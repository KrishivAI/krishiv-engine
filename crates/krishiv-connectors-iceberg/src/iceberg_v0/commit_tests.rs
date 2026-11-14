// SPDX-License-Identifier: Apache-2.0
//! Deterministic simulation of the append commit loop (`commit`, PR-36a;
//! design note §7.2–§7.4, the DST list of §10 row 36a). `TestCatalog`
//! implements `CommitTable` the way `append_tests` applies a request (its
//! requirements checked, its updates applied through iceberg-rust's metadata
//! builder, the metadata JSON written), with a scripted fault per request: a
//! lost response after the apply, an apply delayed by N loads or held until
//! the test lands it, 409, stale 400, duplicate-id 400, 401, 5xx before the
//! apply, a request delivered twice. Foreign writers commit at chosen points
//! (before or after a request, before a load). Time is virtual
//! (`TestClock`). After each scenario the table is checked: every live data
//! file listed once and the rows the model expects, read through SQL and
//! iceberg-rust's planning; and the store holds exactly the files the
//! outcome's cleanup rule keeps (§7.4).

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use iceberg::spec::{FormatVersion, TableMetadata, TableMetadataBuilder};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{ErrorClass, InternalError, KrishivError};

use super::append::{NewManifests, WRITE_ID_PROP, WriteIds, prepare_attempt, write_new_manifests};
use super::append_tests::{MAX, create, file, iceberg_rust_files, ids, rows};
use super::commit::{CommitClock, CommitContext, CommitOutcome, CommitTable, commit_append};
use super::provider_tests::{Env, env};

/// One commit request's body.
type Request = (Vec<TableRequirement>, Vec<TableUpdate>);

/// A foreign writer's request, built on the metadata it is given.
type Foreign =
    Box<dyn FnOnce(TableMetadata) -> Pin<Box<dyn Future<Output = Request> + Send>> + Send>;

/// The snapshot id and write key of the write under test.
const OURS: i64 = 7_000;
const OUR_KEY: &str = "w-ours";
const OUR_UUID: &str = "uuid-ours";

/// What the catalog does with one request.
#[derive(Debug, Clone, Copy)]
enum Fault {
    /// Applied (or refused) and answered.
    Answer,
    /// Applied; the answer is lost (a timeout).
    LostResponse,
    /// Not applied now; applied (if its requirements still hold) after the
    /// given number of further loads; answered 504.
    DelayedApply(usize),
    /// Not applied until the test lands it; answered 504.
    Held,
    /// A 409 without applying.
    Conflict409,
    /// A stale 400 without applying.
    Stale400,
    /// The duplicate-id 400 without applying.
    DuplicateId400,
    /// A 401.
    Unauthorized401,
    /// A 503 before the request reached the table.
    ServerErrorBeforeApply,
    /// Delivered twice: the first copy applied, the client sees the answer
    /// to the second.
    Duplicated,
}

/// When a foreign writer commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum When {
    BeforeRequest(usize),
    AfterRequest(usize),
    BeforeLoad(usize),
}

/// Why a catalog refused a request.
enum Refusal {
    Conflict,
    Stale,
    Duplicate,
    Other(String),
}

/// A request sent but not applied yet.
struct Pending {
    request: Request,
    /// Loads until it lands; `None` until the test lands it.
    loads_left: Option<usize>,
}

struct State {
    meta: TableMetadata,
    location: String,
    version: u32,
    script: VecDeque<Fault>,
    pending: Vec<Pending>,
    requests: usize,
    /// The `all_snapshots` flag of every load.
    loads: Vec<bool>,
    /// Loads still to fail (a 503 on `loadTable`).
    failing_loads: usize,
    foreign: Vec<(When, Foreign)>,
}

/// The catalog under test: one table, a script of faults.
struct TestCatalog {
    state: Mutex<State>,
}

/// The table `m` after the catalog applies `request`, or why not.
fn apply(m: &TableMetadata, location: &str, request: &Request) -> Result<TableMetadata, Refusal> {
    if request.0.iter().any(|r| r.check(Some(m)).is_err()) {
        return Err(Refusal::Conflict);
    }
    let mut b = TableMetadataBuilder::new_from_metadata(m.clone(), Some(location.to_owned()));
    for u in request.1.clone() {
        b = u.apply(b).map_err(|e| {
            let s = e.to_string();
            if s.contains("Snapshot already exists") {
                Refusal::Duplicate
            } else if s.contains("older than last sequence number")
                || s.contains("first-row-id is behind")
                || s.contains("Invalid snapshot timestamp")
            {
                Refusal::Stale
            } else {
                Refusal::Other(s)
            }
        })?;
    }
    b.build()
        .map(|r| r.metadata)
        .map_err(|e| Refusal::Other(e.to_string()))
}

fn error(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

impl State {
    /// Applies `request` and writes the metadata file; the refusal if any.
    fn land(&mut self, request: &Request) -> Result<TableMetadata, Refusal> {
        let next = apply(&self.meta, &self.location, request)?;
        self.persist(next.clone());
        Ok(next)
    }

    /// Writes `next` as the table's new metadata file.
    fn persist(&mut self, next: TableMetadata) {
        self.version += 1;
        let path = format!(
            "{}/metadata/v{}-test.metadata.json",
            next.location(),
            self.version
        );
        std::fs::write(
            path.strip_prefix("file://").unwrap(),
            serde_json::to_vec(&next).unwrap(),
        )
        .unwrap();
        self.meta = next;
        self.location = path;
    }

    fn answer(&mut self, fault: Fault, request: Request) -> CommitOutcome {
        let answered = |r: Result<TableMetadata, Refusal>| match r {
            Ok(m) => CommitOutcome::Committed(Box::new(m)),
            Err(Refusal::Conflict) => CommitOutcome::Conflict { stale: false },
            Err(Refusal::Stale) => CommitOutcome::Conflict { stale: true },
            Err(Refusal::Duplicate) => {
                CommitOutcome::Unknown("400 snapshot already exists".to_owned())
            }
            Err(Refusal::Other(s)) => CommitOutcome::Failed(error(&s)),
        };
        match fault {
            Fault::Answer => answered(self.land(&request)),
            Fault::LostResponse => {
                let _ = self.land(&request);
                CommitOutcome::Unknown("timed out after the request was sent".to_owned())
            }
            Fault::DelayedApply(n) => {
                self.pending.push(Pending {
                    request,
                    loads_left: Some(n),
                });
                CommitOutcome::Unknown("504 gateway timeout".to_owned())
            }
            Fault::Held => {
                self.pending.push(Pending {
                    request,
                    loads_left: None,
                });
                CommitOutcome::Unknown("504 gateway timeout".to_owned())
            }
            Fault::Conflict409 => CommitOutcome::Conflict { stale: false },
            Fault::Stale400 => CommitOutcome::Conflict { stale: true },
            Fault::DuplicateId400 => {
                CommitOutcome::Unknown("400 snapshot already exists".to_owned())
            }
            Fault::Unauthorized401 => CommitOutcome::Failed(error("401 unauthorized")),
            Fault::ServerErrorBeforeApply => {
                CommitOutcome::Unknown("503 service unavailable".to_owned())
            }
            Fault::Duplicated => {
                let _ = self.land(&request);
                answered(self.land(&request))
            }
        }
    }

    /// One load went by: delayed requests whose count runs out land.
    fn tick(&mut self) {
        let mut due = Vec::new();
        self.pending.retain_mut(|p| match &mut p.loads_left {
            Some(0 | 1) => {
                due.push(p.request.clone());
                false
            }
            Some(n) => {
                *n -= 1;
                true
            }
            None => true,
        });
        for r in due {
            let _ = self.land(&r);
        }
    }
}

/// `m` as REST's `snapshots=refs` returns it: only the snapshots refs point at.
fn refs_only(m: &TableMetadata) -> TableMetadata {
    let mut json = serde_json::to_value(m).unwrap();
    let heads: HashSet<i64> = json["refs"]
        .as_object()
        .map(|refs| {
            refs.values()
                .map(|r| r["snapshot-id"].as_i64().unwrap())
                .collect()
        })
        .unwrap_or_default();
    if let Some(snaps) = json["snapshots"].as_array_mut() {
        snaps.retain(|s| heads.contains(&s["snapshot-id"].as_i64().unwrap()));
    }
    serde_json::from_value(json).unwrap()
}

impl TestCatalog {
    fn new(meta: TableMetadata, location: String) -> Self {
        Self {
            state: Mutex::new(State {
                meta,
                location,
                version: 100,
                script: VecDeque::new(),
                pending: Vec::new(),
                requests: 0,
                loads: Vec::new(),
                failing_loads: 0,
                foreign: Vec::new(),
            }),
        }
    }

    fn script(&self, faults: &[Fault]) {
        self.state.lock().unwrap().script.extend(faults);
    }

    fn on(&self, when: When, f: Foreign) {
        self.state.lock().unwrap().foreign.push((when, f));
    }

    fn fail_loads(&self, n: usize) {
        self.state.lock().unwrap().failing_loads = n;
    }

    /// Lands every held request (a gateway's late delivery).
    fn land_held(&self) {
        let mut s = self.state.lock().unwrap();
        let held: Vec<Request> = s.pending.drain(..).map(|p| p.request).collect();
        for r in held {
            let _ = s.land(&r);
        }
    }

    fn metadata(&self) -> (TableMetadata, String) {
        let s = self.state.lock().unwrap();
        (s.meta.clone(), s.location.clone())
    }

    fn requests(&self) -> usize {
        self.state.lock().unwrap().requests
    }

    fn loads(&self) -> Vec<bool> {
        self.state.lock().unwrap().loads.clone()
    }

    /// Commits the foreign writers due `when`, in order.
    async fn foreign(&self, when: When) {
        loop {
            let (f, meta) = {
                let mut s = self.state.lock().unwrap();
                let Some(i) = s.foreign.iter().position(|(w, _)| *w == when) else {
                    return;
                };
                (s.foreign.remove(i).1, s.meta.clone())
            };
            let request = f(meta).await;
            let mut s = self.state.lock().unwrap();
            assert!(s.land(&request).is_ok(), "a foreign commit was refused");
        }
    }
}

#[async_trait]
impl CommitTable for TestCatalog {
    fn name(&self) -> String {
        "db.t".to_owned()
    }

    async fn commit(
        &self,
        requirements: Vec<TableRequirement>,
        updates: Vec<TableUpdate>,
    ) -> CommitOutcome {
        let n = {
            let mut s = self.state.lock().unwrap();
            s.requests += 1;
            s.requests
        };
        self.foreign(When::BeforeRequest(n)).await;
        let outcome = {
            let mut s = self.state.lock().unwrap();
            let fault = s.script.pop_front().unwrap_or(Fault::Answer);
            s.answer(fault, (requirements, updates))
        };
        self.foreign(When::AfterRequest(n)).await;
        outcome
    }

    async fn load(&self, all_snapshots: bool) -> Result<TableMetadata, KrishivError> {
        let n = {
            let mut s = self.state.lock().unwrap();
            s.loads.push(all_snapshots);
            s.loads.len()
        };
        self.foreign(When::BeforeLoad(n)).await;
        let mut s = self.state.lock().unwrap();
        let answer = if s.failing_loads > 0 {
            s.failing_loads -= 1;
            Err(error("503 on loadTable"))
        } else if all_snapshots {
            Ok(s.meta.clone())
        } else {
            Ok(refs_only(&s.meta))
        };
        s.tick();
        answer
    }
}

/// Virtual time: a sleep advances the clock at once and is recorded.
struct TestClock {
    now: AtomicI64,
    waits: Mutex<Vec<u64>>,
    /// Jitter at its largest (`bound - 1`) rather than 0.
    max_jitter: AtomicBool,
}

#[async_trait]
impl CommitClock for TestClock {
    fn now_ms(&self) -> i64 {
        self.now.load(Ordering::SeqCst)
    }

    async fn sleep(&self, wait: Duration) {
        let ms = u64::try_from(wait.as_millis()).unwrap();
        self.waits.lock().unwrap().push(ms);
        self.now
            .fetch_add(i64::try_from(ms).unwrap(), Ordering::SeqCst);
    }

    fn jitter_ms(&self, bound: u64) -> u64 {
        if self.max_jitter.load(Ordering::SeqCst) {
            bound - 1
        } else {
            0
        }
    }
}

/// A foreign append of one row in partition 1 to `branch`, as snapshot `id`.
fn foreign(
    env: &Env,
    clock: &Arc<TestClock>,
    m: &TableMetadata,
    name: &str,
    id: i64,
    branch: &str,
) -> Foreign {
    let f = file(env, m, name, 1, &[(id, name)]);
    let io = env.reg.file_io();
    let clock = Arc::clone(clock);
    let branch = branch.to_owned();
    Box::new(move |m: TableMetadata| {
        Box::pin(async move {
            let uuid = format!("uuid-{id}");
            let mut new = write_new_manifests(&io, &m, vec![f], id, &uuid)
                .await
                .unwrap();
            let ts = clock.now_ms().max(m.last_updated_ms() + 1);
            let ids = WriteIds {
                branch: &branch,
                write_id: &uuid,
                query_id: None,
            };
            let a = prepare_attempt(&io, &m, &mut new, ids, ts, MAX)
                .await
                .unwrap();
            (a.requirements, a.updates)
        })
    })
}

/// A format-version upgrade by another writer.
fn upgrade(to: FormatVersion) -> Foreign {
    Box::new(move |_| {
        Box::pin(async move {
            (
                vec![],
                vec![TableUpdate::UpgradeFormatVersion { format_version: to }],
            )
        })
    })
}

/// A table of format `version` with one committed snapshot (row `1|1|init`),
/// the catalog over it, the clock and the write under test: one data file
/// (`10|2|ours`) and its manifests, written against that snapshot.
struct Case {
    env: Env,
    cat: TestCatalog,
    clock: Arc<TestClock>,
    base: TableMetadata,
    new: NewManifests,
}

fn case_with(version: u8, properties: &[(&str, &str)]) -> Case {
    let env = env();
    let t = create(&env, version);
    let mut m0 = t.metadata().clone();
    if !properties.is_empty() {
        m0 = TableMetadataBuilder::new_from_metadata(m0, None)
            .set_properties(
                properties
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                    .collect(),
            )
            .unwrap()
            .build()
            .unwrap()
            .metadata;
    }
    let cat = TestCatalog::new(m0.clone(), t.metadata_location().unwrap().to_owned());
    let clock = Arc::new(TestClock {
        now: AtomicI64::new(m0.last_updated_ms() + 1_000),
        waits: Mutex::new(Vec::new()),
        max_jitter: AtomicBool::new(false),
    });
    let init = foreign(&env, &clock, &m0, "init.parquet", 1, "main");
    env.rt.block_on(async {
        let request = init(m0.clone()).await;
        cat.state.lock().unwrap().land(&request).ok().unwrap();
    });
    let base = cat.metadata().0;
    let ours = file(&env, &base, "ours.parquet", 2, &[(10, "ours")]);
    let io = env.reg.file_io();
    let new = env
        .rt
        .block_on(write_new_manifests(&io, &base, vec![ours], OURS, OUR_UUID))
        .unwrap();
    Case {
        env,
        cat,
        clock,
        base,
        new,
    }
}

fn case() -> Case {
    case_with(2, &[])
}

impl Case {
    fn run(&mut self) -> Result<TableMetadata, KrishivError> {
        let io = self.env.reg.file_io();
        let cx = CommitContext {
            io: &io,
            clock: self.clock.as_ref(),
            max_metadata_bytes: MAX,
        };
        self.env.rt.block_on(commit_append(
            &self.cat,
            cx,
            &self.base,
            &mut self.new,
            ids(OUR_KEY),
        ))
    }

    fn foreign(&self, name: &str, id: i64, branch: &str) -> Foreign {
        foreign(&self.env, &self.clock, &self.base, name, id, branch)
    }

    /// The location of attempt `n`'s manifest list.
    fn list(&self, n: u32) -> String {
        format!(
            "{}/metadata/snap-{OURS}-{n}-{OUR_UUID}.avro",
            self.base.location()
        )
    }

    fn manifest(&self) -> String {
        self.new.manifests[0].manifest_path.clone()
    }

    fn data(&self) -> String {
        self.env.at("ours.parquet")
    }

    fn waits(&self) -> Vec<u64> {
        self.clock.waits.lock().unwrap().clone()
    }

    /// The table's invariants: every live data file listed once, our
    /// snapshot at most once, and the rows `expected` (`id|x|name`, by id)
    /// through SQL and the same files through iceberg-rust's planning.
    fn check(&self, expected: &[&str]) {
        let (m, loc) = self.cat.metadata();
        let files = iceberg_rust_files(&self.env, &m, &loc);
        let unique: HashSet<&String> = files.iter().collect();
        assert_eq!(unique.len(), files.len(), "a file listed twice: {files:?}");
        let ours = m
            .snapshots()
            .filter(|s| {
                s.summary().additional_properties.get(WRITE_ID_PROP) == Some(&OUR_KEY.to_owned())
            })
            .count();
        assert!(ours <= 1, "the write committed {ours} times");
        assert_eq!(rows(&self.env, &loc), expected);
    }
}

fn exists(url: &str) -> bool {
    std::path::Path::new(url.strip_prefix("file://").unwrap()).exists()
}

const INIT: &str = "1|1|init.parquet";
const MINE: &str = "10|2|ours";

/// The plain path: one request, committed; the list, manifest and data file
/// kept; no wait.
#[test]
fn commits_on_the_first_request() {
    let mut c = case();
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(c.cat.requests(), 1);
    assert!(c.waits().is_empty());
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE]);
}

/// A 409 (another writer moved `main`): the own-key search finds nothing, the
/// write is rebased on the new head with the same snapshot id and sent after
/// `commit.retry.min-wait-ms`; the first attempt's list is deleted, the data
/// file and manifests are kept (§7.4 committed).
#[test]
fn a_conflict_is_rebased_on_the_refreshed_table() {
    let mut c = case();
    c.cat
        .on(When::BeforeRequest(1), c.foreign("f.parquet", 900, "main"));
    let m = c.run().unwrap();
    let s = m.snapshot_by_id(OURS).unwrap();
    assert_eq!(s.parent_snapshot_id(), Some(900));
    assert_eq!(s.manifest_list(), c.list(2));
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(c.waits(), [100]);
    assert!(!exists(&c.list(1)));
    assert!(exists(&c.list(2)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE, "900|1|f.parquet"]);
}

/// R:S1: a commit to another branch takes the table's next sequence number,
/// so the request built before it is stale (iceberg-rust's refusal); the
/// rebase takes `last-sequence-number` + 1 and stays on `main`'s head.
#[test]
fn append_after_foreign_branch_commit() {
    let mut c = case();
    let head = c.base.current_snapshot_id();
    c.cat
        .on(When::BeforeRequest(1), c.foreign("dev.parquet", 901, "dev"));
    let m = c.run().unwrap();
    let s = m.snapshot_by_id(OURS).unwrap();
    assert_eq!(s.parent_snapshot_id(), head);
    assert_eq!(s.sequence_number(), 3);
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(c.cat.requests(), 2);
    c.check(&[INIT, MINE]);
}

/// A stale 400 rebases; the duplicate-id 400 is "probably committed", never a
/// failure: probed (`commit.status-check.*`: immediately, then after 1 s, 2 s,
/// 4 s), and with the ref still at the base the same attempt is resent.
/// Waits are Java's `Tasks` exponential backoff with jitter below 10%.
#[test]
fn stale_and_duplicate_id_answers_are_never_failures() {
    let mut c = case();
    c.clock.max_jitter.store(true, Ordering::SeqCst);
    c.cat.script(&[Fault::Stale400, Fault::DuplicateId400]);
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(2));
    assert_eq!(c.cat.requests(), 3);
    assert_eq!(c.waits(), [109, 1099, 2199, 4399, 219]);
    assert!(!exists(&c.list(1)) && exists(&c.list(2)));
    c.check(&[INIT, MINE]);
}

/// §10 36a: a lost 200 and then a foreign commit on top: the probe finds the
/// snapshot (it is in the table), so the write is committed once.
#[test]
fn lost_200_then_a_foreign_commit_is_found_by_the_probe() {
    let mut c = case();
    c.cat.script(&[Fault::LostResponse]);
    c.cat
        .on(When::AfterRequest(1), c.foreign("f.parquet", 902, "main"));
    let m = c.run().unwrap();
    assert!(m.snapshot_by_id(OURS).is_some());
    assert_eq!(m.current_snapshot_id(), Some(902));
    assert_eq!(c.cat.requests(), 1);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE, "902|1|f.parquet"]);
}

/// §10 36a: a landed snapshot that other writers built on before the probe is
/// no ref's head, so a `snapshots=refs` load lacks it; the probe loads with
/// `snapshots=all` and finds it.
#[test]
fn a_landed_snapshot_is_found_only_through_snapshots_all() {
    let mut c = case();
    c.cat.script(&[Fault::LostResponse]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f.parquet", 903, "main"));
    c.cat
        .on(When::BeforeLoad(1), c.foreign("g.parquet", 904, "main"));
    let m = c.run().unwrap();
    assert!(m.snapshot_by_id(OURS).is_some());
    assert!(c.cat.loads().iter().all(|all| *all), "{:?}", c.cat.loads());
    assert!(refs_only(&m).snapshot_by_id(OURS).is_none());
    c.check(&[INIT, MINE, "903|1|f.parquet", "904|1|g.parquet"]);
}

/// §10 36a: the request lands after the last probe said "not found"; the
/// resend of the same attempt gets 409 and the own-key search finds the
/// snapshot: one snapshot, committed.
#[test]
fn an_apply_delayed_past_the_probe_is_found_after_the_resend() {
    let mut c = case();
    c.cat.script(&[Fault::DelayedApply(4)]);
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(c.cat.loads().len(), 5);
    assert!(exists(&c.list(1)) && !exists(&c.list(2)));
    c.check(&[INIT, MINE]);
}

/// §10 36a: the request is held past the whole budget (every resend a 503
/// before the table): `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` (40003) naming
/// the snapshot and write key, and no file deleted; when the request lands
/// later the table shows the rows once.
#[test]
fn an_apply_delayed_past_the_budget_is_state_unknown_and_deletes_nothing() {
    let mut c = case();
    c.cat.script(&[
        Fault::Held,
        Fault::ServerErrorBeforeApply,
        Fault::ServerErrorBeforeApply,
        Fault::ServerErrorBeforeApply,
        Fault::ServerErrorBeforeApply,
    ]);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown
    );
    assert_eq!(e.sql_state(), Some("40003"));
    assert!(
        e.message().contains(&OURS.to_string()) && e.message().contains(OUR_KEY),
        "{e}"
    );
    // `commit.retry.num-retries` 4: five requests, each probed four times.
    assert_eq!(c.cat.requests(), 5);
    assert_eq!(c.cat.loads().len(), 20);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT]);
    c.cat.land_held();
    c.check(&[INIT, MINE]);
}

/// `commit.retry.total-timeout-ms` bounds the loop, probe waits included.
#[test]
fn the_total_timeout_bounds_the_probes() {
    let mut c = case_with(2, &[("commit.retry.total-timeout-ms", "5000")]);
    c.cat.script(&[Fault::Held]);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown
    );
    assert_eq!(c.cat.requests(), 1);
    assert_eq!(c.waits(), [1000, 2000, 4000]);
    assert!(exists(&c.list(1)) && exists(&c.data()));
}

/// §10 36a: a resend answered 401 says nothing about the first request,
/// which lands later and is found: committed, nothing deleted.
#[test]
fn a_401_on_a_resend_then_the_first_request_lands() {
    let mut c = case();
    c.cat
        .script(&[Fault::DelayedApply(6), Fault::Unauthorized401]);
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(c.cat.requests(), 2);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE]);
}

/// A 401 on every resend after an unknown first request leaves the outcome
/// unknown (§7.2's per-request rule): nothing deleted, and the late apply
/// reads back once.
#[test]
fn a_401_on_every_resend_is_state_unknown_and_deletes_nothing() {
    let mut c = case();
    c.cat.script(&[
        Fault::Held,
        Fault::Unauthorized401,
        Fault::Unauthorized401,
        Fault::Unauthorized401,
        Fault::Unauthorized401,
    ]);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown
    );
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.cat.land_held();
    c.check(&[INIT, MINE]);
}

/// A 401 to an attempt's first request is a known failure: its error, and
/// everything the write produced deleted (§7.4).
#[test]
fn a_401_on_a_first_request_fails_and_deletes_everything() {
    let mut c = case();
    c.cat.script(&[Fault::Unauthorized401]);
    let e = c.run().unwrap_err();
    assert!(e.to_string().contains("401"), "{e}");
    assert_eq!(c.cat.requests(), 1);
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    c.check(&[INIT]);
}

/// A 5xx before the table and failing probes: still unknown, so after the
/// probes the same attempt (same list) is resent, never a new one.
#[test]
fn a_5xx_before_apply_resends_the_same_attempt() {
    let mut c = case();
    c.cat.script(&[Fault::ServerErrorBeforeApply]);
    c.cat.fail_loads(2);
    let m = c.run().unwrap();
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(1));
    assert_eq!(c.cat.requests(), 2);
    assert!(!exists(&c.list(2)));
    c.check(&[INIT, MINE]);
}

/// §10 36a: a request delivered twice is applied once; the client sees the
/// second copy's 409 and the own-key search finds the first.
#[test]
fn a_duplicated_request_is_applied_once() {
    let mut c = case();
    c.cat.script(&[Fault::Duplicated]);
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(c.cat.requests(), 1);
    assert!(exists(&c.list(1)));
    c.check(&[INIT, MINE]);
}

/// Every request refused by a concurrent commit (all outcomes known):
/// `KRISHIV_ICEBERG_COMMIT_FAILED` (40001) after `commit.retry.num-retries`
/// + 1 requests, and everything the write produced deleted.
#[test]
fn a_race_lost_every_time_fails_and_deletes_everything() {
    let mut c = case();
    for n in 1..=5 {
        c.cat.on(
            When::BeforeRequest(n),
            c.foreign(
                &format!("f{n}.parquet"),
                910 + i64::try_from(n).unwrap(),
                "main",
            ),
        );
    }
    let e = c.run().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivIcebergCommitFailed);
    assert_eq!(e.sql_state(), Some("40001"));
    assert_eq!(c.cat.requests(), 5);
    assert_eq!(c.waits(), [100, 200, 400, 800]);
    for n in 1..=5 {
        assert!(!exists(&c.list(n)), "list {n}");
    }
    assert!(!exists(&c.manifest()) && !exists(&c.data()));
    c.check(&[
        INIT,
        "911|1|f1.parquet",
        "912|1|f2.parquet",
        "913|1|f3.parquet",
        "914|1|f4.parquet",
        "915|1|f5.parquet",
    ]);
}

/// §7.2: a stale answer to a resend leads to a rebase, but the first request
/// may still land (the ref never moved), so when the budget runs out the
/// write is unknown and nothing is deleted; the late apply reads back once.
#[test]
fn a_stale_answer_to_a_resend_leaves_the_write_unknown() {
    let mut c = case();
    c.cat.script(&[
        Fault::Held,
        Fault::Stale400,
        Fault::Conflict409,
        Fault::Conflict409,
        Fault::Conflict409,
    ]);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown
    );
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.cat.land_held();
    c.check(&[INIT, MINE]);
}

/// §7.3 step 4a.0: an append rebases across a v2 → v3 upgrade, taking the
/// lineage fields from the upgraded table.
#[test]
fn an_append_commits_across_a_v2_to_v3_upgrade() {
    let mut c = case();
    c.cat.on(When::BeforeRequest(1), upgrade(FormatVersion::V3));
    c.cat
        .on(When::BeforeRequest(1), c.foreign("f.parquet", 920, "main"));
    let m = c.run().unwrap();
    assert_eq!(m.format_version(), FormatVersion::V3);
    let s = m.snapshot_by_id(OURS).unwrap();
    assert_eq!(s.parent_snapshot_id(), Some(920));
    let (first, added) = s.row_range().unwrap();
    assert_eq!(m.next_row_id(), first + added);
    assert!(added >= 1, "{added}");
    c.check(&[INIT, MINE, "920|1|f.parquet"]);
}

/// §7.3 step 4a.0: any other format change under an append is
/// `KRISHIV_ICEBERG_COMMIT_CONFLICT`, and the write's files are deleted.
#[test]
fn a_v1_to_v2_upgrade_is_a_conflict() {
    let mut c = case_with(1, &[]);
    c.cat.on(When::BeforeRequest(1), upgrade(FormatVersion::V2));
    c.cat
        .on(When::BeforeRequest(1), c.foreign("f.parquet", 930, "main"));
    let e = c.run().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivIcebergCommitConflict);
    assert_eq!(e.sql_state(), Some("40001"));
    assert!(e.message().contains("930"), "{e}");
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    c.check(&[INIT, "930|1|f.parquet"]);
}

/// A commit property that is not a number is refused before any request, and
/// the write's files are deleted.
#[test]
fn a_bad_retry_property_is_refused_before_any_request() {
    let mut c = case_with(2, &[("commit.retry.num-retries", "many")]);
    let e = c.run().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("commit.retry.num-retries"), "{e}");
    assert_eq!(c.cat.requests(), 0);
    assert!(!exists(&c.manifest()) && !exists(&c.data()));
}
