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
use iceberg::spec::{
    FormatVersion, SnapshotReference, SnapshotRetention, TableMetadata, TableMetadataBuilder,
};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{ErrorClass, InternalError, KrishivError};

use super::append::{NewManifests, WRITE_ID_PROP, WriteIds, prepare_attempt, write_new_manifests};
use super::append_tests::{MAX, create, file, iceberg_rust_files, ids, rows};
use super::commit::{CommitClock, CommitContext, CommitOutcome, CommitTable, commit_append};
use super::provider_tests::{Env, env};
use super::scan::read_manifest_files;

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

/// What happens at a [`When`] point.
enum Action {
    /// Another writer's request, landed.
    Foreign(Foreign),
    /// Every held request lands (a gateway's late delivery).
    LandHeld,
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
    foreign: Vec<(When, Action)>,
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

    /// Lands every held request; how many were applied.
    fn land_held(&mut self) -> usize {
        let held: Vec<Request> = self.pending.drain(..).map(|p| p.request).collect();
        held.iter().filter(|r| self.land(r).is_ok()).count()
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
        self.state
            .lock()
            .unwrap()
            .foreign
            .push((when, Action::Foreign(f)));
    }

    /// Lands every held request at `when`.
    fn land_held_at(&self, when: When) {
        self.state
            .lock()
            .unwrap()
            .foreign
            .push((when, Action::LandHeld));
    }

    fn fail_loads(&self, n: usize) {
        self.state.lock().unwrap().failing_loads = n;
    }

    /// Lands every held request (a gateway's late delivery); how many
    /// were applied.
    fn land_held(&self) -> usize {
        self.state.lock().unwrap().land_held()
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

    /// Runs the actions due `when`, in order.
    async fn foreign(&self, when: When) {
        loop {
            let (f, meta) = {
                let mut s = self.state.lock().unwrap();
                let Some(i) = s.foreign.iter().position(|(w, _)| *w == when) else {
                    return;
                };
                match s.foreign.remove(i).1 {
                    Action::Foreign(f) => (f, s.meta.clone()),
                    Action::LandHeld => {
                        s.land_held();
                        continue;
                    }
                }
            };
            let request = f(meta).await;
            let mut s = self.state.lock().unwrap();
            if let Err(r) = s.land(&request) {
                let why = match r {
                    Refusal::Conflict => "conflict".to_owned(),
                    Refusal::Stale => "stale".to_owned(),
                    Refusal::Duplicate => "duplicate".to_owned(),
                    Refusal::Other(s) => s,
                };
                panic!("a foreign commit was refused: {why}");
            }
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
    foreign_keyed(env, clock, m, name, id, branch, None)
}

/// [`foreign`] with the write key `key` (default `uuid-<id>`).
fn foreign_keyed(
    env: &Env,
    clock: &Arc<TestClock>,
    m: &TableMetadata,
    name: &str,
    id: i64,
    branch: &str,
    key: Option<&str>,
) -> Foreign {
    let f = file(env, m, name, 1, &[(id, name)]);
    let key = key.map_or_else(|| format!("uuid-{id}"), str::to_owned);
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
                write_id: &key,
                query_id: None,
            };
            let a = prepare_attempt(&io, &m, &mut new, ids, ts, MAX)
                .await
                .unwrap();
            (a.requirements, a.updates)
        })
    })
}

/// Another writer rolls `main` back to snapshot `to` (no requirements).
fn rollback(to: i64) -> Foreign {
    Box::new(move |_| {
        Box::pin(async move {
            let retention = SnapshotRetention::Branch {
                min_snapshots_to_keep: None,
                max_snapshot_age_ms: None,
                max_ref_age_ms: None,
            };
            (
                vec![],
                vec![TableUpdate::SetSnapshotRef {
                    ref_name: "main".to_owned(),
                    reference: SnapshotReference::new(to, retention),
                }],
            )
        })
    })
}

/// The snapshot ids of `main`'s ancestry in `m`, head first.
fn main_ancestry(m: &TableMetadata) -> Vec<i64> {
    let mut out = Vec::new();
    let mut next = m.current_snapshot_id();
    while let Some(id) = next {
        out.push(id);
        next = m.snapshot_by_id(id).and_then(|s| s.parent_snapshot_id());
    }
    out
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

    /// A foreign append carrying the write key `key`.
    fn foreign_keyed(&self, name: &str, id: i64, branch: &str, key: &str) -> Foreign {
        foreign_keyed(
            &self.env,
            &self.clock,
            &self.base,
            name,
            id,
            branch,
            Some(key),
        )
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
    assert_eq!(c.cat.land_held(), 1);
    c.check(&[INIT, MINE]);
}

/// `commit.retry.total-timeout-ms` bounds the loop, probe waits included:
/// the first request's probes pass the timeout (7 s of 5), its resend is
/// still sent (Java's `Tasks`: only a failure after the first stops at the
/// timeout, review N1), and the resend's first probe wait ends the budget.
#[test]
fn the_total_timeout_bounds_the_probes() {
    let mut c = case_with(2, &[("commit.retry.total-timeout-ms", "5000")]);
    c.cat.script(&[Fault::Held, Fault::Held]);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown
    );
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(c.waits(), [1000, 2000, 4000, 100]);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT]);
    assert_eq!(c.cat.land_held(), 1);
    c.check(&[INIT, MINE]);
}

/// Review N1: a first failure after the total timeout is retried once, as
/// Java's `Tasks.runTaskWithRetry` (`durationMs > maxDurationMs && attempt >
/// 1`); a second one ends the loop. Here every request is a known conflict
/// (a foreign commit before each), so the write fails after two requests.
#[test]
fn a_first_failure_after_the_total_timeout_is_retried_once() {
    let mut c = case_with(2, &[("commit.retry.total-timeout-ms", "0")]);
    // The first retry wait (100 ms) passes the 0 ms budget.
    for n in 1..=2 {
        c.cat.on(
            When::BeforeRequest(n),
            c.foreign(
                &format!("t{n}.parquet"),
                940 + i64::try_from(n).unwrap(),
                "main",
            ),
        );
    }
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitFailed,
        "{e}"
    );
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(c.waits(), [100]);
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
    assert_eq!(c.cat.land_held(), 1);
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
    // Four attempts (the fifth request's budget check comes first).
    for n in 1..=4 {
        assert!(exists(&c.list(n)), "list {n}");
    }
    assert!(!exists(&c.list(5)));
    assert!(exists(&c.manifest()) && exists(&c.data()));
    assert_eq!(c.cat.land_held(), 1);
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
    // The first v3 list (the foreign commit's) assigned ids 0 and 1 to the
    // pre-upgrade manifest's row and its own; this write's one row is 2.
    assert_eq!((first, added), (2, 1));
    assert_eq!(m.next_row_id(), 3);
    let listed = c
        .env
        .rt
        .block_on(super::scan::read_manifest_files(
            &c.env.reg.file_io(),
            s.manifest_list(),
            3,
            MAX,
        ))
        .unwrap();
    assert!(
        listed.iter().all(|f| f.first_row_id.is_some()),
        "{listed:?}"
    );
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
    assert!(e.message().contains("removed where possible"), "{e}");
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
    c.check(&[INIT]);
}

/// R:B1: a 409 says nothing about the write until the own-key search has run
/// on a successful load. A request delivered twice lands its first copy and
/// answers the second's 409; when every refresh then fails, the write is
/// `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` and nothing is deleted (the table's
/// head references the files).
#[test]
fn a_conflict_whose_refresh_never_succeeds_is_state_unknown() {
    let mut c = case();
    c.cat.script(&[Fault::Duplicated]);
    c.cat.fail_loads(4);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown,
        "{e}"
    );
    assert_eq!(c.cat.requests(), 1);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE]);
}

/// R:B1, a plain 409 (nothing applied) with failing refreshes: still
/// unknown to the loop, so nothing is deleted.
#[test]
fn a_plain_conflict_whose_refresh_never_succeeds_deletes_nothing() {
    let mut c = case();
    c.cat.script(&[Fault::Conflict409]);
    c.cat.fail_loads(4);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown,
        "{e}"
    );
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT]);
}

/// R:B1, the total timeout ending the refresh: the same.
#[test]
fn a_conflict_whose_refresh_times_out_is_state_unknown() {
    let mut c = case_with(2, &[("commit.retry.total-timeout-ms", "1500")]);
    c.cat.script(&[Fault::Duplicated]);
    c.cat.fail_loads(10);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown,
        "{e}"
    );
    assert_eq!(c.waits(), [1000, 2000]);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE]);
}

/// A refresh that fails fewer times than `commit.status-check.num-retries`
/// and then succeeds: rebased and committed.
#[test]
fn a_refresh_that_recovers_rebases() {
    let mut c = case();
    c.cat.script(&[Fault::Conflict409]);
    c.cat.fail_loads(2);
    let m = c.run().unwrap();
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(2));
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(c.waits(), [1000, 2000, 100]);
    assert!(!exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE]);
}

/// R:S1: with this write's snapshot in the table and another snapshot
/// carrying the same write key (a replayed write on `dev`), the own-key
/// search takes the snapshot id match, whatever the order of the table's
/// snapshot map, so the kept list is the one `main` reads. Repeated, as the
/// map's order is per instance.
#[test]
fn the_own_key_search_prefers_the_snapshot_id() {
    for _ in 0..8 {
        let mut c = case();
        c.cat.script(&[Fault::LostResponse]);
        c.cat.on(
            When::BeforeLoad(1),
            c.foreign_keyed("r.parquet", 950, "dev", OUR_KEY),
        );
        let m = c.run().unwrap();
        assert_eq!(m.current_snapshot_id(), Some(OURS));
        assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
        assert_eq!(rows(&c.env, &c.cat.metadata().1), [INIT, MINE]);
    }
}

/// R:S1: the write key found only under another snapshot id (a replayed
/// write, on `dev`) while this write's own request is still unresolved: no
/// list is deleted, so the held request that lands later reads back. A v1
/// table, where the `dev` commit does not make the held request stale.
#[test]
fn a_key_match_on_another_id_deletes_nothing_while_unresolved() {
    let mut c = case_with(1, &[]);
    c.cat.script(&[Fault::Held]);
    c.cat.on(
        When::BeforeLoad(1),
        c.foreign_keyed("r.parquet", 951, "dev", OUR_KEY),
    );
    let m = c.run().unwrap();
    assert!(m.snapshot_by_id(951).is_some());
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    assert_eq!(c.cat.land_held(), 1);
    let (m, loc) = c.cat.metadata();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(rows(&c.env, &loc), [INIT, MINE]);
}

/// Review of the P1 fix, finding 1: a held request settled when the branch
/// moves, then a 409 whose refresh finds a replay of this write under
/// another id. The write counts as committed, but nothing of it is deleted:
/// after a rollback to the base the held request lands and its files are
/// there. (The write is then in two snapshots, the replay's own exposure,
/// so only the rows are read.) A v1 table, as in the P1 case.
#[test]
fn a_replay_found_after_a_settled_unknown_answer_deletes_nothing() {
    let mut c = case_with(1, &[]);
    c.cat.script(&[Fault::Held, Fault::Conflict409]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f.parquet", 968, "main"));
    c.cat.on(
        When::BeforeLoad(2),
        c.foreign_keyed("r.parquet", 952, "main", OUR_KEY),
    );
    let m = c.run().unwrap();
    assert!(m.snapshot_by_id(952).is_some() && m.snapshot_by_id(OURS).is_none());
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    let back = rollback(c.base.current_snapshot_id().unwrap());
    c.env.rt.block_on(async {
        let (m, _) = c.cat.metadata();
        let request = back(m).await;
        c.cat.state.lock().unwrap().land(&request).ok().unwrap();
    });
    assert_eq!(c.cat.land_held(), 1);
    let (m, loc) = c.cat.metadata();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(rows(&c.env, &loc), [INIT, MINE]);
}

/// Review of the P1 fix, finding 5: a held request settled when the branch
/// moves, then a 401 to the new attempt's first request: that error is
/// returned as is, and the files are kept.
#[test]
fn a_plain_failure_after_a_settled_unknown_answer_keeps_the_files() {
    let mut c = case();
    c.cat.script(&[Fault::Held, Fault::Unauthorized401]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f.parquet", 969, "main"));
    let e = c.run().unwrap_err();
    assert!(e.to_string().contains("401"), "{e}");
    assert_eq!(c.cat.requests(), 2);
    assert!(exists(&c.list(1)) && exists(&c.list(2)));
    assert!(exists(&c.manifest()) && exists(&c.data()));
}

/// R:N3: a foreign snapshot holding this write's snapshot id under another
/// write key is not this write: `KRISHIV_ICEBERG_COMMIT_CONFLICT`, never
/// "committed", and (every outcome known) the write's files deleted.
#[test]
fn a_foreign_snapshot_with_this_id_is_a_conflict() {
    let mut c = case();
    c.cat
        .on(When::BeforeRequest(1), c.foreign("f.parquet", OURS, "main"));
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    c.check(&[INIT, "7000|1|f.parquet"]);
}

/// R:N2: a property value is parsed as Java's `Integer.parseInt` does, so
/// " 4" is not a number.
#[test]
fn a_retry_property_with_spaces_is_not_a_number() {
    let mut c = case_with(2, &[("commit.retry.num-retries", " 4")]);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert_eq!(c.cat.requests(), 0);
    c.check(&[INIT]);
}

/// R:S2, `Probe::Moved`: a 503 before the table, then another writer moves
/// `main` before the first probe: "not applied" is final, the write is
/// rebased on the foreign head, one snapshot, the first list deleted.
#[test]
fn an_unknown_answer_then_a_moved_branch_rebases() {
    let mut c = case();
    c.cat.script(&[Fault::ServerErrorBeforeApply]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f.parquet", 960, "main"));
    let m = c.run().unwrap();
    let s = m.snapshot_by_id(OURS).unwrap();
    assert_eq!(s.parent_snapshot_id(), Some(960));
    assert_eq!(s.manifest_list(), c.list(2));
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(c.cat.loads().len(), 1);
    assert!(!exists(&c.list(1)) && exists(&c.list(2)));
    assert!(exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE, "960|1|f.parquet"]);
}

/// R:S2, `resolve`, and review-1 pass 2 P1: a held request, then the branch
/// moves, so the held request is settled as not applied and the write
/// rebases; every later attempt loses a race: `KRISHIV_ICEBERG_COMMIT_FAILED`,
/// but nothing is deleted, since the held request asserts a base the branch
/// can return to. Another writer rolls `main` back to that base and the held
/// request lands: every file its snapshot names is there, and it reads back
/// once. A v1 table: from v2 on a catalog refuses the held request after the
/// rollback, since its sequence number is behind the table's (Java's
/// `TableMetadata.addSnapshot`), so the exposure needs v1 or a branch moved
/// without a new snapshot.
#[test]
fn a_failed_write_that_had_an_unknown_answer_keeps_its_files() {
    let mut c = case_with(1, &[]);
    c.cat.script(&[Fault::Held]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f1.parquet", 961, "main"));
    for n in 2..=5 {
        c.cat.on(
            When::BeforeRequest(n),
            c.foreign(
                &format!("f{n}.parquet"),
                960 + i64::try_from(n).unwrap(),
                "main",
            ),
        );
    }
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitFailed,
        "{e}"
    );
    assert!(e.message().contains("files were kept"), "{e}");
    assert_eq!(c.cat.requests(), 5);
    for n in 1..=5 {
        assert!(exists(&c.list(n)), "list {n}");
    }
    assert!(exists(&c.manifest()) && exists(&c.data()));
    let base = c.base.current_snapshot_id().unwrap();
    let back = rollback(base);
    c.env.rt.block_on(async {
        let (m, _) = c.cat.metadata();
        let request = back(m).await;
        c.cat.state.lock().unwrap().land(&request).ok().unwrap();
    });
    assert_eq!(c.cat.land_held(), 1);
    let (m, _) = c.cat.metadata();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(1));
    c.check(&[INIT, MINE]);
}

/// Review-1 pass 2 P1: an unknown answer, then a conflict refused by the
/// rebase check: `KRISHIV_ICEBERG_COMMIT_CONFLICT`, its text says the files
/// were kept, and they are.
#[test]
fn a_conflict_after_a_settled_unknown_answer_keeps_the_files() {
    let mut c = case_with(1, &[]);
    c.cat.script(&[Fault::Held]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f.parquet", 967, "main"));
    c.cat.on(When::BeforeLoad(1), upgrade(FormatVersion::V2));
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(e.message().contains("files were kept"), "{e}");
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
}

/// R:S2: a held request, the branch moves, the rebase commits; the held
/// request is refused when it lands, so the rows are in the table once.
#[test]
fn a_held_request_then_a_moved_branch_commits_once() {
    let mut c = case();
    c.cat.script(&[Fault::Held]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f.parquet", 966, "main"));
    let m = c.run().unwrap();
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(2));
    assert!(!exists(&c.list(1)) && exists(&c.list(2)));
    assert_eq!(c.cat.land_held(), 0);
    c.check(&[INIT, MINE, "966|1|f.parquet"]);
}

/// R:S2, `finish` with an error while unresolved: a held request, a stale
/// answer to its resend (the rebase follows, the held request still live),
/// then a 401 to the new attempt's first request: that failure is known for
/// its request only, so `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` and nothing
/// deleted; the held request lands later and reads back once.
#[test]
fn an_error_while_unresolved_is_state_unknown() {
    let mut c = case();
    c.cat
        .script(&[Fault::Held, Fault::Stale400, Fault::Unauthorized401]);
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown,
        "{e}"
    );
    assert!(e.to_string().contains("401"), "{e}");
    assert_eq!(c.cat.requests(), 3);
    assert!(exists(&c.list(1)) && exists(&c.list(2)));
    assert!(exists(&c.manifest()) && exists(&c.data()));
    assert_eq!(c.cat.land_held(), 1);
    c.check(&[INIT, MINE]);
}

/// R:S2, §7.4 "it may be an earlier attempt's": a held request, a stale
/// answer to its resend, a rebase, and the held request lands before the
/// rebased attempt is sent: that one gets 409 and the own-key search finds
/// attempt 1's snapshot: committed, list 1 kept, list 2 deleted.
#[test]
fn an_earlier_attempt_found_after_a_rebase_is_kept() {
    let mut c = case();
    c.cat.script(&[Fault::Held, Fault::Stale400]);
    c.cat.land_held_at(When::BeforeRequest(3));
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(1));
    assert_eq!(c.cat.requests(), 3);
    assert!(exists(&c.list(1)) && !exists(&c.list(2)));
    assert!(exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE]);
}

/// R:S2, §7.3 step 4a.0: a replaced table (another UUID) is
/// `KRISHIV_ICEBERG_COMMIT_CONFLICT`, never a rebase onto it; every outcome
/// known, so the write's files are deleted.
#[test]
fn a_replaced_table_is_a_conflict() {
    let mut c = case();
    let update: TableUpdate = serde_json::from_value(serde_json::json!({
        "action": "assign-uuid",
        "uuid": "0b8f1e0c-1d2e-4f3a-9b4c-5d6e7f809a1b",
    }))
    .unwrap();
    c.cat.on(
        When::BeforeRequest(1),
        Box::new(move |_| Box::pin(async move { (vec![], vec![update]) })),
    );
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(e.message().contains("UUID"), "{e}");
    assert_eq!(c.cat.requests(), 1);
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    c.check(&[INIT]);
}

/// Re-review nit 4: a conflict found while a request is still unresolved is
/// `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` naming the conflict, without the
/// conflict error's "files removed" text, and nothing is deleted.
#[test]
fn a_conflict_while_unresolved_names_it_and_deletes_nothing() {
    let mut c = case();
    c.cat.script(&[Fault::Held]);
    let update: TableUpdate = serde_json::from_value(serde_json::json!({
        "action": "assign-uuid",
        "uuid": "0b8f1e0c-1d2e-4f3a-9b4c-5d6e7f809a1b",
    }))
    .unwrap();
    c.cat.on(
        When::BeforeRequest(2),
        Box::new(move |_| Box::pin(async move { (vec![], vec![update]) })),
    );
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitStateUnknown,
        "{e}"
    );
    assert!(e.message().contains("UUID"), "{e}");
    assert!(!e.message().contains("removed"), "{e}");
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
}

/// R:S2, §7.3 step 4a.0: a v1 → v3 upgrade under an append is a conflict
/// (only v2 → v3 is rebased; Java reuses its manifests across any upgrade,
/// a divergence toward refusal).
#[test]
fn a_v1_to_v3_upgrade_is_a_conflict() {
    let mut c = case_with(1, &[]);
    c.cat.on(When::BeforeRequest(1), upgrade(FormatVersion::V3));
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(e.message().contains("from 1 to 3"), "{e}");
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    c.check(&[INIT]);
}

/// A small deterministic generator (splitmix64) for the schedule sweep.
/// `commit.manifest.min-count-to-merge` 2: the write's first attempt merges
/// its new manifest with the base's.
const MERGE: &[(&str, &str)] = &[("commit.manifest.min-count-to-merge", "2")];

impl Case {
    /// The manifests the list at `list` names.
    fn listed(&self, list: &str) -> Vec<String> {
        let io = self.env.reg.file_io();
        let version = self.base.format_version() as u8;
        self.env
            .rt
            .block_on(read_manifest_files(&io, list, version, MAX))
            .unwrap()
            .into_iter()
            .map(|m| m.manifest_path)
            .collect()
    }
}

/// A merged first attempt that commits: its list names the merged manifest
/// alone; the new manifest it absorbed is deleted, the merged one and the
/// data file kept.
#[test]
fn a_merged_first_attempt_commits_and_drops_the_absorbed_manifest() {
    let mut c = case_with(2, MERGE);
    c.run().unwrap();
    assert_eq!(c.new.merged.len(), 1);
    let merged = c.new.merged[0].clone();
    assert_eq!(c.listed(&c.list(1)), std::slice::from_ref(&merged));
    assert!(exists(&merged) && exists(&c.list(1)) && exists(&c.data()));
    assert!(!exists(&c.manifest()));
    c.check(&[INIT, MINE]);
}

/// A conflict after a merged first attempt: the second attempt lists the
/// unmerged manifests (the new one first, then the foreign head's) and
/// merges nothing; on commit the first attempt's list and merged manifest
/// are deleted, the new manifest kept.
#[test]
fn a_conflict_after_a_merged_first_attempt_lists_the_unmerged_manifests() {
    let mut c = case_with(2, MERGE);
    c.cat
        .on(When::BeforeRequest(1), c.foreign("f.parquet", 900, "main"));
    let m = c.run().unwrap();
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(2));
    assert_eq!(c.new.merged.len(), 1);
    let listed = c.listed(&c.list(2));
    assert_eq!(listed[0], c.manifest());
    assert!(!listed.contains(&c.new.merged[0]), "{listed:?}");
    assert!(!exists(&c.new.merged[0]) && !exists(&c.list(1)));
    assert!(exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, MINE, "900|1|f.parquet"]);
}

/// A known failure of a merged first attempt deletes its merged manifest
/// with everything else.
#[test]
fn a_known_failure_deletes_the_merged_manifest() {
    let mut c = case_with(2, MERGE);
    c.cat.script(&[Fault::Unauthorized401]);
    c.run().unwrap_err();
    assert_eq!(c.new.merged.len(), 1);
    assert!(!exists(&c.new.merged[0]) && !exists(&c.manifest()) && !exists(&c.data()));
    assert!(!exists(&c.list(1)));
    // A base manifest is never deleted.
    let base = c
        .base
        .current_snapshot()
        .unwrap()
        .manifest_list()
        .to_owned();
    assert!(c.listed(&base).iter().all(|m| exists(m)));
    c.check(&[INIT]);
}

/// An unknown answer to a merged first attempt, a later known failure: the
/// merged manifest is kept with everything else, and when a rollback lets
/// the held first attempt land, it reads back.
#[test]
fn an_unknown_merged_attempt_keeps_its_merged_manifest() {
    let mut c = case_with(1, MERGE);
    c.cat.script(&[Fault::Held]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("f1.parquet", 961, "main"));
    for n in 2..=5 {
        c.cat.on(
            When::BeforeRequest(n),
            c.foreign(
                &format!("f{n}.parquet"),
                960 + i64::try_from(n).unwrap(),
                "main",
            ),
        );
    }
    let e = c.run().unwrap_err();
    assert!(e.message().contains("files were kept"), "{e}");
    assert_eq!(c.new.merged.len(), 1);
    let merged = c.new.merged[0].clone();
    assert!(exists(&merged) && exists(&c.manifest()) && exists(&c.data()));
    let base = c.base.current_snapshot_id().unwrap();
    let back = rollback(base);
    c.env.rt.block_on(async {
        let (m, _) = c.cat.metadata();
        let request = back(m).await;
        c.cat.state.lock().unwrap().land(&request).ok().unwrap();
    });
    assert_eq!(c.cat.land_held(), 1);
    let (m, _) = c.cat.metadata();
    assert_eq!(m.snapshot_by_id(OURS).unwrap().manifest_list(), c.list(1));
    assert_eq!(c.listed(&c.list(1)), [merged]);
    c.check(&[INIT, MINE]);
}

struct Seeded(u64);

impl Seeded {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
    }

    fn when(&mut self) -> When {
        match self.below(3) {
            0 => When::BeforeRequest(1 + self.below(5)),
            1 => When::AfterRequest(1 + self.below(5)),
            _ => When::BeforeLoad(1 + self.below(20)),
        }
    }
}

/// One generated schedule: a fault per request, up to three foreign commits
/// (on `main` or `dev`) at random points, failing loads, and held requests
/// landed at a random point or after the run.
fn sweep_one(seed: u64, version: u8) {
    let mut g = Seeded(seed);
    // One schedule in three merges on the first attempt (from the seed, so
    // the other draws, and the schedules, stay as they were).
    let merging = seed.is_multiple_of(3);
    let mut c = case_with(version, if merging { MERGE } else { &[] });
    let mut faults: Vec<Fault> = (0..5)
        .map(|_| match g.below(11) {
            0 => Fault::LostResponse,
            1 => Fault::DelayedApply(g.below(8)),
            2 => Fault::Held,
            3 => Fault::Conflict409,
            4 => Fault::Stale400,
            5 => Fault::DuplicateId400,
            6 => Fault::Unauthorized401,
            7 => Fault::ServerErrorBeforeApply,
            8 => Fault::Duplicated,
            _ => Fault::Answer,
        })
        .collect();
    let mut main_foreign = Vec::new();
    for n in 0..g.below(4) {
        let id = 980 + i64::try_from(n).unwrap();
        let name = format!("s{n}.parquet");
        // v1 metadata keeps no refs, so other branches are v2 only.
        let branch = if g.below(3) == 0 && version > 1 {
            "dev"
        } else {
            "main"
        };
        if branch == "main" {
            main_foreign.push((id, format!("{id}|1|{name}")));
        }
        c.cat.on(g.when(), c.foreign(&name, id, branch));
    }
    // One schedule in four fails 4-8 loads, enough to outlast a conflict's
    // refresh (B1's shape: a conflict, then no successful load).
    let failing = if g.below(4) == 0 {
        4 + g.below(5)
    } else {
        g.below(3)
    };
    c.cat.fail_loads(failing);
    if g.below(2) == 0 {
        c.cat.land_held_at(g.when());
    }
    // One schedule in three rolls `main` back to the write's base (review-1
    // pass 2 P1), during the run or after it: a request settled as "not
    // applied" can then land.
    let rolled = g.below(3) == 0;
    let base = c.base.current_snapshot_id().unwrap();
    let roll_after = rolled && g.below(2) == 0;
    // On v1 those schedules hold the first request: the shape of the P1
    // (unknown, the branch moves, a known failure, the rollback, the held
    // request lands).
    if version == 1 && roll_after {
        faults[0] = Fault::Held;
    }
    c.cat.script(&faults);
    if rolled && !roll_after {
        c.cat.on(g.when(), rollback(base));
    }
    let schedule = format!("seed {seed}: {faults:?}, rollback {rolled}, merging {merging}");
    let out = c.run();
    assert_eq!(
        c.new.merged.len(),
        usize::from(merging),
        "{schedule}: merged only on the first attempt"
    );
    // Files the write produced before any late delivery.
    let data = exists(&c.data());
    let manifest = exists(&c.manifest());
    let merged = c.new.merged.iter().all(|m| exists(m));
    let list1 = exists(&c.list(1));
    let ours_now = |c: &Case| {
        c.cat.metadata().0.snapshots().any(|s| {
            s.summary().additional_properties.get(WRITE_ID_PROP) == Some(&OUR_KEY.to_owned())
        })
    };
    match &out {
        Ok(m) => {
            let s = m.snapshot_by_id(OURS);
            assert!(s.is_some(), "{schedule}");
            // Every manifest the committed list names exists (a merged
            // first attempt's list names the merged manifest, not the new
            // one it absorbed).
            let listed = s.map(|s| c.listed(s.manifest_list())).unwrap_or_default();
            assert!(
                data && listed.iter().all(|p| exists(p)),
                "{schedule}: a committed write's file deleted"
            );
        }
        Err(e) if e.error_class() == ErrorClass::KrishivIcebergCommitStateUnknown => {
            assert!(
                data && manifest && merged && list1,
                "{schedule}: an unresolved write deleted a file: {e}"
            );
        }
        // Every request known: never in the table, a rollback or not.
        Err(e) if e.message().contains("removed where possible") => {
            assert!(!ours_now(&c), "{schedule}: failed but committed: {e}");
        }
        Err(e) => {
            if e.message().contains("files were kept") {
                assert!(
                    data && manifest && merged,
                    "{schedule}: kept files deleted: {e}"
                );
            }
            assert!(
                rolled || !ours_now(&c),
                "{schedule}: failed but committed: {e}"
            );
        }
    }
    if roll_after {
        c.env.rt.block_on(async {
            let (m, _) = c.cat.metadata();
            let request = rollback(base)(m).await;
            assert!(c.cat.state.lock().unwrap().land(&request).is_ok());
        });
    }
    c.cat.land_held();
    // Delayed requests still pending land on later loads.
    for _ in 0..10 {
        let _ = c.env.rt.block_on(c.cat.load(true));
    }
    // A known failure lands later only through a rollback after an unknown
    // answer (the accepted remainder), never when every request was known.
    if let Err(e) = &out
        && e.error_class() != ErrorClass::KrishivIcebergCommitStateUnknown
        && (!rolled || e.message().contains("removed where possible"))
    {
        assert!(
            !ours_now(&c),
            "{schedule}: a known failure landed later: {e}"
        );
    }
    // The rows of `main`'s ancestry; reading them fails if a landed snapshot
    // names a deleted file.
    let (m, _) = c.cat.metadata();
    let ancestry = main_ancestry(&m);
    let mut expected: Vec<String> = vec![INIT.to_owned()];
    expected.extend(
        main_foreign
            .iter()
            .filter(|(id, _)| ancestry.contains(id))
            .map(|(_, row)| row.clone()),
    );
    if ancestry.contains(&OURS) {
        expected.push(MINE.to_owned());
    }
    expected.sort_by_key(|r| r.split('|').next().unwrap().parse::<i64>().unwrap());
    let expected: Vec<&str> = expected.iter().map(String::as_str).collect();
    c.check(&expected);
}

/// R:S3: generated schedules (seeded, on virtual time). After each: the
/// write in at most one snapshot; committed → its files there; a known
/// failure → never in the table, even after late deliveries; unknown →
/// nothing deleted; and once every held or delayed request has landed,
/// every live file listed once and the rows those of the commits that are
/// in the table, read through SQL and iceberg-rust's planning.
#[test]
fn generated_schedules_keep_the_invariants() {
    for seed in 0..300 {
        sweep_one(seed, 2);
    }
}

/// The sweep on a v1 table, where no sequence number stops a request settled
/// as "not applied" from landing after a rollback (review-1 pass 2 P1).
#[test]
fn generated_schedules_on_v1_keep_the_invariants() {
    for seed in 0..200 {
        sweep_one(seed, 1);
    }
}
