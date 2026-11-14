// SPDX-License-Identifier: Apache-2.0
//! The append commit loop (P3-T027, PR-36a; design note
//! `P3-T026-T027-iceberg-scan-and-commit.md` §7, Tier U): [`commit_append`]
//! sends the attempts [`prepare_attempt`] builds through a [`CommitTable`]
//! until one is known to be in the table, the retry budget runs out, or the
//! write fails.
//!
//! - **Never retried blind** (§7.2.1). A request whose outcome is unknown may
//!   still land, so it is followed by a probe (`load` with `snapshots=all`,
//!   immediately and then `commit.status-check.num-retries` more times with
//!   backoff): this write's snapshot id or `krishiv.write-id` in the table's
//!   snapshots means committed; the branch moved off the attempt's base with
//!   neither found means "not applied", final, and the write is rebased; the
//!   branch still at the base means the same attempt (same snapshot id, base
//!   and manifest list) is resent. At most one request of a write can apply:
//!   all carry one snapshot id, and each asserts its base.
//! - **Own-key search first** (§7.3 step 1). Every conflict refreshes the
//!   table and looks for this write's snapshot before rebasing, which closes
//!   "lost response, then 409" and "duplicated request, then 409". A 409 is
//!   known for its request, not for the write: until that search has run on
//!   a successful load the write is unresolved, so a refresh that keeps
//!   failing or runs into the total timeout ends
//!   `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` and deletes nothing (R:B1).
//! - **Per-request outcomes** (§7.2). A failure is "known" only as the answer
//!   to an attempt's first request; once a request of an attempt was unknown,
//!   later answers to it say nothing about that request. An attempt left
//!   behind while still unresolved (a stale answer to a resend, the branch
//!   never moved) stays unresolved until a later load shows the branch off its
//!   base.
//! - **Rebase** (§7.3 steps 4a.0, 4a.1, 4a.3): an append validates nothing; it
//!   keeps its manifests (reset to unassigned by [`prepare_attempt`]) and
//!   recomputes the list, sequence number, totals, v3 row ids and timestamp
//!   against the refreshed table. Across a v2 → v3 upgrade it takes the v3
//!   lineage fields; any other change of format version, or a replaced table
//!   (another UUID), is `KRISHIV_ICEBERG_COMMIT_CONFLICT`. A divergence
//!   toward refusal: Java reuses its manifests across any upgrade
//!   (apache-iceberg 1.12.0 `FastAppend.apply`, `SnapshotProducer.apply`),
//!   v1 → v2 and v1 → v3 included.
//! - **Budget**, Iceberg's table properties and defaults (apache-iceberg
//!   1.12.0 `TableProperties`): `COMMIT_NUM_RETRIES` (4: five requests, resends
//!   and new attempts alike), `COMMIT_MIN_RETRY_WAIT_MS` (100),
//!   `COMMIT_MAX_RETRY_WAIT_MS` (60 s), `COMMIT_TOTAL_RETRY_TIME_MS` (30 min,
//!   probe waits included), `COMMIT_NUM_STATUS_CHECKS` (3),
//!   `COMMIT_STATUS_CHECKS_MIN_WAIT_MS` (1 s), `COMMIT_STATUS_CHECKS_MAX_WAIT_MS`
//!   (60 s). Waits are Java's `Tasks.runTaskWithRetry` backoff: after the n-th
//!   failure `min(min-wait * 2^(n-1), max-wait)` plus a jitter below a tenth of
//!   it. The total time counts from the first request: a wait is skipped and
//!   the budget ends once `COMMIT_TOTAL_RETRY_TIME_MS` has passed, so a
//!   first failure after the timeout retries no more (not yet checked against
//!   Java's `Tasks.runTaskWithRetry`, which may allow one retry there; the
//!   source was not readable in the review sessions, review N1). A property
//!   that is not a number is `KRISHIV_UNSUPPORTED_FEATURE` (Java:
//!   `NumberFormatException`), before any request.
//! - **Outcomes and cleanup** (§7.4): committed → the table's new metadata;
//!   the manifest lists of every other attempt and the manifests the found
//!   snapshot does not list are deleted, never a data file. Every outcome known
//!   and none applied → everything the write produced is deleted (data files,
//!   manifests, lists) and the error is `KRISHIV_ICEBERG_COMMIT_FAILED` when
//!   the budget ran out, else the failure itself. Known, none applied, but
//!   some request of the write once had an unknown answer → the same error,
//!   and **nothing is deleted**: an attempt settled as "not applied" because
//!   the branch left its base still lands if another writer rolls the branch
//!   back to that base, so its files must exist (review-1 pass 2, P1; Java
//!   never cleans up after an unknown outcome either). The error then says
//!   the files were kept (`<files>`), except a plain failure
//!   ([`CommitOutcome::Failed`] or a local error), which is returned as is.
//!   The error can still be wrong in that schedule (`FAILED`, then the rows
//!   appear): the accepted remainder. Any outcome unresolved →
//!   `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` (40003) and **nothing is deleted**:
//!   a late request would otherwise reference deleted files.
//!
//! This differs from Java, which never resends and throws
//! `CommitStateUnknownException` once its status check gives up (R:B2).
//! A resend repeats the attempt exactly, `timestamp-ms` included.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use async_trait::async_trait;
use iceberg::io::FileIO;
use iceberg::spec::{FormatVersion, Snapshot, TableMetadata};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{
    KrishivError, KrishivIcebergCommitConflict, KrishivIcebergCommitFailed,
    KrishivIcebergCommitStateUnknown, KrishivUnsupportedFeature,
};

use super::append::{Attempt, NewManifests, WRITE_ID_PROP, WriteIds, prepare_attempt};
use super::scan::{read_manifest_files, snapshot_refs};

/// Java's `TableProperties.COMMIT_NUM_RETRIES`.
const COMMIT_NUM_RETRIES: &str = "commit.retry.num-retries";
/// Java's `TableProperties.COMMIT_MIN_RETRY_WAIT_MS`.
const COMMIT_MIN_RETRY_WAIT_MS: &str = "commit.retry.min-wait-ms";
/// Java's `TableProperties.COMMIT_MAX_RETRY_WAIT_MS`.
const COMMIT_MAX_RETRY_WAIT_MS: &str = "commit.retry.max-wait-ms";
/// Java's `TableProperties.COMMIT_TOTAL_RETRY_TIME_MS`.
const COMMIT_TOTAL_RETRY_TIME_MS: &str = "commit.retry.total-timeout-ms";
/// Java's `TableProperties.COMMIT_NUM_STATUS_CHECKS`.
const COMMIT_NUM_STATUS_CHECKS: &str = "commit.status-check.num-retries";
/// Java's `TableProperties.COMMIT_STATUS_CHECKS_MIN_WAIT_MS`.
const COMMIT_STATUS_CHECKS_MIN_WAIT_MS: &str = "commit.status-check.min-wait-ms";
/// Java's `TableProperties.COMMIT_STATUS_CHECKS_MAX_WAIT_MS`.
const COMMIT_STATUS_CHECKS_MAX_WAIT_MS: &str = "commit.status-check.max-wait-ms";

/// What one commit request came to. An implementation maps its transport's
/// answers onto these (design note §7.2, Java's `ErrorHandlers`, R:M4); for
/// REST:
///
/// | Response | Outcome |
/// |---|---|
/// | 200 | [`Committed`](Self::Committed), the response's metadata |
/// | 409 `CommitFailedException` | [`Conflict`](Self::Conflict) `{ stale: false }` |
/// | 400 or 409 *stale* (sequence number, `first-row-id` or `timestamp-ms` behind the table) | [`Conflict`](Self::Conflict) `{ stale: true }` |
/// | 400 *duplicate snapshot id* | [`Unknown`](Self::Unknown): probably committed, never "failed" |
/// | 500, 502, 503, 504, a timeout, a reset after the body was sent | [`Unknown`](Self::Unknown) |
/// | 401, 403 (after one token refresh and resend), 404, 422, other 4xx | [`Failed`](Self::Failed) |
///
/// The loop, not the implementation, decides that a `Failed` answer to a
/// resend is still unknown (§7.2's per-request rule).
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "built only by a `CommitTable`: PR-17's catalog clients"
    )
)]
pub(crate) enum CommitOutcome {
    /// Applied; the table's new metadata.
    Committed(Box<TableMetadata>),
    /// Not applied: a requirement failed.
    Conflict {
        /// The request was behind the table (a commit on another branch).
        stale: bool,
    },
    /// It may have been applied; why the answer says nothing.
    Unknown(String),
    /// Refused.
    Failed(KrishivError),
}

/// A catalog's commit endpoint for one table (design note §7.2: Krishiv's,
/// as iceberg-rust 0.11 cannot send a commit Krishiv built).
#[async_trait]
pub(crate) trait CommitTable: Send + Sync {
    /// The table's name, for messages.
    fn name(&self) -> String;

    /// Sends one request: `requirements` checked and `updates` applied
    /// atomically ([`CommitOutcome`] has the status mapping).
    async fn commit(
        &self,
        requirements: Vec<TableRequirement>,
        updates: Vec<TableUpdate>,
    ) -> CommitOutcome;

    /// Loads the table's current metadata; `all_snapshots` asks for every
    /// snapshot (REST `snapshots=all`), not only those refs point at.
    ///
    /// # Errors
    /// The load's failure; the loop counts it as "still unknown".
    async fn load(&self, all_snapshots: bool) -> Result<TableMetadata, KrishivError>;
}

/// Time for the loop, injected so simulations run on virtual time.
#[async_trait]
pub(crate) trait CommitClock: Send + Sync {
    /// Milliseconds since the epoch (the attempts' `timestamp-ms`, the
    /// total timeout).
    fn now_ms(&self) -> i64;

    /// Waits `wait`.
    async fn sleep(&self, wait: Duration);

    /// A uniformly random number of milliseconds in `[0, bound)`; `bound` is
    /// at least 1.
    fn jitter_ms(&self, bound: u64) -> u64;
}

/// What the loop reads and writes through.
#[derive(Clone, Copy)]
pub(crate) struct CommitContext<'a> {
    /// The table's file access (manifest lists written, files deleted).
    pub(crate) io: &'a FileIO,
    /// Time.
    pub(crate) clock: &'a dyn CommitClock,
    /// `krishiv.catalog.maxMetadataBytes`, for the manifest lists read.
    pub(crate) max_metadata_bytes: u64,
}

/// Commits the append `new` (its manifests written against `base`, the
/// table as the write pinned it, S0) to the branch `ids.branch` of `table`
/// (module docs).
///
/// # Errors
/// `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` when any request's outcome is
/// unresolved at the end (nothing deleted); otherwise, with everything the
/// write produced deleted unless some request ever had an unknown answer
/// (then nothing is deleted, module docs): `KRISHIV_ICEBERG_COMMIT_FAILED` when every request
/// was a known "not applied" and the budget ran out,
/// `KRISHIV_ICEBERG_COMMIT_CONFLICT` for a format-version change other than
/// v2 → v3 or a replaced table, `KRISHIV_UNSUPPORTED_FEATURE` for a commit
/// property that is not a number, a failure answered to an attempt's first
/// request, or [`prepare_attempt`]'s errors. A refresh after a conflict that
/// never succeeds is `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN`.
///
/// # Cancellation
/// Dropping the future once it has been polled past its first
/// [`CommitTable::commit`] call is an **unknown outcome** (R:S4): a request
/// may be in flight and still land. Nothing is deleted here, and the caller
/// must not delete the write's data files or manifests either (`sink.rs`
/// disarms its guard before calling this). Only a drop before the first
/// request says nothing was applied. A returned error other than
/// `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` says no request is in the table
/// now; one that had an unknown answer can still land later if the branch
/// returns to its base (module docs), so its files are kept.
pub(crate) async fn commit_append(
    table: &dyn CommitTable,
    cx: CommitContext<'_>,
    base: &TableMetadata,
    new: &mut NewManifests,
    ids: WriteIds<'_>,
) -> Result<TableMetadata, KrishivError> {
    let budget = match Budget::from_properties(base.properties()) {
        Ok(b) => b,
        Err(e) => {
            delete_all(cx.io, new, &[]).await;
            return Err(e);
        }
    };
    let mut run = Loop {
        table,
        cx,
        ids,
        snapshot_id: new.snapshot_id,
        budget,
        start_ms: cx.clock.now_ms(),
        s0_version: base.format_version(),
        s0_uuid: base.uuid().to_string(),
        requests: 0,
        lists: Vec::new(),
        unresolved: Vec::new(),
        searching: false,
        ever_unknown: false,
        last_answer: String::new(),
    };
    match run.drive(base.clone(), new).await {
        Ok(m) => {
            run.clean_committed(&m, new).await;
            Ok(m)
        }
        Err(end) => Err(run.finish(end, new).await),
    }
}

/// Why the loop stopped without a commit.
enum End {
    /// The budget (requests or time) ran out.
    Exhausted,
    /// A conflict with the table (§7.3), kept apart so that a run with a
    /// request still unresolved names it without the error's "files removed"
    /// text (review nit 4).
    Conflict(Box<KrishivIcebergCommitConflict>),
    /// An error; its own request's outcome (if any) known.
    Error(KrishivError),
}

/// What a probe found.
enum Probe {
    /// This write's snapshot is in the table.
    Found(TableMetadata),
    /// The branch moved off the attempt's base and the snapshot is absent.
    Moved(TableMetadata),
    /// Neither: the request may still land.
    StillUnknown,
}

/// An exponential backoff as Java's `Tasks.exponentialBackoff(min, max, _, 2.0)`.
#[derive(Debug, Clone, Copy)]
struct Backoff {
    min_ms: u64,
    max_ms: u64,
}

impl Backoff {
    /// The wait after the `n`-th failure (`n` from 1): Java's
    /// `min(min * 2^(n-1), max)` plus `nextInt(max(1, delay * 0.1))`.
    fn wait(self, n: u32, clock: &dyn CommitClock) -> Duration {
        let factor = 1_u64 << n.saturating_sub(1).min(62);
        let delay = self.min_ms.saturating_mul(factor).min(self.max_ms);
        let jitter = clock.jitter_ms((delay / 10).max(1));
        Duration::from_millis(delay.saturating_add(jitter))
    }
}

/// The retry budget, from the table's properties (module docs).
#[derive(Debug, Clone, Copy)]
struct Budget {
    num_retries: u32,
    retry: Backoff,
    total_ms: u64,
    status_checks: u32,
    status: Backoff,
}

impl Budget {
    fn from_properties(props: &HashMap<String, String>) -> Result<Self, KrishivError> {
        let num = |key: &str, default: i64| -> Result<u64, KrishivError> {
            let n = match props.get(key) {
                None => default,
                // No trim: Java's `Integer.parseInt` refuses " 4" (R:N2).
                Some(v) => v.parse::<i64>().map_err(|_| {
                    KrishivError::from(KrishivUnsupportedFeature {
                        feature: format!(
                            "committing to an Iceberg table whose {key} is {v:?}, not a number"
                        ),
                    })
                })?,
            };
            // A negative count retries never and a negative wait waits not
            // at all, as Java's `Tasks`.
            Ok(u64::try_from(n).unwrap_or(0))
        };
        let count = |key: &str, default: i64| {
            num(key, default).map(|n| u32::try_from(n).unwrap_or(u32::MAX))
        };
        Ok(Self {
            num_retries: count(COMMIT_NUM_RETRIES, 4)?,
            retry: Backoff {
                min_ms: num(COMMIT_MIN_RETRY_WAIT_MS, 100)?,
                max_ms: num(COMMIT_MAX_RETRY_WAIT_MS, 60_000)?,
            },
            total_ms: num(COMMIT_TOTAL_RETRY_TIME_MS, 30 * 60 * 1_000)?,
            status_checks: count(COMMIT_NUM_STATUS_CHECKS, 3)?,
            status: Backoff {
                min_ms: num(COMMIT_STATUS_CHECKS_MIN_WAIT_MS, 1_000)?,
                max_ms: num(COMMIT_STATUS_CHECKS_MAX_WAIT_MS, 60_000)?,
            },
        })
    }
}

/// One run of the loop.
struct Loop<'a> {
    table: &'a dyn CommitTable,
    cx: CommitContext<'a>,
    ids: WriteIds<'a>,
    snapshot_id: i64,
    budget: Budget,
    start_ms: i64,
    s0_version: FormatVersion,
    s0_uuid: String,
    /// Requests sent.
    requests: u32,
    /// Every manifest list an attempt wrote.
    lists: Vec<String>,
    /// The bases of attempts with a request whose outcome is unresolved.
    unresolved: Vec<Option<i64>>,
    /// A conflict was answered and the own-key search has not yet run on a
    /// successful load: the write may be in the table (a duplicated request,
    /// R:B1), so it is unresolved until then.
    searching: bool,
    /// Some request of this write had an unknown answer. Never cleared: an
    /// attempt settled as "not applied" because the branch left its base can
    /// still land if another writer rolls the branch back to that base, so a
    /// write that then fails keeps its files (review-1 pass 2, P1).
    ever_unknown: bool,
    /// The last answer that was not a commit, for messages.
    last_answer: String,
}

impl Loop<'_> {
    async fn drive(
        &mut self,
        mut current: TableMetadata,
        new: &mut NewManifests,
    ) -> Result<TableMetadata, End> {
        'attempts: loop {
            self.before_request().await?;
            let attempt = prepare_attempt(
                self.cx.io,
                &current,
                new,
                self.ids,
                self.cx.clock.now_ms(),
                self.cx.max_metadata_bytes,
            )
            .await
            .map_err(End::Error)?;
            self.lists.push(attempt.snapshot.manifest_list().to_owned());
            let base = attempt_base(&attempt);
            let mut unknown = false;
            let mut first = true;
            loop {
                if !first {
                    self.before_request().await?;
                }
                first = false;
                self.requests += 1;
                let outcome = self
                    .table
                    .commit(attempt.requirements.clone(), attempt.updates.clone())
                    .await;
                match outcome {
                    CommitOutcome::Committed(m) => return Ok(*m),
                    CommitOutcome::Failed(e) if !unknown => return Err(End::Error(e)),
                    CommitOutcome::Failed(e) => self.last_answer = e.to_string(),
                    CommitOutcome::Unknown(reason) => {
                        if !unknown {
                            unknown = true;
                            self.ever_unknown = true;
                            self.unresolved.push(base);
                        }
                        self.last_answer = reason;
                    }
                    CommitOutcome::Conflict { stale } => {
                        self.last_answer = if stale {
                            "a stale request (the table moved on another branch)".to_owned()
                        } else {
                            "a conflict (the branch moved)".to_owned()
                        };
                        // A 409 is known for this request, not for the
                        // write: until the own-key search has run on a
                        // successful load, the write is unresolved (§7.3
                        // step 1).
                        self.searching = true;
                        let m = self.refresh().await?;
                        self.searching = false;
                        if self.found(&m)?.is_some() {
                            return Ok(m);
                        }
                        self.resolve(&m);
                        self.check_rebase(&m)?;
                        current = m;
                        continue 'attempts;
                    }
                }
                match self.probe(base).await? {
                    Probe::Found(m) => return Ok(m),
                    Probe::Moved(m) => {
                        self.check_rebase(&m)?;
                        current = m;
                        continue 'attempts;
                    }
                    Probe::StillUnknown => {}
                }
            }
        }
    }

    /// Before every request but the first: the request count and the time
    /// left, then the retry wait.
    async fn before_request(&mut self) -> Result<(), End> {
        if self.requests == 0 {
            return Ok(());
        }
        if self.requests > self.budget.num_retries {
            return Err(End::Exhausted);
        }
        self.wait(self.budget.retry, self.requests).await
    }

    /// Waits the `n`-th wait of `backoff`, unless the total timeout has passed.
    async fn wait(&mut self, backoff: Backoff, n: u32) -> Result<(), End> {
        let elapsed = u64::try_from(self.cx.clock.now_ms() - self.start_ms).unwrap_or(0);
        if elapsed > self.budget.total_ms {
            return Err(End::Exhausted);
        }
        self.cx.clock.sleep(backoff.wait(n, self.cx.clock)).await;
        Ok(())
    }

    /// The table after a conflict, retried as a probe is.
    async fn refresh(&mut self) -> Result<TableMetadata, End> {
        let mut failures = 0;
        loop {
            match self.table.load(true).await {
                Ok(m) => return Ok(m),
                Err(e) => {
                    failures += 1;
                    if failures > self.budget.status_checks {
                        return Err(End::Error(e));
                    }
                    self.wait(self.budget.status, failures).await?;
                }
            }
        }
    }

    /// §7.2.1 steps 1-3 for an attempt based on `base`.
    async fn probe(&mut self, base: Option<i64>) -> Result<Probe, End> {
        for n in 0..=self.budget.status_checks {
            if n > 0 {
                self.wait(self.budget.status, n).await?;
            }
            let Ok(m) = self.table.load(true).await else {
                // A failed probe is "still unknown" (§7.2.1 step 1).
                continue;
            };
            if self.found(&m)?.is_some() {
                return Ok(Probe::Found(m));
            }
            self.resolve(&m);
            if matches!(self.head(&m), Some(head) if head != base) {
                return Ok(Probe::Moved(m));
            }
        }
        Ok(Probe::StillUnknown)
    }

    /// This write's snapshot in `m` (§7.3 step 1; the snapshot list of a
    /// `snapshots=all` load holds the branch's ancestry and every other
    /// snapshot), deterministically: the snapshot of this write's id when it
    /// carries this write's key, else the lowest id carrying the key (a
    /// replayed write, R:S1).
    ///
    /// # Errors
    /// `KRISHIV_ICEBERG_COMMIT_CONFLICT` when this write's snapshot id is
    /// taken by a snapshot of another (or no) write key and the key is
    /// nowhere (R:N3): never "committed".
    fn found<'m>(&self, m: &'m TableMetadata) -> Result<Option<&'m Snapshot>, End> {
        let ours = |s: &Snapshot| {
            s.summary()
                .additional_properties
                .get(WRITE_ID_PROP)
                .is_some_and(|w| w == self.ids.write_id)
        };
        let by_id = m.snapshot_by_id(self.snapshot_id).map(AsRef::as_ref);
        if let Some(s) = by_id.filter(|s| ours(s)) {
            return Ok(Some(s));
        }
        let by_key = m
            .snapshots()
            .map(AsRef::as_ref)
            .filter(|s| ours(s))
            .min_by_key(|s| s.snapshot_id());
        match (by_key, by_id) {
            (Some(s), _) => Ok(Some(s)),
            (None, None) => Ok(None),
            (None, Some(taken)) => Err(self.conflict(
                m,
                "snapshot id",
                format!(
                    "snapshot {} in the table belongs to another write ({}).",
                    self.snapshot_id,
                    taken
                        .summary()
                        .additional_properties
                        .get(WRITE_ID_PROP)
                        .map_or("no write id", String::as_str)
                ),
            )),
        }
    }

    /// The branch's head in `m` (`Some(None)`: no such branch), as
    /// [`prepare_attempt`] reads it; `None` when the refs cannot be read.
    fn head(&self, m: &TableMetadata) -> Option<Option<i64>> {
        let refs = snapshot_refs(m).ok()?;
        let branch = self.ids.branch;
        Some(match refs.get(branch) {
            Some(r) => Some(r.snapshot_id),
            None if branch == "main" => m.current_snapshot_id(),
            None => None,
        })
    }

    /// Settles the unresolved attempts whose base the branch has left in
    /// `m`, where this write's snapshot is absent (§7.2.1 step 3).
    fn resolve(&mut self, m: &TableMetadata) {
        if let Some(head) = self.head(m) {
            self.unresolved.retain(|base| *base == head);
        }
    }

    /// §7.3 step 4a.0 for an append: a rebase across a v2 → v3 upgrade only.
    fn check_rebase(&self, m: &TableMetadata) -> Result<(), End> {
        let to = m.format_version();
        let from = self.s0_version;
        let problem = if m.uuid().to_string() != self.s0_uuid {
            Some((
                "table UUID",
                format!(
                    "the table was replaced: its UUID is {}, not {}.",
                    m.uuid(),
                    self.s0_uuid
                ),
            ))
        } else if from != to && !(from == FormatVersion::V2 && to == FormatVersion::V3) {
            Some((
                "format version",
                format!(
                    "the table's format version changed from {} to {} after the write started.",
                    from as u8, to as u8
                ),
            ))
        } else {
            None
        };
        match problem {
            None => Ok(()),
            Some((check, message)) => Err(self.conflict(m, check, message)),
        }
    }

    /// `KRISHIV_ICEBERG_COMMIT_CONFLICT` against `m`'s branch head.
    fn conflict(&self, m: &TableMetadata, check: &str, message: String) -> End {
        End::Conflict(Box::new(KrishivIcebergCommitConflict {
            table: self.table.name(),
            snapshot_id: self.snapshot_id.to_string(),
            conflicting_snapshot_id: self
                .head(m)
                .flatten()
                .map_or_else(|| "none".to_owned(), |id| id.to_string()),
            check: check.to_owned(),
            message,
            files: REMOVED.to_owned(),
        }))
    }

    /// §7.4 committed: keep the found snapshot's list and the manifests it
    /// lists; delete every other list and manifest of this write; never a
    /// data file, never a list any snapshot of `m` names. Without the
    /// snapshot (it cannot be read back) nothing is deleted, as Java; nor
    /// when it is another id's (a replayed write) and a request of this one
    /// ever had an unknown answer, so it may still land (R:S1; settled or
    /// not, review of the P1 fix finding 1).
    async fn clean_committed(&self, m: &TableMetadata, new: &NewManifests) {
        let Ok(Some(found)) = self.found(m) else {
            return;
        };
        if found.snapshot_id() != self.snapshot_id && self.ever_unknown {
            return;
        }
        let keep = found.manifest_list();
        let named: HashSet<&str> = m.snapshots().map(|s| s.manifest_list()).collect();
        for list in self.lists.iter().filter(|l| !named.contains(l.as_str())) {
            let _ = self.cx.io.delete(list).await;
        }
        let Ok(listed) = read_manifest_files(
            self.cx.io,
            keep,
            m.format_version() as u8,
            self.cx.max_metadata_bytes,
        )
        .await
        else {
            return;
        };
        let listed: HashSet<&str> = listed.iter().map(|f| f.manifest_path.as_str()).collect();
        for manifest in &new.manifests {
            if !listed.contains(manifest.manifest_path.as_str()) {
                let _ = self.cx.io.delete(&manifest.manifest_path).await;
            }
        }
    }

    /// The error of a run that ended without a commit, after its cleanup.
    async fn finish(&self, end: End, new: &NewManifests) -> KrishivError {
        if !self.unresolved.is_empty() || self.searching {
            let reason = match end {
                End::Exhausted => format!(
                    "the retry budget ran out after {} requests, the last answered with: {}",
                    self.requests, self.last_answer
                ),
                End::Conflict(c) => format!(
                    "a conflict with the table at snapshot {} ({}): {}",
                    c.conflicting_snapshot_id, c.check, c.message
                ),
                End::Error(e) => e.to_string(),
            };
            return KrishivIcebergCommitStateUnknown {
                table: self.table.name(),
                snapshot_id: self.snapshot_id.to_string(),
                write_id: self.ids.write_id.to_owned(),
                reason,
            }
            .into();
        }
        let files = if self.ever_unknown {
            KEPT.to_owned()
        } else {
            delete_all(self.cx.io, new, &self.lists).await;
            REMOVED.to_owned()
        };
        match end {
            End::Exhausted => KrishivIcebergCommitFailed {
                table: self.table.name(),
                snapshot_id: self.snapshot_id.to_string(),
                write_id: self.ids.write_id.to_owned(),
                attempts: self.requests.to_string(),
                reason: self.last_answer.clone(),
                files,
            }
            .into(),
            End::Conflict(mut c) => {
                c.files = files;
                (*c).into()
            }
            End::Error(e) => e,
        }
    }
}

/// The `<files>` text of a failed write whose files were deleted.
const REMOVED: &str = "the written files were removed where possible, and the write \
     can run again";

/// The `<files>` text of a failed write that once had an unknown answer.
const KEPT: &str = "the written files were kept, since an earlier request of this \
     write had no definite answer and could still land if the branch returns to its \
     base: look for the write id in the table's snapshots before running the write \
     again, or its rows may appear twice (remove_orphan_files removes the files if \
     it never lands)";

/// The base an attempt asserts for its branch (`assert-ref-snapshot-id`).
fn attempt_base(attempt: &Attempt) -> Option<i64> {
    attempt.requirements.iter().find_map(|r| match r {
        TableRequirement::RefSnapshotIdMatch { snapshot_id, .. } => *snapshot_id,
        _ => None,
    })
}

/// §7.4 failed with every outcome known: everything the write produced.
/// Failures to delete leave orphans, never a wrong table.
async fn delete_all(io: &FileIO, new: &NewManifests, lists: &[String]) {
    let manifests = new.manifests.iter().map(|m| &m.manifest_path);
    for path in new.data_files.iter().chain(manifests).chain(lists) {
        let _ = io.delete(path).await;
    }
}
