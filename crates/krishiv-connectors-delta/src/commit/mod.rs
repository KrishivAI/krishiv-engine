// SPDX-License-Identifier: Apache-2.0
//! Commits, conflicts and retry (design note P3-T021 section 5, PR-37a's named deliverable).
//!
//! [`commit`] takes what a write or DML statement staged (the add rows of the files it wrote,
//! the scan rows of the files it removes, what it read) and commits it on the snapshot it read:
//!
//! 1. Krishiv's writer allow-list ([`crate::features::check_write`]) first.
//! 2. One kernel transaction per attempt, built on the log pool with the `FileSystemCommitter`
//!    (a `put_if_absent` of the next commit file), carrying `commitInfo.tags."krishiv.writeId"`
//!    (the write's id, fixed for all attempts), `isolationLevel = Serializable` and
//!    `readVersion`.
//! 3. A lost race (`Conflicted`) reads every newer commit ([`winner`]): one carrying this write's
//!    id means the commit landed; otherwise each is checked in order ([`conflict`]), and the
//!    transaction is built again on the snapshot **at the last checked version**, never at the
//!    latest one, so a commit that lands meanwhile is checked on the next round.
//! 4. Any other failure of an attempt (a store failure, or an error from the kernel's commit,
//!    which can come after the commit file was written) first runs the **landed check**: the
//!    commits from the read version on are searched for this write's id. Found means success;
//!    not found raises the failure, as Delta raises a store failure (a kernel error before the
//!    put is deterministic, so retrying it would only spin). The landed check runs on the log
//!    pool, so a query cancelled by then raises `OPERATION_CANCELED` without it: the commit's
//!    outcome is then unknown (a put that started always ran to its end).
//!
//! Attempts are bounded by Delta's `spark.databricks.delta.maxCommitAttempts` (attempts = max +
//! 1, as Delta counts) and a total time ([`RetryLimits`]), checked after each lost race before
//! its winners are checked; exhausting either is `DELTA_MAX_COMMIT_RETRIES_EXCEEDED` with
//! Delta's parameters (the failed version is the one last attempted). Between conflicted attempts the
//! loop waits 100 ms to 60 s, doubling, with jitter; the wait ends early when the query is
//! cancelled.
//!
//! # Known differences
//! - Delta retries with no wait between attempts and no time bound; Krishiv has both.
//! - Delta's idempotency key is `commitInfo.txnId`; the kernel owns that field and regenerates it
//!   per attempt, so Krishiv's is the `krishiv.writeId` tag (ADR 0014 update 2026-11-13).
//! - Every Delta table is checked at Serializable, also one whose stored level is
//!   `WriteSerializable` (stricter, never looser).

pub mod conflict;
mod run;
pub mod winner;

use std::collections::HashMap;
use std::time::Duration;

use arrow_array::RecordBatch;
use delta_kernel::expressions::PredicateRef;

pub use self::run::commit;
pub(crate) use self::run::{map_row, write_id_row, write_id_schema};
use self::winner::FileAction;

/// The commitInfo tag holding a write's id.
pub const WRITE_ID_TAG: &str = "krishiv.writeId";

/// What a statement read of the table it commits to, for every scan of it (section 5.1).
#[derive(Debug, Clone, Default)]
pub struct ReadSet {
    /// It read the whole table (some scan had no partition conjunct).
    pub whole_table: bool,
    /// The partition-column conjuncts pushed to the kernel scans, one predicate per scan, over
    /// the partition columns' names.
    pub predicates: Vec<PredicateRef>,
    /// The files the scans' metadata returned, with their partition values.
    pub files: Vec<FileAction>,
    /// The app ids of `txn` lookups.
    pub app_ids: Vec<String>,
}

impl ReadSet {
    /// Whether the statement read the table at all.
    #[must_use]
    pub fn reads_table(&self) -> bool {
        self.whole_table || !self.predicates.is_empty() || !self.files.is_empty()
    }
}

/// The operation a commit records (`commitInfo.operation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitOperation {
    /// `WRITE`: append, overwrite, `replaceWhere`.
    Write,
    /// `DELETE`.
    Delete,
    /// `UPDATE`.
    Update,
    /// `MERGE`.
    Merge,
}

impl CommitOperation {
    fn kernel(self) -> delta_kernel::transaction::UpdateTableOperation {
        use delta_kernel::transaction::UpdateTableOperation as K;
        match self {
            Self::Write => K::Write,
            Self::Delete => K::Delete,
            Self::Update => K::Update,
            Self::Merge => K::Merge,
        }
    }
}

/// Files a commit removes: rows of the kernel's scan metadata, as its scan returned them, and
/// which of them are removed.
#[derive(Debug, Clone)]
pub struct StagedRemoves {
    /// The scan metadata rows (`path`, `fileConstantValues.partitionValues`, …).
    pub scan_rows: RecordBatch,
    /// One flag per row: removed or not.
    pub selection: Vec<bool>,
}

/// What a statement commits.
#[derive(Debug, Clone)]
pub struct StagedCommit {
    /// The table's name or path, for messages.
    pub table: String,
    /// The write's id (a UUID), fixed for all attempts.
    pub write_id: String,
    /// The operation.
    pub operation: CommitOperation,
    /// `commitInfo.operationParameters`, already stringified.
    pub operation_parameters: Vec<(String, Option<String>)>,
    /// `commitInfo.operationMetrics`, already stringified.
    pub operation_metrics: Vec<(String, Option<String>)>,
    /// Rows of the files written, in the kernel's `add_files_schema`.
    pub adds: Vec<RecordBatch>,
    /// The files removed.
    pub removes: Vec<StagedRemoves>,
    /// What the statement read of the table.
    pub read_set: ReadSet,
}

/// The bounds of the commit loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryLimits {
    /// Delta's `spark.databricks.delta.maxCommitAttempts`: up to this many retries, so one more
    /// attempt in all.
    pub max_attempts: u64,
    /// The total time of all attempts (`krishiv.write.commitRetryTimeout`).
    pub timeout: Duration,
    /// The first wait after a conflicted attempt.
    pub backoff_min: Duration,
    /// The longest wait.
    pub backoff_max: Duration,
}

/// Delta's default `maxCommitAttempts`.
pub const DEFAULT_MAX_COMMIT_ATTEMPTS: u64 = 10_000_000;

impl Default for RetryLimits {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_COMMIT_ATTEMPTS,
            timeout: Duration::from_secs(30 * 60),
            backoff_min: Duration::from_millis(100),
            backoff_max: Duration::from_secs(60),
        }
    }
}

impl RetryLimits {
    /// The limits `config` sets: `krishiv.delta.maxCommitAttempts` (Spark's
    /// `spark.databricks.delta.maxCommitAttempts`) and `krishiv.write.commitRetryTimeout`; the
    /// backoff keeps its defaults.
    #[must_use]
    pub fn from_config(config: &krishiv_common::config::Config) -> Self {
        use krishiv_common::config::{DELTA_MAX_COMMIT_ATTEMPTS, WRITE_COMMIT_RETRY_TIMEOUT};
        let d = Self::default();
        Self {
            max_attempts: config
                .get(&DELTA_MAX_COMMIT_ATTEMPTS)
                .and_then(krishiv_common::config::ConfigValue::as_int)
                .and_then(|n| u64::try_from(n).ok())
                .unwrap_or(d.max_attempts),
            timeout: config
                .get(&WRITE_COMMIT_RETRY_TIMEOUT)
                .and_then(krishiv_common::config::ConfigValue::as_duration)
                .unwrap_or(d.timeout),
            ..d
        }
    }
}

/// A commit that landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Committed {
    /// Its version.
    pub version: u64,
    /// Attempts made.
    pub attempts: u64,
    /// It was found by the landed check after a lost answer, not reported by the commit.
    pub found_by_landed_check: bool,
    /// Commits since the last checkpoint, when the commit reported it (section 4.3).
    pub commits_since_checkpoint: Option<u64>,
}

/// Partition values by physical name.
pub type PartitionValues = HashMap<String, Option<String>>;

#[cfg(test)]
mod hook_store;
#[cfg(test)]
mod tests;
