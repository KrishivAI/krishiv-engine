// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `MERGE INTO` on a Delta table (design note P3-T021 sections 2.3, 6.1 and 6.2;
//! PR-37c): what [`DeltaRowLevel`](super::dml::DeltaRowLevel) commits for a `MERGE`.
//!
//! The session plans the statement (krishiv-session `table_commands/merge.rs`): the touched
//! files, the counts of each kind of change ([`MERGE_COUNTS_OPTION`]) and the clauses
//! ([`MERGE_CLAUSES_OPTION`]) arrive with the rewrite; its rows are the touched files' rows that
//! stay or change and the inserted rows. The commit is delta-spark's (`MergeIntoCommand.scala:
//! 85-205`): the touched files removed, the written files added, operation `MERGE`, nothing
//! committed when nothing changed (`commitIfNeeded`).
//!
//! # Delta's rules ported
//! - `operationParameters` (`DeltaOperations.scala:328-367`): `predicate` (the `ON` condition),
//!   `matchedPredicates`, `notMatchedPredicates` and `notMatchedBySourcePredicates`, each a JSON
//!   array of `{"predicate": .., "actionType": ..}` in clause order (`predicate` absent for a
//!   clause without a condition).
//! - `operationMetrics` (`DeltaOperations.scala:1288-1313`, `MergeIntoCommandBase.scala:206-235`,
//!   `ClassicMergeExecutor.scala:200-220, 470-490`): the source rows, the target rows inserted,
//!   updated and deleted (in total and per clause kind), copied (written less updated and
//!   inserted), `numOutputRows` = inserted + updated + copied, files and bytes added and removed,
//!   the times; deletion vector counts are 0 (no vector is written).
//! - The result row (`MergeIntoCommand.scala:78-82, 183-192`): `num_affected_rows`,
//!   `num_updated_rows`, `num_deleted_rows`, `num_inserted_rows`.
//! - The read set (`ClassicMergeExecutor.scala:85-92`, `InsertOnlyMergeExecutor.scala:76-82`):
//!   the `ON` condition's target conjuncts (Krishiv reads the partition-only ones; a wider read),
//!   the whole table with a `NOT MATCHED BY SOURCE` clause, and every candidate file.
//!
//! # Refusals (before any file is written)
//! - On a table with the change data feed on, a `MERGE` that removes a file
//!   (`UNSUPPORTED_FEATURE.TABLE_OPERATION`, design section 6.2): delta-spark writes change
//!   data files there. A `MERGE` that only inserts commits.
//! - On an append-only table, a `MERGE` that removes a file: `DELTA_CANNOT_MODIFY_APPEND_ONLY`.
//!
//! # Known differences
//! - The predicates are the conditions as written; delta-spark prints Catalyst's
//!   `simpleString` of the resolved expressions (expression ids, unquoted strings).
//! - Deletion vectors are never written (PR-37d): on a table with them on, delta-spark's
//!   default `merge.deletionVectors.persistent = true` marks rows in vectors; Krishiv rewrites.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use serde_json::Value;

use krishiv_common::error::{InternalError, KrishivError};

/// The row-level option holding a `MERGE`'s counts, a JSON object (see krishiv-session
/// `table_commands/merge.rs`).
pub const MERGE_COUNTS_OPTION: &str = "__krishiv_row_level_merge_counts";

/// The row-level option holding a `MERGE`'s clauses, a JSON array.
pub const MERGE_CLAUSES_OPTION: &str = "__krishiv_row_level_merge_clauses";

/// A `MERGE`'s counts as the session found them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MergeCounts {
    pub(crate) source_rows: u64,
    pub(crate) matched_updated: u64,
    pub(crate) matched_deleted: u64,
    pub(crate) by_source_updated: u64,
    pub(crate) by_source_deleted: u64,
    /// `None`: every written row is an insert (an insert-only `MERGE`).
    pub(crate) inserted: Option<u64>,
    pub(crate) materialize_ms: u64,
}

/// One clause, for `operationParameters`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClauseRecord {
    kind: String,
    action: String,
    condition: Option<String>,
}

/// What a `MERGE` commits, beside the touched files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MergeInfo {
    pub(crate) counts: MergeCounts,
    pub(crate) clauses: Vec<ClauseRecord>,
    /// The `ON` condition as written.
    pub(crate) on: String,
    /// The rewrite's observation that counts the source rows, when the session did not count
    /// them first (an insert-only `MERGE` reads its source once): read at commit.
    pub(crate) source_rows_observation: Option<String>,
}

/// The final counts of a committed `MERGE`, for its result row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MergeOutcome {
    pub(crate) updated: u64,
    pub(crate) deleted: u64,
    pub(crate) inserted: u64,
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

impl MergeInfo {
    /// The `MERGE` the session's options describe.
    pub(crate) fn parse(counts: &str, clauses: &str, on: &str) -> Result<Self, KrishivError> {
        let c: Value =
            serde_json::from_str(counts).map_err(|_| internal("malformed MERGE counts"))?;
        let n = |k: &str| c.get(k).and_then(Value::as_u64).unwrap_or(0);
        let counts = MergeCounts {
            source_rows: n("source_rows"),
            matched_updated: n("matched_updated"),
            matched_deleted: n("matched_deleted"),
            by_source_updated: n("not_matched_by_source_updated"),
            by_source_deleted: n("not_matched_by_source_deleted"),
            inserted: c.get("inserted").and_then(Value::as_u64),
            materialize_ms: n("materialize_ms"),
        };
        let list: Vec<Value> =
            serde_json::from_str(clauses).map_err(|_| internal("malformed MERGE clauses"))?;
        let clauses = list
            .iter()
            .map(|v| {
                let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_owned);
                Ok(ClauseRecord {
                    kind: s("kind").ok_or_else(|| internal("a MERGE clause without a kind"))?,
                    action: s("action")
                        .ok_or_else(|| internal("a MERGE clause without an action"))?,
                    condition: s("condition"),
                })
            })
            .collect::<Result<_, KrishivError>>()?;
        Ok(Self {
            counts,
            clauses,
            on: on.to_owned(),
            source_rows_observation: c
                .get("source_rows_observation")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }

    /// These counts with the source rows read from the rewrite's observation
    /// ([`Self::source_rows_observation`]), once the rewrite's rows have all been written.
    ///
    /// # Errors
    /// The observation's own error, or `INTERNAL_ERROR` when the query holds no such
    /// observation (it wrote rows, so it read its source).
    pub(crate) fn with_observed(
        &self,
        query: &krishiv_exec::QueryContext,
    ) -> Result<Self, KrishivError> {
        let Some(name) = &self.source_rows_observation else {
            return Ok(self.clone());
        };
        let batch = query
            .observed_each()
            .remove(name)
            .ok_or_else(|| internal("a MERGE without its source row count"))??;
        let rows = batch
            .columns()
            .first()
            .and_then(|c| c.as_primitive_opt::<Int64Type>())
            .filter(|c| c.len() == 1 && c.is_valid(0))
            .and_then(|c| u64::try_from(c.value(0)).ok())
            .ok_or_else(|| internal("a malformed MERGE source row count"))?;
        let mut out = self.clone();
        out.counts.source_rows = rows;
        Ok(out)
    }

    /// The final counts given the rows written.
    pub(crate) fn outcome(&self, written_rows: u64) -> MergeOutcome {
        let c = &self.counts;
        MergeOutcome {
            updated: c.matched_updated + c.by_source_updated,
            deleted: c.matched_deleted + c.by_source_deleted,
            inserted: c.inserted.unwrap_or(written_rows),
        }
    }

    /// `operationParameters`.
    pub(crate) fn parameters(&self) -> Vec<(String, Option<String>)> {
        let of = |kind: &str| {
            // delta-spark's `JsonUtils` writes `predicate` before `actionType`.
            let string = |v: &str| Value::from(v).to_string();
            let items: Vec<String> = self
                .clauses
                .iter()
                .filter(|c| c.kind == kind)
                .map(|c| match &c.condition {
                    Some(p) => format!(
                        r#"{{"predicate":{},"actionType":{}}}"#,
                        string(&paren(p)),
                        string(&c.action)
                    ),
                    None => format!(r#"{{"actionType":{}}}"#, string(&c.action)),
                })
                .collect();
            format!("[{}]", items.join(","))
        };
        vec![
            (
                "predicate".to_owned(),
                Some(Value::Array(vec![Value::from(paren(&self.on))]).to_string()),
            ),
            ("matchedPredicates".to_owned(), Some(of("matched"))),
            ("notMatchedPredicates".to_owned(), Some(of("notMatched"))),
            (
                "notMatchedBySourcePredicates".to_owned(),
                Some(of("notMatchedBySource")),
            ),
        ]
    }

    /// `operationMetrics`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the commit's totals, each read once"
    )]
    pub(crate) fn metrics(
        &self,
        written_rows: u64,
        added_files: u64,
        added_bytes: u64,
        removed_files: u64,
        removed_bytes: u64,
        scan_ms: u64,
        rewrite_ms: u64,
    ) -> Vec<(String, u64)> {
        let c = &self.counts;
        let o = self.outcome(written_rows);
        let copied = written_rows
            .saturating_sub(o.inserted)
            .saturating_sub(o.updated);
        let mut m = vec![
            ("numSourceRows", c.source_rows),
            ("numTargetRowsInserted", o.inserted),
            ("numTargetRowsUpdated", o.updated),
            ("numTargetRowsMatchedUpdated", c.matched_updated),
            (
                "numTargetRowsNotMatchedBySourceUpdated",
                c.by_source_updated,
            ),
            ("numTargetRowsDeleted", o.deleted),
            ("numTargetRowsMatchedDeleted", c.matched_deleted),
            (
                "numTargetRowsNotMatchedBySourceDeleted",
                c.by_source_deleted,
            ),
            ("numTargetRowsCopied", copied),
            ("numTargetBytesAdded", added_bytes),
            ("numTargetBytesRemoved", removed_bytes),
            ("numOutputRows", o.inserted + o.updated + copied),
            ("numTargetFilesAdded", added_files),
            ("numTargetFilesRemoved", removed_files),
            ("numTargetChangeFilesAdded", 0),
            (
                "executionTimeMs",
                c.materialize_ms
                    .saturating_add(scan_ms)
                    .saturating_add(rewrite_ms),
            ),
            ("materializeSourceTimeMs", c.materialize_ms),
            ("scanTimeMs", scan_ms),
            ("rewriteTimeMs", rewrite_ms),
            ("numTargetDeletionVectorsAdded", 0),
            // delta-spark 4.4.1's copy-on-write MERGE reports 0 even when a removed file carried a
            // deletion vector (`merge/MERGE.json` `deletion-vectors/update_copy_on_write`).
            ("numTargetDeletionVectorsRemoved", 0),
            ("numTargetDeletionVectorsUpdated", 0),
        ];
        m.sort_by_key(|(k, _)| *k);
        m.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
    }
}

/// A condition as delta-spark's `predicate` strings show it: parenthesised.
fn paren(c: &str) -> String {
    let c = c.trim();
    if c.starts_with('(') && c.ends_with(')') {
        c.to_owned()
    } else {
        format!("({c})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts the session counted itself pass through unchanged; a named observation the query
    /// never produced is an internal error, never a zero source count.
    #[test]
    fn the_source_count_comes_from_the_named_observation_only() {
        use std::sync::Arc;

        use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let query = krishiv_exec::QueryContext::new(
            krishiv_exec::QueryId(1),
            krishiv_exec::WorkloadClass::Interactive,
            krishiv_exec::ExecConfig::default(),
            pool,
        );
        let counted = MergeInfo::parse(r#"{"source_rows":4}"#, "[]", "t.id = s.id").unwrap();
        assert_eq!(counted.with_observed(&query).unwrap(), counted);
        let observed = MergeInfo::parse(
            r#"{"source_rows_observation":"__krishiv_merge_source_rows"}"#,
            "[]",
            "t.id = s.id",
        )
        .unwrap();
        let e = observed.with_observed(&query).unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::InternalError,
            "{e}"
        );
        assert!(e.message().contains("source row count"), "{e}");
    }

    #[test]
    fn counts_and_parameters_follow_the_options() {
        let info = MergeInfo::parse(
            r#"{"source_rows":4,"matched_updated":1,"matched_deleted":1,
               "not_matched_by_source_updated":0,"not_matched_by_source_deleted":2,
               "inserted":2,"materialize_ms":3}"#,
            r#"[{"kind":"matched","action":"delete","condition":"s.n > 30"},
                {"kind":"matched","action":"update","condition":null},
                {"kind":"notMatched","action":"insert","condition":null},
                {"kind":"notMatchedBySource","action":"delete","condition":null}]"#,
            "t.id = s.id",
        )
        .unwrap();
        // 6 written rows: 2 inserted, 1 updated, 3 copied.
        let m: std::collections::HashMap<_, _> =
            info.metrics(6, 2, 10, 2, 20, 5, 7).into_iter().collect();
        assert_eq!(m["numTargetRowsCopied"], 3);
        assert_eq!(m["numOutputRows"], 6);
        assert_eq!(m["numTargetRowsDeleted"], 3);
        assert_eq!(m["executionTimeMs"], 15);
        let p = info.parameters();
        assert_eq!(p[0].1.as_deref(), Some(r#"["(t.id = s.id)"]"#));
        assert_eq!(
            p[1].1.as_deref(),
            Some(r#"[{"predicate":"(s.n > 30)","actionType":"delete"},{"actionType":"update"}]"#)
        );
        assert_eq!(p[3].1.as_deref(), Some(r#"[{"actionType":"delete"}]"#));
        // An insert-only MERGE: the written rows are the inserts.
        let only = MergeInfo::parse(r#"{"source_rows":3,"inserted":null}"#, "[]", "x").unwrap();
        assert_eq!(only.outcome(2).inserted, 2);
    }
}
