// SPDX-License-Identifier: Apache-2.0
//! The decide-once pushdown seam: a scan that sees every conjunct of the filter over it at once, once
//! per physical plan, answers per conjunct, and holds the read it planned for the plan's life.
//!
//! The plan stays data: a [`ScanNode`](crate::physical::ScanNode) records a [`DecidedScan`]
//! (the id of the bound scan and the pushed conjunct indices); the [`BoundScan`] itself lives in the
//! query's [`TableProviderResolver`](crate::planner::TableProviderResolver), and the rows of one
//! execution in its [`ExecutionReads`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_schema::SchemaRef;
use krishiv_io::{ColumnPath, ScalarValue};

use super::scan::{PushedFilter, ScanPushdown, ScanPushdowns};
use crate::error::{Result, internal};

/// Spark's `org.apache.spark.sql.sources.Filter` leaves, plus `Not` of one: what a Python data source
/// receives. Not a pruning tree: [`PushdownExpr`](krishiv_io::PushdownExpr) has no `EndsWith` or
/// `EqualNullSafe` and never holds a null value.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceFilter {
    /// `column = value`.
    EqualTo(ColumnPath, ScalarValue),
    /// `column <=> value`.
    EqualNullSafe(ColumnPath, ScalarValue),
    /// `column > value`.
    GreaterThan(ColumnPath, ScalarValue),
    /// `column >= value`.
    GreaterThanOrEqual(ColumnPath, ScalarValue),
    /// `column < value`.
    LessThan(ColumnPath, ScalarValue),
    /// `column <= value`.
    LessThanOrEqual(ColumnPath, ScalarValue),
    /// `column IN (values)`.
    In(ColumnPath, Vec<ScalarValue>),
    /// `column IS NULL`.
    IsNull(ColumnPath),
    /// `column IS NOT NULL`.
    IsNotNull(ColumnPath),
    /// `column LIKE 'prefix%'`.
    StringStartsWith(ColumnPath, String),
    /// `column LIKE '%suffix'`.
    StringEndsWith(ColumnPath, String),
    /// `column LIKE '%part%'`.
    StringContains(ColumnPath, String),
    /// `NOT` of a leaf.
    Not(Box<SourceFilter>),
}

/// The three states of a conjunct in Spark's data source filter translation. Only `Serializable`
/// reaches Python; the residual groups by translated or not.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceConjunct {
    /// A leaf (or `Not` of a leaf) of the 12 kinds, with a value the variant cast encodes.
    Serializable(SourceFilter),
    /// Translated by Spark's `translateFilterWithMapping` but dropped by the Python runner: `And`/`Or`/`Not`
    /// of a non-leaf, `AlwaysTrue`/`AlwaysFalse`, a collated filter, a value the variant cast cannot encode.
    Translated,
    /// No data source filter form.
    Untranslated,
}

/// A conjunct of the filter over a scan, offered to a scan that decides once.
#[derive(Clone, Debug)]
pub struct OfferedConjunct {
    /// The conjunct alone, over the table schema.
    pub filter: PushedFilter,
    /// Its state in Spark's data source filter translation.
    pub source: SourceConjunct,
}

/// What a scan decided: which conjuncts it honours, and the read bound to the decision.
#[derive(Clone, Debug)]
pub struct ScanDecision {
    /// Per offered conjunct: honoured by the scan. The planner leaves it out of the `Filter` above.
    pub pushed: Vec<bool>,
    /// The planned read; it holds no rows.
    pub bound: Arc<dyn BoundScan>,
}

/// A decided scan's planned read. Lives as long as the plan; reads once per execution.
pub trait BoundScan: Send + Sync {
    /// The table schema (as [`ScanPushdown::table_schema`]).
    fn table_schema(&self) -> SchemaRef;
    /// The schema of the batches a read built with `pushdowns` returns, from the schema alone, never a read.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a row selection or `_metadata` columns, which an in-memory scan refuses.
    fn output_schema(&self, pushdowns: &ScanPushdowns) -> Result<SchemaRef>;
    /// The `EXPLAIN` text added to the scan's line.
    fn explain(&self, pushdowns: &ScanPushdowns) -> Option<String>;
    /// Reads the rows for one execution and returns the scan over them, which owns their reservation.
    ///
    /// # Errors
    /// Whatever the read raises (for Python, `PYTHON_EXCEPTION`).
    fn read(&self) -> Result<Arc<dyn ScanPushdown>>;
}

impl std::fmt::Debug for dyn BoundScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BoundScan")
    }
}

/// The number a [`TableProviderResolver`](crate::planner::TableProviderResolver) gave a bound scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BoundScanId(pub u32);

impl BoundScanId {
    /// The id of a deciding scan planned without its decision (a plan that is only looked at): it
    /// is never bound, and a plan that holds it does not run.
    pub const UNDECIDED: Self = Self(u32::MAX);
}

/// What a [`ScanNode`](crate::physical::ScanNode) records of a decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecidedScan {
    /// The bound scan, in the query's resolver.
    pub id: BoundScanId,
    /// Indices of the offered conjuncts the scan honoured, for `EXPLAIN` and tests.
    pub pushed: Vec<u32>,
}

impl DecidedScan {
    /// A deciding scan with a filter over it, planned with decisions off.
    #[must_use]
    pub fn undecided() -> Self {
        Self {
            id: BoundScanId::UNDECIDED,
            pushed: Vec::new(),
        }
    }

    /// Whether this scan was planned without its decision.
    #[must_use]
    pub fn is_undecided(&self) -> bool {
        self.id == BoundScanId::UNDECIDED
    }
}

/// The rows one execution read, by bound scan: shared by the execution's tasks, dropped with the job's
/// worker context (and the reservations with it). A plan run twice reads twice.
#[derive(Default)]
pub struct ExecutionReads(Mutex<HashMap<BoundScanId, Arc<dyn ScanPushdown>>>);

impl std::fmt::Debug for ExecutionReads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ExecutionReads")
    }
}

impl ExecutionReads {
    /// The execution's scan of `id`: the first caller reads (under the lock), the others share it.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a poisoned lock; the read's error.
    pub fn get_or_read(
        &self,
        id: BoundScanId,
        read: impl FnOnce() -> Result<Arc<dyn ScanPushdown>>,
    ) -> Result<Arc<dyn ScanPushdown>> {
        let mut reads = self
            .0
            .lock()
            .map_err(|_| internal("a poisoned execution-reads map".to_owned()))?;
        if let Some(scan) = reads.get(&id) {
            return Ok(Arc::clone(scan));
        }
        let scan = read()?;
        reads.insert(id, Arc::clone(&scan));
        Ok(scan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::MemoryScan;

    fn scan() -> Arc<dyn ScanPushdown> {
        let schema = Arc::new(arrow_schema::Schema::empty());
        Arc::new(MemoryScan::try_new(schema, Vec::new()).expect("empty scan"))
    }

    #[test]
    fn the_first_task_reads_and_the_others_share() {
        let reads = ExecutionReads::default();
        let mut calls = 0;
        let a = reads
            .get_or_read(BoundScanId(0), || {
                calls += 1;
                Ok(scan())
            })
            .expect("first read");
        let b = reads
            .get_or_read(BoundScanId(0), || {
                calls += 1;
                Ok(scan())
            })
            .expect("shared");
        assert_eq!(calls, 1);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn a_failed_read_is_not_kept() {
        let reads = ExecutionReads::default();
        assert!(
            reads
                .get_or_read(BoundScanId(1), || Err(internal("boom".to_owned())))
                .is_err()
        );
        assert!(reads.get_or_read(BoundScanId(1), || Ok(scan())).is_ok());
    }
}
