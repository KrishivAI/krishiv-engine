// SPDX-License-Identifier: Apache-2.0
//! The stateless operators over the executor traits (P2-T051): an in-memory
//! scan with projection and filter pushdown ([`MemoryScan`]), [`Filter`],
//! [`Project`], [`Limit`] and the per-branch alignment of a union
//! ([`UnionAlign`]); the sort breakers (P2-T054, [`sort`]): [`Sort`] and
//! [`TopN`]; and the file scan and write over any file format (P2-T117b):
//! [`FileScan`] / [`FileScanSource`] and [`FileWriteSink`].
//!
//! # Contracts shared by all of them
//! - **Weights** ([`Weights`](krishiv_types::batch::Weights)) are carried
//!   through: filtering and slicing keep each surviving row's weight, a
//!   projection keeps the input's weights as they are. `Limit` is defined
//!   for insert-only input (weight 1) and rejects anything else.
//! - **Empty output.** An operator never emits a batch with no rows
//!   (`NeedMoreInput(None)` instead), so downstream operators and sinks see
//!   only batches that carry work. They still accept empty input.
//! - **Memory.** Their output is handed to the next stage within the same
//!   `execute` call chain and is at most one input batch in size, so they
//!   keep no state worth reserving; the breakers that retain batches
//!   account for them (ADR 0006). The per-driver in-flight bytes are
//!   bounded by `krishiv.exec.batchSize` rows of the widest schema.
//! - **Expressions.** The kernels crate has no expression evaluator yet, so
//!   the physical planner (P2-T074) plugs in [`Predicate`] and
//!   [`ScalarExpr`] implementations; the operators only drive them.
//! - **`UNION`.** See [`union`] for how `UNION ALL` fan-in and the `UNION`
//!   (distinct) breaker are split between tasks.

mod expand;
mod expr;
mod file_scan;
mod file_write;
mod filter;
mod generate;
mod late_fetch;
mod limit;
mod partition_prune;
mod project;
mod range;
mod replicate;
mod runtime_filter;
mod sample;
mod scan;
mod scan_decide;
pub(crate) mod select;
mod sort;
mod spilled_scan;
pub(crate) mod table_write;
mod union;

pub use expand::{Expand, NullOf};
pub use expr::{FnPredicate, Predicate, ScalarExpr};
pub(crate) use file_scan::spark_dir_text;
pub use file_scan::{
    FileScan, FileScanMetrics, FileScanSource, METADATA_COLUMN, PathRoots, ROW_FILE_COLUMN,
    ROW_POSITION_COLUMN,
};
pub use file_write::{FileWrite, FileWriteSink};
pub use filter::Filter;
pub(crate) use filter::compact_sparse_views;
pub use generate::Generate;
pub use late_fetch::{FetchSlot, LateFetch};
pub use limit::Limit;
pub use project::{Project, ProjectItem};
pub(crate) use range::Range;
pub use replicate::ReplicateRows;
#[cfg(test)]
pub(crate) use runtime_filter::RowStage;
pub(crate) use runtime_filter::ScanRows;
pub use runtime_filter::{RowStageSettings, RuntimeFilters, SourceRows};
pub use sample::Sample;
pub(crate) use scan::scan_metadata;
pub use scan::{
    FileSelections, FilterPushdown, MemoryScan, PushedFilter, ScanPushdown, ScanPushdowns, TakeRows,
};
#[cfg(all(test, not(loom)))]
use scan::{ScanGlobal, ScanLocal};
pub use scan_decide::{
    BoundScan, BoundScanId, DecidedScan, ExecutionReads, OfferedConjunct, ScanDecision,
    SourceConjunct, SourceFilter,
};
pub use sort::{DEFAULT_MAX_MERGE_FAN_IN, Sort, SortKey, SortMetrics, TopN};
pub(crate) use sort::{KeyEncoder, blocking, data_bytes, owned_rows};
pub use spilled_scan::SpilledScan;
pub use union::UnionAlign;

use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState};

/// `state` as the operator's own local state.
pub(crate) fn local_mut<'a, T: 'static>(
    state: &'a mut LocalState,
    what: &str,
) -> Result<&'a mut T> {
    state
        .downcast_mut::<T>()
        .ok_or_else(|| internal(format!("wrong local state for {what}")))
}

/// `state` as the operator's own global state.
pub(crate) fn global_ref<'a, T: Send + Sync + 'static>(
    state: &'a GlobalState,
    what: &str,
) -> Result<&'a T> {
    <dyn std::any::Any + Send + Sync>::downcast_ref::<T>(&**state)
        .ok_or_else(|| internal(format!("wrong global state for {what}")))
}

/// Unit local state of the stateless operators.
pub(crate) fn no_state() -> LocalState {
    Box::new(())
}

/// Per-driver local state of an operator evaluating row expressions: the
/// driver's partition id (see [`TaskContext::partition`](crate::context::TaskContext::partition))
/// and the id its input's partitions are numbered from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PartitionState {
    /// The driver's partition id.
    pub(crate) partition: u32,
    /// [`TaskContext::numbering_base`](crate::context::TaskContext::numbering_base).
    pub(crate) base: u32,
}

impl PartitionState {
    /// The state of the operator `ctx` belongs to.
    pub(crate) fn of(ctx: &crate::context::TaskContext) -> Self {
        Self {
            partition: ctx.partition(),
            base: ctx.numbering_base(),
        }
    }

    /// Spark's partition index of a batch tagged `split` (the driver's
    /// partition when untagged): a stateful function's stream, as the
    /// operator hosting it is initialised (`UNION ALL` branches number
    /// their own partitions).
    pub(crate) fn index(&self, split: Option<u32>) -> u32 {
        split.unwrap_or(self.partition).wrapping_sub(self.base)
    }
}

#[cfg(all(test, not(loom)))]
mod tests;
