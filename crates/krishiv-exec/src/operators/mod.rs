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
mod limit;
mod project;
mod replicate;
mod runtime_filter;
mod scan;
mod sort;
mod union;

pub use expand::{Expand, NullOf};
pub use expr::{FnPredicate, Predicate, ScalarExpr};
pub use file_scan::{FileScan, FileScanMetrics, FileScanSource, METADATA_COLUMN};
pub use file_write::{FileWrite, FileWriteSink};
pub use filter::Filter;
pub use limit::Limit;
pub use project::{Project, ProjectItem};
pub use replicate::ReplicateRows;
pub(crate) use runtime_filter::RowStage;
pub use runtime_filter::{RowStageSettings, RuntimeFilters};
pub use scan::{
    FileSelections, FilterPushdown, MemoryScan, PushedFilter, ScanPushdown, ScanPushdowns,
};
#[cfg(all(test, not(loom)))]
use scan::{ScanGlobal, ScanLocal};
pub(crate) use sort::KeyEncoder;
pub use sort::{DEFAULT_MAX_MERGE_FAN_IN, Sort, SortKey, SortMetrics, TopN};
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
/// driver's partition index (see [`TaskContext::partition`](crate::context::TaskContext::partition)).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PartitionState {
    /// The driver's partition index.
    pub(crate) partition: u32,
}

#[cfg(all(test, not(loom)))]
mod tests;
