// SPDX-License-Identifier: Apache-2.0
//! Batch execution engine (layer L4): push-based pipelines over morsels on
//! the Tokio executor (ADR 0006, 0007; spec 13 §2, §4).
//!
//! # Model
//! - **Operators** ([`Source`] → [`Operator`]* → [`Sink`]) are DuckDB-style:
//!   per-driver local state, per-pipeline global state, `combine` per
//!   driver and `finalize` once. `execute`/`sink` are sync and bounded
//!   (≈ ≤ 1 ms); sources and `finalize` are async.
//! - **Pipelines** are split at breakers by the [`PipelineBuilder`] from an
//!   [`ExecNode`] plan; a pipeline starts once the pipelines it depends on
//!   (join build, aggregate, sort) have finalized.
//! - **Drivers**: each pipeline runs `min(worker slots, morsels)` async
//!   drivers on the [`Executor`]; they claim morsels from the source's
//!   shared cursor ([`MorselCursor`]), so load balances by pulling. A driver
//!   yields after every source batch and every 8 `HaveMoreOutput` loops;
//!   longer non-yielding work goes through
//!   [`TaskContext::run_blocking`]. A stall
//!   detector times every driver poll.
//! - **Batch boundaries** are the only pause points: the driver checks the
//!   memory arbitrator's abort, the query's cancellation and its spill
//!   requests there, and on `NeedMemory` reserves outside the operator, then
//!   retries the same input with the bytes parked as grant credit.
//! - **Cancellation and errors**: every await races the query's
//!   cancellation; the first error wins and cancels (and aborts) the rest; a
//!   panic fails only its own query; dropping the result stream cancels.
//!   Memory is released by RAII on every path.
//!
//! ```
//! use std::sync::Arc;
//!
//! use futures::StreamExt;
//! use krishiv_exec::testing::{Filter, Pipe, RangeSource, Scan};
//! use krishiv_exec::{
//!     ExecConfig, ExecutorConfig, QueryContext, QueryId, QueryRunner, Scheduler, TokioExecutor,
//!     WorkloadClass,
//! };
//! use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
//!
//! let executor = TokioExecutor::new(&ExecutorConfig::default())?;
//! let runner = QueryRunner::new(executor);
//! let root = TreePool::new(&MemoryConfig::default());
//! let query = QueryContext::new(
//!     QueryId(1),
//!     WorkloadClass::Interactive,
//!     ExecConfig::default(),
//!     root.child("q1", None),
//! );
//! let scan = Arc::new(Scan(RangeSource::new(10_000)));
//! let plan = Arc::new(Pipe(scan, Filter::new(|v| v % 2 == 0)));
//! let caller = tokio::runtime::Builder::new_current_thread().build().map_err(|e| e.to_string())?;
//! let rows = caller.block_on(async {
//!     let mut stream = runner.submit(plan, Arc::clone(&query)).await?;
//!     let mut rows = 0;
//!     while let Some(batch) = stream.next().await {
//!         rows += batch?.num_rows();
//!     }
//!     query.finished().await;
//!     Ok::<_, krishiv_common::error::KrishivError>(rows)
//! })?;
//! assert_eq!(rows, 5_000);
//! assert_eq!(root.reserved(), 0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]

pub mod aggregate;
mod cancel;
mod channel;
mod context;
mod driver;
mod error;
mod executor;
pub mod files;
pub mod hash;
mod join;
mod metrics;
mod morsel;
mod operator;
mod operators;
mod pipeline;
pub mod planner;
mod runner;
mod sync;
pub mod testing;

pub use cancel::CancellationToken;
pub use context::{
    DEFAULT_BATCH_SIZE, DEFAULT_MORSEL_ROWS, ExecConfig, MemoryHandle, OperatorId, QueryContext,
    QueryId, TaskContext,
};
pub use error::Result;
pub use executor::{
    Executor, ExecutorConfig, TaskHandle, TokioClock, TokioExecutor, WorkloadClass,
};
pub use join::{
    HashJoin, HashJoinNode, JoinSpec, JoinType, NestedLoopJoin, NestedLoopJoinNode, NestedLoopSpec,
};
pub use metrics::{DriverStats, OperatorEntry, OperatorMetrics, QueryMetrics};
pub use morsel::MorselCursor;
pub use operator::{
    FlushOutput, GlobalState, LocalState, Operator, OperatorOutput, Sink, SinkStatus, Source,
};
pub use operators::{
    DEFAULT_MAX_MERGE_FAN_IN, FileScan, FileScanMetrics, FileScanSource, FileSelections, FileWrite,
    FileWriteSink, Filter, FilterPushdown, FnPredicate, Limit, METADATA_COLUMN, MemoryScan,
    Predicate, Project, ProjectItem, PushedFilter, ReplicateRows, ScalarExpr, ScanPushdown,
    ScanPushdowns, Sort, SortKey, SortMetrics, TopN, UnionAlign,
};
pub use pipeline::{ExecNode, PipelineBuilder};
pub use runner::{QueryRunner, Scheduler};

#[cfg(all(test, not(loom)))]
mod tests;

#[cfg(all(test, loom))]
mod loom_tests;
