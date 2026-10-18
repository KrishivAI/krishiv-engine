// SPDX-License-Identifier: Apache-2.0
//! `Source` / `Operator` / `Sink` (spec 13 §2, ADR 0006): DuckDB-style push
//! pipelines with per-driver local state and shared global state. A
//! pipeline breaker is the `Sink` of one pipeline and the `Source` of the
//! next.
//!
//! **Sync and bounded.** [`Operator::execute`] and [`Sink::sink`] do about a
//! millisecond of work or less and never await; the driver yields between
//! calls. On a failed `try_grow` they return `NeedMemory` with their state
//! unchanged, and the driver calls them again with the **same input** once
//! it has the memory (parked as grant credit, see
//! [`MemoryHandle::reservation`](crate::MemoryHandle::reservation)).
//! A batch is never half-applied.
//!
//! **Spill.** Operators that hold memory publish spillable bytes on their
//! [`MemoryHandle::signal`](crate::MemoryHandle::signal). The driver calls
//! `spill` on its own task at a batch boundary when the arbitrator asks;
//! only the owning driver ever touches operator state (ADR 0007).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_types::batch::Batch;

use crate::context::TaskContext;
use crate::error::Result;

/// Per-driver state of an operator, owned by one driver.
pub type LocalState = Box<dyn Any + Send>;
/// Per-pipeline state shared by all drivers.
pub type GlobalState = Arc<dyn Any + Send + Sync>;

/// Produces the batches of a pipeline. Drivers claim morsels from the
/// global state (for example through a [`MorselCursor`](crate::MorselCursor)).
pub trait Source: Send + Sync {
    /// A name for metrics and memory consumers.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    /// Output schema.
    fn schema(&self) -> SchemaRef;
    /// Shared state, made once per pipeline run.
    ///
    /// # Errors
    /// Any error fails the query.
    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState>;
    /// How many drivers can usefully run at once (for example the morsel
    /// count). The pipeline runs `min(worker slots, this)` drivers.
    fn max_parallelism(&self, global: &GlobalState) -> usize {
        let _ = global;
        1
    }
    /// One driver's state.
    ///
    /// # Errors
    /// Any error fails the query.
    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState>;
    /// The next batch, or `None` when no morsel is left. Awaits I/O on the
    /// I/O runtime; never blocks a worker thread (E01/E08).
    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>>;
}

/// What [`Operator::execute`] did with its input.
#[derive(Debug)]
pub enum OperatorOutput {
    /// Done with this input; maybe one output batch.
    NeedMoreInput(Option<Batch>),
    /// One output batch; call `execute` again with the same input.
    HaveMoreOutput(Batch),
    /// `try_grow` failed and the state is unchanged: the driver reserves
    /// `bytes` and calls `execute` again with the same input.
    NeedMemory {
        /// Bytes the operator could not reserve.
        bytes: usize,
    },
    /// The operator needs no more input (a streaming `LIMIT`): the driver
    /// pushes the batch, if any, downstream and stops feeding this
    /// operator and everything upstream of it.
    Finished(Option<Batch>),
}

/// A streaming operator in the middle of a pipeline.
pub trait Operator: Send + Sync {
    /// A name for metrics and memory consumers.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    /// Output schema.
    fn schema(&self) -> SchemaRef;
    /// One driver's state.
    ///
    /// # Errors
    /// Any error fails the query.
    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState>;
    /// CPU-bound and bounded (≈ ≤ 1 ms per call). Reserves memory with
    /// `try_grow` before allocating; never awaits.
    ///
    /// # Errors
    /// Any error fails the query.
    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput>;
    /// Frees at least `target` bytes if it can, on the owning driver at a
    /// batch boundary; returns the bytes freed. Default: not spillable.
    ///
    /// # Errors
    /// Any error fails the query.
    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        let _ = (local, target, ctx);
        Box::pin(async { Ok(0) })
    }
}

/// What [`Sink::sink`] did with its input.
pub enum SinkStatus {
    /// Input consumed; send more.
    NeedMoreInput,
    /// Input consumed; the sink needs no more (a `LIMIT` reached): the
    /// driver stops pulling its source.
    Finished,
    /// `try_grow` failed and the state is unchanged: the driver reserves
    /// `bytes` and passes the same batch again.
    NeedMemory {
        /// Bytes the sink could not reserve.
        bytes: usize,
    },
    /// Input consumed, but the sink is full (the bounded result channel):
    /// the driver awaits this future before pushing the next batch.
    Blocked(BoxFuture<'static, ()>),
}

impl fmt::Debug for SinkStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeedMoreInput => f.write_str("NeedMoreInput"),
            Self::Finished => f.write_str("Finished"),
            Self::NeedMemory { bytes } => {
                f.debug_struct("NeedMemory").field("bytes", bytes).finish()
            }
            Self::Blocked(_) => f.write_str("Blocked"),
        }
    }
}

/// The end of a pipeline: a pipeline breaker (join build, aggregate, sort),
/// or the query's result.
pub trait Sink: Send + Sync {
    /// A name for metrics and memory consumers.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    /// Shared state, made once per pipeline run.
    ///
    /// # Errors
    /// Any error fails the query.
    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState>;
    /// One driver's state.
    ///
    /// # Errors
    /// Any error fails the query.
    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState>;
    /// Consumes one batch. Sync and bounded, like [`Operator::execute`].
    ///
    /// # Errors
    /// Any error fails the query.
    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus>;
    /// Merges one driver's state into the global state, once per driver
    /// after its source is exhausted.
    ///
    /// # Errors
    /// Any error fails the query.
    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()>;
    /// Runs once per pipeline run, on the last driver to combine. May exceed
    /// 1 ms (hash-table build, sort-run merge): chunk the work with yields
    /// or run it through [`TaskContext::run_blocking`], checking
    /// cancellation and the spill flag between chunks.
    ///
    /// # Errors
    /// Any error fails the query.
    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>>;
    /// As [`Operator::spill`]. Default: not spillable.
    ///
    /// # Errors
    /// Any error fails the query.
    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        let _ = (local, target, ctx);
        Box::pin(async { Ok(0) })
    }
}
