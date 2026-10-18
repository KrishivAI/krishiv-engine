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
//! `spill` on its own task when the arbitrator asks; only the owning driver
//! ever touches operator state (ADR 0007). `spill` runs between calls, not
//! only between inputs: at the boundary every 8 `HaveMoreOutput` calls for
//! one input, and while the driver waits for its source or a blocked sink.
//! State kept across `HaveMoreOutput` calls (a probe iterator) must survive
//! a spill. During `finalize` requests go to [`Sink::spill_global`].
//!
//! **Memory in global state.** Reserve in local state (through
//! `ctx.consumer()`), not in `global_state`: a consumer registered there is
//! never asked to spill. State moved into the global state by `combine`
//! stays accounted and is spillable during `finalize` via `spill_global`.

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
    /// Shared state, made once per pipeline run. A breaker's source side
    /// takes its sink side's finalized state from
    /// [`TaskContext::dependency`].
    ///
    /// # Errors
    /// Any error fails the query.
    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState>;
    /// How many drivers can usefully run at once (for example the morsel
    /// count). The pipeline runs `min(worker slots, this)` drivers. No
    /// default, so a scan cannot run serially by omission.
    fn max_parallelism(&self, global: &GlobalState) -> usize;
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
    /// Morsels this driver claimed so far. Sources own the morsel
    /// boundaries, so they count them; the driver reads this once, when it
    /// ends, into [`OperatorMetrics::morsels`](crate::OperatorMetrics::morsels).
    /// Default: not counted.
    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        let _ = local;
        0
    }
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

/// What [`Operator::flush`] produced.
#[derive(Debug)]
pub enum FlushOutput {
    /// One output batch; call `flush` again.
    Batch(Batch),
    /// No output from this step (it read or built state); call `flush` again.
    Continue,
    /// Nothing left to flush.
    Done,
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
    /// Called once no more input will arrive (the driver's source is
    /// exhausted, or an operator upstream returned `Finished`; not when one
    /// downstream did), repeatedly until it returns `None`: one buffered
    /// output batch per call, pushed on through the operators after it
    /// (DuckDB's `FinalExecute`; a partial aggregate flushes its table).
    /// Sync and bounded like `execute`; it cannot ask for memory, so
    /// buffered output must already be reserved. Default: nothing buffered.
    ///
    /// # Errors
    /// Any error fails the query.
    fn finish(&self, local: &mut LocalState) -> Result<Option<Batch>> {
        let _ = local;
        Ok(None)
    }
    /// Operator-specific `EXPLAIN ANALYZE` counters (name, value), read
    /// once when the query ends. Default: none.
    fn extra_metrics(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
    /// Whether the operator has work left that needs **every** driver's
    /// input first (a hash join's spilled partitions). If so, after
    /// [`finish`](Self::finish) a driver calls [`end_input`](Self::end_input),
    /// waits until every driver of the pipeline has (a barrier), then calls
    /// [`flush`](Self::flush) until `Done`, pushing each batch downstream.
    /// Drivers flush in parallel, so the operator shares out the work
    /// itself. Default: `false`.
    fn needs_flush(&self) -> bool {
        false
    }
    /// This driver's input ended: publish what [`flush`](Self::flush) needs
    /// from the local state (before the barrier; may await I/O and
    /// [`TaskContext::run_blocking`]).
    ///
    /// # Errors
    /// Any error fails the query.
    fn end_input<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        let _ = (local, ctx);
        Box::pin(async { Ok(()) })
    }
    /// One bounded step of the work left after every driver's input ended.
    /// May await I/O and [`TaskContext::run_blocking`]; the driver checks
    /// cancellation, aborts and spill requests between steps.
    ///
    /// # Errors
    /// Any error fails the query.
    fn flush<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<FlushOutput>> {
        let _ = (local, ctx);
        Box::pin(async { Ok(FlushOutput::Done) })
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
    /// Frees at least `target` bytes of the **global** state if it can,
    /// while `finalize` runs (the last driver's consumer was asked to spill
    /// after its local state was combined). Runs on the finalizing driver
    /// between the polls of `finalize`, so `finalize` must leave the global
    /// state consistent at every await. Default: not spillable.
    ///
    /// # Errors
    /// Any error fails the query.
    fn spill_global<'a>(
        &'a self,
        global: &'a GlobalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        let _ = (global, target, ctx);
        Box::pin(async { Ok(0) })
    }
    /// As [`Operator::extra_metrics`].
    fn extra_metrics(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
}
