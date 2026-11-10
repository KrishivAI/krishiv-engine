// SPDX-License-Identifier: Apache-2.0
//! Python UDF evaluation (design P2-T059 §2.3-§2.4).
//!
//! **Runtime seam.** A plan names a Python UDF by `(name, version)`
//! ([`FunctionId::PythonUdf`](krishiv_plan::FunctionId::PythonUdf)); the
//! worker resolves it through the [`UdfRuntime`] its
//! [`WorkerContext`](crate::physical::WorkerContext)
//! holds (`WorkerContext::with_python_udfs`). The embedded runtime lives
//! in `krishiv-python` (layer 6), so nothing below it links Python. A
//! worker without a runtime, or a runtime without that version, refuses
//! the plan with `KRISHIV_UDF_NOT_AVAILABLE`.
//!
//! **Evaluation is a pipeline break** ([`PythonEvalSink`] plus a
//! `SharedSource` over its list). A UDF batch
//! can take tens of milliseconds, far over the ≈1 ms bound of
//! [`Sink::sink`], so compute workers never run Python and never wait for
//! it: `sink` slices its input to at most `maxRecordsPerBatch` rows,
//! submits each slice to the runtime (which runs it on threads of its own)
//! and returns at once; while the bounded window of in-flight slices is
//! full it returns [`SinkStatus::Blocked`], whose future the driver awaits
//! between calls (servicing spill requests meanwhile). Finished slices, the
//! input's columns plus one column per call, are kept as the shared
//! plan's list is (`crate::shared`: memory under the driver's consumer,
//! spilled under pressure); the downstream pipeline reads them after
//! `finalize`, which awaits the slices still in flight. Every in-flight
//! slice reserves its input bytes, so a full window is bounded memory (I11).
//!
//! The downstream pipeline does not overlap the UDF calls: running the two
//! pipelines at once needs the local-exchange wiring that P2-T174 unit 3
//! deferred to P2-T176. The sink and the source keep that shape, so the
//! overlap comes with it without changing this module's protocol.
//!
//! **Order.** Slices are kept in submission order per driver; an ordered
//! input (below a sort) runs one driver, so the list keeps its order. Each
//! result is attached to its own input slice, so alignment never depends
//! on completion order.
//!
//! **Errors and cancellation.** The first failed call fails the query.
//! Dropping the sink's state (query end, cancel, error) drops every pending
//! call's future and the reservation of its input: nothing stays charged.
//! `finalize` waits for every slice still in flight at once, so the first
//! failure, or a cancel, ends it without waiting for the slices before it.
//! What the runtime does with a call whose future was dropped is its own
//! (the embedded runtime skips a queued call and interrupts one already
//! inside Python, design P2-T059 §2.5 "Cancel").

use std::collections::VecDeque;
use std::fmt;
use std::future::poll_fn;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use futures::task::noop_waker_ref;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillScope};
use krishiv_plan::PythonEvalType;
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::operators::{ScalarExpr, global_ref, local_mut};
use crate::shared::SharedSink;
use crate::sync::{Mutex, lock};

/// What a runtime prepares one call site of a Python UDF from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdfCallSpec {
    /// The UDF's canonical (lower-case) name.
    pub name: String,
    /// The registration version the plan was analyzed with.
    pub version: u32,
    /// How it is converted and called.
    pub eval_type: PythonEvalType,
    /// Whether calls may overlap.
    pub thread_safe: bool,
    /// The Spark types of the arguments, in order.
    pub arg_types: Vec<DataType>,
    /// The declared Spark return type.
    pub return_type: DataType,
    /// The Arrow type a call's result must have: the return type's client
    /// form (decimals on `Decimal128`, strings and binaries as views).
    pub return_arrow: ArrowType,
}

/// Resolves the Python UDFs a plan names on the worker that runs it.
pub trait UdfRuntime: Send + Sync + fmt::Debug {
    /// The callable of `spec`.
    ///
    /// # Errors
    /// `KRISHIV_UDF_NOT_AVAILABLE` when this runtime has no such
    /// `(name, version)`; any other error fails the query.
    fn prepare(&self, spec: &UdfCallSpec) -> Result<Arc<dyn PreparedUdf>>;
}

/// One prepared call site.
pub trait PreparedUdf: Send + Sync {
    /// Runs the UDF over `args`: one column per argument (their client
    /// forms), `args.num_rows()` rows (a call without arguments has no
    /// columns). The work runs off the caller (on the runtime's own
    /// threads): polling the future only collects the result, so it may be
    /// polled without a waker that ever fires. The result must have
    /// [`UdfCallSpec::return_arrow`]'s type and one value per row.
    fn call(&self, args: RecordBatch) -> BoxFuture<'static, Result<ArrayRef>>;
}

/// One call site of a [`PythonEvalSink`]: its argument expressions over the
/// input, the prepared UDF, and the types of its result.
pub struct EvalCall {
    /// The arguments, evaluated over each input slice.
    pub args: Vec<Arc<dyn ScalarExpr>>,
    /// The prepared UDF.
    pub udf: Arc<dyn PreparedUdf>,
    /// The type the runtime returns ([`UdfCallSpec::return_arrow`]).
    pub client: ArrowType,
    /// The type of the output column (the canonical carrier).
    pub canonical: ArrowType,
}

impl fmt::Debug for EvalCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalCall")
            .field("args", &self.args)
            .field("client", &self.client)
            .field("canonical", &self.canonical)
            .finish_non_exhaustive()
    }
}

/// In-flight slices per driver when nothing else is configured: enough to
/// keep the runtime busy while the driver fetches the next input.
pub const DEFAULT_WINDOW: usize = 2;

/// The sink side of a `PythonEval` node.
pub struct PythonEvalSink {
    calls: Vec<EvalCall>,
    schema: SchemaRef,
    /// Most rows per call; 0 is no limit.
    max_rows: usize,
    window: usize,
    store: Arc<SharedSink>,
}

impl fmt::Debug for PythonEvalSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonEvalSink")
            .field("calls", &self.calls)
            .field("max_rows", &self.max_rows)
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

/// The output schema of a `PythonEval` over `input`: its columns, then one
/// nullable column per call.
#[must_use]
pub fn output_schema(input: &Schema, calls: &[(String, ArrowType)]) -> SchemaRef {
    let mut fields: Vec<Field> = input.fields().iter().map(|f| (**f).clone()).collect();
    fields.extend(
        calls
            .iter()
            .map(|(name, ty)| Field::new(name, ty.clone(), true)),
    );
    Arc::new(Schema::new_with_metadata(fields, input.metadata().clone()))
}

impl PythonEvalSink {
    /// A sink over input of `input` evaluating `calls`, slicing to at most
    /// `max_rows` rows per call (0: no limit), with `window` slices in
    /// flight per driver; finished rows spill to `scope`.
    #[must_use]
    pub fn new(
        input: &Schema,
        calls: Vec<EvalCall>,
        max_rows: usize,
        window: usize,
        scope: Option<SpillScope>,
    ) -> Arc<Self> {
        let named: Vec<(String, ArrowType)> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| (format!("pythonUDF{i}"), c.canonical.clone()))
            .collect();
        let schema = output_schema(input, &named);
        Arc::new(Self {
            calls,
            store: SharedSink::new(Arc::clone(&schema), scope),
            schema,
            max_rows,
            window: window.max(1),
        })
    }

    /// The output schema.
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// The result of one call over one slice.
enum Slot {
    Pending(BoxFuture<'static, Result<ArrayRef>>),
    Done(Result<ArrayRef>),
}

/// One slice in flight: the input rows, one slot per call, and the
/// reservation of the input's bytes (released when the slice is stored or
/// dropped).
struct InFlight {
    input: Batch,
    slots: Vec<Slot>,
    _reservation: Reservation,
}

impl InFlight {
    /// Polls every pending slot; whether all are done.
    fn poll_all(&mut self, cx: &mut Context<'_>) -> bool {
        let mut done = true;
        for slot in &mut self.slots {
            if let Slot::Pending(f) = slot {
                match f.as_mut().poll(cx) {
                    Poll::Ready(r) => *slot = Slot::Done(r),
                    Poll::Pending => done = false,
                }
            }
        }
        done
    }

    /// The error of a call of this slice that failed, if any.
    fn failed(&self) -> Option<&KrishivError> {
        self.slots.iter().find_map(|s| match s {
            Slot::Done(Err(e)) => Some(e),
            _ => None,
        })
    }

    /// The output rows: the input's columns, then each call's result in
    /// the canonical carrier. Every slot must be done.
    fn finish(self, calls: &[EvalCall], schema: &SchemaRef) -> Result<Batch> {
        let rows = self.input.num_rows();
        let (data, weights) = self.input.into_parts();
        let mut columns: Vec<ArrayRef> = data.columns().to_vec();
        for (slot, call) in self.slots.into_iter().zip(calls) {
            let Slot::Done(result) = slot else {
                return Err(internal(
                    "a Python UDF slice was stored unfinished".to_owned(),
                ));
            };
            columns.push(normalize(&result?, call, rows)?);
        }
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        let out = RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
            .map_err(|e| internal(format!("a Python UDF result does not fit its batch: {e}")))?;
        Batch::with_weights(out, weights)
    }
}

/// `result` in `call`'s canonical carrier, checked against the declared
/// type and the slice's row count. The runtime checks both before it hands
/// a result over; this is the engine's own guard.
fn normalize(result: &ArrayRef, call: &EvalCall, rows: usize) -> Result<ArrayRef> {
    if result.len() != rows {
        return Err(internal(format!(
            "a Python UDF returned {} rows for {rows}",
            result.len()
        )));
    }
    if result.data_type() != &call.client {
        return Err(internal(format!(
            "a Python UDF returned {} where {} was declared",
            result.data_type(),
            call.client
        )));
    }
    crate::carrier::recarried_array(result, &call.canonical)
}

/// The pending slices of one driver, shared with the futures its blocked
/// calls return.
#[derive(Default)]
struct Pending {
    slices: VecDeque<InFlight>,
}

impl Pending {
    /// Polls the oldest slice; whether it is done (or there is none).
    fn head_ready(&mut self, cx: &mut Context<'_>) -> bool {
        self.slices.front_mut().is_none_or(|h| h.poll_all(cx))
    }

    /// The first failure among the slices in flight, polling each of them
    /// (so `cx` is woken by any one): a failure behind a slice still
    /// running must not wait for it (design P2-T059 §6.3).
    fn failure(&mut self, cx: &mut Context<'_>) -> Option<KrishivError> {
        self.slices.iter_mut().find_map(|s| {
            s.poll_all(cx);
            s.failed().cloned()
        })
    }

    /// The oldest slice if every call of it is done, without waiting.
    fn take_ready(&mut self) -> Option<InFlight> {
        let mut cx = Context::from_waker(noop_waker_ref());
        if self.slices.front_mut()?.poll_all(&mut cx) {
            self.slices.pop_front()
        } else {
            None
        }
    }
}

/// One driver's state.
struct EvalLocal {
    /// The finished rows, kept by the shared plan's sink.
    store: LocalState,
    pending: Arc<Mutex<Pending>>,
    /// A finished slice the store could not take yet (it asked for memory).
    ready: Option<Batch>,
    partition: u32,
    /// Input rows not submitted yet: fewer than `maxRecordsPerBatch`, kept
    /// until more arrive or the input ends (Spark cuts a partition's rows
    /// into batches of exactly that many, whatever its input batches were;
    /// `research/spark-notes/python-udf-batching.md`).
    held: Vec<Batch>,
    held_rows: usize,
    /// The reservation of the held rows' bytes.
    held_bytes: Reservation,
}

/// The global state: the store's, and the slices drivers left in flight.
struct EvalGlobal {
    store: GlobalState,
    leftover: Mutex<Vec<(Option<Batch>, VecDeque<InFlight>)>>,
}

impl PythonEvalSink {
    /// The arguments of every call over `slice`.
    fn arguments(&self, slice: &RecordBatch, partition: u32) -> Result<Vec<RecordBatch>> {
        let rows = slice.num_rows();
        self.calls
            .iter()
            .map(|call| {
                let mut fields = Vec::with_capacity(call.args.len());
                let mut columns = Vec::with_capacity(call.args.len());
                for (i, arg) in call.args.iter().enumerate() {
                    let a = arg.evaluate_in(slice, partition)?;
                    let client = krishiv_types::arrow::client_arrow_type(a.data_type());
                    let a = crate::carrier::recarried_array(&a, &client)?;
                    fields.push(Field::new(format!("_{i}"), client, arg.nullable()));
                    columns.push(a);
                }
                let options = RecordBatchOptions::new().with_row_count(Some(rows));
                RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
                    .map_err(|e| internal(format!("Python UDF arguments: {e}")))
            })
            .collect()
    }

    /// Submits the held rows in slices of `max_rows` (all of them as one
    /// slice when `max_rows` is 0, Krishiv's "no limit": one call per input
    /// batch); with `flush` the rows left over too, as the driver's last,
    /// shorter slice (Spark's partition's last batch).
    fn submit_held(&self, l: &mut EvalLocal, flush: bool) -> Result<()> {
        if l.held_rows == 0 {
            return Ok(());
        }
        let step = if self.max_rows == 0 {
            l.held_rows
        } else {
            self.max_rows
        };
        if l.held_rows < step && !flush {
            return Ok(());
        }
        let held = std::mem::take(&mut l.held);
        let rows = std::mem::take(&mut l.held_rows);
        let schema = held
            .first()
            .map(Batch::schema)
            .ok_or_else(|| internal("no held rows".to_owned()))?;
        let all = if held.len() == 1 {
            held.into_iter()
                .next()
                .ok_or_else(|| internal("no held rows".to_owned()))?
        } else {
            Batch::concat(&schema, &held)?
        };
        let mut offset = 0;
        let mut slices = Vec::new();
        while offset < rows {
            let len = step.min(rows - offset);
            if len < step && !flush {
                // Too few for a full slice: held for the next input.
                l.held.push(all.slice(offset, len)?);
                l.held_rows = len;
                break;
            }
            let slice = all.slice(offset, len)?;
            let args = self.arguments(slice.data(), l.partition)?;
            // Each slice owns its share of the held rows' reservation; the
            // rows still held keep the rest.
            let share = l.held_bytes.size() * len / (rows - offset);
            let part = l.held_bytes.split(share);
            let slots = self
                .calls
                .iter()
                .zip(args)
                .map(|(call, args)| Slot::Pending(call.udf.call(args)))
                .collect();
            slices.push(InFlight {
                input: slice,
                slots,
                _reservation: part,
            });
            offset += len;
        }
        lock(&l.pending).slices.extend(slices);
        Ok(())
    }

    /// [`Self::submit_held`] without the rows left over.
    fn submit_full(&self, l: &mut EvalLocal) -> Result<()> {
        self.submit_held(l, false)
    }

    /// Stores `batch` in the list, or keeps it for the retry after the
    /// memory the store asked for.
    fn store(&self, l: &mut EvalLocal, batch: Batch, g: &EvalGlobal) -> Result<Option<SinkStatus>> {
        match self.store.sink(batch.clone(), &mut l.store, &g.store)? {
            SinkStatus::NeedMemory { bytes } => {
                l.ready = Some(batch);
                Ok(Some(SinkStatus::NeedMemory { bytes }))
            }
            _ => Ok(None),
        }
    }
}

impl Sink for PythonEvalSink {
    fn name(&self) -> &'static str {
        "PythonEval"
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(EvalGlobal {
            store: self.store.global_state(ctx)?,
            leftover: Mutex::new(Vec::new()),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let g = global_ref::<EvalGlobal>(global, "PythonEval")?;
        Ok(Box::new(EvalLocal {
            store: self.store.local_state(ctx, &g.store)?,
            pending: Arc::new(Mutex::new(Pending::default())),
            ready: None,
            partition: ctx.partition(),
            held: Vec::new(),
            held_rows: 0,
            held_bytes: ctx.consumer().consumer().reservation(),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        let g = global_ref::<EvalGlobal>(global, "PythonEval")?;
        let l = local_mut::<EvalLocal>(local, "PythonEval")?;
        // Store what finished first: a `NeedMemory` here leaves the input
        // unsubmitted, so the retry with the same batch is exact.
        if let Some(ready) = l.ready.take()
            && let Some(status) = self.store(l, ready, g)?
        {
            return Ok(status);
        }
        if let Some(e) = lock(&l.pending).failure(&mut Context::from_waker(noop_waker_ref())) {
            return Err(e);
        }
        loop {
            let next = lock(&l.pending).take_ready();
            let Some(done) = next else { break };
            let batch = done.finish(&self.calls, &self.schema)?;
            if let Some(status) = self.store(l, batch, g)? {
                return Ok(status);
            }
        }
        let rows = input.num_rows();
        if rows > 0 {
            let input = crate::operators::owned_rows(input)?;
            let bytes = input.data().get_array_memory_size();
            if l.held_bytes.try_grow(bytes).is_err() {
                return Ok(SinkStatus::NeedMemory { bytes });
            }
            l.held.push(input);
            l.held_rows += rows;
            self.submit_full(l)?;
        }
        if lock(&l.pending).slices.len() >= self.window {
            let pending = Arc::clone(&l.pending);
            return Ok(SinkStatus::Blocked(Box::pin(poll_fn(move |cx| {
                let mut p = lock(&pending);
                // Free once the oldest slice is done, or any slice failed:
                // the next `sink` call then returns the failure.
                if p.head_ready(cx) || p.failure(cx).is_some() {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }))));
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<EvalGlobal>(global, "PythonEval")?;
        let l = local
            .downcast::<EvalLocal>()
            .map_err(|_| internal("wrong local state for PythonEval".to_owned()))?;
        let mut l = *l;
        // The input ended: the rows held go as the driver's last slice.
        self.submit_held(&mut l, true)?;
        let EvalLocal {
            store,
            pending,
            ready,
            ..
        } = l;
        let slices = std::mem::take(&mut lock(&pending).slices);
        if ready.is_some() || !slices.is_empty() {
            lock(&g.leftover).push((ready, slices));
        }
        self.store.combine(store, &g.store)
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<EvalGlobal>(global, "PythonEval")?;
            let mut leftover = std::mem::take(&mut *lock(&g.leftover));
            // Every slice still in flight at once, not in order: the first
            // failed call (or a cancel) ends the wait, and the error drops
            // the other calls' futures, which interrupts a call still
            // running (design P2-T059 §2.5 "Cancel"). In order, a failure
            // behind a long call would wait for it. Each wake polls every
            // slice left: at most the window times the drivers, so this stays
            // small while the window does (review 2026-11-09 M1).
            let mut cancelled = std::pin::pin!(ctx.query().cancelled());
            poll_fn(|cx| {
                if cancelled.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(Err(ctx.query().cancellation()));
                }
                let mut all = true;
                for slice in leftover.iter_mut().flat_map(|(_, s)| s.iter_mut()) {
                    all &= slice.poll_all(cx);
                    if let Some(e) = slice.failed() {
                        return Poll::Ready(Err(e.clone()));
                    }
                }
                if all {
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Pending
                }
            })
            .await?;
            for (ready, slices) in leftover {
                let mut batches: Vec<Batch> = ready.into_iter().collect();
                for mut slice in slices {
                    ctx.query().check_cancelled()?;
                    poll_fn(|cx| {
                        if slice.poll_all(cx) {
                            Poll::Ready(())
                        } else {
                            Poll::Pending
                        }
                    })
                    .await;
                    batches.push(slice.finish(&self.calls, &self.schema)?);
                }
                let batches = batches
                    .into_iter()
                    .map(crate::operators::owned_rows)
                    .collect::<Result<Vec<_>>>()?;
                let bytes = batches
                    .iter()
                    .map(|b| b.data().get_array_memory_size())
                    .sum();
                let mut reservation = ctx.consumer().reservation();
                reservation.grow(bytes).await.map_err(KrishivError::from)?;
                self.store.add_part(&g.store, batches, reservation)?;
            }
            self.store.finalize(&g.store, ctx).await
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = local_mut::<EvalLocal>(local, "PythonEval")?;
            self.store.spill(&mut l.store, target, ctx).await
        })
    }
}

/// The global state a `PythonEval`'s reader takes its list from: the
/// store's own.
///
/// # Errors
/// `INTERNAL_ERROR` for a state of another sink.
pub(crate) fn store_state(global: &GlobalState) -> Result<GlobalState> {
    Ok(Arc::clone(
        &global_ref::<EvalGlobal>(global, "PythonEval")?.store,
    ))
}

#[cfg(test)]
mod tests;
