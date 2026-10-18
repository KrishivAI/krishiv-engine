// SPDX-License-Identifier: Apache-2.0
//! Test operators for executor tests and benchmarks (the real operators
//! arrive with P2-T051–T054): a generated range scan, filter, pass-through,
//! streaming limit, a count aggregate, a spillable buffering breaker that
//! stands in for sort / join build, and plan nodes to wire them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use std::collections::VecDeque;

use krishiv_memory::{Reservation, SpillSignal};
use krishiv_types::batch::Batch;

use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::morsel::MorselCursor;
use crate::operator::{
    GlobalState, LocalState, Operator, OperatorOutput, Sink, SinkStatus, Source,
};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::sync::{Mutex, lock};

/// The schema of every test batch: one non-null `Int64` column `v`.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]))
}

fn downcast<'a, T: 'static>(state: &'a mut LocalState, what: &str) -> Result<&'a mut T> {
    state
        .downcast_mut::<T>()
        .ok_or_else(|| internal(format!("wrong local state for {what}")))
}

fn downcast_global<T: Send + Sync + 'static>(state: &GlobalState, what: &str) -> Result<Arc<T>> {
    Arc::clone(state)
        .downcast::<T>()
        .map_err(|_| internal(format!("wrong global state for {what}")))
}

fn column(batch: &Batch) -> Result<&Int64Array> {
    batch
        .data()
        .columns()
        .first()
        .and_then(|c| c.as_primitive_opt::<Int64Type>())
        .ok_or_else(|| internal("test batches have one Int64 column".to_owned()))
}

/// Scans `0..rows` as `Int64` batches of `krishiv.exec.batchSize` rows, in
/// morsels of the configured size (seeded order in deterministic mode).
#[derive(Debug, Default)]
pub struct RangeSource {
    rows: u64,
    /// Rows produced so far, by every driver.
    pub produced: AtomicU64,
}

struct RangeGlobal {
    cursor: MorselCursor,
    morsel_rows: u64,
    batch_rows: u64,
}

impl RangeSource {
    /// Scans `0..rows`.
    pub fn new(rows: u64) -> Arc<Self> {
        Arc::new(Self {
            rows,
            produced: AtomicU64::new(0),
        })
    }
}

impl Source for RangeSource {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let config = ctx.query().config();
        let morsel_rows = config.morsel_rows.max(1) as u64;
        let morsels = usize::try_from(self.rows.div_ceil(morsel_rows))
            .map_err(|_| internal("too many morsels".to_owned()))?;
        Ok(Arc::new(RangeGlobal {
            cursor: MorselCursor::new(morsels, config.seed),
            morsel_rows,
            batch_rows: config.batch_size.max(1) as u64,
        }))
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global
            .downcast_ref::<RangeGlobal>()
            .map_or(1, |g| g.cursor.len())
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(0..0u64))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(std::future::ready(self.next_batch(local, global)))
    }
}

impl RangeSource {
    fn next_batch(&self, local: &mut LocalState, global: &GlobalState) -> Result<Option<Batch>> {
        let g = global
            .downcast_ref::<RangeGlobal>()
            .ok_or_else(|| internal("wrong global state for RangeSource".to_owned()))?;
        let range = downcast::<std::ops::Range<u64>>(local, "RangeSource")?;
        if range.is_empty() {
            let Some(m) = g.cursor.claim() else {
                return Ok(None);
            };
            let start = m as u64 * g.morsel_rows;
            *range = start..(start + g.morsel_rows).min(self.rows);
        }
        let end = (range.start + g.batch_rows).min(range.end);
        let values = Int64Array::from_iter_values(
            (range.start..end).map(|v| i64::try_from(v).unwrap_or(i64::MAX)),
        );
        range.start = end;
        let data = RecordBatch::try_new(schema(), vec![Arc::new(values)])
            .map_err(|e| internal(e.to_string()))?;
        self.produced
            .fetch_add(data.num_rows() as u64, Ordering::Relaxed);
        Ok(Some(Batch::new(data)))
    }
}

/// Emits one batch `count` times (shared buffers: no per-batch work), for
/// measuring the driver itself.
#[derive(Debug)]
pub struct Repeat {
    batch: Batch,
    /// Copies left; goes negative once exhausted.
    left: AtomicI64,
}

impl Repeat {
    /// `count` copies of `0..rows`.
    pub fn new(rows: usize, count: u64) -> Result<Arc<Self>> {
        let values = Int64Array::from_iter_values(0..i64::try_from(rows).unwrap_or(i64::MAX));
        let data = RecordBatch::try_new(schema(), vec![Arc::new(values)])
            .map_err(|e| internal(e.to_string()))?;
        Ok(Arc::new(Self {
            batch: Batch::new(data),
            left: AtomicI64::new(i64::try_from(count).unwrap_or(i64::MAX)),
        }))
    }
}

impl Source for Repeat {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(()))
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let more = self.left.fetch_sub(1, Ordering::Relaxed) > 0;
        Box::pin(std::future::ready(Ok(more.then(|| self.batch.clone()))))
    }
}

/// Keeps the rows whose `v` satisfies a predicate.
#[derive(Debug)]
pub struct Filter {
    keep: fn(i64) -> bool,
}

impl Filter {
    /// A filter on `v`.
    pub fn new(keep: fn(i64) -> bool) -> Arc<Self> {
        Arc::new(Self { keep })
    }
}

impl Operator for Filter {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        let mask: BooleanArray = column(input)?
            .values()
            .iter()
            .map(|v| Some((self.keep)(*v)))
            .collect();
        let out = input.filter(&mask)?;
        Ok(OperatorOutput::NeedMoreInput(
            (out.num_rows() > 0).then_some(out),
        ))
    }
}

/// Passes batches through unchanged (driver-overhead benchmark).
#[derive(Debug, Default)]
pub struct PassThrough;

impl Operator for PassThrough {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }
}

/// A streaming `LIMIT`: `Finished` once `limit` rows passed (over all
/// drivers).
#[derive(Debug)]
pub struct Limit {
    limit: u64,
    seen: AtomicU64,
}

impl Limit {
    /// At most `limit` rows.
    pub fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit,
            seen: AtomicU64::new(0),
        })
    }
}

impl Operator for Limit {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        let rows = input.num_rows() as u64;
        let before = self.seen.fetch_add(rows, Ordering::Relaxed);
        if before + rows < self.limit {
            return Ok(OperatorOutput::NeedMoreInput(Some(input.clone())));
        }
        let keep = usize::try_from(self.limit.saturating_sub(before)).unwrap_or(0);
        let out = (keep > 0).then(|| input.slice(0, keep)).transpose()?;
        Ok(OperatorOutput::Finished(out))
    }
}

/// A `COUNT(*), SUM(v)` breaker: its sink side keeps per-driver partials
/// and combines them once per driver into its global state; its source side
/// emits one batch `[count, sum]` (wrapping sum).
#[derive(Debug, Default)]
pub struct Count {
    /// `finalize` calls (exactly one per run).
    pub finalized: AtomicU64,
}

#[derive(Debug, Default)]
struct Totals {
    rows: AtomicU64,
    sum: AtomicI64,
}

impl Count {
    /// A new aggregate.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

/// A breaker's source-side global state: its sink's finalized state.
fn finalized<T: Send + Sync + 'static>(ctx: &TaskContext, what: &str) -> Result<GlobalState> {
    let state = ctx.dependency().cloned();
    let state = state.ok_or_else(|| internal(format!("{what} read before it was built")))?;
    downcast_global::<T>(&state, what)?;
    Ok(state)
}

impl Sink for Count {
    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(Totals::default()))
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new((0u64, 0i64)))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let partial = downcast::<(u64, i64)>(local, "Count")?;
        partial.0 += input.num_rows() as u64;
        partial.1 = column(&input)?
            .values()
            .iter()
            .fold(partial.1, |s, v| s.wrapping_add(*v));
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, mut local: LocalState, global: &GlobalState) -> Result<()> {
        let (rows, sum) = *downcast::<(u64, i64)>(&mut local, "Count")?;
        let totals = downcast_global::<Totals>(global, "Count")?;
        totals.rows.fetch_add(rows, Ordering::Relaxed);
        totals.sum.fetch_add(sum, Ordering::Relaxed);
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        self.finalized.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { Ok(()) })
    }
}

impl Source for Count {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        finalized::<Totals>(ctx, "Count")
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(false))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let out = downcast::<bool>(local, "Count").and_then(|done| {
            if std::mem::replace(done, true) {
                return Ok(None);
            }
            let totals = downcast_global::<Totals>(global, "Count")?;
            let rows = i64::try_from(totals.rows.load(Ordering::Relaxed)).unwrap_or(i64::MAX);
            let values = Int64Array::from(vec![rows, totals.sum.load(Ordering::Relaxed)]);
            let data = RecordBatch::try_new(schema(), vec![Arc::new(values)])
                .map_err(|e| internal(e.to_string()))?;
            Ok(Some(Batch::new(data)))
        });
        Box::pin(std::future::ready(out))
    }
}

/// A pipeline breaker standing in for sort / join build: its sink side
/// buffers every batch under a reservation (a spill moves the buffered
/// batches out of the accounted memory, as if to disk), its source side
/// replays them after `finalize`.
#[derive(Debug)]
pub struct Buffer {
    spillable: bool,
    finalize_work: Option<Duration>,
    finalize_yields: u32,
    /// Batches sunk so far.
    pub sunk: AtomicU64,
    /// Set once `finalize` has started.
    pub finalizing: AtomicBool,
    /// `sunk` at each spill.
    spilled_at: Mutex<Vec<u64>>,
    /// Every driver's spill signal.
    signals: Mutex<Vec<Arc<SpillSignal>>>,
}

#[derive(Debug, Default)]
struct Buffered {
    batches: Mutex<VecDeque<Batch>>,
    reservations: Mutex<Vec<Reservation>>,
}

struct BufferLocal {
    batches: Vec<Batch>,
    reservation: Reservation,
    signal: Arc<SpillSignal>,
}

impl Buffer {
    /// A buffer; `spillable` publishes its bytes for spilling,
    /// `finalize_work` is blocking work its `finalize` runs on the blocking
    /// pool.
    pub fn new(spillable: bool, finalize_work: Option<Duration>) -> Arc<Self> {
        Self::with_finalize_yields(spillable, finalize_work, 0)
    }

    /// As [`new`](Self::new), with a `finalize` that also yields
    /// `finalize_yields` times while its state stays spillable (a chunked
    /// merge).
    pub fn with_finalize_yields(
        spillable: bool,
        finalize_work: Option<Duration>,
        finalize_yields: u32,
    ) -> Arc<Self> {
        Arc::new(Self {
            spillable,
            finalize_work,
            finalize_yields,
            sunk: AtomicU64::new(0),
            finalizing: AtomicBool::new(false),
            spilled_at: Mutex::new(Vec::new()),
            signals: Mutex::new(Vec::new()),
        })
    }

    /// The value of `sunk` at each spill.
    pub fn spilled_at(&self) -> Vec<u64> {
        lock(&self.spilled_at).clone()
    }

    /// The spill signals of the drivers that have run the sink.
    pub fn signals(&self) -> Vec<Arc<SpillSignal>> {
        lock(&self.signals).clone()
    }
}

impl Sink for Buffer {
    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(Buffered::default()))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        lock(&self.signals).push(Arc::clone(memory.signal()));
        Ok(Box::new(BufferLocal {
            batches: Vec::new(),
            reservation: memory.reservation(),
            signal: Arc::clone(memory.signal()),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let l = downcast::<BufferLocal>(local, "Buffer")?;
        let bytes = input.data().get_array_memory_size();
        if l.reservation.try_grow(bytes).is_err() {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        l.batches.push(input);
        self.sunk.fetch_add(1, Ordering::Relaxed);
        if self.spillable {
            l.signal.set_spillable_bytes(l.reservation.size());
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = downcast_global::<Buffered>(global, "Buffer")?;
        let l = local
            .downcast::<BufferLocal>()
            .map_err(|_| internal("wrong local state for Buffer".to_owned()))?;
        let BufferLocal {
            batches,
            reservation,
            signal,
        } = *l;
        lock(&g.batches).extend(batches);
        let mut reservations = lock(&g.reservations);
        reservations.push(reservation);
        // The last driver's signal stays live while it finalizes: offer the
        // whole global state through `spill_global`.
        let total = reservations.iter().map(Reservation::size).sum();
        signal.set_spillable_bytes(if self.spillable { total } else { 0 });
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.finalizing.store(true, Ordering::Release);
            for _ in 0..self.finalize_yields {
                tokio::task::yield_now().await;
            }
            if let Some(d) = self.finalize_work {
                ctx.run_blocking(move || {
                    std::thread::sleep(d);
                    Ok(())
                })
                .await?;
            }
            Ok(())
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = downcast::<BufferLocal>(local, "Buffer")?;
            let freed = l.reservation.size();
            l.reservation.shrink(freed);
            l.signal.set_spillable_bytes(0);
            lock(&self.spilled_at).push(self.sunk.load(Ordering::Relaxed));
            Ok(freed)
        })
    }

    fn spill_global<'a>(
        &'a self,
        global: &'a GlobalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let g = downcast_global::<Buffered>(global, "Buffer")?;
            let mut freed = 0;
            for r in lock(&g.reservations).iter_mut() {
                freed += r.size();
                let size = r.size();
                r.shrink(size);
            }
            ctx.consumer().signal().set_spillable_bytes(0);
            lock(&self.spilled_at).push(self.sunk.load(Ordering::Relaxed));
            Ok(freed)
        })
    }
}

impl Source for Buffer {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        finalized::<Buffered>(ctx, "Buffer")
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let next =
            downcast_global::<Buffered>(global, "Buffer").map(|g| lock(&g.batches).pop_front());
        Box::pin(std::future::ready(next))
    }
}

/// A scan node.
pub struct Scan(pub Arc<dyn Source>);

impl ExecNode for Scan {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.set_source(Arc::clone(&self.0)).map(drop)
    }
}

/// A streaming operator above `child`.
pub struct Pipe(pub Arc<dyn ExecNode>, pub Arc<dyn Operator>);

impl ExecNode for Pipe {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        Arc::clone(&self.0).build_pipelines(b)?;
        b.push_operator(Arc::clone(&self.1));
        Ok(())
    }
}

/// A breaker that is both a sink and a source ([`Count`], [`Buffer`]).
pub trait Breaker: Sink + Source {}

impl<T: Sink + Source> Breaker for T {}

/// A breaker above `child`; `ordered` runs the pipeline above it with one
/// driver, keeping the breaker's output order.
pub struct Break<B>(pub Arc<dyn ExecNode>, pub Arc<B>, pub bool);

impl<B: Breaker + 'static> ExecNode for Break<B> {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(Arc::clone(&self.1) as Arc<dyn Sink>, Arc::clone(&self.0))?;
        b.set_breaker_source(Arc::clone(&self.1) as Arc<dyn Source>, sink)?;
        if self.2 {
            b.preserve_order();
        }
        Ok(())
    }
}

impl<B> std::fmt::Debug for Break<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Break")
    }
}

macro_rules! named_debug {
    ($($t:ident),*) => {$(
        impl std::fmt::Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(stringify!($t))
            }
        }
    )*};
}
named_debug!(Scan, Pipe);
