// SPDX-License-Identifier: Apache-2.0
//! The pipeline driver: the one place where yields, cancellation, memory
//! waits and spill requests happen (E01, ADR 0006/0007).
//!
//! A driver is an async task that owns one local state per operator. It
//! pulls a batch from the source, pushes it through the operators into the
//! sink, and yields; at every batch boundary it checks, in order, the
//! memory arbitrator's abort, the query's cancellation and its
//! [`SpillOwner`]'s spill requests. Every await is raced against
//! cancellation. When its source is exhausted it combines its sink state;
//! the last driver of the pipeline finalizes the sink.

use std::future::{Future, poll_fn};
use std::pin::{Pin, pin};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{MemoryError, SpillOwner, Spillable};
use krishiv_types::batch::Batch;

use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, TaskHandle};
use crate::metrics::{DriverStats, OperatorMetrics};
use crate::morsel::Countdown;
use crate::operator::{GlobalState, LocalState, OperatorOutput, SinkStatus};
use crate::pipeline::Pipeline;

/// An operator's `HaveMoreOutput` loop yields after this many iterations.
const YIELD_EVERY_OUTPUTS: u32 = 8;

/// One run of a pipeline, shared by its drivers.
pub(crate) struct PipelineRun {
    pub(crate) pipeline: Pipeline,
    pub(crate) source_global: GlobalState,
    pub(crate) sink_global: GlobalState,
    pub(crate) countdown: Countdown,
    pub(crate) query: Arc<QueryContext>,
    pub(crate) executor: Arc<dyn Executor>,
}

/// What woke a [`pause`].
enum Wake<T> {
    Done(T),
    Cancelled,
    Aborted(MemoryError),
    Spill,
}

/// Awaits `f` at a pause point. `f` is polled first, so a ready future
/// costs nothing; while it is pending the driver still answers its query's
/// cancellation, a memory abort of the query (failing it with the abort
/// error) and, given `spill`, spill requests for its own consumers.
/// Returns the output and how long `f` was pending.
async fn pause<F: Future>(
    run: &PipelineRun,
    spill: Option<(&SpillOwner, &tokio::sync::Mutex<Slots>)>,
    f: F,
) -> Result<(F::Output, Duration)> {
    let mut f = pin!(f);
    let first = poll_fn(|cx| Poll::Ready(f.as_mut().poll(cx))).await;
    if let Poll::Ready(out) = first {
        return Ok((out, Duration::ZERO));
    }
    let pending_since = run.executor.now();
    let query = &run.query;
    let mut cancel = pin!(query.cancelled());
    let mut abort = query.pool().aborted();
    loop {
        // Listen before checking, so a request in between is not missed.
        let mut requested = pin!(spill.map(|(owner, _)| owner.requested()));
        // Service once, then always wait on `f`, cancel and abort too: a
        // request the pass could not clear must not starve them.
        if let Some((owner, slots)) = spill
            && owner.pending()
        {
            service(run, &mut *slots.lock().await).await?;
        }
        let wake = poll_fn(|cx| {
            if let Poll::Ready(out) = f.as_mut().poll(cx) {
                return Poll::Ready(Wake::Done(out));
            }
            if cancel.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Wake::Cancelled);
            }
            if let Poll::Ready(e) = abort.as_mut().poll(cx) {
                return Poll::Ready(Wake::Aborted(e));
            }
            match requested.as_mut().as_pin_mut().map(|r| r.poll(cx)) {
                Some(Poll::Ready(())) => Poll::Ready(Wake::Spill),
                _ => Poll::Pending,
            }
        })
        .await;
        match wake {
            Wake::Done(out) => return Ok((out, since(&*run.executor, pending_since))),
            Wake::Cancelled => return Err(cancelled()),
            Wake::Aborted(e) => {
                let e = KrishivError::from(e);
                query.fail(e.clone());
                return Err(e);
            }
            Wake::Spill => {}
        }
    }
}

/// Time on `executor`'s clock since `start`.
fn since(executor: &dyn Executor, start: Instant) -> Duration {
    executor.now().saturating_duration_since(start)
}

/// Spawns one driver of `run`.
pub(crate) fn spawn_driver(run: Arc<PipelineRun>) -> TaskHandle {
    let query = Arc::clone(&run.query);
    let executor = Arc::clone(&run.executor);
    let q = Arc::clone(&query);
    let body = async move {
        if let Err(e) = drive(run).await {
            q.fail(e);
        }
    };
    let timed = Timed {
        inner: Box::pin(body),
        stats: DriverStats {
            drivers: 1,
            ..DriverStats::default()
        },
        query: Arc::clone(&query),
        executor: Arc::clone(&executor),
    };
    executor.spawn(query.class(), Box::pin(timed))
}

async fn drive(run: Arc<PipelineRun>) -> Result<()> {
    Driver::new(run)?.run().await
}

/// Times every poll of a driver: the stall detector. A poll is a
/// non-yielding slice of a worker thread.
struct Timed {
    inner: BoxFuture<'static, ()>,
    stats: DriverStats,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
}

impl Future for Timed {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let start = self.executor.now();
        let out = self.inner.as_mut().poll(cx);
        let slice = since(&*self.executor, start);
        let this = &mut *self;
        this.stats.max_slice = this.stats.max_slice.max(slice);
        let config = this.query.config();
        if slice > config.stall_threshold {
            this.stats.stalls += 1;
            if config.fail_on_stall {
                this.query.fail(internal(format!(
                    "a driver ran {slice:?} without yielding (stall threshold {:?}, batch size {})",
                    config.stall_threshold, config.batch_size
                )));
            }
        }
        out
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        let stats = std::mem::take(&mut self.stats);
        self.query.merge_metrics(|m| m.drivers.merge(&stats));
    }
}

/// An operator's or the sink's context and local state (`None` once the
/// sink state is combined).
struct Slot {
    ctx: Arc<TaskContext>,
    local: Option<LocalState>,
}

/// The driver's operator and sink state. It sits behind an async mutex only
/// so the spill service can borrow it from inside `SpillOwner::reserve`;
/// the mutex is never contended.
struct Slots {
    /// Every memory consumer the driver owns, one slot each: the source at
    /// 0 (its state stays in the driver, so `local` is `None`), operators
    /// `1..=n`, the sink at `n + 1`.
    slots: Vec<Slot>,
    /// Source at 0, operators `1..=n`, sink at `n + 1`.
    metrics: Vec<OperatorMetrics>,
}

impl Slots {
    fn slot(&mut self, k: usize) -> Result<&mut Slot> {
        self.slots
            .get_mut(k)
            .ok_or_else(|| internal(format!("no operator slot {k}")))
    }
}

/// What a driver waits for.
#[derive(Clone, Copy)]
enum Blocked {
    Sink,
    Memory,
}

struct Driver {
    run: Arc<PipelineRun>,
    owner: Arc<SpillOwner>,
    source_local: LocalState,
    slots: tokio::sync::Mutex<Slots>,
    /// Inputs in flight, by operator index (`n` = the sink).
    stack: Vec<(usize, Batch)>,
    /// The slot holding grant credit from the last `NeedMemory`.
    credit: Option<usize>,
    /// The wait in progress, timed even if the driver is aborted in it.
    waiting: Option<(usize, Blocked, Instant)>,
    batches: u64,
    yields: u64,
}

impl Driver {
    fn new(run: Arc<PipelineRun>) -> Result<Self> {
        let owner = SpillOwner::new();
        let ctx = |id, name| {
            TaskContext::new(
                Arc::clone(&run.query),
                Arc::clone(&run.executor),
                Arc::clone(&owner),
                id,
                name,
            )
        };
        let p = &run.pipeline;
        let (source_id, source) = &p.source;
        let source_ctx = Arc::new(ctx(*source_id, source.name()));
        let source_local = source.local_state(&source_ctx, &run.source_global)?;
        let mut slots = Vec::with_capacity(p.operators.len() + 2);
        slots.push(Slot {
            ctx: source_ctx,
            local: None,
        });
        for (id, op) in &p.operators {
            let c = Arc::new(ctx(*id, op.name()));
            let local = op.local_state(&c)?;
            slots.push(Slot {
                ctx: c,
                local: Some(local),
            });
        }
        let (sink_id, sink) = &p.sink;
        let c = Arc::new(ctx(*sink_id, sink.name()));
        let local = sink.local_state(&c, &run.sink_global)?;
        slots.push(Slot {
            ctx: c,
            local: Some(local),
        });
        let metrics = vec![OperatorMetrics::default(); p.operators.len() + 2];
        Ok(Self {
            source_local,
            slots: tokio::sync::Mutex::new(Slots { slots, metrics }),
            stack: Vec::new(),
            credit: None,
            waiting: None,
            batches: 0,
            yields: 0,
            owner,
            run,
        })
    }

    fn metrics(&mut self, k: usize) -> &mut OperatorMetrics {
        &mut self.slots.get_mut().metrics[k]
    }

    /// Adds the wait in progress, if any, to its slot's metrics.
    fn end_wait(&mut self) {
        if let Some((k, what, start)) = self.waiting.take() {
            let waited = since(&*self.run.executor, start);
            let m = self.metrics(k);
            match what {
                Blocked::Sink => m.blocked_on_sink += waited,
                Blocked::Memory => m.blocked_on_memory += waited,
            }
        }
    }

    async fn run(mut self) -> Result<()> {
        let run = Arc::clone(&self.run);
        let source = &run.pipeline.source.1;
        loop {
            self.boundary().await?;
            let start = run.executor.now();
            let (next, waited) = pause(
                &run,
                Some((&self.owner, &self.slots)),
                source.get_data(&mut self.source_local, &run.source_global),
            )
            .await?;
            let m = self.metrics(0);
            m.blocked_on_io += waited;
            m.compute_time += since(&*run.executor, start).saturating_sub(waited);
            let next = next?;
            let Some(batch) = next else { break };
            self.metrics(0).output(batch.num_rows());
            self.batches += 1;
            if self.push(batch).await? {
                break;
            }
            self.yield_now().await;
        }
        let sink = &run.pipeline.sink.1;
        let k = run.pipeline.operators.len() + 1;
        let local = self.slots.get_mut().slot(k)?.local.take();
        let local = local.ok_or_else(|| internal("sink state combined twice".to_owned()))?;
        sink.combine(local, &run.sink_global)?;
        if run.countdown.arrive() {
            self.boundary().await?;
            let start = run.executor.now();
            let ctx = Arc::clone(&self.slots.get_mut().slot(k)?.ctx);
            let finalize = sink.finalize(&run.sink_global, &ctx);
            pause(&run, Some((&self.owner, &self.slots)), finalize)
                .await?
                .0?;
            self.metrics(k).compute_time += since(&*run.executor, start);
        }
        Ok(())
    }

    async fn yield_now(&mut self) {
        self.yields += 1;
        tokio::task::yield_now().await;
    }

    /// The batch-boundary checks: abort, cancel, spill requests.
    async fn boundary(&mut self) -> Result<()> {
        let run = Arc::clone(&self.run);
        run.query.pool().check_aborted()?;
        run.query.check_cancelled()?;
        if self.owner.pending() {
            let serviced = service(&run, self.slots.get_mut());
            pause(&run, None, serviced).await?.0?;
        }
        Ok(())
    }

    /// Returns grant credit a retried call did not take.
    fn settle_credit(&mut self) {
        if let Some(k) = self.credit.take()
            && let Ok(slot) = self.slots.get_mut().slot(k)
            && let Some(m) = slot.ctx.registered()
        {
            m.consumer().release_credit();
        }
    }

    /// Pushes one source batch through the operators into the sink;
    /// `true` once the pipeline needs no more input.
    async fn push(&mut self, batch: Batch) -> Result<bool> {
        let run = Arc::clone(&self.run);
        let ops = &run.pipeline.operators;
        let n = ops.len();
        let mut finished = false;
        let mut outputs = 0u32;
        self.stack.push((0, batch));
        while let Some(top) = self.stack.last() {
            let i = top.0;
            if i == n {
                if let Some((_, b)) = self.stack.pop()
                    && self.sink(b).await?
                {
                    finished = true;
                    self.stack.clear();
                }
                continue;
            }
            let rows = top.1.num_rows();
            let out = {
                let slots = self.slots.get_mut();
                let local = slots.slots[i + 1].local.as_mut();
                let local = local.ok_or_else(|| internal("operator state missing".to_owned()))?;
                let start = run.executor.now();
                let out = ops[i].1.execute(&top.1, local);
                slots.metrics[i + 1].compute_time += since(&*run.executor, start);
                out?
            };
            if let OperatorOutput::NeedMemory { bytes } = out {
                self.reserve(i + 1, bytes).await?;
                continue;
            }
            self.settle_credit();
            match out {
                OperatorOutput::NeedMoreInput(b) => {
                    self.stack.pop();
                    self.metrics(i + 1).input(rows);
                    self.forward(i, b);
                }
                OperatorOutput::HaveMoreOutput(b) => {
                    self.forward(i, Some(b));
                    outputs += 1;
                    if outputs.is_multiple_of(YIELD_EVERY_OUTPUTS) {
                        self.boundary().await?;
                        self.yield_now().await;
                    }
                }
                OperatorOutput::Finished(b) => {
                    // Nothing upstream of `i` is needed any more.
                    self.stack.clear();
                    self.metrics(i + 1).input(rows);
                    self.forward(i, b);
                    finished = true;
                }
                OperatorOutput::NeedMemory { .. } => {}
            }
        }
        Ok(finished)
    }

    /// Queues operator `i`'s output for operator `i + 1`.
    fn forward(&mut self, i: usize, batch: Option<Batch>) {
        if let Some(b) = batch {
            self.metrics(i + 1).output(b.num_rows());
            self.stack.push((i + 1, b));
        }
    }

    /// Sinks one batch; `true` if the sink is finished.
    async fn sink(&mut self, batch: Batch) -> Result<bool> {
        let run = Arc::clone(&self.run);
        let sink = &run.pipeline.sink.1;
        let k = run.pipeline.operators.len() + 1;
        let rows = batch.num_rows();
        loop {
            let status = {
                let slots = self.slots.get_mut();
                let local = slots.slot(k)?.local.as_mut();
                let local = local.ok_or_else(|| internal("sink state missing".to_owned()))?;
                let start = run.executor.now();
                // A cheap clone (shared buffers): `NeedMemory` retries the same batch.
                let status = sink.sink(batch.clone(), local, &run.sink_global);
                slots.metrics[k].compute_time += since(&*run.executor, start);
                status?
            };
            if let SinkStatus::NeedMemory { bytes } = status {
                self.reserve(k, bytes).await?;
                continue;
            }
            self.settle_credit();
            self.metrics(k).input(rows);
            return Ok(match status {
                SinkStatus::Finished => true,
                SinkStatus::Blocked(space) => {
                    self.waiting = Some((k, Blocked::Sink, run.executor.now()));
                    let waited = pause(&run, Some((&self.owner, &self.slots)), space).await;
                    self.end_wait();
                    waited?;
                    false
                }
                SinkStatus::NeedMoreInput | SinkStatus::NeedMemory { .. } => false,
            });
        }
    }

    /// `NeedMemory` from slot `k`: reserve the bytes the cooperative way
    /// (self-spill, then arbitration while servicing this driver's own spill
    /// requests) and park them as the slot's grant credit.
    async fn reserve(&mut self, k: usize, bytes: usize) -> Result<()> {
        let run = Arc::clone(&self.run);
        let consumer = Arc::clone(self.slots.get_mut().slot(k)?.ctx.consumer().consumer());
        self.waiting = Some((k, Blocked::Memory, run.executor.now()));
        let slots = &self.slots;
        let run_ref = &*run;
        // Parks the grant as credit in the same step it is granted.
        let reserved = self.owner.reserve_for(&consumer, bytes, || async move {
            let mut guard = slots.lock().await;
            if let Err(e) = service(run_ref, &mut guard).await {
                run_ref.query.fail(e);
            }
        });
        let reserved = pause(run_ref, None, reserved).await;
        self.end_wait();
        let granted = reserved?.0;
        if granted.is_ok() {
            self.credit = Some(k);
        }
        run.query.check_cancelled()?;
        granted.map_err(KrishivError::from)?;
        Ok(())
    }
}

/// Spills every slot whose consumer has a pending request, on the owning
/// driver, and completes every request it finds: one pass over every
/// consumer the driver owns, so no request is left pending (a source has no
/// spill hook and frees 0).
async fn service(run: &PipelineRun, slots: &mut Slots) -> Result<()> {
    let n = run.pipeline.operators.len();
    for k in 0..=n + 1 {
        let slot = slots.slot(k)?;
        let Some(m) = slot.ctx.registered() else {
            continue;
        };
        let Some(target) = m.signal().spill_requested() else {
            continue;
        };
        let op = k.checked_sub(1).and_then(|i| run.pipeline.operators.get(i));
        let freed = match (k, slot.local.as_mut(), op) {
            // Sources have no spill hook.
            (0, _, _) => 0,
            (_, Some(local), Some((_, op))) => op.spill(local, target, &slot.ctx).await?,
            (_, Some(local), None) => run.pipeline.sink.1.spill(local, target, &slot.ctx).await?,
            // Combined (finalize runs): the sink's state is global now.
            (_, None, _) => {
                let sink = &run.pipeline.sink.1;
                sink.spill_global(&run.sink_global, target, &slot.ctx)
                    .await?
            }
        };
        m.signal().complete(freed);
        let metrics = &mut slots.metrics[k];
        metrics.spills += 1;
        metrics.spilled_bytes += freed as u64;
    }
    Ok(())
}

impl Drop for Driver {
    /// Merges this driver's counters into the query, once, on every path
    /// (success, error, cancel, abort).
    fn drop(&mut self) {
        self.end_wait();
        let p = &self.run.pipeline;
        let ids: Vec<_> = std::iter::once(p.source.0)
            .chain(p.operators.iter().map(|(id, _)| *id))
            .chain(std::iter::once(p.sink.0))
            .collect();
        let slots = self.slots.get_mut();
        for (k, slot) in slots.slots.iter().enumerate() {
            if let Some(m) = slot.ctx.registered() {
                slots.metrics[k].peak_reserved = m.consumer().peak() as u64;
            }
        }
        let metrics = std::mem::take(&mut slots.metrics);
        let stats = DriverStats {
            batches: self.batches,
            yields: self.yields,
            ..DriverStats::default()
        };
        self.run.query.merge_metrics(|m| {
            for (id, om) in ids.into_iter().zip(&metrics) {
                m.operators.entry(id).or_default().metrics.merge(om);
            }
            m.drivers.merge(&stats);
        });
    }
}
