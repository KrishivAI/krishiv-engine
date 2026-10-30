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
use crate::morsel::{Barrier, Countdown};
use crate::operator::{FlushOutput, GlobalState, LocalState, OperatorOutput, SinkStatus};
use crate::pipeline::Pipeline;

/// An operator's `HaveMoreOutput` loop yields after this many iterations.
const YIELD_EVERY_OUTPUTS: u32 = 8;

/// One run of a pipeline, shared by its drivers.
pub(crate) struct PipelineRun {
    pub(crate) pipeline: Pipeline,
    /// Per operator: the finalized breaker state it reads, if any.
    pub(crate) operator_deps: Vec<Option<GlobalState>>,
    /// Per operator: its flush barrier (used if it needs a flush), shared
    /// with the other pipelines of a fan-in that run the operator.
    pub(crate) barriers: Vec<Arc<Barrier>>,
    pub(crate) source_global: GlobalState,
    pub(crate) sink_global: GlobalState,
    /// Shared by the pipelines of a fan-in, as is `sink_global`.
    pub(crate) countdown: Arc<Countdown>,
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

/// Spawns one driver of `run`: `partition` is the driver's
/// scheduler-assigned globally unique partition id within its pipeline
/// group (see [`TaskContext::with_partition`]).
pub(crate) fn spawn_driver(run: Arc<PipelineRun>, partition: u32) -> TaskHandle {
    let pipeline = run.pipeline.index;
    let query = Arc::clone(&run.query);
    let executor = Arc::clone(&run.executor);
    let q = Arc::clone(&query);
    let body = async move {
        if let Err(e) = drive(run, partition).await {
            q.fail(e);
        }
    };
    let timed = Timed {
        inner: Box::pin(body),
        stats: DriverStats {
            drivers: 1,
            ..DriverStats::default()
        },
        pipeline,
        started: None,
        threads: Vec::new(),
        query: Arc::clone(&query),
        executor: Arc::clone(&executor),
    };
    let name = crate::console::driver_task_name(query.id(), pipeline, partition);
    crate::console::spawn_named(&executor, &query, query.class(), &name, Box::pin(timed))
}

async fn drive(run: Arc<PipelineRun>, partition: u32) -> Result<()> {
    Driver::new(run, partition)?.run().await
}

/// Times every poll of a driver: the stall detector. A poll is a
/// non-yielding slice of a worker thread.
///
/// In fail-on-stall mode (tests and CI) a slice is measured without the
/// time its thread waited on a run queue (runnable, but another thread had
/// the core): that time is the machine's load, not the driver's work, and
/// counting it failed queries on loaded machines. Running and blocking
/// (a sleep or blocking I/O inside a poll) both still count. Where the
/// kernel does not report run-queue waits, and when stalls are only
/// counted, the slice is wall time.
struct Timed {
    inner: BoxFuture<'static, ()>,
    stats: DriverStats,
    /// The pipeline the driver runs.
    pipeline: u32,
    /// When it was first polled (an offset from the query's start).
    started: Option<Duration>,
    /// The threads that polled it, each once.
    threads: Vec<u64>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
}

impl Future for Timed {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let queued = if self.query.config().fail_on_stall {
            runqueue_wait()
        } else {
            None
        };
        let start = self.executor.now();
        let this = &mut *self;
        if this.started.is_none() {
            this.started = Some(this.query.offset(start));
        }
        let thread = crate::metrics::thread_ordinal();
        if !this.threads.contains(&thread) {
            this.threads.push(thread);
        }
        let out = this.inner.as_mut().poll(cx);
        let slice = since(&*this.executor, start);
        this.stats.max_slice = this.stats.max_slice.max(slice);
        let config = this.query.config();
        // Only a slice longer than the threshold can be a stall, so the
        // second sample is taken for those alone.
        let held = if slice > config.stall_threshold {
            match queued.zip(runqueue_wait()) {
                Some((before, after)) => slice.saturating_sub(after.saturating_sub(before)),
                None => slice,
            }
        } else {
            slice
        };
        if held > config.stall_threshold {
            this.stats.stalls += 1;
            if config.fail_on_stall {
                this.query.fail(internal(format!(
                    "a driver ran {held:?} without yielding (stall threshold {:?}, batch size {})",
                    config.stall_threshold, config.batch_size
                )));
            }
        }
        out
    }
}

/// The time this thread has waited on a run queue so far, from
/// `/proc/thread-self/schedstat` (its second field, in nanoseconds); `None`
/// where the kernel does not report it.
///
/// The file is opened once per thread and read with `pread`: a poll costs
/// one syscall instead of open, statx, two reads and close.
#[cfg(unix)]
#[expect(
    clippy::disallowed_methods,
    reason = "reading a pseudo-file served from memory with one pread per poll never blocks: the only deliberate sync read on a worker"
)]
fn runqueue_wait() -> Option<Duration> {
    use std::os::unix::fs::FileExt;
    thread_local! {
        /// Opened by the thread it describes (`thread-self` is resolved at
        /// open), `None` where the file is missing.
        static SCHEDSTAT: Option<std::fs::File> =
            std::fs::File::open("/proc/thread-self/schedstat").ok();
    }
    SCHEDSTAT.with(|file| {
        let mut buf = [0u8; 128];
        let n = file.as_ref()?.read_at(&mut buf, 0).ok()?;
        let nanos = std::str::from_utf8(&buf[..n])
            .ok()?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()?;
        Some(Duration::from_nanos(nanos))
    })
}

/// Non-Unix targets have no `/proc/thread-self/schedstat`: there is no
/// run-queue wait to subtract, so slices are wall time.
#[cfg(not(unix))]
fn runqueue_wait() -> Option<Duration> {
    None
}

impl Drop for Timed {
    fn drop(&mut self) {
        let stats = std::mem::take(&mut self.stats);
        self.query.merge_metrics(|m| m.drivers.merge(&stats));
        // A driver never polled (cancelled before it started) ran nowhere.
        if let Some(started) = self.started {
            let ended = self.query.offset(self.executor.now());
            self.query
                .record_driver(self.pipeline, started, ended, &self.threads);
        }
    }
}

/// An operator's or the sink's context and local state (`None` once the
/// sink state is combined, once an operator's output is no longer wanted,
/// and while an operator's `end_input` or `flush` step runs, `busy`).
struct Slot {
    ctx: Arc<TaskContext>,
    local: Option<LocalState>,
    /// The state is out of the slot for a step in progress: spill requests
    /// for it wait until the step ends.
    busy: bool,
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
    /// The source position last reported to the sink.
    position: Option<u64>,
}

impl Driver {
    fn new(run: Arc<PipelineRun>, partition: u32) -> Result<Self> {
        let owner = SpillOwner::new();
        let ctx = |id, name| {
            TaskContext::new(
                Arc::clone(&run.query),
                Arc::clone(&run.executor),
                Arc::clone(&owner),
                id,
                name,
            )
            .with_partition(partition)
        };
        let p = &run.pipeline;
        let (source_id, source) = &p.source;
        let source_ctx = Arc::new(ctx(*source_id, source.name()));
        let source_local = source.local_state(&source_ctx, &run.source_global)?;
        let mut slots = Vec::with_capacity(p.operators.len() + 2);
        slots.push(Slot {
            ctx: source_ctx,
            local: None,
            busy: false,
        });
        for (k, (id, op)) in p.operators.iter().enumerate() {
            let mut c = ctx(*id, op.name());
            if let Some(Some(state)) = run.operator_deps.get(k) {
                c = c.with_dependency(Arc::clone(state));
            }
            let c = Arc::new(c);
            let local = op.local_state(&c)?;
            slots.push(Slot {
                ctx: c,
                local: Some(local),
                busy: false,
            });
        }
        let (sink_id, sink) = &p.sink;
        let c = Arc::new(ctx(*sink_id, sink.name()));
        let local = sink.local_state(&c, &run.sink_global)?;
        slots.push(Slot {
            ctx: c,
            local: Some(local),
            busy: false,
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
            position: None,
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
        // The first operator whose buffered output is still wanted.
        let mut finish_from = 0;
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
            self.report_position()?;
            self.metrics(0).output(batch.num_rows());
            self.batches += 1;
            if let Some(from) = self.push(0, batch).await? {
                finish_from = from;
                break;
            }
            self.yield_now().await;
        }
        self.finish_operators(finish_from).await?;
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

    /// Tells the sink when the source moved to another morsel (see
    /// [`Sink::position`](crate::operator::Sink::position)).
    fn report_position(&mut self) -> Result<()> {
        let run = Arc::clone(&self.run);
        let position = run.pipeline.source.1.position(&self.source_local);
        if position == self.position {
            return Ok(());
        }
        self.position = position;
        if let Some(p) = position {
            let k = run.pipeline.operators.len() + 1;
            let slots = self.slots.get_mut();
            let local = slots.slot(k)?.local.as_mut();
            let local = local.ok_or_else(|| internal("sink state missing".to_owned()))?;
            run.pipeline.sink.1.position(local, &run.sink_global, p);
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
        // Under memory pressure, return this thread's freed memory to the
        // OS (the host's release hook; see `krishiv_memory::set_release_hook`).
        krishiv_memory::release_if_pressured_here();
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

    /// Drains the buffered output ([`Operator::finish`]) of operators
    /// `from..` in pipeline order through the operators after each into the
    /// sink, each followed by its flush phase if it needs one. An operator
    /// upstream of one that returned `Finished` is not asked; one downstream
    /// still is.
    ///
    /// [`Operator::finish`]: crate::Operator::finish
    async fn finish_operators(&mut self, mut from: usize) -> Result<()> {
        let run = Arc::clone(&self.run);
        self.release_unwanted(from);
        for (i, (_, op)) in run.pipeline.operators.iter().enumerate() {
            while i >= from {
                self.boundary().await?;
                let out = {
                    let slots = self.slots.get_mut();
                    let local = slots.slots[i + 1].local.as_mut();
                    let local =
                        local.ok_or_else(|| internal("operator state missing".to_owned()))?;
                    let start = run.executor.now();
                    let out = op.finish(local);
                    slots.metrics[i + 1].compute_time += since(&*run.executor, start);
                    out?
                };
                let Some(b) = out else { break };
                self.metrics(i + 1).output(b.num_rows());
                if let Some(next) = self.push(i + 1, b).await? {
                    from = next;
                }
                self.yield_now().await;
            }
            if let Some(barrier) = run.barriers.get(i).filter(|_| op.needs_flush()) {
                from = self.flush(i, barrier, from).await?;
            }
            self.release_unwanted(from);
        }
        Ok(())
    }

    /// Drops the state of operators `..from`, whose output is no longer
    /// wanted: their memory goes back at once, and work they share with the
    /// other drivers (a hash join's claimed grace partition) is given up
    /// before this driver waits at a later operator's flush barrier, where
    /// a driver waiting for that work would otherwise wait forever.
    fn release_unwanted(&mut self, from: usize) {
        let slots = self.slots.get_mut();
        let n = slots.slots.len().saturating_sub(2);
        for slot in slots.slots.iter_mut().skip(1).take(from.min(n)) {
            if slot.local.take().is_some()
                && let Some(m) = slot.ctx.registered()
            {
                m.signal().set_spillable_bytes(0);
            }
        }
    }

    /// Takes operator `i`'s state out of its slot for a step that may wait
    /// (`end_input`, `flush`), so the driver can service spill requests for
    /// its other consumers meanwhile.
    fn take_local(&mut self, i: usize) -> Result<(Arc<TaskContext>, LocalState)> {
        let slot = self.slots.get_mut().slot(i + 1)?;
        let local = slot.local.take();
        let local = local.ok_or_else(|| internal("operator state missing".to_owned()))?;
        slot.busy = true;
        Ok((Arc::clone(&slot.ctx), local))
    }

    fn put_local(&mut self, i: usize, local: LocalState) -> Result<()> {
        let slot = self.slots.get_mut().slot(i + 1)?;
        slot.local = Some(local);
        slot.busy = false;
        Ok(())
    }

    /// The flush phase of operator `i` (see
    /// [`Operator::needs_flush`](crate::Operator::needs_flush)); every
    /// driver arrives at its barrier exactly once, also when operator `i`'s
    /// output is no longer wanted (`i < from`). Returns the new `from`.
    async fn flush(&mut self, i: usize, barrier: &Barrier, mut from: usize) -> Result<usize> {
        let run = Arc::clone(&self.run);
        let Some((_, op)) = run.pipeline.operators.get(i) else {
            return Ok(from);
        };
        if i < from {
            barrier.arrive();
            return Ok(from);
        }
        let (ctx, mut local) = self.take_local(i)?;
        let spill = Some((&*self.owner, &self.slots));
        let ended = pause(&run, spill, op.end_input(&mut local, &ctx)).await;
        self.put_local(i, local)?;
        ended?.0?;
        barrier.arrive();
        pause(&run, Some((&self.owner, &self.slots)), barrier.wait()).await?;
        while i >= from {
            self.boundary().await?;
            let start = run.executor.now();
            let (ctx, mut local) = self.take_local(i)?;
            let spill = Some((&*self.owner, &self.slots));
            let out = pause(&run, spill, op.flush(&mut local, &ctx)).await;
            self.put_local(i, local)?;
            let out = out?.0;
            self.metrics(i + 1).compute_time += since(&*run.executor, start);
            match out? {
                FlushOutput::Done => break,
                FlushOutput::Continue => {}
                FlushOutput::NeedMemory { bytes } => {
                    self.reserve_or_fall_back(i + 1, bytes).await?;
                }
                FlushOutput::Batch(b) => {
                    self.metrics(i + 1).output(b.num_rows());
                    if let Some(next) = self.push(i + 1, b).await? {
                        from = next;
                    }
                }
            }
            self.yield_now().await;
        }
        Ok(from)
    }

    /// Pushes a batch into operator `start` (the sink if `start` is the
    /// operator count) and on through the pipeline. Once the pipeline needs
    /// no more input, returns the first operator whose buffered output is
    /// still wanted: the one after the last that returned `Finished`, or the
    /// operator count when the sink finished.
    async fn push(&mut self, start: usize, batch: Batch) -> Result<Option<usize>> {
        let run = Arc::clone(&self.run);
        let ops = &run.pipeline.operators;
        let n = ops.len();
        let mut finished: Option<usize> = None;
        let mut outputs = 0u32;
        self.stack.push((start, batch));
        while let Some(top) = self.stack.last() {
            let i = top.0;
            if i == n {
                if let Some((_, b)) = self.stack.pop()
                    && self.sink(b).await?
                {
                    finished = Some(n);
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
                    finished = Some(finished.map_or(i + 1, |f| f.max(i + 1)));
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
        let granted = self.reserve_with(k, bytes, true).await?;
        if granted.is_ok() {
            self.credit = Some(k);
        }
        granted.map_err(KrishivError::from)
    }

    /// `FlushOutput::NeedMemory` from slot `k`: as [`reserve`](Self::reserve)
    /// but never aborting a query, and the credit is the operator's to
    /// manage. When spills cannot make room the step finds no new credit
    /// and takes its fallback.
    async fn reserve_or_fall_back(&mut self, k: usize, bytes: usize) -> Result<()> {
        match self.reserve_with(k, bytes, false).await? {
            Ok(()) | Err(MemoryError::Exhausted { .. }) => Ok(()),
            Err(e) => Err(KrishivError::from(e)),
        }
    }

    /// Reserves `bytes` for slot `k`'s consumer, parking the grant as its
    /// credit in the same step; `may_abort` lets the arbitrator abort a
    /// query. The outer error is the driver's own (cancelled, failed).
    async fn reserve_with(
        &mut self,
        k: usize,
        bytes: usize,
        may_abort: bool,
    ) -> Result<Result<(), MemoryError>> {
        let run = Arc::clone(&self.run);
        let consumer = Arc::clone(self.slots.get_mut().slot(k)?.ctx.consumer().consumer());
        self.waiting = Some((k, Blocked::Memory, run.executor.now()));
        let slots = &self.slots;
        let run_ref = &*run;
        let service = || async move {
            let mut guard = slots.lock().await;
            if let Err(e) = service(run_ref, &mut guard).await {
                run_ref.query.fail(e);
            }
        };
        let owner = &self.owner;
        let reserved = async move {
            if may_abort {
                owner.reserve_for(&consumer, bytes, service).await
            } else {
                owner
                    .reserve_for_without_abort(&consumer, bytes, service)
                    .await
            }
        };
        let reserved = pause(run_ref, None, reserved).await;
        self.end_wait();
        let granted = reserved?.0;
        run.query.check_cancelled()?;
        Ok(granted)
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
        if m.signal().spill_requested().is_none() {
            continue;
        }
        let op = k.checked_sub(1).and_then(|i| run.pipeline.operators.get(i));
        // A step of this operator is in progress: it is serviced at the
        // next batch boundary, once the step ended (not begun until then,
        // so the arbitrator's wait for it stays bounded).
        if slot.local.is_none() && slot.busy && op.is_some() {
            continue;
        }
        // From here the request is being serviced: the arbitrator awaits
        // it without a deadline, and an error completes it with 0 bytes.
        let Some(service) = m.signal().begin() else {
            continue;
        };
        let target = service.target();
        let freed = match (k, slot.local.as_mut(), op) {
            // Sources have no spill hook; an operator whose state is gone
            // (no longer wanted) has freed its memory with it.
            (0, _, _) | (_, None, Some(_)) => 0,
            (_, Some(local), Some((_, op))) => op.spill(local, target, &slot.ctx).await?,
            (_, Some(local), None) => run.pipeline.sink.1.spill(local, target, &slot.ctx).await?,
            // Combined (finalize runs): the sink's state is global now.
            (_, None, _) => {
                let sink = &run.pipeline.sink.1;
                sink.spill_global(&run.sink_global, target, &slot.ctx)
                    .await?
            }
        };
        service.complete(freed);
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
        slots.metrics[0].morsels += p.source.1.morsels_claimed(&self.source_local);
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
            for (id, mut om) in ids.into_iter().zip(metrics) {
                om.as_driver();
                m.operators.entry(id).or_default().metrics.merge(&om);
            }
            m.drivers.merge(&stats);
        });
    }
}
