// SPDX-License-Identifier: Apache-2.0
//! The `ORDER BY` breaker: buffer and sort per batch, spill sorted runs
//! under memory pressure, merge them in `finalize` down to a bounded fan-in,
//! and stream the final k-way merge from its source side.

use std::future::Future;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{ConsumerHandle, MemoryError, Reservation, SpillScope, SpillSignal};
use krishiv_types::batch::Batch;

use super::keys::{KeyEncoder, SortKey};
use super::merge::{Merger, SortedBatch, SpillRun, data_bytes, interleave_batches, write_run};
use super::{BATCH_SLACK_BYTES, SPILL_BATCH_BYTES, SPILL_BUFFER_BYTES, blocking, finalized};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::operators::{global_ref, local_mut};
use crate::output_spill::{OutputSpill, SpillUnread, Unfinished, refused};
use crate::sync::{Event, Mutex, lock};

/// Default bound on the runs one merge reads at once.
pub const DEFAULT_MAX_MERGE_FAN_IN: usize = 64;

/// Counters of one sort for `EXPLAIN ANALYZE`, next to the driver's
/// [`OperatorMetrics`](crate::OperatorMetrics) of its sink and source sides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SortMetrics {
    /// Rows sunk (and sorted).
    pub rows_sorted: u64,
    /// Sorted runs written to spill files by spills.
    pub runs_spilled: u64,
    /// Bytes written to spill files, merge passes included.
    pub spilled_bytes: u64,
    /// Merge passes before the final merge: levels of intermediate merges,
    /// each rewriting up to the fan-in's runs into one.
    pub merge_passes: u64,
    /// Runs read at once by the final merge.
    pub final_fan_in: u64,
}

#[derive(Debug, Default)]
struct Counters {
    rows: AtomicU64,
    runs: AtomicU64,
    bytes: AtomicU64,
    passes: AtomicU64,
    fan_in: AtomicU64,
}

/// `ORDER BY` as a pipeline breaker: the sink side of one pipeline and the
/// source side of the next (one driver, so the order holds).
///
/// Spark's order: NULLS FIRST for ASC and LAST for DESC by default, every
/// NaN equal and above all values, `-0.0 = 0.0`, strings by bytes. Stable
/// within a driver (ties keep input order); weights travel with their rows.
/// Spills sorted runs under memory pressure and merges them with a bounded
/// fan-in (see the `operators::sort` module source for the details).
#[derive(Debug)]
pub struct Sort {
    schema: SchemaRef,
    encoder: Arc<KeyEncoder>,
    spill: Option<SpillScope>,
    max_fan_in: usize,
    counters: Arc<Counters>,
}

impl Sort {
    /// Sorts batches of `schema` by `keys`. With a `spill` scope the sort
    /// spills sorted runs under memory pressure; without one it can only
    /// hold its input in memory.
    ///
    /// # Errors
    /// As [`SortKey`] validation: `KRISHIV_UNSUPPORTED_FEATURE` for a key
    /// type the sort does not order yet, `INTERNAL_ERROR` for no keys or an
    /// out-of-range column.
    pub fn new(schema: SchemaRef, keys: &[SortKey], spill: Option<SpillScope>) -> Result<Self> {
        Ok(Self {
            encoder: KeyEncoder::new(&schema, keys)?,
            schema,
            spill,
            max_fan_in: DEFAULT_MAX_MERGE_FAN_IN,
            counters: Arc::default(),
        })
    }

    /// Bounds the runs one merge reads at once (at least 2); more runs are
    /// merged in several passes.
    #[must_use]
    pub fn with_max_fan_in(mut self, max_fan_in: usize) -> Self {
        self.max_fan_in = max_fan_in.max(2);
        self
    }

    /// The counters so far.
    pub fn metrics(&self) -> SortMetrics {
        let c = &self.counters;
        SortMetrics {
            rows_sorted: c.rows.load(Ordering::Relaxed),
            runs_spilled: c.runs.load(Ordering::Relaxed),
            spilled_bytes: c.bytes.load(Ordering::Relaxed),
            merge_passes: c.passes.load(Ordering::Relaxed),
            final_fan_in: c.fan_in.load(Ordering::Relaxed),
        }
    }
}

/// Buffered, per-batch-sorted input with its reservations.
#[derive(Debug)]
struct Buffer {
    batches: Vec<Arc<SortedBatch>>,
    /// Accounts `batches`.
    reservation: Reservation,
    /// Memory kept free for spilling this buffer (the merged output batch
    /// and the writer's scratch), lent to them as grant credit.
    headroom: Reservation,
    /// The most bytes per row seen, for the headroom.
    row_bytes: usize,
    handle: Arc<ConsumerHandle>,
}

impl Buffer {
    fn new(handle: &Arc<ConsumerHandle>) -> Self {
        Self {
            batches: Vec::new(),
            reservation: handle.reservation(),
            headroom: handle.reservation(),
            row_bytes: 0,
            handle: Arc::clone(handle),
        }
    }
}

/// Lends `res` to the next growths of its consumer (as grant credit) while
/// `work` runs, so the scratch memory of a spill or merge is never taken by
/// another consumer; then takes the credit back and tops `res` up to
/// `target` if the pool allows (best effort: a later reservation asks for
/// the rest).
async fn lending<T>(
    res: &mut Reservation,
    handle: &ConsumerHandle,
    target: usize,
    work: impl Future<Output = T>,
) -> T {
    let lent = res.size();
    res.shrink_to_credit(lent);
    let out = work.await;
    // Credit only: this never reaches the pool, so it cannot fail.
    let _ = res.try_grow(handle.credit());
    let _ = res.try_grow(target.saturating_sub(res.size()));
    out
}

/// [`lending`] for a spill write: takes back at most what it lent (or
/// `target`) and returns the rest of the consumer's credit to the pool. The
/// writer's scratch beyond the lent bytes is temporary, and may have come
/// from the pool's spill-scratch floor, which must not stay with one
/// consumer.
async fn lending_for_write<T>(
    res: &mut Reservation,
    handle: &ConsumerHandle,
    target: usize,
    work: impl Future<Output = T>,
) -> T {
    let lent = res.size();
    res.shrink_to_credit(lent);
    let out = work.await;
    // Credit only: this never reaches the pool, so it cannot fail.
    let _ = res.try_grow(handle.credit().min(lent.max(target)));
    handle.release_credit();
    let _ = res.try_grow(target.saturating_sub(res.size()));
    out
}

/// Buffer slack a sunk batch may carry over its data before it is
/// compacted.
const SLICE_SLACK_BYTES: usize = 16 << 10;

/// Rows per spilled batch: about [`SPILL_BATCH_BYTES`], at most a batch.
fn spill_rows(row_bytes: usize, batch_rows: usize) -> usize {
    (SPILL_BATCH_BYTES / row_bytes.max(1)).clamp(1, batch_rows.max(1))
}

/// Headroom to spill one buffer: one merged spill batch of `row_bytes`
/// data bytes per row (plus its buffers' alignment), the writer's scratch
/// (twice that) and its write buffer.
fn headroom(row_bytes: usize, batch_rows: usize) -> usize {
    row_bytes
        .saturating_mul(spill_rows(row_bytes, batch_rows))
        .saturating_add(BATCH_SLACK_BYTES)
        .saturating_mul(3)
        .saturating_add(SPILL_BUFFER_BYTES)
}

struct SortLocal {
    buffer: Buffer,
    runs: Vec<SpillRun>,
    signal: Arc<SpillSignal>,
    batch_rows: usize,
}

/// A combined driver's state, in combine order.
#[derive(Debug)]
enum Part {
    Run(SpillRun),
    Buffer(Buffer),
}

/// What `finalize` leaves for the source side.
enum Output {
    Memory(Arc<Held>),
    Spilled(Vec<SpillRun>),
}

/// An in-memory result, spillable until read (see
/// [`crate::output_spill`]): the buffers, then the source side's merge of
/// them between its steps. A spill writes the rest of the merge as one run.
struct Held {
    state: Mutex<HeldState>,
    /// Notified when a reader's step or a spill gives the merge back.
    changed: Event,
    /// The hook and the scope a spill writes to (with a spill scope only).
    spill: Option<(Arc<OutputSpill>, SpillScope)>,
    schema: SchemaRef,
    encoder: Arc<KeyEncoder>,
    /// The result's consumer: the buffers are charged to it.
    handle: Arc<ConsumerHandle>,
    counters: Arc<Counters>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    batch_rows: usize,
}

enum HeldState {
    /// Not opened yet.
    Buffers(Vec<Buffer>),
    /// The merge between the reader's steps, with what it merges.
    Merging(Box<Merger>, Vec<Buffer>),
    /// A reader's step or a spill holds the merge.
    Busy,
    /// The rest of the merge, spilled: the reader takes the run.
    Spilled(Option<SpillRun>),
    Done,
    /// The merge was lost (a failed spill or step): every reader fails
    /// with the error, never with a short result.
    Failed(KrishivError),
}

/// The merge and the buffers it reads, taken out of a [`Held`].
type Taken = (Box<Merger>, Vec<Buffer>);

impl Held {
    fn set_spillable(&self, bytes: usize) {
        if let Some((hook, _)) = &self.spill {
            hook.set_spillable_bytes(bytes);
        }
    }

    /// Takes the merge (opening it if needed) and leaves `Busy`; `Err` with
    /// the state found otherwise (`Busy`, `Spilled`, `Done`, `Failed`).
    /// Only the `reader` takes a spilled run (and leaves `Done`); for a
    /// spill every other state stays as it was (`Err(Done)`).
    fn take(&self, reader: bool) -> Result<std::result::Result<Taken, HeldState>> {
        let mut state = lock(&self.state);
        Ok(match mem::replace(&mut *state, HeldState::Busy) {
            HeldState::Buffers(held) => {
                let batches = held.iter().flat_map(|b| b.batches.iter().cloned());
                let merger = Merger::new(
                    Arc::clone(&self.schema),
                    Arc::clone(&self.encoder),
                    batches.collect(),
                    &[],
                    &self.handle,
                );
                match merger {
                    Ok(m) => Ok((Box::new(m), held)),
                    Err(e) => {
                        *state = HeldState::Failed(e.clone());
                        return Err(e);
                    }
                }
            }
            HeldState::Merging(m, held) => Ok((m, held)),
            HeldState::Failed(e) => {
                *state = HeldState::Failed(e.clone());
                Err(HeldState::Failed(e))
            }
            HeldState::Busy => Err(HeldState::Busy),
            // The spilled run handed to the reader: done here.
            HeldState::Spilled(run) if reader => {
                *state = HeldState::Done;
                Err(HeldState::Spilled(run))
            }
            // Done, or a run the reader has yet to take: nothing to spill.
            found => {
                *state = found;
                Err(HeldState::Done)
            }
        })
    }

    /// Ends a step or a spill: `next` is the new state.
    fn put(&self, next: HeldState) {
        *lock(&self.state) = next;
        self.changed.notify();
    }
}

impl SpillUnread for Held {
    fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>> {
        Box::pin(async move {
            let Some((_, scope)) = &self.spill else {
                return Ok(0);
            };
            let (mut merger, mut held) = loop {
                let changed = self.changed.listen();
                match self.take(false)? {
                    Ok(taken) => break taken,
                    // A reader's step: the merge comes back after it.
                    Err(HeldState::Busy) => changed.await,
                    Err(_) => return Ok(0),
                }
            };
            // The buffers' headroom is lent to the writer's scratch (as
            // grant credit, so no other consumer takes it meanwhile). The
            // writer first: if its scratch is refused, nothing was merged
            // yet and the merge goes back as it was.
            let mut freed = 0;
            for b in &mut held {
                let lent = b.headroom.size();
                freed += lent + b.reservation.size();
                b.headroom.shrink_to_credit(lent);
            }
            let scratch = self.handle.reservation().for_spill_scratch();
            let writer = match scope.writer(Arc::clone(&self.schema), scratch) {
                Ok(w) => w,
                Err(e) => {
                    self.handle.release_credit();
                    self.put(HeldState::Merging(merger, held));
                    return refused(e.into());
                }
            };
            self.set_spillable(0);
            // Dropped before the spill ends (the query was cancelled): the
            // merge is gone with the blocking job, so readers fail rather
            // than wait on `Busy`.
            let unfinished = Unfinished(Some(&self.state), &self.changed, || {
                HeldState::Failed(cancelled())
            });
            let row_bytes = held.iter().map(|b| b.row_bytes).max().unwrap_or(1);
            let rows = spill_rows(row_bytes, self.batch_rows);
            let query = Arc::clone(&self.query);
            let guard = self.query.blocking_started();
            let written = run_blocking(&*self.executor, move || {
                let run = write_rest(&mut merger, writer, rows, &query);
                // The merged batches go with the merge, whatever happened;
                // the guard travels with the run (a file), so the query
                // does not end before it is dropped.
                drop((merger, held));
                Ok((run, guard))
            })
            .await
            .and_then(|(run, _guard)| run);
            unfinished.disarm();
            self.handle.release_credit();
            match written {
                Ok(run) => {
                    self.counters.runs.fetch_add(1, Ordering::Relaxed);
                    let bytes = run.file.bytes();
                    self.counters.bytes.fetch_add(bytes, Ordering::Relaxed);
                    self.put(HeldState::Spilled(Some(run)));
                    Ok(freed)
                }
                Err(e) => {
                    // Rows already merged are lost: fail the query before
                    // the reader can see the state, and the reader too.
                    self.query.fail(e.clone());
                    self.put(HeldState::Failed(e.clone()));
                    Err(e)
                }
            }
        })
    }
}

#[derive(Default)]
struct SortShared {
    parts: Vec<Part>,
    output: Option<Output>,
}

#[derive(Default)]
struct SortGlobal(Mutex<SortShared>);

impl Sort {
    /// Spills `buffer` (not empty) as one sorted run on the blocking pool.
    /// Returns the run and empties the buffer, or `Ok(Err(_))` and keeps it
    /// when the pool refused the writer's scratch memory.
    async fn spill_buffer(
        &self,
        buffer: &mut Buffer,
        batch_rows: usize,
        ctx: &TaskContext,
    ) -> Result<Result<SpillRun, MemoryError>> {
        let target = headroom(buffer.row_bytes, batch_rows);
        let rows = spill_rows(buffer.row_bytes, batch_rows);
        let (batches, handle) = (buffer.batches.clone(), Arc::clone(&buffer.handle));
        let written = self.write(batches, Vec::new(), rows, &handle, ctx);
        let run = match lending_for_write(&mut buffer.headroom, &handle, target, written).await? {
            Ok(run) => run,
            Err(refused) => return Ok(Err(refused)),
        };
        buffer.batches.clear();
        let freed = buffer.reservation.size();
        buffer.reservation.shrink(freed);
        self.counters.runs.fetch_add(1, Ordering::Relaxed);
        Ok(Ok(run))
    }

    /// The in-memory result of `buffers`, charged to a consumer of its own
    /// whose hook spills the rest of it (with a spill scope).
    fn held(&self, mut buffers: Vec<Buffer>, ctx: &TaskContext) -> Result<Arc<Held>> {
        let (hook, handle) = match &self.spill {
            Some(scope) => {
                let (hook, handle) = OutputSpill::register(ctx, "Sort output");
                for b in &mut buffers {
                    for r in [&mut b.reservation, &mut b.headroom] {
                        r.transfer_to(&handle).map_err(KrishivError::from)?;
                    }
                }
                (Some((hook, scope.clone())), handle)
            }
            None => (None, Arc::clone(ctx.consumer().consumer())),
        };
        let bytes = buffers.iter().map(|b| b.reservation.size()).sum();
        let held = Arc::new(Held {
            state: Mutex::new(HeldState::Buffers(buffers)),
            changed: Event::default(),
            spill: hook,
            schema: Arc::clone(&self.schema),
            encoder: Arc::clone(&self.encoder),
            handle,
            counters: Arc::clone(&self.counters),
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
            batch_rows: ctx.query().config().batch_size.max(1),
        });
        if let Some((hook, _)) = &held.spill {
            let weak: Weak<dyn SpillUnread> = Arc::downgrade(&held) as Weak<Held>;
            hook.attach(weak);
            hook.set_spillable_bytes(bytes);
        }
        Ok(held)
    }

    /// Merges `memory` and `spilled` runs into one new spill file of
    /// batches of `rows` rows, on the blocking pool; scratch memory comes
    /// from `handle` (credit first). `Ok(Err(_))` if the pool refused it.
    async fn write(
        &self,
        memory: Vec<Arc<SortedBatch>>,
        spilled: Vec<SpillRun>,
        rows: usize,
        handle: &Arc<ConsumerHandle>,
        ctx: &TaskContext,
    ) -> Result<Result<SpillRun, MemoryError>> {
        let scope = self.spill.clone();
        let scope = scope.ok_or_else(|| internal("a sort spilled without a scope".to_owned()))?;
        let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
        let (h, query) = (Arc::clone(handle), Arc::clone(ctx.query()));
        let written = ctx
            .run_blocking(move || {
                let mut merger = Merger::new(schema, encoder, memory, &spilled, &h)?;
                let scratch = h.reservation().releasing_to_credit().for_spill_scratch();
                write_run(&mut merger, &scope, scratch, rows, &query)
            })
            .await?;
        if let Ok(run) = &written {
            self.counters
                .bytes
                .fetch_add(run.file.bytes(), Ordering::Relaxed);
        }
        Ok(written)
    }

    /// Merges consecutive runs, up to `fan_in` at a time and level by level,
    /// until at most `fan_in` (what the memory pool allows) remain. A merged
    /// group takes its first run's place, so runs keep their relative order
    /// (and ties their input order).
    async fn reduce(
        &self,
        runs: &mut Vec<SpillRun>,
        batch_rows: usize,
        ctx: &TaskContext,
    ) -> Result<()> {
        if runs.len() <= 2 {
            return Ok(());
        }
        let handle = Arc::clone(ctx.consumer().consumer());
        let (mut budget, fan_in) =
            merge_budget(runs, self.max_fan_in, batch_rows, &handle, true).await?;
        let target = budget.size();
        while runs.len() > fan_in {
            self.counters.passes.fetch_add(1, Ordering::Relaxed);
            let mut level = mem::take(runs).into_iter();
            while level.len() > 0 {
                // Stop merging once the rest fits the fan-in.
                if runs.len() + level.len() <= fan_in {
                    runs.extend(level.by_ref());
                    break;
                }
                let take = group_size(runs.len() + level.len(), fan_in);
                let group: Vec<SpillRun> = level.by_ref().take(take).collect();
                let row_bytes = group.iter().map(|r| r.row_bytes).max().unwrap_or(1);
                let rows = spill_rows(row_bytes, batch_rows);
                let merged = self.write(Vec::new(), group, rows, &handle, ctx);
                let run = lending_for_write(&mut budget, &handle, target, merged).await?;
                runs.push(run.map_err(KrishivError::from)?);
            }
        }
        Ok(())
    }
}

/// Runs to merge next when `total` remain at a level (more than `fan_in`):
/// a group of `g` leaves `g - 1` fewer, so never more than it takes to get
/// down to the fan-in, and as few bytes as possible are written again.
pub(super) fn group_size(total: usize, fan_in: usize) -> usize {
    fan_in.min(total.saturating_sub(fan_in) + 1)
}

/// Reserves (with arbitration) the memory to merge up to `max_fan_in` of
/// `runs` at once. An intermediate merge (`writing` a run) halves the
/// fan-in down to 2 while the pool refuses and also reserves the spill
/// writer's scratch; the final merge needs every run.
pub(super) async fn merge_budget(
    runs: &[SpillRun],
    max_fan_in: usize,
    batch_rows: usize,
    handle: &Arc<ConsumerHandle>,
    writing: bool,
) -> Result<(Reservation, usize)> {
    let per_run = runs.iter().map(SpillRun::merge_bytes).max().unwrap_or(0);
    let row_bytes = runs.iter().map(|r| r.row_bytes).max().unwrap_or(1);
    let out = if writing {
        headroom(row_bytes, batch_rows)
    } else {
        row_bytes.saturating_mul(batch_rows)
    };
    let mut fan_in = max_fan_in.min(runs.len());
    let mut budget = handle.reservation();
    loop {
        // The merger's per-source `slot`/`stamp` metadata (see
        // [`super::merge::metadata_bytes`]): small next to the frames, but
        // reserved with them so the merge never grows past its budget.
        let meta = super::merge::metadata_bytes(fan_in);
        let need = per_run
            .saturating_mul(fan_in)
            .saturating_add(out)
            .saturating_add(meta);
        match budget.grow(need).await {
            Ok(()) => return Ok((budget, fan_in)),
            Err(_) if writing && fan_in > 2 => fan_in = (fan_in / 2).max(2),
            Err(e) => return Err(e.into()),
        }
    }
}

impl Sink for Sort {
    fn name(&self) -> &'static str {
        "Sort"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(SortGlobal::default()))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        Ok(Box::new(SortLocal {
            buffer: Buffer::new(memory.consumer()),
            runs: Vec::new(),
            signal: Arc::clone(memory.signal()),
            batch_rows: ctx.query().config().batch_size.max(1),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let l = local_mut::<SortLocal>(local, "Sort")?;
        let rows = input.num_rows();
        if rows == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        let data = data_bytes(&input);
        let row_bytes = l.buffer.row_bytes.max(data.div_ceil(rows));
        // A small slice of a large batch would keep (and account) the whole
        // batch: keep a copy of its rows instead.
        let input = if input.data().get_array_memory_size() > 2 * data + SLICE_SLACK_BYTES {
            let rows: Vec<(usize, usize)> = (0..rows).map(|r| (0, r)).collect();
            interleave_batches(&input.schema(), &[input], &rows)?
        } else {
            input
        };
        let sorted = SortedBatch::new(input, &self.encoder)?;
        let bytes = sorted.bytes();
        let extra = if self.spill.is_some() {
            headroom(row_bytes, l.batch_rows).saturating_sub(l.buffer.headroom.size())
        } else {
            0
        };
        let b = &mut l.buffer;
        if b.reservation.try_grow(bytes + extra).is_err() {
            return Ok(SinkStatus::NeedMemory {
                bytes: bytes + extra,
            });
        }
        // Moves `extra` to the headroom (through the consumer's credit).
        b.reservation.shrink_to_credit(extra);
        b.headroom.try_grow(extra)?;
        b.row_bytes = row_bytes;
        b.batches.push(Arc::new(sorted));
        self.counters.rows.fetch_add(rows as u64, Ordering::Relaxed);
        if self.spill.is_some() {
            l.signal.set_spillable_bytes(b.reservation.size());
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<SortGlobal>(global, "Sort")?;
        let l = local
            .downcast::<SortLocal>()
            .map_err(|_| internal("wrong local state for Sort".to_owned()))?;
        let SortLocal {
            buffer,
            runs,
            signal,
            ..
        } = *l;
        let mut shared = lock(&g.0);
        shared.parts.extend(runs.into_iter().map(Part::Run));
        if !buffer.batches.is_empty() {
            shared.parts.push(Part::Buffer(buffer));
        }
        // The last driver to combine finalizes: its signal offers the whole
        // global state to `spill_global` meanwhile.
        if self.spill.is_some() {
            signal.set_spillable_bytes(buffered(&shared.parts));
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<SortGlobal>(global, "Sort")?;
            let parts = mem::take(&mut lock(&g.0).parts);
            ctx.consumer().signal().set_spillable_bytes(0);
            let batch_rows = ctx.query().config().batch_size.max(1);
            if !parts.iter().any(|p| matches!(p, Part::Run(_))) {
                let buffers = parts
                    .into_iter()
                    .filter_map(|p| match p {
                        Part::Buffer(mut b) => {
                            // Without a spill scope nothing spills any
                            // more: the headroom goes. With one, the
                            // result's spill writes through it.
                            if self.spill.is_none() {
                                b.headroom.shrink(b.headroom.size());
                            }
                            Some(b)
                        }
                        Part::Run(_) => None,
                    })
                    .collect();
                let held = self.held(buffers, ctx)?;
                lock(&g.0).output = Some(Output::Memory(held));
                return Ok(());
            }
            // External: spill what is still buffered, so the output phase
            // holds only the merge's frames.
            let mut runs = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    Part::Run(r) => runs.push(r),
                    Part::Buffer(mut b) => {
                        let mut run = self.spill_buffer(&mut b, batch_rows, ctx).await?;
                        if run.is_err() {
                            // The scratch outgrew the headroom: widen it,
                            // with arbitration this time, and try again.
                            let more = headroom(b.row_bytes, batch_rows);
                            b.headroom.grow(more).await?;
                            run = self.spill_buffer(&mut b, batch_rows, ctx).await?;
                        }
                        runs.push(run?);
                    }
                }
            }
            self.reduce(&mut runs, batch_rows, ctx).await?;
            lock(&g.0).output = Some(Output::Spilled(runs));
            Ok(())
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = local_mut::<SortLocal>(local, "Sort")?;
            let before = l.buffer.reservation.size();
            if l.buffer.batches.is_empty() || self.spill.is_none() {
                return Ok(0);
            }
            let Ok(run) = self.spill_buffer(&mut l.buffer, l.batch_rows, ctx).await? else {
                // Refused scratch memory: nothing freed this time.
                return Ok(0);
            };
            l.runs.push(run);
            l.signal.set_spillable_bytes(0);
            Ok(before)
        })
    }

    fn spill_global<'a>(
        &'a self,
        global: &'a GlobalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let g = global_ref::<SortGlobal>(global, "Sort")?;
            if self.spill.is_none() {
                return Ok(0);
            }
            // Out of the lock while spilling; `finalize` is not polled
            // meanwhile, and an error fails the query (the parts go).
            let mut parts = mem::take(&mut lock(&g.0).parts);
            let batch_rows = ctx.query().config().batch_size.max(1);
            let mut freed = 0;
            for part in &mut parts {
                let Part::Buffer(b) = part else { continue };
                let before = b.reservation.size();
                if let Ok(run) = self.spill_buffer(b, batch_rows, ctx).await? {
                    freed += before;
                    *part = Part::Run(run);
                }
            }
            let mut shared = lock(&g.0);
            parts.append(&mut shared.parts);
            shared.parts = parts;
            ctx.consumer()
                .signal()
                .set_spillable_bytes(buffered(&shared.parts));
            Ok(freed)
        })
    }
}

/// Bytes held by the buffers among `parts`.
fn buffered(parts: &[Part]) -> usize {
    parts
        .iter()
        .map(|p| match p {
            Part::Buffer(b) => b.reservation.size(),
            Part::Run(_) => 0,
        })
        .sum()
}

/// The source side's per-driver state.
struct SortSource {
    state: SourceState,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    handle: Arc<ConsumerHandle>,
    batch_rows: usize,
}

/// The final merge, holding what it merges: the buffers' reservations (in
/// memory) or the budget for the runs' frames (spilled).
struct Merging {
    merger: Box<Merger>,
    held: Vec<Buffer>,
    budget: Reservation,
}

enum SourceState {
    Start,
    /// Reading an in-memory result through its shared state.
    Held(Arc<Held>),
    Merging(Merging),
    Done,
}

impl Source for Sort {
    fn name(&self) -> &'static str {
        "Sort"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        finalized::<SortGlobal>(ctx, "Sort")
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(SortSource {
            state: SourceState::Start,
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
            handle: Arc::clone(ctx.consumer().consumer()),
            batch_rows: ctx.query().config().batch_size.max(1),
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let l = local_mut::<SortSource>(local, "Sort")?;
            if matches!(l.state, SourceState::Start) {
                l.state = self.start(l, global).await?;
            }
            if let SourceState::Held(held) = &l.state {
                let held = Arc::clone(held);
                match read_held(l, &held).await? {
                    Ok(next) => return Ok(next),
                    // The rest was spilled: merge it from disk.
                    Err(run) => l.state = self.open_runs(l, vec![run]).await?,
                }
            }
            let SourceState::Merging(Merging {
                mut merger,
                held,
                mut budget,
            }) = mem::replace(&mut l.state, SourceState::Done)
            else {
                return Ok(None);
            };
            // Each output batch on the blocking pool: a wide merge over
            // ~1 ms, or reading spilled runs.
            let rows = l.batch_rows;
            let step = blocking(&l.query, &l.executor, move || {
                let next = merger.next(rows)?;
                Ok((merger, next))
            });
            // No top-up: the merger keeps the frames it drew, and frames it
            // frees come back as credit for the next step.
            let (merger, next) = lending(&mut budget, &l.handle, 0, step).await?;
            if next.is_some() {
                l.state = SourceState::Merging(Merging {
                    merger,
                    held,
                    budget,
                });
            } else {
                // Spilled runs' last frames went back as credit.
                drop((merger, held, budget));
                l.handle.release_credit();
            }
            Ok(next)
        })
    }
}

impl Sort {
    /// Takes `finalize`'s output and opens the final merge.
    async fn start(&self, l: &SortSource, global: &GlobalState) -> Result<SourceState> {
        let g = global_ref::<SortGlobal>(global, "Sort")?;
        let output = lock(&g.0).output.take();
        match output {
            None => Ok(SourceState::Done),
            Some(Output::Memory(held)) => Ok(SourceState::Held(held)),
            Some(Output::Spilled(runs)) => self.open_runs(l, runs).await,
        }
    }

    /// Opens the final merge of spilled `runs`.
    async fn open_runs(&self, l: &SortSource, runs: Vec<SpillRun>) -> Result<SourceState> {
        let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
        let handle = Arc::clone(&l.handle);
        let mut budget = merge_budget(&runs, runs.len(), l.batch_rows, &handle, false)
            .await?
            .0;
        self.counters
            .fan_in
            .store(runs.len() as u64, Ordering::Relaxed);
        let open = blocking(&l.query, &l.executor, move || {
            Merger::new(schema, encoder, Vec::new(), &runs, &handle)
        });
        let merger = lending(&mut budget, &l.handle, 0, open).await?;
        Ok(SourceState::Merging(Merging {
            merger: Box::new(merger),
            held: Vec::new(),
            budget,
        }))
    }
}

/// Asks a reader's in-memory result to spill again, as a request made on
/// stale spillable bytes would (tests).
#[cfg(test)]
pub(super) async fn spill_result_again(local: &mut LocalState) -> Result<usize> {
    let l = local_mut::<SortSource>(local, "Sort")?;
    let SourceState::Held(held) = &l.state else {
        return Err(internal("the reader holds no in-memory result".to_owned()));
    };
    Arc::clone(held).spill_unread().await
}

/// One step of an in-memory result's merge: the next batch, or `Err` with
/// the run its rest was spilled to.
async fn read_held(
    l: &mut SortSource,
    held: &Arc<Held>,
) -> Result<std::result::Result<Option<Batch>, SpillRun>> {
    let (mut merger, buffers) = loop {
        let changed = held.changed.listen();
        match held.take(true)? {
            Ok(taken) => break taken,
            // A spill of the rest is being written.
            Err(HeldState::Busy) => changed.await,
            Err(HeldState::Spilled(Some(run))) => return Ok(Err(run)),
            Err(HeldState::Failed(e)) => return Err(e),
            Err(_) => {
                l.state = SourceState::Done;
                return Ok(Ok(None));
            }
        }
    };
    let rows = l.batch_rows;
    let step = blocking(&l.query, &l.executor, move || {
        let next = merger.next(rows);
        Ok((merger, next))
    })
    .await;
    let (merger, next) = match step {
        Ok((merger, next)) => (merger, next),
        Err(e) => {
            held.set_spillable(0);
            held.put(HeldState::Failed(e.clone()));
            return Err(e);
        }
    };
    match next {
        Ok(Some(batch)) => {
            held.put(HeldState::Merging(merger, buffers));
            Ok(Ok(Some(batch)))
        }
        Ok(None) => {
            // Read to the end: the buffers go.
            held.set_spillable(0);
            drop((merger, buffers));
            held.put(HeldState::Done);
            l.state = SourceState::Done;
            Ok(Ok(None))
        }
        Err(e) => {
            held.set_spillable(0);
            drop((merger, buffers));
            held.put(HeldState::Failed(e.clone()));
            Err(e)
        }
    }
}

/// Writes the rest of `merger` with `writer` in batches of `rows` rows,
/// checking cancellation between batches (blocking I/O).
fn write_rest(
    merger: &mut Merger,
    mut writer: krishiv_memory::SpillWriter,
    rows: usize,
    query: &QueryContext,
) -> Result<SpillRun> {
    let (mut max_batch_bytes, mut row_bytes) = (0, 1);
    while let Some(batch) = merger.next(rows)? {
        query.check_cancelled()?;
        max_batch_bytes = max_batch_bytes.max(batch.data().get_array_memory_size());
        row_bytes = row_bytes.max(data_bytes(&batch).div_ceil(batch.num_rows().max(1)));
        writer.write(&batch).map_err(KrishivError::from)?;
    }
    let file = writer.finish().map_err(KrishivError::from)?;
    Ok(SpillRun {
        file,
        max_batch_bytes,
        row_bytes,
    })
}
