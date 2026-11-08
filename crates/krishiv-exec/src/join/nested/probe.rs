// SPDX-License-Identifier: Apache-2.0
//! The nested-loop join's probe operator: streams its input against a build
//! held in memory, or, once the build spilled, buffers and spills its input
//! and joins it build chunk by build chunk in the flush phase (see the
//! [module docs](super)).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{
    MemoryError, Reservation, Seat, SeatState, SpillError, SpillFile, SpillReader, SpillSignal,
};
use krishiv_types::batch::Batch;

use super::build::{Ready, ready, write};
use super::{BuildRows, Pairing, ProbeBits, Rest, Shared, unit_weights};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{FlushOutput, LocalState, Operator, OperatorOutput};
use crate::operators::{local_mut, owned_rows};

/// The most pairing steps (each at most `TARGET_PAIRS` pairs) one
/// `execute` call runs before it returns.
const STEPS_PER_CALL: usize = 4;

pub(super) struct ProbeOp(pub(super) Arc<Shared>);

struct ProbeLocal {
    ready: Arc<Ready>,
    /// The driver's partition: the condition evaluates in it.
    partition: u32,
    /// Progress through the current probe batch (a build in memory).
    cur: Pairing,
    /// The block nested loop of a spilled build.
    disk: Option<Box<Disk>>,
}

/// Where a driver's block nested loop stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The barrier passed: size the chunks.
    Start,
    /// Reading the next build chunk.
    Load,
    /// Joining every probe row with the chunk.
    Probe,
    /// After the last chunk: the probe rows the bitmap selects.
    Unmatched,
    /// Done with the chunks: count this driver out.
    Build,
    /// The build rows of a build-side join (the last driver only).
    Emit,
    Done,
}

/// One driver's state when the build spilled.
struct Disk {
    /// Probe batches not written yet, and their memory.
    buffered: Vec<RecordBatch>,
    reservation: Reservation,
    signal: Arc<SpillSignal>,
    /// The driver's probe rows on disk.
    files: Vec<Arc<SpillFile>>,
    rows: usize,
    /// Which probe rows matched in some chunk, and its memory.
    bits: ProbeBits,
    bits_memory: Reservation,
    phase: Phase,
    /// The most bytes of build rows a chunk reserves.
    budget: usize,
    /// The build read so far: whether the in-memory batches were a chunk,
    /// the next file and its reader, the number of the next row read.
    memory_done: bool,
    next_file: usize,
    build_reader: Option<SpillReader>,
    next_base: u64,
    /// A build batch read past a full chunk: the first of the next one.
    pending: Option<(RecordBatch, u64)>,
    /// The chunk: batches, their first row numbers, their memory.
    chunk: Vec<RecordBatch>,
    chunk_bases: Vec<u64>,
    chunk_memory: Reservation,
    /// This driver's seat among the query's chunked nested loops from its
    /// first chunk until it is done ([`CHUNK_COHORT`]): holding while its
    /// chunk is reserved; with no chunk, a refused frame or batch waits for
    /// other members (`reserve_in_cohort`). Declared after `chunk_memory`, so on a drop the chunk is
    /// released before the seat wakes the members waiting for it.
    seat: Option<Seat>,
    /// The probe rows being re-read: file, reader, the number of the next
    /// row, and the batch being joined with its first row's number.
    probe_file: usize,
    probe_reader: Option<SpillReader>,
    probe_base: usize,
    current: Option<(RecordBatch, usize)>,
    cur: Pairing,
    /// The in-memory build batches emitted so far (phase `Emit`).
    emitted: usize,
}

impl Disk {
    fn new(ctx: &TaskContext) -> Self {
        let memory = ctx.consumer();
        Self {
            buffered: Vec::new(),
            reservation: memory.reservation(),
            signal: Arc::clone(memory.signal()),
            files: Vec::new(),
            rows: 0,
            bits: ProbeBits::default(),
            bits_memory: memory.reservation(),
            phase: Phase::Start,
            budget: 0,
            memory_done: false,
            next_file: 0,
            build_reader: None,
            next_base: 0,
            pending: None,
            chunk: Vec::new(),
            chunk_bases: Vec::new(),
            chunk_memory: memory.reservation(),
            seat: None,
            probe_file: 0,
            probe_reader: None,
            probe_base: 0,
            current: None,
            cur: Pairing::default(),
            emitted: 0,
        }
    }

    /// Whether every build row was in some chunk.
    fn build_read(&self, ready: &Ready) -> bool {
        self.memory_done && self.pending.is_none() && self.next_file >= ready.files.len()
    }

    fn drop_chunk(&mut self) {
        self.chunk.clear();
        self.chunk_bases.clear();
        let held = self.chunk_memory.size();
        self.chunk_memory.shrink(held);
        // After the release: waking members first would find no room.
        if let Some(seat) = &mut self.seat {
            seat.set(SeatState::Idle);
        }
    }
}

/// Grows `r` by `bytes`, which a driver holding no chunk needs to go on.
/// With no seat, as [`Reservation::grow`]. With one, the seat is
/// [`SeatState::Reserving`] while it asks: arbitration spills without
/// aborting; if that is not enough and another member of the cohort is
/// active (holds a chunk, freed at the end of its pass, or is asking), the
/// seat stands by ([`Seat::stand_by`]: idle without waking anyone, so two
/// askers never wait for each other and waiters do not wake each other),
/// waits for a member to go idle and asks again, counting the wait in
/// `waits`. With no other member active, the full arbitration runs, which
/// may fail the request or abort a query. The seat ends in `then` once
/// granted, idle on an error or if the future is dropped (which wakes the
/// cohort).
async fn reserve_in_cohort(
    r: &mut Reservation,
    bytes: usize,
    seat: Option<(&mut Seat, &AtomicU64)>,
    then: SeatState,
) -> Result<()> {
    /// Leaves the seat idle (waking the cohort) unless it ended otherwise.
    struct Asking<'a>(&'a mut Seat);
    impl Drop for Asking<'_> {
        fn drop(&mut self) {
            if self.0.state() == SeatState::Reserving {
                self.0.set(SeatState::Idle);
            }
        }
    }
    let Some((seat, waits)) = seat else {
        return r.grow(bytes).await.map_err(KrishivError::from);
    };
    debug_assert_eq!(seat.state(), SeatState::Idle, "a seat asks from idle");
    let cohort = Arc::clone(seat.cohort());
    let asking = Asking(seat);
    let out = loop {
        asking.0.set(SeatState::Reserving);
        match r.grow_without_abort(bytes).await {
            Ok(()) => break Ok(()),
            Err(MemoryError::Exhausted { .. }) => {}
            Err(e) => break Err(e),
        }
        asking.0.stand_by();
        // Listen before checking: a member going idle meanwhile wakes us.
        let changed = cohort.changed();
        if !cohort.others_active(asking.0) {
            asking.0.set(SeatState::Reserving);
            let out = r.grow(bytes).await;
            if out.is_err() {
                tracing::debug!(
                    bytes,
                    members = cohort.members(),
                    "nested-loop cohort: no other member active, request refused"
                );
            }
            break out;
        }
        waits.fetch_add(1, Ordering::Relaxed);
        changed.await;
    };
    // Granted: `then`; failed: idle (the guard, waking the cohort).
    if out.is_ok() {
        asking.0.set(then);
    }
    drop(asking);
    out.map_err(KrishivError::from)
}

/// The cohort of a query's chunked nested loops: their chunk budgets split
/// one quarter of the limit, and a first batch that does not fit waits for
/// another member's chunk.
const CHUNK_COHORT: &str = "nested-loop chunks";

/// The next batch of the reader in `slot` (opened on `file` when empty),
/// opened and read on the blocking pool. A frame the pool refuses is waited
/// for and read again: given `cohort` (a driver with no chunk reading the
/// first batch of its next one), as [`reserve_in_cohort`] waits. `None` at
/// the end of the file, which closes the reader.
async fn next_batch(
    ctx: &TaskContext,
    slot: &mut Option<SpillReader>,
    file: &Arc<SpillFile>,
    mut cohort: Option<(&mut Seat, &AtomicU64)>,
) -> Result<Option<RecordBatch>> {
    loop {
        let open = slot.take();
        let (f, reservation) = (Arc::clone(file), ctx.consumer().reservation());
        let (reader, item) = ctx
            .run_blocking(move || {
                let mut reader = match open {
                    Some(r) => r,
                    None => f.reader(reservation).map_err(KrishivError::from)?,
                };
                let item = reader.next();
                Ok((reader, item))
            })
            .await?;
        match item {
            None => return Ok(None),
            Some(Ok(b)) => {
                *slot = Some(reader);
                ctx.consumer().consumer().release_credit();
                return Ok(Some(b.into_parts().0));
            }
            Some(Err(SpillError::Memory(_))) => {
                *slot = Some(reader);
                // Wait for the frame's memory, then read it again: the reader
                // draws the bytes from the consumer's credit.
                let bytes = file.max_read_reservation();
                let mut topup = ctx.consumer().reservation();
                let seat = cohort.as_mut().map(|(s, w)| (&mut **s, *w));
                reserve_in_cohort(&mut topup, bytes, seat, SeatState::Idle).await?;
                topup.shrink_to_credit(bytes);
            }
            Some(Err(e)) => return Err(e.into()),
        }
    }
}

impl Operator for ProbeOp {
    fn name(&self) -> &'static str {
        "NestedLoopJoin"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::clone(&self.0.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        let global = ctx
            .dependency()
            .ok_or_else(|| internal("nested-loop probe before its build".to_owned()))?;
        let ready = ready(global)?;
        let disk = ready.spilled().then(|| Box::new(Disk::new(ctx)));
        Ok(Box::new(ProbeLocal {
            ready,
            partition: ctx.partition(),
            cur: Pairing::default(),
            disk,
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        unit_weights(input)?;
        let l = local_mut::<ProbeLocal>(local, "NestedLoopJoin")?;
        if let Some(d) = &mut l.disk {
            // The build spilled: the rows wait for the flush phase, held for
            // their own bytes, not for a larger batch they are a slice of.
            let input = owned_rows(input.clone())?;
            let bytes = input.data().get_array_memory_size();
            if d.reservation.try_grow(bytes).is_err() {
                return Ok(OperatorOutput::NeedMemory { bytes });
            }
            d.buffered.push(input.into_parts().0);
            d.signal.set_spillable_bytes(d.reservation.size());
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let build = BuildRows {
            batches: &l.ready.memory,
            bases: &l.ready.memory_bases,
        };
        for _ in 0..STEPS_PER_CALL {
            let (out, done) = self.0.step(
                input.data(),
                build,
                &mut l.cur,
                l.partition,
                l.ready.marks.as_ref(),
                Rest::Emit,
            )?;
            if done {
                return Ok(OperatorOutput::NeedMoreInput(out.map(Batch::new)));
            }
            if let Some(b) = out {
                return Ok(OperatorOutput::HaveMoreOutput(Batch::new(b)));
            }
        }
        // Nothing matched yet (a selective condition, or a join that
        // outputs no pairs): an empty batch ends the call, so it stays
        // bounded and the driver can yield.
        Ok(OperatorOutput::HaveMoreOutput(Batch::new(
            RecordBatch::new_empty(Arc::clone(&self.0.schema)),
        )))
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let Some(scope) = &self.0.spec.spill else {
                return Ok(0);
            };
            let l = local_mut::<ProbeLocal>(local, "NestedLoopJoin")?;
            let Some(d) = &mut l.disk else {
                return Ok(0);
            };
            if d.buffered.is_empty() {
                return Ok(0);
            }
            let freed = d.reservation.size();
            let batches = d.buffered.clone();
            let schema = Arc::clone(&self.0.spec.probe_schema);
            let file = match write(ctx, scope, schema, batches, &mut d.reservation, false).await {
                Ok(f) => f,
                Err(e) => return crate::output_spill::refused(e),
            };
            d.rows += d.buffered.drain(..).map(|b| b.num_rows()).sum::<usize>();
            d.signal.set_spillable_bytes(0);
            d.files.push(Arc::new(file));
            self.0.counters.probe_spills.fetch_add(1, Ordering::Relaxed);
            Ok(freed)
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        let c = &self.0.counters;
        crate::metrics::counts([
            ("build_spills", c.build_spills.load(Ordering::Relaxed)),
            ("probe_spills", c.probe_spills.load(Ordering::Relaxed)),
            ("build_chunks", c.chunks.load(Ordering::Relaxed)),
            (
                "build_chunks_capped",
                c.capped_chunks.load(Ordering::Relaxed),
            ),
            ("build_chunk_waits", c.chunk_waits.load(Ordering::Relaxed)),
        ])
    }

    fn needs_flush(&self) -> bool {
        self.0.join_type().builds_preserved() || self.0.spec.spill.is_some()
    }

    fn end_input<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let l = local_mut::<ProbeLocal>(local, "NestedLoopJoin")?;
            l.ready.drivers.fetch_add(1, Ordering::AcqRel);
            let Some(d) = &mut l.disk else {
                return Ok(());
            };
            if !d.buffered.is_empty() {
                let scope = self.0.spec.spill.as_ref().ok_or_else(|| {
                    internal("a spilled nested-loop build without a spill scope".to_owned())
                })?;
                let batches = std::mem::take(&mut d.buffered);
                let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
                let schema = Arc::clone(&self.0.spec.probe_schema);
                let file = write(ctx, scope, schema, batches, &mut d.reservation, true).await?;
                d.rows += rows;
                d.signal.set_spillable_bytes(0);
                d.files.push(Arc::new(file));
                self.0.counters.probe_spills.fetch_add(1, Ordering::Relaxed);
            }
            if self.0.tracks_probe() && d.rows > 0 {
                let words = ProbeBits::words(d.rows);
                d.bits_memory
                    .grow(words.saturating_mul(8))
                    .await
                    .map_err(KrishivError::from)?;
                d.bits = ProbeBits(vec![0; words]);
            }
            Ok(())
        })
    }

    fn flush<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<FlushOutput>> {
        Box::pin(async move {
            let l = local_mut::<ProbeLocal>(local, "NestedLoopJoin")?;
            if l.disk.is_some() {
                return self.flush_disk(l, ctx).await;
            }
            // A build in memory: the drivers share out its batches and emit
            // the build rows the join returns.
            let Some(marks) = &l.ready.marks else {
                return Ok(FlushOutput::Done);
            };
            let i = l.ready.claim.fetch_add(1, Ordering::Relaxed);
            let (Some(b), Some(&base)) = (l.ready.memory.get(i), l.ready.memory_bases.get(i))
            else {
                return Ok(FlushOutput::Done);
            };
            Ok(match self.0.build_rows(b, base, marks)? {
                Some(out) => FlushOutput::Batch(Batch::new(out)),
                None => FlushOutput::Continue,
            })
        })
    }
}

impl ProbeOp {
    /// One step of a driver's block nested loop (a spilled build).
    async fn flush_disk(&self, l: &mut ProbeLocal, ctx: &TaskContext) -> Result<FlushOutput> {
        let ready = Arc::clone(&l.ready);
        let partition = l.partition;
        let d = l
            .disk
            .as_deref_mut()
            .ok_or_else(|| internal("nested-loop flush without its disk state".to_owned()))?;
        match d.phase {
            Phase::Start => {
                d.phase = if d.rows == 0 {
                    Phase::Build
                } else {
                    d.seat = ctx
                        .consumer()
                        .consumer()
                        .pool()
                        .cohort(CHUNK_COHORT)
                        .map(|c| c.join());
                    Phase::Load
                };
                d.budget = chunk_budget(ctx, &ready, d.seat.as_ref());
                Ok(FlushOutput::Continue)
            }
            Phase::Load => {
                self.load(d, &ready, ctx).await?;
                Ok(FlushOutput::Continue)
            }
            Phase::Probe => {
                if d.current.is_none() {
                    let Some(file) = d.files.get(d.probe_file).cloned() else {
                        // The chunk met every probe row.
                        d.drop_chunk();
                        d.budget = chunk_budget(ctx, &ready, d.seat.as_ref());
                        (d.probe_file, d.probe_base) = (0, 0);
                        d.phase = if d.build_read(&ready) {
                            Phase::Unmatched
                        } else {
                            Phase::Load
                        };
                        return Ok(FlushOutput::Continue);
                    };
                    match next_batch(ctx, &mut d.probe_reader, &file, None).await? {
                        Some(b) => {
                            let rows = b.num_rows();
                            d.current = Some((b, d.probe_base));
                            d.probe_base += rows;
                        }
                        None => d.probe_file += 1,
                    }
                    return Ok(FlushOutput::Continue);
                }
                let (probe, base) = d
                    .current
                    .as_ref()
                    .ok_or_else(|| internal("nested-loop probe batch".to_owned()))?;
                let build = BuildRows {
                    batches: &d.chunk,
                    bases: &d.chunk_bases,
                };
                let rest = if self.0.tracks_probe() {
                    Rest::Record(&mut d.bits, *base)
                } else {
                    Rest::Emit
                };
                let (out, done) = self.0.step(
                    probe,
                    build,
                    &mut d.cur,
                    partition,
                    ready.marks.as_ref(),
                    rest,
                )?;
                if done {
                    d.current = None;
                }
                Ok(out.map_or(FlushOutput::Continue, |b| FlushOutput::Batch(Batch::new(b))))
            }
            Phase::Unmatched => {
                let file = match d.files.get(d.probe_file).cloned() {
                    Some(f) if self.0.tracks_probe() => f,
                    _ => {
                        d.probe_reader = None;
                        let held = d.bits_memory.size();
                        d.bits = ProbeBits::default();
                        d.bits_memory.shrink(held);
                        d.phase = Phase::Build;
                        return Ok(FlushOutput::Continue);
                    }
                };
                // No chunk is held now: a refused frame waits in the cohort.
                let cohort = d.seat.as_mut().map(|s| (s, &self.0.counters.chunk_waits));
                let Some(b) = next_batch(ctx, &mut d.probe_reader, &file, cohort).await? else {
                    d.probe_file += 1;
                    return Ok(FlushOutput::Continue);
                };
                let matched: Vec<bool> = (0..b.num_rows())
                    .map(|i| d.bits.get(d.probe_base + i))
                    .collect();
                d.probe_base += b.num_rows();
                Ok(match self.0.unmatched(&b, 0, &matched)? {
                    Some(out) => FlushOutput::Batch(Batch::new(out)),
                    None => FlushOutput::Continue,
                })
            }
            Phase::Build => Ok(Self::finish_chunks(d, &ready)),
            Phase::Emit => self.emit_build(d, &ready, ctx).await,
            Phase::Done => Ok(FlushOutput::Done),
        }
    }

    /// Reads one build batch into the chunk; the chunk is complete once it
    /// holds its budget or the build is read (then the probe rows go
    /// through it).
    async fn load(&self, d: &mut Disk, ready: &Ready, ctx: &TaskContext) -> Result<()> {
        if !d.memory_done {
            // The batches the build kept in memory are the first chunk,
            // already reserved.
            d.memory_done = true;
            if !ready.memory.is_empty() {
                d.chunk = ready.memory.to_vec();
                d.chunk_bases = ready.memory_bases.to_vec();
                return self.start_probe(d);
            }
        }
        if let Some((b, base)) = d.pending.take() {
            self.first_batch(d, b.get_array_memory_size()).await?;
            d.chunk.push(b);
            d.chunk_bases.push(base);
        }
        if d.chunk_memory.size() >= d.budget {
            return self.start_probe(d);
        }
        let Some((file, base)) = ready.files.get(d.next_file).cloned() else {
            return self.start_probe(d);
        };
        if d.build_reader.is_none() {
            d.next_base = base;
        }
        // A driver with no chunk waits for its first batch's frame as it
        // waits for the batch (`reserve_in_cohort`); one holding a chunk
        // must not wait on others (two holders would wait for each other).
        let cohort = d
            .seat
            .as_mut()
            .filter(|_| d.chunk.is_empty())
            .map(|s| (s, &self.0.counters.chunk_waits));
        let Some(b) = next_batch(ctx, &mut d.build_reader, &file, cohort).await? else {
            d.next_file += 1;
            if d.next_file >= ready.files.len() {
                return self.start_probe(d);
            }
            return Ok(());
        };
        let (bytes, base) = (b.get_array_memory_size(), d.next_base);
        d.next_base += b.num_rows() as u64;
        if d.chunk.is_empty() {
            self.first_batch(d, bytes).await?;
        } else if !leaves_room(ctx, bytes) {
            self.0
                .counters
                .capped_chunks
                .fetch_add(1, Ordering::Relaxed);
            d.pending = Some((b, base));
            return self.start_probe(d);
        } else if d.chunk_memory.try_grow(bytes).is_err() {
            d.pending = Some((b, base));
            return self.start_probe(d);
        }
        d.chunk.push(b);
        d.chunk_bases.push(base);
        Ok(())
    }

    /// Reserves a chunk's first batch, which it needs to make progress, as
    /// [`reserve_in_cohort`] does; the driver then holds a chunk.
    async fn first_batch(&self, d: &mut Disk, bytes: usize) -> Result<()> {
        let seat = d.seat.as_mut().map(|s| (s, &self.0.counters.chunk_waits));
        reserve_in_cohort(&mut d.chunk_memory, bytes, seat, SeatState::Holding).await
    }

    fn start_probe(&self, d: &mut Disk) -> Result<()> {
        if d.chunk.is_empty() {
            // Nothing left to join.
            d.phase = Phase::Unmatched;
            return Ok(());
        }
        self.0.counters.chunks.fetch_add(1, Ordering::Relaxed);
        d.phase = Phase::Probe;
        Ok(())
    }

    /// After a driver's chunks: the last driver to get here goes on to
    /// emit the build rows of a build-side join (every match is in the
    /// bitmap by then); the others are done.
    fn finish_chunks(d: &mut Disk, ready: &Ready) -> FlushOutput {
        if ready.marks.is_none() {
            d.phase = Phase::Done;
            d.seat = None;
            return FlushOutput::Done;
        }
        let finished = ready.finished.fetch_add(1, Ordering::AcqRel) + 1;
        if finished < ready.drivers.load(Ordering::Acquire) {
            d.phase = Phase::Done;
            d.seat = None;
            return FlushOutput::Done;
        }
        d.build_reader = None;
        (d.next_file, d.next_base, d.emitted) = (0, 0, 0);
        d.phase = Phase::Emit;
        FlushOutput::Continue
    }

    /// One batch of the build rows a build-side join returns: the build's
    /// in-memory batches, then its files read again.
    async fn emit_build(
        &self,
        d: &mut Disk,
        ready: &Ready,
        ctx: &TaskContext,
    ) -> Result<FlushOutput> {
        let marks = ready
            .marks
            .as_ref()
            .ok_or_else(|| internal("nested-loop build rows without a bitmap".to_owned()))?;
        let emit = |out: Option<RecordBatch>| {
            out.map_or(FlushOutput::Continue, |b| FlushOutput::Batch(Batch::new(b)))
        };
        if let (Some(b), Some(&base)) = (
            ready.memory.get(d.emitted),
            ready.memory_bases.get(d.emitted),
        ) {
            d.emitted += 1;
            return Ok(emit(self.0.build_rows(b, base, marks)?));
        }
        let Some((file, base)) = ready.files.get(d.next_file).cloned() else {
            d.phase = Phase::Done;
            d.seat = None;
            return Ok(FlushOutput::Done);
        };
        if d.build_reader.is_none() {
            d.next_base = base;
        }
        let cohort = d.seat.as_mut().map(|s| (s, &self.0.counters.chunk_waits));
        let Some(b) = next_batch(ctx, &mut d.build_reader, &file, cohort).await? else {
            d.next_file += 1;
            return Ok(FlushOutput::Continue);
        };
        let base = d.next_base;
        d.next_base += b.num_rows() as u64;
        Ok(emit(self.0.build_rows(&b, base, marks)?))
    }
}

/// Whether a chunk may take `bytes` more past its first batch: a quarter of
/// the pool's limit stays free for ordinary reservations after it, so nested
/// loops (and everything else) sharing the limit keep room for their own
/// first batches and buffers. Each chunk's budget counts only its own
/// drivers, so without this the chunks of several instantiations together
/// fill the pool. [`MemoryPool::available`] already leaves out the
/// spill-scratch floor (and the resident-set guard's cut), so what stays
/// unreserved is a quarter of the limit plus the floor: under a 4 MiB limit
/// (floor 1/8) chunks stop growing once the pool is about 5/8 full. A chunk
/// cut short costs another pass over the probe files (`build_chunks_capped`).
///
/// [`MemoryPool::available`]: krishiv_memory::MemoryPool::available
fn leaves_room(ctx: &TaskContext, bytes: usize) -> bool {
    let pool = ctx.consumer().consumer().pool();
    match (pool.available(), pool.effective_limit()) {
        (Some(free), Some(limit)) => free.saturating_sub(bytes) >= limit / 4,
        _ => true,
    }
}

/// The bytes of build rows one driver's chunk reserves: a quarter of the
/// pool's limit shared by the drivers reading chunks, those of every
/// nested loop of the query in its cohort, else this one's (at least one
/// batch is always read). Taken again for every chunk, as loops come and go.
fn chunk_budget(ctx: &TaskContext, ready: &Ready, seat: Option<&Seat>) -> usize {
    let drivers = ready.drivers.load(Ordering::Acquire);
    let sharing = seat.map_or(drivers, |s| s.cohort().members().max(drivers));
    ctx.consumer()
        .consumer()
        .pool()
        .effective_limit()
        .map_or(usize::MAX, |l| (l / 4 / sharing.max(1)).max(1))
}

#[cfg(all(test, not(loom)))]
mod tests {
    #![expect(
        clippy::unwrap_used,
        clippy::disallowed_methods,
        reason = "tests: #[tokio::test] blocks on its own runtime"
    )]

    use std::time::Duration;

    use krishiv_common::error::ErrorClass;
    use krishiv_memory::{
        ConsumerId, MemoryConfig, MemoryPool, SpillSignal, Spillable as _, TreePool,
    };

    use super::*;

    const KIB: usize = 1 << 10;

    /// A query pool under a 256 KiB limit; a peer member of the chunk cohort
    /// holding 240 KiB as its chunk (seat holding); the seat and an empty
    /// reservation of another consumer of the query.
    fn setup() -> (Arc<dyn MemoryPool>, ConsumerId, Seat, Seat, Reservation) {
        let root = TreePool::new(&MemoryConfig {
            limit: Some(256 * KIB),
            quantum: 4 * KIB,
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        });
        let q = root.child("q", None);
        let cohort = q.cohort(CHUNK_COHORT).unwrap();
        let mut peer = cohort.join();
        peer.set(SeatState::Holding);
        let held = q.register("peer", None);
        q.try_reserve(held, 240 * KIB).unwrap();
        let id = q.register("chunk", None);
        let r = Reservation::new(Arc::clone(&q), id);
        (q, held, peer, cohort.join(), r)
    }

    /// While another member holds its chunk, a batch that does not fit waits
    /// (no failure, no abort) and is granted once the member frees its
    /// chunk and goes idle; the seat then holds.
    #[tokio::test]
    async fn a_batch_waits_for_an_active_member() {
        let (q, held, mut peer, mut seat, mut r) = setup();
        let waits = AtomicU64::new(0);
        {
            let mut ask = Box::pin(reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
            ));
            let early = tokio::time::timeout(Duration::from_millis(200), ask.as_mut()).await;
            assert!(early.is_err(), "granted while the peer held its chunk");
            // As `Disk::drop_chunk`: the memory goes, then the seat goes idle.
            q.release(held, 240 * KIB);
            peer.set(SeatState::Idle);
            tokio::time::timeout(Duration::from_secs(10), ask)
                .await
                .unwrap()
                .unwrap();
        }
        assert_eq!(seat.state(), SeatState::Holding);
        assert_eq!(r.size(), 32 * KIB);
        assert!(waits.load(Ordering::Relaxed) >= 1);
        assert!(q.check_aborted().is_ok());
        r.shrink(32 * KIB);
        q.unregister(held);
    }

    /// Two batches waiting on one holder do not wake each other: each
    /// waits a bounded number of times (backing off does not notify), and
    /// both are granted once the holder lets go.
    #[tokio::test]
    async fn two_waiters_do_not_wake_each_other() {
        let (q, held, mut peer, mut a, mut ra) = setup();
        let mut b = a.cohort().join();
        let mut rb = Reservation::new(Arc::clone(&q), q.register("chunk2", None));
        let (wa, wb) = (AtomicU64::new(0), AtomicU64::new(0));
        {
            let ask_a =
                reserve_in_cohort(&mut ra, 20 * KIB, Some((&mut a, &wa)), SeatState::Holding);
            let ask_b =
                reserve_in_cohort(&mut rb, 20 * KIB, Some((&mut b, &wb)), SeatState::Holding);
            let both = async { tokio::join!(ask_a, ask_b) };
            let mut both = Box::pin(both);
            let early = tokio::time::timeout(Duration::from_millis(300), both.as_mut()).await;
            assert!(early.is_err(), "granted while the peer held its chunk");
            assert!(
                wa.load(Ordering::Relaxed) <= 2 && wb.load(Ordering::Relaxed) <= 2,
                "waiters woke each other: {wa:?} {wb:?}"
            );
            q.release(held, 240 * KIB);
            peer.set(SeatState::Idle);
            let (x, y) = tokio::time::timeout(Duration::from_secs(10), both)
                .await
                .unwrap();
            x.unwrap();
            y.unwrap();
        }
        assert!(wa.load(Ordering::Relaxed) <= 3 && wb.load(Ordering::Relaxed) <= 3);
        assert_eq!(
            (a.state(), b.state()),
            (SeatState::Holding, SeatState::Holding)
        );
        ra.shrink(20 * KIB);
        rb.shrink(20 * KIB);
        q.unregister(held);
    }

    /// A request dropped while it waits leaves its seat idle.
    #[tokio::test]
    async fn a_dropped_request_leaves_its_seat_idle() {
        let (q, held, _peer, mut seat, mut r) = setup();
        let waits = AtomicU64::new(0);
        {
            let ask = reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
            );
            let early = tokio::time::timeout(Duration::from_millis(200), ask).await;
            assert!(early.is_err());
        }
        assert_eq!(seat.state(), SeatState::Idle);
        q.release(held, 240 * KIB);
        q.unregister(held);
    }

    /// A request dropped while it is still asking (its arbitration waits on
    /// a spill nobody services) leaves its seat idle and wakes the cohort:
    /// the seat would otherwise count as active for good.
    #[tokio::test]
    async fn a_request_dropped_while_asking_leaves_its_seat_idle() {
        let (q, held, mut peer, mut seat, mut r) = setup();
        // The peer's chunk is gone; a spillable consumer holds the memory
        // instead, and its spill is requested but never serviced.
        q.release(held, 240 * KIB);
        q.unregister(held);
        peer.set(SeatState::Idle);
        let signal = Arc::new(SpillSignal::new());
        signal.set_spillable_bytes(240 * KIB);
        let victim = q.register("victim", Some(Arc::clone(&signal) as _));
        q.try_reserve(victim, 240 * KIB).unwrap();
        let waits = AtomicU64::new(0);
        let woken = peer.cohort().changed();
        {
            let ask = reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
            );
            let early = tokio::time::timeout(Duration::from_millis(200), ask).await;
            assert!(early.is_err(), "the request did not wait on the spill");
        }
        assert!(signal.spill_requested().is_some(), "no spill was asked for");
        assert_eq!(waits.load(Ordering::Relaxed), 0, "it stood by instead");
        assert_eq!(seat.state(), SeatState::Idle);
        assert!(!peer.cohort().others_active(&peer), "the seat still counts");
        tokio::time::timeout(Duration::from_secs(1), woken)
            .await
            .expect("the drop did not wake the cohort");
        q.release(victim, 240 * KIB);
        q.unregister(victim);
    }

    /// With no other member active, nothing will free memory: the request
    /// runs the full arbitration as before and fails, and the seat is idle.
    #[tokio::test]
    async fn with_no_active_member_a_batch_fails_as_before() {
        let (q, held, mut peer, mut seat, mut r) = setup();
        peer.set(SeatState::Idle);
        let waits = AtomicU64::new(0);
        let e = tokio::time::timeout(
            Duration::from_secs(10),
            reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
        assert_eq!(seat.state(), SeatState::Idle);
        assert_eq!(waits.load(Ordering::Relaxed), 0);
        q.release(held, 240 * KIB);
        q.unregister(held);
    }
}
