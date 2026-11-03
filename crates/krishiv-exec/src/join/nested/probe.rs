// SPDX-License-Identifier: Apache-2.0
//! The nested-loop join's probe operator: streams its input against a build
//! held in memory, or, once the build spilled, buffers and spills its input
//! and joins it build chunk by build chunk in the flush phase (see the
//! [module docs](super)).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::RecordBatch;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillError, SpillFile, SpillReader, SpillSignal};
use krishiv_types::batch::Batch;

use super::build::{Ready, ready, write};
use super::{BuildRows, Pairing, ProbeBits, Rest, Shared, unit_weights};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{FlushOutput, LocalState, Operator, OperatorOutput};
use crate::operators::local_mut;

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
    }
}

/// The next batch of the reader in `slot` (opened on `file` when empty), read
/// on the blocking pool. A frame the pool refuses is waited for and read
/// again. `None` at the end of the file, which closes the reader.
async fn next_batch(
    ctx: &TaskContext,
    slot: &mut Option<SpillReader>,
    file: &SpillFile,
) -> Result<Option<RecordBatch>> {
    loop {
        let reader = match slot.take() {
            Some(r) => r,
            None => file
                .reader(ctx.consumer().reservation())
                .map_err(KrishivError::from)?,
        };
        let (reader, item) = ctx
            .run_blocking(move || {
                let mut reader = reader;
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
                topup.grow(bytes).await.map_err(KrishivError::from)?;
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
            // The build spilled: the rows wait for the flush phase.
            let bytes = input.data().get_array_memory_size();
            if d.reservation.try_grow(bytes).is_err() {
                return Ok(OperatorOutput::NeedMemory { bytes });
            }
            d.buffered.push(input.data().clone());
            d.signal.set_spillable_bytes(d.reservation.size());
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let build = BuildRows {
            batches: &l.ready.memory,
            bases: &l.ready.memory_bases,
        };
        loop {
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
            let file = match write(ctx, scope, schema, batches, &mut d.reservation).await {
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
                let file = write(ctx, scope, schema, batches, &mut d.reservation).await?;
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
                d.budget = chunk_budget(ctx, ready.drivers.load(Ordering::Acquire));
                d.phase = if d.rows == 0 {
                    Phase::Build
                } else {
                    Phase::Load
                };
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
                        (d.probe_file, d.probe_base) = (0, 0);
                        d.phase = if d.build_read(&ready) {
                            Phase::Unmatched
                        } else {
                            Phase::Load
                        };
                        return Ok(FlushOutput::Continue);
                    };
                    match next_batch(ctx, &mut d.probe_reader, &file).await? {
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
                let Some(b) = next_batch(ctx, &mut d.probe_reader, &file).await? else {
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
            d.chunk_memory
                .grow(b.get_array_memory_size())
                .await
                .map_err(KrishivError::from)?;
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
        let Some(b) = next_batch(ctx, &mut d.build_reader, &file).await? else {
            d.next_file += 1;
            if d.next_file >= ready.files.len() {
                return self.start_probe(d);
            }
            return Ok(());
        };
        let (bytes, base) = (b.get_array_memory_size(), d.next_base);
        d.next_base += b.num_rows() as u64;
        if d.chunk.is_empty() {
            d.chunk_memory
                .grow(bytes)
                .await
                .map_err(KrishivError::from)?;
        } else if d.chunk_memory.try_grow(bytes).is_err() {
            d.pending = Some((b, base));
            return self.start_probe(d);
        }
        d.chunk.push(b);
        d.chunk_bases.push(base);
        Ok(())
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
            return FlushOutput::Done;
        }
        let finished = ready.finished.fetch_add(1, Ordering::AcqRel) + 1;
        if finished < ready.drivers.load(Ordering::Acquire) {
            d.phase = Phase::Done;
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
            return Ok(FlushOutput::Done);
        };
        if d.build_reader.is_none() {
            d.next_base = base;
        }
        let Some(b) = next_batch(ctx, &mut d.build_reader, &file).await? else {
            d.next_file += 1;
            return Ok(FlushOutput::Continue);
        };
        let base = d.next_base;
        d.next_base += b.num_rows() as u64;
        Ok(emit(self.0.build_rows(&b, base, marks)?))
    }
}

/// The bytes of build rows one driver's chunk reserves: a quarter of the
/// pool's limit shared by the drivers (at least one batch is always read).
fn chunk_budget(ctx: &TaskContext, drivers: usize) -> usize {
    ctx.consumer()
        .consumer()
        .pool()
        .effective_limit()
        .map_or(usize::MAX, |l| (l / 4 / drivers.max(1)).max(1))
}
