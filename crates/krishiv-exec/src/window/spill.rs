// SPDX-License-Identifier: Apache-2.0
//! Spilling the window's buffered rows (P2-T055 part 2).
//!
//! # While the input runs
//! Under memory pressure the driver asks the window to spill
//! ([`Operator::spill`](crate::Operator::spill)): every buffered row not
//! emitted yet is written to a spill file, in input order, and its memory
//! freed. From the first spill on, rows are no longer evaluated as they
//! complete: they wait, in memory and in further spill files, so the
//! output keeps the input's order (rows evaluated before the spill are
//! emitted first; everything after it comes from the flush phase).
//!
//! # Flush phase
//! After the input ends ([`Operator::end_input`](crate::Operator::end_input)
//! writes the rows still buffered), the files are read back in order and
//! fed through the same buffer as the input: partitions that fit are
//! evaluated in memory, as without a spill. A partition that does not fit,
//! even after the driver tried to make room for it, is copied to a file of
//! its own and evaluated from there with bounded memory
//! ([`stream`](super::stream)). The spill-path ideas follow Spark's
//! `ExternalAppendOnlyUnsafeRowArray` (a window partition's rows spill and
//! are iterated back) and DuckDB's out-of-core window partitions.
//!
//! # Memory
//! The flush phase keeps a headroom reservation for its readers' frames
//! and writers' scratch; every blocking step lends it as the consumer's
//! grant credit and takes back what is left, so no other consumer takes it
//! in between. An external partition (one that does not fit, evaluated from
//! its own file) needs its readers' frames plus one step's evaluation, which
//! grows with the rows evaluated at once. When the pool refuses the growth
//! for a whole batch, the partition is evaluated in fewer rows per step, as
//! many as the headroom already held covers (`external_narrowed` in the
//! metrics): the query slows down instead of failing, and fails only when
//! one row's evaluation does not fit.
//!
//! # Failure and cancellation
//! Rows handed to a blocking step are owned by that step. Until it returns
//! them, the window's state is failed: a failed write or read, or a step
//! dropped because the query was cancelled, leaves every later call
//! failing with that error, never returning a short result (the discipline
//! of the finished-output spills in [`output_spill`](crate::output_spill)).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_memory::{ConsumerHandle, Reservation, SpillFile, SpillReader};
use krishiv_types::batch::Batch;

use super::stream::{Emit, copy_partition};
use super::{Local, Window};
use crate::context::TaskContext;
use crate::error::{Result, cancelled, internal};
use crate::join::{SPILL_SCRATCH, write_sliced};
use crate::operator::FlushOutput;

/// Spill counters (`EXPLAIN ANALYZE`).
#[derive(Default)]
pub(super) struct Counters {
    pub(super) spills: AtomicU64,
    pub(super) bytes: AtomicU64,
    pub(super) external: AtomicU64,
    /// Batches whose frames did not fit and folded from the file.
    pub(super) file_folds: AtomicU64,
    /// External partitions evaluated in fewer rows per step than a batch, because the headroom for a whole
    /// batch's evaluation was refused.
    pub(super) narrowed: AtomicU64,
}

/// The spilled rows being read back, in order.
struct Cursor {
    files: VecDeque<SpillFile>,
    reader: Option<SpillReader>,
    consumer: Arc<ConsumerHandle>,
    /// Frames consumed so far (batches returned, file footers passed): a refusal at the same count is the same
    /// frame refused again.
    progress: u64,
}

impl Cursor {
    /// The next batch. Blocking.
    fn next(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if let Some(r) = &mut self.reader {
                match r.next() {
                    Some(b) => {
                        let b = b?;
                        self.progress += 1;
                        return Ok(Some(b.data().clone()));
                    }
                    None => {
                        self.reader = None;
                        self.progress += 1;
                    }
                }
            }
            let Some(f) = self.files.pop_front() else {
                return Ok(None);
            };
            self.reader = Some(f.reader(self.consumer.reservation().releasing_to_credit())?);
        }
    }
}

/// The bytes a refused reservation asked for (0 if the error does not say).
fn refused_bytes(e: &KrishivError) -> usize {
    e.message_parameters()
        .iter()
        .find(|(name, _)| *name == "requestedBytes")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0)
}

/// What a failed read of the cursor's next frame turns into.
#[derive(Debug)]
pub(super) enum Refusal {
    /// Ask the driver for this many bytes and repeat the read.
    Ask(usize),
    /// The query fails with this error.
    Fail(KrishivError),
}

/// A read of the frame after `progress` frames failed with `e`; `asked` is the progress at which the driver was
/// last asked for memory. A memory refusal of a frame not yet asked for is asked for; the same frame refused again,
/// or any other error, fails.
pub(super) fn refusal(asked: &mut Option<u64>, progress: u64, e: KrishivError) -> Refusal {
    if e.error_class() != ErrorClass::UnableToAcquireMemory || *asked == Some(progress) {
        *asked = None;
        return Refusal::Fail(e);
    }
    *asked = Some(progress);
    Refusal::Ask(refused_bytes(&e))
}

/// The flush phase's state.
pub(super) struct Replay {
    /// `None` while a blocking step holds it.
    cursor: Option<Cursor>,
    /// A batch read and not taken yet.
    carry: Option<RecordBatch>,
    /// The driver was asked for memory for `carry`.
    asked: bool,
    /// The cursor's [`progress`](Cursor::progress) when the driver was asked for memory for its next frame, which
    /// the pool refused.
    read_asked: Option<u64>,
    /// A partition evaluated from disk.
    emit: Option<Box<Emit>>,
    /// Lent to blocking steps (see the [module docs](self)).
    headroom: Reservation,
    /// What `headroom` holds outside an external partition.
    base: usize,
}

/// Runs `f` on the blocking pool with `headroom` lent as grant credit;
/// takes back what the step left of it. The flush phase's batch read uses [`with_credit_keeping`] with `base`
/// instead: the reader frees its previous batch into the credit while it reads, and the headroom must take those
/// bytes back up to `base`, not only up to what it lent, or one frame larger than the last leaves it at 0 for good.
async fn with_credit<T, F>(ctx: &TaskContext, headroom: &mut Reservation, f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let lent = headroom.size();
    with_credit_keeping(ctx, headroom, lent, f).await
}

/// [`with_credit`], taking back up to `keep` bytes of the consumer's
/// credit afterwards (more than it lent when the step freed other bytes of
/// the consumer's into its credit).
async fn with_credit_keeping<T, F>(
    ctx: &TaskContext,
    headroom: &mut Reservation,
    keep: usize,
    f: F,
) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    headroom.shrink_to_credit(headroom.size());
    let out = ctx.run_blocking(f).await;
    let consumer = ctx.consumer().consumer();
    let back = consumer.credit().min(keep);
    if back > 0 {
        // Drawn from the credit: never refused.
        let _ = headroom.try_grow(back);
    }
    consumer.release_credit();
    out
}

impl Window {
    /// Writes the buffered rows to a spill file and frees their memory
    /// (see the [module docs](self)); returns the bytes freed. Without
    /// `must`, declines (frees nothing) when the writer's scratch cannot be
    /// had without waiting.
    pub(super) async fn spill_pending(
        &self,
        l: &mut Local,
        ctx: &TaskContext,
        must: bool,
    ) -> Result<usize> {
        l.check()?;
        let Some(scope) = &self.spill else {
            return Ok(0);
        };
        if l.replay.is_some() || l.pending.is_empty() {
            return Ok(0);
        }
        // The scratch the rows cannot fund comes from the pool now, or the
        // spill is declined rather than failing mid-write.
        // What the spill frees: a partly emitted front batch's bytes stay
        // reserved for its first rows waiting in `out` (the last of them
        // releases them).
        let shared = match (l.out.is_empty(), l.pending.front()) {
            (false, Some(p)) if p.offset > 0 => p.reserved,
            _ => 0,
        };
        let freed = l.pending_bytes() - shared;
        let mut topup = ctx.consumer().reservation().for_spill_scratch();
        let short = SPILL_SCRATCH.saturating_sub(freed);
        if must {
            topup
                .grow(short)
                .await
                .map_err(krishiv_common::error::KrishivError::from)?;
        } else if topup.try_grow(short).is_err() {
            return Ok(0);
        }
        topup.shrink_to_credit(topup.size());
        // From here until the freed credit is back in the pool, the rows
        // count as about to be released, so another driver's round that
        // cannot spill waits for them instead of failing. No arbitration
        // inside: the write is sync, the blocking job ends on its own.
        let writing = ctx.consumer().consumer().spilling();
        let mut batches = Vec::with_capacity(l.pending.len());
        for p in l.pending.drain(..) {
            let n = p.batch.num_rows();
            batches.push(Batch::new(p.batch.slice(p.offset, n - p.offset)));
        }
        if let Some((_, release)) = l.out.back_mut() {
            *release += shared;
        }
        l.last_start = None;
        // The bytes become the writer's scratch, as grant credit.
        l.reservation.shrink_to_credit(freed);
        l.failed = Some(cancelled());
        let scratch = ctx
            .consumer()
            .reservation()
            .releasing_to_credit()
            .for_spill_scratch();
        let (scope, schema) = (scope.clone(), Arc::clone(&self.env.input));
        let written = ctx
            .run_blocking(move || {
                let mut w = scope.writer(schema, scratch)?;
                for b in &batches {
                    write_sliced(&mut w, b)?;
                }
                drop(batches);
                Ok(w.finish()?)
            })
            .await;
        ctx.consumer().consumer().release_credit();
        drop(writing);
        let file = match written {
            Ok(f) => f,
            Err(e) => {
                l.failed = Some(e.clone());
                return Err(e);
            }
        };
        l.failed = None;
        self.counters.spills.fetch_add(1, Ordering::Relaxed);
        self.counters
            .bytes
            .fetch_add(file.bytes(), Ordering::Relaxed);
        l.spilled.get_or_insert_with(VecDeque::new).push_back(file);
        l.publish();
        Ok(freed)
    }

    /// The input ended: after a spill, writes the rows still buffered and
    /// starts the flush phase.
    pub(super) async fn start_replay(&self, l: &mut Local, ctx: &TaskContext) -> Result<()> {
        l.check()?;
        if l.spilled.is_none() {
            return Ok(());
        }
        self.spill_pending(l, ctx, true).await?;
        if !l.out.is_empty() {
            return Err(internal(
                "window output left before its flush phase".to_owned(),
            ));
        }
        let files = l.spilled.take().unwrap_or_default();
        // The files are this step's until the flush phase holds them: a
        // refused or cancelled reservation leaves the window failed, never
        // a flush phase that finds no rows.
        l.failed = Some(cancelled());
        let read = files
            .iter()
            .map(SpillFile::max_read_reservation)
            .max()
            .unwrap_or(0);
        let base = read + SPILL_SCRATCH;
        let mut headroom = ctx.consumer().reservation();
        if let Err(e) = headroom.grow(base).await {
            let e = krishiv_common::error::KrishivError::from(e);
            l.failed = Some(e.clone());
            return Err(e);
        }
        l.failed = None;
        // The rows are read back as a new input.
        l.started = false;
        l.open_len = 0;
        l.prev_partition = None;
        l.prev_order = None;
        l.last_start = None;
        l.finished = false;
        l.replay = Some(Box::new(Replay {
            cursor: Some(Cursor {
                files,
                reader: None,
                consumer: Arc::clone(ctx.consumer().consumer()),
                progress: 0,
            }),
            carry: None,
            asked: false,
            read_asked: None,
            emit: None,
            headroom,
            base,
        }));
        l.publish();
        Ok(())
    }

    /// One step of the flush phase.
    pub(super) async fn replay(&self, l: &mut Local, ctx: &TaskContext) -> Result<FlushOutput> {
        l.check()?;
        let Some(mut r) = l.replay.take() else {
            return Ok(FlushOutput::Done);
        };
        // Until the step gives the state back (see the module docs).
        l.failed = Some(cancelled());
        match self.replay_step(l, &mut r, ctx).await {
            Ok(out) => {
                l.failed = None;
                l.replay = Some(r);
                Ok(out)
            }
            Err(e) => {
                l.failed = Some(e.clone());
                Err(e)
            }
        }
    }

    async fn replay_step(
        &self,
        l: &mut Local,
        r: &mut Replay,
        ctx: &TaskContext,
    ) -> Result<FlushOutput> {
        if let Some(b) = Self::emit(l) {
            return Ok(FlushOutput::Batch(b));
        }
        if let Some(emit) = r.emit.take() {
            let env = Arc::clone(&self.env);
            let (emit, out) = with_credit(ctx, &mut r.headroom, move || {
                let mut emit = emit;
                let out = emit.step(&env)?;
                Ok((emit, out))
            })
            .await?;
            if let Some(b) = out {
                r.emit = Some(emit);
                return Ok(FlushOutput::Batch(Batch::new(b)));
            }
            // The partition is done: its readers' frames come back.
            drop(emit);
            let consumer = ctx.consumer().consumer();
            let back = consumer
                .credit()
                .min(r.base.saturating_sub(r.headroom.size()));
            if back > 0 {
                let _ = r.headroom.try_grow(back);
            }
            consumer.release_credit();
            let extra = r.headroom.size().saturating_sub(r.base);
            r.headroom.shrink(extra);
            return Ok(FlushOutput::Continue);
        }
        if l.finished {
            return Ok(FlushOutput::Done);
        }
        let batch = match r.carry.take() {
            Some(b) => b,
            None => {
                let mut cursor = r
                    .cursor
                    .take()
                    .ok_or_else(|| internal("window flush without its input".to_owned()))?;
                // The reader frees its previous batch's bytes into the credit while it reads the next one. The
                // headroom takes them back up to `base` (not only up to what it lent: once a read leaves it at 0,
                // that would hand every later read nothing, and the bytes would go to the pool, where buffered rows
                // fill the limit, so the next read of a batch found no memory).
                let keep = r.base;
                let (cursor, next) = with_credit_keeping(ctx, &mut r.headroom, keep, move || {
                    let next = cursor.next();
                    Ok((cursor, next))
                })
                .await?;
                r.cursor = Some(cursor);
                // A refused frame leaves the reader at the frame start (see `SpillReader`): the driver is asked
                // for the frame's bytes once and the read repeats; a second refusal of the same frame fails.
                let progress = r.cursor.as_ref().map_or(0, |c| c.progress);
                let next = match next {
                    Err(e) => match refusal(&mut r.read_asked, progress, e) {
                        Refusal::Ask(bytes) => {
                            return Ok(FlushOutput::NeedMemory {
                                bytes: bytes.max(r.base),
                            });
                        }
                        Refusal::Fail(e) => return Err(e),
                    },
                    ok => {
                        r.read_asked = None;
                        ok?
                    }
                };
                let Some(b) = next else {
                    l.finished = true;
                    self.flush_complete(l)?;
                    return Ok(FlushOutput::Continue);
                };
                b
            }
        };
        if batch.num_rows() == 0 {
            return Ok(FlushOutput::Continue);
        }
        match self.append(l, batch.clone(), false)? {
            Ok(()) => {
                r.asked = false;
                if Self::complete_rows(l) >= l.batch_rows {
                    self.flush_complete(l)?;
                }
                Ok(FlushOutput::Continue)
            }
            Err(bytes) => {
                r.carry = Some(batch);
                if Self::complete_rows(l) > 0 {
                    // Emitting complete partitions frees their memory.
                    self.flush_complete(l)?;
                    Ok(FlushOutput::Continue)
                } else if !l.out.is_empty() {
                    Ok(FlushOutput::Continue)
                } else if !r.asked {
                    r.asked = true;
                    Ok(FlushOutput::NeedMemory { bytes })
                } else {
                    r.asked = false;
                    self.external(l, r, ctx).await?;
                    Ok(FlushOutput::Continue)
                }
            }
        }
    }

    /// The buffered partition and `r.carry` do not fit: copies the
    /// partition to a file of its own and starts evaluating it from there.
    async fn external(&self, l: &mut Local, r: &mut Replay, ctx: &TaskContext) -> Result<()> {
        let mut readers = 3_usize;
        for f in &self.env.plan.functions {
            readers += super::disk::readers(f)?;
            // A frame over a growing state (`collect_list`, `mode`,
            // `listagg`) snapshots a copy per row, quadratic in the
            // partition, which no per-partition headroom covers.
            let grows = match &f.function {
                super::WindowFunction::Aggregate(a) => super::eval::snapshots_grow(a, f.shape()?),
                _ => false,
            };
            if grows {
                return Err(crate::aggregate::unsupported(format!(
                    "the window function {} ({:?}) over a partition larger than memory",
                    f.name, f.frame
                )));
            }
        }
        let carry = r
            .carry
            .take()
            .ok_or_else(|| internal("window partition to copy without its batch".to_owned()))?;
        let scope = self
            .spill
            .clone()
            .ok_or_else(|| internal("window flush phase without a spill scope".to_owned()))?;
        let head: Vec<RecordBatch> = l
            .pending
            .drain(..)
            .map(|p| p.batch.slice(p.offset, p.batch.num_rows() - p.offset))
            .collect();
        l.last_start = None;
        // The buffered rows' bytes become the writer's scratch.
        let held = l.reservation.size();
        l.reservation.shrink_to_credit(held);
        let mut cursor = r
            .cursor
            .take()
            .ok_or_else(|| internal("window flush without its input".to_owned()))?;
        let env = Arc::clone(&self.env);
        let query = Arc::clone(ctx.query());
        let consumer = Arc::clone(ctx.consumer().consumer());
        let scratch = ctx
            .consumer()
            .reservation()
            .releasing_to_credit()
            .for_spill_scratch();
        // The whole-partition folds open on the lent credit: reserve their
        // states on top of the rows the writer's scratch draws from, so a
        // state larger than a small partition is not left to the pool.
        let opening = super::disk::opened_whole(&self.env)?;
        if opening > 0 {
            r.headroom
                .grow(opening)
                .await
                .map_err(krishiv_common::error::KrishivError::from)?;
        }
        // The buffered rows' bytes stay with the headroom: the partition's
        // evaluation from disk needs them.
        let keep = r.headroom.size() + held;
        let (cursor, copied) = with_credit_keeping(ctx, &mut r.headroom, keep, move || {
            let mut carry = Some(carry);
            let writer = scope.writer(Arc::clone(&env.input), scratch)?;
            let copied = copy_partition(
                &env,
                head,
                || match carry.take() {
                    Some(b) => Ok(Some(b)),
                    None => cursor.next(),
                },
                writer,
                &consumer,
                &query,
            )?;
            Ok((cursor, copied))
        })
        .await?;
        r.cursor = Some(cursor);
        self.counters.external.fetch_add(1, Ordering::Relaxed);
        self.counters
            .bytes
            .fetch_add(copied.disk.file.bytes(), Ordering::Relaxed);
        // The rest starts the next partition.
        l.prev_partition = copied.key;
        l.prev_order = None;
        l.started = true;
        r.carry = copied.rest;
        // Frames of the readers' batches (the trailing and lead readers', the
        // lead's batch before, and each frame's or offset function's), and
        // one step's evaluation.
        let rows = copied.disk.batch_rows.max(1);
        let frames = copied
            .disk
            .file
            .max_read_reservation()
            .saturating_mul(readers);
        // What the functions hold as they open: offset functions' values,
        // folds' states.
        let frames = frames.saturating_add(super::disk::opened(
            &self.env,
            copied.disk.file.max_read_reservation(),
        )?);
        let need = |n: usize| -> Result<usize> {
            Ok(frames.saturating_add(super::disk::headroom(&self.env.plan, n)?))
        };
        let full = need(rows)?;
        let spare = r.headroom.size().saturating_sub(full.max(r.base));
        r.headroom.shrink(spare);
        let mut eval_rows = rows;
        if r.headroom.size() < full
            && r.headroom
                .grow_without_abort(full - r.headroom.size())
                .await
                .is_err()
        {
            // Refused (without aborting any query: this request has a
            // fallback, ADR 0007): evaluate fewer rows per step, as many as the headroom held covers (the evaluation's need
            // grows with its rows), so the query slows down instead of failing. Only when one row's evaluation
            // does not fit is the refusal the query's error.
            let held = r.headroom.size();
            let (mut lo, mut hi) = (1_usize, rows);
            if need(lo)? > held {
                r.headroom
                    .grow(need(lo)? - held)
                    .await
                    .map_err(krishiv_common::error::KrishivError::from)?;
            } else {
                while lo < hi {
                    let mid = lo + (hi - lo).div_ceil(2);
                    if need(mid)? <= held {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
            }
            eval_rows = lo;
            self.counters.narrowed.fetch_add(1, Ordering::Relaxed);
        }
        let env = Arc::clone(&self.env);
        let consumer = Arc::clone(ctx.consumer().consumer());
        let query = Arc::clone(ctx.query());
        let disk = copied.disk;
        let counters = Arc::clone(&self.counters);
        let emit = with_credit(ctx, &mut r.headroom, move || {
            Ok(Box::new(Emit::open(
                &env, disk, &consumer, counters, query, eval_rows,
            )?))
        })
        .await?;
        r.emit = Some(emit);
        Ok(())
    }
}
