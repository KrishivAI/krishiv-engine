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
//! in between.
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
use krishiv_memory::{ConsumerHandle, Reservation, SpillFile, SpillReader};
use krishiv_types::batch::Batch;

use super::stream::{Emit, copy_partition};
use super::{Local, Window, eval};
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
}

/// The spilled rows being read back, in order.
struct Cursor {
    files: VecDeque<SpillFile>,
    reader: Option<SpillReader>,
    consumer: Arc<ConsumerHandle>,
}

impl Cursor {
    /// The next batch. Blocking.
    fn next(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if let Some(r) = &mut self.reader {
                match r.next() {
                    Some(b) => return Ok(Some(b?.data().clone())),
                    None => self.reader = None,
                }
            }
            let Some(f) = self.files.pop_front() else {
                return Ok(None);
            };
            self.reader = Some(f.reader(self.consumer.reservation().releasing_to_credit())?);
        }
    }
}

/// The flush phase's state.
pub(super) struct Replay {
    /// `None` while a blocking step holds it.
    cursor: Option<Cursor>,
    /// A batch read and not taken yet.
    carry: Option<RecordBatch>,
    /// The driver was asked for memory for `carry`.
    asked: bool,
    /// A partition evaluated from disk.
    emit: Option<Box<Emit>>,
    /// Lent to blocking steps (see the [module docs](self)).
    headroom: Reservation,
    /// What `headroom` holds outside an external partition.
    base: usize,
}

/// Runs `f` on the blocking pool with `headroom` lent as grant credit;
/// takes back what the step left of it.
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
        l.prev_partition = None;
        l.prev_order = None;
        l.last_start = None;
        l.finished = false;
        l.replay = Some(Box::new(Replay {
            cursor: Some(Cursor {
                files,
                reader: None,
                consumer: Arc::clone(ctx.consumer().consumer()),
            }),
            carry: None,
            asked: false,
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
                let (cursor, next) = with_credit(ctx, &mut r.headroom, move || {
                    let next = cursor.next()?;
                    Ok((cursor, next))
                })
                .await?;
                r.cursor = Some(cursor);
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
        let scratch = ctx
            .consumer()
            .reservation()
            .releasing_to_credit()
            .for_spill_scratch();
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
        // Frames of three readers' batches (the trailing and lead readers',
        // and the lead's batch before), and one batch's evaluation.
        let rows = copied.disk.batch_rows;
        let need = copied
            .disk
            .file
            .max_read_reservation()
            .saturating_mul(3)
            .saturating_add(eval::cost(&self.env.plan, rows, 1, rows)?);
        let spare = r.headroom.size().saturating_sub(need.max(r.base));
        r.headroom.shrink(spare);
        if r.headroom.size() < need {
            r.headroom
                .grow(need - r.headroom.size())
                .await
                .map_err(krishiv_common::error::KrishivError::from)?;
        }
        let env = Arc::clone(&self.env);
        let consumer = Arc::clone(ctx.consumer().consumer());
        let query = Arc::clone(ctx.query());
        let disk = copied.disk;
        let emit = with_credit(ctx, &mut r.headroom, move || {
            Ok(Box::new(Emit::open(&env, disk, &consumer, query)?))
        })
        .await?;
        r.emit = Some(emit);
        Ok(())
    }
}
