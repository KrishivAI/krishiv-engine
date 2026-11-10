// SPDX-License-Identifier: Apache-2.0
//! The disk phase of a spilled aggregate: the partition tasks of the spill
//! runs merged in parallel (design `P2-agg-parallel-disk-merge`, option 1).
//!
//! Every hash partition of the runs is a *task*: its files merge, in run
//! order ([`Run::partition`], so order-sensitive aggregates merge as they
//! would one task at a time), into a table of its own, whose groups go to
//! an output file of its own. Tasks share nothing: each owns its input
//! files, its table and its reservation. They run on the blocking pool
//! from the finalizing driver, at most the executor's parallelism at once
//! (one under `krishiv.deterministicOrder`), as the row tables' in-memory
//! final phase does ([`super::parallel`]). The result is the list of
//! output files, so the output source has as many readers as files.
//! Groups an earlier phase already wrote to the output file stay the first
//! file.
//!
//! **Memory.** A task reserves each batch's growth before it applies it.
//! A task refused memory ends its job holding its table and waits:
//! - while other tasks run, it is retried each time one of them ends, and
//!   no new task starts while one waits;
//! - once none runs (it is alone), a waiting task that still does not fit
//!   is *split*: its table and the input it has not merged are
//!   re-partitioned by the next hash bits into tasks of the next level (up
//!   to [`MAX_LEVEL`]). A split runs alone, so tasks refused at once never
//!   mean as many re-partition rewrites at once.
//!
//! A spill request ([`Sink::spill_global`](crate::Sink::spill_global))
//! stops new tasks; once the running ones ended, a waiting task with all
//! its input merged runs if its write fits (writing frees its table, and a
//! split would rewrite it three times), else the largest waiting task is
//! split. Finished tasks hold no memory (their files are final).
//!
//! A lone task that cannot split (a global aggregate's single group, or
//! the hash bits used up) waits through arbitration with nothing of its
//! own published as spillable: only other consumers can free its memory,
//! and when none can the query fails with `UNABLE_TO_ACQUIRE_MEMORY`, as
//! the serial phase did.
//!
//! The groups' order across partitions is the tasks' key order, not the
//! order the serial phase produced; inside a partition it is unchanged.
//!
//! **Cancellation and errors.** The driver checks for cancellation between
//! task completions and each task between batches; a task's error fails
//! the query. A task's memory goes with its job, which the query waits for.

use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::Ordering as AtomicOrdering;

use arrow_array::UInt32Array;
use arrow_select::take::take_record_batch;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillError, SpillFile, SpillReader, SpillWriter};
use krishiv_types::batch::Batch;

use super::super::table::Table;
use super::{
    Final, Global, HEADROOM, HashAggregate, Output, Run, Spiller, bump, chunks, out_chunk, regrow,
    reserve_to, tasks_of,
};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::hash::{MAX_LEVEL, PARTITIONS};
use crate::sync::lock;

/// Where a task's output file goes in the result: its position among the
/// tasks of its level, after its parent's (a split's children follow one
/// another where the parent was). The empty key sorts first: the groups
/// written before the disk phase.
type Key = Vec<u32>;

/// One partition (or a split's sub-partition) of the disk phase.
struct DiskTask {
    key: Key,
    level: u32,
    table: Table,
    reservation: Reservation,
    /// Input files not opened yet, the next one last.
    files: Vec<SpillFile>,
    reader: Option<SpillReader>,
    /// The open file's largest read (frame and decoded batch).
    read_need: usize,
    /// A batch read but not applied (its memory was refused).
    pending: Option<Batch>,
    /// The output file being written, and the chunks written to it.
    emit: Option<(SpillWriter, usize)>,
    /// Bytes of the reservation lent to each spill read or write: the
    /// headroom, or more once an I/O was refused and that much granted.
    io: usize,
    /// Bytes the task was refused (0 when it runs).
    need: usize,
    /// Tests: a gate passed once the input is merged, before the output
    /// file is opened.
    #[cfg(test)]
    merged_gate: Option<Arc<crate::testing::BlockingGate>>,
}

/// How a task's job ended.
enum Outcome {
    /// Every input merged; its groups' output file (none for no groups).
    Done(Option<SpillFile>),
    /// Refused this many more bytes.
    Stalled(usize),
}

/// What a read gave.
enum Read {
    Batch(Batch),
    End,
    /// The pool refused the read's frame (retryable): this many bytes.
    Refused(usize),
}

impl DiskTask {
    fn new(
        key: Key,
        level: u32,
        table: Table,
        reservation: Reservation,
        files: Vec<SpillFile>,
    ) -> Self {
        Self {
            key,
            level,
            table,
            reservation,
            files,
            reader: None,
            read_need: 0,
            pending: None,
            emit: None,
            io: HEADROOM,
            need: 0,
            #[cfg(test)]
            merged_gate: None,
        }
    }

    /// Whether a split can free its memory: it has groups, is not yet
    /// emitting them, and the hash bits are not used up.
    fn splittable(&self) -> bool {
        self.emit.is_none() && self.level < MAX_LEVEL && !self.table.not_worth_spilling()
    }

    /// Whether all its input is merged: what is left is writing its groups,
    /// which frees its table.
    fn merged(&self) -> bool {
        self.files.is_empty() && self.reader.is_none() && self.pending.is_none()
    }

    /// Grants a waiting task the bytes it was refused, if the pool has
    /// them now (and, with `merged_only`, only a task with all its input
    /// merged); `need` is then 0, so the bytes are never asked twice.
    fn grant(&mut self, merged_only: bool) -> bool {
        let granted =
            (!merged_only || self.merged()) && self.reservation.try_grow(self.need).is_ok();
        if granted {
            self.need = 0;
        }
        granted
    }

    /// Merges and then writes its groups until done, or until memory is
    /// refused (blocking I/O). A refused write or read is retried as it
    /// was once the memory is granted.
    fn run(&mut self, s: &Spiller) -> Result<Outcome> {
        if self.emit.is_none() {
            loop {
                s.query.check_cancelled()?;
                let batch = match self.pending.take() {
                    Some(b) => b,
                    None => match self.next_batch(s)? {
                        Read::Batch(b) => b,
                        Read::End => break,
                        Read::Refused(need) => return Ok(Outcome::Stalled(need)),
                    },
                };
                let p = self.table.prepare(&batch, true)?;
                let want = self.table.projected(&p)? + self.io.max(self.table.transient(&p));
                if let Err(missing) = reserve_to(&mut self.reservation, want) {
                    self.pending = Some(batch);
                    return Ok(Outcome::Stalled(missing));
                }
                self.table.apply(&p)?;
            }
            if self.table.len() == 0 {
                return Ok(Outcome::Done(None));
            }
            // Tests: the task is merged and holds its table, not yet
            // writing.
            #[cfg(test)]
            if let Some(g) = &self.merged_gate {
                g.pass();
            }
            let schema = Arc::clone(&s.core.output_schema);
            let w = s.scope.writer(schema, s.consumer.reservation())?;
            self.emit = Some((w, 0));
        }
        self.write_groups(s)
    }

    /// The next input batch, files in run order; spill I/O borrows `io`
    /// bytes of the reservation.
    fn next_batch(&mut self, s: &Spiller) -> Result<Read> {
        loop {
            if self.reader.is_none() {
                let Some(f) = self.files.pop() else {
                    return Ok(Read::End);
                };
                self.read_need = f.max_read_reservation();
                self.reader = Some(f.reader(s.consumer.reservation())?);
            }
            let Some(reader) = self.reader.as_mut() else {
                return Ok(Read::End);
            };
            let lent = lend(&mut self.reservation, self.io);
            let next = reader.next().transpose();
            regrow(&mut self.reservation, lent);
            match next {
                Ok(Some(b)) => return Ok(Read::Batch(b)),
                // That file is read: the next one.
                Ok(None) => self.reader = None,
                Err(SpillError::Memory(_)) => {
                    self.io = self.io.max(self.read_need);
                    return Ok(Read::Refused(self.read_need));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Writes the table's groups to its output file, from the first chunk
    /// not written yet, and finishes the file.
    fn write_groups(&mut self, s: &Spiller) -> Result<Outcome> {
        let Self {
            table,
            reservation,
            emit,
            io,
            ..
        } = self;
        let Some((w, written)) = emit.as_mut() else {
            return Err(internal("disk task output without a writer".to_owned()));
        };
        let ids = table.ids(0..table.len());
        let size = out_chunk(table.memory_size(), table.len(), s.batch_size);
        for chunk in chunks(&ids, size, table).skip(*written) {
            s.query.check_cancelled()?;
            let b = Batch::new(table.output_batch(chunk)?);
            let lent = lend(reservation, *io);
            let r = w.write(&b);
            regrow(reservation, lent);
            match r {
                Ok(()) => *written += 1,
                Err(SpillError::Memory(_)) => {
                    // About what the write reserves: its encoding scratch
                    // and buffer.
                    let need = 3 * SpillWriter::frame_bytes(&b) + (64 << 10);
                    *io = (*io).max(need);
                    // The unfinished file goes (a writer dropped unfinished
                    // discards it): the task holds only its merged table
                    // again, so it can be split to free memory while it
                    // waits, and writes its groups afresh once it runs.
                    *emit = None;
                    return Ok(Outcome::Stalled(need));
                }
                Err(e) => return Err(e.into()),
            }
        }
        let Some((w, _)) = emit.take() else {
            return Err(internal("disk task output without a writer".to_owned()));
        };
        let n = table.len() as u64;
        bump(&s.counters.groups, n);
        bump(&s.counters.hash_entries, n);
        bump(&s.counters.hash_buckets, table.buckets() as u64);
        s.finish(w).map(|f| Outcome::Done(Some(f)))
    }

    /// Re-partitions the task by the next hash bits (blocking I/O): its
    /// table first, then the input it has not merged, in run order, as two
    /// runs of the next level. `fresh` is an empty table of the same
    /// layout (it computes the partitions).
    fn split(self, s: &Spiller, fresh: &Table) -> Result<Vec<Run>> {
        let level = self.level + 1;
        let Self {
            table,
            mut reservation,
            mut files,
            mut reader,
            pending,
            ..
        } = self;
        // As credit (review m-a): other operators of the query share the
        // pool, and the run's writer must not lose the headroom to them.
        // What it leaves goes back with the phase's `release_credit`.
        lend(&mut reservation, HEADROOM);
        let first = s.write_run(&table, 0, level, u32::MAX, &mut reservation)?;
        drop((table, reservation));
        let mut writers: Vec<Option<SpillWriter>> = (0..PARTITIONS).map(|_| None).collect();
        let mut write = |batch: &Batch| -> Result<()> {
            let parts = fresh.partitions_of(batch.data(), level)?;
            for (rows, slot) in parts.into_iter().zip(writers.iter_mut()) {
                if rows.is_empty() {
                    continue;
                }
                let b = take_record_batch(batch.data(), &UInt32Array::from(rows))
                    .map_err(|e| internal(e.to_string()))?;
                let w = match slot {
                    Some(w) => w,
                    slot => slot.insert(s.run_writer()?),
                };
                w.write(&Batch::new(b)).map_err(KrishivError::from)?;
            }
            Ok(())
        };
        if let Some(b) = &pending {
            write(b)?;
        }
        loop {
            s.query.check_cancelled()?;
            match next_of(&mut reader, &mut files, s)? {
                Some(b) => write(&b)?,
                None if reader.take().is_some() => {}
                None => break,
            }
        }
        let mut rest = Vec::with_capacity(PARTITIONS);
        for w in writers {
            rest.push(w.map(|w| s.finish(w)).transpose()?);
        }
        Ok(vec![first, Run::new(rest, u32::MAX)])
    }
}

/// The next batch of `reader` (a split's input), opening the next of
/// `files` (the last) when none is open; `None` at the end of the open
/// file, or when there is no file left (`reader` is then `None`).
fn next_of(
    reader: &mut Option<SpillReader>,
    files: &mut Vec<SpillFile>,
    s: &Spiller,
) -> Result<Option<Batch>, SpillError> {
    if reader.is_none() {
        let Some(f) = files.pop() else {
            return Ok(None);
        };
        *reader = Some(f.reader(s.consumer.reservation())?);
    }
    match reader.as_mut() {
        Some(r) => r.next().transpose(),
        None => Ok(None),
    }
}

/// Lends up to `bytes` of `r` to spill I/O of the same consumer as credit
/// (not through the pool, as `lend_headroom` does for callers that never
/// `release_credit`). The bytes go to the consumer's credit, not back to
/// the pool, so the I/O's own reservation takes them without racing other
/// consumers for them (review m3); a sibling task of the same consumer may
/// still take them first, and [`regrow`] draws back what is left.
fn lend(r: &mut Reservation, bytes: usize) -> usize {
    let lent = bytes.min(r.size());
    r.shrink_to_credit(lent);
    lent
}

/// A finished disk job.
enum Ended {
    Task(Box<DiskTask>, Outcome),
    /// A split of the task at `Key`: runs of the next level.
    Split(Key, u32, Vec<Run>),
}

type Running<'a> =
    FuturesUnordered<std::pin::Pin<Box<dyn Future<Output = Result<Ended>> + Send + 'a>>>;

/// Tests: what a split adds to the count of disk jobs on the blocking pool
/// (a task adds one).
#[cfg(test)]
const SPLIT: usize = 1 << 16;

/// Tests: counts a disk job on the blocking pool while it lives, and
/// records a split started beside another job.
#[cfg(test)]
struct Active<'a>(&'a super::Counters, usize);

#[cfg(test)]
impl<'a> Active<'a> {
    fn enter(c: &'a super::Counters, split: bool) -> Self {
        let weight = if split { SPLIT } else { 1 };
        let before = c.disk_active.fetch_add(weight, AtomicOrdering::AcqRel);
        if (split && before != 0) || (!split && before >= SPLIT) {
            bump(&c.disk_split_overlaps, 1);
        }
        Self(c, weight)
    }
}

#[cfg(test)]
impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.disk_active.fetch_sub(self.1, AtomicOrdering::AcqRel);
    }
}

impl HashAggregate {
    /// Runs the disk phase (the spill runs' partition tasks in
    /// `g.inner.tasks`, and the output file already started, if any) and
    /// publishes its output files.
    pub(super) async fn finalize_disk(&self, g: &Global, ctx: &TaskContext) -> Result<()> {
        let s = self
            .spiller(ctx)
            .ok_or_else(|| internal("disk phase without spill".to_owned()))?;
        let (tasks, out) = {
            let mut inner = lock(&g.inner);
            inner.disk_merge = true;
            (std::mem::take(&mut inner.tasks), inner.out.take())
        };
        let result = self.run_disk(g, ctx, &s, tasks, out).await;
        let mut inner = lock(&g.inner);
        inner.disk_merge = false;
        inner.spill_asked = false;
        drop(inner);
        ctx.consumer().signal().set_spillable_bytes(0);
        // Bytes `lend` left as credit go back to the pool.
        ctx.consumer().consumer().release_credit();
        let files = result?;
        let chunks = files.len().max(1);
        let out = Output::on_disk(files, Arc::clone(ctx.executor()));
        let _ = g.done.set(Final {
            out: Arc::new(out),
            chunks,
        });
        Ok(())
    }

    /// The task loop of [`finalize_disk`](Self::finalize_disk); returns the
    /// output files in result order.
    async fn run_disk(
        &self,
        g: &Global,
        ctx: &TaskContext,
        s: &Spiller,
        tasks: Vec<(u32, Vec<SpillFile>)>,
        out: Option<SpillWriter>,
    ) -> Result<Vec<SpillFile>> {
        let mut files: Vec<(Key, SpillFile)> = Vec::new();
        if let Some(w) = out {
            // Groups written before the disk phase: the first file.
            let s = s.clone();
            files.push((Key::new(), ctx.run_blocking(move || s.finish(w)).await?));
        }
        let mut queue: VecDeque<(Key, u32, Vec<SpillFile>)> = VecDeque::new();
        push_tasks(&mut queue, &[], tasks)?;
        let limit = if ctx.query().config().deterministic_order {
            1
        } else {
            ctx.executor().parallelism().max(1)
        };
        // Tasks refused memory, in the order they were refused.
        let mut waiting: VecDeque<DiskTask> = VecDeque::new();
        let mut running: Running<'_> = FuturesUnordered::new();
        let mut splitting = false;
        loop {
            ctx.query().check_cancelled()?;
            let asked = lock(&g.inner).spill_asked;
            if !splitting && !asked {
                // Waiting tasks first, whenever their memory is there.
                let mut i = 0;
                while running.len() < limit && i < waiting.len() {
                    let fits = waiting.get_mut(i).is_some_and(|t| t.grant(false));
                    if !fits {
                        i += 1;
                        continue;
                    }
                    if let Some(t) = waiting.remove(i) {
                        running.push(self.launch(ctx, s, t));
                    }
                }
                // New tasks only while none waits for memory, and beside a
                // running one only with room to spare: a task refused memory
                // with every other one waiting can only split, and a split
                // needs scratch beyond its own headroom, which the other
                // tasks' tables must not have taken. Under a small limit the
                // phase stays serial, as it was.
                while running.len() < limit
                    && waiting.is_empty()
                    && (running.is_empty() || roomy(ctx, running.len()))
                {
                    let Some((key, level, files)) = queue.pop_front() else {
                        break;
                    };
                    bump(&self.counters.disk_tasks, 1);
                    let table = self.new_table(g, false)?;
                    let t = DiskTask::new(key, level, table, ctx.consumer().reservation(), files);
                    running.push(self.launch(ctx, s, t));
                }
                self.counters
                    .disk_peak
                    .fetch_max(running.len() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            if running.is_empty() {
                if asked {
                    lock(&g.inner).spill_asked = false;
                    // A merged task frees its table by writing it: run it
                    // when its write fits, rather than split (rewrite) it
                    // (review M2). Without a request the retry pass above
                    // already tried every waiting task; with one it was
                    // skipped, and the tasks not run here are retried next
                    // round, beside this write.
                    if let Some(i) = waiting.iter_mut().position(|t| t.grant(true))
                        && let Some(t) = waiting.remove(i)
                    {
                        #[cfg(test)]
                        bump(&self.counters.disk_merged_runs, 1);
                        running.push(self.launch(ctx, s, t));
                    }
                }
                if !running.is_empty() {
                    // Launched above: wait for it below.
                } else if let Some(mut t) = take_largest(&mut waiting) {
                    if t.splittable() {
                        // Alone and still refused (or asked to free
                        // memory): split it, alone too.
                        bump(&self.counters.repartitions, 1);
                        splitting = true;
                        let fresh = self.new_table(g, false)?;
                        running.push(self.launch_split(ctx, s, t, fresh));
                    } else {
                        // Nothing to split (a global aggregate's single
                        // group, or the hash bits used up): wait for the
                        // memory through arbitration. Nothing of this consumer can be
                        // freed meanwhile (every waiting task is
                        // unsplittable, or it would have been taken), so it
                        // publishes nothing as spillable (review m1).
                        ctx.consumer().signal().set_spillable_bytes(0);
                        let grown = t.reservation.grow(t.need).await;
                        // A request taken while it waited asked this
                        // consumer for nothing it could free.
                        lock(&g.inner).spill_asked = false;
                        grown.map_err(KrishivError::from)?;
                        t.need = 0;
                        running.push(self.launch(ctx, s, t));
                    }
                } else if queue.is_empty() {
                    break;
                } else {
                    continue;
                }
            }
            ctx.consumer()
                .signal()
                .set_spillable_bytes(waiting.iter().map(|t| t.reservation.size()).sum());
            let Some(ended) = running.next().await else {
                continue;
            };
            match ended? {
                Ended::Task(t, Outcome::Done(file)) => {
                    if let Some(f) = file {
                        files.push((t.key.clone(), f));
                    }
                }
                Ended::Task(t, Outcome::Stalled(need)) => {
                    let mut t = *t;
                    t.need = need;
                    if !running.is_empty() {
                        bump(&self.counters.disk_stalls, 1);
                    }
                    waiting.push_back(t);
                }
                Ended::Split(key, level, runs) => {
                    splitting = false;
                    let mut children = Vec::new();
                    tasks_of(runs, level, &mut children);
                    // The children run next, in their order.
                    let mut next = VecDeque::new();
                    push_tasks(&mut next, &key, children)?;
                    next.extend(queue);
                    queue = next;
                }
            }
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(files.into_iter().map(|(_, f)| f).collect())
    }

    /// Runs `t` on the blocking pool until it ends or is refused memory.
    fn launch<'a>(
        &'a self,
        ctx: &'a TaskContext,
        s: &Spiller,
        mut t: DiskTask,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Ended>> + Send + 'a>> {
        let s = s.clone();
        #[cfg(test)]
        let (gate, counters) = (self.disk_gate.clone(), Arc::clone(&self.counters));
        #[cfg(test)]
        {
            t.merged_gate.clone_from(&self.disk_merged_gate);
        }
        Box::pin(async move {
            let (t, outcome) = ctx
                .run_blocking(move || {
                    #[cfg(test)]
                    let _active = Active::enter(&counters, false);
                    let outcome = t.run(&s)?;
                    // Tests: the task holds its table (and output file).
                    #[cfg(test)]
                    if let Some(g) = &gate {
                        g.pass();
                    }
                    Ok((t, outcome))
                })
                .await?;
            Ok(Ended::Task(Box::new(t), outcome))
        })
    }

    /// Splits `t` on the blocking pool (see [`DiskTask::split`]).
    fn launch_split<'a>(
        &'a self,
        ctx: &'a TaskContext,
        s: &Spiller,
        t: DiskTask,
        fresh: Table,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Ended>> + Send + 'a>> {
        let s = s.clone();
        let (key, level) = (t.key.clone(), t.level + 1);
        #[cfg(test)]
        let counters = Arc::clone(&self.counters);
        Box::pin(async move {
            // A split writes the task's table out and drops it, then reads
            // the task's runs back and rewrites them by the next hash bits:
            // until it ends, the consumer's memory (the waiting tasks' tables
            // too) counts as about to be released, so another driver's round
            // that cannot spill waits for it instead of failing. The job ends
            // on its own (its reads and writes never arbitrate).
            // A merge task takes none: it grows its table, and one refused
            // memory keeps it while it waits.
            let _writing = ctx.consumer().consumer().spilling();
            let runs = ctx
                .run_blocking(move || {
                    #[cfg(test)]
                    let _active = Active::enter(&counters, true);
                    t.split(&s, &fresh)
                })
                .await?;
            Ok(Ended::Split(key, level, runs))
        })
    }
}

/// Appends `tasks` (from [`tasks_of`]) to `queue`, keyed after `parent`;
/// their files are reversed so they pop in run order.
fn push_tasks(
    queue: &mut VecDeque<(Key, u32, Vec<SpillFile>)>,
    parent: &[u32],
    tasks: Vec<(u32, Vec<SpillFile>)>,
) -> Result<()> {
    for (i, (level, mut files)) in tasks.into_iter().enumerate() {
        let i = u32::try_from(i).map_err(|_| internal("too many disk tasks".to_owned()))?;
        files.reverse();
        let mut key = parent.to_vec();
        key.push(i);
        queue.push_back((key, level, files));
    }
    Ok(())
}

/// Whether the query's pool has room for another disk task beside the
/// `running` ones: an eighth of its limit, and at least four spill
/// headrooms (a split's scratch), free once each running task is counted
/// at four headrooms too (a task reserves only as it merges, so a task
/// just started holds nothing yet); always without a limit.
fn roomy(ctx: &TaskContext, running: usize) -> bool {
    let pool = ctx.query().pool();
    pool.effective_limit().is_none_or(|limit| {
        let free = limit
            .saturating_sub(pool.reserved())
            .saturating_sub(running.saturating_mul(4 * HEADROOM));
        free >= (limit / 8).max(4 * HEADROOM)
    })
}

/// Takes the waiting task to free memory from: one that can split,
/// holding the most memory (splitting it frees the most).
fn take_largest(waiting: &mut VecDeque<DiskTask>) -> Option<DiskTask> {
    let at = (0..waiting.len()).max_by_key(|&i| {
        waiting
            .get(i)
            .map_or((false, 0), |t| (t.splittable(), t.reservation.size()))
    })?;
    waiting.remove(at)
}
