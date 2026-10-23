// SPDX-License-Identifier: Apache-2.0
//! The in-memory final phase of row tables (P2-T159): radix partitions
//! merged in parallel.
//!
//! Every driver's [`RowTable`] holds its groups in radix partitions of the
//! seedless row hash `h`, each partition with its own rows and states,
//! with partial duplicates where its fixed index was cleared. The final
//! phase is "read partition `p` from every producer" (P2-T171 §5.1, I8):
//! one task per range of partitions ([`final_tasks`], sized from the merged
//! per-partition sketches) takes that range out of every driver table
//! ([`RowTable::split_parts`], with its share of the driver's reservation)
//! and merges it, in table order, into its own growing table, presized
//! from the sketches. Tasks share nothing: each owns its inputs, its table
//! and their reservations. They run on the blocking pool, at most the
//! executor's parallelism at once, each started once its memory is
//! reserved (as the join's partition builds are).
//!
//! **Memory.** A task frees each input partition as soon as it is merged,
//! so the phase holds about the unmerged inputs plus the merged tables,
//! never both in full.
//!
//! **Pressure.** With a spill directory the merge publishes what it holds
//! as spillable. When a task cannot get memory and none is running, or a
//! spill request (`spill_global`) arrives, the finished tasks' groups are
//! written to the output file first: they are final, so they are written
//! once and never merged again, and the merge goes on in memory (the
//! groups merged after that follow them to the file). Only when
//! no task is finished is everything left (every merge table and every
//! input row not merged yet) written as spill runs, for the disk phase.
//! Without a spill directory a task waits for its memory through
//! arbitration.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillWriter};

use super::super::rowtable::{Mode, RowTable, final_tasks, split_ref};
use super::super::table::Table;
use super::{
    CHUNK, Final, Global, HEADROOM, HashAggregate, Run, Spiller, bump, lend_headroom, reserve_to,
    shrink_to, tasks_of,
};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::sync::{Mutex, lock};

/// One final-phase task.
struct Task {
    parts: Range<usize>,
    table: RowTable,
    reservation: Reservation,
    /// Partitions `parts` of every driver table, in table order, each with
    /// its share of the driver's reservation.
    inputs: Vec<(RowTable, Reservation)>,
    /// The next row to merge: (input, partition, row).
    at: (usize, usize, usize),
    /// Bytes to reserve before the task can (re)start.
    need: usize,
    /// Groups to presize the index for on the first start (0 once done).
    presize: usize,
}

impl Task {
    /// Merges until done (`None`), or until the reservation cannot grow
    /// (`Some(missing bytes)`). Each input partition is freed once merged.
    fn run(&mut self, query: &QueryContext) -> Result<Option<usize>> {
        if self.presize > 0 {
            self.table.presize(std::mem::take(&mut self.presize));
        }
        let Self {
            parts,
            table,
            reservation,
            inputs,
            at,
            ..
        } = self;
        loop {
            query.check_cancelled()?;
            let (t, p, row) = *at;
            let Some((input, held)) = inputs.get_mut(t) else {
                shrink_to(reservation, table.memory_size());
                return Ok(None);
            };
            if p >= parts.end {
                *at = (t + 1, parts.start, 0);
                continue;
            }
            let len = input.part_len(p);
            if row >= len {
                input.release_part(p);
                shrink_to(held, input.memory_size());
                *at = (t, p + 1, 0);
                continue;
            }
            let end = (row + CHUNK).min(len);
            let chunk = input.merge_chunk(p, row..end)?;
            let want = table.projected_merge(&chunk);
            if let Err(missing) = reserve_to(reservation, want) {
                return Ok(Some(missing));
            }
            table.apply_merge(input, &chunk)?;
            *at = (t, p, end);
        }
    }

    /// Bytes the task holds.
    fn held(&self) -> usize {
        self.reservation.size() + self.inputs.iter().map(|(_, r)| r.size()).sum::<usize>()
    }
}

type Running<'a> = FuturesUnordered<
    std::pin::Pin<Box<dyn Future<Output = Result<(Task, Option<usize>)>> + Send + 'a>>,
>;

/// What the merge does when it lacks memory.
enum Relief {
    /// Write the finished tasks' groups to the output file and go on.
    Flush,
    /// Write everything left as spill runs; the disk phase takes over.
    Spill,
    /// Wait for the memory (no spill directory).
    Wait,
}

impl HashAggregate {
    /// Finalizes the combined row tables in memory, partitions in parallel.
    /// Returns `false` when nothing was done (spill runs exist: the disk
    /// phase merges everything) or when memory ran out and what was left
    /// was spilled; the step machine then continues with the disk phase.
    pub(super) async fn finalize_rows(&self, g: &Global, ctx: &TaskContext) -> Result<bool> {
        let (mut inputs, held) = {
            let mut inner = lock(&g.inner);
            if !inner.runs.is_empty() || inner.disk {
                return Ok(false);
            }
            let mut tables = std::mem::take(&mut inner.tables);
            // The default path's merge order: the largest table first.
            if let Some(big) = (0..tables.len()).max_by_key(|&i| tables[i].0.len()) {
                tables.swap(0, big);
            }
            inner.busy = true;
            let mut inputs = Vec::with_capacity(tables.len());
            let mut held = Vec::with_capacity(tables.len());
            for (t, r) in tables {
                match t {
                    Table::Rows(t) => inputs.push(t),
                    Table::Keys(_) => {
                        return Err(internal("a key table among row tables".to_owned()));
                    }
                }
                held.push(r);
            }
            (inputs, held)
        };
        // One table without partial duplicates is the result as it is.
        if let [only] = inputs.as_slice()
            && only.is_unique()
        {
            let table = inputs
                .pop()
                .ok_or_else(|| internal("no driver table".to_owned()))?;
            bump(&self.counters.groups, table.len() as u64);
            bump(&self.counters.hash_entries, table.len() as u64);
            bump(&self.counters.hash_buckets, table.buckets() as u64);
            let batch = ctx.query().config().batch_size.max(1);
            let chunks = part_chunks(&table, 0, batch);
            let mut inner = lock(&g.inner);
            inner.busy = false;
            let _ = g.done.set(Final::Parts {
                tables: vec![Table::Rows(table)],
                chunks,
                _held: held,
            });
            return Ok(true);
        }
        let spill = self.spiller(ctx);
        lock(&g.inner).row_merge = spill.is_some();
        let (mut queue, spare) = self.final_queue(g, ctx, inputs, held, spill.is_some())?;
        bump(&self.counters.row_final_tasks, queue.len() as u64);
        let publish = |queue: &VecDeque<Task>, done: &[Task], spare: &Option<Reservation>| {
            let bytes = if spill.is_some() {
                queue.iter().chain(done).map(Task::held).sum::<usize>()
                    + spare.as_ref().map_or(0, Reservation::size)
            } else {
                0
            };
            ctx.consumer().signal().set_spillable_bytes(bytes);
        };
        publish(&queue, &[], &spare);
        let limit = ctx.executor().parallelism().max(1);
        let mut running: Running<'_> = FuturesUnordered::new();
        let mut done: Vec<Task> = Vec::new();
        let mut out: Option<SpillWriter> = None;
        let mut spare = spare;
        loop {
            let asked = spill.is_some() && lock(&g.inner).spill_asked;
            let mut stalled = false;
            while !asked && running.len() < limit {
                let Some(mut task) = queue.pop_front() else {
                    break;
                };
                if task.need > 0 {
                    if task.reservation.try_grow(task.need).is_err() {
                        queue.push_front(task);
                        stalled = running.is_empty();
                        break;
                    }
                    task.need = 0;
                }
                let query = Arc::clone(ctx.query());
                running.push(Box::pin(ctx.run_blocking(move || {
                    let stall = task.run(&query)?;
                    Ok((task, stall))
                })));
            }
            if stalled || (asked && running.is_empty()) {
                let relief = match &spill {
                    None => Relief::Wait,
                    Some(_) if !done.is_empty() => Relief::Flush,
                    Some(_) if asked && queue.is_empty() => {
                        // Asked once everything was merged: nothing to free
                        // that the output does not hold.
                        lock(&g.inner).spill_asked = false;
                        continue;
                    }
                    Some(_) => Relief::Spill,
                };
                match (relief, &spill) {
                    (Relief::Flush, Some(s)) => {
                        lock(&g.inner).spill_asked = false;
                        out = Some(
                            self.flush_done(ctx, s, std::mem::take(&mut done), out, &mut spare)
                                .await?,
                        );
                        publish(&queue, &done, &spare);
                    }
                    (Relief::Spill, Some(s)) => {
                        self.spill_rows(g, ctx, s, queue, spare, out).await?;
                        return Ok(false);
                    }
                    _ => {
                        // No spill directory: wait for the memory.
                        let mut task = queue
                            .pop_front()
                            .ok_or_else(|| internal("a stalled merge without a task".to_owned()))?;
                        task.reservation
                            .grow(task.need)
                            .await
                            .map_err(KrishivError::from)?;
                        task.need = 0;
                        queue.push_front(task);
                    }
                }
                continue;
            }
            let Some(next) = running.next().await else {
                // Every task is done.
                break;
            };
            let (mut task, stall) = next?;
            match stall {
                None => done.push(task),
                Some(missing) => {
                    task.need = missing;
                    queue.push_back(task);
                }
            }
            publish(&queue, &done, &spare);
        }
        drop(running);
        // Groups already went to the output file (memory ran short): the
        // rest follow, so the result is read from disk with nothing else
        // held, as the disk phase's is.
        if let (Some(w), Some(s)) = (out, &spill) {
            let w = self
                .flush_done(ctx, s, std::mem::take(&mut done), Some(w), &mut spare)
                .await?;
            drop(spare);
            let s = s.clone();
            let file = ctx.run_blocking(move || s.finish(w)).await?;
            let mut inner = lock(&g.inner);
            inner.busy = false;
            inner.row_merge = false;
            inner.spill_asked = false;
            let _ = g.done.set(Final::File(Mutex::new(Some(file))));
            return Ok(true);
        }
        drop(spare);
        let batch = ctx.query().config().batch_size.max(1);
        let mut tables = Vec::with_capacity(done.len());
        let mut chunks = Vec::new();
        let mut reservations = Vec::with_capacity(done.len());
        let mut groups = 0;
        done.sort_by_key(|t| t.parts.start);
        for (i, task) in done.into_iter().enumerate() {
            let n = task.table.len();
            groups += n;
            bump(&self.counters.hash_buckets, task.table.buckets() as u64);
            chunks.extend(part_chunks(&task.table, i, batch));
            tables.push(Table::Rows(task.table));
            reservations.push(task.reservation);
        }
        bump(&self.counters.groups, groups as u64);
        bump(&self.counters.hash_entries, groups as u64);
        let mut inner = lock(&g.inner);
        inner.busy = false;
        inner.row_merge = false;
        inner.spill_asked = false;
        let _ = g.done.set(Final::Parts {
            tables,
            chunks,
            _held: reservations,
        });
        Ok(true)
    }

    /// The final phase's tasks: each takes its partitions out of every
    /// driver table with that share of the driver's reservation. What the
    /// drivers keep (their indexes) is freed, but for spill headroom kept
    /// once (`spill`).
    fn final_queue(
        &self,
        g: &Global,
        ctx: &TaskContext,
        mut inputs: Vec<RowTable>,
        held: Vec<Reservation>,
        spill: bool,
    ) -> Result<(VecDeque<Task>, Option<Reservation>)> {
        let plan = {
            let refs: Vec<&RowTable> = inputs.iter().collect();
            final_tasks(&refs)
        };
        let bits = g.row_bits;
        let ranges: Vec<Range<usize>> = plan.iter().map(|t| t.parts.clone()).collect();
        let mut per_task: Vec<Vec<(RowTable, Reservation)>> = ranges
            .iter()
            .map(|_| Vec::with_capacity(inputs.len()))
            .collect();
        let mut spare = None;
        for (t, mut r) in inputs.iter_mut().zip(held) {
            for (slices, part) in per_task.iter_mut().zip(t.split_parts(&ranges)) {
                let share = r.split(part.memory_size());
                slices.push((part, share));
            }
            if spill && spare.is_none() {
                shrink_to(&mut r, HEADROOM);
                spare = Some(r);
            }
        }
        drop(inputs);
        let mut queue = VecDeque::with_capacity(plan.len());
        for (t, inputs) in plan.into_iter().zip(per_task) {
            let table = RowTable::new(&self.core, &g.seed, Mode::Grow, false, bits)?;
            queue.push_back(Task {
                need: table.projected_presize(t.groups),
                presize: t.groups.max(1),
                at: (0, t.parts.start, 0),
                parts: t.parts,
                table,
                reservation: ctx.consumer().reservation(),
                inputs,
            });
        }
        Ok((queue, spare))
    }

    /// Writes the finished tasks' groups to the output file (opened on the
    /// first flush) and frees them.
    async fn flush_done(
        &self,
        ctx: &TaskContext,
        s: &Spiller,
        done: Vec<Task>,
        out: Option<SpillWriter>,
        spare: &mut Option<Reservation>,
    ) -> Result<SpillWriter> {
        bump(&self.counters.row_final_flushes, 1);
        let lent = spare.as_mut().map_or(0, lend_headroom);
        let s = s.clone();
        let out = ctx
            .run_blocking(move || {
                let mut out = match out {
                    Some(w) => w,
                    None => s
                        .scope
                        .writer(Arc::clone(&s.core.output_schema), s.consumer.reservation())
                        .map_err(KrishivError::from)?,
                };
                for task in done {
                    s.write_output(&Table::Rows(task.table), &mut out)?;
                }
                Ok(out)
            })
            .await?;
        if let Some(r) = spare.as_mut() {
            super::regrow(r, lent);
        }
        Ok(out)
    }

    /// Writes every task's table and every input row no task merged yet as
    /// spill runs (on the blocking pool), then hands the runs, and the
    /// output file of the groups already flushed, to the disk phase.
    async fn spill_rows(
        &self,
        g: &Global,
        ctx: &TaskContext,
        s: &Spiller,
        tasks: VecDeque<Task>,
        spare: Option<Reservation>,
        out: Option<SpillWriter>,
    ) -> Result<()> {
        bump(&self.counters.row_final_spills, 1);
        let s = s.clone();
        let mut spare = spare;
        let runs = ctx
            .run_blocking(move || {
                if let Some(r) = spare.as_mut() {
                    lend_headroom(r);
                }
                let mut runs: Vec<Run> = Vec::new();
                for task in tasks {
                    // Per key, merge order: what the task merged precedes
                    // every input row it did not reach.
                    let at = task.at;
                    let all = |_: u32| true;
                    let keeps: Vec<_> = (0..task.inputs.len())
                        .map(|i| {
                            move |r: u32| {
                                let (p, row) = split_ref(r);
                                (i, p, row) >= at
                            }
                        })
                        .collect();
                    let mut sources: Vec<(&RowTable, &dyn Fn(u32) -> bool)> =
                        vec![(&task.table, &all)];
                    for ((input, _), keep) in task.inputs.iter().zip(&keeps) {
                        sources.push((input, keep));
                    }
                    runs.push(s.write_run_sources(&sources, 0)?);
                }
                drop(spare);
                Ok(runs)
            })
            .await?;
        let mut inner = lock(&g.inner);
        inner.busy = false;
        inner.row_merge = false;
        inner.spill_asked = false;
        inner.disk = true;
        inner.out = out;
        tasks_of(runs, 0, &mut inner.tasks);
        Ok(())
    }
}

/// Output chunks of table `i`: positions of at most `batch` rows, each
/// within one radix partition (whose states are one partition's arrays).
fn part_chunks(table: &RowTable, i: usize, batch: usize) -> Vec<(usize, Range<usize>)> {
    let mut out = Vec::new();
    let mut start = 0;
    for p in 0..table.part_count() {
        let end = start + table.part_len(p);
        out.extend(
            (start..end)
                .step_by(batch)
                .map(|s| (i, s..(s + batch).min(end))),
        );
        start = end;
    }
    out
}
