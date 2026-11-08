// SPDX-License-Identifier: Apache-2.0
//! The in-memory final phase of row tables (P2-T159): radix partitions
//! merged in parallel.
//!
//! Every driver's [`RowTable`] holds its groups in radix partitions of the
//! seedless row hash `h`, each partition with its own rows and states,
//! with partial duplicates where its fixed index was cleared. The final
//! phase is "read partition `p` from every producer" (P2-T171 §5.1, I8).
//! Driver tables that stayed narrow (one partition) while another widened
//! are widened first, so all share the radix partitions; when none
//! widened, or a widening's copy cannot be reserved and there is no spill
//! directory, one task with one partition merges them all. Then one task per range of
//! partitions ([`final_tasks`], sized from the merged per-partition
//! sketches) takes that range out of every driver table
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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillError, SpillFile, SpillWriter};

use super::super::is_order_sensitive;
use super::super::rowtable::{FinalTask, Mode, RowTable, final_tasks};
use super::super::table::Table;
use super::parts::{Helper, MAX_WRITERS};
use super::{
    CHUNK, Final, Global, HEADROOM, HashAggregate, Output, Run, Spiller, bump, lend_headroom,
    reserve_to, shrink_to, tasks_of,
};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::hash::PARTITIONS;
use crate::sync::lock;

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
            self.table.presize(std::mem::take(&mut self.presize))?;
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
            let buckets = table.buckets();
            table.apply_merge(input, &chunk)?;
            if table.buckets() != buckets {
                // The peak of a growth was reserved: the old index is gone.
                shrink_to(reservation, table.memory_size());
            }
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
        let (mut inputs, mut held) = {
            let mut inner = lock(&g.inner);
            if !inner.runs.is_empty() || inner.disk {
                return Ok(false);
            }
            let mut tables = std::mem::take(&mut inner.tables);
            // The default path's merge order: the largest table first,
            // unless an order-sensitive aggregate needs partition order.
            let ordered = self
                .core
                .spec
                .aggregates
                .iter()
                .any(|a| is_order_sensitive(&a.function));
            if !ordered && let Some(big) = (0..tables.len()).max_by_key(|&i| tables[i].1.len()) {
                tables.swap(0, big);
            }
            inner.busy = true;
            let mut inputs = Vec::with_capacity(tables.len());
            let mut held = Vec::with_capacity(tables.len());
            for (_, t, r) in tables {
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
            lock(&g.inner).busy = false;
            self.finish_memory(g, ctx, vec![Table::Rows(table)], held)?;
            return Ok(true);
        }
        let spill = self.spiller(ctx);
        // Tests: any final stall also refuses the widening.
        let refuse = self.refused(usize::MAX);
        if !Self::widen_inputs(&mut inputs, &mut held, spill.is_some(), refuse).await? {
            // No memory to widen the narrow tables: everything goes to disk.
            if let Some(s) = &spill {
                self.spill_inputs(g, ctx, s, inputs, held).await?;
            }
            return Ok(false);
        }
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
        // Tasks finished so far (flushed ones included).
        let mut finished = 0;
        loop {
            let asked = spill.is_some() && lock(&g.inner).spill_asked;
            let mut stalled = false;
            while !asked && running.len() < limit {
                let Some(mut task) = queue.pop_front() else {
                    break;
                };
                if self.refused(finished) {
                    queue.push_front(task);
                    stalled = running.is_empty();
                    break;
                }
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
                None => {
                    finished += 1;
                    done.push(task);
                }
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
            let out = Output::on_disk(Some(file), Arc::clone(ctx.executor()));
            let _ = g.done.set(Final {
                out: Arc::new(out),
                chunks: 1,
            });
            return Ok(true);
        }
        drop(spare);
        let mut tables = Vec::with_capacity(done.len());
        let mut reservations = Vec::with_capacity(done.len());
        let mut groups = 0;
        done.sort_by_key(|t| t.parts.start);
        for task in done {
            let n = task.table.len();
            groups += n;
            bump(&self.counters.hash_buckets, task.table.buckets() as u64);
            tables.push(Table::Rows(task.table));
            reservations.push(task.reservation);
        }
        bump(&self.counters.groups, groups as u64);
        bump(&self.counters.hash_entries, groups as u64);
        {
            let mut inner = lock(&g.inner);
            inner.busy = false;
            inner.row_merge = false;
            inner.spill_asked = false;
        }
        self.finish_memory(g, ctx, tables, reservations)?;
        Ok(true)
    }

    /// Whether a test refuses memory to the next task once `finished`
    /// tasks finished (`HashAggregate::with_final_stall`; never outside
    /// tests).
    fn refused(&self, finished: usize) -> bool {
        #[cfg(test)]
        let refused = self.stall_after.is_some_and(|n| finished >= n);
        #[cfg(not(test))]
        let refused = {
            let _ = (self, finished);
            false
        };
        refused
    }

    /// Widens the driver tables that stayed narrow (below the widening
    /// size) when another one widened, so every input has the same radix
    /// partitions; tables that all stayed narrow merge in one task. Each
    /// widening first reserves the copy it makes. When that is refused,
    /// with a spill directory (`spill`) this returns `false` (the caller
    /// spills the inputs); without one the rest stay narrow and one task
    /// merges every partition of every input (see
    /// [`final_queue`](Self::final_queue)), which needs no copy. `refuse`
    /// (a test hook) refuses every reservation.
    async fn widen_inputs(
        inputs: &mut [RowTable],
        held: &mut [Reservation],
        spill: bool,
        refuse: bool,
    ) -> Result<bool> {
        let wide = inputs.iter().map(RowTable::part_count).max().unwrap_or(1);
        for (t, r) in inputs.iter_mut().zip(held.iter_mut()) {
            if t.part_count() == wide {
                continue;
            }
            let before = r.size();
            let extra = t.widen_bytes();
            if refuse || r.try_grow(extra).is_err() {
                // Without a spill directory, the inputs stay of mixed
                // widths and merge in one task.
                return Ok(!spill);
            }
            t.widen()?;
            if t.part_count() != wide {
                return Err(internal(
                    "driver tables of different radix partitions".to_owned(),
                ));
            }
            shrink_to(r, before.max(t.memory_size()));
        }
        Ok(true)
    }

    /// Writes every driver table as a spill run (on the blocking pool) and
    /// hands the runs to the disk phase.
    async fn spill_inputs(
        &self,
        g: &Global,
        ctx: &TaskContext,
        s: &Spiller,
        inputs: Vec<RowTable>,
        held: Vec<Reservation>,
    ) -> Result<()> {
        bump(&self.counters.row_final_spills, 1);
        let s = s.clone();
        let runs = ctx
            .run_blocking(move || {
                let mut runs = Vec::with_capacity(inputs.len());
                for (t, mut r) in inputs.into_iter().zip(held) {
                    lend_headroom(&mut r);
                    runs.push(s.write_run(&Table::Rows(t), 0, 0, u32::MAX)?);
                }
                Ok(runs)
            })
            .await?;
        let mut inner = lock(&g.inner);
        inner.busy = false;
        inner.disk = true;
        tasks_of(runs, 0, &mut inner.tasks);
        Ok(())
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
        // The inputs' radix width (one partition when none widened).
        let wide = inputs.iter().map(RowTable::part_count).max().unwrap_or(1);
        let same = inputs.iter().all(|t| t.part_count() == wide);
        // A one-partition merge table takes inputs of any width.
        let bits = if same { wide.trailing_zeros() } else { 0 };
        let plan = if same {
            let refs: Vec<&RowTable> = inputs.iter().collect();
            final_tasks(&refs)
        } else {
            // Mixed widths (a narrow table could not widen): one task with
            // one partition merges every partition of every input, in
            // table order. It is presized for the largest input only
            // (memory is short; the table grows if there are more groups).
            vec![FinalTask {
                parts: 0..wide,
                groups: inputs.iter().map(RowTable::len).max().unwrap_or(0),
            }]
        };
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
            let table = RowTable::new(&self.core, &g.seed, Mode::Grow, false, bits)?
                .with_resizes(Arc::clone(&self.counters.resizes))
                .with_probes(g.probes.clone());
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
                let runs = s.write_task_runs(tasks)?;
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

/// One final-phase task to write as a spill run: its index (the run's
/// place), the task, and the partition files written so far (a task an
/// extra writer handed back keeps the files it finished).
type Pending = (usize, Task, Vec<Option<SpillFile>>);

/// Writes `task`'s run with writers from `writer`, resuming at the
/// partitions `files` does not hold: its table, then every input row it did
/// not merge yet (per key, merge order: what the task merged precedes every
/// input row it did not reach).
fn write_task(
    s: &Spiller,
    task: &Task,
    files: &mut [Option<SpillFile>],
    writer: &dyn Fn() -> std::result::Result<SpillWriter, SpillError>,
    stop: &AtomicBool,
) -> Result<std::result::Result<(), SpillError>> {
    let at = task.at;
    let all = |_: u32| true;
    let keeps: Vec<_> = task
        .inputs
        .iter()
        .enumerate()
        .map(|(i, (input, _))| {
            move |r: u32| {
                let (p, row) = input.split_ref(r);
                (i, p, row) >= at
            }
        })
        .collect();
    let mut sources: Vec<(&RowTable, &dyn Fn(u32) -> bool)> = vec![(&task.table, &all)];
    for ((input, _), keep) in task.inputs.iter().zip(&keeps) {
        sources.push((input, keep));
    }
    s.write_run_sources(&sources, 0, writer, files, stop)
}

/// What the writers of [`Spiller::write_task_runs`] share.
struct TaskRuns {
    /// Tasks not written yet.
    todo: Mutex<VecDeque<Pending>>,
    /// Each task's run, by index.
    runs: Mutex<Vec<Option<Run>>>,
    /// The first failure; every writer stops at its next chunk.
    failed: Mutex<Option<KrishivError>>,
    /// Set with `failed`: what the writers check while writing.
    stop: AtomicBool,
    /// Extra writers running now.
    live: AtomicUsize,
}

impl TaskRuns {
    fn next(&self) -> Option<Pending> {
        if self.stop.load(Ordering::Acquire) {
            return None;
        }
        lock(&self.todo).pop_front()
    }

    fn fail(&self, e: KrishivError) {
        lock(&self.failed).get_or_insert(e);
        self.stop.store(true, Ordering::Release);
    }

    /// Records the outcome of writing `pending` (taken by `writer`); a
    /// refused writer's task goes back to the front of the queue with the
    /// files it finished. Returns whether the writer goes on.
    fn done(
        &self,
        s: &Spiller,
        (i, task, files): Pending,
        outcome: Result<std::result::Result<(), SpillError>>,
        helper: bool,
    ) -> bool {
        match outcome {
            Ok(Ok(())) => {
                bump(&s.counters.spill_runs, 1);
                if helper {
                    bump(&s.counters.spill_helper_runs, 1);
                }
                lock(&self.runs)[i] = Some(Run::new(files, u32::MAX));
                // `task` drops here: its memory goes back to the pool.
                true
            }
            Ok(Err(SpillError::Memory(_))) if helper => {
                bump(&s.counters.spill_handbacks, 1);
                lock(&self.todo).push_front((i, task, files));
                false
            }
            Ok(Err(e)) => {
                self.fail(e.into());
                false
            }
            Err(e) => {
                self.fail(e);
                false
            }
        }
    }
}

/// Counts one extra writer as running while it lives.
struct Live<'a>(&'a AtomicUsize);

impl Drop for Live<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Spiller {
    /// Writes each task as one spill run (X2b', design P2-X2 §11.3), the
    /// runs in the tasks' order. The calling thread writes from the pool's
    /// spill-scratch floor, as before; each run it finishes drops its task,
    /// whose memory goes back to the pool, and then extra writers start, as
    /// many as the pool gives scratch for ([`Spiller::one_helper`]), up to
    /// [`MAX_WRITERS`] running at once, each taking the next task. An extra
    /// writer refused scratch puts its task back with the partition files
    /// it finished and stops; the next writer resumes it. The tasks share
    /// nothing, so their runs are written independently.
    fn write_task_runs(&self, tasks: VecDeque<Task>) -> Result<Vec<Run>> {
        let n = tasks.len();
        let shared = TaskRuns {
            todo: Mutex::new(
                tasks
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| (i, t, (0..PARTITIONS).map(|_| None).collect()))
                    .collect(),
            ),
            runs: Mutex::new((0..n).map(|_| None).collect()),
            failed: Mutex::new(None),
            stop: AtomicBool::new(false),
            live: AtomicUsize::new(0),
        };
        let floor = || {
            self.scope.writer(
                Arc::clone(&self.core.state_schema),
                self.consumer.reservation().for_spill_scratch(),
            )
        };
        let write_own = |mut pending: Pending| {
            let outcome = write_task(self, &pending.1, &mut pending.2, &floor, &shared.stop);
            shared.done(self, pending, outcome, false);
        };
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            while let Some(pending) = shared.next() {
                write_own(pending);
                // The task's memory is back in the pool: start writers for
                // the tasks left, while there is room for their scratch.
                while shared.live.load(Ordering::Acquire) + 1 < MAX_WRITERS
                    && lock(&shared.todo).len() > 1
                {
                    let Some(h) = self.one_helper() else { break };
                    shared.live.fetch_add(1, Ordering::AcqRel);
                    let live = Live(&shared.live);
                    let shared = &shared;
                    let started = std::thread::Builder::new()
                        .name("krishiv-spill-write".to_owned())
                        .spawn_scoped(scope, move || {
                            let _live = live;
                            self.help_tasks(&h, shared);
                        });
                    match started {
                        Ok(j) => handles.push(j),
                        // The closure (and `live`) dropped: one writer fewer.
                        Err(_) => break,
                    }
                    let writers = shared.live.load(Ordering::Acquire) + 1;
                    self.counters.spill_writers.fetch_max(
                        u64::try_from(writers).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                }
            }
            for j in handles {
                if let Err(p) = j.join() {
                    let what = p
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_owned())
                        .or_else(|| p.downcast_ref::<String>().cloned())
                        .unwrap_or_default();
                    shared.fail(internal(format!("a spill writer thread panicked: {what}")));
                }
            }
        });
        // Tasks handed back after the loop above ended.
        while let Some(pending) = shared.next() {
            write_own(pending);
        }
        if let Some(e) = lock(&shared.failed).take() {
            return Err(e);
        }
        let runs = std::mem::take(&mut *lock(&shared.runs));
        runs.into_iter()
            .map(|r| {
                r.ok_or_else(|| {
                    internal("a spill run of the final phase was not written".to_owned())
                })
            })
            .collect()
    }

    /// An extra writer of [`Self::write_task_runs`]: tasks until none are
    /// left, a run failed, or the pool refuses its scratch (that task goes
    /// back to the front of the queue, with the files it finished).
    fn help_tasks(&self, h: &Helper, shared: &TaskRuns) {
        let writer = || self.helper_writer(h);
        while let Some(mut pending) = shared.next() {
            if self.refused_in_test() {
                bump(&self.counters.spill_handbacks, 1);
                lock(&shared.todo).push_front(pending);
                return;
            }
            let outcome = write_task(self, &pending.1, &mut pending.2, &writer, &shared.stop);
            if !shared.done(self, pending, outcome, true) {
                return;
            }
        }
    }
}
