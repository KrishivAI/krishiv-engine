// SPDX-License-Identifier: Apache-2.0
//! The in-memory final phase of row tables (P2-T159): radix partitions
//! merged in parallel.
//!
//! Every driver's [`RowTable`] holds its groups in [`PARTS`] radix
//! partitions of the seedless row hash `h`, with partial duplicates
//! where its fixed index was cleared. The final phase is "read partition
//! `p` from every producer" (P2-T171 §5.1, I8): one task per range of
//! partitions ([`final_tasks`], sized from the merged per-partition
//! sketches) merges that range's rows from every table, in table order,
//! into its own growing table, presized from the sketches. Tasks share
//! nothing mutable: each owns its table and its reservation and reads the
//! driver tables, which are immutable here. They run on the blocking pool,
//! at most the executor's parallelism at once, each started once its
//! memory is reserved (as the join's partition builds are).
//!
//! **Memory.** The driver tables stay until every task is done (their
//! states are per-driver accumulators, not per partition), so this phase
//! holds the driver tables and the merged tables together: up to about
//! twice the default path's in-place merge. Releasing each partition's
//! input once merged needs per-partition (or inline) states; until then
//! the setting stays off by default.
//!
//! When a task cannot get memory and none is running: with a spill
//! directory, every merge table and every row not yet merged is written as
//! spill runs and the disk phase takes over (the default path's in-memory
//! merge does the same); without one, the task waits for the memory
//! through arbitration.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use krishiv_common::error::KrishivError;
use krishiv_memory::Reservation;

use super::super::rowtable::{Mode, PARTS, RowTable, final_tasks, split_ref};
use super::super::table::Table;
use super::{CHUNK, Final, Global, HashAggregate, Run, bump, lend_headroom, reserve_to, tasks_of};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::sync::lock;

/// One final-phase task.
struct Task {
    parts: Range<usize>,
    table: RowTable,
    reservation: Reservation,
    /// The next row to merge: (input table, partition, row).
    at: (usize, usize, usize),
    /// Bytes to reserve before the task can (re)start.
    need: usize,
    /// Groups to presize the index for on the first start (0 once done).
    presize: usize,
}

impl Task {
    /// Merges until done (`None`), or until the reservation cannot grow
    /// (`Some(missing bytes)`).
    fn run(&mut self, inputs: &[RowTable], query: &QueryContext) -> Result<Option<usize>> {
        if self.presize > 0 {
            self.table.presize(std::mem::take(&mut self.presize));
        }
        loop {
            query.check_cancelled()?;
            let (t, p, row) = self.at;
            let Some(input) = inputs.get(t) else {
                let keep = self.table.memory_size();
                self.reservation
                    .shrink(self.reservation.size().saturating_sub(keep));
                return Ok(None);
            };
            if p >= self.parts.end {
                self.at = (t + 1, self.parts.start, 0);
                continue;
            }
            let len = input.part_len(p);
            if row >= len {
                self.at = (t, p + 1, 0);
                continue;
            }
            let end = (row + CHUNK).min(len);
            let chunk = input.merge_chunk(p, row..end)?;
            let want = self.table.projected_merge(&chunk);
            if let Err(missing) = reserve_to(&mut self.reservation, want) {
                return Ok(Some(missing));
            }
            self.table.apply_merge(input, &chunk)?;
            self.at = (t, p, end);
        }
    }
}

type Running<'a> = FuturesUnordered<
    std::pin::Pin<Box<dyn Future<Output = Result<(Task, Option<usize>)>> + Send + 'a>>,
>;

impl HashAggregate {
    /// Finalizes the combined row tables in memory, partitions in parallel.
    /// Returns `false` when nothing was done (spill runs exist: the disk
    /// phase merges everything) or when memory ran out and everything was
    /// spilled; the step machine then continues with the disk phase.
    pub(super) async fn finalize_rows(&self, g: &Global, ctx: &TaskContext) -> Result<bool> {
        let (inputs, held) = {
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
        let mut inputs = inputs;
        if let [only] = inputs.as_slice()
            && only.is_unique()
        {
            let table = inputs
                .pop()
                .ok_or_else(|| internal("no driver table".to_owned()))?;
            bump(&self.counters.groups, table.len() as u64);
            let mut inner = lock(&g.inner);
            inner.busy = false;
            let _ = g.done.set(Final::Table {
                table: Table::Rows(table),
                _held: held,
            });
            return Ok(true);
        }
        let inputs: Arc<[RowTable]> = inputs.into();
        // The driver tables are read by the tasks, never spilled from here.
        ctx.consumer().signal().set_spillable_bytes(0);
        let mut queue = VecDeque::new();
        {
            let refs: Vec<&RowTable> = inputs.iter().collect();
            for t in final_tasks(&refs) {
                let table = RowTable::new(&self.core, &g.seed, Mode::Grow, false)?;
                queue.push_back(Task {
                    need: table.projected_presize(t.groups),
                    presize: t.groups.max(1),
                    at: (0, t.parts.start, 0),
                    parts: t.parts,
                    table,
                    reservation: ctx.consumer().reservation(),
                });
            }
        }
        let limit = ctx.executor().parallelism().max(1);
        let mut running: Running<'_> = FuturesUnordered::new();
        let mut done = Vec::new();
        loop {
            while running.len() < limit {
                let Some(mut task) = queue.pop_front() else {
                    break;
                };
                if task.need > 0 {
                    if task.reservation.try_grow(task.need).is_err() {
                        if !running.is_empty() {
                            queue.push_front(task);
                            break;
                        }
                        if self.core.spec.spill.is_some() {
                            queue.push_front(task);
                            queue.extend(done);
                            self.spill_rows(g, ctx, inputs, held, queue).await?;
                            return Ok(false);
                        }
                        task.reservation
                            .grow(task.need)
                            .await
                            .map_err(KrishivError::from)?;
                    }
                    task.need = 0;
                }
                let (inputs, query) = (Arc::clone(&inputs), Arc::clone(ctx.query()));
                running.push(Box::pin(ctx.run_blocking(move || {
                    let stall = task.run(&inputs, &query)?;
                    Ok((task, stall))
                })));
            }
            let Some(next) = running.next().await else {
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
        }
        drop(running);
        // Every task is done: the driver tables go.
        drop(inputs);
        drop(held);
        let batch = ctx.query().config().batch_size.max(1);
        let mut tables = Vec::with_capacity(done.len());
        let mut chunks = Vec::new();
        let mut reservations = Vec::with_capacity(done.len());
        let mut groups = 0;
        done.sort_by_key(|t| t.parts.start);
        for (i, task) in done.into_iter().enumerate() {
            let n = task.table.len();
            groups += n;
            chunks.extend((0..n).step_by(batch).map(|s| (i, s..(s + batch).min(n))));
            tables.push(Table::Rows(task.table));
            reservations.push(task.reservation);
        }
        bump(&self.counters.groups, groups as u64);
        let mut inner = lock(&g.inner);
        inner.busy = false;
        let _ = g.done.set(Final::Parts {
            tables,
            chunks,
            _held: reservations,
        });
        Ok(true)
    }

    /// Writes every task's table and every input row no task merged yet as
    /// spill runs (on the blocking pool), then hands the runs to the disk
    /// phase.
    async fn spill_rows(
        &self,
        g: &Global,
        ctx: &TaskContext,
        inputs: Arc<[RowTable]>,
        held: Vec<Reservation>,
        tasks: VecDeque<Task>,
    ) -> Result<()> {
        let s = self
            .spiller(ctx)
            .ok_or_else(|| internal("spilling an unspillable aggregate".to_owned()))?;
        let mut held = held;
        let runs = ctx
            .run_blocking(move || {
                // The driver tables kept spill headroom: the writers' scratch.
                for r in &mut held {
                    lend_headroom(r);
                }
                let mut runs: Vec<Run> = Vec::new();
                // Per partition, the cursor of the task that merges it: rows
                // before it are in that task's table.
                let mut cursor = vec![(0, 0, 0); PARTS];
                for t in &tasks {
                    for p in t.parts.clone() {
                        cursor[p] = t.at;
                    }
                }
                // Merge order: what the tasks merged precedes every row
                // they did not reach.
                for t in tasks {
                    let Task {
                        table, reservation, ..
                    } = t;
                    runs.push(s.write_run(&Table::Rows(table), 0, 0)?);
                    drop(reservation);
                }
                for (i, input) in inputs.iter().enumerate() {
                    let keep = |r: u32| {
                        let (p, row) = split_ref(r);
                        cursor.get(p).is_none_or(|&at| (i, p, row) >= at)
                    };
                    runs.push(s.write_run_rows(input, &keep, 0)?);
                }
                drop(held);
                drop(inputs);
                Ok(runs)
            })
            .await?;
        let mut inner = lock(&g.inner);
        inner.busy = false;
        inner.disk = true;
        tasks_of(runs, 0, &mut inner.tasks);
        Ok(())
    }
}
