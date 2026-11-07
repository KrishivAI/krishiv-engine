// SPDX-License-Identifier: Apache-2.0
//! A run's partition files written at once (X2b, design P2-X2 §10).
//!
//! A spilled run is one file per level-0 partition. Written one after the
//! other, a large table's spill is one thread encoding, compressing (lz4)
//! and writing every partition while the drivers wait for the memory it
//! frees: after the finalize merge one table holds most of a query's
//! groups, so that thread gated q16 at SF1000.
//!
//! [`Spiller::write_parts`] writes a large run's partitions on up to
//! [`MAX_WRITERS`] threads. The calling thread writes as before, its
//! writer's scratch from the pool's spill-scratch floor. Every other
//! writer runs only on scratch it reserved first: a consumer of its own
//! (so no other writer or driver takes that memory) reserves [`HEADROOM`]
//! with an ordinary `try_grow`, into the consumer's credit, which the
//! writer's reservation draws from and returns to. When the pool has no
//! room for that, the run is written by fewer threads, down to the calling
//! one alone, as before. A write that needs more scratch than the writer
//! holds is refused by the pool (nothing is written), and the partition
//! goes back to the calling thread, which writes it after the others.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow_array::RecordBatch;
use krishiv_common::error::KrishivError;
use krishiv_memory::{ConsumerHandle, SpillError, SpillFile, SpillWriter};
use krishiv_types::batch::Batch;

use super::{HEADROOM, Run, Spiller, Table, bump, chunks};
use crate::error::{Result, internal};
use crate::hash::PARTITIONS;

/// Writers of one run at most, the calling thread included.
const MAX_WRITERS: usize = 8;

/// An extra writer starts only while the pool has this many
/// [`HEADROOM`]s free.
const MARGIN: usize = 4;

/// A run of a table smaller than this is written by the calling thread
/// alone: a few writes do not pay for threads and their scratch.
#[cfg(not(test))]
const PARALLEL_MIN_BYTES: usize = 8 << 20;
/// Tests write every run with all the writers they can get.
#[cfg(test)]
const PARALLEL_MIN_BYTES: usize = 1;

/// What one extra writer did: the files it wrote and the partitions it
/// handed back.
#[derive(Default)]
struct Done {
    files: Vec<(usize, SpillFile)>,
    back: Vec<usize>,
}

impl Spiller {
    /// Writes the state batches of the groups of each partition (`parts`,
    /// in chunks of `chunk`, of one radix partition each for a row table)
    /// as one run: a large table's partitions on several threads (see the
    /// module docs).
    pub(super) fn write_parts(
        &self,
        parts: &[Vec<u32>],
        chunk: usize,
        table: &Table,
        state: &(dyn Fn(&[u32]) -> Result<RecordBatch> + Sync),
        partition: u32,
    ) -> Result<Run> {
        let mut files: Vec<Option<SpillFile>> = (0..PARTITIONS).map(|_| None).collect();
        let todo: Vec<usize> = (0..parts.len()).filter(|&p| !parts[p].is_empty()).collect();
        let helpers = self.helpers(table.memory_size(), todo.len());
        let writers = u64::try_from(helpers.len() + 1).unwrap_or(u64::MAX);
        self.counters
            .spill_writers
            .fetch_max(writers, Ordering::Relaxed);
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let claim = || {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            todo.get(next.fetch_add(1, Ordering::AcqRel)).copied()
        };
        let write = |p: usize, w: SpillWriter| self.write_part(w, &parts[p], chunk, table, state);
        let (mine, done) = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for h in &helpers {
                let work = || self.help(h, &claim, &write, &stop);
                // A thread the system refuses is a writer fewer.
                if let Ok(j) = std::thread::Builder::new()
                    .name("krishiv-spill-write".to_owned())
                    .spawn_scoped(s, work)
                {
                    handles.push(j);
                }
            }
            let mine = self.write_own(&claim, &write, &mut files);
            if mine.is_err() {
                stop.store(true, Ordering::Release);
            }
            let done: Vec<_> = handles
                .into_iter()
                .map(std::thread::ScopedJoinHandle::join)
                .collect();
            (mine, done)
        });
        drop(helpers);
        mine?;
        let mut back = Vec::new();
        for d in done {
            let d = d.map_err(|_| internal("a spill writer thread panicked".to_owned()))??;
            for (p, f) in d.files {
                files[p] = Some(f);
            }
            back.extend(d.back);
        }
        // Handed back for want of scratch: written here, from the floor.
        for p in back {
            let w = self.run_writer()?;
            files[p] = Some(self.write_part(w, &parts[p], chunk, table, state)??);
        }
        bump(&self.counters.spill_runs, 1);
        Ok(Run::new(files, partition))
    }

    /// Consumers for the extra writers of a run of `parts` partitions from
    /// a table of `bytes`, each holding [`HEADROOM`] of credit; as many as
    /// the pool gives room for at once, without waiting.
    fn helpers(&self, bytes: usize, parts: usize) -> Vec<Arc<ConsumerHandle>> {
        if bytes < PARALLEL_MIN_BYTES {
            return Vec::new();
        }
        let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let want = parts.min(MAX_WRITERS).min(threads).saturating_sub(1);
        let mut out = Vec::with_capacity(want);
        let pool = self.consumer.pool();
        for _ in 0..want {
            // Leave room for the pool's other consumers: this is optional
            // memory, taken only while plenty is free.
            if pool
                .available()
                .is_some_and(|free| free < MARGIN * HEADROOM)
            {
                break;
            }
            let h = ConsumerHandle::register_counted(
                Arc::clone(pool),
                "HashAggregate spill writer",
                None,
                Arc::clone(self.consumer.spill_stats()),
            );
            let mut r = h.reservation();
            if r.try_grow(HEADROOM).is_err() {
                break;
            }
            r.shrink_to_credit(HEADROOM);
            out.push(h);
        }
        out
    }

    /// The calling thread's share: every partition it claims, its writer's
    /// scratch from the floor; any failure fails the run.
    fn write_own(
        &self,
        claim: &dyn Fn() -> Option<usize>,
        write: &dyn Fn(usize, SpillWriter) -> Result<std::result::Result<SpillFile, SpillError>>,
        files: &mut [Option<SpillFile>],
    ) -> Result<()> {
        while let Some(p) = claim() {
            files[p] = Some(write(p, self.run_writer()?)??);
        }
        Ok(())
    }

    /// An extra writer: partitions it claims, written on `h`'s credit,
    /// until none are left, the run failed, or the pool refuses it scratch
    /// (that partition is handed back).
    fn help(
        &self,
        h: &Arc<ConsumerHandle>,
        claim: &dyn Fn() -> Option<usize>,
        write: &(
             dyn Fn(usize, SpillWriter) -> Result<std::result::Result<SpillFile, SpillError>> + Sync
         ),
        stop: &AtomicBool,
    ) -> Result<Done> {
        let mut done = Done::default();
        let fail = |e: KrishivError| {
            stop.store(true, Ordering::Release);
            Err(e)
        };
        while let Some(p) = claim() {
            let schema = Arc::clone(&self.core.state_schema);
            let w = match self
                .scope
                .writer(schema, h.reservation().releasing_to_credit())
            {
                Ok(w) => w,
                Err(SpillError::Memory(_)) => {
                    done.back.push(p);
                    break;
                }
                Err(e) => return fail(e.into()),
            };
            match write(p, w) {
                Ok(Ok(f)) => done.files.push((p, f)),
                Ok(Err(SpillError::Memory(_))) => {
                    done.back.push(p);
                    break;
                }
                Ok(Err(e)) => return fail(e.into()),
                Err(e) => return fail(e),
            }
        }
        Ok(done)
    }

    /// Writes the groups `ids` with writer `w` as one partition file. The
    /// inner error is the writer's (a refused scratch among them: nothing
    /// of that batch was written, and the dropped writer leaves no file).
    fn write_part(
        &self,
        mut w: SpillWriter,
        ids: &[u32],
        chunk: usize,
        table: &Table,
        state: &(dyn Fn(&[u32]) -> Result<RecordBatch> + Sync),
    ) -> Result<std::result::Result<SpillFile, SpillError>> {
        for c in chunks(ids, chunk, table) {
            self.query.check_cancelled()?;
            if let Err(e) = w.write(&Batch::new(state(c)?)) {
                return Ok(Err(e));
            }
        }
        Ok(Ok(self.finish(w)?))
    }
}
