// SPDX-License-Identifier: Apache-2.0
//! Writing spill runs from row tables: one run of partition files per
//! level, resumable after a refused writer, and the run timing counters.

use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use krishiv_memory::{SpillError, SpillFile, SpillWriter};
use krishiv_types::batch::Batch;

use super::{Spiller, bump, row_chunks, spill_chunk};
use crate::aggregate::rowtable::RowTable;
use crate::error::{Result, internal};

impl Spiller {
    /// Writes the rows that each source's filter keeps as one run of
    /// partition files at `level`: per partition file, the sources in
    /// order (their merge order; blocking I/O). The sources mix driver
    /// partitions, so the run keeps its insertion order.
    ///
    /// `files` holds the run's partition files so far: partitions already
    /// written are skipped, so a run that stopped (its writer refused) is
    /// resumed by another writer without writing or counting a file twice.
    /// Each file's writer comes from `writer`; the inner error is a
    /// writer's (a refused scratch among them: the partition being written
    /// is dropped, the finished ones stay in `files`). `stop` set by another
    /// writer's failure ends the run at its next chunk (an error, which
    /// loses to that failure).
    pub(super) fn write_run_sources(
        &self,
        sources: &[(&RowTable, &dyn Fn(u32) -> bool)],
        level: u32,
        writer: &dyn Fn() -> std::result::Result<SpillWriter, SpillError>,
        files: &mut [Option<SpillFile>],
        stop: &AtomicBool,
    ) -> Result<std::result::Result<(), SpillError>> {
        let started = std::time::Instant::now();
        let per: Vec<Vec<Vec<u32>>> = sources
            .iter()
            .map(|(t, keep)| {
                let mut parts = t.ids_by_partition(level, 0);
                for ids in &mut parts {
                    ids.retain(|&r| keep(r));
                }
                parts
            })
            .collect();
        let outcome = (|| {
            for (p, file) in files.iter_mut().enumerate() {
                if file.is_some() || per.iter().all(|parts| parts[p].is_empty()) {
                    continue;
                }
                let mut w = match writer() {
                    Ok(w) => w,
                    Err(e) => return Ok(Err(e)),
                };
                for ((t, _), parts) in sources.iter().zip(&per) {
                    let chunk = spill_chunk(t.memory_size(), t.len());
                    for ids in row_chunks(&parts[p], chunk, t) {
                        self.query.check_cancelled()?;
                        if stop.load(AtomicOrdering::Acquire) {
                            return Err(internal(
                                "stopped: another spill writer failed".to_owned(),
                            ));
                        }
                        // Cut by bytes too: large sketch states must keep the `binary` offsets under 2 GiB.
                        let mut rest = ids;
                        while !rest.is_empty() {
                            // Each byte-cut frame is materialized and written, so a peer's failure
                            // stops the run before the next one, not after the whole chunk.
                            if stop.load(AtomicOrdering::Acquire) {
                                return Err(internal(
                                    "stopped: another spill writer failed".to_owned(),
                                ));
                            }
                            let n = t.state_cut(rest).clamp(1, rest.len());
                            let (now, later) = rest.split_at(n);
                            if let Err(e) = w.write(&Batch::new(t.state_batch(now)?)) {
                                return Ok(Err(e));
                            }
                            rest = later;
                        }
                    }
                }
                *file = Some(self.finish(w)?);
            }
            Ok(Ok(()))
        })();
        self.time_run(started);
        outcome
    }

    /// Adds one writer's time on a run, since `started`, to
    /// `spill_write_ns` and `spill_write_max_ns`.
    pub(super) fn time_run(&self, started: std::time::Instant) {
        let ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        bump(&self.counters.spill_write_ns, ns);
        self.counters
            .spill_write_max_ns
            .fetch_max(ns, AtomicOrdering::Relaxed);
    }
}
