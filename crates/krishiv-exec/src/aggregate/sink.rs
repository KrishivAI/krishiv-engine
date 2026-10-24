// SPDX-License-Identifier: Apache-2.0
//! The aggregation breaker: per-driver tables, spill runs, and the
//! finalize-time merge (in memory, or partition by partition from disk).
//!
//! # Spill
//! A driver asked to spill writes its table as one *run*: partial states in
//! one file per hash partition (level 0), and starts a new table. `finalize`
//! then has in-memory tables and runs:
//! - **No runs:** the tables merge into the largest one, a chunk of groups
//!   per step, and each merged table is freed. If memory runs out (or a
//!   spill request arrives) midway, everything left is spilled to runs.
//! - **Runs:** each partition is a *task*: its files merge into one table,
//!   whose results go to an output spill file. A task whose table does not
//!   fit is split: the table and the rest of its input are re-partitioned by
//!   the next hash bits into new tasks (up to [`MAX_LEVEL`]).
//!
//! `finalize` keeps all of this in the global state between steps and
//! yields after each, so [`Sink::spill_global`] can free memory at any
//! await. Whole-table writes run on the blocking pool.
//!
//! An in-memory result stays spillable until read: it is charged to a
//! consumer of its own whose hook writes the chunks no reader has claimed
//! to an output file ([`crate::output_spill`]), and its tables go once the
//! last chunk's reader is done with them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock, Weak};

use arrow_array::{RecordBatch, UInt32Array};
use arrow_select::take::take_record_batch;
use futures::future::BoxFuture;
use krishiv_common::error::{KrishivError, UnableToAcquireMemory};
use krishiv_memory::{
    ConsumerHandle, Reservation, SpillFile, SpillReader, SpillScope, SpillSignal, SpillWriter,
};
use krishiv_types::batch::{Batch, Weights};

use super::rowtable::{Mode, RowTable};
use super::table::{AggCore, AggTable, Table};
use super::{AggregateSpec, unsupported};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, run_blocking};
use crate::hash::{HashSeed, MAX_LEVEL, PARTITIONS, ProbeTotals};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::output_spill::{OutputSpill, SpillUnread, Unfinished, refused};
use crate::sync::{Event, Mutex, lock};

mod parallel;

/// Groups per state batch when spilling and merging (one finalize step).
const CHUNK: usize = 1024;
/// Memory a spillable table keeps reserved for the scratch of spill I/O
/// (a writer needs about twice the batch it encodes plus a 64 KiB buffer).
const HEADROOM: usize = 128 << 10;

/// `EXPLAIN ANALYZE` counters of one aggregate, also reported through
/// [`Sink::extra_metrics`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AggregateMetrics {
    /// Groups emitted.
    pub groups: u64,
    /// Spill runs written (per-driver spills, finalize-time spills, splits).
    pub spill_runs: u64,
    /// Spill files written (runs, splits and the output).
    pub spill_files: u64,
    /// Bytes written to spill files.
    pub spill_bytes: u64,
    /// Partitions split again because they did not fit.
    pub repartitions: u64,
}

#[derive(Debug, Default)]
struct Counters {
    groups: AtomicU64,
    /// Distinct keys of the tables the output came from.
    hash_entries: AtomicU64,
    /// Their buckets.
    hash_buckets: AtomicU64,
    probes: Arc<ProbeTotals>,
    spill_runs: AtomicU64,
    spill_files: AtomicU64,
    spill_bytes: AtomicU64,
    repartitions: AtomicU64,
    /// Row tables were used (`krishiv.exec.agg.rowLayout`): their counters
    /// below are reported.
    row_layout: std::sync::atomic::AtomicBool,
    /// Times a thread-local row table's fixed index was cleared when full.
    row_abandons: Arc<AtomicU64>,
    /// Partition tasks of the row tables' final phase.
    row_final_tasks: AtomicU64,
    /// Final phases that moved to disk (out of memory, or a spill request).
    row_final_spills: AtomicU64,
    /// Times the final phase wrote its finished partitions' groups to the
    /// output file to free memory.
    row_final_flushes: AtomicU64,
}

fn bump(c: &AtomicU64, n: u64) {
    c.fetch_add(n, AtomicOrdering::Relaxed);
}

/// A hash-aggregation pipeline breaker (see the [module docs](super)).
#[derive(Debug)]
pub struct HashAggregate {
    core: Arc<AggCore>,
    counters: Arc<Counters>,
    /// Overrides the thread-local row tables' index mode (tests force
    /// abandons).
    row_mode: Option<Mode>,
    /// Tests: once this many final-phase tasks finished, no further task
    /// gets its memory (forces the flush and spill paths).
    #[cfg(test)]
    stall_after: Option<usize>,
    /// A gate the result's spill waits at before writing (tests cancel a
    /// query during the spill).
    #[cfg(test)]
    output_gate: Option<Arc<crate::testing::BlockingGate>>,
}

impl HashAggregate {
    /// An aggregate for `spec`.
    ///
    /// # Errors
    /// Typing errors of the aggregates; `INTERNAL_ERROR` for bad columns.
    pub fn new(spec: AggregateSpec) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            core: Arc::new(AggCore::new(spec)?),
            counters: Arc::default(),
            row_mode: None,
            #[cfg(test)]
            stall_after: None,
            #[cfg(test)]
            output_gate: None,
        }))
    }

    /// [`new`](Self::new) whose row tables (when the row layout is on)
    /// start their thread-local index in `mode`, deterministic mode or not:
    /// tests force abandons and append-only tables with it.
    #[cfg(test)]
    pub(crate) fn with_row_mode(spec: AggregateSpec, mode: Mode) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            core: Arc::new(AggCore::new(spec)?),
            counters: Arc::default(),
            row_mode: Some(mode),
            stall_after: None,
            output_gate: None,
        }))
    }

    /// [`with_row_mode`](Self::with_row_mode) whose final phase refuses
    /// memory to every task after the first `finished` ones finished.
    #[cfg(test)]
    pub(crate) fn with_final_stall(
        spec: AggregateSpec,
        mode: Mode,
        finished: usize,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            core: Arc::new(AggCore::new(spec)?),
            counters: Arc::default(),
            row_mode: Some(mode),
            stall_after: Some(finished),
            output_gate: None,
        }))
    }

    /// [`new`](Self::new) whose result's spill waits at `gate` before it
    /// writes.
    #[cfg(test)]
    pub(crate) fn with_output_gate(
        spec: AggregateSpec,
        gate: Arc<crate::testing::BlockingGate>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            core: Arc::new(AggCore::new(spec)?),
            counters: Arc::default(),
            row_mode: None,
            stall_after: None,
            output_gate: Some(gate),
        }))
    }

    /// The counters so far (complete once the query finished).
    pub fn metrics(&self) -> AggregateMetrics {
        let c = &self.counters;
        let l = |a: &AtomicU64| a.load(AtomicOrdering::Relaxed);
        AggregateMetrics {
            groups: l(&c.groups),
            spill_runs: l(&c.spill_runs),
            spill_files: l(&c.spill_files),
            spill_bytes: l(&c.spill_bytes),
            repartitions: l(&c.repartitions),
        }
    }

    /// A new table: a thread-local one (`local`), or a merge table.
    fn new_table(&self, g: &Global, local: bool) -> Result<Table> {
        Ok(match g.rows {
            Some(mode) => {
                // Merge tables of the disk phase need no radix partitions.
                let (mode, bits) = if local {
                    (mode, g.row_bits)
                } else {
                    (Mode::Grow, 0)
                };
                Table::Rows(
                    RowTable::new(&self.core, &g.seed, mode, local, bits)?
                        .with_abandons(Arc::clone(&self.counters.row_abandons)),
                )
            }
            None => Table::Keys(AggTable::new(&self.core, &g.seed)?.with_probes(g.probes.clone())),
        })
    }

    fn headroom(&self) -> usize {
        if self.core.spec.spill.is_some() {
            HEADROOM
        } else {
            0
        }
    }

    fn spiller(&self, ctx: &TaskContext) -> Option<Spiller> {
        Some(Spiller {
            core: Arc::clone(&self.core),
            counters: Arc::clone(&self.counters),
            query: Arc::clone(ctx.query()),
            scope: self.core.spec.spill.clone()?,
            consumer: Arc::clone(ctx.consumer().consumer()),
            batch_size: ctx.query().config().batch_size.max(1),
        })
    }
}

/// What blocking spill work needs, owned (`'static`).
#[derive(Clone)]
struct Spiller {
    core: Arc<AggCore>,
    counters: Arc<Counters>,
    query: Arc<QueryContext>,
    scope: SpillScope,
    consumer: Arc<ConsumerHandle>,
    batch_size: usize,
}

impl Spiller {
    fn writer(&self) -> Result<SpillWriter> {
        let schema = Arc::clone(&self.core.state_schema);
        let w = self.scope.writer(schema, self.consumer.reservation());
        w.map_err(KrishivError::from)
    }

    /// A writer for one spill run, finished before the spill returns: its
    /// scratch may use the pool's spill-scratch floor, so a spill of a full
    /// pool does not fail for want of its writer's buffers.
    fn run_writer(&self) -> Result<SpillWriter> {
        let schema = Arc::clone(&self.core.state_schema);
        let scratch = self.consumer.reservation().for_spill_scratch();
        let w = self.scope.writer(schema, scratch);
        w.map_err(KrishivError::from)
    }

    fn finish(&self, w: SpillWriter) -> Result<SpillFile> {
        let f = w.finish().map_err(KrishivError::from)?;
        bump(&self.counters.spill_files, 1);
        bump(&self.counters.spill_bytes, f.bytes());
        Ok(f)
    }

    /// Writes groups `from..` of `table` as one run of partition files at
    /// `level` (blocking I/O).
    fn write_run(&self, table: &Table, from: usize, level: u32) -> Result<Run> {
        let parts = table.ids_by_partition(level, from);
        let chunk = spill_chunk(table.memory_size(), table.len());
        self.write_parts(parts, chunk, table, &|ids| table.state_batch(ids))
    }

    /// Writes the rows that each source's filter keeps as one run of
    /// partition files at `level`: per partition file, the sources in
    /// order (their merge order; blocking I/O).
    fn write_run_sources(
        &self,
        sources: &[(&RowTable, &dyn Fn(u32) -> bool)],
        level: u32,
    ) -> Result<Run> {
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
        let mut files: Vec<Option<SpillFile>> = (0..PARTITIONS).map(|_| None).collect();
        for (p, file) in files.iter_mut().enumerate() {
            if per.iter().all(|parts| parts[p].is_empty()) {
                continue;
            }
            let mut writer = self.run_writer()?;
            for ((t, _), parts) in sources.iter().zip(&per) {
                let chunk = spill_chunk(t.memory_size(), t.len());
                for ids in row_chunks(&parts[p], chunk, t) {
                    self.query.check_cancelled()?;
                    writer
                        .write(&Batch::new(t.state_batch(ids)?))
                        .map_err(KrishivError::from)?;
                }
            }
            *file = Some(self.finish(writer)?);
        }
        bump(&self.counters.spill_runs, 1);
        Ok(Run(files))
    }

    /// Writes the state batches of the groups of each partition (`parts`,
    /// in chunks of `chunk`, of one radix partition each for a row table)
    /// as one run.
    fn write_parts(
        &self,
        parts: Vec<Vec<u32>>,
        chunk: usize,
        table: &Table,
        state: &dyn Fn(&[u32]) -> Result<RecordBatch>,
    ) -> Result<Run> {
        let mut files: Vec<Option<SpillFile>> = (0..PARTITIONS).map(|_| None).collect();
        for (p, ids) in parts.into_iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let mut writer = self.run_writer()?;
            for chunk in chunks(&ids, chunk, table) {
                self.query.check_cancelled()?;
                writer
                    .write(&Batch::new(state(chunk)?))
                    .map_err(KrishivError::from)?;
            }
            files[p] = Some(self.finish(writer)?);
        }
        bump(&self.counters.spill_runs, 1);
        Ok(Run(files))
    }

    /// Writes `chunks` of an in-memory result to one output file
    /// (blocking I/O); the writer's scratch may use the pool's
    /// spill-scratch floor.
    fn write_chunks(&self, mem: &InMemory, chunks: std::ops::Range<usize>) -> Result<SpillFile> {
        let schema = Arc::clone(&self.core.output_schema);
        let scratch = self.consumer.reservation().for_spill_scratch();
        let mut w = self.scope.writer(schema, scratch)?;
        for chunk in chunks {
            self.query.check_cancelled()?;
            w.write(&mem.batch(chunk)?)?;
        }
        self.finish(w)
    }

    /// Writes the final output of `table` to `out` (blocking I/O).
    fn write_output(&self, table: &Table, out: &mut SpillWriter) -> Result<()> {
        let n = u32::try_from(table.len()).map_err(|_| internal("too many groups".to_owned()))?;
        let ids = table.ids(0..table.len());
        let size = out_chunk(table.memory_size(), table.len(), self.batch_size);
        for chunk in chunks(&ids, size, table) {
            self.query.check_cancelled()?;
            let b = Batch::new(table.output_batch(chunk)?);
            out.write(&b).map_err(KrishivError::from)?;
        }
        bump(&self.counters.groups, u64::from(n));
        bump(&self.counters.hash_entries, u64::from(n));
        bump(&self.counters.hash_buckets, table.buckets() as u64);
        Ok(())
    }
}

/// Groups per spilled batch, so that encoding one fits [`HEADROOM`]: a
/// table's bytes per group bound its state's.
fn spill_chunk(memory: usize, groups: usize) -> usize {
    let per_group = memory / groups.max(1);
    ((HEADROOM - (64 << 10)) / 3 / per_group.max(1)).clamp(16, CHUNK)
}

/// Group ids of `table` in chunks of at most `size`; a row table's chunks
/// stay within one radix partition, whose states are one partition's
/// arrays (no concatenation).
fn chunks<'a>(
    ids: &'a [u32],
    size: usize,
    table: &'a Table,
) -> Box<dyn Iterator<Item = &'a [u32]> + 'a> {
    match table {
        Table::Rows(t) => Box::new(row_chunks(ids, size, t)),
        Table::Keys(_) => Box::new(ids.chunks(size.max(1))),
    }
}

/// Row references of `table` in chunks of at most `size`, each within one
/// radix partition.
fn row_chunks<'a>(
    ids: &'a [u32],
    size: usize,
    table: &'a RowTable,
) -> impl Iterator<Item = &'a [u32]> {
    ids.chunk_by(move |a, b| table.same_part(*a, *b))
        .flat_map(move |run| run.chunks(size.max(1)))
}

/// Bytes an output frame aims at: reading the output back reserves its
/// largest frame (encoded and decoded), so frames stay well under small
/// memory limits.
const OUT_FRAME: usize = 256 << 10;

/// Groups per output batch written to disk: the batch size, or fewer when
/// a table's bytes per group say a batch would pass [`OUT_FRAME`].
fn out_chunk(memory: usize, groups: usize, batch_size: usize) -> usize {
    let per_group = memory / groups.max(1);
    (OUT_FRAME / per_group.max(1)).clamp(16, batch_size.max(16))
}

/// One spill: a file per hash partition.
#[derive(Debug)]
struct Run(Vec<Option<SpillFile>>);

struct Local {
    table: Table,
    reservation: Reservation,
    signal: Arc<SpillSignal>,
    /// This driver's consumer: combined tables are re-charged to the last
    /// driver to combine, whose spill hook serves `finalize`.
    consumer: Arc<ConsumerHandle>,
    /// The global state: a running driver's spill also spills the tables
    /// that finished drivers combined (nobody else can until `finalize`).
    global: GlobalState,
}

/// A partition being merged from disk.
struct Current {
    level: u32,
    table: Table,
    reservation: Reservation,
    files: Vec<SpillFile>,
    reader: Option<SpillReader>,
    /// A batch read but not yet applied (its memory was refused).
    pending: Option<Batch>,
    /// Re-partitioning at `level + 1`: finished runs and open writers.
    split: Option<(Vec<Run>, Vec<Option<SpillWriter>>)>,
}

#[derive(Default)]
struct Inner {
    /// Combined driver tables (in-memory phase; `tables[0]` is the target).
    tables: Vec<(Table, Reservation)>,
    /// Groups of `tables[1]` already merged into `tables[0]`.
    merged: usize,
    runs: Vec<Run>,
    /// The disk phase has started.
    disk: bool,
    tasks: Vec<(u32, Vec<SpillFile>)>,
    current: Option<Current>,
    out: Option<SpillWriter>,
    /// The finished output file.
    out_file: Option<SpillFile>,
    /// A blocking job owns part of the state: `spill_global` frees nothing.
    busy: bool,
    /// Memory granted by arbitration for the in-memory merge target (no
    /// spill directory): held with it.
    granted: Vec<Reservation>,
    /// The row tables' parallel merge runs and can spill (`spill_global`
    /// asks it through `spill_asked`).
    row_merge: bool,
    /// A spill request arrived during the row tables' merge: it stops
    /// starting tasks and spills what is left once the running ones end.
    spill_asked: bool,
}

/// Whether the row tables' parallel merge is running (tests time spill
/// requests with it).
#[cfg(test)]
pub(super) fn row_merge_running(g: &GlobalState) -> bool {
    global(g).is_ok_and(|g| lock(&g.inner).row_merge)
}

/// The finalized output, read by the source side.
struct Final {
    out: Arc<Output>,
    /// Chunks of an in-memory result: the source side's parallelism.
    chunks: usize,
}

/// An in-memory result: its tables, read in `chunks` (table, positions),
/// and the reservations that hold them.
struct InMemory {
    tables: Vec<Table>,
    chunks: Vec<(usize, std::ops::Range<usize>)>,
    held: Vec<Reservation>,
}

impl InMemory {
    fn batch(&self, chunk: usize) -> Result<Batch> {
        let (t, range) = self
            .chunks
            .get(chunk)
            .ok_or_else(|| internal("aggregate chunk out of range".to_owned()))?;
        let t = self
            .tables
            .get(*t)
            .ok_or_else(|| internal("aggregate chunk without its table".to_owned()))?;
        Ok(Batch::new(t.output_batch(&t.ids(range.clone()))?))
    }

    fn bytes(&self) -> usize {
        self.held.iter().map(Reservation::size).sum()
    }
}

/// Where the source side reads.
enum Out {
    /// In memory; chunks `next..` are unread. Readers share the tables,
    /// which go (with their memory) once the last chunk's reader is done.
    Memory { mem: Arc<InMemory>, next: usize },
    /// The unread chunks are being written to spill files.
    Spilling,
    /// On disk, in order: each reader takes the next file, so several read
    /// at once; `active` readers have a file open.
    Files {
        files: VecDeque<SpillFile>,
        active: usize,
    },
    /// Every chunk was claimed.
    Done,
    /// A spill lost the unread chunks: every reader fails with the error
    /// (never a short result).
    Failed(KrishivError),
}

/// What a reader got from [`Output::claim`].
enum Claim {
    Chunk(Arc<InMemory>, usize),
    /// A file, its reader's frames reserved (`true`) or to be reserved
    /// through arbitration (`false`: the only reader).
    File(SpillFile, bool),
    Wait,
    Done,
    Failed(KrishivError),
}

/// The finalized output with its spill hook: an in-memory result is
/// spillable until read (see [`crate::output_spill`]).
struct Output {
    state: Mutex<Out>,
    /// Notified when a spill of the unread chunks ends.
    changed: Event,
    /// The hook and the writer of the unread chunks (with a spill
    /// directory only).
    spill: Option<(Arc<OutputSpill>, Spiller)>,
    executor: Arc<dyn Executor>,
    #[cfg(test)]
    gate: Option<Arc<crate::testing::BlockingGate>>,
}

impl Output {
    fn on_disk(file: Option<SpillFile>, executor: Arc<dyn Executor>) -> Self {
        Self {
            state: Mutex::new(Out::Files {
                files: file.into_iter().collect(),
                active: 0,
            }),
            changed: Event::default(),
            spill: None,
            executor,
            #[cfg(test)]
            gate: None,
        }
    }

    fn set_spillable(&self, bytes: usize) {
        if let Some((hook, _)) = &self.spill {
            hook.set_spillable_bytes(bytes);
        }
    }

    /// The next part for a reader that has just read a file to its end
    /// (`finished`) or has none open; a file's reader frames come from
    /// `consumer`.
    fn claim(&self, finished: bool, consumer: &Arc<ConsumerHandle>) -> Claim {
        let mut state = lock(&self.state);
        let (mem, chunk) = match &mut *state {
            Out::Memory { mem, next } => {
                *next += 1;
                (Arc::clone(mem), *next - 1)
            }
            Out::Spilling => return Claim::Wait,
            Out::Files { files, active } => {
                if finished {
                    *active = active.saturating_sub(1);
                }
                let Some(file) = files.pop_front() else {
                    return Claim::Done;
                };
                if *active == 0 {
                    // The only reader: it may wait for its frames.
                    *active = 1;
                    return Claim::File(file, false);
                }
                // Another reader only if its frames fit now: a reader that
                // has a file open reads this one otherwise (it claims the
                // next file under this lock when it ends), so nothing is
                // left unread and no reader waits for memory another holds.
                let need = file.max_read_reservation();
                let mut frames = consumer.reservation();
                if frames.try_grow(need).is_err() {
                    files.push_front(file);
                    return Claim::Done;
                }
                frames.shrink_to_credit(need);
                *active += 1;
                return Claim::File(file, true);
            }
            Out::Done => return Claim::Done,
            Out::Failed(e) => return Claim::Failed(e.clone()),
        };
        if chunk + 1 >= mem.chunks.len() {
            // The last chunk: the readers' clones are all that is left.
            *state = Out::Done;
            self.set_spillable(0);
        }
        if chunk < mem.chunks.len() {
            Claim::Chunk(mem, chunk)
        } else {
            Claim::Done
        }
    }
}

impl SpillUnread for Output {
    fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>> {
        Box::pin(async move {
            let Some((_, s)) = &self.spill else {
                return Ok(0);
            };
            let (mem, from) = {
                let mut state = lock(&self.state);
                match std::mem::replace(&mut *state, Out::Spilling) {
                    Out::Memory { mem, next } => (mem, next),
                    other => {
                        *state = other;
                        return Ok(0);
                    }
                }
            };
            self.set_spillable(0);
            // Dropped before the spill ends (the query was cancelled): the
            // unread chunks went with the blocking job, so readers fail
            // rather than wait on `Spilling`.
            let unfinished =
                Unfinished(
                    Some(&self.state),
                    &self.changed,
                    || Out::Failed(cancelled()),
                );
            let freed = mem.bytes();
            let s = s.clone();
            // One file per group of the unread chunks, up to the readers'
            // parallelism: readers go on in parallel from disk.
            let groups = ranges(from..mem.chunks.len(), self.executor.parallelism());
            let query = Arc::clone(&s.query);
            let guard = query.blocking_started();
            #[cfg(test)]
            let gate = self.gate.clone();
            let written = run_blocking(&*self.executor, move || {
                #[cfg(test)]
                if let Some(g) = &gate {
                    g.pass();
                }
                let mut files = VecDeque::with_capacity(groups.len());
                let mut failed = None;
                for chunks in groups {
                    match s.write_chunks(&mem, chunks) {
                        Ok(f) => files.push_back(f),
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                }
                // On success the tables go here (or with the last reader's
                // clone); a failed write leaves them as they were. The
                // guard travels with the result, so the query does not end
                // before what it holds is dropped.
                let out = match failed {
                    None => Ok(files),
                    Some(e) => Err((e, mem)),
                };
                Ok((out, guard))
            })
            .await;
            unfinished.disarm();
            let (next, result) = match written {
                Ok((Ok(files), _guard)) => (Out::Files { files, active: 0 }, Ok(freed)),
                Ok((Err((e, mem)), _guard)) => {
                    // The tables are intact: readers go on from memory.
                    self.set_spillable(mem.bytes());
                    (Out::Memory { mem, next: from }, refused(e))
                }
                Err(e) => {
                    // The unread chunks are lost: fail the query before
                    // any reader can see the state.
                    query.fail(e.clone());
                    (Out::Failed(e.clone()), Err(e))
                }
            };
            *lock(&self.state) = next;
            self.changed.notify();
            result
        })
    }
}

/// `chunks` cut into at most `parts` contiguous ranges of near-equal size.
fn ranges(chunks: std::ops::Range<usize>, parts: usize) -> Vec<std::ops::Range<usize>> {
    let n = chunks.len();
    let parts = parts.clamp(1, n.max(1));
    (0..parts)
        .map(|i| chunks.start + i * n / parts..chunks.start + (i + 1) * n / parts)
        .filter(|r| !r.is_empty())
        .collect()
}

struct Global {
    seed: HashSeed,
    /// Row tables (`krishiv.exec.agg.rowLayout` with fixed-width keys):
    /// the thread-local tables' index mode.
    rows: Option<Mode>,
    /// Radix bits of the thread-local row tables.
    row_bits: u32,
    /// Where tables count probe lengths (a detailed profile only).
    probes: Option<Arc<ProbeTotals>>,
    /// Bytes of the combined tables, before `finalize`.
    combined: AtomicUsize,
    inner: Mutex<Inner>,
    done: OnceLock<Final>,
}

fn downcast<T: 'static>(state: &mut LocalState) -> Result<&mut T> {
    state
        .downcast_mut::<T>()
        .ok_or_else(|| internal("wrong local state for HashAggregate".to_owned()))
}

fn global(state: &GlobalState) -> Result<&Global> {
    state
        .downcast_ref::<Global>()
        .ok_or_else(|| internal("wrong global state for HashAggregate".to_owned()))
}

/// Grows `r` to `want` bytes; `Err(missing)` if the pool refuses.
fn reserve_to(r: &mut Reservation, want: usize) -> Result<(), usize> {
    let need = want.saturating_sub(r.size());
    if need > 0 && r.try_grow(need).is_err() {
        return Err(need);
    }
    Ok(())
}

fn shrink_to(r: &mut Reservation, keep: usize) {
    r.shrink(r.size().saturating_sub(keep));
}

/// Releases the headroom of `r` for a spill file's scratch memory (see
/// [`HEADROOM`]); the writer or reader of the same consumer reserves it
/// right after.
fn lend_headroom(r: &mut Reservation) -> usize {
    let lent = HEADROOM.min(r.size());
    r.shrink(lent);
    lent
}

/// Takes back `lent` bytes of headroom after spill I/O, if the pool still
/// has them (the next `reserve_to` asks again otherwise).
fn regrow(r: &mut Reservation, lent: usize) {
    if r.try_grow(lent).is_err() {
        // Not fatal: the headroom is requested again with the next batch.
    }
}

fn no_memory(bytes: usize) -> KrishivError {
    UnableToAcquireMemory {
        requested_bytes: bytes.to_string(),
        received_bytes: "0".to_owned(),
    }
    .into()
}

/// Applies a finished blocking job's results to the finalize state.
type Apply = Box<dyn FnOnce(&mut Inner) -> Result<()> + Send>;
/// Blocking work of a finalize step (run outside the lock).
type Job = Box<dyn FnOnce() -> Result<Apply> + Send>;

/// One finalize step's outcome.
enum Step {
    Done,
    Yield,
    /// Await this many bytes from arbitration (unspillable merge).
    Grow(usize),
    Job(Job),
}

/// Bytes the finalize state holds in tables.
fn held(inner: &Inner) -> usize {
    inner.tables.iter().map(|(_, r)| r.size()).sum::<usize>()
        + inner.current.as_ref().map_or(0, |c| c.reservation.size())
}

/// Makes the per-partition tasks of `runs` at `level`.
fn tasks_of(runs: Vec<Run>, level: u32, tasks: &mut Vec<(u32, Vec<SpillFile>)>) {
    let mut by_part: Vec<Vec<SpillFile>> = (0..PARTITIONS).map(|_| Vec::new()).collect();
    for run in runs {
        for (p, f) in run.0.into_iter().enumerate() {
            if let Some(f) = f {
                by_part[p].push(f);
            }
        }
    }
    tasks.extend(
        by_part
            .into_iter()
            .filter(|f| !f.is_empty())
            .map(|f| (level, f)),
    );
}

impl HashAggregate {
    /// A blocking job that frees the in-memory finalize state, if any: spill
    /// the tables (in-memory phase) or split the current partition.
    fn free_job(&self, g: &Global, inner: &mut Inner, ctx: &TaskContext) -> Result<Option<Job>> {
        let Some(s) = self.spiller(ctx) else {
            return Ok(None);
        };
        if !inner.disk {
            inner.busy = true;
            let tables = std::mem::take(&mut inner.tables);
            let merged = std::mem::take(&mut inner.merged);
            return Ok(Some(Box::new(move || {
                let mut runs = Vec::new();
                for (i, (t, mut r)) in tables.into_iter().enumerate() {
                    lend_headroom(&mut r);
                    runs.push(s.write_run(&t, if i == 1 { merged } else { 0 }, 0)?);
                }
                Ok(Box::new(move |inner: &mut Inner| {
                    inner.disk = true;
                    // Runs merge in the order their rows arrived: the
                    // drivers' earlier spills, then the tables.
                    let mut all = std::mem::take(&mut inner.runs);
                    all.extend(runs);
                    tasks_of(all, 0, &mut inner.tasks);
                    Ok(())
                }) as Apply)
            })));
        }
        let Some(cur) = inner.current.as_mut() else {
            return Ok(None);
        };
        if cur.split.is_some() || cur.table.not_worth_spilling() {
            return Ok(None);
        }
        if cur.level >= MAX_LEVEL {
            return Err(no_memory(cur.reservation.size()));
        }
        inner.busy = true;
        let fresh = self.new_table(g, false)?;
        let table = std::mem::replace(&mut cur.table, fresh);
        let mut held = std::mem::replace(&mut cur.reservation, s.consumer.reservation());
        let level = cur.level + 1;
        bump(&self.counters.repartitions, 1);
        Ok(Some(Box::new(move || {
            lend_headroom(&mut held);
            let run = s.write_run(&table, 0, level)?;
            drop((table, held));
            Ok(Box::new(move |inner: &mut Inner| {
                if let Some(cur) = inner.current.as_mut() {
                    cur.split = Some((vec![run], (0..PARTITIONS).map(|_| None).collect()));
                }
                Ok(())
            }) as Apply)
        })))
    }

    /// A job to free memory, or the error that none can be freed.
    fn must_free(
        &self,
        g: &Global,
        inner: &mut Inner,
        ctx: &TaskContext,
        want: usize,
    ) -> Result<Step> {
        self.free_job(g, inner, ctx)?
            .map(Step::Job)
            .ok_or_else(|| no_memory(want))
    }

    /// One bounded unit of `finalize` work, under the state lock.
    fn step(&self, g: &Global, ctx: &TaskContext) -> Result<Step> {
        let mut guard = lock(&g.inner);
        let inner = &mut *guard;
        let step = if inner.disk {
            self.disk_step(g, inner, ctx)?
        } else {
            self.memory_step(g, inner, ctx)?
        };
        ctx.consumer()
            .signal()
            .set_spillable_bytes(if self.core.spec.spill.is_some() {
                held(inner)
            } else {
                0
            });
        Ok(step)
    }

    fn memory_step(&self, g: &Global, inner: &mut Inner, ctx: &TaskContext) -> Result<Step> {
        if !inner.runs.is_empty() {
            return self.must_free(g, inner, ctx, 0);
        }
        let tables = &mut inner.tables;
        if inner.merged == 0
            && let Some(big) = (0..tables.len()).max_by_key(|&i| tables[i].0.len())
        {
            tables.swap(0, big);
        }
        if tables.len() <= 1 {
            let (table, reservation) = match tables.pop() {
                Some(t) => t,
                None => (self.new_table(g, false)?, ctx.consumer().reservation()),
            };
            bump(&self.counters.groups, table.len() as u64);
            bump(&self.counters.hash_entries, table.len() as u64);
            bump(&self.counters.hash_buckets, table.buckets() as u64);
            let mut held = std::mem::take(&mut inner.granted);
            held.push(reservation);
            self.finish_memory(g, ctx, vec![table], held)?;
            return Ok(Step::Done);
        }
        let (dst, rest) = tables.split_at_mut(1);
        let (dst, src) = (&mut dst[0], &rest[0].0);
        if inner.merged >= src.len() {
            tables.remove(1);
            inner.merged = 0;
            return Ok(Step::Yield);
        }
        let end = (inner.merged + CHUNK).min(src.len());
        let ids = src.ids(inner.merged..end);
        let p = dst.0.prepare(&Batch::new(src.state_batch(&ids)?), true)?;
        let granted: usize = inner.granted.iter().map(Reservation::size).sum();
        let want = dst.0.projected(&p)? + dst.0.transient(&p);
        if let Err(need) = reserve_to(&mut dst.1, want.saturating_sub(granted)) {
            if self.core.spec.spill.is_none() {
                return Ok(Step::Grow(need));
            }
            return self.must_free(g, inner, ctx, want);
        }
        dst.0.apply(&p)?;
        inner.merged = end;
        Ok(Step::Yield)
    }

    fn disk_step(&self, g: &Global, inner: &mut Inner, ctx: &TaskContext) -> Result<Step> {
        let s = self
            .spiller(ctx)
            .ok_or_else(|| internal("disk phase without spill".to_owned()))?;
        let Some(cur) = inner.current.as_mut() else {
            let Some((level, mut files)) = inner.tasks.pop() else {
                if let Some(w) = inner.out.take() {
                    inner.busy = true;
                    return Ok(Step::Job(Box::new(move || {
                        let f = s.finish(w)?;
                        Ok(Box::new(move |inner: &mut Inner| {
                            inner.out_file = Some(f);
                            Ok(())
                        }) as Apply)
                    })));
                }
                let out = Output::on_disk(inner.out_file.take(), Arc::clone(ctx.executor()));
                let _ = g.done.set(Final {
                    out: Arc::new(out),
                    chunks: 1,
                });
                return Ok(Step::Done);
            };
            // Files are taken from the back: reverse them so they merge in
            // run order (the order their rows arrived; `first`/`last`).
            files.reverse();
            inner.current = Some(Current {
                level,
                table: self.new_table(g, false)?,
                reservation: ctx.consumer().reservation(),
                files,
                reader: None,
                pending: None,
                split: None,
            });
            return Ok(Step::Yield);
        };
        let Some(batch) = cur.pending.take() else {
            if cur.reader.is_none() && cur.files.is_empty() {
                return self.finish_task(inner, ctx);
            }
            // Open the next file if needed and read a batch, off the worker.
            let reader = cur.reader.take();
            let file = if reader.is_none() {
                cur.files.pop()
            } else {
                None
            };
            let lent = lend_headroom(&mut cur.reservation);
            inner.busy = true;
            return Ok(Step::Job(Box::new(move || {
                let mut reader = match (reader, file) {
                    (Some(r), _) => r,
                    (None, Some(f)) => f
                        .reader(s.consumer.reservation())
                        .map_err(KrishivError::from)?,
                    (None, None) => return Err(internal("no spill file to read".to_owned())),
                };
                let next = reader.next().transpose().map_err(KrishivError::from)?;
                Ok(Box::new(move |inner: &mut Inner| {
                    if let Some(cur) = inner.current.as_mut() {
                        regrow(&mut cur.reservation, lent);
                        cur.reader = next.is_some().then_some(reader);
                        cur.pending = next;
                    }
                    Ok(())
                }) as Apply)
            })));
        };
        if let Some((_, writers)) = cur.split.as_mut() {
            let parts = cur.table.partitions_of(batch.data(), cur.level + 1)?;
            let mut picked = Vec::new();
            for (p, rows) in parts.into_iter().enumerate() {
                if !rows.is_empty() {
                    let b = take_record_batch(batch.data(), &UInt32Array::from(rows))
                        .map_err(|e| internal(e.to_string()))?;
                    picked.push((p, Batch::new(b)));
                }
            }
            let mut writers = std::mem::take(writers);
            let lent = lend_headroom(&mut cur.reservation);
            inner.busy = true;
            return Ok(Step::Job(Box::new(move || {
                for (p, b) in &picked {
                    let w = match &mut writers[*p] {
                        Some(w) => w,
                        slot => slot.insert(s.writer()?),
                    };
                    w.write(b).map_err(KrishivError::from)?;
                }
                Ok(Box::new(move |inner: &mut Inner| {
                    if let Some(cur) = inner.current.as_mut() {
                        regrow(&mut cur.reservation, lent);
                        if let Some((_, slot)) = cur.split.as_mut() {
                            *slot = writers;
                        }
                    }
                    Ok(())
                }) as Apply)
            })));
        }
        let p = cur.table.prepare(&batch, true)?;
        let want = cur.table.projected(&p)? + HEADROOM.max(cur.table.transient(&p));
        if reserve_to(&mut cur.reservation, want).is_err() {
            cur.pending = Some(batch);
            return self.must_free(g, inner, ctx, want);
        }
        cur.table.apply(&p)?;
        Ok(Step::Yield)
    }

    /// The current partition's input is exhausted: emit its groups, or turn
    /// its split into tasks of the next level (on the blocking pool).
    fn finish_task(&self, inner: &mut Inner, ctx: &TaskContext) -> Result<Step> {
        let s = self
            .spiller(ctx)
            .ok_or_else(|| internal("disk phase without spill".to_owned()))?;
        let Some(mut cur) = inner.current.take() else {
            return Ok(Step::Yield);
        };
        inner.busy = true;
        if let Some((mut runs, writers)) = cur.split.take() {
            let level = cur.level + 1;
            drop(cur);
            return Ok(Step::Job(Box::new(move || {
                let mut files: Vec<Option<SpillFile>> = Vec::with_capacity(PARTITIONS);
                for w in writers {
                    files.push(w.map(|w| s.finish(w)).transpose()?);
                }
                runs.push(Run(files));
                Ok(Box::new(move |inner: &mut Inner| {
                    tasks_of(runs, level, &mut inner.tasks);
                    Ok(())
                }) as Apply)
            })));
        }
        let out = inner.out.take();
        Ok(Step::Job(Box::new(move || {
            lend_headroom(&mut cur.reservation);
            let mut out = match out {
                Some(w) => w,
                None => s
                    .scope
                    .writer(Arc::clone(&s.core.output_schema), s.consumer.reservation())
                    .map_err(KrishivError::from)?,
            };
            s.write_output(&cur.table, &mut out)?;
            drop(cur);
            Ok(Box::new(move |inner: &mut Inner| {
                inner.out = Some(out);
                Ok(())
            }) as Apply)
        })))
    }

    /// Publishes an in-memory result: `tables` read in chunks of a batch
    /// (a row table's within one radix partition, whose states are one
    /// partition's arrays), held by `held`.
    fn finish_memory(
        &self,
        g: &Global,
        ctx: &TaskContext,
        tables: Vec<Table>,
        held: Vec<Reservation>,
    ) -> Result<()> {
        let batch = ctx.query().config().batch_size.max(1);
        let mut chunks = Vec::new();
        for (i, t) in tables.iter().enumerate() {
            let mut start = 0;
            for len in t.part_lens() {
                let end = start + len;
                chunks.extend(
                    (start..end)
                        .step_by(batch)
                        .map(|s| (i, s..(s + batch).min(end))),
                );
                start = end;
            }
        }
        self.publish_memory(g, ctx, tables, chunks, held)
    }

    /// Publishes an in-memory result read in `chunks` (table, positions).
    /// With a spill directory the result gets a consumer of its own whose
    /// hook spills the unread chunks on request ([`crate::output_spill`]).
    fn publish_memory(
        &self,
        g: &Global,
        ctx: &TaskContext,
        tables: Vec<Table>,
        chunks: Vec<(usize, std::ops::Range<usize>)>,
        mut held: Vec<Reservation>,
    ) -> Result<()> {
        let spill = match self.spiller(ctx) {
            Some(mut s) => {
                let (hook, consumer) = OutputSpill::register(ctx, "HashAggregate output");
                for r in &mut held {
                    r.transfer_to(&consumer).map_err(KrishivError::from)?;
                }
                s.consumer = consumer;
                Some((hook, s))
            }
            None => None,
        };
        let count = chunks.len();
        let mem = Arc::new(InMemory {
            tables,
            chunks,
            held,
        });
        let bytes = mem.bytes();
        let out = Arc::new(Output {
            state: Mutex::new(Out::Memory { mem, next: 0 }),
            changed: Event::default(),
            spill,
            executor: Arc::clone(ctx.executor()),
            #[cfg(test)]
            gate: self.output_gate.clone(),
        });
        if let Some((hook, _)) = &out.spill {
            let weak: Weak<dyn SpillUnread> = Arc::downgrade(&out) as Weak<Output>;
            hook.attach(weak);
            hook.set_spillable_bytes(bytes);
        }
        let _ = g.done.set(Final { out, chunks: count });
        Ok(())
    }

    async fn run_job(&self, g: &Global, ctx: &TaskContext, job: Job) -> Result<()> {
        let apply = ctx.run_blocking(job).await;
        let mut inner = lock(&g.inner);
        inner.busy = false;
        apply?(&mut inner)
    }
}

impl Sink for HashAggregate {
    fn name(&self) -> &'static str {
        "HashAggregate"
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let config = ctx.query().config();
        let threads = ctx.executor().parallelism();
        let rows = (config.agg_row_layout && self.core.rows.is_some()).then(|| {
            self.row_mode
                .unwrap_or_else(|| Mode::local(config.seed.is_some(), threads))
        });
        if rows.is_some() {
            self.counters
                .row_layout
                .store(true, AtomicOrdering::Relaxed);
        }
        Ok(Arc::new(Global {
            seed: HashSeed::new(config.seed),
            rows,
            row_bits: rows.map_or(0, |m| m.radix_bits(threads)),
            probes: config
                .detailed_metrics
                .then(|| Arc::clone(&self.counters.probes)),
            combined: AtomicUsize::new(0),
            inner: Mutex::new(Inner::default()),
            done: OnceLock::new(),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        Ok(Box::new(Local {
            table: self.new_table(global(g)?, true)?,
            reservation: memory.reservation(),
            signal: Arc::clone(memory.signal()),
            consumer: Arc::clone(memory.consumer()),
            global: Arc::clone(g),
        }))
    }

    fn sink(&self, input: Batch, l: &mut LocalState, _g: &GlobalState) -> Result<SinkStatus> {
        let l = downcast::<Local>(l)?;
        if self.core.spec.require_unit_weights && !unit_weights(input.weights()) {
            return Err(unsupported(
                "DISTINCT over rows with weights other than 1 (needs incremental view maintenance)"
                    .to_owned(),
            ));
        }
        let p = l.table.prepare(&input, false)?;
        // The spill headroom is idle while a batch is applied: the apply's
        // transient bytes may use it.
        let want = l.table.projected(&p)? + self.headroom().max(l.table.transient(&p));
        if let Err(bytes) = reserve_to(&mut l.reservation, want) {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        l.table.apply(&p)?;
        shrink_to(&mut l.reservation, l.table.memory_size() + self.headroom());
        if self.core.spec.spill.is_some() {
            let combined = global(&l.global)?.combined.load(AtomicOrdering::Relaxed);
            l.signal
                .set_spillable_bytes(l.table.memory_size() + combined);
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, l: LocalState, g: &GlobalState) -> Result<()> {
        let l = l
            .downcast::<Local>()
            .map_err(|_| internal("wrong local state for HashAggregate".to_owned()))?;
        let Local {
            table,
            mut reservation,
            signal,
            consumer,
            ..
        } = *l;
        // Keep the headroom: a later spill of this table needs scratch.
        shrink_to(&mut reservation, table.memory_size() + self.headroom());
        let g = global(g)?;
        let mut inner = lock(&g.inner);
        g.combined
            .fetch_add(reservation.size(), AtomicOrdering::Relaxed);
        // Earlier drivers have ended: charge their tables to this one, so the
        // arbitrator sees them as this consumer's spillable bytes (the last
        // driver to combine finalizes and spills through `spill_global`).
        for (_, r) in &mut inner.tables {
            r.transfer_to(&consumer).map_err(KrishivError::from)?;
        }
        inner.tables.push((table, reservation));
        // The last driver's signal stays live through `finalize`.
        let spillable = self.core.spec.spill.is_some();
        signal.set_spillable_bytes(if spillable { held(&inner) } else { 0 });
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        g: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global(g)?;
            {
                // The finalizing driver is not necessarily the last one to
                // combine: charge the tables to its live spill hook.
                let mut inner = lock(&g.inner);
                let to = ctx.consumer().consumer();
                for (_, r) in &mut inner.tables {
                    r.transfer_to(to).map_err(KrishivError::from)?;
                }
            }
            // Row tables hold partial duplicates: they merge partition by
            // partition, in parallel, unless they spilled.
            if g.rows.is_some() && self.finalize_rows(g, ctx).await? {
                return Ok(());
            }
            loop {
                ctx.query().check_cancelled()?;
                match self.step(g, ctx)? {
                    Step::Done => return Ok(()),
                    Step::Grow(bytes) => {
                        let mut r = ctx.consumer().reservation();
                        r.grow(bytes).await.map_err(KrishivError::from)?;
                        lock(&g.inner).granted.push(r);
                    }
                    Step::Yield => tokio::task::yield_now().await,
                    Step::Job(job) => self.run_job(g, ctx, job).await?,
                }
            }
        })
    }

    fn spill<'a>(
        &'a self,
        l: &'a mut LocalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = downcast::<Local>(l)?;
            let Some(s) = self.spiller(ctx) else {
                return Ok(0);
            };
            let g = global(&l.global)?;
            let combined = {
                let mut inner = lock(&g.inner);
                g.combined.store(0, AtomicOrdering::Relaxed);
                std::mem::take(&mut inner.tables)
            };
            let freed_combined: usize = combined.iter().map(|(_, r)| r.size()).sum();
            let own = if l.table.not_worth_spilling() {
                None
            } else {
                let fresh = self.new_table(g, true)?;
                Some(std::mem::replace(&mut l.table, fresh))
            };
            if own.is_none() && combined.is_empty() {
                return Ok(0);
            }
            // The table's reservation goes with it; a table kept (nothing
            // worth spilling) keeps its own.
            let mut held = own
                .as_ref()
                .map(|_| std::mem::replace(&mut l.reservation, s.consumer.reservation()));
            let freed_own = held.as_ref().map_or(0, Reservation::size);
            let runs = ctx
                .run_blocking(move || {
                    let mut runs = Vec::new();
                    if let Some(t) = &own {
                        if let Some(h) = held.as_mut() {
                            lend_headroom(h);
                        }
                        runs.push(s.write_run(t, 0, 0)?);
                    }
                    drop((own, held));
                    for (t, mut r) in combined {
                        lend_headroom(&mut r);
                        runs.push(s.write_run(&t, 0, 0)?);
                    }
                    Ok(runs)
                })
                .await?;
            lock(&g.inner).runs.extend(runs);
            l.signal.set_spillable_bytes(0);
            Ok(freed_own + freed_combined)
        })
    }

    fn spill_global<'a>(
        &'a self,
        g: &'a GlobalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let g = global(g)?;
            let (job, before) = {
                let mut inner = lock(&g.inner);
                if inner.row_merge {
                    // Freed by the merge itself, between its tasks.
                    inner.spill_asked = true;
                    return Ok(0);
                }
                if inner.busy || g.done.get().is_some() {
                    return Ok(0);
                }
                let before = held(&inner);
                match self.free_job(g, &mut inner, ctx)? {
                    Some(job) => (job, before),
                    None => return Ok(0),
                }
            };
            self.run_job(g, ctx, job).await?;
            let after = held(&lock(&g.inner));
            ctx.consumer().signal().set_spillable_bytes(after);
            Ok(before.saturating_sub(after))
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        let m = self.metrics();
        let c = &self.counters;
        let mut out = crate::metrics::counts([
            ("groups", m.groups),
            ("spill_runs", m.spill_runs),
            ("spill_files", m.spill_files),
            ("spill_bytes", m.spill_bytes),
            ("repartitions", m.repartitions),
            ("hash_entries", c.hash_entries.load(AtomicOrdering::Relaxed)),
            ("hash_buckets", c.hash_buckets.load(AtomicOrdering::Relaxed)),
        ]);
        out.push((
            std::borrow::Cow::Borrowed("hash_index"),
            crate::metrics::Metric::Text("swiss".to_owned()),
        ));
        if c.row_layout.load(AtomicOrdering::Relaxed) {
            out.extend(crate::metrics::counts([
                ("row_abandons", c.row_abandons.load(AtomicOrdering::Relaxed)),
                (
                    "row_final_tasks",
                    c.row_final_tasks.load(AtomicOrdering::Relaxed),
                ),
                (
                    "row_final_spills",
                    c.row_final_spills.load(AtomicOrdering::Relaxed),
                ),
                (
                    "row_final_flushes",
                    c.row_final_flushes.load(AtomicOrdering::Relaxed),
                ),
            ]));
            out.push((
                std::borrow::Cow::Borrowed("table_layout"),
                crate::metrics::Metric::Text("rows".to_owned()),
            ));
        }
        out.extend(crate::metrics::probe_metrics(&c.probes.get()));
        out
    }
}

pub(super) fn unit_weights(w: &Weights) -> bool {
    match w {
        Weights::Const(c) => *c == 1,
        Weights::Column(c) => c.values().iter().all(|&v| v == 1),
    }
}

/// The source side's per-driver state.
struct SourceLocal {
    consumer: Arc<ConsumerHandle>,
    reader: Option<SpillReader>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
}

impl Source for HashAggregate {
    fn name(&self) -> &'static str {
        "HashAggregate"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::clone(&self.core.output_schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let state = ctx.dependency().cloned();
        let state =
            state.ok_or_else(|| internal("aggregate read before it was built".to_owned()))?;
        global(&state)?;
        Ok(state)
    }

    fn max_parallelism(&self, g: &GlobalState) -> usize {
        global(g)
            .ok()
            .and_then(|g| g.done.get())
            .map_or(1, |f| f.chunks.max(1))
    }

    fn local_state(&self, ctx: &TaskContext, _g: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(SourceLocal {
            consumer: Arc::clone(ctx.consumer().consumer()),
            reader: None,
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
        }))
    }

    fn get_data<'a>(
        &'a self,
        l: &'a mut LocalState,
        g: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let l = downcast::<SourceLocal>(l)?;
            let g = global(g)?;
            let out = &g
                .done
                .get()
                .ok_or_else(|| internal("aggregate read before it finalized".to_owned()))?
                .out;
            // An open file first; once read, claim the next one.
            let mut finished = false;
            if l.reader.is_some() {
                match read_file(l, None, true).await? {
                    Some(b) => return Ok(Some(b)),
                    None => finished = true,
                }
            }
            loop {
                let changed = out.changed.listen();
                match out.claim(std::mem::take(&mut finished), &l.consumer) {
                    Claim::Chunk(mem, chunk) => return mem.batch(chunk).map(Some),
                    Claim::Done => return Ok(None),
                    Claim::Failed(e) => return Err(e),
                    Claim::File(file, reserved) => {
                        match read_file(l, Some(file), reserved).await? {
                            Some(b) => return Ok(Some(b)),
                            None => finished = true,
                        }
                    }
                    // A spill of the unread chunks is being written.
                    Claim::Wait => changed.await,
                }
            }
        })
    }
}

/// Reads the next batch of the output file: `file` (just taken; its
/// reader's frames `reserved` already, or not) or the reader already open.
async fn read_file(
    l: &mut SourceLocal,
    file: Option<SpillFile>,
    reserved: bool,
) -> Result<Option<Batch>> {
    if let Some(f) = file.as_ref().filter(|_| !reserved) {
        // The reader's frames, reserved through the arbitrator before it
        // opens and cycled through the consumer's credit: a read is never
        // refused.
        let need = f.max_read_reservation();
        let mut lent = l.consumer.reservation();
        lent.grow(need).await.map_err(KrishivError::from)?;
        lent.shrink_to_credit(need);
    }
    let reader = l.reader.take();
    let consumer = Arc::clone(&l.consumer);
    // File I/O runs on the blocking pool, never on a worker.
    let guard = l.query.blocking_started();
    let (reader, next) = run_blocking(&*l.executor, move || {
        let _guard = guard;
        let mut r = match (reader, file) {
            (Some(r), _) => r,
            (None, Some(f)) => f
                .reader(consumer.reservation().releasing_to_credit())
                .map_err(KrishivError::from)?,
            (None, None) => return Err(internal("no output file".to_owned())),
        };
        let next = r.next().transpose().map_err(KrishivError::from)?;
        Ok((r, next))
    })
    .await?;
    if next.is_some() {
        l.reader = Some(reader);
    } else {
        drop(reader);
        l.consumer.release_credit();
    }
    Ok(next)
}
