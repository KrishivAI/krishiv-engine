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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

use arrow_array::UInt32Array;
use arrow_select::take::take_record_batch;
use futures::future::BoxFuture;
use krishiv_common::error::{KrishivError, UnableToAcquireMemory};
use krishiv_memory::{
    ConsumerHandle, Reservation, SpillFile, SpillReader, SpillScope, SpillSignal, SpillWriter,
};
use krishiv_types::batch::{Batch, Weights};

use super::table::{AggCore, AggTable};
use super::{AggregateSpec, unsupported};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::executor::{Executor, run_blocking};
use crate::hash::{HashSeed, MAX_LEVEL, PARTITIONS};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::sync::{Mutex, lock};

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
    spill_runs: AtomicU64,
    spill_files: AtomicU64,
    spill_bytes: AtomicU64,
    repartitions: AtomicU64,
}

fn bump(c: &AtomicU64, n: u64) {
    c.fetch_add(n, AtomicOrdering::Relaxed);
}

/// A hash-aggregation pipeline breaker (see the [module docs](super)).
#[derive(Debug)]
pub struct HashAggregate {
    core: Arc<AggCore>,
    counters: Arc<Counters>,
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
    fn write_run(&self, table: &AggTable, from: usize, level: u32) -> Result<Run> {
        let mut files: Vec<Option<SpillFile>> = (0..PARTITIONS).map(|_| None).collect();
        for (p, ids) in table.ids_by_partition(level).into_iter().enumerate() {
            let ids: Vec<u32> = ids.into_iter().filter(|&i| i as usize >= from).collect();
            if ids.is_empty() {
                continue;
            }
            let mut writer = self.run_writer()?;
            for chunk in ids.chunks(spill_chunk(table)) {
                self.query.check_cancelled()?;
                let b = Batch::new(table.state_batch(chunk)?);
                writer.write(&b).map_err(KrishivError::from)?;
            }
            files[p] = Some(self.finish(writer)?);
        }
        bump(&self.counters.spill_runs, 1);
        Ok(Run(files))
    }

    /// Writes the final output of `table` to `out` (blocking I/O).
    fn write_output(&self, table: &AggTable, out: &mut SpillWriter) -> Result<()> {
        let n = u32::try_from(table.len()).map_err(|_| internal("too many groups".to_owned()))?;
        let ids: Vec<u32> = (0..n).collect();
        for chunk in ids.chunks(self.batch_size) {
            self.query.check_cancelled()?;
            let b = Batch::new(table.output_batch(chunk)?);
            out.write(&b).map_err(KrishivError::from)?;
        }
        bump(&self.counters.groups, u64::from(n));
        Ok(())
    }
}

/// Groups per spilled batch, so that encoding one fits [`HEADROOM`]: a
/// table's bytes per group bound its state's.
fn spill_chunk(table: &AggTable) -> usize {
    let per_group = table.memory_size() / table.len().max(1);
    ((HEADROOM - (64 << 10)) / 3 / per_group.max(1)).clamp(16, CHUNK)
}

/// One spill: a file per hash partition.
#[derive(Debug)]
struct Run(Vec<Option<SpillFile>>);

struct Local {
    table: AggTable,
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
    table: AggTable,
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
    tables: Vec<(AggTable, Reservation)>,
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
}

/// The finalized output, read by the source side.
enum Final {
    /// In memory; the reservation is held as long as the table.
    Table {
        table: AggTable,
        _held: Vec<Reservation>,
    },
    /// On disk; the source takes the file when it starts reading.
    File(Mutex<Option<SpillFile>>),
}

struct Global {
    seed: HashSeed,
    /// Bytes of the combined tables, before `finalize`.
    combined: AtomicUsize,
    inner: Mutex<Inner>,
    done: OnceLock<Final>,
    /// The next output chunk (source side of an in-memory result).
    cursor: AtomicUsize,
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
    fn free_job(&self, inner: &mut Inner, ctx: &TaskContext) -> Result<Option<Job>> {
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
                    runs.extend(std::mem::take(&mut inner.runs));
                    tasks_of(runs, 0, &mut inner.tasks);
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
        let fresh = AggTable::new(&self.core, cur.table.seed())?;
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
    fn must_free(&self, inner: &mut Inner, ctx: &TaskContext, want: usize) -> Result<Step> {
        self.free_job(inner, ctx)?
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
            return self.must_free(inner, ctx, 0);
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
                None => (
                    AggTable::new(&self.core, &g.seed)?,
                    ctx.consumer().reservation(),
                ),
            };
            bump(&self.counters.groups, table.len() as u64);
            let mut held = std::mem::take(&mut inner.granted);
            held.push(reservation);
            let _ = g.done.set(Final::Table { table, _held: held });
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
        let ids: Vec<u32> = (inner.merged..end)
            .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
            .collect();
        let p = dst.0.prepare(&Batch::new(src.state_batch(&ids)?), true)?;
        let granted: usize = inner.granted.iter().map(Reservation::size).sum();
        let want = dst.0.projected(&p);
        if let Err(need) = reserve_to(&mut dst.1, want.saturating_sub(granted)) {
            if self.core.spec.spill.is_none() {
                return Ok(Step::Grow(need));
            }
            return self.must_free(inner, ctx, want);
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
            let Some((level, files)) = inner.tasks.pop() else {
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
                let file = Mutex::new(inner.out_file.take());
                let _ = g.done.set(Final::File(file));
                return Ok(Step::Done);
            };
            inner.current = Some(Current {
                level,
                table: AggTable::new(&self.core, &g.seed)?,
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
            let parts = self
                .core
                .partitions_of(batch.data(), &g.seed, cur.level + 1)?;
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
        let want = cur.table.projected(&p) + HEADROOM;
        if reserve_to(&mut cur.reservation, want).is_err() {
            cur.pending = Some(batch);
            return self.must_free(inner, ctx, want);
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
        Ok(Arc::new(Global {
            seed: HashSeed::new(ctx.query().config().seed),
            combined: AtomicUsize::new(0),
            inner: Mutex::new(Inner::default()),
            done: OnceLock::new(),
            cursor: AtomicUsize::new(0),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        Ok(Box::new(Local {
            table: AggTable::new(&self.core, &global(g)?.seed)?,
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
        let want = l.table.projected(&p) + self.headroom();
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
                let fresh = AggTable::new(&self.core, l.table.seed())?;
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
                if inner.busy || g.done.get().is_some() {
                    return Ok(0);
                }
                let before = held(&inner);
                match self.free_job(&mut inner, ctx)? {
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

    fn extra_metrics(&self) -> Vec<(&'static str, u64)> {
        let m = self.metrics();
        vec![
            ("groups", m.groups),
            ("spill_runs", m.spill_runs),
            ("spill_files", m.spill_files),
            ("spill_bytes", m.spill_bytes),
            ("repartitions", m.repartitions),
        ]
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
    batch_size: usize,
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
        match global(g).ok().and_then(|g| g.done.get()) {
            Some(Final::Table { table, .. }) => table.len().div_ceil(DEFAULT_CHUNK_ROWS).max(1),
            _ => 1,
        }
    }

    fn local_state(&self, ctx: &TaskContext, _g: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(SourceLocal {
            consumer: Arc::clone(ctx.consumer().consumer()),
            reader: None,
            batch_size: ctx.query().config().batch_size.max(1),
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
            match g.done.get() {
                Some(Final::Table { table: t, .. }) => {
                    let start = g.cursor.fetch_add(l.batch_size, AtomicOrdering::Relaxed);
                    if start >= t.len() {
                        return Ok(None);
                    }
                    let end = (start + l.batch_size).min(t.len());
                    let ids: Vec<u32> = (start..end)
                        .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
                        .collect();
                    Ok(Some(Batch::new(t.output_batch(&ids)?)))
                }
                Some(Final::File(file)) => {
                    let file = if l.reader.is_none() {
                        match lock(file).take() {
                            Some(f) => Some(f),
                            None => return Ok(None),
                        }
                    } else {
                        None
                    };
                    if let Some(f) = &file {
                        // The reader's frames, reserved through the
                        // arbitrator before it opens and cycled through the
                        // consumer's credit: a read is never refused.
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
                None => Err(internal("aggregate read before it finalized".to_owned())),
            }
        })
    }
}

/// Output rows per source driver claim, for the parallelism estimate.
const DEFAULT_CHUNK_ROWS: usize = crate::context::DEFAULT_BATCH_SIZE;
