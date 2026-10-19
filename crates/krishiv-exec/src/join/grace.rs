// SPDX-License-Identifier: Apache-2.0
//! Grace hash join: spilled partitions, joined after every probe driver's
//! input ended, sharing them out through [`Grace`]. A partition whose build
//! rows do not fit is repartitioned on the next hash bits (build and probe
//! alike) up to [`MAX_DEPTH`] levels deep.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, UInt32Array};
use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivJoinPartitionTooLarge;
use krishiv_memory::{
    ConsumerId, Reservation, SpillError, SpillFile, SpillReader, SpillScope, SpillWriter,
};
use krishiv_types::batch::Batch;

use super::build::BuildGlobal;
use super::keys::{MAX_DEPTH, PARTITIONS, partition};
use super::probe::{Cursor, Tables};
use super::table::{JoinTable, RowArena, key_bytes};
use super::{Counter, Shared, take_batch};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::hash::HashSeed;
use crate::operator::FlushOutput;
use crate::sync::lock;

/// Rows of one partition: chunks in memory (reserved in `memory`) and
/// spill files.
#[derive(Debug, Default)]
pub(crate) struct Part {
    pub(crate) chunks: Vec<Batch>,
    pub(crate) files: Vec<SpillFile>,
    pub(crate) memory: Vec<Reservation>,
    /// Rows in `chunks` and `files`.
    pub(crate) rows: u64,
}

impl Part {
    pub(crate) fn with_memory(reservation: Reservation) -> Self {
        Self {
            memory: vec![reservation],
            ..Self::default()
        }
    }

    /// Bytes reserved for the chunks.
    pub(crate) fn bytes(&self) -> usize {
        self.memory.iter().map(Reservation::size).sum()
    }

    /// What reading the files back reserves at most at once (exact, see
    /// [`SpillFile::max_read_reservation`]); chunks are reserved already.
    pub(crate) fn read_headroom(&self) -> usize {
        let files = self.files.iter().map(SpillFile::max_read_reservation);
        files.max().unwrap_or(0)
    }

    pub(crate) fn mem_rows(&self) -> usize {
        self.chunks.iter().map(Batch::num_rows).sum()
    }

    /// Merges runs of small in-memory chunks (each chunk carries fixed array
    /// overhead), if each merged copy can be reserved first.
    ///
    /// Size-tiered, like an LSM tree: a chunk's tier is `log16` of its bytes,
    /// and only a trailing run of at least `FANOUT` chunks no higher in tier
    /// than the newest one is merged. Merging every chunk held whenever the
    /// count reached the fan-out copied the whole partition again every 15
    /// pushes (quadratic in its size); tiered, a row is copied about once
    /// per tier it climbs.
    pub(crate) fn coalesce(&mut self, schema: &SchemaRef) -> Result<()> {
        const FANOUT: usize = 16;
        let size = |c: &Batch| c.data().get_array_memory_size();
        let tier = |c: &Batch| size(c).max(1).ilog(FANOUT);
        while self.chunks.len() >= FANOUT {
            let Some(top) = self.chunks.last().map(tier) else {
                return Ok(());
            };
            let run = self
                .chunks
                .iter()
                .rev()
                .take_while(|c| tier(c) <= top)
                .count();
            if run < FANOUT {
                return Ok(());
            }
            let from = self.chunks.len() - run;
            let old: usize = self.chunks[from..].iter().map(size).sum();
            let Some(m) = self.memory.first_mut().filter(|m| m.size() >= old) else {
                return Ok(());
            };
            if m.try_grow(old).is_err() {
                return Ok(());
            }
            let merged =
                Batch::concat(schema, &self.chunks[from..]).map_err(|e| internal(e.to_string()));
            let merged = match merged {
                Ok(b) if size(&b) <= old => b,
                other => {
                    m.shrink(old);
                    return other.map(drop);
                }
            };
            m.shrink(old + old - size(&merged));
            self.chunks.truncate(from);
            self.chunks.push(merged);
        }
        Ok(())
    }

    pub(crate) fn absorb(&mut self, other: Self) {
        self.chunks.extend(other.chunks);
        self.files.extend(other.files);
        self.memory.extend(other.memory);
        self.rows += other.rows;
    }
}

/// Writes the in-memory chunks of the `chosen` parts to one new spill file
/// each. Unless it `must`, it declines (returns `(0, 0)`, nothing changed)
/// when the rows plus the pool's free memory cannot fund the writers'
/// scratch; one that must write takes what is missing from the pool as
/// each write needs it. Their memory is released **first**: a writer needs scratch (its
/// buffer and an encoded frame), and with the pool full that can only come
/// from the bytes being spilled. The chunks stay in RAM, unaccounted, only
/// while they are written. Returns the bytes freed and written.
pub(crate) async fn spill_parts(
    parts: &mut [Part],
    chosen: &[usize],
    schema: &SchemaRef,
    scope: &SpillScope,
    ctx: &TaskContext,
    must: bool,
) -> Result<(usize, u64)> {
    // The scratch the rows cannot fund comes from the pool now, or the
    // spill is declined (frees nothing) rather than failing mid-write.
    let held: usize = chosen
        .iter()
        .filter_map(|&p| parts.get(p).filter(|part| !part.chunks.is_empty()))
        .map(Part::bytes)
        .sum();
    let mut topup = ctx.consumer().reservation();
    let funded = topup.try_grow(SPILL_SCRATCH.saturating_sub(held)).is_ok();
    if held == 0 || !(funded || must) {
        // Scratch a caller lent is not needed.
        ctx.consumer().consumer().release_credit();
        return Ok((0, 0));
    }
    topup.shrink_to_credit(topup.size());
    let mut jobs = Vec::new();
    let mut freed = 0;
    let own = ctx.consumer().id();
    for &p in chosen {
        let Some(part) = parts.get_mut(p).filter(|part| !part.chunks.is_empty()) else {
            continue;
        };
        freed += part.bytes();
        for m in &mut part.memory {
            // The bytes become the writers' scratch, as our grant credit.
            // Another driver's (combined into a breaker) are re-charged to
            // us first, without passing through the pool.
            let bytes = m.size();
            if m.consumer() != own && m.transfer_to(ctx.consumer().consumer()).is_err() {
                m.shrink(bytes);
                continue;
            }
            m.shrink_to_credit(bytes);
        }
        jobs.push((
            p,
            std::mem::take(&mut part.chunks),
            ctx.consumer().reservation().releasing_to_credit(),
        ));
    }
    if jobs.is_empty() {
        ctx.consumer().consumer().release_credit();
        return Ok((0, 0));
    }
    let (scope, schema) = (scope.clone(), Arc::clone(schema));
    let files = ctx
        .run_blocking(move || {
            let mut files = Vec::with_capacity(jobs.len());
            for (p, chunks, scratch) in jobs {
                let mut w = scope.writer(Arc::clone(&schema), scratch)?;
                for c in &chunks {
                    write_sliced(&mut w, c)?;
                }
                drop(chunks);
                files.push((p, w.finish()?));
            }
            Ok(files)
        })
        .await;
    // Credit the writers did not take returns to the pool.
    ctx.consumer().consumer().release_credit();
    let files = files?;
    let mut written = 0;
    for (p, file) in files {
        written += file.bytes();
        if let Some(part) = parts.get_mut(p) {
            part.files.push(file);
        }
    }
    Ok((freed, written))
}

/// Writes `batch` in slices of about [`WRITE_SLICE`] bytes: a write's
/// scratch is twice the slice plus the writer's buffer, so it stays small
/// whatever the chunk size.
fn write_sliced(w: &mut SpillWriter, batch: &Batch) -> Result<()> {
    let (rows, bytes) = (batch.num_rows(), batch.data().get_array_memory_size());
    let per = (rows * WRITE_SLICE / bytes.max(1)).clamp(1, rows.max(1));
    for start in (0..rows).step_by(per) {
        let slice = batch
            .slice(start, per.min(rows - start))
            .map_err(|e| internal(e.to_string()))?;
        w.write(&slice)?;
    }
    Ok(())
}

/// Bytes per spill write.
pub(crate) const WRITE_SLICE: usize = 64 << 10;

/// A spill writer's scratch for one write of a [`WRITE_SLICE`] slice: its
/// 64 KiB buffer plus twice the slice it encodes.
pub(crate) const SPILL_SCRATCH: usize = 3 * WRITE_SLICE;

/// The least a spill writes at once when it can: a spill requested by the
/// arbitrator funds its writer's scratch from the bytes it spills (see
/// [`spill_parts`]), so writing at least the scratch keeps it off the pool.
pub(crate) const SPILL_MIN: usize = 256 << 10;
const _: () = assert!(SPILL_MIN >= SPILL_SCRATCH);

/// Reserves `bytes` through arbitration (waiting, or asking other consumers
/// to spill) and parks them as this consumer's grant credit, where spill
/// readers and writers draw their scratch first. The caller returns unused
/// credit with `release_credit`. Only from an operator's own steps, never
/// from a spill request's service.
pub(crate) async fn lend_scratch(ctx: &TaskContext, bytes: usize) -> Result<()> {
    let mut lent = ctx.consumer().reservation();
    lent.grow(bytes)
        .await
        .map_err(krishiv_common::error::KrishivError::from)?;
    lent.shrink_to_credit(bytes);
    Ok(())
}

/// The partitions to spill for `target` bytes (at least [`SPILL_MIN`]):
/// those `first` marks, then the largest; at least one that holds memory.
pub(crate) fn choose(parts: &[Part], first: u64, target: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..parts.len())
        .filter(|&p| !parts[p].chunks.is_empty())
        .collect();
    order.sort_by_key(|&p| (first & (1 << p) == 0, std::cmp::Reverse(parts[p].bytes())));
    let mut freed = 0;
    order
        .into_iter()
        .take_while(|&p| {
            let more = freed < target.max(SPILL_MIN);
            freed += parts[p].bytes();
            more
        })
        .collect()
}

/// Reserves `chunks` (partition, batch) in the parts' first reservations,
/// all or nothing; on failure returns the bytes needed.
pub(crate) fn reserve_chunks(parts: &mut [Part], chunks: &[(usize, Batch)]) -> Result<(), usize> {
    let size = |c: &Batch| c.data().get_array_memory_size();
    for (k, (p, c)) in chunks.iter().enumerate() {
        let ok = parts
            .get_mut(*p)
            .and_then(|part| part.memory.first_mut())
            .is_some_and(|m| m.try_grow(size(c)).is_ok());
        if !ok {
            for (p, c) in &chunks[..k] {
                if let Some(m) = parts.get_mut(*p).and_then(|part| part.memory.first_mut()) {
                    m.shrink(size(c));
                }
            }
            return Err(chunks.iter().map(|(_, c)| size(c)).sum());
        }
    }
    for (p, c) in chunks {
        if let Some(part) = parts.get_mut(*p) {
            part.rows += c.num_rows() as u64;
        }
    }
    Ok(())
}

/// One spilled partition to join.
#[derive(Debug)]
struct Task {
    depth: u32,
    build: Part,
    probe: Part,
}

/// The spilled partitions of a join, shared by its probe drivers.
#[derive(Debug)]
pub(crate) struct Grace {
    /// Level-0 build partitions that spilled (set by `finalize`).
    pub(crate) build: Vec<Part>,
    /// Their probe rows (from every probe driver's `end_input`).
    pub(crate) probe: Vec<Part>,
    started: bool,
    queue: Vec<Task>,
    /// Drivers loading or holding a partition.
    active: usize,
    /// Once a partition did not fit, the one driver that may still claim
    /// partitions: it waits until no other holds one, then works alone, so
    /// repartitioning never competes with other drivers' tables.
    exclusive: Option<ConsumerId>,
}

impl Grace {
    pub(crate) fn new() -> Self {
        Self {
            build: (0..PARTITIONS).map(|_| Part::default()).collect(),
            probe: (0..PARTITIONS).map(|_| Part::default()).collect(),
            started: false,
            queue: Vec::new(),
            active: 0,
            exclusive: None,
        }
    }

    /// The next partition to join (the first call queues level 0).
    fn pop(&mut self) -> Option<Task> {
        if !self.started {
            self.started = true;
            let build = std::mem::take(&mut self.build);
            let probe = std::mem::take(&mut self.probe);
            self.queue = build
                .into_iter()
                .zip(probe)
                .filter(|(_, p)| p.rows > 0)
                .map(|(build, probe)| Task {
                    depth: 0,
                    build,
                    probe,
                })
                .collect();
        }
        self.queue.pop()
    }
}

/// A partition being joined by one driver.
#[derive(Debug)]
struct Active {
    table: JoinTable,
    _memory: Reservation,
    chunks: Vec<Batch>,
    _chunk_memory: Vec<Reservation>,
    files: VecDeque<SpillFile>,
    reader: Option<SpillReader>,
    cursor: Option<Cursor>,
}

/// One driver's claim on a partition (loading or joining it): counted in
/// [`Grace`] until dropped, also when the driver stops early (a `LIMIT`
/// above the join is satisfied) or fails.
#[derive(Debug)]
struct Slot(Arc<BuildGlobal>);

impl Slot {
    fn claim(build: &Arc<BuildGlobal>, g: &mut Grace) -> Self {
        g.active += 1;
        Self(Arc::clone(build))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let Some(built) = self.0.built.get() {
            let mut g = lock(&built.grace);
            g.active = g.active.saturating_sub(1);
        }
    }
}

/// One driver's flush state.
#[derive(Debug, Default)]
pub(crate) struct Flush {
    active: Option<Active>,
    slot: Option<Slot>,
}

/// One flush step: emit a batch of the partition being joined, read its
/// next probe batch, or claim, load or split the next partition.
pub(crate) async fn step(
    shared: &Arc<Shared>,
    seed: &HashSeed,
    build: &Arc<BuildGlobal>,
    st: &mut Flush,
    ctx: &TaskContext,
) -> Result<FlushOutput> {
    let built = build
        .built
        .get()
        .ok_or_else(|| internal("hash join probed before its build".to_owned()))?;
    let grace = &built.grace;
    let limit = ctx.query().config().batch_size.max(1);
    if let Some(a) = &mut st.active {
        if let Some(c) = &mut a.cursor {
            let tables = Tables::Single(&a.table);
            let (out, done) = c.next(shared, &tables, limit)?;
            if done {
                a.cursor = None;
            }
            return Ok(out.map_or(FlushOutput::Continue, FlushOutput::Batch));
        }
        match next_probe(a, ctx).await? {
            Some(b) => {
                let tables = Tables::Single(&a.table);
                a.cursor = Some(Cursor::new(shared, seed, b, &tables)?.0);
            }
            None => {
                st.active = None;
                st.slot = None;
                // The headroom parked for reading the probe rows.
                ctx.consumer().consumer().release_credit();
            }
        }
        return Ok(FlushOutput::Continue);
    }
    let me = ctx.consumer().id();
    let task = {
        let mut g = lock(grace);
        match g.exclusive {
            Some(owner) if owner != me => return Ok(FlushOutput::Done),
            Some(_) if g.active > 0 => return Ok(FlushOutput::Continue),
            _ => {}
        }
        let Some(task) = g.pop() else {
            return Ok(FlushOutput::Done);
        };
        st.slot = Some(Slot::claim(build, &mut g));
        task
    };
    let depth = task.depth;
    let loaded = load(shared, task, ctx).await;
    if !matches!(loaded, Ok(Ok(_))) {
        st.slot = None;
    }
    match loaded? {
        Ok(active) => {
            shared.counters.add(Counter::GracePartitions, 1);
            shared
                .counters
                .max(Counter::RecursionDepth, u64::from(depth));
            st.active = Some(active);
        }
        Err(task) => {
            {
                // It does not fit: from now on one driver works alone, once
                // no other holds a partition (the others stop claiming).
                let mut g = lock(grace);
                let owner = *g.exclusive.get_or_insert(me);
                if owner != me || g.active > 0 {
                    g.queue.push(task);
                    let next = if owner == me {
                        FlushOutput::Continue
                    } else {
                        FlushOutput::Done
                    };
                    return Ok(next);
                }
            }
            let split = split(shared, seed, task, ctx).await;
            ctx.consumer().consumer().release_credit();
            let (tasks, written) = split?;
            shared.counters.add(Counter::Repartitions, 1);
            shared.counters.add(Counter::ProbeSpillBytes, written);
            lock(grace).queue.extend(tasks);
        }
    }
    Ok(FlushOutput::Continue)
}

async fn next_probe(a: &mut Active, ctx: &TaskContext) -> Result<Option<Batch>> {
    if let Some(b) = a.chunks.pop() {
        return Ok(Some(b));
    }
    loop {
        let reader = match a.reader.take() {
            Some(r) => r,
            None => {
                let Some(file) = a.files.pop_front() else {
                    return Ok(None);
                };
                // Draws the headroom `load` lent as credit, and gives it back.
                let memory = ctx.consumer().reservation().releasing_to_credit();
                ctx.run_blocking(move || Ok(file.reader(memory)?)).await?
            }
        };
        let (reader, next) = ctx
            .run_blocking(move || {
                let mut reader = reader;
                let next = reader.next().transpose()?;
                Ok((reader, next))
            })
            .await?;
        if next.is_some() {
            a.reader = Some(reader);
            return Ok(next);
        }
    }
}

/// Every batch of `part`: its chunks, then its files (each file is deleted
/// once read).
fn for_each(
    part: Part,
    memory: impl Fn() -> Reservation,
    mut f: impl FnMut(Batch) -> Result<()>,
) -> Result<()> {
    let Part { chunks, files, .. } = part;
    for c in chunks {
        f(c)?;
    }
    for file in files {
        for b in file.reader(memory())? {
            f(b?)?;
        }
    }
    Ok(())
}

/// Loads a partition's build rows into a table, or gives the task back if
/// they do not fit. The reservation also covers reading the probe rows back
/// (parked as grant credit, which the readers draw), so a partition that
/// loads can always be joined.
async fn load(shared: &Arc<Shared>, task: Task, ctx: &TaskContext) -> Result<Result<Active, Task>> {
    let shared = Arc::clone(shared);
    let mut memory = ctx.consumer().reservation();
    let handle = Arc::clone(ctx.consumer().consumer());
    let query = Arc::clone(ctx.query());
    let loaded = ctx
        .run_blocking(move || {
            // Build chunks in memory are already reserved by their part.
            let mut chunks = task.build.chunks.clone();
            let mut fits = true;
            'files: for file in &task.build.files {
                let reader = match file.reader(handle.reservation().releasing_to_credit()) {
                    Err(SpillError::Memory(_)) => {
                        fits = false;
                        break 'files;
                    }
                    other => other?,
                };
                for b in reader {
                    query.check_cancelled()?;
                    let b = match b {
                        Err(SpillError::Memory(_)) => {
                            fits = false;
                            break 'files;
                        }
                        other => other?,
                    };
                    if memory.try_grow(b.data().get_array_memory_size()).is_err() {
                        fits = false;
                        break 'files;
                    }
                    chunks.push(b);
                }
            }
            let rows = chunks.iter().map(Batch::num_rows).sum();
            let concat = if chunks.len() > 1 { memory.size() } else { 0 };
            let headroom = task.probe.read_headroom();
            let need = JoinTable::index_bytes(rows, key_bytes(&chunks)) + concat + headroom;
            if !fits || memory.try_grow(need).is_err() {
                return Ok(Err(task));
            }
            let arena = RowArena::concat(&shared.arena_schema, &chunks)?;
            drop(chunks);
            memory.shrink(concat);
            let table = JoinTable::build(arena)?;
            memory.shrink_to_credit(headroom);
            let Task { build, probe, .. } = task;
            drop(build);
            Ok(Ok(Active {
                table,
                _memory: memory,
                chunks: probe.chunks,
                _chunk_memory: probe.memory,
                files: probe.files.into(),
                reader: None,
                cursor: None,
            }))
        })
        .await;
    if !matches!(loaded, Ok(Ok(_))) {
        ctx.consumer().consumer().release_credit();
    }
    loaded
}

/// Repartitions a task's build and probe rows one level deeper. Returns the
/// new tasks (those with probe rows) and the bytes written.
async fn split(
    shared: &Arc<Shared>,
    seed: &HashSeed,
    task: Task,
    ctx: &TaskContext,
) -> Result<(Vec<Task>, u64)> {
    if task.depth >= MAX_DEPTH {
        return Err(KrishivJoinPartitionTooLarge {
            build_rows: task.build.rows.to_string(),
            depth: task.depth.to_string(),
        }
        .into());
    }
    let scope = shared
        .spill
        .clone()
        .ok_or_else(|| internal("grace join without spill".to_owned()))?;
    // Scratch for reading the files back and for one write at a time,
    // reserved up front through arbitration: the blocking split itself can
    // only draw it, never wait for memory.
    let scratch = task.build.read_headroom().max(task.probe.read_headroom()) + SPILL_SCRATCH;
    lend_scratch(ctx, scratch).await?;
    let shared = Arc::clone(shared);
    let handle = Arc::clone(ctx.consumer().consumer());
    let query = Arc::clone(ctx.query());
    let depth = task.depth + 1;
    let seed = seed.clone();
    ctx.run_blocking(move || {
        let memory = || handle.reservation().releasing_to_credit();
        let query = &query;
        let build = scatter(
            &scope,
            &shared.arena_schema,
            task.build,
            depth,
            &memory,
            |b| {
                query.check_cancelled()?;
                let h = b
                    .data()
                    .column(0)
                    .as_primitive_opt::<UInt64Type>()
                    .map(|h| h.values().to_vec());
                h.ok_or_else(|| internal("malformed join row arena".to_owned()))
            },
        )?;
        let probe = scatter(
            &scope,
            &shared.probe_schema,
            task.probe,
            depth,
            &memory,
            |b| {
                query.check_cancelled()?;
                let cols: Vec<ArrayRef> = shared
                    .probe_keys
                    .iter()
                    .map(|&k| Arc::clone(b.data().column(k)))
                    .collect();
                Ok(shared.keys.encode(&cols, &seed)?.hashes)
            },
        )?;
        let written = build.1 + probe.1;
        let tasks = build
            .0
            .into_iter()
            .zip(probe.0)
            .filter(|(_, p)| p.rows > 0)
            .map(|(build, probe)| Task {
                depth,
                build,
                probe,
            })
            .collect();
        Ok((tasks, written))
    })
    .await
}

/// Splits `part` into level-`depth` partitions written to new spill files.
fn scatter(
    scope: &SpillScope,
    schema: &SchemaRef,
    part: Part,
    depth: u32,
    memory: &impl Fn() -> Reservation,
    hashes: impl Fn(&Batch) -> Result<Vec<u64>>,
) -> Result<(Vec<Part>, u64)> {
    let mut writers: Vec<Option<SpillWriter>> = (0..PARTITIONS).map(|_| None).collect();
    let mut rows = vec![0u64; PARTITIONS];
    for_each(part, memory, |b| {
        let mut idx = vec![Vec::new(); PARTITIONS];
        for (i, &h) in hashes(&b)?.iter().enumerate() {
            idx[partition(h, depth)].push(u32::try_from(i).unwrap_or(u32::MAX));
        }
        for (p, rows_p) in idx.into_iter().enumerate().filter(|(_, r)| !r.is_empty()) {
            rows[p] += rows_p.len() as u64;
            let w = match &mut writers[p] {
                Some(w) => w,
                slot => slot.insert(scope.writer(Arc::clone(schema), memory())?),
            };
            let piece = take_batch(&b, &UInt32Array::from(rows_p))?;
            write_sliced(w, &piece)?;
        }
        Ok(())
    })?;
    let mut written = 0;
    let mut parts = Vec::with_capacity(PARTITIONS);
    for (w, rows) in writers.into_iter().zip(rows) {
        let mut part = Part {
            rows,
            ..Part::default()
        };
        if let Some(w) = w {
            let file = w.finish()?;
            written += file.bytes();
            part.files.push(file);
        }
        parts.push(part);
    }
    Ok((parts, written))
}

// Real pools and spill files: not under loom (loom state outside a model).
#[cfg(all(test, not(loom)))]
mod tests {
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_memory::{
        MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillOwner, TreePool,
    };

    use super::*;
    use crate::{ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, TokioExecutor};
    use crate::{WorkloadClass, context::TaskContext};

    /// A spill directory under the system temp dir, removed when dropped.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let name = format!("krishiv-{name}-{}", std::process::id());
            Self(std::env::temp_dir().join(name))
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Coalescing is size-tiered: a merged chunk is not copied again by the
    /// next run of small chunks (merging all chunks every 16 pushes was
    /// quadratic), the chunk count stays bounded, and no row is lost.
    #[test]
    fn coalescing_does_not_recopy_merged_chunks() {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let id = pool.register("coalesce", None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let chunk = |i: i64| {
            let k = Int64Array::from_iter_values(i * 300..(i + 1) * 300);
            let s = StringArray::from_iter_values((0..300).map(|j| format!("row-{i}-{j}")));
            Batch::new(
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(s)]).unwrap(),
            )
        };
        let mut parts = vec![Part::with_memory(krishiv_memory::Reservation::new(
            Arc::clone(&pool),
            id,
        ))];
        let push = |parts: &mut Vec<Part>, i: i64| {
            let c = chunk(i);
            reserve_chunks(parts, &[(0, c.clone())]).unwrap();
            parts[0].chunks.push(c);
            parts[0].coalesce(&schema).unwrap();
        };
        for i in 0..16 {
            push(&mut parts, i);
        }
        assert_eq!(parts[0].chunks.len(), 1, "16 small chunks merge");
        let merged = Arc::clone(parts[0].chunks[0].data().column(0));
        for i in 16..31 {
            push(&mut parts, i);
        }
        assert_eq!(parts[0].chunks.len(), 16);
        let first = parts[0].chunks[0].data().column(0);
        assert!(
            Arc::ptr_eq(first, &merged),
            "the merged chunk was copied again"
        );
        let mut most = 0;
        for i in 31..1024 {
            push(&mut parts, i);
            most = most.max(parts[0].chunks.len());
        }
        assert!(most < 3 * 16, "{most} chunks held");
        assert_eq!(parts[0].mem_rows(), 1024 * 300);
        let mut keys: Vec<i64> = parts[0]
            .chunks
            .iter()
            .flat_map(|c| {
                let k = c
                    .data()
                    .column(0)
                    .as_primitive::<arrow_array::types::Int64Type>();
                k.values().to_vec()
            })
            .collect();
        keys.sort_unstable();
        assert!(
            keys.into_iter().eq(0..1024 * 300),
            "rows lost or duplicated"
        );
        drop(parts);
        assert_eq!(pool.reserved(), 0);
    }

    /// Probe rows deferred in many small chunks are coalesced, spilled in
    /// slices, and read back by the grace phase with exactly the headroom
    /// [`load`] reserves: with the pool otherwise full, no read reaches it.
    /// (The headroom used to be twice the largest chunk *before*
    /// coalescing, which a coalesced chunk's slices outgrow: a read then
    /// drew from a pool that other drivers kept full.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn spilled_probe_rows_read_back_within_the_load_headroom() {
        const LIMIT: usize = 64 << 20;
        let root = TreePool::new(&MemoryConfig {
            limit: Some(LIMIT),
            quantum: 1,
            ..MemoryConfig::default()
        });
        let query = QueryContext::new(
            QueryId(1),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            root.child("q", None),
        );
        let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
        let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
        let dir = TempDir::new("grace-headroom");
        let manager = SpillManager::open(&SpillConfig {
            dir: dir.0.clone(),
            ..SpillConfig::default()
        })
        .unwrap();
        let scope = manager.scope().unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let chunk = |i: i64| {
            let k = Int64Array::from_iter_values(i * 300..(i + 1) * 300);
            let s = StringArray::from_iter_values((0..300).map(|j| format!("row-{i}-{j}")));
            Batch::new(
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(s)]).unwrap(),
            )
        };
        let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
        let mut largest_chunk = 0;
        for i in 0..64 {
            let c = chunk(i);
            largest_chunk = largest_chunk.max(c.data().get_array_memory_size());
            reserve_chunks(&mut parts, &[(0, c.clone())]).unwrap();
            parts[0].chunks.push(c);
            parts[0].coalesce(&schema).unwrap();
        }
        assert!(parts[0].chunks.len() < 16, "coalesced");
        spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false)
            .await
            .unwrap();
        let headroom = parts[0].read_headroom();
        assert!(headroom > 2 * largest_chunk, "the old estimate fell short");

        // The pool is full but for the headroom, lent as credit as `load` does.
        let mut lent = ctx.consumer().reservation();
        lent.try_grow(headroom).unwrap();
        lent.shrink_to_credit(headroom);
        let other: Arc<dyn MemoryPool> = root.child("other", None);
        let id = other.register("filler", None);
        let mut filler = krishiv_memory::Reservation::new(Arc::clone(&other), id);
        filler.try_grow(LIMIT - root.reserved()).unwrap();
        let mut rows = 0;
        for file in &parts[0].files {
            for b in file.reader(ctx.consumer().reservation()).unwrap() {
                rows += b.unwrap().num_rows();
            }
        }
        assert_eq!(rows, 64 * 300);
        ctx.consumer().consumer().release_credit();
        drop((filler, parts));
        assert_eq!(root.reserved(), 0);
    }

    /// A spill whose rows and the pool's free memory cannot fund its
    /// writer's scratch is declined: nothing is written or lost, no credit
    /// is left behind, and the query does not fail.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn an_unfundable_spill_is_declined() {
        const LIMIT: usize = 4 << 20;
        let root = TreePool::new(&MemoryConfig {
            limit: Some(LIMIT),
            quantum: 1,
            ..MemoryConfig::default()
        });
        let query = QueryContext::new(
            QueryId(1),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            root.child("q", None),
        );
        let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
        let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
        let dir = TempDir::new("grace-decline");
        let manager = SpillManager::open(&SpillConfig {
            dir: dir.0.clone(),
            ..SpillConfig::default()
        })
        .unwrap();
        let scope = manager.scope().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let small = Batch::new(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from_iter_values(0..100))],
            )
            .unwrap(),
        );
        let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
        reserve_chunks(&mut parts, &[(0, small.clone())]).unwrap();
        parts[0].chunks.push(small);
        let held = parts[0].bytes();
        let other: Arc<dyn MemoryPool> = root.child("other", None);
        let id = other.register("filler", None);
        let mut filler = krishiv_memory::Reservation::new(Arc::clone(&other), id);
        filler.try_grow(LIMIT - root.reserved()).unwrap();
        let out = spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false).await;
        assert_eq!(out.unwrap(), (0, 0));
        assert_eq!((parts[0].chunks.len(), parts[0].bytes()), (1, held));
        assert!(parts[0].files.is_empty());
        assert_eq!(ctx.consumer().consumer().credit(), 0);
        // With room for the scratch, the same spill goes ahead.
        drop(filler);
        let (freed, written) = spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false)
            .await
            .unwrap();
        assert_eq!((freed, parts[0].files.len()), (held, 1));
        assert!(written > 0);
        assert_eq!(ctx.consumer().consumer().credit(), 0);
        drop(parts);
        assert_eq!(root.reserved(), 0);
    }

    /// A spill picks at least `SPILL_MIN` bytes when it can, never a part
    /// without rows in memory.
    #[test]
    fn spills_write_at_least_the_scratch_they_need() {
        assert!(choose(&[Part::default()], 0, 1).is_empty());
    }
}
