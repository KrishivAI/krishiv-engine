// SPDX-License-Identifier: Apache-2.0
//! Sort-merge join (P2-T056): joins over both inputs sorted on the keys, the
//! same semantics as the hash join's (module docs of [`crate::join`]), for
//! every join type but the null-aware anti join.
//!
//! # Shape
//! A [`SortMergeJoinNode`] sorts each input in a child pipeline ([`Sort`] in
//! hash buckets of the keys; equal keys share a bucket on both sides). Its
//! source pairs the buckets: each driver claims a bucket number and merges
//! that bucket of the two sorts, so the join runs on as many drivers as there
//! are buckets, and output is ordered within a bucket only.
//!
//! # The merge
//! [`Merge`] walks the two sorted streams group by group. Rows are compared
//! by their sort keys' binary rows, so equal bytes are equal keys: floats are
//! canonical (`-0.0`, NaN) and the types of the two sides are the same. A row
//! with a NULL in a key compared with `=` matches nothing, however its bytes
//! compare; a key compared with `<=>` matches NULL to NULL by those bytes.
//!
//! The right rows of one key are buffered (the *group*); the left rows of
//! that key stream past them, a slice of a batch at a time, so a key hot on
//! the left costs no memory. A group over its byte bound, or that the pool
//! refuses, moves to a spill file, and each left slice then reads it back a
//! chunk at a time. Spark bounds the group by rows
//! (`spark.sql.sortMergeJoinExec.buffer.*`); here it is bytes
//! (`krishiv.exec.join.smj.bufferBytes`). Marks for the right rows (a right
//! or full join with a residual condition) are one bit per row, reserved.
//!
//! # Memory
//! Group memory is reserved in the driver's consumer and released with the
//! group; output rows reference their input batches as every operator's
//! output does. A merge of spilled runs needs its frames from both sorts, so
//! drivers that would wait on each other for them never hold a bucket
//! meanwhile ([`Claims`]): a driver alone may wait with arbitration, the others
//! take a bucket only with memory to spare, so a join one driver can do never
//! fails for its parallel drivers.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_buffer::NullBuffer;
use arrow_row::Rows;
use arrow_schema::{DataType, SchemaRef};
use arrow_select::interleave::interleave;
use futures::future::BoxFuture;
use krishiv_common::error::{KrishivError, ScalarSubqueryTooManyRows};
use krishiv_memory::{ConsumerHandle, Reservation, SpillFile, SpillReader, SpillWriter};
use krishiv_types::batch::Batch;

use super::{JoinSpec, JoinType, Shared, record};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::executor::Executor;
use crate::metrics::Metric;
use crate::operator::{GlobalState, LocalState, Sink, Source};
use crate::operators::{KeyEncoder, blocking, data_bytes};
use crate::operators::{Sort, SortKey, global_ref, local_mut};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::sync::{Event, Mutex, lock};

/// Candidate pairs whose residual condition is evaluated at once.
const PAIR_BATCH: usize = 1024;
/// A spill writer's scratch ([`super::SPILL_SCRATCH`]).
const WRITE_SCRATCH: usize = super::SPILL_SCRATCH;
/// The group bound without a pool limit.
const UNLIMITED_POOL_GROUP_BYTES: usize = 1 << 30;
/// The least group bound a pool share gives.
const MIN_GROUP_BYTES: usize = 1 << 20;

/// The bytes a group may hold before it spills: `configured`, or (0) a
/// quarter of the pool's limit shared by the drivers.
pub(crate) fn group_bytes(configured: usize, handle: &ConsumerHandle, drivers: usize) -> usize {
    if configured > 0 {
        return configured;
    }
    handle
        .pool()
        .limit()
        .map_or(UNLIMITED_POOL_GROUP_BYTES, |l| {
            (l / 4 / drivers.max(1)).max(MIN_GROUP_BYTES)
        })
}

/// What a merge runs with, owned: a source has no context in `get_data`.
#[derive(Clone)]
pub(crate) struct MergeEnv {
    pub(crate) query: Arc<QueryContext>,
    pub(crate) executor: Arc<dyn Executor>,
    pub(crate) handle: Arc<ConsumerHandle>,
    pub(crate) partition: u32,
}

impl MergeEnv {
    pub(crate) fn new(ctx: &TaskContext) -> Self {
        Self {
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
            handle: Arc::clone(ctx.consumer().consumer()),
            partition: ctx.partition(),
        }
    }

    fn batch_rows(&self) -> usize {
        self.query.config().batch_size.max(1)
    }
}

/// What a merge reads: one bucket of a finalized sort.
pub(crate) struct MergeInput {
    pub(crate) sort: Arc<Sort>,
    pub(crate) global: GlobalState,
    pub(crate) local: LocalState,
}

/// One side of a merge.
struct Stream {
    input: MergeInput,
    keys: Arc<KeyEncoder>,
    /// The key columns compared with `=`: a NULL there matches nothing.
    strict: Arc<[usize]>,
    current: Option<Arc<Keyed>>,
    /// The next row of `current`.
    pos: usize,
    ended: bool,
    serial: u64,
}

/// A batch with its key rows.
struct Keyed {
    batch: Arc<Batch>,
    rows: Rows,
    /// Rows without a NULL in a strict key (none: every row).
    matchable: Option<NullBuffer>,
    /// This batch's number on its side, to spot it in [`Out`].
    serial: u64,
    /// The bytes its rows hold, an average.
    row_bytes: usize,
}

impl Keyed {
    fn key(&self, i: usize) -> &[u8] {
        self.rows.row(i).data()
    }

    fn is_matchable(&self, i: usize) -> bool {
        self.matchable.as_ref().is_none_or(|m| m.is_valid(i))
    }

    /// The first row from `from` whose key is not below `key`.
    fn lower_bound(&self, from: usize, key: &[u8]) -> usize {
        self.search(from, |k| k < key)
    }

    /// The first row from `from` whose key is above `key`.
    fn upper_bound(&self, from: usize, key: &[u8]) -> usize {
        self.search(from, |k| k <= key)
    }

    fn search(&self, from: usize, below: impl Fn(&[u8]) -> bool) -> usize {
        // A group is usually short: look next door before searching.
        let n = self.batch.num_rows();
        if from >= n || !below(self.key(from)) {
            return from;
        }
        let (mut lo, mut hi) = (from + 1, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if below(self.key(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

impl Stream {
    fn new(input: MergeInput, keys: Arc<KeyEncoder>, strict: Arc<[usize]>) -> Self {
        Self {
            input,
            keys,
            strict,
            current: None,
            pos: 0,
            ended: false,
            serial: 0,
        }
    }

    /// Makes `current` hold the next row, unless the stream ended.
    async fn fill(&mut self, query: &QueryContext) -> Result<()> {
        while !self.ended
            && self
                .current
                .as_ref()
                .is_none_or(|c| self.pos >= c.batch.num_rows())
        {
            query.check_cancelled()?;
            let (sort, local, global) =
                (&self.input.sort, &mut self.input.local, &self.input.global);
            let next = Source::get_data(&**sort, local, global).await?;
            let Some(batch) = next else {
                self.ended = true;
                self.current = None;
                break;
            };
            if !batch.is_unweighted() {
                return Err(super::unsupported("a sort-merge join over weighted rows"));
            }
            let rows = self.keys.encode(batch.data())?;
            let mut matchable: Option<NullBuffer> = None;
            for &c in self.strict.iter() {
                matchable = NullBuffer::union(matchable.as_ref(), batch.data().column(c).nulls());
            }
            self.serial += 1;
            let row_bytes = data_bytes(&batch).div_ceil(batch.num_rows().max(1));
            self.current = Some(Arc::new(Keyed {
                batch: Arc::new(batch),
                rows,
                matchable,
                serial: self.serial,
                row_bytes,
            }));
            self.pos = 0;
        }
        Ok(())
    }
}

/// The right rows of one key.
#[derive(Default)]
struct Group {
    key: Vec<u8>,
    chunks: Vec<Chunk>,
    /// Once spilled, every row is in the file (`chunks` is empty).
    writer: Option<SpillWriter>,
    file: Option<Arc<SpillFile>>,
    rows: u64,
    bytes: usize,
    /// A left row matched (no residual: every right row matched then).
    any_left: bool,
    /// Per right row, in order: it matched (right joins with a residual).
    marks: Vec<u64>,
}

impl Group {
    fn mark(&mut self, row: u64) {
        if let Some(w) = usize::try_from(row / 64)
            .ok()
            .and_then(|i| self.marks.get_mut(i))
        {
            *w |= 1 << (row % 64);
        }
    }

    fn marked(&self, row: u64) -> bool {
        usize::try_from(row / 64)
            .ok()
            .and_then(|i| self.marks.get(i))
            .is_some_and(|w| w & (1 << (row % 64)) != 0)
    }
}

/// Right rows `start..end` of a batch: a group's rows stay in the batches
/// they came in (no copy, and one [`Out`] source per batch).
#[derive(Clone)]
struct Chunk {
    batch: Arc<Batch>,
    serial: u64,
    start: usize,
    end: usize,
}

/// Reads a group's chunks in order, from memory or its file.
#[derive(Default)]
struct Chunks {
    next: usize,
    reader: Option<SpillReader>,
    current: Option<Chunk>,
    /// The number of `current`'s first row in the group.
    base: u64,
    /// The next row of `current`.
    row: usize,
}

/// Output rows gathered from their source batches, as `(source, row)`; source
/// 0 of each side is a single row of NULLs.
struct Out {
    left: Vec<RecordBatch>,
    right: Vec<RecordBatch>,
    left_rows: Vec<(usize, usize)>,
    right_rows: Vec<(usize, usize)>,
    exists: Vec<bool>,
    left_null: RecordBatch,
    right_null: RecordBatch,
    /// The serial of the source last added on each side, with its index.
    last_left: Option<(u64, usize)>,
    last_right: Option<(u64, usize)>,
}

impl Out {
    fn new(left_null: RecordBatch, right_null: RecordBatch) -> Self {
        let mut out = Self {
            left: Vec::new(),
            right: Vec::new(),
            left_rows: Vec::new(),
            right_rows: Vec::new(),
            exists: Vec::new(),
            left_null,
            right_null,
            last_left: None,
            last_right: None,
        };
        out.reset();
        out
    }

    fn reset(&mut self) {
        self.left.clear();
        self.right.clear();
        self.left.push(self.left_null.clone());
        self.right.push(self.right_null.clone());
        self.last_left = None;
        self.last_right = None;
    }

    fn len(&self) -> usize {
        self.left_rows.len().max(self.right_rows.len())
    }

    /// Drops the sources no row refers to any more (a flush leaves few rows
    /// behind, and could otherwise keep every batch it ever read).
    fn compact(&mut self) {
        Self::compact_side(&mut self.left, &mut self.left_rows, &mut self.last_left);
        Self::compact_side(&mut self.right, &mut self.right_rows, &mut self.last_right);
    }

    fn compact_side(
        sources: &mut Vec<RecordBatch>,
        rows: &mut [(usize, usize)],
        last: &mut Option<(u64, usize)>,
    ) {
        let mut map: Vec<Option<usize>> = vec![None; sources.len()];
        map[0] = Some(0);
        let mut kept = vec![sources[0].clone()];
        for (source, _) in rows.iter_mut() {
            let to = match map[*source] {
                Some(to) => to,
                None => {
                    kept.push(sources[*source].clone());
                    let to = kept.len() - 1;
                    map[*source] = Some(to);
                    to
                }
            };
            *source = to;
        }
        *last = last.and_then(|(s, i)| map[i].map(|to| (s, to)));
        *sources = kept;
    }

    fn left_source(&mut self, b: &Batch, serial: u64) -> usize {
        match self.last_left {
            Some((s, i)) if s == serial => i,
            _ => {
                self.left.push(b.data().clone());
                let i = self.left.len() - 1;
                self.last_left = Some((serial, i));
                i
            }
        }
    }

    fn right_source(&mut self, b: &Batch, serial: u64) -> usize {
        match self.last_right {
            Some((s, i)) if s == serial => i,
            _ => {
                self.right.push(b.data().clone());
                let i = self.right.len() - 1;
                self.last_right = Some((serial, i));
                i
            }
        }
    }
}

/// Where a [`Merge`] is.
enum Phase {
    /// Comparing the streams' next rows.
    Compare,
    /// Buffering the right rows of `group.key`.
    Collect,
    /// Taking the next left rows of `group.key`.
    NextSlice,
    /// Joining left rows `start..end` of the current batch with the group.
    Pairs(Slice),
    /// Left rows of the key are done: the right rows the join returns by
    /// their marks.
    Drain,
}

/// Left rows being joined with the group.
struct Slice {
    start: usize,
    end: usize,
    /// The left row being paired (with a residual).
    row: usize,
    /// Per left row: it matched (with a residual; without one all do).
    matched: Vec<bool>,
}

/// One merge of two sorted streams.
pub(crate) struct Merge {
    shared: Arc<Shared>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    handle: Arc<ConsumerHandle>,
    left: Stream,
    right: Stream,
    /// The output columns of each side (a fallback's key column is not one).
    left_columns: usize,
    right_columns: usize,
    right_schema: SchemaRef,
    phase: Phase,
    group: Group,
    chunks: Chunks,
    memory: Reservation,
    out: Out,
    batch_rows: usize,
    group_limit: usize,
    partition: u32,
    counters: Arc<MergeCounters>,
}

/// What a merge joins and under which limits.
pub(crate) struct MergeSpec {
    pub(crate) left: MergeInput,
    pub(crate) right: MergeInput,
    pub(crate) left_keys: Arc<KeyEncoder>,
    pub(crate) right_keys: Arc<KeyEncoder>,
    pub(crate) left_strict: Arc<[usize]>,
    pub(crate) right_strict: Arc<[usize]>,
    pub(crate) left_schema: SchemaRef,
    pub(crate) right_schema: SchemaRef,
    pub(crate) left_columns: usize,
    pub(crate) right_columns: usize,
    pub(crate) group_limit: usize,
    pub(crate) counters: Arc<MergeCounters>,
}

/// A merge's counters.
#[derive(Debug, Default)]
pub(crate) struct MergeCounters {
    pub(crate) groups_spilled: AtomicU64,
    pub(crate) spill_bytes: AtomicU64,
}

/// A batch of one row of NULLs, in `schema`'s first `columns` columns.
fn null_row(schema: &SchemaRef, columns: usize) -> Result<RecordBatch> {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .take(columns)
        .map(|f| f.as_ref().clone().with_nullable(true))
        .collect();
    let arrays: Vec<ArrayRef> = fields
        .iter()
        .map(|f| new_null_array(f.data_type(), 1))
        .collect();
    record(&Arc::new(arrow_schema::Schema::new(fields)), arrays, 1)
}

impl Merge {
    pub(crate) fn new(shared: Arc<Shared>, spec: MergeSpec, env: &MergeEnv) -> Result<Self> {
        let out = Out::new(
            null_row(&spec.left_schema, spec.left_columns)?,
            null_row(&spec.right_schema, spec.right_columns)?,
        );
        Ok(Self {
            shared,
            query: Arc::clone(&env.query),
            executor: Arc::clone(&env.executor),
            memory: env.handle.reservation(),
            handle: Arc::clone(&env.handle),
            left: Stream::new(spec.left, spec.left_keys, spec.left_strict),
            right: Stream::new(spec.right, spec.right_keys, spec.right_strict),
            left_columns: spec.left_columns,
            right_columns: spec.right_columns,
            right_schema: spec.right_schema,
            phase: Phase::Compare,
            group: Group::default(),
            chunks: Chunks::default(),
            out,
            batch_rows: env.batch_rows(),
            group_limit: spec.group_limit,
            partition: env.partition,
            counters: spec.counters,
        })
    }

    fn join_type(&self) -> JoinType {
        self.shared.join_type
    }

    /// The next batch of output, `None` when both streams ended.
    pub(crate) async fn next(&mut self) -> Result<Option<Batch>> {
        loop {
            if self.out.len() >= self.batch_rows {
                return self.flush().map(Some);
            }
            match std::mem::replace(&mut self.phase, Phase::Compare) {
                Phase::Compare => {
                    if !self.compare().await? {
                        // Both ended.
                        return if self.out.len() > 0 {
                            self.flush().map(Some)
                        } else {
                            Ok(None)
                        };
                    }
                }
                Phase::Collect => self.collect().await?,
                Phase::NextSlice => self.next_slice().await?,
                Phase::Pairs(slice) => self.pairs(slice).await?,
                Phase::Drain => self.drain().await?,
            }
        }
    }

    /// One step over the streams' next rows; `false` when both ended.
    async fn compare(&mut self) -> Result<bool> {
        self.left.fill(&self.query).await?;
        self.right.fill(&self.query).await?;
        let (lpos, rpos) = (self.left.pos, self.right.pos);
        let (left_end, right_end) = match (&self.left.current, &self.right.current) {
            (None, None) => return Ok(false),
            (Some(l), None) => (Some(l.batch.num_rows()), None),
            (None, Some(r)) => (None, Some(r.batch.num_rows())),
            (Some(l), Some(r)) => {
                let (lk, rk) = (l.key(lpos), r.key(rpos));
                if !l.is_matchable(lpos) {
                    (Some(l.upper_bound(lpos, lk)), None)
                } else if !r.is_matchable(rpos) {
                    (None, Some(r.upper_bound(rpos, rk)))
                } else {
                    match lk.cmp(rk) {
                        std::cmp::Ordering::Less => (Some(l.lower_bound(lpos, rk)), None),
                        std::cmp::Ordering::Greater => (None, Some(r.lower_bound(rpos, lk))),
                        std::cmp::Ordering::Equal => {
                            self.group.key.clear();
                            self.group.key.extend_from_slice(lk);
                            self.phase = Phase::Collect;
                            (None, None)
                        }
                    }
                }
            }
        };
        if let Some(end) = left_end {
            self.left_unmatched(lpos, end);
        }
        if let Some(end) = right_end {
            self.right_unmatched(rpos, end);
        }
        Ok(true)
    }

    /// Left rows `start..end` of the current batch match nothing.
    fn left_unmatched(&mut self, start: usize, end: usize) {
        self.left.pos = end;
        let t = self.join_type();
        if !(t.pads_probe() || matches!(t, JoinType::LeftAnti | JoinType::Existence)) {
            return;
        }
        let Some(l) = &self.left.current else { return };
        let source = self.out.left_source(&l.batch, l.serial);
        for i in start..end {
            self.out.left_rows.push((source, i));
            if t.pads_probe() {
                self.out.right_rows.push((0, 0));
            }
            if t == JoinType::Existence {
                self.out.exists.push(false);
            }
        }
    }

    /// Right rows `start..end` of the current batch match nothing.
    fn right_unmatched(&mut self, start: usize, end: usize) {
        self.right.pos = end;
        let t = self.join_type();
        if !matches!(
            t,
            JoinType::RightOuter | JoinType::FullOuter | JoinType::RightAnti
        ) {
            return;
        }
        let Some(r) = &self.right.current else { return };
        let source = self.out.right_source(&r.batch, r.serial);
        for i in start..end {
            self.out.right_rows.push((source, i));
            if t != JoinType::RightAnti {
                self.out.left_rows.push((0, 0));
            }
        }
    }

    /// Buffers the right rows of the group's key from the current batch.
    async fn collect(&mut self) -> Result<()> {
        self.right.fill(&self.query).await?;
        let more = match &self.right.current {
            Some(r) if r.key(self.right.pos) == self.group.key.as_slice() => {
                let (start, end) = (
                    self.right.pos,
                    r.upper_bound(self.right.pos, &self.group.key),
                );
                let chunk = Chunk {
                    batch: Arc::clone(&r.batch),
                    serial: r.serial,
                    start,
                    end,
                };
                Some((chunk, (end - start).saturating_mul(r.row_bytes)))
            }
            _ => None,
        };
        let Some((chunk, bytes)) = more else {
            self.finish_collect().await?;
            self.phase = Phase::NextSlice;
            return Ok(());
        };
        self.right.pos = chunk.end;
        self.group.rows += (chunk.end - chunk.start) as u64;
        self.add_chunk(chunk, bytes).await?;
        self.phase = Phase::Collect;
        Ok(())
    }

    /// Adds right rows to the group, in memory while they fit.
    async fn add_chunk(&mut self, chunk: Chunk, bytes: usize) -> Result<()> {
        if self.group.writer.is_some() {
            return self.write_chunks(vec![chunk]).await;
        }
        let spills = self.shared.spill.is_some();
        let over = spills && self.group.bytes.saturating_add(bytes) > self.group_limit;
        if over || (spills && self.memory.try_grow(bytes).is_err()) {
            let mut all = std::mem::take(&mut self.group.chunks);
            all.push(chunk);
            return self.write_chunks(all).await;
        }
        if !spills {
            self.memory.grow(bytes).await.map_err(KrishivError::from)?;
        }
        self.group.bytes += bytes;
        self.group.chunks.push(chunk);
        Ok(())
    }

    /// Writes `chunks` to the group's spill file, opening it first.
    async fn write_chunks(&mut self, chunks: Vec<Chunk>) -> Result<()> {
        let mut writer = match self.group.writer.take() {
            Some(w) => w,
            None => {
                let scope = self
                    .shared
                    .spill
                    .clone()
                    .ok_or_else(|| internal("a group spilled without a scope".to_owned()))?;
                let mut lent = self.handle.reservation();
                lent.grow(WRITE_SCRATCH).await.map_err(KrishivError::from)?;
                lent.shrink_to_credit(WRITE_SCRATCH);
                self.counters.groups_spilled.fetch_add(1, Ordering::Relaxed);
                scope
                    .writer(
                        Arc::clone(&self.right_schema),
                        self.handle.reservation().releasing_to_credit(),
                    )
                    .map_err(KrishivError::from)?
            }
        };
        let query = Arc::clone(&self.query);
        let writer = blocking(&self.query, &self.executor, move || {
            for c in &chunks {
                query.check_cancelled()?;
                let rows = c
                    .batch
                    .slice(c.start, c.end - c.start)
                    .map_err(|e| internal(e.to_string()))?;
                super::write_sliced(&mut writer, &rows)?;
            }
            Ok(writer)
        })
        .await?;
        self.group.writer = Some(writer);
        // Everything in memory moved to the file.
        let held = std::mem::take(&mut self.group.bytes);
        self.memory.shrink(held.min(self.memory.size()));
        Ok(())
    }

    /// The group is complete: closes its file, reserves its marks and the
    /// first read-back.
    async fn finish_collect(&mut self) -> Result<()> {
        if let Some(writer) = self.group.writer.take() {
            let file = writer.finish().map_err(KrishivError::from)?;
            self.counters
                .spill_bytes
                .fetch_add(file.bytes(), Ordering::Relaxed);
            self.handle.release_credit();
            self.group.file = Some(Arc::new(file));
        }
        if self.shared.join_type.builds_preserved() && self.shared.filter.is_some() {
            let words = usize::try_from(self.group.rows.div_ceil(64))
                .map_err(|_| internal("a join group too large to mark".to_owned()))?;
            self.memory
                .grow(words * 8)
                .await
                .map_err(KrishivError::from)?;
            self.group.bytes += words * 8;
            self.group.marks.clear();
            self.group.marks.resize(words, 0);
        }
        Ok(())
    }

    /// The next left rows of the group's key, or on to the right rows.
    async fn next_slice(&mut self) -> Result<()> {
        self.left.fill(&self.query).await?;
        let Some(l) = &self.left.current else {
            self.phase = Phase::Drain;
            return Ok(());
        };
        let start = self.left.pos;
        if l.key(start) != self.group.key.as_slice() {
            self.phase = Phase::Drain;
            return Ok(());
        }
        let end = l.upper_bound(start, &self.group.key);
        let t = self.join_type();
        if self.shared.filter.is_none() {
            if self.shared.single_match && self.group.rows > 1 {
                return Err(ScalarSubqueryTooManyRows {}.into());
            }
            self.group.any_left = true;
            // Without a residual every left row matches every right row.
            match t {
                JoinType::LeftSemi | JoinType::Existence => {
                    let source = self.out.left_source(&l.batch, l.serial);
                    for i in start..end {
                        self.out.left_rows.push((source, i));
                        if t == JoinType::Existence {
                            self.out.exists.push(true);
                        }
                    }
                    self.left.pos = end;
                    self.phase = Phase::NextSlice;
                    return Ok(());
                }
                JoinType::LeftAnti | JoinType::RightSemi | JoinType::RightAnti => {
                    self.left.pos = end;
                    self.phase = Phase::NextSlice;
                    return Ok(());
                }
                _ => {}
            }
        }
        self.chunks = Chunks::default();
        let matched = if self.shared.filter.is_some() {
            vec![false; end - start]
        } else {
            Vec::new()
        };
        self.phase = Phase::Pairs(Slice {
            start,
            end,
            row: start,
            matched,
        });
        Ok(())
    }

    /// Makes `chunks.current` the group's chunk holding the next right row;
    /// `false` past the last one.
    async fn next_chunk(&mut self) -> Result<bool> {
        if let Some(c) = &self.chunks.current {
            if self.chunks.row < c.end {
                return Ok(true);
            }
            self.chunks.base += (c.end - c.start) as u64;
            self.chunks.current = None;
        }
        if self.group.file.is_none() {
            let next = self.group.chunks.get(self.chunks.next).cloned();
            self.chunks.next += 1;
            self.chunks.row = next.as_ref().map_or(0, |c| c.start);
            self.chunks.current = next;
            return Ok(self.chunks.current.is_some());
        }
        let reader = match self.chunks.reader.take() {
            Some(r) => r,
            None => {
                let Some(file) = self.group.file.clone() else {
                    return Ok(false);
                };
                let mut lent = self.handle.reservation();
                let need = file.max_read_reservation();
                lent.grow(need).await.map_err(KrishivError::from)?;
                lent.shrink_to_credit(need);
                let memory = self.handle.reservation().releasing_to_credit();
                blocking(&self.query, &self.executor, move || {
                    file.reader(memory).map_err(KrishivError::from)
                })
                .await?
            }
        };
        self.query.check_cancelled()?;
        let (reader, next) = blocking(&self.query, &self.executor, move || {
            let mut reader = reader;
            let next = reader.next().transpose().map_err(KrishivError::from)?;
            Ok((reader, next))
        })
        .await?;
        let Some(next) = next else {
            drop(reader);
            self.handle.release_credit();
            return Ok(false);
        };
        self.chunks.reader = Some(reader);
        self.right.serial += 1;
        self.chunks.row = 0;
        self.chunks.current = Some(Chunk {
            start: 0,
            end: next.num_rows(),
            serial: self.right.serial,
            batch: Arc::new(next),
        });
        Ok(true)
    }

    /// Pairs left rows of the slice with the group, up to the output's room.
    async fn pairs(&mut self, mut slice: Slice) -> Result<()> {
        self.query.check_cancelled()?;
        if !self.next_chunk().await? {
            return self.end_slice(&slice);
        }
        if self.shared.filter.is_some() {
            self.filtered_pairs(&mut slice)?;
        } else {
            self.product(&slice);
        }
        self.phase = Phase::Pairs(slice);
        Ok(())
    }

    /// Without a residual: the product of the slice and the current chunk.
    fn product(&mut self, slice: &Slice) {
        let (Some(l), Some(c)) = (&self.left.current, &self.chunks.current) else {
            return;
        };
        let left = self.out.left_source(&l.batch, l.serial);
        let right = self.out.right_source(&c.batch, c.serial);
        while self.chunks.row < c.end && self.out.len() < self.batch_rows {
            let r = self.chunks.row;
            for i in slice.start..slice.end {
                self.out.left_rows.push((left, i));
                self.out.right_rows.push((right, r));
            }
            self.chunks.row += 1;
        }
    }

    /// With a residual: evaluates up to [`PAIR_BATCH`] candidate pairs of the
    /// current chunk.
    fn filtered_pairs(&mut self, slice: &mut Slice) -> Result<()> {
        let (Some(l), Some(c), Some(filter)) = (
            self.left.current.clone(),
            self.chunks.current.clone(),
            self.shared.filter.clone(),
        ) else {
            return Ok(());
        };
        // (left row, chunk row)
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(PAIR_BATCH);
        let (mut r, mut i) = (self.chunks.row, slice.row);
        while r < c.end && pairs.len() < PAIR_BATCH {
            while i < slice.end && pairs.len() < PAIR_BATCH {
                pairs.push((i, r));
                i += 1;
            }
            if i >= slice.end {
                i = slice.start;
                r += 1;
            }
        }
        slice.row = i;
        self.chunks.row = r;
        if pairs.is_empty() {
            return Ok(());
        }
        let left_idx: Vec<(usize, usize)> = pairs.iter().map(|&(i, _)| (0, i)).collect();
        let right_idx: Vec<(usize, usize)> = pairs.iter().map(|&(_, r)| (0, r)).collect();
        let mut columns = gather(
            std::slice::from_ref(l.batch.data()),
            &left_idx,
            self.left_columns,
        )?;
        columns.extend(gather(
            std::slice::from_ref(c.batch.data()),
            &right_idx,
            self.right_columns,
        )?);
        let candidates = record(&self.shared.filter_schema, columns, pairs.len())?;
        let mask = filter.evaluate_in(&candidates, self.partition)?;
        if mask.len() != pairs.len() {
            return Err(internal(
                "a join filter returned the wrong row count".to_owned(),
            ));
        }
        let t = self.shared.join_type;
        let base = self.chunks.base;
        for (j, &(i, r)) in pairs.iter().enumerate() {
            if !mask.is_valid(j) || !mask.value(j) {
                continue;
            }
            let at = i - slice.start;
            if self.shared.single_match && slice.matched[at] {
                return Err(ScalarSubqueryTooManyRows {}.into());
            }
            slice.matched[at] = true;
            self.group.any_left = true;
            if t.builds_preserved() {
                self.group.mark(base + (r - c.start) as u64);
            }
            if t.emits_build() {
                let left = self.out.left_source(&l.batch, l.serial);
                let right = self.out.right_source(&c.batch, c.serial);
                self.out.left_rows.push((left, i));
                self.out.right_rows.push((right, r));
            }
        }
        Ok(())
    }

    /// Every right row was paired with the slice: its unmatched left rows,
    /// and on to the next left rows.
    fn end_slice(&mut self, slice: &Slice) -> Result<()> {
        self.left.pos = slice.end;
        self.chunks = Chunks::default();
        self.phase = Phase::NextSlice;
        let t = self.shared.join_type;
        // Without a residual the outer joins have nothing left to emit.
        if slice.matched.is_empty()
            || matches!(
                t,
                JoinType::Inner | JoinType::RightSemi | JoinType::RightAnti | JoinType::RightOuter
            )
        {
            return Ok(());
        }
        let Some(l) = &self.left.current else {
            return Ok(());
        };
        let source = self.out.left_source(&l.batch, l.serial);
        for (at, &m) in slice.matched.iter().enumerate() {
            let wanted = match t {
                JoinType::LeftSemi => m,
                JoinType::Existence => true,
                _ => !m,
            };
            if !wanted {
                continue;
            }
            self.out.left_rows.push((source, slice.start + at));
            if t.pads_probe() {
                self.out.right_rows.push((0, 0));
            }
            if t == JoinType::Existence {
                self.out.exists.push(m);
            }
        }
        Ok(())
    }

    /// The right rows the join returns by their marks; then the group goes.
    async fn drain(&mut self) -> Result<()> {
        let t = self.shared.join_type;
        if !t.builds_preserved() || !self.next_chunk().await? {
            self.end_group();
            return Ok(());
        }
        self.phase = Phase::Drain;
        let Some(c) = self.chunks.current.clone() else {
            return Ok(());
        };
        let marked = self.shared.filter.is_some();
        let base = self.chunks.base;
        while self.chunks.row < c.end && self.out.len() < self.batch_rows {
            let r = self.chunks.row;
            self.chunks.row += 1;
            let matched = if marked {
                self.group.marked(base + (r - c.start) as u64)
            } else {
                self.group.any_left
            };
            if matched == (t == JoinType::RightSemi) {
                let source = self.out.right_source(&c.batch, c.serial);
                self.out.right_rows.push((source, r));
                if !matches!(t, JoinType::RightSemi | JoinType::RightAnti) {
                    self.out.left_rows.push((0, 0));
                }
            }
        }
        Ok(())
    }

    fn end_group(&mut self) {
        let held = std::mem::take(&mut self.group.bytes);
        self.memory.shrink(held.min(self.memory.size()));
        let g = &mut self.group;
        // Keeps their buffers for the next group.
        g.chunks.clear();
        g.marks.clear();
        g.file = None;
        g.rows = 0;
        g.any_left = false;
        self.chunks = Chunks::default();
        self.phase = Phase::Compare;
    }

    /// Up to a batch of the gathered output rows.
    fn flush(&mut self) -> Result<Batch> {
        let t = self.shared.join_type;
        let rows = self.batch_rows.min(self.out.len());
        let (left, right) = match t {
            JoinType::Inner | JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter => {
                (true, true)
            }
            JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::NullAwareAnti
            | JoinType::Existence => (true, false),
            JoinType::RightSemi | JoinType::RightAnti => (false, true),
        };
        let mut columns: Vec<ArrayRef> = Vec::new();
        if left {
            let idx: Vec<_> = self.out.left_rows.drain(..rows).collect();
            columns.extend(gather(&self.out.left, &idx, self.left_columns)?);
        }
        if right {
            let idx: Vec<_> = self.out.right_rows.drain(..rows).collect();
            columns.extend(gather(&self.out.right, &idx, self.right_columns)?);
        }
        if t == JoinType::Existence {
            let flags: Vec<bool> = self.out.exists.drain(..rows).collect();
            columns.push(Arc::new(BooleanArray::from(flags)));
        }
        if self.out.len() == 0 {
            self.out.reset();
        } else {
            self.out.compact();
        }
        Ok(Batch::new(record(&self.shared.schema, columns, rows)?))
    }
}

/// The first `columns` columns of the rows `idx` (`(source, row)`).
fn gather(
    sources: &[RecordBatch],
    idx: &[(usize, usize)],
    columns: usize,
) -> Result<Vec<ArrayRef>> {
    (0..columns)
        .map(|c| {
            let arrays: Vec<&dyn Array> = sources.iter().map(|b| b.column(c).as_ref()).collect();
            interleave(&arrays, idx).map_err(|e| internal(e.to_string()))
        })
        .collect()
}

/// What claims a bucket pair: see the module docs.
#[derive(Default)]
struct Claims {
    state: Mutex<ClaimState>,
    changed: Event,
}

#[derive(Default)]
struct ClaimState {
    next: usize,
    active: usize,
}

/// A driver's hold on its bucket pair, given up when dropped.
struct Slot(Arc<Claims>);

impl Drop for Slot {
    fn drop(&mut self) {
        {
            let mut s = lock(&self.0.state);
            s.active = s.active.saturating_sub(1);
        }
        self.0.changed.notify();
    }
}

/// A sort-merge join's source over its two finalized sorts.
struct MergeGlobal {
    left: GlobalState,
    right: GlobalState,
    buckets: usize,
    claims: Arc<Claims>,
}

struct MergeLocal {
    env: MergeEnv,
    merge: Option<(Merge, Slot)>,
    group_limit: usize,
}

/// A sort-merge join: its two sorts and the source that merges them.
pub struct SortMergeJoin {
    shared: Arc<Shared>,
    left_sort: Arc<Sort>,
    right_sort: Arc<Sort>,
    left_keys: Arc<KeyEncoder>,
    right_keys: Arc<KeyEncoder>,
    left_strict: Arc<[usize]>,
    right_strict: Arc<[usize]>,
    left_schema: SchemaRef,
    right_schema: SchemaRef,
    counters: Arc<MergeCounters>,
}

impl fmt::Debug for SortMergeJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SortMergeJoin")
            .field("join_type", &self.shared.join_type)
            .finish_non_exhaustive()
    }
}

/// Whether a sort-merge join can run with `keys` and these inputs: its key
/// types can be ordered (Spark's `RowOrdering.isOrderable`) and no build column
/// changes type through the hash join's row payload (a dictionary).
#[must_use]
pub fn merge_join_supported(
    build: &arrow_schema::Schema,
    probe: &arrow_schema::Schema,
    probe_keys: &[usize],
    join_type: JoinType,
) -> bool {
    let keys: Vec<SortKey> = probe_keys.iter().map(|&k| SortKey::asc(k)).collect();
    join_type != JoinType::NullAwareAnti
        && KeyEncoder::new(probe, &keys).is_ok()
        && !build
            .fields()
            .iter()
            .any(|f| matches!(f.data_type(), DataType::Dictionary(..)))
}

impl SortMergeJoin {
    /// Validates `spec` as [`HashJoin::try_new`](super::HashJoin::try_new)
    /// does, and the keys' types as a sort's.
    ///
    /// # Errors
    /// As the hash join's; `KRISHIV_UNSUPPORTED_FEATURE` for a null-aware anti
    /// join or a key type the sort cannot order.
    pub fn try_new(spec: JoinSpec) -> Result<Self> {
        if spec.join_type == JoinType::NullAwareAnti {
            return Err(super::unsupported("a null-aware anti sort-merge join"));
        }
        let (left_schema, right_schema) = (
            Arc::clone(&spec.probe_schema),
            Arc::clone(&spec.build_schema),
        );
        let sort_keys = |keys: &[usize]| keys.iter().map(|&k| SortKey::asc(k)).collect::<Vec<_>>();
        let (lk, rk) = (sort_keys(&spec.probe_keys), sort_keys(&spec.build_keys));
        let strict = |keys: &[usize], null_safe: &[bool]| -> Arc<[usize]> {
            keys.iter()
                .zip(null_safe)
                .filter(|&(_, &n)| !n)
                .map(|(&k, _)| k)
                .collect()
        };
        let left_strict = strict(&spec.probe_keys, &spec.null_safe);
        let right_strict = strict(&spec.build_keys, &spec.null_safe);
        let scope = spec.spill.clone();
        let left_sort = Sort::new(Arc::clone(&left_schema), &lk, scope.clone())?
            .with_hash_buckets(&spec.probe_keys)?
            .read_by_bucket();
        let right_sort = Sort::new(Arc::clone(&right_schema), &rk, scope)?
            .with_hash_buckets(&spec.build_keys)?
            .read_by_bucket();
        let left_keys = KeyEncoder::new(&left_schema, &lk)?;
        let right_keys = KeyEncoder::new(&right_schema, &rk)?;
        let dictionary = |s: &SchemaRef| {
            s.fields()
                .iter()
                .any(|f| matches!(f.data_type(), DataType::Dictionary(..)))
        };
        if dictionary(&right_schema) {
            return Err(super::unsupported(
                "a sort-merge join over dictionary columns",
            ));
        }
        let shared = Shared::new(spec)?;
        Ok(Self {
            shared: Arc::new(shared),
            left_sort: Arc::new(left_sort),
            right_sort: Arc::new(right_sort),
            left_keys,
            right_keys,
            left_strict,
            right_strict,
            left_schema,
            right_schema,
            counters: Arc::default(),
        })
    }

    /// The output schema.
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.shared.schema)
    }
}

impl Source for SortMergeJoin {
    fn name(&self) -> &'static str {
        "SortMergeJoin"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.shared.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let missing = || internal("a sort-merge join read before its sorts ended".to_owned());
        let left = ctx.dependency().cloned().ok_or_else(missing)?;
        let right = ctx.second_dependency().cloned().ok_or_else(missing)?;
        let buckets = Sort::bucket_count(&left)?;
        if Sort::bucket_count(&right)? != buckets {
            return Err(internal(
                "a merge over sorts of different buckets".to_owned(),
            ));
        }
        Ok(Arc::new(MergeGlobal {
            left,
            right,
            buckets,
            claims: Arc::default(),
        }))
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global_ref::<MergeGlobal>(global, "SortMergeJoin").map_or(1, |g| g.buckets)
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(MergeLocal {
            env: MergeEnv::new(ctx),
            merge: None,
            group_limit: group_bytes(
                ctx.query().config().join_merge_buffer_bytes,
                ctx.consumer().consumer(),
                ctx.executor().parallelism(),
            ),
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let l = local_mut::<MergeLocal>(local, "SortMergeJoin")?;
            let g = global_ref::<MergeGlobal>(global, "SortMergeJoin")?;
            loop {
                if let Some((merge, _slot)) = &mut l.merge {
                    if let Some(batch) = merge.next().await? {
                        return Ok(Some(batch));
                    }
                    l.merge = None;
                    l.env.handle.release_credit();
                }
                let Some((bucket, slot)) = self.claim(g, &l.env).await? else {
                    return Ok(None);
                };
                let spec = MergeSpec {
                    left: MergeInput {
                        sort: Arc::clone(&self.left_sort),
                        global: Arc::clone(&g.left),
                        local: self.left_sort.bucket_reader(&l.env, bucket),
                    },
                    right: MergeInput {
                        sort: Arc::clone(&self.right_sort),
                        global: Arc::clone(&g.right),
                        local: self.right_sort.bucket_reader(&l.env, bucket),
                    },
                    left_keys: Arc::clone(&self.left_keys),
                    right_keys: Arc::clone(&self.right_keys),
                    left_strict: Arc::clone(&self.left_strict),
                    right_strict: Arc::clone(&self.right_strict),
                    left_schema: Arc::clone(&self.left_schema),
                    right_schema: Arc::clone(&self.right_schema),
                    left_columns: self.left_schema.fields().len(),
                    right_columns: self.right_schema.fields().len(),
                    group_limit: l.group_limit,
                    counters: Arc::clone(&self.counters),
                };
                let merge = Merge::new(Arc::clone(&self.shared), spec, &l.env)?;
                l.merge = Some((merge, slot));
            }
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        vec![
            (
                std::borrow::Cow::Borrowed("groups_spilled"),
                Metric::Count(self.counters.groups_spilled.load(Ordering::Relaxed)),
            ),
            (
                std::borrow::Cow::Borrowed("spill_file_bytes"),
                Metric::Count(self.counters.spill_bytes.load(Ordering::Relaxed)),
            ),
        ]
    }
}

impl SortMergeJoin {
    /// The next bucket pair for a driver and its slot, `None` when none is
    /// left; see the module docs.
    async fn claim(&self, g: &MergeGlobal, env: &MergeEnv) -> Result<Option<(usize, Slot)>> {
        let batch_rows = env.batch_rows();
        loop {
            let changed = g.claims.changed.listen();
            {
                let mut s = lock(&g.claims.state);
                if s.next >= g.buckets {
                    return Ok(None);
                }
                let b = s.next;
                let need = Sort::bucket_need(&g.left, b, batch_rows)?
                    .saturating_add(Sort::bucket_need(&g.right, b, batch_rows)?);
                let pool = env.handle.pool();
                let spare = need == 0
                    || pool
                        .limit()
                        .is_none_or(|limit| pool.reserved().saturating_add(need) <= limit / 2);
                if s.active == 0 || spare {
                    s.next += 1;
                    s.active += 1;
                    return Ok(Some((b, Slot(Arc::clone(&g.claims)))));
                }
            }
            env.query.check_cancelled()?;
            changed.await;
        }
    }
}

/// The plan node of a [`SortMergeJoin`].
pub struct SortMergeJoinNode {
    /// The join.
    pub join: Arc<SortMergeJoin>,
    /// The probe (left) input.
    pub probe: Arc<dyn ExecNode>,
    /// The build (right) input.
    pub build: Arc<dyn ExecNode>,
}

impl fmt::Debug for SortMergeJoinNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SortMergeJoinNode")
            .field("join", &self.join)
            .finish_non_exhaustive()
    }
}

impl ExecNode for SortMergeJoinNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let left = b.child_pipeline(
            Arc::clone(&self.join.left_sort) as Arc<dyn Sink>,
            Arc::clone(&self.probe),
        )?;
        let right = b.child_pipeline(
            Arc::clone(&self.join.right_sort) as Arc<dyn Sink>,
            Arc::clone(&self.build),
        )?;
        b.set_merge_source(Arc::clone(&self.join) as Arc<dyn Source>, left, right)
            .map(drop)
    }
}
