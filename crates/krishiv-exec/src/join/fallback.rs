// SPDX-License-Identifier: Apache-2.0
//! A spilled partition repartitioning cannot split (one key too large for
//! memory) joins by sort-merge (P2-T056): both sides are sorted on their
//! encoded key bytes and [`Merge`] walks them, spilling the large key's group.
//! The threshold is the design's provisional one, recursion depth or no
//! shrink (`grace::step`), until the E11 numbers replace it in P5.
//!
//! Build rows arrive as arena rows, so a row keeps its key bytes (the arena's
//! key column, which is why row-layout keys with out-of-line strings stay out:
//! their bytes do not compare) and has its columns decoded; probe rows get the
//! same key bytes. A NULL in that column marks a row that matches nothing.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, make_array};
use arrow_buffer::NullBuffer;
use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_common::error::KrishivError;
use krishiv_memory::{ConsumerHandle, SpillFile, SpillReader};
use krishiv_types::batch::Batch;

use super::grace::{Part, lend_scratch};
use super::keys::Hashing;
use super::merge::{Merge, MergeEnv, MergeInput, MergeSpec, group_bytes};
use super::table::{NULL_KEY, RowArena};
use super::{JoinType, Shared, record};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{FlushOutput, GlobalState, LocalState, Sink, SinkStatus};
use crate::operators::{KeyEncoder, Sort, SortKey};

/// The arena's key column (after the hash).
const ARENA_KEY: usize = 1;

/// Whether a partition of this join can fall back: its key bytes compare and
/// it can spill. `NOT IN` reads build-wide facts a merge does not keep.
pub(super) fn possible(shared: &Shared, hashing: &Hashing) -> bool {
    let comparable = shared
        .keys
        .layout()
        .filter(|_| hashing.rows())
        .is_none_or(|l| !l.has_views());
    comparable && shared.spill.is_some() && shared.join_type != JoinType::NullAwareAnti
}

/// One side's rows being read, in memory or from their files.
struct Rows {
    /// Reading's memory, apart from the sort's consumer: a sort's spill returns
    /// all the credit of its own.
    handle: Arc<ConsumerHandle>,
    part: VecDeque<Batch>,
    files: VecDeque<SpillFile>,
    reader: Option<SpillReader>,
    /// Holds the in-memory rows' reservations until they are read.
    _memory: Part,
}

impl Rows {
    fn new(mut part: Part, handle: Arc<ConsumerHandle>) -> Self {
        Self {
            handle,
            part: std::mem::take(&mut part.chunks).into(),
            files: std::mem::take(&mut part.files).into(),
            reader: None,
            _memory: part,
        }
    }

    async fn next(&mut self, ctx: &TaskContext) -> Result<Option<Batch>> {
        if let Some(b) = self.part.pop_front() {
            return Ok(Some(b));
        }
        loop {
            let reader = match self.reader.take() {
                Some(r) => r,
                None => {
                    let Some(file) = self.files.pop_front() else {
                        return Ok(None);
                    };
                    // Draws the headroom lent as credit, and gives it back.
                    let memory = self.handle.reservation().releasing_to_credit();
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
                self.reader = Some(reader);
                return Ok(next);
            }
        }
    }
}

/// One side's sort.
struct Side {
    sort: Arc<Sort>,
    global: GlobalState,
    local: Option<LocalState>,
    schema: SchemaRef,
    keys: Arc<KeyEncoder>,
    /// The key bytes' column.
    key: usize,
    columns: usize,
}

impl Side {
    fn new(fields: Vec<Field>, key: &Field, shared: &Shared, ctx: &TaskContext) -> Result<Self> {
        let columns = fields.len();
        let mut all = fields;
        all.push(key.clone().with_nullable(true));
        let schema = Arc::new(Schema::new(all));
        let order = [SortKey::asc(columns)];
        let sort = Arc::new(
            Sort::new(Arc::clone(&schema), &order, shared.spill.clone())?.read_by_bucket(),
        );
        let global = Sink::global_state(&*sort, ctx)?;
        let local = Sink::local_state(&*sort, ctx, &global)?;
        Ok(Self {
            keys: KeyEncoder::new(&schema, &order)?,
            sort,
            global,
            local: Some(local),
            schema,
            key: columns,
            columns,
        })
    }

    /// Sinks `batch`, a batch at a time, spilling the sort for its memory.
    async fn feed(&mut self, batch: &Batch, ctx: &TaskContext) -> Result<()> {
        let local = self
            .local
            .as_mut()
            .ok_or_else(|| internal("a fallback's sort fed after its end".to_owned()))?;
        let rows = ctx.query().config().batch_size.max(1);
        for start in (0..batch.num_rows()).step_by(rows) {
            let slice = batch
                .slice(start, rows.min(batch.num_rows() - start))
                .map_err(|e| internal(e.to_string()))?;
            while let SinkStatus::NeedMemory { bytes } =
                Sink::sink(&*self.sort, slice.clone(), local, &self.global)?
            {
                let freed = Sink::spill(&*self.sort, local, bytes, ctx).await?;
                if freed == 0 {
                    lend_scratch(ctx, bytes).await?;
                }
            }
        }
        Ok(())
    }

    async fn finish(&mut self, ctx: &TaskContext) -> Result<()> {
        let local = self
            .local
            .take()
            .ok_or_else(|| internal("a fallback's sort ended twice".to_owned()))?;
        Sink::combine(&*self.sort, local, &self.global)?;
        Sink::finalize(&*self.sort, &self.global, ctx).await
    }

    fn input(&self, env: &MergeEnv, handle: &Arc<ConsumerHandle>) -> MergeInput {
        MergeInput {
            sort: Arc::clone(&self.sort),
            global: Arc::clone(&self.global),
            local: self.sort.bucket_reader(env, handle, 0),
        }
    }
}

/// `key` with NULLs where `valid` says, for a row that matches nothing.
fn with_nulls(key: &ArrayRef, valid: Option<NullBuffer>) -> Result<ArrayRef> {
    let Some(valid) = valid else {
        return Ok(Arc::clone(key));
    };
    let data = key
        .to_data()
        .into_builder()
        .nulls(Some(valid))
        .build()
        .map_err(|e| internal(e.to_string()))?;
    Ok(make_array(data))
}

/// A partition's sort-merge join: sorts its two sides, then merges them.
pub(super) struct Fallback {
    build: Rows,
    probe: Rows,
    left: Side,
    right: Side,
    merge: Option<Merge>,
    fed_build: bool,
}

impl std::fmt::Debug for Fallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fallback").finish_non_exhaustive()
    }
}

impl Fallback {
    /// Starts on a partition. Reading its files is reserved up front.
    pub(super) async fn start(
        shared: &Shared,
        hashing: &Hashing,
        build: Part,
        probe: Part,
        ctx: &TaskContext,
    ) -> Result<Self> {
        let reading = ctx.register_plain_consumer("Join fallback read");
        let headroom = build.read_headroom().max(probe.read_headroom());
        let mut lent = reading.reservation();
        lent.grow(headroom).await.map_err(KrishivError::from)?;
        lent.shrink_to_credit(headroom);
        let arena = shared.arena_schema(hashing);
        let key = arena
            .fields()
            .get(ARENA_KEY)
            .ok_or_else(|| internal("a join row arena without keys".to_owned()))?;
        let fields =
            |s: &Schema| -> Vec<Field> { s.fields().iter().map(|f| f.as_ref().clone()).collect() };
        Ok(Self {
            left: Side::new(fields(&shared.probe_schema), key, shared, ctx)?,
            right: Side::new(shared.build_fields.clone(), key, shared, ctx)?,
            build: Rows::new(build, Arc::clone(&reading)),
            probe: Rows::new(probe, reading),
            merge: None,
            fed_build: false,
        })
    }

    /// One step; `None` once the partition is joined.
    pub(super) async fn step(
        &mut self,
        shared: &Arc<Shared>,
        hashing: &Hashing,
        ctx: &TaskContext,
    ) -> Result<Option<FlushOutput>> {
        if let Some(merge) = &mut self.merge {
            return Ok(merge.next().await?.map(FlushOutput::Batch));
        }
        ctx.query().check_cancelled()?;
        if !self.fed_build {
            match self.build.next(ctx).await? {
                Some(chunk) => {
                    let rows = build_rows(shared, hashing, &chunk, &self.right.schema)?;
                    self.right.feed(&rows, ctx).await?;
                }
                None => {
                    self.fed_build = true;
                    self.right.finish(ctx).await?;
                }
            }
            return Ok(Some(FlushOutput::Continue));
        }
        match self.probe.next(ctx).await? {
            Some(batch) => {
                let rows = probe_rows(shared, hashing, &batch, &self.left.schema)?;
                self.left.feed(&rows, ctx).await?;
            }
            None => {
                self.left.finish(ctx).await?;
                self.merge = Some(self.open(shared, ctx)?);
            }
        }
        Ok(Some(FlushOutput::Continue))
    }

    fn open(&self, shared: &Arc<Shared>, ctx: &TaskContext) -> Result<Merge> {
        let env = MergeEnv::new(ctx);
        let spec = MergeSpec {
            left: self.left.input(&env, &env.left),
            right: self.right.input(&env, &env.right),
            left_keys: Arc::clone(&self.left.keys),
            right_keys: Arc::clone(&self.right.keys),
            left_strict: Arc::from([self.left.key]),
            right_strict: Arc::from([self.right.key]),
            left_schema: Arc::clone(&self.left.schema),
            right_schema: Arc::clone(&self.right.schema),
            left_columns: self.left.columns,
            right_columns: self.right.columns,
            group_limit: group_bytes(
                ctx.query().config().join_merge_buffer_bytes,
                ctx.consumer().consumer(),
                1,
            ),
            counters: Arc::default(),
        };
        Merge::new(Arc::clone(shared), spec, &env)
    }
}

/// An arena chunk as decoded build columns and the key bytes.
fn build_rows(
    shared: &Shared,
    hashing: &Hashing,
    chunk: &Batch,
    schema: &SchemaRef,
) -> Result<Batch> {
    let data = chunk.data();
    let arena = RowArena::concat(shared.arena_schema(hashing), std::slice::from_ref(chunk))?;
    let rows = u32::try_from(arena.len()).map_err(|_| internal("a chunk too large".to_owned()))?;
    let picks: Vec<(usize, u32)> = (0..rows).map(|r| (0, r)).collect();
    let mut columns = shared.build_columns(hashing.rows(), &picks, |_| Some(&arena))?;
    // Rows the build flagged for a NULL `=` key match nothing.
    let valid = data
        .column_by_name(NULL_KEY)
        .and_then(|c| c.as_boolean_opt())
        .map(|flags| NullBuffer::new(!flags.values()));
    columns.push(with_nulls(data.column(ARENA_KEY), valid)?);
    Ok(Batch::new(record(schema, columns, arena.len())?))
}

/// A probe batch and its key bytes, as the build's rows hold them.
fn probe_rows(
    shared: &Shared,
    hashing: &Hashing,
    batch: &Batch,
    schema: &SchemaRef,
) -> Result<Batch> {
    let n = batch.num_rows();
    let cols: Vec<ArrayRef> = shared
        .probe_keys
        .iter()
        .map(|&k| Arc::clone(batch.data().column(k)))
        .collect();
    let keys = shared.keys.encode(&cols, hashing)?;
    let layout = shared.keys.layout().filter(|_| hashing.rows());
    let all: Vec<usize> = (0..n).collect();
    let bytes = keys.arena_columns(layout.map(AsRef::as_ref), &all)?;
    let key = bytes
        .first()
        .ok_or_else(|| internal("a join key without bytes".to_owned()))?;
    let valid = (!(0..n).all(|i| keys.is_valid(i)))
        .then(|| NullBuffer::from_iter((0..n).map(|i| keys.is_valid(i))));
    let mut columns = batch.data().columns().to_vec();
    columns.push(with_nulls(key, valid)?);
    Ok(Batch::new(record(schema, columns, n)?))
}
