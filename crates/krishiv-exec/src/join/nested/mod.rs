// SPDX-License-Identifier: Apache-2.0
//! Nested-loop join (Spark's `BroadcastNestedLoopJoinExec`): joins without
//! an equality condition, such as a cross join or a join on `a < b` or on a
//! disjunction (design P2-T052b §2.2).
//!
//! The **build** side is collected and reserved; under memory pressure it
//! spills (`build.rs`). The **probe** side streams: each step pairs a chunk
//! of probe rows with a slice of the build rows (at most [`TARGET_PAIRS`]
//! pairs), evaluates the condition on the pairs (probe columns, then build
//! columns; NULL is not a match) and emits:
//! - inner: the matching pairs;
//! - left outer: the matching pairs, and each probe row without a match
//!   with NULL build columns once the chunk met every build row;
//! - left semi / anti: each probe row with / without a match, once;
//! - existence: each probe row once with `exists` (whether it matched;
//!   never NULL). Spark plans `k NOT IN (uncorrelated subquery)` outside a
//!   filter conjunct this way, as the negation of the existence join on
//!   `(k = c) OR isnull(k = c)`;
//! - right outer and full outer: the matching pairs (and, full outer, the
//!   unmatched probe rows as left outer does); right semi / anti emit
//!   nothing while probing. Each match sets its build row's bit in the
//!   join's build-match bitmap ([`Marks`]); once every probe driver's input
//!   ended, the flush phase emits the build rows the join returns (never
//!   matched, or for right semi matched), padded with NULL probe columns
//!   for the outer joins. On one worker the drivers OR into one bitmap: the
//!   fused lowering of the note's local → `Exchange(Single)` → final split.
//!
//! **Spilled build** (block nested loop, `probe.rs`): when the build spilled,
//! the probe drivers buffer their input and spill it too; after the barrier
//! each driver reads the build in chunks that fit a share of the pool and
//! re-reads its probe rows once per chunk. A per-driver probe-row bitmap
//! carries "matched in some chunk", so the probe-side rows of the outer,
//! semi, anti and existence joins come after the last chunk; the build rows
//! of the build-side joins come from the last driver to finish.
//!
//! Rows must be insert-only (weight 1); weighted input is
//! `KRISHIV_UNSUPPORTED_FEATURE` until IVM, as for the hash join's build.

mod build;
mod probe;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array, new_null_array,
};
use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_memory::SpillScope;
use krishiv_types::batch::{Batch, Weights};

use super::{JoinType, unsupported};
use crate::error::{Result, internal};
use crate::operator::{Operator, Sink};
use crate::operators::Predicate;
use crate::pipeline::{ExecNode, PipelineBuilder};

/// The most (probe row, build row) pairs one call evaluates.
pub(crate) const TARGET_PAIRS: usize = 8192;

/// What a nested-loop join joins.
#[derive(Debug)]
pub struct NestedLoopSpec {
    /// Inner, left or right outer, full outer, left or right semi or anti,
    /// or existence.
    pub join_type: JoinType,
    /// The probe (left) input's schema.
    pub probe_schema: SchemaRef,
    /// The build (right) input's schema.
    pub build_schema: SchemaRef,
    /// The condition over probe then build columns; `None` joins every
    /// pair (a cross join).
    pub filter: Option<Arc<dyn Predicate>>,
    /// Where the build and, once it spilled, the probe rows spill; without
    /// one the build stays in memory.
    pub spill: Option<SpillScope>,
}

/// Operator counters for `EXPLAIN ANALYZE`.
#[derive(Debug, Default)]
struct Counters {
    /// Build spill files written.
    build_spills: AtomicU64,
    /// Probe spill files written.
    probe_spills: AtomicU64,
    /// Build chunks a probe driver joined from disk.
    chunks: AtomicU64,
}

#[derive(Debug)]
struct Shared {
    spec: NestedLoopSpec,
    /// Probe columns then build columns, as the condition reads them.
    pairs: SchemaRef,
    schema: SchemaRef,
    counters: Counters,
}

impl Shared {
    fn join_type(&self) -> JoinType {
        self.spec.join_type
    }

    /// Whether a match decides which probe rows come out after the probe
    /// rows met every build row (the probe-row bitmap of a spilled build).
    fn tracks_probe(&self) -> bool {
        matches!(
            self.join_type(),
            JoinType::LeftOuter
                | JoinType::FullOuter
                | JoinType::LeftSemi
                | JoinType::LeftAnti
                | JoinType::Existence
        )
    }

    /// Whether matching pairs are output rows.
    fn emits_pairs(&self) -> bool {
        matches!(
            self.join_type(),
            JoinType::Inner | JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter
        )
    }
}

/// A nested-loop join: its build [`Sink`] and probe [`Operator`].
#[derive(Debug, Clone)]
pub struct NestedLoopJoin(Arc<Shared>);

impl NestedLoopJoin {
    /// Validates `spec`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for the null-aware anti join (it needs
    /// an equality key; the planner uses the hash join).
    pub fn try_new(spec: NestedLoopSpec) -> Result<Self> {
        if spec.join_type == JoinType::NullAwareAnti {
            return Err(unsupported("a null-aware anti join without a key"));
        }
        let fields = |probe_nullable: bool, build_nullable: bool| -> Vec<Field> {
            let nullable =
                |f: &Arc<Field>, n: bool| f.as_ref().clone().with_nullable(f.is_nullable() || n);
            spec.probe_schema
                .fields()
                .iter()
                .map(|f| nullable(f, probe_nullable))
                .chain(
                    spec.build_schema
                        .fields()
                        .iter()
                        .map(|f| nullable(f, build_nullable)),
                )
                .collect()
        };
        let pairs = Arc::new(Schema::new(fields(false, false)));
        let schema = match spec.join_type {
            JoinType::Inner => Arc::clone(&pairs),
            JoinType::LeftOuter => Arc::new(Schema::new(fields(false, true))),
            JoinType::RightOuter => Arc::new(Schema::new(fields(true, false))),
            JoinType::FullOuter => Arc::new(Schema::new(fields(true, true))),
            JoinType::Existence => super::existence_schema(&spec.probe_schema),
            JoinType::RightSemi | JoinType::RightAnti => Arc::clone(&spec.build_schema),
            _ => Arc::clone(&spec.probe_schema),
        };
        Ok(Self(Arc::new(Shared {
            spec,
            pairs,
            schema,
            counters: Counters::default(),
        })))
    }

    /// The output schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.0.schema)
    }

    /// The build side's sink.
    pub fn build_sink(&self) -> Arc<dyn Sink> {
        Arc::new(build::BuildSink(Arc::clone(&self.0)))
    }

    /// The probe operator.
    pub fn probe(&self) -> Arc<dyn Operator> {
        Arc::new(probe::ProbeOp(Arc::clone(&self.0)))
    }
}

fn unit_weights(b: &Batch) -> Result<()> {
    match b.weights() {
        Weights::Const(1) => Ok(()),
        _ => Err(unsupported("nested-loop joins of weighted rows")),
    }
}

/// One bit per build row: whether some probe row matched it. Probe drivers
/// set bits as they match; the flush phase reads them once every driver's
/// input ended.
#[derive(Debug)]
struct Marks(Box<[AtomicU64]>);

impl Marks {
    fn new(rows: u64) -> Self {
        Self((0..rows.div_ceil(64)).map(|_| AtomicU64::new(0)).collect())
    }

    /// The bytes it holds for `rows` build rows.
    fn bytes(rows: u64) -> usize {
        usize::try_from(rows.div_ceil(64))
            .unwrap_or(usize::MAX)
            .saturating_mul(8)
    }

    fn word(&self, row: u64) -> Option<(&AtomicU64, u64)> {
        let w = self.0.get(usize::try_from(row / 64).ok()?)?;
        Some((w, 1 << (row % 64)))
    }

    fn set(&self, row: u64) {
        if let Some((w, bit)) = self.word(row)
            && w.load(Ordering::Relaxed) & bit == 0
        {
            w.fetch_or(bit, Ordering::Relaxed);
        }
    }

    fn get(&self, row: u64) -> bool {
        self.word(row)
            .is_some_and(|(w, bit)| w.load(Ordering::Relaxed) & bit != 0)
    }
}

/// One driver's probe rows (a spilled build): whether each matched in some
/// build chunk.
#[derive(Debug, Default)]
struct ProbeBits(Vec<u64>);

impl ProbeBits {
    fn words(rows: usize) -> usize {
        rows.div_ceil(64)
    }

    fn set(&mut self, row: usize) {
        if let Some(w) = self.0.get_mut(row / 64) {
            *w |= 1 << (row % 64);
        }
    }

    fn get(&self, row: usize) -> bool {
        self.0
            .get(row / 64)
            .is_some_and(|w| w & (1 << (row % 64)) != 0)
    }
}

/// What happens to the probe rows of a probe chunk once they met every
/// build row.
enum Rest<'a> {
    /// Emitted at once (the build is all in memory).
    Emit,
    /// Recorded in the driver's probe bitmap, the probe batch's first row
    /// being row `base` of the driver's probe rows (a spilled build).
    Record(&'a mut ProbeBits, usize),
}

/// Progress of one probe batch through a list of build batches.
#[derive(Debug, Default)]
struct Pairing {
    /// The probe chunk `[start, end)` being joined.
    start: usize,
    end: usize,
    /// The build batch and row the next slice starts at.
    batch: usize,
    offset: usize,
    /// Per probe row of the chunk: whether it matched.
    matched: Vec<bool>,
}

/// The build rows a pairing step reads: batches and the build row number
/// of each batch's first row (the index into [`Marks`]).
#[derive(Clone, Copy)]
struct BuildRows<'a> {
    batches: &'a [RecordBatch],
    bases: &'a [u64],
}

impl Shared {
    /// One step of pairing `probe` with `build`: at most [`TARGET_PAIRS`]
    /// pairs. Returns its output, if any, and whether every probe row met
    /// every build row (then `cur` is reset for the next probe batch).
    fn step(
        &self,
        probe: &RecordBatch,
        build: BuildRows<'_>,
        cur: &mut Pairing,
        partition: u32,
        marks: Option<&Marks>,
        rest: Rest<'_>,
    ) -> Result<(Option<RecordBatch>, bool)> {
        let rows = probe.num_rows();
        if rows == 0 {
            return Ok((None, true));
        }
        if cur.start == cur.end {
            // A new probe chunk: as many rows as make `TARGET_PAIRS` pairs
            // with the whole build.
            let total: usize = build.batches.iter().map(RecordBatch::num_rows).sum();
            let chunk = (TARGET_PAIRS / total.clamp(1, TARGET_PAIRS)).max(1);
            cur.end = (cur.start + chunk).min(rows);
            cur.matched = vec![false; cur.end - cur.start];
            (cur.batch, cur.offset) = (0, 0);
        }
        let mut out = Vec::new();
        while cur.batch < build.batches.len() && build.batches[cur.batch].num_rows() == cur.offset {
            (cur.batch, cur.offset) = (cur.batch + 1, 0);
        }
        if let Some(b) = build.batches.get(cur.batch) {
            let n = (b.num_rows() - cur.offset).min((TARGET_PAIRS / (cur.end - cur.start)).max(1));
            let slice = b.slice(cur.offset, n);
            let base = build.bases.get(cur.batch).copied().unwrap_or(0) + cur.offset as u64;
            cur.offset += n;
            let (start, end) = (cur.start, cur.end);
            let marks = marks.map(|m| (m, base));
            if let Some(pairs) = self.pairs(
                probe,
                start,
                end,
                &slice,
                &mut cur.matched,
                partition,
                marks,
            )? {
                out.push(pairs);
            }
        }
        let chunk_done = cur.batch >= build.batches.len()
            || (cur.batch + 1 == build.batches.len()
                && cur.offset == build.batches[cur.batch].num_rows());
        if chunk_done {
            match rest {
                Rest::Emit => {
                    if let Some(r) = self.unmatched(probe, cur.start, &cur.matched)? {
                        out.push(r);
                    }
                }
                Rest::Record(bits, base) => {
                    for (i, _) in cur.matched.iter().enumerate().filter(|(_, m)| **m) {
                        bits.set(base + cur.start + i);
                    }
                }
            }
            cur.start = cur.end;
        }
        let all_done = chunk_done && cur.end == rows;
        if all_done {
            (cur.start, cur.end) = (0, 0);
        }
        let batch = match out.len() {
            0 => None,
            1 => out.pop(),
            _ => Some(
                arrow_select::concat::concat_batches(&self.schema, &out)
                    .map_err(|e| internal(e.to_string()))?,
            ),
        };
        Ok((batch, all_done))
    }

    /// Pairs probe rows `start..end` with every row of `build`, evaluates
    /// the condition, marks matched probe rows (and, with `marks`, the
    /// matched build rows, the slice's first being build row `base`); the
    /// matching pairs of the joins that output them.
    #[expect(
        clippy::too_many_arguments,
        reason = "one call site, all distinct inputs"
    )]
    fn pairs(
        &self,
        probe: &RecordBatch,
        start: usize,
        end: usize,
        build: &RecordBatch,
        matched: &mut [bool],
        partition: u32,
        marks: Option<(&Marks, u64)>,
    ) -> Result<Option<RecordBatch>> {
        let b = build.num_rows();
        if b == 0 {
            return Ok(None);
        }
        let n = (end - start) * b;
        let idx = |f: &dyn Fn(usize) -> usize| -> Result<UInt32Array> {
            (0..n)
                .map(|i| u32::try_from(f(i)).map_err(|_| internal("batch too large".to_owned())))
                .collect()
        };
        let p_idx = idx(&|i| start + i / b)?;
        let b_idx = idx(&|i| i % b)?;
        let take = |a: &ArrayRef, i: &UInt32Array| {
            arrow_select::take::take(a.as_ref(), i, None).map_err(|e| internal(e.to_string()))
        };
        let mut columns = probe
            .columns()
            .iter()
            .map(|c| take(c, &p_idx))
            .collect::<Result<Vec<_>>>()?;
        for c in build.columns() {
            columns.push(take(c, &b_idx)?);
        }
        let options = RecordBatchOptions::new().with_row_count(Some(n));
        let paired = RecordBatch::try_new_with_options(Arc::clone(&self.pairs), columns, &options)
            .map_err(|e| internal(e.to_string()))?;
        let mask = match &self.spec.filter {
            None => BooleanArray::from(vec![true; n]),
            Some(f) => {
                let m = f.evaluate_in(&paired, partition)?;
                // NULL is not a match.
                match m.nulls() {
                    Some(v) => BooleanArray::new(m.values() & v.inner(), None),
                    None => m,
                }
            }
        };
        for i in mask.values().set_indices() {
            matched[i / b] = true;
            if let Some((marks, base)) = marks {
                marks.set(base + (i % b) as u64);
            }
        }
        if !self.emits_pairs() || mask.true_count() == 0 {
            return Ok(None);
        }
        let out = arrow_select::filter::filter_record_batch(&paired, &mask)
            .map_err(|e| internal(e.to_string()))?;
        out.with_schema(Arc::clone(&self.schema))
            .map(Some)
            .map_err(|e| internal(e.to_string()))
    }

    /// Once probe rows `start..` met every build row: semi and anti rows,
    /// the null-padded probe rows of a left or full outer join, and every
    /// probe row with its `exists` of an existence join.
    fn unmatched(
        &self,
        probe: &RecordBatch,
        start: usize,
        matched: &[bool],
    ) -> Result<Option<RecordBatch>> {
        let join_type = self.join_type();
        let keep: Vec<u32> = matched
            .iter()
            .enumerate()
            .filter(|(_, m)| match join_type {
                JoinType::LeftSemi => **m,
                JoinType::LeftAnti | JoinType::LeftOuter | JoinType::FullOuter => !**m,
                JoinType::Existence => true,
                _ => false,
            })
            .map(|(i, _)| {
                u32::try_from(start + i).map_err(|_| internal("batch too large".to_owned()))
            })
            .collect::<Result<_>>()?;
        if keep.is_empty() {
            return Ok(None);
        }
        let idx = UInt32Array::from(keep);
        let mut columns = take_all(probe, &idx)?;
        match join_type {
            JoinType::LeftOuter | JoinType::FullOuter => {
                for f in self.spec.build_schema.fields() {
                    columns.push(new_null_array(f.data_type(), idx.len()));
                }
            }
            JoinType::Existence => {
                columns.push(Arc::new(BooleanArray::from(matched.to_vec())) as ArrayRef);
            }
            _ => {}
        }
        self.batch(columns, idx.len())
    }

    /// The rows of `build` (its first row being build row `base`) that a
    /// build-side join returns once every probe row was joined: matched
    /// ones for right semi, never-matched ones otherwise, padded with NULL
    /// probe columns for the outer joins.
    fn build_rows(
        &self,
        build: &RecordBatch,
        base: u64,
        marks: &Marks,
    ) -> Result<Option<RecordBatch>> {
        let want = self.join_type() == JoinType::RightSemi;
        let keep: Vec<u32> = (0..build.num_rows())
            .filter(|&i| marks.get(base + i as u64) == want)
            .map(|i| u32::try_from(i).map_err(|_| internal("batch too large".to_owned())))
            .collect::<Result<_>>()?;
        if keep.is_empty() {
            return Ok(None);
        }
        let idx = UInt32Array::from(keep);
        let mut columns = Vec::new();
        if matches!(self.join_type(), JoinType::RightOuter | JoinType::FullOuter) {
            for f in self.spec.probe_schema.fields() {
                columns.push(new_null_array(f.data_type(), idx.len()));
            }
        }
        columns.extend(take_all(build, &idx)?);
        self.batch(columns, idx.len())
    }

    fn batch(&self, columns: Vec<ArrayRef>, rows: usize) -> Result<Option<RecordBatch>> {
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        RecordBatch::try_new_with_options(Arc::clone(&self.schema), columns, &options)
            .map(Some)
            .map_err(|e| internal(e.to_string()))
    }
}

fn take_all(batch: &RecordBatch, idx: &UInt32Array) -> Result<Vec<ArrayRef>> {
    batch
        .columns()
        .iter()
        .map(|c| {
            arrow_select::take::take(c.as_ref(), idx, None).map_err(|e| internal(e.to_string()))
        })
        .collect()
}

/// A plan node joining `probe` (left) with `build` (right) by nested loops.
pub struct NestedLoopJoinNode {
    /// The join.
    pub join: NestedLoopJoin,
    /// The build (right) input.
    pub build: Arc<dyn ExecNode>,
    /// The probe (left) input.
    pub probe: Arc<dyn ExecNode>,
}

impl std::fmt::Debug for NestedLoopJoinNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NestedLoopJoinNode")
            .field("join", &self.join)
            .finish_non_exhaustive()
    }
}

impl ExecNode for NestedLoopJoinNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(self.join.build_sink(), Arc::clone(&self.build))?;
        Arc::clone(&self.probe).build_pipelines(b)?;
        b.push_operator_reading(self.join.probe(), sink).map(drop)
    }
}
