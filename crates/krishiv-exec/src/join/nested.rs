// SPDX-License-Identifier: Apache-2.0
//! Nested-loop join (Spark's `BroadcastNestedLoopJoinExec`): joins without
//! an equality condition, such as a cross join or a join on `a < b` or on a
//! disjunction.
//!
//! The **build** side (the right input) is collected in memory, reserved,
//! and never spilled (as Spark broadcasts it). The **probe** side streams:
//! each call pairs a chunk of probe rows with a slice of the build rows
//! (at most [`TARGET_PAIRS`] pairs), evaluates the condition on the pairs
//! (probe columns, then build columns; NULL is not a match) and emits:
//! - inner: the matching pairs;
//! - left outer: the matching pairs, and each probe row without a match
//!   with NULL build columns once the chunk met every build row;
//! - left semi / anti: each probe row with / without a match, once.
//!
//! Rows must be insert-only (weight 1); weighted input is
//! `KRISHIV_UNSUPPORTED_FEATURE` until IVM, as for the hash join's build.

use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array, new_null_array,
};
use arrow_schema::{Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_memory::Reservation;
use krishiv_types::batch::{Batch, Weights};

use super::{JoinType, unsupported};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Operator, OperatorOutput, Sink, SinkStatus};
use crate::operators::Predicate;
use crate::pipeline::{ExecNode, PipelineBuilder};

/// The most (probe row, build row) pairs one call evaluates.
pub(crate) const TARGET_PAIRS: usize = 8192;

/// What a nested-loop join joins.
#[derive(Debug)]
pub struct NestedLoopSpec {
    /// Inner, left outer, left semi or left anti.
    pub join_type: JoinType,
    /// The probe (left) input's schema.
    pub probe_schema: SchemaRef,
    /// The build (right) input's schema.
    pub build_schema: SchemaRef,
    /// The condition over probe then build columns; `None` joins every
    /// pair (a cross join).
    pub filter: Option<Arc<dyn Predicate>>,
}

#[derive(Debug)]
struct Shared {
    spec: NestedLoopSpec,
    /// Probe columns then build columns, as the condition reads them.
    pairs: SchemaRef,
    schema: SchemaRef,
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
        let fields = |nullable: bool| -> Vec<Field> {
            spec.probe_schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .chain(spec.build_schema.fields().iter().map(|f| {
                    f.as_ref()
                        .clone()
                        .with_nullable(f.is_nullable() || nullable)
                }))
                .collect()
        };
        let pairs = Arc::new(Schema::new(fields(false)));
        let schema = match spec.join_type {
            JoinType::Inner => Arc::clone(&pairs),
            JoinType::LeftOuter => Arc::new(Schema::new(fields(true))),
            _ => Arc::clone(&spec.probe_schema),
        };
        Ok(Self(Arc::new(Shared {
            spec,
            pairs,
            schema,
        })))
    }

    /// The output schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.0.schema)
    }

    /// The build side's sink.
    pub fn build_sink(&self) -> Arc<dyn Sink> {
        Arc::new(BuildSink)
    }

    /// The probe operator.
    pub fn probe(&self) -> Arc<dyn Operator> {
        Arc::new(ProbeOp(Arc::clone(&self.0)))
    }
}

fn unit_weights(b: &Batch) -> Result<()> {
    match b.weights() {
        Weights::Const(1) => Ok(()),
        _ => Err(unsupported("nested-loop joins of weighted rows")),
    }
}

/// The build rows, once collected.
#[derive(Debug, Default)]
struct Built {
    batches: Vec<RecordBatch>,
    reservations: Vec<Reservation>,
}

struct BuildSink;

struct BuildLocal {
    batches: Vec<RecordBatch>,
    reservation: Reservation,
}

impl Sink for BuildSink {
    fn name(&self) -> &'static str {
        "NestedLoopJoinBuild"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(Mutex::new(Built::default())))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(BuildLocal {
            batches: Vec::new(),
            reservation: ctx.consumer().reservation(),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        if input.num_rows() == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        unit_weights(&input)?;
        let l = local
            .downcast_mut::<BuildLocal>()
            .ok_or_else(|| internal("nested-loop build state".to_owned()))?;
        let bytes = input.data().get_array_memory_size();
        if l.reservation.try_grow(bytes).is_err() {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        l.batches.push(input.into_parts().0);
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let l = local
            .downcast::<BuildLocal>()
            .map_err(|_| internal("nested-loop build state".to_owned()))?;
        let g = built(global)?;
        let mut g = g.lock().unwrap_or_else(PoisonError::into_inner);
        g.batches.extend(l.batches);
        g.reservations.push(l.reservation);
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

fn built(global: &GlobalState) -> Result<&Mutex<Built>> {
    global
        .downcast_ref::<Mutex<Built>>()
        .ok_or_else(|| internal("nested-loop build state".to_owned()))
}

struct ProbeOp(Arc<Shared>);

/// Progress through one probe batch.
struct ProbeLocal {
    build: Arc<[RecordBatch]>,
    /// The probe chunk `[start, end)` being joined.
    start: usize,
    end: usize,
    /// The build batch and row the next slice starts at.
    batch: usize,
    offset: usize,
    /// Per probe row of the chunk: whether it matched.
    matched: Vec<bool>,
}

impl Operator for ProbeOp {
    fn name(&self) -> &'static str {
        "NestedLoopJoin"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.0.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        let global = ctx
            .dependency()
            .ok_or_else(|| internal("nested-loop probe before its build".to_owned()))?;
        let g = built(global)?;
        let batches: Arc<[RecordBatch]> = g
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .batches
            .clone()
            .into();
        Ok(Box::new(ProbeLocal {
            build: batches,
            start: 0,
            end: 0,
            batch: 0,
            offset: 0,
            matched: Vec::new(),
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let rows = input.num_rows();
        if rows == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        unit_weights(input)?;
        let l = local
            .downcast_mut::<ProbeLocal>()
            .ok_or_else(|| internal("nested-loop probe state".to_owned()))?;
        let total: usize = l.build.iter().map(RecordBatch::num_rows).sum();
        let chunk = (TARGET_PAIRS / total.clamp(1, TARGET_PAIRS)).max(1);
        loop {
            if l.start == l.end {
                // A new chunk.
                l.end = (l.start + chunk).min(rows);
                l.matched = vec![false; l.end - l.start];
                (l.batch, l.offset) = (0, 0);
            }
            let mut out = Vec::new();
            // One build slice.
            while l.batch < l.build.len() && l.build[l.batch].num_rows() == l.offset {
                (l.batch, l.offset) = (l.batch + 1, 0);
            }
            if let Some(b) = l.build.get(l.batch) {
                let n = (b.num_rows() - l.offset).min((TARGET_PAIRS / (l.end - l.start)).max(1));
                let slice = b.slice(l.offset, n);
                l.offset += n;
                if let Some(pairs) =
                    self.pairs(input.data(), l.start, l.end, &slice, &mut l.matched)?
                {
                    out.push(pairs);
                }
            }
            let chunk_done = l.batch >= l.build.len()
                || (l.batch + 1 == l.build.len() && l.offset == l.build[l.batch].num_rows());
            if chunk_done {
                if let Some(rest) = self.unmatched(input.data(), l.start, &l.matched)? {
                    out.push(rest);
                }
                l.start = l.end;
            }
            let all_done = chunk_done && l.end == rows;
            let batch = match out.len() {
                0 => None,
                1 => out.pop(),
                _ => Some(
                    arrow_select::concat::concat_batches(&self.0.schema, &out)
                        .map_err(|e| internal(e.to_string()))?,
                ),
            };
            if all_done {
                (l.start, l.end) = (0, 0);
                return Ok(OperatorOutput::NeedMoreInput(batch.map(Batch::new)));
            }
            if let Some(batch) = batch {
                return Ok(OperatorOutput::HaveMoreOutput(Batch::new(batch)));
            }
        }
    }
}

impl ProbeOp {
    /// Pairs probe rows `start..end` with every row of `build`, evaluates
    /// the condition and marks matched probe rows; the matching pairs for
    /// inner and left outer joins.
    fn pairs(
        &self,
        probe: &RecordBatch,
        start: usize,
        end: usize,
        build: &RecordBatch,
        matched: &mut [bool],
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
        let paired =
            RecordBatch::try_new_with_options(Arc::clone(&self.0.pairs), columns, &options)
                .map_err(|e| internal(e.to_string()))?;
        let mask = match &self.0.spec.filter {
            None => BooleanArray::from(vec![true; n]),
            Some(f) => {
                let m = f.evaluate(&paired)?;
                // NULL is not a match.
                match m.nulls() {
                    Some(v) => BooleanArray::new(m.values() & v.inner(), None),
                    None => m,
                }
            }
        };
        for (i, hit) in mask.values().iter().enumerate() {
            if hit {
                matched[i / b] = true;
            }
        }
        if !matches!(self.0.spec.join_type, JoinType::Inner | JoinType::LeftOuter) {
            return Ok(None);
        }
        if mask.true_count() == 0 {
            return Ok(None);
        }
        let out = arrow_select::filter::filter_record_batch(&paired, &mask)
            .map_err(|e| internal(e.to_string()))?;
        out.with_schema(Arc::clone(&self.0.schema))
            .map(Some)
            .map_err(|e| internal(e.to_string()))
    }

    /// Once a chunk met every build row: semi and anti rows, and the
    /// null-padded probe rows of a left outer join.
    fn unmatched(
        &self,
        probe: &RecordBatch,
        start: usize,
        matched: &[bool],
    ) -> Result<Option<RecordBatch>> {
        let keep: Vec<u32> = matched
            .iter()
            .enumerate()
            .filter(|(_, m)| match self.0.spec.join_type {
                JoinType::LeftSemi => **m,
                JoinType::LeftAnti | JoinType::LeftOuter => !**m,
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
        let mut columns = probe
            .columns()
            .iter()
            .map(|c| {
                arrow_select::take::take(c.as_ref(), &idx, None)
                    .map_err(|e| internal(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        if self.0.spec.join_type == JoinType::LeftOuter {
            for f in self.0.spec.build_schema.fields() {
                columns.push(new_null_array(f.data_type(), idx.len()));
            }
        }
        let options = RecordBatchOptions::new().with_row_count(Some(idx.len()));
        RecordBatch::try_new_with_options(Arc::clone(&self.0.schema), columns, &options)
            .map(Some)
            .map_err(|e| internal(e.to_string()))
    }
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
