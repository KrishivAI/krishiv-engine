// SPDX-License-Identifier: Apache-2.0
//! Plan nodes for aggregation, and the `DISTINCT` / `UNION` plans built
//! from the same operator.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_memory::SpillScope;

use super::{
    AggInput, AggregateExpr, AggregateOutput, AggregateSpec, HashAggregate, PartialAggregate,
    unsupported,
};
use crate::error::{Result, internal};
use crate::operator::{Sink, Source};
use crate::pipeline::{ExecNode, PipelineBuilder};

/// A [`HashAggregate`] over `input`: the sink of a child pipeline and the
/// source of the current one.
#[derive(Debug)]
pub struct AggregateNode {
    input: Arc<dyn ExecNode>,
    agg: Arc<HashAggregate>,
}

impl AggregateNode {
    /// `agg` over `input`.
    pub fn new(input: Arc<dyn ExecNode>, agg: Arc<HashAggregate>) -> Arc<Self> {
        Arc::new(Self { input, agg })
    }
}

impl ExecNode for AggregateNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = Arc::clone(&self.agg) as Arc<dyn Sink>;
        let id = b.child_pipeline(sink, Arc::clone(&self.input))?;
        b.set_breaker_source(Arc::clone(&self.agg) as Arc<dyn Source>, id)
            .map(drop)
    }
}

/// A [`PartialAggregate`] streaming over `input`.
#[derive(Debug)]
pub struct PartialNode {
    input: Arc<dyn ExecNode>,
    op: Arc<PartialAggregate>,
}

impl PartialNode {
    /// `op` over `input`.
    pub fn new(input: Arc<dyn ExecNode>, op: Arc<PartialAggregate>) -> Arc<Self> {
        Arc::new(Self { input, op })
    }
}

impl ExecNode for PartialNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        Arc::clone(&self.input).build_pipelines(b)?;
        b.push_operator(Arc::clone(&self.op) as _);
        Ok(())
    }
}

impl std::fmt::Debug for dyn ExecNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ExecNode")
    }
}

/// An aggregation in which the aggregates flagged in `distinct` are
/// `DISTINCT` (Spark's single-distinct plan): first group by the keys and
/// the distinct arguments, computing the other aggregates' partial states,
/// then group by the keys, aggregating each distinct value once and merging
/// the other states. `spec` has raw inputs and final output.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` when distinct aggregates have different
/// arguments (Spark's `Expand` plan); typing errors.
pub fn distinct_plan(
    input: Arc<dyn ExecNode>,
    spec: AggregateSpec,
    distinct: &[bool],
) -> Result<Arc<dyn ExecNode>> {
    if distinct.len() != spec.aggregates.len()
        || spec.output != AggregateOutput::Final
        || spec.aggregates.iter().any(|a| a.input != AggInput::Raw)
    {
        return Err(internal(
            "a DISTINCT plan takes raw input and final output".to_owned(),
        ));
    }
    let mut dargs: Option<&[usize]> = None;
    for (a, &d) in spec.aggregates.iter().zip(distinct) {
        if d && *dargs.get_or_insert(&a.args) != a.args.as_slice() {
            return Err(unsupported(
                "DISTINCT aggregates over different arguments".to_owned(),
            ));
        }
    }
    let Some(dargs) = dargs.map(<[usize]>::to_vec) else {
        return Ok(AggregateNode::new(input, HashAggregate::new(spec)?));
    };
    let k = spec.group_by.len();
    let first = AggregateSpec {
        group_by: spec.group_by.iter().chain(&dargs).copied().collect(),
        aggregates: spec
            .aggregates
            .iter()
            .zip(distinct)
            .filter(|(_, d)| !**d)
            .map(|(a, _)| a.clone())
            .collect(),
        output: AggregateOutput::Partial,
        require_unit_weights: true,
        ..spec.clone()
    };
    let first_schema = first.output_schema()?;
    let mut state_at = k + dargs.len();
    let mut aggregates = Vec::with_capacity(spec.aggregates.len());
    for (a, &d) in spec.aggregates.iter().zip(distinct) {
        if d {
            aggregates.push(AggregateExpr {
                args: (k..k + dargs.len()).collect(),
                ..a.clone()
            });
        } else {
            let m = a.merging(state_at);
            state_at += m.args.len();
            aggregates.push(m);
        }
    }
    let second = AggregateSpec {
        input: first_schema,
        group_by: (0..k).collect(),
        aggregates,
        ..spec
    };
    let inner = AggregateNode::new(input, HashAggregate::new(first)?);
    Ok(AggregateNode::new(inner, HashAggregate::new(second)?))
}

/// `UNION` (distinct) of the rows of `input` (the `UNION ALL` of the
/// branches): a grouping on every column with no aggregates.
///
/// # Errors
/// `INTERNAL_ERROR` for an unsupported key type.
pub fn union_distinct(
    input: Arc<dyn ExecNode>,
    schema: SchemaRef,
    spill: Option<SpillScope>,
) -> Result<Arc<dyn ExecNode>> {
    let spec = AggregateSpec {
        group_by: (0..schema.fields().len()).collect(),
        input: schema,
        aggregates: Vec::new(),
        output: AggregateOutput::Final,
        ansi: true,
        spill,
        require_unit_weights: true,
        legacy_statistical: false,
    };
    Ok(AggregateNode::new(input, HashAggregate::new(spec)?))
}
