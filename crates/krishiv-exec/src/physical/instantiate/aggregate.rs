// SPDX-License-Identifier: Apache-2.0
//! Aggregate specs and nodes, and the output filter a `Filter` above a
//! grouped aggregate hands it (`krishiv.exec.agg.outputFilter`,
//! `aggregate/output_filter.rs`): the `Filter`'s predicate, when it is
//! deterministic, reads no subquery and only the aggregate's values through
//! plain column projections, is also compiled over those values and set on
//! the aggregate. The `Filter` stays in the plan.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use krishiv_memory::SpillScope;
use krishiv_plan::{AttrId, Expr, ExprNode, ResolvedExpr};

use super::{Instantiation, Values, compile_one};
use crate::aggregate::{AggregateNode, AggregateOutput, AggregateSpec, HashAggregate};
use crate::aggregate::{OutputFilter, UdafEnv};
use crate::error::Result;
use crate::operators::Predicate;
use crate::physical::{AggregatePlan, PhysNode, ProjectItemPlan, SubqueryPlan};
use crate::pipeline::ExecNode;
use crate::planner::eval::{Compiled, EvalEnv};

pub(crate) fn aggregate_spec(
    input: &SchemaRef,
    a: &AggregatePlan,
    env: &EvalEnv,
    spill: Option<SpillScope>,
) -> AggregateSpec {
    AggregateSpec {
        input: Arc::clone(input),
        group_by: a.group_by.clone(),
        aggregates: a.aggregates.clone(),
        output: a.output,
        ansi: env.ansi,
        spill,
        require_unit_weights: false,
        legacy_statistical: env.legacy_statistical_aggregate,
        udafs: UdafEnv::of(env.udfs.as_ref(), env.zone),
    }
}

/// The aggregate of `spec` over `input` (two for `DISTINCT` aggregates, or
/// one counting at the merge with `count_at_merge`: see
/// [`crate::aggregate::distinct_plan_with`]).
pub(crate) fn aggregate_node(
    input: Arc<dyn ExecNode>,
    spec: AggregateSpec,
    distinct: &[bool],
    count_at_merge: bool,
) -> Result<Arc<dyn ExecNode>> {
    if distinct.iter().any(|d| *d) {
        crate::aggregate::distinct_plan_with(input, spec, distinct, count_at_merge)
    } else {
        Ok(AggregateNode::new(input, HashAggregate::new(spec)?))
    }
}

/// A `Filter`'s predicate as an aggregate's output filter, before it is
/// compiled: over the attributes `layout`, which are the aggregate output
/// columns `columns`.
#[derive(Debug, Clone)]
pub(super) struct PendingFilter {
    predicate: Arc<Expr>,
    layout: Vec<AttrId>,
    columns: Vec<usize>,
}

/// The aggregate (by node address) that a `Filter` over `input` with
/// `predicate` over `layout` may hand its predicate, and what it hands: a
/// final grouped aggregate without `DISTINCT` aggregates below plain column
/// projections only, the predicate deterministic, unable to raise under
/// `ansi` (the disk phase evaluates it on every group, which a `LIMIT` above
/// may not read: review NIT-1) and reading value columns only.
pub(super) fn having_target(
    input: &PhysNode,
    layout: &[AttrId],
    predicate: &Arc<Expr>,
    ansi: bool,
) -> Option<(usize, PendingFilter)> {
    // Each Filter input column's position in the node reached so far.
    let mut at: Vec<Option<usize>> = (0..layout.len()).map(Some).collect();
    let mut node = input;
    let aggregate = loop {
        match node {
            PhysNode::Project {
                input,
                items,
                subqueries,
                ..
            } if subqueries.is_empty() => {
                for a in &mut at {
                    *a = a.and_then(|i| match items.get(i) {
                        Some(ProjectItemPlan::Column { index, .. }) => Some(*index),
                        _ => None,
                    });
                }
                node = input.as_ref();
            }
            PhysNode::Aggregate { aggregate, .. } => break aggregate,
            _ => return None,
        }
    };
    if aggregate.output != AggregateOutput::Final
        || aggregate.group_by.is_empty()
        || aggregate.distinct.iter().any(|d| *d)
        || !krishiv_plan::is_deterministic(predicate)
        || krishiv_plan::may_raise(predicate, ansi)
    {
        return None;
    }
    let keys = aggregate.group_by.len();
    let mut used = BTreeMap::new();
    if !columns_of(predicate, &mut |id| {
        let column = layout
            .iter()
            .position(|a| *a == id)
            .and_then(|i| at.get(i).copied().flatten())
            .filter(|&c| c >= keys);
        column.map(|c| used.insert(id, c)).is_some()
    }) {
        return None;
    }
    if used.is_empty() {
        return None;
    }
    let (layout, columns) = used.into_iter().unzip();
    let addr = std::ptr::from_ref(node).addr();
    Some((
        addr,
        PendingFilter {
            predicate: Arc::clone(predicate),
            layout,
            columns,
        },
    ))
}

/// Calls `f` on every column `e` reads; false as soon as `f` refuses one.
fn columns_of(e: &Expr, f: &mut impl FnMut(AttrId) -> bool) -> bool {
    if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node
        && !f(c.id)
    {
        return false;
    }
    let mut ok = true;
    e.for_each_child(&mut |child| ok = ok && columns_of(child, f));
    ok
}

impl crate::physical::WorkerContext {
    /// `krishiv.exec.agg.outputFilter` from this worker's configuration.
    pub(crate) fn agg_output_filter(&self) -> bool {
        use krishiv_common::config::{ConfigValue, EXEC_AGG_OUTPUT_FILTER as E};
        self.config
            .as_deref()
            .map_or(E.default.as_ref(), |c| c.get(&E))
            .and_then(ConfigValue::as_bool)
            .unwrap_or(true)
    }
}

impl Instantiation<'_> {
    /// Records the output filter a `Filter` over `input` hands the
    /// aggregate below it, when it may (see [`having_target`]).
    pub(super) fn note_output_filter(
        &mut self,
        input: &PhysNode,
        layout: &[AttrId],
        predicate: &Arc<Expr>,
        subqueries: &[SubqueryPlan],
    ) {
        if !subqueries.is_empty() || !self.worker.agg_output_filter() {
            return;
        }
        if let Some((addr, pending)) = having_target(input, layout, predicate, self.env.ansi) {
            self.output_filters.insert(addr, pending);
        }
    }

    /// `agg`, the aggregate of node `n` with output `schema`, with the
    /// output filter a `Filter` above recorded for it, if any.
    pub(super) fn with_output_filter(
        &self,
        n: &PhysNode,
        schema: &SchemaRef,
        agg: Arc<HashAggregate>,
    ) -> Result<Arc<HashAggregate>> {
        let Some(p) = self.output_filters.get(&std::ptr::from_ref(n).addr()) else {
            return Ok(agg);
        };
        let fields: Vec<_> = p
            .columns
            .iter()
            .filter_map(|&c| schema.fields().get(c).cloned())
            .collect();
        let values = Arc::new(Schema::new(fields));
        let v = Values {
            env: &self.env,
            subqueries: &[],
        };
        let predicate = compile_one(&p.predicate, &p.layout, &values, &v)?;
        let filter = OutputFilter::new(
            Arc::new(Compiled(predicate)) as Arc<dyn Predicate>,
            p.columns.clone(),
            values,
        );
        Ok(HashAggregate::with_output_filter(agg, filter))
    }

    /// [`aggregate_node`] for node `n`, with its output filter.
    pub(super) fn aggregate_node_of(
        &self,
        n: &PhysNode,
        input: Arc<dyn ExecNode>,
        spec: AggregateSpec,
        distinct: &[bool],
    ) -> Result<Arc<dyn ExecNode>> {
        if distinct.iter().any(|d| *d)
            || !self
                .output_filters
                .contains_key(&std::ptr::from_ref(n).addr())
        {
            return aggregate_node(input, spec, distinct, self.worker.distinct_count_at_merge());
        }
        let schema = spec.output_schema()?;
        let agg = self.with_output_filter(n, &schema, HashAggregate::new(spec)?)?;
        Ok(AggregateNode::new(input, agg))
    }
}
