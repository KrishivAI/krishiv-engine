// SPDX-License-Identifier: Apache-2.0
//! The operator of a `CollectMetrics` node (P3-T084 `Dataset.observe`).

use super::*;

/// The operator of a `CollectMetrics` node over rows of `schema`: the
/// shape of `measure` is what `planner/observe.rs` makes.
pub(crate) fn collect_metrics_op(
    name: &str,
    instance: u32,
    limit: Option<u64>,
    measure: &PhysNode,
    schema: &SchemaRef,
    v: &Values<'_>,
) -> Result<Arc<dyn Operator>> {
    let bad = || {
        internal(format!(
            "the metrics of observation {name} have an unexpected plan"
        ))
    };
    let PhysNode::Project {
        input: aggregate,
        layout: post_layout,
        items: post_items,
        nullable: post_nullable,
        subqueries,
    } = measure
    else {
        return Err(bad());
    };
    let PhysNode::Aggregate {
        input: args,
        aggregate: plan,
    } = aggregate.as_ref()
    else {
        return Err(bad());
    };
    if !subqueries.is_empty() {
        return Err(krishiv_common::error::KrishivUnsupportedFeature {
            feature: "a subquery in an observed metric (Dataset.observe)".to_owned(),
        }
        .into());
    }
    if !plan.group_by.is_empty()
        || plan.output != AggregateOutput::Final
        || plan.distinct.iter().any(|d| *d)
    {
        return Err(bad());
    }
    let (pre, pre_schema) = match args.as_ref() {
        PhysNode::OneRow => (None, Arc::clone(schema)),
        PhysNode::Project {
            input: leaf,
            layout,
            items,
            nullable,
            subqueries,
        } if matches!(leaf.as_ref(), PhysNode::OneRow) && subqueries.is_empty() => {
            let p = project_op(schema, layout, items, nullable.as_deref(), v)?;
            let s = p.schema();
            (Some(p), s)
        }
        // A registered Python UDF in a metric's argument plans a `PythonEval` stage under the
        // projection; the observing operator folds rows in place and does not run stages.
        other if has_python_eval(other) => {
            return Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: "a Python UDF in an observed metric (Dataset.observe)".to_owned(),
            }
            .into());
        }
        _ => return Err(bad()),
    };
    let partial = AggregatePlan {
        output: AggregateOutput::Partial,
        ..plan.clone()
    };
    let merge = final_of(&partial)?;
    CollectMetrics::try_new(
        name,
        instance,
        limit,
        Arc::clone(schema),
        pre,
        aggregate_spec(&pre_schema, &partial, v.env, None),
        |states| Ok(aggregate_spec(states, &merge, v.env, None)),
        |results| {
            project_op(
                results,
                post_layout,
                post_items,
                post_nullable.as_deref(),
                v,
            )
        },
    )
    .map(|op| op as Arc<dyn Operator>)
}

/// Whether `node` or a node below it is a `PythonEval`.
fn has_python_eval(node: &PhysNode) -> bool {
    match node {
        PhysNode::PythonEval { .. } => true,
        PhysNode::Project { input, .. } => has_python_eval(input),
        _ => false,
    }
}
