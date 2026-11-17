// SPDX-License-Identifier: Apache-2.0
//! The phases of an aggregation around an exchange (design P2-T174 §2.2):
//! the final phase of a partial one, Spark's single-distinct phases, and
//! their recognition as one aggregation when one worker runs them
//! ([`fused_aggregate`], [`fused_distinct`]).

use crate::aggregate::{AggInput, AggregateExpr, AggregateOutput};
use crate::error::{Result, internal};
use crate::physical::{AggregatePlan, Partitioning, PhysNode};

/// The final phase reading the output of `partial` (its keys, then each
/// aggregate's state columns): grouped by those keys, each aggregate merging
/// its states. [`fused_aggregate`] recognizes the pair.
pub(crate) fn final_of(partial: &AggregatePlan) -> Result<AggregatePlan> {
    let k = partial.group_by.len();
    let mut at = k;
    let mut aggregates = Vec::with_capacity(partial.aggregates.len());
    for a in &partial.aggregates {
        let m = a.merging(at);
        at += m.args.len();
        aggregates.push(m);
    }
    if partial.output != AggregateOutput::Partial || partial.distinct.iter().any(|d| *d) {
        return Err(internal(
            "a final phase over a plan that is not a partial one".to_owned(),
        ));
    }
    Ok(AggregatePlan {
        group_by: (0..k).collect(),
        distinct: vec![false; aggregates.len()],
        aggregates,
        output: AggregateOutput::Final,
        route: false,
    })
}

/// `Final ← Exchange(Hash | Single) ← Partial` as one aggregation (design
/// P2-T174 §2.2): the partial node, the node below it, and the plan of the
/// single breaker over that node, whose radix partitions stand for the
/// exchange's and whose finalize merge is its reader side. `None` for any
/// other shape: `aggregate` is not exactly the final phase
/// [`final_of`] plans over the partial.
pub(crate) fn fused_aggregate<'p>(
    aggregate: &AggregatePlan,
    input: &'p PhysNode,
) -> Option<(&'p PhysNode, &'p PhysNode, AggregatePlan)> {
    let PhysNode::Exchange(e) = input else {
        return None;
    };
    if !matches!(
        e.partitioning,
        Partitioning::Hash { .. } | Partitioning::Single
    ) {
        return None;
    }
    let partial = e.input.as_ref();
    let PhysNode::Aggregate {
        input: below,
        aggregate: p,
    } = partial
    else {
        return None;
    };
    if p.output != AggregateOutput::Partial || final_of(p).ok().as_ref() != Some(aggregate) {
        return None;
    }
    Some((
        partial,
        below.as_ref(),
        AggregatePlan {
            output: AggregateOutput::Final,
            ..p.clone()
        },
    ))
}

/// The phases of `single`, an aggregation with `DISTINCT` aggregates over raw
/// input, around an exchange (Spark's single-distinct plan, as
/// [`distinct_plan`](crate::aggregate::distinct_plan) runs it in one place): a partial phase grouped by the
/// keys and the distinct arguments computing the other aggregates' states; a
/// partial merge after the exchange on the same columns, which removes the
/// pairs several tasks sent; and the final phase grouped by the keys, which
/// aggregates each distinct value once (as plain aggregates over the now
/// distinct argument columns) and merges the other states. `None` where
/// [`distinct_plan`](crate::aggregate::distinct_plan) has no such plan: no distinct aggregate, distinct
/// aggregates over different arguments or with a filter, or input that is not
/// raw.
pub(crate) fn distinct_phases(single: &AggregatePlan) -> Option<[AggregatePlan; 3]> {
    if single.output != AggregateOutput::Final
        || single.distinct.len() != single.aggregates.len()
        || single.aggregates.iter().any(|a| a.input != AggInput::Raw)
    {
        return None;
    }
    let mut dargs: Option<&[usize]> = None;
    for (a, &d) in single.aggregates.iter().zip(&single.distinct) {
        if d && (a.mask.is_some() || *dargs.get_or_insert(&a.args) != a.args.as_slice()) {
            return None;
        }
    }
    let dargs = dargs?;
    let k = single.group_by.len();
    let width = k + dargs.len();
    let others: Vec<AggregateExpr> = single
        .aggregates
        .iter()
        .zip(&single.distinct)
        .filter(|(_, d)| !**d)
        .map(|(a, _)| a.clone())
        .collect();
    let first = AggregatePlan {
        group_by: single.group_by.iter().chain(dargs).copied().collect(),
        distinct: vec![false; others.len()],
        aggregates: others,
        output: AggregateOutput::Partial,
        route: false,
    };
    let mut at = width;
    let mut merged = Vec::with_capacity(first.aggregates.len());
    for a in &first.aggregates {
        let m = a.merging(at);
        at += m.args.len();
        merged.push(m);
    }
    let merge = AggregatePlan {
        group_by: (0..width).collect(),
        distinct: vec![false; merged.len()],
        aggregates: merged.clone(),
        output: AggregateOutput::Partial,
        route: false,
    };
    let mut merged = merged.into_iter();
    let mut aggregates = Vec::with_capacity(single.aggregates.len());
    for (a, &d) in single.aggregates.iter().zip(&single.distinct) {
        aggregates.push(if d {
            AggregateExpr {
                args: (k..width).collect(),
                ..a.clone()
            }
        } else {
            merged.next()?
        });
    }
    // The final phase is where one worker fuses the phases back into the
    // distinct plan (`fused_distinct`): it carries the routing.
    let last = AggregatePlan {
        group_by: (0..k).collect(),
        distinct: vec![false; aggregates.len()],
        aggregates,
        output: AggregateOutput::Final,
        route: single.route,
    };
    Some([first, merge, last])
}

/// `Final ← PartialMerge ← Exchange(Hash | Single) ← Partial`, the phases
/// [`distinct_phases`] plans, as the one aggregation they came from: the node
/// the partial node, the node below it and that aggregation. `None` for any
/// other shape.
pub(crate) fn fused_distinct<'p>(
    last: &AggregatePlan,
    input: &'p PhysNode,
) -> Option<(&'p PhysNode, &'p PhysNode, AggregatePlan)> {
    let PhysNode::Aggregate {
        input: exchange,
        aggregate: merge,
    } = input
    else {
        return None;
    };
    let PhysNode::Exchange(e) = exchange.as_ref() else {
        return None;
    };
    if !matches!(
        e.partitioning,
        Partitioning::Hash { .. } | Partitioning::Single
    ) {
        return None;
    }
    let partial = e.input.as_ref();
    let PhysNode::Aggregate {
        input: below,
        aggregate: first,
    } = partial
    else {
        return None;
    };
    let k = last.group_by.len();
    let dargs = first.group_by.get(k..)?;
    // The final phase's raw aggregates are the distinct ones; the others
    // merge the partial phase's states, in order.
    let mut others = first.aggregates.iter();
    let mut aggregates = Vec::with_capacity(last.aggregates.len());
    let mut distinct = Vec::with_capacity(last.aggregates.len());
    for a in &last.aggregates {
        if a.input == AggInput::Raw {
            aggregates.push(AggregateExpr {
                args: dargs.to_vec(),
                ..a.clone()
            });
            distinct.push(true);
        } else {
            aggregates.push(others.next()?.clone());
            distinct.push(false);
        }
    }
    let single = AggregatePlan {
        group_by: first.group_by.get(..k)?.to_vec(),
        aggregates,
        distinct,
        output: AggregateOutput::Final,
        route: last.route,
    };
    let phases = distinct_phases(&single)?;
    (phases[0] == *first && phases[1] == *merge && phases[2] == *last).then_some((
        partial,
        below.as_ref(),
        single,
    ))
}
