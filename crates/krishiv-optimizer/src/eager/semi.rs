// SPDX-License-Identifier: Apache-2.0
//! A distinct over a semi or anti join runs on each input instead (TPC-DS
//! q14's `INTERSECT`, lever K5).
//!
//! `Distinct(L SEMI JOIN R ON c)`, the analyzer's `INTERSECT` (and `EXCEPT`
//! with an anti join), becomes `Distinct(L) SEMI JOIN Distinct(R')`, where
//! `R'` keeps the columns of `R` that `c` reads:
//!
//! - a semi or anti join only filters `L`'s rows by a test of their values,
//!   so with `c` deterministic, reading of `L` only the distinct's keys, the
//!   rows of one group pass or fail together, and the distinct may run first;
//!   its output is already distinct, so no aggregate stays above;
//! - only whether a row of `R` matches counts, so equal rows of `R'` can go.
//!
//! The distinct below the join is then an aggregate over `L`'s own joins,
//! which [`super::Shape::push`] can push further down (the fact table grouped
//! by its item key before the item join, in q14's three channels).
//!
//! Gates: keys and the condition's columns of `R` are of types whose equal
//! values cannot be told apart (no floating point, no collation, nothing
//! nested); each side's distinct must have a known distinct count and reduce
//! its rows [`MIN_REDUCTION`] times, else that side keeps its rows (`L`
//! keeps the distinct above the join). An `L` that is already distinct on
//! the keys (the left of a chained `INTERSECT`) gets no second distinct.

use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, ExprNode, Grouping, JoinCondition, JoinKind, LogicalPlan, PlanNode,
    ResolvedExpr,
};
use krishiv_types::DataType;

use super::{MIN_REDUCTION, attrs_of, contains, groups_of};
use crate::estimate::Estimator;
use crate::expr::{column, has_outer_reference, has_subquery, is_deterministic, references};

/// What [`split`] built.
pub(super) struct Split {
    /// The new plan: the join, or the distinct over it when `L` kept its rows.
    pub(super) plan: Arc<LogicalPlan>,
    /// The `EXPLAIN VERBOSE` note.
    pub(super) note: String,
}

/// `plan` with its distinct split across the semi or anti join below it, or
/// why not (`None`: not the shape at all).
/// `then` gets each new distinct below the join and returns what replaces it
/// (eager aggregation pushing it further down).
pub(super) fn split(
    plan: &LogicalPlan,
    est: &mut Estimator<'_>,
    then: &mut dyn FnMut(Arc<LogicalPlan>, &mut Estimator<'_>) -> Arc<LogicalPlan>,
) -> Option<Result<Split, String>> {
    let keys = distinct_keys(plan)?;
    let PlanNode::Aggregate { input: join, .. } = &plan.node else {
        return None;
    };
    let PlanNode::Join {
        left,
        right,
        kind: kind @ (JoinKind::LeftSemi | JoinKind::LeftAnti),
        condition: JoinCondition::On(on),
    } = &join.node
    else {
        return None;
    };
    Some(build(plan, join, left, right, *kind, on, &keys, est, then))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the matched parts of one plan shape"
)]
fn build(
    plan: &LogicalPlan,
    join: &Arc<LogicalPlan>,
    left: &Arc<LogicalPlan>,
    right: &Arc<LogicalPlan>,
    kind: JoinKind,
    on: &Arc<krishiv_plan::Expr>,
    keys: &[BoundColumn],
    est: &mut Estimator<'_>,
    then: &mut dyn FnMut(Arc<LogicalPlan>, &mut Estimator<'_>) -> Arc<LogicalPlan>,
) -> Result<Split, String> {
    if !is_deterministic(on) || has_subquery(on) || has_outer_reference(on) {
        return Err("the condition is nondeterministic or holds a subquery".to_owned());
    }
    let (left_cols, right_cols) = (attrs_of(left)?, attrs_of(right)?);
    let read = references(on);
    if read
        .iter()
        .any(|id| contains(&left_cols, *id) && !keys.iter().any(|k| k.id == *id))
    {
        return Err("the condition reads a column of the left side the distinct drops".to_owned());
    }
    let right_keys: Vec<BoundColumn> = right_cols
        .iter()
        .filter(|c| read.contains(&c.id))
        .cloned()
        .collect();
    if let Some(k) = keys
        .iter()
        .chain(&right_keys)
        .find(|k| !exact(&k.data_type))
    {
        return Err(format!(
            "{} is a floating point, collated or nested column: equal values can still differ",
            k.name
        ));
    }
    let mut moved = false;
    let mut notes = Vec::new();
    let new_right = match reduces(right, &right_keys, est) {
        Ok(()) if !right_keys.is_empty() && !is_distinct_on(right, &right_keys) => {
            moved = true;
            notes.push("right distinct".to_owned());
            under_shift(right, &right_keys, join, est, then)
        }
        Ok(()) => Arc::clone(right),
        Err(why) => {
            notes.push(format!("right kept: {why}"));
            Arc::clone(right)
        }
    };
    let (new_left, left_distinct) = if is_distinct_on(left, keys) {
        moved = true;
        notes.push("left already distinct".to_owned());
        (Arc::clone(left), true)
    } else {
        match reduces(left, keys, est) {
            Ok(()) => {
                moved = true;
                notes.push("left distinct".to_owned());
                (under_shift(left, keys, plan, est, then), true)
            }
            Err(why) => {
                notes.push(format!("left kept: {why}"));
                (Arc::clone(left), false)
            }
        }
    };
    if !moved {
        return Err(notes.join("; "));
    }
    let joined = Arc::new(
        LogicalPlan::new(PlanNode::Join {
            left: new_left,
            right: new_right,
            kind,
            condition: JoinCondition::On(Arc::clone(on)),
        })
        .with_origin(join.origin),
    );
    let in_order =
        left_cols.len() == keys.len() && left_cols.iter().zip(keys).all(|(c, k)| c.id == k.id);
    let out = if left_distinct && in_order {
        joined
    } else if left_distinct {
        // Distinct rows in another column order.
        Arc::new(
            LogicalPlan::new(PlanNode::Project {
                input: joined,
                exprs: keys.iter().map(column).collect(),
            })
            .with_origin(plan.origin),
        )
    } else {
        // The join's output is `L`'s, wider than the keys, or `L` kept its
        // rows: the distinct stays above.
        let PlanNode::Aggregate {
            grouping,
            aggregates,
            ..
        } = &plan.node
        else {
            return Err("not an aggregate".to_owned());
        };
        Arc::new(
            LogicalPlan::new(PlanNode::Aggregate {
                input: joined,
                grouping: grouping.clone(),
                aggregates: aggregates.clone(),
            })
            .with_origin(plan.origin),
        )
    };
    Ok(Split {
        plan: out,
        note: format!("split across the {kind:?} join: {}", notes.join("; ")),
    })
}

/// The keys of a distinct: an aggregate grouped by columns that outputs
/// exactly those columns, in order, and nothing else.
fn distinct_keys(p: &LogicalPlan) -> Option<Vec<BoundColumn>> {
    let PlanNode::Aggregate {
        grouping: Grouping::GroupBy(keys),
        aggregates,
        ..
    } = &p.node
    else {
        return None;
    };
    if keys.is_empty() || keys.len() != aggregates.len() {
        return None;
    }
    keys.iter()
        .zip(aggregates)
        .map(|(k, o)| match (&k.node, &o.node) {
            (
                ExprNode::Resolved(ResolvedExpr::Column(k)),
                ExprNode::Resolved(ResolvedExpr::Column(o)),
            ) if k.id == o.id => Some(k.clone()),
            _ => None,
        })
        .collect()
}

/// Whether `p`'s rows are already distinct on exactly `keys` (its output):
/// a distinct on them, or a semi or anti join over one.
fn is_distinct_on(p: &LogicalPlan, keys: &[BoundColumn]) -> bool {
    match &p.node {
        PlanNode::Join {
            left,
            kind: JoinKind::LeftSemi | JoinKind::LeftAnti,
            ..
        } => is_distinct_on(left, keys),
        PlanNode::Shift { input, offset } => {
            let inner: Option<Vec<BoundColumn>> = keys
                .iter()
                .map(|k| {
                    let mut c = k.clone();
                    c.id = AttrId(k.id.0.checked_sub(*offset)?);
                    Some(c)
                })
                .collect();
            inner.is_some_and(|inner| is_distinct_on(input, &inner))
        }
        _ => distinct_keys(p).is_some_and(|have| {
            have.len() == keys.len() && keys.iter().all(|k| have.iter().any(|h| h.id == k.id))
        }),
    }
}

/// Whether a distinct on `keys` pays on `p`: known distinct counts, and
/// [`MIN_REDUCTION`] rows per group.
fn reduces(
    p: &Arc<LogicalPlan>,
    keys: &[BoundColumn],
    est: &mut Estimator<'_>,
) -> Result<(), String> {
    let stats = est.plan(p);
    let ids: Vec<AttrId> = keys.iter().map(|k| k.id).collect();
    let bound = source(p, ids, 0).map(|scan| est.plan(&scan).rows + 1.0);
    let groups = match (groups_of(keys, &stats), bound) {
        (Ok(g), b) => b.map_or(g, |b| g.min(b)),
        (Err(_), Some(b)) => b,
        (Err(why), None) => return Err(why),
    };
    if stats.rows < MIN_REDUCTION * groups {
        let counts: Vec<String> = keys
            .iter()
            .map(|k| {
                let ndv = stats.cols.get(&k.id).map_or(f64::NAN, |c| c.ndv);
                format!("{} {ndv:.0}", k.name)
            })
            .collect();
        return Err(format!(
            "reduction below {MIN_REDUCTION}: rows {:.0}, groups {groups:.0} ({})",
            stats.rows,
            counts.join(", ")
        ));
    }
    Ok(())
}

/// The scan every one of `ids` (columns of `p`) is a column of, through
/// projections of columns, filters, joins and `Shift`s: the keys' groups are
/// at most its rows (and a NULL an outer join adds). Distinct counts
/// estimated from integral ranges over sparse ids (TPC-DS `i_brand_id`) run
/// far above that.
fn source(p: &Arc<LogicalPlan>, ids: Vec<AttrId>, depth: usize) -> Option<Arc<LogicalPlan>> {
    if depth > 64 {
        return None;
    }
    let has = |c: &Arc<LogicalPlan>, ids: &[AttrId]| {
        c.resolved_output()
            .is_some_and(|out| ids.iter().all(|id| out.iter().any(|o| o.id == *id)))
    };
    match &p.node {
        PlanNode::Scan(_) => Some(Arc::clone(p)),
        PlanNode::Filter { input, .. } => source(input, ids, depth + 1),
        PlanNode::Project { input, exprs } => {
            let below = ids
                .iter()
                .map(|id| {
                    exprs.iter().find_map(|e| match &e.node {
                        ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == *id => Some(c.id),
                        ExprNode::Resolved(ResolvedExpr::Alias { expr, output })
                            if output.id == *id =>
                        {
                            match &expr.node {
                                ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
                                _ => None,
                            }
                        }
                        _ => None,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            source(input, below, depth + 1)
        }
        PlanNode::Shift { input, offset } => {
            let below = ids
                .iter()
                .map(|id| id.0.checked_sub(*offset).map(AttrId))
                .collect::<Option<Vec<_>>>()?;
            source(input, below, depth + 1)
        }
        PlanNode::Join { left, right, .. } => {
            if has(left, &ids) {
                source(left, ids, depth + 1)
            } else if has(right, &ids) {
                source(right, ids, depth + 1)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A distinct of `input` on `keys`, pushed on by `then`. Over a `Shift` (an
/// instance of a shared subtree, q14's catalog and web channels) it goes
/// inside, on the keys less the offset: a distinct makes no attribute ids, nor
/// does eager aggregation pushing one (no calls, no states), so the instance
/// stays an instance of its own subtree.
fn under_shift(
    input: &Arc<LogicalPlan>,
    keys: &[BoundColumn],
    like: &LogicalPlan,
    est: &mut Estimator<'_>,
    then: &mut dyn FnMut(Arc<LogicalPlan>, &mut Estimator<'_>) -> Arc<LogicalPlan>,
) -> Arc<LogicalPlan> {
    if let PlanNode::Shift {
        input: body,
        offset,
    } = &input.node
    {
        let inner: Option<Vec<BoundColumn>> = keys
            .iter()
            .map(|k| {
                let mut c = k.clone();
                c.id = AttrId(k.id.0.checked_sub(*offset)?);
                Some(c)
            })
            .collect();
        if let Some(inner) = inner {
            let pushed = under_shift(body, &inner, like, est, then);
            return Arc::new(
                LogicalPlan::new(PlanNode::Shift {
                    input: pushed,
                    offset: *offset,
                })
                .with_origin(input.origin),
            );
        }
    }
    then(distinct(input, keys, like), est)
}

/// A distinct of `input` on `keys`, with the origin of `like`.
fn distinct(
    input: &Arc<LogicalPlan>,
    keys: &[BoundColumn],
    like: &LogicalPlan,
) -> Arc<LogicalPlan> {
    let cols: Vec<_> = keys.iter().map(column).collect();
    Arc::new(
        LogicalPlan::new(PlanNode::Aggregate {
            input: Arc::clone(input),
            grouping: Grouping::GroupBy(cols.clone()),
            aggregates: cols,
        })
        .with_origin(like.origin),
    )
}

/// A type whose equal values are the same value: grouping them loses
/// nothing a condition or a reader above could see.
fn exact(t: &DataType) -> bool {
    !matches!(t, DataType::Float | DataType::Double) && !t.contains_collation() && !t.is_nested()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_point_collated_and_nested_types_are_not_exact() {
        assert!(exact(&DataType::Long));
        assert!(exact(&DataType::String));
        assert!(!exact(&DataType::Double));
        assert!(!exact(&DataType::Float));
    }
}
