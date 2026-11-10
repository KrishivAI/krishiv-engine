// SPDX-License-Identifier: Apache-2.0
//! Compute a wide expression below the join that builds it (S4', study 13 §5).
//!
//! TPC-H q09 builds the `orders` join's hash table from `lineitem ⋈ partsupp'`
//! with four decimal columns that only feed `amount = l_extendedprice * (1 -
//! l_discount) - ps_supplycost * l_quantity`, computed above the join. Computing
//! `amount` in a projection over that input first carries one column instead of
//! four, so the build is narrower and spills less.
//!
//! `Project(.. e AS a ..)(S JOIN T)` becomes `Project(.. a' AS a ..)(S' JOIN T)`
//! with `S' = Project(S's columns, e AS a')(S)`; pruning then drops the columns
//! only `e` read.
//!
//! # When it applies
//!
//! All of the following, else the item stays where it is:
//!
//! 1. The join is inner, on a deterministic condition, and the projection sits
//!    directly above it.
//! 2. `e` is deterministic, reads two or more columns, all from one input `S`,
//!    and cannot raise: only `+`, `-` and `*` of decimals whose result type is
//!    the exact one (no rounding, at most 38 digits, so no value can overflow),
//!    casts that never fail and literals. The join may drop rows `e` would have
//!    been evaluated on, so a possible error would change the result.
//! 3. The columns only `e` reads (nothing else in the projection or the join
//!    condition) are strictly wider, by `EXPLAIN COST`'s column sizes, than the
//!    result: a rule never widens a build.
//!
//! # Status: off by default (`Options::compute_below_join`)
//!
//! The rewrite is right on q09, but join ordering then reads the computed
//! input as an opaque leaf whose `Estimator` rows (6e9 at SF1000, against the
//! tree's own 3.3e8, the E9 composite-key defect in the orderer) are larger
//! than `orders`, so the second `optimize` puts `orders` on the build side
//! and the plan is not a fixed point. It needs the orderer to estimate the
//! composite key (or to see through a computed projection) before it is on.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, JoinCondition, JoinKind, LogicalPlan,
    PlanNode, ResolvedExpr, may_raise,
};
use krishiv_types::DataType;

use crate::dag;
use crate::estimate::Estimator;
use crate::expr::{Attrs, column, data_type, is_deterministic, references};
use crate::plan_stats::default_size;
use crate::restrict::{MAX_ID, max_id};
use crate::stats::Statistics;

/// Moves wide expressions below the join that builds them, entering subquery
/// plans. Also returns one line per item moved.
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut memo = HashMap::new();
    let mut next = max_id(&root, &mut memo).saturating_add(1);
    let mut est = Estimator::with_propagation(stats, propagate);
    let mut notes = Vec::new();
    let out = dag::rewrite_up(root, &mut |p| {
        if next > MAX_ID {
            return Ok(p);
        }
        Ok(match moved(&p, &mut est, &mut next, &mut notes) {
            Some(q) => q,
            None => p,
        })
    })?;
    Ok((out, notes))
}

/// `p` with its movable items computed below the join, when it has any.
fn moved(
    p: &Arc<LogicalPlan>,
    _est: &mut Estimator<'_>,
    next: &mut u64,
    notes: &mut Vec<String>,
) -> Option<Arc<LogicalPlan>> {
    let PlanNode::Project { input, exprs } = &p.node else {
        return None;
    };
    let PlanNode::Join {
        left,
        right,
        kind: JoinKind::Inner,
        condition: JoinCondition::On(on),
    } = &input.node
    else {
        return None;
    };
    if !is_deterministic(on) {
        return None;
    }
    let sides = [left, right];
    let outputs = [left.resolved_output()?, right.resolved_output()?];
    // How many times each column is read by the projection and the condition.
    let mut reads: HashMap<AttrId, usize> = HashMap::new();
    let mut count = |e: &Expr| {
        for id in references(e) {
            *reads.entry(id).or_default() += 1;
        }
    };
    count(on);
    for e in exprs {
        count(e);
    }
    let mut below: [Vec<Arc<Expr>>; 2] = [Vec::new(), Vec::new()];
    let mut items = exprs.clone();
    for item in &mut items {
        let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &item.node else {
            continue;
        };
        let used = references(expr);
        if used.len() < 2 || !is_deterministic(expr) || !cannot_raise(expr) {
            continue;
        }
        let Some(side) = (0..2).find(|&s| covers(&outputs[s], &used)) else {
            continue;
        };
        // Columns nothing else reads leave with the item.
        let own = |id: &AttrId| reads.get(id).copied().unwrap_or(0) == 1;
        let freed: u64 = outputs[side]
            .iter()
            .filter(|c| used.contains(&c.id) && own(&c.id))
            .map(|c| default_size(&c.data_type))
            .sum();
        if freed <= default_size(&output.data_type) {
            continue;
        }
        let moved = BoundColumn {
            id: AttrId(*next),
            name: output.name.clone(),
            data_type: output.data_type.clone(),
            nullable: output.nullable,
        };
        *next += 1;
        notes.push(format!(
            "{}#{} computed below the join: {freed} bytes of inputs become {}",
            output.name,
            output.id.0,
            default_size(&output.data_type)
        ));
        below[side].push(Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Alias {
                expr: Arc::clone(expr),
                output: moved.clone(),
            },
        ))));
        *item = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
            expr: column(&moved),
            output: output.clone(),
        })));
    }
    if below.iter().all(Vec::is_empty) {
        return None;
    }
    let widened = |s: usize| -> Arc<LogicalPlan> {
        if below[s].is_empty() {
            return Arc::clone(sides[s]);
        }
        let mut exprs: Vec<Arc<Expr>> = outputs[s].iter().map(column).collect();
        exprs.extend(below[s].iter().cloned());
        Arc::new(
            LogicalPlan::new(PlanNode::Project {
                input: Arc::clone(sides[s]),
                exprs,
            })
            .with_origin(sides[s].origin),
        )
    };
    let join = Arc::new(
        LogicalPlan::new(PlanNode::Join {
            left: widened(0),
            right: widened(1),
            kind: JoinKind::Inner,
            condition: JoinCondition::On(Arc::clone(on)),
        })
        .with_origin(input.origin),
    );
    Some(Arc::new(
        LogicalPlan::new(PlanNode::Project {
            input: join,
            exprs: items,
        })
        .with_origin(p.origin),
    ))
}

/// Whether `cols` has every attribute of `used`.
fn covers(cols: &[BoundColumn], used: &Attrs) -> bool {
    used.iter().all(|id| cols.iter().any(|c| c.id == *id))
}

/// Whether evaluating `e` on any row never raises, under ANSI too.
fn cannot_raise(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(_) | ExprNode::Resolved(ResolvedExpr::Column(_)) => true,
        ExprNode::Cast { .. } => !may_raise(e, true),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let op = matches!(
                f.id,
                FunctionId::Builtin(Builtin::Add | Builtin::Subtract | Builtin::Multiply)
            );
            op && f.filter.is_none()
                && f.args.len() == 2
                && f.args.iter().all(|a| cannot_raise(a))
                && exact_decimal(f, &f.args)
        }
        _ => false,
    }
}

/// Whether the call's decimal result type holds every value of its operands'
/// types: the exact precision and scale, at most 38 digits.
fn exact_decimal(f: &krishiv_plan::ResolvedFunction, args: &[Arc<Expr>]) -> bool {
    let (Some((p1, s1)), Some((p2, s2)), DataType::Decimal(r)) =
        (digits(&args[0]), digits(&args[1]), &f.data_type)
    else {
        return false;
    };
    let (precision, scale) = if f.id == FunctionId::Builtin(Builtin::Multiply) {
        (p1 + p2 + 1, s1 + s2)
    } else {
        let scale = s1.max(s2);
        ((p1 - s1).max(p2 - s2) + scale + 1, scale)
    };
    precision <= 38 && i32::from(r.precision()) == precision && i32::from(r.scale()) == scale
}

/// The precision and scale Spark gives an operand: a decimal's own, an
/// integral type's digits at scale 0.
fn digits(e: &Expr) -> Option<(i32, i32)> {
    match data_type(e)? {
        DataType::Decimal(d) => Some((i32::from(d.precision()), i32::from(d.scale()))),
        DataType::Byte => Some((3, 0)),
        DataType::Short => Some((5, 0)),
        DataType::Integer => Some((10, 0)),
        DataType::Long => Some((20, 0)),
        _ => None,
    }
}
