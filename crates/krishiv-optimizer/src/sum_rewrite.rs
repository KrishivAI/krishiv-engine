// SPDX-License-Identifier: Apache-2.0
//! `SUM(x + c)` as `SUM(x) + c * COUNT(x)` (and `x - c`, `c - x`).
//!
//! Many sums of one column shifted by different constants (ClickBench's
//! 30th query computes ninety) become one `SUM` and one `COUNT` of the
//! column, and a projection above the aggregate that adds the constants.
//! DuckDB has a rewrite of this shape (`SumRewriteRule` in
//! `src/optimizer/aggregate_function_rewriter.cpp`, formerly its sum
//! rewriter); it matches integer `x + C` without grouping and widens the
//! result. This one is stricter, because Spark evaluates `x + c` per row
//! and an overflow there is an error (ANSI) or a wrap (legacy) that
//! `SUM(x) + c * COUNT(x)` would not reproduce.
//!
//! # When it applies
//!
//! All of the following, else the aggregate is left alone:
//!
//! 1. The item is exactly `sum(e) AS a` (not `DISTINCT`, no `FILTER`), `e`
//!    is a call of `+` or `-` of an integral column expression `x` and an
//!    integral literal `c`, and the call has the type of `x` and `c`.
//!    Floating point is excluded: IEEE addition is not associative, so
//!    `SUM(x + c)` and `SUM(x) + c * COUNT(x)` differ in the last bits,
//!    and the result must be identical. Decimals are excluded: their
//!    result precision and scale follow the operands and the rewrite would
//!    have to reproduce each rounding.
//! 2. **No per-row overflow.** The type of `x` bounds it
//!    ([`bounds`]: a column's type, through casts that cannot fail) and
//!    every value `x + c` can take fits the type of the addition. For
//!    `SMALLINT` and `TINYINT` columns promoted to `INT` or `BIGINT` (what
//!    Spark's coercion makes of `smallint_col + 1`) this always holds. For
//!    an `INT` column and `+ 1` it does not (`INT_MAX + 1` overflows), so
//!    that is left alone.
//! 3. **No overflow in the accumulation either.** `SUM` of an integral type
//!    accumulates in `BIGINT` (an error under ANSI, a wrap in legacy
//!    mode); the rewrite accumulates `SUM(x)` and `c * COUNT(x)` there.
//!    Legacy mode wraps modulo 2^64, which is associative, so both agree
//!    for any input. Under ANSI, an error is raised when a partial sum
//!    leaves the `BIGINT` range. With `|x| <= 2^16` and `|c| <= 2^16`
//!    ([`MAX_MAGNITUDE`]) every row adds less than 2^17 in magnitude, so
//!    neither form can leave the range before 2^46 rows; the two can differ
//!    only on aggregates over more than 2^46 (7e13) rows.
//! 4. NULL: `SUM` skips NULL values of `x + c`, which are those of `x`;
//!    `COUNT(x)` counts the same rows, so the result is NULL exactly when
//!    `SUM(x)` is, and `NULL + c * 0` is NULL.
//!
//! The result type is `BIGINT`, as the sum's was, and the output attribute
//! (id, name, nullability) is the aggregate item's own, now computed by a
//! projection over the aggregate.

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_functions::registry::{Lookup, builtins};
use krishiv_functions::signature::SparkProfile;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, LogicalPlan,
    PlanNode, ResolvedExpr, ResolvedFunction,
};
use krishiv_types::DataType;

use crate::dag;
use crate::expr::{
    call, column, data_type, has_aggregate, has_outer_reference, has_subquery, is_deterministic,
    output_of, same,
};
use crate::restrict::{MAX_ID, max_id};

/// The largest magnitude of the column and of the constant (2^16): see the
/// module documentation, condition 3.
pub(crate) const MAX_MAGNITUDE: i128 = 1 << 16;

/// Rewrites the sums of shifted columns in every aggregate of `root`.
///
/// # Errors
/// As [`dag::rewrite_up`].
pub(crate) fn rewrite(root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut next = max_id(&root, &mut std::collections::HashMap::new()).saturating_add(1);
    dag::rewrite_up(root, &mut |p| Ok(rewrite_aggregate(p, &mut next)))
}

/// One `sum(x ± c)` item, taken apart.
struct Shifted<'a> {
    /// The column expression.
    x: &'a Arc<Expr>,
    /// The constant, with the sign it has in the sum: `+c` or `-c`, and
    /// whether the sum is `c - x` (the column is subtracted).
    c: i128,
    negated: bool,
    /// The `sum` call.
    sum: &'a ResolvedFunction,
}

fn rewrite_aggregate(p: Arc<LogicalPlan>, next: &mut u64) -> Arc<LogicalPlan> {
    let PlanNode::Aggregate {
        input,
        grouping,
        aggregates,
    } = &p.node
    else {
        return p;
    };
    let shifted: Vec<Option<Shifted<'_>>> = aggregates.iter().map(shifted).collect();
    if shifted.iter().all(Option::is_none) {
        return p;
    }
    // Every item must be a column or an alias to be passed up unchanged.
    let outputs: Option<Vec<&BoundColumn>> = aggregates.iter().map(|e| output_of(e)).collect();
    let Some(outputs) = outputs else {
        return p;
    };
    let mut kept: Vec<Arc<Expr>> = Vec::new();
    let mut project: Vec<Arc<Expr>> = Vec::new();
    // The `sum(x)` and `count(x)` attributes of each distinct `x`.
    let mut made: Vec<(Arc<Expr>, BoundColumn, BoundColumn)> = Vec::new();
    let mut fresh = |name: &str, data_type: DataType, nullable: bool| -> Option<BoundColumn> {
        let id = *next;
        if id > MAX_ID {
            return None;
        }
        *next += 1;
        Some(BoundColumn {
            id: AttrId(id),
            name: name.to_owned(),
            data_type,
            nullable,
        })
    };
    let count_id = count_function();
    let mut extra: Vec<Arc<Expr>> = Vec::new();
    for ((item, shift), out) in aggregates.iter().zip(&shifted).zip(&outputs) {
        let Some(s) = shift else {
            kept.push(Arc::clone(item));
            project.push(column(out));
            continue;
        };
        let Some(count_id) = count_id.clone() else {
            return p;
        };
        let (sum_col, count_col) = match made.iter().find(|(x, _, _)| same(x, s.x)) {
            Some((_, a, b)) => (a.clone(), b.clone()),
            None => {
                // An existing `sum(x)` item of the aggregate is reused.
                let plain = plain_sum(aggregates, s.x);
                let (Some(sum_col), Some(count_col)) = (
                    match plain {
                        Some(c) => Some(c.clone()),
                        None => fresh("sum", DataType::Long, true),
                    },
                    fresh("count", DataType::Long, false),
                ) else {
                    return p;
                };
                if plain.is_none() {
                    extra.push(alias(
                        sum_call(s.sum, vec![Arc::clone(s.x)]),
                        sum_col.clone(),
                    ));
                }
                extra.push(alias(
                    count_call(count_id, vec![Arc::clone(s.x)]),
                    count_col.clone(),
                ));
                made.push((Arc::clone(s.x), sum_col.clone(), count_col.clone()));
                (sum_col, count_col)
            }
        };
        // `c * COUNT(x)`, then added to or subtracted from `SUM(x)`.
        let Ok(c) = i64::try_from(s.c.abs()) else {
            return p;
        };
        let scaled = call(
            Builtin::Multiply,
            vec![
                Arc::new(Expr::literal(Literal::Long(c))),
                column(&count_col),
            ],
            DataType::Long,
        );
        let total = match (s.c < 0, s.negated) {
            // c - SUM(x), with c >= 0 or c < 0 (then -(|c| COUNT) - SUM).
            (false, true) => call(
                Builtin::Subtract,
                vec![scaled, column(&sum_col)],
                DataType::Long,
            ),
            (true, true) => call(
                Builtin::Subtract,
                vec![
                    call(Builtin::Negative, vec![scaled], DataType::Long),
                    column(&sum_col),
                ],
                DataType::Long,
            ),
            (false, false) => call(Builtin::Add, vec![column(&sum_col), scaled], DataType::Long),
            (true, false) => call(
                Builtin::Subtract,
                vec![column(&sum_col), scaled],
                DataType::Long,
            ),
        };
        project.push(alias(total, (*out).clone()));
    }
    // A `sum(x)` item that was reused stays in `kept` (it was not a shifted
    // sum); the new aggregates follow it.
    kept.extend(extra);
    let aggregate = Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(input),
        grouping: grouping.clone(),
        aggregates: kept,
    }));
    let mut out = LogicalPlan::new(PlanNode::Project {
        input: aggregate,
        exprs: project,
    });
    out.origin = p.origin;
    Arc::new(out)
}

/// `e` as `sum(x ± c)`, when the module's conditions hold.
fn shifted(e: &Arc<Expr>) -> Option<Shifted<'_>> {
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) = &e.node else {
        return None;
    };
    let ExprNode::Resolved(ResolvedExpr::Function(sum)) = &expr.node else {
        return None;
    };
    if sum.id.name() != "sum"
        || sum.kind != FunctionKind::Aggregate
        || sum.distinct
        || sum.filter.is_some()
        || sum.data_type != DataType::Long
    {
        return None;
    }
    let [arg] = sum.args.as_slice() else {
        return None;
    };
    let ExprNode::Resolved(ResolvedExpr::Function(add)) = &arg.node else {
        return None;
    };
    let FunctionId::Builtin(op @ (Builtin::Add | Builtin::Subtract)) = add.id else {
        return None;
    };
    let [l, r] = add.args.as_slice() else {
        return None;
    };
    // Which operand is the constant.
    let (x, c, constant_first) = match (literal_value(l), literal_value(r)) {
        (None, Some(c)) => (l, c, false),
        (Some(c), None) => (r, c, true),
        _ => return None,
    };
    let t = add.data_type.clone();
    if !integral(&t)
        || data_type(x).as_ref() != Some(&t)
        || data_type(l).as_ref() != Some(&t)
        || data_type(r).as_ref() != Some(&t)
        || !is_deterministic(x)
        || has_subquery(x)
        || has_outer_reference(x)
        || has_aggregate(x)
        || c.abs() > MAX_MAGNITUDE
    {
        return None;
    }
    // The signed constant of `x + k`: `+c`, `-c` for `x - c`; `c - x` is
    // `c` with the column negated.
    let (k, negated) = match (op, constant_first) {
        (Builtin::Add, _) => (c, false),
        (_, false) => (-c, false),
        (_, true) => (c, true),
    };
    let (lo, hi) = bounds(x)?;
    if lo < -MAX_MAGNITUDE || hi > MAX_MAGNITUDE {
        return None;
    }
    // The values of the addition, per row, fit its type.
    let (v_lo, v_hi) = if negated {
        (k - hi, k - lo)
    } else {
        (lo + k, hi + k)
    };
    let (t_lo, t_hi) = type_range(&t)?;
    if v_lo < t_lo || v_hi > t_hi {
        return None;
    }
    Some(Shifted {
        x,
        c: k,
        negated,
        sum,
    })
}

fn integral(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long
    )
}

/// The range of an integral type.
fn type_range(t: &DataType) -> Option<(i128, i128)> {
    Some(match t {
        DataType::Byte => (i128::from(i8::MIN), i128::from(i8::MAX)),
        DataType::Short => (i128::from(i16::MIN), i128::from(i16::MAX)),
        DataType::Integer => (i128::from(i32::MIN), i128::from(i32::MAX)),
        DataType::Long => (i128::from(i64::MIN), i128::from(i64::MAX)),
        _ => return None,
    })
}

fn literal_value(e: &Expr) -> Option<i128> {
    match &e.node {
        ExprNode::Literal(Literal::Byte(v)) => Some(i128::from(*v)),
        ExprNode::Literal(Literal::Short(v)) => Some(i128::from(*v)),
        ExprNode::Literal(Literal::Integer(v)) => Some(i128::from(*v)),
        ExprNode::Literal(Literal::Long(v)) => Some(i128::from(*v)),
        _ => None,
    }
}

/// The values an integral expression can take, from the types of its
/// columns: a column's type range, through casts to a type that holds
/// that range (so the cast neither fails nor wraps).
fn bounds(e: &Expr) -> Option<(i128, i128)> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => type_range(&c.data_type),
        ExprNode::Cast { expr, to, .. } => {
            let (lo, hi) = bounds(expr)?;
            let (t_lo, t_hi) = type_range(to)?;
            (t_lo <= lo && hi <= t_hi).then_some((lo, hi))
        }
        _ => None,
    }
}

/// The output attribute of an item `sum(x) AS a` already in `aggregates`.
fn plain_sum<'a>(aggregates: &'a [Arc<Expr>], x: &Arc<Expr>) -> Option<&'a BoundColumn> {
    aggregates.iter().find_map(|e| {
        let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &e.node else {
            return None;
        };
        let ExprNode::Resolved(ResolvedExpr::Function(f)) = &expr.node else {
            return None;
        };
        let plain = f.id.name() == "sum"
            && f.kind == FunctionKind::Aggregate
            && !f.distinct
            && f.filter.is_none()
            && f.data_type == DataType::Long
            && matches!(f.args.as_slice(), [a] if same(a, x));
        plain.then_some(output)
    })
}

fn alias(expr: Arc<Expr>, output: BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr,
        output,
    })))
}

fn sum_call(like: &ResolvedFunction, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            args,
            ..like.clone()
        },
    ))))
}

fn count_call(id: FunctionId, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id,
            kind: FunctionKind::Aggregate,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Long,
            nullable: false,
            deterministic: true,
        },
    ))))
}

/// The registry's `count`.
fn count_function() -> Option<FunctionId> {
    let profile = SparkProfile::ALL.into_iter().max()?;
    match builtins().lookup("count", profile) {
        Lookup::Found(def) => Some(FunctionId::Registry {
            name: def.name.to_owned(),
            id: def.id,
        }),
        _ => None,
    }
}
