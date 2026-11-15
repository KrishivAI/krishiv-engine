// SPDX-License-Identifier: Apache-2.0
//! Group keys determined by other keys (`GROUP BY x, x - 1`) leave the keys.
//!
//! ClickBench's 36th query groups by `ClientIP, ClientIP - 1, ClientIP - 2,
//! ClientIP - 3`: four keys hashed, compared, stored and exchanged where
//! one decides the group. A key that is a deterministic function of key
//! columns of the same aggregate splits no group those columns form, so the
//! aggregate groups by the columns alone and computes the dependent key as
//! one of its outputs, over the key columns, once per group.
//!
//! # When it applies
//!
//! A key of a `GROUP BY` aggregate (not `ROLLUP`, `CUBE` or `GROUPING
//! SETS`) is dropped when all of the following hold, else it stays:
//!
//! 1. It is not a bare column, and it reads only columns that are bare
//!    column keys of the same aggregate (a literal key reads none). At
//!    least one such key stays, so the aggregate is never made global
//!    (a global aggregate returns a row over no input; a grouped one does
//!    not).
//! 2. It is deterministic, holds no subquery, outer reference or aggregate
//!    call, and calls only built-in and registry functions (no UDF, whose
//!    determinism the session declares).
//! 3. **Equal keys are identical values.** Each column it reads has a type
//!    whose grouping equality is identity: integral, boolean, binary,
//!    decimal, date, timestamp and `STRING` with the binary collation.
//!    Floating point is excluded (`0.0` and `-0.0`, and every NaN, form
//!    one group, which `1 / x` or a cast to string would split), as are
//!    collated strings (`'a'` and `'A'` form one group under a
//!    case-insensitive collation; `upper` or `length` of them need not).
//!
//! # Errors and NULL
//!
//! The dependent key is computed by the aggregate itself (its output
//! expressions, evaluated over the key columns of each group), not by a
//! projection above it, so no filter or limit moves between the aggregate
//! and the computation. Each group's key values are those of some input
//! row, so the computation raises (ANSI overflow, a raising function) for
//! some group exactly when the per-row computation raised for some row:
//! the query fails or succeeds as before, with the same error class. When
//! another computation of the query also raises, which of the two errors is
//! reported may differ, as it may between two runs. A NULL key column is one
//! group, and the expression of NULL is one value.

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    Expr, ExprNode, FunctionId, FunctionKind, Grouping, LogicalPlan, PlanNode, ResolvedExpr,
};
use krishiv_types::DataType;

use crate::dag;
use crate::expr::{Attrs, has_outer_reference, has_subquery, is_deterministic, references};

/// Drops the dependent keys of every aggregate of `root`.
///
/// # Errors
/// As [`dag::rewrite_up`].
pub(crate) fn rewrite(root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
    dag::rewrite_up(root, &mut |p| Ok(rewrite_aggregate(p)))
}

fn rewrite_aggregate(p: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(keys),
        aggregates,
    } = &p.node
    else {
        return p;
    };
    // The bare column keys whose values decide their group exactly.
    let mut determinants = Attrs::default();
    for k in keys {
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &k.node
            && identity_equality(&c.data_type)
        {
            determinants.insert(c.id);
        }
    }
    if determinants.is_empty() {
        return p;
    }
    let kept: Vec<Arc<Expr>> = keys
        .iter()
        .filter(|k| !dependent(k, &determinants))
        .cloned()
        .collect();
    if kept.len() == keys.len() {
        return p;
    }
    let mut out = LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(input),
        grouping: Grouping::GroupBy(kept),
        aggregates: aggregates.clone(),
    });
    out.origin = p.origin;
    Arc::new(out)
}

/// Whether key `k` is determined by the columns in `determinants`
/// (conditions 1 and 2 of the module documentation).
fn dependent(k: &Arc<Expr>, determinants: &Attrs) -> bool {
    if matches!(k.node, ExprNode::Resolved(ResolvedExpr::Column(_))) {
        return false;
    }
    is_deterministic(k)
        && !has_subquery(k)
        && !has_outer_reference(k)
        && only_builtin_scalars(k)
        && references(k).iter().all(|id| determinants.contains(id))
}

/// Whether every call in `e` is a scalar built-in or registry function.
fn only_builtin_scalars(e: &Arc<Expr>) -> bool {
    let mut stack: Vec<&Arc<Expr>> = vec![e];
    while let Some(x) = stack.pop() {
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && (f.kind == FunctionKind::Aggregate
                || !matches!(f.id, FunctionId::Builtin(_) | FunctionId::Registry { .. }))
        {
            return false;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    true
}

/// Whether two values of `t` that group together are the same value
/// (condition 3 of the module documentation).
fn identity_equality(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Boolean
            | DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::String
            | DataType::Binary
            | DataType::Decimal(_)
            | DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
    )
}
