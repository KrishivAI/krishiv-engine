// SPDX-License-Identifier: Apache-2.0
//! Group keys determined by other keys (`GROUP BY x, x - 1`).
//!
//! ClickBench's 36th query groups by `ClientIP, ClientIP - 1, ClientIP - 2,
//! ClientIP - 3`: four keys hashed, compared, stored and exchanged where
//! one decides the group. A key that is a deterministic function of key
//! columns of the same aggregate splits no group those columns form, so the
//! physical planner groups by the columns alone and computes the dependent
//! key from the key columns of each group (`krishiv-exec`
//! `planner/aggregate.rs`). This module only decides which keys are
//! dependent; the logical plan is not changed.
//!
//! # Which keys are dependent
//!
//! A key of a `GROUP BY` aggregate (not `ROLLUP`, `CUBE` or `GROUPING
//! SETS`) is dependent when all of the following hold:
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
//! # Errors
//!
//! Spark evaluates a grouping key on every input row, so a key that can
//! raise (ANSI overflow in `ClientIP - 1`) raises for any row, whether or
//! not the key is read above the aggregate and however few groups a
//! `LIMIT` takes. The physical planner therefore still computes a dependent
//! key that may raise on every input row, below the aggregate (a breaker,
//! which reads all its input before its first output row), and only leaves
//! it out of the hashed keys; a key that cannot raise is computed per group
//! only.

use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, FunctionId, FunctionKind, ResolvedExpr};
use krishiv_types::DataType;

use crate::expr::{Attrs, has_outer_reference, has_subquery, is_deterministic, references};

/// For each of `keys` (an aggregate's `GROUP BY` keys), whether it is
/// determined by the bare column keys among them (see the module
/// documentation). All `false` when no bare column key could decide a
/// group.
#[must_use]
pub fn dependent_group_keys(keys: &[Arc<Expr>]) -> Vec<bool> {
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
        return vec![false; keys.len()];
    }
    keys.iter().map(|k| dependent(k, &determinants)).collect()
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
