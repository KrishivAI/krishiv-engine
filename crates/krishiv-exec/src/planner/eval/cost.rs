// SPDX-License-Identifier: Apache-2.0
//! A static cost class of a predicate: whether evaluating it costs much more per row than
//! copying the row out (study 60 (d), `research/perf-parity/60-planner-rewrites-step0.md`).
//!
//! A costly right side of `AND`/`OR` runs over the rows the left side left open once they are
//! few (`AndOr` in [`super::logic`]); a cheap one runs over the whole batch, because a numeric
//! or string comparison costs less than filtering the batch to the open rows and scattering the
//! result back.

use krishiv_plan::{Builtin, Expr, ExprNode, FunctionId, ResolvedExpr};

/// How deep the walk looks before it gives up (and calls the predicate cheap, today's path).
const MAX_DEPTH: usize = 64;

/// Whether `e` holds a pattern match over strings (`LIKE` and kin, `RLIKE`), an `IN` list over
/// strings, a call of a registry function on a string argument (`contains`, `lower`, `regexp_*`,
/// ...) or a user-defined function: work that costs far more per row than a comparison.
pub(super) fn costly(e: &Expr) -> bool {
    costly_at(e, 0)
}

fn costly_at(e: &Expr, depth: usize) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let next = depth + 1;
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let on_string = || {
                f.args
                    .iter()
                    .any(|a| a.resolved_type().is_some_and(|t| t.is_string_type()))
            };
            let here = match &f.id {
                FunctionId::Builtin(
                    Builtin::Like
                    | Builtin::ILike
                    | Builtin::RLike
                    | Builtin::LikeAny
                    | Builtin::LikeAll
                    | Builtin::NotLikeAny
                    | Builtin::NotLikeAll
                    | Builtin::ILikeAny
                    | Builtin::ILikeAll
                    | Builtin::NotILikeAny
                    | Builtin::NotILikeAll,
                ) => true,
                FunctionId::Builtin(Builtin::In) | FunctionId::Registry { .. } => on_string(),
                FunctionId::Builtin(_) => false,
                // UDFs, Python UDFs and any later kind of call.
                _ => true,
            };
            here || f.args.iter().any(|a| costly_at(a, next))
        }
        ExprNode::Resolved(
            ResolvedExpr::Alias { expr, .. } | ResolvedExpr::EvalMode { expr, .. },
        )
        | ExprNode::Cast { expr, .. } => costly_at(expr, next),
        ExprNode::Resolved(ResolvedExpr::Field { base, .. }) => costly_at(base, next),
        _ => false,
    }
}

/// What evaluating a predicate costs per row, from cheapest: the order a
/// [`super::conjunction`] runs its movable conjuncts in, before their kept fractions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Cost {
    /// Comparisons and arithmetic over fixed-width values.
    Cheap,
    /// A comparison of strings (`SearchPhrase <> ''`): a byte compare per row,
    /// still cheaper than copying the open rows out unless very few are open.
    String,
    /// See [`costly`].
    Costly,
}

/// The cost class of `e`.
pub(super) fn class(e: &Expr) -> Cost {
    if costly(e) {
        Cost::Costly
    } else if compares_strings(e, 0) {
        Cost::String
    } else {
        Cost::Cheap
    }
}

fn compares_strings(e: &Expr, depth: usize) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let next = depth + 1;
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let here = matches!(
                f.id,
                FunctionId::Builtin(
                    Builtin::Eq
                        | Builtin::EqNullSafe
                        | Builtin::Lt
                        | Builtin::LtEq
                        | Builtin::Gt
                        | Builtin::GtEq
                )
            ) && f
                .args
                .iter()
                .any(|a| a.resolved_type().is_some_and(|t| t.is_string_type()));
            here || f.args.iter().any(|a| compares_strings(a, next))
        }
        ExprNode::Resolved(
            ResolvedExpr::Alias { expr, .. } | ResolvedExpr::EvalMode { expr, .. },
        )
        | ExprNode::Cast { expr, .. } => compares_strings(expr, next),
        // As `costly_at`: a field of a compared struct.
        ExprNode::Resolved(ResolvedExpr::Field { base, .. }) => compares_strings(base, next),
        _ => false,
    }
}
