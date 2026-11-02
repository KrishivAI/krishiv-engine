// SPDX-License-Identifier: Apache-2.0
//! Semantic keys of resolved expressions (`semantic_key`): equality up to Spark's canonicalization.

use super::*;

/// How Spark's canonicalisation (`Expression.canonicalized`) orders a call's
/// operands.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Canon {
    /// Operands in order.
    Ordered,
    /// `+ * AND OR`: nested calls of the same operator are flattened and
    /// the operands sorted (`greatest`/`least` and `= <=>` sort too, with
    /// nothing to flatten for the latter).
    Commutative { flatten: bool },
    /// `a > b` is `b < a`, `a >= b` is `b <= a`: the call is renamed and
    /// its operands swapped.
    Flipped(Builtin),
}

pub(super) fn canon(f: &ResolvedFunction) -> Canon {
    match &f.id {
        FunctionId::Builtin(Builtin::Add | Builtin::Multiply | Builtin::And | Builtin::Or) => {
            Canon::Commutative { flatten: true }
        }
        FunctionId::Builtin(Builtin::Eq | Builtin::EqNullSafe) => {
            Canon::Commutative { flatten: false }
        }
        FunctionId::Builtin(Builtin::Gt) => Canon::Flipped(Builtin::Lt),
        FunctionId::Builtin(Builtin::GtEq) => Canon::Flipped(Builtin::LtEq),
        FunctionId::Registry { name, .. } if matches!(name.as_str(), "greatest" | "least") => {
            Canon::Commutative { flatten: false }
        }
        _ => Canon::Ordered,
    }
}

/// A semantic key: equal for expressions that compute the same value,
/// whatever their origins, column spellings or (top-level) alias, and
/// canonicalised as Spark's `semanticEquals` is (commutative operands
/// sorted, `a > b` as `b < a`).
pub(in crate::analyzer) fn semantic_key(e: &Expr, memo: &mut HashMap<usize, u128>) -> u128 {
    let e = match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => &**expr,
        _ => e,
    };
    let mut stack = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if memo.contains_key(&addr(x)) {
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(&**c));
        if !expanded {
            stack.push((x, true));
            stack.extend(kids.into_iter().map(|c| (c, false)));
            continue;
        }
        let mut name = None;
        let mut order = Canon::Ordered;
        let mut b = match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => format!("c{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => format!("o{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                order = canon(f);
                let id = match order {
                    Canon::Flipped(to) => to.name(),
                    _ => f.id.name(),
                };
                name = Some(f);
                // The ordering keys are children; their directions are not.
                let directions: Vec<_> = f
                    .within_group
                    .iter()
                    .map(|o| (o.direction, o.nulls))
                    .collect();
                format!("f{id}:{:?}:{}:{directions:?}", f.kind, f.distinct)
            }
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => format!("a{}", output.id.0),
            ExprNode::Literal(v) => format!("l{v:?}"),
            ExprNode::Cast { to, mode, .. } => format!("x{}:{mode:?}", to.simple_string()),
            ExprNode::Subquery { plan, .. } => format!("s{}", addr(&**plan)),
            // The children (function, partitioning, ordering, offsets) are
            // keyed below; the shape says which is which.
            ExprNode::Window(w) => match &w.window {
                crate::WindowSpecRef::Named(name) => format!("wn{name}"),
                crate::WindowSpecRef::Spec(spec) => format!(
                    "w{}:{:?}:{:?}",
                    spec.partition_by.len(),
                    spec.order_by
                        .iter()
                        .map(|o| (o.direction, o.nulls))
                        .collect::<Vec<_>>(),
                    spec.frame.as_ref().map(|f| (
                        f.units,
                        frame_shape(&f.lower),
                        frame_shape(&f.upper)
                    )),
                ),
            },
            other => format!("u{other:?}"),
        }
        .into_bytes();
        let mut hashes: Vec<u128> = Vec::new();
        match (order, name) {
            (Canon::Commutative { flatten }, Some(f)) => {
                // Flattening reads a nested operator's operands again for each
                // way it is shared, so it stops at a bound; past it the
                // operands are the call's own (equal expressions written the
                // same way still match).
                let mut budget = FLATTEN_BUDGET;
                if !(flatten && collect_operands(x, f, memo, &mut hashes, &mut budget)) {
                    hashes.clear();
                    hashes.extend(
                        kids.iter()
                            .map(|c| memo.get(&addr(*c)).copied().unwrap_or(0)),
                    );
                }
                hashes.sort_unstable();
            }
            (Canon::Flipped(_), _) => {
                hashes.extend(
                    kids.iter()
                        .rev()
                        .map(|c| memo.get(&addr(*c)).copied().unwrap_or(0)),
                );
            }
            _ => hashes.extend(
                kids.iter()
                    .map(|c| memo.get(&addr(*c)).copied().unwrap_or(0)),
            ),
        }
        for h in hashes {
            b.extend_from_slice(&h.to_le_bytes());
        }
        memo.insert(addr(x), XxHash3_128::oneshot(&b));
    }
    memo.get(&addr(e)).copied().unwrap_or(0)
}

/// The most nodes [`collect_operands`] reads for one call.
pub(super) const FLATTEN_BUDGET: usize = 64;

/// Adds the operand hashes of `x`, a call of `f`, to `out`: those of nested
/// calls of the same operator and result type are gathered too (Spark's
/// `gatherCommutative`). `false` when `budget` nodes were not enough, which
/// keeps a shared operand (a DAG that is exponentially large as a tree) from
/// being read once per path.
pub(super) fn collect_operands(
    x: &Expr,
    f: &ResolvedFunction,
    memo: &HashMap<usize, u128>,
    out: &mut Vec<u128>,
    budget: &mut usize,
) -> bool {
    let mut done = true;
    x.for_each_child(&mut |c| {
        if !done {
            return;
        }
        let Some(left) = budget.checked_sub(1) else {
            done = false;
            return;
        };
        *budget = left;
        if let ExprNode::Resolved(ResolvedExpr::Function(g)) = &c.node
            && g.id == f.id
            && g.data_type == f.data_type
        {
            done = collect_operands(c, g, memo, out, budget);
        } else {
            out.push(memo.get(&addr(&**c)).copied().unwrap_or(0));
        }
    });
    done
}

/// A frame bound's kind, for [`semantic_key`] (an offset's value is a
/// child).
pub(super) const fn frame_shape(b: &crate::FrameBound) -> u8 {
    match b {
        crate::FrameBound::UnboundedPreceding => 0,
        crate::FrameBound::CurrentRow => 1,
        crate::FrameBound::UnboundedFollowing => 2,
        crate::FrameBound::Offset(_) => 3,
    }
}
