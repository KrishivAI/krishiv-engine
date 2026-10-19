// SPDX-License-Identifier: Apache-2.0
//! Expression helpers shared by the rules: references, determinism,
//! conjuncts, substitution and semantic equality.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, LogicalPlan,
    ResolvedExpr, ResolvedFunction,
};
use krishiv_types::DataType;

/// A set of attribute ids.
pub(crate) type Attrs = HashSet<AttrId>;

/// A relation or an expression, for walks that enter subquery plans.
#[derive(Clone, Copy)]
enum Item<'a> {
    Plan(&'a LogicalPlan),
    /// An expression, and whether it sits inside a subquery plan.
    Expr(&'a Expr, bool),
}

/// Visits `e` and every expression and relation of its subquery plans
/// once each, without recursing.
fn walk<'a>(e: &'a Expr, mut f: impl FnMut(Item<'a>)) {
    let mut seen: HashSet<*const ()> = HashSet::new();
    let mut stack = vec![Item::Expr(e, false)];
    while let Some(item) = stack.pop() {
        let key = match item {
            Item::Plan(p) => std::ptr::from_ref(p).cast(),
            Item::Expr(x, _) => std::ptr::from_ref(x).cast(),
        };
        if !seen.insert(key) {
            continue;
        }
        f(item);
        match item {
            Item::Plan(p) => {
                p.for_each_expr(&mut |x| stack.push(Item::Expr(x, true)));
                p.for_each_child(&mut |c| stack.push(Item::Plan(c)));
            }
            Item::Expr(x, inside) => {
                if let ExprNode::Subquery { plan, .. } = &x.node {
                    stack.push(Item::Plan(plan));
                }
                x.for_each_child(&mut |c| stack.push(Item::Expr(c, inside)));
            }
        }
    }
}

/// Adds the attributes `e` reads to `out`: its columns, and every outer
/// reference inside its subquery plans (some may belong to a query further
/// out; callers intersect with the attributes they own).
pub(crate) fn references_into(e: &Expr, out: &mut Attrs) {
    walk(e, |item| {
        if let Item::Expr(x, inside) = item {
            match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) if !inside => {
                    out.insert(c.id);
                }
                ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
                    out.insert(c.id);
                }
                _ => {}
            }
        }
    });
}

/// The attributes `e` reads; see [`references_into`].
pub(crate) fn references(e: &Expr) -> Attrs {
    let mut out = Attrs::new();
    references_into(e, &mut out);
    out
}

/// Whether `e` (subquery plans included) calls no nondeterministic function
/// (the registry's flag, which the analyzer copies into each call): a
/// predicate holding one is never moved, and a projection holding one is
/// never inlined.
pub(crate) fn is_deterministic(e: &Expr) -> bool {
    let mut ok = true;
    walk(e, |item| {
        if let Item::Expr(x, _) = item
            && let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && !f.deterministic
        {
            ok = false;
        }
    });
    ok
}

/// Whether `e` holds a subquery expression.
pub(crate) fn has_subquery(e: &Expr) -> bool {
    let mut found = false;
    walk(e, |item| {
        if let Item::Expr(x, false) = item {
            found |= matches!(x.node, ExprNode::Subquery { .. });
        }
    });
    found
}

/// Whether `e` reads a column of an enclosing query.
pub(crate) fn has_outer_reference(e: &Expr) -> bool {
    let mut found = false;
    walk(e, |item| {
        if let Item::Expr(x, _) = item {
            found |= matches!(x.node, ExprNode::Resolved(ResolvedExpr::OuterColumn(_)));
        }
    });
    found
}

/// Whether `e` calls an aggregate function.
pub(crate) fn has_aggregate(e: &Expr) -> bool {
    let mut found = false;
    walk(e, |item| {
        if let Item::Expr(x, false) = item
            && let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
        {
            found |= f.kind == FunctionKind::Aggregate;
        }
    });
    found
}

/// The operator and arguments of a resolved scalar call of a builtin.
pub(crate) fn builtin(e: &Expr) -> Option<(Builtin, &[Arc<Expr>], &ResolvedFunction)> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(
            f @ ResolvedFunction {
                id: FunctionId::Builtin(b),
                kind: FunctionKind::Scalar,
                ..
            },
        )) => Some((*b, &f.args, f)),
        _ => None,
    }
}

/// The conjuncts of `e`, splitting nested `and`s, in order.
pub(crate) fn conjuncts(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        // A repeated (shared) operand adds nothing: `x and x` is `x`.
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        match builtin(&x) {
            Some((Builtin::And, [a, b], _)) => {
                stack.push(Arc::clone(b));
                stack.push(Arc::clone(a));
            }
            _ => out.push(x),
        }
    }
    out
}

/// The disjuncts of `e`, splitting nested `or`s, in order.
pub(crate) fn disjuncts(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        // A repeated (shared) operand adds nothing: `x or x` is `x`.
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        match builtin(&x) {
            Some((Builtin::Or, [a, b], _)) => {
                stack.push(Arc::clone(b));
                stack.push(Arc::clone(a));
            }
            _ => out.push(x),
        }
    }
    out
}

/// A resolved scalar call of a builtin.
pub(crate) fn call(op: Builtin, args: Vec<Arc<Expr>>, data_type: DataType) -> Arc<Expr> {
    let nullable = args.iter().any(|a| nullable(a));
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(op),
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            data_type,
            nullable,
            deterministic: true,
        },
    ))))
}

/// A call like `like` (its type and nullability kept) of builtin `op` on
/// `args`.
pub(crate) fn recall(like: &ResolvedFunction, op: Builtin, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(op),
            args,
            ..like.clone()
        },
    ))))
}

/// From a predicate over both sides of an inner join, a predicate over
/// `side` alone that it implies: for `(a1 and b1) or (a2 and b2)`, `a1 or
/// a2` when each disjunct has a conjunct over `side` alone. Filter pushdown
/// adds it below the join, and join ordering discounts it from the
/// predicate's selectivity.
pub(crate) fn implied(pred: &Arc<Expr>, side: &Attrs) -> Option<Arc<Expr>> {
    let parts = disjuncts(pred);
    if parts.len() < 2 || has_subquery(pred) {
        return None;
    }
    let mut out = Vec::new();
    for d in &parts {
        let mine: Vec<_> = conjuncts(d)
            .into_iter()
            .filter(|c| references(c).is_subset(side))
            .collect();
        out.push(and_all(mine)?);
    }
    or_all(out)
}

/// `a and b and ...`, left-deep; `None` for no predicates.
pub(crate) fn and_all(preds: impl IntoIterator<Item = Arc<Expr>>) -> Option<Arc<Expr>> {
    preds
        .into_iter()
        .reduce(|a, b| call(Builtin::And, vec![a, b], DataType::Boolean))
}

/// `a or b or ...`, left-deep; `None` for no predicates.
pub(crate) fn or_all(preds: impl IntoIterator<Item = Arc<Expr>>) -> Option<Arc<Expr>> {
    preds
        .into_iter()
        .reduce(|a, b| call(Builtin::Or, vec![a, b], DataType::Boolean))
}

/// A literal expression.
pub(crate) fn lit(value: Literal) -> Arc<Expr> {
    Arc::new(Expr::literal(value))
}

/// A column reference.
pub(crate) fn column(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        c.clone(),
    ))))
}

/// Whether `e` may be null, conservatively.
pub(crate) fn nullable(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(l) => matches!(l, Literal::Null(_)),
        ExprNode::Resolved(
            ResolvedExpr::Column(c)
            | ResolvedExpr::OuterColumn(c)
            | ResolvedExpr::Alias { output: c, .. },
        ) => c.nullable,
        ExprNode::Resolved(ResolvedExpr::Function(f)) => f.nullable,
        _ => true,
    }
}

/// The type of `e`, when known.
pub(crate) fn data_type(e: &Expr) -> Option<DataType> {
    match &e.node {
        ExprNode::Literal(l) => crate::fold::literal_type(l),
        ExprNode::Resolved(
            ResolvedExpr::Column(c)
            | ResolvedExpr::OuterColumn(c)
            | ResolvedExpr::Alias { output: c, .. },
        ) => Some(c.data_type.clone()),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => Some(f.data_type.clone()),
        ExprNode::Cast { to, .. } => Some(to.clone()),
        _ => None,
    }
}

/// The expression an output entry computes: an alias's child, or the
/// expression itself.
pub(crate) fn strip_alias(e: &Arc<Expr>) -> &Arc<Expr> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
        _ => e,
    }
}

/// The attribute an output entry (a column or an alias) produces.
pub(crate) fn output_of(e: &Expr) -> Option<&BoundColumn> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::Alias { output: c, .. }) => {
            Some(c)
        }
        _ => None,
    }
}

/// `e` with each column in `map` replaced by its expression. Subquery plans
/// are not entered: callers never substitute into a predicate whose
/// subqueries reference a replaced attribute.
pub(crate) fn substitute(e: &Arc<Expr>, map: &HashMap<AttrId, Arc<Expr>>) -> Arc<Expr> {
    if map.is_empty() {
        return Arc::clone(e);
    }
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => map.get(&c.id).map_or(x, Arc::clone),
            _ => x,
        })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// `e` with every column id lowered by `offset` (no subqueries inside).
pub(crate) fn unshift(e: &Arc<Expr>, offset: u64) -> Option<Arc<Expr>> {
    let out: Result<_, ()> = Arc::clone(e).transform_up(&mut |x| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            let id = AttrId(c.id.0.checked_sub(offset).ok_or(())?);
            Ok(column(&BoundColumn { id, ..c.clone() }))
        }
        _ => Ok(x),
    });
    out.ok()
}

/// Whether `a` and `b` compute the same thing: equal apart from origins.
/// Conservative: subqueries, aliases and filtered calls are never equal.
/// Linear in the distinct node pairs, so shared subexpressions stay cheap.
pub(crate) fn same(a: &Expr, b: &Expr) -> bool {
    same_in(a, b, &mut HashSet::new())
}

fn same_in(a: &Expr, b: &Expr, equal: &mut HashSet<(*const Expr, *const Expr)>) -> bool {
    let key = (std::ptr::from_ref(a), std::ptr::from_ref(b));
    if key.0 == key.1 || equal.contains(&key) {
        return true;
    }
    let result = match (&a.node, &b.node) {
        (ExprNode::Literal(x), ExprNode::Literal(y)) => x == y,
        (
            ExprNode::Resolved(ResolvedExpr::Column(x)),
            ExprNode::Resolved(ResolvedExpr::Column(y)),
        )
        | (
            ExprNode::Resolved(ResolvedExpr::OuterColumn(x)),
            ExprNode::Resolved(ResolvedExpr::OuterColumn(y)),
        ) => x.id == y.id,
        (
            ExprNode::Resolved(ResolvedExpr::Function(x)),
            ExprNode::Resolved(ResolvedExpr::Function(y)),
        ) => {
            (x.id.clone(), x.kind, x.distinct, &x.data_type)
                == (y.id.clone(), y.kind, y.distinct, &y.data_type)
                && x.filter.is_none()
                && y.filter.is_none()
                && x.args.len() == y.args.len()
                && x.args
                    .iter()
                    .zip(&y.args)
                    .all(|(p, q)| same_in(p, q, equal))
        }
        (
            ExprNode::Cast {
                expr: x,
                to: t,
                mode: m,
            },
            ExprNode::Cast {
                expr: y,
                to: u,
                mode: n,
            },
        ) => t == u && m == n && same_in(x, y, equal),
        _ => false,
    };
    if result {
        equal.insert(key);
    }
    result
}
