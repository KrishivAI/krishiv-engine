// SPDX-License-Identifier: Apache-2.0
//! The lift of aggregate calls out of expressions, and the check of what an expression contains.

use super::*;

/// The state of [`Resolver::lift`]: the aggregate it reads (and grows),
/// its grouping keys and the lateral aliases of the select list.
pub(in crate::analyzer) struct Lift<'a> {
    pub(in crate::analyzer) info: &'a mut AggInfo,
    pub(in crate::analyzer) keys: &'a [(u128, ArgType)],
    pub(in crate::analyzer) lateral: &'a HashSet<AttrId>,
    pub(in crate::analyzer) memo: &'a mut HashMap<usize, u128>,
    pub(in crate::analyzer) global: bool,
    pub(in crate::analyzer) origin: Origin,
}

/// The expansion column evaluating `a`: the input column it already is,
/// or the fresh column computed for its semantic key (`None` when neither
/// holds, for the caller to leave the rewrite).
/// Whether `e` reads a subquery.
pub(in crate::analyzer) fn has_subquery(e: &Arc<Expr>) -> bool {
    crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n
            && matches!(x.node, ExprNode::Subquery { .. })
        {
            return std::ops::ControlFlow::Break(());
        }
        std::ops::ControlFlow::Continue(())
    })
    .is_break()
}

pub(in crate::analyzer) fn expand_col(
    a: &Arc<Expr>,
    scope: &Rel,
    fresh: &[(u128, Attr)],
    memo: &mut HashMap<usize, u128>,
) -> Option<Attr> {
    if let ExprNode::Resolved(ResolvedExpr::Column(col)) = &a.node {
        return scope.out.iter().find(|x| x.col.id == col.id).cloned();
    }
    let k = semantic_key(a, memo);
    fresh.iter().find(|(j, _)| *j == k).map(|(_, x)| x.clone())
}

/// `x` with the rewritten aggregate's calls and grouping expressions
/// replaced: a call the second aggregate's call (keyed by the original's
/// identity string, attribute ids included), a grouping expression its column.
pub(in crate::analyzer) fn subst_distinct(
    x: &Arc<Expr>,
    new_call: &HashMap<String, Arc<Expr>>,
    key_map: &HashMap<u128, Arc<Expr>>,
    memo: &mut HashMap<usize, u128>,
) -> Arc<Expr> {
    if let Some(n) = new_call.get(&crate::identity_string(x)) {
        return Arc::clone(n);
    }
    if let Some(n) = key_map.get(&semantic_key(x, memo)) {
        return Arc::clone(n);
    }
    Arc::clone(x)
}

pub(in crate::analyzer) fn out_attr(b: &Bound, scope: &[Attr]) -> Attr {
    match &b.expr.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            match scope.iter().find(|a| a.col.id == c.id) {
                Some(a) => Attr {
                    col: c.clone(),
                    ..a.clone()
                },
                None => Attr {
                    col: c.clone(),
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                },
            }
        }
        ExprNode::Resolved(ResolvedExpr::Alias { output, expr }) => Attr {
            col: output.clone(),
            qual: no_qual(),
            source: None,
            // Spark's `Alias` of an attribute keeps the attribute's metadata.
            duplicate: match &expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                    scope.iter().any(|a| a.col.id == c.id && a.duplicate)
                }
                _ => false,
            },
        },
        _ => Attr {
            col: BoundColumn {
                id: AttrId(u64::MAX),
                name: pretty(&b.expr, None),
                data_type: krishiv_types::DataType::Boolean,
                nullable: true,
            },
            qual: no_qual(),
            source: None,
            duplicate: false,
        },
    }
}

/// A select-list item with its top-level alias removed (Spark's
/// `CleanupAliases.trimNonTopLevelAliases` for one level: the analyzer
/// never nests aliases).
pub(in crate::analyzer) fn trim_alias(e: &Arc<Expr>) -> &Arc<Expr> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
        _ => e,
    }
}

/// Whether `e` is a generator call (today only `json_tuple`).
pub(in crate::analyzer) fn is_generator(e: &Arc<Expr>) -> bool {
    matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Generator)
}

/// Whether unresolved `e` is a direct `json_tuple(...)` call, with or
/// without a single `AS name` alias (ASCII case-insensitive, as registry
/// lookup is).
pub(in crate::analyzer) fn is_json_tuple_call(e: &Arc<Expr>) -> bool {
    match &e.node {
        ExprNode::Function(f) => f.name.eq_ignore_ascii_case("json_tuple"),
        ExprNode::Alias { expr, .. } | ExprNode::Pipe { expr, .. } => is_json_tuple_call(expr),
        _ => false,
    }
}

/// Whether `e` holds a generator call anywhere inside (subquery plans
/// resolve on their own; [`Expr::for_each_child`] does not enter them).
/// Expressions are DAGs, so shared nodes are visited once.
pub(in crate::analyzer) fn has_generator(e: &Arc<Expr>) -> bool {
    let mut seen: HashSet<*const ()> = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x).cast::<()>()) {
            continue;
        }
        if is_generator(&x) {
            return true;
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    false
}

/// Spark's `DATATYPE_MISMATCH.FILTER_NOT_BOOLEAN`.
pub(in crate::analyzer) fn boolean_filter(b: &Bound, origin: Origin) -> Result<(), AnalysisError> {
    if b.ty.data_type == Some(krishiv_types::DataType::Boolean) {
        return Ok(());
    }
    let text = format!("\"{}\"", pretty(&b.shown(), None));
    Err(AnalysisError::new(
        DatatypeMismatchFilterNotBoolean {
            sql_expr: text.clone(),
            filter: text,
            r#type: quoted_type(b.ty.data_type.as_ref()),
        },
        origin,
    ))
}

/// `e` with every column renamed by `rename` (an attribute's name in
/// generated text only; ids and types stay).
pub(in crate::analyzer) fn renamed_columns(
    e: &Arc<Expr>,
    rename: &dyn Fn(&BoundColumn) -> Option<String>,
) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |n: Arc<Expr>| {
        Ok::<_, std::convert::Infallible>(match &n.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => match rename(c) {
                Some(name) => col_expr(&BoundColumn { name, ..c.clone() }, n.origin),
                None => n,
            },
            _ => n,
        })
    });
    out.unwrap_or_else(|never| match never {})
}

/// `e` with its columns fully qualified, as Spark's `sql` prints them.
pub(in crate::analyzer) fn qualified(e: &Arc<Expr>, scope: &[Attr]) -> Arc<Expr> {
    renamed_columns(e, &|c| {
        let mut parts: Vec<String> = scope
            .iter()
            .find(|a| a.col.id == c.id)
            .map(|a| a.qual.to_vec())
            .unwrap_or_default();
        parts.push(c.name.clone());
        Some(parts.join("."))
    })
}

/// `e` with each reference to a lateral alias in `ids` read as Spark names
/// it: `lateralAliasReference(name)`.
pub(in crate::analyzer) fn lateral_refs(e: &Arc<Expr>, ids: &[AttrId]) -> Arc<Expr> {
    if ids.is_empty() {
        return Arc::clone(e);
    }
    renamed_columns(e, &|c| {
        ids.contains(&c.id)
            .then(|| format!("lateralAliasReference({})", c.name))
    })
}

/// Spark's `quoteIfNeeded`: a name of letters, digits and `_` (not all
/// digits) stays bare.
pub(in crate::analyzer) fn quote_if_needed(name: &str) -> String {
    let bare = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.chars().all(|c| c.is_ascii_digit());
    if bare {
        name.to_owned()
    } else {
        quoted(&[name.to_owned()])
    }
}

/// The attribute an aliased item defines.
pub(in crate::analyzer) fn alias_id(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => Some(output.id),
        _ => None,
    }
}

/// Whether `keys` is a lone `GROUP BY ALL` / `ORDER BY ALL`: one
/// unqualified, unresolvable-by-shape column named `all` (case-insensitive;
/// the analyzer resolves a real column first, so this only gates the
/// expansion).
pub(in crate::analyzer) fn is_bare_all(keys: &[Arc<Expr>]) -> bool {
    matches!(keys, [k] if is_all(k))
}

/// Whether `e` is an unqualified column named `all` (case-insensitive)
/// without a DataFrame owner.
pub(in crate::analyzer) fn is_all(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Column(c) => {
            c.plan_id.is_none() && matches!(&c.name[..], [n] if n.eq_ignore_ascii_case("all"))
        }
        _ => false,
    }
}

/// Whether `e` calls an aggregate function outside any subquery: a scalar
/// subquery groups like any other value (checked against Spark 4.2.0),
/// whatever it computes inside.
pub(in crate::analyzer) fn contains_aggregate(e: &Arc<Expr>) -> bool {
    let mut found = false;
    contains_walk(e, false, &mut found, &mut |x, _| {
        if matches!(&x.node, ExprNode::Subquery { .. }) {
            Contains::Skip
        } else if matches!(&x.node, ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate)
        {
            Contains::Yes
        } else {
            Contains::No
        }
    });
    found
}

/// Whether `e` names a column outside any aggregate call's arguments: an
/// attribute, or a correlated (`OuterColumn`) reference inside a subquery
/// (columns a subquery reads from its own scope do not count).
pub(in crate::analyzer) fn contains_bare(e: &Arc<Expr>) -> bool {
    let mut found = false;
    contains_walk(e, false, &mut found, &mut |x, in_subquery| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
            Contains::Skip
        }
        ExprNode::Resolved(ResolvedExpr::Column(_)) if !in_subquery => Contains::Yes,
        ExprNode::Resolved(ResolvedExpr::OuterColumn(_)) => Contains::Yes,
        _ => Contains::No,
    });
    found
}

/// What a [`contains_walk`] visitor reports for a node.
pub(in crate::analyzer) enum Contains {
    /// Counts: stop the walk.
    Yes,
    /// Does not count: visit children.
    No,
    /// Does not count: skip the subtree (aggregate arguments).
    Skip,
}

/// Walks `e` and the plans of the subqueries it holds, outside in;
/// `in_subquery` says whether the node is inside a subquery plan. A
/// visitor answering [`Contains::Yes`] stops the walk.
pub(in crate::analyzer) fn contains_walk(
    e: &Arc<Expr>,
    in_subquery: bool,
    found: &mut bool,
    visit: &mut impl FnMut(&Arc<Expr>, bool) -> Contains,
) {
    let mut stack = vec![(Arc::clone(e), in_subquery)];
    while let Some((x, inner)) = stack.pop() {
        if *found {
            return;
        }
        match visit(&x, inner) {
            Contains::Yes => {
                *found = true;
                return;
            }
            Contains::Skip => continue,
            Contains::No => {}
        }
        if let ExprNode::Subquery { plan, .. } = &x.node {
            let mut plans = vec![Arc::clone(plan)];
            while let Some(p) = plans.pop() {
                let mut exprs = Vec::new();
                p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
                stack.extend(exprs.into_iter().map(|e| (e, true)));
                p.for_each_child(&mut |k| plans.push(Arc::clone(k)));
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push((Arc::clone(k), inner)));
    }
}

/// The `QUALIFY` predicate of a wrapped filter predicate, if `e` is the
/// wrapper lowering stamps (see `lower.rs`).
pub(in crate::analyzer) fn qualify_expression(e: &Arc<Expr>) -> Option<&Arc<Expr>> {
    match &e.node {
        ExprNode::Function(f) if is_qualify_wrapper(&f.name) && f.args.len() == 1 => {
            Some(&f.args[0])
        }
        _ => None,
    }
}

/// Whether `name` is lowering's `QUALIFY` marker (see `lower.rs`): the
/// analyzer resolves and strips it, so it is never a routine call.
pub(in crate::analyzer) fn is_qualify_wrapper(name: &str) -> bool {
    name == "qualify"
}

/// The first aggregate call in a `QUALIFY` predicate, outside window
/// expressions and subqueries (neither counts, as for the select list's
/// own aggregates): `QUALIFY` forbids aggregates.
pub(in crate::analyzer) fn first_qualify_aggregate(
    q: &Arc<Expr>,
    r: &Resolver<'_>,
) -> Option<Arc<Expr>> {
    let mut stack = vec![Arc::clone(q)];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Window(_) | ExprNode::Subquery { .. } => continue,
            _ => {}
        }
        if r.is_aggregate(&x) {
            return Some(x);
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    None
}

/// Whether a `QUALIFY` predicate holds a window expression, outside
/// subqueries.
pub(in crate::analyzer) fn has_qualify_window(q: &Arc<Expr>) -> bool {
    let mut stack = vec![Arc::clone(q)];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Subquery { .. } => continue,
            ExprNode::Window(_) => return true,
            _ => {}
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    false
}

/// Whether `e` calls an aggregate function. Subquery plans are not
/// expressions and are not visited, so their own aggregates never count.
pub(in crate::analyzer) fn has_aggregate(e: &Arc<Expr>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(super::super::bind::addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            return true;
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    false
}

/// Spark's `UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_AGGREGATE_FUNC`:
/// an aggregate call (its `FILTER` included) of `b` reads a lateral alias.
pub(in crate::analyzer) fn in_aggregate_call(b: &Bound) -> Result<(), AnalysisError> {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(&b.expr)];
    while let Some(x) = stack.pop() {
        if !seen.insert(super::super::bind::addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            let mut inner = vec![Arc::clone(&x)];
            while let Some(y) = inner.pop() {
                if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &y.node
                    && b.lateral.contains(&c.id)
                {
                    return Err(AnalysisError::new(
                        UnsupportedFeatureLateralColumnAliasInAggregateFunc {
                            agg_func: format!(
                                "\"{}\"",
                                pretty(&lateral_refs(&x, &b.lateral), None)
                            ),
                            lca: quoted(std::slice::from_ref(&c.name)),
                        },
                        Origin::UNKNOWN,
                    ));
                }
                y.for_each_child(&mut |k| inner.push(Arc::clone(k)));
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    Ok(())
}

/// Whether `e` is not foldable in Spark's sense: it reads a column or a
/// subquery, calls a non-deterministic function, or calls an aggregate,
/// window or generator function (Spark's `foldable` is false for those
/// whatever their arguments). Walks each distinct node once, so a shared
/// subtree costs one visit however many paths reach it.
pub(in crate::analyzer) fn not_constant(e: &Expr) -> bool {
    crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        let crate::walk::Node::Expr(x) = n else {
            // A plan is reached only through a subquery, stopped below.
            return std::ops::ControlFlow::Continue(());
        };
        let stop = matches!(
            &x.node,
            ExprNode::Subquery { .. }
                | ExprNode::Window(_)
                | ExprNode::Resolved(
                    ResolvedExpr::Column(_)
                        | ResolvedExpr::OuterColumn(_)
                        | ResolvedExpr::Function(ResolvedFunction {
                            deterministic: false,
                            ..
                        })
                )
        ) || matches!(
            &x.node,
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.kind != FunctionKind::Scalar
        );
        if stop {
            std::ops::ControlFlow::Break(())
        } else {
            std::ops::ControlFlow::Continue(())
        }
    })
    .is_break()
}

/// Whether `p` returns at most one row, as Spark's `maxRows` has it.
pub(in crate::analyzer) fn at_most_one_row(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::OneRowRelation => true,
        PlanNode::InlineTable { rows, .. } => rows.len() <= 1,
        PlanNode::Limit { limit, .. } => matches!(
            &limit.node,
            ExprNode::Literal(Literal::Integer(v)) if (0..=1).contains(v)
        ),
        PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            ..
        } if keys.is_empty() => true,
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Repartition { input, .. }
        | PlanNode::SubqueryAlias { input, .. }
        | PlanNode::Shift { input, .. }
        | PlanNode::Window { input, .. } => at_most_one_row(input),
        _ => false,
    }
}
