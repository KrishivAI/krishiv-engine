// SPDX-License-Identifier: Apache-2.0
//! Plan predicates and helpers of the subquery rewrite: shape tests, no-rows values and the legacy checks.

use super::*;

/// Whether a node holds what only a domain join can decorrelate: a set
/// operation (as the analyzer rewrites them) or a join.
pub(super) fn set_op_or_join(p: &LogicalPlan) -> bool {
    matches!(
        p.node,
        PlanNode::Union { .. } | PlanNode::ReplicateRows { .. } | PlanNode::Join { .. }
    )
}

/// [`set_op_or_join`], a window over a correlation that is not an equality
/// (partitioned by the domain), or a select list that reads the outer query: below a
/// domain it reads the domain's columns, and `pull` cannot lift it.
pub(in crate::analyzer) fn needs_domain(p: &LogicalPlan) -> bool {
    set_op_or_join(p)
        || matches!(&p.node, PlanNode::Project { exprs, .. }
            if exprs.iter().any(|e| has_outer(Node::Expr(e))))
        || matches!(&p.node, PlanNode::Window { input, .. } if non_equality_correlation(input))
}

/// What sends an `EXISTS` through the domain: [`needs_domain`], or a
/// `LIMIT 0` / `OFFSET` over a correlation that is not an equality (a
/// positive `LIMIT` alone is pulled away, the answer is the same).
pub(super) fn exists_hit(p: &LogicalPlan) -> bool {
    needs_domain(p)
        || (split::limit_hit(p) && bounded(p).is_some_and(|b| b.offset > 0 || b.limit == Some(0)))
}

/// The first operator of `p`, from the top, whose own expressions read the
/// outer query (Spark's `failOnOuterReferenceInPlan` walk).
pub(super) fn outer_host(p: &Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
    // A nested subquery's reference to a column `p` itself defines stays
    // inside `p`: only references to attributes `p` does not define escape.
    let mut defined: HashSet<AttrId> = HashSet::new();
    let _ = visit_distinct(Node::Plan(p), |n| {
        if let Node::Plan(q) = n {
            defined.extend(q.resolved_output().unwrap_or_default().iter().map(|c| c.id));
        }
        ControlFlow::<()>::Continue(())
    });
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(p)];
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(&*x)) {
            continue;
        }
        let mut reads = false;
        x.for_each_expr(&mut |e| reads |= escapes(e, &defined));
        if reads {
            return Some(x);
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(Arc::clone(c)));
        stack.extend(kids.into_iter().rev());
    }
    None
}

/// [`needs_domain`], a limit (ranked within each domain value), a global
/// aggregate (one row for every outer row, matches or not) or a grouped
/// aggregate over a correlation that is not an equality (`pull` would group
/// by the correlated column, which only an equality allows).
pub(super) fn lateral_hit(p: &LogicalPlan) -> bool {
    needs_domain(p)
        || matches!(&p.node, PlanNode::Limit { .. } | PlanNode::Offset { .. })
        || matches!(&p.node, PlanNode::Aggregate { grouping: Grouping::GroupBy(keys), input, .. }
            if keys.is_empty() || non_equality_correlation(input))
}

/// Whether a filter on the correlation path from `p` down holds an outer
/// reference outside an equality of an inner column and an outer expression.
pub(in crate::analyzer) fn non_equality_correlation(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Filter { input, predicate } => {
            // As `pull` reads it: the common conjuncts of a disjunction first.
            let predicate = &factor_or(predicate);
            conjuncts(predicate)
                .iter()
                .any(|c| has_outer(Node::Expr(c)) && !is_outer_equality(c))
                || non_equality_correlation(input)
        }
        PlanNode::Project { input, .. }
        | PlanNode::Generate { input, .. }
        | PlanNode::Window { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Repartition { input, .. } => non_equality_correlation(input),
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
            non_equality_correlation(input)
        }
        _ => false,
    }
}

/// The select list of `p` when it is a projection over no relation (under
/// plain aliases): one row whatever the outer row.
pub(super) fn one_row_select(p: &Arc<LogicalPlan>) -> Option<Vec<Arc<Expr>>> {
    match &p.node {
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
            one_row_select(input)
        }
        PlanNode::Project { input, exprs } if matches!(input.node, PlanNode::OneRowRelation) => {
            Some(exprs.clone())
        }
        _ => None,
    }
}

/// Whether no expression of the plan `p` is nondeterministic.
pub(in crate::analyzer) fn plan_is_deterministic(p: &Arc<LogicalPlan>) -> bool {
    visit_distinct(Node::Plan(p), |n| match n {
        Node::Expr(e)
            if matches!(
                &e.node,
                ExprNode::Resolved(ResolvedExpr::Function(f)) if !f.deterministic
            ) =>
        {
            ControlFlow::Break(())
        }
        _ => ControlFlow::Continue(()),
    })
    .is_continue()
}

/// Whether every filter on the correlation path from `p` down is
/// deterministic: a nondeterministic predicate lifted into a domain join's
/// condition would be evaluated per domain row and inner row, not per row of
/// the subquery's input.
pub(super) fn deterministic_filters(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Filter { input, predicate } => {
            super::super::sqlfn::is_deterministic(predicate) && deterministic_filters(input)
        }
        PlanNode::Project { input, .. }
        | PlanNode::Generate { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Repartition { input, .. } => deterministic_filters(input),
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
            deterministic_filters(input)
        }
        _ => true,
    }
}

/// Whether `hit` holds on the correlation path from `p` down, through what
/// [`Rewriter::domained`] goes through. A global aggregate ends the path: it
/// has one row for an outer row without matches, which a domain would lose.
pub(super) fn on_path(p: &Arc<LogicalPlan>, hit: fn(&LogicalPlan) -> bool) -> bool {
    on_path_through(p, hit, false)
}

/// [`on_path`], also through a global aggregate when `global` (a lateral
/// join's right side: the domain is kept by an outer join above it).
pub(super) fn on_path_through(
    p: &Arc<LogicalPlan>,
    hit: fn(&LogicalPlan) -> bool,
    global: bool,
) -> bool {
    if !has_outer(Node::Plan(p)) {
        return false;
    }
    if hit(p) {
        return true;
    }
    match &p.node {
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Generate { input, .. }
        | PlanNode::Window { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Repartition { input, .. } => on_path_through(input, hit, global),
        PlanNode::Limit { input, .. } | PlanNode::Offset { input, .. } if global => {
            on_path_through(input, hit, global)
        }
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
            on_path_through(input, hit, global)
        }
        PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            ..
        } if !keys.is_empty() || global => on_path_through(input, hit, global),
        _ => false,
    }
}

/// Whether a semi or anti join is what the analyzer makes of `INTERSECT` or
/// `EXCEPT`: the null-safe equality of every column, pairwise.
pub(super) fn is_set_operation(
    left: &Arc<LogicalPlan>,
    right: &Arc<LogicalPlan>,
    on: Option<&Arc<Expr>>,
) -> bool {
    let (Some(on), Some(l), Some(r)) = (on, left.resolved_output(), right.resolved_output()) else {
        return false;
    };
    let parts = conjuncts(on);
    parts.len() == l.len()
        && l.len() == r.len()
        && parts.iter().zip(l.iter().zip(&r)).all(|(c, (a, b))| {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.node else {
                return false;
            };
            f.id == FunctionId::Builtin(Builtin::EqNullSafe)
                && matches!(&f.args[..], [x, y]
                    if column_id(x) == Some(a.id) && column_id(y) == Some(b.id))
        })
}

/// Whether `keys` group by exactly `input`'s columns, in order.
pub(super) fn distinct_of(input: &Arc<LogicalPlan>, keys: &[Arc<Expr>]) -> bool {
    input.resolved_output().is_some_and(|out| {
        out.len() == keys.len()
            && out
                .iter()
                .zip(keys)
                .all(|(c, k)| column_id(k) == Some(c.id))
    })
}

pub(super) fn column_id(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        _ => None,
    }
}

/// The domain's columns as `renumber` placed them.
pub(super) fn domain_columns(domain: &[BoundColumn], renumber: Renumber) -> Vec<BoundColumn> {
    domain
        .iter()
        .map(|c| BoundColumn {
            id: renumber.id(c.id),
            ..c.clone()
        })
        .collect()
}

/// `out` and the domain's columns after it.
pub(super) fn with_domain(
    mut out: Vec<BoundColumn>,
    domain: &[BoundColumn],
    renumber: Renumber,
) -> Vec<BoundColumn> {
    out.extend(domain_columns(domain, renumber));
    out
}

/// `value` as the attribute `output`.
pub(in crate::analyzer) fn alias(value: Arc<Expr>, output: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr: value,
        output: output.clone(),
    })))
}

/// `e`, an aggregate output, over no rows: each aggregate call read as its
/// value over no input (`count` 0, the others NULL).
pub(super) fn over_no_rows(e: &Arc<Expr>) -> R<Arc<Expr>> {
    Arc::clone(e).transform_up(&mut |x: Arc<Expr>| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
            let zero = match builtin_name(f) {
                "count" | "count_if" | "approx_count_distinct" => Literal::Long(0),
                "sum" | "avg" | "min" | "max" | "any_value" | "first" | "last" | "first_value"
                | "last_value" | "stddev" | "stddev_samp" | "stddev_pop" | "variance"
                | "var_samp" | "var_pop" | "bool_and" | "bool_or" | "every" | "some" | "any"
                | "bit_and" | "bit_or" | "bit_xor" | "median" | "mode" | "corr" => {
                    Literal::Null(Some(f.data_type.clone()))
                }
                _ => {
                    return Err(unsupported(
                        "this aggregate in a lateral subquery over no rows",
                        x.origin,
                    ));
                }
            };
            Ok(Arc::new(Expr::literal(zero)))
        }
        _ => Ok(x),
    })
}

/// A `LIMIT`'s input split for ranking: what the limit reads rows of, the
/// order it takes them in, and the select lists and derived tables above
/// the sort, outermost first, which rebuild above the ranking.
pub(super) type PeeledLimit = (Arc<LogicalPlan>, Vec<SortOrder>, Vec<Arc<LogicalPlan>>);

pub(super) fn peel_limit(plan: &Arc<LogicalPlan>, input: &Arc<LogicalPlan>) -> R<PeeledLimit> {
    let unsupported_limit =
        || unsupported("a correlated subquery's LIMIT of this form", plan.origin);
    // The analyzer puts a Project between the Limit and the Sort when
    // the sort columns are not selected: the window ranks below it.
    let ordered = |sort: &Arc<LogicalPlan>| match &sort.node {
        PlanNode::Sort { input, order, .. }
            if !order.iter().any(|o| has_outer(Node::Expr(&o.expr))) =>
        {
            Some((Arc::clone(input), order.clone()))
        }
        _ => None,
    };
    // The projection moves above the ranking filter, so a nondeterministic
    // one would see only the kept rows.
    if let PlanNode::Project { exprs, .. } = &input.node
        && !exprs.iter().all(super::super::sqlfn::is_deterministic)
    {
        return Err(unsupported_limit());
    }
    // The select lists and derived tables above the sort rebuild above
    // the ranking; they keep its order.
    let mut layers = Vec::new();
    let mut at = input;
    loop {
        match &at.node {
            PlanNode::Project { input, exprs }
                if !exprs.iter().any(|e| has_outer(Node::Expr(e))) =>
            {
                layers.push(Arc::clone(at));
                at = input;
            }
            PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
                layers.push(Arc::clone(at));
                at = input;
            }
            _ => break,
        }
    }
    let (below, order) = match ordered(at) {
        Some((below, order)) => {
            let moves = |l: &Arc<LogicalPlan>| match &l.node {
                PlanNode::Project { exprs, .. } => {
                    !exprs.iter().all(super::super::sqlfn::is_deterministic)
                }
                _ => false,
            };
            if layers.iter().any(moves) {
                return Err(unsupported_limit());
            }
            (below, order)
        }
        None => {
            layers.clear();
            (Arc::clone(input), Vec::new())
        }
    };
    // `pull` drops a sort, which is right only with no limit above it.
    if order.is_empty() && sorts(&below) {
        return Err(unsupported_limit());
    }
    Ok((below, order, layers))
}

/// Whether `right` reads a column of `left`: a lateral join's right side.
pub(in crate::analyzer) fn reads_left(left: &Arc<LogicalPlan>, right: &Arc<LogicalPlan>) -> bool {
    let Ok(out) = output(left) else {
        return false;
    };
    let ids: HashSet<AttrId> = out.iter().map(|c| c.id).collect();
    plan_outer_columns(right)
        .iter()
        .any(|c| ids.contains(&c.id))
}

/// The outer columns `plan` reads, in the order they are met.
pub(super) fn plan_outer_columns(plan: &Arc<LogicalPlan>) -> Vec<BoundColumn> {
    let mut out: Vec<BoundColumn> = Vec::new();
    let _ = visit_distinct(Node::Plan(plan), |n| {
        if let Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &x.node
            && !out.iter().any(|o| o.id == c.id)
        {
            out.push(c.clone());
        }
        ControlFlow::<()>::Continue(())
    });
    out
}

/// The columns (not outer references) `e` reads.
pub(in crate::analyzer) fn inner_columns(e: &Arc<Expr>) -> Vec<BoundColumn> {
    let mut out: Vec<BoundColumn> = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
            && !out.iter().any(|o| o.id == c.id)
        {
            out.push(c.clone());
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    out
}

/// What a scalar subquery's value `e` is over no rows: `None` when NULL,
/// `Some(0)` for a bare `count`; `None` of the outer option when unknown
/// (Spark evaluates other forms; Krishiv does not yet).
pub(super) fn on_no_rows(e: &Arc<Expr>) -> Option<Option<Literal>> {
    // A bare count is 0 over no rows.
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == FunctionKind::Aggregate
        && builtin_name(f) == "count"
    {
        return Some(Some(Literal::Long(0)));
    }
    if let Some(v) = integer_on_no_rows(e) {
        return Some(v.map(Literal::Long));
    }
    // Otherwise NULL when every path to a leaf meets an aggregate that is
    // NULL over no rows through operators that keep a NULL.
    let mut stack = vec![Arc::clone(e)];
    let mut any_aggregate = false;
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
                if !matches!(builtin_name(f), "sum" | "avg" | "min" | "max" | "any_value") {
                    return None;
                }
                any_aggregate = true;
            }
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if matches!(
                    f.id,
                    FunctionId::Builtin(
                        Builtin::Add
                            | Builtin::Subtract
                            | Builtin::Multiply
                            | Builtin::Divide
                            | Builtin::Remainder
                            | Builtin::IntegralDivide
                            | Builtin::Negative
                    )
                ) =>
            {
                // NULL if either side is: one aggregate side suffices.
                let mut nulls = false;
                for a in &f.args {
                    if on_no_rows(a) == Some(None) {
                        nulls = true;
                    }
                }
                if nulls {
                    any_aggregate = true;
                } else {
                    return None;
                }
            }
            ExprNode::Cast { expr, .. } => stack.push(Arc::clone(expr)),
            _ => return None,
        }
    }
    any_aggregate.then_some(None)
}

/// `e` over no rows when it is integer arithmetic over `count`s (0), other
/// aggregates (NULL) and integer literals: `Some(None)` is NULL.
pub(super) fn integer_on_no_rows(e: &Arc<Expr>) -> Option<Option<i64>> {
    match &e.node {
        ExprNode::Literal(Literal::Long(v)) => Some(Some(*v)),
        ExprNode::Literal(Literal::Integer(v)) => Some(Some(i64::from(*v))),
        ExprNode::Cast {
            expr,
            to: DataType::Long,
            ..
        } => integer_on_no_rows(expr),
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
            match builtin_name(f) {
                "count" => Some(Some(0)),
                "sum" | "avg" | "min" | "max" | "any_value" => Some(None),
                _ => None,
            }
        }
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.args.len() == 2 => {
            let op: fn(i64, i64) -> Option<i64> = match f.id {
                FunctionId::Builtin(Builtin::Add) => i64::checked_add,
                FunctionId::Builtin(Builtin::Subtract) => i64::checked_sub,
                FunctionId::Builtin(Builtin::Multiply) => i64::checked_mul,
                _ => return None,
            };
            let a = integer_on_no_rows(&f.args[0])?;
            let b = integer_on_no_rows(&f.args[1])?;
            match (a, b) {
                (Some(a), Some(b)) => op(a, b).map(Some),
                _ => Some(None),
            }
        }
        _ => None,
    }
}

/// Each correlated predicate is `inner = outer`: the inner and the outer
/// sides, pairwise.
pub(super) fn equality_keys(corr: &[Arc<Expr>], origin: Origin) -> R<Keys> {
    let mut inner_keys = Vec::new();
    let mut outer_keys = Vec::new();
    for c in corr {
        let pair = match &c.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.id == FunctionId::Builtin(Builtin::Eq) && f.args.len() == 2 =>
            {
                let (a, b) = (&f.args[0], &f.args[1]);
                match (has_outer(Node::Expr(a)), has_outer(Node::Expr(b))) {
                    (false, true) if inner_columns(b).is_empty() => Some((a, b)),
                    (true, false) if inner_columns(a).is_empty() => Some((b, a)),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some((i, o)) = pair else {
            return Err(unsupported(
                "correlated scalar subqueries whose correlation is not an equality",
                origin,
            ));
        };
        inner_keys.push(Arc::clone(i));
        outer_keys.push(Arc::clone(o));
    }
    Ok((inner_keys, outer_keys))
}

/// `plan` as a global aggregate of one value under a `HAVING` (a filter on
/// its value, possibly under a projection of that value): the aggregate, the
/// predicate and the value's column.
pub(super) fn having_over_global(
    plan: &Arc<LogicalPlan>,
) -> Option<(Arc<LogicalPlan>, Arc<Expr>, BoundColumn)> {
    let filter = match &plan.node {
        PlanNode::Project { input, exprs } if exprs.len() == 1 => {
            // The projection must be the aggregate's value itself.
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &exprs[0].node else {
                return None;
            };
            let PlanNode::Filter { input: agg, .. } = &input.node else {
                return None;
            };
            let PlanNode::Aggregate { aggregates, .. } = &agg.node else {
                return None;
            };
            let [item] = &aggregates[..] else {
                return None;
            };
            let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) = &item.node else {
                return None;
            };
            if output.id != c.id {
                return None;
            }
            input
        }
        PlanNode::Filter { .. } => plan,
        _ => return None,
    };
    let PlanNode::Filter { input, predicate } = &filter.node else {
        return None;
    };
    let PlanNode::Aggregate {
        grouping: Grouping::GroupBy(keys),
        aggregates,
        ..
    } = &input.node
    else {
        return None;
    };
    let [item] = &aggregates[..] else {
        return None;
    };
    let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) = &item.node else {
        return None;
    };
    keys.is_empty()
        .then(|| (Arc::clone(input), Arc::clone(predicate), output.clone()))
}

/// Under Spark's legacy count handling a grouped scalar subquery reads its
/// value over no rows (a count's 0) where nothing matched; the single join
/// [`Rewriter::scalar_grouped`] builds gives NULL there, so a value that is
/// not NULL over no rows (`empty`) is refused rather than answered NULL.
pub(super) fn legacy_grouped<T>(empty: Option<&T>, origin: Origin) -> R<()> {
    match empty {
        None => Ok(()),
        Some(_) => Err(unsupported(
            "a grouped correlated scalar subquery whose value over no rows is not NULL under \
             spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled",
            origin,
        )),
    }
}

/// Spark's refusal of a correlated reference where it cannot be lifted, naming
/// those of `exprs` that hold one.
pub(in crate::analyzer) fn correlated_reference(
    exprs: &[Arc<Expr>],
    origin: Origin,
) -> AnalysisError {
    let outer: Vec<String> = exprs
        .iter()
        .filter(|e| has_outer(Node::Expr(e)))
        .map(|e| format!("\"{}\"", pretty(e, None)))
        .collect();
    AnalysisError::new(
        UnsupportedSubqueryExpressionCategoryCorrelatedReference {
            sql_exprs: outer.join(", "),
        },
        origin,
    )
}

/// Spark's refusal of an order that reads the outer query.
pub(super) fn correlated_order(order: &[SortOrder], origin: Origin) -> AnalysisError {
    let outer: Vec<String> = order
        .iter()
        .filter(|o| has_outer(Node::Expr(&o.expr)))
        .map(|o| {
            let dir = match o.direction {
                SortDirection::Ascending => "ASC",
                SortDirection::Descending => "DESC",
            };
            let nulls = match o.nulls {
                crate::NullOrdering::NullsFirst => "NULLS FIRST",
                crate::NullOrdering::NullsLast => "NULLS LAST",
            };
            format!("\"{} {dir} {nulls}\"", pretty(&o.expr, None))
        })
        .collect();
    AnalysisError::new(
        UnsupportedSubqueryExpressionCategoryCorrelatedReference {
            sql_exprs: outer.join(", "),
        },
        origin,
    )
}

/// Whether a sort lies on the path from `plan` down that keeps rows in order.
pub(super) fn sorts(plan: &Arc<LogicalPlan>) -> bool {
    match &plan.node {
        PlanNode::Sort { .. } => true,
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::SubqueryAlias { input, .. } => sorts(input),
        _ => false,
    }
}
