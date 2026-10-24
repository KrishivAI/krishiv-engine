// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes on pipe operators (`|>`) and on the
//! query result clauses they share with queries.
//!
//! | Check | Error class |
//! |---|---|
//! | `ORDER BY` with `CLUSTER BY`, or `SORT BY` with `ORDER BY` | `UNSUPPORTED_FEATURE.COMBINATION_QUERY_RESULT_CLAUSES` |
//! | two result clauses in one operator (`ORDER BY ... LIMIT`) | `MULTIPLE_QUERY_RESULT_CLAUSES_WITH_PIPE_OPERATORS` |
//! | `WINDOW` as an operator of its own | `UNSUPPORTED_FEATURE.CLAUSE_WITH_PIPE_OPERATORS` |
//! | `WHERE ... WINDOW` | `NOT_ALLOWED_IN_PIPE_OPERATOR_WHERE.WINDOW_CLAUSE` |
//! | `PIVOT` with `UNPIVOT` in one operator | `NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT` |
//! | a dotted `SET` key, a bare `AGGREGATE` | `_LEGACY_ERROR_TEMP_0035` |
//! | `AGGREGATE ... GROUP BY` with `ALL`, `ROLLUP`, `CUBE`, `GROUPING SETS`, `GROUPING`, window functions | `UNSUPPORTED_FEATURE.PIPE_OPERATOR_AGGREGATE_UNSUPPORTED_CASE` |

use krishiv_common::error::{
    LegacyErrorTemp0035, MultipleQueryResultClausesWithPipeOperators,
    NotAllowedInFromUnpivotWithPivot, NotAllowedInPipeOperatorWhereWindowClause,
    UnsupportedFeatureClauseWithPipeOperators, UnsupportedFeatureCombinationQueryResultClauses,
    UnsupportedFeaturePipeOperatorAggregateUnsupportedCase,
};

use super::{Checker, R, at};
use crate::ast::{
    Expr, ExprKind, GroupBy, GroupItemKind, GroupWith, Pipe, PipeAggregate, PipeOp, PipeOpKind,
    Query, QueryOrganization, Span,
};

fn not_allowed(message: &str, span: Span) -> crate::error::ParseError {
    at(
        LegacyErrorTemp0035 {
            message: message.to_owned(),
        },
        span,
    )
}

/// The clause a query result carries, as Spark names it: `ORDER BY`, `SORT
/// BY`, `DISTRIBUTE BY`, both of the last two, or `CLUSTER BY`; `None` for
/// no sorting clause, `Err` for a combination Spark does not support.
fn result_clause(
    order: bool,
    sort: bool,
    distribute: bool,
    cluster: bool,
) -> Result<Option<&'static str>, ()> {
    match (order, sort, distribute, cluster) {
        (false, false, false, false) => Ok(None),
        (true, false, false, false) => Ok(Some("ORDER BY")),
        (false, true, false, false) => Ok(Some("SORT BY")),
        (false, false, true, false) => Ok(Some("DISTRIBUTE BY")),
        (false, true, true, false) => Ok(Some("SORT BY ... DISTRIBUTE BY")),
        (false, false, false, true) => Ok(Some("CLUSTER BY")),
        _ => Err(()),
    }
}

impl Checker<'_> {
    pub(super) fn pipe(&self, p: &Pipe) -> R {
        self.set_expr(&p.input)?;
        p.ops.iter().try_for_each(|op| self.pipe_op(op))
    }

    fn pipe_op(&self, op: &PipeOp) -> R {
        match &op.kind {
            PipeOpKind::Select(s) => {
                self.select_items(&s.projection)?;
                if let Some(g) = &s.group_by {
                    self.group_by(g)?;
                }
                self.windows(&s.windows)?;
                s.hints
                    .iter()
                    .try_for_each(|h| h.params.iter().try_for_each(|e| self.expr(e)))
            }
            PipeOpKind::Extend(items) => self.select_items(items),
            PipeOpKind::Set(assignments) => {
                if assignments.iter().any(|a| a.key.len() > 1) {
                    return Err(not_allowed(
                        "SQL pipe syntax |> SET operator with multi-part assignment key \
                         (only single-part keys are allowed)",
                        op.span,
                    ));
                }
                assignments.iter().try_for_each(|a| self.expr(&a.value))
            }
            PipeOpKind::Where(w) => {
                if !w.windows.is_empty() {
                    return Err(at(NotAllowedInPipeOperatorWhereWindowClause {}, op.span));
                }
                self.expr(&w.condition)
            }
            PipeOpKind::Pivot(p, other) => {
                if other.is_some() {
                    return Err(at(NotAllowedInFromUnpivotWithPivot {}, op.span));
                }
                self.select_items(&p.aggregates)?;
                self.select_items(&p.values)
            }
            PipeOpKind::Unpivot(_, other) => {
                if other.is_some() {
                    return Err(at(NotAllowedInFromUnpivotWithPivot {}, op.span));
                }
                Ok(())
            }
            PipeOpKind::Sample(s) => self.sample(s),
            PipeOpKind::Join(j) => self.join(j),
            PipeOpKind::SetOp { right, .. } => self.set_expr(right),
            PipeOpKind::Organization(o) => self.pipe_organization(o),
            PipeOpKind::Aggregate(a) => self.pipe_aggregate(a, op.span),
            _ => Ok(()),
        }
    }

    /// The combination check of a query's result clauses (their expressions
    /// are checked with the query).
    pub(super) fn query_result_clauses(&self, q: &Query) -> R {
        let (o, s, d, c) = (
            !q.order_by.is_empty(),
            !q.sort_by.is_empty(),
            !q.distribute_by.is_empty(),
            !q.cluster_by.is_empty(),
        );
        result_clause(o, s, d, c)
            .map(drop)
            .map_err(|()| at(UnsupportedFeatureCombinationQueryResultClauses {}, q.span))
    }

    fn pipe_organization(&self, o: &QueryOrganization) -> R {
        let mut clause = result_clause(
            !o.order_by.is_empty(),
            !o.sort_by.is_empty(),
            !o.distribute_by.is_empty(),
            !o.cluster_by.is_empty(),
        )
        .map_err(|()| at(UnsupportedFeatureCombinationQueryResultClauses {}, o.span))?;
        self.sort_items(&o.order_by)?;
        o.cluster_by
            .iter()
            .chain(&o.distribute_by)
            .try_for_each(|e| self.expr(e))?;
        self.sort_items(&o.sort_by)?;
        self.windows(&o.windows)?;
        if !o.windows.is_empty() {
            return Err(at(
                UnsupportedFeatureClauseWithPipeOperators {
                    clauses: "the WINDOW clause".to_owned(),
                },
                o.span,
            ));
        }
        let multiple = |first: &str, second: &str| -> crate::error::ParseError {
            at(
                MultipleQueryResultClausesWithPipeOperators {
                    clause1: first.to_owned(),
                    clause2: second.to_owned(),
                },
                o.span,
            )
        };
        if let Some(e) = &o.offset {
            if let Some(first) = clause {
                return Err(multiple(first, "OFFSET"));
            }
            self.expr(e)?;
            clause = Some("OFFSET");
        }
        if let Some(e) = o.limit.as_ref().and_then(|l| l.value.as_ref()) {
            if let Some(first) = clause.filter(|c| *c != "OFFSET") {
                return Err(multiple(first, "LIMIT"));
            }
            self.expr(e)?;
        }
        Ok(())
    }

    fn pipe_aggregate(&self, a: &PipeAggregate, span: Span) -> R {
        if a.aggregates.is_empty() && a.group_by.is_none() {
            return Err(not_allowed(
                "The AGGREGATE clause requires a list of aggregate expressions or a list of \
                 grouping expressions, or both",
                span,
            ));
        }
        if let Some(g) = &a.group_by {
            self.group_by(g)?;
        }
        self.select_items(&a.aggregates)?;
        let Some(g) = &a.group_by else {
            return Ok(());
        };
        let unsupported = |case: &str| -> crate::error::ParseError {
            at(
                UnsupportedFeaturePipeOperatorAggregateUnsupportedCase {
                    case: case.to_owned(),
                },
                g.span,
            )
        };
        if let Some(case) = unsupported_grouping(g, a) {
            return Err(unsupported(case));
        }
        Ok(())
    }
}

/// The first construct of an `AGGREGATE` with `GROUP BY` that Spark does not
/// support, in the order it finds them: the grouping, then the aggregates.
fn unsupported_grouping(g: &GroupBy, a: &PipeAggregate) -> Option<&'static str> {
    if let [item] = g.items.as_slice()
        && g.with.is_none()
        && let GroupItemKind::Expr(e) = &item.kind
        && let ExprKind::Column(parts) = &e.kind
        && let [name] = parts.as_slice()
        && !name.quoted
        && name.value.eq_ignore_ascii_case("ALL")
    {
        return Some("GROUP BY ALL");
    }
    match &g.with {
        Some(GroupWith::Rollup) => return Some("GROUP BY ROLLUP"),
        Some(GroupWith::Cube) => return Some("GROUP BY CUBE"),
        Some(GroupWith::GroupingSets(_)) => return Some("GROUPING SETS"),
        _ => {}
    }
    for item in &g.items {
        let found = match &item.kind {
            GroupItemKind::Rollup(_) => Some("GROUP BY ROLLUP"),
            GroupItemKind::Cube(_) => Some("GROUP BY CUBE"),
            GroupItemKind::GroupingSets(_) => Some("GROUPING SETS"),
            GroupItemKind::Expr(e) => unsupported_in(e),
            GroupItemKind::Named(i) => unsupported_in(&i.expr),
            _ => None,
        };
        if found.is_some() {
            return found;
        }
    }
    a.aggregates.iter().find_map(|i| unsupported_in(&i.expr))
}

const WINDOW_CASE: &str = "window functions; please update the query to move the window \
                           functions to a subsequent |> SELECT operator instead";

/// The first `GROUPING` / `GROUPING_ID` call or window function in `e`
/// (subqueries aside), parents before children.
fn unsupported_in(e: &Expr) -> Option<&'static str> {
    let mut stack = vec![e];
    while let Some(e) = stack.pop() {
        if let ExprKind::Function(f) = &e.kind {
            if let [name] = f.name.parts.as_slice()
                && f.args.len() == 1
            {
                for case in ["GROUPING", "GROUPING_ID"] {
                    if name.value.eq_ignore_ascii_case(case) {
                        return Some(case);
                    }
                }
            }
            if f.over.is_some() {
                return Some(WINDOW_CASE);
            }
        }
        let before = stack.len();
        children(e, &mut stack);
        stack[before..].reverse();
    }
    None
}

/// The subexpressions of `e`, in order, pushed onto `out`.
fn children<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
    match &e.kind {
        ExprKind::Field(a, _)
        | ExprKind::Collate(a, _)
        | ExprKind::VariantGet(a, _)
        | ExprKind::Unary(_, a)
        | ExprKind::Not(a)
        | ExprKind::Nested(a)
        | ExprKind::IsNull { expr: a, .. }
        | ExprKind::IsTruth { expr: a, .. }
        | ExprKind::Cast { expr: a, .. }
        | ExprKind::Lambda { body: a, .. }
        | ExprKind::InSubquery { expr: a, .. } => out.push(a),
        ExprKind::Subscript(a, b)
        | ExprKind::Binary(a, _, b)
        | ExprKind::IsDistinctFrom {
            left: a, right: b, ..
        }
        | ExprKind::Like {
            expr: a,
            pattern: b,
            ..
        }
        | ExprKind::RLike {
            expr: a,
            pattern: b,
            ..
        } => out.extend([&**a, &**b]),
        ExprKind::Between {
            expr, low, high, ..
        } => out.extend([&**expr, &**low, &**high]),
        ExprKind::InList { expr, list, .. } => {
            out.push(expr);
            out.extend(list);
        }
        ExprKind::LikeQuantified { expr, patterns, .. } => {
            out.push(expr);
            out.extend(patterns);
        }
        ExprKind::Case {
            operand,
            branches,
            else_result,
        } => {
            out.extend(operand.as_deref());
            for b in branches {
                out.extend([&b.condition, &b.result]);
            }
            out.extend(else_result.as_deref());
        }
        ExprKind::Function(f) => {
            out.extend(f.args.iter().map(|a| &a.expr));
            out.extend(f.within_group.iter().map(|s| &s.expr));
            out.extend(f.filter.as_deref());
        }
        ExprKind::Row(items) | ExprKind::Struct(items) => out.extend(items.iter().map(|i| &i.expr)),
        _ => {}
    }
}
