// SPDX-License-Identifier: Apache-2.0
//! ORDER BY, LIMIT and OFFSET binding, and subquery aliases.

use super::*;

impl Resolver<'_> {
    /// A `LIMIT` (`offset` false) or `OFFSET` over `input`. Kept out of line:
    /// `resolve` recurses per plan level.
    #[inline(never)]
    pub(in crate::analyzer) fn limit_node(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        count: &Arc<Expr>,
        offset: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let name = if offset { "offset" } else { "limit" };
        let expr = self.limit_like(count, name, &c, env)?;
        // `LIMIT l OFFSET o` is a limit over an offset; Spark bounds `l + o`.
        if let (
            false,
            PlanNode::Offset { offset: below, .. },
            ExprNode::Literal(Literal::Integer(l)),
        ) = (offset, &c.plan.node, &expr.node)
            && let ExprNode::Literal(Literal::Integer(o)) = &below.node
            && i64::from(*l) + i64::from(*o) > i64::from(i32::MAX)
        {
            return Err(AnalysisError::new(
                SumOfLimitAndOffsetExceedsMaxInt {
                    limit: l.to_string(),
                    offset: o.to_string(),
                },
                count.origin,
            ));
        }
        let input = Arc::clone(&c.plan);
        let node = if offset {
            PlanNode::Offset {
                input,
                offset: expr,
            }
        } else {
            PlanNode::Limit { input, limit: expr }
        };
        let mut r = self.make(p.origin, node, c.out.to_vec(), &[&c]);
        r.widen = Some(Arc::new(Widen::Through { child: c }));
        Ok(r)
    }

    /// The count of a `LIMIT` or `OFFSET` (`name`), checked as Spark does:
    /// it must be foldable (`IS_UNFOLDABLE` when it is not: a column, a
    /// nondeterministic call, a subquery), then an `INT`, not NULL and not
    /// negative. Columns resolve against the input's output (the select
    /// list the limit sits on top of), so `LIMIT key` over an input holding `key` is
    /// unfoldable rather than unresolved; a name the input outputs twice
    /// stays ambiguous when nothing below resolves it.
    pub(in crate::analyzer) fn limit_like(
        &mut self,
        e: &Arc<Expr>,
        name: &str,
        input: &Rel,
        env: &Arc<Env>,
    ) -> Result<Arc<Expr>, AnalysisError> {
        // Columns resolve against the input (so `OFFSET id` is "not
        // constant", as in Spark, not an unresolved column). A name the
        // input outputs twice is looked up below it, as for `ORDER BY`.
        let rels = [input];
        let mut cx = Self::cx(
            &input.out,
            &rels,
            env,
            if name == "limit" { "Limit" } else { "Offset" },
        );
        cx.lenient = true;
        let b = self.bind(e, &cx)?;
        // Spark's order: foldable, then the type, then the value.
        let unfoldable = not_constant(&b.expr);
        if !unfoldable && b.ty.data_type != Some(krishiv_types::DataType::Integer) {
            return Err(AnalysisError::new(
                InvalidLimitLikeExpressionDataType {
                    expr: format!("\"{}\"", pretty(&b.shown(), self.ctx.conf.session_zone)),
                    name: name.to_owned(),
                    data_type: quoted_type(b.ty.data_type.as_ref()),
                },
                e.origin,
            ));
        }
        let folded = self.check_limit(&b, name, unfoldable, e.origin, input.plan.origin)?;
        match folded {
            // `CAST('3' AS INT)` and the like: the planner reads a literal.
            Some(v) if !matches!(b.expr.node, ExprNode::Literal(_)) => {
                let v = i32::try_from(v)
                    .map_err(|_| unsupported("a LIMIT or OFFSET out of range", e.origin))?;
                Ok(Arc::new(Expr::literal(Literal::Integer(v))))
            }
            _ => Ok(b.expr),
        }
    }

    pub(in crate::analyzer) fn sort(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        order: &[SortOrder],
        global: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        // `ORDER BY ALL [ASC|DESC] [NULLS ...]`: every output column in
        // order, with that direction (a column named `all` takes
        // precedence, as Spark resolves it first). Only a global sort: in
        // `SORT BY ALL` and `sortWithinPartitions("all")` it is an unknown
        // column (`UNRESOLVED_COLUMN`, checked against Spark 4.2.0). Inside a window
        // specification `all` stays a column: only the sort's own key
        // expands here.
        let expanded;
        let order = if global
            && order.len() == 1
            && is_all(&order[0].expr)
            && !c
                .out
                .iter()
                .any(|a| same(&a.col.name, "all", self.ctx.conf.case_sensitive))
            && !c.agg.as_ref().is_some_and(|agg| {
                agg.keys
                    .iter()
                    .any(|a| same(&a.col.name, "all", self.ctx.conf.case_sensitive))
                    || agg
                        .input
                        .out
                        .iter()
                        .any(|a| same(&a.col.name, "all", self.ctx.conf.case_sensitive))
            }) {
            let template = &order[0];
            expanded = c
                .out
                .iter()
                .map(|a| SortOrder {
                    expr: col_expr(&a.col, template.expr.origin),
                    ..template.clone()
                })
                .collect::<Vec<_>>();
            &expanded[..]
        } else {
            order
        };
        let mut exprs = Vec::new();
        for o in order {
            exprs.push(match &o.expr.node {
                // `ORDER BY n`: the `n`-th output column.
                ExprNode::Literal(Literal::Integer(n)) if self.ctx.conf.order_by_ordinal => {
                    let i = usize::try_from(*n)
                        .ok()
                        .filter(|i| (1..=c.out.len()).contains(i));
                    let Some(i) = i else {
                        return Err(AnalysisError::new(
                            OrderByPosOutOfRange {
                                index: n.to_string(),
                                size: c.out.len().to_string(),
                            },
                            o.expr.origin,
                        ));
                    };
                    col_expr(&c.out[i - 1].col, o.expr.origin)
                }
                _ => Arc::clone(&o.expr),
            });
        }
        let sorted = |exprs: Vec<Arc<Expr>>| -> Vec<SortOrder> {
            order
                .iter()
                .zip(exprs)
                .map(|(o, expr)| SortOrder { expr, ..o.clone() })
                .collect()
        };
        // Over no grouping sets, `ORDER BY grouping(..)` is rejected before
        // its argument resolves (`HAVING` reports the argument first).
        if c.agg.as_ref().is_none_or(|a| a.analytics.is_none()) {
            super::super::analytics::reject_grouping_calls(&exprs)?;
        }
        // Over an aggregate with windows, Spark's `ORDER BY` reads the
        // select list only (nothing resolves through the windows).
        // A name the aggregate outputs twice is `UNRESOLVED_COLUMN` in
        // `ORDER BY` (not `AMBIGUOUS_REFERENCE`): Spark lists the
        // duplicates as proposals (checked against Spark 4.2.0).
        // The lookup below a duplicated name reports it as ambiguous; an
        // `ORDER BY` reports it as unresolved (Spark 4.2.0:
        // `SELECT a AS k, b AS k FROM t ORDER BY k`), only `LIMIT` and
        // `HAVING` say ambiguous. `AMBIGUOUS_REFERENCE` always carries its
        // `name` parameter, so without one (or without a matching sort
        // key) the error passes through unchanged.
        let reported = |e: AnalysisError| -> AnalysisError {
            if !e
                .error
                .error_class()
                .name()
                .starts_with("AMBIGUOUS_REFERENCE")
            {
                return e;
            }
            e.error
                .message_parameters()
                .iter()
                .find_map(|(k, v)| (*k == "name").then(|| v.clone()))
                .and_then(|name| {
                    exprs.iter().find_map(|x| match &x.node {
                        ExprNode::Column(col)
                            if col.plan_id.is_none()
                                && col.name.len() == 1
                                && quoted(&col.name) == name =>
                        {
                            Some(unresolved(&col.name, &c.out, x.origin))
                        }
                        _ => None,
                    })
                })
                .unwrap_or(e)
        };
        let mut no_aggregate = false;
        let (bounds, below, extended) = if c.agg.as_ref().is_some_and(|i| !i.has_windows()) {
            // The grouping attributes resolve a name the aggregate outputs
            // twice (Spark 4.2.0: `SELECT 1 AS a, 2 AS a FROM t(a) GROUP BY a
            // ORDER BY a` succeeds); with no grouping attribute of that name
            // the duplicates are `UNRESOLVED_COLUMN` and not ambiguous.
            match self.over_agg(&c, &exprs, env, "Sort") {
                Ok(found) => found,
                Err(e) => return Err(reported(e)),
            }
        } else {
            // A name the select list outputs twice is looked up below it,
            // as Spark does.
            match self.bind_missing(&c, &exprs, env, "Sort", true) {
                Ok((bounds, wide)) => {
                    // No aggregate below to run a call in (Spark 4.2.0:
                    // `SELECT x FROM t ORDER BY sum(x)` and a sort over
                    // `DISTRIBUTE BY` over an aggregate): refused after the
                    // ordering-type check below, which Spark makes first.
                    no_aggregate = c.agg.is_none();
                    let extended = wide.out.len() > c.out.len();
                    (bounds, wide, extended)
                }
                // `HAVING ... ORDER BY`: the sort reads the aggregate below
                // the filter, which grows what the sort needs. Duplicates in
                // the select list are looked up below it first
                // (`bind_missing` with leniency); when that is ambiguous
                // too, the aggregate below still resolves grouping columns.
                Err(e)
                    if e.error
                        .error_class()
                        .name()
                        .starts_with("UNRESOLVED_COLUMN")
                        || e.error
                            .error_class()
                            .name()
                            .starts_with("AMBIGUOUS_REFERENCE") =>
                {
                    let Some((layers, inner)) = Self::through_to_agg(&c) else {
                        return Err(reported(e));
                    };
                    // Over windows, Spark's `ORDER BY` reads the select list
                    // only: nothing resolves through them to the aggregate.
                    if inner.agg.as_ref().is_some_and(|i| i.has_windows()) {
                        return Err(reported(e));
                    }
                    let Ok((bounds, grown, extended)) = self.over_agg(&inner, &exprs, env, "Sort")
                    else {
                        return Err(reported(e));
                    };
                    let top = if extended {
                        self.rewrapped(&layers, grown)
                    } else {
                        c.clone()
                    };
                    (bounds, top, extended)
                }
                Err(e) => return Err(reported(e)),
            }
        };
        // A map does not order (`INVALID_ORDERING_TYPE`, as `SortOrder`).
        for (o, b) in order.iter().zip(&bounds) {
            if let Some(t) =
                b.ty.data_type
                    .as_ref()
                    .filter(|t| !super::super::coerce::orderable(t))
            {
                let direction = match o.direction {
                    crate::SortDirection::Ascending => "ASC",
                    crate::SortDirection::Descending => "DESC",
                };
                let nulls = match o.nulls {
                    crate::NullOrdering::NullsFirst => "NULLS FIRST",
                    crate::NullOrdering::NullsLast => "NULLS LAST",
                };
                let zone = self.ctx.conf.session_zone;
                return Err(AnalysisError::new(
                    DatatypeMismatchInvalidOrderingType {
                        sql_expr: format!("\"{} {direction} {nulls}\"", pretty(&b.shown(), zone)),
                        function_name: "`sortorder`".to_owned(),
                        data_type: quoted_type(Some(t)),
                    },
                    o.expr.origin,
                ));
            }
        }
        if no_aggregate && let Some(e) = self.misplaced_aggregates(&bounds, p.origin) {
            return Err(e);
        }
        // A generator between the sort and the aggregate: the aggregate
        // cannot grow through it, so Spark leaves the call in the sort and
        // refuses it there. The same for windows over the aggregate: an
        // aggregate call in the `ORDER BY` stays in the sort, which no
        // aggregate may be in (Spark 4.2.0, `window-with-aggregates`).
        let over_windows = c.agg.as_ref().is_some_and(|i| i.has_windows());
        if bounds.iter().any(|b| b.agg.is_some())
            && (Self::holds_generator(&c.plan) || over_windows)
        {
            // Each call as Spark prints it: unqualified (`sum(col1)`), in the order the sort keys read them, a call
            // of several arguments whole (`max_by(b, c)`, `count(DISTINCT b)`), a repeated call listed twice
            // (Spark 4.2.0, `window-with-aggregates` and a local run over a generator).
            let calls: Vec<String> = bounds
                .iter()
                .flat_map(|b| super::super::aggregate_text::aggregate_calls(&b.shown()))
                .map(|c| format!("\"{c}\""))
                .collect();
            if !calls.is_empty() {
                return Err(AnalysisError::new(
                    UnsupportedExprForOperator {
                        invalid_expr_sqls: calls.join(", "),
                    },
                    p.origin,
                ));
            }
        }
        // A key naming a column a `USING` join merged away reads it from the join's output.
        let revealed = self.reveal(&below, &bounds)?;
        let extended = extended || revealed.out.len() > below.out.len();
        let below = revealed;
        let node = PlanNode::Sort {
            input: Arc::clone(&below.plan),
            order: sorted(bounds.into_iter().map(|b| b.expr).collect()),
            global,
        };
        let mut s = self.make(p.origin, node, below.out.to_vec(), &[&below]);
        s.widen = Some(Arc::new(Widen::Through { child: below }));
        Ok(if extended {
            let mut kept = self.restore(&s, &c.out);
            kept.hidden = Arc::clone(&c.hidden);
            kept
        } else {
            s
        })
    }
}
