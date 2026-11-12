// SPDX-License-Identifier: Apache-2.0
//! Domain joins: the outer values a correlated subquery is evaluated for.

use super::*;

impl Rewriter {
    /// `input` decorrelated through the domain of the outer columns it
    /// reads (Spark's `DecorrelateInnerQuery` on a set operation): each
    /// correlated part joins the distinct values of those columns and they
    /// travel up beside its output, so a set operation or a join runs once
    /// per value. Returns the new input, the domain columns in it and the
    /// outer columns they match, null-safely (a NULL outer value owns its
    /// own rows).
    pub(super) fn domain(
        &mut self,
        base: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
    ) -> R<Lifted> {
        // The outer rows serve as the left input and under the domain: a
        // nondeterministic outer input would be evaluated twice.
        if !plan_is_deterministic(base) {
            return Err(unsupported(
                "a correlated subquery over a nondeterministic outer input",
                input.origin,
            ));
        }
        let domain = plan_outer_columns(input);
        let visible: HashSet<AttrId> = output(base)?.iter().map(|c| c.id).collect();
        if !domain.iter().all(|c| visible.contains(&c.id)) {
            return Err(unsupported(
                "a set operation or join that reads a query further out",
                input.origin,
            ));
        }
        let keys = columns(Arc::clone(base), &domain);
        let exprs: Vec<_> = domain
            .iter()
            .map(|c| col_expr(c, Origin::UNKNOWN))
            .collect();
        let distinct = node(
            PlanNode::Aggregate {
                input: keys,
                grouping: Grouping::GroupBy(exprs.clone()),
                aggregates: exprs,
            },
            Origin::UNKNOWN,
        );
        let (inner, renumber) = self.domained(input, &distinct, &domain)?;
        let inner_keys = domain_columns(&domain, renumber)
            .iter()
            .map(|c| col_expr(c, Origin::UNKNOWN))
            .collect();
        let outer_keys = domain
            .iter()
            .map(|c| {
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(
                    c.clone(),
                ))))
            })
            .collect();
        Ok((inner, inner_keys, outer_keys))
    }

    /// `plan` with the domain columns beside its output and no outer
    /// reference left; the [`Renumber`] gives their ids here.
    pub(super) fn domained(
        &mut self,
        plan: &Arc<LogicalPlan>,
        distinct: &Arc<LogicalPlan>,
        domain: &[BoundColumn],
    ) -> R<(Arc<LogicalPlan>, Renumber)> {
        // Below the last set operation or join, the correlation lifts as
        // it does anywhere else and meets the domain in a join's condition.
        let hit: fn(&LogicalPlan) -> bool = if self.lateral {
            lateral_hit
        } else {
            split::limit_hit
        };
        if !on_path_through(plan, hit, self.lateral) {
            let (inner, corr) = self.pull(plan, true)?;
            let (right, renumber) = self.shifted(Arc::clone(distinct))?;
            let cond = corr.iter().map(|c| renumber.outer(c)).collect();
            let join = node(
                PlanNode::Join {
                    left: inner,
                    right,
                    kind: JoinKind::Inner,
                    condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
                },
                Origin::UNKNOWN,
            );
            let out = with_domain(output(plan)?, domain, renumber);
            return Ok((columns(join, &out), renumber));
        }
        match &plan.node {
            PlanNode::Filter { input, predicate } => {
                if conjuncts(predicate)
                    .iter()
                    .any(|c| has_outer(Node::Expr(c)) && has_subquery(Node::Expr(c)))
                {
                    return Err(unsupported(
                        "correlated predicates with subqueries",
                        plan.origin,
                    ));
                }
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let filter = PlanNode::Filter {
                    input: inner,
                    predicate: renumber.outer(predicate),
                };
                Ok((node(filter, plan.origin), renumber))
            }
            PlanNode::Project { input, exprs } => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let exprs = exprs
                    .iter()
                    .map(|e| renumber.outer(e))
                    .chain(
                        domain_columns(domain, renumber)
                            .iter()
                            .map(|c| col_expr(c, Origin::UNKNOWN)),
                    )
                    .collect();
                let project = PlanNode::Project {
                    input: inner,
                    exprs,
                };
                Ok((node(project, plan.origin), renumber))
            }
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } if columns.is_empty() => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let aliased = PlanNode::SubqueryAlias {
                    input: inner,
                    alias: alias.clone(),
                    columns: Vec::new(),
                };
                Ok((node(aliased, plan.origin), renumber))
            }
            PlanNode::Repartition { input, exprs } => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let node_ = PlanNode::Repartition {
                    input: inner,
                    exprs: exprs.iter().map(|e| renumber.outer(e)).collect(),
                };
                Ok((node(node_, plan.origin), renumber))
            }
            PlanNode::Sort { input, order, .. } => {
                if order.iter().any(|o| has_outer(Node::Expr(&o.expr))) {
                    return Err(correlated_order(order, plan.origin));
                }
                // No limit above (it stops the path): the order means nothing.
                self.domained(input, distinct, domain)
            }
            // A global aggregate has a row for an outer row without matches
            // (a lateral join's right side): aggregated per domain value,
            // it joins the domain on the left, and a row that found no
            // match reads the aggregates over no rows.
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } if keys.is_empty() && self.lateral => {
                if aggregates.iter().any(|e| has_outer(Node::Expr(e))) {
                    return Err(correlated_reference(aggregates, plan.origin));
                }
                self.global_aggregate(plan, input, aggregates, distinct, domain)
            }
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } if !keys.is_empty() => {
                if aggregates.iter().any(|e| has_outer(Node::Expr(e))) {
                    return Err(correlated_reference(aggregates, plan.origin));
                }
                if keys.iter().any(|k| has_outer(Node::Expr(k))) {
                    return Err(correlated_reference(keys, plan.origin));
                }
                let (inner, renumber) = match &input.node {
                    // The analyzer's `INTERSECT` / `EXCEPT`: nothing between
                    // the distinct and the join, where a `SELECT` has a
                    // projection.
                    PlanNode::Join {
                        left,
                        right,
                        kind,
                        condition,
                    } if distinct_of(input, keys) => self.domained_join(
                        input, left, right, *kind, condition, distinct, domain, true,
                    )?,
                    _ => self.domained(input, distinct, domain)?,
                };
                let extra: Vec<Arc<Expr>> = domain_columns(domain, renumber)
                    .iter()
                    .map(|c| col_expr(c, Origin::UNKNOWN))
                    .collect();
                let agg = PlanNode::Aggregate {
                    input: inner,
                    grouping: Grouping::GroupBy(keys.iter().chain(&extra).cloned().collect()),
                    aggregates: aggregates.iter().chain(&extra).cloned().collect(),
                };
                Ok((node(agg, plan.origin), renumber))
            }
            PlanNode::Generate {
                input,
                kind,
                outer,
                arg,
                output,
                names,
                qualifier,
            } => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let generate = PlanNode::Generate {
                    input: inner,
                    kind: *kind,
                    outer: *outer,
                    arg: renumber.outer(arg),
                    output: output.clone(),
                    names: names.clone(),
                    qualifier: qualifier.clone(),
                };
                Ok((node(generate, plan.origin), renumber))
            }
            PlanNode::Limit { .. } | PlanNode::Offset { .. }
                if bounded(plan)
                    .is_some_and(|b| self.lateral || non_equality_correlation(&b.below)) =>
            {
                let Some(b) = bounded(plan) else {
                    return Err(unsupported(
                        "outer references below a join, aggregate, limit or set operation",
                        plan.origin,
                    ));
                };
                self.limited_domain(plan, &b, distinct, domain)
            }
            PlanNode::Window {
                input,
                exprs,
                partition_by,
                order_by,
            } if !window_reads_outer(exprs, partition_by, order_by) => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let extra = domain_columns(domain, renumber);
                let window = repartitioned(plan, inner, exprs, partition_by, order_by, &extra)?;
                Ok((window, renumber))
            }
            PlanNode::ReplicateRows { input, count } => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let replicate = PlanNode::ReplicateRows {
                    input: inner,
                    count: Arc::clone(count),
                };
                Ok((node(replicate, plan.origin), renumber))
            }
            PlanNode::Union {
                inputs,
                by_name: false,
                ..
            } => {
                let mut branches = Vec::with_capacity(inputs.len());
                let mut first = None;
                for branch in inputs {
                    let (inner, renumber) = self.domained(branch, distinct, domain)?;
                    // A union matches by position.
                    let out = with_domain(output(branch)?, domain, renumber);
                    branches.push(columns(inner, &out));
                    first.get_or_insert(renumber);
                }
                let union = PlanNode::Union {
                    inputs: branches,
                    by_name: false,
                    allow_missing_columns: false,
                };
                Ok((node(union, plan.origin), first.unwrap_or(Renumber::NONE)))
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => self.domained_join(plan, left, right, *kind, condition, distinct, domain, false),
            _ => Err(unsupported(
                "outer references below a join, aggregate, limit or set operation",
                plan.origin,
            )),
        }
    }

    /// A global aggregate `plan` (over `input`) on a lateral join's
    /// correlation path: the aggregates per domain value, outer-joined to
    /// the domain, with the value over no rows where nothing matched.
    pub(super) fn global_aggregate(
        &mut self,
        plan: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        aggregates: &[Arc<Expr>],
        distinct: &Arc<LogicalPlan>,
        domain: &[BoundColumn],
    ) -> R<(Arc<LogicalPlan>, Renumber)> {
        let (inner, renumber) = self.domained(input, distinct, domain)?;
        let keys: Vec<Arc<Expr>> = domain_columns(domain, renumber)
            .iter()
            .map(|c| col_expr(c, Origin::UNKNOWN))
            .collect();
        let out = output(plan)?;
        let marker = BoundColumn {
            id: self.fresh(),
            name: "matched".to_owned(),
            data_type: DataType::Boolean,
            nullable: false,
        };
        // The aggregates under fresh attributes, then the keys and a marker.
        let mut inner_out = Vec::with_capacity(out.len());
        let mut grouped = Vec::with_capacity(aggregates.len() + keys.len() + 1);
        let mut defaults = Vec::with_capacity(aggregates.len());
        for (e, o) in aggregates.iter().zip(&out) {
            let value = match &e.node {
                ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
                _ => Arc::clone(e),
            };
            let fresh = BoundColumn {
                id: self.fresh(),
                ..o.clone()
            };
            defaults.push(over_no_rows(&value)?);
            grouped.push(alias(value, &fresh));
            inner_out.push(fresh);
        }
        grouped.extend(keys.iter().cloned());
        grouped.push(alias(
            Arc::new(Expr::literal(Literal::Boolean(true))),
            &marker,
        ));
        let agg = node(
            PlanNode::Aggregate {
                input: inner,
                grouping: Grouping::GroupBy(keys.clone()),
                aggregates: grouped,
            },
            plan.origin,
        );
        let (domain_side, dom) = self.shifted(Arc::clone(distinct))?;
        let cond = domain_columns(domain, dom)
            .iter()
            .zip(&keys)
            .map(|(d, k)| {
                boolean(
                    Builtin::EqNullSafe,
                    vec![col_expr(d, Origin::UNKNOWN), Arc::clone(k)],
                )
            })
            .collect();
        let join = node(
            PlanNode::Join {
                left: domain_side,
                right: agg,
                kind: JoinKind::LeftOuter,
                condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
            },
            plan.origin,
        );
        let unmatched = boolean(
            Builtin::IsNull,
            vec![col_expr(
                &BoundColumn {
                    nullable: true,
                    ..marker
                },
                Origin::UNKNOWN,
            )],
        );
        let mut exprs: Vec<Arc<Expr>> = Vec::with_capacity(out.len() + domain.len());
        for ((o, fresh), default) in out.iter().zip(&inner_out).zip(defaults) {
            let read = col_expr(
                &BoundColumn {
                    nullable: true,
                    ..fresh.clone()
                },
                Origin::UNKNOWN,
            );
            let value = call(
                Builtin::When,
                vec![Arc::clone(&unmatched), default, read],
                o.data_type.clone(),
                o.nullable,
            );
            exprs.push(alias(value, o));
        }
        exprs.extend(
            domain_columns(domain, dom)
                .iter()
                .map(|c| col_expr(c, Origin::UNKNOWN)),
        );
        let project = PlanNode::Project { input: join, exprs };
        Ok((node(project, plan.origin), dom))
    }

    /// A join on the correlation path: the domain goes under the side a
    /// correlated condition must not drop rows of, and under both sides when
    /// both read the outer query. What Spark's `CheckAnalysis` refuses for
    /// the join type is refused with its class.
    #[expect(clippy::too_many_arguments, reason = "the parts of the join")]
    pub(super) fn domained_join(
        &mut self,
        plan: &Arc<LogicalPlan>,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
        distinct: &Arc<LogicalPlan>,
        domain: &[BoundColumn],
        below_distinct: bool,
    ) -> R<(Arc<LogicalPlan>, Renumber)> {
        let on = match condition {
            JoinCondition::On(on) => Some(on),
            JoinCondition::None => None,
            _ => return Err(unsupported("this join condition", plan.origin)),
        };
        let (in_left, in_right) = (has_outer(Node::Plan(left)), has_outer(Node::Plan(right)));
        // Which sides take the domain.
        let (under_left, under_right) = match kind {
            JoinKind::Inner => (in_left || !in_right, in_right),
            // `INTERSECT` / `EXCEPT` are a semi / anti join here; Spark checks
            // them as set operations, so both sides may be correlated.
            JoinKind::LeftSemi | JoinKind::LeftAnti
                if in_right && below_distinct && is_set_operation(left, right, on) =>
            {
                (true, true)
            }
            JoinKind::LeftOuter | JoinKind::LeftSemi | JoinKind::LeftAnti if !in_right => {
                (true, false)
            }
            JoinKind::RightOuter if !in_left => (false, true),
            JoinKind::LeftOuter
            | JoinKind::LeftSemi
            | JoinKind::LeftAnti
            | JoinKind::RightOuter
            | JoinKind::FullOuter
                if in_left || in_right =>
            {
                let side = if in_right && kind != JoinKind::RightOuter {
                    right
                } else {
                    left
                };
                return Err(AnalysisError::new(
                    UnsupportedSubqueryExpressionCategoryAccessingOuterQueryColumnIsNotAllowed {
                        tree_node: side.to_string(),
                    },
                    plan.origin,
                ));
            }
            // Spark's answer for a FULL JOIN correlated only in `ON` is not
            // the per-row join: refused rather than answered differently.
            JoinKind::FullOuter => {
                return Err(unsupported(
                    "outer references in the condition of a FULL JOIN",
                    plan.origin,
                ));
            }
            _ => {
                return Err(unsupported(
                    "outer references below a join of this kind",
                    plan.origin,
                ));
            }
        };
        let (l, left_domain) = if under_left {
            let (l, r) = self.domained(left, distinct, domain)?;
            (l, Some(r))
        } else {
            (Arc::clone(left), None)
        };
        let (r, right_domain) = if under_right {
            let (r, n) = self.domained(right, distinct, domain)?;
            (r, Some(n))
        } else {
            (Arc::clone(right), None)
        };
        let renumber = left_domain.or(right_domain).unwrap_or(Renumber::NONE);
        // The domain widens the left side's ids over the right side's, so
        // the right side moves (a join's sides never overlap, and re-analysing
        // must change nothing). Above the join its columns read their old ids.
        let right_ids: HashSet<AttrId> = output(&r)?.iter().map(|c| c.id).collect();
        let (r, moved) = self.disjoint(&l, r)?;
        let moved_column = |c: &BoundColumn| BoundColumn {
            id: if right_ids.contains(&c.id) {
                moved.id(c.id)
            } else {
                c.id
            },
            ..c.clone()
        };
        let mut cond: Vec<Arc<Expr>> = on.iter().map(|c| renumber.outer(c)).collect();
        if let (Some(a), Some(b)) = (left_domain, right_domain) {
            let pairs = domain_columns(domain, a)
                .into_iter()
                .zip(domain_columns(domain, b));
            for (x, y) in pairs {
                let args = vec![col_expr(&x, Origin::UNKNOWN), col_expr(&y, Origin::UNKNOWN)];
                cond.push(boolean(Builtin::EqNullSafe, args));
            }
        }
        let cond = cond
            .into_iter()
            .map(|c| {
                c.transform_up(&mut |x: Arc<Expr>| {
                    Ok::<_, std::convert::Infallible>(match &x.node {
                        ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                            col_expr(&moved_column(col), x.origin)
                        }
                        _ => x,
                    })
                })
                .unwrap_or_else(|never| match never {})
            })
            .collect();
        let join = node(
            PlanNode::Join {
                left: l,
                right: r,
                kind,
                condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
            },
            plan.origin,
        );
        let exprs = with_domain(output(plan)?, domain, renumber)
            .iter()
            .map(|c| {
                let read = col_expr(&moved_column(c), Origin::UNKNOWN);
                if moved_column(c).id == c.id {
                    read
                } else {
                    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: read,
                        output: c.clone(),
                    })))
                }
            })
            .collect();
        let project = PlanNode::Project { input: join, exprs };
        Ok((node(project, plan.origin), renumber))
    }
}
