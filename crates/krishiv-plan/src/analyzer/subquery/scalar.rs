// SPDX-License-Identifier: Apache-2.0
//! Scalar subqueries: joins, HAVING and single-row forms.

use super::*;

impl Rewriter {
    /// The correlated scalar subqueries of `exprs`, each a left outer join
    /// of `base` with its aggregate; `exprs` read the aggregates' values.
    pub(super) fn scalars(
        &mut self,
        base: Arc<LogicalPlan>,
        exprs: Vec<Arc<Expr>>,
    ) -> R<(Arc<LogicalPlan>, Vec<Arc<Expr>>)> {
        let mut base = base;
        let mut values: HashMap<usize, Arc<Expr>> = HashMap::new();
        for e in &exprs {
            for s in subqueries(e) {
                if !self.correlated(&s) || values.contains_key(&addr(&*s)) {
                    continue;
                }
                let ExprNode::Subquery {
                    kind: SubqueryKind::Scalar,
                    plan,
                } = &s.node
                else {
                    return Err(unsupported(
                        "correlated EXISTS and IN subqueries inside OR, NOT or other expressions",
                        s.origin,
                    ));
                };
                let (joined, value) = self.scalar(base, plan)?;
                base = joined;
                values.insert(addr(&*s), value);
            }
        }
        let mut out = Vec::new();
        for e in exprs {
            let replaced = e.transform_up(&mut |x: Arc<Expr>| {
                Ok::<_, std::convert::Infallible>(values.get(&addr(&*x)).cloned().unwrap_or(x))
            });
            out.push(replaced.unwrap_or_else(|never| match never {}));
        }
        Ok((base, out))
    }

    /// A correlated scalar subquery as a left outer join of `base` with
    /// its aggregate grouped by its correlated keys, and the expression
    /// that reads its value.
    pub(super) fn scalar(
        &mut self,
        base: Arc<LogicalPlan>,
        plan: &Arc<LogicalPlan>,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        // A `DISTRIBUTE BY` over the subquery's value (Spark's Category 2)
        // sets partitions a joined subquery does not have; it goes when no
        // expression of it can raise (`DISTRIBUTE BY 1`), as nothing then
        // observes it. One that may raise stays, and the shapes below refuse
        // it rather than drop its evaluation. "May raise" is read as under
        // ANSI (resolved casts carry their own mode), so with ANSI off an
        // arithmetic expression is refused that Spark would accept: an
        // honest refusal, never a dropped error.
        if let PlanNode::Repartition { input, exprs } = &plan.node
            && exprs.iter().all(|e| !crate::raises::may_raise(e, true))
        {
            return self.scalar(base, input);
        }
        if let Some((agg, predicate, value)) = having_over_global(plan) {
            return self.scalar_having(base, &agg, &predicate, &value);
        }
        if let Some(c) = split::composite(plan) {
            return self.scalar_composite(base, &c);
        }
        let PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } = &plan.node
        else {
            self.check_single(plan)?;
            return self.scalar_single(base, plan, None);
        };
        if keys.iter().any(|k| has_outer(Node::Expr(k))) {
            return Err(correlated_reference(keys, plan.origin));
        }
        let [item] = &aggregates[..] else {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        };
        // A `SELECT DISTINCT c` is a grouping by `c` whose item is the bare
        // column; its value is that column.
        let (expr, value): (&Arc<Expr>, BoundColumn) = match &item.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => (expr, output.clone()),
            ExprNode::Resolved(ResolvedExpr::Column(c)) if !keys.is_empty() => (item, c.clone()),
            _ => {
                return Err(unsupported(
                    "a correlated scalar subquery of this shape",
                    plan.origin,
                ));
            }
        };
        let value = &value;
        // Spark rejects outer references in the value
        // (`AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES`, or an outer
        // aggregate's `CORRELATED_REFERENCE`); the join could not bind them.
        if has_outer(Node::Expr(item)) {
            return Err(unsupported(
                "outer references in a correlated scalar subquery's value",
                item.origin,
            ));
        }
        // Grouped, the subquery has no row for an outer row without matches:
        // its value is NULL, whatever the expression (a bare `count` too);
        // with Spark's legacy flag, the value over no rows as for a global one.
        let empty = if keys.is_empty() || self.legacy_scalar_count {
            on_no_rows(expr).ok_or_else(|| {
                unsupported(
                    "a correlated scalar subquery whose value over no rows is not NULL or a count",
                    plan.origin,
                )
            })?
        } else {
            None
        };
        if self.scalar_needs_domain(input, keys.is_empty()) {
            if !keys.is_empty() {
                legacy_grouped(empty.as_ref(), plan.origin)?;
                return self.scalar_grouped(base, plan, input, keys);
            }
            let (inner, inner_keys, outer_keys) = self.domain_counting(&base, input)?;
            return self.scalar_join(
                base,
                plan,
                item,
                value,
                empty,
                inner,
                &inner_keys,
                &outer_keys,
                true,
            );
        }
        let (inner, corr) = self.pull(input, true)?;
        let (inner_keys, outer_keys) = equality_keys(&corr, plan.origin)?;
        // Each grouping expression is a constant or a correlated key, so
        // there is at most one group per outer row.
        let determined = |k: &Arc<Expr>| match &k.node {
            ExprNode::Literal(_) => true,
            ExprNode::Resolved(ResolvedExpr::Column(c)) => inner_keys.iter().any(
                |i| matches!(&i.node, ExprNode::Resolved(ResolvedExpr::Column(o)) if o.id == c.id),
            ),
            _ => false,
        };
        if !keys.iter().all(determined) {
            legacy_grouped(empty.as_ref(), plan.origin)?;
            return self.scalar_grouped(base, plan, input, keys);
        }
        self.scalar_join(
            base,
            plan,
            item,
            value,
            empty,
            inner,
            &inner_keys,
            &outer_keys,
            false,
        )
    }

    /// `(SELECT agg FROM .. WHERE <correlation> HAVING pred(agg))`: the
    /// aggregate's value per outer row, NULL where the `HAVING` rejects it
    /// (a global aggregate has one row, so a rejected row is no row). A
    /// count over no matches is 0 and meets the `HAVING` as such.
    pub(super) fn scalar_having(
        &mut self,
        base: Arc<LogicalPlan>,
        agg: &Arc<LogicalPlan>,
        predicate: &Arc<Expr>,
        value: &BoundColumn,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        let (base, v) = self.scalar(base, agg)?;
        let id = value.id;
        let mut other = false;
        let pred = Arc::clone(predicate).transform_up(&mut |x: Arc<Expr>| {
            Ok::<_, std::convert::Infallible>(match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == id => Arc::clone(&v),
                ExprNode::Resolved(ResolvedExpr::Column(_)) => {
                    other = true;
                    x
                }
                _ => x,
            })
        });
        let pred = pred.unwrap_or_else(|never| match never {});
        if other || has_outer(Node::Expr(&pred)) || has_subquery(Node::Expr(&pred)) {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                predicate.origin,
            ));
        }
        let null = Arc::new(Expr::literal(Literal::Null(Some(value.data_type.clone()))));
        let guarded = call(
            Builtin::When,
            vec![pred, v, null],
            value.data_type.clone(),
            true,
        );
        Ok((base, guarded))
    }

    /// The left outer join of `base` with the aggregate of `inner` (the
    /// subquery's input, its correlation lifted out) grouped by
    /// `inner_keys`, matched to `outer_keys`; and the value it reads.
    #[expect(clippy::too_many_arguments, reason = "the pieces `scalar` has built")]
    pub(super) fn scalar_join(
        &mut self,
        base: Arc<LogicalPlan>,
        plan: &Arc<LogicalPlan>,
        item: &Arc<Expr>,
        value: &BoundColumn,
        empty: Option<Literal>,
        inner: Arc<LogicalPlan>,
        inner_keys: &[Arc<Expr>],
        outer_keys: &[Arc<Expr>],
        null_safe: bool,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        // Over a union the keys are the domain of outer values, NULL
        // included, and a NULL outer value owns the NULL group (Spark joins
        // with `<=>`); a lifted `inner = outer` leaves no NULL group, and
        // the inner NULL keys must not match.
        let op = if null_safe {
            Builtin::EqNullSafe
        } else {
            Builtin::Eq
        };
        // The aggregate outputs its value and its keys.
        let mut aggregates = vec![Arc::clone(item)];
        let mut key_cols: Vec<BoundColumn> = Vec::new();
        let mut grouping: Vec<Arc<Expr>> = Vec::new();
        for k in inner_keys {
            let col = match &k.node {
                // A column met again (`u.a = t.a AND u.a = t.x`) is one key:
                // the aggregate outputs each attribute once.
                ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                    if !key_cols.iter().any(|o| o.id == c.id) {
                        aggregates.push(Arc::clone(k));
                        grouping.push(Arc::clone(k));
                    }
                    c.clone()
                }
                _ => {
                    let (data_type, nullable) = typed(k);
                    let data_type = data_type
                        .ok_or_else(|| unsupported("a correlation key of this form", k.origin))?;
                    let output = BoundColumn {
                        id: self.fresh(),
                        name: pretty(k, None),
                        data_type,
                        nullable,
                    };
                    aggregates.push(Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Alias {
                            expr: Arc::clone(k),
                            output: output.clone(),
                        },
                    ))));
                    grouping.push(Arc::clone(k));
                    output
                }
            };
            key_cols.push(col);
        }
        let agg = node(
            PlanNode::Aggregate {
                input: inner,
                grouping: Grouping::GroupBy(grouping),
                aggregates,
            },
            plan.origin,
        );
        let (right, renumber) = self.disjoint(&base, agg)?;
        let cond = outer_keys
            .iter()
            .zip(&key_cols)
            .map(|(o, k)| {
                let k = renumber.inner(&col_expr(k, Origin::UNKNOWN));
                boolean(op, vec![renumber.inner(o), k])
            })
            .collect();
        let join = PlanNode::Join {
            left: base,
            right,
            kind: JoinKind::LeftOuter,
            condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
        };
        // The value, NULL where no row matched (an outer join's right side).
        let value = BoundColumn {
            id: renumber.id(value.id),
            nullable: true,
            ..value.clone()
        };
        let read = col_expr(&value, Origin::UNKNOWN);
        let read = match empty {
            None => read,
            Some(zero) => {
                let missing = boolean(Builtin::IsNull, vec![Arc::clone(&read)]);
                let zero = Arc::new(Expr::literal(zero));
                call(
                    Builtin::When,
                    vec![missing, zero, read],
                    value.data_type,
                    false,
                )
            }
        };
        Ok((node(join, Origin::UNKNOWN), read))
    }

    /// A correlated scalar subquery without an aggregate, as a `Single`
    /// join of `base` with the subquery (its correlation lifted out) on the
    /// correlated equalities: a left row matching more than one row fails
    /// with `SCALAR_SUBQUERY_TOO_MANY_ROWS`, none reads NULL.
    pub(super) fn scalar_single(
        &mut self,
        base: Arc<LogicalPlan>,
        plan: &Arc<LogicalPlan>,
        forced: Option<JoinKind>,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        let out = output(plan)?;
        let [value] = &out[..] else {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        };
        let value = value.clone();
        // `(SELECT outer_col = 1)`: a select list over no relation is one
        // row whatever the outer row, so its value is the expression read
        // over the outer query's own columns.
        if let PlanNode::Project { input, exprs } = &plan.node
            && matches!(input.node, PlanNode::OneRowRelation)
            && let [e] = &exprs[..]
            && !has_subquery(Node::Expr(e))
        {
            let read = Renumber::NONE.inner(e);
            let read = match &read.node {
                ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
                _ => read,
            };
            return Ok((base, read));
        }
        let through_domain = forced.is_some() || on_path(plan, split::limit_hit);
        if !through_domain && aggregates_below(plan) && !self.grouped_having_fixed(plan)? {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        }
        let (inner, inner_keys, outer_keys) = if through_domain {
            self.domain(&base, plan)?
        } else {
            let (inner, corr) = self.pull(plan, true)?;
            let (inner_keys, outer_keys) = equality_keys(&corr, plan.origin)?;
            (inner, inner_keys, outer_keys)
        };
        let op = if through_domain {
            Builtin::EqNullSafe
        } else {
            Builtin::Eq
        };
        if inner_keys.is_empty() {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        }
        let mut key_cols = Vec::new();
        for k in &inner_keys {
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &k.node else {
                return Err(unsupported("a correlation key of this form", k.origin));
            };
            key_cols.push(c.clone());
        }
        let (right, renumber) = self.disjoint(&base, inner)?;
        let cond = outer_keys
            .iter()
            .zip(&key_cols)
            .map(|(o, k)| {
                let k = renumber.inner(&col_expr(k, Origin::UNKNOWN));
                boolean(op, vec![renumber.inner(o), k])
            })
            .collect();
        let join = PlanNode::Join {
            left: base,
            right,
            kind: forced.unwrap_or(JoinKind::Single),
            condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
        };
        // NULL where no row matched (an outer join's right side).
        let value = BoundColumn {
            id: renumber.id(value.id),
            nullable: true,
            ..value
        };
        Ok((
            node(join, Origin::UNKNOWN),
            col_expr(&value, Origin::UNKNOWN),
        ))
    }

    /// `right` renumbered to fresh attribute ids whatever it overlaps.
    pub(super) fn shifted(&mut self, right: Arc<LogicalPlan>) -> R<(Arc<LogicalPlan>, Renumber)> {
        let Some((lo, hi)) = hull(&right) else {
            return Ok((right, Renumber::NONE));
        };
        let offset = self.next_id.saturating_sub(lo);
        if hi.checked_add(offset).is_none_or(|top| top > MAX_ID) {
            return Err(AnalysisError::new(FailedToParseTooComplex {}, right.origin));
        }
        self.next_id += hi - lo + 1;
        let shift = PlanNode::Shift {
            input: right,
            offset,
        };
        Ok((node(shift, Origin::UNKNOWN), Renumber { lo, hi, offset }))
    }
}
