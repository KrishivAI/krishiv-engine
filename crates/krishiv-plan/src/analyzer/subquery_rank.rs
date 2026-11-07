// SPDX-License-Identifier: Apache-2.0
//! Ranking and partitioning on a subquery's correlation path (Spark's
//! `DecorrelateInnerQuery`): a `LIMIT` / `OFFSET` keeps ranks `offset + 1`
//! to `offset + limit` of a `row_number()` within each correlation value (or
//! domain value, on a lateral join's path); a window is partitioned by the
//! correlation's inner columns (or the domain's) too.

use super::*;

impl Rewriter {
    /// `LIMIT limit` over `input` (a subquery read for its values) with
    /// its correlation lifted out: the rows ranked up to `limit` within
    /// each value of the correlated keys, in `input`'s order.
    pub(super) fn limited(
        &mut self,
        plan: &Arc<LogicalPlan>,
        b: &Bounded,
    ) -> R<(Arc<LogicalPlan>, Vec<Arc<Expr>>)> {
        let unsupported_limit =
            || unsupported("a correlated subquery's LIMIT of this form", plan.origin);
        let (below, order, layers) = peel_limit(plan, &b.below)?;
        let (inner, corr) = self.pull(&below, true)?;
        let mut partition_by = Vec::new();
        for c in &corr {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.node else {
                return Err(unsupported_limit());
            };
            if f.id != FunctionId::Builtin(Builtin::Eq) || f.args.len() != 2 {
                return Err(unsupported_limit());
            }
            let (a, b) = (&f.args[0], &f.args[1]);
            // The window partitions on one evaluation, the join reads another.
            if !crate::analyzer::sqlfn::is_deterministic(a)
                || !crate::analyzer::sqlfn::is_deterministic(b)
            {
                return Err(unsupported_limit());
            }
            match (has_outer(Node::Expr(a)), has_outer(Node::Expr(b))) {
                (false, true) if inner_columns(b).is_empty() => partition_by.push(Arc::clone(a)),
                (true, false) if inner_columns(a).is_empty() => partition_by.push(Arc::clone(b)),
                _ => return Err(unsupported_limit()),
            }
        }
        let extra: Vec<BoundColumn> = corr.iter().flat_map(inner_columns).collect();
        let out_plan = self.rank(plan, inner, partition_by, order, b, &layers, &extra)?;
        Ok((out_plan, corr))
    }

    /// `LIMIT limit` over `input` on a lateral join's correlation path: the
    /// rows ranked up to `limit` within each domain value, in `input`'s
    /// order. Returns the plan and the domain columns' renumbering.
    pub(super) fn limited_domain(
        &mut self,
        plan: &Arc<LogicalPlan>,
        b: &Bounded,
        distinct: &Arc<LogicalPlan>,
        domain: &[BoundColumn],
    ) -> R<(Arc<LogicalPlan>, Renumber)> {
        let (below, order, layers) = peel_limit(plan, &b.below)?;
        let (inner, renumber) = self.domained(&below, distinct, domain)?;
        let extra = domain_columns(domain, renumber);
        let partition_by = extra.iter().map(|c| col_expr(c, Origin::UNKNOWN)).collect();
        let out_plan = self.rank(plan, inner, partition_by, order, b, &layers, &extra)?;
        Ok((out_plan, renumber))
    }

    /// The rows of `inner` ranked up to `limit` within each `partition_by`,
    /// in `order`, under the `layers` the limit's input had; the `extra`
    /// columns pass through the layers' projections.
    #[expect(clippy::too_many_arguments, reason = "one step of two callers")]
    pub(super) fn rank(
        &mut self,
        plan: &Arc<LogicalPlan>,
        inner: Arc<LogicalPlan>,
        partition_by: Vec<Arc<Expr>>,
        order: Vec<SortOrder>,
        b: &Bounded,
        layers: &[Arc<LogicalPlan>],
        extra: &[BoundColumn],
    ) -> R<Arc<LogicalPlan>> {
        let unsupported_limit =
            || unsupported("a correlated subquery's LIMIT of this form", plan.origin);
        let Some((id, kind)) = self.row_number.clone() else {
            return Err(unsupported_limit());
        };
        let out = output(&inner)?;
        let rank = BoundColumn {
            id: self.fresh(),
            name: "row_number".to_owned(),
            data_type: DataType::Integer,
            nullable: false,
        };
        let function = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id,
                kind,
                args: Vec::new(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Integer,
                nullable: false,
                deterministic: true,
            },
        ))));
        let window = Arc::new(Expr::new(ExprNode::Window(WindowExpr {
            function,
            window: WindowSpecRef::Spec(WindowSpec {
                partition_by: partition_by.clone(),
                order_by: order.clone(),
                frame: Some(WindowFrame::running(FrameUnits::Rows)),
            }),
        })));
        let alias = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
            expr: window,
            output: rank.clone(),
        })));
        let ranked = node(
            PlanNode::Window {
                input: inner,
                exprs: vec![alias],
                partition_by,
                order_by: order,
            },
            plan.origin,
        );
        // Ranks `offset + 1` to `offset + limit` (Spark: `rn > offset AND
        // rn <= offset + limit`); `row_number` is an INT, so a bound past
        // its range is the range's end.
        let bound = |n: i64| {
            Arc::new(Expr::literal(Literal::Integer(
                i32::try_from(n).unwrap_or(i32::MAX),
            )))
        };
        let rn = || col_expr(&rank, Origin::UNKNOWN);
        let mut bounds = Vec::new();
        if b.offset > 0 {
            bounds.push(boolean(Builtin::Gt, vec![rn(), bound(b.offset)]));
        }
        if let Some(l) = b.limit {
            bounds.push(boolean(
                Builtin::LtEq,
                vec![rn(), bound(b.offset.saturating_add(l))],
            ));
        }
        let Some(keep) = and_all(bounds) else {
            return Err(unsupported_limit());
        };
        let filtered = node(
            PlanNode::Filter {
                input: ranked,
                predicate: keep,
            },
            plan.origin,
        );
        let mut out_plan = columns(filtered, &out);
        for layer in layers.iter().rev() {
            let rebuilt = match &layer.node {
                PlanNode::Project { exprs, .. } => {
                    let mut exprs = exprs.clone();
                    let mut have: HashSet<AttrId> = output(layer)?.iter().map(|c| c.id).collect();
                    for col in extra {
                        if have.insert(col.id) {
                            exprs.push(col_expr(col, Origin::UNKNOWN));
                        }
                    }
                    PlanNode::Project {
                        input: out_plan,
                        exprs,
                    }
                }
                PlanNode::SubqueryAlias { alias, .. } => PlanNode::SubqueryAlias {
                    input: out_plan,
                    alias: alias.clone(),
                    columns: Vec::new(),
                },
                _ => return Err(unsupported_limit()),
            };
            out_plan = node(rebuilt, layer.origin);
        }
        Ok(out_plan)
    }
}

/// Whether a window's own expressions read the outer query (Spark refuses
/// it: "Correlated column is not allowed in window function").
pub(super) fn window_reads_outer(
    exprs: &[Arc<Expr>],
    partition_by: &[Arc<Expr>],
    order_by: &[SortOrder],
) -> bool {
    exprs.iter().any(|e| has_outer(Node::Expr(e)))
        || partition_by.iter().any(|e| has_outer(Node::Expr(e)))
        || order_by.iter().any(|o| has_outer(Node::Expr(&o.expr)))
}

/// The window `plan` over `input`, partitioned by `extra` too (the
/// correlation's inner columns, or the domain's), in the node and in each
/// window expression's spec, which must agree.
pub(super) fn repartitioned(
    plan: &Arc<LogicalPlan>,
    input: Arc<LogicalPlan>,
    exprs: &[Arc<Expr>],
    partition_by: &[Arc<Expr>],
    order_by: &[SortOrder],
    extra: &[BoundColumn],
) -> R<Arc<LogicalPlan>> {
    let added: Vec<Arc<Expr>> = extra
        .iter()
        .filter(|c| {
            !partition_by.iter().any(
                |k| matches!(&k.node, ExprNode::Resolved(ResolvedExpr::Column(o)) if o.id == c.id),
            )
        })
        .map(|c| col_expr(c, Origin::UNKNOWN))
        .collect();
    let mut keys = partition_by.to_vec();
    keys.extend(added.iter().cloned());
    let mut items = Vec::with_capacity(exprs.len());
    for e in exprs {
        let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &e.node else {
            return Err(unsupported("a correlated window of this form", plan.origin));
        };
        let ExprNode::Window(WindowExpr {
            function,
            window: WindowSpecRef::Spec(spec),
        }) = &expr.node
        else {
            return Err(unsupported("a correlated window of this form", plan.origin));
        };
        let mut spec = spec.clone();
        spec.partition_by.extend(added.iter().cloned());
        let window = Arc::new(
            Expr::new(ExprNode::Window(WindowExpr {
                function: Arc::clone(function),
                window: WindowSpecRef::Spec(spec),
            }))
            .with_origin(expr.origin),
        );
        items.push(Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: window,
                output: output.clone(),
            }))
            .with_origin(e.origin),
        ));
    }
    Ok(node(
        PlanNode::Window {
            input,
            exprs: items,
            partition_by: keys,
            order_by: order_by.to_vec(),
        },
        plan.origin,
    ))
}
