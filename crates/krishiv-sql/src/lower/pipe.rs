// SPDX-License-Identifier: Apache-2.0
//! Lowering of pipe operators (`input |> operator |> ...`), as Spark's
//! `visitOperatorPipeRightSide` builds them.

use super::*;

/// `item` of a pipe operator under its [`ExprNode::Pipe`] check (Spark's
/// `PipeExpression`): inside the alias when the item has one, as Spark's
/// `AstBuilder` puts it.
fn pipe_item(item: Arc<IrExpr>, clause: PipeClause) -> Arc<IrExpr> {
    let wrap = |expr: Arc<IrExpr>| {
        let origin = expr.origin;
        Arc::new(IrExpr::new(ExprNode::Pipe { expr, clause }).with_origin(origin))
    };
    match &item.node {
        ExprNode::Alias { expr, name } => Arc::new(
            IrExpr::new(ExprNode::Alias {
                expr: wrap(Arc::clone(expr)),
                name: name.clone(),
            })
            .with_origin(item.origin),
        ),
        _ => wrap(item),
    }
}

impl Lowerer {
    /// `input |> operator |> ...`: each operator reads the plan of the
    /// ones before it, as Spark's `visitOperatorPipeRightSide` builds them.
    pub(super) fn pipe(&mut self, p: &Pipe) -> R<LogicalPlan> {
        let mut plan = self.deeper(|l| l.body(&p.input))?;
        for op in &p.ops {
            // Spark's `PipeOperator(left)`: the operator reads what the one
            // before it output, never a column that one dropped.
            plan = self.node(PlanNode::PipeOperator { input: arc(plan) }, op.span);
            plan = self.pipe_op(plan, op)?;
        }
        Ok(plan)
    }

    fn pipe_op(&mut self, plan: LogicalPlan, op: &PipeOp) -> R<LogicalPlan> {
        let star = |l: &Self, except: Vec<Vec<String>>| {
            arc(l.ex(
                ExprNode::Star(Star {
                    regex: None,
                    replace: Vec::new(),
                    position: None,
                    qualifier: Vec::new(),
                    except,
                    plan_id: None,
                }),
                op.span,
            ))
        };
        Ok(match &op.kind {
            PipeOpKind::Select(s) => {
                let exprs = self.with_windows(&s.windows, |l| {
                    s.projection
                        .iter()
                        .map(|i| l.select_item(i))
                        .collect::<R<Vec<_>>>()
                })?;
                // `|> SELECT items GROUP BY keys` is an aggregate whose
                // select list is the items.
                let plan = match &s.group_by {
                    Some(g) => {
                        let grouping = self.grouping(g)?;
                        self.node(
                            PlanNode::Aggregate {
                                input: arc(plan),
                                grouping,
                                aggregates: exprs,
                            },
                            s.span,
                        )
                    }
                    None => self.node(
                        PlanNode::Project {
                            input: arc(plan),
                            exprs,
                        },
                        s.span,
                    ),
                };
                self.wrap_distinct_hints(plan, s.quantifier, &s.hints, s.span)?
            }
            PipeOpKind::Extend(items) => {
                let mut exprs = vec![star(self, Vec::new())];
                for item in items {
                    let item = self.select_item(item)?;
                    exprs.push(pipe_item(item, PipeClause::Extend));
                }
                self.node(
                    PlanNode::Project {
                        input: arc(plan),
                        exprs,
                    },
                    op.span,
                )
            }
            PipeOpKind::Drop(names) => {
                let except = names
                    .iter()
                    .map(|n| n.parts.iter().map(|p| p.value.clone()).collect())
                    .collect();
                let exprs = vec![star(self, except)];
                self.node(
                    PlanNode::Project {
                        input: arc(plan),
                        exprs,
                    },
                    op.span,
                )
            }
            PipeOpKind::As(name) => self.node(
                PlanNode::SubqueryAlias {
                    input: arc(plan),
                    alias: name.value.clone(),
                    columns: Vec::new(),
                },
                op.span,
            ),
            PipeOpKind::Where(w) => {
                let predicate = arc(self.expr(&w.condition)?);
                self.node(
                    PlanNode::Filter {
                        input: arc(plan),
                        predicate,
                    },
                    op.span,
                )
            }
            PipeOpKind::Pivot(p, None) => self.pivot(plan, p)?,
            PipeOpKind::Unpivot(u, None) => self.unpivot(plan, u)?,
            PipeOpKind::Join(j) => self.join(plan, j)?,
            PipeOpKind::SetOp {
                op: set,
                quantifier,
                right,
            } => {
                let right = arc(self.deeper(|l| l.body(right))?);
                self.set_operation(*set, *quantifier, arc(plan), right, op.span)
            }
            PipeOpKind::Organization(o) => self.organize_parts(
                plan,
                &Organization {
                    order_by: &o.order_by,
                    cluster_by: &o.cluster_by,
                    distribute_by: &o.distribute_by,
                    sort_by: &o.sort_by,
                    order_span: o.order_span,
                    offset: o.offset.as_ref(),
                    limit: o.limit.as_ref(),
                    span: o.span,
                },
            )?,
            PipeOpKind::Aggregate(a) => self.pipe_aggregate(plan, a)?,
            PipeOpKind::Set(assignments) => {
                // One projection per assignment, so a later one reads the
                // value an earlier one set (`SET z = 1, zz = z + 1`).
                let mut plan = plan;
                for a in assignments {
                    // Validation rejects multi-part keys.
                    let Some(name) = a.key.last() else { continue };
                    let value = pipe_item(arc(self.expr(&a.value)?), PipeClause::Set);
                    let star = self.ex(
                        ExprNode::Star(Star {
                            regex: None,
                            qualifier: Vec::new(),
                            except: Vec::new(),
                            replace: vec![(name.value.clone(), value)],
                            position: None,
                            plan_id: None,
                        }),
                        a.span,
                    );
                    plan = self.node(
                        PlanNode::Project {
                            input: arc(plan),
                            exprs: vec![arc(star)],
                        },
                        a.span,
                    );
                }
                plan
            }
            PipeOpKind::Sample(t) => self.sample(plan, t)?,
            PipeOpKind::Pivot(_, Some(_)) | PipeOpKind::Unpivot(_, Some(_)) => {
                return Err(unsupported("PIVOT with UNPIVOT", op.span));
            }
        })
    }

    /// The input column a pipe `GROUP BY` number names.
    fn position(&self, e: &Expr) -> Option<usize> {
        match &e.kind {
            ExprKind::Literal(Literal::Number {
                text,
                kind: NumKind::Int,
            }) => text
                .parse::<i32>()
                .ok()
                .and_then(|n| usize::try_from(n).ok()),
            _ => None,
        }
    }

    /// `AGGREGATE [aggregates] [GROUP BY keys]`: the result is the keys,
    /// then the aggregates.
    fn pipe_aggregate(&mut self, plan: LogicalPlan, a: &PipeAggregate) -> R<LogicalPlan> {
        let mut keys = Vec::new();
        let mut items = Vec::new();
        for g in a.group_by.iter().flat_map(|g| &g.items) {
            match &g.kind {
                GroupItemKind::Expr(e) => {
                    // A number names an input column, unlike `GROUP BY n`
                    // of a query, which names a select item.
                    let key = match self.position(e) {
                        Some(position) => arc(self.ex(
                            ExprNode::Star(Star {
                                regex: None,
                                qualifier: Vec::new(),
                                except: Vec::new(),
                                replace: Vec::new(),
                                position: Some(position),
                                plan_id: None,
                            }),
                            e.span,
                        )),
                        None => arc(self.expr(e)?),
                    };
                    items.push(Arc::clone(&key));
                    keys.push(key);
                }
                GroupItemKind::Named(item) => {
                    keys.push(arc(self.expr(&item.expr)?));
                    items.push(self.select_item(item)?);
                }
                _ => return Err(unsupported("grouping sets in a pipe AGGREGATE", g.span)),
            }
        }
        for item in &a.aggregates {
            let item = self.select_item(item)?;
            items.push(pipe_item(item, PipeClause::Aggregate));
        }
        Ok(self.node(
            PlanNode::Aggregate {
                input: arc(plan),
                grouping: Grouping::GroupBy(keys),
                aggregates: items,
            },
            a.span,
        ))
    }
}
