// SPDX-License-Identifier: Apache-2.0
//! Lowering: the syntax tree of a query into the logical-plan IR.
//!
//! Purely syntactic, as Spark's `AstBuilder`: no catalog, no types beyond
//! literals. Every node's origin is its span in the SQL text. Comma joins
//! fold left into inner joins without a condition; `<>` is `not(==)`; a
//! `SELECT` with aggregates but no `GROUP BY` stays a `Project` (the
//! analyzer turns it into an aggregate); `HAVING` is a `Filter` over the
//! `Aggregate`. Constructs the v1 IR lacks (set operations, `VALUES`,
//! windows, generators, pivots, ...) fail with `KRISHIV_UNSUPPORTED_FEATURE`
//! naming the construct. Join lists and dotted names are lowered with
//! loops; recursion follows the tree's nesting, which the parser bounds,
//! and moves to a fresh stack every `STEP` levels. The plan is checked
//! against `MAX_DEPTH` at the end.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::{
    FailedToParseTooComplex, InvalidNumericLiteralRange, InvalidTypedLiteral, KrishivError,
    KrishivUnsupportedFeature,
};
use krishiv_common::secret::OptionValue;
use krishiv_common::stack::{STEP, on_big_stack};
use krishiv_plan::{
    Builtin, CastMode, ColumnRef, Cte, Expr as IrExpr, ExprNode, FunctionCall, Grouping,
    JoinCondition, JoinKind as IrJoin, Literal as IrLiteral, LogicalPlan, Origin, Plan, PlanNode,
    ReadSource, SortOrder, SourceId, Star, SubqueryKind,
};
use krishiv_types::DecimalType;

use crate::ast::{
    Alias, BinaryOp, Expr, ExprKind, FromClause, GroupItemKind, JoinConstraint, JoinKind, Literal,
    NumKind, Query, Relation, RelationExt, Select, SetExprKind, SetQuantifier, SortItem, Span,
    TableFactor, TableFactorKind, Truth, UnaryOp,
};
use crate::{ParseError, ParserOptions, parse_query};

type R<T> = Result<T, ParseError>;

/// Parses and lowers one query; spans index `source`.
///
/// # Errors
/// A [`ParseError`]: the parser's, or a lowering error with its span.
pub fn parse_and_lower(sql: &str, options: &ParserOptions, source: SourceId) -> R<Plan> {
    lower(&parse_query(sql, options)?, source).map_err(|e| e.redacted_in(sql))
}

/// Lowers `query` into an unresolved plan whose spans index `source`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for constructs the IR lacks, Spark's
/// literal errors (`INVALID_NUMERIC_LITERAL_RANGE`, `INVALID_TYPED_LITERAL`,
/// decimal bounds, unsupported cast types) and `FAILED_TO_PARSE_TOO_COMPLEX`
/// for plans deeper than `MAX_DEPTH`.
pub fn lower(query: &Query, source: SourceId) -> R<Plan> {
    let mut l = Lowerer { source, depth: 0 };
    let plan = l.query(query)?;
    Plan::try_new(plan).map_err(|e| err(e, query.span))
}

fn err(e: impl Into<KrishivError>, span: Span) -> ParseError {
    ParseError {
        error: e.into(),
        span,
    }
}

/// Rejects the query-organization clauses lowering does not support.
fn reject_organization(q: &Query) -> R<()> {
    for (present, what) in [
        (!q.cluster_by.is_empty(), "CLUSTER BY"),
        (!q.distribute_by.is_empty(), "DISTRIBUTE BY"),
        (!q.windows.is_empty(), "WINDOW clauses"),
        (q.offset.is_some(), "OFFSET"),
    ] {
        if present {
            return Err(unsupported(what, q.span));
        }
    }
    Ok(())
}

fn unsupported(feature: &str, span: Span) -> ParseError {
    err(
        KrishivUnsupportedFeature {
            feature: feature.to_owned(),
        },
        span,
    )
}

struct Lowerer {
    source: SourceId,
    depth: usize,
}

fn arc<T>(v: T) -> Arc<T> {
    Arc::new(v)
}

impl Lowerer {
    fn origin(&self, span: Span) -> Origin {
        let at = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        Origin::sql(
            self.source,
            krishiv_plan::Span::new(at(span.start), at(span.end)),
        )
    }

    fn node(&self, node: PlanNode, span: Span) -> LogicalPlan {
        LogicalPlan::new(node).with_origin(self.origin(span))
    }

    fn ex(&self, node: ExprNode, span: Span) -> IrExpr {
        IrExpr::new(node).with_origin(self.origin(span))
    }

    fn call(&self, name: &str, args: Vec<Arc<IrExpr>>, span: Span) -> IrExpr {
        self.ex(
            ExprNode::Function(FunctionCall {
                name: name.to_owned(),
                args,
                distinct: false,
                filter: None,
            }),
            span,
        )
    }

    fn op(&self, b: Builtin, args: Vec<IrExpr>, span: Span) -> IrExpr {
        self.call(b.name(), args.into_iter().map(arc).collect(), span)
    }

    fn not(&self, e: IrExpr, negated: bool, span: Span) -> IrExpr {
        if negated {
            self.op(Builtin::Not, vec![e], span)
        } else {
            e
        }
    }

    /// Runs `f` one level deeper, on a fresh stack every `STEP` levels.
    fn deeper<T: Send>(&mut self, span: Span, f: impl FnOnce(&mut Self) -> R<T> + Send) -> R<T> {
        self.depth += 1;
        let r = if self.depth.is_multiple_of(STEP) {
            on_big_stack(|| f(self)).unwrap_or_else(|| Err(err(FailedToParseTooComplex {}, span)))
        } else {
            f(self)
        };
        self.depth -= 1;
        r
    }

    fn query(&mut self, q: &Query) -> R<LogicalPlan> {
        self.deeper(q.span, |l| l.query_inner(q))
    }

    fn query_inner(&mut self, q: &Query) -> R<LogicalPlan> {
        reject_organization(q)?;
        let mut plan = match &q.body.kind {
            SetExprKind::Select(s) => self.select(s, s.from.as_ref())?,
            SetExprKind::Query(inner) => self.query(inner)?,
            SetExprKind::Table(name) => self.node(
                PlanNode::Read(ReadSource::Table {
                    name: name.parts.iter().map(|p| p.value.clone()).collect(),
                    options: BTreeMap::new(),
                }),
                name.span,
            ),
            SetExprKind::FromStatement(f) => match &f.bodies[..] {
                [body] => match &body.body.kind {
                    SetExprKind::Select(s) => {
                        reject_organization(body)?;
                        let inner = self.select(s, Some(&f.from))?;
                        self.organize(inner, body)?
                    }
                    _ => return Err(unsupported("FROM ... SELECT with a set body", f.span)),
                },
                _ => return Err(unsupported("FROM with several SELECT bodies", f.span)),
            },
            SetExprKind::SetOp { .. } => return Err(unsupported("set operations", q.body.span)),
            SetExprKind::Values(_) => return Err(unsupported("VALUES", q.body.span)),
        };
        plan = self.organize(plan, q)?;
        if let Some(with) = &q.with {
            if with.recursive {
                return Err(unsupported("recursive CTEs", with.span));
            }
            let mut ctes = Vec::new();
            for c in &with.ctes {
                if c.max_recursion_level.is_some() {
                    return Err(unsupported("MAX RECURSION LEVEL", c.span));
                }
                let mut body = self.query(&c.query)?;
                if !c.columns.is_empty() {
                    body = self.node(
                        PlanNode::SubqueryAlias {
                            input: arc(body),
                            alias: c.name.value.clone(),
                            columns: c.columns.iter().map(|i| i.value.clone()).collect(),
                        },
                        c.span,
                    );
                }
                ctes.push(Cte {
                    name: c.name.value.clone(),
                    plan: arc(body),
                });
            }
            plan = self.node(
                PlanNode::WithCte {
                    input: arc(plan),
                    ctes,
                },
                q.span,
            );
        }
        Ok(plan)
    }

    /// `ORDER BY`, `SORT BY` and `LIMIT` of `q` over `plan`.
    fn organize(&mut self, mut plan: LogicalPlan, q: &Query) -> R<LogicalPlan> {
        for (items, global) in [(&q.order_by, true), (&q.sort_by, false)] {
            if !items.is_empty() {
                let order = items.iter().map(|s| self.sort_item(s)).collect::<R<_>>()?;
                // Spark's errors cite the ORDER BY clause.
                let span = q.order_span.filter(|_| global).unwrap_or(q.span);
                plan = self.node(
                    PlanNode::Sort {
                        input: arc(plan),
                        order,
                        global,
                    },
                    span,
                );
            }
        }
        if let Some(limit) = &q.limit
            && let Some(v) = &limit.value
        {
            let limit_expr = arc(self.expr(v)?);
            plan = self.node(
                PlanNode::Limit {
                    input: arc(plan),
                    limit: limit_expr,
                },
                limit.span,
            );
        }
        Ok(plan)
    }

    fn sort_item(&mut self, s: &SortItem) -> R<SortOrder> {
        let expr = arc(self.expr(&s.expr)?);
        let asc = s.asc.unwrap_or(true);
        let mut o = if asc {
            SortOrder::asc(expr)
        } else {
            SortOrder::desc(expr)
        };
        if let Some(first) = s.nulls_first {
            o.nulls = if first {
                krishiv_plan::NullOrdering::NullsFirst
            } else {
                krishiv_plan::NullOrdering::NullsLast
            };
        }
        Ok(o)
    }

    fn select(&mut self, s: &Select, from: Option<&FromClause>) -> R<LogicalPlan> {
        for (present, what) in [
            (!s.hints.is_empty(), "hints"),
            (
                s.quantifier == Some(SetQuantifier::Distinct),
                "SELECT DISTINCT",
            ),
            (!s.lateral_views.is_empty(), "LATERAL VIEW"),
            (!s.windows.is_empty(), "WINDOW clauses"),
            (s.qualify.is_some(), "QUALIFY"),
        ] {
            if present {
                return Err(unsupported(what, s.span));
            }
        }
        let Some(from) = from else {
            return Err(unsupported("SELECT without FROM", s.span));
        };
        let mut plan = self.from(from)?;
        if let Some(w) = &s.selection {
            let predicate = arc(self.expr(w)?);
            plan = self.node(
                PlanNode::Filter {
                    input: arc(plan),
                    predicate,
                },
                s.span,
            );
        }
        let mut items = Vec::new();
        for item in &s.projection {
            let e = self.expr(&item.expr)?;
            items.push(arc(match &item.alias {
                None => e,
                Some(Alias::Name(name)) => self.ex(
                    ExprNode::Alias {
                        expr: arc(e),
                        name: name.value.clone(),
                    },
                    item.span,
                ),
                Some(_) => return Err(unsupported("multi-name aliases", item.span)),
            }));
        }
        if s.group_by.is_none() && s.having.is_none() {
            return Ok(self.node(
                PlanNode::Project {
                    input: arc(plan),
                    exprs: items,
                },
                s.span,
            ));
        }
        let mut keys = Vec::new();
        if let Some(g) = &s.group_by {
            if g.with.is_some() {
                return Err(unsupported("ROLLUP, CUBE and GROUPING SETS", g.span));
            }
            for item in &g.items {
                match &item.kind {
                    GroupItemKind::Expr(e) => keys.push(arc(self.expr(e)?)),
                    _ => return Err(unsupported("ROLLUP, CUBE and GROUPING SETS", item.span)),
                }
            }
        }
        plan = self.node(
            PlanNode::Aggregate {
                input: arc(plan),
                grouping: Grouping::GroupBy(keys),
                aggregates: items,
            },
            s.span,
        );
        if let Some(h) = &s.having {
            let predicate = self.expr(h)?;
            // As Spark's AstBuilder: a `HAVING` that is not a predicate is
            // cast to boolean.
            let predicate = arc(if is_predicate(&predicate) {
                predicate
            } else {
                let span = h.span;
                self.ex(
                    ExprNode::Cast {
                        expr: arc(predicate),
                        to: krishiv_types::DataType::Boolean,
                        mode: CastMode::Default,
                    },
                    span,
                )
            });
            plan = self.node(
                PlanNode::Filter {
                    input: arc(plan),
                    predicate,
                },
                s.span,
            );
        }
        Ok(plan)
    }

    fn from(&mut self, f: &FromClause) -> R<LogicalPlan> {
        if f.pivot.is_some() || f.unpivot.is_some() {
            return Err(unsupported("PIVOT and UNPIVOT", f.span));
        }
        if !f.lateral_views.is_empty() {
            return Err(unsupported("LATERAL VIEW", f.span));
        }
        let mut plan: Option<LogicalPlan> = None;
        for r in &f.relations {
            let right = self.relation(r)?;
            plan = Some(match plan {
                None => right,
                Some(left) => self.node(
                    PlanNode::Join {
                        left: arc(left),
                        right: arc(right),
                        kind: IrJoin::Inner,
                        condition: JoinCondition::None,
                    },
                    r.span,
                ),
            });
        }
        plan.ok_or_else(|| unsupported("an empty FROM clause", f.span))
    }

    fn relation(&mut self, r: &Relation) -> R<LogicalPlan> {
        if r.lateral {
            return Err(unsupported("LATERAL subqueries", r.span));
        }
        let mut plan = self.factor(&r.primary)?;
        for ext in &r.extensions {
            let RelationExt::Join(j) = ext else {
                return Err(unsupported("PIVOT and UNPIVOT", r.span));
            };
            if j.natural || j.lateral {
                return Err(unsupported(
                    if j.natural {
                        "NATURAL JOIN"
                    } else {
                        "LATERAL joins"
                    },
                    j.span,
                ));
            }
            let kind = match j.kind {
                JoinKind::Inner => IrJoin::Inner,
                JoinKind::Cross => IrJoin::Cross,
                JoinKind::Left => IrJoin::LeftOuter,
                JoinKind::Right => IrJoin::RightOuter,
                JoinKind::Full => IrJoin::FullOuter,
                JoinKind::LeftSemi => IrJoin::LeftSemi,
                JoinKind::LeftAnti => IrJoin::LeftAnti,
            };
            let right = self.factor(&j.right)?;
            let condition = match &j.constraint {
                JoinConstraint::None => JoinCondition::None,
                JoinConstraint::On(e) => JoinCondition::On(arc(self.expr(e)?)),
                JoinConstraint::Using(cols) => {
                    JoinCondition::Using(cols.iter().map(|c| c.value.clone()).collect())
                }
            };
            plan = self.node(
                PlanNode::Join {
                    left: arc(plan),
                    right: arc(right),
                    kind,
                    condition,
                },
                j.span,
            );
        }
        Ok(plan)
    }

    fn factor(&mut self, tf: &TableFactor) -> R<LogicalPlan> {
        let plan = match &tf.kind {
            TableFactorKind::Table {
                name,
                temporal,
                options,
            } => {
                if let Some(t) = temporal {
                    return Err(unsupported("time travel", t.span));
                }
                let mut opts = BTreeMap::new();
                for o in options {
                    let key: Vec<&str> = o.key.parts.iter().map(|p| p.value.as_str()).collect();
                    let key = key.join(".");
                    let value = match o.value.as_ref().map(|v| &v.kind) {
                        Some(ExprKind::Literal(Literal::String(s))) => s.clone(),
                        Some(ExprKind::Literal(Literal::Number { text, .. })) => text.clone(),
                        Some(ExprKind::Literal(Literal::Boolean(b))) => b.to_string(),
                        _ => return Err(unsupported("this table option value", o.span)),
                    };
                    let value = OptionValue::from_user(&key, value);
                    opts.insert(key, value);
                }
                let parts: Vec<String> = name.parts.iter().map(|p| p.value.clone()).collect();
                let source = match &name.parts[..] {
                    // `parquet.`s3://...``: a path, kept typed so that its
                    // credentials never sit in a name.
                    [format, path] if path.quoted && is_path(&path.value) => ReadSource::Files {
                        // As written: Spark names it so in errors.
                        format: format.value.clone(),
                        paths: vec![OptionValue::from_user("path", path.value.clone())],
                        options: opts,
                    },
                    _ => ReadSource::Table {
                        name: parts,
                        options: opts,
                    },
                };
                self.node(PlanNode::Read(source), name.span)
            }
            TableFactorKind::Subquery(q) => {
                let inner = self.query(q)?;
                if tf.alias.is_none() {
                    return Ok(self.node(
                        PlanNode::SubqueryAlias {
                            input: arc(inner),
                            alias: "__auto_generated_subquery_name".to_owned(),
                            columns: Vec::new(),
                        },
                        tf.span,
                    ));
                }
                inner
            }
            TableFactorKind::Nested(r) => self.deeper(tf.span, |l| l.relation(r))?,
            TableFactorKind::Values(_) => return Err(unsupported("VALUES", tf.span)),
            _ => return Err(unsupported("table-valued functions", tf.span)),
        };
        Ok(match &tf.alias {
            None => plan,
            Some(a) => self.node(
                PlanNode::SubqueryAlias {
                    input: arc(plan),
                    alias: a.name.value.clone(),
                    columns: a.columns.iter().map(|c| c.value.clone()).collect(),
                },
                tf.span,
            ),
        })
    }

    fn expr(&mut self, e: &Expr) -> R<IrExpr> {
        self.deeper(e.span, |l| l.expr_inner(e))
    }

    fn arg(&mut self, e: &Expr) -> R<Arc<IrExpr>> {
        self.expr(e).map(arc)
    }

    #[expect(clippy::too_many_lines, reason = "one arm per expression kind")]
    fn expr_inner(&mut self, e: &Expr) -> R<IrExpr> {
        let sp = e.span;
        Ok(match &e.kind {
            ExprKind::Literal(v) => self.ex(ExprNode::Literal(literal(v, sp)?), sp),
            ExprKind::Keyword(k) => {
                self.ex(ExprNode::Literal(IrLiteral::String(k.value.clone())), sp)
            }
            ExprKind::Column(parts) => {
                self.column(parts.iter().map(|p| p.value.clone()).collect(), sp)
            }
            ExprKind::Field(..) => {
                // `a.b.c` written through parentheses: a longer name.
                let mut fields = Vec::new();
                let mut base = e;
                loop {
                    match &base.kind {
                        ExprKind::Field(inner, f) => {
                            fields.push(f.value.clone());
                            base = inner;
                        }
                        // Parentheses make no node: `(t.a).b` is `t.a.b`.
                        ExprKind::Nested(inner) => base = inner,
                        _ => break,
                    }
                }
                let ExprKind::Column(parts) = &base.kind else {
                    return Err(unsupported("field access on an expression", sp));
                };
                let mut name: Vec<String> = parts.iter().map(|p| p.value.clone()).collect();
                name.extend(fields.into_iter().rev());
                self.column(name, sp)
            }
            ExprKind::Wildcard { qualifier, except } => {
                if !except.is_empty() {
                    return Err(unsupported("SELECT * EXCEPT", sp));
                }
                self.ex(
                    ExprNode::Star(Star {
                        qualifier: qualifier.iter().map(|p| p.value.clone()).collect(),
                        plan_id: None,
                    }),
                    sp,
                )
            }
            ExprKind::Unary(op, inner) => {
                let name = match op {
                    UnaryOp::Minus => Builtin::Negative.name(),
                    UnaryOp::Plus => "positive",
                    UnaryOp::BitNot => "~",
                };
                let arg = self.arg(inner)?;
                self.call(name, vec![arg], sp)
            }
            ExprKind::Binary(l, op, r) => {
                let (l, r) = (self.expr(l)?, self.expr(r)?);
                let name = match op {
                    BinaryOp::NotEq => {
                        let eq = self.op(Builtin::Eq, vec![l, r], sp);
                        return Ok(self.not(eq, true, sp));
                    }
                    BinaryOp::Or => Builtin::Or.name(),
                    BinaryOp::And => Builtin::And.name(),
                    BinaryOp::Eq => Builtin::Eq.name(),
                    BinaryOp::Lt => Builtin::Lt.name(),
                    BinaryOp::LtEq => Builtin::LtEq.name(),
                    BinaryOp::Gt => Builtin::Gt.name(),
                    BinaryOp::GtEq => Builtin::GtEq.name(),
                    BinaryOp::NullSafeEq => Builtin::EqNullSafe.name(),
                    BinaryOp::Plus => Builtin::Add.name(),
                    BinaryOp::Minus => Builtin::Subtract.name(),
                    BinaryOp::Multiply => Builtin::Multiply.name(),
                    BinaryOp::Divide => Builtin::Divide.name(),
                    BinaryOp::Modulo => Builtin::Remainder.name(),
                    BinaryOp::Concat => Builtin::Concat.name(),
                    BinaryOp::IntegerDivide => "div",
                    BinaryOp::BitAnd => "&",
                    BinaryOp::BitOr => "|",
                    BinaryOp::BitXor => "^",
                    BinaryOp::ShiftLeft => "shiftleft",
                    BinaryOp::ShiftRight => "shiftright",
                    BinaryOp::ShiftRightUnsigned => "shiftrightunsigned",
                };
                self.call(name, vec![arc(l), arc(r)], sp)
            }
            ExprKind::Not(inner) => {
                let v = self.expr(inner)?;
                self.not(v, true, sp)
            }
            ExprKind::IsNull { expr, negated } => {
                let b = if *negated {
                    Builtin::IsNotNull
                } else {
                    Builtin::IsNull
                };
                let v = self.expr(expr)?;
                self.op(b, vec![v], sp)
            }
            ExprKind::IsTruth {
                expr,
                truth,
                negated,
            } => {
                let v = self.expr(expr)?;
                match truth {
                    Truth::Unknown => {
                        let b = if *negated {
                            Builtin::IsNotNull
                        } else {
                            Builtin::IsNull
                        };
                        self.op(b, vec![v], sp)
                    }
                    t => {
                        let lit =
                            self.ex(ExprNode::Literal(IrLiteral::Boolean(*t == Truth::True)), sp);
                        let eq = self.op(Builtin::EqNullSafe, vec![v, lit], sp);
                        self.not(eq, *negated, sp)
                    }
                }
            }
            ExprKind::IsDistinctFrom {
                left,
                right,
                negated,
            } => {
                let (l, r) = (self.expr(left)?, self.expr(right)?);
                let eq = self.op(Builtin::EqNullSafe, vec![l, r], sp);
                self.not(eq, !negated, sp)
            }
            // Spark's errors cite a predicate from its keyword (or `NOT`).
            ExprKind::Between {
                expr,
                low,
                high,
                negated,
                op,
            } => {
                let args = vec![self.expr(expr)?, self.expr(low)?, self.expr(high)?];
                let b = self.op(Builtin::Between, args, *op);
                self.not(b, *negated, *op)
            }
            ExprKind::InList {
                expr,
                list,
                negated,
                op,
            } => {
                let mut args = vec![self.arg(expr)?];
                for item in list {
                    args.push(self.arg(item)?);
                }
                let b = self.call(Builtin::In.name(), args, *op);
                self.not(b, *negated, *op)
            }
            ExprKind::InSubquery {
                expr,
                query,
                negated,
                op,
            } => {
                let values = match &expr.kind {
                    ExprKind::Row(items) if items.iter().all(|i| i.alias.is_none()) => {
                        items.iter().map(|i| self.arg(&i.expr)).collect::<R<_>>()?
                    }
                    _ => vec![self.arg(expr)?],
                };
                let plan = arc(self.query(query)?);
                let s = self.ex(
                    ExprNode::Subquery {
                        kind: SubqueryKind::In(values),
                        plan,
                    },
                    *op,
                );
                self.not(s, *negated, *op)
            }
            ExprKind::Like {
                expr,
                pattern,
                escape,
                negated,
                case_insensitive,
            } => {
                let b = if *case_insensitive {
                    Builtin::ILike
                } else {
                    Builtin::Like
                };
                let mut args = vec![self.arg(expr)?, self.arg(pattern)?];
                if let Some(esc) = escape {
                    args.push(arc(
                        self.ex(ExprNode::Literal(IrLiteral::String(esc.clone())), sp)
                    ));
                }
                let like = self.call(b.name(), args, sp);
                self.not(like, *negated, sp)
            }
            ExprKind::RLike {
                expr,
                pattern,
                negated,
            } => {
                let args = vec![self.expr(expr)?, self.expr(pattern)?];
                let like = self.op(Builtin::RLike, args, sp);
                self.not(like, *negated, sp)
            }
            ExprKind::Exists(q) => {
                let plan = arc(self.query(q)?);
                self.ex(
                    ExprNode::Subquery {
                        kind: SubqueryKind::Exists,
                        plan,
                    },
                    sp,
                )
            }
            ExprKind::Subquery(q) => {
                let plan = arc(self.query(q)?);
                self.ex(
                    ExprNode::Subquery {
                        kind: SubqueryKind::Scalar,
                        plan,
                    },
                    sp,
                )
            }
            ExprKind::Cast {
                expr,
                data_type,
                try_cast,
            } => {
                let to = data_type
                    .to_data_type()
                    .map_err(|e| err(e, data_type.span))?;
                let mode = if *try_cast {
                    CastMode::Try
                } else {
                    CastMode::Default
                };
                let expr = self.arg(expr)?;
                self.ex(ExprNode::Cast { expr, to, mode }, sp)
            }
            ExprKind::Case {
                operand,
                branches,
                else_result,
            } => {
                let operand = operand.as_ref().map(|o| self.arg(o)).transpose()?;
                let mut args = Vec::new();
                for b in branches {
                    let cond = self.arg(&b.condition)?;
                    args.push(match &operand {
                        None => cond,
                        Some(o) => {
                            arc(self.call(Builtin::Eq.name(), vec![Arc::clone(o), cond], b.span))
                        }
                    });
                    args.push(self.arg(&b.result)?);
                }
                if let Some(e) = else_result {
                    args.push(self.arg(e)?);
                }
                self.call(Builtin::When.name(), args, sp)
            }
            ExprKind::Function(f) => {
                let [name] = &f.name.parts[..] else {
                    return Err(unsupported("qualified function names", f.name.span));
                };
                for (present, what) in [
                    (f.over.is_some(), "window functions"),
                    (!f.within_group.is_empty(), "WITHIN GROUP"),
                    (f.null_treatment.is_some(), "IGNORE NULLS and RESPECT NULLS"),
                ] {
                    if present {
                        return Err(unsupported(what, sp));
                    }
                }
                let mut args = Vec::new();
                for a in &f.args {
                    if a.name.is_some() {
                        return Err(unsupported("named function arguments", a.span));
                    }
                    args.push(self.arg(&a.expr)?);
                }
                let filter = f.filter.as_ref().map(|x| self.arg(x)).transpose()?;
                self.ex(
                    ExprNode::Function(FunctionCall {
                        name: name.value.clone(),
                        args,
                        distinct: f.quantifier == Some(SetQuantifier::Distinct),
                        filter,
                    }),
                    sp,
                )
            }
            ExprKind::Nested(inner) => self.expr(inner)?,
            ExprKind::Parameter(_) => return Err(unsupported("parameter markers", sp)),
            ExprKind::Subscript(..) => return Err(unsupported("subscripts", sp)),
            ExprKind::Lambda { .. } => return Err(unsupported("lambda functions", sp)),
            ExprKind::Row(_) | ExprKind::Struct(_) => {
                return Err(unsupported("row and struct constructors", sp));
            }
            _ => return Err(unsupported("LIKE ANY, SOME and ALL", sp)),
        })
    }

    fn column(&self, name: Vec<String>, sp: Span) -> IrExpr {
        self.ex(
            ExprNode::Column(ColumnRef {
                name,
                plan_id: None,
            }),
            sp,
        )
    }
}

/// A path written as a quoted name part (`parquet.`s3://b/k``).
/// Whether `e` is one of Spark's `Predicate` expressions: a comparison,
/// `AND` / `OR` / `NOT`, `IN`, `BETWEEN`, `LIKE`, `IS [NOT] NULL`,
/// `EXISTS` or `IN (subquery)`.
fn is_predicate(e: &IrExpr) -> bool {
    match &e.node {
        ExprNode::Function(f) => matches!(
            Builtin::from_name(&f.name),
            Some(
                Builtin::Eq
                    | Builtin::EqNullSafe
                    | Builtin::Lt
                    | Builtin::LtEq
                    | Builtin::Gt
                    | Builtin::GtEq
                    | Builtin::And
                    | Builtin::Or
                    | Builtin::Not
                    | Builtin::In
                    | Builtin::Between
                    | Builtin::Like
                    | Builtin::ILike
                    | Builtin::RLike
                    | Builtin::IsNull
                    | Builtin::IsNotNull
            )
        ),
        ExprNode::Subquery { kind, .. } => !matches!(kind, SubqueryKind::Scalar),
        _ => false,
    }
}

fn is_path(s: &str) -> bool {
    s.contains('/') || s.contains(':')
}

fn literal(v: &Literal, sp: Span) -> R<IrLiteral> {
    Ok(match v {
        Literal::Null => IrLiteral::Null(None),
        Literal::Boolean(b) => IrLiteral::Boolean(*b),
        Literal::String(s) => IrLiteral::String(s.clone()),
        Literal::Number { text, kind } => number(text, *kind, sp)?,
        Literal::Typed { type_name, value } => {
            let t = type_name.value.to_ascii_uppercase();
            match t.as_str() {
                "DATE" if !type_name.quoted => IrLiteral::Date(date(value).ok_or_else(|| {
                    err(
                        InvalidTypedLiteral {
                            value_type: "\"DATE\"".to_owned(),
                            value: format!("'{value}'"),
                        },
                        sp,
                    )
                })?),
                "X" if !type_name.quoted => IrLiteral::Binary(hex(value).ok_or_else(|| {
                    err(
                        InvalidTypedLiteral {
                            value_type: "\"X\"".to_owned(),
                            value: format!("'{value}'"),
                        },
                        sp,
                    )
                })?),
                _ => return Err(unsupported(&format!("{t} literals"), sp)),
            }
        }
        _ => return Err(unsupported("INTERVAL literals", sp)),
    })
}

fn out_of_range(text: &str, ty: &str, min: &str, max: &str, sp: Span) -> ParseError {
    err(
        InvalidNumericLiteralRange {
            raw_stripped_qualifier: text.to_owned(),
            type_name: ty.to_owned(),
            min_value: min.to_owned(),
            max_value: max.to_owned(),
        },
        sp,
    )
}

/// A numeric literal as Spark's `AstBuilder` types it.
fn number(text: &str, kind: NumKind, sp: Span) -> R<IrLiteral> {
    let range = |ty: &str, min: String, max: String| out_of_range(text, ty, &min, &max, sp);
    Ok(match kind {
        NumKind::Int => match (text.parse::<i32>(), text.parse::<i64>()) {
            (Ok(v), _) => IrLiteral::Integer(v),
            (_, Ok(v)) => IrLiteral::Long(v),
            _ => decimal(text, sp)?,
        },
        NumKind::Long => IrLiteral::Long(
            text.parse()
                .map_err(|_| range("bigint", i64::MIN.to_string(), i64::MAX.to_string()))?,
        ),
        NumKind::Short => IrLiteral::Short(
            text.parse()
                .map_err(|_| range("smallint", i16::MIN.to_string(), i16::MAX.to_string()))?,
        ),
        NumKind::Byte => IrLiteral::Byte(
            text.parse()
                .map_err(|_| range("tinyint", i8::MIN.to_string(), i8::MAX.to_string()))?,
        ),
        NumKind::Decimal | NumKind::BigDecimal if !text.contains(['e', 'E']) => decimal(text, sp)?,
        // Bounds as Java prints them.
        NumKind::Float => match text.parse::<f32>() {
            Ok(v) if v.is_finite() => IrLiteral::Float(v),
            _ => {
                return Err(range(
                    "float",
                    "-3.4028235E38".into(),
                    "3.4028235E38".into(),
                ));
            }
        },
        NumKind::Exponent | NumKind::Double => match text.parse::<f64>() {
            Ok(v) if v.is_finite() => IrLiteral::Double(v),
            _ => {
                let max = "1.7976931348623157E308";
                return Err(range("double", format!("-{max}"), max.into()));
            }
        },
        _ => return Err(unsupported("this numeric literal", sp)),
    })
}

/// A decimal literal: precision = significant digits (at least the
/// scale), as Java's `BigDecimal` counts them.
fn decimal(text: &str, sp: Span) -> R<IrLiteral> {
    let (neg, body) = text.strip_prefix('-').map_or((false, text), |b| (true, b));
    let (int, frac) = body.split_once('.').unwrap_or((body, ""));
    let digits = format!("{int}{frac}");
    let trimmed = digits.trim_start_matches('0');
    let precision = u32::try_from(trimmed.len().max(1)).unwrap_or(u32::MAX);
    let scale = u32::try_from(frac.len()).unwrap_or(u32::MAX);
    let data_type = DecimalType::try_new(precision.max(scale), scale).map_err(|e| err(e, sp))?;
    let unscaled: i128 = if trimmed.is_empty() {
        0
    } else {
        trimmed
            .parse()
            .map_err(|_| unsupported("this decimal literal", sp))?
    };
    Ok(IrLiteral::Decimal {
        unscaled: if neg { -unscaled } else { unscaled },
        data_type,
    })
}

/// Days since 1970-01-01 of a date string as Spark's `stringToDate` reads
/// it: `[+-]y[yyy...][-m[m][-d[d]]]`, optionally followed by a space or `T`
/// and anything.
fn date(s: &str) -> Option<i32> {
    let s = s.trim();
    let s = s.split([' ', 'T']).next()?;
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let mut parts = body.split('-');
    let field = |p: Option<&str>, lo: usize, hi: usize| -> Option<i64> {
        p.filter(|p| (lo..=hi).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|p| p.parse().ok())
    };
    let year = field(parts.next(), 4, 7)?;
    let month = parts.next().map_or(Some(1), |m| field(Some(m), 1, 2))?;
    let day = parts.next().map_or(Some(1), |d| field(Some(d), 1, 2))?;
    if parts.next().is_some() || !(1..=12).contains(&month) {
        return None;
    }
    let year = if neg { -year } else { year };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day < 1 || day > days_in[usize::try_from(month - 1).ok()?] {
        return None;
    }
    // H. Hinnant's `days_from_civil`.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    i32::try_from(era * 146_097 + doe - 719_468).ok()
}

/// The bytes of a hex string (`X'1F'`); an odd length gets a leading 0.
fn hex(s: &str) -> Option<Vec<u8>> {
    let padded = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_owned()
    };
    (0..padded.len())
        .step_by(2)
        .map(|i| {
            padded
                .get(i..i + 2)
                .and_then(|b| u8::from_str_radix(b, 16).ok())
        })
        .collect()
}
