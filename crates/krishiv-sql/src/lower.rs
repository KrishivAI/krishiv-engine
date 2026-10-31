// SPDX-License-Identifier: Apache-2.0
//! Lowering: the syntax tree of a query into the logical-plan IR.
//!
//! Purely syntactic, as Spark's `AstBuilder`: no catalog, no types beyond
//! literals. Every node's origin is its span in the SQL text. Comma joins
//! fold left into inner joins without a condition; `<>` is `not(==)`; a
//! `SELECT` with aggregates but no `GROUP BY` stays a `Project` (the
//! analyzer turns it into an aggregate); `HAVING` is a `Filter` over the
//! `Aggregate`. An inline `VALUES` table lowers to the unresolved
//! [`PlanNode::InlineTable`] (row aliases stripped, columns `col1`, ...
//! unless the table alias renames them; the alias itself stays a
//! `SubqueryAlias`, as other table factors do). Constructs the IR lacks
//! (windows, generators, ...) fail with `KRISHIV_UNSUPPORTED_FEATURE`
//! naming the construct. Join lists and dotted names are lowered with
//! loops; recursion follows the tree's nesting, which the parser bounds,
//! and moves to a fresh stack every `STEP` levels. The plan is checked
//! against `MAX_DEPTH` at the end.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::{
    FailedToParseTooComplex, KrishivError, KrishivUnsupportedFeature, MissingWindowSpecification,
    WrongNumArgsWithoutSuggestion,
};
use krishiv_common::secret::OptionValue;
use krishiv_common::stack::{STEP, on_big_stack};
use krishiv_plan::{
    Builtin, CastMode, ColumnRef, Cte, Expr as IrExpr, ExprNode, FrameBound as IrFrameBound,
    FrameUnits as IrFrameUnits, FunctionCall, Grouping, JoinCondition, JoinKind as IrJoin,
    Literal as IrLiteral, LogicalPlan, Origin, PivotValue, Plan, PlanNode, ReadSource, SortOrder,
    SourceId, Star, SubqueryKind, TypedLiteralKind as IrTypedKind, UnpivotColumnSet, WindowExpr,
    WindowFrame as IrWindowFrame, WindowSpec as IrWindowSpec, WindowSpecRef,
};
use krishiv_types::DecimalType;

use crate::ast::{
    Alias, BinaryOp, Expr, ExprKind, FrameBound, FrameBoundKind, FrameUnits, FromClause,
    FromStatement, GroupBy, GroupItem, GroupItemKind, GroupWith, GroupingSet, Hint, Ident, Join,
    JoinConstraint, JoinKind, LateralView, Limit, Literal, NamedWindow, NullTreatment, NumKind,
    Pipe, PipeAggregate, PipeOp, PipeOpKind, Pivot, Query, Relation, RelationExt, Select,
    SelectItem, SetExpr, SetExprKind, SetOperator, SetQuantifier, SortItem, Span, TableAlias,
    TableFactor, TableFactorKind, Truth, UnaryOp, Unpivot, WindowRef, WindowSpec,
};
use crate::validate::literal::{self, Number, TypedKind};
use crate::{ParseError, ParserOptions, parse_named_expression, parse_query};

type R<T> = Result<T, ParseError>;

/// The most grouping sets one `GROUP BY` expands to.
const MAX_GROUPING_SETS: usize = 1 << 16;

/// Parses and lowers one query; spans index `source`.
///
/// Typed datetime literals (`TIMESTAMP '...'`, `TIMESTAMP_LTZ '...'`,
/// `TIMESTAMP_NTZ '...'`, `DATE '...'`) lower to an unresolved
/// [`ExprNode::TypedLiteral`] (the type keyword and the raw string): the
/// analyzer resolves them with the session's time zone and timestamp type.
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
    let mut l = Lowerer {
        source,
        depth: 0,
        root: query.span,
        windows: Vec::new(),
    };
    let plan = l.query(query)?;
    Plan::try_new(plan).map_err(|e| err(e, query.span))
}

/// Lowers one parsed expression into the IR `Session::sql` builds for
/// `SELECT` items, so it resolves against a DataFrame's columns in the
/// analyzer. Origins index `source`; too deep a tree fails with
/// `FAILED_TO_PARSE_TOO_COMPLEX` at the expression's span.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for constructs the IR lacks and Spark's
/// literal errors, as for [`lower`].
pub fn lower_expression(expr: &Expr, source: SourceId) -> R<IrExpr> {
    let mut l = Lowerer {
        source,
        depth: 0,
        root: expr.span,
        windows: Vec::new(),
    };
    l.expr(expr)
}

/// Lowers one parsed select item (`expr [[AS] alias]`), as [`lower`]
/// lowers the items of a `SELECT` list.
///
/// # Errors
/// As for [`lower_expression`].
pub fn lower_select_item(item: &SelectItem, source: SourceId) -> R<Arc<IrExpr>> {
    let mut l = Lowerer {
        source,
        depth: 0,
        root: item.span,
        windows: Vec::new(),
    };
    l.select_item(item)
}

/// Parses and lowers one named-expression string (`F.expr`,
/// `selectExpr` once per argument string, `filter`/`where` strings); spans
/// index `source`.
///
/// # Errors
/// A [`ParseError`]: the parser's, or a lowering error with its span.
pub fn parse_and_lower_named_expression(
    sql: &str,
    options: &ParserOptions,
    source: SourceId,
) -> R<Arc<IrExpr>> {
    lower_select_item(&parse_named_expression(sql, options)?, source)
        .map_err(|e| e.redacted_in(sql))
}

fn err(e: impl Into<KrishivError>, span: Span) -> ParseError {
    ParseError {
        error: e.into(),
        span,
    }
}

/// The clauses `ORDER BY`, `SORT BY`, `OFFSET` and `LIMIT` of a query or of
/// a pipe operator.
struct Organization<'a> {
    order_by: &'a [SortItem],
    sort_by: &'a [SortItem],
    order_span: Option<Span>,
    offset: Option<&'a Expr>,
    limit: Option<&'a Limit>,
    span: Span,
}

/// Rejects the query-organization clauses lowering does not support.
fn reject_organization(q: &Query) -> R<()> {
    for (present, what) in [
        (!q.cluster_by.is_empty(), "CLUSTER BY"),
        (!q.distribute_by.is_empty(), "DISTRIBUTE BY"),
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
    /// The whole statement: `FAILED_TO_PARSE_TOO_COMPLEX` refers to it.
    root: Span,
    /// The `WINDOW` clauses around the expression being lowered, innermost
    /// last; `None` stands for a subquery expression's boundary (Spark's
    /// `WindowsSubstitution` does not enter subquery plans).
    windows: Vec<Option<Arc<WindowDefs>>>,
}

/// A `WINDOW` clause: each name's specification, references to other
/// names followed (`validate` checked them).
type WindowDefs = Vec<(String, WindowSpec)>;

fn arc<T>(v: T) -> Arc<T> {
    Arc::new(v)
}

/// The definitions of a `WINDOW` clause, as Spark's `AstBuilder` resolves
/// them: a name defined as another name takes that one's specification.
fn window_defs(windows: &[NamedWindow]) -> WindowDefs {
    let spec = |r: &WindowRef| match r {
        WindowRef::Spec(spec) => Some((**spec).clone()),
        WindowRef::Named(_) => None,
    };
    windows
        .iter()
        .filter_map(|w| {
            let s = match &w.window {
                WindowRef::Spec(s) => Some((**s).clone()),
                WindowRef::Named(other) => windows
                    .iter()
                    .find(|d| d.name.value == other.value)
                    .and_then(|d| spec(&d.window)),
            };
            s.map(|s| (w.name.value.clone(), s))
        })
        .collect()
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
                qualifier: Vec::new(),
                arg_names: Vec::new(),
                args,
                distinct: false,
                filter: None,
                within_group: Vec::new(),
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
    fn deeper<T: Send>(&mut self, f: impl FnOnce(&mut Self) -> R<T> + Send) -> R<T> {
        self.depth += 1;
        let root = self.root;
        let r = if self.depth.is_multiple_of(STEP) {
            on_big_stack(|| f(self)).unwrap_or_else(|| Err(err(FailedToParseTooComplex {}, root)))
        } else {
            f(self)
        };
        self.depth -= 1;
        r
    }

    fn query(&mut self, q: &Query) -> R<LogicalPlan> {
        self.deeper(|l| l.query_inner(q))
    }

    /// A subquery expression's query: the `WINDOW` clauses around it do
    /// not reach into it.
    fn subquery(&mut self, q: &Query) -> R<LogicalPlan> {
        self.windows.push(None);
        let r = self.query(q);
        self.windows.pop();
        r
    }

    /// Runs `f` with the definitions of `windows` in scope.
    fn with_windows<T>(
        &mut self,
        windows: &[NamedWindow],
        f: impl FnOnce(&mut Self) -> R<T>,
    ) -> R<T> {
        if windows.is_empty() {
            return f(self);
        }
        self.windows.push(Some(Arc::new(window_defs(windows))));
        let r = f(self);
        self.windows.pop();
        r
    }

    /// `OVER name`, as Spark's `WindowsSubstitution` resolves it: the
    /// innermost `WINDOW` clause around decides (a name it lacks is
    /// `MISSING_WINDOW_SPECIFICATION`); with none around, the name stays
    /// for the analyzer, which reports it once the operator's input
    /// resolved.
    fn named_window(&mut self, name: &Ident, span: Span) -> R<WindowSpecRef> {
        let scope = self.windows.last().cloned().flatten();
        match scope {
            None => Ok(WindowSpecRef::Named(name.value.clone())),
            Some(defs) => match defs.iter().find(|(n, _)| *n == name.value) {
                Some((_, spec)) => Ok(WindowSpecRef::Spec(self.window_spec(spec)?)),
                None => Err(err(
                    MissingWindowSpecification {
                        window_name: name.value.clone(),
                        docroot: "https://spark.apache.org/docs/latest".to_owned(),
                    },
                    span,
                )),
            },
        }
    }

    /// `(PARTITION BY ... ORDER BY ... frame)`; `DISTRIBUTE BY` and
    /// `CLUSTER BY` partition, `SORT BY` orders, as in Spark.
    fn window_spec(&mut self, w: &WindowSpec) -> R<IrWindowSpec> {
        let partition_by = w
            .partition_by
            .iter()
            .map(|e| self.arg(e))
            .collect::<R<Vec<_>>>()?;
        let order_by = w
            .order_by
            .iter()
            .map(|s| self.sort_item(s))
            .collect::<R<Vec<_>>>()?;
        let frame = match &w.frame {
            None => None,
            Some(f) => Some(IrWindowFrame {
                units: match f.units {
                    FrameUnits::Rows => IrFrameUnits::Rows,
                    FrameUnits::Range => IrFrameUnits::Range,
                },
                lower: self.frame_bound(&f.start)?,
                upper: match &f.end {
                    Some(b) => self.frame_bound(b)?,
                    None => IrFrameBound::CurrentRow,
                },
            }),
        };
        Ok(IrWindowSpec {
            partition_by,
            order_by,
            frame,
        })
    }

    /// A frame bound; `n PRECEDING` is `-(n)`, as Spark's `AstBuilder`
    /// writes it.
    fn frame_bound(&mut self, b: &FrameBound) -> R<IrFrameBound> {
        Ok(match &b.kind {
            FrameBoundKind::UnboundedPreceding => IrFrameBound::UnboundedPreceding,
            FrameBoundKind::UnboundedFollowing => IrFrameBound::UnboundedFollowing,
            FrameBoundKind::CurrentRow => IrFrameBound::CurrentRow,
            FrameBoundKind::Preceding(e) => {
                let value = self.expr(e)?;
                IrFrameBound::Offset(arc(self.op(Builtin::Negative, vec![value], b.span)))
            }
            FrameBoundKind::Following(e) => IrFrameBound::Offset(self.arg(e)?),
        })
    }

    fn query_inner(&mut self, q: &Query) -> R<LogicalPlan> {
        reject_organization(q)?;
        let mut plan = self.with_windows(&q.windows, |l| {
            let plan = l.body(&q.body)?;
            l.organize(plan, q)
        })?;
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

    /// `FROM ... SELECT ... [SELECT ...]`: the shared `FROM` lowered
    /// once, its plan shared across the bodies (as Spark's
    /// `AstBuilder.visitFromStatement` shares its `from` plan), every
    /// body reading it. One body reads as itself; several combine with
    /// `UNION ALL` (the analyzer coerces the branches); none returns
    /// the relation, as Spark does for a bare `FROM t`.
    fn multi_select_from(&mut self, f: &FromStatement) -> R<LogicalPlan> {
        let from = arc(self.from(&f.from)?);
        let mut inputs = Vec::with_capacity(f.bodies.len());
        for body in &f.bodies {
            reject_organization(body)?;
            let plan = match &body.body.kind {
                SetExprKind::Select(s) => self.with_windows(&body.windows, |l| {
                    let inner = l.select_over(s, (*from).clone())?;
                    l.organize(inner, body)
                })?,
                _ => return Err(unsupported("FROM ... SELECT with a set body", f.span)),
            };
            inputs.push(arc(plan));
        }
        if inputs.is_empty() {
            return Ok(match Arc::try_unwrap(from) {
                Ok(plan) => plan,
                Err(shared) => (*shared).clone(),
            });
        }
        if inputs.len() == 1 {
            return Ok(match Arc::try_unwrap(inputs.remove(0)) {
                Ok(plan) => plan,
                Err(shared) => (*shared).clone(),
            });
        }
        Ok(self.node(
            PlanNode::Union {
                inputs,
                by_name: false,
                allow_missing_columns: false,
            },
            f.span,
        ))
    }

    /// A query body: a `SELECT`, a set operation, a table, ...
    fn body(&mut self, body: &SetExpr) -> R<LogicalPlan> {
        Ok(match &body.kind {
            SetExprKind::Pipe(p) => self.pipe(p)?,
            SetExprKind::Transform(_) => return Err(unsupported("TRANSFORM", body.span)),
            SetExprKind::Select(s) => self.select(s, s.from.as_ref())?,
            SetExprKind::Query(inner) => self.query(inner)?,
            SetExprKind::DynamicTable(_) => return Err(unsupported("IDENTIFIER()", body.span)),
            SetExprKind::Table(name) => self.node(
                PlanNode::Read(ReadSource::Table {
                    name: name.parts.iter().map(|p| p.value.clone()).collect(),
                    options: BTreeMap::new(),
                }),
                name.span,
            ),
            SetExprKind::FromStatement(f) => self.multi_select_from(f)?,
            SetExprKind::SetOp {
                op,
                quantifier,
                left,
                right,
            } => {
                let left = arc(self.deeper(|l| l.body(left))?);
                let right = arc(self.deeper(|l| l.body(right))?);
                self.set_operation(*op, *quantifier, left, right, body.span)
            }
            SetExprKind::Values(v) => {
                let plan = self.inline_table(&v.rows, v.alias.as_ref(), v.span)?;
                match &v.alias {
                    None => plan,
                    Some(a) => self.with_alias(plan, a, v.span),
                }
            }
        })
    }

    /// `left op right` as Spark's `visitSetOperationImpl` builds it:
    /// `UNION` is a `Union` (under `Distinct` unless `ALL`), the rest a
    /// `SetOp`.
    fn set_operation(
        &mut self,
        op: SetOperator,
        quantifier: Option<SetQuantifier>,
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        span: Span,
    ) -> LogicalPlan {
        let all = quantifier == Some(SetQuantifier::All);
        let op = match op {
            SetOperator::Union => {
                let union = self.node(
                    PlanNode::Union {
                        inputs: vec![left, right],
                        by_name: false,
                        allow_missing_columns: false,
                    },
                    span,
                );
                return if all {
                    union
                } else {
                    self.node(PlanNode::Distinct { input: arc(union) }, span)
                };
            }
            SetOperator::Intersect => krishiv_plan::SetOperation::Intersect,
            SetOperator::Except => krishiv_plan::SetOperation::Except,
        };
        self.node(
            PlanNode::SetOp {
                left,
                right,
                op,
                all,
            },
            span,
        )
    }

    /// `input |> operator |> ...`: each operator reads the plan of the
    /// ones before it, as Spark's `visitOperatorPipeRightSide` builds them.
    fn pipe(&mut self, p: &Pipe) -> R<LogicalPlan> {
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
                    qualifier: Vec::new(),
                    except,
                    plan_id: None,
                }),
                op.span,
            ))
        };
        Ok(match &op.kind {
            PipeOpKind::Select(s) => {
                if let Some(g) = &s.group_by {
                    return Err(unsupported("GROUP BY in a pipe SELECT", g.span));
                }
                let exprs = self.with_windows(&s.windows, |l| {
                    s.projection
                        .iter()
                        .map(|i| l.select_item(i))
                        .collect::<R<Vec<_>>>()
                })?;
                let plan = self.node(
                    PlanNode::Project {
                        input: arc(plan),
                        exprs,
                    },
                    s.span,
                );
                self.wrap_distinct_hints(plan, s.quantifier, &s.hints, s.span)?
            }
            PipeOpKind::Extend(items) => {
                let mut exprs = vec![star(self, Vec::new())];
                for item in items {
                    exprs.push(self.select_item(item)?);
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
            PipeOpKind::Organization(o) => {
                if !o.cluster_by.is_empty() {
                    return Err(unsupported("CLUSTER BY", o.span));
                }
                if !o.distribute_by.is_empty() {
                    return Err(unsupported("DISTRIBUTE BY", o.span));
                }
                self.organize_parts(
                    plan,
                    &Organization {
                        order_by: &o.order_by,
                        sort_by: &o.sort_by,
                        order_span: o.order_span,
                        offset: o.offset.as_ref(),
                        limit: o.limit.as_ref(),
                        span: o.span,
                    },
                )?
            }
            PipeOpKind::Aggregate(a) => self.pipe_aggregate(plan, a)?,
            PipeOpKind::Set(_) => return Err(unsupported("pipe SET", op.span)),
            PipeOpKind::Sample(t) => return Err(unsupported("TABLESAMPLE", t.span)),
            PipeOpKind::Pivot(_, Some(_)) | PipeOpKind::Unpivot(_, Some(_)) => {
                return Err(unsupported("PIVOT with UNPIVOT", op.span));
            }
        })
    }

    /// `AGGREGATE [aggregates] [GROUP BY keys]`: the result is the keys,
    /// then the aggregates.
    fn pipe_aggregate(&mut self, plan: LogicalPlan, a: &PipeAggregate) -> R<LogicalPlan> {
        let mut keys = Vec::new();
        let mut items = Vec::new();
        for g in a.group_by.iter().flat_map(|g| &g.items) {
            match &g.kind {
                GroupItemKind::Expr(e) => {
                    if matches!(e.kind, ExprKind::Literal(Literal::Number { .. })) {
                        return Err(unsupported(
                            "a position in a pipe AGGREGATE GROUP BY",
                            g.span,
                        ));
                    }
                    let key = arc(self.expr(e)?);
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
            items.push(self.select_item(item)?);
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

    /// `ORDER BY`, `SORT BY`, `OFFSET` and `LIMIT` of `q` over `plan`.
    fn organize(&mut self, plan: LogicalPlan, q: &Query) -> R<LogicalPlan> {
        self.organize_parts(
            plan,
            &Organization {
                order_by: &q.order_by,
                sort_by: &q.sort_by,
                order_span: q.order_span,
                offset: q.offset.as_ref(),
                limit: q.limit.as_ref(),
                span: q.span,
            },
        )
    }

    /// The clauses of [`Self::organize`] given as parts, which a pipe
    /// operator shares.
    fn organize_parts(&mut self, mut plan: LogicalPlan, q: &Organization<'_>) -> R<LogicalPlan> {
        for (items, global) in [(q.order_by, true), (q.sort_by, false)] {
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
        // `LIMIT n OFFSET m` skips `m` rows, then keeps `n`.
        if let Some(offset) = q.offset {
            let offset_expr = arc(self.expr(offset)?);
            plan = self.node(
                PlanNode::Offset {
                    input: arc(plan),
                    offset: offset_expr,
                },
                offset.span,
            );
        }
        if let Some(limit) = q.limit
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

    /// A `SELECT` with its `DISTINCT` and hints: as Spark's AstBuilder,
    /// the hints wrap the rest, the first one outermost.
    fn select(&mut self, s: &Select, from: Option<&FromClause>) -> R<LogicalPlan> {
        let plan = self.with_windows(&s.windows, |l| {
            let input = match from {
                Some(from) => l.from(from)?,
                None => l.node(PlanNode::OneRowRelation, s.span),
            };
            l.select_list(s, input)
        })?;
        self.select_wrap(s, plan)
    }

    /// A `SELECT` over an already-lowered input: the bodies of a
    /// `FROM ... SELECT` statement share one. The input lowered before
    /// the body's `WINDOW` clause is in scope, as Spark lowers the
    /// shared `FROM` once outside the bodies.
    fn select_over(&mut self, s: &Select, input: LogicalPlan) -> R<LogicalPlan> {
        let plan = self.with_windows(&s.windows, |l| l.select_list(s, input))?;
        self.select_wrap(s, plan)
    }

    /// A `SELECT`'s `DISTINCT` and hints around its lowered list.
    fn select_wrap(&mut self, s: &Select, plan: LogicalPlan) -> R<LogicalPlan> {
        self.wrap_distinct_hints(plan, s.quantifier, &s.hints, s.span)
    }

    fn wrap_distinct_hints(
        &mut self,
        mut plan: LogicalPlan,
        quantifier: Option<SetQuantifier>,
        hints: &[Hint],
        span: Span,
    ) -> R<LogicalPlan> {
        if quantifier == Some(SetQuantifier::Distinct) {
            plan = self.node(PlanNode::Distinct { input: arc(plan) }, span);
        }
        for h in hints.iter().rev() {
            let params = h
                .params
                .iter()
                .map(|e| self.arg(e))
                .collect::<R<Vec<_>>>()?;
            plan = self.node(
                PlanNode::Hint {
                    input: arc(plan),
                    name: h.name.value.clone(),
                    params,
                },
                h.span,
            );
        }
        Ok(plan)
    }

    fn select_item(&mut self, item: &SelectItem) -> R<Arc<IrExpr>> {
        let e = self.expr(&item.expr)?;
        Ok(arc(match &item.alias {
            None => e,
            Some(Alias::Name(name)) => self.ex(
                ExprNode::Alias {
                    expr: arc(e),
                    name: name.value.clone(),
                },
                item.span,
            ),
            Some(_) => return Err(unsupported("multi-name aliases", item.span)),
        }))
    }

    /// `LATERAL VIEW [OUTER] generator(arg) table AS columns` over `plan`:
    /// a [`PlanNode::Generate`] per view, in order, as Spark's `AstBuilder`
    /// stacks them. Only the one-argument generators (`explode`,
    /// `posexplode`, `inline` and their `_outer` forms) are lowered.
    fn lateral_views(&mut self, mut plan: LogicalPlan, views: &[LateralView]) -> R<LogicalPlan> {
        for v in views {
            let [name] = &v.function.parts[..] else {
                return Err(unsupported("a qualified generator name", v.span));
            };
            let lower = name.value.to_ascii_lowercase();
            let (base, outer_name) = match lower.strip_suffix("_outer") {
                Some(base) => (base, true),
                None => (lower.as_str(), false),
            };
            let kind = match base {
                "explode" => krishiv_plan::GeneratorKind::Explode,
                "posexplode" => krishiv_plan::GeneratorKind::PosExplode,
                "inline" => krishiv_plan::GeneratorKind::Inline,
                _ => {
                    return Err(unsupported(
                        &format!("the generator function {}", name.value),
                        v.span,
                    ));
                }
            };
            let [arg] = &v.args[..] else {
                return Err(err(
                    WrongNumArgsWithoutSuggestion {
                        function_name: format!("`{}`", name.value),
                        expected_num: "1".to_owned(),
                        actual_num: v.args.len().to_string(),
                        docroot: "https://spark.apache.org/docs/latest".to_owned(),
                    },
                    v.span,
                ));
            };
            let arg = arc(self.expr(arg)?);
            plan = self.node(
                PlanNode::Generate {
                    input: arc(plan),
                    kind,
                    outer: v.outer || outer_name,
                    arg,
                    output: Vec::new(),
                    names: v.columns.iter().map(|c| c.value.clone()).collect(),
                    qualifier: Some(v.table.value.clone()),
                },
                v.span,
            );
        }
        Ok(plan)
    }

    /// The generator a select item is, when its expression is a call of
    /// `explode`, `posexplode`, `inline` or one of their `_outer` forms:
    /// the kind, whether it is `_outer`, and the arguments.
    fn generator_of(
        e: &Expr,
    ) -> Option<(
        krishiv_plan::GeneratorKind,
        bool,
        &[crate::ast::FunctionArg],
        &Ident,
    )> {
        let ExprKind::Function(f) = &e.kind else {
            return None;
        };
        let [name] = &f.name.parts[..] else {
            return None;
        };
        let lower = name.value.to_ascii_lowercase();
        let (base, outer) = match lower.strip_suffix("_outer") {
            Some(base) => (base, true),
            None => (lower.as_str(), false),
        };
        let kind = match base {
            "explode" => krishiv_plan::GeneratorKind::Explode,
            "posexplode" => krishiv_plan::GeneratorKind::PosExplode,
            "inline" => krishiv_plan::GeneratorKind::Inline,
            _ => return None,
        };
        // A window, a filter or DISTINCT makes it no generator call.
        (f.over.is_none() && f.filter.is_none() && f.quantifier.is_none()).then_some((
            kind,
            outer,
            f.args.as_slice(),
            name,
        ))
    }

    /// `generator(arg)` over `plan` (a [`PlanNode::Generate`]) and the
    /// select item standing for its columns: a star over the generator's
    /// own qualifier, in the place of the call.
    fn select_generator(
        &mut self,
        plan: LogicalPlan,
        item: &SelectItem,
        index: usize,
    ) -> R<(LogicalPlan, Arc<IrExpr>)> {
        let Some((kind, outer, args, name)) = Self::generator_of(&item.expr) else {
            return Err(unsupported("a select item that is no generator", item.span));
        };
        let [arg] = args else {
            return Err(err(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{}`", name.value),
                    expected_num: "1".to_owned(),
                    actual_num: args.len().to_string(),
                    docroot: "https://spark.apache.org/docs/latest".to_owned(),
                },
                item.span,
            ));
        };
        let names = match &item.alias {
            None => Vec::new(),
            Some(Alias::Name(n)) => vec![n.value.clone()],
            Some(Alias::List(l)) => l.iter().map(|n| n.value.clone()).collect(),
        };
        // A reserved qualifier the generated columns are read through; no
        // user table can have it.
        let qualifier = format!("{}{index}", krishiv_plan::GENERATE_QUALIFIER_PREFIX);
        let arg = self.arg(&arg.expr)?;
        let plan = self.node(
            PlanNode::Generate {
                input: arc(plan),
                kind,
                outer,
                arg,
                output: Vec::new(),
                names,
                qualifier: Some(qualifier.clone()),
            },
            item.span,
        );
        let star = self.ex(
            ExprNode::Star(Star {
                qualifier: vec![qualifier],
                except: Vec::new(),
                plan_id: None,
            }),
            item.span,
        );
        Ok((plan, arc(star)))
    }

    fn select_list(&mut self, s: &Select, input: LogicalPlan) -> R<LogicalPlan> {
        let mut plan = self.lateral_views(input, &s.lateral_views)?;
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
        for (index, item) in s.projection.iter().enumerate() {
            if s.group_by.is_none()
                && s.having.is_none()
                && Self::generator_of(&item.expr).is_some()
            {
                let taken =
                    std::mem::replace(&mut plan, LogicalPlan::new(PlanNode::OneRowRelation));
                let (generated, star) = self.select_generator(taken, item, index)?;
                plan = generated;
                items.push(star);
                continue;
            }
            items.push(self.select_item(item)?);
        }
        if s.group_by.is_none() && s.having.is_none() {
            plan = self.node(
                PlanNode::Project {
                    input: arc(plan),
                    exprs: items,
                },
                s.span,
            );
        } else {
            let grouping = match &s.group_by {
                Some(g) => self.grouping(g)?,
                None => Grouping::GroupBy(Vec::new()),
            };
            plan = self.node(
                PlanNode::Aggregate {
                    input: arc(plan),
                    grouping,
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
        }
        if let Some(q) = &s.qualify {
            // `QUALIFY`, as Spark's `AstBuilder` shapes it: a filter with
            // the qualify predicate over the select list's projection (the
            // analyzer resolves it like `HAVING`, extracting window
            // functions the same way). The predicate is wrapped so the
            // analyzer tells it apart from `WHERE` and `HAVING`; the
            // wrapper is stripped during analysis and never reaches the
            // executor.
            let inner = arc(self.expr(q)?);
            let predicate = arc(self.ex(
                ExprNode::Function(FunctionCall {
                    qualifier: Vec::new(),
                    name: "qualify".to_owned(),
                    args: vec![inner],
                    arg_names: Vec::new(),
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                }),
                q.span,
            ));
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

    /// `GROUP BY`, as Spark's `AstBuilder` shapes it: plain keys, `WITH
    /// ROLLUP` / `WITH CUBE` over them, `GROUPING SETS (..)` after them (the
    /// keys only fix the grouping expressions), or items that are
    /// `ROLLUP`, `CUBE` and `GROUPING SETS` among keys, whose grouping sets
    /// are the cross product of the items' (a key is a set of itself).
    #[inline(never)]
    fn grouping(&mut self, g: &GroupBy) -> R<Grouping> {
        let mut plain = Vec::with_capacity(g.items.len());
        for item in &g.items {
            if let GroupItemKind::Expr(e) = &item.kind {
                plain.push(arc(self.expr(e)?));
            }
        }
        let all_plain = plain.len() == g.items.len();
        match (&g.with, all_plain) {
            (None, true) => {
                // `GROUP BY ALL` reads as a column named `all` here; the
                // analyzer expands it (or resolves the column). The key
                // keeps the whole clause's span, so the analyzer's
                // `UNRESOLVED_ALL_IN_GROUP_BY` cites `group by all`.
                if plain.len() == 1 && Self::is_group_by_all(&g.items[0]) {
                    let node = plain[0].node.clone();
                    plain[0] = arc(IrExpr::new(node).with_origin(self.origin(g.span)));
                }
                return Ok(Grouping::GroupBy(plain));
            }
            (Some(GroupWith::Rollup), true) => return Ok(Grouping::Rollup(plain)),
            (Some(GroupWith::Cube), true) => return Ok(Grouping::Cube(plain)),
            (Some(GroupWith::GroupingSets(sets)), true) => {
                let sets = sets.iter().map(|s| self.set(s)).collect::<R<Vec<_>>>()?;
                return Ok(Grouping::GroupingSets { exprs: plain, sets });
            }
            (Some(_), false) => {
                return Err(unsupported(
                    "ROLLUP, CUBE and GROUPING SETS items with WITH ROLLUP, WITH CUBE or GROUPING SETS",
                    g.span,
                ));
            }
            (None, false) => {}
        }
        let mut exprs = Vec::new();
        let mut product: Vec<Vec<Arc<IrExpr>>> = vec![Vec::new()];
        for item in &g.items {
            let sets = self.item_sets(item, &mut exprs)?;
            if product.len().saturating_mul(sets.len()) > MAX_GROUPING_SETS {
                return Err(unsupported(
                    &format!("more than {MAX_GROUPING_SETS} grouping sets"),
                    g.span,
                ));
            }
            product = product
                .iter()
                .flat_map(|head| {
                    sets.iter()
                        .map(move |tail| head.iter().chain(tail).cloned().collect::<Vec<_>>())
                })
                .collect();
        }
        Ok(Grouping::GroupingSets {
            exprs,
            sets: product,
        })
    }

    /// The grouping sets one `GROUP BY` item stands for; the expressions it
    /// names are appended to `written` in order.
    fn item_sets(
        &mut self,
        item: &GroupItem,
        written: &mut Vec<Arc<IrExpr>>,
    ) -> R<Vec<Vec<Arc<IrExpr>>>> {
        Ok(match &item.kind {
            GroupItemKind::Named(_) => return Err(unsupported("named grouping items", item.span)),
            GroupItemKind::Expr(e) => {
                let e = arc(self.expr(e)?);
                written.push(Arc::clone(&e));
                vec![vec![e]]
            }
            GroupItemKind::Set(s) => {
                let set = self.set(s)?;
                written.extend(set.iter().cloned());
                vec![set]
            }
            GroupItemKind::GroupingSets(items) => {
                let mut sets = Vec::new();
                let span = item.span;
                for inner in items {
                    sets.extend(self.deeper(|l| l.item_sets(inner, &mut *written))?);
                    if sets.len() > MAX_GROUPING_SETS {
                        return Err(unsupported(
                            &format!("more than {MAX_GROUPING_SETS} grouping sets"),
                            span,
                        ));
                    }
                }
                sets
            }
            GroupItemKind::Rollup(elements) | GroupItemKind::Cube(elements) => {
                let rollup = matches!(item.kind, GroupItemKind::Rollup(_));
                let mut parts = Vec::with_capacity(elements.len());
                for element in elements {
                    let part = self.set(element)?;
                    written.extend(part.iter().cloned());
                    parts.push(part);
                }
                // `ROLLUP (a, b)`: (a, b), (a), (); `CUBE (a, b)`: every
                // subset of the elements.
                let n = parts.len();
                if !rollup && n >= usize::BITS as usize - 1
                    || !rollup && (1_usize << n) > MAX_GROUPING_SETS
                {
                    return Err(unsupported(
                        &format!("more than {MAX_GROUPING_SETS} grouping sets"),
                        item.span,
                    ));
                }
                let chosen: Vec<Vec<usize>> = if rollup {
                    (0..=n).rev().map(|k| (0..k).collect()).collect()
                } else {
                    (0..1_usize << n)
                        .rev()
                        .map(|m| (0..n).filter(|i| m >> (n - 1 - i) & 1 == 1).collect())
                        .collect()
                };
                chosen
                    .into_iter()
                    .map(|c| {
                        c.into_iter()
                            .flat_map(|i| parts[i].iter().cloned())
                            .collect()
                    })
                    .collect()
            }
        })
    }

    /// The expressions of one grouping set.
    fn set(&mut self, s: &GroupingSet) -> R<Vec<Arc<IrExpr>>> {
        s.exprs.iter().map(|e| Ok(arc(self.expr(e)?))).collect()
    }

    /// Whether a `GROUP BY` item is a bare `ALL` (`GROUP BY ALL`), written as
    /// a column of that name by the parser (see `column_named_all`).
    fn is_group_by_all(item: &GroupItem) -> bool {
        let GroupItemKind::Expr(e) = &item.kind else {
            return false;
        };
        let ExprKind::Column(parts) = &e.kind else {
            return false;
        };
        matches!(&parts[..], [p] if p.value.eq_ignore_ascii_case("all"))
    }

    fn from(&mut self, f: &FromClause) -> R<LogicalPlan> {
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
        let plan = plan.ok_or_else(|| unsupported("an empty FROM clause", f.span))?;
        let mut plan = self.lateral_views(plan, &f.lateral_views)?;
        if f.pivot.is_some() && f.unpivot.is_some() {
            return Err(err(
                krishiv_common::error::NotAllowedInFromUnpivotWithPivot {},
                f.span,
            ));
        }
        if let Some(p) = &f.pivot {
            plan = self.pivot(plan, p)?;
        }
        if let Some(u) = &f.unpivot {
            plan = self.unpivot(plan, u)?;
        }
        Ok(plan)
    }

    fn relation(&mut self, r: &Relation) -> R<LogicalPlan> {
        if r.lateral {
            return Err(unsupported("LATERAL subqueries", r.span));
        }
        // Spark rejects a relation carrying both clauses
        // (`NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT`, also raised for the
        // pipe-operator chain in the goldens): a `PIVOT ... UNPIVOT ...`
        // chain on one relation lowers to no plan.
        if r.extensions
            .iter()
            .any(|e| matches!(e, RelationExt::Pivot(_)))
            && r.extensions
                .iter()
                .any(|e| matches!(e, RelationExt::Unpivot(_)))
        {
            return Err(err(
                krishiv_common::error::NotAllowedInFromUnpivotWithPivot {},
                r.span,
            ));
        }
        let mut plan = self.factor(&r.primary)?;
        for ext in &r.extensions {
            match ext {
                RelationExt::Join(j) => {
                    plan = self.join(plan, j)?;
                }
                RelationExt::Pivot(p) => {
                    plan = self.pivot(plan, p)?;
                }
                RelationExt::Unpivot(u) => {
                    plan = self.unpivot(plan, u)?;
                }
            }
        }
        Ok(plan)
    }

    /// `[INNER] JOIN right constraint` onto `plan`.
    fn join(&mut self, plan: LogicalPlan, j: &Join) -> R<LogicalPlan> {
        if j.lateral {
            return Err(unsupported("LATERAL joins", j.span));
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
        let condition = if j.natural {
            // `NATURAL JOIN` has no condition in the syntax tree (the
            // parser leaves it empty); the analyzer computes the
            // common columns.
            JoinCondition::Natural
        } else {
            match &j.constraint {
                JoinConstraint::None => JoinCondition::None,
                JoinConstraint::On(e) => JoinCondition::On(arc(self.expr(e)?)),
                JoinConstraint::Using(cols) => {
                    JoinCondition::Using(cols.iter().map(|c| c.value.clone()).collect())
                }
                JoinConstraint::NearestBy(n) => {
                    return Err(unsupported("NEAREST BY joins", n.span));
                }
            }
        };
        Ok(self.node(
            PlanNode::Join {
                left: arc(plan),
                right: arc(right),
                kind,
                condition,
            },
            j.span,
        ))
    }

    /// `PIVOT (aggregates FOR columns IN (values)) [alias]` over `plan`,
    /// as Spark's parser builds it: the aggregates and values keep their
    /// aliases, several pivot columns stand for a struct value, and the
    /// alias becomes a `SubqueryAlias`.
    fn pivot(&mut self, plan: LogicalPlan, p: &Pivot) -> R<LogicalPlan> {
        let mut aggregates = Vec::with_capacity(p.aggregates.len());
        for item in &p.aggregates {
            aggregates.push(self.select_item(item)?);
        }
        let mut pivot_columns = Vec::with_capacity(p.columns.len());
        for col in &p.columns {
            pivot_columns.push(arc(self.column(vec![col.value.clone()], col.span)));
        }
        let mut values = Vec::with_capacity(p.values.len());
        for item in &p.values {
            let alias = match &item.alias {
                None => None,
                Some(Alias::Name(name)) => Some(name.value.clone()),
                Some(_) => return Err(unsupported("multi-name aliases", item.span)),
            };
            // `(a, b)` is one struct value of two parts.
            let exprs = match &item.expr.kind {
                ExprKind::Row(items) => items
                    .iter()
                    .map(|i| self.expr(&i.expr).map(arc))
                    .collect::<R<Vec<_>>>()?,
                _ => vec![arc(self.expr(&item.expr)?)],
            };
            values.push(PivotValue { exprs, alias });
        }
        let plan = self.node(
            PlanNode::Pivot {
                input: arc(plan),
                pivot_columns,
                values,
                aggregates,
            },
            p.span,
        );
        Ok(match &p.alias {
            None => plan,
            Some(alias) => self.node(
                PlanNode::SubqueryAlias {
                    input: arc(plan),
                    alias: alias.value.clone(),
                    columns: Vec::new(),
                },
                p.span,
            ),
        })
    }

    /// `UNPIVOT [INCLUDE | EXCLUDE NULLS] (values FOR name IN (columns))
    /// [alias]` over `plan`, as Spark's parser builds it: the alias
    /// becomes a `SubqueryAlias`.
    fn unpivot(&mut self, plan: LogicalPlan, u: &Unpivot) -> R<LogicalPlan> {
        let mut column_sets = Vec::with_capacity(u.columns.len());
        for set in &u.columns {
            let mut columns = Vec::with_capacity(set.columns.len());
            for col in &set.columns {
                columns.push(arc(self.column(
                    col.parts.iter().map(|p| p.value.clone()).collect(),
                    col.span,
                )));
            }
            column_sets.push(UnpivotColumnSet {
                columns,
                alias: set.alias.as_ref().map(|a| a.value.clone()),
            });
        }
        let plan = self.node(
            PlanNode::Unpivot {
                input: arc(plan),
                value_columns: u.value_columns.iter().map(|c| c.value.clone()).collect(),
                name_column: u.name_column.value.clone(),
                column_sets,
                include_nulls: u.include_nulls.unwrap_or(false),
            },
            u.span,
        );
        Ok(match &u.alias {
            None => plan,
            Some(alias) => self.node(
                PlanNode::SubqueryAlias {
                    input: arc(plan),
                    alias: alias.value.clone(),
                    columns: Vec::new(),
                },
                u.span,
            ),
        })
    }

    fn factor(&mut self, tf: &TableFactor) -> R<LogicalPlan> {
        if let Some(sample) = &tf.sample {
            return Err(unsupported("TABLESAMPLE", sample.span));
        }
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
            TableFactorKind::Nested(r) => self.deeper(|l| l.relation(r))?,
            TableFactorKind::DynamicTable { name, .. } => {
                return Err(unsupported("IDENTIFIER()", name.span));
            }
            TableFactorKind::Values(rows) => self.inline_table(rows, tf.alias.as_ref(), tf.span)?,
            TableFactorKind::TableFunction {
                name,
                dynamic_name,
                args,
            } => {
                if dynamic_name.is_some() {
                    return Err(unsupported("table-valued functions", tf.span));
                }
                if name.parts.len() > 3 {
                    return Err(unsupported("table-valued functions", tf.span));
                }
                for a in args {
                    if a.name.is_some() {
                        return Err(unsupported("table-valued functions", a.span));
                    }
                    if matches!(a.expr.kind, crate::ast::ExprKind::TableArgument(_)) {
                        return Err(unsupported("table-valued functions", a.span));
                    }
                }
                let lowered = args
                    .iter()
                    .map(|a| self.arg(&a.expr))
                    .collect::<R<Vec<_>>>()?;
                // A generator as a table: `FROM explode(array(1, 2))`
                // generates from one empty row.
                let lname = name.parts[0].value.to_ascii_lowercase();
                let (base, outer) = match lname.strip_suffix("_outer") {
                    Some(base) => (base, true),
                    None => (lname.as_str(), false),
                };
                let kind = match base {
                    "explode" => Some(krishiv_plan::GeneratorKind::Explode),
                    "posexplode" => Some(krishiv_plan::GeneratorKind::PosExplode),
                    "inline" => Some(krishiv_plan::GeneratorKind::Inline),
                    _ => None,
                };
                if let (Some(kind), 1) = (kind, name.parts.len()) {
                    let [arg] = &lowered[..] else {
                        return Err(err(
                            WrongNumArgsWithoutSuggestion {
                                function_name: format!("`{}`", name.parts[0].value),
                                expected_num: "1".to_owned(),
                                actual_num: lowered.len().to_string(),
                                docroot: "https://spark.apache.org/docs/latest".to_owned(),
                            },
                            tf.span,
                        ));
                    };
                    let one_row = self.node(PlanNode::OneRowRelation, tf.span);
                    self.node(
                        PlanNode::Generate {
                            input: arc(one_row),
                            kind,
                            outer,
                            arg: Arc::clone(arg),
                            output: Vec::new(),
                            names: Vec::new(),
                            qualifier: None,
                        },
                        tf.span,
                    )
                } else {
                    self.node(
                        PlanNode::TableFunction {
                            // A qualified name (`ns.f`) is kept as one string, its
                            // parts joined by `TABLE_FUNCTION_NAME_SEP`.
                            name: name
                                .parts
                                .iter()
                                .map(|p| p.value.as_str())
                                .collect::<Vec<_>>()
                                .join(&krishiv_plan::TABLE_FUNCTION_NAME_SEP.to_string()),
                            args: lowered,
                        },
                        tf.span,
                    )
                }
            }
        };
        Ok(match &tf.alias {
            None => plan,
            Some(a) => self.with_alias(plan, a, tf.span),
        })
    }

    /// `plan` under the `SubqueryAlias` of `a` (the one wrapper both the
    /// query-body and table-factor paths use). `AS (columns)` names no
    /// table: a generated name, as for no alias at all.
    fn with_alias(&mut self, plan: LogicalPlan, a: &TableAlias, span: Span) -> LogicalPlan {
        self.node(
            PlanNode::SubqueryAlias {
                input: arc(plan),
                alias: a.name.as_ref().map_or_else(
                    || "__auto_generated_subquery_name".to_owned(),
                    |n| n.value.clone(),
                ),
                columns: a.columns.iter().map(|c| c.value.clone()).collect(),
            },
            span,
        )
    }

    /// `VALUES rows [AS alias(columns)]` at `span`: an unresolved
    /// [`PlanNode::InlineTable`] (Spark's `UnresolvedInlineTable`); the
    /// caller wraps it in a `SubqueryAlias` for the alias. A row is either
    /// a row constructor (one cell per item; item aliases are ignored, the
    /// schema comes from `columns`) or a single expression (one column, as
    /// Spark's grammar has it). Rows of different widths are rejected by
    /// [`crate::validate`] before lowering.
    fn inline_table(
        &mut self,
        rows: &[Expr],
        alias: Option<&TableAlias>,
        span: Span,
    ) -> R<LogicalPlan> {
        let mut lowered = Vec::with_capacity(rows.len());
        for row in rows {
            let cells = match &row.kind {
                ExprKind::Row(items) => items
                    .iter()
                    .map(|i| self.arg(&i.expr))
                    .collect::<R<Vec<_>>>()?,
                _ => vec![self.arg(row)?],
            };
            lowered.push(cells);
        }
        let width = lowered.first().map_or(0, Vec::len);
        let columns = match alias.map(|a| &a.columns) {
            Some(cols) if !cols.is_empty() => cols.iter().map(|c| c.value.clone()).collect(),
            _ => (1..=width).map(|i| format!("col{i}")).collect(),
        };
        Ok(self.node(
            PlanNode::InlineTable {
                rows: lowered,
                columns,
            },
            span,
        ))
    }

    fn expr(&mut self, e: &Expr) -> R<IrExpr> {
        self.deeper(|l| l.expr_inner(e))
    }

    fn arg(&mut self, e: &Expr) -> R<Arc<IrExpr>> {
        self.expr(e).map(arc)
    }

    #[expect(clippy::too_many_lines, reason = "one arm per expression kind")]
    fn expr_inner(&mut self, e: &Expr) -> R<IrExpr> {
        let sp = e.span;
        Ok(match &e.kind {
            ExprKind::Literal(v) => self.ex(literal(v, sp)?, sp),
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
                let sp = base.span;
                if let ExprKind::Column(parts) = &base.kind {
                    let mut name: Vec<String> = parts.iter().map(|p| p.value.clone()).collect();
                    name.extend(fields.into_iter().rev());
                    self.column(name, sp)
                } else {
                    // `f(x).y`: field access on an expression.
                    let mut base = self.expr(base)?;
                    for name in fields.into_iter().rev() {
                        base = self.ex(
                            ExprNode::Field {
                                base: arc(base),
                                name,
                            },
                            sp,
                        );
                    }
                    base
                }
            }
            ExprKind::Wildcard { qualifier, except } => self.ex(
                ExprNode::Star(Star {
                    qualifier: qualifier.iter().map(|p| p.value.clone()).collect(),
                    except: except
                        .iter()
                        .map(|n| n.parts.iter().map(|p| p.value.clone()).collect())
                        .collect(),
                    plan_id: None,
                }),
                sp,
            ),
            ExprKind::Unary(op, inner) => {
                let name = match op {
                    // Unary `-` keeps `"-"` so the analyzer sees it apart
                    // from the function `negative(x)` (Spark's `UnaryMinus`
                    // with and without its `FUNC_ALIAS`).
                    UnaryOp::Minus => "-",
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
                    BinaryOp::ShiftLeft => "<<",
                    BinaryOp::ShiftRight => ">>",
                    BinaryOp::ShiftRightUnsigned => ">>>",
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
                let plan = arc(self.subquery(query)?);
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
                let plan = arc(self.subquery(q)?);
                self.ex(
                    ExprNode::Subquery {
                        kind: SubqueryKind::Exists,
                        plan,
                    },
                    sp,
                )
            }
            ExprKind::Subquery(q) => {
                let plan = arc(self.subquery(q)?);
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
                if let Some(dynamic) = &f.dynamic_name {
                    return Err(unsupported("IDENTIFIER() function names", dynamic.span));
                }
                let (qualifier, name) = match &f.name.parts[..] {
                    [name] => (Vec::new(), name),
                    [qualifier @ .., name] => {
                        (qualifier.iter().map(|p| p.value.clone()).collect(), name)
                    }
                    [] => return Err(unsupported("a function name", f.name.span)),
                };
                // `WITHIN GROUP (ORDER BY ...)` rides along for the analyzer:
                // the ordered-set aggregates lower it into their arguments,
                // any other function keeps Spark's
                // `FUNCTION_WITH_UNSUPPORTED_SYNTAX` error (raised during
                // analysis, which quotes the call).
                if f.null_treatment.is_some() && !takes_null_treatment(&name.value) {
                    return Err(unsupported("IGNORE NULLS and RESPECT NULLS", sp));
                }
                let mut args = Vec::new();
                let mut arg_names = Vec::new();
                let mut any_named = false;
                for a in &f.args {
                    if let Some(n) = &a.name {
                        any_named = true;
                        arg_names.push(Some(n.value.clone()));
                    } else {
                        arg_names.push(None);
                    }
                    args.push(self.arg(&a.expr)?);
                }
                if !any_named {
                    arg_names.clear();
                }
                // Spark's builder of `lag` and `lead`: `input` NULL, `offset`
                // 1 and `default` NULL when not given.
                let lname = name.value.to_ascii_lowercase();
                if matches!(lname.as_str(), "lag" | "lead") && args.len() < 3 {
                    let fill = [
                        IrLiteral::Null(None),
                        IrLiteral::Integer(1),
                        IrLiteral::Null(None),
                    ];
                    for v in fill.into_iter().skip(args.len()) {
                        args.push(arc(self.ex(ExprNode::Literal(v), sp)));
                        if !arg_names.is_empty() {
                            arg_names.push(None);
                        }
                    }
                }
                // `listagg(expr)` is `listagg(expr, NULL)` (and `string_agg`
                // likewise): Spark's constructor writes the default
                // delimiter, so names and errors show it.
                if matches!(lname.as_str(), "listagg" | "string_agg") && args.len() == 1 {
                    args.push(arc(self.ex(ExprNode::Literal(IrLiteral::Null(None)), sp)));
                    if !arg_names.is_empty() {
                        arg_names.push(None);
                    }
                }
                // `first(x) IGNORE NULLS` is `first(x, true)` and `RESPECT
                // NULLS` is `first(x, false)`, whatever flag was written
                // (`lag`/`lead`'s fourth argument, `nth_value`'s third); a
                // flag that is not a literal stays for the analyzer to
                // refuse, as Spark does.
                let flag_at = match lname.as_str() {
                    "lag" | "lead" => 3,
                    "nth_value" => 2,
                    _ => 1,
                };
                if let Some(t) = f.null_treatment {
                    let flag = t == NullTreatment::Ignore;
                    match args.get(flag_at).map(|a| &a.node) {
                        None | Some(ExprNode::Literal(IrLiteral::Boolean(_))) => {
                            let flag =
                                arc(self.ex(ExprNode::Literal(IrLiteral::Boolean(flag)), sp));
                            // Only the flag slot is replaced: surplus (or
                            // missing) arguments stay for the analyzer's
                            // arity error.
                            if args.len() == flag_at {
                                args.push(flag);
                                if !arg_names.is_empty() {
                                    arg_names.push(None);
                                }
                            } else if args.len() > flag_at {
                                args[flag_at] = flag;
                            }
                        }
                        Some(_) => {}
                    }
                }
                let filter = f.filter.as_ref().map(|x| self.arg(x)).transpose()?;
                // `DATEADD(unit, n, t)` and `DATEDIFF(unit, a, b)` with a unit
                // keyword are `TIMESTAMPADD` and `TIMESTAMPDIFF` (Spark's
                // AstBuilder), and are named so in output columns; any other
                // three-argument call stays the day-count function.
                let unit_form = qualifier.is_empty()
                    && f.args.len() == 3
                    && matches!(f.args[0].expr.kind, ExprKind::Keyword(_));
                let function_name = match (unit_form, lname.as_str()) {
                    (true, "timestampadd" | "dateadd" | "date_add") => "timestampadd".to_owned(),
                    (true, "timestampdiff" | "datediff" | "date_diff") => {
                        "timestampdiff".to_owned()
                    }
                    _ => name.value.clone(),
                };
                let within_group = f
                    .within_group
                    .iter()
                    .map(|s| self.sort_item(s))
                    .collect::<R<Vec<_>>>()?;
                let call = self.ex(
                    ExprNode::Function(FunctionCall {
                        name: function_name,
                        qualifier,
                        arg_names,
                        args,
                        distinct: f.quantifier == Some(SetQuantifier::Distinct),
                        filter,
                        within_group,
                    }),
                    sp,
                );
                match &f.over {
                    None => call,
                    Some(over) => {
                        let window = match over {
                            WindowRef::Spec(spec) => WindowSpecRef::Spec(self.window_spec(spec)?),
                            WindowRef::Named(name) => self.named_window(name, sp)?,
                        };
                        self.ex(
                            ExprNode::Window(WindowExpr {
                                function: arc(call),
                                window,
                            }),
                            sp,
                        )
                    }
                }
            }
            ExprKind::LikeQuantified {
                expr,
                quantifier,
                patterns,
                negated,
                case_insensitive,
            } => {
                use crate::ast::LikeQuantifier::{All, Any, Some};
                let b = match (quantifier, negated, case_insensitive) {
                    (Any | Some, false, false) => Builtin::LikeAny,
                    (All, false, false) => Builtin::LikeAll,
                    (Any | Some, true, false) => Builtin::NotLikeAny,
                    (All, true, false) => Builtin::NotLikeAll,
                    (Any | Some, false, true) => Builtin::ILikeAny,
                    (All, false, true) => Builtin::ILikeAll,
                    (Any | Some, true, true) => Builtin::NotILikeAny,
                    (All, true, true) => Builtin::NotILikeAll,
                };
                let mut args = vec![self.arg(expr)?];
                for p in patterns {
                    args.push(self.arg(p)?);
                }
                self.call(b.name(), args, sp)
            }
            ExprKind::Nested(inner) => self.expr(inner)?,
            ExprKind::Parameter(_) | ExprKind::NamedParameter(_) => {
                return Err(unsupported("parameter markers", sp));
            }
            ExprKind::Subscript(base, index) => {
                let args = vec![self.arg(base)?, self.arg(index)?];
                self.call("__subscript", args, sp)
            }
            ExprKind::Collate(..) => return Err(unsupported("COLLATE", sp)),
            ExprKind::VariantGet(..) => return Err(unsupported("variant paths", sp)),
            ExprKind::TableArgument(_) => return Err(unsupported("TABLE arguments", sp)),
            ExprKind::Lambda { params, body } => {
                let params = params.iter().map(|p| p.value.clone()).collect();
                let body = self.arg(body)?;
                self.ex(
                    IrExpr::lambda(params, Arc::unwrap_or_clone(body))
                        .node
                        .clone(),
                    sp,
                )
            }
            ExprKind::Row(_) => {
                return Err(unsupported("row and struct constructors", sp));
            }
            ExprKind::Struct(items) => self.struct_ctor(items, sp)?,
            // Every variant is lowered above; the wildcard stays because
            // `ExprKind` is non-exhaustive.
            #[expect(unreachable_patterns, reason = "every ExprKind variant is lowered")]
            _ => return Err(unsupported("this expression form", sp)),
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

    /// `STRUCT(items)`: `struct(...)` with the items as written (an aliased
    /// item stays an alias); the analyzer names the fields (an alias, an
    /// unaliased column or field its own name, anything else `colN`) and
    /// rewrites a call with any named child to `named_struct(name, ...)`,
    /// keeping `struct(...)` as the shown form. A `*` item passes through
    /// for the analyzer to expand, and a multi-name alias is not a field
    /// name.
    fn struct_ctor(&mut self, items: &[SelectItem], sp: Span) -> R<IrExpr> {
        let mut args = Vec::with_capacity(items.len());
        for item in items.iter() {
            let star = matches!(item.expr.kind, ExprKind::Wildcard { .. });
            if star && item.alias.is_some() {
                return Err(unsupported("row and struct constructors", sp));
            }
            let arg = self.arg(&item.expr)?;
            args.push(match &item.alias {
                None => arg,
                Some(Alias::Name(id)) => arc(self.ex(
                    ExprNode::Alias {
                        expr: arg,
                        name: id.value.clone(),
                    },
                    sp,
                )),
                Some(Alias::List(_)) => {
                    return Err(unsupported("row and struct constructors", sp));
                }
            });
        }
        Ok(self.call("struct", args, sp))
    }
}

/// The aggregates that take `IGNORE NULLS` / `RESPECT NULLS` (as their
/// `ignoreNulls` argument).
fn takes_null_treatment(name: &str) -> bool {
    [
        "first",
        "last",
        "first_value",
        "last_value",
        "any_value",
        "lag",
        "lead",
        "nth_value",
    ]
    .iter()
    .any(|n| name.eq_ignore_ascii_case(n))
}

/// A path written as a quoted name part (`parquet.`s3://b/k``).
/// Whether `e` is one of Spark's `Predicate` expressions: a comparison,
/// `AND` / `OR` / `NOT`, `IN`, `BETWEEN`, `LIKE`, `IS [NOT] NULL`,
/// `EXISTS` or `IN (subquery)`.
fn is_predicate(e: &IrExpr) -> bool {
    match &e.node {
        ExprNode::Function(f) => matches!(
            f.unqualified().and_then(Builtin::from_name),
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
                    | Builtin::LikeAny
                    | Builtin::LikeAll
                    | Builtin::NotLikeAny
                    | Builtin::NotLikeAll
                    | Builtin::ILikeAny
                    | Builtin::ILikeAll
                    | Builtin::NotILikeAny
                    | Builtin::NotILikeAll
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

fn literal(v: &Literal, sp: Span) -> R<ExprNode> {
    Ok(match v {
        Literal::Null => ExprNode::Literal(IrLiteral::Null(None)),
        Literal::Boolean(b) => ExprNode::Literal(IrLiteral::Boolean(*b)),
        Literal::String(s) => ExprNode::Literal(IrLiteral::String(s.clone())),
        Literal::Number { text, kind } => ExprNode::Literal(number(text, *kind, sp)?),
        Literal::Typed { type_name, value } => {
            let kind = TypedKind::of(&type_name.value, type_name.quoted).map_err(|e| err(e, sp))?;
            match kind {
                TypedKind::Date => ExprNode::TypedLiteral {
                    kind: IrTypedKind::Date,
                    value: value.clone(),
                },
                TypedKind::Timestamp => ExprNode::TypedLiteral {
                    kind: IrTypedKind::Timestamp,
                    value: value.clone(),
                },
                TypedKind::TimestampNtz => ExprNode::TypedLiteral {
                    kind: IrTypedKind::TimestampNtz,
                    value: value.clone(),
                },
                TypedKind::TimestampLtz => ExprNode::TypedLiteral {
                    kind: IrTypedKind::TimestampLtz,
                    value: value.clone(),
                },
                TypedKind::Binary => ExprNode::Literal(IrLiteral::Binary(
                    literal::hex(value).ok_or_else(|| err(kind.invalid(value), sp))?,
                )),
                _ => {
                    let t = type_name.value.to_ascii_uppercase();
                    return Err(unsupported(&format!("{t} literals"), sp));
                }
            }
        }
        _ => return Err(unsupported("INTERVAL literals", sp)),
    })
}

/// A numeric literal as Spark's `AstBuilder` types it.
fn number(text: &str, kind: NumKind, sp: Span) -> R<IrLiteral> {
    Ok(match literal::number(text, kind).map_err(|e| err(e, sp))? {
        Number::Byte(v) => IrLiteral::Byte(v),
        Number::Short(v) => IrLiteral::Short(v),
        Number::Integer(v) => IrLiteral::Integer(v),
        Number::Long(v) => IrLiteral::Long(v),
        Number::Float(v) => IrLiteral::Float(v),
        Number::Double(v) => IrLiteral::Double(v),
        Number::Decimal {
            unscaled,
            precision,
            scale,
        } => IrLiteral::Decimal {
            unscaled,
            data_type: DecimalType::try_new(precision, scale).map_err(|e| err(e, sp))?,
        },
    })
}
