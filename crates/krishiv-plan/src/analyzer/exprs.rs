// SPDX-License-Identifier: Apache-2.0
//! Expression binding: names, stars, functions, casts, subqueries, output
//! names (Spark's `toPrettySQL`) and semantic keys.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{
    AmbiguousColumnReference, AmbiguousLateralColumnAlias, AmbiguousReference,
    CannotResolveDataframeColumn, CannotResolveStarExpand, DatatypeMismatchBinaryOpDiffTypes,
    DatatypeMismatchBinaryOpWrongType, DatatypeMismatchCastWithConfSuggestion,
    DatatypeMismatchCastWithFuncSuggestion, DatatypeMismatchCastWithoutSuggestion,
    DatatypeMismatchDataDiffTypes, DatatypeMismatchInSubqueryDataTypeMismatch,
    DatatypeMismatchInSubqueryLengthMismatch, DatatypeMismatchUnexpectedInputType,
    InvalidAggregateFilterNotBoolean, InvalidExtractBaseFieldType,
    InvalidSubqueryExpressionScalarSubqueryReturnMoreThanOneOutputColumn,
    InvalidUsageOfStarOrRegex, KrishivError, NestedAggregateFunction,
    UnresolvedColumnWithSuggestion, UnresolvedColumnWithoutSuggestion, UnresolvedRoutine,
    WrongNumArgsWithoutSuggestion,
};
use krishiv_types::DataType;
use twox_hash::XxHash3_128;

use super::analytics::GroupingFn;
use super::bind::{addr, same, unsupported};
use super::catalog::quoted;
use super::coerce::{self, Arg, BadCast, Mismatch, ordinal, quoted_type};
use super::fold;
use super::resolve::{Attr, Env, Rel, Resolver};
use super::{AnalysisError, ArgType, CallTypes, Constant, FunctionEntry, FunctionLookup, Inputs};
use crate::{
    AttrId, BoundColumn, Builtin, CastMode, ColumnRef, Expr, ExprNode, FunctionCall, FunctionId,
    FunctionKind, Literal, LogicalPlan, Origin, ResolvedExpr, ResolvedFunction, Star, SubqueryKind,
};

/// Where an operator's expressions resolve.
#[derive(Clone, Copy)]
pub(super) struct Cx<'s> {
    /// The operator's input attributes.
    pub(super) scope: &'s [Attr],
    /// The operator's children, whose `plan_id` carriers `df["a"]` uses.
    pub(super) rels: &'s [&'s Rel],
    /// Aliases defined earlier in the same select list (lateral column
    /// aliases), searched after the input; `None` where Spark has none.
    pub(super) lateral: Option<&'s [Attr]>,
    /// Select-list aliases a `GROUP BY` may name: (name, unresolved expr).
    pub(super) aliases: &'s [(String, Arc<Expr>)],
    /// Attributes searched after `scope`: an aggregate's grouping
    /// attributes (`HAVING`), a projection's input (`ORDER BY`).
    pub(super) extra: Option<&'s [Attr]>,
    /// Over an aggregate: its input, where aggregate-function arguments
    /// resolve.
    pub(super) agg_scope: Option<&'s [Attr]>,
    /// Whether `grouping(..)` and `grouping_id(..)` may appear: in the
    /// select list, `HAVING` and `ORDER BY` of an aggregate over grouping
    /// sets (see `analytics`).
    pub(super) grouping: bool,
    /// A name `scope` holds several times is looked up further rather
    /// than ambiguous (Spark's `ORDER BY` over a select list).
    pub(super) lenient: bool,
    /// Whether names may resolve in enclosing queries.
    pub(super) outer: bool,
    pub(super) env: &'s Arc<Env>,
    /// The operator, as Spark names it in errors.
    pub(super) op: &'static str,
}

/// A bound expression.
#[derive(Clone)]
pub(super) struct Bound {
    pub(super) expr: Arc<Expr>,
    pub(super) ty: ArgType,
    /// The first aggregate call inside (outside subqueries).
    pub(super) agg: Option<Origin>,
    /// The outermost query level referenced, for outer references.
    pub(super) outer: Option<usize>,
    /// The lateral column aliases referenced (their attribute ids).
    pub(super) lateral: Vec<AttrId>,
    /// The expression as Spark shows it in names and errors, when that is
    /// not `expr`: before implicit casts and rewrites.
    pub(super) shown: Option<Arc<Expr>>,
}

/// The function Spark rewrites `extract(field FROM x)` (or `date_part`)
/// to, when `field` is a literal naming one.
fn extract_function(function: &str, call: &Expr) -> Option<&'static str> {
    use krishiv_kernels::datetime::DatePart as P;
    if !matches!(function, "extract" | "date_part") {
        return None;
    }
    let ExprNode::Function(f) = &call.node else {
        return None;
    };
    let ExprNode::Literal(Literal::String(field)) = &f.args.first()?.node else {
        return None;
    };
    Some(match P::parse(field)? {
        P::Year => "year",
        P::YearOfWeek => "yearofweek",
        P::Quarter => "quarter",
        P::Month => "month",
        P::Week => "weekofyear",
        P::Day => "dayofmonth",
        P::DayOfWeek => "dayofweek",
        P::DayOfWeekIso => "weekday",
        P::DayOfYear => "dayofyear",
        P::Hour => "hour",
        P::Minute => "minute",
        P::Second => "secondwithfraction",
    })
}

/// Functions Spark resolves from a bare name no column matches, with the
/// function each calls (`USER` and `SESSION_USER` are `current_user`).
const LITERAL_FUNCTIONS: &[(&str, &str)] = &[
    ("current_date", "current_date"),
    ("current_timestamp", "current_timestamp"),
    ("current_user", "current_user"),
    ("user", "current_user"),
    ("session_user", "current_user"),
];

impl Bound {
    /// The expression as Spark shows it.
    pub(super) fn shown(&self) -> Arc<Expr> {
        self.shown.clone().unwrap_or_else(|| Arc::clone(&self.expr))
    }
}

/// `e` with its direct children replaced, in order, by `kids`.
fn with_children(e: &Expr, kids: Vec<Arc<Expr>>) -> Arc<Expr> {
    let mut x = e.clone();
    let mut kids = kids.into_iter();
    x.for_each_child_mut(&mut |slot| {
        if let Some(k) = kids.next() {
            *slot = k;
        }
    });
    Arc::new(x)
}

/// The shown form of a call whose bound form is `expr`, the arguments
/// `cast` of `kids`: the call over what `kids` show, when that differs.
fn shown_call(expr: &Expr, kids: &[Bound], cast: &[Bound]) -> Option<Arc<Expr>> {
    let differs = kids.iter().any(|k| k.shown.is_some())
        || kids
            .iter()
            .zip(cast)
            .any(|(k, c)| !Arc::ptr_eq(&k.expr, &c.expr));
    differs.then(|| with_children(expr, kids.iter().map(Bound::shown).collect()))
}

/// Spark's error for a cast of `arg` to `to` it refuses.
fn bad_cast(
    bad: BadCast,
    origin: Origin,
    arg: &Bound,
    to: &DataType,
    mode: CastMode,
) -> AnalysisError {
    let shown = ExprNode::Cast {
        expr: arg.shown(),
        to: to.clone(),
        mode,
    };
    let sql_expr = format!("\"{}\"", pretty(&Expr::new(shown)));
    let src_type = quoted_type(arg.ty.data_type.as_ref());
    let target_type = quoted_type(Some(to));
    let error: KrishivError = match bad {
        BadCast::Without => DatatypeMismatchCastWithoutSuggestion {
            sql_expr,
            src_type,
            target_type,
        }
        .into(),
        BadCast::Func(f) => DatatypeMismatchCastWithFuncSuggestion {
            sql_expr,
            src_type,
            target_type,
            function_names: f.to_owned(),
        }
        .into(),
        BadCast::Conf => DatatypeMismatchCastWithConfSuggestion {
            sql_expr,
            src_type,
            target_type,
            config: "\"spark.sql.ansi.enabled\"".to_owned(),
            config_val: "'false'".to_owned(),
        }
        .into(),
    };
    AnalysisError::new(error, origin)
}

/// A bound argument as coercion sees it.
fn arg_of(b: &Bound) -> Arg<'_> {
    let literal = match &b.expr.node {
        ExprNode::Literal(Literal::Byte(v)) => Some(i128::from(*v)),
        ExprNode::Literal(Literal::Short(v)) => Some(i128::from(*v)),
        ExprNode::Literal(Literal::Integer(v)) => Some(i128::from(*v)),
        ExprNode::Literal(Literal::Long(v)) => Some(i128::from(*v)),
        _ => None,
    };
    Arg {
        ty: b.ty.data_type.as_ref(),
        nullable: b.ty.nullable,
        literal,
    }
}

/// Spark's documentation root, as its errors cite it.
pub(super) const SPARK_DOC_ROOT: &str = "https://spark.apache.org/docs/latest";

fn column(c: &BoundColumn) -> ArgType {
    arg_type(c.data_type.clone(), c.nullable)
}

/// The type of a value as typing sees it: `VOID` is an untyped `NULL`
/// (`data_type: None`) everywhere but in plans and schemas.
fn arg_type(data_type: DataType, nullable: bool) -> ArgType {
    ArgType {
        data_type: (data_type != DataType::Null).then_some(data_type),
        nullable,
    }
}

pub(super) fn literal_type(v: &Literal) -> Option<DataType> {
    Some(match v {
        Literal::Null(t) => return t.clone().filter(|t| *t != DataType::Null),
        Literal::Boolean(_) => DataType::Boolean,
        Literal::Byte(_) => DataType::Byte,
        Literal::Short(_) => DataType::Short,
        Literal::Integer(_) => DataType::Integer,
        Literal::Long(_) => DataType::Long,
        Literal::Float(_) => DataType::Float,
        Literal::Double(_) => DataType::Double,
        Literal::Decimal { data_type, .. } => DataType::Decimal(*data_type),
        Literal::String(_) => DataType::String,
        Literal::Binary(_) => DataType::Binary,
        Literal::Date(_) => DataType::Date,
        Literal::Timestamp(_) => DataType::Timestamp,
        Literal::TimestampNtz(_) => DataType::TimestampNtz,
        Literal::YearMonthInterval { .. } | Literal::DayTimeInterval { .. } => return None,
    })
}

enum Step {
    Enter(Arc<Expr>, bool),
    Exit(Arc<Expr>, bool),
    /// The first expression's result is the second's (a rewritten call).
    Link(usize, Arc<Expr>, bool),
}

impl Resolver<'_> {
    /// Binds `e` in `cx`, post-order with an explicit stack; a node shared
    /// within the expression is bound once.
    pub(super) fn expr(&mut self, e: &Arc<Expr>, cx: &Cx<'_>) -> Result<Bound, AnalysisError> {
        let mut done: HashMap<(usize, bool), Bound> = HashMap::new();
        // Rewritten calls stay alive for the pass: `done` is keyed by
        // address.
        let mut rewritten_calls: Vec<Arc<Expr>> = Vec::new();
        let mut stack = vec![Step::Enter(Arc::clone(e), false)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(x, in_agg) => {
                    if done.contains_key(&(addr(&*x), in_agg)) {
                        continue;
                    }
                    if let Some(rewritten) = self.star_args(&x, cx)? {
                        rewritten_calls.push(Arc::clone(&rewritten));
                        stack.push(Step::Link(addr(&*x), Arc::clone(&rewritten), in_agg));
                        stack.push(Step::Enter(rewritten, in_agg));
                        continue;
                    }
                    let inner = in_agg || self.is_aggregate(&x);
                    stack.push(Step::Exit(Arc::clone(&x), in_agg));
                    let mut kids = Vec::new();
                    x.for_each_child(&mut |c| kids.push(Arc::clone(c)));
                    stack.extend(kids.into_iter().rev().map(|c| Step::Enter(c, inner)));
                }
                Step::Link(from, to, in_agg) => {
                    if let Some(b) = done.get(&(addr(&*to), in_agg)).cloned() {
                        done.insert((from, in_agg), b);
                    }
                }
                Step::Exit(x, in_agg) => {
                    let inner = in_agg || self.is_aggregate(&x);
                    let mut kids = Vec::new();
                    x.for_each_child(&mut |c| {
                        if let Some(b) = done.get(&(addr(&**c), inner)) {
                            kids.push(b.clone());
                        }
                    });
                    let b = self.expr_node(&x, &kids, in_agg, cx)?;
                    done.insert((addr(&*x), in_agg), b);
                }
            }
        }
        done.remove(&(addr(&**e), false))
            .ok_or_else(|| unsupported("an expression shape", e.origin))
    }

    fn is_aggregate(&self, e: &Expr) -> bool {
        match &e.node {
            ExprNode::Function(f) => matches!(
                self.ctx.functions.lookup(std::slice::from_ref(&f.name), self.ctx.profile),
                FunctionLookup::Found(entry) if entry.kind == FunctionKind::Aggregate
            ),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => f.kind == FunctionKind::Aggregate,
            _ => false,
        }
    }

    /// A call with `*` arguments, rewritten: `count(*)` is `count(1)`, any
    /// other call takes the input columns.
    fn star_args(&mut self, e: &Expr, cx: &Cx<'_>) -> Result<Option<Arc<Expr>>, AnalysisError> {
        let ExprNode::Function(f) = &e.node else {
            return Ok(None);
        };
        if !f.args.iter().any(|a| matches!(a.node, ExprNode::Star(_))) {
            return Ok(None);
        }
        let count_star = f.name.eq_ignore_ascii_case("count")
            && matches!(&f.args[..], [a] if matches!(&a.node, ExprNode::Star(s) if s.qualifier.is_empty() && s.plan_id.is_none()));
        let mut args = Vec::new();
        for a in &f.args {
            match &a.node {
                _ if count_star => args.push(Arc::new(
                    Expr::literal(Literal::Integer(1)).with_origin(a.origin),
                )),
                ExprNode::Star(s) => {
                    for attr in self.expand(s, cx, a.origin)? {
                        args.push(Arc::new(
                            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col)))
                                .with_origin(a.origin),
                        ));
                    }
                }
                _ => args.push(Arc::clone(a)),
            }
        }
        let call = FunctionCall { args, ..f.clone() };
        Ok(Some(Arc::new(
            Expr::new(ExprNode::Function(call)).with_origin(e.origin),
        )))
    }

    /// The attributes `*`, `t.*` or `df["*"]` stands for.
    pub(super) fn expand(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<Attr>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let out: Vec<Attr> = match s.plan_id {
            Some(id) => carriers(cx, id)
                .into_iter()
                .filter(|a| cx.scope.iter().any(|b| b.col.id == a.col.id))
                .collect(),
            None => cx
                .scope
                .iter()
                .filter(|a| {
                    let q = s.qualifier.len();
                    a.qual.len() >= q
                        && a.qual[a.qual.len() - q..]
                            .iter()
                            .zip(&s.qualifier)
                            .all(|(x, y)| same(x, y, cs))
                })
                .cloned()
                .collect(),
        };
        if out.is_empty() && !s.qualifier.is_empty() {
            return Err(AnalysisError::new(
                CannotResolveStarExpand {
                    target_string: quoted(&s.qualifier),
                    columns: names(cx.scope, false),
                },
                origin,
            ));
        }
        out.iter().for_each(|a| self.used(a));
        Ok(out)
    }

    /// Records a column of a securable as referenced.
    pub(super) fn used(&mut self, a: &Attr) {
        if let Some((i, name)) = &a.source
            && let Some(set) = self.columns.get_mut(*i)
        {
            set.insert(name.to_string());
        }
    }

    fn expr_node(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        in_agg: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let outer = kids.iter().filter_map(|k| k.outer).min();
        let agg = kids.iter().find_map(|k| k.agg);
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        let keep = |ty| Bound {
            expr: Arc::clone(e),
            ty,
            agg,
            outer,
            lateral: lateral.clone(),
            shown: None,
        };
        let rebuilt = |node: ExprNode| Arc::new(Expr::new(node).with_origin(e.origin));
        let args: Vec<Arc<Expr>> = kids.iter().map(|k| Arc::clone(&k.expr)).collect();
        let mut b = match &e.node {
            ExprNode::Column(c) => self.column(c, e.origin, in_agg, cx)?,
            ExprNode::Literal(
                Literal::YearMonthInterval { .. } | Literal::DayTimeInterval { .. },
            ) => return Err(unsupported("interval values", e.origin)),
            ExprNode::Literal(v) => keep(ArgType {
                nullable: matches!(v, Literal::Null(_)),
                data_type: literal_type(v),
            }),
            ExprNode::Star(_) => {
                return Err(AnalysisError::new(
                    InvalidUsageOfStarOrRegex {
                        elem: "'*'".to_owned(),
                        pretty_name: cx.op.to_owned(),
                    },
                    e.origin,
                ));
            }
            ExprNode::Function(f) => {
                let n = f.args.len();
                if let Some(filter) = kids.get(n)
                    && filter.ty.data_type != Some(DataType::Boolean)
                {
                    return Err(AnalysisError::new(
                        InvalidAggregateFilterNotBoolean {
                            filter_expr: format!("\"{}\"", pretty(&filter.shown())),
                        },
                        e.origin,
                    ));
                }
                if let Some(which) = GroupingFn::from_name(&f.name) {
                    return self.grouping_call(which, e, &kids[..n], cx);
                }
                if let Some(op) = Builtin::from_name(&f.name) {
                    return self.operator_call(op, e.origin, &kids[..n]);
                }
                let entry = match self
                    .ctx
                    .functions
                    .lookup(std::slice::from_ref(&f.name), self.ctx.profile)
                {
                    FunctionLookup::Found(entry) => entry,
                    FunctionLookup::KnownUnsupported => {
                        return Err(unsupported(format!("function {}", f.name), e.origin));
                    }
                    FunctionLookup::NotFound => {
                        return Err(AnalysisError::new(
                            UnresolvedRoutine {
                                routine_name: quoted(std::slice::from_ref(&f.name)),
                                search_path: self.b.search_path.clone(),
                            },
                            e.origin,
                        ));
                    }
                };
                if entry.kind == FunctionKind::Aggregate
                    && let Some(inner) = kids.iter().find_map(|k| k.agg)
                {
                    return Err(AnalysisError::new(NestedAggregateFunction {}, inner));
                }
                let (mut call_kids, ty) = self.typed_call(&entry, e, &kids[..n])?;
                let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
                call_kids.extend(kids.get(n).cloned());
                let call = ResolvedFunction {
                    id: entry.id,
                    kind: entry.kind,
                    args: call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect(),
                    distinct: f.distinct,
                    filter: call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                    data_type,
                    nullable: ty.nullable,
                    deterministic: entry.deterministic,
                };
                let expr = rebuilt(ExprNode::Resolved(ResolvedExpr::Function(call)));
                Bound {
                    agg: if entry.kind == FunctionKind::Aggregate {
                        Some(e.origin)
                    } else {
                        agg
                    },
                    shown: shown_call(&expr, kids, &call_kids),
                    expr,
                    ..keep(ty)
                }
            }
            ExprNode::Cast { to, mode, .. } => {
                let mode = match mode {
                    CastMode::Default if self.ctx.conf.ansi => CastMode::Ansi,
                    CastMode::Default => CastMode::Legacy,
                    m => *m,
                };
                let child = &kids[0].ty;
                if let Err(bad) = coerce::can_cast(child.data_type.as_ref(), to, mode) {
                    return Err(bad_cast(bad, e.origin, &kids[0], to, mode));
                }
                let nullable = child.nullable
                    || child.data_type.as_ref().is_some_and(|from| {
                        if mode == CastMode::Try {
                            coerce::try_nullable(from, to)
                        } else {
                            coerce::force_nullable(from, to)
                        }
                    });
                Bound {
                    expr: rebuilt(ExprNode::Cast {
                        expr: Arc::clone(&args[0]),
                        to: to.clone(),
                        mode,
                    }),
                    ..keep(arg_type(to.clone(), nullable))
                }
            }
            ExprNode::Alias { name, .. } => {
                let output = self.attr(name, &kids[0].ty);
                Bound {
                    expr: rebuilt(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(&args[0]),
                        output: output.clone(),
                    })),
                    ..keep(column(&output))
                }
            }
            ExprNode::Subquery { kind, plan } => {
                let env = self.sub_env(cx);
                let rel = self.rel(plan, &env)?;
                let (data_type, nullable) = match kind {
                    SubqueryKind::Scalar => {
                        if rel.out.len() != 1 {
                            return Err(AnalysisError::new(
                                InvalidSubqueryExpressionScalarSubqueryReturnMoreThanOneOutputColumn {
                                    number: rel.out.len().to_string(),
                                },
                                e.origin,
                            ));
                        }
                        (
                            arg_type(rel.out[0].col.data_type.clone(), true).data_type,
                            true,
                        )
                    }
                    SubqueryKind::Exists => (Some(DataType::Boolean), false),
                    SubqueryKind::In(_) => (Some(DataType::Boolean), true),
                };
                let (kind, plan, values) = match kind {
                    SubqueryKind::In(_) => {
                        let (values, plan) = self.in_subquery(e, kids, &rel, cx)?;
                        let exprs = values.iter().map(|v| Arc::clone(&v.expr)).collect();
                        (SubqueryKind::In(exprs), plan, values)
                    }
                    k => (k.clone(), Arc::clone(&rel.plan), Vec::new()),
                };
                let level = cx.env.outer.len();
                let expr = rebuilt(ExprNode::Subquery { kind, plan });
                Bound {
                    shown: shown_call(&expr, kids, &values),
                    expr,
                    outer: [outer, rel.outer.filter(|&l| l < level)]
                        .into_iter()
                        .flatten()
                        .min(),
                    ..keep(ArgType {
                        data_type,
                        nullable,
                    })
                }
            }
            ExprNode::Resolved(r) => keep(match r {
                ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) => column(c),
                ResolvedExpr::Alias { output, .. } => column(output),
                ResolvedExpr::Function(f) => arg_type(f.data_type.clone(), f.nullable),
            }),
        };
        if b.shown.is_none() && kids.iter().any(|k| k.shown.is_some()) {
            b.shown = Some(with_children(
                &b.expr,
                kids.iter().map(Bound::shown).collect(),
            ));
        }
        Ok(b)
    }

    /// `b` implicitly cast to `to`, if given (named and shown as `b`).
    /// A registry call's arguments after their implicit casts, and its
    /// result type: Spark's arity check, then the function's input types
    /// and result rule. Out of line: it keeps the recursive `expr` frame
    /// small (deep plans resolve on a small stack).
    #[inline(never)]
    fn typed_call(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        kids: &[Bound],
    ) -> Result<(Vec<Bound>, ArgType), AnalysisError> {
        let n = kids.len();
        if !entry.arity.accepts(n) {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{}`", entry.id.name()),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: n.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        let constants: Vec<Option<Constant>> = kids
            .iter()
            .map(|k| fold::constant(&k.expr, self.ctx.conf.ansi))
            .collect();
        // Rendered only if a rule's error quotes an argument.
        let sql = |i: usize| kids.get(i).map(|k| pretty(&k.shown()));
        let given: Vec<ArgType> = kids.iter().map(|k| k.ty.clone()).collect();
        let ansi = self.ctx.conf.ansi;
        let loss = self.ctx.conf.decimal_precision_loss;
        let call = CallTypes::new(&given, &constants)
            .with_sql(&sql)
            .with_ansi(ansi)
            .with_decimal_loss(loss);
        let coerced: Vec<Arg<'_>> = kids.iter().map(arg_of).collect();
        let casts = coerce::inputs(&entry.inputs, &coerced, &call)
            .map_err(|m| self.mismatch(m, e, kids, entry.id.name(), e.origin))?;
        // `try_add('x', 1)` casts its arguments in TRY mode: NULL, not an error.
        let try_casts = matches!(entry.inputs, Inputs::TryOperator(_));
        let call_kids: Vec<Bound> = kids
            .iter()
            .zip(casts)
            .map(|(k, to)| self.implicit_cast_as(k, to, try_casts))
            .collect();
        let types: Vec<ArgType> = call_kids.iter().map(|k| k.ty.clone()).collect();
        let call = CallTypes::new(&types, &constants)
            .with_sql(&sql)
            .with_ansi(ansi)
            .with_decimal_loss(loss);
        let ty = (entry.result)(&call).map_err(|err| AnalysisError::new(err, e.origin))?;
        Ok((call_kids, ty))
    }

    pub(super) fn implicit_cast(&self, b: &Bound, to: Option<DataType>) -> Bound {
        self.implicit_cast_as(b, to, false)
    }

    /// [`Self::implicit_cast`], as a `TRY_CAST` when `try_cast`.
    fn implicit_cast_as(&self, b: &Bound, to: Option<DataType>, try_cast: bool) -> Bound {
        let Some(to) = to else {
            return b.clone();
        };
        let mode = match (try_cast, self.ctx.conf.ansi) {
            (true, _) => CastMode::Try,
            (false, true) => CastMode::Ansi,
            (false, false) => CastMode::Legacy,
        };
        let may_null = |f: &DataType| {
            if try_cast {
                coerce::try_nullable(f, &to)
            } else {
                coerce::force_nullable(f, &to)
            }
        };
        let cast = ExprNode::Cast {
            expr: Arc::clone(&b.expr),
            to: to.clone(),
            mode,
        };
        Bound {
            expr: Arc::new(Expr::new(cast).with_origin(b.expr.origin)),
            ty: ArgType {
                nullable: b.ty.nullable || b.ty.data_type.as_ref().is_some_and(may_null),
                data_type: Some(to.clone()),
            },
            shown: Some(b.shown()),
            ..b.clone()
        }
    }

    /// A call of operator `op` on `kids`, typed as Spark types it.
    /// `BETWEEN` is Spark's replacement, `>=` and `<=` joined by `AND`,
    /// shown as written.
    pub(super) fn operator_call(
        &mut self,
        op: Builtin,
        origin: Origin,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        if let (Builtin::Between, [v, lo, hi]) = (op, kids) {
            // The value is shared, as Spark's errors show it.
            let shared = Expr::function("commonexpressionref", Vec::new());
            let v = Bound {
                shown: Some(Arc::new(shared)),
                ..v.clone()
            };
            let ge = self.operator_call(Builtin::GtEq, origin, &[v.clone(), lo.clone()])?;
            let le = self.operator_call(Builtin::LtEq, origin, &[v, hi.clone()])?;
            let both = self.operator_call(Builtin::And, origin, &[ge, le])?;
            let shown = ResolvedFunction {
                id: FunctionId::Builtin(op),
                kind: FunctionKind::Scalar,
                args: kids.iter().map(Bound::shown).collect(),
                distinct: false,
                filter: None,
                data_type: DataType::Boolean,
                nullable: both.ty.nullable,
                deterministic: true,
            };
            let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(shown)));
            return Ok(Bound {
                shown: Some(Arc::new(shown.with_origin(origin))),
                ..both
            });
        }
        let args: Vec<Arg<'_>> = kids.iter().map(arg_of).collect();
        let conf = self.ctx.conf;
        let typed = coerce::operator(op, &args, conf.ansi, conf.decimal_precision_loss);
        let call = |args: Vec<Arc<Expr>>, data_type: DataType, nullable: bool| {
            let f = ResolvedFunction {
                id: FunctionId::Builtin(op),
                kind: FunctionKind::Scalar,
                args,
                distinct: false,
                filter: None,
                data_type,
                nullable,
                deterministic: true,
            };
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(origin))
        };
        let typed = typed.map_err(|m| {
            let shown = call(
                kids.iter().map(Bound::shown).collect(),
                DataType::Boolean,
                true,
            );
            self.mismatch(m, &shown, kids, op.name(), origin)
        })?;
        let cast: Vec<Bound> = kids
            .iter()
            .zip(typed.casts)
            .map(|(k, to)| self.implicit_cast(k, to))
            .collect();
        let expr = call(
            cast.iter().map(|k| Arc::clone(&k.expr)).collect(),
            typed.data_type.clone(),
            typed.nullable,
        );
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            shown: shown_call(&expr, kids, &cast),
            expr,
            ty: arg_type(typed.data_type, typed.nullable),
            agg: kids.iter().find_map(|k| k.agg),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
        })
    }

    /// `(values) IN (subquery)`: the values and the subquery's columns, each
    /// pair cast to the type it meets at (the subquery under a projection
    /// that casts). Returns the values and the subquery plan.
    fn in_subquery(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        rel: &Rel,
        cx: &Cx<'_>,
    ) -> Result<(Vec<Bound>, Arc<LogicalPlan>), AnalysisError> {
        let shown = with_children(e, kids.iter().map(Bound::shown).collect());
        let sql_expr = format!("\"{}\"", pretty(&shown));
        let texts = |v: Vec<String>| v.join(", ");
        if kids.len() != rel.out.len() {
            return Err(AnalysisError::new(
                DatatypeMismatchInSubqueryLengthMismatch {
                    sql_expr,
                    left_length: kids.len().to_string(),
                    left_columns: texts(
                        kids.iter()
                            .map(|k| format!("\"{}\"", pretty(&k.shown())))
                            .collect(),
                    ),
                    right_length: rel.out.len().to_string(),
                    right_columns: texts(
                        rel.out
                            .iter()
                            .map(|a| format!("\"{}\"", a.col.name))
                            .collect(),
                    ),
                },
                e.origin,
            ));
        }
        let qualified = |qual: &[String], name: &str, ty: Option<&DataType>| {
            let mut parts = qual.to_vec();
            parts.push(name.to_owned());
            let ty = ty.map_or_else(|| "void".to_owned(), DataType::simple_string);
            format!("{}:{ty}", parts.join("."))
        };
        let mut targets = Vec::new();
        let mut bad = Vec::new();
        for (k, a) in kids.iter().zip(rel.out.iter()) {
            let out = Arg {
                ty: Some(&a.col.data_type),
                nullable: a.col.nullable,
                literal: None,
            };
            match coerce::common_type(&arg_of(k), &out) {
                Some(t) => targets.push(t),
                None => {
                    let left = format!(
                        "{}:{}",
                        pretty(&super::ops::qualified(&k.shown(), cx.scope)),
                        k.ty.data_type
                            .as_ref()
                            .map_or_else(|| "void".to_owned(), DataType::simple_string)
                    );
                    let right = qualified(&a.qual, &a.col.name, Some(&a.col.data_type));
                    bad.push(format!("{left}, {right}"));
                }
            }
        }
        if !bad.is_empty() {
            return Err(AnalysisError::new(
                DatatypeMismatchInSubqueryDataTypeMismatch {
                    sql_expr,
                    mismatched_columns: format!("({})", bad.join(", ")),
                    left_type: texts(
                        kids.iter()
                            .map(|k| quoted_type(k.ty.data_type.as_ref()))
                            .collect(),
                    ),
                    right_type: texts(
                        rel.out
                            .iter()
                            .map(|a| quoted_type(Some(&a.col.data_type)))
                            .collect(),
                    ),
                },
                e.origin,
            ));
        }
        let values = kids
            .iter()
            .zip(&targets)
            .map(|(k, t)| {
                self.implicit_cast(k, coerce::cast_to(k.ty.data_type.as_ref(), t.as_ref()))
            })
            .collect();
        let needs = rel
            .out
            .iter()
            .zip(&targets)
            .any(|(a, t)| t.as_ref().is_some_and(|t| *t != a.col.data_type));
        if !needs {
            return Ok((values, Arc::clone(&rel.plan)));
        }
        let mut exprs = Vec::new();
        let mut out = Vec::new();
        for (a, t) in rel.out.iter().zip(targets) {
            let column = Bound {
                expr: super::ops::col_expr(&a.col, Origin::UNKNOWN),
                ty: column(&a.col),
                agg: None,
                outer: None,
                lateral: Vec::new(),
                shown: None,
            };
            let to = coerce::cast_to(Some(&a.col.data_type), t.as_ref());
            if to.is_none() {
                exprs.push(column.expr);
                out.push(a.clone());
                continue;
            }
            let cast = self.implicit_cast(&column, to);
            let output = self.attr(&a.col.name, &cast.ty);
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: cast.expr,
                    output: output.clone(),
                },
            ))));
            out.push(Attr {
                col: output,
                ..a.clone()
            });
        }
        Ok((
            values,
            self.projection(Origin::UNKNOWN, rel, exprs, out).plan,
        ))
    }

    /// The analysis error of a call that does not type.
    fn mismatch(
        &self,
        m: Mismatch,
        call: &Arc<Expr>,
        kids: &[Bound],
        function: &str,
        origin: Origin,
    ) -> AnalysisError {
        let sql_expr = format!("\"{}\"", pretty(call));
        let error: KrishivError = match m {
            Mismatch::DiffTypes(left, right) => DatatypeMismatchBinaryOpDiffTypes {
                sql_expr,
                left: quoted_type(left.as_ref()),
                right: quoted_type(right.as_ref()),
            }
            .into(),
            Mismatch::WrongType(actual, input) => DatatypeMismatchBinaryOpWrongType {
                sql_expr,
                input_type: input.to_owned(),
                actual_data_type: quoted_type(actual.as_ref()),
            }
            .into(),
            Mismatch::Unexpected {
                arg,
                param,
                ty,
                required,
            } => {
                // `extract(year FROM x)` and `date_part('year', x)` are
                // `year(x)` to Spark, which names that in the error.
                let (sql_expr, param) = match extract_function(function, call) {
                    Some(name) if param == 1 => (
                        format!(
                            "\"{name}({})\"",
                            kids.get(1).map(|k| pretty(&k.shown())).unwrap_or_default()
                        ),
                        0,
                    ),
                    _ => (sql_expr, param),
                };
                DatatypeMismatchUnexpectedInputType {
                    sql_expr,
                    param_index: ordinal(param),
                    required_type: required,
                    input_sql: format!(
                        "\"{}\"",
                        kids.get(arg)
                            .map(|k| pretty(&k.shown()))
                            .unwrap_or_default()
                    ),
                    input_type: quoted_type(ty.as_ref()),
                }
                .into()
            }
            Mismatch::DataDiff(types, list) => {
                let types: Vec<String> = types.iter().map(|t| quoted_type(t.as_ref())).collect();
                let data_type = if list {
                    format!("[{}]", types.join(", "))
                } else {
                    format!("({})", types.join(" or "))
                };
                let function_name = match function {
                    "when" => "casewhen",
                    other => other,
                };
                DatatypeMismatchDataDiffTypes {
                    sql_expr,
                    function_name: quoted(&[function_name.to_owned()]),
                    data_type,
                }
                .into()
            }
            Mismatch::Unsupported(what) => return unsupported(what, origin),
        };
        AnalysisError::new(error, origin)
    }

    /// A new attribute named `name` of type `ty`.
    pub(super) fn attr(&mut self, name: &str, ty: &ArgType) -> BoundColumn {
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        BoundColumn {
            id: self.fresh(),
            name: name.to_owned(),
            data_type,
            nullable: ty.nullable,
        }
    }

    /// A column reference, in Spark's order: the input, lateral column
    /// aliases (a select list), select-list aliases (a `GROUP BY`), the
    /// enclosing query (only the immediately enclosing one).
    fn column(
        &mut self,
        c: &ColumnRef,
        origin: Origin,
        in_agg: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let bound = |expr: ExprNode, col: &BoundColumn, outer| Bound {
            expr: Arc::new(Expr::new(expr).with_origin(origin)),
            ty: column(col),
            agg: None,
            outer,
            lateral: Vec::new(),
            shown: None,
        };
        let renamed = |a: &Attr, parts: &[String], used: usize| {
            // A reference keeps the spelling it was written with.
            let mut col = a.col.clone();
            col.name.clone_from(&parts[used - 1]);
            col
        };
        if let Some(id) = c.plan_id {
            let a = self.plan_column(id, &c.name, cx, origin)?;
            self.used(&a);
            let col = renamed(&a, &c.name, c.name.len());
            return Ok(bound(
                ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                &col,
                None,
            ));
        }
        let (scope, extra) = match cx.agg_scope {
            Some(input) if in_agg => (input, None),
            _ => (cx.scope, cx.extra),
        };
        for (i, level) in [Some(scope), extra].into_iter().enumerate() {
            let Some(level) = level else { continue };
            let lenient = cx.lenient && i == 0;
            if let Some((a, used)) = self.find(level, &c.name, origin, lenient)? {
                self.used(&a);
                let col = renamed(&a, &c.name, used);
                return Ok(bound(
                    ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                    &col,
                    None,
                ));
            }
        }
        // A name no input column has may be a function written without
        // parentheses (what the parser leaves a column unless ANSI reserved
        // keywords are on). Spark 4.2 tries it before lateral aliases,
        // GROUP BY aliases and outer references.
        if let [name] = &c.name[..]
            && let Some(function) = LITERAL_FUNCTIONS
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(name))
                .map(|(_, f)| *f)
            // A release without the function reads the name as a column.
            && matches!(
                self.ctx.functions.lookup(&[function.to_owned()], self.ctx.profile),
                FunctionLookup::Found(_) | FunctionLookup::KnownUnsupported
            )
        {
            let call = Arc::new(
                Expr::new(ExprNode::Function(FunctionCall {
                    name: function.to_owned(),
                    args: Vec::new(),
                    distinct: false,
                    filter: None,
                }))
                .with_origin(origin),
            );
            return self.expr(&call, cx);
        }
        if let ([name], Some(lateral)) = (&c.name[..], cx.lateral) {
            let cs = self.ctx.conf.case_sensitive;
            let hits: Vec<&Attr> = lateral
                .iter()
                .filter(|a| same(&a.col.name, name, cs))
                .collect();
            match hits[..] {
                [] => {}
                [a] => {
                    let mut col = a.col.clone();
                    col.name.clone_from(name);
                    let mut b = bound(
                        ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                        &col,
                        None,
                    );
                    b.lateral.push(col.id);
                    return Ok(b);
                }
                _ => {
                    return Err(AnalysisError::new(
                        AmbiguousLateralColumnAlias {
                            name: quoted(std::slice::from_ref(name)),
                            n: hits.len().to_string(),
                        },
                        Origin::UNKNOWN,
                    ));
                }
            }
        }
        if let [name] = &c.name[..]
            && let Some((_, alias)) = cx
                .aliases
                .iter()
                .find(|(n, _)| same(n, name, self.ctx.conf.case_sensitive))
        {
            let inner = Cx {
                aliases: &[],
                ..*cx
            };
            return self.expr(alias, &inner);
        }
        // Only the immediately enclosing query, as Spark 4.2 resolves.
        let levels = cx.env.outer.iter().enumerate().rev().take(1);
        for (level, outer) in levels.filter(|_| cx.outer) {
            if let Some((a, used)) = self.find(&outer.scope, &c.name, origin, false)? {
                self.used(&a);
                let col = renamed(&a, &c.name, used);
                return Ok(bound(
                    ExprNode::Resolved(ResolvedExpr::OuterColumn(col.clone())),
                    &col,
                    Some(level),
                ));
            }
        }
        Err(unresolved(&c.name, cx.scope, origin))
    }

    /// `df["a"]`: the column `name` of the relations carrying `id` in the
    /// operator's subtree.
    fn plan_column(
        &self,
        id: crate::PlanId,
        name: &[String],
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Attr, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let pick = |carried: Vec<Attr>, scope: &[Attr]| {
            let mut hits: Vec<Attr> = Vec::new();
            for a in carried {
                let visible = scope.iter().any(|s| s.col.id == a.col.id);
                if visible
                    && name.len() == 1
                    && same(&a.col.name, &name[0], cs)
                    && !hits.iter().any(|h| h.col.id == a.col.id)
                {
                    hits.push(a);
                }
            }
            hits
        };
        let sql_name = format!("\"{}\"", name.join("."));
        // The error names the DataFrame the reference belongs to.
        let origin = match origin.plan_id {
            Some(_) => origin,
            None => origin.with_plan_id(id),
        };
        // Only the operator's own subtree: Spark 4.2 does not resolve a
        // DataFrame reference against an enclosing query (golden case
        // `outer`).
        let mut hits = pick(carriers(cx, id), cx.scope);
        match hits.len() {
            1 => Ok(hits.remove(0)),
            0 => Err(AnalysisError::new(
                CannotResolveDataframeColumn { name: sql_name },
                origin,
            )),
            _ => Err(AnalysisError::new(
                AmbiguousColumnReference { name: sql_name },
                origin,
            )),
        }
    }

    /// The attribute `parts` names in `scope` and how many parts it used:
    /// with a qualifier of 3, 2 or 1 parts (a suffix of the attribute's),
    /// then by name alone; the first form with any match decides. Several
    /// matches are `AMBIGUOUS_REFERENCE`, or no match when `lenient`.
    fn find(
        &self,
        scope: &[Attr],
        parts: &[String],
        origin: Origin,
        lenient: bool,
    ) -> Result<Option<(Attr, usize)>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        for q in (0..parts.len().min(4)).rev() {
            let mut hits: Vec<&Attr> = Vec::new();
            for a in scope {
                let qualified = a.qual.len() >= q
                    && a.qual[a.qual.len() - q..]
                        .iter()
                        .zip(&parts[..q])
                        .all(|(x, y)| same(x, y, cs));
                if qualified
                    && same(&a.col.name, &parts[q], cs)
                    && !hits.iter().any(|h| h.col.id == a.col.id)
                {
                    hits.push(a);
                }
            }
            match hits[..] {
                [] => continue,
                [a] if parts.len() > q + 1 => {
                    // Struct fields: no nested types exist yet, so any
                    // field access is on a non-struct column.
                    return Err(AnalysisError::new(
                        InvalidExtractBaseFieldType {
                            base: format!("\"{}\"", parts[q]),
                            other: format!("\"{}\"", a.col.data_type.sql()),
                        },
                        origin,
                    ));
                }
                [a] => return Ok(Some((a.clone(), q + 1))),
                _ if lenient => return Ok(None),
                _ => {
                    let mut names: Vec<String> = hits
                        .iter()
                        .map(|a| {
                            let mut p = a.qual.to_vec();
                            p.push(a.col.name.clone());
                            quoted(&p)
                        })
                        .collect();
                    names.sort();
                    return Err(AnalysisError::new(
                        AmbiguousReference {
                            name: quoted(parts),
                            reference_names: format!("[{}]", names.join(", ")),
                        },
                        origin,
                    ));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn sub_env(&mut self, cx: &Cx<'_>) -> Arc<Env> {
        let mut outer = cx.env.outer.clone();
        // Over an aggregate, a subquery also sees its grouping attributes.
        let mut scope = cx.scope.to_vec();
        if cx.agg_scope.is_some() {
            scope.extend(cx.extra.unwrap_or_default().iter().cloned());
        }
        outer.push(super::resolve::Level {
            scope: scope.into(),
        });
        self.env(cx.env.ctes.clone(), cx.env.ctes_id, outer)
    }
}

/// The attributes carrying `id` below the operator.
fn carriers(cx: &Cx<'_>, id: crate::PlanId) -> Vec<Attr> {
    cx.rels
        .iter()
        .filter_map(|r| r.carriers.get(&id))
        .flatten()
        .flat_map(|out| out.iter().cloned())
        .collect()
}

/// `UNRESOLVED_COLUMN`, with up to five of `scope`'s columns ranked by edit
/// distance (qualified by their last qualifier part when `parts` is).
pub(super) fn unresolved(parts: &[String], scope: &[Attr], origin: Origin) -> AnalysisError {
    let target = parts.join(".");
    let qualified = parts.len() > 1;
    let mut ranked: Vec<(usize, Vec<String>)> = scope
        .iter()
        .map(|a| {
            let mut p: Vec<String> = a
                .qual
                .last()
                .filter(|_| qualified)
                .cloned()
                .into_iter()
                .collect();
            p.push(a.col.name.clone());
            (levenshtein(&p.join("."), &target), p)
        })
        .collect();
    // Closest first; ties in name order, as Spark orders them.
    ranked.sort();
    let object_name = quoted(parts);
    if ranked.is_empty() {
        return AnalysisError::new(UnresolvedColumnWithoutSuggestion { object_name }, origin);
    }
    let proposal = ranked
        .iter()
        .take(5)
        .map(|(_, p)| quoted(p))
        .collect::<Vec<_>>()
        .join(", ");
    AnalysisError::new(
        UnresolvedColumnWithSuggestion {
            object_name,
            proposal,
        },
        origin,
    )
}

/// The quoted names of `attrs`, comma-separated.
pub(super) fn names(attrs: &[Attr], _qualified: bool) -> String {
    attrs
        .iter()
        .map(|a| quoted(std::slice::from_ref(&a.col.name)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(x != *y)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// The longest generated name, in bytes.
const MAX_NAME: usize = 64 << 10;

/// Spark's generated name for an unaliased expression (`toPrettySQL`):
/// `(a + 1)`, `count(1)`, `CASE WHEN ... END`. Built bottom-up with an
/// explicit stack, once per distinct node.
pub(super) fn pretty(e: &Expr) -> String {
    let mut done: HashMap<usize, String> = HashMap::new();
    let mut stack = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if done.contains_key(&addr(x)) {
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(&**c));
        if !expanded {
            stack.push((x, true));
            stack.extend(kids.into_iter().rev().map(|c| (c, false)));
            continue;
        }
        let k: Vec<&str> = kids
            .iter()
            .map(|c| done.get(&addr(*c)).map_or("", String::as_str))
            .collect();
        let mut s = pretty_node(x, &k);
        if s.len() > MAX_NAME {
            // A shared subexpression repeated in a name doubles it per level:
            // names stay bounded (Spark's would not).
            let mut end = MAX_NAME;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            s.truncate(end);
            s.push_str("...");
        }
        done.insert(addr(x), s);
    }
    done.remove(&addr(e)).unwrap_or_default()
}

/// The names of the outer references of a subquery plan, in order, each
/// once.
fn outer_names(plan: &crate::LogicalPlan) -> String {
    use crate::walk::{Node, visit_distinct};
    let mut names: Vec<(AttrId, String)> = Vec::new();
    let _ = visit_distinct(Node::Plan(plan), |n| {
        if let Node::Expr(e) = n
            && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
            && !names.iter().any(|(id, _)| *id == c.id)
        {
            names.push((c.id, c.name.clone()));
        }
        std::ops::ControlFlow::Continue(())
    });
    names
        .into_iter()
        .map(|(_, n)| n)
        .collect::<Vec<_>>()
        .join(", ")
}

fn pretty_node(x: &Expr, k: &[&str]) -> String {
    match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            c.name.clone()
        }
        ExprNode::Column(c) => c.name.last().cloned().unwrap_or_default(),
        ExprNode::Literal(v) => pretty_literal(v),
        ExprNode::Cast { to, mode, .. } => {
            let cast = if *mode == CastMode::Try {
                "TRY_CAST"
            } else {
                "CAST"
            };
            format!("{cast}({} AS {})", k[0], to.sql())
        }
        ExprNode::Alias { .. } | ExprNode::Resolved(ResolvedExpr::Alias { .. }) => k[0].to_owned(),
        ExprNode::Subquery { kind, plan } => {
            // Spark lists the outer references.
            let outer = outer_names(plan);
            match kind {
                SubqueryKind::Scalar => format!("scalarsubquery({outer})"),
                SubqueryKind::Exists => format!("exists({outer})"),
                SubqueryKind::In(_) if k.len() == 1 => {
                    format!("({} IN (listquery({outer})))", k[0])
                }
                SubqueryKind::In(_) => {
                    let fields: Vec<String> = k.iter().map(|v| format!("'{v}', {v}")).collect();
                    format!(
                        "(named_struct({}) IN (listquery({outer})))",
                        fields.join(", ")
                    )
                }
            }
        }
        ExprNode::Star(_) => "*".to_owned(),
        ExprNode::Function(FunctionCall { name, distinct, .. }) => call(name, None, *distinct, k),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let op = match &f.id {
                FunctionId::Builtin(b) => Some(*b),
                FunctionId::Registry { name, .. } => Builtin::from_name(name),
            };
            let n = f.args.len().min(k.len());
            let text = call(f.id.name(), op, f.distinct, &k[..n]);
            match k.get(n) {
                Some(filter) => format!("{text} FILTER (WHERE {filter})"),
                None => text,
            }
        }
    }
}

fn call(name: &str, op: Option<Builtin>, distinct: bool, k: &[&str]) -> String {
    let infix = |sym: &str| format!("({} {sym} {})", k[0], k[1]);
    match (op, k.len()) {
        (Some(Builtin::Eq), 2) => infix("="),
        (Some(Builtin::EqNullSafe), 2) => infix("<=>"),
        (Some(Builtin::Lt), 2) => infix("<"),
        (Some(Builtin::LtEq), 2) => infix("<="),
        (Some(Builtin::Gt), 2) => infix(">"),
        (Some(Builtin::GtEq), 2) => infix(">="),
        (Some(Builtin::And), 2) => infix("AND"),
        (Some(Builtin::Or), 2) => infix("OR"),
        (Some(Builtin::Add), 2) => infix("+"),
        (Some(Builtin::Subtract), 2) => infix("-"),
        (Some(Builtin::Multiply), 2) => infix("*"),
        (Some(Builtin::Divide), 2) => infix("/"),
        (Some(Builtin::Remainder), 2) => infix("%"),
        (Some(Builtin::IntegralDivide), 2) => infix("div"),
        (Some(Builtin::Like), 2) => format!("{} LIKE {}", k[0], k[1]),
        (Some(Builtin::Not), 1) => format!("(NOT {})", k[0]),
        (Some(Builtin::Negative), 1) => format!("(- {})", k[0]),
        (Some(Builtin::IsNull), 1) => format!("({} IS NULL)", k[0]),
        (Some(Builtin::IsNotNull), 1) => format!("({} IS NOT NULL)", k[0]),
        (Some(Builtin::In), n) if n > 0 => format!("({} IN ({}))", k[0], k[1..].join(", ")),
        (Some(Builtin::When), _) => {
            let mut s = "CASE".to_owned();
            for pair in k.chunks(2) {
                match pair {
                    [c, v] => s.push_str(&format!(" WHEN {c} THEN {v}")),
                    [e] => s.push_str(&format!(" ELSE {e}")),
                    _ => {}
                }
            }
            s + " END"
        }
        // Spark's own spellings of `if`, `positive` and `mod`.
        _ if name.eq_ignore_ascii_case("if") && k.len() == 3 => {
            format!("(IF({}, {}, {}))", k[0], k[1], k[2])
        }
        _ if name.eq_ignore_ascii_case("positive") && k.len() == 1 => format!("(+ {})", k[0]),
        _ if name.eq_ignore_ascii_case("mod") && k.len() == 2 => infix("%"),
        _ if name.eq_ignore_ascii_case("extract") && k.len() == 2 => {
            format!("extract({} FROM {})", k[0], k[1])
        }
        // Spark's name of `first(x, ignoreNulls)` leaves the flag out.
        _ if k.len() == 2
            && matches!(k[1], "true" | "false")
            && ["first", "last", "first_value", "last_value"]
                .iter()
                .any(|n| name.eq_ignore_ascii_case(n)) =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), k[0])
        }
        _ => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), k.join(", "))
        }
    }
}

fn pretty_literal(v: &Literal) -> String {
    let text = v.to_string();
    match v {
        // A URL's credentials never reach a name (ADR 0038).
        Literal::String(s) => krishiv_common::secret::SecretUrl::parse(s)
            .map_or_else(|| s.clone(), |url| url.to_string()),
        Literal::Null(_) => "NULL".to_owned(),
        Literal::Byte(_) | Literal::Short(_) | Literal::Long(_) => {
            text.trim_end_matches(['Y', 'S', 'L']).to_owned()
        }
        Literal::Float(x) => format!("{x:?}"),
        Literal::Double(x) => format!("{x:?}"),
        Literal::Decimal { .. } => text.trim_end_matches("BD").to_owned(),
        _ => text,
    }
}

/// A semantic key: equal for expressions that compute the same value,
/// whatever their origins, column spellings or (top-level) alias.
pub(super) fn semantic_key(e: &Expr, memo: &mut HashMap<usize, u128>) -> u128 {
    let e = match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => &**expr,
        _ => e,
    };
    let mut stack = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if memo.contains_key(&addr(x)) {
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(&**c));
        if !expanded {
            stack.push((x, true));
            stack.extend(kids.into_iter().map(|c| (c, false)));
            continue;
        }
        let mut b = match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => format!("c{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => format!("o{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                format!("f{}:{:?}:{}", f.id.name(), f.kind, f.distinct)
            }
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => format!("a{}", output.id.0),
            ExprNode::Literal(v) => format!("l{v:?}"),
            ExprNode::Cast { to, mode, .. } => format!("x{}:{mode:?}", to.simple_string()),
            ExprNode::Subquery { plan, .. } => format!("s{}", addr(&**plan)),
            other => format!("u{other:?}"),
        }
        .into_bytes();
        for c in kids {
            b.extend_from_slice(&memo.get(&addr(c)).copied().unwrap_or(0).to_le_bytes());
        }
        memo.insert(addr(x), XxHash3_128::oneshot(&b));
    }
    memo.get(&addr(e)).copied().unwrap_or(0)
}
