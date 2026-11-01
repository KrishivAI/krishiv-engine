// SPDX-License-Identifier: Apache-2.0
//! Expression binding: names, stars, functions, casts, subqueries, output
//! names (Spark's `toPrettySQL`) and semantic keys.

use super::SessionZone;
use krishiv_kernels::datetime::TimeZoneRules;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::{
    AggregateFunctionWithNondeterministicExpression, AmbiguousColumnReference,
    AmbiguousLateralColumnAlias, AmbiguousReference, CannotResolveDataframeColumn,
    CannotResolveStarExpand, DatatypeMismatchBinaryOpDiffTypes, DatatypeMismatchBinaryOpWrongType,
    DatatypeMismatchCannotDropAllFields, DatatypeMismatchCastWithConfSuggestion,
    DatatypeMismatchCastWithFuncSuggestion, DatatypeMismatchCastWithoutSuggestion,
    DatatypeMismatchCreateMapKeyDiffTypes, DatatypeMismatchCreateMapValueDiffTypes,
    DatatypeMismatchDataDiffTypes, DatatypeMismatchInSubqueryDataTypeMismatch,
    DatatypeMismatchInSubqueryLengthMismatch, DatatypeMismatchInputSizeNotOne,
    DatatypeMismatchInvalidOrderingType, DatatypeMismatchMapConcatDiffTypes,
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchUnexpectedNull, DatatypeMismatchValueOutOfRange,
    DuplicateRoutineParameterAssignmentBothPositionalAndNamed,
    DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference, DuplicatedMapKey,
    ExceptOverlappingColumns, FailedFunctionCall, FailedToParseTooComplex, FieldNotFound,
    GroupByPosOutOfRange, InternalError, InvalidAggregateFilterNotBoolean,
    InvalidExtractBaseFieldType, InvalidExtractFieldType,
    InvalidSqlSyntaxFunctionWithUnsupportedSyntax,
    InvalidSubqueryExpressionScalarSubqueryReturnMoreThanOneOutputColumn,
    InvalidUsageOfStarOrRegex, InvalidUsageOfStarWithTableIdentifierInCount,
    InvalidWithinGroupExpressionDistinctUnsupported,
    InvalidWithinGroupExpressionMismatchWithDistinctInput,
    InvalidWithinGroupExpressionMismatchWithDistinctInputUnsafeCast,
    InvalidWithinGroupExpressionWithinGroupMissing, InvalidWithinGroupExpressionWrongNumOrderings,
    KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp1104, NamedParametersNotSupported,
    NestedAggregateFunction, NonFoldableArgument, RequiredParameterNotFound, UnexpectedInputType,
    UnexpectedPositionalArgument, UnrecognizedParameterName, UnresolvedColumnWithSuggestion,
    UnresolvedColumnWithoutSuggestion, UnresolvedRoutine, WrongNumArgsWithSuggestion,
    WrongNumArgsWithoutSuggestion,
};
use krishiv_common::stack::on_big_stack;
use krishiv_types::{ArrayType, DataType, DecimalType, StructField, StructType};
use twox_hash::XxHash3_128;

use super::analytics::GroupingFn;
use super::bind::{addr, qualified_parts, same, unsupported};
use super::builtins::{NamedDefault, NamedParam, named_signature};
use super::catalog::quoted;
use super::coerce::{self, Arg, BadCast, Mismatch, list_types, ordinal, quoted_type};
use super::fold;
use super::resolve::{Attr, Env, RESOLVE_STEP, Rel, Resolver};
use super::suggest::closest;
use super::{
    AnalysisError, ArgType, CallTypes, Constant, FunctionEntry, FunctionLookup, InputType, Inputs,
};
use crate::{
    AttrId, BoundColumn, Builtin, CastMode, ColumnRef, Expr, ExprNode, FieldOp, FunctionCall,
    FunctionId, FunctionKind, Literal, LogicalPlan, NullOrdering, Origin, QuantifiedLike,
    ResolvedExpr, ResolvedFunction, SortDirection, SortOrder, Star, SubqueryKind, TypedLiteralKind,
};

mod aggregates;
mod calls;
mod columns;
mod conditionals;
mod in_subquery;
mod literals;
mod node;
mod operators;
mod patterns;
mod pretty;
mod semantic;
mod star;

use calls::*;
pub(super) use columns::*;
use conditionals::*;
pub(super) use literals::*;
use operators::*;
use patterns::*;
pub(super) use pretty::*;
pub(super) use semantic::*;

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
    /// than ambiguous (Spark's `ORDER BY` over a select list, `HAVING` and
    /// `ORDER BY` over an aggregate's duplicate outputs among its grouping
    /// attributes); when nothing further holds it either, the duplicates
    /// stay ambiguous rather than unresolved.
    pub(super) lenient: bool,
    /// Whether names may resolve in enclosing queries.
    pub(super) outer: bool,
    /// Where window expressions may appear.
    pub(super) windows: WindowUse,
    pub(super) env: &'s Arc<Env>,
    /// The operator, as Spark names it in errors.
    pub(super) op: &'static str,
}

/// Where an operator's expressions may hold window expressions, and the
/// error Spark raises where they may not.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WindowUse {
    /// A select list (`Project`, `Aggregate`): windows are extracted.
    Allowed,
    /// `WHERE`: `WINDOW_FUNCTION_NOT_ALLOWED_IN_CLAUSE`.
    Where,
    /// `HAVING`: `WINDOW_FUNCTION_NOT_ALLOWED_IN_CLAUSE`.
    Having,
    /// Anywhere else: `UNSUPPORTED_EXPR_FOR_OPERATOR`.
    Other,
}

/// A bound expression.
#[derive(Clone)]
pub(super) struct Bound {
    pub(super) expr: Arc<Expr>,
    pub(super) ty: ArgType,
    /// The first aggregate call inside (outside subqueries), not counting
    /// window aggregates (`sum(x) OVER w`).
    pub(super) agg: Option<Origin>,
    /// The first window expression inside (outside subqueries).
    pub(super) win: Option<Origin>,
    /// The outermost query level referenced, for outer references.
    pub(super) outer: Option<usize>,
    /// The lateral column aliases referenced (their attribute ids).
    pub(super) lateral: Vec<AttrId>,
    /// The expression as Spark shows it in names and errors, when that is
    /// not `expr`: before implicit casts and rewrites.
    pub(super) shown: Option<Arc<Expr>>,
}

impl Bound {
    /// Adds the analysis markers of `kids` to this expression: rebinding an
    /// already resolved child drops its markers, so a rewrite that rebinds
    /// its children carries them over.
    pub(super) fn carry_markers(&mut self, kids: &[Self]) {
        for k in kids {
            self.agg = self.agg.or(k.agg);
            self.win = self.win.or(k.win);
            self.outer = match (self.outer, k.outer) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (x, y) => x.or(y),
            };
            for id in &k.lateral {
                if !self.lateral.contains(id) {
                    self.lateral.push(*id);
                }
            }
        }
    }
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

/// The replacement characters `mask` takes when a call omits them, which
/// Spark's constructors write into the call (so the schema names them):
/// `mask(s)` is `mask(s, 'X', 'x', 'n', NULL)`.
const MASK_DEFAULTS: [Option<&str>; 4] = [Some("X"), Some("x"), Some("n"), None];

/// A literal argument written by a constructor, at `origin`.
fn default_argument(
    literal: Literal,
    data_type: DataType,
    nullable: bool,
    origin: Origin,
) -> Bound {
    Bound {
        expr: Arc::new(Expr::new(ExprNode::Literal(literal)).with_origin(origin)),
        ty: arg_type(data_type, nullable),
        agg: None,
        win: None,
        outer: None,
        lateral: Vec::new(),
        shown: None,
    }
}

/// Whether `entry` is a built-in (a canonical operator or a registry
/// function), rather than a session UDF. Every special-cased built-in path
/// (`assert_true`'s default message, the `Conditional` rewrites, `ntile`'s
/// folding) goes through this one rule: built-ins win over a same-name
/// session UDF for unqualified names, and a qualified call to the session
/// UDF reaches the UDF.
fn is_builtin_entry(entry: &FunctionEntry) -> bool {
    !matches!(entry.id, FunctionId::Udf { .. })
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

/// Where a node is bound: inside an aggregate call's arguments (`agg`),
/// and whether it is the function of a window expression (`window_fn`:
/// `sum(x)` in `sum(x) OVER w` is a window aggregate, not an aggregate of
/// the operator).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Ctx {
    pub(super) agg: bool,
    pub(super) window_fn: bool,
}

enum Step {
    Enter(Arc<Expr>, Ctx),
    Exit(Arc<Expr>, Ctx),
    /// The first expression's result is the second's (a rewritten call).
    Link(usize, Arc<Expr>, Ctx),
}

impl Resolver<'_> {
    /// Binds `e` in `cx`, post-order with an explicit stack; a node shared
    /// within the expression is bound once.
    pub(super) fn expr(&mut self, e: &Arc<Expr>, cx: &Cx<'_>) -> Result<Bound, AnalysisError> {
        let top = Ctx {
            agg: false,
            window_fn: false,
        };
        self.expr_in(top, e, cx)
    }

    /// Binds `e` in `cx` under the step context `top` (see [`Self::expr`]).
    /// Higher-order calls bind their children through this with each
    /// child's own context, so names inside lambdas resolve as the generic
    /// path would resolve them. Every [`RESOLVE_STEP`] levels of
    /// higher-order nesting run on a fresh stack (see [`Resolver::rel`).
    pub(super) fn expr_in(
        &mut self,
        top: Ctx,
        e: &Arc<Expr>,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        self.expr_depth += 1;
        let r = if self.expr_depth.is_multiple_of(RESOLVE_STEP) {
            on_big_stack(|| self.expr_in_on_stack(top, e, cx))
                .unwrap_or_else(|| Err(AnalysisError::new(FailedToParseTooComplex {}, e.origin)))
        } else {
            self.expr_in_on_stack(top, e, cx)
        };
        self.expr_depth -= 1;
        r
    }

    /// [`Self::expr_in`] on the current stack.
    fn expr_in_on_stack(
        &mut self,
        top: Ctx,
        e: &Arc<Expr>,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let mut done: HashMap<(usize, Ctx), Bound> = HashMap::new();
        // Rewritten calls stay alive for the pass: `done` is keyed by
        // address.
        let mut rewritten_calls: Vec<Arc<Expr>> = Vec::new();
        let mut stack = vec![Step::Enter(Arc::clone(e), top)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(x, ctx) => {
                    if done.contains_key(&(addr(&*x), ctx)) {
                        continue;
                    }
                    if let Some(rewritten) = self.star_args(&x, cx)? {
                        rewritten_calls.push(Arc::clone(&rewritten));
                        stack.push(Step::Link(addr(&*x), Arc::clone(&rewritten), ctx));
                        stack.push(Step::Enter(rewritten, ctx));
                        continue;
                    }
                    // A lambda binds where it is used: a higher-order
                    // call binds its array and lambda children itself
                    // (see `hof`), while a lambda anywhere else is no
                    // higher-order call's argument. Only unqualified and
                    // builtin-qualified calls take the higher-order path:
                    // a session-qualified UDF named like one (`session.filter`,
                    // ...) runs the UDF.
                    let hof = |f: &FunctionCall| {
                        (f.unqualified().is_some()
                            || super::bind::is_builtin_qualifier(&f.qualifier))
                        .then(|| super::hof::Hof::of(&f.name))
                        .flatten()
                    };
                    if let ExprNode::Function(f) = &x.node
                        && (hof(f).is_some() || super::hof::has_direct_lambda(f))
                    {
                        let Some(h) = hof(f) else {
                            return Err(super::hof::not_higher_order(&f.name, x.origin));
                        };
                        let kids = self.child_ctx(&x, ctx);
                        for (key, bound) in self.hof_children(f, h, &kids, cx, x.origin)? {
                            let step = kids
                                .iter()
                                .find(|(c, _)| addr(&**c) == key)
                                .map(|(_, k)| *k)
                                .unwrap_or(ctx);
                            done.insert((key, step), bound);
                        }
                        stack.push(Step::Exit(Arc::clone(&x), ctx));
                        let lambdas: Vec<usize> = h
                            .lambdas()
                            .iter()
                            .filter(|&&i| i < f.args.len())
                            .map(|&i| addr(&*f.args[i]))
                            .collect();
                        stack.extend(
                            kids.into_iter()
                                .rev()
                                .filter(|(c, _)| !lambdas.contains(&addr(&**c)))
                                .map(|(c, k)| Step::Enter(c, k)),
                        );
                        continue;
                    }
                    self.unknown_routine_first(&x)?;
                    stack.push(Step::Exit(Arc::clone(&x), ctx));
                    let kids = self.child_ctx(&x, ctx);
                    stack.extend(kids.into_iter().rev().map(|(c, k)| Step::Enter(c, k)));
                }
                Step::Link(from, to, ctx) => {
                    if let Some(b) = done.get(&(addr(&*to), ctx)).cloned() {
                        done.insert((from, ctx), b);
                    }
                }
                Step::Exit(x, ctx) => {
                    let kids: Vec<Bound> = self
                        .child_ctx(&x, ctx)
                        .iter()
                        .filter_map(|(c, k)| done.get(&(addr(&**c), *k)).cloned())
                        .collect();
                    let b = self.expr_node(&x, &kids, ctx.agg, ctx.window_fn, cx)?;
                    done.insert((addr(&*x), ctx), b);
                }
            }
        }
        done.remove(&(addr(&**e), top))
            .ok_or_else(|| unsupported("an expression shape", e.origin))
    }

    /// The children of `x` (bound in context `ctx`) with their contexts:
    /// inside an aggregate call's arguments once inside one (a window
    /// aggregate's arguments are not), the window function of a window
    /// expression marked.
    #[inline(never)]
    fn child_ctx(&self, x: &Expr, ctx: Ctx) -> Vec<(Arc<Expr>, Ctx)> {
        let agg = ctx.agg || (!ctx.window_fn && self.is_aggregate(x));
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| {
            kids.push((
                Arc::clone(c),
                Ctx {
                    agg,
                    window_fn: false,
                },
            ));
        });
        if let ExprNode::Window(_) = &x.node
            && let Some((_, first)) = kids.first_mut()
        {
            // The window function: `for_each_child` visits it first.
            first.window_fn = true;
        }
        kids
    }

    pub(super) fn is_aggregate(&self, e: &Expr) -> bool {
        match &e.node {
            ExprNode::Function(f) => {
                // A session-qualified call (`session.f`) resolves through
                // its full qualifier (UDFs only), as resolution does; any
                // other call resolves by its bare name.
                let name;
                let lookup: &[String] = if super::bind::is_session_qualifier(&f.qualifier) {
                    let mut parts = f.qualifier.clone();
                    parts.push(f.name.clone());
                    name = parts;
                    &name
                } else {
                    std::slice::from_ref(&f.name)
                };
                matches!(
                    self.ctx.functions.lookup(lookup, self.ctx.profile),
                    FunctionLookup::Found(entry) if entry.kind == FunctionKind::Aggregate
                )
            }
            ExprNode::Resolved(ResolvedExpr::Function(f)) => f.kind == FunctionKind::Aggregate,
            _ => false,
        }
    }
}

/// `DATATYPE_MISMATCH.NON_FOLDABLE_INPUT` for an unfoldable quantile
/// percentage, quoting the call as Spark's `toPrettySQL` does.
fn non_foldable(
    name: &str,
    value: &Bound,
    pct: &Bound,
    array: bool,
    zone: Option<SessionZone>,
    origin: Origin,
) -> AnalysisError {
    let sql = format!(
        "\"{name}({})\"",
        [value.shown(), pct.shown()]
            .iter()
            .map(|a| pretty(a, zone))
            .collect::<Vec<_>>()
            .join(", ")
    );
    AnalysisError::new(
        DatatypeMismatchNonFoldableInput {
            sql_expr: sql,
            input_name: "`percentage`".to_owned(),
            input_type: if array {
                "\"ARRAY<DOUBLE>\"".to_owned()
            } else {
                "\"DOUBLE\"".to_owned()
            },
            input_expr: format!("\"{}\"", pretty(&pct.shown(), zone)),
        },
        origin,
    )
}

/// The elements of an array percentage: through casts to the `array`
/// call the analyzer built, `None` for anything else.
fn array_elements(e: &Arc<Expr>) -> Option<Vec<Arc<Expr>>> {
    let mut e = e;
    loop {
        match &e.node {
            ExprNode::Cast { expr, .. } => e = expr,
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.id.name() == "array" => {
                return Some(f.args.clone());
            }
            _ => return None,
        }
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

/// Whether `e` holds no nondeterministic call: a struct update repeats its
/// base (in the null check, in every kept field and in the new struct), so a
/// nondeterministic base would read a different value per use (the analyzer
/// has no projection below that binds it once, as Spark's
/// `PullOutNondeterministic` does). An unresolved call counts as
/// nondeterministic: a bound expression never holds one.
fn is_deterministic(e: &Expr) -> bool {
    let mut nondeterministic = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n {
            match &x.node {
                ExprNode::Function(_) => {
                    nondeterministic = true;
                    return std::ops::ControlFlow::Break(());
                }
                ExprNode::Resolved(ResolvedExpr::Function(f)) if !f.deterministic => {
                    nondeterministic = true;
                    return std::ops::ControlFlow::Break(());
                }
                _ => {}
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    !nondeterministic
}

/// The longest generated name, in bytes.
const MAX_NAME: usize = 64 << 10;

impl Resolver<'_> {
    /// A call of a function that does not exist is reported before the
    /// errors of its arguments (Spark's `CheckAnalysis` reports the
    /// outermost unresolved node first).
    fn unknown_routine_first(&self, x: &Expr) -> Result<(), AnalysisError> {
        let ExprNode::Function(f) = &x.node else {
            return Ok(());
        };
        let n = f.args.len();
        if !f.qualifier.is_empty()
            || GroupingFn::from_name(&f.name).is_some()
            || builtin_op(&f.name, n).is_some()
            || !matches!(
                self.ctx
                    .functions
                    .lookup(std::slice::from_ref(&f.name), self.ctx.profile),
                FunctionLookup::NotFound
            )
        {
            return Ok(());
        }
        let (parts, _) = qualified_parts(f, &self.b.search_path, x.origin)?;
        Err(AnalysisError::new(
            UnresolvedRoutine {
                routine_name: quoted(&parts),
                search_path: self.b.search_path.clone(),
            },
            x.origin,
        ))
    }

    /// `AGGREGATE_FUNCTION_WITH_NONDETERMINISTIC_EXPRESSION`: an argument
    /// of an aggregate calls a nondeterministic function. A call of a SQL
    /// function is as deterministic as its body, whatever its arguments.
    fn nondeterministic_arguments(&mut self, e: &Arc<Expr>, args: &[Bound]) {
        if self.pending_nondeterministic.is_some() {
            return;
        }
        let Some(bad) = args.iter().find(|k| {
            !super::sqlfn::is_deterministic(&k.expr)
                && self.nondeterministic(&k.shown(), &mut HashSet::new())
        }) else {
            return;
        };
        let shown = with_children(e, args.iter().map(Bound::shown).collect());
        self.pending_nondeterministic = Some(AnalysisError::new(
            AggregateFunctionWithNondeterministicExpression {
                sql_expr: format!("\"{}\"", pretty(&shown, self.ctx.conf.session_zone)),
            },
            bad.expr.origin,
        ));
    }

    /// Whether `e`, as written, calls a nondeterministic function; a call
    /// of a SQL function is as deterministic as its body, whatever its
    /// arguments. `seen` holds the nodes visited (expressions are DAGs).
    fn nondeterministic(&self, e: &Expr, seen: &mut HashSet<usize>) -> bool {
        if !seen.insert(addr(e)) {
            return false;
        }
        if let ExprNode::Function(f) = &e.node {
            let mut name = f.qualifier.clone();
            name.push(f.name.clone());
            if let FunctionLookup::Sql(def) = self.ctx.functions.lookup(&name, self.ctx.profile) {
                return self.nondeterministic(&def.body, seen);
            }
            if f.qualifier.is_empty()
                && self.ctx.functions.spark_deterministic(&f.name) == Some(false)
            {
                return true;
            }
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
            && !f.deterministic
        {
            return true;
        }
        let mut found = false;
        e.for_each_child(&mut |c| found = found || self.nondeterministic(c, seen));
        found
    }
}

/// The checks of a call that is not a window's function, out of line (the
/// binding frame stays small): a window function needs `OVER`; an
/// aggregate holds no aggregate and no window. A window aggregate
/// (`sum(sum(x)) OVER w`) may hold both.
#[inline(never)]
fn call_checks(
    entry: &FunctionEntry,
    e: &Arc<Expr>,
    kids: &[Bound],
    n: usize,
) -> Result<(), AnalysisError> {
    if entry.kind == FunctionKind::Window {
        let shown = with_children(e, kids.iter().map(Bound::shown).collect());
        return Err(super::windows::without_over(
            entry.id.name(),
            &shown,
            e.origin,
        ));
    }
    if entry.kind == FunctionKind::Aggregate {
        if let Some(inner) = kids.iter().find_map(|k| k.agg) {
            return Err(AnalysisError::new(NestedAggregateFunction {}, inner));
        }
        if kids[..n].iter().any(|k| k.win.is_some()) {
            return Err(super::windows::window_in_aggregate(e.origin));
        }
    }
    Ok(())
}

/// The canonical operator for an unresolved call, if it is one.
///
/// Unary `-` (lowered as `"-"` with one argument) is `Negative`; the
/// function `negative(x)` is not an operator but a registry function (as
/// Spark's `UnaryMinus` with alias `negative` is), so it never maps here.
fn builtin_op(name: &str, arity: usize) -> Option<crate::Builtin> {
    if name == "-" && arity == 1 {
        return Some(crate::Builtin::Negative);
    }
    if name.eq_ignore_ascii_case("negative") {
        return None;
    }
    crate::Builtin::from_name(name).filter(|b| !b.is_analyzer_internal())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::{Arity, Constant, ResultType};
    use super::*;
    use super::{Bound, Resolver};
    use crate::{
        AttrId, BoundColumn, Expr, ExprNode, FunctionCall, FunctionId, FunctionKind, Literal,
        ResolvedExpr,
    };
    use krishiv_types::DataType;

    /// A `FunctionEntry` for `id`: what the session registry builds for the
    /// built-in window function `ntile` and for a session UDF of that name.
    fn entry(id: FunctionId) -> super::super::FunctionEntry {
        let result: ResultType = Arc::new(|_| {
            Ok(super::super::ArgType {
                data_type: None,
                nullable: true,
            })
        });
        super::super::FunctionEntry {
            id,
            kind: FunctionKind::Scalar,
            arity: Arity::any(),
            inputs: super::super::Inputs::Any,
            result,
            deterministic: true,
        }
    }

    /// One bound `1 + 1` argument, as `typed_call` passes it.
    fn kid() -> Bound {
        let one = || Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(1))));
        Bound {
            expr: Arc::new(Expr::new(ExprNode::Function(FunctionCall {
                within_group: Vec::new(),
                name: "+".to_owned(),
                qualifier: Vec::new(),
                arg_names: Vec::new(),
                args: vec![one(), one()],
                distinct: false,
                filter: None,
            }))),
            ty: super::super::ArgType {
                data_type: Some(DataType::Integer),
                nullable: false,
            },
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        }
    }

    /// `ntile`'s bucket count folds to its literal for the built-in only: a
    /// session UDF named `ntile` keeps its arguments as written. The
    /// difference is analyzer-level (the optimizer folds `1 + 1` either
    /// way), so it is checked here rather than through query results.
    #[test]
    fn folded_ntile_only_folds_for_builtins() {
        let constants = vec![Some(Constant::Integral(2))];
        let builtin = entry(FunctionId::Registry {
            name: "ntile".to_owned(),
            id: 0,
        });
        let folded = Resolver::folded_ntile(&builtin, &constants, vec![kid()]);
        assert!(
            matches!(&folded[0].expr.node, ExprNode::Literal(Literal::Integer(2))),
            "the built-in folds its bucket count"
        );
        let udf = entry(FunctionId::Udf {
            name: "ntile".to_owned(),
            version: 1,
        });
        let kept = Resolver::folded_ntile(&udf, &constants, vec![kid()]);
        assert!(
            matches!(
                &kept[0].expr.node,
                ExprNode::Function(f) if f.name == "+" && f.qualifier.is_empty()
            ),
            "a session UDF keeps its argument as written"
        );
    }

    #[test]
    fn a_map_literal_types_and_prints_as_a_string_map() {
        let literal = Literal::Map(vec![("mode".to_owned(), "FAILFAST".to_owned())]);
        assert_eq!(
            literal_type(&literal),
            Some(DataType::Map(
                krishiv_types::MapType::new(DataType::String, DataType::String, false).into()
            ))
        );
        assert_eq!(literal.to_string(), "MAP('mode', 'FAILFAST')");
    }

    /// `lambdafunction(body, arg1, ...)`: a bound lambda variable prints
    /// as `namedlambdavariable()`, like Spark's `NamedLambdaVariable`;
    /// an unbound lambda keeps its written names.
    #[test]
    fn lambdas_print_as_sparks_lambdafunction() {
        let var = BoundColumn {
            id: AttrId(9),
            name: "y".to_owned(),
            data_type: DataType::Integer,
            nullable: false,
        };
        let body = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::LambdaVar(
            var.clone(),
        ))));
        assert_eq!(pretty(&body, None), "namedlambdavariable()");
        let lam = Expr::new(ExprNode::Lambda {
            params: vec![var],
            body,
        });
        assert_eq!(
            pretty(&lam, None),
            "lambdafunction(namedlambdavariable(), namedlambdavariable())"
        );
        // Placeholders keep their written names (error quotes of calls
        // whose lambdas never bound).
        let unbound = Expr::lambda(vec!["y".to_owned()], Expr::column(["y"]));
        assert_eq!(pretty(&unbound, None), "lambdafunction(y, y)");
    }

    /// `*` prints as itself (the arm beside the lambda printers).
    #[test]
    fn stars_print_as_sql() {
        let star = Expr::new(ExprNode::Star(crate::Star {
            replace: Vec::new(),
            position: None,
            qualifier: Vec::new(),
            except: Vec::new(),
            plan_id: None,
        }));
        assert_eq!(pretty(&star, None), "*");
    }

    /// `call()` renders named arguments with their `name => ` prefixes and
    /// quantified forms in their SQL shape.
    #[test]
    fn calls_print_names_and_quantifiers() {
        let names: Vec<Option<String>> =
            vec![None, Some("lowerChar".to_owned()), Some("str".to_owned())];
        assert_eq!(
            call("mask", None, false, &["a", "q", "b"], Some(&names), None),
            "mask(a, lowerChar => q, str => b)"
        );
        assert_eq!(
            call("mask", None, false, &["a", "q"], None, None),
            "mask(a, q)"
        );
        assert_eq!(
            call(
                "like_any",
                Some(Builtin::LikeAny),
                false,
                &["c", "'%oo%'", "'fa%'"],
                None,
                None
            ),
            "likeany(c)"
        );
        assert_eq!(
            call(
                "not_ilike_all",
                Some(Builtin::NotILikeAll),
                false,
                &["c", "'a'"],
                None,
                None
            ),
            "notlikeall(c)"
        );
    }

    /// `from_json` is named after its JSON argument alone, with or without
    /// options (Spark's `JsonToStructs` has one child).
    #[test]
    fn from_json_is_named_after_the_json_argument() {
        for k in [
            &["value", "a INT"][..],
            &["value", "a INT", "MAP('mode', 'FAILFAST')"][..],
        ] {
            assert_eq!(
                call("from_json", None, false, k, None, None),
                "from_json(value)"
            );
        }
    }

    /// The `UNRECOGNIZED_PARAMETER_NAME` proposal follows Spark's similarity
    /// order, ties in Scala-`Set` order.
    #[test]
    fn proposals_follow_spark_similarity() {
        let params = ["str", "upperChar", "lowerChar", "digitChar", "otherChar"];
        assert_eq!(
            suggestions("cellular", params.into_iter()),
            "`str` `upperChar` `otherChar`"
        );
        assert_eq!(
            suggestions("str", params.into_iter()),
            "`str` `otherChar` `digitChar`"
        );
    }

    /// Operators are never qualified: only a bare name matches an operator,
    /// so `session.+` prints as written.
    #[test]
    fn qualified_operators_print_as_written() {
        let qualified = Expr::new(ExprNode::Function(FunctionCall {
            within_group: Vec::new(),
            name: "+".to_owned(),
            qualifier: vec!["session".to_owned()],
            args: vec![
                Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(1)))),
                Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(2)))),
            ],
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
        }));
        assert_eq!(pretty(&qualified, None), "session.+(1, 2)");
        let bare = Expr::new(ExprNode::Function(FunctionCall {
            within_group: Vec::new(),
            name: "+".to_owned(),
            qualifier: Vec::new(),
            args: vec![
                Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(1)))),
                Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(2)))),
            ],
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
        }));
        assert_eq!(pretty(&bare, None), "(1 + 2)");
    }

    /// The test helper's result rule answers typeless and nullable.
    #[test]
    fn test_entries_answer_without_a_type() {
        use krishiv_functions::signature::CallTypes;
        let call = CallTypes::new(&[], &[]);
        for id in [
            FunctionId::Builtin(Builtin::Add),
            FunctionId::Udf {
                name: "my_udf".to_owned(),
                version: 1,
            },
        ] {
            let e = entry(id);
            let ty = (e.result)(&call).unwrap();
            assert_eq!(ty.data_type, None);
            assert!(ty.nullable);
        }
    }
}
