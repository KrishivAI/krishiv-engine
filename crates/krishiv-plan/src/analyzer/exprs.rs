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
    InvalidUsageOfStarOrRegex, InvalidUsageOfStarWithTableIdentifierInCount, KrishivError,
    NestedAggregateFunction, UnresolvedColumnWithSuggestion, UnresolvedColumnWithoutSuggestion,
    UnresolvedRoutine, WrongNumArgsWithSuggestion, WrongNumArgsWithoutSuggestion,
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
    TypedLiteralKind,
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

/// The default message of `assert_true(cond)`: `'<cond>' is not true!`,
/// with casts stripped as Spark's `toPrettySQL` strips them.
fn assert_message(cond: &Bound, origin: Origin, zone: Option<i32>) -> Bound {
    let mut shown = cond.shown();
    while let ExprNode::Cast { expr, .. } = &shown.node {
        shown = Arc::clone(expr);
    }
    let message = format!("'{}' is not true!", pretty(&shown, zone));
    Bound {
        expr: Arc::new(Expr::new(ExprNode::Literal(Literal::String(message))).with_origin(origin)),
        ty: arg_type(DataType::String, false),
        agg: None,
        win: None,
        outer: None,
        lateral: Vec::new(),
        shown: None,
    }
}

/// A conditional function rewritten during analysis (Spark's
/// `RuntimeReplaceable`).
#[derive(Clone, Copy)]
enum Conditional {
    /// `if(c, t, e)`: `CASE WHEN c THEN t ELSE e END`.
    If,
    /// `nvl(a, b)` / `ifnull(a, b)`: `coalesce(a, b)`.
    Coalesce,
    /// `nvl2(x, t, e)`: `CASE WHEN x IS NOT NULL THEN t ELSE e END`.
    Nvl2,
    /// `nullif(a, b)`: `a` at its own type with the `=`-cast `b` to compare
    /// against.
    NullIf,
    /// `nullifzero(x)`: `nullif(x, 0)`.
    NullIfZero,
    /// `zeroifnull(x)`: `coalesce(x, 0)` at the met type.
    ZeroIfNull,
    /// `nanvl(a, b)`: both sides are `float`/`double` already.
    Nanvl,
}

impl Conditional {
    /// Which conditional `name` is, ASCII case-insensitively (function
    /// names are); `None` for any other function, `coalesce` included.
    fn of(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "if" => Self::If,
            "nvl" | "ifnull" => Self::Coalesce,
            "nvl2" => Self::Nvl2,
            "nullif" => Self::NullIf,
            "nullifzero" => Self::NullIfZero,
            "zeroifnull" => Self::ZeroIfNull,
            "nanvl" => Self::Nanvl,
            _ => return None,
        })
    }
}

/// Whether `t` is a numeric type.
fn is_conditional_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
    )
}

/// The cast `b` needs for a legacy `nullif(a, b)` comparison that `=`
/// refuses (`None` when no legacy shape applies): the string side meets
/// the other side's type, a number meets a boolean at `BOOLEAN`. `None` as
/// the cast (rather than a type) leaves `b` as it is: `a` is the side the
/// evaluator casts in place.
fn legacy_nullif_cast(a: &Bound, b: &Bound) -> Option<Option<DataType>> {
    let (x, y) = (a.ty.data_type.as_ref(), b.ty.data_type.as_ref());
    match (x, y) {
        (Some(DataType::String), _) => Some(None),
        (_, Some(DataType::String)) => Some(x.cloned()),
        (Some(DataType::Boolean), Some(n)) if is_conditional_numeric(n) => {
            Some(Some(DataType::Boolean))
        }
        (Some(n), Some(DataType::Boolean)) if is_conditional_numeric(n) => Some(None),
        _ => None,
    }
}

/// The literal zero of `t`: what `zeroifnull` meets its argument with (`'0'`
/// for a legacy `STRING`). `None` for any other type, which a met argument
/// never has (it always meets the hidden `INT`, so never stays a narrower
/// integral type).
fn zero_literal(t: &DataType, origin: Origin) -> Option<Arc<Expr>> {
    let literal = match t {
        DataType::Integer => Literal::Integer(0),
        DataType::Long => Literal::Long(0),
        DataType::Float => Literal::Float(0.0),
        DataType::Double => Literal::Double(0.0),
        DataType::Decimal(d) => Literal::Decimal {
            unscaled: 0,
            data_type: *d,
        },
        DataType::String => Literal::String("0".to_owned()),
        _ => return None,
    };
    Some(Arc::new(
        Expr::new(ExprNode::Literal(literal)).with_origin(origin),
    ))
}

/// Spark's error for a cast of `arg` to `to` it refuses.
fn bad_cast(
    bad: BadCast,
    origin: Origin,
    arg: &Bound,
    to: &DataType,
    mode: CastMode,
    zone: Option<i32>,
) -> AnalysisError {
    let shown = ExprNode::Cast {
        expr: arg.shown(),
        to: to.clone(),
        mode,
    };
    let sql_expr = format!("\"{}\"", pretty(&Expr::new(shown), zone));
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

/// A typed datetime literal as Spark's `AstBuilder` builds it: the same
/// `CAST('...' AS TIMESTAMP)` kernels execution uses, evaluated here with
/// the session's resolved time zone and timestamp type. Plain `TIMESTAMP`
/// means `TIMESTAMP_NTZ` when `spark.sql.timestampType = TIMESTAMP_NTZ`
/// and the value names no zone, else an instant. Values needing the query
/// clock (`now`, `today`, a bare time of day) or the time-zone database
/// (a region id in the value, or a region session zone for a value
/// without a zone) are `KRISHIV_UNSUPPORTED_FEATURE`, as the kernels
/// report them.
///
/// # Errors
/// `INVALID_TYPED_LITERAL` for text the type does not read, at the
/// literal's origin; `KRISHIV_UNSUPPORTED_FEATURE` for clock- or
/// time-zone-dependent values.
fn typed_literal(
    kind: TypedLiteralKind,
    value: &str,
    conf: &super::ResolutionConf,
    origin: Origin,
) -> Result<Literal, AnalysisError> {
    use krishiv_common::error::{ErrorClass, InvalidTypedLiteral, KrishivUnsupportedFeature};
    use krishiv_kernels::cast::{
        CastMode as KernelMode, CastOptions, FixedOffset, TimestampText, cast, timestamp_has_zone,
        timestamp_text,
    };
    let invalid = |value_type: &str| {
        AnalysisError::new(
            InvalidTypedLiteral {
                value: format!("'{value}'"),
                value_type: format!("\"{value_type}\""),
            },
            origin,
        )
    };
    let unsupported = |feature: &str| {
        AnalysisError::new(
            KrishivUnsupportedFeature {
                feature: feature.to_owned(),
            },
            origin,
        )
    };
    let type_name = match kind {
        TypedLiteralKind::Date => "DATE",
        TypedLiteralKind::Timestamp => "TIMESTAMP",
        TypedLiteralKind::TimestampNtz => "TIMESTAMP_NTZ",
        TypedLiteralKind::TimestampLtz => "TIMESTAMP_LTZ",
    };
    // Clock-dependent spellings evaluate with the query clock, which the
    // analyzer has not got.
    if is_special(value) {
        return Err(unsupported(
            "a timestamp literal that needs the query clock",
        ));
    }
    let shape = timestamp_text(value);
    // A `DATE` is checked by its own cast below: the timestamp grammar
    // rejects dates whose instant is past the microsecond range
    // (`DATE '999999-03-18'`, valid in Spark), so only a bare time of day
    // is read from the timestamp shape for it.
    if shape == TimestampText::Malformed && kind != TypedLiteralKind::Date {
        return Err(invalid(type_name));
    }
    // A bare time of day is valid only for an instant type, and needs the
    // clock there too.
    if shape == TimestampText::TimeOnly {
        let without_zone = kind == TypedLiteralKind::Date
            || kind == TypedLiteralKind::TimestampNtz
            || (kind == TypedLiteralKind::Timestamp && conf.timestamp_ntz);
        if without_zone {
            return Err(invalid(type_name));
        }
        return Err(unsupported(
            "a timestamp literal that needs the query clock",
        ));
    }
    // One string cast through the kernels, as a microsecond instant or
    // NTZ wall-clock value.
    let cast_one = |value: &str, to: &arrow_schema::DataType, zone: &FixedOffset| {
        use arrow_array::StringArray;
        use arrow_array::cast::AsArray as _;
        use arrow_array::types::TimestampMicrosecondType;
        let input = StringArray::from(vec![value]);
        let opts = CastOptions {
            mode: KernelMode::Ansi,
            time_zone: zone,
        };
        let out = cast(&input, to, &opts).map_err(KrishivError::from)?;
        if out.is_null(0) {
            // ANSI casts raise rather than null malformed input; a NULL
            // here is unreachable.
            return Err(KrishivError::from(KrishivUnsupportedFeature {
                feature: "a timestamp literal".to_owned(),
            }));
        }
        Ok(out.as_primitive::<TimestampMicrosecondType>().value(0))
    };
    let ltz_arrow = || {
        arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into()))
    };
    let ntz_arrow = || arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None);
    let map_error = |e: KrishivError| {
        if e.error_class() == ErrorClass::KrishivUnsupportedFeature {
            AnalysisError::new(e, origin)
        } else {
            invalid(type_name)
        }
    };
    match kind {
        TypedLiteralKind::Date => {
            use arrow_array::cast::AsArray as _;
            use arrow_array::types::Date32Type;
            let input = arrow_array::StringArray::from(vec![value]);
            let opts = CastOptions {
                mode: KernelMode::Ansi,
                time_zone: &FixedOffset(0),
            };
            let out = cast(&input, &arrow_schema::DataType::Date32, &opts)
                .map_err(|e| map_error(KrishivError::from(e)))?;
            if out.is_null(0) {
                return Err(invalid(type_name));
            }
            Ok(Literal::Date(out.as_primitive::<Date32Type>().value(0)))
        }
        TypedLiteralKind::TimestampNtz => {
            let micros = cast_one(value, &ntz_arrow(), &FixedOffset(0)).map_err(map_error)?;
            Ok(Literal::TimestampNtz(micros))
        }
        TypedLiteralKind::TimestampLtz => {
            // Session-independent values (an explicit zone) evaluate under
            // any zone; only session-dependent ones read the session zone.
            if timestamp_has_zone(value) {
                let micros = cast_one(value, &ltz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::Timestamp(micros))
            } else {
                match conf.session_zone {
                    Some(off) => {
                        let micros =
                            cast_one(value, &ltz_arrow(), &FixedOffset(off)).map_err(map_error)?;
                        Ok(Literal::Timestamp(micros))
                    }
                    None => Err(unsupported(
                        "a timestamp literal in a session time zone other than a fixed offset",
                    )),
                }
            }
        }
        TypedLiteralKind::Timestamp if conf.timestamp_ntz => {
            // Under `TIMESTAMP_NTZ`, a value naming a zone is still an
            // instant; without one it is wall-clock time.
            if timestamp_has_zone(value) {
                let micros = cast_one(value, &ltz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::Timestamp(micros))
            } else {
                let micros = cast_one(value, &ntz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::TimestampNtz(micros))
            }
        }
        TypedLiteralKind::Timestamp => {
            if timestamp_has_zone(value) {
                let micros = cast_one(value, &ltz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::Timestamp(micros))
            } else {
                match conf.session_zone {
                    Some(off) => {
                        let micros =
                            cast_one(value, &ltz_arrow(), &FixedOffset(off)).map_err(map_error)?;
                        Ok(Literal::Timestamp(micros))
                    }
                    None => Err(unsupported(
                        "a timestamp literal in a session time zone other than a fixed offset",
                    )),
                }
            }
        }
    }
}

/// Whether `value` is one of Spark's special date/timestamp spellings
/// (`now`, `today`, ...), which depend on the session's clock and zone.
fn is_special(value: &str) -> bool {
    let v = value.trim().to_ascii_lowercase();
    ["epoch", "now", "today", "yesterday", "tomorrow"].contains(&v.as_str())
}

/// Where a node is bound: inside an aggregate call's arguments (`agg`),
/// and whether it is the function of a window expression (`window_fn`:
/// `sum(x)` in `sum(x) OVER w` is a window aggregate, not an aggregate of
/// the operator).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Ctx {
    agg: bool,
    window_fn: bool,
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
        let mut done: HashMap<(usize, Ctx), Bound> = HashMap::new();
        // Rewritten calls stay alive for the pass: `done` is keyed by
        // address.
        let mut rewritten_calls: Vec<Arc<Expr>> = Vec::new();
        let top = Ctx {
            agg: false,
            window_fn: false,
        };
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
    /// other call takes the input columns. `count(tbl.*)` with a single
    /// qualified star is `INVALID_USAGE_OF_STAR_WITH_TABLE_IDENTIFIER_IN_COUNT`
    /// unless `spark.sql.legacy.allowStarWithSingleTableIdentifierInCount`.
    fn star_args(&mut self, e: &Expr, cx: &Cx<'_>) -> Result<Option<Arc<Expr>>, AnalysisError> {
        let ExprNode::Function(f) = &e.node else {
            return Ok(None);
        };
        if f.name.eq_ignore_ascii_case("count")
            && let [a] = &f.args[..]
            && let ExprNode::Star(s) = &a.node
            && !s.qualifier.is_empty()
            && s.plan_id.is_none()
            && !self
                .ctx
                .conf
                .allow_star_with_single_table_identifier_in_count
            && self.qualifier_matches_scope(&s.qualifier, cx)
        {
            return Err(AnalysisError::new(
                InvalidUsageOfStarWithTableIdentifierInCount {
                    table_name: quoted(&s.qualifier),
                },
                a.origin,
            ));
        }
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

    /// Whether `qualifier` names the qualifier of an input attribute
    /// (a table or alias the scope knows).
    fn qualifier_matches_scope(&self, qualifier: &[String], cx: &Cx<'_>) -> bool {
        let cs = self.ctx.conf.case_sensitive;
        cx.scope.iter().any(|a| {
            let q = qualifier.len();
            a.qual.len() >= q
                && a.qual[a.qual.len() - q..]
                    .iter()
                    .zip(qualifier)
                    .all(|(x, y)| same(x, y, cs))
        })
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
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let outer = kids.iter().filter_map(|k| k.outer).min();
        let agg = kids.iter().find_map(|k| k.agg);
        let win = kids.iter().find_map(|k| k.win);
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
            win,
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
            ExprNode::TypedLiteral { kind, value } => {
                let literal = typed_literal(*kind, value, &self.ctx.conf, e.origin)?;
                let ty = ArgType {
                    nullable: false,
                    data_type: literal_type(&literal),
                };
                // Shown as the resolved literal: Spark's names show the
                // normalized value (`TIMESTAMP '2016-03-11 20:54:00'` for
                // `.000`), formatted in the session zone by `pretty`.
                Bound {
                    expr: Arc::new(Expr::new(ExprNode::Literal(literal)).with_origin(e.origin)),
                    ..keep(ty)
                }
            }
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
                            filter_expr: format!(
                                "\"{}\"",
                                pretty(&filter.shown(), self.ctx.conf.session_zone)
                            ),
                        },
                        e.origin,
                    ));
                }
                if let Some(filter) = kids.get(n)
                    && filter.win.is_some()
                {
                    return Err(super::windows::window_in_filter(filter, e.origin));
                }
                if let Some(which) = GroupingFn::from_name(&f.name) {
                    return self.grouping_call(which, e, &kids[..n], cx);
                }
                if let Some(op) = builtin_op(&f.name, n) {
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
                if !window_fn {
                    call_checks(&entry, e, kids, n)?;
                }
                if entry.id.name() == "count" && n == 0 {
                    return self.parameterless_count(e, &entry, kids, agg, window_fn);
                }
                // `assert_true(c)` is `assert_true(c, '<c>' is not true!)`:
                // Spark's constructor materializes the default message, so
                // the schema names it (`assert_true(true, 'true' is not
                // true!)`) and the kernel raises it.
                let mut owned: Vec<Bound>;
                let (kids, n) = if entry.id.name() == "assert_true" && n == 1 {
                    owned = kids.to_vec();
                    owned.insert(
                        1,
                        assert_message(&owned[0], e.origin, self.ctx.conf.session_zone),
                    );
                    (&owned[..], 2)
                } else {
                    (kids, n)
                };
                if let Some(which) = Conditional::of(&f.name) {
                    return self.conditional_call(which, &entry, e, f, kids);
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
                let shown = self.call_shown(&f.name, &expr, kids, &call_kids);
                Bound {
                    agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                        Some(e.origin)
                    } else {
                        agg
                    },
                    shown,
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
                    return Err(bad_cast(
                        bad,
                        e.origin,
                        &kids[0],
                        to,
                        mode,
                        self.ctx.conf.session_zone,
                    ));
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
            ExprNode::Window(w) => self.window(e, w, kids, cx)?,
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
            .map(|k| self.fold(&k.expr, self.ctx.conf.ansi))
            .collect();
        let literals: Vec<bool> = kids.iter().map(|k| fold::is_literal(&k.expr)).collect();
        // Rendered only if a rule's error quotes an argument.
        let zone = self.ctx.conf.session_zone;
        let sql = |i: usize| kids.get(i).map(|k| pretty(&k.shown(), zone));
        let given: Vec<ArgType> = kids.iter().map(|k| k.ty.clone()).collect();
        let ansi = self.ctx.conf.ansi;
        let loss = self.ctx.conf.decimal_precision_loss;
        let call = CallTypes::new(&given, &constants)
            .with_literals(&literals)
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
            .with_literals(&literals)
            .with_sql(&sql)
            .with_ansi(ansi)
            .with_decimal_loss(loss);
        let ty = (entry.result)(&call).map_err(|err| {
            // An error folding an argument is Spark's, cited at the argument.
            let origin = constants
                .iter()
                .zip(kids)
                .find_map(|(c, k)| match c {
                    Some(Constant::Error(folded)) if *folded == err => Some(k.expr.origin),
                    _ => None,
                })
                .unwrap_or(e.origin);
            AnalysisError::new(err, origin)
        })?;
        Ok((Self::folded_ntile(entry, &constants, call_kids), ty))
    }

    /// `count()` with no arguments: `WRONG_NUM_ARGS.WITH_SUGGESTION`,
    /// or `count(NULL)` — which counts no rows — when
    /// `spark.sql.legacy.allowParameterlessCount`. The output still shows
    /// as `count()`, as Spark names it.
    fn parameterless_count(
        &self,
        e: &Arc<Expr>,
        entry: &FunctionEntry,
        kids: &[Bound],
        agg: Option<Origin>,
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        if !self.ctx.conf.allow_parameterless_count {
            return Err(AnalysisError::new(
                WrongNumArgsWithSuggestion {
                    function_name: format!("`{}`", entry.id.name()),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: "0".to_owned(),
                    legacy_num: "0".to_owned(),
                    legacy_conf_key: "\"spark.sql.legacy.allowParameterlessCount\"".to_owned(),
                    legacy_conf_value: "\"true\"".to_owned(),
                },
                e.origin,
            ));
        }
        let null = Bound {
            expr: Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(e.origin)),
            ty: ArgType {
                data_type: Some(DataType::Null),
                nullable: true,
            },
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        };
        let (mut call_kids, ty) = self.typed_call(entry, e, std::slice::from_ref(&null))?;
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        // `kids[0]` is the `FILTER`, if any (`n == 0`).
        call_kids.extend(kids.first().cloned());
        let call = ResolvedFunction {
            id: entry.id.clone(),
            kind: entry.kind,
            args: call_kids[..1].iter().map(|k| Arc::clone(&k.expr)).collect(),
            distinct: match &e.node {
                ExprNode::Function(f) => f.distinct,
                _ => false,
            },
            filter: call_kids.get(1).map(|k| Arc::clone(&k.expr)),
            data_type,
            nullable: ty.nullable,
            deterministic: entry.deterministic,
        };
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(call))).with_origin(e.origin),
        );
        Ok(Bound {
            expr,
            ty,
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(e.origin)
            } else {
                agg
            },
            win: None,
            outer: None,
            lateral: Vec::new(),
            // Shown as written (`count()`), not as bound (`count(NULL)`).
            shown: Some(Arc::clone(e)),
        })
    }

    /// `ntile`'s bucket count as the literal it folds to (`ntile(1 + 1)`):
    /// the window operator reads a literal. The rule above checked it is an
    /// `int` constant.
    #[inline(never)]
    fn folded_ntile(
        entry: &FunctionEntry,
        constants: &[Option<Constant>],
        mut kids: Vec<Bound>,
    ) -> Vec<Bound> {
        if entry.id.name() == "ntile"
            && let (Some(Some(Constant::Integral(n))), Some(k)) =
                (constants.first(), kids.first_mut())
            && let Ok(n) = i32::try_from(*n)
        {
            k.expr = Arc::new(
                Expr::new(ExprNode::Literal(Literal::Integer(n))).with_origin(k.expr.origin),
            );
        }
        kids
    }

    /// The shown form of the call `expr` (see [`shown_call`]); `date_add(d,
    /// ' 12 ')` and `date_add(d, '1' || '2')` without ANSI show the day count
    /// as an integer (`date_add(d, 12)`): Spark folds the string it casts to
    /// `INT`. Out of line: it is on the path of every call.
    #[inline(never)]
    fn call_shown(
        &self,
        name: &str,
        expr: &Expr,
        kids: &[Bound],
        cast: &[Bound],
    ) -> Option<Arc<Expr>> {
        let days = self.legacy_day_count(name, kids);
        shown_call(expr, days.as_deref().unwrap_or(kids), cast)
    }

    /// The value of `e` as a constant (`fold::constant`) under this
    /// session's settings, with `ansi` for the arithmetic: shared with
    /// every other fold of the pass.
    pub(super) fn fold(&self, e: &Arc<Expr>, ansi: bool) -> Option<Constant> {
        let conf = fold::Conf::new(ansi, self.ctx.conf.decimal_precision_loss, self.ctx.profile);
        fold::constant(e, conf, &mut self.folds.borrow_mut())
    }

    fn legacy_day_count(&self, name: &str, kids: &[Bound]) -> Option<Vec<Bound>> {
        if self.ctx.conf.ansi
            || !["date_add", "dateadd", "date_sub"]
                .iter()
                .any(|n| name.eq_ignore_ascii_case(n))
        {
            return None;
        }
        let kid = kids.get(1)?;
        if kid.ty.data_type != Some(DataType::String) {
            return None;
        }
        let Some(Constant::String(text)) = self.fold(&kid.expr, false) else {
            return None;
        };
        let days = krishiv_functions::registry::legacy_day_count(&text)?;
        let mut shown = kids.to_vec();
        shown[1].shown = Some(Arc::new(Expr::new(ExprNode::Literal(Literal::String(
            days.to_string(),
        )))));
        Some(shown)
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
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
        })
    }

    /// A conditional function rewritten during analysis (Spark's
    /// `RuntimeReplaceable`): `if` and `nvl2` become `CASE WHEN`, `nvl` and
    /// `ifnull` become `coalesce`, `zeroifnull` becomes `coalesce` with a
    /// zero, and `nullif` / `nullifzero` become a same-typed `nullif` the
    /// evaluator runs once. `coalesce` keeps its name (the evaluator lowers
    /// it lazily), as does `nanvl` (both sides are `float`/`double`).
    fn conditional_call(
        &mut self,
        which: Conditional,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        let n = f.args.len();
        // The arity and the input checks quote the call as written.
        let (mut call_kids, ty) = self.typed_call(entry, e, &kids[..n])?;
        call_kids.extend(kids.get(n).cloned());
        let origin = e.origin;
        let node = |node: ExprNode| Arc::new(Expr::new(node).with_origin(origin));
        // The call as written (before casts), naming the output and quoting
        // errors: never the rewritten form.
        let shown = node(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args: kids[..n].iter().map(Bound::shown).collect(),
                distinct: f.distinct,
                filter: kids.get(n).map(Bound::shown),
                data_type: ty.data_type.clone().unwrap_or(DataType::Null),
                nullable: ty.nullable,
                deterministic: entry.deterministic,
            },
        )));
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        let resolved =
            |id: FunctionId, args: Vec<Arc<Expr>>, distinct: bool, filter: Option<Arc<Expr>>| {
                node(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id,
                        kind: FunctionKind::Scalar,
                        args,
                        distinct,
                        filter,
                        data_type: data_type.clone(),
                        nullable: ty.nullable,
                        deterministic: entry.deterministic,
                    },
                )))
            };
        let expr = match which {
            // `if(c, t, e)`: the branches' common type, as `when` types it.
            Conditional::If => {
                let args = call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                resolved(
                    FunctionId::Builtin(Builtin::When),
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `nvl(a, b)` / `ifnull(a, b)`: `coalesce`, cast likewise.
            Conditional::Coalesce => {
                let args = call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                self.coalesce_node(
                    origin,
                    e,
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                    &ty,
                )?
            }
            // `nvl2(x, t, e)`: `t` where `x` is not null, else `e`.
            Conditional::Nvl2 => {
                let cond = self.operator_call(Builtin::IsNotNull, origin, &call_kids[..1])?;
                let mut args = vec![cond.expr];
                args.extend(call_kids[1..n].iter().map(|k| Arc::clone(&k.expr)));
                resolved(
                    FunctionId::Builtin(Builtin::When),
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `nullif(a, b)`: `NULL` where `a = b`, else `a` (evaluated
            // once by the evaluator); `b` carries the analyzer's own `=`
            // cast, `a` keeps its type.
            Conditional::NullIf => {
                let other =
                    self.nullif_other(&call_kids[0], &call_kids[1], &shown, kids, f, origin)?;
                resolved(
                    entry.id.clone(),
                    vec![Arc::clone(&call_kids[0].expr), other],
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `nullifzero(x)`: `nullif(x, 0)`, named as written.
            Conditional::NullIfZero => {
                let zero = Bound {
                    expr: node(ExprNode::Literal(Literal::Integer(0))),
                    ty: arg_type(DataType::Integer, false),
                    agg: None,
                    win: None,
                    outer: None,
                    lateral: Vec::new(),
                    shown: None,
                };
                let other = self.nullif_other(&call_kids[0], &zero, &shown, kids, f, origin)?;
                resolved(
                    entry.id.clone(),
                    vec![Arc::clone(&call_kids[0].expr), other],
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `zeroifnull(x)`: `coalesce(x, 0)` at the met type.
            Conditional::ZeroIfNull => {
                let Some(to) = ty.data_type.as_ref() else {
                    return Err(unsupported("zeroifnull of an untyped NULL", origin));
                };
                let Some(zero) = zero_literal(to, origin) else {
                    return Err(unsupported(format!("zeroifnull of {}", to.sql()), origin));
                };
                let args = vec![Arc::clone(&call_kids[0].expr), zero];
                self.coalesce_node(
                    origin,
                    e,
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                    &ty,
                )?
            }
            // `nanvl(a, b)`: `b` where `a` is NaN, else `a`; both sides are
            // already `float`/`double`.
            Conditional::Nanvl => {
                let args = call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                resolved(
                    entry.id.clone(),
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
        };
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            expr,
            ty: ArgType {
                data_type: ty.data_type.clone(),
                nullable: ty.nullable,
            },
            agg: kids.iter().find_map(|k| k.agg),
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
            shown: Some(shown),
        })
    }

    /// A resolved `coalesce` call: `nvl` / `ifnull` / `zeroifnull` keep the
    /// result type their own casts met at.
    fn coalesce_node(
        &mut self,
        origin: Origin,
        e: &Arc<Expr>,
        args: Vec<Arc<Expr>>,
        distinct: bool,
        filter: Option<Arc<Expr>>,
        ty: &ArgType,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let id = match self
            .ctx
            .functions
            .lookup(&["coalesce".to_owned()], self.ctx.profile)
        {
            FunctionLookup::Found(entry) => entry.id,
            _ => return Err(unsupported("function coalesce", e.origin)),
        };
        Ok(Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id,
                    kind: FunctionKind::Scalar,
                    args,
                    distinct,
                    filter,
                    data_type: ty.data_type.clone().unwrap_or(DataType::Null),
                    nullable: ty.nullable,
                    deterministic: true,
                },
            )))
            .with_origin(origin),
        ))
    }

    /// The second argument of `nullif(a, b)`: `b` cast to the type `=`
    /// compares at, through the analyzer's own comparison coercion. In
    /// legacy mode `=` leaves two shapes to P2-T071d (a string against
    /// another type, a boolean against a number); there Spark casts the
    /// string to the other side's type and the number to `BOOLEAN`, which
    /// this mirrors so `nullif` keeps Spark's rows.
    fn nullif_other(
        &mut self,
        a: &Bound,
        b: &Bound,
        shown: &Arc<Expr>,
        kids: &[Bound],
        f: &FunctionCall,
        origin: Origin,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let args = [arg_of(a), arg_of(b)];
        let ansi = self.ctx.conf.ansi;
        match coerce::operator(
            Builtin::Eq,
            &args,
            ansi,
            self.ctx.conf.decimal_precision_loss,
        ) {
            Ok(typed) => Ok(self.implicit_cast(b, typed.casts[1].clone()).expr),
            Err(m) => {
                if !ansi && let Some(to) = legacy_nullif_cast(a, b) {
                    return Ok(self.implicit_cast(b, to).expr);
                }
                Err(self.mismatch(m, shown, kids, &f.name, origin))
            }
        }
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
        let zone = self.ctx.conf.session_zone;
        let sql_expr = format!("\"{}\"", pretty(&shown, zone));
        let texts = |v: Vec<String>| v.join(", ");
        if kids.len() != rel.out.len() {
            return Err(AnalysisError::new(
                DatatypeMismatchInSubqueryLengthMismatch {
                    sql_expr,
                    left_length: kids.len().to_string(),
                    left_columns: texts(
                        kids.iter()
                            .map(|k| format!("\"{}\"", pretty(&k.shown(), zone)))
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
                        pretty(
                            &super::ops::qualified(&k.shown(), cx.scope),
                            self.ctx.conf.session_zone
                        ),
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
                win: None,
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
        let zone = self.ctx.conf.session_zone;
        let sql_expr = format!("\"{}\"", pretty(call, zone));
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
                            kids.get(1)
                                .map(|k| pretty(&k.shown(), zone))
                                .unwrap_or_default()
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
                            .map(|k| pretty(&k.shown(), zone))
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
            win: None,
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
        // A name no column or outer reference has may be a session
        // variable (`v`, `session.v` or `system.session.v`): Spark
        // resolves columns and outer references first, then variables. A
        // name of another shape, or one naming no variable, stays an
        // unresolved column. The reference becomes the variable's current
        // value as a typed literal, so no physical, executor or plan-codec
        // change is needed; the expression keeps the spelling it was
        // written with for names and errors.
        if let Some(var) = self.ctx.variables.get(&c.name) {
            let Some(name) = c.name.last() else {
                debug_assert!(false, "a variable name is never empty");
                return Err(unresolved(&c.name, cx.scope, origin));
            };
            let col = BoundColumn {
                id: self.fresh(),
                name: name.clone(),
                data_type: var.data_type.clone(),
                nullable: matches!(var.value, Literal::Null(_)),
            };
            return Ok(Bound {
                expr: Arc::new(Expr::new(ExprNode::Literal(var.value.clone())).with_origin(origin)),
                ty: column(&col),
                agg: None,
                win: None,
                outer: None,
                lateral: Vec::new(),
                shown: Some(Arc::new(
                    Expr::new(ExprNode::Column(c.clone())).with_origin(origin),
                )),
            });
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
    crate::Builtin::from_name(name)
}

/// Spark's generated name for an unaliased expression (`toPrettySQL`):
/// `(a + 1)`, `count(1)`, `CASE WHEN ... END`. Built bottom-up with an
/// explicit stack, once per distinct node. Timestamp literals render in
/// `zone`, the session's resolved zone, through the same cast-to-string
/// kernels `CAST(ts AS STRING)` uses.
pub(super) fn pretty(e: &Expr, zone: Option<i32>) -> String {
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
        let mut s = pretty_node(x, &k, zone);
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

fn pretty_node(x: &Expr, k: &[&str], zone: Option<i32>) -> String {
    match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            c.name.clone()
        }
        ExprNode::Column(c) => c.name.last().cloned().unwrap_or_default(),
        ExprNode::Literal(v) => pretty_literal(v, zone),
        ExprNode::TypedLiteral { kind, value } => {
            let keyword = match kind {
                TypedLiteralKind::Date => "DATE",
                TypedLiteralKind::Timestamp => "TIMESTAMP",
                TypedLiteralKind::TimestampNtz => "TIMESTAMP_NTZ",
                TypedLiteralKind::TimestampLtz => "TIMESTAMP_LTZ",
            };
            format!("{keyword} '{value}'")
        }
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
        ExprNode::Window(w) => pretty_window(w, k, zone),
        ExprNode::Function(FunctionCall { name, distinct, .. }) => {
            call(name, Builtin::from_name(name), *distinct, k, None)
        }
        // Spark's `lag`/`lead` show `input, offset, default`, and
        // `nth_value` `input, offset` then ` ignore nulls` when set.
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.kind == FunctionKind::Window
                && matches!(f.id.name(), "lag" | "lead" | "nth_value")
                && f.filter.is_none() =>
        {
            let shown = if f.id.name() == "nth_value" { 2 } else { 3 };
            let n = shown.min(k.len());
            let text = call(f.id.name(), None, f.distinct, &k[..n], Some(&f.data_type));
            let ignore = matches!(
                f.args.get(shown).map(|a| &a.node),
                Some(ExprNode::Literal(Literal::Boolean(true)))
            );
            if ignore && f.id.name() == "nth_value" {
                format!("{text} ignore nulls")
            } else {
                text
            }
        }
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let op = match &f.id {
                FunctionId::Builtin(b) => Some(*b),
                FunctionId::Registry { name, .. } => builtin_op(name, f.args.len()),
            };
            let n = f.args.len().min(k.len());
            let text = call(f.id.name(), op, f.distinct, &k[..n], Some(&f.data_type));
            match k.get(n) {
                Some(filter) => format!("{text} FILTER (WHERE {filter})"),
                None => text,
            }
        }
    }
}

/// `f OVER (...)` as Spark names it: the function (`RANK()`,
/// `DENSE_RANK()` and `PERCENT_RANK()` upper case, as Spark's `RankLike`
/// writes them), then the window as `WindowSpecDefinition.sql` writes it.
/// `k` holds the children's names in [`Expr::for_each_child`] order.
fn pretty_window(w: &crate::WindowExpr, k: &[&str], zone: Option<i32>) -> String {
    use crate::{FrameBound, FrameUnits, WindowSpecRef};
    let function = match &w.function.node {
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.kind == FunctionKind::Window
                && ["rank", "dense_rank", "percent_rank"].contains(&f.id.name()) =>
        {
            format!("{}()", f.id.name().to_uppercase())
        }
        _ => k[0].to_owned(),
    };
    let spec = match &w.window {
        WindowSpecRef::Named(name) => return format!("{function} OVER {name}"),
        WindowSpecRef::Spec(spec) => spec,
    };
    let mut parts = Vec::new();
    let mut rest = k.get(1..).unwrap_or_default().iter();
    if !spec.partition_by.is_empty() {
        let names: Vec<&str> = rest
            .by_ref()
            .take(spec.partition_by.len())
            .copied()
            .collect();
        parts.push(format!("PARTITION BY {}", names.join(", ")));
    }
    if !spec.order_by.is_empty() {
        let names: Vec<String> = spec
            .order_by
            .iter()
            .zip(rest.by_ref())
            .map(|(o, n)| format!("{n} {}", order_sql(o)))
            .collect();
        parts.push(format!("ORDER BY {}", names.join(", ")));
    }
    let Some(f) = &spec.frame else {
        // An unresolved window, as Spark's errors show one.
        parts.push("unspecifiedframe$()".to_owned());
        return format!("{function} OVER ({})", parts.join(" "));
    };
    {
        let bound = |b: &FrameBound, rest: &mut std::slice::Iter<'_, &str>| match b {
            FrameBound::UnboundedPreceding => "UNBOUNDED PRECEDING".to_owned(),
            FrameBound::CurrentRow => "CURRENT ROW".to_owned(),
            FrameBound::UnboundedFollowing => "UNBOUNDED FOLLOWING".to_owned(),
            FrameBound::Offset(e) => {
                let own = rest.next().copied().unwrap_or_default();
                match crate::expr::negated(e) {
                    Some(n) => format!("{} PRECEDING", pretty(n, zone)),
                    None => format!("{own} FOLLOWING"),
                }
            }
        };
        let units = match f.units {
            FrameUnits::Rows => "ROWS",
            FrameUnits::Range => "RANGE",
        };
        let lower = bound(&f.lower, &mut rest);
        let upper = bound(&f.upper, &mut rest);
        parts.push(format!("{units} BETWEEN {lower} AND {upper}"));
    }
    format!("{function} OVER ({})", parts.join(" "))
}

/// `ASC NULLS FIRST` and the like (Spark's `SortOrder.sql` after the key).
pub(super) const fn order_sql(o: &crate::SortOrder) -> &'static str {
    use crate::{NullOrdering, SortDirection};
    match (o.direction, o.nulls) {
        (SortDirection::Ascending, NullOrdering::NullsFirst) => "ASC NULLS FIRST",
        (SortDirection::Ascending, NullOrdering::NullsLast) => "ASC NULLS LAST",
        (SortDirection::Descending, NullOrdering::NullsFirst) => "DESC NULLS FIRST",
        (SortDirection::Descending, NullOrdering::NullsLast) => "DESC NULLS LAST",
    }
}

fn call(
    name: &str,
    op: Option<Builtin>,
    distinct: bool,
    k: &[&str],
    data_type: Option<&DataType>,
) -> String {
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
        // Spark's own spellings of `if` and `positive`.
        _ if name.eq_ignore_ascii_case("if") && k.len() == 3 => {
            format!("(IF({}, {}, {}))", k[0], k[1], k[2])
        }
        _ if name.eq_ignore_ascii_case("positive") && k.len() == 1 => format!("(+ {})", k[0]),
        _ if name.eq_ignore_ascii_case("extract") && k.len() == 2 => {
            format!("extract({} FROM {})", k[0], k[1])
        }
        // `trim` family: one argument is `trim(src)` / `ltrim(src)` /
        // `rtrim(src)` / `btrim(...)`; two are Spark's `TRIM(...)` form.
        // The analyzer keeps `trim(trimStr, src)` (characters first, as
        // Spark's two-parameter call is) and `btrim(src, chars)` last.
        _ if name.eq_ignore_ascii_case("trim") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("trim({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("trim") && k.len() == 2 => {
            format!("TRIM(BOTH {} FROM {})", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("ltrim") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("ltrim({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("ltrim") && k.len() == 2 => {
            format!("TRIM(LEADING {} FROM {})", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("rtrim") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("rtrim({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("rtrim") && k.len() == 2 => {
            format!("TRIM(TRAILING {} FROM {})", k[0], k[1])
        }
        // Spark fills in the missing arguments: `substr(s, p)` is
        // `substr(s, p, 2147483647)`, `lpad(s, n)` is `lpad(s, n, ' ')`
        // (`X'00'` for binary), `replace(s, search)` is
        // `replace(s, search, '')`, `round(x)` is `round(x, 0)` and `log(x)`
        // is `LOG(E(), x)`.
        _ if name.eq_ignore_ascii_case("substr") && k.len() == 2 => {
            format!("substr({}, {}, 2147483647)", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("substring") && k.len() == 2 => {
            format!("substring({}, {}, 2147483647)", k[0], k[1])
        }
        _ if (name.eq_ignore_ascii_case("lpad") || name.eq_ignore_ascii_case("rpad"))
            && k.len() == 2 =>
        {
            let pad = if data_type == Some(&DataType::Binary) {
                "X'00'"
            } else {
                " "
            };
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{}, {}, {pad})", name.to_lowercase(), k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("replace") && k.len() == 2 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("replace({d}{}, {}, )", k[0], k[1])
        }
        _ if (name.eq_ignore_ascii_case("round") || name.eq_ignore_ascii_case("bround"))
            && k.len() == 1 =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{}, 0)", name.to_lowercase(), k[0])
        }
        _ if name.eq_ignore_ascii_case("ceil") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("CEIL({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("floor") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("FLOOR({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("log") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("LOG({d}E(), {})", k[0])
        }
        _ if name.eq_ignore_ascii_case("log") && k.len() == 2 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("LOG({d}{}, {})", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("ln") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("ln({d}{})", k[0])
        }
        // Spark's name of `first(x, ignoreNulls)` leaves the flag out.
        _ if k.len() == 2
            && matches!(k[1], "true" | "false")
            && ["first", "last", "first_value", "last_value", "any_value"]
                .iter()
                .any(|n| name.eq_ignore_ascii_case(n)) =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), k[0])
        }
        // The bitwise operators show infix (`(20181117 >> 2)`), `~`
        // prefix; `to_varchar` shows as `to_char`.
        _ if k.len() == 2 && matches!(name, "&" | "|" | "^" | "<<" | ">>" | ">>>") => infix(name),
        _ if k.len() == 1 && name == "~" => format!("~{}", k[0]),
        _ if name.eq_ignore_ascii_case("to_varchar") => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("to_char({d}{})", k.join(", "))
        }
        // Single-argument math (`sec`, `csc`, `cot`, `exp`, ...) and
        // two-argument math (`power`, `atan2`, `hypot`) print upper case,
        // as Spark's `UnaryMathExpression` / `BinaryMathExpression` do;
        // aliases (`pow`, `sign`) and the rest print lower case.
        _ if is_upper_math(name) => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_uppercase(), k.join(", "))
        }
        _ => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), k.join(", "))
        }
    }
}

/// Whether Spark prints `name` upper case: the `UnaryMathExpression` /
/// `BinaryMathExpression` functions without an alias (`sec` is `SEC`,
/// `power` is `POWER`), while aliases (`pow`, `sign`) and `ln` stay lower.
fn is_upper_math(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "sqrt"
            | "cbrt"
            | "exp"
            | "expm1"
            | "log10"
            | "log2"
            | "log1p"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "cot"
            | "csc"
            | "sec"
            | "degrees"
            | "radians"
            | "rint"
            | "signum"
            | "power"
            | "atan2"
            | "hypot"
            | "log"
            | "pi"
            | "e"
    )
}

fn pretty_literal(v: &Literal, zone: Option<i32>) -> String {
    match v {
        // A URL's credentials never reach a name (ADR 0038).
        Literal::String(s) => krishiv_common::secret::SecretUrl::parse(s)
            .map_or_else(|| s.clone(), |url| url.to_string()),
        Literal::Null(_) => "NULL".to_owned(),
        Literal::Boolean(v) => v.to_string(),
        Literal::Byte(v) => v.to_string(),
        Literal::Short(v) => v.to_string(),
        Literal::Integer(v) => v.to_string(),
        Literal::Long(v) => v.to_string(),
        Literal::Float(v) if v.is_finite() => crate::spark_fmt::float(*v),
        Literal::Float(v) => format!("CAST('{}' AS FLOAT)", special_float(f64::from(*v))),
        Literal::Double(v) if v.is_finite() => crate::spark_fmt::double(*v),
        Literal::Double(v) => format!("CAST('{}' AS DOUBLE)", special_float(*v)),
        Literal::Decimal {
            unscaled,
            data_type,
        } => crate::spark_fmt::decimal(*unscaled, u32::from(data_type.scale())),
        Literal::Binary(b) => {
            use std::fmt::Write as _;
            let mut out = String::from("X'");
            for byte in b {
                let _ = write!(out, "{byte:02X}");
            }
            out.push('\'');
            out
        }
        Literal::Date(days) => crate::spark_fmt::date(*days),
        // An instant renders in the session zone (a fixed offset; UTC when
        // unset), as Spark's `toPrettySQL` of a timestamp literal does.
        Literal::Timestamp(us) => crate::spark_fmt::timestamp(
            us.saturating_add(i64::from(zone.unwrap_or(0)).saturating_mul(1_000_000)),
            false,
        ),
        Literal::TimestampNtz(us) => crate::spark_fmt::timestamp(*us, true),
        _ => v.to_string(),
    }
}

/// `NaN`, `Infinity` or `-Infinity`, as Spark spells them in a name.
fn special_float(v: f64) -> &'static str {
    if v.is_nan() {
        "NaN"
    } else if v > 0.0 {
        "Infinity"
    } else {
        "-Infinity"
    }
}

/// How Spark's canonicalisation (`Expression.canonicalized`) orders a call's
/// operands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Canon {
    /// Operands in order.
    Ordered,
    /// `+ * AND OR`: nested calls of the same operator are flattened and
    /// the operands sorted (`greatest`/`least` and `= <=>` sort too, with
    /// nothing to flatten for the latter).
    Commutative { flatten: bool },
    /// `a > b` is `b < a`, `a >= b` is `b <= a`: the call is renamed and
    /// its operands swapped.
    Flipped(Builtin),
}

fn canon(f: &ResolvedFunction) -> Canon {
    match &f.id {
        FunctionId::Builtin(Builtin::Add | Builtin::Multiply | Builtin::And | Builtin::Or) => {
            Canon::Commutative { flatten: true }
        }
        FunctionId::Builtin(Builtin::Eq | Builtin::EqNullSafe) => {
            Canon::Commutative { flatten: false }
        }
        FunctionId::Builtin(Builtin::Gt) => Canon::Flipped(Builtin::Lt),
        FunctionId::Builtin(Builtin::GtEq) => Canon::Flipped(Builtin::LtEq),
        FunctionId::Registry { name, .. } if matches!(name.as_str(), "greatest" | "least") => {
            Canon::Commutative { flatten: false }
        }
        _ => Canon::Ordered,
    }
}

/// A semantic key: equal for expressions that compute the same value,
/// whatever their origins, column spellings or (top-level) alias, and
/// canonicalised as Spark's `semanticEquals` is (commutative operands
/// sorted, `a > b` as `b < a`).
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
        let mut name = None;
        let mut order = Canon::Ordered;
        let mut b = match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => format!("c{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => format!("o{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                order = canon(f);
                let id = match order {
                    Canon::Flipped(to) => to.name(),
                    _ => f.id.name(),
                };
                name = Some(f);
                format!("f{id}:{:?}:{}", f.kind, f.distinct)
            }
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => format!("a{}", output.id.0),
            ExprNode::Literal(v) => format!("l{v:?}"),
            ExprNode::Cast { to, mode, .. } => format!("x{}:{mode:?}", to.simple_string()),
            ExprNode::Subquery { plan, .. } => format!("s{}", addr(&**plan)),
            // The children (function, partitioning, ordering, offsets) are
            // keyed below; the shape says which is which.
            ExprNode::Window(w) => match &w.window {
                crate::WindowSpecRef::Named(name) => format!("wn{name}"),
                crate::WindowSpecRef::Spec(spec) => format!(
                    "w{}:{:?}:{:?}",
                    spec.partition_by.len(),
                    spec.order_by
                        .iter()
                        .map(|o| (o.direction, o.nulls))
                        .collect::<Vec<_>>(),
                    spec.frame.as_ref().map(|f| (
                        f.units,
                        frame_shape(&f.lower),
                        frame_shape(&f.upper)
                    )),
                ),
            },
            other => format!("u{other:?}"),
        }
        .into_bytes();
        let mut hashes: Vec<u128> = Vec::new();
        match (order, name) {
            (Canon::Commutative { flatten }, Some(f)) => {
                // Flattening reads a nested operator's operands again for each
                // way it is shared, so it stops at a bound; past it the
                // operands are the call's own (equal expressions written the
                // same way still match).
                let mut budget = FLATTEN_BUDGET;
                if !(flatten && collect_operands(x, f, memo, &mut hashes, &mut budget)) {
                    hashes.clear();
                    hashes.extend(
                        kids.iter()
                            .map(|c| memo.get(&addr(*c)).copied().unwrap_or(0)),
                    );
                }
                hashes.sort_unstable();
            }
            (Canon::Flipped(_), _) => {
                hashes.extend(
                    kids.iter()
                        .rev()
                        .map(|c| memo.get(&addr(*c)).copied().unwrap_or(0)),
                );
            }
            _ => hashes.extend(
                kids.iter()
                    .map(|c| memo.get(&addr(*c)).copied().unwrap_or(0)),
            ),
        }
        for h in hashes {
            b.extend_from_slice(&h.to_le_bytes());
        }
        memo.insert(addr(x), XxHash3_128::oneshot(&b));
    }
    memo.get(&addr(e)).copied().unwrap_or(0)
}

/// The most nodes [`collect_operands`] reads for one call.
const FLATTEN_BUDGET: usize = 64;

/// Adds the operand hashes of `x`, a call of `f`, to `out`: those of nested
/// calls of the same operator and result type are gathered too (Spark's
/// `gatherCommutative`). `false` when `budget` nodes were not enough, which
/// keeps a shared operand (a DAG that is exponentially large as a tree) from
/// being read once per path.
fn collect_operands(
    x: &Expr,
    f: &ResolvedFunction,
    memo: &HashMap<usize, u128>,
    out: &mut Vec<u128>,
    budget: &mut usize,
) -> bool {
    let mut done = true;
    x.for_each_child(&mut |c| {
        if !done {
            return;
        }
        let Some(left) = budget.checked_sub(1) else {
            done = false;
            return;
        };
        *budget = left;
        if let ExprNode::Resolved(ResolvedExpr::Function(g)) = &c.node
            && g.id == f.id
            && g.data_type == f.data_type
        {
            done = collect_operands(c, g, memo, out, budget);
        } else {
            out.push(memo.get(&addr(&**c)).copied().unwrap_or(0));
        }
    });
    done
}

/// A frame bound's kind, for [`semantic_key`] (an offset's value is a
/// child).
const fn frame_shape(b: &crate::FrameBound) -> u8 {
    match b {
        crate::FrameBound::UnboundedPreceding => 0,
        crate::FrameBound::CurrentRow => 1,
        crate::FrameBound::UnboundedFollowing => 2,
        crate::FrameBound::Offset(_) => 3,
    }
}
