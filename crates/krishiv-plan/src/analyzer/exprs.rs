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
    UnresolvedColumnWithoutSuggestion, UnresolvedRoutine, UnsupportedGroupingExpression,
    WrongNumArgsWithSuggestion, WrongNumArgsWithoutSuggestion,
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
    fn carry_markers(&mut self, kids: &[Self]) {
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

/// The default message of `assert_true(cond)`: `'<cond>' is not true!`,
/// with casts stripped as Spark's `toPrettySQL` strips them.
fn assert_message(cond: &Bound, origin: Origin, zone: Option<SessionZone>) -> Bound {
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

/// The replacement characters `mask` takes when a call omits them, which
/// Spark's constructors write into the call (so the schema names them):
/// `mask(s)` is `mask(s, 'X', 'x', 'n', NULL)`.
const MASK_DEFAULTS: [Option<&str>; 4] = [Some("X"), Some("x"), Some("n"), None];

/// The position of the options argument of the CSV and JSON functions.
fn options_index(name: &str) -> Option<usize> {
    match name {
        "from_csv" | "from_json" | "from_xml" => Some(2),
        "to_csv" | "schema_of_csv" | "to_json" | "schema_of_json" | "to_xml" | "schema_of_xml" => {
            Some(1)
        }
        _ => None,
    }
}

/// The pairs of a `map('k', 'v', ...)` call whose arguments are all `STRING`
/// literals, if `e` is one (a call with any other argument keeps its call).
fn literal_map_pairs(e: &Expr) -> Option<Vec<(String, String)>> {
    let ExprNode::Resolved(ResolvedExpr::Function(call)) = &e.node else {
        return None;
    };
    if call.id.name() != "map" || call.args.len() % 2 != 0 || call.filter.is_some() {
        return None;
    }
    let text = |x: &Arc<Expr>| match &x.node {
        ExprNode::Literal(Literal::String(s)) => Some(s.clone()),
        _ => None,
    };
    call.args
        .chunks(2)
        .map(|kv| Some((text(&kv[0])?, text(&kv[1])?)))
        .collect()
}

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
    /// `decode(expr, search, result, ...)`: a `CASE` over null-safe
    /// equality (the two-argument form is the charset decoder, a function).
    Decode,
}

impl Conditional {
    /// Which conditional `name` is, ASCII case-insensitively (function
    /// names are); `None` for any other function, `coalesce` included.
    fn of(name: &str, args: usize) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "decode" if args >= 3 => Self::Decode,
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

/// Whether `entry` is a built-in (a canonical operator or a registry
/// function), rather than a session UDF. Every special-cased built-in path
/// (`assert_true`'s default message, the `Conditional` rewrites, `ntile`'s
/// folding) goes through this one rule: built-ins win over a same-name
/// session UDF for unqualified names, and a qualified call to the session
/// UDF reaches the UDF.
fn is_builtin_entry(entry: &FunctionEntry) -> bool {
    !matches!(entry.id, FunctionId::Udf { .. })
}

/// The target type of a cast-named function call (`float(x)`,
/// `int(x)`, ...), ASCII case-insensitively; `None` for any other
/// function. Spark analyzes these names as `CAST(x AS target)` (its
/// `RuntimeReplaceable`): `decimal` is `decimal(10,0)`.
fn cast_target(name: &str) -> Option<DataType> {
    Some(match name.to_ascii_lowercase().as_str() {
        "boolean" => DataType::Boolean,
        "tinyint" => DataType::Byte,
        "smallint" => DataType::Short,
        "int" => DataType::Integer,
        "bigint" => DataType::Long,
        "float" => DataType::Float,
        "double" => DataType::Double,
        "decimal" => DataType::Decimal(DecimalType::USER_DEFAULT),
        "string" => DataType::String,
        "binary" => DataType::Binary,
        "date" => DataType::Date,
        "timestamp" => DataType::Timestamp,
        "time" => DataType::Time { precision: 6 },
        _ => return None,
    })
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
    zone: Option<SessionZone>,
) -> AnalysisError {
    let shown = ExprNode::Cast {
        expr: arg.shown(),
        to: to.clone(),
        mode,
    };
    let sql_expr = format!("\"{}\"", pretty(&Expr::new(shown), zone));
    let error = cast_mismatch(
        bad,
        sql_expr,
        quoted_type(arg.ty.data_type.as_ref()),
        quoted_type(Some(to)),
    );
    AnalysisError::new(error, origin)
}

/// The error for a cast of `arg` to `to` that `bad` says is refused;
/// `sql_expr` is the quoted expression Spark shows in it.
pub(super) fn cast_error(
    bad: BadCast,
    sql_expr: String,
    arg: &Bound,
    to: &DataType,
) -> KrishivError {
    cast_mismatch(
        bad,
        sql_expr,
        quoted_type(arg.ty.data_type.as_ref()),
        quoted_type(Some(to)),
    )
}

/// Why a cast fails: the error of [`BadCast`], quoting `sql_expr` (the
/// expression Spark's message names) with the source and target types.
/// A `CAST` quotes the cast; a cast-named function (`float(x)`, ...)
/// quotes its argument, as Spark does.
fn cast_mismatch(
    bad: BadCast,
    sql_expr: String,
    src_type: String,
    target_type: String,
) -> KrishivError {
    match bad {
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
    }
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

/// Whether `qualifier` matches the trailing parts of scope qualifier `qual`,
/// whole identifier by whole identifier (case-folded unless
/// `case_sensitive`). An empty qualifier matches everything; `t` matches
/// `db.t`; `db.t` matches `db.t` and `catalog.db.t` but not `t` alone or
/// `xdb.t`.
fn qualifier_matches(qual: &[String], qualifier: &[String], case_sensitive: bool) -> bool {
    let q = qualifier.len();
    qual.len() >= q
        && qual[qual.len() - q..]
            .iter()
            .zip(qualifier.iter())
            .all(|(x, y)| same(x, y, case_sensitive))
}

fn column(c: &BoundColumn) -> ArgType {
    arg_type(c.data_type.clone(), c.nullable)
}

/// The type of a value as typing sees it: `VOID` is an untyped `NULL`
/// (`data_type: None`) everywhere but in plans and schemas.
pub(super) fn arg_type(data_type: DataType, nullable: bool) -> ArgType {
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
        Literal::Map(_) => DataType::Map(
            krishiv_types::MapType::new(DataType::String, DataType::String, false).into(),
        ),
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
        // A time literal is resolved by the caller, as a cast.
        TypedLiteralKind::Time => return Err(unsupported("a TIME literal here")),
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
    let cast_one = |value: &str, to: &arrow_schema::DataType, zone: &dyn TimeZoneRules| {
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
        TypedLiteralKind::Time => Err(unsupported("a TIME literal here")),
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
                    Some(zone) => {
                        let micros = cast_one(value, &ltz_arrow(), &zone).map_err(map_error)?;
                        Ok(Literal::Timestamp(micros))
                    }
                    None => Err(unsupported(
                        "a timestamp literal in an unresolvable session time zone",
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
                    Some(zone) => {
                        let micros = cast_one(value, &ltz_arrow(), &zone).map_err(map_error)?;
                        Ok(Literal::Timestamp(micros))
                    }
                    None => Err(unsupported(
                        "a timestamp literal in an unresolvable session time zone",
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

/// What [`strength_reduce_comparison`] decided for an integral expression
/// compared with a decimal literal.
enum ComparisonReduction {
    /// Not an integral-vs-decimal-literal comparison; bind as usual.
    Keep,
    /// The decimal literal at `index` becomes the `BIGINT` literal `value`.
    Narrow { index: usize, value: i64 },
    /// The literal is outside `BIGINT` range and decides the comparison.
    Decided(bool),
}

/// Spark's strength reduction for an integral expression compared with a
/// decimal literal (`DecimalPrecisionTypeCoercion.integralAndDecimalLiteral`
/// in Spark 4.2, which runs before the implicit casts are inserted). With
/// the literal on the right, `i > d` is `i > floor(d)`, `i >= d` is
/// `i >= ceil(d)`, `i < d` is `i < ceil(d)` and `i <= d` is `i <= floor(d)`;
/// with the literal on the left the bounds swap sides (`d > i` is
/// `ceil(d) > i`, `d >= i` is `floor(d) >= i`, `d < i` is `floor(d) < i` and
/// `d <= i` is `ceil(d) <= i`). A literal below `BIGINT` range or above it
/// decides the comparison outright. The output name shows the rewritten
/// comparison (`select 1 > 0.00001` names its column `(1 > 0)`).
fn strength_reduce_comparison(op: Builtin, left: &Bound, right: &Bound) -> ComparisonReduction {
    /// The value of a bound decimal literal, if the bound expression is one:
    /// its unscaled value and scale.
    fn decimal(b: &Bound) -> Option<(i128, u32)> {
        // Only a written decimal literal: a session variable binds its value
        // as a literal too but keeps its name (`shown`), and Spark's rule
        // matches `DecimalLiteral`, not `VariableReference`.
        if b.shown.is_some() {
            return None;
        }
        match &b.expr.node {
            ExprNode::Literal(Literal::Decimal {
                unscaled,
                data_type,
            }) => Some((*unscaled, u32::from(data_type.scale()))),
            _ => None,
        }
    }
    /// Whether the bound expression is integral, as Spark's
    /// `IntegralTypeExpression` matches (`BYTE`/`SHORT`/`INT`/`LONG`).
    fn integral(b: &Bound) -> bool {
        matches!(
            b.ty.data_type,
            Some(DataType::Byte | DataType::Short | DataType::Integer | DataType::Long)
        )
    }
    // The decimal literal's side, when the other side is integral. Two
    // decimal literals never match (Spark widens those instead).
    let (index, (unscaled, scale)) = match (decimal(left), decimal(right)) {
        (Some(d), None) if integral(right) => (0, d),
        (None, Some(d)) if integral(left) => (1, d),
        _ => return ComparisonReduction::Keep,
    };
    // `floor` and `ceil` of the decimal value, exactly: the scale is at most
    // 38, so `10^scale` fits an `i128`, and the divisions are exact. Any
    // overflow (unreachable for a decimal literal) keeps the comparison.
    let Some(pow) = 10i128.checked_pow(scale) else {
        return ComparisonReduction::Keep;
    };
    let floor = unscaled.div_euclid(pow);
    let Some(ceil) = unscaled
        .checked_neg()
        .map(|n| n.div_euclid(pow))
        .and_then(i128::checked_neg)
    else {
        return ComparisonReduction::Keep;
    };
    // Spark's `DecimalLiteral.largerThanLargestLong` /
    // `smallerThanSmallestLong`: exact, as both bounds are integers. A
    // literal above `BIGINT` range exceeds every integral value, so only
    // the comparisons that the other side cannot reach fail.
    let long_max = i128::from(i64::MAX);
    let long_min = i128::from(i64::MIN);
    if ceil > long_max {
        return ComparisonReduction::Decided(matches!(
            (op, index),
            (Builtin::Gt | Builtin::GtEq, 0) | (Builtin::Lt | Builtin::LtEq, 1)
        ));
    }
    // A literal below `BIGINT` range is exceeded by every integral value.
    if floor < long_min {
        return ComparisonReduction::Decided(matches!(
            (op, index),
            (Builtin::Gt | Builtin::GtEq, 1) | (Builtin::Lt | Builtin::LtEq, 0)
        ));
    }
    // The narrowed bound fits a `BIGINT`: `floor`/`ceil` lie between two
    // consecutive integers around a value within `BIGINT` range.
    let narrowed = match (op, index) {
        (Builtin::Gt | Builtin::LtEq, 1) | (Builtin::GtEq | Builtin::Lt, 0) => floor,
        _ => ceil,
    };
    let Ok(value) = i64::try_from(narrowed) else {
        return ComparisonReduction::Keep;
    };
    ComparisonReduction::Narrow { index, value }
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

    /// A call with `*` arguments, rewritten: `count(*)` is `count(1)`, any
    /// other call takes the input columns. `count(tbl.*)` with a single
    /// qualified star is `INVALID_USAGE_OF_STAR_WITH_TABLE_IDENTIFIER_IN_COUNT`
    /// unless `spark.sql.legacy.allowStarWithSingleTableIdentifierInCount`.
    /// A qualified star of a struct
    /// column in `struct`/`named_struct` (the `STRUCT(t.*)` syntax) takes
    /// the struct's fields instead; in `named_struct` it replaces a
    /// preceding placeholder name from that same syntax.
    fn star_args(&mut self, e: &Expr, cx: &Cx<'_>) -> Result<Option<Arc<Expr>>, AnalysisError> {
        let ExprNode::Function(f) = &e.node else {
            return Ok(None);
        };
        if f.unqualified()
            .is_some_and(|n| n.eq_ignore_ascii_case("count"))
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
        let count_star = f
            .unqualified()
            .is_some_and(|n| n.eq_ignore_ascii_case("count"))
            && matches!(&f.args[..], [a] if matches!(&a.node, ExprNode::Star(s) if s.qualifier.is_empty() && s.plan_id.is_none()));
        let struct_star =
            f.name.eq_ignore_ascii_case("struct") || f.name.eq_ignore_ascii_case("named_struct");
        let mut args = Vec::new();
        let mut arg_names = Vec::new();
        for (i, a) in f.args.iter().enumerate() {
            let name = f.arg_names.get(i).cloned().flatten();
            match &a.node {
                _ if count_star => {
                    args.push(Arc::new(
                        Expr::literal(Literal::Integer(1)).with_origin(a.origin),
                    ));
                    arg_names.push(name);
                }
                ExprNode::Star(s) if struct_star => {
                    let is_named = f.name.eq_ignore_ascii_case("named_struct");
                    // A `STRUCT(t.*)` placeholder name precedes the star in
                    // a `named_struct` the syntax lowered: the expansion
                    // carries its own names.
                    if is_named
                        && matches!(
                            args.last().map(|a| &a.node),
                            Some(ExprNode::Literal(Literal::String(_)))
                        )
                    {
                        args.pop();
                        arg_names.pop();
                    }
                    for (name, field) in self.struct_star(s, cx, a.origin)? {
                        if is_named {
                            args.push(Arc::new(
                                Expr::new(ExprNode::Literal(Literal::String(name)))
                                    .with_origin(a.origin),
                            ));
                            arg_names.push(None);
                        }
                        args.push(field);
                        arg_names.push(None);
                    }
                }
                ExprNode::Star(s) => {
                    for attr in self.expand(s, cx, a.origin)? {
                        args.push(Arc::new(
                            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col)))
                                .with_origin(a.origin),
                        ));
                        arg_names.push(None);
                    }
                }
                _ => {
                    args.push(Arc::clone(a));
                    arg_names.push(name);
                }
            }
        }
        // An all-positional call stays all-positional.
        if arg_names.iter().all(Option::is_none) {
            arg_names.clear();
        }
        let call = FunctionCall {
            within_group: Vec::new(),
            args,
            arg_names,
            ..f.clone()
        };
        Ok(Some(Arc::new(
            Expr::new(ExprNode::Function(call)).with_origin(e.origin),
        )))
    }

    /// Whether `qualifier` names the qualifier of an input attribute
    /// (a table or alias the scope knows): it must match the scope
    /// qualifier's trailing parts exactly, whole identifier by whole
    /// identifier (case-folded unless `case_sensitive`).
    fn qualifier_matches_scope(&self, qualifier: &[String], cx: &Cx<'_>) -> bool {
        let cs = self.ctx.conf.case_sensitive;
        cx.scope
            .iter()
            .any(|a| qualifier_matches(&a.qual, qualifier, cs))
    }

    /// The fields of the struct `s` (a qualified `t.*`) names, as resolved
    /// field expressions. An unqualified star, a star with `EXCEPT` or a
    /// DataFrame star, a qualifier no column has, or a qualifier of a
    /// non-struct column falls back to the input columns
    /// ([`Self::expand_star_as_columns`]): an unqualified star is all of
    /// them, while a qualified one that is not a struct errors there when
    /// nothing matches.
    fn struct_star(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<(String, Arc<Expr>)>, AnalysisError> {
        let st = match s.qualifier[..] {
            [] => None,
            [..] if !s.except.is_empty() || s.plan_id.is_some() => None,
            _ => match self.find(cx.scope, &s.qualifier, origin, false)? {
                Some((a, used)) if used == s.qualifier.len() => match &a.col.data_type {
                    DataType::Struct(st) => Some((a.clone(), st.clone())),
                    _ => return self.columns_with_names(s, cx, origin),
                },
                _ => return self.columns_with_names(s, cx, origin),
            },
        };
        let Some((attr, st)) = st else {
            return self.columns_with_names(s, cx, origin);
        };
        self.used(&attr);
        Ok(st
            .fields()
            .iter()
            .enumerate()
            .map(|(ordinal, field)| {
                let nullable = field.nullable() || attr.col.nullable;
                let name = field.name().to_owned();
                let field = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                        base: Arc::new(
                            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col.clone())))
                                .with_origin(origin),
                        ),
                        ordinal,
                        name: name.clone(),
                        data_type: field.data_type().clone(),
                        nullable,
                    }))
                    .with_origin(origin),
                );
                (name, field)
            })
            .collect())
    }

    /// [`Self::expand`] as resolved column expressions with their names.
    fn columns_with_names(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<(String, Arc<Expr>)>, AnalysisError> {
        Ok(self
            .expand(s, cx, origin)?
            .into_iter()
            .map(|attr| {
                let name = attr.col.name.clone();
                let expr = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col)))
                        .with_origin(origin),
                );
                (name, expr)
            })
            .collect())
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
                .filter(|a| qualifier_matches(&a.qual, &s.qualifier, cs))
                // The columns of a select-list generator are read by its own
                // star, not by `*`.
                .filter(|a| {
                    !s.qualifier.is_empty()
                        || !a
                            .qual
                            .iter()
                            .any(|q| q.starts_with(crate::GENERATE_QUALIFIER_PREFIX))
                })
                .cloned()
                .collect(),
        };
        let out = match s.position {
            Some(n) if (1..=out.len()).contains(&n) => vec![out[n - 1].clone()],
            Some(n) => {
                return Err(AnalysisError::new(
                    GroupByPosOutOfRange {
                        index: n.to_string(),
                        size: out.len().to_string(),
                    },
                    origin,
                ));
            }
            None => out,
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
        if s.except.is_empty() {
            return Ok(out);
        }
        self.apply_except(s, &out, origin)
    }

    /// What `* EXCEPT (cols)` drops from `out` (the star's own expansion).
    /// Each entry names one of those columns, bare or qualified with the
    /// table (as in `ids.id`); longer paths name nested struct fields,
    /// which need struct support (`KRISHIV_UNSUPPORTED_FEATURE`), and
    /// unknown names are `UNRESOLVED_COLUMN`. Overlapping entries are
    /// `EXCEPT_OVERLAPPING_COLUMNS`, checked before anything resolves (as
    /// Spark checks them).
    fn apply_except(
        &mut self,
        s: &Star,
        out: &[Attr],
        origin: Origin,
    ) -> Result<Vec<Attr>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let folded: Vec<Vec<String>> = s
            .except
            .iter()
            .map(|p| p.iter().map(|x| fold_case(x, cs)).collect())
            .collect();
        for (i, a) in folded.iter().enumerate() {
            for (j, b) in folded.iter().enumerate().take(i) {
                if is_prefix(b, a) || is_prefix(a, b) {
                    return Err(AnalysisError::new(
                        ExceptOverlappingColumns {
                            columns: format!(
                                "{}, {}",
                                s.except[j].join("."),
                                s.except[i].join(".")
                            ),
                        },
                        origin,
                    ));
                }
            }
        }
        let mut dropped: Vec<(AttrId, String)> = Vec::new();
        for path in &s.except {
            let raw = path.join(".");
            match self.except_target(path, out, cs)? {
                ExceptTarget::Top(a) => {
                    if let Some((_, first)) = dropped.iter().find(|(id, _)| *id == a.col.id) {
                        return Err(AnalysisError::new(
                            ExceptOverlappingColumns {
                                columns: format!("{first}, {raw}"),
                            },
                            origin,
                        ));
                    }
                    dropped.push((a.col.id, raw));
                }
                ExceptTarget::Nested => {
                    return Err(AnalysisError::new(
                        KrishivUnsupportedFeature {
                            feature: format!("SELECT * EXCEPT with nested column `{raw}`"),
                        },
                        origin,
                    ));
                }
                ExceptTarget::Missing => {
                    return Err(unresolved(path, out, origin));
                }
            }
        }
        let gone: HashSet<AttrId> = dropped.into_iter().map(|(id, _)| id).collect();
        Ok(out
            .iter()
            .filter(|a| !gone.contains(&a.col.id))
            .cloned()
            .collect())
    }

    /// Records a column of a securable as referenced.
    pub(super) fn used(&mut self, a: &Attr) {
        if let Some((i, name)) = &a.source
            && let Some(set) = self.columns.get_mut(*i)
        {
            set.insert(name.to_string());
        }
    }

    /// The column of `out` an `EXCEPT` entry names: a top-level column
    /// (`Top`), a field below one (`Nested`), or nothing (`Missing`). A
    /// bare name matches by column name; a longer path matches when its
    /// last part names the column and the leading parts match the
    /// qualifier (as in `ids.id`).
    fn except_target(
        &self,
        path: &[String],
        out: &[Attr],
        cs: bool,
    ) -> Result<ExceptTarget, AnalysisError> {
        if path.is_empty() {
            return Err(unsupported("an empty EXCEPT entry", Origin::UNKNOWN));
        }
        match self.except_top(path, out, cs)[..] {
            [ref a] => Ok(ExceptTarget::Top(a.clone())),
            [] => {
                // A field below a top-level column when a strict prefix of
                // the entry names one (as in `t.id.sub`, or `id.sub`).
                let nested = path.len() > 1
                    && (1..path.len()).any(|j| !self.except_top(&path[..j], out, cs).is_empty());
                if nested {
                    Ok(ExceptTarget::Nested)
                } else {
                    Ok(ExceptTarget::Missing)
                }
            }
            _ => Err(AnalysisError::new(
                AmbiguousReference {
                    name: quoted(path),
                    reference_names: format!("[{}]", names(out, false)),
                },
                Origin::UNKNOWN,
            )),
        }
    }

    /// The columns of `out` an `EXCEPT` entry prefix names: a bare name
    /// matches by column name; a longer path matches when its last part
    /// names the column and the leading parts match the qualifier (as in
    /// `ids.id`).
    fn except_top(&self, path: &[String], out: &[Attr], cs: bool) -> Vec<Attr> {
        let Some((last, leading)) = path.split_last() else {
            return Vec::new();
        };
        if leading.is_empty() {
            return out
                .iter()
                .filter(|a| same(&a.col.name, last, cs))
                .cloned()
                .collect();
        }
        out.iter()
            .filter(|a| {
                same(&a.col.name, last, cs)
                    && leading.len() <= a.qual.len()
                    && a.qual[a.qual.len() - leading.len()..]
                        .iter()
                        .zip(leading)
                        .all(|(x, y)| same(x, y, cs))
            })
            .cloned()
            .collect()
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
            ExprNode::TypedLiteral {
                kind: TypedLiteralKind::Time,
                value,
            } => {
                if !self.ctx.conf.time_type {
                    return Err(AnalysisError::new(
                        krishiv_common::error::UnsupportedTimeType {},
                        e.origin,
                    ));
                }
                // A time is a string cast to `TIME(6)` (design P2-T181): the
                // value is checked again here, in case the plan came from a
                // front end other than the SQL parser.
                let Some(nanos) = krishiv_kernels::cast::parse_time(value) else {
                    return Err(AnalysisError::new(
                        krishiv_common::error::InvalidTypedLiteral {
                            value: format!("'{value}'"),
                            value_type: "\"TIME\"".to_owned(),
                        },
                        e.origin,
                    ));
                };
                let text = krishiv_kernels::cast::format_time(nanos);
                let text = Arc::new(Expr::from(Literal::String(text)).with_origin(e.origin));
                let cast = Expr::new(ExprNode::Cast {
                    expr: text,
                    to: DataType::Time { precision: 6 },
                    mode: CastMode::Ansi,
                })
                .with_origin(e.origin);
                Bound {
                    expr: Arc::new(cast),
                    shown: Some(Arc::clone(e)),
                    ..keep(ArgType {
                        nullable: false,
                        data_type: Some(DataType::Time { precision: 6 }),
                    })
                }
            }
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
            // A lambda outside a higher-order function's argument list:
            // lambdas in argument position are bound by the higher-order
            // call itself (see `hof`), never here. An already-bound lambda
            // (re-analysis of a resolved plan) is preserved as-is; its body
            // already holds `LambdaVar` leaves.
            ExprNode::Lambda { body, .. } => {
                let bound = crate::walk::visit_distinct(crate::walk::Node::Expr(body), |n| {
                    if let crate::walk::Node::Expr(x) = n
                        && matches!(&x.node, ExprNode::Resolved(ResolvedExpr::LambdaVar(_)))
                    {
                        return std::ops::ControlFlow::Break(());
                    }
                    std::ops::ControlFlow::Continue(())
                })
                .is_break();
                if bound && let Some(kid) = kids.first() {
                    keep(kid.ty.clone())
                } else {
                    return Err(unsupported(
                        "lambda functions outside a higher-order function",
                        e.origin,
                    ));
                }
            }
            ExprNode::Field { base: _, name } => {
                let (expr, ty) = self.struct_field(e, &kids[0], name)?;
                Bound {
                    expr,
                    ty: ty.clone(),
                    ..keep(ty)
                }
            }
            ExprNode::UpdateFields { ops, .. } => {
                return self.update_fields(e, ops, kids, cx);
            }
            ExprNode::Function(f) => {
                let n = f.args.len();
                // The filter is the child after the arguments, when the
                // call has one; any `WITHIN GROUP` ordering keys follow it
                // and are never the filter.
                if f.filter.is_some()
                    && let Some(filter) = kids.get(n)
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
                if f.filter.is_some()
                    && let Some(filter) = kids.get(n)
                    && filter.win.is_some()
                {
                    return Err(super::windows::window_in_filter(filter, e.origin));
                }
                // `builtin.f` / `system.builtin.f` resolve to the builtin;
                // `session.f` / `system.session.f` reach a session UDF only;
                // any other qualifier is an unresolved routine.
                // Grouping functions are never qualified.
                // A function stored in a namespace of the session's catalog,
                // named by its qualifier (`default.f`, `spark_catalog.default.f`).
                if !f.qualifier.is_empty()
                    && !super::bind::is_session_qualifier(&f.qualifier)
                    && !super::bind::is_builtin_qualifier(&f.qualifier)
                {
                    let mut named = f.qualifier.clone();
                    named.push(f.name.clone());
                    if let FunctionLookup::Sql(def) =
                        self.ctx.functions.lookup(&named, self.ctx.profile)
                    {
                        return self.sql_call(&def, e, f, kids, cx.op);
                    }
                }
                let (parts, builtin) = qualified_parts(f, &self.b.search_path, e.origin)?;
                let session_qualified = super::bind::is_session_qualifier(&f.qualifier);
                let lookup_name = builtin.as_deref().unwrap_or(&f.name);
                // A SQL-bodied function of the session: the call is replaced by
                // its body. Built-ins and the operators rank first.
                if !session_qualified
                    && builtin.is_none()
                    && f.qualifier.is_empty()
                    && GroupingFn::from_name(lookup_name).is_none()
                    && builtin_op(lookup_name, n).is_none()
                    && let FunctionLookup::Sql(def) = self
                        .ctx
                        .functions
                        .lookup(std::slice::from_ref(&f.name), self.ctx.profile)
                {
                    return self.sql_call(&def, e, f, kids, cx.op);
                }
                // `session.f` / `system.session.f` reach a temporary SQL
                // function as well as a session UDF.
                if session_qualified
                    && let FunctionLookup::Sql(def) =
                        self.ctx.functions.lookup(&parts, self.ctx.profile)
                {
                    return self.sql_call(&def, e, f, kids, cx.op);
                }
                // A call of a function with a named-parameter signature is
                // rearranged into positional order (Spark's
                // `FunctionBuilderBase` rearrangement), filling defaults; any
                // other call with named arguments is unsupported.
                let signature = if session_qualified {
                    None
                } else {
                    named_signature(lookup_name)
                };
                let mut kids = kids.to_vec();
                if signature.is_some() {
                    let filter = kids.get(n).cloned();
                    // Any `WITHIN GROUP` ordering keys follow the filter;
                    // keep them after it (only `mask` and the unimplemented
                    // `count_min_sketch` rearrange, and both refuse the
                    // ordering, but the slice below must still count it).
                    let orders: Vec<Bound> = kids
                        .get(n + usize::from(f.filter.is_some())..)
                        .unwrap_or(&[])
                        .to_vec();
                    let mut reordered = self.rearrange_named(
                        &parts,
                        signature.unwrap_or(&[]),
                        f,
                        &kids[..n],
                        e.origin,
                    )?;
                    reordered.extend(filter);
                    reordered.extend(orders);
                    kids = reordered;
                } else if f.arg_names.iter().any(Option::is_some) {
                    return Err(AnalysisError::new(
                        NamedParametersNotSupported {
                            function_name: quoted(&parts),
                        },
                        e.origin,
                    ));
                }
                // The bound argument children: after any named-argument
                // rearrangement (which fills defaults, so `f.args.len()`
                // undercounts `mask`), minus the filter and any
                // `WITHIN GROUP` ordering keys (which follow it, so
                // `kids.len()` overcounts `mode`/`listagg`).
                let n = kids.len() - usize::from(f.filter.is_some()) - f.within_group.len();
                // Spark's `SimpleDateFormat` time parser is not implemented: a
                // pattern function under it would answer with the new parser.
                if self.ctx.conf.legacy_time_parser
                    && !session_qualified
                    && is_pattern_function(lookup_name, &kids[..n])
                {
                    return Err(unsupported(
                        "spark.sql.legacy.timeParserPolicy=LEGACY (the SimpleDateFormat time parser)",
                        e.origin,
                    ));
                }
                if !session_qualified && let Some(which) = GroupingFn::from_name(lookup_name) {
                    return self.grouping_call(which, e, &kids[..n], cx);
                }
                // `to_char(datetime, fmt)` and `to_varchar` are `date_format` to
                // Spark (`ToCharacterBuilder`), which names them so; a number or
                // a binary keeps `to_char`.
                if !session_qualified
                    && n == 2
                    && !window_fn
                    && (lookup_name.eq_ignore_ascii_case("to_char")
                        || lookup_name.eq_ignore_ascii_case("to_varchar"))
                    && matches!(
                        kids[0].ty.data_type,
                        Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
                    )
                {
                    // Spark builds `date_format` first, so its clauses are
                    // refused under that name.
                    let syntax = if f.distinct {
                        Some("DISTINCT")
                    } else if f.filter.is_some() {
                        Some("FILTER CLAUSE")
                    } else if !f.within_group.is_empty() {
                        Some("WITHIN GROUP (ORDER BY ...)")
                    } else {
                        None
                    };
                    if let Some(syntax) = syntax {
                        return Err(AnalysisError::new(
                            InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                                pretty_name: "`date_format`".to_owned(),
                                syntax: syntax.to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    let call = Expr::function(
                        "date_format",
                        vec![Expr::clone(&kids[0].expr), Expr::clone(&kids[1].expr)],
                    )
                    .with_origin(e.origin);
                    let mut b = self.expr(&Arc::new(call), cx)?;
                    b.carry_markers(&kids[..n]);
                    b.shown = Some(Arc::new(
                        Expr::function(
                            "date_format",
                            vec![kids[0].shown(), kids[1].shown()]
                                .into_iter()
                                .map(|k| Expr::clone(&k))
                                .collect(),
                        )
                        .with_origin(e.origin),
                    ));
                    return Ok(b);
                }
                // `date + int`, `int + date` and `date - int` are `date_add`
                // and `date_sub` to Spark, which names them so.
                if !session_qualified
                    && n == 2
                    && let Some((name, swap)) =
                        date_with_days(lookup_name, &kids[0].ty, &kids[1].ty)
                {
                    let (first, second) = if swap { (1, 0) } else { (0, 1) };
                    let call = Expr::function(
                        name,
                        vec![
                            Expr::clone(&kids[first].expr),
                            Expr::clone(&kids[second].expr),
                        ],
                    )
                    .with_origin(e.origin);
                    // The children are bound already, and rebinding a
                    // resolved child drops its markers: carry them over.
                    let mut b = self.expr(&Arc::new(call), cx)?;
                    b.carry_markers(&kids[..n]);
                    // Shown as Spark names it, over the operands as written
                    // (a session variable keeps its name).
                    b.shown = Some(Arc::new(
                        Expr::function(
                            name,
                            vec![kids[first].shown(), kids[second].shown()]
                                .into_iter()
                                .map(|k| Expr::clone(&k))
                                .collect(),
                        )
                        .with_origin(e.origin),
                    ));
                    return Ok(b);
                }
                if !session_qualified && let Some(op) = builtin_op(lookup_name, n) {
                    // `ILIKE` and `LIKE|ILIKE ANY|SOME|ALL` are rewritten to
                    // the calls they denote, keeping Spark's display name.
                    if op == Builtin::ILike {
                        return self.ilike_call(e, &kids[..n]);
                    }
                    if op.quantified_like().is_some() {
                        return self.like_quantified_call(op, e, &kids[..n]);
                    }
                    return self.operator_call(op, e.origin, &kids[..n]);
                }
                let mut entry = if session_qualified {
                    match self.ctx.functions.lookup(&parts, self.ctx.profile) {
                        FunctionLookup::Found(entry) => entry,
                        FunctionLookup::KnownUnsupported => {
                            return Err(unsupported(
                                format!("function {}", parts.join(".")),
                                e.origin,
                            ));
                        }
                        FunctionLookup::NotFound | FunctionLookup::Sql(_) => {
                            return Err(AnalysisError::new(
                                UnresolvedRoutine {
                                    routine_name: quoted(&parts),
                                    search_path: self.b.search_path.clone(),
                                },
                                e.origin,
                            ));
                        }
                    }
                } else {
                    match self.ctx.functions.lookup(
                        std::slice::from_ref(&lookup_name.to_owned()),
                        self.ctx.profile,
                    ) {
                        // A builtin-qualified call (`builtin.f`,
                        // `system.builtin.f`) reaches built-ins only: a
                        // session UDF of the same name stays unresolved.
                        FunctionLookup::Found(entry)
                            if builtin.is_some() && matches!(entry.id, FunctionId::Udf { .. }) =>
                        {
                            return Err(AnalysisError::new(
                                UnresolvedRoutine {
                                    routine_name: quoted(&parts),
                                    search_path: self.b.search_path.clone(),
                                },
                                e.origin,
                            ));
                        }
                        FunctionLookup::Found(entry) => entry,
                        FunctionLookup::KnownUnsupported => {
                            return Err(unsupported(
                                format!("function {}", parts.join(".")),
                                e.origin,
                            ));
                        }
                        FunctionLookup::NotFound | FunctionLookup::Sql(_) => {
                            return Err(AnalysisError::new(
                                UnresolvedRoutine {
                                    routine_name: quoted(&parts),
                                    search_path: self.b.search_path.clone(),
                                },
                                e.origin,
                            ));
                        }
                    }
                };
                // `array_agg` is Spark's spelling of `collect_list` (its
                // result column is named `collect_list(...)`): resolve it
                // as `collect_list` throughout.
                if entry.id.name() == "array_agg" {
                    entry = match self
                        .ctx
                        .functions
                        .lookup(&["collect_list".to_owned()], self.ctx.profile)
                    {
                        FunctionLookup::Found(entry) => entry,
                        _ => {
                            return Err(unsupported("function array_agg".to_owned(), e.origin));
                        }
                    };
                }
                // `string_agg` is Spark's alias of `listagg` (its result
                // column is named `listagg(...)`): resolve it as `listagg`
                // throughout.
                if entry.id.name() == "string_agg" {
                    entry = match self
                        .ctx
                        .functions
                        .lookup(&["listagg".to_owned()], self.ctx.profile)
                    {
                        FunctionLookup::Found(entry) => entry,
                        _ => {
                            return Err(unsupported("function string_agg".to_owned(), e.origin));
                        }
                    };
                }
                // `mode` and `listagg` take `WITHIN GROUP` orderings, which
                // lower into the resolved call (see `mode_call` and
                // `listagg_call`); any other function keeps Spark's
                // `FUNCTION_WITH_UNSUPPORTED_SYNTAX` error.
                if !f.within_group.is_empty()
                    && !matches!(
                        entry.id.name(),
                        "mode" | "listagg" | "percentile_cont" | "percentile_disc"
                    )
                {
                    return Err(AnalysisError::new(
                        InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                            pretty_name: format!("`{lookup_name}`"),
                            syntax: "WITHIN GROUP (ORDER BY ...)".to_owned(),
                        },
                        e.origin,
                    ));
                }
                if !window_fn && entry.kind == FunctionKind::Aggregate {
                    self.nondeterministic_arguments(e, &kids[..n]);
                }
                if entry.id.name() == "mode" {
                    return self.mode_call(&entry, e, f, &kids, window_fn);
                }
                if entry.id.name() == "listagg" {
                    return self.listagg_call(&entry, e, f, &kids, window_fn);
                }
                if matches!(
                    entry.id.name(),
                    "percentile"
                        | "percentile_approx"
                        | "approx_percentile"
                        | "median"
                        | "percentile_cont"
                        | "percentile_disc"
                        | "approx_count_distinct"
                ) {
                    return self.quantile_call(&entry, e, f, &kids, n, window_fn);
                }
                if !window_fn {
                    call_checks(&entry, e, &kids, n)?;
                }
                // `struct(a, ...)`: children with names keep them (a column
                // contributes its name, a struct field its field name), so a
                // call with any named child becomes `named_struct(name, ...)`
                // with `colN` defaults; all-default calls stay `struct`
                // (even one written with aliases: the alias is its default).
                // Only for the built-in: a qualified session UDF named
                // `struct` runs the UDF.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "struct"
                    && let Some(rewritten) = self.struct_names(f, &kids[..n], e.origin)
                {
                    // The children are bound already, and rebinding a
                    // resolved child drops its markers: carry them over.
                    let mut b = self.expr(&rewritten, cx)?;
                    b.carry_markers(&kids[..n]);
                    // Shown as written (`struct(1 AS a)`), not the
                    // `named_struct` it binds as: aliases keep `AS name`.
                    let mut shown_kids = Vec::with_capacity(n);
                    for (arg, k) in f.args.iter().zip(&kids[..n]) {
                        if let ExprNode::Alias { name, .. } = &arg.node {
                            shown_kids.push(Arc::new(
                                Expr::new(ExprNode::Alias {
                                    expr: k.shown(),
                                    name: name.clone(),
                                })
                                .with_origin(arg.origin),
                            ));
                        } else {
                            shown_kids.push(k.shown());
                        }
                    }
                    b.shown = Some(with_children(e, shown_kids));
                    return Ok(b);
                }
                if entry.id.name() == "count" && n == 0 {
                    return self.parameterless_count(e, &entry, &kids, agg, window_fn);
                }
                // `assert_true(c)` is `assert_true(c, '<c>' is not true!)`:
                // Spark's constructor materializes the default message, so
                // the schema names it (`assert_true(true, 'true' is not
                // true!)`) and the kernel raises it. Only for the built-in:
                // a qualified session UDF named `assert_true` takes its
                // arguments as written.
                // Only for built-ins: a qualified session UDF named `assert_true`
                // takes its arguments as written, as do the other built-in
                // constructors below and the conditional rewrites.
                let mut owned: Vec<Bound>;
                let (kids, n) = if is_builtin_entry(&entry)
                    && entry.id.name() == "assert_true"
                    && n == 1
                {
                    owned = kids.clone();
                    owned.insert(
                        1,
                        assert_message(&owned[0], e.origin, self.ctx.conf.session_zone),
                    );
                    (&owned[..], 2)
                } else if is_builtin_entry(&entry)
                    && entry.id.name() == "mask"
                    && (1..5).contains(&n)
                {
                    owned = kids[..n].to_vec();
                    owned.extend(MASK_DEFAULTS.iter().skip(n - 1).map(|d| match d {
                        Some(text) => default_argument(
                            Literal::String((*text).to_owned()),
                            DataType::String,
                            false,
                            e.origin,
                        ),
                        None => default_argument(
                            Literal::Null(Some(DataType::String)),
                            DataType::String,
                            true,
                            e.origin,
                        ),
                    }));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 5)
                } else if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "locate" | "position")
                    && n == 2
                {
                    // `locate(sub, str)` is `locate(sub, str, 1)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Integer(1),
                        DataType::Integer,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 3)
                } else if is_builtin_entry(&entry) && entry.id.name() == "sort_array" && n == 1 {
                    // `sort_array(arr)` is `sort_array(arr, true)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Boolean(true),
                        DataType::Boolean,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 2)
                } else if is_builtin_entry(&entry)
                    && let Some(default) = regexp_default_index(entry.id.name(), n)
                {
                    // `regexp_extract(str, re)` is `regexp_extract(str, re, 1)`,
                    // `regexp_instr(str, re)` takes index 0 and
                    // `regexp_replace(str, re, rep)` position 1; Spark names
                    // the column with the default shown.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Integer(default),
                        DataType::Integer,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], n + 1)
                } else if is_builtin_entry(&entry) && entry.id.name() == "split" && n == 2 {
                    // `split(str, pattern)` is `split(str, pattern, -1)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Integer(-1),
                        DataType::Integer,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 3)
                } else if is_builtin_entry(&entry)
                    && let Some(at) = options_index(entry.id.name())
                    && n > at
                    && let Some(pairs) = literal_map_pairs(&kids[at].expr)
                {
                    // A `map('k', 'v', ...)` of string literals is the
                    // options literal Spark's `convertToMapData` reads at
                    // analysis; the executor binds it as a literal.
                    // Spark evaluates the options `map(...)`: a repeated key
                    // is `DUPLICATED_MAP_KEY` unless the session keeps the
                    // last one.
                    let mut distinct_pairs: Vec<(String, String)> = Vec::new();
                    for (key, value) in pairs {
                        if let Some(seen) = distinct_pairs.iter_mut().find(|(k, _)| *k == key) {
                            if !self.ctx.conf.map_key_dedup_last_win {
                                return Err(AnalysisError::new(
                                    DuplicatedMapKey {
                                        key,
                                        map_key_dedup_policy: "\"spark.sql.mapKeyDedupPolicy\""
                                            .to_owned(),
                                    },
                                    e.origin,
                                ));
                            }
                            seen.1 = value;
                        } else {
                            distinct_pairs.push((key, value));
                        }
                    }
                    let pairs = distinct_pairs;
                    owned = kids[..n].to_vec();
                    let original = Arc::clone(&owned[at].expr);
                    owned[at].shown = Some(Arc::clone(&original));
                    owned[at].expr = Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Map(pairs)))
                            .with_origin(original.origin),
                    );
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], n)
                } else if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "vector_norm" | "vector_normalize")
                    && n == 1
                {
                    // `vector_norm(v)` is `vector_norm(v, 2.0)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Float(2.0),
                        DataType::Float,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 2)
                } else if is_builtin_entry(&entry) && entry.id.name() == "named_struct" && n > 0 {
                    owned = kids.clone();
                    // A foldable name that is not a literal becomes one, so
                    // the executor binds it (the result rule already read it
                    // folded; the shown form keeps the written argument for
                    // error quotes).
                    for k in owned.iter_mut().step_by(2) {
                        if !matches!(k.expr.node, ExprNode::Literal(_))
                            && let Some(Constant::String(name)) =
                                self.fold(&k.expr, self.ctx.conf.ansi)
                        {
                            let lit = Arc::new(
                                Expr::new(ExprNode::Literal(Literal::String(name)))
                                    .with_origin(k.expr.origin),
                            );
                            k.shown = Some(Arc::clone(&k.expr));
                            k.expr = lit;
                        }
                    }
                    (&owned[..], n)
                } else {
                    (&kids[..], n)
                };
                // Only for built-ins: a qualified session UDF named like a
                // conditional (`session.nvl`, `system.session.nvl`, ...) runs
                // the UDF, while a bare `nvl` runs the built-in (which wins
                // the search path).
                if is_builtin_entry(&entry)
                    && let Some(which) = Conditional::of(lookup_name, n)
                {
                    return self.conditional_call(which, &entry, e, f, kids);
                }
                // A cast-named function (`float(x)`, `int(x)`, ...) is a
                // cast, never a call: the plan only ever holds the `Cast`
                // node. `DISTINCT` and `FILTER` have no cast form: Spark
                // rejects them as unsupported syntax on the function. Only for
                // built-ins: a qualified session UDF named `int` runs the UDF.
                if is_builtin_entry(&entry)
                    && let Some(to) = cast_target(&f.name)
                {
                    if f.distinct {
                        return Err(AnalysisError::new(
                            InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                                pretty_name: "`cast`".to_owned(),
                                syntax: "DISTINCT".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    if f.filter.is_some() {
                        return Err(AnalysisError::new(
                            InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                                pretty_name: "`cast`".to_owned(),
                                syntax: "FILTER CLAUSE".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    return self.cast_named_call(&entry, e, f, kids, to);
                }
                // `s['a']` on a struct is the field `a` (Spark's
                // `ExtractValue`): the extraction is a non-null string.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "__subscript"
                    && n == 2
                    && matches!(kids[0].ty.data_type, Some(DataType::Struct(_)))
                {
                    let ExprNode::Literal(Literal::String(name)) = &kids[1].expr.node else {
                        return Err(AnalysisError::new(
                            InvalidExtractFieldType {
                                extraction: format!(
                                    "\"{}\"",
                                    pretty(&kids[1].shown(), self.ctx.conf.session_zone)
                                ),
                            },
                            e.origin,
                        ));
                    };
                    let (expr, ty) = self.struct_field(e, &kids[0], name)?;
                    return Ok(Bound {
                        expr,
                        ty: ty.clone(),
                        ..keep(ty)
                    });
                }
                // `array_of_structs['a']` is Spark's `GetArrayStructFields`: the
                // field of every element, a NULL element giving NULL. It is
                // built as `transform(arr, x -> x.a)` once the field is known.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "__subscript"
                    && n == 2
                    && let Some(DataType::Array(arr)) = &kids[0].ty.data_type
                    && let DataType::Struct(st) = arr.element_type()
                    && let ExprNode::Literal(Literal::String(name)) = &kids[1].expr.node
                {
                    let cs = self.ctx.conf.case_sensitive;
                    let Some(field) = st.fields().iter().find(|f| same(f.name(), name, cs)) else {
                        return Err(unresolved_field(
                            &[
                                pretty(&kids[0].shown(), self.ctx.conf.session_zone),
                                name.clone(),
                            ],
                            st,
                            e.origin,
                        ));
                    };
                    let element = Expr::new(ExprNode::Column(ColumnRef {
                        name: vec!["__subscript_element".to_owned()],
                        plan_id: None,
                    }))
                    .with_origin(e.origin);
                    let body = Expr::new(ExprNode::Field {
                        base: Arc::new(element),
                        name: field.name().to_owned(),
                    })
                    .with_origin(e.origin);
                    let call = Expr::function(
                        "transform",
                        vec![
                            Expr::clone(&kids[0].expr),
                            Expr::lambda(vec!["__subscript_element".to_owned()], body),
                        ],
                    )
                    .with_origin(e.origin);
                    let mut b = self.expr(&Arc::new(call), cx)?;
                    b.carry_markers(&kids[..n]);
                    // Spark names it as the field of the array (`x.a`).
                    b.shown = Some(Arc::new(
                        Expr::new(ExprNode::Field {
                            base: Arc::new(Expr::clone(&kids[0].shown())),
                            name: field.name().to_owned(),
                        })
                        .with_origin(e.origin),
                    ));
                    return Ok(b);
                }
                let (mut call_kids, mut ty) = self.typed_call(&entry, e, &kids[..n])?;
                // `arrays_zip(a, b)` names its struct fields after columns,
                // struct fields and aliases (`0`, `1`, ... otherwise): the
                // result is renamed and the names follow the arrays as
                // string literals, which the executor reads.
                let mut zip_names: Vec<Arc<Expr>> = Vec::new();
                // Under `spark.sql.legacy.negativeIndexInArrayInsert`,
                // `array_insert` carries the switch as a trailing literal.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "array_insert"
                    && self.ctx.conf.legacy_negative_index_in_array_insert
                {
                    zip_names.push(Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Boolean(true))).with_origin(e.origin),
                    ));
                }
                // `extract(SECOND FROM t)` reads a `time(p)` to `p` digits
                // whatever digits the value holds (a string cast keeps all
                // of them, and Arrow's `Time64` does not carry `p`): the
                // precision follows as a trailing literal.
                if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "extract" | "date_part")
                    && n == 2
                    && let Some(DataType::Time { precision }) = &call_kids[1].ty.data_type
                    && *precision < 6
                {
                    zip_names.push(Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Integer(i32::from(*precision))))
                            .with_origin(e.origin),
                    ));
                }
                if is_builtin_entry(&entry)
                    && entry.id.name() == "arrays_zip"
                    && let Some(DataType::Array(a)) = &ty.data_type
                    && let DataType::Struct(st) = a.element_type()
                {
                    let names: Vec<String> = kids[..n]
                        .iter()
                        .enumerate()
                        .map(|(i, k)| match &k.expr.node {
                            ExprNode::Resolved(
                                ResolvedExpr::Column(c)
                                | ResolvedExpr::OuterColumn(c)
                                | ResolvedExpr::LambdaVar(c),
                            ) => c.name.clone(),
                            ExprNode::Resolved(ResolvedExpr::Field { name, .. }) => name.clone(),
                            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
                                output.name.clone()
                            }
                            _ => i.to_string(),
                        })
                        .collect();
                    let fields = st
                        .fields()
                        .iter()
                        .zip(&names)
                        .map(|(f, name)| {
                            StructField::new(name.clone(), f.data_type().clone(), f.nullable())
                        })
                        .collect::<Vec<_>>();
                    ty.data_type = Some(DataType::Array(
                        ArrayType::new(
                            DataType::Struct(StructType::new(fields)),
                            a.contains_null(),
                        )
                        .into(),
                    ));
                    zip_names = names
                        .into_iter()
                        .map(|name| {
                            Arc::new(
                                Expr::new(ExprNode::Literal(Literal::String(name)))
                                    .with_origin(e.origin),
                            )
                        })
                        .collect();
                }
                // `schema_of_csv(lit)` resolves below to the DDL literal.
                // The registry lookup is ASCII case-insensitive and
                // `id.name()` is the canonical lower-case name, so a
                // case-variant call (`SCHEMA_OF_CSV`) resolves here too.
                // Only for the built-in: a qualified session UDF of the
                // same name runs the UDF.
                let is_schema_of_csv = is_builtin_entry(&entry)
                    && matches!(
                        entry.id.name(),
                        "schema_of_csv" | "schema_of_xml" | "schema_of_json"
                    );
                let of_xml = entry.id.name() == "schema_of_xml";
                let of_json = entry.id.name() == "schema_of_json";
                // `from_csv` and `to_csv` show only their first argument.
                let is_csv = is_builtin_entry(&entry)
                    && matches!(
                        entry.id.name(),
                        "from_csv" | "to_csv" | "from_xml" | "to_xml"
                    );
                if is_builtin_entry(&entry) && entry.id.name() == "mask" {
                    self.check_mask_replacements(lookup_name, &kids[..n], &call_kids[..n])?;
                }
                let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
                call_kids.extend(kids.get(n).cloned());
                // `typeof(x)` is Spark's `RuntimeReplaceable`; its type
                // name is known once the call is typed. Only for the
                // built-in: a qualified session UDF named `typeof` runs
                // the UDF.
                let is_typeof = is_builtin_entry(&entry) && entry.id.name() == "typeof";
                let session_value_early = if is_builtin_entry(&entry) {
                    match entry.id.name() {
                        "current_schema" | "current_database" => {
                            Some(self.ctx.catalogs.current_namespace().join("."))
                        }
                        "current_path" => Some(super::current_path(&*self.ctx.catalogs)),
                        _ => None,
                    }
                } else {
                    None
                };
                let plain_args: Vec<Arc<Expr>> =
                    call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                let zipped = !zip_names.is_empty();
                let call = ResolvedFunction {
                    id: entry.id,
                    kind: entry.kind,
                    args: plain_args.iter().cloned().chain(zip_names).collect(),
                    distinct: f.distinct,
                    filter: call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                    within_group: Vec::new(),
                    data_type,
                    nullable: ty.nullable,
                    deterministic: entry.deterministic,
                };
                // The names `arrays_zip` carries are not part of the call as
                // written.
                let plain = zipped.then(|| {
                    rebuilt(ExprNode::Resolved(ResolvedExpr::Function(
                        ResolvedFunction {
                            args: plain_args,
                            ..call.clone()
                        },
                    )))
                });
                let expr = rebuilt(ExprNode::Resolved(ResolvedExpr::Function(call)));
                let shown = match plain {
                    Some(plain) => self
                        .call_shown(&parts, &plain, kids, &call_kids)
                        .or(Some(plain)),
                    None => self.call_shown(&parts, &expr, kids, &call_kids),
                };
                // `typeof(x)` is Spark's `RuntimeReplaceable`: the analyzer
                // replaces it with the type name literal, keeping the call
                // as the shown form (the executor gets no type logic).
                // `current_schema()`, `current_database()` and `current_path()`
                // read the session: the analyzer replaces them with the
                // value, as the query sees it.
                let session_value = session_value_early;
                let (expr, shown) = if let Some(value) = session_value {
                    (
                        rebuilt(ExprNode::Literal(Literal::String(value))),
                        Some(expr),
                    )
                } else if is_typeof {
                    let name = call_kids
                        .first()
                        .and_then(|k| k.ty.data_type.clone())
                        .map_or_else(|| "void".to_owned(), |t| t.catalog_string());
                    (
                        rebuilt(ExprNode::Literal(Literal::String(name))),
                        Some(expr),
                    )
                } else {
                    (expr, shown)
                };
                let mut bound = Bound {
                    agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                        Some(e.origin)
                    } else {
                        agg
                    },
                    shown,
                    expr,
                    ..keep(ty)
                };
                // `schema_of_csv(lit)` resolves during analysis (Spark's
                // `SchemaOfCsv` evaluates its foldable child): the DDL
                // literal, shown as the call.
                if is_schema_of_csv {
                    self.resolve_schema_of_csv(e, &kids[..n], &mut bound, of_xml, of_json)?;
                }
                // `from_csv` and `to_csv` show only their first argument
                // (Spark's `toPrettySQL` drops the schema and options).
                if is_csv
                    && let Some(first) = kids.first()
                    && let ExprNode::Resolved(ResolvedExpr::Function(call)) = &bound.expr.node
                {
                    let mut new_call = call.clone();
                    new_call.args = vec![first.shown()];
                    bound.shown = Some(Arc::new(
                        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(new_call)))
                            .with_origin(bound.expr.origin),
                    ));
                }
                bound
            }
            ExprNode::Cast { to, mode, .. } => {
                let mode = match mode {
                    CastMode::Default if self.ctx.conf.ansi => CastMode::Ansi,
                    CastMode::Default => CastMode::Legacy,
                    m => *m,
                };
                // Spark's parser rejects the `TIME` type before any cast
                // is checked (`spark.sql.timeType.enabled` is false by
                // default; TIME cannot be evaluated here either way).
                if !self.ctx.conf.time_type && to.any_type(&|t| matches!(t, DataType::Time { .. }))
                {
                    return Err(AnalysisError::new(
                        krishiv_common::error::UnsupportedTimeType {},
                        e.origin,
                    ));
                }
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
                // A subquery resolves without the enclosing lambda scopes:
                // its columns must not bind to enclosing lambda variables.
                let saved = std::mem::take(&mut self.lambdas);
                let rel = self.rel(plan, &env);
                let stale = std::mem::replace(&mut self.lambdas, saved);
                debug_assert!(stale.is_empty(), "a subquery leaks lambda scopes");
                let rel = rel?;
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
            ExprNode::Resolved(r) => keep({
                use ResolvedExpr::{Column, LambdaVar, OuterColumn};
                match r {
                    Column(c) | OuterColumn(c) | LambdaVar(c) => column(c),
                    ResolvedExpr::Alias { output, .. } => column(output),
                    ResolvedExpr::Function(f) => arg_type(f.data_type.clone(), f.nullable),
                    ResolvedExpr::StateInput { raw_type, .. } => arg_type(raw_type.clone(), true),
                    ResolvedExpr::Field {
                        data_type,
                        nullable,
                        ..
                    } => arg_type(data_type.clone(), *nullable),
                }
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
    /// `mode` with its `WITHIN GROUP` ordering and deterministic flag
    /// (Spark's `ModeBuilder` and `SupportsOrderingWithinGroup` checks):
    /// the ordering lowers to the value the executor reads, and the shown
    /// form keeps what was written (`mode() WITHIN GROUP (ORDER BY ...)`
    /// for the ordered forms, with Spark's swapped direction).
    ///
    /// `kids` holds the bound arguments, the filter, then the bound
    /// ordering keys; the executor reads the first resolved argument as
    /// the value and the second (a folded boolean literal, when present)
    /// as the tie-break (largest when true).
    #[inline(never)]
    fn mode_call(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        let zone = self.ctx.conf.session_zone;
        let ansi = self.ctx.conf.ansi;
        let n_args = f.args.len();
        let at = |i: usize| kids.get(i);
        let filter_at = n_args + usize::from(f.filter.is_some());
        let orders: &[Bound] = kids.get(filter_at..).unwrap_or(&[]);
        let sql = |i: usize| at(i).map(|k| pretty(&k.shown(), zone));
        if f.distinct {
            return Err(AnalysisError::new(
                InvalidWithinGroupExpressionDistinctUnsupported {
                    func_name: "`mode`".to_owned(),
                },
                e.origin,
            ));
        }
        // The nested-aggregate and window checks, over the arguments, the
        // filter and the ordering keys alike.
        let mut checked: Vec<Bound> = kids[..filter_at.min(kids.len())].to_vec();
        checked.extend(orders.iter().cloned());
        if !window_fn {
            call_checks(entry, e, &checked, filter_at.min(checked.len()))?;
        }
        let wrong_arity = || {
            AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`mode`".to_owned(),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: n_args.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            )
        };
        // The value the executor reads, the tie-break flag (largest when
        // true) and the shown ordering (Spark's swapped direction: an
        // ascending written order shows as descending and picks the
        // smallest, a descending one shows bare and picks the largest).
        enum Form {
            Plain,
            Smallest,
            Largest,
        }
        let (value, form): (&Bound, Form) = match (n_args, orders.len()) {
            (0, 0) => {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWithinGroupMissing {
                        func_name: "`mode`".to_owned(),
                    },
                    e.origin,
                ));
            }
            (0, 1) => {
                let descending = f.within_group[0].direction == SortDirection::Descending;
                (
                    &orders[0],
                    if descending {
                        Form::Largest
                    } else {
                        Form::Smallest
                    },
                )
            }
            (0, n) => {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWrongNumOrderings {
                        func_name: "`mode`".to_owned(),
                        expected_num: "1".to_owned(),
                        actual_num: n.to_string(),
                    },
                    e.origin,
                ));
            }
            (1, 0) => (at(0).ok_or_else(wrong_arity)?, Form::Plain),
            (2, 0) => {
                let flag = at(1).ok_or_else(wrong_arity)?;
                // The flag must fold (a column does not); a literal
                // without a folded value (binary, timestamp) still counts
                // as folded here, and fails the type check below.
                let folded = self.fold(&flag.expr, ansi);
                let literal = matches!(flag.expr.node, ExprNode::Literal(_));
                match (folded, literal) {
                    (None, false) => {
                        return Err(AnalysisError::new(
                            NonFoldableArgument {
                                func_name: "`mode`".to_owned(),
                                param_name: "`deterministic`".to_owned(),
                                param_type: "\"BOOLEAN\"".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    (Some(Constant::Error(err)), _) => {
                        return Err(AnalysisError::new(err, flag.expr.origin));
                    }
                    (Some(Constant::Null), _) => {
                        return Err(AnalysisError::new(
                            DatatypeMismatchUnexpectedNull {
                                sql_expr: format!("\"{}\"", sql(1).unwrap_or_default()),
                                expr_name: "`deterministic`".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    (Some(Constant::Boolean(true)), _) => {
                        (at(0).ok_or_else(wrong_arity)?, Form::Smallest)
                    }
                    (Some(Constant::Boolean(false)), _) => {
                        (at(0).ok_or_else(wrong_arity)?, Form::Plain)
                    }
                    // A literal without a folded value (binary, timestamp)
                    // or any other non-boolean foldable fails the type
                    // check, as Spark checks the type after folding.
                    _ if flag.ty.data_type != Some(DataType::Boolean) => {
                        let input = flag
                            .ty
                            .data_type
                            .clone()
                            .map_or_else(|| "VOID".to_owned(), |t| t.sql());
                        return Err(AnalysisError::new(
                            UnexpectedInputType {
                                param_index: "second".to_owned(),
                                function_name: "`mode`".to_owned(),
                                required_type: "\"BOOLEAN\"".to_owned(),
                                input_sql: format!("\"{}\"", sql(1).unwrap_or_default()),
                                input_type: format!("\"{input}\""),
                            },
                            e.origin,
                        ));
                    }
                    // A boolean-typed foldable that folds to no constant
                    // (unreachable through the current constant folder).
                    _ => (at(0).ok_or_else(wrong_arity)?, Form::Plain),
                }
            }
            (1 | 2, _) => {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWrongNumOrderings {
                        func_name: "`mode`".to_owned(),
                        expected_num: "0".to_owned(),
                        actual_num: orders.len().to_string(),
                    },
                    e.origin,
                ));
            }
            _ => return Err(wrong_arity()),
        };
        let data_type = value.ty.data_type.clone().unwrap_or(DataType::Null);
        let mut args = vec![Arc::clone(&value.expr)];
        // The tie-break the executor reads (largest when true).
        let largest = match form {
            Form::Plain => None,
            Form::Smallest => Some(false),
            Form::Largest => Some(true),
        };
        if let Some(largest) = largest {
            args.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::Boolean(largest))).with_origin(e.origin),
            ));
        }
        // The shown ordering: Spark's swapped direction (ascending shows
        // as descending, descending shows bare).
        let shown_order = |o: &Bound, descending: bool, shown: bool| SortOrder {
            expr: if shown {
                o.shown()
            } else {
                Arc::clone(&o.expr)
            },
            direction: if descending {
                SortDirection::Ascending
            } else {
                SortDirection::Descending
            },
            nulls: NullOrdering::NullsFirst,
        };
        let within_group = match form {
            Form::Plain => Vec::new(),
            Form::Smallest => vec![shown_order(value, false, false)],
            Form::Largest => vec![shown_order(value, true, false)],
        };
        let call = ResolvedFunction {
            id: entry.id.clone(),
            kind: entry.kind,
            args,
            distinct: f.distinct,
            filter: at(n_args)
                .filter(|_| f.filter.is_some())
                .map(|k| Arc::clone(&k.expr)),
            within_group,
            data_type: data_type.clone(),
            nullable: true,
            deterministic: entry.deterministic,
        };
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(call))).with_origin(e.origin),
        );
        // The shown form keeps what was written (the ordering for the
        // ordered forms, at Spark's swapped direction).
        let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args: vec![value.shown()],
                distinct: f.distinct,
                filter: at(n_args).filter(|_| f.filter.is_some()).map(Bound::shown),
                within_group: match form {
                    Form::Plain => Vec::new(),
                    Form::Smallest => vec![shown_order(value, false, true)],
                    Form::Largest => vec![shown_order(value, true, true)],
                },
                data_type,
                nullable: true,
                deterministic: entry.deterministic,
            },
        )));
        let shown = Arc::new(shown.with_origin(e.origin));
        let bound = Bound {
            expr,
            ty: arg_type(value.ty.data_type.clone().unwrap_or(DataType::Null), true),
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(e.origin)
            } else {
                kids.iter().find_map(|k| k.agg)
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral: kids
                .iter()
                .flat_map(|k| k.lateral.iter().copied())
                .collect(),
            shown: Some(shown),
        };
        Ok(bound)
    }

    /// `listagg(expr[, delimiter])` with its `WITHIN GROUP` ordering
    /// (Spark's `ListAgg` checks): the delimiter is implicitly cast to
    /// the value's type by the declared inputs, `DISTINCT` needs the
    /// ordering to be the value itself (or a cast that preserves
    /// equality, as Spark's `hasDistinctOrderAmbiguity` computes it).
    ///
    /// `kids` holds the bound arguments, the filter, then the bound
    /// ordering keys; the executor reads the value, then the ordering
    /// keys, and the folded delimiter from the resolved call.
    #[inline(never)]
    fn listagg_call(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        let n_args = f.args.len();
        let filter_at = n_args + usize::from(f.filter.is_some());
        let orders: &[Bound] = kids.get(filter_at..).unwrap_or(&[]);
        let zone = self.ctx.conf.session_zone;
        // The nested-aggregate and window checks, over the arguments, the
        // filter and the ordering keys alike.
        let mut checked: Vec<Bound> = kids[..filter_at.min(kids.len())].to_vec();
        checked.extend(orders.iter().cloned());
        if !window_fn {
            call_checks(entry, e, &checked, filter_at.min(checked.len()))?;
        }
        // The value and the delimiter, implicitly cast (Spark's arity
        // check, then `listagg`'s own delimiter checks in the registry).
        // The SQL layer fills a missing delimiter with NULL, so the call
        // always has both.
        if n_args == 0 {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`listagg`".to_owned(),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: "0".to_owned(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        // A call built outside SQL (`call_function("listagg", [x])`) has no
        // delimiter: it is NULL, as the SQL layer writes it.
        let mut head: Vec<Bound> = kids[..n_args].to_vec();
        if n_args == 1 {
            head.push(default_argument(
                Literal::Null(None),
                DataType::Null,
                true,
                e.origin,
            ));
        }
        let (mut call_kids, ty) = self.typed_call(entry, e, &head)?;
        // `DISTINCT` with an ordering: the ordering must be the value
        // itself. When the value is a cast of the ordering key, the cast
        // must preserve equality (Spark's `hasDistinctOrderAmbiguity`,
        // with the looser cast rule off): anything else is ambiguous
        // (`MISMATCH_WITH_DISTINCT_INPUT`), an unsafe cast (`DOUBLE`,
        // `FLOAT`, `TIMESTAMP` to `STRING`) its
        // `MISMATCH_WITH_DISTINCT_INPUT_UNSAFE_CAST`. The value is the
        // implicitly cast one (a `DOUBLE` column reads as its `STRING`
        // cast); the error quotes what was written.
        if f.distinct && !orders.is_empty() {
            // Structural equality, ignoring source spans (either side may
            // be written elsewhere).
            let mut memo = std::collections::HashMap::new();
            let mut key = |e: &Arc<Expr>| semantic_key(e, &mut memo);
            let value = &call_kids[0].expr;
            let same =
                orders.len() == 1 && orders.first().is_some_and(|o| key(&o.expr) == key(value));
            if !same {
                let cast_of = match &call_kids[0].expr.node {
                    ExprNode::Cast { expr, to, .. }
                        if orders.len() == 1
                            && orders.first().is_some_and(|o| key(&o.expr) == key(expr))
                            && matches!(to, DataType::String | DataType::Binary) =>
                    {
                        Some(to.clone())
                    }
                    _ => None,
                };
                match cast_of {
                    // A cast that preserves equality deduplicates exactly.
                    Some(to)
                        if orders
                            .first()
                            .and_then(|o| o.ty.data_type.clone())
                            .is_some_and(|from| Self::cast_preserves_equality(&from, &to)) => {}
                    Some(to) => {
                        let input = orders
                            .first()
                            .and_then(|o| o.ty.data_type.clone())
                            .map_or_else(|| "VOID".to_owned(), |t| t.sql());
                        return Err(AnalysisError::new(
                            InvalidWithinGroupExpressionMismatchWithDistinctInputUnsafeCast {
                                func_name: "`listagg`".to_owned(),
                                input_type: format!("\"{input}\""),
                                cast_type: format!("\"{}\"", to.sql()),
                            },
                            e.origin,
                        ));
                    }
                    None => {
                        let arg = |k: &Bound| pretty(&k.shown(), zone);
                        return Err(AnalysisError::new(
                            InvalidWithinGroupExpressionMismatchWithDistinctInput {
                                func_name: "`listagg`".to_owned(),
                                func_arg: format!("\"{}\"", arg(&kids[0])),
                                ordering_expr: orders
                                    .iter()
                                    .map(|o| format!("\"{}\"", pretty(&o.shown(), zone)))
                                    .collect::<Vec<_>>()
                                    .join(", "),
                            },
                            e.origin,
                        ));
                    }
                }
            }
        }
        // A NULL delimiter stays NULL (it reads as empty): the implicit
        // cast must not give it a type, or the same-type check below
        // would compare the value against the cast.
        if head.len() > 1
            && matches!(
                head.get(1).map(|k| &k.expr.node),
                Some(ExprNode::Literal(Literal::Null(_)))
            )
        {
            call_kids[1] = default_argument(Literal::Null(None), DataType::Null, true, e.origin);
        }
        // The executor reads a literal delimiter: fold one here (the
        // shown form keeps what was written). Binary and other literals
        // the folder has no value for stay as written for the planner.
        if let Some(delim) = call_kids.get_mut(1) {
            let folded = match self.fold(&delim.expr, self.ctx.conf.ansi) {
                Some(Constant::String(s)) => Some((Literal::String(s), false)),
                Some(Constant::Null) => Some((Literal::Null(None), true)),
                Some(Constant::Boolean(b)) => Some((Literal::String(b.to_string()), false)),
                Some(Constant::Integral(i)) => Some((Literal::String(i.to_string()), false)),
                Some(Constant::Error(err)) => {
                    return Err(AnalysisError::new(err, delim.expr.origin));
                }
                Some(_) | None => None,
            };
            if let Some((lit, nullable)) = folded {
                let ty = arg_type(DataType::String, nullable);
                if delim.ty.data_type == Some(DataType::String) {
                    delim.expr =
                        Arc::new(Expr::new(ExprNode::Literal(lit)).with_origin(delim.expr.origin));
                    delim.ty = ty;
                }
            }
        }
        // The resolved ordering keys, with their written directions.
        let mut args: Vec<Arc<Expr>> = call_kids.iter().map(|k| Arc::clone(&k.expr)).collect();
        args.extend(orders.iter().map(|k| Arc::clone(&k.expr)));
        let resolved_orders: Vec<SortOrder> = orders
            .iter()
            .zip(&f.within_group)
            .map(|(k, o)| SortOrder {
                expr: Arc::clone(&k.expr),
                direction: o.direction,
                nulls: o.nulls,
            })
            .collect();
        let shown_orders: Vec<SortOrder> = orders
            .iter()
            .zip(&f.within_group)
            .map(|(k, o)| SortOrder {
                expr: k.shown(),
                direction: o.direction,
                nulls: o.nulls,
            })
            .collect();
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        let call = ResolvedFunction {
            id: entry.id.clone(),
            kind: entry.kind,
            args,
            distinct: f.distinct,
            filter: kids
                .get(n_args)
                .filter(|_| f.filter.is_some())
                .map(|k| Arc::clone(&k.expr)),
            within_group: resolved_orders,
            data_type: data_type.clone(),
            nullable: ty.nullable,
            deterministic: entry.deterministic,
        };
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(call))).with_origin(e.origin),
        );
        // The shown form keeps what was written (the delimiter default
        // included, as Spark's constructor writes it).
        let shown_args: Vec<Arc<Expr>> = call_kids.iter().map(Bound::shown).collect();
        let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args: shown_args,
                distinct: f.distinct,
                filter: kids
                    .get(n_args)
                    .filter(|_| f.filter.is_some())
                    .map(Bound::shown),
                within_group: shown_orders,
                data_type,
                nullable: ty.nullable,
                deterministic: entry.deterministic,
            },
        )));
        let shown = Arc::new(shown.with_origin(e.origin));
        Ok(Bound {
            expr,
            ty,
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(e.origin)
            } else {
                kids.iter().find_map(|k| k.agg)
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral: kids
                .iter()
                .flat_map(|k| k.lateral.iter().copied())
                .collect(),
            shown: Some(shown),
        })
    }

    /// Whether casting `from` to string or binary preserves equality, so
    /// `listagg(DISTINCT CAST(x AS STRING)) WITHIN GROUP (ORDER BY x)`
    /// deduplicates exactly (Spark's `isCastEqualityPreserving` and
    /// `isCastTargetEqualityPreserving`, with the engine's types: no
    /// interval values, and every string is binary-collated).
    fn cast_preserves_equality(from: &DataType, to: &DataType) -> bool {
        if !matches!(to, DataType::String | DataType::Binary) {
            return false;
        }
        matches!(
            from,
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Decimal(_)
                | DataType::Date
                | DataType::Time { .. }
                | DataType::TimestampNtz
                | DataType::Boolean
                | DataType::Binary
                | DataType::String,
        )
    }

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
            .with_decimal_loss(loss)
            .with_case_sensitive(self.ctx.conf.case_sensitive);
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
        let bound_sql = |i: usize| call_kids.get(i).map(|k| pretty(&k.expr, zone));
        let call = CallTypes::new(&types, &constants)
            .with_literals(&literals)
            .with_sql(&sql)
            .with_bound_sql(&bound_sql)
            .with_ansi(ansi)
            .with_decimal_loss(loss)
            .with_case_sensitive(self.ctx.conf.case_sensitive);
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
        // `kids.first()` is the `FILTER`, if any (`n == 0`).
        call_kids.extend(kids.first().cloned());
        let call = ResolvedFunction {
            within_group: Vec::new(),
            id: entry.id.clone(),
            kind: entry.kind,
            args: call_kids
                .first()
                .map(|k| vec![Arc::clone(&k.expr)])
                .unwrap_or_default(),
            distinct: match &e.node {
                ExprNode::Function(f) => f.distinct,
                _ => false,
            },
            filter: call_kids.get(1).map(|k| Arc::clone(&k.expr)),
            data_type,
            nullable: ty.nullable,
            deterministic: entry.deterministic,
        };
        // Shown as written: no argument, the `FILTER` after the call.
        let shown = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    args: Vec::new(),
                    ..call.clone()
                },
            )))
            .with_origin(e.origin),
        );
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(call))).with_origin(e.origin),
        );
        // The `FILTER`'s markers ride along, as for any other call.
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            expr,
            ty,
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(e.origin)
            } else {
                agg
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
            // Shown as written (`count()`), not as bound (`count(NULL)`).
            shown: Some(shown),
        })
    }

    /// `ntile`'s bucket count as the literal it folds to (`ntile(1 + 1)`):
    /// the window operator reads a literal. The rule above checked it is an
    /// `int` constant. Only for the built-in: a session UDF named `ntile`
    /// keeps its arguments as written.
    #[inline(never)]
    fn folded_ntile(
        entry: &FunctionEntry,
        constants: &[Option<Constant>],
        mut kids: Vec<Bound>,
    ) -> Vec<Bound> {
        if is_builtin_entry(entry)
            && entry.id.name() == "ntile"
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

    /// `schema_of_csv(csv[, options])` as the DDL literal it resolves to (Spark's
    /// `SchemaOfCsv` evaluates its foldable child during analysis, keeping
    /// the call as the output name). The signature's result rule already
    /// rejected a non-foldable, null or non-string child, so anything but
    /// a folded string here (a custom registry's rule) keeps the call;
    /// that call has no scalar binding and fails at execution binding, so
    /// it never resolves silently. The options argument resolves when it is
    /// a map literal (the `map('k', 'v')` rewrite above); any other map
    /// keeps the call. Out of line: it is on the path of every call.
    #[inline(never)]
    fn resolve_schema_of_csv(
        &self,
        e: &Arc<Expr>,
        kids: &[Bound],
        bound: &mut Bound,
        xml: bool,
        json: bool,
    ) -> Result<(), AnalysisError> {
        if kids.is_empty() || kids.len() > 2 {
            return Ok(());
        }
        let csv = kids
            .first()
            .and_then(|k| self.fold(&k.expr, self.ctx.conf.ansi));
        let Some(Constant::String(csv)) = csv else {
            return Ok(());
        };
        let ddl = if json {
            // `schema_of_json`: only a `map(...)` literal of options folds;
            // anything else keeps the call.
            let pairs = match kids.get(1) {
                None => Vec::new(),
                Some(opts) => match &opts.expr.node {
                    ExprNode::Literal(Literal::Map(pairs)) => pairs.clone(),
                    _ => return Ok(()),
                },
            };
            match krishiv_functions::scalar::schema_of_json_ddl(&csv, &pairs)
                .map_err(|err| AnalysisError::new(err, e.origin))?
            {
                Some(ddl) => ddl,
                None => return Ok(()),
            }
        } else if xml {
            // `schema_of_xml`: the document under the written options.
            let pairs = match kids.get(1) {
                None => Vec::new(),
                Some(opts) => match &opts.expr.node {
                    ExprNode::Literal(Literal::Map(pairs)) => pairs.clone(),
                    // Spark reads the options only from a `map(...)` call.
                    _ => {
                        return Err(AnalysisError::new(
                            krishiv_common::error::InvalidOptionsNonMapFunction {},
                            e.origin,
                        ));
                    }
                },
            };
            krishiv_functions::scalar::schema_of_xml_ddl(&csv, &pairs)
                .map_err(|err| AnalysisError::new(err, e.origin))?
        } else {
            // Options other than a map literal keep the call (the signature
            // accepts them; the executor has no binding for it).
            if kids
                .get(1)
                .is_some_and(|o| !matches!(o.expr.node, ExprNode::Literal(Literal::Map(_))))
            {
                return Ok(());
            }
            self.csv_ddl(e, &csv, kids.get(1))?
        };
        self.show_schema_literal(e, kids.len(), bound, ddl);
        Ok(())
    }

    /// The DDL `schema_of_csv(csv[, options])` resolves to (`None`: defaults).
    fn csv_ddl(
        &self,
        e: &Arc<Expr>,
        csv: &str,
        options: Option<&Bound>,
    ) -> Result<String, AnalysisError> {
        let options = match options {
            None => krishiv_io::FormatOptions::defaults(krishiv_io::csv::option_schema()),
            Some(opts) => {
                // Options other than a literal map keep the call (the
                // signature accepts them; the executor has no binding).
                let ExprNode::Literal(Literal::Map(pairs)) = &opts.expr.node else {
                    return Err(AnalysisError::new(
                        InternalError {
                            message: "schema_of_csv options are not a map literal".to_owned(),
                        },
                        e.origin,
                    ));
                };
                // Option names are case-insensitive: of two spellings of one
                // name, the one written last wins (Spark's `CaseInsensitiveMap`).
                let mut user: std::collections::BTreeMap<
                    String,
                    krishiv_common::secret::OptionValue,
                > = std::collections::BTreeMap::new();
                for (k, v) in pairs {
                    user.retain(|seen, _| !seen.eq_ignore_ascii_case(k));
                    user.insert(
                        k.clone(),
                        krishiv_common::secret::OptionValue::from_user(k, v.clone()),
                    );
                }
                krishiv_io::csv::option_schema()
                    .parse(&user, &krishiv_common::config::Config::new())
                    .map_err(|err| AnalysisError::new(err, e.origin))?
                    .0
            }
        };
        krishiv_io::csv::ddl_of_string(csv, &options).map_err(|err| {
            AnalysisError::new(
                match err {
                    krishiv_io::FormatError::Other(e) => e,
                    other => InternalError {
                        message: other.to_string(),
                    }
                    .into(),
                },
                e.origin,
            )
        })
    }

    /// Replaces the resolved call by its DDL literal, shown as the call
    /// without its options.
    fn show_schema_literal(&self, e: &Arc<Expr>, args: usize, bound: &mut Bound, ddl: String) {
        // Keep the form the call already shows (a session variable's name,
        // not its value); otherwise show the call itself.
        if bound.shown.is_none() {
            bound.shown = Some(Arc::clone(&bound.expr));
        }
        // Spark's output name drops the options (`schema_of_csv(1|abc)`).
        if args == 2
            && let Some(shown) = &bound.shown
            && let ExprNode::Resolved(ResolvedExpr::Function(call)) = &shown.node
        {
            let mut one = call.clone();
            one.args.truncate(1);
            bound.shown = Some(Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(one)))
                    .with_origin(shown.origin),
            ));
        }
        bound.expr =
            Arc::new(Expr::new(ExprNode::Literal(Literal::String(ddl))).with_origin(e.origin));
    }

    /// The shown form of the call `expr` (see [`shown_call`]); `date_add(d,
    /// ' 12 ')` and `date_add(d, '1' || '2')` without ANSI show the day count
    /// as an integer (`date_add(d, 12)`): Spark folds the string it casts to
    /// `INT`. Out of line: it is on the path of every call.
    #[inline(never)]
    fn call_shown(
        &self,
        name: &[String],
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
        let conf = fold::Conf::new(
            ansi,
            self.ctx.conf.decimal_precision_loss,
            self.ctx.conf.map_key_dedup_last_win,
            (
                self.ctx.conf.legacy_java_charsets,
                self.ctx.conf.legacy_coding_error_action,
            ),
            self.ctx.profile,
            self.ctx.conf.session_zone.and_then(SessionZone::fixed),
        );
        fold::constant(
            e,
            conf,
            &mut self.folds.borrow_mut(),
            self.ctx.functions.as_ref(),
        )
    }

    fn legacy_day_count(&self, name: &[String], kids: &[Bound]) -> Option<Vec<Bound>> {
        let last = name.last().map(String::as_str).unwrap_or("");
        if self.ctx.conf.ansi
            || !["date_add", "dateadd", "date_sub"]
                .iter()
                .any(|n| last.eq_ignore_ascii_case(n))
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
    pub(super) fn implicit_cast_as(
        &self,
        b: &Bound,
        to: Option<DataType>,
        try_cast: bool,
    ) -> Bound {
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
                within_group: Vec::new(),
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
        if matches!(
            op,
            Builtin::Gt | Builtin::GtEq | Builtin::Lt | Builtin::LtEq
        ) && let [a, b] = kids
        {
            match strength_reduce_comparison(op, a, b) {
                ComparisonReduction::Keep => {}
                ComparisonReduction::Decided(v) => {
                    let mut lateral: Vec<AttrId> = Vec::new();
                    for id in kids.iter().flat_map(|k| k.lateral.iter()) {
                        if !lateral.contains(id) {
                            lateral.push(*id);
                        }
                    }
                    return Ok(Bound {
                        shown: None,
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Boolean(v))).with_origin(origin),
                        ),
                        ty: arg_type(DataType::Boolean, false),
                        agg: kids.iter().find_map(|k| k.agg),
                        win: kids.iter().find_map(|k| k.win),
                        outer: kids.iter().filter_map(|k| k.outer).min(),
                        lateral,
                    });
                }
                ComparisonReduction::Narrow { index, value } => {
                    let mut rewritten = kids.to_vec();
                    let literal_origin = rewritten[index].expr.origin;
                    rewritten[index] = Bound {
                        shown: None,
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Long(value)))
                                .with_origin(literal_origin),
                        ),
                        ty: arg_type(DataType::Long, false),
                        agg: None,
                        win: None,
                        outer: None,
                        lateral: Vec::new(),
                    };
                    return self.operator_call(op, origin, &rewritten);
                }
            }
        }
        let args: Vec<Arg<'_>> = kids.iter().map(arg_of).collect();
        let conf = self.ctx.conf;
        let rules = coerce::Rules {
            ansi: conf.ansi,
            case_sensitive: conf.case_sensitive,
        };
        let typed = coerce::operator(op, &args, rules, conf.decimal_precision_loss);
        let call = |args: Vec<Arc<Expr>>, data_type: DataType, nullable: bool| {
            let f = ResolvedFunction {
                id: FunctionId::Builtin(op),
                kind: FunctionKind::Scalar,
                args,
                distinct: false,
                filter: None,
                within_group: Vec::new(),
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

    /// Rearranges a call of a function with a named-parameter signature
    /// into positional order (Spark's `FunctionBuilderBase`
    /// rearrangement): positional arguments first, then named arguments in
    /// signature order with defaults filled in. Short positional calls are
    /// expanded with defaults too.
    ///
    /// `parts` names the routine in errors, as written (qualified when the
    /// call was). `kids` holds the bound arguments (no filter).
    fn rearrange_named(
        &self,
        parts: &[String],
        signature: &[NamedParam],
        f: &FunctionCall,
        kids: &[Bound],
        origin: Origin,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let routine = || quoted(parts);
        let param = |name: &str| quoted(std::slice::from_ref(&name.to_owned()));
        // Spark's split: positional arguments first, then named ones; a
        // positional argument after a named one, and a twice-named
        // parameter, are errors.
        let mut seen: Vec<&str> = Vec::new();
        if let Some(first) = f.arg_names.iter().position(Option::is_some) {
            for (i, name) in f.arg_names.iter().enumerate().skip(first) {
                match name {
                    None => {
                        let prev = i
                            .checked_sub(1)
                            .and_then(|j| f.arg_names.get(j))
                            .and_then(Option::as_deref)
                            .unwrap_or_default();
                        return Err(AnalysisError::new(
                            UnexpectedPositionalArgument {
                                routine_name: routine(),
                                parameter_name: param(prev),
                            },
                            origin,
                        ));
                    }
                    Some(name) => {
                        if seen.iter().any(|s| same(s, name, cs)) {
                            return Err(AnalysisError::new(
                                DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference {
                                    routine_name: routine(),
                                    parameter_name: param(name),
                                },
                                origin,
                            ));
                        }
                        seen.push(name);
                    }
                }
            }
        }
        let positional = f
            .arg_names
            .iter()
            .position(Option::is_some)
            .unwrap_or(kids.len());
        // Unrecognized names, and names that are both positional (by their
        // position) and named, are errors.
        for name in &seen {
            if !signature.iter().any(|p| same(p.name, name, cs)) {
                return Err(AnalysisError::new(
                    UnrecognizedParameterName {
                        routine_name: routine(),
                        argument_name: param(name),
                        proposal: suggestions(name, signature.iter().map(|p| p.name)),
                    },
                    origin,
                ));
            }
            if signature[..positional.min(signature.len())]
                .iter()
                .any(|p| same(p.name, name, cs))
            {
                return Err(AnalysisError::new(
                    DuplicateRoutineParameterAssignmentBothPositionalAndNamed {
                        routine_name: routine(),
                        parameter_name: param(name),
                    },
                    origin,
                ));
            }
        }
        // Spark's arity check over the valid sizes.
        let required = signature
            .iter()
            .take_while(|p| matches!(p.default, NamedDefault::Required))
            .count();
        if signature.len() < kids.len() {
            let valid: Vec<String> = (required..=signature.len())
                .map(|v| v.to_string())
                .collect();
            let expected = match valid.as_slice() {
                [only] => only.clone(),
                _ => format!("[{}]", valid.join(", ")),
            };
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: routine(),
                    expected_num: expected,
                    actual_num: kids.len().to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                origin,
            ));
        }
        // Positional arguments keep their order; the rest follows the
        // signature, filling defaults.
        let mut reordered: Vec<Bound> = kids[..positional.min(kids.len())].to_vec();
        for (i, p) in signature.iter().enumerate().skip(positional) {
            if let Some((_, kid)) =
                f.arg_names
                    .iter()
                    .zip(kids)
                    .find_map(|(name, kid)| match name {
                        Some(name) if same(name, p.name, cs) => Some((name, kid)),
                        _ => None,
                    })
            {
                reordered.push(kid.clone());
            } else if matches!(p.default, NamedDefault::Required) {
                return Err(AnalysisError::new(
                    RequiredParameterNotFound {
                        routine_name: routine(),
                        parameter_name: param(p.name),
                        index: (i - positional).to_string(),
                    },
                    origin,
                ));
            } else {
                let literal = match p.default {
                    NamedDefault::Null => Literal::Null(None),
                    NamedDefault::Text(s) => Literal::String(s.to_owned()),
                    NamedDefault::Required => {
                        return Err(unsupported("a required parameter default", origin));
                    }
                };
                let data_type = literal_type(&literal);
                let nullable = matches!(literal, Literal::Null(_));
                reordered.push(Bound {
                    expr: Arc::new(Expr::new(ExprNode::Literal(literal)).with_origin(origin)),
                    ty: ArgType {
                        data_type,
                        nullable,
                    },
                    agg: None,
                    win: None,
                    outer: None,
                    lateral: Vec::new(),
                    shown: None,
                });
            }
        }
        Ok(reordered)
    }

    /// `lower(b)` as a call of the registry's `lower` (`entry`), keeping the
    /// argument's aggregate, window and outer-reference bookkeeping.
    fn lowered(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        b: &Bound,
    ) -> Result<Bound, AnalysisError> {
        let (mut v, ty) = self.typed_call(entry, e, std::slice::from_ref(b))?;
        if v.is_empty() {
            return Err(unsupported("function lower", e.origin));
        }
        let arg = v.remove(0);
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        Ok(Bound {
            expr: Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: entry.id.clone(),
                        kind: entry.kind,
                        args: vec![Arc::clone(&arg.expr)],
                        distinct: false,
                        filter: None,
                        within_group: Vec::new(),
                        data_type,
                        nullable: ty.nullable,
                        deterministic: entry.deterministic,
                    },
                )))
                .with_origin(e.origin),
            ),
            ty,
            agg: arg.agg,
            win: arg.win,
            outer: arg.outer,
            lateral: arg.lateral,
            shown: None,
        })
    }

    /// `ilike(a, b[, escape])` as Spark's `ILike` replacement:
    /// `lower(a) LIKE lower(b)` with the same escape, shown as written.
    fn ilike_call(&mut self, e: &Arc<Expr>, kids: &[Bound]) -> Result<Bound, AnalysisError> {
        let n = kids.len();
        if n != 2 && n != 3 {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`ilike`".to_owned(),
                    expected_num: "[2, 3]".to_owned(),
                    actual_num: n.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        let FunctionLookup::Found(lower) = self
            .ctx
            .functions
            .lookup(&["lower".to_owned()], self.ctx.profile)
        else {
            return Err(unsupported("function lower", e.origin));
        };
        let origin = e.origin;
        let mut like_kids = vec![
            self.lowered(&lower, e, &kids[0])?,
            self.lowered(&lower, e, &kids[1])?,
        ];
        if n == 3 {
            like_kids.push(kids[2].clone());
        }
        let like = self.operator_call(Builtin::Like, origin, &like_kids)?;
        // Shown without the escape, as Spark shows `ILike`.
        let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(Builtin::ILike),
                kind: FunctionKind::Scalar,
                args: kids[..2].iter().map(Bound::shown).collect(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Boolean,
                nullable: like.ty.nullable,
                deterministic: true,
            },
        )));
        Ok(Bound {
            shown: Some(Arc::new(shown.with_origin(origin))),
            ..like
        })
    }

    /// `x [NOT] LIKE|ILIKE ANY|SOME|ALL (patterns)` as an `OR` (`ANY`/`SOME`)
    /// or `AND` (`ALL`) chain of single `LIKE`s (the `NOT` inside, as Spark
    /// negates each pattern match), with the case-insensitive variants built
    /// from the `ILIKE` rewrite. Shown in the quantified form.
    fn like_quantified_call(
        &mut self,
        which: Builtin,
        e: &Arc<Expr>,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        let Some(QuantifiedLike {
            insensitive,
            negated,
            all,
        }) = which.quantified_like()
        else {
            return Err(unsupported("a LIKE ANY, SOME or ALL form", e.origin));
        };
        let origin = e.origin;
        let Some((written, patterns)) = kids.split_first() else {
            return Err(unsupported("a LIKE ANY, SOME or ALL form", origin));
        };
        let child = written;
        let lower = if insensitive {
            let FunctionLookup::Found(entry) = self
                .ctx
                .functions
                .lookup(&["lower".to_owned()], self.ctx.profile)
            else {
                return Err(unsupported("function lower", origin));
            };
            Some(entry)
        } else {
            None
        };
        let wrap = |b: &Bound| match &lower {
            None => Ok(b.clone()),
            Some(entry) => self.lowered(entry, e, b),
        };
        let child = wrap(child)?;
        let mut lowered_patterns = Vec::with_capacity(patterns.len());
        for p in patterns {
            lowered_patterns.push(wrap(p)?);
        }
        let mut acc: Option<Bound> = None;
        for p in lowered_patterns {
            let like = self.operator_call(Builtin::Like, origin, &[child.clone(), p])?;
            let next = if negated {
                self.operator_call(Builtin::Not, origin, std::slice::from_ref(&like))?
            } else {
                like
            };
            acc = Some(match acc {
                None => next,
                Some(done) => self.operator_call(
                    if all { Builtin::And } else { Builtin::Or },
                    origin,
                    &[done, next],
                )?,
            });
        }
        let Some(acc) = acc else {
            return Err(unsupported("a LIKE ANY, SOME or ALL form", origin));
        };
        // Spark's name: with only non-NULL string literal patterns it builds
        // `LikeAny`/`LikeAll` (shown `likeany(child)`, the child lowered for
        // ILIKE); otherwise the chain of single `LIKE`s or `ilike` calls.
        let nullable = acc.ty.nullable;
        let shown_call = |op: Builtin, args: Vec<Arc<Expr>>| {
            Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: FunctionId::Builtin(op),
                        kind: FunctionKind::Scalar,
                        args,
                        distinct: false,
                        filter: None,
                        within_group: Vec::new(),
                        data_type: DataType::Boolean,
                        nullable,
                        deterministic: true,
                    },
                )))
                .with_origin(origin),
            )
        };
        let ansi = self.ctx.conf.ansi;
        let literal_patterns = patterns
            .iter()
            .all(|p| matches!(self.fold(&p.expr, ansi), Some(Constant::String(_))));
        let shown = if literal_patterns {
            shown_call(which, vec![child.shown()])
        } else {
            let single = if insensitive {
                Builtin::ILike
            } else {
                Builtin::Like
            };
            patterns
                .iter()
                .map(|p| {
                    let one = shown_call(single, vec![written.shown(), p.shown()]);
                    if negated {
                        shown_call(Builtin::Not, vec![one])
                    } else {
                        one
                    }
                })
                .reduce(|done, next| {
                    shown_call(
                        if all { Builtin::And } else { Builtin::Or },
                        vec![done, next],
                    )
                })
                .unwrap_or_else(|| shown_call(which, vec![child.shown()]))
        };
        Ok(Bound {
            shown: Some(shown),
            ..acc
        })
    }

    /// Spark's checks of the replacement characters of a `mask` call: each
    /// must fold to a single character (or `NULL`), quoted and cited at the
    /// argument. `kids` are the call's bound arguments before their implicit
    /// casts, `call_kids` after.
    fn check_mask_replacements(
        &self,
        lookup_name: &str,
        kids: &[Bound],
        call_kids: &[Bound],
    ) -> Result<(), AnalysisError> {
        const NAMES: [&str; 4] = ["upperChar", "lowerChar", "digitChar", "otherChar"];
        let zone = self.ctx.conf.session_zone;
        let ansi = self.ctx.conf.ansi;
        // The call as Spark quotes it: the rearranged positional form.
        let sql_expr = || {
            format!(
                "\"{}({})\"",
                lookup_name.to_lowercase(),
                call_kids
                    .iter()
                    .map(|k| pretty(&k.shown(), zone))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        for (i, name) in NAMES.iter().enumerate() {
            let (Some(arg), Some(written)) = (call_kids.get(1 + i), kids.get(1 + i)) else {
                return Err(unsupported(
                    "a mask call without its replacements",
                    call_kids
                        .first()
                        .map_or_else(Origin::default, |k| k.expr.origin),
                ));
            };
            match self.fold(&arg.expr, ansi) {
                Some(Constant::Null) => {}
                Some(Constant::String(s)) if s.chars().count() == 1 => {}
                Some(Constant::String(_)) => {
                    return Err(AnalysisError::new(
                        DatatypeMismatchInputSizeNotOne {
                            sql_expr: sql_expr(),
                            expr_name: (*name).to_owned(),
                        },
                        arg.expr.origin,
                    ));
                }
                Some(Constant::Error(err)) => {
                    return Err(AnalysisError::new(err, arg.expr.origin));
                }
                _ => {
                    return Err(AnalysisError::new(
                        DatatypeMismatchNonFoldableInput {
                            sql_expr: sql_expr(),
                            input_name: format!("`{name}`"),
                            input_type: "\"STRING\"".to_owned(),
                            input_expr: format!("\"{}\"", pretty(&written.shown(), zone)),
                        },
                        arg.expr.origin,
                    ));
                }
            }
        }
        Ok(())
    }

    /// `decode(expr, search1, result1, ..., [default])` (Oracle style): a
    /// `CASE` over null-safe equality, so a NULL `expr` matches a NULL
    /// search. A trailing unpaired argument is the default; without one the
    /// result is NULL. The column is named by the call as written.
    fn decode_case(&mut self, e: &Arc<Expr>, kids: &[Bound]) -> Result<Bound, AnalysisError> {
        let origin = e.origin;
        let mut parts = Vec::with_capacity(kids.len());
        for pair in kids[1..].chunks(2) {
            if let [search, result] = pair {
                let cond = self.operator_call(
                    Builtin::EqNullSafe,
                    origin,
                    &[kids[0].clone(), search.clone()],
                )?;
                parts.push(cond);
                parts.push(result.clone());
            } else {
                parts.extend(pair.iter().cloned());
            }
        }
        // Without a default Spark ends the CASE with a string-typed NULL,
        // which takes part in the result type (`typeof(decode(1, 1, 2))` is
        // `string` outside ANSI mode).
        if kids.len() % 2 == 1 {
            let null = Arc::new(
                Expr::new(ExprNode::Literal(Literal::Null(Some(DataType::String))))
                    .with_origin(origin),
            );
            parts.push(Bound {
                expr: null,
                ty: arg_type(DataType::String, true),
                agg: None,
                win: None,
                outer: None,
                lateral: Vec::new(),
                shown: None,
            });
        }
        let mut b = self.operator_call(Builtin::When, origin, &parts)?;
        b.carry_markers(kids);
        b.shown = Some(with_children(e, kids.iter().map(Bound::shown).collect()));
        Ok(b)
    }

    /// The field `name` of the struct `b`, as `b.name` or `b['name']`.
    fn struct_field(
        &self,
        e: &Expr,
        b: &Bound,
        name: &str,
    ) -> Result<(Arc<Expr>, ArgType), AnalysisError> {
        let st = match &b.ty.data_type {
            Some(DataType::Struct(st)) => st.clone(),
            _ => {
                let other =
                    b.ty.data_type
                        .as_ref()
                        .map_or_else(|| "VOID".to_owned(), DataType::sql);
                return Err(AnalysisError::new(
                    InvalidExtractBaseFieldType {
                        base: format!("\"{}\"", pretty(&b.shown(), self.ctx.conf.session_zone)),
                        other: format!("\"{other}\""),
                    },
                    e.origin,
                ));
            }
        };
        let cs = self.ctx.conf.case_sensitive;
        let Some((ordinal, field)) = st
            .fields()
            .iter()
            .enumerate()
            .find(|(_, f)| same(f.name(), name, cs))
        else {
            return Err(unresolved_field(
                &[
                    pretty(&b.shown(), self.ctx.conf.session_zone),
                    name.to_owned(),
                ],
                &st,
                e.origin,
            ));
        };
        let ty = arg_type(field.data_type().clone(), field.nullable() || b.ty.nullable);
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                base: Arc::clone(&b.expr),
                ordinal,
                name: name.to_owned(),
                data_type: ty.data_type.clone().unwrap_or(DataType::Null),
                nullable: ty.nullable,
            }))
            .with_origin(e.origin),
        );
        Ok((expr, ty))
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
        if matches!(which, Conditional::Decode) {
            // A scalar function has no `DISTINCT` or `FILTER` form.
            for (present, syntax) in [
                (f.distinct, "DISTINCT"),
                (f.filter.is_some(), "FILTER CLAUSE"),
            ] {
                if present {
                    return Err(AnalysisError::new(
                        InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                            pretty_name: "`decode`".to_owned(),
                            syntax: syntax.to_owned(),
                        },
                        e.origin,
                    ));
                }
            }
            return self.decode_case(e, &kids[..n]);
        }
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
                within_group: Vec::new(),
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
                        within_group: Vec::new(),
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
            Conditional::Decode => return self.decode_case(e, &kids[..n]),
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

    /// A cast-named function call (`float(x)`, `int(x)`, ...), rewritten
    /// during analysis to `CAST(x AS target)` in the session's default
    /// mode (Spark's `RuntimeReplaceable`): the plan only ever holds the
    /// `Cast` node, typed by the cast path. The output column is named by
    /// the argument, and cast errors quote the argument, as Spark does;
    /// the arity error quotes the call. A pair the cast path does not
    /// support fails exactly as `CAST(x AS target)` fails, apart from the
    /// quoted expression.
    fn cast_named_call(
        &mut self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        to: DataType,
    ) -> Result<Bound, AnalysisError> {
        let n = f.args.len();
        // The arity check quotes the call as written, as `typed_call` does.
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
        let arg = &kids[0];
        let mode = if self.ctx.conf.ansi {
            CastMode::Ansi
        } else {
            CastMode::Legacy
        };
        if let Err(bad) = coerce::can_cast(arg.ty.data_type.as_ref(), &to, mode) {
            let error = cast_mismatch(
                bad,
                format!("\"{}\"", pretty(&arg.shown(), self.ctx.conf.session_zone)),
                quoted_type(arg.ty.data_type.as_ref()),
                quoted_type(Some(&to)),
            );
            return Err(AnalysisError::new(error, e.origin));
        }
        let nullable = arg.ty.nullable
            || arg
                .ty
                .data_type
                .as_ref()
                .is_some_and(|from| coerce::force_nullable(from, &to));
        let expr = Arc::new(
            Expr::new(ExprNode::Cast {
                expr: Arc::clone(&arg.expr),
                to: to.clone(),
                mode,
            })
            .with_origin(e.origin),
        );
        Ok(Bound {
            expr,
            ty: arg_type(to, nullable),
            agg: arg.agg,
            win: arg.win,
            outer: arg.outer,
            lateral: arg.lateral.clone(),
            shown: Some(arg.shown()),
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
                    within_group: Vec::new(),
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
        let rules = coerce::Rules {
            ansi,
            case_sensitive: self.ctx.conf.case_sensitive,
        };
        match coerce::operator(
            Builtin::Eq,
            &args,
            rules,
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
            match coerce::common_type(&arg_of(k), &out, self.ctx.conf.case_sensitive) {
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
        // `mask` materializes its omitted arguments before typing (Spark's
        // constructor writes them into the call), so an input-type error
        // quotes the expanded call, as Spark does
        // (`mask(c1, replaceArg, x, n, NULL)`).
        let sql_expr = match function {
            "mask" if kids.len() == 5 => {
                let args: Vec<String> = kids.iter().map(|k| pretty(&k.shown(), zone)).collect();
                format!("\"{function}({})\"", args.join(", "))
            }
            _ => format!("\"{}\"", pretty(call, zone)),
        };
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
            Mismatch::DateDays { sub, date, ty } => {
                let shown = |i: usize| {
                    kids.get(i)
                        .map(|k| pretty(&k.shown(), zone))
                        .unwrap_or_default()
                };
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: format!(
                        "\"{}({}, {})\"",
                        if sub { "date_sub" } else { "date_add" },
                        shown(date),
                        shown(1 - date)
                    ),
                    param_index: ordinal(1),
                    required_type: r#"("INT" or "SMALLINT" or "TINYINT")"#.to_owned(),
                    input_sql: format!("\"{}\"", shown(1 - date)),
                    input_type: quoted_type(ty.as_ref()),
                }
                .into()
            }

            Mismatch::DateAddText => {
                let shown = |i: usize| {
                    kids.get(i)
                        .map(|k| pretty(&k.shown(), zone))
                        .unwrap_or_default()
                };
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: format!("\"date_add({}, {})\"", shown(0), shown(1)),
                    param_index: ordinal(1),
                    required_type: r#"("INT" or "SMALLINT" or "TINYINT")"#.to_owned(),
                    input_sql: format!("\"{}\"", shown(1)),
                    input_type: r#""DATE""#.to_owned(),
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
            Mismatch::CreateMapKeyDiff(types) => DatatypeMismatchCreateMapKeyDiffTypes {
                sql_expr,
                function_name: "`map`".to_owned(),
                data_type: list_types(&types),
            }
            .into(),
            Mismatch::CreateMapValueDiff(types) => DatatypeMismatchCreateMapValueDiffTypes {
                sql_expr,
                function_name: "`map`".to_owned(),
                data_type: list_types(&types),
            }
            .into(),
            Mismatch::MapConcatDiffTypes(types) => DatatypeMismatchMapConcatDiffTypes {
                sql_expr,
                function_name: "`map_concat`".to_owned(),
                data_type: list_types(&types),
            }
            .into(),
            Mismatch::InvalidOrdering(t) => DatatypeMismatchInvalidOrderingType {
                sql_expr,
                function_name: quoted(&[function.to_owned()]),
                data_type: quoted_type(Some(&t)),
            }
            .into(),
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
        if c.plan_id.is_none()
            && let [name, _, ..] = &c.name[..]
            && let Some(var) = self.lambda_var(name)
        {
            // `x.field` of a lambda variable `x`: a field of its struct.
            let attr = Attr {
                col: var,
                qual: Arc::from(Vec::new()),
                source: None,
            };
            return self.resolve_attr_as(&attr, 1, &c.name, origin, None, true);
        }
        if c.plan_id.is_none()
            && let [name] = &c.name[..]
            && let Some(var) = self.lambda_var(name)
        {
            // A lambda variable shadows the enclosing query's columns.
            let mut col = var;
            col.name.clone_from(name);
            return Ok(bound(
                ExprNode::Resolved(ResolvedExpr::LambdaVar(col.clone())),
                &col,
                None,
            ));
        }
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
        // With leniency (an `ORDER BY` over duplicate outputs, a `HAVING`
        // or `ORDER BY` over an aggregate whose select list repeats a
        // name), duplicates in the first level are looked up below it; when
        // nothing below resolves them either, they stay ambiguous rather
        // than unresolved.
        let mut lenient_dup = false;
        for (i, level) in [Some(scope), extra].into_iter().enumerate() {
            let Some(level) = level else { continue };
            let lenient = cx.lenient && i == 0;
            match self.find(level, &c.name, origin, lenient) {
                Ok(Some((a, used))) => {
                    self.used(&a);
                    return self.resolve_attr(&a, used, &c.name, origin, None);
                }
                Ok(None) => {
                    lenient_dup = lenient_dup || lenient;
                }
                Err(e) => return Err(e),
            }
        }
        if lenient_dup {
            // Duplicates in the first level, nothing below resolved them:
            // they stay ambiguous rather than unresolved, before a
            // function, alias, outer column or session variable can match.
            self.find(scope, &c.name, origin, false)?;
        }
        // A name no input column has may be a function written without
        // parentheses (what the parser leaves a column unless ANSI reserved
        // keywords are on). Spark 4.2 tries it before lateral aliases,
        // GROUP BY aliases and outer references.
        if let [name] = &c.name[..]
            && !(self.ctx.conf.udf_parameter_shadows_function && self.has_parameter(name))
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
                    qualifier: Vec::new(),
                    arg_names: Vec::new(),
                    args: Vec::new(),
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
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
                return self.resolve_attr(&a, used, &c.name, origin, Some(level));
            }
        }
        // `grouping__id` is the grouping id Spark's aggregate rule reads
        // before a function parameter of that name can: outside a grouping
        // context it is `UNSUPPORTED_GROUPING_EXPRESSION`.
        if let [name] = &c.name[..]
            && name.eq_ignore_ascii_case("grouping__id")
            && !cx.grouping
            && !self.ctx.conf.udf_parameter_shadows_function
            && self.has_parameter(name)
        {
            return Err(AnalysisError::new(UnsupportedGroupingExpression {}, origin));
        }
        // The parameters of the SQL function being resolved rank after
        // everything else, but before a session variable.
        if let Some(b) = self.parameter(&c.name, origin, cx.env.outer.len())? {
            return Ok(b);
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
        if self.parameter_attrs().is_empty() {
            return Err(unresolved(&c.name, cx.scope, origin));
        }
        // A parameter the name matches is out of reach here (the body
        // nests deeper than a parameter is visible), and Spark proposes
        // nothing for it.
        let cs = self.ctx.conf.case_sensitive;
        let candidates: Vec<Attr> = cx
            .scope
            .iter()
            .chain(
                self.parameter_attrs()
                    .iter()
                    .filter(|a| !matches!(&c.name[..], [n] if same(&a.col.name, n, cs))),
            )
            .cloned()
            .collect();
        Err(unresolved(&c.name, &candidates, origin))
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
    /// matches are `AMBIGUOUS_REFERENCE`, or no match when `lenient`. A
    /// single match with leftover parts resolves them as struct fields in
    /// [`Self::resolve_attr`].
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
                    // A longer path may name struct fields below the
                    // column; the caller resolves them (or reports the
                    // base type when the column is not a struct).
                    return Ok(Some((a.clone(), q + 1)));
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

    /// `struct` with named children as `named_struct`, or `None` when every
    /// child defaults to `colN`: a column contributes its (written) name, a
    /// struct field its field name, an alias its output name.
    fn struct_names(&self, f: &FunctionCall, kids: &[Bound], origin: Origin) -> Option<Arc<Expr>> {
        let mut names = Vec::with_capacity(kids.len());
        for (i, k) in kids.iter().enumerate() {
            let name = match &k.expr.node {
                ExprNode::Resolved(
                    ResolvedExpr::Column(c)
                    | ResolvedExpr::OuterColumn(c)
                    | ResolvedExpr::LambdaVar(c),
                ) => c.name.clone(),
                ExprNode::Resolved(ResolvedExpr::Field { name, .. }) => name.clone(),
                ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
                _ => format!("col{}", i + 1),
            };
            names.push(name);
        }
        if names
            .iter()
            .enumerate()
            .all(|(i, n)| *n == format!("col{}", i + 1))
            && !f
                .args
                .iter()
                .any(|a| matches!(a.node, ExprNode::Alias { .. }))
        {
            return None;
        }
        let mut args = Vec::with_capacity(2 * kids.len());
        for (name, k) in names.into_iter().zip(kids) {
            args.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::String(name))).with_origin(origin),
            ));
            args.push(Arc::clone(&k.expr));
        }
        Some(Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                name: "named_struct".to_owned(),
                qualifier: Vec::new(),
                args,
                arg_names: Vec::new(),
                distinct: f.distinct,
                filter: None,
                within_group: Vec::new(),
            }))
            .with_origin(origin),
        ))
    }

    /// `UpdateFields(base, ops)` (PySpark `withField`/`dropFields`): the
    /// struct is rebuilt by `named_struct`, keeping every field the ops do
    /// not touch, so the executor and the kernels get nothing new. A `With`
    /// replaces every field matching its name (compared per
    /// `spark.sql.caseSensitive`) in place, else appends the field last; a
    /// `Drop` removes every matching field and ignores names that match
    /// nothing. The rewrite is `CASE WHEN base IS NULL THEN CAST(NULL AS
    /// <new type>) ELSE named_struct(...) END`, bound here, so its field
    /// order, names, types and nullability are the rewritten
    /// `named_struct`'s and no resolved plan holds the node. The base
    /// appears several times in the rewrite: a nondeterministic base would
    /// read a different value per use, and nothing below binds it once, so
    /// that is `KRISHIV_UNSUPPORTED_FEATURE`.
    fn update_fields(
        &mut self,
        e: &Arc<Expr>,
        ops: &[FieldOp],
        kids: &[Bound],
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let base = &kids[0];
        let struct_type = match &base.ty.data_type {
            Some(DataType::Struct(st)) => st.clone(),
            _ => {
                let shown = with_children(e, kids.iter().map(Bound::shown).collect());
                return Err(AnalysisError::new(
                    DatatypeMismatchUnexpectedInputType {
                        sql_expr: format!("\"{}\"", pretty(&shown, self.ctx.conf.session_zone)),
                        param_index: ordinal(0),
                        required_type: "\"STRUCT\"".to_owned(),
                        input_sql: format!(
                            "\"{}\"",
                            pretty(&base.shown(), self.ctx.conf.session_zone)
                        ),
                        input_type: quoted_type(base.ty.data_type.as_ref()),
                    },
                    e.origin,
                ));
            }
        };
        if !is_deterministic(&base.expr) {
            return Err(unsupported(
                "withField/dropFields of a nondeterministic expression",
                e.origin,
            ));
        }
        let cs = self.ctx.conf.case_sensitive;
        // One output field per kept struct field (a field extraction of the
        // resolved base, as the SQL path reads it), then the appended ones.
        let mut out: Vec<(String, Arc<Expr>, DataType, bool)> = Vec::new();
        for (ordinal, field) in struct_type.fields().iter().enumerate() {
            let nullable = field.nullable() || base.ty.nullable;
            let value = Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                    base: Arc::clone(&base.expr),
                    ordinal,
                    name: field.name().to_owned(),
                    data_type: field.data_type().clone(),
                    nullable,
                }))
                .with_origin(e.origin),
            );
            out.push((
                field.name().to_owned(),
                value,
                field.data_type().clone(),
                nullable,
            ));
        }
        let mut values = kids[1..].iter();
        for op in ops {
            match op {
                FieldOp::With { name, .. } => {
                    let Some(v) = values.next() else {
                        return Err(unsupported("a struct update without its value", e.origin));
                    };
                    let updated = (
                        name.clone(),
                        Arc::clone(&v.expr),
                        v.ty.data_type.clone().unwrap_or(DataType::Null),
                        v.ty.nullable,
                    );
                    let mut matched = false;
                    for slot in out.iter_mut() {
                        if same(&slot.0, name, cs) {
                            *slot = updated.clone();
                            matched = true;
                        }
                    }
                    if !matched {
                        out.push(updated);
                    }
                }
                FieldOp::Drop { name } => {
                    out.retain(|slot| !same(&slot.0, name, cs));
                }
            }
        }
        if out.is_empty() {
            return Err(AnalysisError::new(
                DatatypeMismatchCannotDropAllFields {
                    sql_expr: format!(
                        "\"{}\"",
                        pretty(
                            &with_children(e, kids.iter().map(Bound::shown).collect()),
                            self.ctx.conf.session_zone
                        )
                    ),
                },
                e.origin,
            ));
        }
        let new_type = DataType::Struct(StructType::new(
            out.iter()
                .map(|(name, _, t, n)| StructField::new(name.clone(), t.clone(), *n)),
        ));
        // `named_struct(name0, val0, ...)`: bound here, so the output field
        // names, types and nullability are the analyzer's own.
        let mut struct_args: Vec<Arc<Expr>> = Vec::with_capacity(2 * out.len());
        for (name, value, _, _) in &out {
            struct_args.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::String(name.clone()))).with_origin(e.origin),
            ));
            struct_args.push(Arc::clone(value));
        }
        let struct_call = Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                within_group: Vec::new(),
                name: "named_struct".to_owned(),
                qualifier: Vec::new(),
                args: struct_args,
                arg_names: Vec::new(),
                distinct: false,
                filter: None,
            }))
            .with_origin(e.origin),
        );
        let rebuilt = self.expr(&struct_call, cx)?;
        let cond = self.operator_call(Builtin::IsNull, e.origin, std::slice::from_ref(base))?;
        let null_branch = {
            let null =
                Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(e.origin));
            let cast = Arc::new(
                Expr::new(ExprNode::Cast {
                    expr: null,
                    to: new_type,
                    mode: CastMode::Default,
                })
                .with_origin(e.origin),
            );
            self.expr(&cast, cx)?
        };
        let mut b = self.operator_call(Builtin::When, e.origin, &[cond, null_branch, rebuilt])?;
        // The field values were bound before and rebinding them in the
        // `named_struct` dropped their markers (an aggregate, a window
        // expression, an outer or lateral reference): carry them over.
        b.carry_markers(kids);
        // Shown as written (`update_fields(...)`), not the rewrite.
        b.shown = Some(with_children(e, kids.iter().map(Bound::shown).collect()));
        Ok(b)
    }

    /// The attribute `a` matched with `used` of `parts`: the column itself,
    /// or the struct fields the leftover parts name below it (Spark's
    /// `GetStructField` chain), resolved under `spark.sql.caseSensitive`.
    /// `outer` is `Some` for a column of the enclosing query.
    fn resolve_attr(
        &mut self,
        a: &Attr,
        used: usize,
        parts: &[String],
        origin: Origin,
        outer: Option<usize>,
    ) -> Result<Bound, AnalysisError> {
        self.resolve_attr_as(a, used, parts, origin, outer, false)
    }

    /// [`Self::resolve_attr`]; `lambda` makes the base a lambda variable
    /// (`x.field` inside a higher-order function's lambda).
    fn resolve_attr_as(
        &mut self,
        a: &Attr,
        used: usize,
        parts: &[String],
        origin: Origin,
        outer: Option<usize>,
        lambda: bool,
    ) -> Result<Bound, AnalysisError> {
        // A reference keeps the spelling it was written with.
        let mut col = a.col.clone();
        col.name.clone_from(&parts[used - 1]);
        let node = match outer {
            _ if lambda => ExprNode::Resolved(ResolvedExpr::LambdaVar(col.clone())),
            None => ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
            Some(_) => ExprNode::Resolved(ResolvedExpr::OuterColumn(col.clone())),
        };
        if used == parts.len() {
            return Ok(Bound {
                expr: Arc::new(Expr::new(node).with_origin(origin)),
                ty: column(&col),
                agg: None,
                win: None,
                outer,
                lateral: Vec::new(),
                shown: None,
            });
        }
        let cs = self.ctx.conf.case_sensitive;
        let mut ty = column(&col);
        let mut base = Arc::new(Expr::new(node).with_origin(origin));
        let mut parent = parts[used - 1].clone();
        for part in &parts[used..] {
            let st = match &ty.data_type {
                Some(DataType::Struct(st)) => st.clone(),
                _ => {
                    let other = ty
                        .data_type
                        .as_ref()
                        .map_or_else(|| "VOID".to_owned(), DataType::sql);
                    return Err(AnalysisError::new(
                        InvalidExtractBaseFieldType {
                            base: format!("\"{parent}\""),
                            other: format!("\"{other}\""),
                        },
                        origin,
                    ));
                }
            };
            let Some((ordinal, field)) = st
                .fields()
                .iter()
                .enumerate()
                .find(|(_, f)| same(f.name(), part, cs))
            else {
                return Err(unresolved_field(parts, &st, origin));
            };
            parent.clone_from(part);
            let field_ty = arg_type(field.data_type().clone(), field.nullable() || ty.nullable);
            base = Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                    base: Arc::clone(&base),
                    ordinal,
                    name: part.clone(),
                    data_type: field_ty.data_type.clone().unwrap_or(DataType::Null),
                    nullable: field_ty.nullable,
                }))
                .with_origin(origin),
            );
            ty = field_ty;
        }
        Ok(Bound {
            expr: base,
            ty,
            agg: None,
            win: None,
            outer,
            lateral: Vec::new(),
            shown: None,
        })
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

    /// The ordered-set and quantile aggregates (`percentile`,
    /// `percentile_approx`, `approx_percentile`, `median`,
    /// `percentile_cont`, `percentile_disc`, `approx_count_distinct`),
    /// typed here rather than through the registry: Spark's builders fold
    /// the percentage, accuracy, frequency and `relativeSD` while typing,
    /// lower `WITHIN GROUP` to the ordering key, and fill the frequency
    /// and accuracy defaults. The executor reads the folded literals.
    ///
    /// The resolved call keeps the lowered arguments (the value, the cast
    /// percentage, the frequency or accuracy, the descending flag); the
    /// shown form keeps the call as written (with `WITHIN GROUP`), which
    /// names the output and quotes errors, as Spark's `toPrettySQL` does.
    #[inline(never)]
    fn quantile_call(
        &mut self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        n: usize,
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        use krishiv_functions::registry::{approx_percentile_result, exact_percentile_result};
        let name = entry.id.name();
        let origin = e.origin;
        let zone = self.ctx.conf.session_zone;
        let wrong_args = |actual: usize| {
            AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{name}`"),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: actual.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                origin,
            )
        };
        if !entry.arity.accepts(f.args.len()) {
            return Err(wrong_args(f.args.len()));
        }
        let ordered = matches!(name, "percentile_cont" | "percentile_disc");
        if ordered {
            if f.within_group.is_empty() {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWithinGroupMissing {
                        func_name: format!("`{name}`"),
                    },
                    origin,
                ));
            }
            if f.distinct {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionDistinctUnsupported {
                        func_name: format!("`{name}`"),
                    },
                    origin,
                ));
            }
            if f.within_group.len() != 1 {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWrongNumOrderings {
                        func_name: format!("`{name}`"),
                        expected_num: "1".to_owned(),
                        actual_num: f.within_group.len().to_string(),
                    },
                    origin,
                ));
            }
        }
        // The nesting and window checks see the filter and the ordering key
        // too (a filter sits between the arguments and the ordering keys).
        call_checks(entry, e, kids, kids.len())?;
        let filter = kids.get(n).filter(|_| f.filter.is_some());
        let wstart = kids.len() - f.within_group.len();
        // The lowered call's SQL as Spark quotes it in errors: `name(args)`
        // over the shown arguments (the frequency and accuracy defaults
        // included, the descending flag excluded).
        let lowered_sql = |args: &[Arc<Expr>]| {
            format!(
                "\"{name}({})\"",
                args.iter()
                    .map(|a| pretty(a, zone))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        // The value (or ordering key): Spark's numeric input, which the
        // approximate path extends to dates and timestamps.
        let value = if ordered {
            kids[wstart].clone()
        } else {
            kids[0].clone()
        };
        let approx = matches!(name, "percentile_approx" | "approx_percentile");
        let temporal = approx
            && matches!(
                value.ty.data_type,
                Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
            );
        // `approx_count_distinct` hashes a value of any type as it is: no
        // numeric requirement and no cast (the executor refuses what it
        // cannot hash).
        let hashes_any_type = name == "approx_count_distinct";
        let child_target: Option<DataType> = if temporal || hashes_any_type {
            None
        } else {
            let given = [value.ty.clone()];
            let call = CallTypes::new(&given, &[]);
            let inputs = Inputs::Expects(vec![InputType::Numeric { intervals: true }]);
            let arg = arg_of(&value);
            match coerce::inputs(&inputs, &[arg], &call) {
                Ok(casts) => casts.into_iter().next().flatten(),
                Err(m) => {
                    let lowered = Arc::new(
                        Expr::new(ExprNode::Function(FunctionCall {
                            name: name.to_owned(),
                            qualifier: Vec::new(),
                            args: vec![value.shown()],
                            arg_names: Vec::new(),
                            distinct: false,
                            filter: None,
                            within_group: Vec::new(),
                        }))
                        .with_origin(origin),
                    );
                    return Err(self.mismatch(
                        m,
                        &lowered,
                        std::slice::from_ref(&value),
                        name,
                        origin,
                    ));
                }
            }
        };
        let mut value = value;
        if child_target.is_some() {
            value = self.implicit_cast(&value, child_target);
        }
        // `median` fixes its percentage at 0.5 and `approx_count_distinct`
        // takes none: neither reads the percentage argument.
        let median = name == "median";
        let appr = name == "approx_count_distinct";
        // Spark quotes errors over the lowered call, with defaulted
        // arguments filled in: the frequency, or the accuracy. The
        // ordered-set calls lower to two arguments.
        let third_shown: Option<Arc<Expr>> = if ordered || appr {
            None
        } else {
            let default = if approx {
                Literal::Integer(10000)
            } else {
                Literal::Long(1)
            };
            Some(if f.args.len() == 3 {
                kids[2].shown()
            } else {
                Arc::new(Expr::new(ExprNode::Literal(default)).with_origin(origin))
            })
        };
        // The percentage: a double or an array of doubles after the implicit
        // cast, folded here; the analyzer, not the executor, validates it.
        let pct_index = if ordered { 0 } else { 1 };
        // The lowered call as Spark quotes percentage errors: with the
        // defaulted third argument filled in.
        let pct_sql = |pct: &Bound| match &third_shown {
            Some(t) => lowered_sql(&[value.shown(), pct.shown(), Arc::clone(t)]),
            None => lowered_sql(&[value.shown(), pct.shown()]),
        };
        let half = || Bound {
            expr: Arc::new(Expr::new(ExprNode::Literal(Literal::Double(0.5))).with_origin(origin)),
            ty: arg_type(DataType::Double, false),
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        };
        let (mut pct, mut array, mut percentages) = (half(), false, vec![0.5]);
        if !median && !appr {
            pct = kids[pct_index].clone();
            let pct_type = pct.ty.data_type.clone();
            array = matches!(pct_type, Some(DataType::Array(_)));
            if array {
                let (element, contains_null) = match &pct_type {
                    Some(DataType::Array(a)) => (a.element_type().clone(), a.contains_null()),
                    _ => (DataType::Null, true),
                };
                let castable = !contains_null
                    && matches!(
                        element,
                        DataType::Byte
                            | DataType::Short
                            | DataType::Integer
                            | DataType::Long
                            | DataType::Float
                            | DataType::Double
                            | DataType::Decimal(_)
                            | DataType::Null
                    )
                    || (!self.ctx.conf.ansi && element == DataType::String);
                // (Under ANSI Spark refuses an array of strings, where it casts a
                // string percentage.)
                if !castable {
                    let sql = pct_sql(&pct);
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: sql,
                            param_index: ordinal(1),
                            required_type: "\"ARRAY<DOUBLE>\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&pct.shown(), zone)),
                            input_type: quoted_type(pct_type.as_ref()),
                        },
                        origin,
                    ));
                }
            } else {
                let castable = matches!(
                    pct_type,
                    None | Some(
                        DataType::Byte
                            | DataType::Short
                            | DataType::Integer
                            | DataType::Long
                            | DataType::Float
                            | DataType::Double
                            | DataType::Decimal(_)
                            | DataType::String
                            | DataType::Null
                    )
                );
                if !castable {
                    let sql = pct_sql(&pct);
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: sql,
                            param_index: ordinal(1),
                            required_type: "\"DOUBLE\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&pct.shown(), zone)),
                            input_type: quoted_type(pct_type.as_ref()),
                        },
                        origin,
                    ));
                }
            }
            let pct_target = if array {
                DataType::Array(ArrayType::new(DataType::Double, false).into())
            } else {
                DataType::Double
            };
            let pct = self.implicit_cast(&pct, Some(pct_target));
            let conf = fold::Conf::new(
                self.ctx.conf.ansi,
                self.ctx.conf.decimal_precision_loss,
                self.ctx.conf.map_key_dedup_last_win,
                (
                    self.ctx.conf.legacy_java_charsets,
                    self.ctx.conf.legacy_coding_error_action,
                ),
                self.ctx.profile,
                self.ctx.conf.session_zone.and_then(SessionZone::fixed),
            );
            // An array percentage folds elementwise: a cast of the whole array
            // does not fold, so each element of the array call is cast to
            // double and folded on its own (as Spark converts its array).
            // Anything else folds as one value.
            let folded: Vec<Option<f64>> = if array {
                match array_elements(&pct.expr) {
                    Some(elements) => {
                        let mut out = Vec::with_capacity(elements.len());
                        for el in elements {
                            let cast = Arc::new(
                                Expr::new(ExprNode::Cast {
                                    expr: el,
                                    to: DataType::Double,
                                    mode: if self.ctx.conf.ansi {
                                        crate::CastMode::Ansi
                                    } else {
                                        crate::CastMode::Legacy
                                    },
                                })
                                .with_origin(origin),
                            );
                            match fold::doubles(
                                &cast,
                                conf,
                                &mut self.folds.borrow_mut(),
                                self.ctx.functions.as_ref(),
                            ) {
                                None => {
                                    return Err(non_foldable(
                                        name, &value, &pct, array, zone, origin,
                                    ));
                                }
                                Some(Err(e)) => return Err(AnalysisError::new(e, origin)),
                                Some(Ok(mut v)) => {
                                    if v.len() != 1 {
                                        return Err(non_foldable(
                                            name, &value, &pct, array, zone, origin,
                                        ));
                                    }
                                    out.append(&mut v);
                                }
                            }
                        }
                        out
                    }
                    None => {
                        return Err(non_foldable(name, &value, &pct, array, zone, origin));
                    }
                }
            } else {
                match fold::doubles(
                    &pct.expr,
                    conf,
                    &mut self.folds.borrow_mut(),
                    self.ctx.functions.as_ref(),
                ) {
                    None => {
                        return Err(non_foldable(name, &value, &pct, array, zone, origin));
                    }
                    Some(Err(e)) => return Err(AnalysisError::new(e, origin)),
                    Some(Ok(v)) => v,
                }
            };
            if folded.iter().any(Option::is_none) {
                let sql = pct_sql(&pct);
                return Err(AnalysisError::new(
                    DatatypeMismatchUnexpectedNull {
                        expr_name: "percentage".to_owned(),
                        sql_expr: sql,
                    },
                    origin,
                ));
            }
            percentages = folded.into_iter().flatten().collect();
            if percentages.iter().any(|&v| v < 0.0 || v > 1.0) {
                let sql = pct_sql(&pct);
                let current: Vec<String> = percentages
                    .iter()
                    .map(|v| format!("{}D", crate::spark_fmt::double(*v)))
                    .collect();
                return Err(AnalysisError::new(
                    DatatypeMismatchValueOutOfRange {
                        expr_name: "percentage".to_owned(),
                        value_range: "[0.0, 1.0]".to_owned(),
                        current_value: current.join(","),
                        sql_expr: sql,
                    },
                    origin,
                ));
            }
        }
        // The executor reads the folded percentage literals (Spark folds
        // them while typing too): a `double`, or an `array` of them.
        let pct_lit = if array {
            let elements: Vec<Arc<Expr>> = percentages
                .iter()
                .map(|v| {
                    Arc::new(Expr::new(ExprNode::Literal(Literal::Double(*v))).with_origin(origin))
                })
                .collect();
            let FunctionLookup::Found(entry) = self
                .ctx
                .functions
                .lookup(&["array".to_owned()], self.ctx.profile)
            else {
                return Err(super::bind::unsupported("the array function", origin));
            };
            Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: entry.id,
                        kind: FunctionKind::Scalar,
                        args: elements,
                        distinct: false,
                        filter: None,
                        within_group: Vec::new(),
                        data_type: DataType::Array(ArrayType::new(DataType::Double, false).into()),
                        nullable: false,
                        deterministic: true,
                    },
                )))
                .with_origin(origin),
            )
        } else {
            // An empty array is legal (Spark yields NULL for it); a scalar
            // percentage always folds to one value (NULL fails above).
            let v = percentages.first().copied().unwrap_or(f64::NAN);
            Arc::new(Expr::new(ExprNode::Literal(Literal::Double(v))).with_origin(origin))
        };
        // The frequency, accuracy, `relativeSD` and direction, per function.
        // The resolved call keeps the lowered arguments; the shown form keeps
        // the call as written (with `WITHIN GROUP`), which names the output.
        let mut resolved: Vec<Arc<Expr>> = vec![Arc::clone(&value.expr)];
        let mut shown: Vec<Arc<Expr>> = vec![value.shown()];
        let data_type: DataType;
        let mut nullable = true;
        if ordered {
            // The percentage (already validated) and the descending flag.
            // The shown form keeps the call as written: the percentage
            // argument, then `WITHIN GROUP`.
            resolved.push(Arc::clone(&pct_lit));
            shown = vec![kids[0].shown()];
            let descending = matches!(
                f.within_group[0].direction,
                crate::SortDirection::Descending
            );
            resolved.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::Boolean(descending))).with_origin(origin),
            ));
            // `percentile_disc` reads the session's legacy calculation
            // (`spark.sql.legacy.percentileDiscCalculation`); the executor
            // takes it as a literal like the descending flag.
            resolved.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::Boolean(
                    self.ctx.conf.legacy_percentile_disc,
                )))
                .with_origin(origin),
            ));
            data_type = exact_percentile_result(array);
        } else if median {
            data_type = exact_percentile_result(false);
        } else if appr {
            // `relativeSD`: a double (or decimal) literal, as Spark's
            // `validateDoubleLiteral` requires; the default is 0.05. It is
            // the second SQL argument, not the filter.
            let rsd = match (f.args.len() == 2).then(|| kids.get(1)).flatten() {
                None => Bound {
                    expr: Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Double(0.05))).with_origin(origin),
                    ),
                    ty: arg_type(DataType::Double, false),
                    agg: None,
                    win: None,
                    outer: None,
                    lateral: Vec::new(),
                    shown: None,
                },
                Some(k) => k.clone(),
            };
            // The written value as a double (`0.5` is a decimal literal).
            let written = match &rsd.expr.node {
                ExprNode::Literal(Literal::Double(d)) => Some(*d),
                ExprNode::Literal(Literal::Decimal {
                    unscaled,
                    data_type,
                }) => Some(*unscaled as f64 / 10_f64.powi(i32::from(data_type.scale()))),
                _ => None,
            };
            let Some(written) = written else {
                return Err(AnalysisError::new(LegacyErrorTemp1104 {}, origin));
            };
            // Spark's `require(p >= 4)` on the addressing bits
            // (`p = ceil(2 * log2(1.106 / relativeSD))`) fails the call when
            // the error is above about 39%.
            let p = (2.0 * (1.106 / written).ln() / std::f64::consts::LN_2).ceil();
            if written.is_finite() && written > 0.0 && p < 4.0 {
                return Err(AnalysisError::new(
                    FailedFunctionCall {
                        func_name: "`approx_count_distinct`".to_owned(),
                    },
                    origin,
                ));
            }
            // The executor derives the addressing bits and refuses precisions
            // Spark's `require` refuses.
            resolved.push(Arc::clone(&rsd.expr));
            data_type = DataType::Long;
            nullable = false;
        } else {
            // `percentile` and `percentile_approx` share the percentage.
            resolved.push(Arc::clone(&pct_lit));
            shown.push(pct.shown());
            if approx {
                // The accuracy: the third argument, or 10000 (Spark's
                // builder default); it must be a foldable integral in
                // `(0, 2147483647]`. It is the third SQL argument, not the
                // filter.
                let accuracy = match (f.args.len() == 3).then(|| kids.get(2)).flatten() {
                    Some(k) => k.clone(),
                    None => Bound {
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Integer(10000)))
                                .with_origin(origin),
                        ),
                        ty: arg_type(DataType::Integer, false),
                        agg: None,
                        win: None,
                        outer: None,
                        lateral: Vec::new(),
                        shown: None,
                    },
                };
                let accuracy_sql = lowered_sql(&[value.shown(), pct.shown(), accuracy.shown()]);
                let acc_type = accuracy.ty.data_type.clone();
                if !matches!(
                    acc_type,
                    None | Some(
                        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long
                    )
                ) {
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: accuracy_sql,
                            param_index: ordinal(2),
                            required_type: "\"INTEGRAL\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&accuracy.shown(), zone)),
                            input_type: quoted_type(acc_type.as_ref()),
                        },
                        origin,
                    ));
                }
                let n = match self.fold(&accuracy.expr, self.ctx.conf.ansi) {
                    Some(Constant::Integral(n)) => n,
                    Some(Constant::Null) => {
                        return Err(AnalysisError::new(
                            DatatypeMismatchUnexpectedNull {
                                expr_name: "accuracy".to_owned(),
                                sql_expr: accuracy_sql,
                            },
                            origin,
                        ));
                    }
                    Some(Constant::Error(e)) => return Err(AnalysisError::new(e, origin)),
                    _ => {
                        return Err(AnalysisError::new(
                            DatatypeMismatchNonFoldableInput {
                                sql_expr: accuracy_sql,
                                input_name: "`accuracy`".to_owned(),
                                input_type: quoted_type(acc_type.as_ref()),
                                input_expr: format!("\"{}\"", pretty(&accuracy.shown(), zone)),
                            },
                            origin,
                        ));
                    }
                };
                if n <= 0 || n > i64::from(i32::MAX) {
                    return Err(AnalysisError::new(
                        DatatypeMismatchValueOutOfRange {
                            expr_name: "accuracy".to_owned(),
                            value_range: "(0, 2147483647]".to_owned(),
                            current_value: n.to_string(),
                            sql_expr: accuracy_sql,
                        },
                        origin,
                    ));
                }
                // The executor reads the folded literal.
                resolved.push(Arc::new(
                    Expr::new(ExprNode::Literal(Literal::Long(n))).with_origin(origin),
                ));
                // Spark names the call with the defaulted accuracy.
                shown.push(accuracy.shown());
                let child = value.ty.data_type.clone().unwrap_or(DataType::Null);
                data_type = approx_percentile_result(&child, array);
            } else {
                // The frequency: the third argument, or 1; a per-row weight
                // which stays an expression (negative values fail per row).
                // It is the third SQL argument, not the filter.
                let frequency = match (f.args.len() == 3).then(|| kids.get(2)).flatten() {
                    Some(k) => k.clone(),
                    None => Bound {
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Long(1))).with_origin(origin),
                        ),
                        ty: arg_type(DataType::Long, false),
                        agg: None,
                        win: None,
                        outer: None,
                        lateral: Vec::new(),
                        shown: None,
                    },
                };
                let freq_type = frequency.ty.data_type.clone();
                if !matches!(
                    freq_type,
                    None | Some(
                        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long
                    )
                ) {
                    let sql = lowered_sql(&[value.shown(), pct.shown(), frequency.shown()]);
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: sql,
                            param_index: ordinal(2),
                            required_type: "\"INTEGRAL\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&frequency.shown(), zone)),
                            input_type: quoted_type(freq_type.as_ref()),
                        },
                        origin,
                    ));
                }
                resolved.push(Arc::clone(&frequency.expr));
                // Spark names the call with the defaulted frequency.
                shown.push(frequency.shown());
                data_type = exact_percentile_result(array);
            }
        }
        let shown_within: Vec<SortOrder> = f
            .within_group
            .iter()
            .zip(kids[wstart..].iter())
            .map(|(o, k)| SortOrder {
                expr: k.shown(),
                ..o.clone()
            })
            .collect();
        let shown_call = Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                name: f.name.clone(),
                qualifier: f.qualifier.clone(),
                args: shown,
                arg_names: Vec::new(),
                distinct: f.distinct,
                filter: filter.map(Bound::shown),
                within_group: shown_within,
            }))
            .with_origin(origin),
        );
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: entry.id.clone(),
                    kind: entry.kind,
                    args: resolved,
                    distinct: f.distinct,
                    filter: filter.map(|b| Arc::clone(&b.expr)),
                    within_group: Vec::new(),
                    data_type: data_type.clone(),
                    nullable,
                    deterministic: entry.deterministic,
                },
            )))
            .with_origin(origin),
        );
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            expr,
            ty: arg_type(data_type, nullable),
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(origin)
            } else {
                kids.iter().find_map(|k| k.agg)
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
            shown: Some(shown_call),
        })
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

/// One `EXCEPT` entry, resolved against its star's expansion.
#[derive(Debug, Clone)]
enum ExceptTarget {
    /// A top-level column to drop.
    Top(Attr),
    /// A field below a top-level column: struct support is needed.
    Nested,
    /// Nothing by that name.
    Missing,
}

/// `name` folded for a case-insensitive comparison when `case_sensitive`
/// is off.
fn fold_case(name: &str, case_sensitive: bool) -> String {
    if case_sensitive {
        name.to_owned()
    } else {
        name.to_lowercase()
    }
}

/// Whether folded path `a` is a prefix of (or equal to) folded path `b`.
fn is_prefix(a: &[String], b: &[String]) -> bool {
    a.len() <= b.len() && a.iter().zip(b).all(|(x, y)| x == y)
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

/// `FIELD_NOT_FOUND` for a struct field `parts` names below a struct with
/// `fields` (Spark's `GetStructField` by an unknown name, checked against
/// Spark 4.2.0 for `s.f`, `(s).f`, nested paths and `Column.getField`): the
/// missing name, and every field of the struct.
pub(super) fn unresolved_field(
    parts: &[String],
    fields: &krishiv_types::StructType,
    origin: Origin,
) -> AnalysisError {
    let name = parts.last().map_or_else(String::new, |n| format!("`{n}`"));
    let list = fields
        .fields()
        .iter()
        .map(|f| format!("`{}`", f.name()))
        .collect::<Vec<_>>()
        .join(", ");
    AnalysisError::new(
        FieldNotFound {
            field_name: name,
            fields: list,
        },
        origin,
    )
}

/// `UNRESOLVED_COLUMN`, with up to five of `scope`'s columns ranked by edit
/// distance (qualified by their last qualifier part when `parts` is).
/// Distances are over the quoted candidate and the unquoted target with
/// ties in name order, as Spark's `orderSuggestedIdentifiersBySimilarity`
/// orders them (checked against Spark 4.2.0: `` `id`, `name`, `power`,
/// `city`, `country` `` for `non_exist`, `` `id`, `name`, `data` `` for
/// `invalid_column`).
pub(super) fn unresolved(parts: &[String], scope: &[Attr], origin: Origin) -> AnalysisError {
    let qualified = parts.len() > 1;
    let candidates = scope
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
            p
        })
        .collect();
    let ranked = closest(&parts.join("."), candidates);
    let object_name = quoted(parts);
    if ranked.is_empty() {
        return AnalysisError::new(UnresolvedColumnWithoutSuggestion { object_name }, origin);
    }
    let proposal = ranked
        .iter()
        .map(|p| quoted(p))
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

/// The `UNRECOGNIZED_PARAMETER_NAME` proposal for `name`: the parameter
/// names most similar to it, up to three, as Spark's
/// `orderSuggestedIdentifiersBySimilarity` orders them (Levenshtein over
/// the backticked names, ties in Scala-`Set` order).
fn suggestions(name: &str, params: impl Iterator<Item = &'static str>) -> String {
    let params: Vec<&str> = params.collect();
    let order = scala_set_order(&params);
    let mut scored: Vec<(usize, usize, String)> = params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let quoted = format!("`{}`", p.replace('`', "``"));
            let rank = order.iter().position(|o| *o == i).unwrap_or(i);
            (utf16_distance(&quoted, name), rank, quoted)
        })
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(3)
        .map(|(_, _, q)| q)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The Levenshtein distance of `a` and `b` in UTF-16 units, as Apache
/// Commons Text computes it.
fn utf16_distance(a: &str, b: &str) -> usize {
    let a: Vec<u16> = a.encode_utf16().collect();
    let b: Vec<u16> = b.encode_utf16().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(x != y)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// Java's `String.hashCode` in UTF-16 units.
fn java_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// Scala's hash-code spread for hash tries.
fn improve(hcode: i32) -> i32 {
    let h = hcode.wrapping_add(!hcode.wrapping_shl(9));
    let h = h ^ (h.cast_unsigned() >> 14) as i32;
    let h = h.wrapping_add(h.wrapping_shl(4));
    h ^ (h.cast_unsigned() >> 10) as i32
}

/// The iteration order of a small Scala `Set` of `strings`, as position in
/// the returned order: a hash trie walked in ascending 5-bit chunks of the
/// improved hash.
fn scala_set_order(strings: &[&str]) -> Vec<usize> {
    fn walk(items: Vec<(u32, usize)>, shift: u32, out: &mut Vec<usize>) {
        if items.len() <= 1 || shift >= 32 {
            out.extend(items.into_iter().map(|(_, i)| i));
            return;
        }
        let mut groups: Vec<(u32, Vec<(u32, usize)>)> = Vec::new();
        for (h, i) in items {
            let chunk = (h >> shift) & 31;
            match groups.iter_mut().find(|(c, _)| *c == chunk) {
                Some((_, g)) => g.push((h, i)),
                None => groups.push((chunk, vec![(h, i)])),
            }
        }
        groups.sort_by_key(|(c, _)| *c);
        for (_, g) in groups {
            walk(g, shift + 5, out);
        }
    }
    let items: Vec<(u32, usize)> = strings
        .iter()
        .enumerate()
        .map(|(i, s)| (improve(java_hash(s)).cast_unsigned(), i))
        .collect();
    let mut out = Vec::new();
    walk(items, 0, &mut out);
    out
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
    crate::Builtin::from_name(name)
}

/// Spark's generated name for an unaliased expression (`toPrettySQL`):
/// `(a + 1)`, `count(1)`, `CASE WHEN ... END`. Built bottom-up with an
/// explicit stack, once per distinct node. Timestamp literals render in
/// `zone`, the session's resolved zone, through the same cast-to-string
/// kernels `CAST(ts AS STRING)` uses.
pub(super) fn pretty(e: &Expr, zone: Option<SessionZone>) -> String {
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
    fn collect(node: Node<'_>) -> Vec<(AttrId, String)> {
        let mut names: Vec<(AttrId, String)> = Vec::new();
        let _ = visit_distinct(node, |n| {
            if let Node::Expr(e) = n
                && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
                && !names.iter().any(|(id, _)| *id == c.id)
            {
                names.push((c.id, c.name.clone()));
            }
            std::ops::ControlFlow::Continue(())
        });
        names
    }
    let mut names = collect(Node::Plan(plan));
    // Spark lists each reference of a set operation's branches, not once per
    // column. `INTERSECT` and `EXCEPT` are a semi or an anti join by now.
    let mut union = None;
    let _ = visit_distinct(Node::Plan(plan), |n| {
        let Node::Plan(p) = n else {
            return std::ops::ControlFlow::Continue(());
        };
        match &p.node {
            crate::PlanNode::Union { inputs, .. } => union = Some(inputs.clone()),
            crate::PlanNode::Join {
                left,
                right,
                kind: crate::JoinKind::LeftSemi | crate::JoinKind::LeftAnti,
                ..
            } => union = Some(vec![Arc::clone(left), Arc::clone(right)]),
            _ => return std::ops::ControlFlow::Continue(()),
        }
        std::ops::ControlFlow::Break(())
    });
    // Only when a branch reads the outer query: over an uncorrelated set
    // operation, the references above it are listed as anywhere else.
    if union.is_some_and(|inputs: Vec<_>| {
        inputs
            .iter()
            .any(|input| !collect(Node::Plan(input)).is_empty())
    }) {
        names = Vec::new();
        let _ = visit_distinct(Node::Plan(plan), |n| {
            if let Node::Expr(e) = n
                && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
            {
                names.push((c.id, c.name.clone()));
            }
            std::ops::ControlFlow::Continue(())
        });
    }
    names
        .into_iter()
        .map(|(_, n)| n)
        .collect::<Vec<_>>()
        .join(", ")
}

fn pretty_node(x: &Expr, k: &[&str], zone: Option<SessionZone>) -> String {
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
                TypedLiteralKind::Time => "TIME",
            };
            // Spark names a time by its value, not by the text written.
            if *kind == TypedLiteralKind::Time
                && let Some(n) = krishiv_kernels::cast::parse_time(value)
            {
                return format!("{keyword} '{}'", krishiv_kernels::cast::format_time(n));
            }
            format!("{keyword} '{value}'")
        }
        // Spark's pretty SQL shows a cast to an array, map or struct type as
        // its operand (`CAST(array() AS ARRAY<FLOAT>)` is `array()`).
        ExprNode::Cast { to, mode, .. }
            if *mode != CastMode::Try
                && matches!(
                    to,
                    DataType::Array(_) | DataType::Map(_) | DataType::Struct(_)
                ) =>
        {
            k[0].to_owned()
        }
        ExprNode::Cast { to, mode, .. } => {
            let cast = if *mode == CastMode::Try {
                "TRY_CAST"
            } else {
                "CAST"
            };
            format!("{cast}({} AS {})", k[0], to.sql())
        }
        // An unresolved alias keeps `AS name` (a `struct` call rewritten
        // to `named_struct` shows its children as written); a resolved
        // one is its expression (its output names the attribute).
        ExprNode::Alias { name, .. } => format!("{} AS {name}", k[0]),
        ExprNode::Resolved(ResolvedExpr::Alias { .. }) => k[0].to_owned(),
        ExprNode::Resolved(ResolvedExpr::StateInput { .. }) => k.join(", "),
        ExprNode::Field { name, .. } | ExprNode::Resolved(ResolvedExpr::Field { name, .. }) => {
            format!("{}.{}", k[0], name)
        }
        // As written (`update_fields(base, ...)`): the `shown` form of an
        // update, before the analyzer rewrites it.
        ExprNode::UpdateFields { ops, .. } => {
            let mut values = k[1..].iter();
            let rendered: Vec<String> = ops
                .iter()
                .map(|op| match op {
                    FieldOp::With { name, .. } => format!(
                        "WithField({name}, {})",
                        values.next().copied().unwrap_or("?")
                    ),
                    FieldOp::Drop { name } => format!("DropField({name})"),
                })
                .collect();
            format!("update_fields({}, {})", k[0], rendered.join(", "))
        }
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
        // Spark's `lambdafunction(body, arg1, ...)`: bound variables render
        // as `namedlambdavariable()` (Spark's `NamedLambdaVariable.sql`
        // ignores the written name); an unbound lambda keeps its written
        // names, as error quotes of not-yet-bound calls show them.
        ExprNode::Lambda { params, .. } => {
            let mut args = vec![k[0].to_owned()];
            let bound = params.iter().any(|p| p.id != AttrId(u64::MAX));
            args.extend(params.iter().map(|p| {
                if bound {
                    "namedlambdavariable()".to_owned()
                } else {
                    p.name.clone()
                }
            }));
            format!("lambdafunction({})", args.join(", "))
        }
        ExprNode::Resolved(ResolvedExpr::LambdaVar(_)) => "namedlambdavariable()".to_owned(),
        ExprNode::Window(w) => pretty_window(w, k, zone),
        ExprNode::Function(FunctionCall {
            name, qualifier, ..
        }) if name == "__subscript" && qualifier.is_empty() && k.len() == 2 => {
            format!("{}[{}]", k[0], k[1])
        }
        ExprNode::Function(FunctionCall {
            name,
            qualifier,
            arg_names,
            distinct,
            filter,
            within_group,
            ..
        }) => {
            let qualified = if qualifier.is_empty() {
                name.clone()
            } else {
                format!("{}.{name}", qualifier.join("."))
            };
            // Operators are never qualified, so only a bare name matches.
            let op = if qualifier.is_empty() {
                Builtin::from_name(name)
            } else {
                None
            };
            // The argument children come first; the filter child follows
            // them when the call has one, and any `WITHIN GROUP`
            // ordering keys follow it (a named-argument rearrangement
            // fills defaults, so the written count undercounts `mask`).
            let n = k
                .len()
                .saturating_sub(usize::from(filter.is_some()) + within_group.len());
            let text = call(
                &qualified,
                op,
                *distinct,
                &k[..n.min(k.len())],
                Some(arg_names),
                None,
            );
            // The filter child follows the arguments, when the call has
            // one; any `WITHIN GROUP` ordering keys follow it.
            let mut rest = k.get(n.min(k.len())..).unwrap_or_default().iter();
            let filter = filter.as_ref().and_then(|_| rest.next());
            let mut text = text;
            if !within_group.is_empty() {
                let orders: Vec<String> = within_group
                    .iter()
                    .zip(rest)
                    .map(|(o, key)| within_key(name, o, key))
                    .collect();
                text = format!("{text} WITHIN GROUP (ORDER BY {})", orders.join(", "));
            }
            // Spark appends the filter after the whole function text.
            match filter {
                Some(filter) => format!("{text} FILTER (WHERE {filter})"),
                None => text,
            }
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
            let text = call(
                f.id.name(),
                None,
                f.distinct,
                &k[..n],
                None,
                Some(&f.data_type),
            );
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
        // Spark's `ExtractValue` shows `child[ordinal]`.
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.id.name() == "__subscript" && k.len() == 2 && f.filter.is_none() =>
        {
            format!("{}[{}]", k[0], k[1])
        }
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let op = match &f.id {
                FunctionId::Builtin(b) => Some(*b),
                FunctionId::Registry { name, .. } => builtin_op(name, f.args.len()),
                FunctionId::Udf { .. } => None,
            };
            // As above: the argument children come first, then the
            // filter child, then any ordering keys.
            let n = k
                .len()
                .saturating_sub(usize::from(f.filter.is_some()) + f.within_group.len());
            // `aggregate`/`reduce` of three arguments show the identity
            // `finish` Spark's constructor fills in, once the merge is bound
            // (see `hof`): the call then reads as the four-argument form.
            let identity = matches!(f.id.name(), "aggregate" | "reduce")
                && f.args.len() == 3
                && k.len() >= 3
                && matches!(&f.args[2].node, ExprNode::Lambda { params, .. } if params.iter().any(|p| p.id != AttrId(u64::MAX)));
            let owned;
            let kids: &[&str] = if identity {
                owned = vec![
                    k[0],
                    k[1],
                    k[2],
                    "lambdafunction(namedlambdavariable(), namedlambdavariable())",
                ];
                &owned
            } else {
                &k[..n.min(k.len())]
            };
            let text = call(f.id.name(), op, f.distinct, kids, None, Some(&f.data_type));
            // The filter child follows the arguments, when the call has
            // one; any `WITHIN GROUP` ordering keys follow it.
            let mut rest = k.get(n.min(k.len())..).unwrap_or_default().iter();
            let filter = f.filter.as_ref().and_then(|_| rest.next());
            let mut text = text;
            if !f.within_group.is_empty() {
                // `mode` keeps Spark's `Mode.sql` form: the ordering is
                // the tie-break and takes no arguments.
                if f.id.name() == "mode" {
                    if let (Some(name), Some(o)) = (rest.next(), f.within_group.first()) {
                        let desc = if o.direction == SortDirection::Descending {
                            " DESC"
                        } else {
                            ""
                        };
                        text = format!("mode() WITHIN GROUP (ORDER BY {name}{desc})");
                    }
                } else {
                    let orders: Vec<String> = f
                        .within_group
                        .iter()
                        .zip(rest)
                        .map(|(o, key)| within_key(f.id.name(), o, key))
                        .collect();
                    text = format!("{text} WITHIN GROUP (ORDER BY {})", orders.join(", "));
                }
            }
            // Spark appends the filter after the whole function text.
            match filter {
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
fn pretty_window(w: &crate::WindowExpr, k: &[&str], zone: Option<SessionZone>) -> String {
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

/// One `WITHIN GROUP` ordering key as Spark names it. `percentile_cont` and
/// `percentile_disc` print the key and ` DESC` for a descending order (their
/// own `sql`); the other ordered-set aggregates print the full sort order.
fn within_key(function: &str, o: &crate::SortOrder, key: &str) -> String {
    if function.eq_ignore_ascii_case("percentile_cont")
        || function.eq_ignore_ascii_case("percentile_disc")
    {
        let desc = if o.direction == SortDirection::Descending {
            " DESC"
        } else {
            ""
        };
        format!("{key}{desc}")
    } else {
        format!("{key} {}", order_sql(o))
    }
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
    arg_names: Option<&[Option<String>]>,
    data_type: Option<&DataType>,
) -> String {
    let infix = |sym: &str| format!("({} {sym} {})", k[0], k[1]);
    // `[NOT] LIKE|ILIKE ANY|SOME|ALL` with literal patterns: Spark's
    // `LikeAny`-family name, `likeany(child)` (the child is `lower(child)`
    // for ILIKE; the patterns are not shown), in both its name and SQL form.
    let quantified = |q: QuantifiedLike| {
        let not = if q.negated { "not" } else { "" };
        let any = if q.all { "all" } else { "any" };
        format!("{not}like{any}({})", k[0])
    };
    match (op, k.len()) {
        (Some(op), n) if n >= 1 && op.quantified_like().is_some() => {
            quantified(op.quantified_like().unwrap_or_default())
        }
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
        // A third argument is the precision the analyzer appends for a
        // `time(p)`: not part of the call as written.
        _ if name.eq_ignore_ascii_case("extract") && (2..=3).contains(&k.len()) => {
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
        // `JsonToStructs` has the JSON text as its only child: the schema and
        // the options are not shown.
        // A map schema is the exception: Spark 4.2 names the column `entries`.
        _ if name.eq_ignore_ascii_case("from_json")
            && k.get(1).is_some_and(|schema| {
                let text = schema.trim_matches('\'').trim_start();
                text.get(..3)
                    .is_some_and(|head| head.eq_ignore_ascii_case("map"))
                    && text[3..].trim_start().starts_with('<')
            }) =>
        {
            "entries".to_owned()
        }
        _ if name.eq_ignore_ascii_case("from_json") && !k.is_empty() => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("from_json({d}{})", k[0])
        }
        // `StructsToJson` and `SchemaOfJson` likewise show only their first
        // child, not the options.
        _ if (name.eq_ignore_ascii_case("to_json")
            || name.eq_ignore_ascii_case("schema_of_json"))
            && !k.is_empty() =>
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
        // `current_database` and `current_schema` are one expression
        // (Spark's `CurrentDatabase`, shown as `current_schema`);
        // `current_catalog` shows without parentheses.
        _ if name.eq_ignore_ascii_case("current_database") => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("current_schema({d}{})", k.join(", "))
        }
        _ if name.eq_ignore_ascii_case("current_catalog") && k.is_empty() && !distinct => {
            "current_catalog".to_owned()
        }
        // `RLike` prints its function name upper case, whichever spelling
        // (`rlike`, `regexp`, `regexp_like`) the query used.
        _ if matches!(
            name.to_ascii_lowercase().as_str(),
            "rlike" | "regexp" | "regexp_like"
        ) =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_uppercase(), named(k, arg_names))
        }
        // Single-argument math (`sec`, `csc`, `cot`, `exp`, ...) and
        // two-argument math (`power`, `atan2`, `hypot`) print upper case,
        // as Spark's `UnaryMathExpression` / `BinaryMathExpression` do;
        // aliases (`pow`, `sign`) and the rest print lower case.
        _ if is_upper_math(name) => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_uppercase(), named(k, arg_names))
        }
        _ => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), named(k, arg_names))
        }
    }
}

/// The arguments `k` with their `name => ` prefixes, when `arg_names[i]`
/// names one.
fn named(k: &[&str], arg_names: Option<&[Option<String>]>) -> String {
    k.iter()
        .enumerate()
        .map(
            |(i, v)| match arg_names.and_then(|n| n.get(i)).and_then(Option::as_ref) {
                Some(n) => format!("{n} => {v}"),
                None => (*v).to_owned(),
            },
        )
        .collect::<Vec<_>>()
        .join(", ")
}

/// The default of the optional trailing integer argument of a regular-expression
/// function called with `n` arguments: the group index of `regexp_extract` /
/// `regexp_extract_all` (1) and `regexp_instr` (0), the start position of
/// `regexp_replace` (1).
fn regexp_default_index(name: &str, n: usize) -> Option<i32> {
    match (name, n) {
        ("regexp_extract" | "regexp_extract_all", 2) | ("regexp_replace", 3) => Some(1),
        ("regexp_instr", 2) => Some(0),
        _ => None,
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

fn pretty_literal(v: &Literal, zone: Option<SessionZone>) -> String {
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
        Literal::Timestamp(us) => {
            crate::spark_fmt::timestamp(SessionZone::local_micros(zone, *us), false)
        }
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
                // The ordering keys are children; their directions are not.
                let directions: Vec<_> = f
                    .within_group
                    .iter()
                    .map(|o| (o.direction, o.nulls))
                    .collect();
                format!("f{id}:{:?}:{}:{directions:?}", f.kind, f.distinct)
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

/// The datetime functions whose pattern the legacy time parser reads: the
/// formatters, the parsers with a format argument and `from_unixtime`.
/// `unix_timestamp(str)` and `to_unix_timestamp(str)` parse their string with
/// the default pattern, so a string argument alone counts too (a timestamp or
/// date argument has no text to parse).
fn is_pattern_function(name: &str, args: &[Bound]) -> bool {
    match name.to_ascii_lowercase().as_str() {
        "date_format" | "from_unixtime" => true,
        "unix_timestamp" | "to_unix_timestamp" => {
            args.len() >= 2
                || args
                    .first()
                    .is_some_and(|a| a.ty.data_type == Some(DataType::String))
        }
        "to_date" | "try_to_date" | "to_timestamp" | "try_to_timestamp" | "to_timestamp_ntz"
        | "to_timestamp_ltz" => args.len() >= 2,
        _ => false,
    }
}

/// The function Spark rewrites `+`/`-` to when one operand is a date and
/// the other a byte, short or int (`date_add`, `date_sub`), and whether the
/// operands swap (`int + date`). Other mixes of dates and numbers stay
/// refused by the coercion.
fn date_with_days(op: &str, left: &ArgType, right: &ArgType) -> Option<(&'static str, bool)> {
    let days = |t: &ArgType| {
        matches!(
            t.data_type,
            Some(DataType::Byte | DataType::Short | DataType::Integer)
        )
    };
    let date = |t: &ArgType| matches!(t.data_type, Some(DataType::Date));
    match op {
        "+" if date(left) && days(right) => Some(("date_add", false)),
        "+" if days(left) && date(right) => Some(("date_add", true)),
        "-" if date(left) && days(right) => Some(("date_sub", false)),
        _ => None,
    }
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
