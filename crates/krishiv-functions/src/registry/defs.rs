// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the implemented functions: Spark 4.2's
//! implicit casts (`inputTypes`, expression builders) and result types.
//! Arity, determinism and availability come from Spark's catalog.

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchValueOutOfRange, InvalidExtractField, KrishivError, LegacyErrorTemp1210,
    NonFoldableArgument, SecondFunctionArgumentNotInteger,
};
use krishiv_kernels::datetime::DatePart;
use krishiv_types::{DataType, DecimalType};

use super::{Aggregate, Execution};
use crate::signature::{
    ArgType, CallTypes, Constant, FunctionKind, InputBuilder, InputType, Inputs,
};

/// A function's signature before the catalog adds arity, determinism and
/// availability.
pub(super) struct Spec {
    pub(super) name: &'static str,
    pub(super) kind: FunctionKind,
    pub(super) inputs: Inputs,
    pub(super) result: fn(&CallTypes<'_>) -> Result<ArgType, KrishivError>,
    pub(super) execution: Execution,
}

type Rule = fn(&CallTypes<'_>) -> Result<ArgType, KrishivError>;

pub(super) fn exact(t: DataType) -> InputType {
    InputType::Exact(t)
}

pub(super) fn scalar(names: &[&'static str], inputs: &Inputs, result: Rule) -> Vec<Spec> {
    names
        .iter()
        .map(|name| Spec {
            name,
            kind: FunctionKind::Scalar,
            inputs: inputs.clone(),
            result,
            execution: Execution::Scalar,
        })
        .collect()
}

fn aggregate(name: &'static str, inputs: Inputs, result: Rule, agg: Aggregate) -> Spec {
    Spec {
        name,
        kind: FunctionKind::Aggregate,
        inputs,
        result,
        execution: Execution::Aggregate(agg),
    }
}

// ---- result rules ----

pub(super) fn typed(t: DataType, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(t, c.any_nullable()))
}

pub(super) fn string(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::String, c)
}

pub(super) fn int(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Integer, c)
}

/// `startswith`/`endswith`/`contains`: the binary predicates may be null
/// whatever their arguments (Spark's `BinaryPredicate` is a nullable
/// static invocation).
fn matches(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let binary = is_binary(c, 0) && is_binary(c, 1);
    Ok(ArgType::new(DataType::Boolean, binary || c.any_nullable()))
}

fn date(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Date, c)
}

/// Whether Spark's legacy string-to-`int` cast reads `s` (surrounding
/// whitespace, a sign, digits and a fraction it truncates).
fn legacy_int(s: &str) -> bool {
    let t = s.trim();
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    !whole.is_empty()
        && whole.bytes().all(|b| b.is_ascii_digit())
        && fraction.bytes().all(|b| b.is_ascii_digit())
        && t.split('.')
            .next()
            .and_then(|w| w.parse::<i32>().ok())
            .is_some()
}

/// `date_add`/`date_sub` with ANSI off: a string literal day count that is
/// not an integer is an analysis error (`SECOND_FUNCTION_ARGUMENT_NOT_INTEGER`).
fn day_arith(function: &str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if !c.ansi()
        && let Some(Constant::String(s)) = c.constant(1)
        && !legacy_int(s)
    {
        return Err(SecondFunctionArgumentNotInteger {
            function_name: function.to_owned(),
        }
        .into());
    }
    date(c)
}

fn date_add(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    day_arith("date_add", c)
}

fn date_sub(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    day_arith("date_sub", c)
}

/// The type of the first (cast) argument: `binary` or `string` overloads.
fn first_type(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = c.data_type(0).cloned().unwrap_or(DataType::String);
    typed(t, c)
}

fn clock_date(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Date, false))
}

fn clock_timestamp(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Timestamp, false))
}

fn clock_timestamp_ntz(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::TimestampNtz, false))
}

fn user(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, false))
}

// ---- strings ----

fn is_binary(c: &CallTypes<'_>, i: usize) -> bool {
    c.data_type(i) == Some(&DataType::Binary)
}

/// `startswith`/`endswith`/`contains`: the binary predicate when both
/// arguments are binary, else the string one.
fn match_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let t = if is_binary(c, 0) && is_binary(c, 1) {
        DataType::Binary
    } else {
        DataType::String
    };
    vec![exact(t)]
}

/// `lpad`/`rpad`: the binary pad when the string (and the pad, if any) is
/// binary, else the string pad.
fn pad_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let bytes = is_binary(c, 0) && (c.args().len() < 3 || is_binary(c, 2));
    let t = if bytes {
        DataType::Binary
    } else {
        DataType::String
    };
    vec![exact(t.clone()), exact(DataType::Integer), exact(t)]
}

fn strings() -> Vec<Spec> {
    let text = Inputs::Expects(vec![exact(DataType::String)]);
    let text_or_bytes = InputType::OneOf(vec![DataType::String, DataType::Binary]);
    let mut out = scalar(&["upper", "ucase", "lower", "lcase"], &text, string);
    out.extend(scalar(
        &["length", "char_length", "character_length"],
        &Inputs::Expects(vec![text_or_bytes.clone()]),
        int,
    ));
    out.extend(scalar(
        &["trim", "ltrim", "rtrim", "btrim", "replace"],
        &text,
        string,
    ));
    out.extend(scalar(
        &["startswith", "endswith", "contains"],
        &Inputs::Built(InputBuilder(match_inputs)),
        matches,
    ));
    out.extend(scalar(
        &["lpad", "rpad"],
        &Inputs::Built(InputBuilder(pad_inputs)),
        first_type,
    ));
    out.extend(scalar(
        &["substring", "substr"],
        &Inputs::Expects(vec![
            text_or_bytes,
            exact(DataType::Integer),
            exact(DataType::Integer),
        ]),
        first_type,
    ));
    out
}

// ---- datetime ----

/// The field of `extract`/`date_part`: `Ok(None)` for a NULL field.
fn extract_field(c: &CallTypes<'_>) -> Result<Option<DatePart>, KrishivError> {
    let name = match c.constant(0) {
        Some(Constant::String(s)) => s,
        Some(Constant::Null) => return Ok(None),
        _ => {
            return Err(NonFoldableArgument {
                func_name: "`date_part`".to_owned(),
                param_name: "`field`".to_owned(),
                param_type: "\"STRING\"".to_owned(),
            }
            .into());
        }
    };
    DatePart::parse(name).map(Some).ok_or_else(|| {
        InvalidExtractField {
            field: format!("`{name}`"),
            expr: format!("\"{}\"", c.sql(1).unwrap_or_default()),
        }
        .into()
    })
}

fn is_time_field(part: DatePart) -> bool {
    matches!(part, DatePart::Hour | DatePart::Minute | DatePart::Second)
}

/// `extract(field FROM source)`: the date fields read a `DATE` (a
/// timestamp is cast), the time fields a timestamp (a date is cast).
fn extract_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let source = match extract_field(c) {
        Ok(Some(part)) if is_time_field(part) => {
            InputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz])
        }
        Ok(Some(_)) => exact(DataType::Date),
        // The result rule reports the field.
        Ok(None) | Err(_) => InputType::Any,
    };
    vec![exact(DataType::String), source]
}

fn extract_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = c.args().get(1).is_some_and(|a| a.nullable);
    Ok(match extract_field(c)? {
        // `date_part(NULL, x)` is a NULL `double`.
        None => ArgType::new(DataType::Double, true),
        Some(DatePart::Second) => {
            ArgType::new(DataType::Decimal(DecimalType::try_new(8, 6)?), nullable)
        }
        Some(_) => ArgType::new(DataType::Integer, nullable),
    })
}

fn datetime() -> Vec<Spec> {
    let days = InputType::OneOf(vec![DataType::Integer, DataType::Short, DataType::Byte]);
    let day_inputs = Inputs::Expects(vec![exact(DataType::Date), days]);
    let mut out = scalar(&["date_add", "dateadd"], &day_inputs, date_add);
    out.extend(scalar(&["date_sub"], &day_inputs, date_sub));
    out.extend(scalar(
        &["datediff", "date_diff"],
        &Inputs::Expects(vec![exact(DataType::Date)]),
        int,
    ));
    out.extend(scalar(
        &["year", "month", "day", "dayofmonth"],
        &Inputs::Expects(vec![exact(DataType::Date)]),
        int,
    ));
    out.extend(scalar(
        &["extract", "date_part"],
        &Inputs::Built(InputBuilder(extract_inputs)),
        extract_result,
    ));
    out.extend(scalar(
        &["current_date", "curdate"],
        &Inputs::Any,
        clock_date,
    ));
    out.extend(scalar(
        &["current_timestamp", "now"],
        &Inputs::Any,
        clock_timestamp,
    ));
    out.extend(scalar(
        &["current_user", "user", "session_user"],
        &Inputs::Any,
        user,
    ));
    out.extend(scalar(
        &["localtimestamp"],
        &Inputs::Any,
        clock_timestamp_ntz,
    ));
    out
}

// ---- aggregates ----

fn bounded_decimal(precision: u32, scale: u32) -> Result<DataType, KrishivError> {
    Ok(DataType::Decimal(DecimalType::try_new(
        precision.min(38),
        scale.min(38),
    )?))
}

/// `sum`: `decimal(min(38, p + 10), s)`, `bigint` for integral input,
/// `double` for floating point.
fn sum(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = match c.data_type(0) {
        Some(DataType::Decimal(d)) => bounded_decimal(
            u32::from(d.precision()).saturating_add(10),
            d.scale().into(),
        )?,
        Some(DataType::Double | DataType::Float) | None => DataType::Double,
        Some(_) => DataType::Long,
    };
    Ok(ArgType::new(t, true))
}

/// `avg`: `decimal(min(38, p + 4), min(38, s + 4))`, else `double`.
fn avg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = match c.data_type(0) {
        Some(DataType::Decimal(d)) => bounded_decimal(
            u32::from(d.precision()).saturating_add(4),
            u32::from(d.scale()).saturating_add(4),
        )?,
        _ => DataType::Double,
    };
    Ok(ArgType::new(t, true))
}

fn count(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, false))
}

fn min_max(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: c.data_type(0).cloned(),
        nullable: true,
    })
}

/// `first(x [, ignoreNulls])`: the type of `x`; the flag must be a boolean
/// literal.
fn pick(c: &CallTypes<'_>, function: &str) -> Result<ArgType, KrishivError> {
    if c.args().len() > 1 && !matches!(c.constant(1), Some(Constant::Boolean(_))) {
        return Err(LegacyErrorTemp1210 {
            func_name: function.to_owned(),
        }
        .into());
    }
    min_max(c)
}

fn first(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pick(c, "first")
}

fn last(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pick(c, "last")
}

fn aggregates() -> Vec<Spec> {
    let numeric = || Inputs::Expects(vec![InputType::Numeric { intervals: true }]);
    vec![
        aggregate("count", Inputs::Any, count, Aggregate::Count),
        aggregate("sum", numeric(), sum, Aggregate::Sum),
        aggregate("avg", numeric(), avg, Aggregate::Avg),
        aggregate("mean", numeric(), avg, Aggregate::Avg),
        aggregate("min", Inputs::Any, min_max, Aggregate::Min),
        aggregate("max", Inputs::Any, min_max, Aggregate::Max),
        aggregate("first", Inputs::Any, first, Aggregate::First),
        aggregate("first_value", Inputs::Any, first, Aggregate::First),
        aggregate("last", Inputs::Any, last, Aggregate::Last),
        aggregate("last_value", Inputs::Any, last, Aggregate::Last),
    ]
}

// ---- window functions ----

fn window(name: &'static str, result: Rule) -> Spec {
    Spec {
        name,
        kind: FunctionKind::Window,
        inputs: Inputs::Any,
        result,
        execution: Execution::Window,
    }
}

/// `row_number`, `rank`, `dense_rank`, `ntile`: `int`, never NULL.
fn rank_like(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Integer, false))
}

/// `percent_rank`, `cume_dist`: `double`, never NULL.
fn distribution(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Double, false))
}

/// `ntile(buckets)`: Spark's `NTile.checkInputDataTypes`, in its order: a
/// foldable argument, of type `int` exactly (no implicit cast), above 0.
/// `ntile()` is `ntile(1)`.
fn ntile(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().is_empty() {
        return rank_like(c);
    }
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    let call = format!("\"ntile({})\"", sql(0));
    let Some(value) = c.constant(0) else {
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call,
            input_name: "`buckets`".to_owned(),
            input_type: "\"INT\"".to_owned(),
            input_expr: format!("\"{}\"", sql(0)),
        }
        .into());
    };
    if c.data_type(0) != Some(&DataType::Integer) {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call,
            param_index: "first".to_owned(),
            required_type: "\"INT\"".to_owned(),
            input_sql: format!("\"{}\"", sql(0)),
            input_type: c
                .data_type(0)
                .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql())),
        }
        .into());
    }
    match value {
        Constant::Integral(n) if *n > 0 => rank_like(c),
        Constant::Integral(n) => Err(DatatypeMismatchValueOutOfRange {
            sql_expr: call,
            expr_name: "buckets".to_owned(),
            value_range: format!("(0, {}]", i32::MAX),
            current_value: n.to_string(),
        }
        .into()),
        Constant::Error(e) => Err(e.clone()),
        // A NULL `int` (`ntile(CAST(NULL AS INT))`): Spark's check reads
        // it as 0.
        _ => Err(DatatypeMismatchValueOutOfRange {
            sql_expr: call,
            expr_name: "buckets".to_owned(),
            value_range: format!("(0, {}]", i32::MAX),
            current_value: "0".to_owned(),
        }
        .into()),
    }
}

fn windows() -> Vec<Spec> {
    vec![
        window("row_number", rank_like),
        window("rank", rank_like),
        window("dense_rank", rank_like),
        window("percent_rank", distribution),
        window("cume_dist", distribution),
        window("ntile", ntile),
    ]
}

/// Every implemented function.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = strings();
    out.extend(datetime());
    out.extend(aggregates());
    out.extend(windows());
    out.extend(super::conditional::specs());
    out.extend(super::math::specs());
    out.extend(super::hashing::specs());
    out
}
