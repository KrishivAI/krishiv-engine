// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the implemented functions: Spark 4.2's
//! implicit casts (`inputTypes`, expression builders) and result types.
//! Arity, determinism and availability come from Spark's catalog.

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchValueOutOfRange, InvalidExtractField, InvalidParameterValueBoolean,
    KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp1210, NonFoldableArgument,
    SecondFunctionArgumentNotInteger,
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

/// `regexp_replace`: Spark checks `position` while typing the call, so it
/// must be a foldable `INT` in `(0, 2147483647]` (NULL is fine) whatever the
/// rows hold.
fn regexp_replace_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().len() == 4 {
        let call = || {
            let args: Vec<String> = (0..4).map(|n| c.sql(n).unwrap_or_default()).collect();
            format!("\"regexp_replace({})\"", args.join(", "))
        };
        match c.constant(3) {
            Some(Constant::Integral(n)) if *n < 1 => {
                return Err(DatatypeMismatchValueOutOfRange {
                    sql_expr: call(),
                    expr_name: "position".to_owned(),
                    value_range: "(0, 2147483647]".to_owned(),
                    current_value: n.to_string(),
                }
                .into());
            }
            Some(Constant::Error(e)) => return Err(e.clone()),
            Some(Constant::Integral(_) | Constant::Null) => {}
            _ => {
                return Err(DatatypeMismatchNonFoldableInput {
                    sql_expr: call(),
                    input_name: "`position`".to_owned(),
                    input_type: "\"INT\"".to_owned(),
                    input_expr: format!("\"{}\"", c.sql(3).unwrap_or_default()),
                }
                .into());
            }
        }
    }
    string(c)
}

pub(super) fn int(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Integer, c)
}

pub(super) fn boolean(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Boolean, c)
}

/// `regexp_substr`: Spark marks it nullable whatever its arguments.
fn always_null(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, true))
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

/// The day count `date_add`/`date_sub` without ANSI read `s` as: an `int`
/// with an optional sign, surrounded by whitespace or control characters
/// (`UTF8String.toIntExact`). A fraction is not read (`'1.5'`, `'1.'`),
/// nor an exponent, a hex number or one past `int` (checked against a
/// Spark 4.2 run of each). The typing rule here and the analyzer's shown
/// form of the call both read the day count through this.
#[must_use]
pub fn legacy_day_count(s: &str) -> Option<i32> {
    s.trim_matches(|c: char| c.is_whitespace() || c.is_control())
        .parse::<i32>()
        .ok()
}

/// `date_add`/`date_sub` with ANSI off: a string literal day count that is
/// not an integer is an analysis error (`SECOND_FUNCTION_ARGUMENT_NOT_INTEGER`).
fn day_arith(function: &str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if !c.ansi()
        && let Some(Constant::String(s)) = c.constant(1)
        && legacy_day_count(s).is_none()
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
pub(super) fn first_type(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = c.data_type(0).cloned().unwrap_or(DataType::String);
    typed(t, c)
}

fn always_null_date(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Date, true))
}

fn always_null_timestamp(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Timestamp, true))
}

fn double(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Double, c)
}

/// `make_date`: an invalid date is NULL with ANSI off, so the result may be
/// null then whatever the arguments.
fn make_date_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Date, !c.ansi() || c.any_nullable()))
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

pub(super) fn is_binary(c: &CallTypes<'_>, i: usize) -> bool {
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
        &["length", "len", "char_length", "character_length"],
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
        &["regexp_replace"],
        &Inputs::Expects(vec![
            exact(DataType::String),
            exact(DataType::String),
            exact(DataType::String),
            exact(DataType::Integer),
        ]),
        regexp_replace_result,
    ));
    out.extend(scalar(
        &["regexp", "regexp_like"],
        &Inputs::Expects(vec![exact(DataType::String), exact(DataType::String)]),
        boolean,
    ));
    out.extend(scalar(
        &["regexp_extract"],
        &Inputs::Expects(vec![
            exact(DataType::String),
            exact(DataType::String),
            exact(DataType::Integer),
        ]),
        string,
    ));
    out.extend(scalar(
        &["regexp_substr"],
        &Inputs::Expects(vec![exact(DataType::String), exact(DataType::String)]),
        always_null,
    ));
    out.extend(scalar(
        &["regexp_count"],
        &Inputs::Expects(vec![exact(DataType::String), exact(DataType::String)]),
        int,
    ));
    out.extend(scalar(
        &["regexp_instr"],
        &Inputs::Expects(vec![
            exact(DataType::String),
            exact(DataType::String),
            exact(DataType::Integer),
        ]),
        int,
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
    out.extend(scalar(
        &["mask"],
        &Inputs::Expects(vec![
            InputType::Strict(DataType::String),
            InputType::Strict(DataType::String),
            InputType::Strict(DataType::String),
            InputType::Strict(DataType::String),
            InputType::Strict(DataType::String),
        ]),
        string,
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

/// `TimestampType` as `date_trunc` and `months_between` expect it: other
/// types are cast to `TIMESTAMP` or refused naming it, but a
/// `TIMESTAMP_NTZ` is kept (Spark casts it, as the binding does, in the
/// session zone).
fn instant_input(c: &CallTypes<'_>, i: usize) -> InputType {
    if c.data_type(i) == Some(&DataType::TimestampNtz) {
        InputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz])
    } else {
        exact(DataType::Timestamp)
    }
}

fn date_trunc_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![exact(DataType::String), instant_input(c, 1)]
}

fn months_between_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![
        instant_input(c, 0),
        instant_input(c, 1),
        exact(DataType::Boolean),
    ]
}

/// `TimestampNTZType` as `convert_timezone` expects it: a `TIMESTAMP` is
/// kept (Spark casts it, as the binding does, to the wall clock of the
/// session zone).
fn wall_clock_input(c: &CallTypes<'_>, i: usize) -> InputType {
    if c.data_type(i) == Some(&DataType::Timestamp) {
        InputType::OneOf(vec![DataType::TimestampNtz, DataType::Timestamp])
    } else {
        exact(DataType::TimestampNtz)
    }
}

fn zone_shift_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![instant_input(c, 0), exact(DataType::String)]
}

fn convert_timezone_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    match c.args().len() {
        2 => vec![exact(DataType::String), wall_clock_input(c, 1)],
        _ => vec![
            exact(DataType::String),
            exact(DataType::String),
            wall_clock_input(c, 2),
        ],
    }
}

fn timestamp_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Timestamp, c)
}

fn timestamp_ntz_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::TimestampNtz, c)
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
    let date_only = Inputs::Expects(vec![exact(DataType::Date)]);
    out.extend(scalar(
        &[
            "year",
            "month",
            "day",
            "dayofmonth",
            "dayofweek",
            "dayofyear",
            "weekofyear",
            "weekday",
            "quarter",
        ],
        &date_only,
        int,
    ));
    // `UnixDate` expects a date and casts nothing else to one.
    out.extend(scalar(
        &["unix_date"],
        &Inputs::Expects(vec![InputType::Strict(DataType::Date)]),
        int,
    ));
    // The time fields read a timestamp of either kind; a date is cast to
    // one.
    let any_timestamp = InputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz]);
    out.extend(scalar(
        &["hour", "minute", "second"],
        &Inputs::Expects(vec![any_timestamp]),
        int,
    ));
    out.extend(scalar(&["monthname", "dayname"], &date_only, string));
    out.extend(scalar(
        &["add_months"],
        &Inputs::Expects(vec![exact(DataType::Date), exact(DataType::Integer)]),
        date,
    ));
    out.extend(scalar(&["last_day"], &date_only, date));
    // `NextDay`, `TruncDate` and `TruncTimestamp` are nullable whatever
    // their arguments: an unknown weekday or unit gives NULL.
    out.extend(scalar(
        &["next_day", "trunc"],
        &Inputs::Expects(vec![exact(DataType::Date), exact(DataType::String)]),
        always_null_date,
    ));
    out.extend(scalar(
        &["date_trunc"],
        &Inputs::Built(InputBuilder(date_trunc_inputs)),
        always_null_timestamp,
    ));
    out.extend(scalar(
        &["months_between"],
        &Inputs::Built(InputBuilder(months_between_inputs)),
        double,
    ));
    out.extend(scalar(
        &["from_utc_timestamp", "to_utc_timestamp"],
        &Inputs::Built(InputBuilder(zone_shift_inputs)),
        timestamp_result,
    ));
    out.extend(scalar(
        &["convert_timezone"],
        &Inputs::Built(InputBuilder(convert_timezone_inputs)),
        timestamp_ntz_result,
    ));
    out.extend(scalar(
        &["make_date"],
        &Inputs::Expects(vec![exact(DataType::Integer)]),
        make_date_result,
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
/// literal (Spark matches `Literal(b, BooleanType)`, so `1 = 1` is rejected).
fn pick(c: &CallTypes<'_>, function: &str) -> Result<ArgType, KrishivError> {
    if c.args().len() > 1
        && !(c.is_literal(1) && matches!(c.constant(1), Some(Constant::Boolean(_))))
    {
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

fn any_value(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pick(c, "any_value")
}

/// The moment aggregates (`CentralMomentAgg`, `Covariance`, `Corr`): every
/// argument is cast to `DOUBLE`; the result is a nullable `DOUBLE`.
fn moment(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Double, true))
}

/// `bool_and`, `bool_or` and their aliases: `boolean`, NULL for no non-NULL
/// input.
fn nullable_boolean(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Boolean, true))
}

/// `max_by(x, y)`, `min_by(x, y)`: the type of `x`. The three-argument
/// form (`max_by(x, y, k)`) returns an array of up to `k` values, which
/// this engine has no type for yet.
fn by(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().len() > 2 {
        return Err(KrishivUnsupportedFeature {
            feature: "max_by and min_by with a third argument (they return an array)".to_owned(),
        }
        .into());
    }
    min_max(c)
}

fn aggregates() -> Vec<Spec> {
    let numeric = || Inputs::Expects(vec![InputType::Numeric { intervals: true }]);
    let boolean_arg = || Inputs::Expects(vec![exact(DataType::Boolean)]);
    let doubles = || Inputs::Expects(vec![exact(DataType::Double)]);
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
        aggregate("any_value", Inputs::Any, any_value, Aggregate::AnyValue),
        aggregate("stddev", doubles(), moment, Aggregate::StddevSamp),
        aggregate("std", doubles(), moment, Aggregate::StddevSamp),
        aggregate("stddev_samp", doubles(), moment, Aggregate::StddevSamp),
        aggregate("stddev_pop", doubles(), moment, Aggregate::StddevPop),
        aggregate("variance", doubles(), moment, Aggregate::VarSamp),
        aggregate("var_samp", doubles(), moment, Aggregate::VarSamp),
        aggregate("var_pop", doubles(), moment, Aggregate::VarPop),
        aggregate("skewness", doubles(), moment, Aggregate::Skewness),
        aggregate("kurtosis", doubles(), moment, Aggregate::Kurtosis),
        aggregate("corr", doubles(), moment, Aggregate::Corr),
        aggregate("covar_samp", doubles(), moment, Aggregate::CovarSamp),
        aggregate("covar_pop", doubles(), moment, Aggregate::CovarPop),
        // Spark runs these as `count`, `min` and `max` of a rewritten argument.
        aggregate("count_if", boolean_arg(), count, Aggregate::Count),
        aggregate("bool_and", boolean_arg(), nullable_boolean, Aggregate::Min),
        aggregate("every", boolean_arg(), nullable_boolean, Aggregate::Min),
        aggregate("bool_or", boolean_arg(), nullable_boolean, Aggregate::Max),
        aggregate("some", boolean_arg(), nullable_boolean, Aggregate::Max),
        aggregate("any", boolean_arg(), nullable_boolean, Aggregate::Max),
        aggregate("max_by", Inputs::Any, by, Aggregate::MaxBy),
        aggregate("min_by", Inputs::Any, by, Aggregate::MinBy),
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

/// The `ignoreNulls` flag (argument `i`) of `name`, when given, must be a
/// boolean literal (`INVALID_PARAMETER_VALUE.BOOLEAN`, Spark's builder).
fn null_flag(c: &CallTypes<'_>, i: usize, name: &str) -> Result<(), KrishivError> {
    if c.args().len() > i && !matches!(c.constant(i), Some(Constant::Boolean(_))) {
        return Err(InvalidParameterValueBoolean {
            function_name: format!("`{name}`"),
            parameter: "`ignoreNulls`".to_owned(),
            invalid_value: format!("\"{}\"", c.sql(i).unwrap_or_default()),
        }
        .into());
    }
    Ok(())
}

/// `lag`/`lead(input, offset, default[, ignoreNulls])` (the parser fills
/// `offset` 1 and `default` NULL): the input's type, nullable. The offset
/// and the default are checked with the window (the frame they make).
fn offset_value(c: &CallTypes<'_>, name: &str) -> Result<ArgType, KrishivError> {
    null_flag(c, 3, name)?;
    Ok(ArgType::new(
        c.data_type(0).cloned().unwrap_or(DataType::Null),
        true,
    ))
}

fn lag(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    offset_value(c, "lag")
}

fn lead(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    offset_value(c, "lead")
}

/// `nth_value(input, offset[, ignoreNulls])`: the input's type, nullable.
/// Spark's `NthValue` checks, in its order: `offset` foldable, then its
/// value (implicitly an `INT`) above 0 (a NULL reads as 0).
fn nth_value(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    null_flag(c, 2, "nth_value")?;
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    let call = format!("\"nth_value({}, {})\"", sql(0), sql(1));
    let Some(offset) = c.constant(1) else {
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call,
            input_name: "`offset`".to_owned(),
            input_type: "\"INT\"".to_owned(),
            input_expr: format!("\"{}\"", sql(1)),
        }
        .into());
    };
    let value = match offset {
        Constant::Integral(v) | Constant::Fractional(v) => *v,
        Constant::String(t) => t.trim().parse::<i64>().unwrap_or(0),
        Constant::Error(e) => return Err(e.clone()),
        _ => 0,
    };
    if value <= 0 {
        return Err(DatatypeMismatchValueOutOfRange {
            sql_expr: call,
            expr_name: "offset".to_owned(),
            value_range: format!("(0, {}]", i64::MAX),
            current_value: format!("{value}L"),
        }
        .into());
    }
    Ok(ArgType::new(
        c.data_type(0).cloned().unwrap_or(DataType::Null),
        true,
    ))
}

fn windows() -> Vec<Spec> {
    vec![
        window("lag", lag),
        window("lead", lead),
        window("nth_value", nth_value),
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
    out.extend(super::json::specs());
    out.extend(super::math::specs());
    out.extend(super::hashing::specs());
    out.extend(super::bitwise::specs());
    out.extend(super::predicate::specs());
    out.extend(super::number::specs());
    out.extend(super::misc::specs());
    out.extend(super::url::specs());
    out.extend(super::csv::specs());
    out.extend(super::text::specs());
    out.extend(super::hof::specs());
    out.extend(super::r#struct::specs());
    out
}
