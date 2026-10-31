// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the implemented functions: Spark 4.2's
//! implicit casts (`inputTypes`, expression builders) and result types.
//! Arity, determinism and availability come from Spark's catalog.

use krishiv_common::error::{
    DatatypeMismatchDataDiffTypes, DatatypeMismatchInvalidOrderingType,
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchUnsupportedInputType, DatatypeMismatchValueOutOfRange, InvalidExtractField,
    InvalidParameterValueBoolean, KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp1210,
    NonFoldableArgument, SecondFunctionArgumentNotInteger,
};
use krishiv_kernels::datetime::DatePart;
use krishiv_types::{ArrayType, DataType, DecimalType};

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

fn byte(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Byte, c)
}

fn short(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Short, c)
}

fn long(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Long, c)
}

fn float(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Float, c)
}

fn decimal_default(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Decimal(DecimalType::USER_DEFAULT), c)
}

fn binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Binary, c)
}

fn timestamp(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Timestamp, c)
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

/// Whether `t` is a numeric type (Spark's `NumericType`).
fn is_numeric_type(t: &DataType) -> bool {
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

/// A date, timestamp or string (what `to_date` and `to_timestamp_ntz`
/// read): anything else takes the first member it may be cast to, as
/// Spark's `TypeCollection` does (a number becomes a string).
fn date_like() -> InputType {
    InputType::OneOf(vec![
        DataType::String,
        DataType::Date,
        DataType::Timestamp,
        DataType::TimestampNtz,
    ])
}

/// A date, timestamp or string, and nothing else (Spark's
/// `ExpectsInputTypes` over the collection: `date_format`,
/// `unix_timestamp` and `from_unixtime`'s pattern take no numbers).
fn strict_date_like() -> InputType {
    InputType::StrictOneOf(vec![
        DataType::String,
        DataType::Date,
        DataType::Timestamp,
        DataType::TimestampNtz,
    ])
}

/// `to_date`/`try_to_date` (and `to_timestamp_ntz`): one or two arguments;
/// the pattern is a string.
fn to_date_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let mut out = vec![date_like()];
    if c.args().len() > 1 {
        out.push(exact(DataType::String));
    }
    out
}

/// `to_timestamp`/`try_to_timestamp`/`to_timestamp_ltz`: a numeric stays
/// numeric (Spark's `NumericType` member: integrals are seconds since the
/// epoch); anything else takes the string/date/timestamp collection.
fn to_timestamp_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let first = match c.data_type(0) {
        Some(t) if is_numeric_type(t) => InputType::Numeric { intervals: false },
        _ => date_like(),
    };
    let mut out = vec![first];
    if c.args().len() > 1 {
        out.push(exact(DataType::String));
    }
    out
}

/// `unix_timestamp`/`to_unix_timestamp`: no arguments (now), or a
/// date-like value with an optional pattern. Neither takes a number
/// (Spark analyses `to_unix_timestamp(1, fmt)` as
/// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`); the pattern is exactly a
/// string.
fn unix_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let mut out = Vec::new();
    if !c.args().is_empty() {
        out.push(strict_date_like());
    }
    if c.args().len() > 1 {
        out.push(InputType::Strict(DataType::String));
    }
    out
}

/// `from_unixtime(seconds[, fmt])`: seconds widen to `BIGINT`.
fn from_unixtime_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let mut out = vec![exact(DataType::Long)];
    if c.args().len() > 1 {
        out.push(exact(DataType::String));
    }
    out
}

/// Whether Spark's one-argument `to_*` replacement (`CAST`) may yield
/// NULL from a non-NULL value of `from` cast to `to`: strings (booleans
/// and binaries reach it as strings) and floating-point values may, as
/// `Cast.forceNullable` says (mirrored from the analyzer's coercion; the
/// function matrix pins it against Spark 4.2 runs).
fn temporal_cast_nullable(from: &DataType, to: &DataType) -> bool {
    if from == to {
        return false;
    }
    match from {
        DataType::String
        | DataType::Boolean
        | DataType::Binary
        | DataType::Float
        | DataType::Double => true,
        DataType::Date => !matches!(to, DataType::Timestamp),
        DataType::TimestampNtz => matches!(to, DataType::Date),
        _ => false,
    }
}

/// The nullability of a one-argument `to_*` call over `from`: the
/// argument's, or Spark's `CAST` replacement wherever it may yield NULL.
fn one_arg_temporal_nullable(c: &CallTypes<'_>, to: &DataType) -> bool {
    c.any_nullable()
        || match c.data_type(0) {
            // Booleans and binaries reach the cast as strings.
            Some(DataType::Boolean | DataType::Binary) => {
                temporal_cast_nullable(&DataType::String, to)
            }
            Some(from) => temporal_cast_nullable(from, to),
            // An untyped `NULL` is NULL.
            None => true,
        }
}

/// `to_date`: one argument is Spark's `CAST` (nullable where the cast may
/// yield NULL); two are `failOnError`-nullable.
fn to_date_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = if c.args().len() > 1 {
        !c.ansi() || c.any_nullable()
    } else {
        one_arg_temporal_nullable(c, &DataType::Date)
    };
    Ok(ArgType::new(DataType::Date, nullable))
}

/// `try_to_date`: one argument is Spark's `TRY_CAST`; two are always
/// nullable.
fn try_to_date_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = if c.args().len() > 1 {
        true
    } else {
        one_arg_temporal_nullable(c, &DataType::Date)
    };
    Ok(ArgType::new(DataType::Date, nullable))
}

/// `to_timestamp`/`to_timestamp_ltz`, as [`to_date_result`].
fn to_timestamp_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = if c.args().len() > 1 {
        !c.ansi() || c.any_nullable()
    } else {
        one_arg_temporal_nullable(c, &DataType::Timestamp)
    };
    Ok(ArgType::new(DataType::Timestamp, nullable))
}

/// `try_to_timestamp`, as [`try_to_date_result`].
fn try_to_timestamp_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = if c.args().len() > 1 {
        true
    } else {
        one_arg_temporal_nullable(c, &DataType::Timestamp)
    };
    Ok(ArgType::new(DataType::Timestamp, nullable))
}

/// `to_timestamp_ntz`, as [`to_date_result`].
fn to_timestamp_ntz_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = if c.args().len() > 1 {
        !c.ansi() || c.any_nullable()
    } else {
        one_arg_temporal_nullable(c, &DataType::TimestampNtz)
    };
    Ok(ArgType::new(DataType::TimestampNtz, nullable))
}

/// `unix_timestamp`/`to_unix_timestamp`: NULL with ANSI off whatever the
/// arguments, else NULL when an argument may be (Spark's `ToTimestamp`
/// nullability, which `UnixTime` inherits).
fn unix_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, !c.ansi() || c.any_nullable()))
}

/// `date_format`: NULL when an argument may be, or when the value's
/// implicit cast to `TIMESTAMP` may yield NULL.
fn date_format_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = c.any_nullable()
        || match c.data_type(0) {
            Some(DataType::Boolean | DataType::Binary) => {
                temporal_cast_nullable(&DataType::String, &DataType::Timestamp)
            }
            Some(from) => temporal_cast_nullable(from, &DataType::Timestamp),
            None => true,
        };
    Ok(ArgType::new(DataType::String, nullable))
}

fn current_timezone_result(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, false))
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
        &["date_format"],
        &Inputs::Expects(vec![strict_date_like(), exact(DataType::String)]),
        date_format_result,
    ));
    out.extend(scalar(
        &["to_date"],
        &Inputs::Built(InputBuilder(to_date_inputs)),
        to_date_result,
    ));
    out.extend(scalar(
        &["try_to_date"],
        &Inputs::Built(InputBuilder(to_date_inputs)),
        try_to_date_result,
    ));
    out.extend(scalar(
        &["to_timestamp", "to_timestamp_ltz"],
        &Inputs::Built(InputBuilder(to_timestamp_inputs)),
        to_timestamp_result,
    ));
    out.extend(scalar(
        &["try_to_timestamp"],
        &Inputs::Built(InputBuilder(to_timestamp_inputs)),
        try_to_timestamp_result,
    ));
    out.extend(scalar(
        &["to_timestamp_ntz"],
        &Inputs::Built(InputBuilder(to_date_inputs)),
        to_timestamp_ntz_result,
    ));
    out.extend(scalar(
        &["unix_timestamp", "to_unix_timestamp"],
        &Inputs::Built(InputBuilder(unix_inputs)),
        unix_result,
    ));
    out.extend(scalar(
        &["from_unixtime"],
        &Inputs::Built(InputBuilder(from_unixtime_inputs)),
        always_null,
    ));
    out.extend(scalar(
        &["current_timezone"],
        &Inputs::Any,
        current_timezone_result,
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

/// Whether Spark orders `t` (`RowOrdering.isOrderable`): atomic types,
/// arrays of an orderable element and structs of orderable fields; maps do
/// not.
fn spark_orderable(t: &DataType) -> bool {
    match t {
        DataType::Map(_) => false,
        DataType::Array(a) => spark_orderable(a.element_type()),
        DataType::Struct(s) => s.fields().iter().all(|f| spark_orderable(f.data_type())),
        _ => true,
    }
}

/// `min`/`max` over `function`'s argument: its type, which must order. A
/// struct orders field by field (recursively). One Spark cannot order (a
/// map field) is `DATATYPE_MISMATCH.INVALID_ORDERING_TYPE`; one Spark orders
/// but the struct comparator does not take
/// ([`krishiv_kernels::r#struct::comparable`], the same predicate the
/// executor compares with: an array or interval field) is
/// `KRISHIV_UNSUPPORTED_FEATURE`.
fn ordered_min_max(c: &CallTypes<'_>, function: &str) -> Result<ArgType, KrishivError> {
    if let Some(t) = c.data_type(0)
        && matches!(t, DataType::Struct(_))
        && !krishiv_kernels::r#struct::comparable(&t.to_arrow())
    {
        if spark_orderable(t) {
            return Err(KrishivUnsupportedFeature {
                feature: format!(
                    "{function} over {}, a struct with a field the struct comparator does not take",
                    t.sql()
                ),
            }
            .into());
        }
        return Err(DatatypeMismatchInvalidOrderingType {
            sql_expr: format!("\"{function}({})\"", c.sql(0).unwrap_or_default()),
            function_name: format!("`{function}`"),
            data_type: format!("\"{}\"", t.sql()),
        }
        .into());
    }
    min_max(c)
}

/// `min`: as [`ordered_min_max`].
fn min(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    ordered_min_max(c, "min")
}

/// `max`: as [`ordered_min_max`].
fn max(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    ordered_min_max(c, "max")
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

/// `collect_list(x)`, `collect_set(x)`, `array_agg(x)`: an array of the
/// input's type, never NULL (an empty group is `[]`; NULL elements are
/// skipped by the executor). Spark names `array_agg`'s column
/// `collect_list(...)`, so the analyzer resolves it as `collect_list`
/// (see `krishiv_plan::analyzer`).
fn collect(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let element = c.data_type(0).cloned().unwrap_or(DataType::Null);
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, false).into()),
        false,
    ))
}

/// Whether `t` is or contains a map at any depth.
fn has_map(t: &DataType) -> bool {
    match t {
        DataType::Map(_) => true,
        DataType::Array(a) => has_map(a.element_type()),
        DataType::Struct(s) => s.fields().iter().any(|f| has_map(f.data_type())),
        _ => false,
    }
}

/// `collect_set(x)`: as [`collect`], but Spark refuses an input that
/// contains a map at any depth (`DATATYPE_MISMATCH.UNSUPPORTED_INPUT_TYPE`),
/// as a map has no equality to deduplicate by.
fn collect_set(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if let Some(t) = c.data_type(0).filter(|t| has_map(t)) {
        return Err(DatatypeMismatchUnsupportedInputType {
            sql_expr: format!("\"collect_set({})\"", c.sql(0).unwrap_or_default()),
            function_name: "`collect_set`".to_owned(),
            data_type: format!("\"{}\"", t.sql()),
        }
        .into());
    }
    collect(c)
}

/// `vector_sum(x)` / `vector_avg(x)`: element-wise aggregates over
/// `ARRAY<FLOAT>` only. Any other type is Spark's
/// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` (no implicit cast is
/// declared, so the check below sees the written type, as `ntile`'s
/// bucket check does). The result is an array of `float`, NULL for no
/// non-NULL vector.
fn vector(c: &CallTypes<'_>, name: &str) -> Result<ArgType, KrishivError> {
    let is_floats = matches!(
        c.data_type(0),
        Some(DataType::Array(a)) if *a.element_type() == DataType::Float
    );
    if !is_floats {
        let input = c
            .data_type(0)
            .map_or_else(|| "VOID".to_owned(), DataType::sql);
        let sql = |i: usize| c.sql(i).unwrap_or_default();
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: format!("\"{name}({})\"", sql(0)),
            param_index: "first".to_owned(),
            required_type: "\"ARRAY<FLOAT>\"".to_owned(),
            input_sql: format!("\"{}\"", sql(0)),
            input_type: format!("\"{input}\""),
        }
        .into());
    }
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(DataType::Float, false).into()),
        true,
    ))
}

fn vector_sum(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    vector(c, "vector_sum")
}

fn vector_avg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    vector(c, "vector_avg")
}

/// `bit_and`, `bit_or`, `bit_xor` over `TINYINT` .. `BIGINT`: the result
/// is the input type (NULL for no input). Any other type is Spark's
/// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` (no implicit cast is
/// declared, so the check below sees the written type, as `ntile`'s
/// bucket check and `vector` do). The analyzer calls this for aggregate
/// bit operations only; the scalar bitwise operators live in
/// `bitwise.rs`.
fn bit(c: &CallTypes<'_>, name: &str) -> Result<ArgType, KrishivError> {
    let integral = matches!(
        c.data_type(0),
        Some(DataType::Byte | DataType::Short | DataType::Integer | DataType::Long,)
    );
    if !integral {
        let input = c
            .data_type(0)
            .map_or_else(|| "VOID".to_owned(), DataType::sql);
        let sql = |i: usize| c.sql(i).unwrap_or_default();
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: format!("\"{name}({})\"", sql(0)),
            param_index: "first".to_owned(),
            required_type: "\"BYTE\", \"SHORT\", \"INT\" or \"BIGINT\"".to_owned(),
            input_sql: format!("\"{}\"", sql(0)),
            input_type: format!("\"{input}\""),
        }
        .into());
    }
    Ok(ArgType::new(
        c.data_type(0).cloned().unwrap_or(DataType::Integer),
        true,
    ))
}

fn bit_and(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    bit(c, "bit_and")
}

fn bit_or(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    bit(c, "bit_or")
}

fn bit_xor(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    bit(c, "bit_xor")
}

/// `mode(col)`: the most frequent non-NULL value, of the input's type.
/// The analyzer types the `mode(col, deterministic)` and
/// `mode() WITHIN GROUP (ORDER BY col)` forms itself (this rule only
/// types the plain call, where the argument is the value); the executor
/// breaks ties as Spark does.
fn mode(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    min_max(c)
}

/// `regr_count(y, x)`: the number of non-NULL pairs, a non-NULL `BIGINT`.
fn regr_count(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, false))
}

/// `listagg(expr[, delimiter])` / `string_agg`: the concatenation of the
/// non-NULL values, of the value's type (`STRING` or `BINARY`; NULL for
/// no input). The delimiter is implicitly cast to the value's type by the
/// declared inputs; a non-foldable delimiter is Spark's
/// `NON_FOLDABLE_INPUT`, and one that cannot cast to the value's type (a
/// string delimiter for binary values) Spark's `DATA_DIFF_TYPES`, after
/// Spark's `checkForSameTypeInputExpr`. A NULL delimiter reads as empty;
/// the analyzer fills a missing one with NULL, so the call always shows
/// it.
fn listagg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    // The delimiter must fold (a column does not); a missing one is the
    // NULL default. A delimiter that raises while folding raises here.
    // (`is_plain_literal` covers the literals the constant folder has no
    // value for, like binary ones.)
    if c.args().len() > 1 && !c.is_plain_literal(1) && c.constant(1).is_none() {
        let input = c
            .data_type(1)
            .map_or_else(|| "VOID".to_owned(), DataType::sql);
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: format!("\"listagg({}, {})\"", sql(0), sql(1)),
            input_name: "`delimiter`".to_owned(),
            input_type: format!("\"{input}\""),
            input_expr: format!("\"{}\"", sql(1)),
        }
        .into());
    }
    if c.args().len() > 1
        && let Some(Constant::Error(err)) = c.constant(1)
    {
        return Err(err.clone());
    }
    let (Some(value), mut delimiter) = (c.data_type(0), c.data_type(1).cloned()) else {
        return typed(DataType::String, c);
    };
    // A written NULL delimiter stays NULL (it reads as empty): the
    // implicit cast must not give it a type for the check below. This
    // matches the call as written (an explicit `CAST(NULL AS STRING)`
    // keeps its type, as Spark's check sees it; the SQL layer fills a
    // missing delimiter with NULL, which shows the same way).
    if c.sql(1).is_some_and(|s| s == "NULL") {
        delimiter = Some(DataType::Null);
    }
    // Spark renders the failing delimiter by value, not as SQL: the
    // golden quotes `listagg(c1, , )` for a `', '` delimiter (and `NULL`
    // for a NULL one, as SQL renders it too).
    let delim = match &delimiter {
        None | Some(DataType::Null) => "NULL".to_owned(),
        Some(DataType::String) if matches!(c.constant(1), Some(Constant::String(_))) => {
            match c.constant(1) {
                Some(Constant::String(s)) => s.clone(),
                _ => sql(1),
            }
        }
        Some(_) => sql(1),
    };
    if !matches!(delimiter, None | Some(DataType::Null)) && Some(value) != c.data_type(1) {
        let delim_type = c.data_type(1).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchDataDiffTypes {
            sql_expr: format!("\"listagg({}, {delim})\"", sql(0)),
            function_name: "`listagg`".to_owned(),
            data_type: format!("(\"{}\" or \"{}\")", value.sql(), delim_type.sql()),
        }
        .into());
    }
    // NULL for no input: always nullable.
    Ok(ArgType::new(value.clone(), true))
}

/// `try_sum(x)`: as `sum` (NULL on overflow, ANSI on or off).
fn try_sum(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    sum(c)
}

/// `try_avg(x)`: as `avg` (NULL on overflow, ANSI on or off).
fn try_avg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    avg(c)
}

/// Spark's `PercentileBase.dataType` over the types this engine
/// represents: `double`, or `array<double>` for an array percentage.
/// (Spark keeps an ANSI interval child as an interval; this engine has
/// no interval type, so an interval never reaches typing.)
pub fn exact_percentile_result(array: bool) -> DataType {
    if array {
        DataType::Array(ArrayType::new(DataType::Double, false).into())
    } else {
        DataType::Double
    }
}

/// Spark's `ApproximatePercentile.dataType`: the child's type, or an
/// array of it for an array percentage.
pub fn approx_percentile_result(child: &DataType, array: bool) -> DataType {
    if array {
        DataType::Array(ArrayType::new(child.clone(), false).into())
    } else {
        child.clone()
    }
}

/// Whether the percentage argument at `i` selects an array of
/// percentiles (Spark's `returnPercentileArray`).
fn array_percentage(c: &CallTypes<'_>, i: usize) -> bool {
    matches!(c.data_type(i), Some(DataType::Array(_)))
}

/// `percentile`, `percentile_cont`, `percentile_disc` and `median` (which
/// is `percentile(child, 0.5)`): [`exact_percentile_result`].
fn percentile(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        exact_percentile_result(array_percentage(c, 1)),
        true,
    ))
}

/// Spark's `Median`: `percentile(child, 0.5)`.
fn median(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Double, true))
}

/// Spark's `ApproximatePercentile.dataType`: the child's type, or an
/// array of it for an array percentage.
fn percentile_approx(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let child = c.data_type(0).cloned().unwrap_or(DataType::Null);
    Ok(ArgType::new(
        approx_percentile_result(&child, array_percentage(c, 1)),
        true,
    ))
}

/// Spark's `PercentileCont` / `PercentileDisc` result over the user
/// form (the percentage is the only argument): `double`, or
/// `array<double>` for an array percentage.
fn percentile_ordered(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        exact_percentile_result(array_percentage(c, 0)),
        true,
    ))
}

/// `approx_count_distinct`: `bigint`, never NULL (an empty group is 0).
fn approx_count_distinct(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, false))
}

fn aggregates() -> Vec<Spec> {
    let numeric = || Inputs::Expects(vec![InputType::Numeric { intervals: true }]);
    let boolean_arg = || Inputs::Expects(vec![exact(DataType::Boolean)]);
    let doubles = || Inputs::Expects(vec![exact(DataType::Double)]);
    let integrals = || {
        Inputs::Expects(vec![InputType::OneOf(vec![
            DataType::Byte,
            DataType::Short,
            DataType::Integer,
            DataType::Long,
        ])])
    };
    let string_or_bytes = || {
        Inputs::Expects(vec![
            InputType::OneOf(vec![DataType::String, DataType::Binary]),
            InputType::OneOf(vec![DataType::String, DataType::Binary, DataType::Null]),
        ])
    };
    vec![
        aggregate("count", Inputs::Any, count, Aggregate::Count),
        aggregate("sum", numeric(), sum, Aggregate::Sum),
        aggregate("avg", numeric(), avg, Aggregate::Avg),
        aggregate("mean", numeric(), avg, Aggregate::Avg),
        aggregate("min", Inputs::Any, min, Aggregate::Min),
        aggregate("max", Inputs::Any, max, Aggregate::Max),
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
        // The collect and vector aggregates take any single argument with
        // no implicit casts (the vector ones check the type themselves);
        // `array_agg` is `collect_list` under Spark's display name.
        aggregate("collect_list", Inputs::Any, collect, Aggregate::CollectList),
        aggregate(
            "collect_set",
            Inputs::Any,
            collect_set,
            Aggregate::CollectSet,
        ),
        aggregate("array_agg", Inputs::Any, collect, Aggregate::CollectList),
        aggregate("vector_sum", Inputs::Any, vector_sum, Aggregate::VectorSum),
        aggregate("vector_avg", Inputs::Any, vector_avg, Aggregate::VectorAvg),
        // Bitwise aggregates over `TINYINT` .. `BIGINT`.
        aggregate("bit_and", integrals(), bit_and, Aggregate::BitAnd),
        aggregate("bit_or", integrals(), bit_or, Aggregate::BitOr),
        aggregate("bit_xor", integrals(), bit_xor, Aggregate::BitXor),
        // The linear-regression aggregates read both arguments as
        // `DOUBLE`, like the other moment aggregates.
        aggregate("regr_count", doubles(), regr_count, Aggregate::RegrCount),
        aggregate("regr_avgx", doubles(), moment, Aggregate::RegrAvgx),
        aggregate("regr_avgy", doubles(), moment, Aggregate::RegrAvgy),
        aggregate("regr_sxx", doubles(), moment, Aggregate::RegrSxx),
        aggregate("regr_syy", doubles(), moment, Aggregate::RegrSyy),
        aggregate("regr_sxy", doubles(), moment, Aggregate::RegrSxy),
        aggregate("regr_slope", doubles(), moment, Aggregate::RegrSlope),
        aggregate(
            "regr_intercept",
            doubles(),
            moment,
            Aggregate::RegrIntercept,
        ),
        aggregate("regr_r2", doubles(), moment, Aggregate::RegrR2),
        // `mode(col)`: the analyzer types the deterministic and
        // `WITHIN GROUP` forms itself (see `mode_call`); the plain call
        // takes any single value.
        aggregate("mode", Inputs::Any, mode, Aggregate::Mode),
        // `listagg(expr[, delimiter])`: the analyzer fills a missing
        // delimiter with NULL and lowers `WITHIN GROUP` into the ordering
        // keys (see `listagg_call`); `string_agg` is its alias, resolved
        // as `listagg` throughout for Spark's display name.
        aggregate("listagg", string_or_bytes(), listagg, Aggregate::ListAgg),
        aggregate("string_agg", string_or_bytes(), listagg, Aggregate::ListAgg),
        // `try_sum` / `try_avg`: `sum` / `avg` that give NULL instead of
        // an overflow error, ANSI on or off.
        aggregate("try_sum", numeric(), try_sum, Aggregate::TrySum),
        aggregate("try_avg", numeric(), try_avg, Aggregate::TryAvg),
        // The ordered-set and quantile aggregates: the analyzer checks the
        // percentage, accuracy and frequency itself (they must be foldable
        // and in range, and `WITHIN GROUP` lowers to the ordering key), so
        // no implicit casts are declared here.
        aggregate("percentile", Inputs::Any, percentile, Aggregate::Percentile),
        aggregate(
            "percentile_approx",
            Inputs::Any,
            percentile_approx,
            Aggregate::PercentileApprox,
        ),
        aggregate(
            "approx_percentile",
            Inputs::Any,
            percentile_approx,
            Aggregate::PercentileApprox,
        ),
        aggregate("median", Inputs::Any, median, Aggregate::Median),
        aggregate(
            "percentile_cont",
            Inputs::Any,
            percentile_ordered,
            Aggregate::PercentileCont,
        ),
        aggregate(
            "percentile_disc",
            Inputs::Any,
            percentile_ordered,
            Aggregate::PercentileDisc,
        ),
        aggregate(
            "approx_count_distinct",
            Inputs::Any,
            approx_count_distinct,
            Aggregate::ApproxCountDistinct,
        ),
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
    out.extend(super::map::specs());
    out.extend(super::math::specs());
    out.extend(super::hashing::specs());
    out.extend(super::bitwise::specs());
    out.extend(super::predicate::specs());
    out.extend(super::number::specs());
    out.extend(super::misc::specs());
    out.extend(super::url::specs());
    out.extend(super::csv::specs());
    out.extend(super::xml::specs());
    out.extend(super::vector::specs());
    out.extend(super::element::specs());
    out.extend(super::array_edit::specs());
    out.extend(super::text::specs());
    out.extend(super::hof::specs());
    out.extend(super::collection::specs());
    out.extend(super::r#struct::specs());
    out.extend(casts());
    out
}

/// The cast-named functions (`float(x)`, `int(x)`, ...): Spark analyzes
/// them as `CAST(x AS target)` (its `RuntimeReplaceable`), with
/// `decimal(10,0)` for `decimal`. One definition per name: kind scalar,
/// arity exactly 1 from Spark's catalog, any input, result the target
/// type. The analyzer rewrites the call to a cast during analysis (a call
/// with `DISTINCT` or `FILTER` is rejected as unsupported syntax: it has no
/// cast form), so they run as [`Execution::Cast`] and are never executed as
/// functions.
fn casts() -> Vec<Spec> {
    let cast = |name: &'static str, result: Rule| Spec {
        name,
        kind: FunctionKind::Scalar,
        inputs: Inputs::Any,
        result,
        execution: Execution::Cast,
    };
    vec![
        cast("boolean", boolean),
        cast("tinyint", byte),
        cast("smallint", short),
        cast("int", int),
        cast("bigint", long),
        cast("float", float),
        cast("double", double),
        cast("decimal", decimal_default),
        cast("string", string),
        cast("binary", binary),
        cast("date", date),
        cast("timestamp", timestamp),
    ]
}
