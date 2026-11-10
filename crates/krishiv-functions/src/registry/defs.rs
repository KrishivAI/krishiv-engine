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
use krishiv_types::{ArrayType, DataType, DayTimeField, DecimalType, YearMonthField};

use super::math::is_ansi_interval;
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

/// The `TIME` types a time function takes: the first is what an untyped
/// `NULL` becomes.
fn any_time() -> InputType {
    InputType::StrictOneOf(
        [6, 0, 1, 2, 3, 4, 5]
            .into_iter()
            .map(|precision| DataType::Time { precision })
            .collect(),
    )
}

/// A `TIME(6)` that is NULL for any row that does not parse.
fn time6_nullable(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Time { precision: 6 }, true))
}

fn time6(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Time { precision: 6 }, c)
}

/// The seconds of `time_from_seconds` and `time_to_seconds`: `DECIMAL(14,6)`.
fn time_decimal() -> DataType {
    DataType::Decimal(DecimalType::try_new(14, 6).unwrap_or(DecimalType::USER_DEFAULT))
}

fn time_seconds_input() -> InputType {
    exact(time_decimal())
}

fn time_seconds_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(time_decimal(), c)
}

/// `time_trunc(unit, time)`: the time's own type.
fn time_trunc_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = c
        .args()
        .get(1)
        .and_then(|a| a.data_type.clone())
        .unwrap_or(DataType::Time { precision: 6 });
    typed(t, c)
}

fn double(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Double, c)
}

/// `make_date`: an invalid date is NULL with ANSI off, so the result may be
/// null then whatever the arguments.
fn make_date_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Date, !c.ansi() || c.any_nullable()))
}

/// A `DATE` that is NULL when an argument is.
fn date_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Date, c)
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

/// `current_time([p])`: `TIME(p)`, `p` a foldable integer in `[0, 6]`
/// (`TIME(6)` without it); never NULL.
fn clock_time(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let precision = match c.constant(0) {
        None if c.args().is_empty() => 6,
        Some(Constant::Integral(n)) if (0..=6).contains(n) => u8::try_from(*n).unwrap_or(6),
        Some(Constant::Integral(n)) => {
            return Err(DatatypeMismatchValueOutOfRange {
                sql_expr: format!("\"current_time({n})\""),
                expr_name: "precision".to_owned(),
                value_range: "[0, 6]".to_owned(),
                current_value: n.to_string(),
            }
            .into());
        }
        Some(Constant::Error(e)) => return Err(e.clone()),
        _ => {
            return Err(DatatypeMismatchNonFoldableInput {
                sql_expr: format!("\"current_time({})\"", c.sql(0).unwrap_or_default()),
                input_name: "`precision`".to_owned(),
                input_type: "\"INT\"".to_owned(),
                input_expr: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            }
            .into());
        }
    };
    Ok(ArgType::new(DataType::Time { precision }, false))
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
        &["regexp_extract_all"],
        &Inputs::Expects(vec![
            exact(DataType::String),
            exact(DataType::String),
            exact(DataType::Integer),
        ]),
        super::collection::extract_all_result,
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
    let Some(name) = extract_name(c)? else {
        return Ok(None);
    };
    DatePart::parse(name)
        .map(Some)
        .ok_or_else(|| invalid_extract_field(c, name))
}

fn invalid_extract_field(c: &CallTypes<'_>, name: &str) -> KrishivError {
    InvalidExtractField {
        field: format!("`{name}`"),
        expr: format!("\"{}\"", c.sql(1).unwrap_or_default()),
    }
    .into()
}

/// The literal field of `extract`/`date_part`; `None` for a NULL one.
fn extract_name<'a>(c: &CallTypes<'a>) -> Result<Option<&'a str>, KrishivError> {
    match c.constant(0) {
        Some(Constant::String(s)) => Ok(Some(s)),
        Some(Constant::Null) => Ok(None),
        _ => Err(NonFoldableArgument {
            func_name: "`date_part`".to_owned(),
            param_name: "`field`".to_owned(),
            param_type: "\"STRING\"".to_owned(),
        }
        .into()),
    }
}

/// Spark's `ExtractIntervalPart.parseExtractField`: the names an interval
/// takes, fewer than a date's, and only the fields its type holds.
fn interval_field(name: &str, t: &DataType) -> Option<DatePart> {
    use DayTimeField as D;
    use YearMonthField as Y;
    let part = match name.to_ascii_uppercase().as_str() {
        "YEAR" | "Y" | "YEARS" | "YR" | "YRS" => DatePart::Year,
        "MONTH" | "MON" | "MONS" | "MONTHS" => DatePart::Month,
        "DAY" | "D" | "DAYS" => DatePart::Day,
        "HOUR" | "H" | "HOURS" | "HR" | "HRS" => DatePart::Hour,
        "MINUTE" | "M" | "MIN" | "MINS" | "MINUTES" => DatePart::Minute,
        "SECOND" | "S" | "SEC" | "SECONDS" | "SECS" => DatePart::Second,
        _ => return None,
    };
    let holds = match (t, part) {
        (DataType::CalendarInterval, _) => true,
        (DataType::YearMonthInterval { start, .. }, DatePart::Year) => *start == Y::Year,
        (DataType::YearMonthInterval { end, .. }, DatePart::Month) => *end == Y::Month,
        (DataType::DayTimeInterval { start, end }, _) => {
            let field = match part {
                DatePart::Day => D::Day,
                DatePart::Hour => D::Hour,
                DatePart::Minute => D::Minute,
                DatePart::Second => D::Second,
                _ => return None,
            };
            (*start..=*end).contains(&field)
        }
        _ => false,
    };
    holds.then_some(part)
}

const fn is_interval(t: &DataType) -> bool {
    matches!(
        t,
        DataType::YearMonthInterval { .. }
            | DataType::DayTimeInterval { .. }
            | DataType::CalendarInterval
    )
}

/// `extract`/`date_part` of an interval: a month, hour or minute is a `TINYINT`.
fn interval_extract_result(c: &CallTypes<'_>, t: &DataType) -> Result<ArgType, KrishivError> {
    let nullable = c.args().get(1).is_some_and(|a| a.nullable);
    let Some(name) = extract_name(c)? else {
        return Ok(ArgType::new(DataType::Double, true));
    };
    let part = interval_field(name, t).ok_or_else(|| invalid_extract_field(c, name))?;
    Ok(match part {
        DatePart::Month | DatePart::Hour | DatePart::Minute => {
            ArgType::new(DataType::Byte, nullable)
        }
        DatePart::Second => ArgType::new(DataType::Decimal(DecimalType::try_new(8, 6)?), nullable),
        _ => ArgType::new(DataType::Integer, nullable),
    })
}

fn is_time_field(part: DatePart) -> bool {
    matches!(part, DatePart::Hour | DatePart::Minute | DatePart::Second)
}

/// `extract(field FROM source)`: the date fields read a `DATE` (a
/// timestamp is cast), the time fields a timestamp (a date is cast).
fn extract_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    if c.data_type(1).is_some_and(is_interval) {
        return vec![exact(DataType::String), InputType::Any];
    }
    let source = match extract_field(c) {
        Ok(Some(part)) if is_time_field(part) => InputType::OneOf(
            [DataType::Timestamp, DataType::TimestampNtz]
                .into_iter()
                .chain((0..=6).map(|precision| DataType::Time { precision }))
                .collect(),
        ),
        Ok(Some(_)) => exact(DataType::Date),
        // The result rule reports the field.
        Ok(None) | Err(_) => InputType::Any,
    };
    vec![exact(DataType::String), source]
}

fn extract_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if let Some(t) = c.data_type(1).filter(|t| is_interval(t)) {
        return interval_extract_result(c, t);
    }
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

fn timestampadd_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![
        exact(DataType::String),
        exact(DataType::Long),
        instant_input(c, 2),
    ]
}

fn timestampdiff_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![
        exact(DataType::String),
        instant_input(c, 1),
        instant_input(c, 2),
    ]
}

/// `timestampadd` keeps a `TIMESTAMP_NTZ` and makes every other input
/// (a date included) a `TIMESTAMP`.
fn timestampadd_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.data_type(2) == Some(&DataType::TimestampNtz) {
        timestamp_ntz_result(c)
    } else {
        timestamp_result(c)
    }
}

/// The seconds of `make_timestamp`: `DECIMAL(16,6)`.
fn seconds_input() -> InputType {
    exact(DataType::Decimal(
        DecimalType::try_new(16, 6).unwrap_or(DecimalType::USER_DEFAULT),
    ))
}

/// The `(date[, time[, zone]])` forms of `make_timestamp`.
fn make_timestamp_date_inputs(n: usize) -> Vec<InputType> {
    let time = InputType::OneOf(
        (0..=6)
            .map(|precision| DataType::Time { precision })
            .collect(),
    );
    [
        InputType::Strict(DataType::Date),
        time,
        exact(DataType::String),
    ]
    .into_iter()
    .take(n)
    .collect()
}

fn make_timestamp_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    if c.args().len() <= 3 {
        return make_timestamp_date_inputs(c.args().len());
    }
    let mut inputs = vec![exact(DataType::Integer); 5];
    inputs.push(seconds_input());
    if c.args().len() == 7 {
        inputs.push(exact(DataType::String));
    }
    inputs
}

fn make_timestamp_ntz_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    if c.args().len() <= 2 {
        return make_timestamp_date_inputs(c.args().len());
    }
    let mut inputs = vec![exact(DataType::Integer); 5];
    inputs.push(seconds_input());
    inputs
}

/// A malformed field is NULL with ANSI off and always for the `try_`
/// forms, so the result may be null then whatever the arguments.
fn make_timestamp_result(c: &CallTypes<'_>, ty: DataType, try_form: bool) -> ArgType {
    // The date forms have no field that can be malformed.
    let can_fail = c.args().len() > 3 && (try_form || !c.ansi());
    ArgType::new(ty, can_fail || c.any_nullable())
}

fn make_timestamp_ltz(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(make_timestamp_result(c, DataType::Timestamp, false))
}

fn try_make_timestamp_ltz(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(make_timestamp_result(c, DataType::Timestamp, true))
}

fn make_timestamp_ntz(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(make_timestamp_result(c, DataType::TimestampNtz, false))
}

fn try_make_timestamp_ntz(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(make_timestamp_result(c, DataType::TimestampNtz, true))
}

/// The seconds of `make_dt_interval` and `make_interval`: `DECIMAL(18,6)`.
fn interval_seconds_input() -> InputType {
    exact(DataType::Decimal(
        DecimalType::try_new(18, 6).unwrap_or(DecimalType::USER_DEFAULT),
    ))
}

fn make_dt_interval_inputs(_: &CallTypes<'_>) -> Vec<InputType> {
    let mut inputs = vec![exact(DataType::Integer); 3];
    inputs.push(interval_seconds_input());
    inputs
}

fn make_interval_inputs(_: &CallTypes<'_>) -> Vec<InputType> {
    let mut inputs = vec![exact(DataType::Integer); 6];
    inputs.push(interval_seconds_input());
    inputs
}

fn make_ym_interval(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let ty = DataType::YearMonthInterval {
        start: YearMonthField::Year,
        end: YearMonthField::Month,
    };
    Ok(ArgType::new(ty, c.any_nullable()))
}

fn make_dt_interval(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let ty = DataType::DayTimeInterval {
        start: DayTimeField::Day,
        end: DayTimeField::Second,
    };
    Ok(ArgType::new(ty, c.any_nullable()))
}

/// An overflow is NULL with ANSI off, so the result may be null then whatever
/// the arguments.
fn make_interval(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        DataType::CalendarInterval,
        !c.ansi() || c.any_nullable(),
    ))
}

fn try_make_interval(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::CalendarInterval, true))
}

fn months_between_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![
        instant_input(c, 0),
        instant_input(c, 1),
        // Spark casts a string `roundOff` only under ANSI; without it the
        // call is a type mismatch.
        if c.ansi() {
            exact(DataType::Boolean)
        } else {
            InputType::Strict(DataType::Boolean)
        },
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

/// `timestamp_seconds`: a float or double NaN or infinity is NULL, so the
/// result may be null whatever the arguments.
fn timestamp_seconds_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let float = matches!(c.data_type(0), Some(DataType::Float | DataType::Double));
    Ok(ArgType::new(DataType::Timestamp, float || c.any_nullable()))
}

fn timestamp_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Timestamp, c)
}

fn timestamp_ntz_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::TimestampNtz, c)
}

/// Whether `t` is a numeric type (Spark's `NumericType`).
pub(super) fn is_numeric_type(t: &DataType) -> bool {
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
    // `timestamp_seconds` takes any number; `timestamp_millis` and
    // `timestamp_micros` an integral type, nothing implicit (a string or a
    // decimal is `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`).
    out.extend(scalar(
        &["timestamp_seconds"],
        &Inputs::Expects(vec![InputType::StrictNumeric]),
        timestamp_seconds_result,
    ));
    out.extend(scalar(
        &["timestamp_millis", "timestamp_micros"],
        &Inputs::Expects(vec![InputType::StrictOneOf(vec![
            DataType::Byte,
            DataType::Short,
            DataType::Integer,
            DataType::Long,
        ])]),
        timestamp_result,
    ));
    out.extend(scalar(
        &["unix_seconds", "unix_millis", "unix_micros"],
        &Inputs::Expects(vec![InputType::Strict(DataType::Timestamp)]),
        long,
    ));
    out.extend(scalar(
        &["date_from_unix_date"],
        &Inputs::Expects(vec![exact(DataType::Integer)]),
        date_result,
    ));
    out.extend(scalar(
        &["unix_date"],
        &Inputs::Expects(vec![InputType::Strict(DataType::Date)]),
        int,
    ));
    // The time fields read a timestamp of either kind; a date is cast to
    // one.
    let any_timestamp = InputType::OneOf(
        [DataType::Timestamp, DataType::TimestampNtz]
            .into_iter()
            .chain((0..=6).map(|precision| DataType::Time { precision }))
            .collect(),
    );
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
        &["time_trunc"],
        &Inputs::Expects(vec![exact(DataType::String), InputType::AnyTimeOrString]),
        time_trunc_result,
    ));
    out.extend(scalar(
        &["make_time"],
        &Inputs::Expects(vec![
            exact(DataType::Integer),
            exact(DataType::Integer),
            seconds_input(),
        ]),
        time6,
    ));
    out.extend(scalar(
        &["time_from_seconds"],
        &Inputs::Expects(vec![time_seconds_input()]),
        time6,
    ));
    out.extend(scalar(
        &["time_from_millis", "time_from_micros"],
        &Inputs::Expects(vec![exact(DataType::Long)]),
        time6,
    ));
    out.extend(scalar(
        &["time_to_seconds"],
        &Inputs::Expects(vec![any_time()]),
        time_seconds_result,
    ));
    out.extend(scalar(
        &["time_to_millis", "time_to_micros"],
        &Inputs::Expects(vec![any_time()]),
        long,
    ));
    out.extend(scalar(
        &["to_time"],
        &Inputs::Expects(vec![InputType::Strict(DataType::String)]),
        time6,
    ));
    out.extend(scalar(
        &["try_to_time"],
        &Inputs::Expects(vec![InputType::Strict(DataType::String)]),
        time6_nullable,
    ));
    out.extend(scalar(
        &["time_diff"],
        &Inputs::Expects(vec![exact(DataType::String), any_time(), any_time()]),
        long,
    ));
    out.extend(scalar(
        &["timestampadd"],
        &Inputs::Built(InputBuilder(timestampadd_inputs)),
        timestampadd_result,
    ));
    out.extend(scalar(
        &["timestampdiff"],
        &Inputs::Built(InputBuilder(timestampdiff_inputs)),
        long,
    ));
    let make = Inputs::Built(InputBuilder(make_timestamp_inputs));
    let make_ntz = Inputs::Built(InputBuilder(make_timestamp_ntz_inputs));
    out.extend(scalar(
        &["make_timestamp", "make_timestamp_ltz"],
        &make,
        make_timestamp_ltz,
    ));
    out.extend(scalar(
        &["try_make_timestamp", "try_make_timestamp_ltz"],
        &make,
        try_make_timestamp_ltz,
    ));
    out.extend(scalar(
        &["make_timestamp_ntz"],
        &make_ntz,
        make_timestamp_ntz,
    ));
    out.extend(scalar(
        &["try_make_timestamp_ntz"],
        &make_ntz,
        try_make_timestamp_ntz,
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
        &["make_ym_interval"],
        &Inputs::Expects(vec![exact(DataType::Integer); 2]),
        make_ym_interval,
    ));
    out.extend(scalar(
        &["make_dt_interval"],
        &Inputs::Built(InputBuilder(make_dt_interval_inputs)),
        make_dt_interval,
    ));
    let calendar = Inputs::Built(InputBuilder(make_interval_inputs));
    out.extend(scalar(&["make_interval"], &calendar, make_interval));
    out.extend(scalar(&["try_make_interval"], &calendar, try_make_interval));
    out.extend(scalar(
        &["make_date"],
        &Inputs::Expects(vec![exact(DataType::Integer)]),
        make_date_result,
    ));
    out.extend(scalar(
        &["extract", "date_part", "datepart"],
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
        &["current_time"],
        &Inputs::Expects(vec![exact(DataType::Integer)]),
        clock_time,
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
        // An interval or timestamp offset has no implicit cast to `INT` (Spark's `ImplicitCastInputTypes`).
        Constant::Interval(_) | Constant::Timestamp(_) => {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call,
                param_index: "second".to_owned(),
                required_type: "\"INT\"".to_owned(),
                input_sql: format!("\"{}\"", sql(1)),
                input_type: c
                    .data_type(1)
                    .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql())),
            }
            .into());
        }
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
    out.extend(super::generator::specs());
    out.extend(super::json::specs());
    out.extend(super::map::specs());
    out.extend(super::math::specs());
    out.extend(super::hashing::specs());
    out.extend(super::crypto::specs());
    out.extend(super::bitwise::specs());
    out.extend(super::predicate::specs());
    out.extend(super::number::specs());
    out.extend(super::misc::specs());
    out.extend(super::geo::specs());
    out.extend(super::url::specs());
    out.extend(super::csv::specs());
    out.extend(super::xml::specs());
    out.extend(super::vector::specs());
    out.extend(super::element::specs());
    out.extend(super::array_edit::specs());
    out.extend(super::array_scan::specs());
    out.extend(super::array_set::specs());
    out.extend(super::array_shape::specs());
    out.extend(super::text::specs());
    out.extend(super::time::specs());
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
        cast("time", time6),
    ]
}

mod aggregates;
pub(super) use aggregates::spark_orderable;
use aggregates::*;
pub use aggregates::{approx_percentile_result, exact_percentile_result};
