// SPDX-License-Identifier: Apache-2.0
//! Datetime functions (P2-T025a kernels): `date_add`/`date_sub`/`datediff`,
//! the field functions (`year` ... `second`, `extract`/`date_part`), the
//! calendar functions (`add_months`, `last_day`, `next_day`, `trunc`,
//! `date_trunc`, `months_between`, `make_date`, `monthname`, `dayname`,
//! `unix_date`) and the query clock (`current_date`, `current_timestamp`,
//! `localtimestamp`).
//!
//! Day arithmetic raises `ARITHMETIC_OVERFLOW` under ANSI where the day
//! number leaves the `int` range and wraps like Java's `int` otherwise (ADR
//! 0023: Spark wraps under ANSI too).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Int64Array, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{KrishivError, UnableToAcquireMemory};
use krishiv_kernels::OverflowMode;
use krishiv_kernels::cast::{CastMode, CastOptions, cast};
use krishiv_kernels::datetime::{
    DatePart, ParsePolicy, ZoneRef, Zones, convert_timezone, current_date, date_add,
    date_add_months, date_diff, date_sub, day_name, extract_date, extract_timestamp,
    format_timestamps, from_utc_timestamp, last_day, make_date, month_name, months_between,
    next_day, parse_date_one, parse_dates, parse_ntz, parse_ntz_one, parse_timestamp_one,
    parse_timestamps, seconds_to_timestamps, to_utc_timestamp, trunc_date, trunc_timestamp,
    unix_date,
};
use krishiv_kernels::datetime::{
    date_from_unix_date, timestamp_from_integral, timestamp_from_seconds,
    timestamp_from_seconds_decimal, timestamp_from_seconds_double, unix_time,
};
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone};

use super::args::{Inputs, internal, is_string, is_timestamp, to_utf8, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "date_add",
    "dateadd",
    "date_sub",
    "datediff",
    "date_diff",
    "year",
    "month",
    "day",
    "dayofmonth",
    "dayofweek",
    "dayofyear",
    "weekofyear",
    "weekday",
    "quarter",
    "hour",
    "minute",
    "second",
    "monthname",
    "dayname",
    "unix_date",
    "timestamp_seconds",
    "timestamp_millis",
    "timestamp_micros",
    "unix_seconds",
    "unix_millis",
    "unix_micros",
    "date_from_unix_date",
    "add_months",
    "last_day",
    "next_day",
    "trunc",
    "date_trunc",
    "months_between",
    "make_date",
    "from_utc_timestamp",
    "to_utc_timestamp",
    "convert_timezone",
    "extract",
    "date_part",
    "current_date",
    "curdate",
    "current_timestamp",
    "now",
    "localtimestamp",
    "date_format",
    "to_date",
    "try_to_date",
    "to_timestamp",
    "try_to_timestamp",
    "to_timestamp_ltz",
    "to_timestamp_ntz",
    "unix_timestamp",
    "to_unix_timestamp",
    "from_unixtime",
    "current_timezone",
];

/// The Arrow type of `timestamp` results.
fn timestamp() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")))
}

/// The Arrow type of `timestamp_ntz`.
fn ntz_type() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, None)
}

/// Argument `i`, a timestamp column, cast to `to` in the session zone when
/// `convert` (a `timestamp_ntz` read as an instant, or an instant as a wall
/// clock), as the implicit cast does.
fn timestamps_as(
    x: &Inputs<'_>,
    i: usize,
    convert: bool,
    to: &DataType,
    opts: &BindOptions,
) -> Result<TimestampMicrosecondArray, KrishivError> {
    let t = x.timestamps(i)?;
    if !convert {
        return Ok(t);
    }
    let cast_options = CastOptions {
        mode: CastMode::Ansi,
        time_zone: opts.time_zone.as_ref(),
    };
    let out = cast(&t, to, &cast_options)?;
    out.as_primitive_opt::<TimestampMicrosecondType>()
        .cloned()
        .ok_or_else(|| internal("a timestamp cast to another carrier".to_owned()))
}

/// Argument `i` as instants: a `timestamp_ntz` (`ntz`) is read in the
/// session zone.
fn instants(
    x: &Inputs<'_>,
    i: usize,
    ntz: bool,
    opts: &BindOptions,
) -> Result<TimestampMicrosecondArray, KrishivError> {
    timestamps_as(x, i, ntz, &timestamp(), opts)
}

/// The zone a text argument names, resolved against the bundled
/// time-zone database: unknown ids are `INVALID_TIMEZONE`.
fn resolve_zone(text: &str) -> Result<ZoneRef, KrishivError> {
    krishiv_kernels::datetime::resolve_zone(text)
}

fn is_integral_type(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    )
}

/// What `timestamp_seconds` reads: any integral, float or decimal column.
fn timestamp_seconds_input(t: &DataType) -> bool {
    is_integral_type(t)
        || matches!(
            t,
            DataType::Float32 | DataType::Float64 | DataType::Decimal128(..)
        )
}

/// Argument 0 cast to `to` (a widening of an integral or a float).
fn widen<T: arrow_array::ArrowPrimitiveType>(
    x: &Inputs<'_>,
    to: &DataType,
) -> Result<arrow_array::PrimitiveArray<T>, KrishivError> {
    let column = x.column(0)?;
    let options = CastOptions {
        mode: CastMode::Ansi,
        time_zone: &krishiv_kernels::cast::FixedOffset(0),
    };
    let widened = cast(column.as_ref(), to, &options)?;
    widened
        .as_primitive_opt::<T>()
        .cloned()
        .ok_or_else(|| internal(format!("a column of type {to} after the cast")))
}

/// A zone argument: a literal is resolved when the call is bound (`None` is
/// a NULL zone), a column at each invocation.
enum ZoneArg {
    Literal(Option<ZoneRef>),
    Column,
}

fn zone_arg(args: &[BindArg], i: usize) -> Result<ZoneArg, KrishivError> {
    let arg = args
        .get(i)
        .ok_or_else(|| internal(format!("missing zone argument {i}")))?;
    let BindArg::Literal(a) = arg else {
        return Ok(ZoneArg::Column);
    };
    let text = to_utf8(a.as_ref())?;
    if text.is_null(0) {
        return Ok(ZoneArg::Literal(None));
    }
    resolve_zone(text.value(0)).map(|z| ZoneArg::Literal(Some(z)))
}

/// Memory-tracked scratch, reporting allocation failure as
/// `UNABLE_TO_ACQUIRE_MEMORY` instead of aborting.
fn reserve<T>(len: usize) -> Result<Vec<T>, KrishivError> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| UnableToAcquireMemory {
            requested_bytes: len.saturating_mul(size_of::<T>()).to_string(),
            received_bytes: "0".to_owned(),
        })?;
    Ok(out)
}

/// The zone of each row of a zone column; a NULL is `None`. Runs of equal
/// names resolve once.
fn column_zones(names: &StringArray) -> Result<Vec<Option<ZoneRef>>, KrishivError> {
    let mut out = reserve(names.len())?;
    let mut last: Option<(&str, ZoneRef)> = None;
    for name in names.iter() {
        out.push(match name {
            None => None,
            Some(n) => match last {
                Some((seen, ref zone)) if seen == n => Some(zone.clone()),
                _ => {
                    let zone = resolve_zone(n)?;
                    last = Some((n, zone.clone()));
                    Some(zone)
                }
            },
        });
    }
    Ok(out)
}

/// A `timestamp` (an instant) column of `rows` NULLs, for a NULL literal zone.
fn null_instants(rows: usize) -> TimestampMicrosecondArray {
    TimestampMicrosecondArray::new_null(rows).with_timezone("UTC")
}

/// A `timestamp_ntz` column of `rows` NULLs, for a NULL literal zone.
fn null_wall_clocks(rows: usize) -> TimestampMicrosecondArray {
    TimestampMicrosecondArray::new_null(rows)
}

/// Argument `i` as a wall clock (`timestamp_ntz`): a `timestamp` (`instant`)
/// is read in the session zone.
fn wall_clocks(
    x: &Inputs<'_>,
    i: usize,
    instant: bool,
    opts: &BindOptions,
) -> Result<TimestampMicrosecondArray, KrishivError> {
    timestamps_as(x, i, instant, &ntz_type(), opts)
}

fn is_day_count(t: &DataType) -> bool {
    matches!(t, DataType::Int8 | DataType::Int16 | DataType::Int32)
}

/// The field a literal `extract` field argument (a string) names; `None`
/// for a NULL field.
fn field(arg: &BindArg) -> Result<Option<DatePart>, KrishivError> {
    let BindArg::Literal(a) = arg else {
        return Err(unsupported(
            "extract with a field that is not a literal".to_owned(),
        ));
    };
    let name = to_utf8(a.as_ref())?;
    if name.is_null(0) {
        return Ok(None);
    }
    let name = name.value(0);
    DatePart::parse(name)
        .map(Some)
        .ok_or_else(|| unsupported(format!("extract field {name}")))
}

/// Extracts `part` from the date or timestamp column `i`.
/// `source` is the argument's type: a date, or a timestamp (with a zone
/// label for an instant).
fn extract(
    x: &Inputs<'_>,
    i: usize,
    part: DatePart,
    source: &DataType,
    opts: &BindOptions,
) -> Result<ArrayRef, KrishivError> {
    if *source == DataType::Date32 {
        return Ok(extract_date(part, &x.dates(i)?)?);
    }
    let zone = matches!(source, DataType::Timestamp(_, Some(_))).then(|| opts.time_zone.as_ref());
    Ok(extract_timestamp(part, &x.timestamps(i)?, zone)?)
}

// ---------------------------------------------------------------------
// Pattern-driven functions (`date_format`, `to_date`, `to_timestamp`,
// `unix_timestamp`, `to_unix_timestamp`, `from_unixtime`).
// ---------------------------------------------------------------------

/// How many distinct column patterns one invocation compiles at most: the
/// cache clears when full, which changes nothing observable.
const PATTERN_CACHE_CAP: usize = 8;

/// A format argument: a literal compiles once per bind (`Null` is a NULL
/// literal), a column per distinct value at each invocation.
enum PatternArg {
    Null,
    Literal(DateTimePattern),
    Column,
}

fn pattern_arg(args: &[BindArg], i: usize, purpose: Purpose) -> Result<PatternArg, KrishivError> {
    let arg = args
        .get(i)
        .ok_or_else(|| internal(format!("missing pattern argument {i}")))?;
    let BindArg::Literal(a) = arg else {
        return Ok(PatternArg::Column);
    };
    let text = to_utf8(a.as_ref())?;
    if text.is_null(0) {
        return Ok(PatternArg::Null);
    }
    DateTimePattern::new(text.value(0), purpose).map(PatternArg::Literal)
}

/// The compiled column patterns of one invocation, keyed by pattern text.
struct PatternCache {
    purpose: Purpose,
    map: std::collections::HashMap<String, DateTimePattern>,
}

impl PatternCache {
    fn new(purpose: Purpose) -> Self {
        Self {
            purpose,
            map: std::collections::HashMap::new(),
        }
    }

    fn get(&mut self, text: &str) -> Result<&DateTimePattern, KrishivError> {
        if !self.map.contains_key(text) {
            if self.map.len() >= PATTERN_CACHE_CAP {
                self.map.clear();
            }
            let compiled = DateTimePattern::new(text, self.purpose)?;
            self.map.insert(text.to_owned(), compiled);
        }
        self.map
            .get(text)
            .ok_or_else(|| internal("a pattern cache lost its pattern".to_owned()))
    }
}

/// The session zone for formatting: its id and rules.
fn session_zone(opts: &BindOptions) -> Zone<'_> {
    Zone {
        id: &opts.time_zone_name,
        rules: opts.time_zone.as_ref(),
    }
}

/// The cast mode of a one-argument `to_*` call: `try_*` is `Try`, ANSI is
/// `Ansi`, else `Legacy` (NULL for malformed input).
fn cast_mode(opts: &BindOptions, is_try: bool) -> CastMode {
    if is_try {
        CastMode::Try
    } else if opts.ansi {
        CastMode::Ansi
    } else {
        CastMode::Legacy
    }
}

/// The parse policy of a pattern call: `try_*` never raises for
/// unparseable input, otherwise ANSI decides.
fn parse_policy(opts: &BindOptions, is_try: bool, func: &'static str) -> ParsePolicy {
    ParsePolicy {
        fail_on_error: !is_try && opts.ansi,
        func,
    }
}

/// One-argument `to_date`/`to_timestamp` over a non-string, non-numeric
/// value: Spark's replacement is a cast, so this reuses the cast kernel
/// (there is no second cast path here).
fn cast_one_arg(
    x: &Inputs<'_>,
    to: &DataType,
    opts: &BindOptions,
    mode: CastMode,
) -> Result<ArrayRef, KrishivError> {
    let input = x.plain_column(0)?;
    let options = CastOptions {
        mode,
        time_zone: opts.time_zone.as_ref(),
    };
    Ok(cast(input.as_ref(), to, &options)?)
}

// ---------------------------------------------------------------------
// Small predicates over bound Arrow types.
// ---------------------------------------------------------------------

/// Whether `t` is a `DATE` or a microsecond timestamp of either kind.
fn is_date_or_timestamp(t: &DataType) -> bool {
    matches!(t, DataType::Date32) || is_timestamp(t)
}

/// Whether `t` is a date, a timestamp or a string carrier.
fn is_date_or_timestamp_or_string(t: &DataType) -> bool {
    is_date_or_timestamp(t) || is_string(t)
}

/// Whether `t` is a numeric carrier `to_timestamp` reads as seconds.
fn is_numeric_arrow(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(..)
    )
}

/// Whether `t` is an integral carrier `from_unixtime` reads.
fn is_longish(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    )
}

/// The default pattern (`yyyy-MM-dd HH:mm:ss`) for the calls that omit
/// one: it always compiles.
fn default_pattern(purpose: Purpose) -> Result<PatternArg, KrishivError> {
    DateTimePattern::new("yyyy-MM-dd HH:mm:ss", purpose).map(PatternArg::Literal)
}

/// Any date, timestamp or string value as an instant in the session zone
/// (Spark's implicit cast to `TIMESTAMP`).
fn cast_to_instants(
    x: &Inputs<'_>,
    o: &BindOptions,
    mode: CastMode,
) -> Result<TimestampMicrosecondArray, KrishivError> {
    let out = cast_one_arg(x, &timestamp(), o, mode)?;
    out.as_primitive_opt::<TimestampMicrosecondType>()
        .cloned()
        .ok_or_else(|| internal("a timestamp cast to another carrier".to_owned()))
}

/// ANDs the validity of `a` with non-NULL pattern rows: a NULL pattern is
/// NULL, as Spark's `nullIntolerant` binary functions answer.
fn apply_fmt_nulls<T: arrow_array::ArrowPrimitiveType>(
    a: &mut arrow_array::PrimitiveArray<T>,
    fmts: &StringArray,
) where
    arrow_array::PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
{
    if fmts.null_count() == 0 {
        return;
    }
    let kept: Vec<Option<T::Native>> = a
        .iter()
        .zip(fmts.iter())
        .map(|(v, f)| v.filter(|_| f.is_some()))
        .collect();
    *a = arrow_array::PrimitiveArray::<T>::from(kept);
}

/// Tags a pattern error with its row.
fn row_error(row: usize, e: KrishivError) -> KrishivError {
    krishiv_kernels::KernelError::with_row(Some(row), e).into()
}

/// `to_date` over a column pattern: per-row compile through the bounded
/// cache, a NULL pattern row is NULL.
fn parse_dates_column(
    strings: &StringArray,
    fmts: &StringArray,
    o: &BindOptions,
    policy: ParsePolicy,
    operation: &str,
) -> Result<ArrayRef, KrishivError> {
    let mut cache = PatternCache::new(Purpose::Parse);
    let mut out = reserve::<Option<i32>>(strings.len())?;
    for i in 0..strings.len() {
        if strings.is_null(i) || fmts.is_null(i) {
            out.push(None);
            continue;
        }
        let p = cache.get(fmts.value(i)).map_err(|e| row_error(i, e))?;
        out.push(parse_date_one(
            p,
            strings.value(i),
            o.time_zone.as_ref(),
            policy,
            operation,
            i,
        )?);
    }
    Ok(Arc::new(Date32Array::from(out)) as ArrayRef)
}

/// `to_timestamp` over a column pattern, as above, in the session zone.
fn parse_timestamps_column(
    strings: &StringArray,
    fmts: &StringArray,
    o: &BindOptions,
    policy: ParsePolicy,
    operation: &str,
) -> Result<TimestampMicrosecondArray, KrishivError> {
    let mut cache = PatternCache::new(Purpose::Parse);
    let mut out = reserve::<Option<i64>>(strings.len())?;
    for i in 0..strings.len() {
        if strings.is_null(i) || fmts.is_null(i) {
            out.push(None);
            continue;
        }
        let p = cache.get(fmts.value(i)).map_err(|e| row_error(i, e))?;
        out.push(parse_timestamp_one(
            p,
            strings.value(i),
            o.time_zone.as_ref(),
            policy,
            operation,
            i,
        )?);
    }
    Ok(TimestampMicrosecondArray::from(out))
}

/// `to_timestamp_ntz` over a column pattern, as above.
fn parse_ntz_column(
    strings: &StringArray,
    fmts: &StringArray,
    policy: ParsePolicy,
    operation: &str,
) -> Result<ArrayRef, KrishivError> {
    let mut cache = PatternCache::new(Purpose::Parse);
    let mut out = reserve::<Option<i64>>(strings.len())?;
    for i in 0..strings.len() {
        if strings.is_null(i) || fmts.is_null(i) {
            out.push(None);
            continue;
        }
        let p = cache.get(fmts.value(i)).map_err(|e| row_error(i, e))?;
        out.push(parse_ntz_one(p, strings.value(i), policy, operation, i)?);
    }
    Ok(Arc::new(TimestampMicrosecondArray::from(out)) as ArrayRef)
}

/// Renders a bound argument for the unresolved-replacement error below:
/// literals read as Spark renders them (strings bare, numbers plain), anything
/// else as its type. Approximate where Spark prints SQL text: no golden or
/// matrix case compares the message.
fn bind_arg_text(arg: &BindArg) -> String {
    let lit = |a: &ArrayRef| {
        if a.len() != 1 || a.is_null(0) {
            return None;
        }
        Some(match a.data_type() {
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => {
                to_utf8(a.as_ref()).ok()?.value(0).to_owned()
            }
            DataType::Int8 => a.as_primitive::<Int8Type>().value(0).to_string(),
            DataType::Int16 => a.as_primitive::<Int16Type>().value(0).to_string(),
            DataType::Int32 => a.as_primitive::<Int32Type>().value(0).to_string(),
            DataType::Int64 => a.as_primitive::<Int64Type>().value(0).to_string(),
            _ => return None,
        })
    };
    match arg {
        BindArg::Literal(a) => lit(a).unwrap_or_else(|| a.data_type().to_string()),
        BindArg::Column { ty, .. } => ty.to_string(),
    }
}

/// An integral `from_unixtime` argument as seconds.
fn longs_column(x: &Inputs<'_>, i: usize) -> Result<Int64Array, KrishivError> {
    use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
    let c = x.column(i)?;
    Ok(match c.data_type() {
        DataType::Int8 => c.as_primitive::<Int8Type>().unary(i64::from),
        DataType::Int16 => c.as_primitive::<Int16Type>().unary(i64::from),
        DataType::Int32 => c.as_primitive::<Int32Type>().unary(i64::from),
        DataType::Int64 => c.as_primitive::<Int64Type>().clone(),
        t => {
            return Err(internal(format!("a from_unixtime argument of type {t}")));
        }
    })
}

/// Binds the datetime function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let ctx = opts.kernel;
    let mode = if opts.ansi {
        OverflowMode::AnsiError
    } else {
        OverflowMode::JavaWrap
    };
    let o = opts.clone();
    let binding = match (listed, types.as_slice()) {
        ("date_add" | "dateadd" | "date_sub", [DataType::Date32, n]) if is_day_count(n) => {
            let sub = listed == "date_sub";
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                let (d, n) = (x.date(0)?, x.int(1)?);
                let out = if sub {
                    date_sub(d.operand(), n.operand(), mode, &ctx)?
                } else {
                    date_add(d.operand(), n.operand(), mode, &ctx)?
                };
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("datediff" | "date_diff", [DataType::Date32, DataType::Date32]) => {
            Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
                let (end, start) = (x.date(0)?, x.date(1)?);
                Ok(Arc::new(date_diff(end.operand(), start.operand(), mode, &ctx)?) as ArrayRef)
            })
        }
        (
            "year" | "month" | "day" | "dayofmonth" | "dayofweek" | "dayofyear" | "weekofyear"
            | "weekday" | "quarter",
            [t],
        ) if *t == DataType::Date32 || is_timestamp(t) => {
            let part = match listed {
                "year" => DatePart::Year,
                "month" => DatePart::Month,
                "dayofweek" => DatePart::DayOfWeek,
                "dayofyear" => DatePart::DayOfYear,
                "weekofyear" => DatePart::Week,
                // `weekday` is the ISO day of the week less one.
                "weekday" => DatePart::DayOfWeekIso,
                "quarter" => DatePart::Quarter,
                _ => DatePart::Day,
            };
            let (source, weekday) = (t.clone(), listed == "weekday");
            Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
                let field = extract(x, 0, part, &source, &o)?;
                if !weekday {
                    return Ok(field);
                }
                // `weekday` is the ISO day of the week less one (1 to 7).
                let days = field.as_primitive::<Int32Type>();
                Ok(Arc::new(days.unary::<_, Int32Type>(|d| {
                    debug_assert!((1..=7).contains(&d), "extract DAYOFWEEK_ISO is 1 to 7");
                    d.wrapping_sub(1)
                })) as ArrayRef)
            })
        }
        ("hour" | "minute" | "second", [t]) if is_timestamp(t) => {
            let part = match listed {
                "hour" => DatePart::Hour,
                "minute" => DatePart::Minute,
                _ => DatePart::Second,
            };
            let source = t.clone();
            Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
                let field = extract(x, 0, part, &source, &o)?;
                if part != DatePart::Second {
                    return Ok(field);
                }
                // `second` is the whole seconds of `extract`'s `decimal(8, 6)`
                // (the time field is below one minute, so 0 to 59 whole
                // seconds: the conversion is infallible).
                let micros = field.as_primitive::<Decimal128Type>();
                let whole = micros.unary::<_, Int32Type>(|m| {
                    debug_assert!(
                        (0..60_000_000).contains(&m),
                        "extract SECOND is below one minute"
                    );
                    i32::try_from(m / 1_000_000).unwrap_or_default()
                });
                Ok(Arc::new(whole) as ArrayRef)
            })
        }
        ("monthname" | "dayname", [DataType::Date32]) => {
            let months = listed == "monthname";
            Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                let d = x.dates(0)?;
                Ok(Arc::new(if months { month_name(&d) } else { day_name(&d) }) as ArrayRef)
            })
        }
        ("timestamp_seconds", [t]) if timestamp_seconds_input(t) => {
            let source = t.clone();
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let out = match &source {
                    DataType::Float32 | DataType::Float64 => timestamp_from_seconds_double(
                        &widen::<Float64Type>(x, &DataType::Float64)?,
                    )?,
                    DataType::Decimal128(..) => {
                        timestamp_from_seconds_decimal(&x.array::<Decimal128Type>(0)?)?
                    }
                    _ => timestamp_from_seconds(&widen::<Int64Type>(x, &DataType::Int64)?)?,
                };
                Ok(Arc::new(out.with_timezone("UTC")) as ArrayRef)
            })
        }
        ("timestamp_millis" | "timestamp_micros", [t]) if is_integral_type(t) => {
            let per = if listed == "timestamp_millis" {
                1_000
            } else {
                1
            };
            let name = listed;
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let out =
                    timestamp_from_integral(&widen::<Int64Type>(x, &DataType::Int64)?, per, name)?;
                Ok(Arc::new(out.with_timezone("UTC")) as ArrayRef)
            })
        }
        ("unix_seconds" | "unix_millis" | "unix_micros", [t])
            if is_timestamp(t) && *t != ntz_type() =>
        {
            let per = match listed {
                "unix_seconds" => 1_000_000,
                "unix_millis" => 1_000,
                _ => 1,
            };
            Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
                Ok(Arc::new(unix_time(&x.timestamps(0)?, per)?) as ArrayRef)
            })
        }
        ("date_from_unix_date", [DataType::Int32]) => {
            Binding::new(DataType::Date32, |x: &Inputs<'_>| {
                Ok(Arc::new(date_from_unix_date(&x.array::<Int32Type>(0)?)?) as ArrayRef)
            })
        }
        ("unix_date", [DataType::Date32]) => Binding::new(DataType::Int32, |x: &Inputs<'_>| {
            Ok(Arc::new(unix_date(&x.dates(0)?)) as ArrayRef)
        }),
        ("add_months", [DataType::Date32, n]) if is_day_count(n) => {
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                let (d, n) = (x.date(0)?, x.int(1)?);
                let out = date_add_months(d.operand(), n.operand(), false, false, &ctx)?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("last_day", [DataType::Date32]) => Binding::new(DataType::Date32, |x: &Inputs<'_>| {
            Ok(Arc::new(last_day(&x.dates(0)?)) as ArrayRef)
        }),
        ("next_day", [DataType::Date32, s]) if is_string(s) => {
            let ansi = opts.ansi;
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                Ok(Arc::new(next_day(&x.dates(0)?, &x.strings(1)?, ansi)?) as ArrayRef)
            })
        }
        ("trunc", [DataType::Date32, s]) if is_string(s) => {
            let ansi = opts.ansi;
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                Ok(Arc::new(trunc_date(&x.dates(0)?, &x.strings(1)?, ansi)?) as ArrayRef)
            })
        }
        ("date_trunc", [s, t]) if is_string(s) && is_timestamp(t) => {
            let ntz = *t == ntz_type();
            let ansi = opts.ansi;
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let utc = instants(x, 1, ntz, &o)?;
                let out = trunc_timestamp(&x.strings(0)?, &utc, o.time_zone.as_ref(), ansi)?;
                Ok(Arc::new(out.with_timezone("UTC")) as ArrayRef)
            })
        }
        ("months_between", [a, b] | [a, b, DataType::Boolean])
            if is_timestamp(a) && is_timestamp(b) =>
        {
            let explicit = types.len() == 3;
            let (end_ntz, start_ntz) = (*a == ntz_type(), *b == ntz_type());
            let ansi = opts.ansi;
            Binding::new(DataType::Float64, move |x: &Inputs<'_>| {
                let round_off = if explicit {
                    x.booleans(2)?
                } else {
                    // The default `roundOff` is true for every row.
                    let mut values = reserve::<bool>(x.rows())?;
                    values.resize(x.rows(), true);
                    BooleanArray::new(values.into(), None)
                };
                let out = months_between(
                    &instants(x, 0, end_ntz, &o)?,
                    &instants(x, 1, start_ntz, &o)?,
                    &round_off,
                    o.time_zone.as_ref(),
                    ansi,
                )?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("from_utc_timestamp" | "to_utc_timestamp", [t, z]) if is_timestamp(t) && is_string(z) => {
            let zone = match zone_arg(args, 1) {
                Ok(z) => z,
                Err(e) => return Some((listed, Err(e))),
            };
            let (ntz, from) = (*t == ntz_type(), listed == "from_utc_timestamp");
            let ansi = opts.ansi;
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let utc = instants(x, 0, ntz, &o)?;
                // A literal zone reads every row with one lookup; a column
                // resolves each row (runs of equal names once).
                let out = match &zone {
                    ZoneArg::Literal(None) => null_instants(x.rows()),
                    ZoneArg::Literal(Some(one)) => {
                        let zones = Zones::One(one);
                        (if from {
                            from_utc_timestamp(&utc, &zones, ansi)
                        } else {
                            to_utc_timestamp(&utc, &zones, ansi)
                        })?
                        .with_timezone("UTC")
                    }
                    ZoneArg::Column => {
                        let names = column_zones(&x.strings(1)?)?;
                        let zones = Zones::PerRow(&names);
                        (if from {
                            from_utc_timestamp(&utc, &zones, ansi)
                        } else {
                            to_utc_timestamp(&utc, &zones, ansi)
                        })?
                        .with_timezone("UTC")
                    }
                };
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("convert_timezone", [.., t]) if is_timestamp(t) && matches!(types.len(), 2 | 3) => {
            let three = types.len() == 3;
            let target_index = types.len().saturating_sub(2);
            let target = match zone_arg(args, target_index) {
                Ok(z) => z,
                Err(e) => return Some((listed, Err(e))),
            };
            let source = if three {
                match zone_arg(args, 0) {
                    Ok(z) => Some(z),
                    Err(e) => return Some((listed, Err(e))),
                }
            } else {
                None
            };
            let instant = *t != ntz_type();
            let ansi = opts.ansi;
            Binding::new(ntz_type(), move |x: &Inputs<'_>| {
                let clocks = wall_clocks(x, target_index + 1, instant, &o)?;
                // Literals read every row with one lookup; a NULL literal
                // zone is NULL for every row.
                let target_vec;
                let target_zones = match &target {
                    ZoneArg::Literal(Some(one)) => Zones::One(one),
                    ZoneArg::Literal(None) => {
                        return Ok(Arc::new(null_wall_clocks(x.rows())) as ArrayRef);
                    }
                    ZoneArg::Column => {
                        target_vec = column_zones(&x.strings(target_index)?)?;
                        Zones::PerRow(&target_vec)
                    }
                };
                let out = if let Some(arg) = &source {
                    let source_vec;
                    let source_zones = match arg {
                        ZoneArg::Literal(Some(one)) => Zones::One(one),
                        ZoneArg::Literal(None) => {
                            return Ok(Arc::new(null_wall_clocks(x.rows())) as ArrayRef);
                        }
                        ZoneArg::Column => {
                            source_vec = column_zones(&x.strings(0)?)?;
                            Zones::PerRow(&source_vec)
                        }
                    };
                    convert_timezone(&source_zones, &target_zones, &clocks, ansi)?
                } else {
                    convert_timezone(
                        &Zones::One(o.time_zone.as_ref()),
                        &target_zones,
                        &clocks,
                        ansi,
                    )?
                };
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("make_date", [DataType::Int32, DataType::Int32, DataType::Int32]) => {
            let ansi = opts.ansi;
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                let (y, m, d) = (
                    x.array::<Int32Type>(0)?,
                    x.array::<Int32Type>(1)?,
                    x.array::<Int32Type>(2)?,
                );
                Ok(Arc::new(make_date(&y, &m, &d, ansi, &ctx)?) as ArrayRef)
            })
        }
        ("extract" | "date_part", [f, t]) if is_string(f) => match field(&args[0]) {
            Err(e) => return Some((listed, Err(e))),
            // `date_part(NULL, x)` is a NULL `double`, whatever `x` is.
            Ok(None) => Binding::new(DataType::Float64, |x: &Inputs<'_>| {
                Ok(arrow_array::new_null_array(&DataType::Float64, x.rows()))
            }),
            Ok(Some(part)) if *t == DataType::Date32 || is_timestamp(t) => {
                let source = t.clone();
                Binding::new(part.result_type(), move |x: &Inputs<'_>| {
                    extract(x, 1, part, &source, &o)
                })
            }
            Ok(Some(_)) => return Some((listed, Err(bad_types(listed, args)))),
        },
        ("current_date" | "curdate", []) => {
            let day = match current_date(opts.query_start_micros, opts.time_zone.as_ref()) {
                Ok(d) => d,
                Err(e) => return Some((listed, Err(e.into()))),
            };
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                Ok(Arc::new(Date32Array::from(vec![day; x.rows()])) as ArrayRef)
            })
        }
        ("current_timestamp" | "now", []) => {
            let now = opts.query_start_micros;
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let a = TimestampMicrosecondArray::from(vec![now; x.rows()]).with_timezone("UTC");
                Ok(Arc::new(a) as ArrayRef)
            })
        }
        ("localtimestamp", []) => {
            // The wall clock of the session zone, as a `timestamp_ntz`.
            let now = opts.query_start_micros;
            let offset = i64::from(opts.time_zone.offset_at_utc(now.div_euclid(1_000_000)));
            let Some(local) = offset
                .checked_mul(1_000_000)
                .and_then(|o| now.checked_add(o))
            else {
                return Some((
                    listed,
                    Err(internal(
                        "localtimestamp: the query clock leaves the timestamp range".to_owned(),
                    )),
                ));
            };
            Binding::new(
                DataType::Timestamp(TimeUnit::Microsecond, None),
                move |x: &Inputs<'_>| {
                    Ok(
                        Arc::new(TimestampMicrosecondArray::from(vec![local; x.rows()]))
                            as ArrayRef,
                    )
                },
            )
        }
        ("date_format", [t, s]) if is_string(s) && is_date_or_timestamp_or_string(t) => {
            let pat = match pattern_arg(args, 1, Purpose::Format) {
                Ok(p) => p,
                Err(e) => return Some((listed, Err(e))),
            };
            let mode = cast_mode(opts, false);
            Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                // Dates, timestamps and strings all format as the instant
                // the session zone gives them (Spark's implicit cast to
                // `TIMESTAMP`); a NULL literal pattern is NULL for every
                // row.
                let instants = cast_to_instants(x, &o, mode)?;
                let zone = session_zone(&o);
                match &pat {
                    PatternArg::Null => Ok(Arc::new(StringArray::new_null(x.rows())) as ArrayRef),
                    PatternArg::Literal(p) => {
                        Ok(Arc::new(format_timestamps(p, &instants, zone)?) as ArrayRef)
                    }
                    PatternArg::Column => {
                        let fmts = x.strings(1)?;
                        let mut cache = PatternCache::new(Purpose::Format);
                        let mut out = reserve::<Option<String>>(x.rows())?;
                        for i in 0..x.rows() {
                            if instants.is_null(i) || fmts.is_null(i) {
                                out.push(None);
                                continue;
                            }
                            let p = cache
                                .get(fmts.value(i))
                                .map_err(|e| krishiv_kernels::KernelError::with_row(Some(i), e))?;
                            out.push(Some(p.format_timestamp(instants.value(i), zone).map_err(
                                |e| krishiv_kernels::KernelError::with_row(Some(i), e),
                            )?));
                        }
                        Ok(
                            Arc::new(StringArray::from_iter(out.iter().map(|s| s.as_deref())))
                                as ArrayRef,
                        )
                    }
                }
            })
        }
        ("to_date" | "try_to_date", [t]) if is_date_or_timestamp_or_string(t) => {
            let is_try = listed == "try_to_date";
            let mode = cast_mode(opts, is_try);
            // One argument is Spark's `CAST`: the default grammar in the
            // session zone.
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                cast_one_arg(x, &DataType::Date32, &o, mode)
            })
        }
        ("to_date" | "try_to_date", [t, s])
            if is_string(s) && is_date_or_timestamp_or_string(t) =>
        {
            let is_try = listed == "try_to_date";
            let first = t.clone();
            if !is_string(&first) {
                // Dates and timestamps ignore the pattern (Spark never
                // evaluates it for them); a NULL pattern is still NULL.
                let mode = cast_mode(opts, is_try);
                return Some((
                    listed,
                    Ok(Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                        let out = cast_one_arg(x, &DataType::Date32, &o, mode)?;
                        let mut dates = out
                            .as_primitive_opt::<Date32Type>()
                            .cloned()
                            .ok_or_else(|| internal("a date cast to another carrier".to_owned()))?;
                        apply_fmt_nulls(&mut dates, &x.strings(1)?);
                        Ok(Arc::new(dates) as ArrayRef)
                    })),
                ));
            }
            let pat = match pattern_arg(args, 1, Purpose::Parse) {
                Ok(p) => p,
                Err(e) => return Some((listed, Err(e))),
            };
            let policy = parse_policy(opts, is_try, "try_to_date");
            Binding::new(DataType::Date32, move |x: &Inputs<'_>| {
                let strings = x.strings(0)?;
                match &pat {
                    PatternArg::Null => Ok(Arc::new(Date32Array::new_null(x.rows())) as ArrayRef),
                    PatternArg::Literal(p) => Ok(Arc::new(parse_dates(
                        p,
                        &strings,
                        o.time_zone.as_ref(),
                        policy,
                        "to_date",
                    )?) as ArrayRef),
                    PatternArg::Column => {
                        let fmts = x.strings(1)?;
                        parse_dates_column(&strings, &fmts, &o, policy, "to_date")
                    }
                }
            })
        }
        ("to_timestamp" | "try_to_timestamp" | "to_timestamp_ltz", [t]) => {
            let is_try = listed == "try_to_timestamp";
            let mode = cast_mode(opts, is_try);
            if is_numeric_arrow(t) {
                // Integrals are seconds since the epoch, as Spark's
                // `longToTimestamp` reads them.
                return Some((
                    listed,
                    Ok(Binding::new(timestamp(), move |x: &Inputs<'_>| {
                        let out = seconds_to_timestamps(&x.plain_column(0)?, "to_timestamp")?
                            .with_timezone("UTC");
                        Ok(Arc::new(out) as ArrayRef)
                    })),
                ));
            }
            if !is_date_or_timestamp_or_string(t) {
                return Some((listed, Err(bad_types(listed, args))));
            }
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                cast_one_arg(x, &timestamp(), &o, mode)
            })
        }
        ("to_timestamp" | "try_to_timestamp" | "to_timestamp_ltz", [_, s]) if is_string(s) => {
            let is_try = listed == "try_to_timestamp";
            let first = args[0].data_type();
            if !is_string(&first) {
                // Dates and timestamps ignore the pattern; a NULL pattern
                // is still NULL.
                if is_numeric_arrow(&first) {
                    // A numeric with a pattern is Spark's unresolved
                    // `gettimestamp` replacement: an internal error for
                    // every non-NULL row, NULL rows stay NULL.
                    let message = format!(
                        "Cannot resolve the runtime replaceable expression \"{listed}({}, {})\". The replacement is unresolved: \"gettimestamp({}, {})\".",
                        bind_arg_text(&args[0]),
                        bind_arg_text(&args[1]),
                        bind_arg_text(&args[0]),
                        bind_arg_text(&args[1]),
                    );
                    return Some((
                        listed,
                        Ok(Binding::new(timestamp(), move |x: &Inputs<'_>| {
                            let (values, fmts) = (x.plain_column(0)?, x.strings(1)?);
                            let evaluable =
                                (0..x.rows()).any(|i| !values.is_null(i) && !fmts.is_null(i));
                            if evaluable {
                                return Err(internal(message.clone()));
                            }
                            Ok(Arc::new(null_instants(x.rows())) as ArrayRef)
                        })),
                    ));
                }
                if !is_date_or_timestamp(&first) {
                    return Some((listed, Err(bad_types(listed, args))));
                }
                let mode = cast_mode(opts, is_try);
                return Some((
                    listed,
                    Ok(Binding::new(timestamp(), move |x: &Inputs<'_>| {
                        let out = cast_one_arg(x, &timestamp(), &o, mode)?;
                        let stamps = out
                            .as_primitive_opt::<TimestampMicrosecondType>()
                            .cloned()
                            .ok_or_else(|| {
                                internal("a timestamp cast to another carrier".to_owned())
                            })?;
                        let mut out = stamps;
                        apply_fmt_nulls(&mut out, &x.strings(1)?);
                        Ok(Arc::new(out.with_timezone("UTC")) as ArrayRef)
                    })),
                ));
            }
            let pat = match pattern_arg(args, 1, Purpose::Parse) {
                Ok(p) => p,
                Err(e) => return Some((listed, Err(e))),
            };
            let policy = parse_policy(opts, is_try, "try_to_timestamp");
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let strings = x.strings(0)?;
                let out: ArrayRef = match &pat {
                    PatternArg::Null => Arc::new(null_instants(x.rows())),
                    PatternArg::Literal(p) => Arc::new(
                        parse_timestamps(
                            p,
                            &strings,
                            o.time_zone.as_ref(),
                            policy,
                            "to_timestamp",
                        )?
                        .with_timezone("UTC"),
                    ),
                    PatternArg::Column => Arc::new(
                        parse_timestamps_column(
                            &strings,
                            &x.strings(1)?,
                            &o,
                            policy,
                            "to_timestamp",
                        )?
                        .with_timezone("UTC"),
                    ),
                };
                Ok(out)
            })
        }
        ("to_timestamp_ntz", [t]) if is_date_or_timestamp_or_string(t) => {
            let is_try = false;
            let mode = cast_mode(opts, is_try);
            Binding::new(ntz_type(), move |x: &Inputs<'_>| {
                cast_one_arg(x, &ntz_type(), &o, mode)
            })
        }
        ("to_timestamp_ntz", [t, s]) if is_string(s) && is_date_or_timestamp_or_string(t) => {
            let first = t.clone();
            if !is_string(&first) {
                let mode = cast_mode(opts, false);
                return Some((
                    listed,
                    Ok(Binding::new(ntz_type(), move |x: &Inputs<'_>| {
                        let out = cast_one_arg(x, &ntz_type(), &o, mode)?;
                        let stamps = out
                            .as_primitive_opt::<TimestampMicrosecondType>()
                            .cloned()
                            .ok_or_else(|| {
                                internal("a timestamp cast to another carrier".to_owned())
                            })?;
                        let mut out = stamps;
                        apply_fmt_nulls(&mut out, &x.strings(1)?);
                        Ok(Arc::new(out) as ArrayRef)
                    })),
                ));
            }
            let pat = match pattern_arg(args, 1, Purpose::Parse) {
                Ok(p) => p,
                Err(e) => return Some((listed, Err(e))),
            };
            let policy = parse_policy(opts, false, "try_to_timestamp");
            Binding::new(ntz_type(), move |x: &Inputs<'_>| {
                let strings = x.strings(0)?;
                match &pat {
                    PatternArg::Null => Ok(Arc::new(null_wall_clocks(x.rows())) as ArrayRef),
                    PatternArg::Literal(p) => {
                        Ok(
                            Arc::new(parse_ntz(p, &strings, policy, "to_timestamp_ntz")?)
                                as ArrayRef,
                        )
                    }
                    PatternArg::Column => {
                        parse_ntz_column(&strings, &x.strings(1)?, policy, "to_timestamp_ntz")
                    }
                }
            })
        }
        ("unix_timestamp", []) => {
            let now = opts.query_start_micros;
            Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
                Ok(Arc::new(Int64Array::from(vec![now / 1_000_000; x.rows()])) as ArrayRef)
            })
        }
        ("unix_timestamp" | "to_unix_timestamp", [t] | [t, _])
            if is_date_or_timestamp_or_string(t) =>
        {
            let has_fmt = types.len() == 2;
            let first = t.clone();
            // Dates and timestamps ignore the pattern; strings without one
            // use `yyyy-MM-dd HH:mm:ss`.
            let pat = if !is_string(&first) {
                PatternArg::Null
            } else if has_fmt {
                match pattern_arg(args, 1, Purpose::Parse) {
                    Ok(p) => p,
                    Err(e) => return Some((listed, Err(e))),
                }
            } else {
                match default_pattern(Purpose::Parse) {
                    Ok(p) => p,
                    Err(e) => return Some((listed, Err(e))),
                }
            };
            let policy = parse_policy(opts, false, "try_to_timestamp");
            Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
                let stamps = if is_string(&first) {
                    let strings = x.strings(0)?;
                    match &pat {
                        PatternArg::Literal(p) => parse_timestamps(
                            p,
                            &strings,
                            o.time_zone.as_ref(),
                            policy,
                            "unix_timestamp",
                        )?,
                        PatternArg::Column => parse_timestamps_column(
                            &strings,
                            &x.strings(1)?,
                            &o,
                            policy,
                            "unix_timestamp",
                        )?,
                        // A NULL literal pattern is NULL for every row.
                        PatternArg::Null => TimestampMicrosecondArray::new_null(x.rows()),
                    }
                } else {
                    // The instant the session zone gives the value (Spark's
                    // replacement), with a NULL pattern still NULL.
                    let out = cast_one_arg(x, &timestamp(), &o, cast_mode(&o, false))?;
                    let stamps = out
                        .as_primitive_opt::<TimestampMicrosecondType>()
                        .cloned()
                        .ok_or_else(|| {
                            internal("a timestamp cast to another carrier".to_owned())
                        })?;
                    let mut out = stamps.with_timezone("UTC");
                    if has_fmt {
                        apply_fmt_nulls(&mut out, &x.strings(1)?);
                    }
                    out
                };
                // Seconds, truncated toward zero as Java divides them.
                let seconds = stamps.unary::<_, Int64Type>(|m| m / 1_000_000);
                Ok(Arc::new(seconds) as ArrayRef)
            })
        }
        ("from_unixtime", [n] | [n, _]) if is_longish(n) => {
            let has_fmt = types.len() == 2;
            let pat = if has_fmt {
                match pattern_arg(args, 1, Purpose::Format) {
                    Ok(p) => p,
                    Err(e) => return Some((listed, Err(e))),
                }
            } else {
                match default_pattern(Purpose::Format) {
                    Ok(p) => p,
                    Err(e) => return Some((listed, Err(e))),
                }
            };
            Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                let secs = longs_column(x, 0)?;
                // Seconds to micros, wrapping as Java's `long` multiply
                // does; the wall clock then renders in the session zone.
                let stamps: TimestampMicrosecondArray = secs
                    .iter()
                    .map(|s| s.map(|v| v.wrapping_mul(1_000_000)))
                    .collect();
                let zone = session_zone(&o);
                match &pat {
                    // A NULL literal pattern is NULL for every row.
                    PatternArg::Null => Ok(Arc::new(StringArray::new_null(x.rows())) as ArrayRef),
                    PatternArg::Literal(p) => {
                        Ok(Arc::new(format_timestamps(p, &stamps, zone)?) as ArrayRef)
                    }
                    PatternArg::Column => {
                        let fmts = x.strings(1)?;
                        let mut cache = PatternCache::new(Purpose::Format);
                        let mut out = reserve::<Option<String>>(x.rows())?;
                        for i in 0..x.rows() {
                            if stamps.is_null(i) || fmts.is_null(i) {
                                out.push(None);
                                continue;
                            }
                            let p = cache
                                .get(fmts.value(i))
                                .map_err(|e| krishiv_kernels::KernelError::with_row(Some(i), e))?;
                            out.push(Some(p.format_timestamp(stamps.value(i), zone).map_err(
                                |e| krishiv_kernels::KernelError::with_row(Some(i), e),
                            )?));
                        }
                        Ok(
                            Arc::new(StringArray::from_iter(out.iter().map(|s| s.as_deref())))
                                as ArrayRef,
                        )
                    }
                }
            })
        }
        ("current_timezone", []) => {
            let name = Arc::clone(&opts.configured_time_zone);
            Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                Ok(Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    name.as_ref(),
                    x.rows(),
                ))) as ArrayRef)
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
