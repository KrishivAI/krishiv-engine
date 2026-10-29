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
use arrow_array::types::{Decimal128Type, Int32Type, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{InvalidTimezone, KrishivError, UnableToAcquireMemory};
use krishiv_kernels::OverflowMode;
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
use krishiv_kernels::datetime::{
    DatePart, Zones, convert_timezone, current_date, date_add, date_add_months, date_diff,
    date_sub, day_name, extract_date, extract_timestamp, from_utc_timestamp, last_day, make_date,
    month_name, months_between, next_day, to_utc_timestamp, trunc_date, trunc_timestamp, unix_date,
};

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

/// Whether `text` could be a zone id that is not an offset `FixedOffset`
/// reads: a plausible region id (`Asia/Seoul`), a short id (`CET`), or an
/// offset spelled another way (`+0530`, seconds). Anything else (`xx`,
/// `foo`, `utc`, `abc`, `true`, `etc/utc`) is text no zone id can have, and
/// is `INVALID_TIMEZONE`: Spark's own `INVALID_TIMEZONE` text requires
/// region ids to have the form `area/city` (recorded in
/// `tests/fixtures/spark-4.2.0/function-matrix`).
fn is_zone_id(text: &str) -> bool {
    let rest = ["UTC", "GMT", "UT"]
        .iter()
        .find_map(|p| text.strip_prefix(p))
        .unwrap_or(text);
    if let Some(signed) = rest.strip_prefix(['+', '-']) {
        return offset_parts(signed).is_some();
    }
    is_region_id(text) || SHORT_ZONE_IDS.contains(&text)
}

/// Short zone ids without an `area/city` form that Java's `ZoneId.of`
/// accepts: its `SHORT_IDS` aliases plus the single-component TZDB regions
/// (`CET`, `EET`, `WET`, `MET`).
const SHORT_ZONE_IDS: &[&str] = &[
    "ACT", "AET", "AGT", "ART", "AST", "BET", "BST", "CAT", "CNT", "CST", "CTT", "EAT", "ECT",
    "IET", "IST", "JST", "MIT", "NET", "NST", "PLT", "PNT", "PRT", "PST", "SST", "VST", "EST",
    "MST", "HST", "CET", "EET", "WET", "MET",
];

/// Whether `text` looks like a region id Java's `ZoneId.of` accepts: an
/// `area/city` form of non-empty parts in zone-name characters whose area
/// starts with an uppercase letter (every TZDB area does, so `etc/utc` is
/// not one).
fn is_region_id(text: &str) -> bool {
    if !text.contains('/') {
        return false;
    }
    let mut parts = text.split('/');
    if !parts
        .next()
        .is_some_and(|first| first.starts_with(|c: char| c.is_ascii_uppercase()))
    {
        return false;
    }
    text.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-' | '+' | '.'))
        && text.split('/').all(|part| !part.is_empty())
}

/// The `(hours, minutes, seconds)` of a sign-stripped offset `text` where it
/// parses as an in-range offset (`+HHMM`, `+HHMMSS`, `+H[:MM[:SS]]`), `None`
/// otherwise. Malformed offsets (`+ab`) and out-of-range ones (`+19:00`,
/// which Spark's `INVALID_TIMEZONE` text limits to +-18:00) are not zones at
/// all; in-range ones `FixedOffset::parse` does not read need a fuller
/// offset parser.
fn offset_parts(text: &str) -> Option<(u32, u32, u32)> {
    let (hours, minutes, seconds) = if text.contains(':') {
        let mut fields = text.split(':');
        let (h, m, s) = (fields.next()?, fields.next(), fields.next());
        if fields.next().is_some() {
            return None;
        }
        if !(1..=2).contains(&h.len())
            || !h.bytes().all(|b| b.is_ascii_digit())
            || m.is_some_and(|m| {
                !(1..=2).contains(&m.len()) || !m.bytes().all(|b| b.is_ascii_digit())
            })
            || s.is_some_and(|s| {
                !(1..=2).contains(&s.len()) || !s.bytes().all(|b| b.is_ascii_digit())
            })
        {
            return None;
        }
        (
            h.parse().ok()?,
            m.map(str::parse).transpose().ok()??,
            s.map(str::parse).transpose().ok()??,
        )
    } else {
        let digits = text.as_bytes();
        if !digits.iter().all(u8::is_ascii_digit) {
            return None;
        }
        match digits.len() {
            4 => (text[..2].parse().ok()?, text[2..].parse().ok()?, 0),
            6 => (
                text[..2].parse().ok()?,
                text[2..4].parse().ok()?,
                text[4..].parse().ok()?,
            ),
            _ => return None,
        }
    };
    (minutes < 60 && seconds < 60 && (hours < 18 || (hours == 18 && minutes == 0 && seconds == 0)))
        .then_some((hours, minutes, seconds))
}

/// The zone the text names. Only offsets `FixedOffset::parse` reads
/// resolve: a plausible region id (`Asia/Seoul`, a short id such as `CET`)
/// or an offset spelled another way (`+0530`, seconds) needs the time-zone
/// database or a fuller offset parser, and is `KRISHIV_UNSUPPORTED_FEATURE`;
/// text no zone id can have (`xx`, `abc`, `etc/utc`) is `INVALID_TIMEZONE`.
fn resolve_zone(text: &str) -> Result<FixedOffset, KrishivError> {
    if text == text.trim()
        && let Some(zone) = FixedOffset::parse(text)
    {
        return Ok(zone);
    }
    if is_zone_id(text) {
        return Err(unsupported(format!(
            "the time zone {text} (region ids need the time-zone database)"
        )));
    }
    Err(InvalidTimezone {
        time_zone: text.to_owned(),
    }
    .into())
}

/// A zone argument: a literal is resolved when the call is bound (`None` is
/// a NULL zone), a column at each invocation.
enum ZoneArg {
    Literal(Option<FixedOffset>),
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
fn column_zones(names: &StringArray) -> Result<Vec<Option<FixedOffset>>, KrishivError> {
    let mut out = reserve(names.len())?;
    let mut last: Option<(&str, FixedOffset)> = None;
    for name in names.iter() {
        out.push(match name {
            None => None,
            Some(n) => match last {
                Some((seen, zone)) if seen == n => Some(zone),
                _ => {
                    let zone = resolve_zone(n)?;
                    last = Some((n, zone));
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
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
