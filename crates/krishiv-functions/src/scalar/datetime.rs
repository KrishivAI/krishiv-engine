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
use arrow_array::{Array, ArrayRef, BooleanArray, Date32Array, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{InvalidTimezone, KrishivError};
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
/// reads: a region id (`Asia/Seoul`, `CET`) or an offset spelled another way.
/// Text that cannot name any zone in Java's `ZoneId.of` is `INVALID_TIMEZONE`.
fn is_zone_id(text: &str) -> bool {
    let rest = ["UTC", "GMT", "UT"]
        .iter()
        .find_map(|p| text.strip_prefix(p))
        .unwrap_or(text);
    let offset_like = rest.starts_with(['+', '-'])
        && rest.chars().skip(1).all(|c| c.is_ascii_digit() || c == ':');
    let region = text.chars().count() >= 2
        && text.chars().enumerate().all(|(i, c)| {
            c.is_ascii_alphabetic()
                || (i > 0 && (c.is_ascii_digit() || matches!(c, '/' | '~' | '.' | '_' | '+' | '-')))
        });
    offset_like || region
}

/// The zone the text names. Only offsets resolve (`UTC`, `+08:00`): a region
/// id needs the time-zone database, which the engine does not have, and is
/// `KRISHIV_UNSUPPORTED_FEATURE`; text no zone can have is `INVALID_TIMEZONE`.
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

/// The zone of each of `x.rows()` rows from argument `i`; a NULL is `None`.
fn zones_of(
    x: &Inputs<'_>,
    i: usize,
    arg: &ZoneArg,
) -> Result<Vec<Option<FixedOffset>>, KrishivError> {
    match arg {
        ZoneArg::Literal(zone) => Ok(vec![*zone; x.rows()]),
        ZoneArg::Column => {
            let names = x.strings(i)?;
            let mut last: Option<(&str, FixedOffset)> = None;
            let mut out = Vec::with_capacity(names.len());
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
    }
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
                let days = field.as_primitive::<Int32Type>();
                Ok(Arc::new(days.unary::<_, Int32Type>(|d| d.wrapping_sub(1))) as ArrayRef)
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
                // `second` is the whole seconds of `extract`'s `decimal(8, 6)`.
                let micros = field.as_primitive::<Decimal128Type>();
                let whole = micros
                    .unary::<_, Int32Type>(|m| i32::try_from(m / 1_000_000).unwrap_or_default());
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
            Binding::new(DataType::Date32, |x: &Inputs<'_>| {
                Ok(Arc::new(trunc_date(&x.dates(0)?, &x.strings(1)?)?) as ArrayRef)
            })
        }
        ("date_trunc", [s, t]) if is_string(s) && is_timestamp(t) => {
            let ntz = *t == ntz_type();
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let utc = instants(x, 1, ntz, &o)?;
                let out = trunc_timestamp(&x.strings(0)?, &utc, o.time_zone.as_ref())?;
                Ok(Arc::new(out.with_timezone("UTC")) as ArrayRef)
            })
        }
        ("months_between", [a, b] | [a, b, DataType::Boolean])
            if is_timestamp(a) && is_timestamp(b) =>
        {
            let explicit = types.len() == 3;
            let (end_ntz, start_ntz) = (*a == ntz_type(), *b == ntz_type());
            Binding::new(DataType::Float64, move |x: &Inputs<'_>| {
                let round_off = if explicit {
                    x.booleans(2)?
                } else {
                    BooleanArray::from(vec![true; x.rows()])
                };
                let out = months_between(
                    &instants(x, 0, end_ntz, &o)?,
                    &instants(x, 1, start_ntz, &o)?,
                    &round_off,
                    o.time_zone.as_ref(),
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
            Binding::new(timestamp(), move |x: &Inputs<'_>| {
                let utc = instants(x, 0, ntz, &o)?;
                let zones = zones_of(x, 1, &zone)?;
                let zones = Zones::PerRow(&zones);
                let out = if from {
                    from_utc_timestamp(&utc, &zones)?
                } else {
                    to_utc_timestamp(&utc, &zones)?
                };
                Ok(Arc::new(out.with_timezone("UTC")) as ArrayRef)
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
            Binding::new(ntz_type(), move |x: &Inputs<'_>| {
                let clocks = wall_clocks(x, target_index + 1, instant, &o)?;
                let to = zones_of(x, target_index, &target)?;
                let from = match &source {
                    Some(arg) => zones_of(x, 0, arg)?,
                    None => Vec::new(),
                };
                let to = Zones::PerRow(&to);
                let out = if source.is_some() {
                    convert_timezone(&Zones::PerRow(&from), &to, &clocks)?
                } else {
                    convert_timezone(&Zones::One(o.time_zone.as_ref()), &to, &clocks)?
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
