// SPDX-License-Identifier: Apache-2.0
//! Datetime functions (P2-T025a kernels): `date_add`/`date_sub`/`datediff`,
//! the field functions (`year`, `month`, `day`, `extract`/`date_part`) and
//! the query clock (`current_date`, `current_timestamp`).
//!
//! Day arithmetic raises `ARITHMETIC_OVERFLOW` under ANSI where the day
//! number leaves the `int` range and wraps like Java's `int` otherwise (ADR
//! 0023: Spark wraps under ANSI too).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Date32Array, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::KrishivError;
use krishiv_kernels::OverflowMode;
use krishiv_kernels::datetime::{
    DatePart, current_date, date_add, date_diff, date_sub, extract_date, extract_timestamp,
};

use super::args::{Inputs, is_string, is_timestamp, to_utf8, unsupported};
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
    "extract",
    "date_part",
    "current_date",
    "curdate",
    "current_timestamp",
    "now",
];

/// The Arrow type of `timestamp` results.
fn timestamp() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")))
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
        ("year" | "month" | "day" | "dayofmonth", [t])
            if *t == DataType::Date32 || is_timestamp(t) =>
        {
            let part = match listed {
                "year" => DatePart::Year,
                "month" => DatePart::Month,
                _ => DatePart::Day,
            };
            let source = t.clone();
            Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
                extract(x, 0, part, &source, &o)
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
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
