// SPDX-License-Identifier: Apache-2.0
//! The interval constructors: `make_ym_interval`, `make_dt_interval`,
//! `make_interval` and `try_make_interval`. The analyzer fills a short call's
//! defaults, so each binds at its full arity.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::types::{Decimal128Type, Int32Type};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use krishiv_kernels::interval_make::{
    CalendarFields, DayTimeFields, make_calendar, make_day_time, make_year_month,
};

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "make_ym_interval",
    "make_dt_interval",
    "make_interval",
    "try_make_interval",
];

/// Whether `types` are `ints` integers, then a `DECIMAL(18,6)` if `seconds`.
fn takes(types: &[DataType], ints: usize, seconds: bool) -> bool {
    let (head, tail) = types.split_at(types.len().min(ints));
    head.len() == ints
        && head.iter().all(|t| *t == DataType::Int32)
        && match tail {
            [] => !seconds,
            [DataType::Decimal128(18, 6)] => seconds,
            _ => false,
        }
}

/// Binds the interval constructor `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match listed {
        "make_ym_interval" if takes(&types, 2, false) => Binding::new(
            DataType::Interval(IntervalUnit::YearMonth),
            |x: &Inputs<'_>| {
                let out = make_year_month(&x.array::<Int32Type>(0)?, &x.array::<Int32Type>(1)?)?;
                Ok(Arc::new(out) as ArrayRef)
            },
        ),
        "make_dt_interval" if takes(&types, 3, true) => Binding::new(
            DataType::Duration(TimeUnit::Microsecond),
            |x: &Inputs<'_>| {
                let out = make_day_time(DayTimeFields {
                    days: &x.array::<Int32Type>(0)?,
                    hours: &x.array::<Int32Type>(1)?,
                    minutes: &x.array::<Int32Type>(2)?,
                    seconds: &x.array::<Decimal128Type>(3)?,
                })?;
                Ok(Arc::new(out) as ArrayRef)
            },
        ),
        "make_interval" | "try_make_interval" if takes(&types, 6, true) => {
            let fail = listed == "make_interval" && opts.ansi;
            Binding::new(
                DataType::Interval(IntervalUnit::MonthDayNano),
                move |x: &Inputs<'_>| {
                    let out = make_calendar(
                        CalendarFields {
                            years: &x.array::<Int32Type>(0)?,
                            months: &x.array::<Int32Type>(1)?,
                            weeks: &x.array::<Int32Type>(2)?,
                            days: &x.array::<Int32Type>(3)?,
                            hours: &x.array::<Int32Type>(4)?,
                            minutes: &x.array::<Int32Type>(5)?,
                            seconds: &x.array::<Decimal128Type>(6)?,
                        },
                        fail,
                    )?;
                    Ok(Arc::new(out) as ArrayRef)
                },
            )
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
