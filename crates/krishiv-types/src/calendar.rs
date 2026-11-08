// SPDX-License-Identifier: Apache-2.0
//! Calendar interval columns: the struct carrier of [`crate::arrow::calendar_struct`] (months, days,
//! microseconds, as Spark's `CalendarInterval`; ADR 0009 update 2026-11-06), read through [`Calendars`] and built
//! with [`calendar_array`], and the conversions to and from Arrow's `Interval(MonthDayNano)`, which only the
//! session edges use: [`to_month_day_nano`] multiplies the microseconds by 1,000 and reports the first row that
//! overflows (Spark's `ArrowWriter` overflows there with `Math.multiplyExact`), [`from_month_day_nano`] divides
//! toward zero (Spark's `ArrowColumnVector` reads `nanoseconds / 1000`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, Int32Array, Int64Array, IntervalMonthDayNanoArray, StructArray,
};
use arrow_schema::{DataType as ArrowType, Fields};

use crate::arrow::calendar_struct;

/// One calendar interval value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CalendarValue {
    /// Months.
    pub months: i32,
    /// Days.
    pub days: i32,
    /// Microseconds.
    pub micros: i64,
}

impl CalendarValue {
    /// `months, days, micros`.
    #[must_use]
    pub const fn new(months: i32, days: i32, micros: i64) -> Self {
        Self {
            months,
            days,
            micros,
        }
    }
}

/// A read view of a calendar interval column.
#[derive(Debug, Clone, Copy)]
pub struct Calendars<'a> {
    array: &'a StructArray,
    months: &'a Int32Array,
    days: &'a Int32Array,
    micros: &'a Int64Array,
}

impl<'a> Calendars<'a> {
    /// The view of `a`, when it is a calendar interval column.
    #[must_use]
    pub fn of(a: &'a dyn Array) -> Option<Self> {
        if !crate::arrow::is_calendar_struct(a.data_type()) {
            return None;
        }
        let array = a.as_struct_opt()?;
        Some(Self {
            array,
            months: array.column(0).as_primitive_opt::<Int32Type>()?,
            days: array.column(1).as_primitive_opt::<Int32Type>()?,
            micros: array.column(2).as_primitive_opt::<Int64Type>()?,
        })
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Whether row `i` is NULL.
    #[must_use]
    pub fn is_null(&self, i: usize) -> bool {
        self.array.is_null(i)
    }

    /// Row `i` (a NULL row is `None`).
    #[must_use]
    pub fn get(&self, i: usize) -> Option<CalendarValue> {
        self.array.is_valid(i).then(|| self.value(i))
    }

    /// Row `i`'s stored value, NULL or not.
    #[must_use]
    pub fn value(&self, i: usize) -> CalendarValue {
        CalendarValue::new(
            self.months.value(i),
            self.days.value(i),
            self.micros.value(i),
        )
    }

    /// Every row, in order.
    pub fn iter(&self) -> impl Iterator<Item = Option<CalendarValue>> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }
}

/// A calendar interval column of `values`.
#[must_use]
pub fn calendar_array(values: impl IntoIterator<Item = Option<CalendarValue>>) -> ArrayRef {
    let values: Vec<Option<CalendarValue>> = values.into_iter().collect();
    let pick = |f: fn(&CalendarValue) -> i64| values.iter().map(move |v| v.as_ref().map_or(0, f));
    let months: Int32Array = values
        .iter()
        .map(|v| v.map_or(0, |v| v.months))
        .collect::<Vec<_>>()
        .into();
    let days: Int32Array = values
        .iter()
        .map(|v| v.map_or(0, |v| v.days))
        .collect::<Vec<_>>()
        .into();
    let micros: Int64Array = pick(|v| v.micros).collect::<Vec<_>>().into();
    let nulls = values.iter().any(Option::is_none).then(|| {
        values
            .iter()
            .map(Option::is_some)
            .collect::<arrow_buffer::NullBuffer>()
    });
    let ArrowType::Struct(fields) = calendar_struct() else {
        return Arc::new(StructArray::new_empty_fields(values.len(), nulls));
    };
    Arc::new(StructArray::new(
        fields,
        vec![Arc::new(months), Arc::new(days), Arc::new(micros)],
        nulls,
    ))
}

/// A calendar interval column of `len` NULLs.
#[must_use]
pub fn null_calendar_array(len: usize) -> ArrayRef {
    calendar_array(std::iter::repeat_n(None, len))
}

/// `a` (a calendar interval column) as Arrow's `Interval(MonthDayNano)`, for the session edges.
///
/// # Errors
/// The first row whose microseconds overflow `i64` nanoseconds.
pub fn to_month_day_nano(a: &Calendars<'_>) -> Result<IntervalMonthDayNanoArray, usize> {
    let mut out = Vec::with_capacity(a.len());
    for (row, v) in a.iter().enumerate() {
        out.push(match v {
            None => None,
            Some(v) => Some(arrow_buffer::IntervalMonthDayNano::new(
                v.months,
                v.days,
                v.micros.checked_mul(1_000).ok_or(row)?,
            )),
        });
    }
    Ok(IntervalMonthDayNanoArray::from(out))
}

/// An Arrow `Interval(MonthDayNano)` column as a calendar interval column (nanoseconds divided toward zero).
#[must_use]
pub fn from_month_day_nano(a: &IntervalMonthDayNanoArray) -> ArrayRef {
    calendar_array(
        a.iter()
            .map(|v| v.map(|v| CalendarValue::new(v.months, v.days, v.nanoseconds / 1_000))),
    )
}

/// The fields of the carrier, for builders that need them.
#[must_use]
pub fn calendar_fields() -> Fields {
    match calendar_struct() {
        ArrowType::Struct(f) => f,
        _ => Fields::empty(),
    }
}

/// Builds a calendar interval column row by row.
#[derive(Debug, Default)]
pub struct CalendarBuilder {
    values: Vec<Option<CalendarValue>>,
}

impl CalendarBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder with room for `n` rows.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            values: Vec::with_capacity(n),
        }
    }

    /// Adds `v`.
    pub fn append_value(&mut self, v: CalendarValue) {
        self.values.push(Some(v));
    }

    /// Adds a NULL.
    pub fn append_null(&mut self) {
        self.values.push(None);
    }

    /// Adds `v`, NULL for `None`.
    pub fn append_option(&mut self, v: Option<CalendarValue>) {
        self.values.push(v);
    }

    /// The rows added so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether no row was added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The column, leaving the builder empty.
    pub fn finish(&mut self) -> ArrayRef {
        calendar_array(std::mem::take(&mut self.values))
    }
}
