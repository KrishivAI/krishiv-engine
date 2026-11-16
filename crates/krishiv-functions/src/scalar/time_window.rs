// SPDX-License-Identifier: Apache-2.0
//! `__time_windows(ts, window, slide, start)`: the windows of Spark's `window(time_column, ...)` that
//! contain each row's time, as `array<struct<start, end>>` (NULL for a NULL time).
//!
//! Spark's `TimeWindowing` (`TimeWindowResolution.buildTimeWindowRewrite`) computes, for
//! `i` in `0 until ceil(window / slide)`, `lastStart - i * slide` with
//! `lastStart = ts - (ts - start) mod slide` (a positive remainder) and keeps the windows
//! `[s, s + window)` that contain `ts`. The durations are microseconds, literals at bind time.

use std::sync::Arc;

use arrow_array::types::TimestampMicrosecondType;
use arrow_array::{Array, ArrayRef, PrimitiveArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use krishiv_common::error::KrishivError;

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["__time_windows", "__window_time"];

/// The long literal at argument `i`, if there is one.
fn literal(args: &[BindArg], i: usize) -> Option<i64> {
    let BindArg::Literal(a) = args.get(i)? else {
        return None;
    };
    let a = a.as_any().downcast_ref::<arrow_array::Int64Array>()?;
    a.is_valid(0).then(|| a.value(0))
}

/// Binds `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    if listed == "__window_time" {
        return Some((listed, bind_window_time(args)));
    }
    let ty = args.first().map(BindArg::data_type);
    let (Some(ty @ DataType::Timestamp(TimeUnit::Microsecond, _)), Some(w), Some(s), Some(t)) =
        (ty, literal(args, 1), literal(args, 2), literal(args, 3))
    else {
        return Some((listed, Err(bad_types(listed, args))));
    };
    if w <= 0 || s <= 0 || s > w || t.unsigned_abs() >= s.unsigned_abs() {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let fields: Fields = Fields::from(vec![
        Field::new("start", ty.clone(), true),
        Field::new("end", ty.clone(), true),
    ]);
    let element = Arc::new(Field::new(
        "element",
        DataType::Struct(fields.clone()),
        false,
    ));
    let ret = DataType::List(Arc::clone(&element));
    Some((
        listed,
        Ok(Binding::new(ret, move |x: &Inputs<'_>| {
            windows(x, &ty, &element, &fields, (w, s, t))
        })),
    ))
}

/// `window_time(window)`: the window's `end - 1` microsecond, NULL for a NULL window or end.
fn bind_window_time(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let Some(DataType::Struct(fields)) = args.first().map(BindArg::data_type) else {
        return Err(bad_types("window_time", args));
    };
    let Some(end) = fields.get(1) else {
        return Err(bad_types("window_time", args));
    };
    let ty = end.data_type().clone();
    if !matches!(ty, DataType::Timestamp(TimeUnit::Microsecond, _)) {
        return Err(bad_types("window_time", args));
    }
    let ret = ty.clone();
    Ok(Binding::new(ret, move |x: &Inputs<'_>| {
        let column = x.column(0)?;
        let window = column
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| internal("window_time of a column that is not a struct".to_owned()))?;
        let ends = window
            .column(1)
            .as_any()
            .downcast_ref::<PrimitiveArray<TimestampMicrosecondType>>()
            .ok_or_else(|| {
                internal("window_time of a window without a timestamp end".to_owned())
            })?;
        let mut out: PrimitiveArray<TimestampMicrosecondType> =
            ends.unary_opt(|v| v.checked_sub(1));
        if let Some(nulls) = window.nulls() {
            let merged = arrow_buffer::NullBuffer::union(out.nulls(), Some(nulls));
            out = PrimitiveArray::new(out.values().clone(), merged);
        }
        let out = match &ty {
            DataType::Timestamp(_, Some(tz)) => out.with_timezone(Arc::clone(tz)),
            _ => out,
        };
        Ok(Arc::new(out) as ArrayRef)
    }))
}

/// The windows of every row of argument 0.
fn windows(
    x: &Inputs<'_>,
    ty: &DataType,
    element: &Arc<Field>,
    fields: &Fields,
    (window, slide, start): (i64, i64, i64),
) -> Result<ArrayRef, KrishivError> {
    let ts: PrimitiveArray<TimestampMicrosecondType> = x.timestamps(0)?;
    let overlapping = window / slide + i64::from(window % slide != 0);
    // The windows of a batch are counted by an `i32` offset: refuse a batch that would pass it before
    // allocating (one row of a day-long window over a one microsecond slide is 86.4 billion).
    let valid_rows = i64::try_from(ts.len() - ts.null_count()).unwrap_or(i64::MAX);
    if overlapping.saturating_mul(valid_rows) > i64::from(i32::MAX) {
        return Err(internal(format!(
            "more than 2^31 windows in one batch: {valid_rows} rows of {overlapping} overlapping windows"
        )));
    }
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut offsets: Vec<i32> = Vec::with_capacity(ts.len() + 1);
    offsets.push(0);
    let mut valid = Vec::with_capacity(ts.len());
    for row in 0..ts.len() {
        if ts.is_null(row) {
            valid.push(false);
        } else {
            valid.push(true);
            let v = ts.value(row);
            let rem = v.wrapping_sub(start).rem_euclid(slide);
            let last = v.wrapping_sub(rem);
            for i in 0..overlapping {
                let Some(s) = i.checked_mul(slide).and_then(|d| last.checked_sub(d)) else {
                    continue;
                };
                let Some(e) = s.checked_add(window) else {
                    continue;
                };
                if s <= v && v < e {
                    starts.push(s);
                    ends.push(e);
                }
            }
        }
        offsets.push(
            i32::try_from(starts.len())
                .map_err(|_| internal("more than 2^31 windows in one batch".to_owned()))?,
        );
    }
    let carrier = |values: Vec<i64>| -> ArrayRef {
        let a = PrimitiveArray::<TimestampMicrosecondType>::from(values);
        match ty {
            DataType::Timestamp(_, Some(tz)) => Arc::new(a.with_timezone(Arc::clone(tz))),
            _ => Arc::new(a),
        }
    };
    let values = arrow_array::StructArray::try_new(
        fields.clone(),
        vec![carrier(starts), carrier(ends)],
        None,
    )
    .map_err(|e| internal(e.to_string()))?;
    let nulls = arrow_buffer::NullBuffer::from(valid);
    let list = arrow_array::ListArray::try_new(
        Arc::clone(element),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(values),
        Some(nulls),
    )
    .map_err(|e| internal(e.to_string()))?;
    Ok(Arc::new(list))
}
