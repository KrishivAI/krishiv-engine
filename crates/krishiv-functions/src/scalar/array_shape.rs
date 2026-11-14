// SPDX-License-Identifier: Apache-2.0
//! Array reshaping: `flatten`, `slice` and `array_join`.
//!
//! `flatten` is NULL for a NULL array or when any inner array is NULL.
//! `slice(arr, start, length)` counts `start` from 1, or from the end when
//! negative; a start before the first element or past the last gives an
//! empty array, `start = 0` and a negative length are errors. `array_join`
//! skips NULL elements unless a replacement is given, and is NULL when the
//! array, delimiter or replacement is NULL (Spark 4.2.0 run).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, StringArray};
use arrow_schema::{DataType, Field};
use krishiv_common::error::{
    CollectionSizeLimitExceededParameter, InvalidParameterValueLength, InvalidParameterValueStart,
    KrishivError, LegacyErrorTemp3242, LegacyErrorTemp3243,
};

use super::args::{Inputs, internal, to_utf8};
use super::array_edit::{MAX_ARRAY_LENGTH, index, list_of, rebuild, span, too_long};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["array_join", "flatten", "sequence", "slice"];

/// Binds the function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("flatten", [DataType::List(outer)]) => match outer.data_type() {
            DataType::List(inner) => {
                let field = Arc::new(Field::new(
                    inner.name(),
                    inner.data_type().clone(),
                    outer.is_nullable() || inner.is_nullable(),
                ));
                Ok(Binding::new(
                    DataType::List(Arc::clone(&field)),
                    move |x: &Inputs<'_>| flatten(x, &field),
                ))
            }
            _ => Err(bad_types(listed, args)),
        },
        ("sequence", [first, rest @ ..])
            if matches!(
                first,
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            ) && rest.len() <= 2
                && rest.iter().all(|t| t == first) =>
        {
            let element = first.clone();
            let field = Arc::new(Field::new("item", element.clone(), false));
            let out = DataType::List(Arc::clone(&field));
            let steps = types.len() > 2;
            Ok(Binding::new(out, move |x: &Inputs<'_>| {
                sequence(x, &element, &field, steps)
            }))
        }
        ("sequence", [first, second, rest @ ..])
            if temporal_bound(first) && first == second && rest.len() <= 1 =>
        {
            temporal_binding(first, rest.first(), opts)
        }
        ("slice", [t @ DataType::List(f), DataType::Int32, DataType::Int32]) => {
            let field = Arc::clone(f);
            Ok(Binding::new(t.clone(), move |x: &Inputs<'_>| {
                slice(x, &field)
            }))
        }
        ("array_join", [DataType::List(f), d, ..])
            if is_text(f.data_type()) && is_text(d) && types.len() <= 3 =>
        {
            let replaced = types.len() == 3;
            Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                join(x, replaced)
            }))
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

/// Whether `t` is a string type.
fn is_text(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

/// `flatten(arr)`.
fn flatten(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let outer = list_of(&c, "flatten")?;
    let inner = outer
        .values()
        .as_list_opt::<i32>()
        .ok_or_else(|| internal("flatten of a non-nested array".to_owned()))?;
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if outer.is_null(row) {
            rows.push(None);
            continue;
        }
        let (start, len) = span(outer, row);
        let mut picks: Option<Vec<Option<u32>>> = Some(Vec::new());
        for k in start..start + len {
            if inner.is_null(k) {
                picks = None;
                break;
            }
            let (first, n) = span(inner, k);
            if let Some(p) = picks.as_mut() {
                if p.len() + n > MAX_ARRAY_LENGTH {
                    return Err(too_long("flatten", p.len() + n));
                }
                for at in first..first + n {
                    p.push(Some(index(at)?));
                }
            }
        }
        rows.push(picks);
    }
    rebuild(field, inner.values(), &rows, "flatten", MAX_ARRAY_LENGTH)
}

/// The `INT` column `i`, one value per row.
fn ints(x: &Inputs<'_>, i: usize) -> Result<Int32Array, KrishivError> {
    Ok(x.column(i)?
        .as_primitive_opt::<arrow_array::types::Int32Type>()
        .ok_or_else(|| internal("slice with a non-INT bound".to_owned()))?
        .clone())
}

/// Column `i` of an integral type as `i64`.
fn longs(x: &Inputs<'_>, i: usize) -> Result<Vec<Option<i64>>, KrishivError> {
    use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
    let c = x.column(i)?;
    Ok(match c.data_type() {
        DataType::Int8 => c
            .as_primitive::<Int8Type>()
            .iter()
            .map(|v| v.map(i64::from))
            .collect(),
        DataType::Int16 => c
            .as_primitive::<Int16Type>()
            .iter()
            .map(|v| v.map(i64::from))
            .collect(),
        DataType::Int32 => c
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| v.map(i64::from))
            .collect(),
        DataType::Int64 => c.as_primitive::<Int64Type>().iter().collect(),
        DataType::Date32 => c
            .as_primitive::<arrow_array::types::Date32Type>()
            .iter()
            .map(|v| v.map(i64::from))
            .collect(),
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => c
            .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .iter()
            .collect(),
        t => return Err(internal(format!("sequence over {t}"))),
    })
}

/// `sequence(start, stop[, step])` over `element`: the step defaults to 1,
/// or -1 when `stop` is below `start`; a step that cannot reach `stop` is an
/// error (Spark's `Illegal sequence boundaries`).
fn sequence(
    x: &Inputs<'_>,
    element: &DataType,
    field: &Arc<Field>,
    has_step: bool,
) -> Result<ArrayRef, KrishivError> {
    use arrow_array::builder::{Int8Builder, Int16Builder, Int32Builder, Int64Builder};
    let (starts, stops) = (longs(x, 0)?, longs(x, 1)?);
    let steps = if has_step { Some(longs(x, 2)?) } else { None };
    let mut offsets: Vec<i32> = vec![0];
    let mut valid = Vec::with_capacity(x.rows());
    let mut values: Vec<i64> = Vec::new();
    for row in 0..x.rows() {
        let step_cell = steps.as_ref().map(|s| s[row]);
        let (Some(start), Some(stop), true) = (starts[row], stops[row], step_cell != Some(None))
        else {
            valid.push(false);
            offsets.push(offsets[row]);
            continue;
        };
        let step = step_cell
            .flatten()
            .unwrap_or(if start <= stop { 1 } else { -1 });
        let ok = (step > 0 && start <= stop)
            || (step < 0 && start >= stop)
            || (step == 0 && start == stop);
        if !ok {
            return Err(LegacyErrorTemp3243 {
                start: start.to_string(),
                stop: stop.to_string(),
                step: step.to_string(),
            }
            .into());
        }
        let len: i128 = if step == 0 || start == stop {
            1
        } else {
            1 + (i128::from(stop) - i128::from(start)) / i128::from(step)
        };
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        if len > MAX_ARRAY_LENGTH || values.len() + len > MAX_ARRAY_LENGTH {
            return Err(CollectionSizeLimitExceededParameter {
                function_name: "`sequence`".to_owned(),
                parameter: "`count`".to_owned(),
                number_of_elements: len.to_string(),
                max_rounded_array_length: MAX_ARRAY_LENGTH.to_string(),
            }
            .into());
        }
        let mut at = start;
        for _ in 0..len {
            values.push(at);
            at = at.wrapping_add(step);
        }
        valid.push(true);
        offsets.push(i32::try_from(values.len()).map_err(|_| too_long("sequence", values.len()))?);
    }
    let narrow = |v: i64| internal(format!("sequence element {v} outside {element}"));
    let child: ArrayRef = match element {
        DataType::Int8 => {
            let mut b = Int8Builder::with_capacity(values.len());
            for &v in &values {
                b.append_value(i8::try_from(v).map_err(|_| narrow(v))?);
            }
            Arc::new(b.finish())
        }
        DataType::Int16 => {
            let mut b = Int16Builder::with_capacity(values.len());
            for &v in &values {
                b.append_value(i16::try_from(v).map_err(|_| narrow(v))?);
            }
            Arc::new(b.finish())
        }
        DataType::Int32 => {
            let mut b = Int32Builder::with_capacity(values.len());
            for &v in &values {
                b.append_value(i32::try_from(v).map_err(|_| narrow(v))?);
            }
            Arc::new(b.finish())
        }
        _ => {
            let mut b = Int64Builder::with_capacity(values.len());
            for &v in &values {
                b.append_value(v);
            }
            Arc::new(b.finish())
        }
    };
    let list = arrow_array::ListArray::try_new(
        Arc::clone(field),
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        child,
        Some(arrow_buffer::NullBuffer::from(valid)),
    )
    .map_err(|e| internal(e.to_string()))?;
    Ok(Arc::new(list))
}

/// `reverse(array)`: the elements in the opposite order, NULL elements kept in place, NULL for a NULL array (the string and
/// binary forms are in `chars`).
pub(super) fn reverse_binding(t: &DataType) -> Option<Binding> {
    let DataType::List(f) = t else {
        return None;
    };
    let field = Arc::clone(f);
    Some(Binding::new(t.clone(), move |x: &Inputs<'_>| {
        reverse(x, &field)
    }))
}

fn reverse(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "reverse")?;
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) {
            rows.push(None);
            continue;
        }
        let (first, size) = span(list, row);
        let mut picks = Vec::with_capacity(size);
        for at in (first..first + size).rev() {
            picks.push(Some(index(at)?));
        }
        rows.push(Some(picks));
    }
    rebuild(field, list.values(), &rows, "reverse", MAX_ARRAY_LENGTH)
}

/// `slice(arr, start, length)`.
fn slice(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "slice")?;
    let (starts, lengths) = (ints(x, 1)?, ints(x, 2)?);
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) || starts.is_null(row) || lengths.is_null(row) {
            rows.push(None);
            continue;
        }
        let (start, length) = (starts.value(row), lengths.value(row));
        if start == 0 {
            return Err(InvalidParameterValueStart {
                function_name: "`slice`".to_owned(),
                parameter: "`start`".to_owned(),
            }
            .into());
        }
        if length < 0 {
            return Err(InvalidParameterValueLength {
                function_name: "`slice`".to_owned(),
                parameter: "`length`".to_owned(),
                length: length.to_string(),
            }
            .into());
        }
        let (first, size) = span(list, row);
        let size_i = i64::try_from(size).map_err(|_| internal("array past 2^63".to_owned()))?;
        let from = if start > 0 {
            i64::from(start) - 1
        } else {
            size_i + i64::from(start)
        };
        if from < 0 || from >= size_i {
            rows.push(Some(Vec::new()));
            continue;
        }
        let to = size_i.min(from + i64::from(length));
        let (from, to) = (
            usize::try_from(from).unwrap_or(0),
            usize::try_from(to).unwrap_or(0),
        );
        let mut picks = Vec::with_capacity(to - from);
        for at in first + from..first + to {
            picks.push(Some(index(at)?));
        }
        rows.push(Some(picks));
    }
    rebuild(field, list.values(), &rows, "slice", MAX_ARRAY_LENGTH)
}

/// `array_join(arr, delimiter[, replacement])`.
fn join(x: &Inputs<'_>, replaced: bool) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "array_join")?;
    let elements = to_utf8(list.values().as_ref())?;
    let delimiters = x.strings(1)?;
    let replacements: Option<StringArray> = replaced.then(|| x.strings(2)).transpose()?;
    let mut out: Vec<Option<String>> = Vec::with_capacity(x.rows());
    let mut total = 0usize;
    for row in 0..x.rows() {
        let replacement = replacements
            .as_ref()
            .map(|r| (r.is_valid(row)).then(|| r.value(row)));
        if list.is_null(row) || delimiters.is_null(row) || replacement == Some(None) {
            out.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let delimiter = delimiters.value(row);
        let mut text = String::new();
        let mut first = true;
        for at in start..start + len {
            let piece = if elements.is_valid(at) {
                elements.value(at)
            } else if let Some(Some(r)) = replacement {
                r
            } else {
                continue;
            };
            if !first {
                text.push_str(delimiter);
            }
            first = false;
            text.push_str(piece);
            if i32::try_from(text.len()).is_err() {
                return Err(too_long("array_join", text.len()));
            }
        }
        total = total.saturating_add(text.len());
        if i32::try_from(total).is_err() {
            return Err(too_long("array_join", total));
        }
        out.push(Some(text));
    }
    Ok(Arc::new(StringArray::from(out)))
}

/// Whether `t` is a bound `sequence` takes over time: a date or a (zoned or zone-less) microsecond timestamp.
fn temporal_bound(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Date32 | DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _)
    )
}

/// How a step is read: its `(months, days, micros)` per row.
#[derive(Clone, Copy)]
enum StepKind {
    YearMonth,
    DayTime,
    Calendar,
}

fn temporal_binding(
    bound: &DataType,
    step: Option<&DataType>,
    opts: &BindOptions,
) -> Result<Binding, KrishivError> {
    use arrow_schema::{IntervalUnit, TimeUnit};
    let kind = match step {
        None => None,
        Some(DataType::Interval(IntervalUnit::YearMonth)) => Some(StepKind::YearMonth),
        Some(DataType::Duration(TimeUnit::Microsecond)) => Some(StepKind::DayTime),
        Some(t) if krishiv_types::arrow::is_calendar_struct(t) => Some(StepKind::Calendar),
        Some(_) => {
            return Err(internal(
                "a sequence step that is not an interval".to_owned(),
            ));
        }
    };
    let element = bound.clone();
    let field = Arc::new(Field::new("item", element.clone(), false));
    let zone = Arc::clone(&opts.time_zone);
    Ok(Binding::new(
        DataType::List(Arc::clone(&field)),
        move |x: &Inputs<'_>| temporal_sequence(x, &element, &field, kind, zone.as_ref()),
    ))
}

/// A step as `(months, days, micros)`.
type Step = (i64, i64, i64);

/// Every row's step as `(months, days, micros)` (`None` for a NULL step): a year-month interval is months; a day-time one splits into
/// whole days and the remainder; a calendar interval is its three fields. Read once per batch.
fn read_steps(x: &Inputs<'_>, kind: StepKind) -> Result<Vec<Option<Step>>, KrishivError> {
    use arrow_array::types::{DurationMicrosecondType, IntervalYearMonthType};
    let col = x.column(2)?;
    Ok(match kind {
        StepKind::YearMonth => col
            .as_primitive::<IntervalYearMonthType>()
            .iter()
            .map(|v| v.map(|m| (i64::from(m), 0, 0)))
            .collect(),
        StepKind::DayTime => col
            .as_primitive::<DurationMicrosecondType>()
            .iter()
            .map(|v| v.map(|m| (0, m / MICROS_PER_DAY, m % MICROS_PER_DAY)))
            .collect(),
        StepKind::Calendar => {
            let c = krishiv_types::calendar::Calendars::of(col.as_ref()).ok_or_else(|| {
                internal("a sequence step that is not a calendar interval".to_owned())
            })?;
            c.iter()
                .map(|v| v.map(|v| (i64::from(v.months), i64::from(v.days), v.micros)))
                .collect()
        }
    })
}

const MICROS_PER_DAY: i64 = 86_400_000_000;

/// `sequence(start, stop[, step])` over dates or timestamps (Spark 4.2.0, goldens `functions/sequence_xml`): element `i` is `start`
/// plus `i * step` (months, then calendar days on the local clock, then elapsed microseconds), taken from the start each time,
/// while it has not passed `stop`; a missing step is one day towards `stop`.
fn temporal_sequence(
    x: &Inputs<'_>,
    element: &DataType,
    field: &Arc<Field>,
    kind: Option<StepKind>,
    zone: &dyn krishiv_kernels::datetime::TimeZoneRules,
) -> Result<ArrayRef, KrishivError> {
    use krishiv_kernels::datetime::{
        add_day_time_to_timestamp, add_months_to_date, add_months_to_timestamp,
    };
    let is_date = matches!(element, DataType::Date32);
    let tz: Option<&dyn krishiv_kernels::datetime::TimeZoneRules> = match element {
        DataType::Timestamp(_, Some(_)) => Some(zone),
        _ => None,
    };
    let (starts, stops) = (longs(x, 0)?, longs(x, 1)?);
    let steps = kind.map(|k| read_steps(x, k)).transpose()?;
    let mut offsets: Vec<i32> = vec![0];
    let mut valid = Vec::with_capacity(x.rows());
    let mut values: Vec<i64> = Vec::new();
    for row in 0..x.rows() {
        let step = match &steps {
            Some(all) => all[row],
            None => Some((0, 0, 0)),
        };
        let (Some(start), Some(stop), Some(step)) = (starts[row], stops[row], step) else {
            valid.push(false);
            offsets.push(offsets[row]);
            continue;
        };
        let (mut months, mut days, micros) = step;
        if kind.is_none() {
            days = if start <= stop { 1 } else { -1 };
            months = 0;
        }
        let type_name = match kind {
            Some(StepKind::YearMonth) => "year to month",
            Some(StepKind::DayTime) => "day to second",
            _ => "interval",
        };
        if is_date && months == 0 && days == 0 {
            return Err(LegacyErrorTemp3242 {
                interval_type: format!("interval {type_name}")
                    .replace("interval interval", "interval"),
            }
            .into());
        }
        // The direction is the one from start to stop; the step must move that way, its sign being that of its first non-zero field
        // (months, then days, then microseconds), except that equal bounds give one element for any step a timestamp takes.
        let sign = if months != 0 {
            months.signum()
        } else if days != 0 {
            days.signum()
        } else {
            micros.signum()
        };
        let up = start <= stop;
        if !(start == stop || (up && sign > 0) || (!up && sign < 0)) {
            return Err(LegacyErrorTemp3243 {
                start: start.to_string(),
                stop: stop.to_string(),
                step: if months != 0 && days == 0 && micros == 0 {
                    months.to_string()
                } else {
                    (i128::from(days) * i128::from(MICROS_PER_DAY) + i128::from(micros)).to_string()
                },
            }
            .into());
        }
        // A step of days and microseconds only has its length up front, so a huge result fails before any element is built
        // (`1 microsecond` over a year is `numberOfElements` 31449600000001); a month step cannot reach the limit.
        if months == 0 {
            let step = i128::from(days) * i128::from(MICROS_PER_DAY) + i128::from(micros);
            let unit = if is_date {
                i128::from(MICROS_PER_DAY)
            } else {
                1
            };
            let span = (i128::from(stop) - i128::from(start)) * unit;
            let count = if step == 0 { 1 } else { span / step + 1 };
            if count > i128::try_from(MAX_ARRAY_LENGTH).unwrap_or(i128::MAX) {
                return Err(CollectionSizeLimitExceededParameter {
                    function_name: "`sequence`".to_owned(),
                    parameter: "`count`".to_owned(),
                    number_of_elements: count.to_string(),
                    max_rounded_array_length: MAX_ARRAY_LENGTH.to_string(),
                }
                .into());
            }
        }
        // Whole-day steps over dates have their length from the micros (`(stop - start) / step + 1`), as Spark computes it before
        // taking the elements: `1 day 12 hours` from Jan 1 to Jan 5 gives Jan 1, Jan 2, Jan 4.
        let date_count = (is_date && months == 0).then(|| {
            let step = i128::from(days) * i128::from(MICROS_PER_DAY) + i128::from(micros);
            let span = (i128::from(stop) - i128::from(start)) * i128::from(MICROS_PER_DAY);
            if step == 0 { 1 } else { span / step + 1 }
        });
        let first = values.len();
        let overflow = |what: &str| internal(format!("sequence {what} out of range"));
        let mut i: i64 = 0;
        loop {
            let at = if is_date {
                let moved = add_months_to_date(
                    i32::try_from(start).map_err(|_| overflow("date"))?,
                    months.checked_mul(i).ok_or_else(|| overflow("months"))?,
                )
                .ok_or_else(|| overflow("date"))?;
                // A date's step is whole days: the remainder of `i * micros` below a day is dropped (`1 day 12 hours` gives
                // Jan 1, Jan 2, Jan 4).
                let extra =
                    micros.checked_mul(i).ok_or_else(|| overflow("micros"))? / MICROS_PER_DAY;
                i64::from(moved)
                    .checked_add(days.checked_mul(i).ok_or_else(|| overflow("days"))?)
                    .and_then(|d| d.checked_add(extra))
                    .ok_or_else(|| overflow("date"))?
            } else {
                let by_months = add_months_to_timestamp(
                    start,
                    months.checked_mul(i).ok_or_else(|| overflow("months"))?,
                    tz,
                )
                .ok_or_else(|| overflow("timestamp"))?;
                let by_days = add_day_time_to_timestamp(
                    by_months,
                    days.checked_mul(i)
                        .and_then(|d| d.checked_mul(MICROS_PER_DAY))
                        .ok_or_else(|| overflow("days"))?,
                    tz,
                )
                .ok_or_else(|| overflow("timestamp"))?;
                by_days
                    .checked_add(micros.checked_mul(i).ok_or_else(|| overflow("micros"))?)
                    .ok_or_else(|| overflow("timestamp"))?
            };
            let past = if up { at > stop } else { at < stop };
            if past || date_count.is_some_and(|n| i128::from(i) >= n) {
                break;
            }
            if start == stop && i > 0 {
                break;
            }
            values.push(at);
            if values.len() - first > MAX_ARRAY_LENGTH {
                return Err(CollectionSizeLimitExceededParameter {
                    function_name: "`sequence`".to_owned(),
                    parameter: "`count`".to_owned(),
                    number_of_elements: (values.len() - first).to_string(),
                    max_rounded_array_length: MAX_ARRAY_LENGTH.to_string(),
                }
                .into());
            }
            i += 1;
        }
        valid.push(true);
        offsets.push(i32::try_from(values.len()).map_err(|_| too_long("sequence", values.len()))?);
    }
    let child: ArrayRef = match element {
        DataType::Date32 => Arc::new(arrow_array::Date32Array::from(
            values
                .iter()
                .map(|v| {
                    i32::try_from(*v)
                        .map_err(|_| internal("a sequence date out of range".to_owned()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        DataType::Timestamp(_, tzname) => Arc::new(
            arrow_array::TimestampMicrosecondArray::from(values).with_timezone_opt(tzname.clone()),
        ),
        other => return Err(internal(format!("a sequence over {other}"))),
    };
    let list = arrow_array::ListArray::try_new(
        Arc::clone(field),
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        child,
        Some(arrow_buffer::NullBuffer::from(valid)),
    )
    .map_err(|e| internal(e.to_string()))?;
    Ok(Arc::new(list))
}
