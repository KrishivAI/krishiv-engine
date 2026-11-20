// SPDX-License-Identifier: Apache-2.0
//! `counter_diff(counter[, start_time])` (Spark 4.3, `CounterDiff` / `CounterDiffWithStartTime`): the increase of
//! a cumulative counter since the previous row of the partition that has a counter.
//!
//! The state is Spark's aggregation buffer: the last non-NULL counter and the start time that came with it
//! (which is NULL when that row's start time was). A row with a NULL counter is NULL and changes nothing. For
//! any other row, in Spark's order: a negative counter is an error; a start time that is lower than the previous
//! one is an error, a higher one a reset (NULL); a counter that is lower than the previous one is a reset (NULL);
//! else the difference, in the counter's own type (a decimal is not widened). The first row is NULL. A NULL
//! start time on either row skips the start time checks. Floats order as Spark orders them: `NaN` above every
//! value and equal to itself, so `NaN - x` and `NaN - NaN` are `NaN`, and `x - NaN` too.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, Decimal64Array, Decimal128Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{CounterDiffNegativeCounterValue, CounterDiffStartTimeDecreased};
use krishiv_kernels::cast::{CastMode, CastOptions, cast};
use krishiv_kernels::datetime::ZoneRef;

use crate::error::{Result, internal};

/// The name Spark's errors give the function.
const FUNCTION: &str = "`counter_diff`";

/// One counter value, in the arithmetic of its type.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Cell {
    /// An integral type or a decimal's unscaled value.
    Int(i128),
    F32(f32),
    F64(f64),
}

impl Cell {
    /// Whether the value is below zero (`NaN` is not).
    fn negative(self) -> bool {
        match self {
            Self::Int(v) => v < 0,
            Self::F32(v) => v < 0.0,
            Self::F64(v) => v < 0.0,
        }
    }

    /// Spark's `self < other`: `NaN` is the greatest value and equals itself.
    fn below(self, other: Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a < b,
            (Self::F32(a), Self::F32(b)) => !a.is_nan() && (b.is_nan() || a < b),
            (Self::F64(a), Self::F64(b)) => !a.is_nan() && (b.is_nan() || a < b),
            _ => false,
        }
    }

    fn minus(self, other: Self) -> Self {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Self::Int(a - b),
            (Self::F32(a), Self::F32(b)) => Self::F32(a - b),
            (Self::F64(a), Self::F64(b)) => Self::F64(a - b),
            (a, _) => a,
        }
    }
}

/// What a partition has seen so far.
#[derive(Clone, Debug, Default)]
pub(super) struct CounterState {
    /// The last non-NULL counter, and the start time of its row.
    prev: Option<(Cell, Option<i64>)>,
}

/// The counter column as cells.
fn cells(counter: &ArrayRef) -> Result<Vec<Option<Cell>>> {
    macro_rules! ints {
        ($t:ty) => {
            counter
                .as_primitive::<$t>()
                .iter()
                .map(|v| v.map(|v| Cell::Int(i128::from(v))))
                .collect()
        };
    }
    Ok(match counter.data_type() {
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::Decimal64(..) => ints!(Decimal64Type),
        DataType::Decimal128(..) => counter
            .as_primitive::<Decimal128Type>()
            .iter()
            .map(|v| v.map(Cell::Int))
            .collect(),
        DataType::Float32 => counter
            .as_primitive::<Float32Type>()
            .iter()
            .map(|v| v.map(Cell::F32))
            .collect(),
        DataType::Float64 => counter
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.map(Cell::F64))
            .collect(),
        other => return Err(internal(format!("counter_diff of {other}"))),
    })
}

/// The result column: `values` in the counter's type `t`.
fn column(t: &DataType, values: &[Option<Cell>]) -> Result<ArrayRef> {
    let int = |c: &Option<Cell>| match c {
        Some(Cell::Int(v)) => Some(*v),
        _ => None,
    };
    let narrow = |v: i128| internal(format!("counter_diff result {v} outside {t}"));
    macro_rules! ints {
        ($arr:ty, $prim:ty) => {{
            let out: Result<Vec<Option<$prim>>> = values
                .iter()
                .map(|c| {
                    int(c)
                        .map(|v| <$prim>::try_from(v).map_err(|_| narrow(v)))
                        .transpose()
                })
                .collect();
            Arc::new(<$arr>::from(out?)) as ArrayRef
        }};
    }
    Ok(match t {
        DataType::Int8 => ints!(Int8Array, i8),
        DataType::Int16 => ints!(Int16Array, i16),
        DataType::Int32 => ints!(Int32Array, i32),
        DataType::Int64 => ints!(Int64Array, i64),
        DataType::Decimal64(p, s) => Arc::new(
            ints_decimal64(values, int, &narrow)?
                .with_precision_and_scale(*p, *s)
                .map_err(|e| internal(e.to_string()))?,
        ),
        DataType::Decimal128(p, s) => Arc::new(
            Decimal128Array::from(values.iter().map(int).collect::<Vec<_>>())
                .with_precision_and_scale(*p, *s)
                .map_err(|e| internal(e.to_string()))?,
        ),
        DataType::Float32 => Arc::new(Float32Array::from(
            values
                .iter()
                .map(|c| match c {
                    Some(Cell::F32(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        DataType::Float64 => Arc::new(Float64Array::from(
            values
                .iter()
                .map(|c| match c {
                    Some(Cell::F64(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )),
        other => return Err(internal(format!("counter_diff of {other}"))),
    })
}

fn ints_decimal64(
    values: &[Option<Cell>],
    int: impl Fn(&Option<Cell>) -> Option<i128>,
    narrow: &dyn Fn(i128) -> krishiv_common::error::KrishivError,
) -> Result<Decimal64Array> {
    let out: Result<Vec<Option<i64>>> = values
        .iter()
        .map(|c| {
            int(c)
                .map(|v| i64::try_from(v).map_err(|_| narrow(v)))
                .transpose()
        })
        .collect();
    Ok(Decimal64Array::from(out?))
}

/// Row `i` of `a` as the text Spark's cast to `STRING` gives.
fn text(a: &ArrayRef, i: usize, zone: &ZoneRef) -> Result<String> {
    let one = a.slice(i, 1);
    let opts = CastOptions {
        mode: CastMode::Legacy,
        time_zone: zone,
    };
    let s = cast(one.as_ref(), &DataType::Utf8, &opts).map_err(|e| internal(e.to_string()))?;
    Ok(s.as_string::<i32>().value(0).to_owned())
}

/// The start times as microseconds.
fn starts(start: &ArrayRef) -> Result<Vec<Option<i64>>> {
    match start.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, _) => Ok(start
            .as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect()),
        other => Err(internal(format!("counter_diff start time of {other}"))),
    }
}

impl CounterState {
    /// The results for the next rows of the partition: `counter` and, with a start time, `start`.
    ///
    /// # Errors
    /// `COUNTER_DIFF_NEGATIVE_COUNTER_VALUE`, `COUNTER_DIFF_START_TIME_DECREASED`.
    pub(super) fn run(
        &mut self,
        counter: &ArrayRef,
        start: Option<&ArrayRef>,
        zone: &ZoneRef,
    ) -> Result<ArrayRef> {
        let values = cells(counter)?;
        let times = start.map(starts).transpose()?;
        let mut out = Vec::with_capacity(values.len());
        for (i, cell) in values.iter().enumerate() {
            let Some(cur) = *cell else {
                out.push(None);
                continue;
            };
            if cur.negative() {
                return Err(CounterDiffNegativeCounterValue {
                    function: FUNCTION.to_owned(),
                    value: text(counter, i, zone)?,
                }
                .into());
            }
            let cur_start = times.as_ref().and_then(|t| t[i]);
            let result = match self.prev {
                None => None,
                Some((prev, prev_start)) => {
                    let (lower, higher) = match (prev_start, cur_start) {
                        (Some(p), Some(c)) => (c < p, p < c),
                        _ => (false, false),
                    };
                    if lower {
                        let at = start
                            .ok_or_else(|| internal("a start time without a column".to_owned()))?;
                        let previous = prev_start.map(|p| render_start(at, p, zone)).transpose()?;
                        return Err(CounterDiffStartTimeDecreased {
                            function: FUNCTION.to_owned(),
                            previous_start_time: previous.unwrap_or_default(),
                            current_start_time: text(at, i, zone)?,
                        }
                        .into());
                    }
                    if higher || cur.below(prev) {
                        None
                    } else {
                        Some(cur.minus(prev))
                    }
                }
            };
            self.prev = Some((cur, cur_start));
            out.push(result);
        }
        column(counter.data_type(), &out)
    }
}

/// A start time of `microseconds` in the type of `like`, as text.
fn render_start(like: &ArrayRef, microseconds: i64, zone: &ZoneRef) -> Result<String> {
    let one: ArrayRef = Arc::new(
        arrow_array::TimestampMicrosecondArray::from(vec![microseconds])
            .with_data_type(like.data_type().clone()),
    );
    text(&one, 0, zone)
}

#[cfg(test)]
#[path = "counter_diff_tests.rs"]
mod tests;
