// SPDX-License-Identifier: Apache-2.0
//! Per-group accumulators, one vector per aggregate (column-oriented), with
//! Spark 4.2 arithmetic: `sum` of integers in `bigint` (ANSI raises, legacy
//! wraps), decimals in `i128`, `avg` of decimals by Spark's division
//! (quotient at scale 39, then rounded half-up to the result scale).
//!
//! Decimal sums overflow as soon as a running sum leaves the sum type's
//! precision, even if later rows bring it back (`ARITHMETIC_OVERFLOW
//! "Overflow in sum of decimals"` under ANSI, NULL otherwise), grouped or
//! not, so the result never depends on the plan. Spark 4.2 keeps grouped
//! and partial buffers in that precision too; only its single-partition
//! global aggregate tolerates a temporary overflow (a documented divergence,
//! ADR 0023).

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Decimal128Array, Float64Array,
    Int64Array, PrimitiveArray, StringArray, StringViewArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_types::batch::Weights;

use super::{
    AggInput, AggregateExpr, AggregateFunction, decimal_sum_overflow, long_overflow, out_of_range,
    unsupported,
};
use crate::error::{Result, internal};

/// Whether `min`/`max` take arguments of type `t`.
pub(crate) fn min_max_supported(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(..)
            | DataType::Date32
            | DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Utf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::BinaryView
    )
}

/// `min`/`max` values, by physical kind.
#[derive(Debug)]
enum Values {
    /// Integral, boolean, decimal, date and timestamp values.
    Int(Vec<i128>),
    Float(Vec<f64>),
    Bytes(Vec<Box<[u8]>>),
}

#[derive(Debug)]
enum Kind {
    Count(Vec<i64>),
    SumInt {
        sum: Vec<i64>,
        seen: Vec<bool>,
    },
    SumFloat {
        sum: Vec<f64>,
        seen: Vec<bool>,
    },
    /// `state`: 0 no value, 1 a value, 2 overflowed.
    SumDec {
        sum: Vec<i128>,
        state: Vec<u8>,
    },
    AvgFloat {
        sum: Vec<f64>,
        count: Vec<i64>,
    },
    AvgDec {
        sum: Vec<i128>,
        count: Vec<i64>,
        overflow: Vec<bool>,
    },
    MinMax {
        max: bool,
        values: Values,
        set: Vec<bool>,
        /// Heap bytes of `Values::Bytes`.
        heap: usize,
    },
}

/// One aggregate's state for every group of a table.
#[derive(Debug)]
pub(crate) struct Accumulator {
    kind: Kind,
    expr: AggregateExpr,
    result: DataType,
    ansi: bool,
    /// The bound of a running decimal sum's magnitude (`10^p` of the sum
    /// type); `None` for other aggregates.
    sum_bound: Option<u128>,
}

/// The weight of row `i`.
fn weight(w: &Weights, i: usize) -> i64 {
    match w {
        Weights::Const(c) => *c,
        Weights::Column(c) => c.value(i),
    }
}

fn is_unit(w: &Weights) -> bool {
    match w {
        Weights::Const(c) => *c == 1,
        Weights::Column(c) => c.values().iter().all(|&v| v == 1),
    }
}

fn arg(cols: &[ArrayRef], i: usize) -> Result<&ArrayRef> {
    cols.get(i)
        .ok_or_else(|| internal(format!("aggregate argument {i} out of range")))
}

fn ints(a: &ArrayRef) -> Result<Cow<'_, [i64]>> {
    Ok(match a.data_type() {
        DataType::Int64 => Cow::Borrowed(a.as_primitive::<Int64Type>().values()),
        DataType::Int32 => widen(a.as_primitive::<Int32Type>()),
        DataType::Int16 => widen(a.as_primitive::<Int16Type>()),
        DataType::Int8 => widen(a.as_primitive::<Int8Type>()),
        t => return Err(internal(format!("integral argument of type {t}"))),
    })
}

fn widen<T>(a: &PrimitiveArray<T>) -> Cow<'static, [i64]>
where
    T: arrow_array::ArrowPrimitiveType,
    T::Native: Into<i64>,
{
    Cow::Owned(a.values().iter().map(|&v| v.into()).collect())
}

fn floats(a: &ArrayRef) -> Result<Cow<'_, [f64]>> {
    Ok(match a.data_type() {
        DataType::Float64 => Cow::Borrowed(a.as_primitive::<Float64Type>().values()),
        DataType::Float32 => Cow::Owned(
            a.as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|&v| f64::from(v))
                .collect(),
        ),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "Spark's avg of integers sums in double"
            )]
            let v = ints(a)?.iter().map(|&v| v as f64).collect();
            Cow::Owned(v)
        }
        t => return Err(internal(format!("floating argument of type {t}"))),
    })
}

/// A `bigint` state column (counts).
fn counts_of(a: &ArrayRef) -> Result<&[i64]> {
    a.as_primitive_opt::<Int64Type>()
        .map(|c| c.values().as_ref())
        .ok_or_else(|| internal(format!("count state of type {}", a.data_type())))
}

/// A `boolean` state column.
fn booleans_of(a: &ArrayRef) -> Result<&BooleanArray> {
    a.as_boolean_opt()
        .ok_or_else(|| internal(format!("boolean state of type {}", a.data_type())))
}

fn decimals(a: &ArrayRef) -> Result<&[i128]> {
    a.as_primitive_opt::<Decimal128Type>()
        .map(|d| d.values().as_ref())
        .ok_or_else(|| internal(format!("decimal argument of type {}", a.data_type())))
}

/// `min`/`max` input as `i128`s (integral, boolean, decimal, temporal).
fn int_like(a: &ArrayRef) -> Result<Cow<'_, [i128]>> {
    let conv = |v: Vec<i128>| Cow::Owned(v);
    Ok(match a.data_type() {
        DataType::Decimal128(..) => Cow::Borrowed(decimals(a)?),
        DataType::Boolean => {
            let b = a.as_boolean();
            conv((0..b.len()).map(|i| i128::from(b.value(i))).collect())
        }
        DataType::Date32 => conv(
            a.as_primitive::<Date32Type>()
                .values()
                .iter()
                .map(|&v| i128::from(v))
                .collect(),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => conv(
            a.as_primitive::<TimestampMicrosecondType>()
                .values()
                .iter()
                .map(|&v| i128::from(v))
                .collect(),
        ),
        _ => conv(ints(a)?.iter().map(|&v| i128::from(v)).collect()),
    })
}

/// Byte values of a string or binary array (empty under NULL slots).
fn byte_values(a: &ArrayRef) -> Result<Vec<&[u8]>> {
    let n = a.len();
    Ok(match a.data_type() {
        DataType::Utf8View => {
            let s = a.as_string_view();
            (0..n).map(|i| s.value(i).as_bytes()).collect()
        }
        DataType::Utf8 => {
            let s = a.as_string::<i32>();
            (0..n).map(|i| s.value(i).as_bytes()).collect()
        }
        DataType::BinaryView => {
            let s = a.as_binary_view();
            (0..n).map(|i| s.value(i)).collect()
        }
        DataType::Binary => {
            let s = a.as_binary::<i32>();
            (0..n).map(|i| s.value(i)).collect()
        }
        t => return Err(internal(format!("string argument of type {t}"))),
    })
}

/// An upper bound of the bytes `min`/`max` may keep from `a`: the lengths
/// of its values (not the buffers it shares with other arrays).
fn value_bytes(a: &ArrayRef) -> usize {
    let span = |o: &[i32]| match (o.first(), o.last()) {
        (Some(&s), Some(&e)) => usize::try_from(e.saturating_sub(s)).unwrap_or(0),
        _ => 0,
    };
    // The low 32 bits of a view are the value's length.
    let views = |v: &[u128]| -> usize {
        v.iter()
            .map(|v| usize::try_from(v & 0xFFFF_FFFF).unwrap_or(0))
            .sum()
    };
    match a.data_type() {
        DataType::Utf8 => span(a.as_string::<i32>().value_offsets()),
        DataType::Binary => span(a.as_binary::<i32>().value_offsets()),
        DataType::Utf8View => views(a.as_string_view().views()),
        DataType::BinaryView => views(a.as_binary_view().views()),
        _ => a.get_buffer_memory_size(),
    }
}

/// Spark ordering of doubles: NaN is the largest and equals itself; `-0.0`
/// equals `0.0`.
fn float_lt(a: f64, b: f64) -> bool {
    !a.is_nan() && (b.is_nan() || a < b)
}

/// `10^p` for `p ≤ 38`.
fn pow10(p: u8) -> i128 {
    10i128.pow(u32::from(p))
}

/// Spark's `toPlainString` of an unscaled decimal.
pub(crate) fn plain(v: i128, scale: i8) -> String {
    let s = usize::try_from(scale).unwrap_or(0);
    plain_digits(&v.unsigned_abs().to_string(), s, v < 0)
}

/// `digits` (an unscaled magnitude) with `scale` digits after the point.
fn plain_digits(digits: &str, scale: usize, negative: bool) -> String {
    let sign = if negative { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = padded.split_at(padded.len() - scale);
    format!("{sign}{int}.{frac}")
}

/// Spark's decimal `avg`: `sum / count` as `Decimal./` (scale 39, half-up)
/// then `toPrecision(p, s)` (half-up). `Ok(None)` when the value does not
/// fit `decimal(p, s)`, with the scale-39 quotient's text for the error.
fn avg_decimal(sum: i128, sum_scale: i8, count: i64, p: u8, s: i8) -> (Option<i128>, String) {
    let negative = (sum < 0) != (count < 0);
    let (a, b) = (sum.unsigned_abs(), u128::from(count.unsigned_abs()));
    let ss = usize::try_from(sum_scale).unwrap_or(0);
    let mut int = a / b;
    let mut rem = a % b;
    // `b < 2^63`, so `rem * 10 < 2^67` never overflows.
    let mut digit = || {
        rem *= 10;
        let d = u8::try_from(rem / b).unwrap_or(9);
        rem %= b;
        d
    };
    // Digits after the sum's scale up to scale 39, and the rounding digit.
    let mut frac: Vec<u8> = (0..39 - ss).map(|_| digit()).collect();
    if digit() >= 5 && round_up(&mut frac) {
        int += 1;
    }
    let mut digits = int.to_string();
    digits.extend(frac.iter().map(|d| char::from(b'0' + d)));
    let text = plain_digits(&digits, 39, negative && digits.bytes().any(|d| d != b'0'));
    let keep = usize::try_from(s).unwrap_or(0).saturating_sub(ss);
    let up = frac.get(keep).is_some_and(|&d| d >= 5);
    let mut v = Some(int);
    for &d in frac.iter().take(keep) {
        v = v
            .and_then(|v| v.checked_mul(10))
            .and_then(|v| v.checked_add(u128::from(d)));
    }
    let v = v
        .and_then(|v| v.checked_add(u128::from(up)))
        .and_then(|v| i128::try_from(v).ok())
        .filter(|&v| v < pow10(p))
        .map(|v| if negative { -v } else { v });
    (v, text)
}

/// Adds one unit in the last place of `digits`; `true` if it carries out.
fn round_up(digits: &mut [u8]) -> bool {
    for d in digits.iter_mut().rev() {
        if *d == 9 {
            *d = 0;
        } else {
            *d += 1;
            return false;
        }
    }
    true
}

impl Accumulator {
    /// An accumulator for `expr` (decimal sums are bounded while running,
    /// see the module docs).
    pub(crate) fn new(expr: &AggregateExpr, ansi: bool) -> Result<Self> {
        use AggregateFunction as F;
        let result = expr.result_type()?;
        let kind = match (expr.function, &result) {
            (F::Count, _) => Kind::Count(Vec::new()),
            (F::Sum, DataType::Int64) => Kind::SumInt {
                sum: Vec::new(),
                seen: Vec::new(),
            },
            (F::Sum, DataType::Float64) => Kind::SumFloat {
                sum: Vec::new(),
                seen: Vec::new(),
            },
            (F::Sum, _) => Kind::SumDec {
                sum: Vec::new(),
                state: Vec::new(),
            },
            (F::Avg, DataType::Float64) => Kind::AvgFloat {
                sum: Vec::new(),
                count: Vec::new(),
            },
            (F::Avg, _) => Kind::AvgDec {
                sum: Vec::new(),
                count: Vec::new(),
                overflow: Vec::new(),
            },
            (f @ (F::Min | F::Max), t) => Kind::MinMax {
                max: f == F::Max,
                values: match t {
                    DataType::Float32 | DataType::Float64 => Values::Float(Vec::new()),
                    DataType::Utf8
                    | DataType::Utf8View
                    | DataType::Binary
                    | DataType::BinaryView => Values::Bytes(Vec::new()),
                    _ => Values::Int(Vec::new()),
                },
                set: Vec::new(),
                heap: 0,
            },
        };
        let sum_bound = match (expr.function, expr.avg_sum_type()) {
            (F::Sum | F::Avg, DataType::Decimal128(p, _)) => Some(pow10(p).unsigned_abs()),
            _ => None,
        };
        Ok(Self {
            kind,
            expr: expr.clone(),
            result,
            ansi,
            sum_bound,
        })
    }

    /// Visits every per-group vector as (length, capacity, element size).
    fn vecs(&self) -> Vec<(usize, usize, usize)> {
        fn v<T>(x: &Vec<T>) -> (usize, usize, usize) {
            (x.len(), x.capacity(), size_of::<T>())
        }
        match &self.kind {
            Kind::Count(c) => vec![v(c)],
            Kind::SumInt { sum, seen } => vec![v(sum), v(seen)],
            Kind::SumFloat { sum, seen } => vec![v(sum), v(seen)],
            Kind::SumDec { sum, state } => vec![v(sum), v(state)],
            Kind::AvgFloat { sum, count } => vec![v(sum), v(count)],
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => vec![v(sum), v(count), v(overflow)],
            Kind::MinMax { values, set, .. } => vec![
                match values {
                    Values::Int(x) => v(x),
                    Values::Float(x) => v(x),
                    Values::Bytes(x) => v(x),
                },
                v(set),
            ],
        }
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        let heap = match &self.kind {
            Kind::MinMax { heap, .. } => *heap,
            _ => 0,
        };
        heap + self.vecs().iter().map(|(_, c, e)| c * e).sum::<usize>()
    }

    /// [`size`](Self::size) after [`resize`](Self::resize) to `new` more
    /// groups and an update with `input` (string values it may keep).
    /// `arg` is the value column the update reads (raw or state).
    pub(crate) fn projected(&self, new: usize, arg: Option<&ArrayRef>) -> usize {
        let vecs: usize = self
            .vecs()
            .iter()
            .map(|&(len, cap, e)| {
                let need = len.saturating_add(new);
                let c = if need <= cap {
                    cap
                } else {
                    need.max(cap * 2).max(if e == 1 { 8 } else { 4 })
                };
                c * e
            })
            .sum();
        let heap = match &self.kind {
            Kind::MinMax {
                values: Values::Bytes(_),
                heap,
                ..
            } => heap + arg.map_or(0, value_bytes),
            Kind::MinMax { heap, .. } => *heap,
            _ => 0,
        };
        vecs + heap
    }

    /// Adds groups up to `n` with empty states.
    pub(crate) fn resize(&mut self, n: usize) {
        match &mut self.kind {
            Kind::Count(c) => c.resize(n, 0),
            Kind::SumInt { sum, seen } => {
                sum.resize(n, 0);
                seen.resize(n, false);
            }
            Kind::SumFloat { sum, seen } => {
                sum.resize(n, 0.0);
                seen.resize(n, false);
            }
            Kind::SumDec { sum, state } => {
                sum.resize(n, 0);
                state.resize(n, 0);
            }
            Kind::AvgFloat { sum, count } => {
                sum.resize(n, 0.0);
                count.resize(n, 0);
            }
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => {
                sum.resize(n, 0);
                count.resize(n, 0);
                overflow.resize(n, false);
            }
            Kind::MinMax { values, set, .. } => {
                set.resize(n, false);
                match values {
                    Values::Int(v) => v.resize(n, 0),
                    Values::Float(v) => v.resize(n, 0.0),
                    Values::Bytes(v) => v.resize(n, Box::default()),
                }
            }
        }
    }

    /// Reserves room for `new` more groups (as [`projected`](Self::projected)).
    pub(crate) fn reserve(&mut self, new: usize) {
        match &mut self.kind {
            Kind::Count(c) => c.reserve(new),
            Kind::SumInt { sum, seen } => {
                sum.reserve(new);
                seen.reserve(new);
            }
            Kind::SumFloat { sum, seen } => {
                sum.reserve(new);
                seen.reserve(new);
            }
            Kind::SumDec { sum, state } => {
                sum.reserve(new);
                state.reserve(new);
            }
            Kind::AvgFloat { sum, count } => {
                sum.reserve(new);
                count.reserve(new);
            }
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => {
                sum.reserve(new);
                count.reserve(new);
                overflow.reserve(new);
            }
            Kind::MinMax { values, set, .. } => {
                set.reserve(new);
                match values {
                    Values::Int(v) => v.reserve(new),
                    Values::Float(v) => v.reserve(new),
                    Values::Bytes(v) => v.reserve(new),
                }
            }
        }
    }

    /// Folds rows of `cols` into `groups` (one group id per row): raw
    /// arguments, or partial states, per the aggregate's input.
    pub(crate) fn update(&mut self, groups: &[u32], cols: &[ArrayRef], w: &Weights) -> Result<()> {
        let args: Vec<&ArrayRef> = self
            .expr
            .args
            .iter()
            .map(|&i| arg(cols, i))
            .collect::<Result<_>>()?;
        let state = self.expr.input == AggInput::State;
        self.fold(groups, &args, state, w)
    }

    /// Merges partial states (`states`: this aggregate's state columns).
    pub(crate) fn merge(&mut self, groups: &[u32], states: &[ArrayRef]) -> Result<()> {
        let args: Vec<&ArrayRef> = states.iter().collect();
        self.fold(groups, &args, true, &Weights::Const(1))
    }

    fn fold(&mut self, groups: &[u32], args: &[&ArrayRef], state: bool, w: &Weights) -> Result<()> {
        let want = if state {
            self.state_len()
        } else {
            self.expr.args.len()
        };
        if args.len() != want || (want == 0 && !matches!(self.kind, Kind::Count(_))) {
            return Err(internal(format!(
                "{:?} got {} argument columns, wants {want}",
                self.expr.function,
                args.len()
            )));
        }
        if (state || matches!(self.kind, Kind::MinMax { .. })) && !is_unit(w) {
            return Err(unsupported(format!(
                "{:?} over rows with weights other than 1 (needs incremental view maintenance)",
                self.expr.function
            )));
        }
        let ansi = self.ansi;
        let bound = self.sum_bound;
        let fits = |s: &i128| bound.is_none_or(|b| s.unsigned_abs() < b);
        // Validity bitmaps, read once per batch (no virtual call per row).
        let nulls: Vec<Option<&arrow_buffer::NullBuffer>> =
            args.iter().map(|a| a.nulls()).collect();
        let first = nulls.first().copied().flatten();
        let valid = |i: usize| first.is_none_or(|n| n.is_valid(i));
        match (&mut self.kind, state) {
            (Kind::Count(c), false) => {
                for (i, &g) in groups.iter().enumerate() {
                    if nulls.iter().all(|n| n.is_none_or(|n| n.is_valid(i))) {
                        let x = &mut c[g as usize];
                        *x = x.wrapping_add(weight(w, i));
                    }
                }
            }
            (Kind::Count(c), true) => {
                let s = counts_of(args[0])?;
                for (&g, &v) in groups.iter().zip(s) {
                    let x = &mut c[g as usize];
                    *x = x.wrapping_add(v);
                }
            }
            (Kind::SumInt { sum, seen }, _) => {
                let a = args[0];
                let v = ints(a)?;
                for (i, &g) in groups.iter().enumerate() {
                    if valid(i) {
                        let g = g as usize;
                        let wv = if state {
                            Some(v[i])
                        } else {
                            v[i].checked_mul(weight(w, i))
                        };
                        let s = wv.and_then(|x| sum[g].checked_add(x));
                        sum[g] = match s {
                            Some(s) => s,
                            None if ansi => return Err(long_overflow()),
                            None => sum[g].wrapping_add(v[i].wrapping_mul(if state {
                                1
                            } else {
                                weight(w, i)
                            })),
                        };
                        seen[g] = true;
                    }
                }
            }
            (Kind::SumFloat { sum, seen }, _) => {
                let a = args[0];
                let v = floats(a)?;
                for (i, &g) in groups.iter().enumerate() {
                    if valid(i) {
                        #[expect(clippy::cast_precision_loss, reason = "a weight scales a double")]
                        let wv = if state { 1.0 } else { weight(w, i) as f64 };
                        sum[g as usize] += v[i] * wv;
                        seen[g as usize] = true;
                    }
                }
            }
            (Kind::SumDec { sum, state: st }, _) => {
                let a = args[0];
                let v = decimals(a)?;
                let nonempty = args.get(1).map(|b| booleans_of(b)).transpose()?;
                for (i, &g) in groups.iter().enumerate() {
                    let g = g as usize;
                    if state {
                        if nonempty.is_some_and(|b| b.value(i)) {
                            if !valid(i) {
                                st[g] = 2;
                            } else if st[g] != 2 {
                                (sum[g], st[g]) = sum[g]
                                    .checked_add(v[i])
                                    .filter(fits)
                                    .map_or((0, 2), |s| (s, 1));
                            }
                        }
                    } else if valid(i) && st[g] != 2 {
                        let s = v[i]
                            .checked_mul(i128::from(weight(w, i)))
                            .and_then(|x| sum[g].checked_add(x))
                            .filter(fits);
                        (sum[g], st[g]) = s.map_or((0, 2), |s| (s, 1));
                    }
                }
            }
            (Kind::AvgFloat { sum, count }, _) => {
                let a = args[0];
                let v = floats(a)?;
                let counts = args.get(1).map(|c| counts_of(c)).transpose()?;
                for (i, &g) in groups.iter().enumerate() {
                    let g = g as usize;
                    if let Some(c) = counts {
                        if valid(i) {
                            sum[g] += v[i];
                        }
                        count[g] = count[g].wrapping_add(c[i]);
                    } else if valid(i) {
                        let wt = weight(w, i);
                        #[expect(clippy::cast_precision_loss, reason = "a weight scales a double")]
                        let wv = wt as f64;
                        sum[g] += v[i] * wv;
                        count[g] = count[g].wrapping_add(wt);
                    }
                }
            }
            (
                Kind::AvgDec {
                    sum,
                    count,
                    overflow,
                },
                _,
            ) => {
                let a = args[0];
                let v = decimals(a)?;
                let counts = args.get(1).map(|c| counts_of(c)).transpose()?;
                for (i, &g) in groups.iter().enumerate() {
                    let g = g as usize;
                    let (add, n) = match counts {
                        Some(c) if valid(i) => (Some(v[i]), c[i]),
                        Some(c) => {
                            overflow[g] |= c[i] != 0;
                            (Some(0), c[i])
                        }
                        None if valid(i) => {
                            let wt = weight(w, i);
                            (v[i].checked_mul(i128::from(wt)), wt)
                        }
                        None => continue,
                    };
                    match add.and_then(|x| sum[g].checked_add(x)).filter(fits) {
                        Some(s) => sum[g] = s,
                        None => overflow[g] = true,
                    }
                    count[g] = count[g].wrapping_add(n);
                }
            }
            (
                Kind::MinMax {
                    max,
                    values,
                    set,
                    heap,
                },
                _,
            ) => {
                let a = args[0];
                let max = *max;
                match values {
                    Values::Int(vals) => {
                        let v = int_like(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if valid(i) && (!set[g] || (v[i] > vals[g]) == max && v[i] != vals[g]) {
                                vals[g] = v[i];
                                set[g] = true;
                            }
                        }
                    }
                    Values::Float(vals) => {
                        let v = floats(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            let better = if max {
                                float_lt(vals[g], v[i])
                            } else {
                                float_lt(v[i], vals[g])
                            };
                            if valid(i) && (!set[g] || better) {
                                vals[g] = v[i];
                                set[g] = true;
                            }
                        }
                    }
                    Values::Bytes(vals) => {
                        let v = byte_values(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            let cur: &[u8] = &vals[g];
                            let better = if max { v[i] > cur } else { v[i] < cur };
                            if valid(i) && (!set[g] || better) {
                                *heap = heap.saturating_sub(vals[g].len()) + v[i].len();
                                vals[g] = v[i].into();
                                set[g] = true;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The raw argument columns' indices.
    pub(crate) fn args(&self) -> &[usize] {
        &self.expr.args
    }

    /// Number of state columns.
    pub(crate) fn state_len(&self) -> usize {
        match self.kind {
            Kind::Count(_) | Kind::SumInt { .. } | Kind::SumFloat { .. } | Kind::MinMax { .. } => 1,
            Kind::SumDec { .. } | Kind::AvgFloat { .. } | Kind::AvgDec { .. } => 2,
        }
    }

    /// The partial states of groups `ids`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let pick = |i: &u32| *i as usize;
        Ok(match &self.kind {
            Kind::Count(c) => vec![Arc::new(Int64Array::from_iter_values(
                ids.iter().map(|i| c[pick(i)]),
            ))],
            Kind::SumInt { sum, seen } => vec![Arc::new(Int64Array::from_iter(
                ids.iter().map(|i| seen[pick(i)].then(|| sum[pick(i)])),
            ))],
            Kind::SumFloat { sum, seen } => vec![Arc::new(Float64Array::from_iter(
                ids.iter().map(|i| seen[pick(i)].then(|| sum[pick(i)])),
            ))],
            Kind::SumDec { sum, state } => vec![
                self.decimal(
                    ids.iter()
                        .map(|i| (state[pick(i)] == 1).then(|| sum[pick(i)])),
                    &self.result,
                )?,
                Arc::new(BooleanArray::from_iter(
                    ids.iter().map(|i| Some(state[pick(i)] != 0)),
                )),
            ],
            Kind::AvgFloat { sum, count } => vec![
                Arc::new(Float64Array::from_iter_values(
                    ids.iter().map(|i| sum[pick(i)]),
                )),
                Arc::new(Int64Array::from_iter_values(
                    ids.iter().map(|i| count[pick(i)]),
                )),
            ],
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => vec![
                self.decimal(
                    ids.iter()
                        .map(|i| (!overflow[pick(i)]).then(|| sum[pick(i)])),
                    &self.expr.avg_sum_type(),
                )?,
                Arc::new(Int64Array::from_iter_values(
                    ids.iter().map(|i| count[pick(i)]),
                )),
            ],
            Kind::MinMax { .. } => vec![self.min_max(ids)?],
        })
    }

    fn decimal(&self, v: impl Iterator<Item = Option<i128>>, t: &DataType) -> Result<ArrayRef> {
        let a = Decimal128Array::from_iter(v).with_data_type(t.clone());
        Ok(Arc::new(a))
    }

    fn min_max(&self, ids: &[u32]) -> Result<ArrayRef> {
        let Kind::MinMax { values, set, .. } = &self.kind else {
            return Err(internal("not min/max".to_owned()));
        };
        let t = &self.result;
        let on = |i: &u32| set[*i as usize];
        let at = |i: &u32| *i as usize;
        Ok(match values {
            Values::Float(v) => {
                let it = ids.iter().map(|i| on(i).then(|| v[at(i)]));
                if *t == DataType::Float32 {
                    #[expect(clippy::cast_possible_truncation, reason = "the value was a float")]
                    let a: PrimitiveArray<Float32Type> = it.map(|x| x.map(|x| x as f32)).collect();
                    Arc::new(a)
                } else {
                    Arc::new(it.collect::<Float64Array>())
                }
            }
            Values::Bytes(v) => {
                let it = ids.iter().map(|i| on(i).then(|| &*v[at(i)]));
                match t {
                    DataType::Binary => Arc::new(it.collect::<BinaryArray>()),
                    DataType::BinaryView => Arc::new(it.collect::<BinaryViewArray>()),
                    _ => {
                        let s: Vec<Option<&str>> = it
                            .map(|b| b.map(std::str::from_utf8).transpose())
                            .collect::<Result<_, _>>()
                            .map_err(|e| internal(e.to_string()))?;
                        if *t == DataType::Utf8 {
                            Arc::new(StringArray::from(s))
                        } else {
                            Arc::new(StringViewArray::from(s))
                        }
                    }
                }
            }
            Values::Int(v) => {
                let it = ids.iter().map(|i| on(i).then(|| v[at(i)]));
                narrow(it, t)?
            }
        })
    }

    /// The final values of groups `ids`.
    ///
    /// # Errors
    /// ANSI overflow errors.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let at = |i: &u32| *i as usize;
        let ansi = self.ansi;
        Ok(match &self.kind {
            Kind::SumDec { sum, state } => {
                let DataType::Decimal128(p, s) = self.result else {
                    return Err(internal("decimal sum type".to_owned()));
                };
                let mut out = Vec::with_capacity(ids.len());
                for i in ids {
                    let v = sum[at(i)];
                    out.push(match state[at(i)] {
                        0 => None,
                        1 if v.unsigned_abs() < pow10(p).unsigned_abs() => Some(v),
                        _ if !ansi => None,
                        1 => return Err(out_of_range(plain(v, s), p, s)),
                        _ => return Err(decimal_sum_overflow("sum")),
                    });
                }
                self.decimal(out.into_iter(), &self.result)?
            }
            Kind::AvgFloat { sum, count } => {
                Arc::new(Float64Array::from_iter(ids.iter().map(|i| {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "Spark divides by count as double"
                    )]
                    let n = count[at(i)] as f64;
                    (count[at(i)] != 0).then(|| sum[at(i)] / n)
                })))
            }
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => {
                let (DataType::Decimal128(p, s), DataType::Decimal128(_, ss)) =
                    (&self.result, self.expr.avg_sum_type())
                else {
                    return Err(internal("decimal avg type".to_owned()));
                };
                let mut out = Vec::with_capacity(ids.len());
                for i in ids {
                    let g = at(i);
                    out.push(if count[g] == 0 {
                        None
                    } else if overflow[g] {
                        if ansi {
                            return Err(decimal_sum_overflow("avg"));
                        }
                        None
                    } else {
                        match avg_decimal(sum[g], ss, count[g], *p, *s) {
                            (Some(v), _) => Some(v),
                            (None, text) if ansi => return Err(out_of_range(text, *p, *s)),
                            (None, _) => None,
                        }
                    });
                }
                self.decimal(out.into_iter(), &self.result)?
            }
            Kind::Count(_) | Kind::SumInt { .. } | Kind::SumFloat { .. } | Kind::MinMax { .. } => {
                let mut s = self.state(ids)?;
                s.swap_remove(0)
            }
        })
    }
}

/// Builds an array of type `t` from `i128` values (inverse of `int_like`).
fn narrow(it: impl Iterator<Item = Option<i128>>, t: &DataType) -> Result<ArrayRef> {
    fn prim<T>(it: impl Iterator<Item = Option<i128>>, t: &DataType) -> Result<ArrayRef>
    where
        T: arrow_array::ArrowPrimitiveType,
        T::Native: TryFrom<i128>,
    {
        let v: Vec<Option<T::Native>> = it
            .map(|x| x.map(T::Native::try_from).transpose())
            .collect::<Result<_, _>>()
            .map_err(|_| internal(format!("min/max value out of range of {t}")))?;
        let a: PrimitiveArray<T> = v.into_iter().collect();
        Ok(Arc::new(a.with_data_type(t.clone())))
    }
    match t {
        DataType::Boolean => Ok(Arc::new(
            it.map(|x| x.map(|v| v != 0)).collect::<BooleanArray>(),
        )),
        DataType::Int8 => prim::<Int8Type>(it, t),
        DataType::Int16 => prim::<Int16Type>(it, t),
        DataType::Int32 => prim::<Int32Type>(it, t),
        DataType::Int64 => prim::<Int64Type>(it, t),
        DataType::Date32 => prim::<Date32Type>(it, t),
        DataType::Timestamp(..) => prim::<TimestampMicrosecondType>(it, t),
        _ => prim::<Decimal128Type>(it, t),
    }
}

#[cfg(test)]
#[cfg(not(loom))]
mod tests {
    use super::*;

    #[test]
    fn avg_decimal_rounds_like_spark() {
        // 5 / 3 at decimal(38,22) (Spark: 1.6666666666666666666667).
        let (v, _) = avg_decimal(5_000_000_000_000_000_000, 18, 3, 38, 22);
        assert_eq!(plain(v.unwrap(), 22), "1.6666666666666666666667");
        let (v, _) = avg_decimal(-5, 0, 3, 38, 4);
        assert_eq!(plain(v.unwrap(), 4), "-1.6667");
        // Spark: avg of 1e20 - 1e-18 and 1 does not fit decimal(38,22).
        let sum = 100_000_000_000_000_000_000_999_999_999_999_999_999i128;
        let (v, text) = avg_decimal(sum, 18, 2, 38, 22);
        assert_eq!(v, None);
        assert_eq!(
            text,
            "50000000000000000000.499999999999999999500000000000000000000"
        );
        let big = 10i128.pow(38) - 1;
        let (v, text) = avg_decimal(big, 0, 1, 38, 4);
        assert_eq!(v, None);
        assert!(text.starts_with("99999999999999999999999999999999999999.000"));
        // Half-up carry through every digit.
        let (v, _) = avg_decimal(2, 0, 3, 10, 0);
        assert_eq!(v, Some(1));
        assert!(round_up(&mut [9, 9]));
        let long = "x".repeat(100);
        let all: ArrayRef = Arc::new(StringArray::from(vec![long.as_str(), "ab"]));
        assert_eq!(value_bytes(&all.slice(1, 1)), 2);
        let views: ArrayRef = Arc::new(StringViewArray::from(vec![long.as_str(), "ab"]));
        assert_eq!(value_bytes(&views.slice(1, 1)), 2);
        let bin: ArrayRef = Arc::new(BinaryArray::from(vec![&b"abc"[..]]));
        assert_eq!(value_bytes(&bin), 3);
        let bv: ArrayRef = Arc::new(BinaryViewArray::from(vec![&b"abcd"[..]]));
        assert_eq!(value_bytes(&bv), 4);
        assert_eq!(plain(-5, 3), "-0.005");
        assert!(float_lt(1.0, f64::NAN) && !float_lt(f64::NAN, f64::NAN) && !float_lt(-0.0, 0.0));
    }
}
