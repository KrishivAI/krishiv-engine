// SPDX-License-Identifier: Apache-2.0
//! Per-group accumulators, one vector per aggregate (column-oriented), with
//! Spark 4.2 arithmetic: `sum` of integers in `bigint` (ANSI raises, legacy
//! wraps), decimals in `i128`, `avg` of decimals by Spark's division
//! (quotient at scale 39, then rounded half-up to the result scale).
//!
//! Decimal sums overflow as soon as a running sum leaves the sum type's
//! precision, even if later rows bring it back (`ARITHMETIC_OVERFLOW
//! "Overflow in sum of decimals"` under ANSI, NULL otherwise), grouped or
//! not. Whether a temporary overflow happens depends on the summation
//! order, as in Spark 4.2's grouped and partial buffers. Spark's global
//! aggregate (when no partial buffer overflows) and its grouped aggregate
//! over a local relation tolerate a temporary overflow; Krishiv does not (a
//! documented divergence, ADR 0023).
//!
//! **Per-batch proof (P2-T160):** each decimal `sum`/`avg` keeps an upper
//! bound of every group's `|sum|`; a batch whose `Σ |weight·value|` keeps
//! that bound below the sum type's `10^p` cannot overflow any group in any
//! order, so it folds without per-row checks. Other batches take the
//! per-row checked path. Decimal `avg` with a result scale of at most 19
//! divides once instead of digit by digit (see [`avg_fast`]).

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Decimal128Array, Float64Array,
    Int64Array, PrimitiveArray, StringArray, StringViewArray,
};
use arrow_buffer::{NullBuffer, i256};
use arrow_schema::{DataType, TimeUnit};
use krishiv_kernels::decimal::DecimalNative;
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
            | DataType::Decimal64(..)
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

impl Values {
    /// No groups yet, of the physical kind of Spark type `t`.
    fn empty(t: &DataType) -> Self {
        match t {
            DataType::Float32 | DataType::Float64 => Self::Float(Vec::new()),
            DataType::Utf8 | DataType::Utf8View | DataType::Binary | DataType::BinaryView => {
                Self::Bytes(Vec::new())
            }
            _ => Self::Int(Vec::new()),
        }
    }

    fn resize(&mut self, n: usize) {
        match self {
            Self::Int(v) => v.resize(n, 0),
            Self::Float(v) => v.resize(n, 0.0),
            Self::Bytes(v) => v.resize(n, Box::default()),
        }
    }

    fn reserve(&mut self, n: usize) {
        match self {
            Self::Int(v) => v.reserve(n),
            Self::Float(v) => v.reserve(n),
            Self::Bytes(v) => v.reserve(n),
        }
    }

    fn vec(&self) -> (usize, usize, usize) {
        fn v<T>(x: &Vec<T>) -> (usize, usize, usize) {
            (x.len(), x.capacity(), size_of::<T>())
        }
        match self {
            Self::Int(x) => v(x),
            Self::Float(x) => v(x),
            Self::Bytes(x) => v(x),
        }
    }
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
    /// `first` / `last`: `set` = a row was taken, `null` = its value is NULL.
    Pick {
        last: bool,
        ignore_nulls: bool,
        values: Values,
        set: Vec<bool>,
        null: Vec<bool>,
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
    /// An upper bound of `|sum|` over every group of a decimal `sum`/`avg`,
    /// refreshed after each fold ([`refresh_max`]).
    max_sum: u128,
    /// Whether decimal folds may take the per-batch proof path (tests turn
    /// it off to compare against the per-row path).
    proofs: bool,
}

/// `Σ |w_i|·|v_i|` over the valid rows of a batch (saturating; unit
/// weights when `w` is `None`, as for merged partial states).
fn batch_abs<N: DecimalNative>(v: &[N], nulls: Option<&NullBuffer>, w: Option<&Weights>) -> u128 {
    let at = |i: usize, x: N| match w {
        None | Some(Weights::Const(1)) => x.wide().unsigned_abs(),
        Some(w) => x
            .wide()
            .unsigned_abs()
            .saturating_mul(u128::from(weight(w, i).unsigned_abs())),
    };
    match nulls {
        None => v
            .iter()
            .enumerate()
            .fold(0_u128, |s, (i, &x)| s.saturating_add(at(i, x))),
        Some(n) => v
            .iter()
            .zip(n.iter())
            .enumerate()
            .fold(0_u128, |s, (i, (&x, ok))| {
                s.saturating_add(if ok { at(i, x) } else { 0 })
            }),
    }
}

/// The per-batch overflow proof of a decimal `sum`/`avg` fold (design
/// P2-T160): every stored sum is within `max_sum` and the batch moves any
/// one group's sum by at most `Σ |w|·|v|`, so if `max_sum + Σ |w|·|v|` is
/// below the bound, no running sum of any group leaves the bound during
/// the batch, in any row order. The fold may then add with wrapping
/// arithmetic and no range test, and its result is exactly the per-row
/// path's. Proofs are local to one batch; the next batch is tested afresh.
fn proven<N: DecimalNative>(
    enabled: bool,
    bound: Option<u128>,
    max_sum: u128,
    v: &[N],
    nulls: Option<&NullBuffer>,
    w: Option<&Weights>,
) -> bool {
    enabled
        && bound.is_some_and(|b| {
            max_sum.saturating_add(quick_abs(v, w)) < b
                || max_sum.saturating_add(batch_abs(v, nulls, w)) < b
        })
}

/// A cheap upper bound of [`batch_abs`]: rows × max |v| × max |w|, with
/// each maximum bounded by an OR of ones'-complement magnitudes (`|x| <=
/// OR + 1`, branch-free) over every slot, NULL ones included. When it is
/// too loose (values near the bound, large values under NULLs) the caller
/// takes the exact sum.
fn quick_abs<N: DecimalNative>(v: &[N], w: Option<&Weights>) -> u128 {
    let or = v
        .iter()
        .fold(0_u128, |m, &x| m | x.ones_magnitude())
        .saturating_add(1);
    let wmax = match w {
        None => 1,
        Some(Weights::Const(c)) => u128::from(c.unsigned_abs()),
        Some(Weights::Column(c)) => u128::from(
            c.values()
                .iter()
                .fold(0_u64, |m, &x| m | (x ^ (x >> 63)).cast_unsigned()),
        )
        .saturating_add(1),
    };
    let rows = u128::try_from(v.len()).unwrap_or(u128::MAX);
    or.saturating_mul(rows).saturating_mul(wmax)
}

/// The new bound of every group's `|sum|` after a fold touched `groups`:
/// the exact maximum over all groups when there are no more groups than
/// rows (it can then shrink, e.g. after IVM retractions), else the old
/// bound raised by the touched groups' sums (untouched sums are within it).
fn refresh_max(old: u128, sum: &[i128], groups: &[u32]) -> u128 {
    let abs = |v: &i128| v.unsigned_abs();
    if sum.len() <= groups.len() {
        sum.iter().map(abs).max().unwrap_or(0)
    } else {
        groups
            .iter()
            .filter_map(|&g| sum.get(g as usize))
            .map(abs)
            .fold(old, u128::max)
    }
}

/// One batch of a decimal `sum`/`avg` fold, over input values of either
/// carrier (`N`): raw rows (`w` is `Some`) or merged partial states.
struct DecFold<'a> {
    groups: &'a [u32],
    /// The validity of the value column.
    nulls: Option<&'a NullBuffer>,
    /// Row weights; `None` for partial states.
    w: Option<&'a Weights>,
    bound: Option<u128>,
    max_sum: u128,
    proofs: bool,
}

impl DecFold<'_> {
    fn valid(&self, i: usize) -> bool {
        self.nulls.is_none_or(|n| n.is_valid(i))
    }

    fn fits(&self, s: i128) -> bool {
        self.bound.is_none_or(|b| s.unsigned_abs() < b)
    }

    /// Whether the batch takes the proof path (see [`proven`]).
    fn proven<N: DecimalNative>(&self, v: &[N]) -> bool {
        proven(self.proofs, self.bound, self.max_sum, v, self.nulls, self.w)
    }

    /// `sum` (`st`: 0 no value, 1 a value, 2 overflowed); a partial state
    /// has a `nonempty` column.
    fn sum<N: DecimalNative>(
        &self,
        sum: &mut [i128],
        st: &mut [u8],
        v: &[N],
        nonempty: Option<&BooleanArray>,
    ) {
        let fast = self.proven(v);
        for (i, &g) in self.groups.iter().enumerate() {
            let g = g as usize;
            let x = v[i].wide();
            match self.w {
                None => {
                    if nonempty.is_some_and(|b| b.value(i)) {
                        // An overflowed partial (sum NULL, nonempty) has
                        // no magnitude: it is sticky on both paths.
                        if !self.valid(i) {
                            st[g] = 2;
                        } else if st[g] != 2 {
                            (sum[g], st[g]) = if fast {
                                (sum[g].wrapping_add(x), 1)
                            } else {
                                sum[g]
                                    .checked_add(x)
                                    .filter(|&s| self.fits(s))
                                    .map_or((0, 2), |s| (s, 1))
                            };
                        }
                    }
                }
                Some(w) if self.valid(i) && st[g] != 2 => {
                    let s = if fast && matches!(w, Weights::Const(1)) {
                        Some(sum[g].wrapping_add(x))
                    } else if fast {
                        Some(sum[g].wrapping_add(x.wrapping_mul(i128::from(weight(w, i)))))
                    } else {
                        x.checked_mul(i128::from(weight(w, i)))
                            .and_then(|x| sum[g].checked_add(x))
                            .filter(|&s| self.fits(s))
                    };
                    (sum[g], st[g]) = s.map_or((0, 2), |s| (s, 1));
                }
                Some(_) => {}
            }
        }
    }

    /// `avg`; a partial state has a `counts` column.
    fn avg<N: DecimalNative>(
        &self,
        sum: &mut [i128],
        count: &mut [i64],
        overflow: &mut [bool],
        v: &[N],
        counts: Option<&[i64]>,
    ) {
        let fast = self.proven(v);
        for (i, &g) in self.groups.iter().enumerate() {
            let g = g as usize;
            let x = v[i].wide();
            let (add, n) = match (counts, self.w) {
                (Some(c), _) if self.valid(i) => (Some(x), c[i]),
                (Some(c), _) => {
                    overflow[g] |= c[i] != 0;
                    (Some(0), c[i])
                }
                (None, Some(w)) if self.valid(i) => {
                    let wt = weight(w, i);
                    if fast {
                        (Some(x.wrapping_mul(i128::from(wt))), wt)
                    } else {
                        (x.checked_mul(i128::from(wt)), wt)
                    }
                }
                (None, _) => continue,
            };
            let s = if fast {
                add.map(|x| sum[g].wrapping_add(x))
            } else {
                add.and_then(|x| sum[g].checked_add(x))
                    .filter(|&s| self.fits(s))
            };
            match s {
                Some(s) => sum[g] = s,
                None => overflow[g] = true,
            }
            count[g] = count[g].wrapping_add(n);
        }
    }
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

/// The values of a decimal column of either carrier.
#[derive(Clone, Copy)]
enum Decimals<'a> {
    I64(&'a [i64]),
    I128(&'a [i128]),
}

fn decimals(a: &ArrayRef) -> Result<Decimals<'_>> {
    match a.data_type() {
        DataType::Decimal64(..) => Ok(Decimals::I64(a.as_primitive::<Decimal64Type>().values())),
        DataType::Decimal128(..) => Ok(Decimals::I128(a.as_primitive::<Decimal128Type>().values())),
        t => Err(internal(format!("decimal argument of type {t}"))),
    }
}

/// `min`/`max` input as `i128`s (integral, boolean, decimal, temporal).
fn int_like(a: &ArrayRef) -> Result<Cow<'_, [i128]>> {
    let conv = |v: Vec<i128>| Cow::Owned(v);
    Ok(match a.data_type() {
        DataType::Decimal128(..) | DataType::Decimal64(..) => match decimals(a)? {
            Decimals::I128(v) => Cow::Borrowed(v),
            Decimals::I64(v) => conv(v.iter().map(|&x| i128::from(x)).collect()),
        },
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

/// [`avg_decimal`]'s value by one `HALF_UP` division (design P2-T160 D3),
/// or `None` when that shortcut does not apply (result scale above 19 or
/// below the sum's, numerator beyond `u128`) or the value does not fit
/// `decimal(p, s)` (the caller then takes the digit loop for the error
/// text).
///
/// Spark rounds the quotient `HALF_UP` to scale 39, then `HALF_UP` to `s`.
/// For `s <= 19` that equals one rounding of the exact quotient to `s`: the
/// first rounding can only change the second when the exact quotient is
/// within `½·10^-39` of a half point at scale `s` without being on it, but
/// such a distance is a nonzero multiple of `1 / (2·count·10^s)`, at least
/// `½·10^-(19+s)` since `|count| < 10^19`; exact half points are
/// representable at scale 39 and round up both ways.
fn avg_fast(sum: i128, sum_scale: i8, count: i64, p: u8, s: i8) -> Option<i128> {
    let (ss, rs) = (u8::try_from(sum_scale).ok()?, u8::try_from(s).ok()?);
    if rs > 19 || rs < ss || count == 0 {
        return None;
    }
    let n = sum
        .unsigned_abs()
        .checked_mul(pow10(rs - ss).unsigned_abs())?;
    let d = u128::from(count.unsigned_abs());
    let (q, r) = (n / d, n % d);
    // `r < d < 2^64`, so `d - r` cannot underflow; `2r >= d` rounds up.
    let q = q + u128::from(r >= d - r);
    let v = i128::try_from(q).ok().filter(|&v| v < pow10(p))?;
    Some(if (sum < 0) != (count < 0) { -v } else { v })
}

/// A decimal `sum` that was kept exact (a window frame's, which Spark
/// keeps as an unbounded `Decimal`; only the evaluation checks it), as
/// `decimal(p, s)`: NULL for no value, else the value if it fits; past
/// `p` digits NULL, or with `ansi` `NUMERIC_VALUE_OUT_OF_RANGE`.
///
/// # Errors
/// `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION` as above.
pub(crate) fn wide_sum_value(sum: Option<i256>, p: u8, s: i8, ansi: bool) -> Result<Option<i128>> {
    let Some(sum) = sum else {
        return Ok(None);
    };
    let bound = i256::from_i128(pow10(p));
    let fits = sum < bound && sum > bound.wrapping_neg();
    match sum.to_i128() {
        Some(v) if fits => Ok(Some(v)),
        _ if !ansi => Ok(None),
        _ => {
            let digits = sum.wrapping_abs().to_string();
            let scale = usize::try_from(s).unwrap_or(0);
            Err(out_of_range(
                plain_digits(&digits, scale, sum.is_negative()),
                p,
                s,
            ))
        }
    }
}

/// Spark's decimal `avg` of an exact `sum` (scale `sum_scale`) over
/// `count` rows as `decimal(p, s)`, as [`avg_decimal`] computes it for a
/// sum within `i128`: `Ok(None)` when it does not fit, with the scale-39
/// quotient's text for the error.
pub(crate) fn avg_decimal_wide(
    sum: i256,
    sum_scale: i8,
    count: i64,
    p: u8,
    s: i8,
) -> (Option<i128>, String) {
    if let Some(sum) = sum.to_i128() {
        if let Some(v) = avg_fast(sum, sum_scale, count, p, s) {
            return (Some(v), String::new());
        }
        return avg_decimal(sum, sum_scale, count, p, s);
    }
    let negative = sum.is_negative() != (count < 0);
    let a = sum.wrapping_abs();
    let b = i256::from_i128(i128::from(count.unsigned_abs()));
    let ss = usize::try_from(sum_scale).unwrap_or(0);
    let (Some(mut int), Some(rem)) = (a.checked_div(b), a.checked_rem(b)) else {
        return (None, String::new());
    };
    // `rem < count < 2^63`: the digit loop runs in `u128`, as `avg_decimal`.
    let b = u128::from(count.unsigned_abs());
    let mut rem = rem
        .to_i128()
        .and_then(|r| u128::try_from(r).ok())
        .unwrap_or(0);
    let mut digit = || {
        rem *= 10;
        let d = u8::try_from(rem / b).unwrap_or(9);
        rem %= b;
        d
    };
    let mut frac: Vec<u8> = (0..39 - ss).map(|_| digit()).collect();
    if digit() >= 5 && round_up(&mut frac) {
        int = int.wrapping_add(i256::ONE);
    }
    let mut digits = int.to_string();
    digits.extend(frac.iter().map(|d| char::from(b'0' + d)));
    let text = plain_digits(&digits, 39, negative && digits.bytes().any(|d| d != b'0'));
    let keep = usize::try_from(s).unwrap_or(0).saturating_sub(ss);
    let up = frac.get(keep).is_some_and(|&d| d >= 5);
    let ten = i256::from_i128(10);
    let mut v = Some(int);
    for &d in frac.iter().take(keep) {
        v = v
            .and_then(|v| v.checked_mul(ten))
            .and_then(|v| v.checked_add(i256::from_i128(i128::from(d))));
    }
    let v = v
        .and_then(|v| v.checked_add(i256::from_i128(i128::from(up))))
        .and_then(i256::to_i128)
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

/// The snapshots of a running fold ([`Accumulator::update_running`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Running<'a> {
    /// The group holding each row's partition state.
    pub(crate) parts: &'a [u32],
    /// The group that takes the state after each row, or [`Running::NONE`].
    pub(crate) snapshots: &'a [u32],
}

impl Running<'_> {
    /// No snapshot after this row.
    pub(crate) const NONE: u32 = u32::MAX;
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
                values: Values::empty(t),
                set: Vec::new(),
                heap: 0,
            },
            (f @ (F::First { ignore_nulls } | F::Last { ignore_nulls }), t) => Kind::Pick {
                last: matches!(f, F::Last { .. }),
                ignore_nulls,
                values: Values::empty(t),
                set: Vec::new(),
                null: Vec::new(),
                heap: 0,
            },
        };
        let sum_bound = match (expr.function, expr.avg_sum_type()) {
            (F::Sum | F::Avg, DataType::Decimal64(p, _) | DataType::Decimal128(p, _)) => {
                Some(pow10(p).unsigned_abs())
            }
            _ => None,
        };
        Ok(Self {
            kind,
            expr: expr.clone(),
            result,
            ansi,
            sum_bound,
            max_sum: 0,
            proofs: true,
        })
    }

    /// Bytes one group's state takes (heap bytes of string values aside).
    pub(crate) fn group_bytes(&self) -> usize {
        self.vecs().iter().map(|&(_, _, e)| e).sum()
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
            Kind::MinMax { values, set, .. } => vec![values.vec(), v(set)],
            Kind::Pick {
                values, set, null, ..
            } => vec![values.vec(), v(set), v(null)],
        }
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        let heap = match &self.kind {
            Kind::MinMax { heap, .. } | Kind::Pick { heap, .. } => *heap,
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
            }
            | Kind::Pick {
                values: Values::Bytes(_),
                heap,
                ..
            } => heap + arg.map_or(0, value_bytes),
            Kind::MinMax { heap, .. } | Kind::Pick { heap, .. } => *heap,
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
                values.resize(n);
            }
            Kind::Pick {
                values, set, null, ..
            } => {
                set.resize(n, false);
                null.resize(n, false);
                values.resize(n);
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
                values.reserve(new);
            }
            Kind::Pick {
                values, set, null, ..
            } => {
                set.reserve(new);
                null.reserve(new);
                values.reserve(new);
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
        self.fold(groups, &args, state, w, None)
    }

    /// [`update`](Self::update) over raw arguments, keeping the state after
    /// chosen rows: after row `i` is folded, if `run.snapshots[i]` is a
    /// group, `run.parts[i]`'s state is copied into it. A window's running
    /// frame takes the state after each row or peer group this way: one
    /// pass over the rows, folded one by one and in order as Spark's frames
    /// fold them, with the argument columns, validity and conversions read
    /// once.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for the aggregates a window evaluates elsewhere
    /// (decimal `sum`/`avg`, string `min`/`max`/`first`/`last`), partial
    /// states, or slices of other lengths than the rows.
    pub(crate) fn update_running(
        &mut self,
        groups: &[u32],
        cols: &[ArrayRef],
        run: &Running<'_>,
    ) -> Result<()> {
        if run.parts.len() != groups.len() || run.snapshots.len() != groups.len() {
            return Err(internal(
                "a running fold over slices of different lengths".to_owned(),
            ));
        }
        let args: Vec<&ArrayRef> = self
            .expr
            .args
            .iter()
            .map(|&i| arg(cols, i))
            .collect::<Result<_>>()?;
        if self.expr.input == AggInput::State {
            return Err(internal("a running fold over partial states".to_owned()));
        }
        self.fold(groups, &args, false, &Weights::Const(1), Some(run))
    }

    /// Merges partial states (`states`: this aggregate's state columns).
    pub(crate) fn merge(&mut self, groups: &[u32], states: &[ArrayRef]) -> Result<()> {
        let args: Vec<&ArrayRef> = states.iter().collect();
        self.fold(groups, &args, true, &Weights::Const(1), None)
    }

    fn fold(
        &mut self,
        groups: &[u32],
        args: &[&ArrayRef],
        state: bool,
        w: &Weights,
        run: Option<&Running<'_>>,
    ) -> Result<()> {
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
        if (state || matches!(self.kind, Kind::MinMax { .. } | Kind::Pick { .. })) && !is_unit(w) {
            return Err(unsupported(format!(
                "{:?} over rows with weights other than 1 (needs incremental view maintenance)",
                self.expr.function
            )));
        }
        let ansi = self.ansi;
        let bound = self.sum_bound;
        let (proofs, max_sum) = (self.proofs, self.max_sum);
        // Validity bitmaps, read once per batch (no virtual call per row);
        // logical, so an untyped NULL column (`count(NULL)`) counts nothing.
        let nulls: Vec<Option<NullBuffer>> =
            args.iter().map(arrow_array::Array::logical_nulls).collect();
        let first = nulls.first().and_then(Option::as_ref);
        let valid = |i: usize| first.is_none_or(|n| n.is_valid(i));
        // After row `i` of a running fold: the state of its partition's
        // group is copied into the row's snapshot group, if it has one.
        macro_rules! snapshot {
            ($i:expr, $($v:ident),+) => {
                if let Some(r) = run {
                    let to = r.snapshots[$i];
                    if to != Running::NONE {
                        let (to, from) = (to as usize, r.parts[$i] as usize);
                        $( $v[to] = $v[from].clone(); )+
                    }
                }
            };
        }
        if run.is_some()
            && matches!(
                self.kind,
                Kind::SumDec { .. }
                    | Kind::AvgDec { .. }
                    | Kind::MinMax {
                        values: Values::Bytes(_),
                        ..
                    }
                    | Kind::Pick {
                        values: Values::Bytes(_),
                        ..
                    }
            )
        {
            return Err(internal(format!(
                "{:?} over {:?} has no running fold",
                self.expr.function, self.expr.arg_type
            )));
        }
        match (&mut self.kind, state) {
            (Kind::Count(c), false) => {
                for (i, &g) in groups.iter().enumerate() {
                    if nulls
                        .iter()
                        .all(|n| n.as_ref().is_none_or(|n| n.is_valid(i)))
                    {
                        let x = &mut c[g as usize];
                        *x = x.wrapping_add(weight(w, i));
                    }
                    snapshot!(i, c);
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
                    snapshot!(i, sum, seen);
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
                    snapshot!(i, sum, seen);
                }
            }
            (Kind::SumDec { sum, state: st }, _) => {
                let nonempty = args.get(1).map(|b| booleans_of(b)).transpose()?;
                let f = DecFold {
                    groups,
                    nulls: first,
                    w: (!state).then_some(w),
                    bound,
                    max_sum,
                    proofs,
                };
                match decimals(args[0])? {
                    Decimals::I64(v) => f.sum(sum, st, v, nonempty),
                    Decimals::I128(v) => f.sum(sum, st, v, nonempty),
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
                    snapshot!(i, sum, count);
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
                let counts = args.get(1).map(|c| counts_of(c)).transpose()?;
                let f = DecFold {
                    groups,
                    nulls: first,
                    w: counts.is_none().then_some(w),
                    bound,
                    max_sum,
                    proofs,
                };
                match decimals(args[0])? {
                    Decimals::I64(v) => f.avg(sum, count, overflow, v, counts),
                    Decimals::I128(v) => f.avg(sum, count, overflow, v, counts),
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
                            snapshot!(i, vals, set);
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
                            snapshot!(i, vals, set);
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
            (
                Kind::Pick {
                    last,
                    ignore_nulls,
                    values,
                    set,
                    null,
                    heap,
                },
                _,
            ) => {
                let a = args[0];
                let (last, ignore_nulls) = (*last, *ignore_nulls);
                // A partial state's row is taken when its `valueSet` says so
                // (its value may be NULL); a raw row unless it is a NULL to
                // skip.
                let taken = if state {
                    Some(booleans_of(args[1])?)
                } else {
                    None
                };
                let take = |i: usize, g: usize, set: &[bool]| {
                    let present = taken.map_or(valid(i) || !ignore_nulls, |t| t.value(i));
                    present && (last || !set[g])
                };
                match values {
                    Values::Int(vals) => {
                        let v = int_like(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if take(i, g, set) {
                                set[g] = true;
                                null[g] = !valid(i);
                                if valid(i) {
                                    vals[g] = v[i];
                                }
                            }
                            snapshot!(i, vals, set, null);
                        }
                    }
                    Values::Float(vals) => {
                        let v = floats(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if take(i, g, set) {
                                set[g] = true;
                                null[g] = !valid(i);
                                if valid(i) {
                                    vals[g] = v[i];
                                }
                            }
                            snapshot!(i, vals, set, null);
                        }
                    }
                    Values::Bytes(vals) => {
                        let v = byte_values(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if take(i, g, set) {
                                set[g] = true;
                                null[g] = !valid(i);
                                if valid(i) {
                                    *heap = heap.saturating_sub(vals[g].len()) + v[i].len();
                                    vals[g] = v[i].into();
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Kind::SumDec { sum, .. } | Kind::AvgDec { sum, .. } = &self.kind {
            self.max_sum = refresh_max(self.max_sum, sum, groups);
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
            Kind::SumDec { .. }
            | Kind::AvgFloat { .. }
            | Kind::AvgDec { .. }
            | Kind::Pick { .. } => 2,
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
            Kind::MinMax { values, set, .. } => {
                vec![self.value_array(values, |i| set[i], ids)?]
            }
            Kind::Pick {
                values, set, null, ..
            } => vec![
                self.value_array(values, |i| set[i] && !null[i], ids)?,
                Arc::new(BooleanArray::from_iter(
                    ids.iter().map(|i| Some(set[pick(i)])),
                )),
            ],
        })
    }

    /// A decimal state or result column of type `t` (either carrier).
    fn decimal(&self, v: impl Iterator<Item = Option<i128>>, t: &DataType) -> Result<ArrayRef> {
        match t {
            DataType::Decimal64(..) => narrow(v, t),
            _ => Ok(Arc::new(
                Decimal128Array::from_iter(v).with_data_type(t.clone()),
            )),
        }
    }

    /// `values` of groups `ids` as the result type; `on(group)` says
    /// whether the group has a value (else NULL).
    fn value_array(
        &self,
        values: &Values,
        on: impl Fn(usize) -> bool,
        ids: &[u32],
    ) -> Result<ArrayRef> {
        let on = |i: &u32| on(*i as usize);
        let at = |i: &u32| *i as usize;
        let t = &self.result;
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
                let (DataType::Decimal64(p, s) | DataType::Decimal128(p, s)) = self.result else {
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
                let (
                    DataType::Decimal64(p, s) | DataType::Decimal128(p, s),
                    DataType::Decimal64(_, ss) | DataType::Decimal128(_, ss),
                ) = (&self.result, self.expr.avg_sum_type())
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
                    } else if let Some(v) = avg_fast(sum[g], ss, count[g], *p, *s) {
                        Some(v)
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
            Kind::Count(_)
            | Kind::SumInt { .. }
            | Kind::SumFloat { .. }
            | Kind::MinMax { .. }
            | Kind::Pick { .. } => {
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
            .map_err(|_| internal(format!("value out of range of {t}")))?;
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
        DataType::Decimal64(..) => prim::<Decimal64Type>(it, t),
        _ => prim::<Decimal128Type>(it, t),
    }
}

#[cfg(test)]
#[cfg(not(loom))]
mod differential;

/// The per-batch proofs (design P2-T160) against the per-row path.
#[cfg(test)]
#[cfg(not(loom))]
mod proof_tests {
    use super::*;
    use proptest::prelude::*;

    /// `10^38 - 1`, Spark's largest decimal.
    const M: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

    type Row = (u32, i128, bool, i64);

    fn acc(f: AggregateFunction, t: &DataType, ansi: bool, proofs: bool) -> Accumulator {
        let mut a = Accumulator::new(&AggregateExpr::new(f, 0, t.clone(), "a"), ansi).unwrap();
        a.proofs = proofs;
        a
    }

    fn column(rows: &[Row], t: &DataType) -> ArrayRef {
        let v = rows.iter().map(|&(_, v, ok, _)| ok.then_some(v));
        Arc::new(Decimal128Array::from_iter(v).with_data_type(t.clone()))
    }

    fn weights(rows: &[Row], mode: u8) -> Weights {
        match mode {
            0 => Weights::Const(1),
            1 => Weights::Const(rows.first().map_or(1, |r| r.3)),
            _ => Weights::Column(Int64Array::from_iter_values(rows.iter().map(|r| r.3))),
        }
    }

    fn feed(a: &mut Accumulator, groups: u32, batch: &[Row], t: &DataType, mode: u8) {
        a.resize(groups as usize);
        let g: Vec<u32> = batch.iter().map(|r| r.0).collect();
        a.update(&g, &[column(batch, t)], &weights(batch, mode))
            .unwrap();
    }

    /// `state` and `evaluate` (value or error text) of every group.
    fn outcome(
        a: &Accumulator,
        groups: u32,
    ) -> (Vec<ArrayRef>, std::result::Result<ArrayRef, String>) {
        let ids: Vec<u32> = (0..groups).collect();
        (
            a.state(&ids).unwrap(),
            a.evaluate(&ids).map_err(|e| e.to_string()),
        )
    }

    /// The largest stored `|sum|`, which `max_sum` must bound.
    fn stored_max(a: &Accumulator) -> u128 {
        match &a.kind {
            Kind::SumDec { sum, .. } | Kind::AvgDec { sum, .. } => {
                sum.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
            }
            _ => 0,
        }
    }

    fn decimal_type() -> impl Strategy<Value = (u8, i8)> {
        prop_oneof![
            Just((38, 0)),
            Just((28, 0)),
            Just((5, 2)),
            Just((1, 0)),
            Just((18, 0)),
            Just((30, 5)),
            Just((28, 18)),
        ]
    }

    fn value(p: u8) -> impl Strategy<Value = i128> + Clone {
        let m = 10_i128.pow(u32::from(p)) - 1;
        prop_oneof![
            Just(m),
            Just(-m),
            Just(m / 2 + 1),
            Just(1),
            Just(-5),
            Just(0),
            -1000..1000_i128,
            -m..=m,
        ]
    }

    fn weight() -> impl Strategy<Value = i64> + Clone {
        prop_oneof![
            4 => Just(1_i64),
            2 => Just(-1),
            1 => -3..=3_i64,
            1 => Just(i64::MAX),
            1 => Just(i64::MIN),
            1 => any::<i64>(),
        ]
    }

    /// `(input type, groups, weights mode, batches)`.
    type Input = ((u8, i8), u32, u8, Vec<Vec<Row>>);

    fn input() -> impl Strategy<Value = Input> {
        (decimal_type(), 1..=3_u32, 0..3_u8).prop_flat_map(|(t, groups, mode)| {
            let row = (
                0..groups,
                value(t.0),
                proptest::bool::weighted(0.9),
                weight(),
            );
            // Short batches put a near-bound value alone in a batch, so
            // the next batch's proof depends on the running bound.
            let batch = prop_oneof![
                proptest::collection::vec(row.clone(), 0..3),
                proptest::collection::vec(row, 0..24),
            ];
            (
                Just(t),
                Just(groups),
                Just(mode),
                proptest::collection::vec(batch, 1..8),
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig { failure_persistence: None, ..ProptestConfig::default() })]

        /// Raw rows (with IVM weights of both signs): the proof path equals
        /// the per-row path for `sum` and `avg`, ANSI on and off.
        #[test]
        fn fold_proof_equals_per_row(((p, s), groups, mode, batches) in input()) {
            let t = DataType::Decimal128(p, s);
            for f in [AggregateFunction::Sum, AggregateFunction::Avg] {
                for ansi in [true, false] {
                    let (mut fast, mut slow) = (acc(f, &t, ansi, true), acc(f, &t, ansi, false));
                    for b in &batches {
                        feed(&mut fast, groups, b, &t, mode);
                        feed(&mut slow, groups, b, &t, mode);
                        prop_assert!(stored_max(&fast) <= fast.max_sum);
                        prop_assert!(stored_max(&slow) <= slow.max_sum);
                    }
                    prop_assert_eq!(outcome(&fast, groups), outcome(&slow, groups));
                }
            }
        }

        /// Merged partial states, overflowed ones included (sum NULL with a
        /// nonempty flag or a count): the proof path equals the per-row path
        /// in both merge orders.
        #[test]
        fn merge_proof_equals_per_row(((p, s), groups, _mode, batches) in input()) {
            let t = DataType::Decimal128(p, s);
            for f in [AggregateFunction::Sum, AggregateFunction::Avg] {
                for ansi in [true, false] {
                    let ids: Vec<u32> = (0..groups).collect();
                    let partials: Vec<Vec<ArrayRef>> = batches
                        .iter()
                        .map(|b| {
                            let mut a = acc(f, &t, ansi, false);
                            feed(&mut a, groups, b, &t, 0);
                            a.state(&ids).unwrap()
                        })
                        .collect();
                    for order in [false, true] {
                        let (mut fast, mut slow) =
                            (acc(f, &t, ansi, true), acc(f, &t, ansi, false));
                        fast.resize(groups as usize);
                        slow.resize(groups as usize);
                        let mut list: Vec<&Vec<ArrayRef>> = partials.iter().collect();
                        if order {
                            list.reverse();
                        }
                        for st in list {
                            fast.merge(&ids, st).unwrap();
                            slow.merge(&ids, st).unwrap();
                        }
                        prop_assert_eq!(outcome(&fast, groups), outcome(&slow, groups));
                    }
                }
            }
        }

        /// One `HALF_UP` division equals Spark's two roundings (scale 39,
        /// then the result scale) whenever it applies.
        #[test]
        fn avg_single_division_equals_digit_loop(
            sum in prop_oneof![Just(M), Just(-M), Just(M / 3), -M..=M, -100_000..100_000_i128],
            ss in 0..=38_i8,
            count in prop_oneof![Just(1_i64), Just(2), Just(3), Just(-7), Just(i64::MAX), Just(i64::MIN), any::<i64>()],
            p in 1..=38_u8,
            extra in 0..=8_i8,
        ) {
            prop_assume!(count != 0);
            let s = (ss + extra).min(38);
            let (slow, _) = avg_decimal(sum, ss, count, p, s);
            match avg_fast(sum, ss, count, p, s) {
                Some(v) => prop_assert_eq!(Some(v), slow),
                None => {
                    // The shortcut declines only when it does not apply or
                    // the value does not fit.
                    let k = u32::try_from(s - ss).unwrap();
                    let applies = s <= 19
                        && sum.unsigned_abs().checked_mul(10_u128.pow(k)).is_some();
                    prop_assert!(!applies || slow.is_none());
                }
            }
        }
    }

    /// Spark 4.2's grouped `sum`/`avg` over rows in a fixed order, ANSI on
    /// and off (`tests/scripts/gen_decimal_sum_avg.py`): row `i` is in group
    /// `i % groups` and in Spark's `range` partition (contiguous ranges).
    /// Each partition is folded in one batch or one row per batch (the
    /// running bound decides each proof), or on the per-row path; several
    /// partitions are merged as partial states in partition order. Every
    /// run gives Spark's values or error.
    #[test]
    fn spark_sum_avg_goldens() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/spark-4.2.0/decimal-sum-avg.json"
        );
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(cases.len(), 172);
        let num = |v: &serde_json::Value| usize::try_from(v.as_u64().unwrap()).unwrap();
        for c in &cases {
            let ansi = c["ansi"].as_bool().unwrap();
            let f = match c["fn"].as_str().unwrap() {
                "sum" => AggregateFunction::Sum,
                _ => AggregateFunction::Avg,
            };
            let ty = c["type"].as_array().unwrap();
            let p = u8::try_from(ty[0].as_u64().unwrap()).unwrap();
            let s = i8::try_from(ty[1].as_u64().unwrap()).unwrap();
            let t = DataType::Decimal128(p, s);
            let (groups, parts) = (num(&c["groups"]), num(&c["partitions"]));
            let g32 = u32::try_from(groups).unwrap();
            let rows: Vec<Row> = c["values"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let g = u32::try_from(i % groups).unwrap();
                    (g, v.as_str().unwrap().parse().unwrap(), true, 1)
                })
                .collect();
            let n = rows.len();
            let ids: Vec<u32> = (0..g32).collect();
            let mut runs = Vec::new();
            for (proofs, one_batch) in [(true, true), (true, false), (false, true)] {
                let fold = |part: &[Row], a: &mut Accumulator| {
                    if one_batch {
                        feed(a, g32, part, &t, 0);
                    } else {
                        for r in part {
                            feed(a, g32, std::slice::from_ref(r), &t, 0);
                        }
                    }
                };
                let mut a = acc(f, &t, ansi, proofs);
                a.resize(groups);
                if parts == 1 {
                    fold(&rows, &mut a);
                } else {
                    for k in 0..parts {
                        let part = &rows[k * n / parts..(k + 1) * n / parts];
                        let mut partial = acc(f, &t, ansi, proofs);
                        partial.resize(groups);
                        fold(part, &mut partial);
                        a.merge(&ids, &partial.state(&ids).unwrap()).unwrap();
                    }
                }
                runs.push(a.evaluate(&ids).map_err(|e| e.to_string()));
            }
            for got in &runs {
                match (got, c.get("error")) {
                    (Err(e), Some(want)) => {
                        assert!(e.contains(want.as_str().unwrap()), "{c}: {e}");
                    }
                    (Ok(v), None) => {
                        let DataType::Decimal128(op, os) = v.data_type() else {
                            panic!("{c}: {:?}", v.data_type());
                        };
                        assert_eq!(c["type_out"], format!("decimal({op},{os})"), "{c}");
                        let v = v.as_primitive::<Decimal128Type>();
                        let want = c["value"].as_array().unwrap();
                        assert_eq!(want.len(), groups.min(n), "{c}");
                        for (g, w) in want.iter().enumerate() {
                            let text = (!v.is_null(g)).then(|| plain(v.value(g), *os));
                            assert_eq!(w.as_str(), text.as_deref(), "{c} group {g}");
                        }
                    }
                    (got, _) => panic!("{c}: {got:?}"),
                }
            }
        }
    }

    /// Half points round away from zero, as Spark's `HALF_UP` does.
    #[test]
    fn avg_single_division_half_points() {
        assert_eq!(avg_fast(5, 0, 2, 10, 0), Some(3));
        assert_eq!(avg_fast(-5, 0, 2, 10, 0), Some(-3));
        assert_eq!(avg_fast(5, 0, -2, 10, 0), Some(-3));
        assert_eq!(avg_fast(-5, 0, 3, 5, 4), Some(-16_667));
        assert_eq!(avg_decimal(-5, 0, 3, 5, 4).0, Some(-16_667));
        // Numerator beyond u128: `M * 10^4`; the digit loop decides.
        assert_eq!(avg_fast(M, 0, 3, 38, 4), None);
        assert_eq!(avg_fast(1, 0, 3, 38, 20), None);
        assert_eq!(avg_fast(1, 2, 3, 38, 1), None);
        assert_eq!(avg_fast(1, 0, 0, 38, 1), None);
        // Does not fit: the caller takes the loop for the error text.
        assert_eq!(avg_fast(99_999, 0, 1, 5, 1), None);
    }

    /// The proof holds strictly below the bound, Σ|w|·|v| counts weights,
    /// and NULL slots do not count.
    #[test]
    fn proof_boundaries() {
        let b = Some(100_u128);
        assert!(proven(true, b, 89, &[10_i128], None, None));
        assert!(!proven(true, b, 90, &[10_i128], None, None));
        assert!(!proven(false, b, 0, &[1_i128], None, None));
        assert!(!proven(true, None, 0, &[1_i128], None, None));
        let w = Weights::Column(Int64Array::from(vec![-3_i64, 2]));
        assert_eq!(batch_abs(&[-10_i128, 5], None, Some(&w)), 40);
        assert_eq!(batch_abs(&[7_i64], None, Some(&Weights::Const(-4))), 28);
        let nulls = NullBuffer::from(vec![true, false]);
        assert_eq!(batch_abs(&[1, i128::MAX], Some(&nulls), None), 1);
        assert_eq!(batch_abs(&[i128::MIN, i128::MIN], None, None), u128::MAX);
        let big = Weights::Const(i64::MIN);
        assert_eq!(batch_abs(&[M], None, Some(&big)), u128::MAX);
    }

    /// Spark R6 shape: a partial that overflowed upstream (sum NULL,
    /// nonempty) merged with a proof-eligible batch stays overflowed.
    #[test]
    fn overflowed_partial_is_sticky_on_the_proof_path() {
        let t = DataType::Decimal128(38, 0);
        for f in [AggregateFunction::Sum, AggregateFunction::Avg] {
            let mut part = acc(f, &t, true, true);
            feed(&mut part, 1, &[(0, M, true, 1), (0, 1, true, 1)], &t, 0);
            let st = part.state(&[0]).unwrap();
            assert!(st[0].is_null(0), "{f:?} partial overflowed");
            let mut small = acc(f, &t, true, true);
            feed(&mut small, 1, &[(0, -5, true, 1)], &t, 0);
            let small = small.state(&[0]).unwrap();
            for order in [[&st, &small], [&small, &st]] {
                let mut m = acc(f, &t, true, true);
                m.resize(1);
                for s in order {
                    m.merge(&[0], s).unwrap();
                }
                let err = m.evaluate(&[0]).unwrap_err().to_string();
                assert!(err.contains("ARITHMETIC_OVERFLOW"), "{err}");
                let mut legacy = acc(f, &t, false, true);
                legacy.resize(1);
                for s in order {
                    legacy.merge(&[0], s).unwrap();
                }
                assert!(legacy.evaluate(&[0]).unwrap().is_null(0));
            }
        }
    }

    /// Spark R6: partials `M | 1 | -5` overflow when `M` meets `1` first and
    /// not when `-5` comes between; the outcome follows the merge order (not
    /// a Spark golden: Spark's merge order is its shuffle-fetch order).
    #[test]
    fn r6_partials_follow_merge_order() {
        let t = DataType::Decimal128(38, 0);
        for f in [AggregateFunction::Sum, AggregateFunction::Avg] {
            let partial = |v: i128| {
                let mut a = acc(f, &t, true, true);
                feed(&mut a, 1, &[(0, v, true, 1)], &t, 0);
                a.state(&[0]).unwrap()
            };
            let (m, one, minus5) = (partial(M), partial(1), partial(-5));
            for proofs in [true, false] {
                let merged = |order: [&Vec<ArrayRef>; 3]| {
                    let mut a = acc(f, &t, true, proofs);
                    a.resize(1);
                    for st in order {
                        a.merge(&[0], st).unwrap();
                    }
                    a.evaluate(&[0]).map_err(|e| e.to_string())
                };
                let err = merged([&m, &one, &minus5]).unwrap_err();
                assert!(err.contains("ARITHMETIC_OVERFLOW"), "{err}");
                let ok = merged([&m, &minus5, &one]);
                if f == AggregateFunction::Sum {
                    let v = ok.unwrap();
                    assert_eq!(v.as_primitive::<Decimal128Type>().value(0), M - 4);
                } else {
                    // The average (M - 4) / 3 is out of decimal(38, 4)'s range.
                    assert!(ok.unwrap_err().contains("NUMERIC_VALUE_OUT_OF_RANGE"));
                }
            }
        }
    }

    /// Spark R1 vs R3: M, 1, -5 overflows in that order (grouped), M, -5, 1
    /// does not; the proof path keeps the processing order.
    #[test]
    fn transient_overflow_follows_row_order() {
        let t = DataType::Decimal128(38, 0);
        let mut a = acc(AggregateFunction::Sum, &t, true, true);
        feed(
            &mut a,
            1,
            &[(0, M, true, 1), (0, 1, true, 1), (0, -5, true, 1)],
            &t,
            0,
        );
        assert!(a.evaluate(&[0]).is_err());
        // The same across batches: the second batch alone is provable, the
        // running bound is not.
        for f in [AggregateFunction::Sum, AggregateFunction::Avg] {
            let mut a = acc(f, &t, true, true);
            feed(&mut a, 1, &[(0, M, true, 1)], &t, 0);
            feed(&mut a, 1, &[(0, 1, true, 1)], &t, 0);
            feed(&mut a, 1, &[(0, -5, true, 1)], &t, 0);
            let err = a.evaluate(&[0]).unwrap_err().to_string();
            assert!(err.contains("ARITHMETIC_OVERFLOW"), "{err}");
        }
        let mut a = acc(AggregateFunction::Sum, &t, true, true);
        feed(
            &mut a,
            1,
            &[(0, M, true, 1), (0, -5, true, 1), (0, 1, true, 1)],
            &t,
            0,
        );
        let v = a.evaluate(&[0]).unwrap();
        assert_eq!(v.as_primitive::<Decimal128Type>().value(0), M - 4);
        // IVM: a retraction lowers the sum; with no more groups than rows
        // the bound is recomputed and shrinks with it.
        let mut a = acc(AggregateFunction::Sum, &t, true, true);
        feed(&mut a, 1, &[(0, M, true, 1)], &t, 0);
        assert_eq!(a.max_sum, M.unsigned_abs());
        feed(&mut a, 1, &[(0, M, true, -1)], &t, 2);
        assert_eq!(a.max_sum, 0);
        // Many groups, one row: only the touched group raises the bound.
        assert_eq!(refresh_max(9, &[1, -20, 3], &[2]), 9);
        assert_eq!(refresh_max(9, &[1, -20, 3], &[1]), 20);
        assert_eq!(
            quick_abs(&[3_i128, -2], Some(&Weights::Const(-2))),
            2 * 4 * 2
        );
        let w = Weights::Column(Int64Array::from(vec![1_i64, -3]));
        assert_eq!(quick_abs(&[3_i64, -2], Some(&w)), 2 * 4 * 4);
        feed(&mut a, 1, &[(0, 7, true, 1)], &t, 0);
        let v = a.evaluate(&[0]).unwrap();
        assert_eq!(v.as_primitive::<Decimal128Type>().value(0), 7);
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

    /// Beyond `i128`, the exact average rounds as Spark's (`HALF_UP` to
    /// scale 39, then to the result's scale; expected values from Python's
    /// `decimal`), and reports the scale-39 quotient when it does not fit.
    #[test]
    fn wide_averages_round_as_spark() {
        let wide = |v: i128, mul: i128| {
            i256::from_i128(v)
                .checked_mul(i256::from_i128(mul))
                .unwrap()
        };
        let three = wide(3, 10_i128.pow(38));
        assert_eq!(
            avg_decimal_wide(three, 0, 7, 38, 0).0,
            Some(42_857_142_857_142_857_142_857_142_857_142_857_143)
        );
        let (none, text) = avg_decimal_wide(three, 0, 7, 38, 2);
        assert_eq!(none, None);
        assert_eq!(
            text,
            "42857142857142857142857142857142857142.857142857142857142857142857142857142857"
        );
        let negative = wide(-5, 10_i128.pow(38)).wrapping_add(i256::from_i128(3));
        assert_eq!(
            avg_decimal_wide(negative, 0, 9, 38, 0).0,
            Some(-55_555_555_555_555_555_555_555_555_555_555_555_555)
        );
        let half = wide(2, 10_i128.pow(38)).wrapping_add(i256::ONE);
        assert_eq!(
            avg_decimal_wide(half, 0, 4, 38, 0).0,
            Some(50_000_000_000_000_000_000_000_000_000_000_000_000)
        );
        // Within `i128` it is `avg_decimal` itself.
        assert_eq!(
            avg_decimal_wide(i256::from_i128(15), 1, 2, 10, 5),
            (Some(75_000), String::new())
        );
        assert_eq!(
            avg_decimal_wide(i256::from_i128(10_i128.pow(38) - 1), 0, 1, 38, 4).0,
            None
        );
    }

    #[test]
    fn wide_sums_are_checked_when_evaluated() {
        let big = i256::from_i128(10_i128.pow(38));
        assert_eq!(wide_sum_value(None, 38, 0, true).unwrap(), None);
        assert_eq!(
            wide_sum_value(Some(i256::from_i128(-7)), 3, 1, true).unwrap(),
            Some(-7)
        );
        assert_eq!(wide_sum_value(Some(big), 38, 0, false).unwrap(), None);
        let e = wide_sum_value(Some(big.wrapping_neg()), 38, 2, true).unwrap_err();
        assert!(
            e.to_string()
                .contains("-1000000000000000000000000000000000000.00"),
            "{e}"
        );
    }

    /// A running fold keeps, after each row, the state the row-by-row fold
    /// of its partition has: every snapshot evaluates as the aggregate of
    /// the rows folded so far (NULLs and unfolded rows included), for every
    /// kind a window evaluates this way; the rest are refused.
    #[test]
    fn running_snapshots_equal_row_by_row_folds() {
        use AggregateFunction as F;
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(3),
            None,
            Some(-7),
            Some(5),
            None,
            Some(2),
            Some(9),
        ]));
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.5),
            None,
            Some(-1.25),
            Some(2.0),
            None,
            Some(8.0),
            Some(-3.5),
        ]));
        // Partition 0: rows 0..4, partition 1: rows 4..7; group 2 takes rows
        // the filter drops (rows 3 and 6); a snapshot after every row.
        let (n, parts_of) = (7, [0_u32, 0, 0, 0, 1, 1, 1]);
        let groups = [0_u32, 0, 0, 2, 1, 1, 2];
        let snapshots: Vec<u32> = (3..3 + 7).collect();
        for (col, t) in [(&ints, DataType::Int64), (&floats, DataType::Float64)] {
            for f in [
                F::Count,
                F::Sum,
                F::Avg,
                F::Min,
                F::Max,
                F::First {
                    ignore_nulls: false,
                },
                F::First { ignore_nulls: true },
                F::Last {
                    ignore_nulls: false,
                },
                F::Last { ignore_nulls: true },
            ] {
                let expr = AggregateExpr::new(f, 0, t.clone(), "x");
                let mut fast = Accumulator::new(&expr, true).unwrap();
                fast.resize(3 + n);
                // Two calls: a snapshot never needs rows of the same call.
                for (lo, hi) in [(0, 3), (3, 7)] {
                    fast.update_running(
                        &groups[lo..hi],
                        &[col.slice(lo, hi - lo)],
                        &Running {
                            parts: &parts_of[lo..hi],
                            snapshots: &snapshots[lo..hi],
                        },
                    )
                    .unwrap();
                }
                let ids: Vec<u32> = snapshots.clone();
                let got = fast.evaluate(&ids).unwrap();
                // The same rows, one at a time, evaluated after each.
                let mut slow = Accumulator::new(&expr, true).unwrap();
                slow.resize(3);
                for row in 0..n {
                    slow.update(&[groups[row]], &[col.slice(row, 1)], &Weights::Const(1))
                        .unwrap();
                    let want = slow.evaluate(&[parts_of[row]]).unwrap();
                    assert_eq!(
                        got.slice(row, 1).to_data(),
                        want.to_data(),
                        "{f:?} over {t:?}, row {row}"
                    );
                }
            }
        }
        // Rows without a snapshot only fold.
        let expr = AggregateExpr::new(F::Sum, 0, DataType::Int64, "x");
        let mut a = Accumulator::new(&expr, true).unwrap();
        a.resize(2);
        a.update_running(
            &[0, 0],
            &[ints.slice(0, 2)],
            &Running {
                parts: &[0, 0],
                snapshots: &[Running::NONE, Running::NONE],
            },
        )
        .unwrap();
        assert_eq!(
            a.evaluate(&[0])
                .unwrap()
                .as_primitive::<Int64Type>()
                .value(0),
            3
        );
        // Kinds a window evaluates elsewhere, and bad shapes, are errors.
        for (f, t) in [
            (F::Sum, DataType::Decimal128(10, 2)),
            (F::Max, DataType::Utf8View),
        ] {
            let mut a = Accumulator::new(&AggregateExpr::new(f, 0, t.clone(), "x"), true).unwrap();
            a.resize(2);
            let col: ArrayRef = Arc::new(Int64Array::from(vec![1]));
            let run = Running {
                parts: &[0],
                snapshots: &[1],
            };
            let e = a.update_running(&[0], &[col], &run).unwrap_err();
            assert!(
                e.to_string().contains("no running fold"),
                "{f:?} {t:?}: {e}"
            );
        }
        let mut a = Accumulator::new(&expr, true).unwrap();
        a.resize(2);
        let short = Running {
            parts: &[0],
            snapshots: &[],
        };
        let e = a
            .update_running(&[0, 0], &[ints.slice(0, 2)], &short)
            .unwrap_err();
        assert!(e.to_string().contains("different lengths"), "{e}");
    }
}
