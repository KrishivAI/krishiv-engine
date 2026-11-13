// SPDX-License-Identifier: Apache-2.0
//! Helpers of the accumulator: decimal folding proofs, column readers and decimal text and rounding.

use super::*;

/// `Σ |w_i|·|v_i|` over the valid rows of a batch (saturating; unit
/// weights when `w` is `None`, as for merged partial states).
pub(super) fn batch_abs<N: DecimalNative>(
    v: &[N],
    nulls: Option<&NullBuffer>,
    w: Option<&Weights>,
) -> u128 {
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

/// [`proven_moves`] as a yes or no (the proof tests' form).
#[cfg(test)]
pub(super) fn proven<N: DecimalNative>(
    enabled: bool,
    bound: Option<u128>,
    max_sum: u128,
    v: &[N],
    nulls: Option<&NullBuffer>,
    w: Option<&Weights>,
) -> bool {
    proven_moves(enabled, bound, max_sum, v, nulls, w).is_some()
}

/// The per-batch overflow proof of a decimal `sum`/`avg` fold (design
/// P2-T160): every stored sum is within `max_sum` and the batch moves any
/// one group's sum by at most `Σ |w|·|v|`, so if `max_sum + Σ |w|·|v|` is
/// below the bound, no running sum of any group leaves the bound during
/// the batch, in any row order. The fold may then add with wrapping
/// arithmetic and no range test, and its result is exactly the per-row
/// path's. Proofs are local to one batch; the next batch is tested afresh.
/// Returns the bound of `Σ |w|·|v|` the proof used (over every slot when
/// `nulls` is `None`), or `None` if the batch is not proven.
pub(super) fn proven_moves<N: DecimalNative>(
    enabled: bool,
    bound: Option<u128>,
    max_sum: u128,
    v: &[N],
    nulls: Option<&NullBuffer>,
    w: Option<&Weights>,
) -> Option<u128> {
    let b = bound.filter(|_| enabled)?;
    let quick = quick_abs(v, w);
    if max_sum.saturating_add(quick) < b {
        return Some(quick);
    }
    let exact = batch_abs(v, nulls, w);
    (max_sum.saturating_add(exact) < b).then_some(exact)
}

/// Whether a batch whose `Σ |v|` is at most `moves` (see [`proven_moves`])
/// can be totalled on 64-bit lanes: every value and every partial total of
/// any group then fits an `i64`, so the 64-bit wrapping adds are exact.
pub(super) fn fits64_moves(moves: u128) -> bool {
    moves <= u128::from(i64::MAX.cast_unsigned())
}

/// A cheap upper bound of [`batch_abs`]: rows × max |v| × max |w|, with
/// each maximum bounded by an OR of ones'-complement magnitudes (`|x| <=
/// OR + 1`, branch-free) over every slot, NULL ones included. When it is
/// too loose (values near the bound, large values under NULLs) the caller takes the exact sum.
pub(super) fn quick_abs<N: DecimalNative>(v: &[N], w: Option<&Weights>) -> u128 {
    let or = N::or_magnitudes(v).saturating_add(1);
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
pub(super) fn refresh_max(old: u128, sum: &[i128], groups: &[u32]) -> u128 {
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
pub(super) struct DecFold<'a> {
    pub(super) groups: &'a [u32],
    /// The validity of the value column.
    pub(super) nulls: Option<&'a NullBuffer>,
    /// Rows the mask drops (`FILTER (WHERE ...)`); `None` reads all rows.
    pub(super) mask: Option<&'a BooleanArray>,
    /// Row weights; `None` for partial states.
    pub(super) w: Option<&'a Weights>,
    pub(super) bound: Option<u128>,
    pub(super) max_sum: u128,
    pub(super) proofs: bool,
}

impl DecFold<'_> {
    pub(super) fn valid(&self, i: usize) -> bool {
        self.nulls.is_none_or(|n| n.is_valid(i))
            && self.mask.is_none_or(|m| m.is_valid(i) && m.value(i))
    }

    /// Raw rows (not partial states) with unit weights, no NULL value and
    /// no `FILTER` mask: every row adds its value once.
    pub(super) fn dense_unit(&self) -> bool {
        matches!(self.w, Some(Weights::Const(1)))
            && self.mask.is_none()
            && self.nulls.is_none_or(|n| n.null_count() == 0)
    }

    pub(super) fn fits(&self, s: i128) -> bool {
        self.bound.is_none_or(|b| s.unsigned_abs() < b)
    }

    /// Whether the batch takes the proof path, and the bound it proved (see
    /// [`proven_moves`]).
    pub(super) fn proven<N: DecimalNative>(&self, v: &[N]) -> Option<u128> {
        proven_moves(self.proofs, self.bound, self.max_sum, v, self.nulls, self.w)
    }

    /// `sum` (`st`: 0 no value, 1 a value, 2 overflowed); a partial state
    /// has a `nonempty` column.
    pub(super) fn sum<N: DecimalNative>(
        &self,
        sum: &mut [i128],
        st: &mut [u8],
        v: &[N],
        nonempty: Option<&BooleanArray>,
    ) {
        let moves = self.proven(v);
        let fast = moves.is_some();
        if let Some(m) = moves.filter(|_| self.dense_unit()) {
            debug_assert_eq!(v.len(), self.groups.len(), "one value per group id");
            small_fold::dense_sum(self.groups, v, sum, st, fits64_moves(m));
            return;
        }
        for (i, &g) in self.groups.iter().enumerate() {
            let g = g as usize;
            let x = v[i].wide();
            match self.w {
                None => {
                    // A NULL `nonempty` is an unmatched outer join row's empty
                    // state, not an overflow.
                    if nonempty.is_some_and(|b| b.is_valid(i) && b.value(i)) {
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
    pub(super) fn avg<N: DecimalNative>(
        &self,
        sum: &mut [i128],
        count: &mut [i64],
        overflow: &mut [bool],
        v: &[N],
        counts: Option<&[i64]>,
    ) {
        let moves = self.proven(v);
        let fast = moves.is_some();
        if let Some(m) = moves.filter(|_| counts.is_none() && self.dense_unit()) {
            debug_assert_eq!(v.len(), self.groups.len(), "one value per group id");
            // As in `sum`: raw rows, all valid, unit weights, proven.
            small_fold::dense_avg(self.groups, v, sum, count, fits64_moves(m));
            return;
        }
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
pub(super) fn weight(w: &Weights, i: usize) -> i64 {
    match w {
        Weights::Const(c) => *c,
        Weights::Column(c) => c.value(i),
    }
}

pub(super) fn is_unit(w: &Weights) -> bool {
    match w {
        Weights::Const(c) => *c == 1,
        Weights::Column(c) => c.values().iter().all(|&v| v == 1),
    }
}

pub(super) fn arg(cols: &[ArrayRef], i: usize) -> Result<&ArrayRef> {
    cols.get(i)
        .ok_or_else(|| internal(format!("aggregate argument {i} out of range")))
}

pub(super) fn ints(a: &ArrayRef) -> Result<Cow<'_, [i64]>> {
    Ok(match a.data_type() {
        DataType::Int64 => Cow::Borrowed(a.as_primitive::<Int64Type>().values()),
        DataType::Int32 => widen(a.as_primitive::<Int32Type>()),
        DataType::Int16 => widen(a.as_primitive::<Int16Type>()),
        DataType::Int8 => widen(a.as_primitive::<Int8Type>()),
        // An untyped NULL column: every row is NULL, so no value is read.
        DataType::Null => Cow::Owned(vec![0; a.len()]),
        t => return Err(internal(format!("integral argument of type {t}"))),
    })
}

/// An ANSI interval's months or microseconds; a partial state holds them as
/// `bigint`.
pub(super) fn interval_values(a: &ArrayRef) -> Result<Cow<'_, [i64]>> {
    Ok(match a.data_type() {
        DataType::Interval(IntervalUnit::YearMonth) => {
            widen(a.as_primitive::<IntervalYearMonthType>())
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            Cow::Borrowed(a.as_primitive::<DurationMicrosecondType>().values())
        }
        _ => ints(a)?,
    })
}

/// `a + b` within the range of the interval family: a year-month value is an
/// `int` of months, whatever the width of its running sum.
pub(super) fn interval_add(a: i64, b: i64, months: bool) -> Option<i64> {
    a.checked_add(b)
        .filter(|s| !months || i32::try_from(*s).is_ok())
}

/// `sum / count` rounded `HALF_UP`, as Spark's `DivideYMInterval` and
/// `DivideDTInterval` divide an interval `avg`; `count` is positive.
pub(super) fn interval_mean(sum: i64, count: i64) -> i128 {
    let (s, c) = (i128::from(sum), i128::from(count));
    let (q, r) = (s / c, s % c);
    if r.abs() * 2 >= c { q + s.signum() } else { q }
}

pub(super) fn widen<T>(a: &PrimitiveArray<T>) -> Cow<'static, [i64]>
where
    T: arrow_array::ArrowPrimitiveType,
    T::Native: Into<i64>,
{
    Cow::Owned(a.values().iter().map(|&v| v.into()).collect())
}

pub(super) fn floats(a: &ArrayRef) -> Result<Cow<'_, [f64]>> {
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
pub(super) fn counts_of(a: &ArrayRef) -> Result<Cow<'_, [i64]>> {
    let c = a
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| internal(format!("count state of type {}", a.data_type())))?;
    // A NULL state is the empty state of an unmatched outer join row: it
    // counts nothing, whatever its slot holds.
    Ok(if c.null_count() == 0 {
        Cow::Borrowed(c.values().as_ref())
    } else {
        Cow::Owned(c.iter().map(|v| v.unwrap_or(0)).collect())
    })
}

/// A moment aggregate's state column: `double`s, none NULL (a buffer is
/// never NULL; a NULL would silently skip a partial's rows).
pub(super) fn state_doubles(a: &ArrayRef) -> Result<&[f64]> {
    match a.as_primitive_opt::<Float64Type>() {
        Some(c) if c.null_count() == 0 => Ok(c.values().as_ref()),
        _ => Err(internal(format!(
            "moment state of type {} with NULLs or not double",
            a.data_type()
        ))),
    }
}

/// A `boolean` state column.
pub(super) fn booleans_of(a: &ArrayRef) -> Result<&BooleanArray> {
    a.as_boolean_opt()
        .ok_or_else(|| internal(format!("boolean state of type {}", a.data_type())))
}

/// The values of a decimal column of either carrier.
#[derive(Clone, Copy)]
pub(super) enum Decimals<'a> {
    I64(&'a [i64]),
    I128(&'a [i128]),
}

pub(super) fn decimals(a: &ArrayRef) -> Result<Decimals<'_>> {
    match a.data_type() {
        DataType::Decimal64(..) => Ok(Decimals::I64(a.as_primitive::<Decimal64Type>().values())),
        DataType::Decimal128(..) => Ok(Decimals::I128(a.as_primitive::<Decimal128Type>().values())),
        t => Err(internal(format!("decimal argument of type {t}"))),
    }
}

/// `min`/`max` input as `i128`s (integral, boolean, decimal, temporal).
pub(super) fn int_like(a: &ArrayRef) -> Result<Cow<'_, [i128]>> {
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
        DataType::Interval(IntervalUnit::YearMonth) | DataType::Duration(TimeUnit::Microsecond) => {
            conv(interval_values(a)?.iter().map(|&v| i128::from(v)).collect())
        }
        _ => conv(ints(a)?.iter().map(|&v| i128::from(v)).collect()),
    })
}

/// Byte values of a string or binary array (empty under NULL slots).
pub(super) fn byte_values(a: &ArrayRef) -> Result<Vec<&[u8]>> {
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
pub(super) fn value_bytes(a: &ArrayRef) -> usize {
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
pub(super) fn float_lt(a: f64, b: f64) -> bool {
    !a.is_nan() && (b.is_nan() || a < b)
}

/// `10^p` for `p ≤ 38`.
pub(super) fn pow10(p: u8) -> i128 {
    10i128.pow(u32::from(p))
}

/// Spark's `toPlainString` of an unscaled decimal.
pub(crate) fn plain(v: i128, scale: i8) -> String {
    let s = usize::try_from(scale).unwrap_or(0);
    plain_digits(&v.unsigned_abs().to_string(), s, v < 0)
}

/// `digits` (an unscaled magnitude) with `scale` digits after the point.
pub(super) fn plain_digits(digits: &str, scale: usize, negative: bool) -> String {
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
pub(super) fn avg_decimal(
    sum: i128,
    sum_scale: i8,
    count: i64,
    p: u8,
    s: i8,
) -> (Option<i128>, String) {
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
pub(super) fn avg_fast(sum: i128, sum_scale: i8, count: i64, p: u8, s: i8) -> Option<i128> {
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
pub(super) fn round_up(digits: &mut [u8]) -> bool {
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
