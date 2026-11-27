// SPDX-License-Identifier: Apache-2.0
//! Dense folds of integer `sum`, integer-like `min`/`max` and `count(col)`
//! (gap ledger §4 item 15): raw rows with unit weights or partial states, no
//! `FILTER` mask, no running window. The generic fold tests the mask, the
//! weight and the validity of every row through closures; these loops read
//! the native values and the validity bitmap directly, and a table of one
//! group (a global aggregate) folds into a register. Each gives the state the
//! per-row loop gives.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType, Int8Type, Int16Type,
    Int32Type, Int64Type, IntervalYearMonthType, Time64NanosecondType, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, IntervalUnit, TimeUnit};

/// `nulls` unless it marks no row NULL.
fn some_null(nulls: Option<&NullBuffer>) -> Option<&NullBuffer> {
    nulls.filter(|n| n.null_count() > 0)
}

/// An integer carrier `sum` folds into `i64`: `ABS_MAX` bounds `|x|` of
/// every value.
pub(super) trait SumInt: Copy + Into<i64> {
    const ABS_MAX: u64;
}

impl SumInt for i8 {
    const ABS_MAX: u64 = 1 << 7;
}
impl SumInt for i16 {
    const ABS_MAX: u64 = 1 << 15;
}
impl SumInt for i32 {
    const ABS_MAX: u64 = 1 << 31;
}
impl SumInt for i64 {
    const ABS_MAX: u64 = 1 << 63;
}

/// Adds the valid rows of `v` to their groups' `sum` (wrapping) and marks
/// the groups `seen`. Returns whether any add overflowed `i64`: the
/// per-row loop's checked add fails at the first such row (ANSI), and its
/// wrapping fallback gives these sums (non-ANSI). A NULL row adds 0, which
/// never overflows. Values are read at their own width (no widened copy).
///
/// One group: when `|sum| + len * ABS_MAX` fits `i64`, no running sum of the
/// batch can overflow (every prefix lies within that bound), so the batch is
/// summed in one reduction the compiler vectorises and added once; else the
/// per-row overflow-tracking loop runs. Both give the same `i64`. This is a
/// per-batch range proof in the spirit of DuckDB's `sum_no_overflow`, which
/// proves the bound from column statistics instead.
pub(super) fn sum_ints<T: SumInt>(
    groups: &[u32],
    v: &[T],
    nulls: Option<&NullBuffer>,
    sum: &mut [i64],
    seen: &mut [bool],
) -> bool {
    debug_assert_eq!(groups.len(), v.len(), "one group per row");
    let nulls = some_null(nulls);
    let mut overflow = false;
    if let ([s], [any]) = (&mut *sum, &mut *seen) {
        *any |= nulls.map_or(!v.is_empty(), |n| n.null_count() < v.len());
        let bound = u64::try_from(v.len())
            .ok()
            .and_then(|l| l.checked_mul(T::ABS_MAX))
            .filter(|&b| b <= i64::MAX.unsigned_abs());
        if let Some(b) = bound
            && s.unsigned_abs() <= i64::MAX.unsigned_abs() - b
        {
            // |batch| <= b and |s| + b <= i64::MAX: neither add overflows.
            *s += match nulls {
                None => v.iter().map(|&x| x.into()).sum::<i64>(),
                Some(n) => masked_sum(v, n),
            };
            return false;
        }
        let mut acc = *s;
        match nulls {
            None => {
                for &x in v {
                    let (a, o) = acc.overflowing_add(x.into());
                    acc = a;
                    overflow |= o;
                }
            }
            Some(n) => {
                for (&x, ok) in v.iter().zip(n.inner().iter()) {
                    let (a, o) = acc.overflowing_add(if ok { x.into() } else { 0 });
                    acc = a;
                    overflow |= o;
                }
            }
        }
        *s = acc;
        return overflow;
    }
    match nulls {
        None => {
            for (&g, &x) in groups.iter().zip(v) {
                let g = g as usize;
                let (a, o) = sum[g].overflowing_add(x.into());
                sum[g] = a;
                seen[g] = true;
                overflow |= o;
            }
        }
        Some(n) => {
            for ((&g, &x), ok) in groups.iter().zip(v).zip(n.inner().iter()) {
                let g = g as usize;
                let (a, o) = sum[g].overflowing_add(if ok { x.into() } else { 0 });
                sum[g] = a;
                seen[g] |= ok;
                overflow |= o;
            }
        }
    }
    overflow
}

/// The sum of the rows of `v` that `n` marks valid, a 64-row word of the
/// bitmap at a time (a NULL slot's value is masked to 0, branch-free). The
/// caller proved it cannot overflow.
fn masked_sum<T: SumInt>(v: &[T], n: &NullBuffer) -> i64 {
    let chunks = n.inner().bit_chunks();
    let (body, tail) = v.split_at(chunks.chunk_len() * 64);
    let word = |xs: &[T], bits: u64| -> i64 {
        xs.iter()
            .enumerate()
            .map(|(j, &x)| x.into() & -i64::from((bits >> j) & 1 == 1))
            .sum()
    };
    let mut t: i64 = body
        .as_chunks::<64>()
        .0
        .iter()
        .zip(chunks.iter())
        .map(|(xs, bits)| word(xs, bits))
        .sum();
    t += word(tail, chunks.remainder_bits());
    t
}

/// The native values of an integer-like column, as `min`/`max` order them
/// (`Values::Int`): integers, dates, microsecond timestamps and durations,
/// year-month intervals and decimals. `None` for any other type (booleans,
/// an untyped NULL column), which the per-row loop folds.
pub(super) enum NativeInts<'a> {
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    I128(&'a [i128]),
}

pub(super) fn native_ints(a: &ArrayRef) -> Option<NativeInts<'_>> {
    Some(match a.data_type() {
        DataType::Int8 => NativeInts::I8(a.as_primitive::<Int8Type>().values()),
        DataType::Int16 => NativeInts::I16(a.as_primitive::<Int16Type>().values()),
        DataType::Int32 => NativeInts::I32(a.as_primitive::<Int32Type>().values()),
        DataType::Date32 => NativeInts::I32(a.as_primitive::<Date32Type>().values()),
        DataType::Interval(IntervalUnit::YearMonth) => {
            NativeInts::I32(a.as_primitive::<IntervalYearMonthType>().values())
        }
        DataType::Int64 => NativeInts::I64(a.as_primitive::<Int64Type>().values()),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            NativeInts::I64(a.as_primitive::<TimestampMicrosecondType>().values())
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            NativeInts::I64(a.as_primitive::<DurationMicrosecondType>().values())
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            NativeInts::I64(a.as_primitive::<Time64NanosecondType>().values())
        }
        DataType::Decimal64(..) => NativeInts::I64(a.as_primitive::<Decimal64Type>().values()),
        DataType::Decimal128(..) => NativeInts::I128(a.as_primitive::<Decimal128Type>().values()),
        _ => return None,
    })
}

/// `min` (`max`: `MAX`) of the valid rows of `v` into their groups'
/// `vals`, setting `set`. A group's value changes only for a strictly
/// better row, as in the per-row loop (equal values are equal).
pub(super) fn extremum(
    max: bool,
    groups: &[u32],
    v: &NativeInts<'_>,
    nulls: Option<&NullBuffer>,
    vals: &mut [i128],
    set: &mut [bool],
) {
    let nulls = some_null(nulls);
    match (max, v) {
        (true, NativeInts::I8(v)) => fold_extremum::<true, _>(groups, v, nulls, vals, set),
        (true, NativeInts::I16(v)) => fold_extremum::<true, _>(groups, v, nulls, vals, set),
        (true, NativeInts::I32(v)) => fold_extremum::<true, _>(groups, v, nulls, vals, set),
        (true, NativeInts::I64(v)) => fold_extremum::<true, _>(groups, v, nulls, vals, set),
        (true, NativeInts::I128(v)) => fold_extremum::<true, _>(groups, v, nulls, vals, set),
        (false, NativeInts::I8(v)) => fold_extremum::<false, _>(groups, v, nulls, vals, set),
        (false, NativeInts::I16(v)) => fold_extremum::<false, _>(groups, v, nulls, vals, set),
        (false, NativeInts::I32(v)) => fold_extremum::<false, _>(groups, v, nulls, vals, set),
        (false, NativeInts::I64(v)) => fold_extremum::<false, _>(groups, v, nulls, vals, set),
        (false, NativeInts::I128(v)) => fold_extremum::<false, _>(groups, v, nulls, vals, set),
    }
}

#[inline]
fn better<const MAX: bool, T: Ord + Copy>(x: T, cur: T) -> bool {
    if MAX { x > cur } else { x < cur }
}

fn fold_extremum<const MAX: bool, T: Copy + Ord + Into<i128>>(
    groups: &[u32],
    v: &[T],
    nulls: Option<&NullBuffer>,
    vals: &mut [i128],
    set: &mut [bool],
) {
    debug_assert_eq!(groups.len(), v.len(), "one group per row");
    if let ([cur], [s]) = (&mut *vals, &mut *set) {
        let pick = |a: T, b: T| if better::<MAX, T>(b, a) { b } else { a };
        let best = match nulls {
            None => v.iter().copied().reduce(pick),
            Some(n) => v
                .iter()
                .zip(n.inner().iter())
                .filter_map(|(&x, ok)| ok.then_some(x))
                .reduce(pick),
        };
        if let Some(b) = best.map(Into::into)
            && (!*s || better::<MAX, i128>(b, *cur))
        {
            *cur = b;
            *s = true;
        }
        return;
    }
    let mut row = |g: u32, x: T| {
        let g = g as usize;
        let x: i128 = x.into();
        if !set[g] || better::<MAX, i128>(x, vals[g]) {
            vals[g] = x;
            set[g] = true;
        }
    };
    match nulls {
        None => groups.iter().zip(v).for_each(|(&g, &x)| row(g, x)),
        Some(n) => groups
            .iter()
            .zip(v)
            .zip(n.inner().iter())
            .filter(|&(_, ok)| ok)
            .for_each(|((&g, &x), _)| row(g, x)),
    }
}

/// The validity of a row of every argument: the union of the arguments'
/// NULLs (a row counts when no argument is NULL). `None`: no row is NULL.
pub(super) fn all_valid(nulls: &[Option<NullBuffer>]) -> Option<NullBuffer> {
    nulls
        .iter()
        .fold(None, |acc, n| NullBuffer::union(acc.as_ref(), n.as_ref()))
        .filter(|n| n.null_count() > 0)
}

/// `count` of the rows `valid` marks, each with weight `k`, into their
/// groups (wrapping, as the per-row loop).
pub(super) fn count_valid(groups: &[u32], valid: &NullBuffer, k: i64, count: &mut [i64]) {
    debug_assert_eq!(groups.len(), valid.len(), "one group per row");
    if let [c] = count {
        #[expect(clippy::cast_possible_wrap, reason = "wrapping, as the per-row adds")]
        let n = (valid.len() - valid.null_count()) as i64;
        *c = c.wrapping_add(k.wrapping_mul(n));
        return;
    }
    for (&g, ok) in groups.iter().zip(valid.inner().iter()) {
        let c = &mut count[g as usize];
        *c = c.wrapping_add(if ok { k } else { 0 });
    }
}

/// Adds the valid rows of `v` to their groups' double `sum` in row order
/// (a double sum depends on the order, so each group adds its rows as the
/// per-row loop does; a NULL row is skipped, not added as `0.0`, which
/// would turn a `-0.0` total into `0.0`), counting them in `count` (`avg`)
/// or marking them in `seen` (`sum`).
pub(super) fn sum_f64(
    groups: &[u32],
    v: &[f64],
    nulls: Option<&NullBuffer>,
    sum: &mut [f64],
    mut tally: Tally<'_>,
) {
    debug_assert_eq!(groups.len(), v.len(), "one group per row");
    let nulls = some_null(nulls);
    if sum.len() == 1 {
        let mut acc = sum[0];
        let mut n = 0_i64;
        match nulls {
            None => {
                for &x in v {
                    acc += x;
                }
                #[expect(clippy::cast_possible_wrap, reason = "a batch length")]
                let len = v.len() as i64;
                n = len;
            }
            Some(b) => {
                for (&x, ok) in v.iter().zip(b.inner().iter()) {
                    if ok {
                        acc += x;
                        n += 1;
                    }
                }
            }
        }
        sum[0] = acc;
        tally.add(0, n);
        return;
    }
    match nulls {
        None => {
            for (&g, &x) in groups.iter().zip(v) {
                sum[g as usize] += x;
                tally.add(g as usize, 1);
            }
        }
        Some(b) => {
            for ((&g, &x), ok) in groups.iter().zip(v).zip(b.inner().iter()) {
                if ok {
                    sum[g as usize] += x;
                    tally.add(g as usize, 1);
                }
            }
        }
    }
}

/// What [`sum_f64`] records of a group's rows: their count (`avg`) or that
/// it has one (`sum`).
pub(super) enum Tally<'a> {
    Count(&'a mut [i64]),
    Seen(&'a mut [bool]),
}

impl Tally<'_> {
    #[inline]
    fn add(&mut self, g: usize, n: i64) {
        match self {
            Self::Count(c) => c[g] = c[g].wrapping_add(n),
            Self::Seen(s) => s[g] |= n > 0,
        }
    }
}
