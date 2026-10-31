// SPDX-License-Identifier: Apache-2.0
//! Row-at-a-time reference implementations (the oracle).
//!
//! Each function evaluates one row at a time on `Option` values with the
//! overflow policy as a runtime value, using exact wide arithmetic instead
//! of the fast paths' flags. They define the expected result of every
//! specialization in property tests, the differential fuzz target and the
//! benchmarks (where they are the "generic path"). They are deliberately
//! simple, not fast; production code uses [`crate::integer`],
//! [`crate::decimal`] and [`crate::float`].

use std::cmp::Ordering;

use arrow_array::types::{Decimal128Type, Int64Type};
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, Decimal128Array, PrimitiveArray};
use arrow_buffer::i256;

use crate::compare::CompareKind;
use crate::decimal::{DecimalOp, DecimalPlan};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::float::SparkFloat;
use crate::integer::{SparkIntegral, UnaryOp};
use crate::ops::{ArithOp, Operand};
use crate::policy::{Action, Fault, OverflowMode};
use crate::version::SparkVersion;

/// Row `i` of an operand (`None` for NULL).
fn value<T: ArrowPrimitiveType>(o: Operand<'_, T>, i: usize) -> Option<T::Native> {
    match o {
        Operand::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Operand::Scalar(v) => v,
    }
}

fn length<T: ArrowPrimitiveType>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<usize, KernelError> {
    match (l, r) {
        (Operand::Array(a), Operand::Array(b)) if a.len() != b.len() => {
            Err(KernelError::internal("operand lengths differ"))
        }
        (Operand::Array(a), _) | (_, Operand::Array(a)) => Ok(a.len()),
        _ => Err(KernelError::internal("two scalar operands")),
    }
}

fn exact_i128(op: ArithOp, a: i128, b: i128) -> Option<i128> {
    match op {
        ArithOp::Add => a.checked_add(b),
        ArithOp::Subtract => a.checked_sub(b),
        ArithOp::Multiply => a.checked_mul(b),
    }
}

/// What policy `mode` makes of a faulting row: the wrapped value, NULL or
/// the error.
fn settle<T>(
    mode: OverflowMode,
    fault: Fault,
    wrapped: T,
    error: impl FnOnce() -> KernelError,
) -> Result<Option<T>, KernelError> {
    match mode.action(fault) {
        Action::Wrap => Ok(Some(wrapped)),
        Action::Null => Ok(None),
        Action::Raise => Err(error()),
    }
}

/// Row-at-a-time loop over two operands; `f` maps each non-NULL pair.
fn rows<T: ArrowPrimitiveType, U: ArrowPrimitiveType>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    mut f: impl FnMut(usize, T::Native, T::Native) -> Result<Option<U::Native>, KernelError>,
) -> Result<PrimitiveArray<U>, KernelError> {
    let len = length(l, r)?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(match (value(l, i), value(r, i)) {
            (Some(a), Some(b)) => f(i, a, b)?,
            _ => None,
        });
    }
    Ok(out.into_iter().collect())
}

/// Reference for [`crate::integer::binary`].
///
/// # Errors
/// As the fast path.
pub fn integer_binary<T: SparkIntegral>(
    op: ArithOp,
    mode: OverflowMode,
    version: SparkVersion,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<PrimitiveArray<T>, KernelError> {
    // A plain loop: this oracle is also the benchmarks' "generic path".
    let len = length(l, r)?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let (Some(a), Some(b)) = (value(l, i), value(r, i)) else {
            out.push(None);
            continue;
        };
        // Every Spark integral fits an i128 with room for + - *.
        let exact = exact_i128(op, T::widen(a), T::widen(b)).unwrap_or_default();
        let (wrapped, overflow) = T::narrow(exact);
        out.push(match (mode, overflow) {
            (_, false) | (OverflowMode::Unchecked | OverflowMode::JavaWrap, true) => Some(wrapped),
            (OverflowMode::NullOnOverflow, true) => None,
            (OverflowMode::AnsiError, true) => {
                let ctx = KernelCtx::new().with_spark_version(version);
                return Err(crate::integer::overflow_error::<T>(&ctx, i, op, a, b));
            }
        });
    }
    Ok(out.into_iter().collect())
}

/// Reference for [`crate::integer::remainder`].
///
/// # Errors
/// As the fast path.
pub fn integer_remainder<T: SparkIntegral>(
    mode: OverflowMode,
    version: SparkVersion,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<PrimitiveArray<T>, KernelError> {
    rows(l, r, |i, a, b| {
        let (a, b) = (T::widen(a), T::widen(b));
        if b == 0 {
            return settle(mode, Fault::DivideByZero, T::Native::default(), || {
                KernelError::remainder_by_zero(i, version)
            });
        }
        // Truncated remainder: the sign of the dividend, |result| < |b|.
        Ok(Some(T::narrow(a.checked_rem(b).unwrap_or_default()).0))
    })
}

/// Reference for [`crate::integer::integral_divide`].
///
/// # Errors
/// As the fast path.
pub fn integer_divide(
    mode: OverflowMode,
    l: Operand<'_, Int64Type>,
    r: Operand<'_, Int64Type>,
) -> Result<PrimitiveArray<Int64Type>, KernelError> {
    rows(l, r, |i, a, b| {
        if b == 0 {
            return settle(mode, Fault::DivideByZero, 0, || {
                KernelError::divide_by_zero(i)
            });
        }
        // b != 0 and i64 operands: exact in i128.
        let q = i128::from(a).checked_div(i128::from(b)).unwrap_or_default();
        match Int64Type::narrow(q) {
            (v, false) => Ok(Some(v)),
            (v, true) => settle(mode, Fault::Overflow, v, || {
                KernelError::arithmetic_overflow(
                    i,
                    "Overflow in integral divide",
                    Some("try_divide"),
                )
            }),
        }
    })
}

/// Reference for [`crate::integer::unary`].
///
/// # Errors
/// As the fast path.
pub fn integer_unary<T: SparkIntegral>(
    op: UnaryOp,
    mode: OverflowMode,
    version: SparkVersion,
    a: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, KernelError> {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if a.is_null(i) {
            out.push(None);
            continue;
        }
        let v = T::widen(a.value(i));
        let exact = match op {
            UnaryOp::Negate => v.checked_neg(),
            UnaryOp::Abs => v.checked_abs(),
        };
        // Integral values never reach i128::MIN.
        out.push(match T::narrow(exact.unwrap_or_default()) {
            (n, false) => Some(n),
            (n, true) => settle(mode, Fault::Overflow, n, || {
                KernelError::arithmetic_overflow(i, version.overflow_message(T::WIDTH), None)
            })?,
        });
    }
    Ok(out.into_iter().collect())
}

/// `v` (at some scale) with its last `drop` digits removed, rounded half
/// away from zero, computed on the decimal digits.
fn round_digits(v: i256, drop: usize) -> i256 {
    let text = v.to_string();
    let (neg, digits) = text
        .strip_prefix('-')
        .map_or((false, text.as_str()), |d| (true, d));
    if drop == 0 {
        return v;
    }
    let keep = digits.len().saturating_sub(drop);
    let (kept, dropped) = digits.split_at(keep);
    let mut q = if kept.is_empty() {
        i256::ZERO
    } else {
        kept.parse().unwrap_or_default()
    };
    // A dropped run shorter than `drop` has an implicit leading 0 digit.
    if dropped.len() == drop && dropped.as_bytes().first().is_some_and(|&d| d >= b'5') {
        q = q.wrapping_add(i256::ONE);
    }
    if neg { q.wrapping_neg() } else { q }
}

/// `v` truncated (towards zero) to `digits` significant digits; returns
/// the value and how many digits were removed.
fn truncate_significant(v: i256, digits: usize) -> (i256, usize) {
    let text = v.to_string();
    let (neg, all) = text
        .strip_prefix('-')
        .map_or((false, text.as_str()), |d| (true, d));
    if all.len() <= digits {
        return (v, 0);
    }
    let kept: i256 = all
        .get(..digits)
        .unwrap_or_default()
        .parse()
        .unwrap_or_default();
    (
        if neg { kept.wrapping_neg() } else { kept },
        all.len().saturating_sub(digits),
    )
}

/// Reference for [`crate::decimal::decimal_arith`] (and
/// [`crate::decimal::decimal_binary`]): follows Spark's `Decimal` steps
/// literally on decimal digits — exact `+ - %`, products truncated to 39
/// digits and quotients to 39 decimals (`MATH_CONTEXT`, `DOWN`), then
/// `HALF_UP` to the result scale — and range-checks by counting digits.
///
/// # Errors
/// As the fast path (policy/plan mismatches excepted).
#[expect(
    clippy::arithmetic_side_effects,
    reason = "scale arithmetic on validated scales (0..=38) cannot overflow"
)]
pub fn decimal_arith(
    plan: &DecimalPlan,
    mode: OverflowMode,
    version: SparkVersion,
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
) -> Result<Decimal128Array, KernelError> {
    let len = length(l, r)?;
    let (op, (_, ls), (_, rs), (out_p, os)) = plan.types();
    let (ls, rs, os) = (ls.unsigned_abs(), rs.unsigned_abs(), os.unsigned_abs());
    let ten = i256::from_i128(10);
    let pow = |k: u8| ten.checked_pow(u32::from(k)).unwrap_or(i256::ONE);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let (Some(a), Some(b)) = (value(l, i), value(r, i)) else {
            out.push(None);
            continue;
        };
        let (wa, wb) = (i256::from_i128(a), i256::from_i128(b));
        let es = ls.max(rs);
        // (result at scale os, Spark's pre-rounding text for the error).
        let computed = match op {
            DecimalOp::Add | DecimalOp::Subtract => {
                let (x, y) = (wa.wrapping_mul(pow(es - ls)), wb.wrapping_mul(pow(es - rs)));
                let exact = if op == DecimalOp::Add {
                    x.wrapping_add(y)
                } else {
                    x.wrapping_sub(y)
                };
                Some((
                    round_digits(exact, usize::from(es - os)),
                    plan.spark_value(exact),
                ))
            }
            DecimalOp::Multiply => {
                let exact = wa.wrapping_mul(wb);
                let (t, removed) = truncate_significant(exact, 39);
                let scale = usize::from(ls + rs).saturating_sub(removed);
                let v = if scale >= usize::from(os) {
                    round_digits(t, scale - usize::from(os))
                } else {
                    t.wrapping_mul(
                        ten.checked_pow(u32::try_from(usize::from(os) - scale).unwrap_or(0))
                            .unwrap_or(i256::ONE),
                    )
                };
                Some((v, plan.spark_value(exact)))
            }
            DecimalOp::Remainder | DecimalOp::Divide | DecimalOp::IntegralDivide if b == 0 => None,
            DecimalOp::Remainder => {
                let (n, d) = (wa.wrapping_mul(pow(es - ls)), wb.wrapping_mul(pow(es - rs)));
                // Java's remainder cuts the integral quotient to 39
                // significant digits and fails if a dropped digit is not 0.
                let q = crate::decimal::quotient_text(n, d, 0);
                let q = q.trim_start_matches('-');
                if q.get(39..)
                    .is_some_and(|dropped| dropped.bytes().any(|c| c != b'0'))
                {
                    return Err(KernelError::decimal_remainder_impossible(
                        i,
                        plan.spark_operand(a, plan.types().1.1),
                        plan.spark_operand(b, plan.types().2.1),
                    ));
                }
                let v = n.checked_rem(d).unwrap_or_default();
                Some((v, plan.spark_value(v)))
            }
            DecimalOp::Divide | DecimalOp::IntegralDivide => {
                let (n, d) = (wa.wrapping_mul(pow(rs)), wb.wrapping_mul(pow(ls)));
                let q39 = crate::decimal::quotient_text(n, d, 39);
                // HALF_UP to `os` decimals reads only the first `os + 1` of
                // them (truncating the rest never moves that decision), so
                // drop the others before parsing: the full 39-decimal
                // quotient can exceed 256 bits even when the result fits.
                let (int, frac) = q39.split_once('.').unwrap_or((q39.as_str(), ""));
                let kept = frac.get(..=usize::from(os)).unwrap_or(frac);
                // Beyond 256 bits it is far outside any decimal type:
                // i256::MAX has more digits than any precision.
                let q: i256 = format!("{int}{kept}").parse().unwrap_or(i256::MAX);
                Some((round_digits(q, kept.len() - usize::from(os)), q39))
            }
        };
        let Some((v, text)) = computed else {
            out.push(settle(mode, Fault::DivideByZero, 0, || {
                if op == DecimalOp::Remainder {
                    KernelError::remainder_by_zero(i, version)
                } else {
                    KernelError::divide_by_zero(i)
                }
            })?);
            continue;
        };
        let digits = v.to_string().trim_start_matches('-').len();
        let fits = v == i256::ZERO || digits <= usize::from(out_p);
        out.push(match (fits, v.to_i128()) {
            (true, Some(v)) => Some(v),
            // Decimals never wrap: an overflowing row raises or is NULL.
            _ if mode == OverflowMode::AnsiError => {
                return Err(KernelError::decimal_out_of_range(
                    i,
                    text,
                    out_p,
                    plan.out().1,
                ));
            }
            _ => None,
        });
    }
    let a: Decimal128Array = out.into_iter().collect();
    a.with_precision_and_scale(out_p, plan.out().1)
        .map_err(|e| KernelError::internal(e.to_string()))
}

/// Reference for [`crate::decimal::decimal_integral_divide`]: the
/// quotient's integral digits, wrapped to 64 bits.
///
/// # Errors
/// As the fast path.
pub fn decimal_integral_divide(
    left: (u8, i8),
    right: (u8, i8),
    mode: OverflowMode,
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
) -> Result<PrimitiveArray<Int64Type>, KernelError> {
    let ten = i256::from_i128(10);
    let pow = |s: i8| {
        ten.checked_pow(u32::from(s.unsigned_abs()))
            .unwrap_or(i256::ONE)
    };
    rows(l, r, |i, a, b| {
        if b == 0 {
            return settle(mode, Fault::DivideByZero, 0, || {
                KernelError::divide_by_zero(i)
            });
        }
        let n = i256::from_i128(a).wrapping_mul(pow(right.1));
        let d = i256::from_i128(b).wrapping_mul(pow(left.1));
        let text = crate::decimal::quotient_text(n, d, 0);
        if text.trim_start_matches('-').len() > 38 {
            return settle(mode, Fault::Overflow, 0, || {
                KernelError::decimal_out_of_range(i, text, 38, 0)
            });
        }
        let q: i256 = text.parse().unwrap_or_default();
        // BigDecimal.longValue: the low 64 bits.
        Ok(Some(i64::from_le_bytes(
            q.to_le_bytes()
                .first_chunk::<8>()
                .copied()
                .unwrap_or_default(),
        )))
    })
}

/// Spark's `SQLOrderingUtil.compareDoubles` / `compareFloats`.
fn spark_order<F: SparkFloat>(x: F, y: F) -> Ordering {
    if x == y {
        return Ordering::Equal;
    }
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
    }
}

/// Reference for [`crate::compare::compare`] on floats.
///
/// # Errors
/// As the fast path.
pub fn compare<T>(
    kind: CompareKind,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<BooleanArray, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
{
    compare_by(kind, l, r, spark_order)
}

/// Reference for [`crate::compare::compare`] on integral, decimal, date
/// and timestamp values (their natural order).
///
/// # Errors
/// As the fast path.
pub fn compare_ord<T>(
    kind: CompareKind,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<BooleanArray, KernelError>
where
    T: ArrowPrimitiveType<Native: Ord>,
{
    compare_by(kind, l, r, |a: T::Native, b| a.cmp(&b))
}

fn compare_by<T: ArrowPrimitiveType>(
    kind: CompareKind,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ord: impl Fn(T::Native, T::Native) -> Ordering,
) -> Result<BooleanArray, KernelError> {
    let len = length(l, r)?;
    Ok((0..len)
        .map(|i| Some(kind.holds(ord(value(l, i)?, value(r, i)?))))
        .collect())
}

/// Reference for [`crate::compare::compare_bytes`]: `None` rows are NULL.
#[must_use]
pub fn compare_bytes(kind: CompareKind, l: &[Option<&[u8]>], r: &[Option<&[u8]>]) -> BooleanArray {
    l.iter()
        .zip(r)
        .map(|(a, b)| Some(kind.holds((*a)?.cmp((*b)?))))
        .collect()
}

/// Reference for [`crate::compare::null_safe`] over row values.
#[must_use]
pub fn null_safe_equal<T: PartialEq>(l: &[Option<T>], r: &[Option<T>]) -> BooleanArray {
    l.iter().zip(r).map(|(a, b)| Some(a == b)).collect()
}

/// Reference for [`crate::float::binary`], [`crate::float::divide`]
/// (`op == None`, `rem == false`) and [`crate::float::remainder`]
/// (`rem == true`).
///
/// # Errors
/// As the fast path.
pub fn float_arith<T>(
    op: Option<ArithOp>,
    rem: bool,
    mode: OverflowMode,
    version: SparkVersion,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
{
    rows(l, r, |i, a, b| match op {
        Some(op) => Ok(Some(a.arith(op, b))),
        None if b.is_zero() => settle(mode, Fault::DivideByZero, a, || {
            if rem {
                KernelError::remainder_by_zero(i, version)
            } else {
                KernelError::divide_by_zero(i)
            }
        }),
        None if rem => Ok(Some(a.rem(b))),
        None => Ok(Some(a.div(b))),
    })
}

/// Reference for [`crate::list::create`]: one list per row holding each
/// argument's value, built row-at-a-time over `Int32` arguments.
///
/// # Errors
/// As the fast path.
pub fn list_create(
    args: &[arrow_array::ArrayRef],
    rows: usize,
    contains_null: bool,
) -> Result<arrow_array::ListArray, KernelError> {
    use arrow_array::cast::AsArray as _;
    use arrow_array::types::Int32Type;
    for a in args {
        if a.len() != rows {
            return Err(KernelError::internal(format!(
                "array of {rows} rows over an argument of {} rows",
                a.len()
            )));
        }
    }
    let len = rows
        .checked_mul(args.len().max(1))
        .ok_or_else(|| KernelError::internal(format!("a list of {rows} rows")))?;
    if args.is_empty() {
        let zeros = usize::checked_add(rows, 1)
            .ok_or_else(|| KernelError::internal(format!("a list of {rows} rows")))?;
        return Ok(arrow_array::ListArray::new(
            std::sync::Arc::new(arrow_schema::Field::new(
                "element",
                arrow_schema::DataType::Null,
                contains_null,
            )),
            arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0_i32; zeros])),
            std::sync::Arc::new(arrow_array::NullArray::new(0)),
            None,
        ));
    }
    let mut values: Vec<Option<i32>> = Vec::with_capacity(len);
    let mut offsets = Vec::with_capacity(
        len.checked_add(1)
            .ok_or_else(|| KernelError::internal(format!("a list of {rows} rows")))?,
    );
    offsets.push(0_i32);
    for r in 0..rows {
        for a in args {
            let col = a.as_primitive::<Int32Type>();
            values.push(col.is_valid(r).then(|| col.value(r)));
        }
        offsets.push(
            i32::try_from(values.len())
                .map_err(|_| KernelError::internal(format!("a list of {rows} rows")))?,
        );
    }
    Ok(arrow_array::ListArray::new(
        std::sync::Arc::new(arrow_schema::Field::new(
            "element",
            arrow_schema::DataType::Int32,
            contains_null,
        )),
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
        std::sync::Arc::new(arrow_array::Int32Array::from(values)),
        None,
    ))
}

/// Nanoseconds since midnight for the text of a `TIME`, read one byte at a
/// time (the grammar of [`crate::cast::parse_time`] written as a scanner):
/// `[T]h[h]:m[m][:s[s][.digits]][ AM|PM]` inside ASCII whitespace, a fraction
/// cut to microseconds, 12-hour clock hours 1 to 12.
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "the oracle: positions are bounded by the text, fields by two digits"
)]
pub fn time_parse(text: &str) -> Option<i64> {
    let b = text
        .trim_matches(|c: char| c.is_ascii_whitespace())
        .as_bytes();
    let mut at = usize::from(b.first() == Some(&b'T'));
    let number = |at: &mut usize| -> Option<i64> {
        let start = *at;
        while *at < b.len() && *at - start < 2 && b[*at].is_ascii_digit() {
            *at += 1;
        }
        let digits = b.get(start..*at)?;
        let mut v = 0_i64;
        for d in digits {
            v = v * 10 + i64::from(d - b'0');
        }
        (!digits.is_empty()).then_some(v)
    };
    let hour = number(&mut at)?;
    if b.get(at) != Some(&b':') {
        return None;
    }
    at += 1;
    let minute = number(&mut at)?;
    let (mut second, mut nanos) = (0, 0);
    if b.get(at) == Some(&b':') {
        at += 1;
        second = number(&mut at)?;
        if b.get(at) == Some(&b'.') {
            at += 1;
            let mut scale = 100_000_000_i64;
            while at < b.len() && b[at].is_ascii_digit() {
                if scale >= 1_000 {
                    nanos += i64::from(b[at] - b'0') * scale;
                    scale /= 10;
                }
                at += 1;
            }
        }
    }
    let mut pm = None;
    if at < b.len() {
        while b.get(at).is_some_and(u8::is_ascii_whitespace) {
            at += 1;
        }
        let tail = b.get(at..)?;
        if !b[at - 1].is_ascii_whitespace() {
            return None;
        }
        pm = if tail.eq_ignore_ascii_case(b"AM") {
            Some(false)
        } else if tail.eq_ignore_ascii_case(b"PM") {
            Some(true)
        } else {
            return None;
        };
    }
    if minute > 59 || second > 59 {
        return None;
    }
    let hour = match pm {
        None if hour < 24 => hour,
        Some(is_pm) if (1..=12).contains(&hour) => hour % 12 + if is_pm { 12 } else { 0 },
        _ => return None,
    };
    Some(((hour * 60 + minute) * 60 + second) * 1_000_000_000 + nanos)
}

#[cfg(test)]
mod tests {
    use arrow_array::{Float32Array, Int64Array};

    use super::*;

    #[test]
    fn engine_errors() {
        let a = Int64Array::from(vec![1, 2]);
        let b = Int64Array::from(vec![1]);
        let add = |l, r| {
            integer_binary::<Int64Type>(
                ArithOp::Add,
                OverflowMode::AnsiError,
                SparkVersion::V4_2,
                l,
                r,
            )
        };
        assert!(add(Operand::Array(&a), Operand::Array(&b)).is_err());
        assert!(add(Operand::Scalar(Some(1)), Operand::Scalar(Some(1))).is_err());
        assert_eq!(
            add(Operand::Scalar(Some(1)), Operand::Array(&a)).unwrap(),
            Int64Array::from(vec![2, 3])
        );
    }

    #[test]
    fn float_order() {
        assert_eq!(spark_order(f32::NAN, 1.0), Ordering::Greater);
        assert_eq!(spark_order(-0.0_f64, 0.0), Ordering::Equal);
        assert_eq!(spark_order(1.0_f64, f64::NAN), Ordering::Less);
        let a = Float32Array::from(vec![f32::NAN]);
        let ge = compare(
            CompareKind::GreaterEq,
            Operand::Array(&a),
            Operand::Scalar(Some(f32::INFINITY)),
        );
        assert_eq!(ge.unwrap(), BooleanArray::from(vec![true]));
    }
}
