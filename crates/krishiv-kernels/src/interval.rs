// SPDX-License-Identifier: Apache-2.0
//! Year-month and day-time interval arithmetic: `+`, `-`, unary `-`, `*` and
//! `/` by a number, and the differences of dates and timestamps.
//!
//! | Situation | Error class |
//! |---|---|
//! | `+`, `-` overflow | `INTERVAL_ARITHMETIC_OVERFLOW.WITH_SUGGESTION` (`try_add`, `try_subtract`) |
//! | unary `-` overflow | `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION` |
//! | `*` overflow, or a fraction that rounds out of range or is NaN | `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION` (Spark: no class) |
//! | `/` by zero | `INTERVAL_DIVIDED_BY_ZERO` |
//! | the lowest value `/ -1` | `INTERVAL_ARITHMETIC_OVERFLOW.WITH_SUGGESTION` (`try_divide`) |
//!
//! None of these depend on ANSI. Fractions round `HALF_UP`, as Spark's
//! `DoubleMath`, `IntMath`, `LongMath` and `BigDecimal.setScale` do.

use arrow_array::builder::PrimitiveBuilder;
use arrow_array::cast::AsArray as _;
use arrow_array::types::{
    Date32Type, Decimal128Type, DurationMicrosecondType, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, IntervalYearMonthType, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrowPrimitiveType, DurationMicrosecondArray, Int32Array, IntervalYearMonthArray,
    PrimitiveArray,
};
use arrow_buffer::i256;
use arrow_schema::DataType;

use crate::cast::datetime::utc_to_local;
use crate::datetime::{TimeZoneRules, map2};
use crate::error::KernelError;
use crate::ops::Operand;

const MICROS_PER_DAY: i64 = 86_400_000_000;

/// `+` or `-` of an interval operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddOp {
    /// `+`, `try_add`.
    Add,
    /// `-`, `try_subtract`.
    Subtract,
}

impl AddOp {
    const fn function(self) -> &'static str {
        match self {
            Self::Add => "try_add",
            Self::Subtract => "try_subtract",
        }
    }
}

/// `*` or `/` of an interval by a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleOp {
    /// `*`.
    Multiply,
    /// `/`.
    Divide,
}

/// Year-month `l op r`; an overflow is NULL if `null_on_overflow`.
///
/// # Errors
/// `INTERVAL_ARITHMETIC_OVERFLOW.WITH_SUGGESTION`.
pub fn add_year_month(
    l: Operand<'_, IntervalYearMonthType>,
    r: Operand<'_, IntervalYearMonthType>,
    op: AddOp,
    null_on_overflow: bool,
) -> Result<IntervalYearMonthArray, KernelError> {
    map2(
        l,
        r,
        |a, b| match op {
            AddOp::Add => a.checked_add(b),
            AddOp::Subtract => a.checked_sub(b),
        },
        null_on_overflow,
        |row| KernelError::interval_overflow(row, Some(op.function())),
    )
}

/// Day-time `l op r`; an overflow is NULL if `null_on_overflow`.
///
/// # Errors
/// `INTERVAL_ARITHMETIC_OVERFLOW.WITH_SUGGESTION`.
pub fn add_day_time(
    l: Operand<'_, DurationMicrosecondType>,
    r: Operand<'_, DurationMicrosecondType>,
    op: AddOp,
    null_on_overflow: bool,
) -> Result<DurationMicrosecondArray, KernelError> {
    map2(
        l,
        r,
        |a, b| match op {
            AddOp::Add => a.checked_add(b),
            AddOp::Subtract => a.checked_sub(b),
        },
        null_on_overflow,
        |row| KernelError::interval_overflow(row, Some(op.function())),
    )
}

fn negate<T>(
    a: &PrimitiveArray<T>,
    f: impl Fn(T::Native) -> Option<T::Native>,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType,
{
    let mut out = PrimitiveBuilder::<T>::with_capacity(a.len());
    for (row, v) in a.iter().enumerate() {
        match v {
            Some(v) => {
                out.append_value(f(v).ok_or_else(|| KernelError::interval_overflow(row, None))?);
            }
            None => out.append_null(),
        }
    }
    Ok(out.finish())
}

/// A year-month column in a type whose end field is `unit` months: what is
/// below it is dropped.
#[must_use]
pub fn truncate_year_month(a: &IntervalYearMonthArray, unit: i32) -> IntervalYearMonthArray {
    a.unary(|v| v.checked_div(unit).map_or(v, |q| q.saturating_mul(unit)))
}

/// A day-time column in a type whose end field is `unit` microseconds: what is
/// below it is dropped.
#[must_use]
pub fn truncate_day_time(a: &DurationMicrosecondArray, unit: i64) -> DurationMicrosecondArray {
    a.unary(|v| v.checked_div(unit).map_or(v, |q| q.saturating_mul(unit)))
}

/// Unary `-` of a year-month interval.
///
/// # Errors
/// `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION`.
pub fn negate_year_month(
    a: &IntervalYearMonthArray,
) -> Result<IntervalYearMonthArray, KernelError> {
    negate(a, i32::checked_neg)
}

/// Unary `-` of a day-time interval.
///
/// # Errors
/// `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION`.
pub fn negate_day_time(
    a: &DurationMicrosecondArray,
) -> Result<DurationMicrosecondArray, KernelError> {
    negate(a, i64::checked_neg)
}

/// Spark's `ExtractANSIIntervalDays`: the whole days of a day-time interval.
#[must_use]
pub fn extract_days(a: &DurationMicrosecondArray) -> Int32Array {
    a.unary(|micros| i32::try_from(micros / MICROS_PER_DAY).unwrap_or(i32::MAX))
}

/// `date - date`, an `INTERVAL DAY`.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths or two literals.
pub fn subtract_dates(
    l: Operand<'_, Date32Type>,
    r: Operand<'_, Date32Type>,
) -> Result<DurationMicrosecondArray, KernelError> {
    map2(
        l,
        r,
        |a, b| {
            i64::from(a)
                .checked_sub(i64::from(b))?
                .checked_mul(MICROS_PER_DAY)
        },
        false,
        |row| KernelError::interval_overflow(row, None),
    )
}

/// `timestamp - timestamp`, an `INTERVAL DAY TO SECOND`: the difference of the
/// local date-times in `zone` (`None` for `TIMESTAMP_NTZ`), so a day across a
/// DST change is 24 hours.
///
/// # Errors
/// `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION` beyond a long of
/// microseconds; `INTERNAL_ERROR` for operands of different lengths or two
/// literals.
pub fn subtract_timestamps(
    l: Operand<'_, TimestampMicrosecondType>,
    r: Operand<'_, TimestampMicrosecondType>,
    zone: Option<&dyn TimeZoneRules>,
) -> Result<DurationMicrosecondArray, KernelError> {
    map2(
        l,
        r,
        |a, b| match zone {
            Some(tz) => utc_to_local(a, tz)?.checked_sub(utc_to_local(b, tz)?),
            None => a.checked_sub(b),
        },
        false,
        |row| KernelError::interval_overflow(row, None),
    )
}

/// A number an interval is scaled by.
#[derive(Clone, Copy)]
enum Number {
    /// A `tinyint`, `smallint`, `int` or `bigint`.
    Integral(i64),
    /// A `float` or `double`.
    Fraction(f64),
    /// A decimal's unscaled value and scale.
    Decimal(i128, u32),
}

/// Why a scaled value has no result.
enum Fault {
    Overflow,
    DivideOverflow,
    DivideByZero,
}

fn fault(row: usize, f: &Fault) -> KernelError {
    match f {
        Fault::Overflow => KernelError::interval_overflow(row, None),
        Fault::DivideOverflow => KernelError::interval_overflow(row, Some("try_divide")),
        Fault::DivideByZero => KernelError::interval_divided_by_zero(row),
    }
}

/// `n / d` rounded `HALF_UP`, for `d != 0`.
fn divide_half_up(n: i256, d: i256) -> Option<i256> {
    let (q, r) = (n.checked_div(d)?, n.checked_rem(d)?);
    let (r2, ad) = (
        r.checked_abs()?.checked_mul(i256::from_i128(2))?,
        d.checked_abs()?,
    );
    if r2 >= ad {
        let away = if (n < i256::ZERO) == (d < i256::ZERO) {
            i256::ONE
        } else {
            i256::MINUS_ONE
        };
        q.checked_add(away)
    } else {
        Some(q)
    }
}

/// `value op by`, in the range `lo..=hi`, the interval's whole range.
fn scale(value: i64, by: Number, op: ScaleOp, lo: i64, hi: i64) -> Result<i64, Fault> {
    let is_lowest = value == lo;
    let fits = |v: i128| match i64::try_from(v) {
        Ok(v) if (lo..=hi).contains(&v) => Ok(v),
        _ => Err(Fault::Overflow),
    };
    let rounded = |v: f64| {
        if !v.is_finite() {
            return Err(Fault::Overflow);
        }
        let v = v.round();
        // 2^63 is exact in a double; every double below it fits an `i128`.
        if v.abs() > 1.0e19 {
            return Err(Fault::Overflow);
        }
        #[expect(clippy::cast_possible_truncation, reason = "checked just above")]
        fits(v as i128)
    };
    let pow10 = |s: u32| i256::from_i128(10).checked_pow(s).ok_or(Fault::Overflow);
    let zero = match by {
        Number::Integral(n) => n == 0,
        Number::Fraction(x) => x == 0.0,
        Number::Decimal(u, _) => u == 0,
    };
    if op == ScaleOp::Divide && zero {
        return Err(Fault::DivideByZero);
    }
    match (op, by) {
        (ScaleOp::Multiply, Number::Integral(n)) => i128::from(value)
            .checked_mul(i128::from(n))
            .map_or(Err(Fault::Overflow), fits),
        #[expect(clippy::cast_precision_loss, reason = "Spark multiplies as a double")]
        (ScaleOp::Multiply, Number::Fraction(x)) => rounded(value as f64 * x),
        (ScaleOp::Multiply, Number::Decimal(u, s)) => {
            let n = i256::from_i128(i128::from(value))
                .checked_mul(i256::from_i128(u))
                .ok_or(Fault::Overflow)?;
            let q = divide_half_up(n, pow10(s)?).ok_or(Fault::Overflow)?;
            q.to_i128().map_or(Err(Fault::Overflow), fits)
        }
        (ScaleOp::Divide, Number::Integral(-1)) if is_lowest => Err(Fault::DivideOverflow),
        (ScaleOp::Divide, Number::Integral(n)) => {
            let q = divide_half_up(
                i256::from_i128(i128::from(value)),
                i256::from_i128(i128::from(n)),
            )
            .ok_or(Fault::Overflow)?;
            q.to_i128().map_or(Err(Fault::Overflow), fits)
        }
        #[expect(clippy::cast_precision_loss, reason = "Spark divides as a double")]
        (ScaleOp::Divide, Number::Fraction(x)) => rounded(value as f64 / x),
        (ScaleOp::Divide, Number::Decimal(u, s)) => {
            let n = i256::from_i128(i128::from(value))
                .checked_mul(pow10(s)?)
                .ok_or(Fault::Overflow)?;
            let q = divide_half_up(n, i256::from_i128(u)).ok_or(Fault::Overflow)?;
            q.to_i128().map_or(Err(Fault::Overflow), fits)
        }
    }
}

/// A numeric column, its type resolved once per batch.
enum Numbers<'a> {
    I8(&'a PrimitiveArray<Int8Type>),
    I16(&'a PrimitiveArray<Int16Type>),
    I32(&'a PrimitiveArray<Int32Type>),
    I64(&'a PrimitiveArray<Int64Type>),
    F32(&'a PrimitiveArray<Float32Type>),
    F64(&'a PrimitiveArray<Float64Type>),
    Decimal(&'a PrimitiveArray<Decimal128Type>, u32),
}

impl<'a> Numbers<'a> {
    fn of(by: &'a dyn Array) -> Result<Self, KernelError> {
        Ok(match by.data_type() {
            DataType::Int8 => Self::I8(by.as_primitive()),
            DataType::Int16 => Self::I16(by.as_primitive()),
            DataType::Int32 => Self::I32(by.as_primitive()),
            DataType::Int64 => Self::I64(by.as_primitive()),
            DataType::Float32 => Self::F32(by.as_primitive()),
            DataType::Float64 => Self::F64(by.as_primitive()),
            DataType::Decimal128(_, s) => Self::Decimal(
                by.as_primitive(),
                u32::try_from(*s).map_err(|_| KernelError::internal("a negative decimal scale"))?,
            ),
            other => {
                return Err(KernelError::internal(format!(
                    "an interval scaled by {other}"
                )));
            }
        })
    }

    /// The value at `i`, which is valid.
    fn at(&self, i: usize) -> Number {
        match self {
            Self::I8(a) => Number::Integral(i64::from(a.value(i))),
            Self::I16(a) => Number::Integral(i64::from(a.value(i))),
            Self::I32(a) => Number::Integral(i64::from(a.value(i))),
            Self::I64(a) => Number::Integral(a.value(i)),
            Self::F32(a) => Number::Fraction(f64::from(a.value(i))),
            Self::F64(a) => Number::Fraction(a.value(i)),
            Self::Decimal(a, s) => Number::Decimal(a.value(i), *s),
        }
    }
}

fn scale_column<T>(
    a: &PrimitiveArray<T>,
    by: &dyn Array,
    op: ScaleOp,
    null_on_overflow: bool,
    bounds: (i64, i64),
    wide: impl Fn(T::Native) -> i64,
    narrow: impl Fn(i64) -> Option<T::Native>,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType,
{
    let len = a.len().max(by.len());
    if a.len() != len && a.len() != 1 {
        return Err(KernelError::internal("operands differ in length"));
    }
    let numbers = Numbers::of(by)?;
    let mut out = PrimitiveBuilder::<T>::with_capacity(len);
    for row in 0..len {
        let i = if a.len() == 1 { 0 } else { row };
        // A one-row column stands for every row.
        let j = if by.len() == 1 { 0 } else { row };
        if a.is_null(i) || by.is_null(j) {
            out.append_null();
            continue;
        }
        match scale(wide(a.value(i)), numbers.at(j), op, bounds.0, bounds.1) {
            Ok(r) => out.append_option(narrow(r)),
            Err(_) if null_on_overflow => out.append_null(),
            Err(f) => return Err(fault(row, &f)),
        }
    }
    Ok(out.finish())
}

/// A year-month interval `*` or `/` a numeric column.
///
/// # Errors
/// As in the module docs, unless `null_on_overflow`.
pub fn scale_year_month(
    a: &IntervalYearMonthArray,
    by: &dyn Array,
    op: ScaleOp,
    null_on_overflow: bool,
) -> Result<IntervalYearMonthArray, KernelError> {
    scale_column(
        a,
        by,
        op,
        null_on_overflow,
        (i64::from(i32::MIN), i64::from(i32::MAX)),
        i64::from,
        |v| i32::try_from(v).ok(),
    )
}

/// A day-time interval `*` or `/` a numeric column.
///
/// # Errors
/// As in the module docs, unless `null_on_overflow`.
pub fn scale_day_time(
    a: &DurationMicrosecondArray,
    by: &dyn Array,
    op: ScaleOp,
    null_on_overflow: bool,
) -> Result<DurationMicrosecondArray, KernelError> {
    scale_column(
        a,
        by,
        op,
        null_on_overflow,
        (i64::MIN, i64::MAX),
        |v| v,
        Some,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Decimal128Array, Float64Array, Int32Array, Int64Array};
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn months(v: i32) -> IntervalYearMonthArray {
        IntervalYearMonthArray::from(vec![v])
    }

    fn micros(v: i64) -> DurationMicrosecondArray {
        DurationMicrosecondArray::from(vec![v])
    }

    fn ym(v: i32, by: &ArrayRef, op: ScaleOp) -> Result<i32, ErrorClass> {
        scale_year_month(&months(v), by, op, false)
            .map(|a| a.value(0))
            .map_err(|e| e.error_class())
    }

    fn int(v: i32) -> ArrayRef {
        Arc::new(Int32Array::from(vec![v]))
    }

    fn long(v: i64) -> ArrayRef {
        Arc::new(Int64Array::from(vec![v]))
    }

    fn double(v: f64) -> ArrayRef {
        Arc::new(Float64Array::from(vec![v]))
    }

    fn dec(unscaled: i128, scale: i8) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(vec![unscaled])
                .with_precision_and_scale(38, scale)
                .unwrap(),
        )
    }

    #[test]
    fn fractions_round_half_up() {
        use ScaleOp::{Divide, Multiply};
        assert_eq!(ym(1, &double(1.5), Multiply), Ok(2));
        assert_eq!(ym(-1, &double(1.5), Multiply), Ok(-2));
        assert_eq!(ym(1, &dec(15, 1), Multiply), Ok(2));
        assert_eq!(ym(-1, &dec(15, 1), Multiply), Ok(-2));
        assert_eq!(ym(3, &int(2), Divide), Ok(2));
        assert_eq!(ym(-3, &int(2), Divide), Ok(-2));
        assert_eq!(ym(1, &dec(15, 1), Divide), Ok(1));
        assert_eq!(ym(1, &double(f64::INFINITY), Divide), Ok(0));
    }

    #[test]
    fn scale_errors_are_spark_classes() {
        use ScaleOp::{Divide, Multiply};
        let overflow = ErrorClass::IntervalArithmeticOverflowWithoutSuggestion;
        let with = ErrorClass::IntervalArithmeticOverflowWithSuggestion;
        let zero = ErrorClass::IntervalDividedByZero;
        assert_eq!(ym(i32::MAX, &int(2), Multiply), Err(overflow));
        assert_eq!(ym(i32::MAX, &long(1 << 31), Multiply), Err(overflow));
        assert_eq!(ym(1, &double(f64::NAN), Multiply), Err(overflow));
        assert_eq!(ym(i32::MAX, &double(1e300), Multiply), Err(overflow));
        assert_eq!(ym(1, &int(0), Divide), Err(zero));
        assert_eq!(ym(1, &double(0.0), Divide), Err(zero));
        assert_eq!(ym(1, &dec(0, 1), Divide), Err(zero));
        assert_eq!(ym(i32::MIN, &int(-1), Divide), Err(with));
        let r = scale_day_time(&micros(i64::MIN), &long(-1), Divide, false);
        assert_eq!(r.unwrap_err().error_class(), with);
        let r = scale_day_time(&micros(i64::MAX), &long(2), Multiply, false);
        assert_eq!(r.unwrap_err().error_class(), overflow);
    }

    #[test]
    fn null_on_overflow_and_nulls_give_null() {
        let r = scale_year_month(&months(1), &int(0), ScaleOp::Divide, true).unwrap();
        assert!(r.is_null(0));
        let none: ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
        let r = scale_year_month(&months(1), &none, ScaleOp::Multiply, false).unwrap();
        assert!(r.is_null(0));
    }

    #[test]
    fn a_literal_operand_stands_for_every_row() {
        let a = IntervalYearMonthArray::from(vec![1, 2]);
        let r = scale_year_month(&a, &int(2), ScaleOp::Multiply, false).unwrap();
        assert_eq!(r.values().as_ref(), &[2, 4]);
        let by: ArrayRef = Arc::new(Int32Array::from(vec![2, 3]));
        let r = scale_year_month(&months(2), &by, ScaleOp::Multiply, false).unwrap();
        assert_eq!(r.values().as_ref(), &[4, 6]);
    }

    #[test]
    fn add_and_negate_overflow_as_spark() {
        let (a, b) = (months(i32::MAX), months(1));
        let e =
            add_year_month(Operand::Array(&a), Operand::Array(&b), AddOp::Add, false).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::IntervalArithmeticOverflowWithSuggestion
        );
        let r = add_year_month(Operand::Array(&a), Operand::Array(&b), AddOp::Add, true).unwrap();
        assert!(r.is_null(0));
        let (a, b) = (micros(5), micros(7));
        let r = add_day_time(
            Operand::Array(&a),
            Operand::Array(&b),
            AddOp::Subtract,
            false,
        )
        .unwrap();
        assert_eq!(r.value(0), -2);
        let e = negate_year_month(&months(i32::MIN)).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::IntervalArithmeticOverflowWithoutSuggestion
        );
        assert_eq!(negate_day_time(&micros(1)).unwrap().value(0), -1);
        assert!(negate_day_time(&micros(i64::MIN)).is_err());
    }

    #[test]
    fn differences_and_days() {
        use arrow_array::{Date32Array, TimestampMicrosecondArray};
        let (a, b) = (Date32Array::from(vec![30]), Date32Array::from(vec![0]));
        let r = subtract_dates(Operand::Array(&a), Operand::Array(&b)).unwrap();
        assert_eq!(r.value(0), 30 * MICROS_PER_DAY);
        let (a, b) = (
            TimestampMicrosecondArray::from(vec![9]),
            TimestampMicrosecondArray::from(vec![2]),
        );
        let r = subtract_timestamps(Operand::Array(&a), Operand::Array(&b), None).unwrap();
        assert_eq!(r.value(0), 7);
        assert_eq!(extract_days(&micros(3 * MICROS_PER_DAY)).value(0), 3);
        assert_eq!(extract_days(&micros(-MICROS_PER_DAY)).value(0), -1);
    }

    #[test]
    fn a_coarser_end_field_drops_what_is_below_it() {
        let m = IntervalYearMonthArray::from(vec![Some(13), Some(-13), None]);
        let out = truncate_year_month(&m, 12);
        assert_eq!(
            (out.value(0), out.value(1), out.is_null(2)),
            (12, -12, true)
        );
        let d = DurationMicrosecondArray::from(vec![Some(90), Some(-90), None]);
        let out = truncate_day_time(&d, 60);
        assert_eq!(
            (out.value(0), out.value(1), out.is_null(2)),
            (60, -60, true)
        );
    }
}
