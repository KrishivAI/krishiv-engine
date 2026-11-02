// SPDX-License-Identifier: Apache-2.0
//! Casts between year-month or day-time intervals and numbers. Spark 4.2.0 raises `CAST_OVERFLOW` with ANSI on or off;
//! only `TRY_CAST` gives NULL (`null_on_overflow`).
//!
//! A value counts in its type's end field: `unit` is that field's size in the
//! carrier (months or microseconds). Fractions round `HALF_UP`, as
//! `BigDecimal.setScale` does.

use std::sync::Arc;

use arrow_array::builder::PrimitiveBuilder;
use arrow_array::cast::AsArray as _;
use arrow_array::types::{
    ArrowPrimitiveType, Decimal64Type, Decimal128Type, DurationMicrosecondType, Int8Type,
    Int16Type, Int32Type, Int64Type, IntervalYearMonthType,
};
use arrow_array::{
    Array, ArrayRef, DurationMicrosecondArray, IntervalYearMonthArray, PrimitiveArray,
};
use arrow_buffer::i256;
use arrow_schema::DataType;

use crate::error::KernelError;

const MICROS_PER_SECOND: i64 = 1_000_000;

/// How an overflow reads in `CAST_OVERFLOW`.
#[derive(Clone, Copy)]
pub struct CastNames<'a> {
    /// Spark's SQL type name of the source, such as `INTERVAL DAY`.
    pub from: &'a str,
    /// Spark's SQL type name of the target.
    pub to: &'a str,
    /// A source value (the carrier's) as SQL text; unused for a number source.
    pub text: &'a dyn Fn(i64) -> String,
}

impl std::fmt::Debug for CastNames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CastNames")
            .field("from", &self.from)
            .field("to", &self.to)
            .finish_non_exhaustive()
    }
}

/// The carrier values of a year-month or day-time column.
fn carrier(a: &dyn Array) -> Result<PrimitiveArray<Int64Type>, KernelError> {
    match a.data_type() {
        DataType::Interval(_) => Ok(a.as_primitive::<IntervalYearMonthType>().unary(i64::from)),
        DataType::Duration(_) => Ok(a
            .as_primitive::<DurationMicrosecondType>()
            .reinterpret_cast::<Int64Type>()),
        other => Err(KernelError::internal(format!(
            "an interval cast of {other}"
        ))),
    }
}

/// Each value through `f`; `None` is an overflow.
fn map_values<T: ArrowPrimitiveType>(
    values: &PrimitiveArray<Int64Type>,
    null_on_overflow: bool,
    names: CastNames<'_>,
    f: impl Fn(i64) -> Option<T::Native>,
) -> Result<PrimitiveArray<T>, KernelError> {
    let mut out = PrimitiveBuilder::<T>::with_capacity(values.len());
    for (row, v) in values.iter().enumerate() {
        let Some(v) = v else {
            out.append_null();
            continue;
        };
        match f(v) {
            Some(n) => out.append_value(n),
            None if null_on_overflow => out.append_null(),
            None => {
                return Err(KernelError::cast_overflow(
                    row,
                    (names.text)(v),
                    names.from,
                    names.to,
                ));
            }
        }
    }
    Ok(out.finish())
}

/// A year-month or day-time column as `to`, an integral type: truncated to
/// whole `unit`s.
///
/// # Errors
/// `CAST_OVERFLOW`, unless `null_on_overflow`.
pub fn interval_to_integral(
    a: &dyn Array,
    unit: i64,
    to: &DataType,
    null_on_overflow: bool,
    names: CastNames<'_>,
) -> Result<ArrayRef, KernelError> {
    let v = carrier(a)?;
    Ok(match to {
        DataType::Int8 => Arc::new(map_values::<Int8Type>(&v, null_on_overflow, names, |x| {
            i8::try_from(x.checked_div(unit)?).ok()
        })?),
        DataType::Int16 => Arc::new(map_values::<Int16Type>(&v, null_on_overflow, names, |x| {
            i16::try_from(x.checked_div(unit)?).ok()
        })?),
        DataType::Int32 => Arc::new(map_values::<Int32Type>(&v, null_on_overflow, names, |x| {
            i32::try_from(x.checked_div(unit)?).ok()
        })?),
        DataType::Int64 => Arc::new(map_values::<Int64Type>(&v, null_on_overflow, names, |x| {
            x.checked_div(unit)
        })?),
        other => {
            return Err(KernelError::internal(format!(
                "an interval cast to {other}"
            )));
        }
    })
}

/// `10^n`, for a decimal scale in range.
fn ten_pow(n: u32) -> Option<i128> {
    10_i128.checked_pow(n)
}

/// `v / 10^down`, rounding `HALF_UP`.
fn divide_half_up(v: i128, down: u32) -> Option<i128> {
    let d = ten_pow(down)?;
    let (q, r) = (v.checked_div(d)?, v.checked_rem(d)?);
    if r.unsigned_abs().checked_mul(2)? >= d.unsigned_abs() {
        q.checked_add(v.signum())
    } else {
        Some(q)
    }
}

/// `unscaled` at scale `from` as an unscaled value at scale `to`, if it fits
/// `precision` digits.
fn rescale(unscaled: i128, from: i8, precision: u8, to: i8) -> Option<i128> {
    let shift = i32::from(to).saturating_sub(i32::from(from));
    let out = if shift >= 0 {
        unscaled.checked_mul(ten_pow(shift.unsigned_abs())?)?
    } else {
        divide_half_up(unscaled, shift.unsigned_abs())?
    };
    (out.unsigned_abs() < ten_pow(u32::from(precision))?.unsigned_abs()).then_some(out)
}

/// Microseconds as seconds with all six digits, as Spark's message shows them.
fn micros_text(micros: i64) -> String {
    let sign = if micros < 0 { "-" } else { "" };
    let abs = micros.unsigned_abs();
    let per_second = MICROS_PER_SECOND.unsigned_abs();
    let (whole, fraction) = (
        abs.checked_div(per_second).unwrap_or(0),
        abs.checked_rem(per_second).unwrap_or(0),
    );
    format!("{sign}{whole}.{fraction:06}")
}

/// A year-month or day-time column as `decimal(precision, scale)`, counting
/// `unit`s; only a day-time value ending in SECOND keeps a fraction.
///
/// # Errors
/// `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION`, ANSI on or off, unless
/// `null_on_overflow`.
pub fn interval_to_decimal(
    a: &dyn Array,
    unit: i64,
    to: &DataType,
    null_on_overflow: bool,
) -> Result<ArrayRef, KernelError> {
    let (DataType::Decimal64(precision, scale) | DataType::Decimal128(precision, scale)) = *to
    else {
        return Err(KernelError::internal(format!("an interval cast to {to}")));
    };
    let v = carrier(a)?;
    let seconds = matches!(a.data_type(), DataType::Duration(_)) && unit == MICROS_PER_SECOND;
    let mut out = PrimitiveBuilder::<Decimal128Type>::with_capacity(v.len());
    for (row, x) in v.iter().enumerate() {
        let Some(x) = x else {
            out.append_null();
            continue;
        };
        let counted = x.checked_div(unit).unwrap_or(x);
        let cast = if seconds {
            rescale(i128::from(x), 6, precision, scale)
        } else {
            rescale(i128::from(counted), 0, precision, scale)
        };
        match cast {
            Some(n) => out.append_value(n),
            None if null_on_overflow => out.append_null(),
            None => {
                let text = if seconds {
                    micros_text(x)
                } else {
                    counted.to_string()
                };
                return Err(KernelError::decimal_out_of_range(
                    row, text, precision, scale,
                ));
            }
        }
    }
    let out = out.finish();
    let out = out
        .with_precision_and_scale(precision, scale)
        .map_err(|e| KernelError::internal(e.to_string()))?;
    if matches!(to, DataType::Decimal128(..)) {
        return Ok(Arc::new(out));
    }
    // `rescale` kept the value within a precision that fits this carrier.
    let out = out
        .try_unary::<_, Decimal64Type, _>(|v| {
            i64::try_from(v).map_err(|e| arrow_schema::ArrowError::CastError(e.to_string()))
        })
        .and_then(|d| d.with_precision_and_scale(precision, scale))
        .map_err(|e| KernelError::internal(e.to_string()))?;
    Ok(Arc::new(out))
}

/// A number's values as unscaled decimals, with their scale.
fn number_values(a: &dyn Array) -> Result<(PrimitiveArray<Decimal128Type>, i8), KernelError> {
    Ok(match a.data_type() {
        DataType::Int8 => (a.as_primitive::<Int8Type>().unary(i128::from), 0),
        DataType::Int16 => (a.as_primitive::<Int16Type>().unary(i128::from), 0),
        DataType::Int32 => (a.as_primitive::<Int32Type>().unary(i128::from), 0),
        DataType::Int64 => (a.as_primitive::<Int64Type>().unary(i128::from), 0),
        DataType::Decimal64(_, s) => (a.as_primitive::<Decimal64Type>().unary(i128::from), *s),
        DataType::Decimal128(_, s) => (a.as_primitive::<Decimal128Type>().clone(), *s),
        other => {
            return Err(KernelError::internal(format!(
                "an interval cast of {other}"
            )));
        }
    })
}

/// `v` to 34 significant digits, half to even (Scala's `DECIMAL128`).
fn to_34_digits(v: i256) -> Option<i256> {
    let abs = v.checked_abs()?;
    let mut limit = i256::from_i128(ten_pow(34)?);
    let mut unit = i256::ONE;
    let ten = i256::from_i128(10);
    while abs >= limit {
        limit = limit.checked_mul(ten)?;
        unit = unit.checked_mul(ten)?;
    }
    if unit == i256::ONE {
        return Some(v);
    }
    let (q, r) = (abs.checked_div(unit)?, abs.checked_rem(unit)?);
    let twice = r.checked_mul(i256::from_i128(2))?;
    let odd = q.checked_rem(i256::from_i128(2))? == i256::ONE;
    let q = if twice > unit || (twice == unit && odd) {
        q.checked_add(i256::ONE)?
    } else {
        q
    };
    let rounded = q.checked_mul(unit)?;
    if v < i256::ZERO {
        rounded.checked_neg()
    } else {
        Some(rounded)
    }
}

/// The carrier value of `unscaled`, at `scale`, in `unit`s.
fn number_as_carrier(unscaled: i128, scale: i8, unit: i64) -> Option<i64> {
    // A `decimal(38, s)` times a day's microseconds can pass `i128` however small
    // the result, so the product is wide.
    let v = i256::from_i128(unscaled).checked_mul(i256::from_i128(i128::from(unit)))?;
    // Spark's product has 34 significant digits, so a fraction just under a half
    // can reach one. With nothing to multiply by, there is no product to round.
    let v = if unit == 1 { v } else { to_34_digits(v)? };
    let power = i256::from_i128(ten_pow(u32::from(scale.unsigned_abs()))?);
    let v = if scale >= 0 {
        crate::interval::divide_half_up(v, power)?
    } else {
        v.checked_mul(power)?
    };
    i64::try_from(v.to_i128()?).ok()
}

/// SQL text of a number for `CAST_OVERFLOW`.
fn number_text(unscaled: i128, a: &dyn Array) -> String {
    match a.data_type() {
        DataType::Int8 => format!("{unscaled}Y"),
        DataType::Int16 => format!("{unscaled}S"),
        DataType::Int64 => format!("{unscaled}L"),
        DataType::Decimal64(_, s) | DataType::Decimal128(_, s) => {
            let digits = unscaled.unsigned_abs().to_string();
            let sign = if unscaled < 0 { "-" } else { "" };
            let scale = usize::from(s.unsigned_abs());
            if *s <= 0 || scale == 0 {
                format!("{sign}{digits}BD")
            } else {
                let digits = format!("{digits:0>width$}", width = scale.saturating_add(1));
                let (whole, fraction) = digits.split_at(digits.len().saturating_sub(scale));
                format!("{sign}{whole}.{fraction}BD")
            }
        }
        _ => unscaled.to_string(),
    }
}

/// Runs `f` over a number column's unscaled values, row by row.
fn from_number<T: ArrowPrimitiveType>(
    a: &dyn Array,
    null_on_overflow: bool,
    names: CastNames<'_>,
    f: impl Fn(i128, i8) -> Option<T::Native>,
) -> Result<PrimitiveArray<T>, KernelError> {
    let (values, scale) = number_values(a)?;
    let mut out = PrimitiveBuilder::<T>::with_capacity(values.len());
    for (row, v) in values.iter().enumerate() {
        let Some(v) = v else {
            out.append_null();
            continue;
        };
        match f(v, scale) {
            Some(n) => out.append_value(n),
            None if null_on_overflow => out.append_null(),
            None => {
                return Err(KernelError::cast_overflow(
                    row,
                    number_text(v, a),
                    names.from,
                    names.to,
                ));
            }
        }
    }
    Ok(out.finish())
}

/// An integral or decimal column as a year-month interval whose end field is
/// `unit` months.
///
/// # Errors
/// `CAST_OVERFLOW`, unless `null_on_overflow`.
pub fn number_to_year_month(
    a: &dyn Array,
    unit: i64,
    null_on_overflow: bool,
    names: CastNames<'_>,
) -> Result<IntervalYearMonthArray, KernelError> {
    from_number::<IntervalYearMonthType>(a, null_on_overflow, names, |v, scale| {
        i32::try_from(number_as_carrier(v, scale, unit)?).ok()
    })
}

/// An integral or decimal column as a day-time interval whose end field is
/// `unit` microseconds.
///
/// # Errors
/// `CAST_OVERFLOW`, unless `null_on_overflow`.
pub fn number_to_day_time(
    a: &dyn Array,
    unit: i64,
    null_on_overflow: bool,
    names: CastNames<'_>,
) -> Result<DurationMicrosecondArray, KernelError> {
    from_number::<DurationMicrosecondType>(a, null_on_overflow, names, |v, scale| {
        number_as_carrier(v, scale, unit)
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{Decimal128Array, Int64Array};

    use super::*;

    /// Each from a Spark 4.2.0 run.
    #[test]
    fn a_high_scale_decimal_scales_without_an_intermediate_overflow() {
        const SECOND: i64 = 1_000_000;
        const DAY: i64 = 86_400 * SECOND;
        let thirty_digits = 1_123_456_789_012_345_678_901_234_567_890_i128;
        assert_eq!(
            number_as_carrier(thirty_digits, 30, SECOND),
            Some(1_123_457)
        );
        assert_eq!(
            number_as_carrier(-thirty_digits, 30, SECOND),
            Some(-1_123_457)
        );
        assert_eq!(
            number_as_carrier(thirty_digits, 30, DAY),
            Some(97_066_666_571)
        );
        assert_eq!(number_as_carrier(5, 38, DAY), Some(0));
        assert_eq!(
            number_as_carrier(9_223_372_036_854_775_807, 6, SECOND),
            Some(i64::MAX)
        );
        assert_eq!(
            number_as_carrier(92_233_720_368_547_758_075, 7, SECOND),
            None
        );
        assert_eq!(number_as_carrier(1_067_519_915, 1, DAY), None);
    }

    /// Each from a Spark 4.2.0 run.
    #[test]
    fn a_product_past_34_digits_rounds_there_first() {
        const SECOND: i64 = 1_000_000;
        // 1.0000004999999999999999999999999999999
        let just_under = 10_000_004_999_999_999_999_999_999_999_999_999_999_i128;
        assert_eq!(number_as_carrier(just_under, 37, SECOND), Some(1_000_001));
        assert_eq!(number_as_carrier(-just_under, 37, SECOND), Some(-1_000_001));
        // 5165.2083333333333333333333333333333330
        let months = 51_652_083_333_333_333_333_333_333_333_333_333_330_i128;
        assert_eq!(number_as_carrier(months, 34, 12), Some(61_983));
        // 9223372036854.7758074999999999999999999
        let top = 92_233_720_368_547_758_074_999_999_999_999_999_999_i128;
        assert_eq!(number_as_carrier(top, 25, SECOND), None);
        assert_eq!(number_as_carrier(-top, 25, SECOND), Some(i64::MIN));
        let below = 92_233_720_368_547_758_084_999_999_999_999_999_999_i128;
        assert_eq!(number_as_carrier(-below, 25, SECOND), None);
        // 10.49999999999999999999999999999999999 to INTERVAL MONTH: nothing to round.
        let one_unit = 1_049_999_999_999_999_999_999_999_999_999_999_999_i128;
        assert_eq!(number_as_carrier(one_unit, 35, 1), Some(10));
    }

    fn names() -> CastNames<'static> {
        CastNames {
            from: "INTERVAL DAY",
            to: "TINYINT",
            text: &|v: i64| format!("v{v}"),
        }
    }

    #[test]
    fn integral_truncates_and_overflows() {
        let a = DurationMicrosecondArray::from(vec![Some(1_500_000), Some(-1_500_000), None]);
        let out =
            interval_to_integral(&a, MICROS_PER_SECOND, &DataType::Int32, false, names()).unwrap();
        let out = out.as_primitive::<Int32Type>();
        assert_eq!((out.value(0), out.value(1)), (1, -1));
        assert!(out.is_null(2));

        let big = DurationMicrosecondArray::from(vec![200_000_000]);
        let e = interval_to_integral(&big, MICROS_PER_SECOND, &DataType::Int8, false, names())
            .unwrap_err();
        assert!(e.to_string().contains("v200000000"), "{e}");
        let out =
            interval_to_integral(&big, MICROS_PER_SECOND, &DataType::Int8, true, names()).unwrap();
        assert!(out.is_null(0));
    }

    #[test]
    fn decimal_keeps_a_seconds_fraction_and_rounds_half_up() {
        let a = DurationMicrosecondArray::from(vec![1_500_000, -1_500_000]);
        let out = interval_to_decimal(&a, MICROS_PER_SECOND, &DataType::Decimal128(10, 0), false)
            .unwrap();
        let out = out.as_primitive::<Decimal128Type>();
        assert_eq!((out.value(0), out.value(1)), (2, -2));
        let out = interval_to_decimal(&a, MICROS_PER_SECOND, &DataType::Decimal128(10, 3), false)
            .unwrap();
        assert_eq!(out.as_primitive::<Decimal128Type>().value(0), 1_500);
        let out =
            interval_to_decimal(&a, MICROS_PER_SECOND, &DataType::Decimal64(10, 3), false).unwrap();
        assert_eq!(out.data_type(), &DataType::Decimal64(10, 3));
        assert_eq!(out.as_primitive::<Decimal64Type>().value(1), -1_500);
        let e = interval_to_decimal(&a, MICROS_PER_SECOND, &DataType::Decimal128(2, 2), false)
            .unwrap_err();
        let text = e.to_string();
        assert!(text.contains("NUMERIC_VALUE_OUT_OF_RANGE"), "{text}");
        assert!(text.contains("1.500000"), "{text}");
        assert_eq!(micros_text(-99_500_000), "-99.500000");
        let months = IntervalYearMonthArray::from(vec![10]);
        let e = interval_to_decimal(&months, 1, &DataType::Decimal128(1, 0), false).unwrap_err();
        assert!(e.to_string().contains(" 10 "), "{e}");
        let out = interval_to_decimal(&months, 1, &DataType::Decimal128(1, 0), true).unwrap();
        assert!(out.is_null(0));
    }

    #[test]
    fn numbers_become_intervals() {
        let n = Int64Array::from(vec![Some(5), None, Some(3_000_000_000)]);
        let e = number_to_year_month(&n, 1, false, names()).unwrap_err();
        assert!(e.to_string().contains("3000000000L"), "{e}");
        let out = number_to_year_month(&n, 1, true, names()).unwrap();
        assert_eq!(out.value(0), 5);
        assert!(out.is_null(1) && out.is_null(2));

        let d = Decimal128Array::from(vec![15, -25])
            .with_precision_and_scale(3, 1)
            .unwrap();
        let out = number_to_year_month(&d, 1, false, names()).unwrap();
        assert_eq!((out.value(0), out.value(1)), (2, -3));
        let out = number_to_day_time(&d, MICROS_PER_SECOND, false, names()).unwrap();
        assert_eq!(out.value(0), 1_500_000);
    }

    #[test]
    fn number_text_reads_as_sql() {
        let short = arrow_array::Decimal64Array::from(vec![15])
            .with_precision_and_scale(2, 1)
            .unwrap();
        let out = number_to_year_month(&short, 1, false, names()).unwrap();
        assert_eq!(out.value(0), 2);
        let d = Decimal128Array::from(vec![5])
            .with_precision_and_scale(3, 2)
            .unwrap();
        assert_eq!(number_text(-5, &d), "-0.05BD");
        assert_eq!(number_text(5, &arrow_array::Int8Array::from(vec![1])), "5Y");
        assert_eq!(
            number_text(5, &arrow_array::Int16Array::from(vec![1])),
            "5S"
        );
        assert_eq!(number_text(5, &arrow_array::Int32Array::from(vec![1])), "5");
    }
}
