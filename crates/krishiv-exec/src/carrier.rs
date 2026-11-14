// SPDX-License-Identifier: Apache-2.0
//! Moving a value between decimal carriers (design P2-T160), and a calendar interval between its internal struct
//! and Arrow's `Interval(MonthDayNano)` at the session edges (ADR 0009 update 2026-11-06).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::types::{Decimal32Type, Decimal64Type, Decimal128Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, GenericListArray, ListArray, MapArray, OffsetSizeTrait,
    PrimitiveArray, StructArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{ArrowError, DataType, FieldRef};
use krishiv_common::error::KrishivError;

use crate::error::internal;

fn arrow_error(e: &ArrowError) -> KrishivError {
    internal(e.to_string())
}

fn list<O: OffsetSizeTrait>(
    a: &GenericListArray<O>,
    item: &FieldRef,
) -> Result<GenericListArray<O>, KrishivError> {
    GenericListArray::try_new(
        Arc::clone(item),
        a.offsets().clone(),
        recarried_array(a.values(), item.data_type())?,
        a.nulls().cloned(),
    )
    .map_err(|e| arrow_error(&e))
}

fn structure(a: &StructArray, fields: &arrow_schema::Fields) -> Result<StructArray, KrishivError> {
    let columns = a
        .columns()
        .iter()
        .zip(fields)
        .map(|(c, f)| recarried_array(c, f.data_type()))
        .collect::<Result<Vec<_>, _>>()?;
    StructArray::try_new(fields.clone(), columns, a.nulls().cloned()).map_err(|e| arrow_error(&e))
}

/// A calendar interval whose microseconds overflow `i64` nanoseconds at the edge: Spark's `ArrowWriter` fails there
/// with `Math.multiplyExact`'s `ArithmeticException` ("long overflow"), which carries no error class. The nearest class
/// is `ARITHMETIC_OVERFLOW`, whose template always names `spark.sql.ansi.enabled`, although turning ANSI off does not
/// avoid this error (the edge conversion is exact in every mode).
fn calendar_overflow() -> KrishivError {
    krishiv_common::error::ArithmeticOverflow {
        message: "long overflow".to_owned(),
        alternative: String::new(),
        config: "\"spark.sql.ansi.enabled\"".to_owned(),
    }
    .into()
}

/// `a` on the decimal carrier `(p, s)` of `to`.
fn decimal(a: &ArrayRef, to: &DataType, p: u8, s: i8) -> Result<ArrayRef, KrishivError> {
    let unsupported = || internal(format!("no carrier change from {} to {to}", a.data_type()));
    let out: ArrayRef = match (a.data_type(), to) {
        (DataType::Decimal32(..), DataType::Decimal64(..)) => Arc::new(
            a.as_primitive::<Decimal32Type>()
                .unary::<_, Decimal64Type>(i64::from)
                .with_precision_and_scale(p, s)
                .map_err(|e| arrow_error(&e))?,
        ),
        (DataType::Decimal128(..), DataType::Decimal64(..)) => Arc::new(
            a.as_primitive::<Decimal128Type>()
                .try_unary::<_, Decimal64Type, _>(|v| {
                    i64::try_from(v).map_err(|e| ArrowError::CastError(e.to_string()))
                })
                .map_err(|e| arrow_error(&e))?
                .with_precision_and_scale(p, s)
                .map_err(|e| arrow_error(&e))?,
        ),
        (DataType::Decimal32(..), DataType::Decimal128(..)) => Arc::new(
            a.as_primitive::<Decimal32Type>()
                .unary::<_, Decimal128Type>(i128::from)
                .with_precision_and_scale(p, s)
                .map_err(|e| arrow_error(&e))?,
        ),
        (DataType::Decimal64(..), DataType::Decimal128(..)) => Arc::new(
            a.as_primitive::<Decimal64Type>()
                .unary::<_, Decimal128Type>(i128::from)
                .with_precision_and_scale(p, s)
                .map_err(|e| arrow_error(&e))?,
        ),
        _ => return Err(unsupported()),
    };
    Ok(out)
}

/// A vector carrier as the plain list every expression and every edge reads (design P3-T075 §5): the offsets are
/// `0, N, 2N, ..` and the values (and their nulls) are shared, not copied. A null row becomes an empty null slot, as a
/// list builder writes it, so the values of the non-null rows are gathered.
///
/// # Errors
/// `INTERNAL_ERROR` when `N` times the rows does not fit the `i32` offsets (it cannot under the length bound for a
/// batch of 32,767 rows or fewer; it is checked, not assumed).
pub fn fixed_to_list(l: &FixedSizeListArray, item: &FieldRef) -> Result<ListArray, KrishivError> {
    let n = usize::try_from(l.value_length())
        .map_err(|_| internal("a negative vector length".to_owned()))?;
    let rows = l.len();
    let overflow = || {
        internal(format!(
            "{rows} vectors of length {n} overflow list offsets"
        ))
    };
    let total = rows.checked_mul(n).ok_or_else(overflow)?;
    i32::try_from(total).map_err(|_| overflow())?;
    if l.null_count() == 0 {
        let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(n, rows));
        return GenericListArray::try_new(
            Arc::clone(item),
            offsets,
            recarried_array(l.values(), item.data_type())?,
            None,
        )
        .map_err(|e| arrow_error(&e));
    }
    // Valid rows only: each contributes its `n` values, a null row none.
    let mut take = Vec::with_capacity(total);
    let mut lengths = Vec::with_capacity(rows);
    for row in 0..rows {
        if l.is_valid(row) {
            let start = u32::try_from(row * n).map_err(|_| overflow())?;
            take.extend(start..start + u32::try_from(n).map_err(|_| overflow())?);
            lengths.push(n);
        } else {
            lengths.push(0);
        }
    }
    let values = arrow_select::take::take(
        l.values().as_ref(),
        &PrimitiveArray::<UInt32Type>::from(take),
        None,
    )
    .map_err(|e| arrow_error(&e))?;
    GenericListArray::try_new(
        Arc::clone(item),
        OffsetBuffer::from_lengths(lengths),
        recarried_array(&values, item.data_type())?,
        l.nulls().cloned(),
    )
    .map_err(|e| arrow_error(&e))
}

/// `a` as `to`, which differs at most in decimal carriers, nested ones included, and in a vector carrier
/// (`FixedSizeList`) read as a plain list.
///
/// # Errors
/// `INTERNAL_ERROR` for any other difference, or a value too large for the
/// narrower carrier (it is above its declared precision).
pub fn recarried_array(a: &ArrayRef, to: &DataType) -> Result<ArrayRef, KrishivError> {
    if a.data_type() == to {
        return Ok(Arc::clone(a));
    }
    // A calendar interval at the session edges (ADR 0009 update 2026-11-06): the internal struct of
    // microseconds becomes Arrow's `Interval(MonthDayNano)` and back.
    if let Some(c) = krishiv_types::calendar::Calendars::of(a.as_ref())
        && matches!(
            to,
            DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano)
        )
    {
        return krishiv_types::calendar::to_month_day_nano(&c)
            .map(|m| Arc::new(m) as ArrayRef)
            .map_err(|_| calendar_overflow());
    }
    if krishiv_types::arrow::is_calendar_struct(to)
        && let Some(m) = a.as_primitive_opt::<arrow_array::types::IntervalMonthDayNanoType>()
    {
        return Ok(krishiv_types::calendar::from_month_day_nano(m));
    }
    // A vector carrier read as a plain list (an expression input, an edge or a sink).
    if let (Some(l), DataType::List(item)) = (a.as_fixed_size_list_opt(), to) {
        return Ok(Arc::new(fixed_to_list(l, item)?));
    }
    Ok(match to {
        DataType::Decimal64(p, s) | DataType::Decimal128(p, s) => decimal(a, to, *p, *s)?,
        DataType::List(item) => Arc::new(list(a.as_list::<i32>(), item)?),
        DataType::LargeList(item) => Arc::new(list(a.as_list::<i64>(), item)?),
        DataType::FixedSizeList(item, n) => {
            let l = a.as_fixed_size_list();
            Arc::new(
                FixedSizeListArray::try_new(
                    Arc::clone(item),
                    *n,
                    recarried_array(l.values(), item.data_type())?,
                    l.nulls().cloned(),
                )
                .map_err(|e| arrow_error(&e))?,
            )
        }
        DataType::Map(entries, sorted) => {
            let DataType::Struct(fields) = entries.data_type() else {
                return Err(internal(format!("map entries of type {to}")));
            };
            let m = a.as_map();
            Arc::new(
                MapArray::try_new(
                    Arc::clone(entries),
                    m.offsets().clone(),
                    structure(m.entries(), fields)?,
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(|e| arrow_error(&e))?,
            )
        }
        DataType::Struct(fields) => Arc::new(structure(a.as_struct(), fields)?),
        other => {
            return Err(internal(format!(
                "no carrier change from {} to {other}",
                a.data_type()
            )));
        }
    })
}
