// SPDX-License-Identifier: Apache-2.0
//! Moving a value between decimal carriers (design P2-T160), and a calendar interval between its internal struct
//! and Arrow's `Interval(MonthDayNano)` at the session edges (ADR 0009 update 2026-11-06).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal32Type, Decimal64Type, Decimal128Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, GenericListArray, MapArray, OffsetSizeTrait, StructArray,
};
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

/// `a` as `to`, which differs at most in decimal carriers, nested ones included.
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
