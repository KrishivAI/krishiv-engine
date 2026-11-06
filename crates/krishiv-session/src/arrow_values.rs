// SPDX-License-Identifier: Apache-2.0
//! Values read back from Arrow arrays as typed literals: a session variable's value, an
//! `EXECUTE IMMEDIATE` argument, a scalar subquery result (the single [`arrow_krishiv_type`] map).
//! An ANSI interval is held only where its fields are known (a tagged field or the analyzed
//! column's type): Arrow carries no fields, and Spark's defaults would be a silent divergence.

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_plan::Literal;
use krishiv_types::{DataType, DecimalType};

/// The value of `array` at `row` as a typed literal. The type comes from
/// the single [`arrow_krishiv_type`] map, so a NULL slot and a value never
/// disagree; anything the engine cannot hold in a variable fails there,
/// naming its Arrow type.
pub(crate) fn arrow_value(
    array: &arrow_array::ArrayRef,
    row: usize,
) -> Result<(DataType, Literal), KrishivError> {
    typed_value(array, row, arrow_krishiv_type(array.data_type())?)
}

/// [`arrow_value`] of column `column` of `batch`, whose field also says an interval's fields
/// (`INTERVAL '3' DAY` is a `Duration` tagged `interval day`).
pub(crate) fn arrow_field_value(
    batch: &RecordBatch,
    column: usize,
    row: usize,
) -> Result<(DataType, Literal), KrishivError> {
    let tagged = DataType::from_arrow_field(batch.schema().field(column)).ok();
    declared_value(batch.column(column), row, tagged.as_ref())
}

/// [`arrow_value`], also holding an ANSI interval whose fields `declared` (the analyzed column's
/// type, or the field's tag) gives: an interval of another kind, or of none, is refused as
/// [`arrow_value`] refuses it.
pub(crate) fn declared_value(
    array: &arrow_array::ArrayRef,
    row: usize,
    declared: Option<&DataType>,
) -> Result<(DataType, Literal), KrishivError> {
    let data_type = match (array.data_type(), declared) {
        (
            ArrowType::Interval(arrow_schema::IntervalUnit::YearMonth),
            Some(d @ DataType::YearMonthInterval { .. }),
        )
        | (
            ArrowType::Duration(arrow_schema::TimeUnit::Microsecond),
            Some(d @ DataType::DayTimeInterval { .. }),
        ) => d.clone(),
        (other, _) => arrow_krishiv_type(other)?,
    };
    typed_value(array, row, data_type)
}

/// The value of `array` at `row` as a literal of `data_type` (the array's Krishiv type).
fn typed_value(
    array: &arrow_array::ArrayRef,
    row: usize,
    data_type: DataType,
) -> Result<(DataType, Literal), KrishivError> {
    use arrow_array::types::{
        Date32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType, Int8Type, Int16Type,
        Int32Type, Int64Type, IntervalYearMonthType, TimestampMicrosecondType,
    };
    // A `NULL` array holds no validity bitmap (`is_null` is false), but
    // every slot is logically NULL.
    if matches!(array.data_type(), ArrowType::Null) || array.is_null(row) {
        let value = if data_type == DataType::Null {
            Literal::Null(None)
        } else {
            Literal::Null(Some(data_type.clone()))
        };
        return Ok((data_type, value));
    }
    let value = match array.data_type() {
        ArrowType::Boolean => Literal::Boolean(array.as_boolean().value(row)),
        ArrowType::Int8 => Literal::Byte(array.as_primitive::<Int8Type>().value(row)),
        ArrowType::Int16 => Literal::Short(array.as_primitive::<Int16Type>().value(row)),
        ArrowType::Int32 => Literal::Integer(array.as_primitive::<Int32Type>().value(row)),
        ArrowType::Int64 => Literal::Long(array.as_primitive::<Int64Type>().value(row)),
        ArrowType::Float32 => Literal::Float(
            array
                .as_primitive::<arrow_array::types::Float32Type>()
                .value(row),
        ),
        ArrowType::Float64 => Literal::Double(
            array
                .as_primitive::<arrow_array::types::Float64Type>()
                .value(row),
        ),
        ArrowType::Decimal64(..) | ArrowType::Decimal128(..) => {
            let DataType::Decimal(decimal) = &data_type else {
                return Err(crate::session::internal(
                    "a variable decimal changed type".to_owned(),
                ));
            };
            Literal::Decimal {
                unscaled: match array.as_primitive_opt::<Decimal64Type>() {
                    Some(a) => i128::from(a.value(row)),
                    None => array.as_primitive::<Decimal128Type>().value(row),
                },
                data_type: *decimal,
            }
        }
        ArrowType::Utf8 => Literal::String(array.as_string::<i32>().value(row).to_owned()),
        ArrowType::LargeUtf8 => Literal::String(array.as_string::<i64>().value(row).to_owned()),
        ArrowType::Utf8View => Literal::String(array.as_string_view().value(row).to_owned()),
        ArrowType::Binary => Literal::Binary(array.as_binary::<i32>().value(row).to_owned()),
        ArrowType::LargeBinary => Literal::Binary(array.as_binary::<i64>().value(row).to_owned()),
        ArrowType::BinaryView => Literal::Binary(array.as_binary_view().value(row).to_owned()),
        ArrowType::Date32 => Literal::Date(array.as_primitive::<Date32Type>().value(row)),
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            let micros = array.as_primitive::<TimestampMicrosecondType>().value(row);
            match &data_type {
                DataType::Timestamp => Literal::Timestamp(micros),
                DataType::TimestampNtz => Literal::TimestampNtz(micros),
                _ => {
                    return Err(crate::session::internal(
                        "a variable timestamp changed type".to_owned(),
                    ));
                }
            }
        }
        ArrowType::Interval(arrow_schema::IntervalUnit::YearMonth) => {
            let DataType::YearMonthInterval { start, end } = data_type else {
                return Err(crate::session::internal(
                    "a variable interval changed type".to_owned(),
                ));
            };
            Literal::YearMonthInterval {
                months: array.as_primitive::<IntervalYearMonthType>().value(row),
                start,
                end,
            }
        }
        ArrowType::Duration(arrow_schema::TimeUnit::Microsecond) => {
            let DataType::DayTimeInterval { start, end } = data_type else {
                return Err(crate::session::internal(
                    "a variable interval changed type".to_owned(),
                ));
            };
            Literal::DayTimeInterval {
                micros: array.as_primitive::<DurationMicrosecondType>().value(row),
                start,
                end,
            }
        }
        // `arrow_krishiv_type` above already failed on any other type.
        _ => {
            return Err(crate::session::internal(
                "a variable value changed type".to_owned(),
            ));
        }
    };
    Ok((data_type, value))
}

/// The Krishiv type of Arrow `data_type`: the single Arrow-to-Krishiv
/// type map [`arrow_value`] reads, so NULL slots and values agree.
pub(crate) fn arrow_krishiv_type(data_type: &ArrowType) -> Result<DataType, KrishivError> {
    match data_type {
        ArrowType::Null => Ok(DataType::Null),
        ArrowType::Boolean => Ok(DataType::Boolean),
        ArrowType::Int8 => Ok(DataType::Byte),
        ArrowType::Int16 => Ok(DataType::Short),
        ArrowType::Int32 => Ok(DataType::Integer),
        ArrowType::Int64 => Ok(DataType::Long),
        ArrowType::Float32 => Ok(DataType::Float),
        ArrowType::Float64 => Ok(DataType::Double),
        ArrowType::Decimal64(precision, scale) | ArrowType::Decimal128(precision, scale) => {
            decimal_type(*precision, *scale).map(DataType::Decimal)
        }
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => Ok(DataType::String),
        ArrowType::Binary | ArrowType::LargeBinary | ArrowType::BinaryView => Ok(DataType::Binary),
        ArrowType::Date32 => Ok(DataType::Date),
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some(_)) => {
            Ok(DataType::Timestamp)
        }
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, None) => {
            Ok(DataType::TimestampNtz)
        }
        // An ANSI interval's fields are not in its Arrow type: only [`declared_value`] holds one.
        other => Err(unsupported(other)),
    }
}

fn unsupported(data_type: &ArrowType) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("session variable values of type {data_type}"),
    }
    .into()
}

/// A decimal type from Arrow's precision and scale, for variable values.
fn decimal_type(precision: u8, scale: i8) -> Result<DecimalType, KrishivError> {
    DecimalType::try_new(
        u32::from(precision),
        u32::try_from(scale).map_err(|_| {
            crate::session::internal("an invalid decimal type in a variable".to_owned())
        })?,
    )
    .map_err(|e| crate::session::internal(format!("an invalid decimal type in a variable: {e}")))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_common::error::ErrorClass;

    use super::*;

    /// Every Arrow type the engine produces extracts to its literal.
    #[test]
    fn arrow_values_extract_every_type() {
        use arrow_array::{
            BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
            Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
            LargeStringArray, NullArray, StringArray, StringViewArray, TimestampMicrosecondArray,
        };
        let value = |array: arrow_array::ArrayRef| arrow_value(&array, 0).unwrap();
        assert_eq!(
            value(Arc::new(BooleanArray::from(vec![true]))),
            (DataType::Boolean, Literal::Boolean(true))
        );
        assert_eq!(
            value(Arc::new(Int8Array::from(vec![7]))),
            (DataType::Byte, Literal::Byte(7))
        );
        assert_eq!(
            value(Arc::new(Int16Array::from(vec![7]))),
            (DataType::Short, Literal::Short(7))
        );
        assert_eq!(
            value(Arc::new(Int32Array::from(vec![7]))),
            (DataType::Integer, Literal::Integer(7))
        );
        assert_eq!(
            value(Arc::new(Int64Array::from(vec![7]))),
            (DataType::Long, Literal::Long(7))
        );
        assert_eq!(
            value(Arc::new(Float32Array::from(vec![1.5]))),
            (DataType::Float, Literal::Float(1.5))
        );
        assert_eq!(
            value(Arc::new(Float64Array::from(vec![1.5]))),
            (DataType::Double, Literal::Double(1.5))
        );
        let decimal = Decimal128Array::from(vec![15])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert_eq!(
            value(Arc::new(decimal)),
            (
                DataType::Decimal(DecimalType::try_new(10, 2).unwrap()),
                Literal::Decimal {
                    unscaled: 15,
                    data_type: DecimalType::try_new(10, 2).unwrap(),
                }
            )
        );
        assert_eq!(
            value(Arc::new(StringArray::from(vec!["a"]))),
            (DataType::String, Literal::String("a".to_owned()))
        );
        assert_eq!(
            value(Arc::new(LargeStringArray::from(vec!["a"]))),
            (DataType::String, Literal::String("a".to_owned()))
        );
        assert_eq!(
            value(Arc::new(StringViewArray::from(vec!["a"]))),
            (DataType::String, Literal::String("a".to_owned()))
        );
        assert_eq!(
            value(Arc::new(BinaryArray::from(vec![b"a".as_slice()]))),
            (DataType::Binary, Literal::Binary(vec![b'a']))
        );
        assert_eq!(
            value(Arc::new(LargeBinaryArray::from(vec![b"a".as_slice()]))),
            (DataType::Binary, Literal::Binary(vec![b'a']))
        );
        assert_eq!(
            value(Arc::new(BinaryViewArray::from(vec![b"a".as_slice()]))),
            (DataType::Binary, Literal::Binary(vec![b'a']))
        );
        assert_eq!(
            value(Arc::new(Date32Array::from(vec![0]))),
            (DataType::Date, Literal::Date(0))
        );
        assert_eq!(
            value(Arc::new(
                TimestampMicrosecondArray::from(vec![1_000_000]).with_timezone("UTC")
            )),
            (DataType::Timestamp, Literal::Timestamp(1_000_000))
        );
        assert_eq!(
            value(Arc::new(TimestampMicrosecondArray::from(vec![1]))),
            (DataType::TimestampNtz, Literal::TimestampNtz(1))
        );
        // Untyped and typed nulls.
        assert_eq!(
            value(Arc::new(NullArray::new(1))),
            (DataType::Null, Literal::Null(None))
        );
        assert_eq!(
            value(Arc::new(Int32Array::from(vec![None]))),
            (DataType::Integer, Literal::Null(Some(DataType::Integer)))
        );
        assert_eq!(
            value(Arc::new(StringViewArray::from(vec![None as Option<&str>]))),
            (DataType::String, Literal::Null(Some(DataType::String)))
        );
        // Anything else names its Arrow type.
        let other: arrow_array::ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
        let _ = arrow_krishiv_type(other.data_type()).unwrap();
        let e = arrow_value(
            &{
                let array: arrow_array::ArrayRef =
                    Arc::new(arrow_array::DurationSecondArray::from(vec![1]));
                array
            },
            0,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        let e = decimal_type(99, 99).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }

    /// An ANSI interval keeps the fields it was declared with; without
    /// them it is refused, never given Spark's default fields.
    #[test]
    fn intervals_need_their_declared_fields() {
        use arrow_array::{DurationMicrosecondArray, IntervalYearMonthArray};
        use krishiv_types::DayTimeField;

        let day: arrow_array::ArrayRef =
            Arc::new(DurationMicrosecondArray::from(vec![86_400_000_000]));
        let declared = DataType::DayTimeInterval {
            start: DayTimeField::Day,
            end: DayTimeField::Day,
        };
        let (ty, value) = declared_value(&day, 0, Some(&declared)).unwrap();
        assert_eq!(ty, declared);
        assert!(matches!(
            value,
            Literal::DayTimeInterval {
                micros: 86_400_000_000,
                ..
            }
        ));
        let months: arrow_array::ArrayRef = Arc::new(IntervalYearMonthArray::from(vec![14]));
        // The wrong kind of interval, or none: refused.
        for e in [
            declared_value(&months, 0, Some(&declared)).unwrap_err(),
            declared_value(&day, 0, None).unwrap_err(),
            arrow_value(&day, 0).unwrap_err(),
            arrow_value(&months, 0).unwrap_err(),
        ] {
            assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        }
    }
}
