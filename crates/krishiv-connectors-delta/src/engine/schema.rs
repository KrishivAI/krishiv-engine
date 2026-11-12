// SPDX-License-Identifier: Apache-2.0
//! Kernel schema types to Arrow types, and kernel scalars to Arrow arrays.
//!
//! Follows `delta_kernel` 0.30.1 `engine/arrow_conversion` (Apache-2.0), reduced to what the data bridge
//! needs: `timestamp` is microseconds UTC (zone `+00:00`: arrow-cast and arrow-json parse only offset zones without `chrono-tz`, which ADR 0028 bans), `timestamp_ntz` microseconds without a zone, there is no
//! nanosecond type, and the unshredded Variant is a struct of `metadata` and `value`. Parquet field ids
//! (column mapping) arrive with the Parquet handler.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::builder::{
    ArrayBuilder, BinaryBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
    Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder, StringBuilder,
    TimestampMicrosecondBuilder, make_builder,
};
use arrow_schema::{DataType as ArrowType, Field, Schema, TimeUnit};
use delta_kernel::expressions::{ArrayData, Scalar};
use delta_kernel::schema::{
    ArrayType, DataType, MapType, MetadataValue, PrimitiveType, StructField, StructType,
};
use delta_kernel::{KernelError, KernelResult};

const LIST_ELEMENT: &str = "element";
const MAP_ENTRIES: &str = "key_value";

/// The Arrow schema of a kernel struct type.
pub fn arrow_schema(s: &StructType) -> KernelResult<Schema> {
    Ok(Schema::new(arrow_fields(s)?))
}

fn arrow_fields(s: &StructType) -> KernelResult<Vec<Field>> {
    s.fields().map(arrow_field).collect()
}

fn arrow_field(f: &StructField) -> KernelResult<Field> {
    let metadata = f
        .metadata()
        .iter()
        .map(|(k, v)| {
            let value = match v {
                MetadataValue::String(s) => s.clone(),
                other => serde_json::to_string(other)
                    .map_err(|e| KernelError::generic(format!("field metadata {k}: {e}")))?,
            };
            Ok((k.clone(), value))
        })
        .collect::<KernelResult<std::collections::HashMap<_, _>>>()?;
    Ok(Field::new(f.name(), arrow_type(f.data_type())?, f.is_nullable()).with_metadata(metadata))
}

/// The Arrow type of a kernel data type.
pub fn arrow_type(t: &DataType) -> KernelResult<ArrowType> {
    Ok(match t {
        DataType::Primitive(p) => match p {
            PrimitiveType::String => ArrowType::Utf8,
            PrimitiveType::Long | PrimitiveType::IntervalDayTime => ArrowType::Int64,
            PrimitiveType::Integer | PrimitiveType::IntervalYearMonth => ArrowType::Int32,
            PrimitiveType::Short => ArrowType::Int16,
            PrimitiveType::Byte => ArrowType::Int8,
            PrimitiveType::Float => ArrowType::Float32,
            PrimitiveType::Double => ArrowType::Float64,
            PrimitiveType::Boolean => ArrowType::Boolean,
            PrimitiveType::Binary => ArrowType::Binary,
            PrimitiveType::Decimal(d) => {
                let scale = i8::try_from(d.scale())
                    .map_err(|_| KernelError::schema(format!("decimal scale {}", d.scale())))?;
                ArrowType::Decimal128(d.precision(), scale)
            }
            PrimitiveType::Date => ArrowType::Date32,
            PrimitiveType::Timestamp => {
                ArrowType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into()))
            }
            PrimitiveType::TimestampNtz => ArrowType::Timestamp(TimeUnit::Microsecond, None),
            PrimitiveType::Void => ArrowType::Null,
        },
        DataType::Struct(s) => ArrowType::Struct(arrow_fields(s)?.into()),
        DataType::Array(a) => ArrowType::List(Arc::new(Field::new(
            LIST_ELEMENT,
            arrow_type(a.element_type())?,
            a.contains_null(),
        ))),
        DataType::Map(m) => {
            let entries = ArrowType::Struct(
                vec![
                    Field::new("key", arrow_type(m.key_type())?, false),
                    Field::new(
                        "value",
                        arrow_type(m.value_type())?,
                        m.value_contains_null(),
                    ),
                ]
                .into(),
            );
            ArrowType::Map(Arc::new(Field::new(MAP_ENTRIES, entries, false)), false)
        }
        DataType::Variant(s) => {
            if *t != DataType::unshredded_variant() {
                return Err(KernelError::schema(format!(
                    "only the unshredded variant schema is supported, got {t}"
                )));
            }
            ArrowType::Struct(arrow_fields(s)?.into())
        }
    })
}

/// The kernel struct type of an Arrow schema.
pub fn kernel_schema(s: &Schema) -> KernelResult<StructType> {
    StructType::try_new(
        s.fields()
            .iter()
            .map(|f| kernel_field(f))
            .collect::<KernelResult<Vec<_>>>()?,
    )
    .map_err(|e| KernelError::schema(e.to_string()))
}

fn kernel_field(f: &Field) -> KernelResult<StructField> {
    let metadata: std::collections::HashMap<String, MetadataValue> = f
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), MetadataValue::from(v)))
        .collect();
    Ok(StructField::new(
        f.name().clone(),
        kernel_type(f.data_type())?,
        f.is_nullable(),
    )
    .with_metadata(metadata))
}

/// The kernel data type of an Arrow type. Wider-than-Delta layouts (large, view, dictionary) fold onto
/// their logical type; millisecond and nanosecond timestamps map to the microsecond logical type.
pub fn kernel_type(t: &ArrowType) -> KernelResult<DataType> {
    let utc = |tz: &Option<Arc<str>>| {
        tz.as_deref().is_some_and(|z| {
            z.eq_ignore_ascii_case("utc") || matches!(z, "+00:00" | "-00:00" | "+0000" | "Z")
        })
    };
    Ok(match t {
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => DataType::STRING,
        ArrowType::Int64 | ArrowType::UInt64 => DataType::LONG,
        ArrowType::Int32 | ArrowType::UInt32 => DataType::INTEGER,
        ArrowType::Int16 | ArrowType::UInt16 => DataType::SHORT,
        ArrowType::Int8 | ArrowType::UInt8 => DataType::BYTE,
        ArrowType::Null => DataType::VOID,
        ArrowType::Float32 => DataType::FLOAT,
        ArrowType::Float64 => DataType::DOUBLE,
        ArrowType::Boolean => DataType::BOOLEAN,
        ArrowType::Binary
        | ArrowType::LargeBinary
        | ArrowType::BinaryView
        | ArrowType::FixedSizeBinary(_) => DataType::BINARY,
        ArrowType::Decimal128(p, sc) => {
            let scale = u8::try_from(*sc).map_err(|_| {
                KernelError::invalid_decimal("Negative scales are not supported in Delta")
            })?;
            DataType::decimal(*p, scale)?
        }
        ArrowType::Date32 | ArrowType::Date64 => DataType::DATE,
        ArrowType::Timestamp(_, tz) if utc(tz) => DataType::TIMESTAMP,
        ArrowType::Timestamp(_, None) => DataType::TIMESTAMP_NTZ,
        ArrowType::Struct(fields) => {
            DataType::try_struct_type_from_results(fields.iter().map(|f| kernel_field(f)))?
        }
        ArrowType::List(f)
        | ArrowType::LargeList(f)
        | ArrowType::ListView(f)
        | ArrowType::LargeListView(f)
        | ArrowType::FixedSizeList(f, _) => {
            ArrayType::new(kernel_type(f.data_type())?, f.is_nullable()).into()
        }
        ArrowType::Map(entries, _) => {
            let ArrowType::Struct(kv) = entries.data_type() else {
                return Err(KernelError::schema(
                    "a map type must have a struct entries field",
                ));
            };
            let [key, value] = kv.as_ref() else {
                return Err(KernelError::schema(
                    "a map entries struct must have a key and a value",
                ));
            };
            MapType::new(
                kernel_type(key.data_type())?,
                kernel_type(value.data_type())?,
                value.is_nullable(),
            )
            .into()
        }
        ArrowType::Dictionary(_, v) => kernel_type(v)?,
        other => {
            return Err(KernelError::schema(format!(
                "Invalid data type for Delta Lake: {other}"
            )));
        }
    })
}

/// The Arrow array holding the elements of one kernel [`ArrayData`]; only primitive elements.
pub fn array_data_to_arrow(a: &ArrayData) -> KernelResult<ArrayRef> {
    let ty = arrow_type(a.array_type().element_type())?;
    let elements = a.array_elements();
    let mut b = make_builder(&ty, elements.len());
    for e in elements {
        append_scalar(b.as_mut(), e)?;
    }
    Ok(b.finish())
}

macro_rules! put {
    ($b:expr, $B:ty, $v:expr, $s:expr) => {
        $b.as_any_mut()
            .downcast_mut::<$B>()
            .ok_or_else(|| mismatch($s))?
            .append_value($v)
    };
}

fn mismatch(s: &Scalar) -> KernelError {
    KernelError::generic(format!("scalar {s:?} does not match the builder type"))
}

fn append_null(b: &mut dyn ArrayBuilder, s: &Scalar, t: &DataType) -> KernelResult<()> {
    macro_rules! n {
        ($B:ty) => {
            b.as_any_mut()
                .downcast_mut::<$B>()
                .ok_or_else(|| mismatch(s))?
                .append_null()
        };
    }
    match t {
        DataType::Primitive(p) => match p {
            PrimitiveType::String => {
                n!(StringBuilder);
            }
            PrimitiveType::Long | PrimitiveType::IntervalDayTime => {
                n!(Int64Builder);
            }
            PrimitiveType::Integer | PrimitiveType::IntervalYearMonth => {
                n!(Int32Builder);
            }
            PrimitiveType::Short => {
                n!(Int16Builder);
            }
            PrimitiveType::Byte => {
                n!(Int8Builder);
            }
            PrimitiveType::Float => {
                n!(Float32Builder);
            }
            PrimitiveType::Double => {
                n!(Float64Builder);
            }
            PrimitiveType::Boolean => {
                n!(BooleanBuilder);
            }
            PrimitiveType::Binary => {
                n!(BinaryBuilder);
            }
            PrimitiveType::Decimal(_) => {
                n!(Decimal128Builder);
            }
            PrimitiveType::Date => {
                n!(Date32Builder);
            }
            PrimitiveType::Timestamp | PrimitiveType::TimestampNtz => {
                n!(TimestampMicrosecondBuilder);
            }
            PrimitiveType::Void => {
                return Err(KernelError::unsupported("a void scalar in an array column"));
            }
        },
        _ => {
            return Err(KernelError::unsupported(
                "a complex null scalar in an array column",
            ));
        }
    }
    Ok(())
}

fn append_scalar(b: &mut dyn ArrayBuilder, s: &Scalar) -> KernelResult<()> {
    match s {
        Scalar::Integer(v) | Scalar::IntervalYearMonth(v) => {
            put!(b, Int32Builder, *v, s);
        }
        Scalar::Long(v) | Scalar::IntervalDayTime(v) => {
            put!(b, Int64Builder, *v, s);
        }
        Scalar::Short(v) => {
            put!(b, Int16Builder, *v, s);
        }
        Scalar::Byte(v) => {
            put!(b, Int8Builder, *v, s);
        }
        Scalar::Float(v) => {
            put!(b, Float32Builder, *v, s);
        }
        Scalar::Double(v) => {
            put!(b, Float64Builder, *v, s);
        }
        Scalar::String(v) => {
            put!(b, StringBuilder, v, s);
        }
        Scalar::Boolean(v) => {
            put!(b, BooleanBuilder, *v, s);
        }
        Scalar::Timestamp(v) | Scalar::TimestampNtz(v) => {
            put!(b, TimestampMicrosecondBuilder, *v, s);
        }
        Scalar::Date(v) => {
            put!(b, Date32Builder, *v, s);
        }
        Scalar::Binary(v) => {
            put!(b, BinaryBuilder, v, s);
        }
        Scalar::Decimal(d) => {
            put!(b, Decimal128Builder, d.bits(), s);
        }
        Scalar::Null(t) => append_null(b, s, t)?,
        other => {
            return Err(KernelError::unsupported(format!(
                "a {other:?} scalar in an array column"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kernel_schema_round_trips_through_arrow() {
        let schema = StructType::new_unchecked([
            StructField::not_null("id", DataType::LONG),
            StructField::nullable("ts", DataType::TIMESTAMP),
            StructField::nullable("ntz", DataType::TIMESTAMP_NTZ),
            StructField::nullable("d", DataType::decimal(10, 2).unwrap()),
            StructField::nullable("tags", ArrayType::new(DataType::STRING, true)),
            StructField::nullable("m", MapType::new(DataType::STRING, DataType::INTEGER, true)),
        ]);
        let arrow = arrow_schema(&schema).unwrap();
        assert_eq!(
            arrow.field(1).data_type(),
            &ArrowType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into()))
        );
        assert_eq!(kernel_schema(&arrow).unwrap(), schema);
    }

    #[test]
    fn a_variant_reads_back_as_its_struct() {
        let arrow = ArrowType::Struct(
            arrow_fields(&StructType::new_unchecked([
                StructField::not_null("metadata", DataType::BINARY),
                StructField::not_null("value", DataType::BINARY),
            ]))
            .unwrap()
            .into(),
        );
        assert_eq!(arrow_type(&DataType::unshredded_variant()).unwrap(), arrow);
        assert!(matches!(kernel_type(&arrow).unwrap(), DataType::Struct(_)));
    }

    #[test]
    fn wide_arrow_layouts_fold_onto_the_logical_type() {
        assert_eq!(
            kernel_type(&ArrowType::LargeUtf8).unwrap(),
            DataType::STRING
        );
        assert_eq!(
            kernel_type(&ArrowType::Timestamp(TimeUnit::Millisecond, None)).unwrap(),
            DataType::TIMESTAMP_NTZ
        );
        assert_eq!(
            kernel_type(&ArrowType::Timestamp(
                TimeUnit::Nanosecond,
                Some("utc".into())
            ))
            .unwrap(),
            DataType::TIMESTAMP
        );
        assert!(kernel_type(&ArrowType::Decimal128(10, -1)).is_err());
        assert!(kernel_type(&ArrowType::Float16).is_err());
    }
}
