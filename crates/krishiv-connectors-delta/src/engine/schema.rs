// SPDX-License-Identifier: Apache-2.0
//! Kernel schema types to Arrow types, and kernel scalars to Arrow arrays.
//!
//! Follows `delta_kernel` 0.30.1 `engine/arrow_conversion` (Apache-2.0), reduced to what the data bridge
//! needs: `timestamp` is microseconds UTC, `timestamp_ntz` microseconds without a zone, there is no
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
use delta_kernel::schema::{DataType, PrimitiveType, StructField, StructType};
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
                delta_kernel::schema::MetadataValue::String(s) => s.clone(),
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
                ArrowType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
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
