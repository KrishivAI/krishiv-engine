// SPDX-License-Identifier: Apache-2.0
//! Kernel schemas and values as Arrow 60 and back (design note P3-T021 section 3.1).
//!
//! The shape follows the conversions of the kernel's default engine (`delta_kernel` 0.30.1,
//! `src/engine/arrow_conversion/`, Apache-2.0), which are compiled only with the kernel's Arrow
//! features; Krishiv builds the kernel without them, so the conversions live here. Rules:
//! - `timestamp` is `Timestamp(Microsecond, "UTC")` and `timestamp_ntz` is
//!   `Timestamp(Microsecond, None)`. There is no nanosecond or millisecond type: such an Arrow
//!   type is refused (Krishiv's readers already widen millisecond timestamps to microseconds).
//!   Any non-empty zone label reads as `timestamp` (the values are UTC instants).
//! - A Variant is the struct of `metadata` and `value` (both non-null binary); only the
//!   unshredded Variant converts.
//! - A list's element field is `element`, a map's entries are `key_value` with `key` and
//!   `value`; map keys are never null.
//! - Field ids: the kernel's `parquet.field.id` is Arrow's `PARQUET:field_id`, and the ids of a
//!   list's or map's synthesized fields (`delta.columnMapping.nested.ids` on the nearest named
//!   field) go onto those Arrow fields, both ways.
//! - Unsigned integers, `Date64` and other types Delta has no column type for are refused.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    ArrayBuilder, BinaryBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
    Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder, ListBuilder, MapBuilder,
    NullBuilder, StringBuilder, StructBuilder, TimestampMicrosecondBuilder, make_builder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType as ArrowType, Field as ArrowField, Schema as ArrowSchema, TimeUnit};
use delta_kernel::expressions::{ArrayData, Scalar};
use delta_kernel::schema::{
    ArrayType, ColumnMetadataKey, DataType, MapType, MetadataValue, PrimitiveType, StructField,
    StructType,
};
use delta_kernel::{KernelError, KernelResult};

/// Arrow's field-id metadata key (as `krishiv_io::FIELD_ID_KEY`).
pub const ARROW_FIELD_ID: &str = "PARQUET:field_id";

/// The zone label of a `timestamp` column (as Krishiv's type mapping).
pub const UTC: &str = "UTC";

const LIST_ELEMENT: &str = "element";
const MAP_ENTRIES: &str = "key_value";
const MAP_KEY: &str = "key";
const MAP_VALUE: &str = "value";

fn schema_error(message: impl ToString) -> KernelError {
    KernelError::schema(message)
}

// ---------------------------------------------------------------------------------------------
// Kernel to Arrow.

/// The Arrow schema of the kernel schema `schema`.
///
/// # Errors
/// A type with no Arrow form here (a shredded Variant), or malformed field-id metadata.
pub fn to_arrow_schema(schema: &StructType) -> KernelResult<ArrowSchema> {
    Ok(ArrowSchema::new(
        schema
            .fields()
            .map(to_arrow_field)
            .collect::<KernelResult<Vec<_>>>()?,
    ))
}

/// The Arrow field of the kernel field `field`, with its metadata and field ids.
///
/// # Errors
/// As [`to_arrow_schema`].
pub fn to_arrow_field(field: &StructField) -> KernelResult<ArrowField> {
    let mut metadata = HashMap::new();
    for (key, value) in field.metadata() {
        if key == ColumnMetadataKey::ColumnMappingNestedIds.as_ref() {
            // Consumed below, onto the synthesized list and map fields.
            continue;
        }
        let key = if key == ColumnMetadataKey::ParquetFieldId.as_ref() {
            ARROW_FIELD_ID.to_owned()
        } else {
            key.clone()
        };
        let value = match value {
            MetadataValue::String(s) => s.clone(),
            other => serde_json::to_string(other).map_err(|e| schema_error(e.to_string()))?,
        };
        metadata.insert(key, value);
    }
    let data_type = with_nested_ids(field, field.name(), field.data_type())?;
    Ok(ArrowField::new(field.name(), data_type, field.is_nullable()).with_metadata(metadata))
}

/// The nested field id stored on `field` for the dotted `path`, if any.
fn nested_id(field: &StructField, path: &str) -> KernelResult<Option<i64>> {
    let key = ColumnMetadataKey::ColumnMappingNestedIds.as_ref();
    let Some(value) = field.metadata().get(key) else {
        return Ok(None);
    };
    let MetadataValue::Other(serde_json::Value::Object(ids)) = value else {
        return Err(schema_error(format!(
            "'{key}' on field '{}' must be a JSON object",
            field.name()
        )));
    };
    ids.get(path).map_or(Ok(None), |id| {
        id.as_i64().map(Some).ok_or_else(|| {
            schema_error(format!(
                "'{key}['{path}']' on field '{}' must be an integer, got {id}",
                field.name()
            ))
        })
    })
}

fn id_metadata(id: Option<i64>) -> HashMap<String, String> {
    id.map(|id| HashMap::from([(ARROW_FIELD_ID.to_owned(), id.to_string())]))
        .unwrap_or_default()
}

/// `data_type` as Arrow, with the nested ids of `ancestor` on list and map fields.
fn with_nested_ids(
    ancestor: &StructField,
    path: &str,
    data_type: &DataType,
) -> KernelResult<ArrowType> {
    match data_type {
        DataType::Array(a) => {
            let element_path = format!("{path}.{LIST_ELEMENT}");
            let element = with_nested_ids(ancestor, &element_path, a.element_type())?;
            let field = ArrowField::new(LIST_ELEMENT, element, a.contains_null())
                .with_metadata(id_metadata(nested_id(ancestor, &element_path)?));
            Ok(ArrowType::List(Arc::new(field)))
        }
        DataType::Map(m) => {
            let key_path = format!("{path}.{MAP_KEY}");
            let value_path = format!("{path}.{MAP_VALUE}");
            let key = ArrowField::new(
                MAP_KEY,
                with_nested_ids(ancestor, &key_path, m.key_type())?,
                false,
            )
            .with_metadata(id_metadata(nested_id(ancestor, &key_path)?));
            let value = ArrowField::new(
                MAP_VALUE,
                with_nested_ids(ancestor, &value_path, m.value_type())?,
                m.value_contains_null(),
            )
            .with_metadata(id_metadata(nested_id(ancestor, &value_path)?));
            Ok(map_type(key, value))
        }
        DataType::Struct(_) | DataType::Primitive(_) | DataType::Variant(_) => {
            to_arrow_type(data_type)
        }
    }
}

fn map_type(key: ArrowField, value: ArrowField) -> ArrowType {
    // `MapArray::try_new` refuses nullable entries.
    let entries = ArrowField::new(
        MAP_ENTRIES,
        ArrowType::Struct(vec![key, value].into()),
        false,
    );
    ArrowType::Map(Arc::new(entries), false)
}

/// The Arrow type of the kernel type `data_type` (without nested field ids).
///
/// # Errors
/// A shredded Variant, or a geo type.
pub fn to_arrow_type(data_type: &DataType) -> KernelResult<ArrowType> {
    Ok(match data_type {
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
            PrimitiveType::Date => ArrowType::Date32,
            PrimitiveType::Timestamp => {
                ArrowType::Timestamp(TimeUnit::Microsecond, Some(UTC.into()))
            }
            PrimitiveType::TimestampNtz => ArrowType::Timestamp(TimeUnit::Microsecond, None),
            PrimitiveType::Void => ArrowType::Null,
            PrimitiveType::Decimal(d) => ArrowType::Decimal128(
                d.precision(),
                i8::try_from(d.scale())
                    .map_err(|_| schema_error(format!("decimal scale {}", d.scale())))?,
            ),
        },
        DataType::Struct(s) => ArrowType::Struct(to_arrow_schema(s)?.fields().clone()),
        DataType::Array(a) => ArrowType::List(Arc::new(ArrowField::new(
            LIST_ELEMENT,
            to_arrow_type(a.element_type())?,
            a.contains_null(),
        ))),
        DataType::Map(m) => map_type(
            ArrowField::new(MAP_KEY, to_arrow_type(m.key_type())?, false),
            ArrowField::new(
                MAP_VALUE,
                to_arrow_type(m.value_type())?,
                m.value_contains_null(),
            ),
        ),
        DataType::Variant(s) => {
            if *data_type != DataType::unshredded_variant() {
                return Err(schema_error(format!(
                    "a shredded Variant ({data_type}) has no Arrow form here; only the unshredded \
                     struct of metadata and value"
                )));
            }
            ArrowType::Struct(to_arrow_schema(s)?.fields().clone())
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Arrow to kernel.

/// The kernel schema of the Arrow schema `schema`.
///
/// # Errors
/// A type Delta has no column type for (see the module docs), or a malformed field id.
pub fn to_kernel_schema(schema: &ArrowSchema) -> KernelResult<StructType> {
    StructType::try_new(
        schema
            .fields()
            .iter()
            .map(|f| to_kernel_field(f))
            .collect::<KernelResult<Vec<_>>>()?,
    )
}

/// The kernel field of the Arrow field `field`, with its metadata and field ids.
///
/// # Errors
/// As [`to_kernel_schema`].
pub fn to_kernel_field(field: &ArrowField) -> KernelResult<StructField> {
    let kernel_id = ColumnMetadataKey::ParquetFieldId.as_ref();
    let meta = field.metadata();
    if let (Some(a), Some(k)) = (meta.get(ARROW_FIELD_ID), meta.get(kernel_id))
        && a != k
    {
        return Err(schema_error(format!(
            "field '{}': conflicting field ids {a} ({ARROW_FIELD_ID}) and {k} ({kernel_id})",
            field.name()
        )));
    }
    let mut metadata: HashMap<String, MetadataValue> = HashMap::with_capacity(meta.len());
    for (key, value) in meta {
        if key == ARROW_FIELD_ID || key == kernel_id {
            metadata.insert(
                kernel_id.to_owned(),
                MetadataValue::Number(parse_id(field.name(), key, value)?),
            );
        } else {
            metadata.insert(key.clone(), MetadataValue::from(value));
        }
    }
    let mut nested = serde_json::Map::new();
    collect_nested_ids(field.data_type(), field.name(), &mut nested)?;
    if !nested.is_empty() {
        metadata.insert(
            ColumnMetadataKey::ColumnMappingNestedIds
                .as_ref()
                .to_owned(),
            MetadataValue::Other(serde_json::Value::Object(nested)),
        );
    }
    Ok(StructField::new(
        field.name().clone(),
        to_kernel_type(field.data_type())?,
        field.is_nullable(),
    )
    .with_metadata(metadata))
}

fn parse_id(name: &str, key: &str, value: &str) -> KernelResult<i64> {
    value.parse().map_err(|_| {
        schema_error(format!(
            "'{key}' on field '{name}' must be an integer, got '{value}'"
        ))
    })
}

/// The field ids of the synthesized list and map fields under `data_type`, by dotted path.
fn collect_nested_ids(
    data_type: &ArrowType,
    path: &str,
    out: &mut serde_json::Map<String, serde_json::Value>,
) -> KernelResult<()> {
    let inner: Vec<(&ArrowField, &str)> = match data_type {
        ArrowType::List(f)
        | ArrowType::LargeList(f)
        | ArrowType::ListView(f)
        | ArrowType::LargeListView(f) => {
            vec![(f.as_ref(), LIST_ELEMENT)]
        }
        ArrowType::Map(entries, _) => match entries.data_type() {
            ArrowType::Struct(kv) if kv.len() == 2 => {
                vec![(kv[0].as_ref(), MAP_KEY), (kv[1].as_ref(), MAP_VALUE)]
            }
            other => {
                return Err(schema_error(format!(
                    "map entries must be a struct of key and value, got {other}"
                )));
            }
        },
        _ => return Ok(()),
    };
    for (field, segment) in inner {
        let child = format!("{path}.{segment}");
        if let Some(id) = field.metadata().get(ARROW_FIELD_ID) {
            out.insert(
                child.clone(),
                parse_id(field.name(), ARROW_FIELD_ID, id)?.into(),
            );
        }
        collect_nested_ids(field.data_type(), &child, out)?;
    }
    Ok(())
}

/// The kernel type of the Arrow type `data_type`.
///
/// # Errors
/// A type Delta has no column type for (see the module docs).
pub fn to_kernel_type(data_type: &ArrowType) -> KernelResult<DataType> {
    Ok(match data_type {
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => DataType::STRING,
        ArrowType::Int64 => DataType::LONG,
        ArrowType::Int32 => DataType::INTEGER,
        ArrowType::Int16 => DataType::SHORT,
        ArrowType::Int8 => DataType::BYTE,
        ArrowType::Null => DataType::VOID,
        ArrowType::Float32 => DataType::FLOAT,
        ArrowType::Float64 => DataType::DOUBLE,
        ArrowType::Boolean => DataType::BOOLEAN,
        ArrowType::Binary
        | ArrowType::LargeBinary
        | ArrowType::BinaryView
        | ArrowType::FixedSizeBinary(_) => DataType::BINARY,
        ArrowType::Decimal128(p, s) => {
            let scale = u8::try_from(*s).map_err(|_| {
                KernelError::invalid_decimal("negative scales are not supported in Delta")
            })?;
            DataType::decimal(*p, scale)?
        }
        ArrowType::Date32 => DataType::DATE,
        ArrowType::Timestamp(TimeUnit::Microsecond, Some(zone)) if !zone.is_empty() => {
            DataType::TIMESTAMP
        }
        ArrowType::Timestamp(TimeUnit::Microsecond, _) => DataType::TIMESTAMP_NTZ,
        ArrowType::Struct(fields) => {
            let fields = fields
                .iter()
                .map(|f| to_kernel_field(f))
                .collect::<KernelResult<Vec<_>>>()?;
            // A Variant's Arrow layout is a plain struct of `metadata` and `value`, so it reads
            // back as a struct; the table schema, not the Arrow layout, says Variant.
            DataType::from(StructType::try_new(fields)?)
        }
        ArrowType::List(f)
        | ArrowType::LargeList(f)
        | ArrowType::ListView(f)
        | ArrowType::LargeListView(f) => {
            ArrayType::new(to_kernel_type(f.data_type())?, f.is_nullable()).into()
        }
        ArrowType::Map(entries, _) => match entries.data_type() {
            ArrowType::Struct(kv) if kv.len() == 2 => MapType::new(
                to_kernel_type(kv[0].data_type())?,
                to_kernel_type(kv[1].data_type())?,
                kv[1].is_nullable(),
            )
            .into(),
            other => {
                return Err(schema_error(format!(
                    "map entries must be a struct of key and value, got {other}"
                )));
            }
        },
        ArrowType::Dictionary(_, value) => to_kernel_type(value)?,
        other => {
            return Err(schema_error(format!(
                "the Arrow type {other} has no Delta column type"
            )));
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Values.

/// `scalar` repeated `rows` times, as an Arrow array of [`to_arrow_type`] of its type.
///
/// # Errors
/// A type with no Arrow form here (see [`to_arrow_type`]).
pub fn scalar_to_array(scalar: &Scalar, rows: usize) -> KernelResult<ArrayRef> {
    let mut builder = make_builder(&to_arrow_type(&scalar.data_type())?, rows);
    append_scalar(scalar, builder.as_mut(), rows)?;
    Ok(builder.finish())
}

/// The elements of `data` as an Arrow array (one row per element).
///
/// # Errors
/// As [`scalar_to_array`].
pub fn array_data_to_arrow(data: &ArrayData) -> KernelResult<ArrayRef> {
    let elements = data.array_elements();
    let mut builder = make_builder(
        &to_arrow_type(data.array_type().element_type())?,
        elements.len(),
    );
    for element in elements {
        append_scalar(element, builder.as_mut(), 1)?;
    }
    Ok(builder.finish())
}

fn wrong_builder(what: &dyn std::fmt::Display) -> KernelError {
    KernelError::invalid_expression(format!("no Arrow builder for {what}"))
}

macro_rules! builder_as {
    ($builder:expr, $t:ty, $what:expr) => {
        $builder
            .as_any_mut()
            .downcast_mut::<$t>()
            .ok_or_else(|| wrong_builder($what))?
    };
}

/// Appends `scalar` `rows` times. Struct builders need an entry per child for every row, null
/// rows included; list and map builders do not.
fn append_scalar(scalar: &Scalar, builder: &mut dyn ArrayBuilder, rows: usize) -> KernelResult<()> {
    let what = &scalar.data_type();
    match scalar {
        Scalar::Integer(v) | Scalar::IntervalYearMonth(v) => {
            builder_as!(builder, Int32Builder, what).append_value_n(*v, rows);
        }
        Scalar::Long(v) | Scalar::IntervalDayTime(v) => {
            builder_as!(builder, Int64Builder, what).append_value_n(*v, rows);
        }
        Scalar::Short(v) => builder_as!(builder, Int16Builder, what).append_value_n(*v, rows),
        Scalar::Byte(v) => builder_as!(builder, Int8Builder, what).append_value_n(*v, rows),
        Scalar::Float(v) => builder_as!(builder, Float32Builder, what).append_value_n(*v, rows),
        Scalar::Double(v) => builder_as!(builder, Float64Builder, what).append_value_n(*v, rows),
        Scalar::Boolean(v) => builder_as!(builder, BooleanBuilder, what).append_n(rows, *v),
        Scalar::Timestamp(v) | Scalar::TimestampNtz(v) => {
            builder_as!(builder, TimestampMicrosecondBuilder, what).append_value_n(*v, rows);
        }
        Scalar::Date(v) => builder_as!(builder, Date32Builder, what).append_value_n(*v, rows),
        Scalar::Decimal(v) => {
            builder_as!(builder, Decimal128Builder, what).append_value_n(v.bits(), rows);
        }
        Scalar::String(v) => {
            let b = builder_as!(builder, StringBuilder, what);
            for _ in 0..rows {
                b.append_value(v);
            }
        }
        Scalar::Binary(v) => {
            let b = builder_as!(builder, BinaryBuilder, what);
            for _ in 0..rows {
                b.append_value(v);
            }
        }
        Scalar::Struct(data) => {
            let b = builder_as!(builder, StructBuilder, what);
            if b.num_fields() != data.fields().len() {
                return Err(wrong_builder(what));
            }
            for (child, value) in b.field_builders_mut().iter_mut().zip(data.values()) {
                append_scalar(value, child.as_mut(), rows)?;
            }
            for _ in 0..rows {
                b.append(true);
            }
        }
        Scalar::Array(data) => {
            let b = builder_as!(builder, ListBuilder<Box<dyn ArrayBuilder>>, what);
            for _ in 0..rows {
                for value in data.array_elements() {
                    append_scalar(value, b.values().as_mut(), 1)?;
                }
                b.append(true);
            }
        }
        Scalar::Map(data) => {
            let b = builder_as!(
                builder,
                MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>,
                what
            );
            for _ in 0..rows {
                for (key, value) in data.pairs() {
                    append_scalar(key, b.keys().as_mut(), 1)?;
                    append_scalar(value, b.values().as_mut(), 1)?;
                }
                b.append(true)
                    .map_err(|e| KernelError::generic(e.to_string()))?;
            }
        }
        Scalar::Null(data_type) => append_null(data_type, builder, rows)?,
    }
    Ok(())
}

fn append_null(
    data_type: &DataType,
    builder: &mut dyn ArrayBuilder,
    rows: usize,
) -> KernelResult<()> {
    macro_rules! nulls {
        ($t:ty) => {
            builder_as!(builder, $t, data_type).append_nulls(rows)
        };
    }
    match data_type {
        DataType::Primitive(p) => match p {
            PrimitiveType::Integer | PrimitiveType::IntervalYearMonth => nulls!(Int32Builder),
            PrimitiveType::Long | PrimitiveType::IntervalDayTime => nulls!(Int64Builder),
            PrimitiveType::Short => nulls!(Int16Builder),
            PrimitiveType::Byte => nulls!(Int8Builder),
            PrimitiveType::Float => nulls!(Float32Builder),
            PrimitiveType::Double => nulls!(Float64Builder),
            PrimitiveType::String => nulls!(StringBuilder),
            PrimitiveType::Boolean => nulls!(BooleanBuilder),
            PrimitiveType::Timestamp | PrimitiveType::TimestampNtz => {
                nulls!(TimestampMicrosecondBuilder);
            }
            PrimitiveType::Date => nulls!(Date32Builder),
            PrimitiveType::Binary => nulls!(BinaryBuilder),
            PrimitiveType::Decimal(_) => nulls!(Decimal128Builder),
            PrimitiveType::Void => nulls!(NullBuilder),
        },
        DataType::Struct(st) | DataType::Variant(st) => {
            let b = builder_as!(builder, StructBuilder, data_type);
            if b.num_fields() != st.num_fields() {
                return Err(wrong_builder(data_type));
            }
            for (child, field) in b.field_builders_mut().iter_mut().zip(st.fields()) {
                append_null(field.data_type(), child.as_mut(), rows)?;
            }
            b.append_nulls(rows);
        }
        DataType::Array(_) => nulls!(ListBuilder<Box<dyn ArrayBuilder>>),
        DataType::Map(_) => {
            let b = builder_as!(
                builder,
                MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>,
                data_type
            );
            for _ in 0..rows {
                b.append(false)
                    .map_err(|e| KernelError::generic(e.to_string()))?;
            }
        }
    }
    Ok(())
}

/// The value at `row` of the primitive Arrow array `array` as a kernel [`Scalar`]
/// (`Scalar::Null` of the column's type when null).
///
/// # Errors
/// `row` out of bounds, or a non-primitive or non-Delta Arrow type.
pub fn scalar_at(array: &dyn Array, row: usize) -> KernelResult<Scalar> {
    if row >= array.len() {
        return Err(KernelError::generic(format!(
            "row {row} out of bounds for an array of {} rows",
            array.len()
        )));
    }
    let data_type = to_kernel_type(array.data_type())?;
    if data_type.as_primitive_opt().is_none() {
        return Err(KernelError::generic(format!(
            "a {data_type} value is not a primitive scalar"
        )));
    }
    if array.is_null(row) {
        return Ok(Scalar::Null(data_type));
    }
    let missing = || KernelError::generic(format!("unexpected Arrow layout for {data_type}"));
    Ok(match array.data_type() {
        ArrowType::Int8 => Scalar::Byte(array.as_primitive::<Int8Type>().value(row)),
        ArrowType::Int16 => Scalar::Short(array.as_primitive::<Int16Type>().value(row)),
        ArrowType::Int32 => Scalar::Integer(array.as_primitive::<Int32Type>().value(row)),
        ArrowType::Int64 => Scalar::Long(array.as_primitive::<Int64Type>().value(row)),
        ArrowType::Float32 => Scalar::Float(array.as_primitive::<Float32Type>().value(row)),
        ArrowType::Float64 => Scalar::Double(array.as_primitive::<Float64Type>().value(row)),
        ArrowType::Boolean => Scalar::Boolean(array.as_boolean().value(row)),
        ArrowType::Date32 => Scalar::Date(array.as_primitive::<Date32Type>().value(row)),
        ArrowType::Timestamp(..) => {
            let v = array.as_primitive::<TimestampMicrosecondType>().value(row);
            if data_type == DataType::TIMESTAMP {
                Scalar::Timestamp(v)
            } else {
                Scalar::TimestampNtz(v)
            }
        }
        ArrowType::Decimal128(p, s) => Scalar::decimal(
            array.as_primitive::<Decimal128Type>().value(row),
            *p,
            u8::try_from(*s).map_err(|_| missing())?,
        )?,
        ArrowType::Utf8 => Scalar::String(array.as_string::<i32>().value(row).to_owned()),
        ArrowType::LargeUtf8 => Scalar::String(array.as_string::<i64>().value(row).to_owned()),
        ArrowType::Utf8View => Scalar::String(array.as_string_view().value(row).to_owned()),
        ArrowType::Binary => Scalar::Binary(array.as_binary::<i32>().value(row).to_vec()),
        ArrowType::LargeBinary => Scalar::Binary(array.as_binary::<i64>().value(row).to_vec()),
        ArrowType::BinaryView => Scalar::Binary(array.as_binary_view().value(row).to_vec()),
        _ => return Err(missing()),
    })
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
