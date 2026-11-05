// SPDX-License-Identifier: Apache-2.0
//! `VARIANT` columns in Parquet (design P2-T001c section 2.3, Parquet
//! `VariantShredding.md`).
//!
//! **Reading.** A top-level group is a variant when it is annotated
//! `VARIANT` or, for files Spark wrote with
//! `spark.sql.parquet.variant.annotateLogicalType.enabled=false`, when the
//! footer's Spark schema (`org.apache.spark.sql.parquet.row.metadata`) types
//! it `variant`. Its children are found by name, never position (Spark writes
//! `value, metadata` with shredding off). arrow-rs decodes the group as a
//! plain struct; [`to_memory`] turns it into the in-memory form (one
//! unshredded `(metadata, value)` pair per row), rebuilding shredded values
//! as the specification's `construct_variant` does: a missing `typed_value`
//! and `value` is a missing field (a variant null at the top), an object
//! merges its shredded fields with the residual object in `value`, an array
//! rebuilds each element. Rebuilt values get a fresh dictionary, so they are
//! equal to the stored ones by `to_json`, not by bytes.
//!
//! **Writing** is unshredded, as Spark 4.2 writes by default: the group
//! `{metadata, value}` (both required binary) annotated `VARIANT(1)`
//! ([`annotate`]); the footer's Spark schema says `variant` too.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::{
    Date32Type, Decimal32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType,
};
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::{DataType, FieldRef, TimeUnit};
use krishiv_common::error::KrishivError;
use krishiv_kernels::variant::VariantColumnBuilder;
use krishiv_types::DataType as SparkType;
use krishiv_types::variant::{VariantRef, VariantWriter};
use parquet::basic::LogicalType;
use parquet::file::metadata::KeyValue;
use parquet::schema::types::{SchemaDescriptor, Type, TypePtr};

use crate::error::{FormatError, FormatResult};

/// Footer key of Spark's schema JSON.
const ROW_METADATA_KEY: &str = "org.apache.spark.sql.parquet.row.metadata";

fn corrupt(what: &str) -> FormatError {
    FormatError::Corrupt(format!("a VARIANT column {what}"))
}

fn other(e: KrishivError) -> FormatError {
    FormatError::Other(e)
}

/// The names of the top-level fields of the file that are variants.
pub(super) fn variant_columns(
    descr: &SchemaDescriptor,
    footer: Option<&Vec<KeyValue>>,
) -> HashSet<String> {
    let spark = footer
        .into_iter()
        .flatten()
        .find(|kv| kv.key == ROW_METADATA_KEY)
        .and_then(|kv| kv.value.as_deref())
        .and_then(|json| SparkType::from_json(json).ok());
    let spark_variant = |name: &str| match &spark {
        Some(SparkType::Struct(s)) => s
            .fields()
            .iter()
            .any(|f| f.name() == name && *f.data_type() == SparkType::Variant),
        _ => false,
    };
    descr
        .root_schema()
        .get_fields()
        .iter()
        .filter(|t| {
            t.is_group()
                && has_children(t)
                && (matches!(
                    t.get_basic_info().logical_type_ref(),
                    Some(LogicalType::Variant(_))
                ) || spark_variant(t.name()))
        })
        .map(|t| t.name().to_owned())
        .collect()
}

/// Whether a group has the `metadata` and `value` (or `typed_value`)
/// children of a variant.
fn has_children(t: &TypePtr) -> bool {
    let names: Vec<&str> = t.get_fields().iter().map(|c| c.name()).collect();
    names.contains(&"metadata") && (names.contains(&"value") || names.contains(&"typed_value"))
}

/// `field` as the engine's variant field, when `variants` names it.
pub(super) fn engine_field(field: &FieldRef, variants: &HashSet<String>) -> FieldRef {
    if variants.contains(field.name()) {
        Arc::new(SparkType::Variant.to_arrow_field(field.name(), field.is_nullable()))
    } else {
        Arc::clone(field)
    }
}

/// The child `name` of `s`, if any.
fn child<'a>(s: &'a StructArray, name: &str) -> Option<&'a ArrayRef> {
    s.column_by_name(name)
}

/// The bytes of row `i` of a binary child, `None` when NULL.
fn bytes(a: &ArrayRef, i: usize) -> FormatResult<Option<&[u8]>> {
    if a.is_null(i) {
        return Ok(None);
    }
    Ok(Some(match a.data_type() {
        DataType::BinaryView => a.as_binary_view().value(i),
        DataType::Binary => a.as_binary::<i32>().value(i),
        DataType::LargeBinary => a.as_binary::<i64>().value(i),
        other => return Err(corrupt(&format!("with a {other} binary child"))),
    }))
}

/// Converts a decoded variant group (physical struct) to the in-memory form.
///
/// # Errors
/// `Corrupt` for a group without the variant children, `MALFORMED_VARIANT`
/// for bytes that are no variant.
pub(super) fn to_memory(col: &ArrayRef) -> FormatResult<ArrayRef> {
    if krishiv_types::arrow::is_variant_struct(col.data_type()) && !has_typed(col) {
        return Ok(Arc::clone(col));
    }
    let s = col
        .as_struct_opt()
        .ok_or_else(|| corrupt("that is not a group"))?;
    let metadata = child(s, "metadata").ok_or_else(|| corrupt("without metadata"))?;
    let value = child(s, "value");
    let typed = child(s, "typed_value");
    let mut out = VariantColumnBuilder::with_capacity(s.len());
    for i in 0..s.len() {
        if s.is_null(i) {
            out.append_null();
            continue;
        }
        let m = bytes(metadata, i)?.ok_or_else(|| corrupt("with a NULL metadata"))?;
        match typed {
            // Unshredded: the stored bytes as they are.
            None => match value.map(|v| bytes(v, i)).transpose()?.flatten() {
                Some(v) => out.append(m, v),
                None => out.append(m, &[0]),
            },
            Some(_) => {
                let mut w = VariantWriter::new();
                if !rebuild(&mut w, m, value, typed, i)? {
                    // Required but missing: a variant null.
                    w.null();
                }
                let (meta, v) = w.finish().map_err(other)?;
                out.append(&meta, &v);
            }
        }
    }
    out.finish().map_err(other)
}

fn has_typed(col: &ArrayRef) -> bool {
    col.as_struct_opt()
        .is_some_and(|s| s.column_by_name("typed_value").is_some())
}

/// Appends the value of row `i` of a `(value, typed_value)` pair to `w`;
/// `false` (nothing written) when both are NULL (a missing value).
fn rebuild(
    w: &mut VariantWriter,
    metadata: &[u8],
    value: Option<&ArrayRef>,
    typed: Option<&ArrayRef>,
    i: usize,
) -> FormatResult<bool> {
    let residual = value.map(|v| bytes(v, i)).transpose()?.flatten();
    let residual = residual
        .map(|v| VariantRef::new(metadata, v))
        .transpose()
        .map_err(other)?;
    let Some(typed) = typed.filter(|t| t.is_valid(i)) else {
        return Ok(match residual {
            Some(r) => {
                w.append_variant(&r).map_err(other)?;
                true
            }
            None => false,
        });
    };
    match typed.data_type() {
        DataType::Struct(_) => {
            let fields = typed.as_struct();
            let start = w.position();
            let mut entries = Vec::new();
            let mut shredded = HashSet::new();
            for (f, column) in fields.fields().iter().zip(fields.columns()) {
                shredded.insert(f.name().as_str());
                let pair = column
                    .as_struct_opt()
                    .ok_or_else(|| corrupt("with a shredded field that is not a group"))?;
                let entry = w.field(start, f.name()).map_err(other)?;
                if rebuild(
                    w,
                    metadata,
                    child(pair, "value"),
                    child(pair, "typed_value"),
                    i,
                )? {
                    // A missing field writes nothing and has no entry.
                    entries.push(entry);
                }
            }
            // A partially shredded object: the residual holds the rest.
            if let Some(r) = residual {
                if r.kind() != krishiv_types::variant::VariantKind::Object {
                    return Err(corrupt("with a residual value that is not an object"));
                }
                for k in 0..r.len().map_err(other)? {
                    let key = r.field_key_at(k).map_err(other)?;
                    if shredded.contains(key) {
                        continue;
                    }
                    entries.push(w.field(start, key).map_err(other)?);
                    w.append_variant(&r.field_at(k).map_err(other)?)
                        .map_err(other)?;
                }
            }
            w.finish_object(start, entries).map_err(other)?;
        }
        DataType::List(_) | DataType::LargeList(_) => {
            let (values, from, to) = match typed.data_type() {
                DataType::List(_) => {
                    let l = typed.as_list::<i32>();
                    let o = l.value_offsets();
                    (Arc::clone(l.values()), to_usize(o[i])?, to_usize(o[i + 1])?)
                }
                _ => {
                    let l = typed.as_list::<i64>();
                    let o = l.value_offsets();
                    (Arc::clone(l.values()), to_usize(o[i])?, to_usize(o[i + 1])?)
                }
            };
            let element = values
                .as_struct_opt()
                .ok_or_else(|| corrupt("with a shredded element that is not a group"))?;
            let start = w.position();
            let mut offsets = Vec::with_capacity(to - from);
            for e in from..to {
                offsets.push(w.position() - start);
                if !rebuild(
                    w,
                    metadata,
                    child(element, "value"),
                    child(element, "typed_value"),
                    e,
                )? {
                    w.null();
                }
            }
            w.finish_array(start, &offsets).map_err(other)?;
        }
        _ => primitive(w, typed, i)?,
    }
    Ok(true)
}

fn to_usize<T: TryInto<usize>>(v: T) -> FormatResult<usize> {
    v.try_into()
        .map_err(|_| corrupt("with a negative list offset"))
}

/// Appends a shredded primitive as the variant type the specification maps
/// its Parquet type to.
fn primitive(w: &mut VariantWriter, a: &ArrayRef, i: usize) -> FormatResult<()> {
    let bad = |t: &DataType| corrupt(&format!("with a shredded {t}"));
    match a.data_type() {
        DataType::Boolean => w.boolean(a.as_boolean().value(i)),
        DataType::Int8 => w.long(i64::from(a.as_primitive::<Int8Type>().value(i))),
        DataType::Int16 => w.long(i64::from(a.as_primitive::<Int16Type>().value(i))),
        DataType::Int32 => w.long(i64::from(a.as_primitive::<Int32Type>().value(i))),
        DataType::Int64 => w.long(a.as_primitive::<Int64Type>().value(i)),
        DataType::Float32 => w.float(a.as_primitive::<Float32Type>().value(i)),
        DataType::Float64 => w.double(a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal32(_, s) => w.decimal(
            i128::from(a.as_primitive::<Decimal32Type>().value(i)),
            scale(*s)?,
        ),
        DataType::Decimal64(_, s) => w.decimal(
            i128::from(a.as_primitive::<Decimal64Type>().value(i)),
            scale(*s)?,
        ),
        DataType::Decimal128(_, s) => {
            w.decimal(a.as_primitive::<Decimal128Type>().value(i), scale(*s)?);
        }
        DataType::Date32 => w.date(a.as_primitive::<Date32Type>().value(i)),
        DataType::Timestamp(unit, tz) => {
            let micros = match unit {
                TimeUnit::Microsecond => a.as_primitive::<TimestampMicrosecondType>().value(i),
                TimeUnit::Millisecond => a
                    .as_primitive::<TimestampMillisecondType>()
                    .value(i)
                    .checked_mul(1000)
                    .ok_or_else(|| corrupt("with a timestamp past the range"))?,
                TimeUnit::Nanosecond => a
                    .as_primitive::<TimestampNanosecondType>()
                    .value(i)
                    .div_euclid(1000),
                TimeUnit::Second => return Err(bad(a.data_type())),
            };
            if tz.is_some() {
                w.timestamp(micros);
            } else {
                w.timestamp_ntz(micros);
            }
        }
        DataType::Utf8 => w.string(a.as_string::<i32>().value(i)).map_err(other)?,
        DataType::LargeUtf8 => w.string(a.as_string::<i64>().value(i)).map_err(other)?,
        DataType::Utf8View => w.string(a.as_string_view().value(i)).map_err(other)?,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            let b = bytes(a, i)?.ok_or_else(|| corrupt("with a NULL typed binary"))?;
            w.binary(b).map_err(other)?;
        }
        DataType::FixedSizeBinary(16) => {
            let b: [u8; 16] = a
                .as_fixed_size_binary()
                .value(i)
                .try_into()
                .map_err(|_| corrupt("with a short UUID"))?;
            w.uuid(b);
        }
        other => return Err(bad(other)),
    }
    Ok(())
}

fn scale(s: i8) -> FormatResult<u8> {
    u8::try_from(s).map_err(|_| corrupt("with a negative decimal scale"))
}

/// The Parquet schema of `fields` as arrow-rs converts it, with each
/// top-level variant group annotated `VARIANT(1)`.
///
/// # Errors
/// `WriteFailed` when the schema cannot be converted or rebuilt.
pub(super) fn annotate(
    descr: SchemaDescriptor,
    fields: &[FieldRef],
) -> FormatResult<SchemaDescriptor> {
    let variants: HashSet<&str> = fields
        .iter()
        .filter(|f| krishiv_types::arrow::is_variant_struct(f.data_type()))
        .map(|f| f.name().as_str())
        .collect();
    if variants.is_empty() {
        return Ok(descr);
    }
    let fail = |e: parquet::errors::ParquetError| FormatError::WriteFailed(e.to_string());
    let root = descr.root_schema();
    let children = root
        .get_fields()
        .iter()
        .map(|t| {
            if !variants.contains(t.name()) {
                return Ok(Arc::clone(t));
            }
            let info = t.get_basic_info();
            let mut b = Type::group_type_builder(t.name())
                .with_fields(t.get_fields().to_vec())
                .with_logical_type(Some(LogicalType::variant(Some(1))));
            if info.has_repetition() {
                b = b.with_repetition(info.repetition());
            }
            if info.has_id() {
                b = b.with_id(Some(info.id()));
            }
            b.build().map(Arc::new).map_err(fail)
        })
        .collect::<FormatResult<Vec<_>>>()?;
    let root = Type::group_type_builder(root.name())
        .with_fields(children)
        .build()
        .map_err(fail)?;
    Ok(SchemaDescriptor::new(Arc::new(root)))
}
