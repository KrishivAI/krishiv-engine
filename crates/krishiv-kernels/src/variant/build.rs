// SPDX-License-Identifier: Apache-2.0
//! `castToVariant`: a value of a Spark type as a variant.
//!
//! Integers become the smallest integer width, decimals keep their scale,
//! strings, binaries, dates and timestamps keep their types, a variant is
//! re-keyed into the new value, arrays become arrays and (for
//! `to_variant_object`) maps with string keys and structs become objects;
//! a NULL nested inside is a variant null, a NULL at the top is SQL `NULL`.
//! A struct or map with a repeated key is `VARIANT_DUPLICATE_KEY`.

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::KrishivError;
use krishiv_types::DataType;
use krishiv_types::variant::VariantWriter;

use super::{VariantColumn, VariantColumnBuilder, internal};

/// Every row of `col` (of Spark type `from`) as a variant.
///
/// # Errors
/// `VARIANT_DUPLICATE_KEY`, `VARIANT_SIZE_LIMIT`, and `INTERNAL_ERROR` for a
/// type with no variant form (the analyzer refuses those first).
pub fn cast_to_variant(col: &dyn Array, from: &DataType) -> Result<ArrayRef, KrishivError> {
    let mut out = VariantColumnBuilder::with_capacity(col.len());
    for row in 0..col.len() {
        if col.is_null(row) || *from == DataType::Null {
            out.append_null();
            continue;
        }
        let mut w = VariantWriter::new();
        write(&mut w, col, row, from)?;
        let (m, v) = w.finish()?;
        out.append(&m, &v);
    }
    out.finish()
}

fn bytes_at(col: &dyn Array, row: usize) -> Option<&[u8]> {
    match col.data_type() {
        ArrowType::BinaryView => Some(col.as_binary_view().value(row)),
        ArrowType::Binary => Some(col.as_binary::<i32>().value(row)),
        ArrowType::LargeBinary => Some(col.as_binary::<i64>().value(row)),
        ArrowType::Utf8View => Some(col.as_string_view().value(row).as_bytes()),
        ArrowType::Utf8 => Some(col.as_string::<i32>().value(row).as_bytes()),
        ArrowType::LargeUtf8 => Some(col.as_string::<i64>().value(row).as_bytes()),
        _ => None,
    }
}

fn mismatch(col: &dyn Array, t: &DataType) -> KrishivError {
    internal(&format!(
        "a {} value on an Arrow {} carrier",
        t.sql(),
        col.data_type()
    ))
}

/// Writes row `row` of `col` (non-null or nested) as type `t`.
fn write(
    w: &mut VariantWriter,
    col: &dyn Array,
    row: usize,
    t: &DataType,
) -> Result<(), KrishivError> {
    if col.is_null(row) {
        w.null();
        return Ok(());
    }
    let bad = || mismatch(col, t);
    match t {
        DataType::Boolean => w.boolean(col.as_boolean_opt().ok_or_else(bad)?.value(row)),
        DataType::Byte => w.long(i64::from(
            col.as_primitive_opt::<Int8Type>()
                .ok_or_else(bad)?
                .value(row),
        )),
        DataType::Short => w.long(i64::from(
            col.as_primitive_opt::<Int16Type>()
                .ok_or_else(bad)?
                .value(row),
        )),
        DataType::Integer => w.long(i64::from(
            col.as_primitive_opt::<Int32Type>()
                .ok_or_else(bad)?
                .value(row),
        )),
        DataType::Long => w.long(
            col.as_primitive_opt::<Int64Type>()
                .ok_or_else(bad)?
                .value(row),
        ),
        DataType::Float => w.float(
            col.as_primitive_opt::<Float32Type>()
                .ok_or_else(bad)?
                .value(row),
        ),
        DataType::Double => w.double(
            col.as_primitive_opt::<Float64Type>()
                .ok_or_else(bad)?
                .value(row),
        ),
        DataType::Decimal(d) => {
            let unscaled = match col.data_type() {
                ArrowType::Decimal64(..) => {
                    i128::from(col.as_primitive::<Decimal64Type>().value(row))
                }
                ArrowType::Decimal128(..) => col.as_primitive::<Decimal128Type>().value(row),
                _ => return Err(bad()),
            };
            let scale = d.scale();
            w.decimal(unscaled, scale);
        }
        DataType::String | DataType::CollatedString(_) => {
            let b = bytes_at(col, row).ok_or_else(bad)?;
            w.string(&String::from_utf8_lossy(b))?;
        }
        DataType::Binary => w.binary(bytes_at(col, row).ok_or_else(bad)?)?,
        DataType::Date => w.date(
            col.as_primitive_opt::<Date32Type>()
                .ok_or_else(bad)?
                .value(row),
        ),
        DataType::Timestamp => w.timestamp(
            col.as_primitive_opt::<TimestampMicrosecondType>()
                .ok_or_else(bad)?
                .value(row),
        ),
        DataType::TimestampNtz => w.timestamp_ntz(
            col.as_primitive_opt::<TimestampMicrosecondType>()
                .ok_or_else(bad)?
                .value(row),
        ),
        DataType::Variant => {
            let c = VariantColumn::new(col)?;
            if let Some(v) = c.get(row)? {
                w.append_variant(&v)?;
            } else {
                w.null();
            }
        }
        DataType::Array(a) => {
            let (values, start, end) = match col.data_type() {
                ArrowType::List(_) => {
                    let l = col.as_list::<i32>();
                    let o = l.value_offsets();
                    (
                        Arc::clone(l.values()),
                        as_usize(o[row])?,
                        as_usize(o[row.saturating_add(1)])?,
                    )
                }
                ArrowType::LargeList(_) => {
                    let l = col.as_list::<i64>();
                    let o = l.value_offsets();
                    (
                        Arc::clone(l.values()),
                        as_usize(o[row])?,
                        as_usize(o[row.saturating_add(1)])?,
                    )
                }
                _ => return Err(bad()),
            };
            let at = w.position();
            let mut offsets = Vec::with_capacity(end.saturating_sub(start));
            for i in start..end {
                offsets.push(w.position().saturating_sub(at));
                write(w, values.as_ref(), i, a.element_type())?;
            }
            w.finish_array(at, &offsets)?;
        }
        DataType::Map(m) => {
            let map = col.as_map_opt().ok_or_else(bad)?;
            let o = map.value_offsets();
            let (start, end) = (as_usize(o[row])?, as_usize(o[row.saturating_add(1)])?);
            let keys = map.keys();
            let values = map.values();
            let at = w.position();
            let mut fields = Vec::with_capacity(end.saturating_sub(start));
            for i in start..end {
                let key = bytes_at(keys.as_ref(), i).ok_or_else(bad)?;
                fields.push(w.field(at, &String::from_utf8_lossy(key))?);
                write(w, values.as_ref(), i, m.value_type())?;
            }
            w.finish_object(at, fields)?;
        }
        DataType::Struct(s) => {
            let st = col.as_struct_opt().ok_or_else(bad)?;
            let at = w.position();
            let mut fields = Vec::with_capacity(s.fields().len());
            for (i, f) in s.fields().iter().enumerate() {
                fields.push(w.field(at, f.name())?);
                write(w, st.column(i).as_ref(), row, f.data_type())?;
            }
            w.finish_object(at, fields)?;
        }
        other => return Err(internal(&format!("a cast of {} to VARIANT", other.sql()))),
    }
    Ok(())
}

fn as_usize<T: TryInto<usize>>(v: T) -> Result<usize, KrishivError> {
    v.try_into().map_err(|_| internal("a negative list offset"))
}
