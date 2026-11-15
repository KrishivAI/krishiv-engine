// SPDX-License-Identifier: Apache-2.0
//! `ParseJson` and `ToJson` (ports of the kernel's `parse_json_impl` and `to_json`,
//! `delta_kernel` 0.30.1 `engine/arrow_utils/mod.rs` and
//! `engine/arrow_expression/evaluate_expression.rs`, Apache-2.0).
//!
//! `ParseJson`:
//! - one JSON object per string, a NULL string a row of NULLs (`{}`);
//! - leaves of type `timestamp`, `timestamp_ntz`, `date` and `decimal` inside structs are
//!   decoded as strings and cast back safely, so a bad cell of one of them is NULL and the
//!   rest of the row survives (an extended-year timestamp, a decimal over its precision);
//! - anything else unparseable (an empty string, a malformed object, a string where the
//!   schema has a number) makes the whole batch's result a NULL struct per row, so the files
//!   it describes are kept rather than the scan failing;
//! - the parse limits ([`crate::engine::DeltaLimits`]) apply per string and raise
//!   `KRISHIV_CATALOG_LIMIT_EXCEEDED`;
//! - numbers and booleans read into a string leaf as their text (`with_coerce_primitive`).
//!
//! `ToJson` writes a struct column as one JSON object per row (a NULL row is NULL). Timestamps
//! are truncated to milliseconds (`%.3f`, floored below the epoch) and a `timestamp` ends in a
//! literal `Z`; binary is hex.

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray, new_null_array};
use arrow_json::writer::{EncoderOptions, make_encoder};
use arrow_json::{ReaderBuilder, StructMode};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use delta_kernel::schema::{DataType, PrimitiveType, SchemaRef, StructField, StructType};
use delta_kernel::{KernelError, KernelResult};

use super::types::cast_to;
use super::{Ctx, arrow_err};
use crate::engine::json::{check_depth, check_size};
use crate::engine::relabel::{ARROW_UTC, relabel, with_zone};
use crate::engine::schema::to_arrow_schema;
use crate::error::to_kernel;

/// `ToJson`'s `timestamp_ntz` format: milliseconds, truncated.
const NTZ_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.3f";
/// `ToJson`'s `timestamp` format: milliseconds, truncated, and a literal `Z` (every Delta
/// writer emits `Z`; a `timestamp` stat is always UTC).
const TZ_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.3fZ";

/// The JSON text of each row of the struct column `input`.
///
/// # Errors
/// A non-struct input, or an encoder failure.
pub(crate) fn to_json(input: &ArrayRef) -> KernelResult<ArrayRef> {
    let Some(s) = input.as_struct_opt() else {
        return Err(KernelError::generic(format!(
            "TO_JSON can only be applied to struct arrays, got {:?}",
            input.data_type()
        )));
    };
    let rows = s.len();
    // Zoned timestamps are formatted at `+00:00`, which Arrow can resolve without chrono-tz.
    let zoned = relabel(input, &with_zone(input.data_type(), ARROW_UTC)).map_err(arrow_err)?;
    let s = zoned.as_struct();
    let field = Arc::new(Field::new_struct("root", s.fields().clone(), true));
    let options = EncoderOptions::default()
        .with_struct_mode(StructMode::ObjectOnly)
        .with_timestamp_format(NTZ_FORMAT.to_owned())
        .with_timestamp_tz_format(TZ_FORMAT.to_owned());
    let mut encoder = make_encoder(&field, s, &options).map_err(arrow_err)?;
    let mut out = StringBuilder::with_capacity(rows, rows.saturating_mul(64));
    let mut buf = Vec::with_capacity(256);
    for row in 0..rows {
        if s.is_null(row) {
            out.append_null();
            continue;
        }
        buf.clear();
        encoder.encode(row, &mut buf);
        let text = std::str::from_utf8(&buf)
            .map_err(|e| KernelError::generic(format!("ToJson wrote invalid UTF-8: {e}")))?;
        out.append_value(text);
    }
    Ok(Arc::new(out.finish()))
}

/// Whether `t` is a leaf whose typed decoder fails the whole batch on one bad cell.
fn failure_prone(t: &PrimitiveType) -> bool {
    matches!(
        t,
        PrimitiveType::Timestamp
            | PrimitiveType::TimestampNtz
            | PrimitiveType::Date
            | PrimitiveType::Decimal(_)
    )
}

/// `schema` with its failure-prone leaves (inside structs only) read as strings, or borrowed
/// when there are none.
fn stringify(schema: &StructType) -> KernelResult<Cow<'_, StructType>> {
    fn field(f: &StructField) -> KernelResult<Option<StructField>> {
        let t = match f.data_type() {
            DataType::Primitive(p) if failure_prone(p) => DataType::STRING,
            DataType::Struct(s) => match stringify(s)? {
                Cow::Borrowed(_) => return Ok(None),
                Cow::Owned(s) => DataType::Struct(Box::new(s)),
            },
            _ => return Ok(None),
        };
        Ok(Some(
            StructField::new(f.name(), t, f.is_nullable()).with_metadata(f.metadata().clone()),
        ))
    }
    let mut changed = false;
    let mut fields = Vec::with_capacity(schema.num_fields());
    for f in schema.fields() {
        match field(f)? {
            Some(new) => {
                changed = true;
                fields.push(new);
            }
            None => fields.push(f.clone()),
        }
    }
    if changed {
        Ok(Cow::Owned(StructType::try_new(fields)?))
    } else {
        Ok(Cow::Borrowed(schema))
    }
}

/// Decodes one object per string into `schema` (zones labelled for arrow-json).
fn decode<'s>(
    strings: impl Iterator<Item = Option<&'s str>>,
    rows: usize,
    schema: Arc<Schema>,
) -> Result<RecordBatch, String> {
    if rows == 0 {
        return Ok(RecordBatch::new_empty(schema));
    }
    let mut decoder = ReaderBuilder::new(schema)
        .with_batch_size(rows)
        .with_coerce_primitive(true)
        .build_decoder()
        .map_err(|e| e.to_string())?;
    for (text, row) in strings.zip(1..) {
        let line = text.unwrap_or("{}");
        let read = decoder.decode(line.as_bytes()).map_err(|e| e.to_string())?;
        if read != line.len() || decoder.has_partial_record() || decoder.len() != row {
            return Err(format!(
                "Malformed JSON: Multiple, partial, or 0 JSON objects on row {row}"
            ));
        }
    }
    match decoder.flush().map_err(|e| e.to_string())? {
        Some(b) if b.num_rows() == rows => Ok(b),
        _ => Err("Malformed JSON: nothing decoded".to_owned()),
    }
}

/// Visits each string of the string column `strings`.
fn with_strings<T>(
    strings: &ArrayRef,
    f: impl FnOnce(&mut dyn Iterator<Item = Option<&str>>) -> T,
) -> KernelResult<T> {
    if let Some(s) = strings.as_string_opt::<i32>() {
        Ok(f(&mut s.iter()))
    } else if let Some(s) = strings.as_string_opt::<i64>() {
        Ok(f(&mut s.iter()))
    } else if let Some(s) = strings.as_string_view_opt() {
        Ok(f(&mut s.iter()))
    } else {
        Err(KernelError::generic(format!(
            "Expected string array for JSON parsing, got {}",
            strings.data_type()
        )))
    }
}

/// Evaluates `ParseJson` over the string column `strings` (see the module docs).
///
/// # Errors
/// A non-string input, or a parse limit exceeded.
pub(crate) fn parse_json(
    ctx: &Ctx,
    strings: &ArrayRef,
    schema: &SchemaRef,
) -> KernelResult<ArrayRef> {
    let source = "a JSON value of the Delta log";
    with_strings(strings, |it| {
        for text in it.flatten() {
            check_size(&ctx.limits, source, text.len() as u64)?;
            check_depth(&ctx.limits, source, text.as_bytes())?;
        }
        Ok(())
    })?
    .map_err(to_kernel)?;
    let target = Arc::new(to_arrow_schema(schema)?);
    let target_type = ArrowType::Struct(target.fields().clone());
    let rows = strings.len();
    let relaxed = stringify(schema)?;
    let decode_schema = Arc::new(to_arrow_schema(&relaxed)?);
    let decode_schema = Arc::new(crate::engine::relabel::schema_with_zone(
        &decode_schema,
        ARROW_UTC,
    ));
    let decoded = with_strings(strings, |it| decode(it, rows, decode_schema))?;
    let parsed = decoded.and_then(|b| {
        let array: ArrayRef = Arc::new(StructArray::from(b));
        cast_to(&array, &with_zone(&target_type, ARROW_UTC), true)
            .and_then(|a| relabel(&a, &target_type).map_err(arrow_err))
            .map_err(|e| e.to_string())
    });
    Ok(match parsed {
        Ok(a) => a,
        // The kernel's rule: unparseable stats become NULL stats, which keep their files.
        Err(_) => new_null_array(&target_type, rows),
    })
}
