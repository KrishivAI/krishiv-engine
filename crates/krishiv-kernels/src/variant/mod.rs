// SPDX-License-Identifier: Apache-2.0
//! `VARIANT` kernels: reading and building variant columns, Spark's
//! `variant_get` path language, the JSON printer behind `to_json` and
//! `CAST(v AS STRING)`, and the casts into and out of `VARIANT`.
//!
//! A variant column is the struct `{metadata, value}` of
//! [`krishiv_types::arrow`] (children found by name; `BinaryView`, `Binary`
//! or `LargeBinary`). Values are the unshredded Parquet Variant v1 bytes
//! read through [`krishiv_types::variant::VariantRef`].
//!
//! Ground truth: apache/spark v4.2.0 `variantExpressions.scala`
//! (`VariantPathParser`, `VariantGet.cast`), `Variant.java` (`toJson`),
//! `VariantExpressionEvalUtils.scala` (`castToVariant`), checked against
//! the `variant-field-extractions.sql` goldens.

mod build;
mod cast;

use std::sync::Arc;

use arrow_array::builder::{BinaryViewBuilder, StringViewBuilder};
use arrow_array::cast::AsArray as _;
use arrow_array::{Array, ArrayRef, BinaryViewArray, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{InvalidVariantGetPath, KrishivError, UnknownPrimitiveTypeInVariant};
use krishiv_types::DataType;
use krishiv_types::arrow::{VARIANT_METADATA_FIELD, VARIANT_VALUE_FIELD};
use krishiv_types::variant::{VariantKind, VariantRef, strip_trailing_zeros};

use crate::cast::TimeZoneRules;
use crate::cast::datetime::{format_date, format_timestamp, utc_to_local};

pub use build::cast_to_variant;
pub use cast::{cast_from_variant, cast_values};

fn internal(message: &str) -> KrishivError {
    krishiv_common::error::InternalError {
        message: format!("variant kernel: {message}"),
    }
    .into()
}

/// A variant column's two byte children and its validity.
#[derive(Debug)]
pub struct VariantColumn {
    nulls: Option<NullBuffer>,
    metadata: BinaryViewArray,
    value: BinaryViewArray,
}

/// A binary child as `BinaryView`.
fn binary_view(a: &ArrayRef) -> Result<BinaryViewArray, KrishivError> {
    match a.data_type() {
        ArrowType::BinaryView => Ok(a.as_binary_view().clone()),
        ArrowType::Binary => Ok(a.as_binary::<i32>().iter().collect()),
        ArrowType::LargeBinary => Ok(a.as_binary::<i64>().iter().collect()),
        other => Err(internal(&format!("a variant child of type {other}"))),
    }
}

impl VariantColumn {
    /// Reads `col`, a variant struct (or an all-NULL `Null` column).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `col` is not a variant column.
    pub fn new(col: &dyn Array) -> Result<Self, KrishivError> {
        if col.data_type() == &ArrowType::Null {
            let empty = BinaryViewArray::from_iter_values(std::iter::repeat_n(&[][..], col.len()));
            return Ok(Self {
                nulls: Some(NullBuffer::new_null(col.len())),
                metadata: empty.clone(),
                value: empty,
            });
        }
        let s = col
            .as_struct_opt()
            .ok_or_else(|| internal(&format!("a variant argument of type {}", col.data_type())))?;
        let child = |name: &str| {
            s.column_by_name(name)
                .ok_or_else(|| internal(&format!("a variant without `{name}`")))
                .and_then(binary_view)
        };
        Ok(Self {
            nulls: s.nulls().cloned(),
            metadata: child(VARIANT_METADATA_FIELD)?,
            value: child(VARIANT_VALUE_FIELD)?,
        })
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// Whether row `i` is SQL `NULL`.
    #[must_use]
    pub fn is_null(&self, i: usize) -> bool {
        self.nulls.as_ref().is_some_and(|n| n.is_null(i)) || self.value.is_null(i)
    }

    /// The value at row `i`, `None` for SQL `NULL`.
    ///
    /// # Errors
    /// `MALFORMED_VARIANT` for bytes that are not a valid variant.
    pub fn get(&self, i: usize) -> Result<Option<VariantRef<'_>>, KrishivError> {
        if self.is_null(i) {
            return Ok(None);
        }
        VariantRef::new(self.metadata.value(i), self.value.value(i)).map(Some)
    }
}

/// Builds a variant column.
#[derive(Debug)]
pub struct VariantColumnBuilder {
    metadata: BinaryViewBuilder,
    value: BinaryViewBuilder,
    valid: Vec<bool>,
}

impl VariantColumnBuilder {
    /// A builder for about `rows` rows.
    #[must_use]
    pub fn with_capacity(rows: usize) -> Self {
        Self {
            metadata: BinaryViewBuilder::with_capacity(rows),
            value: BinaryViewBuilder::with_capacity(rows),
            valid: Vec::with_capacity(rows),
        }
    }

    /// Appends a value.
    pub fn append(&mut self, metadata: &[u8], value: &[u8]) {
        self.metadata.append_value(metadata);
        self.value.append_value(value);
        self.valid.push(true);
    }

    /// Appends SQL `NULL` (both children hold empty bytes: they are never null).
    pub fn append_null(&mut self) {
        self.metadata.append_value([]);
        self.value.append_value([]);
        self.valid.push(false);
    }

    /// The finished column, of type `DataType::Variant.to_arrow()`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if Arrow refuses the struct.
    pub fn finish(mut self) -> Result<ArrayRef, KrishivError> {
        let ArrowType::Struct(fields) = DataType::Variant.to_arrow() else {
            return Err(internal("the variant type is not a struct"));
        };
        let nulls = self
            .valid
            .contains(&false)
            .then(|| NullBuffer::from(self.valid));
        let children: Vec<ArrayRef> = vec![
            Arc::new(self.metadata.finish()),
            Arc::new(self.value.finish()),
        ];
        StructArray::try_new(fields, children, nulls)
            .map(|a| Arc::new(a) as ArrayRef)
            .map_err(|e| internal(&e.to_string()))
    }
}

/// One step of a `variant_get` path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    /// `.key`, `['key']` or `["key"]`.
    Key(String),
    /// `[n]`.
    Index(usize),
}

/// Spark's `VariantPathParser`: `$` followed by `.key` (up to the next `.`
/// or `[`), `['key']`, `["key"]` (no escapes) or `[digits]`; `None` when
/// the text does not parse. An index past `INT` is `None` too, so it raises
/// `INVALID_VARIANT_GET_PATH` where Spark's `toInt` lets a
/// `NumberFormatException` escape (a deliberate divergence).
#[must_use]
pub fn parse_path(path: &str) -> Option<Vec<PathSegment>> {
    let mut rest = path.strip_prefix('$')?;
    let mut out = vec![];
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix('.') {
            let end = r.find(['.', '[']).unwrap_or(r.len());
            if end == 0 {
                return None;
            }
            out.push(PathSegment::Key(r[..end].to_owned()));
            rest = &r[end..];
        } else if let Some(r) = rest.strip_prefix("['") {
            let end = r.find('\'')?;
            let after = r[end..].strip_prefix("']")?;
            out.push(PathSegment::Key(r[..end].to_owned()));
            rest = after;
        } else if let Some(r) = rest.strip_prefix("[\"") {
            let end = r.find('"')?;
            let after = r[end..].strip_prefix("\"]")?;
            out.push(PathSegment::Key(r[..end].to_owned()));
            rest = after;
        } else {
            let r = rest.strip_prefix('[')?;
            let end = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
            if end == 0 {
                return None;
            }
            let after = r[end..].strip_prefix(']')?;
            // `index.toInt` on more digits than an int holds throws.
            let index = r[..end].parse::<i32>().ok()?;
            out.push(PathSegment::Index(usize::try_from(index).ok()?));
            rest = after;
        }
    }
    Some(out)
}

/// [`parse_path`], or `INVALID_VARIANT_GET_PATH` naming `function`.
///
/// # Errors
/// `INVALID_VARIANT_GET_PATH` when the path does not parse.
pub fn parse_path_or_raise(path: &str, function: &str) -> Result<Vec<PathSegment>, KrishivError> {
    parse_path(path).ok_or_else(|| {
        InvalidVariantGetPath {
            path: path.to_owned(),
            function_name: format!("`{function}`"),
        }
        .into()
    })
}

/// The value at `path` inside `v`, `None` when a step misses (a key that
/// is absent, an index out of range, or a step of the wrong kind).
///
/// # Errors
/// `MALFORMED_VARIANT` for bytes that fail validation on access.
pub fn extract<'a>(
    v: VariantRef<'a>,
    path: &[PathSegment],
) -> Result<Option<VariantRef<'a>>, KrishivError> {
    let mut v = v;
    for step in path {
        v = match (step, v.kind()) {
            (PathSegment::Key(k), VariantKind::Object) => match v.get_field(k)? {
                Some(f) => f,
                None => return Ok(None),
            },
            (PathSegment::Index(i), VariantKind::Array) if *i < v.len()? => v.get_index(*i)?,
            _ => return Ok(None),
        };
    }
    Ok(Some(v))
}

/// Jackson's `writeString`: quotes, `\"`, `\\`, the short escapes of
/// `\b \t \n \f \r`, `\u00XX` (upper-case hex) for the other control
/// characters; everything else as is.
pub fn push_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `BigDecimal.toPlainString` of `unscaled × 10^-scale`.
pub(crate) fn plain_decimal(unscaled: i128, scale: u8, out: &mut String) {
    let digits = unscaled.unsigned_abs().to_string();
    if unscaled < 0 {
        out.push('-');
    }
    let scale = usize::from(scale);
    if scale == 0 {
        out.push_str(&digits);
    } else if digits.len() > scale {
        let (int, frac) = digits.split_at(digits.len().saturating_sub(scale));
        out.push_str(int);
        out.push('.');
        out.push_str(frac);
    } else {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', scale.saturating_sub(digits.len())));
        out.push_str(&digits);
    }
}

/// The canonical text of a UUID: 8-4-4-4-12 lower-case hex digits.
pub(crate) fn uuid_string(b: [u8; 16]) -> String {
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// The offset suffix of Java's `appendOffset("+HH:MM", "+00:00")`: hours
/// and minutes only (an offset's seconds, as in a zone's local mean time, are
/// not printed).
fn push_offset(seconds: i32, out: &mut String) {
    let sign = if seconds < 0 { '-' } else { '+' };
    let abs = seconds.unsigned_abs();
    out.push(sign);
    out.push_str(&format!("{:02}:{:02}", abs / 3600, abs / 60 % 60));
}

fn unknown_type(v: &VariantRef<'_>) -> KrishivError {
    let id = v.value_bytes().first().map_or(0, |h| h >> 2);
    UnknownPrimitiveTypeInVariant { id: id.to_string() }.into()
}

/// Spark's `Variant.toJson`: object keys in stored (sorted) order, strings
/// escaped as Jackson does, doubles and floats as Java prints them (quoted
/// when not finite), decimals plain without trailing zeros, dates
/// `yyyy-MM-dd`, timestamps in `zone` with the offset, binary as base64,
/// all quoted.
///
/// # Errors
/// `MALFORMED_VARIANT` for bytes that fail validation on access;
/// `UNKNOWN_PRIMITIVE_TYPE_IN_VARIANT` for the time and nanosecond types
/// Spark 4.2.0 does not print.
pub fn write_json(
    v: &VariantRef<'_>,
    zone: &dyn TimeZoneRules,
    out: &mut String,
) -> Result<(), KrishivError> {
    match v.kind() {
        VariantKind::Object => {
            out.push('{');
            for i in 0..v.len()? {
                if i > 0 {
                    out.push(',');
                }
                push_json_string(v.field_key_at(i)?, out);
                out.push(':');
                write_json(&v.field_at(i)?, zone, out)?;
            }
            out.push('}');
        }
        VariantKind::Array => {
            out.push('[');
            for i in 0..v.len()? {
                if i > 0 {
                    out.push(',');
                }
                write_json(&v.get_index(i)?, zone, out)?;
            }
            out.push(']');
        }
        VariantKind::Null => out.push_str("null"),
        VariantKind::Bool => out.push_str(if v.as_bool()? { "true" } else { "false" }),
        VariantKind::Int8 | VariantKind::Int16 | VariantKind::Int32 | VariantKind::Int64 => {
            out.push_str(&v.as_i64()?.to_string());
        }
        VariantKind::String => push_json_string(v.as_str()?, out),
        VariantKind::Double => {
            let d = v.as_f64()?;
            let s = crate::text::java_double_string(d);
            if d.is_finite() {
                out.push_str(&s);
            } else {
                push_json_string(&s, out);
            }
        }
        VariantKind::Float => {
            let f = v.as_f32()?;
            let s = crate::text::java_float_string32(f);
            if f.is_finite() {
                out.push_str(&s);
            } else {
                push_json_string(&s, out);
            }
        }
        VariantKind::Decimal { .. } => {
            let (unscaled, scale) = v.as_decimal()?;
            let (unscaled, scale) = strip_trailing_zeros(unscaled, scale);
            plain_decimal(unscaled, scale, out);
        }
        VariantKind::Date => {
            let mut s = String::new();
            format_date(i64::from(v.as_date()?), &mut s);
            push_json_string(&s, out);
        }
        VariantKind::Timestamp => {
            let utc = v.as_timestamp_micros()?;
            let offset = zone.offset_at_utc(utc.div_euclid(1_000_000));
            let local =
                utc_to_local(utc, zone).ok_or_else(|| internal("a timestamp past the range"))?;
            let mut s = String::new();
            format_timestamp(local, &mut s);
            push_offset(offset, &mut s);
            push_json_string(&s, out);
        }
        VariantKind::TimestampNtz => {
            let mut s = String::new();
            format_timestamp(v.as_timestamp_micros()?, &mut s);
            push_json_string(&s, out);
        }
        VariantKind::Binary => push_json_string(&crate::json::base64_encode(v.as_bytes()?), out),
        VariantKind::Uuid => push_json_string(&uuid_string(v.as_uuid()?), out),
        _ => return Err(unknown_type(v)),
    }
    Ok(())
}

/// [`write_json`] of every row of `col` as a `Utf8View` column; SQL `NULL`
/// stays `NULL`.
///
/// # Errors
/// As [`write_json`] and [`VariantColumn::new`].
pub fn to_json(col: &dyn Array, zone: &dyn TimeZoneRules) -> Result<ArrayRef, KrishivError> {
    let col = VariantColumn::new(col)?;
    let mut out = StringViewBuilder::with_capacity(col.len());
    let mut text = String::new();
    for i in 0..col.len() {
        match col.get(i)? {
            Some(v) => {
                text.clear();
                write_json(&v, zone, &mut text)?;
                out.append_value(&text);
            }
            None => out.append_null(),
        }
    }
    Ok(Arc::new(out.finish()))
}

#[cfg(test)]
mod tests;
