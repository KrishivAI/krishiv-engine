// SPDX-License-Identifier: Apache-2.0
//! The Avro ↔ Arrow schema mapping, following Spark's `SchemaConverters`.
//!
//! Read (Avro → Arrow):
//! primitives map to their Arrow counterpart (`int` → `Int32`, `long` →
//! `Int64`, `float` → `Float32`, `double` → `Float64`, `boolean` →
//! `Boolean`, `string` → `Utf8View`, `bytes` and `fixed` → `BinaryView`);
//! unions follow Spark's `SchemaConverters`: the null branches are dropped and
//! the union is nullable; what is left is `T` for one branch, `long` for
//! `int` and `long`, `double` for `float` and `double`, and otherwise a
//! struct of `member0..N` (one nullable member per remaining branch);
//! `enum` maps to
//! `Utf8View` (a string), `uuid` to the string or bytes holding it; `date` →
//! `Date32`; `timestamp-millis` and `timestamp-micros` → zoned microsecond
//! `Timestamp` (millis are scaled); `local-timestamp-millis`/`-micros` →
//! unzoned microsecond `Timestamp`; `time-millis` → `Int32`, `time-micros` →
//! `Int64`; `decimal(p, s)` → `Decimal128`; records → structs, arrays →
//! lists, maps → maps with string keys. `duration`, `BigDecimal`,
//! nanosecond timestamps and recursive (`ref`) schemas are refused
//! (`KRISHIV_UNSUPPORTED_FEATURE`), as they have no Spark type here.
//!
//! Write (Arrow → Avro) maps the same way back: nullable fields become
//! `["null", T]` unions, `Decimal128` a `fixed` decimal, zoned timestamps
//! `timestamp-micros` and unzoned ones `local-timestamp-micros`.
//! `byte`/`short` widen to `int` on the next read, as in Spark.

use std::sync::Arc;

use apache_avro::{Schema as AvroSchema, types::Value as AvroValue};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema, TimeUnit};

use crate::error::{FormatError, FormatResult};

/// `KRISHIV_UNSUPPORTED_FEATURE` as a format error.
fn unsupported(feature: String) -> FormatError {
    FormatError::Other(crate::error::unsupported(feature))
}

/// How Spark maps a union's branches once the null branches are dropped
/// (`SchemaConverters.toSqlTypeHelper`).
pub(crate) enum UnionShape<'a> {
    /// One branch: the union is that branch.
    Single(&'a AvroSchema),
    /// Exactly `int` and `long`: a `long`.
    Long,
    /// Exactly `float` and `double`: a `double`.
    Double,
    /// A struct of `member0..N`, one member per non-null branch.
    Members,
}

/// The shape of the union of `variants` (see [`UnionShape`]).
pub(crate) fn union_shape(variants: &[AvroSchema]) -> UnionShape<'_> {
    let branches: Vec<&AvroSchema> = variants
        .iter()
        .filter(|s| !matches!(s, AvroSchema::Null))
        .collect();
    match branches.as_slice() {
        [one] => UnionShape::Single(one),
        [a, b] => {
            let has = |wanted: &AvroSchema| {
                std::mem::discriminant(*a) == std::mem::discriminant(wanted)
                    || std::mem::discriminant(*b) == std::mem::discriminant(wanted)
            };
            if has(&AvroSchema::Int) && has(&AvroSchema::Long) {
                UnionShape::Long
            } else if has(&AvroSchema::Float) && has(&AvroSchema::Double) {
                UnionShape::Double
            } else {
                UnionShape::Members
            }
        }
        _ => UnionShape::Members,
    }
}

/// Whether `schema` is a union with a null branch.
fn is_nullable_union(schema: &AvroSchema) -> bool {
    null_branch_index(schema).is_some()
}

/// The Arrow type of an Avro schema.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a schema with no Spark type here.
fn arrow_of(schema: &AvroSchema) -> FormatResult<DataType> {
    Ok(match schema {
        AvroSchema::Null => DataType::Null,
        AvroSchema::Boolean => DataType::Boolean,
        AvroSchema::Int | AvroSchema::TimeMillis => DataType::Int32,
        AvroSchema::Long | AvroSchema::TimeMicros => DataType::Int64,
        AvroSchema::Float => DataType::Float32,
        AvroSchema::Double => DataType::Float64,
        AvroSchema::Bytes | AvroSchema::Fixed(_) => DataType::BinaryView,
        AvroSchema::String | AvroSchema::Enum(_) => DataType::Utf8View,
        AvroSchema::Array(a) => {
            let item = field_of("element", &a.items)?;
            DataType::List(Arc::new(item))
        }
        AvroSchema::Map(m) => {
            let value = field_of("value", &m.types)?;
            let entries = Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8View, false),
                        value.with_name("value"),
                    ]
                    .into(),
                ),
                false,
            );
            DataType::Map(Arc::new(entries), false)
        }
        AvroSchema::Union(u) => {
            let variants = u.variants();
            match union_shape(variants) {
                UnionShape::Single(inner) => return arrow_of(inner),
                UnionShape::Long => DataType::Int64,
                UnionShape::Double => DataType::Float64,
                UnionShape::Members => {
                    // Its fields are nullable as only one holds a value per
                    // row.
                    let members = variants
                        .iter()
                        .filter(|s| !matches!(s, AvroSchema::Null))
                        .enumerate()
                        .map(|(i, s)| {
                            field_of(&format!("member{i}"), s).map(|f| f.with_nullable(true))
                        })
                        .collect::<FormatResult<Vec<_>>>()?;
                    DataType::Struct(members.into())
                }
            }
        }
        AvroSchema::Record(r) => {
            let fields = r
                .fields
                .iter()
                .map(|f| field_of(&f.name, &f.schema))
                .collect::<FormatResult<Vec<_>>>()?;
            DataType::Struct(fields.into())
        }
        AvroSchema::Decimal(d) => {
            let precision = u8::try_from(d.precision)
                .ok()
                .filter(|p| (1..=38).contains(p));
            let scale = i8::try_from(d.scale).ok();
            match (precision, scale) {
                (Some(p), Some(s)) => DataType::Decimal128(p, s),
                _ => {
                    return Err(unsupported(format!(
                        "the Avro decimal precision {} scale {} (only precision 1..=38 is supported)",
                        d.precision, d.scale
                    )));
                }
            }
        }
        AvroSchema::BigDecimal => {
            return Err(unsupported(
                "the Avro bigdecimal logical type (no fixed scale)".to_owned(),
            ));
        }
        AvroSchema::Uuid(u) => match u {
            apache_avro::schema::UuidSchema::String => DataType::Utf8View,
            apache_avro::schema::UuidSchema::Bytes | apache_avro::schema::UuidSchema::Fixed(_) => {
                DataType::BinaryView
            }
        },
        AvroSchema::Date => DataType::Date32,
        AvroSchema::TimestampMillis | AvroSchema::TimestampMicros => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        AvroSchema::TimestampNanos | AvroSchema::LocalTimestampNanos => {
            return Err(unsupported(
                "the Avro (local-)timestamp-nanos logical type (only millis and micros are supported)"
                    .to_owned(),
            ));
        }
        AvroSchema::LocalTimestampMillis | AvroSchema::LocalTimestampMicros => {
            DataType::Timestamp(TimeUnit::Microsecond, None)
        }
        AvroSchema::Duration(_) => {
            return Err(unsupported(
                "the Avro duration logical type (it has no Spark type)".to_owned(),
            ));
        }
        AvroSchema::Ref { name } => {
            return Err(unsupported(format!(
                "the recursive Avro schema {} (recursive types are not supported)",
                name.fullname(None)
            )));
        }
    })
}

/// An Arrow field of an Avro schema: nullable exactly for unions with a null
/// branch.
fn field_of(name: &str, schema: &AvroSchema) -> FormatResult<Field> {
    let dt = arrow_of(schema)?;
    Ok(Field::new(name, dt, is_nullable_union(schema)))
}

/// The Arrow fields of a top-level Avro record schema.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a non-record schema or a field with
/// no Spark type.
pub(crate) fn fields_of_record(schema: &AvroSchema) -> FormatResult<Vec<Field>> {
    match schema {
        AvroSchema::Record(r) => r
            .fields
            .iter()
            .map(|f| field_of(&f.name, &f.schema))
            .collect(),
        other => Err(unsupported(format!(
            "the Avro top-level type {kind} (only records are supported)",
            kind = schema_kind(other)
        ))),
    }
}

/// A short kind name of an Avro schema, for messages.
pub(crate) fn schema_kind(schema: &AvroSchema) -> &'static str {
    match schema {
        AvroSchema::Null => "null",
        AvroSchema::Boolean => "boolean",
        AvroSchema::Int => "int",
        AvroSchema::Long => "long",
        AvroSchema::Float => "float",
        AvroSchema::Double => "double",
        AvroSchema::Bytes => "bytes",
        AvroSchema::String => "string",
        AvroSchema::Array(_) => "array",
        AvroSchema::Map(_) => "map",
        AvroSchema::Union(_) => "union",
        AvroSchema::Record(_) => "record",
        AvroSchema::Enum(_) => "enum",
        AvroSchema::Fixed(_) => "fixed",
        AvroSchema::Decimal(_) => "decimal",
        AvroSchema::BigDecimal => "bigdecimal",
        AvroSchema::Uuid(_) => "uuid",
        AvroSchema::Date => "date",
        AvroSchema::TimeMillis => "time-millis",
        AvroSchema::TimeMicros => "time-micros",
        AvroSchema::TimestampMillis => "timestamp-millis",
        AvroSchema::TimestampMicros => "timestamp-micros",
        AvroSchema::TimestampNanos => "timestamp-nanos",
        AvroSchema::LocalTimestampMillis => "local-timestamp-millis",
        AvroSchema::LocalTimestampMicros => "local-timestamp-micros",
        AvroSchema::LocalTimestampNanos => "local-timestamp-nanos",
        AvroSchema::Duration(_) => "duration",
        AvroSchema::Ref { .. } => "ref",
    }
}

/// The JSON of an Avro string.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// An Avro name: letters, digits and underscores, not starting with a
/// digit. Anything else becomes an underscore.
fn avro_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.starts_with(|c: char| c.is_ascii_digit()) || out.is_empty() {
        out.insert(0, '_');
    }
    out
}

/// Smallest `fixed` size holding `precision` decimal digits: the least `n`
/// with `2 * 10^precision <= 256^n` (the two's complement range).
fn decimal_size(precision: u8) -> usize {
    let mut need: u128 = 2;
    for _ in 0..precision {
        need *= 10;
    }
    let (mut size, mut cap) = (1, 256_u128);
    while cap < need {
        cap = cap.saturating_mul(256);
        size += 1;
        if size >= 16 {
            break;
        }
    }
    size.min(16)
}

/// A writer of Avro schema JSON from Arrow fields.
struct Writer {
    namespace: String,
}

impl Writer {
    /// The JSON of `dt` (an Avro type fragment), with named types scoped
    /// under `scope`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an Arrow type with no Avro type.
    fn fragment(&self, dt: &DataType, scope: &str) -> FormatResult<String> {
        Ok(match dt {
            DataType::Null => r#""null""#.to_owned(),
            DataType::Boolean => r#""boolean""#.to_owned(),
            DataType::Int8 | DataType::Int16 | DataType::Int32 => r#""int""#.to_owned(),
            DataType::Int64 => r#""long""#.to_owned(),
            DataType::Float32 => r#""float""#.to_owned(),
            DataType::Float64 => r#""double""#.to_owned(),
            DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8 => r#""string""#.to_owned(),
            DataType::BinaryView | DataType::Binary | DataType::LargeBinary => {
                r#""bytes""#.to_owned()
            }
            DataType::Decimal128(p, s) => {
                let name = avro_name(&format!("{scope}_decimal"));
                format!(
                    r#"{{"type":"fixed","name":{name},"size":{size},"logicalType":"decimal","precision":{p},"scale":{s}}}"#,
                    name = json_string(&name),
                    size = decimal_size(*p),
                )
            }
            DataType::Date32 => r#"{"type":"int","logicalType":"date"}"#.to_owned(),
            DataType::Time32(TimeUnit::Millisecond) => {
                r#"{"type":"int","logicalType":"time-millis"}"#.to_owned()
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                r#"{"type":"long","logicalType":"time-micros"}"#.to_owned()
            }
            DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
                r#"{"type":"long","logicalType":"timestamp-micros"}"#.to_owned()
            }
            DataType::Timestamp(TimeUnit::Millisecond, Some(_)) => {
                r#"{"type":"long","logicalType":"timestamp-millis"}"#.to_owned()
            }
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                r#"{"type":"long","logicalType":"local-timestamp-micros"}"#.to_owned()
            }
            DataType::Timestamp(TimeUnit::Millisecond, None) => {
                r#"{"type":"long","logicalType":"local-timestamp-millis"}"#.to_owned()
            }
            DataType::Struct(fields) => self.record(scope, fields)?,
            DataType::List(item) | DataType::LargeList(item) | DataType::FixedSizeList(item, _) => {
                let inner = self.fragment(item.data_type(), scope)?;
                let items = if item.is_nullable() {
                    format!("[\"null\",{inner}]")
                } else {
                    inner
                };
                format!(r#"{{"type":"array","items":{items}}}"#)
            }
            DataType::Map(entries, _) => {
                let (key, value) = match entries.data_type() {
                    DataType::Struct(fields) => {
                        let key = fields.first().map(|f| (f.name(), f.data_type()));
                        let value = fields.get(1).map(|f| f.as_ref().clone());
                        (key, value)
                    }
                    _ => (None, None),
                };
                let (
                    Some((_, DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8)),
                    Some(value),
                ) = (key, value)
                else {
                    return Err(unsupported(
                        "an Avro map needs string keys (only UTF-8 keys are supported)".to_owned(),
                    ));
                };
                let inner = self.fragment(value.data_type(), scope)?;
                let values = if value.is_nullable() {
                    format!("[\"null\",{inner}]")
                } else {
                    inner
                };
                format!(r#"{{"type":"map","values":{values}}}"#)
            }
            other => {
                return Err(unsupported(format!(
                    "writing the Arrow type {other} as Avro"
                )));
            }
        })
    }

    /// The JSON of a record named `scope` with `fields`.
    ///
    /// # Errors
    /// As [`Self::fragment`].
    fn record(&self, scope: &str, fields: &Fields) -> FormatResult<String> {
        let name = avro_name(scope);
        let mut out = format!(
            r#"{{"type":"record","name":{name}"#,
            name = json_string(&name)
        );
        if !self.namespace.is_empty() {
            out.push_str(&format!(
                r#","namespace":{ns}"#,
                ns = json_string(&self.namespace)
            ));
        }
        out.push_str(r#","fields":["#);
        for (i, f) in fields.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let inner = self.fragment(f.data_type(), &format!("{name}_{}", avro_name(f.name())))?;
            let ty = if f.is_nullable() {
                format!("[\"null\",{inner}]")
            } else {
                inner
            };
            out.push_str(&format!(
                r#"{{"name":{name},"type":{ty}}}"#,
                name = json_string(f.name())
            ));
        }
        out.push_str("]}");
        Ok(out)
    }
}

/// The Avro schema JSON (a record) of the Arrow `schema`, named
/// `record_name` in `record_namespace`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for an Arrow type with no Avro type.
pub(crate) fn schema_json_of_arrow(
    schema: &ArrowSchema,
    record_name: &str,
    record_namespace: &str,
) -> FormatResult<String> {
    Writer {
        namespace: record_namespace.to_owned(),
    }
    .record(record_name, schema.fields())
}

/// The default value of an Avro record field, for schema evolution: missing
/// writer fields read as their default (in reader terms, union-wrapped when
/// the reader is nullable), then null when the reader allows it.
///
/// # Errors
/// [`FormatError::Corrupt`] for a missing field without a usable default or
/// a default the reader schema cannot hold.
pub(crate) fn field_default(
    field_schema: &AvroSchema,
    default: Option<&serde_json::Value>,
) -> FormatResult<AvroValue> {
    let Some(default) = default else {
        if let Some(idx) = null_branch_index(field_schema) {
            return Ok(AvroValue::Union(idx, Box::new(AvroValue::Null)));
        }
        return Err(crate::error::FormatError::Corrupt(
            "the read schema has a field the file lacks with no default".to_owned(),
        ));
    };
    json_default(field_schema, default)
}

/// Whether `json` is the JSON null.
fn is_json_null(json: &serde_json::Value) -> bool {
    matches!(json, serde_json::Value::Null)
}

/// The Avro value of the JSON `default` in `schema` (reader) terms.
///
/// # Errors
/// [`FormatError::Corrupt`] for a default the schema cannot hold.
fn json_default(schema: &AvroSchema, json: &serde_json::Value) -> FormatResult<AvroValue> {
    use serde_json::Value as J;
    if let AvroSchema::Union(u) = schema {
        if u.variants().len() == 2 {
            let null = u
                .variants()
                .iter()
                .position(|s| matches!(s, AvroSchema::Null));
            let inner = u
                .variants()
                .iter()
                .enumerate()
                .find(|(_, s)| !matches!(s, AvroSchema::Null));
            if let (Some(null), Some((branch, inner))) = (null, inner) {
                let index = |i: usize| {
                    u32::try_from(i).map_err(|_| {
                        crate::error::FormatError::Corrupt(
                            "a union with too many branches".to_owned(),
                        )
                    })
                };
                if is_json_null(json) {
                    return Ok(AvroValue::Union(index(null)?, Box::new(AvroValue::Null)));
                }
                return Ok(AvroValue::Union(
                    index(branch)?,
                    Box::new(json_default(inner, json)?),
                ));
            }
        }
        for (i, branch) in u.variants().iter().enumerate() {
            if let Ok(v) = json_default(branch, json) {
                let i = u32::try_from(i).map_err(|_| {
                    crate::error::FormatError::Corrupt("a union with too many branches".to_owned())
                })?;
                return Ok(AvroValue::Union(i, Box::new(v)));
            }
        }
        return Err(crate::error::FormatError::Corrupt(
            "a default no union branch accepts".to_owned(),
        ));
    }
    let corrupt = |what: &str| {
        crate::error::FormatError::Corrupt(format!("a default the Avro schema cannot hold: {what}"))
    };
    Ok(match (schema, json) {
        (_, J::Null) => return Err(corrupt("null for a non-nullable field")),
        (AvroSchema::Null, _) => AvroValue::Null,
        (AvroSchema::Boolean, J::Bool(v)) => AvroValue::Boolean(*v),
        (AvroSchema::Int, J::Number(v)) => AvroValue::Int(
            v.as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| corrupt("an int default"))?,
        ),
        (AvroSchema::Long, J::Number(v)) => {
            AvroValue::Long(v.as_i64().ok_or_else(|| corrupt("a long default"))?)
        }
        (AvroSchema::Float, J::Number(v)) => AvroValue::Float(float_default(
            v.as_f64().ok_or_else(|| corrupt("a float default"))?,
        )),
        (AvroSchema::Double, J::Number(v)) => {
            AvroValue::Double(v.as_f64().ok_or_else(|| corrupt("a double default"))?)
        }
        (AvroSchema::String, J::String(v)) => AvroValue::String(v.clone()),
        (AvroSchema::Bytes, J::String(v)) => {
            AvroValue::Bytes(latin1_bytes(v).ok_or_else(|| corrupt("a bytes default"))?)
        }
        (AvroSchema::Enum(e), J::String(v)) => {
            let index = e
                .symbols
                .iter()
                .position(|s| s == v)
                .and_then(|i| u32::try_from(i).ok())
                .ok_or_else(|| corrupt("an enum default"))?;
            AvroValue::Enum(index, v.clone())
        }
        (AvroSchema::Fixed(f), J::String(v)) => {
            let bytes = latin1_bytes(v)
                .filter(|b| b.len() == f.size)
                .ok_or_else(|| corrupt("a fixed default"))?;
            AvroValue::Fixed(f.size, bytes)
        }
        (AvroSchema::Decimal(_), J::String(v)) => AvroValue::Decimal(apache_avro::Decimal::from(
            latin1_bytes(v).ok_or_else(|| corrupt("a decimal default"))?,
        )),
        (AvroSchema::Uuid(_), J::String(v)) => {
            AvroValue::Uuid(v.parse().map_err(|_| corrupt("a uuid default"))?)
        }
        (AvroSchema::Date, J::Number(v)) => AvroValue::Date(
            v.as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| corrupt("a date default"))?,
        ),
        (AvroSchema::TimeMillis, J::Number(v)) => AvroValue::TimeMillis(
            v.as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| corrupt("a time default"))?,
        ),
        (AvroSchema::TimeMicros, J::Number(v)) => {
            AvroValue::TimeMicros(v.as_i64().ok_or_else(|| corrupt("a time default"))?)
        }
        (AvroSchema::TimestampMillis, J::Number(v)) => {
            AvroValue::TimestampMillis(v.as_i64().ok_or_else(|| corrupt("a timestamp default"))?)
        }
        (AvroSchema::TimestampMicros, J::Number(v)) => {
            AvroValue::TimestampMicros(v.as_i64().ok_or_else(|| corrupt("a timestamp default"))?)
        }
        (AvroSchema::LocalTimestampMillis, J::Number(v)) => AvroValue::LocalTimestampMillis(
            v.as_i64().ok_or_else(|| corrupt("a timestamp default"))?,
        ),
        (AvroSchema::LocalTimestampMicros, J::Number(v)) => AvroValue::LocalTimestampMicros(
            v.as_i64().ok_or_else(|| corrupt("a timestamp default"))?,
        ),
        (AvroSchema::Array(a), J::Array(items)) => AvroValue::Array(
            items
                .iter()
                .map(|v| json_default(&a.items, v))
                .collect::<FormatResult<Vec<_>>>()?,
        ),
        (AvroSchema::Map(m), J::Object(entries)) => AvroValue::Map(
            entries
                .iter()
                .map(|(k, v)| json_default(&m.types, v).map(|v| (k.clone(), v)))
                .collect::<FormatResult<std::collections::HashMap<_, _>>>()?,
        ),
        (AvroSchema::Record(r), J::Object(entries)) => AvroValue::Record(
            r.fields
                .iter()
                .map(|f| {
                    let default = entries.get(&f.name);
                    let v = match default {
                        Some(v) => json_default(&f.schema, v)?,
                        None => json_default(&f.schema, &J::Null)?,
                    };
                    Ok((f.name.clone(), v))
                })
                .collect::<FormatResult<Vec<_>>>()?,
        ),
        _ => return Err(corrupt("a mismatched default")),
    })
}

/// The bytes of an Avro `bytes` or `fixed` default: a JSON string with one
/// code point per byte (U+0000 to U+00FF, ISO-8859-1); `None` for a code
/// point above U+00FF.
fn latin1_bytes(s: &str) -> Option<Vec<u8>> {
    s.chars().map(|c| u8::try_from(u32::from(c)).ok()).collect()
}

/// The index of the `null` branch of a nullable union.
pub(crate) fn null_branch_index(schema: &AvroSchema) -> Option<u32> {
    match schema {
        AvroSchema::Union(u) => u
            .variants()
            .iter()
            .position(|s| matches!(s, AvroSchema::Null))
            .and_then(|i| u32::try_from(i).ok()),
        _ => None,
    }
}

/// A JSON number as an Avro `float` default (narrowing, as Avro reads it).
#[expect(
    clippy::cast_possible_truncation,
    reason = "Avro float defaults are JSON numbers; narrowing is the documented read"
)]
fn float_default(v: f64) -> f32 {
    v as f32
}
