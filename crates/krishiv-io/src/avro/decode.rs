// SPDX-License-Identifier: Apache-2.0
//! The Avro decoder: container blocks into Arrow batches, batch by batch.
//!
//! A file is its header (magic, metadata map, sync marker) then blocks of
//! `count` records, their byte size, the records and the sync marker. The
//! metadata pass scans every block header once (offsets, counts, first
//! rows); the decoder then reads the owned blocks one at a time: it asks
//! for one block's bytes ([`Step::NeedData`]), validates its sync marker,
//! decompresses it, decodes each datum with the file's (writer) schema into
//! an Avro value, resolves the value against the read schema (writer fields
//! matched to reader fields by name, or by position with
//! `positionalFieldMatching`; missing fields read their default, then null)
//! and appends it to the Arrow builders. A batch of [`BATCH_ROWS`] rows is
//! emitted in the file-level request schema, with `_tmp_metadata_row_index`
//! appended when asked.
//!
//! Corrupt records follow `mode`: `FAILFAST` fails the scan, `DROPMALFORMED`
//! drops the rest of the block, `PERMISSIVE` appends one null row (when the
//! output allows it) and drops the rest of the block. A record can never be
//! skipped and resumed mid-block: the framing past a bad datum is
//! unknowable. Dates and timestamps before the calendar switch follow
//! `datetimeRebaseMode` (`EXCEPTION` fails, `CORRECTED` reads them as
//! stored, `LEGACY` is refused when such a value appears).

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use apache_avro::{Schema as AvroSchema, types::Value as AvroValue};
use arrow_array::builder::{
    ArrayBuilder, BinaryViewBuilder, BooleanBuilder, Date32Builder, Decimal128Builder,
    Float32Builder, Float64Builder, Int32Builder, Int64Builder, StringViewBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, RecordBatchOptions, StructArray};
use arrow_buffer::{NullBufferBuilder, OffsetBuffer};
use arrow_schema::{DataType, Field, SchemaRef, TimeUnit};
use bytes::Bytes;

use super::AvroMetadata;
use super::options::{AvroOptions, Codec as OptCodec, Mode, Rebase};
use super::schema::{field_default, schema_kind};
use crate::error::{FormatError, FormatResult, internal, unsupported};
use crate::format::{Decoder, ScanUnitMetrics, Step, UnitPlan};
use crate::request::{ROW_INDEX_COLUMN, ScanRequest, project_batch};

/// Rows per emitted batch, as the text formats'.
pub(crate) const BATCH_ROWS: usize = 4096;

/// The magic of an Avro container file.
const MAGIC: &[u8; 4] = b"Obj\x01";

/// 1582-10-15 as days since the epoch: dates before it predate the
/// Gregorian switch.
const DATE_CUTOFF: i32 = -141427;

/// 1900-01-01T00:00:00Z as milliseconds since the epoch.
const TS_MILLIS_CUTOFF: i64 = -2_208_988_800_000;

/// 1900-01-01T00:00:00Z as microseconds since the epoch.
const TS_MICROS_CUTOFF: i64 = -2_208_988_800_000_000;

/// A parsed header: the metadata entries and the header's length.
type Header = (BTreeMap<Vec<u8>, Vec<u8>>, usize);

/// The outcome of parsing a byte prefix: more bytes, the value, or a bad
/// magic.
pub(crate) enum Parsed<T> {
    Need(usize),
    Done(T),
    BadMagic,
}

/// A zigzag long of `buf` at `pos`: the value and the bytes it took
/// (`None`: truncated or overflowing).
pub(crate) fn parse_long(buf: &[u8], pos: usize) -> Option<(i64, usize)> {
    let mut shift = 0_u32;
    let mut raw = 0_u64;
    for (i, b) in buf[pos..].iter().enumerate().take(10) {
        raw |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            let v = (raw >> 1) as i64 ^ -((raw & 1) as i64);
            return Some((v, i + 1));
        }
        shift += 7;
    }
    None
}

/// A length-delimited slice of `buf` at `pos`.
fn parse_slice(buf: &[u8], pos: usize) -> Option<(&[u8], usize)> {
    let (len, n) = parse_long(buf, pos)?;
    let len = usize::try_from(len).ok()?;
    let end = pos.checked_add(n)?.checked_add(len)?;
    (end <= buf.len()).then(|| (&buf[pos + n..end], n + len))
}

/// The parsed header: the metadata entries and the header's length.
pub(crate) fn parse_header(buf: &[u8]) -> Parsed<Header> {
    if buf.len() < MAGIC.len() {
        return Parsed::Need(MAGIC.len());
    }
    if &buf[..MAGIC.len()] != MAGIC {
        // Not a header at all; the caller reports the magic.
        return Parsed::BadMagic;
    }
    let mut pos = MAGIC.len();
    let mut meta = BTreeMap::new();
    loop {
        let Some((count, n)) = parse_long(buf, pos) else {
            return Parsed::Need(pos + 1);
        };
        pos += n;
        if count == 0 {
            break;
        }
        // A negative count is followed by the block's byte size, which a
        // header walk skips (the entries follow it).
        if count < 0 {
            let Some((_, n)) = parse_long(buf, pos) else {
                return Parsed::Need(pos + 1);
            };
            pos += n;
        }
        let Ok(entries) = usize::try_from(count.unsigned_abs()) else {
            return Parsed::BadMagic;
        };
        if entries > 1_000_000 {
            return Parsed::BadMagic;
        }
        for _ in 0..entries {
            let Some((k, n)) = parse_slice(buf, pos) else {
                return Parsed::Need(pos + 1);
            };
            let Some((v, m)) = parse_slice(buf, pos + n) else {
                return Parsed::Need(pos + n + 1);
            };
            pos += n + m;
            meta.insert(k.to_vec(), v.to_vec());
        }
    }
    if buf.len() < pos + 16 {
        return Parsed::Need(pos + 16);
    }
    Parsed::Done((meta, pos))
}

/// A corrupt-data error.
fn corrupt(reason: impl Into<String>) -> FormatError {
    FormatError::Corrupt(reason.into())
}

/// The nullable union of `schema`, if it is one: the null branch index and
/// the single non-null branch.
fn nullable_parts(schema: &AvroSchema) -> Option<(u32, &AvroSchema)> {
    match schema {
        AvroSchema::Union(u) if u.variants().len() == 2 => {
            let null = u
                .variants()
                .iter()
                .position(|s| matches!(s, AvroSchema::Null))?;
            let inner = u
                .variants()
                .iter()
                .find(|s| !matches!(s, AvroSchema::Null))?;
            Some((u32::try_from(null).ok()?, inner))
        }
        _ => None,
    }
}

/// Resolves `value` (in `writer` terms) against the `reader` schema: null
/// unions are stripped on either side, missing record fields read their
/// default (then null), promotions follow the Avro specification
/// (`int` → `long`/`float`/`double`, `long` → `float`/`double`,
/// `float` → `double`, `bytes` ↔ `string`, `enum` ↔ `string`).
///
/// # Errors
/// [`FormatError::Corrupt`] for a value the reader schema cannot hold.
fn resolve(
    writer: &AvroSchema,
    reader: &AvroSchema,
    value: AvroValue,
    positional: bool,
) -> FormatResult<AvroValue> {
    match (nullable_parts(writer), nullable_parts(reader)) {
        (Some((_, w)), Some((r_null, r))) => {
            let inner = match value {
                AvroValue::Union(_, inner) => *inner,
                AvroValue::Null => AvroValue::Null,
                other => other,
            };
            if matches!(inner, AvroValue::Null) {
                Ok(AvroValue::Union(r_null, Box::new(AvroValue::Null)))
            } else {
                Ok(AvroValue::Union(
                    r_null,
                    Box::new(resolve(w, r, inner, positional)?),
                ))
            }
        }
        (Some((_, w)), None) => {
            let inner = match value {
                AvroValue::Union(_, inner) => *inner,
                AvroValue::Null => {
                    return Err(corrupt("a null value for a non-nullable field"));
                }
                other => other,
            };
            if matches!(inner, AvroValue::Null) {
                return Err(corrupt("a null value for a non-nullable field"));
            }
            resolve(w, reader, inner, positional)
        }
        (None, Some((r_null, r))) => Ok(AvroValue::Union(
            r_null,
            Box::new(resolve(writer, r, value, positional)?),
        )),
        (None, None) => resolve_inner(writer, reader, value, positional),
    }
}

/// Resolves `value` with neither side a nullable union.
///
/// # Errors
/// [`FormatError::Corrupt`] for a value the reader schema cannot hold.
fn resolve_inner(
    writer: &AvroSchema,
    reader: &AvroSchema,
    value: AvroValue,
    positional: bool,
) -> FormatResult<AvroValue> {
    // A writer union selects its branch by index; a reader union takes the
    // first branch the value resolves against.
    if let AvroSchema::Union(u) = writer {
        let (index, inner) = match value {
            AvroValue::Union(i, inner) => (i, *inner),
            _ => return Err(corrupt("a union value without a branch index")),
        };
        let branch = u
            .variants()
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .ok_or_else(|| corrupt("a union value with an out-of-range branch index"))?;
        return match reader {
            AvroSchema::Union(r) => {
                for (j, candidate) in r.variants().iter().enumerate() {
                    if let Ok(resolved) = resolve(branch, candidate, inner.clone(), positional) {
                        let j = u32::try_from(j).map_err(|_| {
                            corrupt("a union value with an out-of-range branch index")
                        })?;
                        return Ok(AvroValue::Union(j, Box::new(resolved)));
                    }
                }
                Err(corrupt("a union value no reader branch accepts"))
            }
            _ => resolve(branch, reader, inner, positional),
        };
    }
    if let AvroSchema::Union(r) = reader {
        for (j, candidate) in r.variants().iter().enumerate() {
            if let Ok(resolved) = resolve(writer, candidate, value.clone(), positional) {
                let j = u32::try_from(j)
                    .map_err(|_| corrupt("a union value with too many branches"))?;
                return Ok(AvroValue::Union(j, Box::new(resolved)));
            }
        }
        return Err(corrupt("a value no reader union branch accepts"));
    }
    resolve_scalar(writer, reader, value, positional)
}

/// The effective string or bytes kind of a schema, treating `uuid` as its
/// underlying type.
fn string_or_bytes(schema: &AvroSchema) -> Option<(&'static str, Option<usize>)> {
    match schema {
        AvroSchema::String => Some(("string", None)),
        AvroSchema::Bytes => Some(("bytes", None)),
        AvroSchema::Fixed(f) => Some(("fixed", Some(f.size))),
        AvroSchema::Uuid(u) => match u {
            apache_avro::schema::UuidSchema::String => Some(("string", None)),
            apache_avro::schema::UuidSchema::Bytes => Some(("bytes", None)),
            apache_avro::schema::UuidSchema::Fixed(f) => Some(("fixed", Some(f.size))),
        },
        _ => None,
    }
}

/// Resolves a scalar, record, enum, array or map `value` with neither side
/// a union.
///
/// # Errors
/// [`FormatError::Corrupt`] for a value the reader schema cannot hold.
fn resolve_scalar(
    writer: &AvroSchema,
    reader: &AvroSchema,
    value: AvroValue,
    positional: bool,
) -> FormatResult<AvroValue> {
    // Records resolve field by field.
    if let (AvroSchema::Record(w), AvroSchema::Record(r)) = (writer, reader) {
        let AvroValue::Record(pairs) = value else {
            return Err(corrupt("a record value that is not a record"));
        };
        let mut resolved = Vec::with_capacity(r.fields.len());
        for (i, rf) in r.fields.iter().enumerate() {
            let pair: Option<(&AvroSchema, &AvroValue)> = if positional {
                match (w.fields.get(i), pairs.get(i)) {
                    (Some(wf), Some((_, v))) => Some((&wf.schema, v)),
                    _ => None,
                }
            } else {
                match w.fields.iter().find(|f| f.name == rf.name) {
                    Some(wf) => pairs
                        .iter()
                        .find(|(name, _)| name == &rf.name)
                        .map(|(_, v)| (&wf.schema, v)),
                    None => None,
                }
            };
            let v = match pair {
                Some((ws, v)) => resolve(ws, &rf.schema, v.clone(), positional)?,
                None => field_default(&rf.schema, rf.default.as_ref())?,
            };
            resolved.push((rf.name.clone(), v));
        }
        return Ok(AvroValue::Record(resolved));
    }
    // Arrays and maps resolve elementwise.
    if let (AvroSchema::Array(w), AvroSchema::Array(r)) = (writer, reader) {
        let AvroValue::Array(items) = value else {
            return Err(corrupt("an array value that is not an array"));
        };
        return items
            .into_iter()
            .map(|v| resolve(&w.items, &r.items, v, positional))
            .collect::<FormatResult<Vec<_>>>()
            .map(AvroValue::Array);
    }
    if let (AvroSchema::Map(w), AvroSchema::Map(r)) = (writer, reader) {
        let AvroValue::Map(entries) = value else {
            return Err(corrupt("a map value that is not a map"));
        };
        return entries
            .into_iter()
            .map(|(k, v)| resolve(&w.types, &r.types, v, positional).map(|v| (k, v)))
            .collect::<FormatResult<std::collections::HashMap<_, _>>>()
            .map(AvroValue::Map);
    }
    // Enums resolve by symbol, to an enum or a string.
    if let AvroSchema::Enum(w) = writer {
        let symbol = match &value {
            AvroValue::Enum(_, symbol) => symbol.clone(),
            _ => return Err(corrupt("an enum value that is not an enum")),
        };
        if !w.symbols.contains(&symbol) {
            return Err(corrupt("an enum value outside the writer symbols"));
        }
        return match reader {
            AvroSchema::Enum(r) => {
                let index = r
                    .symbols
                    .iter()
                    .position(|s| s == &symbol)
                    .and_then(|i| u32::try_from(i).ok())
                    .ok_or_else(|| corrupt("an enum symbol the read schema lacks"))?;
                Ok(AvroValue::Enum(index, symbol))
            }
            AvroSchema::String => Ok(AvroValue::String(symbol)),
            _ => Err(corrupt(format!(
                "an enum value resolved as {}",
                schema_kind(reader)
            ))),
        };
    }
    if let AvroSchema::Enum(r) = reader {
        let symbol = match &value {
            AvroValue::String(s) => s.clone(),
            _ => {
                return Err(corrupt(format!(
                    "a {} value resolved as an enum",
                    schema_kind(writer)
                )));
            }
        };
        let index = r
            .symbols
            .iter()
            .position(|s| s == &symbol)
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| corrupt("a string the read enum lacks"))?;
        return Ok(AvroValue::Enum(index, symbol));
    }
    // Fixed sizes must match; decimals their precision and scale.
    if let (AvroSchema::Fixed(w), AvroSchema::Fixed(r)) = (writer, reader) {
        if w.size != r.size {
            return Err(corrupt("a fixed value of another size"));
        }
        return Ok(value);
    }
    if let (AvroSchema::Decimal(w), AvroSchema::Decimal(r)) = (writer, reader) {
        if w.precision != r.precision || w.scale != r.scale {
            return Err(corrupt("a decimal value of another precision or scale"));
        }
        return Ok(value);
    }
    // Strings and bytes convert either way; fixed only to itself.
    if let (Some((wk, ws)), Some((rk, rs))) = (string_or_bytes(writer), string_or_bytes(reader)) {
        if wk == rk && ws == rs {
            return Ok(value);
        }
        if wk == "fixed" || rk == "fixed" {
            return Err(corrupt("a fixed value resolved as another type"));
        }
        return match (value, rk) {
            (AvroValue::String(s), "bytes") => Ok(AvroValue::Bytes(s.into_bytes())),
            (AvroValue::Bytes(b), "string") => {
                Ok(AvroValue::String(String::from_utf8(b).map_err(|_| {
                    FormatError::InvalidUtf8("an Avro string".to_owned())
                })?))
            }
            (..) => Err(corrupt(format!("a {wk} value resolved as {rk}",))),
        };
    }
    // Numeric promotions resolve to reader terms: after this, every value
    // matches its reader schema exactly (append never converts).
    match (writer, &value) {
        (AvroSchema::Int, AvroValue::Int(v)) => {
            let v = *v;
            match reader {
                AvroSchema::Long => return Ok(AvroValue::Long(i64::from(v))),
                AvroSchema::Float => return Ok(AvroValue::Float(v as f32)),
                AvroSchema::Double => return Ok(AvroValue::Double(f64::from(v))),
                _ => {}
            }
        }
        (AvroSchema::Long, AvroValue::Long(v)) => {
            let v = *v;
            match reader {
                AvroSchema::Float => return Ok(AvroValue::Float(v as f32)),
                AvroSchema::Double => return Ok(AvroValue::Double(v as f64)),
                _ => {}
            }
        }
        (AvroSchema::Float, AvroValue::Float(v)) => {
            if matches!(reader, AvroSchema::Double) {
                return Ok(AvroValue::Double(f64::from(*v)));
            }
        }
        _ => {}
    }
    let promotion = matches!(
        (writer, reader),
        (
            AvroSchema::Int,
            AvroSchema::Long | AvroSchema::Float | AvroSchema::Double
        ) | (AvroSchema::Long, AvroSchema::Float | AvroSchema::Double)
            | (AvroSchema::Float, AvroSchema::Double)
    );
    if promotion || std::mem::discriminant(writer) == std::mem::discriminant(reader) {
        return Ok(value);
    }
    Err(corrupt(format!(
        "a {} value resolved as {}",
        schema_kind(writer),
        schema_kind(reader)
    )))
}

/// Checks a date or timestamp against `mode`: values before the calendar
/// switch fail in `EXCEPTION`, are refused in `LEGACY` and pass through in
/// `CORRECTED`.
///
/// # Errors
/// [`FormatError::Corrupt`] in `EXCEPTION`; the unsupported error in
/// `LEGACY`.
fn check_rebase(kind: &str, ancient: bool, mode: Rebase) -> FormatResult<()> {
    if !ancient {
        return Ok(());
    }
    match mode {
        Rebase::Corrected => Ok(()),
        Rebase::Exception => Err(corrupt(format!(
            "a {kind} value before 1582-10-15 with datetimeRebaseMode EXCEPTION"
        ))),
        Rebase::Legacy => Err(FormatError::Other(unsupported(format!(
            "the LEGACY calendar for a {kind} value before 1582-10-15"
        )))),
    }
}

/// One output column's builders: leaves hold Arrow builders, containers
/// hold their children's builders with offsets and validity. Containers
/// keep the reader schema of their variable parts, so values append
/// against the right types.
#[derive(Debug)]
enum Node {
    Null {
        len: usize,
    },
    Bool(BooleanBuilder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    String(StringViewBuilder),
    Binary(BinaryViewBuilder),
    Decimal {
        builder: Decimal128Builder,
        precision: u8,
    },
    Date(Date32Builder),
    Timestamp(TimestampMicrosecondBuilder),
    Struct {
        schemas: Vec<AvroSchema>,
        children: Vec<Node>,
        nulls: NullBufferBuilder,
        len: usize,
    },
    List {
        item: Arc<Field>,
        item_schema: AvroSchema,
        values: Box<Node>,
        offsets: Vec<i32>,
        nulls: NullBufferBuilder,
    },
    Map {
        entries: Arc<Field>,
        value_schema: AvroSchema,
        keys: Box<Node>,
        values: Box<Node>,
        offsets: Vec<i32>,
        nulls: NullBufferBuilder,
    },
}

/// Fresh builders for `field` holding `schema` values.
fn node_of(field: &Field, schema: &AvroSchema) -> FormatResult<Node> {
    // Nullable unions hold their single branch; the builders are the
    // branch's (nulls append as nulls).
    if let Some((_, inner)) = nullable_parts(schema) {
        return node_of(field, inner);
    }
    Ok(match field.data_type() {
        DataType::Null => Node::Null { len: 0 },
        DataType::Boolean => Node::Bool(BooleanBuilder::new()),
        DataType::Int8 | DataType::Int16 | DataType::Int32 => Node::Int32(Int32Builder::new()),
        DataType::Int64 => Node::Int64(Int64Builder::new()),
        DataType::Float32 => Node::Float32(Float32Builder::new()),
        DataType::Float64 => Node::Float64(Float64Builder::new()),
        DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8 => {
            Node::String(StringViewBuilder::new())
        }
        DataType::BinaryView | DataType::Binary | DataType::LargeBinary => {
            Node::Binary(BinaryViewBuilder::new())
        }
        DataType::Decimal128(p, s) => {
            let builder = Decimal128Builder::new()
                .with_precision_and_scale(*p, *s)
                .map_err(|e| corrupt(format!("an invalid decimal type: {e}")))?;
            Node::Decimal {
                builder,
                precision: *p,
            }
        }
        DataType::Date32 => Node::Date(Date32Builder::new()),
        DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => {
            Node::Timestamp(TimestampMicrosecondBuilder::new().with_timezone(tz.as_ref()))
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            Node::Timestamp(TimestampMicrosecondBuilder::new())
        }
        DataType::Struct(fields) => {
            let schemas = match schema {
                AvroSchema::Record(r) => r.fields.iter().map(|f| f.schema.clone()).collect(),
                AvroSchema::Union(u) => {
                    let members: Vec<AvroSchema> = u.variants().to_vec();
                    if members.len() != fields.len() {
                        return Err(corrupt("a struct column of another shape"));
                    }
                    members
                }
                _ => return Err(corrupt("a struct column of another shape")),
            };
            Node::Struct {
                children: fields
                    .iter()
                    .zip(schemas.iter())
                    .map(|(f, s)| node_of(f, s))
                    .collect::<FormatResult<Vec<_>>>()?,
                schemas,
                nulls: NullBufferBuilder::new(0),
                len: 0,
            }
        }
        DataType::List(item) | DataType::LargeList(item) | DataType::FixedSizeList(item, _) => {
            let item_schema = match schema {
                AvroSchema::Array(a) => (*a.items).clone(),
                AvroSchema::Union(u) => {
                    let members: Vec<AvroSchema> = u.variants().to_vec();
                    if members.len() != 1 {
                        return Err(corrupt("a list column of another shape"));
                    }
                    members.into_iter().next().unwrap_or(AvroSchema::Null)
                }
                _ => return Err(corrupt("a list column of another shape")),
            };
            Node::List {
                values: Box::new(node_of(item, &item_schema)?),
                item: Arc::clone(item),
                item_schema,
                offsets: vec![0],
                nulls: NullBufferBuilder::new(0),
            }
        }
        DataType::Map(entries, _) => {
            let (key_field, value_field) = match entries.data_type() {
                DataType::Struct(fields) => (fields.first().cloned(), fields.get(1).cloned()),
                _ => (None, None),
            };
            let (Some(key_field), Some(value_field)) = (key_field, value_field) else {
                return Err(corrupt("a map column without key and value fields"));
            };
            let value_schema = match schema {
                AvroSchema::Map(m) => (*m.types).clone(),
                AvroSchema::Union(u) => {
                    let members: Vec<AvroSchema> = u.variants().to_vec();
                    if members.len() != 1 {
                        return Err(corrupt("a map column of another shape"));
                    }
                    members.into_iter().next().unwrap_or(AvroSchema::Null)
                }
                _ => return Err(corrupt("a map column of another shape")),
            };
            Node::Map {
                keys: Box::new(node_of(&key_field, &AvroSchema::String)?),
                values: Box::new(node_of(&value_field, &value_schema)?),
                entries: Arc::clone(entries),
                value_schema,
                offsets: vec![0],
                nulls: NullBufferBuilder::new(0),
            }
        }
        other => {
            return Err(corrupt(format!(
                "a column of the Arrow type {other} (not an Avro type)"
            )));
        }
    })
}

/// The unscaled `i128` of decimal bytes (big-endian two's complement).
///
/// # Errors
/// [`FormatError::Corrupt`] for more than 16 bytes.
fn decimal_i128(bytes: &[u8]) -> FormatResult<i128> {
    if bytes.len() > 16 {
        return Err(corrupt("a decimal value of more than 16 bytes"));
    }
    let mut buf = if bytes.first().is_some_and(|b| b & 0x80 != 0) {
        [0xFF; 16]
    } else {
        [0x00; 16]
    };
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    Ok(i128::from_be_bytes(buf))
}

/// Appends a null to `node`.
///
/// # Errors
/// [`FormatError::Corrupt`] for an inconsistent node.
fn append_null(node: &mut Node) -> FormatResult<()> {
    match node {
        Node::Null { len } => *len += 1,
        Node::Bool(b) => b.append_null(),
        Node::Int32(b) => b.append_null(),
        Node::Int64(b) => b.append_null(),
        Node::Float32(b) => b.append_null(),
        Node::Float64(b) => b.append_null(),
        Node::String(b) => b.append_null(),
        Node::Binary(b) => b.append_null(),
        Node::Decimal { builder, .. } => builder.append_null(),
        Node::Date(b) => b.append_null(),
        Node::Timestamp(b) => b.append_null(),
        Node::Struct {
            children,
            nulls,
            len,
            ..
        } => {
            for c in children {
                append_null(c)?;
            }
            nulls.append_null();
            *len += 1;
        }
        Node::List { offsets, nulls, .. } | Node::Map { offsets, nulls, .. } => {
            offsets.push(*offsets.last().unwrap_or(&0));
            nulls.append_null();
        }
    }
    Ok(())
}

/// Appends `value` (in `reader` terms) to `node`.
///
/// # Errors
/// [`FormatError::Corrupt`] for a value the column cannot hold (overflow,
/// a short fixed value, an out-of-range decimal) and
/// [`FormatError::InvalidUtf8`] for a string that is not UTF-8.
#[expect(clippy::too_many_lines, reason = "one arm per leaf type")]
fn append(node: &mut Node, reader: &AvroSchema, value: &AvroValue) -> FormatResult<()> {
    if matches!(value, AvroValue::Null) {
        return append_null(node);
    }
    // A union into a struct sets the selected member and nulls the rest
    // (Spark's `member0..N` mapping). Nullable unions never reach this:
    // they strip to their branch above.
    if let (
        Node::Struct {
            schemas,
            children,
            nulls,
            len,
        },
        AvroValue::Union(i, inner),
    ) = (&mut *node, value)
        && matches!(reader, AvroSchema::Union(_))
        && nullable_parts(reader).is_none()
    {
        let i = usize::try_from(*i).unwrap_or(usize::MAX);
        if i >= children.len() {
            return Err(corrupt("a union value with an out-of-range branch index"));
        }
        for (j, (child, s)) in children.iter_mut().zip(schemas.iter()).enumerate() {
            if j == i {
                append(child, s, inner)?;
            } else {
                append_null(child)?;
            }
        }
        nulls.append_non_null();
        *len += 1;
        return Ok(());
    }
    if let AvroValue::Union(i, inner) = value {
        if matches!(**inner, AvroValue::Null) {
            return append_null(node);
        }
        let inner_reader = match reader {
            AvroSchema::Union(u) => u
                .variants()
                .get(usize::try_from(*i).unwrap_or(usize::MAX))
                .ok_or_else(|| corrupt("a union value with an out-of-range branch index"))?,
            _ => return Err(corrupt("a union value for a non-union column")),
        };
        return append(node, inner_reader, inner);
    }
    // A nullable union's branch was stripped above; a bare union here wraps
    // the value's own branch.
    if let AvroSchema::Union(u) = reader {
        let branches: Vec<&AvroSchema> = u.variants().iter().collect();
        if branches.len() == 1 {
            return append(node, branches[0], value);
        }
    }
    match (node, value) {
        (Node::Bool(b), AvroValue::Boolean(v)) => b.append_value(*v),
        (Node::Int32(b), AvroValue::Int(v) | AvroValue::Date(v) | AvroValue::TimeMillis(v)) => {
            b.append_value(*v);
        }
        (Node::Int64(b), AvroValue::Long(v) | AvroValue::TimeMicros(v)) => b.append_value(*v),
        (Node::Float32(b), AvroValue::Float(v)) => b.append_value(*v),
        (Node::Float64(b), AvroValue::Float(v)) => b.append_value(f64::from(*v)),
        (Node::Float64(b), AvroValue::Double(v)) => b.append_value(*v),
        (Node::String(b), AvroValue::String(v) | AvroValue::Enum(_, v)) => b.append_value(v),
        (Node::String(b), AvroValue::Bytes(v)) => b.append_value(
            std::str::from_utf8(v)
                .map_err(|_| FormatError::InvalidUtf8("an Avro string".to_owned()))?,
        ),
        (Node::String(b), AvroValue::Uuid(v)) => b.append_value(v.to_string()),
        (Node::Binary(b), AvroValue::Bytes(v) | AvroValue::Fixed(_, v)) => b.append_value(v),
        (Node::Binary(b), AvroValue::String(v)) => b.append_value(v.as_bytes()),
        (Node::Binary(b), AvroValue::Uuid(v)) => b.append_value(v.as_bytes().as_slice()),
        (Node::Decimal { builder, precision }, AvroValue::Decimal(v)) => {
            let bytes = Vec::<u8>::try_from(v)
                .map_err(|e| corrupt(format!("an undecodable decimal: {e}")))?;
            let unscaled = decimal_i128(&bytes)?;
            let max = 10_i128.pow(u32::from(*precision));
            if unscaled.abs() >= max {
                return Err(corrupt("a decimal value outside its precision"));
            }
            builder.append_value(unscaled);
        }
        (Node::Date(b), AvroValue::Date(v)) => b.append_value(*v),
        (
            Node::Timestamp(b),
            AvroValue::TimestampMillis(v) | AvroValue::LocalTimestampMillis(v),
        ) => {
            b.append_value(
                v.checked_mul(1000)
                    .ok_or_else(|| corrupt("a timestamp out of range"))?,
            );
        }
        (
            Node::Timestamp(b),
            AvroValue::TimestampMicros(v) | AvroValue::LocalTimestampMicros(v),
        ) => b.append_value(*v),
        (
            Node::Struct {
                schemas,
                children,
                nulls,
                len,
            },
            AvroValue::Record(pairs),
        ) => {
            if pairs.len() != children.len() {
                return Err(corrupt("a struct value of another shape"));
            }
            for ((_, v), (child, s)) in pairs.iter().zip(children.iter_mut().zip(schemas.iter())) {
                append(child, s, v)?;
            }
            nulls.append_non_null();
            *len += 1;
        }
        (
            Node::List {
                item_schema,
                values,
                offsets,
                nulls,
                ..
            },
            AvroValue::Array(items),
        ) => {
            for v in items {
                append(values, item_schema, v)?;
            }
            let end = i32::try_from(node_len(values))
                .map_err(|_| corrupt("a list value with too many elements"))?;
            offsets.push(end);
            nulls.append_non_null();
        }
        (
            Node::Map {
                value_schema,
                keys,
                values,
                offsets,
                nulls,
                ..
            },
            AvroValue::Map(entries),
        ) => {
            // Sorted by key, so the output does not depend on the hash map.
            let mut entries: Vec<(&String, &AvroValue)> = entries.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (k, v) in entries {
                append(keys, &AvroSchema::String, &AvroValue::String(k.clone()))?;
                append(values, value_schema, v)?;
            }
            let end = i32::try_from(node_len(values))
                .map_err(|_| corrupt("a map value with too many entries"))?;
            offsets.push(end);
            nulls.append_non_null();
        }
        _ => {
            return Err(corrupt(format!(
                "a {} value for an Avro {} column",
                value_kind(value),
                schema_kind(reader)
            )));
        }
    }
    Ok(())
}

/// Rows buffered in `node`.
fn node_len(node: &Node) -> usize {
    match node {
        Node::Null { len } | Node::Struct { len, .. } => *len,
        Node::Bool(b) => b.len(),
        Node::Int32(b) => b.len(),
        Node::Int64(b) => b.len(),
        Node::Float32(b) => b.len(),
        Node::Float64(b) => b.len(),
        Node::String(b) => b.len(),
        Node::Binary(b) => b.len(),
        Node::Decimal { builder, .. } => builder.len(),
        Node::Date(b) => b.len(),
        Node::Timestamp(b) => b.len(),
        Node::List { offsets, .. } | Node::Map { offsets, .. } => offsets.len().saturating_sub(1),
    }
}

/// A short kind name of a value, for messages.
fn value_kind(value: &AvroValue) -> &'static str {
    match value {
        AvroValue::Null => "null",
        AvroValue::Boolean(_) => "boolean",
        AvroValue::Int(_) | AvroValue::Date(_) | AvroValue::TimeMillis(_) => "int",
        AvroValue::Long(_) | AvroValue::TimeMicros(_) => "long",
        AvroValue::Float(_) => "float",
        AvroValue::Double(_) => "double",
        AvroValue::Bytes(_) => "bytes",
        AvroValue::String(_) => "string",
        AvroValue::Fixed(_, _) => "fixed",
        AvroValue::Enum(_, _) => "enum",
        AvroValue::Union(_, _) => "union",
        AvroValue::Array(_) => "array",
        AvroValue::Map(_) => "map",
        AvroValue::Record(_) => "record",
        AvroValue::Decimal(_) => "decimal",
        AvroValue::BigDecimal(_) => "bigdecimal",
        AvroValue::TimestampMillis(_) => "timestamp-millis",
        AvroValue::TimestampMicros(_) => "timestamp-micros",
        AvroValue::TimestampNanos(_) => "timestamp-nanos",
        AvroValue::LocalTimestampMillis(_) => "local-timestamp-millis",
        AvroValue::LocalTimestampMicros(_) => "local-timestamp-micros",
        AvroValue::LocalTimestampNanos(_) => "local-timestamp-nanos",
        AvroValue::Duration(_) => "duration",
        AvroValue::Uuid(_) => "uuid",
    }
}

/// Finishes `node` into its array (`field` names the nested parts).
///
/// # Errors
/// `INTERNAL_ERROR` for an inconsistent builder (a broken invariant, not a
/// data error: every value was validated on append).
fn finish_node(node: Node, field: &Field) -> FormatResult<ArrayRef> {
    Ok(match node {
        Node::Null { len } => Arc::new(arrow_array::NullArray::new(len)),
        Node::Bool(mut b) => Arc::new(b.finish()),
        Node::Int32(mut b) => Arc::new(b.finish()),
        Node::Int64(mut b) => Arc::new(b.finish()),
        Node::Float32(mut b) => Arc::new(b.finish()),
        Node::Float64(mut b) => Arc::new(b.finish()),
        Node::String(mut b) => Arc::new(b.finish()),
        Node::Binary(mut b) => Arc::new(b.finish()),
        Node::Decimal { mut builder, .. } => Arc::new(builder.finish()),
        Node::Date(mut b) => Arc::new(b.finish()),
        Node::Timestamp(mut b) => Arc::new(b.finish()),
        Node::Struct {
            children,
            mut nulls,
            ..
        } => {
            let DataType::Struct(fields) = field.data_type() else {
                return Err(FormatError::Other(internal(
                    "a struct builder for a non-struct field".to_owned(),
                )));
            };
            let arrays = children
                .into_iter()
                .zip(fields.iter())
                .map(|(c, f)| finish_node(c, f))
                .collect::<FormatResult<Vec<_>>>()?;
            Arc::new(StructArray::new(fields.clone(), arrays, nulls.finish()))
        }
        Node::List {
            item,
            values,
            offsets,
            mut nulls,
            ..
        } => {
            let values = finish_node(*values, &item)?;
            let offsets = OffsetBuffer::new(offsets.into());
            Arc::new(arrow_array::ListArray::new(
                Arc::clone(&item),
                offsets,
                values,
                nulls.finish(),
            ))
        }
        Node::Map {
            entries,
            keys,
            values,
            offsets,
            mut nulls,
            ..
        } => {
            let DataType::Struct(entry_fields) = entries.data_type() else {
                return Err(FormatError::Other(internal(
                    "a map builder for a non-map field".to_owned(),
                )));
            };
            let (Some(key_field), Some(value_field)) = (entry_fields.first(), entry_fields.get(1))
            else {
                return Err(FormatError::Other(internal(
                    "a map builder without key and value fields".to_owned(),
                )));
            };
            let key_array = finish_node(*keys, key_field)?;
            let value_array = finish_node(*values, value_field)?;
            let entries_array =
                StructArray::new(entry_fields.clone(), vec![key_array, value_array], None);
            let offsets = OffsetBuffer::new(offsets.into());
            Arc::new(arrow_array::MapArray::new(
                Arc::clone(&entries),
                offsets,
                entries_array,
                nulls.finish(),
                false,
            ))
        }
    })
}

/// Checks every date and timestamp of `value` (in `reader` terms) against
/// `mode`. Unlike corrupt data this always fails: a value needing a rebase
/// is rejected even in `PERMISSIVE` mode.
///
/// # Errors
/// As [`check_rebase`].
fn check_rebase_value(reader: &AvroSchema, value: &AvroValue, rebase: Rebase) -> FormatResult<()> {
    match (reader, value) {
        (AvroSchema::Date, AvroValue::Date(v)) => check_rebase("date", *v < DATE_CUTOFF, rebase),
        (AvroSchema::TimestampMillis, AvroValue::TimestampMillis(v))
        | (AvroSchema::LocalTimestampMillis, AvroValue::LocalTimestampMillis(v)) => {
            check_rebase("timestamp", *v < TS_MILLIS_CUTOFF, rebase)
        }
        (AvroSchema::TimestampMicros, AvroValue::TimestampMicros(v))
        | (AvroSchema::LocalTimestampMicros, AvroValue::LocalTimestampMicros(v)) => {
            check_rebase("timestamp", *v < TS_MICROS_CUTOFF, rebase)
        }
        (AvroSchema::Union(u), AvroValue::Union(i, inner)) => {
            let branch = u
                .variants()
                .get(usize::try_from(*i).unwrap_or(usize::MAX))
                .ok_or_else(|| corrupt("a union value with an out-of-range branch index"))?;
            check_rebase_value(branch, inner, rebase)
        }
        (AvroSchema::Record(r), AvroValue::Record(pairs)) => {
            for ((_, v), f) in pairs.iter().zip(r.fields.iter()) {
                check_rebase_value(&f.schema, v, rebase)?;
            }
            Ok(())
        }
        (AvroSchema::Array(a), AvroValue::Array(items)) => {
            for v in items {
                check_rebase_value(&a.items, v, rebase)?;
            }
            Ok(())
        }
        (AvroSchema::Map(m), AvroValue::Map(entries)) => {
            for v in entries.values() {
                check_rebase_value(&m.types, v, rebase)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// A block being decoded: its decompressed bytes, the read position, the
/// records left and the file row of the next record.
#[derive(Debug)]
struct OpenBlock {
    bytes: Vec<u8>,
    pos: usize,
    left: u64,
    file_row: u64,
}

/// The sans-IO Avro decoder: one owned block at a time into Arrow builders
/// (see the module docs).
#[derive(Debug)]
pub(crate) struct AvroDecoder {
    meta: Arc<AvroMetadata>,
    blocks: Vec<usize>,
    next: usize,
    req: ScanRequest,
    out_schema: SchemaRef,
    columns: Vec<(Field, AvroSchema)>,
    o: AvroOptions,
    pending: BTreeMap<u64, Bytes>,
    pending_bytes: usize,
    open: Option<OpenBlock>,
    nodes: Vec<Node>,
    file_rows: Vec<u64>,
    batch_rows: usize,
    row_width: usize,
    all_nullable: bool,
    metrics: ScanUnitMetrics,
    done: bool,
}

/// A rough width of one row of `schema`, for the working-set estimate.
fn row_width(schema: &SchemaRef) -> usize {
    fn width(dt: &DataType) -> usize {
        match dt {
            DataType::Null => 0,
            DataType::Boolean => 1,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Float32
            | DataType::Date32
            | DataType::Date64 => 4,
            DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Dictionary(_, _)
            | DataType::RunEndEncoded(_, _) => 8,
            DataType::Float16 => 2,
            DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Duration(_)
            | DataType::Interval(_) => 16,
            DataType::Utf8View | DataType::BinaryView => 48,
            DataType::Utf8 | DataType::Binary => 32,
            DataType::LargeUtf8
            | DataType::LargeBinary
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::Map(_, _)
            | DataType::Union(_, _) => 64,
            DataType::FixedSizeBinary(n) => usize::try_from(*n).unwrap_or(usize::MAX),
            DataType::Struct(fields) => fields.iter().map(|f| width(f.data_type())).sum(),
        }
    }
    schema.fields().iter().map(|f| width(f.data_type())).sum()
}

impl AvroDecoder {
    /// A decoder of `units` (block indexes of `meta`) into the file-level
    /// `req` schema.
    ///
    /// # Errors
    /// [`FormatError::Corrupt`] if the output cannot be built.
    pub(crate) fn new(
        meta: &Arc<AvroMetadata>,
        units: &UnitPlan,
        req: &ScanRequest,
        o: &AvroOptions,
    ) -> FormatResult<Self> {
        let out_schema = req.output_schema().map_err(FormatError::Other)?;
        let AvroSchema::Record(reader) = &meta.reader else {
            return Err(corrupt("a read schema that is not a record"));
        };
        let columns = req
            .table_schema
            .fields()
            .iter()
            .map(|f| {
                let s = reader
                    .fields
                    .iter()
                    .find(|rf| rf.name == *f.name())
                    .map(|rf| rf.schema.clone())
                    .ok_or_else(|| corrupt("a file column the read schema lacks"))?;
                Ok((f.as_ref().clone(), s))
            })
            .collect::<FormatResult<Vec<_>>>()?;
        let nodes = columns
            .iter()
            .map(|(f, s)| node_of(f, s))
            .collect::<FormatResult<Vec<_>>>()?;
        let all_nullable = req.table_schema.fields().iter().all(|f| f.is_nullable());
        Ok(Self {
            meta: Arc::clone(meta),
            blocks: units.units.iter().map(|u| u.index).collect(),
            next: 0,
            req: req.clone(),
            out_schema,
            columns,
            o: o.clone(),
            pending: BTreeMap::new(),
            pending_bytes: 0,
            open: None,
            nodes,
            file_rows: Vec::new(),
            batch_rows: 0,
            row_width: row_width(&req.table_schema).max(1),
            all_nullable,
            metrics: ScanUnitMetrics {
                units: u64::try_from(units.units.len()).unwrap_or(u64::MAX),
                ..ScanUnitMetrics::default()
            },
            done: false,
        })
    }

    /// Whether the whole of `range` was pushed.
    fn has(&self, range: &Range<u64>) -> bool {
        self.pending
            .get(&range.start)
            .is_some_and(|b| b.len() as u64 == range.end.saturating_sub(range.start))
    }

    /// Takes the pushed bytes of `range`.
    fn take(&mut self, range: &Range<u64>) -> FormatResult<Bytes> {
        let Some(b) = self.pending.remove(&range.start) else {
            return Err(corrupt("a block whose bytes were never pushed"));
        };
        self.pending_bytes = self.pending_bytes.saturating_sub(b.len());
        if b.len() as u64 != range.end.saturating_sub(range.start) {
            return Err(corrupt("a truncated block"));
        }
        Ok(b)
    }

    /// Opens block `index`: takes its bytes, checks the sync marker and
    /// decompresses the records.
    fn open_block(&mut self, index: usize) -> FormatResult<()> {
        use apache_avro::Codec as AvCodec;
        let block = self.meta.blocks[index].clone();
        let raw = self.take(&block.bytes)?;
        if raw.len() < 16 || raw[raw.len() - 16..] != self.meta.sync {
            return Err(corrupt("a block with a bad sync marker"));
        }
        let data_start = usize::try_from(block.data.start.saturating_sub(block.bytes.start))
            .unwrap_or(usize::MAX);
        let data_end =
            usize::try_from(block.data.end.saturating_sub(block.bytes.start)).unwrap_or(usize::MAX);
        if data_start > data_end || data_end > raw.len().saturating_sub(16) {
            return Err(corrupt("a block with a bad length"));
        }
        let mut data = raw[data_start..data_end].to_vec();
        let codec = match self.meta.codec {
            OptCodec::Null => AvCodec::Null,
            OptCodec::Deflate => AvCodec::Deflate(apache_avro::DeflateSettings::default()),
            OptCodec::Snappy => AvCodec::Snappy,
            OptCodec::Zstandard => AvCodec::Zstandard(apache_avro::ZstandardSettings::default()),
            OptCodec::Bzip2 | OptCodec::Xz => {
                return Err(FormatError::Other(unsupported(format!(
                    "the Avro {} codec (not enabled in this build)",
                    self.meta.codec.header_name()
                ))));
            }
        };
        codec
            .decompress(&mut data)
            .map_err(|e| corrupt(format!("an undecodable block: {e}")))?;
        self.open = Some(OpenBlock {
            bytes: data,
            pos: 0,
            left: block.count,
            file_row: block.first_row,
        });
        Ok(())
    }

    /// Handles a corrupt record per `mode`: `FAILFAST` fails; otherwise the
    /// rest of the block is dropped, with one null row in `PERMISSIVE` mode
    /// when the output allows it.
    fn corrupt_record(&mut self, reason: String) -> FormatResult<bool> {
        match self.o.mode {
            Mode::FailFast => Err(corrupt(reason)),
            Mode::DropMalformed => {
                self.open = None;
                Ok(false)
            }
            Mode::Permissive => {
                let row = self.open.as_mut().map(|open| {
                    let row = open.file_row;
                    open.file_row += 1;
                    row
                });
                if let Some(file_row) = row
                    && self.all_nullable
                {
                    for node in &mut self.nodes {
                        append_null(node)?;
                    }
                    self.file_rows.push(file_row);
                    self.batch_rows += 1;
                }
                self.open = None;
                Ok(false)
            }
        }
    }

    /// Decodes one record of the open block into the batch (`false`: the
    /// block is exhausted).
    fn decode_one(&mut self) -> FormatResult<bool> {
        let Some(open) = self.open.as_mut() else {
            return Ok(false);
        };
        if open.left == 0 {
            if open.pos != open.bytes.len() {
                return Err(corrupt("a block with trailing bytes"));
            }
            return Ok(false);
        }
        let reader = apache_avro::reader::datum::GenericDatumReader::builder(&self.meta.writer)
            .build()
            .map_err(|e| corrupt(format!("an unreadable schema: {e}")))?;
        let mut cursor = std::io::Cursor::new(&open.bytes[open.pos..]);
        // `apache-avro`'s `GenericDatumReader::read_value` can panic on
        // malformed datum bytes (out-of-range indexes, unchecked slices)
        // instead of returning `Err`; catching the unwind turns such input
        // into the same typed corrupt-record path as a decode error. Both
        // paths leave position accounting to `corrupt_record` (which drops
        // the rest of the block), so neither advances `pos`/`left`/`file_row`
        // here.
        let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            reader.read_value(&mut cursor)
        }));
        let value = match value {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                return self.corrupt_record(format!("an undecodable record: {e}"));
            }
            Err(_) => {
                return self.corrupt_record("an undecodable record".to_owned());
            }
        };
        let consumed = usize::try_from(cursor.position()).unwrap_or(usize::MAX);
        open.pos = open.pos.saturating_add(consumed);
        open.left -= 1;
        let file_row = open.file_row;
        open.file_row += 1;
        // A value the read schema cannot hold is a schema error, not a
        // corrupt record: it fails in every mode.
        let resolved = resolve(
            &self.meta.writer,
            &self.meta.reader,
            value,
            self.o.positional,
        )?;
        let AvroValue::Record(pairs) = resolved else {
            return self.corrupt_record("a record value that is not a record".to_owned());
        };
        let selected = self
            .req
            .selection
            .as_ref()
            .is_none_or(|s| s.contains(file_row));
        if selected {
            let mut bad: Option<String> = None;
            for ((_, v), (node, (_, s))) in pairs
                .iter()
                .zip(self.nodes.iter_mut().zip(self.columns.iter()))
            {
                if let Err(e) = append(node, s, v) {
                    bad = Some(e.to_string());
                    break;
                }
            }
            if let Some(reason) = bad {
                return self.corrupt_record(reason);
            }
            check_rebase_value(&self.meta.reader, &AvroValue::Record(pairs), self.o.rebase)?;
            self.file_rows.push(file_row);
            self.batch_rows += 1;
        }
        Ok(true)
    }

    /// Emits the buffered rows as one batch of the request schema.
    fn emit(&mut self) -> FormatResult<Step> {
        let nodes = self
            .columns
            .iter()
            .map(|(f, s)| node_of(f, s))
            .collect::<FormatResult<Vec<_>>>()?;
        let built = std::mem::replace(&mut self.nodes, nodes);
        let arrays = built
            .into_iter()
            .zip(self.req.table_schema.fields().iter())
            .map(|(n, f)| finish_node(n, f))
            .collect::<FormatResult<Vec<_>>>()?;
        let full = RecordBatch::try_new_with_options(
            Arc::clone(&self.req.table_schema),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(self.batch_rows)),
        )
        .map_err(|e| corrupt(format!("an invalid batch: {e}")))?;
        let mut batch = project_batch(&full, &self.req.projection).map_err(FormatError::Other)?;
        if self.req.metadata_cols.row_index {
            let index: Vec<i64> = self
                .file_rows
                .iter()
                .map(|r| i64::try_from(*r).unwrap_or(i64::MAX))
                .collect();
            let column: ArrayRef = Arc::new(Int64Array::from(index));
            let mut fields: Vec<Field> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            fields.push(Field::new(ROW_INDEX_COLUMN, DataType::Int64, false));
            batch = RecordBatch::try_new_with_options(
                Arc::new(arrow_schema::Schema::new_with_metadata(
                    fields,
                    batch.schema().metadata().clone(),
                )),
                batch.columns().iter().cloned().chain([column]).collect(),
                &RecordBatchOptions::new().with_row_count(Some(self.batch_rows)),
            )
            .map_err(|e| corrupt(format!("an invalid batch: {e}")))?;
        }
        debug_assert_eq!(batch.schema(), self.out_schema);
        self.metrics.rows += self.batch_rows as u64;
        self.batch_rows = 0;
        self.file_rows.clear();
        Ok(Step::Batch(batch))
    }
}

impl Decoder for AvroDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if self.done {
            return Ok(Step::Done);
        }
        loop {
            if self.batch_rows >= BATCH_ROWS {
                return self.emit();
            }
            if self.open.is_none() {
                let Some(&index) = self.blocks.get(self.next) else {
                    if self.batch_rows > 0 {
                        return self.emit();
                    }
                    self.done = true;
                    return Ok(Step::Done);
                };
                let range = self.meta.blocks[index].bytes.clone();
                if !self.has(&range) {
                    return Ok(Step::NeedData(vec![range]));
                }
                self.open_block(index)?;
                self.next += 1;
            }
            while self.batch_rows < BATCH_ROWS {
                if !self.decode_one()? {
                    self.open = None;
                    break;
                }
            }
        }
    }

    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()> {
        for (r, b) in data {
            if r.start > r.end || b.len() as u64 != r.end.saturating_sub(r.start) {
                return Err(FormatError::Other(crate::error::internal(format!(
                    "pushed {} bytes for the range {r:?}",
                    b.len()
                ))));
            }
            self.pending_bytes += b.len();
            self.metrics.bytes_pushed += b.len() as u64;
            self.pending.insert(r.start, b);
        }
        Ok(())
    }

    fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        let mut out = Vec::new();
        let mut bytes = 0_u64;
        for &index in &self.blocks[self.next..] {
            let range = self.meta.blocks[index].bytes.clone();
            bytes += range.end.saturating_sub(range.start);
            if bytes > max_bytes {
                break;
            }
            if !self.has(&range) {
                out.push(range);
            }
        }
        out
    }

    fn buffered_bytes(&self) -> usize {
        self.pending_bytes + self.open.as_ref().map_or(0, |o| o.bytes.len() - o.pos)
    }

    fn working_bytes(&self) -> usize {
        self.batch_rows.saturating_mul(self.row_width)
    }

    fn next_decode_bytes(&self) -> usize {
        self.blocks
            .get(self.next)
            .map(|&i| {
                let b = &self.meta.blocks[i];
                usize::try_from(b.bytes.end.saturating_sub(b.bytes.start)).unwrap_or(usize::MAX)
            })
            .unwrap_or(0)
    }

    fn metrics(&self) -> ScanUnitMetrics {
        self.metrics
    }
}
