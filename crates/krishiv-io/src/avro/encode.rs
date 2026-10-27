// SPDX-License-Identifier: Apache-2.0
//! The Avro writer: Arrow batches into one container file, framed
//! incrementally (see the module docs).
//!
//! The header (magic, the `avro.schema` and `avro.codec` metadata, a sync
//! marker derived from the write job's seed and the schema JSON) is queued
//! first, then records are encoded straight into the open block's bytes: a
//! block is framed once it holds [`BLOCK_TARGET_BYTES`] of encoded records
//! (or [`BLOCK_ROWS`] rows, or a [`Encoder::flush`]) and drained through
//! [`Encoder::take_bytes`], so the framework uploads while the next block
//! builds. The sync marker is fixed per write job and file (never random):
//! two writes of the same rows, options and job agree byte for byte, while
//! the marker cannot be computed from the schema alone, so row data cannot
//! forge a block boundary for readers that split by scanning for it. Files
//! stay readable because readers use the marker stored in each file's
//! header.

use std::sync::Arc;

use apache_avro::Schema as AvroSchema;
use apache_avro::schema::InnerDecimalSchema;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, RecordBatch};
use arrow_schema::{DataType, SchemaRef, TimeUnit};
use bytes::Bytes;

use super::block_codec;
use super::options::AvroOptions;
use super::schema::{schema_json_of_arrow, schema_kind};
use crate::error::{FormatError, FormatResult};
use crate::format::{Encoder, WrittenFile};
use crate::options::FormatOptions;
use crate::stats::{FileStatistics, Precision};

/// Encoded records per block before framing, bounding one block's memory.
const BLOCK_ROWS: usize = 4096;

/// Encoded bytes per block before framing.
const BLOCK_TARGET_BYTES: usize = 256 << 10;

/// The magic of an Avro container file.
const MAGIC: &[u8; 4] = b"Obj\x01";

/// A write error: the batch cannot be encoded as the writer schema.
fn write_failed(reason: impl Into<String>) -> FormatError {
    FormatError::WriteFailed(reason.into())
}

/// The zigzag varint of `v` appended to `out`.
fn put_long(out: &mut Vec<u8>, v: i64) {
    let mut raw = v
        .unsigned_abs()
        .wrapping_mul(2)
        .wrapping_sub(u64::from(v.is_negative()));
    loop {
        let b = (raw & 0x7F) as u8;
        raw >>= 7;
        if raw == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// The length-delimited `bytes` appended to `out`.
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_long(out, i64::try_from(bytes.len()).unwrap_or(i64::MAX));
    out.extend_from_slice(bytes);
}

/// The minimal big-endian two's complement bytes of `v` (at least one).
fn minimal_bytes(v: i128) -> Vec<u8> {
    let full = v.to_be_bytes();
    let mut start = 0;
    while start < 15 {
        let (a, b) = (full[start], full[start + 1]);
        if (a == 0x00 && b & 0x80 == 0) || (a == 0xFF && b & 0x80 != 0) {
            start += 1;
        } else {
            break;
        }
    }
    full[start..].to_vec()
}

/// The row window of list offsets (validated: offsets come from validated
/// batches, so this is a broken invariant, never untrusted input).
///
/// # Errors
/// [`FormatError::WriteFailed`] for a window outside the values.
fn window<O>(offsets: &[O], row: usize, len: usize) -> FormatResult<(usize, usize)>
where
    O: TryInto<usize> + Copy,
{
    let bad = || write_failed("a list value outside its offsets");
    let (Some(&a), Some(&b)) = (offsets.get(row), offsets.get(row + 1)) else {
        return Err(bad());
    };
    let (a, b) = (
        a.try_into().map_err(|_| bad())?,
        b.try_into().map_err(|_| bad())?,
    );
    if a > b || b > len {
        return Err(bad());
    }
    Ok((a, b))
}

/// The `row`th string of a UTF-8 array of any of its three layouts.
fn str_at(array: &ArrayRef, row: usize) -> Option<&str> {
    match array.data_type() {
        DataType::Utf8View => Some(array.as_string_view().value(row)),
        DataType::Utf8 => Some(array.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => Some(array.as_string::<i64>().value(row)),
        _ => None,
    }
}

/// The `row`th value of a primitive array of type `T`.
fn prim<T: ArrowPrimitiveType>(array: &ArrayRef, row: usize) -> T::Native {
    array.as_primitive::<T>().value(row)
}

/// The count-prefixed block of the elements `start..end` of `values`
/// (nothing but the closing zero block when empty), as an array's items.
///
/// # Errors
/// [`FormatError::WriteFailed`] for an element the schema cannot hold.
fn put_items(
    out: &mut Vec<u8>,
    items: &AvroSchema,
    values: &ArrayRef,
    start: usize,
    end: usize,
) -> FormatResult<()> {
    if end > start {
        put_long(out, i64::try_from(end - start).unwrap_or(i64::MAX));
        for i in start..end {
            put_cell(out, items, values, i)?;
        }
    }
    put_long(out, 0);
    Ok(())
}

/// `v`, the unscaled value of an Arrow decimal of scale `from`, as the
/// unscaled value of an Avro decimal of `precision` digits and scale `to`.
///
/// # Errors
/// [`FormatError::WriteFailed`] when the rescale overflows, drops non-zero
/// digits or leaves more than `precision` digits.
fn rescale(v: i128, from: i8, to: usize, precision: usize) -> FormatResult<i128> {
    let fail = || write_failed("a decimal value the Avro schema cannot hold");
    if v == 0 {
        return Ok(0);
    }
    let diff = i64::try_from(to)
        .map_err(|_| fail())?
        .checked_sub(i64::from(from))
        .ok_or_else(fail)?;
    let factor = 10_i128
        .checked_pow(u32::try_from(diff.unsigned_abs()).map_err(|_| fail())?)
        .ok_or_else(fail)?;
    let v = if diff >= 0 {
        v.checked_mul(factor).ok_or_else(fail)?
    } else if v % factor == 0 {
        v / factor
    } else {
        return Err(fail());
    };
    let limit = u32::try_from(precision)
        .ok()
        .and_then(|p| 10_u128.checked_pow(p));
    if limit.is_some_and(|limit| v.unsigned_abs() >= limit) {
        return Err(fail());
    }
    Ok(v)
}

/// The bytes of cell `row` of `array` in `schema` (writer) terms, appended
/// to `out`.
///
/// # Errors
/// [`FormatError::WriteFailed`] for a null in a non-nullable schema or a
/// value the schema cannot hold.
fn put_cell(
    out: &mut Vec<u8>,
    schema: &AvroSchema,
    array: &ArrayRef,
    row: usize,
) -> FormatResult<()> {
    // Nullable unions wrap every value; the null branch holds nulls.
    if let AvroSchema::Union(u) = schema {
        let too_many = || write_failed("a union with too many branches");
        if array.is_null(row) {
            let Some(null) = u
                .variants()
                .iter()
                .position(|s| matches!(s, AvroSchema::Null))
            else {
                return Err(write_failed("a null value for a non-nullable field"));
            };
            put_long(out, i64::try_from(null).map_err(|_| too_many())?);
            return Ok(());
        }
        let mut branches = u
            .variants()
            .iter()
            .enumerate()
            .filter(|(_, s)| !matches!(s, AvroSchema::Null));
        let (Some((index, branch)), None) = (branches.next(), branches.next()) else {
            return Err(write_failed(
                "a value for a union without exactly one non-null branch",
            ));
        };
        put_long(out, i64::try_from(index).map_err(|_| too_many())?);
        return put_cell(out, branch, array, row);
    }
    if array.is_null(row) {
        return Err(write_failed("a null value for a non-nullable field"));
    }
    put_scalar(out, schema, array, row)
}

/// The bytes of the non-null cell `row` of `array` in `schema` terms,
/// appended to `out` in Avro's binary encoding.
///
/// # Errors
/// [`FormatError::WriteFailed`] for a value the schema cannot hold.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per Arrow type; each validates its range"
)]
fn put_scalar(
    out: &mut Vec<u8>,
    schema: &AvroSchema,
    array: &ArrayRef,
    row: usize,
) -> FormatResult<()> {
    use arrow_array::types as t;
    let fail = |what: &str| write_failed(format!("a {what} value the Avro schema cannot hold"));
    match (schema, array.data_type()) {
        (AvroSchema::Boolean, DataType::Boolean) => {
            out.push(u8::from(array.as_boolean().value(row)));
        }
        (AvroSchema::Int, DataType::Int8) => {
            put_long(out, i64::from(prim::<t::Int8Type>(array, row)));
        }
        (AvroSchema::Int, DataType::Int16) => {
            put_long(out, i64::from(prim::<t::Int16Type>(array, row)));
        }
        (AvroSchema::Int | AvroSchema::Long, DataType::Int32) => {
            put_long(out, i64::from(prim::<t::Int32Type>(array, row)));
        }
        (AvroSchema::Long, DataType::Int64) => put_long(out, prim::<t::Int64Type>(array, row)),
        (AvroSchema::Float, DataType::Float32) => {
            out.extend_from_slice(&prim::<t::Float32Type>(array, row).to_le_bytes());
        }
        (AvroSchema::Double, DataType::Float32) => {
            out.extend_from_slice(&f64::from(prim::<t::Float32Type>(array, row)).to_le_bytes());
        }
        (AvroSchema::Double, DataType::Float64) => {
            out.extend_from_slice(&prim::<t::Float64Type>(array, row).to_le_bytes());
        }
        (AvroSchema::Bytes, DataType::BinaryView) => {
            put_bytes(out, array.as_binary_view().value(row));
        }
        (AvroSchema::Bytes, DataType::Binary) => {
            put_bytes(out, array.as_binary::<i32>().value(row));
        }
        (AvroSchema::Bytes, DataType::LargeBinary) => {
            put_bytes(out, array.as_binary::<i64>().value(row));
        }
        (AvroSchema::Bytes, DataType::Utf8View) => {
            put_bytes(out, array.as_string_view().value(row).as_bytes());
        }
        (AvroSchema::String, DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8) => {
            let s = str_at(array, row).ok_or_else(|| fail("string"))?;
            put_bytes(out, s.as_bytes());
        }
        (AvroSchema::String, DataType::BinaryView) => {
            let v = array.as_binary_view().value(row);
            std::str::from_utf8(v).map_err(|_| write_failed("binary bytes that are not UTF-8"))?;
            put_bytes(out, v);
        }
        (AvroSchema::Fixed(f), DataType::BinaryView | DataType::Binary) => {
            let v = match array.data_type() {
                DataType::Binary => array.as_binary::<i32>().value(row),
                _ => array.as_binary_view().value(row),
            };
            if v.len() != f.size {
                return Err(fail("fixed"));
            }
            out.extend_from_slice(v);
        }
        (AvroSchema::Decimal(d), DataType::Decimal128(_, scale)) => {
            let v = rescale(
                prim::<t::Decimal128Type>(array, row),
                *scale,
                d.scale,
                d.precision,
            )?;
            let minimal = minimal_bytes(v);
            match &d.inner {
                InnerDecimalSchema::Bytes => put_bytes(out, &minimal),
                InnerDecimalSchema::Fixed(f) => {
                    let pad = f
                        .size
                        .checked_sub(minimal.len())
                        .ok_or_else(|| fail("decimal"))?;
                    out.resize(out.len() + pad, if v < 0 { 0xFF } else { 0x00 });
                    out.extend_from_slice(&minimal);
                }
            }
        }
        (AvroSchema::Date, DataType::Date32) => {
            put_long(out, i64::from(prim::<t::Date32Type>(array, row)));
        }
        (AvroSchema::TimeMillis, DataType::Time32(TimeUnit::Millisecond)) => {
            put_long(out, i64::from(prim::<t::Time32MillisecondType>(array, row)));
        }
        (AvroSchema::TimeMicros, DataType::Time64(TimeUnit::Microsecond)) => {
            put_long(out, prim::<t::Time64MicrosecondType>(array, row));
        }
        (
            AvroSchema::TimestampMillis | AvroSchema::LocalTimestampMillis,
            DataType::Timestamp(TimeUnit::Millisecond, _),
        ) => put_long(out, prim::<t::TimestampMillisecondType>(array, row)),
        (
            AvroSchema::TimestampMicros | AvroSchema::LocalTimestampMicros,
            DataType::Timestamp(TimeUnit::Microsecond, _),
        ) => put_long(out, prim::<t::TimestampMicrosecondType>(array, row)),
        (AvroSchema::Record(r), DataType::Struct(_)) => {
            let s = array.as_struct();
            for f in &r.fields {
                let Some(child) = s.column_by_name(&f.name) else {
                    return Err(write_failed(format!(
                        "a batch without the field {}",
                        f.name
                    )));
                };
                put_cell(out, &f.schema, child, row)?;
            }
        }
        (AvroSchema::Array(a), DataType::List(_)) => {
            let l = array.as_list::<i32>();
            let (start, end) = window(l.value_offsets(), row, l.values().len())?;
            put_items(out, &a.items, l.values(), start, end)?;
        }
        (AvroSchema::Array(a), DataType::LargeList(_)) => {
            let l = array.as_list::<i64>();
            let (start, end) = window(l.value_offsets(), row, l.values().len())?;
            put_items(out, &a.items, l.values(), start, end)?;
        }
        (AvroSchema::Array(a), DataType::FixedSizeList(_, _)) => {
            let l = array.as_fixed_size_list();
            let bad = || write_failed("a list value outside its offsets");
            let len = usize::try_from(l.value_length()).map_err(|_| bad())?;
            let start = row.checked_mul(len).ok_or_else(bad)?;
            let end = start
                .checked_add(len)
                .filter(|end| *end <= l.values().len())
                .ok_or_else(bad)?;
            put_items(out, &a.items, l.values(), start, end)?;
        }
        (AvroSchema::Map(m), DataType::Map(_, _)) => {
            let map = array.as_map();
            let (start, end) = window(map.value_offsets(), row, map.entries().len())?;
            let entries = map.entries();
            let keys = entries.column(0);
            // Sorted by key, so the output does not depend on storage order;
            // of equal keys the last one wins.
            let mut order = (start..end)
                .map(|i| {
                    str_at(keys, i)
                        .map(|k| (k, i))
                        .ok_or_else(|| write_failed("a map key that is not a UTF-8 string"))
                })
                .collect::<FormatResult<Vec<_>>>()?;
            order.sort_by(|a, b| a.0.cmp(b.0));
            let mut kept = Vec::with_capacity(order.len());
            for (n, &(key, i)) in order.iter().enumerate() {
                if order.get(n + 1).is_none_or(|next| next.0 != key) {
                    kept.push((key, i));
                }
            }
            if !kept.is_empty() {
                put_long(out, i64::try_from(kept.len()).unwrap_or(i64::MAX));
                for (key, i) in kept {
                    put_bytes(out, key.as_bytes());
                    put_cell(out, &m.types, entries.column(1), i)?;
                }
            }
            put_long(out, 0);
        }
        (AvroSchema::Enum(e), DataType::Utf8View) => {
            let symbol = array.as_string_view().value(row);
            let index = e
                .symbols
                .iter()
                .position(|s| s == symbol)
                .ok_or_else(|| fail("enum"))?;
            put_long(out, i64::try_from(index).map_err(|_| fail("enum"))?);
        }
        _ => {
            return Err(write_failed(format!(
                "a {} value for an Avro {} column",
                array.data_type(),
                schema_kind(schema)
            )));
        }
    }
    Ok(())
}

/// The sans-IO Avro encoder: batches into framed, compressed blocks.
#[derive(Debug)]
pub(crate) struct AvroEncoder {
    schema: Arc<AvroSchema>,
    codec: apache_avro::Codec,
    sync: [u8; 16],
    block_rows: usize,
    /// Framed bytes ready to upload.
    out: Vec<u8>,
    /// The open block's encoded records.
    pending: Vec<u8>,
    pending_rows: usize,
    rows: u64,
    taken: u64,
}

impl AvroEncoder {
    /// An encoder of `schema` rows with `o`.
    ///
    /// # Errors
    /// `CODEC_NOT_AVAILABLE.WITH_AVAILABLE_CODECS_SUGGESTION` for the
    /// zstandard codec and `KRISHIV_UNSUPPORTED_FEATURE` for the other
    /// codecs this build cannot write;
    /// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for a bad `avroSchema`;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a schema with no Avro type.
    pub(crate) fn new(schema: &SchemaRef, o: &FormatOptions) -> FormatResult<Self> {
        let opts = AvroOptions::new(o, true)?;
        let codec = block_codec(opts.codec)?;
        let json = match &opts.avro_schema {
            Some(s) => s.clone(),
            None => schema_json_of_arrow(schema, &opts.record_name, &opts.record_namespace)?,
        };
        let parsed = apache_avro::Schema::parse_str(&json).map_err(|e| {
            FormatError::Other(crate::text::invalid_option(
                "avro",
                "avroSchema",
                &json,
                &format!("the schema does not parse: {e}"),
            ))
        })?;
        let AvroSchema::Record(_) = parsed else {
            return Err(FormatError::Other(crate::text::invalid_option(
                "avro",
                "avroSchema",
                &json,
                "the schema is not a record",
            )));
        };
        let sync = sync_of(opts.sync_seed.as_deref(), &json);
        let mut out = Vec::with_capacity(1024);
        out.extend_from_slice(MAGIC);
        // The metadata map: exactly the schema and the codec, as Spark writes.
        put_long(&mut out, 2);
        put_bytes(&mut out, b"avro.schema");
        put_bytes(&mut out, json.as_bytes());
        put_bytes(&mut out, b"avro.codec");
        put_bytes(&mut out, opts.codec.header_name().as_bytes());
        put_long(&mut out, 0);
        out.extend_from_slice(&sync);
        Ok(Self {
            schema: Arc::new(parsed),
            codec,
            sync,
            block_rows: opts.block_rows.map_or(BLOCK_ROWS, |n| n.min(BLOCK_ROWS)),
            out,
            pending: Vec::new(),
            pending_rows: 0,
            rows: 0,
            taken: 0,
        })
    }

    /// Frames the open block, if it holds rows.
    fn flush_block(&mut self) -> FormatResult<()> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let mut data = std::mem::take(&mut self.pending);
        self.codec
            .compress(&mut data)
            .map_err(|e| write_failed(format!("an incompressible block: {e}")))?;
        put_long(
            &mut self.out,
            i64::try_from(self.pending_rows).unwrap_or(i64::MAX),
        );
        put_bytes(&mut self.out, &data);
        self.out.extend_from_slice(&self.sync);
        self.pending_rows = 0;
        Ok(())
    }
}

/// The sync marker of a file: XXH64 (two seeds) of the write job's `seed`
/// and the schema JSON, so a job's file is reproducible while the marker
/// cannot be computed from the schema alone. Without a seed (a write outside
/// a write job) the marker is a function of the schema only.
fn sync_of(seed: Option<&str>, json: &str) -> [u8; 16] {
    let mut material = Vec::with_capacity(json.len() + seed.map_or(0, str::len) + 1);
    material.extend_from_slice(seed.unwrap_or_default().as_bytes());
    material.push(0);
    material.extend_from_slice(json.as_bytes());
    let (a, b) = (
        twox_hash::XxHash64::oneshot(0xcbf2_9ce4_8422_2325, &material),
        twox_hash::XxHash64::oneshot(0x8422_2325_cbf2_9ce4, &material),
    );
    let mut sync = [0_u8; 16];
    sync[..8].copy_from_slice(&a.to_le_bytes());
    sync[8..].copy_from_slice(&b.to_le_bytes());
    sync
}

impl Encoder for AvroEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        let schema = Arc::clone(&self.schema);
        let AvroSchema::Record(record) = &*schema else {
            return Err(write_failed("a writer schema that is not a record"));
        };
        // Each field's column, resolved once for the batch.
        let batch_schema = batch.schema();
        let columns = record
            .fields
            .iter()
            .map(|f| {
                let index = batch_schema
                    .index_of(&f.name)
                    .map_err(|_| write_failed(format!("a batch without the field {}", f.name)))?;
                Ok((&f.schema, batch.column(index)))
            })
            .collect::<FormatResult<Vec<_>>>()?;
        for row in 0..batch.num_rows() {
            let mark = self.pending.len();
            for (field, column) in &columns {
                if let Err(e) = put_cell(&mut self.pending, field, column, row) {
                    self.pending.truncate(mark);
                    return Err(e);
                }
            }
            self.pending_rows += 1;
            self.rows += 1;
            if self.pending_rows >= self.block_rows || self.pending.len() >= BLOCK_TARGET_BYTES {
                self.flush_block()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> FormatResult<()> {
        self.flush_block()
    }

    fn take_bytes(&mut self) -> Option<Bytes> {
        if self.out.is_empty() {
            return None;
        }
        let out = std::mem::take(&mut self.out);
        self.taken += out.len() as u64;
        Some(Bytes::from(out))
    }

    fn finish(mut self: Box<Self>) -> FormatResult<WrittenFile> {
        self.flush_block()?;
        let size = self.taken + self.out.len() as u64;
        let tail = Bytes::from(std::mem::take(&mut self.out));
        Ok(WrittenFile::new(
            tail,
            FileStatistics {
                rows: Precision::Exact(self.rows),
                bytes: Precision::Exact(size),
                ..FileStatistics::default()
            },
        ))
    }

    fn buffered_bytes(&self) -> usize {
        self.out.capacity() + self.pending.capacity()
    }

    fn written_bytes(&self) -> u64 {
        self.taken + self.out.len() as u64 + self.pending.len() as u64
    }
}
