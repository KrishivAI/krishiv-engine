// SPDX-License-Identifier: Apache-2.0
//! The Avro writer: Arrow batches into one container file, framed
//! incrementally (see the module docs).
//!
//! The header (magic, the `avro.schema` and `avro.codec` metadata, a
//! deterministic sync marker derived from the schema JSON) is queued first,
//! then records accumulate into blocks: a block is framed once it holds
//! [`BLOCK_TARGET_BYTES`] of encoded records (or [`BLOCK_ROWS`] rows, or a
//! [`Encoder::flush`]) and drained through [`Encoder::take_bytes`], so the
//! framework uploads while the next block builds. The sync marker is fixed
//! per schema (never random): two writes of the same rows and options agree
//! byte for byte.

use std::collections::BTreeMap;

use apache_avro::{Decimal as AvroDecimal, Schema as AvroSchema, types::Value as AvroValue};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, SchemaRef, TimeUnit};
use bytes::Bytes;

use super::options::{AvroOptions, Codec as OptCodec};
use super::schema::{schema_json_of_arrow, schema_kind};
use crate::error::{FormatError, FormatResult, unsupported};
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

/// The value of cell `row` of `array` in `schema` (writer) terms.
///
/// # Errors
/// [`FormatError::WriteFailed`] for a null in a non-nullable schema or a
/// value the schema cannot hold.
fn cell(schema: &AvroSchema, array: &ArrayRef, row: usize) -> FormatResult<AvroValue> {
    // Nullable unions wrap every value; the null branch holds nulls.
    if let AvroSchema::Union(u) = schema {
        let branches: Vec<(usize, &AvroSchema)> = u
            .variants()
            .iter()
            .enumerate()
            .filter(|(_, s)| !matches!(s, AvroSchema::Null))
            .collect();
        let null = u
            .variants()
            .iter()
            .position(|s| matches!(s, AvroSchema::Null));
        if array.is_null(row) {
            let Some(null) = null else {
                return Err(write_failed("a null value for a non-nullable field"));
            };
            let index =
                u32::try_from(null).map_err(|_| write_failed("a union with too many branches"))?;
            return Ok(AvroValue::Union(index, Box::new(AvroValue::Null)));
        }
        let [(branch_index, branch)] = branches.as_slice() else {
            return Err(write_failed(
                "a value for a union without exactly one non-null branch",
            ));
        };
        let index = u32::try_from(*branch_index)
            .map_err(|_| write_failed("a union with too many branches"))?;
        return Ok(AvroValue::Union(index, Box::new(cell(branch, array, row)?)));
    }
    if array.is_null(row) {
        return Err(write_failed("a null value for a non-nullable field"));
    }
    scalar(schema, array, row)
}

/// The non-null value of cell `row` of `array` in `schema` terms.
///
/// # Errors
/// [`FormatError::WriteFailed`] for a value the schema cannot hold.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per Arrow type; each validates its range"
)]
fn scalar(schema: &AvroSchema, array: &ArrayRef, row: usize) -> FormatResult<AvroValue> {
    let fail = |what: &str| write_failed(format!("a {what} value the Avro schema cannot hold"));
    Ok(match (schema, array.data_type()) {
        (AvroSchema::Boolean, DataType::Boolean) => {
            AvroValue::Boolean(array.as_boolean().value(row))
        }
        (AvroSchema::Int, DataType::Int8) => AvroValue::Int(i32::from(
            array
                .as_primitive::<arrow_array::types::Int8Type>()
                .value(row),
        )),
        (AvroSchema::Int, DataType::Int16) => AvroValue::Int(i32::from(
            array
                .as_primitive::<arrow_array::types::Int16Type>()
                .value(row),
        )),
        (AvroSchema::Int, DataType::Int32) => AvroValue::Int(
            array
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(row),
        ),
        (AvroSchema::Long, DataType::Int64) => AvroValue::Long(
            array
                .as_primitive::<arrow_array::types::Int64Type>()
                .value(row),
        ),
        (AvroSchema::Long, DataType::Int32) => AvroValue::Long(i64::from(
            array
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(row),
        )),
        (AvroSchema::Float, DataType::Float32) => AvroValue::Float(
            array
                .as_primitive::<arrow_array::types::Float32Type>()
                .value(row),
        ),
        (AvroSchema::Double, DataType::Float32) => AvroValue::Double(f64::from(
            array
                .as_primitive::<arrow_array::types::Float32Type>()
                .value(row),
        )),
        (AvroSchema::Double, DataType::Float64) => AvroValue::Double(
            array
                .as_primitive::<arrow_array::types::Float64Type>()
                .value(row),
        ),
        (AvroSchema::Bytes, DataType::BinaryView) => {
            AvroValue::Bytes(array.as_binary_view().value(row).to_vec())
        }
        (AvroSchema::Bytes, DataType::Binary) => {
            AvroValue::Bytes(array.as_binary::<i32>().value(row).to_vec())
        }
        (AvroSchema::Bytes, DataType::LargeBinary) => {
            AvroValue::Bytes(array.as_binary::<i64>().value(row).to_vec())
        }
        (AvroSchema::Bytes, DataType::Utf8View) => {
            AvroValue::Bytes(array.as_string_view().value(row).as_bytes().to_vec())
        }
        (AvroSchema::String, DataType::Utf8View) => {
            AvroValue::String(array.as_string_view().value(row).to_owned())
        }
        (AvroSchema::String, DataType::Utf8) => {
            AvroValue::String(array.as_string::<i32>().value(row).to_owned())
        }
        (AvroSchema::String, DataType::LargeUtf8) => {
            AvroValue::String(array.as_string::<i64>().value(row).to_owned())
        }
        (AvroSchema::Fixed(f), DataType::BinaryView) => {
            let v = array.as_binary_view().value(row);
            if v.len() != f.size {
                return Err(fail("fixed"));
            }
            AvroValue::Fixed(f.size, v.to_vec())
        }
        (AvroSchema::Fixed(f), DataType::Binary) => {
            let v = array.as_binary::<i32>().value(row);
            if v.len() != f.size {
                return Err(fail("fixed"));
            }
            AvroValue::Fixed(f.size, v.to_vec())
        }
        (AvroSchema::Decimal(_), DataType::Decimal128(_, _)) => {
            let v = array
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(row);
            AvroValue::Decimal(AvroDecimal::from(minimal_bytes(v)))
        }
        (AvroSchema::Date, DataType::Date32) => AvroValue::Date(
            array
                .as_primitive::<arrow_array::types::Date32Type>()
                .value(row),
        ),
        (AvroSchema::TimeMillis, DataType::Time32(TimeUnit::Millisecond)) => AvroValue::TimeMillis(
            array
                .as_primitive::<arrow_array::types::Time32MillisecondType>()
                .value(row),
        ),
        (AvroSchema::TimeMicros, DataType::Time64(TimeUnit::Microsecond)) => AvroValue::TimeMicros(
            array
                .as_primitive::<arrow_array::types::Time64MicrosecondType>()
                .value(row),
        ),
        (AvroSchema::TimestampMillis, DataType::Timestamp(TimeUnit::Millisecond, _)) => {
            AvroValue::TimestampMillis(
                array
                    .as_primitive::<arrow_array::types::TimestampMillisecondType>()
                    .value(row),
            )
        }
        (AvroSchema::TimestampMicros, DataType::Timestamp(TimeUnit::Microsecond, _)) => {
            AvroValue::TimestampMicros(
                array
                    .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                    .value(row),
            )
        }
        (AvroSchema::LocalTimestampMillis, DataType::Timestamp(TimeUnit::Millisecond, _)) => {
            AvroValue::LocalTimestampMillis(
                array
                    .as_primitive::<arrow_array::types::TimestampMillisecondType>()
                    .value(row),
            )
        }
        (AvroSchema::LocalTimestampMicros, DataType::Timestamp(TimeUnit::Microsecond, _)) => {
            AvroValue::LocalTimestampMicros(
                array
                    .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                    .value(row),
            )
        }
        (AvroSchema::Record(r), DataType::Struct(_)) => {
            let s = array.as_struct();
            let mut pairs = Vec::with_capacity(r.fields.len());
            for f in &r.fields {
                let Some(child) = s.column_by_name(&f.name) else {
                    return Err(write_failed(format!(
                        "a batch without the field {}",
                        f.name
                    )));
                };
                pairs.push((f.name.clone(), cell(&f.schema, child, row)?));
            }
            AvroValue::Record(pairs)
        }
        (AvroSchema::Array(a), DataType::List(_)) => {
            let l = array.as_list::<i32>();
            let (start, end) = window(l.value_offsets(), row, l.values().len())?;
            (start..end)
                .map(|i| cell(&a.items, l.values(), i))
                .collect::<FormatResult<Vec<_>>>()
                .map(AvroValue::Array)?
        }
        (AvroSchema::Array(a), DataType::LargeList(_)) => {
            let l = array.as_list::<i64>();
            let (start, end) = window(l.value_offsets(), row, l.values().len())?;
            (start..end)
                .map(|i| cell(&a.items, l.values(), i))
                .collect::<FormatResult<Vec<_>>>()
                .map(AvroValue::Array)?
        }
        (AvroSchema::Array(a), DataType::FixedSizeList(_, _)) => {
            let l = array.as_fixed_size_list();
            let len = usize::try_from(l.value_length())
                .map_err(|_| write_failed("a list value outside its offsets"))?;
            let start = row
                .checked_mul(len)
                .ok_or_else(|| write_failed("a list value outside its offsets"))?;
            let end = start
                .checked_add(len)
                .filter(|end| *end <= l.values().len())
                .ok_or_else(|| write_failed("a list value outside its offsets"))?;
            (start..end)
                .map(|i| cell(&a.items, l.values(), i))
                .collect::<FormatResult<Vec<_>>>()
                .map(AvroValue::Array)?
        }
        (AvroSchema::Map(m), DataType::Map(_, _)) => {
            let map = array.as_map();
            let (start, end) = window(map.value_offsets(), row, map.entries().len())?;
            let entries = map.entries();
            let keys = entries.column(0).as_string_view();
            // Sorted by key, so the output does not depend on storage order.
            let mut order: Vec<usize> = (start..end).collect();
            order.sort_by(|&a, &b| keys.value(a).cmp(keys.value(b)));
            let mut out = BTreeMap::new();
            for i in order {
                out.insert(
                    keys.value(i).to_owned(),
                    cell(&m.types, entries.column(1), i)?,
                );
            }
            AvroValue::Map(out.into_iter().collect())
        }
        (AvroSchema::Enum(e), DataType::Utf8View) => {
            let symbol = array.as_string_view().value(row).to_owned();
            let index = e
                .symbols
                .iter()
                .position(|s| s == &symbol)
                .and_then(|i| u32::try_from(i).ok())
                .ok_or_else(|| fail("enum"))?;
            AvroValue::Enum(index, symbol)
        }
        (AvroSchema::String, DataType::BinaryView) => {
            let v = array.as_binary_view().value(row);
            AvroValue::String(
                std::str::from_utf8(v)
                    .map_err(|_| write_failed("binary bytes that are not UTF-8"))?
                    .to_owned(),
            )
        }
        _ => {
            return Err(write_failed(format!(
                "a {} value for an Avro {} column",
                array.data_type(),
                schema_kind(schema)
            )));
        }
    })
}

/// The sans-IO Avro encoder: batches into framed, compressed blocks.
#[derive(Debug)]
pub(crate) struct AvroEncoder {
    schema: AvroSchema,
    codec: OptCodec,
    sync: [u8; 16],
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
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a codec this build cannot write;
    /// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for a bad `avroSchema`;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a schema with no Avro type.
    pub(crate) fn new(schema: &SchemaRef, o: &FormatOptions) -> FormatResult<Self> {
        let opts = AvroOptions::new(o, true)?;
        match opts.codec {
            OptCodec::Bzip2 | OptCodec::Xz => {
                return Err(FormatError::Other(unsupported(format!(
                    "writing Avro with the {} codec (only uncompressed, snappy, deflate and zstandard are supported)",
                    opts.codec.header_name()
                ))));
            }
            OptCodec::Null | OptCodec::Snappy | OptCodec::Deflate | OptCodec::Zstandard => {}
        }
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
        let sync = sync_of(&json);
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
            schema: parsed,
            codec: opts.codec,
            sync,
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
        use apache_avro::Codec as AvCodec;
        let codec = match self.codec {
            OptCodec::Null => AvCodec::Null,
            OptCodec::Deflate => AvCodec::Deflate(apache_avro::DeflateSettings::default()),
            OptCodec::Snappy => AvCodec::Snappy,
            OptCodec::Zstandard => AvCodec::Zstandard(apache_avro::ZstandardSettings::default()),
            OptCodec::Bzip2 | OptCodec::Xz => {
                return Err(FormatError::Other(unsupported(format!(
                    "writing Avro with the {} codec (only uncompressed, snappy, deflate and zstandard are supported)",
                    self.codec.header_name()
                ))));
            }
        };
        let mut data = std::mem::take(&mut self.pending);
        codec
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

/// The sync marker of a schema: FNV-1a of its JSON with two seeds, so two
/// writes of the same schema agree byte for byte.
fn sync_of(json: &str) -> [u8; 16] {
    fn fnv(json: &str, seed: u64) -> u64 {
        let mut h = seed;
        for b in json.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
    let (a, b) = (
        fnv(json, 0xcbf2_9ce4_8422_2325),
        fnv(json, 0x8422_2325_cbf2_9ce4),
    );
    let mut sync = [0_u8; 16];
    sync[..8].copy_from_slice(&a.to_le_bytes());
    sync[8..].copy_from_slice(&b.to_le_bytes());
    sync
}

impl Encoder for AvroEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        let fields = match &self.schema {
            AvroSchema::Record(r) => r.fields.clone(),
            _ => return Err(write_failed("a writer schema that is not a record")),
        };
        let schema = self.schema.clone();
        let writer = apache_avro::writer::datum::GenericDatumWriter::builder(&schema)
            .build()
            .map_err(|e| write_failed(format!("an unwritable schema: {e}")))?;
        for row in 0..batch.num_rows() {
            let mut pairs = Vec::with_capacity(fields.len());
            for f in &fields {
                let index = batch
                    .schema()
                    .index_of(&f.name)
                    .map_err(|_| write_failed(format!("a batch without the field {}", f.name)))?;
                pairs.push((f.name.clone(), cell(&f.schema, batch.column(index), row)?));
            }
            let datum = writer
                .write_value_to_vec(AvroValue::Record(pairs))
                .map_err(|e| write_failed(format!("an unencodable row: {e}")))?;
            self.pending.extend_from_slice(&datum);
            self.pending_rows += 1;
            self.rows += 1;
            if self.pending_rows >= BLOCK_ROWS || self.pending.len() >= BLOCK_TARGET_BYTES {
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
