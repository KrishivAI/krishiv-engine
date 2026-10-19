// SPDX-License-Identifier: Apache-2.0
//! `MockFormat`: a small columnar format built on Arrow IPC that implements
//! every part of [`FileFormat`] the way a real columnar format would, so the
//! conformance suite runs against it (P2-T117).
//!
//! Layout: `KMCK`, then units of rows (`batchRows` per unit); each unit
//! stores each top-level column as its own chunk (a one-column Arrow IPC
//! stream followed by its CRC-32), so projection reads only the needed
//! chunks and coalescing has gaps to merge. The footer holds the schema,
//! per-unit chunk offsets and min/max/null-count statistics, and ends with
//! its length and `KMCK`. With the `encryptionKey` option the footer is
//! XOR-enciphered (a stand-in for encrypted Parquet footers: the metadata
//! cache must key it by the key's fingerprint).
//!
//! Options exercise the framework: `batchRows`, `statsTruncateLength`
//! (truncated string bounds, inexact), `legacyStats` (every bound inexact,
//! as statistics of writers that ordered them wrongly), `pageChecksumVerification`
//! (CRC check, conf fallback `spark.sql.parquet.pageChecksumVerification`),
//! `asyncDecoder` (decode through [`AsyncDecoder`] over [`PushReadAt`]),
//! `encryptionKey` (secret).

use std::collections::VecDeque;
use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, RecordBatchOptions, StructArray,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Fields, Schema, SchemaRef};
use bytes::Bytes;
use futures::future::BoxFuture;
use krishiv_common::secret::SecretString;
use krishiv_io::{
    AsyncDecoder, ByteRange, Capabilities, ColumnPath, ColumnStatistics, Decoder, Encoder,
    FileFormat, FileMetadata, FileRef, FileStatistics, FormatError, FormatOptions, FormatResult,
    IoContext, MetadataNeeds, OptionDefault, OptionKind, OptionSchema, OptionSpec, Precision,
    PushReadAt, ROW_INDEX_COLUMN, ReadAt, RowSelection, ScalarValue, ScanRequest, ScanUnit,
    ScanUnitMetrics, Splittable, Step, TriState, UnitPlan, WrittenFile, prune,
};

const MAGIC: &[u8; 4] = b"KMCK";
const FOOTER_MAGIC: &[u8; 4] = b"FTR1";

static OPTIONS: OptionSchema = OptionSchema {
    format: "mock",
    options: &[
        OptionSpec {
            name: "batchRows",
            kind: OptionKind::Int {
                min: 1,
                max: 1 << 30,
            },
            default: OptionDefault::Int(1000),
            conf_fallback: None,
            metadata_key: false,
            doc: "Rows per unit when writing.",
        },
        OptionSpec {
            name: "statsTruncateLength",
            kind: OptionKind::Int {
                min: 0,
                max: 1 << 20,
            },
            default: OptionDefault::Int(0),
            conf_fallback: None,
            metadata_key: false,
            doc: "Truncate string statistics to this many bytes (0: never).",
        },
        OptionSpec {
            name: "legacyStats",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(false),
            conf_fallback: None,
            metadata_key: false,
            doc: "Mark every statistic inexact.",
        },
        OptionSpec {
            name: "pageChecksumVerification",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(true),
            conf_fallback: Some("spark.sql.parquet.pageChecksumVerification"),
            metadata_key: false,
            doc: "Verify chunk checksums.",
        },
        OptionSpec {
            name: "asyncDecoder",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(false),
            conf_fallback: None,
            metadata_key: false,
            doc: "Decode through the async read-at bridge.",
        },
        OptionSpec {
            name: "selfDescribing",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(true),
            conf_fallback: None,
            metadata_key: false,
            doc: "Report files as self-describing (false: read like CSV, with the table schema).",
        },
        OptionSpec {
            name: "encryptionKey",
            kind: OptionKind::Secret,
            default: OptionDefault::None,
            conf_fallback: None,
            metadata_key: true,
            doc: "Footer key.",
        },
    ],
};

/// The mock format (see the module docs).
#[derive(Debug, Default, Clone, Copy)]
pub struct MockFormat;

#[derive(Debug, Clone)]
struct Chunk {
    range: Range<u64>,
}

#[derive(Debug, Clone)]
struct Unit {
    first_row: u64,
    rows: u64,
    chunks: Vec<Chunk>,
    stats: FileStatistics,
}

/// Decoded footer.
#[derive(Debug)]
pub struct MockMetadata {
    schema: SchemaRef,
    units: Vec<Unit>,
    stats: FileStatistics,
    bytes: usize,
}

impl FileMetadata for MockMetadata {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
    fn statistics(&self) -> FileStatistics {
        self.stats.clone()
    }
    fn heap_size(&self) -> usize {
        self.bytes
    }
}

fn corrupt(msg: impl Into<String>) -> FormatError {
    FormatError::Corrupt(msg.into())
}
fn bad_footer(msg: impl Into<String>) -> FormatError {
    FormatError::CorruptFooter(msg.into())
}

/// CRC-32 (IEEE), table-driven.
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0_u32; 256];
        for (i, slot) in (0_u32..).zip(t.iter_mut()) {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 == 1 {
                    (c >> 1) ^ 0xEDB8_8320
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        t
    });
    !data.iter().fold(0xFFFF_FFFF_u32, |crc, &b| {
        table[usize::from((crc ^ u32::from(b)).to_le_bytes()[0])] ^ (crc >> 8)
    })
}

fn xor(data: &mut [u8], key: &[u8]) {
    if key.is_empty() {
        return;
    }
    for (i, b) in data.iter_mut().enumerate() {
        *b ^= key[i % key.len()];
    }
}

// ---- footer encoding -------------------------------------------------------

struct Out(Vec<u8>);
impl Out {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.0.extend_from_slice(b);
    }
    fn scalar(&mut self, p: &Precision<ScalarValue>) {
        let (exact, v) = match p {
            Precision::Exact(v) => (1, v),
            Precision::Inexact(v) => (2, v),
            Precision::Absent => return self.u8(0),
        };
        self.u8(exact);
        match v {
            ScalarValue::Boolean(b) => {
                self.u8(1);
                self.u8(u8::from(*b));
            }
            ScalarValue::Int8(x) => self.num(2, i128::from(*x)),
            ScalarValue::Int16(x) => self.num(3, i128::from(*x)),
            ScalarValue::Int32(x) => self.num(4, i128::from(*x)),
            ScalarValue::Int64(x) => self.num(5, i128::from(*x)),
            ScalarValue::Float32(x) => {
                self.u8(6);
                self.u64(u64::from(x.to_bits()));
            }
            ScalarValue::Float64(x) => {
                self.u8(7);
                self.u64(x.to_bits());
            }
            ScalarValue::Decimal128 {
                unscaled,
                precision,
                scale,
            } => {
                self.u8(8);
                self.0.extend_from_slice(&unscaled.to_le_bytes());
                self.u8(*precision);
                self.0.extend_from_slice(&scale.to_le_bytes());
            }
            ScalarValue::Utf8(s) => {
                self.u8(9);
                self.bytes(s.as_bytes());
            }
            ScalarValue::Binary(b) => {
                self.u8(10);
                self.bytes(b);
            }
            ScalarValue::Date32(d) => self.num(11, i128::from(*d)),
            ScalarValue::TimestampMicros(t) => self.num(12, i128::from(*t)),
            _ => self.0.push(0),
        }
    }
    fn num(&mut self, tag: u8, v: i128) {
        self.u8(tag);
        self.0.extend_from_slice(&v.to_le_bytes());
    }
}

struct In<'a>(&'a [u8]);
impl In<'_> {
    fn take(&mut self, n: usize) -> FormatResult<&[u8]> {
        if n > self.0.len() {
            return Err(bad_footer("footer truncated"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u8(&mut self) -> FormatResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u64(&mut self) -> FormatResult<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(
            b.try_into().map_err(|_| bad_footer("bad u64"))?,
        ))
    }
    fn i128(&mut self) -> FormatResult<i128> {
        let b = self.take(16)?;
        Ok(i128::from_le_bytes(
            b.try_into().map_err(|_| bad_footer("bad i128"))?,
        ))
    }
    fn bytes(&mut self) -> FormatResult<Vec<u8>> {
        let n = usize::try_from(self.u64()?).map_err(|_| bad_footer("length overflow"))?;
        Ok(self.take(n)?.to_vec())
    }
    fn scalar(&mut self) -> FormatResult<Precision<ScalarValue>> {
        let exact = self.u8()?;
        if exact == 0 {
            return Ok(Precision::Absent);
        }
        let int = |v: i128| v;
        let v = match self.u8()? {
            1 => ScalarValue::Boolean(self.u8()? != 0),
            2 => {
                ScalarValue::Int8(i8::try_from(int(self.i128()?)).map_err(|_| bad_footer("int8"))?)
            }
            3 => ScalarValue::Int16(i16::try_from(self.i128()?).map_err(|_| bad_footer("int16"))?),
            4 => ScalarValue::Int32(i32::try_from(self.i128()?).map_err(|_| bad_footer("int32"))?),
            5 => ScalarValue::Int64(i64::try_from(self.i128()?).map_err(|_| bad_footer("int64"))?),
            6 => ScalarValue::Float32(f32::from_bits(
                u32::try_from(self.u64()?).map_err(|_| bad_footer("f32"))?,
            )),
            7 => ScalarValue::Float64(f64::from_bits(self.u64()?)),
            8 => {
                let unscaled = self.i128()?;
                let precision = self.u8()?;
                let scale = i8::from_le_bytes([self.u8()?]);
                ScalarValue::Decimal128 {
                    unscaled,
                    precision,
                    scale,
                }
            }
            9 => ScalarValue::Utf8(
                String::from_utf8(self.bytes()?).map_err(|_| bad_footer("utf8 stat"))?,
            ),
            10 => ScalarValue::Binary(self.bytes()?),
            11 => ScalarValue::Date32(i32::try_from(self.i128()?).map_err(|_| bad_footer("date"))?),
            12 => ScalarValue::TimestampMicros(
                i64::try_from(self.i128()?).map_err(|_| bad_footer("ts"))?,
            ),
            _ => return Err(bad_footer("unknown statistic tag")),
        };
        Ok(if exact == 1 {
            Precision::Exact(v)
        } else {
            Precision::Inexact(v)
        })
    }
}

fn schema_bytes(schema: &Schema) -> FormatResult<Vec<u8>> {
    let mut w = StreamWriter::try_new(Vec::new(), schema).map_err(|e| corrupt(e.to_string()))?;
    w.finish().map_err(|e| corrupt(e.to_string()))?;
    w.into_inner().map_err(|e| corrupt(e.to_string()))
}

fn encode_footer(schema: &Schema, units: &[Unit], legacy: bool) -> FormatResult<Vec<u8>> {
    let mut o = Out(FOOTER_MAGIC.to_vec());
    o.bytes(&schema_bytes(schema)?);
    o.u8(u8::from(legacy));
    o.u64(units.len() as u64);
    for u in units {
        o.u64(u.first_row);
        o.u64(u.rows);
        o.u64(u.chunks.len() as u64);
        for (c, s) in u.chunks.iter().zip(&u.stats.columns) {
            o.u64(c.range.start);
            o.u64(c.range.end);
            match s.null_count {
                Precision::Exact(n) => {
                    o.u8(1);
                    o.u64(n);
                }
                _ => o.u8(0),
            }
            o.scalar(&s.min);
            o.scalar(&s.max);
        }
    }
    Ok(o.0)
}

fn decode_footer(data: &[u8], body_end: u64) -> FormatResult<MockMetadata> {
    let mut i = In(data);
    if i.take(4)? != FOOTER_MAGIC {
        return Err(bad_footer("bad footer magic (wrong key?)"));
    }
    let schema_raw = i.bytes()?;
    let schema = StreamReader::try_new(Cursor::new(schema_raw), None)
        .map_err(|e| bad_footer(e.to_string()))?
        .schema();
    let _legacy = i.u8()?;
    let n = i.u64()?;
    if n > body_end {
        return Err(bad_footer("unit count out of range"));
    }
    let mut units = Vec::new();
    for _ in 0..n {
        let first_row = i.u64()?;
        let rows = i.u64()?;
        let cols = i.u64()?;
        if cols != schema.fields().len() as u64 {
            return Err(bad_footer("column count mismatch"));
        }
        let mut chunks = Vec::new();
        let mut columns = Vec::new();
        for f in schema.fields() {
            let start = i.u64()?;
            let end = i.u64()?;
            if start > end || end > body_end {
                return Err(bad_footer("chunk outside the file"));
            }
            let null_count = if i.u8()? == 1 {
                Precision::Exact(i.u64()?)
            } else {
                Precision::Absent
            };
            let min = i.scalar()?;
            let max = i.scalar()?;
            chunks.push(Chunk { range: start..end });
            columns.push(col_stats(ColumnPath::new(f.name()), null_count, min, max));
        }
        units.push(Unit {
            first_row,
            rows,
            chunks,
            stats: file_stats(rows, columns),
        });
    }
    let stats = merge_stats(&schema, &units);
    Ok(MockMetadata {
        schema,
        units,
        stats,
        bytes: data.len() * 2 + 256,
    })
}

/// Column statistics without a NaN count: NaN rows may hide behind the
/// bounds, so pruning stays conservative for floats.
fn col_stats(
    path: ColumnPath,
    null_count: Precision<u64>,
    min: Precision<ScalarValue>,
    max: Precision<ScalarValue>,
) -> ColumnStatistics {
    let mut c = ColumnStatistics::new(path);
    c.null_count = null_count;
    c.min = min;
    c.max = max;
    c
}

fn file_stats(rows: u64, columns: Vec<ColumnStatistics>) -> FileStatistics {
    let mut f = FileStatistics::with_rows(Precision::Exact(rows));
    f.columns = columns;
    f
}

fn merge_stats(schema: &Schema, units: &[Unit]) -> FileStatistics {
    let rows: u64 = units.iter().map(|u| u.rows).sum();
    let columns = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(c, f)| {
            let mut nulls = Some(0_u64);
            let mut min: Option<Precision<ScalarValue>> = None;
            let mut max: Option<Precision<ScalarValue>> = None;
            for u in units {
                let s = &u.stats.columns[c];
                nulls = nulls.zip(s.null_count.exact().copied()).map(|(a, b)| a + b);
                min = Some(pick(min, &s.min, std::cmp::Ordering::Less));
                max = Some(pick(max, &s.max, std::cmp::Ordering::Greater));
            }
            col_stats(
                ColumnPath::new(f.name()),
                nulls.map_or(Precision::Absent, Precision::Exact),
                min.unwrap_or_default(),
                max.unwrap_or_default(),
            )
        })
        .collect();
    file_stats(rows, columns)
}

fn pick(
    acc: Option<Precision<ScalarValue>>,
    next: &Precision<ScalarValue>,
    want: std::cmp::Ordering,
) -> Precision<ScalarValue> {
    let Some(acc) = acc else { return next.clone() };
    let exact = matches!(acc, Precision::Exact(_)) && matches!(next, Precision::Exact(_));
    match (acc.value(), next.value()) {
        (Some(a), Some(b)) => {
            let v = if cmp_scalar(b, a) == Some(want) {
                b.clone()
            } else {
                a.clone()
            };
            if exact {
                Precision::Exact(v)
            } else {
                Precision::Inexact(v)
            }
        }
        // A unit without a bound (all null) does not widen the file's.
        (Some(_), None) if matches!(next, Precision::Absent) => acc,
        (None, Some(_)) if matches!(acc, Precision::Absent) => next.clone(),
        _ => Precision::Absent,
    }
}

fn cmp_scalar(a: &ScalarValue, b: &ScalarValue) -> Option<std::cmp::Ordering> {
    use ScalarValue as S;
    match (a, b) {
        (S::Boolean(x), S::Boolean(y)) => Some(x.cmp(y)),
        (S::Int8(x), S::Int8(y)) => Some(x.cmp(y)),
        (S::Int16(x), S::Int16(y)) => Some(x.cmp(y)),
        (S::Int32(x), S::Int32(y)) | (S::Date32(x), S::Date32(y)) => Some(x.cmp(y)),
        (S::Int64(x), S::Int64(y)) | (S::TimestampMicros(x), S::TimestampMicros(y)) => {
            Some(x.cmp(y))
        }
        (S::Float32(x), S::Float32(y)) => Some(x.total_cmp(y)),
        (S::Float64(x), S::Float64(y)) => Some(x.total_cmp(y)),
        (S::Decimal128 { unscaled: x, .. }, S::Decimal128 { unscaled: y, .. }) => Some(x.cmp(y)),
        (S::Utf8(x), S::Utf8(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        (S::Binary(x), S::Binary(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

// ---- statistics of written chunks -----------------------------------------

/// Min and max of a column chunk, the way the Parquet specification has
/// writers record them: floats by IEEE total order (so -0.0 can be the min)
/// with NaN left out of both bounds and no NaN count, so pruning must allow
/// for NaN rows the bounds do not show.
fn column_stats(path: ColumnPath, a: &ArrayRef, truncate: usize, legacy: bool) -> ColumnStatistics {
    let mut s = ColumnStatistics::new(path);
    s.null_count = Precision::Exact(a.null_count() as u64);
    macro_rules! prim {
        ($t:ty, $v:ident) => {
            a.as_primitive::<$t>()
                .iter()
                .flatten()
                .map(ScalarValue::$v)
                .collect()
        };
    }
    let vals: Vec<ScalarValue> = match a.data_type() {
        DataType::Boolean => a
            .as_boolean()
            .iter()
            .flatten()
            .map(ScalarValue::Boolean)
            .collect(),
        DataType::Int8 => prim!(Int8Type, Int8),
        DataType::Int16 => prim!(Int16Type, Int16),
        DataType::Int32 => prim!(Int32Type, Int32),
        DataType::Int64 => prim!(Int64Type, Int64),
        DataType::Float32 => a
            .as_primitive::<Float32Type>()
            .iter()
            .flatten()
            .filter(|v| !v.is_nan())
            .map(ScalarValue::Float32)
            .collect(),
        DataType::Float64 => a
            .as_primitive::<Float64Type>()
            .iter()
            .flatten()
            .filter(|v| !v.is_nan())
            .map(ScalarValue::Float64)
            .collect(),
        DataType::Date32 => prim!(Date32Type, Date32),
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            prim!(TimestampMicrosecondType, TimestampMicros)
        }
        DataType::Decimal128(p, sc) => a
            .as_primitive::<Decimal128Type>()
            .iter()
            .flatten()
            .map(|u| ScalarValue::Decimal128 {
                unscaled: u,
                precision: *p,
                scale: *sc,
            })
            .collect(),
        DataType::Utf8View => a
            .as_string_view()
            .iter()
            .flatten()
            .map(|v| ScalarValue::Utf8(v.to_owned()))
            .collect(),
        DataType::Utf8 => a
            .as_string::<i32>()
            .iter()
            .flatten()
            .map(|v| ScalarValue::Utf8(v.to_owned()))
            .collect(),
        DataType::BinaryView => a
            .as_binary_view()
            .iter()
            .flatten()
            .map(|v| ScalarValue::Binary(v.to_vec()))
            .collect(),
        _ => return s,
    };
    let min = vals
        .iter()
        .min_by(|x, y| cmp_scalar(x, y).unwrap_or(std::cmp::Ordering::Equal))
        .cloned();
    let max = vals
        .iter()
        .max_by(|x, y| cmp_scalar(x, y).unwrap_or(std::cmp::Ordering::Equal))
        .cloned();
    let (mut min, mut max) = (
        min.map_or(Precision::Absent, Precision::Exact),
        max.map_or(Precision::Absent, Precision::Exact),
    );
    if truncate > 0 {
        if let Precision::Exact(ScalarValue::Utf8(v)) = &min
            && v.len() > truncate
        {
            let cut = floor_char(v, truncate);
            min = Precision::Inexact(ScalarValue::Utf8(v[..cut].to_owned()));
        }
        if let Precision::Exact(ScalarValue::Utf8(v)) = &max
            && v.len() > truncate
        {
            let cut = floor_char(v, truncate);
            // An upper bound: the prefix with its last byte raised.
            let mut b = v.as_bytes()[..cut].to_vec();
            if let Some(last) = b.last_mut() {
                *last = last.saturating_add(1).min(0x7F);
            }
            max = Precision::Inexact(ScalarValue::Utf8(String::from_utf8_lossy(&b).into_owned()));
        }
    }
    if legacy {
        min = min.to_inexact();
        max = max.to_inexact();
    }
    s.min = min;
    s.max = max;
    s
}

fn floor_char(s: &str, n: usize) -> usize {
    (0..=n).rev().find(|i| s.is_char_boundary(*i)).unwrap_or(0)
}

// ---- encoder ---------------------------------------------------------------

struct MockEncoder {
    schema: SchemaRef,
    batch_rows: usize,
    truncate: usize,
    legacy: bool,
    key: Option<SecretString>,
    pending: Vec<RecordBatch>,
    pending_rows: usize,
    out: VecDeque<Bytes>,
    out_bytes: usize,
    offset: u64,
    rows: u64,
    units: Vec<Unit>,
}

impl MockEncoder {
    fn emit(&mut self, b: Vec<u8>) {
        self.offset += b.len() as u64;
        self.out_bytes += b.len();
        self.out.push_back(Bytes::from(b));
    }

    fn encode_unit(&mut self) -> FormatResult<()> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let batch = arrow_select_concat(&self.schema, &self.pending)?;
        self.pending.clear();
        self.pending_rows = 0;
        let mut chunks = Vec::new();
        let mut columns = Vec::new();
        let schema = Arc::clone(&self.schema);
        for (i, f) in schema.fields().iter().enumerate() {
            let col = batch.column(i);
            let one = Arc::new(Schema::new(vec![f.as_ref().clone()]));
            let rb = RecordBatch::try_new(Arc::clone(&one), vec![Arc::clone(col)])
                .map_err(|e| corrupt(e.to_string()))?;
            let mut w =
                StreamWriter::try_new(Vec::new(), &one).map_err(|e| corrupt(e.to_string()))?;
            w.write(&rb).map_err(|e| corrupt(e.to_string()))?;
            w.finish().map_err(|e| corrupt(e.to_string()))?;
            let mut data = w.into_inner().map_err(|e| corrupt(e.to_string()))?;
            let crc = crc32(&data);
            data.extend_from_slice(&crc.to_le_bytes());
            let start = self.offset;
            self.emit(data);
            chunks.push(Chunk {
                range: start..self.offset,
            });
            columns.push(column_stats(
                ColumnPath::new(f.name()),
                col,
                self.truncate,
                self.legacy,
            ));
        }
        let rows = batch.num_rows() as u64;
        self.units.push(Unit {
            first_row: self.rows,
            rows,
            chunks,
            stats: file_stats(rows, columns),
        });
        self.rows += rows;
        Ok(())
    }
}

fn arrow_select_concat(schema: &SchemaRef, batches: &[RecordBatch]) -> FormatResult<RecordBatch> {
    arrow_select::concat::concat_batches(schema, batches).map_err(|e| corrupt(e.to_string()))
}

impl Encoder for MockEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        if batch.schema().fields() != self.schema.fields() {
            return Err(corrupt("batch schema differs from the writer's"));
        }
        let mut off = 0;
        while off < batch.num_rows() {
            let n = (self.batch_rows - self.pending_rows).min(batch.num_rows() - off);
            self.pending.push(batch.slice(off, n));
            self.pending_rows += n;
            off += n;
            if self.pending_rows >= self.batch_rows {
                self.encode_unit()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> FormatResult<()> {
        self.encode_unit()
    }

    fn take_bytes(&mut self) -> Option<Bytes> {
        let b = self.out.pop_front()?;
        self.out_bytes -= b.len();
        Some(b)
    }

    fn finish(mut self: Box<Self>) -> FormatResult<WrittenFile> {
        self.encode_unit()?;
        let mut footer = encode_footer(&self.schema, &self.units, self.legacy)?;
        if let Some(k) = &self.key {
            xor(&mut footer, k.expose_secret().as_bytes());
        }
        let len = u32::try_from(footer.len()).map_err(|_| corrupt("footer too large"))?;
        let mut tail = Vec::new();
        while let Some(b) = self.take_bytes() {
            tail.extend_from_slice(&b);
        }
        tail.extend_from_slice(&footer);
        tail.extend_from_slice(&len.to_le_bytes());
        tail.extend_from_slice(MAGIC);
        let statistics = merge_stats(&self.schema, &self.units);
        Ok(WrittenFile::new(Bytes::from(tail), statistics))
    }

    fn buffered_bytes(&self) -> usize {
        self.pending
            .iter()
            .map(RecordBatch::get_array_memory_size)
            .sum::<usize>()
            + self.out_bytes
    }

    fn written_bytes(&self) -> u64 {
        self.offset
    }
}

// ---- decoder ---------------------------------------------------------------

/// Decodes one chunk (verifying its CRC) and returns its column.
fn decode_chunk(data: &[u8], verify: bool) -> FormatResult<ArrayRef> {
    if data.len() < 4 {
        return Err(corrupt("chunk truncated"));
    }
    let (body, crc) = data.split_at(data.len() - 4);
    if verify {
        let want = u32::from_le_bytes(crc.try_into().map_err(|_| corrupt("crc"))?);
        if crc32(body) != want {
            return Err(corrupt("chunk checksum mismatch"));
        }
    }
    let mut r =
        StreamReader::try_new(Cursor::new(body), None).map_err(|e| corrupt(e.to_string()))?;
    let batch = r
        .next()
        .ok_or_else(|| corrupt("empty chunk"))?
        .map_err(|e| corrupt(e.to_string()))?;
    Ok(Arc::clone(batch.column(0)))
}

/// `a` pruned to the nested type `want` (struct children by name).
fn project_array(a: &ArrayRef, want: &DataType) -> FormatResult<ArrayRef> {
    match (a.data_type(), want) {
        (DataType::Struct(_), DataType::Struct(wf)) if a.data_type() != want => {
            let sa = a.as_struct();
            let mut cols = Vec::new();
            for f in wf {
                let c = sa
                    .column_by_name(f.name())
                    .ok_or_else(|| corrupt(format!("missing struct field {}", f.name())))?;
                cols.push(project_array(c, f.data_type())?);
            }
            let fields: Fields = wf.clone();
            Ok(Arc::new(
                StructArray::try_new_with_length(fields, cols, sa.nulls().cloned(), sa.len())
                    .map_err(|e| corrupt(e.to_string()))?,
            ))
        }
        _ => Ok(Arc::clone(a)),
    }
}

/// What a decoder needs to read: the plan and request, shared with the async variant.
struct Plan {
    meta: Arc<MockMetadata>,
    units: VecDeque<ScanUnit>,
    out_schema: SchemaRef,
    /// File column index per output column (before the row index).
    cols: Vec<usize>,
    selection: Option<RowSelection>,
    row_index: bool,
    verify: bool,
    dynamic: Vec<krishiv_io::ScanFilter>,
}

impl Plan {
    fn new(
        meta: Arc<MockMetadata>,
        units: UnitPlan,
        req: &ScanRequest,
        verify: bool,
    ) -> FormatResult<Self> {
        let out_schema = req.output_schema()?;
        let schema = meta.schema();
        let mut cols = Vec::new();
        for f in out_schema.fields() {
            if f.name() == ROW_INDEX_COLUMN && req.metadata_cols.row_index {
                continue;
            }
            cols.push(
                schema
                    .index_of(f.name())
                    .map_err(|e| corrupt(e.to_string()))?,
            );
        }
        Ok(Self {
            meta,
            units: units.units.into(),
            out_schema,
            cols,
            selection: req.selection.clone(),
            row_index: req.metadata_cols.row_index,
            verify,
            dynamic: req.dynamic.clone(),
        })
    }

    /// The next unit not pruned by a runtime filter (re-read per unit).
    fn next_unit(&mut self) -> Option<ScanUnit> {
        while let Some(u) = self.units.pop_front() {
            let stats = &self.meta.units[u.index].stats;
            let pruned = self
                .dynamic
                .iter()
                .filter_map(krishiv_io::ScanFilter::predicate)
                .any(|e| prune(&e, stats) == TriState::False);
            if !pruned {
                return Some(u);
            }
        }
        None
    }

    fn ranges(&self, u: &ScanUnit) -> Vec<Range<u64>> {
        let mut r: Vec<Range<u64>> = self
            .cols
            .iter()
            .map(|&c| self.meta.units[u.index].chunks[c].range.clone())
            .collect();
        r.sort_by_key(|x| x.start);
        r.dedup();
        r
    }

    /// The unit's batch from its chunks' bytes.
    fn build(
        &self,
        u: &ScanUnit,
        chunk: impl Fn(&Range<u64>) -> Option<Bytes>,
    ) -> FormatResult<Option<RecordBatch>> {
        let mut columns = Vec::new();
        for (i, &c) in self.cols.iter().enumerate() {
            let range = &self.meta.units[u.index].chunks[c].range;
            let data = chunk(range).ok_or_else(|| corrupt("chunk not pushed"))?;
            let a = decode_chunk(&data, self.verify)?;
            if a.len() as u64 != u.rows {
                return Err(corrupt("chunk row count differs from the footer"));
            }
            columns.push(project_array(&a, self.out_schema.field(i).data_type())?);
        }
        let window = u.first_row..u.first_row + u.rows;
        let keep: Vec<Range<u64>> = match &self.selection {
            None => std::iter::once(0..u.rows).collect(),
            Some(s) => s.within(window),
        };
        let selected: u64 = keep.iter().map(|r| r.end - r.start).sum();
        if selected == 0 {
            return Ok(None);
        }
        let mask = (selected != u.rows).then(|| {
            let mut m = vec![false; usize::try_from(u.rows).unwrap_or(0)];
            for r in &keep {
                for i in r.clone() {
                    if let Some(x) = usize::try_from(i).ok().and_then(|i| m.get_mut(i)) {
                        *x = true;
                    }
                }
            }
            BooleanArray::from(m)
        });
        if let Some(mask) = &mask {
            columns = columns
                .iter()
                .map(|c| arrow_select::filter::filter(c, mask).map_err(|e| corrupt(e.to_string())))
                .collect::<FormatResult<_>>()?;
        }
        if self.row_index {
            let idx: Int64Array = keep
                .iter()
                .flat_map(Clone::clone)
                .map(|i| i64::try_from(u.first_row + i).unwrap_or(i64::MAX))
                .collect();
            columns.push(Arc::new(idx));
        }
        let rows = usize::try_from(selected).map_err(|_| corrupt("rows overflow"))?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.out_schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map(Some)
        .map_err(|e| corrupt(e.to_string()))
    }
}

struct MockDecoder {
    plan: Plan,
    current: Option<ScanUnit>,
    data: Vec<(Range<u64>, Bytes)>,
    metrics: ScanUnitMetrics,
    done: bool,
}

impl Decoder for MockDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        loop {
            if self.done {
                return Ok(Step::Done);
            }
            if self.current.is_none() {
                match self.plan.next_unit() {
                    Some(u) => self.current = Some(u),
                    None => {
                        self.done = true;
                        self.data.clear();
                        continue;
                    }
                }
            }
            let Some(u) = self.current.clone() else {
                continue;
            };
            let missing: Vec<Range<u64>> = self
                .plan
                .ranges(&u)
                .into_iter()
                .filter(|r| !self.data.iter().any(|(d, _)| d == r))
                .collect();
            if !missing.is_empty() {
                return Ok(Step::NeedData(missing));
            }
            let data = std::mem::take(&mut self.data);
            let out = self.plan.build(&u, |r| {
                data.iter().find(|(d, _)| d == r).map(|(_, b)| b.clone())
            })?;
            self.current = None;
            if let Some(b) = out {
                self.metrics.rows += b.num_rows() as u64;
                return Ok(Step::Batch(b));
            }
        }
    }

    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()> {
        for (r, b) in data {
            if b.len() as u64 != r.end - r.start {
                return Err(corrupt("pushed bytes do not match the range"));
            }
            self.metrics.bytes_pushed += b.len() as u64;
            self.data.push((r, b));
        }
        Ok(())
    }

    fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        let mut out = Vec::new();
        let mut total = 0;
        for u in self.plan.units.iter().take(2) {
            for r in self.plan.ranges(u) {
                total += r.end - r.start;
                if total > max_bytes {
                    return out;
                }
                out.push(r);
            }
        }
        out
    }

    fn buffered_bytes(&self) -> usize {
        self.data.iter().map(|(_, b)| b.len()).sum()
    }

    fn metrics(&self) -> ScanUnitMetrics {
        self.metrics
    }
}

/// The async variant: a stream over a [`PushReadAt`], driven by [`AsyncDecoder`].
fn async_decoder(plan: Plan, size: u64) -> AsyncDecoder {
    let reader = PushReadAt::new(size);
    let r = Arc::clone(&reader);
    let stream = futures::stream::unfold((plan, r), |(mut plan, r)| async move {
        loop {
            let u = plan.next_unit()?;
            let mut got = Vec::new();
            for range in plan.ranges(&u) {
                match r.read_at(range.clone()).await {
                    Ok(b) => got.push((range, b)),
                    Err(e) => return Some((Err(e), (plan, r))),
                }
            }
            match plan.build(&u, |x| {
                got.iter().find(|(d, _)| d == x).map(|(_, b)| b.clone())
            }) {
                Ok(Some(b)) => return Some((Ok(b), (plan, r))),
                Ok(None) => {}
                Err(e) => return Some((Err(e), (plan, r))),
            }
        }
    });
    AsyncDecoder::new(Box::pin(stream), reader)
}

fn downcast(m: Arc<dyn FileMetadata>) -> FormatResult<Arc<MockMetadata>> {
    let any: Arc<dyn std::any::Any + Send + Sync> = m;
    any.downcast::<MockMetadata>().map_err(|_| {
        FormatError::Other(
            krishiv_common::error::InternalError {
                message: "not mock metadata".into(),
            }
            .into(),
        )
    })
}

impl FileFormat for MockFormat {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn options(&self) -> &'static OptionSchema {
        &OPTIONS
    }

    fn capabilities(&self, o: &FormatOptions) -> Capabilities {
        {
            let mut c = Capabilities::new(Splittable::Yes);
            c.self_describing = o.bool("selfDescribing").unwrap_or(true);
            c.statistics = true;
            c.row_index = true;
            c.selection = true;
            c.page_checksums = true;
            c.write = true;
            c
        }
    }

    fn file_suffix(&self, _o: &FormatOptions) -> String {
        ".mock".to_owned()
    }

    fn read_metadata<'a>(
        &'a self,
        io: &'a IoContext,
        file: &'a FileRef,
        _needs: MetadataNeeds,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Arc<dyn FileMetadata>>> {
        Box::pin(async move {
            if file.size < 12 {
                return Err(bad_footer("file too small"));
            }
            let held = io.get_tail(file, 0).await?;
            let (start, tail) = (held.start, &held.bytes);
            let n = tail.len();
            if n < 8 || &tail[n - 4..] != MAGIC {
                return Err(bad_footer("bad magic"));
            }
            let len = u64::from(u32::from_le_bytes(
                tail[n - 8..n - 4]
                    .try_into()
                    .map_err(|_| bad_footer("len"))?,
            ));
            let footer_start = file
                .size
                .checked_sub(8 + len)
                .filter(|s| *s >= 4)
                .ok_or_else(|| bad_footer("footer length out of range"))?;
            let mut footer = if footer_start >= start {
                let a = usize::try_from(footer_start - start).map_err(|_| bad_footer("offset"))?;
                tail[a..n - 8].to_vec()
            } else {
                // A footer larger than the speculative read: one more exact read.
                let footer_range = footer_start..file.size - 8;
                let v = io
                    .get_ranges_held(file, std::slice::from_ref(&footer_range))
                    .await?;
                v.first().map(|b| b.to_vec()).unwrap_or_default()
            };
            if let Some(key) = o.resolve_secret("encryptionKey", io.secrets()).await? {
                xor(&mut footer, key.expose_secret().as_bytes());
            }
            let meta = decode_footer(&footer, footer_start)?;
            Ok(Arc::new(meta) as Arc<dyn FileMetadata>)
        })
    }

    fn plan_units(
        &self,
        m: &dyn FileMetadata,
        split: &ByteRange,
        req: &ScanRequest,
    ) -> FormatResult<UnitPlan> {
        let any: &dyn std::any::Any = m;
        let meta = any
            .downcast_ref::<MockMetadata>()
            .ok_or_else(|| corrupt("not mock metadata"))?;
        let mut plan = UnitPlan::default();
        for (i, u) in meta.units.iter().enumerate() {
            let start = u.chunks.first().map_or(0, |c| c.range.start);
            let end = u.chunks.last().map_or(0, |c| c.range.end);
            if !split.owns(&(start..end)) {
                continue;
            }
            plan.owned += 1;
            if let Some(p) = &req.predicate
                && prune(p, &u.stats) == TriState::False
            {
                plan.pruned_by_statistics += 1;
                continue;
            }
            plan.units.push(ScanUnit {
                index: i,
                first_row: u.first_row,
                rows: u.rows,
                bytes: start..end,
            });
        }
        Ok(plan)
    }

    fn open<'a>(
        &'a self,
        _io: &'a IoContext,
        m: Arc<dyn FileMetadata>,
        units: UnitPlan,
        req: &'a ScanRequest,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Decoder>>> {
        Box::pin(async move {
            let meta = downcast(m)?;
            let size = meta
                .units
                .last()
                .and_then(|u| u.chunks.last())
                .map_or(0, |c| c.range.end);
            let n = units.units.len() as u64;
            let plan = Plan::new(
                meta,
                units,
                req,
                o.bool("pageChecksumVerification").unwrap_or(true),
            )?;
            if o.bool("asyncDecoder") == Some(true) {
                return Ok(Box::new(async_decoder(plan, size)) as Box<dyn Decoder>);
            }
            Ok(Box::new(MockDecoder {
                plan,
                current: None,
                data: Vec::new(),
                metrics: {
                    let mut m = ScanUnitMetrics::default();
                    m.units = n;
                    m
                },
                done: false,
            }) as Box<dyn Decoder>)
        })
    }

    fn writer<'a>(
        &'a self,
        io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Encoder>>> {
        Box::pin(async move {
            let key = o.resolve_secret("encryptionKey", io.secrets()).await?;
            let mut e = MockEncoder {
                schema,
                batch_rows: usize::try_from(o.int("batchRows").unwrap_or(1000)).unwrap_or(1000),
                truncate: usize::try_from(o.int("statsTruncateLength").unwrap_or(0)).unwrap_or(0),
                legacy: o.bool("legacyStats").unwrap_or(false),
                key,
                pending: Vec::new(),
                pending_rows: 0,
                out: VecDeque::new(),
                out_bytes: 0,
                offset: 0,
                rows: 0,
                units: Vec::new(),
            };
            e.emit(MAGIC.to_vec());
            Ok(Box::new(e) as Box<dyn Encoder>)
        })
    }
}
