// SPDX-License-Identifier: Apache-2.0
//! Reading, checking and describing a Parquet footer.
//!
//! - **One speculative tail read** ([`IoContext::get_tail`]) feeds arrow-rs's
//!   sans-IO `ParquetMetaDataPushDecoder`; a footer larger than the tail
//!   costs exactly one more read of the missing bytes.
//! - **Checked before use.** arrow-rs trusts some footer fields (column
//!   chunk offsets and sizes are asserted non-negative when a chunk is
//!   read). Every row group and column chunk is checked here against the
//!   file size, so a corrupt footer is a `CANNOT_READ_FILE_FOOTER` error,
//!   never a panic later.
//! - **The file schema in engine types**: strings as `Utf8View`, binary as
//!   `BinaryView`, decimals as `Decimal128`, `INT96` as `timestamp`
//!   (`int96AsTimestamp`), unsigned integers widened and
//!   millisecond timestamps in microseconds, as Spark reads them
//!   (`UINT_8`→`smallint`, `UINT_16`→`int`, `UINT_32`→`bigint`,
//!   `UINT_64`→`decimal(20,0)`); field ids and extension metadata kept.
//! - **Statistics per row group** in the column's own (engine) type, exact
//!   only where they are exact: integer, float, boolean and date bounds;
//!   string, binary and decimal-in-bytes bounds only when the file marks them
//!   exact and they are not legacy (deprecated `min`/`max` fields, which old
//!   writers ordered by signed bytes); never for `INT96` or unsigned types.

use std::ops::Range;
use std::sync::Arc;

use super::thrift_guard;
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef, TimeUnit};
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataPushDecoder};
use parquet::file::statistics::Statistics;
use parquet::schema::types::ColumnDescriptor;

use crate::error::{FormatError, FormatResult};
use crate::format::FileMetadata;
use crate::io::{FileRef, IoContext};
use crate::request::{ColumnPath, ScalarValue};
use crate::stats::{ColumnStatistics, FileStatistics, Precision};

/// Footer magic of plaintext and encrypted-footer files.
const MAGIC: &[u8; 4] = b"PAR1";
const ENCRYPTED_MAGIC: &[u8; 4] = b"PARE";
/// Footer reads needed at most: the tail, then the rest of a large footer.
const MAX_FOOTER_READS: usize = 4;

/// How the file schema is presented (options that change it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SchemaOptions {
    /// `binaryAsString`: unannotated `BINARY` columns are strings.
    pub(crate) binary_as_string: bool,
    /// `int96AsTimestamp`: `INT96` columns are timestamps.
    pub(crate) int96_as_timestamp: bool,
}

/// One row group as planned and pruned.
#[derive(Debug, Clone)]
pub(crate) struct RowGroup {
    pub(crate) first_row: u64,
    pub(crate) rows: u64,
    /// From the first byte of its first column chunk to the end of its last.
    pub(crate) bytes: Range<u64>,
    pub(crate) statistics: FileStatistics,
}

/// A Parquet file's metadata as the framework caches it.
#[derive(Debug)]
pub struct ParquetMetadata {
    /// The calendars its dates and timestamps are in.
    pub(crate) calendars: super::calendar::Calendars,
    /// The footer with the arrow-rs schema the decoders read with.
    pub(crate) reader: ArrowReaderMetadata,
    /// Options the reader schema was built with (a decoder needing virtual
    /// columns rebuilds it from the same footer).
    pub(crate) reader_options: ArrowReaderOptions,
    /// The file-native schema in engine types (what decoders produce).
    pub(crate) schema: SchemaRef,
    pub(crate) row_groups: Vec<RowGroup>,
    statistics: FileStatistics,
    heap: usize,
}

impl FileMetadata for ParquetMetadata {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn statistics(&self) -> FileStatistics {
        self.statistics.clone()
    }

    fn heap_size(&self) -> usize {
        self.heap
    }
}

fn footer_error(e: impl std::fmt::Display) -> FormatError {
    FormatError::CorruptFooter(e.to_string())
}

fn guard_error(why: &str) -> FormatError {
    footer_error(format!("the footer is malformed: {why}"))
}

/// Reads and checks the footer of `file`.
pub(crate) async fn read(
    io: &IoContext,
    file: &FileRef,
    options: SchemaOptions,
) -> FormatResult<ParquetMetadata> {
    if file.size < 12 {
        return Err(footer_error(format!(
            "a {}-byte file is too small for Parquet",
            file.size
        )));
    }
    // The magic and the footer length must be in the tail whatever the
    // configured speculative read size.
    let tail = io.get_tail(file, 8).await?;
    let magic = tail.bytes.get(tail.bytes.len().saturating_sub(4)..);
    if magic == Some(ENCRYPTED_MAGIC.as_slice()) {
        return Err(FormatError::Other(
            krishiv_common::error::KrishivUnsupportedFeature {
                feature: "Parquet files with encrypted footers".to_owned(),
            }
            .into(),
        ));
    }
    if magic != Some(MAGIC.as_slice()) {
        return Err(footer_error("no Parquet magic at the end of the file"));
    }
    // The footer length is untrusted: check it against the file before
    // arrow-rs computes offsets from it.
    let n = tail.bytes.len();
    let footer_len = tail
        .bytes
        .get(n.saturating_sub(8)..n.saturating_sub(4))
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .map(|b| u64::from(u32::from_le_bytes(b)))
        .ok_or_else(|| footer_error("no footer length"))?;
    if footer_len + 12 > file.size {
        return Err(footer_error(format!(
            "a {footer_len}-byte footer does not fit a {}-byte file",
            file.size
        )));
    }
    let mut decoder = ParquetMetaDataPushDecoder::try_new(file.size)
        .map_err(footer_error)?
        .with_page_index_policy(PageIndexPolicy::Skip);
    // arrow-rs trusts the element counts of the footer's Thrift lists and
    // maps, so the whole footer is walked first. The part before the tail,
    // if any, is the one read the decoder would ask for.
    let footer_start = file.size - 8 - footer_len;
    let mut reads = 1;
    let tail_footer = usize::try_from(footer_start.saturating_sub(tail.start))
        .ok()
        .and_then(|skipped| tail.bytes.get(skipped..n - 8))
        .ok_or_else(|| footer_error("the footer lies outside the tail"))?;
    if footer_start >= tail.start {
        thrift_guard::check(tail_footer).map_err(guard_error)?;
    } else {
        reads += 1;
        let missing = footer_start..tail.start;
        let held = io
            .get_ranges_held(file, std::slice::from_ref(&missing))
            .await?;
        let head = held
            .first()
            .filter(|b| b.len() as u64 == tail.start - footer_start)
            .ok_or_else(|| footer_error("the footer read came back short"))?;
        let mut whole = Vec::with_capacity(head.len() + tail_footer.len());
        whole.extend_from_slice(head);
        whole.extend_from_slice(tail_footer);
        thrift_guard::check(&whole).map_err(guard_error)?;
        decoder
            .push_range(missing, head.clone())
            .map_err(footer_error)?;
    }
    decoder
        .push_range(tail.start..file.size, tail.bytes.clone())
        .map_err(footer_error)?;
    drop(tail);
    let meta = loop {
        let step = super::contain("reading the footer", FormatError::CorruptFooter, || {
            decoder.try_decode().map_err(footer_error)
        })?;
        match step {
            DecodeResult::Data(m) => break m,
            DecodeResult::Finished => return Err(footer_error("no footer")),
            DecodeResult::NeedsData(ranges) => {
                reads += 1;
                if reads > MAX_FOOTER_READS {
                    return Err(footer_error("the footer needs too many reads"));
                }
                if let Some(bad) = ranges.iter().find(|r| r.start > r.end || r.end > file.size) {
                    return Err(footer_error(format!(
                        "footer range {bad:?} is out of the file"
                    )));
                }
                let held = io.get_ranges_held(file, &ranges).await?;
                decoder
                    .push_ranges(ranges, held.to_vec())
                    .map_err(footer_error)?;
            }
        }
    };
    describe(&Arc::new(meta), file.size, options)
}

/// Checks `meta` (of a `size`-byte file) and describes it for the framework.
///
/// # Errors
/// [`FormatError::CorruptFooter`] for row groups or column chunks outside
/// the file, negative counts, or a schema arrow-rs cannot read.
pub(crate) fn describe(
    meta: &Arc<ParquetMetaData>,
    size: u64,
    options: SchemaOptions,
) -> FormatResult<ParquetMetadata> {
    super::contain("reading the schema", FormatError::CorruptFooter, || {
        describe_checked(meta, size, options)
    })
}

fn describe_checked(
    meta: &Arc<ParquetMetaData>,
    size: u64,
    options: SchemaOptions,
) -> FormatResult<ParquetMetadata> {
    let row_groups = check(meta, size)?;
    let inferred = ArrowReaderMetadata::try_new(Arc::clone(meta), ArrowReaderOptions::new())
        .map_err(footer_error)?;
    let descr = meta.file_metadata().schema_descr();
    let hint = Arc::new(Schema::new_with_metadata(
        inferred
            .schema()
            .fields()
            .iter()
            .map(|f| hint_field(f, &[f.name().as_str()], descr, options))
            .collect::<Vec<_>>(),
        inferred.schema().metadata().clone(),
    ));
    let hint = match super::collation::restore(&hint, meta.file_metadata().key_value_metadata()) {
        Some(restored) => Arc::new(restored),
        None => hint,
    };
    let reader_options = ArrowReaderOptions::new().with_schema(Arc::clone(&hint));
    let reader = ArrowReaderMetadata::try_new(Arc::clone(meta), reader_options.clone())
        .map_err(footer_error)?;
    let schema = Arc::new(Schema::new_with_metadata(
        reader
            .schema()
            .fields()
            .iter()
            .map(|f| Arc::new(spark_read_field(f)))
            .collect::<Vec<FieldRef>>(),
        reader.schema().metadata().clone(),
    ));
    let legacy_writer = legacy_binary_stats(meta.file_metadata().created_by());
    let mut first_row = 0u64;
    let mut planned = Vec::with_capacity(row_groups.len());
    for (i, (rows, bytes)) in row_groups.into_iter().enumerate() {
        let rg = meta.row_group(i);
        let mut columns = Vec::new();
        for (j, col) in rg.columns().iter().enumerate() {
            let d = descr.column(j);
            if d.max_rep_level() > 0 {
                continue;
            }
            let path = ColumnPath(d.path().parts().to_vec());
            let Some(dt) = leaf_type(&schema, &path.0) else {
                continue;
            };
            columns.push(column_statistics(
                path,
                &d,
                dt,
                col.statistics(),
                legacy_writer,
            ));
        }
        planned.push(RowGroup {
            first_row,
            rows,
            bytes: bytes.clone(),
            statistics: FileStatistics {
                rows: Precision::Exact(rows),
                bytes: Precision::Exact(bytes.end - bytes.start),
                columns,
            },
        });
        first_row = first_row
            .checked_add(rows)
            .ok_or_else(|| footer_error("the row groups hold more than 2^64 rows"))?;
    }
    let statistics = file_statistics(&planned);
    let heap = meta.memory_size() + planned.len() * std::mem::size_of::<RowGroup>();
    let calendars = super::calendar::calendars(meta.file_metadata().key_value_metadata());
    Ok(ParquetMetadata {
        calendars,
        reader,
        reader_options,
        schema,
        row_groups: planned,
        statistics,
        heap,
    })
}

/// The row count and byte extent of every row group, checked against the
/// file.
fn check(meta: &ParquetMetaData, size: u64) -> FormatResult<Vec<(u64, Range<u64>)>> {
    let leaves = meta.file_metadata().schema_descr().num_columns();
    let mut out = Vec::with_capacity(meta.num_row_groups());
    for (i, rg) in meta.row_groups().iter().enumerate() {
        let rows = u64::try_from(rg.num_rows())
            .map_err(|_| footer_error(format!("row group {i} has a negative row count")))?;
        if rg.num_columns() != leaves {
            return Err(footer_error(format!(
                "row group {i} has {} column chunks for {leaves} columns",
                rg.num_columns()
            )));
        }
        let mut extent: Option<Range<u64>> = None;
        for (j, c) in rg.columns().iter().enumerate() {
            if c.file_path().is_some() {
                return Err(footer_error(format!(
                    "column chunk {j} of row group {i} is stored in another file"
                )));
            }
            let start = c.dictionary_page_offset().unwrap_or(c.data_page_offset());
            let bad = || {
                footer_error(format!(
                    "column chunk {j} of row group {i} is out of the file"
                ))
            };
            let start = u64::try_from(start).map_err(|_| bad())?;
            let len = u64::try_from(c.compressed_size()).map_err(|_| bad())?;
            let end = start
                .checked_add(len)
                .filter(|e| *e <= size)
                .ok_or_else(bad)?;
            if c.data_page_offset() < 0
                || u64::try_from(c.data_page_offset()).is_ok_and(|d| d > end)
            {
                return Err(bad());
            }
            extent = Some(match extent {
                None => start..end,
                Some(r) => r.start.min(start)..r.end.max(end),
            });
        }
        out.push((rows, extent.unwrap_or(0..0)));
    }
    Ok(out)
}

/// The reading type of `field` (inferred by arrow-rs) in engine types.
fn hint_field(
    field: &FieldRef,
    path: &[&str],
    descr: &parquet::schema::types::SchemaDescriptor,
    o: SchemaOptions,
) -> FieldRef {
    let leaf = || {
        descr.columns().iter().find(|c| {
            c.path()
                .parts()
                .iter()
                .map(String::as_str)
                .eq(path.iter().copied())
        })
    };
    let dt = match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => DataType::Utf8View,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            if o.binary_as_string {
                DataType::Utf8View
            } else {
                DataType::BinaryView
            }
        }
        DataType::Dictionary(_, value) => {
            let inner = Arc::new(Field::new(field.name(), value.as_ref().clone(), true));
            hint_field(&inner, path, descr, o).data_type().clone()
        }
        // arrow-rs decodes `BINARY` decimals only into `Decimal128`; the scan
        // casts those to the carrier (design P2-T160).
        DataType::Decimal32(p, s) | DataType::Decimal64(p, s) | DataType::Decimal128(p, s) => {
            let binary = |c: &parquet::schema::types::ColumnDescriptor| {
                c.physical_type() == PhysicalType::BYTE_ARRAY
                    && c.converted_type() == parquet::basic::ConvertedType::DECIMAL
            };
            // A nested leaf has no path here: any binary decimal in the file
            // keeps the whole file's nested decimals on `Decimal128`.
            let readable = match leaf() {
                Some(c) => c.physical_type() != PhysicalType::BYTE_ARRAY,
                None => !descr.columns().iter().any(|c| binary(c)),
            };
            if *p <= krishiv_types::arrow::DECIMAL64_MAX_PRECISION && readable {
                DataType::Decimal64(*p, *s)
            } else {
                DataType::Decimal128(*p, *s)
            }
        }
        DataType::Timestamp(_, None)
            if o.int96_as_timestamp
                && leaf().is_some_and(|c| c.physical_type() == PhysicalType::INT96) =>
        {
            krishiv_types::DataType::Timestamp.to_arrow()
        }
        DataType::Struct(children) => DataType::Struct(Fields::from(
            children
                .iter()
                .map(|c| {
                    let mut p = path.to_vec();
                    p.push(c.name());
                    hint_field(c, &p, descr, o)
                })
                .collect::<Vec<_>>(),
        )),
        DataType::List(e) => DataType::List(hint_nested(e, descr, o)),
        DataType::LargeList(e) => DataType::LargeList(hint_nested(e, descr, o)),
        DataType::Map(e, sorted) => DataType::Map(hint_nested(e, descr, o), *sorted),
        other => other.clone(),
    };
    Arc::new(field.as_ref().clone().with_data_type(dt))
}

/// Children of lists and maps: their leaves have repetition, so no `INT96`
/// lookup by path is needed (nested `INT96` keeps arrow-rs's reading).
fn hint_nested(
    field: &FieldRef,
    descr: &parquet::schema::types::SchemaDescriptor,
    o: SchemaOptions,
) -> FieldRef {
    hint_field(field, &["\u{0}"], descr, o)
}

/// Unsigned integers and millisecond timestamps as Spark reads them (wider
/// signed integers, microsecond timestamps); nested types recursively.
pub(crate) fn spark_read_type(dt: &DataType) -> DataType {
    match dt {
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            DataType::Timestamp(TimeUnit::Microsecond, tz.clone())
        }
        DataType::Time32(TimeUnit::Millisecond) | DataType::Time64(TimeUnit::Microsecond) => {
            DataType::Time64(TimeUnit::Nanosecond)
        }
        DataType::UInt8 => DataType::Int16,
        DataType::UInt16 => DataType::Int32,
        DataType::UInt32 => DataType::Int64,
        DataType::UInt64 => DataType::Decimal128(20, 0),
        DataType::Struct(f) => {
            DataType::Struct(f.iter().map(|c| Arc::new(spark_read_field(c))).collect())
        }
        DataType::List(e) => DataType::List(Arc::new(spark_read_field(e))),
        DataType::LargeList(e) => DataType::LargeList(Arc::new(spark_read_field(e))),
        DataType::Map(e, s) => DataType::Map(Arc::new(spark_read_field(e)), *s),
        other => other.clone(),
    }
}

fn spark_read_field(f: &FieldRef) -> Field {
    f.as_ref()
        .clone()
        .with_data_type(spark_read_type(f.data_type()))
}

/// The type of the leaf at `path` (structs only).
fn leaf_type<'a>(schema: &'a Schema, path: &[String]) -> Option<&'a DataType> {
    let (first, rest) = path.split_first()?;
    let mut dt = schema.field_with_name(first).ok()?.data_type();
    for name in rest {
        let DataType::Struct(children) = dt else {
            return None;
        };
        dt = children.iter().find(|c| c.name() == name)?.data_type();
    }
    Some(dt)
}

/// Whether `created_by` names a writer whose binary statistics are ordered
/// by signed bytes (parquet-mr before 1.10, PARQUET-1025).
fn legacy_binary_stats(created_by: Option<&str>) -> bool {
    let Some(rest) = created_by.and_then(|c| c.strip_prefix("parquet-mr version ")) else {
        return false;
    };
    let version: Vec<u32> = rest
        .split(|c: char| !c.is_ascii_digit())
        .take(2)
        .filter_map(|p| p.parse().ok())
        .collect();
    matches!(version.as_slice(), [1, minor] if *minor < 10) || matches!(version.as_slice(), [0, _])
}

/// Statistics of one leaf column chunk in its engine type `dt`.
fn column_statistics(
    path: ColumnPath,
    d: &ColumnDescriptor,
    dt: &DataType,
    stats: Option<&Statistics>,
    legacy_writer: bool,
) -> ColumnStatistics {
    let mut c = ColumnStatistics::new(path);
    let Some(s) = stats else {
        return c;
    };
    if let Some(n) = s.null_count_opt() {
        c.null_count = Precision::Exact(n);
    }
    if let Some(n) = s.nan_count_opt() {
        c.nan_count = Precision::Exact(n);
    }
    let bytes_typed = matches!(
        d.physical_type(),
        PhysicalType::BYTE_ARRAY | PhysicalType::FIXED_LEN_BYTE_ARRAY
    );
    let unsigned = matches!(d.logical_type_ref(), Some(LogicalType::Integer(i)) if !i.is_signed)
        || matches!(
            d.converted_type(),
            ConvertedType::UINT_8
                | ConvertedType::UINT_16
                | ConvertedType::UINT_32
                | ConvertedType::UINT_64
        );
    let exact = |is_exact: bool| {
        !unsigned && (!bytes_typed || (is_exact && !s.is_min_max_deprecated() && !legacy_writer))
    };
    let precise = |v: Option<ScalarValue>, is_exact: bool| match v {
        None => Precision::Absent,
        Some(v) if exact(is_exact) => Precision::Exact(v),
        Some(v) => Precision::Inexact(v),
    };
    let (min, max) = scalar_bounds(s, dt, micros_per_unit(d));
    c.min = precise(min, s.min_is_exact() || !bytes_typed);
    c.max = precise(max, s.max_is_exact() || !bytes_typed);
    c
}

/// Microseconds per stored unit of a timestamp leaf (`None`: another
/// unit, or not a timestamp): a millisecond column reads as microseconds,
/// so its statistics are scaled the same way.
pub(super) fn micros_per_unit(d: &ColumnDescriptor) -> Option<i64> {
    match d.logical_type_ref() {
        Some(LogicalType::Timestamp(t)) => match t.unit {
            parquet::basic::TimeUnit::MILLIS => Some(1000),
            parquet::basic::TimeUnit::MICROS => Some(1),
            parquet::basic::TimeUnit::NANOS => None,
        },
        _ => match d.converted_type() {
            ConvertedType::TIMESTAMP_MILLIS => Some(1000),
            ConvertedType::TIMESTAMP_MICROS => Some(1),
            _ => None,
        },
    }
}

/// The statistic's bounds as values of `dt`, where they can be expressed;
/// timestamps are scaled by `micros_per_unit`.
fn scalar_bounds(
    s: &Statistics,
    dt: &DataType,
    micros_per_unit: Option<i64>,
) -> (Option<ScalarValue>, Option<ScalarValue>) {
    let int = |v: i64| -> Option<ScalarValue> {
        Some(match dt {
            DataType::Int8 => ScalarValue::Int8(i8::try_from(v).ok()?),
            DataType::Int16 => ScalarValue::Int16(i16::try_from(v).ok()?),
            DataType::Int32 => ScalarValue::Int32(i32::try_from(v).ok()?),
            DataType::Int64 => ScalarValue::Int64(v),
            DataType::Date32 => ScalarValue::Date32(i32::try_from(v).ok()?),
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                ScalarValue::TimestampMicros(v.checked_mul(micros_per_unit?)?)
            }
            DataType::Decimal64(p, sc) | DataType::Decimal128(p, sc) => ScalarValue::Decimal128 {
                unscaled: i128::from(v),
                precision: *p,
                scale: *sc,
            },
            _ => return None,
        })
    };
    let bytes = |b: &[u8]| -> Option<ScalarValue> {
        Some(match dt {
            DataType::Utf8View => ScalarValue::Utf8(std::str::from_utf8(b).ok()?.to_owned()),
            DataType::BinaryView => ScalarValue::Binary(b.to_vec()),
            DataType::Decimal64(p, sc) | DataType::Decimal128(p, sc) => ScalarValue::Decimal128 {
                unscaled: be_i128(b)?,
                precision: *p,
                scale: *sc,
            },
            _ => return None,
        })
    };
    match s {
        Statistics::Boolean(v) if *dt == DataType::Boolean => (
            v.min_opt().map(|b| ScalarValue::Boolean(*b)),
            v.max_opt().map(|b| ScalarValue::Boolean(*b)),
        ),
        Statistics::Int32(v) => (
            v.min_opt().and_then(|x| int(i64::from(*x))),
            v.max_opt().and_then(|x| int(i64::from(*x))),
        ),
        Statistics::Int64(v) => (
            v.min_opt().and_then(|x| int(*x)),
            v.max_opt().and_then(|x| int(*x)),
        ),
        Statistics::Float(v) if *dt == DataType::Float32 => (
            v.min_opt().map(|x| ScalarValue::Float32(*x)),
            v.max_opt().map(|x| ScalarValue::Float32(*x)),
        ),
        Statistics::Double(v) if *dt == DataType::Float64 => (
            v.min_opt().map(|x| ScalarValue::Float64(*x)),
            v.max_opt().map(|x| ScalarValue::Float64(*x)),
        ),
        Statistics::ByteArray(v) => (
            v.min_opt().and_then(|x| bytes(x.data())),
            v.max_opt().and_then(|x| bytes(x.data())),
        ),
        Statistics::FixedLenByteArray(v) => (
            v.min_opt().and_then(|x| bytes(x.data())),
            v.max_opt().and_then(|x| bytes(x.data())),
        ),
        _ => (None, None),
    }
}

/// A big-endian two's-complement integer of at most 16 bytes.
fn be_i128(b: &[u8]) -> Option<i128> {
    if b.is_empty() || b.len() > 16 {
        return None;
    }
    let fill = if b[0] & 0x80 == 0 { 0 } else { 0xFF };
    let mut buf = [fill; 16];
    buf[16 - b.len()..].copy_from_slice(b);
    Some(i128::from_be_bytes(buf))
}

/// Whole-file statistics: rows, bytes, and per column the null count and
/// the bounds over every row group (exact only if every row group's is).
fn file_statistics(row_groups: &[RowGroup]) -> FileStatistics {
    let rows: u64 = row_groups.iter().map(|r| r.rows).sum();
    let bytes: u64 = row_groups.iter().map(|r| r.bytes.end - r.bytes.start).sum();
    let mut columns: Vec<ColumnStatistics> = Vec::new();
    for (i, rg) in row_groups.iter().enumerate() {
        for c in &rg.statistics.columns {
            match columns.iter_mut().find(|m| m.path == c.path) {
                None if i == 0 => columns.push(c.clone()),
                None => {}
                Some(m) => {
                    m.null_count = sum(&m.null_count, &c.null_count);
                    m.nan_count = sum(&m.nan_count, &c.nan_count);
                    m.min = bound(&m.min, &c.min, std::cmp::Ordering::Less);
                    m.max = bound(&m.max, &c.max, std::cmp::Ordering::Greater);
                }
            }
        }
    }
    // A column missing from some row group's statistics is unknown.
    columns.retain(|c| {
        row_groups
            .iter()
            .all(|rg| rg.statistics.columns.iter().any(|x| x.path == c.path))
    });
    FileStatistics {
        rows: Precision::Exact(rows),
        bytes: Precision::Exact(bytes),
        columns,
    }
}

fn sum(a: &Precision<u64>, b: &Precision<u64>) -> Precision<u64> {
    match (a, b) {
        (Precision::Exact(x), Precision::Exact(y)) => Precision::Exact(x.saturating_add(*y)),
        _ => Precision::Absent,
    }
}

/// The smaller (`Less`) or larger (`Greater`) of two exact bounds.
fn bound(
    a: &Precision<ScalarValue>,
    b: &Precision<ScalarValue>,
    keep: std::cmp::Ordering,
) -> Precision<ScalarValue> {
    match (a, b) {
        (Precision::Exact(x), Precision::Exact(y)) => match compare(x, y) {
            Some(o) if o == keep => Precision::Exact(x.clone()),
            Some(_) => Precision::Exact(y.clone()),
            None => Precision::Absent,
        },
        _ => Precision::Absent,
    }
}

/// Orders two bounds of one column type (`None` when they cannot be
/// compared, e.g. NaN).
fn compare(a: &ScalarValue, b: &ScalarValue) -> Option<std::cmp::Ordering> {
    use ScalarValue as S;
    match (a, b) {
        (S::Boolean(x), S::Boolean(y)) => Some(x.cmp(y)),
        (S::Int8(x), S::Int8(y)) => Some(x.cmp(y)),
        (S::Int16(x), S::Int16(y)) => Some(x.cmp(y)),
        (S::Int32(x), S::Int32(y)) | (S::Date32(x), S::Date32(y)) => Some(x.cmp(y)),
        (S::Int64(x), S::Int64(y)) | (S::TimestampMicros(x), S::TimestampMicros(y)) => {
            Some(x.cmp(y))
        }
        (S::Float32(x), S::Float32(y)) => x.partial_cmp(y),
        (S::Float64(x), S::Float64(y)) => x.partial_cmp(y),
        (
            S::Decimal128 {
                unscaled: x,
                scale: sx,
                ..
            },
            S::Decimal128 {
                unscaled: y,
                scale: sy,
                ..
            },
        ) if sx == sy => Some(x.cmp(y)),
        (S::Utf8(x), S::Utf8(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        (S::Binary(x), S::Binary(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_writers_and_big_endian_decimals() {
        assert!(legacy_binary_stats(Some(
            "parquet-mr version 1.8.1 (build x)"
        )));
        assert!(!legacy_binary_stats(Some("parquet-mr version 1.10.0")));
        assert!(!legacy_binary_stats(Some(
            "parquet-cpp-arrow version 19.0.0"
        )));
        assert!(!legacy_binary_stats(None));
        assert_eq!(be_i128(&[0xFF, 0xFE]), Some(-2));
        assert_eq!(be_i128(&[0x01, 0x00]), Some(256));
        assert_eq!(be_i128(&[]), None);
        assert_eq!(be_i128(&[0; 17]), None);
        assert_eq!(
            compare(&ScalarValue::Float64(f64::NAN), &ScalarValue::Float64(1.0)),
            None
        );
    }
}
