// SPDX-License-Identifier: Apache-2.0
//! The Parquet writer (P2-T082a): arrow-rs's `ArrowWriter` over an
//! in-memory buffer that the framework drains and uploads (the sans-IO
//! [`Encoder`], ADR 0014). No custom encoders.
//!
//! - **Spark's settings and file names.** The codec comes from the
//!   `compression` option, then `parquet.compression`, then
//!   `spark.sql.parquet.compression.codec` (default `snappy`), and names the
//!   file as parquet-mr does (`.snappy.parquet`, `.zstd.parquet`, ...).
//!   `parquet.block.size` (128 MiB) bounds a row group by its encoded size,
//!   `parquet.page.size` a page, as in parquet-mr.
//! - **Page indexes are always written** (column and offset indexes, E08).
//!   Column-index bounds are truncated at 64 bytes, as parquet-mr; footer
//!   bounds at 4 KiB (parquet-mr drops longer ones; arrow-rs truncates them
//!   and marks them inexact, which readers treat as such), so huge values
//!   cannot bloat the footer.
//! - **Footer metadata:** `org.apache.spark.version` (below) and arrow-rs's
//!   `ARROW:schema`; Spark's own schema JSON
//!   (`org.apache.spark.sql.parquet.row.metadata`) is written only when a
//!   column holds a collated string (see `collation.rs`); otherwise Spark
//!   reads the Parquet types instead (the same Spark types for every type
//!   the engine has; checked in the interop fixtures).
//! - **Strings are UTF-8.** A `string` column stored as binary (non-UTF-8
//!   bytes, tagged `krishiv.spark_type`) is refused: it can be written
//!   neither as a valid `STRING` nor as `binary` without changing its type.
//! - **Timestamps** are `INT64` microseconds by default (Spark writes
//!   `INT96`, which arrow-rs cannot write: `outputTimestampType=INT96` is
//!   refused when the data has timestamps); `TIMESTAMP_MILLIS` rounds down
//!   as Spark does. `timestamp_ntz` is always microseconds.
//! - **Calendars.** Values are proleptic Gregorian, and the footer says so
//!   the way Spark's readers understand (`org.apache.spark.version`), so
//!   any Spark 3.x/4.x reader takes them as stored.
//!   `datetimeRebaseModeInWrite=EXCEPTION` fails on a date before
//!   1582-10-15 or a timestamp before 1900 (Spark wraps the same error in
//!   `TASK_WRITE_FAILED`; there is no cause chain yet); `LEGACY` marks
//!   the file as Spark does and refuses such values (the rebase itself is
//!   not implemented), so no value is ever shifted silently.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Time64NanosecondType, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, LargeListArray, ListArray, MapArray, RecordBatch, StructArray,
    Time64MicrosecondArray, TimestampMillisecondArray,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use bytes::Bytes;
use krishiv_common::error::{
    CodecNotAvailableWithAvailableCodecsSuggestion,
    InconsistentBehaviorCrossVersionWriteAncientDatetime, KrishivUnsupportedFeature,
};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_writer::ArrowWriterOptions;
use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{
    DEFAULT_CREATED_BY, EnabledStatistics, WriterProperties, WriterPropertiesBuilder, WriterVersion,
};

use super::ParquetFormat;
use super::calendar;
use crate::error::{FormatError, FormatResult};
use crate::format::FileMetadata;
use crate::format::{Encoder, WrittenFile};
use crate::options::FormatOptions;

/// The Spark version whose file conventions the writer follows: Spark's
/// readers take the values of a file marked with it (or any version from
/// 3.0) as proleptic Gregorian.
pub const SPARK_FILE_VERSION: &str = "4.2.0";

/// Most rows handed to arrow-rs at a time (see [`ParquetEncoder`]'s
/// `write`).
const SLICE_ROWS: usize = 4096;

/// Longest footer statistics bound, as parquet-mr's `MAX_STATS_SIZE`.
const FOOTER_BOUND_BYTES: usize = 4096;

/// Bounds of a column index entry, as parquet-mr truncates them.
const COLUMN_INDEX_TRUNCATE_LENGTH: usize = 64;

/// Spark's codec names (`ParquetOptions`) with the file name part
/// parquet-mr gives each (checked in the Spark 4.2 container).
const CODECS: &[(&str, &str)] = &[
    ("brotli", ".br"),
    ("gzip", ".gz"),
    ("lz4", ".lz4hadoop"),
    ("lz4_raw", ".lz4raw"),
    ("lzo", ".lzo"),
    ("none", ""),
    ("snappy", ".snappy"),
    ("uncompressed", ""),
    ("zstd", ".zstd"),
];

/// arrow-rs's codec for a name of [`CODECS`] (`None`: not available).
fn compression(name: &str) -> Option<Compression> {
    Some(match name {
        "none" | "uncompressed" => Compression::UNCOMPRESSED,
        "snappy" => Compression::SNAPPY,
        "gzip" => Compression::GZIP(GzipLevel::default()),
        "brotli" => Compression::BROTLI(BrotliLevel::default()),
        "lz4" => Compression::LZ4,
        "lz4_raw" => Compression::LZ4_RAW,
        // parquet-mr's default level (`parquet.compression.codec.zstd.level`).
        "zstd" => Compression::ZSTD(ZstdLevel::try_new(3).ok()?),
        _ => return None,
    })
}

/// How a write treats dates and timestamps before the calendar switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rebase {
    /// Written as they are (the default).
    Corrected,
    /// Such a value fails the write.
    Exception,
    /// The file is marked as legacy-calendar; such a value is refused.
    Legacy,
}

/// The settings of a write, from its options.
#[derive(Debug, Clone)]
pub(crate) struct Settings {
    extension: &'static str,
    compression: Compression,
    millis: bool,
    /// Whether batches need `to_parquet` (milliseconds or `TIME`).
    convert: bool,
    rebase: Rebase,
    /// Whether the data has dates or timestamps (values to check).
    datetimes: bool,
    block_size: usize,
    page_size: usize,
    dictionary: bool,
    version: WriterVersion,
}

fn int(o: &FormatOptions, name: &str, default: usize) -> usize {
    o.int(name)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

impl Settings {
    /// The settings of a write of `schema` with `o`.
    ///
    /// # Errors
    /// `CODEC_NOT_AVAILABLE.WITH_AVAILABLE_CODECS_SUGGESTION` for an unknown
    /// codec; `KRISHIV_UNSUPPORTED_FEATURE` for `lzo`, and for `INT96` output
    /// or `LEGACY` rebasing when `schema` has values they would apply to.
    pub(crate) fn new(schema: &Schema, o: &FormatOptions) -> FormatResult<Self> {
        let name = o
            .str("compression")
            .or_else(|| o.str("parquet.compression"))
            .unwrap_or("snappy")
            .to_ascii_lowercase();
        let &(codec, extension) = CODECS.iter().find(|c| c.0 == name).ok_or_else(|| {
            let names: Vec<&str> = CODECS.iter().map(|c| c.0).collect();
            FormatError::Other(
                CodecNotAvailableWithAvailableCodecsSuggestion {
                    codec_name: name.clone(),
                    available_codecs: names.join(", "),
                }
                .into(),
            )
        })?;
        let compression =
            compression(codec).ok_or_else(|| unsupported(&format!("the {codec} Parquet codec")))?;
        let has =
            |f: fn(&DataType) -> bool| schema.fields().iter().any(|x| any_type(x.data_type(), f));
        let millis = match o.str("outputTimestampType").unwrap_or("TIMESTAMP_MICROS") {
            "INT96" if has(is_timestamp) => {
                return Err(unsupported(
                    "writing INT96 Parquet timestamps (spark.sql.parquet.outputTimestampType); \
                     use TIMESTAMP_MICROS",
                ));
            }
            "TIMESTAMP_MILLIS" => true,
            _ => false,
        };
        let rebase = match o.str("datetimeRebaseModeInWrite").unwrap_or("CORRECTED") {
            "EXCEPTION" => Rebase::Exception,
            "LEGACY" => Rebase::Legacy,
            _ => Rebase::Corrected,
        };
        let version = match o.str("parquet.writer.version").unwrap_or("v1") {
            "v2" | "PARQUET_2_0" => WriterVersion::PARQUET_2_0,
            _ => WriterVersion::PARQUET_1_0,
        };
        if schema.fields().iter().any(|f| binary_string(f)) {
            return Err(unsupported(
                "writing a string column holding invalid UTF-8 to Parquet",
            ));
        }
        Ok(Self {
            extension,
            compression,
            millis: millis && has(is_timestamp),
            convert: (millis && has(is_timestamp)) || has(is_time),
            rebase,
            datetimes: has(is_datetime),
            block_size: int(o, "parquet.block.size", 128 << 20),
            page_size: int(o, "parquet.page.size", 1 << 20),
            dictionary: o.bool("parquet.enable.dictionary").unwrap_or(true),
            version,
        })
    }

    /// The file name part of the codec (`.snappy`, `.zstd`, or nothing).
    pub(crate) fn extension(&self) -> &'static str {
        self.extension
    }

    /// arrow-rs's properties for these settings.
    fn properties(&self, row_metadata: Option<KeyValue>) -> WriterPropertiesBuilder {
        let mut kv = vec![KeyValue::new(
            "org.apache.spark.version".to_owned(),
            SPARK_FILE_VERSION.to_owned(),
        )];
        kv.extend(row_metadata);
        if self.rebase == Rebase::Legacy {
            kv.push(KeyValue::new(
                "org.apache.spark.legacyDateTime".to_owned(),
                String::new(),
            ));
        }
        WriterProperties::builder()
            .set_compression(self.compression)
            .set_writer_version(self.version)
            .set_max_row_group_row_count(None)
            .set_max_row_group_bytes(Some(self.block_size))
            .set_data_page_size_limit(self.page_size)
            .set_dictionary_enabled(self.dictionary)
            // Page statistics give the column index; the offset index is on.
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_offset_index_disabled(false)
            .set_statistics_truncate_length(Some(FOOTER_BOUND_BYTES))
            .set_column_index_truncate_length(Some(COLUMN_INDEX_TRUNCATE_LENGTH))
            .set_created_by(format!(
                "krishiv version {} (build {DEFAULT_CREATED_BY})",
                env!("CARGO_PKG_VERSION")
            ))
            .set_key_value_metadata(Some(kv))
    }
}

fn unsupported(feature: &str) -> FormatError {
    FormatError::Other(
        KrishivUnsupportedFeature {
            feature: feature.to_owned(),
        }
        .into(),
    )
}

/// Whether `f`, or a field nested in it, is a `string` stored as binary.
fn binary_string(f: &Field) -> bool {
    let tagged = f
        .metadata()
        .get(krishiv_types::arrow::SPARK_TYPE_METADATA_KEY)
        .is_some_and(|t| t == "string");
    let binary = matches!(
        f.data_type(),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    );
    (tagged && binary)
        || match f.data_type() {
            DataType::Struct(fields) => fields.iter().any(|c| binary_string(c)),
            DataType::List(c) | DataType::LargeList(c) | DataType::Map(c, _) => binary_string(c),
            _ => false,
        }
}

fn is_timestamp(t: &DataType) -> bool {
    matches!(t, DataType::Timestamp(TimeUnit::Microsecond, Some(_)))
}

fn is_time(t: &DataType) -> bool {
    matches!(t, DataType::Time64(TimeUnit::Nanosecond))
}

fn is_datetime(t: &DataType) -> bool {
    matches!(t, DataType::Date32 | DataType::Timestamp(_, Some(_)))
}

/// Whether `t` or a type nested in it satisfies `f`.
fn any_type(t: &DataType, f: fn(&DataType) -> bool) -> bool {
    f(t) || match t {
        DataType::Struct(fields) => fields.iter().any(|x| any_type(x.data_type(), f)),
        DataType::List(x) | DataType::LargeList(x) | DataType::Map(x, _) => {
            any_type(x.data_type(), f)
        }
        _ => false,
    }
}

/// `a` as Parquet stores it: timestamps (with a time zone) in milliseconds
/// when `millis` (rounded down as Spark does, `floorDiv`), and `TIME` in
/// microseconds (Spark's `TIME(MICROS)`), at any depth.
fn to_parquet(a: &ArrayRef, millis: bool) -> FormatResult<ArrayRef> {
    let bad = |e: arrow_schema::ArrowError| FormatError::WriteFailed(e.to_string());
    let retyped = |f: &Field, t: &DataType| Arc::new(f.clone().with_data_type(t.clone()));
    Ok(match a.data_type() {
        DataType::Time64(TimeUnit::Nanosecond) => {
            let nanos = a.as_primitive::<Time64NanosecondType>();
            let micros: Time64MicrosecondArray = nanos.unary(|v| v.div_euclid(1000));
            Arc::new(micros)
        }
        DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) if millis => {
            let micros = a.as_primitive::<TimestampMicrosecondType>();
            let millis: TimestampMillisecondArray = micros.unary(|v| v.div_euclid(1000));
            Arc::new(millis.with_timezone(Arc::clone(tz)))
        }
        DataType::Struct(fields) => {
            let s = a.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(|c| to_parquet(c, millis))
                .collect::<FormatResult<Vec<_>>>()?;
            let fields: Fields = fields
                .iter()
                .zip(&columns)
                .map(|(f, c)| retyped(f, c.data_type()))
                .collect();
            Arc::new(StructArray::try_new(fields, columns, s.nulls().cloned()).map_err(bad)?)
        }
        DataType::List(f) => {
            let l = a.as_list::<i32>();
            let values = to_parquet(l.values(), millis)?;
            let f = retyped(f, values.data_type());
            Arc::new(
                ListArray::try_new(f, l.offsets().clone(), values, l.nulls().cloned())
                    .map_err(bad)?,
            )
        }
        DataType::LargeList(f) => {
            let l = a.as_list::<i64>();
            let values = to_parquet(l.values(), millis)?;
            let f = retyped(f, values.data_type());
            Arc::new(
                LargeListArray::try_new(f, l.offsets().clone(), values, l.nulls().cloned())
                    .map_err(bad)?,
            )
        }
        DataType::Map(f, sorted) => {
            let m = a.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = to_parquet(&entries, millis)?;
            let f = retyped(f, entries.data_type());
            Arc::new(
                MapArray::try_new(
                    f,
                    m.offsets().clone(),
                    entries.as_struct().clone(),
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(bad)?,
            )
        }
        _ => Arc::clone(a),
    })
}

/// The Parquet encoder of one file (see the module docs).
pub(crate) struct ParquetEncoder {
    writer: ArrowWriter<Vec<u8>>,
    /// The schema of the batches the writer takes (milliseconds applied).
    schema: SchemaRef,
    settings: Settings,
    options: FormatOptions,
}

impl ParquetEncoder {
    /// An encoder of `schema` with `o`, its arrow-rs properties adjusted by
    /// `tweak`.
    pub(crate) fn new(
        schema: &SchemaRef,
        o: &FormatOptions,
        tweak: impl FnOnce(WriterPropertiesBuilder) -> WriterPropertiesBuilder,
    ) -> FormatResult<Self> {
        let settings = Settings::new(schema, o)?;
        let schema = if settings.convert {
            let empty = RecordBatch::new_empty(Arc::clone(schema));
            let fields: Vec<Field> = schema
                .fields()
                .iter()
                .zip(empty.columns())
                .map(|(f, c)| {
                    Ok(f.as_ref()
                        .clone()
                        .with_data_type(to_parquet(c, settings.millis)?.data_type().clone()))
                })
                .collect::<FormatResult<_>>()?;
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
        } else {
            Arc::clone(schema)
        };
        // The footer keeps arrow-rs's Arrow schema next to the Parquet
        // types: other engines read the Parquet types, Krishiv's reader
        // also gets back what Parquet cannot say (extension tags).
        let options = ArrowWriterOptions::new().with_properties(
            tweak(settings.properties(super::collation::row_metadata(&schema))).build(),
        );
        let writer = ArrowWriter::try_new_with_options(Vec::new(), Arc::clone(&schema), options)
            .map_err(|e| FormatError::WriteFailed(e.to_string()))?;
        Ok(Self {
            writer,
            schema,
            settings,
            options: o.clone(),
        })
    }
}

impl Encoder for ParquetEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        match self.settings.rebase {
            _ if !self.settings.datetimes => {}
            Rebase::Corrected => {}
            Rebase::Exception => {
                if batch.columns().iter().any(calendar::ancient) {
                    return Err(FormatError::Other(
                        InconsistentBehaviorCrossVersionWriteAncientDatetime {
                            format: "Parquet".to_owned(),
                            config: "\"spark.sql.parquet.datetimeRebaseModeInWrite\"".to_owned(),
                        }
                        .into(),
                    ));
                }
            }
            Rebase::Legacy => {
                if batch.columns().iter().any(calendar::ancient) {
                    return Err(unsupported(
                        "rebasing Parquet dates before 1582-10-15 or timestamps before \
                         1900-01-01 to the legacy hybrid calendar \
                         (spark.sql.parquet.datetimeRebaseModeInWrite=LEGACY)",
                    ));
                }
            }
        }
        let converted;
        let batch = if self.settings.convert {
            let columns = batch
                .columns()
                .iter()
                .map(|c| to_parquet(c, self.settings.millis))
                .collect::<FormatResult<_>>()?;
            converted = RecordBatch::try_new(Arc::clone(&self.schema), columns)
                .map_err(|e| FormatError::WriteFailed(e.to_string()))?;
            &converted
        } else {
            batch
        };
        // arrow-rs sizes a row group from the rows it already holds and
        // takes the first write of each one whole, so batches go in slices
        // of at most a sixteenth of `parquet.block.size` (by the batch's
        // in-memory row size) and at most `SLICE_ROWS` rows: a large batch
        // then cannot overshoot the block size by much, even with wide rows.
        let row_bytes = batch.get_array_memory_size() / batch.num_rows().max(1);
        let slice = (self.settings.block_size / 16 / row_bytes.max(1)).clamp(1, SLICE_ROWS);
        let mut offset = 0;
        while offset < batch.num_rows() {
            let n = slice.min(batch.num_rows() - offset);
            self.writer
                .write(&batch.slice(offset, n))
                .map_err(|e| FormatError::WriteFailed(e.to_string()))?;
            offset += n;
        }
        Ok(())
    }

    fn flush(&mut self) -> FormatResult<()> {
        self.writer
            .flush()
            .map_err(|e| FormatError::WriteFailed(e.to_string()))
    }

    fn take_bytes(&mut self) -> Option<Bytes> {
        let buf = std::mem::take(self.writer.inner_mut());
        (!buf.is_empty()).then(|| Bytes::from(buf))
    }

    fn finish(mut self: Box<Self>) -> FormatResult<WrittenFile> {
        let meta = self
            .writer
            .finish()
            .map_err(|e| FormatError::WriteFailed(e.to_string()))?;
        let size = self.writer.bytes_written() as u64;
        let tail = Bytes::from(std::mem::take(self.writer.inner_mut()));
        // Our own footer: a failure to describe it is a write failure.
        let described = ParquetFormat::describe(&Arc::new(meta), size, &self.options)
            .map_err(|e| FormatError::WriteFailed(e.to_string()))?;
        Ok(WrittenFile::new(tail, described.statistics()))
    }

    fn buffered_bytes(&self) -> usize {
        self.writer
            .memory_size()
            .saturating_add(self.writer.inner().len())
    }

    fn written_bytes(&self) -> u64 {
        // The open row group counts at arrow-rs's estimate of its encoded
        // size (before compression), so a target file size smaller than a
        // row group still rolls over; with compressible data a file can end
        // up below the target.
        (self.writer.bytes_written() as u64).saturating_add(self.writer.in_progress_size() as u64)
    }
}

#[cfg(test)]
mod tests;
