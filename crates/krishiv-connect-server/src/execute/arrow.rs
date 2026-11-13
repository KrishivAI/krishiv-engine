// SPDX-License-Identifier: Apache-2.0
//! Result batches as Spark Connect sends them (design P3-T011a §3.2;
//! Spark's `SparkConnectPlanExecution.processAsArrowBatches` and
//! `ArrowConverters`, v4.3.0):
//!
//! - columns in Spark's Arrow shapes: `Utf8` and `Binary` (Spark's
//!   `useLargeVarTypes` is false; the engine holds views), timestamps
//!   labelled with the session time zone, geometry, geography and variant
//!   as `krishiv_connect::arrow` builds them, no Krishiv field metadata;
//! - batches cut at `maxRecordsPerBatch` rows and at `0.7 × maxBatchSize`
//!   bytes (the first row always goes; each row's size is added as the
//!   batch grows, estimated as its `UnsafeRow`, as Spark's
//!   `ArrowBatchWithSchemaIterator` does);
//! - each batch one complete Arrow IPC stream (schema, batch,
//!   end-of-stream), with the buffer compression of
//!   `spark.sql.execution.arrow.compression.codec`;
//! - the IPC bytes cut into chunks when the client allows it.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::TimestampMicrosecondType;
use arrow_array::{
    Array, ArrayRef, BinaryArray, ListArray, MapArray, RecordBatch, RecordBatchOptions,
    StringArray, StructArray,
};
use arrow_ipc::CompressionType;
use arrow_ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow_schema::{DataType as ArrowType, Field, Fields, Schema, SchemaRef, TimeUnit};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_types::DataType;

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// How result batches are cut and encoded.
#[derive(Debug, Clone)]
pub struct EncodeSettings {
    /// Most rows per batch (`spark.sql.execution.arrow.maxRecordsPerBatch`;
    /// `None` when zero or negative: no limit).
    pub max_rows: Option<usize>,
    /// Most estimated bytes per batch (`0.7 × maxBatchSize`).
    pub max_bytes: usize,
    /// IPC buffer compression (`none`, `zstd`, `lz4`).
    pub compression: Option<CompressionType>,
    /// The chunk size when chunking is on.
    pub chunk_size: Option<usize>,
    /// The session time zone timestamps are labelled with; `None` keeps the
    /// engine's label.
    pub time_zone: Option<Arc<str>>,
}

/// The Arrow type Spark sends for a column held as `t`.
fn spark_arrow_type(t: &ArrowType, tz: Option<&Arc<str>>) -> ArrowType {
    match t {
        ArrowType::Utf8View => ArrowType::Utf8,
        ArrowType::BinaryView => ArrowType::Binary,
        ArrowType::Timestamp(TimeUnit::Microsecond, Some(old)) => ArrowType::Timestamp(
            TimeUnit::Microsecond,
            Some(tz.map_or_else(|| Arc::clone(old), Arc::clone)),
        ),
        ArrowType::Struct(fields) => ArrowType::Struct(spark_fields(fields, tz)),
        ArrowType::List(f) => ArrowType::List(Arc::new(spark_field(f, tz))),
        ArrowType::Map(f, sorted) => ArrowType::Map(Arc::new(spark_field(f, tz)), *sorted),
        other => other.clone(),
    }
}

fn spark_field(f: &Field, tz: Option<&Arc<str>>) -> Field {
    Field::new(
        f.name(),
        spark_arrow_type(f.data_type(), tz),
        f.is_nullable(),
    )
}

fn spark_fields(fields: &Fields, tz: Option<&Arc<str>>) -> Fields {
    fields.iter().map(|f| spark_field(f, tz)).collect()
}

/// `array` with the type [`spark_arrow_type`] gives.
fn spark_array(array: &ArrayRef, tz: Option<&Arc<str>>) -> Result<ArrayRef, KrishivError> {
    let arrow = |e: arrow_schema::ArrowError| internal(format!("a result column: {e}"));
    Ok(match array.data_type() {
        ArrowType::Utf8View => Arc::new(array.as_string_view().iter().collect::<StringArray>()),
        ArrowType::BinaryView => Arc::new(array.as_binary_view().iter().collect::<BinaryArray>()),
        ArrowType::Timestamp(TimeUnit::Microsecond, Some(_)) => match tz {
            Some(tz) => Arc::new(
                array
                    .as_primitive::<TimestampMicrosecondType>()
                    .clone()
                    .with_timezone(Arc::clone(tz)),
            ),
            None => Arc::clone(array),
        },
        ArrowType::Struct(fields) => {
            let s = array.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(|c| spark_array(c, tz))
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                StructArray::try_new(spark_fields(fields, tz), columns, s.nulls().cloned())
                    .map_err(arrow)?,
            )
        }
        ArrowType::List(f) => {
            let l = array.as_list::<i32>();
            Arc::new(
                ListArray::try_new(
                    Arc::new(spark_field(f, tz)),
                    l.offsets().clone(),
                    spark_array(l.values(), tz)?,
                    l.nulls().cloned(),
                )
                .map_err(arrow)?,
            )
        }
        ArrowType::Map(f, sorted) => {
            let m = array.as_map();
            let entries = spark_array(&(Arc::new(m.entries().clone()) as ArrayRef), tz)?;
            Arc::new(
                MapArray::try_new(
                    Arc::new(spark_field(f, tz)),
                    m.offsets().clone(),
                    entries.as_struct().clone(),
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(arrow)?,
            )
        }
        _ => Arc::clone(array),
    })
}

/// `batch` in the shapes Spark sends (module docs).
///
/// # Errors
/// `INTERNAL_ERROR` for a column that is not what its field says (an
/// engine bug).
pub fn spark_batch(
    batch: &RecordBatch,
    tz: Option<&Arc<str>>,
) -> Result<RecordBatch, KrishivError> {
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        let spark = DataType::from_arrow_field(field).ok();
        let (f, c) = match &spark {
            Some(t @ (DataType::Geometry { .. } | DataType::Geography { .. })) => {
                match krishiv_connect::arrow::spatial_field(field.name(), t, field.is_nullable()) {
                    Some(f) => (f, krishiv_connect::arrow::spatial_column(column, t)?),
                    None => (spark_field(field, tz), spark_array(column, tz)?),
                }
            }
            Some(DataType::Variant) => (
                krishiv_connect::arrow::variant_field(field.name(), field.is_nullable()),
                krishiv_connect::arrow::variant_column(column)?,
            ),
            _ => (spark_field(field, tz), spark_array(column, tz)?),
        };
        fields.push(f);
        columns.push(c);
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| internal(format!("a result batch: {e}")))
}

/// The rows of `batch` in slices of at most `settings.max_rows` rows and
/// about `settings.max_bytes` bytes (at least one row each). As Spark's
/// `ArrowBatchWithSchemaIterator`, the size of each row is added as the
/// slice grows, and the slice ends with the row that reaches the limit; a
/// row's size is estimated as its `UnsafeRow` (`row_sizes`).
#[must_use]
pub fn slices(batch: &RecordBatch, settings: &EncodeSettings) -> Vec<RecordBatch> {
    let rows = batch.num_rows();
    if rows == 0 {
        return Vec::new();
    }
    let max_rows = settings.max_rows.unwrap_or(usize::MAX).max(1);
    if settings.max_bytes == 0 && max_rows >= rows {
        return vec![batch.clone()];
    }
    let sizes = row_sizes(batch);
    let mut out = Vec::new();
    let (mut start, mut count, mut bytes) = (0, 0, 0_usize);
    for (row, size) in sizes.iter().enumerate() {
        count += 1;
        bytes = bytes.saturating_add(*size);
        let full = (settings.max_bytes > 0 && bytes >= settings.max_bytes) || count >= max_rows;
        if full || row + 1 == rows {
            out.push(batch.slice(start, count));
            start = row + 1;
            count = 0;
            bytes = 0;
        }
    }
    out
}

/// `bytes` rounded up to a multiple of 8 (`UnsafeRow` words).
fn words(bytes: usize) -> usize {
    bytes.div_ceil(8).saturating_mul(8)
}

/// The estimated size of each row of `batch`: Spark's `UnsafeRow`, a null
/// bit set, eight bytes a field, then the variable-length part of each
/// value ([`var_sizes`]).
fn row_sizes(batch: &RecordBatch) -> Vec<usize> {
    let fields = batch.num_columns();
    let fixed = fields.div_ceil(64) * 8 + fields * 8;
    let mut sizes = vec![fixed; batch.num_rows()];
    for column in batch.columns() {
        for (size, var) in sizes.iter_mut().zip(var_sizes(column.as_ref())) {
            *size = size.saturating_add(var);
        }
    }
    sizes
}

/// The variable-length bytes each value of `array` adds to an `UnsafeRow`:
/// none for a fixed-width value, the (word-rounded) bytes of a string or
/// binary, a nested row for a struct, an `UnsafeArrayData` for an array or
/// a map. A type without a cheap per-value size takes the array's average.
fn var_sizes(array: &dyn Array) -> Vec<usize> {
    let n = array.len();
    match array.data_type() {
        ArrowType::Utf8 => array
            .as_string::<i32>()
            .iter()
            .map(|v| words(v.map_or(0, str::len)))
            .collect(),
        ArrowType::LargeUtf8 => array
            .as_string::<i64>()
            .iter()
            .map(|v| words(v.map_or(0, str::len)))
            .collect(),
        ArrowType::Utf8View => array
            .as_string_view()
            .iter()
            .map(|v| words(v.map_or(0, str::len)))
            .collect(),
        ArrowType::Binary => array
            .as_binary::<i32>()
            .iter()
            .map(|v| words(v.map_or(0, <[u8]>::len)))
            .collect(),
        ArrowType::LargeBinary => array
            .as_binary::<i64>()
            .iter()
            .map(|v| words(v.map_or(0, <[u8]>::len)))
            .collect(),
        ArrowType::BinaryView => array
            .as_binary_view()
            .iter()
            .map(|v| words(v.map_or(0, <[u8]>::len)))
            .collect(),
        ArrowType::Decimal128(p, _) if *p > 18 => vec![16; n],
        ArrowType::Struct(fields) => {
            let s = array.as_struct();
            let fixed = fields.len().div_ceil(64) * 8 + fields.len() * 8;
            let mut sizes = vec![fixed; n];
            for column in s.columns() {
                for (size, var) in sizes.iter_mut().zip(var_sizes(column.as_ref())) {
                    *size = size.saturating_add(var);
                }
            }
            sizes
        }
        ArrowType::List(_) => {
            let l = array.as_list::<i32>();
            element_sizes(
                l.offsets().iter().map(|o| usize::try_from(*o).unwrap_or(0)),
                l.values().as_ref(),
            )
        }
        ArrowType::LargeList(_) => {
            let l = array.as_list::<i64>();
            element_sizes(
                l.offsets().iter().map(|o| usize::try_from(*o).unwrap_or(0)),
                l.values().as_ref(),
            )
        }
        ArrowType::Map(_, _) => {
            let m = array.as_map();
            // Two arrays (keys, values) and the eight bytes of their length.
            let keys = element_sizes(
                m.offsets().iter().map(|o| usize::try_from(*o).unwrap_or(0)),
                m.keys().as_ref(),
            );
            let values = element_sizes(
                m.offsets().iter().map(|o| usize::try_from(*o).unwrap_or(0)),
                m.values().as_ref(),
            );
            keys.iter().zip(values).map(|(k, v)| 8 + k + v).collect()
        }
        t if t.is_primitive() || matches!(t, ArrowType::Boolean | ArrowType::Null) => vec![0; n],
        _ => {
            let average = array.get_array_memory_size().div_ceil(n.max(1));
            vec![average; n]
        }
    }
}

/// The `UnsafeArrayData` of each list of `values` between consecutive
/// `offsets`: a length word, a null bit set, eight bytes an element and the
/// elements' variable parts.
fn element_sizes(offsets: impl Iterator<Item = usize>, values: &dyn Array) -> Vec<usize> {
    let var = var_sizes(values);
    let mut prefix = Vec::with_capacity(var.len() + 1);
    prefix.push(0_usize);
    for v in &var {
        let last = prefix.last().copied().unwrap_or(0);
        prefix.push(last.saturating_add(*v));
    }
    let offsets: Vec<usize> = offsets.collect();
    offsets
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0].min(var.len()), w[1].min(var.len()));
            let len = b.saturating_sub(a);
            let elements = prefix[b].saturating_sub(prefix[a]);
            8 + len.div_ceil(64) * 8 + len * 8 + elements
        })
        .collect()
}

/// `batch` as one Arrow IPC stream: schema, the batch, end of stream.
///
/// # Errors
/// `INTERNAL_ERROR` when Arrow cannot write it.
pub fn ipc_stream(
    batch: &RecordBatch,
    compression: Option<CompressionType>,
) -> Result<Vec<u8>, KrishivError> {
    let arrow = |e: arrow_schema::ArrowError| internal(format!("writing a result batch: {e}"));
    let options = IpcWriteOptions::default()
        .try_with_compression(compression)
        .map_err(arrow)?;
    let mut writer =
        StreamWriter::try_new_with_options(Vec::new(), &batch.schema(), options).map_err(arrow)?;
    writer.write(batch).map_err(arrow)?;
    writer.finish().map_err(arrow)?;
    writer.into_inner().map_err(arrow)
}

/// An empty batch of `schema`, in Spark's shapes (Spark's
/// `createEmptyArrowBatch`, sent when the result has no row).
///
/// # Errors
/// As [`spark_batch`].
pub fn empty_batch(schema: &SchemaRef, tz: Option<&Arc<str>>) -> Result<RecordBatch, KrishivError> {
    spark_batch(&RecordBatch::new_empty(Arc::clone(schema)), tz)
}

/// The codec named by `spark.sql.execution.arrow.compression.codec`
/// (`none`, `zstd`, `lz4`; Spark 4.1+). An unknown name is `none`.
#[must_use]
pub fn compression_codec(name: Option<&str>) -> Option<CompressionType> {
    match name.map(str::to_ascii_lowercase).as_deref() {
        Some("zstd") => Some(CompressionType::ZSTD),
        Some("lz4") => Some(CompressionType::LZ4_FRAME),
        _ => None,
    }
}

/// `bytes` in chunks of at most `size` bytes (one empty chunk for no
/// bytes).
#[must_use]
pub fn chunks(bytes: &[u8], size: usize) -> Vec<&[u8]> {
    if bytes.is_empty() {
        return vec![bytes];
    }
    bytes.chunks(size.max(1)).collect()
}
