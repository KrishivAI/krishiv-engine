// SPDX-License-Identifier: Apache-2.0
//! The Parquet [`Decoder`]: arrow-rs's `ParquetPushDecoder` (sans-IO) behind
//! the framework's step/push interface. The framework fetches what it asks
//! for on the I/O runtime; decoding runs where `step` is called (a compute
//! worker), one batch per step.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    TimestampMicrosecondType, TimestampMillisecondType, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, LargeListArray, ListArray, MapArray, RecordBatch, RecordBatchOptions,
    StructArray,
};
use arrow_schema::{DataType, Field, SchemaRef, TimeUnit};
use bytes::Bytes;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, RowSelection as ParquetSelection};
use parquet::arrow::push_decoder::{
    ParquetPushDecoder, ParquetPushDecoderBuilder, RowGroupSelection,
};
use parquet::arrow::{ProjectionMask, RowNumber};
use parquet::errors::ParquetError;

use super::calendar::Guard;
use super::meta::{ParquetMetadata, spark_read_type};
use crate::error::{FormatError, FormatResult};
use crate::format::{Decoder, ScanUnitMetrics, Step, UnitPlan};
use crate::options::FormatOptions;
use crate::request::{ColumnPath, ScanRequest, project_batch};

/// Rows per decoded batch (`krishiv.exec.batchSize`'s default).
pub(crate) const BATCH_ROWS: usize = 4096;

/// A decode error: invalid UTF-8 in a `STRING` column is its own class
/// (`INVALID_UTF8_STRING`), anything else is corrupt data.
fn decode_error(e: &ParquetError) -> FormatError {
    let text = e.to_string();
    if text.contains("UTF-8") || text.contains("UTF8") || text.contains("utf-8") {
        FormatError::InvalidUtf8(text)
    } else {
        FormatError::Corrupt(text)
    }
}

/// See the module docs.
pub(crate) struct ParquetDecoder {
    decoder: ParquetPushDecoder,
    /// Planned row groups, in order, with their projected chunk ranges.
    planned: Vec<Vec<Range<u64>>>,
    projection: Vec<ColumnPath>,
    output: SchemaRef,
    row_index: bool,
    widen: bool,
    metrics: ScanUnitMetrics,
    /// arrow-rs panicked: its state is not used again.
    poisoned: bool,
    /// Spark's calendar rules for the output columns.
    guard: Guard,
}

impl std::fmt::Debug for ParquetDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetDecoder")
            .field("row_groups", &self.planned.len())
            .field("metrics", &self.metrics)
            .finish_non_exhaustive()
    }
}

/// The leaves of `paths`: every leaf under a path through structs, or the
/// whole top-level column for a path into a list or map (pruned after
/// decoding).
fn projection_mask(meta: &ParquetMetadata, paths: &[ColumnPath]) -> (ProjectionMask, Vec<usize>) {
    let descr = meta.reader.parquet_schema();
    let leaves: Vec<&[String]> = descr.columns().iter().map(|c| c.path().parts()).collect();
    let mut picked = Vec::new();
    for p in paths {
        let mut hits: Vec<usize> = (0..leaves.len())
            .filter(|&i| leaves[i].starts_with(&p.0))
            .collect();
        if hits.is_empty() {
            hits = (0..leaves.len())
                .filter(|&i| leaves[i].first().map(String::as_str) == Some(p.root()))
                .collect();
        }
        picked.extend(hits);
    }
    picked.sort_unstable();
    picked.dedup();
    (
        ProjectionMask::leaves(descr, picked.iter().copied()),
        picked,
    )
}

impl ParquetDecoder {
    pub(crate) fn new(
        meta: &Arc<ParquetMetadata>,
        units: &UnitPlan,
        req: &ScanRequest,
        o: &FormatOptions,
    ) -> FormatResult<Self> {
        let output = req.output_schema().map_err(FormatError::Other)?;
        let roots: Vec<&str> = output.fields().iter().map(|f| f.name().as_str()).collect();
        let guard = Guard::new(meta.calendars, meta.reader.parquet_schema(), &roots, o);
        let row_index = req.metadata_cols.row_index;
        let (mask, leaves) = projection_mask(meta, &req.projection);
        let mut selections = Vec::with_capacity(units.units.len());
        let mut planned = Vec::with_capacity(units.units.len());
        let mut pruned = 0u64;
        for u in &units.units {
            let rows = usize::try_from(u.rows)
                .map_err(|_| FormatError::Corrupt("row group too large".to_owned()))?;
            let selection = match &req.selection {
                None => None,
                Some(s) => {
                    let within = s.within(u.first_row..u.first_row + u.rows);
                    if within.is_empty() {
                        pruned += 1;
                        continue;
                    }
                    let ranges = within.into_iter().filter_map(|r| {
                        Some(usize::try_from(r.start).ok()?..usize::try_from(r.end).ok()?)
                    });
                    Some(ParquetSelection::from_consecutive_ranges(ranges, rows))
                }
            };
            selections.push(RowGroupSelection::new(u.index, selection));
            let rg = meta.reader.metadata().row_group(u.index);
            planned.push(
                leaves
                    .iter()
                    .filter_map(|&l| {
                        let c = rg.columns().get(l)?;
                        let start = u64::try_from(
                            c.dictionary_page_offset().unwrap_or(c.data_page_offset()),
                        )
                        .ok()?;
                        let len = u64::try_from(c.compressed_size()).ok()?;
                        Some(start..start + len)
                    })
                    .collect(),
            );
        }
        let reader = if row_index {
            let rn = Arc::new(
                Field::new("row_number", DataType::Int64, false).with_extension_type(RowNumber),
            );
            let options = meta
                .reader_options
                .clone()
                .with_virtual_columns(vec![rn])
                .map_err(|e| FormatError::Corrupt(e.to_string()))?;
            super::contain("planning", FormatError::Corrupt, || {
                ArrowReaderMetadata::try_new(Arc::clone(meta.reader.metadata()), options)
                    .map_err(|e| FormatError::Corrupt(e.to_string()))
            })?
        } else {
            meta.reader.clone()
        };
        let decoder = super::contain("planning", FormatError::Corrupt, || {
            ParquetPushDecoderBuilder::new_with_metadata(reader)
                .with_projection(mask)
                .with_batch_size(BATCH_ROWS)
                .with_row_group_selections(selections)
                .build()
                .map_err(|e| FormatError::Corrupt(e.to_string()))
        })?;
        let metrics = ScanUnitMetrics {
            units: planned.len() as u64,
            units_pruned: pruned,
            ..ScanUnitMetrics::default()
        };
        Ok(Self {
            decoder,
            planned,
            projection: req.projection.clone(),
            output,
            row_index,
            widen: spark_read_type(&DataType::Struct(meta.reader.schema().fields().clone()))
                != DataType::Struct(meta.reader.schema().fields().clone()),
            metrics,
            poisoned: false,
            guard,
        })
    }

    /// The decoded batch in the file-level output schema.
    fn finish(&self, batch: RecordBatch) -> FormatResult<RecordBatch> {
        let rows = batch.num_rows();
        let mut batch = batch;
        let row_number = if self.row_index {
            let last = batch.num_columns().checked_sub(1).ok_or_else(|| {
                FormatError::Corrupt("the decoder returned no row numbers".to_owned())
            })?;
            let c = Arc::clone(batch.column(last));
            batch.remove_column(last);
            Some(c)
        } else {
            None
        };
        if self.widen {
            let schema = batch.schema();
            let columns = batch
                .columns()
                .iter()
                .map(spark_read_array)
                .collect::<FormatResult<Vec<_>>>()?;
            let fields: Vec<Field> = schema
                .fields()
                .iter()
                .zip(&columns)
                .map(|(f, c)| f.as_ref().clone().with_data_type(c.data_type().clone()))
                .collect();
            batch = RecordBatch::try_new_with_options(
                Arc::new(arrow_schema::Schema::new_with_metadata(
                    fields,
                    schema.metadata().clone(),
                )),
                columns,
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .map_err(|e| FormatError::Corrupt(e.to_string()))?;
        }
        let projected = project_batch(&batch, &self.projection).map_err(FormatError::Other)?;
        let mut columns = projected.columns().to_vec();
        if let Some(rn) = row_number {
            columns.push(rn);
        }
        let out = RecordBatch::try_new_with_options(
            Arc::clone(&self.output),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|e| {
            FormatError::Corrupt(format!("decoded batch does not match the request: {e}"))
        })?;
        self.guard.check(&out)?;
        Ok(out)
    }
}

impl Decoder for ParquetDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if self.poisoned {
            return Err(FormatError::Corrupt(
                "the decoder failed earlier".to_owned(),
            ));
        }
        let decoder = &mut self.decoder;
        let result = super::contain("decoding", FormatError::Corrupt, || {
            decoder.try_decode().map_err(|e| decode_error(&e))
        });
        self.poisoned = result.is_err();
        match result? {
            DecodeResult::NeedsData(ranges) => Ok(Step::NeedData(ranges)),
            DecodeResult::Data(batch) => {
                self.metrics.rows += batch.num_rows() as u64;
                Ok(Step::Batch(self.finish(batch)?))
            }
            DecodeResult::Finished => Ok(Step::Done),
        }
    }

    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()> {
        let (ranges, bytes): (Vec<_>, Vec<_>) = data.into_iter().unzip();
        self.metrics.bytes_pushed += bytes.iter().map(|b| b.len() as u64).sum::<u64>();
        let decoder = &mut self.decoder;
        let result = super::contain("buffering", FormatError::Corrupt, || {
            decoder
                .push_ranges(ranges, bytes)
                .map_err(|e| FormatError::Corrupt(e.to_string()))
        });
        self.poisoned |= result.is_err();
        result
    }

    fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        let next = self
            .planned
            .len()
            .saturating_sub(self.decoder.row_groups_remaining());
        let mut out = Vec::new();
        let mut total = 0u64;
        for chunks in self.planned.iter().skip(next) {
            let size: u64 = chunks.iter().map(|r| r.end - r.start).sum();
            if total + size > max_bytes {
                break;
            }
            total += size;
            out.extend(chunks.iter().cloned());
        }
        out
    }

    fn buffered_bytes(&self) -> usize {
        usize::try_from(self.decoder.buffered_bytes()).unwrap_or(usize::MAX)
    }

    fn metrics(&self) -> ScanUnitMetrics {
        self.metrics
    }
}

/// `a` as [`spark_read_type`] types it.
///
/// # Errors
/// A millisecond timestamp beyond the microsecond range (Spark's
/// `multiplyExact` fails on it too).
fn spark_read_array(a: &ArrayRef) -> FormatResult<ArrayRef> {
    let bad = |e: arrow_schema::ArrowError| FormatError::Corrupt(e.to_string());
    Ok(match a.data_type() {
        DataType::Timestamp(TimeUnit::Millisecond, tz) => Arc::new(
            a.as_primitive::<TimestampMillisecondType>()
                .try_unary::<_, TimestampMicrosecondType, FormatError>(|ms| {
                    ms.checked_mul(1000).ok_or_else(|| {
                        FormatError::Corrupt(format!(
                            "the millisecond timestamp {ms} is out of the microsecond range"
                        ))
                    })
                })?
                .with_timezone_opt(tz.clone()),
        ) as ArrayRef,
        DataType::UInt8 => Arc::new(
            a.as_primitive::<UInt8Type>()
                .unary::<_, arrow_array::types::Int16Type>(i16::from),
        ) as ArrayRef,
        DataType::UInt16 => Arc::new(
            a.as_primitive::<UInt16Type>()
                .unary::<_, arrow_array::types::Int32Type>(i32::from),
        ),
        DataType::UInt32 => Arc::new(
            a.as_primitive::<UInt32Type>()
                .unary::<_, arrow_array::types::Int64Type>(i64::from),
        ),
        DataType::UInt64 => Arc::new(
            a.as_primitive::<UInt64Type>()
                .unary::<_, arrow_array::types::Decimal128Type>(i128::from)
                .with_precision_and_scale(20, 0)
                .map_err(bad)?,
        ),
        DataType::Struct(fields) => {
            let s = a.as_struct();
            let children = s
                .columns()
                .iter()
                .map(spark_read_array)
                .collect::<FormatResult<Vec<_>>>()?;
            let fields = fields
                .iter()
                .zip(&children)
                .map(|(f, c)| Arc::new(f.as_ref().clone().with_data_type(c.data_type().clone())))
                .collect::<Vec<_>>();
            Arc::new(
                StructArray::try_new_with_length(
                    fields.into(),
                    children,
                    s.nulls().cloned(),
                    s.len(),
                )
                .map_err(bad)?,
            )
        }
        DataType::List(f) => {
            let l = a.as_list::<i32>();
            let values = spark_read_array(l.values())?;
            let f = Arc::new(
                f.as_ref()
                    .clone()
                    .with_data_type(values.data_type().clone()),
            );
            Arc::new(
                ListArray::try_new(f, l.offsets().clone(), values, l.nulls().cloned())
                    .map_err(bad)?,
            )
        }
        DataType::LargeList(f) => {
            let l = a.as_list::<i64>();
            let values = spark_read_array(l.values())?;
            let f = Arc::new(
                f.as_ref()
                    .clone()
                    .with_data_type(values.data_type().clone()),
            );
            Arc::new(
                LargeListArray::try_new(f, l.offsets().clone(), values, l.nulls().cloned())
                    .map_err(bad)?,
            )
        }
        DataType::Map(f, sorted) => {
            let m = a.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = spark_read_array(&entries)?;
            let f = Arc::new(
                f.as_ref()
                    .clone()
                    .with_data_type(entries.data_type().clone()),
            );
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
