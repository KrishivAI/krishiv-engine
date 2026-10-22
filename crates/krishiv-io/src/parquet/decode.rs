// SPDX-License-Identifier: Apache-2.0
//! The Parquet [`Decoder`]: arrow-rs's `ParquetPushDecoder` (sans-IO) behind
//! the framework's step/push interface. The framework fetches what it asks
//! for on the I/O runtime; decoding runs where `step` is called (a compute
//! worker), one batch per step.
//!
//! The decoder hands out one row group's reader at a time, so between row
//! groups it sits at a clean boundary: there, when a runtime filter
//! published a new generation since the units were planned, the row groups
//! not started yet are pruned again with it and the push decoder is
//! rebuilt over the survivors (its buffered bytes carry over).
//!
//! **Memory.** [`Decoder::buffered_bytes`] covers the push decoder's
//! buffered column chunks; [`Decoder::working_bytes`] the row group being
//! decoded: once the push decoder hands out a row group's reader it no
//! longer counts that row group's column chunks, but the reader holds them
//! until it ends, plus one decompressed page per column and the batch being
//! built. From when the reader is handed out until it ends it is charged its
//! row group's compressed chunks (exact) and [`Unit::decode`], an estimate of
//! the rest: per column `min(uncompressed chunk, PAGE_ESTIMATE)` plus one
//! batch of decoded rows.

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
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReader, RowSelection as ParquetSelection,
};
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
use crate::runtime_filter::ScanFilter;
use crate::stats::{TriState, prune};

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

/// A planned row group.
struct Unit {
    index: usize,
    selection: Option<ParquetSelection>,
    /// Projected column chunk ranges.
    chunks: Vec<Range<u64>>,
    /// Compressed bytes of the row group.
    bytes: u64,
    /// Bytes of `chunks`: what the row group's reader holds compressed.
    chunk_bytes: usize,
    /// What decoding the row group holds on top of its chunks (module
    /// docs).
    decode: usize,
}

/// Decompressed bytes one column holds per page while it is decoded: the
/// common writers' data page size limit (parquet-rs, parquet-mr and
/// DuckDB all default to 1 MiB pages), capped by the column chunk's own
/// uncompressed size.
const PAGE_ESTIMATE: u64 = 1 << 20;

/// [`Unit::decode`] of row group `rg` over the projected `leaves`: a
/// decompressed page per column and one output batch (`BATCH_ROWS` rows at
/// the row group's average uncompressed row width). The sizes come from the
/// footer (untrusted): saturating, and only ever an estimate to reserve.
fn decode_bytes(rg: &parquet::file::metadata::RowGroupMetaData, leaves: &[usize]) -> usize {
    let mut pages = 0u64;
    let mut uncompressed = 0u64;
    for c in leaves.iter().filter_map(|&l| rg.columns().get(l)) {
        let unc = u64::try_from(c.uncompressed_size()).unwrap_or(0);
        pages = pages.saturating_add(unc.min(PAGE_ESTIMATE));
        uncompressed = uncompressed.saturating_add(unc);
    }
    let rows = u64::try_from(rg.num_rows()).unwrap_or(0).max(1);
    let batch_rows = (BATCH_ROWS as u64).min(rows);
    let batch = (uncompressed / rows).saturating_mul(batch_rows);
    usize::try_from(pages.saturating_add(batch)).unwrap_or(usize::MAX)
}

/// See the module docs.
pub(crate) struct ParquetDecoder {
    /// `None` only while it is rebuilt.
    decoder: Option<ParquetPushDecoder>,
    /// The current row group's reader.
    reader: Option<ParquetRecordBatchReader>,
    /// What the current reader holds: its row group's chunk bytes and
    /// decoding estimate (0 without a reader).
    reading: usize,
    /// The row group (file index) whose chunks the decoder asked for last:
    /// the push decoder asks for every row group's chunks before it hands
    /// out that row group's reader.
    fetching: Option<usize>,
    /// Planned row groups, in order.
    planned: Vec<Unit>,
    meta: Arc<ParquetMetadata>,
    /// Runtime filters (file level) and the generation each had when the
    /// remaining units were last pruned with it.
    dynamic: Vec<(ScanFilter, u64)>,
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
            let rg = meta.reader.metadata().row_group(u.index);
            let chunks: Vec<Range<u64>> = leaves
                .iter()
                .filter_map(|&l| {
                    let c = rg.columns().get(l)?;
                    let start =
                        u64::try_from(c.dictionary_page_offset().unwrap_or(c.data_page_offset()))
                            .ok()?;
                    let len = u64::try_from(c.compressed_size()).ok()?;
                    Some(start..start.checked_add(len)?)
                })
                .collect();
            let chunk_bytes = chunks
                .iter()
                .fold(0u64, |n, r| n.saturating_add(r.end - r.start));
            planned.push(Unit {
                index: u.index,
                selection,
                chunks,
                bytes: u.bytes.end - u.bytes.start,
                chunk_bytes: usize::try_from(chunk_bytes).unwrap_or(usize::MAX),
                decode: decode_bytes(rg, &leaves),
            });
        }
        let selections = planned
            .iter()
            .map(|u| RowGroupSelection::new(u.index, u.selection.clone()))
            .collect();
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
            decoder: Some(decoder),
            reader: None,
            reading: 0,
            fetching: None,
            planned,
            meta: Arc::clone(meta),
            // The units were planned with these generations' filters.
            dynamic: req
                .dynamic
                .iter()
                .map(|d| (d.clone(), d.generation()))
                .collect(),
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

impl ParquetDecoder {
    fn decoder(&mut self) -> FormatResult<&mut ParquetPushDecoder> {
        self.decoder
            .as_mut()
            .ok_or_else(|| FormatError::Corrupt("the decoder failed earlier".to_owned()))
    }

    /// At a row-group boundary, prunes the row groups not started yet with
    /// the runtime filters that changed since they were planned (see the
    /// module docs).
    fn prune_remaining(&mut self) -> FormatResult<()> {
        let Some(decoder) = self.decoder.as_ref() else {
            return Ok(());
        };
        let changed = self
            .dynamic
            .iter()
            .any(|(f, seen)| f.filter().generation_hint() != *seen);
        if !changed || !decoder.is_at_row_group_boundary() {
            return Ok(());
        }
        let remaining = decoder.row_groups_remaining();
        let start = self.planned.len().saturating_sub(remaining);
        let mut predicates = Vec::new();
        for (f, seen) in &mut self.dynamic {
            *seen = f.generation();
            if let Some(p) = f.predicate() {
                predicates.push((p, Arc::clone(f.filter())));
            }
        }
        let rows = &self.meta.row_groups;
        let pruned_by = |u: &Unit| {
            let stats = &rows.get(u.index)?.statistics;
            predicates
                .iter()
                .find(|(p, _)| prune(p, stats) == TriState::False)
                .map(|(_, f)| f)
        };
        let tail = self.planned.split_off(start);
        let before = tail.len();
        for u in tail {
            match pruned_by(&u) {
                Some(f) => {
                    f.stats().add_unit(u.bytes);
                    self.metrics.units_pruned += 1;
                }
                None => self.planned.push(u),
            }
        }
        if self.planned.len() - start == before {
            return Ok(());
        }
        let selections = self.planned[start..]
            .iter()
            .map(|u| RowGroupSelection::new(u.index, u.selection.clone()))
            .collect();
        let decoder = self
            .decoder
            .take()
            .ok_or_else(|| FormatError::Corrupt("the decoder failed earlier".to_owned()))?;
        let rebuilt = super::contain("planning", FormatError::Corrupt, || {
            decoder
                .into_builder()
                .and_then(|b| b.with_row_group_selections(selections).build())
                .map_err(|e| FormatError::Corrupt(e.to_string()))
        })?;
        self.decoder = Some(rebuilt);
        Ok(())
    }

    fn step_inner(&mut self) -> FormatResult<Step> {
        loop {
            if let Some(reader) = self.reader.as_mut() {
                let next = super::contain("decoding", FormatError::Corrupt, || {
                    reader
                        .next()
                        .transpose()
                        .map_err(|e| decode_error(&ParquetError::from(e)))
                })?;
                match next {
                    Some(batch) => {
                        self.metrics.rows += batch.num_rows() as u64;
                        return Ok(Step::Batch(self.finish(batch)?));
                    }
                    None => {
                        self.reader = None;
                        self.reading = 0;
                        continue;
                    }
                }
            }
            self.prune_remaining()?;
            let decoder = self.decoder()?;
            let result = super::contain("decoding", FormatError::Corrupt, || {
                decoder.try_next_reader().map_err(|e| decode_error(&e))
            })?;
            match result {
                DecodeResult::NeedsData(ranges) => {
                    let unit = ranges.first().and_then(|r| {
                        self.planned.iter().find(|u| {
                            u.chunks
                                .iter()
                                .any(|c| c.start <= r.start && r.start < c.end)
                        })
                    });
                    self.fetching = unit.map(|u| u.index);
                    return Ok(Step::NeedData(ranges));
                }
                DecodeResult::Data(reader) => {
                    // No chunk was asked for (no column projected): the
                    // reader holds no column data.
                    let index = self.fetching.take();
                    self.reading = index
                        .and_then(|i| self.planned.iter().find(|u| u.index == i))
                        .map_or(0, |u| u.chunk_bytes.saturating_add(u.decode));
                    self.reader = Some(reader);
                }
                DecodeResult::Finished => return Ok(Step::Done),
            }
        }
    }
}

impl Decoder for ParquetDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if self.poisoned {
            return Err(FormatError::Corrupt(
                "the decoder failed earlier".to_owned(),
            ));
        }
        let result = self.step_inner();
        self.poisoned = result.is_err();
        result
    }

    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()> {
        let (ranges, bytes): (Vec<_>, Vec<_>) = data.into_iter().unzip();
        self.metrics.bytes_pushed += bytes.iter().map(|b| b.len() as u64).sum::<u64>();
        let decoder = self.decoder()?;
        let result = super::contain("buffering", FormatError::Corrupt, || {
            decoder
                .push_ranges(ranges, bytes)
                .map_err(|e| FormatError::Corrupt(e.to_string()))
        });
        self.poisoned |= result.is_err();
        result
    }

    fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        let Some(decoder) = self.decoder.as_ref() else {
            return Vec::new();
        };
        let next = self
            .planned
            .len()
            .saturating_sub(decoder.row_groups_remaining());
        let mut out = Vec::new();
        let mut total = 0u64;
        for chunks in self.planned.iter().skip(next).map(|u| &u.chunks) {
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
        self.decoder.as_ref().map_or(0, |d| {
            usize::try_from(d.buffered_bytes()).unwrap_or(usize::MAX)
        })
    }

    fn working_bytes(&self) -> usize {
        self.reading
    }

    fn next_decode_bytes(&self) -> usize {
        self.fetching
            .and_then(|i| self.planned.iter().find(|u| u.index == i))
            .map_or(0, |u| u.decode)
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
