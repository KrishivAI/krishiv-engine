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
    Time32MillisecondType, Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampMillisecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
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
use super::row_filter::{CACHE_BYTES, LateMaterialization, Worth};
use crate::error::{FormatError, FormatResult};
use crate::format::{Decoder, ScanUnitMetrics, Step, UnitPlan};
use crate::options::FormatOptions;
use crate::request::{ColumnPath, PushdownExpr, RowFilterPolicy, ScanRequest, project_batch};
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
    /// The column chunk ranges of every unit planned at the start, sorted
    /// by start: what [`Self::check_page_headers`] looks pushed ranges up
    /// in (pruned units may stay; checking their chunks is harmless).
    chunk_ranges: Vec<Range<u64>>,
    meta: Arc<ParquetMetadata>,
    /// Runtime filters (file level) and the generation each had when the
    /// remaining units were last pruned with it.
    dynamic: Vec<(ScanFilter, u64)>,
    /// Whether the row filter may evaluate the runtime filters
    /// ([`crate::RowLevel::runtime_filters`]).
    dynamic_rows: bool,
    projection: Vec<ColumnPath>,
    output: SchemaRef,
    row_index: bool,
    widen: bool,
    metrics: ScanUnitMetrics,
    /// arrow-rs panicked: its state is not used again.
    poisoned: bool,
    /// Spark's calendar rules for the output columns.
    guard: Guard,
    /// Late materialisation (`row_filter.rs`).
    late: LateMaterialization,
    /// The projected leaves.
    leaves: Vec<usize>,
    /// The row filter's leaves while it is on (the only ones prefetched).
    late_leaves: Vec<usize>,
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
        let widen = spark_read_type(&DataType::Struct(meta.reader.schema().fields().clone()))
            != DataType::Struct(meta.reader.schema().fields().clone());
        let policy = match (req.row_filter, o.str("lateMaterialization")) {
            (RowFilterPolicy::Auto, Some("on")) => RowFilterPolicy::On,
            (RowFilterPolicy::Auto, Some("off")) => RowFilterPolicy::Off,
            (p, _) => p,
        };
        // Rows are dropped only by what the request allows at row level
        // (`RowLevel`); pruning below uses the whole predicate.
        let row_predicate = req.row_level.predicate(req.predicate.as_ref());
        let kept = row_predicate.and_then(|p| {
            let p = p.to_nnf();
            let (mut rows, mut kept) = (0f64, 0f64);
            for u in &planned {
                let rg = meta.row_groups.get(u.index)?;
                #[expect(clippy::cast_precision_loss, reason = "an estimate")]
                let n = rg.rows as f64;
                rows += n;
                kept += n * super::row_filter::estimate(&p, &rg.statistics)?;
            }
            (rows > 0.0).then(|| kept / rows)
        });
        let mut late = LateMaterialization::new(
            policy,
            row_predicate,
            widen.then_some(widen_column as super::row_filter::Widen),
            !row_index && guard.is_empty(),
            kept,
        )
        .resuming(req.row_filter_start);
        let dynamic: Vec<PushdownExpr> = req
            .dynamic
            .iter()
            .filter(|_| req.row_level.runtime_filters())
            .filter_map(crate::ScanFilter::predicate)
            .collect();
        let mut late_leaves = Vec::new();
        let row_filter = late.filter(late.is_on(), &dynamic, |cols| {
            predicate_mask(meta, cols, &leaves, &mut late_leaves, &planned)
        });
        let decoder = super::contain("planning", FormatError::Corrupt, || {
            ParquetPushDecoderBuilder::new_with_metadata(reader)
                .with_projection(mask)
                .with_batch_size(BATCH_ROWS)
                .with_row_group_selections(selections)
                .with_row_filter(row_filter)
                .with_max_predicate_cache_size(CACHE_BYTES)
                .build()
                .map_err(|e| FormatError::Corrupt(e.to_string()))
        })?;
        let metrics = ScanUnitMetrics {
            units: planned.len() as u64,
            units_pruned: pruned,
            ..ScanUnitMetrics::default()
        };
        let mut chunk_ranges: Vec<Range<u64>> = planned
            .iter()
            .flat_map(|u| u.chunks.iter().cloned())
            .collect();
        chunk_ranges.sort_by_key(|r| (r.start, r.end));
        Ok(Self {
            decoder: Some(decoder),
            reader: None,
            reading: 0,
            fetching: None,
            planned,
            chunk_ranges,
            meta: Arc::clone(meta),
            // The units were planned with these generations' filters.
            dynamic: req
                .dynamic
                .iter()
                .map(|d| (d.clone(), d.generation()))
                .collect(),
            dynamic_rows: req.row_level.runtime_filters(),
            projection: req.projection.clone(),
            output,
            row_index,
            widen,
            metrics,
            poisoned: false,
            guard,
            late,
            leaves,
            late_leaves,
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
        if !decoder.is_at_row_group_boundary() {
            return Ok(());
        }
        let changed = self
            .dynamic
            .iter()
            .any(|(f, seen)| f.filter().generation_hint() != *seen);
        if changed {
            self.late
                .runtime_filters_changed(&self.dynamic_predicates());
        }
        let late = self.late.want();
        if !changed {
            return self.rebuild(None, late, false);
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
        let selections = (self.planned.len() - start != before).then(|| {
            self.planned[start..]
                .iter()
                .map(|u| RowGroupSelection::new(u.index, u.selection.clone()))
                .collect()
        });
        self.rebuild(selections, late, true)
    }

    /// Rebuilds the decoder at a row-group boundary when the remaining row
    /// groups changed (`selections`) or the row filter should be switched
    /// or refreshed (`late`, and a new runtime-filter generation:
    /// `regenerated`; see `row_filter.rs`).
    fn rebuild(
        &mut self,
        selections: Option<Vec<RowGroupSelection>>,
        late: bool,
        regenerated: bool,
    ) -> FormatResult<()> {
        let refresh = late != self.late.is_on() || (late && regenerated);
        if selections.is_none() && !refresh {
            return Ok(());
        }
        let decoder = self
            .decoder
            .take()
            .ok_or_else(|| FormatError::Corrupt("the decoder failed earlier".to_owned()))?;
        let filter = refresh.then(|| {
            let dynamic = self.dynamic_predicates();
            let (meta, leaves, picked, planned) = (
                &self.meta,
                &self.leaves,
                &mut self.late_leaves,
                &self.planned,
            );
            let f = self.late.filter(late, &dynamic, |cols| {
                predicate_mask(meta, cols, leaves, picked, planned)
            });
            if !self.late.is_on() {
                picked.clear();
            }
            f
        });
        let rebuilt = super::contain("planning", FormatError::Corrupt, || {
            decoder
                .into_builder()
                .and_then(|mut b| {
                    if let Some(s) = selections {
                        b = b.with_row_group_selections(s);
                    }
                    if let Some(f) = filter {
                        b = b.with_row_filter(f);
                    }
                    b.build()
                })
                .map_err(|e| FormatError::Corrupt(e.to_string()))
        })?;
        self.decoder = Some(rebuilt);
        Ok(())
    }

    /// The current runtime filters as file-level predicates, for the row
    /// filter (none when the request does not allow them at row level).
    fn dynamic_predicates(&self) -> Vec<PushdownExpr> {
        self.dynamic
            .iter()
            .filter(|_| self.dynamic_rows)
            .filter_map(|(f, _)| f.predicate())
            .collect()
    }

    /// Walks the page headers of every planned column chunk that `ranges`
    /// hold before arrow-rs reads them ([`super::thrift_guard::check_pages`]).
    ///
    /// The push decoder asks for whole column chunks: the footer is read
    /// without the page index (`PageIndexPolicy::Skip` in `meta.rs`), and
    /// only an offset index lets arrow-rs fetch part of a chunk. Loading the
    /// page index would need this check to handle partial ranges.
    fn check_page_headers(&self, ranges: &[Range<u64>], bytes: &[Bytes]) -> FormatResult<()> {
        for (range, held) in ranges.iter().zip(bytes) {
            let first = self.chunk_ranges.partition_point(|c| c.start < range.start);
            let inside = self.chunk_ranges[first..]
                .iter()
                .take_while(|c| c.start < range.end)
                .filter(|c| c.end <= range.end);
            for c in inside {
                let at = |offset: u64| usize::try_from(offset - range.start).ok();
                if let Some(chunk) = at(c.start)
                    .zip(at(c.end))
                    .and_then(|(from, to)| held.get(from..to))
                {
                    super::thrift_guard::check_pages(chunk).map_err(|why| {
                        FormatError::Corrupt(format!("a page header is malformed: {why}"))
                    })?;
                }
            }
        }
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
                        if self.late.sampling() {
                            let dynamic = self.dynamic_predicates();
                            self.late.observe(&batch, &dynamic);
                        }
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
                    self.late.start_row_group();
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
        if let Err(e) = self.check_page_headers(&ranges, &bytes) {
            self.poisoned = true;
            return Err(e);
        }
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
        // With the filter on by default (nothing measured yet), the row
        // group in flight asks for the filter's chunks first: fetch its
        // other chunks meanwhile, as without the filter.
        if self.late.is_guess()
            && self.reader.is_none()
            && !decoder.is_at_row_group_boundary()
            && let Some(u) = next.checked_sub(1).and_then(|i| self.planned.get(i))
        {
            let rg = self.meta.reader.metadata().row_group(u.index);
            for l in &self.leaves {
                if self.late_leaves.binary_search(l).is_err()
                    && let Some(r) = chunk(rg, *l)
                    && total + (r.end - r.start) <= max_bytes
                {
                    total += r.end - r.start;
                    out.push(r);
                }
            }
        }
        // With the filter on after a selective row group, a row group's
        // other chunks are fetched only if some row passes (none for most
        // groups of a selective filter). Otherwise every chunk is prefetched,
        // as the next row group may be read without the filter.
        if self.late.is_on() && !self.late.is_guess() {
            for u in self.planned.iter().skip(next) {
                let rg = self.meta.reader.metadata().row_group(u.index);
                let chunks: Vec<Range<u64>> = self
                    .late_leaves
                    .iter()
                    .filter_map(|&l| chunk(rg, l))
                    .collect();
                let size: u64 = chunks.iter().map(|r| r.end - r.start).sum();
                if total + size > max_bytes {
                    break;
                }
                total += size;
                out.extend(chunks);
            }
            return out;
        }
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

    fn row_filter_measured(&self) -> Option<bool> {
        self.late.measurement()
    }

    fn metrics(&self) -> ScanUnitMetrics {
        let (units_filtered, units_unfiltered, row_filter_rows, row_filter_kept) =
            self.late.report();
        ScanUnitMetrics {
            units_filtered,
            units_unfiltered,
            row_filter_rows,
            row_filter_kept,
            ..self.metrics
        }
    }
}

/// The byte range of leaf `l`'s column chunk in `rg` (as the planned
/// units' chunks).
fn chunk(rg: &parquet::file::metadata::RowGroupMetaData, l: usize) -> Option<Range<u64>> {
    let c = rg.columns().get(l)?;
    let start = u64::try_from(c.dictionary_page_offset().unwrap_or(c.data_page_offset())).ok()?;
    let len = u64::try_from(c.compressed_size()).ok()?;
    Some(start..start + len)
}

/// The filter pays only where the projected columns outside the predicate
/// hold at least this many times the compressed bytes of those inside it:
/// the predicate columns are decoded a second time for the rows that pass
/// (arrow-rs's predicate cache is off, see `row_filter.rs`), and with the
/// filter wrongly on for a row group that costs that much again.
const SIZE_RATIO: u64 = 2;

/// The predicate's projection over `cols` (top-level columns), stored in
/// `picked` as sorted leaves, and whether filtering may pay: some projected
/// leaf lies outside it, holding at least [`SIZE_RATIO`] times the
/// predicate leaves' compressed bytes in the `planned` row groups.
fn predicate_mask(
    meta: &ParquetMetadata,
    cols: &[ColumnPath],
    leaves: &[usize],
    picked: &mut Vec<usize>,
    planned: &[Unit],
) -> (ProjectionMask, Worth) {
    let (mask, p) = projection_mask(meta, cols);
    let (mut inside, mut outside) = (0u64, 0u64);
    for u in planned {
        let rg = meta.reader.metadata().row_group(u.index);
        for &l in leaves {
            let size = chunk(rg, l).map_or(0, |r| r.end - r.start);
            if p.binary_search(&l).is_ok() {
                inside = inside.saturating_add(size);
            } else {
                outside = outside.saturating_add(size);
            }
        }
    }
    let worth = if leaves.iter().all(|l| p.binary_search(l).is_ok()) {
        Worth::Never
    } else if outside >= inside.saturating_mul(SIZE_RATIO) {
        Worth::Yes
    } else {
        Worth::Small
    };
    *picked = p;
    (mask, worth)
}

/// [`spark_read_array`] for the row filter: a column it cannot widen is
/// not evaluated there.
fn widen_column(a: &ArrayRef) -> Option<ArrayRef> {
    spark_read_array(a).ok()
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
        DataType::Time32(TimeUnit::Millisecond) => Arc::new(
            a.as_primitive::<Time32MillisecondType>()
                .unary::<_, Time64NanosecondType>(|ms| i64::from(ms) * 1_000_000),
        ) as ArrayRef,
        DataType::Time64(TimeUnit::Microsecond) => Arc::new(
            a.as_primitive::<Time64MicrosecondType>()
                .try_unary::<_, Time64NanosecondType, FormatError>(|us| {
                    us.checked_mul(1000).ok_or_else(|| {
                        FormatError::Corrupt(format!("the TIME value {us} is out of range"))
                    })
                })?,
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
