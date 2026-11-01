// SPDX-License-Identifier: Apache-2.0
//! The file scan: [`FileScan`] (what the planner pushes into, over any
//! [`FileFormat`]) and the [`FileScanSource`] it builds.
//!
//! - **Morsels are pieces of Spark's file partitions**
//!   ([`krishiv_io::split::morsels`]): about four per driver, each a run of
//!   one partition's splits or a byte range of one split, in partition
//!   order. Drivers claim them as they finish, so a filter that keeps
//!   one stretch of a sorted file (the row groups of a few dates) is read
//!   by every driver, not by the two or three that own that stretch's
//!   split. A driver reads a morsel's pieces in order, each through a
//!   [`ScanStream`] (metadata from the cache, schema mapping, the format's
//!   sans-IO decoder, fetches and prefetch through the I/O scheduler); a
//!   piece reads the units its range owns, as a split does, so the rows
//!   and their order are those of the partitions.
//! - **Footers are pipelined**: when a driver opens a split it starts
//!   reading the footer of the next split's file on the I/O runtime
//!   ([`IoContext::prefetch_metadata`]), so it is cached by the time that
//!   file is opened.
//! - **Pushdown** ([`ScanPushdown`]) becomes the immutable table-level
//!   [`ScanRequest`]: nested projection, the typed predicates of the pushed
//!   filters, a limit hint, per-file row selections, field mapping and
//!   `_metadata` columns. Pruning keeps a superset of the matching rows, so
//!   pushed filters are never [`FilterPushdown::Exact`]: the `Filter` above
//!   the scan stays. When a conjunct of a pushed filter may raise, rows are
//!   dropped only by the conjuncts written before it and no runtime filter
//!   applies ([`RowLevel`]): the `Filter` sees every row it would raise on.
//! - **Runtime filters** (design P2-T163), attached after planning, join
//!   the request when the scan starts. A split whose filter drops every row
//!   (an empty join build) is skipped without opening its file; the format
//!   prunes row groups with them (re-read at row-group boundaries); each
//!   driver's row stage drops rows from the mapped batches, with its
//!   adaptive switch-off.
//! - **`_metadata`** columns are one `_metadata` struct column appended
//!   after the projection, with Spark's field names, types and order
//!   (`file_path`, `file_name`, `file_size`, `file_block_start`,
//!   `file_block_length`, `file_modification_time`, `row_index`); the file
//!   constants are built once per batch without copying the strings per
//!   row, `row_index` comes from the decoder.
//! - **Memory**: each split's stream reserves its decoder buffers and
//!   prefetch through the scan's consumer in the driver; prefetch is only
//!   ever `try_grow`n, so it never forces arbitration.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::builder::make_view;
use arrow_array::{
    ArrayRef, Int64Array, RecordBatch, RecordBatchOptions, StringViewArray, StructArray,
    TimestampMicrosecondArray,
};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_io::split::{FilePartition, MORSELS_PER_DRIVER, MorselSplit, SplitConfig, SplitFile};
use krishiv_io::{
    ByteRange, ColumnPath, FileFormat, FileMetadata, FileRef, FormatOptions, IoContext,
    MetadataCols, PushdownExpr, ROW_INDEX_COLUMN, RowLevel, ScanOptions, ScanRequest, ScanStream,
    ScanUnitMetrics, SchemaMapper,
};
use krishiv_memory::ConsumerHandle;
use krishiv_types::batch::Batch;

use super::scan::{FileSelections, FilterPushdown, PushedFilter, ScanPushdown, ScanPushdowns};
use super::{global_ref, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::morsel::MorselCursor;
use crate::operator::{GlobalState, LocalState, Source};

/// Name of the `_metadata` struct column (Spark's).
pub const METADATA_COLUMN: &str = "_metadata";

/// How a file scan finds its partitions.
#[derive(Debug, Clone)]
enum Partitions {
    /// Spark's split planning over the listed files.
    Plan(SplitConfig),
    /// Given (a distributed task's partitions, tests).
    Given(Arc<[FilePartition]>),
}

/// A scan of listed files of one format, before pushdown (see the module
/// docs). [`ScanPushdown::build`] makes its [`FileScanSource`].
#[derive(Clone)]
pub struct FileScan {
    format: Arc<dyn FileFormat>,
    io: IoContext,
    table: SchemaRef,
    files: Arc<[FileRef]>,
    format_options: FormatOptions,
    scan_options: ScanOptions,
    partitions: Partitions,
    /// The partition columns (the last columns of `table`): constant per
    /// file, read from the `name=value` directories above it, never from
    /// the file.
    partition_columns: Arc<[FieldRef]>,
    /// The smallest morsel: `spark.sql.files.openCostInBytes`.
    min_morsel_bytes: u64,
}

impl std::fmt::Debug for FileScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileScan")
            .field("format", &self.format.name())
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

impl FileScan {
    /// A scan of `files` (in listing order) as the table `table`, planned
    /// into partitions with `splits` (Spark's file split planning).
    #[must_use]
    pub fn new(
        format: Arc<dyn FileFormat>,
        io: IoContext,
        table: SchemaRef,
        files: Vec<FileRef>,
        format_options: FormatOptions,
        scan_options: ScanOptions,
        splits: SplitConfig,
    ) -> Self {
        Self {
            format,
            io,
            table,
            files: files.into(),
            format_options,
            scan_options,
            partitions: Partitions::Plan(splits),
            partition_columns: Arc::from([]),
            min_morsel_bytes: splits.open_cost_in_bytes,
        }
    }

    /// The same scan with `columns` as its partition columns: the last
    /// columns of the table, whose values are the `name=value` directories
    /// above each file (Spark's partition discovery).
    #[must_use]
    pub fn with_partition_columns(mut self, columns: Vec<FieldRef>) -> Self {
        self.partition_columns = columns.into();
        self
    }

    /// The same scan reading exactly `partitions` (split file indexes refer
    /// to the listed files), as a distributed task does.
    ///
    /// Prefer [`Self::with_file_splits`] when the files differ too: split
    /// file indexes always refer to the files the scan was given.
    #[must_use]
    pub fn with_partitions(mut self, partitions: Vec<FilePartition>) -> Self {
        self.partitions = Partitions::Given(partitions.into());
        self
    }

    /// The same scan reading exactly `files` in `partitions` (split file
    /// indexes refer to `files`), as a distributed task does: the listed
    /// files with their versions pinned.
    #[must_use]
    pub fn with_file_splits(mut self, files: Vec<FileRef>, partitions: Vec<FilePartition>) -> Self {
        self.files = files.into();
        self.partitions = Partitions::Given(partitions.into());
        self
    }

    /// The source reading `pushdowns` ([`ScanPushdown::build`] without the
    /// type erasure).
    ///
    /// # Errors
    /// As [`ScanPushdown::build`].
    pub fn source(&self, pushdowns: ScanPushdowns) -> Result<FileScanSource> {
        FileScanSource::new(self, pushdowns)
    }

    /// The listed files, in listing order (split file indexes refer to
    /// them).
    #[must_use]
    pub fn files(&self) -> &[FileRef] {
        &self.files
    }

    /// The format the files are read with.
    #[must_use]
    pub fn format(&self) -> &Arc<dyn FileFormat> {
        &self.format
    }

    /// The parsed read options.
    #[must_use]
    pub fn format_options(&self) -> &FormatOptions {
        &self.format_options
    }

    fn plan(&self) -> Arc<[FilePartition]> {
        match &self.partitions {
            Partitions::Given(p) => Arc::clone(p),
            Partitions::Plan(cfg) => {
                let caps = self.format.capabilities(&self.format_options);
                let files: Vec<SplitFile> = self
                    .files
                    .iter()
                    .map(|f| SplitFile {
                        size: f.size,
                        splittable: caps.is_splittable(f.path.as_ref()),
                    })
                    .collect();
                krishiv_io::split::plan(cfg, &files).into()
            }
        }
    }
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

impl ScanPushdown for FileScan {
    fn table_schema(&self) -> SchemaRef {
        Arc::clone(&self.table)
    }

    fn filter_pushdown(&self, filter: &PushedFilter) -> FilterPushdown {
        if reads_partition_column(self, filter) {
            return FilterPushdown::Unsupported;
        }
        // Pruning keeps a superset of the matching rows; the row evaluator
        // is never run by the scan.
        if filter.expr.is_some() {
            FilterPushdown::Inexact
        } else {
            FilterPushdown::Unsupported
        }
    }

    fn supports_selection(&self) -> bool {
        self.format.capabilities(&self.format_options).selection
    }

    fn build(&self, pushdowns: ScanPushdowns) -> Result<Arc<dyn Source>> {
        Ok(Arc::new(self.source(pushdowns)?))
    }

    fn with_file_splits(
        &self,
        files: Vec<FileRef>,
        partitions: Vec<FilePartition>,
    ) -> Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(self.clone().with_file_splits(files, partitions)))
    }

    fn output_schema(&self, pushdowns: &ScanPushdowns) -> Result<SchemaRef> {
        check_capabilities(self, pushdowns)?;
        Ok(table_request(self, pushdowns)?.output)
    }
}

/// Counters of a file scan, summed over every split read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileScanMetrics {
    /// Splits opened.
    pub splits: u64,
    /// Units (row groups) decoders were opened with, after statistics pruning.
    pub units: u64,
    /// Units skipped by dynamic filters or dictionaries.
    pub units_pruned: u64,
    /// Pages skipped by page indexes.
    pub pages_pruned: u64,
    /// Rows decoded.
    pub rows: u64,
    /// Bytes pushed into decoders.
    pub bytes: u64,
    /// Bytes of decode working sets taken without a reservation: the pool
    /// was full and nothing could be spilled for them
    /// (`krishiv_io::ScanGate`).
    pub decode_overcommit_bytes: u64,
    /// Units decoded with the row filter (predicate columns first).
    pub units_filtered: u64,
    /// Units decoded without it.
    pub units_unfiltered: u64,
    /// Rows the row filter evaluated while decoding.
    pub row_filter_rows: u64,
    /// Rows it kept.
    pub row_filter_kept: u64,
}

impl FileScanMetrics {
    /// The `EXPLAIN ANALYZE` counters, once a split was read: units
    /// (row groups) decoded and skipped, rows and bytes decoded, and how
    /// the row filter ran (`row_filter_*` only where it did); the decode
    /// over-commit when there was any. Page-index pruning is not counted:
    /// the Parquet reader does not read page indexes.
    fn extra(&self) -> crate::metrics::ExtraMetrics {
        let mut out = Vec::new();
        if self.splits > 0 {
            out = crate::metrics::counts([
                ("scan_splits", self.splits),
                ("scan_units", self.units),
                ("scan_units_pruned", self.units_pruned),
                ("scan_rows_decoded", self.rows),
                ("scan_bytes", self.bytes),
            ]);
        }
        if self.units_filtered > 0 || self.row_filter_rows > 0 {
            out.extend(crate::metrics::counts([
                ("row_filter_units", self.units_filtered),
                ("row_filter_units_skipped", self.units_unfiltered),
                ("row_filter_rows", self.row_filter_rows),
                ("row_filter_kept", self.row_filter_kept),
            ]));
        }
        if self.decode_overcommit_bytes > 0 {
            out.extend(crate::metrics::counts([(
                "decode_overcommit_bytes",
                self.decode_overcommit_bytes,
            )]));
        }
        out
    }

    fn add(&mut self, m: &ScanUnitMetrics, overcommitted: u64) {
        self.decode_overcommit_bytes += overcommitted;
        self.splits += 1;
        self.units += m.units;
        self.units_pruned += m.units_pruned;
        self.pages_pruned += m.pages_pruned;
        self.rows += m.rows;
        self.bytes += m.bytes_pushed;
        self.units_filtered += m.units_filtered;
        self.units_unfiltered += m.units_unfiltered;
        self.row_filter_rows += m.row_filter_rows;
        self.row_filter_kept += m.row_filter_kept;
    }
}

/// The source of a file scan (see the module docs).
pub struct FileScanSource {
    format: Arc<dyn FileFormat>,
    io: IoContext,
    files: Arc<[FileRef]>,
    partitions: Arc<[FilePartition]>,
    min_morsel_bytes: u64,
    /// Table level, immutable.
    request: ScanRequest,
    selection: Option<Arc<dyn FileSelections>>,
    format_options: FormatOptions,
    scan_options: ScanOptions,
    output: SchemaRef,
    /// The decoded batches' schema (`output` before `_metadata`).
    mapped: SchemaRef,
    /// The partition columns as the table types them.
    partition_fields: Arc<[FieldRef]>,
    /// Where each column of `mapped` comes from; empty without partition
    /// columns (the decoded batch is `mapped`).
    layout: Vec<Slot>,
    runtime: super::RuntimeFilters,
    metrics: Mutex<FileScanMetrics>,
}

impl std::fmt::Debug for FileScanSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileScanSource")
            .field("format", &self.format.name())
            .field("partitions", &self.partitions.len())
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

struct FileScanGlobal {
    /// The morsels, in partition order.
    morsels: Vec<Vec<MorselSplit>>,
    cursor: MorselCursor,
    /// The table-level request with the runtime filters attached.
    request: ScanRequest,
    /// Rows returned by every driver (for a limit hint).
    returned: AtomicU64,
}

/// The piece being read.
struct Current {
    stream: ScanStream,
    file: usize,
    /// The planned split the piece belongs to (for `_metadata`).
    split: ByteRange,
    /// The file's partition values, one row each, as the partition fields
    /// type them.
    partition_values: Vec<ArrayRef>,
}

type MetadataFuture = BoxFuture<'static, krishiv_io::Result<Arc<dyn FileMetadata>>>;

struct FileScanLocal {
    consumer: Arc<ConsumerHandle>,
    /// The query's scan decode gate.
    gate: Arc<krishiv_io::ScanGate>,
    /// The claimed morsel and its next piece.
    morsel: Option<(usize, usize)>,
    current: Option<Current>,
    /// The footer read in flight for a file this driver opens next.
    ahead: Option<(usize, MetadataFuture)>,
    claimed: u64,
    /// The claim ordinal of the claimed morsel.
    position: Option<u64>,
    /// What the last piece read measured for its row-level filter, and its
    /// file: the next piece of that file starts from it instead of guessing
    /// again.
    row_filter: Option<(usize, bool)>,
    rows: super::RowStage,
}

/// Rejects pushdowns the format cannot honour (`build` and
/// `output_schema` alike).
fn check_capabilities(scan: &FileScan, p: &ScanPushdowns) -> Result<()> {
    let caps = scan.format.capabilities(&scan.format_options);
    if p.selection.is_some() && !caps.selection {
        return Err(unsupported(format!(
            "row selections over the {} format",
            scan.format.name()
        )));
    }
    if p.metadata.row_index && !caps.row_index {
        return Err(unsupported(format!(
            "_metadata.row_index over the {} format",
            scan.format.name()
        )));
    }
    Ok(())
}

/// Whether the typed tree of `filter` reads a partition column of `scan`
/// (a filter without one prunes nothing).
fn reads_partition_column(scan: &FileScan, filter: &PushedFilter) -> bool {
    if scan.partition_columns.is_empty() {
        return false;
    }
    filter.expr.as_ref().is_some_and(|e| {
        e.columns()
            .iter()
            .any(|c| scan.partition_columns.iter().any(|p| p.name() == c.root()))
    })
}

/// Where a column of a partitioned scan's output comes from.
#[derive(Clone, Copy, Debug)]
enum Slot {
    /// The decoded batch's column.
    Data(usize),
    /// The partition column (index into the scan's partition fields).
    Partition(usize),
}

/// What [`table_request`] decides for a scan.
struct TableRequest {
    request: ScanRequest,
    /// The output before `_metadata`, partition columns included.
    mapped: SchemaRef,
    /// The output with `_metadata`.
    output: SchemaRef,
    partition_fields: Arc<[FieldRef]>,
    layout: Vec<Slot>,
}

/// The table-level request of a scan of `scan` with `p`, the schema its
/// columns map to (the mapped output does not depend on the file: it maps
/// onto the table) and the output schema (with `_metadata`).
///
/// Partition columns are not in the files: the format is asked for the
/// data columns only (a CSV reader would otherwise find every record
/// short), and the partition columns are added from each file's path.
fn table_request(scan: &FileScan, p: &ScanPushdowns) -> Result<TableRequest> {
    let projection: Vec<ColumnPath> = p.projection.clone().unwrap_or_else(|| {
        scan.table
            .fields()
            .iter()
            .map(|f| ColumnPath::new(f.name()))
            .collect()
    });
    // The partition columns with the types the table (the analyzer's
    // binding) gives them, which may differ from this listing's.
    let partition_fields: Arc<[FieldRef]> = scan
        .partition_columns
        .iter()
        .map(|f| {
            scan.table
                .field_with_name(f.name())
                .map_or_else(|_| Arc::clone(f), |t| Arc::new(t.clone()))
        })
        .collect();
    let partition_of = |c: &ColumnPath| {
        partition_fields
            .iter()
            .position(|f| c.0.len() == 1 && f.name() == c.root())
    };
    let data_table: SchemaRef = if partition_fields.is_empty() {
        Arc::clone(&scan.table)
    } else {
        Arc::new(Schema::new_with_metadata(
            scan.table
                .fields()
                .iter()
                .filter(|f| !partition_fields.iter().any(|p| p.name() == f.name()))
                .cloned()
                .collect::<Vec<_>>(),
            scan.table.metadata().clone(),
        ))
    };
    let data_projection: Vec<ColumnPath> = projection
        .iter()
        .filter(|c| partition_of(c).is_none())
        .cloned()
        .collect();
    // A filter on a partition column is not applied here, nor any filter
    // written after it: the filter above evaluates all of them.
    let usable: Vec<PushedFilter> = p
        .filters
        .iter()
        .take_while(|f| !reads_partition_column(scan, f))
        .cloned()
        .collect();
    let exprs: Vec<PushdownExpr> = usable.iter().filter_map(|f| f.expr.clone()).collect();
    let mut request = ScanRequest::new(Arc::clone(&data_table), data_projection);
    request.predicate = match exprs.len() {
        0 => None,
        1 => exprs.into_iter().next(),
        _ => Some(PushdownExpr::And(exprs)),
    };
    request.row_level = scan_row_level(&p.filters, usable.len());
    request.limit = p.limit.filter(|_| p.filters.is_empty());
    request.metadata_cols = p.metadata;
    request.mapping = p.mapping.clone();
    let decoded = Arc::clone(SchemaMapper::new(&request, data_table, "")?.output_schema());
    let (mapped, layout) = if partition_fields.is_empty() {
        (decoded, Vec::new())
    } else {
        // The decoded batch has one column per distinct top-level column
        // of the projection, in order of first appearance (nested paths
        // of one column merge into it).
        let mut fields: Vec<FieldRef> = Vec::new();
        let mut layout = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        let mut data = 0;
        for c in &projection {
            if seen.contains(&c.root()) {
                continue;
            }
            seen.push(c.root());
            if let Some(k) = partition_of(c) {
                fields.push(Arc::new(
                    partition_fields[k].as_ref().clone().with_nullable(true),
                ));
                layout.push(Slot::Partition(k));
            } else {
                fields.push(Arc::clone(&decoded.fields()[data]));
                layout.push(Slot::Data(data));
                data += 1;
            }
        }
        // What the format adds after the projection (the row index).
        for i in data..decoded.fields().len() {
            fields.push(Arc::clone(&decoded.fields()[i]));
            layout.push(Slot::Data(i));
        }
        (
            Arc::new(Schema::new_with_metadata(
                fields,
                decoded.metadata().clone(),
            )),
            layout,
        )
    };
    let output = with_metadata_column(&mapped, &request.metadata_cols);
    Ok(TableRequest {
        request,
        mapped,
        output,
        partition_fields,
        layout,
    })
}

/// [`row_level`] over the first `usable` of `filters`, the rest being left
/// to the filter above the scan. A filter left out that may raise must
/// still see every row it could raise on, so then no runtime filter and no
/// later conjunct may drop rows: the prefix is all the scan has.
fn scan_row_level(filters: &[PushedFilter], usable: usize) -> RowLevel {
    let level = row_level(&filters[..usable]);
    let left_out_may_raise = filters[usable..]
        .iter()
        .any(|f| matches!(f.rows, RowLevel::Prefix(_)));
    if !matches!(level, RowLevel::Full) || !left_out_may_raise {
        return level;
    }
    let mut parts: Vec<PushdownExpr> = filters[..usable]
        .iter()
        .filter_map(|f| f.expr.clone())
        .collect();
    RowLevel::Prefix(match parts.len() {
        0 | 1 => parts.pop(),
        _ => Some(PushdownExpr::And(parts)),
    })
}

/// What the scan may drop rows by, over the pushed `filters` (a
/// conjunction, in written order): every filter's typed form while none
/// has a conjunct that may raise; at the first that has one, its prefix,
/// and no later filter.
fn row_level(filters: &[PushedFilter]) -> RowLevel {
    let mut parts = Vec::new();
    for f in filters {
        match &f.rows {
            RowLevel::Prefix(p) => {
                parts.extend(p.iter().cloned());
                return RowLevel::Prefix(match parts.len() {
                    0 | 1 => parts.pop(),
                    _ => Some(PushdownExpr::And(parts)),
                });
            }
            _ => parts.extend(f.expr.iter().cloned()),
        }
    }
    RowLevel::Full
}

impl FileScanSource {
    fn new(scan: &FileScan, p: ScanPushdowns) -> Result<Self> {
        check_capabilities(scan, &p)?;
        let TableRequest {
            request,
            mapped,
            output,
            partition_fields,
            layout,
        } = table_request(scan, &p)?;
        Ok(Self {
            format: Arc::clone(&scan.format),
            io: scan.io.clone(),
            files: Arc::clone(&scan.files),
            partitions: scan.plan(),
            min_morsel_bytes: scan.min_morsel_bytes,
            request,
            selection: p.selection,
            format_options: scan.format_options.clone(),
            scan_options: scan.scan_options,
            output,
            partition_fields,
            layout,
            mapped,
            runtime: p.dynamic,
            metrics: Mutex::new(FileScanMetrics::default()),
        })
    }

    /// The table-level request every split reads.
    #[must_use]
    pub fn request(&self) -> &ScanRequest {
        &self.request
    }

    /// The planned partitions (cut into morsels when the scan starts).
    #[must_use]
    pub fn partitions(&self) -> &[FilePartition] {
        &self.partitions
    }

    /// Counters of the splits read so far.
    #[must_use]
    pub fn metrics(&self) -> FileScanMetrics {
        *self.metrics.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn file(&self, i: usize) -> Result<&FileRef> {
        self.files
            .get(i)
            .ok_or_else(|| internal(format!("split of file {i} out of {}", self.files.len())))
    }

    fn end_split(&self, current: &Current) {
        self.metrics
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .add(&current.stream.metrics(), current.stream.overcommitted());
    }

    /// The morsels the drivers of `ctx`'s executor claim: whole splits
    /// when the scan reports `row_index` (a text format would count the
    /// lines before every piece again), else pieces no smaller than
    /// `spark.sql.files.openCostInBytes`.
    fn morsels(&self, ctx: &TaskContext) -> Vec<Vec<MorselSplit>> {
        let target = if self.request.metadata_cols.row_index {
            u64::MAX
        } else {
            krishiv_io::split::morsel_bytes(
                &self.partitions,
                ctx.executor().parallelism(),
                self.min_morsel_bytes,
            )
        };
        let caps = self.format.capabilities(&self.format_options);
        krishiv_io::split::morsels(&self.partitions, target, |f| {
            self.files
                .get(f)
                .is_some_and(|f| caps.is_splittable(f.path.as_ref()))
        })
    }

    /// Starts reading the footer of the file after piece `next` of
    /// `morsel` if it is another file and none is in flight for it.
    fn read_ahead(&self, local: &mut FileScanLocal, morsel: &[MorselSplit], next: usize) {
        let Some(piece) = morsel.get(next + 1) else {
            return;
        };
        let opening = morsel[next].file;
        let wanted = piece.file;
        if wanted == opening || local.ahead.as_ref().is_some_and(|(f, _)| *f == wanted) {
            return;
        }
        if let Some(file) = self.files.get(wanted) {
            let fut = self.io.prefetch_metadata(
                Arc::clone(&self.format),
                file.clone(),
                self.scan_options.needs,
                self.format_options.clone(),
            );
            local.ahead = Some((wanted, fut));
        }
    }

    async fn open(
        &self,
        local: &mut FileScanLocal,
        request: &ScanRequest,
        piece: MorselSplit,
    ) -> Result<()> {
        let MorselSplit { file, range, split } = piece;
        // Stage 0: a filter that drops every row skips the piece unopened.
        if let Some(f) = request.dynamic.iter().find(|f| f.drops_everything()) {
            f.filter().stats().add_split(range.len());
            return Ok(());
        }
        if let Some((f, _)) = &local.ahead
            && *f == file
            && let Some((_, fut)) = local.ahead.take()
        {
            // The read's own error (if any) is reported by the open below.
            drop(fut.await);
        }
        let f = self.file(file)?.clone();
        let partition_values = self.partition_values(&f)?;
        let row_filter_start = local
            .row_filter
            .and_then(|(last, on)| (last == file).then_some(on));
        let selected;
        let req = if self.selection.is_some() || row_filter_start.is_some() {
            let mut r = request.clone();
            if let Some(s) = &self.selection {
                r.selection = s.selection(&f);
            }
            r.row_filter_start = row_filter_start;
            selected = r;
            &selected
        } else {
            request
        };
        let stream = ScanStream::open(
            &self.io,
            self.format.as_ref(),
            f,
            range,
            req,
            &self.format_options,
            local.consumer.reservation(),
            self.scan_options,
        )
        .await?
        .with_gate(Arc::clone(&local.gate));
        local.current = Some(Current {
            stream,
            file,
            split,
            partition_values,
        });
        Ok(())
    }

    async fn next_batch(
        &self,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<Option<Batch>> {
        let g = global_ref::<FileScanGlobal>(global, "FileScan")?;
        let local = local_mut::<FileScanLocal>(local, "FileScan")?;
        loop {
            if self
                .request
                .limit
                .is_some_and(|l| g.returned.load(Ordering::Relaxed) >= l)
            {
                return Ok(None);
            }
            if let Some(cur) = local.current.as_mut() {
                match cur.stream.next_batch().await? {
                    Some(b) => {
                        let file = self.file(cur.file)?;
                        let b = self.with_partition_columns(b, &cur.partition_values)?;
                        let b = local.rows.apply(b)?;
                        let b = self.append_metadata(b, file, cur.split)?;
                        g.returned.fetch_add(b.num_rows() as u64, Ordering::Relaxed);
                        return Ok(Some(Batch::new(b)));
                    }
                    None => {
                        if let Some(done) = local.current.take() {
                            self.end_split(&done);
                            if let Some(on) = done.stream.row_filter_measured() {
                                local.row_filter = Some((done.file, on));
                            }
                        }
                        continue;
                    }
                }
            }
            if let Some((m, next)) = local.morsel
                && let Some(piece) = g.morsels[m].get(next).copied()
            {
                local.morsel = Some((m, next + 1));
                self.read_ahead(local, &g.morsels[m], next);
                self.open(local, &g.request, piece).await?;
                continue;
            }
            let Some((ordinal, m)) = g.cursor.claim_ordered() else {
                return Ok(None);
            };
            local.position = Some(ordinal);
            if m >= g.morsels.len() {
                return Err(internal(format!("morsel {m} out of {}", g.morsels.len())));
            }
            local.morsel = Some((m, 0));
            local.claimed += 1;
            // Runtime filters are measured afresh about as often as a driver
            // reads a planned split's worth of rows, not at every morsel.
            if local.claimed % MORSELS_PER_DRIVER == 1 {
                local.rows.start_partition();
            }
        }
    }

    /// The values of `file`'s partition columns, one row each.
    fn partition_values(&self, file: &FileRef) -> Result<Vec<ArrayRef>> {
        if self.layout.is_empty() {
            return Ok(Vec::new());
        }
        let texts = krishiv_io::file_values(&file.path, self.partition_fields.len())?;
        self.partition_fields
            .iter()
            .zip(&texts)
            .map(|(field, text)| krishiv_io::constant_column(field, text.as_deref(), 1))
            .collect()
    }

    /// The decoded `batch` as the mapped output: the partition columns
    /// between its data columns, each `values` repeated.
    fn with_partition_columns(
        &self,
        batch: RecordBatch,
        values: &[ArrayRef],
    ) -> Result<RecordBatch> {
        if self.layout.is_empty() {
            return Ok(batch);
        }
        let rows = batch.num_rows();
        let zeros = arrow_array::UInt32Array::from(vec![0_u32; rows]);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.layout.len());
        for slot in &self.layout {
            columns.push(match *slot {
                Slot::Data(i) => Arc::clone(batch.column(i)),
                Slot::Partition(k) => arrow_select::take::take(&values[k], &zeros, None)
                    .map_err(|e| internal(format!("partition column of {rows} rows: {e}")))?,
            });
        }
        RecordBatch::try_new_with_options(
            Arc::clone(&self.mapped),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|e| internal(format!("invalid partition columns: {e}")))
    }

    /// `batch` (the mapped table-level output) with the `_metadata` column.
    fn append_metadata(
        &self,
        batch: RecordBatch,
        file: &FileRef,
        range: ByteRange,
    ) -> Result<RecordBatch> {
        let cols = &self.request.metadata_cols;
        if !any_metadata(cols) {
            return Ok(batch);
        }
        let rows = batch.num_rows();
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        let row_index = if cols.row_index { columns.pop() } else { None };
        let Some(DataType::Struct(fields)) =
            self.output.fields().last().map(|f| f.data_type().clone())
        else {
            return Err(internal("file scan output without _metadata".to_owned()));
        };
        let long = |v: u64| -> Result<ArrayRef> {
            let v = i64::try_from(v).map_err(|_| internal(format!("{v} overflows a long")))?;
            Ok(Arc::new(Int64Array::from_value(v, rows)))
        };
        let mut children = Vec::with_capacity(fields.len());
        for f in &fields {
            children.push(match f.name().as_str() {
                "file_path" => constant_string(&spark_file_path(file), rows)?,
                "file_name" => constant_string(file.path.filename().unwrap_or(""), rows)?,
                "file_size" => long(file.size)?,
                "file_block_start" => long(range.start)?,
                "file_block_length" => long(range.len())?,
                "file_modification_time" => {
                    let micros = file.modified_micros.ok_or_else(|| {
                        internal(format!(
                            "{} was listed without a modification time",
                            file.display_path()
                        ))
                    })?;
                    Arc::new(
                        TimestampMicrosecondArray::from_value(micros, rows)
                            .with_data_type(f.data_type().clone()),
                    )
                }
                "row_index" => row_index
                    .clone()
                    .ok_or_else(|| internal("decoder returned no row index".to_owned()))?,
                other => return Err(internal(format!("unknown _metadata field {other}"))),
            });
        }
        let meta = StructArray::try_new_with_length(fields, children, None, rows)
            .map_err(|e| internal(format!("invalid _metadata column: {e}")))?;
        columns.push(Arc::new(meta));
        RecordBatch::try_new_with_options(
            Arc::clone(&self.output),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|e| internal(format!("invalid file scan batch: {e}")))
    }
}

fn any_metadata(c: &MetadataCols) -> bool {
    c.file_path
        || c.file_name
        || c.file_size
        || c.file_block_start
        || c.file_block_length
        || c.file_modification_time
        || c.row_index
}

/// `mapped` with the decoder's row-index column replaced by a `_metadata`
/// struct of the requested fields (Spark's order and types, not null).
fn with_metadata_column(mapped: &SchemaRef, c: &MetadataCols) -> SchemaRef {
    if !any_metadata(c) {
        return Arc::clone(mapped);
    }
    let string = krishiv_types::DataType::String.to_arrow();
    let timestamp = krishiv_types::DataType::Timestamp.to_arrow();
    let mut children = Vec::new();
    for (wanted, name, dt) in [
        (c.file_path, "file_path", &string),
        (c.file_name, "file_name", &string),
        (c.file_size, "file_size", &DataType::Int64),
        (c.file_block_start, "file_block_start", &DataType::Int64),
        (c.file_block_length, "file_block_length", &DataType::Int64),
        (
            c.file_modification_time,
            "file_modification_time",
            &timestamp,
        ),
        (c.row_index, "row_index", &DataType::Int64),
    ] {
        if wanted {
            children.push(Field::new(name, dt.clone(), false));
        }
    }
    let mut fields: Vec<Field> = mapped
        .fields()
        .iter()
        .filter(|f| f.name() != ROW_INDEX_COLUMN)
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(
        METADATA_COLUMN,
        DataType::Struct(Fields::from(children)),
        false,
    ));
    Arc::new(Schema::new_with_metadata(fields, mapped.metadata().clone()))
}

/// `value` repeated `rows` times, sharing one copy of its bytes.
fn constant_string(value: &str, rows: usize) -> Result<ArrayRef> {
    let bytes = value.as_bytes();
    let len = u32::try_from(bytes.len())
        .map_err(|_| internal("a _metadata string over 4 GiB".to_owned()))?;
    let view = make_view(bytes, 0, 0);
    let buffers = if len > 12 {
        vec![Buffer::from(bytes)]
    } else {
        Vec::new()
    };
    let views = ScalarBuffer::from(vec![view; rows]);
    let array = StringViewArray::try_new(views, buffers, None)
        .map_err(|e| internal(format!("invalid _metadata string: {e}")))?;
    Ok(Arc::new(array))
}

/// The file's URI as Spark prints `_metadata.file_path`: `file:/dir/f` for
/// local files, `<scheme>://<authority>/key` otherwise, the path quoted as
/// `java.net.URI` quotes it (Spark 4.2: `file:/tmp/données%20x/é%2541/...`
/// for `/tmp/données x/é%41/...`).
fn spark_file_path(file: &FileRef) -> String {
    let store = file.store.to_string();
    let path = uri_path(&decode_store_path(file.path.as_ref()));
    match store.strip_prefix("file://") {
        Some(rest) => format!("file:{}/{path}", rest.trim_end_matches('/')),
        None => format!("{}/{path}", store.trim_end_matches('/')),
    }
}

/// The text of an `object_store` path, whose segments percent-encode the
/// characters stores reject (`%` among them, so decoding is unambiguous).
fn decode_store_path(path: &str) -> String {
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = bytes
            .get(i + 1..i + 3)
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u8::from_str_radix(h, 16).ok());
        match (bytes[i], hex) {
            (b'%', Some(b)) => {
                out.push(b);
                i += 3;
            }
            (b, _) => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Quotes a path as `java.net.URI` does: unreserved and path punctuation
/// stay, other non-ASCII characters stay literal, everything else (space,
/// `%`, controls, `#`, `?`, ...) is percent-encoded as UTF-8.
fn uri_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for c in path.chars() {
        let keep = if c.is_ascii() {
            c.is_ascii_alphanumeric() || "-._~!$&'()*+,;=:@/".contains(c)
        } else {
            !c.is_control() && !c.is_whitespace()
        };
        if keep {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            for b in c.encode_utf8(&mut buf).bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

impl Source for FileScanSource {
    fn name(&self) -> &'static str {
        "FileScan"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let mut request = self.request.clone();
        // With a filter conjunct that may raise, no runtime filter: Spark
        // evaluates its runtime filter as the filter's last conjunct, so it
        // skips no split, row group or row that conjunct still has to see.
        if request.row_level.runtime_filters() {
            let table = &self.request.table_schema;
            request.dynamic = self.runtime.scan_filters(table);
        }
        let morsels = self.morsels(ctx);
        Ok(Arc::new(FileScanGlobal {
            request,
            cursor: MorselCursor::new(morsels.len(), ctx.query().config().seed),
            morsels,
            returned: AtomicU64::new(0),
        }))
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global_ref::<FileScanGlobal>(global, "FileScan").map_or(1, |g| g.cursor.len().max(1))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let g = global_ref::<FileScanGlobal>(global, "FileScan")?;
        Ok(Box::new(FileScanLocal {
            rows: super::RowStage::new(&g.request.dynamic, &self.mapped, self.runtime.settings()),
            consumer: Arc::clone(ctx.consumer().consumer()),
            gate: Arc::clone(ctx.query().scan_gate()),
            morsel: None,
            current: None,
            ahead: None,
            claimed: 0,
            row_filter: None,
            position: None,
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(self.next_batch(local, global))
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        local
            .downcast_ref::<FileScanLocal>()
            .map_or(0, |l| l.claimed)
    }

    fn reports_positions(&self) -> bool {
        true
    }

    fn position(&self, local: &LocalState) -> Option<u64> {
        local
            .downcast_ref::<FileScanLocal>()
            .and_then(|l| l.position)
    }

    fn runtime_filters(&self) -> Vec<krishiv_io::FilterId> {
        if !self.request.row_level.runtime_filters() {
            return Vec::new();
        }
        self.runtime.filters().iter().map(|f| f.id()).collect()
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        let mut m = self.runtime.extra_metrics();
        m.extend(
            self.metrics
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extra(),
        );
        m
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn value(extra: &crate::metrics::ExtraMetrics, name: &str) -> Option<u64> {
        extra
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, v)| v.value())
    }

    /// The decode over-commit shows in `EXPLAIN ANALYZE` only when there was
    /// any, summed over the scan's splits.
    #[test]
    fn decode_over_commit_is_reported_when_there_was_any() {
        let mut m = FileScanMetrics::default();
        assert!(m.extra().is_empty());
        m.add(&ScanUnitMetrics::default(), 0);
        assert_eq!(value(&m.extra(), "decode_overcommit_bytes"), None);
        m.add(&ScanUnitMetrics::default(), 3);
        m.add(&ScanUnitMetrics::default(), 4);
        assert_eq!(value(&m.extra(), "decode_overcommit_bytes"), Some(7));
    }

    /// The scan counters are summed over splits; the row filter's appear
    /// only where it ran.
    #[test]
    fn scan_counters_sum_over_splits() {
        let mut m = FileScanMetrics::default();
        let mut unit = ScanUnitMetrics::default();
        unit.units = 4;
        unit.units_pruned = 1;
        unit.rows = 100;
        unit.bytes_pushed = 1000;
        m.add(&unit, 0);
        let extra = m.extra();
        assert_eq!(value(&extra, "scan_splits"), Some(1));
        assert_eq!(value(&extra, "row_filter_rows"), None);
        let mut filtered = unit;
        filtered.units_filtered = 2;
        filtered.units_unfiltered = 1;
        filtered.row_filter_rows = 50;
        filtered.row_filter_kept = 5;
        m.add(&filtered, 0);
        let extra = m.extra();
        for (name, want) in [
            ("scan_splits", 2),
            ("scan_units", 8),
            ("scan_units_pruned", 2),
            ("scan_rows_decoded", 200),
            ("scan_bytes", 2000),
            ("row_filter_units", 2),
            ("row_filter_units_skipped", 1),
            ("row_filter_rows", 50),
            ("row_filter_kept", 5),
        ] {
            assert_eq!(value(&extra, name), Some(want), "{name}");
        }
    }

    #[test]
    fn spark_paths_and_constants() {
        let file = FileRef {
            store: krishiv_io::StoreId::new("file://"),
            path: object_store::path::Path::from("tmp/pq/a b.parquet"),
            size: 10,
            version: krishiv_io::VersionToken::Immutable,
            modified_micros: Some(5),
        };
        assert_eq!(spark_file_path(&file), "file:/tmp/pq/a%20b.parquet");
        // Spark 4.2 keeps non-ASCII text and quotes spaces and `%`.
        let odd = FileRef {
            path: object_store::path::Path::from("tmp/données x/é%41/f.parquet"),
            ..file.clone()
        };
        assert_eq!(
            spark_file_path(&odd),
            "file:/tmp/données%20x/é%2541/f.parquet"
        );
        let s3 = FileRef {
            store: krishiv_io::StoreId::new("s3://bucket"),
            ..file
        };
        assert_eq!(spark_file_path(&s3), "s3://bucket/tmp/pq/a%20b.parquet");
        for s in ["short", "a value longer than twelve bytes"] {
            let a = constant_string(s, 3).unwrap_or_else(|e| unreachable!("{e}"));
            let a = a
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap_or_else(|| unreachable!());
            assert_eq!(a.iter().collect::<Vec<_>>(), [Some(s); 3]);
        }
    }
}

#[cfg(test)]
mod row_level_tests {
    use arrow_array::BooleanArray;

    use super::*;
    use crate::operators::expr::FnPredicate;

    fn filter(rows: RowLevel) -> PushedFilter {
        let predicate: Arc<dyn super::super::expr::Predicate> =
            Arc::new(FnPredicate(|b: &RecordBatch| {
                Ok(BooleanArray::from(vec![true; b.num_rows()]))
            }));
        PushedFilter {
            predicate,
            expr: Some(PushdownExpr::Constant(true)),
            rows,
        }
    }

    #[test]
    fn a_filter_left_out_that_may_raise_keeps_runtime_filters_off() {
        let safe = filter(RowLevel::Full);
        let raising = filter(RowLevel::Prefix(None));
        // Nothing left out: as `row_level`.
        assert!(matches!(
            scan_row_level(std::slice::from_ref(&safe), 1),
            RowLevel::Full
        ));
        // A safe filter left out changes nothing.
        assert!(matches!(
            scan_row_level(&[safe.clone(), safe.clone()], 1),
            RowLevel::Full
        ));
        // A raising filter left out: only the usable prefix may drop rows.
        let level = scan_row_level(&[safe.clone(), raising.clone()], 1);
        assert!(
            matches!(&level, RowLevel::Prefix(Some(PushdownExpr::Constant(true)))),
            "{level:?}"
        );
        assert!(matches!(
            scan_row_level(std::slice::from_ref(&raising), 0),
            RowLevel::Prefix(None)
        ));
        // A later raising filter counts too.
        assert!(matches!(
            scan_row_level(&[safe.clone(), safe, raising], 1),
            RowLevel::Prefix(Some(_))
        ));
    }
}
