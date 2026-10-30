// SPDX-License-Identifier: Apache-2.0
//! The file scan: [`FileScan`] (what the planner pushes into, over any
//! [`FileFormat`]) and the [`FileScanSource`] it builds.
//!
//! - **Morsels are Spark's file partitions** ([`krishiv_io::split`]): a
//!   driver claims one and reads its splits in order, each through a
//!   [`ScanStream`] (metadata from the cache, schema mapping, the format's
//!   sans-IO decoder, fetches and prefetch through the I/O scheduler).
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
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_io::split::{FilePartition, SplitConfig, SplitFile};
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
        }
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
        Ok(table_request(self, pushdowns)?.2)
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
}

impl FileScanMetrics {
    /// The `EXPLAIN ANALYZE` counters: the decode over-commit, when there
    /// was any.
    fn extra(&self) -> crate::metrics::ExtraMetrics {
        if self.decode_overcommit_bytes == 0 {
            return Vec::new();
        }
        crate::metrics::counts([("decode_overcommit_bytes", self.decode_overcommit_bytes)])
    }

    fn add(&mut self, m: &ScanUnitMetrics, overcommitted: u64) {
        self.decode_overcommit_bytes += overcommitted;
        self.splits += 1;
        self.units += m.units;
        self.units_pruned += m.units_pruned;
        self.pages_pruned += m.pages_pruned;
        self.rows += m.rows;
        self.bytes += m.bytes_pushed;
    }
}

/// The source of a file scan (see the module docs).
pub struct FileScanSource {
    format: Arc<dyn FileFormat>,
    io: IoContext,
    files: Arc<[FileRef]>,
    partitions: Arc<[FilePartition]>,
    /// Table level, immutable.
    request: ScanRequest,
    selection: Option<Arc<dyn FileSelections>>,
    format_options: FormatOptions,
    scan_options: ScanOptions,
    output: SchemaRef,
    /// The decoded batches' schema (`output` before `_metadata`).
    mapped: SchemaRef,
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
    cursor: MorselCursor,
    /// The table-level request with the runtime filters attached.
    request: ScanRequest,
    /// Rows returned by every driver (for a limit hint).
    returned: AtomicU64,
}

/// The split being read.
struct Current {
    stream: ScanStream,
    file: usize,
    range: ByteRange,
}

type MetadataFuture = BoxFuture<'static, krishiv_io::Result<Arc<dyn FileMetadata>>>;

struct FileScanLocal {
    consumer: Arc<ConsumerHandle>,
    /// The query's scan decode gate.
    gate: Arc<krishiv_io::ScanGate>,
    /// The claimed partition and its next split.
    partition: Option<(usize, usize)>,
    current: Option<Current>,
    /// The footer read in flight for a file this driver opens next.
    ahead: Option<(usize, MetadataFuture)>,
    claimed: u64,
    /// The claim ordinal of the claimed partition.
    position: Option<u64>,
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

/// The table-level request of a scan of `scan` with `p`, the schema its
/// columns map to (the mapped output does not depend on the file: it maps
/// onto the table) and the output schema (with `_metadata`).
fn table_request(
    scan: &FileScan,
    p: &ScanPushdowns,
) -> Result<(ScanRequest, SchemaRef, SchemaRef)> {
    let projection = p.projection.clone().unwrap_or_else(|| {
        scan.table
            .fields()
            .iter()
            .map(|f| ColumnPath::new(f.name()))
            .collect()
    });
    let exprs: Vec<PushdownExpr> = p.filters.iter().filter_map(|f| f.expr.clone()).collect();
    let mut request = ScanRequest::new(Arc::clone(&scan.table), projection);
    request.predicate = match exprs.len() {
        0 => None,
        1 => exprs.into_iter().next(),
        _ => Some(PushdownExpr::And(exprs)),
    };
    request.row_level = row_level(&p.filters);
    request.limit = p.limit.filter(|_| p.filters.is_empty());
    request.metadata_cols = p.metadata;
    request.mapping = p.mapping.clone();
    let mapped =
        Arc::clone(SchemaMapper::new(&request, Arc::clone(&scan.table), "")?.output_schema());
    let output = with_metadata_column(&mapped, &request.metadata_cols);
    Ok((request, mapped, output))
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
        let (request, mapped, output) = table_request(scan, &p)?;
        Ok(Self {
            format: Arc::clone(&scan.format),
            io: scan.io.clone(),
            files: Arc::clone(&scan.files),
            partitions: scan.plan(),
            request,
            selection: p.selection,
            format_options: scan.format_options.clone(),
            scan_options: scan.scan_options,
            output,
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

    /// The partitions (morsels).
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

    /// Starts reading the footer of the file after `next` in `partition`
    /// if it is another file and none is in flight for it.
    fn read_ahead(&self, local: &mut FileScanLocal, partition: usize, next: usize) {
        let Some(split) = self.partitions[partition].splits.get(next + 1) else {
            return;
        };
        let opening = self.partitions[partition].splits[next].file;
        let wanted = split.file;
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
        file: usize,
        range: ByteRange,
    ) -> Result<()> {
        // Stage 0: a filter that drops every row skips the split unopened.
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
        let selected;
        let req = match &self.selection {
            Some(s) => {
                let mut r = request.clone();
                r.selection = s.selection(&f);
                selected = r;
                &selected
            }
            None => request,
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
            range,
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
                        let b = local.rows.apply(b)?;
                        let b = self.append_metadata(b, file, cur.range)?;
                        g.returned.fetch_add(b.num_rows() as u64, Ordering::Relaxed);
                        return Ok(Some(Batch::new(b)));
                    }
                    None => {
                        if let Some(done) = local.current.take() {
                            self.end_split(&done);
                        }
                        continue;
                    }
                }
            }
            if let Some((p, next)) = local.partition
                && let Some(split) = self.partitions[p].splits.get(next).copied()
            {
                local.partition = Some((p, next + 1));
                self.read_ahead(local, p, next);
                self.open(local, &g.request, split.file, split.range)
                    .await?;
                continue;
            }
            let Some((ordinal, p)) = g.cursor.claim_ordered() else {
                return Ok(None);
            };
            local.position = Some(ordinal);
            if p >= self.partitions.len() {
                return Err(internal(format!(
                    "partition {p} out of {}",
                    self.partitions.len()
                )));
            }
            local.partition = Some((p, 0));
            local.claimed += 1;
            // Runtime filter decisions depend on this partition's rows only.
            local.rows.start_partition();
        }
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
            request.dynamic = self.runtime.scan_filters(&self.request.table_schema);
        }
        Ok(Arc::new(FileScanGlobal {
            request,
            cursor: MorselCursor::new(self.partitions.len(), ctx.query().config().seed),
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
            partition: None,
            current: None,
            ahead: None,
            claimed: 0,
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

    /// The decode over-commit shows in `EXPLAIN ANALYZE` only when there was
    /// any, summed over the scan's splits.
    #[test]
    fn decode_over_commit_is_reported_when_there_was_any() {
        let mut m = FileScanMetrics::default();
        m.add(&ScanUnitMetrics::default(), 0);
        assert!(m.extra().is_empty());
        m.add(&ScanUnitMetrics::default(), 3);
        m.add(&ScanUnitMetrics::default(), 4);
        let extra = m.extra();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].0, "decode_overcommit_bytes");
        assert_eq!(extra[0].1.value(), Some(7));
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
