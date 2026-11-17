// SPDX-License-Identifier: Apache-2.0
//! One write task's data files for an append (P3-T027, PR-36a; design note
//! §6.1, distribution `none`): rows of the table's Arrow schema
//! ([`arrow_schema`]) in, Parquet files with the table's field ids out, each
//! described by a [`WrittenDataFile`] for its manifest entry.
//!
//! Java references are apache-iceberg 1.12.0 classes and methods (no line
//! numbers verified).
//!
//! - **Partitioning.** Each row's partition key is computed from the default
//!   spec's fields ([`partition_value`] over the source column; a null source
//!   or `void` gives a null value, a transform Krishiv cannot compute fails the
//!   write). Rows are grouped by key (floats as Java's `Float.equals`:
//!   NaN payloads one key, -0.0 and 0.0 two) and taken per key. A spec whose
//!   fields are all `void` (an unpartitioned one included) does no per-row
//!   work. The per-row cost is one `ScalarValue` per partition field and row
//!   (a string source is copied), the key bytes in one reused buffer.
//! - **Fanout** (Java's `FanoutWriter` shape): one open file per key, at most
//!   `max_open_files` open, the oldest-opened closed first; a key seen again
//!   after its file closed gets a new file. The cap is Krishiv's own (design
//!   note §6.1); Java's `FanoutWriter` has none. With more keys per batch
//!   than the cap (say 150 bucket keys and the cap 100), every batch closes
//!   and reopens each key, a new small file per key per batch: such writes
//!   want `hash` distribution (a later step).
//! - **Rollover** as Java's `RollingFileWriter.shouldRollToNewFile`:
//!   `currentFileRows % ROWS_DIVISOR == 0 && currentWriter.length() >=
//!   targetFileSizeInBytes`, `ROWS_DIVISOR` = 1000, the target
//!   `write.target-file-size-bytes` (`TableProperties.
//!   WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT`, 512 MiB). Java deletes an empty
//!   file on close; this writer opens a file only for rows, so none is empty.
//! - **Parquet settings** from the table properties, as Java's
//!   `Parquet.WriteBuilder` reads them (`TableProperties`):
//!   `write.parquet.compression-codec` is the codec, `gzip` when the table
//!   does not set it (`PARQUET_COMPRESSION_DEFAULT`; Java writes `zstd`,
//!   `PARQUET_COMPRESSION_DEFAULT_SINCE_1_4_0`, only into the properties of
//!   the tables it creates; iceberg-rust 0.11's fallback `zstd` differs).
//!   Names as `Parquet.toCodec` (`CompressionCodecName.valueOf` of the upper
//!   case; otherwise "Unsupported compression codec: " + codec); `lzo`, which
//!   Java accepts, is refused (a divergence: Krishiv cannot write it).
//!   `write.parquet.row-group-size-bytes` (`PARQUET_ROW_GROUP_SIZE_BYTES_DEFAULT`,
//!   128 MiB) is the row-group size (`parquet.block.size`),
//!   `write.parquet.page-size-bytes` (`PARQUET_PAGE_SIZE_BYTES_DEFAULT`,
//!   1 MiB) the page size; timestamps are INT64 microseconds, never rebased.
//!   `write.parquet.compression-level` (`PARQUET_COMPRESSION_LEVEL_DEFAULT`
//!   null, the codec's own) is not applied: the Parquet writer has no level
//!   option (zstd writes level 3, parquet-mr's default). Neither are
//!   `write.parquet.dict-size-bytes`, `page-row-limit` nor the bloom-filter
//!   properties. `write.format.default` (`DEFAULT_FILE_FORMAT_DEFAULT`
//!   `parquet`) other than `parquet` is refused.
//! - **Names and locations:** [`file_name`], Java's `OutputFileFactory`
//!   (`"%05d-%d-%s-%05d%s"` over partition id, task attempt id, operation id
//!   and `fileCount.incrementAndGet()` from 0, so the first file is 1), placed
//!   by [`DataLocations::location`] for the file's partition tuple. The
//!   attempt id must be unique per attempt ([`TaskIds::attempt_id`]).
//! - **Abort:** [`DataFileWriter::finish`] failing, [`DataFileWriter::abort`]
//!   and a drop before `finish` abort the open uploads (nothing becomes
//!   visible) and delete the files already finished (a drop deletes in a
//!   task spawned on the I/O runtime, else the current runtime). An abort
//!   dropped midway leaves what it did not delete to the drop. A close
//!   cancelled after the store completed the upload leaves an unreferenced
//!   file (in no manifest); an open upload dropped outside a runtime is not
//!   aborted (`Upload`'s drop).
//!
//! Columns of type `uuid` or `fixed` are refused: Java writes them as
//! `FIXED_LEN_BYTE_ARRAY`, the Spark schema carries them as string and
//! binary. Files carry no `split_offsets` (the writer's statistics have no
//! row-group offsets) and no `iceberg.schema` footer key.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, LargeListArray, ListArray, MapArray, RecordBatch, StructArray, UInt32Array,
};
use arrow_schema::{DataType, TimeUnit};
use iceberg::spec::{PartitionSpecRef, PrimitiveType, SchemaRef, Struct, TableMetadata, Type};
use krishiv_common::config::Config;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_io::io::upload::Upload;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{Encoder, FileFormat, FormatOptions, IoContext, PinnedStore, ScalarValue};
use krishiv_memory::Reservation;

use super::data_file::{DataLocations, WrittenDataFile, file_name, partition_tuple};
use super::scan::{iceberg_type, plan_transform, schema_fields};
use super::storage::Registered;
#[cfg(test)]
use super::storage::StorageRegistration;
use super::types::arrow_schema;
use super::write_metrics::WriteMetricsConfig;
use crate::plan::file_metrics::file_metrics;
use crate::plan::transform::partition_value;
use crate::plan::types::{IcebergType, SchemaField, Transform};

/// Open files per task by default (`krishiv.iceberg.write.maxOpenFiles`).
pub(crate) const DEFAULT_MAX_OPEN_FILES: usize = 100;

/// `write.target-file-size-bytes` and its default (Java's
/// `TableProperties.WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT`, 512 MiB).
pub(crate) const TARGET_FILE_SIZE: &str = "write.target-file-size-bytes";
const TARGET_FILE_SIZE_DEFAULT: i64 = 512 * 1024 * 1024;
/// `write.parquet.compression-codec` and Java's fallback when a table does
/// not set it (`PARQUET_COMPRESSION_DEFAULT`, `gzip`; new tables carry `zstd`
/// as a property instead).
pub(crate) const PARQUET_CODEC: &str = "write.parquet.compression-codec";
const PARQUET_CODEC_DEFAULT: &str = "gzip";
/// `write.parquet.row-group-size-bytes` and its default (128 MiB).
pub(crate) const ROW_GROUP_SIZE: &str = "write.parquet.row-group-size-bytes";
const ROW_GROUP_SIZE_DEFAULT: i32 = 128 * 1024 * 1024;
/// `write.parquet.page-size-bytes` and its default (1 MiB).
pub(crate) const PAGE_SIZE: &str = "write.parquet.page-size-bytes";
const PAGE_SIZE_DEFAULT: i32 = 1024 * 1024;
/// `write.format.default`.
const FILE_FORMAT: &str = "write.format.default";

/// Rows between two checks of the target file size (Java's
/// `RollingFileWriter.ROWS_DIVISOR`).
const ROWS_DIVISOR: u64 = 1000;

/// Who writes: the task's partition id, the attempt's id and the write's
/// id, which name its files ([`file_name`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskIds<'a> {
    /// The task's partition id.
    pub(crate) partition_id: u32,
    /// The id of this attempt of the task, unique per attempt (Java's
    /// `OutputFileFactory.builderFor(table, partitionId, taskId)` gets Spark's
    /// `taskAttemptId`). Never a stable task index: a retried attempt would
    /// write the failed attempt's names, and that attempt's abort could then
    /// delete the files the retry commits.
    pub(crate) attempt_id: u64,
    /// The write's id (Java's operation id).
    pub(crate) write_id: &'a str,
}

/// How one partition field's value is found in a row.
#[derive(Debug, Clone)]
struct FieldPlan {
    transform: Transform,
    /// The source column: its top-level index, the names below it, its type;
    /// `None` for `void`.
    source: Option<(usize, Vec<String>, IcebergType)>,
}

struct OpenFile {
    key: Vec<u8>,
    partition: Struct,
    location: String,
    at: LocatedPath,
    store: PinnedStore,
    encoder: Box<dyn Encoder>,
    upload: Upload,
    rows: u64,
}

struct Finished {
    file: WrittenDataFile,
    at: LocatedPath,
    store: PinnedStore,
}

/// One task's data-file writer (see the module docs).
pub(crate) struct DataFileWriter {
    io: IoContext,
    paths: Arc<dyn PathLocator>,
    schema: SchemaRef,
    arrow: arrow_schema::SchemaRef,
    spec: PartitionSpecRef,
    fields: Vec<FieldPlan>,
    columns: Vec<SchemaField>,
    metrics: WriteMetricsConfig,
    locations: DataLocations,
    options: FormatOptions,
    target_file_bytes: u64,
    max_open_files: usize,
    partition_id: u32,
    attempt_id: u64,
    write_id: String,
    /// Files opened so far (the last file's number).
    files: u32,
    /// Open files, oldest-opened first.
    open: Vec<OpenFile>,
    finished: Vec<Finished>,
    mem: Reservation,
    /// Set once the files were handed out or deleted.
    done: bool,
}

impl std::fmt::Debug for DataFileWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFileWriter")
            .field("write_id", &self.write_id)
            .field("open", &self.open.len())
            .field("finished", &self.finished.len())
            .finish_non_exhaustive()
    }
}

impl DataFileWriter {
    /// A writer of the table `metadata` describes (its current schema,
    /// default spec, properties and sort order) through `registration`, at
    /// most `max_open_files` (at least 1) open at once, holding encoder and
    /// upload memory in `mem`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a format other than Parquet, a codec
    /// Krishiv cannot write, a size property that is not a number, a `uuid`,
    /// `fixed` or `variant` column, a partition source Krishiv cannot read
    /// (inside a list or map), and [`DataLocations::for_table`]'s; the
    /// schema's conversion errors ([`arrow_schema`]); `INTERNAL_ERROR` when
    /// the registration is gone.
    #[cfg(test)]
    pub(crate) fn new(
        metadata: &TableMetadata,
        registration: &StorageRegistration,
        ids: TaskIds<'_>,
        max_open_files: usize,
        mem: Reservation,
    ) -> Result<Self, KrishivError> {
        let registered = registration.registered()?;
        Self::with_registered(metadata, registered, ids, max_open_files, mem)
    }

    /// As `new`, through a registration already resolved (what an opened
    /// table keeps; the sink's writers).
    ///
    /// # Errors
    /// As `new`, except the registration's.
    pub(crate) fn with_registered(
        metadata: &TableMetadata,
        registered: Registered,
        ids: TaskIds<'_>,
        max_open_files: usize,
        mem: Reservation,
    ) -> Result<Self, KrishivError> {
        let props = metadata.properties();
        if let Some(format) = props.get(FILE_FORMAT)
            && !format.eq_ignore_ascii_case("parquet")
        {
            return Err(unsupported(format!(
                "writing Iceberg {format} data files ({FILE_FORMAT})"
            )));
        }
        let schema = Arc::clone(metadata.current_schema());
        for f in schema.as_struct().fields() {
            check_writable(&f.name, &f.field_type)?;
        }
        let arrow = arrow_schema(&schema)?;
        let spec = Arc::clone(metadata.default_partition_spec());
        let fields = spec
            .fields()
            .iter()
            .map(|f| field_plan(&schema, &arrow, f.source_id, plan_transform(f.transform)))
            .collect::<Result<Vec<_>, _>>()?;
        let options = parquet_options(props)?;
        ParquetFormat
            .check_write(&arrow, &options)
            .map_err(|e| e.into_error(metadata.location()))?;
        Ok(Self {
            io: registered.io,
            paths: registered.paths,
            columns: schema_fields(&schema),
            metrics: WriteMetricsConfig::from_table(
                props,
                &schema,
                Some(metadata.default_sort_order()),
            ),
            locations: DataLocations::for_table(metadata)?,
            target_file_bytes: target_file_bytes(props)?,
            schema,
            arrow,
            spec,
            fields,
            options,
            max_open_files: max_open_files.max(1),
            partition_id: ids.partition_id,
            attempt_id: ids.attempt_id,
            write_id: ids.write_id.to_owned(),
            files: 0,
            open: Vec::new(),
            finished: Vec::new(),
            mem,
            done: false,
        })
    }

    /// Writes `batch` (of the table's Arrow schema), each row into its
    /// partition's file.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a batch of another schema; a transform Krishiv
    /// cannot compute (`KRISHIV_UNSUPPORTED_FEATURE`); encode, upload and
    /// memory errors. The writer should then be aborted.
    pub(crate) async fn write(&mut self, batch: &RecordBatch) -> Result<(), KrishivError> {
        let batch = self.conform(batch)?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if self.fields.iter().all(|f| f.source.is_none()) {
            let values = vec![None; self.fields.len()];
            let mut key = Vec::new();
            key_bytes(&values, &mut key);
            return self.write_group(&key, &values, &batch).await;
        }
        for (key, values, rows) in self.groups(&batch)? {
            let slice = if rows.len() == batch.num_rows() {
                batch.clone()
            } else {
                arrow_select::take::take_record_batch(&batch, &UInt32Array::from(rows))
                    .map_err(|e| internal(format!("taking a partition's rows: {e}")))?
            };
            self.write_group(&key, &values, &slice).await?;
        }
        Ok(())
    }

    /// Ends every open file; returns every file written, in the order they
    /// were closed.
    ///
    /// # Errors
    /// Encode and upload errors; every file of the task is then deleted.
    pub(crate) async fn finish(mut self) -> Result<Vec<WrittenDataFile>, KrishivError> {
        while !self.open.is_empty() {
            if let Err(e) = self.close(0).await {
                self.abort_files().await;
                return Err(e);
            }
        }
        self.done = true;
        Ok(std::mem::take(&mut self.finished)
            .into_iter()
            .map(|f| f.file)
            .collect())
    }

    /// Abandons the write: open uploads are aborted, finished files deleted.
    pub(crate) async fn abort(mut self) {
        self.abort_files().await;
    }

    /// Cancellation-safe: a file leaves `open` or `finished` only once its
    /// abort or delete is done (an upload dropped midway aborts on its own
    /// drop), and `done` is set last, so a dropped abort leaves the rest to
    /// [`Drop`].
    async fn abort_files(&mut self) {
        while let Some(f) = self.open.pop() {
            f.upload.abort().await;
        }
        while let Some(f) = self.finished.last() {
            // Best effort: a file left behind is not in any manifest.
            let _ = self.io.delete_pinned(&f.store, &f.at.path).await;
            self.finished.pop();
        }
        let held = self.mem.size();
        self.mem.shrink(held);
        self.done = true;
    }

    /// `batch` under the table's Arrow schema (field ids included).
    fn conform(&self, batch: &RecordBatch) -> Result<RecordBatch, KrishivError> {
        let fields = self.arrow.fields();
        let got = batch.schema();
        if got.fields().len() != fields.len()
            || got
                .fields()
                .iter()
                .zip(fields.iter())
                .any(|(a, b)| a.name() != b.name())
        {
            return Err(internal(format!(
                "an Iceberg write of {} got a batch of {}",
                self.arrow, got
            )));
        }
        let columns = batch
            .columns()
            .iter()
            .zip(fields.iter())
            .map(|(c, f)| retype(c, f.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        RecordBatch::try_new(Arc::clone(&self.arrow), columns)
            .map_err(|e| internal(format!("an Iceberg write's batch: {e}")))
    }

    /// The rows of `batch` by partition key, keys in first-row order.
    #[expect(clippy::type_complexity, reason = "a local triple")]
    fn groups(
        &self,
        batch: &RecordBatch,
    ) -> Result<Vec<(Vec<u8>, Vec<Option<ScalarValue>>, Vec<u32>)>, KrishivError> {
        let sources = self
            .fields
            .iter()
            .map(|f| {
                f.source
                    .as_ref()
                    .map(|(i, path, _)| source_chain(batch, *i, path))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut index: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut groups = Vec::new();
        let mut values = Vec::with_capacity(self.fields.len());
        let mut key = Vec::new();
        for row in 0..batch.num_rows() {
            values.clear();
            for (f, chain) in self.fields.iter().zip(&sources) {
                let value = match (chain, &f.source) {
                    (Some(chain), Some((_, _, ty))) => {
                        let v = value_at(chain, row)?;
                        partition_value(f.transform, ty, v.as_ref())?
                    }
                    _ => None,
                };
                values.push(value);
            }
            key.clear();
            key_bytes(&values, &mut key);
            let at = match index.get(key.as_slice()) {
                Some(at) => *at,
                None => {
                    index.insert(key.clone(), groups.len());
                    groups.push((key.clone(), values.clone(), Vec::new()));
                    groups.len() - 1
                }
            };
            let row = u32::try_from(row).map_err(|_| internal("batch too long".to_owned()))?;
            groups[at].2.push(row);
        }
        Ok(groups)
    }

    /// Writes `batch` (rows of one key) into the key's file, rolling over.
    async fn write_group(
        &mut self,
        key: &[u8],
        values: &[Option<ScalarValue>],
        batch: &RecordBatch,
    ) -> Result<(), KrishivError> {
        let mut offset = 0;
        while offset < batch.num_rows() {
            let at = self.file_for(key, values).await?;
            let f = &mut self.open[at];
            let room = ROWS_DIVISOR - f.rows % ROWS_DIVISOR;
            let n = usize::try_from(room)
                .unwrap_or(usize::MAX)
                .min(batch.num_rows() - offset);
            f.encoder
                .write(&batch.slice(offset, n))
                .map_err(|e| e.into_error(&f.at.display))?;
            f.rows += n as u64;
            offset += n;
            drain(f).await?;
            self.account().await?;
            let f = &self.open[at];
            if f.rows.is_multiple_of(ROWS_DIVISOR)
                && f.encoder.written_bytes() >= self.target_file_bytes
            {
                self.close(at).await?;
            }
        }
        Ok(())
    }

    /// The index of the open file of `key`, opening one (and closing the
    /// oldest when `max_open_files` are open).
    async fn file_for(
        &mut self,
        key: &[u8],
        values: &[Option<ScalarValue>],
    ) -> Result<usize, KrishivError> {
        if let Some(at) = self.open.iter().position(|f| f.key == key) {
            return Ok(at);
        }
        if self.open.len() >= self.max_open_files {
            self.close(0).await?;
        }
        self.files = self
            .files
            .checked_add(1)
            .ok_or_else(|| internal("too many files in one task".to_owned()))?;
        let name = file_name(
            self.partition_id,
            self.attempt_id,
            &self.write_id,
            self.files,
        );
        let partition = partition_tuple(&self.spec, &self.schema, values)?;
        let location = self
            .locations
            .location(&self.spec, &self.schema, &partition, &name)?;
        let at = self.paths.locate(&location)?;
        let store = self.io.pin(&at.store)?;
        let seed = format!(
            "{}/{}/{}/{}",
            self.write_id, self.partition_id, self.attempt_id, self.files
        );
        let encoder = ParquetFormat
            .writer(
                &self.io,
                Arc::clone(&self.arrow),
                &self.options.clone().with_write_seed(seed),
            )
            .await
            .map_err(|e| e.into_error(&at.display))?;
        let upload = self.io.upload_pinned(&store, at.path.clone())?;
        self.open.push(OpenFile {
            key: key.to_vec(),
            partition,
            location,
            at,
            store,
            encoder,
            upload,
            rows: 0,
        });
        Ok(self.open.len() - 1)
    }

    /// Ends the open file at `at`.
    async fn close(&mut self, at: usize) -> Result<(), KrishivError> {
        let mut f = self.open.remove(at);
        drain(&mut f).await?;
        let OpenFile {
            partition,
            location,
            at,
            store,
            encoder,
            mut upload,
            rows,
            ..
        } = f;
        let done = encoder.finish().map_err(|e| e.into_error(&at.display))?;
        upload.write(done.tail).await?;
        let size = upload.finish().await?;
        let metrics = file_metrics(&done.statistics, &self.columns, |id| self.metrics.mode(id));
        self.finished.push(Finished {
            file: WrittenDataFile {
                location,
                size,
                rows,
                metrics,
                partition,
                spec_id: self.spec.spec_id(),
                split_offsets: None,
                sort_order_id: None,
            },
            at,
            store,
        });
        self.account().await
    }

    /// Charges what the open files hold to the reservation; under pressure
    /// encodes and ships what they buffer first (as `krishiv-io`'s `FileWriter`).
    async fn account(&mut self) -> Result<(), KrishivError> {
        let held = self.held();
        let size = self.mem.size();
        if held <= size {
            self.mem.shrink(size - held);
            return Ok(());
        }
        if self.mem.try_grow(held - size).is_ok() {
            return Ok(());
        }
        for f in &mut self.open {
            f.encoder.flush().map_err(|e| e.into_error(&f.at.display))?;
            drain(f).await?;
            f.upload.wait_idle().await?;
        }
        let grow = self.held().saturating_sub(self.mem.size());
        if grow > 0 {
            self.mem.grow(grow).await?;
        }
        Ok(())
    }

    fn held(&self) -> usize {
        self.open
            .iter()
            .map(|f| {
                f.encoder
                    .buffered_bytes()
                    .saturating_add(usize::try_from(f.upload.held_bytes()).unwrap_or(usize::MAX))
            })
            .fold(0, usize::saturating_add)
    }
}

impl Drop for DataFileWriter {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        // Open uploads abort on their own drop; finished files are deleted
        // in the background.
        let files = std::mem::take(&mut self.finished);
        if files.is_empty() {
            return;
        }
        let io = self.io.clone();
        let delete = async move {
            for f in files {
                let _ = io.delete_pinned(&f.store, &f.at.path).await;
            }
        };
        // The I/O runtime first: it outlives a query's compute runtime, which
        // may be shutting down when a cancelled driver drops its writer.
        let handle = self
            .io
            .scheduler()
            .runtime()
            .cloned()
            .or_else(|| tokio::runtime::Handle::try_current().ok());
        if let Some(h) = handle {
            h.spawn(delete);
        }
    }
}

async fn drain(f: &mut OpenFile) -> Result<(), KrishivError> {
    while let Some(bytes) = f.encoder.take_bytes() {
        f.upload.write(bytes).await?;
    }
    Ok(())
}

/// The Parquet options of a write into a table with `props` (module docs).
fn parquet_options(props: &HashMap<String, String>) -> Result<FormatOptions, KrishivError> {
    let codec = props
        .get(PARQUET_CODEC)
        .map_or(PARQUET_CODEC_DEFAULT, String::as_str);
    // Java's `CompressionCodecName.valueOf(codec.toUpperCase(Locale.ENGLISH))`.
    let name = codec.to_ascii_lowercase();
    match name.as_str() {
        "uncompressed" | "snappy" | "gzip" | "brotli" | "lz4" | "lz4_raw" | "zstd" => {}
        "lzo" => return Err(unsupported("writing the lzo Parquet codec".to_owned())),
        _ => {
            return Err(unsupported(format!(
                "writing Iceberg data files: Unsupported compression codec: {codec}"
            )));
        }
    }
    let row_group = int_property(props, ROW_GROUP_SIZE, ROW_GROUP_SIZE_DEFAULT)?;
    let page = int_property(props, PAGE_SIZE, PAGE_SIZE_DEFAULT)?;
    let user = [
        ("compression", name),
        ("parquet.block.size", row_group.to_string()),
        ("parquet.page.size", page.to_string()),
        ("outputTimestampType", "TIMESTAMP_MICROS".to_owned()),
        ("datetimeRebaseModeInWrite", "CORRECTED".to_owned()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), OptionValue::Plain(v)))
    .collect();
    let (options, _) = ParquetFormat.options().parse(&user, &Config::new())?;
    Ok(options)
}

/// A positive int property (Java's `PropertyUtil.propertyAsInt`; Krishiv
/// also refuses a value below 1, which parquet-mr cannot use).
fn int_property(
    props: &HashMap<String, String>,
    key: &str,
    default: i32,
) -> Result<i32, KrishivError> {
    match props.get(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<i32>()
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| bad_property(key, v)),
    }
}

/// `write.target-file-size-bytes` (Java's `propertyAsLong`); a value below 0
/// rolls over as 0 does (every 1,000 rows).
fn target_file_bytes(props: &HashMap<String, String>) -> Result<u64, KrishivError> {
    let n = match props.get(TARGET_FILE_SIZE) {
        None => TARGET_FILE_SIZE_DEFAULT,
        Some(v) => v
            .parse::<i64>()
            .map_err(|_| bad_property(TARGET_FILE_SIZE, v))?,
    };
    Ok(u64::try_from(n).unwrap_or(0))
}

fn bad_property(key: &str, value: &str) -> KrishivError {
    unsupported(format!(
        "writing an Iceberg table whose {key} is {value:?}, not a positive number"
    ))
}

/// Refuses column types Krishiv cannot write as Java does.
fn check_writable(name: &str, ty: &Type) -> Result<(), KrishivError> {
    match ty {
        Type::Primitive(PrimitiveType::Uuid | PrimitiveType::Fixed(_)) => Err(unsupported(
            format!("writing the Iceberg column {name} of type {ty}"),
        )),
        Type::Variant(_) => Err(unsupported(format!(
            "writing the Iceberg variant column {name}"
        ))),
        Type::Primitive(_) => Ok(()),
        Type::Struct(s) => s
            .fields()
            .iter()
            .try_for_each(|f| check_writable(&f.name, &f.field_type)),
        Type::List(l) => check_writable(name, &l.element_field.field_type),
        Type::Map(m) => {
            check_writable(name, &m.key_field.field_type)?;
            check_writable(name, &m.value_field.field_type)
        }
    }
}

/// How the partition field over `source_id` with `transform` is read.
fn field_plan(
    schema: &SchemaRef,
    arrow: &arrow_schema::SchemaRef,
    source_id: i32,
    transform: Transform,
) -> Result<FieldPlan, KrishivError> {
    if transform == Transform::Void {
        return Ok(FieldPlan {
            transform,
            source: None,
        });
    }
    let column = schema_fields(schema)
        .into_iter()
        .find(|f| f.id == source_id)
        .ok_or_else(|| {
            unsupported(format!(
                "writing an Iceberg partition of source column {source_id}, which is not a \
                 primitive reachable through structs"
            ))
        })?;
    let names = &column.path.0;
    let top = names
        .first()
        .and_then(|n| arrow.index_of(n).ok())
        .ok_or_else(|| {
            internal(format!(
                "partition source {} not in the schema",
                column.path
            ))
        })?;
    let p = schema
        .field_by_id(source_id)
        .and_then(|f| f.field_type.as_primitive_type().cloned())
        .ok_or_else(|| internal(format!("partition source {source_id} is not primitive")))?;
    Ok(FieldPlan {
        transform,
        source: Some((top, names[1..].to_vec(), iceberg_type(&p))),
    })
}

/// The arrays from the top-level column `top` down `path` (structs).
fn source_chain(
    batch: &RecordBatch,
    top: usize,
    path: &[String],
) -> Result<Vec<ArrayRef>, KrishivError> {
    let mut chain = vec![Arc::clone(batch.column(top))];
    for name in path {
        let parent = chain.last().map(AsArray::as_struct_opt);
        let child = parent
            .flatten()
            .and_then(|s| s.column_by_name(name))
            .ok_or_else(|| internal(format!("partition source field {name} not found")))?;
        chain.push(Arc::clone(child));
    }
    Ok(chain)
}

/// The value of the last array of `chain` at `row`; `None` when it or a
/// struct above it is null.
fn value_at(chain: &[ArrayRef], row: usize) -> Result<Option<ScalarValue>, KrishivError> {
    if chain.iter().any(|a| a.is_null(row)) {
        return Ok(None);
    }
    let Some(a) = chain.last() else {
        return Ok(None);
    };
    Ok(Some(match a.data_type() {
        DataType::Boolean => ScalarValue::Boolean(a.as_boolean().value(row)),
        DataType::Int32 => ScalarValue::Int32(a.as_primitive::<Int32Type>().value(row)),
        DataType::Int64 => ScalarValue::Int64(a.as_primitive::<Int64Type>().value(row)),
        DataType::Float32 => ScalarValue::Float32(a.as_primitive::<Float32Type>().value(row)),
        DataType::Float64 => ScalarValue::Float64(a.as_primitive::<Float64Type>().value(row)),
        DataType::Date32 => ScalarValue::Date32(a.as_primitive::<Date32Type>().value(row)),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            ScalarValue::TimestampMicros(a.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        DataType::Decimal128(precision, scale) => ScalarValue::Decimal128 {
            unscaled: a.as_primitive::<Decimal128Type>().value(row),
            precision: *precision,
            scale: *scale,
        },
        DataType::Utf8 => ScalarValue::Utf8(a.as_string::<i32>().value(row).to_owned()),
        DataType::LargeUtf8 => ScalarValue::Utf8(a.as_string::<i64>().value(row).to_owned()),
        DataType::Utf8View => ScalarValue::Utf8(a.as_string_view().value(row).to_owned()),
        DataType::Binary => ScalarValue::Binary(a.as_binary::<i32>().value(row).to_vec()),
        DataType::LargeBinary => ScalarValue::Binary(a.as_binary::<i64>().value(row).to_vec()),
        DataType::BinaryView => ScalarValue::Binary(a.as_binary_view().value(row).to_vec()),
        other => {
            return Err(internal(format!("partition source of Arrow type {other}")));
        }
    }))
}

/// A partition key as bytes: equal tuples give equal bytes. Floats compare by
/// bits with every NaN made one NaN, as Java's `Float.equals`/`Double.equals`
/// (`floatToIntBits` collapses NaN payloads; -0.0 and 0.0 stay apart).
fn key_bytes(values: &[Option<ScalarValue>], out: &mut Vec<u8>) {
    for v in values {
        let mut put = |tag: u8, bytes: &[u8]| {
            out.push(tag);
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(bytes);
        };
        match v {
            None => put(0, &[]),
            Some(ScalarValue::Boolean(b)) => put(1, &[u8::from(*b)]),
            Some(ScalarValue::Int32(x)) => put(2, &x.to_le_bytes()),
            Some(ScalarValue::Int64(x)) => put(3, &x.to_le_bytes()),
            Some(ScalarValue::Float32(x)) => {
                let x = if x.is_nan() { f32::NAN } else { *x };
                put(4, &x.to_bits().to_le_bytes());
            }
            Some(ScalarValue::Float64(x)) => {
                let x = if x.is_nan() { f64::NAN } else { *x };
                put(5, &x.to_bits().to_le_bytes());
            }
            Some(ScalarValue::Decimal128 { unscaled, .. }) => put(6, &unscaled.to_le_bytes()),
            Some(ScalarValue::Utf8(s)) => put(7, s.as_bytes()),
            Some(ScalarValue::Binary(b)) => put(8, b),
            Some(ScalarValue::Date32(x)) => put(9, &x.to_le_bytes()),
            Some(ScalarValue::TimestampMicros(x)) => put(10, &x.to_le_bytes()),
            Some(other) => put(11, format!("{other:?}").as_bytes()),
        }
    }
}

/// `array` with the type `target`: the same data with the target's nested
/// field names, nullability and metadata (field ids).
///
/// # Errors
/// `INTERNAL_ERROR` when the shapes differ (another type, another struct
/// field name or count).
fn retype(array: &ArrayRef, target: &DataType) -> Result<ArrayRef, KrishivError> {
    if array.data_type() == target {
        return Ok(Arc::clone(array));
    }
    let mismatch = || {
        internal(format!(
            "an Iceberg write's column of type {} where {target} is expected",
            array.data_type()
        ))
    };
    let arrow_err =
        |e: arrow_schema::ArrowError| internal(format!("an Iceberg write's column: {e}"));
    Ok(match (array.data_type(), target) {
        (DataType::Struct(got), DataType::Struct(fields)) => {
            if got.len() != fields.len()
                || got
                    .iter()
                    .zip(fields.iter())
                    .any(|(a, b)| a.name() != b.name())
            {
                return Err(mismatch());
            }
            let s = array.as_struct();
            let children = s
                .columns()
                .iter()
                .zip(fields.iter())
                .map(|(c, f)| retype(c, f.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                StructArray::try_new(fields.clone(), children, s.nulls().cloned())
                    .map_err(arrow_err)?,
            )
        }
        (DataType::List(_), DataType::List(f)) => {
            let l = array.as_list::<i32>();
            Arc::new(
                ListArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    retype(l.values(), f.data_type())?,
                    l.nulls().cloned(),
                )
                .map_err(arrow_err)?,
            )
        }
        (DataType::LargeList(_), DataType::LargeList(f)) => {
            let l = array.as_list::<i64>();
            Arc::new(
                LargeListArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    retype(l.values(), f.data_type())?,
                    l.nulls().cloned(),
                )
                .map_err(arrow_err)?,
            )
        }
        (DataType::Map(_, _), DataType::Map(f, sorted)) => {
            let m = array.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = retype(&entries, f.data_type())?;
            Arc::new(
                MapArray::try_new(
                    Arc::clone(f),
                    m.offsets().clone(),
                    entries.as_struct().clone(),
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(arrow_err)?,
            )
        }
        _ => return Err(mismatch()),
    })
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod key_tests {
    use krishiv_io::ScalarValue;

    use super::key_bytes as into;

    fn key_bytes(values: &[Option<ScalarValue>]) -> Vec<u8> {
        let mut out = Vec::new();
        into(values, &mut out);
        out
    }

    #[test]
    fn nan_payloads_are_one_key_and_signed_zeros_two() {
        let f64_key = |x: f64| key_bytes(&[Some(ScalarValue::Float64(x))]);
        let f32_key = |x: f32| key_bytes(&[Some(ScalarValue::Float32(x))]);
        assert_eq!(
            f64_key(f64::NAN),
            f64_key(f64::from_bits(0x7ff0_0000_0000_0001))
        );
        assert_eq!(f64_key(f64::NAN), f64_key(-f64::NAN));
        assert_eq!(f32_key(f32::NAN), f32_key(f32::from_bits(0x7f80_0001)));
        assert_ne!(f64_key(0.0), f64_key(-0.0));
        assert_ne!(f32_key(0.0), f32_key(-0.0));
    }
}
