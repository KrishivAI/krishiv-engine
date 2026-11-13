// SPDX-License-Identifier: Apache-2.0
//! One write task's data files for an append (P3-T027, PR-36a; design note
//! §6.1, distribution `none`): rows of the table's Arrow schema
//! ([`arrow_schema`]) in, Parquet files with the table's field ids out, each
//! described by a [`WrittenDataFile`] for its manifest entry.
//!
//! - **Partitioning.** Each row's partition key is computed from the default
//!   spec's fields ([`partition_value`] over the source column; a null source
//!   or `void` gives a null value, a transform Krishiv cannot compute fails the
//!   write). Rows are grouped by key and taken per key. A spec whose fields
//!   are all `void` (an unpartitioned one included) does no per-row work.
//!   The per-row cost is one `ScalarValue` per partition field and row.
//! - **Fanout** (Java's `FanoutDataWriter` shape): one open file per key, at
//!   most `max_open_files` open (Krishiv's own cap, design note §6.1; Java's
//!   fanout writer has none), the oldest-opened closed first; a key seen
//!   again after its file closed gets a new file. A file rolls over to a new
//!   one when, at a multiple of 1,000 rows, its size reaches
//!   `write.target-file-size-bytes` (Java's `RollingFileWriter`:
//!   `ROWS_DIVISOR` = 1000, `length >= targetFileSize`).
//! - **Parquet settings** from the table properties, as Java's `Parquet`
//!   write builder reads them (defaults as iceberg-rust 0.11's port,
//!   `src/spec/table_properties.rs:437, 507, 516, 521`):
//!   `write.parquet.compression-codec` (`zstd`) is the codec,
//!   `write.parquet.row-group-size-bytes` (128 MiB) the row-group size
//!   (`parquet.block.size`), `write.parquet.page-size-bytes` (1 MiB) the page
//!   size; timestamps are INT64 microseconds, never rebased. The codec's
//!   level (`write.parquet.compression-level`) is not applied: the Parquet
//!   writer has no level option (zstd writes level 3, parquet-mr's default).
//!   `write.format.default` other than `parquet` is refused.
//! - **Names and locations:** [`file_name`] with a counter per task from 1
//!   (Java's `OutputFileFactory`), placed by [`DataLocations::location`] for
//!   the file's partition tuple.
//! - **Abort:** [`DataFileWriter::finish`] failing, [`DataFileWriter::abort`]
//!   and a drop before `finish` abort the open uploads (nothing becomes
//!   visible) and delete the files already finished (a drop deletes in a
//!   task spawned on the current runtime).
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
use super::storage::StorageRegistration;
use super::types::arrow_schema;
use super::write_metrics::WriteMetricsConfig;
use crate::plan::file_metrics::file_metrics;
use crate::plan::transform::partition_value;
use crate::plan::types::{IcebergType, SchemaField, Transform};

/// Open files per task by default (the future `krishiv.iceberg.write.maxOpenFiles`).
pub(crate) const DEFAULT_MAX_OPEN_FILES: usize = 100;

/// `write.target-file-size-bytes` and its default (Java's
/// `WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT`, 512 MiB).
pub(crate) const TARGET_FILE_SIZE: &str = "write.target-file-size-bytes";
const TARGET_FILE_SIZE_DEFAULT: i64 = 512 * 1024 * 1024;
/// `write.parquet.compression-codec` and its default (`zstd` since Iceberg 1.4).
pub(crate) const PARQUET_CODEC: &str = "write.parquet.compression-codec";
const PARQUET_CODEC_DEFAULT: &str = "zstd";
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

/// Who writes: the task's partition and task ids and the write's id, which
/// name its files ([`file_name`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskIds<'a> {
    /// The task's partition id.
    pub(crate) partition_id: u32,
    /// The task id.
    pub(crate) task_id: u64,
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
    task_id: u64,
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
    pub(crate) fn new(
        metadata: &TableMetadata,
        registration: &StorageRegistration,
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
        let registered = registration.registered()?;
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
            task_id: ids.task_id,
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
            return self.write_group(&key_bytes(&values), &values, &batch).await;
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

    async fn abort_files(&mut self) {
        self.done = true;
        for f in std::mem::take(&mut self.open) {
            f.upload.abort().await;
        }
        for f in std::mem::take(&mut self.finished) {
            // Best effort: a file left behind is not in any manifest.
            let _ = self.io.delete_pinned(&f.store, &f.at.path).await;
        }
        let held = self.mem.size();
        self.mem.shrink(held);
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
            let key = key_bytes(&values);
            let at = match index.get(&key) {
                Some(at) => *at,
                None => {
                    index.insert(key.clone(), groups.len());
                    groups.push((key, values.clone(), Vec::new()));
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
        let name = file_name(self.partition_id, self.task_id, &self.write_id, self.files);
        let partition = partition_tuple(&self.spec, &self.schema, values)?;
        let location = self
            .locations
            .location(&self.spec, &self.schema, &partition, &name)?;
        let at = self.paths.locate(&location)?;
        let store = self.io.pin(&at.store)?;
        let seed = format!(
            "{}/{}/{}/{}",
            self.write_id, self.partition_id, self.task_id, self.files
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
        let handle = tokio::runtime::Handle::try_current()
            .ok()
            .or_else(|| self.io.scheduler().runtime().cloned());
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

/// A partition key as bytes: equal tuples give equal bytes (floats by bits).
fn key_bytes(values: &[Option<ScalarValue>]) -> Vec<u8> {
    let mut out = Vec::new();
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
            Some(ScalarValue::Float32(x)) => put(4, &x.to_bits().to_le_bytes()),
            Some(ScalarValue::Float64(x)) => put(5, &x.to_bits().to_le_bytes()),
            Some(ScalarValue::Decimal128 { unscaled, .. }) => put(6, &unscaled.to_le_bytes()),
            Some(ScalarValue::Utf8(s)) => put(7, s.as_bytes()),
            Some(ScalarValue::Binary(b)) => put(8, b),
            Some(ScalarValue::Date32(x)) => put(9, &x.to_le_bytes()),
            Some(ScalarValue::TimestampMicros(x)) => put(10, &x.to_le_bytes()),
            Some(other) => put(11, format!("{other:?}").as_bytes()),
        }
    }
    out
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
