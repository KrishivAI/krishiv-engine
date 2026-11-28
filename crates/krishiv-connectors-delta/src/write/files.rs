// SPDX-License-Identifier: Apache-2.0
//! The data files of one write (design note P3-T021 section 4.1, "Data files" and "Adds"),
//! written while the rows arrive: each driver of the write owns a [`DriverWriter`], which keeps
//! one open file per partition it has seen and rolls it over at `maxRecordsPerFile` and at the
//! target file size ([`FileLimits`]).
//!
//! - Each partition is bound once to the transaction's write state ([`BoundWriteContext`]); its
//!   rows are turned into the physical schema by the kernel's `logical_to_physical` expression
//!   (physical names and Parquet field ids under column mapping) and streamed through the
//!   Parquet encoder into a multipart upload under the context's `write_dir`. A file buffers at
//!   most the row group being built and the upload's parts; every byte of it is held in the
//!   driver's reservation, and [`DriverWriter::flush`] (the sink's spill) encodes and ships the
//!   buffered row groups when memory is short.
//! - Each file is described by one add row with Krishiv's own statistics ([`super::stats`]),
//!   accumulated over its batches.
//! - Files are invisible until the commit names them. A write that fails before its commit
//!   deletes the files it finished and aborts its open uploads ([`DriverWriter::abort`]), best
//!   effort; a file left behind is an orphan no snapshot reads (as a failed delta-spark task's
//!   files are).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{Field, Schema, SchemaRef};
use delta_kernel::expressions::{MapData, PredicateRef, Scalar};
use delta_kernel::schema::{DataType, MapType, StructType};
use delta_kernel::transaction::{BoundWriteContext, WriteState};
use delta_kernel::{EvaluationHandler as _, ExpressionEvaluator};
use krishiv_common::error::{DeltaNotNullConstraintViolated, InternalError, KrishivError};
use krishiv_io::Encoder;
use krishiv_io::io::Upload;
use krishiv_memory::Reservation;
use object_store::path::Path;
use url::Url;

use super::stats::StatsAccumulator;
use crate::engine::eval::create_many;
use crate::engine::relabel::relabel_batch;
use crate::engine::schema::scalar_at;
use crate::engine::{DeltaEngine, KrishivEngineData};
use crate::error::from_kernel;

/// The target size of a data file (delta-spark leaves it to Spark's task output; 128 MiB is
/// the `delta.targetFileSize` default of its optimized writes). A `krishiv.delta.*` key for it
/// is a follow-up.
pub(crate) const DEFAULT_TARGET_FILE_BYTES: u64 = 128 * 1024 * 1024;

/// Rows written between two checks of the target file size.
const ROLLOVER_ROWS: usize = 1024;

/// When a data file is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileLimits {
    /// Delta's `maxRecordsPerFile` write option; 0 = unlimited.
    pub(crate) max_records: u64,
    /// Roll over once the encoder produced this many bytes; 0 = never.
    pub(crate) target_bytes: u64,
}

impl Default for FileLimits {
    fn default() -> Self {
        Self {
            max_records: 0,
            target_bytes: DEFAULT_TARGET_FILE_BYTES,
        }
    }
}

/// What the files of one write are written against, made once per write.
pub(crate) struct Plan {
    /// The kernel's write state of the transaction.
    pub(crate) state: Arc<WriteState>,
    /// The kernel's stats schema of the transaction.
    pub(crate) stats_schema: Arc<StructType>,
    /// The table's logical columns in the kernel's Arrow form.
    pub(crate) logical: SchemaRef,
    /// The logical partition columns, in the table's order.
    pub(crate) partition_columns: Vec<String>,
    /// Whether the table supports deletion vectors (`tightBounds`).
    pub(crate) tight_bounds: bool,
    /// The table, for messages.
    pub(crate) table: String,
    /// When a file is full.
    pub(crate) limits: FileLimits,
    /// `replaceWhere`'s constraint: every row's partition values satisfy the predicate (its
    /// text for the message), with the table's logical schema.
    pub(crate) check: Option<(PredicateRef, String, Arc<StructType>)>,
}

/// What one write produced: its add rows (in the kernel's `add_files_schema`, `stats` holding
/// Krishiv's statistics) and its metrics.
#[derive(Debug, Default)]
pub(crate) struct Written {
    /// One row per file.
    pub(crate) adds: Vec<RecordBatch>,
    /// Files written.
    pub(crate) files: u64,
    /// Rows written.
    pub(crate) rows: u64,
    /// Bytes written.
    pub(crate) bytes: u64,
    /// The physical partition values of each partition written (once per partition).
    pub(crate) partitions: Vec<HashMap<String, Option<String>>>,
    /// The logical partition values of each partition written, in the partition columns'
    /// order (once per partition).
    pub(crate) partition_values: Vec<Vec<Scalar>>,
    /// The files, for a failed write's clean-up.
    pub(crate) paths: Vec<Path>,
    /// The sorted physical values of each of `partitions`, for [`Self::merge`]'s dedup.
    seen: HashSet<PartitionKey>,
}

/// A partition's physical values, sorted by column: a hashable form of its map.
type PartitionKey = Vec<(String, Option<String>)>;

fn partition_key(values: &HashMap<String, Option<String>>) -> PartitionKey {
    let mut key: PartitionKey = values.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    key.sort_unstable();
    key
}

impl Written {
    /// Adds `other`'s files; a partition both wrote is kept once (a hash set: linear in the
    /// partitions over every driver).
    pub(crate) fn merge(&mut self, other: Self) {
        self.adds.extend(other.adds);
        self.files += other.files;
        self.rows += other.rows;
        self.bytes += other.bytes;
        self.paths.extend(other.paths);
        if self.seen.len() != self.partitions.len() {
            self.seen = self.partitions.iter().map(partition_key).collect();
        }
        for (physical, logical) in other.partitions.into_iter().zip(other.partition_values) {
            if self.seen.insert(partition_key(&physical)) {
                self.partitions.push(physical);
                self.partition_values.push(logical);
            }
        }
    }
}

/// A batch's rows of one partition: its key (the row format of its values), its logical
/// values and the rows.
type Group = (Vec<u8>, Vec<Scalar>, RecordBatch);

/// One partition a driver writes: its binding, evaluator and open file.
struct Partition {
    ctx: BoundWriteContext,
    evaluator: Arc<dyn ExpressionEvaluator>,
    input_schema: SchemaRef,
    values: Vec<Scalar>,
    file: Option<OpenFile>,
}

/// A file being written.
struct OpenFile {
    encoder: Box<dyn Encoder>,
    upload: Upload,
    url: Url,
    path: Path,
    schema: SchemaRef,
    rows: u64,
    stats: StatsAccumulator,
}

impl OpenFile {
    /// Bytes it holds in memory: the encoder's buffers and the upload's parts.
    fn held(&self) -> usize {
        self.encoder
            .buffered_bytes()
            .saturating_add(usize::try_from(self.upload.held_bytes()).unwrap_or(usize::MAX))
    }

    fn display(&self) -> String {
        self.url.to_string()
    }

    async fn drain(&mut self) -> Result<(), KrishivError> {
        while let Some(bytes) = self.encoder.take_bytes() {
            self.upload.write(bytes).await?;
        }
        Ok(())
    }
}

/// One driver's files (see the module docs).
pub(crate) struct DriverWriter {
    engine: DeltaEngine,
    mem: Reservation,
    /// Bytes held past the reservation after the last batch (the pool refused them): the next
    /// batch asks for them first.
    pub(crate) deficit: usize,
    /// The first error of a batch written on the driver's behalf.
    pub(crate) failed: Option<KrishivError>,
    /// The partition key encoder (the partition columns' row format).
    converter: Option<RowConverter>,
    partitions: HashMap<Vec<u8>, Partition>,
    done: Written,
}

impl std::fmt::Debug for DriverWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverWriter")
            .field("partitions", &self.partitions.len())
            .field("files", &self.done.files)
            .finish_non_exhaustive()
    }
}

impl DriverWriter {
    /// A writer holding its memory in `mem` (the driver's consumer).
    pub(crate) fn new(engine: DeltaEngine, mem: Reservation) -> Self {
        Self {
            engine,
            mem,
            deficit: 0,
            failed: None,
            converter: None,
            partitions: HashMap::new(),
            done: Written::default(),
        }
    }

    /// The reservation, for the sink's `try_grow` before a batch.
    pub(crate) fn reservation(&mut self) -> &mut Reservation {
        &mut self.mem
    }

    /// Bytes held by the open files.
    fn held(&self) -> usize {
        self.partitions
            .values()
            .filter_map(|p| p.file.as_ref())
            .map(OpenFile::held)
            .fold(0, usize::saturating_add)
    }

    /// Writes `batch` (rows of the table's logical columns, in its order), then fits the
    /// reservation to what the open files hold.
    ///
    /// # Errors
    /// `replaceWhere`'s constraint; the kernel's partition-value and write-state errors; the
    /// evaluator's; encode and storage errors.
    pub(crate) async fn write(
        &mut self,
        plan: &Plan,
        batch: &RecordBatch,
    ) -> Result<(), KrishivError> {
        if batch.num_rows() > 0 {
            let batch = conform(batch, &plan.logical)?;
            let batch = null_empty_partition_values(batch, &plan.partition_columns)?;
            if let Some((pred, text, schema)) = &plan.check {
                super::sink::check_filter(
                    &self.engine,
                    schema,
                    &plan.partition_columns,
                    std::slice::from_ref(&batch),
                    pred,
                    text,
                    &plan.table,
                )?;
            }
            for (key, values, rows) in self.groups(plan, batch)? {
                self.write_group(plan, key, values, &rows).await?;
            }
        }
        self.settle().await
    }

    /// The rows of `batch` by partition: key, logical values and rows.
    fn groups(&mut self, plan: &Plan, batch: RecordBatch) -> Result<Vec<Group>, KrishivError> {
        if plan.partition_columns.is_empty() {
            return Ok(vec![(Vec::new(), Vec::new(), batch)]);
        }
        let table = plan.table.as_str();
        let columns =
            plan.partition_columns
                .iter()
                .map(|p| {
                    batch.column_by_name(p).cloned().ok_or_else(|| {
                        internal(&format!("the partition column {p} is not written"))
                    })
                })
                .collect::<Result<Vec<ArrayRef>, _>>()?;
        if self.converter.is_none() {
            let fields = columns
                .iter()
                .map(|c| SortField::new(c.data_type().clone()))
                .collect();
            self.converter = Some(
                RowConverter::new(fields)
                    .map_err(|e| internal(&format!("the partition key format: {e}")))?,
            );
        }
        let converter = self
            .converter
            .as_ref()
            .ok_or_else(|| internal("no partition key format"))?;
        let keys = converter
            .convert_columns(&columns)
            .map_err(|e| internal(&format!("the partition keys: {e}")))?;
        // One entry per partition of the batch, in first-seen order.
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let mut groups: Vec<(usize, Vec<u32>)> = Vec::new();
        for row in 0..batch.num_rows() {
            let key = keys.row(row);
            let at = u32::try_from(row).map_err(|_| internal("a batch row past u32"))?;
            match index.get(key.as_ref()) {
                Some(&g) => groups[g].1.push(at),
                None => {
                    index.insert(key.data(), groups.len());
                    groups.push((row, vec![at]));
                }
            }
        }
        let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
        let single = groups.len() == 1;
        let mut out = Vec::with_capacity(groups.len());
        for (first, rows) in groups {
            let values = columns
                .iter()
                .map(|c| scalar_at(c.as_ref(), first))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| kernel(&e))?;
            let key = keys.row(first).as_ref().to_vec();
            let part = if single {
                batch.clone()
            } else {
                let indices = arrow_array::UInt32Array::from(rows);
                arrow_select::take::take_record_batch(&batch, &indices)
                    .map_err(|e| internal(&format!("grouping rows by partition: {e}")))?
            };
            out.push((key, values, part));
        }
        Ok(out)
    }

    /// Writes one partition's rows, opening and rolling over its files.
    async fn write_group(
        &mut self,
        plan: &Plan,
        key: Vec<u8>,
        values: Vec<Scalar>,
        rows: &RecordBatch,
    ) -> Result<(), KrishivError> {
        let table = plan.table.as_str();
        if !self.partitions.contains_key(&key) {
            let part = self.bind(plan, values).await?;
            self.partitions.insert(key.clone(), part);
        }
        let Some(part) = self.partitions.get_mut(&key) else {
            return Err(internal("a partition that was just bound is missing"));
        };
        let mut offset = 0;
        while offset < rows.num_rows() {
            self.engine
                .check_cancelled()
                .map_err(|e| from_kernel(&e, table))?;
            let room = match (plan.limits.max_records, &part.file) {
                (0, _) | (_, None) => usize::MAX,
                (max, Some(f)) => usize::try_from(max.saturating_sub(f.rows)).unwrap_or(usize::MAX),
            };
            let room = if plan.limits.max_records > 0 && part.file.is_none() {
                usize::try_from(plan.limits.max_records).unwrap_or(usize::MAX)
            } else {
                room
            };
            let step = if plan.limits.target_bytes > 0 {
                ROLLOVER_ROWS
            } else {
                usize::MAX
            };
            let n = room.min(step).min(rows.num_rows() - offset).max(1);
            let slice = rows.slice(offset, n);
            offset += n;
            let physical = physical(part, &slice, table)?;
            if part.file.is_none() {
                part.file = Some(open(&self.engine, &part.ctx, physical.schema()).await?);
            }
            let Some(f) = part.file.as_mut() else {
                return Err(internal("a file that was just opened is missing"));
            };
            f.stats.update(&physical, &plan.stats_schema)?;
            let physical = relabel_batch(&physical, Arc::clone(&f.schema))
                .map_err(|e| internal(&format!("a data file's rows: {e}")))?;
            let display = f.display();
            f.encoder
                .write(&physical)
                .map_err(|e| e.into_error(&display))?;
            f.rows += n as u64;
            f.drain().await?;
            let full = (plan.limits.max_records > 0 && f.rows >= plan.limits.max_records)
                || (plan.limits.target_bytes > 0
                    && f.encoder.written_bytes() >= plan.limits.target_bytes);
            if full && let Some(f) = part.file.take() {
                // Listed before the close: a close that fails (or is dropped) after its upload
                // landed still has its file deleted by `abort`.
                self.done.paths.push(f.path.clone());
                let add = close(plan, &part.ctx, f).await?;
                self.done.push(add);
            }
        }
        Ok(())
    }

    /// Binds a partition's values to the write state (a kernel call: on the log pool).
    async fn bind(&mut self, plan: &Plan, values: Vec<Scalar>) -> Result<Partition, KrishivError> {
        let state = Arc::clone(&plan.state);
        let columns = plan.partition_columns.clone();
        let table = plan.table.clone();
        let engine = self.engine.clone();
        engine
            .delta_blocking(move |e| bind(e, &state, &columns, values, &table))
            .await?
    }

    /// Fits the reservation to the bytes the open files hold: shrinks it, or grows it; when
    /// the pool refuses, encodes and ships the buffered row groups first, and what is still
    /// refused is the next batch's [`deficit`](Self::deficit).
    pub(crate) async fn settle(&mut self) -> Result<(), KrishivError> {
        self.deficit = 0;
        let held = self.held();
        let size = self.mem.size();
        if held <= size {
            self.mem.shrink(size - held);
            return Ok(());
        }
        if self.mem.try_grow(held - size).is_ok() {
            return Ok(());
        }
        self.flush().await?;
        let held = self.held();
        let size = self.mem.size();
        if held <= size {
            self.mem.shrink(size - held);
        } else if self.mem.try_grow(held - size).is_err() {
            self.deficit = held - size;
        }
        Ok(())
    }

    /// Encodes the buffered row groups of every open file and waits for their parts to be
    /// uploaded (the sink's spill); returns the bytes given back.
    ///
    /// # Errors
    /// Encode and upload errors.
    pub(crate) async fn flush(&mut self) -> Result<usize, KrishivError> {
        for p in self.partitions.values_mut() {
            if let Some(f) = p.file.as_mut() {
                let display = f.display();
                f.encoder.flush().map_err(|e| e.into_error(&display))?;
                f.drain().await?;
                f.upload.wait_idle().await?;
            }
        }
        let held = self.held();
        let size = self.mem.size();
        let freed = size.saturating_sub(held);
        self.mem.shrink(freed);
        Ok(freed)
    }

    /// Ends every open file and returns what this driver wrote; its memory is given back. The
    /// files stay listed in the writer, so an [`abort`](Self::abort) after `finish` deletes them.
    ///
    /// # Errors
    /// Encode and upload errors (the caller then aborts the write).
    pub(crate) async fn finish(&mut self, plan: &Plan) -> Result<Written, KrishivError> {
        let mut keys: Vec<Vec<u8>> = self.partitions.keys().cloned().collect();
        keys.sort();
        for key in keys {
            let Some(part) = self.partitions.get_mut(&key) else {
                continue;
            };
            if let Some(f) = part.file.take() {
                self.done.paths.push(f.path.clone());
                let add = close(plan, &part.ctx, f).await?;
                self.done.push(add);
            }
        }
        let mut written = std::mem::take(&mut self.done);
        // The writer keeps its files' paths until the write lets it go
        // ([`WriteGlobal::release_writers`](super::sink)): a cancellation or failure before
        // the commit still deletes them through `abort`.
        self.done.paths.clone_from(&written.paths);
        for part in self.partitions.values() {
            if !plan.partition_columns.is_empty() {
                written
                    .partitions
                    .push(part.ctx.physical_partition_values().clone());
                written.partition_values.push(part.values.clone());
            }
        }
        let size = self.mem.size();
        self.mem.shrink(size);
        Ok(written)
    }

    /// Abandons the write: the open uploads are aborted and the finished files deleted, best
    /// effort; the memory is given back.
    pub(crate) async fn abort(&mut self) {
        for (_, p) in self.partitions.drain() {
            if let Some(f) = p.file {
                f.upload.abort().await;
            }
        }
        let paths = std::mem::take(&mut self.done.paths);
        delete_files(&self.engine, &paths).await;
        let size = self.mem.size();
        self.mem.shrink(size);
    }
}

/// A partition's binding, evaluator and logical input schema.
fn bind(
    e: &DeltaEngine,
    state: &Arc<WriteState>,
    partition_columns: &[String],
    values: Vec<Scalar>,
    table: &str,
) -> Result<Partition, KrishivError> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    let mut builder = state.write_context_builder();
    if !partition_columns.is_empty() {
        let map: HashMap<String, Scalar> = partition_columns
            .iter()
            .cloned()
            .zip(values.iter().cloned())
            .collect();
        builder = builder.with_partition_values(map);
    }
    let ctx = builder.build().map_err(|err| kernel(&err))?;
    let logical_data = Arc::clone(ctx.logical_data_schema());
    let input_schema: SchemaRef = Arc::new(
        crate::engine::schema::to_arrow_schema(&logical_data).map_err(|err| kernel(&err))?,
    );
    let evaluator = e
        .evaluation()
        .new_expression_evaluator(
            logical_data,
            ctx.logical_to_physical(),
            DataType::Struct(Box::new(ctx.physical_data_schema().as_ref().clone())),
        )
        .map_err(|err| kernel(&err))?;
    Ok(Partition {
        ctx,
        evaluator,
        input_schema,
        values,
        file: None,
    })
}

/// Deletes the data files `paths` of a failed write, best effort.
pub(crate) async fn delete_files(engine: &DeltaEngine, paths: &[Path]) {
    for path in paths {
        if let Err(e) = engine.io().delete(engine.store(), path).await {
            tracing::warn!(error = %e, %path, "could not delete a data file of a failed Delta write");
        }
    }
}

/// One closed file: its add row and size (its path was listed before the close).
struct Closed {
    add: RecordBatch,
    rows: u64,
    size: u64,
}

impl Written {
    fn push(&mut self, c: Closed) {
        self.adds.push(c.add);
        self.files += 1;
        self.rows += c.rows;
        self.bytes += c.size;
    }
}

/// `slice` of a partition's logical rows in the physical schema.
fn physical(
    part: &Partition,
    slice: &RecordBatch,
    table: &str,
) -> Result<RecordBatch, KrishivError> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    let columns = part
        .input_schema
        .fields()
        .iter()
        .map(|f| {
            let c = slice
                .column_by_name(f.name())
                .ok_or_else(|| internal(&format!("the column {} is not written", f.name())))?;
            // The kernel's logical schema labels nested fields with their column-mapping
            // metadata; the rows' types differ only in labels.
            crate::engine::relabel::relabel(c, f.data_type())
                .map_err(|e| internal(&format!("the column {}: {e}", f.name())))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input = RecordBatch::try_new(Arc::clone(&part.input_schema), columns)
        .map_err(|e| internal(&format!("a write's logical rows: {e}")))?;
    let out = part
        .evaluator
        .evaluate(&KrishivEngineData::new(input))
        .map_err(|e| kernel(&e))?;
    Ok(KrishivEngineData::from_engine_data(out)
        .map_err(|e| kernel(&e))?
        .into_batch())
}

/// Opens a new file of `ctx`'s partition for rows of `schema`.
async fn open(
    engine: &DeltaEngine,
    ctx: &BoundWriteContext,
    schema: SchemaRef,
) -> Result<OpenFile, KrishivError> {
    let name = format!(
        "part-00000-{}.c000{}",
        file_uuid(),
        crate::engine::columnar::file_suffix()?
    );
    let url = ctx
        .write_dir()
        .join(&name)
        .map_err(|err| internal(&format!("the data file {name}: {err}")))?;
    let path = engine.root().confine(&url)?;
    let format = crate::engine::columnar::format()?;
    let options = crate::engine::columnar::format_options()?;
    let encoder = format
        .writer(engine.io(), Arc::clone(&schema), &options)
        .await
        .map_err(|e| e.into_error(url.as_str()))?;
    let upload = engine.io().upload(engine.store(), path.clone())?;
    Ok(OpenFile {
        encoder,
        upload,
        url,
        path,
        schema,
        rows: 0,
        stats: StatsAccumulator::default(),
    })
}

/// Ends `f` and describes it.
async fn close(
    plan: &Plan,
    ctx: &BoundWriteContext,
    mut f: OpenFile,
) -> Result<Closed, KrishivError> {
    let table = plan.table.as_str();
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    f.drain().await?;
    let OpenFile {
        encoder,
        mut upload,
        url,
        rows,
        stats,
        ..
    } = f;
    let done = encoder.finish().map_err(|e| e.into_error(url.as_str()))?;
    upload.write(done.tail).await?;
    let size = upload.finish().await?;
    let stats = stats.finish(&plan.stats_schema, plan.tight_bounds)?;
    let relative = ctx.resolve_file_path(&url).map_err(|e| kernel(&e))?;
    let add = add_row(ctx, &relative, size, stats, table)?;
    Ok(Closed { add, rows, size })
}

/// The add row of one file: path, partition values, size, modification time, and the
/// statistics struct in place of the schema's empty one.
fn add_row(
    ctx: &BoundWriteContext,
    path: &str,
    size: u64,
    stats: StructArray,
    table: &str,
) -> Result<RecordBatch, KrishivError> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    let mandatory = StructType::new_unchecked([
        delta_kernel::schema::StructField::not_null("path", DataType::STRING),
        delta_kernel::schema::StructField::not_null(
            "partitionValues",
            MapType::new(DataType::STRING, DataType::STRING, true),
        ),
        delta_kernel::schema::StructField::not_null("size", DataType::LONG),
        delta_kernel::schema::StructField::not_null("modificationTime", DataType::LONG),
    ]);
    let mut values: Vec<(&String, &Option<String>)> =
        ctx.physical_partition_values().iter().collect();
    values.sort();
    let map = MapData::try_new(
        MapType::new(DataType::STRING, DataType::STRING, true),
        values.into_iter().map(|(k, v)| {
            (
                Scalar::String(k.clone()),
                v.as_ref().map_or(Scalar::Null(DataType::STRING), |v| {
                    Scalar::String(v.clone())
                }),
            )
        }),
    )
    .map_err(|e| kernel(&e))?;
    let size = i64::try_from(size).map_err(|_| internal("a file size past a long"))?;
    let base = create_many(
        &mandatory,
        &[vec![
            Scalar::String(path.to_owned()),
            Scalar::Map(map),
            Scalar::Long(size),
            Scalar::Long(now_ms()),
        ]],
    )
    .map_err(|e| kernel(&e))?;
    let mut fields: Vec<Field> = base
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut columns = base.columns().to_vec();
    fields.push(Field::new("stats", stats.data_type().clone(), true));
    columns.push(Arc::new(stats));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| internal(&format!("an add row: {e}")))
}

/// `batch` with each column in the type of the same column of `target` (the kernel's Arrow
/// form: the engine's string views, small decimals and zone labels to the kernel's).
/// Delta stores an empty string or binary partition value as null (the kernel's
/// `serialize_partition_value`; delta-spark turns `''` into null before it writes), so a
/// partition column's empty values become NULL before the rows are checked against
/// `replaceWhere` and grouped: the key, the add's logical values, the dynamic overwrite's
/// predicate and the check all see the value that is stored (`WRITE-EDGES.json`
/// `empty_partition`).
///
/// # Errors
/// An internal error when a partition column is missing or the NULL mask cannot be applied.
pub(crate) fn null_empty_partition_values(
    batch: RecordBatch,
    partitions: &[String],
) -> Result<RecordBatch, KrishivError> {
    use arrow_array::BooleanArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType as ArrowType;
    let mut columns: Option<Vec<ArrayRef>> = None;
    for p in partitions {
        let Some((at, _)) = batch.schema().column_with_name(p) else {
            return Err(internal(&format!(
                "the partition column {p} is not written"
            )));
        };
        let c = batch.column(at);
        let empty: Option<BooleanArray> = match c.data_type() {
            ArrowType::Utf8 => Some(
                c.as_string::<i32>()
                    .iter()
                    .map(|v| Some(v == Some("")))
                    .collect(),
            ),
            ArrowType::LargeUtf8 => Some(
                c.as_string::<i64>()
                    .iter()
                    .map(|v| Some(v == Some("")))
                    .collect(),
            ),
            ArrowType::Utf8View => Some(
                c.as_string_view()
                    .iter()
                    .map(|v| Some(v == Some("")))
                    .collect(),
            ),
            ArrowType::Binary => Some(
                c.as_binary::<i32>()
                    .iter()
                    .map(|v| Some(v.is_some_and(<[u8]>::is_empty)))
                    .collect(),
            ),
            ArrowType::LargeBinary => Some(
                c.as_binary::<i64>()
                    .iter()
                    .map(|v| Some(v.is_some_and(<[u8]>::is_empty)))
                    .collect(),
            ),
            ArrowType::BinaryView => Some(
                c.as_binary_view()
                    .iter()
                    .map(|v| Some(v.is_some_and(<[u8]>::is_empty)))
                    .collect(),
            ),
            _ => None,
        };
        let Some(empty) = empty.filter(|m| m.true_count() > 0) else {
            continue;
        };
        let nulled = arrow_select::nullif::nullif(c.as_ref(), &empty)
            .map_err(|e| internal(&format!("the partition column {p}: {e}")))?;
        // The NOT NULL check sees the stored value: delta-spark converts first, so `''` into a
        // NOT NULL partition column is `DELTA_NOT_NULL_CONSTRAINT_VIOLATED` naming it
        // (`WRITE-EDGES.json` `not_null_partition`).
        check_not_null(nulled.as_ref(), batch.schema().field(at), p, None)?;
        columns.get_or_insert_with(|| batch.columns().to_vec())[at] = nulled;
    }
    match columns {
        None => Ok(batch),
        Some(columns) => RecordBatch::try_new(batch.schema(), columns)
            .map_err(|e| internal(&format!("a write's rows: {e}"))),
    }
}

pub(crate) fn conform(
    batch: &RecordBatch,
    target: &SchemaRef,
) -> Result<RecordBatch, KrishivError> {
    if batch.num_columns() != target.fields().len() {
        return Err(internal(&format!(
            "a write of {} columns into a table of {}",
            batch.num_columns(),
            target.fields().len()
        )));
    }
    for (c, f) in batch.columns().iter().zip(target.fields()) {
        check_not_null(c.as_ref(), f, f.name(), None)?;
    }
    let columns = batch
        .columns()
        .iter()
        .zip(target.fields())
        .map(|(c, f)| {
            if c.data_type() == f.data_type() {
                Ok(Arc::clone(c))
            } else {
                crate::engine::relabel::relabel(c, f.data_type())
                    .or_else(|_| {
                        // Not the default safe cast: a value that does not fit fails, never
                        // turns into NULL (the session casts first; this catches a gap).
                        arrow_cast::cast_with_options(
                            c,
                            f.data_type(),
                            &arrow_cast::CastOptions {
                                safe: false,
                                ..arrow_cast::CastOptions::default()
                            },
                        )
                    })
                    .map_err(|e| {
                        internal(&format!(
                            "the column {} as {}: {e}",
                            f.name(),
                            f.data_type()
                        ))
                    })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(target), columns)
        .map_err(|e| internal(&format!("a write's rows in the table's types: {e}")))
}

/// delta-spark's NOT NULL invariant (`DeltaInvariantChecker`: `<path> IS NOT NULL` for each
/// non-nullable column, nested struct fields included): a row whose value at `path` is NULL,
/// because it or a struct above it is NULL, fails with `DELTA_NOT_NULL_CONSTRAINT_VIOLATED`
/// naming the dotted path (`WRITE-EDGES.json` `not_null`). Checked on the incoming column,
/// before any cast, so no cast refuses the NULL first. Struct fields are matched by name, then
/// by position. Fields inside arrays and maps are not checked here (Arrow's own validation
/// refuses a NULL there).
fn check_not_null(
    column: &dyn Array,
    field: &Field,
    path: &str,
    above: Option<&arrow_buffer::NullBuffer>,
) -> Result<(), KrishivError> {
    let own = column.logical_nulls();
    let nulls = arrow_buffer::NullBuffer::union(above, own.as_ref());
    if !field.is_nullable() && nulls.as_ref().is_some_and(|n| n.null_count() > 0) {
        return Err(DeltaNotNullConstraintViolated {
            column_name: path.to_owned(),
        }
        .into());
    }
    if let (arrow_schema::DataType::Struct(children), Some(s)) = (
        field.data_type(),
        column.as_any().downcast_ref::<StructArray>(),
    ) {
        for (i, child) in children.iter().enumerate() {
            let c = s
                .column_by_name(child.name())
                .or_else(|| s.columns().get(i));
            if let Some(c) = c {
                check_not_null(
                    c.as_ref(),
                    child,
                    &format!("{path}.{}", child.name()),
                    nulls.as_ref(),
                )?;
            }
        }
    }
    Ok(())
}

/// A fresh random UUID (version 4 layout) for a file name.
pub(crate) fn file_uuid() -> String {
    let hex: String = uuid_bytes().iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// The 16 bytes of a fresh random UUID (version 4 layout, big-endian).
pub(crate) fn uuid_bytes() -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let half = |salt: u64| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(salt);
        h.write_u64(n);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        h.finish()
    };
    let (a, b) = (half(1), half(2));
    let bits = (u128::from(a) << 64 | u128::from(b)) & !(0xF000_u128 << 64) | (0x4000_u128 << 64);
    let bits = bits & !(0xC000_u128 << 48) | (0x8000_u128 << 48);
    bits.to_be_bytes()
}

/// Milliseconds since the epoch.
pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}
