// SPDX-License-Identifier: Apache-2.0
//! [`FileWriter`]: the framework's write loop for one task — encoder bytes
//! to uploads, Spark file naming, rollover at `maxRecordsPerFile` and at a
//! target file size, staged files removed on abort.
//!
//! The executor's file sink feeds a `FileWriter` running on its own task
//! through a bounded channel, so its `sink` stays sync and reports
//! `Blocked` while the upload window is full. Files land under a staging
//! prefix; publishing them is the commit protocol's job (P2-T082a/b),
//! never a rename.
//!
//! Partitioned writes (P2-T082b unit 1) keep exactly one file open: the
//! planner sorts the input by the partition columns, so each partition's
//! rows arrive in one run and the writer starts a new file when the
//! partition key changes ([`partition`](crate::partition)).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use krishiv_memory::Reservation;
use object_store::path::Path;

use crate::error::{Result, internal, unsupported};
use crate::format::{Encoder, FileFormat};
use crate::io::{IoContext, StoreId, Upload};
use crate::options::FormatOptions;
use crate::partition::{
    check_partition_column, partition_dir_name, partition_segment, partition_texts,
    spark_partitioned_file_name,
};
use crate::stats::FileStatistics;

/// Rows written between two checks of the target file size.
const ROLLOVER_ROWS: usize = 1024;

/// Spark's data file name: `part-<task:05>-<job uuid>-c<file:03><suffix>`,
/// e.g. `part-00000-<uuid>-c000.snappy.parquet`.
#[must_use]
pub fn spark_file_name(task: u32, job_uuid: &str, file: u32, suffix: &str) -> String {
    format!("part-{task:05}-{job_uuid}-c{file:03}{suffix}")
}

/// The job-id segment of a task's file names: `base` for attempt 0 (names
/// exactly as before), `base_<attempt>` after that, so a retried attempt
/// can never overwrite a committed attempt's file.
#[must_use]
pub fn effective_job_uuid(base: &str, attempt: u32) -> String {
    if attempt == 0 {
        base.to_owned()
    } else {
        format!("{base}_{attempt}")
    }
}

/// Settings of one task's writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteOptions {
    /// Store to write to.
    pub store: StoreId,
    /// Directory (staging prefix) of the files.
    pub dir: Path,
    /// Task (partition) number in file names.
    pub task: u32,
    /// Job UUID in file names.
    pub job_uuid: String,
    /// `maxRecordsPerFile` (`spark.sql.files.maxRecordsPerFile`); 0 = unlimited.
    pub max_records_per_file: u64,
    /// Roll over to a new file once this many bytes were produced; 0 = never.
    pub target_file_bytes: u64,
    /// Partition column indices into the input (in `partitionBy` order);
    /// empty for unpartitioned output.
    pub partition_by: Vec<usize>,
    /// Partition column names (one per [`Self::partition_by`]).
    pub partition_names: Vec<String>,
    /// The session time zone as a fixed offset in seconds, which renders
    /// timestamp partition values; `None` when it is not a fixed offset.
    pub zone: Option<i32>,
}

/// One file written.
#[derive(Debug, Clone, PartialEq)]
pub struct WrittenFileInfo {
    /// Its path.
    pub path: Path,
    /// Its size.
    pub size: u64,
    /// Rows in it.
    pub rows: u64,
    /// Its statistics (for manifests and the optimizer).
    pub statistics: FileStatistics,
}

struct OpenFile {
    encoder: Box<dyn Encoder>,
    upload: Upload,
    path: Path,
    rows: u64,
}

/// Writes one task's output files (see the module docs).
pub struct FileWriter {
    io: IoContext,
    format: Arc<dyn FileFormat>,
    /// The schema of the data files: the input schema without the
    /// partition columns (which live in the directory names instead).
    data_schema: SchemaRef,
    /// Input columns kept in the data files (every column but the
    /// partition ones, in input order).
    data_columns: Vec<usize>,
    format_options: FormatOptions,
    options: WriteOptions,
    suffix: String,
    current: Option<OpenFile>,
    /// The partition key of the open file, and its directory relative to
    /// [`WriteOptions::dir`] (`None` for unpartitioned output).
    current_key: Option<Vec<String>>,
    current_rel: Option<String>,
    /// The next file number (`c000`, `c001`, ...), shared by every writer
    /// of one task.
    next_file: Arc<AtomicU32>,
    written: Vec<WrittenFileInfo>,
    mem: Reservation,
}

impl std::fmt::Debug for FileWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileWriter")
            .field("dir", &self.options.dir)
            .field("files", &self.written.len())
            .finish_non_exhaustive()
    }
}

impl FileWriter {
    /// A writer of `schema` in `format`, holding encoder memory in `mem`.
    /// Partition columns ([`WriteOptions::partition_by`]) are rendered
    /// into directory names and dropped from the data files.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` if the format cannot write, or a
    /// partition column cannot be rendered as a string; the format's
    /// [`FileFormat::check_write`] error; `INTERNAL_ERROR` for a partition
    /// column out of range, a repeated one, or a name missing for it.
    pub fn new(
        io: &IoContext,
        format: Arc<dyn FileFormat>,
        schema: &SchemaRef,
        format_options: FormatOptions,
        options: WriteOptions,
        mem: Reservation,
    ) -> Result<Self> {
        if !format.capabilities(&format_options).write {
            return Err(unsupported(format!("writing the {} format", format.name())));
        }
        let fields = schema.fields();
        if options.partition_names.len() != options.partition_by.len() {
            return Err(internal(
                "partition names do not match the partition columns".to_owned(),
            ));
        }
        let mut partitioned = vec![false; fields.len()];
        for (i, &c) in options.partition_by.iter().enumerate() {
            let field = fields
                .get(c)
                .ok_or_else(|| internal(format!("partition column {c} is out of range")))?;
            if partitioned[c] {
                return Err(internal(format!("partition column {c} is listed twice")));
            }
            partitioned[c] = true;
            check_partition_column(&options.partition_names[i], field.data_type())?;
        }
        let data_columns: Vec<usize> = (0..fields.len()).filter(|i| !partitioned[*i]).collect();
        let data_schema: SchemaRef = Arc::new(
            schema
                .project(&data_columns)
                .map_err(|e| internal(format!("cannot drop the partition columns: {e}")))?,
        );
        format
            .check_write(&data_schema, &format_options)
            .map_err(|e| e.into_error(&format!("{}/{}", options.store, options.dir)))?;
        let suffix = format.file_suffix(&format_options);
        Ok(Self {
            io: io.clone(),
            format,
            data_schema,
            data_columns,
            format_options,
            options,
            suffix,
            current: None,
            current_key: None,
            current_rel: None,
            next_file: Arc::new(AtomicU32::new(0)),
            written: Vec::new(),
            mem,
        })
    }

    /// This writer numbering its files from `counter`, shared with the other
    /// writers of the same task: several writers (one per executor driver)
    /// then write one task's files under distinct names
    /// (`part-<task>-<uuid>-c<n>`, `n` unique within the task).
    #[must_use]
    pub fn sharing_file_numbers(mut self, counter: Arc<AtomicU32>) -> Self {
        self.next_file = counter;
        self
    }

    async fn open_file(&mut self) -> Result<()> {
        let file = self.next_file.fetch_add(1, Ordering::Relaxed);
        let name = if self.options.partition_by.is_empty() {
            spark_file_name(
                self.options.task,
                &self.options.job_uuid,
                file,
                &self.suffix,
            )
        } else {
            spark_partitioned_file_name(
                self.options.task,
                &self.options.job_uuid,
                file,
                &self.suffix,
            )
        };
        // Partition directories carry Spark's `%`-escapes literally: they
        // are parsed as path segments, never joined (which would encode
        // the `%` again).
        let path = match &self.current_rel {
            None => self.options.dir.clone().join(name),
            Some(rel) => {
                let base = self.options.dir.as_ref();
                let full = if base.is_empty() {
                    format!("{rel}/{name}")
                } else {
                    format!("{base}/{rel}/{name}")
                };
                Path::parse(full.as_str())
                    .map_err(|e| internal(format!("invalid partition path {full:?}: {e}")))?
            }
        };
        let display = format!("{}/{}", self.options.store, path);
        let seed = format!("{}/{}/{file}", self.options.job_uuid, self.options.task);
        let encoder = self
            .format
            .writer(
                &self.io,
                Arc::clone(&self.data_schema),
                &self.format_options.clone().with_write_seed(seed),
            )
            .await
            .map_err(|e| e.into_error(&display))?;
        let upload = self.io.upload(&self.options.store, path.clone())?;
        self.current = Some(OpenFile {
            encoder,
            upload,
            path,
            rows: 0,
        });
        Ok(())
    }

    /// Writes `batch`, rolling over to new files as needed.
    ///
    /// # Errors
    /// Encode, upload and memory errors; the writer should then be aborted.
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.options.partition_by.is_empty() {
            return self.write_to_current(batch).await;
        }
        let texts = self.partition_texts(batch)?;
        let mut start = 0;
        while start < batch.num_rows() {
            // Runs are found on the raw texts; only a run's first row is escaped.
            let mut end = start + 1;
            while end < batch.num_rows() && texts.iter().all(|t| t[end] == t[start]) {
                end += 1;
            }
            let key: Vec<String> = texts
                .iter()
                .map(|t| partition_segment(t[start].as_deref()))
                .collect();
            self.write_partition_run(&batch.slice(start, end - start), &key)
                .await?;
            start = end;
        }
        Ok(())
    }

    /// The text of every row of `batch` for each partition column.
    fn partition_texts(&self, batch: &RecordBatch) -> Result<Vec<Vec<Option<String>>>> {
        self.options
            .partition_by
            .iter()
            .map(|&c| {
                let column = batch
                    .columns()
                    .get(c)
                    .ok_or_else(|| internal(format!("partition column {c} is out of range")))?;
                partition_texts(column, self.options.zone)
            })
            .collect()
    }

    /// Writes one partition's run: a new file when the key changed, then
    /// the run's data columns (without the partition columns).
    async fn write_partition_run(&mut self, slice: &RecordBatch, key: &[String]) -> Result<()> {
        if self.current_key.as_deref() != Some(key) {
            self.close_file().await?;
            let rel = partition_dir_name(&self.options.partition_names, key);
            self.current_rel = Some(rel);
            self.open_file().await?;
            self.current_key = Some(key.to_vec());
        }
        let data = self.data_batch(slice)?;
        self.write_to_current(&data).await
    }

    /// `batch` without the partition columns.
    fn data_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = self
            .data_columns
            .iter()
            .map(|&c| {
                batch
                    .columns()
                    .get(c)
                    .cloned()
                    .ok_or_else(|| internal(format!("column {c} is out of range")))
            })
            .collect::<Result<_>>()?;
        RecordBatch::try_new(Arc::clone(&self.data_schema), columns)
            .map_err(|e| internal(format!("cannot drop the partition columns: {e}")))
    }

    /// Writes `batch` to the open file, opening and rolling over files
    /// in the current partition (or the root) as needed.
    async fn write_to_current(&mut self, batch: &RecordBatch) -> Result<()> {
        let mut offset = 0;
        while offset < batch.num_rows() {
            if self.current.is_none() {
                self.open_file().await?;
            }
            let room = match (self.options.max_records_per_file, &self.current) {
                (0, _) | (_, None) => usize::MAX,
                (max, Some(f)) => usize::try_from(max.saturating_sub(f.rows)).unwrap_or(usize::MAX),
            };
            // With a target size, check it at least every `ROLLOVER_ROWS` rows.
            let step = if self.options.target_file_bytes > 0 {
                ROLLOVER_ROWS
            } else {
                usize::MAX
            };
            let n = room.min(step).min(batch.num_rows() - offset);
            let slice = batch.slice(offset, n);
            offset += n;
            self.write_slice(&slice).await?;
            let full = self.current.as_ref().is_some_and(|f| {
                (self.options.max_records_per_file > 0
                    && f.rows >= self.options.max_records_per_file)
                    || (self.options.target_file_bytes > 0
                        && f.encoder.written_bytes() >= self.options.target_file_bytes)
            });
            if full {
                self.close_file().await?;
            }
        }
        Ok(())
    }

    async fn write_slice(&mut self, slice: &RecordBatch) -> Result<()> {
        let Some(f) = self.current.as_mut() else {
            return Ok(());
        };
        let display = format!("{}/{}", self.options.store, f.path);
        f.encoder.write(slice).map_err(|e| e.into_error(&display))?;
        f.rows += slice.num_rows() as u64;
        drain(f).await?;
        let held = held_bytes(f);
        if held > self.mem.size() {
            let grow = held - self.mem.size();
            if self.mem.try_grow(grow).is_err() {
                // Under pressure: encode what is buffered and ship it, let
                // the parts in flight complete (freeing their bytes), then
                // wait for what is still needed.
                f.encoder.flush().map_err(|e| e.into_error(&display))?;
                drain(f).await?;
                f.upload.wait_idle().await?;
                let held = held_bytes(f);
                let grow = held.saturating_sub(self.mem.size());
                if grow > 0 {
                    self.mem.grow(grow).await?;
                }
            }
        } else {
            let extra = self.mem.size() - held;
            self.mem.shrink(extra);
        }
        Ok(())
    }

    async fn close_file(&mut self) -> Result<()> {
        let Some(mut f) = self.current.take() else {
            return Ok(());
        };
        let display = format!("{}/{}", self.options.store, f.path);
        drain(&mut f).await?;
        let OpenFile {
            encoder,
            mut upload,
            path,
            rows,
        } = f;
        let done = encoder.finish().map_err(|e| e.into_error(&display))?;
        upload.write(done.tail).await?;
        let size = upload.finish().await?;
        let held = self.mem.size();
        self.mem.shrink(held);
        self.written.push(WrittenFileInfo {
            path,
            size,
            rows,
            statistics: done.statistics,
        });
        Ok(())
    }

    /// Ends the last file; returns every file written.
    ///
    /// # Errors
    /// Encode and upload errors; the files written so far are then deleted.
    pub async fn finish(mut self) -> Result<Vec<WrittenFileInfo>> {
        if let Err(e) = self.close_file().await {
            // The caller no longer holds the writer: clean up here, so no
            // file of a failed write stays visible.
            self.abort().await;
            return Err(e);
        }
        Ok(std::mem::take(&mut self.written))
    }

    /// Abandons the write: the open upload is aborted and the files already
    /// finished by this writer are deleted, so nothing stays visible.
    pub async fn abort(mut self) {
        if let Some(f) = self.current.take() {
            f.upload.abort().await;
        }
        for w in std::mem::take(&mut self.written) {
            if let Err(e) = self.io.delete(&self.options.store, &w.path).await {
                tracing::warn!(error = %e, "could not delete a staged file");
            }
        }
    }
}

/// Bytes an open file holds in memory: the encoder's buffers, and the
/// upload's buffered and in-flight parts.
fn held_bytes(f: &OpenFile) -> usize {
    f.encoder
        .buffered_bytes()
        .saturating_add(usize::try_from(f.upload.held_bytes()).unwrap_or(usize::MAX))
}

async fn drain(f: &mut OpenFile) -> Result<()> {
    while let Some(bytes) = f.encoder.take_bytes() {
        f.upload.write(bytes).await?;
    }
    Ok(())
}
