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

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_memory::Reservation;
use object_store::path::Path;

use crate::error::{Result, unsupported};
use crate::format::{Encoder, FileFormat};
use crate::io::{IoContext, StoreId, Upload};
use crate::options::FormatOptions;
use crate::stats::FileStatistics;

/// Rows written between two checks of the target file size.
const ROLLOVER_ROWS: usize = 1024;

/// Spark's data file name: `part-<task:05>-<job uuid>-c<file:03><suffix>`,
/// e.g. `part-00000-<uuid>-c000.snappy.parquet`.
#[must_use]
pub fn spark_file_name(task: u32, job_uuid: &str, file: u32, suffix: &str) -> String {
    format!("part-{task:05}-{job_uuid}-c{file:03}{suffix}")
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
    schema: SchemaRef,
    format_options: FormatOptions,
    options: WriteOptions,
    suffix: String,
    current: Option<OpenFile>,
    next_file: u32,
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
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` if the format cannot write.
    pub fn new(
        io: &IoContext,
        format: Arc<dyn FileFormat>,
        schema: SchemaRef,
        format_options: FormatOptions,
        options: WriteOptions,
        mem: Reservation,
    ) -> Result<Self> {
        if !format.capabilities(&format_options).write {
            return Err(unsupported(format!("writing the {} format", format.name())));
        }
        let suffix = format.file_suffix(&format_options);
        Ok(Self {
            io: io.clone(),
            format,
            schema,
            format_options,
            options,
            suffix,
            current: None,
            next_file: 0,
            written: Vec::new(),
            mem,
        })
    }

    async fn open_file(&mut self) -> Result<()> {
        let name = spark_file_name(
            self.options.task,
            &self.options.job_uuid,
            self.next_file,
            &self.suffix,
        );
        self.next_file += 1;
        let path = self.options.dir.clone().join(name);
        let display = format!("{}/{}", self.options.store, path);
        let encoder = self
            .format
            .writer(&self.io, Arc::clone(&self.schema), &self.format_options)
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
