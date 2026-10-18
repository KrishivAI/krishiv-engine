// SPDX-License-Identifier: Apache-2.0
//! The file write: [`FileWriteSink`] writes a pipeline's rows as files of
//! one [`FileFormat`] and, as the source side of the breaker, returns one
//! row per file written ([`FileWrite`] wires both).
//!
//! - **One writer task per driver.** `sink` stays sync and bounded: it
//!   hands the batch to a [`FileWriter`] running on its own task through a
//!   bounded channel, and reports [`SinkStatus::Blocked`] while the channel
//!   is full (the encoder or the upload window is behind). Each driver
//!   writes its own files (Spark's `part-<task>` names, one task number per
//!   driver), with `maxRecordsPerFile` and target-size rollover.
//! - **Nothing of a failed or cancelled write stays visible.** `combine`
//!   tells the driver's writer to finish its last file; `finalize` waits
//!   for every writer and then tells them to keep their files, or, if any
//!   failed, to delete them. A writer whose driver ended without `combine`
//!   (cancellation, another task's error) aborts its open upload and
//!   deletes its finished files; one whose `finalize` never came deletes
//!   them too. The query ends only after every writer task has (they count
//!   as the query's blocking work), so its memory pool is empty by then.
//! - Files land under the write's staging directory; publishing them is
//!   the commit protocol's job, never a rename.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_io::{FileFormat, FileWriter, FormatOptions, IoContext, WriteOptions, WrittenFileInfo};
use krishiv_types::batch::Batch;
use tokio::sync::{mpsc, oneshot};

use super::{global_ref, local_mut};
use crate::context::{InFlight, TaskContext};
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::pipeline::{ExecNode, PipelineBuilder};

/// Batches queued per driver before its sink blocks.
const QUEUED_BATCHES: usize = 2;

type Outcome = Result<Vec<WrittenFileInfo>>;

/// Writes its input as files and returns them (see the module docs).
pub struct FileWriteSink {
    format: Arc<dyn FileFormat>,
    io: IoContext,
    schema: SchemaRef,
    format_options: FormatOptions,
    options: WriteOptions,
    output: SchemaRef,
    /// The files of the finished write, with their statistics.
    files: Mutex<Vec<WrittenFileInfo>>,
}

impl std::fmt::Debug for FileWriteSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileWriteSink")
            .field("format", &self.format.name())
            .field("dir", &self.options.dir)
            .finish_non_exhaustive()
    }
}

/// One driver's writer task, as `finalize` sees it.
struct Writer {
    done: oneshot::Receiver<Outcome>,
    keep: oneshot::Sender<bool>,
}

struct WriteGlobal {
    next_task: AtomicU32,
    writers: Mutex<Vec<Writer>>,
    written: Mutex<Vec<WrittenFileInfo>>,
}

struct WriteLocal {
    batches: mpsc::Sender<RecordBatch>,
    commit: Option<oneshot::Sender<()>>,
    writer: Option<Writer>,
    /// The writer's error, once it failed (it then closes the channel).
    failed: Arc<Mutex<Option<KrishivError>>>,
}

/// The channels of one writer task.
struct WriterTask {
    writer: FileWriter,
    batches: mpsc::Receiver<RecordBatch>,
    commit: oneshot::Receiver<()>,
    done: oneshot::Sender<Outcome>,
    keep: oneshot::Receiver<bool>,
    failed: Arc<Mutex<Option<KrishivError>>>,
    io: IoContext,
    store: krishiv_io::StoreId,
    /// The query does not end before this task has.
    _in_flight: InFlight,
}

impl WriterTask {
    async fn run(self) {
        let Self {
            mut writer,
            mut batches,
            commit,
            done,
            keep,
            failed,
            io,
            store,
            _in_flight,
        } = self;
        while let Some(b) = batches.recv().await {
            if let Err(e) = writer.write(&b).await {
                *failed.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
                drop(batches);
                writer.abort().await;
                drop(done.send(Err(e)));
                return;
            }
        }
        if commit.await.is_err() {
            // The driver ended without `combine`: cancelled or failed.
            writer.abort().await;
            return;
        }
        let files = match writer.finish().await {
            Ok(files) => files,
            Err(e) => {
                drop(done.send(Err(e)));
                return;
            }
        };
        let kept = done.send(Ok(files.clone())).is_ok() && keep.await.unwrap_or(false);
        if !kept {
            for f in files {
                if let Err(e) = io.delete(&store, &f.path).await {
                    tracing::warn!(error = %e, "could not delete a file of a failed write");
                }
            }
        }
    }
}

impl FileWriteSink {
    /// Writes batches of `schema` in `format` with `options` (the task
    /// number there is the first driver's; each driver takes the next).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` if the format cannot write.
    pub fn try_new(
        format: Arc<dyn FileFormat>,
        io: IoContext,
        schema: SchemaRef,
        format_options: FormatOptions,
        options: WriteOptions,
    ) -> Result<Self> {
        if !format.capabilities(&format_options).write {
            return Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: format!("writing the {} format", format.name()),
            }
            .into());
        }
        let output = Arc::new(Schema::new(vec![
            Field::new("path", krishiv_types::DataType::String.to_arrow(), false),
            Field::new("size", DataType::Int64, false),
            Field::new("rows", DataType::Int64, false),
        ]));
        Ok(Self {
            format,
            io,
            schema,
            format_options,
            options,
            output,
            files: Mutex::new(Vec::new()),
        })
    }

    /// The files written, with their statistics (for manifests and the
    /// optimizer), once the write has finished; empty before.
    #[must_use]
    pub fn files(&self) -> Vec<WrittenFileInfo> {
        self.files
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn written_batch(&self, files: &[WrittenFileInfo]) -> Result<RecordBatch> {
        let long = |v: u64| i64::try_from(v).map_err(|_| internal(format!("{v} overflows")));
        let paths: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            files.iter().map(|f| f.path.as_ref()),
        ));
        let sizes: ArrayRef = Arc::new(Int64Array::from(
            files
                .iter()
                .map(|f| long(f.size))
                .collect::<Result<Vec<_>>>()?,
        ));
        let rows: ArrayRef = Arc::new(Int64Array::from(
            files
                .iter()
                .map(|f| long(f.rows))
                .collect::<Result<Vec<_>>>()?,
        ));
        RecordBatch::try_new(Arc::clone(&self.output), vec![paths, sizes, rows])
            .map_err(|e| internal(format!("invalid written-files batch: {e}")))
    }
}

impl Sink for FileWriteSink {
    fn name(&self) -> &'static str {
        "FileWrite"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(WriteGlobal {
            next_task: AtomicU32::new(self.options.task),
            writers: Mutex::new(Vec::new()),
            written: Mutex::new(Vec::new()),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let g = global_ref::<WriteGlobal>(global, "FileWrite")?;
        let options = WriteOptions {
            task: g.next_task.fetch_add(1, Ordering::Relaxed),
            ..self.options.clone()
        };
        let writer = FileWriter::new(
            &self.io,
            Arc::clone(&self.format),
            Arc::clone(&self.schema),
            self.format_options.clone(),
            options,
            ctx.consumer().reservation(),
        )?;
        let (batches_tx, batches_rx) = mpsc::channel(QUEUED_BATCHES);
        let (commit_tx, commit_rx) = oneshot::channel();
        let (done_tx, done_rx) = oneshot::channel();
        let (keep_tx, keep_rx) = oneshot::channel();
        let failed = Arc::new(Mutex::new(None));
        let task = WriterTask {
            writer,
            batches: batches_rx,
            commit: commit_rx,
            done: done_tx,
            keep: keep_rx,
            failed: Arc::clone(&failed),
            io: self.io.clone(),
            store: self.options.store.clone(),
            _in_flight: ctx.query().blocking_started(),
        };
        // Detached: it ends on its own once its channels close.
        drop(
            ctx.executor()
                .spawn(ctx.query().class(), Box::pin(task.run())),
        );
        Ok(Box::new(WriteLocal {
            batches: batches_tx,
            commit: Some(commit_tx),
            writer: Some(Writer {
                done: done_rx,
                keep: keep_tx,
            }),
            failed,
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let local = local_mut::<WriteLocal>(local, "FileWrite")?;
        if !input.is_unweighted() {
            return Err(internal(
                "a file write takes insert-only rows (weight 1)".to_owned(),
            ));
        }
        let (data, _) = input.into_parts();
        match local.batches.try_send(data) {
            Ok(()) if local.batches.capacity() == 0 => {
                let tx = local.batches.clone();
                Ok(SinkStatus::Blocked(Box::pin(async move {
                    drop(tx.reserve().await);
                })))
            }
            Ok(()) => Ok(SinkStatus::NeedMoreInput),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(local
                .failed
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
                .unwrap_or_else(|| internal("the file writer ended early".to_owned()))),
            Err(mpsc::error::TrySendError::Full(_)) => Err(internal(
                "the file writer's queue was full after a wait".to_owned(),
            )),
        }
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let mut local = local
            .downcast::<WriteLocal>()
            .map_err(|_| internal("wrong local state for FileWrite".to_owned()))?;
        let g = global_ref::<WriteGlobal>(global, "FileWrite")?;
        // Finish the last file once the queued batches are written.
        if let Some(commit) = local.commit.take() {
            let _ = commit.send(());
        }
        if let Some(w) = local.writer.take() {
            g.writers
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(w);
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<WriteGlobal>(global, "FileWrite")?;
            let writers =
                std::mem::take(&mut *g.writers.lock().unwrap_or_else(PoisonError::into_inner));
            let mut keeps = Vec::with_capacity(writers.len());
            let mut files = Vec::new();
            let mut first_error = None;
            for w in writers {
                keeps.push(w.keep);
                match w.done.await {
                    Ok(Ok(f)) => files.extend(f),
                    Ok(Err(e)) => {
                        first_error.get_or_insert(e);
                    }
                    Err(_) => {
                        first_error
                            .get_or_insert_with(|| internal("a file writer task died".to_owned()));
                    }
                }
            }
            let keep = first_error.is_none();
            for k in keeps {
                let _ = k.send(keep);
            }
            if let Some(e) = first_error {
                return Err(e);
            }
            self.files
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone_from(&files);
            *g.written.lock().unwrap_or_else(PoisonError::into_inner) = files;
            Ok(())
        })
    }
}

/// The source side: one batch listing the files written.
struct Emitted(bool);

impl Source for FileWriteSink {
    fn name(&self) -> &'static str {
        "FileWriteResult"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let dep = ctx
            .dependency()
            .ok_or_else(|| internal("file write result without its sink".to_owned()))?;
        let g = global_ref::<WriteGlobal>(dep, "FileWrite")?;
        let files = std::mem::take(&mut *g.written.lock().unwrap_or_else(PoisonError::into_inner));
        Ok(Arc::new(self.written_batch(&files)?))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(Emitted(false)))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let out = (|| {
            let emitted = local_mut::<Emitted>(local, "FileWriteResult")?;
            if emitted.0 {
                return Ok(None);
            }
            emitted.0 = true;
            let batch = global_ref::<RecordBatch>(global, "FileWriteResult")?;
            Ok(Some(Batch::new(batch.clone())))
        })();
        Box::pin(std::future::ready(out))
    }
}

/// A plan node writing `input` through `sink` and returning the files.
pub struct FileWrite {
    /// The rows to write.
    pub input: Arc<dyn ExecNode>,
    /// The sink (and result source).
    pub sink: Arc<FileWriteSink>,
}

impl std::fmt::Debug for FileWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileWrite")
            .field("sink", &self.sink)
            .finish_non_exhaustive()
    }
}

impl ExecNode for FileWrite {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(
            Arc::clone(&self.sink) as Arc<dyn Sink>,
            Arc::clone(&self.input),
        )?;
        b.set_breaker_source(Arc::clone(&self.sink) as Arc<dyn Source>, sink)?;
        Ok(())
    }
}
