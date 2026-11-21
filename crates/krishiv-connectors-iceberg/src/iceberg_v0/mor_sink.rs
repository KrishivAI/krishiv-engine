// SPDX-License-Identifier: Apache-2.0
//! [`MergeOnReadSink`]: the [`Sink`] of a merge-on-read `DELETE` or `UPDATE` (P3-T027, PR-36c
//! part 2), the adapter between a statement's rows and [`MergeOnReadWrite`] (`row_delta.rs`).
//!
//! The rows it receives end with the row id columns `_krishiv_row_file` and
//! `_krishiv_row_index` (what [`IcebergRowLevel::rewriter`](super::row_level) recognizes it by).
//! A `DELETE` receives only those two: each row deletes `(file, position)`. An `UPDATE` receives
//! the table's columns first: each row deletes `(file, position)` and inserts the new row.
//!
//! One [`MorTask`](super::row_delta::MorTask) per driver, on its own task as
//! [`IcebergWriteSink`](super::sink::IcebergWriteSink): `sink` is sync and bounded and hands each
//! batch over a bounded channel. A driver that ends without `combine` stops its task, which
//! aborts its output. `finalize` collects every task's output and commits them as one `RowDelta`
//! ([`MergeOnReadWrite::commit`]); a task whose output `finalize` does not keep deletes it. A
//! statement that deleted and inserted nothing still commits an `overwrite` snapshot (Java).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::{GlobalState, LocalState, Sink, SinkStatus, TaskContext};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN, planner::RowLevelOperation};
use krishiv_memory::ConsumerHandle;
use krishiv_types::batch::Batch;
use tokio::sync::{mpsc, oneshot};

use super::row_delta::{MergeOnReadWrite, TaskOutput};

/// Batches queued per driver before its sink blocks.
const QUEUED_BATCHES: usize = 2;

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether `schema` is the input of a merge-on-read statement: its last two columns are the row
/// id.
#[must_use]
pub(crate) fn is_merge_on_read_input(schema: &arrow_schema::Schema) -> bool {
    let n = schema.fields().len();
    n >= 2
        && schema.field(n - 2).name() == ROW_FILE_COLUMN
        && schema.field(n - 1).name() == ROW_POSITION_COLUMN
}

/// The sink (module docs).
#[derive(Debug)]
pub(crate) struct MergeOnReadSink {
    write: Arc<MergeOnReadWrite>,
    op: RowLevelOperation,
}

impl MergeOnReadSink {
    /// The sink of `write` for `op`.
    pub(crate) fn new(write: Arc<MergeOnReadWrite>, op: RowLevelOperation) -> Self {
        Self { write, op }
    }
}

type Outcome = Result<TaskOutput, KrishivError>;

struct Writer {
    done: oneshot::Receiver<Outcome>,
    /// Tells the task whether its output is kept; a task told otherwise (or whose finalize never
    /// comes) deletes its files.
    keep: oneshot::Sender<bool>,
}

struct SinkLocal {
    batches: mpsc::Sender<RecordBatch>,
    commit: Option<oneshot::Sender<()>>,
    writer: Option<Writer>,
    failed: Arc<Mutex<Option<KrishivError>>>,
    stop: Arc<AtomicBool>,
}

impl Drop for SinkLocal {
    fn drop(&mut self) {
        if self.commit.is_some() {
            self.stop.store(true, Ordering::Release);
        }
    }
}

struct SinkGlobal {
    writers: Mutex<Vec<Writer>>,
}

/// Applies one batch to the task: its deletes, and for an `UPDATE` its new rows.
async fn apply(
    task: &mut super::row_delta::MorTask,
    batch: &RecordBatch,
    update: bool,
    write: &MergeOnReadWrite,
) -> Result<(), KrishivError> {
    let n = batch.num_columns();
    if n < 2 {
        return Err(internal("a merge-on-read batch without the row id columns"));
    }
    let files = batch.column(n - 2);
    let positions = batch.column(n - 1);
    let positions = positions
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| internal("a merge-on-read row position that is not a bigint"))?;
    for row in 0..batch.num_rows() {
        if files.is_null(row) || positions.is_null(row) {
            return Err(internal("a merge-on-read row without a row id"));
        }
        let file: &[u8] = if let Some(a) = files.as_binary_view_opt() {
            a.value(row)
        } else if let Some(a) = files.as_binary_opt::<i32>() {
            a.value(row)
        } else if let Some(a) = files.as_binary_opt::<i64>() {
            a.value(row)
        } else {
            return Err(internal("a merge-on-read row file that is not binary"));
        };
        task.delete(file, positions.value(row))?;
    }
    if update && batch.num_rows() > 0 {
        let rows = batch
            .project(&(0..n - 2).collect::<Vec<_>>())
            .map_err(|e| internal(&e.to_string()))?;
        write.check_nulls(&rows)?;
        task.insert(&rows).await?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "one task's channels and shared state, built once in local_state"
)]
async fn run_task(
    mut task: super::row_delta::MorTask,
    update: bool,
    mut batches: mpsc::Receiver<RecordBatch>,
    commit: oneshot::Receiver<()>,
    done: oneshot::Sender<Outcome>,
    keep: oneshot::Receiver<bool>,
    write: Arc<MergeOnReadWrite>,
    failed: Arc<Mutex<Option<KrishivError>>>,
    stop: Arc<AtomicBool>,
) {
    while let Some(b) = batches.recv().await {
        if stop.load(Ordering::Acquire) {
            task.abort().await;
            return;
        }
        if let Err(e) = apply(&mut task, &b, update, &write).await {
            *lock(&failed) = Some(e.clone());
            drop(batches);
            task.abort().await;
            drop(done.send(Err(e)));
            return;
        }
    }
    if commit.await.is_err() {
        task.abort().await;
        return;
    }
    match task.finish().await {
        Err(e) => drop(done.send(Err(e))),
        Ok(out) => {
            // Armed until `finalize` keeps the output: a finalize that never comes (another
            // driver failed, the query was cancelled) or says no deletes the files here.
            let mut guard = write.written_by(std::slice::from_ref(&out));
            if done.send(Ok(out)).is_ok() && keep.await.unwrap_or(false) {
                guard.armed = false;
            } else {
                guard.delete().await;
            }
        }
    }
}

impl Sink for MergeOnReadSink {
    fn name(&self) -> &'static str {
        "IcebergMergeOnRead"
    }

    fn global_state(&self, _ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(SinkGlobal {
            writers: Mutex::new(Vec::new()),
        }))
    }

    fn local_state(
        &self,
        ctx: &TaskContext,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        let mem = ConsumerHandle::register(Arc::clone(ctx.query().pool()), "iceberg-mor", None);
        let task = self.write.task(ctx.partition(), mem);
        let (batches_tx, batches_rx) = mpsc::channel(QUEUED_BATCHES);
        let (commit_tx, commit_rx) = oneshot::channel();
        let (done_tx, done_rx) = oneshot::channel();
        let (keep_tx, keep_rx) = oneshot::channel();
        let failed = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let fut = run_task(
            task,
            self.op != RowLevelOperation::Delete,
            batches_rx,
            commit_rx,
            done_tx,
            keep_rx,
            Arc::clone(&self.write),
            Arc::clone(&failed),
            Arc::clone(&stop),
        );
        drop(krishiv_exec::console::spawn_named(
            ctx.executor(),
            ctx.query(),
            ctx.query().class(),
            &krishiv_exec::console::io_task_name("iceberg-mor"),
            Box::pin(fut),
        ));
        Ok(Box::new(SinkLocal {
            batches: batches_tx,
            commit: Some(commit_tx),
            writer: Some(Writer {
                done: done_rx,
                keep: keep_tx,
            }),
            failed,
            stop,
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<SinkStatus> {
        let local = local
            .downcast_mut::<SinkLocal>()
            .ok_or_else(|| internal("wrong local state for IcebergMergeOnRead"))?;
        if !input.is_unweighted() {
            return Err(internal("a merge-on-read change takes rows of weight 1"));
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
            Err(mpsc::error::TrySendError::Closed(_)) => Err(lock(&local.failed)
                .clone()
                .unwrap_or_else(|| internal("the merge-on-read task ended early"))),
            Err(mpsc::error::TrySendError::Full(_)) => {
                Err(internal("the merge-on-read queue was full after a wait"))
            }
        }
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> krishiv_exec::Result<()> {
        let mut local = local
            .downcast::<SinkLocal>()
            .map_err(|_| internal("wrong local state for IcebergMergeOnRead"))?;
        let global = global
            .downcast_ref::<SinkGlobal>()
            .ok_or_else(|| internal("wrong global state for IcebergMergeOnRead"))?;
        if let Some(commit) = local.commit.take() {
            let _ = commit.send(());
        }
        if let Some(w) = local.writer.take() {
            lock(&global.writers).push(w);
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, krishiv_exec::Result<()>> {
        Box::pin(async move {
            let global = global
                .downcast_ref::<SinkGlobal>()
                .ok_or_else(|| internal("wrong global state for IcebergMergeOnRead"))?;
            let writers = std::mem::take(&mut *lock(&global.writers));
            let mut outputs: Vec<TaskOutput> = Vec::new();
            let mut first_error = None;
            let mut keeps = Vec::new();
            for w in writers {
                keeps.push(w.keep);
                match w.done.await {
                    Ok(Ok(o)) => outputs.push(o),
                    Ok(Err(e)) => {
                        first_error.get_or_insert(e);
                    }
                    Err(_) => {
                        first_error.get_or_insert_with(|| internal("a merge-on-read task died"));
                    }
                }
            }
            // Each task deletes its own files unless told to keep them (dropping `keeps` says no).
            if let Some(e) = first_error {
                return Err(e);
            }
            ctx.query().check_cancelled()?;
            // An empty statement still commits (an `overwrite` snapshot, as Java's RowDelta).
            for k in keeps {
                let _ = k.send(true);
            }
            let query_id = ctx.query().id().0.to_string();
            self.write.commit(outputs, Some(&query_id)).await?;
            Ok(())
        })
    }
}
