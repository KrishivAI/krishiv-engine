// SPDX-License-Identifier: Apache-2.0
//! [`IcebergWriteSink`]: an append into an Iceberg table, the [`Sink`] that
//! [`IcebergTableProvider::writer`](krishiv_exec::planner::TableProvider::writer)
//! returns (P3-T027, PR-36a; design note §6.1 distribution `none`, §7.1, §7.4).
//!
//! - **One [`DataFileWriter`] per driver, on its own task** (as
//!   [`FileWriteSink`](krishiv_exec::FileWriteSink)): `sink` stays sync and
//!   bounded, hands the batch over a bounded channel and reports
//!   [`SinkStatus::Blocked`] while the channel is full. The task is spawned
//!   on the query's compute runtime ([`krishiv_exec::console::spawn_named`]).
//!   **Gap:** `FileWriteSink`'s tasks count as the query's blocking work, so
//!   the query ends only after them; that counter
//!   (`QueryContext::blocking_started`) is private to `krishiv-exec`, so a
//!   cancelled or failed write's task may still be aborting (deleting files,
//!   releasing its memory) for a moment after the query ended.
//! - **Nothing of a failed or cancelled write stays visible, and nothing is
//!   committed.** A batch's write that fails aborts the task's writer
//!   (deleting its files) and fails the next `sink` or `combine`. A driver
//!   that ends without `combine` (cancelled, or another driver failed) stops
//!   its task, which aborts its writer before the next batch. `finalize`
//!   waits for every task to finish its files and then tells each to keep
//!   them or, if any failed, to delete them; a task whose `finalize` never
//!   comes deletes them too.
//! - **`finalize`** then writes the manifests ([`write_new_manifests`]) and
//!   commits ([`commit_append`]) against the table as the provider pinned it
//!   when it was opened (design note §7.1, S0), with the request's `write_id`
//!   as `krishiv.write-id` (a fresh UUID when the request has none) and the
//!   query id as `krishiv.query-id`. Until the commit starts, the data files
//!   and manifests are deleted if anything fails or the finalize is dropped;
//!   once [`commit_append`] runs, cleanup is its job (§7.4: an unresolved
//!   outcome deletes nothing). A finalize dropped while a request is in
//!   flight leaves orphans, never a table that references deleted files. A
//!   write of no rows still commits a snapshot with no files.
//! - **Statistics** (PR-54b, `statistics_file.rs`): when the target has them
//!   on, each driver's writer sketches its rows (charged to a reservation of
//!   the driver's consumer, refused → that task has none), `finalize`
//!   unions the tasks' sketches (charged to its own consumer) into the
//!   write's, and each commit attempt writes them into a statistics file. A
//!   task without sketches leaves the whole write without statistics.
//! - **Input checks.** [`IcebergWriteSink::try_new`] refuses a request whose
//!   `input_schema` does not have the table's columns, by name in order and of
//!   the table's types (nullability and field ids aside), with
//!   `KRISHIV_UNSUPPORTED_FEATURE` before anything is written: Spark's
//!   `INCOMPATIBLE_DATA_FOR_TABLE.*` classes belong to the analyzer's
//!   resolution of the write, which brings the rows to the table's columns
//!   before the sink sees them. A NULL in a column (or a struct field, list
//!   element or map value) the table declares required fails the write with
//!   `NOT_NULL_ASSERT_VIOLATION` (42000): Spark v4.2.0 wraps such a column in
//!   `AssertNotNull(input, colPath)` (`TableOutputResolver.checkNullability`,
//!   sql/catalyst/.../analysis/TableOutputResolver.scala:493-510, when the input
//!   is nullable and `storeAssignmentPolicy` is not `LEGACY`), which throws
//!   `QueryExecutionErrors.notNullAssertViolation` with the path joined by
//!   newlines, a newline before and after (`AssertNotNull.errMsg`,
//!   sql/catalyst/.../expressions/objects/objects.scala:1955-1967); nested
//!   positions are named `element`, `key` and `value` (`resolveArrayType`,
//!   `resolveMapType`).
//! - **File names** carry `<partition id>-<attempt id>-<write id>-<n>` (Java's
//!   `OutputFileFactory`). The partition id is the driver's
//!   ([`TaskContext::partition`]). The attempt id must differ between every
//!   driver, worker and retry of one write ([`TaskIds::attempt_id`]), and
//!   `TaskContext` carries no task attempt: each driver draws one from
//!   [`unique_u64`], a per-process counter put through a bijective 64-bit mix
//!   keyed by a random per-process value. Two drivers of one process never draw
//!   the same id (the mix of distinct counters is distinct); drivers of two
//!   processes (workers, or a query retried after a restart) collide only if
//!   their random keys line up, about 2^-64 per pair.
//!
//! The other commits are a copy-on-write rewrite's and an `INSERT
//! OVERWRITE`'s ([`IcebergWriteSink::try_rewrite`], `row_level.rs`,
//! `overwrite.rs`): the same writers (with the lineage columns on v3), then
//! an `OverwriteFiles` or `ReplacePartitions` through [`commit_validated`]
//! with what it removes and its [`Validation`]; a dynamic overwrite takes
//! its partitions from the files written and commits nothing when it wrote
//! none. A
//! distributed write would need the `WrittenDataFile`s of
//! every worker's tasks gathered to one place that runs the commit once; the
//! `Sink` contract commits once per pipeline run of one process, which is what
//! this sink does.

use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, SchemaRef};
use async_trait::async_trait;
use futures::future::BoxFuture;
use iceberg::io::FileIO;
use iceberg::spec::TableMetadata;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivUnsupportedFeature, NotNullAssertViolation,
};
use krishiv_exec::planner::WriteRequest;
use krishiv_exec::{GlobalState, LocalState, Sink, SinkStatus, TaskContext};
use krishiv_types::batch::Batch;
use tokio::sync::{mpsc, oneshot};

use super::append::{WriteIds, write_new_manifests};
use super::commit::{CommitClock, CommitContext, CommitTable, commit_append, commit_validated};
use super::data_file::{WrittenDataFile, data_file};
use super::filter::{Removals, partition_key};
use super::statistics_file::{SketchColumn, TaskSketches, sketch_columns, union_tasks};
use super::storage::Registered;
use super::types::arrow_schema;
use super::validate::{PartitionSet, Validation};
use super::write::{DataFileWriter, TaskIds};

/// The branch an append commits to.
const BRANCH: &str = "main";

/// What a write into one opened table needs: the committer, the table as the
/// provider pinned it (S0), its I/O and limits.
pub(crate) struct WriteTarget {
    /// The catalog's commit endpoint for the table.
    pub(crate) committer: Arc<dyn CommitTable>,
    /// The table's metadata when the provider was opened (design note §7.1).
    pub(crate) base: TableMetadata,
    /// Data files are written through it.
    pub(crate) registered: Registered,
    /// Manifests and lists are written, and files deleted, through it.
    pub(crate) file_io: FileIO,
    /// `krishiv.catalog.maxMetadataBytes`.
    pub(crate) max_metadata_bytes: u64,
    /// `krishiv.iceberg.write.maxOpenFiles`, as the provider was opened.
    pub(crate) max_open_files: usize,
    /// Whether the write writes statistics (`krishiv.iceberg.write.statistics.enabled`
    /// and the table's `krishiv.write.statistics.enabled`, as the provider was opened).
    pub(crate) statistics: bool,
    /// `krishiv.iceberg.commit.maxIntervening`, as the provider was opened.
    pub(crate) max_intervening: usize,
}

/// A copy-on-write rewrite's commit (`row_level.rs`): an `OverwriteFiles`
/// removing `removals`, validated by `validation` on every rebase.
#[derive(Debug, Clone)]
pub(crate) struct Overwrite {
    /// The files removed.
    pub(crate) removals: Removals,
    /// The rebase checks.
    pub(crate) validation: Validation,
    /// The rows carry `_row_id` and `_last_updated_sequence_number` after
    /// the table's columns (a v3 table), written under their reserved ids.
    pub(crate) lineage: bool,
}

impl std::fmt::Debug for WriteTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteTarget")
            .field("table", &self.committer.name())
            .field("location", &self.base.location())
            .finish_non_exhaustive()
    }
}

/// The append sink (see the module docs).
pub(crate) struct IcebergWriteSink {
    target: Arc<WriteTarget>,
    /// The table's Arrow schema (its required columns non-nullable).
    table: SchemaRef,
    /// Whether any column or nested position of the table is required.
    has_required: bool,
    /// The commit's idempotency key, also the files' operation id.
    write_id: String,
    /// The caller's `krishiv.write.idempotencyKey` (E34).
    idempotency_key: Option<String>,
    clock: Arc<dyn CommitClock>,
    /// The columns sketched, when the write has statistics (PR-54b).
    sketch_columns: Option<Arc<[SketchColumn]>>,
    /// `None`: an append.
    overwrite: Option<Overwrite>,
}

impl std::fmt::Debug for IcebergWriteSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergWriteSink")
            .field("target", &self.target)
            .field("write_id", &self.write_id)
            .finish_non_exhaustive()
    }
}

/// Batches queued per driver before its sink blocks.
const QUEUED_BATCHES: usize = 2;

/// A task's files and, when the write has statistics, its sketches.
type Outcome = Result<(Vec<WrittenDataFile>, Option<TaskSketches>), KrishivError>;

/// One driver's writer task, as `finalize` sees it.
struct Writer {
    done: oneshot::Receiver<Outcome>,
    keep: oneshot::Sender<bool>,
}

struct SinkLocal {
    batches: mpsc::Sender<RecordBatch>,
    commit: Option<oneshot::Sender<()>>,
    writer: Option<Writer>,
    /// The task's error, once it failed (it then closes the channel).
    failed: Arc<Mutex<Option<KrishivError>>>,
    /// Set when the driver ends without `combine`: the task aborts before
    /// its next batch instead of writing the queued ones.
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
    /// The writer tasks of the drivers that combined.
    writers: Mutex<Vec<Writer>>,
}

/// One driver's writer task (see the module docs).
struct WriterTask {
    writer: DataFileWriter,
    batches: mpsc::Receiver<RecordBatch>,
    commit: oneshot::Receiver<()>,
    done: oneshot::Sender<Outcome>,
    keep: oneshot::Receiver<bool>,
    failed: Arc<Mutex<Option<KrishivError>>>,
    stop: Arc<AtomicBool>,
    file_io: FileIO,
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
            stop,
            file_io,
        } = self;
        while let Some(b) = batches.recv().await {
            if stop.load(Ordering::Acquire) {
                writer.abort().await;
                return;
            }
            if let Err(e) = writer.write(&b).await {
                *lock(&failed) = Some(e.clone());
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
        let sketches = writer.take_sketches();
        let files = match writer.finish().await {
            Ok(files) => files,
            Err(e) => {
                drop(done.send(Err(e)));
                return;
            }
        };
        let locations: Vec<String> = files.iter().map(|f| f.location.clone()).collect();
        let kept = done.send(Ok((files, sketches))).is_ok() && keep.await.unwrap_or(false);
        if !kept {
            delete_paths(&file_io, locations).await;
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl IcebergWriteSink {
    /// The sink of `request` into `target`'s table.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when `request.input_schema` does not
    /// have the table's columns and types (module docs); the table schema's
    /// conversion errors; `INTERNAL_ERROR` when the I/O scheduler has no
    /// runtime (the commit's clock sleeps on it).
    pub(crate) fn try_new(
        target: Arc<WriteTarget>,
        request: &WriteRequest,
    ) -> Result<Self, KrishivError> {
        Self::build(
            target,
            &request.input_schema,
            &request.table_name,
            request.write_id.clone(),
            None,
        )
        .map(|s| s.with_idempotency_key(request.idempotency_key()))
    }

    /// Records the caller's idempotency key in the snapshot's summary; a table that already has a
    /// snapshot with it gets no commit (E34).
    #[must_use]
    pub(crate) fn with_idempotency_key(mut self, key: Option<String>) -> Self {
        self.idempotency_key = key;
        self
    }

    /// The sink of a copy-on-write rewrite into `target`'s table: the rows
    /// written replace the removed files in one `OverwriteFiles` commit
    /// (`row_level.rs`). With lineage the input ends with `_row_id` and
    /// `_last_updated_sequence_number`.
    ///
    /// # Errors
    /// As [`Self::try_new`].
    pub(crate) fn try_rewrite(
        target: Arc<WriteTarget>,
        input_schema: &SchemaRef,
        table_name: &str,
        write_id: Option<String>,
        overwrite: Overwrite,
    ) -> Result<Self, KrishivError> {
        Self::build(target, input_schema, table_name, write_id, Some(overwrite))
    }

    /// Whether this write sketches its columns (appends only).
    #[cfg(test)]
    pub(super) fn sketches_columns(&self) -> bool {
        self.sketch_columns.is_some()
    }

    fn build(
        target: Arc<WriteTarget>,
        input_schema: &SchemaRef,
        table_name: &str,
        write_id: Option<String>,
        overwrite: Option<Overwrite>,
    ) -> Result<Self, KrishivError> {
        let table = arrow_schema(target.base.current_schema())?;
        let expected = if overwrite.as_ref().is_some_and(|o| o.lineage) {
            let mut fields: Vec<FieldRef> = table.fields().iter().cloned().collect();
            fields.extend(super::lineage::lineage_fields());
            Arc::new(arrow_schema::Schema::new(fields))
        } else {
            Arc::clone(&table)
        };
        check_input(input_schema, &expected, table_name)?;
        let handle = target
            .registered
            .io
            .scheduler()
            .runtime()
            .cloned()
            .ok_or_else(|| {
                internal("an Iceberg commit needs the I/O scheduler's runtime".to_owned())
            })?;
        let has_required = table.fields().iter().any(|f| has_required(f));
        let write_id = write_id.unwrap_or_else(random_uuid);
        // Statistics are for appends only: a copy-on-write rewrite would union the parent's
        // blob with the rewritten rows and keep counting deleted keys.
        let sketch_columns = (target.statistics && overwrite.is_none())
            .then(|| sketch_columns(target.base.current_schema()).into());
        Ok(Self {
            target,
            table,
            has_required,
            write_id,
            idempotency_key: None,
            clock: Arc::new(IoClock { handle }),
            sketch_columns,
            overwrite,
        })
    }

    /// Fails on a NULL in a required position of `batch` (module docs).
    fn check_nulls(&self, batch: &RecordBatch) -> Result<(), KrishivError> {
        if !self.has_required {
            return Ok(());
        }
        check_required(&self.table, batch)
    }

    /// Finishes every combined writer, then commits (module docs).
    async fn commit(&self, global: &SinkGlobal, ctx: &TaskContext) -> Result<(), KrishivError> {
        let writers = std::mem::take(&mut *lock(&global.writers));
        let t = &self.target;
        let mut written = Written {
            file_io: t.file_io.clone(),
            runtime: t.registered.io.scheduler().runtime().cloned(),
            data: Vec::new(),
            manifests: Vec::new(),
            armed: true,
        };
        let mut keeps = Vec::with_capacity(writers.len());
        let mut files: Vec<WrittenDataFile> = Vec::new();
        let mut sketches = Vec::with_capacity(writers.len());
        let mut first_error = None;
        for w in writers {
            keeps.push(w.keep);
            match w.done.await {
                Ok(Ok((f, s))) => {
                    files.extend(f);
                    sketches.push(s);
                }
                Ok(Err(e)) => {
                    first_error.get_or_insert(e);
                }
                Err(_) => {
                    first_error
                        .get_or_insert_with(|| internal("an Iceberg writer task died".to_owned()));
                }
            }
        }
        // Each task deletes its own files unless told to keep them; once
        // kept, they are this finalize's to delete.
        let keep = first_error.is_none();
        for k in keeps {
            let _ = k.send(keep);
        }
        if let Some(e) = first_error {
            return Err(e);
        }
        written.data = files.iter().map(|f| f.location.clone()).collect();
        if let Err(e) = ctx.query().check_cancelled() {
            written.delete().await;
            return Err(e);
        }
        let schema = t.base.current_schema();
        let entries = match files
            .iter()
            .map(|f| data_file(f, schema).map(|d| (f.spec_id, d)))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(e) => e,
            Err(e) => {
                written.delete().await;
                return Err(e);
            }
        };
        let mut overwrite = self.overwrite.clone();
        if let Some(o) = overwrite.as_mut().filter(|o| o.removals.replace_partitions) {
            // Spark skips an empty dynamic overwrite (`SparkWrite.java:341-344`).
            if entries.is_empty() {
                written.delete().await;
                return Ok(());
            }
            if let Err(e) = replaced_partitions(o, &t.base, &entries) {
                written.delete().await;
                return Err(e);
            }
        }
        // E34: the statement committed before (a retry opens the table anew, so its base has the
        // earlier snapshot): this attempt's files are orphans and nothing is committed.
        if let Some(key) = &self.idempotency_key
            && t.base.snapshots().any(|s| {
                s.summary()
                    .additional_properties
                    .get(super::append::IDEMPOTENCY_KEY_PROP)
                    .is_some_and(|v| v == key)
            })
        {
            written.delete().await;
            return Ok(());
        }
        let snapshot_id = snapshot_id(&t.base);
        let commit_uuid = random_uuid();
        let mut new = match write_new_manifests(
            &t.file_io,
            &t.base,
            entries,
            snapshot_id,
            &commit_uuid,
            // Each manifest's path is in the list before its file exists, so
            // a failure or a dropped finalize deletes what was written.
            &mut written.manifests,
        )
        .await
        {
            Ok(n) => n,
            Err(e) => {
                written.delete().await;
                return Err(e);
            }
        };
        // The tasks' sketches, unioned (statistics never fail the write).
        if let Some(columns) = &self.sketch_columns {
            new.statistics = union_tasks(columns, sketches, ctx.consumer().reservation());
        }
        // From here the commit loop owns cleanup (design note §7.4).
        written.armed = false;
        let query_id = ctx.query().id().0.to_string();
        let cx = CommitContext {
            io: &t.file_io,
            clock: self.clock.as_ref(),
            max_metadata_bytes: t.max_metadata_bytes,
        };
        let ids = WriteIds {
            branch: BRANCH,
            write_id: &self.write_id,
            query_id: Some(&query_id),
            idempotency_key: self.idempotency_key.as_deref(),
        };
        match &overwrite {
            None => commit_append(t.committer.as_ref(), cx, &t.base, &mut new, ids)
                .await
                .map(drop),
            Some(o) => {
                new.removals = Some(o.removals.clone());
                commit_validated(
                    t.committer.as_ref(),
                    cx,
                    &t.base,
                    &mut new,
                    ids,
                    &o.validation,
                )
                .await
                .map(drop)
            }
        }
    }
}

/// A `ReplacePartitions`' removals and conflict scope from the files it
/// writes (`BaseReplacePartitions.java:51-56`, `:113-117`): their partitions
/// (spec id and tuple, compared by value: `filter::partition_key`), or
/// every row when none of their spec's fields is non-`void` (validated over
/// `true`, Java's `alwaysTrue`).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a written partition value that cannot
/// be read back (never matched, so never dropped: refused instead).
fn replaced_partitions(
    o: &mut Overwrite,
    base: &TableMetadata,
    entries: &[(i32, iceberg::spec::DataFile)],
) -> Result<(), KrishivError> {
    let schema = base.current_schema();
    // Java's `PartitionSpec.isUnpartitioned`: no field, or only `void` fields.
    let unpartitioned = entries.iter().all(|(spec_id, _)| {
        base.partition_spec_by_id(*spec_id).is_some_and(|s| {
            s.fields()
                .iter()
                .all(|f| f.transform == iceberg::spec::Transform::Void)
        })
    });
    if unpartitioned {
        o.removals.row_filter = Some(krishiv_io::PushdownExpr::Constant(true));
        o.validation.filter = None;
        "true".clone_into(&mut o.validation.filter_text);
        return Ok(());
    }
    let specs = super::scan::partition_specs(base, schema);
    let mut paths = std::collections::BTreeSet::new();
    for (spec_id, f) in entries {
        let tuple = super::scan::content_file(f, *spec_id).partition;
        let key = partition_key(&specs, *spec_id, &tuple).ok_or_else(|| {
            KrishivError::from(KrishivUnsupportedFeature {
                feature: format!(
                    "a dynamic overwrite writing a partition value Krishiv cannot read back ({})",
                    f.file_path()
                ),
            })
        })?;
        o.removals.drop_partitions.insert(key);
        if let Some(spec) = base.partition_spec_by_id(*spec_id) {
            paths.insert(spec.partition_to_path(f.partition(), Arc::clone(schema)));
        }
    }
    o.validation.partitions = Some(PartitionSet {
        keys: o.removals.drop_partitions.clone(),
        text: format!("[{}]", paths.into_iter().collect::<Vec<_>>().join(", ")),
    });
    Ok(())
}

/// The files of a write not yet handed to the commit loop: deleted on
/// failure, or by `Drop` (in a task on the I/O runtime) when the finalize is
/// dropped first.
pub(crate) struct Written {
    file_io: FileIO,
    /// The I/O runtime.
    runtime: Option<tokio::runtime::Handle>,
    /// Data (and delete) files written.
    pub(crate) data: Vec<String>,
    /// Manifests written.
    pub(crate) manifests: Vec<String>,
    /// Cleared once the commit loop owns the files.
    pub(crate) armed: bool,
}

impl Written {
    /// An armed guard over nothing yet, deleting through `file_io` (on the
    /// I/O runtime of `registered` when dropped).
    pub(crate) fn new(file_io: FileIO, registered: &Registered) -> Self {
        Self {
            file_io,
            runtime: registered.io.scheduler().runtime().cloned(),
            data: Vec::new(),
            manifests: Vec::new(),
            armed: true,
        }
    }

    /// Cancellation-safe: a path leaves the lists only once its delete is
    /// done, so a dropped delete leaves the rest to `Drop`.
    pub(crate) async fn delete(&mut self) {
        while let Some(p) = self.manifests.last().or(self.data.last()).cloned() {
            let _ = self.file_io.delete(&p).await;
            if self.manifests.pop().is_none() {
                self.data.pop();
            }
        }
        self.armed = false;
    }
}

impl Drop for Written {
    fn drop(&mut self) {
        if !self.armed || (self.data.is_empty() && self.manifests.is_empty()) {
            return;
        }
        let paths: Vec<String> = self
            .data
            .drain(..)
            .chain(self.manifests.drain(..))
            .collect();
        let io = self.file_io.clone();
        let handle = self
            .runtime
            .clone()
            .or_else(|| tokio::runtime::Handle::try_current().ok());
        if let Some(h) = handle {
            h.spawn(async move { delete_paths(&io, paths).await });
        }
    }
}

/// Deletes `paths`, best effort: a file left behind is in no manifest list.
async fn delete_paths(io: &FileIO, paths: Vec<String>) {
    for p in paths {
        let _ = io.delete(&p).await;
    }
}

impl Sink for IcebergWriteSink {
    fn name(&self) -> &'static str {
        "IcebergWrite"
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
        let ids = TaskIds {
            partition_id: ctx.partition(),
            attempt_id: unique_u64(),
            write_id: &self.write_id,
        };
        let sketches = self
            .sketch_columns
            .as_ref()
            .and_then(|c| TaskSketches::new(Arc::clone(c), ctx.consumer().reservation()));
        let mut writer = DataFileWriter::with_registered(
            &self.target.base,
            self.target.registered.clone(),
            ids,
            self.target.max_open_files,
            ctx.consumer().reservation(),
        )?
        .with_sketches(sketches);
        if self.overwrite.as_ref().is_some_and(|o| o.lineage) {
            writer = writer.with_lineage()?;
        }
        let (batches_tx, batches_rx) = mpsc::channel(QUEUED_BATCHES);
        let (commit_tx, commit_rx) = oneshot::channel();
        let (done_tx, done_rx) = oneshot::channel();
        let (keep_tx, keep_rx) = oneshot::channel();
        let failed = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let task = WriterTask {
            writer,
            batches: batches_rx,
            commit: commit_rx,
            done: done_tx,
            keep: keep_rx,
            failed: Arc::clone(&failed),
            stop: Arc::clone(&stop),
            file_io: self.target.file_io.clone(),
        };
        // Detached: it ends on its own once its channels close.
        drop(krishiv_exec::console::spawn_named(
            ctx.executor(),
            ctx.query(),
            ctx.query().class(),
            &krishiv_exec::console::io_task_name("iceberg-write"),
            Box::pin(task.run()),
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
            .ok_or_else(|| internal("wrong local state for IcebergWrite".to_owned()))?;
        if !input.is_unweighted() {
            return Err(internal(
                "an Iceberg append takes insert-only rows (weight 1)".to_owned(),
            ));
        }
        let (data, _) = input.into_parts();
        self.check_nulls(&data)?;
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
                .unwrap_or_else(|| internal("the Iceberg writer task ended early".to_owned()))),
            Err(mpsc::error::TrySendError::Full(_)) => Err(internal(
                "the Iceberg writer's queue was full after a wait".to_owned(),
            )),
        }
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> krishiv_exec::Result<()> {
        let mut local = local
            .downcast::<SinkLocal>()
            .map_err(|_| internal("wrong local state for IcebergWrite".to_owned()))?;
        let global = global
            .downcast_ref::<SinkGlobal>()
            .ok_or_else(|| internal("wrong global state for IcebergWrite".to_owned()))?;
        // Finish the files once the queued batches are written.
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
                .ok_or_else(|| internal("wrong global state for IcebergWrite".to_owned()))?;
            self.commit(global, ctx).await
        })
    }
}

/// Refuses `input` unless it has `table`'s columns in order, by name, of the
/// table's types (module docs).
fn check_input(input: &SchemaRef, table: &SchemaRef, name: &str) -> Result<(), KrishivError> {
    let show = |s: &SchemaRef| {
        s.fields()
            .iter()
            .map(|f| format!("{}: {}", f.name(), f.data_type()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let fits = input.fields().len() == table.fields().len()
        && input
            .fields()
            .iter()
            .zip(table.fields())
            .all(|(a, b)| a.name() == b.name() && same_shape(a.data_type(), b.data_type()));
    if fits {
        Ok(())
    } else {
        Err(KrishivUnsupportedFeature {
            feature: format!(
                "a write into the Iceberg table {name} of rows ({}) that are not its columns ({})",
                show(input),
                show(table)
            ),
        }
        .into())
    }
}

/// Whether `a` and `b` are one type up to nullability and field metadata
/// (what the writer's retyping accepts): struct fields by name in order.
fn same_shape(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::Struct(x), DataType::Struct(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| p.name() == q.name() && same_shape(p.data_type(), q.data_type()))
        }
        (DataType::List(x), DataType::List(y))
        | (DataType::LargeList(x), DataType::LargeList(y))
        | (DataType::Map(x, _), DataType::Map(y, _)) => same_shape(x.data_type(), y.data_type()),
        _ => a == b,
    }
}

/// Whether `field` or anything below it is required.
fn has_required(field: &Field) -> bool {
    if !field.is_nullable() {
        return true;
    }
    match field.data_type() {
        DataType::Struct(fields) => fields.iter().any(|f| has_required(f)),
        DataType::List(f) | DataType::LargeList(f) | DataType::Map(f, _) => has_required(f),
        _ => false,
    }
}

/// Fails on a NULL in a required position of `batch`, whose columns follow `table`'s fields.
pub(super) fn check_required(table: &SchemaRef, batch: &RecordBatch) -> Result<(), KrishivError> {
    let mut path = Vec::new();
    for (column, field) in batch.columns().iter().zip(table.fields()) {
        path.push(field.name().clone());
        if null_in_required(column, field, None, &mut path) {
            return Err(NotNullAssertViolation {
                walked_type_path: format!("\n{}\n", path.join("\n")),
            }
            .into());
        }
        path.pop();
    }
    Ok(())
}

/// Whether `array` (of the table field `field`) holds a NULL in a required
/// position among the rows `reach` marks (`None`: every row); `path` ends
/// at that position when it does.
pub(super) fn null_in_required(
    array: &ArrayRef,
    field: &Field,
    reach: Option<&[bool]>,
    path: &mut Vec<String>,
) -> bool {
    let reachable = |i: usize| reach.is_none_or(|r| r.get(i).copied().unwrap_or(false));
    if !field.is_nullable()
        && let Some(nulls) = array.logical_nulls()
        && (0..array.len()).any(|i| nulls.is_null(i) && reachable(i))
    {
        return true;
    }
    if !has_required_below(field) {
        return false;
    }
    // Rows below a NULL are not reachable.
    let live = |i: usize| reachable(i) && array.is_valid(i);
    let mut child = |a: &ArrayRef, f: &FieldRef, name: &str, mask: &[bool]| {
        path.push(name.to_owned());
        let found = null_in_required(a, f, Some(mask), path);
        if !found {
            path.pop();
        }
        found
    };
    match field.data_type() {
        DataType::Struct(fields) => {
            let Some(s) = array.as_struct_opt() else {
                return false;
            };
            let mask: Vec<bool> = (0..array.len()).map(live).collect();
            s.columns()
                .iter()
                .zip(fields.iter())
                .any(|(a, f)| child(a, f, f.name(), &mask))
        }
        DataType::List(f) => array.as_list_opt::<i32>().is_some_and(|l| {
            let offsets: Vec<usize> = l
                .offsets()
                .iter()
                .map(|o| o.unsigned_abs() as usize)
                .collect();
            let mask = ranges(&offsets, l.values().len(), live);
            child(l.values(), f, "element", &mask)
        }),
        DataType::LargeList(f) => array.as_list_opt::<i64>().is_some_and(|l| {
            let offsets: Vec<usize> = l
                .offsets()
                .iter()
                .map(|o| usize::try_from(*o).unwrap_or(0))
                .collect();
            let mask = ranges(&offsets, l.values().len(), live);
            child(l.values(), f, "element", &mask)
        }),
        DataType::Map(entries, _) => {
            let (Some(m), DataType::Struct(kv)) = (array.as_map_opt(), entries.data_type()) else {
                return false;
            };
            let offsets: Vec<usize> = m
                .offsets()
                .iter()
                .map(|o| o.unsigned_abs() as usize)
                .collect();
            let mask = ranges(&offsets, m.entries().len(), live);
            let names = ["key", "value"];
            m.entries()
                .columns()
                .iter()
                .zip(kv.iter())
                .zip(names)
                .any(|((a, f), name)| child(a, f, name, &mask))
        }
        _ => false,
    }
}

/// Whether anything strictly below `field` is required.
fn has_required_below(field: &Field) -> bool {
    match field.data_type() {
        DataType::Struct(fields) => fields.iter().any(|f| has_required(f)),
        DataType::List(f) | DataType::LargeList(f) | DataType::Map(f, _) => has_required(f),
        _ => false,
    }
}

/// The child positions `offsets` gives the rows `live` keeps, as a mask of
/// `len` values.
fn ranges(offsets: &[usize], len: usize, live: impl Fn(usize) -> bool) -> Vec<bool> {
    let mut mask = vec![false; len];
    for (row, w) in offsets.windows(2).enumerate() {
        if live(row) {
            for m in mask.iter_mut().take(w[1]).skip(w[0]) {
                *m = true;
            }
        }
    }
    mask
}

/// A random key per process (module docs).
static PROCESS_KEY: LazyLock<u64> = LazyLock::new(|| {
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(std::process::id());
    if let Ok(t) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        h.write_u128(t.as_nanos());
    }
    h.finish()
});
static NEXT: AtomicU64 = AtomicU64::new(0);

/// A 64-bit value no other call in this process returns, and unpredictable
/// across processes (module docs): splitmix64's finalizer, a bijection, over
/// the counter's Weyl sequence keyed by [`PROCESS_KEY`].
pub(crate) fn unique_u64() -> u64 {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut z = PROCESS_KEY.wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A random version-4 UUID's text.
pub(crate) fn random_uuid() -> String {
    let hi = unique_u64();
    let lo = unique_u64();
    let hi = (hi & !0xF000) | 0x4000;
    let lo = (lo & !(0xC000_u64 << 48)) | (0x8000_u64 << 48);
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        hi >> 32,
        (hi >> 16) & 0xFFFF,
        hi & 0xFFFF,
        lo >> 48,
        lo & 0xFFFF_FFFF_FFFF
    )
}

/// A positive snapshot id the table does not have (Java's
/// `SnapshotIdGenerator`: a random positive long).
pub(crate) fn snapshot_id(base: &TableMetadata) -> i64 {
    loop {
        let id = i64::try_from(unique_u64() >> 1).unwrap_or(i64::MAX);
        if id != 0 && base.snapshot_by_id(id).is_none() {
            return id;
        }
    }
}

/// The production commit clock: wall time, sleeps on the I/O runtime.
struct IoClock {
    handle: tokio::runtime::Handle,
}

/// The production commit clock on `registered`'s I/O runtime.
///
/// # Errors
/// `INTERNAL_ERROR` when the I/O scheduler has no runtime.
pub(crate) fn io_clock(registered: &Registered) -> Result<Arc<dyn CommitClock>, KrishivError> {
    let handle = registered
        .io
        .scheduler()
        .runtime()
        .cloned()
        .ok_or_else(|| {
            internal("an Iceberg commit needs the I/O scheduler's runtime".to_owned())
        })?;
    Ok(Arc::new(IoClock { handle }))
}

#[async_trait]
impl CommitClock for IoClock {
    fn now_ms(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
    }

    async fn sleep(&self, wait: Duration) {
        // A join error (the runtime shutting down) ends the wait early.
        drop(self.handle.spawn(tokio::time::sleep(wait)).await);
    }

    fn jitter_ms(&self, bound: u64) -> u64 {
        unique_u64() % bound.max(1)
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod null_tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, ListArray, StructArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field, Fields};

    use super::null_in_required;

    fn path_of(array: &ArrayRef, field: &Field) -> Option<String> {
        let mut path = vec![field.name().clone()];
        null_in_required(array, field, None, &mut path).then(|| path.join("."))
    }

    /// A required struct field is checked only under a non-null struct.
    #[test]
    fn struct_fields_under_null_parents_are_not_violations() {
        // The table's field is required; the rows' (input) field is not.
        let field = Field::new(
            "s",
            DataType::Struct(Fields::from(vec![Field::new("k", DataType::Int32, false)])),
            true,
        );
        let k = Field::new("k", DataType::Int32, true);
        let values: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let under_null: ArrayRef = Arc::new(StructArray::new(
            Fields::from(vec![k.clone()]),
            vec![Arc::clone(&values)],
            Some(NullBuffer::from(vec![true, false])),
        ));
        assert_eq!(path_of(&under_null, &field), None);
        let under_valid: ArrayRef =
            Arc::new(StructArray::new(Fields::from(vec![k]), vec![values], None));
        assert_eq!(path_of(&under_valid, &field).as_deref(), Some("s.k"));
    }

    /// A required list element is named `element`; values outside every
    /// valid list's range are not looked at.
    #[test]
    fn list_elements_are_named_element() {
        let field = Field::new(
            "l",
            DataType::List(Arc::new(Field::new("element", DataType::Int32, false))),
            true,
        );
        let item = Arc::new(Field::new("item", DataType::Int32, true));
        let values: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
        // Rows [1], [null, 3]: a violation.
        let bad: ArrayRef = Arc::new(ListArray::new(
            Arc::clone(&item),
            OffsetBuffer::new(vec![0, 1, 3].into()),
            Arc::clone(&values),
            None,
        ));
        assert_eq!(path_of(&bad, &field).as_deref(), Some("l.element"));
        // Rows [1], null (over [null, 3]): none.
        let fine: ArrayRef = Arc::new(ListArray::new(
            item,
            OffsetBuffer::new(vec![0, 1, 3].into()),
            values,
            Some(NullBuffer::from(vec![true, false])),
        ));
        assert_eq!(path_of(&fine, &field), None);
    }
}
