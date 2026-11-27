// SPDX-License-Identifier: Apache-2.0
//! [`DeltaWriteSink`] and [`NewDeltaTable`] (see the parent module).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, MapArray, RecordBatch, StructArray};
use arrow_schema::SchemaRef;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::expressions::{Expression as KExpr, Predicate as Pred, PredicateRef, Scalar};
use delta_kernel::schema::{DataType, StructType};
use delta_kernel::transaction::create_table::{CreateTableTransaction, create_table};
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::transaction::{CommitResult, UpdateTableOperation};
use delta_kernel::{EvaluationHandler as _, Snapshot, SnapshotRef};
use futures::future::BoxFuture;
use krishiv_common::error::{
    DeltaCannotModifyAppendOnly, DeltaReplaceWhereMismatch, InternalError, KrishivError,
    TableOrViewAlreadyExists, UnsupportedFeatureTableOperation,
};
use krishiv_exec::planner::{TableCapabilities, TableProvider, WriteMode, WriteRequest};
use krishiv_exec::{
    CancellationToken, GlobalState, LocalState, QueryContext, ScanPushdown, Sink, SinkStatus,
    TaskContext,
};
use krishiv_memory::{Seat, SeatState, SpillSignal};
use krishiv_types::batch::Batch;
use tokio::sync::{Mutex as AsyncMutex, OnceCell};

use super::dml::RowLevelPlan;
use super::files::{DriverWriter, FileLimits, Plan, Written, delete_files};
use super::filter::{FilterRefusal, partition_predicate};
use super::{PREDICATE_TEXT_OPTION, READS_TARGET_OPTION};
use crate::commit::{CommitOperation, ReadSet, RetryLimits, StagedCommit, StagedRemoves, commit};
use crate::engine::{DeltaEngine, KrishivEngineData};
use crate::error::from_kernel;
use crate::features::{WriteProtocol, check_create, check_write};
use crate::table::{DeltaTable, hadoop_path, partition_scalar};

/// The default `delta.checkpointInterval`.
const DEFAULT_CHECKPOINT_INTERVAL: u64 = 10;

/// What a Delta write does to the rows already in the table.
#[derive(Debug, Clone)]
enum Mode {
    Append,
    Truncate,
    /// The files of the partitions the predicate selects; its text for `commitInfo`.
    Filter(PredicateRef, String),
    /// The files of the partitions written.
    Dynamic,
    /// A copy-on-write `DELETE` or `UPDATE` ([`super::dml`]).
    RowLevel(Arc<RowLevelPlan>),
}

impl Mode {
    fn removes(&self) -> bool {
        !matches!(self, Self::Append)
    }
}

/// Where the rows go.
#[derive(Debug, Clone)]
enum Target {
    /// An existing table, at the snapshot the statement read.
    Existing(DeltaTable),
    /// A table this write creates.
    Create(Arc<NewTable>),
}

/// A table a statement creates: its root, schema, partitioning and properties.
#[derive(Debug)]
pub(super) struct NewTable {
    engine: DeltaEngine,
    schema: Arc<StructType>,
    partition_columns: Vec<String>,
    properties: BTreeMap<String, String>,
    name: String,
}

#[cfg(test)]
impl NewTable {
    /// The table as errors name it.
    pub(super) fn name(&self) -> &str {
        &self.name
    }
}

/// The provider of a table `CREATE TABLE .. USING delta [AS SELECT]` creates (design section
/// 4.1, "CTAS"): its writer creates the table at version 0 with the written rows in one commit.
#[derive(Debug, Clone)]
pub struct NewDeltaTable {
    table: Arc<NewTable>,
    arrow: SchemaRef,
}

impl NewDeltaTable {
    /// The table at `engine`'s root with the columns of `schema` (the engine's Arrow forms),
    /// partitioned by `partition_columns`, with the table properties `properties`; `name` names
    /// it in errors.
    ///
    /// # Errors
    /// A column type Delta has no type for.
    pub fn new(
        engine: DeltaEngine,
        schema: &SchemaRef,
        partition_columns: Vec<String>,
        properties: BTreeMap<String, String>,
        name: Option<String>,
    ) -> Result<Self, KrishivError> {
        let name = name.unwrap_or_else(|| hadoop_path(engine.table_root()));
        let kernel =
            crate::engine::schema::to_kernel_schema(schema).map_err(|e| from_kernel(&e, &name))?;
        Ok(Self {
            table: Arc::new(NewTable {
                engine,
                schema: Arc::new(kernel),
                partition_columns,
                properties,
                name,
            }),
            arrow: Arc::clone(schema),
        })
    }
}

impl NewDeltaTable {
    /// The table this creates (for tests of the create commit).
    #[cfg(test)]
    pub(super) fn new_table(&self) -> &NewTable {
        &self.table
    }

    /// Creates the table with no rows (`CREATE TABLE .. USING delta` without a query): version
    /// 0 with the protocol and metadata only, carrying `write_id`.
    ///
    /// # Errors
    /// The kernel's create errors (a table already at the root); storage errors.
    pub async fn create_empty(&self, write_id: &str) -> Result<(), KrishivError> {
        let new = Arc::clone(&self.table);
        let write_id = write_id.to_owned();
        new.engine
            .clone()
            .delta_blocking(move |e| -> Result<(), KrishivError> {
                match plan_create(e, &new, &write_id, None, FileLimits::default())? {
                    Planned::Landed => Ok(()),
                    Planned::Ready { create, .. } => {
                        let txn = take_create(&create)?;
                        commit_create(e, txn, Vec::new(), &write_id, &new.name)
                            .map_err(Failure::into_error)
                    }
                }
            })
            .await?
    }
}

impl TableProvider for NewDeltaTable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        krishiv_exec::planner::MemoryTable::new(Arc::clone(&self.arrow), Vec::new()).scan()
    }

    fn capabilities(&self) -> TableCapabilities {
        let mut c = TableCapabilities::default();
        c.batch_write = true;
        c
    }

    fn writer(&self, request: &WriteRequest) -> krishiv_exec::Result<Arc<dyn Sink>> {
        self.capabilities()
            .check(&request.mode, &request.table_name)?;
        Ok(Arc::new(DeltaWriteSink {
            target: Target::Create(Arc::clone(&self.table)),
            mode: Mode::Append,
            write_id: write_id(request)?,
            idempotency_key: request.idempotency_key(),
            reads_target: false,
            limits: RetryLimits::default(),
            files: file_limits(request)?,
        }))
    }
}

/// The capabilities of an existing Delta table: append, truncate, overwrite by filter and
/// dynamic overwrite.
#[must_use]
pub(crate) fn capabilities() -> TableCapabilities {
    TableCapabilities::all_write_modes()
}

fn write_id(request: &WriteRequest) -> Result<String, KrishivError> {
    request
        .write_id
        .clone()
        .ok_or_else(|| internal("a Delta write without a write id"))
}

/// The writer of an existing table (see the parent module's refusals).
pub(crate) fn writer(
    table: &DeltaTable,
    request: &WriteRequest,
    limits: RetryLimits,
) -> Result<Arc<dyn Sink>, KrishivError> {
    capabilities().check(&request.mode, &request.table_name)?;
    let name = table.name().to_owned();
    let snapshot = table.snapshot();
    check_write(&name, &WriteProtocol::of(snapshot))?;
    let mode = match &request.mode {
        WriteMode::Append => Mode::Append,
        WriteMode::Truncate => Mode::Truncate,
        WriteMode::OverwriteDynamic => {
            if table.partition_columns().is_empty() {
                Mode::Truncate
            } else {
                Mode::Dynamic
            }
        }
        WriteMode::OverwriteByFilter(filter) => {
            let partitions = table.partition_columns();
            let pred = partition_predicate(filter, &snapshot.schema(), &partitions, &[]).map_err(
                |refusal| {
                    let what = match refusal {
                        FilterRefusal::DataColumn(c) => {
                            format!("an overwrite whose filter names the data column {c}")
                        }
                        FilterRefusal::Unsupported(w) => {
                            format!("an overwrite by a filter with {w}")
                        }
                    };
                    KrishivError::from(UnsupportedFeatureTableOperation {
                        table_name: request.table_name.clone(),
                        operation: format!(
                            "{what} (only partition-column filters are supported on Delta tables)"
                        ),
                    })
                },
            )?;
            let text = request
                .options
                .get(PREDICATE_TEXT_OPTION)
                .cloned()
                .unwrap_or_else(|| format!("{pred}"));
            Mode::Filter(Arc::new(pred), text)
        }
        other => {
            return Err(UnsupportedFeatureTableOperation {
                table_name: request.table_name.clone(),
                operation: other.operation().to_owned(),
            }
            .into());
        }
    };
    if mode.removes() {
        let props = snapshot.table_properties();
        append_only_refusal(table)?;
        if props.enable_change_data_feed == Some(true) {
            return Err(UnsupportedFeatureTableOperation {
                table_name: request.table_name.clone(),
                operation: format!(
                    "{} on a Delta table with change data feed enabled",
                    request.mode.operation()
                ),
            }
            .into());
        }
    }
    Ok(Arc::new(DeltaWriteSink {
        target: Target::Existing(table.clone()),
        mode,
        write_id: write_id(request)?,
        idempotency_key: request.idempotency_key(),
        reads_target: request
            .options
            .get(READS_TARGET_OPTION)
            .is_some_and(|v| v == "true"),
        limits,
        files: file_limits(request)?,
    }))
}

/// `DELTA_CANNOT_MODIFY_APPEND_ONLY` when `table` is append-only (`delta.appendOnly`), as
/// delta-spark's `DeltaLog.assertRemovable` raises it before a write that removes rows.
pub(super) fn append_only_refusal(table: &DeltaTable) -> Result<(), KrishivError> {
    if table.snapshot().table_properties().append_only != Some(true) {
        return Ok(());
    }
    let name = table.name().to_owned();
    // delta-spark names a path table `null` here (`tests/fixtures/delta-4.4.1/WRITES.json`).
    let table_name = if name == hadoop_path(table.engine().table_root()) {
        "null".to_owned()
    } else {
        name
    };
    Err(DeltaCannotModifyAppendOnly {
        table_name,
        config: "delta.appendOnly".to_owned(),
    }
    .into())
}

/// The sink of a copy-on-write row-level change of `table` ([`super::dml`]): the rewritten rows
/// are written as an overwrite's are, and the commit removes the touched files.
pub(super) fn row_level_sink(
    table: &DeltaTable,
    plan: RowLevelPlan,
    write_id: String,
    idempotency_key: Option<String>,
    limits: RetryLimits,
) -> Arc<dyn Sink> {
    Arc::new(DeltaWriteSink {
        target: Target::Existing(table.clone()),
        mode: Mode::RowLevel(Arc::new(plan)),
        write_id,
        idempotency_key,
        reads_target: true,
        limits,
        files: FileLimits::default(),
    })
}

/// The file limits of a write: Delta's `maxRecordsPerFile` write option (`DeltaOptions`, case
/// insensitive; read as a long, a value not above 0 is no limit) and the default target size.
fn file_limits(request: &WriteRequest) -> Result<FileLimits, KrishivError> {
    let mut limits = FileLimits::default();
    if let Some((key, raw)) = request
        .options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("maxRecordsPerFile"))
    {
        // Read as a long, as delta-spark does (no spaces around it); a value not above 0 is no
        // limit (`-1` too).
        let value = raw.parse::<i64>().map_err(|_| {
            KrishivError::from(krishiv_common::error::KrishivInvalidFormatOptionValue {
                value: format!("'{raw}'"),
                option: format!("\"{key}\""),
                format: "\"delta\"".to_owned(),
                reason: "expected a long integer".to_owned(),
            })
        })?;
        limits.max_records = u64::try_from(value).unwrap_or(0);
    }
    Ok(limits)
}

/// The sink of a Delta write (see the parent module).
#[derive(Debug)]
pub struct DeltaWriteSink {
    target: Target,
    mode: Mode,
    write_id: String,
    /// `krishiv.write.idempotencyKey` (E34).
    idempotency_key: Option<String>,
    reads_target: bool,
    limits: RetryLimits,
    files: FileLimits,
}

/// What a write's files are written against, made once on the first need.
pub(super) enum Planned {
    /// Write the files with `plan`; a create commits `create` at the end.
    Ready {
        plan: Arc<Plan>,
        create: Mutex<Option<Box<CreateTableTransaction>>>,
    },
    /// A create whose version 0 already carries this write's id (a retried task): nothing is
    /// written again.
    Landed,
}

/// The state shared by the drivers of one write.
struct WriteGlobal {
    /// The engine bound to the query's cancellation.
    engine: DeltaEngine,
    query: Arc<QueryContext>,
    target: Target,
    mode: Mode,
    write_id: String,
    /// `krishiv.write.idempotencyKey` (E34): a create records it in version 0.
    idempotency_key: Option<String>,
    files: FileLimits,
    planned: OnceCell<Arc<Planned>>,
    /// The writers of the drivers that ended their input.
    writers: Mutex<Vec<Arc<AsyncMutex<DriverWriter>>>>,
    /// A driver's batch failed: the write is abandoned.
    failed: std::sync::atomic::AtomicBool,
    /// When the sink started (a row-level change's `rewriteTimeMs`).
    started: std::time::Instant,
}

impl WriteGlobal {
    /// Abandons the write after a driver's failure: the writers combined so far, and any that
    /// combine later, are aborted.
    async fn abandon_all(&self) {
        self.failed.store(true, Ordering::Release);
        abort_all(&self.take_writers()).await;
    }

    fn take_writers(&self) -> Vec<Arc<AsyncMutex<DriverWriter>>> {
        std::mem::take(&mut *self.writers.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Lets the writers go just before the commit is attempted: from here a cancellation (or
    /// [`Drop`]) no longer deletes their files, since the commit may land (review 2, N2).
    /// Until then the writers stay here, so a cancellation anywhere in `finalize` aborts their
    /// uploads and deletes their finished files.
    fn release_writers(&self) {
        drop(self.take_writers());
    }
}

impl WriteGlobal {
    /// The write's plan, made on the log pool by the first driver that needs it.
    async fn planned(&self) -> Result<Arc<Planned>, KrishivError> {
        self.planned
            .get_or_try_init(|| async {
                let target = self.target.clone();
                let mode = self.mode.clone();
                let write_id = self.write_id.clone();
                let key = self.idempotency_key.clone();
                let files = self.files;
                let planned = self
                    .engine
                    .delta_blocking(move |e| match &target {
                        Target::Existing(table) => plan_existing(e, table, &mode, files),
                        Target::Create(new) => {
                            plan_create(e, new, &write_id, key.as_deref(), files)
                        }
                    })
                    .await??;
                Ok(Arc::new(planned))
            })
            .await
            .cloned()
    }
}

impl Drop for WriteGlobal {
    /// Writers combined but never finalized (another driver failed, the query was
    /// cancelled): their files are deleted in the background, best effort.
    fn drop(&mut self) {
        let writers =
            std::mem::take(&mut *self.writers.lock().unwrap_or_else(PoisonError::into_inner));
        for w in writers {
            abandon(&self.engine, w);
        }
    }
}

/// Gives back a writer's memory now and deletes its files in the background.
fn abandon(engine: &DeltaEngine, writer: Arc<AsyncMutex<DriverWriter>>) {
    if let Ok(mut w) = writer.try_lock() {
        let size = w.reservation().size();
        w.reservation().shrink(size);
    }
    drop(engine.handle().spawn(async move {
        writer.lock().await.abort().await;
    }));
}

/// One driver's state: its writer.
struct WriteLocal {
    engine: DeltaEngine,
    writer: Arc<AsyncMutex<DriverWriter>>,
    signal: Arc<SpillSignal>,
    /// The driver's seat in [`WRITE_COHORT`], holding while a batch is written: the writer is
    /// locked then, so a spill request cannot reach it, and another driver's round waits for
    /// the write to end instead of failing. The round counts the driver's whole use as about to
    /// be released, but the write gives all of it back only when a spill request set
    /// [`flush_wanted`](Self::flush_wanted); otherwise only the batch's copy. A round that
    /// waited and is still short re-plans in the abort phase, which asks for no spills, so it
    /// then fails as it did before (bounded by `abortReleaseTimeout`).
    seat: Arc<Mutex<Option<Seat>>>,
    /// A spill request found the writer busy: the batch's write flushes when it ends.
    flush_wanted: Arc<AtomicBool>,
    combined: bool,
}

/// The cohort of a query's Delta writers ([`WriteLocal::seat`]).
const WRITE_COHORT: &str = "delta-write";

/// Sets the driver's seat while a batch's write runs; idle again when it drops, also when the
/// write's future is dropped (cancellation).
struct Holding(Arc<Mutex<Option<Seat>>>);

impl Holding {
    fn new(seat: &Arc<Mutex<Option<Seat>>>) -> Self {
        set_seat(seat, SeatState::Holding);
        Self(Arc::clone(seat))
    }
}

impl Drop for Holding {
    fn drop(&mut self) {
        set_seat(&self.0, SeatState::Idle);
    }
}

fn set_seat(seat: &Mutex<Option<Seat>>, state: SeatState) {
    if let Some(seat) = seat.lock().unwrap_or_else(PoisonError::into_inner).as_mut() {
        seat.set(state);
    }
}

impl Drop for WriteLocal {
    fn drop(&mut self) {
        if !self.combined {
            // The driver ended without `combine`: cancelled or failed.
            abandon(&self.engine, Arc::clone(&self.writer));
        }
    }
}

impl Sink for DeltaWriteSink {
    fn name(&self) -> &'static str {
        "DeltaWrite"
    }

    fn global_state(&self, ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        let engine = match &self.target {
            Target::Existing(table) => table.engine().for_query(ctx.query()),
            Target::Create(new) => new.engine.for_query(ctx.query()),
        };
        let global = Arc::new(WriteGlobal {
            engine,
            query: Arc::clone(ctx.query()),
            target: self.target.clone(),
            mode: self.mode.clone(),
            write_id: self.write_id.clone(),
            idempotency_key: self.idempotency_key.clone(),
            files: self.files,
            planned: OnceCell::new(),
            writers: Mutex::new(Vec::new()),
            failed: std::sync::atomic::AtomicBool::new(false),
            started: std::time::Instant::now(),
        });
        #[cfg(test)]
        super::test_hooks::record(&self.write_id, &global);
        Ok(global)
    }

    /// The driver's writer, holding its memory in the driver's own consumer, where a
    /// `NeedMemory` grant lands.
    fn local_state(
        &self,
        ctx: &TaskContext,
        global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        let g = global_of(global)?;
        let memory = ctx.consumer();
        Ok(Box::new(WriteLocal {
            engine: g.engine.clone(),
            writer: Arc::new(AsyncMutex::new(DriverWriter::new(
                g.engine.clone(),
                memory.reservation(),
            ))),
            signal: Arc::clone(memory.signal()),
            seat: Arc::new(Mutex::new(memory.join_cohort(WRITE_COHORT))),
            flush_wanted: Arc::new(AtomicBool::new(false)),
            combined: false,
        }))
    }

    /// Reserves twice the batch (the batch and its partition copy) and what the open files
    /// held past the last reservation, or asks for it with `NeedMemory`; then writes the batch
    /// while the driver waits.
    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> krishiv_exec::Result<SinkStatus> {
        let l = local
            .downcast_mut::<WriteLocal>()
            .ok_or_else(|| internal("wrong local state for the Delta write"))?;
        let batch = input.into_parts().0;
        if batch.num_rows() == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        if let Mode::RowLevel(plan) = &self.mode {
            plan.check_rows()?;
        }
        let mut w = Arc::clone(&l.writer)
            .try_lock_owned()
            .map_err(|_| internal("a Delta writer is busy between batches"))?;
        if let Some(e) = &w.failed {
            return Err(e.clone());
        }
        let bytes = batch
            .get_array_memory_size()
            .saturating_mul(2)
            .saturating_add(w.deficit);
        if w.reservation().try_grow(bytes).is_err() {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        w.deficit = 0;
        let global = Arc::clone(global);
        let signal = Arc::clone(&l.signal);
        let flush_wanted = Arc::clone(&l.flush_wanted);
        #[cfg(test)]
        let driver = Arc::as_ptr(&l.writer) as usize;
        let holding = Holding::new(&l.seat);
        Ok(SinkStatus::Blocked(Box::pin(async move {
            let Ok(g) = global_of(&global) else {
                return;
            };
            let mut out = match g.planned().await {
                Ok(p) => match &*p {
                    Planned::Ready { plan, .. } => w.write(plan, &batch).await,
                    Planned::Landed => {
                        // Nothing is written: the batch's reservation goes back now.
                        let size = w.reservation().size();
                        w.reservation().shrink(size);
                        Ok(())
                    }
                },
                Err(e) => Err(e),
            };
            drop(batch);
            #[cfg(test)]
            if out.is_ok() {
                let site = super::test_hooks::Site {
                    point: super::test_hooks::Point::Sunk,
                    driver,
                };
                super::test_hooks::at(&g.write_id, site, &g.query, Some(&signal)).await;
            }
            // A round asked this driver to spill while the batch was written.
            if out.is_ok() && flush_wanted.swap(false, Ordering::AcqRel) {
                let flushed = w.flush().await;
                // A request that came during this flush is met by it (the writer was still
                // locked, so it set the flag again).
                flush_wanted.store(false, Ordering::Release);
                #[cfg(test)]
                if let Ok(freed) = &flushed {
                    super::test_hooks::note(
                        &g.write_id,
                        super::test_hooks::Path::WantedFlush,
                        *freed,
                    );
                }
                out = flushed.map(drop);
            }
            if let Err(e) = out {
                // Clean up before failing the query: its cancellation drops this future.
                w.abort().await;
                g.abandon_all().await;
                g.query.fail(e.clone());
                w.failed = Some(e);
            }
            signal.set_spillable_bytes(w.reservation().size());
            // The memory a waiting round counted on is given back (or the write failed).
            drop(holding);
            #[cfg(test)]
            {
                // The seat is idle and the writer free: a spill request reaches it again.
                drop(w);
                let site = super::test_hooks::Site {
                    point: super::test_hooks::Point::Written,
                    driver,
                };
                super::test_hooks::at(&g.write_id, site, &g.query, None).await;
            }
        })))
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> krishiv_exec::Result<()> {
        let mut l = local
            .downcast::<WriteLocal>()
            .map_err(|_| internal("wrong local state for the Delta write"))?;
        l.combined = true;
        // The driver ends here: nothing will service a spill request on its signal again
        // (review 2, N3), so its buffered row groups are shipped now instead.
        l.signal.set_spillable_bytes(0);
        let g = global_of(global)?;
        let writer = Arc::clone(&l.writer);
        if g.failed.load(Ordering::Acquire) {
            abandon(&g.engine, writer);
            return Ok(());
        }
        release_combined(g, &writer, &l.seat);
        g.writers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::clone(&writer));
        if let Ok(w) = writer.try_lock()
            && let Some(e) = &w.failed
        {
            return Err(e.clone());
        }
        Ok(())
    }

    /// Encodes and ships the buffered row groups of the driver's open files.
    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, krishiv_exec::Result<usize>> {
        Box::pin(async move {
            let Some(l) = local.downcast_mut::<WriteLocal>() else {
                return Err(internal("wrong local state for the Delta write"));
            };
            let Ok(mut w) = l.writer.try_lock() else {
                // A batch is being written (its seat holds): it flushes when it ends.
                l.flush_wanted.store(true, Ordering::Release);
                return Ok(0);
            };
            l.flush_wanted.store(false, Ordering::Release);
            let freed = w.flush().await?;
            l.signal.set_spillable_bytes(w.reservation().size());
            Ok(freed)
        })
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, krishiv_exec::Result<()>> {
        Box::pin(async move {
            let g = global_of(global)?;
            // A copy: the writers stay in `g` until the commit is attempted, so a cancellation
            // of this future leaves them to `Drop for WriteGlobal` (review 2, N2).
            let writers = g
                .writers
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            let planned = match g.planned().await {
                Ok(p) => p,
                Err(e) => {
                    g.abandon_all().await;
                    return Err(e);
                }
            };
            let (plan, create) = match &*planned {
                Planned::Landed => {
                    g.abandon_all().await;
                    return Ok(());
                }
                Planned::Ready { plan, create } => (plan, create),
            };
            let mut written = Written::default();
            let mut first = None;
            for w in &writers {
                let mut w = w.lock().await;
                let one = match w.failed.clone() {
                    // Its background flush after `combine` failed (`release_combined`).
                    Some(e) => Err(e),
                    None => w.finish(plan).await,
                };
                match one {
                    Ok(one) => written.merge(one),
                    Err(e) => {
                        first.get_or_insert(e);
                    }
                }
            }
            drop(writers);
            if let Some(e) = first {
                // Every writer still lists its finished files: `abort` deletes them.
                g.abandon_all().await;
                return Err(e);
            }
            #[cfg(test)]
            {
                let site = super::test_hooks::Site {
                    point: super::test_hooks::Point::Finished,
                    driver: 0,
                };
                super::test_hooks::at(&g.write_id, site, &g.query, None).await;
            }
            let paths = written.paths.clone();
            let out = match &self.target {
                Target::Existing(table) => self.commit_existing(g, table, written).await,
                Target::Create(new) => {
                    let txn = match take_create(create) {
                        Ok(txn) => txn,
                        Err(e) => {
                            g.abandon_all().await;
                            return Err(e);
                        }
                    };
                    g.release_writers();
                    let (new, write_id) = (Arc::clone(new), self.write_id.clone());
                    let adds = std::mem::take(&mut written.adds);
                    match g
                        .engine
                        .delta_blocking(move |e| commit_create(e, txn, adds, &write_id, &new.name))
                        .await
                    {
                        Ok(r) => r,
                        Err(e) => Err(Failure::Unknown(e)),
                    }
                }
            };
            match out {
                Ok(()) => Ok(()),
                Err(Failure::NotCommitted(e)) => {
                    // Released first, so `Drop` does not delete them a second time.
                    g.release_writers();
                    delete_files(&g.engine, &paths).await;
                    Err(e)
                }
                // The commit's outcome is not known: its files may be live.
                Err(Failure::Unknown(e)) => Err(e),
            }
        })
    }
}

/// Gives back what a driver's writer holds once the driver has ended (`combine`): no spill
/// request reaches it again, but another driver may still need memory (`UnableToAcquireMemory`
/// on CI, 2026-11-19, when one driver ran out of input first). No more rows come to its files,
/// so their buffered row groups are encoded and shipped now, in the background, rather than at
/// `finalize`, which changes no file's layout. While that runs the driver's seat holds, so
/// another driver's round waits for the bytes instead of failing; the bytes still held past the
/// reservation after it are reserved or left as the deficit, as after a batch. `finalize`
/// waits on the writer's lock; a failure here fails the write.
fn release_combined(
    g: &WriteGlobal,
    writer: &Arc<AsyncMutex<DriverWriter>>,
    seat: &Arc<Mutex<Option<Seat>>>,
) {
    let holding = Holding::new(seat);
    let writer = Arc::clone(writer);
    #[cfg(test)]
    let (write_id, query) = (g.write_id.clone(), Arc::clone(&g.query));
    drop(g.engine.handle().spawn(async move {
        let mut w = writer.lock().await;
        #[cfg(test)]
        {
            let site = super::test_hooks::Site {
                point: super::test_hooks::Point::Combined,
                driver: Arc::as_ptr(&writer) as usize,
            };
            super::test_hooks::at(&write_id, site, &query, None).await;
        }
        let out = match w.flush().await {
            #[cfg_attr(
                not(test),
                expect(unused_variables, reason = "counted by the test hooks")
            )]
            Ok(freed) => {
                #[cfg(test)]
                super::test_hooks::note(&write_id, super::test_hooks::Path::CombinedRelease, freed);
                w.settle().await
            }
            Err(e) => Err(e),
        };
        if let Err(e) = out {
            w.failed.get_or_insert(e);
        }
        drop(w);
        drop(holding);
    }));
}

/// Aborts every writer (their files are deleted).
async fn abort_all(writers: &[Arc<AsyncMutex<DriverWriter>>]) {
    for w in writers {
        w.lock().await.abort().await;
    }
}

fn global_of(global: &GlobalState) -> Result<&WriteGlobal, KrishivError> {
    <dyn std::any::Any + Send + Sync>::downcast_ref::<WriteGlobal>(&**global)
        .ok_or_else(|| internal("wrong global state for the Delta write"))
}

/// How a write failed after its files were written.
pub(super) enum Failure {
    /// Before its commit, or its commit surely did not land: the files are deleted.
    NotCommitted(KrishivError),
    /// The commit may have landed: the files stay.
    Unknown(KrishivError),
}

impl Failure {
    pub(super) fn into_error(self) -> KrishivError {
        match self {
            Self::NotCommitted(e) | Self::Unknown(e) => e,
        }
    }
}

/// The create transaction of a plan, once.
pub(super) fn take_create(
    create: &Mutex<Option<Box<CreateTableTransaction>>>,
) -> Result<Box<CreateTableTransaction>, KrishivError> {
    create
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .ok_or_else(|| internal("a Delta create without its transaction"))
}

/// The plan of a write into an existing table: one kernel transaction on the snapshot the
/// statement read gives the write state and the stats schema.
fn plan_existing(
    e: &DeltaEngine,
    table: &DeltaTable,
    mode: &Mode,
    files: FileLimits,
) -> Result<Planned, KrishivError> {
    let name = table.name();
    let snapshot = table.snapshot();
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, name);
    let txn = Arc::clone(snapshot)
        .update_table(
            UpdateTableOperation::Write,
            format!("krishiv {}", env!("CARGO_PKG_VERSION")),
            true,
        )
        .build(e, Box::new(FileSystemCommitter::new()))
        .map_err(|err| kernel(&err))?;
    let state = txn.write_state().map_err(|err| kernel(&err))?;
    let stats_schema = txn.stats_schema().map_err(|err| kernel(&err))?;
    let schema = snapshot.schema();
    let tight = snapshot
        .table_configuration()
        .protocol()
        .writer_features()
        .is_some_and(|f| f.iter().any(|f| f.to_string() == "deletionVectors"));
    let check = match mode {
        Mode::Filter(pred, text) => Some((Arc::clone(pred), text.clone(), Arc::clone(&schema))),
        Mode::Append | Mode::Truncate | Mode::Dynamic | Mode::RowLevel(_) => None,
    };
    Ok(Planned::Ready {
        plan: Arc::new(Plan {
            state,
            stats_schema,
            logical: Arc::new(
                crate::engine::schema::to_arrow_schema(&schema).map_err(|err| kernel(&err))?,
            ),
            partition_columns: table.partition_columns(),
            tight_bounds: tight,
            table: name.to_owned(),
            limits: files,
            check,
        }),
        create: Mutex::new(None),
    })
}

/// The plan of a create: the kernel's `create_table` transaction with the table-features
/// protocol, carrying the write's id and, for a keyed write, the key's `txn` action (E34: in
/// version 0 itself, so a retry of the statement on the created table finds it). When the transaction cannot be built because version 0
/// exists, it is this write's when version 0 carries its id (a retried task: [`Planned::Landed`]),
/// else `TABLE_OR_VIEW_ALREADY_EXISTS`.
pub(super) fn plan_create(
    e: &DeltaEngine,
    new: &NewTable,
    write_id: &str,
    key: Option<&str>,
    files: FileLimits,
) -> Result<Planned, KrishivError> {
    let name = new.name.as_str();
    check_create(name, &new.properties, &new.schema)?;
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, name);
    let mut builder = create_table(
        e.table_root().as_str(),
        Arc::clone(&new.schema),
        format!("krishiv {}", env!("CARGO_PKG_VERSION")),
        true,
    )
    .with_table_properties(new.properties.clone());
    if !new.partition_columns.is_empty() {
        builder = builder.with_data_layout(DataLayout::partitioned(new.partition_columns.clone()));
    }
    let built = builder
        .with_commit_info(
            Box::new(KrishivEngineData::new(crate::commit::write_id_row(
                write_id, name,
            )?)),
            crate::commit::write_id_schema(),
        )
        .build(e, Box::new(FileSystemCommitter::new()));
    let txn = match built {
        Ok(txn) => match key {
            Some(k) => txn.with_transaction_id(crate::commit::idempotency_app_id(k), 1),
            None => txn,
        },
        Err(err) => {
            return match version_zero(e, name)? {
                Some(owner) if owner.as_deref() == Some(write_id) => Ok(Planned::Landed),
                Some(_) => Err(TableOrViewAlreadyExists {
                    relation_name: name.to_owned(),
                }
                .into()),
                None => Err(kernel(&err)),
            };
        }
    };
    let state = txn.write_state().map_err(|err| kernel(&err))?;
    let stats_schema = txn.stats_schema().map_err(|err| kernel(&err))?;
    let tight = new
        .properties
        .get("delta.enableDeletionVectors")
        .is_some_and(|v| v.eq_ignore_ascii_case("true"));
    Ok(Planned::Ready {
        plan: Arc::new(Plan {
            state,
            stats_schema,
            logical: Arc::new(
                crate::engine::schema::to_arrow_schema(&new.schema).map_err(|err| kernel(&err))?,
            ),
            partition_columns: new.partition_columns.clone(),
            tight_bounds: tight,
            table: name.to_owned(),
            limits: files,
            check: None,
        }),
        create: Mutex::new(Some(Box::new(txn))),
    })
}

/// Commits a create (version 0) with `adds`, on the log pool. A lost race (`Conflicted`) or a
/// failure reads version 0: carrying this write's id, the create landed (the answer was lost,
/// or a retried task's first attempt committed); otherwise a lost race is
/// `TABLE_OR_VIEW_ALREADY_EXISTS` and a failure is raised.
pub(super) fn commit_create(
    e: &DeltaEngine,
    mut txn: Box<CreateTableTransaction>,
    adds: Vec<RecordBatch>,
    write_id: &str,
    name: &str,
) -> Result<(), Failure> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, name);
    for add in adds {
        txn.add_files(Box::new(KrishivEngineData::new(add)));
    }
    let failed = match (*txn).commit(e) {
        Ok(CommitResult::Committed(_)) => return Ok(()),
        Ok(CommitResult::Conflicted(_)) => None,
        Ok(CommitResult::Retryable(r)) => Some(kernel(&r.error)),
        Err(err) => Some(kernel(&err)),
    };
    let owner = version_zero(e, name).map_err(|err| match &failed {
        // Version 0 could not be read: the outcome is unknown; the commit's error is kept.
        Some(f) => Failure::Unknown(f.clone()),
        None => Failure::Unknown(err),
    })?;
    match (owner, failed) {
        (Some(Some(id)), _) if id == write_id => Ok(()),
        (Some(_), None) => Err(Failure::NotCommitted(
            TableOrViewAlreadyExists {
                relation_name: name.to_owned(),
            }
            .into(),
        )),
        (_, Some(f)) => Err(Failure::NotCommitted(f)),
        (None, None) => Err(Failure::Unknown(internal(
            "a Delta create conflicted with no version 0",
        ))),
    }
}

/// Version 0's commit, if there is one: the write id it carries.
fn version_zero(e: &DeltaEngine, name: &str) -> Result<Option<Option<String>>, KrishivError> {
    let commits = crate::commit::winner::read_from(e, 0, 1, name)?;
    Ok(commits
        .first()
        .map(|w| w.tag(crate::commit::WRITE_ID_TAG).map(str::to_owned)))
}

/// Post-commit checkpoints that failed or were cut short (design section 4.3).
static CHECKPOINT_FAILURES: AtomicU64 = AtomicU64::new(0);

/// How many post-commit checkpoints failed in this process: a checkpoint after a landed commit
/// never fails its statement, it is logged and counted here.
#[must_use]
pub fn checkpoint_failures() -> u64 {
    CHECKPOINT_FAILURES.load(Ordering::Relaxed)
}

impl DeltaWriteSink {
    /// Removes, checks and commits a write into an existing table.
    async fn commit_existing(
        &self,
        g: &WriteGlobal,
        table: &DeltaTable,
        written: Written,
    ) -> Result<(), Failure> {
        if let Mode::RowLevel(plan) = &self.mode {
            let orphans = written.paths.clone();
            let staged = super::dml::stage(
                &g.engine,
                g.started,
                table,
                plan,
                written,
                &self.write_id,
                &g.query,
            )
            .await?;
            return match staged {
                Some(mut staged) => {
                    staged.idempotency_key.clone_from(&self.idempotency_key);
                    if self.commit_staged(g, table, staged, orphans).await? {
                        // An earlier attempt made the change: this one claims no rows.
                        plan.deduplicated();
                    }
                    Ok(())
                }
                None => {
                    // Nothing changed: no commit, as delta-spark's `commitIfNeeded`.
                    g.release_writers();
                    Ok(())
                }
            };
        }
        let engine = &g.engine;
        let name = table.name().to_owned();
        let snapshot = Arc::clone(table.snapshot());
        let partitions = table.partition_columns();
        let predicate: Option<Option<PredicateRef>> = match &self.mode {
            Mode::RowLevel(_) => return Err(Failure::NotCommitted(internal("a row-level mode"))),
            Mode::Append => None,
            Mode::Truncate => Some(None),
            Mode::Filter(pred, _) => Some(Some(Arc::clone(pred))),
            Mode::Dynamic if written.partition_values.is_empty() => None,
            Mode::Dynamic => Some(Some(Arc::new(written_partitions(&partitions, &written)))),
        };
        let (removes, removed_files, removed_bytes) = match predicate {
            None => (Vec::new(), 0, 0),
            Some(predicate) => {
                let (s, n) = (Arc::clone(&snapshot), name.clone());
                engine
                    .delta_blocking(move |e| scan_removes(e, &s, &n, &partitions, predicate))
                    .await
                    .and_then(|r| r)
                    .map_err(Failure::NotCommitted)?
            }
        };
        let mut read_set = ReadSet::default();
        match &self.mode {
            Mode::Append => read_set.whole_table = self.reads_target,
            // Truncate and dynamic overwrite remove what they find anywhere in the table.
            Mode::Truncate | Mode::Dynamic | Mode::RowLevel(_) => read_set.whole_table = true,
            Mode::Filter(pred, _) => {
                read_set.predicates.push(Arc::clone(pred));
                read_set.whole_table = self.reads_target;
            }
        }
        let mut parameters = vec![
            (
                "mode".to_owned(),
                Some(
                    if self.mode.removes() {
                        "Overwrite"
                    } else {
                        "Append"
                    }
                    .to_owned(),
                ),
            ),
            ("partitionBy".to_owned(), Some("[]".to_owned())),
        ];
        match &self.mode {
            Mode::Dynamic => {
                parameters.push((
                    "isDynamicPartitionOverwrite".to_owned(),
                    Some("true".to_owned()),
                ));
            }
            Mode::Filter(_, text) => parameters.push(("predicate".to_owned(), Some(text.clone()))),
            _ => {}
        }
        let mut metrics = vec![
            ("numFiles".to_owned(), Some(written.files.to_string())),
            ("numOutputRows".to_owned(), Some(written.rows.to_string())),
            ("numOutputBytes".to_owned(), Some(written.bytes.to_string())),
        ];
        if self.mode.removes() {
            metrics.push((
                "numRemovedFiles".to_owned(),
                Some(removed_files.to_string()),
            ));
            metrics.push((
                "numRemovedBytes".to_owned(),
                Some(removed_bytes.to_string()),
            ));
        }
        if matches!(self.mode, Mode::Filter(..)) {
            // delta-spark's `replaceWhere` metrics (`tests/fixtures/delta-4.4.1/WRITES.json`):
            // a partition filter copies no rows and writes no change files.
            metrics.push(("numCopiedRows".to_owned(), Some("0".to_owned())));
            metrics.push(("numAddedChangeFiles".to_owned(), Some("0".to_owned())));
        }
        // An append or a dynamic overwrite that wrote no file commits nothing: delta-spark 4.4.1
        // writes no version for them (an empty `INSERT INTO`, one reading its own target, a
        // DataFrame append, a dynamic `INSERT OVERWRITE`; probe
        // `ops/reports/2026-11-18-o2-empty-append.md` in krishiv-internal), and the kernel refuses
        // a blind append without files. A truncate or `replaceWhere` still commits its removes.
        if matches!(self.mode, Mode::Append | Mode::Dynamic) && written.adds.is_empty() {
            g.release_writers();
            return Ok(());
        }
        let written_paths = written.paths.clone();
        let staged = StagedCommit {
            table: name.clone(),
            write_id: self.write_id.clone(),
            idempotency_key: self.idempotency_key.clone(),
            operation: CommitOperation::Write,
            operation_parameters: parameters,
            operation_metrics: metrics,
            adds: written.adds,
            removes,
            read_set,
        };
        self.commit_staged(g, table, staged, written_paths)
            .await
            .map(|_| ())
    }

    /// Commits `staged` on the snapshot the statement read, then starts a checkpoint when one
    /// is due. The writers are let go first: from here their files stay (the commit may land).
    /// Returns whether the commit was deduplicated (an earlier attempt of the statement landed).
    async fn commit_staged(
        &self,
        g: &WriteGlobal,
        table: &DeltaTable,
        staged: StagedCommit,
        orphans: Vec<object_store::path::Path>,
    ) -> Result<bool, Failure> {
        let engine = &g.engine;
        let name = table.name().to_owned();
        let snapshot = Arc::clone(table.snapshot());
        g.release_writers();
        let committed = commit(engine, snapshot, staged, self.limits)
            .await
            .map_err(Failure::Unknown)?;
        if committed.deduplicated {
            // An earlier attempt of this statement committed: this one's files are orphans.
            delete_files(engine, &orphans).await;
            return Ok(true);
        }
        let interval = table
            .snapshot()
            .table_properties()
            .checkpoint_interval
            .map_or(DEFAULT_CHECKPOINT_INTERVAL, std::num::NonZero::get);
        if committed
            .commits_since_checkpoint
            .is_some_and(|n| n >= interval)
        {
            // The commit has landed: the checkpoint runs detached from the statement (its
            // cancellation and its end), so it can never turn a landed write into a failure.
            spawn_checkpoint(engine, committed.version, name);
        }
        Ok(false)
    }
}

/// `OR` over the written partitions of `AND` over the partition columns of `col <=> value`.
fn written_partitions(partitions: &[String], written: &Written) -> Pred {
    Pred::or_from(written.partition_values.iter().map(|values| {
        Pred::and_from(partitions.iter().zip(values).map(|(p, v)| {
            Pred::not(Pred::distinct(
                KExpr::column([p.as_str()]),
                KExpr::literal(v.clone()),
            ))
        }))
    }))
}

/// The live files an overwrite removes, as scan rows, with their count and bytes: every file
/// for a truncate (`None`), else the files whose partition values satisfy `predicate`.
///
/// The kernel's scan selects a superset (a skipping filter it cannot build keeps every file),
/// so each selected file's partition values are checked again exactly, as delta-spark filters
/// its candidates (`WriteIntoDelta.scala:397-418`, `txn.filterFiles`): only TRUE removes. A
/// partition value that does not parse or a predicate that does not evaluate refuses the
/// overwrite rather than removing too much.
fn scan_removes(
    e: &DeltaEngine,
    snapshot: &SnapshotRef,
    name: &str,
    partitions: &[String],
    predicate: Option<PredicateRef>,
) -> Result<(Vec<StagedRemoves>, u64, u64), KrishivError> {
    let exact = predicate.clone();
    select_removes(e, snapshot, name, partitions, predicate, exact.as_ref())
}

/// [`scan_removes`] with the kernel's scan given `scan` (which may select more) and each
/// selected file checked against `exact`.
pub(super) fn select_removes(
    e: &DeltaEngine,
    snapshot: &SnapshotRef,
    name: &str,
    partitions: &[String],
    scan: Option<PredicateRef>,
    exact: Option<&PredicateRef>,
) -> Result<(Vec<StagedRemoves>, u64, u64), KrishivError> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, name);
    let exact = match exact {
        Some(p) => Some(ExactCheck::new(e, snapshot, partitions, p, name)?),
        None => None,
    };
    let scan = Arc::clone(snapshot)
        .scan_builder()
        .with_predicate(scan)
        .with_cancellation_token(e.kernel_cancellation())
        .build()
        .map_err(|err| kernel(&err))?;
    let mut out = Vec::new();
    let (mut files, mut bytes) = (0_u64, 0_u64);
    for metadata in scan.scan_metadata(e).map_err(|err| kernel(&err))? {
        let (data, selection) = metadata
            .map_err(|err| kernel(&err))?
            .scan_files
            .into_parts();
        let batch = KrishivEngineData::from_engine_data(data)
            .map_err(|err| kernel(&err))?
            .into_batch();
        let mut selection: Vec<bool> = (0..batch.num_rows())
            .map(|i| selection.get(i).copied().unwrap_or(true))
            .collect();
        if let Some(exact) = &exact {
            exact.narrow(&batch, &mut selection)?;
        }
        if let Some(size) = batch.column_by_name("size") {
            let size = arrow_cast::cast(size, &arrow_schema::DataType::Int64)
                .map_err(|err| internal(&format!("a scan row size: {err}")))?;
            let size = size.as_primitive::<Int64Type>();
            for (i, s) in selection.iter().enumerate() {
                if *s {
                    files += 1;
                    if size.is_valid(i) {
                        bytes += u64::try_from(size.value(i)).unwrap_or(0);
                    }
                }
            }
        }
        if selection.iter().any(|s| *s) {
            out.push(StagedRemoves {
                scan_rows: batch,
                selection,
            });
        }
    }
    Ok((out, files, bytes))
}

/// Every scan metadata batch of `snapshot`'s live files, with its selection, given to `f` as
/// the listing goes (nothing is kept here).
pub(super) fn for_each_scan_batch(
    e: &DeltaEngine,
    snapshot: &SnapshotRef,
    name: &str,
    mut f: impl FnMut(RecordBatch, Vec<bool>) -> Result<(), KrishivError>,
) -> Result<(), KrishivError> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, name);
    let scan = Arc::clone(snapshot)
        .scan_builder()
        .with_cancellation_token(e.kernel_cancellation())
        .build()
        .map_err(|err| kernel(&err))?;
    for metadata in scan.scan_metadata(e).map_err(|err| kernel(&err))? {
        let (data, selection) = metadata
            .map_err(|err| kernel(&err))?
            .scan_files
            .into_parts();
        let batch = KrishivEngineData::from_engine_data(data)
            .map_err(|err| kernel(&err))?
            .into_batch();
        let selection: Vec<bool> = (0..batch.num_rows())
            .map(|i| selection.get(i).copied().unwrap_or(true))
            .collect();
        f(batch, selection)?;
    }
    Ok(())
}

/// An overwrite predicate evaluated exactly over scan rows' partition values.
pub(super) struct ExactCheck {
    /// Per partition column: its physical name (the `partitionValues` key) and type.
    columns: Vec<(String, DataType)>,
    schema: Arc<StructType>,
    predicate: PredicateRef,
    handler: Arc<crate::engine::eval::KrishivEvaluationHandler>,
    name: String,
}

impl ExactCheck {
    pub(super) fn new(
        e: &DeltaEngine,
        snapshot: &SnapshotRef,
        partitions: &[String],
        predicate: &PredicateRef,
        name: &str,
    ) -> Result<Self, KrishivError> {
        let schema = snapshot.schema();
        let mapping = snapshot.table_configuration().column_mapping_mode();
        let mut columns = Vec::with_capacity(partitions.len());
        let mut fields = Vec::with_capacity(partitions.len());
        for p in partitions {
            let f = schema.field(p).ok_or_else(|| {
                internal(&format!("the partition column {p} is not in the schema"))
            })?;
            columns.push((f.physical_name(mapping).to_owned(), f.data_type().clone()));
            fields.push(f.clone());
        }
        Ok(Self {
            columns,
            schema: Arc::new(StructType::new_unchecked(fields)),
            predicate: Arc::clone(predicate),
            handler: e.evaluation(),
            name: name.to_owned(),
        })
    }

    fn refuse(&self, why: &str) -> KrishivError {
        UnsupportedFeatureTableOperation {
            table_name: self.name.clone(),
            operation: format!(
                "an overwrite whose partition values cannot be checked exactly ({why})"
            ),
        }
        .into()
    }

    /// Keeps selected only the rows of `batch` whose partition values make the predicate TRUE.
    ///
    /// # Errors
    /// The refusal (`UNSUPPORTED_FEATURE.TABLE_OPERATION`) when a value does not parse, the
    /// predicate does not evaluate, or a partitioned table's scan row carries no partition
    /// values (never read as all-NULL: that would remove the NULL partition's files for every
    /// file, review 2, N4).
    pub(super) fn narrow(
        &self,
        batch: &RecordBatch,
        selection: &mut [bool],
    ) -> Result<(), KrishivError> {
        let values = batch
            .column_by_name("fileConstantValues")
            .and_then(|c| c.as_any().downcast_ref::<StructArray>())
            .and_then(|s| s.column_by_name("partitionValues"))
            .and_then(|c| c.as_any().downcast_ref::<MapArray>());
        let mut rows = Vec::new();
        let mut at = Vec::new();
        for (i, selected) in selection.iter().enumerate() {
            if !*selected {
                continue;
            }
            let map: HashMap<String, Option<String>> = match values {
                Some(m) if m.is_valid(i) => crate::commit::map_row(m, i)?,
                _ if self.columns.is_empty() => HashMap::new(),
                _ => return Err(self.refuse("a file without its partition values")),
            };
            let row = self
                .columns
                .iter()
                .map(|(physical, t)| {
                    partition_scalar(map.get(physical).and_then(Option::as_ref), t)
                })
                .collect::<Result<Vec<Scalar>, _>>()
                .map_err(|err| self.refuse(&format!("a partition value: {err}")))?;
            rows.push(row);
            at.push(i);
        }
        if rows.is_empty() {
            return Ok(());
        }
        let data = self
            .handler
            .create_many(Arc::clone(&self.schema), rows)
            .map_err(|err| self.refuse(&err.to_string()))?;
        let evaluator = self
            .handler
            .new_predicate_evaluator(Arc::clone(&self.schema), Arc::clone(&self.predicate))
            .map_err(|err| self.refuse(&err.to_string()))?;
        let out = evaluator
            .evaluate(data.as_ref())
            .map_err(|err| self.refuse(&err.to_string()))?;
        let out = KrishivEngineData::from_engine_data(out)
            .map_err(|err| self.refuse(&err.to_string()))?
            .into_batch();
        let hits = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| self.refuse("the predicate is not a boolean"))?;
        if hits.len() != at.len() {
            return Err(self.refuse("the predicate returned another row count"));
        }
        for (k, i) in at.into_iter().enumerate() {
            selection[i] = hits.is_valid(k) && hits.value(k);
        }
        Ok(())
    }
}

/// delta-spark's `replaceWhere` constraint (`CheckDeltaInvariant.scala:51-53`: NULL or false
/// fails): every written row's partition values satisfy the filter.
pub(super) fn check_filter(
    e: &DeltaEngine,
    schema: &StructType,
    partitions: &[String],
    batches: &[RecordBatch],
    pred: &PredicateRef,
    text: &str,
    name: &str,
) -> Result<(), KrishivError> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, name);
    let fields = partitions
        .iter()
        .map(|p| {
            schema
                .field(p)
                .cloned()
                .ok_or_else(|| internal(&format!("the partition column {p} is not in the schema")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let part_schema = Arc::new(StructType::new_unchecked(fields));
    let arrow: SchemaRef =
        Arc::new(crate::engine::schema::to_arrow_schema(&part_schema).map_err(|err| kernel(&err))?);
    let evaluator = e
        .evaluation()
        .new_predicate_evaluator(Arc::clone(&part_schema), Arc::clone(pred))
        .map_err(|err| kernel(&err))?;
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let columns = partitions
            .iter()
            .zip(arrow.fields())
            .map(|(p, f)| {
                let c = batch
                    .column_by_name(p)
                    .ok_or_else(|| internal(&format!("the partition column {p} is not written")))?;
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
                    .map_err(|err| internal(&format!("the partition column {p}: {err}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let part = RecordBatch::try_new(Arc::clone(&arrow), columns)
            .map_err(|err| internal(&format!("a write's partition values: {err}")))?;
        let out = evaluator
            .evaluate(&KrishivEngineData::new(part.clone()))
            .map_err(|err| kernel(&err))?;
        let out = KrishivEngineData::from_engine_data(out)
            .map_err(|err| kernel(&err))?
            .into_batch();
        let result = out.column(0).as_boolean();
        if let Some(row) = (0..result.len()).find(|i| result.is_null(*i) || !result.value(*i)) {
            let values = partitions
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let v = arrow_cast::display::array_value_to_string(part.column(i), row)
                        .unwrap_or_default();
                    format!(" - {p} : {v}")
                })
                .collect::<Vec<_>>()
                .join("\n");
            return Err(DeltaReplaceWhereMismatch {
                replace_where: text.to_owned(),
                message: format!(
                    "[DELTA_VIOLATE_CONSTRAINT_WITH_VALUES] CHECK constraint EXPRESSION({text}) \
                     ({text}) violated by row with values:\n{values}"
                ),
            }
            .into());
        }
    }
    Ok(())
}

/// The time budget of a post-commit checkpoint: past it the checkpoint is cancelled (its
/// kernel calls end at their next handler call, giving back the log-pool slot) and counted as
/// failed. A `krishiv.delta.checkpointTimeout` key is a follow-up.
const CHECKPOINT_BUDGET: std::time::Duration = std::time::Duration::from_secs(600);

/// The table roots with a post-commit checkpoint running in this process.
static CHECKPOINTING: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Post-commit checkpoints skipped because one of the same table was running.
static CHECKPOINTS_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// How many post-commit checkpoints this process skipped because one of the same table root
/// was still running (at most one runs per root).
#[must_use]
pub fn checkpoints_skipped() -> u64 {
    CHECKPOINTS_SKIPPED.load(Ordering::Relaxed)
}

/// The one running checkpoint of a table root; dropping it lets the next one start.
#[derive(Debug)]
pub(super) struct CheckpointSlot(String);

impl CheckpointSlot {
    /// The slot of `root`, unless a checkpoint of `root` holds it.
    pub(super) fn begin(root: &str) -> Option<Self> {
        CHECKPOINTING
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(root.to_owned())
            .then(|| Self(root.to_owned()))
    }
}

impl Drop for CheckpointSlot {
    fn drop(&mut self) {
        CHECKPOINTING
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.0);
    }
}

/// Cancels its token when dropped: the checkpoint task's end, its time budget, or the runtime
/// shutting down (which drops the task) all end the kernel calls it started.
struct CancelOnDrop(Arc<CancellationToken>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Starts the checkpoint of `version` in the background (section 4.3): at most one per table
/// root at a time (another is skipped and counted), within [`CHECKPOINT_BUDGET`], cancelled
/// when the runtime drops it.
fn spawn_checkpoint(engine: &DeltaEngine, version: u64, name: String) {
    let Some(slot) = CheckpointSlot::begin(engine.table_root().as_str()) else {
        CHECKPOINTS_SKIPPED.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let token = Arc::new(CancellationToken::new());
    let detached = engine.detached(Arc::clone(&token));
    drop(engine.handle().spawn(async move {
        let _slot = slot;
        let _cancel = CancelOnDrop(token);
        let run = checkpoint(&detached, version, &name);
        if tokio::time::timeout(CHECKPOINT_BUDGET, run).await.is_err() {
            CHECKPOINT_FAILURES.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                table = name,
                version,
                "a Delta checkpoint after a commit ran past its time budget and was cancelled"
            );
        }
    }));
}

/// Writes a checkpoint of `version` (section 4.3) on `engine` (detached from the statement);
/// a failure is logged and counted, never raised: the commit has landed.
async fn checkpoint(engine: &DeltaEngine, version: u64, name: &str) {
    let root = engine.table_root().as_str().to_owned();
    let table = name.to_owned();
    let done = engine
        .delta_blocking(move |e| -> Result<(), KrishivError> {
            let snapshot = Snapshot::builder_for(&root)
                .with_cancellation_token(e.kernel_cancellation())
                .at_version(version)
                .build(e)
                .map_err(|err| from_kernel(&err, &table))?;
            snapshot
                .checkpoint(e, None)
                .map_err(|err| from_kernel(&err, &table))?;
            Ok(())
        })
        .await
        .and_then(|r| r);
    if let Err(e) = done {
        CHECKPOINT_FAILURES.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(error = %e, table = name, version, "a Delta checkpoint after a commit failed");
    }
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}
