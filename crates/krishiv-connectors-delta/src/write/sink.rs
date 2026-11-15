// SPDX-License-Identifier: Apache-2.0
//! [`DeltaWriteSink`] and [`NewDeltaTable`] (see the parent module).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
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
    GlobalState, LocalState, QueryContext, ScanPushdown, Sink, SinkStatus, TaskContext,
};
use krishiv_memory::SpillSignal;
use krishiv_types::batch::Batch;
use tokio::sync::{Mutex as AsyncMutex, OnceCell};

use super::files::{DriverWriter, FileLimits, Plan, Written, delete_files};
use super::filter::{FilterRefusal, partition_predicate};
use super::{PREDICATE_TEXT_OPTION, READS_TARGET_OPTION};
use crate::commit::{CommitOperation, ReadSet, RetryLimits, StagedCommit, StagedRemoves, commit};
use crate::engine::{DeltaEngine, KrishivEngineData};
use crate::error::from_kernel;
use crate::features::{WriteProtocol, check_write};
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
                match plan_create(e, &new, &write_id, FileLimits::default())? {
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
            let pred = partition_predicate(filter, &snapshot.schema(), &partitions).map_err(
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
        if props.append_only == Some(true) {
            // delta-spark names a path table `null` here (`tests/fixtures/delta-4.4.1/WRITES.json`).
            let table_name = if name == hadoop_path(table.engine().table_root()) {
                "null".to_owned()
            } else {
                name
            };
            return Err(DeltaCannotModifyAppendOnly {
                table_name,
                config: "delta.appendOnly".to_owned(),
            }
            .into());
        }
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
        reads_target: request
            .options
            .get(READS_TARGET_OPTION)
            .is_some_and(|v| v == "true"),
        limits,
        files: file_limits(request)?,
    }))
}

/// The file limits of a write: Delta's `maxRecordsPerFile` write option (`DeltaOptions`, case
/// insensitive) and the default target size.
fn file_limits(request: &WriteRequest) -> Result<FileLimits, KrishivError> {
    let mut limits = FileLimits::default();
    if let Some((key, raw)) = request
        .options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("maxRecordsPerFile"))
    {
        limits.max_records = raw.trim().parse::<u64>().map_err(|_| {
            KrishivError::from(krishiv_common::error::KrishivInvalidFormatOptionValue {
                value: format!("'{raw}'"),
                option: format!("\"{key}\""),
                format: "\"delta\"".to_owned(),
                reason: "expected an integer in 0..=18446744073709551615".to_owned(),
            })
        })?;
    }
    Ok(limits)
}

/// The sink of a Delta write (see the parent module).
#[derive(Debug)]
pub struct DeltaWriteSink {
    target: Target,
    mode: Mode,
    write_id: String,
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
    files: FileLimits,
    planned: OnceCell<Arc<Planned>>,
    /// The writers of the drivers that ended their input.
    writers: Mutex<Vec<Arc<AsyncMutex<DriverWriter>>>>,
    /// A driver's batch failed: the write is abandoned.
    failed: std::sync::atomic::AtomicBool,
}

impl WriteGlobal {
    /// Abandons the write after a driver's failure: the writers combined so far, and any that
    /// combine later, are aborted.
    async fn abandon_all(&self) {
        self.failed.store(true, Ordering::Release);
        let writers =
            std::mem::take(&mut *self.writers.lock().unwrap_or_else(PoisonError::into_inner));
        abort_all(&writers).await;
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
                let files = self.files;
                let planned = self
                    .engine
                    .delta_blocking(move |e| match &target {
                        Target::Existing(table) => plan_existing(e, table, &mode, files),
                        Target::Create(new) => plan_create(e, new, &write_id, files),
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
    combined: bool,
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
        Ok(Arc::new(WriteGlobal {
            engine,
            query: Arc::clone(ctx.query()),
            target: self.target.clone(),
            mode: self.mode.clone(),
            write_id: self.write_id.clone(),
            files: self.files,
            planned: OnceCell::new(),
            writers: Mutex::new(Vec::new()),
            failed: std::sync::atomic::AtomicBool::new(false),
        }))
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
        Ok(SinkStatus::Blocked(Box::pin(async move {
            let Ok(g) = global_of(&global) else {
                return;
            };
            let out = match g.planned().await {
                Ok(p) => match &*p {
                    Planned::Ready { plan, .. } => w.write(plan, &batch).await,
                    Planned::Landed => Ok(()),
                },
                Err(e) => Err(e),
            };
            drop(batch);
            if let Err(e) = out {
                // Clean up before failing the query: its cancellation drops this future.
                w.abort().await;
                g.abandon_all().await;
                g.query.fail(e.clone());
                w.failed = Some(e);
            }
            signal.set_spillable_bytes(w.reservation().size());
        })))
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> krishiv_exec::Result<()> {
        let mut l = local
            .downcast::<WriteLocal>()
            .map_err(|_| internal("wrong local state for the Delta write"))?;
        l.combined = true;
        let g = global_of(global)?;
        let writer = Arc::clone(&l.writer);
        if g.failed.load(Ordering::Acquire) {
            abandon(&g.engine, writer);
            return Ok(());
        }
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
                return Ok(0);
            };
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
            let writers =
                std::mem::take(&mut *g.writers.lock().unwrap_or_else(PoisonError::into_inner));
            let planned = match g.planned().await {
                Ok(p) => p,
                Err(e) => {
                    abort_all(&writers).await;
                    return Err(e);
                }
            };
            let (plan, create) = match &*planned {
                Planned::Landed => {
                    abort_all(&writers).await;
                    return Ok(());
                }
                Planned::Ready { plan, create } => (plan, create),
            };
            let mut written = Written::default();
            let mut first = None;
            for w in &writers {
                match w.lock().await.finish(plan).await {
                    Ok(one) => written.merge(one),
                    Err(e) => {
                        first.get_or_insert(e);
                    }
                }
            }
            if let Some(e) = first {
                abort_all(&writers).await;
                delete_files(&g.engine, &written.paths).await;
                return Err(e);
            }
            let paths = written.paths.clone();
            let out = match &self.target {
                Target::Existing(table) => self.commit_existing(&g.engine, table, written).await,
                Target::Create(new) => {
                    let txn = take_create(create)?;
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
                    delete_files(&g.engine, &paths).await;
                    Err(e)
                }
                // The commit's outcome is not known: its files may be live.
                Err(Failure::Unknown(e)) => Err(e),
            }
        })
    }
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
        _ => None,
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
/// protocol, carrying the write's id. When the transaction cannot be built because version 0
/// exists, it is this write's when version 0 carries its id (a retried task: [`Planned::Landed`]),
/// else `TABLE_OR_VIEW_ALREADY_EXISTS`.
pub(super) fn plan_create(
    e: &DeltaEngine,
    new: &NewTable,
    write_id: &str,
    files: FileLimits,
) -> Result<Planned, KrishivError> {
    let name = new.name.as_str();
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
        Ok(txn) => txn,
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
        engine: &DeltaEngine,
        table: &DeltaTable,
        written: Written,
    ) -> Result<(), Failure> {
        let name = table.name().to_owned();
        let snapshot = Arc::clone(table.snapshot());
        let partitions = table.partition_columns();
        let predicate: Option<Option<PredicateRef>> = match &self.mode {
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
            Mode::Truncate | Mode::Dynamic => read_set.whole_table = true,
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
        let staged = StagedCommit {
            table: name.clone(),
            write_id: self.write_id.clone(),
            operation: CommitOperation::Write,
            operation_parameters: parameters,
            operation_metrics: metrics,
            adds: written.adds,
            removes,
            read_set,
        };
        let committed = commit(engine, snapshot, staged, self.limits)
            .await
            .map_err(Failure::Unknown)?;
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
            let detached = engine.detached();
            let version = committed.version;
            drop(engine.handle().spawn(async move {
                checkpoint(&detached, version, &name).await;
            }));
        }
        Ok(())
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

/// An overwrite predicate evaluated exactly over scan rows' partition values.
struct ExactCheck {
    /// Per partition column: its physical name (the `partitionValues` key) and type.
    columns: Vec<(String, DataType)>,
    schema: Arc<StructType>,
    predicate: PredicateRef,
    handler: Arc<crate::engine::eval::KrishivEvaluationHandler>,
    name: String,
}

impl ExactCheck {
    fn new(
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
    fn narrow(&self, batch: &RecordBatch, selection: &mut [bool]) -> Result<(), KrishivError> {
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
                _ => HashMap::new(),
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
                    .or_else(|_| arrow_cast::cast(c, f.data_type()))
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
