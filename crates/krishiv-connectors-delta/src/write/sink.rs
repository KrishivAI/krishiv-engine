// SPDX-License-Identifier: Apache-2.0
//! [`DeltaWriteSink`] and [`NewDeltaTable`] (see the parent module).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use arrow_schema::SchemaRef;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::expressions::{Expression as KExpr, Predicate as Pred, PredicateRef};
use delta_kernel::schema::StructType;
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::transaction::{CommitResult, UpdateTableOperation};
use delta_kernel::{EvaluationHandler as _, Snapshot, SnapshotRef};
use futures::future::BoxFuture;
use krishiv_common::error::{
    DeltaCannotModifyAppendOnly, DeltaReplaceWhereMismatch, InternalError, KrishivError,
    TableOrViewAlreadyExists, UnsupportedFeatureTableOperation,
};
use krishiv_exec::planner::{TableCapabilities, TableProvider, WriteMode, WriteRequest};
use krishiv_exec::{GlobalState, LocalState, ScanPushdown, Sink, SinkStatus, TaskContext};
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::files::{Layout, Written, write_files};
use super::filter::{FilterRefusal, partition_predicate};
use super::{PREDICATE_TEXT_OPTION, READS_TARGET_OPTION};
use crate::commit::{CommitOperation, ReadSet, RetryLimits, StagedCommit, StagedRemoves, commit};
use crate::engine::{DeltaEngine, KrishivEngineData};
use crate::error::from_kernel;
use crate::features::{WriteProtocol, check_write};
use crate::table::{DeltaTable, hadoop_path};

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
#[derive(Debug)]
enum Target {
    /// An existing table, at the snapshot the statement read.
    Existing(DeltaTable),
    /// A table this write creates.
    Create(Arc<NewTable>),
}

/// A table a statement creates: its root, schema, partitioning and properties.
#[derive(Debug)]
struct NewTable {
    engine: DeltaEngine,
    schema: Arc<StructType>,
    partition_columns: Vec<String>,
    properties: BTreeMap<String, String>,
    name: String,
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
            .delta_blocking(move |e| create(e, &new, &[], &write_id))
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
    }))
}

/// The sink of a Delta write (see the parent module).
#[derive(Debug)]
pub struct DeltaWriteSink {
    target: Target,
    mode: Mode,
    write_id: String,
    reads_target: bool,
    limits: RetryLimits,
}

/// The rows a write buffered, and their memory.
struct Buffered {
    batches: Mutex<(Vec<RecordBatch>, Option<Reservation>)>,
}

impl Sink for DeltaWriteSink {
    fn name(&self) -> &'static str {
        "DeltaWrite"
    }

    fn global_state(&self, ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(Buffered {
            batches: Mutex::new((Vec::new(), Some(ctx.consumer().reservation()))),
        }))
    }

    fn local_state(
        &self,
        _ctx: &TaskContext,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        Ok(Box::new(()))
    }

    fn sink(
        &self,
        input: Batch,
        _local: &mut LocalState,
        global: &GlobalState,
    ) -> krishiv_exec::Result<SinkStatus> {
        let buffered = buffered(global)?;
        let batch = input.into_parts().0;
        let bytes = batch.get_array_memory_size();
        let mut guard = buffered
            .batches
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(r) = guard.1.as_mut()
            && r.try_grow(bytes).is_err()
        {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        guard.0.push(batch);
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, _local: LocalState, _global: &GlobalState) -> krishiv_exec::Result<()> {
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, krishiv_exec::Result<()>> {
        Box::pin(async move {
            let buffered = buffered(global)?;
            let (batches, reservation) = {
                let mut guard = buffered
                    .batches
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                (std::mem::take(&mut guard.0), guard.1.take())
            };
            let out = match &self.target {
                Target::Existing(table) => {
                    let engine = table.engine().for_query(ctx.query());
                    self.write_existing(&engine, table, batches).await
                }
                Target::Create(new) => {
                    let engine = new.engine.for_query(ctx.query());
                    self.write_create(&engine, new, batches).await
                }
            };
            drop(reservation);
            out
        })
    }
}

fn buffered(global: &GlobalState) -> Result<&Buffered, KrishivError> {
    <dyn std::any::Any + Send + Sync>::downcast_ref::<Buffered>(&**global)
        .ok_or_else(|| internal("wrong global state for the Delta write"))
}

/// What the files of an existing table's write came to, on the log pool.
struct Prepared {
    written: Written,
    removes: Vec<StagedRemoves>,
    removed_files: u64,
    removed_bytes: u64,
}

impl DeltaWriteSink {
    async fn write_existing(
        &self,
        engine: &DeltaEngine,
        table: &DeltaTable,
        batches: Vec<RecordBatch>,
    ) -> Result<(), KrishivError> {
        let name = table.name().to_owned();
        let snapshot = Arc::clone(table.snapshot());
        let partitions = table.partition_columns();
        let mode = self.mode.clone();
        let (s, n, p) = (Arc::clone(&snapshot), name.clone(), partitions.clone());
        let prepared = engine
            .delta_blocking(move |e| prepare(e, &s, &n, &p, &mode, &batches))
            .await??;
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
        let Prepared {
            written,
            removes,
            removed_files,
            removed_bytes,
        } = prepared;
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
        let committed = commit(engine, snapshot, staged, self.limits).await?;
        let interval = table
            .snapshot()
            .table_properties()
            .checkpoint_interval
            .map_or(DEFAULT_CHECKPOINT_INTERVAL, std::num::NonZero::get);
        if committed
            .commits_since_checkpoint
            .is_some_and(|n| n >= interval)
        {
            checkpoint(engine, committed.version, &name).await;
        }
        Ok(())
    }

    async fn write_create(
        &self,
        engine: &DeltaEngine,
        new: &Arc<NewTable>,
        batches: Vec<RecordBatch>,
    ) -> Result<(), KrishivError> {
        let new = Arc::clone(new);
        let write_id = self.write_id.clone();
        engine
            .delta_blocking(move |e| create(e, &new, &batches, &write_id))
            .await?
    }
}

/// The data files and removes of a write on an existing table, on the log pool.
fn prepare(
    e: &DeltaEngine,
    snapshot: &SnapshotRef,
    name: &str,
    partitions: &[String],
    mode: &Mode,
    batches: &[RecordBatch],
) -> Result<Prepared, KrishivError> {
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
    let layout = Layout {
        schema: &schema,
        partition_columns: partitions,
        state,
        stats_schema: &stats_schema,
        tight_bounds: tight,
        table: name,
    };
    if let Mode::Filter(pred, text) = mode {
        check_filter(e, &schema, partitions, batches, pred, text, name)?;
    }
    let written = write_files(e, &layout, batches)?;
    let predicate: Option<PredicateRef> = match mode {
        Mode::Append => return Ok(Prepared::only(written)),
        Mode::Truncate => None,
        Mode::Filter(pred, _) => Some(Arc::clone(pred)),
        Mode::Dynamic => {
            if written.partition_values.is_empty() {
                return Ok(Prepared::only(written));
            }
            Some(Arc::new(written_partitions(partitions, &written)))
        }
    };
    let (removes, files, bytes) =
        scan_removes(e, snapshot, predicate).map_err(|err| kernel(&err))?;
    Ok(Prepared {
        written,
        removes,
        removed_files: files,
        removed_bytes: bytes,
    })
}

impl Prepared {
    fn only(written: Written) -> Self {
        Self {
            written,
            removes: Vec::new(),
            removed_files: 0,
            removed_bytes: 0,
        }
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

/// The live files `predicate` selects (all with `None`), as scan rows to remove, with their
/// count and bytes.
fn scan_removes(
    e: &DeltaEngine,
    snapshot: &SnapshotRef,
    predicate: Option<PredicateRef>,
) -> Result<(Vec<StagedRemoves>, u64, u64), delta_kernel::KernelError> {
    let scan = Arc::clone(snapshot)
        .scan_builder()
        .with_predicate(predicate)
        .with_cancellation_token(e.kernel_cancellation())
        .build()?;
    let mut out = Vec::new();
    let (mut files, mut bytes) = (0_u64, 0_u64);
    for metadata in scan.scan_metadata(e)? {
        let (data, selection) = metadata?.scan_files.into_parts();
        let batch = KrishivEngineData::from_engine_data(data)?.into_batch();
        let selection: Vec<bool> = (0..batch.num_rows())
            .map(|i| selection.get(i).copied().unwrap_or(true))
            .collect();
        if let Some(size) = batch.column_by_name("size") {
            let size = arrow_cast::cast(size, &arrow_schema::DataType::Int64)
                .map_err(|err| delta_kernel::KernelError::generic(err.to_string()))?;
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

/// delta-spark's `replaceWhere` constraint (`CheckDeltaInvariant.scala:51-53`: NULL or false
/// fails): every written row's partition values satisfy the filter.
fn check_filter(
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

/// Creates the table `new` with the rows `batches` in one commit (version 0), on the log pool.
fn create(
    e: &DeltaEngine,
    new: &NewTable,
    batches: &[RecordBatch],
    write_id: &str,
) -> Result<(), KrishivError> {
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
    let mut txn = builder
        .with_commit_info(
            Box::new(KrishivEngineData::new(crate::commit::write_id_row(
                write_id, name,
            )?)),
            crate::commit::write_id_schema(),
        )
        .build(e, Box::new(FileSystemCommitter::new()))
        .map_err(|err| kernel(&err))?;
    let state = txn.write_state().map_err(|err| kernel(&err))?;
    let stats_schema = txn.stats_schema().map_err(|err| kernel(&err))?;
    let tight = new
        .properties
        .get("delta.enableDeletionVectors")
        .is_some_and(|v| v.eq_ignore_ascii_case("true"));
    let layout = Layout {
        schema: &new.schema,
        partition_columns: &new.partition_columns,
        state,
        stats_schema: &stats_schema,
        tight_bounds: tight,
        table: name,
    };
    let written = write_files(e, &layout, batches)?;
    for add in written.adds {
        txn.add_files(Box::new(KrishivEngineData::new(add)));
    }
    match txn.commit(e) {
        Ok(CommitResult::Committed(_)) => Ok(()),
        Ok(CommitResult::Conflicted(_)) => Err(TableOrViewAlreadyExists {
            relation_name: name.to_owned(),
        }
        .into()),
        Ok(CommitResult::Retryable(r)) => landed_create(e, write_id, name, &kernel(&r.error)),
        Err(err) => landed_create(e, write_id, name, &kernel(&err)),
    }
}

/// After a failed create commit: success when version 0 carries this write's id.
fn landed_create(
    e: &DeltaEngine,
    write_id: &str,
    name: &str,
    error: &KrishivError,
) -> Result<(), KrishivError> {
    let commits = crate::commit::winner::read_from(e, 0, 1, name)?;
    if commits
        .first()
        .is_some_and(|w| w.tag(crate::commit::WRITE_ID_TAG) == Some(write_id))
    {
        Ok(())
    } else {
        Err(error.clone())
    }
}

/// Writes a checkpoint of `version` (section 4.3); a failure is logged and the statement
/// still succeeds.
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
        .await;
    // The commit has landed: a failed checkpoint is abandoned, and the next commit past the
    // interval tries again (the crate has no logger to report it to).
    drop(done);
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}
