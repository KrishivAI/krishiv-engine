// SPDX-License-Identifier: Apache-2.0
//! Test tables for other crates (feature `testing`): a small Iceberg table written into a local
//! directory by iceberg-rust's in-memory catalog over Krishiv's I/O, with Parquet data files, so a
//! catalog test can serve its metadata and read it back through [`crate::IcebergTableProvider`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Struct, Type,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::files::LocalPaths;
use krishiv_io::{FIELD_ID_KEY, IoConfig, IoScheduler, StoreId};

use super::storage::{KrishivStorageFactory, StorageRegistration, from_iceberg};

/// A table [`write_table`] wrote.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WrittenTable {
    /// The current metadata file (`file://...`).
    pub metadata_location: String,
    /// That file's document, as a catalog returns it in `loadTable`.
    pub metadata: serde_json::Value,
    /// The table's root (`file://<dir>/db/t`).
    pub location: String,
    /// The snapshot ids, oldest first: the first holds rows 1 and 2, the second adds row 3.
    pub snapshots: Vec<i64>,
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// Writes `db.t (id BIGINT NOT NULL, name STRING)` under `dir` in two appends: `(1, 'a'), (2, 'b')`,
/// then `(3, 'c')`.
///
/// # Errors
/// Any failure of the runtime, the catalog or the writes, as `INTERNAL_ERROR`.
pub fn write_table(dir: &Path) -> Result<WrittenTable, KrishivError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(|e| internal(format!("a runtime: {e}")))?;
    let sched = IoScheduler::with_runtime(IoConfig::default(), rt.handle().clone());
    let id = StoreId::new("file://");
    sched.register_store(
        id.clone(),
        Arc::new(object_store::local::LocalFileSystem::new()),
    );
    let paths = Arc::new(LocalPaths::new(id, dir.to_string_lossy().into_owned()));
    let reg = {
        let _entered = rt.enter();
        StorageRegistration::new(sched.context(), paths)
    };
    let warehouse = format!("file://{}", dir.display());
    rt.block_on(write(&reg, dir, &warehouse))
}

async fn write(
    reg: &StorageRegistration,
    dir: &Path,
    warehouse: &str,
) -> Result<WrittenTable, KrishivError> {
    let catalog = iceberg::memory::MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(KrishivStorageFactory {
            name: reg.name().to_owned(),
        }))
        .with_runtime(reg.runtime()?)
        .load(
            "mem",
            [(
                iceberg::memory::MEMORY_CATALOG_WAREHOUSE.to_owned(),
                warehouse.to_owned(),
            )]
            .into(),
        )
        .await
        .map_err(from_iceberg)?;
    let ns = NamespaceIdent::new("db".to_owned());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .map_err(from_iceberg)?;
    let schema = iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                .with_doc("the name")
                .into(),
        ])
        .build()
        .map_err(from_iceberg)?;
    let mut table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("t".to_owned())
                .schema(schema)
                .build(),
        )
        .await
        .map_err(from_iceberg)?;
    let data = dir.join("db").join("t").join("data");
    std::fs::create_dir_all(&data).map_err(|e| internal(format!("a data directory: {e}")))?;
    let mut snapshots = Vec::new();
    for (n, rows) in [vec![(1, "a"), (2, "b")], vec![(3, "c")]]
        .into_iter()
        .enumerate()
    {
        let file = data.join(format!("{n}.parquet"));
        let size = write_parquet(&file, &rows)?;
        let entry = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("file://{}", file.display()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(size)
            .record_count(rows.len() as u64)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .map_err(|e| internal(format!("a data file entry: {e}")))?;
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![entry])
            .apply(tx)
            .map_err(from_iceberg)?;
        table = tx.commit(&catalog).await.map_err(from_iceberg)?;
        if let Some(s) = table.metadata().current_snapshot() {
            snapshots.push(s.snapshot_id());
        }
    }
    let metadata_location = table
        .metadata_location()
        .ok_or_else(|| internal("the table has no metadata location".to_owned()))?
        .to_owned();
    let local = metadata_location
        .strip_prefix("file://")
        .unwrap_or(&metadata_location);
    let text = std::fs::read(local).map_err(|e| internal(format!("the metadata file: {e}")))?;
    let metadata = serde_json::from_slice(&text)
        .map_err(|e| internal(format!("the metadata file's JSON: {e}")))?;
    Ok(WrittenTable {
        metadata_location,
        metadata,
        location: table.metadata().location().to_owned(),
        snapshots,
    })
}

/// `rows` as one Parquet file at `path` with the table's field ids; its size.
fn write_parquet(path: &Path, rows: &[(i64, &str)]) -> Result<u64, KrishivError> {
    let field = |name: &str, ty: DataType, nullable: bool, id: i32| {
        Field::new(name, ty, nullable)
            .with_metadata(HashMap::from([(FIELD_ID_KEY.to_owned(), id.to_string())]))
    };
    let schema = Arc::new(Schema::new(vec![
        field("id", DataType::Int64, false, 1),
        field("name", DataType::Utf8, true, 2),
    ]));
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0)));
    let names: ArrayRef = Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.1)));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, names])
        .map_err(|e| internal(format!("a batch: {e}")))?;
    let file = std::fs::File::create(path).map_err(|e| internal(format!("a data file: {e}")))?;
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None)
        .map_err(|e| internal(format!("a Parquet writer: {e}")))?;
    w.write(&batch)
        .map_err(|e| internal(format!("a Parquet write: {e}")))?;
    w.close()
        .map_err(|e| internal(format!("a Parquet close: {e}")))?;
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| internal(format!("a data file's size: {e}")))
}

/// Local I/O for a catalog test that opens [`crate::IcebergTableProvider`]s itself: a scheduler on
/// a runtime of its own with the local file system registered for `file://`.
pub struct LocalIo {
    runtime: tokio::runtime::Runtime,
    sched: Arc<IoScheduler>,
    paths: Arc<LocalPaths>,
}

impl std::fmt::Debug for LocalIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalIo").finish_non_exhaustive()
    }
}

impl LocalIo {
    /// The I/O of `dir` (relative paths resolve under it).
    ///
    /// # Errors
    /// The runtime cannot start (`INTERNAL_ERROR`).
    pub fn new(dir: &Path) -> Result<Self, KrishivError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| internal(format!("a runtime: {e}")))?;
        let sched = IoScheduler::with_runtime(IoConfig::default(), runtime.handle().clone());
        let id = StoreId::new("file://");
        sched.register_store(
            id.clone(),
            Arc::new(object_store::local::LocalFileSystem::new()),
        );
        let paths = Arc::new(LocalPaths::new(id, dir.to_string_lossy().into_owned()));
        Ok(Self {
            runtime,
            sched,
            paths,
        })
    }

    /// The query I/O.
    #[must_use]
    pub fn io(&self) -> krishiv_io::IoContext {
        self.sched.context()
    }

    /// The path placement.
    #[must_use]
    pub fn paths(&self) -> Arc<dyn krishiv_exec::files::PathLocator> {
        Arc::clone(&self.paths) as Arc<dyn krishiv_exec::files::PathLocator>
    }

    /// The runtime the I/O runs on.
    #[must_use]
    pub fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }
}

/// Why [`apply_commit`] refused a request, as an Iceberg REST server answers it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CommitRefusal {
    /// The HTTP status (409 for a failed requirement, 400 for a refused update).
    pub status: u16,
    /// The `ErrorModel` type.
    pub kind: String,
    /// The `ErrorModel` message (iceberg-rust's text for a refused update).
    pub message: String,
}

impl CommitRefusal {
    /// The `ErrorModel` body of the refusal.
    #[must_use]
    pub fn body(&self) -> String {
        serde_json::json!({
            "error": {"message": self.message, "type": self.kind, "code": self.status}
        })
        .to_string()
    }
}

/// What a fake Iceberg REST server does with the `CommitTableRequest` `body` against the table
/// whose metadata is `current` (read from `current_location`): every requirement checked (a failed
/// one is a 409 `CommitFailedException`), the updates applied through iceberg-rust's metadata
/// builder (a refusal is a 400 with iceberg-rust's text: stale, duplicate snapshot id, ...), and
/// the new metadata written to `next_location` (`file://`). The new metadata document.
///
/// # Errors
/// The refusal as the server answers it; a body or metadata that does not parse, or a write that
/// fails, is a 400 / 500 refusal.
pub fn apply_commit(
    current: &serde_json::Value,
    current_location: &str,
    body: &[u8],
    next_location: &str,
) -> Result<serde_json::Value, CommitRefusal> {
    use iceberg::spec::{TableMetadata, TableMetadataBuilder};
    use iceberg::{TableRequirement, TableUpdate};
    let bad = |status: u16, kind: &str, message: String| CommitRefusal {
        status,
        kind: kind.to_owned(),
        message,
    };
    let request: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| bad(400, "BadRequestException", format!("not JSON: {e}")))?;
    let requirements: Vec<TableRequirement> =
        serde_json::from_value(request["requirements"].clone())
            .map_err(|e| bad(400, "BadRequestException", format!("requirements: {e}")))?;
    let updates: Vec<TableUpdate> = serde_json::from_value(request["updates"].clone())
        .map_err(|e| bad(400, "BadRequestException", format!("updates: {e}")))?;
    let m: TableMetadata = serde_json::from_value(current.clone())
        .map_err(|e| bad(500, "ServiceFailureException", format!("metadata: {e}")))?;
    for r in &requirements {
        if let Err(e) = r.check(Some(&m)) {
            return Err(bad(
                409,
                "CommitFailedException",
                format!("Requirement failed: {e}"),
            ));
        }
    }
    let mut b = TableMetadataBuilder::new_from_metadata(m, Some(current_location.to_owned()));
    for u in updates {
        b = u
            .apply(b)
            .map_err(|e| bad(400, "BadRequestException", e.message().to_owned()))?;
    }
    let next = b
        .build()
        .map_err(|e| bad(400, "BadRequestException", e.message().to_owned()))?
        .metadata;
    let json = serde_json::to_value(&next)
        .map_err(|e| bad(500, "ServiceFailureException", format!("metadata: {e}")))?;
    let local = next_location
        .strip_prefix("file://")
        .unwrap_or(next_location);
    let text = serde_json::to_vec(&json)
        .map_err(|e| bad(500, "ServiceFailureException", format!("metadata: {e}")))?;
    std::fs::write(local, text)
        .map_err(|e| bad(500, "ServiceFailureException", format!("write: {e}")))?;
    Ok(json)
}

/// Appends `rows` (`id`, `name`, the [`write_table`] schema) through `provider`'s writer, run to
/// the end by the executor on a runtime of its own: the commit happens when the sink finalizes.
///
/// # Errors
/// The write's or the commit's failure.
pub fn append(
    provider: &dyn krishiv_exec::planner::TableProvider,
    rows: &[(i64, &str)],
) -> Result<(), KrishivError> {
    use arrow_array::StringViewArray;
    use futures::StreamExt;
    use krishiv_exec::planner::{WriteMode, WriteRequest};
    use krishiv_exec::{
        ExecConfig, ExecutorConfig, QueryContext, QueryId, QueryRunner, TokioExecutor,
        WorkloadClass,
    };
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};

    let schema = provider.scan()?.table_schema();
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0)));
    let names: ArrayRef = match schema.field(1).data_type() {
        DataType::Utf8View => Arc::new(StringViewArray::from_iter_values(rows.iter().map(|r| r.1))),
        _ => Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.1))),
    };
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, names])
        .map_err(|e| internal(format!("a batch: {e}")))?;
    let request = WriteRequest::new(WriteMode::Append, "`db`.`t`", Arc::clone(&schema));
    let sink = provider.writer(&request)?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| internal(format!("a runtime: {e}")))?;
    let root = TreePool::new(&MemoryConfig::default());
    let pool = root.child("append", None);
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        pool,
    );
    let runner = QueryRunner::new(TokioExecutor::new(&ExecutorConfig::default())?);
    let node = Arc::new(Write {
        input: Arc::new(krishiv_exec::testing::Scan(
            krishiv_exec::testing::Batches::new(
                schema,
                vec![krishiv_types::batch::Batch::new(batch)],
            ),
        )),
        sink,
    });
    rt.block_on(async {
        let mut stream = runner.execute(node, Arc::clone(&query)).await?;
        let mut result = Ok(());
        while let Some(b) = stream.next().await {
            if let Err(e) = b {
                result = Err(e);
            }
        }
        query.finished().await;
        result
    })
}

/// `input` written through `sink`.
struct Write {
    input: Arc<dyn krishiv_exec::ExecNode>,
    sink: Arc<dyn krishiv_exec::Sink>,
}

impl std::fmt::Debug for Write {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Write")
    }
}

impl krishiv_exec::ExecNode for Write {
    fn build_pipelines(
        self: Arc<Self>,
        b: &mut krishiv_exec::PipelineBuilder,
    ) -> krishiv_exec::Result<()> {
        let sink = b.child_pipeline(Arc::clone(&self.sink), Arc::clone(&self.input))?;
        b.set_breaker_source(Arc::new(Nothing), sink)?;
        Ok(())
    }
}

/// The source side of the write node: no rows.
struct Nothing;

impl krishiv_exec::Source for Nothing {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::empty())
    }
    fn global_state(
        &self,
        _ctx: &krishiv_exec::TaskContext,
    ) -> krishiv_exec::Result<krishiv_exec::GlobalState> {
        Ok(Arc::new(()))
    }
    fn max_parallelism(&self, _global: &krishiv_exec::GlobalState) -> usize {
        1
    }
    fn local_state(
        &self,
        _ctx: &krishiv_exec::TaskContext,
        _global: &krishiv_exec::GlobalState,
    ) -> krishiv_exec::Result<krishiv_exec::LocalState> {
        Ok(Box::new(()))
    }
    fn get_data<'a>(
        &'a self,
        _local: &'a mut krishiv_exec::LocalState,
        _global: &'a krishiv_exec::GlobalState,
    ) -> futures::future::BoxFuture<'a, krishiv_exec::Result<Option<krishiv_types::batch::Batch>>>
    {
        Box::pin(std::future::ready(Ok(None)))
    }
}
