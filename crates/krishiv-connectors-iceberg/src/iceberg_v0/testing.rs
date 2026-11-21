// SPDX-License-Identifier: Apache-2.0
//! Test tables for other crates (feature `testing`): a small Iceberg table written into a local
//! directory (or, `write_table_with`, any store a path locator places) by iceberg-rust's in-memory
//! catalog over Krishiv's I/O, with Parquet data files, so a catalog test can serve its metadata and
//! read it back through [`crate::IcebergTableProvider`].

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
use krishiv_exec::files::{LocalPaths, PathLocator};
use krishiv_io::{FIELD_ID_KEY, IoConfig, IoContext, IoScheduler, StoreId};

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
    let root = dir.to_string_lossy().into_owned();
    write_table_with(&format!("file://{}", dir.display()), move |sched| {
        let id = StoreId::new("file://");
        sched.register_store(
            id.clone(),
            Arc::new(object_store::local::LocalFileSystem::new()),
        );
        Arc::new(LocalPaths::new(id, root))
    })
}

/// [`write_table`]'s table under `warehouse` (a URL such as `s3://bucket/wh`), every byte written
/// through the stores of the locator `paths` builds on the writer's own I/O scheduler (where it
/// registers them): metadata, manifests and the Parquet data files under `<warehouse>/db/t/data/`.
///
/// # Errors
/// As [`write_table`].
pub fn write_table_with(
    warehouse: &str,
    paths: impl FnOnce(&Arc<IoScheduler>) -> Arc<dyn PathLocator>,
) -> Result<WrittenTable, KrishivError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| internal(format!("a runtime: {e}")))?;
    let sched = IoScheduler::with_runtime(IoConfig::default(), rt.handle().clone());
    let paths = paths(&sched);
    let io = sched.context();
    let reg = {
        let _entered = rt.enter();
        StorageRegistration::new(io.clone(), Arc::clone(&paths))
    };
    let files = Files {
        io,
        paths: paths.as_ref(),
    };
    rt.block_on(write(&reg, &files, warehouse))
}

/// Where [`write`] puts the data files and reads the metadata back from.
struct Files<'a> {
    io: IoContext,
    paths: &'a dyn PathLocator,
}

impl Files<'_> {
    async fn put(&self, url: &str, bytes: Vec<u8>) -> Result<(), KrishivError> {
        let at = self.paths.locate(url)?;
        self.io.put_object(&at.store, &at.path, bytes.into()).await
    }

    async fn get(&self, url: &str) -> Result<Vec<u8>, KrishivError> {
        let at = self.paths.locate(url)?;
        self.io
            .get_object(&at.store, &at.path)
            .await?
            .map(|b| b.to_vec())
            .ok_or_else(|| internal(format!("{url} is missing")))
    }
}

async fn write(
    reg: &StorageRegistration,
    files: &Files<'_>,
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
    let data = format!("{}/data", table.metadata().location().trim_end_matches('/'));
    let mut snapshots = Vec::new();
    for (n, rows) in [vec![(1, "a"), (2, "b")], vec![(3, "c")]]
        .into_iter()
        .enumerate()
    {
        let file = format!("{data}/{n}.parquet");
        let bytes = parquet_bytes(&rows)?;
        let size = bytes.len() as u64;
        files.put(&file, bytes).await?;
        let entry = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(file)
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
    let text = files.get(&metadata_location).await?;
    let metadata = serde_json::from_slice(&text)
        .map_err(|e| internal(format!("the metadata file's JSON: {e}")))?;
    Ok(WrittenTable {
        metadata_location,
        metadata,
        location: table.metadata().location().to_owned(),
        snapshots,
    })
}

/// `rows` as one Parquet file with the table's field ids.
fn parquet_bytes(rows: &[(i64, &str)]) -> Result<Vec<u8>, KrishivError> {
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
    let mut out = Vec::new();
    let mut w = parquet::arrow::ArrowWriter::try_new(&mut out, schema, None)
        .map_err(|e| internal(format!("a Parquet writer: {e}")))?;
    w.write(&batch)
        .map_err(|e| internal(format!("a Parquet write: {e}")))?;
    w.close()
        .map_err(|e| internal(format!("a Parquet close: {e}")))?;
    Ok(out)
}
