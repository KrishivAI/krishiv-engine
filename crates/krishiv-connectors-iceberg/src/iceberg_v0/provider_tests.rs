// SPDX-License-Identifier: Apache-2.0
//! Tests of [`IcebergTableProvider`] through SQL (analyzer, optimizer, planner
//! and executor) over tables written by iceberg-rust's in-memory catalog with
//! Parquet files written here: identity partitions from the manifest, files
//! without field ids, delete files refused, time travel, and the 22 TPC-H
//! queries against the same Parquet files read as a plain file table.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Decimal64Type, Decimal128Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as A, Field, Schema, SchemaRef};
use futures::StreamExt;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, NestedField,
    PartitionSpec, PrimitiveType, Struct, Transform as IceTransform, Type,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::physical::WorkerContext;
use krishiv_exec::planner::{
    DefaultPlanner, MemoryTables, PhysicalPlanner, PlanningContext, TableProvider,
};
use krishiv_exec::{
    ExecConfig, ExecutorConfig, QueryContext, QueryId, QueryRunner, Scheduler as _, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::{FIELD_ID_KEY, FileFormat as _};
use krishiv_memory::{MemoryConfig, MemoryPool as _, TreePool};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_types::StructType;

use super::provider::IcebergTableProvider;
use super::storage::{KrishivStorageFactory, StorageRegistration};
use super::table::IcebergTable;
use super::tests::local;
use crate::plan::VersionSpec;

/// A runtime for the I/O side, a registration over the local file system and
/// iceberg-rust's in-memory catalog with namespace `db`.
pub(super) struct Env {
    pub(super) rt: tokio::runtime::Runtime,
    pub(super) dir: tempfile::TempDir,
    pub(super) _sched: Arc<krishiv_io::IoScheduler>,
    pub(super) reg: StorageRegistration,
    pub(super) catalog: iceberg::memory::MemoryCatalog,
}

pub(super) fn env() -> Env {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (sched, reg) = {
        let _entered = rt.enter();
        local(dir.path())
    };
    let factory = Arc::new(KrishivStorageFactory {
        name: reg.name().to_owned(),
    });
    let catalog = rt.block_on(async {
        let catalog = iceberg::memory::MemoryCatalogBuilder::default()
            .with_storage_factory(factory)
            .with_runtime(reg.runtime().unwrap())
            .load(
                "mem",
                [(
                    iceberg::memory::MEMORY_CATALOG_WAREHOUSE.to_owned(),
                    format!("file://{}", dir.path().display()),
                )]
                .into(),
            )
            .await
            .unwrap();
        catalog
            .create_namespace(&NamespaceIdent::new("db".to_owned()), HashMap::new())
            .await
            .unwrap();
        catalog
    });
    Env {
        rt,
        dir,
        _sched: sched,
        reg,
        catalog,
    }
}

impl Env {
    /// `file://<dir>/<name>`.
    pub(super) fn at(&self, name: &str) -> String {
        format!("file://{}/{name}", self.dir.path().display())
    }

    pub(super) fn create(
        &self,
        name: &str,
        schema: iceberg::spec::Schema,
        spec: Option<PartitionSpec>,
    ) -> iceberg::table::Table {
        let creation = match spec {
            Some(spec) => TableCreation::builder()
                .name(name.to_owned())
                .schema(schema)
                .partition_spec(spec.into_unbound())
                .build(),
            None => TableCreation::builder()
                .name(name.to_owned())
                .schema(schema)
                .build(),
        };
        self.rt.block_on(async {
            self.catalog
                .create_table(&NamespaceIdent::new("db".to_owned()), creation)
                .await
                .unwrap()
        })
    }

    pub(super) fn append(
        &self,
        table: &iceberg::table::Table,
        files: Vec<DataFile>,
    ) -> iceberg::table::Table {
        self.rt.block_on(async {
            let tx = Transaction::new(table);
            let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
            tx.commit(&self.catalog).await.unwrap()
        })
    }

    pub(super) fn load(&self, table: &iceberg::table::Table) -> IcebergTable {
        let location = table.metadata_location().unwrap().to_owned();
        self.rt
            .block_on(IcebergTable::load(&self.reg, &location, "db", "t"))
            .unwrap()
    }

    pub(super) fn provider(
        &self,
        table: &IcebergTable,
        version: &VersionSpec,
    ) -> Result<IcebergTableProvider, KrishivError> {
        self.rt.block_on(IcebergTableProvider::open(
            table,
            version,
            &Config::new(),
            4,
        ))
    }
}

/// `batch` written as Parquet at `path` (a local path); its size.
pub(super) fn write_parquet(path: &str, batch: &RecordBatch) -> u64 {
    let local = path.strip_prefix("file://").unwrap();
    let file = std::fs::File::create(local).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    w.write(batch).unwrap();
    w.close().unwrap();
    std::fs::metadata(local).unwrap().len()
}

/// A field with an Iceberg id.
pub(super) fn id_field(name: &str, dt: A, id: i32) -> Field {
    Field::new(name, dt, true)
        .with_metadata(HashMap::from([(FIELD_ID_KEY.to_owned(), id.to_string())]))
}

pub(super) fn data_file(path: &str, size: u64, rows: u64, partition: Struct) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_owned())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(rows)
        .partition(partition)
        .partition_spec_id(0)
        .build()
        .unwrap()
}

/// Tables by name, each with the columns the analyzer resolves.
pub(super) struct Session {
    pub(super) ctx: krishiv_plan::analyzer::AnalysisContext,
    pub(super) tables: Arc<MemoryTables>,
    pub(super) config: Arc<Config>,
}

pub(super) fn session(tables: Vec<(&str, StructType, Arc<dyn TableProvider>)>) -> Session {
    let mut config = Config::new();
    config.set("spark.sql.ansi.enabled", "true").unwrap();
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut providers = MemoryTables::new();
    for (name, schema, provider) in tables {
        let columns = schema
            .fields()
            .iter()
            .map(|f| TableColumn {
                name: f.name().to_owned(),
                data_type: f.data_type().clone(),
                nullable: f.nullable(),
                char_varchar: None,
            })
            .collect();
        catalog = catalog.with_table(&["default"], name, columns, None);
        providers = providers.with_table(&["spark_catalog", "default", name], provider);
    }
    let conf = ResolutionConf {
        ansi: true,
        ..ResolutionConf::default()
    };
    Session {
        ctx: context(catalog, conf),
        tables: Arc::new(providers),
        config: Arc::new(config),
    }
}

impl Session {
    /// Runs `sql` (joins in the optimizer's default order) to completion.
    pub(super) fn run(&self, sql: &str) -> Result<Vec<RecordBatch>, KrishivError> {
        let plan =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .map_err(KrishivError::from)?;
        let plan = block_on(DefaultAnalyzer.analyze(&plan, &self.ctx))?.plan;
        let plan = krishiv_optimizer::optimize(&plan)?;
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::clone(&self.config));
        let physical = DefaultPlanner.plan(&plan, &cx)?;
        let worker = WorkerContext::new(Arc::clone(&self.tables) as _);
        let root = TreePool::new(&MemoryConfig::default());
        let query = QueryContext::new(
            QueryId(1),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            root.child("q", None),
        );
        let runner = QueryRunner::new(TokioExecutor::new(&ExecutorConfig::default())?);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| {
                KrishivError::from(krishiv_common::error::InternalError {
                    message: e.to_string(),
                })
            })?;
        rt.block_on(async {
            let mut stream = runner.submit(physical, worker, Arc::clone(&query)).await?;
            let mut out = Vec::new();
            while let Some(b) = stream.next().await {
                out.push(b?.into_parts().0);
            }
            query.finished().await;
            Ok(out)
        })
    }

    /// [`Self::run`]'s rows rendered as `a|b|c`.
    pub(super) fn rows(&self, sql: &str) -> Vec<String> {
        render(&self.run(sql).unwrap_or_else(|e| panic!("{sql}: {e}")))
    }
}

/// One value as text (`NULL` for null).
pub(super) fn cell(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        A::Int32 => a.as_primitive::<Int32Type>().value(row).to_string(),
        A::Int64 => a.as_primitive::<Int64Type>().value(row).to_string(),
        A::Date32 => a.as_primitive::<Date32Type>().value(row).to_string(),
        A::Decimal64(_, s) => format!("{}e-{s}", a.as_primitive::<Decimal64Type>().value(row)),
        A::Decimal128(_, s) => format!("{}e-{s}", a.as_primitive::<Decimal128Type>().value(row)),
        A::Utf8View => a.as_string_view().value(row).to_owned(),
        A::Utf8 => a.as_string::<i32>().value(row).to_owned(),
        A::Float64 => a
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(row)
            .to_string(),
        other => panic!("no rendering for {other}"),
    }
}

pub(super) fn render(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            out.push(cells.join("|"));
        }
    }
    out
}

fn int64s(v: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn strings(v: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(v.to_vec()))
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// `id long (1), x int (2), name string (3)`.
fn ixn_schema() -> iceberg::spec::Schema {
    iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "x", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

fn provider_session(p: IcebergTableProvider) -> Session {
    let schema = p.snapshot().schema.clone();
    session(vec![("t", schema, Arc::new(p))])
}

/// FC S4: identity-partition values come from the manifest; a column the file
/// lacks is filled from it, and a column the file has with another value is
/// read as the manifest says. A bucket field beside it reads from the file.
#[test]
fn identity_partitions_come_from_the_manifest() {
    let env = env();
    let schema = ixn_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("id", "id_bucket", IceTransform::Bucket(4))
        .unwrap()
        .add_partition_field("x", "x", IceTransform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let t = env.create("t", schema, Some(spec));
    let bucket = |id: i64| {
        let v = crate::plan::apply_transform(
            crate::plan::Transform::Bucket(4),
            &crate::plan::IcebergType::Long,
            &krishiv_io::ScalarValue::Int64(id),
        )
        .unwrap();
        match v {
            krishiv_io::ScalarValue::Int32(b) => Literal::int(b),
            other => panic!("{other:?}"),
        }
    };
    // a: no `x` column (a Hive-layout file); b: `x` = 99 in the file, 8 in the
    // manifest; c: a null partition value.
    let a = batch(
        vec![id_field("id", A::Int64, 1), id_field("name", A::Utf8, 3)],
        vec![int64s(&[1, 1]), strings(&["a1", "a2"])],
    );
    let b = batch(
        vec![
            id_field("id", A::Int64, 1),
            id_field("x", A::Int32, 2),
            id_field("name", A::Utf8, 3),
        ],
        vec![
            int64s(&[2]),
            Arc::new(Int32Array::from(vec![99])),
            strings(&["b"]),
        ],
    );
    let c = batch(
        vec![
            id_field("id", A::Int64, 1),
            id_field("x", A::Int32, 2),
            id_field("name", A::Utf8, 3),
        ],
        vec![
            int64s(&[3]),
            Arc::new(Int32Array::from(vec![5])),
            strings(&["c"]),
        ],
    );
    let mut files = Vec::new();
    for (name, batch, id, x) in [
        ("a.parquet", &a, 1, Some(7)),
        ("b.parquet", &b, 2, Some(8)),
        ("c.parquet", &c, 3, None),
    ] {
        let path = env.at(name);
        let size = write_parquet(&path, batch);
        let partition = Struct::from_iter([Some(bucket(id)), x.map(Literal::int)]);
        files.push(data_file(&path, size, batch.num_rows() as u64, partition));
    }
    let t = env.append(&t, files);
    let table = env.load(&t);
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    assert_eq!(p.constant_columns(), ["x"]);
    // Pruning by the manifest's partition values: `x = 8` opens only b, `x = 99`
    // nothing (b's file says 99, its manifest 8), a bucket of `id` the files in its bucket.
    let x = |v| {
        krishiv_io::PushdownExpr::cmp(
            krishiv_io::ColumnPath::new("x"),
            krishiv_io::CmpOp::Eq,
            krishiv_io::ScalarValue::Int32(v),
        )
    };
    assert_eq!(p.files_read(x(8)), 1);
    assert_eq!(p.files_read(x(99)), 0);
    assert_eq!(
        p.files_read(krishiv_io::PushdownExpr::IsNull(
            krishiv_io::ColumnPath::new("x")
        )),
        1
    );
    assert_eq!(
        p.files_read(krishiv_io::PushdownExpr::cmp(
            krishiv_io::ColumnPath::new("id"),
            krishiv_io::CmpOp::Eq,
            krishiv_io::ScalarValue::Int64(1),
        )),
        // The files whose bucket is 1's (bucket 4 of 1, 2 and 3 may collide).
        [1, 2, 3]
            .iter()
            .filter(|i| bucket(**i) == bucket(1))
            .count()
    );
    let s = provider_session(p);
    assert_eq!(
        s.rows("SELECT id, x, name FROM t ORDER BY name"),
        ["1|7|a1", "1|7|a2", "2|8|b", "3|NULL|c"]
    );
    // Filters on the constant column and on the bucketed one prune files and
    // are still applied to the rows.
    assert_eq!(s.rows("SELECT name FROM t WHERE x = 8"), ["b"]);
    assert_eq!(
        s.rows("SELECT name FROM t WHERE x = 99"),
        Vec::<String>::new()
    );
    assert_eq!(s.rows("SELECT name FROM t WHERE x IS NULL"), ["c"]);
    assert_eq!(
        s.rows("SELECT name FROM t WHERE id = 1 ORDER BY name"),
        ["a1", "a2"]
    );
    assert_eq!(s.rows("SELECT count(*) FROM t"), ["4"]);
}

/// FC S4: a file with no field ids reads its top-level columns as ids 1..n by
/// position (names do not matter); with a name mapping it is refused until
/// PR-25c reads the mapping.
#[test]
fn a_file_without_field_ids_reads_by_position() {
    let env = env();
    let t = env.create("t", ixn_schema(), None);
    let plain = batch(
        vec![
            Field::new("c0", A::Int64, true),
            Field::new("c1", A::Int32, true),
            Field::new("c2", A::Utf8, true),
        ],
        vec![
            int64s(&[10]),
            Arc::new(Int32Array::from(vec![20])),
            strings(&["thirty"]),
        ],
    );
    let path = env.at("plain.parquet");
    let size = write_parquet(&path, &plain);
    let t = env.append(&t, vec![data_file(&path, size, 1, Struct::empty())]);
    let table = env.load(&t);
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    assert!(p.constant_columns().is_empty());
    let s = provider_session(p);
    assert_eq!(s.rows("SELECT name, id, x FROM t"), ["thirty|10|20"]);

    // The same table with a name mapping: refused, naming the file.
    let t = env.rt.block_on(async {
        let tx = Transaction::new(&t);
        let tx = tx
            .update_table_properties()
            .set(
                "schema.name-mapping.default".to_owned(),
                r#"[{"field-id": 1, "names": ["c0"]}]"#.to_owned(),
            )
            .apply(tx)
            .unwrap();
        tx.commit(&env.catalog).await.unwrap()
    });
    let table = env.load(&t);
    let s = provider_session(env.provider(&table, &VersionSpec::Current).unwrap());
    let e = s.run("SELECT id FROM t").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.message().contains("plain.parquet"), "{e}");
}

/// Time travel through the provider reads the older snapshot's files with its
/// own schema; the current version reads the added column as null in the old
/// file.
#[test]
fn time_travel_reads_the_older_snapshot_and_schema() {
    let env = env();
    let schema = iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let t = env.create("t", schema, None);
    let old = batch(
        vec![id_field("id", A::Int64, 1), id_field("name", A::Utf8, 2)],
        vec![int64s(&[1]), strings(&["old"])],
    );
    let path = env.at("old.parquet");
    let size = write_parquet(&path, &old);
    let t = env.append(&t, vec![data_file(&path, size, 1, Struct::empty())]);
    let first = t.metadata().current_snapshot().unwrap().snapshot_id();
    // Add `extra int` (id 3), then a file that has it; rename `name` to `label`.
    let t = env.rt.block_on(async {
        let tx = Transaction::new(&t);
        let tx = tx
            .update_schema()
            .add_column(iceberg::transaction::AddColumn::optional(
                "extra",
                Type::Primitive(PrimitiveType::Int),
            ))
            .apply(tx)
            .unwrap();
        tx.commit(&env.catalog).await.unwrap()
    });
    let new = batch(
        vec![
            id_field("id", A::Int64, 1),
            id_field("name", A::Utf8, 2),
            id_field("extra", A::Int32, 3),
        ],
        vec![
            int64s(&[2]),
            strings(&["new"]),
            Arc::new(Int32Array::from(vec![42])),
        ],
    );
    let path = env.at("new.parquet");
    let size = write_parquet(&path, &new);
    let t = env.append(&t, vec![data_file(&path, size, 1, Struct::empty())]);
    let table = env.load(&t);

    let current = env.provider(&table, &VersionSpec::Current).unwrap();
    assert_eq!(current.files().len(), 2);
    let s = provider_session(current);
    assert_eq!(
        s.rows("SELECT * FROM t ORDER BY id"),
        ["1|old|NULL", "2|new|42"]
    );
    let older = env.provider(&table, &VersionSpec::Id(first)).unwrap();
    assert_eq!(older.files().len(), 1);
    assert_eq!(older.snapshot().schema.fields().len(), 2);
    let s = provider_session(older);
    assert_eq!(s.rows("SELECT * FROM t"), ["1|old"]);
    // An unknown snapshot: Iceberg's text, Krishiv's user-facing class.
    let e = env.provider(&table, &VersionSpec::Id(7)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivTimeTravelNotFound);
    assert_eq!(e.message(), "Cannot find snapshot with ID 7");
}

/// A snapshot with a live delete file is refused (deletes are PR-25b), naming
/// the file: never read without its deletes. The snapshot is written by hand:
/// a data manifest, a position-delete manifest and a manifest list, added to
/// the metadata as a new current snapshot.
#[test]
fn a_snapshot_with_delete_files_is_refused() {
    use iceberg::spec::{ManifestListWriter, ManifestWriterBuilder};
    let env = env();
    let schema = ixn_schema();
    let t = env.create("t", schema.clone(), None);
    let rows = batch(
        vec![
            id_field("id", A::Int64, 1),
            id_field("x", A::Int32, 2),
            id_field("name", A::Utf8, 3),
        ],
        vec![
            int64s(&[1]),
            Arc::new(Int32Array::from(vec![1])),
            strings(&["a"]),
        ],
    );
    let data = env.at("data.parquet");
    let size = write_parquet(&data, &rows);
    let t = env.append(&t, vec![data_file(&data, size, 1, Struct::empty())]);
    let base = Arc::clone(t.metadata().current_snapshot().unwrap());
    let io = env.reg.file_io();
    let schema = Arc::new(schema);
    let spec = PartitionSpec::unpartition_spec();
    let snapshot_id = base.snapshot_id() + 1;
    let seq = base.sequence_number() + 1;
    let list = env.at("snap-deletes.avro");
    let deletes = env.at("deletes.parquet");
    env.rt.block_on(async {
        let mut w = ManifestWriterBuilder::new(
            io.new_output(env.at("data-m.avro")).unwrap(),
            Some(snapshot_id),
            Arc::clone(&schema),
            spec.clone(),
        )
        .build_v2_data();
        w.add_file(data_file(&data, size, 1, Struct::empty()), seq - 1)
            .unwrap();
        let data_manifest = w.write_manifest_file().await.unwrap();
        let mut w = ManifestWriterBuilder::new(
            io.new_output(env.at("delete-m.avro")).unwrap(),
            Some(snapshot_id),
            Arc::clone(&schema),
            spec,
        )
        .build_v2_deletes();
        let delete = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(deletes.clone())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(10)
            .record_count(1)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();
        w.add_file(delete, seq).unwrap();
        let delete_manifest = w.write_manifest_file().await.unwrap();
        let mut lw = ManifestListWriter::v2(
            io.new_output(&list).unwrap().writer().await.unwrap(),
            snapshot_id,
            Some(base.snapshot_id()),
            seq,
        );
        lw.add_manifests([data_manifest, delete_manifest].into_iter())
            .unwrap();
        lw.close().await.unwrap();
    });
    // The metadata with the new snapshot current.
    let location = t.metadata_location().unwrap().to_owned();
    let text = env
        .rt
        .block_on(async { io.new_input(&location).unwrap().read().await.unwrap() });
    let mut json: serde_json::Value = serde_json::from_slice(&text).unwrap();
    let ts = base.timestamp_ms() + 1;
    json["snapshots"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({
            "snapshot-id": snapshot_id,
            "parent-snapshot-id": base.snapshot_id(),
            "sequence-number": seq,
            "timestamp-ms": ts,
            "manifest-list": list,
            "summary": {"operation": "overwrite"},
            "schema-id": 0
        }));
    json["current-snapshot-id"] = serde_json::json!(snapshot_id);
    json["last-sequence-number"] = serde_json::json!(seq);
    json["refs"]["main"] = serde_json::json!({"snapshot-id": snapshot_id, "type": "branch"});
    json["snapshot-log"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"snapshot-id": snapshot_id, "timestamp-ms": ts}));
    let edited = env.at("99-deletes.metadata.json");
    env.rt.block_on(async {
        io.new_output(&edited)
            .unwrap()
            .write(bytes::Bytes::from(serde_json::to_vec(&json).unwrap()))
            .await
            .unwrap();
    });
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, &edited, "db", "t"))
        .unwrap();
    let e = env.provider(&table, &VersionSpec::Current).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.message().contains("deletes.parquet"), "{e}");
    // The snapshot before it has no deletes and reads.
    let p = env
        .provider(&table, &VersionSpec::Id(base.snapshot_id()))
        .unwrap();
    assert_eq!(provider_session(p).rows("SELECT name FROM t"), ["a"]);
}

/// The Iceberg type of a TPC-H column as the Parquet file stores it.
fn iceberg_type(dt: &A) -> PrimitiveType {
    match dt {
        A::Int32 => PrimitiveType::Int,
        A::Int64 => PrimitiveType::Long,
        A::Float64 => PrimitiveType::Double,
        A::Date32 => PrimitiveType::Date,
        A::Utf8 | A::Utf8View | A::LargeUtf8 => PrimitiveType::String,
        A::Decimal32(p, s) | A::Decimal64(p, s) | A::Decimal128(p, s) => PrimitiveType::Decimal {
            precision: u32::from(*p),
            scale: u32::try_from(*s).unwrap(),
        },
        other => panic!("no Iceberg type for {other}"),
    }
}

/// The 22 TPC-H queries over Iceberg tables registered on the SF 0.01 Parquet
/// files (DuckDB's dbgen, `krishiv-exec/tests/fixtures/tpch-sf0.01-parquet`)
/// return what the same files return read as a plain Parquet file table. The
/// files carry no field ids, so the Iceberg read maps them by position (FC S4).
#[test]
fn tpch_over_iceberg_equals_tpch_over_parquet() {
    const TABLES: [&str; 8] = [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let data = root
        .join("crates/krishiv-exec/tests/fixtures/tpch-sf0.01-parquet")
        .canonicalize()
        .unwrap();
    let env = env();
    let io = env.reg.registered().unwrap().io;
    let format: Arc<dyn krishiv_io::FileFormat> = Arc::new(krishiv_io::parquet::ParquetFormat);
    let options = krishiv_io::FormatOptions::defaults(krishiv_io::parquet::ParquetFormat.options());
    let mut parquet_tables = Vec::new();
    let mut iceberg_tables = Vec::new();
    let mut without_ids = 0;
    for name in TABLES {
        let path = data.join(format!("{name}.parquet"));
        let location = format!("file://{}", path.display());
        let at = env
            .reg
            .registered()
            .unwrap()
            .paths
            .locate(&location)
            .unwrap();
        let (file, file_schema): (krishiv_io::FileRef, SchemaRef) = env.rt.block_on(async {
            let meta = io.head(&at.store, &at.path).await.unwrap().unwrap();
            let file = krishiv_io::FileRef::from_meta(at.store.clone(), &meta);
            let m = io
                .read_metadata(
                    format.as_ref(),
                    &file,
                    krishiv_io::MetadataNeeds::default(),
                    &options,
                )
                .await
                .unwrap();
            (file, m.schema())
        });
        // The plain table: typed as Spark reads the files.
        let mut columns = Vec::new();
        let mut arrow = Vec::new();
        let mut nested = Vec::new();
        for (i, f) in file_schema.fields().iter().enumerate() {
            let dt = krishiv_types::DataType::from_arrow_field(f).unwrap();
            arrow.push(dt.to_arrow_field(f.name(), f.is_nullable()));
            columns.push(krishiv_types::StructField::new(
                f.name().clone(),
                dt,
                f.is_nullable(),
            ));
            let id = f
                .metadata()
                .get(FIELD_ID_KEY)
                .map_or(i32::try_from(i).unwrap() + 1, |v| v.parse().unwrap());
            if !f.metadata().contains_key(FIELD_ID_KEY) {
                without_ids += 1;
            }
            let ty = Type::Primitive(iceberg_type(f.data_type()));
            nested.push(if f.is_nullable() {
                NestedField::optional(id, f.name(), ty).into()
            } else {
                NestedField::required(id, f.name(), ty).into()
            });
        }
        let rows = env.rt.block_on(async {
            io.read_metadata(
                format.as_ref(),
                &file,
                krishiv_io::MetadataNeeds::default(),
                &options,
            )
            .await
            .unwrap()
            .statistics()
            .rows
            .exact()
            .copied()
            .unwrap()
        });
        let plain = env
            .rt
            .block_on(krishiv_exec::planner::FileTable::open(
                Arc::clone(&format),
                io.clone(),
                Arc::new(Schema::new(arrow)),
                vec![file.clone()],
                options.clone(),
                &Config::new(),
                4,
            ))
            .unwrap();
        parquet_tables.push((
            name,
            StructType::new(columns),
            Arc::new(plain) as Arc<dyn TableProvider>,
        ));

        let schema = iceberg::spec::Schema::builder()
            .with_fields(nested)
            .build()
            .unwrap();
        let t = env.rt.block_on(async {
            let creation = TableCreation::builder()
                .name(name.to_owned())
                .schema(schema)
                .build();
            env.catalog
                .create_table(&NamespaceIdent::new("db".to_owned()), creation)
                .await
                .unwrap()
        });
        let t = env.append(
            &t,
            vec![data_file(&location, file.size, rows, Struct::empty())],
        );
        let table = env.load(&t);
        let p = env.provider(&table, &VersionSpec::Current).unwrap();
        iceberg_tables.push((
            name,
            p.snapshot().schema.clone(),
            Arc::new(p) as Arc<dyn TableProvider>,
        ));
    }
    assert!(without_ids > 0, "the fixture files carry no field ids");
    let plain = session(parquet_tables);
    let iceberg = session(iceberg_tables);
    let mut failed = Vec::new();
    let mut rows_seen = 0;
    for q in 1..=22 {
        let sql =
            std::fs::read_to_string(root.join(format!("bench/queries/tpch/q{q:02}.sql"))).unwrap();
        let want = plain
            .run(&sql)
            .unwrap_or_else(|e| panic!("q{q:02} parquet: {e}"));
        let got = match iceberg.run(&sql) {
            Ok(b) => b,
            Err(e) => {
                failed.push(format!("q{q:02}: {e}"));
                continue;
            }
        };
        let cat = |b: &[RecordBatch]| -> Vec<ArrayRef> {
            let Some(first) = b.first() else {
                return Vec::new();
            };
            let schema = first.schema();
            let all = arrow_select::concat::concat_batches(&schema, b).unwrap();
            all.columns().to_vec()
        };
        let (w, g) = (cat(&want), cat(&got));
        rows_seen += w.first().map_or(0, Array::len);
        if w != g {
            failed.push(format!(
                "q{q:02}: {} rows vs {} over Parquet",
                g.first().map_or(0, Array::len),
                w.first().map_or(0, Array::len)
            ));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    // The answers have 932 rows at SF 0.01 (Spark 4.2's, see
    // `tpch-sf0.01-parquet/spark-answers.json`).
    assert_eq!(rows_seen, 932);
}
