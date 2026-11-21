// SPDX-License-Identifier: Apache-2.0
//! Tests of the storage adapter, the bridged `inspect` batches and the type mapping.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use krishiv_exec::files::LocalPaths;
use krishiv_io::{IoConfig, IoScheduler, StoreId};
use object_store::ObjectStore;

use super::storage::{KrishivStorageFactory, StorageRegistration, from_iceberg};

/// A scheduler with the local file system registered as `file://` and a
/// registration whose relative paths resolve under `dir`.
pub(super) fn local(dir: &std::path::Path) -> (Arc<IoScheduler>, StorageRegistration) {
    let store = Arc::new(object_store::local::LocalFileSystem::new());
    local_on(dir, store, IoConfig::default())
}

/// [`local`] with `store` registered as `file://` (a wrapper of the local
/// file system) and the scheduler's settings `config`.
pub(super) fn local_on(
    dir: &std::path::Path,
    store: Arc<dyn ObjectStore>,
    config: IoConfig,
) -> (Arc<IoScheduler>, StorageRegistration) {
    // The I/O runtime is the test's own (iceberg-rust's `Runtime` needs one).
    let sched = match tokio::runtime::Handle::try_current() {
        Ok(h) => IoScheduler::with_runtime(config, h),
        Err(_) => IoScheduler::new(config),
    };
    let id = StoreId::new("file://");
    sched.register_store(id.clone(), store);
    let paths = Arc::new(LocalPaths::new(id, dir.to_string_lossy().into_owned()));
    let reg = StorageRegistration::new(sched.context(), paths);
    (sched, reg)
}

fn requests(sched: &IoScheduler) -> u64 {
    sched.totals().requests.load(Ordering::Relaxed)
}

#[tokio::test]
async fn storage_reads_and_writes_through_krishiv_io() {
    let dir = tempfile::tempdir().unwrap();
    let (sched, reg) = local(dir.path());
    let io = reg.file_io();
    let loc = format!("file://{}/m/v1.metadata.json", dir.path().display());

    let before = requests(&sched);
    let out = io.new_output(&loc).unwrap();
    out.write(bytes::Bytes::from_static(b"{\"a\":1}"))
        .await
        .unwrap();
    let input = io.new_input(&loc).unwrap();
    assert!(input.exists().await.unwrap());
    assert_eq!(input.metadata().await.unwrap().size, 7);
    assert_eq!(&input.read().await.unwrap()[..], b"{\"a\":1}");
    let reader = input.reader().await.unwrap();
    assert_eq!(&reader.read(2..5).await.unwrap()[..], b"a\":");
    // Every call above was a request of the Krishiv scheduler.
    assert!(
        requests(&sched) - before >= 5,
        "{}",
        requests(&sched) - before
    );

    let mut w = io.new_output(&loc).unwrap().writer().await.unwrap();
    w.write(bytes::Bytes::from_static(b"xy")).await.unwrap();
    w.write(bytes::Bytes::from_static(b"z")).await.unwrap();
    assert_eq!(w.close().await.unwrap().size, 3);
    assert_eq!(
        &io.new_input(&loc).unwrap().read().await.unwrap()[..],
        b"xyz"
    );

    io.delete_prefix(format!("file://{}/m", dir.path().display()))
        .await
        .unwrap();
    assert!(!io.exists(&loc).await.unwrap());
}

#[tokio::test]
async fn missing_file_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let io = reg.file_io();
    let loc = format!("{}/none.json", dir.path().display());
    let input = io.new_input(&loc).unwrap();
    assert!(!input.exists().await.unwrap());
    let e = input.read().await.unwrap_err();
    assert_eq!(e.kind(), iceberg::ErrorKind::DataInvalid);
    assert!(e.message().contains("File not found"), "{e}");
}

#[tokio::test]
async fn file_io_round_trips_through_serde_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let bytes = reg.file_io().serialize_all().unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(text.contains(reg.name()), "{text}");
    let io = iceberg::io::FileIO::deserialize_all(&bytes).unwrap();
    let loc = format!("{}/a.json", dir.path().display());
    io.new_output(&loc)
        .unwrap()
        .write(bytes::Bytes::from_static(b"1"))
        .await
        .unwrap();
    assert!(io.exists(&loc).await.unwrap());
}

#[test]
fn unknown_store_name_fails_cleanly() {
    let json = r#"{"config":{"props":{}},"factory":{"type":"krishiv","name":"krishiv-nope"}}"#;
    let io = iceberg::io::FileIO::deserialize_all(json.as_bytes()).unwrap();
    let e = io.new_input("/tmp/x").unwrap_err();
    assert_eq!(e.kind(), iceberg::ErrorKind::PreconditionFailed);
    assert!(e.message().contains("krishiv-nope"), "{e}");
    // A dropped registration is gone too.
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let name = reg.name().to_owned();
    drop(reg);
    let factory: &dyn iceberg::io::StorageFactory = &KrishivStorageFactory { name };
    assert!(factory.build(&iceberg::io::StorageConfig::new()).is_err());
}

#[test]
fn krishiv_errors_keep_their_class_through_iceberg() {
    let k: krishiv_common::error::KrishivError = krishiv_common::error::InternalError {
        message: "boom".to_owned(),
    }
    .into();
    let e = iceberg::Error::new(iceberg::ErrorKind::Unexpected, "x").with_source(k.clone());
    assert_eq!(from_iceberg(e), k);
    let plain = from_iceberg(iceberg::Error::new(iceberg::ErrorKind::DataInvalid, "bad"));
    assert!(plain.to_string().contains("iceberg-rust"), "{plain}");
}

fn test_schema() -> iceberg::spec::Schema {
    use iceberg::spec::{ListType, MapType, NestedField, PrimitiveType, Type};
    iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                .with_doc("the name")
                .into(),
            NestedField::optional(
                3,
                "tags",
                Type::List(ListType::new(
                    NestedField::list_element(4, Type::Primitive(PrimitiveType::String), true)
                        .into(),
                )),
            )
            .into(),
            NestedField::optional(
                5,
                "attrs",
                Type::Map(MapType::new(
                    NestedField::map_key_element(6, Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::map_value_element(
                        7,
                        Type::Primitive(PrimitiveType::Double),
                        false,
                    )
                    .into(),
                )),
            )
            .into(),
            NestedField::optional(8, "at", Type::Primitive(PrimitiveType::Timestamptz)).into(),
        ])
        .build()
        .unwrap()
}

/// A table created and appended to by iceberg-rust's in-memory catalog over the
/// Krishiv storage (so its metadata, manifest list and manifest are real files
/// written through `krishiv-io`); returns its metadata location.
async fn make_table(reg: &StorageRegistration, warehouse: &std::path::Path) -> (String, i64) {
    use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat, Struct};
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

    let factory = Arc::new(KrishivStorageFactory {
        name: reg.name().to_owned(),
    });
    let catalog = iceberg::memory::MemoryCatalogBuilder::default()
        .with_storage_factory(factory)
        .with_runtime(reg.runtime().unwrap())
        .load(
            "mem",
            [(
                iceberg::memory::MEMORY_CATALOG_WAREHOUSE.to_owned(),
                format!("file://{}", warehouse.display()),
            )]
            .into(),
        )
        .await
        .unwrap();
    let ns = NamespaceIdent::new("db".to_owned());
    catalog
        .create_namespace(&ns, Default::default())
        .await
        .unwrap();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("t".to_owned())
                .schema(test_schema())
                .build(),
        )
        .await
        .unwrap();
    let file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!(
            "file://{}/db/t/data/0.parquet",
            warehouse.display()
        ))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(10)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .build()
        .unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let snapshot = table.metadata().current_snapshot().unwrap().snapshot_id();
    (table.metadata_location().unwrap().to_owned(), snapshot)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_maps_schema_and_bridges_inspect_tables() {
    use arrow_array::Array;

    let dir = tempfile::tempdir().unwrap();
    let (sched, reg) = local(dir.path());
    let (location, snapshot) = make_table(&reg, dir.path()).await;

    let before = requests(&sched);
    let table = super::table::IcebergTable::load(&reg, &location, "db", "t")
        .await
        .unwrap();
    let after_load = requests(&sched);
    assert!(
        after_load > before,
        "the metadata file is read through krishiv-io"
    );

    let schema = table.schema().unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&krishiv_types::DataType::Struct(schema).json()).unwrap();
    let expected = serde_json::json!({"type": "struct", "fields": [
        {"name": "id", "type": "long", "nullable": false, "metadata": {}},
        {"name": "name", "type": "string", "nullable": true, "metadata": {"comment": "the name"}},
        {"name": "tags", "type": {"type": "array", "elementType": "string", "containsNull": false},
         "nullable": true, "metadata": {}},
        {"name": "attrs", "type": {"type": "map", "keyType": "string", "valueType": "double",
         "valueContainsNull": true}, "nullable": true, "metadata": {}},
        {"name": "at", "type": "timestamp", "nullable": true, "metadata": {}},
    ]});
    assert_eq!(json, expected);

    // `bridge_roundtrip` for the inspect tables: each Arrow 60 batch equals iceberg-rust's
    // own Arrow 58 batch, schema and every column (Debug renders both majors alike).
    for (bridged, theirs) in [
        (
            table.snapshots().await.unwrap(),
            table.table.inspect().snapshots().scan().await.unwrap(),
        ),
        (
            table.manifests().await.unwrap(),
            table.table.inspect().manifests().scan().await.unwrap(),
        ),
    ] {
        use futures::TryStreamExt;
        let theirs: Vec<_> = theirs.try_collect().await.unwrap();
        assert_eq!(bridged.len(), theirs.len());
        for (b, t) in bridged.iter().zip(&theirs) {
            assert_eq!(format!("{:?}", b.schema()), format!("{:?}", t.schema()));
            assert_eq!(b.num_columns(), t.num_columns());
            for i in 0..b.num_columns() {
                assert_eq!(
                    format!("{:?}", b.column(i)),
                    format!("{:?}", t.column(i)),
                    "column {i}"
                );
            }
        }
    }
    let snapshots = table.snapshots().await.unwrap();
    assert_eq!(
        snapshots
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
    let ids = snapshots[0]
        .column_by_name("snapshot_id")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), snapshot);

    let manifests = table.manifests().await.unwrap();
    assert_eq!(
        manifests
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
    let added = manifests[0]
        .column_by_name("added_data_files_count")
        .unwrap();
    assert_eq!(added.len(), 1);
    assert!(
        requests(&sched) > after_load,
        "the manifest list is read through krishiv-io"
    );
}

#[tokio::test]
async fn bad_metadata_fails_load() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let loc = format!("{}/v1.metadata.json", dir.path().display());
    std::fs::write(&loc, b"{\"format-version\": 3, \"not\": \"metadata\"}").unwrap();
    let e = super::table::IcebergTable::load(&reg, &loc, "db", "t")
        .await
        .unwrap_err();
    assert!(e.to_string().contains("iceberg-rust"), "{e}");
    let missing = super::table::IcebergTable::load(&reg, &format!("{loc}.none"), "db", "t")
        .await
        .unwrap_err();
    // Spark's class, not an internal error (review S1).
    assert!(
        missing
            .to_string()
            .contains("FAILED_READ_FILE.FILE_NOT_EXIST"),
        "{missing}"
    );
}

#[tokio::test]
async fn runtime_needs_the_io_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let sched = IoScheduler::new(IoConfig::default());
    let paths = Arc::new(LocalPaths::new(
        StoreId::new("file://"),
        dir.path().to_string_lossy().into_owned(),
    ));
    let reg = StorageRegistration::new(sched.context(), paths);
    // No fallback to the calling task's runtime, which may be a compute worker.
    let e = reg.runtime().unwrap_err();
    assert!(e.to_string().contains("I/O scheduler's runtime"), "{e}");
}

#[test]
fn names_differ_from_another_process() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    // `krishiv-<per-process random>-<n>`: the same counter in a process with the same
    // pid (a container's pid 1) is another name.
    let (token, n) = reg
        .name()
        .strip_prefix("krishiv-")
        .unwrap()
        .split_once('-')
        .unwrap();
    assert_eq!(token.len(), 16);
    let other = format!(
        "krishiv-{:016x}-{n}",
        u64::from_str_radix(token, 16).unwrap() ^ 1
    );
    let factory: &dyn iceberg::io::StorageFactory = &KrishivStorageFactory { name: other };
    assert!(factory.build(&iceberg::io::StorageConfig::new()).is_err());
}

#[test]
fn column_defaults_are_refused() {
    use iceberg::spec::{Literal, NestedField, PrimitiveType, Type};
    for f in [
        NestedField::optional(1, "a", Type::Primitive(PrimitiveType::Int))
            .with_initial_default(Literal::int(1)),
        NestedField::optional(1, "a", Type::Primitive(PrimitiveType::Int))
            .with_write_default(Literal::int(1)),
    ] {
        let e = super::types::spark_field(&f).unwrap_err();
        assert!(e.to_string().contains("column default values"), "{e}");
    }
}

/// The Arrow table schema carries the id of every nested field: struct
/// children, list elements, map keys and values.
#[test]
fn arrow_schema_carries_nested_field_ids() {
    use arrow_schema::{DataType as A, Field};
    use iceberg::spec::{ListType, NestedField, PrimitiveType, StructType, Type};
    let schema = iceberg::spec::Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(
                9,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::required(10, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        11,
                        "l",
                        Type::List(ListType::new(
                            NestedField::list_element(
                                12,
                                Type::Primitive(PrimitiveType::Int),
                                false,
                            )
                            .into(),
                        )),
                    )
                    .into(),
                ])),
            )
            .into(),
        ])
        .build()
        .unwrap();
    let id = |f: &Field| f.metadata().get(krishiv_io::FIELD_ID_KEY).cloned();
    let arrow = super::types::arrow_schema(&schema).unwrap();
    assert_eq!(id(arrow.field(0)).as_deref(), Some("1"));
    let s = arrow.field(1);
    assert_eq!(id(s).as_deref(), Some("9"));
    let A::Struct(children) = s.data_type() else {
        panic!("{s:?}")
    };
    assert_eq!(id(&children[0]).as_deref(), Some("10"));
    assert_eq!(id(&children[1]).as_deref(), Some("11"));
    let A::List(element) = children[1].data_type() else {
        panic!("{:?}", children[1])
    };
    assert_eq!(id(element).as_deref(), Some("12"));

    let arrow = super::types::arrow_schema(&test_schema()).unwrap();
    let A::List(element) = arrow.field(2).data_type() else {
        panic!("{:?}", arrow.field(2))
    };
    assert_eq!(id(element).as_deref(), Some("4"));
    let A::Map(entries, _) = arrow.field(3).data_type() else {
        panic!("{:?}", arrow.field(3))
    };
    let A::Struct(kv) = entries.data_type() else {
        panic!("{entries:?}")
    };
    assert_eq!(
        (id(&kv[0]).as_deref(), id(&kv[1]).as_deref()),
        (Some("6"), Some("7"))
    );
}

/// A manifest read fetches only its declared length: a file larger than its
/// manifest list says is refused (never parsed short, never collected whole),
/// as is one smaller; either size past the cap is refused before any byte is
/// read.
#[tokio::test]
async fn capped_reads_fetch_at_most_the_declared_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let io = reg.file_io();
    let loc = format!("file://{}/m/big.avro", dir.path().display());
    io.new_output(&loc)
        .unwrap()
        .write(bytes::Bytes::from(vec![7u8; 4096]))
        .await
        .unwrap();
    let read = |declared, max| super::scan::read_capped(&io, "manifest", &loc, declared, max);
    assert_eq!(read(Some(4096), 8192).await.unwrap().len(), 4096);
    assert_eq!(read(None, 8192).await.unwrap().len(), 4096);
    let longer = read(Some(10), 100).await.unwrap_err();
    assert!(
        longer.to_string().contains("is not 10 bytes long"),
        "{longer}"
    );
    let shorter = read(Some(5000), 8192).await.unwrap_err();
    assert_eq!(
        shorter.error_class(),
        krishiv_common::error::ErrorClass::FailedReadFileNoHint,
        "{shorter}"
    );
    for (declared, max) in [(Some(200), 100), (None, 100)] {
        let e = read(declared, max).await.unwrap_err();
        assert!(
            e.to_string().contains("KRISHIV_CATALOG_LIMIT_EXCEEDED"),
            "{e}"
        );
    }
}
