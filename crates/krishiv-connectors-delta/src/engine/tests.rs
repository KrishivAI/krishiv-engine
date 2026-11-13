// SPDX-License-Identifier: Apache-2.0
//! Handler tests on a local temporary table: the storage and JSON handlers, cancellation, the
//! parse limits, path confinement and a kernel snapshot read through them.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use bytes::Bytes;
use delta_kernel::engine_data::FilteredEngineData;
use delta_kernel::schema::{DataType, MapType, StructField, StructType};
use delta_kernel::{
    EngineData, EvaluationHandler, FileMeta, JsonHandler, KernelError, ParquetHandler, Snapshot,
    StorageHandler,
};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::CancellationToken;
use krishiv_io::{IoConfig, IoScheduler, SettledLocalFileSystem, StoreId};
use object_store::ObjectStore;
use url::Url;

use super::pool::LogPool;
use super::*;
use crate::error::from_kernel;

const V0: &str = concat!(
    r#"{"commitInfo":{"timestamp":1700000000000,"operation":"CREATE TABLE","operationParameters":{},"isBlindAppend":true}}"#,
    "\n",
    r#"{"protocol":{"minReaderVersion":1,"minWriterVersion":2}}"#,
    "\n",
    r#"{"metaData":{"id":"6f3e4f0a-1111-2222-3333-444455556666","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"id\",\"type\":\"long\",\"nullable\":true,\"metadata\":{}},{\"name\":\"p\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}","partitionColumns":["p"],"configuration":{},"createdTime":1700000000000}}"#,
    "\n",
);

const V1: &str = concat!(
    r#"{"commitInfo":{"timestamp":1700000001000,"operation":"WRITE","operationParameters":{"mode":"Append"},"isBlindAppend":true}}"#,
    "\n",
    r#"{"add":{"path":"p=a/part-0.parquet","partitionValues":{"p":"a"},"size":100,"modificationTime":1700000001000,"dataChange":true,"stats":"{\"numRecords\":1}"}}"#,
    "\n",
);

struct Fixture {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    root: Url,
    rt: tokio::runtime::Runtime,
    pool: &'static LogPool,
    cancel: Arc<CancellationToken>,
    engine: DeltaEngine,
}

fn pool(threads: usize) -> &'static LogPool {
    Box::leak(Box::new(LogPool::new(threads)))
}

impl Fixture {
    fn new() -> Self {
        Self::with(DeltaLimits::default(), pool(2), None)
    }

    fn with(
        limits: DeltaLimits,
        pool: &'static LogPool,
        store: Option<Arc<dyn ObjectStore>>,
    ) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        std::fs::create_dir_all(dir.join("_delta_log")).unwrap();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let sched = IoScheduler::new(IoConfig::default());
        let id = StoreId::new("file");
        let store = store
            .unwrap_or_else(|| Arc::new(SettledLocalFileSystem::new()) as Arc<dyn ObjectStore>);
        sched.register_whole_listing_store(id.clone(), store);
        let root = Url::from_directory_path(&dir).unwrap();
        let cancel = Arc::new(CancellationToken::new());
        let engine = DeltaEngine::new(
            sched.context(),
            id,
            root.clone(),
            rt.handle().clone(),
            limits,
            Arc::clone(&cancel),
            pool,
        );
        Self {
            _tmp: tmp,
            dir,
            root,
            rt,
            pool,
            cancel,
            engine,
        }
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn url(&self, rel: &str) -> Url {
        self.root.join(rel).unwrap()
    }

    /// Runs `f` on the log pool.
    fn run<T: Send + 'static>(&self, f: impl FnOnce(&DeltaEngine) -> T + Send + 'static) -> T {
        self.rt.block_on(self.engine.delta_blocking(f)).unwrap()
    }

    fn class(&self, e: &KernelError) -> ErrorClass {
        from_kernel(e, self.root.as_str()).error_class()
    }
}

fn names(root: &Url, metas: &[FileMeta]) -> Vec<String> {
    metas
        .iter()
        .map(|m| {
            m.location
                .as_str()
                .strip_prefix(root.as_str())
                .unwrap()
                .to_owned()
        })
        .collect()
}

#[test]
fn storage_handler_lists_reads_writes_heads_and_deletes() {
    let f = Fixture::new();
    f.write("_delta_log/00000000000000000001.json", V1);
    f.write("_delta_log/00000000000000000000.json", V0);
    f.write("_delta_log/_last_checkpoint", r#"{"version":0,"size":3}"#);
    f.write("_delta_log/_sidecars/a.parquet", "x");
    let root = f.root.clone();
    let (from_zero, from_dir, ranged, whole) = f.run({
        let root = root.clone();
        move |e| {
            let s = e.storage();
            let list = |u: &str| -> Vec<FileMeta> {
                s.list_from(&root.join(u).unwrap())
                    .unwrap()
                    .map(Result::unwrap)
                    .collect()
            };
            let from_zero = list("_delta_log/00000000000000000000.json");
            let from_dir = list("_delta_log/");
            let mut read = s
                .read_files(vec![
                    (
                        root.join("_delta_log/00000000000000000001.json").unwrap(),
                        Some(2..14),
                    ),
                    (root.join("_delta_log/_last_checkpoint").unwrap(), None),
                ])
                .unwrap();
            let ranged = read.next().unwrap().unwrap();
            let whole = read.next().unwrap().unwrap();
            (from_zero, from_dir, ranged, whole)
        }
    });
    assert_eq!(
        names(&root, &from_zero),
        [
            "_delta_log/00000000000000000001.json",
            "_delta_log/_last_checkpoint",
            "_delta_log/_sidecars/a.parquet"
        ]
    );
    assert_eq!(from_dir.len(), 4);
    assert_eq!(from_zero[0].size, V1.len() as u64);
    assert!(from_zero[0].last_modified > 0);
    assert_eq!(&ranged[..], &V1.as_bytes()[2..14]);
    assert_eq!(&whole[..], br#"{"version":0,"size":3}"#);

    let u = f.url("_delta_log/00000000000000000002.json");
    let results = f.run(move |e| {
        let s = e.storage();
        let created = s.put(&u, Bytes::from_static(b"a"), false);
        let again = s.put(&u, Bytes::from_static(b"b"), false);
        let overwritten = s.put(&u, Bytes::from_static(b"cc"), true);
        let head = s.head(&u).map(|m| m.size);
        let deleted = s.delete(&u);
        let deleted_twice = s.delete(&u);
        let missing = s.head(&u).map(|m| m.size);
        let missing_read = s
            .read_files(vec![(u.clone(), None)])
            .unwrap()
            .next()
            .unwrap()
            .map(|_| ());
        let copy = s.copy_atomic(&u, &u);
        (
            created,
            again,
            overwritten,
            head,
            deleted,
            deleted_twice,
            missing,
            missing_read,
            copy,
        )
    });
    let (created, again, overwritten, head, deleted, deleted_twice, missing, missing_read, copy) =
        results;
    assert!(created.is_ok());
    assert!(
        matches!(again, Err(KernelError::FileAlreadyExists(_))),
        "{again:?}"
    );
    assert!(overwritten.is_ok());
    assert_eq!(head.unwrap(), 2);
    assert!(deleted.is_ok() && deleted_twice.is_ok());
    assert!(
        matches!(missing, Err(KernelError::FileNotFound(_))),
        "{missing:?}"
    );
    assert!(
        matches!(missing_read, Err(KernelError::FileNotFound(_))),
        "{missing_read:?}"
    );
    let copy = copy.unwrap_err();
    assert_eq!(f.class(&copy), ErrorClass::UnsupportedFeatureTableOperation);
}

fn action_schema() -> Arc<StructType> {
    let add = StructType::try_new([
        StructField::nullable("path", DataType::STRING),
        StructField::nullable(
            "partitionValues",
            MapType::new(DataType::STRING, DataType::STRING, true),
        ),
        StructField::nullable("size", DataType::LONG),
    ])
    .unwrap();
    let meta = StructType::try_new([
        StructField::nullable("id", DataType::STRING),
        StructField::nullable("schemaString", DataType::STRING),
    ])
    .unwrap();
    Arc::new(
        StructType::try_new([
            StructField::nullable("add", add),
            StructField::nullable("metaData", meta),
            StructField::nullable("ts", DataType::TIMESTAMP),
        ])
        .unwrap(),
    )
}

#[test]
fn json_handler_reads_in_order_parses_and_commits() {
    let f = Fixture::new();
    f.write("_delta_log/00000000000000000000.json", V0);
    f.write("_delta_log/00000000000000000001.json", V1);
    let files: Vec<FileMeta> = ["00000000000000000001.json", "00000000000000000000.json"]
        .iter()
        .map(|n| FileMeta {
            location: f.url(&format!("_delta_log/{n}")),
            last_modified: 0,
            size: 0,
        })
        .collect();
    let commit = f.url("_delta_log/00000000000000000002.json");
    let (batches, parsed, written, bytes, conflict) = f.run(move |e| {
        let j = e.json();
        let batches: Vec<RecordBatch> = j
            .read_json_files(&files, action_schema(), None)
            .unwrap()
            .map(|d| KrishivEngineData::from_engine_data(d.unwrap()).unwrap().into_batch())
            .collect();
        let strings = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", ArrowType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec![
                Some(r#"{"add":{"path":"x","partitionValues":{"p":null},"size":5},"ts":"2024-01-02T03:04:05.000006Z"}"#),
                None,
            ]))],
        )
        .unwrap();
        let parsed = j
            .parse_json(Box::new(KrishivEngineData::new(strings)), action_schema())
            .unwrap();
        let parsed_batch = KrishivEngineData::batch_of(parsed.as_ref()).unwrap().clone();
        // Commit the first parsed row only.
        let data = FilteredEngineData::try_new(parsed, vec![true, false]).unwrap();
        let written = j.write_json_file(&commit, Box::new(std::iter::once(Ok(data))), false).unwrap();
        let bytes = e.storage().read_files(vec![(commit.clone(), None)]).unwrap().next().unwrap().unwrap();
        let again = FilteredEngineData::with_all_rows_selected(Box::new(KrishivEngineData::new(parsed_batch.clone())));
        let conflict = j.write_json_file(&commit, Box::new(std::iter::once(Ok(again))), false);
        (batches, parsed_batch, written, bytes, conflict)
    });
    // File order, then row order: V1's two rows before V0's three.
    let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
    assert_eq!(rows, [2, 3]);
    let add = batches[0].column(0).as_struct();
    assert_eq!(
        add.column(0).as_string::<i32>().value(1),
        "p=a/part-0.parquet"
    );
    // A null string parses as a row of nulls; a timestamp keeps the UTC label.
    assert_eq!(parsed.num_rows(), 2);
    assert!(parsed.column(0).is_null(1));
    assert_eq!(
        parsed.schema().field(2).data_type(),
        &ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into()))
    );
    assert_eq!(
        parsed
            .column(2)
            .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0),
        1_704_164_645_000_006
    );
    let text = std::str::from_utf8(&bytes).unwrap();
    assert_eq!(written, bytes.len() as u64);
    assert!(text.starts_with(r#"{"add":{"path":"x","partitionValues":{"p":null},"size":5},"ts":"2024-01-02T03:04:05.000006"#), "{text}");
    assert!(!text.contains("metaData"), "nulls are omitted: {text}");
    assert_eq!(text.lines().count(), 1);
    assert!(
        matches!(conflict, Err(KernelError::FileAlreadyExists(_))),
        "{conflict:?}"
    );
}

/// The kernel reads a two-commit table's log through these handlers.
#[test]
fn the_kernel_builds_a_snapshot_through_our_handlers() {
    let f = Fixture::new();
    f.write("_delta_log/00000000000000000000.json", V0);
    f.write("_delta_log/00000000000000000001.json", V1);
    let root = f.root.clone();
    let (version, schema, at_zero) = f.run(move |e| {
        let engine = TestEngine(e.clone());
        let snapshot = Snapshot::builder_for(root.as_str())
            .with_cancellation_token(e.kernel_cancellation())
            .build(&engine)
            .unwrap();
        let old = Snapshot::builder_for(root.as_str())
            .at_version(0)
            .build(&engine)
            .unwrap();
        (snapshot.version(), snapshot.schema(), old.version())
    });
    assert_eq!((version, at_zero), (1, 0));
    let fields: Vec<_> = schema
        .fields()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    assert_eq!(
        fields,
        [
            ("id".to_owned(), DataType::LONG),
            ("p".to_owned(), DataType::STRING)
        ]
    );
}

/// The kernel's other two handlers are part 2 (the evaluator, the Parquet handler).
struct TestEngine(DeltaEngine);

struct NoEval;

impl EvaluationHandler for NoEval {
    fn new_expression_evaluator(
        &self,
        _: delta_kernel::schema::SchemaRef,
        _: delta_kernel::expressions::ExpressionRef,
        _: DataType,
    ) -> delta_kernel::KernelResult<Arc<dyn delta_kernel::ExpressionEvaluator>> {
        Err(KernelError::unsupported("the evaluator is PR-08a part 2"))
    }

    fn new_predicate_evaluator(
        &self,
        _: delta_kernel::schema::SchemaRef,
        _: delta_kernel::expressions::PredicateRef,
    ) -> delta_kernel::KernelResult<Arc<dyn delta_kernel::PredicateEvaluator>> {
        Err(KernelError::unsupported("the evaluator is PR-08a part 2"))
    }

    fn create_many(
        &self,
        _: delta_kernel::schema::SchemaRef,
        _: Vec<Vec<delta_kernel::expressions::Scalar>>,
    ) -> delta_kernel::KernelResult<Box<dyn EngineData>> {
        Err(KernelError::unsupported("the evaluator is PR-08a part 2"))
    }
}

struct NoParquet;

impl ParquetHandler for NoParquet {
    fn read_parquet_files(
        &self,
        _: &[FileMeta],
        _: delta_kernel::schema::SchemaRef,
        _: Option<delta_kernel::expressions::PredicateRef>,
    ) -> delta_kernel::KernelResult<delta_kernel::FileDataReadResultIterator> {
        Err(KernelError::unsupported(
            "the Parquet handler is PR-08a part 2",
        ))
    }

    fn write_parquet_file(
        &self,
        _: Url,
        _: delta_kernel::KernelResultIteratorStatic<Box<dyn EngineData>>,
    ) -> delta_kernel::KernelResult<delta_kernel::FileSize> {
        Err(KernelError::unsupported(
            "the Parquet handler is PR-08a part 2",
        ))
    }

    fn read_parquet_footer(
        &self,
        _: &FileMeta,
    ) -> delta_kernel::KernelResult<delta_kernel::ParquetFooter> {
        Err(KernelError::unsupported(
            "the Parquet handler is PR-08a part 2",
        ))
    }
}

impl delta_kernel::Engine for TestEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        Arc::new(NoEval)
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.0.storage()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.0.json()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        Arc::new(NoParquet)
    }
}

#[test]
fn delta_handler_cancellation() {
    let f = Fixture::new();
    f.write("_delta_log/00000000000000000000.json", V0);
    let root = f.root.clone();
    // The query is cancelled while a call is on the pool (the pool itself is driven with a live
    // token here, so the handlers' own checks are what is tested).
    f.cancel.cancel();
    let engine = f.engine.clone();
    let live = CancellationToken::new();
    let errors =
        f.rt.block_on(f.pool.run(f.rt.handle(), &live, move || {
            let e = &engine;
            let s = e.storage();
            let j = e.json();
            let u = root.join("_delta_log/00000000000000000000.json").unwrap();
            let meta = FileMeta {
                location: u.clone(),
                last_modified: 0,
                size: 0,
            };
            vec![
                s.list_from(&u).map(|_| ()).unwrap_err(),
                s.read_files(vec![(u.clone(), None)])
                    .map(|_| ())
                    .unwrap_err(),
                s.put(&u, Bytes::new(), true).unwrap_err(),
                s.head(&u).map(|_| ()).unwrap_err(),
                s.delete(&u).unwrap_err(),
                j.read_json_files(&[meta], action_schema(), None)
                    .map(|_| ())
                    .unwrap_err(),
                j.write_json_file(&u, Box::new(std::iter::empty()), true)
                    .unwrap_err(),
            ]
        }))
        .unwrap();
    for e in &errors {
        assert!(matches!(e, KernelError::Cancelled), "{e:?}");
        assert_eq!(f.class(e), ErrorClass::OperationCanceled);
    }
    // A cancelled query does not start the next kernel call.
    let e = f.rt.block_on(f.engine.delta_blocking(|_| ())).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::OperationCanceled);
}

#[test]
fn the_pool_bounds_kernel_calls_and_a_cancelled_wait_ends() {
    let shared = pool(1);
    let busy = Fixture::with(DeltaLimits::default(), shared, None);
    let waiting = Fixture::with(DeltaLimits::default(), shared, None);
    let (release, hold) = mpsc::channel::<()>();
    let (started, is_started) = mpsc::channel::<()>();
    let first = busy.rt.spawn({
        let engine = busy.engine.clone();
        async move {
            engine
                .delta_blocking(move |_| {
                    started.send(()).unwrap();
                    hold.recv().unwrap();
                })
                .await
        }
    });
    is_started.recv().unwrap();
    assert_eq!(shared.free(), 0);
    let second = waiting.rt.spawn({
        let engine = waiting.engine.clone();
        async move { engine.delta_blocking(|_| 7).await }
    });
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(!second.is_finished(), "the second call waits for the slot");
    waiting.cancel.cancel();
    let e: KrishivError = waiting.rt.block_on(second).unwrap().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::OperationCanceled);
    release.send(()).unwrap();
    busy.rt.block_on(first).unwrap().unwrap();
    assert_eq!(shared.free(), 1);
    assert_eq!(
        shared.waits().count(),
        1,
        "only the call that got a slot records a wait"
    );
    assert_eq!(shared.threads(), 1);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "outside the log pool")]
fn a_handler_called_off_the_pool_is_refused() {
    let f = Fixture::new();
    let _ = f.engine.storage().head(&f.url("_delta_log/x"));
}

#[test]
fn catalog_parse_limits() {
    let limits = DeltaLimits {
        max_metadata_bytes: 600,
        max_depth: 64,
        max_columns: 2,
    };
    let f = Fixture::with(limits, pool(1), None);
    // A commit file over the byte limit.
    f.write("_delta_log/00000000000000000000.json", &"{}\n".repeat(300));
    // A commit nested 65 deep, small enough to pass the size limit.
    let deep = format!("{}[]{}\n", r#"{"a":"#.repeat(64), "}".repeat(64));
    f.write("_delta_log/00000000000000000001.json", &deep);
    // A metaData schema string with three columns (limit two).
    let wide = r#"{"metaData":{"id":"x","schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"a\",\"type\":\"long\",\"nullable\":true,\"metadata\":{}},{\"name\":\"s\",\"type\":{\"type\":\"struct\",\"fields\":[{\"name\":\"b\",\"type\":\"long\",\"nullable\":true,\"metadata\":{}}]},\"nullable\":true,\"metadata\":{}}]}"}}"#;
    f.write("_delta_log/00000000000000000002.json", &format!("{wide}\n"));
    // `_last_checkpoint` and `.crc`, which the kernel parses itself.
    f.write("_delta_log/_last_checkpoint", &"x".repeat(700));
    f.write("_delta_log/00000000000000000001.crc", &deep);
    let root = f.root.clone();
    let errors = f.run(move |e| {
        let j = e.json();
        let s = e.storage();
        let read = |n: &str| {
            let meta = FileMeta {
                location: root.join(&format!("_delta_log/{n}")).unwrap(),
                last_modified: 0,
                size: 0,
            };
            j.read_json_files(&[meta], action_schema(), None)
                .unwrap()
                .find_map(Result::err)
                .unwrap()
        };
        let strings = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", ArrowType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["x".repeat(700)]))],
        )
        .unwrap();
        vec![
            read("00000000000000000000.json"),
            read("00000000000000000001.json"),
            read("00000000000000000002.json"),
            s.read_files(vec![(
                root.join("_delta_log/_last_checkpoint").unwrap(),
                None,
            )])
            .unwrap()
            .next()
            .unwrap()
            .unwrap_err(),
            s.read_files(vec![(
                root.join("_delta_log/00000000000000000001.crc").unwrap(),
                None,
            )])
            .unwrap()
            .next()
            .unwrap()
            .unwrap_err(),
            j.parse_json(Box::new(KrishivEngineData::new(strings)), action_schema())
                .map(|_| ())
                .unwrap_err(),
        ]
    });
    let messages: Vec<String> = errors
        .iter()
        .map(|e| from_kernel(e, "t").to_string())
        .collect();
    for (e, m) in errors.iter().zip(&messages) {
        assert_eq!(f.class(e), ErrorClass::KrishivCatalogLimitExceeded, "{m}");
    }
    assert!(
        messages[0].contains("krishiv.catalog.maxMetadataBytes"),
        "{}",
        messages[0]
    );
    assert!(
        messages[1].contains("nesting depth limit of 64"),
        "{}",
        messages[1]
    );
    assert!(
        messages[2].contains("3 columns") && messages[2].contains("krishiv.catalog.maxColumns"),
        "{}",
        messages[2]
    );
    assert!(messages[4].contains("nesting depth"), "{}", messages[4]);
    // Exactly at the limits passes.
    let ok = DeltaLimits {
        max_metadata_bytes: 600,
        max_depth: 64,
        max_columns: 3,
    };
    let at_depth = format!("{}{}", "[".repeat(64), "]".repeat(64));
    assert!(json::check_depth(&ok, "s", at_depth.as_bytes()).is_ok());
    assert!(json::check_depth(&ok, "s", format!("[{at_depth}]").as_bytes()).is_err());
    // Brackets inside strings do not count.
    assert!(json::check_depth(&ok, "s", format!("\"{}\\\"\"", "[".repeat(100)).as_bytes()).is_ok());
    let wide_schema = r#"{"type":"struct","fields":[{"name":"a","type":"long"},{"name":"s","type":{"type":"struct","fields":[{"name":"b","type":"long"}]}}]}"#;
    assert!(json::check_schema_string(&ok, "s", wide_schema).is_ok());
}

#[test]
fn delta_log_paths_confined() {
    let f = Fixture::new();
    f.write("_delta_log/00000000000000000000.json", V0);
    let outside_dir = f.dir.parent().unwrap().join("other");
    std::fs::create_dir_all(&outside_dir).unwrap();
    std::fs::write(outside_dir.join("x.json"), V0).unwrap();
    let root = f.root.clone();
    let urls = vec![
        // `..` resolved by the URL parser lands outside the root.
        root.join("../other/x.json").unwrap(),
        root.join("_delta_log/../../other/x.json").unwrap(),
        Url::from_file_path(outside_dir.join("x.json")).unwrap(),
        Url::parse("file:///etc/passwd").unwrap(),
        Url::parse("s3://bucket/t/_delta_log/00000000000000000000.json").unwrap(),
        Url::parse("file://otherhost/t/x").unwrap(),
        // A sibling whose name extends the root's.
        Url::from_file_path(format!("{}2/_delta_log/0.json", f.dir.display())).unwrap(),
    ];
    let errors = f.run(move |e| {
        let s = e.storage();
        let j = e.json();
        let mut out = Vec::new();
        for u in urls {
            out.push(
                s.read_files(vec![(u.clone(), None)])
                    .unwrap()
                    .next()
                    .unwrap()
                    .unwrap_err(),
            );
            out.push(s.head(&u).map(|_| ()).unwrap_err());
            out.push(s.list_from(&u).map(|_| ()).unwrap_err());
            out.push(s.put(&u, Bytes::new(), true).unwrap_err());
            out.push(s.delete(&u).unwrap_err());
            let meta = FileMeta {
                location: u,
                last_modified: 0,
                size: 0,
            };
            out.push(
                j.read_json_files(&[meta], action_schema(), None)
                    .unwrap()
                    .find_map(Result::err)
                    .unwrap(),
            );
        }
        out
    });
    assert_eq!(errors.len(), 42);
    for e in &errors {
        let ours = from_kernel(e, "t");
        assert_eq!(
            ours.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation,
            "{ours}"
        );
        assert!(
            ours.to_string().contains("outside the table root"),
            "{ours}"
        );
    }
    // The file outside was neither read nor overwritten.
    assert_eq!(
        std::fs::read_to_string(outside_dir.join("x.json")).unwrap(),
        V0
    );
    // Inside the root is fine, including the root itself as a listing.
    let ok = f.run({
        let root = f.root.clone();
        move |e| e.storage().list_from(&root).map(Iterator::count)
    });
    assert_eq!(ok.unwrap(), 1);
}

/// A store that refuses conditional puts, as one without `If-None-Match` does.
#[derive(Debug)]
struct NoConditionalPut(object_store::memory::InMemory);

impl std::fmt::Display for NoConditionalPut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NoConditionalPut")
    }
}

#[async_trait::async_trait]
impl ObjectStore for NoConditionalPut {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        if matches!(opts.mode, object_store::PutMode::Create) {
            return Err(object_store::Error::NotImplemented {
                operation: "put_opts(Create)".to_owned(),
                implementer: "NoConditionalPut".to_owned(),
            });
        }
        self.0.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.0.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.0.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.0.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.0.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.0.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.0.copy_opts(from, to, options).await
    }
}

#[test]
fn a_commit_on_a_store_without_conditional_put_is_refused() {
    let store =
        Arc::new(NoConditionalPut(object_store::memory::InMemory::new())) as Arc<dyn ObjectStore>;
    let f = Fixture::with(DeltaLimits::default(), pool(1), Some(store));
    let u = f.url("_delta_log/00000000000000000000.json");
    let (commit, overwrite) = f.run(move |e| {
        let empty = || {
            Box::new(std::iter::empty())
                as delta_kernel::KernelResultIterator<'static, FilteredEngineData>
        };
        (
            e.json().write_json_file(&u, empty(), false),
            e.json().write_json_file(&u, empty(), true),
        )
    });
    let e = from_kernel(&commit.unwrap_err(), "t");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.to_string()
            .contains("Delta commit on a store without conditional put"),
        "{e}"
    );
    assert_eq!(overwrite.unwrap(), 0);
}
