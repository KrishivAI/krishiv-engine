// SPDX-License-Identifier: Apache-2.0
//! Reads by path (P2-T154) through `krishiv_exec::files`: listing as Spark
//! 4.2 lists (hidden files, sort order, missing and empty paths), schema
//! inference through the mock and Parquet formats, the local file system,
//! and the analyzer end to end with `FileReads` as its resolver.
#![cfg(test)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{FileReads, LocalPaths, LocatedPath, PathLocator};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{FormatRegistry, IoConfig, IoScheduler, StoreId};
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, FileSchemaResolver, MemoryCatalog, ResolutionConf, SourceKind,
    TableColumn,
};
use krishiv_plan::{LogicalPlan, Plan, PlanNode, ReadSource};
use krishiv_testkit::file_format::mock::MockFormat;
use krishiv_testkit::file_format::parquet::ParquetFixture;
use krishiv_testkit::file_format::{Env, MockFixture};
use krishiv_types::DataType;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path;

/// Paths `/a/b` in the harness's in-memory store; counts lookups.
#[derive(Debug, Default)]
struct MemPaths(AtomicUsize);

impl PathLocator for MemPaths {
    fn locate(&self, path: &str) -> krishiv_exec::Result<LocatedPath> {
        self.0.fetch_add(1, Ordering::Relaxed);
        // Its own display form places the same path again.
        let path = path.strip_prefix("memory:").unwrap_or(path);
        Ok(LocatedPath {
            store: StoreId::new("memory://suite"),
            path: Path::from(path),
            display: format!("memory:{path}"),
        })
    }
}

fn schema(fields: &[(&str, ArrowType)]) -> SchemaRef {
    Arc::new(Schema::new(
        fields
            .iter()
            .map(|(n, t)| Field::new(*n, t.clone(), false))
            .collect::<Vec<_>>(),
    ))
}

fn ints(s: &SchemaRef) -> RecordBatch {
    RecordBatch::try_new(Arc::clone(s), vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap()
}

fn longs(s: &SchemaRef) -> RecordBatch {
    RecordBatch::try_new(Arc::clone(s), vec![Arc::new(Int64Array::from(vec![1_i64]))]).unwrap()
}

fn reads(
    env: &Env,
    formats: Vec<Arc<dyn krishiv_io::FileFormat>>,
    paths: Arc<dyn PathLocator>,
) -> FileReads {
    FileReads::new(
        env.io.clone(),
        registry_of(formats),
        paths,
        Arc::new(Config::new()),
    )
}

fn registry_of(formats: Vec<Arc<dyn krishiv_io::FileFormat>>) -> Arc<FormatRegistry> {
    let mut registry = FormatRegistry::new();
    for f in formats {
        registry.register(f).unwrap();
    }
    Arc::new(registry)
}

fn paths(p: &[&str]) -> Vec<OptionValue> {
    p.iter()
        .map(|p| OptionValue::from_user("path", *p))
        .collect()
}

fn columns(r: Result<Vec<TableColumn>, KrishivError>) -> Vec<(String, DataType)> {
    r.unwrap()
        .into_iter()
        .map(|c| (c.name, c.data_type))
        .collect()
}

fn class<T: std::fmt::Debug>(r: Result<T, KrishivError>) -> (ErrorClass, String) {
    let e = r.unwrap_err();
    let first = e
        .message_parameters()
        .first()
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    (e.error_class(), first)
}

#[test]
fn directories_list_like_spark() {
    let env = Env::new(None).unwrap();
    let fx = MockFixture::default();
    let a = schema(&[("a", ArrowType::Int32)]);
    let z = schema(&[("z", ArrowType::Int64)]);
    // The schema comes from the first file by path, whatever the order of
    // writing or of the paths given.
    env.write(&fx, "d/part-1", &z, &[longs(&z)], &[]).unwrap();
    env.write(&fx, "d/part-0", &a, &[ints(&a)], &[]).unwrap();
    env.put_raw(&Path::from("d/_SUCCESS"), Vec::new()).unwrap();
    env.put_raw(&Path::from("d/.part-0.crc"), b"junk".to_vec())
        .unwrap();
    env.put_raw(&Path::from("d/_temporary/0/x"), b"junk".to_vec())
        .unwrap();
    env.put_raw(&Path::from("d/x._COPYING_"), b"junk".to_vec())
        .unwrap();
    let locator = Arc::new(MemPaths::default());
    let r = reads(&env, vec![Arc::new(MockFormat)], Arc::clone(&locator) as _);
    let none = BTreeMap::new();
    let got = env.block_on(r.infer("MOCK", &paths(&["/d"]), &none));
    assert_eq!(columns(got), [("a".to_owned(), DataType::Integer)]);
    let listing = env
        .block_on(r.listing("mock", &paths(&["/d"]), &none))
        .unwrap();
    let names: Vec<_> = listing.files.iter().map(|f| f.path.to_string()).collect();
    assert_eq!(names, ["d/part-0", "d/part-1"]);
    // Listed once per query: the second call reused the listing.
    assert_eq!(locator.0.load(Ordering::Relaxed), 1);
    // A file by itself; a hidden one is skipped even when named.
    let got = env.block_on(r.infer("mock", &paths(&["/d/part-1"]), &none));
    assert_eq!(columns(got), [("z".to_owned(), DataType::Long)]);
    let got = env.block_on(r.infer("mock", &paths(&["/d/_SUCCESS"]), &none));
    assert_eq!(
        class(got),
        (ErrorClass::UnableToInferSchema, "mock".to_owned())
    );
    let got = env.block_on(r.infer("mock", &[], &none));
    assert_eq!(class(got).0, ErrorClass::UnableToInferSchema);
    // Several paths: one missing fails the read.
    let got = env.block_on(r.infer("mock", &paths(&["/d", "/nope"]), &none));
    assert_eq!(
        class(got),
        (ErrorClass::PathNotFound, "memory:/nope".to_owned())
    );
    // A `name=value` subdirectory is a partition column, after the data's.
    env.write(&fx, "p/k=1/part-0", &a, &[ints(&a)], &[])
        .unwrap();
    let got = env
        .block_on(r.infer("mock", &paths(&["/p"]), &none))
        .unwrap();
    assert_eq!(got.last().map(|c| c.name.as_str()), Some("k"));
    // Any other subdirectory is refused.
    env.write(&fx, "q/plain/part-0", &a, &[ints(&a)], &[])
        .unwrap();
    let got = env.block_on(r.infer("mock", &paths(&["/q"]), &none));
    assert_eq!(class(got).0, ErrorClass::KrishivUnsupportedFeature);
    // Globs and credentials in paths.
    for p in ["/d/*", "s3://user:secret@bucket/k"] {
        let got = env.block_on(r.infer("mock", &paths(&[p]), &none));
        assert_eq!(class(got).0, ErrorClass::KrishivUnsupportedFeature, "{p}");
    }
    // Paths come before the format: a missing path of a format without a
    // reader here is PATH_NOT_FOUND, an existing one of a Spark built-in
    // source without an implementation is unsupported.
    let got = env.block_on(r.infer("json", &paths(&["/nope"]), &none));
    assert_eq!(class(got).0, ErrorClass::PathNotFound);
    for format in ["json", "orc", "avro"] {
        let got = env.block_on(r.infer(format, &paths(&["/d"]), &none));
        let (kind, param) = class(got);
        assert_eq!(kind, ErrorClass::KrishivUnsupportedFeature, "{format}");
        assert!(
            param.contains(&format!("the {format} file format")),
            "{format}: {param}"
        );
    }
    // A name Spark does not know at all is Spark's DATA_SOURCE_NOT_FOUND.
    for format in ["delta", "nosuchformat"] {
        let got = env.block_on(r.infer(format, &paths(&["/d"]), &none));
        let (kind, param) = class(got);
        assert_eq!(kind, ErrorClass::DataSourceNotFound, "{format}");
        assert_eq!(kind.sql_state(), Some("42K02"), "{format}");
        assert_eq!(param, format.to_owned(), "{format}");
    }
    // Without the three-way resolution this reported an unknown name as
    // unsupported; DATA_SOURCE_NOT_FOUND carries SQLSTATE 42K02.
    // The registered test format resolves case-insensitively, like SQL USING.
    let got = env.block_on(r.infer("MOCK", &paths(&["/d"]), &none));
    assert_eq!(columns(got), [("a".to_owned(), DataType::Integer)]);
    // SQL's `mock`.`/d` reads through the same resolver.
    let mut ctx = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    ctx.files = Some(Arc::new(r) as Arc<dyn FileSchemaResolver>);
    let plan = Plan::try_new(LogicalPlan::table(["mock", "/d"])).unwrap();
    let analyzed = env.block_on(DefaultAnalyzer.analyze(&plan, &ctx)).unwrap();
    assert_eq!(analyzed.output[0].name, "a");
    // An unknown `USING` format is Spark's DATA_SOURCE_NOT_FOUND (42K02).
    let read = LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "delta".to_owned(),
        paths: paths(&["/d"]),
        options: BTreeMap::new(),
    }));
    let e = env
        .block_on(DefaultAnalyzer.analyze(&Plan::try_new(read).unwrap(), &ctx))
        .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::DataSourceNotFound);
    assert_eq!(e.error.error_class().sql_state(), Some("42K02"));
}

#[test]
fn source_kinds_follow_spark() {
    let env = Env::new(None).unwrap();
    let r = reads(
        &env,
        vec![Arc::new(MockFormat)],
        Arc::new(MemPaths::default()),
    );
    for (name, kind) in [
        ("parquet", SourceKind::File),
        ("CSV", SourceKind::File),
        ("ORC", SourceKind::File),
        ("avro", SourceKind::File),
        ("binaryFile", SourceKind::File),
        ("mock", SourceKind::File),
        ("jdbc", SourceKind::Other),
        ("noop", SourceKind::Other),
        ("delta", SourceKind::Unknown),
        ("nosuchformat", SourceKind::Unknown),
    ] {
        assert_eq!(r.source_kind(name), kind, "{name}");
    }
}

#[test]
fn parquet_types_and_options() {
    let env = Env::new(None).unwrap();
    let fx = ParquetFixture;
    let s = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, false),
        Field::new("name", ArrowType::Utf8, true),
        Field::new("amount", ArrowType::Decimal128(12, 2), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&s),
        vec![
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(StringArray::from(vec![Some("x")])),
            Arc::new(
                arrow_array::Decimal128Array::from(vec![Some(1234_i128)])
                    .with_precision_and_scale(12, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    env.write(&fx, "t/part-0.parquet", &s, &[batch], &[])
        .unwrap();
    let r = reads(
        &env,
        vec![Arc::new(ParquetFormat)],
        Arc::new(MemPaths::default()),
    );
    let got = env.block_on(r.infer("parquet", &paths(&["/t"]), &BTreeMap::new()));
    assert_eq!(
        columns(got),
        [
            ("id".to_owned(), DataType::Long),
            ("name".to_owned(), DataType::String),
            (
                "amount".to_owned(),
                DataType::Decimal(krishiv_types::DecimalType::try_new(12, 2).unwrap())
            ),
        ]
    );
    // Options are parsed against the format's schema.
    let opt = |k: &str, v: &str| BTreeMap::from([(k.to_owned(), OptionValue::from_user(k, v))]);
    let got = env.block_on(r.infer("parquet", &paths(&["/t"]), &opt("binaryAsString", "maybe")));
    assert_eq!(class(got).0, ErrorClass::KrishivInvalidFormatOptionValue);
    // Merging schemas is not ignored: it is not supported yet.
    let got = env.block_on(r.infer("parquet", &paths(&["/t"]), &opt("MERGESCHEMA", " True")));
    assert_eq!(class(got).0, ErrorClass::KrishivUnsupportedFeature);
    let got = env.block_on(r.infer("parquet", &paths(&["/t"]), &opt("mergeSchema", "false")));
    assert_eq!(columns(got).len(), 3);
    let (session, _) = Config::from_pairs([("spark.sql.parquet.mergeSchema", "true")]).unwrap();
    let merging = FileReads::new(
        env.io.clone(),
        registry_of(vec![Arc::new(ParquetFormat)]),
        Arc::new(MemPaths::default()),
        Arc::new(session),
    );
    let got = env.block_on(merging.infer("parquet", &paths(&["/t"]), &BTreeMap::new()));
    assert_eq!(class(got).0, ErrorClass::KrishivUnsupportedFeature);
    // The read option wins over the session's.
    let got = env.block_on(merging.infer("parquet", &paths(&["/t"]), &opt("mergeSchema", "false")));
    assert_eq!(columns(got).len(), 3);
}

#[test]
fn the_analyzer_reads_parquet_paths() {
    let env = Env::new(None).unwrap();
    let a = schema(&[("a", ArrowType::Int32)]);
    env.write(&ParquetFixture, "t/part-0.parquet", &a, &[ints(&a)], &[])
        .unwrap();
    let r = Arc::new(reads(
        &env,
        vec![Arc::new(ParquetFormat)],
        Arc::new(MemPaths::default()),
    ));
    let mut ctx = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    ctx.files = Some(Arc::clone(&r) as Arc<dyn FileSchemaResolver>);
    let analyze = |plan: LogicalPlan| {
        env.block_on(DefaultAnalyzer.analyze(&Plan::try_new(plan).unwrap(), &ctx))
    };
    let read = LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "parquet".to_owned(),
        paths: paths(&["/t"]),
        options: BTreeMap::new(),
    }));
    let analyzed = analyze(read).unwrap();
    assert_eq!(analyzed.output.len(), 1);
    // Non-null in the file, nullable as Spark reads it.
    assert!(analyzed.output[0].nullable);
    assert_eq!(analyzed.output[0].data_type, DataType::Integer);
    assert!(analyzed.reads_files);
    // SQL's parquet.`path`.
    let analyzed = analyze(LogicalPlan::table(["parquet", "/t"])).unwrap();
    assert_eq!(analyzed.output[0].name, "a");
    let e = analyze(LogicalPlan::table(["parquet", "/missing"])).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::PathNotFound);
}

#[test]
fn local_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_str().unwrap().to_owned();
    std::fs::create_dir(dir.path().join("empty")).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let sched = IoScheduler::new(IoConfig::default());
    let store = StoreId::new("file");
    sched.register_store(
        store.clone(),
        Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>,
    );
    let r = FileReads::new(
        sched.context(),
        registry_of(vec![Arc::new(ParquetFormat)]),
        Arc::new(LocalPaths::new(store, root.clone())),
        Arc::new(Config::new()),
    );
    let none = BTreeMap::new();
    // An empty directory exists but has no files.
    let got = rt.block_on(r.infer("parquet", &paths(&[&format!("{root}/empty")]), &none));
    assert_eq!(
        class(got),
        (ErrorClass::UnableToInferSchema, "Parquet".to_owned())
    );
    // A relative path, qualified against the working directory.
    let got = rt.block_on(r.infer("parquet", &paths(&["nope"]), &none));
    assert_eq!(
        class(got),
        (ErrorClass::PathNotFound, format!("file:{root}/nope"))
    );
    let got = rt.block_on(r.infer("parquet", &paths(&[&format!("file://{root}/empty")]), &none));
    assert_eq!(class(got).0, ErrorClass::UnableToInferSchema);
}

/// Writes `batch` as a Parquet file at `path`.
fn write_parquet(path: &std::path::Path, batch: &RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    w.write(batch).unwrap();
    w.close().unwrap();
}

/// The tree `gen_file_read_probe.py` lays out under its root.
fn probe_tree(root: &std::path::Path) {
    let one = root.join("one");
    std::fs::create_dir(&one).unwrap();
    let s = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, false),
        Field::new("B", ArrowType::Utf8, false),
        Field::new("c", ArrowType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&s),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["x"])),
            Arc::new(Int32Array::from(vec![None])),
        ],
    )
    .unwrap();
    write_parquet(&one.join("part-00000-1.snappy.parquet"), &batch);
    std::fs::write(one.join("_SUCCESS"), b"").unwrap();
    std::fs::write(one.join(".part-00000-1.snappy.parquet.crc"), b"junk").unwrap();
    let h = root.join("h");
    std::fs::create_dir(&h).unwrap();
    let a = schema(&[("a", ArrowType::Int32)]);
    write_parquet(&h.join("part-00000-2.snappy.parquet"), &ints(&a));
    std::fs::write(h.join(".hidden.parquet"), b"junk").unwrap();
    std::fs::write(h.join("_junk"), b"junk").unwrap();
    std::fs::create_dir(root.join("empty")).unwrap();
    std::fs::create_dir(root.join("onlyhidden")).unwrap();
    std::fs::write(root.join("onlyhidden/_SUCCESS"), b"").unwrap();
}

/// Every case of the Spark 4.2.0 probe: the same schema (names, types,
/// nullability) or the same error class and parameters.
#[test]
fn agrees_with_spark_on_the_file_read_probe() {
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/spark-4.2.0/file-read-probe.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_str().unwrap().to_owned();
    probe_tree(dir.path());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let sched = IoScheduler::new(IoConfig::default());
    let store = StoreId::new("file");
    sched.register_store(
        store.clone(),
        Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>,
    );
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let kind = case["kind"].as_str().unwrap();
        let conf = ResolutionConf {
            run_sql_on_files: kind != "sql-off",
            ..ResolutionConf::default()
        };
        let mut ctx = context(MemoryCatalog::new("spark_catalog"), conf);
        // One query, one resolver, as a session runs it.
        ctx.files = Some(Arc::new(FileReads::new(
            sched.context(),
            registry_of(vec![Arc::new(ParquetFormat)]),
            Arc::new(LocalPaths::new(store.clone(), "/")),
            Arc::new(Config::new()),
        )));
        let sub = |s: &str| s.replace("<root>", &root);
        let plan = if kind == "read" {
            let paths: Vec<String> = case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| sub(p.as_str().unwrap()))
                .collect();
            let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
            Plan::try_new(LogicalPlan::new(PlanNode::Read(ReadSource::Files {
                format: "parquet".to_owned(),
                paths: self::paths(&refs),
                options: BTreeMap::new(),
            })))
            .unwrap()
        } else {
            let sql = sub(case["input"].as_str().unwrap());
            krishiv_sql::parse_and_lower(
                &sql,
                &krishiv_sql::ParserOptions::default(),
                krishiv_plan::SourceId(0),
            )
            .unwrap()
        };
        let got = rt.block_on(DefaultAnalyzer.analyze(&plan, &ctx));
        if let Some(schema) = case.get("schema") {
            let a = got.unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let ours: Vec<_> = a
                .output
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "name": c.name,
                        "type": c.data_type.simple_string(),
                        "nullable": c.nullable,
                    })
                })
                .collect();
            assert_eq!(&serde_json::Value::from(ours), schema, "{name}");
        } else {
            let e = got.unwrap_err();
            assert_eq!(
                e.error.error_class().name(),
                case["error_class"].as_str().unwrap(),
                "{name}"
            );
            // Suggestions are ordered by their distance to the name, which
            // holds the root: compare them as a set.
            let normalize = |k: &str, v: &str| {
                let v = v.replace(&root, "<root>");
                if k == "proposal" {
                    let mut parts: Vec<&str> = v.split(", ").collect();
                    parts.sort_unstable();
                    parts.join(", ")
                } else {
                    v
                }
            };
            let ours: BTreeMap<String, String> = e
                .error
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), normalize(k, v)))
                .collect();
            let spark: BTreeMap<String, String> = case["message_parameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), normalize(k, v.as_str().unwrap())))
                .collect();
            assert_eq!(ours, spark, "{name}");
        }
        checked += 1;
    }
    assert_eq!(checked, 20);
}

#[test]
fn concurrent_reads_keep_one_listing() {
    let env = Env::new(None).unwrap();
    let a = schema(&[("a", ArrowType::Int32)]);
    env.write(&MockFixture::default(), "d/part-0", &a, &[ints(&a)], &[])
        .unwrap();
    let r = reads(
        &env,
        vec![Arc::new(MockFormat)],
        Arc::new(MemPaths::default()),
    );
    let none = BTreeMap::new();
    let p = paths(&["/d"]);
    // Both miss the cache, list, and end with the same listing.
    let (x, y) = env.block_on(async {
        futures::join!(r.listing("mock", &p, &none), r.listing("MOCK", &p, &none))
    });
    let (x, y) = (x.unwrap(), y.unwrap());
    assert!(Arc::ptr_eq(&x, &y));
    let shown = format!("{x:?} {r:?}");
    assert!(
        shown.contains("FileListing") && shown.contains("\"mock\""),
        "{shown}"
    );
    assert!(
        shown.contains("FileReads") && shown.contains("MemPaths"),
        "{shown}"
    );
}
