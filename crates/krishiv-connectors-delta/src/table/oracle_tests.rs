// SPDX-License-Identifier: Apache-2.0
//! The read-path oracle tables of `tests/fixtures/delta-4.4.1/` (Spark 4.2.0 + delta-spark
//! 4.4.1) through [`DeltaTable`] and its scan, run by the executor: every readable table against
//! Spark's `typed_rows` and against the kernel's own `Scan::execute` over the same engine; the
//! refused tables; type widening; pruning by a pushed filter.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, SchemaRef};
use futures::StreamExt;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::{
    CancellationToken, ExecConfig, ExecNode, FilterPushdown, PipelineBuilder, PushedFilter,
    QueryContext, QueryId, QueryRunner, ScanPushdown, ScanPushdowns, Source, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::{ColumnPath, IoConfig, IoScheduler, PushdownExpr, ScalarValue, StoreId};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use serde_json::{Value, json};
use url::Url;

use super::*;
use crate::engine::pool::LogPool;
use crate::engine::{DeltaLimits, KrishivEngineData};

pub(super) fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A table copied to a temporary root, with the engine, runtimes and pools to read it.
pub(super) struct Env {
    _tmp: tempfile::TempDir,
    pub(super) dir: PathBuf,
    rt: tokio::runtime::Runtime,
    query_pool: Arc<dyn MemoryPool>,
    pub(super) engine: DeltaEngine,
}

impl Env {
    /// The oracle table `name`.
    pub(super) fn fixture(name: &str) -> Self {
        let env = Self::empty();
        copy_dir(&fixtures().join(name), &env.dir);
        env
    }

    /// An empty table directory.
    pub(super) fn empty() -> Self {
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
        sched.register_whole_listing_store(
            id.clone(),
            Arc::new(krishiv_io::SettledLocalFileSystem::new()),
        );
        let root = TreePool::new(&MemoryConfig::default());
        let query_pool = root.child("query", None);
        let process: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        sched.attach_memory(&process);
        let pool: &'static LogPool = Box::leak(Box::new(LogPool::new(2)));
        let engine = DeltaEngine::new(
            sched.context().with_memory(Arc::clone(&query_pool)),
            id,
            Url::from_directory_path(&dir).unwrap(),
            rt.handle().clone(),
            DeltaLimits::default(),
            Arc::new(CancellationToken::new()),
            pool,
        )
        .unwrap();
        Self {
            _tmp: tmp,
            dir,
            rt,
            query_pool,
            engine,
        }
    }

    pub(super) fn open(&self) -> Result<DeltaTable, KrishivError> {
        self.rt
            .block_on(DeltaTable::open(self.engine.clone(), None, &Config::new()))
    }

    /// Every batch `source` returns, run by the deterministic executor with two drivers.
    pub(super) fn run(&self, source: Arc<dyn Source>) -> Vec<RecordBatch> {
        let ct = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let query_pool = Arc::clone(&self.query_pool);
        ct.block_on(async move {
            let runner = QueryRunner::new(TokioExecutor::deterministic(2).unwrap());
            let query = QueryContext::new(
                QueryId(1),
                WorkloadClass::Interactive,
                ExecConfig::default(),
                query_pool,
            );
            let mut stream = runner
                .execute(Arc::new(Leaf(source)), Arc::clone(&query))
                .await
                .unwrap();
            let mut out = Vec::new();
            while let Some(b) = stream.next().await {
                out.push(b.unwrap().into_parts().0);
            }
            query.finished().await;
            out
        })
    }

    /// The kernel's own read of the table (`Scan::execute` over the same engine).
    fn kernel_read(&self, table: &DeltaTable) -> Vec<RecordBatch> {
        let snapshot = Arc::clone(table.snapshot());
        self.rt
            .block_on(self.engine.delta_blocking(move |e| {
                let scan = snapshot.scan_builder().build().unwrap();
                let engine: Arc<dyn delta_kernel::Engine> = Arc::new(e.clone());
                scan.execute(engine)
                    .unwrap()
                    .map(|d| {
                        KrishivEngineData::from_engine_data(d.unwrap())
                            .unwrap()
                            .into_batch()
                    })
                    .collect::<Vec<_>>()
            }))
            .unwrap()
    }
}

struct Leaf(Arc<dyn Source>);

impl ExecNode for Leaf {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> krishiv_exec::Result<()> {
        b.set_source(Arc::clone(&self.0)).map(drop)
    }
}

fn expected(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(fixtures().join(name).join("EXPECTED.json")).unwrap())
        .unwrap()
}

/// Python's `repr` of a float.
fn float(v: f64) -> Value {
    let s = format!("{v}");
    let s = if s.contains(['.', 'e', 'N', 'i']) {
        s
    } else {
        format!("{s}.0")
    };
    json!({ "float": s })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn binary_at(a: &dyn Array, row: usize) -> Vec<u8> {
    match a.data_type() {
        ArrowType::BinaryView => a.as_binary_view().value(row).to_vec(),
        ArrowType::Binary => a.as_binary::<i32>().value(row).to_vec(),
        other => panic!("not binary: {other}"),
    }
}

/// The value at `row` of `a` in the generator's typed form; `table` is the table's field (the
/// Variant is told by it, as the kernel's read has no Variant marker).
fn typed(a: &dyn Array, row: usize, table: &Field) -> Value {
    if a.is_null(row) {
        return Value::Null;
    }
    if table.data_type() == &krishiv_types::DataType::Variant.to_arrow() {
        let s = a.as_struct();
        let part = |n: &str| hex(&binary_at(s.column_by_name(n).unwrap().as_ref(), row));
        return json!({ "variant": { "metadata": part("metadata"), "value": part("value") } });
    }
    match a.data_type() {
        ArrowType::Boolean => json!(a.as_boolean().value(row)),
        ArrowType::Int8 => json!(a.as_primitive::<Int8Type>().value(row)),
        ArrowType::Int16 => json!(a.as_primitive::<Int16Type>().value(row)),
        ArrowType::Int32 => json!(a.as_primitive::<Int32Type>().value(row)),
        ArrowType::Int64 => json!(a.as_primitive::<Int64Type>().value(row)),
        ArrowType::Float32 => float(f64::from(a.as_primitive::<Float32Type>().value(row))),
        ArrowType::Float64 => float(a.as_primitive::<Float64Type>().value(row)),
        ArrowType::Utf8 => json!(a.as_string::<i32>().value(row)),
        ArrowType::Utf8View => json!(a.as_string_view().value(row)),
        ArrowType::Binary | ArrowType::BinaryView => json!({ "binary": hex(&binary_at(a, row)) }),
        ArrowType::Date32 => {
            let d = chrono::DateTime::from_timestamp(
                i64::from(a.as_primitive::<Date32Type>().value(row)) * 86_400,
                0,
            )
            .unwrap();
            json!({ "date": d.format("%Y-%m-%d").to_string() })
        }
        ArrowType::Timestamp(_, _) => {
            let us = a.as_primitive::<TimestampMicrosecondType>().value(row);
            let t = chrono::DateTime::from_timestamp_micros(us)
                .unwrap()
                .naive_utc();
            let mut s = t.format("%Y-%m-%dT%H:%M:%S").to_string();
            if us.rem_euclid(1_000_000) != 0 {
                s.push_str(&format!(".{:06}", us.rem_euclid(1_000_000)));
            }
            json!({ "timestamp": s })
        }
        ArrowType::Struct(_) => {
            let s = a.as_struct();
            let ArrowType::Struct(tf) = table.data_type() else {
                panic!("struct read for {table}");
            };
            let fields: serde_json::Map<String, Value> = tf
                .iter()
                .zip(s.columns())
                .map(|(f, c)| (f.name().clone(), typed(c.as_ref(), row, f)))
                .collect();
            json!({ "struct": fields })
        }
        other => panic!("no typed form for {other}"),
    }
}

/// The rows of `batches` in the typed form, ordered by the first column (`id`).
fn rows(batches: &[RecordBatch], table: &SchemaRef) -> Vec<Vec<Value>> {
    let mut out: Vec<Vec<Value>> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |r| {
                b.columns()
                    .iter()
                    .zip(table.fields())
                    .map(|(c, f)| typed(c.as_ref(), r, f))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    out.sort_by_key(|r| r[0].as_i64());
    out
}

/// The readable oracle tables.
const READABLE_TABLES: &[&str] = &[
    "deletion-vectors",
    "column-mapping-name",
    "column-mapping-id",
    "type-widening",
    "partitioned",
    "timestamp-ntz",
    "variant",
    "vacuum-protocol-check",
    "v1-checkpoint",
    "v2-checkpoint",
    "multipart-checkpoint",
];

/// Spark's rows, in the typed form; the checkpoint tables (PR-08a) recorded only `rows`.
fn spark_rows(name: &str) -> Option<(Vec<String>, Vec<Vec<Value>>)> {
    let e = expected(name);
    let fields = e["fields"]
        .as_array()?
        .iter()
        .map(|f| f.as_str().unwrap().to_owned())
        .collect();
    let rows = serde_json::from_value(e["typed_rows"].clone()).unwrap();
    Some((fields, rows))
}

#[test]
fn every_readable_table_matches_spark() {
    for name in READABLE_TABLES {
        let env = Env::fixture(name);
        let table = env.open().unwrap_or_else(|e| panic!("{name}: {e}"));
        let scan = table.delta_scan();
        let source = scan.build(ScanPushdowns::default()).unwrap();
        let batches = env.run(source);
        let schema = table.schema();
        let got = rows(&batches, &schema);
        match spark_rows(name) {
            Some((fields, want)) => {
                let names: Vec<&String> = schema.fields().iter().map(|f| f.name()).collect();
                assert_eq!(names, fields.iter().collect::<Vec<_>>(), "{name}: columns");
                assert_eq!(got, want, "{name}");
            }
            None => {
                // PR-08a's tables: ids only.
                let want: Vec<i64> = expected(name)["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| r["id"].as_i64().unwrap())
                    .collect();
                let ids: Vec<i64> = got.iter().map(|r| r[0].as_i64().unwrap()).collect();
                assert_eq!(ids, want, "{name}");
            }
        }
        // Statistics: the live rows, deletion vectors subtracted.
        assert_eq!(
            table.statistics().map(|s| s.rows),
            Some(got.len() as u64),
            "{name}: statistics"
        );
        assert_eq!(table.scan_units(), Some(table.units().len() as u64));
    }
}

#[test]
fn scan_shape_matches_kernel() {
    for name in READABLE_TABLES {
        let env = Env::fixture(name);
        let table = env.open().unwrap();
        let ours = env.run(table.delta_scan().build(ScanPushdowns::default()).unwrap());
        let kernel = env.kernel_read(&table);
        let schema = table.schema();
        // Same columns, of the same Spark types (the kernel's Arrow forms are not the
        // engine's; its Variant carries no marker, so it is compared by its struct form).
        let kernel_schema = kernel.first().map(RecordBatch::schema);
        if let Some(ks) = &kernel_schema {
            assert_eq!(ks.fields().len(), schema.fields().len(), "{name}");
            for (k, o) in ks.fields().iter().zip(schema.fields()) {
                assert_eq!(k.name(), o.name(), "{name}");
                if o.data_type() != &krishiv_types::DataType::Variant.to_arrow() {
                    assert_eq!(
                        krishiv_types::DataType::from_arrow(k.data_type()).unwrap(),
                        krishiv_types::DataType::from_arrow(o.data_type()).unwrap(),
                        "{name}: {}",
                        k.name()
                    );
                }
            }
        }
        assert_eq!(rows(&ours, &schema), rows(&kernel, &schema), "{name}");
    }
}

#[test]
fn delta_type_widening_read() {
    let env = Env::fixture("type-widening");
    let table = env.open().unwrap();
    let schema = table.schema();
    assert_eq!(schema.field(1).data_type(), &ArrowType::Int64);
    assert_eq!(schema.field(2).data_type(), &ArrowType::Float64);
    // A file written before the widening holds the old types.
    let format = crate::engine::columnar::format().unwrap();
    let options = crate::engine::columnar::format_options().unwrap();
    let mut old = 0;
    for u in table.units() {
        let file = krishiv_io::FileRef {
            store: env.engine.store().clone(),
            path: env
                .engine
                .root()
                .confine(&env.engine.table_root().join(&u.path).unwrap())
                .unwrap(),
            size: u.size,
            version: krishiv_io::VersionToken::Immutable,
            modified_micros: None,
        };
        let meta = env
            .rt
            .block_on(env.engine.io().read_metadata(
                format.as_ref(),
                &file,
                krishiv_io::MetadataNeeds::default(),
                &options,
            ))
            .unwrap();
        let s = meta.schema();
        if s.field_with_name("i").unwrap().data_type() == &ArrowType::Int32 {
            assert_eq!(
                s.field_with_name("f").unwrap().data_type(),
                &ArrowType::Float32
            );
            old += 1;
        }
    }
    assert_eq!(old, 1, "one file in the pre-widening types");
    let got = rows(
        &env.run(table.delta_scan().build(ScanPushdowns::default()).unwrap()),
        &schema,
    );
    assert_eq!(got, spark_rows("type-widening").unwrap().1);
}

/// One pushed conjunct (`column op value`).
fn filter(expr: PushdownExpr) -> PushedFilter {
    PushedFilter::typed(
        Arc::new(krishiv_exec::FnPredicate(|b: &RecordBatch| {
            Ok(arrow_array::BooleanArray::from(vec![true; b.num_rows()]))
        })),
        expr,
    )
}

#[test]
fn a_pushed_filter_prunes_files_through_the_kernel() {
    let env = Env::fixture("partitioned");
    let table = env.open().unwrap();
    let scan = table.delta_scan();
    let by_partition = filter(PushdownExpr::eq(
        ColumnPath::new("n"),
        ScalarValue::Int32(2),
    ));
    assert_eq!(scan.filter_pushdown(&by_partition), FilterPushdown::Inexact);
    let contains = filter(PushdownExpr::Contains {
        column: ColumnPath::new("p"),
        needle: "a".to_owned(),
        collation: krishiv_io::Collation::Utf8Binary,
    });
    assert_eq!(scan.filter_pushdown(&contains), FilterPushdown::Unsupported);
    let pushdowns = ScanPushdowns {
        filters: vec![contains, by_partition],
        ..ScanPushdowns::default()
    };
    let source = scan.source(&pushdowns).unwrap();
    assert!(
        source.pushed_predicate().is_some(),
        "recorded for the read set"
    );
    let got = rows(&env.run(Arc::new(source)), &table.schema());
    // The kernel kept the one file of n = 2 (the filter above the scan does the rest).
    let ids: Vec<i64> = got.iter().map(|r| r[0].as_i64().unwrap()).collect();
    assert_eq!(ids, [4]);
}

#[test]
fn a_projection_reads_partition_and_data_columns_in_its_order() {
    let env = Env::fixture("column-mapping-name");
    let table = env.open().unwrap();
    let pushdowns = ScanPushdowns {
        projection: Some(vec![ColumnPath::new("p"), ColumnPath::new("id")]),
        ..ScanPushdowns::default()
    };
    let source = table.delta_scan().build(pushdowns).unwrap();
    let batches = env.run(source);
    let schema = batches[0].schema();
    let names: Vec<&String> = schema.fields().iter().map(|f| f.name()).collect();
    assert_eq!(names, ["p", "id"]);
    let mut got: Vec<(i64, Option<String>)> = batches
        .iter()
        .flat_map(|b| {
            let p = b.column(0).as_string_view().clone();
            let id = b.column(1).as_primitive::<Int64Type>().clone();
            (0..b.num_rows())
                .map(move |r| (id.value(r), (!p.is_null(r)).then(|| p.value(r).to_owned())))
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        [
            (1, Some("p1".to_owned())),
            (2, None),
            (3, Some("p1".to_owned()))
        ]
    );
}

#[test]
fn delta_unsupported_features_refused_on_oracle_tables() {
    let env = Env::fixture("variant-shredding");
    let e = env.open().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DeltaUnsupportedFeaturesForRead);
    assert!(
        e.message().contains("variantShredding-preview"),
        "{}",
        e.message()
    );
}

/// A protocol the oracle cannot write (section 3.5's rows delta-spark 4.4.1 refuses to create),
/// appended as a new commit to an oracle table.
fn with_commit(name: &str, actions: &[Value]) -> Env {
    let env = Env::fixture(name);
    let log = env.dir.join("_delta_log");
    let next = std::fs::read_dir(&log)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".json")?.parse::<u64>().ok()
        })
        .max()
        .unwrap()
        + 1;
    let text: String = actions.iter().map(|a| format!("{a}\n")).collect();
    std::fs::write(log.join(format!("{next:020}.json")), text).unwrap();
    env
}

fn protocol(reader: &[&str], writer: &[&str]) -> Value {
    json!({ "protocol": {
        "minReaderVersion": 3,
        "minWriterVersion": 7,
        "readerFeatures": reader,
        "writerFeatures": writer,
    }})
}

/// The first `metaData` action of the oracle table `name`.
fn metadata_of(name: &str) -> Value {
    std::fs::read_to_string(
        fixtures()
            .join(name)
            .join("_delta_log/00000000000000000000.json"),
    )
    .unwrap()
    .lines()
    .map(|l| serde_json::from_str::<Value>(l).unwrap())
    .find(|a| a.get("metaData").is_some())
    .unwrap()
}

#[test]
fn catalog_managed_read_refused() {
    // A valid catalog-managed protocol (it requires in-commit timestamps, enabled) reaches the
    // kernel's own refusal: no catalog version is given, so no stale log is ever read.
    let mut metadata = metadata_of("vacuum-protocol-check");
    metadata["metaData"]["configuration"]["delta.enableInCommitTimestamps"] = json!("true");
    let env = with_commit(
        "vacuum-protocol-check",
        &[
            json!({ "commitInfo": { "timestamp": 1_800_000_000_000_i64, "inCommitTimestamp": 1_800_000_000_000_i64, "operation": "SET TBLPROPERTIES" } }),
            protocol(
                &["vacuumProtocolCheck", "catalogManaged"],
                &["vacuumProtocolCheck", "catalogManaged", "inCommitTimestamp"],
            ),
            metadata,
        ],
    );
    let e = env.open().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(e.message().contains("catalog-managed"), "{}", e.message());
}

#[test]
fn edited_protocols_are_refused_with_their_classes() {
    let unknown = with_commit(
        "vacuum-protocol-check",
        &[protocol(
            &["vacuumProtocolCheck", "krishivProbeFeature"],
            &["vacuumProtocolCheck", "krishivProbeFeature"],
        )],
    );
    let e = unknown.open().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DeltaUnsupportedFeaturesForRead,
        "{e}"
    );
    assert!(
        e.message().contains("krishivProbeFeature"),
        "{}",
        e.message()
    );

    // Coordinated commits by configuration: the metadata of the table with one more key.
    let mut metadata = metadata_of("vacuum-protocol-check");
    metadata["metaData"]["configuration"]["delta.coordinatedCommits.commitCoordinator-preview"] =
        json!("dynamodb");
    let coordinated = with_commit("vacuum-protocol-check", &[metadata]);
    let e = coordinated.open().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(
        e.message().contains("coordinated commits"),
        "{}",
        e.message()
    );
}
