// SPDX-License-Identifier: Apache-2.0
//! The write path against the Spark 4.2.0 + delta-spark 4.4.1 oracle (`tests/fixtures/delta-4.4.1/`,
//! `gen_delta_fixtures.py /out writes`): `delta_stats_match_spark` (`write-stats`: each
//! oracle file's rows written again by Krishiv, its stats compared with the oracle's),
//! `write_modes_match_oracle` (`WRITES.json`: every step replayed on a Krishiv-created table, rows
//! and commitInfo compared), `delta_write_allowed_features` and `delta_write_refusals`
//! (`write-features/`), a round trip and the query memory pool after each write.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::{TableProvider, WriteMode, WriteRequest};
use krishiv_exec::{
    CancellationToken, ExecConfig, ExecNode, PipelineBuilder, QueryContext, QueryId, QueryRunner,
    ScanPushdowns, Sink, Source, TokioExecutor, WorkloadClass,
};
use krishiv_io::{IoConfig, IoScheduler, StoreId};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_plan::{Expr, Literal};
use serde_json::Value;
use url::Url;

use super::NewDeltaTable;
use crate::engine::pool::LogPool;
use crate::engine::{DeltaEngine, DeltaLimits};
use crate::table::DeltaTable;

fn fixtures() -> PathBuf {
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

struct Env {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    pub(super) rt: tokio::runtime::Runtime,
    query_pool: Arc<dyn MemoryPool>,
    /// The runtime-root pool the I/O scheduler attached.
    process_pool: Arc<dyn MemoryPool>,
    engine: DeltaEngine,
    /// The metrics of the last query run.
    metrics: std::sync::Mutex<Option<krishiv_exec::QueryMetrics>>,
}

impl Env {
    /// An empty table root.
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        std::fs::create_dir_all(&dir).unwrap();
        Self::at(tmp, dir)
    }

    /// An empty table root, the query's memory limited to `limit` bytes.
    fn with_limit(limit: usize) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        std::fs::create_dir_all(&dir).unwrap();
        let config = MemoryConfig {
            limit: Some(limit),
            quantum: 64 << 10,
            ..MemoryConfig::default()
        };
        Self::at_with(tmp, dir, &config)
    }

    /// A copy of the oracle table `name` (a path under the fixtures).
    fn fixture(name: &str) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        copy_dir(&fixtures().join(name), &dir);
        Self::at(tmp, dir)
    }

    fn at(tmp: tempfile::TempDir, dir: PathBuf) -> Self {
        Self::at_with(tmp, dir, &MemoryConfig::default())
    }

    fn at_with(tmp: tempfile::TempDir, dir: PathBuf, memory: &MemoryConfig) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let sched = IoScheduler::new(IoConfig::default());
        let id = StoreId::new("file");
        sched.register_whole_listing_store(id.clone(), Arc::new(crate::TestLocalStore::new()));
        let root = TreePool::new(memory);
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
            process_pool: process,
            engine,
            metrics: std::sync::Mutex::new(None),
        }
    }

    fn open_at(&self, version: Option<u64>) -> DeltaTable {
        self.rt
            .block_on(DeltaTable::open(
                self.engine.clone(),
                version,
                &Config::new(),
            ))
            .unwrap()
    }

    fn open(&self) -> Result<DeltaTable, KrishivError> {
        self.rt
            .block_on(DeltaTable::open(self.engine.clone(), None, &Config::new()))
    }

    /// Every row of `provider`'s scan.
    fn read(&self, provider: &dyn TableProvider) -> Vec<RecordBatch> {
        let source = provider
            .scan()
            .unwrap()
            .build(ScanPushdowns::default())
            .unwrap();
        self.run(Arc::new(Leaf(source))).unwrap()
    }

    /// Writes `batches` through `provider`'s writer for `mode`, checking the query's memory is
    /// all given back.
    fn write(
        &self,
        provider: &dyn TableProvider,
        mode: WriteMode,
        batches: Vec<RecordBatch>,
        options: &[(&str, &str)],
    ) -> Result<(), KrishivError> {
        self.write_as(provider, mode, batches, options, &super::files::file_uuid())
    }

    /// [`Self::write`] with the write id `write_id`.
    fn write_as(
        &self,
        provider: &dyn TableProvider,
        mode: WriteMode,
        batches: Vec<RecordBatch>,
        options: &[(&str, &str)],
        write_id: &str,
    ) -> Result<(), KrishivError> {
        let schema = batches[0].schema();
        let mut request = WriteRequest::new(mode, "`t`", Arc::clone(&schema));
        request.write_id = Some(write_id.to_owned());
        for (k, v) in options {
            request.options.insert((*k).to_owned(), (*v).to_owned());
        }
        let sink = provider.writer(&request)?;
        let input = krishiv_exec::planner::MemoryTable::new(
            schema,
            batches
                .into_iter()
                .map(krishiv_types::batch::Batch::new)
                .collect(),
        )
        .scan()?
        .build(ScanPushdowns::default())?;
        let out = self.run(Arc::new(Writing {
            input: Arc::new(Leaf(input)),
            sink,
        }));
        self.assert_memory_returned("the write");
        out.map(drop)
    }

    /// Asserts the query pool is back to 0 after `what` ran. An ended driver's row groups are
    /// shipped by a background flush (`release_combined`) that returns its memory a moment after
    /// the query finished: wait for it, but not forever.
    fn assert_memory_returned(&self, what: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while self.query_pool.reserved() != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(self.query_pool.reserved(), 0, "{what} gave its memory back");
    }

    #[expect(clippy::unwrap_in_result, reason = "test helper")]
    fn run(&self, node: Arc<dyn ExecNode>) -> Result<Vec<RecordBatch>, KrishivError> {
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
            let mut stream = runner.execute(node, Arc::clone(&query)).await?;
            let mut out = Vec::new();
            let mut first = None;
            while let Some(b) = stream.next().await {
                match b {
                    Ok(b) => out.push(b.into_parts().0),
                    Err(e) => {
                        first.get_or_insert(e);
                    }
                }
            }
            query.finished().await;
            *self.metrics.lock().unwrap() = Some(query.metrics());
            first.map_or(Ok(out), Err)
        })
    }

    /// The JSON actions of commit `version`.
    fn log(&self, version: u64) -> Vec<Value> {
        std::fs::read_to_string(self.dir.join(format!("_delta_log/{version:020}.json")))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn latest(&self) -> u64 {
        std::fs::read_dir(self.dir.join("_delta_log"))
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().into_string().unwrap();
                n.strip_suffix(".json")?.parse::<u64>().ok()
            })
            .max()
            .unwrap()
    }
}

struct Leaf(Arc<dyn Source>);

impl ExecNode for Leaf {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> krishiv_exec::Result<()> {
        b.set_source(Arc::clone(&self.0)).map(drop)
    }
}

/// `input` into `sink`, then nothing (as the executor's table write).
struct Writing {
    input: Arc<dyn ExecNode>,
    sink: Arc<dyn Sink>,
}

impl ExecNode for Writing {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> krishiv_exec::Result<()> {
        let sink = b.child_pipeline(Arc::clone(&self.sink), Arc::clone(&self.input))?;
        let empty = krishiv_exec::planner::MemoryTable::new(Arc::new(Schema::empty()), Vec::new())
            .scan()?
            .build(ScanPushdowns::default())?;
        b.set_breaker_source(empty, sink)?;
        Ok(())
    }
}

fn expected(path: &str) -> Value {
    serde_json::from_slice(&std::fs::read(fixtures().join(path)).unwrap()).unwrap()
}

/// The commitInfo and adds of a commit's actions.
fn commit_info(actions: &[Value]) -> Value {
    actions
        .iter()
        .find_map(|a| a.get("commitInfo").cloned())
        .unwrap()
}

fn adds(actions: &[Value]) -> Vec<Value> {
    actions
        .iter()
        .filter_map(|a| a.get("add").cloned())
        .collect()
}

fn removes(actions: &[Value]) -> usize {
    actions.iter().filter(|a| a.get("remove").is_some()).count()
}

/// The rows of `batches` whose `id` (column 0) is not in `seen`.
fn new_rows(batches: &[RecordBatch], seen: &[i64]) -> Vec<RecordBatch> {
    batches
        .iter()
        .map(|b| {
            let ids = arrow_cast::cast(b.column(0), &ArrowType::Int64).unwrap();
            let ids = ids.as_primitive::<Int64Type>();
            let keep: arrow_array::BooleanArray = (0..b.num_rows())
                .map(|i| Some(!seen.contains(&ids.value(i))))
                .collect();
            arrow_select::filter::filter_record_batch(b, &keep).unwrap()
        })
        .filter(|b| b.num_rows() > 0)
        .collect()
}

fn ids(batches: &[RecordBatch]) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        let ids = arrow_cast::cast(b.column(0), &ArrowType::Int64).unwrap();
        out.extend(ids.as_primitive::<Int64Type>().iter().flatten());
    }
    out.sort_unstable();
    out
}

#[test]
fn delta_stats_match_spark() {
    let oracle = Env::fixture("write-stats");
    let expected = expected("write-stats/EXPECTED.json");
    // Krishiv's copy: the oracle table's create commit only.
    let ours = Env::new();
    std::fs::create_dir_all(ours.dir.join("_delta_log")).unwrap();
    std::fs::copy(
        oracle.dir.join("_delta_log/00000000000000000000.json"),
        ours.dir.join("_delta_log/00000000000000000000.json"),
    )
    .unwrap();
    let commits = expected["commits"].as_array().unwrap();
    let mut seen = Vec::new();
    let mut checked = 0;
    for c in commits {
        let version = c["version"].as_u64().unwrap();
        if version == 0 {
            continue;
        }
        let rows = oracle.read(&oracle.open_at(Some(version)));
        let fresh = new_rows(&rows, &seen);
        seen = ids(&rows);
        ours.write(&ours.open().unwrap(), WriteMode::Append, fresh, &[])
            .unwrap();
        let actions = ours.log(ours.latest());
        let ours_adds = adds(&actions);
        let theirs = c["adds"].as_array().unwrap();
        assert_eq!(ours_adds.len(), theirs.len(), "version {version}");
        let stats: Value = serde_json::from_str(ours_adds[0]["stats"].as_str().unwrap()).unwrap();
        assert_eq!(stats, theirs[0]["stats"], "the stats of version {version}");
        assert_eq!(
            ours_adds[0]["partitionValues"], theirs[0]["partitionValues"],
            "version {version}"
        );
        checked += 1;
    }
    assert_eq!(checked, commits.len() - 1);
}

fn modes_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("v", ArrowType::Utf8, true),
        Field::new("p", ArrowType::Utf8, true),
    ]))
}

fn rows(values: &[(i64, &str, Option<&str>)]) -> Vec<RecordBatch> {
    let id: ArrayRef = Arc::new(Int64Array::from(
        values.iter().map(|r| r.0).collect::<Vec<_>>(),
    ));
    let v: ArrayRef = Arc::new(StringArray::from(
        values.iter().map(|r| Some(r.1)).collect::<Vec<_>>(),
    ));
    let p: ArrayRef = Arc::new(StringArray::from(
        values.iter().map(|r| r.2).collect::<Vec<_>>(),
    ));
    vec![RecordBatch::try_new(modes_schema(), vec![id, v, p]).unwrap()]
}

/// The table's rows as the oracle's `typed` JSON, by `id`.
fn typed_rows(batches: &[RecordBatch]) -> Vec<Value> {
    let mut out = Vec::new();
    for b in batches {
        let b = crate::write::files::conform(b, &modes_schema()).unwrap();
        let id = b.column(0).as_primitive::<Int64Type>();
        let v = b.column(1).as_string::<i32>();
        let p = b.column(2).as_string::<i32>();
        for i in 0..b.num_rows() {
            let s = |a: &StringArray| {
                if a.is_null(i) {
                    Value::Null
                } else {
                    Value::from(a.value(i))
                }
            };
            out.push(Value::Array(vec![Value::from(id.value(i)), s(v), s(p)]));
        }
    }
    out.sort_by_key(|r| r[0].as_i64());
    out
}

fn eq(column: &str, value: &str, null_safe: bool) -> Arc<Expr> {
    Arc::new(Expr::function(
        if null_safe { "<=>" } else { "==" },
        vec![
            Expr::column([column]),
            Expr::literal(Literal::String(value.to_owned())),
        ],
    ))
}

/// The oracle's commitInfo fields Krishiv writes the same: the operation, its parameters and
/// its metrics' names and counts (not bytes).
fn compare_commit(step: &str, ours: &Value, theirs: &Value) {
    assert_eq!(ours["operation"], theirs["operation"], "{step}");
    let mut want = theirs["operationParameters"].clone();
    if let Some(d) = want.get_mut("isDynamicPartitionOverwrite") {
        // A recorded difference: the kernel's parameters are strings.
        *d = Value::from("true");
    }
    assert_eq!(ours["operationParameters"], want, "{step}");
    let keys = |v: &Value| {
        let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
        k.sort();
        k
    };
    let theirs_m = &theirs["operationMetrics"];
    let ours_m = &ours["operationMetrics"];
    let mut want_keys = keys(theirs_m);
    // `numDeletedRows`: delta-spark counts the removed files' rows from their stats; a
    // partition filter's removes are whole files (left out, a recorded difference).
    want_keys.retain(|k| k != "numDeletedRows");
    assert_eq!(keys(ours_m), want_keys, "{step}");
    for k in ["numFiles", "numOutputRows", "numRemovedFiles"] {
        assert_eq!(ours_m.get(k), theirs_m.get(k), "{step} {k}");
    }
    // The kernel writes `isBlindAppend` only when true (a recorded difference).
    let blind = |v: &Value| v.as_bool().unwrap_or(false);
    assert_eq!(
        blind(&ours["isBlindAppend"]),
        blind(&theirs["isBlindAppend"]),
        "{step}"
    );
    assert_eq!(ours["isolationLevel"], theirs["isolationLevel"], "{step}");
    assert_eq!(ours["readVersion"], theirs["readVersion"], "{step}");
}

#[test]
fn write_modes_match_oracle() {
    let env = Env::new();
    let oracle = expected("WRITES.json");
    let steps = oracle["steps"].as_array().unwrap();
    let step = |name: &str| {
        steps
            .iter()
            .find(|s| s["step"] == name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    let check_rows = |name: &str| {
        let table = env.open().unwrap();
        assert_eq!(
            Value::Array(typed_rows(&env.read(&table))),
            step(name)["rows"],
            "{name}"
        );
    };
    let check_commit = |name: &str| {
        let ours = commit_info(&env.log(env.latest()));
        let theirs = &step(name)["commits"][0];
        compare_commit(name, &ours, &theirs["commitInfo"]);
        assert_eq!(
            removes(&env.log(env.latest())),
            theirs["removes"].as_array().unwrap().len(),
            "{name}"
        );
        assert_eq!(
            adds(&env.log(env.latest())).len(),
            theirs["adds"].as_array().unwrap().len(),
            "{name}"
        );
    };

    // CTAS: version 0 with the first rows.
    let new = NewDeltaTable::new(
        env.engine.clone(),
        &modes_schema(),
        vec!["p".to_owned()],
        std::collections::BTreeMap::new(),
        None,
    )
    .unwrap();
    env.write(
        &new,
        WriteMode::Append,
        rows(&[(1, "a", Some("x")), (2, "b", Some("y"))]),
        &[],
    )
    .unwrap();
    assert_eq!(env.latest(), 0);
    check_rows("ctas");
    let v0 = env.log(0);
    assert_eq!(adds(&v0).len(), 2);
    let protocol = v0.iter().find_map(|a| a.get("protocol")).unwrap();
    // The kernel's table-features protocol (design section 2.1, a recorded difference).
    assert_eq!(protocol["minWriterVersion"], 7, "{protocol}");

    let append = |values: &[(i64, &str, Option<&str>)]| {
        env.write(&env.open().unwrap(), WriteMode::Append, rows(values), &[])
    };
    append(&[(3, "c", Some("x")), (4, "d", None)]).unwrap();
    check_rows("append");
    check_commit("append");

    env.write(
        &env.open().unwrap(),
        WriteMode::Truncate,
        rows(&[(5, "e", Some("x")), (6, "f", Some("y"))]),
        &[],
    )
    .unwrap();
    check_rows("overwrite");
    check_commit("overwrite");

    append(&[(7, "g", Some("y")), (8, "h", Some("z"))]).unwrap();
    check_rows("append_again");
    check_commit("append_again");

    env.write(
        &env.open().unwrap(),
        WriteMode::OverwriteDynamic,
        rows(&[(20, "dyn", Some("x")), (21, "dyn", Some("z"))]),
        &[],
    )
    .unwrap();
    check_rows("dynamic_overwrite");
    check_commit("dynamic_overwrite");

    env.write(
        &env.open().unwrap(),
        WriteMode::OverwriteByFilter(eq("p", "y", false)),
        rows(&[(30, "rw", Some("y"))]),
        &[(super::PREDICATE_TEXT_OPTION, "p = 'y'")],
    )
    .unwrap();
    check_rows("replace_where");
    check_commit("replace_where");

    let before = env.latest();
    let e = env
        .write(
            &env.open().unwrap(),
            WriteMode::OverwriteByFilter(eq("p", "y", false)),
            rows(&[(31, "rw", Some("x"))]),
            &[(super::PREDICATE_TEXT_OPTION, "p = 'y'")],
        )
        .unwrap_err();
    let theirs = &step("replace_where_mismatch")["error"];
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap(),
        "{e}"
    );
    assert_eq!(e.sql_state(), theirs["sqlstate"].as_str());
    assert!(
        e.to_string().contains(
            "Written data does not conform to partial table overwrite condition or constraint \
             'p = 'y''."
        ),
        "{e}"
    );
    assert!(e.to_string().contains(" - p : x"), "{e}");
    assert_eq!(env.latest(), before, "nothing committed");
    check_rows("replace_where_mismatch");

    env.write(
        &env.open().unwrap(),
        WriteMode::OverwriteByFilter(eq("p", "z", true)),
        rows(&[(40, "st", Some("z"))]),
        &[(super::PREDICATE_TEXT_OPTION, "(p <=> 'z')")],
    )
    .unwrap();
    check_rows("static_partition_overwrite");
    check_commit("static_partition_overwrite");

    env.write(
        &env.open().unwrap(),
        WriteMode::Truncate,
        rows(&[(50, "df", Some("x"))]),
        &[],
    )
    .unwrap();
    check_rows("insert_overwrite_dataframe_mode");
    check_commit("insert_overwrite_dataframe_mode");

    append(&[(51, "df", Some("w"))]).unwrap();
    check_rows("append_dataframe");
    check_commit("append_dataframe");
}

#[test]
fn a_filter_on_a_data_column_is_refused_before_anything_is_written() {
    let env = Env::new();
    let new = NewDeltaTable::new(
        env.engine.clone(),
        &modes_schema(),
        vec!["p".to_owned()],
        std::collections::BTreeMap::new(),
        None,
    )
    .unwrap();
    env.write(&new, WriteMode::Append, rows(&[(1, "a", Some("x"))]), &[])
        .unwrap();
    let e = env
        .write(
            &env.open().unwrap(),
            WriteMode::OverwriteByFilter(eq("v", "a", false)),
            rows(&[(2, "a", Some("x"))]),
            &[],
        )
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation
    );
    assert!(e.to_string().contains("data column v"), "{e}");
    assert_eq!(env.latest(), 0);
}

#[test]
fn an_overwrite_of_an_append_only_table_fails_as_delta_spark() {
    let oracle = expected("WRITES.json");
    let theirs = &oracle["append_only"][1]["error"];
    let env = Env::fixture("write-features/append-only");
    let table = env.open().unwrap();
    let schema = table.schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![3_i64]))],
    )
    .unwrap();
    let e = env
        .write(&table, WriteMode::Truncate, vec![batch.clone()], &[])
        .unwrap_err();
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap(),
        "{e}"
    );
    assert!(
        e.to_string().contains(theirs["message"].as_str().unwrap()),
        "{e}"
    );
    // Appends are fine.
    env.write(&table, WriteMode::Append, vec![batch], &[])
        .unwrap();
}

/// One row (`id` = 2, other columns NULL) in `table`'s columns.
fn one_row(table: &DeltaTable) -> RecordBatch {
    let schema = table.schema();
    let columns = schema
        .fields()
        .iter()
        .map(|f| {
            if f.name() == "id" {
                Arc::new(Int64Array::from(vec![2_i64])) as ArrayRef
            } else {
                arrow_array::new_null_array(f.data_type(), 1)
            }
        })
        .collect();
    RecordBatch::try_new(schema, columns).unwrap()
}

#[test]
fn delta_write_allowed_features() {
    let features = expected("write-features/FEATURES.json");
    let mut written = Vec::new();
    for (name, t) in features["tables"].as_object().unwrap() {
        if t["refused"].as_bool().unwrap() {
            continue;
        }
        let env = Env::fixture(&format!("write-features/{name}"));
        let table = env.open().unwrap();
        env.write(&table, WriteMode::Append, vec![one_row(&table)], &[])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        // The oracle's row and Krishiv's, read back by Krishiv.
        let back = env.read(&env.open().unwrap());
        assert_eq!(ids(&back), vec![1, 2], "{name}");
        written.push(name.clone());
    }
    assert!(written.len() >= 10, "{written:?}");
}

#[test]
fn delta_write_refusals() {
    let features = expected("write-features/FEATURES.json");
    let mut refused = Vec::new();
    for (name, t) in features["tables"].as_object().unwrap() {
        if !t["refused"].as_bool().unwrap() {
            continue;
        }
        let env = Env::fixture(&format!("write-features/{name}"));
        let table = env.open().unwrap();
        let files_before = std::fs::read_dir(&env.dir).unwrap().count();
        let e = env
            .write(&table, WriteMode::Append, vec![one_row(&table)], &[])
            .unwrap_err();
        let writer = t["protocol"]["minWriterVersion"].as_i64().unwrap();
        let want = if writer == 7 {
            ErrorClass::DeltaUnsupportedFeaturesForWrite
        } else {
            ErrorClass::DeltaInvalidProtocolVersion
        };
        assert_eq!(e.error_class(), want, "{name}: {e}");
        assert_eq!(
            std::fs::read_dir(&env.dir).unwrap().count(),
            files_before,
            "{name}: nothing written"
        );
        refused.push(name.clone());
    }
    assert!(refused.len() >= 10, "{refused:?}");
}

/// An empty-string partition value is stored as null, as delta-spark does: a dynamic overwrite
/// of `''` replaces the null partition, an append of `''` lands there, and `replaceWhere`
/// checks the stored value (`WRITE-EDGES.json` `empty_partition`).
#[test]
fn an_empty_partition_value_is_the_null_partition() {
    let oracle = expected("WRITE-EDGES.json");
    let steps = oracle["empty_partition"].as_array().unwrap();
    let step = |name: &str| {
        steps
            .iter()
            .find(|s| s["step"] == name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    let env = Env::new();
    let check_rows = |name: &str| {
        assert_eq!(
            Value::Array(typed_rows(&env.read(&env.open().unwrap()))),
            step(name)["rows"],
            "{name}"
        );
    };
    let new = NewDeltaTable::new(
        env.engine.clone(),
        &modes_schema(),
        vec!["p".to_owned()],
        std::collections::BTreeMap::new(),
        None,
    )
    .unwrap();
    env.write(
        &new,
        WriteMode::Append,
        rows(&[(1, "a", None), (2, "b", Some("x"))]),
        &[],
    )
    .unwrap();

    env.write(
        &env.open().unwrap(),
        WriteMode::OverwriteDynamic,
        rows(&[(3, "c", Some(""))]),
        &[],
    )
    .unwrap();
    check_rows("dynamic_overwrite_empty");
    let theirs = &step("dynamic_overwrite_empty")["commits"][0];
    assert_eq!(
        removes(&env.log(env.latest())),
        theirs["removes"].as_array().unwrap().len()
    );
    assert_eq!(
        adds(&env.log(env.latest()))[0]["partitionValues"]["p"],
        Value::Null
    );

    env.write(
        &env.open().unwrap(),
        WriteMode::Append,
        rows(&[(4, "d", Some("")), (5, "e", None)]),
        &[],
    )
    .unwrap();
    check_rows("append_empty");
    // `''` and NULL rows share one file: one partition.
    assert_eq!(adds(&env.log(env.latest())).len(), 1);

    let before = env.latest();
    let e = env
        .write(
            &env.open().unwrap(),
            WriteMode::OverwriteByFilter(eq("p", "", false)),
            rows(&[(6, "f", Some(""))]),
            &[],
        )
        .unwrap_err();
    let theirs = &step("replace_where_empty")["error"];
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap(),
        "{e}"
    );
    assert_eq!(env.latest(), before);
    check_rows("replace_where_empty");

    let is_null = Arc::new(Expr::function("isnull", vec![Expr::column(["p"])]));
    env.write(
        &env.open().unwrap(),
        WriteMode::OverwriteByFilter(is_null),
        rows(&[(7, "g", Some(""))]),
        &[],
    )
    .unwrap();
    check_rows("replace_where_null_with_empty");
}

/// A NULL written into a `NOT NULL` column, at the top level, in a nested struct field, or as
/// a NULL struct above one, fails as delta-spark does (`WRITE-EDGES.json` `not_null`), with
/// rows that come in nullable (a DataFrame's), and writes nothing.
#[test]
fn a_null_in_a_not_null_column_fails_as_delta_spark() {
    use arrow_array::{Int32Array, StructArray};
    let oracle = expected("WRITE-EDGES.json");
    let steps = oracle["not_null"].as_array().unwrap();
    let step = |name: &str| {
        &steps
            .iter()
            .find(|s| s["step"] == name)
            .unwrap_or_else(|| panic!("{name}"))["error"]
    };
    let s_fields = |strict: bool| {
        arrow_schema::Fields::from(vec![
            Field::new("x", ArrowType::Int32, !strict),
            Field::new("y", ArrowType::Int32, true),
        ])
    };
    let table_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("a", ArrowType::Int32, false),
        Field::new("s", ArrowType::Struct(s_fields(true)), true),
    ]));
    let loose: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("a", ArrowType::Int32, true),
        Field::new("s", ArrowType::Struct(s_fields(false)), true),
    ]));
    let batch = |id: i64, a: Option<i32>, x: Option<i32>, s_null: bool| {
        let s = StructArray::try_new(
            s_fields(false),
            vec![
                Arc::new(Int32Array::from(vec![x])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(1)])),
            ],
            s_null.then(|| arrow_buffer::NullBuffer::from(vec![false])),
        )
        .unwrap();
        RecordBatch::try_new(
            Arc::clone(&loose),
            vec![
                Arc::new(Int64Array::from(vec![id])),
                Arc::new(Int32Array::from(vec![a])),
                Arc::new(s),
            ],
        )
        .unwrap()
    };
    let env = Env::new();
    let new = NewDeltaTable::new(
        env.engine.clone(),
        &table_schema,
        Vec::new(),
        std::collections::BTreeMap::new(),
        None,
    )
    .unwrap();
    env.write(
        &new,
        WriteMode::Append,
        vec![batch(1, Some(1), Some(1), false)],
        &[],
    )
    .unwrap();
    for (name, row) in [
        ("dataframe_null_top_level", batch(2, None, Some(2), false)),
        ("dataframe_null_nested", batch(3, Some(3), None, false)),
        ("dataframe_null_struct", batch(4, Some(4), Some(4), true)),
    ] {
        let theirs = step(name);
        let e = env
            .write(&env.open().unwrap(), WriteMode::Append, vec![row], &[])
            .unwrap_err();
        assert_eq!(
            e.error_class().name(),
            theirs["condition"].as_str().unwrap(),
            "{name}: {e}"
        );
        assert_eq!(e.sql_state(), theirs["sqlstate"].as_str(), "{name}");
        assert!(
            e.to_string().contains(theirs["message"].as_str().unwrap()),
            "{name}: {e}"
        );
        assert_eq!(env.latest(), 0, "{name}: nothing committed");
    }
}

/// An empty string written into a `NOT NULL` partition column is stored as NULL, so it fails
/// the NOT NULL check as delta-spark's does (`WRITE-EDGES.json` `not_null_partition`): the
/// check sees the stored value, never an internal error, and nothing is committed.
#[test]
fn an_empty_value_in_a_not_null_partition_column_fails_as_delta_spark() {
    let oracle = expected("WRITE-EDGES.json");
    let steps = oracle["not_null_partition"].as_array().unwrap();
    let table_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("p", ArrowType::Utf8, false),
    ]));
    let loose: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("p", ArrowType::Utf8, true),
    ]));
    let batch = |id: i64, p: &str| {
        RecordBatch::try_new(
            Arc::clone(&loose),
            vec![
                Arc::new(Int64Array::from(vec![id])) as ArrayRef,
                Arc::new(StringArray::from(vec![p])),
            ],
        )
        .unwrap()
    };
    let env = Env::new();
    let new = NewDeltaTable::new(
        env.engine.clone(),
        &table_schema,
        vec!["p".to_owned()],
        std::collections::BTreeMap::new(),
        None,
    )
    .unwrap();
    env.write(&new, WriteMode::Append, vec![batch(1, "x")], &[])
        .unwrap();
    for step in steps {
        let name = step["step"].as_str().unwrap();
        let theirs = &step["error"];
        let e = env
            .write(
                &env.open().unwrap(),
                WriteMode::Append,
                vec![batch(2, "")],
                &[],
            )
            .unwrap_err();
        assert_eq!(
            e.error_class().name(),
            theirs["condition"].as_str().unwrap(),
            "{name}: {e}"
        );
        assert_eq!(e.sql_state(), theirs["sqlstate"].as_str(), "{name}");
        assert!(
            e.to_string().contains(theirs["message"].as_str().unwrap()),
            "{name}: {e}"
        );
        assert_eq!(env.latest(), 0, "{name}: nothing committed");
    }
}

/// A create (CTAS and an empty `CREATE TABLE`) under a feature the kernel's create admits but
/// Krishiv's writer does not honour (design section 4.2) is refused before anything is written.
#[test]
fn a_create_under_a_refused_feature_writes_nothing() {
    let cases: &[(&str, &str, &str)] = &[
        ("delta.feature.rowTracking", "supported", "rowTracking"),
        ("delta.enableRowTracking", "true", "rowTracking"),
        ("delta.feature.variantType", "supported", "variantType"),
        (
            "delta.feature.variantShredding",
            "supported",
            "variantShredding",
        ),
        (
            "delta.feature.materializePartitionColumns",
            "supported",
            "materializePartitionColumns",
        ),
        (
            "delta.feature.icebergCompatV2",
            "supported",
            "icebergCompatV2",
        ),
        ("delta.enableIcebergCompatV2", "true", "icebergCompatV2"),
        (
            "delta.feature.icebergCompatV3",
            "supported",
            "icebergCompatV3",
        ),
        ("delta.enableIcebergCompatV3", "true", "icebergCompatV3"),
        (
            "delta.feature.catalogManaged",
            "supported",
            "catalogManaged",
        ),
    ];
    for (key, value, feature) in cases {
        for ctas in [true, false] {
            let env = Env::new();
            let properties =
                std::collections::BTreeMap::from([((*key).to_owned(), (*value).to_owned())]);
            let new = NewDeltaTable::new(
                env.engine.clone(),
                &modes_schema(),
                vec!["p".to_owned()],
                properties,
                None,
            )
            .unwrap();
            let e = if ctas {
                env.write(&new, WriteMode::Append, rows(&[(1, "a", Some("x"))]), &[])
                    .unwrap_err()
            } else {
                env.rt.block_on(new.create_empty("w")).unwrap_err()
            };
            assert_eq!(
                e.error_class(),
                ErrorClass::DeltaUnsupportedFeaturesForWrite,
                "{key} (ctas {ctas}): {e}"
            );
            assert_eq!(e.sql_state(), Some("56038"));
            assert!(e.to_string().contains(feature), "{key}: {e}");
            if *key == "delta.feature.variantShredding" {
                // The kernel's create enables Variant support beside shredding.
                assert!(e.to_string().contains("variantType"), "{key}: {e}");
            }
            assert_eq!(
                std::fs::read_dir(&env.dir).unwrap().count(),
                0,
                "{key} (ctas {ctas}): nothing written"
            );
        }
    }
    // A feature Krishiv writes is still created.
    let env = Env::new();
    let new = NewDeltaTable::new(
        env.engine.clone(),
        &modes_schema(),
        vec!["p".to_owned()],
        std::collections::BTreeMap::from([(
            "delta.feature.deletionVectors".to_owned(),
            "supported".to_owned(),
        )]),
        None,
    )
    .unwrap();
    env.rt.block_on(new.create_empty("w")).unwrap();
    assert_eq!(env.latest(), 0);
}

/// Every field of `logical` is in `written` under its physical name with its column-mapping id
/// as the Parquet field id, at every level: struct in struct, and the structs inside array
/// elements and map keys and values.
fn check_field_ids(
    logical: &delta_kernel::schema::StructType,
    written: &delta_kernel::schema::StructType,
    path: &str,
    checked: &mut Vec<String>,
) {
    use delta_kernel::schema::{ColumnMetadataKey, MetadataValue};
    use delta_kernel::table_features::ColumnMappingMode;
    let num = |v: Option<&MetadataValue>| match v {
        Some(MetadataValue::Number(n)) => Some(*n),
        Some(MetadataValue::String(s)) => s.parse().ok(),
        _ => None,
    };
    for field in logical.fields() {
        let name = format!("{path}{}", field.name());
        let physical = field.physical_name(ColumnMappingMode::Name);
        assert_ne!(physical, field.name(), "{name} has a physical name");
        let w = written
            .field(physical)
            .unwrap_or_else(|| panic!("{name} as {physical} in {written:?}"));
        let want = num(field.get_config_value(&ColumnMetadataKey::ColumnMappingId));
        assert!(want.is_some(), "{name}: {field:?}");
        assert_eq!(
            num(w.get_config_value(&ColumnMetadataKey::ParquetFieldId)),
            want,
            "{name}"
        );
        checked.push(name.clone());
        check_nested_ids(
            field.data_type(),
            w.data_type(),
            &format!("{name}."),
            checked,
        );
    }
}

fn check_nested_ids(
    logical: &delta_kernel::schema::DataType,
    written: &delta_kernel::schema::DataType,
    path: &str,
    checked: &mut Vec<String>,
) {
    use delta_kernel::schema::DataType;
    match (logical, written) {
        (DataType::Struct(l), DataType::Struct(w)) => check_field_ids(l, w, path, checked),
        (DataType::Array(l), DataType::Array(w)) => {
            check_nested_ids(
                l.element_type(),
                w.element_type(),
                &format!("{path}element."),
                checked,
            );
        }
        (DataType::Map(l), DataType::Map(w)) => {
            check_nested_ids(l.key_type(), w.key_type(), &format!("{path}key."), checked);
            check_nested_ids(
                l.value_type(),
                w.value_type(),
                &format!("{path}value."),
                checked,
            );
        }
        _ => {}
    }
}

/// The kernel schema of the footer of the data file at `path` (relative to the table root).
fn footer_schema(env: &Env, path: &str) -> delta_kernel::schema::SchemaRef {
    use delta_kernel::ParquetHandler as _;
    let path = env.dir.join(path);
    let size = std::fs::metadata(&path).unwrap().len();
    let meta = delta_kernel::FileMeta {
        location: Url::from_file_path(&path).unwrap(),
        last_modified: 0,
        size,
    };
    env.rt
        .block_on(
            env.engine
                .delta_blocking(move |e| e.parquet().read_parquet_footer(&meta)),
        )
        .unwrap()
        .unwrap()
        .schema
}

#[test]
fn column_mapping_writes_physical_names_and_field_ids() {
    let env = Env::fixture("write-features/column-mapping");
    let table = env.open().unwrap();
    env.write(&table, WriteMode::Append, vec![one_row(&table)], &[])
        .unwrap();
    let add = &adds(&env.log(env.latest()))[0];
    let footer = footer_schema(&env, add["path"].as_str().unwrap());
    let schema = env.open().unwrap().snapshot().schema();
    let mut checked = Vec::new();
    check_field_ids(&schema, &footer, "", &mut checked);
    // `id`, `s` and the nested `s.x`, each with its own id and physical name.
    assert!(checked.len() >= 3, "{checked:?}");
    assert!(checked.iter().any(|p| p == "s.x"), "{checked:?}");
    // Krishiv reads its own file back under the table's logical names.
    assert_eq!(ids(&env.read(&env.open().unwrap())), vec![1, 2]);
}

mod dml;
mod lifecycle;
mod merge;
mod readback;
mod streaming;
