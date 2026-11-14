// SPDX-License-Identifier: Apache-2.0
//! Tests of the append sink (`sink`, PR-36a): appends through the executor
//! into an unpartitioned and a partitioned table, committed by a catalog that
//! applies each request as `append_tests` does and read back through SQL; a
//! second append on top; the write id in the snapshot summary; a failing
//! driver and a cancelled write leaving no file, no commit and no memory; the
//! capability check without a committer; a schema mismatch refused before
//! anything is written.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as A, Field, Schema, SchemaRef};
use async_trait::async_trait;
use futures::StreamExt;
use futures::future::BoxFuture;
use iceberg::spec::{
    NestedField, PartitionSpec, PrimitiveType, TableMetadata, TableMetadataBuilder, Transform, Type,
};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::{TableProvider, WriteMode, WriteRequest};
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, ExecutorConfig, GlobalState, LocalState, PipelineBuilder, QueryContext,
    QueryId, QueryRunner, Sink, Source, TaskContext, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::Batch;

use super::append::{QUERY_ID_PROP, WRITE_ID_PROP};
use super::commit::{CommitOutcome, CommitTable};
use super::provider::IcebergTableProvider;
use super::provider_tests::{Env, env, ixn_schema, provider_session};
use super::table::IcebergTable;
use crate::plan::VersionSpec;

/// A catalog holding one table: each request's requirements checked, its
/// updates applied and the new metadata file written.
struct Catalog {
    state: Mutex<(TableMetadata, String, u32)>,
    /// A request never answers (it is in flight forever) when set.
    hang: AtomicBool,
    /// A request was sent.
    sent: AtomicBool,
}

impl Catalog {
    fn new(table: &iceberg::table::Table) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new((
                table.metadata().clone(),
                table.metadata_location().unwrap().to_owned(),
                0,
            )),
            hang: AtomicBool::new(false),
            sent: AtomicBool::new(false),
        })
    }

    fn location(&self) -> String {
        self.state.lock().unwrap().1.clone()
    }

    fn metadata(&self) -> TableMetadata {
        self.state.lock().unwrap().0.clone()
    }
}

#[async_trait]
impl CommitTable for Catalog {
    fn name(&self) -> String {
        "db.t".to_owned()
    }

    async fn commit(
        &self,
        requirements: Vec<TableRequirement>,
        updates: Vec<TableUpdate>,
    ) -> CommitOutcome {
        self.sent.store(true, Ordering::SeqCst);
        if self.hang.load(Ordering::SeqCst) {
            return std::future::pending().await;
        }
        let mut s = self.state.lock().unwrap();
        if requirements.iter().any(|r| r.check(Some(&s.0)).is_err()) {
            return CommitOutcome::Conflict { stale: false };
        }
        let mut b = TableMetadataBuilder::new_from_metadata(s.0.clone(), Some(s.1.clone()));
        for u in updates {
            b = u.apply(b).unwrap();
        }
        let next = b.build().unwrap().metadata;
        s.2 += 1;
        let path = format!("{}/metadata/v{}-sink.metadata.json", next.location(), s.2);
        std::fs::write(
            path.strip_prefix("file://").unwrap(),
            serde_json::to_vec(&next).unwrap(),
        )
        .unwrap();
        s.0 = next.clone();
        s.1 = path;
        CommitOutcome::Committed(Box::new(next))
    }

    async fn load(&self, _all_snapshots: bool) -> Result<TableMetadata, KrishivError> {
        Ok(self.metadata())
    }
}

/// The source side of the test's write node: no rows.
struct Nothing;

impl Source for Nothing {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::empty())
    }
    fn global_state(&self, _ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(()))
    }
    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }
    fn local_state(
        &self,
        _ctx: &TaskContext,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        Ok(Box::new(()))
    }
    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, krishiv_exec::Result<Option<Batch>>> {
        Box::pin(std::future::ready(Ok(None)))
    }
}

/// `input` written through `sink`.
struct Write {
    input: Arc<dyn ExecNode>,
    sink: Arc<dyn Sink>,
}

impl std::fmt::Debug for Write {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Write")
    }
}

impl ExecNode for Write {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> krishiv_exec::Result<()> {
        let sink = b.child_pipeline(Arc::clone(&self.sink), Arc::clone(&self.input))?;
        b.set_breaker_source(Arc::new(Nothing), sink)?;
        Ok(())
    }
}

/// A source of `batches`, one per driver (so as many drivers as batches),
/// that never ends after them when `stall`.
struct PerDriver {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    stall: bool,
}

impl Source for PerDriver {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
    fn global_state(&self, _ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(std::sync::atomic::AtomicUsize::new(0)))
    }
    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        self.batches.len()
    }
    fn local_state(
        &self,
        _ctx: &TaskContext,
        global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        let next = global
            .downcast_ref::<std::sync::atomic::AtomicUsize>()
            .unwrap()
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Box::new(Some(next)))
    }
    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, krishiv_exec::Result<Option<Batch>>> {
        let mine = local.downcast_mut::<Option<usize>>().unwrap();
        match mine.take() {
            Some(i) => Box::pin(std::future::ready(Ok(Some(Batch::new(
                self.batches[i].clone(),
            ))))),
            None if self.stall => Box::pin(std::future::pending()),
            None => Box::pin(std::future::ready(Ok(None))),
        }
    }
}

/// The schema of the rows written into [`ixn_schema`] tables.
fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", A::Int64, true),
        Field::new("x", A::Int32, true),
        Field::new("name", krishiv_types::DataType::String.to_arrow(), true),
    ]))
}

/// Rows `ids` with `x = id % 3` and `name = n<id>` (a `None` id stays null).
fn rows(ids: &[Option<i64>]) -> RecordBatch {
    let names: Vec<String> = ids
        .iter()
        .map(|i| format!("n{}", i.unwrap_or(-1)))
        .collect();
    let names: ArrayRef = match krishiv_types::DataType::String.to_arrow() {
        A::Utf8View => Arc::new(arrow_array::StringViewArray::from(names)),
        _ => Arc::new(arrow_array::StringArray::from(names)),
    };
    RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(Int32Array::from(
                ids.iter()
                    .map(|i| i.map_or(0, |i| (i % 3) as i32))
                    .collect::<Vec<_>>(),
            )),
            names,
        ],
    )
    .unwrap()
}

fn range(from: i64, to: i64) -> RecordBatch {
    rows(&(from..to).map(Some).collect::<Vec<_>>())
}

/// A table `t` of `schema`, partitioned by `identity(x)` when `partitioned`,
/// with `props`.
fn create(
    env: &Env,
    schema: iceberg::spec::Schema,
    partitioned: bool,
    props: &[(&str, &str)],
) -> iceberg::table::Table {
    let props = props
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect::<std::collections::HashMap<_, _>>();
    let creation = if partitioned {
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        iceberg::TableCreation::builder()
            .name("t".to_owned())
            .properties(props)
            .schema(schema)
            .partition_spec(spec.into_unbound())
            .build()
    } else {
        iceberg::TableCreation::builder()
            .name("t".to_owned())
            .properties(props)
            .schema(schema)
            .build()
    };
    env.rt.block_on(async {
        use iceberg::Catalog as _;
        env.catalog
            .create_table(&iceberg::NamespaceIdent::new("db".to_owned()), creation)
            .await
            .unwrap()
    })
}

/// The provider of the table at `location`, committing through `catalog`.
fn provider(env: &Env, location: &str, catalog: &Arc<Catalog>) -> IcebergTableProvider {
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, location, "db", "t"))
        .unwrap()
        .with_committer(Arc::clone(catalog) as Arc<dyn CommitTable>);
    env.provider(&table, &VersionSpec::Current).unwrap()
}

fn request(write_id: Option<&str>) -> WriteRequest {
    let mut r = WriteRequest::new(WriteMode::Append, "`db`.`t`", input_schema());
    r.write_id = write_id.map(str::to_owned);
    r
}

/// Runs `input` into `sink` to the end (cancelled once `cancel_when`
/// holds); the result, the query's pool and the runner (whose runtime the
/// writer tasks run on: keep it while they settle).
fn run(
    env: &Env,
    sink: Arc<dyn Sink>,
    input: Arc<dyn Source>,
    cancel_when: Option<Box<dyn Fn() -> bool + Send>>,
) -> (Result<(), KrishivError>, Arc<dyn MemoryPool>, QueryRunner) {
    let root = TreePool::new(&MemoryConfig::default());
    let pool = root.child("q", None);
    let query = QueryContext::new(
        QueryId(7),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        Arc::clone(&pool),
    );
    let runner = QueryRunner::new(TokioExecutor::new(&ExecutorConfig::default()).unwrap());
    let node = Arc::new(Write {
        input: Arc::new(Scan(input)),
        sink,
    });
    let out = env.rt.block_on(async {
        let mut stream = runner.execute(node, Arc::clone(&query)).await?;
        if let Some(ready) = cancel_when {
            for _ in 0..1_000 {
                if ready() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(ready(), "the write never got going");
            query.cancel();
        }
        let mut result = Ok(());
        while let Some(b) = stream.next().await {
            if let Err(e) = b {
                result = Err(e);
            }
        }
        query.finished().await;
        result
    });
    (out, pool, runner)
}

/// What `pool` holds once a cancelled or failed write's tasks had up to 5 s
/// to abort (they are not the query's blocking work, see `sink`).
fn settled(pool: &Arc<dyn MemoryPool>) -> usize {
    for _ in 0..500 {
        if pool.reserved() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    pool.reserved()
}

/// Every `.parquet` file under `dir`, once background deletes had up to 5 s
/// to empty it when `settle`.
fn parquet_files(env: &Env, settle: bool) -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "parquet") {
                out.push(p.display().to_string());
            }
        }
    }
    let list = || {
        let mut out = Vec::new();
        walk(env.dir.path(), &mut out);
        out.sort();
        out
    };
    for _ in 0..500 {
        if !settle || list().is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    list()
}

fn read_back(env: &Env, catalog: &Catalog) -> Vec<String> {
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, &catalog.location(), "db", "t"))
        .unwrap();
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    super::provider_tests::render(
        &provider_session(p)
            .run("SELECT id, x, name FROM t ORDER BY id")
            .unwrap(),
    )
}

fn expected(ids: impl IntoIterator<Item = i64>) -> Vec<String> {
    ids.into_iter()
        .map(|i| format!("{i}|{}|n{i}", i % 3))
        .collect()
}

fn append_twice(partitioned: bool) {
    let env = env();
    let t = create(&env, ixn_schema(), partitioned, &[]);
    let catalog = Catalog::new(&t);
    let p = provider(&env, &catalog.location(), &catalog);
    assert!(p.capabilities().batch_write);
    let sink = p.writer(&request(Some("w-1"))).unwrap();
    // Two drivers, each with its own writer.
    let input = Arc::new(PerDriver {
        schema: input_schema(),
        batches: vec![range(0, 50), range(50, 100)],
        stall: false,
    });
    let (out, pool, _runner) = run(&env, sink, input, None);
    out.unwrap();
    assert_eq!(pool.reserved(), 0);
    let m = catalog.metadata();
    let s = m.current_snapshot().unwrap();
    let summary = &s.summary().additional_properties;
    assert_eq!(summary.get(WRITE_ID_PROP).map(String::as_str), Some("w-1"));
    assert_eq!(summary.get(QUERY_ID_PROP).map(String::as_str), Some("7"));
    assert_eq!(
        summary.get("added-records").map(String::as_str),
        Some("100")
    );
    let files = parquet_files(&env, false);
    // Partitioned by x (3 values), each driver writes a file per value.
    assert_eq!(files.len(), if partitioned { 6 } else { 2 }, "{files:?}");
    for driver in ["/00000-", "/00001-"] {
        assert!(files.iter().any(|f| f.contains(driver)), "{files:?}");
    }
    assert_eq!(read_back(&env, &catalog), expected(0..100));

    // A second append on top, with a write id of the sink's own.
    let p = provider(&env, &catalog.location(), &catalog);
    let sink = p.writer(&request(None)).unwrap();
    let input = Batches::new(input_schema(), vec![Batch::new(range(100, 130))]);
    let (out, pool, _runner) = run(&env, sink, input, None);
    out.unwrap();
    assert_eq!(pool.reserved(), 0);
    let m = catalog.metadata();
    assert_eq!(m.snapshots().count(), 2);
    let summary = &m
        .current_snapshot()
        .unwrap()
        .summary()
        .additional_properties;
    let id = summary.get(WRITE_ID_PROP).unwrap();
    assert!(id.len() == 36 && id != "w-1", "{id}");
    assert_eq!(
        summary.get("total-records").map(String::as_str),
        Some("130")
    );
    assert_eq!(read_back(&env, &catalog), expected(0..130));
}

#[test]
fn append_into_an_unpartitioned_table_and_again() {
    append_twice(false);
}

#[test]
fn append_into_a_partitioned_table_and_again() {
    append_twice(true);
}

/// `ixn_schema` with `id` required.
fn required_id_schema() -> iceberg::spec::Schema {
    iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "x", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

/// One driver writes good rows into several partitions while the other hits
/// a NULL id: Spark's `NOT_NULL_ASSERT_VIOLATION`, no commit, no file left.
#[test]
fn a_null_in_a_required_column_fails_the_write_and_leaves_nothing() {
    let env = env();
    let t = create(
        &env,
        required_id_schema(),
        true,
        &[("write.target-file-size-bytes", "1")],
    );
    let catalog = Catalog::new(&t);
    let p = provider(&env, &catalog.location(), &catalog);
    let sink = p.writer(&request(Some("w-null"))).unwrap();
    let mut bad: Vec<Option<i64>> = (0..3_000).map(Some).collect();
    bad[2_500] = None;
    let input = Arc::new(PerDriver {
        schema: input_schema(),
        batches: vec![range(10_000, 13_000), rows(&bad)],
        stall: false,
    });
    let (out, pool, _runner) = run(&env, sink, input, None);
    let e = out.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NotNullAssertViolation, "{e}");
    assert_eq!(settled(&pool), 0);
    assert_eq!(e.sql_state(), Some("42000"));
    assert!(
        e.to_string()
            .contains("NULL value appeared in non-nullable field: \nid\nIf the schema"),
        "{e}"
    );
    assert!(catalog.metadata().current_snapshot().is_none());
    assert_eq!(parquet_files(&env, true), Vec::<String>::new());
}

/// A write cancelled while both drivers hold finished files: no commit, no
/// file left, the pool back to 0.
#[test]
fn a_cancelled_write_leaves_nothing() {
    let env = env();
    let t = create(
        &env,
        ixn_schema(),
        true,
        &[("write.target-file-size-bytes", "1")],
    );
    let catalog = Catalog::new(&t);
    let p = provider(&env, &catalog.location(), &catalog);
    let sink = p.writer(&request(Some("w-cancel"))).unwrap();
    let input = Arc::new(PerDriver {
        schema: input_schema(),
        batches: vec![range(0, 3_000), range(3_000, 6_000)],
        stall: true,
    });
    let dir = env.dir.path().to_owned();
    let started = move || {
        let mut n = 0;
        let mut stack = vec![dir.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "parquet") {
                    n += 1;
                }
            }
        }
        n >= 2
    };
    let (out, pool, _runner) = run(&env, sink, input, Some(Box::new(started)));
    assert!(out.is_err());
    assert_eq!(settled(&pool), 0);
    assert!(catalog.metadata().current_snapshot().is_none());
    assert_eq!(parquet_files(&env, true), Vec::<String>::new());
}

/// R:S4: a write cancelled while its commit request is in flight is an
/// unknown outcome: the finalize (and with it `commit_append`) is dropped,
/// and nothing is deleted, as the request may still land.
#[test]
fn a_write_cancelled_during_its_commit_deletes_nothing() {
    let env = env();
    let t = create(&env, ixn_schema(), false, &[]);
    let catalog = Catalog::new(&t);
    catalog.hang.store(true, Ordering::SeqCst);
    let p = provider(&env, &catalog.location(), &catalog);
    let sink = p.writer(&request(Some("w-inflight"))).unwrap();
    let input = Batches::new(input_schema(), vec![Batch::new(range(0, 10))]);
    let sent = Arc::clone(&catalog);
    let (out, pool, _runner) = run(
        &env,
        sink,
        input,
        Some(Box::new(move || sent.sent.load(Ordering::SeqCst))),
    );
    assert!(out.is_err());
    assert_eq!(settled(&pool), 0);
    // Give any (wrong) background delete time to run.
    std::thread::sleep(Duration::from_millis(500));
    let files = parquet_files(&env, false);
    assert_eq!(files.len(), 1, "{files:?}");
    let metadata = env.dir.path().join("db/t/metadata");
    let avro: Vec<_> = std::fs::read_dir(&metadata)
        .unwrap()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "avro"))
        .collect();
    // The manifest and the attempt's manifest list.
    assert_eq!(avro.len(), 2, "{avro:?}");
    assert!(catalog.metadata().current_snapshot().is_none());
}

/// Without a committer the table declares no write mode, so the session's
/// check (and `writer` itself) refuses an append; with one, only appends.
#[test]
fn only_a_table_with_a_committer_takes_appends() {
    let env = env();
    let t = create(&env, ixn_schema(), false, &[]);
    let table = env.load(&t);
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    assert!(!p.capabilities().batch_write);
    let e = p.writer(&request(None)).err().unwrap();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(e.to_string().contains("append in batch mode"), "{e}");

    let catalog = Catalog::new(&t);
    let p = provider(&env, &catalog.location(), &catalog);
    let caps = p.capabilities();
    assert!(caps.batch_write && !caps.truncate && !caps.overwrite_by_filter);
    let mut truncate = request(None);
    truncate.mode = WriteMode::Truncate;
    let e = p.writer(&truncate).err().unwrap();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
}

/// Rows that are not the table's columns are refused when the sink is
/// built, before any file is written.
#[test]
fn a_schema_mismatch_is_refused_before_any_file() {
    let env = env();
    let t = create(&env, ixn_schema(), false, &[]);
    let catalog = Catalog::new(&t);
    let p = provider(&env, &catalog.location(), &catalog);
    let wrong = [
        // A column missing.
        Schema::new(vec![
            Field::new("id", A::Int64, true),
            Field::new("x", A::Int32, true),
        ]),
        // Another type.
        Schema::new(vec![
            Field::new("id", A::Int32, true),
            Field::new("x", A::Int32, true),
            Field::new("name", krishiv_types::DataType::String.to_arrow(), true),
        ]),
        // Another name.
        Schema::new(vec![
            Field::new("id", A::Int64, true),
            Field::new("y", A::Int32, true),
            Field::new("name", krishiv_types::DataType::String.to_arrow(), true),
        ]),
    ];
    for schema in wrong {
        let r = WriteRequest::new(WriteMode::Append, "`db`.`t`", Arc::new(schema));
        let e = p.writer(&r).err().unwrap();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        );
        assert!(e.to_string().contains("`db`.`t`"), "{e}");
    }
    // Nullability is not the type: a non-nullable input column fits.
    let r = WriteRequest::new(
        WriteMode::Append,
        "`db`.`t`",
        Arc::new(Schema::new(vec![
            Field::new("id", A::Int64, false),
            Field::new("x", A::Int32, true),
            Field::new("name", krishiv_types::DataType::String.to_arrow(), true),
        ])),
    );
    p.writer(&r).unwrap();
    assert_eq!(parquet_files(&env, false), Vec::<String>::new());
}

/// Attempt ids never repeat within a process.
#[test]
fn attempt_ids_are_unique() {
    let ids: std::collections::HashSet<u64> =
        (0..100_000).map(|_| super::sink::unique_u64()).collect();
    assert_eq!(ids.len(), 100_000);
}
