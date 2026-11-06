// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries on a simulated cluster of two workers (design
//! P2-T176 unit 2): the same rows as one worker, the same trace from the
//! same seed, and the same rows when a map task is killed and rerun (I10).
//! The coordinator deals the scans' file splits over the tasks, so each
//! table is read once across the workers (P2-T178), and each worker builds
//! a broadcast join's table once for its tasks (P2-T177 unit 3).
//!
//! The data is the SF 0.01 Parquet fixture of the planner tests, or the
//! directory `KRISHIV_SIM_TPCH_DIR` names (the SF 0.1 CI job generates it
//! with `tests/scripts/gen_tpch_parquet.py`). Every host opens its own
//! tables over one shared in-memory object store: the only thing hosts
//! share besides simulated TCP.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arrow_array::{Array as _, Int64Array, RecordBatch, StringViewArray};
use futures::StreamExt as _;
use futures::future::BoxFuture;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::{PhysicalPlan, SplitFileRef, WorkerContext, WritePlan};
use krishiv_exec::planner::{
    DefaultPlanner, FileTable, MemoryTables, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_exec::{
    ExecConfig, Executor, QueryContext, QueryId, QueryRunner, Scheduler, TokioClock, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::SplitConfig;
use krishiv_io::{
    DirectoryWrite, FileStatistics, FormatRegistry, IoContext, Precision, SaveMode, WrittenFileInfo,
};
use krishiv_io::{
    FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, MetadataNeeds, StoreId,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
use krishiv_scheduler::TableDealer;
use krishiv_sim::{ClusterConfig, Coordinator, Fault, HostSetup, SimConfig, simulate};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path as StorePath;

const TABLES: [&str; 8] = [
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn data_dir() -> PathBuf {
    std::env::var_os("KRISHIV_SIM_TPCH_DIR").map_or_else(
        || repo().join("crates/krishiv-exec/tests/fixtures/tpch-sf0.01-parquet"),
        PathBuf::from,
    )
}

/// The TPC-H tables as one in-memory object store, the cluster's shared
/// storage.
fn store() -> Arc<dyn ObjectStore> {
    let store = InMemory::new();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for t in TABLES {
        let bytes = std::fs::read(data_dir().join(format!("{t}.parquet"))).unwrap();
        let path = object_store::path::Path::from(format!("{t}.parquet").as_str());
        rt.block_on(object_store::ObjectStoreExt::put(
            &store,
            &path,
            bytes.into(),
        ))
        .unwrap();
    }
    Arc::new(store)
}

fn config() -> Config {
    let mut config = Config::new();
    config.set("spark.sql.ansi.enabled", "true").unwrap();
    config
}

/// One host's TPC-H tables and catalog, opened over `store`.
struct Tpch {
    ctx: AnalysisContext,
    tables: Arc<MemoryTables>,
    config: Arc<Config>,
    /// Each table's files, for dealing scans to tasks.
    files: BTreeMap<String, Vec<SplitFileRef>>,
    /// The host's I/O over the shared store (`memory://`).
    io: IoContext,
}

async fn open(store: Arc<dyn ObjectStore>) -> Result<Tpch, KrishivError> {
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("memory://");
    sched.register_store(id.clone(), Arc::clone(&store));
    let io = sched.context();
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let config = config();
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    let mut listed = BTreeMap::new();
    for t in TABLES {
        let path = object_store::path::Path::from(format!("{t}.parquet").as_str());
        let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
            .await
            .unwrap();
        let files = vec![FileRef::from_meta(id.clone(), &meta)];
        listed.insert(
            format!("spark_catalog.default.{t}"),
            files.iter().map(SplitFileRef::from_file_ref).collect(),
        );
        let file_schema = io
            .read_metadata(
                format.as_ref(),
                &files[0],
                MetadataNeeds::default(),
                &options,
            )
            .await?
            .schema();
        let mut columns = Vec::new();
        let mut fields = Vec::new();
        for f in file_schema.fields() {
            let data_type = krishiv_types::DataType::from_arrow_field(f).unwrap();
            fields.push(data_type.to_arrow_field(f.name(), f.is_nullable()));
            columns.push(TableColumn {
                name: f.name().clone(),
                data_type,
                nullable: f.is_nullable(),
            });
        }
        catalog = catalog.with_table(&["default"], t, columns, None);
        let table = FileTable::open(
            Arc::clone(&format),
            io.clone(),
            Arc::new(arrow_schema::Schema::new(fields)),
            files,
            options.clone(),
            &config,
            12,
        )
        .await?;
        tables = tables.with_table(&["spark_catalog", "default", t], Arc::new(table));
    }
    Ok(Tpch {
        ctx: context(
            catalog,
            ResolutionConf {
                ansi: true,
                ..ResolutionConf::default()
            },
        ),
        tables: Arc::new(tables),
        config: Arc::new(config),
        files: listed,
        io,
    })
}

impl Tpch {
    async fn plan(&self, q: usize) -> Result<PhysicalPlan, KrishivError> {
        let text = std::fs::read_to_string(repo().join(format!("bench/queries/tpch/q{q:02}.sql")))
            .unwrap();
        self.plan_sql(&text).await
    }

    async fn plan_sql(&self, text: &str) -> Result<PhysicalPlan, KrishivError> {
        self.plan_sql_with(text, Arc::clone(&self.config)).await
    }

    async fn plan_sql_with(
        &self,
        text: &str,
        config: Arc<Config>,
    ) -> Result<PhysicalPlan, KrishivError> {
        let plan =
            krishiv_sql::parse_and_lower(text, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .map_err(KrishivError::from)?;
        let plan: Plan = DefaultAnalyzer.analyze(&plan, &self.ctx).await?.plan;
        let stats = ProviderStatistics(Arc::clone(&self.tables) as _);
        let plan = krishiv_optimizer::optimize_with(&plan, &stats)?;
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, config);
        DefaultPlanner.plan(&plan, &cx)
    }
}

/// Each host opens its own tables over the shared store.
struct Setup(Arc<dyn ObjectStore>);

impl HostSetup for Setup {
    fn worker(&self) -> BoxFuture<'static, Result<WorkerContext, KrishivError>> {
        let store = Arc::clone(&self.0);
        Box::pin(async move {
            let tpch = open(store).await?;
            // Writes go to the shared store through the host's own I/O.
            Ok(WorkerContext::new(tpch.tables as _)
                .with_io(tpch.io)
                .with_formats(Arc::new(FormatRegistry::with_defaults()))
                .with_config(tpch.config))
        })
    }
}

/// Deals the scans of `tpch`'s tables over a stage's tasks in splits of
/// `bytes`.
fn dealer(tpch: &Tpch, bytes: u64) -> Arc<TableDealer> {
    Arc::new(TableDealer::new(
        tpch.files.clone(),
        SplitConfig {
            max_partition_bytes: bytes,
            open_cost_in_bytes: 0,
            min_partition_num: 1,
            max_partition_num: None,
        },
    ))
}

/// `plan`'s batches through `scheduler`.
async fn batches(
    seed: u64,
    scheduler: &dyn Scheduler,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    pool: &Arc<dyn MemoryPool>,
    id: u64,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let query = QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        ExecConfig {
            seed: Some(seed),
            ..ExecConfig::default()
        },
        pool.child(&format!("q{id}"), None),
    );
    let mut stream = scheduler
        .submit(plan.clone(), worker.clone(), Arc::clone(&query))
        .await?;
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        out.push(b?.into_parts().0);
    }
    drop(stream);
    query.finished().await;
    Ok(out)
}

/// `plan`'s rows through `scheduler`, as text, sorted.
async fn rows(
    seed: u64,
    scheduler: &dyn Scheduler,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    pool: &Arc<dyn MemoryPool>,
    id: u64,
) -> Result<Vec<String>, KrishivError> {
    let query = QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        ExecConfig {
            seed: Some(seed),
            ..ExecConfig::default()
        },
        pool.child(&format!("q{id}"), None),
    );
    let mut stream = scheduler
        .submit(plan.clone(), worker.clone(), Arc::clone(&query))
        .await?;
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        let (batch, _) = b?.into_parts();
        for r in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|c| format!("{:?}", c.slice(r, 1)))
                .collect();
            out.push(cells.join(" | "));
        }
    }
    drop(stream);
    query.finished().await;
    out.sort();
    Ok(out)
}

/// What one simulated run produced: per query, the cluster's rows and one
/// worker's, and the coordinator's trace.
struct Run {
    results: Vec<(usize, Vec<String>, Vec<String>)>,
    trace: Vec<String>,
}

fn run(seed: u64, queries: &[usize], fault: Option<Fault>) -> Run {
    let store = store();
    let config = ClusterConfig {
        sim: SimConfig::with_seed(seed),
        workers: 2,
        partitions: 4,
        slots: 2,
        fault,
    };
    let results = Arc::new(Mutex::new(Vec::new()));
    let out = Arc::clone(&results);
    let queries = queries.to_vec();
    let client_store = Arc::clone(&store);
    let trace = simulate(
        config,
        Arc::new(Setup(store)),
        move |coordinator: Coordinator, worker: WorkerContext| async move {
            let tpch = open(client_store).await?;
            // Scans dealt over the workers in 256 KiB splits (P2-T178): the
            // tasks' splits travel in their descriptors.
            coordinator.deal_scans(dealer(&tpch, 256 << 10));
            let executor: Arc<dyn Executor> = TokioExecutor::deterministic(config.slots)?;
            let plain = QueryRunner::new(executor);
            let pool: Arc<dyn MemoryPool> =
                TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
            for (i, q) in queries.into_iter().enumerate() {
                let plan = tpch.plan(q).await?;
                let id = 2 * i as u64;
                let want = rows(seed, &plain, &plan, &worker, &pool, id).await?;
                let got = rows(seed, &coordinator, &plan, &worker, &pool, id + 1)
                    .await
                    .map_err(|e| {
                        KrishivError::from(krishiv_common::error::InternalError {
                            message: format!("q{q:02} on the cluster: {e}"),
                        })
                    })?;
                out.lock().unwrap().push((q, got, want));
            }
            Ok(())
        },
    )
    .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    let results = std::mem::take(&mut *results.lock().unwrap());
    Run { results, trace }
}

fn assert_equal(run: &Run) {
    let mut failed = Vec::new();
    for (q, got, want) in &run.results {
        if got != want {
            failed.push(format!(
                "q{q:02}: {} rows vs {}\n  got  {:?}\n  want {:?}",
                got.len(),
                want.len(),
                got.first(),
                want.first()
            ));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

fn trace_hash(run: &Run) -> u64 {
    let mut h = DefaultHasher::new();
    run.trace.hash(&mut h);
    for (q, got, _) in &run.results {
        (q, got).hash(&mut h);
    }
    h.finish()
}

#[test]
fn tpch_on_two_workers_equals_one_worker() {
    let all: Vec<usize> = (1..=22).collect();
    let run = run(1, &all, None);
    assert_eq!(run.results.len(), 22);
    assert_equal(&run);
    // Not an empty match: nearly every query returns rows (at SF 0.01 a
    // selective one may return none).
    let with_rows = run
        .results
        .iter()
        .filter(|(_, got, _)| !got.is_empty())
        .count();
    assert!(with_rows >= 20, "{with_rows} queries returned rows");
    // Every query ran map tasks on both workers.
    for w in ["worker-0", "worker-1"] {
        assert!(
            run.trace.iter().any(|e| e.contains(w)),
            "{w}: {:?}",
            run.trace
        );
    }
}

#[test]
fn the_same_seed_gives_the_same_trace() {
    let queries = [3, 9, 18];
    let a = run(7, &queries, None);
    let b = run(7, &queries, None);
    assert_equal(&a);
    assert_eq!(a.trace, b.trace);
    assert_eq!(trace_hash(&a), trace_hash(&b));
}

#[test]
fn a_killed_map_task_reruns_with_the_same_rows() {
    let fault = Fault {
        map: 0,
        attempt: 0,
        after: 1,
    };
    let run = run(3, &[1, 3, 13, 18], Some(fault));
    assert_equal(&run);
    assert!(
        run.trace
            .iter()
            .any(|e| e.contains("attempt 0") && e.contains("failed: ")),
        "no task failed: {:?}",
        run.trace
    );
    assert!(
        run.trace
            .iter()
            .any(|e| e.contains("attempt 1") && e.ends_with("done"))
    );
}

#[test]
fn an_unseeded_query_is_refused() {
    let store = store();
    let config = ClusterConfig {
        sim: SimConfig::with_seed(5),
        workers: 2,
        partitions: 2,
        slots: 1,
        fault: None,
    };
    let client_store = Arc::clone(&store);
    let refused = simulate(
        config,
        Arc::new(Setup(store)),
        move |coordinator: Coordinator, worker: WorkerContext| async move {
            let tpch = open(client_store).await?;
            let plan = tpch.plan(6).await?;
            let pool: Arc<dyn MemoryPool> =
                TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
            let query = QueryContext::new(
                QueryId(0),
                WorkloadClass::Interactive,
                ExecConfig::default(),
                pool.child("q", None),
            );
            let err = coordinator
                .submit(plan, worker, query)
                .await
                .err()
                .map(|e| e.message().to_owned())
                .unwrap_or_default();
            assert!(err.contains("execution seed"), "{err}");
            Ok(())
        },
    );
    assert!(refused.is_ok(), "{refused:?}");
}

/// A file write dealt over two workers (design P2-T178 §7): every task
/// writes the files its partition names, Spark's names for the same task
/// partitions (`part-<partition %05d>-<job>-c000.snappy.parquet`, checked
/// on Spark 4.2.0: one file per task with rows, one job id), and one
/// commit from the gathered rows makes them a table with every row once.
#[test]
fn a_write_on_two_workers_names_files_by_task_and_commits_once() {
    let (names, _) = write_on_two_workers(None);
    check_spark_names(names);
}

/// A write task whose attempt fails after it wrote its files is rerun under
/// its own file names; the commit, told the coordinator retried a task,
/// sweeps the failed attempt's files, so the table holds every row once and
/// no data file the commit did not list.
#[test]
fn a_retried_write_task_commits_only_the_retry_files() {
    let fault = Fault {
        map: 0,
        attempt: 0,
        after: 0,
    };
    let (mut committed, mut listed) = write_on_two_workers(Some(fault));
    committed.sort();
    listed.retain(|p| !p.ends_with("/_SUCCESS"));
    listed.sort();
    assert_eq!(listed, committed);
}

/// Runs `SELECT ... FROM lineitem` into `memory://out` on two workers with
/// `fault`, checks every row is written once and `_SUCCESS` exists, and
/// returns the committed files and every object under `out/`.
fn write_on_two_workers(fault: Option<Fault>) -> (Vec<String>, Vec<String>) {
    let store = store();
    let config = ClusterConfig {
        sim: SimConfig::with_seed(11),
        workers: 2,
        partitions: 4,
        slots: 2,
        fault,
    };
    let names = Arc::new(Mutex::new((Vec::new(), Vec::new())));
    let out = Arc::clone(&names);
    let client_store = Arc::clone(&store);
    simulate(
        config,
        Arc::new(Setup(Arc::clone(&store))),
        move |coordinator: Coordinator, worker: WorkerContext| async move {
            let tpch = open(Arc::clone(&client_store)).await?;
            coordinator.deal_scans(dealer(&tpch, 128 << 10));
            let executor: Arc<dyn Executor> = TokioExecutor::deterministic(config.slots)?;
            let plain = QueryRunner::new(executor);
            let pool: Arc<dyn MemoryPool> =
                TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
            let sql = "SELECT l_orderkey, l_linenumber, l_comment FROM lineitem";
            let plan = tpch.plan_sql(sql).await?;
            let schema = Arc::clone(plan.schema());
            let want = rows(11, &plain, &plan, &worker, &pool, 0).await?;
            let w = DirectoryWrite::begin(
                &tpch.io,
                StoreId::new("memory://"),
                StorePath::from("out"),
                SaveMode::ErrorIfExists,
                "job-sim",
                false,
            )
            .await?
            .unwrap();
            let write = plan.into_write(WritePlan {
                format: "parquet".to_owned(),
                options: BTreeMap::new(),
                store: "memory://".to_owned(),
                dir: "out".to_owned(),
                job_uuid: "job-sim".to_owned(),
                max_records_per_file: 0,
                target_file_bytes: 0,
                partition_by: Vec::new(),
                column_types: BTreeMap::new(),
            })?;
            let mut files = Vec::new();
            for b in batches(11, &coordinator, &write, &worker, &pool, 1).await? {
                let path = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                let size = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                let count = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
                for r in 0..b.num_rows() {
                    let n = u64::try_from(count.value(r)).unwrap();
                    files.push(WrittenFileInfo {
                        path: StorePath::from(path.value(r)),
                        size: u64::try_from(size.value(r)).unwrap(),
                        rows: n,
                        statistics: FileStatistics::with_rows(Precision::Exact(n)),
                    });
                }
            }
            let options = FormatOptions::defaults(ParquetFormat.options());
            w.note_attempt(coordinator.max_attempt());
            if fault.is_some() {
                assert!(coordinator.max_attempt() > 0, "{:?}", coordinator.trace());
            }
            let committed = w
                .commit(&tpch.io, &ParquetFormat, schema, &options, files)
                .await?;
            let written: u64 = committed.iter().map(|f| f.rows).sum();
            assert_eq!(usize::try_from(written).unwrap(), want.len());
            // One commit: its marker, and the files it listed.
            let success = object_store::path::Path::from("out/_SUCCESS");
            object_store::ObjectStoreExt::head(client_store.as_ref(), &success)
                .await
                .unwrap();
            let prefix = object_store::path::Path::from("out");
            let listed: Vec<String> = client_store
                .list(Some(&prefix))
                .map(|m| m.unwrap().location.to_string())
                .collect()
                .await;
            *out.lock().unwrap() = (
                committed.iter().map(|f| f.path.to_string()).collect(),
                listed,
            );
            Ok(())
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    std::mem::take(&mut *names.lock().unwrap())
}

/// Checks `names` are Spark's file names of a write with several tasks.
fn check_spark_names(mut names: Vec<String>) {
    names.sort();
    assert!(names.len() >= 2, "{names:?}");
    // Spark's `part-<task>-<job>-c<file>`: a task whose drivers each write
    // a file numbers them from c000 (how many depends on its splits).
    let mut files: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for n in &names {
        let (task, file) = n
            .strip_prefix("out/part-")
            .and_then(|r| r.strip_suffix(".snappy.parquet"))
            .and_then(|r| r.split_once("-job-sim-c"))
            .unwrap_or_else(|| panic!("not a Spark name: {n}"));
        assert_eq!((task.len(), file.len()), (5, 3), "{n}");
        files
            .entry(task.parse().unwrap())
            .or_default()
            .push(file.parse().unwrap());
    }
    assert!(files.len() >= 2, "files of one task only: {names:?}");
    for (task, mut numbers) in files {
        numbers.sort_unstable();
        let expected: Vec<u32> = (0..u32::try_from(numbers.len()).unwrap()).collect();
        assert_eq!(numbers, expected, "task {task}: {names:?}");
    }
}

/// A file changed between the coordinator's listing and a worker's read
/// fails the query with `KRISHIV_FILE_CHANGED_DURING_READ` instead of
/// mixing versions: the dealt splits carry the listed version.
#[test]
fn a_file_changed_after_listing_fails_the_read_on_a_worker() {
    let store = store();
    let config = ClusterConfig {
        sim: SimConfig::with_seed(5),
        workers: 2,
        partitions: 4,
        slots: 2,
        fault: None,
    };
    let client_store = Arc::clone(&store);
    let outcome = Arc::new(Mutex::new(None));
    let out = Arc::clone(&outcome);
    simulate(
        config,
        Arc::new(Setup(Arc::clone(&store))),
        move |coordinator: Coordinator, worker: WorkerContext| async move {
            let tpch = open(Arc::clone(&client_store)).await?;
            coordinator.deal_scans(dealer(&tpch, 128 << 10));
            let plan = tpch
                .plan_sql("SELECT sum(l_quantity) FROM lineitem")
                .await?;
            // Rewritten after listing: same bytes, a new version.
            let path = object_store::path::Path::from("lineitem.parquet");
            let bytes = object_store::ObjectStoreExt::get(client_store.as_ref(), &path)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            object_store::ObjectStoreExt::put(client_store.as_ref(), &path, bytes.into())
                .await
                .unwrap();
            let pool: Arc<dyn MemoryPool> =
                TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
            let got = rows(5, &coordinator, &plan, &worker, &pool, 0).await;
            *out.lock().unwrap() = Some(got.map_err(|e| e.to_string()));
            Ok(())
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let got = outcome.lock().unwrap().take().unwrap();
    let e = got.unwrap_err();
    assert!(e.contains("KRISHIV_FILE_CHANGED_DURING_READ"), "{e}");
}

/// The retry case of design P2-T177 §7 across stages: a partitioned join
/// whose build stage collects the probe scan's filter in several tasks,
/// one of which is killed once and reruns. Only the committed attempt's
/// partial is merged (the coordinator takes the partials of the attempt
/// that succeeded), and the rows equal one worker's.
#[test]
fn a_rerun_build_task_gives_one_partial_and_the_same_rows() {
    let store = store();
    let config = ClusterConfig {
        sim: SimConfig::with_seed(3),
        workers: 2,
        partitions: 4,
        slots: 2,
        fault: Some(Fault {
            map: 1,
            attempt: 0,
            after: 0,
        }),
    };
    let client_store = Arc::clone(&store);
    let results = Arc::new(Mutex::new(None));
    let out = Arc::clone(&results);
    let trace = simulate(
        config,
        Arc::new(Setup(Arc::clone(&store))),
        move |coordinator: Coordinator, worker: WorkerContext| async move {
            let tpch = open(Arc::clone(&client_store)).await?;
            coordinator.deal_scans(dealer(&tpch, 64 << 10));
            // Partitioned joins: no broadcast at this scale otherwise.
            let mut partitioned = Config::clone(&tpch.config);
            partitioned
                .set("spark.sql.autoBroadcastJoinThreshold", "-1")
                .unwrap();
            let sql = "SELECT l_orderkey, l_linenumber, o_orderdate FROM lineitem JOIN orders \
                       ON l_orderkey = o_orderkey WHERE o_orderdate < DATE '1992-03-01' \
                       AND o_orderpriority = '1-URGENT'";
            let plan = tpch.plan_sql_with(sql, Arc::new(partitioned)).await?;
            let schemas = krishiv_exec::physical::exchange_schemas(&plan, &worker)?;
            let collecting = krishiv_scheduler::fragment(&plan, &schemas)?
                .iter()
                .any(|s| {
                    matches!(s.plan.root(), krishiv_exec::physical::PhysNode::Exchange(e) if !e.filters.is_empty())
                });
            let executor: Arc<dyn Executor> = TokioExecutor::deterministic(config.slots)?;
            let plain = QueryRunner::new(executor);
            let pool: Arc<dyn MemoryPool> =
                TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
            let want = rows(3, &plain, &plan, &worker, &pool, 0).await?;
            let got = rows(3, &coordinator, &plan, &worker, &pool, 1).await?;
            *out.lock().unwrap() = Some((collecting, got, want));
            Ok(())
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let (collecting, got, want) = results.lock().unwrap().take().unwrap();
    assert!(collecting, "no filter crosses stages");
    assert!(!want.is_empty());
    assert_eq!(got, want);
    assert!(
        trace
            .iter()
            .any(|e| e.contains("task 1 attempt 0") && e.contains("failed")),
        "{trace:?}"
    );
}
