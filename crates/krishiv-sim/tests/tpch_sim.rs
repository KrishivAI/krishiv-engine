// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries on a simulated cluster of two workers (design
//! P2-T176 unit 2): the same rows as one worker, the same trace from the
//! same seed, and the same rows when a map task is killed and rerun (I10).
//!
//! The data is the SF 0.01 Parquet fixture of the planner tests, or the
//! directory `KRISHIV_SIM_TPCH_DIR` names (the SF 0.1 CI job generates it
//! with `tests/scripts/gen_tpch_parquet.py`). Every host opens its own
//! tables over one shared in-memory object store: the only thing hosts
//! share besides simulated TCP.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::StreamExt as _;
use futures::future::BoxFuture;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::{PhysicalPlan, WorkerContext};
use krishiv_exec::planner::{
    DefaultPlanner, FileTable, MemoryTables, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_exec::{
    ExecConfig, Executor, QueryContext, QueryId, QueryRunner, Scheduler, TokioClock, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, MetadataNeeds, StoreId,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
use krishiv_sim::{ClusterConfig, Coordinator, Fault, HostSetup, SimConfig, simulate};
use object_store::ObjectStore;
use object_store::memory::InMemory;

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
    for t in TABLES {
        let path = object_store::path::Path::from(format!("{t}.parquet").as_str());
        let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
            .await
            .unwrap();
        let files = vec![FileRef::from_meta(id.clone(), &meta)];
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
    })
}

impl Tpch {
    async fn plan(&self, q: usize) -> Result<PhysicalPlan, KrishivError> {
        let text = std::fs::read_to_string(repo().join(format!("bench/queries/tpch/q{q:02}.sql")))
            .unwrap();
        let plan = krishiv_sql::parse_and_lower(
            &text,
            &krishiv_sql::ParserOptions::default(),
            SourceId(0),
        )
        .map_err(KrishivError::from)?;
        let plan: Plan = DefaultAnalyzer.analyze(&plan, &self.ctx).await?.plan;
        let stats = ProviderStatistics(Arc::clone(&self.tables) as _);
        let plan = krishiv_optimizer::optimize_with(&plan, &stats)?;
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::clone(&self.config));
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
            Ok(WorkerContext::new(tpch.tables as _))
        })
    }
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
