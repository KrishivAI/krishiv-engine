// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries over the SF 0.01 Parquet fixture, run as stages of
//! tasks over a local shuffle ([`StageRunner`], 4 reduce partitions per
//! hash exchange) and on one worker ([`QueryRunner`]): the same rows
//! (design P2-T176 unit 1).

#![expect(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "test helpers sit outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::RecordBatch;
use futures::StreamExt as _;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::{
    PhysicalPlan, ScanId, ScanSplits, SpillScopes, SplitFileRef, WorkerContext,
};
use krishiv_exec::planner::{
    DefaultPlanner, FileTable, MemoryTables, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_exec::{
    ExecConfig, ExecutorConfig, QueryContext, QueryId, QueryRunner, Scheduler, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::SplitConfig;
use krishiv_io::{
    FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, MetadataNeeds, StoreId,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
use krishiv_scheduler::{StageRunner, StageSplits, TableDealer, fragment};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

const TABLES: [&str; 8] = [
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Tpch {
    ctx: AnalysisContext,
    tables: Arc<MemoryTables>,
    config: Arc<Config>,
    /// Each table's files, for dealing scans to tasks.
    files: BTreeMap<String, Vec<SplitFileRef>>,
}

/// Deals the scans of a stage's streaming pipeline (probe sides, not
/// builds or subqueries) to its tasks in byte ranges of 64 KiB, as the
/// coordinator would (P2-T178).
fn dealer(files: &BTreeMap<String, Vec<SplitFileRef>>) -> TableDealer {
    TableDealer::new(
        files.clone(),
        SplitConfig {
            max_partition_bytes: 64 << 10,
            open_cost_in_bytes: 0,
            min_partition_num: 1,
            max_partition_num: None,
        },
    )
}

fn tpch(rt: &tokio::runtime::Runtime) -> Tpch {
    let dir = repo().join("crates/krishiv-exec/tests/fixtures/tpch-sf0.01-parquet");
    let store = Arc::new(LocalFileSystem::new_with_prefix(&dir).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let mut config = Config::new();
    config.set("spark.sql.ansi.enabled", "true").unwrap();
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    let mut listed = BTreeMap::new();
    for t in TABLES {
        let path = object_store::path::Path::from(format!("{t}.parquet").as_str());
        let meta = rt
            .block_on(object_store::ObjectStoreExt::head(store.as_ref(), &path))
            .unwrap();
        let files = vec![FileRef::from_meta(id.clone(), &meta)];
        listed.insert(
            format!("spark_catalog.default.{t}"),
            files.iter().map(SplitFileRef::from_file_ref).collect(),
        );
        let file_schema = rt
            .block_on(io.read_metadata(
                format.as_ref(),
                &files[0],
                MetadataNeeds::default(),
                &options,
            ))
            .unwrap()
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
        let table = rt
            .block_on(FileTable::open(
                Arc::clone(&format),
                io.clone(),
                Arc::new(arrow_schema::Schema::new(fields)),
                files,
                options.clone(),
                &config,
                12,
            ))
            .unwrap();
        tables = tables.with_table(&["spark_catalog", "default", t], Arc::new(table));
    }
    Tpch {
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
    }
}

impl Tpch {
    fn plan(&self, q: usize) -> Result<PhysicalPlan, KrishivError> {
        let text = std::fs::read_to_string(repo().join(format!("bench/queries/tpch/q{q:02}.sql")))
            .unwrap();
        self.plan_sql(&text, Arc::clone(&self.config))
    }

    fn plan_sql(&self, text: &str, config: Arc<Config>) -> Result<PhysicalPlan, KrishivError> {
        let plan =
            krishiv_sql::parse_and_lower(text, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .map_err(KrishivError::from)?;
        let plan: Plan = block_on(DefaultAnalyzer.analyze(&plan, &self.ctx))?.plan;
        let stats = ProviderStatistics(Arc::clone(&self.tables) as _);
        let plan = krishiv_optimizer::optimize_with(&plan, &stats)?;
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, config);
        DefaultPlanner.plan(&plan, &cx)
    }
}

/// `plan`'s rows through `scheduler`, as text, sorted unless the plan
/// orders them.
fn run(
    scheduler: &dyn Scheduler,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    pool: &Arc<dyn MemoryPool>,
    id: u64,
) -> Result<Vec<String>, KrishivError> {
    let query = QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        pool.child(&format!("q{id}"), None),
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = rt.block_on(async {
        let mut stream = scheduler
            .submit(plan.clone(), worker.clone(), Arc::clone(&query))
            .await?;
        let mut out = Vec::new();
        while let Some(b) = stream.next().await {
            out.push(b?.into_parts().0);
        }
        query.finished().await;
        Ok::<_, KrishivError>(out)
    })?;
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| format!("{:?}", c.slice(r, 1)))
                .collect();
            rows.push(cells.join(" | "));
        }
    }
    Ok(rows)
}

#[test]
fn tpch_sf0_01_on_stages_equals_one_worker() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let plain = QueryRunner::new(Arc::clone(&executor) as _);
    let stages = StageRunner::new(Arc::clone(&executor) as _, 4).unwrap();
    let split = StageRunner::new(Arc::clone(&executor) as _, 4)
        .unwrap()
        .with_splits(Arc::new(dealer(&tpch.files)));
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let dir = tempfile::tempdir().unwrap();
    let spill = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap();
    let only: Option<usize> = std::env::var("KRISHIV_TPCH_QUERY")
        .ok()
        .map(|q| q.parse().unwrap());
    let mut failed = Vec::new();
    let mut cut = 0;
    for q in 1..=22 {
        if only.is_some_and(|o| o != q) {
            continue;
        }
        let plan = tpch.plan(q).unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        let worker =
            WorkerContext::new(Arc::clone(&tpch.tables) as _).with_spill(spill.scope().unwrap());
        let schemas = krishiv_exec::physical::exchange_schemas(&plan, &worker).unwrap();
        cut += fragment(&plan, &schemas).unwrap().len() - 1;
        let want = run(&plain, &plan, &worker, &root, 1).unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        for (runner, name) in [(&stages, "whole scans"), (&split, "dealt scans")] {
            let got = match run(runner, &plan, &worker, &root, 2) {
                Ok(rows) => rows,
                Err(e) => {
                    failed.push(format!("q{q:02} ({name}): {e}"));
                    continue;
                }
            };
            let (mut got_s, mut want_s) = (got.clone(), want.clone());
            got_s.sort();
            want_s.sort();
            if got_s != want_s {
                failed.push(format!(
                    "q{q:02} ({name}): {} rows vs {}\n  got  {:?}\n  want {:?}",
                    got.len(),
                    want.len(),
                    got.first(),
                    want.first()
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    // The suite crosses stages: every query has at least one exchange.
    assert!(only.is_some() || cut >= 22, "{cut} cuts");
    // Dealt scans run several tasks per stage: their broadcast joins read
    // one table built once per query (design P2-T177 unit 3), and its
    // memory is back once the query ended.
    assert!(
        only.is_some() || split.shared_builds() > 0,
        "no broadcast build was shared"
    );
    assert_eq!(root.reserved(), 0, "memory still held after the queries");
}

/// Scopes from one spill manager, counted.
#[derive(Debug)]
struct Counted {
    manager: SpillManager,
    made: AtomicUsize,
}

impl SpillScopes for Counted {
    fn scope(&self) -> Result<SpillScope, KrishivError> {
        self.made.fetch_add(1, Ordering::Relaxed);
        Ok(self.manager.scope()?)
    }
}

/// Under a 16 MiB pool the exchange writers' buffers spill as frames to
/// the query's spill directory, and the rows still match. A worker with a
/// scope factory (as a session's) spills its shuffle and its tasks into one
/// scope per query.
#[test]
fn stages_under_a_16_mib_pool_match() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let plain = QueryRunner::new(Arc::clone(&executor) as _);
    let stages = StageRunner::new(Arc::clone(&executor) as _, 8)
        .unwrap()
        .with_splits(Arc::new(dealer(&tpch.files)));
    let big: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let small: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
        limit: Some(16 << 20),
        ..MemoryConfig::default()
    });
    let dir = tempfile::tempdir().unwrap();
    let spill = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap();
    let scopes = Arc::new(Counted {
        manager: SpillManager::open(&SpillConfig {
            dir: dir.path().join("scopes"),
            ..SpillConfig::default()
        })
        .unwrap(),
        made: AtomicUsize::new(0),
    });
    let queries = [3, 9, 13, 18];
    for q in queries {
        let plan = tpch.plan(q).unwrap();
        let worker =
            WorkerContext::new(Arc::clone(&tpch.tables) as _).with_spill(spill.scope().unwrap());
        let mut want = run(&plain, &plan, &worker, &big, 1).unwrap();
        let factory = WorkerContext::new(Arc::clone(&tpch.tables) as _)
            .with_spill_scopes(Arc::clone(&scopes) as _);
        let mut got =
            run(&stages, &plan, &factory, &small, 2).unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        want.sort();
        got.sort();
        assert_eq!(got, want, "q{q:02}");
    }
    assert_eq!(scopes.made.load(Ordering::Relaxed), queries.len());
    assert_eq!(small.reserved(), 0);
}

/// Deals as [`dealer`] does, but task 1 of every dealt stage reads a file
/// that does not exist.
#[derive(Debug)]
struct Failing(TableDealer);

impl StageSplits for Failing {
    fn splits(
        &self,
        fragment: &PhysicalPlan,
        tasks: u32,
    ) -> Result<Option<Vec<BTreeMap<ScanId, ScanSplits>>>, KrishivError> {
        let mut dealt = self.0.splits(fragment, tasks)?;
        if let Some(task) = dealt.as_mut().and_then(|d| d.get_mut(1)) {
            for splits in task.values_mut() {
                if let ScanSplits::Files { files, .. } = splits {
                    let moved: Vec<SplitFileRef> = files
                        .iter()
                        .map(|f| SplitFileRef {
                            path: object_store::path::Path::from("missing.parquet"),
                            ..f.clone()
                        })
                        .collect();
                    *files = Arc::from(moved);
                }
            }
        }
        Ok(dealt)
    }
}

fn submit_and_drain(
    scheduler: &dyn Scheduler,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    query: &Arc<QueryContext>,
    before: impl FnOnce(),
) -> Result<usize, KrishivError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        let submitted = scheduler.submit(plan.clone(), worker.clone(), Arc::clone(query));
        before();
        let mut stream = submitted.await?;
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b?.num_rows();
        }
        query.finished().await;
        Ok(rows)
    })
}

#[test]
fn a_failing_map_task_ends_its_stage_and_frees_its_memory() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let stages = StageRunner::new(Arc::clone(&executor) as _, 4)
        .unwrap()
        .with_splits(Arc::new(Failing(dealer(&tpch.files))));
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let worker = WorkerContext::new(Arc::clone(&tpch.tables) as _);
    let plan = tpch.plan(1).unwrap();
    let query = QueryContext::new(
        QueryId(5),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q5", None),
    );
    let e = submit_and_drain(&stages, &plan, &worker, &query, || {}).unwrap_err();
    assert_ne!(
        e.error_class(),
        krishiv_common::error::ErrorClass::OperationCanceled,
        "{e}"
    );
    // Every task has released its memory when `submit` fails, and the
    // query carries the error.
    assert_eq!(root.reserved(), 0);
    assert!(query.error().is_some());
}

#[test]
fn a_cancel_before_the_stages_run_ends_the_query() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let stages = StageRunner::new(Arc::clone(&executor) as _, 4).unwrap();
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let worker = WorkerContext::new(Arc::clone(&tpch.tables) as _);
    let plan = tpch.plan(5).unwrap();
    let query = QueryContext::new(
        QueryId(6),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q6", None),
    );
    let e = submit_and_drain(&stages, &plan, &worker, &query, || {
        stages.cancel(QueryId(6));
    })
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::OperationCanceled,
        "{e}"
    );
    assert_eq!(root.reserved(), 0);
    // The runner forgets the query: a later cancel of its id is a no-op.
    stages.cancel(QueryId(6));
    assert_eq!(stages.running_queries(), 0);
}

#[test]
fn a_submission_dropped_unpolled_is_forgotten() {
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let stages = StageRunner::new(Arc::clone(&executor) as _, 4).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(7),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q7", None),
    );
    let submitted = stages.submit(
        tpch.plan(1).unwrap(),
        WorkerContext::new(Arc::clone(&tpch.tables) as _),
        query,
    );
    assert_eq!(stages.running_queries(), 1);
    // A timeout or a client gone drops the submission unfinished.
    drop(submitted);
    assert_eq!(stages.running_queries(), 0);
}

/// The hazard of design P2-T177 §7, across stages: a partitioned join whose
/// build stage runs as several tasks, each seeing part of the build keys.
/// The probe scan's filter is collected by the build exchange in the
/// earlier stage, merged from every task's partial, and carried to the
/// probe scan's stage; a partial alone would drop probe rows whose keys
/// another task saw. The rows equal one worker's (filters local) and the
/// run with runtime filters off.
#[test]
fn a_filter_merged_from_every_build_task_crosses_stages() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let plain = QueryRunner::new(Arc::clone(&executor) as _);
    let split = StageRunner::new(Arc::clone(&executor) as _, 4)
        .unwrap()
        .with_splits(Arc::new(dealer(&tpch.files)));
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let worker = WorkerContext::new(Arc::clone(&tpch.tables) as _);
    // Partitioned joins: no broadcast at this scale otherwise.
    let mut config = Config::clone(&tpch.config);
    config
        .set("spark.sql.autoBroadcastJoinThreshold", "-1")
        .unwrap();
    let mut off = config.clone();
    off.set("krishiv.runtimeFilter.enabled", "false").unwrap();
    let sql = "SELECT l_orderkey, l_linenumber, o_orderdate FROM lineitem JOIN orders \
               ON l_orderkey = o_orderkey WHERE o_orderdate < DATE '1992-03-01' \
               AND o_orderpriority = '1-URGENT'";
    let plan = tpch.plan_sql(sql, Arc::new(config)).unwrap();
    let schemas = krishiv_exec::physical::exchange_schemas(&plan, &worker).unwrap();
    let stages = fragment(&plan, &schemas).unwrap();
    let collecting: Vec<usize> = stages
        .iter()
        .filter(|s| {
            matches!(s.plan.root(), krishiv_exec::physical::PhysNode::Exchange(e) if !e.filters.is_empty())
        })
        .map(|s| s.id)
        .collect();
    assert!(!collecting.is_empty(), "no filter crosses stages:\n{plan}");
    // Several tasks collect the filter: each partial alone is the hazard.
    let tasks = krishiv_scheduler::stage_tasks(&stages, 4, Some(&dealer(&tpch.files))).unwrap();
    assert!(
        collecting.iter().any(|&s| tasks[s].len() > 1),
        "the collecting stage runs as one task"
    );
    let sorted = |mut rows: Vec<String>| {
        rows.sort();
        rows
    };
    let want = sorted(run(&plain, &plan, &worker, &root, 1).unwrap());
    assert!(!want.is_empty());
    let got = sorted(run(&split, &plan, &worker, &root, 2).unwrap());
    assert_eq!(got.len(), want.len());
    assert_eq!(got, want);
    assert!(split.cross_stage_filters() > 0, "no value crossed");
    let unfiltered = tpch.plan_sql(sql, Arc::new(off)).unwrap();
    assert_eq!(
        sorted(run(&plain, &unfiltered, &worker, &root, 3).unwrap()),
        want
    );
    assert_eq!(root.reserved(), 0);
}

/// A file its format cannot split (gzip or `multiLine` CSV) is dealt whole,
/// as a local scan keeps it: one task reads all of its bytes, so its
/// `_metadata.file_block_length` is the file's length. Other files of the
/// same size are cut into ranges.
#[test]
fn an_unsplittable_file_is_dealt_whole() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let plan = tpch
        .plan_sql(
            "SELECT l_orderkey FROM lineitem",
            Arc::new(Config::clone(&tpch.config)),
        )
        .unwrap();
    let ranges = |dealer: &TableDealer| {
        let mut ranges = Vec::new();
        for task in dealer.splits(&plan, 4).unwrap().unwrap() {
            for splits in task.values() {
                let ScanSplits::Files { files, partitions } = splits else {
                    panic!("not file splits: {splits:?}");
                };
                for s in partitions.iter().flat_map(|p| &p.splits) {
                    ranges.push((s.range.start, s.range.end, files[s.file].size));
                }
            }
        }
        ranges
    };
    let cut = ranges(&dealer(&tpch.files));
    assert!(
        cut.iter()
            .any(|&(start, end, size)| start > 0 || end < size),
        "{cut:?}"
    );
    let paths = tpch.files["spark_catalog.default.lineitem"]
        .iter()
        .map(|f| f.path.to_string());
    let whole = ranges(&dealer(&tpch.files).with_unsplittable(paths));
    assert!(!whole.is_empty());
    assert!(
        whole
            .iter()
            .all(|&(start, end, size)| start == 0 && end == size),
        "{whole:?}"
    );
}

/// A shared broadcast build that does not fit in the query's memory falls
/// back to one build per task (owner decision 2026-11-06, row 30c (b)):
/// each task builds its own table under its own spill scope, and the rows
/// match. With room, the same plan shares the build.
#[test]
fn a_broadcast_that_does_not_fit_is_built_by_each_task() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let plain = QueryRunner::new(Arc::clone(&executor) as _);
    // The query's shared broadcast tables may hold 256 KiB: the build of
    // `orders` (about 1 MiB) does not fit, the tasks' own builds do.
    let capped = StageRunner::new(Arc::clone(&executor) as _, 4)
        .unwrap()
        .with_splits(Arc::new(dealer(&tpch.files)))
        .with_broadcast_limit(256 << 10);
    let split = StageRunner::new(Arc::clone(&executor) as _, 4)
        .unwrap()
        .with_splits(Arc::new(dealer(&tpch.files)));
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let dir = tempfile::tempdir().unwrap();
    let spill = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap();
    let worker =
        WorkerContext::new(Arc::clone(&tpch.tables) as _).with_spill(spill.scope().unwrap());
    let mut config = Config::clone(&tpch.config);
    config
        .set("spark.sql.autoBroadcastJoinThreshold", "1g")
        .unwrap();
    let sql = "SELECT o_orderpriority, count(*), sum(l_quantity), max(o_comment) \
               FROM lineitem JOIN orders ON l_orderkey = o_orderkey GROUP BY o_orderpriority";
    let plan = tpch.plan_sql(sql, Arc::new(config)).unwrap();
    let sorted = |mut rows: Vec<String>| {
        rows.sort();
        rows
    };
    let want = sorted(run(&plain, &plan, &worker, &root, 1).unwrap());
    assert_eq!(want.len(), 5, "{want:?}");
    let got = sorted(run(&capped, &plan, &worker, &root, 2).unwrap());
    assert_eq!(got, want);
    assert_eq!(
        capped.broadcast_fallbacks(),
        1,
        "the shared build fit:\n{plan}"
    );
    assert_eq!(root.reserved(), 0, "memory still held after a fallback");
    // Uncapped, the same plan shares its build.
    assert_eq!(sorted(run(&split, &plan, &worker, &root, 3).unwrap()), want);
    assert_eq!(split.broadcast_fallbacks(), 0);
    assert_eq!(split.shared_builds(), 1);
    assert_eq!(root.reserved(), 0);
}

/// The schemas the planner writes into the plan (T176 unit 1 review m4)
/// are the ones instantiation builds: the fragmenter takes them instead
/// of instantiating the plan, and an exchange read refuses other columns.
#[test]
fn the_planners_exchange_schemas_are_the_instantiated_ones() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = tpch(&rt);
    let worker = WorkerContext::new(Arc::clone(&tpch.tables) as _);
    let mut partitioned = Config::clone(&tpch.config);
    partitioned
        .set("spark.sql.autoBroadcastJoinThreshold", "-1")
        .unwrap();
    let partitioned = Arc::new(partitioned);
    for q in 1..=22 {
        for plan in [
            tpch.plan(q).unwrap(),
            tpch.plan_sql(
                &std::fs::read_to_string(repo().join(format!("bench/queries/tpch/q{q:02}.sql")))
                    .unwrap(),
                Arc::clone(&partitioned),
            )
            .unwrap(),
        ] {
            let planned = krishiv_scheduler::plan_data_schemas(&plan)
                .unwrap_or_else(|| panic!("q{q:02}: an exchange without a schema"));
            let built = krishiv_exec::physical::plan_schemas(&plan, &worker).unwrap();
            assert_eq!(
                planned.exchanges.keys().collect::<Vec<_>>(),
                built.exchanges.keys().collect::<Vec<_>>(),
                "q{q:02}"
            );
            for (id, schema) in &planned.exchanges {
                assert_eq!(
                    schema.fields(),
                    built.exchanges[id].fields(),
                    "q{q:02} exchange {}",
                    id.0
                );
            }
            for (id, schema) in &planned.nested_loop_probes {
                assert_eq!(
                    schema.fields(),
                    built.nested_loop_probes[id].fields(),
                    "q{q:02} nested loop over {}",
                    id.0
                );
            }
        }
    }
}
