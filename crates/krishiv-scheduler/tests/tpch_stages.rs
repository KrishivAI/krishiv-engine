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

use arrow_array::RecordBatch;
use futures::StreamExt as _;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::{
    PhysNode, PhysicalPlan, ScanId, ScanNode, ScanSplits, SplitFileRef, WorkerContext,
    assign_file_splits, plan_file_partitions,
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
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, TreePool};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, ReadSource, SourceId};
use krishiv_scheduler::{StageRunner, StageSplits, fragment};
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
#[derive(Debug)]
struct Dealer(BTreeMap<String, Vec<SplitFileRef>>);

impl StageSplits for Dealer {
    fn splits(
        &self,
        fragment: &PhysicalPlan,
        tasks: u32,
    ) -> Result<Option<Vec<BTreeMap<ScanId, ScanSplits>>>, KrishivError> {
        fn streaming<'a>(n: &'a PhysNode, out: &mut Vec<&'a ScanNode>) {
            match n {
                PhysNode::Scan(s) => out.push(s),
                PhysNode::HashJoin(j) => streaming(&j.probe, out),
                PhysNode::SortMergeJoin(j) => streaming(&j.probe, out),
                PhysNode::NestedLoopJoin(j) => streaming(&j.probe, out),
                PhysNode::UnionLoop { .. } | PhysNode::LateFetch { .. } => {}
                PhysNode::Filter { input, .. }
                | PhysNode::Project { input, .. }
                | PhysNode::ReplicateRows { input, .. }
                | PhysNode::Generate { input, .. }
                | PhysNode::Expand { input, .. } => streaming(input, out),
                other => other.children().into_iter().for_each(|c| streaming(c, out)),
            }
        }
        let mut scans = Vec::new();
        streaming(fragment.root(), &mut scans);
        if scans.is_empty() {
            return Ok(None);
        }
        let tasks = tasks as usize;
        let cfg = SplitConfig {
            max_partition_bytes: 64 << 10,
            open_cost_in_bytes: 0,
            min_partition_num: 1,
            max_partition_num: None,
        };
        let mut out = vec![BTreeMap::new(); tasks];
        for scan in scans {
            let ReadSource::Table { name, .. } = &scan.scan.source else {
                return Ok(None);
            };
            let files = &self.0[name.last().unwrap()];
            let partitions = plan_file_partitions(&cfg, files, |_| true);
            for (t, splits) in assign_file_splits(files, &partitions, tasks)
                .into_iter()
                .enumerate()
            {
                out[t].insert(scan.id, splits);
            }
        }
        Ok(Some(out))
    }
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
            t.to_owned(),
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
        let plan = krishiv_sql::parse_and_lower(
            &text,
            &krishiv_sql::ParserOptions::default(),
            SourceId(0),
        )
        .map_err(KrishivError::from)?;
        let plan: Plan = block_on(DefaultAnalyzer.analyze(&plan, &self.ctx))?.plan;
        let stats = ProviderStatistics(Arc::clone(&self.tables) as _);
        let plan = krishiv_optimizer::optimize_with(&plan, &stats)?;
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::clone(&self.config));
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
        .with_splits(Arc::new(Dealer(tpch.files.clone())));
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
}

/// Under a 16 MiB pool the exchange writers' buffers spill as frames to
/// the query's spill directory, and the rows still match.
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
        .with_splits(Arc::new(Dealer(tpch.files.clone())));
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
    for q in [3, 9, 13, 18] {
        let plan = tpch.plan(q).unwrap();
        let worker =
            WorkerContext::new(Arc::clone(&tpch.tables) as _).with_spill(spill.scope().unwrap());
        let mut want = run(&plain, &plan, &worker, &big, 1).unwrap();
        let mut got =
            run(&stages, &plan, &worker, &small, 2).unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        want.sort();
        got.sort();
        assert_eq!(got, want, "q{q:02}");
    }
    assert_eq!(small.reserved(), 0);
}
