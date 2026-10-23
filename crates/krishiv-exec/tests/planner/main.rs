// SPDX-License-Identifier: Apache-2.0
//! Planner tests: SQL through the analyzer, the optimizer, the planner and
//! the executor, over in-memory tables.
#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::unwrap_in_result,
    clippy::print_stderr,
    reason = "test helpers sit outside #[test] fns; the TPC-H runs print their timings"
)]

mod exchanges;
mod hints;
mod listing;
mod physical;
mod runtime_filters;
mod semantics;
mod tpch;
mod window;
mod write;

use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_memory::{MemoryConfig, MemoryPool as _, TreePool};
use krishiv_optimizer::Statistics as _;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{AttrId, Expr, ExprNode, LogicalPlan, Plan, PlanNode, ResolvedExpr, SourceId};
use krishiv_types::batch::Batch;

use krishiv_exec::physical::WorkerContext;
use krishiv_exec::planner::{
    DefaultPlanner, ExplainNode, MemoryTable, MemoryTables, PhysicalPlan, PhysicalPlanner,
    PlanningContext, ProviderStatistics, TableProvider, TableProviderResolver,
};
use krishiv_exec::{
    ExecConfig, ExecutorConfig, FileWriteSink, QueryContext, QueryId, QueryRunner, Scheduler as _,
    TokioExecutor, WorkloadClass,
};

/// A plan and the worker context it runs with.
#[derive(Debug)]
pub(crate) struct Runnable {
    plan: PhysicalPlan,
    worker: WorkerContext,
}

impl Runnable {
    /// `plan` reading `tables`, without spilling.
    pub(crate) fn new(plan: PhysicalPlan, tables: Arc<dyn TableProviderResolver>) -> Self {
        Self {
            plan,
            worker: WorkerContext::new(tables),
        }
    }

    /// The same plan spilling into `scope`.
    pub(crate) fn with_spill(mut self, scope: krishiv_memory::SpillScope) -> Self {
        self.worker = self.worker.with_spill(scope);
        self
    }

    /// The plan writing its rows through `sink`.
    pub(crate) fn into_write(self, sink: Arc<FileWriteSink>) -> Result<Self, KrishivError> {
        Ok(Self {
            plan: self.plan.into_write(&sink)?,
            worker: self.worker.with_write_sink(sink),
        })
    }
}

impl std::ops::Deref for Runnable {
    type Target = PhysicalPlan;

    fn deref(&self) -> &PhysicalPlan {
        &self.plan
    }
}

impl std::fmt::Display for Runnable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.plan.fmt(f)
    }
}

/// `t(a INT, b BIGINT, s STRING)`: rows `(i, 10 * i, "s{i}")` for `i` in
/// `0..n`, every fourth `a` NULL, in batches of 3.
fn table(n: i32) -> (SchemaRef, Vec<Batch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("b", ArrowType::Int64, false),
        Field::new("s", ArrowType::Utf8View, false),
    ]));
    let mut batches = Vec::new();
    let mut i = 0;
    while i < n {
        let rows: Vec<i32> = (i..n.min(i + 3)).collect();
        let a: Int32Array = rows.iter().map(|v| (v % 4 != 3).then_some(*v)).collect();
        let b: Int64Array = rows.iter().map(|v| i64::from(*v) * 10).collect();
        let s: StringViewArray = rows.iter().map(|v| Some(format!("s{v}"))).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(a), Arc::new(b), Arc::new(s)],
        )
        .unwrap();
        batches.push(Batch::new(batch));
        i += 3;
    }
    (schema, batches)
}

struct Env {
    ctx: krishiv_plan::analyzer::AnalysisContext,
    tables: Arc<MemoryTables>,
}

fn env() -> Env {
    let (schema, batches) = table(10);
    let columns = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: f.is_nullable(),
        })
        .collect();
    let catalog = MemoryCatalog::new("spark_catalog").with_table(&["default"], "t", columns, None);
    let tables = MemoryTables::new().with_table(
        &["spark_catalog", "default", "t"],
        Arc::new(MemoryTable::new(schema, batches)),
    );
    Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    }
}

impl Env {
    fn analyzed(&self, sql: &str) -> krishiv_plan::analyzer::AnalyzedPlan {
        let plan =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .unwrap_or_else(|e| panic!("{sql}: {e}"));
        block_on(DefaultAnalyzer.analyze(&plan, &self.ctx)).unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    fn analyze(&self, sql: &str) -> Plan {
        self.analyzed(sql).plan
    }

    fn cx(&self) -> PlanningContext {
        PlanningContext::new(Arc::clone(&self.tables) as _, Arc::new(Config::new()))
    }

    fn plan(&self, sql: &str) -> Result<Runnable, KrishivError> {
        let plan = krishiv_optimizer::optimize(&self.analyze(sql)).unwrap();
        Ok(self.runnable(DefaultPlanner.plan(&plan, &self.cx())?))
    }

    fn runnable(&self, plan: PhysicalPlan) -> Runnable {
        Runnable::new(plan, Arc::clone(&self.tables) as _)
    }

    fn rows(&self, sql: &str) -> (SchemaRef, Vec<String>) {
        let plan = self.plan(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        let schema = Arc::clone(plan.schema());
        let batches = run(&plan).unwrap_or_else(|e| panic!("{sql}: {e}"));
        let mut rows = Vec::new();
        for b in &batches {
            assert_eq!(b.schema().fields(), schema.fields(), "{sql}");
            for r in 0..b.num_rows() {
                let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
                rows.push(cells.join("|"));
            }
        }
        (schema, rows)
    }
}

/// Runs `plan` to completion.
fn run(plan: &Runnable) -> Result<Vec<RecordBatch>, KrishivError> {
    let root = TreePool::new(&MemoryConfig::default());
    run_in(plan, &root).0
}

thread_local! {
    /// Run aggregates on the row-layout tables (`krishiv.exec.agg.rowLayout`),
    /// as `KRISHIV_AGG_ROW_LAYOUT=1` does for every test
    /// (`KRISHIV_JOIN_ROW_LAYOUT=1` turns the join's row layout on).
    static AGG_ROW_LAYOUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `plan` to completion as a query of `root`; also returns the query
/// (for its metrics).
fn run_in(
    plan: &Runnable,
    root: &TreePool,
) -> (Result<Vec<RecordBatch>, KrishivError>, Arc<QueryContext>) {
    let agg_row_layout =
        AGG_ROW_LAYOUT.get() || std::env::var("KRISHIV_AGG_ROW_LAYOUT").is_ok_and(|v| v == "1");
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig {
            agg_row_layout,
            join_row_layout: std::env::var("KRISHIV_JOIN_ROW_LAYOUT").is_ok_and(|v| v == "1"),
            ..ExecConfig::default()
        },
        root.child("q", None),
    );
    let runner = match TokioExecutor::new(&ExecutorConfig::default()) {
        Ok(executor) => QueryRunner::new(executor),
        Err(e) => return (Err(e), query),
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let out = rt.block_on(async {
        let mut stream = runner
            .submit(plan.plan.clone(), plan.worker.clone(), Arc::clone(&query))
            .await?;
        let mut out = Vec::new();
        while let Some(b) = stream.next().await {
            out.push(b?.into_parts().0);
        }
        query.finished().await;
        Ok(out)
    });
    (out, query)
}

/// One value as text (`NULL` for null).
fn cell(a: &dyn Array, row: usize) -> String {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type};
    if a.is_null(row) || a.data_type() == &ArrowType::Null {
        return "NULL".to_owned();
    }
    match a.data_type() {
        ArrowType::Int32 => a.as_primitive::<Int32Type>().value(row).to_string(),
        ArrowType::Int64 => a.as_primitive::<Int64Type>().value(row).to_string(),
        ArrowType::Utf8View => a.as_string_view().value(row).to_owned(),
        ArrowType::Boolean => a.as_boolean().value(row).to_string(),
        other => panic!("no test formatting for {other}"),
    }
}

fn names(schema: &SchemaRef) -> Vec<&str> {
    schema.fields().iter().map(|f| f.name().as_str()).collect()
}

#[test]
fn projections_rename_and_reorder() {
    let env = env();
    let (schema, rows) = env.rows("SELECT s, a AS x, b FROM t");
    assert_eq!(names(&schema), ["s", "x", "b"]);
    let mut rows = rows;
    rows.sort();
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[0], "s0|0|0");
    assert_eq!(rows[3], "s3|NULL|30");
    // A column selected twice, under two names.
    let (schema, rows) = env.rows("SELECT a, a AS c FROM t");
    assert_eq!(names(&schema), ["a", "c"]);
    assert!(rows.iter().all(|r| {
        let (x, y) = r.split_once('|').unwrap();
        x == y
    }));
}

#[test]
fn order_by_sorts_and_limits() {
    let env = env();
    let (_, rows) = env.rows("SELECT a, b FROM t ORDER BY a DESC");
    assert_eq!(rows.first().map(String::as_str), Some("9|90"));
    // DESC puts NULLs last; the two NULL rows tie, so their order is free.
    assert!(rows[8..].iter().all(|r| r.starts_with("NULL|")), "{rows:?}");
    let (_, rows) = env.rows("SELECT a, b FROM t ORDER BY a ASC NULLS LAST, b DESC");
    assert_eq!(rows[..2], ["0|0", "1|10"]);
    assert_eq!(rows[8..], ["NULL|70", "NULL|30"]);
    // `ORDER BY ... LIMIT` is a top-N.
    let (_, rows) = env.rows("SELECT a AS x FROM t ORDER BY x DESC LIMIT 3");
    assert_eq!(rows, ["9", "8", "6"]);
    let plan = env.plan("SELECT a FROM t ORDER BY a LIMIT 2").unwrap();
    assert!(plan.to_string().starts_with("TopN 2"), "{plan}");
    // A limit without a sort.
    let (_, rows) = env.rows("SELECT b FROM t LIMIT 4");
    assert_eq!(rows.len(), 4);
    let (_, rows) = env.rows("SELECT b FROM (SELECT b FROM t LIMIT 4) AS v LIMIT 0");
    assert!(rows.is_empty());
}

#[test]
fn a_top_n_too_large_for_top_n_is_a_sort_and_a_limit() {
    let env = env();
    let sql = format!(
        "SELECT a FROM t ORDER BY a LIMIT {}",
        krishiv_exec::TopN::MAX_ROWS + 1
    );
    let plan = env.plan(&sql).unwrap();
    insta::assert_snapshot!(plan.to_string(), @"
    Limit 65537
    +- Exchange SinglePartition
       +- LocalLimit 65537
          +- Sort [a#0 ASC NULLS FIRST]
             +- Exchange rangepartitioning(a#0 ASC NULLS FIRST)
                +- Scan spark_catalog.default.t [a#0]
    ");
    let (_, rows) = env.rows(&sql);
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[..3], ["NULL", "NULL", "0"]);
}

#[test]
fn explain_prints_the_tree() {
    let env = env();
    let plan = env
        .plan("SELECT b AS y, a FROM t ORDER BY y LIMIT 5")
        .unwrap();
    insta::assert_snapshot!(plan.to_string(), @"
    TopN 5, [y#3 ASC NULLS FIRST]
    +- Exchange SinglePartition
       +- LocalTopN 5, [y#3 ASC NULLS FIRST]
          +- Project [b#1 AS y#3, a#0]
             +- Scan spark_catalog.default.t [a#0, b#1]
    ");
    assert!(plan.warnings().is_empty());
    assert!(format!("{plan:?}").contains("TopN 5"));
    let tree = ExplainNode::new(
        "Join",
        "",
        vec![
            ExplainNode::new("A", "x", vec![ExplainNode::leaf("C", "")]),
            ExplainNode::leaf("B", "y"),
        ],
    );
    assert_eq!(tree.to_string(), "Join\n:- A x\n:  +- C\n+- B y\n");
    let plan = plan.plan.with_warnings(vec!["w".to_owned()]);
    assert_eq!(plan.warnings(), ["w"]);
    assert_eq!(plan.explain().name, "TopN");
}

#[test]
fn a_self_join_instance_reads_shifted_ids() {
    let env = env();
    let plan = env.analyze("SELECT a, b FROM t");
    let PlanNode::Project { input, exprs } = &plan.root().node else {
        panic!("{plan}")
    };
    // Every id inside the instance reads as `id + 100`.
    let shifted: Vec<Arc<Expr>> = exprs
        .iter()
        .map(|e| {
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node else {
                panic!("{e}")
            };
            let mut c = c.clone();
            c.id = AttrId(c.id.0 + 100);
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c))))
        })
        .rev()
        .collect();
    let shift = LogicalPlan::new(PlanNode::Shift {
        input: Arc::clone(input),
        offset: 100,
    });
    let root = LogicalPlan::new(PlanNode::Project {
        input: Arc::new(shift),
        exprs: shifted,
    });
    let plan = env.runnable(
        DefaultPlanner
            .plan(&Plan::try_new(root).unwrap(), &env.cx())
            .unwrap(),
    );
    let batches = run(&plan).unwrap();
    assert_eq!(names(plan.schema()), ["b", "a"]);
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 10);
    // An id overflowing the shift is an engine error.
    let root = LogicalPlan::new(PlanNode::Shift {
        input: Arc::clone(input),
        offset: u64::MAX,
    });
    let e = DefaultPlanner
        .plan(&Plan::try_new(root).unwrap(), &env.cx())
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn unsupported_and_inconsistent_plans_fail_clearly() {
    let env = env();
    for sql in [
        "SELECT x.a FROM t AS x FULL OUTER JOIN t AS y ON x.a = y.a",
        "SELECT coalesce(a, 1) FROM t",
    ] {
        let e = env.plan(sql).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}: {e}"
        );
    }
    // An unresolved plan.
    let unresolved = Plan::try_new(LogicalPlan::table(["t"])).unwrap();
    let e = DefaultPlanner.plan(&unresolved, &env.cx()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // A table the providers do not have.
    let cx = PlanningContext::new(Arc::new(MemoryTables::new()), Arc::new(Config::new()));
    let plan = krishiv_optimizer::optimize(&env.analyze("SELECT a FROM t")).unwrap();
    let e = DefaultPlanner.plan(&plan, &cx).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert!(
        e.to_string().contains("`spark_catalog`.`default`.`t`"),
        "{e}"
    );
}

#[test]
fn nullability_follows_spark() {
    let env = env();
    // Spark 4.2: `SELECT CAST(NULL AS INT), (SELECT a > 0 FROM t WHERE
    // false), (SELECT max(a > 100) FROM t), EXISTS (SELECT 1 FROM t)` has
    // nullable fields but the last (over one row of `t`: Krishiv has no
    // SELECT without FROM, nor NULL-typed result columns, yet).
    let (schema, rows) = env.rows(
        "SELECT CAST(NULL AS INT) AS c, \
         (SELECT a > 0 FROM t WHERE false) AS s, \
         (SELECT max(a > 100) FROM t) AS m, \
         EXISTS (SELECT 1 FROM t) AS e FROM t WHERE b = 0",
    );
    let nullable: Vec<bool> = schema.fields().iter().map(|f| f.is_nullable()).collect();
    assert_eq!(nullable, [true, true, true, false]);
    assert_eq!(rows, ["NULL|NULL|false|true"]);
    // `<=>` with an untyped NULL side: true exactly for the NULL rows.
    let (schema, rows) = env.rows("SELECT a, a <=> NULL, NULL <=> NULL FROM t WHERE b < 50");
    assert!(!schema.field(1).is_nullable() && !schema.field(2).is_nullable());
    assert_eq!(
        rows,
        [
            "0|false|true",
            "1|false|true",
            "2|false|true",
            "NULL|true|true",
            "4|false|true"
        ]
    );
}

#[test]
fn a_scalar_subquery_of_several_rows_fails_like_spark() {
    let env = env();
    let plan = env.plan("SELECT (SELECT a FROM t) AS x FROM t").unwrap();
    let e = run(&plan).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::ScalarSubqueryTooManyRows,
        "{e}"
    );
}

#[test]
fn tables_and_statistics() {
    let env = env();
    let plan = env.analyze("SELECT a FROM t");
    let PlanNode::Project { input, .. } = &plan.root().node else {
        panic!("{plan}")
    };
    let PlanNode::Scan(scan) = &input.node else {
        panic!("{plan}")
    };
    // Names compare case-insensitively.
    let (schema, batches) = table(4);
    let tables = MemoryTables::new().with_table(
        &["SPARK_CATALOG", "Default", "T"],
        Arc::new(MemoryTable::new(Arc::clone(&schema), batches.clone())),
    );
    let provider = tables.provider(scan).unwrap();
    assert_eq!(provider.statistics().map(|s| s.rows), Some(4));
    assert_eq!(provider.scan().unwrap().table_schema(), schema);
    let stats = ProviderStatistics(Arc::new(tables));
    assert_eq!(stats.table(scan).map(|s| s.rows), Some(4));
    assert!(format!("{stats:?}").contains("MemoryTables"));
    let none = MemoryTable::new(schema, batches).with_statistics(None);
    assert!(none.statistics().is_none());
    assert!(
        ProviderStatistics(Arc::new(MemoryTables::new()))
            .table(scan)
            .is_none()
    );
    // Reading files by path needs a resolver that lists files.
    let mut files = scan.clone();
    files.source = krishiv_plan::ReadSource::Files {
        format: "parquet".to_owned(),
        paths: Vec::new(),
        options: std::collections::BTreeMap::new(),
    };
    let e = MemoryTables::new().provider(&files).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let dir = tempfile::tempdir().unwrap();
    let worker = WorkerContext::new(Arc::clone(&env.tables) as _).with_spill(
        krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
            dir: dir.path().join("spill"),
            ..krishiv_memory::SpillConfig::default()
        })
        .unwrap()
        .scope()
        .unwrap(),
    );
    assert!(worker.spill().is_some());
    assert!(format!("{worker:?}").contains("spill: true"));
}

/// Every scan reads the same in-memory table, files included.
#[derive(Debug)]
struct AnyTable(Arc<dyn TableProvider>);

impl TableProviderResolver for AnyTable {
    fn provider(&self, _: &krishiv_plan::Scan) -> Result<Arc<dyn TableProvider>, KrishivError> {
        Ok(Arc::clone(&self.0))
    }
}

/// A provider without statistics.
#[derive(Debug)]
struct Bare(MemoryTable);

impl TableProvider for Bare {
    fn scan(&self) -> Result<Arc<dyn krishiv_exec::ScanPushdown>, KrishivError> {
        self.0.scan()
    }
}

#[test]
fn unoptimized_plans_and_file_sources() {
    let env = env();
    // Planned as analyzed: the alias is still there.
    let plan = env.analyze("SELECT v.x FROM (SELECT a AS x FROM t) AS v LIMIT 3");
    let physical = env.runnable(DefaultPlanner.plan(&plan, &env.cx()).unwrap());
    assert_eq!(names(physical.schema()), ["x"]);
    let rows: usize = run(&physical)
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 3);
    // Column aliases of a derived table rename the output.
    let renamed = env.analyze("SELECT * FROM t AS v(x, y, z)");
    let physical = env.runnable(DefaultPlanner.plan(&renamed, &env.cx()).unwrap());
    assert_eq!(names(physical.schema()), ["x", "y", "z"]);
    assert_eq!(run(&physical).unwrap()[0].schema().field(2).name(), "z");
    // A scan of files reads through whatever the resolver opens.
    let PlanNode::Limit { input, .. } = &plan.root().node else {
        panic!("{plan}")
    };
    let mut scan = None;
    let mut find = vec![Arc::clone(input)];
    while let Some(p) = find.pop() {
        if let PlanNode::Scan(s) = &p.node {
            scan = Some(s.clone());
        }
        p.for_each_child(&mut |c| find.push(Arc::clone(c)));
    }
    let mut scan = scan.unwrap();
    scan.source = krishiv_plan::ReadSource::Files {
        format: "parquet".to_owned(),
        paths: Vec::new(),
        options: std::collections::BTreeMap::new(),
    };
    let (schema, batches) = table(5);
    let any = AnyTable(Arc::new(Bare(MemoryTable::new(schema, batches))));
    assert!(any.0.statistics().is_none());
    let any: Arc<dyn TableProviderResolver> = Arc::new(any);
    let cx = PlanningContext::new(Arc::clone(&any), Arc::new(Config::new()));
    let root = Plan::try_new(LogicalPlan::new(PlanNode::Scan(scan))).unwrap();
    let physical = Runnable::new(DefaultPlanner.plan(&root, &cx).unwrap(), any);
    assert!(
        physical.to_string().starts_with("Scan parquet files"),
        "{physical}"
    );
    let rows: usize = run(&physical)
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5);
}

#[test]
fn limits_must_be_literal_row_counts() {
    let env = env();
    let plan = env.analyze("SELECT a FROM t");
    let with_limit = |limit: Expr| {
        let root = LogicalPlan::new(PlanNode::Limit {
            input: Arc::clone(plan.root()),
            limit: Arc::new(limit),
        });
        DefaultPlanner
            .plan(&Plan::try_new(root).unwrap(), &env.cx())
            .unwrap_err()
            .error_class()
    };
    assert_eq!(
        with_limit(Expr::literal(krishiv_plan::Literal::Integer(-1))),
        ErrorClass::InternalError
    );
    assert_eq!(
        with_limit(Expr::literal(krishiv_plan::Literal::Long(1))),
        ErrorClass::KrishivUnsupportedFeature
    );
    // A table whose batches do not match its schema cannot be scanned.
    let (schema, _) = table(1);
    let (_, batches) = table(3);
    let narrow = Arc::new(Schema::new(vec![schema.field(0).clone()]));
    assert!(MemoryTable::new(narrow, batches).scan().is_err());
}
