// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries through the analyzer, optimizer, planner and
//! executor over Parquet files: physical-plan snapshots, and results
//! compared with Spark 4.2.0's.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::{
    DefaultPlanner, FileTable, MemoryTables, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, MetadataNeeds, StoreId,
};
use krishiv_memory::{MemoryConfig, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};

use super::Runnable;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

pub(crate) const TABLES: [&str; 8] = [
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

pub(crate) fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub(crate) fn sql(q: usize) -> String {
    std::fs::read_to_string(root().join(format!("bench/queries/tpch/q{q:02}.sql"))).unwrap()
}

/// Tables over the Parquet files `<dir>/<table>.parquet`, the catalog
/// typing them as their files do (as Spark does when it reads them), and
/// the session configuration.
pub(crate) struct Tpch {
    pub(crate) ctx: AnalysisContext,
    pub(crate) tables: Arc<MemoryTables>,
    pub(crate) config: Arc<Config>,
}

pub(crate) fn parquet_tables(dir: &Path, rt: &tokio::runtime::Runtime) -> Tpch {
    parquet_dir(dir, &TABLES, true, rt)
}

pub(crate) fn parquet_dir(
    dir: &Path,
    names: &[&str],
    ansi: bool,
    rt: &tokio::runtime::Runtime,
) -> Tpch {
    let tables: Vec<(&str, Vec<String>)> = names
        .iter()
        .map(|t| (*t, vec![format!("{t}.parquet")]))
        .collect();
    parquet_files(dir, &tables, ansi, rt)
}

/// Tables of the Parquet files under `dir` (relative paths; the first
/// file's schema is the table's).
pub(crate) fn parquet_files(
    dir: &Path,
    sources: &[(&str, Vec<String>)],
    ansi: bool,
    rt: &tokio::runtime::Runtime,
) -> Tpch {
    parquet_files_with(dir, sources, ansi, &[], rt)
}

/// As [`parquet_files`], with `settings` in the session the tables are
/// opened under (format options such as `krishiv.io.lateMaterialization`).
pub(crate) fn parquet_files_with(
    dir: &Path,
    sources: &[(&str, Vec<String>)],
    ansi: bool,
    settings: &[(&str, &str)],
    rt: &tokio::runtime::Runtime,
) -> Tpch {
    let store = Arc::new(LocalFileSystem::new_with_prefix(dir).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let mut config = Config::new();
    config
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    for (k, v) in settings {
        config.set(k, v).unwrap();
    }
    // Format options fall back to the session's settings.
    let options = if settings.is_empty() {
        options
    } else {
        ParquetFormat
            .options()
            .parse(&Default::default(), &config)
            .unwrap()
            .0
    };
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    for (t, paths) in sources {
        let files: Vec<FileRef> = paths
            .iter()
            .map(|p| {
                let path = object_store::path::Path::from(p.as_str());
                let meta = rt
                    .block_on(object_store::ObjectStoreExt::head(store.as_ref(), &path))
                    .unwrap();
                FileRef::from_meta(id.clone(), &meta)
            })
            .collect();
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
    let conf = ResolutionConf {
        ansi,
        ..ResolutionConf::default()
    };
    Tpch {
        ctx: context(catalog, conf),
        tables: Arc::new(tables),
        config: Arc::new(config),
    }
}

/// How a test plan orders its joins.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Order {
    /// By the tables' footer statistics.
    Statistics,
    /// By the optimizer's default estimates.
    Defaults,
    /// As the query writes them.
    Written,
}

impl Tpch {
    pub(crate) fn plan(&self, q: usize) -> Runnable {
        self.plan_sql(&sql(q))
            .unwrap_or_else(|e| panic!("q{q:02}: {e}"))
    }

    /// Parses, analyzes, optimizes (ordering joins by the tables'
    /// statistics) and plans `text`.
    pub(crate) fn plan_sql(&self, text: &str) -> Result<Runnable, KrishivError> {
        self.plan_sql_with(text, Order::Statistics)
    }

    /// As [`Self::plan_sql`], ordering the joins by `order`.
    pub(crate) fn plan_sql_with(&self, text: &str, order: Order) -> Result<Runnable, KrishivError> {
        self.plan_sql_in(text, order, None)
    }

    /// `text` parsed, analyzed and optimized (joins ordered by `order`).
    pub(crate) fn optimized(&self, text: &str, order: Order) -> Result<Plan, KrishivError> {
        let plan =
            krishiv_sql::parse_and_lower(text, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .map_err(KrishivError::from)?;
        let plan: Plan = block_on(DefaultAnalyzer.analyze(&plan, &self.ctx))?.plan;
        Ok(match order {
            Order::Statistics => {
                let stats = ProviderStatistics(Arc::clone(&self.tables) as _);
                krishiv_optimizer::optimize_with(&plan, &stats)?
            }
            Order::Defaults => krishiv_optimizer::optimize(&plan)?,
            Order::Written => krishiv_optimizer::optimize_as_written(&plan)?,
        })
    }

    /// As [`Self::plan_sql_with`], its operators spilling into `spill`.
    pub(crate) fn plan_sql_in(
        &self,
        text: &str,
        order: Order,
        spill: Option<&SpillScope>,
    ) -> Result<Runnable, KrishivError> {
        let plan = self.optimized(text, order)?;
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::clone(&self.config));
        let mut runnable = Runnable::new(
            DefaultPlanner.plan(&plan, &cx)?,
            Arc::clone(&self.tables) as _,
        );
        if let Some(scope) = spill {
            runnable = runnable.with_spill(scope.clone());
        }
        Ok(runnable)
    }
}

/// One value as Spark's answers render it.
pub(crate) fn render(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) || a.data_type() == &ArrowType::Null {
        return "NULL".to_owned();
    }
    match a.data_type() {
        ArrowType::Int32 => a.as_primitive::<Int32Type>().value(row).to_string(),
        ArrowType::Int64 => a.as_primitive::<Int64Type>().value(row).to_string(),
        ArrowType::Boolean => a.as_boolean().value(row).to_string(),
        ArrowType::Int8 => a.as_primitive::<Int8Type>().value(row).to_string(),
        ArrowType::Int16 => a.as_primitive::<Int16Type>().value(row).to_string(),
        ArrowType::Float32 => python_float(f64::from(
            a.as_primitive::<arrow_array::types::Float32Type>()
                .value(row),
        )),
        ArrowType::Float64 => python_float(a.as_primitive::<Float64Type>().value(row)),
        ArrowType::Utf8View => a.as_string_view().value(row).to_owned(),
        ArrowType::Utf8 => a.as_string::<i32>().value(row).to_owned(),
        ArrowType::Decimal64(_, s) => {
            decimal(i128::from(a.as_primitive::<Decimal64Type>().value(row)), *s)
        }
        ArrowType::Decimal128(_, s) => decimal(a.as_primitive::<Decimal128Type>().value(row), *s),
        ArrowType::Date32 => {
            let days = a.as_primitive::<Date32Type>().value(row);
            date(days)
        }
        ArrowType::Timestamp(_, _) => {
            let micros = a
                .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                .value(row);
            let (days, rest) = (
                micros.div_euclid(86_400_000_000),
                micros.rem_euclid(86_400_000_000),
            );
            let (h, m, s, us) = (
                rest / 3_600_000_000,
                rest / 60_000_000 % 60,
                rest / 1_000_000 % 60,
                rest % 1_000_000,
            );
            let day = date(i32::try_from(days).unwrap());
            if us == 0 {
                format!("{day} {h:02}:{m:02}:{s:02}")
            } else {
                format!("{day} {h:02}:{m:02}:{s:02}.{us:06}")
            }
        }
        other => panic!("no rendering for {other}"),
    }
}

/// A double as Python's `repr` prints it (the answers come from PySpark):
/// the shortest round-trip digits, positional between 1e-4 and 1e16 (with
/// `.0` when integral), else in exponent form (`1e+16`, `1e-05`).
fn python_float(v: f64) -> String {
    let a = v.abs();
    if !v.is_finite() {
        return if v.is_nan() {
            "nan".to_owned()
        } else if v > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    if a == 0.0 || (1e-4..1e16).contains(&a) {
        let s = v.to_string();
        return if s.contains('.') { s } else { format!("{s}.0") };
    }
    let s = format!("{v:e}");
    let (mantissa, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exp.abs())
}

fn decimal(v: i128, scale: i8) -> String {
    let neg = v < 0;
    let digits = v.unsigned_abs().to_string();
    let scale = usize::try_from(scale).unwrap();
    let digits = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = digits.split_at(digits.len() - scale);
    let sign = if neg { "-" } else { "" };
    if scale == 0 {
        format!("{sign}{int}")
    } else {
        format!("{sign}{int}.{frac}")
    }
}

fn date(days: i32) -> String {
    // Howard Hinnant's civil_from_days.
    let z = i64::from(days) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// The rows of `batches`, one string per row (`|`-separated).
pub(crate) fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| render(c.as_ref(), r)).collect();
            out.push(cells.join("|"));
        }
    }
    out
}

/// Spark's answers (`tests/scripts/gen_planner_goldens.py`).
pub(crate) fn answers(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Spark's rows of query `name`, as [`rows`] renders them.
pub(crate) fn expected_rows(answers: &serde_json::Value, name: &str) -> Vec<String> {
    answers["queries"][name]["rows"]
        .as_array()
        .unwrap_or_else(|| panic!("{name} has no answer"))
        .iter()
        .map(|r| r.as_str().unwrap().to_owned())
        .collect()
}

/// Runs all 22 queries over `dir` and compares them with `answers`.
pub(crate) fn check(dir: &Path, answers_path: &Path) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = parquet_tables(dir, &rt);
    let answers = answers(answers_path);
    let only: Option<usize> = std::env::var("KRISHIV_TPCH_QUERY")
        .ok()
        .map(|q| q.parse().unwrap());
    let mut failed = Vec::new();
    for q in 1..=22 {
        if only.is_some_and(|o| o != q) {
            continue;
        }
        let plan = tpch.plan(q);
        let start = std::time::Instant::now();
        let got = match super::run(&plan) {
            Ok(b) => rows(&b),
            Err(e) => {
                failed.push(format!("q{q:02}: {e}"));
                continue;
            }
        };
        let elapsed = start.elapsed();
        let want = expected_rows(&answers, &format!("q{q:02}"));
        eprintln!("q{q:02}: {} rows in {elapsed:.2?}", got.len());
        if got != want {
            let diff: Vec<String> = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| g != w)
                .take(3)
                .map(|(g, w)| format!("  got  {g}\n  want {w}"))
                .collect();
            failed.push(format!(
                "q{q:02}: {} rows vs {} expected\n{}\n{plan}",
                got.len(),
                want.len(),
                diff.join("\n")
            ));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
}

/// `n` without its exchange nodes.
fn without_exchanges(n: &krishiv_exec::planner::ExplainNode) -> krishiv_exec::planner::ExplainNode {
    if n.name.contains("Exchange")
        && let [input] = n.children.as_slice()
    {
        return without_exchanges(input);
    }
    krishiv_exec::planner::ExplainNode::new(
        n.name.clone(),
        n.detail.clone(),
        n.children.iter().map(without_exchanges).collect(),
    )
}

/// The optimizer differential (P2-T072a): each query planned with the
/// tables' statistics and with default estimates, which order the joins
/// differently; both plans return Spark's answer. Returns how many queries
/// got different plans.
fn join_orders_agree(dir: &Path, answers_path: &Path, skip: &[usize]) -> usize {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = parquet_tables(dir, &rt);
    let answers = answers(answers_path);
    let mut failed = Vec::new();
    let mut differ = 0;
    for q in (1..=22).filter(|q| !skip.contains(q)) {
        let want = expected_rows(&answers, &format!("q{q:02}"));
        let with = tpch.plan_sql_with(&sql(q), Order::Statistics).unwrap();
        let without = tpch.plan_sql_with(&sql(q), Order::Defaults).unwrap();
        // Join order only: estimates also pick broadcast or partitioned
        // joins, which run alike on one worker.
        if without_exchanges(with.explain()).to_string()
            == without_exchanges(without.explain()).to_string()
        {
            continue;
        }
        differ += 1;
        for (name, plan) in [("statistics", &with), ("default estimates", &without)] {
            let start = std::time::Instant::now();
            match super::run(plan) {
                Ok(b) if rows(&b) == want => {
                    eprintln!("q{q:02} ({name}): {:.2?}", start.elapsed());
                }
                Ok(b) => failed.push(format!("q{q:02} ({name}): {:?}\n{plan}", rows(&b))),
                Err(e) => failed.push(format!("q{q:02} ({name}): {e}")),
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
    differ
}

#[test]
fn tpch_sf0_01_join_orders_agree() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let differ = join_orders_agree(&dir, &dir.join("spark-answers.json"), &[]);
    assert!(differ >= 5, "only {differ} queries got another join order");
}

/// [`join_orders_agree`] at SF1 (`KRISHIV_TPCH_SF1_DIR`, as
/// [`tpch_sf1_matches_spark`]). Without statistics Q5 joins `customer` and
/// `supplier` on their nation first (a many-to-many join, ~36 billion rows
/// into the `lineitem` join at SF1), so it runs at SF 0.01 only.
#[test]
#[ignore = "needs the TPC-H SF1 data set"]
fn tpch_sf1_join_orders_agree() {
    let dir = std::env::var("KRISHIV_TPCH_SF1_DIR").expect("KRISHIV_TPCH_SF1_DIR");
    let answers =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf1-spark-answers.json");
    let differ = join_orders_agree(Path::new(&dir), &answers, &[5]);
    eprintln!("{differ} queries got another join order");
}

/// TPC-H SF 0.01 (`tests/fixtures/tpch-sf0.01-parquet`, DuckDB's dbgen)
/// against Spark 4.2.0's answers.
#[test]
fn tpch_sf0_01_matches_spark() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    check(&dir, &dir.join("spark-answers.json"));
}

/// [`tpch_sf0_01_matches_spark`] with the columnar aggregate tables
/// (`krishiv.exec.agg.rowLayout` off): both layouts return Spark's answers.
#[test]
fn tpch_sf0_01_matches_spark_with_columnar_aggregates() {
    super::AGG_ROW_LAYOUT.set(false);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    check(&dir, &dir.join("spark-answers.json"));
}

/// TPC-H SF1 (`KRISHIV_TPCH_SF1_DIR`: dbgen's tables as Parquet, one file
/// per table) against Spark 4.2.0's answers over the same files
/// (`tests/fixtures/tpch-sf1-spark-answers.json`). Run with
/// `cargo test --release -p krishiv-exec --test planner tpch_sf1 -- --ignored`.
#[test]
#[ignore = "needs the TPC-H SF1 data set"]
fn tpch_sf1_matches_spark() {
    let dir = std::env::var("KRISHIV_TPCH_SF1_DIR").expect("KRISHIV_TPCH_SF1_DIR");
    let answers =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf1-spark-answers.json");
    check(Path::new(&dir), &answers);
}

/// TPC-H Q9 and Q18 at SF1 (`KRISHIV_TPCH_SF1_DIR`) under a tight memory
/// limit (`KRISHIV_TPCH_MEMORY_MIB`, default 64) on the threaded executor
/// with the system clock: both spill and still match Spark. Run with
/// `cargo test --release -p krishiv-exec --test planner tpch_sf1_tight -- --ignored`.
#[test]
#[ignore = "needs the TPC-H SF1 data set"]
fn tpch_sf1_tight_memory_spills_and_matches_spark() {
    let dir = std::env::var("KRISHIV_TPCH_SF1_DIR").expect("KRISHIV_TPCH_SF1_DIR");
    let mib: usize = std::env::var("KRISHIV_TPCH_MEMORY_MIB")
        .map_or(64, |v| v.parse().expect("KRISHIV_TPCH_MEMORY_MIB"));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = parquet_tables(Path::new(&dir), &rt);
    let answers = answers(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf1-spark-answers.json"),
    );
    let spill_dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tpch-sf1-tight-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: spill_dir,
        ..SpillConfig::default()
    })
    .unwrap();
    for q in [9, 18] {
        let scope = manager.scope().unwrap();
        let plan = tpch
            .plan_sql_in(&sql(q), Order::Statistics, Some(&scope))
            .unwrap();
        let root = TreePool::new(&MemoryConfig {
            limit: Some(mib << 20),
            ..MemoryConfig::default()
        });
        let start = std::time::Instant::now();
        let (out, query) = super::run_in(&plan, &root);
        let elapsed = start.elapsed();
        let got = rows(&out.unwrap_or_else(|e| panic!("q{q:02} under {mib} MiB: {e}")));
        let metrics = query.metrics();
        // Bytes written to spill files, by any operator.
        let spilled: u64 = metrics
            .operators
            .values()
            .flat_map(|e| &e.extra)
            .filter(|(name, _)| name.contains("spill") && name.ends_with("bytes"))
            .filter_map(|(_, v)| v.value())
            .sum();
        let stats = root.stats();
        eprintln!(
            "q{q:02} under {mib} MiB: {elapsed:.2?}, {spilled} bytes spilled, {} rounds, {} spill requests, {} aborts",
            stats.rounds, stats.spill_requests, stats.aborts
        );
        assert_eq!(got, expected_rows(&answers, &format!("q{q:02}")), "q{q:02}");
        assert!(spilled > 0, "q{q:02} spilled under {mib} MiB");
        assert_eq!(stats.aborts, 0, "q{q:02}");
    }
}

/// The physical plans of the 22 queries over the SF 0.01 tables (their
/// footers' row counts order the joins).
#[test]
fn tpch_physical_plans() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let tpch = parquet_tables(&dir, &rt);
    for q in 1..=22 {
        insta::assert_snapshot!(format!("q{q:02}"), tpch.plan(q).to_string());
    }
}

/// Footer statistics of a table of several files: rows and nulls add up,
/// bounds widen, and integral ranges bound the distinct values.
#[test]
fn file_table_statistics_merge_footers() {
    use krishiv_exec::planner::TableProvider as _;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/planner");
    let store = Arc::new(LocalFileSystem::new_with_prefix(&dir).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let meta = rt
        .block_on(object_store::ObjectStoreExt::head(
            store.as_ref(),
            &object_store::path::Path::from("t.parquet"),
        ))
        .unwrap();
    let file = FileRef::from_meta(id, &meta);
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let schema = rt
        .block_on(io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &options))
        .unwrap()
        .schema();
    let table: arrow_schema::SchemaRef = Arc::new(arrow_schema::Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| {
                krishiv_types::DataType::from_arrow_field(f)
                    .unwrap()
                    .to_arrow_field(f.name(), f.is_nullable())
            })
            .collect::<Vec<_>>(),
    ));
    let open = |files: Vec<FileRef>| {
        rt.block_on(FileTable::open(
            Arc::clone(&format),
            io.clone(),
            Arc::clone(&table),
            files,
            options.clone(),
            &Config::new(),
            2,
        ))
        .unwrap()
        .statistics()
        .unwrap()
    };
    let one = open(vec![file.clone()]);
    let two = open(vec![file.clone(), file]);
    assert_eq!((one.rows, two.rows), (8, 16));
    let (a1, a2) = (&one.columns["a"], &two.columns["a"]);
    assert_eq!((a1.nulls, a2.nulls), (Some(2), Some(4)));
    assert_eq!(a2.min, Some(krishiv_plan::Literal::Integer(0)));
    assert_eq!(a2.max, Some(krishiv_plan::Literal::Integer(4)));
    // 0..=4: at most 5 distinct values.
    assert_eq!((a1.distinct, a2.distinct), (Some(5), Some(5)));
    assert_eq!(
        two.columns["s"].max,
        Some(krishiv_plan::Literal::String("xyz".to_owned()))
    );
    assert_eq!(two.columns["d"].min.as_ref().map(|_| ()), Some(()));
    assert!(two.columns["dt"].distinct.is_some());
}

/// TPC-H Q5, Q7, Q8 and Q9 with their joins written in a hand-picked
/// order: `lineitem` probes, every other table is built (filters stay in
/// `WHERE`, where the optimizer pushes them into the scans).
const HAND_ORDERED: [(usize, &str); 4] = [
    (
        5,
        "SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue \
         FROM lineitem JOIN orders ON l_orderkey = o_orderkey \
         JOIN customer ON c_custkey = o_custkey \
         JOIN supplier ON l_suppkey = s_suppkey AND c_nationkey = s_nationkey \
         JOIN nation ON s_nationkey = n_nationkey \
         JOIN region ON n_regionkey = r_regionkey \
         WHERE r_name = 'ASIA' AND o_orderdate >= CAST('1994-01-01' AS date) \
         AND o_orderdate < CAST('1995-01-01' AS date) \
         GROUP BY n_name ORDER BY revenue DESC",
    ),
    (
        7,
        "SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM ( \
         SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, \
         extract(year FROM l_shipdate) AS l_year, l_extendedprice * (1 - l_discount) AS volume \
         FROM lineitem JOIN supplier ON s_suppkey = l_suppkey \
         JOIN orders ON o_orderkey = l_orderkey \
         JOIN customer ON c_custkey = o_custkey \
         JOIN nation n1 ON s_nationkey = n1.n_nationkey \
         JOIN nation n2 ON c_nationkey = n2.n_nationkey \
         WHERE ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') \
         OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE')) \
         AND l_shipdate BETWEEN CAST('1995-01-01' AS date) AND CAST('1996-12-31' AS date)) AS shipping \
         GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year",
    ),
    (
        8,
        "SELECT o_year, sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / sum(volume) AS mkt_share \
         FROM (SELECT extract(year FROM o_orderdate) AS o_year, \
         l_extendedprice * (1 - l_discount) AS volume, n2.n_name AS nation \
         FROM lineitem JOIN part ON p_partkey = l_partkey \
         JOIN supplier ON s_suppkey = l_suppkey \
         JOIN orders ON l_orderkey = o_orderkey \
         JOIN customer ON o_custkey = c_custkey \
         JOIN nation n1 ON c_nationkey = n1.n_nationkey \
         JOIN region ON n1.n_regionkey = r_regionkey \
         JOIN nation n2 ON s_nationkey = n2.n_nationkey \
         WHERE r_name = 'AMERICA' \
         AND o_orderdate BETWEEN CAST('1995-01-01' AS date) AND CAST('1996-12-31' AS date) \
         AND p_type = 'ECONOMY ANODIZED STEEL') AS all_nations \
         GROUP BY o_year ORDER BY o_year",
    ),
    (
        9,
        "SELECT nation, o_year, sum(amount) AS sum_profit FROM ( \
         SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, \
         l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount \
         FROM lineitem JOIN part ON p_partkey = l_partkey \
         JOIN supplier ON s_suppkey = l_suppkey \
         JOIN partsupp ON ps_suppkey = l_suppkey AND ps_partkey = l_partkey \
         JOIN orders ON o_orderkey = l_orderkey \
         JOIN nation ON s_nationkey = n_nationkey \
         WHERE p_name LIKE '%green%') AS profit \
         GROUP BY nation, o_year ORDER BY nation, o_year DESC",
    ),
];

/// The hand-ordered queries return the same rows as the queries (SF 0.01).
#[test]
fn tpch_hand_ordered_queries_agree() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let tpch = parquet_tables(&dir, &rt);
    let answers = answers(&dir.join("spark-answers.json"));
    for (q, text) in HAND_ORDERED {
        let plan = tpch.plan_sql_with(text, Order::Written).unwrap();
        let got = rows(&super::run(&plan).unwrap());
        assert_eq!(got, expected_rows(&answers, &format!("q{q:02}")), "q{q:02}");
    }
}

/// P2-T073a: Q5, Q7, Q8 and Q9 at SF10 (`KRISHIV_TPCH_SF10_DIR`) with the
/// optimizer's join order within 2x of the hand-ordered plan; runs
/// interleaved, the median of `KRISHIV_TPCH_RUNS` (default 3) each. Run
/// with `cargo test --release -p krishiv-exec --test planner tpch_sf10 --
/// --ignored --nocapture`.
#[test]
#[ignore = "needs the TPC-H SF10 data set"]
fn tpch_sf10_join_order_within_2x_of_hand_ordered() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = std::env::var("KRISHIV_TPCH_SF10_DIR").expect("KRISHIV_TPCH_SF10_DIR");
    let runs: usize = std::env::var("KRISHIV_TPCH_RUNS")
        .ok()
        .map_or(3, |r| r.parse().unwrap());
    let tpch = parquet_tables(Path::new(&dir), &rt);
    let mut slow = Vec::new();
    for (q, text) in HAND_ORDERED {
        let mut times = [Vec::new(), Vec::new()];
        let mut results = [Vec::new(), Vec::new()];
        for _ in 0..runs {
            for (i, (query, order)) in [
                (sql(q), Order::Statistics),
                (text.to_owned(), Order::Written),
            ]
            .into_iter()
            .enumerate()
            {
                let plan = tpch.plan_sql_with(&query, order).unwrap();
                let start = std::time::Instant::now();
                results[i] = rows(&super::run(&plan).unwrap());
                times[i].push(start.elapsed());
            }
        }
        assert_eq!(results[0], results[1], "q{q:02}");
        let median = |v: &mut Vec<std::time::Duration>| {
            v.sort();
            v[v.len() / 2]
        };
        let (opt, hand) = (median(&mut times[0]), median(&mut times[1]));
        let ratio = opt.as_secs_f64() / hand.as_secs_f64();
        eprintln!("q{q:02}: optimizer {opt:.2?}, hand-ordered {hand:.2?}, ratio {ratio:.2}");
        if ratio > 2.0 {
            slow.push(q);
        }
    }
    assert!(slow.is_empty(), "over 2x the hand-ordered time: {slow:?}");
}

/// TPC-H Q21 at SF10 (`KRISHIV_TPCH_SF10_DIR`) under a memory limit
/// (`KRISHIV_TPCH_MEMORY_MIB`, default 12 GiB) completes, without a spill
/// scope (as a session runs it) or with one (`KRISHIV_TPCH_SPILL=1`), and
/// reports the operators' peak reservations. Run with `cargo test --release
/// -p krishiv-exec --test planner tpch_sf10_q21 -- --ignored --nocapture`.
#[test]
#[ignore = "needs the TPC-H SF10 data set"]
fn tpch_sf10_q21_completes_under_the_memory_limit() {
    let dir = std::env::var("KRISHIV_TPCH_SF10_DIR").expect("KRISHIV_TPCH_SF10_DIR");
    let mib: usize = std::env::var("KRISHIV_TPCH_MEMORY_MIB")
        .map_or(12 << 10, |v| v.parse().expect("KRISHIV_TPCH_MEMORY_MIB"));
    let spill = std::env::var("KRISHIV_TPCH_SPILL").is_ok_and(|v| v == "1");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let tpch = parquet_tables(Path::new(&dir), &rt);
    let spill_dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tpch-sf10-q21-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: spill_dir,
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let plan = tpch
        .plan_sql_in(&sql(21), Order::Statistics, spill.then_some(&scope))
        .unwrap();
    let root = TreePool::new(&MemoryConfig {
        limit: Some(mib << 20),
        ..MemoryConfig::default()
    });
    let start = std::time::Instant::now();
    let (out, query) = super::run_in(&plan, &root);
    let elapsed = start.elapsed();
    let metrics = query.metrics();
    let mut peaks: Vec<(u64, &str)> = metrics
        .operators
        .values()
        .map(|e| (e.metrics.peak_reserved, e.name.as_ref()))
        .collect();
    peaks.sort_unstable_by(|a, b| b.cmp(a));
    let total: u64 = peaks.iter().map(|(p, _)| p).sum();
    eprintln!(
        "q21 under {mib} MiB (spill {spill}): {elapsed:.2?}, operator peaks summed {total} bytes, largest {:?}",
        &peaks[..peaks.len().min(4)]
    );
    let got = rows(&out.unwrap_or_else(|e| panic!("q21 under {mib} MiB: {e}")));
    assert_eq!(got.len(), 100, "q21 returns 100 rows");
}
