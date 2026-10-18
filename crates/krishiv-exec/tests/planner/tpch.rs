// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries through the analyzer, optimizer, planner and
//! executor over Parquet files: physical-plan snapshots, and results
//! compared with Spark 4.2.0's.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::{
    DefaultPlanner, FileTable, MemoryTables, PhysicalPlan, PhysicalPlanner, PlanningContext,
    ProviderStatistics,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, MetadataNeeds, StoreId,
};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
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
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    for &t in names {
        let path = object_store::path::Path::from(format!("{t}.parquet"));
        let meta = rt
            .block_on(object_store::ObjectStoreExt::head(store.as_ref(), &path))
            .unwrap();
        let file = FileRef::from_meta(id.clone(), &meta);
        let file_schema = rt
            .block_on(io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &options))
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
                vec![file],
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

impl Tpch {
    pub(crate) fn plan(&self, q: usize) -> PhysicalPlan {
        self.plan_sql(&sql(q))
            .unwrap_or_else(|e| panic!("q{q:02}: {e}"))
    }

    /// Parses, analyzes, optimizes (ordering joins by the tables'
    /// statistics) and plans `text`.
    pub(crate) fn plan_sql(&self, text: &str) -> Result<PhysicalPlan, KrishivError> {
        let plan =
            krishiv_sql::parse_and_lower(text, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .map_err(KrishivError::from)?;
        let plan: Plan = block_on(DefaultAnalyzer.analyze(&plan, &self.ctx))?.plan;
        let stats = ProviderStatistics(Arc::clone(&self.tables) as _);
        let plan = krishiv_optimizer::optimize_with(&plan, &stats)?;
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::clone(&self.config));
        DefaultPlanner.plan(&plan, &cx)
    }
}

/// One value as Spark's answers render it.
pub(crate) fn render(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) {
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

/// TPC-H SF 0.01 (`tests/fixtures/tpch-sf0.01-parquet`, DuckDB's dbgen)
/// against Spark 4.2.0's answers.
#[test]
fn tpch_sf0_01_matches_spark() {
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
