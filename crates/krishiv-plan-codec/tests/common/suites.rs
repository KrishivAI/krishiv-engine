// SPDX-License-Identifier: Apache-2.0
//! The TPC-H and TPC-DS queries planned as the engine plans them: parse,
//! analyze, optimize, physical planning. TPC-H reads the Parquet fixture
//! (footer statistics drive join order and runtime filters); TPC-DS has no
//! data in the repository, so it plans over empty in-memory tables of the
//! Spark 4.2 TPC-DS schemas with the optimizer's default estimates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_schema::Schema;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::PhysicalPlan;
use krishiv_exec::planner::{
    DefaultPlanner, FileTable, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext,
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
use krishiv_types::{DataType, parse_data_type};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

pub(crate) struct Suite {
    ctx: AnalysisContext,
    tables: Arc<MemoryTables>,
    config: Arc<Config>,
    /// Order joins by the tables' statistics (else the defaults).
    statistics: bool,
}

pub(crate) fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `(name, text)` of each query of `suite` (`tpch` or `tpcds`), by name.
pub(crate) fn queries(suite: &str) -> Vec<(String, String)> {
    let dir = repo().join("bench/queries").join(suite);
    let mut out: Vec<(String, String)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "sql"))
        .map(|p| {
            (
                p.file_stem().unwrap().to_string_lossy().into_owned(),
                std::fs::read_to_string(&p).unwrap(),
            )
        })
        .collect();
    out.sort();
    out
}

fn config(ansi: bool) -> Config {
    let mut config = Config::new();
    config
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    config
}

impl Suite {
    /// TPC-H over the SF 0.01 Parquet files of the planner tests.
    pub(crate) fn tpch(rt: &tokio::runtime::Runtime) -> Self {
        let dir = repo().join("crates/krishiv-exec/tests/fixtures/tpch-sf0.01-parquet");
        let store = Arc::new(LocalFileSystem::new_with_prefix(&dir).unwrap());
        let sched = IoScheduler::new(IoConfig::default());
        let id = StoreId::new("file://");
        sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
        let io = sched.context();
        let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
        let options = FormatOptions::defaults(ParquetFormat.options());
        let config = config(true);
        let mut catalog = MemoryCatalog::new("spark_catalog");
        let mut tables = MemoryTables::new();
        for t in [
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ] {
            let path = object_store::path::Path::from(format!("{t}.parquet"));
            let meta = rt
                .block_on(object_store::ObjectStoreExt::head(store.as_ref(), &path))
                .unwrap();
            let files = vec![FileRef::from_meta(id.clone(), &meta)];
            let file_schema = rt
                .block_on(io.read_metadata(
                    format.as_ref(),
                    &files[0],
                    MetadataNeeds::default(),
                    &options,
                ))
                .unwrap()
                .schema();
            let (columns, fields) = columns_of(file_schema.fields().iter().map(|f| {
                (
                    f.name().clone(),
                    DataType::from_arrow_field(f).unwrap(),
                    f.is_nullable(),
                )
            }));
            catalog = catalog.with_table(&["default"], t, columns, None);
            let table = rt
                .block_on(FileTable::open(
                    Arc::clone(&format),
                    io.clone(),
                    Arc::new(Schema::new(fields)),
                    files,
                    options.clone(),
                    &config,
                    12,
                ))
                .unwrap();
            tables = tables.with_table(&["spark_catalog", "default", t], Arc::new(table));
        }
        Self {
            ctx: context(catalog, ResolutionConf::default()),
            tables: Arc::new(tables),
            config: Arc::new(config),
            statistics: true,
        }
    }

    /// TPC-DS over empty tables of the Spark 4.2 schemas.
    pub(crate) fn tpcds() -> Self {
        let file = repo().join("tests/fixtures/spark-4.2.0/golden-tables/tpcds-schemas.json");
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
        let mut catalog = MemoryCatalog::new("spark_catalog");
        let mut tables = MemoryTables::new();
        for t in doc["tables"].as_array().unwrap() {
            let name = t["name"].as_str().unwrap();
            let (columns, fields) = columns_of(t["columns"].as_array().unwrap().iter().map(|c| {
                // Char and varchar are strings to the engine.
                let ty = c["type"].as_str().unwrap();
                let ty = if ty.starts_with("char") || ty.starts_with("varchar") {
                    "string"
                } else {
                    ty
                };
                (
                    c["name"].as_str().unwrap().to_owned(),
                    parse_data_type(ty).unwrap(),
                    c["nullable"].as_bool().unwrap(),
                )
            }));
            catalog = catalog.with_table(&["default"], name, columns, None);
            let table =
                MemoryTable::new(Arc::new(Schema::new(fields)), Vec::new()).with_statistics(None);
            tables = tables.with_table(&["spark_catalog", "default", name], Arc::new(table));
        }
        Self {
            ctx: context(catalog, ResolutionConf::default()),
            tables: Arc::new(tables),
            config: Arc::new(config(true)),
            statistics: false,
        }
    }

    /// Plans `sql`.
    pub(crate) fn plan(&self, sql: &str) -> Result<PhysicalPlan, KrishivError> {
        let parsed =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .map_err(KrishivError::from)?;
        let plan: Plan = block_on(DefaultAnalyzer.analyze(&parsed, &self.ctx))?.plan;
        let plan = if self.statistics {
            krishiv_optimizer::optimize_with(
                &plan,
                &ProviderStatistics(Arc::clone(&self.tables) as _),
            )?
        } else {
            krishiv_optimizer::optimize(&plan)?
        };
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::clone(&self.config));
        DefaultPlanner.plan(&plan, &cx)
    }

    /// The tables, for instantiating a plan.
    pub(crate) fn tables(&self) -> Arc<MemoryTables> {
        Arc::clone(&self.tables)
    }
}

fn columns_of(
    cols: impl Iterator<Item = (String, DataType, bool)>,
) -> (Vec<TableColumn>, Vec<arrow_schema::Field>) {
    let mut columns = Vec::new();
    let mut fields = Vec::new();
    for (name, data_type, nullable) in cols {
        fields.push(data_type.to_arrow_field(&name, nullable));
        columns.push(TableColumn {
            name,
            data_type,
            nullable,
        });
    }
    (columns, fields)
}
