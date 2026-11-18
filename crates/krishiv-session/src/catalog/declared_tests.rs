// SPDX-License-Identifier: Apache-2.0
//! The read path of declared catalogs' tables in the session (design P3-T067 PR-58b) over a fake
//! catalog whose table opens as a memory table: scans read the provider opened before planning,
//! distributed stages refuse them, and #836 follow-ups N4 (a cache scan under a declared current
//! catalog) and N5 (`REFRESH TABLE spark_catalog.ns.t`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType as A, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::{MemoryTable, TableProvider};
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{BoxFuture, NameForm, TableColumn, TableIdent, TableKind, TableMeta};
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

use crate::EngineRuntime;
use crate::Session;
use crate::catalog_decl::{CatalogDecl, CatalogType, DeclMode};
use crate::catalog_registry::{
    CallerCatalog, CatalogCaller, CatalogDeps, CatalogFactory, CatalogSetup, DeclaredTable,
    LoadedDeclared, ScanInputs,
};

/// `db.t (x INT)` holding 1, 2, 3; counts its opens and refreshes.
#[derive(Debug, Default)]
struct Lake {
    opens: Arc<AtomicUsize>,
    refreshes: AtomicUsize,
}

#[derive(Debug)]
struct Rows(Arc<AtomicUsize>);

impl DeclaredTable for Rows {
    fn open(
        self: Arc<Self>,
        _inputs: ScanInputs,
    ) -> BoxFuture<'static, Result<Arc<dyn TableProvider>, KrishivError>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            let schema = Arc::new(Schema::new(vec![Field::new("x", A::Int32, true)]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            )
            .map_err(|e| crate::session::internal(e.to_string()))?;
            Ok(Arc::new(MemoryTable::new(schema, vec![Batch::new(batch)]))
                as Arc<dyn TableProvider>)
        })
    }
}

impl CallerCatalog for Lake {
    fn epoch(&self) -> u64 {
        self.refreshes.load(Ordering::SeqCst) as u64
    }

    fn load_table<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            Ok(self
                .load_declared(caller, id, version, form)
                .await?
                .map(|l| l.meta))
        })
    }

    fn load_declared<'a>(
        &'a self,
        _caller: &'a CatalogCaller,
        id: &'a TableIdent,
        _version: Option<TableVersion>,
        _form: NameForm,
    ) -> BoxFuture<'a, Result<Option<LoadedDeclared>, KrishivError>> {
        Box::pin(async move {
            if id.namespace != ["db"] || id.name != "t" {
                return Ok(None);
            }
            let meta = TableMeta {
                ident: TableIdent {
                    catalog: "lake".to_owned(),
                    namespace: vec!["db".to_owned()],
                    name: "t".to_owned(),
                },
                columns: vec![TableColumn::new("x", DataType::Integer, true)],
                version: Some(TableVersion(7)),
                kind: TableKind::Table,
                file_metadata: Default::default(),
                column_comments: Default::default(),
            };
            let table: Arc<dyn DeclaredTable> = Arc::new(Rows(Arc::clone(&self.opens)));
            Ok(Some(LoadedDeclared::new(meta, Some(table))))
        })
    }

    fn refresh<'a>(
        &'a self,
        _caller: &'a CatalogCaller,
        id: &'a TableIdent,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        self.refreshes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(id.namespace == ["db"] && id.name == "t") })
    }

    fn list_namespaces<'a>(
        &'a self,
        _caller: &'a CatalogCaller,
        parent: &'a [String],
    ) -> BoxFuture<'a, Result<Option<Vec<Vec<String>>>, KrishivError>> {
        Box::pin(async move { Ok(parent.is_empty().then(|| vec![vec!["db".to_owned()]])) })
    }

    fn list_tables<'a>(
        &'a self,
        _caller: &'a CatalogCaller,
        namespace: &'a [String],
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, KrishivError>> {
        Box::pin(async move { Ok((namespace == ["db"]).then(|| vec!["t".to_owned()])) })
    }
}

#[derive(Debug, Default)]
struct Factory(Arc<Lake>);

impl CatalogFactory for Factory {
    fn kind(&self) -> CatalogType {
        CatalogType::Memory
    }

    fn build(
        &self,
        _decl: &CatalogDecl,
        _deps: &CatalogDeps,
    ) -> Result<Arc<dyn CallerCatalog>, KrishivError> {
        Ok(Arc::clone(&self.0) as Arc<dyn CallerCatalog>)
    }
}

fn session(lake: &Arc<Lake>) -> Session {
    let rt = EngineRuntime::builder()
        .catalogs(
            CatalogSetup::new(DeclMode::Embedded)
                .factory(Arc::new(Factory(Arc::clone(lake))))
                .setting("krishiv.catalog.lake.type", "memory"),
        )
        .build()
        .unwrap();
    Session::builder().runtime(rt).build().unwrap()
}

fn xs(s: &Session, sql: &str) -> Vec<i32> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for b in batches {
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        out.extend(c.values().iter().copied());
    }
    out.sort_unstable();
    out
}

#[test]
fn a_declared_table_is_read_through_the_provider_it_opened() {
    let lake = Arc::new(Lake::default());
    let s = session(&lake);
    assert_eq!(xs(&s, "SELECT x FROM lake.db.t"), [1, 2, 3]);
    assert_eq!(xs(&s, "SELECT x FROM lake.db.t WHERE x > 1"), [2, 3]);
    // A self join reads one provider: the table is opened once per query.
    let before = lake.opens.load(Ordering::SeqCst);
    assert_eq!(
        xs(
            &s,
            "SELECT a.x FROM lake.db.t a JOIN lake.db.t b ON a.x = b.x"
        ),
        [1, 2, 3]
    );
    assert_eq!(lake.opens.load(Ordering::SeqCst), before + 1);
    // `spark_catalog`'s table of the same namespace and name stays its own.
    block_on(s.sql("CREATE NAMESPACE db")).unwrap();
    block_on(async {
        s.sql("CREATE TABLE db.t (x INT) USING parquet")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert!(xs(&s, "SELECT x FROM spark_catalog.db.t").is_empty());
    assert_eq!(xs(&s, "SELECT x FROM lake.db.t"), [1, 2, 3]);
}

#[test]
fn declared_read_refused_in_distributed_mode() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let lake = Arc::new(Lake::default());
    let s = session(&lake);
    let df = rt.block_on(s.sql("SELECT x FROM lake.db.t")).unwrap();
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(s.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    let e = analysis
        .env
        .catalog
        .task_specs(&physical, 1, 1)
        .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    assert!(
        e.to_string()
            .contains("declared catalog `lake` in a distributed stage"),
        "{e}"
    );
}

/// #836 N4: a memory-cached relation scanned under a declared current catalog is dealt as a
/// memory table, not refused as a declared table.
#[test]
fn a_cache_scan_under_a_declared_catalog_is_not_refused_in_a_stage() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let lake = Arc::new(Lake::default());
    let s = session(&lake);
    rt.block_on(async {
        s.sql("CACHE TABLE tc AS SELECT 1 AS a")
            .await?
            .collect()
            .await
    })
    .unwrap();
    rt.block_on(async { s.sql("USE lake").await?.collect().await })
        .unwrap();
    let df = rt.block_on(s.sql("SELECT a FROM tc")).unwrap();
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(s.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    analysis.env.catalog.task_specs(&physical, 1, 1).unwrap();
}

/// `REFRESH TABLE` of a declared table reaches its catalog (which moves its epoch); #836 N5:
/// `REFRESH TABLE spark_catalog.ns.t` refreshes the embedded table under any current catalog.
#[test]
fn refresh_table_reaches_declared_and_three_part_session_tables() {
    let lake = Arc::new(Lake::default());
    let s = session(&lake);
    let run = |sql: &str| block_on(async { s.sql(sql).await?.collect().await });
    run("CREATE TABLE default.e (a INT) USING parquet").unwrap();
    run("REFRESH TABLE spark_catalog.default.e").unwrap();
    run("REFRESH TABLE lake.db.t").unwrap();
    assert_eq!(lake.refreshes.load(Ordering::SeqCst), 1);
    let e = run("REFRESH TABLE lake.db.missing").unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    run("USE lake").unwrap();
    run("REFRESH TABLE spark_catalog.default.e").unwrap();
    let e = run("REFRESH TABLE spark_catalog.default.missing").unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
}

fn strings(s: &Session, sql: &str, col: usize) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for b in batches {
        let c = arrow_cast::cast(b.column(col), &A::Utf8).unwrap();
        let c = c
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        out.extend(c.iter().flatten().map(str::to_owned));
    }
    out
}

/// Listings of a declared catalog reach its listing calls: SQL and the `spark.catalog` API.
#[test]
fn declared_listings_reach_the_catalog() {
    let lake = Arc::new(Lake::default());
    let s = session(&lake);
    assert_eq!(strings(&s, "SHOW NAMESPACES IN lake", 0), ["db"]);
    assert_eq!(strings(&s, "SHOW TABLES IN lake.db", 1), ["t"]);
    let e =
        block_on(async { s.sql("SHOW TABLES IN lake.nope").await?.collect().await }).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_FOUND");
    let tables = block_on(s.catalog().list_tables(Some("lake.db"), None)).unwrap();
    assert_eq!(
        tables.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["t"]
    );
    block_on(s.catalog().set_current_catalog("lake")).unwrap();
    let dbs = block_on(s.catalog().list_databases(None)).unwrap();
    assert_eq!(
        dbs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
        ["db"]
    );
}
