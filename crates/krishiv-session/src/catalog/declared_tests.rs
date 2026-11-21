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

/// `db.t (x INT)` holding 1, 2, 3; counts its opens and refreshes. Lists the namespaces `db`
/// and `ops`, the tables `db.t`, `db.u` and `ops.v` (only `db.t` loads). With `nested`, also
/// `a` holding `a.b` (table `a.b.t`) and `locked`, whose table listing fails; `table_listings`
/// counts the calls of `list_tables`.
#[derive(Debug, Default)]
struct Lake {
    opens: Arc<AtomicUsize>,
    refreshes: AtomicUsize,
    nested: bool,
    table_listings: AtomicUsize,
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
        Box::pin(async move {
            let ns = |parts: &[&str]| parts.iter().map(|p| (*p).to_owned()).collect::<Vec<_>>();
            Ok(match parent {
                [] if self.nested => {
                    Some(vec![ns(&["a"]), ns(&["db"]), ns(&["locked"]), ns(&["ops"])])
                }
                [] => Some(vec![ns(&["db"]), ns(&["ops"])]),
                [n] if n == "db" || n == "ops" => Some(Vec::new()),
                [n] if self.nested && n == "a" => Some(vec![ns(&["a", "b"])]),
                [n] if self.nested && n == "locked" => Some(Vec::new()),
                [a, b] if self.nested && a == "a" && b == "b" => Some(Vec::new()),
                _ => None,
            })
        })
    }

    fn list_tables<'a>(
        &'a self,
        _caller: &'a CatalogCaller,
        namespace: &'a [String],
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, KrishivError>> {
        self.table_listings.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(match namespace {
                [n] if self.nested && n == "locked" => {
                    return Err(crate::session::internal("403 Forbidden".to_owned()));
                }
                [n] if self.nested && n == "a" => Some(Vec::new()),
                [a, b] if self.nested && a == "a" && b == "b" => Some(vec!["t".to_owned()]),
                [n] if n == "db" => Some(vec!["t".to_owned(), "u".to_owned()]),
                [n] if n == "ops" => Some(vec!["v".to_owned()]),
                _ => None,
            })
        })
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
    assert_eq!(strings(&s, "SHOW NAMESPACES IN lake", 0), ["db", "ops"]);
    assert_eq!(strings(&s, "SHOW TABLES IN lake.db", 1), ["t", "u"]);
    let e =
        block_on(async { s.sql("SHOW TABLES IN lake.nope").await?.collect().await }).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_FOUND");
    let tables = block_on(s.catalog().list_tables(Some("lake.db"), None)).unwrap();
    assert_eq!(
        tables.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["t", "u"]
    );
    block_on(s.catalog().set_current_catalog("lake")).unwrap();
    let dbs = block_on(s.catalog().list_databases(None)).unwrap();
    assert_eq!(
        dbs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
        ["db", "ops"]
    );
}

/// A runtime declaring `lake`.
fn lake_runtime(lake: &Arc<Lake>) -> Arc<EngineRuntime> {
    EngineRuntime::builder()
        .catalogs(
            CatalogSetup::new(DeclMode::Embedded)
                .factory(Arc::new(Factory(Arc::clone(lake))))
                .setting("krishiv.catalog.lake.type", "memory"),
        )
        .build()
        .unwrap()
}

/// A session of `name` on `rt` under the `grants` provider (viewer role underneath), as an
/// admin when `admin`.
fn grants_session(
    rt: &Arc<EngineRuntime>,
    grants: &Arc<crate::grants::Grants>,
    name: &str,
    admin: bool,
) -> Session {
    use crate::auth::{Role, RolePolicy};
    use crate::{AccessPolicy, AllowAll, Principal, PrincipalKind};
    let system: Arc<dyn AccessPolicy> = if admin {
        Arc::new(AllowAll)
    } else {
        Arc::new(RolePolicy::new(
            Some(Role::Viewer),
            Role::Admin,
            krishiv_plan::analyzer::PolicyVersion(1),
        ))
    };
    let policy = Arc::new(crate::grants::GrantsPolicy::new(
        Arc::clone(grants),
        system,
        admin,
    ));
    Session::builder()
        .runtime(Arc::clone(rt))
        .principal(Principal::new(name, PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .grants(policy)
        .build()
        .unwrap()
}

fn error_class(s: &Session, sql: &str) -> String {
    let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
    e.error_class().name().to_owned()
}

/// Under the `grants` provider a declared catalog's listings show only what the principal may
/// list (review of #902 at 29da71de4, P1): with `none`, `token` and `client-credentials` every
/// principal reaches the catalog as the server, so the engine policy is the only per-user check
/// (design P3-T067 §3.1). `SELECT` on `lake.db.t` shows that table and its namespace, never
/// `db.u` or `ops`; a hidden namespace answers as a missing one, by SQL and by `spark.catalog`.
#[test]
fn declared_listings_show_only_what_the_principal_may_list() {
    let lake = Arc::new(Lake::default());
    let grants = crate::grants::Grants::new(Arc::new(crate::grants::MemoryGrantStore::new()));
    let rt = lake_runtime(&lake);
    let root = grants_session(&rt, &grants, "root", true);
    for statement in [
        "GRANT USAGE ON CATALOG lake TO user:ana",
        "GRANT SELECT ON TABLE lake.db.t TO user:ana",
    ] {
        block_on(async { root.sql(statement).await?.collect().await })
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }
    // The admin sees everything.
    assert_eq!(strings(&root, "SHOW TABLES IN lake.db", 1), ["t", "u"]);
    let ana = grants_session(&rt, &grants, "ana", false);
    assert_eq!(strings(&ana, "SHOW NAMESPACES IN lake", 0), ["db"]);
    assert_eq!(strings(&ana, "SHOW TABLES IN lake.db", 1), ["t"]);
    assert_eq!(
        error_class(&ana, "SHOW TABLES IN lake.ops"),
        "SCHEMA_NOT_FOUND"
    );
    assert_eq!(
        error_class(&ana, "SHOW TABLES IN lake.nope"),
        "SCHEMA_NOT_FOUND"
    );
    assert_eq!(
        error_class(&ana, "SHOW NAMESPACES IN lake.ops"),
        "SCHEMA_NOT_FOUND"
    );
    assert!(strings(&ana, "SHOW NAMESPACES IN lake.db", 0).is_empty());
    let tables = block_on(ana.catalog().list_tables(Some("lake.db"), None)).unwrap();
    assert_eq!(
        tables.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["t"]
    );
    let e = block_on(ana.catalog().list_tables(Some("lake.ops"), None)).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_FOUND");
    block_on(ana.catalog().set_current_catalog("lake")).unwrap();
    let dbs = block_on(ana.catalog().list_databases(None)).unwrap();
    assert_eq!(
        dbs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
        ["db"]
    );
    assert_eq!(strings(&ana, "SHOW NAMESPACES", 0), ["db"]);
}

/// `SHOW NAMESPACES IN <db>` of a session-catalog database the grants provider hides answers as
/// for a missing one (review of #902 at 29da71de4, P2; ADR 0046 §8).
#[test]
fn show_namespaces_in_a_hidden_database_is_schema_not_found() {
    let lake = Arc::new(Lake::default());
    let grants = crate::grants::Grants::new(Arc::new(crate::grants::MemoryGrantStore::new()));
    let rt = lake_runtime(&lake);
    let root = grants_session(&rt, &grants, "root", true);
    for statement in [
        "CREATE NAMESPACE vaultns",
        // Statement-level `Browse` passes, so only the listing rule hides `vaultns`.
        "GRANT USAGE ON CATALOG spark_catalog TO user:ana",
    ] {
        block_on(async { root.sql(statement).await?.collect().await })
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }
    assert!(strings(&root, "SHOW NAMESPACES IN vaultns", 0).is_empty());
    let ana = grants_session(&rt, &grants, "ana", false);
    assert_eq!(
        error_class(&ana, "SHOW NAMESPACES IN vaultns"),
        "SCHEMA_NOT_FOUND"
    );
    assert_eq!(
        error_class(&ana, "SHOW NAMESPACES IN nope"),
        "SCHEMA_NOT_FOUND"
    );
}

/// A grant deep in nested namespaces shows every namespace above it, and the listing rule asks
/// the grant state, not the catalog (review of #902 at 60b8e6e44, P2 1 and 2): `SELECT` on
/// `lake.a.b.t` shows `a` in `lake` and `a.b` in `lake.a`; `SHOW TABLES IN lake.a` is empty, not
/// `SCHEMA_NOT_FOUND`. `SHOW NAMESPACES` makes no table listing, so `locked`, whose table listing
/// fails, fails no listing.
#[test]
fn nested_namespace_grants_show_their_parents_without_table_listings() {
    let lake = Arc::new(Lake {
        nested: true,
        ..Lake::default()
    });
    let grants = crate::grants::Grants::new(Arc::new(crate::grants::MemoryGrantStore::new()));
    let rt = lake_runtime(&lake);
    let root = grants_session(&rt, &grants, "root", true);
    for statement in [
        "GRANT USAGE ON CATALOG lake TO user:ana",
        "GRANT SELECT ON TABLE lake.a.b.t TO user:ana",
    ] {
        block_on(async { root.sql(statement).await?.collect().await })
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }
    assert_eq!(
        strings(&root, "SHOW NAMESPACES IN lake", 0),
        ["a", "db", "locked", "ops"]
    );
    let ana = grants_session(&rt, &grants, "ana", false);
    let before = lake.table_listings.load(Ordering::SeqCst);
    assert_eq!(strings(&ana, "SHOW NAMESPACES IN lake", 0), ["a"]);
    assert_eq!(strings(&ana, "SHOW NAMESPACES IN lake.a", 0), ["a.b"]);
    block_on(ana.catalog().set_current_catalog("lake")).unwrap();
    let dbs = block_on(ana.catalog().list_databases(None)).unwrap();
    assert_eq!(
        dbs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
        ["a"]
    );
    assert_eq!(lake.table_listings.load(Ordering::SeqCst), before);
    assert!(strings(&ana, "SHOW TABLES IN lake.a", 1).is_empty());
    assert_eq!(strings(&ana, "SHOW TABLES IN lake.a.b", 1), ["t"]);
    assert_eq!(
        error_class(&ana, "SHOW NAMESPACES IN lake.locked"),
        "SCHEMA_NOT_FOUND"
    );
    assert_eq!(
        error_class(&ana, "SHOW NAMESPACES IN lake.db"),
        "SCHEMA_NOT_FOUND"
    );
}

/// A grant on a namespace alone shows the namespace, and `SHOW TABLES` in it is empty (review of
/// #902 at 60b8e6e44, nit).
#[test]
fn a_namespace_only_grant_lists_the_namespace_and_no_tables() {
    let lake = Arc::new(Lake::default());
    let grants = crate::grants::Grants::new(Arc::new(crate::grants::MemoryGrantStore::new()));
    let rt = lake_runtime(&lake);
    let root = grants_session(&rt, &grants, "root", true);
    for statement in [
        "GRANT USAGE ON CATALOG lake TO user:ana",
        "GRANT USAGE ON NAMESPACE lake.ops TO user:ana",
    ] {
        block_on(async { root.sql(statement).await?.collect().await })
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }
    let ana = grants_session(&rt, &grants, "ana", false);
    assert_eq!(strings(&ana, "SHOW NAMESPACES IN lake", 0), ["ops"]);
    assert!(strings(&ana, "SHOW TABLES IN lake.ops", 1).is_empty());
}
