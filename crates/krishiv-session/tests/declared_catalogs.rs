// SPDX-License-Identifier: Apache-2.0
//! Declared catalogs on a runtime (design P3-T067 §2.2-§2.4, §7; PR-58a): the registry and its
//! factories, startup refusals, per-tenant visibility, `system.catalogs` and the server-mode
//! refusal of runtime catalog keys. The factory here is a fake (the `iceberg-rest` one lives in
//! `krishiv-catalogs`, feature `session`).
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, NameForm, PolicyVersion, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::catalog_decl::{CatalogDecl, CatalogType, DeclMode};
use krishiv_session::catalog_registry::{
    CallerCatalog, CatalogDeps, CatalogFactory, CatalogSetup, ServerIdentity,
};
use krishiv_session::{EngineRuntime, Principal, PrincipalKind, Session};
use krishiv_types::DataType;

/// What a [`OneTable`] holds, shared with the test: its epoch, and whether `x` is a string.
#[derive(Debug, Default)]
struct Shape {
    epoch: AtomicU64,
    string: AtomicBool,
}

/// A catalog holding `db.t (x INT)` (`x STRING` once its [`Shape`] says so).
#[derive(Debug)]
struct OneTable(String, Arc<Shape>);

impl Catalog for OneTable {
    fn name(&self) -> &str {
        &self.0
    }

    fn epoch(&self) -> u64 {
        self.1.epoch.load(Ordering::SeqCst)
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _version: Option<TableVersion>,
        _form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        let ty = if self.1.string.load(Ordering::SeqCst) {
            DataType::String
        } else {
            DataType::Integer
        };
        Box::pin(async move {
            Ok(
                (id.namespace == ["db"] && id.name == "t").then(|| TableMeta {
                    ident: id.clone(),
                    columns: vec![TableColumn::new("x", ty, true)],
                    version: None,
                    kind: TableKind::Table,
                    file_metadata: Default::default(),
                    column_comments: Default::default(),
                }),
            )
        })
    }
}

/// Builds [`OneTable`] for `memory` declarations; the first `fail_first` builds fail; each
/// build takes `delay`.
#[derive(Debug, Default)]
struct Fake {
    builds: AtomicUsize,
    fail_first: usize,
    delay: std::time::Duration,
    shape: Arc<Shape>,
}

impl CatalogFactory for Fake {
    fn kind(&self) -> CatalogType {
        CatalogType::Memory
    }

    fn build(
        &self,
        decl: &CatalogDecl,
        _deps: &CatalogDeps,
    ) -> Result<Arc<dyn CallerCatalog>, KrishivError> {
        let n = self.builds.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.delay);
        if n < self.fail_first {
            return Err(krishiv_common::error::KrishivCatalogFailure {
                catalog: decl.name.clone(),
                relation_name: "the catalog".to_owned(),
            }
            .into());
        }
        Ok(Arc::new(ServerIdentity(Arc::new(OneTable(
            decl.name.clone(),
            Arc::clone(&self.shape),
        )))))
    }
}

fn setup(mode: DeclMode, pairs: &[(&str, &str)], factory: Arc<Fake>) -> CatalogSetup {
    let mut s = CatalogSetup::new(mode).factory(factory);
    for (k, v) in pairs {
        s = s.setting(*k, *v);
    }
    s
}

fn runtime(mode: DeclMode, pairs: &[(&str, &str)]) -> Arc<EngineRuntime> {
    EngineRuntime::builder()
        .catalogs(setup(mode, pairs, Arc::default()))
        .build()
        .unwrap()
}

/// A session on `rt` for a principal of `tenant` with `role` (a server's per-caller session).
fn session_as(rt: &Arc<EngineRuntime>, tenant: Option<&str>, role: Role) -> Session {
    let base = Session::builder().runtime(Arc::clone(rt)).build().unwrap();
    let mut p = Principal::new("u", PrincipalKind::User);
    p.tenant = tenant.map(str::to_owned);
    base.new_session_as(
        p,
        Arc::new(RolePolicy::new(Some(role), Role::Admin, PolicyVersion(1))),
    )
}

fn run(s: &Session, sql: &str) -> Result<Vec<arrow_array::RecordBatch>, KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await })
}

/// One column of `batches` as text (strings, string views and booleans).
fn column(batches: &[arrow_array::RecordBatch], column: usize) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        let c = b.column(column);
        if let Some(v) = c.as_string_view_opt() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else if let Some(v) = c.as_string_opt::<i32>() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else if let Some(v) = c.as_boolean_opt() {
            out.extend((0..v.len()).map(|i| v.value(i).to_string()));
        } else {
            let v = c.as_string::<i64>();
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        }
    }
    out
}

fn strings(s: &Session, sql: &str, index: usize) -> Vec<String> {
    column(&run(s, sql).unwrap(), index)
}

fn schema_of(s: &Session, sql: &str) -> Result<Vec<String>, KrishivError> {
    block_on(async {
        let schema = s.sql(sql).await?.schema().await?;
        Ok(schema.fields().iter().map(|f| f.name().clone()).collect())
    })
}

const LAKE: [(&str, &str); 1] = [("krishiv.catalog.lake.type", "memory")];

#[test]
fn declared_catalog_resolves_through_its_factory() {
    let rt = runtime(DeclMode::Server, &LAKE);
    let s = session_as(&rt, None, Role::Operator);
    assert_eq!(schema_of(&s, "SELECT * FROM lake.db.t").unwrap(), ["x"]);
    assert_eq!(schema_of(&s, "SELECT * FROM LAKE.db.t").unwrap(), ["x"]);
    assert_eq!(
        strings(&s, "SHOW CATALOGS", 0),
        ["lake", "spark_catalog", "system"]
    );
    // The Catalog API leaves out `system` (Spark has no such catalog); `SHOW CATALOGS` lists it.
    assert_eq!(s.catalog().list_catalogs(), ["lake", "spark_catalog"]);
    run(&s, "USE lake").unwrap();
    assert_eq!(strings(&s, "SELECT current_catalog()", 0), ["lake"]);
    assert_eq!(strings(&s, "SHOW CURRENT NAMESPACE", 0), ["lake"]);
}

#[test]
fn type_without_a_factory_fails_startup() {
    let e = EngineRuntime::builder()
        .catalogs(setup(
            DeclMode::Server,
            &[
                ("krishiv.catalog.h.type", "hive"),
                ("krishiv.catalog.h.uri", "thrift://metastore:9083"),
            ],
            Arc::default(),
        ))
        .build()
        .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    let text = e.to_string();
    assert!(text.contains("`hive`") && text.contains("`h`"), "{text}");
}

#[test]
fn unresolved_secret_reference_fails_startup() {
    let e = EngineRuntime::builder()
        .catalogs(setup(
            DeclMode::Server,
            &[
                ("krishiv.catalog.lake.type", "memory"),
                ("krishiv.catalog.lake.auth", "token"),
                (
                    "krishiv.catalog.lake.token",
                    "env:KRISHIV_TEST_SURELY_UNSET_CATALOG_TOKEN",
                ),
            ],
            Arc::default(),
        ))
        .build()
        .unwrap_err();
    assert!(e.to_string().contains("krishiv.catalog.lake.token"), "{e}");
}

#[test]
fn a_failed_build_is_not_kept() {
    let factory = Arc::new(Fake {
        fail_first: 1,
        ..Fake::default()
    });
    let rt = EngineRuntime::builder()
        .catalogs(setup(DeclMode::Embedded, &LAKE, Arc::clone(&factory)))
        .build()
        .unwrap();
    let s = Session::builder().runtime(rt).build().unwrap();
    let e = schema_of(&s, "SELECT * FROM lake.db.t").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_CATALOG_FAILURE");
    assert_eq!(schema_of(&s, "SELECT * FROM lake.db.t").unwrap(), ["x"]);
    assert_eq!(schema_of(&s, "SELECT * FROM lake.db.t").unwrap(), ["x"]);
    assert_eq!(
        factory.builds.load(Ordering::SeqCst),
        2,
        "built once, then shared"
    );
}

/// A frame's analysis is reused only while every catalog it read is at the epoch it read: a
/// declared catalog's table that changed is analyzed again (CodeRabbit #836).
#[test]
fn a_frame_is_analyzed_again_when_a_declared_catalog_changes() {
    let factory = Arc::new(Fake::default());
    let rt = EngineRuntime::builder()
        .catalogs(setup(DeclMode::Embedded, &LAKE, Arc::clone(&factory)))
        .build()
        .unwrap();
    let s = Session::builder().runtime(rt).build().unwrap();
    let df = block_on(s.sql("SELECT * FROM lake.db.t")).unwrap();
    let type_of = || {
        let schema = block_on(df.schema()).unwrap();
        schema.field(0).data_type().clone()
    };
    assert_eq!(type_of(), arrow_schema::DataType::Int32);
    factory.shape.string.store(true, Ordering::SeqCst);
    // Unchanged epoch: the analysis is reused.
    assert_eq!(type_of(), arrow_schema::DataType::Int32);
    factory.shape.epoch.fetch_add(1, Ordering::SeqCst);
    assert_eq!(type_of(), arrow_schema::DataType::Utf8View);
}

/// With a declared catalog current, the Catalog API refuses what it would answer from the
/// embedded catalog's namespaces and tables (the catalog resolves names only, PR-58a), never
/// labelling them with the declared catalog's name; `spark_catalog.<db>` stays the embedded one.
#[test]
fn the_catalog_api_never_lists_embedded_namespaces_as_a_declared_catalogs() {
    let rt = runtime(DeclMode::Embedded, &LAKE);
    let s = Session::builder().runtime(rt).build().unwrap();
    run(&s, "CREATE TABLE default.e (a INT) USING parquet").unwrap();
    let c = s.catalog();
    block_on(c.set_current_catalog("lake")).unwrap();
    fn class<T: std::fmt::Debug>(r: Result<T, KrishivError>) -> String {
        r.unwrap_err().error_class().name().to_owned()
    }
    let unsupported = "KRISHIV_UNSUPPORTED_FEATURE";
    assert_eq!(class(block_on(c.list_databases(None))), unsupported);
    // Flight SQL's listing names `spark_catalog` itself, so it does not follow `USE lake`.
    let embedded: Vec<_> = block_on(c.list_embedded_databases())
        .unwrap()
        .into_iter()
        .map(|d| (d.catalog, d.name))
        .collect();
    assert_eq!(
        embedded,
        [("spark_catalog".to_owned(), "default".to_owned())]
    );
    assert_eq!(class(block_on(c.list_tables(None, None))), unsupported);
    for name in ["default", "lake.db"] {
        assert_eq!(class(block_on(c.get_database(name))), unsupported, "{name}");
        assert_eq!(
            class(block_on(c.database_exists(name))),
            unsupported,
            "{name}"
        );
        assert_eq!(
            class(block_on(c.list_tables(Some(name), None))),
            unsupported,
            "{name}"
        );
    }
    for name in ["e", "default.e", "lake.db.t"] {
        assert_eq!(class(block_on(c.get_table(name))), unsupported, "{name}");
        assert_eq!(class(block_on(c.table_exists(name))), unsupported, "{name}");
    }
    // Qualified with `spark_catalog`, the embedded namespace and table are its own.
    let d = block_on(c.get_database("spark_catalog.default")).unwrap();
    assert_eq!(d.catalog, "spark_catalog");
    assert!(block_on(c.database_exists("spark_catalog.default")).unwrap());
    let t = block_on(c.get_table("spark_catalog.default.e")).unwrap();
    assert_eq!(t.catalog.as_deref(), Some("spark_catalog"));
}

/// With a declared catalog current, the SQL namespace and table commands that would resolve in it
/// refuse rather than read or change the embedded catalog in its name, as the Catalog API does;
/// `spark_catalog.<db>` forms, `global_temp` and temporary views keep working.
#[test]
fn sql_namespace_commands_never_answer_for_a_declared_catalog() {
    let rt = runtime(DeclMode::Embedded, &LAKE);
    let s = Session::builder().runtime(rt).build().unwrap();
    run(&s, "CREATE TABLE default.e (a INT) USING parquet").unwrap();
    run(&s, "CREATE NAMESPACE n0").unwrap();
    run(&s, "CREATE FUNCTION default.f() RETURNS INT RETURN 5").unwrap();
    run(&s, "CACHE TABLE default.e").unwrap();
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS a").unwrap();
    run(&s, "USE lake").unwrap();
    let mut wrong = Vec::new();
    for sql in [
        "SHOW NAMESPACES",
        "SHOW DATABASES",
        "SHOW TABLES",
        "SHOW VIEWS",
        "SHOW TABLE EXTENDED LIKE 'e'",
        "SHOW TABLES IN default",
        "SHOW VIEWS IN default",
        "DESCRIBE NAMESPACE default",
        "CREATE NAMESPACE n1",
        "DROP NAMESPACE n0",
        "ALTER NAMESPACE default SET DBPROPERTIES ('k' = 'v')",
        "USE NAMESPACE default",
        "USE default",
        "CREATE TABLE t2 (a INT) USING parquet",
        "CREATE TABLE default.t3 (a INT) USING parquet",
        "CREATE VIEW v1 AS SELECT 1 AS a",
        "DROP TABLE e",
        "DROP TABLE default.e",
        "SHOW COLUMNS IN e",
        "SHOW TBLPROPERTIES default.e",
        "SHOW CREATE TABLE default.e",
        "ALTER TABLE default.e ADD COLUMNS (b INT)",
        "CREATE FUNCTION f2() RETURNS INT RETURN 1",
        "DROP FUNCTION f",
        "DROP FUNCTION IF EXISTS f",
        "REFRESH TABLE e",
        "REFRESH TABLE default.e",
        "UNCACHE TABLE e",
        "UNCACHE TABLE default.e",
        "UNCACHE TABLE IF EXISTS e",
    ] {
        match run(&s, sql) {
            Err(e)
                if e.error_class().name() == "KRISHIV_UNSUPPORTED_FEATURE"
                    && e.to_string().contains("`lake`") => {}
            Err(e) => wrong.push(format!("{sql}: {e}")),
            Ok(b) => wrong.push(format!(
                "{sql}: ok, {} rows",
                b.iter()
                    .map(arrow_array::RecordBatch::num_rows)
                    .sum::<usize>()
            )),
        }
    }
    // The Catalog API calls that run those statements.
    for (call, r) in [
        ("refresh_table", block_on(s.catalog().refresh_table("e"))),
        ("uncache_table", block_on(s.catalog().uncache_table("e"))),
    ] {
        match r {
            Err(e) if e.error_class().name() == "KRISHIV_UNSUPPORTED_FEATURE" => {}
            other => wrong.push(format!("{call}: {other:?}")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    // Qualified with `spark_catalog`, temporary views and `global_temp` keep working.
    assert_eq!(
        strings(&s, "SHOW TABLES IN spark_catalog.default", 1),
        ["e", "tv"]
    );
    let describe = run(&s, "DESCRIBE NAMESPACE spark_catalog.default").unwrap();
    assert_eq!(
        &column(&describe, 0)[..2],
        ["Catalog Name", "Namespace Name"]
    );
    assert_eq!(&column(&describe, 1)[..2], ["spark_catalog", "default"]);
    assert_eq!(strings(&s, "SHOW TABLES IN global_temp", 1), ["tv"]);
    // Review 58b2 F1: a catalog named in `SHOW NAMESPACES IN` wins over the current one.
    assert_eq!(
        strings(&s, "SHOW NAMESPACES IN spark_catalog", 0),
        ["default", "n0"]
    );
    assert_eq!(strings(&s, "SELECT CAST(a AS STRING) FROM tv", 0), ["1"]);
    assert!(run(&s, "DROP VIEW tv").unwrap().is_empty());
    assert!(run(&s, "SELECT * FROM tv").is_err());
    run(&s, "CREATE NAMESPACE spark_catalog.n2").unwrap();
    assert!(block_on(s.catalog().database_exists("spark_catalog.n2")).unwrap());
    run(&s, "DROP NAMESPACE spark_catalog.n2").unwrap();
    assert!(!block_on(s.catalog().database_exists("spark_catalog.n2")).unwrap());
    // Nothing was changed in the embedded catalog: its namespaces, table, function and cache.
    run(&s, "USE spark_catalog").unwrap();
    assert_eq!(strings(&s, "SHOW NAMESPACES", 0), ["default", "n0"]);
    assert_eq!(strings(&s, "SHOW TABLES", 1), ["e"]);
    assert_eq!(strings(&s, "SHOW COLUMNS IN e", 0), ["a"]);
    assert_eq!(strings(&s, "SELECT CAST(f() AS STRING)", 0), ["5"]);
    assert!(!block_on(s.catalog().function_exists("f2")).unwrap());
    assert!(block_on(s.catalog().is_cached("e")).unwrap());
}

/// With a declared catalog current, a two-part name in a query resolves in that catalog, as in
/// Spark, never as the embedded catalog's view or function; a table the declared catalog has is
/// refused before it is read (the catalog resolves names only for now). `spark_catalog.` names, and
/// the tables and functions a stored view of `spark_catalog` reads, stay the embedded ones.
#[test]
fn queries_under_a_declared_catalog_never_read_embedded_views_or_functions() {
    let rt = runtime(DeclMode::Embedded, &LAKE);
    let s = Session::builder().runtime(rt).build().unwrap();
    for sql in [
        "CREATE TABLE default.e (a INT) USING parquet",
        "INSERT INTO default.e VALUES (1)",
        "CREATE VIEW default.v AS SELECT 42 AS a",
        "CREATE FUNCTION default.f() RETURNS INT RETURN 5",
        "CREATE VIEW default.w AS SELECT default.f() + a AS b FROM default.e",
    ] {
        run(&s, sql).unwrap();
    }
    run(&s, "USE lake").unwrap();
    let error = |sql: &str| {
        let e = run(&s, sql).unwrap_err();
        (e.error_class().name().to_owned(), e.to_string())
    };
    let (class, text) = error("SELECT * FROM default.v");
    assert_eq!(class, "TABLE_OR_VIEW_NOT_FOUND", "{text}");
    assert!(text.contains("`default`.`v`"), "{text}");
    let (class, text) = error("SELECT default.f()");
    assert_eq!(class, "UNRESOLVED_ROUTINE", "{text}");
    assert!(text.contains("`default`.`f`"), "{text}");
    // A table `lake` has: its name resolves, its read is refused, never an internal error.
    for sql in ["SELECT * FROM db.t", "SELECT * FROM lake.db.t"] {
        let (class, text) = error(sql);
        assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE", "{sql}: {text}");
        assert!(text.contains("`lake`"), "{sql}: {text}");
    }
    assert_eq!(schema_of(&s, "SELECT * FROM db.t").unwrap(), ["x"]);
    // No write statement reaches a declared catalog's table in 0.3 (docs/guides/lakehouse-tables.md says so): it is
    // refused, whatever the catalog's `option.write`, and nothing of the embedded catalog changes.
    for sql in [
        "INSERT INTO lake.db.t VALUES (1)",
        "INSERT OVERWRITE lake.db.t VALUES (1)",
        "DELETE FROM lake.db.t WHERE x = 1",
        "UPDATE lake.db.t SET x = 2",
        "MERGE INTO lake.db.t USING default.e ON x = a WHEN MATCHED THEN DELETE",
    ] {
        // `error` panics when the statement succeeds; the class says which refusal it is.
        let (class, text) = error(sql);
        assert!(!class.is_empty(), "{sql}: {text}");
    }
    // The embedded catalog's objects by their full names.
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(a AS STRING) FROM spark_catalog.default.v",
            0
        ),
        ["42"]
    );
    assert_eq!(
        strings(&s, "SELECT CAST(spark_catalog.default.f() AS STRING)", 0),
        ["5"]
    );
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(b AS STRING) FROM spark_catalog.default.w",
            0
        ),
        ["6"]
    );
}

/// The error text with the catalog name replaced, so a hidden and an undeclared name compare.
fn error_as(s: &Session, sql: &str, name: &str) -> String {
    let e = run(s, sql).unwrap_err();
    format!("{}: {}", e.error_class().name(), e).replace(name, "<c>")
}

#[test]
fn catalog_hidden_from_other_tenant() {
    let rt = runtime(
        DeclMode::Server,
        &[
            ("krishiv.catalog.lake.type", "memory"),
            ("krishiv.catalog.lake.tenants", "acme"),
        ],
    );
    let acme = session_as(&rt, Some("acme"), Role::Admin);
    assert_eq!(schema_of(&acme, "SELECT * FROM lake.db.t").unwrap(), ["x"]);
    for (tenant, who) in [(Some("globex"), "other tenant"), (None, "no tenant")] {
        let s = session_as(&rt, tenant, Role::Admin);
        // Absent, not denied: the same answer as for a name nobody declared.
        for sql in ["SELECT * FROM {}.db.t", "USE {}", "DESCRIBE TABLE {}.db.t"] {
            assert_eq!(
                error_as(&s, &sql.replace("{}", "lake"), "lake"),
                error_as(&s, &sql.replace("{}", "nope"), "nope"),
                "{who}: {sql}"
            );
        }
        assert_eq!(strings(&s, "SHOW CATALOGS", 0), ["spark_catalog", "system"]);
        assert!(strings(&s, "SELECT name FROM system.catalogs", 0).is_empty());
        // The Catalog API answers as for an undeclared name too, whatever the case.
        let set_err = |name: &str| {
            let e = block_on(s.catalog().set_current_catalog(name)).unwrap_err();
            format!("{}: {}", e.error_class().name(), e).replace(name, "<c>")
        };
        for name in ["lake", "LAKE"] {
            assert_eq!(set_err(name), set_err("nope"), "{who}: {name}");
        }
        assert_eq!(s.catalog().list_catalogs(), ["spark_catalog"], "{who}");
    }
}

#[test]
fn system_catalogs_shows_no_secret_or_uri() {
    // A canary: the token is a reference to a variable every test environment has, so
    // its value is known here and must appear nowhere; the uri carries userinfo.
    let canary = std::env::var("PATH").unwrap();
    let pairs = [
        ("krishiv.catalog.lake.type", "memory"),
        ("krishiv.catalog.lake.auth", "token"),
        ("krishiv.catalog.lake.token", "env:PATH"),
        (
            "krishiv.catalog.lake.uri",
            "https://user:canary-pw@lake.example.com/api",
        ),
        ("krishiv.catalog.other.type", "memory"),
        ("krishiv.catalog.default", "lake"),
    ];
    let rt = runtime(DeclMode::Server, &pairs);
    let s = session_as(&rt, None, Role::Operator);
    let batches = run(&s, "SELECT * FROM system.catalogs ORDER BY name").unwrap();
    let names: Vec<String> = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(
        names,
        ["name", "type", "auth", "storage", "delegation", "default"]
    );
    let rows: Vec<Vec<String>> = (0..6).map(|i| column(&batches, i)).collect();
    assert_eq!(rows[0], ["lake", "other"]);
    assert_eq!(rows[1], ["memory", "memory"]);
    assert_eq!(rows[2], ["token", "none"]);
    assert_eq!(rows[3], ["allowed-prefixes", "allowed-prefixes"]);
    assert_eq!(rows[5], ["true", "false"]);
    let shown = format!("{rows:?}");
    for leak in [canary.as_str(), "canary-pw", "user:", "lake.example.com"] {
        assert!(!shown.contains(leak), "{leak} in {shown}");
    }
    let debug = format!("{:?}", rt.catalogs());
    for leak in [canary.as_str(), "canary-pw"] {
        assert!(!debug.contains(leak), "{leak} in {debug}");
    }
    // `krishiv.catalog.default`: a new session starts in it.
    assert_eq!(strings(&s, "SELECT current_catalog()", 0), ["lake"]);
}

const KEYS: [&str; 4] = [
    "krishiv.catalog.lake.type",
    "krishiv.catalog.x.uri",
    "spark.sql.catalog.x",
    "spark.sql.catalog.lake.uri",
];

#[test]
fn session_set_catalog_refused_in_server_mode() {
    let rt = runtime(DeclMode::Server, &LAKE);
    for role in [Role::Viewer, Role::Operator, Role::Admin] {
        let s = session_as(&rt, None, role);
        for key in KEYS {
            for sql in [format!("SET {key} = memory"), format!("RESET {key}")] {
                let e = run(&s, &sql).unwrap_err();
                assert_eq!(
                    e.error_class().name(),
                    "CANNOT_MODIFY_STATIC_CONFIG",
                    "{role:?}: {sql}"
                );
            }
            let e = s.conf().set(key, "memory").unwrap_err();
            assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG");
            assert!(!s.conf().is_modifiable(key));
        }
        // A registered catalog setting is not a declaration: its own rules apply.
        assert!(s.conf().is_modifiable("spark.sql.ansi.enabled"));
    }
    // A session builder on a server runtime cannot declare one either (decision 7).
    for key in KEYS {
        let e = Session::builder()
            .runtime(Arc::clone(&rt))
            .config(key, "memory")
            .build()
            .unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "CANNOT_MODIFY_STATIC_CONFIG",
            "{key}"
        );
    }
    // The refusal survives a clone and a new session.
    let s = session_as(&rt, None, Role::Admin);
    for copy in [s.clone_session().unwrap(), s.new_session()] {
        assert!(copy.conf().set("spark.sql.catalog.x", "y").is_err());
    }
}

#[test]
fn embedded_set_catalog_unchanged() {
    let s = Session::new().unwrap();
    // Spark's form is kept as any unknown Spark key; Krishiv's is an unknown key, as before.
    s.conf()
        .set(
            "spark.sql.catalog.x",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .unwrap();
    assert!(s.conf().is_modifiable("spark.sql.catalog.x"));
    let e = s
        .conf()
        .set("krishiv.catalog.x.type", "memory")
        .unwrap_err();
    assert_ne!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG");
    // An embedded runtime with declarations still takes runtime keys from its sessions.
    let rt = runtime(DeclMode::Embedded, &LAKE);
    let s = Session::builder()
        .runtime(rt)
        .config("spark.sql.catalog.y", "z")
        .build()
        .unwrap();
    assert_eq!(schema_of(&s, "SELECT * FROM lake.db.t").unwrap(), ["x"]);
}

/// Review 58a2 N2: two first uses at once both succeed with the same catalog; later uses build
/// nothing more.
#[test]
fn concurrent_first_uses_both_succeed() {
    let factory = Arc::new(Fake {
        delay: std::time::Duration::from_millis(200),
        ..Fake::default()
    });
    let rt = EngineRuntime::builder()
        .catalogs(setup(DeclMode::Embedded, &LAKE, Arc::clone(&factory)))
        .build()
        .unwrap();
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let rt = Arc::clone(&rt);
            std::thread::spawn(move || {
                let s = Session::builder().runtime(rt).build().unwrap();
                schema_of(&s, "SELECT * FROM lake.db.t").map_err(|e| e.to_string())
            })
        })
        .collect();
    for t in threads {
        assert_eq!(t.join().unwrap().unwrap(), ["x"]);
    }
    let built = factory.builds.load(Ordering::SeqCst);
    assert!((1..=2).contains(&built), "{built}");
    let s = Session::builder().runtime(rt).build().unwrap();
    assert_eq!(schema_of(&s, "SELECT * FROM lake.db.t").unwrap(), ["x"]);
    assert_eq!(
        factory.builds.load(Ordering::SeqCst),
        built,
        "the stored catalog is shared"
    );
}

/// Review 58a2 M3: a server runtime takes no declaration through its configuration either.
#[test]
fn server_runtime_refuses_declarations_in_its_configuration() {
    for key in KEYS {
        let e = EngineRuntime::builder()
            .config(key, "memory")
            .catalogs(setup(DeclMode::Server, &[], Arc::default()))
            .build()
            .unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "CANNOT_MODIFY_STATIC_CONFIG",
            "{key}"
        );
    }
    // Registered catalog settings stay configuration.
    EngineRuntime::builder()
        .config("krishiv.catalog.requestTimeout", "5s")
        .catalogs(setup(DeclMode::Server, &[], Arc::default()))
        .build()
        .unwrap();
}

/// The callers a [`Recording`] catalog saw: principal id, method, credential token.
type Calls = Arc<std::sync::Mutex<Vec<(Option<String>, Option<String>, Option<String>)>>>;

/// [`OneTable`] that records the caller of every load.
#[derive(Debug)]
struct Recording(Arc<dyn CallerCatalog>, Calls);

impl CallerCatalog for Recording {
    fn load_table<'a>(
        &'a self,
        caller: &'a krishiv_session::catalog_registry::CatalogCaller,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        self.1.lock().unwrap().push((
            caller.identity.principal_id.clone(),
            caller.method.map(|m| m.as_str().to_owned()),
            caller
                .credential
                .as_ref()
                .map(|c| c.token().expose_secret().to_owned()),
        ));
        self.0.load_table(caller, id, version, form)
    }
}

#[derive(Debug, Default)]
struct RecordingFactory(Calls);

impl CatalogFactory for RecordingFactory {
    fn kind(&self) -> CatalogType {
        CatalogType::Memory
    }

    fn build(
        &self,
        decl: &CatalogDecl,
        _deps: &CatalogDeps,
    ) -> Result<Arc<dyn CallerCatalog>, KrishivError> {
        let inner = Arc::new(ServerIdentity(Arc::new(OneTable(
            decl.name.clone(),
            Arc::default(),
        ))));
        Ok(Arc::new(Recording(inner, Arc::clone(&self.0))))
    }
}

/// P3-T067 §3.2/§3.3: a delegated catalog gets each query's caller (identity key and the
/// session's latest user token, a fresher one replacing it); `system.catalogs` shows the mode.
#[test]
fn delegated_catalog_calls_carry_the_sessions_caller() {
    use krishiv_session::auth::{CredentialKind, ExchangeClaims, InboundCredential, Method};
    let factory = Arc::new(RecordingFactory::default());
    let calls = Arc::clone(&factory.0);
    let mut setup = CatalogSetup::new(DeclMode::Server).factory(factory);
    for (k, v) in [
        ("krishiv.catalog.lake.type", "memory"),
        ("krishiv.catalog.lake.auth", "delegated"),
        ("krishiv.catalog.lake.delegation", "forward"),
        ("krishiv.catalog.lake.audience", "lake"),
        ("krishiv.catalog.plain.type", "memory"),
    ] {
        setup = setup.setting(k, v);
    }
    let rt = EngineRuntime::builder().catalogs(setup).build().unwrap();
    let s = session_as(&rt, None, Role::Operator);
    let rows = run(
        &s,
        "SELECT name, auth, delegation FROM system.catalogs ORDER BY name",
    )
    .unwrap();
    assert_eq!(column(&rows, 0), ["lake", "plain"]);
    assert_eq!(column(&rows, 1), ["delegated", "none"]);
    // `delegation` is null for a catalog that is not delegated.
    assert_eq!(rows[0].column(2).null_count(), 1);
    assert_eq!(column(&rows, 2)[0], "forward");

    let credential = |token: &str| {
        InboundCredential::new(
            CredentialKind::Oidc,
            krishiv_common::secret::SecretString::new(token),
            ExchangeClaims::with_audiences(vec!["lake".to_owned()]),
            std::time::SystemTime::now() + std::time::Duration::from_secs(60),
        )
    };
    // No RPC recorded yet (an embedded caller): no method, no token.
    schema_of(&s, "SELECT * FROM lake.db.t").unwrap();
    s.set_inbound_credential(Method::Oidc, Some(credential("first-jwt")));
    schema_of(&s, "SELECT * FROM lake.db.t").unwrap();
    s.set_inbound_credential(Method::Oidc, Some(credential("fresher-jwt")));
    schema_of(&s, "SELECT * FROM lake.db.t").unwrap();
    // A catalog of another mode gets the server identity, never the user's token.
    schema_of(&s, "SELECT * FROM plain.db.t").unwrap();
    let id = Principal::new("u", PrincipalKind::User).id();
    let seen = calls.lock().unwrap().clone();
    let lake: Vec<_> = seen.iter().filter(|c| c.0.is_some()).cloned().collect();
    assert_eq!(
        lake,
        [
            (Some(id.clone()), None, None),
            (
                Some(id.clone()),
                Some("oidc".to_owned()),
                Some("first-jwt".to_owned())
            ),
            (
                Some(id),
                Some("oidc".to_owned()),
                Some("fresher-jwt".to_owned())
            ),
        ]
    );
    assert!(
        seen.contains(&(None, None, None)),
        "the plain catalog's call is the server's: {seen:?}"
    );
    // The token never reaches the configuration or `system.catalogs`.
    let all = format!(
        "{:?} {:?}",
        run(&s, "SELECT * FROM system.catalogs").unwrap(),
        s.conf().entries()
    );
    assert!(!all.contains("fresher-jwt"), "{all}");
}

/// Review 58b1-r1 m6: a statement uses the token of the request that started it
/// (`QueryOptions::inbound_credential`), not the session's latest; nit 10: a plain act-as session
/// is named as such in the refusal, not by the service's method.
#[test]
fn a_statement_uses_its_own_requests_token() {
    use krishiv_session::QueryOptions;
    use krishiv_session::auth::{CredentialKind, ExchangeClaims, InboundCredential, Method};
    use krishiv_session::catalog_registry::{CatalogCaller, IdentityKey};
    let factory = Arc::new(RecordingFactory::default());
    let calls = Arc::clone(&factory.0);
    let mut setup = CatalogSetup::new(DeclMode::Server).factory(factory);
    for (k, v) in [
        ("krishiv.catalog.lake.type", "memory"),
        ("krishiv.catalog.lake.auth", "delegated"),
        ("krishiv.catalog.lake.delegation", "forward"),
        ("krishiv.catalog.lake.audience", "lake"),
    ] {
        setup = setup.setting(k, v);
    }
    let rt = EngineRuntime::builder().catalogs(setup).build().unwrap();
    let s = session_as(&rt, None, Role::Operator);
    let credential = |token: &str| {
        InboundCredential::new(
            CredentialKind::Oidc,
            krishiv_common::secret::SecretString::new(token),
            ExchangeClaims::with_audiences(vec!["lake".to_owned()]),
            std::time::SystemTime::now() + std::time::Duration::from_secs(60),
        )
    };
    // Another request of the session stored a wider token meanwhile.
    s.set_inbound_credential(Method::Oidc, Some(credential("wide-jwt")));
    let options =
        QueryOptions::new().inbound_credential(Method::Oidc, Some(credential("narrow-jwt")));
    block_on(async {
        s.sql_with("SELECT * FROM lake.db.t", options)
            .await?
            .schema()
            .await
    })
    .unwrap();
    schema_of(&s, "SELECT * FROM lake.db.t").unwrap();
    let tokens: Vec<_> = calls.lock().unwrap().iter().map(|c| c.2.clone()).collect();
    assert_eq!(
        tokens,
        [Some("narrow-jwt".to_owned()), Some("wide-jwt".to_owned())]
    );

    let mut end_user = Principal::new("alice", PrincipalKind::User);
    end_user.delegated_by = Some("portal".to_owned());
    let caller = CatalogCaller::new(IdentityKey::of("lake", &end_user), Some(Method::Oidc), None);
    assert!(
        caller.authenticated_with().contains("act-as"),
        "{}",
        caller.authenticated_with()
    );
}
