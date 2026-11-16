// SPDX-License-Identifier: Apache-2.0
//! Declared catalogs on a runtime (design P3-T067 §2.2-§2.4, §7; PR-58a): the registry and its
//! factories, startup refusals, per-tenant visibility, `system.catalogs` and the server-mode
//! refusal of runtime catalog keys. The factory here is a fake (the `iceberg-rest` one lives in
//! `krishiv-catalogs`, feature `session`).
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
use krishiv_session::catalog_registry::{CatalogDeps, CatalogFactory, CatalogSetup};
use krishiv_session::{EngineRuntime, Principal, PrincipalKind, Session};
use krishiv_types::DataType;

/// A catalog holding `db.t (x INT)`.
#[derive(Debug)]
struct OneTable(String);

impl Catalog for OneTable {
    fn name(&self) -> &str {
        &self.0
    }

    fn epoch(&self) -> u64 {
        0
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _version: Option<TableVersion>,
        _form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            Ok(
                (id.namespace == ["db"] && id.name == "t").then(|| TableMeta {
                    ident: id.clone(),
                    columns: vec![TableColumn::new("x", DataType::Integer, true)],
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
}

impl CatalogFactory for Fake {
    fn kind(&self) -> CatalogType {
        CatalogType::Memory
    }

    fn build(
        &self,
        decl: &CatalogDecl,
        _deps: &CatalogDeps,
    ) -> Result<Arc<dyn Catalog>, KrishivError> {
        let n = self.builds.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.delay);
        if n < self.fail_first {
            return Err(krishiv_common::error::KrishivCatalogFailure {
                catalog: decl.name.clone(),
                relation_name: "the catalog".to_owned(),
            }
            .into());
        }
        Ok(Arc::new(OneTable(decl.name.clone())))
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
    assert_eq!(
        s.catalog().list_catalogs(),
        ["lake", "spark_catalog", "system"]
    );
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
        assert!(block_on(s.catalog().set_current_catalog("lake")).is_err());
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
