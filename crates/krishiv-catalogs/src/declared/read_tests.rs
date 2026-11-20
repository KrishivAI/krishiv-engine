// SPDX-License-Identifier: Apache-2.0
//! Declared Iceberg REST tables read through a session (design P3-T067 PR-58b): a fake REST
//! catalog serves the `loadTable` of a small Iceberg table the connector's test helper wrote into a
//! local directory, and the session reads it through `krishiv-connectors-iceberg`.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_connectors_iceberg::testing::{WrittenTable, write_table};
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::TableIdent;
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::{CallerCatalog, CatalogCaller, CatalogSetup};
use krishiv_session::{EngineRuntime, Session};
use krishiv_store::egress::PrivateAddresses;

use krishiv_session::catalog_registry::CatalogFactory;

use super::IcebergRestFactory;
use super::tests::{FORM, decl, deps, ident, path};
use crate::iceberg_rest::test_server::{Reply, Server, serve};

const NO_TABLE: &str =
    r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#;

/// The `loadTable` body of `table` with `config` (a JSON object) and `extra` members.
fn body(table: &WrittenTable, config: &str, extra: &str) -> String {
    format!(
        r#"{{"metadata-location": {}, "metadata": {}, "config": {config}{extra}}}"#,
        serde_json::Value::String(table.metadata_location.clone()),
        table.metadata
    )
}

/// A catalog with `db.t` answering `load` (and `db` in its listings).
fn catalog(load: String) -> Server {
    serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()),
        "/v1/namespaces" => Reply::Json(200, r#"{"namespaces": [["db"]]}"#.into()),
        "/v1/namespaces/db/tables" => Reply::Json(
            200,
            r#"{"identifiers": [{"namespace": ["db"], "name": "t"}]}"#.into(),
        ),
        "/v1/namespaces/db/tables/t" => Reply::Json(200, load.clone()),
        "/v1/namespaces/nope/tables" => Reply::Json(
            404,
            r#"{"error": {"message": "no", "type": "NoSuchNamespaceException", "code": 404}}"#
                .into(),
        ),
        _ => Reply::Json(404, NO_TABLE.into()),
    })
}

fn loads(server: &Server) -> usize {
    server
        .seen()
        .iter()
        .filter(|s| path(s) == "/v1/namespaces/db/tables/t")
        .count()
}

fn session_with(mode: DeclMode, pairs: &[(&str, &str)], config: &[(&str, &str)]) -> Session {
    let mut setup = CatalogSetup::new(mode).factory(Arc::new(IcebergRestFactory));
    for (k, v) in pairs {
        setup = setup.setting(*k, *v);
    }
    let mut builder = EngineRuntime::builder().catalogs(setup);
    for (k, v) in config {
        builder = builder.config(*k, *v);
    }
    Session::builder()
        .runtime(builder.build().unwrap())
        .build()
        .unwrap()
}

/// An embedded session with `lake` on `server` (and `storage`, when given).
pub(super) fn session(server: &Server, storage: Option<&str>) -> Session {
    let mut pairs = vec![
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", server.base.as_str()),
    ];
    pairs.push((
        "krishiv.catalog.lake.storage",
        storage.unwrap_or("allowed-prefixes"),
    ));
    session_with(DeclMode::Embedded, &pairs, &[])
}

fn run(s: &Session, sql: &str) -> Result<Vec<arrow_array::RecordBatch>, KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await })
}

/// `(id, name)` rows of `sql`, sorted.
pub(super) fn rows(s: &Session, sql: &str) -> Vec<(i64, String)> {
    let mut out = Vec::new();
    for b in run(s, sql).unwrap() {
        let ids = b.column(0).as_primitive::<Int64Type>();
        let names = b.column(1);
        for i in 0..b.num_rows() {
            let name = names
                .as_string_view_opt()
                .map(|v| v.value(i).to_owned())
                .or_else(|| names.as_string_opt::<i32>().map(|v| v.value(i).to_owned()))
                .unwrap_or_default();
            out.push((ids.value(i), name));
        }
    }
    out.sort();
    out
}

pub(super) fn pairs(v: &[(i64, &str)]) -> Vec<(i64, String)> {
    v.iter().map(|(i, s)| (*i, (*s).to_owned())).collect()
}

#[test]
fn declared_table_reads_rows() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let server = catalog(body(&table, "{}", ""));
    let s = session(&server, None);
    assert_eq!(
        rows(&s, "SELECT id, name FROM lake.db.t"),
        pairs(&[(1, "a"), (2, "b"), (3, "c")])
    );
    assert_eq!(
        rows(&s, "SELECT id, name FROM lake.db.t WHERE id > 1"),
        pairs(&[(2, "b"), (3, "c")])
    );
    // A projection reads one column.
    let ids = run(&s, "SELECT id FROM lake.db.t WHERE name = 'b'").unwrap();
    assert_eq!(ids[0].column(0).as_primitive::<Int64Type>().value(0), 2);
    // The Iceberg `doc` is the column comment.
    // Time travel by snapshot id (the analyzer has no `VERSION AS OF` lowering yet, so the catalog
    // is asked directly): the first snapshot's version and schema.
    let built = IcebergRestFactory
        .build(
            &decl(
                &[
                    ("krishiv.catalog.lake.type", "iceberg-rest"),
                    ("krishiv.catalog.lake.uri", server.base.as_str()),
                ],
                DeclMode::Embedded,
            ),
            &deps(DeclMode::Embedded, PrivateAddresses::Allow),
        )
        .unwrap();
    let first = TableVersion(table.snapshots[0]);
    let at = block_on(built.load_declared(
        &CatalogCaller::server("lake"),
        &ident(&["db"], "t"),
        Some(first),
        FORM,
    ))
    .unwrap()
    .unwrap();
    assert_eq!(at.meta.version, Some(first));
    assert_eq!(
        at.meta.column_comments.get("name").map(String::as_str),
        Some("the name")
    );
    assert!(at.table.is_some());
    let current = block_on(built.load_table(
        &CatalogCaller::server("lake"),
        &ident(&["db"], "t"),
        None,
        FORM,
    ))
    .unwrap()
    .unwrap();
    assert_eq!(current.version, Some(TableVersion(table.snapshots[1])));
}

#[test]
fn declared_table_read_restrictions_refused_before_data() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    // The data would read fine: only the restriction refuses it.
    let restricted = body(
        &table,
        "{}",
        r#", "read-restrictions": {"required-column-projections": [{"field-id": 2, "type": "mask-alphanum"}]}"#,
    );
    let server = catalog(restricted);
    let built = IcebergRestFactory
        .build(
            &decl(
                &[
                    ("krishiv.catalog.lake.type", "iceberg-rest"),
                    ("krishiv.catalog.lake.uri", server.base.as_str()),
                ],
                DeclMode::Embedded,
            ),
            &deps(DeclMode::Embedded, PrivateAddresses::Allow),
        )
        .unwrap();
    let e = block_on(built.load_declared(
        &CatalogCaller::server("lake"),
        &ident(&["db"], "t"),
        None,
        FORM,
    ))
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogTableUnsupported);
    assert!(e.to_string().contains("read restrictions"), "{e}");
    // Through SQL the query fails and no data file is opened: the data directory's files are
    // moved away, so a read would fail with a missing file instead.
    std::fs::rename(dir.path().join("db/t/data"), dir.path().join("moved")).unwrap();
    let s = session(&server, None);
    // Review 58b2 F2: the refusal itself, not a generic catalog failure, for SQL and `spark.table`.
    let e = run(&s, "SELECT id FROM lake.db.t").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogTableUnsupported,
        "{e}"
    );
    assert!(e.to_string().contains("read restrictions"), "{e}");
    let e = block_on(s.table("lake.db.t").collect()).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogTableUnsupported,
        "{e}"
    );
    assert!(e.to_string().contains("read restrictions"), "{e}");
}

#[test]
fn declared_vended_table_refused_in_this_build() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let server = catalog(body(&table, "{}", ""));
    let s = session(&server, Some("vended"));
    // Metadata is shown; a read is refused at planning, before any storage request.
    block_on(
        block_on(s.sql("SELECT id FROM lake.db.t"))
            .unwrap()
            .schema(),
    )
    .unwrap();
    std::fs::rename(dir.path().join("db/t"), dir.path().join("moved")).unwrap();
    let e = run(&s, "SELECT id FROM lake.db.t").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.to_string().contains("vended credentials"), "{e}");
    // The catalog was asked for vended credentials.
    let seen = server.seen();
    let load = seen
        .iter()
        .find(|r| path(r) == "/v1/namespaces/db/tables/t")
        .unwrap();
    assert_eq!(
        load.header("x-iceberg-access-delegation"),
        Some("vended-credentials")
    );
}

#[test]
fn declared_table_outside_allowed_prefixes_refused() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    // A server-mode catalog names `s3://b/t`; its manifest list is still the local file, outside
    // every allowed root and prefix, so the read is refused before it is opened.
    let mut metadata = table.metadata.clone();
    metadata["location"] = serde_json::Value::String("s3://b/t".to_owned());
    let mut moved: WrittenTable = table;
    moved.metadata_location = "s3://b/t/metadata/v1.metadata.json".to_owned();
    moved.metadata = metadata;
    let server = catalog(body(&moved, "{}", ""));
    let s = session_with(
        DeclMode::Server,
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            ("krishiv.catalog.lake.uri", server.base.as_str()),
            ("krishiv.catalog.lake.allowInsecure", "true"),
            ("krishiv.catalog.lake.storage", "allowed-prefixes"),
        ],
        &[("krishiv.io.objectStore.allowedPrefixes", "s3://other/x")],
    );
    let e = run(&s, "SELECT id FROM lake.db.t").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogLocationRefused,
        "{e}"
    );
    assert!(
        !e.to_string().contains(&dir.path().display().to_string()),
        "{e}"
    );
    assert!(e.to_string().contains("`lake`.`db`.`t`"), "{e}");
}

#[test]
fn declared_listing_tables_and_namespaces() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let server = catalog(body(&table, "{}", ""));
    let built = IcebergRestFactory
        .build(
            &decl(
                &[
                    (
                        "spark.sql.catalog.lake",
                        "org.apache.iceberg.spark.SparkCatalog",
                    ),
                    ("spark.sql.catalog.lake.type", "rest"),
                    ("spark.sql.catalog.lake.uri", server.base.as_str()),
                    ("spark.sql.catalog.lake.token", "static-token"),
                ],
                DeclMode::Embedded,
            ),
            &deps(DeclMode::Embedded, PrivateAddresses::Allow),
        )
        .unwrap();
    let caller = CatalogCaller::server("lake");
    let ns = block_on(built.list_namespaces(&caller, &[])).unwrap();
    assert_eq!(ns, Some(vec![vec!["db".to_owned()]]));
    let tables = block_on(built.list_tables(&caller, &["db".to_owned()])).unwrap();
    assert_eq!(tables, Some(vec!["t".to_owned()]));
    let missing = block_on(built.list_tables(&caller, &["nope".to_owned()])).unwrap();
    assert_eq!(missing, None);
    for r in server.seen() {
        assert_eq!(
            r.header("authorization"),
            Some("Bearer static-token"),
            "{}",
            r.target
        );
    }
}

#[test]
fn declared_metadata_cache_ttl_and_refresh() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let server = catalog(body(&table, "{}", ""));
    // The catalog level, with a short TTL.
    let d = decl(
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            ("krishiv.catalog.lake.uri", server.base.as_str()),
        ],
        DeclMode::Embedded,
    );
    let config =
        IcebergRestFactory::config(&d, &deps(DeclMode::Embedded, PrivateAddresses::Allow), true)
            .unwrap();
    let client = crate::iceberg_rest::RestClient::new(
        "lake",
        config,
        Arc::new(super::table::EmbeddedLocations::default()),
    )
    .unwrap();
    let ttl = Duration::from_millis(300);
    let built = super::RestCatalog::new(&d, Arc::new(client), ttl, false);
    let caller = CatalogCaller::server("lake");
    let t: TableIdent = ident(&["db"], "t");
    let load = || {
        block_on(built.load_table(&caller, &t, None, FORM))
            .unwrap()
            .unwrap()
    };
    load();
    load();
    assert_eq!(loads(&server), 1, "the second load is cached");
    std::thread::sleep(ttl + Duration::from_millis(50));
    load();
    assert_eq!(loads(&server), 2, "past the TTL the table is loaded again");
    let epoch = built.epoch();
    assert!(block_on(built.refresh(&caller, &t)).unwrap());
    assert_eq!(built.epoch(), epoch + 1);
    assert_eq!(loads(&server), 3, "a refresh drops the entry");
    assert!(!block_on(built.refresh(&caller, &ident(&["db"], "missing"))).unwrap());

    // Through SQL: queries share the cached answer; `REFRESH TABLE` loads it again.
    let server = catalog(body(&table, "{}", ""));
    let s = session(&server, None);
    let df = block_on(s.sql("SELECT id, name FROM lake.db.t")).unwrap();
    let n: usize = block_on(df.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(n, 3);
    run(&s, "SELECT id FROM lake.db.t").unwrap();
    assert_eq!(loads(&server), 1);
    run(&s, "REFRESH TABLE lake.db.t").unwrap();
    assert_eq!(loads(&server), 2);
    // The frame analyzed before the refresh is analyzed again (the catalog's epoch moved).
    block_on(df.collect()).unwrap();
    assert_eq!(
        loads(&server),
        2,
        "served from the entry the refresh loaded"
    );
    // Review 58b2 nit: names reach the catalog as written; another letter case is another name.
    let e = run(&s, "REFRESH TABLE lake.DB.T").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    let e = run(&s, "REFRESH TABLE lake.db.missing").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
}

#[test]
fn table_config_signer_and_refresh_urls_never_followed() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    // Every URL the table config names is this catalog's own host, so a request to one would be
    // seen; a table token would replace the catalog's.
    let probe = serve(|_| Reply::Json(500, "{}".into()));
    let config = format!(
        r#"{{"token": "table-token",
            "urn:ietf:params:oauth:token-type:access_token": "table-token",
            "gcs.oauth2.refresh-credentials-endpoint": "{0}/refresh",
            "gcs.oauth2.refresh-credentials-enabled": "true",
            "adls.refresh-credentials-endpoint": "{0}/refresh",
            "signer.uri": "{0}/signer", "signer.endpoint": "v1/sign",
            "s3.access-key-id": "AKIA", "s3.secret-access-key": "secret"}}"#,
        probe.base
    );
    let server = catalog(body(&table, &config, ""));
    let s = session_with(
        DeclMode::Embedded,
        &[
            (
                "spark.sql.catalog.lake",
                "org.apache.iceberg.spark.SparkCatalog",
            ),
            ("spark.sql.catalog.lake.type", "rest"),
            ("spark.sql.catalog.lake.uri", server.base.as_str()),
            ("spark.sql.catalog.lake.token", "catalog-token"),
            ("krishiv.catalog.lake.storage", "allowed-prefixes"),
        ],
        &[],
    );
    assert_eq!(rows(&s, "SELECT id, name FROM lake.db.t").len(), 3);
    run(&s, "REFRESH TABLE lake.db.t").unwrap();
    assert_eq!(rows(&s, "SELECT id, name FROM lake.db.t").len(), 3);
    assert!(probe.seen().is_empty(), "{:?}", probe.seen());
    for r in server.seen() {
        assert_eq!(r.header("authorization"), Some("Bearer catalog-token"));
    }

    // A table that asks for remote signing is refused when read.
    for config in [
        r#"{"s3.remote-signing-enabled": "true"}"#.to_owned(),
        format!(r#"{{"signer.uri": "{}/signer"}}"#, probe.base),
    ] {
        let server = catalog(body(&table, &config, ""));
        let s = session(&server, None);
        let e = run(&s, "SELECT id FROM lake.db.t").unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        );
        assert!(e.to_string().contains("remote signing"), "{e}");
    }
    assert!(probe.seen().is_empty());
}

/// Review 58b2 F4 (§4.1): `lake.db.t.files` and the other Iceberg metadata tables of a declared
/// table are refused, never loaded as the table `files` of namespace `db.t`; a metadata table name
/// under a prefix that is no table is an ordinary (here absent) name.
#[test]
fn declared_metadata_tables_refused() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let server = catalog(body(&table, "{}", ""));
    let s = session(&server, None);
    for name in ["files", "SNAPSHOTS", "all_manifests", "position_deletes"] {
        let e = run(&s, &format!("SELECT * FROM lake.db.t.{name}")).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivCatalogTableUnsupported,
            "{e}"
        );
        assert!(e.to_string().contains("Iceberg metadata table"), "{e}");
        assert!(
            e.to_string().contains("catalog `lake` serves `db`.`t`"),
            "{e}"
        );
    }
    let e = run(&s, "SELECT * FROM lake.db.nope.files").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound, "{e}");
    // A refused name never reached the catalog as a table of namespace `db.t`.
    let asked: Vec<String> = server
        .seen()
        .iter()
        .map(|r| path(r).to_owned())
        .filter(|p| p.contains("/tables/"))
        .collect();
    assert!(
        !asked
            .iter()
            .any(|p| p.starts_with("/v1/namespaces/db%1Ft/")),
        "{asked:?}"
    );
    assert!(
        asked
            .iter()
            .any(|p| p == "/v1/namespaces/db%1Fnope/tables/files"),
        "{asked:?}"
    );
}

/// Review 58b2 F5 (§3.3): no streaming read of a declared table exists yet. A streaming SQL source
/// over one is refused before the catalog is asked (never an internal error, never a batch read):
/// when the first streaming catalog read lands, this test fails and the delegated-catalog rules
/// must be applied to it. The Connect `readStream.table` leg is in krishiv-connect-server
/// `tests/delegated_catalog.rs`.
#[test]
fn declared_streaming_on_delegated_refused() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let server = catalog(body(&table, "{}", ""));
    let s = session(&server, None);
    for (sql, class) in [
        (
            "SELECT * FROM STREAM(lake.db.t)",
            "UNRESOLVABLE_TABLE_VALUED_FUNCTION",
        ),
        (
            "CREATE STREAMING TABLE st AS SELECT * FROM STREAM(lake.db.t)",
            "UNSUPPORTED_FEATURE.CREATE_PIPELINE_DATASET_QUERY_EXECUTION",
        ),
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(e.error_class().name(), class, "{sql}: {e}");
    }
    assert_eq!(loads(&server), 0, "no streaming source loaded the table");
}

/// The embedded rules take only absolute, unescaped `file:` paths on top of the remote ones, and
/// let the `file` scheme through while the remote list still decides every other scheme.
#[test]
fn embedded_locations_take_plain_absolute_file_paths_only() {
    use crate::location::{LocationPolicy, LocationRule};
    let policy = super::table::EmbeddedLocations::default();
    assert_eq!(policy.check("file:///data/t/metadata/v1.json"), Ok(()));
    assert_eq!(policy.check("FILE:/data/t"), Ok(()));
    assert_eq!(policy.check("s3://bucket/t"), Ok(()));
    assert_eq!(
        policy.check("file:///data\\t"),
        Err(LocationRule::Malformed)
    );
    assert_eq!(
        policy.check("file:///data/\u{7}t"),
        Err(LocationRule::Malformed)
    );
    assert_eq!(policy.check("file:data/t"), Err(LocationRule::NoScheme));
    assert_eq!(policy.check("file://data/t"), Err(LocationRule::NoScheme));
    for escape in [
        "file:///data/../etc",
        "file:///data/%2E%2E/etc",
        "file:///a%2fb",
        "file:///a%5Cb",
    ] {
        assert_eq!(
            policy.check(escape),
            Err(LocationRule::PathEscape),
            "{escape}"
        );
    }
    assert_eq!(policy.check_scheme("file"), Ok(()));
    assert_eq!(policy.check_scheme("FILE"), Ok(()));
    assert_eq!(policy.check_scheme("s3"), Ok(()));
    assert_eq!(
        policy.check_scheme("ftp"),
        Err(LocationRule::SchemeNotAllowed)
    );
}
