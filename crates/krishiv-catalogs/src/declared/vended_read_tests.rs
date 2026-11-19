// SPDX-License-Identifier: Apache-2.0
//! Vended reads end to end (design P3-T067 §4.1, §8): a table written to `s3://bucket/wh` on a loopback S3 stand-in
//! with a writer key, served by a fake REST catalog that vends another key, read by a session that has no S3
//! credential of its own. The stand-in records the access key id every request was signed with.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_connectors_iceberg::testing::WrittenTable;
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::CatalogSetup;
use krishiv_session::{EngineRuntime, Session};

use super::IcebergRestFactory;
use super::tests::path;
use crate::iceberg_rest::test_server::{Reply, Server, serve};
use crate::s3_test_server::{S3Server, serve_s3, write_s3_table};

const WRITER: &str = "AKIAWRITERKEY";

/// `db.t` written to `s3://bucket/wh` on `s3`, every request signed with [`WRITER`].
fn s3_table(s3: &S3Server) -> WrittenTable {
    let table = write_s3_table(s3, "s3://bucket/wh", WRITER);
    assert!(table.metadata_location.starts_with("s3://bucket/wh/db/t/"));
    assert!(
        s3.objects().iter().any(|o| o.ends_with(".parquet")),
        "{:?}",
        s3.objects()
    );
    table
}

fn ms_from_now(d: Duration) -> u128 {
    (SystemTime::now() + d)
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

/// An S3 credential for `prefix` signed as `key_id`, expiring `ttl` from now.
fn vended(prefix: &str, key_id: &str, ttl: Duration) -> String {
    format!(
        r#"{{"prefix": "{prefix}", "config": {{"s3.access-key-id": "{key_id}", "s3.secret-access-key": "secret-of-{key_id}",
            "s3.session-token": "session-of-{key_id}", "s3.session-token-expires-at-ms": "{}"}}}}"#,
        ms_from_now(ttl)
    )
}

/// The `loadTable` answer for `table` with `config` and `storage-credentials`.
fn load_body(table: &WrittenTable, config: &str, credentials: &[String]) -> String {
    format!(
        r#"{{"metadata-location": {}, "metadata": {}, "config": {config}, "storage-credentials": [{}]}}"#,
        serde_json::Value::String(table.metadata_location.clone()),
        table.metadata,
        credentials.join(",")
    )
}

/// A catalog serving `db.t` with `load`; with `credentials`, it advertises `loadCredentials` and answers it.
fn catalog(load: String, credentials: Option<String>) -> Server {
    let endpoints = if credentials.is_some() {
        r#", "endpoints": ["GET /v1/{prefix}/namespaces", "GET /v1/{prefix}/namespaces/{namespace}/tables",
            "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}",
            "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"]"#
    } else {
        ""
    };
    let config = format!(r#"{{"defaults": {{}}, "overrides": {{}}{endpoints}}}"#);
    serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, config.clone()),
        "/v1/namespaces/db/tables/t" => Reply::Json(200, load.clone()),
        "/v1/namespaces/db/tables/t/credentials" => match &credentials {
            Some(c) => Reply::Json(200, format!(r#"{{"storage-credentials": [{c}]}}"#)),
            None => Reply::Json(404, "{}".into()),
        },
        _ => Reply::Json(
            404,
            r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#.into(),
        ),
    })
}

/// An embedded session with the vended catalog `lake` on `server`, `fs.s3a.endpoint` pointing at `s3` and no S3
/// key of its own.
fn session(server: &Server, s3: &S3Server) -> Session {
    let setup = CatalogSetup::new(DeclMode::Embedded)
        .factory(Arc::new(IcebergRestFactory))
        .setting("krishiv.catalog.lake.type", "iceberg-rest")
        .setting("krishiv.catalog.lake.uri", server.base.as_str())
        .setting("krishiv.catalog.lake.storage", "vended");
    let runtime = EngineRuntime::builder()
        .catalogs(setup)
        .config("fs.s3a.endpoint", s3.endpoint.as_str())
        .config("fs.s3a.region", "us-east-1")
        .build()
        .unwrap();
    Session::builder().runtime(runtime).build().unwrap()
}

/// The sorted ids of `SELECT id FROM lake.db.t`.
fn ids(s: &Session) -> Vec<i64> {
    let batches =
        block_on(async { s.sql("SELECT id FROM lake.db.t").await?.collect().await }).unwrap();
    let mut out: Vec<i64> = batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect();
    out.sort_unstable();
    out
}

fn signed_with(key_ids: &[&str]) -> BTreeSet<Option<String>> {
    key_ids.iter().map(|k| Some((*k).to_owned())).collect()
}

/// Design §8 `vended_credentials_only`: every storage request of the read is signed with the vended key, none with
/// the writer's or unsigned (no ambient credential).
#[test]
fn vended_credentials_only() {
    let s3 = serve_s3();
    let table = s3_table(&s3);
    let server = catalog(
        load_body(
            &table,
            "{}",
            &[vended(
                "s3://bucket/wh",
                "AKIAVENDEDKEY",
                Duration::from_secs(3600),
            )],
        ),
        None,
    );
    let s = session(&server, &s3);
    s3.deny(WRITER);
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert!(!s3.seen().is_empty());
    assert_eq!(
        s3.key_ids(),
        signed_with(&["AKIAVENDEDKEY"]),
        "{:?}",
        s3.seen()
    );
    // A second query reads with the cached credential: no new `loadTable` past the metadata cache.
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(s3.key_ids(), signed_with(&["AKIAVENDEDKEY"]));
}

/// Design §8 `vended_credential_refreshed_via_credentials_endpoint` and `catalog_refresh_and_signer_urls_never_followed`:
/// the `loadTable` credential expires inside the refresh skew, so the read asks the catalog's own
/// `loadCredentials` for a new one and signs every request with that; the refresh endpoint the table config names
/// is never called.
#[test]
fn vended_credential_refreshed_via_credentials_endpoint() {
    let s3 = serve_s3();
    let table = s3_table(&s3);
    let probe = serve(|_| Reply::Json(500, "{}".into()));
    let config = format!(
        r#"{{"s3.refresh-credentials-endpoint": "{0}/refresh", "s3.refresh-credentials-enabled": "true",
            "client.refresh-credentials-endpoint": "{0}/refresh", "token": "table-token"}}"#,
        probe.base
    );
    let server = catalog(
        load_body(
            &table,
            &config,
            &[vended(
                "s3://bucket/wh",
                "AKIAOLDKEY",
                Duration::from_secs(60),
            )],
        ),
        Some(vended(
            "s3://bucket/wh",
            "AKIANEWKEY",
            Duration::from_secs(3600),
        )),
    );
    let s = session(&server, &s3);
    s3.deny(WRITER);
    s3.deny("AKIAOLDKEY");
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(
        s3.key_ids(),
        signed_with(&["AKIANEWKEY"]),
        "{:?}",
        s3.seen()
    );
    let asked: Vec<String> = server.seen().iter().map(|r| path(r).to_owned()).collect();
    assert!(
        asked
            .iter()
            .any(|p| p == "/v1/namespaces/db/tables/t/credentials"),
        "{asked:?}"
    );
    assert!(probe.seen().is_empty(), "{:?}", probe.seen());
    // The table's `token` replaces the catalog's on no request.
    for r in server.seen() {
        assert_ne!(r.header("authorization"), Some("Bearer table-token"));
    }
}

/// All text of the batches `sql` returns (every column displayed).
fn text_of(s: &Session, sql: &str) -> String {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = String::new();
    for b in &batches {
        for c in b.columns() {
            let f = arrow_cast_free_display(c);
            out.push_str(&f);
        }
    }
    out
}

/// A column's values as text without `arrow-cast`: strings as they are, everything else through `Debug`.
fn arrow_cast_free_display(c: &arrow_array::ArrayRef) -> String {
    if let Some(v) = c.as_string_opt::<i32>() {
        return v.iter().flatten().collect::<Vec<_>>().join("\n");
    }
    if let Some(v) = c.as_string_view_opt() {
        return v.iter().flatten().collect::<Vec<_>>().join("\n");
    }
    format!("{c:?}")
}

/// Design §3.2 / §8 (T-C4), the vended part of the planted-secret check: a vended secret key and session token and
/// a catalog-supplied table `token` never appear in `EXPLAIN`, an error, `system.catalogs`, `system.queries` or the
/// configuration listing.
#[test]
fn vended_secrets_never_shown() {
    const SECRET: &str = "vended-secret-canary-7c1d";
    const SESSION: &str = "vended-session-canary-7c1d";
    const TABLE_TOKEN: &str = "table-token-canary-7c1d";
    let s3 = serve_s3();
    let table = s3_table(&s3);
    let credential = format!(
        r#"{{"prefix": "s3://bucket/wh", "config": {{"s3.access-key-id": "AKIACANARYKEY", "s3.secret-access-key": "{SECRET}",
            "s3.session-token": "{SESSION}", "s3.session-token-expires-at-ms": "{}"}}}}"#,
        ms_from_now(Duration::from_secs(3600))
    );
    let server = catalog(
        load_body(
            &table,
            &format!(r#"{{"token": "{TABLE_TOKEN}"}}"#),
            &[credential],
        ),
        None,
    );
    let s = session(&server, &s3);
    assert_eq!(ids(&s), vec![1, 2, 3]);
    let mut shown = String::new();
    for sql in [
        "EXPLAIN EXTENDED SELECT id FROM lake.db.t WHERE id > 1",
        "EXPLAIN COST SELECT id FROM lake.db.t",
        "SELECT * FROM system.catalogs",
        "SELECT * FROM system.queries",
        "SET",
        "SET -v",
    ] {
        shown.push_str(&text_of(&s, sql));
    }
    // A refusal: another runtime declares `lake` against a catalog that vends only `s3://bucket/other`. Its read is
    // refused, never served the first runtime's cached credential (vended answers are keyed by catalog instance).
    let moved = catalog(
        load_body(
            &table,
            &format!(r#"{{"token": "{TABLE_TOKEN}"}}"#),
            &[vended(
                "s3://bucket/other",
                "AKIACANARYKEY",
                Duration::from_secs(3600),
            )
            .replace("secret-of-AKIACANARYKEY", SECRET)],
        ),
        None,
    );
    let e = block_on(async {
        session(&moved, &s3)
            .sql("SELECT id FROM lake.db.t")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivCatalogLocationRefused,
        "{e}"
    );
    assert!(
        e.to_string()
            .contains("outside the vended credential prefixes"),
        "{e}"
    );
    shown.push_str(&format!("{e} {e:?}"));
    assert!(shown.contains("lake"), "the listings ran");
    for planted in [SECRET, SESSION, TABLE_TOKEN] {
        assert!(!shown.contains(planted), "{planted} shown");
    }
    for r in server.seen().iter().chain(moved.seen().iter()) {
        assert!(!format!("{r:?}").contains(TABLE_TOKEN), "{r:?}");
    }
}
