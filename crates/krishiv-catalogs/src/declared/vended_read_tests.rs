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

/// A short-lived credential's lifetime, and how long to wait until it is past half of it: a credential is used for
/// at most half its lifetime when that is shorter than the refresh skew (review c27440b53 F1), so a test that needs
/// a credential "about to expire" reads once while it is fresh, waits [`AGE`], and reads again.
const SHORT: Duration = Duration::from_secs(10);
const AGE: Duration = Duration::from_secs(6);

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
/// once the `loadTable` credential is past half its short life, the next read asks the catalog's own
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
            &[vended("s3://bucket/wh", "AKIAOLDKEY", SHORT)],
        ),
        Some(vended(
            "s3://bucket/wh",
            "AKIANEWKEY",
            Duration::from_secs(3600),
        )),
    );
    let s = session(&server, &s3);
    s3.deny(WRITER);
    // Fresh: read with the `loadTable` credential, no `loadCredentials`.
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(count(&server, "/v1/namespaces/db/tables/t/credentials"), 0);
    std::thread::sleep(AGE);
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
    // Design §4.1: the table's `token` is the bearer of that table's `loadCredentials`, and of nothing else.
    for r in &server.seen() {
        let table_bearer = r.header("authorization") == Some("Bearer table-token");
        assert_eq!(
            table_bearer,
            path(r) == "/v1/namespaces/db/tables/t/credentials",
            "{}",
            path(r)
        );
    }
}

/// Design §4.1: the table's `token` expired before its vended credential. The catalog answers 401 to that token on
/// `loadCredentials` and 200 to the caller's own (none here: the catalog has no auth), so the refresh falls back to
/// the caller's token once. The refused token is not kept with the new answer, so the next refresh goes straight to
/// the caller's token.
#[test]
fn vended_table_token_refused_falls_back_to_caller_token() {
    let s3 = serve_s3();
    let table = s3_table(&s3);
    let load = load_body(
        &table,
        r#"{"token": "table-token"}"#,
        &[vended("s3://bucket/wh", "AKIAOLDKEY", SHORT)],
    );
    let config = r#"{"defaults": {}, "overrides": {}, "endpoints": ["GET /v1/{prefix}/namespaces/{namespace}/tables/{table}",
        "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"]}"#;
    // The first credential answered is itself short-lived, so a read past half its life refreshes a second time.
    let mut answered = 0_usize;
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, config.into()),
        "/v1/namespaces/db/tables/t" => Reply::Json(200, load.clone()),
        "/v1/namespaces/db/tables/t/credentials" => {
            if r.header("authorization") == Some("Bearer table-token") {
                return Reply::Json(
                    401,
                    r#"{"error": {"message": "expired", "type": "NotAuthorizedException", "code": 401}}"#
                        .into(),
                );
            }
            answered += 1;
            let (key, ttl) = if answered == 1 {
                ("AKIAMIDKEY", SHORT)
            } else {
                ("AKIANEWKEY", Duration::from_secs(3600))
            };
            Reply::Json(
                200,
                format!(
                    r#"{{"storage-credentials": [{}]}}"#,
                    vended("s3://bucket/wh", key, ttl)
                ),
            )
        }
        _ => Reply::Json(
            404,
            r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#.into(),
        ),
    });
    let s = session(&server, &s3);
    s3.deny(WRITER);
    assert_eq!(ids(&s), vec![1, 2, 3]);
    // Past half the `loadTable` credential's life: the refresh sends the table token, refused, then the caller's.
    std::thread::sleep(AGE);
    s3.deny("AKIAOLDKEY");
    assert_eq!(ids(&s), vec![1, 2, 3]);
    // Past half the refreshed credential's life: the next refresh goes straight to the caller's token.
    std::thread::sleep(AGE);
    s3.deny("AKIAMIDKEY");
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert!(
        s3.key_ids().contains(&Some("AKIANEWKEY".to_owned())),
        "{:?}",
        s3.seen()
    );
    let bearers: Vec<Option<String>> = server
        .seen()
        .iter()
        .filter(|r| path(r) == "/v1/namespaces/db/tables/t/credentials")
        .map(|r| r.header("authorization").map(str::to_owned))
        .collect();
    // The table token once (refused), then the caller's token for this refresh and the next one.
    assert_eq!(
        bearers,
        vec![Some("Bearer table-token".to_owned()), None, None],
        "{bearers:?}"
    );
}

/// An S3 credential for `prefix` signed as `key_id` without an expiry key (valid for `vendedDefaultTtl`).
fn vended_no_expiry(prefix: &str, key_id: &str) -> String {
    format!(
        r#"{{"prefix": "{prefix}", "config": {{"s3.access-key-id": "{key_id}", "s3.secret-access-key": "secret-of-{key_id}"}}}}"#
    )
}

/// How a scripted catalog answers `loadCredentials`.
#[derive(Clone, Copy)]
enum Credentials {
    /// Not advertised in `/v1/config`.
    NotAdvertised,
    /// Advertised, answered 404 `NoSuchTableException`.
    Missing,
    /// Advertised, answered a 404 of another error type (`NotFoundException`).
    PlainNotFound,
}

/// A catalog whose `n`-th `loadTable` (from 0) answers `load(n)`.
fn reloading_catalog(
    load: impl Fn(usize) -> String + Send + Sync + 'static,
    credentials: Credentials,
) -> Server {
    let endpoints = match credentials {
        Credentials::NotAdvertised => "",
        Credentials::Missing | Credentials::PlainNotFound => {
            r#", "endpoints": ["GET /v1/{prefix}/namespaces/{namespace}/tables/{table}",
            "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"]"#
        }
    };
    let config = format!(r#"{{"defaults": {{}}, "overrides": {{}}{endpoints}}}"#);
    let loads = std::sync::atomic::AtomicUsize::new(0);
    let missing = r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#;
    let plain = r#"{"error": {"message": "no", "type": "NotFoundException", "code": 404}}"#;
    let plain_not_found = matches!(credentials, Credentials::PlainNotFound);
    serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, config.clone()),
        "/v1/namespaces/db/tables/t/credentials" if plain_not_found => {
            Reply::Json(404, plain.into())
        }
        "/v1/namespaces/db/tables/t" => Reply::Json(
            200,
            load(loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst)),
        ),
        _ => Reply::Json(404, missing.into()),
    })
}

fn count(server: &Server, p: &str) -> usize {
    server.seen().iter().filter(|r| path(r) == p).count()
}

/// Design §4.1: a catalog without `loadCredentials`, or one that answers it 404 (of any error type), refreshes by
/// reloading the table.
/// The first answer's credential is short-lived: read once while fresh, then past half its life the reload's is
/// used, and every storage request of that read is signed with it.
#[test]
fn vended_credential_refreshed_by_table_reload() {
    for mode in [
        Credentials::NotAdvertised,
        Credentials::Missing,
        Credentials::PlainNotFound,
    ] {
        let s3 = serve_s3();
        let table = s3_table(&s3);
        let t = table.clone();
        let server = reloading_catalog(
            move |n| {
                let (key, ttl) = if n == 0 {
                    ("AKIAOLDKEY", SHORT)
                } else {
                    ("AKIARELOADKEY", Duration::from_secs(3600))
                };
                load_body(
                    &t,
                    r#"{"token": "table-token"}"#,
                    &[vended("s3://bucket/wh", key, ttl)],
                )
            },
            mode,
        );
        let s = session(&server, &s3);
        s3.deny(WRITER);
        assert_eq!(ids(&s), vec![1, 2, 3]);
        std::thread::sleep(AGE);
        s3.deny("AKIAOLDKEY");
        s3.clear_seen();
        assert_eq!(ids(&s), vec![1, 2, 3]);
        assert_eq!(
            s3.key_ids(),
            signed_with(&["AKIARELOADKEY"]),
            "{:?}",
            s3.seen()
        );
        assert_eq!(count(&server, "/v1/namespaces/db/tables/t"), 2);
        let asked = count(&server, "/v1/namespaces/db/tables/t/credentials");
        match mode {
            Credentials::NotAdvertised => assert_eq!(asked, 0),
            Credentials::Missing | Credentials::PlainNotFound => assert_eq!(asked, 1),
        }
        // The reload is sent with the table's token, the first load with the caller's (here: none).
        let bearers: Vec<Option<String>> = server
            .seen()
            .iter()
            .filter(|r| path(r) == "/v1/namespaces/db/tables/t")
            .map(|r| r.header("authorization").map(str::to_owned))
            .collect();
        assert_eq!(bearers, vec![None, Some("Bearer table-token".to_owned())]);
    }
}

/// Design §4.1 minor 5, §4.3: a credential without an expiry key that storage refuses as expired (403
/// `ExpiredToken`) is refreshed once and the request retried, though the cache still thinks it good for the default
/// TTL; the refused answer is replaced, so the next query never signs with it. A read starts with HEADs, which storage
/// answers with a bare 403 (no `ExpiredToken` body): those are refreshed and retried once too (review thread `Lf1s`).
#[test]
fn vended_credential_refused_mid_read_is_refreshed() {
    let s3 = serve_s3();
    let table = s3_table(&s3);
    let server = catalog(
        load_body(
            &table,
            "{}",
            &[vended_no_expiry("s3://bucket/wh", "AKIAOLDKEY")],
        ),
        Some(vended(
            "s3://bucket/wh",
            "AKIANEWKEY",
            Duration::from_secs(3600),
        )),
    );
    let s = session(&server, &s3);
    s3.deny(WRITER);
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(s3.key_ids(), signed_with(&["AKIAOLDKEY"]));
    let credentials = "/v1/namespaces/db/tables/t/credentials";
    assert_eq!(count(&server, credentials), 0);
    // Storage revokes the key; the cache still holds it, valid for the default TTL.
    s3.expire("AKIAOLDKEY");
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(count(&server, credentials), 1);
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(
        s3.key_ids(),
        signed_with(&["AKIANEWKEY"]),
        "{:?}",
        s3.seen()
    );
    assert_eq!(count(&server, credentials), 1);
}

/// Design §4.1 confinement, §8 `vended_prefix_does_not_open_path_reads`: after a vended read of `lake.db.t` in
/// server mode, a path read of the same files by the same session never uses the vended key: it reads with the
/// session's own storage settings under `allowedPrefixes`, or is refused.
#[test]
fn vended_prefix_does_not_open_path_reads() {
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
    let setup = CatalogSetup::new(DeclMode::Server)
        .factory(Arc::new(IcebergRestFactory))
        .setting("krishiv.catalog.lake.type", "iceberg-rest")
        .setting("krishiv.catalog.lake.uri", server.base.as_str())
        .setting("krishiv.catalog.lake.allowInsecure", "true")
        .setting("krishiv.catalog.lake.storage", "vended");
    let runtime = EngineRuntime::builder()
        .catalogs(setup)
        .config("fs.s3a.endpoint", s3.endpoint.as_str())
        .config("fs.s3a.region", "us-east-1")
        .config("krishiv.io.objectStore.allowedPrefixes", "s3://bucket/wh")
        // The session's own key, so a path read signs with it (no ambient credential chain).
        .config("fs.s3a.access.key", "AKIASESSIONKEY")
        .config("fs.s3a.secret.key", "secret-of-the-session")
        .build()
        .unwrap();
    let s = Session::builder().runtime(runtime).build().unwrap();
    s3.deny(WRITER);
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(s3.key_ids(), signed_with(&["AKIAVENDEDKEY"]));
    let data = s3
        .objects()
        .into_iter()
        .find(|o| o.ends_with(".parquet"))
        .unwrap();
    s3.clear_seen();
    let read = block_on(async {
        s.sql(&format!("SELECT * FROM parquet.`s3://{data}`"))
            .await?
            .collect()
            .await
    });
    let seen = s3.seen();
    assert!(
        seen.iter()
            .all(|r| r.key_id.as_deref() != Some("AKIAVENDEDKEY")),
        "a path read used the vended key: {seen:?} ({read:?})"
    );
    // Either refused before any request, or read with the session's own key.
    assert!(
        seen.iter()
            .all(|r| r.key_id.as_deref() == Some("AKIASESSIONKEY")),
        "{seen:?} ({read:?})"
    );
    // The path is inside the allowed prefixes and the stand-in accepts the session's key: the read succeeds with
    // the rows of that one data file (some of the table's 3), signed by the session alone.
    let rows: usize = read
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert!((1..=3).contains(&rows), "{rows} rows");
    assert!(!seen.is_empty(), "the path read reached storage");
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

/// Design §4.5, §8 `remote_signing_refused_cleanly`, the `storage = vended` leg (the declaration leg is
/// `catalog_decl::tests::remote_signing_refused_in_this_build`, the allowed-prefixes leg
/// `read_tests::table_config_signer_and_refresh_urls_never_followed`): a vended table whose answer sets
/// `s3.remote-signing-enabled` is refused when read, with Krishiv's own error, before any storage request; `signer.*`
/// keys beside vended credentials ask for nothing (§4.5), so that read signs with the vended key and the signer is
/// never called.
#[test]
fn remote_signing_refused_cleanly() {
    let s3 = serve_s3();
    let table = s3_table(&s3);
    s3.deny(WRITER);
    let signer = serve(|_| Reply::Json(500, "{}".into()));
    let lake = |config: &str| {
        catalog(
            load_body(
                &table,
                config,
                &[vended(
                    "s3://bucket/wh",
                    "AKIAVENDEDKEY",
                    Duration::from_secs(3600),
                )],
            ),
            None,
        )
    };
    let flagged = lake(r#"{"s3.remote-signing-enabled": "true"}"#);
    let s = session(&flagged, &s3);
    s3.clear_seen();
    let e =
        block_on(async { s.sql("SELECT id FROM lake.db.t").await?.collect().await }).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.to_string().contains("remote signing"), "{e}");
    assert!(s3.seen().is_empty(), "{:?}", s3.seen());

    let with_signer = lake(&format!(
        r#"{{"signer.uri": "{0}/signer", "signer.endpoint": "v1/sign"}}"#,
        signer.base
    ));
    let s = session(&with_signer, &s3);
    s3.clear_seen();
    assert_eq!(ids(&s), vec![1, 2, 3]);
    assert_eq!(s3.key_ids(), signed_with(&["AKIAVENDEDKEY"]));
    assert!(signer.seen().is_empty(), "{:?}", signer.seen());
}

/// Review c27440b53 F1: a catalog that only ever vends 60 s credentials (less than the 5-minute refresh skew) is asked
/// for them about once per scope, not once per object-store request: the skew is at most half the credential's
/// lifetime, in the store's cache and in `VendedSource`.
#[test]
fn short_lived_vended_credentials_are_not_fetched_per_request() {
    let s3 = serve_s3();
    let table = s3_table(&s3);
    let short = vended("s3://bucket/wh", "AKIASHORTKEY", Duration::from_secs(60));
    let server = catalog(
        load_body(&table, "{}", std::slice::from_ref(&short)),
        Some(short),
    );
    let s = session(&server, &s3);
    s3.deny(WRITER);
    s3.clear_seen();
    for _ in 0..3 {
        assert_eq!(ids(&s), vec![1, 2, 3]);
    }
    let requests = s3.seen().len();
    let asked = count(&server, "/v1/namespaces/db/tables/t/credentials");
    assert!(requests > 3, "the reads reached storage: {requests}");
    // At most one per query per scope (one scope here); before the fix it was about one per S3 request.
    assert!(
        asked <= 3,
        "{asked} credential calls for {requests} S3 requests"
    );
}
