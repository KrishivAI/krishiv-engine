// SPDX-License-Identifier: Apache-2.0
//! The console of one embedded session: one test per rule of the embedded mode (ADR 0024).
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_server::embedded::EmbeddedSource;
use krishiv_server::internal::{Row, SystemTable, UiSource};
use krishiv_server::session_source::EmbeddedUi;
use krishiv_session::{EngineRuntime, Session};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A cross join no machine finishes soon.
const ENDLESS: &str =
    "SELECT count(*) FROM range(0, 1000000000) a CROSS JOIN range(0, 1000000000) b";

/// Two sessions of one runtime: the one the console shows, and another one.
fn two_sessions() -> (Session, Session) {
    let rt = EngineRuntime::builder().build().unwrap();
    let mine = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let other = Session::builder().runtime(rt).build().unwrap();
    (mine, other)
}

async fn run(session: &Session, sql: &str) {
    session.sql(sql).await.unwrap().collect().await.unwrap();
}

fn texts(rows: &[Row]) -> Vec<String> {
    rows.iter()
        .filter_map(|r| r["query_text"].as_str())
        .map(str::to_owned)
        .collect()
}

/// Whether the query `row` is a statement of the SQL editor: it carries the tag of an editor run.
fn of_editor(row: &Row) -> bool {
    row["tags"]
        .as_object()
        .is_some_and(|tags| tags.keys().any(|k| k.starts_with("krishiv-ui-run-")))
}

/// The editor run `id` once it has left `running`.
async fn ended(source: &EmbeddedSource, id: &str) -> Value {
    let start = Instant::now();
    loop {
        let v = source.sql_status(id, None).await.unwrap().unwrap();
        if v["state"] != "running" {
            return v;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "still running: {v}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The id of the running query of `session` whose text holds `marker`, once it shows.
async fn running_id(session: &Session, marker: &str) -> String {
    let reader = Session::builder()
        .runtime(Arc::clone(session.runtime()))
        .build()
        .unwrap();
    let sql = format!(
        "SELECT query_id FROM system.queries WHERE session_id = '{}' AND state = 'running' \
         AND query_text LIKE '%{marker}%'",
        session.id()
    );
    let start = Instant::now();
    loop {
        let batches = reader.sql(&sql).await.unwrap().collect().await.unwrap();
        let rows = krishiv_server::session_source::batches_to_rows(&batches);
        if let Some(id) = rows.first().and_then(|r| r["query_id"].as_str()) {
            return id.to_owned();
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the query never ran"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn it_names_its_mode_and_its_session() {
    let (mine, _) = two_sessions();
    let source = EmbeddedSource::new(&mine).unwrap();
    assert_eq!(source.mode(), "embedded");
    assert_eq!(source.session_id().as_deref(), Some(mine.id()));
    assert!(format!("{source:?}").contains(mine.id()));
}

#[tokio::test]
async fn only_the_sessions_own_queries_and_operators_are_listed() {
    let (mine, other) = two_sessions();
    run(
        &mine,
        "SELECT id % 3 AS mine_marker, sum(id) FROM range(1000) GROUP BY id % 3",
    )
    .await;
    run(
        &other,
        "SELECT id % 3 AS other_marker, sum(id) FROM range(1000) GROUP BY id % 3",
    )
    .await;
    let source = EmbeddedSource::new(&mine).unwrap();

    let queries = source.rows(SystemTable::Queries).await.unwrap();
    let seen = texts(&queries);
    assert!(seen.iter().any(|t| t.contains("mine_marker")), "{seen:?}");
    assert!(!seen.iter().any(|t| t.contains("other_marker")), "{seen:?}");
    assert!(queries.iter().all(|r| r["session_id"] == mine.id()));

    let ids: Vec<&str> = queries
        .iter()
        .filter_map(|r| r["query_id"].as_str())
        .collect();
    let operators = source.rows(SystemTable::Operators).await.unwrap();
    assert!(!operators.is_empty(), "the session's query has operators");
    assert!(
        operators
            .iter()
            .all(|r| ids.contains(&r["query_id"].as_str().unwrap())),
        "an operator of another session's query is listed"
    );
    // Reading the pages adds nothing to the session's own history.
    let again = source.rows(SystemTable::Queries).await.unwrap();
    assert_eq!(again.len(), queries.len());
}

#[tokio::test]
async fn a_query_of_another_session_cannot_be_cancelled_and_an_own_one_can() {
    let (mine, other) = two_sessions();
    let source = EmbeddedSource::new(&mine).unwrap();

    let theirs = other.clone();
    let their_task = tokio::spawn(async move {
        let sql = format!("{ENDLESS} /* theirs */");
        theirs.sql(&sql).await.unwrap().collect().await
    });
    let their_id = running_id(&other, "theirs").await;
    assert!(!source.cancel(&their_id).await.unwrap());
    assert!(!source.cancel("not-an-id").await.unwrap());
    assert!(
        !their_task.is_finished(),
        "the other session's query was cancelled"
    );
    // Its memory pool is not among this console's resources either.
    let memory = source.rows(SystemTable::Memory).await.unwrap();
    assert!(
        memory.iter().all(|r| r["query_id"] != their_id.as_str()),
        "{memory:?}"
    );
    assert!(
        memory.iter().any(|r| r["query_id"].is_null()),
        "the process's pools are shown"
    );
    assert!(source.rows(SystemTable::Caches).await.is_ok());

    let own = mine.clone();
    let own_task = tokio::spawn(async move {
        let sql = format!("{ENDLESS} /* own */");
        own.sql(&sql).await.unwrap().collect().await
    });
    let own_id = running_id(&mine, "own").await;
    assert!(source.cancel(&own_id).await.unwrap());
    let err = tokio::time::timeout(Duration::from_secs(10), own_task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(err.error_class().name(), "OPERATION_CANCELED");

    assert_eq!(other.interrupt_all().len(), 1);
    assert!(their_task.await.unwrap().is_err());
}

/// The names the catalog page lists under `path`.
async fn listed(source: &EmbeddedSource, path: &[&str]) -> Vec<String> {
    let path: Vec<String> = path.iter().map(|p| (*p).to_owned()).collect();
    source
        .catalog(&path)
        .await
        .unwrap()
        .iter()
        .filter_map(|r| r["name"].as_str())
        .map(str::to_owned)
        .collect()
}

/// The page's statements name their catalog, so what is listed under `spark_catalog` does not change when
/// the program makes another catalog the current one.
#[tokio::test]
async fn spark_catalog_is_listed_whatever_the_sessions_current_catalog() {
    let (mine, _other) = two_sessions();
    run(&mine, "CREATE TEMPORARY VIEW seen AS SELECT 1 AS x").await;
    let source = EmbeddedSource::new(&mine).unwrap();
    let schemas = listed(&source, &["spark_catalog"]).await;
    let tables = listed(&source, &["spark_catalog", "default"]).await;
    assert!(schemas.contains(&"default".to_owned()), "{schemas:?}");
    assert!(tables.contains(&"seen".to_owned()), "{tables:?}");

    run(&mine, "USE system").await;
    let current = mine.catalog().current_catalog();
    assert_eq!(current, "system");
    assert_eq!(listed(&source, &["spark_catalog"]).await, schemas);
    assert_eq!(listed(&source, &["spark_catalog", "default"]).await, tables);
    // The current catalog's own schemas are another list.
    assert_ne!(listed(&source, &["system"]).await, schemas);
}

/// A temporary view is listed under every schema but has a one-part name: its columns are read by that name.
/// A table of the schema with the same name comes first, as the three-part name the page asked for names it.
#[tokio::test]
async fn the_catalog_lists_the_columns_of_a_temporary_view() {
    let (mine, _other) = two_sessions();
    run(
        &mine,
        "CREATE TEMPORARY VIEW only_a_view AS SELECT 1 AS from_the_view",
    )
    .await;
    run(
        &mine,
        "CREATE TABLE shadowed (from_the_table INT) USING parquet",
    )
    .await;
    run(
        &mine,
        "CREATE TEMPORARY VIEW shadowed AS SELECT 1 AS from_the_view",
    )
    .await;
    let source = EmbeddedSource::new(&mine).unwrap();
    for (name, column) in [
        ("only_a_view", "from_the_view"),
        ("shadowed", "from_the_table"),
    ] {
        let columns = listed(&source, &["spark_catalog", "default", name]).await;
        assert_eq!(columns, [column], "{name}");
    }
    // A name that is neither is still an error, and the first statement's.
    let path = ["spark_catalog", "default", "no_such_table"].map(str::to_owned);
    let missing = source.catalog(&path).await.unwrap_err();
    assert!(
        missing.message.contains("no_such_table"),
        "{}",
        missing.message
    );
}

#[tokio::test]
async fn the_editor_shares_the_sessions_views_and_the_catalog_lists_them() {
    let (mine, other) = two_sessions();
    run(&mine, "CREATE TEMPORARY VIEW from_program AS SELECT 7 AS x").await;
    let source = EmbeddedSource::new(&mine).unwrap();

    // The editor reads what the program made.
    let id = source
        .sql_start("SELECT x FROM from_program", None)
        .await
        .unwrap();
    let done = ended(&source, &id).await;
    assert_eq!(done["state"], "finished", "{done}");
    assert_eq!(done["rows"], serde_json::json!([[7]]));

    // The program reads what the editor made, and another session does not.
    let id = source
        .sql_start("CREATE TEMPORARY VIEW from_editor AS SELECT 8 AS y", None)
        .await
        .unwrap();
    assert_eq!(ended(&source, &id).await["state"], "finished");
    run(&mine, "SELECT y FROM from_editor").await;
    assert!(other.sql("SELECT y FROM from_editor").await.is_err());

    // The editor's statements are the session's own queries. A row of the editor is known by its run's tag
    // (the program's statements above carry none): both statements are listed, under this session.
    let queries = source.rows(SystemTable::Queries).await.unwrap();
    let from_editor: Vec<_> = queries.iter().filter(|r| of_editor(r)).collect();
    assert_eq!(from_editor.len(), 2, "{queries:?}");
    assert!(
        from_editor.iter().all(|r| r["session_id"] == mine.id()),
        "{from_editor:?}"
    );

    // The catalog shows the session's temporary views; listing them is not one of the session's queries.
    let before = source.rows(SystemTable::Queries).await.unwrap().len();
    let path = ["spark_catalog".to_owned(), "default".to_owned()];
    let tables = source.catalog(&path).await.unwrap();
    let names: Vec<&str> = tables.iter().filter_map(|r| r["name"].as_str()).collect();
    assert!(
        names.contains(&"from_program") && names.contains(&"from_editor"),
        "{names:?}"
    );
    assert_eq!(
        source.rows(SystemTable::Queries).await.unwrap().len(),
        before
    );

    // A running editor statement can be cancelled from the editor.
    let id = source.sql_start(ENDLESS, None).await.unwrap();
    let start = Instant::now();
    loop {
        let queries = source.rows(SystemTable::Queries).await.unwrap();
        if queries
            .iter()
            .any(|r| of_editor(r) && r["state"] == "running")
        {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(20), "{queries:?}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(source.sql_cancel(&id, None).await.unwrap());
    assert_eq!(ended(&source, &id).await["state"], "cancelled");
    // It ends as the session's own cancel, not as an operator's kill: this process has no operator.
    let queries = source.rows(SystemTable::Queries).await.unwrap();
    let cancelled: Vec<_> = queries
        .iter()
        .filter(|r| of_editor(r) && r["state"] == "cancelled")
        .collect();
    assert_eq!(cancelled.len(), 1, "{queries:?}");
    assert_eq!(cancelled[0]["cancel_reason"], "user", "{cancelled:?}");
    // (The row's error class is not asserted: `system.queries` leaves it null for every cancelled query.)
    assert_eq!(cancelled[0]["session_id"], mine.id());
}

#[tokio::test]
async fn the_environment_is_the_sessions_settings() {
    let (mine, other) = two_sessions();
    mine.conf().set("my.app.flag", "mine").unwrap();
    other.conf().set("my.app.flag", "theirs").unwrap();
    let source = EmbeddedSource::new(&mine).unwrap();
    let settings = source.environment().await.unwrap();
    let flag = settings.iter().find(|(k, _)| k == "my.app.flag");
    assert_eq!(flag.map(|(_, v)| v.as_str()), Some("mine"), "{settings:?}");
}

#[tokio::test]
async fn a_closed_session_serves_nothing() {
    let (mine, _) = two_sessions();
    let source = EmbeddedSource::new(&mine).unwrap();
    mine.close();
    for table in [
        SystemTable::Queries,
        SystemTable::Operators,
        SystemTable::Memory,
        SystemTable::Caches,
    ] {
        let e = source.rows(table).await.unwrap_err();
        assert!(e.to_string().contains("closed"), "{e}");
    }
    assert!(
        source
            .cancel("00112233-4455-6677-8899-aabbccddeeff")
            .await
            .is_err()
    );
    assert!(source.catalog(&[]).await.is_err());
    assert!(source.environment().await.is_err());
    assert!(source.sql_start("SELECT 1", None).await.is_err());
    assert!(source.sql_status("nothing", None).await.unwrap().is_none());
    assert!(!source.sql_cancel("nothing", None).await.unwrap());
    assert!(EmbeddedUi::for_session(&mine, 0).is_err());
}

async fn get(addr: std::net::SocketAddr, path: &str, token: Option<&str>) -> String {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    s.read_to_end(&mut out).await.unwrap();
    String::from_utf8(out).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_server_is_on_loopback_behind_a_token_and_stops_when_the_session_closes() {
    let (mine, other) = two_sessions();
    run(&mine, "SELECT 1 AS mine_marker").await;
    run(&other, "SELECT 1 AS other_marker").await;
    let ui = EmbeddedUi::for_session(&mine, 0).unwrap();
    let addr = ui.addr();
    assert!(addr.ip().is_loopback());
    assert_ne!(addr.port(), 0);
    assert!(
        ui.url()
            .starts_with(&format!("http://127.0.0.1:{}/?token=", addr.port()))
    );
    assert!(!ui.is_stopped());

    let denied = get(addr, "/api/internal/health", None).await;
    assert!(denied.starts_with("HTTP/1.1 401"), "{denied}");
    let wrong = get(addr, "/api/internal/health", Some("0000")).await;
    assert!(wrong.starts_with("HTTP/1.1 401"), "{wrong}");

    let health = get(addr, "/api/internal/health", Some(ui.token())).await;
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
    assert!(health.contains(r#""mode":"embedded""#), "{health}");
    assert!(
        health.contains(&format!(r#""session_id":"{}""#, mine.id())),
        "{health}"
    );

    let history = get(addr, "/api/internal/history", Some(ui.token())).await;
    assert!(history.contains("mine_marker"), "{history}");
    assert!(!history.contains("other_marker"), "{history}");

    // Closing the session stops the server without the handle being dropped.
    mine.close();
    let start = Instant::now();
    while !ui.is_stopped() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the server outlived its session"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(tokio::net::TcpStream::connect(addr).await.is_err());
    drop(ui);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_handle_stops_the_server_and_leaves_the_session_open() {
    let (mine, _) = two_sessions();
    let ui = EmbeddedUi::for_session(&mine, 0).unwrap();
    let addr = ui.addr();
    tokio::task::spawn_blocking(move || drop(ui)).await.unwrap();
    assert!(tokio::net::TcpStream::connect(addr).await.is_err());
    assert!(!mine.is_closed());
    run(&mine, "SELECT 1").await;
}
