// SPDX-License-Identifier: Apache-2.0
//! The UI backend over a real engine runtime: the system tables through `SessionSource`.
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use krishiv_server::internal::{SystemTable, UiSource};
use krishiv_server::session_source::{SessionSource, UI_PRINCIPAL};
use krishiv_session::{EngineRuntime, Session};
use serde_json::Value;

fn runtime() -> Arc<EngineRuntime> {
    EngineRuntime::builder().build().unwrap()
}

#[tokio::test]
async fn a_users_query_shows_up_in_the_queries_table_but_the_uis_own_polling_does_not() {
    let rt = runtime();
    let user = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    user.sql("SELECT 41 + 1 AS answer")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let source = SessionSource::new(rt).unwrap();
    let first = source.rows(SystemTable::Queries).await.unwrap();
    let again = source.rows(SystemTable::Queries).await.unwrap();
    assert!(
        first.iter().any(|r| r["query_text"]
            .as_str()
            .is_some_and(|t| t.contains("41 + 1"))),
        "{first:?}"
    );
    assert!(
        first
            .iter()
            .chain(&again)
            .all(|r| r["principal"] != Value::String(UI_PRINCIPAL.to_owned())),
        "the UI's own queries are hidden"
    );
    assert_eq!(
        first.len(),
        again.len(),
        "polling does not add rows to the list"
    );
    let row = first
        .iter()
        .find(|r| {
            r["query_text"]
                .as_str()
                .is_some_and(|t| t.contains("41 + 1"))
        })
        .unwrap();
    assert!(
        row["submitted_at"].is_i64(),
        "timestamps are microseconds: {row:?}"
    );
    assert_eq!(row["state"], "finished");
}

#[tokio::test]
async fn memory_and_caches_come_through_with_numbers_and_flags() {
    let source = SessionSource::new(runtime()).unwrap();
    let memory = source.rows(SystemTable::Memory).await.unwrap();
    assert!(!memory.is_empty());
    assert!(memory[0]["reserved_bytes"].is_i64());
    assert!(memory[0]["spillable"].is_boolean());
    let caches = source.rows(SystemTable::Caches).await.unwrap();
    assert!(caches.iter().all(|r| r["hits"].is_i64()));
}

#[tokio::test]
async fn a_populated_operator_row_converts_every_allowlisted_column() {
    let rt = runtime();
    let user = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    user.sql("SELECT id % 3 AS k, sum(id) FROM range(1000) GROUP BY id % 3")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let source = SessionSource::new(rt).unwrap();
    let operators = source.rows(SystemTable::Operators).await.unwrap();
    assert!(!operators.is_empty(), "the query recorded no operators");
    let sink = operators.iter().find(|r| r["name"] == "Result").unwrap();
    assert!(
        sink["query_id"].as_str().is_some_and(|q| q.len() == 36),
        "{sink:?}"
    );
    assert_eq!(sink["role"], "sink");
    for column in [
        "operator_id",
        "drivers",
        "rows_in",
        "compute_ns",
        "max_driver_compute_ns",
        "blocked_io_ns",
        "blocked_memory_ns",
        "blocked_sink_ns",
        "peak_reserved_bytes",
        "spilled_bytes",
    ] {
        // A column the engine has not filled yet is null; every filled one converts to a number.
        assert!(
            sink[column].is_i64() || sink[column].is_null(),
            "{column}: {sink:?}"
        );
    }
    assert_eq!(
        sink["rows_in"], 3,
        "the three groups reached the sink: {sink:?}"
    );
}

#[tokio::test]
async fn cancelling_an_unknown_query_is_false_and_a_malformed_id_is_an_error() {
    let source = SessionSource::new(runtime()).unwrap();
    assert!(
        !source
            .cancel("00112233-4455-6677-8899-aabbccddeeff")
            .await
            .unwrap()
    );
    assert!(
        !source.cancel("not-a-query-id").await.unwrap(),
        "an id that is not an id names no running query"
    );
}

#[tokio::test]
async fn an_embedded_ui_shows_the_hosts_own_queries_and_stops_on_drop() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    async fn get(addr: std::net::SocketAddr, path: &str, token: Option<&str>) -> String {
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
        let req =
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await.unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).await.unwrap();
        String::from_utf8(out).unwrap()
    }
    let rt = runtime();
    let user = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    user.sql("SELECT 7 * 6 AS embedded_marker")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ui = krishiv_server::session_source::EmbeddedUi::start(rt, "127.0.0.1:0".parse().unwrap())
        .unwrap();
    let addr = ui.addr();
    let token = ui.url().split("token=").nth(1).unwrap().to_owned();
    let denied = get(addr, "/api/internal/queries", None).await;
    assert!(denied.starts_with("HTTP/1.1 401"), "{denied}");
    let shown = get(addr, "/api/internal/history", Some(&token)).await;
    assert!(shown.starts_with("HTTP/1.1 200"), "{shown}");
    assert!(shown.contains("embedded_marker"), "{shown}");
    // An open event stream must not keep the stop waiting forever.
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "GET /api/internal/events?token={} HTTP/1.1\r\nHost: localhost\r\n\r\n",
        ui.token()
    );
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = [0_u8; 64];
    let _ = stream.read(&mut buf).await.unwrap();
    let stopped = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::task::spawn_blocking(move || drop(ui)),
    )
    .await;
    assert!(stopped.is_ok(), "drop hung on an open event stream");
    assert!(tokio::net::TcpStream::connect(addr).await.is_err());
}

/// ADR 0024, the modes table: a server's history survives a restart. The second runtime's own tables start empty;
/// the source adds what the store held when it was built, and a query in both is listed once, as the runtime has
/// it.
#[tokio::test]
async fn a_servers_history_lists_what_ran_before_its_restart() {
    use krishiv_session::QueryOptions;
    const OP: &str = "0193b1c2-7a00-7a01-8001-00000000aa01";
    async fn run(rt: &Arc<EngineRuntime>, sql: &str, options: QueryOptions) {
        let session = Session::builder().runtime(Arc::clone(rt)).build().unwrap();
        let df = session.sql_with(sql, options).await.unwrap();
        df.collect().await.unwrap();
    }
    fn of<'a>(rows: &'a [krishiv_server::internal::Row], marker: &str) -> Vec<&'a Value> {
        rows.iter()
            .filter(|r| r["query_text"].as_str().is_some_and(|t| t.contains(marker)))
            .map(|r| &r["query_id"])
            .collect()
    }
    fn operators_of(rows: &[krishiv_server::internal::Row], id: &Value) -> usize {
        rows.iter().filter(|r| &r["query_id"] == id).count()
    }
    let dir = tempfile::tempdir().unwrap();
    let with_store = || {
        EngineRuntime::builder()
            .config("krishiv.history.path", dir.path().to_str().unwrap())
            .build()
            .unwrap()
    };
    {
        let rt = with_store();
        let before = "SELECT count(*) AS before_restart FROM range(0, 100)";
        run(&rt, before, QueryOptions::new()).await;
        let again = "SELECT count(*) AS first_run FROM range(0, 100)";
        run(&rt, again, QueryOptions::new().operation_id(OP)).await;
        // Dropping the runtime flushes the history writer.
    }
    let stored = krishiv_session::read_profile_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 2, "{stored:?}");

    let rt = with_store();
    run(&rt, "SELECT 3 AS after_restart", QueryOptions::new()).await;
    let again = "SELECT count(*) AS second_run FROM range(0, 100)";
    run(&rt, again, QueryOptions::new().operation_id(OP)).await;
    let plain = SessionSource::new(Arc::clone(&rt)).unwrap();
    let source = SessionSource::new(rt).unwrap().with_stored_history(&stored);

    let queries = source.rows(SystemTable::Queries).await.unwrap();
    let before = of(&queries, "before_restart");
    assert_eq!(before.len(), 1, "{queries:?}");
    assert_eq!(of(&queries, "after_restart").len(), 1, "{queries:?}");
    // The id that ran in both lives is the runtime's row, not the store's.
    assert_eq!(
        of(&queries, "second_run"),
        [&Value::from(OP)],
        "{queries:?}"
    );
    assert!(of(&queries, "first_run").is_empty(), "{queries:?}");

    let operators = source.rows(SystemTable::Operators).await.unwrap();
    assert!(operators_of(&operators, before[0]) > 0, "{operators:?}");
    let live = plain.rows(SystemTable::Operators).await.unwrap();
    let op = Value::from(OP);
    assert!(operators_of(&live, &op) > 0, "{live:?}");
    assert_eq!(operators_of(&operators, &op), operators_of(&live, &op));

    // Without the store the restarted server knows this run only.
    let alone = plain.rows(SystemTable::Queries).await.unwrap();
    assert!(of(&alone, "before_restart").is_empty(), "{alone:?}");
}

/// A client that reuses its operation id after a restart: while the new run has reported no operator yet, its
/// id is in the queries table only. The stored operators of the earlier run must not stand in for it, or the
/// query's page would show the new run with the old run's metrics.
#[tokio::test]
async fn a_running_query_does_not_show_the_stored_operators_of_an_earlier_run_of_its_id() {
    use krishiv_session::QueryOptions;
    const OP: &str = "0193b1c2-7a00-7a01-8001-00000000bb01";
    let dir = tempfile::tempdir().unwrap();
    let with_store = || {
        EngineRuntime::builder()
            .config("krishiv.history.path", dir.path().to_str().unwrap())
            .build()
            .unwrap()
    };
    {
        let rt = with_store();
        let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
        let first = "SELECT count(*) AS first_run FROM range(0, 100)";
        let df = session
            .sql_with(first, QueryOptions::new().operation_id(OP))
            .await
            .unwrap();
        df.collect().await.unwrap();
    }
    let stored = krishiv_session::read_profile_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1, "{stored:?}");
    let doc: Value = serde_json::from_str(&stored[0].profile).unwrap();
    assert!(!doc["operators"].as_array().unwrap().is_empty());

    let rt = with_store();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let runner = session.clone();
    let task = tokio::spawn(async move {
        // A cross join no machine finishes soon.
        let sql = "SELECT count(*) FROM range(0, 1000000000) a CROSS JOIN range(0, 1000000000) b";
        let df = runner
            .sql_with(sql, QueryOptions::new().operation_id(OP))
            .await?;
        df.collect().await
    });
    let plain = SessionSource::new(Arc::clone(&rt)).unwrap();
    let source = SessionSource::new(rt).unwrap().with_stored_history(&stored);
    let op = Value::from(OP);
    let start = std::time::Instant::now();
    loop {
        let queries = plain.rows(SystemTable::Queries).await.unwrap();
        if queries
            .iter()
            .any(|q| q["query_id"] == op && q["state"] == "running")
        {
            break;
        }
        assert!(start.elapsed().as_secs() < 30, "the query never ran");
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let of_op = |rows: Vec<krishiv_server::internal::Row>| {
        rows.into_iter()
            .filter(|r| r["query_id"] == op)
            .collect::<Vec<_>>()
    };
    // One row for the id, the running one; and exactly the operators the runtime itself reports for it.
    let queries = of_op(source.rows(SystemTable::Queries).await.unwrap());
    assert_eq!(queries.len(), 1, "{queries:?}");
    assert_eq!(queries[0]["state"], "running");
    let live = of_op(plain.rows(SystemTable::Operators).await.unwrap());
    let shown = of_op(source.rows(SystemTable::Operators).await.unwrap());
    assert_eq!(shown.len(), live.len(), "{shown:?}");

    assert_eq!(session.interrupt_all().len(), 1);
    assert!(task.await.unwrap().is_err());
}

/// The console's own reads are stored like any query, about one a second while a page is open. A restarted
/// server must still list what its users ran: the console's profiles take no place among the newest it shows, and
/// none of them is listed.
#[tokio::test]
async fn the_consoles_own_reads_do_not_push_the_users_history_out_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let with_store = || {
        EngineRuntime::builder()
            .config("krishiv.history.path", dir.path().to_str().unwrap())
            .build()
            .unwrap()
    };
    {
        let rt = with_store();
        let user = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
        let df = user.sql("SELECT 5 AS the_users_query").await.unwrap();
        df.collect().await.unwrap();
        // An open page: each read is one stored query, so these are more than the 1,000 a server lists.
        let console = SessionSource::new(Arc::clone(&rt)).unwrap();
        for _ in 0..1_040 {
            console.rows(SystemTable::Queries).await.unwrap();
        }
        // Dropping the runtime flushes the history writer.
    }
    let stored = krishiv_session::read_profile_history(dir.path()).unwrap();
    let of_console = stored
        .iter()
        .filter(|p| p.principal == UI_PRINCIPAL)
        .count();
    assert!(
        of_console > 1_000,
        "{} stored, too few of them the console's to reach the cut",
        stored.len()
    );

    let principals = |rows: &[krishiv_server::internal::Row]| -> Vec<String> {
        rows.iter()
            .filter_map(|r| r["principal"].as_str().map(str::to_owned))
            .collect()
    };
    let source = SessionSource::new(with_store())
        .unwrap()
        .with_stored_history(&stored);
    let queries = source.rows(SystemTable::Queries).await.unwrap();
    // The store keeps the statement as `system.queries` shows it, its literals replaced.
    let texts: Vec<_> = queries.iter().map(|r| &r["query_text"]).collect();
    assert_eq!(
        texts,
        ["SELECT ? AS the_users_query"],
        "{of_console} stored"
    );
    assert!(!principals(&queries).iter().any(|p| p == UI_PRINCIPAL));

    // `krishiv ui --profiles` over a server's folder leaves them out too.
    let saved = krishiv_server::session_source::ProfilesSource::from_profiles(&stored);
    let saved = saved.rows(SystemTable::Queries).await.unwrap();
    assert_eq!(saved.len(), 1, "{saved:?}");
    assert!(!principals(&saved).iter().any(|p| p == UI_PRINCIPAL));
}

/// A query's row carries its tags (the session's and the request's) and the client type of its session, so the
/// server console can filter on them.
#[tokio::test]
async fn a_query_row_carries_its_tags_and_its_sessions_client_type() {
    let rt = runtime();
    let user = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    user.set_client_info(
        krishiv_session::Transport::Connect,
        Some("_SPARK_CONNECT_PYTHON spark/4.2.0"),
        None,
        None,
    );
    user.set_tag("team", "risk").unwrap();
    user.sql_with(
        "SELECT 7 AS tagged",
        krishiv_session::QueryOptions::new().tags(["nightly"]),
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let plain = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    plain
        .sql("SELECT 8 AS plain")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let rows = SessionSource::new(rt)
        .unwrap()
        .rows(SystemTable::Queries)
        .await
        .unwrap();
    let of = |text: &str| {
        rows.iter()
            .find(|r| r["query_text"].as_str().is_some_and(|t| t.contains(text)))
            .unwrap_or_else(|| panic!("{text}: {rows:?}"))
    };
    let tagged = of("7 AS tagged");
    assert_eq!(tagged["client_type"], "_SPARK_CONNECT_PYTHON spark/4.2.0");
    assert_eq!(tagged["tags"]["team"], "risk");
    assert_eq!(tagged["tags"]["nightly"], "");
    let plain = of("8 AS plain");
    assert_eq!(plain["client_type"], Value::Null);
    assert_eq!(plain["tags"], serde_json::json!({}));
}

/// An operation id is the client's and is scoped to its session, so two sessions may run a query under the
/// same one: the console lists both, each with its own session's client type.
#[tokio::test]
async fn two_sessions_queries_under_one_operation_id_are_both_listed() {
    const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";
    let rt = runtime();
    let mut sessions = Vec::new();
    for client in ["client-a", "client-b"] {
        let s = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
        s.set_client_info(
            krishiv_session::Transport::Connect,
            Some(client),
            None,
            None,
        );
        s.sql_with(
            "SELECT 9 AS shared_id",
            krishiv_session::QueryOptions::new().operation_id(OP),
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
        sessions.push(s);
    }
    let rows = SessionSource::new(rt)
        .unwrap()
        .rows(SystemTable::Queries)
        .await
        .unwrap();
    let mut clients: Vec<&str> = rows
        .iter()
        .filter(|r| r["query_id"] == OP)
        .filter_map(|r| r["client_type"].as_str())
        .collect();
    clients.sort_unstable();
    assert_eq!(clients, ["client-a", "client-b"], "{rows:?}");
}

/// How many statements the console itself has run on `rt`, read through an ordinary session.
async fn console_statements(rt: &Arc<EngineRuntime>) -> usize {
    let reader = Session::builder().runtime(Arc::clone(rt)).build().unwrap();
    let batches = reader
        .sql("SELECT principal FROM system.queries")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    krishiv_server::session_source::batches_to_rows(&batches)
        .iter()
        .filter(|r| r["principal"] == UI_PRINCIPAL)
        .count()
}

/// A read of the queries is one engine statement, the session's client type included, and a read of another
/// table is one too: the console's polling is not what fills the runtime's history.
#[tokio::test]
async fn one_read_of_a_table_is_one_statement() {
    let rt = runtime();
    let source = SessionSource::new(Arc::clone(&rt)).unwrap();
    let before = console_statements(&rt).await;
    source.rows(SystemTable::Queries).await.unwrap();
    assert_eq!(console_statements(&rt).await, before + 1);
    source.rows(SystemTable::Sessions).await.unwrap();
    assert_eq!(console_statements(&rt).await, before + 2);
}

#[tokio::test]
async fn a_source_lists_udf_workers_only_when_it_was_given_a_pool_and_reads_it_each_time() {
    use std::sync::atomic::{AtomicU64, Ordering};

    use krishiv_server::udf_workers::{UdfWorkers, UdfWorkersProvider};

    let none = SessionSource::new(runtime()).unwrap();
    assert_eq!(none.udf_workers().await.unwrap(), None);

    let live = Arc::new(AtomicU64::new(1));
    let read = Arc::clone(&live);
    let provider: UdfWorkersProvider = Arc::new(move || UdfWorkers {
        live: read.load(Ordering::SeqCst),
        max_total: 64,
        ..UdfWorkers::default()
    });
    let source = SessionSource::new(runtime())
        .unwrap()
        .with_udf_workers(provider);
    assert_eq!(source.udf_workers().await.unwrap().unwrap().live, 1);
    live.store(3, Ordering::SeqCst);
    let pool = source.udf_workers().await.unwrap().unwrap();
    assert_eq!((pool.live, pool.max_total), (3, 64));
}
