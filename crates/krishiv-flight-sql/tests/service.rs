// SPDX-License-Identifier: Apache-2.0
//! The Flight SQL service end to end on a Connect listener (design P3-T101 §11): a raw tonic
//! Flight client against `ConnectServer::with_routes(FlightSqlServer::service())`, token auth.
#![cfg(test)]
#![cfg(unix)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::sql::{CommandGetSqlInfo, CommandStatementQuery, ProstMessageExt};
use arrow_flight::{
    Action, CancelFlightInfoRequest, CancelFlightInfoResult, CancelStatus, FlightDescriptor,
    FlightInfo, HandshakeRequest, Ticket,
};
use futures::{StreamExt, TryStreamExt};
use krishiv_common::config::Config;
use krishiv_connect_server::auth::role_sessions;
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, SessionLimits, SessionManager,
};
use krishiv_flight_sql::{FlightSqlConfig, FlightSqlServer, ServerInfo};
use krishiv_session::Session;
use krishiv_session::auth::{AuthConfig, Authenticator, DeniedCounts, ServerAuth};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::service::Routes;
use tonic::transport::Channel;
use tonic::{Code, Request, Status};

const OPS: &str = "flight-operator-token-0123456789abcdef";
const OTHER: &str = "flight-other-token-0123456789abcdefgh";

fn secret(dir: &Path, name: &str, token: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(name);
    std::fs::write(&path, token).expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    format!("file:{}", path.display())
}

/// Two operators: `ops` and `other`.
fn server_auth(dir: &Path) -> Arc<ServerAuth> {
    let o = secret(dir, "o", OPS);
    let t = secret(dir, "t", OTHER);
    let config = AuthConfig::from_settings([
        ("krishiv.auth.tokens.o.secret", o.as_str()),
        ("krishiv.auth.tokens.o.principal", "ops"),
        ("krishiv.auth.tokens.o.role", "operator"),
        ("krishiv.auth.tokens.t.secret", t.as_str()),
        ("krishiv.auth.tokens.t.principal", "other"),
        ("krishiv.auth.tokens.t.role", "operator"),
    ])
    .expect("auth config");
    Arc::new(ServerAuth::new(&config).expect("auth"))
}

struct Harness {
    flight: FlightSqlServer,
    manager: Arc<SessionManager>,
    draining: Arc<std::sync::atomic::AtomicBool>,
    client: FlightServiceClient<Channel>,
    _stop: oneshot::Sender<()>,
    _dir: tempfile::TempDir,
}

async fn start(config: FlightSqlConfig) -> Harness {
    let dir = tempfile::tempdir().expect("tempdir");
    let auth = server_auth(dir.path());
    let denials = Arc::new(DeniedCounts::new());
    let factory = role_sessions(
        Session::new().expect("session"),
        &Config::new(),
        Arc::clone(&denials),
    )
    .expect("factory");
    let manager = Arc::new(SessionManager::new(factory, SessionLimits::default()));
    let connect_auth = || {
        ConnectAuth::new(Arc::clone(&auth) as Arc<dyn Authenticator>)
            .with_denials(Arc::clone(&denials))
    };
    let server = ConnectServer::new(Arc::clone(&manager), ConnectServerConfig::default())
        .with_auth(connect_auth());
    let flight = FlightSqlServer::new(
        Arc::clone(&manager),
        Arc::new(connect_auth()),
        server.draining(),
        server.response_budget().clone(),
        &ServerInfo::default(),
        config,
    )
    .expect("flight server");
    let draining = server.draining();
    let server = server.with_routes(Routes::new(flight.service()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    Harness {
        flight,
        manager,
        draining,
        client: FlightServiceClient::new(channel),
        _stop: stop,
        _dir: dir,
    }
}

fn authed<T>(message: T, token: &str, cookie: Option<&str>) -> Request<T> {
    let mut r = Request::new(message);
    r.metadata_mut().insert(
        "authorization",
        format!("Bearer {token}").parse().expect("header"),
    );
    if let Some(id) = cookie {
        r.metadata_mut().insert(
            "cookie",
            format!("arrow_flight_session_id={id}")
                .parse()
                .expect("header"),
        );
    }
    r
}

/// The session id a response's `set-cookie` carries.
fn set_cookie(metadata: &tonic::metadata::MetadataMap) -> Option<String> {
    let v = metadata.get("set-cookie")?.to_str().ok()?;
    let rest = v.strip_prefix("arrow_flight_session_id=")?;
    Some(rest.split(';').next()?.to_owned())
}

/// `GetFlightInfo(CommandStatementQuery{sql})`: the info and the session id set.
async fn info(
    h: &mut Harness,
    token: &str,
    cookie: Option<&str>,
    sql: &str,
) -> Result<(FlightInfo, Option<String>), Status> {
    let cmd = CommandStatementQuery {
        query: sql.to_owned(),
        transaction_id: None,
    };
    let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
    let response = h
        .client
        .get_flight_info(authed(descriptor, token, cookie))
        .await?;
    let id = set_cookie(response.metadata());
    Ok((response.into_inner(), id))
}

fn ticket(info: &FlightInfo) -> Ticket {
    info.endpoint[0].ticket.clone().expect("ticket")
}

/// `DoGet(ticket)`, decoded.
async fn fetch(
    h: &mut Harness,
    token: &str,
    cookie: Option<&str>,
    ticket: Ticket,
) -> Result<Vec<RecordBatch>, Status> {
    let stream = h.client.do_get(authed(ticket, token, cookie)).await?;
    let data = stream
        .into_inner()
        .map_err(|s| FlightError::Tonic(Box::new(s)));
    FlightRecordBatchStream::new_from_flight_data(data)
        .try_collect()
        .await
        .map_err(|e| match e {
            FlightError::Tonic(s) => *s,
            other => Status::internal(other.to_string()),
        })
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Waits until the server's session table has `n` rows (releases run when a stream ends).
async fn wait_sessions(h: &Harness, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while h.flight.session_count() != n {
        assert!(
            Instant::now() < deadline,
            "{} sessions, want {n}",
            h.flight.session_count()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_query_round_trips_with_spark_types() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (info, cookie) = info(&mut h, OPS, None, "SELECT 1 AS a, 'x' AS b, 2L AS c")
        .await
        .expect("info");
    assert!(cookie.is_some(), "a cookie-less call gets a session cookie");
    assert_eq!(info.total_records, -1);
    let batches = fetch(&mut h, OPS, None, ticket(&info))
        .await
        .expect("fetch");
    assert_eq!(rows(&batches), 1);
    let b = &batches[0];
    let a = b
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("INT");
    let s = b
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("STRING");
    let c = b
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("BIGINT");
    assert_eq!((a.value(0), s.value(0), c.value(0)), (1, "x", 2));
    assert_eq!(b.schema().field(0).name(), "a");
}

#[tokio::test(flavor = "multi_thread")]
async fn cookieless_session_released_after_doget() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (info, _) = info(&mut h, OPS, None, "SELECT 1").await.expect("info");
    assert_eq!(h.flight.session_count(), 1, "provisional until fetched");
    fetch(&mut h, OPS, None, ticket(&info))
        .await
        .expect("fetch");
    wait_sessions(&h, 0).await;
    assert_eq!(h.flight.handle_count(), 0, "its handle went with it");
    // One fetch per ticket without a cookie.
    let again = fetch(&mut h, OPS, None, ticket(&info))
        .await
        .expect_err("released");
    assert_eq!(again.code(), Code::NotFound, "{again:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cookie_session_keeps_state_and_a_command_runs_once() {
    let mut h = start(FlightSqlConfig::default()).await;
    // A command runs at GetFlightInfo, once: a second run would fail as the view exists.
    let (created, cookie) = info(
        &mut h,
        OPS,
        None,
        "CREATE TEMPORARY VIEW v AS SELECT 7 AS n",
    )
    .await
    .expect("info");
    let cookie = cookie.expect("cookie");
    let t = ticket(&created);
    fetch(&mut h, OPS, Some(&cookie), t.clone())
        .await
        .expect("first fetch");
    fetch(&mut h, OPS, Some(&cookie), t)
        .await
        .expect("re-fetch serves the stored rows");
    // The cookie session keeps the view for the next statement.
    let (q, set) = info(&mut h, OPS, Some(&cookie), "SELECT n FROM v")
        .await
        .expect("info");
    assert_eq!(set, None, "a presented cookie sets none");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("INT");
    assert_eq!(n.value(0), 7);
    assert_eq!(h.flight.session_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_query_ticket_is_refetched_with_the_same_clock() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (info, cookie) = info(&mut h, OPS, None, "SELECT current_timestamp() AS t")
        .await
        .expect("info");
    let cookie = cookie.expect("cookie");
    let first = fetch(&mut h, OPS, Some(&cookie), ticket(&info))
        .await
        .expect("fetch");
    tokio::time::sleep(Duration::from_millis(5)).await;
    let second = fetch(&mut h, OPS, Some(&cookie), ticket(&info))
        .await
        .expect("re-fetch");
    assert_eq!(first, second);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_ticket_is_its_owners_only() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (info, cookie) = info(&mut h, OPS, None, "SELECT 1").await.expect("info");
    let e = fetch(&mut h, OTHER, cookie.as_deref(), ticket(&info))
        .await
        .expect_err("another principal");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    // The owner still fetches it.
    fetch(&mut h, OPS, cookie.as_deref(), ticket(&info))
        .await
        .expect("owner");
}

#[tokio::test(flavor = "multi_thread")]
async fn flight_sql_requires_auth() {
    let mut h = start(FlightSqlConfig::default()).await;
    let cmd = CommandStatementQuery {
        query: "SELECT 1".to_owned(),
        transaction_id: None,
    };
    let e = h
        .client
        .get_flight_info(Request::new(FlightDescriptor::new_cmd(
            cmd.as_any().encode_to_vec(),
        )))
        .await
        .expect_err("no token");
    assert_eq!(e.code(), Code::Unauthenticated, "{e:?}");
    assert_eq!(h.flight.session_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn handshake_echoes_bearer() {
    let mut h = start(FlightSqlConfig::default()).await;
    let request = authed(
        futures::stream::iter([HandshakeRequest::default()]),
        OPS,
        None,
    );
    let response = h.client.handshake(request).await.expect("handshake");
    assert_eq!(
        response
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some(format!("Bearer {OPS}").as_str())
    );
    assert_eq!(h.flight.session_count(), 0, "Handshake creates no session");
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_info_needs_no_session() {
    let mut h = start(FlightSqlConfig::default()).await;
    let cmd = CommandGetSqlInfo { info: vec![] };
    let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
    let info = h
        .client
        .get_flight_info(authed(descriptor, OPS, None))
        .await
        .expect("info")
        .into_inner();
    let batches = fetch(&mut h, OPS, None, ticket(&info))
        .await
        .expect("fetch");
    assert!(rows(&batches) > 0);
    assert_eq!(h.flight.session_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn statement_over_max_bytes_invalid_argument() {
    let mut h = start(FlightSqlConfig {
        max_statement_bytes: 16,
        ..FlightSqlConfig::default()
    })
    .await;
    let e = info(&mut h, OPS, None, "SELECT 1 AS a_long_name")
        .await
        .expect_err("too long");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(h.flight.session_count(), 0, "refused before any session");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_statement_gives_back_its_provisional_session() {
    let mut h = start(FlightSqlConfig::default()).await;
    let e = info(&mut h, OPS, None, "SELECT * FROM no_such_table")
        .await
        .expect_err("unknown table");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    assert_eq!(
        e.metadata()
            .get("x-krishiv-error-class")
            .and_then(|v| v.to_str().ok()),
        Some("TABLE_OR_VIEW_NOT_FOUND")
    );
    wait_sessions(&h, 0).await;
    assert_eq!(h.flight.handle_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_flight_info_before_doget() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (info, cookie) = info(&mut h, OPS, None, "SELECT 1").await.expect("info");
    let body = CancelFlightInfoRequest::new(info.clone()).encode_to_vec();
    let action = Action::new("CancelFlightInfo", body);
    let mut results = h
        .client
        .do_action(authed(action, OPS, cookie.as_deref()))
        .await
        .expect("cancel")
        .into_inner();
    let result = results.next().await.expect("one result").expect("ok");
    let status = CancelFlightInfoResult::decode(result.body)
        .expect("decode")
        .status;
    assert_eq!(status, CancelStatus::Cancelled as i32);
    let e = fetch(&mut h, OPS, cookie.as_deref(), ticket(&info))
        .await
        .expect_err("released");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    // Its provisional session held only that handle: released (rule 3).
    wait_sessions(&h, 0).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn unsupported_actions_unimplemented() {
    let mut h = start(FlightSqlConfig::default()).await;
    let e = h
        .client
        .do_action(authed(Action::new("NoSuchAction", Vec::new()), OPS, None))
        .await
        .expect_err("unknown action");
    assert_eq!(e.code(), Code::Unimplemented, "{e:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn stalled_reader_cancelled_after_idle_timeout() {
    let mut h = start(FlightSqlConfig {
        fetch_idle_timeout: Duration::from_millis(200),
        encode: krishiv_flight_sql::EncodeLimits {
            batch_bytes: 64 << 10,
            data_bytes: 16 << 10,
        },
        ..FlightSqlConfig::default()
    })
    .await;
    let (info, _) = info(
        &mut h,
        OPS,
        None,
        "SELECT id, id * 2 AS d FROM range(0, 5000000)",
    )
    .await
    .expect("info");
    let mut stream = h
        .client
        .do_get(authed(ticket(&info), OPS, None))
        .await
        .expect("do_get")
        .into_inner();
    stream
        .message()
        .await
        .expect("first message")
        .expect("schema");
    // The reader stops: tonic fills the HTTP/2 window and then owes the stream a poll.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let swept_at = Instant::now() + Duration::from_secs(1);
    h.flight.sweep_at(swept_at);
    // Its query was cancelled and its session released; the stream ends with the typed error
    // after what was already buffered.
    wait_sessions(&h, 0).await;
    let end = loop {
        match stream.message().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the stream ended without the idle error"),
            Err(status) => break status,
        }
    };
    assert_eq!(end.code(), Code::DeadlineExceeded, "{end:?}");
    assert_eq!(
        end.metadata()
            .get("x-krishiv-error-class")
            .and_then(|v| v.to_str().ok()),
        Some("KRISHIV_FLIGHT_SQL_FETCH_IDLE")
    );
}

/// Waits until the manager reports `n` busy units: running queries plus live operations (a
/// running `DoGet` is one of each).
async fn wait_busy(h: &Harness, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while h.manager.busy() != n {
        assert!(
            Instant::now() < deadline,
            "{} busy, want {n}",
            h.manager.busy()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A `DoGet` that has delivered its first message (the schema) and is then left unread.
async fn open_fetch(
    h: &mut Harness,
    cookie: Option<&str>,
    t: Ticket,
) -> tonic::Streaming<arrow_flight::FlightData> {
    let mut s = h
        .client
        .do_get(authed(t, OPS, cookie))
        .await
        .expect("do_get")
        .into_inner();
    s.message().await.expect("first message").expect("schema");
    s
}

const LONG: &str = "SELECT id, id * 2 AS d FROM range(0, 50000000)";

#[tokio::test(flavor = "multi_thread")]
async fn flight_sql_refused_while_draining() {
    let mut h = start(FlightSqlConfig::default()).await;
    h.draining.store(true, std::sync::atomic::Ordering::Release);
    let e = info(&mut h, OPS, None, "SELECT 1")
        .await
        .expect_err("draining");
    assert_eq!(e.code(), Code::Unavailable, "{e:?}");
    assert_eq!(e.message(), "server shutting down");
    assert_eq!(h.flight.session_count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn dropped_doget_stream_cancels_query() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (i, cookie) = info(&mut h, OPS, None, LONG).await.expect("info");
    let cookie = cookie.expect("cookie");
    let stream = open_fetch(&mut h, Some(&cookie), ticket(&i)).await;
    wait_busy(&h, 2).await;
    drop(stream);
    // The dropped stream ends its operation and cancels its query; the cookie session stays.
    wait_busy(&h, 0).await;
    assert_eq!(h.flight.session_count(), 1);
    assert_eq!(h.flight.handle_count(), 1, "the ticket stays fetchable");
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_doget_are_two_operations() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (i, cookie) = info(&mut h, OPS, None, LONG).await.expect("info");
    let cookie = cookie.expect("cookie");
    let a = open_fetch(&mut h, Some(&cookie), ticket(&i)).await;
    let b = open_fetch(&mut h, Some(&cookie), ticket(&i)).await;
    // Two operations, each with its running query.
    wait_busy(&h, 4).await;
    drop(a);
    wait_busy(&h, 2).await;
    assert_eq!(h.manager.running_queries(), 1, "the other fetch still runs");
    drop(b);
    wait_busy(&h, 0).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn cookie_session_runs_more_statements_than_handle_cap() {
    let mut h = start(FlightSqlConfig {
        handles: krishiv_flight_sql::HandleLimits {
            per_session: 2,
            ..krishiv_flight_sql::HandleLimits::default()
        },
        ..FlightSqlConfig::default()
    })
    .await;
    let (first, cookie) = info(&mut h, OPS, None, "SELECT 1").await.expect("first");
    let cookie = cookie.expect("cookie");
    let (second, _) = info(&mut h, OPS, Some(&cookie), "SELECT 2")
        .await
        .expect("second");
    let (third, _) = info(&mut h, OPS, Some(&cookie), "SELECT 3")
        .await
        .expect("third: evicts");
    assert_eq!(h.flight.handle_count(), 2);
    let e = fetch(&mut h, OPS, Some(&cookie), ticket(&first))
        .await
        .expect_err("the oldest idle ticket was evicted");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    for t in [ticket(&second), ticket(&third)] {
        fetch(&mut h, OPS, Some(&cookie), t).await.expect("kept");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn refetch_schema_changed_fails_precondition() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (v, cookie) = info(
        &mut h,
        OPS,
        None,
        "CREATE TEMPORARY VIEW v AS SELECT 1 AS a",
    )
    .await
    .expect("view");
    let cookie = cookie.expect("cookie");
    drop(v);
    let (q, _) = info(&mut h, OPS, Some(&cookie), "SELECT a FROM v")
        .await
        .expect("query");
    info(
        &mut h,
        OPS,
        Some(&cookie),
        "CREATE OR REPLACE TEMPORARY VIEW v AS SELECT 'x' AS a",
    )
    .await
    .expect("replace");
    let e = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect_err("schema changed");
    assert_eq!(e.code(), Code::FailedPrecondition, "{e:?}");
    assert_eq!(
        e.metadata()
            .get("x-krishiv-error-class")
            .and_then(|v| v.to_str().ok()),
        Some("KRISHIV_FLIGHT_SQL_SCHEMA_CHANGED")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn command_result_not_stored_says_it_ran() {
    let mut h = start(FlightSqlConfig {
        max_command_result_bytes: 1,
        ..FlightSqlConfig::default()
    })
    .await;
    let e = info(&mut h, OPS, None, "SET spark.sql.shuffle.partitions = 7")
        .await
        .expect_err("its one-row result is over 1 byte");
    assert_eq!(e.code(), Code::FailedPrecondition, "{e:?}");
    assert_eq!(
        e.metadata()
            .get("x-krishiv-error-class")
            .and_then(|v| v.to_str().ok()),
        Some("KRISHIV_FLIGHT_SQL_RESULT_NOT_STORED")
    );
    assert!(e.message().contains("The statement ran"), "{}", e.message());
    assert_eq!(h.flight.handle_count(), 0, "the slot was given back");
}

/// `CancelFlightInfo` for one ticket cancels that ticket's fetches only (§4.5, review M1): another
/// ticket's fetch in the same session keeps running.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_flight_info_cancels_only_its_handle() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (a, cookie) = info(&mut h, OPS, None, LONG).await.expect("info a");
    let cookie = cookie.expect("cookie");
    let (b, _) = info(&mut h, OPS, Some(&cookie), LONG)
        .await
        .expect("info b");
    let mut sa = open_fetch(&mut h, Some(&cookie), ticket(&a)).await;
    let mut sb = open_fetch(&mut h, Some(&cookie), ticket(&b)).await;
    wait_busy(&h, 4).await;
    let body = CancelFlightInfoRequest::new(a.clone()).encode_to_vec();
    let mut results = h
        .client
        .do_action(authed(
            Action::new("CancelFlightInfo", body),
            OPS,
            Some(&cookie),
        ))
        .await
        .expect("cancel")
        .into_inner();
    let result = results.next().await.expect("one result").expect("ok");
    assert_eq!(
        CancelFlightInfoResult::decode(result.body)
            .expect("decode")
            .status,
        CancelStatus::Cancelled as i32
    );
    // A's fetch ends; B's still runs and still delivers.
    let end = loop {
        match sa.message().await {
            Ok(Some(_)) => {}
            Ok(None) => break None,
            Err(status) => break Some(status),
        }
    };
    assert!(end.is_some(), "the cancelled fetch ends with an error");
    assert_eq!(h.manager.running_queries(), 1, "B's query still runs");
    sb.message().await.expect("B reads on").expect("a piece");
    drop(sb);
    wait_busy(&h, 0).await;
}

/// A stale cookie, then a failing statement: the reset session the call created is released, so
/// retries don't pile up to the session cap (review M4 b).
#[tokio::test(flavor = "multi_thread")]
async fn refused_cookie_then_error_releases_the_reset_session() {
    let mut h = start(FlightSqlConfig::default()).await;
    for _ in 0..3 {
        let e = info(
            &mut h,
            OPS,
            Some("stale-session"),
            "SELECT * FROM no_such_table",
        )
        .await
        .expect_err("unknown table");
        assert_eq!(e.code(), Code::NotFound, "{e:?}");
    }
    wait_sessions(&h, 0).await;
    assert_eq!(h.manager.counts().live, 0, "the manager released them too");
}

/// A `GetFlightInfo` dropped while its command runs (its session already marked changed) gives
/// back the session and the slot it created: the client never saw either (§16 M3, review M4 a).
#[tokio::test(flavor = "multi_thread")]
async fn dropped_get_flight_info_returns_slot() {
    let mut h = start(FlightSqlConfig::default()).await;
    let r = tokio::time::timeout(
        Duration::from_millis(300),
        info(
            &mut h,
            OPS,
            None,
            "EXECUTE IMMEDIATE 'SELECT sum(id) FROM range(0, 50000000000)'",
        ),
    )
    .await;
    assert!(r.is_err(), "the command was still running: {r:?}");
    wait_sessions(&h, 0).await;
    assert_eq!(h.flight.handle_count(), 0);
    wait_busy(&h, 0).await;
}
