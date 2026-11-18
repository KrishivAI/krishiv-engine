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
    /// Sending (or dropping) it starts `serve`'s shutdown.
    stop: Option<oneshot::Sender<()>>,
    /// `serve`'s task: it ends once shutdown has waited for the running operations.
    served: tokio::task::JoinHandle<Result<(), krishiv_common::error::KrishivError>>,
    /// The listener's `maxResponseBytes`, shared by Connect operations and Flight fetches.
    budget: krishiv_connect_server::execute::ResponseBudget,
    /// The server's authenticator, for a test that builds a Connect session itself.
    auth: Arc<ServerAuth>,
    _dir: tempfile::TempDir,
}

async fn start(config: FlightSqlConfig) -> Harness {
    start_with(
        config,
        ConnectServerConfig::default(),
        None,
        SessionLimits::default(),
    )
    .await
}

/// `stream_window`: the client's HTTP/2 stream window, so a test can keep a fetch's last pieces
/// from fitting in the transport's buffers.
async fn start_with(
    config: FlightSqlConfig,
    server_config: ConnectServerConfig,
    stream_window: Option<u32>,
    limits: SessionLimits,
) -> Harness {
    let dir = tempfile::tempdir().expect("tempdir");
    let auth = server_auth(dir.path());
    let denials = Arc::new(DeniedCounts::new());
    let factory = role_sessions(
        Session::new().expect("session"),
        &Config::new(),
        Arc::clone(&denials),
    )
    .expect("factory");
    let manager = Arc::new(SessionManager::new(factory, limits));
    let connect_auth = || {
        ConnectAuth::new(Arc::clone(&auth) as Arc<dyn Authenticator>)
            .with_denials(Arc::clone(&denials))
    };
    let server = ConnectServer::new(Arc::clone(&manager), server_config).with_auth(connect_auth());
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
    let budget = server.response_budget().clone();
    let server = server.with_routes(Routes::new(flight.service()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (stop, stopped) = oneshot::channel::<()>();
    let served = tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .initial_stream_window_size(stream_window)
        .connect()
        .await
        .expect("connect");
    Harness {
        flight,
        manager,
        draining,
        client: FlightServiceClient::new(channel),
        stop: Some(stop),
        served,
        budget,
        auth,
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
    // The reader stops: once tonic has filled the HTTP/2 window it owes the stream a poll, which
    // may take a while under load, so the sweep is repeated until it expires the fetch (m11).
    let deadline = Instant::now() + Duration::from_secs(10);
    while h.flight.session_count() != 0 {
        assert!(Instant::now() < deadline, "the fetch never went idle");
        tokio::time::sleep(Duration::from_millis(100)).await;
        h.flight.sweep_at(Instant::now() + Duration::from_secs(1));
    }
    // Its query was cancelled and its session released; the stream ends with the typed error
    // after what was already buffered.
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

/// The request counts the server's `/metrics` merges (design §7, review m6).
#[tokio::test(flavor = "multi_thread")]
async fn requests_reach_the_metrics_snapshot() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (i, cookie) = info(&mut h, OPS, None, "SELECT 1").await.expect("info");
    fetch(&mut h, OPS, cookie.as_deref(), ticket(&i))
        .await
        .expect("fetch");
    let snapshot = h.flight.metrics();
    let ok_query = snapshot.counters.iter().any(|c| {
        c.name == "krishiv.flight_sql.requests"
            && c.labels.contains(&("outcome".to_owned(), "ok".to_owned()))
            && c.value >= 1.0
    });
    assert!(ok_query, "{:?}", snapshot.counters);
}

/// Fail closed (design §16 m5, review m7): what 65a does not serve is `UNIMPLEMENTED` before
/// arrow-rs decodes its body (an action body that is not a protobuf still answers so), and each
/// refusal is counted as `unsupported`.
#[tokio::test(flavor = "multi_thread")]
async fn unsupported_rpcs_unimplemented_before_decoding() {
    let mut h = start(FlightSqlConfig::default()).await;
    for kind in [
        "ClosePreparedStatement",
        "BeginTransaction",
        "CreatePreparedStatement",
    ] {
        let e = h
            .client
            .do_action(authed(
                Action::new(kind, b"not a protobuf".to_vec()),
                OPS,
                None,
            ))
            .await
            .expect_err(kind);
        assert_eq!(e.code(), Code::Unimplemented, "{kind}: {e:?}");
    }
    let catalogs = arrow_flight::sql::CommandGetCatalogs {};
    let e = h
        .client
        .get_flight_info(authed(
            FlightDescriptor::new_cmd(catalogs.as_any().encode_to_vec()),
            OPS,
            None,
        ))
        .await
        .expect_err("GetCatalogs is 65b");
    assert_eq!(e.code(), Code::Unimplemented, "{e:?}");
    let e = h
        .client
        .do_put(authed(
            futures::stream::empty::<arrow_flight::FlightData>(),
            OPS,
            None,
        ))
        .await
        .expect_err("DoPut is 65b");
    assert_eq!(e.code(), Code::Unimplemented, "{e:?}");
    let e = h
        .client
        .do_get(authed(Ticket::new(vec![0xff, 0xff, 0xff]), OPS, None))
        .await
        .expect_err("not a command");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(h.flight.session_count(), 0, "no session work");
    let unsupported: f64 = h
        .flight
        .metrics()
        .counters
        .iter()
        .filter(|c| {
            c.labels
                .contains(&("command".to_owned(), "unsupported".to_owned()))
        })
        .map(|c| c.value)
        .sum();
    assert!(unsupported >= 6.0, "{unsupported}");
}

/// The deprecated `CancelQuery` (§4.5): `CancelFlightInfo`'s rules, `CANCEL_RESULT_CANCELLED`.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_query_releases_its_handle() {
    use arrow_flight::sql::{ActionCancelQueryRequest, ActionCancelQueryResult, Any};
    let mut h = start(FlightSqlConfig::default()).await;
    let (i, cookie) = info(&mut h, OPS, None, "SELECT 1").await.expect("info");
    let body = ActionCancelQueryRequest {
        info: i.encode_to_vec().into(),
    }
    .as_any()
    .encode_to_vec();
    let mut results = h
        .client
        .do_action(authed(
            Action::new("CancelQuery", body),
            OPS,
            cookie.as_deref(),
        ))
        .await
        .expect("cancel")
        .into_inner();
    let result = results.next().await.expect("one result").expect("ok");
    let answer: ActionCancelQueryResult = Any::decode(result.body)
        .expect("any")
        .unpack()
        .expect("unpack")
        .expect("a CancelQuery result");
    assert_eq!(answer.result, 1, "CANCEL_RESULT_CANCELLED");
    let e = fetch(&mut h, OPS, cookie.as_deref(), ticket(&i))
        .await
        .expect_err("released");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
}

/// A cancel whose ticket is not a statement ticket is `INVALID_HANDLE.FORMAT`, `INVALID_ARGUMENT`
/// (§6, review m7); another principal's ticket is `NOT_FOUND` like an unknown one (§4.5).
#[tokio::test(flavor = "multi_thread")]
async fn cancel_ticket_format_and_owner() {
    let mut h = start(FlightSqlConfig::default()).await;
    let bad = FlightInfo::new().with_endpoint(
        arrow_flight::FlightEndpoint::new().with_ticket(Ticket::new(b"garbage".to_vec())),
    );
    let e = h
        .client
        .do_action(authed(
            Action::new(
                "CancelFlightInfo",
                CancelFlightInfoRequest::new(bad).encode_to_vec(),
            ),
            OPS,
            None,
        ))
        .await
        .expect_err("malformed");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(
        e.metadata()
            .get("x-krishiv-error-class")
            .and_then(|v| v.to_str().ok()),
        Some("INVALID_HANDLE.FORMAT")
    );
    let (i, cookie) = info(&mut h, OPS, None, "SELECT 1").await.expect("info");
    let e = h
        .client
        .do_action(authed(
            Action::new(
                "CancelFlightInfo",
                CancelFlightInfoRequest::new(i.clone()).encode_to_vec(),
            ),
            OTHER,
            None,
        ))
        .await
        .expect_err("not theirs");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    fetch(&mut h, OPS, cookie.as_deref(), ticket(&i))
        .await
        .expect("the owner's ticket still works");
}

/// Another principal's cookie gets a new session and a new cookie, never the owner's (§3.2).
#[tokio::test(flavor = "multi_thread")]
async fn foreign_cookie_gets_new_session() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (_, mine) = info(&mut h, OPS, None, "SET spark.sql.shuffle.partitions = 3")
        .await
        .expect("info");
    let mine = mine.expect("cookie");
    let (_, theirs) = info(&mut h, OTHER, Some(&mine), "SELECT 1")
        .await
        .expect("a new session for the other principal");
    let theirs = theirs.expect("a new cookie");
    assert_ne!(theirs, mine);
}

/// A fetch longer than the handle TTL keeps its handle: the TTL runs from the end of the last
/// operation, never while one is live (§4.1).
#[tokio::test(flavor = "multi_thread")]
async fn long_doget_outlives_handle_ttl() {
    let mut h = start(FlightSqlConfig {
        handles: krishiv_flight_sql::HandleLimits {
            ttl: Duration::from_millis(100),
            ..krishiv_flight_sql::HandleLimits::default()
        },
        ..FlightSqlConfig::default()
    })
    .await;
    let (i, cookie) = info(&mut h, OPS, None, LONG).await.expect("info");
    let cookie = cookie.expect("cookie");
    let mut s = open_fetch(&mut h, Some(&cookie), ticket(&i)).await;
    h.flight.sweep_at(Instant::now() + Duration::from_secs(5));
    assert_eq!(h.flight.handle_count(), 1, "a live fetch keeps its handle");
    s.message().await.expect("reads on").expect("a piece");
    drop(s);
    wait_busy(&h, 0).await;
    h.flight.sweep_at(Instant::now() + Duration::from_secs(5));
    assert_eq!(h.flight.handle_count(), 0, "idle past its TTL: gone");
}

/// An idle ticket past the handle TTL is gone (§4.1).
#[tokio::test(flavor = "multi_thread")]
async fn ticket_expires() {
    let mut h = start(FlightSqlConfig {
        handles: krishiv_flight_sql::HandleLimits {
            ttl: Duration::from_millis(100),
            ..krishiv_flight_sql::HandleLimits::default()
        },
        ..FlightSqlConfig::default()
    })
    .await;
    let (i, cookie) = info(&mut h, OPS, None, "SELECT 1").await.expect("info");
    h.flight.sweep_at(Instant::now() + Duration::from_secs(5));
    let e = fetch(&mut h, OPS, cookie.as_deref(), ticket(&i))
        .await
        .expect_err("expired");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
}

/// While the server drains, a call with a stale cookie is `UNAVAILABLE` and creates no session
/// (§3.2, §2 "Drain").
#[tokio::test(flavor = "multi_thread")]
async fn reset_during_shutdown_is_unavailable() {
    let mut h = start(FlightSqlConfig::default()).await;
    h.draining.store(true, std::sync::atomic::Ordering::Release);
    let e = info(&mut h, OPS, Some("stale-session"), "SELECT 1")
        .await
        .expect_err("draining");
    assert_eq!(e.code(), Code::Unavailable, "{e:?}");
    assert_eq!(h.flight.session_count(), 0);
    assert_eq!(h.manager.counts().live, 0);
}

/// A listener whose shutdown grace is long enough that only the running fetch decides when
/// `serve` returns, a client with a 16 KiB stream window and pieces of 4 KiB: an unread fetch
/// stays open however small its result.
async fn start_graceful() -> Harness {
    start_with(
        FlightSqlConfig {
            encode: krishiv_flight_sql::EncodeLimits {
                batch_bytes: 16 << 10,
                data_bytes: 4 << 10,
            },
            ..FlightSqlConfig::default()
        },
        ConnectServerConfig {
            shutdown_timeout: Duration::from_secs(30),
            ..ConnectServerConfig::default()
        },
        Some(16 << 10),
        SessionLimits::default(),
    )
    .await
}

/// Starts `serve`'s shutdown and checks it is still waiting after a while: a live operation holds it.
async fn shut_down_and_see_it_wait(h: &mut Harness) {
    let _ = h.stop.take().expect("not stopped yet").send(());
    let deadline = Instant::now() + Duration::from_secs(10);
    while !h.draining.load(std::sync::atomic::Ordering::Acquire) {
        assert!(Instant::now() < deadline, "shutdown never started");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !h.served.is_finished(),
        "serve returned with a fetch running"
    );
}

/// A `DoGet` that has delivered its first message (the schema), kept for [`read_to_end`].
async fn open_fetch_keeping_schema(
    h: &mut Harness,
    cookie: &str,
    t: Ticket,
) -> (
    arrow_flight::FlightData,
    tonic::Streaming<arrow_flight::FlightData>,
) {
    let mut s = h
        .client
        .do_get(authed(t, OPS, Some(cookie)))
        .await
        .expect("do_get")
        .into_inner();
    let schema = s.message().await.expect("first message").expect("schema");
    (schema, s)
}

/// Reads a stream to its end after its first message: the rows, or the status it ended with.
async fn read_to_end(
    (schema, stream): (
        arrow_flight::FlightData,
        tonic::Streaming<arrow_flight::FlightData>,
    ),
) -> Result<Vec<RecordBatch>, Status> {
    let data = futures::stream::once(async { Ok(schema) })
        .chain(stream)
        .map_err(|s| FlightError::Tonic(Box::new(s)));
    FlightRecordBatchStream::new_from_flight_data(data)
        .try_collect()
        .await
        .map_err(|e| match e {
            FlightError::Tonic(s) => *s,
            other => Status::internal(other.to_string()),
        })
}

/// `serve` returns, and `close_all` (what `krishiv-cli` calls next) leaves no Flight session or
/// handle (§16 m7).
async fn served_then_closed(h: Harness) {
    tokio::time::timeout(Duration::from_secs(20), h.served)
        .await
        .expect("serve returned after the fetch")
        .expect("serve task")
        .expect("serve");
    assert_eq!(
        h.manager.counts().live,
        0,
        "the manager closed its sessions"
    );
    h.flight.close_all();
    assert_eq!(h.flight.session_count(), 0);
    assert_eq!(h.flight.handle_count(), 0);
}

/// A running query `DoGet` is a live operation: shutdown waits for it within
/// `krishiv.server.shutdownTimeout`, the reader gets every row, then the sessions are closed
/// (design §7 "Drain", §11).
#[tokio::test(flavor = "multi_thread")]
async fn shutdown_waits_for_running_doget_then_closes_sessions() {
    let mut h = start_graceful().await;
    let (i, cookie) = info(&mut h, OPS, None, "SELECT id FROM range(0, 200000)")
        .await
        .expect("info");
    let cookie = cookie.expect("cookie");
    let stream = open_fetch_keeping_schema(&mut h, &cookie, ticket(&i)).await;
    wait_busy(&h, 2).await;
    shut_down_and_see_it_wait(&mut h).await;
    let batches = read_to_end(stream).await.expect("the fetch finishes");
    assert_eq!(rows(&batches), 200_000);
    served_then_closed(h).await;
}

/// The `DoGet` of a command's stored rows is a live operation too (§4.1): shutdown waits for it
/// and its reader gets the stored rows, not `UNAVAILABLE`.
#[tokio::test(flavor = "multi_thread")]
async fn command_doget_waited_for_at_shutdown() {
    let mut h = start_graceful().await;
    // `SET -v` lists every key with its documentation: many 4 KiB pieces.
    let (i, cookie) = info(&mut h, OPS, None, "SET -v").await.expect("info");
    let cookie = cookie.expect("cookie");
    let stream = open_fetch_keeping_schema(&mut h, &cookie, ticket(&i)).await;
    wait_busy(&h, 1).await;
    shut_down_and_see_it_wait(&mut h).await;
    let batches = read_to_end(stream).await.expect("the stored rows");
    assert!(rows(&batches) > 100, "{} rows", rows(&batches));
    served_then_closed(h).await;
}

/// Stalled readers hold their share of `maxResponseBytes` (§16 M1): once two fetches hold the
/// whole budget, a third `DoGet` waits before it answers, and goes on when one of them ends.
#[tokio::test(flavor = "multi_thread")]
async fn stalled_readers_wait_at_the_response_budget() {
    // Each fetch reserves `batch_bytes + data_bytes` = 500 MiB of the default 1 GiB.
    let mut h = start(FlightSqlConfig {
        encode: krishiv_flight_sql::EncodeLimits {
            batch_bytes: 400 << 20,
            data_bytes: 100 << 20,
        },
        ..FlightSqlConfig::default()
    })
    .await;
    let cap = h.budget.cap();
    assert!((1000 << 20..1500 << 20).contains(&cap), "budget {cap}");
    let (i, cookie) = info(&mut h, OPS, None, LONG).await.expect("info");
    let cookie = cookie.expect("cookie");
    let a = open_fetch(&mut h, Some(&cookie), ticket(&i)).await;
    let b = open_fetch(&mut h, Some(&cookie), ticket(&i)).await;
    assert_eq!(h.budget.available(), cap - 2 * (500 << 20));
    let mut client = h.client.clone();
    let third = tokio::spawn({
        let request = authed(ticket(&i), OPS, Some(&cookie));
        async move { client.do_get(request).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !third.is_finished(),
        "a third fetch answered past the budget"
    );
    drop(a);
    let mut c = tokio::time::timeout(Duration::from_secs(10), third)
        .await
        .expect("the third fetch goes on once a reader ends")
        .expect("task")
        .expect("do_get")
        .into_inner();
    c.message().await.expect("first message").expect("schema");
    drop((b, c));
    wait_busy(&h, 0).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while h.budget.available() != cap {
        assert!(Instant::now() < deadline, "the budget was not given back");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Calls that read no session state (`GetSqlInfo`, `Handshake`) create none and set no cookie
/// (§3.2 "No session").
#[tokio::test(flavor = "multi_thread")]
async fn stateless_call_without_cookie_sets_no_cookie() {
    let mut h = start(FlightSqlConfig::default()).await;
    let cmd = CommandGetSqlInfo { info: vec![] };
    let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
    let response = h
        .client
        .get_flight_info(authed(descriptor, OPS, None))
        .await
        .expect("info");
    assert_eq!(set_cookie(response.metadata()), None);
    let hello = HandshakeRequest {
        protocol_version: 0,
        payload: Default::default(),
    };
    let response = h
        .client
        .handshake(authed(futures::stream::iter([hello]), OPS, None))
        .await
        .expect("handshake");
    assert_eq!(set_cookie(response.metadata()), None);
    assert_eq!(h.flight.session_count(), 0);
    assert_eq!(h.manager.counts().live, 0);
}

/// A `DoGet` runs in its handle's session whatever session its cookie names (two first calls
/// in parallel), and that cookie still promotes its own session (§3.2, v3 review M4, m10).
#[tokio::test(flavor = "multi_thread")]
async fn doget_with_other_sessions_cookie_uses_handle_session() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (a, cookie_a) = info(&mut h, OPS, None, "SELECT current_database() AS db, 1 AS n")
        .await
        .expect("info a");
    let (_, cookie_b) = info(&mut h, OPS, None, "USE default")
        .await
        .expect("info b");
    let (cookie_a, cookie_b) = (cookie_a.expect("cookie a"), cookie_b.expect("cookie b"));
    assert_ne!(cookie_a, cookie_b);
    assert_eq!(h.flight.session_count(), 2);
    let batches = fetch(&mut h, OPS, Some(&cookie_b), ticket(&a))
        .await
        .expect("served from the handle's session");
    assert_eq!(rows(&batches), 1);
    // b was promoted by its cookie: past `provisionalTtl` only a's provisional session goes
    // (it was fetched with a cookie, so rule 1 did not release it at the stream's end).
    wait_busy(&h, 0).await;
    h.flight.sweep_at(Instant::now() + Duration::from_secs(120));
    assert_eq!(h.flight.session_count(), 1);
    let (_, set) = info(&mut h, OPS, Some(&cookie_b), "SELECT 1")
        .await
        .expect("b is still a session");
    assert_eq!(set, None, "b's cookie was accepted, not reset");
    let (_, set) = info(&mut h, OPS, Some(&cookie_a), "SELECT 1")
        .await
        .expect("a reset");
    assert!(set.is_some_and(|id| id != cookie_a), "a was released");
}

/// Cookie-less `GetFlightInfo` calls hold a provisional session each until fetched or past
/// `provisionalTtl`, so a loop of them meets `maxSessionsPerPrincipal` (§3.2 "Caps", v2 review
/// m7) and the sweep gives the slots back.
#[tokio::test(flavor = "multi_thread")]
async fn cookieless_get_flight_info_loop_hits_session_cap() {
    let mut h = start_with(
        FlightSqlConfig::default(),
        ConnectServerConfig::default(),
        None,
        SessionLimits {
            max_sessions_per_principal: 3,
            ..SessionLimits::default()
        },
    )
    .await;
    for _ in 0..3 {
        info(&mut h, OPS, None, "SELECT 1")
            .await
            .expect("under the cap");
    }
    let e = info(&mut h, OPS, None, "SELECT 1")
        .await
        .expect_err("the fourth is over the cap");
    assert_eq!(e.code(), Code::ResourceExhausted, "{e:?}");
    assert_eq!(
        e.metadata()
            .get("x-krishiv-error-class")
            .and_then(|v| v.to_str().ok()),
        Some("KRISHIV_SESSION_LIMIT")
    );
    assert_eq!(
        h.flight.session_count(),
        3,
        "the refused call kept no session"
    );
    h.flight.sweep_at(Instant::now() + Duration::from_secs(120));
    wait_sessions(&h, 0).await;
    info(&mut h, OPS, None, "SELECT 1")
        .await
        .expect("the swept slots are free");
}

/// A Connect session's id is not a Flight session: Flight resets the cookie to a new session
/// and leaves the Connect session alone (§3.2 "Cross-protocol ids", v2 review m9, v3 review m3).
#[tokio::test(flavor = "multi_thread")]
async fn connect_session_cookie_resets_on_flight() {
    let mut h = start(FlightSqlConfig::default()).await;
    let mut credentials = krishiv_session::auth::Credentials::new(
        "127.0.0.1:1".parse().expect("addr"),
        krishiv_session::auth::Surface::Connect,
    );
    credentials.bearer = Some(OPS);
    let caller = h
        .auth
        .authenticate(&credentials)
        .await
        .expect("authenticated");
    let connect_id = "00000000-0000-4000-8000-000000000001";
    h.manager
        .get_or_create(&caller, connect_id, None, "ops")
        .expect("a Connect session");
    let (_, set) = info(&mut h, OPS, Some(connect_id), "SELECT 1")
        .await
        .expect("served in a new session");
    let set = set.expect("a new cookie");
    assert_ne!(set, connect_id);
    assert_eq!(h.flight.session_count(), 1);
    assert_eq!(
        h.manager.counts().live,
        2,
        "the Connect session is untouched"
    );
}

/// The idle clock runs only while the stream owes the reader a piece: a query slow to its
/// first batch is not an idle fetch, however many sweeps run meanwhile (§4.2).
#[tokio::test(flavor = "multi_thread")]
async fn slow_first_batch_is_not_idle() {
    let mut h = start(FlightSqlConfig {
        fetch_idle_timeout: Duration::from_millis(100),
        ..FlightSqlConfig::default()
    })
    .await;
    let (i, cookie) = info(
        &mut h,
        OPS,
        None,
        "SELECT count(*) AS n FROM range(0, 100000000) WHERE id % 7 = 3",
    )
    .await
    .expect("info");
    let cookie = cookie.expect("cookie");
    let flight = h.flight.clone();
    let sweeper = tokio::spawn(async move {
        loop {
            flight.sweep_at(Instant::now() + Duration::from_secs(1));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    let started = Instant::now();
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&i)).await;
    let took = started.elapsed();
    sweeper.abort();
    let batches = batches.expect("the slow query is not cancelled as idle");
    assert_eq!(rows(&batches), 1);
    assert!(
        took > Duration::from_millis(300),
        "the first batch came in {took:?}: not slow enough to test the idle clock"
    );
}
