// SPDX-License-Identifier: Apache-2.0
//! The Connect listener end to end over a raw tonic client: session RPCs,
//! limits, refusals and graceful shutdown (design P3-T001 §5). `ExecutePlan`
//! is `tests/execute.rs`; the PySpark tests live in `krishiv-cli`.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
use krishiv_session::Session;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::transport::Channel;

const ID: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";

struct Running {
    manager: Arc<SessionManager>,
    channel: Channel,
    stop: Option<oneshot::Sender<()>>,
    done: tokio::task::JoinHandle<Result<(), krishiv_common::error::KrishivError>>,
    draining: Arc<std::sync::atomic::AtomicBool>,
}

async fn start(config: ConnectServerConfig, limits: SessionLimits) -> Running {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        limits,
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(Arc::clone(&manager), config);
    let draining = server.draining();
    let (stop, stopped) = oneshot::channel::<()>();
    let done = tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    Running {
        manager,
        channel,
        stop: Some(stop),
        done,
        draining,
    }
}

async fn unary<Req, Resp>(
    channel: &Channel,
    rpc: &'static str,
    request: Req,
) -> Result<Resp, tonic::Status>
where
    Req: prost::Message + Send + Sync + 'static,
    Resp: prost::Message + Default + Send + Sync + 'static,
{
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let path = format!("/spark.connect.SparkConnectService/{rpc}")
        .parse()
        .expect("path");
    grpc.unary(
        tonic::Request::new(request),
        path,
        tonic_prost::ProstCodec::<Req, Resp>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

fn release(session_id: &str, allow_reconnect: bool) -> pb::ReleaseSessionRequest {
    pb::ReleaseSessionRequest {
        session_id: session_id.to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "someone".to_owned(),
            ..Default::default()
        }),
        allow_reconnect,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_session_frees_state() {
    let run = start(ConnectServerConfig::default(), SessionLimits::default()).await;
    let local = Principal::local_user();
    let entry = run
        .manager
        .get_or_create(&local, ID, None, "someone")
        .expect("create");
    let server_side = entry.server_side_id().to_owned();
    let session = entry.session().clone();
    let memory = session.runtime().memory();
    let before = memory.reserved();
    session
        .sql("CREATE TEMPORARY VIEW v AS SELECT 1 AS a")
        .await
        .expect("view");
    let cached = session
        .sql("SELECT id FROM range(20000)")
        .await
        .expect("df")
        .cache();
    cached.collect().await.expect("cache");
    assert!(
        memory.reserved() > before,
        "the cached relation holds memory"
    );
    drop(cached);
    drop(session);
    drop(entry);

    let response: pb::ReleaseSessionResponse =
        unary(&run.channel, "ReleaseSession", release(ID, false))
            .await
            .expect("release");
    assert_eq!(response.session_id, ID);
    assert_eq!(response.server_side_session_id, server_side);
    assert_eq!(run.manager.counts().live, 0);
    assert_eq!(memory.reserved(), before, "the session's cache was freed");
    let e = run.manager.get(&local, ID, None).unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");

    // Releasing again (or an unknown id) answers with no server-side id.
    let response: pb::ReleaseSessionResponse =
        unary(&run.channel, "ReleaseSession", release(ID, false))
            .await
            .expect("release");
    assert_eq!(response.server_side_session_id, "");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_rpc_is_unimplemented() {
    let run = start(ConnectServerConfig::default(), SessionLimits::default()).await;
    // GetStatus only looks a session up: an unknown one is not created.
    let e = unary::<_, pb::GetStatusResponse>(
        &run.channel,
        "GetStatus",
        pb::GetStatusRequest {
            session_id: ID.to_owned(),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert_eq!(e.code(), Code::Internal);
    assert!(
        e.message().contains("INVALID_HANDLE.SESSION_NOT_FOUND"),
        "{e:?}"
    );
    assert_eq!(run.manager.counts().live, 0);
    let e = unary::<_, pb::ReleaseSessionResponse>(&run.channel, "NoSuchRpc", release(ID, false))
        .await
        .unwrap_err();
    assert_eq!(e.code(), Code::Unimplemented);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_message_over_the_inbound_limit_is_refused() {
    let run = start(
        ConnectServerConfig {
            max_inbound_message_size: 1024,
            ..ConnectServerConfig::default()
        },
        SessionLimits::default(),
    )
    .await;
    let big = "x".repeat(2048);
    let e = unary::<_, pb::ReleaseSessionResponse>(
        &run.channel,
        "ReleaseSession",
        release(&big, false),
    )
    .await
    .unwrap_err();
    // tonic refuses it from the length prefix, before buffering, with
    // OUT_OF_RANGE (grpc-java answers RESOURCE_EXHAUSTED): a recorded
    // difference, `docs/known-differences.md`.
    assert_eq!(e.code(), Code::OutOfRange);
    // One that fits passes on the same connection.
    unary::<_, pb::ReleaseSessionResponse>(&run.channel, "ReleaseSession", release(ID, false))
        .await
        .expect("fits");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_field_the_version_does_not_define_is_refused_by_the_codec() {
    let run = start(ConnectServerConfig::default(), SessionLimits::default()).await;
    // ReleaseSessionRequest with an unknown field 99 (a varint): the bounded
    // decoder refuses it before prost sees it.
    #[derive(Clone, PartialEq, prost::Message)]
    struct Future {
        #[prost(string, tag = "1")]
        session_id: String,
        #[prost(uint64, tag = "99")]
        unknown: u64,
    }
    let e = unary::<_, pb::ReleaseSessionResponse>(
        &run.channel,
        "ReleaseSession",
        Future {
            session_id: ID.to_owned(),
            unknown: 7,
        },
    )
    .await
    .unwrap_err();
    assert_eq!(e.code(), Code::Internal);
    assert_eq!(
        e.message(),
        "[KRISHIV_UNSUPPORTED_FEATURE] The feature field 99 of \
         spark.connect.ReleaseSessionRequest, which Spark Connect 4.3 does not define \
         (sent by a newer client?) is not supported by Krishiv yet. SQLSTATE: 0A000"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_cancels_a_stuck_query_at_the_timeout_and_closes_every_session() {
    let mut run = start(
        ConnectServerConfig {
            shutdown_timeout: Duration::from_millis(300),
            ..ConnectServerConfig::default()
        },
        SessionLimits::default(),
    )
    .await;
    let local = Principal::local_user();
    let entry = run
        .manager
        .get_or_create(&local, ID, None, "u")
        .expect("create");
    let session = entry.session().clone();
    // A query that runs far longer than the grace period.
    let df = session
        .sql("SELECT count(*) FROM range(0, 100000000000, 1, 4)")
        .await
        .expect("df");
    let query = tokio::spawn(async move { df.collect().await });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while session.running_queries() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "query never started"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let started = tokio::time::Instant::now();
    run.stop.take().expect("stop").send(()).expect("send");
    let served = tokio::time::timeout(Duration::from_secs(20), run.done)
        .await
        .expect("serve returns")
        .expect("join");
    served.expect("clean exit");
    assert!(started.elapsed() >= Duration::from_millis(300));
    assert!(run.draining.load(Ordering::Acquire));
    assert!(session.is_closed());
    let e = query.await.expect("join").unwrap_err();
    assert_eq!(e.error_class().name(), "OPERATION_CANCELED");
    assert_eq!(run.manager.counts().live, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_with_nothing_running_returns_at_once() {
    let mut run = start(ConnectServerConfig::default(), SessionLimits::default()).await;
    run.manager
        .get_or_create(&Principal::local_user(), ID, None, "u")
        .expect("create");
    run.stop.take().expect("stop").send(()).expect("send");
    // Well inside the 30 s grace period: no query to wait for.
    tokio::time::timeout(Duration::from_secs(10), run.done)
        .await
        .expect("serve returns")
        .expect("join")
        .expect("clean exit");
    assert_eq!(run.manager.counts().live, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_idle_sweeper_runs_on_the_maintenance_interval() {
    let run = start(
        ConnectServerConfig {
            maintenance_interval: Duration::from_millis(20),
            ..ConnectServerConfig::default()
        },
        SessionLimits {
            idle_timeout: Some(Duration::from_millis(50)),
            ..SessionLimits::default()
        },
    )
    .await;
    let entry = run
        .manager
        .get_or_create(&Principal::local_user(), ID, None, "u")
        .expect("create");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while run.manager.counts().live > 0 {
        assert!(tokio::time::Instant::now() < deadline, "never swept");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(entry.session().is_closed());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_with_a_running_query_is_not_swept() {
    let manager = SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits {
            idle_timeout: Some(Duration::from_millis(1)),
            ..SessionLimits::default()
        },
    );
    let entry = manager
        .get_or_create(&Principal::local_user(), ID, None, "u")
        .expect("create");
    let session = entry.session().clone();
    let df = session
        .sql("SELECT count(*) FROM range(0, 100000000000, 1, 4)")
        .await
        .expect("df");
    let query = tokio::spawn(async move { df.collect().await });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while session.running_queries() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "query never started"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(manager.sweep(), 0, "a running query keeps its session");
    session.interrupt_all();
    let _ = query.await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while session.running_queries() > 0 {
        assert!(tokio::time::Instant::now() < deadline, "query never ended");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(manager.sweep(), 1);
}

#[tokio::test]
async fn a_draining_service_refuses_new_rpcs_with_unavailable() {
    use krishiv_connect_server::service::SparkConnectService;
    use prost::Message;
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits::default(),
    ));
    let draining = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let service = krishiv_connect_server::ConnectService::new(manager, draining);
    let bytes = release(ID, false).encode_to_vec();
    let decoded =
        krishiv_connect::wire::decode_request::<pb::ReleaseSessionRequest>(&bytes).expect("decode");
    let e = service
        .release_session(tonic::Request::new(decoded))
        .await
        .unwrap_err();
    assert_eq!(e.code(), Code::Unavailable);
    assert_eq!(e.message(), "server shutting down");
}

#[test]
fn server_settings_come_from_the_config() {
    let mut c = krishiv_common::config::Config::new();
    let d = ConnectServerConfig::from_config(&c);
    assert_eq!(d.max_inbound_message_size, 128 << 20);
    assert_eq!(d.max_concurrent_streams, 100);
    assert_eq!(
        d.keepalive,
        Some((Duration::from_secs(60), Duration::from_secs(20)))
    );
    c.set("spark.connect.grpc.keepAlive.enabled", "false")
        .unwrap();
    c.set("spark.connect.grpc.maxInboundMessageSize", "1048576")
        .unwrap();
    c.set("krishiv.server.shutdownTimeout", "2s").unwrap();
    let s = ConnectServerConfig::from_config(&c);
    assert_eq!(s.keepalive, None);
    assert_eq!(s.max_inbound_message_size, 1 << 20);
    assert_eq!(s.shutdown_timeout, Duration::from_secs(2));
}

fn spill_entries(root: &std::path::Path) -> Vec<String> {
    std::fs::read_dir(root)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("krishiv-spill-"))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_then_dropping_the_runtime_leaves_no_spill_directory() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("spill");
    let runtime = krishiv_session::EngineRuntime::builder()
        .config("krishiv.spill.dir", root.to_str().expect("utf-8"))
        .config("krishiv.memory.limit", "4m")
        .build()
        .expect("runtime");
    let base = Session::builder()
        .runtime(Arc::clone(&runtime))
        .build()
        .expect("session");
    let manager = Arc::new(SessionManager::from_base(base, SessionLimits::default()));
    let entry = manager
        .get_or_create(&Principal::local_user(), ID, None, "u")
        .expect("create");
    // A sort larger than the pool spills, which makes the spill directory.
    let rows = entry
        .session()
        .sql("SELECT id FROM range(0, 1000000, 1, 2) ORDER BY id DESC")
        .await
        .expect("df")
        .collect()
        .await
        .expect("sorted");
    assert!(!rows.is_empty());
    assert!(
        !spill_entries(&root).is_empty(),
        "the query spilled, so the spill directory exists"
    );
    drop(entry);

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let server = ConnectServer::new(Arc::clone(&manager), ConnectServerConfig::default());
    server
        .serve(listener, async {})
        .await
        .expect("serve and shut down");
    assert_eq!(manager.counts().live, 0);
    // Design P3-T001 §3.4 step 4: the last handle on the runtime goes, and
    // with it the spill directory and its lock.
    // Aborted tasks (the sweeper) drop their handles when tokio next runs
    // them; `krishiv server` drops the whole tokio runtime instead.
    drop(manager);
    drop(runtime);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !spill_entries(&root).is_empty() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "spill directory left: {:?}",
            spill_entries(&root)
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_silent_socket_loses_its_slot_after_the_handshake_deadline() {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits::default(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(
        manager,
        ConnectServerConfig {
            max_connections: 1,
            handshake_timeout: Duration::from_millis(300),
            ..ConnectServerConfig::default()
        },
    );
    let (stop, stopped) = oneshot::channel::<()>();
    let done = tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    // Takes the only slot and sends nothing. `started` is taken first: the
    // server's deadline starts at accept, which is after it, so the lower
    // bound below holds however late this task runs.
    let started = tokio::time::Instant::now();
    let silent = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    let response: pb::ReleaseSessionResponse = tokio::time::timeout(
        Duration::from_secs(10),
        unary(&channel, "ReleaseSession", release(ID, false)),
    )
    .await
    .expect("served once the silent socket was dropped")
    .expect("release");
    assert_eq!(response.session_id, ID);
    assert!(
        started.elapsed() >= Duration::from_millis(250),
        "the second connection waited for the slot"
    );
    drop(silent);
    stop.send(()).expect("stop");
    done.await.expect("join").expect("serve");
}

#[test]
fn every_krishiv_key_sets_its_setting() {
    let mut c = krishiv_common::config::Config::new();
    for (k, v) in [
        ("krishiv.connect.maxInboundMessageSize", "2MB"),
        ("krishiv.connect.keepAlive.time", "7s"),
        ("krishiv.connect.keepAlive.timeout", "3s"),
        ("krishiv.connect.session.maintenanceInterval", "5s"),
        ("krishiv.connect.session.closedTombstones", "7"),
        ("krishiv.connect.maxConnections", "9"),
        ("krishiv.connect.handshakeTimeout", "4s"),
        ("krishiv.connect.http2.maxConcurrentStreams", "11"),
        ("krishiv.connect.http2.maxHeaderListBytes", "8KB"),
        ("krishiv.connect.http2.maxPendingAcceptResets", "13"),
    ] {
        c.set(k, v).unwrap_or_else(|e| panic!("{k}: {e}"));
    }
    let s = ConnectServerConfig::from_config(&c);
    assert_eq!(s.max_inbound_message_size, 2 << 20);
    assert_eq!(
        s.keepalive,
        Some((Duration::from_secs(7), Duration::from_secs(3)))
    );
    assert_eq!(s.maintenance_interval, Duration::from_secs(5));
    assert_eq!(s.max_connections, 9);
    assert_eq!(s.handshake_timeout, Duration::from_secs(4));
    assert_eq!(s.max_concurrent_streams, 11);
    assert_eq!(s.max_header_list_bytes, 8 << 10);
    assert_eq!(s.max_pending_accept_resets, 13);
    assert_eq!(SessionLimits::from_config(&c).tombstones, 7);
    c.set("krishiv.connect.keepAlive.enabled", "false").unwrap();
    assert_eq!(ConnectServerConfig::from_config(&c).keepalive, None);
}

fn decoded<T: krishiv_connect::wire::Request>() -> krishiv_connect::wire::Decoded<T> {
    krishiv_connect::wire::decode_request::<T>(&T::default().encode_to_vec()).expect("decode")
}

/// Every RPC answers `UNAVAILABLE` once the server drains.
#[tokio::test]
async fn every_rpc_is_unavailable_once_the_server_drains() {
    use krishiv_connect_server::service::SparkConnectService;
    use tonic::Request;
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits::default(),
    ));
    let draining = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let service =
        krishiv_connect_server::ConnectService::new(Arc::clone(&manager), Arc::clone(&draining));
    assert!(Arc::ptr_eq(service.manager(), &manager));
    // No artifact directory until an artifact is written.
    assert!(service.artifacts().root_path().is_none());
    {
        let (drain, code) = (true, Code::Unavailable);
        draining.store(drain, Ordering::Release);
        let results: Vec<(&str, tonic::Status)> = vec![
            (
                "CloneSession",
                service
                    .clone_session(Request::new(decoded()))
                    .await
                    .unwrap_err(),
            ),
            (
                "GetStatus",
                service
                    .get_status(Request::new(decoded()))
                    .await
                    .unwrap_err(),
            ),
        ];
        if drain {
            // The other RPCs refuse too while draining.
            for e in [
                service
                    .artifact_status(Request::new(decoded()))
                    .await
                    .unwrap_err(),
                service
                    .fetch_error_details(Request::new(decoded()))
                    .await
                    .unwrap_err(),
                service
                    .analyze_plan(Request::new(decoded()))
                    .await
                    .unwrap_err(),
                service.config(Request::new(decoded())).await.unwrap_err(),
                service
                    .interrupt(Request::new(decoded()))
                    .await
                    .unwrap_err(),
            ] {
                assert_eq!(e.code(), Code::Unavailable);
            }
            let e = service
                .execute_plan(Request::new(decoded()))
                .await
                .err()
                .expect("err");
            assert_eq!(e.code(), Code::Unavailable);
            let e = service
                .reattach_execute(Request::new(decoded()))
                .await
                .err()
                .expect("err");
            assert_eq!(e.code(), Code::Unavailable);
            let e = service
                .release_execute(Request::new(decoded()))
                .await
                .unwrap_err();
            assert_eq!(e.code(), Code::Unavailable);
        }
        for (rpc, status) in results {
            assert_eq!(status.code(), code, "{rpc}");
            assert_eq!(status.message(), "server shutting down", "{rpc}");
        }
    }
}

/// A connection cap past the semaphore's capacity is clamped, not a panic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connection_cap_past_the_semaphore_limit_is_clamped() {
    let run = start(
        ConnectServerConfig {
            max_connections: usize::MAX,
            ..ConnectServerConfig::default()
        },
        SessionLimits::default(),
    )
    .await;
    let response: pb::ReleaseSessionResponse =
        unary(&run.channel, "ReleaseSession", release(ID, false))
            .await
            .expect("release");
    assert_eq!(response.session_id, ID);
}
