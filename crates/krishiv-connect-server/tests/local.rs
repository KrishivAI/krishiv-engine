// SPDX-License-Identifier: Apache-2.0
//! The embedded Connect server (P3-T016): start and stop leave nothing behind, the listener is
//! loopback only, and a connection without the per-start token is refused.
#![cfg(target_os = "linux")]
#![expect(clippy::expect_used, reason = "integration tests")]

use std::net::{Ipv4Addr, SocketAddr, TcpListener};

use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect_server::LocalServer;
use krishiv_session::Session;
use tonic::Code;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d11";

fn threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("task dir")
        .count()
}

/// The thread count once it has not changed for 300 ms.
fn quiesced_threads() -> usize {
    let mut last = threads();
    let mut since = std::time::Instant::now();
    while since.elapsed() < std::time::Duration::from_millis(300) {
        std::thread::sleep(std::time::Duration::from_millis(20));
        let now = threads();
        if now != last {
            last = now;
            since = std::time::Instant::now();
        }
    }
    last
}

/// Waits (at most 10 s) for the thread count to fall to `expected`. `stop()` joins the server's
/// own thread; the engine's compute and I/O pools are released with `shutdown_background`, so
/// their threads leave within moments of it.
fn assert_threads_settle(expected: usize, what: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while threads() > expected && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(threads(), expected, "{what}");
}

async fn config_rpc(addr: SocketAddr, token: Option<&str>) -> Result<(), tonic::Status> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    let mut request = tonic::Request::new(pb::ConfigRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext::default()),
        operation: Some(pb::config_request::Operation {
            op_type: Some(pb::config_request::operation::OpType::GetAll(
                pb::config_request::GetAll::default(),
            )),
        }),
        ..Default::default()
    });
    if let Some(t) = token {
        request.metadata_mut().insert(
            "authorization",
            format!("Bearer {t}").parse().expect("ascii"),
        );
    }
    let mut grpc = tonic::client::Grpc::new(channel);
    grpc.ready().await.expect("ready");
    grpc.unary(
        request,
        "/spark.connect.SparkConnectService/Config"
            .parse()
            .expect("path"),
        tonic_prost::ProstCodec::<pb::ConfigRequest, pb::ConfigResponse>::default(),
    )
    .await
    .map(|_| ())
}

#[test]
fn start_stop_100_times_leaves_no_port_and_no_thread() {
    // One start first: lazily created process-wide threads (allocator, runtimes) exist after it.
    let mut warm = LocalServer::start(Session::new().expect("session")).expect("start");
    warm.stop().expect("stop");
    let before = quiesced_threads();
    for _ in 0..100 {
        let mut server = LocalServer::start(Session::new().expect("session")).expect("start");
        assert!(server.is_running());
        let addr = server.addr();
        server.stop().expect("stop");
        assert!(!server.is_running());
        server.stop().expect("a second stop is a no-op");
        // The port-in-use check, right after the stop: a sibling test of this binary may take
        // the same ephemeral port later, so a check after the loop could meet its listener.
        drop(TcpListener::bind(addr).unwrap_or_else(|e| panic!("{addr} is still bound: {e}")));
    }
    assert_threads_settle(before, "a server thread outlived stop()");
}

#[test]
fn dropping_a_server_stops_it() {
    // One start first: lazily created process-wide threads (allocator, runtimes, the Delta log
    // pool of a build with the `delta` feature) exist after it and are not the server's.
    let mut warm = LocalServer::start(Session::new().expect("session")).expect("start");
    warm.stop().expect("stop");
    let before = quiesced_threads();
    let addr = {
        let server = LocalServer::start(Session::new().expect("session")).expect("start");
        server.addr()
    };
    TcpListener::bind(addr).expect("the port is free after drop");
    assert_threads_settle(before, "a server thread outlived drop");
}

#[test]
fn binds_loopback_only_and_the_string_names_the_port_and_token() {
    let mut server = LocalServer::start(Session::new().expect("session")).expect("start");
    assert!(server.addr().ip().is_loopback());
    assert_eq!(server.addr().ip(), Ipv4Addr::LOCALHOST);
    let s = server.connection_string();
    assert_eq!(
        s,
        format!(
            "sc://localhost:{}/;token={}",
            server.addr().port(),
            server.token()
        )
    );
    assert_eq!(server.token().len(), 64);
    // Another start has another token and port.
    let mut other = LocalServer::start(Session::new().expect("session")).expect("start");
    assert_ne!(server.token(), other.token());
    assert_ne!(server.addr(), other.addr());
    other.stop().expect("stop");
    server.stop().expect("stop");
}

fn has_ipv6_loopback() -> bool {
    TcpListener::bind("[::1]:0").is_ok()
}

/// `localhost` may resolve to `::1` first: nobody else may be able to hold `[::1]:<port>` and
/// receive the token a client sends there.
#[test]
fn the_ipv6_loopback_of_the_port_is_ours_too() {
    if !has_ipv6_loopback() {
        eprintln!("SKIPPED: this host has no IPv6 loopback");
        return;
    }
    let mut server = LocalServer::start(Session::new().expect("session")).expect("start");
    let port = server.addr().port();
    assert!(
        TcpListener::bind(("::1", port)).is_err(),
        "[::1]:{port} is bindable by another process while the server runs"
    );
    assert!(server.addrs().iter().all(|a| a.ip().is_loopback()));
    assert!(server.addrs().iter().any(SocketAddr::is_ipv6));
    server.stop().expect("stop");
    TcpListener::bind(("::1", port)).expect("[::1] is free after stop");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_token_works_over_the_ipv6_loopback() {
    if !has_ipv6_loopback() {
        eprintln!("SKIPPED: this host has no IPv6 loopback");
        return;
    }
    let mut server = LocalServer::start(Session::new().expect("session")).expect("start");
    let v6 = server
        .addrs()
        .into_iter()
        .find(SocketAddr::is_ipv6)
        .expect("v6");
    let token = server.token().to_owned();
    let status = config_rpc(v6, None).await.expect_err("no token");
    assert_eq!(status.code(), Code::Unauthenticated);
    config_rpc(v6, Some(&token)).await.expect("token");
    tokio::task::spawn_blocking(move || server.stop().expect("stop"))
        .await
        .expect("join");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connection_without_the_token_is_refused() {
    let mut server = LocalServer::start(Session::new().expect("session")).expect("start");
    let addr = server.addr();
    let token = server.token().to_owned();
    let no_token = config_rpc(addr, None).await.expect_err("no token");
    assert_eq!(no_token.code(), Code::Unauthenticated);
    let wrong = config_rpc(addr, Some(&"0".repeat(64)))
        .await
        .expect_err("wrong token");
    assert_eq!(wrong.code(), Code::Unauthenticated);
    config_rpc(addr, Some(&token))
        .await
        .expect("the token is accepted");
    // The token of another start is not this server's.
    let mut other = LocalServer::start(Session::new().expect("session")).expect("start");
    let foreign = config_rpc(addr, Some(other.token()))
        .await
        .expect_err("foreign token");
    assert_eq!(foreign.code(), Code::Unauthenticated);
    other.stop().expect("stop");
    tokio::task::spawn_blocking(move || server.stop().expect("stop"))
        .await
        .expect("join");
}

/// Every local client has the peer address 127.0.0.1: wrong tokens from another process must not
/// lock the owner out (the default failure throttle would after 20).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_tokens_never_lock_the_owner_out() {
    let mut server = LocalServer::start(Session::new().expect("session")).expect("start");
    let addr = server.addr();
    let token = server.token().to_owned();
    for _ in 0..25 {
        let wrong = config_rpc(addr, Some(&"0".repeat(64)))
            .await
            .expect_err("wrong token");
        assert_eq!(wrong.code(), Code::Unauthenticated);
    }
    config_rpc(addr, Some(&token))
        .await
        .expect("the right token still works after 25 wrong ones");
    tokio::task::spawn_blocking(move || server.stop().expect("stop"))
        .await
        .expect("join");
}

/// A host that runs nothing: only its presence is under test.
#[derive(Debug)]
struct NoHost;

#[derive(Debug)]
struct NoRuntime;

impl krishiv_exec::python_eval::UdfRuntime for NoRuntime {
    fn prepare(
        &self,
        spec: &krishiv_exec::python_eval::UdfCallSpec,
    ) -> krishiv_exec::Result<std::sync::Arc<dyn krishiv_exec::python_eval::PreparedUdf>> {
        Err(krishiv_common::error::KrishivUdfNotAvailable {
            name: spec.name.clone(),
            version: spec.version.to_string(),
        }
        .into())
    }
}

impl krishiv_connect_server::SessionUdfRuntime for NoRuntime {
    fn register(&self, _: &str, _: u32, _: &krishiv_connect::translate::PythonUdf) {}
    fn release(&self) {}
}

impl krishiv_connect_server::UdfHost for NoHost {
    fn session_runtime(
        &self,
        _: &str,
        _: Option<&str>,
        _: bool,
        _: &str,
        _: krishiv_connect_server::RunnerConf,
    ) -> std::sync::Arc<dyn krishiv_connect_server::SessionUdfRuntime> {
        std::sync::Arc::new(NoRuntime)
    }
}

#[test]
fn a_local_server_starts_with_a_udf_host_and_stops() {
    let mut server = LocalServer::start_with_udf_host(
        Session::new().expect("session"),
        std::sync::Arc::new(NoHost),
    )
    .expect("start");
    assert!(server.is_running());
    server.stop().expect("stop");
    assert!(!server.is_running());
}
