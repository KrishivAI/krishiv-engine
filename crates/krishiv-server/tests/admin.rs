// SPDX-License-Identifier: Apache-2.0
//! `/healthz`, `/readyz` and the admin listener's limits over a real socket.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::routing::get;
use krishiv_server::admin::{HttpLimits, health_routes, serve_capped, with_limits};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn get_raw(addr: std::net::SocketAddr, path: &str) -> String {
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes())
        .await
        .expect("write");
    let mut out = String::new();
    s.read_to_string(&mut out).await.expect("read");
    out
}

async fn start(
    app: Router,
    limits: HttpLimits,
) -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(serve_capped(
        listener,
        with_limits(app, limits),
        limits.max_connections,
        async {
            let _ = stopped.await;
        },
    ));
    (addr, stop)
}

#[tokio::test]
async fn healthz_is_ok_and_readyz_follows_the_flag() {
    let ready = Arc::new(AtomicBool::new(false));
    let (addr, _stop) = start(health_routes(Arc::clone(&ready)), HttpLimits::default()).await;
    let h = get_raw(addr, "/healthz").await;
    assert!(h.starts_with("HTTP/1.1 200"), "{h}");
    assert!(h.ends_with(r#"{"status":"ok"}"#), "{h}");
    let r = get_raw(addr, "/readyz").await;
    assert!(r.starts_with("HTTP/1.1 503"), "{r}");
    assert!(r.ends_with(r#"{"status":"not_ready"}"#), "{r}");
    ready.store(true, Ordering::Release);
    let r = get_raw(addr, "/readyz").await;
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.ends_with(r#"{"status":"ready"}"#), "{r}");
}

#[tokio::test]
async fn a_slow_request_times_out_but_an_event_stream_does_not() {
    let slow = || async {
        tokio::time::sleep(Duration::from_millis(400)).await;
        "done"
    };
    let app = Router::new()
        .route("/slow", get(slow))
        .route("/api/internal/events", get(slow));
    let limits = HttpLimits {
        request_timeout: Duration::from_millis(100),
        ..HttpLimits::default()
    };
    let (addr, _stop) = start(app, limits).await;
    let r = get_raw(addr, "/slow").await;
    assert!(r.starts_with("HTTP/1.1 408"), "{r}");
    let r = get_raw(addr, "/api/internal/events").await;
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
}

#[tokio::test]
async fn connections_past_the_cap_wait_for_one_to_close() {
    let limits = HttpLimits {
        max_connections: 1,
        ..HttpLimits::default()
    };
    let (addr, _stop) = start(health_routes(Arc::new(AtomicBool::new(true))), limits).await;
    // One idle connection holds the only permit.
    let held = TcpStream::connect(addr).await.expect("connect");
    tokio::time::sleep(Duration::from_millis(50)).await;
    let second = tokio::time::timeout(Duration::from_millis(300), get_raw(addr, "/healthz")).await;
    assert!(second.is_err(), "served past the cap");
    drop(held);
    let third = tokio::time::timeout(Duration::from_secs(5), get_raw(addr, "/healthz"))
        .await
        .expect("served once the first closed");
    assert!(third.starts_with("HTTP/1.1 200"), "{third}");
}

#[tokio::test]
async fn the_api_router_serves_no_page() {
    let (addr, _stop) = start(krishiv_server::api_router(), HttpLimits::default()).await;
    let v = get_raw(addr, "/api/v1/version").await;
    assert!(v.starts_with("HTTP/1.1 200"), "{v}");
    for path in ["/", "/index.html", "/queries"] {
        let r = get_raw(addr, path).await;
        assert!(r.starts_with("HTTP/1.1 404"), "{path}: {r}");
        assert!(!r.contains("<html"), "{path}: {r}");
    }
}
