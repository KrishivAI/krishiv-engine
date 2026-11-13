// SPDX-License-Identifier: Apache-2.0
//! The admin listener with authentication configured (P3-T002 design §3.1, ADR 0046 §3, §3c):
//! `/metrics` and the UI over a real socket through `serve_capped`, guarded by a `SessionGuard`
//! over the server's `ServerAuth` with one token per role.
#![cfg(all(unix, feature = "session", feature = "metrics-prometheus"))]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use krishiv_exec::MetricsSnapshot;
use krishiv_server::admin::{HttpLimits, health_routes, serve_capped, with_limits};
use krishiv_server::guard::Guard;
use krishiv_server::internal::{Row, SystemTable, UiFuture, UiSource, router_with_ui_guard};
use krishiv_server::metrics::metrics_route_guarded;
use krishiv_server::session_guard::SessionGuard;
use krishiv_session::auth::{AuthConfig, ServerAuth};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const VIEWER: &str = "viewer-token-0123456789abcdef0123456789";
const OPERATOR: &str = "operator-token-0123456789abcdef012345678";

struct Empty;

impl UiSource for Empty {
    fn rows(&self, _: SystemTable) -> UiFuture<'_, Vec<Row>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn cancel<'a>(&'a self, _: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async { Ok(false) })
    }
}

fn secret(dir: &std::path::Path, name: &str, value: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, value).expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    format!("file:{}", path.display())
}

/// The admin router of `krishiv server` with auth on: a viewer and an operator token, and a
/// throttle that blocks a peer after two failures.
async fn start() -> SocketAddr {
    let dir = tempfile::tempdir().expect("tempdir");
    let (v, o) = (
        secret(dir.path(), "v", VIEWER),
        secret(dir.path(), "o", OPERATOR),
    );
    let settings = [
        ("krishiv.auth.tokens.prom.secret", v.as_str()),
        ("krishiv.auth.tokens.prom.principal", "prometheus"),
        ("krishiv.auth.tokens.prom.kind", "service"),
        ("krishiv.auth.tokens.prom.role", "viewer"),
        ("krishiv.auth.tokens.ops.secret", o.as_str()),
        ("krishiv.auth.tokens.ops.principal", "ann"),
        ("krishiv.auth.tokens.ops.role", "operator"),
        ("krishiv.auth.failureThrottle.maxFailures", "2"),
    ];
    let config = AuthConfig::from_settings(settings).expect("auth config");
    let auth = ServerAuth::new(&config).expect("server auth");
    let guard: Arc<dyn Guard> = Arc::new(SessionGuard::new(Arc::new(auth)));
    let metrics = metrics_route_guarded(
        true,
        Arc::new(MetricsSnapshot::default),
        Some(Arc::clone(&guard)),
    );
    let app = router_with_ui_guard(Arc::new(Empty), guard)
        .merge(health_routes(Arc::new(AtomicBool::new(true))))
        .merge(metrics);
    let limits = HttpLimits::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(serve_capped(
        listener,
        with_limits(app, limits),
        limits.max_connections,
        std::future::pending(),
    ));
    addr
}

/// The raw response to `method path` with `token` as a bearer header (if any).
async fn call(addr: SocketAddr, method: &str, path: &str, token: Option<&str>) -> String {
    let mut s = TcpStream::connect(addr).await.expect("connect");
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: x\r\n{auth}Content-Length: 0\r\nConnection: close\r\n\r\n"
    );
    s.write_all(request.as_bytes()).await.expect("write");
    let mut out = String::new();
    s.read_to_string(&mut out).await.expect("read");
    out
}

fn status(response: &str) -> &str {
    response.get(9..12).unwrap_or("")
}

#[tokio::test]
async fn each_role_reaches_what_the_matrix_gives_it() {
    let addr = start().await;
    let queries = "/api/internal/queries";
    let cancel = "/api/internal/queries/q1/cancel";
    // ADR 0046 §3c: a viewer scrapes /metrics, and sees no other principal's work.
    assert_eq!(
        status(&call(addr, "GET", "/metrics", Some(VIEWER)).await),
        "200"
    );
    assert_eq!(
        status(&call(addr, "GET", queries, Some(VIEWER)).await),
        "403"
    );
    assert_eq!(
        status(&call(addr, "POST", cancel, Some(VIEWER)).await),
        "403"
    );
    // An operator monitors and acts on others' running work.
    assert_eq!(
        status(&call(addr, "GET", "/metrics", Some(OPERATOR)).await),
        "200"
    );
    assert_eq!(
        status(&call(addr, "GET", queries, Some(OPERATOR)).await),
        "200"
    );
    assert_ne!(
        status(&call(addr, "POST", cancel, Some(OPERATOR)).await),
        "403"
    );
    // The configured token works in the UI's query-string form too (GET only).
    let page = format!("{queries}?token={OPERATOR}");
    assert_eq!(status(&call(addr, "GET", &page, None).await), "200");
    // The probes stay open.
    assert_eq!(status(&call(addr, "GET", "/healthz", None).await), "200");
    assert_eq!(status(&call(addr, "GET", "/readyz", None).await), "200");
}

#[tokio::test]
async fn failures_are_401_then_the_peer_is_throttled() {
    let addr = start().await;
    let missing = call(addr, "GET", "/metrics", None).await;
    assert_eq!(status(&missing), "401", "{missing}");
    assert!(missing.contains("www-authenticate: Bearer"), "{missing}");
    let wrong = call(addr, "GET", "/api/internal/queries", Some("not-a-token")).await;
    assert_eq!(status(&wrong), "401", "{wrong}");
    assert!(!wrong.contains("not-a-token"), "{wrong}");
    // A second invalid token reaches the throttle (maxFailures = 2): the peer is refused for a
    // while, even with a valid token.
    assert_eq!(
        status(&call(addr, "GET", "/metrics", Some("again")).await),
        "401"
    );
    assert_eq!(
        status(&call(addr, "GET", "/metrics", Some(VIEWER)).await),
        "503"
    );
    assert_eq!(status(&call(addr, "GET", "/healthz", None).await), "200");
}
