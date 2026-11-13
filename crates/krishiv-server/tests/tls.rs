// SPDX-License-Identifier: Apache-2.0
//! The admin listener over TLS (P3-T002 design §3.4, §3.5): `serve_capped_tls` with the server's
//! `ServerTls`, a bearer token over TLS, plain HTTP refused, and a client certificate that
//! authenticates by itself with the role its name is listed under. Fixtures:
//! `crates/krishiv-session/tests/fixtures/tls/README.md`.
#![cfg(all(
    unix,
    feature = "session",
    feature = "metrics-prometheus",
    feature = "tls"
))]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use krishiv_exec::MetricsSnapshot;
use krishiv_server::admin::{HttpLimits, health_routes, with_limits};
use krishiv_server::guard::Guard;
use krishiv_server::internal::{Row, SystemTable, UiFuture, UiSource, router_with_ui_guard};
use krishiv_server::metrics::metrics_route_guarded;
use krishiv_server::session_guard::SessionGuard;
use krishiv_server::tls::serve_capped_tls;
use krishiv_session::auth::{AuthConfig, ServerAuth, ServerTls, TlsConfig};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsConnector;
use tokio_rustls::rustls::pki_types::pem::PemObject as _;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use tokio_rustls::rustls::{ClientConfig, RootCertStore};

const VIEWER: &str = "viewer-token-0123456789abcdef0123456789";

struct Empty;

impl UiSource for Empty {
    fn rows(&self, _: SystemTable) -> UiFuture<'_, Vec<Row>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn cancel<'a>(&'a self, _: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async { Ok(false) })
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../krishiv-session/tests/fixtures/tls/{name}"))
}

/// `src` copied into `dir` with mode 0600 (git keeps no modes), as a path string.
fn private_copy(dir: &Path, src: &Path, name: &str) -> String {
    let path = dir.join(name);
    std::fs::copy(src, &path).expect("copy");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    path.to_str().expect("utf-8 path").to_owned()
}

/// The admin router of `krishiv server` over TLS: a viewer token, client certificates
/// `optional` with the principal from the CN, `etl-job` an operator.
async fn start() -> (SocketAddr, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let secret = dir.path().join("v");
    std::fs::write(&secret, VIEWER).expect("write secret");
    let secret = private_copy(dir.path(), &secret, "viewer");
    let cert = fixture("server.pem");
    let key = private_copy(dir.path(), &fixture("server.key"), "server.key");
    let ca = format!("file:{}", fixture("ca.pem").display());
    let secret_ref = format!("file:{secret}");
    let settings = [
        ("krishiv.auth.tokens.prom.secret", secret_ref.as_str()),
        ("krishiv.auth.tokens.prom.principal", "prometheus"),
        ("krishiv.auth.tokens.prom.role", "viewer"),
        ("krishiv.auth.clientCertificates", "optional"),
        ("krishiv.auth.clientCa", ca.as_str()),
        ("krishiv.auth.mtls.principalFrom", "cn"),
        ("krishiv.auth.roles.operator", "etl-job"),
        ("krishiv.server.tls.certFile", cert.to_str().expect("utf-8")),
        ("krishiv.server.tls.keyFile", key.as_str()),
    ];
    let auth_config = AuthConfig::from_settings(settings).expect("auth config");
    let tls_config = TlsConfig::from_settings(settings).expect("tls config");
    let tls = ServerTls::new(&tls_config, &auth_config.mtls)
        .expect("server tls")
        .expect("tls on");
    let auth = ServerAuth::new(&auth_config).expect("server auth");
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
    let config = tls.server_config();
    tokio::spawn(async move {
        serve_capped_tls(
            listener,
            with_limits(app, limits),
            limits.max_connections,
            &config,
            std::future::pending(),
        )
        .await
    });
    (addr, dir)
}

/// A client trusting the test CA, presenting the fixture `identity` if given.
fn client(identity: Option<&str>) -> TlsConnector {
    let mut roots = RootCertStore::empty();
    roots
        .add(CertificateDer::from_pem_file(fixture("ca.pem")).expect("ca"))
        .expect("add ca");
    let provider = Arc::new(tokio_rustls::rustls::crypto::aws_lc_rs::default_provider());
    let builder = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("versions")
        .with_root_certificates(roots);
    let config = match identity {
        None => builder.with_no_client_auth(),
        Some(name) => {
            let chain =
                vec![CertificateDer::from_pem_file(fixture(&format!("{name}.pem"))).expect("cert")];
            let key = PrivateKeyDer::from_pem_file(fixture(&format!("{name}.key"))).expect("key");
            builder
                .with_client_auth_cert(chain, key)
                .expect("client auth")
        }
    };
    TlsConnector::from(Arc::new(config))
}

async fn exchange<S: AsyncRead + AsyncWrite + Unpin>(
    mut s: S,
    path: &str,
    token: Option<&str>,
) -> String {
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    let request = format!("GET {path} HTTP/1.1\r\nHost: x\r\n{auth}Connection: close\r\n\r\n");
    s.write_all(request.as_bytes()).await.expect("write");
    let mut out = Vec::new();
    // A server that closes without `close_notify` ends the read with an error; keep what came.
    let _ = s.read_to_end(&mut out).await;
    String::from_utf8_lossy(&out).into_owned()
}

/// The response to `GET path` over TLS, presenting `identity` and `token`.
async fn get(addr: SocketAddr, path: &str, identity: Option<&str>, token: Option<&str>) -> String {
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let name = ServerName::try_from("localhost").expect("name");
    let tls = client(identity)
        .connect(name, tcp)
        .await
        .expect("handshake");
    exchange(tls, path, token).await
}

fn status(response: &str) -> &str {
    response.get(9..12).unwrap_or("")
}

#[tokio::test]
async fn a_bearer_token_works_over_tls_and_plain_http_gets_nothing() {
    let (addr, _dir) = start().await;
    assert_eq!(status(&get(addr, "/healthz", None, None).await), "200");
    assert_eq!(
        status(&get(addr, "/metrics", None, Some(VIEWER)).await),
        "200"
    );
    assert_eq!(status(&get(addr, "/metrics", None, None).await), "401");
    // HTTP in clear on the TLS port: the handshake fails and no HTTP answer comes back.
    let plain = TcpStream::connect(addr).await.expect("connect");
    let answer = exchange(plain, "/healthz", None).await;
    assert!(!answer.starts_with("HTTP/"), "{answer:?}");
}

#[tokio::test]
async fn a_client_certificate_authenticates_with_its_listed_role() {
    let (addr, _dir) = start().await;
    let queries = "/api/internal/queries";
    // `etl-job` (the CN) is an operator: it monitors other principals' queries, no token.
    assert_eq!(
        status(&get(addr, queries, Some("client"), None).await),
        "200"
    );
    // The viewer token alone does not reach them (403), and nothing at all is 401.
    assert_eq!(status(&get(addr, queries, None, Some(VIEWER)).await), "403");
    assert_eq!(status(&get(addr, queries, None, None).await), "401");
    // `reporting` holds no role: authenticated, but `/metrics` needs one.
    assert_eq!(
        status(&get(addr, "/metrics", Some("nosan"), None).await),
        "403"
    );
}
