// SPDX-License-Identifier: Apache-2.0
//! The Connect listener over TLS (P3-T002 design §3.4, §3.5): `ConnectServer::with_tls` with the
//! server's `ServerTls`; a bearer token over TLS, a client certificate that authenticates by
//! itself with the role its name is listed under, and a client in clear refused. Fixtures:
//! `crates/krishiv-session/tests/fixtures/tls/README.md`.
#![cfg(unix)]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hyper_util::rt::TokioIo;
use krishiv_common::config::Config;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, SessionLimits, SessionManager, role_sessions,
};
use krishiv_session::Session;
use krishiv_session::auth::{AuthConfig, ServerAuth, ServerTls, TlsConfig};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_rustls::TlsConnector;
use tokio_rustls::rustls::pki_types::pem::PemObject as _;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tonic::Code;
use tonic::transport::{Channel, Endpoint};

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const VIEWER: &str = "viewer-token-0123456789abcdefghijklmnopqrstuv";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../krishiv-session/tests/fixtures/tls/{name}"))
}

/// `src` copied into `dir` with mode 0600 (git keeps no modes).
fn private_copy(dir: &Path, src: &Path, name: &str) -> String {
    let path = dir.join(name);
    std::fs::copy(src, &path).expect("copy");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    path.to_str().expect("utf-8 path").to_owned()
}

/// A TLS server: a viewer token (`carol`), client certificates `optional` with the principal
/// from the CN, `etl-job` an operator.
async fn start() -> (SocketAddr, oneshot::Sender<()>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let token = dir.path().join("token");
    std::fs::write(&token, VIEWER).expect("write");
    let secret = format!("file:{}", private_copy(dir.path(), &token, "v"));
    let key = private_copy(dir.path(), &fixture("server.key"), "server.key");
    let cert = fixture("server.pem");
    let ca = format!("file:{}", fixture("ca.pem").display());
    let settings = [
        ("krishiv.auth.tokens.v.secret", secret.as_str()),
        ("krishiv.auth.tokens.v.principal", "carol"),
        ("krishiv.auth.tokens.v.role", "viewer"),
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
    let auth = Arc::new(ServerAuth::new(&auth_config).expect("auth"));
    let factory = role_sessions(Session::new().expect("session"), &Config::new()).expect("factory");
    let manager = Arc::new(SessionManager::new(factory, SessionLimits::default()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(manager, ConnectServerConfig::default())
        .with_auth(ConnectAuth::new(auth))
        .with_tls(&tls.server_config());
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    (addr, stop, dir)
}

/// A channel over TLS to `addr`, trusting the test CA and presenting the fixture `identity`.
async fn tls_channel(addr: SocketAddr, identity: Option<&str>) -> Channel {
    let mut roots = RootCertStore::empty();
    roots
        .add(CertificateDer::from_pem_file(fixture("ca.pem")).expect("ca"))
        .expect("add ca");
    let provider = Arc::new(tokio_rustls::rustls::crypto::aws_lc_rs::default_provider());
    let builder = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("versions")
        .with_root_certificates(roots);
    let mut config = match identity {
        None => builder.with_no_client_auth(),
        Some(name) => builder
            .with_client_auth_cert(
                vec![CertificateDer::from_pem_file(fixture(&format!("{name}.pem"))).expect("cert")],
                PrivateKeyDer::from_pem_file(fixture(&format!("{name}.key"))).expect("key"),
            )
            .expect("client auth"),
    };
    config.alpn_protocols = vec![b"h2".to_vec()];
    let connector = TlsConnector::from(Arc::new(config));
    Endpoint::from_static("http://localhost")
        .connect_with_connector(tower::service_fn(move |_| {
            let connector = connector.clone();
            async move {
                let tcp = TcpStream::connect(addr).await?;
                let name = ServerName::try_from("localhost").map_err(std::io::Error::other)?;
                let tls = connector.connect(name, tcp).await?;
                Ok::<_, std::io::Error>(TokioIo::new(tls))
            }
        }))
        .await
        .expect("connect")
}

async fn release(channel: &Channel, token: Option<&str>) -> Result<(), tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready()
        .await
        .map_err(|e| tonic::Status::unavailable(e.to_string()))?;
    let mut request = tonic::Request::new(pb::ReleaseSessionRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext::default()),
        ..Default::default()
    });
    if let Some(t) = token {
        request.metadata_mut().insert(
            "authorization",
            format!("Bearer {t}").parse().expect("ascii"),
        );
    }
    grpc.unary(
        request,
        "/spark.connect.SparkConnectService/ReleaseSession"
            .parse()
            .expect("path"),
        tonic_prost::ProstCodec::<pb::ReleaseSessionRequest, pb::ReleaseSessionResponse>::default(),
    )
    .await
    .map(|_| ())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_token_and_a_certificate_authenticate_over_tls() {
    let (addr, _stop, _dir) = start().await;
    let plain_tls = tls_channel(addr, None).await;
    release(&plain_tls, Some(VIEWER))
        .await
        .expect("token over TLS");
    let s = release(&plain_tls, None).await.expect_err("no credential");
    assert_eq!(s.code(), Code::Unauthenticated);
    // The certificate alone: `etl-job` holds the operator role.
    let with_cert = tls_channel(addr, Some("client")).await;
    release(&with_cert, None).await.expect("certificate");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_in_clear_gets_no_answer() {
    let (addr, _stop, _dir) = start().await;
    // h2 in clear on the TLS port: the handshake fails, so no RPC gets through.
    let failed = match Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
    {
        Err(_) => true,
        Ok(channel) => release(&channel, Some(VIEWER)).await.is_err(),
    };
    assert!(failed, "an RPC in clear reached a TLS listener");
}
