// SPDX-License-Identifier: Apache-2.0
//! The Connect crate's hooks for Flight SQL (P3-T101 note §2, H1-H3): a service added with
//! `ConnectServer::with_routes` shares the listener and `AuthLayer`; the layer labels each
//! request with the surface of its path, and every site that records a request (credentials and
//! the failure counter, the no-role refusal and its denial counter, act-as records, the
//! unauthenticated-service log, the foreign-id record, a session's own denials) names
//! `flight-sql` for a Flight request and `connect` for a Connect one. One test per binary: the
//! log subscriber is process-wide.
#![cfg(test)]

use std::convert::Infallible;
use std::fmt::Write as _;
use std::future::{Ready, ready};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use krishiv_common::config::Config;
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect_server::auth::{ACT_AS_GROUPS, ACT_AS_USER, request_surface};
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, SessionLimits, SessionManager, role_sessions,
};
use krishiv_session::auth::{
    AuthConfig, AuthFailure, Authenticated, Authenticator, Credentials, DeniedCounts, Method, Role,
    ServerAuth, Surface,
};
use krishiv_session::{BoxFuture, Privilege, Session, Transport};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::codegen::http::{self, HeaderMap, HeaderValue};
use tonic::server::NamedService;
use tonic::service::Routes;
use tonic::transport::Channel;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Keeps every `krishiv::audit` and `krishiv::auth` event as `target field=value ...`.
struct Capture;

struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        let _ = write!(self.0, "{}={value} ", field.name());
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let target = event.metadata().target();
        if target == "krishiv::audit" || target == "krishiv::auth" {
            let mut fields = Fields(format!("{target} "));
            event.record(&mut fields);
            RECORDS.lock().unwrap().push(fields.0);
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

fn take() -> Vec<String> {
    std::mem::take(&mut *RECORDS.lock().unwrap())
}

const CAROL: &str = "carol-token-0123456789abcdefghijklmnopqrstuvw";
const PORTAL: &str = "portal-token-0123456789abcdefghijklmnopqrstu";
const ALICE: &str = "alice-token-0123456789abcdefghijklmnopqrstuv";
const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";

fn secret(dir: &Path, name: &str, token: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, token).expect("write secret");
    // A secret file readable by group or others refuses startup on Unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    }
    format!("file:{}", path.display())
}

/// Carol (viewer), alice (viewer) and portal, a service that may act for alice.
fn server_auth(dir: &Path) -> Arc<ServerAuth> {
    let c = secret(dir, "c", CAROL);
    let p = secret(dir, "p", PORTAL);
    let a = secret(dir, "a", ALICE);
    let config = AuthConfig::from_settings([
        ("krishiv.auth.tokens.c.secret", c.as_str()),
        ("krishiv.auth.tokens.c.principal", "carol"),
        ("krishiv.auth.tokens.c.role", "viewer"),
        ("krishiv.auth.tokens.a.secret", a.as_str()),
        ("krishiv.auth.tokens.a.principal", "alice"),
        ("krishiv.auth.tokens.a.role", "viewer"),
        ("krishiv.auth.tokens.p.secret", p.as_str()),
        ("krishiv.auth.tokens.p.principal", "portal"),
        ("krishiv.auth.tokens.p.kind", "service"),
        ("krishiv.auth.tokens.p.role", "admin"),
        ("krishiv.auth.principals.portal.expires", "2099-01-01"),
        ("krishiv.auth.principals.portal.privileges", "impersonate"),
        ("krishiv.auth.principals.portal.impersonates", "alice"),
    ])
    .expect("auth config");
    Arc::new(ServerAuth::new(&config).expect("auth"))
}

/// A stand-in for the Flight service: it answers every RPC with `FAILED_PRECONDITION` whose
/// message is `<surface>|<caller>`, the surface and caller `AuthLayer` attached.
#[derive(Clone)]
struct Probe;

impl NamedService for Probe {
    const NAME: &'static str = "arrow.flight.protocol.FlightService";
}

impl tower::Service<http::Request<tonic::body::Body>> for Probe {
    type Response = http::Response<tonic::body::Body>;
    type Error = Infallible;
    type Future = Ready<Result<Self::Response, Infallible>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
        let surface = request
            .extensions()
            .get::<Surface>()
            .map_or("none", |s| s.as_str());
        let caller = request
            .extensions()
            .get::<Authenticated>()
            .map_or("none", |c| c.principal.name.as_str());
        ready(Ok(tonic::Status::failed_precondition(format!(
            "{surface}|{caller}"
        ))
        .into_http()))
    }
}

fn with_headers<T>(
    message: T,
    token: Option<&str>,
    extra: &[(&'static str, &str)],
) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    if let Some(token) = token {
        request.metadata_mut().insert(
            "authorization",
            format!("Bearer {token}").parse().expect("header"),
        );
    }
    for (name, value) in extra {
        request
            .metadata_mut()
            .insert(*name, value.parse().expect("header"));
    }
    request
}

/// One unary call of `path` (the body is a `ReleaseSessionRequest` for `SESSION`).
async fn call(
    channel: &Channel,
    path: &'static str,
    token: Option<&str>,
    extra: &[(&'static str, &str)],
) -> Result<pb::ReleaseSessionResponse, tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let body = pb::ReleaseSessionRequest {
        session_id: SESSION.to_owned(),
        ..Default::default()
    };
    grpc.unary(
        with_headers(body, token, extra),
        http::uri::PathAndQuery::from_static(path),
        tonic_prost::ProstCodec::<pb::ReleaseSessionRequest, pb::ReleaseSessionResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

const FLIGHT: &str = "/arrow.flight.protocol.FlightService/Handshake";
const CONNECT: &str = "/spark.connect.SparkConnectService/ReleaseSession";

/// An authenticator whose callers map to no role.
struct NoRole;

impl Authenticator for NoRole {
    fn authenticate<'a>(
        &'a self,
        _credentials: &'a Credentials<'a>,
    ) -> BoxFuture<'a, Result<Authenticated, AuthFailure>> {
        Box::pin(async {
            Ok(Authenticated::new(
                Principal::new("nobody", PrincipalKind::User),
                None,
                Method::Oidc,
            ))
        })
    }
}

fn peer() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 40_000))
}

fn single(records: &[String], what: &str) -> String {
    assert_eq!(records.len(), 1, "{what}: {records:?}");
    records[0].clone()
}

fn operator(name: &str) -> Authenticated {
    Authenticated::new(
        Principal::new(name, PrincipalKind::User),
        Some(Role::Operator),
        Method::Token,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_requests_are_labelled_flight_sql_at_every_site() {
    tracing::subscriber::set_global_default(Capture).expect("subscriber");
    let dir = tempfile::tempdir().expect("tempdir");
    let auth = server_auth(dir.path());

    // H1: a server with the stand-in Flight service beside Connect, behind the one layer.
    let denials = Arc::new(DeniedCounts::new());
    let factory = role_sessions(
        Session::new().expect("session"),
        &Config::new(),
        Arc::clone(&denials),
    )
    .expect("factory");
    let manager = Arc::new(SessionManager::new(factory, SessionLimits::default()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let connect_auth = ConnectAuth::new(Arc::clone(&auth) as Arc<dyn Authenticator>)
        .with_denials(Arc::clone(&denials))
        .with_impersonation(Arc::clone(&auth));
    let server = ConnectServer::new(Arc::clone(&manager), ConnectServerConfig::default())
        .with_auth(connect_auth)
        .with_routes(Routes::new(Probe));
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");

    // The Flight path reaches the added service with its surface and caller; Connect still
    // answers its own path.
    let s = call(&channel, FLIGHT, Some(CAROL), &[])
        .await
        .expect_err("probe");
    assert_eq!(s.code(), Code::FailedPrecondition, "{s:?}");
    assert_eq!(s.message(), "flight-sql|carol");
    call(&channel, CONNECT, Some(CAROL), &[])
        .await
        .expect("connect release");
    take();

    // H2, the credentials' surface: a wrong token on each path counts under its surface.
    let failed = |surface: Surface| -> u64 {
        auth.failure_counts()
            .iter()
            .filter(|c| c.surface == surface)
            .map(|c| c.count)
            .sum()
    };
    let s = call(&channel, FLIGHT, Some("wrong"), &[])
        .await
        .expect_err("wrong");
    assert_eq!(s.code(), Code::Unauthenticated);
    assert_eq!(
        (failed(Surface::FlightSql), failed(Surface::Connect)),
        (1, 0)
    );
    let s = call(&channel, CONNECT, Some("wrong"), &[])
        .await
        .expect_err("wrong");
    assert_eq!(s.code(), Code::Unauthenticated);
    assert_eq!(
        (failed(Surface::FlightSql), failed(Surface::Connect)),
        (1, 1)
    );
    take();

    // H2, act-as: allowed and refused impersonation records name the request's listener.
    let s = call(&channel, FLIGHT, Some(PORTAL), &[(ACT_AS_USER, "alice")])
        .await
        .expect_err("probe");
    assert_eq!(s.message(), "flight-sql|alice");
    let r = single(&take(), "allowed act-as");
    assert!(r.contains("listener=flight-sql "), "{r}");
    let s = call(&channel, FLIGHT, Some(PORTAL), &[(ACT_AS_GROUPS, "g")])
        .await
        .expect_err("groups without a user");
    assert_eq!(s.code(), Code::PermissionDenied);
    let r = single(&take(), "malformed act-as");
    assert!(r.contains("listener=flight-sql "), "{r}");
    let s = call(&channel, CONNECT, Some(PORTAL), &[(ACT_AS_GROUPS, "g")])
        .await
        .expect_err("groups without a user");
    assert_eq!(s.code(), Code::PermissionDenied);
    let r = single(&take(), "malformed act-as on Connect");
    assert!(r.contains("listener=connect "), "{r}");
    drop(stop);

    // H2, the no-role refusal: its audit record and denial counter.
    for surface in [Surface::FlightSql, Surface::Connect] {
        let counts = Arc::new(DeniedCounts::new());
        let no_role = ConnectAuth::new(Arc::new(NoRole)).with_denials(Arc::clone(&counts));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer any"));
        let s = no_role
            .authenticate(&headers, Some(peer()), None, surface)
            .await
            .expect_err("no role");
        assert_eq!(s.code(), Code::PermissionDenied);
        let r = single(&take(), "no role");
        assert!(r.contains(&format!("surface={} ", surface.as_str())), "{r}");
        let denied: Vec<_> = counts
            .counts()
            .into_iter()
            .filter(|c| c.count > 0)
            .map(|c| (c.surface, c.privilege, c.count))
            .collect();
        assert_eq!(denied, [(surface, Privilege::Browse, 1)]);

        // No peer address: refused, logged with the surface.
        no_role
            .authenticate(&headers, None, None, surface)
            .await
            .expect_err("no peer");
        let r = single(&take(), "no peer");
        assert!(r.contains(&format!("surface={} ", surface.as_str())), "{r}");
        assert!(
            r.contains("refused a request without a peer address"),
            "{r}"
        );

        // Act-as on a listener that decides none.
        let mut asking = HeaderMap::new();
        asking.insert(ACT_AS_USER, HeaderValue::from_static("alice"));
        ConnectAuth::off()
            .authenticate(&asking, Some(peer()), None, surface)
            .await
            .expect_err("no impersonation");
        let r = single(&take(), "no impersonation");
        assert!(
            r.contains(&format!("listener={} ", surface.as_str())),
            "{r}"
        );

        // `caller` without the layer's caller: the request's surface from its extension.
        let mut request = tonic::Request::new(());
        request.extensions_mut().insert(surface);
        assert_eq!(request_surface(&request), surface);
        no_role.caller(&request).expect_err("fails closed");
        let r = single(&take(), "no caller");
        assert!(r.contains(&format!("surface={} ", surface.as_str())), "{r}");
    }
    // Without the layer's extension the surface is Connect.
    assert_eq!(request_surface(&tonic::Request::new(())), Surface::Connect);

    // H3, the foreign-id record: `get_or_create_with` and `get_with` name their surface.
    let manager =
        SessionManager::from_base(Session::new().expect("session"), SessionLimits::default());
    manager
        .get_or_create(&operator("alice"), SESSION, None, "u")
        .expect("alice's");
    take();
    let bob = operator("bob");
    let e = manager
        .get_or_create_with(&bob, SESSION, None, "u", Transport::FlightSql)
        .expect_err("alice's id");
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_NOT_FOUND");
    let r = single(&take(), "foreign create");
    assert!(r.contains("surface=flight-sql "), "{r}");
    assert!(r.contains("refused an id another principal holds"), "{r}");
    // Another owner's id answers as an unknown one, never SESSION_CLOSED.
    let e = manager
        .get_with(&bob, SESSION, Surface::FlightSql)
        .expect_err("alice's id");
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_NOT_FOUND");
    let r = single(&take(), "foreign get");
    assert!(r.contains("surface=flight-sql "), "{r}");
    let e = manager.get(&bob, SESSION, None).expect_err("alice's id");
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_NOT_FOUND");
    let r = single(&take(), "foreign get on Connect");
    assert!(r.contains("surface=connect "), "{r}");

    // H3, the session's own denials: `role_sessions` counts under the transport's surface.
    let data = dir.path().join("data");
    std::fs::create_dir(&data).expect("data dir");
    let mut config = Config::new();
    config
        .set(
            "krishiv.io.local.allowedRoots",
            &data
                .canonicalize()
                .expect("canonical")
                .display()
                .to_string(),
        )
        .expect("allowed roots");
    let outside = dir.path().join("outside.parquet");
    for (transport, surface) in [
        (Transport::FlightSql, Surface::FlightSql),
        (Transport::Connect, Surface::Connect),
    ] {
        let counts = Arc::new(DeniedCounts::new());
        let factory = role_sessions(
            Session::new().expect("session"),
            &config,
            Arc::clone(&counts),
        )
        .expect("factory");
        let manager = SessionManager::new(factory, SessionLimits::default());
        let entry = manager
            .get_or_create_with(&operator("olga"), SESSION, None, "u", transport)
            .expect("session");
        let refused = entry
            .session()
            .sql(&format!("SELECT * FROM parquet.`{}`", outside.display()))
            .await;
        let e = match refused {
            Ok(frame) => frame.collect().await.expect_err("confined"),
            Err(e) => e,
        };
        assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        let denied: Vec<_> = counts
            .counts()
            .into_iter()
            .filter(|c| c.count > 0)
            .map(|c| (c.surface, c.privilege, c.count))
            .collect();
        assert_eq!(denied, [(surface, Privilege::Select, 1)]);
    }
}
