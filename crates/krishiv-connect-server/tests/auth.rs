// SPDX-License-Identifier: Apache-2.0
//! The authenticated Connect listener end to end over a raw tonic client (P3-T002, design
//! §3.1, §3.6 and §4; ADR 0046 §3b): every RPC authenticates, sessions are the caller's, and a
//! command's class is checked against the caller's role before it is refused as unsupported.
#![cfg(unix)]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::command::CommandType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, SessionLimits, SessionManager, role_sessions,
};
use krishiv_session::BoxFuture;
use krishiv_session::Privilege;
use krishiv_session::Session;
use krishiv_session::auth::{
    AuthConfig, AuthFailure, Authenticated, Authenticator, Credentials, Method, Role, ServerAuth,
    Surface,
};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const VIEWER: &str = "viewer-token-0123456789abcdefghijklmnopqrstuv";
const ADMIN: &str = "admin-token-0123456789abcdefghijklmnopqrstuvw";
const OPERATOR: &str = "operator-token-0123456789abcdefghijklmnopqrstu";

struct Running {
    manager: Arc<SessionManager>,
    /// The denial counter the sessions and the no-role refusal count into.
    denials: Arc<krishiv_session::auth::DeniedCounts>,
    channel: Channel,
    _stop: oneshot::Sender<()>,
    _dir: tempfile::TempDir,
}

fn secret(dir: &Path, name: &str, token: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(name);
    std::fs::write(&path, token).expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    format!("file:{}", path.display())
}

/// A server with three tokens, `carol` (viewer), `olga` (operator) and `dana` (admin), whose
/// sessions are built by the factory `krishiv server` uses (`role_sessions`), with local paths
/// confined to `<dir>/data`.
async fn start() -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let v = secret(dir.path(), "v", VIEWER);
    let o = secret(dir.path(), "o", OPERATOR);
    let a = secret(dir.path(), "a", ADMIN);
    let config = AuthConfig::from_settings([
        ("krishiv.auth.tokens.v.secret", v.as_str()),
        ("krishiv.auth.tokens.v.principal", "carol"),
        ("krishiv.auth.tokens.v.role", "viewer"),
        ("krishiv.auth.tokens.o.secret", o.as_str()),
        ("krishiv.auth.tokens.o.principal", "olga"),
        ("krishiv.auth.tokens.o.role", "operator"),
        ("krishiv.auth.tokens.a.secret", a.as_str()),
        ("krishiv.auth.tokens.a.principal", "dana"),
        ("krishiv.auth.tokens.a.role", "admin"),
    ])
    .expect("auth config");
    let auth = Arc::new(ServerAuth::new(&config).expect("auth"));
    start_with(auth, dir).await
}

async fn start_with(auth: Arc<dyn Authenticator>, dir: tempfile::TempDir) -> Running {
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
    let denials = Arc::new(krishiv_session::auth::DeniedCounts::new());
    let factory = role_sessions(
        Session::new().expect("session"),
        &config,
        Arc::clone(&denials),
    )
    .expect("factory");
    let manager = Arc::new(SessionManager::new(factory, SessionLimits::default()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(Arc::clone(&manager), ConnectServerConfig::default())
        .with_auth(ConnectAuth::new(auth).with_denials(Arc::clone(&denials)));
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    Running {
        manager,
        denials,
        channel,
        _stop: stop,
        _dir: dir,
    }
}

fn path(rpc: &str) -> tonic::codegen::http::uri::PathAndQuery {
    format!("/spark.connect.SparkConnectService/{rpc}")
        .parse()
        .expect("path")
}

fn with_token<T>(request: T, token: Option<&str>) -> tonic::Request<T> {
    let mut r = tonic::Request::new(request);
    if let Some(t) = token {
        r.metadata_mut().insert(
            "authorization",
            format!("Bearer {t}").parse().expect("ascii"),
        );
    }
    r
}

async fn unary<Req, Resp>(
    channel: &Channel,
    rpc: &str,
    request: Req,
    token: Option<&str>,
) -> Result<Resp, tonic::Status>
where
    Req: prost::Message + Send + Sync + 'static,
    Resp: prost::Message + Default + Send + Sync + 'static,
{
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        with_token(request, token),
        path(rpc),
        tonic_prost::ProstCodec::<Req, Resp>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// Every response of an `ExecutePlan`, or the status it ended with.
async fn execute(
    channel: &Channel,
    plan: pb::Plan,
    token: Option<&str>,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let request = pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext {
            // Never identity: the token decides who this is.
            user_id: "dana".to_owned(),
            ..Default::default()
        }),
        plan: Some(plan),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let mut s = grpc
        .server_streaming(
            with_token(request, token),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(r) = s.message().await? {
        out.push(r);
    }
    Ok(out)
}

fn range(end: i64) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Range(pb::Range {
                start: Some(0),
                end,
                step: 1,
                num_partitions: None,
            })),
        }))),
    }
}

fn command(c: CommandType) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(c),
        }))),
    }
}

#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ErrorInfo {
    #[prost(string, tag = "1")]
    reason: String,
    #[prost(string, tag = "2")]
    domain: String,
    #[prost(btree_map = "string, string", tag = "3")]
    metadata: BTreeMap<String, String>,
}

fn error_class(status: &tonic::Status) -> String {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    ErrorInfo::decode(rpc.details[0].value.as_slice())
        .expect("ErrorInfo")
        .metadata
        .get("errorClass")
        .cloned()
        .unwrap_or_default()
}

fn release() -> pb::ReleaseSessionRequest {
    pb::ReleaseSessionRequest {
        session_id: SESSION.to_owned(),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_rpc_needs_a_credential() {
    let run = start().await;
    // A session RPC, an implemented streaming RPC and an unimplemented one: all are
    // `UNAUTHENTICATED` without a token (Spark's message), so an unauthenticated client learns
    // nothing about the server.
    let s = unary::<_, pb::ReleaseSessionResponse>(&run.channel, "ReleaseSession", release(), None)
        .await
        .expect_err("no token");
    assert_eq!(s.code(), Code::Unauthenticated);
    assert_eq!(s.message(), "No authentication token provided");
    let s = execute(&run.channel, range(3), None)
        .await
        .expect_err("no token");
    assert_eq!(s.code(), Code::Unauthenticated);
    let s =
        unary::<_, pb::ConfigResponse>(&run.channel, "Config", pb::ConfigRequest::default(), None)
            .await
            .expect_err("no token");
    assert_eq!(s.code(), Code::Unauthenticated);
    let s = unary::<_, pb::ConfigResponse>(
        &run.channel,
        "Config",
        pb::ConfigRequest::default(),
        Some("not-a-configured-token"),
    )
    .await
    .expect_err("wrong token");
    assert_eq!(s.code(), Code::Unauthenticated);
    assert_eq!(s.message(), "Invalid authentication token");
    assert_eq!(run.manager.counts().live, 0, "nothing was created");
    // With a token, the same RPCs get past authentication.
    let s = unary::<_, pb::ConfigResponse>(
        &run.channel,
        "Config",
        pb::ConfigRequest::default(),
        Some(VIEWER),
    )
    .await
    .expect_err("not implemented yet");
    assert_eq!(s.code(), Code::Unimplemented);
    unary::<_, pb::ReleaseSessionResponse>(&run.channel, "ReleaseSession", release(), Some(VIEWER))
        .await
        .expect("an unknown session is a no-op");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sessions_belong_to_the_authenticated_principal() {
    let run = start().await;
    let responses = execute(&run.channel, range(3), Some(VIEWER))
        .await
        .expect("a viewer runs a plan that reads no table");
    assert!(!responses.is_empty());
    // The session is carol's, whatever `user_id` said; its principal is carol.
    let carol = run
        .manager
        .get_if_present(
            &Authenticated::new(
                Principal::new("carol", PrincipalKind::User),
                Some(Role::Viewer),
                Method::Token,
            ),
            SESSION,
        )
        .expect("carol's session");
    assert_eq!(carol.caller().principal.name, "carol");
    assert_eq!(carol.client_user_id(), "dana", "recorded, never trusted");
    // dana's token does not reach carol's session id: it is not found, as an unknown id is.
    let s = execute(&run.channel, range(3), Some(ADMIN))
        .await
        .expect_err("another principal's session id");
    assert_eq!(error_class(&s), "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert_eq!(run.manager.counts().live, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_is_checked_against_the_role_before_it_is_refused() {
    let run = start().await;
    let register = || command(CommandType::RegisterFunction(Box::default()));
    // Client code needs `Usage` on `System("udf")`: admin only until T014b (ADR 0046 N1).
    let s = execute(&run.channel, register(), Some(VIEWER))
        .await
        .expect_err("a viewer sends no client code");
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    // The admin passes the check and meets the refusal every command gets until P3-T012.
    release_as(&run, VIEWER).await;
    let s = execute(&run.channel, register(), Some(ADMIN))
        .await
        .expect_err("commands are not run yet");
    assert!(
        error_class(&s).starts_with("CONNECT_INVALID_PLAN"),
        "{}",
        error_class(&s)
    );
    // A global temporary view is a shared object: the `Ddl` class.
    release_as(&run, ADMIN).await;
    let global = command(CommandType::CreateDataframeView(Box::new(
        pb::CreateDataFrameViewCommand {
            name: "v".to_owned(),
            is_global: true,
            ..Default::default()
        },
    )));
    let s = execute(&run.channel, global, Some(VIEWER))
        .await
        .expect_err("a viewer creates no shared object");
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
}

async fn release_as(run: &Running, token: &str) {
    unary::<_, pb::ReleaseSessionResponse>(
        &run.channel,
        "ReleaseSession",
        pb::ReleaseSessionRequest {
            allow_reconnect: true,
            ..release()
        },
        Some(token),
    )
    .await
    .expect("release");
}

fn sql(query: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Sql(pb::Sql {
                query: query.to_owned(),
                ..Default::default()
            })),
        }))),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sessions_get_the_role_policy_and_the_path_confinement() {
    let run = start().await;
    // A viewer reads no rows, even of a file inside the allowed roots.
    let s = execute(&run.channel, sql("SELECT * FROM range(1)"), Some(VIEWER))
        .await
        .expect("a plan that reads no table needs no privilege");
    assert!(!s.is_empty());
    release_as(&run, VIEWER).await;
    // An operator may read rows, but not by a path outside `krishiv.io.local.allowedRoots`.
    let outside = run._dir.path().join("outside.parquet");
    let s = execute(
        &run.channel,
        sql(&format!("SELECT * FROM parquet.`{}`", outside.display())),
        Some(OPERATOR),
    )
    .await
    .expect_err("confined");
    assert_eq!(s.code(), Code::PermissionDenied);
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    // ADR 0046 §8: the session counted it once, under `connect` and the privilege asked for.
    assert_eq!(denied(&run), [(Surface::Connect, Privilege::Select, 1)]);
    // Inside the roots the same read gets past the policy (the file does not exist).
    let inside = run._dir.path().join("data").join("missing.parquet");
    let s = execute(
        &run.channel,
        sql(&format!("SELECT * FROM parquet.`{}`", inside.display())),
        Some(OPERATOR),
    )
    .await
    .expect_err("no such file");
    assert_ne!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
}

/// An authenticator whose callers map to no role (an OIDC subject in no role list, a
/// certificate principal in none).
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_caller_with_no_role_gets_no_session() {
    let run = start_with(Arc::new(NoRole), tempfile::tempdir().expect("tempdir")).await;
    // ADR 0046 §2: refused when the session would be created, before anything is built.
    let s = execute(&run.channel, range(3), Some("any"))
        .await
        .expect_err("no role");
    assert_eq!(s.code(), Code::PermissionDenied);
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    let s = unary::<_, pb::ReleaseSessionResponse>(
        &run.channel,
        "ReleaseSession",
        release(),
        Some("any"),
    )
    .await
    .expect_err("no role");
    assert_eq!(s.code(), Code::PermissionDenied);
    assert_eq!(run.manager.counts().live, 0);
    // Each refusal is one denial of `BROWSE` on `connect`.
    assert_eq!(denied(&run), [(Surface::Connect, Privilege::Browse, 2)]);
}

/// The non-zero denial counters.
fn denied(run: &Running) -> Vec<(Surface, Privilege, u64)> {
    run.denials
        .counts()
        .into_iter()
        .filter(|c| c.count > 0)
        .map(|c| (c.surface, c.privilege, c.count))
        .collect()
}

/// Sends `body` (a message that is no `ExecutePlanRequest`: field 1 as a varint where a
/// string is expected) as an `ExecutePlan`.
async fn undecodable(channel: &Channel, token: Option<&str>) -> tonic::Status {
    let body = prost_types::Duration {
        seconds: 5,
        nanos: 0,
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    match grpc
        .server_streaming(
            with_token(body, token),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<prost_types::Duration, pb::ExecutePlanResponse>::default(),
        )
        .await
    {
        Ok(mut s) => s.get_mut().message().await.expect_err("refused"),
        Err(status) => status,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn authentication_comes_before_the_body_is_decoded() {
    let run = start().await;
    // Without a credential the body is never decoded: the answer is about the credential.
    let s = undecodable(&run.channel, None).await;
    assert_eq!(s.code(), Code::Unauthenticated, "{s:?}");
    // With one, the same body reaches the decoder and fails there.
    let s = undecodable(&run.channel, Some(VIEWER)).await;
    assert_ne!(s.code(), Code::Unauthenticated, "{s:?}");
    assert_ne!(s.code(), Code::Ok);
}
