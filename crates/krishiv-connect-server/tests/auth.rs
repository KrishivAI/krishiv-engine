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
    .expect_err("an empty session id");
    assert_ne!(s.code(), Code::Unauthenticated, "{s:?}");
    assert_ne!(s.code(), Code::PermissionDenied, "{s:?}");
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

/// ADR 0046 §5 (threat model T-R3): the client's `UserContext.user_id` is never the identity,
/// but its session's `system.sessions` row records it as the client reported it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_clients_user_id_is_recorded_not_trusted() {
    use arrow_array::Array as _;
    let run = start().await;
    // The helper's requests say `user_id = "dana"`; the token is carol's (a viewer, who sees
    // her own rows of `system.sessions`).
    let responses = execute(
        &run.channel,
        sql("SELECT principal, client_user_id, transport FROM system.sessions"),
        Some(VIEWER),
    )
    .await
    .expect("a viewer reads her own sessions");
    let rows: Vec<(String, String, String)> = responses
        .iter()
        .filter_map(|r| match &r.response_type {
            Some(pb::execute_plan_response::ResponseType::ArrowBatch(b)) => Some(b),
            _ => None,
        })
        .flat_map(|b| {
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data.clone()), None)
                .expect("ipc")
                .collect::<Result<Vec<_>, _>>()
                .expect("batches")
        })
        .flat_map(|batch| {
            let text = |i: usize| {
                batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .expect("string column")
                    .clone()
            };
            let (p, u, t) = (text(0), text(1), text(2));
            (0..batch.num_rows())
                .map(|r| {
                    let get = |a: &arrow_array::StringArray| {
                        if a.is_null(r) {
                            String::new()
                        } else {
                            a.value(r).to_owned()
                        }
                    };
                    (get(&p), get(&u), get(&t))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        rows,
        [("carol".to_owned(), "dana".to_owned(), "connect".to_owned())]
    );
}

/// The `krishiv::audit` events of this thread, as `field=value` lines.
#[derive(Clone, Default)]
struct AuditCapture(Arc<std::sync::Mutex<Vec<String>>>);

impl AuditCapture {
    fn take(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

struct AuditFields(String);

impl tracing::field::Visit for AuditFields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write as _;
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        use std::fmt::Write as _;
        let _ = write!(self.0, "{}={value} ", field.name());
    }
}

impl tracing::Subscriber for AuditCapture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        if event.metadata().target() == "krishiv::audit" {
            let mut fields = AuditFields(String::new());
            event.record(&mut fields);
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(fields.0);
        }
    }

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

/// Review N2 (ADR 0046 §3b): `ReattachExecute` of another principal's operation id, end to
/// end. It is answered as an unknown id and writes one audit record naming the caller only.
/// A current-thread runtime, so the handler runs on this thread and the thread's subscriber
/// sees its record.
#[tokio::test(flavor = "current_thread")]
async fn reattaching_another_principals_operation_is_unknown_and_recorded() {
    use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
    const OP: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";
    const UNKNOWN: &str = "5b1f7a2e-0c3d-4e8f-9a6b-2d4c6e8f0a1b";
    const OTHER_SESSION: &str = "2c4e6a8b-0d1f-4a3c-8e5b-7d9f1b3d5f70";
    let capture = AuditCapture::default();
    let _guard = tracing::subscriber::set_default(capture.clone());
    let run = start().await;

    // carol's reattachable operation, read to its end: it stays registered until released.
    let request = pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        operation_id: Some(OP.to_owned()),
        plan: Some(range(3)),
        request_options: vec![pb::execute_plan_request::RequestOption {
            request_option: Some(RequestOption::ReattachOptions(pb::ReattachOptions {
                reattachable: true,
            })),
        }],
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let mut stream = grpc
        .server_streaming(
            with_token(request, Some(VIEWER)),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await
        .expect("execute")
        .into_inner();
    while stream.message().await.expect("response").is_some() {}
    assert!(
        capture.take().is_empty(),
        "carol's own operation writes nothing"
    );

    let reattach = |operation_id: &str| pb::ReattachExecuteRequest {
        session_id: OTHER_SESSION.to_owned(),
        operation_id: operation_id.to_owned(),
        ..Default::default()
    };
    let reattach_as_dana = async |operation_id: &str| {
        let mut grpc = tonic::client::Grpc::new(run.channel.clone());
        grpc.ready().await.expect("ready");
        grpc.server_streaming(
            with_token(reattach(operation_id), Some(ADMIN)),
            path("ReattachExecute"),
            tonic_prost::ProstCodec::<pb::ReattachExecuteRequest, pb::ExecutePlanResponse>::default(
            ),
        )
        .await
        .expect_err("not dana's operation")
    };

    // Dana (an admin) creates her own session first, so the reattach finds it.
    execute_in(&run.channel, OTHER_SESSION, ADMIN).await;
    capture.take();

    let unknown = reattach_as_dana(UNKNOWN).await;
    assert_eq!(error_class(&unknown), "INVALID_HANDLE.OPERATION_NOT_FOUND");
    assert!(capture.take().is_empty(), "an unknown id writes nothing");

    let foreign = reattach_as_dana(OP).await;
    assert_eq!(
        error_class(&foreign),
        "INVALID_HANDLE.OPERATION_NOT_FOUND",
        "answered as the unknown id is"
    );
    let records = capture.take();
    assert_eq!(records.len(), 1, "{records:?}");
    let r = &records[0];
    assert!(r.contains("principal=dana "), "{r}");
    assert!(r.contains("handle=operation "), "{r}");
    assert!(r.contains("surface=connect "), "{r}");
    assert!(!r.contains("carol"), "the owner is never named: {r}");
}

/// Runs a small plan in `session_id` as the holder of `token`, to create that session.
async fn execute_in(channel: &Channel, session_id: &str, token: &str) {
    let request = pb::ExecutePlanRequest {
        session_id: session_id.to_owned(),
        plan: Some(range(1)),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let mut s = grpc
        .server_streaming(
            with_token(request, Some(token)),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await
        .expect("execute")
        .into_inner();
    while s.message().await.expect("response").is_some() {}
}

/// A `Config` request of `op` in [`SESSION`] as `token`.
async fn config_as(
    run: &Running,
    token: &str,
    op: krishiv_connect::proto::v4_3::config_request::operation::OpType,
) -> Result<pb::ConfigResponse, tonic::Status> {
    unary(
        &run.channel,
        "Config",
        pb::ConfigRequest {
            session_id: SESSION.to_owned(),
            operation: Some(pb::config_request::Operation { op_type: Some(op) }),
            ..Default::default()
        },
        Some(token),
    )
    .await
}

/// `Config` set and unset of an admin-only key or a server-scoped store key ask the session's
/// policy as SQL `SET` does: a refusal is `PERMISSION_DENIED`, audited and counted under
/// `connect` and `MANAGE` (ADR 0046 §8), a silent set turns it into a warning (still counted)
/// and changes nothing, an admin-only key named by its Spark alias is refused the same way, and
/// an admin may.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_refusals_are_counted_like_sql_set() {
    use krishiv_connect::proto::v4_3::config_request::{self as creq, operation::OpType as Op};
    let kv = |key: &str, value: &str| pb::KeyValue {
        key: key.to_owned(),
        value: Some(value.to_owned()),
    };
    let run = start().await;
    let set = |silent| {
        Op::Set(creq::Set {
            pairs: vec![kv("krishiv.resultCache.ttl", "1m")],
            silent: Some(silent),
        })
    };
    let s = config_as(&run, OPERATOR, set(false))
        .await
        .expect_err("admin key");
    assert_eq!(s.code(), Code::PermissionDenied);
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    assert_eq!(denied(&run), [(Surface::Connect, Privilege::Manage, 1)]);

    let quiet = config_as(&run, OPERATOR, set(true)).await.expect("silent");
    assert!(
        quiet
            .warnings
            .iter()
            .any(|w| w.contains("krishiv.resultCache.ttl")),
        "{:?}",
        quiet.warnings
    );
    let s = config_as(
        &run,
        OPERATOR,
        Op::Unset(creq::Unset {
            keys: vec!["krishiv.resultCache.ttl".to_owned()],
        }),
    )
    .await
    .expect_err("admin key");
    assert_eq!(s.code(), Code::PermissionDenied);
    let s = config_as(
        &run,
        OPERATOR,
        Op::Set(creq::Set {
            pairs: vec![kv("fs.s3a.endpoint", "http://169.254.169.254")],
            silent: None,
        }),
    )
    .await
    .expect_err("server-scoped store key");
    assert_eq!(s.code(), Code::PermissionDenied);
    // `krishiv.sql.autoBroadcastJoinThreshold` (admin) by its Spark name.
    let s = config_as(
        &run,
        OPERATOR,
        Op::Set(creq::Set {
            pairs: vec![kv("spark.sql.autoBroadcastJoinThreshold", "1")],
            silent: None,
        }),
    )
    .await
    .expect_err("admin key by its Spark alias");
    assert_eq!(s.code(), Code::PermissionDenied);
    assert_eq!(denied(&run), [(Surface::Connect, Privilege::Manage, 5)]);
    let got = config_as(
        &run,
        OPERATOR,
        Op::GetOption(creq::GetOption {
            keys: vec!["krishiv.resultCache.ttl".to_owned()],
        }),
    )
    .await
    .expect("get");
    assert_ne!(
        got.pairs[0].value.as_deref(),
        Some("1min"),
        "a refused set changes nothing"
    );
    release_as(&run, OPERATOR).await;

    config_as(&run, ADMIN, set(false)).await.expect("admin may");
    let got = config_as(
        &run,
        ADMIN,
        Op::GetOption(creq::GetOption {
            keys: vec!["krishiv.resultCache.ttl".to_owned()],
        }),
    )
    .await
    .expect("get");
    assert_eq!(
        got.pairs[0].value.as_deref(),
        Some("1min"),
        "the admin's set holds (1m, read back normalized)"
    );
    assert_eq!(denied(&run), [(Surface::Connect, Privilege::Manage, 5)]);
}

/// A silent set that fails never sends the value back when the redaction rules match it: an
/// operator's refused store endpoint with URL credentials comes back as
/// `Some(*********(redacted))` (a difference toward safety: Spark echoes the value).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_silent_set_warning_hides_a_secret_value() {
    use krishiv_connect::proto::v4_3::config_request::{self as creq, operation::OpType as Op};
    let run = start().await;
    let quiet = config_as(
        &run,
        OPERATOR,
        Op::Set(creq::Set {
            pairs: vec![pb::KeyValue {
                key: "fs.s3a.endpoint".to_owned(),
                value: Some("http://u:canary-pw-7f3@minio.local:9000".to_owned()),
            }],
            silent: Some(true),
        }),
    )
    .await
    .expect("silent");
    assert_eq!(quiet.warnings.len(), 1, "{:?}", quiet.warnings);
    let w = &quiet.warnings[0];
    assert!(
        w.starts_with("Failed to set fs.s3a.endpoint to Some(*********(redacted)) due to "),
        "{w}"
    );
    assert!(!w.contains("canary-pw-7f3"), "{w}");
    assert!(w.contains("KRISHIV_INSUFFICIENT_PERMISSIONS"), "{w}");
}

/// A non-silent `Config.set` that fails returns the error without the value when the pair is
/// redacted (a credential in a URL here), as `RuntimeConf::set` hides it; the class stays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_set_error_hides_a_secret_value() {
    use krishiv_connect::proto::v4_3::config_request::{self as creq, operation::OpType as Op};
    let run = start().await;
    let e = config_as(
        &run,
        ADMIN,
        Op::Set(creq::Set {
            pairs: vec![pb::KeyValue {
                key: "spark.sql.session.timeZone".to_owned(),
                value: Some("postgres://u:canary-pw-8e1@db.local/x".to_owned()),
            }],
            silent: Some(false),
        }),
    )
    .await
    .expect_err("not a zone");
    let shown = format!("{e:?}");
    assert!(!shown.contains("canary-pw-8e1"), "{shown}");
    assert!(shown.contains("INVALID_CONF_VALUE.TIME_ZONE"), "{shown}");
    assert!(shown.contains("*********(redacted)"), "{shown}");
}

// ---------------------------------------------------- cache rows (P3-T083, ADR 0046 §3b)

fn rel_range(end: i64) -> pb::Relation {
    match range(end).op_type {
        Some(OpType::Root(r)) => *r,
        _ => unreachable!("range is a root relation"),
    }
}

async fn analyze_with(
    run: &Running,
    analyze: pb::analyze_plan_request::Analyze,
    token: &str,
) -> Result<pb::AnalyzePlanResponse, tonic::Status> {
    unary(
        &run.channel,
        "AnalyzePlan",
        pb::AnalyzePlanRequest {
            session_id: SESSION.to_owned(),
            analyze: Some(analyze),
            ..Default::default()
        },
        Some(token),
    )
    .await
}

fn checkpoint(eager: bool, local: bool) -> pb::Plan {
    command(CommandType::CheckpointCommand(Box::new(
        pb::CheckpointCommand {
            relation: Some(rel_range(3)),
            local,
            eager,
            storage_level: None,
        },
    )))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_cache_rows_need_the_operator_floor_except_the_reads() {
    use pb::analyze_plan_request::{self as areq, Analyze};
    // A session id belongs to its caller: the viewer and the operator get a server each.
    let run = start().await;
    let op = start().await;
    for (r, token) in [(&run, VIEWER), (&op, OPERATOR)] {
        // `AnalyzePlan` needs the caller's session: each caller runs a plan first.
        execute(&r.channel, range(1), Some(token))
            .await
            .expect("opens the session");
    }
    let persist = || {
        Analyze::Persist(Box::new(areq::Persist {
            relation: Some(rel_range(3)),
            storage_level: None,
        }))
    };
    let unpersist = || {
        Analyze::Unpersist(Box::new(areq::Unpersist {
            relation: Some(rel_range(3)),
            blocking: None,
        }))
    };
    // A viewer: refused with 42501, and nothing is registered or held.
    let s = analyze_with(&run, persist(), VIEWER).await.unwrap_err();
    assert_eq!(s.code(), Code::PermissionDenied, "{s:?}");
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    let s = analyze_with(&run, unpersist(), VIEWER).await.unwrap_err();
    assert_eq!(s.code(), Code::PermissionDenied, "{s:?}");
    for (eager, local) in [(true, true), (false, true), (false, false)] {
        let s = execute(&run.channel, checkpoint(eager, local), Some(VIEWER))
            .await
            .unwrap_err();
        assert_eq!(
            s.code(),
            Code::PermissionDenied,
            "eager={eager} local={local}: {s:?}"
        );
        assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    }
    // The viewer's refused persist registered nothing.
    let level = |r: pb::AnalyzePlanResponse| match r.result {
        Some(pb::analyze_plan_response::Result::GetStorageLevel(l)) => l.storage_level,
        other => unreachable!("{other:?}"),
    };
    let get = || {
        Analyze::GetStorageLevel(Box::new(areq::GetStorageLevel {
            relation: Some(rel_range(3)),
        }))
    };
    // `get_storage_level` is a read: a viewer may ask, and sees NONE.
    let l = level(
        analyze_with(&run, get(), VIEWER)
            .await
            .expect("viewer reads"),
    )
    .expect("level");
    assert!(!l.use_disk && !l.use_memory);
    // An operator persists, reads, unpersists and checkpoints.
    analyze_with(&op, persist(), OPERATOR)
        .await
        .expect("operator persists");
    let l = level(analyze_with(&op, get(), OPERATOR).await.expect("reads")).expect("level");
    assert!(l.use_memory);
    analyze_with(&op, unpersist(), OPERATOR)
        .await
        .expect("operator unpersists");
    for (eager, local) in [(true, true), (false, true)] {
        execute(&op.channel, checkpoint(eager, local), Some(OPERATOR))
            .await
            .expect("operator checkpoints");
    }
    // Removing an id forgets what the session holds: no floor (own session).
    let removal = command(CommandType::RemoveCachedRemoteRelationCommand(Box::new(
        pb::RemoveCachedRemoteRelationCommand {
            relation: Some(pb::CachedRemoteRelation {
                relation_id: "x".to_owned(),
            }),
        },
    )));
    execute(&run.channel, removal, Some(VIEWER))
        .await
        .expect("a viewer removes an id");
}

/// A viewer's checkpoint of a plan that names a cached local relation is refused for its class
/// (42501) before the relation is built, not with `NOT_FOUND_CACHED_LOCAL_RELATION`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_viewer_is_refused_a_checkpoint_before_its_local_relations_are_built() {
    let run = start().await;
    execute(&run.channel, range(1), Some(VIEWER))
        .await
        .expect("opens the session");
    let cached = pb::Relation {
        common: None,
        rel_type: Some(pb::relation::RelType::CachedLocalRelation(
            pb::CachedLocalRelation {
                hash: "0123abcd".to_owned(),
            },
        )),
    };
    let plan = command(CommandType::CheckpointCommand(Box::new(
        pb::CheckpointCommand {
            relation: Some(cached),
            local: true,
            eager: true,
            storage_level: None,
        },
    )));
    let s = execute(&run.channel, plan, Some(VIEWER)).await.unwrap_err();
    assert_eq!(s.code(), Code::PermissionDenied, "{s:?}");
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
}

/// One `AddArtifacts` stream of `artifacts` (name, bytes) on `SESSION` as `token`.
async fn add_artifacts_as(
    channel: &Channel,
    artifacts: &[(&str, &[u8])],
    token: &str,
) -> Result<pb::AddArtifactsResponse, tonic::Status> {
    use krishiv_connect::proto::v4_3::add_artifacts_request::{
        ArtifactChunk, Batch, Payload, SingleChunkArtifact,
    };
    let artifacts = artifacts
        .iter()
        .map(|(name, data)| SingleChunkArtifact {
            name: (*name).to_owned(),
            data: Some(ArtifactChunk {
                data: data.to_vec(),
                crc: i64::from(krishiv_connect_server::artifacts::crc32(data)),
            }),
        })
        .collect();
    let request = pb::AddArtifactsRequest {
        session_id: SESSION.to_owned(),
        payload: Some(Payload::Batch(Batch { artifacts })),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.client_streaming(
        with_token(futures::stream::iter([request]), Some(token)),
        path("AddArtifacts"),
        tonic_prost::ProstCodec::<pb::AddArtifactsRequest, pb::AddArtifactsResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// ADR 0046 §3b: `AddArtifacts` is client code, `Usage` on `System("udf")` (admin only until
/// T014b), checked on the session before anything is staged, for every name, `cache/`
/// included; a refusal is counted like every other denial.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn adding_artifacts_needs_the_udf_privilege() {
    let run = start().await;
    let cache: &[(&str, &[u8])] = &[("cache/abc123", b"rows")];
    let code: &[(&str, &[u8])] = &[("pyfiles/util.py", b"def f(): return 1\n")];
    for token in [VIEWER, OPERATOR] {
        for artifacts in [cache, code] {
            let s = add_artifacts_as(&run.channel, artifacts, token)
                .await
                .expect_err("only an admin sends artifacts");
            assert_eq!(s.code(), Code::PermissionDenied, "{s:?}");
            assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        }
        release_as(&run, token).await;
    }
    assert_eq!(
        denied(&run),
        [(Surface::Connect, Privilege::Usage, 4)],
        "each refusal is counted once"
    );
    let added = add_artifacts_as(&run.channel, &[cache[0], code[0]], ADMIN)
        .await
        .expect("an admin adds artifacts");
    let names: Vec<&str> = added.artifacts.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["cache/abc123", "pyfiles/util.py"]);
    assert!(added.artifacts.iter().all(|a| a.is_crc_successful));
}

/// PySpark's exit-time `ml_command.clean_cache` is `OwnSession` (ADR 0046 §3b): it succeeds for
/// every role, a viewer included, and is no denial.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ml_clean_cache_succeeds_for_a_viewer() {
    let run = start().await;
    let clean = command(CommandType::MlCommand(Box::new(pb::MlCommand {
        command: Some(pb::ml_command::Command::CleanCache(
            pb::ml_command::CleanCache {},
        )),
    })));
    let responses = execute(&run.channel, clean, Some(VIEWER))
        .await
        .expect("nothing to clean, for any role");
    assert!(responses.is_empty(), "{} responses", responses.len());
    assert!(denied(&run).is_empty(), "{:?}", denied(&run));
}
