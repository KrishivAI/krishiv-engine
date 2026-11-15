// SPDX-License-Identifier: Apache-2.0
//! Trusted impersonation on the Connect listener end to end (P3-T061 I2/I3, design §4 and
//! amendment A12; the security review's wiring requirements): only `x-krishiv-act-as-*`
//! metadata asks, every refusal is `KRISHIV_AUTH_IMPERSONATION_DENIED`, the end user's
//! session belongs to the end user through that front end only, and the act-as identity is
//! pinned per session.
#![cfg(unix)]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{
    ACT_AS_GROUPS, ACT_AS_TOKEN, ACT_AS_USER, ConnectAuth, ConnectServer, ConnectServerConfig,
    SessionLimits, SessionManager, role_sessions,
};
use krishiv_session::Session;
use krishiv_session::auth::{AuthConfig, ImpersonationCount, ServerAuth};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::transport::Channel;

const SESSION: &str = "1d7a3e52-8c0b-4f6e-a9d2-5b4c3a2f1e09";
const PORTAL: &str = "portal-token-0123456789abcdefghijklmnopqrstu";
const CAROL: &str = "carol-token-0123456789abcdefghijklmnopqrstuv";
const ALICE: &str = "alice-token-0123456789abcdefghijklmnopqrstuv";

struct Running {
    auth: Arc<ServerAuth>,
    manager: Arc<SessionManager>,
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

/// A server with the service `portal` (admin, may act for `alice` and members of `analysts`),
/// and the users `carol` (viewer) and `alice` (viewer). `impersonating` decides whether the
/// listener is given the `ServerAuth` for act-as requests.
async fn start(impersonating: bool) -> Running {
    start_with(impersonating, true, SessionLimits::default()).await
}

/// [`start`], with authentication on or off and the given session limits.
async fn start_with(impersonating: bool, authenticated: bool, limits: SessionLimits) -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = secret(dir.path(), "p", PORTAL);
    let c = secret(dir.path(), "c", CAROL);
    let a = secret(dir.path(), "a", ALICE);
    let config = AuthConfig::from_settings([
        ("krishiv.auth.tokens.p.secret", p.as_str()),
        ("krishiv.auth.tokens.p.principal", "portal"),
        ("krishiv.auth.tokens.p.kind", "service"),
        ("krishiv.auth.tokens.p.role", "admin"),
        ("krishiv.auth.principals.portal.expires", "2099-01-01"),
        ("krishiv.auth.principals.portal.privileges", "impersonate"),
        (
            "krishiv.auth.principals.portal.impersonates",
            "alice, group:analysts",
        ),
        ("krishiv.auth.impersonation.roles.operator", "analysts"),
        ("krishiv.auth.tokens.c.secret", c.as_str()),
        ("krishiv.auth.tokens.c.principal", "carol"),
        ("krishiv.auth.tokens.c.role", "viewer"),
        ("krishiv.auth.tokens.a.secret", a.as_str()),
        ("krishiv.auth.tokens.a.principal", "alice"),
        ("krishiv.auth.tokens.a.role", "viewer"),
    ])
    .expect("auth config");
    let auth = Arc::new(ServerAuth::new(&config).expect("auth"));
    let denials = Arc::new(krishiv_session::auth::DeniedCounts::new());
    let factory = role_sessions(
        Session::new().expect("session"),
        &Config::new(),
        Arc::clone(&denials),
    )
    .expect("factory");
    let manager = Arc::new(SessionManager::new(factory, limits));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let mut connect = ConnectAuth::new(Arc::clone(&auth) as Arc<_>).with_denials(denials);
    if impersonating {
        connect = connect.with_impersonation(Arc::clone(&auth));
    }
    let mut server = ConnectServer::new(Arc::clone(&manager), ConnectServerConfig::default());
    if authenticated {
        server = server.with_auth(connect);
    }
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
        auth,
        manager,
        channel,
        _stop: stop,
        _dir: dir,
    }
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

/// The string rows of an `ExecutePlan` of `plan` in `session` with `token` and the act-as
/// `metadata`, or the status it ended with.
async fn execute(
    run: &Running,
    session: &str,
    plan: pb::Plan,
    token: &str,
    metadata: &[(&'static str, &'static str)],
) -> Result<Vec<Vec<String>>, tonic::Status> {
    use arrow_array::Array as _;
    let request = pb::ExecutePlanRequest {
        session_id: session.to_owned(),
        user_context: Some(pb::UserContext {
            // Never identity, and never an act-as request on its own.
            user_id: "mallory".to_owned(),
            ..Default::default()
        }),
        plan: Some(plan),
        ..Default::default()
    };
    let mut request = tonic::Request::new(request);
    request.metadata_mut().insert(
        "authorization",
        format!("Bearer {token}").parse().expect("ascii"),
    );
    for (k, v) in metadata {
        request.metadata_mut().append(*k, v.parse().expect("ascii"));
    }
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let mut stream = grpc
        .server_streaming(
            request,
            "/spark.connect.SparkConnectService/ExecutePlan"
                .parse()
                .expect("path"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut rows = Vec::new();
    while let Some(r) = stream.message().await? {
        let Some(pb::execute_plan_response::ResponseType::ArrowBatch(b)) = r.response_type else {
            continue;
        };
        for batch in arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data), None)
            .expect("ipc")
        {
            let batch = batch.expect("batch");
            let columns: Vec<_> = batch.columns().to_vec();
            for row in 0..batch.num_rows() {
                rows.push(
                    columns
                        .iter()
                        .map(|c| {
                            let c = c
                                .as_any()
                                .downcast_ref::<arrow_array::StringArray>()
                                .expect("string column");
                            if c.is_null(row) {
                                "NULL".to_owned()
                            } else {
                                c.value(row).to_owned()
                            }
                        })
                        .collect(),
                );
            }
        }
    }
    Ok(rows)
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

fn assert_impersonation_denied(result: Result<Vec<Vec<String>>, tonic::Status>) {
    let status = result.expect_err("refused");
    assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");
    assert_eq!(error_class(&status), "KRISHIV_AUTH_IMPERSONATION_DENIED");
}

fn counts(allowed: u64, denied: u64) -> ImpersonationCount {
    let mut c = ImpersonationCount::default();
    c.allowed = allowed;
    c.denied = denied;
    c
}

const OWN_SESSIONS: &str = "SELECT principal, delegated_by, client_user_id FROM system.sessions";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_service_acts_for_an_allowed_user() {
    let run = start(true).await;
    let rows = execute(
        &run,
        SESSION,
        sql(OWN_SESSIONS),
        PORTAL,
        &[(ACT_AS_USER, "alice")],
    )
    .await
    .expect("portal acts for alice");
    // The end user's own rows: alice through portal; `user_id` stays the client's claim.
    assert_eq!(rows, [["alice", "portal", "mallory"]]);
    assert_eq!(run.auth.impersonation_counts(), counts(1, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_user_id_alone_never_asks() {
    let run = start(true).await;
    // `user_id = mallory` in the body, no act-as metadata: carol is carol.
    let rows = execute(&run, SESSION, sql(OWN_SESSIONS), CAROL, &[])
        .await
        .expect("carol");
    assert_eq!(rows, [["carol", "NULL", "mallory"]]);
    assert_eq!(run.auth.impersonation_counts(), counts(0, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn act_as_is_refused_without_the_right_or_outside_the_allowlist() {
    let run = start(true).await;
    // carol holds no `impersonate`.
    assert_impersonation_denied(
        execute(
            &run,
            SESSION,
            sql("SELECT 'ok'"),
            CAROL,
            &[(ACT_AS_USER, "alice")],
        )
        .await,
    );
    // portal may act for alice and analysts only.
    assert_impersonation_denied(
        execute(
            &run,
            SESSION,
            sql("SELECT 'ok'"),
            PORTAL,
            &[(ACT_AS_USER, "bob")],
        )
        .await,
    );
    assert_eq!(run.auth.impersonation_counts(), counts(0, 2));
    assert_eq!(run.manager.counts().live, 0, "no session for a refusal");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unreadable_act_as_metadata_is_refused_not_ignored() {
    let run = start(true).await;
    for metadata in [
        // Groups for no one.
        &[(ACT_AS_GROUPS, "analysts")][..],
        // Twice the same header.
        &[(ACT_AS_USER, "alice"), (ACT_AS_USER, "bob")][..],
        // A broken escape.
        &[(ACT_AS_USER, "al%ice")][..],
    ] {
        assert_impersonation_denied(
            execute(&run, SESSION, sql("SELECT 'ok'"), PORTAL, metadata).await,
        );
    }
    assert_eq!(run.auth.impersonation_counts(), counts(0, 3));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_signed_assertion_is_refused_while_signed_assertions_are_off() {
    let run = start(true).await;
    assert_impersonation_denied(
        execute(
            &run,
            SESSION,
            sql("SELECT 'ok'"),
            PORTAL,
            &[(ACT_AS_TOKEN, "e30.e30.c2ln")],
        )
        .await,
    );
    assert_eq!(run.auth.impersonation_counts(), counts(0, 1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_listener_without_impersonation_refuses_act_as() {
    let run = start(false).await;
    assert_impersonation_denied(
        execute(
            &run,
            SESSION,
            sql("SELECT 'ok'"),
            PORTAL,
            &[(ACT_AS_USER, "alice")],
        )
        .await,
    );
    // Without the act-as metadata the service is itself (an admin, which sees every row).
    let own = format!("{OWN_SESSIONS} WHERE principal = 'portal'");
    let rows = execute(&run, SESSION, sql(&own), PORTAL, &[])
        .await
        .expect("portal as itself");
    assert_eq!(rows, [["portal", "NULL", "mallory"]]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_made_through_a_front_end_is_not_the_end_users_own() {
    let run = start(true).await;
    execute(
        &run,
        SESSION,
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice")],
    )
    .await
    .expect("portal for alice");
    // alice herself, with the same session id: another owner's session, answered as unknown.
    let status = execute(&run, SESSION, sql("SELECT 'ok'"), ALICE, &[])
        .await
        .expect_err("not alice's own session");
    assert_eq!(error_class(&status), "INVALID_HANDLE.SESSION_NOT_FOUND");
    // Nor does portal itself get it.
    let status = execute(&run, SESSION, sql("SELECT 'ok'"), PORTAL, &[])
        .await
        .expect_err("not portal's own session");
    assert_eq!(error_class(&status), "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert_eq!(run.manager.counts().live, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_act_as_identity_is_pinned_per_session() {
    let run = start(true).await;
    execute(
        &run,
        SESSION,
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice"), (ACT_AS_GROUPS, "analysts")],
    )
    .await
    .expect("portal for alice, an analyst");
    // The same front end, the same user, other groups: not that session.
    let status = execute(
        &run,
        SESSION,
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice")],
    )
    .await
    .expect_err("groups changed mid-session");
    assert_eq!(error_class(&status), "INVALID_HANDLE.SESSION_NOT_FOUND");
    // Repeated identically, it is.
    execute(
        &run,
        SESSION,
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice"), (ACT_AS_GROUPS, "analysts")],
    )
    .await
    .expect("the same act-as identity");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_end_users_role_comes_from_its_groups() {
    let run = start(true).await;
    // carol runs a query: a row of hers in `system.queries`.
    execute(&run, SESSION, sql("SELECT 'ok'"), CAROL, &[])
        .await
        .expect("carol");
    let principals = "SELECT DISTINCT principal FROM system.queries ORDER BY principal";
    // alice without groups is a viewer (the default role of an impersonated user, not
    // portal's admin): her own rows only.
    let rows = execute(
        &run,
        "2e8b4f63-9d1c-4a7f-b0e3-6c5d4b3a2f10",
        sql(principals),
        PORTAL,
        &[(ACT_AS_USER, "alice")],
    )
    .await
    .expect("alice the viewer");
    assert_eq!(rows, [["alice"]]);
    // As an analyst she is an operator, whose `Monitor` shows every principal's rows: the role
    // `ServerAuth::impersonate` returned is the one the session got (review requirement 6).
    let rows = execute(
        &run,
        "3f9c5a74-0e2d-4b8a-81f4-7d6e5c4b3a21",
        sql(principals),
        PORTAL,
        &[(ACT_AS_USER, "alice"), (ACT_AS_GROUPS, "analysts")],
    )
    .await
    .expect("alice the analyst");
    assert!(rows.contains(&vec!["carol".to_owned()]), "{rows:?}");
}

/// Review M2 (A12): a server without authentication refuses act-as metadata instead of
/// serving it as its local user.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_server_without_authentication_refuses_act_as() {
    let run = start_with(true, false, SessionLimits::default()).await;
    for metadata in [
        &[(ACT_AS_USER, "alice")][..],
        &[(ACT_AS_GROUPS, "analysts")][..],
        &[(ACT_AS_TOKEN, "e30.e30.c2ln")][..],
    ] {
        assert_impersonation_denied(
            execute(&run, SESSION, sql("SELECT 'ok'"), PORTAL, metadata).await,
        );
    }
    assert_eq!(run.manager.counts().live, 0, "no session for a refusal");
    // Without act-as metadata the local user is served as before.
    execute(&run, SESSION, sql("SELECT 'ok'"), PORTAL, &[])
        .await
        .expect("the local user");
}

/// Review m4: the pinned groups are a set, so their order does not matter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_pinned_groups_are_a_set() {
    let run = start(true).await;
    execute(
        &run,
        SESSION,
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice"), (ACT_AS_GROUPS, "analysts,eng")],
    )
    .await
    .expect("portal for alice");
    execute(
        &run,
        SESSION,
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice"), (ACT_AS_GROUPS, "eng,analysts,eng")],
    )
    .await
    .expect("the same groups in another order");
    assert_eq!(run.manager.counts().live, 1);
}

/// Review m1: the groups a front end sends do not multiply the per-principal session cap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn other_groups_do_not_multiply_the_session_cap() {
    let limits = SessionLimits {
        max_sessions_per_principal: 1,
        ..SessionLimits::default()
    };
    let run = start_with(true, true, limits).await;
    execute(
        &run,
        SESSION,
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice")],
    )
    .await
    .expect("the first session of alice through portal");
    let status = execute(
        &run,
        "4a0d6b85-1f3e-4c9b-92a5-8e7f6d5c4b32",
        sql("SELECT 'ok'"),
        PORTAL,
        &[(ACT_AS_USER, "alice"), (ACT_AS_GROUPS, "analysts")],
    )
    .await
    .expect_err("a second session past the cap");
    assert_eq!(error_class(&status), "KRISHIV_SESSION_LIMIT");
    // alice herself has her own cap.
    execute(
        &run,
        "5b1e7c96-2a4f-4d0c-a3b6-9f8e7d6c5b43",
        sql("SELECT 'ok'"),
        ALICE,
        &[],
    )
    .await
    .expect("alice's own session");
}
