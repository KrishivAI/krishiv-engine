// SPDX-License-Identifier: Apache-2.0
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;

use krishiv_common::principal::PrincipalKind;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::command::CommandType;
use krishiv_connect::proto::v4_3::ml_command::Command as MlCommandType;
use krishiv_session::auth::Authenticated;
use krishiv_session::auth::{AuthConfig, Method, Role, ServerAuth};
use krishiv_session::{Privilege, Securable, StatementClass};
use tonic::codegen::http::HeaderMap;
use tonic::{Code, Request};

use super::{Check, ConnectAuth, Rpc, SessionOwner, check_command, check_rpc};

const TOKEN: &str = "cccccccccccccccccccccccccccccccccccccccc";

#[expect(
    clippy::disallowed_methods,
    reason = "test setup: the secret file a token config reads"
)]
fn secret_file(dir: &Path, contents: &str) -> String {
    let path = dir.join("token");
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(contents.as_bytes()).expect("write");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("mode");
    }
    format!("file:{}", path.display())
}

/// A `ConnectAuth` over a real `ServerAuth` with one token: `carol`, viewer.
fn token_auth(dir: &Path) -> ConnectAuth {
    let secret = secret_file(dir, TOKEN);
    let config = AuthConfig::from_settings([
        ("krishiv.auth.tokens.c.secret", secret.as_str()),
        ("krishiv.auth.tokens.c.principal", "carol"),
        ("krishiv.auth.tokens.c.role", "viewer"),
    ])
    .expect("config");
    ConnectAuth::new(Arc::new(ServerAuth::new(&config).expect("auth")))
}

fn peer(last: u8) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 1, 2, last)), 50_000)
}

/// The headers of a request with the `authorization` values `values`.
fn headers(values: &[&str]) -> HeaderMap {
    let mut h = HeaderMap::new();
    for v in values {
        h.append("authorization", v.parse().expect("ascii"));
    }
    h
}

/// Authenticates a request from `peer` with the `authorization` values `values`.
fn authenticate(
    auth: &ConnectAuth,
    peer: Option<SocketAddr>,
    values: &[&str],
) -> Result<Authenticated, tonic::Status> {
    block_on(auth.authenticate(&headers(values), peer, None))
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    futures::executor::block_on(f)
}

#[test]
fn off_is_the_local_user_without_a_credential() {
    let auth = ConnectAuth::off();
    assert!(!auth.is_enabled());
    let caller = auth.caller(&Request::new(())).expect("local");
    assert_eq!(
        caller.principal.name,
        krishiv_common::principal::Principal::local_user().name
    );
    assert_eq!(caller.method, Method::None);
    assert_eq!(caller.role, None);
}

#[test]
fn a_token_names_its_principal_and_role() {
    let dir = tempfile::tempdir().expect("dir");
    let auth = token_auth(dir.path());
    assert!(auth.is_enabled());
    let caller =
        authenticate(&auth, Some(peer(1)), &[&format!("Bearer {TOKEN}")]).expect("authenticated");
    assert_eq!(caller.principal.name, "carol");
    assert_eq!(caller.principal.kind, PrincipalKind::User);
    assert_eq!(caller.role, Some(Role::Viewer));
    assert_eq!(caller.method, Method::Token);
}

#[test]
fn failures_are_spark_unauthenticated_messages() {
    let dir = tempfile::tempdir().expect("dir");
    let auth = token_auth(dir.path());
    let status = |values: &[&str]| authenticate(&auth, Some(peer(2)), values).expect_err("refused");
    let missing = status(&[]);
    assert_eq!(missing.code(), Code::Unauthenticated);
    assert_eq!(missing.message(), "No authentication token provided");
    let wrong = status(&["Bearer not-the-token"]);
    assert_eq!(wrong.code(), Code::Unauthenticated);
    assert_eq!(wrong.message(), "Invalid authentication token");
    // Another scheme, and two `authorization` entries, are malformed: invalid, not missing.
    let basic = status(&["Basic dXNlcjpwdw=="]);
    assert_eq!(basic.message(), "Invalid authentication token");
    let two = status(&[&format!("Bearer {TOKEN}"), &format!("Bearer {TOKEN}")]);
    assert_eq!(two.code(), Code::Unauthenticated);
    assert_eq!(two.message(), "Invalid authentication token");
    // The credential never appears in the status.
    for s in [&missing, &wrong, &basic, &two] {
        assert!(!s.message().contains(TOKEN));
    }
}

#[test]
fn a_request_without_a_peer_address_is_refused() {
    let dir = tempfile::tempdir().expect("dir");
    let auth = token_auth(dir.path());
    let s = authenticate(&auth, None, &[&format!("Bearer {TOKEN}")]).expect_err("fails closed");
    assert_eq!(s.code(), Code::Unauthenticated);
}

#[test]
fn a_throttled_peer_is_unavailable() {
    let dir = tempfile::tempdir().expect("dir");
    let auth = token_auth(dir.path());
    let mut last = None;
    // The default throttle blocks a peer after 20 failures a minute.
    for _ in 0..25 {
        last = authenticate(&auth, Some(peer(3)), &["Bearer wrong"]).err();
    }
    let s = last.expect("refused");
    assert_eq!(s.code(), Code::Unavailable);
    // Even the right token waits out the block, and another peer is unaffected.
    let blocked = authenticate(&auth, Some(peer(3)), &[&format!("Bearer {TOKEN}")]);
    assert_eq!(blocked.expect_err("blocked").code(), Code::Unavailable);
    authenticate(&auth, Some(peer(4)), &[&format!("Bearer {TOKEN}")]).expect("other peer");
}

#[test]
fn the_owner_leaves_out_groups_and_keeps_the_method() {
    let dir = tempfile::tempdir().expect("dir");
    let auth = token_auth(dir.path());
    let caller =
        authenticate(&auth, Some(peer(5)), &[&format!("Bearer {TOKEN}")]).expect("authenticated");
    let mut grouped = caller.clone();
    grouped.principal.groups = vec!["bi".to_owned()];
    assert_eq!(SessionOwner::of(&caller), SessionOwner::of(&grouped));
    let mut oidc = caller.clone();
    oidc.method = Method::Oidc;
    assert_ne!(SessionOwner::of(&caller), SessionOwner::of(&oidc));
}

fn command(c: CommandType) -> pb::Command {
    pb::Command {
        command_type: Some(c),
    }
}

fn udf_usage() -> Check {
    Check::Privilege {
        securable: Securable::System("udf".to_owned()),
        privilege: Privilege::Usage,
        class: StatementClass::Session,
    }
}

#[test]
fn every_rpc_has_its_class() {
    for rpc in [Rpc::ExecutePlan, Rpc::AnalyzePlan] {
        assert_eq!(check_rpc(rpc), Check::Plan, "{rpc:?}");
    }
    assert_eq!(check_rpc(Rpc::AddArtifacts), udf_usage());
    for rpc in [
        Rpc::Config,
        Rpc::ArtifactStatus,
        Rpc::Interrupt,
        Rpc::ReattachExecute,
        Rpc::ReleaseExecute,
        Rpc::ReleaseSession,
        Rpc::FetchErrorDetails,
        Rpc::CloneSession,
        Rpc::GetStatus,
    ] {
        assert_eq!(check_rpc(rpc), Check::OwnSession, "{rpc:?}");
    }
}

#[test]
fn every_command_has_its_class() {
    assert_eq!(check_command(&pb::Command::default()), Check::Plan);
    let cases: Vec<(CommandType, Check)> = vec![
        (CommandType::SqlCommand(Box::default()), Check::Plan),
        (
            CommandType::CheckpointCommand(Box::default()),
            Check::SideEffect,
        ),
        (CommandType::WriteOperation(Box::default()), Check::Write),
        (CommandType::WriteOperationV2(Box::default()), Check::Write),
        (
            CommandType::MergeIntoTableCommand(Box::default()),
            Check::Write,
        ),
        (CommandType::RegisterFunction(Box::default()), udf_usage()),
        (
            CommandType::RegisterTableFunction(Box::default()),
            udf_usage(),
        ),
        (CommandType::RegisterDataSource(Box::default()), udf_usage()),
        (
            CommandType::CreateDataframeView(Box::default()),
            Check::OwnSession,
        ),
        (
            CommandType::CreateDataframeView(Box::new(pb::CreateDataFrameViewCommand {
                is_global: true,
                ..Default::default()
            })),
            Check::Privilege {
                securable: Securable::Namespace(vec![
                    "spark_catalog".to_owned(),
                    "global_temp".to_owned(),
                ]),
                privilege: Privilege::Create,
                class: StatementClass::Ddl,
            },
        ),
        (
            CommandType::GetResourcesCommand(Box::default()),
            Check::OwnSession,
        ),
        (
            CommandType::RemoveCachedRemoteRelationCommand(Box::default()),
            Check::OwnSession,
        ),
        (
            CommandType::MlCommand(Box::new(pb::MlCommand {
                command: Some(MlCommandType::CleanCache(Default::default())),
            })),
            Check::OwnSession,
        ),
        (
            CommandType::MlCommand(Box::new(pb::MlCommand {
                command: Some(MlCommandType::GetCacheInfo(Default::default())),
            })),
            Check::Unsupported("ml_command"),
        ),
        (
            CommandType::MlCommand(Box::default()),
            Check::Unsupported("ml_command"),
        ),
        (
            CommandType::PipelineCommand(Box::default()),
            Check::Unsupported("pipeline_command"),
        ),
        (
            CommandType::ExecuteExternalCommand(Box::default()),
            Check::Unsupported("execute_external_command"),
        ),
        (
            CommandType::CreateResourceProfileCommand(Box::default()),
            Check::Unsupported("create_resource_profile_command"),
        ),
        (
            CommandType::StreamingQueryCommand(Box::default()),
            Check::Unsupported("streaming commands"),
        ),
        (
            CommandType::Extension(Box::default()),
            Check::Unsupported("extension"),
        ),
    ];
    for (c, want) in cases {
        assert_eq!(check_command(&command(c.clone())), want, "{c:?}");
    }
}

#[test]
fn enforce_asks_the_session_policy() {
    use krishiv_session::auth::RolePolicy;
    let base = krishiv_session::Session::new().expect("session");
    let viewer = base.new_session_as(
        krishiv_common::principal::Principal::new("v", PrincipalKind::User),
        Arc::new(RolePolicy::new(
            Some(Role::Viewer),
            Role::Admin,
            krishiv_session::PolicyVersion(1),
        )),
    );
    let e = block_on(super::enforce(&udf_usage(), &viewer)).expect_err("denied");
    assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    for allowed in [
        Check::OwnSession,
        Check::Plan,
        Check::Write,
        Check::Unsupported("x"),
    ] {
        block_on(super::enforce(&allowed, &viewer)).expect("decided elsewhere");
    }
    let admin = base.new_session_as(
        krishiv_common::principal::Principal::new("a", PrincipalKind::User),
        Arc::new(RolePolicy::new(
            Some(Role::Admin),
            Role::Admin,
            krishiv_session::PolicyVersion(1),
        )),
    );
    block_on(super::enforce(&udf_usage(), &admin)).expect("admin sends client code");
}

#[test]
fn with_authentication_on_a_request_without_the_layers_caller_is_refused() {
    // The service reads the caller the layer attached; without one it fails closed.
    let dir = tempfile::tempdir().expect("dir");
    let auth = token_auth(dir.path());
    let s = auth.caller(&Request::new(())).expect_err("no layer ran");
    assert_eq!(s.code(), Code::Unauthenticated);
    let mut attached = Request::new(());
    let carol = authenticate(&auth, Some(peer(6)), &[&format!("Bearer {TOKEN}")]).expect("token");
    attached.extensions_mut().insert(carol.clone());
    assert_eq!(auth.caller(&attached).expect("attached"), carol);
}
