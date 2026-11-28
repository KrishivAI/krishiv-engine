// SPDX-License-Identifier: Apache-2.0
//! A Spark Connect client of a served runtime (`krishiv server`: `DeclMode::Server`) never reads
//! the server's environment through `${env:NAME}` / `${system:NAME}` in its SQL (rev-f2 code
//! health 0.3, finding I-1), whatever it sets through `Config` or `SET`; only the startup key
//! `krishiv.server.sql.substituteEnvironment` restores Spark's substitution.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;

use krishiv_common::config::Config;
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, SessionLimits, SessionManager, role_sessions,
};
use krishiv_session::auth::{AuthFailure, Authenticated, Authenticator, Credentials, Method, Role};
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::CatalogSetup;
use krishiv_session::{BoxFuture, EngineRuntime, Session};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d20";
const TOKEN: &str = "admin-token";

/// Every bearer is the admin `svc`: the most a client can be.
struct AdminAuth;

impl Authenticator for AdminAuth {
    fn authenticate<'a>(
        &'a self,
        credentials: &'a Credentials<'a>,
    ) -> BoxFuture<'a, Result<Authenticated, AuthFailure>> {
        Box::pin(async move {
            credentials.bearer.ok_or(AuthFailure::Missing)?;
            Ok(Authenticated::new(
                Principal::new("svc", PrincipalKind::Service),
                Some(Role::Admin),
                Method::Token,
            ))
        })
    }
}

/// A Connect server as `krishiv server` builds it, over a server-mode runtime with `config`.
async fn start(config: &[(&str, &str)]) -> (Channel, oneshot::Sender<()>) {
    let runtime = config
        .iter()
        .fold(
            EngineRuntime::builder().catalogs(CatalogSetup::new(DeclMode::Server)),
            |b, (k, v)| b.config(*k, *v),
        )
        .build()
        .expect("runtime");
    let base = Session::builder()
        .runtime(runtime)
        .build()
        .expect("session");
    let denials = Arc::new(krishiv_session::auth::DeniedCounts::new());
    let factory = role_sessions(base, &Config::new(), Arc::clone(&denials)).expect("factory");
    let manager = Arc::new(SessionManager::new(factory, SessionLimits::default()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(manager, ConnectServerConfig::default())
        .with_auth(ConnectAuth::new(Arc::new(AdminAuth)).with_denials(denials));
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    (channel, stop)
}

fn authorized<T>(message: T) -> tonic::Request<T> {
    let mut r = tonic::Request::new(message);
    r.metadata_mut().insert(
        "authorization",
        format!("Bearer {TOKEN}").parse().expect("ascii"),
    );
    r
}

fn user() -> Option<pb::UserContext> {
    Some(pb::UserContext {
        user_id: "anyone".to_owned(),
        ..Default::default()
    })
}

fn plan_of(op_type: OpType) -> pb::Plan {
    pb::Plan {
        op_type: Some(op_type),
    }
}

/// A root `SQL` relation (what a client sends again after `spark.sql` answered it).
fn query(sql: &str) -> pb::Plan {
    plan_of(OpType::Root(Box::new(pb::Relation {
        common: None,
        rel_type: Some(RelType::Sql(pb::Sql {
            query: sql.to_owned(),
            ..Default::default()
        })),
    })))
}

/// The command PySpark 4.x sends for `spark.sql(sql)`: a command runs at once.
fn command(sql: &str) -> pb::Plan {
    let command = pb::SqlCommand {
        input: Some(pb::Relation {
            common: None,
            rel_type: Some(RelType::Sql(pb::Sql {
                query: sql.to_owned(),
                ..Default::default()
            })),
        }),
        ..Default::default()
    };
    plan_of(OpType::Command(Box::new(pb::Command {
        command_type: Some(pb::command::CommandType::SqlCommand(Box::new(command))),
    })))
}

/// `plan` run over Connect: the first string value of its batches, or the failure's status.
async fn run(channel: &Channel, plan: pb::Plan) -> Result<String, tonic::Status> {
    let request = pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        user_context: user(),
        plan: Some(plan),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let mut s = grpc
        .server_streaming(
            authorized(request),
            "/spark.connect.SparkConnectService/ExecutePlan"
                .parse()
                .expect("path"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(m) = s.message().await? {
        let Some(ResponseType::ArrowBatch(b)) = m.response_type else {
            continue;
        };
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data.clone()), None)
                .expect("ipc");
        for batch in reader {
            let batch = batch.expect("batch");
            let column = batch.column(0);
            let text = match column.as_string_opt::<i32>() {
                Some(a) => a.value(0).to_owned(),
                None => column.as_string_view().value(0).to_owned(),
            };
            out.push(text);
        }
    }
    Ok(out.into_iter().next().unwrap_or_default())
}

/// What a view made by `spark.sql("CREATE TEMPORARY VIEW .. AS SELECT '<reference>'")` holds.
async fn stored(channel: &Channel, view: &str, reference: &str) -> Result<String, tonic::Status> {
    run(
        channel,
        command(&format!(
            "CREATE OR REPLACE TEMPORARY VIEW {view} AS SELECT '{reference}' AS s"
        )),
    )
    .await?;
    run(channel, query(&format!("SELECT s FROM {view}"))).await
}

/// Spark Connect `Config.set` of one pair on the session.
async fn config_set(channel: &Channel, key: &str, value: &str) -> Result<(), tonic::Status> {
    let request = pb::ConfigRequest {
        session_id: SESSION.to_owned(),
        user_context: user(),
        operation: Some(pb::config_request::Operation {
            op_type: Some(pb::config_request::operation::OpType::Set(
                pb::config_request::Set {
                    pairs: vec![pb::KeyValue {
                        key: key.to_owned(),
                        value: Some(value.to_owned()),
                    }],
                    silent: None,
                },
            )),
        }),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        authorized(request),
        "/spark.connect.SparkConnectService/Config"
            .parse()
            .expect("path"),
        tonic_prost::ProstCodec::<pb::ConfigRequest, pb::ConfigResponse>::default(),
    )
    .await
    .map(|_| ())
}

fn path() -> String {
    std::env::var("PATH").expect("PATH is set")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connect_client_never_reads_the_servers_environment() {
    let (channel, _stop) = start(&[]).await;
    for reference in ["${env:PATH}", "${system:PATH}"] {
        let got = stored(&channel, "v", reference).await.expect("view");
        assert_eq!(got, reference);
    }
    // No client setting brings it back: substitution itself is on already, and the server key is
    // static, for `Config` and for SQL `SET` alike.
    config_set(&channel, "spark.sql.variable.substitute", "true")
        .await
        .expect("set");
    let refused = config_set(&channel, "krishiv.server.sql.substituteEnvironment", "true").await;
    assert!(refused.is_err(), "a client set the startup key");
    let refused = run(
        &channel,
        command("SET krishiv.server.sql.substituteEnvironment = true"),
    )
    .await;
    assert!(refused.is_err(), "SQL SET of the startup key");
    let got = stored(&channel, "w", "${env:PATH}").await.expect("view");
    assert_eq!(got, "${env:PATH}");
    assert!(!got.contains(&path()));
    // A configuration value is still substituted.
    run(&channel, command("SET probe = substituted"))
        .await
        .expect("set");
    assert_eq!(
        stored(&channel, "x", "${probe}").await.expect("view"),
        "substituted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_operator_restores_sparks_substitution_at_startup() {
    let (channel, _stop) = start(&[("krishiv.server.sql.substituteEnvironment", "true")]).await;
    let got = stored(&channel, "v", "${env:PATH}").await.expect("view");
    assert_eq!(got, path());
}

/// A setting `Config.get` hides (`is_redacted_setting`) is not read through `${key}` either: the
/// operator's startup value stays behind the reference; a plain one is still substituted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connect_client_never_reads_a_hidden_setting_through_substitution() {
    let (channel, _stop) =
        start(&[("spark.my.token", "tok-123"), ("spark.my.plain", "visible")]).await;
    for (view, reference) in [
        ("v", "${spark.my.token}"),
        ("w", "${hiveconf:spark.my.token}"),
    ] {
        let got = stored(&channel, view, reference).await.expect("view");
        assert_eq!(got, reference);
    }
    assert_eq!(
        stored(&channel, "x", "${spark.my.plain}")
            .await
            .expect("view"),
        "visible"
    );
}
