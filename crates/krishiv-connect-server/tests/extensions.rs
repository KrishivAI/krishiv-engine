// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.v1` extensions over a raw tonic client (P3-T015): the capability set a Krishiv client
//! reads, the worked example `Echo` run end to end, an unknown extension answered with its capability,
//! and the rule that a stock client never meets an extension message.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use krishiv_connect::extension::{CAPABILITIES_KEY, Echo, TYPE_URL_PREFIX};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::config_request::operation::OpType as ConfOp;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{
    ConnectServer, ConnectServerConfig, ExecuteLimits, SessionLimits, SessionManager,
};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";

struct Running {
    channel: Channel,
    _stop: oneshot::Sender<()>,
}

async fn start() -> Running {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits::default(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(
        manager,
        ConnectServerConfig {
            execute: ExecuteLimits::default(),
            ..ConnectServerConfig::default()
        },
    );
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
        channel,
        _stop: stop,
    }
}

fn path(rpc: &str) -> tonic::codegen::http::uri::PathAndQuery {
    format!("/spark.connect.SparkConnectService/{rpc}")
        .parse()
        .expect("path")
}

async fn config(run: &Running, op: ConfOp) -> Result<pb::ConfigResponse, tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        tonic::Request::new(pb::ConfigRequest {
            session_id: SESSION.to_owned(),
            operation: Some(pb::config_request::Operation { op_type: Some(op) }),
            ..Default::default()
        }),
        path("Config"),
        tonic_prost::ProstCodec::<pb::ConfigRequest, pb::ConfigResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

async fn execute(
    run: &Running,
    relation: pb::Relation,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let mut stream = grpc
        .server_streaming(
            tonic::Request::new(pb::ExecutePlanRequest {
                session_id: SESSION.to_owned(),
                user_context: Some(pb::UserContext {
                    user_id: "someone".to_owned(),
                    ..Default::default()
                }),
                plan: Some(pb::Plan {
                    op_type: Some(OpType::Root(Box::new(relation))),
                }),
                ..Default::default()
            }),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(r) = stream.message().await? {
        out.push(r);
    }
    Ok(out)
}

fn extension(name: &str, value: Vec<u8>) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(RelType::Extension(prost_types::Any {
            type_url: format!("{TYPE_URL_PREFIX}{name}"),
            value,
        })),
    }
}

fn sql(query: &str) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(RelType::Sql(pb::Sql {
            query: query.to_owned(),
            ..Default::default()
        })),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_krishiv_client_reads_the_capabilities() {
    let run = start().await;
    for op in [
        ConfOp::Get(pb::config_request::Get {
            keys: vec![CAPABILITIES_KEY.to_owned()],
        }),
        ConfOp::GetOption(pb::config_request::GetOption {
            keys: vec![CAPABILITIES_KEY.to_owned()],
        }),
    ] {
        let r = config(&run, op).await.expect("get");
        assert_eq!(r.pairs.len(), 1);
        assert_eq!(
            r.pairs[0].value.as_deref(),
            Some("echo,persistpythonfunction")
        );
    }
    // The same set is a `surface` row of `system.capabilities`.
    let rows = krishiv_session::capabilities();
    assert!(
        rows.iter().any(|r| r[0] == "surface"
            && r[1] == "krishiv.v1"
            && r[3] == "echo,persistpythonfunction"),
        "{rows:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn echo_runs_end_to_end_as_a_one_row_relation() {
    let run = start().await;
    let echo = Echo {
        message: "hello extension".to_owned(),
    };
    let responses = execute(&run, extension("Echo", echo.encode_to_vec()))
        .await
        .expect("execute");
    let batch = responses
        .iter()
        .find_map(|r| match &r.response_type {
            Some(ResponseType::ArrowBatch(b)) => Some(b),
            _ => None,
        })
        .expect("a batch");
    let batches: Vec<_> =
        arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(batch.data.clone()), None)
            .expect("ipc")
            .map(|b| b.expect("batch"))
            .collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "message");
    assert_eq!(
        batches[0].column(0).as_string::<i32>().value(0),
        "hello extension"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_extension_names_its_capability() {
    let run = start().await;
    // Bytes a newer client's message would hold: not decodable as anything the server knows.
    let status = execute(&run, extension("Vacuum", vec![0xFF, 0xFE, 0x01]))
        .await
        .unwrap_err();
    let text = status.message();
    assert!(text.contains("capability `vacuum`"), "{text}");
    assert!(!text.to_lowercase().contains("decode"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stock_client_never_sees_an_extension() {
    let run = start().await;
    let needle = b"krishiv.v1";
    let has = |bytes: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);
    // Config: every read a stock client makes, getAll included.
    for op in [
        ConfOp::GetAll(pb::config_request::GetAll { prefix: None }),
        ConfOp::GetAll(pb::config_request::GetAll {
            prefix: Some("krishiv".to_owned()),
        }),
        ConfOp::Get(pb::config_request::Get {
            keys: vec!["spark.sql.ansi.enabled".to_owned()],
        }),
    ] {
        let r = config(&run, op).await.expect("config");
        assert!(!has(&r.encode_to_vec()), "{r:?}");
    }
    // Execution: a SQL query and a range answer without a krishiv.v1 message.
    for relation in [
        sql("SELECT 1 AS a, 'x' AS b"),
        pb::Relation {
            common: None,
            rel_type: Some(RelType::Range(pb::Range {
                start: Some(0),
                end: 5,
                step: 1,
                num_partitions: Some(1),
            })),
        },
    ] {
        for r in execute(&run, relation).await.expect("execute") {
            assert!(!has(&r.encode_to_vec()), "{r:?}");
        }
    }
}
