// SPDX-License-Identifier: Apache-2.0
//! `write_operation` and `write_operation_v2` (P3-T012 PR-24, group 11) end to end over a raw tonic
//! client: a path write read back, `save()` with the `path` option, `save()` with neither
//! (`_LEGACY_ERROR_TEMP_2050`), `saveAsTable` and `insertInto`, a V2 create and append, and the
//! `errorifexists` default on an existing path.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]
#![expect(
    clippy::disallowed_methods,
    reason = "the tests read back the server's status details and Arrow data"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::proto::v4_3::write_operation::{SaveMode, SaveTable, SaveType, save_table};
use krishiv_connect::proto::v4_3::write_operation_v2::Mode;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
use krishiv_session::Session;
use krishiv_session::auth::{Authenticated, Method};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "5b1f7c0e-3a52-4f0b-8d6e-1c2a9e4b7d33";

struct Running {
    manager: Arc<SessionManager>,
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
    let server = ConnectServer::new(Arc::clone(&manager), ConnectServerConfig::default());
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
        channel,
        _stop: stop,
    }
}

async fn collect(
    channel: &Channel,
    plan: pb::Plan,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let request = pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "someone".to_owned(),
            ..Default::default()
        }),
        plan: Some(plan),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let mut stream = grpc
        .server_streaming(
            tonic::Request::new(request),
            tonic::codegen::http::uri::PathAndQuery::from_static(
                "/spark.connect.SparkConnectService/ExecutePlan",
            ),
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

fn sql(query: &str) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(RelType::Sql(pb::Sql {
            query: query.to_owned(),
            ..Default::default()
        })),
    }
}

fn root(query: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(sql(query)))),
    }
}

fn command(c: pb::command::CommandType) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(c),
        }))),
    }
}

const ROWS: &str = "SELECT * FROM VALUES (1, 'a'), (2, 'b'), (3, 'c') AS t(id, name)";

fn write_op(save_type: Option<SaveType>, mode: SaveMode) -> pb::WriteOperation {
    pb::WriteOperation {
        input: Some(sql(ROWS)),
        source: Some("parquet".to_owned()),
        mode: mode as i32,
        save_type,
        ..Default::default()
    }
}

async fn write(
    run: &Running,
    op: pb::WriteOperation,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    collect(
        &run.channel,
        command(pb::command::CommandType::WriteOperation(Box::new(op))),
    )
    .await
}

fn write_v2_op(table: &str, mode: Mode) -> pb::WriteOperationV2 {
    pb::WriteOperationV2 {
        input: Some(sql(ROWS)),
        table_name: table.to_owned(),
        provider: Some("parquet".to_owned()),
        mode: mode as i32,
        ..Default::default()
    }
}

/// The ids the plan `query` answers, sorted.
async fn ids(run: &Running, query: &str) -> Vec<i64> {
    let responses = collect(
        &run.channel,
        root(&format!("SELECT CAST(id AS BIGINT) FROM ({query})")),
    )
    .await
    .expect("the query");
    let mut ids = Vec::new();
    for response in responses {
        if let Some(ResponseType::ArrowBatch(batch)) = response.response_type {
            let reader = arrow_ipc::reader::StreamReader::try_new(
                std::io::Cursor::new(batch.data.to_vec()),
                None,
            )
            .expect("ipc");
            for batch in reader {
                let batch: RecordBatch = batch.expect("batch");
                let column = batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int64Type>();
                ids.extend(column.values().iter().copied());
            }
        }
    }
    ids.sort_unstable();
    ids
}

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp dir")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_parquet_save_is_read_back() {
    let run = start().await;
    let dir = tmp();
    let target = dir.path().join("out").display().to_string();
    let responses = write(
        &run,
        write_op(
            Some(SaveType::Path(target.clone())),
            SaveMode::ErrorIfExists,
        ),
    )
    .await
    .expect("the write");
    assert!(
        responses
            .iter()
            .all(|r| !matches!(r.response_type, Some(ResponseType::ArrowBatch(_)))),
        "a write answers no rows"
    );
    assert_eq!(
        ids(&run, &format!("SELECT id FROM parquet.`{target}`")).await,
        [1, 2, 3]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn save_without_a_path_takes_the_path_option() {
    let run = start().await;
    let dir = tmp();
    let target = dir.path().join("by-option").display().to_string();
    let mut op = write_op(None, SaveMode::ErrorIfExists);
    op.options.insert("PATH".to_owned(), target.clone());
    write(&run, op).await.expect("the write");
    assert_eq!(
        ids(&run, &format!("SELECT id FROM parquet.`{target}`")).await,
        [1, 2, 3]
    );
}

/// The write is a one-shot statement of its operation: the request's operation id and tags reach
/// the engine query (so does its credential: `QueryOptions::inbound_credential`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_writes_query_carries_the_requests_operation_id_and_tags() {
    let run = start().await;
    let dir = tmp();
    let target = dir.path().join("tagged").display().to_string();
    let request = pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        operation_id: Some("7d2f4c1a-9b3e-4a60-8c15-2e6f0a1b3c44".to_owned()),
        tags: vec!["etl".to_owned()],
        plan: Some(command(pb::command::CommandType::WriteOperation(Box::new(
            write_op(Some(SaveType::Path(target)), SaveMode::ErrorIfExists),
        )))),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let mut stream = grpc
        .server_streaming(
            tonic::Request::new(request),
            tonic::codegen::http::uri::PathAndQuery::from_static(
                "/spark.connect.SparkConnectService/ExecutePlan",
            ),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await
        .expect("the call")
        .into_inner();
    while stream.message().await.expect("message").is_some() {}
    let entry = run
        .manager
        .get_or_create(
            &Authenticated::new(Principal::local_user(), None, Method::None),
            SESSION,
            None,
            "u",
        )
        .expect("session");
    let batches = entry
        .session()
        .sql("select count(*) from system.queries where tags['etl'] is not null")
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    assert!(
        batches[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0)
            >= 1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn save_with_a_path_and_a_path_option_is_spark_legacy_1307() {
    let run = start().await;
    let dir = tmp();
    let a = dir.path().join("a").display().to_string();
    let b = dir.path().join("b").display().to_string();
    let mut op = write_op(Some(SaveType::Path(b)), SaveMode::ErrorIfExists);
    op.options.insert("Path".to_owned(), a);
    let status = write(&run, op).await.expect_err("both forms");
    assert_eq!(error_class(&status), "_LEGACY_ERROR_TEMP_1307");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn save_with_neither_a_path_nor_the_option_is_spark_legacy_2050() {
    let run = start().await;
    let status = write(&run, write_op(None, SaveMode::ErrorIfExists))
        .await
        .expect_err("no path");
    assert_eq!(error_class(&status), "_LEGACY_ERROR_TEMP_2050");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn errorifexists_is_the_default_and_append_adds_rows() {
    let run = start().await;
    let dir = tmp();
    let target = dir.path().join("twice").display().to_string();
    write(
        &run,
        write_op(Some(SaveType::Path(target.clone())), SaveMode::Unspecified),
    )
    .await
    .expect("the first write");
    let status = write(
        &run,
        write_op(Some(SaveType::Path(target.clone())), SaveMode::Unspecified),
    )
    .await
    .expect_err("the path exists");
    assert_eq!(error_class(&status), "PATH_ALREADY_EXISTS");
    write(
        &run,
        write_op(Some(SaveType::Path(target.clone())), SaveMode::Append),
    )
    .await
    .expect("the append");
    assert_eq!(
        ids(&run, &format!("SELECT id FROM parquet.`{target}`")).await,
        [1, 1, 2, 2, 3, 3]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn save_as_table_and_insert_into() {
    let run = start().await;
    let table = |method: save_table::TableSaveMethod| {
        Some(SaveType::Table(SaveTable {
            table_name: "written".to_owned(),
            save_method: method as i32,
        }))
    };
    write(
        &run,
        write_op(
            table(save_table::TableSaveMethod::SaveAsTable),
            SaveMode::ErrorIfExists,
        ),
    )
    .await
    .expect("saveAsTable");
    assert_eq!(ids(&run, "SELECT id FROM written").await, [1, 2, 3]);
    write(
        &run,
        write_op(
            table(save_table::TableSaveMethod::InsertInto),
            SaveMode::Append,
        ),
    )
    .await
    .expect("insertInto");
    assert_eq!(
        ids(&run, "SELECT id FROM written").await,
        [1, 1, 2, 2, 3, 3]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_v2_create_then_append() {
    let run = start().await;
    collect(
        &run.channel,
        command(pb::command::CommandType::WriteOperationV2(Box::new(
            write_v2_op("v2_table", Mode::Create),
        ))),
    )
    .await
    .expect("create");
    collect(
        &run.channel,
        command(pb::command::CommandType::WriteOperationV2(Box::new(
            write_v2_op("v2_table", Mode::Append),
        ))),
    )
    .await
    .expect("append");
    assert_eq!(
        ids(&run, "SELECT id FROM v2_table").await,
        [1, 1, 2, 2, 3, 3]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unspecified_table_save_method_is_invalid_enum() {
    let run = start().await;
    let status = write(
        &run,
        write_op(
            Some(SaveType::Table(SaveTable {
                table_name: "t".to_owned(),
                save_method: 0,
            })),
            SaveMode::ErrorIfExists,
        ),
    )
    .await
    .expect_err("unspecified");
    assert_eq!(error_class(&status), "CONNECT_INVALID_PLAN.INVALID_ENUM");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_dataframe_view_registers_a_temp_view() {
    let run = start().await;
    let view_of = |rows: &str, name: &str, is_global: bool, replace: bool| {
        command(pb::command::CommandType::CreateDataframeView(Box::new(
            pb::CreateDataFrameViewCommand {
                input: Some(sql(rows)),
                name: name.to_owned(),
                is_global,
                replace,
            },
        )))
    };
    let view = |name: &str, is_global: bool, replace: bool| view_of(ROWS, name, is_global, replace);
    // Rows that differ from `ROWS`, so a replace that does nothing is seen.
    let other = "SELECT * FROM VALUES (7, 'g'), (8, 'h') AS t(id, name)";
    collect(&run.channel, view("tv", false, false))
        .await
        .expect("createTempView");
    assert_eq!(ids(&run, "SELECT id FROM tv").await, [1, 2, 3]);
    let again = collect(&run.channel, view("tv", false, false))
        .await
        .expect_err("the name is taken");
    assert_eq!(error_class(&again), "TEMP_TABLE_OR_VIEW_ALREADY_EXISTS");
    collect(&run.channel, view_of(other, "tv", false, true))
        .await
        .expect("createOrReplaceTempView");
    assert_eq!(ids(&run, "SELECT id FROM tv").await, [7, 8]);
    collect(&run.channel, view("gv", true, false))
        .await
        .expect("createGlobalTempView");
    assert_eq!(ids(&run, "SELECT id FROM global_temp.gv").await, [1, 2, 3]);
    collect(&run.channel, view_of(other, "gv", true, true))
        .await
        .expect("createOrReplaceGlobalTempView");
    assert_eq!(ids(&run, "SELECT id FROM global_temp.gv").await, [7, 8]);
    // Spark Connect keeps the last part of a two-part name, local or global.
    for (name, global, read) in [
        ("db.`two.parts`", false, "`two.parts`"),
        ("db.gv2", true, "global_temp.gv2"),
    ] {
        collect(&run.channel, view(name, global, false))
            .await
            .expect(name);
        let query = format!("SELECT id FROM {read}");
        assert_eq!(ids(&run, &query).await, [1, 2, 3], "{name}");
    }
    for name in ["a.b.v", "a b", "db.", ""] {
        let refused = collect(&run.channel, view(name, false, false))
            .await
            .expect_err(name);
        assert_eq!(error_class(&refused), "_LEGACY_ERROR_TEMP_1321", "{name}");
    }
}
