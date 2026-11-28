// SPDX-License-Identifier: Apache-2.0
//! `ExecutePlan`, `ReattachExecute` and `ReleaseExecute` end to end over a
//! raw tonic client (design P3-T011a §5, PR-11a).
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]
#![expect(
    clippy::disallowed_methods,
    reason = "the tests read back the server's status details and Arrow data"
)]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_request::RequestOption as ReqOpt;
use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{
    ConnectServer, ConnectServerConfig, ExecuteLimits, OperationState, ProfileChoice,
    SessionLimits, SessionManager,
};
use krishiv_session::Session;
use krishiv_session::auth::{Authenticated, Method};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const OTHER_SESSION: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";
const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";

struct Running {
    manager: Arc<SessionManager>,
    channel: Channel,
    _stop: oneshot::Sender<()>,
}

async fn start_with(execute: ExecuteLimits) -> Running {
    serve(
        SessionManager::from_base(Session::new().expect("session"), SessionLimits::default()),
        execute,
    )
    .await
}

async fn serve(manager: SessionManager, execute: ExecuteLimits) -> Running {
    let manager = Arc::new(manager);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(
        Arc::clone(&manager),
        ConnectServerConfig {
            execute,
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
        manager,
        channel,
        _stop: stop,
    }
}

async fn start() -> Running {
    start_with(ExecuteLimits::default()).await
}

fn path(rpc: &str) -> tonic::codegen::http::uri::PathAndQuery {
    format!("/spark.connect.SparkConnectService/{rpc}")
        .parse()
        .expect("path")
}

async fn unary<Req, Resp>(channel: &Channel, rpc: &str, request: Req) -> Result<Resp, tonic::Status>
where
    Req: prost::Message + Send + Sync + 'static,
    Resp: prost::Message + Default + Send + Sync + 'static,
{
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        tonic::Request::new(request),
        path(rpc),
        tonic_prost::ProstCodec::<Req, Resp>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// Starts `rpc` and returns its response stream.
async fn stream<Req>(
    channel: &Channel,
    rpc: &str,
    request: Req,
) -> Result<tonic::Streaming<pb::ExecutePlanResponse>, tonic::Status>
where
    Req: prost::Message + Send + Sync + 'static,
{
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.server_streaming(
        tonic::Request::new(request),
        path(rpc),
        tonic_prost::ProstCodec::<Req, pb::ExecutePlanResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// Every response of `rpc`, or the status it ended with.
async fn collect(
    channel: &Channel,
    request: pb::ExecutePlanRequest,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let mut s = stream(channel, "ExecutePlan", request).await?;
    let mut out = Vec::new();
    while let Some(r) = s.message().await? {
        out.push(r);
    }
    Ok(out)
}

fn relation(rel: RelType) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(rel),
    }
}

fn range(end: i64) -> pb::Relation {
    relation(RelType::Range(pb::Range {
        start: Some(0),
        end,
        step: 1,
        num_partitions: Some(1),
    }))
}

fn sql(query: &str) -> pb::Relation {
    relation(RelType::Sql(pb::Sql {
        query: query.to_owned(),
        ..Default::default()
    }))
}

fn root(r: pb::Relation) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(r))),
    }
}

fn request(plan: pb::Plan) -> pb::ExecutePlanRequest {
    pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "someone".to_owned(),
            ..Default::default()
        }),
        plan: Some(plan),
        ..Default::default()
    }
}

fn reattachable(mut r: pb::ExecutePlanRequest) -> pb::ExecutePlanRequest {
    r.request_options.push(ReqOpt {
        request_option: Some(RequestOption::ReattachOptions(pb::ReattachOptions {
            reattachable: true,
        })),
    });
    r
}

fn with_id(mut r: pb::ExecutePlanRequest, id: &str) -> pb::ExecutePlanRequest {
    r.operation_id = Some(id.to_owned());
    r
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

/// The `ErrorInfo` metadata of a status.
fn info(status: &tonic::Status) -> BTreeMap<String, String> {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    ErrorInfo::decode(rpc.details[0].value.as_slice())
        .expect("ErrorInfo")
        .metadata
}

fn error_class(status: &tonic::Status) -> String {
    info(status).get("errorClass").cloned().unwrap_or_default()
}

fn batches(responses: &[pb::ExecutePlanResponse]) -> Vec<&pb::execute_plan_response::ArrowBatch> {
    responses
        .iter()
        .filter_map(|r| match &r.response_type {
            Some(ResponseType::ArrowBatch(b)) => Some(b),
            _ => None,
        })
        .collect()
}

fn read_ipc(data: &[u8]) -> Vec<RecordBatch> {
    arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(data.to_vec()), None)
        .expect("ipc")
        .collect::<Result<Vec<_>, _>>()
        .expect("batches")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn execute_response_order() {
    let run = start().await;
    // One engine batch of 25,000 rows, cut at maxRecordsPerBatch (10,000):
    // small engine batches are not merged (design P3-T011a §3.2), so the
    // engine's batch size decides how many come below the limits.
    run.manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session")
        .session()
        .conf()
        .set("krishiv.exec.batchSize", "25000")
        .expect("batch size");
    let responses = collect(&run.channel, reattachable(request(root(range(25_000)))))
        .await
        .expect("execute");
    // schema, 3 batches, metrics, result_complete
    assert_eq!(responses.len(), 6, "{responses:?}");
    let schema = responses[0].schema.as_ref().expect("schema first");
    let pb::data_type::Kind::Struct(s) = schema.kind.as_ref().expect("kind") else {
        panic!("a struct schema");
    };
    assert_eq!(s.fields.len(), 1);
    assert_eq!(s.fields[0].name, "id");
    assert!(matches!(
        s.fields[0].data_type.as_ref().and_then(|t| t.kind.as_ref()),
        Some(pb::data_type::Kind::Long(_))
    ));
    let b = batches(&responses);
    let rows: Vec<i64> = b.iter().map(|b| b.row_count).collect();
    assert_eq!(rows, [10_000, 10_000, 5_000]);
    let offsets: Vec<Option<i64>> = b.iter().map(|b| b.start_offset).collect();
    assert_eq!(offsets, [Some(0), Some(10_000), Some(20_000)]);
    let mut next = 0_i64;
    for batch in &b {
        assert_eq!(batch.chunk_index, None);
        let read = read_ipc(&batch.data);
        assert_eq!(read.len(), 1, "one batch per IPC stream");
        assert_eq!(
            i64::try_from(read[0].num_rows()).expect("rows"),
            batch.row_count
        );
        let ids = read[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("long");
        assert_eq!(ids.value(0), next);
        next += batch.row_count;
    }
    assert!(responses[4].metrics.is_some(), "metrics after the batches");
    assert!(matches!(
        responses[5].response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    let entry = run
        .manager
        .get(&local_caller(), SESSION, None)
        .expect("session");
    let ids: HashSet<&str> = responses.iter().map(|r| r.response_id.as_str()).collect();
    assert_eq!(ids.len(), responses.len(), "fresh response ids");
    let op = &responses[0].operation_id;
    for r in &responses {
        assert_eq!(&r.operation_id, op);
        assert_eq!(r.session_id, SESSION);
        assert_eq!(r.server_side_session_id, entry.server_side_id());
    }
    assert!(krishiv_session::QueryId::parse(op).is_ok(), "a minted UUID");
    // A finished reattachable operation stays until the client releases it
    // (Spark's `ReleaseExecute(release_all)`).
    assert!(entry.operations().get(op).is_some());
    let _: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        pb::ReleaseExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: op.clone(),
            release: Some(pb::release_execute_request::Release::ReleaseAll(
                pb::release_execute_request::ReleaseAll {},
            )),
            ..Default::default()
        },
    )
    .await
    .expect("release all");
    assert!(entry.operations().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_empty_result_is_one_empty_batch_and_no_result_complete_unless_reattachable() {
    let run = start().await;
    let responses = collect(&run.channel, request(root(range(0))))
        .await
        .expect("execute");
    let b = batches(&responses);
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].row_count, 0);
    assert_eq!(b[0].start_offset, Some(0));
    let read = read_ipc(&b[0].data);
    assert_eq!(read.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    assert!(
        !responses
            .iter()
            .any(|r| matches!(r.response_type, Some(ResponseType::ResultComplete(_)))),
        "not reattachable: the stream just ends"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn strings_are_utf8_and_chunks_reassemble() {
    let run = start().await;
    let mut r = request(root(sql(
        "SELECT CAST(id AS STRING) AS s, TIMESTAMP'2024-01-01 00:00:00' AS t FROM range(3000)",
    )));
    r.request_options.push(ReqOpt {
        request_option: Some(RequestOption::ResultChunkingOptions(
            pb::ResultChunkingOptions {
                allow_arrow_batch_chunking: true,
                preferred_arrow_chunk_size: Some(1024),
            },
        )),
    });
    let entry = run
        .manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session");
    entry
        .session()
        .conf()
        .set("spark.sql.session.timeZone", "America/Los_Angeles")
        .expect("tz");
    let responses = collect(&run.channel, r).await.expect("execute");
    // The chunks of the first batch.
    let b: Vec<_> = batches(&responses)
        .into_iter()
        .filter(|b| b.start_offset == Some(0))
        .collect();
    let rows = b[0].row_count;
    assert!(rows > 0);
    let n = b[0].num_chunks_in_batch.expect("chunked");
    assert!(n > 1, "{n}");
    let mut data = Vec::new();
    for (i, chunk) in b.iter().enumerate() {
        assert_eq!(chunk.chunk_index, Some(i64::try_from(i).expect("i")));
        assert_eq!(chunk.num_chunks_in_batch, Some(n));
        assert_eq!(chunk.row_count, rows);
        assert!(chunk.data.len() <= 1024);
        data.extend_from_slice(&chunk.data);
    }
    let read = read_ipc(&data);
    assert_eq!(i64::try_from(read[0].num_rows()).expect("rows"), rows);
    let schema = read[0].schema();
    assert_eq!(schema.field(0).data_type(), &arrow_schema::DataType::Utf8);
    assert_eq!(
        schema.field(1).data_type(),
        &arrow_schema::DataType::Timestamp(
            arrow_schema::TimeUnit::Microsecond,
            Some("America/Los_Angeles".into())
        )
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compressed_ipc_buffers_follow_the_session_codec() {
    let run = start().await;
    let entry = run
        .manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session");
    for (codec, expected) in [
        ("zstd", arrow_ipc::CompressionType::ZSTD),
        ("lz4", arrow_ipc::CompressionType::LZ4_FRAME),
    ] {
        entry
            .session()
            .conf()
            .set("spark.sql.execution.arrow.compression.codec", codec)
            .expect("codec");
        let responses = collect(&run.channel, request(root(range(100))))
            .await
            .expect("execute");
        let b = batches(&responses);
        let read = read_ipc(&b[0].data);
        assert_eq!(read[0].num_rows(), 100, "{codec}");
        assert_eq!(ipc_codecs(&b[0].data), vec![Some(expected)], "{codec}");
    }
}

/// The buffer compression of each record batch of an IPC stream.
fn ipc_codecs(data: &[u8]) -> Vec<Option<arrow_ipc::CompressionType>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 8 <= data.len() {
        if data[pos..pos + 4] == [0xff; 4] {
            pos += 4;
        }
        let len = i32::from_le_bytes(data[pos..pos + 4].try_into().expect("length"));
        let len = usize::try_from(len).expect("length");
        pos += 4;
        if len == 0 {
            break;
        }
        let message = arrow_ipc::root_as_message(&data[pos..pos + len]).expect("message");
        pos += len + usize::try_from(message.bodyLength()).expect("body");
        if let Some(batch) = message.header_as_record_batch() {
            out.push(batch.compression().map(|c| c.codec()));
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compressed_plans_decode_and_a_bomb_fails_before_it_expands() {
    let run = start_with(ExecuteLimits {
        max_plan_size: 4096,
        ..ExecuteLimits::default()
    })
    .await;
    let compressed = |data: Vec<u8>, op: i32, codec: i32| {
        request(pb::Plan {
            op_type: Some(OpType::CompressedOperation(Box::new(
                pb::plan::CompressedOperation {
                    data,
                    op_type: op,
                    compression_codec: codec,
                },
            ))),
        })
    };
    let zstd_codec = pb::CompressionCodec::Zstd as i32;
    let rel = pb::plan::compressed_operation::OpType::Relation as i32;
    let data = zstd::bulk::compress(&range(7).encode_to_vec(), 3).expect("zstd");
    let responses = collect(&run.channel, compressed(data, rel, zstd_codec))
        .await
        .expect("execute");
    assert_eq!(batches(&responses)[0].row_count, 7);

    // A frame that declares its size: refused from the header.
    let bomb = zstd::bulk::compress(&vec![0_u8; 1 << 20], 3).expect("zstd");
    assert!(bomb.len() < 1024);
    let e = collect(&run.channel, compressed(bomb, rel, zstd_codec))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.PLAN_SIZE_LARGER_THAN_MAX"
    );
    assert!(e.message().contains("1048576 vs. 4096"), "{}", e.message());
    // Exactly the maximum: refused, as Spark's bounded stream refuses it
    // (the parser asks for more once `max` bytes were read).
    let exact = zstd::bulk::compress(&[0_u8; 4096], 3).expect("zstd");
    let e = collect(&run.channel, compressed(exact, rel, zstd_codec))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.PLAN_SIZE_LARGER_THAN_MAX"
    );
    assert!(e.message().contains("unknown vs. 4096"), "{}", e.message());
    // A streamed frame without a declared size: refused at the bound.
    let streamed = {
        let mut enc = zstd::stream::write::Encoder::new(Vec::new(), 3).expect("encoder");
        std::io::Write::write_all(&mut enc, &vec![0_u8; 1 << 20]).expect("write");
        enc.finish().expect("finish")
    };
    let e = collect(&run.channel, compressed(streamed, rel, zstd_codec))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.PLAN_SIZE_LARGER_THAN_MAX"
    );
    assert!(e.message().contains("unknown vs. 4096"), "{}", e.message());

    let e = collect(&run.channel, compressed(vec![1, 2, 3], rel, 0))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.INVALID_ENUM");
    let e = collect(&run.channel, compressed(vec![1, 2, 3], rel, zstd_codec))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.CANNOT_PARSE");
    let empty = zstd::bulk::compress(&[], 3).expect("zstd");
    // An unset operation type: Spark wraps the parse error in CANNOT_PARSE.
    let e = collect(&run.channel, compressed(empty.clone(), 0, zstd_codec))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.CANNOT_PARSE");
    assert!(
        e.message().contains("OP_TYPE_UNSPECIFIED"),
        "{}",
        e.message()
    );
    let command = pb::Command {
        command_type: Some(pb::command::CommandType::GetResourcesCommand(Box::new(
            pb::GetResourcesCommand {},
        ))),
    };
    let data = zstd::bulk::compress(&command.encode_to_vec(), 3).expect("zstd");
    let cmd = pb::plan::compressed_operation::OpType::Command as i32;
    let e = collect(&run.channel, compressed(data, cmd, zstd_codec))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_requests_are_refused_with_sparks_classes() {
    let run = start().await;
    let e = collect(&run.channel, with_id(request(root(range(1))), "nope"))
        .await
        .unwrap_err();
    assert_eq!(e.code(), Code::Internal);
    assert_eq!(error_class(&e), "INVALID_HANDLE.FORMAT");
    let mut tagged = request(root(range(1)));
    tagged.tags = vec!["a,b".to_owned()];
    let e = collect(&run.channel, tagged).await.unwrap_err();
    assert_eq!(error_class(&e), "KRISHIV_INVALID_OPERATION_TAG");
    assert_eq!(info(&e)["sqlState"], "22023");
    let e = collect(&run.channel, request(pb::Plan::default()))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET"
    );
    let mut no_plan = request(pb::Plan::default());
    no_plan.plan = None;
    let e = collect(&run.channel, no_plan).await.unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET"
    );
    let command = request(pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(pb::command::CommandType::GetResourcesCommand(Box::new(
                pb::GetResourcesCommand {},
            ))),
        }))),
    });
    let e = collect(&run.channel, command).await.unwrap_err();
    assert_eq!(info(&e)["sqlState"], "56K00");
    // An analysis error: AnalysisException, so PySpark raises its own.
    let e = collect(
        &run.channel,
        request(root(sql("SELECT * FROM no_such_table"))),
    )
    .await
    .unwrap_err();
    assert_eq!(e.code(), Code::Internal);
    let meta = info(&e);
    assert_eq!(meta["errorClass"], "TABLE_OR_VIEW_NOT_FOUND");
    assert!(
        meta["classes"].starts_with(r#"["org.apache.spark.sql.AnalysisException""#),
        "{}",
        meta["classes"]
    );
    let entry = run
        .manager
        .get(&local_caller(), SESSION, None)
        .expect("session");
    assert!(
        entry.operations().is_empty(),
        "a failed start leaves nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_finished_operation_id_is_abandoned_for_reuse() {
    let run = start().await;
    collect(&run.channel, with_id(request(root(range(1))), OP))
        .await
        .expect("first");
    let e = collect(&run.channel, with_id(request(root(range(1))), OP))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_ABANDONED");
}

/// A query that streams far more rows than the test ever reads.
fn endless() -> pb::Plan {
    root(range(100_000_000_000))
}

/// Waits until `cond` holds (or fails after 10 s).
async fn eventually(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(tokio::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Waits until the encoder of `op` waits for its client to read (or fails
/// after 20 s: filling the windows takes a while on a loaded machine).
async fn blocked_on_the_client(op: &krishiv_connect_server::operation::Operation) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while !op.is_waiting_for_client() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the encoder never waits ({} responses)",
            op.sent_count()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_operations_reattach_release_and_cap() {
    let run = start_with(ExecuteLimits {
        max_operations_per_session: 1,
        ..ExecuteLimits::default()
    })
    .await;
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        with_id(reattachable(request(endless())), OP),
    )
    .await
    .expect("execute");
    let first = live.message().await.expect("ok").expect("schema");
    assert!(first.schema.is_some());

    // The cap: RESOURCE_EXHAUSTED.
    let e = collect(&run.channel, request(root(range(1))))
        .await
        .unwrap_err();
    assert_eq!(e.code(), Code::ResourceExhausted);
    assert_eq!(error_class(&e), "KRISHIV_OPERATION_LIMIT");
    // The same id with another plan: ALREADY_EXISTS (the same plan
    // reattaches, `tests/reattach.rs`).
    let e = collect(&run.channel, with_id(request(root(range(1))), OP))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_ALREADY_EXISTS");

    let reattach = |session: &str, op: &str| pb::ReattachExecuteRequest {
        session_id: session.to_owned(),
        operation_id: op.to_owned(),
        ..Default::default()
    };
    let e = stream(
        &run.channel,
        "ReattachExecute",
        reattach(SESSION, "00112233-4455-6677-8899-000000000000"),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_NOT_FOUND");
    // Another session never sees the operation (operation_bound_to_session).
    run.manager
        .get_or_create(&local_caller(), OTHER_SESSION, None, "u")
        .expect("other");
    let e = stream(&run.channel, "ReattachExecute", reattach(OTHER_SESSION, OP))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_NOT_FOUND");
    let e = stream(
        &run.channel,
        "ReattachExecute",
        reattach("8f0e4c55-0000-4d1e-9a3c-3a1b0b2c7d10", OP),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");

    let release = |op: &str, how: pb::release_execute_request::Release| pb::ReleaseExecuteRequest {
        session_id: SESSION.to_owned(),
        operation_id: op.to_owned(),
        release: Some(how),
        ..Default::default()
    };
    use pb::release_execute_request::{Release, ReleaseAll, ReleaseUntil};
    let until = |id: &str| {
        Release::ReleaseUntil(ReleaseUntil {
            response_id: id.to_owned(),
        })
    };
    // release_until a response it sent: success; another id: not found.
    let ok: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        release(OP, until(&first.response_id)),
    )
    .await
    .expect("release until");
    assert_eq!(ok.operation_id.as_deref(), Some(OP));
    assert_eq!(ok.session_id, SESSION);
    assert_eq!(ok.server_side_session_id, first.server_side_session_id);
    let e = unary::<_, pb::ReleaseExecuteResponse>(
        &run.channel,
        "ReleaseExecute",
        release(OP, until("not-sent")),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_CURSOR.POSITION_NOT_FOUND");
    let mut unset = release(OP, Release::ReleaseAll(ReleaseAll {}));
    unset.release = None;
    let e = unary::<_, pb::ReleaseExecuteResponse>(&run.channel, "ReleaseExecute", unset)
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET"
    );
    // An unknown operation: success without an operation id.
    let ok: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        release(
            "00112233-4455-6677-8899-000000000000",
            Release::ReleaseAll(ReleaseAll {}),
        ),
    )
    .await
    .expect("unknown");
    assert_eq!(ok.operation_id, None);

    // release_all closes the live operation: its stream is disconnected
    // (Spark's `ExecuteHolder.close` interrupts the attached sender).
    let entry = run
        .manager
        .get(&local_caller(), SESSION, None)
        .expect("session");
    let session = entry.session().clone();
    let _: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        release(OP, Release::ReleaseAll(ReleaseAll {})),
    )
    .await
    .expect("release all");
    assert!(entry.operations().is_empty());
    let status = loop {
        match live.message().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the stream ended without an error"),
            Err(s) => break s,
        }
    };
    assert_eq!(error_class(&status), "INVALID_CURSOR.DISCONNECTED");
    eventually("the query stops", || session.running_queries() == 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_live_non_reattachable_operation_is_not_reattachable() {
    let run = start().await;
    let mut live = stream(&run.channel, "ExecutePlan", with_id(request(endless()), OP))
        .await
        .expect("execute");
    live.message().await.expect("ok").expect("schema");
    let e = stream(
        &run.channel,
        "ReattachExecute",
        pb::ReattachExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: OP.to_owned(),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_CURSOR.NOT_REATTACHABLE");
    let e = unary::<_, pb::ReleaseExecuteResponse>(
        &run.channel,
        "ReleaseExecute",
        pb::ReleaseExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: OP.to_owned(),
            release: Some(pb::release_execute_request::Release::ReleaseAll(
                pb::release_execute_request::ReleaseAll {},
            )),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_CURSOR.NOT_REATTACHABLE");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_session_closes_its_operations() {
    let run = start().await;
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        reattachable(request(endless())),
    )
    .await
    .expect("execute");
    live.message().await.expect("ok").expect("schema");
    let _: pb::ReleaseSessionResponse = unary(
        &run.channel,
        "ReleaseSession",
        pb::ReleaseSessionRequest {
            session_id: SESSION.to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("release");
    // Spark's `removeAllExecutionsForSession` closes them: the attached
    // sender is interrupted.
    let status = loop {
        match live.message().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the stream ended without an error"),
            Err(s) => break s,
        }
    };
    assert_eq!(error_class(&status), "INVALID_CURSOR.DISCONNECTED");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_stream_cancels_the_query_and_frees_its_memory() {
    let run = start_with(ExecuteLimits {
        ahead_bytes: 64 * 1024,
        ..ExecuteLimits::default()
    })
    .await;
    let entry = run
        .manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session");
    let session = entry.session().clone();
    session
        .conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .expect("conf");
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let mut live = stream(&run.channel, "ExecutePlan", with_id(request(endless()), OP))
        .await
        .expect("execute");
    live.message().await.expect("ok").expect("schema");
    let op = entry.operations().get(OP).expect("live");
    // Backpressure: nothing is read, so the encoder stops once the budget
    // and the transport's windows are full (about 8 KB per response) and
    // waits for room (it never would without backpressure).
    blocked_on_the_client(&op).await;
    let ahead = op.sent_count();
    // 64 KiB of budget plus the HTTP/2 windows and tonic's buffer, far from
    // the 100 million rows the query would produce.
    assert!(ahead < 2_000, "{ahead} responses produced ahead");
    // backpressure_no_materialise: the responses waiting for the client
    // never hold more than the budget plus one response (about 8 KB here).
    let peak = op.buffered_peak();
    assert!(peak > 0);
    assert!(peak <= 64 * 1024 + 16 * 1024, "{peak} bytes buffered");
    drop(live);
    eventually("the operation is removed", || entry.operations().is_empty()).await;
    eventually("the query stops", || session.running_queries() == 0).await;
    eventually("the responses' memory returns", || {
        memory.reserved() == before
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_stream_cancels_a_query_that_has_not_produced_yet() {
    let run = start().await;
    let entry = run
        .manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session");
    let session = entry.session().clone();
    // An aggregate sends nothing until it has read its whole input.
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        with_id(
            request(root(sql(
                "SELECT count(*) FROM range(0, 100000000000, 1, 4)",
            ))),
            OP,
        ),
    )
    .await
    .expect("execute");
    live.message().await.expect("ok").expect("schema");
    eventually("the query runs", || session.running_queries() == 1).await;
    drop(live);
    eventually("the query is cancelled", || session.running_queries() == 0).await;
    eventually("the operation leaves the map", || {
        entry.operations().is_empty()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_released_operation_does_not_wait_for_a_client_that_reads_nothing() {
    let run = start_with(ExecuteLimits {
        ahead_bytes: 64 * 1024,
        ..ExecuteLimits::default()
    })
    .await;
    let entry = run
        .manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session");
    let session = entry.session().clone();
    session
        .conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "1")
        .expect("conf");
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        with_id(reattachable(request(endless())), OP),
    )
    .await
    .expect("execute");
    live.message().await.expect("ok").expect("schema");
    let op = entry.operations().get(OP).expect("live");
    // Tiny responses fill the ahead budget and the transport.
    blocked_on_the_client(&op).await;
    let _: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        pb::ReleaseExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: OP.to_owned(),
            release: Some(pb::release_execute_request::Release::ReleaseAll(
                pb::release_execute_request::ReleaseAll {},
            )),
            ..Default::default()
        },
    )
    .await
    .expect("release all");
    // The encoder ends and the buffer is dropped while the client still
    // reads nothing.
    eventually("the query stops", || session.running_queries() == 0).await;
    eventually("the buffer's memory returns", || {
        memory.reserved() == before
    })
    .await;
    drop(live);
}

/// large_row_result, scaled down: a row larger than the batch limit (here
/// 2 KiB instead of 128 MiB) is sent as one batch of one row without
/// chunking, and in chunks of the session's maximum with it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn large_row_result() {
    let run = start_with(ExecuteLimits {
        max_batch_size: 2048,
        ..ExecuteLimits::default()
    })
    .await;
    let query = || request(root(sql("SELECT repeat('x', 65536) AS s FROM range(2)")));
    let responses = collect(&run.channel, query()).await.expect("execute");
    let b = batches(&responses);
    assert_eq!(b.len(), 2, "one row per batch: the first row always goes");
    for batch in &b {
        assert_eq!(batch.row_count, 1);
        assert!(batch.data.len() > 65_536);
        assert_eq!(batch.num_chunks_in_batch, None);
    }
    run.manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session")
        .session()
        .conf()
        .set("spark.connect.session.resultChunking.maxChunkSize", "4096")
        .expect("conf");
    let mut chunked = query();
    chunked.request_options.push(ReqOpt {
        request_option: Some(RequestOption::ResultChunkingOptions(
            pb::ResultChunkingOptions {
                allow_arrow_batch_chunking: true,
                preferred_arrow_chunk_size: None,
            },
        )),
    });
    let responses = collect(&run.channel, chunked).await.expect("execute");
    let first: Vec<_> = batches(&responses)
        .into_iter()
        .filter(|b| b.start_offset == Some(0))
        .collect();
    assert!(first.len() > 16, "{}", first.len());
    let mut data = Vec::new();
    for chunk in &first {
        assert!(chunk.data.len() <= 4096);
        data.extend_from_slice(&chunk.data);
    }
    let read = read_ipc(&data);
    let s = read[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .expect("utf8");
    assert_eq!(s.value(0).len(), 65_536);
}

// ---- local relations, schemas, qualifiers and expression strings (P3-T012a part 2b) --------

/// An Arrow IPC stream of `(id: Int64, name: Utf8)` rows, as PySpark's `createDataFrame` sends.
fn people_ipc() -> Vec<u8> {
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .expect("batch");
    let mut out = Vec::new();
    {
        let mut w = arrow_ipc::writer::StreamWriter::try_new(&mut out, &schema).expect("writer");
        w.write(&batch).expect("write");
        w.finish().expect("finish");
    }
    out
}

/// An Arrow IPC stream of `(id: Int64, name: Utf8)` rows other than [`people_ipc`]'s.
fn others_ipc() -> Vec<u8> {
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![10, 20])),
            Arc::new(StringArray::from(vec![Some("x"), Some("y")])),
        ],
    )
    .expect("batch");
    let mut out = Vec::new();
    {
        let mut w = arrow_ipc::writer::StreamWriter::try_new(&mut out, &schema).expect("writer");
        w.write(&batch).expect("write");
        w.finish().expect("finish");
    }
    out
}

fn local(data: Option<Vec<u8>>, schema: Option<&str>) -> pb::Relation {
    relation(RelType::LocalRelation(pb::LocalRelation {
        data,
        schema: schema.map(str::to_owned),
    }))
}

/// The one Arrow batch of a small result.
fn rows(responses: &[pb::ExecutePlanResponse]) -> RecordBatch {
    let mut all: Vec<RecordBatch> = batches(responses)
        .iter()
        .flat_map(|b| read_ipc(&b.data))
        .collect();
    assert_eq!(all.len(), 1, "a small result is one batch");
    all.remove(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_local_relation_is_collected() {
    let run = start().await;
    let r = collect(&run.channel, request(root(local(Some(people_ipc()), None))))
        .await
        .expect("runs");
    let got = rows(&r);
    assert_eq!(got.num_rows(), 3);
    assert_eq!(got.schema().field(0).name(), "id");
    assert_eq!(got.schema().field(1).name(), "name");
    // The batches live once, in the session's store, held by the plan's lease.
    let again = collect(&run.channel, request(root(local(Some(people_ipc()), None))))
        .await
        .expect("runs again");
    assert_eq!(rows(&again).num_rows(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_local_relation_applies_its_schema_eagerly() {
    let run = start().await;
    // The user schema renames by position and casts `id` to int.
    let r = collect(
        &run.channel,
        request(root(local(Some(people_ipc()), Some("a INT, b STRING")))),
    )
    .await
    .expect("runs");
    let got = rows(&r);
    assert_eq!(got.schema().field(0).name(), "a");
    assert_eq!(
        got.schema().field(0).data_type(),
        &arrow_schema::DataType::Int32
    );
    assert_eq!(got.num_rows(), 3);
    // A schema of a different column count fails while the plan is built.
    let e = collect(
        &run.channel,
        request(root(local(Some(people_ipc()), Some("a INT")))),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "NUM_COLUMNS_MISMATCH");
    // A schema alone is an empty relation; neither is invalid.
    let empty = collect(
        &run.channel,
        request(root(local(None, Some("a INT, b STRING")))),
    )
    .await
    .expect("runs");
    assert_eq!(rows(&empty).num_rows(), 0);
    let e = collect(&run.channel, request(root(local(None, None))))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.SCHEMA_REQUIRED_FOR_LOCAL_RELATION"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cached_local_relation_without_its_artifact_cannot_be_found() {
    let run = start().await;
    let cached = relation(RelType::CachedLocalRelation(pb::CachedLocalRelation {
        hash: "0123abcd".to_owned(),
    }));
    let e = collect(&run.channel, request(root(cached)))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.CANNOT_FIND_CACHED_LOCAL_RELATION"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_local_relations_join_in_one_plan() {
    let run = start().await;
    let join = relation(RelType::Join(Box::new(pb::Join {
        left: Some(Box::new(local(Some(people_ipc()), None))),
        right: Some(Box::new(local(
            Some(others_ipc()),
            Some("id2 BIGINT, n2 STRING"),
        ))),
        join_type: pb::join::JoinType::Cross as i32,
        ..Default::default()
    })));
    let r = collect(&run.channel, request(root(join)))
        .await
        .expect("runs");
    let out = rows(&r);
    // The two relations are bound to their own data: 3 left rows by 2 right rows, in the left's
    // columns then the right's, with the right's names from its own schema text.
    let names: Vec<_> = out
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["id", "name", "id2", "n2"]);
    let left = out
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>();
    let left_names = out.column(1).as_string::<i32>();
    let right = out
        .column(2)
        .as_primitive::<arrow_array::types::Int64Type>();
    let right_names = out.column(3).as_string::<i32>();
    let mut tuples: Vec<(i64, Option<String>, i64, String)> = (0..out.num_rows())
        .map(|i| {
            (
                left.value(i),
                (!left_names.is_null(i)).then(|| left_names.value(i).to_owned()),
                right.value(i),
                right_names.value(i).to_owned(),
            )
        })
        .collect();
    tuples.sort();
    let expected: Vec<(i64, Option<String>, i64, String)> = [
        (1, Some("a"), 10, "x"),
        (1, Some("a"), 20, "y"),
        (2, None, 10, "x"),
        (2, None, 20, "y"),
        (3, Some("c"), 10, "x"),
        (3, Some("c"), 20, "y"),
    ]
    .into_iter()
    .map(|(a, b, c, d)| (a, b.map(str::to_owned), c, d.to_owned()))
    .collect();
    assert_eq!(tuples, expected);
}

fn column(name: &str) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::UnresolvedAttribute(
            pb::expression::UnresolvedAttribute {
                unparsed_identifier: name.to_owned(),
                plan_id: None,
                is_metadata_column: None,
            },
        )),
    }
}

fn expression_string(text: &str) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::ExpressionString(
            pb::expression::ExpressionString {
                expression: text.to_owned(),
            },
        )),
    }
}

fn project(input: pb::Relation, exprs: Vec<pb::Expression>) -> pb::Relation {
    relation(RelType::Project(Box::new(pb::Project {
        input: Some(Box::new(input)),
        expressions: exprs,
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_expression_string_runs_as_select_expr() {
    let run = start().await;
    let plan = project(
        local(Some(people_ipc()), None),
        vec![expression_string("id * 10 AS tens")],
    );
    let r = collect(&run.channel, request(root(plan)))
        .await
        .expect("runs");
    let got = rows(&r);
    assert_eq!(got.schema().field(0).name(), "tens");
    let values = got
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("longs");
    assert_eq!(values.values().to_vec(), vec![10, 20, 30]);
    let e = collect(
        &run.channel,
        request(root(project(
            local(Some(people_ipc()), None),
            vec![expression_string("id +")],
        ))),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "PARSE_SYNTAX_ERROR");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subquery_alias_qualifier_qualifies_the_columns() {
    let run = start().await;
    let aliased = |qualifier: &[&str]| {
        relation(RelType::SubqueryAlias(Box::new(pb::SubqueryAlias {
            input: Some(Box::new(local(Some(people_ipc()), None))),
            alias: "t".to_owned(),
            qualifier: qualifier.iter().map(|q| (*q).to_owned()).collect(),
        })))
    };
    // `db.t.id` and `t.id` both name the column of `db.t`.
    for column_name in ["db.t.id", "t.id", "id"] {
        let r = collect(
            &run.channel,
            request(root(project(aliased(&["db"]), vec![column(column_name)]))),
        )
        .await
        .unwrap_or_else(|e| panic!("{column_name}: {e}"));
        assert_eq!(rows(&r).num_rows(), 3, "{column_name}");
    }
    // The qualifier is not a prefix of another relation's name.
    let e = collect(
        &run.channel,
        request(root(project(aliased(&["db"]), vec![column("other.t.id")]))),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "UNRESOLVED_COLUMN.WITH_SUGGESTION");
    // Without a qualifier `db.t.id` does not resolve.
    let e = collect(
        &run.channel,
        request(root(project(aliased(&[]), vec![column("db.t.id")]))),
    )
    .await
    .unwrap_err();
    assert!(
        error_class(&e).starts_with("UNRESOLVED_COLUMN"),
        "{}",
        error_class(&e)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_data_source_schema_declares_the_columns() {
    let dir = tempfile::tempdir().expect("dir");
    let path = dir.path().join("t.csv");
    std::fs::write(&path, "1,x\n2,y\n").expect("write");
    let read = relation(RelType::Read(pb::Read {
        is_streaming: false,
        read_type: Some(pb::read::ReadType::DataSource(pb::read::DataSource {
            format: Some("csv".to_owned()),
            schema: Some("n INT, s STRING".to_owned()),
            paths: vec![path.to_string_lossy().into_owned()],
            ..Default::default()
        })),
    }));
    let run = start().await;
    let r = collect(&run.channel, request(root(read)))
        .await
        .expect("runs");
    let got = rows(&r);
    assert_eq!(got.schema().field(0).name(), "n");
    assert_eq!(
        got.schema().field(0).data_type(),
        &arrow_schema::DataType::Int32
    );
    assert_eq!(got.num_rows(), 2);
}

fn sql_command(query: &str) -> pb::Plan {
    #[expect(deprecated, reason = "the field every client version sends")]
    let command = pb::SqlCommand {
        sql: query.to_owned(),
        ..Default::default()
    };
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(pb::command::CommandType::SqlCommand(Box::new(command))),
        }))),
    }
}

/// The command PySpark 4.x sends for `spark.sql(query)`: the statement as `input`, a `SQL`
/// relation (the deprecated fields unset).
fn sql_command_input(query: &str, plan_id: i64) -> pb::Plan {
    let command = pb::SqlCommand {
        input: Some(pb::Relation {
            common: Some(pb::RelationCommon {
                plan_id: Some(plan_id),
                ..Default::default()
            }),
            rel_type: Some(RelType::Sql(pb::Sql {
                query: query.to_owned(),
                ..Default::default()
            })),
        }),
        ..Default::default()
    };
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(pb::command::CommandType::SqlCommand(Box::new(command))),
        }))),
    }
}

/// The relation of the one `sql_command_result` response.
fn sql_result(responses: &[pb::ExecutePlanResponse]) -> pb::Relation {
    let results: Vec<_> = responses
        .iter()
        .filter_map(|r| match &r.response_type {
            Some(ResponseType::SqlCommandResult(c)) => c.relation.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1, "one sql_command_result");
    assert!(
        batches(responses).is_empty(),
        "a SQL command streams no batch"
    );
    results.into_iter().next().expect("one")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sql_query_answers_the_sql_relation_the_client_sends_again() {
    let run = start().await;
    let r = collect(
        &run.channel,
        reattachable(request(sql_command("SELECT 1 AS a, 'x' AS b"))),
    )
    .await
    .expect("runs");
    // Only the result and `result_complete`: no schema, batch or metrics.
    assert_eq!(r.len(), 2, "{r:?}");
    assert!(matches!(
        r[1].response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    let relation = sql_result(&r);
    let Some(RelType::Sql(sql)) = &relation.rel_type else {
        panic!("a query answers a SQL relation: {relation:?}");
    };
    assert_eq!(sql.query, "SELECT 1 AS a, 'x' AS b");
    // The client sends it as a root relation.
    let again = collect(&run.channel, request(root(relation)))
        .await
        .expect("runs");
    let got = rows(&again);
    assert_eq!(got.num_rows(), 1);
    assert_eq!(got.schema().field(0).name(), "a");
    // An analysis error is raised by `spark.sql` itself, as Spark does.
    let e = collect(
        &run.channel,
        request(sql_command("SELECT * FROM no_such_table")),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "TABLE_OR_VIEW_NOT_FOUND");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sql_command_runs_at_once_and_answers_its_rows_as_a_local_relation() {
    let run = start().await;
    let r = collect(
        &run.channel,
        request(sql_command("CREATE TEMPORARY VIEW v AS SELECT 7 AS n")),
    )
    .await
    .expect("runs");
    // Spark's order for a command: the result, then metrics (then `result_complete`).
    assert!(
        matches!(r[0].response_type, Some(ResponseType::SqlCommandResult(_))),
        "{r:?}"
    );
    assert!(r[1].metrics.is_some(), "{r:?}");
    let relation = sql_result(&r);
    assert!(
        matches!(relation.rel_type, Some(RelType::LocalRelation(_))),
        "{relation:?}"
    );
    // It ran: the view is there for the next request, and no later request creates it again.
    let entry = run
        .manager
        .get(&local_caller(), SESSION, None)
        .expect("session");
    let df = entry
        .session()
        .sql("SELECT n FROM v")
        .await
        .expect("the view exists");
    assert_eq!(df.collect().await.expect("rows").len(), 1);
    // `SET key=value` returns its rows (key, value) as a local relation the client re-sends.
    let r = collect(
        &run.channel,
        request(sql_command("SET spark.sql.shuffle.partitions=3")),
    )
    .await
    .expect("runs");
    let relation = sql_result(&r);
    let again = collect(&run.channel, request(root(relation)))
        .await
        .expect("a command's relation runs");
    let got = rows(&again);
    assert_eq!(got.num_rows(), 1);
    assert_eq!(got.num_columns(), 2);
    assert_eq!(
        entry.session().conf().get("spark.sql.shuffle.partitions"),
        Some("3".to_owned())
    );
    // A failing command is a failed start.
    let e = collect(&run.channel, request(sql_command("DROP VIEW nope_view")))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "TABLE_OR_VIEW_NOT_FOUND");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_compressed_sql_command_runs_and_unsupported_forms_are_refused() {
    let run = start().await;
    #[expect(deprecated, reason = "the field every client version sends")]
    let command = pb::Command {
        command_type: Some(pb::command::CommandType::SqlCommand(Box::new(
            pb::SqlCommand {
                sql: "SELECT 2".to_owned(),
                ..Default::default()
            },
        ))),
    };
    let data = zstd::bulk::compress(&command.encode_to_vec(), 3).expect("zstd");
    let plan = pb::Plan {
        op_type: Some(OpType::CompressedOperation(Box::new(
            pb::plan::CompressedOperation {
                data,
                op_type: pb::plan::compressed_operation::OpType::Command as i32,
                compression_codec: pb::CompressionCodec::Zstd as i32,
            },
        ))),
    };
    let r = collect(&run.channel, request(plan)).await.expect("runs");
    assert!(matches!(sql_result(&r).rel_type, Some(RelType::Sql(_))));
    // Parameters are refused up front (golden-pending: group 4).
    let mut with_args = sql_command("SELECT ?");
    if let Some(OpType::Command(c)) = &mut with_args.op_type
        && let Some(pb::command::CommandType::SqlCommand(s)) = &mut c.command_type
    {
        #[expect(deprecated, reason = "the field every client version sends")]
        {
            s.pos_arguments = vec![column("x")];
        }
    }
    let e = collect(&run.channel, request(with_args)).await.unwrap_err();
    assert_eq!(error_class(&e), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pyspark_4_sql_command_sends_its_statement_as_input() {
    let run = start().await;
    // A query: the `input` relation comes back as it was, plan id included.
    let r = collect(
        &run.channel,
        reattachable(request(sql_command_input("SELECT 1 AS a", 41))),
    )
    .await
    .expect("runs");
    assert_eq!(r.len(), 2, "{r:?}");
    let relation = sql_result(&r);
    assert_eq!(relation.common.as_ref().and_then(|c| c.plan_id), Some(41));
    assert!(
        matches!(&relation.rel_type, Some(RelType::Sql(s)) if s.query == "SELECT 1 AS a"),
        "{relation:?}"
    );
    // A command through `input` runs at once.
    let r = collect(
        &run.channel,
        request(sql_command_input(
            "CREATE TEMPORARY VIEW iv AS SELECT 5 AS n",
            42,
        )),
    )
    .await
    .expect("runs");
    assert!(matches!(
        sql_result(&r).rel_type,
        Some(RelType::LocalRelation(_))
    ));
    let entry = run
        .manager
        .get(&local_caller(), SESSION, None)
        .expect("session");
    let df = entry
        .session()
        .sql("SELECT n FROM iv")
        .await
        .expect("the view exists");
    assert_eq!(df.collect().await.expect("rows").len(), 1);
    // Another input relation is refused with Spark's class; WithRelations is group 4.
    let mut other = sql_command_input("x", 1);
    if let Some(OpType::Command(c)) = &mut other.op_type
        && let Some(pb::command::CommandType::SqlCommand(s)) = &mut c.command_type
    {
        s.input = Some(range(1));
    }
    let e = collect(&run.channel, request(other)).await.unwrap_err();
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.SQL_COMMAND_EXPECTS_SQL_OR_WITH_RELATIONS"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_result_over_the_batch_cap_is_refused() {
    let run = start_with(ExecuteLimits {
        max_batch_size: 64,
        ..ExecuteLimits::default()
    })
    .await;
    let e = collect(&run.channel, request(sql_command("SHOW TABLES")))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "KRISHIV_UNSUPPORTED_FEATURE");
}

fn literal_long(v: i64) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::Literal(pb::expression::Literal {
            literal_type: Some(pb::expression::literal::LiteralType::Long(v)),
            data_type: None,
        })),
    }
}

fn aliased_count(alias: &str) -> pb::Expression {
    let call = pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::UnresolvedFunction(
            pb::expression::UnresolvedFunction {
                function_name: "count".to_owned(),
                arguments: vec![literal_long(1)],
                ..Default::default()
            },
        )),
    };
    pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::Alias(Box::new(
            pb::expression::Alias {
                expr: Some(Box::new(call)),
                name: vec![alias.to_owned()],
                metadata: None,
            },
        ))),
    }
}

fn observed(input: pb::Relation, plan_id: i64) -> pb::Relation {
    pb::Relation {
        common: Some(pb::RelationCommon {
            plan_id: Some(plan_id),
            ..Default::default()
        }),
        rel_type: Some(RelType::CollectMetrics(Box::new(pb::CollectMetrics {
            input: Some(Box::new(input)),
            name: "m".to_owned(),
            metrics: vec![aliased_count("n")],
        }))),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observed_metrics_follow_the_result() {
    let run = start().await;
    let responses = collect(&run.channel, request(root(observed(range(10), 7))))
        .await
        .expect("execute");
    let metrics: Vec<_> = responses.iter().flat_map(|r| &r.observed_metrics).collect();
    assert_eq!(metrics.len(), 1, "{responses:?}");
    let m = metrics[0];
    assert_eq!((m.name.as_str(), m.plan_id), ("m", 7));
    assert_eq!(m.keys, ["n"]);
    assert_eq!(
        m.values[0].literal_type,
        Some(pb::expression::literal::LiteralType::Long(10))
    );
    // After the arrow batches and the metrics response.
    let at =
        |f: fn(&pb::ExecutePlanResponse) -> bool| responses.iter().position(f).expect("present");
    assert!(at(|r| !r.observed_metrics.is_empty()) > at(|r| r.metrics.is_some()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_query_sends_no_observed_metrics() {
    let run = start().await;
    let failing = pb::Relation {
        common: None,
        rel_type: Some(RelType::CollectMetrics(Box::new(pb::CollectMetrics {
            input: Some(Box::new(sql("SELECT * FROM no_such_table"))),
            name: "m".to_owned(),
            metrics: vec![aliased_count("n")],
        }))),
    };
    // Read the stream directly so responses sent before the error are kept and checked.
    let mut retained = Vec::new();
    let status = match stream(&run.channel, "ExecutePlan", request(root(failing))).await {
        Err(status) => status,
        Ok(mut s) => loop {
            match s.message().await {
                Ok(Some(r)) => retained.push(r),
                Ok(None) => panic!("the table is missing, yet the stream ended: {retained:?}"),
                Err(status) => break status,
            }
        },
    };
    assert_eq!(status.code(), Code::Internal);
    assert!(
        retained.iter().all(|r| r.observed_metrics.is_empty()),
        "{retained:?}"
    );
}

// ---- the Connect leg of tests/fixtures/spark-4.2.0/observe-goldens.json (P3-T084) ----
//
// Connect returns the metrics of the last action that ran the plan (no first-wins), all names
// in one `ExecutePlanResponse`, and nothing for a query that failed. The cases below are the
// fixture's Connect-only `flows_*` and the ones where the Connect leg differs from the classic
// one; the classic leg (`krishiv-session/tests/observe_goldens.rs`) covers the rest. Cases the
// engine does not match yet are asserted as what they give now (a gap), so a fix fails them.

use pb::expression::literal::LiteralType;

fn expression(e: pb::expression::ExprType) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(e),
    }
}

fn attr(name: &str) -> pb::Expression {
    expression(pb::expression::ExprType::UnresolvedAttribute(
        pb::expression::UnresolvedAttribute {
            unparsed_identifier: name.to_owned(),
            ..Default::default()
        },
    ))
}

fn call(function: &str, arguments: Vec<pb::Expression>) -> pb::Expression {
    expression(pb::expression::ExprType::UnresolvedFunction(
        pb::expression::UnresolvedFunction {
            function_name: function.to_owned(),
            arguments,
            ..Default::default()
        },
    ))
}

fn aliased(e: pb::Expression, name: &str) -> pb::Expression {
    expression(pb::expression::ExprType::Alias(Box::new(
        pb::expression::Alias {
            expr: Some(Box::new(e)),
            name: vec![name.to_owned()],
            metadata: None,
        },
    )))
}

fn base() -> pb::Relation {
    sql(
        "SELECT * FROM VALUES ('a', 1, 1.5D), ('b', 2, CAST(NULL AS DOUBLE)), \
         ('c', CAST(NULL AS INT), 3.0D) AS t(k, v, x)",
    )
}

fn observe_as(
    input: pb::Relation,
    name: &str,
    metrics: Vec<pb::Expression>,
    plan_id: i64,
) -> pb::Relation {
    pb::Relation {
        common: Some(pb::RelationCommon {
            plan_id: Some(plan_id),
            ..Default::default()
        }),
        rel_type: Some(RelType::CollectMetrics(Box::new(pb::CollectMetrics {
            input: Some(Box::new(input)),
            name: name.to_owned(),
            metrics,
        }))),
    }
}

fn count_n() -> pb::Expression {
    aliased(call("count", vec![literal_long(1)]), "n")
}

fn limit_of(input: pb::Relation, limit: i32) -> pb::Relation {
    relation(RelType::Limit(Box::new(pb::Limit {
        input: Some(Box::new(input)),
        limit,
    })))
}

fn union_of(left: pb::Relation, right: pb::Relation) -> pb::Relation {
    relation(RelType::SetOp(Box::new(pb::SetOperation {
        left_input: Some(Box::new(left)),
        right_input: Some(Box::new(right)),
        set_op_type: pb::set_operation::SetOpType::Union as i32,
        is_all: Some(true),
        by_name: None,
        allow_missing_columns: None,
    })))
}

/// The observed metrics of a run: name -> (keys, values), and how many responses carried any.
/// Metric name -> (key, value) cells, and how many responses carried metrics.
type Observed = (BTreeMap<String, Vec<(String, Option<LiteralType>)>>, usize);

fn observed_of(responses: &[pb::ExecutePlanResponse]) -> Observed {
    let carrying = responses
        .iter()
        .filter(|r| !r.observed_metrics.is_empty())
        .count();
    let all = responses
        .iter()
        .flat_map(|r| &r.observed_metrics)
        .map(|m| {
            assert_eq!(m.keys.len(), m.values.len(), "{m:?}");
            let cells = m
                .keys
                .iter()
                .cloned()
                .zip(m.values.iter().map(|v| v.literal_type.clone()))
                .collect();
            (m.name.clone(), cells)
        })
        .collect();
    (all, carrying)
}

fn long(v: i64) -> Option<LiteralType> {
    Some(LiteralType::Long(v))
}

async fn observed_run(plan: pb::Relation) -> Observed {
    let run = start().await;
    let responses = collect(&run.channel, request(root(plan)))
        .await
        .expect("execute");
    observed_of(&responses)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_flows_string_observe_and_no_alias() {
    // flows_string_observe: an unaliased literal is named by its text.
    let plan = observe_as(
        base(),
        "nm",
        vec![
            count_n(),
            aliased(call("sum", vec![attr("v")]), "s"),
            literal_long(7),
        ],
        1,
    );
    let (got, _) = observed_run(plan).await;
    assert_eq!(
        got["nm"],
        vec![
            ("n".to_owned(), long(3)),
            ("s".to_owned(), long(3)),
            ("7".to_owned(), long(7))
        ]
    );
    // flows_no_alias: metrics named as Spark names them.
    let plan = observe_as(
        base(),
        "x",
        vec![
            call("count", vec![literal_long(1)]),
            call("max", vec![attr("v")]),
        ],
        2,
    );
    let (got, _) = observed_run(plan).await;
    assert_eq!(
        got["x"],
        vec![
            ("count(1)".to_owned(), long(3)),
            ("max(v)".to_owned(), Some(LiteralType::Integer(2)))
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_flows_two_nested_in_one_response() {
    let inner = observe_as(base(), "a", vec![count_n()], 1);
    let outer = observe_as(
        inner,
        "b",
        vec![aliased(call("sum", vec![attr("v")]), "s")],
        2,
    );
    let (got, carrying) = observed_run(outer).await;
    assert_eq!(got["a"], vec![("n".to_owned(), long(3))]);
    assert_eq!(got["b"], vec![("s".to_owned(), long(3))]);
    assert_eq!(carrying, 1, "Spark sends all the names in one response");
}

/// Connect has no first-wins: each run reports its own metrics (`first_then_second_action`:
/// the count after a `limit(1)` run is 3), and the limit's run counts the rows it took.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_each_action_reports_its_own_metrics() {
    let run = start().await;
    let observed = observe_as(base(), "m", vec![count_n()], 1);
    let first = collect(&run.channel, request(root(limit_of(observed.clone(), 1))))
        .await
        .expect("execute");
    assert_eq!(observed_of(&first).0["m"], vec![("n".to_owned(), long(1))]);
    let second = collect(&run.channel, request(root(observed)))
        .await
        .expect("execute");
    assert_eq!(observed_of(&second).0["m"], vec![("n".to_owned(), long(3))]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_self_union_reports_one_instance_not_the_sum() {
    let d = observe_as(base(), "m", vec![count_n()], 1);
    let (got, _) = observed_run(union_of(d.clone(), d)).await;
    assert_eq!(got["m"], vec![("n".to_owned(), long(3))]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_limit_zero_sends_no_metrics() {
    let observed = observe_as(base(), "m", vec![count_n()], 1);
    let (got, carrying) = observed_run(limit_of(observed, 0)).await;
    assert!(got.is_empty(), "{got:?}");
    assert_eq!(carrying, 0);
}

/// Spark prunes an observe node under an empty relation (P3-T084c): no metrics are sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_where_false_above_observe_prunes_the_node() {
    let observed = observe_as(base(), "e", vec![count_n()], 1);
    let filtered = relation(RelType::Filter(Box::new(pb::Filter {
        input: Some(Box::new(observed)),
        condition: Some(call("==", vec![literal_long(1), literal_long(0)])),
    })));
    let (got, carrying) = observed_run(filtered).await;
    assert!(got.is_empty(), "{got:?}");
    assert_eq!(carrying, 0);
}

/// Decimal, NaN and array metric values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_decimal_nan_and_array_metrics() {
    let run = start().await;
    let types = sql(
        "SELECT CAST(1.10 AS DECIMAL(5,2)) d, CAST('NaN' AS DOUBLE) n, CAST(-0.0 AS DOUBLE) z, 1 i",
    );
    let plan = observe_as(
        types.clone(),
        "m",
        vec![
            aliased(call("sum", vec![attr("d")]), "sd"),
            aliased(call("max", vec![attr("n")]), "mn"),
            aliased(call("collect_list", vec![attr("i")]), "l"),
        ],
        1,
    );
    let responses = collect(&run.channel, request(root(plan)))
        .await
        .expect("execute");
    let (got, _) = observed_of(&responses);
    let cells = &got["m"];
    // Spark 4.2.0 (`types` golden): sd = Decimal('1.10'), mn = NaN, l = [1]. The type of
    // sum(DECIMAL(5,2)) is DECIMAL(15,2) (Spark's `Sum.resultType`: bounded(p + 10, s)).
    assert_eq!(cells[0].0, "sd");
    assert_eq!(
        cells[0].1,
        Some(LiteralType::Decimal(pb::expression::literal::Decimal {
            value: "1.10".to_owned(),
            precision: Some(15),
            scale: Some(2),
        })),
        "{cells:?}"
    );
    assert_eq!(cells[1].0, "mn");
    assert!(
        matches!(&cells[1].1, Some(LiteralType::Double(d)) if d.is_nan()),
        "{cells:?}"
    );
    assert_eq!(cells[2].0, "l");
    let Some(LiteralType::Array(array)) = &cells[2].1 else {
        panic!("l is not an array: {cells:?}");
    };
    let elements: Vec<_> = array
        .elements
        .iter()
        .map(|e| e.literal_type.clone())
        .collect();
    assert_eq!(elements, vec![Some(LiteralType::Integer(1))]);
}

/// Struct, map, binary and interval metrics go out in Spark's literal forms
/// (`observe-goldens-followup.json`, `value_types`, Connect leg). Spark's Python client raises
/// `UNSUPPORTED_LITERAL` for the year-month and calendar forms; the server sends them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    deprecated,
    reason = "the literal's struct and map types are read as Spark's clients do"
)]
async fn connect_struct_map_binary_and_interval_metrics() {
    let max = |c: &str| call("max", vec![attr(c)]);
    let plan = observe_as(
        sql("SELECT 1 i, 'x' s, X'FF00FE' b, INTERVAL '2' DAY dt, INTERVAL '2' MONTH ym"),
        "m",
        vec![
            aliased(call("map", vec![literal_long(7), max("i")]), "m"),
            aliased(max("b"), "b"),
            aliased(max("dt"), "dt"),
            aliased(max("ym"), "ym"),
            aliased(call("struct", vec![max("i"), max("s")]), "st"),
        ],
        1,
    );
    let (got, _) = observed_run(plan).await;
    let cell = |k: &str| {
        got["m"]
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, v)| v.clone())
            .expect(k)
    };
    let LiteralType::Map(map) = cell("m") else {
        panic!("map")
    };
    assert_eq!((map.keys.len(), map.values.len()), (1, 1), "{map:?}");
    assert_eq!(map.keys[0].literal_type, Some(LiteralType::Long(7)));
    assert_eq!(map.values[0].literal_type, Some(LiteralType::Integer(1)));
    assert!(map.key_type.is_some() && map.value_type.is_some());
    assert_eq!(cell("b"), LiteralType::Binary(vec![0xFF, 0x00, 0xFE]));
    assert_eq!(cell("dt"), LiteralType::DayTimeInterval(172_800_000_000));
    assert_eq!(cell("ym"), LiteralType::YearMonthInterval(2));
    let LiteralType::Struct(st) = cell("st") else {
        panic!("struct")
    };
    assert_eq!(st.elements.len(), 2);
    assert!(st.struct_type.is_some());
    assert_eq!(
        st.elements[1].literal_type,
        Some(LiteralType::String("x".into()))
    );
}

/// The caller of every request to a server without authentication.
fn local_caller() -> Authenticated {
    Authenticated::new(Principal::local_user(), None, Method::None)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_compressed_sql_command_runs_like_a_plain_one() {
    let run = start().await;
    let Some(OpType::Command(command)) =
        sql_command("CREATE TEMPORARY VIEW zc AS SELECT 7 AS n").op_type
    else {
        panic!("not a command");
    };
    let data = zstd::bulk::compress(&command.encode_to_vec(), 3).expect("zstd");
    let plan = pb::Plan {
        op_type: Some(OpType::CompressedOperation(Box::new(
            pb::plan::CompressedOperation {
                data,
                op_type: pb::plan::compressed_operation::OpType::Command as i32,
                compression_codec: pb::CompressionCodec::Zstd as i32,
            },
        ))),
    };
    let r = collect(&run.channel, request(plan)).await.expect("runs");
    // The command ran, and its answer is the one `sql_command_result`.
    let relation = sql_result(&r);
    assert!(
        matches!(relation.rel_type, Some(RelType::LocalRelation(_))),
        "{relation:?}"
    );
    // The view exists afterwards: the compressed command ran, it was not only answered.
    let entry = run
        .manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session");
    let batches = entry
        .session()
        .sql("SELECT n FROM zc")
        .await
        .expect("reads the view")
        .collect()
        .await
        .expect("collects");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(
        batches[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        7
    );
}

/// A plan of `Command{ml_command{..}}`.
fn ml_command(command: pb::ml_command::Command) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(pb::command::CommandType::MlCommand(Box::new(
                pb::MlCommand {
                    command: Some(command),
                },
            ))),
        }))),
    }
}

fn clean_cache() -> pb::Plan {
    ml_command(pb::ml_command::Command::CleanCache(
        pb::ml_command::CleanCache {},
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ml_clean_cache_succeeds_with_nothing_to_clean() {
    // PySpark 4.3.0's `atexit` cleanup (`_cleanup_ml_cache`) sends it before
    // `release_session` (connect-mapping §4 row 17, decision 11).
    let run = start().await;
    let entry = run
        .manager
        .get_or_create(&local_caller(), SESSION, None, "u")
        .expect("session");
    // Not reattachable: no response, the stream ends OK.
    let first = "00112233-4455-6677-8899-aabbccddee02";
    let responses = collect(&run.channel, with_id(request(clean_cache()), first))
        .await
        .expect("clean_cache");
    assert!(responses.is_empty(), "{} responses", responses.len());
    eventually("the operation leaves", || entry.operations().is_empty()).await;
    assert_eq!(
        entry.operations().state_of(first),
        Some(OperationState::Succeeded)
    );

    // Reattachable: only `ResultComplete`, then it reattaches and releases
    // like any other operation.
    let id = "00112233-4455-6677-8899-aabbccddee01";
    let responses = collect(
        &run.channel,
        with_id(reattachable(request(clean_cache())), id),
    )
    .await
    .expect("clean_cache");
    assert_eq!(responses.len(), 1, "only ResultComplete");
    assert!(matches!(
        responses[0].response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    assert_eq!(responses[0].operation_id, id);
    assert_eq!(responses[0].session_id, SESSION);
    assert_eq!(entry.operations().len(), 1, "kept until released");
    assert_eq!(
        entry.operations().state_of(id),
        Some(OperationState::Succeeded)
    );
    let mut again = stream(
        &run.channel,
        "ReattachExecute",
        pb::ReattachExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: id.to_owned(),
            last_response_id: Some(responses[0].response_id.clone()),
            ..Default::default()
        },
    )
    .await
    .expect("reattach");
    assert!(
        again.message().await.expect("ok").is_none(),
        "nothing after"
    );
    let released: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        pb::ReleaseExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: id.to_owned(),
            release: Some(pb::release_execute_request::Release::ReleaseAll(
                pb::release_execute_request::ReleaseAll {},
            )),
            ..Default::default()
        },
    )
    .await
    .expect("release all");
    assert_eq!(released.operation_id.as_deref(), Some(id));
    assert!(entry.operations().is_empty());
    assert_eq!(
        entry.operations().state_of(id),
        Some(OperationState::Succeeded)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn other_ml_commands_stay_unsupported() {
    let run = start().await;
    let delete = ml_command(pb::ml_command::Command::Delete(
        pb::ml_command::Delete::default(),
    ));
    let info_of = ml_command(pb::ml_command::Command::GetCacheInfo(
        pb::ml_command::GetCacheInfo {},
    ));
    let empty = pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(pb::command::CommandType::MlCommand(Box::default())),
        }))),
    };
    for plan in [delete, info_of, empty] {
        for r in [request(plan.clone()), reattachable(request(plan))] {
            let e = collect(&run.channel, r).await.unwrap_err();
            assert_eq!(info(&e)["sqlState"], "56K00", "{e:?}");
        }
    }
    let entry = run
        .manager
        .get(&local_caller(), SESSION, None)
        .expect("session");
    assert!(entry.operations().is_empty());
}

/// CON-RC-3: a temporary view made by `spark.sql` over Connect reads back through the SQL
/// relation the client sends again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_temp_view_made_by_spark_sql_can_be_read() {
    let run = start().await;
    collect(
        &run.channel,
        request(sql_command_input(
            "CREATE OR REPLACE TEMP VIEW t AS SELECT id, id % 7 AS k FROM range(300)",
            1,
        )),
    )
    .await
    .expect("creates the view");
    let r = collect(
        &run.channel,
        request(sql_command_input("SELECT count(*) AS n FROM t", 2)),
    )
    .await
    .expect("analyzes the query");
    let rel = sql_result(&r);
    let r = collect(&run.channel, request(root(rel)))
        .await
        .expect("reads the view");
    let got = rows(&r);
    assert_eq!(
        got.column(0)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0),
        300
    );
}

/// The `show_string` text a ROOT `ShowString` answers for `input`.
async fn shown(input: pb::Relation, num_rows: i32, truncate: i32, vertical: bool) -> String {
    let run = start().await;
    let show = relation(RelType::ShowString(Box::new(pb::ShowString {
        input: Some(Box::new(input)),
        num_rows,
        truncate,
        vertical,
    })));
    let responses = collect(&run.channel, request(root(show)))
        .await
        .expect("show_string");
    let b = batches(&responses);
    assert_eq!(b.len(), 1, "one batch");
    let read = read_ipc(&b[0].data);
    assert_eq!(read[0].num_rows(), 1);
    assert_eq!(read[0].schema().field(0).name(), "show_string");
    read[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .expect("string")
        .value(0)
        .to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn show_string_lays_out_the_first_rows_like_dataset_show() {
    let text = shown(
        sql("SELECT id, CAST(NULL AS STRING) AS s, X'0102' AS b FROM range(3)"),
        2,
        20,
        false,
    )
    .await;
    assert_eq!(
        text,
        "+---+----+-------+\n| id|   s|      b|\n+---+----+-------+\n|  0|NULL|[01 02]|\n|  1|NULL|[01 02]|\n+---+----+-------+\nonly showing top 2 rows"
    );
}

/// Every case of the Spark 4.3.0 `show_string` fixture (`tests/scripts/gen_show_string_goldens.py`, a real
/// Spark run) answers the same text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn show_string_matches_the_spark_430_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.3.0/show-string.json"
    );
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json");
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().expect("cases") {
        let Some(expected) = case["expected"].as_str() else {
            // The fixture lists no refusals today; an oracle error is a regenerated-fixture surprise.
            panic!("fixture case without an expected text (oracle error?): {case}");
        };
        let got = shown(
            sql(case["sql"].as_str().expect("sql")),
            i32::try_from(case["num_rows"].as_i64().expect("n")).expect("i32"),
            i32::try_from(case["truncate"].as_i64().expect("t")).expect("i32"),
            case["vertical"].as_bool().expect("v"),
        )
        .await;
        if got != expected {
            failures.push(format!(
                "{}:\n--- spark\n{expected}\n--- krishiv\n{got}",
                case["name"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// A compressed `show_string` relation answers as the uncompressed one does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_compressed_show_string_answers_the_same_text() {
    let run = start().await;
    let show = relation(RelType::ShowString(Box::new(pb::ShowString {
        input: Some(Box::new(sql("SELECT 1 AS x"))),
        num_rows: 20,
        truncate: 20,
        vertical: false,
    })));
    let data = zstd::bulk::compress(&show.encode_to_vec(), 3).expect("zstd");
    let plan = pb::Plan {
        op_type: Some(OpType::CompressedOperation(Box::new(
            pb::plan::CompressedOperation {
                data,
                op_type: pb::plan::compressed_operation::OpType::Relation as i32,
                compression_codec: pb::CompressionCodec::Zstd as i32,
            },
        ))),
    };
    let responses = collect(&run.channel, request(plan)).await.expect("show");
    let b = batches(&responses);
    let read = read_ipc(&b[0].data);
    assert_eq!(read[0].schema().field(0).name(), "show_string");
    let text = read[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .expect("string")
        .value(0)
        .to_owned();
    assert_eq!(text, "+---+\n|  x|\n+---+\n|  1|\n+---+\n");
}

/// Result columns that share a name show by position, as `Dataset.showString` does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn show_string_binds_duplicate_named_columns_by_position() {
    let text = shown(sql("SELECT 1 AS x, 2 AS x"), 20, 20, false).await;
    assert_eq!(
        text,
        "+---+---+\n|  x|  x|\n+---+---+\n|  1|  2|\n+---+---+\n"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn html_string_is_refused_until_it_has_goldens() {
    let run = start().await;
    let html = relation(RelType::HtmlString(Box::new(pb::HtmlString {
        input: Some(Box::new(sql("SELECT 1 AS x"))),
        num_rows: 5,
        truncate: 20,
    })));
    let err = collect(&run.channel, request(root(html)))
        .await
        .expect_err("html_string is refused");
    assert!(err.to_string().contains("html_string"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_show_string_result_over_the_byte_cap_is_refused() {
    let run = start_with(ExecuteLimits {
        max_batch_size: 1024,
        ..ExecuteLimits::default()
    })
    .await;
    // Empty cells still cost their headers, so many rows pass the cap without any text.
    for input in [
        "SELECT '' AS x FROM range(100000)",
        "SELECT repeat('a', 5000) AS x",
    ] {
        let show = relation(RelType::ShowString(Box::new(pb::ShowString {
            input: Some(Box::new(sql(input))),
            num_rows: 1_000_000,
            truncate: 0,
            vertical: false,
        })));
        let e = collect(&run.channel, request(root(show)))
            .await
            .unwrap_err();
        assert_eq!(error_class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{input}");
    }
}

/// The path issue S4M-3 was found on (ADR 0003 update 2026-11-27): under `krishiv.connect.sparkProfile = auto`,
/// a PySpark 3.5.10 client's `SELECT 1/0` is NULL, as on Spark 3.5 (ANSI off), and a 4.x client's raises.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_3_5_client_gets_the_3_5_ansi_default_through_execute_plan() {
    let manager =
        SessionManager::from_base(Session::new().expect("session"), SessionLimits::default())
            .with_profile_choice(ProfileChoice::Auto);
    let running = serve(manager, ExecuteLimits::default()).await;
    let mut old = request(root(sql("SELECT 1/0")));
    old.client_type = Some("_SPARK_CONNECT_PYTHON spark/3.5.10 os/linux python/3.11.9".to_owned());
    let responses = collect(&running.channel, old).await.expect("3.5 runs");
    assert!(rows(&responses).column(0).is_null(0), "1/0 is NULL on 3.5");

    let mut new = request(root(sql("SELECT 1/0")));
    new.session_id = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d11".to_owned();
    new.client_type = Some("_SPARK_CONNECT_PYTHON spark/4.1.0 os/linux python/3.11.9".to_owned());
    let e = collect(&running.channel, new)
        .await
        .expect_err("4.1 raises");
    assert_eq!(error_class(&e), "DIVIDE_BY_ZERO");
}
