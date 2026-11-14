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

use arrow_array::RecordBatch;
use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_request::RequestOption as ReqOpt;
use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
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
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits::default(),
    ));
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
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
        .get(&Principal::local_user(), SESSION, None)
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
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
        command_type: Some(pb::command::CommandType::SqlCommand(Box::default())),
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
            command_type: Some(pb::command::CommandType::SqlCommand(Box::default())),
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
        .get(&Principal::local_user(), SESSION, None)
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
        .get_or_create(&Principal::local_user(), OTHER_SESSION, None, "u")
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
        .get(&Principal::local_user(), SESSION, None)
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
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
        .get_or_create(&Principal::local_user(), SESSION, None, "u")
        .expect("session");
    // Not reattachable: no response, the stream ends OK.
    let responses = collect(&run.channel, request(clean_cache()))
        .await
        .expect("clean_cache");
    assert!(responses.is_empty(), "{} responses", responses.len());
    eventually("the operation leaves", || entry.operations().is_empty()).await;

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
        .get(&Principal::local_user(), SESSION, None)
        .expect("session");
    assert!(entry.operations().is_empty());
}
