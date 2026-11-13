// SPDX-License-Identifier: Apache-2.0
//! The replay buffer, `ReattachExecute`, the sender caps and detach end to
//! end over a raw tonic client (design P3-T011a §3.8, PR-15a; Spark 4.3.0's
//! `ExecuteResponseObserver`, `ExecuteGrpcResponseSender`, `ExecuteHolder`
//! and `SparkConnectExecutionManager`).
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]
#![expect(
    clippy::disallowed_methods,
    reason = "the tests read back the server's status details"
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_request::RequestOption as ReqOpt;
use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::proto::v4_3::release_execute_request::{Release, ReleaseAll, ReleaseUntil};
use krishiv_connect_server::{
    ConnectServer, ConnectServerConfig, ExecuteLimits, SessionEntry, SessionLimits, SessionManager,
};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";
const OP2: &str = "00112233-4455-6677-8899-aabbccddee00";

struct Running {
    manager: Arc<SessionManager>,
    channel: Channel,
    _stop: oneshot::Sender<()>,
}

impl Running {
    fn entry(&self) -> Arc<SessionEntry> {
        self.manager
            .get_or_create(&Principal::local_user(), SESSION, None, "u")
            .expect("session")
    }

    /// The session, with `rows` rows per Arrow batch.
    fn session(&self, rows: usize) -> Session {
        let session = self.entry().session().clone();
        session
            .conf()
            .set(
                "spark.sql.execution.arrow.maxRecordsPerBatch",
                &rows.to_string(),
            )
            .expect("conf");
        session
    }
}

async fn start_with(execute: ExecuteLimits, sessions: SessionLimits) -> Running {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        sessions,
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

async fn start(execute: ExecuteLimits) -> Running {
    start_with(execute, SessionLimits::default()).await
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

type Responses = tonic::Streaming<pb::ExecutePlanResponse>;

async fn stream<Req>(channel: &Channel, rpc: &str, request: Req) -> Result<Responses, tonic::Status>
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

fn range(end: i64) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Range(pb::Range {
                start: Some(0),
                end,
                step: 1,
                num_partitions: Some(1),
            })),
        }))),
    }
}

/// A reattachable `ExecutePlan` of `plan` with the operation id `op`.
fn execute(plan: pb::Plan, op: &str) -> pb::ExecutePlanRequest {
    pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        operation_id: Some(op.to_owned()),
        plan: Some(plan),
        request_options: vec![ReqOpt {
            request_option: Some(RequestOption::ReattachOptions(pb::ReattachOptions {
                reattachable: true,
            })),
        }],
        ..Default::default()
    }
}

fn reattach(op: &str, last: Option<&str>) -> pb::ReattachExecuteRequest {
    pb::ReattachExecuteRequest {
        session_id: SESSION.to_owned(),
        operation_id: op.to_owned(),
        last_response_id: last.map(str::to_owned),
        ..Default::default()
    }
}

fn release(op: &str, how: Release) -> pb::ReleaseExecuteRequest {
    pb::ReleaseExecuteRequest {
        session_id: SESSION.to_owned(),
        operation_id: op.to_owned(),
        release: Some(how),
        ..Default::default()
    }
}

fn until(id: &str) -> Release {
    Release::ReleaseUntil(ReleaseUntil {
        response_id: id.to_owned(),
    })
}

async fn release_all(run: &Running, op: &str) {
    let _: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        release(op, Release::ReleaseAll(ReleaseAll {})),
    )
    .await
    .expect("release all");
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
    let Ok(rpc) = RpcStatus::decode(status.details()) else {
        return format!("<no details: {status:?}>");
    };
    let Some(any) = rpc.details.first() else {
        return format!("<no details: {status:?}>");
    };
    ErrorInfo::decode(any.value.as_slice())
        .expect("ErrorInfo")
        .metadata
        .get("errorClass")
        .cloned()
        .unwrap_or_default()
}

fn is_complete(r: &pb::ExecutePlanResponse) -> bool {
    matches!(r.response_type, Some(ResponseType::ResultComplete(_)))
}

/// What one stream gave: its responses and how it ended.
struct Read {
    responses: Vec<pb::ExecutePlanResponse>,
    end: Result<(), tonic::Status>,
}

/// Reads `s` to its end.
async fn drain(mut s: Responses) -> Read {
    let mut responses = Vec::new();
    let end = loop {
        match s.message().await {
            Ok(Some(r)) => responses.push(r),
            Ok(None) => break Ok(()),
            Err(e) => break Err(e),
        }
    };
    Read { responses, end }
}

/// Reattaches after `last` until `ResultComplete`, as the stock client
/// does; returns every response and how many streams it took.
async fn reattach_to_end(
    run: &Running,
    op: &str,
    mut last: Option<String>,
) -> (Vec<pb::ExecutePlanResponse>, usize) {
    let mut all = Vec::new();
    let mut streams = 0;
    loop {
        streams += 1;
        assert!(streams < 1000, "the client never got ResultComplete");
        let s = stream(
            &run.channel,
            "ReattachExecute",
            reattach(op, last.as_deref()),
        )
        .await
        .expect("reattach");
        let read = drain(s).await;
        read.end.expect("the stream ends with OK");
        let done = read.responses.iter().any(is_complete);
        if let Some(r) = read.responses.last() {
            last = Some(r.response_id.clone());
        }
        all.extend(read.responses);
        if done {
            return (all, streams);
        }
    }
}

/// Waits until `cond` holds (or fails after 10 s).
async fn eventually(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(tokio::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The batches of `responses` must be exactly the `rows` rows of a range,
/// each once and in order (engine batches are cut at the session's
/// `maxRecordsPerBatch`, so their sizes vary).
fn assert_every_batch_once(responses: &[pb::ExecutePlanResponse], rows: i64, per: i64) {
    let mut next = 0_i64;
    for r in responses {
        if let Some(ResponseType::ArrowBatch(b)) = &r.response_type {
            assert_eq!(
                b.start_offset,
                Some(next),
                "every batch exactly once, in order"
            );
            assert!(b.row_count <= per);
            next += b.row_count;
        }
    }
    assert_eq!(next, rows, "every row");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reattach_resumes_after_disconnect() {
    let run = start(ExecuteLimits::default()).await;
    run.session(10);
    let mut first = stream(&run.channel, "ExecutePlan", execute(range(1000), OP))
        .await
        .expect("execute");
    let mut seen = Vec::new();
    // The schema and five batches, then the connection is cut.
    for _ in 0..6 {
        seen.push(first.message().await.expect("ok").expect("response"));
    }
    drop(first);
    let entry = run.entry();
    // A reattachable operation detaches; it is not cancelled.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let op = entry.operations().get(OP).expect("still live");
    assert!(!op.is_cancelled());
    let last = seen.last().map(|r| r.response_id.clone());
    let (rest, _) = reattach_to_end(&run, OP, last).await;
    seen.extend(rest);
    assert_every_batch_once(&seen, 1000, 10);
    assert!(seen.iter().filter(|r| is_complete(r)).count() == 1);
    // Still live until released (Spark's `ReleaseExecute(release_all)`).
    assert!(entry.operations().get(OP).is_some());
    release_all(&run, OP).await;
    assert!(entry.operations().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn execute_plan_with_a_live_id_reattaches_from_the_start() {
    let run = start(ExecuteLimits::default()).await;
    run.session(10);
    let mut first = stream(&run.channel, "ExecutePlan", execute(range(100), OP))
        .await
        .expect("execute");
    first.message().await.expect("ok").expect("schema");
    // The same plan again: a new sender from response 1 (Spark's
    // `SparkConnectExecutePlanHandler` lines 47-55); the old one ends with
    // INVALID_CURSOR.DISCONNECTED.
    let again = stream(&run.channel, "ExecutePlan", execute(range(100), OP))
        .await
        .expect("reattach from the start");
    let old = drain(first).await;
    let e = old.end.expect_err("the old sender is interrupted");
    assert_eq!(error_class(&e), "INVALID_CURSOR.DISCONNECTED");
    let read = drain(again).await;
    read.end.expect("ok");
    assert!(read.responses[0].schema.is_some(), "from the start");
    assert_every_batch_once(&read.responses, 100, 10);
    assert!(read.responses.last().is_some_and(is_complete));
    // Another plan under the live id is still OPERATION_ALREADY_EXISTS.
    let e = stream(&run.channel, "ExecutePlan", execute(range(7), OP))
        .await
        .expect_err("another plan");
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_ALREADY_EXISTS");
    release_all(&run, OP).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_cursor_cases() {
    // Retry buffer 0: a consumed response is dropped once the next one is
    // consumed (Spark's `removeCachedResponses`).
    let run = start(ExecuteLimits {
        retry_buffer_size: 0,
        ..ExecuteLimits::default()
    })
    .await;
    run.session(10);
    // Far more than the transport's windows hold, so the first sender is
    // still sending when it is replaced.
    let mut live = stream(&run.channel, "ExecutePlan", execute(range(100_000), OP))
        .await
        .expect("execute");
    let mut seen = Vec::new();
    for _ in 0..4 {
        seen.push(live.message().await.expect("ok").expect("response"));
    }

    // POSITION_NOT_FOUND: an id this operation never sent. The live sender
    // is not interrupted (Spark looks the id up before it attaches).
    let s = stream(&run.channel, "ReattachExecute", reattach(OP, Some("nope"))).await;
    let e = match s {
        Err(e) => e,
        Ok(s) => drain(s).await.end.expect_err("refused"),
    };
    assert_eq!(error_class(&e), "INVALID_CURSOR.POSITION_NOT_FOUND");
    seen.push(live.message().await.expect("ok").expect("still sending"));

    // POSITION_NOT_AVAILABLE: resuming after response 1 needs response 2,
    // dropped already; this new sender replaced the old one, which ends
    // with DISCONNECTED.
    let s = stream(
        &run.channel,
        "ReattachExecute",
        reattach(OP, Some(&seen[0].response_id)),
    )
    .await;
    let e = match s {
        Err(e) => e,
        Ok(s) => drain(s).await.end.expect_err("refused"),
    };
    assert_eq!(error_class(&e), "INVALID_CURSOR.POSITION_NOT_AVAILABLE");
    let old = drain(live).await;
    assert_eq!(
        error_class(&old.end.expect_err("interrupted")),
        "INVALID_CURSOR.DISCONNECTED"
    );
    // (With no retry buffer, what the old sender handed to the transport
    // beyond what this client read is gone too: Spark's trade-off.)
    release_all(&run, OP).await;

    // NOT_REATTACHABLE: a live operation that did not ask for it.
    let mut plain = execute(range(100_000_000_000), OP2);
    plain.request_options.clear();
    let mut live = stream(&run.channel, "ExecutePlan", plain)
        .await
        .expect("execute");
    live.message().await.expect("ok").expect("schema");
    let e = stream(&run.channel, "ReattachExecute", reattach(OP2, None))
        .await
        .expect_err("not reattachable");
    assert_eq!(error_class(&e), "INVALID_CURSOR.NOT_REATTACHABLE");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_until_trims_and_accepts_a_released_id_again() {
    let run = start(ExecuteLimits::default()).await;
    run.session(10);
    let mut live = stream(&run.channel, "ExecutePlan", execute(range(1000), OP))
        .await
        .expect("execute");
    let mut seen = Vec::new();
    for _ in 0..5 {
        seen.push(live.message().await.expect("ok").expect("response"));
    }
    let ok: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        release(OP, until(&seen[3].response_id)),
    )
    .await
    .expect("release until");
    assert_eq!(ok.operation_id.as_deref(), Some(OP));
    // Spark never forgets a response id (`responseIdToIndex`), so releasing
    // it again succeeds, and reattaching after a released id resumes.
    let _: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        release(OP, until(&seen[3].response_id)),
    )
    .await
    .expect("again");
    let e = unary::<_, pb::ReleaseExecuteResponse>(
        &run.channel,
        "ReleaseExecute",
        release(OP, until("never-sent")),
    )
    .await
    .expect_err("unknown id");
    assert_eq!(error_class(&e), "INVALID_CURSOR.POSITION_NOT_FOUND");
    drop(live);
    let last = seen.last().map(|r| r.response_id.clone());
    let (rest, _) = reattach_to_end(&run, OP, last).await;
    seen.extend(rest);
    assert_every_batch_once(&seen, 1000, 10);
    release_all(&run, OP).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_all_frees_buffer() {
    let run = start(ExecuteLimits::default()).await;
    let session = run.session(10);
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let read = drain(
        stream(&run.channel, "ExecutePlan", execute(range(1000), OP))
            .await
            .expect("execute"),
    )
    .await;
    read.end.expect("ok");
    assert!(read.responses.last().is_some_and(is_complete));
    // Everything was sent, and the retry buffer still holds it.
    assert!(memory.reserved() > before, "the retry buffer is held");
    release_all(&run, OP).await;
    assert!(run.entry().operations().is_empty());
    eventually("the buffer's memory returns", || {
        memory.reserved() == before
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_released_live_operation_disconnects_its_stream_and_stops() {
    let run = start(ExecuteLimits::default()).await;
    let session = run.session(10);
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        execute(range(100_000_000_000), OP),
    )
    .await
    .expect("execute");
    live.message().await.expect("ok").expect("schema");
    release_all(&run, OP).await;
    // Spark's `ExecuteHolder.close` interrupts the attached sender.
    let read = drain(live).await;
    assert_eq!(
        error_class(&read.end.expect_err("interrupted")),
        "INVALID_CURSOR.DISCONNECTED"
    );
    eventually("the query stops", || session.running_queries() == 0).await;
    eventually("the memory returns", || memory.reserved() == before).await;
    let e = stream(&run.channel, "ReattachExecute", reattach(OP, None))
        .await
        .expect_err("released");
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_NOT_FOUND");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn detached_abandoned() {
    let run = start(ExecuteLimits {
        detached_timeout: Duration::from_secs(1),
        maintenance_interval: Duration::from_millis(100),
        ..ExecuteLimits::default()
    })
    .await;
    let session = run.session(10);
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        execute(range(100_000_000_000), OP),
    )
    .await
    .expect("execute");
    live.message().await.expect("ok").expect("schema");
    drop(live);
    let entry = run.entry();
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(
        entry.operations().get(OP).is_some(),
        "detached, not yet abandoned"
    );
    eventually("abandoned after the timeout", || {
        entry.operations().get(OP).is_none()
    })
    .await;
    eventually("the query stops", || session.running_queries() == 0).await;
    eventually("the memory returns", || memory.reserved() == before).await;
    let e = stream(&run.channel, "ReattachExecute", reattach(OP, None))
        .await
        .expect_err("abandoned");
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_ABANDONED");
    let e = stream(
        &run.channel,
        "ExecutePlan",
        execute(range(100_000_000_000), OP),
    )
    .await
    .expect_err("abandoned id");
    assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_ABANDONED");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_attached_operation_is_never_abandoned() {
    let run = start(ExecuteLimits {
        detached_timeout: Duration::from_millis(200),
        maintenance_interval: Duration::from_millis(50),
        ..ExecuteLimits::default()
    })
    .await;
    run.session(10);
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        execute(range(100_000_000_000), OP),
    )
    .await
    .expect("execute");
    live.message().await.expect("ok").expect("schema");
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert!(run.entry().operations().get(OP).is_some());
    live.message().await.expect("ok").expect("still sending");
    release_all(&run, OP).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_cap_reattach() {
    // A 2 KiB cap instead of 1 GiB: each stream ends with OK and no
    // ResultComplete after about 2 KiB; the client reattaches.
    let run = start(ExecuteLimits {
        sender_max_stream_size: 2048,
        ..ExecuteLimits::default()
    })
    .await;
    run.session(10);
    let first = drain(
        stream(&run.channel, "ExecutePlan", execute(range(1000), OP))
            .await
            .expect("execute"),
    )
    .await;
    first.end.expect("the cap ends the stream with OK");
    assert!(!first.responses.iter().any(is_complete));
    let sent: usize = first.responses.iter().map(Message::encoded_len).sum();
    let without_last = sent - first.responses.last().map_or(0, Message::encoded_len);
    assert!(
        without_last <= 2048,
        "the crossing response is the last one"
    );
    assert!(sent > 2048);
    let mut seen = first.responses;
    let last = seen.last().map(|r| r.response_id.clone());
    let (rest, streams) = reattach_to_end(&run, OP, last).await;
    assert!(streams > 2, "{streams} streams");
    seen.extend(rest);
    assert_every_batch_once(&seen, 1000, 10);
    release_all(&run, OP).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_duration_cap_reattach() {
    let run = start(ExecuteLimits {
        sender_max_stream_duration: Duration::from_millis(300),
        ..ExecuteLimits::default()
    })
    .await;
    run.session(10);
    // Far more than the transport's windows hold, so the sender is still
    // sending at its deadline.
    let mut first = stream(&run.channel, "ExecutePlan", execute(range(100_000), OP))
        .await
        .expect("execute");
    let mut seen = vec![first.message().await.expect("ok").expect("schema")];
    tokio::time::sleep(Duration::from_millis(500)).await;
    // Past the deadline: what was handed to the transport arrives, then
    // the stream ends with OK and no ResultComplete.
    let read = drain(first).await;
    read.end.expect("ok");
    assert!(!read.responses.iter().any(is_complete));
    seen.extend(read.responses);
    let last = seen.last().map(|r| r.response_id.clone());
    let (rest, _) = reattach_to_end(&run, OP, last).await;
    seen.extend(rest);
    assert_every_batch_once(&seen, 100_000, 10);
    release_all(&run, OP).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn global_response_cap() {
    // 256 KiB across the server, 1 MiB ahead per operation: the global cap
    // binds first.
    let run = start(ExecuteLimits {
        ahead_bytes: 1 << 20,
        max_response_bytes: 256 * 1024,
        ..ExecuteLimits::default()
    })
    .await;
    let session = run.session(1000);
    let mut hog = stream(
        &run.channel,
        "ExecutePlan",
        execute(range(100_000_000_000), OP),
    )
    .await
    .expect("execute");
    hog.message().await.expect("ok").expect("schema");
    let entry = run.entry();
    let hog_op = entry.operations().get(OP).expect("live");
    // Nothing more is read: the hog fills the global budget and pauses.
    let mut last = 0;
    loop {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let now = hog_op.sent_count();
        if now == last {
            break;
        }
        last = now;
    }
    assert!(
        hog_op.buffered_peak() <= 256 * 1024 + 16 * 1024,
        "{} bytes held",
        hog_op.buffered_peak()
    );
    // A new operation waits for room rather than failing.
    let small = tokio::spawn({
        let channel = run.channel.clone();
        async move {
            let s = stream(&channel, "ExecutePlan", execute(range(5), OP2))
                .await
                .expect("admitted");
            drain(s).await
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(!small.is_finished(), "it waits for the global budget");
    release_all(&run, OP).await;
    let read = tokio::time::timeout(Duration::from_secs(10), small)
        .await
        .expect("it runs once the hog is released")
        .expect("join");
    read.end.expect("ok");
    assert!(read.responses.last().is_some_and(is_complete));
    release_all(&run, OP2).await;
    drop(hog);
    eventually("the queries stop", || session.running_queries() == 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reattach_racing_the_old_sender() {
    let run = start(ExecuteLimits::default()).await;
    run.session(10);
    let mut first = stream(&run.channel, "ExecutePlan", execute(range(2000), OP))
        .await
        .expect("execute");
    let schema = first.message().await.expect("ok").expect("schema");
    // Eight reattaches from the same point race each other and the first
    // stream; each sender either sends or is interrupted with DISCONNECTED.
    let mut racers = Vec::new();
    for _ in 0..8 {
        let channel = run.channel.clone();
        let last = schema.response_id.clone();
        racers.push(tokio::spawn(async move {
            match stream(&channel, "ReattachExecute", reattach(OP, Some(&last))).await {
                Ok(s) => drain(s).await.end,
                Err(e) => Err(e),
            }
        }));
    }
    let old = drain(first).await;
    if let Err(e) = old.end {
        assert_eq!(error_class(&e), "INVALID_CURSOR.DISCONNECTED");
    }
    for r in racers {
        if let Err(e) = r.await.expect("join") {
            assert_eq!(error_class(&e), "INVALID_CURSOR.DISCONNECTED");
        }
    }
    // A last reattach from the schema gets every batch exactly once.
    let (rest, _) = reattach_to_end(&run, OP, Some(schema.response_id)).await;
    assert_every_batch_once(&rest, 2000, 10);
    release_all(&run, OP).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn release_racing_the_encoder() {
    let run = start(ExecuteLimits {
        ahead_bytes: 16 * 1024,
        ..ExecuteLimits::default()
    })
    .await;
    let session = run.session(10);
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let mut live = stream(&run.channel, "ExecutePlan", execute(range(5000), OP))
        .await
        .expect("execute");
    // As the stock client: release_until after every response, sent
    // concurrently while the encoder keeps producing.
    let mut seen = Vec::new();
    let mut releases = Vec::new();
    while let Some(r) = live.message().await.expect("ok") {
        let channel = run.channel.clone();
        let id = r.response_id.clone();
        releases.push(tokio::spawn(async move {
            unary::<_, pb::ReleaseExecuteResponse>(
                &channel,
                "ReleaseExecute",
                release(OP, until(&id)),
            )
            .await
        }));
        let done = is_complete(&r);
        seen.push(r);
        if done {
            break;
        }
    }
    for r in releases {
        r.await.expect("join").expect("release until");
    }
    assert_every_batch_once(&seen, 5000, 10);
    release_all(&run, OP).await;
    eventually("the memory returns", || memory.reserved() == before).await;

    // release_all racing a producing encoder.
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        execute(range(100_000_000_000), OP2),
    )
    .await
    .expect("execute");
    for _ in 0..20 {
        live.message().await.expect("ok").expect("response");
    }
    release_all(&run, OP2).await;
    drop(live);
    eventually("the query stops", || session.running_queries() == 0).await;
    eventually("the memory returns", || memory.reserved() == before).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn maintenance_racing_reattach() {
    let run = start(ExecuteLimits {
        detached_timeout: Duration::from_millis(30),
        maintenance_interval: Duration::from_millis(10),
        ..ExecuteLimits::default()
    })
    .await;
    let session = run.session(10);
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let mut live = stream(
        &run.channel,
        "ExecutePlan",
        execute(range(100_000_000_000), OP),
    )
    .await
    .expect("execute");
    let mut last = live
        .message()
        .await
        .expect("ok")
        .expect("schema")
        .response_id;
    drop(live);
    // Reattach over and over across the detach timeout: each reattach
    // either resumes or finds the operation abandoned, never anything else.
    let mut abandoned = false;
    for i in 0_u64..200 {
        tokio::time::sleep(Duration::from_millis((i % 5) * 10)).await;
        match stream(&run.channel, "ReattachExecute", reattach(OP, Some(&last))).await {
            Ok(mut s) => match s.message().await {
                Ok(Some(r)) => last = r.response_id,
                Ok(None) => {}
                Err(e) => {
                    let class = error_class(&e);
                    assert!(
                        class == "INVALID_CURSOR.DISCONNECTED"
                            || class == "INVALID_HANDLE.OPERATION_ABANDONED",
                        "{class}"
                    );
                }
            },
            Err(e) => {
                assert_eq!(error_class(&e), "INVALID_HANDLE.OPERATION_ABANDONED");
                abandoned = true;
                break;
            }
        }
    }
    if !abandoned {
        // Detach for good: maintenance abandons it.
        eventually("abandoned", || run.entry().operations().is_empty()).await;
    }
    eventually("the query stops", || session.running_queries() == 0).await;
    eventually("the memory returns", || memory.reserved() == before).await;
}
