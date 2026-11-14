// SPDX-License-Identifier: Apache-2.0
//! `ExecutePlan` end to end over a raw tonic client with the test-only
//! fault points of `execute::hooks`: an encoder that panics (review L4,
//! PR-15a) and a cancel or release while the plan is being prepared (review
//! L3, carried from #727).
#![expect(
    clippy::disallowed_methods,
    reason = "the tests block on their runtime and read back status details"
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_request::RequestOption as ReqOpt;
use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
use krishiv_connect::proto::v4_3::interrupt_request::{Interrupt, InterruptType};
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::proto::v4_3::release_execute_request::{Release, ReleaseAll};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

use super::ResponseBudget;
use super::hooks::{self, Fault};
use crate::{ConnectServer, ConnectServerConfig, SessionEntry, SessionLimits, SessionManager};

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";

struct Running {
    manager: Arc<SessionManager>,
    channel: Channel,
    budget: ResponseBudget,
    _stop: oneshot::Sender<()>,
}

impl Running {
    fn entry(&self) -> Arc<SessionEntry> {
        self.manager
            .get_or_create(&Principal::local_user(), SESSION, None, "u")
            .expect("session")
    }
}

async fn start() -> Running {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits::default(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(Arc::clone(&manager), ConnectServerConfig::default());
    let budget = server.response_budget().clone();
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
        budget,
        _stop: stop,
    }
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

/// A fresh operation id: fault points are keyed by it.
fn fresh_id() -> String {
    crate::operation::new_operation_id()
        .expect("id")
        .as_str()
        .to_owned()
}

fn execute(op: &str, reattachable: bool) -> pb::ExecutePlanRequest {
    pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        operation_id: Some(op.to_owned()),
        plan: Some(pb::Plan {
            op_type: Some(OpType::Root(Box::new(pb::Relation {
                common: None,
                rel_type: Some(RelType::Range(pb::Range {
                    start: Some(0),
                    end: 10,
                    step: 1,
                    num_partitions: Some(1),
                })),
            }))),
        }),
        request_options: vec![ReqOpt {
            request_option: Some(RequestOption::ReattachOptions(pb::ReattachOptions {
                reattachable,
            })),
        }],
        ..Default::default()
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

/// How `s` ends after its responses (within 10 s: an orphan never ends).
async fn end_of(mut s: Responses) -> (usize, Result<(), tonic::Status>) {
    let mut n = 0;
    let end = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match s.message().await {
                Ok(Some(_)) => n += 1,
                Ok(None) => break Ok(()),
                Err(e) => break Err(e),
            }
        }
    })
    .await
    .expect("the stream ends");
    (n, end)
}

async fn eventually(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(tokio::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn release_all(run: &Running, op: &str) {
    let _: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        pb::ReleaseExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: op.to_owned(),
            release: Some(Release::ReleaseAll(ReleaseAll {})),
            ..Default::default()
        },
    )
    .await
    .expect("release all");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_encoder_panic_ends_a_non_reattachable_stream_with_internal_error() {
    let run = start().await;
    let entry = run.entry();
    let op = fresh_id();
    hooks::inject(&op, Fault::PanicInEncoder);
    let s = stream(&run.channel, "ExecutePlan", execute(&op, false))
        .await
        .expect("execute");
    let (n, end) = end_of(s).await;
    assert!(hooks::reached(&op));
    assert_eq!(n, 1, "the schema, then the error");
    let e = end.expect_err("the stream fails");
    assert_eq!(error_class(&e), "INTERNAL_ERROR");
    eventually("the operation leaves", || entry.operations().is_empty()).await;
    eventually("the budget returns", || {
        run.budget.available() == run.budget.cap()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_encoder_panic_ends_a_reattachable_operation_with_internal_error() {
    let run = start().await;
    let entry = run.entry();
    let op = fresh_id();
    hooks::inject(&op, Fault::PanicInEncoder);
    let s = stream(&run.channel, "ExecutePlan", execute(&op, true))
        .await
        .expect("execute");
    let (_, end) = end_of(s).await;
    assert_eq!(error_class(&end.expect_err("fails")), "INTERNAL_ERROR");
    // Not an orphan: a reattach ends with the same error at once rather
    // than reattaching forever.
    let again = stream(
        &run.channel,
        "ReattachExecute",
        pb::ReattachExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: op.clone(),
            ..Default::default()
        },
    )
    .await
    .expect("reattach");
    let (_, end) = end_of(again).await;
    assert_eq!(error_class(&end.expect_err("fails")), "INTERNAL_ERROR");
    let live = entry.operations().get(&op).expect("kept until released");
    assert!(!live.is_active(), "settled: shutdown does not wait for it");
    release_all(&run, &op).await;
    assert!(entry.operations().is_empty());
    eventually("the budget returns", || {
        run.budget.available() == run.budget.cap()
    })
    .await;
}

/// Starts an `ExecutePlan` whose preparation holds, and waits until it does.
async fn held(
    run: &Running,
    op: &str,
    reattachable: bool,
) -> tokio::task::JoinHandle<Result<Responses, tonic::Status>> {
    hooks::inject(op, Fault::HoldPreparation);
    let channel = run.channel.clone();
    let request = execute(op, reattachable);
    let call = tokio::spawn(async move { stream(&channel, "ExecutePlan", request).await });
    eventually("the preparation holds", || hooks::reached(op)).await;
    assert_eq!(run.entry().operations().len(), 1, "registered");
    call
}

async fn refused(call: tokio::task::JoinHandle<Result<Responses, tonic::Status>>) -> String {
    let answer = tokio::time::timeout(Duration::from_secs(10), call)
        .await
        .expect("the preparation stops")
        .expect("joined");
    match answer {
        Ok(_) => panic!("the operation started"),
        Err(e) => error_class(&e),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_interrupt_during_plan_preparation_cancels_it() {
    let run = start().await;
    let op = fresh_id();
    let call = held(&run, &op, false).await;
    let answer: pb::InterruptResponse = unary(
        &run.channel,
        "Interrupt",
        pb::InterruptRequest {
            session_id: SESSION.to_owned(),
            interrupt_type: InterruptType::OperationId as i32,
            interrupt: Some(Interrupt::OperationId(op.clone())),
            ..Default::default()
        },
    )
    .await
    .expect("interrupt");
    assert_eq!(answer.interrupted_ids, vec![op.clone()]);
    assert_eq!(refused(call).await, "OPERATION_CANCELED");
    assert!(run.entry().operations().is_empty(), "nothing leaks");
    assert_eq!(run.budget.available(), run.budget.cap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_release_during_plan_preparation_cancels_it() {
    let run = start().await;
    let op = fresh_id();
    let call = held(&run, &op, true).await;
    release_all(&run, &op).await;
    assert_eq!(refused(call).await, "OPERATION_CANCELED");
    assert!(run.entry().operations().is_empty(), "nothing leaks");
    assert_eq!(run.budget.available(), run.budget.cap());
    assert_eq!(run.entry().session().running_queries(), 0);
}
