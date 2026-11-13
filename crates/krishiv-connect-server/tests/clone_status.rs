// SPDX-License-Identifier: Apache-2.0
//! `CloneSession` and `GetStatus` end to end over a raw tonic client
//! (design P3-T011a §3.11, §3.12, §5: `clone_session_*`,
//! `get_status_states`; PR-15c).
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]
#![expect(
    clippy::disallowed_methods,
    reason = "the tests read back status details and what is on disk"
)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::add_artifacts_request::{
    ArtifactChunk, Batch, Payload, SingleChunkArtifact,
};
use krishiv_connect::proto::v4_3::config_request::operation::OpType;
use krishiv_connect::proto::v4_3::config_request::{self as creq, Operation};
use krishiv_connect::proto::v4_3::execute_plan_request::RequestOption as ReqOpt;
use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
use krishiv_connect::proto::v4_3::get_status_request::OperationStatusRequest;
use krishiv_connect::proto::v4_3::get_status_response::operation_status::OperationState;
use krishiv_connect::proto::v4_3::interrupt_request::{Interrupt, InterruptType};
use krishiv_connect::proto::v4_3::plan::OpType as PlanOp;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::proto::v4_3::release_execute_request::{Release, ReleaseAll};
use krishiv_connect_server::artifacts::{ARTIFACT_OVERHEAD, crc32};
use krishiv_connect_server::{
    ArtifactLimits, ConnectServer, ConnectServerConfig, ExecuteLimits, SessionEntry, SessionLimits,
    SessionManager,
};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const TARGET: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";
const TARGET2: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f22";
const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";
const OP2: &str = "00112233-4455-6677-8899-aabbccddeef0";

struct Running {
    manager: Arc<SessionManager>,
    channel: Channel,
    parent: tempfile::TempDir,
    _stop: oneshot::Sender<()>,
}

impl Running {
    fn entry(&self) -> Arc<SessionEntry> {
        self.manager
            .get_or_create(&Principal::local_user(), SESSION, None, "u")
            .expect("session")
    }

    fn get(&self, id: &str) -> Arc<SessionEntry> {
        self.manager
            .get(&Principal::local_user(), id, None)
            .expect("session")
    }
}

#[derive(Default)]
struct Limits {
    sessions: SessionLimits,
    artifacts: ArtifactLimits,
    execute: ExecuteLimits,
}

async fn start_with(limits: Limits) -> Running {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        limits.sessions,
    ));
    let parent = tempfile::tempdir().expect("tempdir");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let config = ConnectServerConfig {
        artifacts: ArtifactLimits {
            parent: Some(parent.path().to_owned()),
            ..limits.artifacts
        },
        execute: limits.execute,
        ..ConnectServerConfig::default()
    };
    let server = ConnectServer::new(Arc::clone(&manager), config);
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
        parent,
        _stop: stop,
    }
}

async fn start() -> Running {
    start_with(Limits::default()).await
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

async fn execute(
    channel: &Channel,
    request: pb::ExecutePlanRequest,
) -> Result<tonic::Streaming<pb::ExecutePlanResponse>, tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.server_streaming(
        tonic::Request::new(request),
        path("ExecutePlan"),
        tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// Every response of a stream, or the status it ended with.
async fn drain(
    mut s: tonic::Streaming<pb::ExecutePlanResponse>,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let mut out = Vec::new();
    while let Some(r) = s.message().await? {
        out.push(r);
    }
    Ok(out)
}

fn sql_plan(query: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Sql(pb::Sql {
                query: query.to_owned(),
                ..Default::default()
            })),
        }))),
    }
}

fn range_plan(end: i64) -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(pb::Relation {
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

fn exec_request(session: &str, plan: pb::Plan, op: Option<&str>) -> pb::ExecutePlanRequest {
    pb::ExecutePlanRequest {
        session_id: session.to_owned(),
        plan: Some(plan),
        operation_id: op.map(str::to_owned),
        request_options: vec![ReqOpt {
            request_option: Some(RequestOption::ReattachOptions(pb::ReattachOptions {
                reattachable: true,
            })),
        }],
        ..Default::default()
    }
}

/// The rows of `query` on `session`, as the text of the first column.
async fn rows(channel: &Channel, session: &str, query: &str) -> Result<usize, tonic::Status> {
    let responses =
        drain(execute(channel, exec_request(session, sql_plan(query), None)).await?).await?;
    Ok(responses
        .iter()
        .filter_map(|r| match &r.response_type {
            Some(pb::execute_plan_response::ResponseType::ArrowBatch(b)) => {
                Some(usize::try_from(b.row_count).expect("rows"))
            }
            _ => None,
        })
        .sum())
}

fn clone_request(new: Option<&str>) -> pb::CloneSessionRequest {
    pb::CloneSessionRequest {
        session_id: SESSION.to_owned(),
        new_session_id: new.map(str::to_owned),
        client_type: Some("python".to_owned()),
        ..Default::default()
    }
}

async fn clone(
    channel: &Channel,
    request: pb::CloneSessionRequest,
) -> Result<pb::CloneSessionResponse, tonic::Status> {
    unary(channel, "CloneSession", request).await
}

fn config_set(session: &str, pairs: &[(&str, &str)]) -> pb::ConfigRequest {
    pb::ConfigRequest {
        session_id: session.to_owned(),
        operation: Some(Operation {
            op_type: Some(OpType::Set(creq::Set {
                pairs: pairs
                    .iter()
                    .map(|(k, v)| pb::KeyValue {
                        key: (*k).to_owned(),
                        value: Some((*v).to_owned()),
                    })
                    .collect(),
                silent: None,
            })),
        }),
        ..Default::default()
    }
}

async fn add_artifacts(
    channel: &Channel,
    session: &str,
    artifacts: &[(&str, &[u8])],
) -> Result<pb::AddArtifactsResponse, tonic::Status> {
    let request = pb::AddArtifactsRequest {
        session_id: session.to_owned(),
        payload: Some(Payload::Batch(Batch {
            artifacts: artifacts
                .iter()
                .map(|(name, data)| SingleChunkArtifact {
                    name: (*name).to_owned(),
                    data: Some(ArtifactChunk {
                        data: data.to_vec(),
                        crc: i64::from(crc32(data)),
                    }),
                })
                .collect(),
        })),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.client_streaming(
        tonic::Request::new(futures::stream::iter(vec![request])),
        path("AddArtifacts"),
        tonic_prost::ProstCodec::<pb::AddArtifactsRequest, pb::AddArtifactsResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
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

fn info(status: &tonic::Status) -> BTreeMap<String, String> {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    ErrorInfo::decode(rpc.details[0].value.as_slice())
        .expect("ErrorInfo")
        .metadata
}

fn error_class(status: &tonic::Status) -> String {
    info(status).get("errorClass").cloned().unwrap_or_default()
}

fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_owned()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries {
            let p = e.expect("entry").path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(
                    p.strip_prefix(dir)
                        .expect("under")
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    out.sort();
    out
}

async fn eventually(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ----------------------------------------------------------- CloneSession

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_session_copies_the_session_and_its_artifacts() {
    let run = start().await;
    let source = run.entry();
    source
        .session()
        .sql("CREATE TEMPORARY VIEW v AS SELECT 1 AS x UNION ALL SELECT 2")
        .await
        .expect("view");
    unary::<_, pb::ConfigResponse>(
        &run.channel,
        "Config",
        config_set(SESSION, &[("my.flag", "on")]),
    )
    .await
    .expect("set");
    add_artifacts(
        &run.channel,
        SESSION,
        &[("pyfiles/a/b.py", b"print(1)"), ("cache/abc123", b"blob")],
    )
    .await
    .expect("add");
    let before = run.manager.counts().live;

    let response = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .expect("clone");
    assert_eq!(response.session_id, SESSION);
    assert_eq!(response.server_side_session_id, source.server_side_id());
    assert_eq!(response.new_session_id, TARGET);
    let clone = run.get(TARGET);
    assert_eq!(response.new_server_side_session_id, clone.server_side_id());
    assert_ne!(clone.server_side_id(), source.server_side_id());
    assert_eq!(run.manager.counts().live, before + 1);

    // Configuration and temporary views are copied, then diverge.
    assert_eq!(clone.session().conf().get("my.flag").as_deref(), Some("on"));
    // (A Connect `Sql` relation cannot read a view yet: asked of the engine.)
    let batches = clone
        .session()
        .sql("SELECT x FROM v")
        .await
        .expect("view")
        .collect()
        .await
        .expect("rows");
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    assert_eq!(
        rows(&run.channel, TARGET, "SELECT 1").await.expect("query"),
        1
    );
    unary::<_, pb::ConfigResponse>(
        &run.channel,
        "Config",
        config_set(TARGET, &[("my.flag", "off")]),
    )
    .await
    .expect("set");
    assert_eq!(
        source.session().conf().get("my.flag").as_deref(),
        Some("on")
    );

    // Artifacts: the file in a directory of its own, the cached blob, the
    // same charge as the source's.
    let source_dir = source.artifacts().dir_path().expect("source dir");
    let clone_dir = clone.artifacts().dir_path().expect("clone dir");
    assert_ne!(source_dir, clone_dir);
    assert_eq!(files_under(&clone_dir), vec!["pyfiles/a/b.py".to_owned()]);
    assert_eq!(
        std::fs::read(clone_dir.join("pyfiles/a/b.py")).expect("read"),
        b"print(1)"
    );
    assert_eq!(
        clone.artifacts().cached("abc123").as_deref(),
        Some(&b"blob"[..])
    );
    assert_eq!(
        clone.artifacts().used_bytes(),
        source.artifacts().used_bytes()
    );
    let status: pb::ArtifactStatusesResponse = unary(
        &run.channel,
        "ArtifactStatus",
        pb::ArtifactStatusesRequest {
            session_id: TARGET.to_owned(),
            names: vec!["cache/abc123".to_owned()],
            ..Default::default()
        },
    )
    .await
    .expect("status");
    assert!(status.statuses["cache/abc123"].exists);

    // Releasing the source leaves the clone's artifacts in place.
    unary::<_, pb::ReleaseSessionResponse>(
        &run.channel,
        "ReleaseSession",
        pb::ReleaseSessionRequest {
            session_id: SESSION.to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("release");
    eventually("the source's directory to go", || !source_dir.exists()).await;
    assert_eq!(files_under(&clone_dir), vec!["pyfiles/a/b.py".to_owned()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_session_without_an_id_mints_a_uuid() {
    let run = start().await;
    let _ = run.entry();
    let response = clone(&run.channel, clone_request(None))
        .await
        .expect("clone");
    assert_eq!(response.new_session_id.len(), 36);
    assert_eq!(&response.new_session_id[14..15], "4", "a version 4 UUID");
    let entry = run.get(&response.new_session_id);
    assert_eq!(response.new_server_side_session_id, entry.server_side_id());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_session_target_checks() {
    let run = start().await;
    let source = run.entry();
    // Not a UUID.
    let e = clone(&run.channel, clone_request(Some("not-a-uuid")))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_FORMAT"
    );
    let meta = info(&e);
    assert_eq!(meta["sqlState"], "42K04");
    assert!(
        meta["classes"].contains("org.apache.spark.SparkSQLException"),
        "{meta:?}"
    );
    assert!(meta.contains_key("errorId"), "sent on the source session");
    // Live: the source itself, and an existing clone.
    let e = clone(&run.channel, clone_request(Some(SESSION)))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_ALREADY_EXISTS"
    );
    clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .expect("clone");
    let e = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_ALREADY_EXISTS"
    );
    // Released (tombstoned).
    run.manager.release(&Principal::local_user(), TARGET, false);
    let e = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&e),
        "INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_ALREADY_CLOSED"
    );
    assert_eq!(info(&e)["sqlState"], "42K04");
    // Held by another principal: refused as on every create.
    let bob = Principal::new("bob", PrincipalKind::User);
    run.manager
        .get_or_create(&bob, TARGET2, None, "bob")
        .expect("bob's");
    let e = clone(&run.channel, clone_request(Some(TARGET2)))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert!(
        run.manager
            .get(&Principal::local_user(), TARGET2, None)
            .is_err()
    );
    // Nothing was created by the failures: the source and bob's session.
    assert_eq!(run.manager.counts().live, 2);
    assert!(Arc::ptr_eq(&run.entry(), &source));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_session_source_checks() {
    let run = start().await;
    // Unknown: lookup only, nothing created.
    let e = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert!(!info(&e).contains_key("errorId"));
    assert_eq!(run.manager.counts().live, 0);
    // A changed server-side id.
    let _ = run.entry();
    let mut request = clone_request(Some(TARGET));
    request.client_observed_server_side_session_id = Some("someone-else".to_owned());
    let e = clone(&run.channel, request).await.unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_CHANGED");
    // Released.
    run.manager
        .release(&Principal::local_user(), SESSION, false);
    let e = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_CLOSED");
    assert_eq!(run.manager.counts().live, 0);
}

/// ADR 0046 Validation, threat model T-Z6: another principal's session
/// cannot be cloned, and the answer is the one an unknown id gets.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_own_session_only() {
    let run = start().await;
    let bob = Principal::new("bob", PrincipalKind::User);
    run.manager
        .get_or_create(&bob, SESSION, None, "bob")
        .expect("bob's");
    let e = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert_eq!(run.manager.counts().live, 1);
    assert!(run.manager.get(&bob, TARGET, None).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_session_counts_against_the_session_caps() {
    let run = start_with(Limits {
        sessions: SessionLimits {
            max_sessions_per_principal: 2,
            ..SessionLimits::default()
        },
        ..Limits::default()
    })
    .await;
    let _ = run.entry();
    clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .expect("first clone");
    let e = clone(&run.channel, clone_request(Some(TARGET2)))
        .await
        .unwrap_err();
    assert_eq!(e.code(), tonic::Code::ResourceExhausted);
    assert_eq!(error_class(&e), "KRISHIV_SESSION_LIMIT");
    assert_eq!(run.manager.counts().live, 2);

    let run = start_with(Limits {
        sessions: SessionLimits {
            max_sessions: 1,
            ..SessionLimits::default()
        },
        ..Limits::default()
    })
    .await;
    let _ = run.entry();
    let e = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "KRISHIV_SESSION_LIMIT");
    assert!(info(&e)["messageParameters"].contains("krishiv.connect.maxSessions"));
    assert_eq!(run.manager.counts().live, 1);
}

/// The copy is charged to the server-wide cap; one that does not fit fails
/// and leaves no session, no file and no quota behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_artifacts_past_the_server_cap_leave_nothing_behind() {
    let data = vec![7u8; 40_000];
    let run = start_with(Limits {
        artifacts: ArtifactLimits {
            max_total_bytes: 64_000,
            max_artifact_bytes: 64_000,
            ..ArtifactLimits::default()
        },
        ..Limits::default()
    })
    .await;
    let source = run.entry();
    add_artifacts(
        &run.channel,
        SESSION,
        &[("files/small.txt", b"tiny"), ("files/big.bin", &data)],
    )
    .await
    .expect("add");
    let held = source.artifacts().used_bytes();
    assert_eq!(held, 4 + 40_000 + 4 * ARTIFACT_OVERHEAD);
    let root = source
        .artifacts()
        .dir_path()
        .and_then(|d| d.parent().map(Path::to_owned))
        .expect("root");
    let dirs_before = std::fs::read_dir(&root).expect("root").count();
    let e = clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .unwrap_err();
    assert_eq!(e.code(), tonic::Code::ResourceExhausted);
    assert_eq!(error_class(&e), "KRISHIV_ARTIFACT_LIMIT");
    assert!(info(&e)["messageParameters"].contains("krishiv.connect.artifacts.maxTotalBytes"));
    assert_eq!(run.manager.counts().live, 1);
    assert!(
        run.manager
            .get(&Principal::local_user(), TARGET, None)
            .is_err()
    );
    eventually("the clone's directory to go", || {
        std::fs::read_dir(&root).expect("root").count() == dirs_before
    })
    .await;
    assert_eq!(source.artifacts().used_bytes(), held);
    // The target id is still free: with room, the clone succeeds.
    assert!(run.parent.path().exists());
    let run2 = start().await;
    let _ = run2.entry();
    add_artifacts(&run2.channel, SESSION, &[("files/big.bin", &data)])
        .await
        .expect("add");
    clone(&run2.channel, clone_request(Some(TARGET)))
        .await
        .expect("clone with room");
}

/// The per-session quota counts the copy too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clone_artifacts_fit_the_session_quota_as_uploads_do() {
    let run = start_with(Limits {
        artifacts: ArtifactLimits {
            max_session_bytes: 3 * ARTIFACT_OVERHEAD + 10,
            ..ArtifactLimits::default()
        },
        ..Limits::default()
    })
    .await;
    let source = run.entry();
    add_artifacts(&run.channel, SESSION, &[("files/a/b", b"0123456789")])
        .await
        .expect("add");
    clone(&run.channel, clone_request(Some(TARGET)))
        .await
        .expect("an exact fit");
    assert_eq!(
        run.get(TARGET).artifacts().used_bytes(),
        source.artifacts().used_bytes()
    );
}

/// Threaded: a clone of a session receiving concurrent `Config` sets sees
/// each request whole (both keys of one request, or neither), and a clone
/// does not wait for a running operation's stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clone_serialises_with_concurrent_statements_on_the_source() {
    let run = Arc::new(start().await);
    let _ = run.entry();
    unary::<_, pb::ConfigResponse>(
        &run.channel,
        "Config",
        config_set(SESSION, &[("a", "0"), ("b", "0")]),
    )
    .await
    .expect("set");
    // A running operation whose client reads nothing.
    let held = execute(
        &run.channel,
        exec_request(SESSION, range_plan(10_000_000), Some(OP)),
    )
    .await
    .expect("stream");
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer = {
        let run = Arc::clone(&run);
        let stop = Arc::clone(&stop);
        tokio::spawn(async move {
            let mut i = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                i += 1;
                let v = i.to_string();
                let pairs: Vec<(&str, &str)> = (0..32)
                    .map(|_| ("a", v.as_str()))
                    .chain(std::iter::once(("b", v.as_str())))
                    .collect();
                unary::<_, pb::ConfigResponse>(&run.channel, "Config", config_set(SESSION, &pairs))
                    .await
                    .expect("set");
            }
            i
        })
    };
    for n in 0..40u32 {
        let target = format!("1d6a3c1e-77b2-4c55-8d0f-{n:012x}");
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            clone(&run.channel, clone_request(Some(&target))),
        )
        .await
        .expect("a clone does not wait for the stream")
        .expect("clone");
        let conf = run.get(&response.new_session_id).session().conf();
        assert_eq!(
            conf.get("a"),
            conf.get("b"),
            "clone {n} copied half a request"
        );
        run.manager
            .release(&Principal::local_user(), &target, false);
    }
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert!(writer.await.expect("writer") > 0);
    drop(held);
}

// -------------------------------------------------------------- GetStatus

fn status_request(ids: Option<&[&str]>) -> pb::GetStatusRequest {
    pb::GetStatusRequest {
        session_id: SESSION.to_owned(),
        operation_status: ids.map(|ids| OperationStatusRequest {
            operation_ids: ids.iter().map(|s| (*s).to_owned()).collect(),
            extensions: Vec::new(),
        }),
        // Ignored.
        extensions: vec![prost_types::Any {
            type_url: "type.googleapis.com/some.Extension".to_owned(),
            value: vec![1, 2, 3],
        }],
        ..Default::default()
    }
}

async fn get_status(run: &Running, ids: Option<&[&str]>) -> Vec<(String, OperationState)> {
    let response: pb::GetStatusResponse = unary(&run.channel, "GetStatus", status_request(ids))
        .await
        .expect("status");
    assert_eq!(response.session_id, SESSION);
    assert_eq!(
        response.server_side_session_id,
        run.entry().server_side_id()
    );
    assert!(response.extensions.is_empty());
    response
        .operation_statuses
        .into_iter()
        .map(|s| {
            let state = OperationState::try_from(s.state).expect("state");
            assert_ne!(state, OperationState::Unspecified);
            (s.operation_id, state)
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_status_states() {
    let run = start_with(Limits {
        execute: ExecuteLimits {
            ahead_bytes: 1024,
            ..ExecuteLimits::default()
        },
        ..Limits::default()
    })
    .await;
    let _ = run.entry();
    assert!(get_status(&run, None).await.is_empty(), "absent: none");
    assert!(
        get_status(&run, Some(&[])).await.is_empty(),
        "no operation yet"
    );

    // Finished, its stream read but not released: SUCCEEDED.
    let done = drain(
        execute(&run.channel, exec_request(SESSION, range_plan(3), Some(OP)))
            .await
            .expect("stream"),
    )
    .await
    .expect("rows");
    assert!(!done.is_empty());
    // Running: its client reads nothing past the ahead budget.
    let running = execute(
        &run.channel,
        exec_request(SESSION, range_plan(10_000_000), Some(OP2)),
    )
    .await
    .expect("stream");
    let mut statuses = get_status(&run, Some(&[OP, OP2, OP, "not-a-uuid", SESSION])).await;
    assert_eq!(
        statuses,
        vec![
            (OP.to_owned(), OperationState::Succeeded),
            (OP2.to_owned(), OperationState::Running),
            ("not-a-uuid".to_owned(), OperationState::Unknown),
            (SESSION.to_owned(), OperationState::Unknown),
        ],
        "deduplicated, in the order asked"
    );
    // Interrupted: CANCELLED once its encoder stopped.
    let _: pb::InterruptResponse = unary(
        &run.channel,
        "Interrupt",
        pb::InterruptRequest {
            session_id: SESSION.to_owned(),
            interrupt_type: InterruptType::OperationId.into(),
            interrupt: Some(Interrupt::OperationId(OP2.to_owned())),
            ..Default::default()
        },
    )
    .await
    .expect("interrupt");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        statuses = get_status(&run, Some(&[OP2])).await;
        let state = statuses[0].1;
        assert!(
            matches!(
                state,
                OperationState::Terminating | OperationState::Cancelled
            ),
            "{state:?}"
        );
        if state == OperationState::Cancelled {
            break;
        }
        assert!(Instant::now() < deadline, "never cancelled");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    drop(running);

    // Released: it leaves the session, its final state stays.
    let _: pb::ReleaseExecuteResponse = unary(
        &run.channel,
        "ReleaseExecute",
        pb::ReleaseExecuteRequest {
            session_id: SESSION.to_owned(),
            operation_id: OP.to_owned(),
            release: Some(Release::ReleaseAll(ReleaseAll {})),
            ..Default::default()
        },
    )
    .await
    .expect("release");
    assert!(run.entry().operations().get(OP).is_none());
    assert_eq!(
        get_status(&run, Some(&[OP])).await,
        vec![(OP.to_owned(), OperationState::Succeeded)]
    );

    // Failed at run time.
    let op3 = "00112233-4455-6677-8899-aabbccddeef1";
    let failed = drain(
        execute(
            &run.channel,
            exec_request(SESSION, sql_plan("SELECT raise_error('boom')"), Some(op3)),
        )
        .await
        .expect("stream"),
    )
    .await;
    assert!(failed.is_err());
    // Failed before it started (analysis).
    let op4 = "00112233-4455-6677-8899-aabbccddeef2";
    assert!(
        execute(
            &run.channel,
            exec_request(SESSION, sql_plan("SELECT * FROM no_such_table"), Some(op4)),
        )
        .await
        .is_err()
    );
    assert_eq!(
        get_status(&run, Some(&[op3, op4])).await,
        vec![
            (op3.to_owned(), OperationState::Failed),
            (op4.to_owned(), OperationState::Failed),
        ]
    );

    // Empty ids: every operation the session knows, live and ended.
    let mut all = get_status(&run, Some(&[])).await;
    all.sort();
    let mut want = vec![
        (OP.to_owned(), OperationState::Succeeded),
        (OP2.to_owned(), OperationState::Cancelled),
        (op3.to_owned(), OperationState::Failed),
        (op4.to_owned(), OperationState::Failed),
    ];
    want.sort();
    assert_eq!(all, want);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_status_only_looks_up_and_checks_the_server_side_id() {
    let run = start().await;
    let e = unary::<_, pb::GetStatusResponse>(&run.channel, "GetStatus", status_request(Some(&[])))
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert_eq!(run.manager.counts().live, 0);
    let _ = run.entry();
    let mut request = status_request(None);
    request.client_observed_server_side_session_id = Some("someone-else".to_owned());
    let e = unary::<_, pb::GetStatusResponse>(&run.channel, "GetStatus", request)
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_CHANGED");
    // Another principal's operations are not seen.
    let bob = Principal::new("bob", PrincipalKind::User);
    let bobs = run
        .manager
        .get_or_create(&bob, TARGET, None, "bob")
        .expect("bob's");
    assert!(bobs.operations().is_empty());
    let mut request = status_request(Some(&[]));
    request.session_id = TARGET.to_owned();
    let e = unary::<_, pb::GetStatusResponse>(&run.channel, "GetStatus", request)
        .await
        .unwrap_err();
    assert_eq!(error_class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");
}
