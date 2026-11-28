// SPDX-License-Identifier: Apache-2.0
//! The cache arms over Spark Connect (design P3-T083 §3.4, PR-47), end to end over a raw tonic
//! client: `AnalyzePlan` `persist`, `unpersist` and `get_storage_level`; `checkpoint_command`
//! answering a `CachedRemoteRelation` the session holds; the relation read back (alone, under
//! another relation, twice); `remove_cached_remote_relation_command`; the error of an id the
//! session does not hold. Spark's handlers: `SparkConnectAnalyzeHandler` and
//! `SparkConnectPlanner.handleCheckpointCommand` (identical at v4.0.0 and v4.2.0).
#![expect(
    clippy::expect_used,
    clippy::panic,
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
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::analyze_plan_request::{self as areq, Analyze};
use krishiv_connect::proto::v4_3::analyze_plan_response::Result as AResult;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
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

/// A server whose sessions have `spark.checkpoint.dir` set to `dir`, when given.
async fn start(dir: Option<String>) -> Running {
    let factory = move |_: &krishiv_session::auth::Authenticated,
                        _: krishiv_session::Transport,
                        _: Option<krishiv_plan::analyzer::SparkProfile>| {
        let s = Session::new()?;
        if let Some(dir) = &dir {
            s.conf().set("spark.checkpoint.dir", dir)?;
        }
        Ok(s)
    };
    let manager = Arc::new(SessionManager::new(
        Box::new(factory),
        SessionLimits::default(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(manager, ConnectServerConfig::default());
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

fn remote(id: &str) -> pb::Relation {
    relation(RelType::CachedRemoteRelation(pb::CachedRemoteRelation {
        relation_id: id.to_owned(),
    }))
}

fn limit(input: pb::Relation, n: i32) -> pb::Relation {
    relation(RelType::Limit(Box::new(pb::Limit {
        input: Some(Box::new(input)),
        limit: n,
    })))
}

fn root(r: pb::Relation) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(r))),
    }
}

fn command(c: pb::command::CommandType) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(c),
        }))),
    }
}

async fn analyze(run: &Running, analyze: Analyze) -> Result<AResult, tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let response: pb::AnalyzePlanResponse = grpc
        .unary(
            tonic::Request::new(pb::AnalyzePlanRequest {
                session_id: SESSION.to_owned(),
                analyze: Some(analyze),
                ..Default::default()
            }),
            path("AnalyzePlan"),
            tonic_prost::ProstCodec::<pb::AnalyzePlanRequest, pb::AnalyzePlanResponse>::default(),
        )
        .await?
        .into_inner();
    Ok(response.result.expect("a result"))
}

async fn execute(
    run: &Running,
    plan: pb::Plan,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let mut live = grpc
        .server_streaming(
            tonic::Request::new(pb::ExecutePlanRequest {
                session_id: SESSION.to_owned(),
                plan: Some(plan),
                ..Default::default()
            }),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(m) = live.message().await? {
        out.push(m);
    }
    Ok(out)
}

/// The ids of the first column of the rows of `responses`.
fn ids(responses: &[pb::ExecutePlanResponse]) -> Vec<i64> {
    let mut out = Vec::new();
    for r in responses {
        if let Some(ResponseType::ArrowBatch(b)) = &r.response_type {
            let batches: Vec<RecordBatch> = arrow_ipc::reader::StreamReader::try_new(
                std::io::Cursor::new(b.data.clone()),
                None,
            )
            .expect("ipc")
            .collect::<Result<_, _>>()
            .expect("batches");
            for batch in batches {
                out.extend(
                    batch
                        .column(0)
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .iter()
                        .flatten(),
                );
            }
        }
    }
    out
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

fn error_info(status: &tonic::Status) -> BTreeMap<String, String> {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    let info = ErrorInfo::decode(rpc.details[0].value.as_slice()).expect("ErrorInfo");
    let params = info
        .metadata
        .get("messageParameters")
        .cloned()
        .unwrap_or_default();
    serde_json::from_str(&params).expect("messageParameters is a JSON object")
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

fn level(use_disk: bool, use_memory: bool, replication: i32) -> pb::StorageLevel {
    pb::StorageLevel {
        use_disk,
        use_memory,
        use_off_heap: false,
        // Spark's `MEMORY_*` levels are deserialized, `DISK_ONLY` is not.
        deserialized: use_memory,
        replication,
    }
}

async fn storage_level(run: &Running, plan: &pb::Relation) -> pb::StorageLevel {
    let AResult::GetStorageLevel(r) = analyze(
        run,
        Analyze::GetStorageLevel(Box::new(areq::GetStorageLevel {
            relation: Some(plan.clone()),
        })),
    )
    .await
    .expect("storage level") else {
        panic!("get_storage_level");
    };
    r.storage_level.expect("a level")
}

async fn persist(
    run: &Running,
    plan: &pb::Relation,
    storage_level: Option<pb::StorageLevel>,
) -> Result<AResult, tonic::Status> {
    analyze(
        run,
        Analyze::Persist(Box::new(areq::Persist {
            relation: Some(plan.clone()),
            storage_level,
        })),
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persist_reports_and_unpersist_forgets_the_storage_level() {
    let run = start(None).await;
    let plan = range(10);
    // Uncached: Spark's `StorageLevel.NONE`.
    assert_eq!(storage_level(&run, &plan).await, level(false, false, 1));
    // Registered at once (Spark's `persist` runs no job), at the level asked, replication included.
    assert!(matches!(
        persist(&run, &plan, Some(level(true, false, 2))).await,
        Ok(AResult::Persist(_))
    ));
    assert_eq!(storage_level(&run, &plan).await, level(true, false, 2));
    // A second persist keeps the first level.
    persist(&run, &plan, Some(level(false, true, 1)))
        .await
        .expect("persist again");
    assert_eq!(storage_level(&run, &plan).await, level(true, false, 2));
    // The same plan built elsewhere sees it; another plan does not.
    assert_eq!(storage_level(&run, &range(10)).await, level(true, false, 2));
    assert_eq!(
        storage_level(&run, &range(11)).await,
        level(false, false, 1)
    );
    // `blocking` is accepted.
    for blocking in [None, Some(true)] {
        let r = analyze(
            &run,
            Analyze::Unpersist(Box::new(areq::Unpersist {
                relation: Some(plan.clone()),
                blocking,
            })),
        )
        .await
        .expect("unpersist");
        assert!(matches!(r, AResult::Unpersist(_)));
        assert_eq!(storage_level(&run, &plan).await, level(false, false, 1));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persist_without_a_level_uses_the_default_level() {
    let run = start(None).await;
    let plan = range(5);
    persist(&run, &plan, None).await.expect("persist");
    // `spark.sql.defaultCacheStorageLevel`: MEMORY_AND_DISK since Spark 4.0.0.
    assert_eq!(storage_level(&run, &plan).await, level(true, true, 1));
    // The cached plan still answers its rows.
    let r = execute(&run, root(plan)).await.expect("runs");
    assert_eq!(ids(&r), (0..5).collect::<Vec<_>>());
}

fn checkpoint(input: pb::Relation, local: bool, eager: bool) -> pb::Plan {
    command(pb::command::CommandType::CheckpointCommand(Box::new(
        pb::CheckpointCommand {
            relation: Some(input),
            local,
            eager,
            storage_level: None,
        },
    )))
}

/// The id a `checkpoint_command` answered.
fn relation_id(responses: &[pb::ExecutePlanResponse]) -> String {
    let results: Vec<_> = responses
        .iter()
        .filter_map(|r| match &r.response_type {
            Some(ResponseType::CheckpointCommandResult(c)) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1, "one checkpoint_command_result");
    results[0]
        .relation
        .as_ref()
        .expect("a relation")
        .relation_id
        .clone()
}

fn remove(id: &str) -> pb::Plan {
    command(pb::command::CommandType::RemoveCachedRemoteRelationCommand(
        Box::new(pb::RemoveCachedRemoteRelationCommand {
            relation: Some(pb::CachedRemoteRelation {
                relation_id: id.to_owned(),
            }),
        }),
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_local_checkpoint_is_held_under_an_id_and_read_back() {
    let run = start(None).await;
    for eager in [true, false] {
        let answered = execute(&run, checkpoint(range(5), true, eager))
            .await
            .expect("checkpoints");
        let id = relation_id(&answered);
        // A version 4 UUID, as Spark's `UUID.randomUUID().toString`.
        assert_eq!(id.len(), 36, "{id}");
        assert_eq!(id.as_bytes()[14], b'4', "{id}");
        // Alone, under another relation, and as both inputs of one plan.
        let r = execute(&run, root(remote(&id))).await.expect("reads");
        assert_eq!(ids(&r), (0..5).collect::<Vec<_>>(), "eager = {eager}");
        let r = execute(&run, root(limit(remote(&id), 2)))
            .await
            .expect("derived");
        assert_eq!(ids(&r), vec![0, 1], "eager = {eager}");
        let join = relation(RelType::Join(Box::new(pb::Join {
            left: Some(Box::new(remote(&id))),
            right: Some(Box::new(remote(&id))),
            join_type: pb::join::JoinType::Cross as i32,
            ..Default::default()
        })));
        let r = execute(&run, root(limit(join, 100))).await.expect("joined");
        assert_eq!(ids(&r).len(), 25, "a cross join of 5 rows with 5 rows");
        // `AnalyzePlan` resolves it too, and it is not a cache entry.
        let AResult::Schema(s) = analyze(
            &run,
            Analyze::Schema(Box::new(areq::Schema {
                plan: Some(root(remote(&id))),
            })),
        )
        .await
        .expect("schema") else {
            panic!("schema");
        };
        assert!(s.schema.is_some());
        assert_eq!(
            storage_level(&run, &remote(&id)).await,
            level(false, false, 1)
        );
        // Removing it ends the id; removing it again is not an error.
        execute(&run, remove(&id)).await.expect("removes");
        execute(&run, remove(&id)).await.expect("removes again");
        let e = execute(&run, root(remote(&id))).await.unwrap_err();
        assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND");
        // Spark's `SessionHolder.getDataFrameOrThrow`: the client's session id.
        assert_eq!(
            e.message(),
            format!(
                "[CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND] The Spark Connect plan is invalid. No DataFrame with id {id} is found in the session {SESSION} SQLSTATE: 56K00"
            )
        );
        let fields = error_info(&e);
        assert_eq!(fields.get("dfId"), Some(&id), "{fields:?}");
        assert_eq!(
            fields.get("sessionId").map(String::as_str),
            Some(SESSION),
            "{fields:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_id_is_not_found_and_removing_it_is_fine() {
    let run = start(None).await;
    let e = execute(&run, root(remote("no-such-id"))).await.unwrap_err();
    assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND");
    let e = analyze(
        &run,
        Analyze::Schema(Box::new(areq::Schema {
            plan: Some(root(remote("no-such-id"))),
        })),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND");
    execute(&run, remove("no-such-id"))
        .await
        .expect("not an error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reliable_checkpoint_needs_a_directory_and_keeps_its_rows() {
    // Without `spark.checkpoint.dir`: Spark's `_LEGACY_ERROR_TEMP_3016`, for a lazy one too.
    let run = start(None).await;
    for eager in [true, false] {
        let e = execute(&run, checkpoint(range(3), false, eager))
            .await
            .unwrap_err();
        assert_eq!(
            error_class(&e),
            "_LEGACY_ERROR_TEMP_3016",
            "eager = {eager}"
        );
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let cp_dir = tmp.path().join("checkpoints");
    let run = start(Some(cp_dir.to_str().expect("utf-8").to_owned())).await;
    for eager in [true, false] {
        let id = relation_id(
            &execute(&run, checkpoint(range(4), false, eager))
                .await
                .expect("checkpoints"),
        );
        let r = execute(&run, root(remote(&id))).await.expect("reads");
        assert_eq!(ids(&r), vec![0, 1, 2, 3], "eager = {eager}");
        assert_eq!(
            spill_files(&cp_dir),
            1,
            "eager = {eager}: the rows are on disk"
        );
        // The id ends with the frame; the files go with the last frame that reads them.
        execute(&run, remove(&id)).await.expect("removes");
        assert_eq!(
            spill_files(&cp_dir),
            0,
            "eager = {eager}: the files go with the frame"
        );
    }
}

/// The checkpoint files (`.spill`) under `dir`.
fn spill_files(dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                spill_files(&p)
            } else {
                usize::from(p.extension().is_some_and(|x| x == "spill"))
            }
        })
        .sum()
}

/// Pinned: analyzing a plan that reads a lazy checkpoint takes its rows (the schema is the
/// snapshot's), over Connect as embedded (`DataFrame::schema`), where Spark's lazy checkpoint runs
/// no job on analysis. Recorded in the coverage row and the connect-server CLAUDE.md.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyzing_a_lazy_checkpoint_takes_its_rows() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cp_dir = tmp.path().join("checkpoints");
    let run = start(Some(cp_dir.to_str().expect("utf-8").to_owned())).await;
    let id = relation_id(
        &execute(&run, checkpoint(range(4), false, false))
            .await
            .expect("checkpoints"),
    );
    assert_eq!(spill_files(&cp_dir), 0, "lazy: nothing is taken yet");
    analyze(
        &run,
        Analyze::Schema(Box::new(areq::Schema {
            plan: Some(root(remote(&id))),
        })),
    )
    .await
    .expect("schema");
    assert_eq!(spill_files(&cp_dir), 1, "the analysis took the rows");
}

/// GraphFrames' loop over Connect: each round is a lazy local checkpoint of the previous round's
/// held frame, and the client removes the one before last (PySpark does that when a `DataFrame`
/// is collected). Fifty rounds finish at once, the rows stay right, the latest two ids still read
/// and every removed id is `DATAFRAME_NOT_FOUND`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fifty_lazy_checkpoints_leave_only_the_frames_the_client_still_holds() {
    let run = start(None).await;
    let mut ids_held: Vec<String> = Vec::new();
    let mut input = range(5);
    for _ in 0..50 {
        let answered = execute(&run, checkpoint(limit(input, 10), true, false))
            .await
            .expect("checkpoints");
        let id = relation_id(&answered);
        input = remote(&id);
        ids_held.push(id);
        if ids_held.len() > 2 {
            let old = ids_held.remove(0);
            execute(&run, remove(&old)).await.expect("removes");
            let e = execute(&run, root(remote(&old))).await.unwrap_err();
            assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND");
        }
    }
    assert_eq!(ids_held.len(), 2);
    for id in &ids_held {
        let r = execute(&run, root(remote(id))).await.expect("reads");
        assert_eq!(ids(&r), (0..5).collect::<Vec<_>>());
    }
}
