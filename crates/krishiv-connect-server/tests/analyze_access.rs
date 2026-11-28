// SPDX-License-Identifier: Apache-2.0
//! `AnalyzePlan` under the ADR 0046 role matrix (§3b, P3 PR-14c), end to end over a raw tonic client:
//! a viewer (`RolePolicy`: `Browse`, no `Select`) gets the schema, tree string, flags and semantic
//! hash of a table it may not read, and its `explain` with the statistics withheld (P3 PR-14d);
//! `ExecutePlan` of the same plan, `input_files` and `explain` in `cost` mode stay
//! `PERMISSION_DENIED`. An operator gets everything, its explain unchanged. A read by path keeps `Select` even for
//! `AnalyzePlan` (its schema is inferred from file contents): a viewer's schema of it is denied.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::analyze_plan_request::{self as areq, Analyze};
use krishiv_connect::proto::v4_3::analyze_plan_response::Result as AResult;
use krishiv_connect::proto::v4_3::plan::OpType as PlanOp;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::Session;
use krishiv_session::auth::{Authenticated, Role, RolePolicy};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";

struct Running {
    channel: Channel,
    _stop: oneshot::Sender<()>,
}

/// A server whose sessions run as `role`, each with the table `t` (column `a`, three rows).
async fn start(role: Role) -> Running {
    let factory = move |_: &Authenticated,
                        _: krishiv_session::Transport,
                        _: Option<krishiv_plan::analyzer::SparkProfile>| {
        let policy = RolePolicy::new(Some(role), Role::Admin, PolicyVersion(1));
        let s = Session::builder().access_policy(Arc::new(policy)).build()?;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .expect("batch");
        s.register_batches("t", schema, vec![batch])?;
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

/// The table `t`, as a DataFrame plan (`spark.table("t")`).
fn table() -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Read(pb::Read {
                is_streaming: false,
                read_type: Some(pb::read::ReadType::NamedTable(pb::read::NamedTable {
                    unparsed_identifier: "t".to_owned(),
                    ..Default::default()
                })),
            })),
        }))),
    }
}

/// `spark.read.parquet(dir)`, as a DataFrame plan.
fn by_path(dir: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Read(pb::Read {
                is_streaming: false,
                read_type: Some(pb::read::ReadType::DataSource(pb::read::DataSource {
                    format: Some("parquet".to_owned()),
                    paths: vec![dir.to_owned()],
                    ..Default::default()
                })),
            })),
        }))),
    }
}

/// A directory of parquet files of `t` (column `a`), written by an unrestricted session.
async fn parquet_dir() -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("data").to_str().expect("utf-8").to_owned();
    let s = Session::new().expect("session");
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .expect("batch");
    s.register_batches("t", schema, vec![batch])
        .expect("register");
    s.table("t").write().parquet(&dir).await.expect("write");
    (tmp, dir)
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

/// Runs `plan`; the number of Arrow batches, or the first error.
async fn execute(run: &Running, plan: pb::Plan) -> Result<usize, tonic::Status> {
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
    let mut batches = 0;
    while let Some(m) = live.message().await? {
        if matches!(
            m.response_type,
            Some(pb::execute_plan_response::ResponseType::ArrowBatch(_))
        ) {
            batches += 1;
        }
    }
    Ok(batches)
}

fn schema_of(plan: pb::Plan) -> Analyze {
    Analyze::Schema(Box::new(areq::Schema { plan: Some(plan) }))
}

fn denied<T: std::fmt::Debug>(r: &Result<T, tonic::Status>, what: &str) {
    let code = r.as_ref().map_err(tonic::Status::code);
    assert_eq!(code.err(), Some(Code::PermissionDenied), "{what}: {r:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_viewer_analyzes_a_table_it_may_not_read() {
    let run = start(Role::Viewer).await;
    let AResult::Schema(s) = analyze(&run, schema_of(table())).await.expect("schema") else {
        panic!("schema");
    };
    assert!(s.schema.is_some());
    let AResult::TreeString(t) = analyze(
        &run,
        Analyze::TreeString(Box::new(areq::TreeString {
            plan: Some(table()),
            level: None,
        })),
    )
    .await
    .expect("tree string") else {
        panic!("tree string");
    };
    assert!(t.tree_string.contains("a: integer"), "{}", t.tree_string);
    for (what, a) in [
        (
            "is_local",
            Analyze::IsLocal(Box::new(areq::IsLocal {
                plan: Some(table()),
            })),
        ),
        (
            "is_streaming",
            Analyze::IsStreaming(Box::new(areq::IsStreaming {
                plan: Some(table()),
            })),
        ),
        (
            "semantic_hash",
            Analyze::SemanticHash(Box::new(areq::SemanticHash {
                plan: Some(table()),
            })),
        ),
        (
            "same_semantics",
            Analyze::SameSemantics(Box::new(areq::SameSemantics {
                target_plan: Some(table()),
                other_plan: Some(table()),
            })),
        ),
    ] {
        analyze(&run, a)
            .await
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
    }

    // Reading it is not metadata, before or after the analyses.
    denied(&execute(&run, table()).await, "ExecutePlan");
    denied(
        &analyze(
            &run,
            Analyze::InputFiles(Box::new(areq::InputFiles {
                plan: Some(table()),
            })),
        )
        .await,
        "input_files",
    );
    // `explain` (P3 PR-14d): the plan with the data withheld; `cost` is the data.
    let explained = analyze(&run, explain_of(areq::explain::ExplainMode::Simple))
        .await
        .expect("explain");
    let AResult::Explain(e) = explained else {
        panic!("an explain: {explained:?}")
    };
    assert!(
        e.explain_string.contains("Statistics withheld")
            && !e.explain_string.contains("Statistics("),
        "{}",
        e.explain_string
    );
    denied(
        &analyze(&run, explain_of(areq::explain::ExplainMode::Cost)).await,
        "explain cost",
    );
    // And the schema still answers after the denials.
    analyze(&run, schema_of(table()))
        .await
        .expect("schema again");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_operator_runs_what_it_analyzed() {
    let run = start(Role::Operator).await;
    analyze(&run, schema_of(table())).await.expect("schema");
    assert!(execute(&run, table()).await.expect("execute") > 0);
    for mode in [
        areq::explain::ExplainMode::Simple,
        areq::explain::ExplainMode::Cost,
    ] {
        let explained = analyze(&run, explain_of(mode)).await.expect("explain");
        let AResult::Explain(e) = explained else {
            panic!("an explain: {explained:?}")
        };
        assert!(
            !e.explain_string.contains("Statistics withheld"),
            "{}",
            e.explain_string
        );
    }
}

fn explain_of(mode: areq::explain::ExplainMode) -> Analyze {
    Analyze::Explain(Box::new(areq::Explain {
        plan: Some(table()),
        explain_mode: mode.into(),
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_read_by_path_needs_select_for_analyze_plan_too() {
    let (_tmp, dir) = parquet_dir().await;
    let viewer = start(Role::Viewer).await;
    denied(&analyze(&viewer, schema_of(by_path(&dir))).await, "schema");
    denied(
        &analyze(
            &viewer,
            Analyze::InputFiles(Box::new(areq::InputFiles {
                plan: Some(by_path(&dir)),
            })),
        )
        .await,
        "input_files",
    );
    let operator = start(Role::Operator).await;
    let AResult::Schema(s) = analyze(&operator, schema_of(by_path(&dir)))
        .await
        .expect("schema")
    else {
        panic!("schema");
    };
    assert!(s.schema.is_some());
    let AResult::InputFiles(f) = analyze(
        &operator,
        Analyze::InputFiles(Box::new(areq::InputFiles {
            plan: Some(by_path(&dir)),
        })),
    )
    .await
    .expect("input_files") else {
        panic!("input_files");
    };
    assert!(!f.files.is_empty(), "{:?}", f.files);
    assert!(
        f.files.iter().all(|file| file.contains("data")),
        "{:?}",
        f.files
    );
}
