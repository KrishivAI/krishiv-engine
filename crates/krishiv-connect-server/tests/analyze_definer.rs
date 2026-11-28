// SPDX-License-Identifier: Apache-2.0
//! `AnalyzePlan` over a `SQL SECURITY DEFINER` view (P3-T062 review r3 X2, ADR 0048 A1 §3): the
//! arms that hash or list the analyzed plan (`semantic_hash`, `same_semantics`, `input_files`)
//! ask the reader, as `EXPLAIN` does, for what the definer's view reads before they answer, so
//! a reader holding `Select` on the view only cannot confirm a guessed definition or learn the
//! definer's files; the arms that show no plan (`schema`, `is_local`, `is_streaming`) still
//! answer it. Once the reader may read the base table, every arm answers.
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::analyze_plan_request::{self as areq, Analyze};
use krishiv_connect::proto::v4_3::analyze_plan_response::Result as AResult;
use krishiv_connect::proto::v4_3::plan::OpType as PlanOp;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::transport::Channel;

const SESSION: &str = "4b1d2c55-5ad0-4d1e-9a3c-3a1b0b2c7d11";

struct World {
    rt: Arc<EngineRuntime>,
    grants: Arc<Grants>,
    _dir: tempfile::TempDir,
}

fn session(w: &World, name: &str, admin: bool) -> Session {
    let policy = Arc::new(GrantsPolicy::new(
        Arc::clone(&w.grants),
        Arc::new(AllowAll),
        admin,
    ));
    Session::builder()
        .runtime(Arc::clone(&w.rt))
        .principal(Principal::new(name, PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .grants(policy)
        .build()
        .expect("session")
}

async fn run(s: &Session, sql: &str) {
    let df = s.sql(sql).await.unwrap_or_else(|e| panic!("{sql}: {e}"));
    df.collect().await.unwrap_or_else(|e| panic!("{sql}: {e}"));
}

struct Running {
    channel: Channel,
    root: Session,
    _world: Arc<World>,
    _stop: oneshot::Sender<()>,
}

/// root's `t (a, b)`, ana's definer view `dv` over it granted to bob, and a server whose
/// sessions run as bob.
async fn start() -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let rt = EngineRuntime::builder()
        .config(
            "spark.sql.warehouse.dir",
            dir.path().join("wh").to_str().expect("utf-8"),
        )
        .build()
        .expect("runtime");
    let world = Arc::new(World {
        rt,
        grants: Grants::new(Arc::new(MemoryGrantStore::new())),
        _dir: dir,
    });
    let (root, ana) = (session(&world, "root", true), session(&world, "ana", false));
    for sql in [
        "CREATE TABLE t (a INT, b INT) USING parquet",
        "INSERT INTO t VALUES (1, 2)",
        "GRANT CREATE ON SCHEMA default TO user:ana",
        "GRANT SELECT ON TABLE t TO user:ana",
    ] {
        run(&root, sql).await;
    }
    for sql in [
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t WHERE b = 2 AND a < 424242",
        "GRANT SELECT ON VIEW dv TO user:bob",
    ] {
        run(&ana, sql).await;
    }
    let shared = Arc::clone(&world);
    let factory = move |_: &krishiv_session::auth::Authenticated,
                        _: krishiv_session::Transport,
                        _: Option<krishiv_plan::analyzer::SparkProfile>| {
        Ok(session(&shared, "bob", false))
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
        root,
        _world: world,
        _stop: stop,
    }
}

/// `spark.table(name)`, as a DataFrame plan.
fn table(name: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Read(pb::Read {
                is_streaming: false,
                read_type: Some(pb::read::ReadType::NamedTable(pb::read::NamedTable {
                    unparsed_identifier: name.to_owned(),
                    ..Default::default()
                })),
            })),
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
            "/spark.connect.SparkConnectService/AnalyzePlan"
                .parse()
                .expect("path"),
            tonic_prost::ProstCodec::<pb::AnalyzePlanRequest, pb::AnalyzePlanResponse>::default(),
        )
        .await?
        .into_inner();
    Ok(response.result.expect("a result"))
}

fn semantic_hash() -> Analyze {
    Analyze::SemanticHash(Box::new(areq::SemanticHash {
        plan: Some(table("dv")),
    }))
}

fn same_semantics() -> Analyze {
    Analyze::SameSemantics(Box::new(areq::SameSemantics {
        target_plan: Some(table("dv")),
        other_plan: Some(table("dv")),
    }))
}

fn input_files() -> Analyze {
    Analyze::InputFiles(Box::new(areq::InputFiles {
        plan: Some(table("dv")),
    }))
}

fn plan_free() -> Vec<(&'static str, Analyze)> {
    vec![
        (
            "schema",
            Analyze::Schema(Box::new(areq::Schema {
                plan: Some(table("dv")),
            })),
        ),
        (
            "is_local",
            Analyze::IsLocal(Box::new(areq::IsLocal {
                plan: Some(table("dv")),
            })),
        ),
        (
            "is_streaming",
            Analyze::IsStreaming(Box::new(areq::IsStreaming {
                plan: Some(table("dv")),
            })),
        ),
    ]
}

/// A refusal as the reader sees it: `PERMISSION_DENIED`, nothing of the definer's text.
fn refused(r: &Result<AResult, tonic::Status>, what: &str) {
    let status = r.as_ref().expect_err(what);
    assert_eq!(status.code(), Code::PermissionDenied, "{what}: {status:?}");
    for hidden in ["424242", "`t`", "`b`", "`ana`"] {
        assert!(
            !status.message().contains(hidden),
            "{what}: {hidden}: {status:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn semantic_hash_asks_the_definer_views_reads() {
    let run = start().await;
    refused(&analyze(&run, semantic_hash()).await, "semantic_hash");
    run_grant(&run).await;
    let r = analyze(&run, semantic_hash()).await.expect("semantic_hash");
    assert!(matches!(r, AResult::SemanticHash(_)), "{r:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_semantics_asks_the_definer_views_reads() {
    let run = start().await;
    refused(&analyze(&run, same_semantics()).await, "same_semantics");
    run_grant(&run).await;
    let AResult::SameSemantics(s) = analyze(&run, same_semantics())
        .await
        .expect("same_semantics")
    else {
        panic!("same_semantics");
    };
    assert!(s.result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn input_files_asks_the_definer_views_reads() {
    let run = start().await;
    refused(&analyze(&run, input_files()).await, "input_files");
    run_grant(&run).await;
    let r = analyze(&run, input_files()).await.expect("input_files");
    assert!(matches!(r, AResult::InputFiles(_)), "{r:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_arms_that_show_no_plan_still_answer_the_reader() {
    let run = start().await;
    for (what, arm) in plan_free() {
        analyze(&run, arm)
            .await
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
    }
}

async fn run_grant(r: &Running) {
    run(&r.root, "GRANT SELECT ON TABLE t TO user:bob").await;
}
