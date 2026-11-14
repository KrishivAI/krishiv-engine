// SPDX-License-Identifier: Apache-2.0
//! `AnalyzePlan`, `Config` and `Interrupt` end to end over a raw tonic
//! client (design P3-T011a §5, PR-11b).
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
use std::time::{Duration, Instant};

use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::analyze_plan_request::{self as areq, Analyze};
use krishiv_connect::proto::v4_3::analyze_plan_response::Result as AResult;
use krishiv_connect::proto::v4_3::config_request::operation::OpType;
use krishiv_connect::proto::v4_3::config_request::{self as creq, Operation};
use krishiv_connect::proto::v4_3::interrupt_request::{Interrupt, InterruptType};
use krishiv_connect::proto::v4_3::plan::OpType as PlanOp;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{
    ConnectServer, ConnectServerConfig, SessionEntry, SessionLimits, SessionManager,
};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const OTHER_SESSION: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";
const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";
const OP2: &str = "00112233-4455-6677-8899-aabbccddeef0";

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

async fn stream(
    channel: &Channel,
    request: pb::ExecutePlanRequest,
) -> tonic::Streaming<pb::ExecutePlanResponse> {
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.server_streaming(
        tonic::Request::new(request),
        path("ExecutePlan"),
        tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
    )
    .await
    .expect("execute")
    .into_inner()
}

fn relation(rel: RelType) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(rel),
    }
}

fn sql(query: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(relation(RelType::Sql(pb::Sql {
            query: query.to_owned(),
            ..Default::default()
        }))))),
    }
}

fn range(end: i64) -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(relation(RelType::Range(
            pb::Range {
                start: Some(0),
                end,
                step: 1,
                num_partitions: Some(1),
            },
        ))))),
    }
}

/// A query that runs until it is stopped.
fn endless() -> pb::Plan {
    range(100_000_000_000)
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

fn info(status: &tonic::Status) -> ErrorInfo {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    ErrorInfo::decode(rpc.details[0].value.as_slice()).expect("ErrorInfo")
}

fn error_class(status: &tonic::Status) -> String {
    info(status)
        .metadata
        .get("errorClass")
        .cloned()
        .unwrap_or_default()
}

// ---------------------------------------------------------------- Analyze

async fn analyze(
    run: &Running,
    analyze: Analyze,
) -> Result<pb::AnalyzePlanResponse, tonic::Status> {
    unary(
        &run.channel,
        "AnalyzePlan",
        pb::AnalyzePlanRequest {
            session_id: SESSION.to_owned(),
            analyze: Some(analyze),
            ..Default::default()
        },
    )
    .await
}

async fn analyzed(run: &Running, a: Analyze) -> AResult {
    let r = analyze(run, a).await.expect("analyze");
    assert_eq!(r.session_id, SESSION);
    assert_eq!(r.server_side_session_id, run.entry().server_side_id());
    r.result.expect("a result")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyze_schema_matches_result() {
    let run = start().await;
    let plan = sql("SELECT 1 AS a, 'x' AS b, CAST(NULL AS DECIMAL(10, 2)) AS c");
    let AResult::Schema(s) = analyzed(
        &run,
        Analyze::Schema(Box::new(areq::Schema {
            plan: Some(plan.clone()),
        })),
    )
    .await
    else {
        panic!("schema");
    };
    let mut live = stream(
        &run.channel,
        pb::ExecutePlanRequest {
            session_id: SESSION.to_owned(),
            plan: Some(plan),
            ..Default::default()
        },
    )
    .await;
    let first = live.message().await.expect("ok").expect("schema");
    assert_eq!(s.schema, first.schema);
    assert!(s.schema.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyze_explain_tree_string_and_flags() {
    let run = start().await;
    let plan = sql("SELECT id, named_struct('x', id, 'y', array(id)) AS s FROM range(3)");
    let AResult::Explain(e) = analyzed(
        &run,
        Analyze::Explain(Box::new(areq::Explain {
            plan: Some(plan.clone()),
            explain_mode: areq::explain::ExplainMode::Extended as i32,
        })),
    )
    .await
    else {
        panic!("explain");
    };
    assert!(
        e.explain_string.starts_with("== Parsed Logical Plan =="),
        "{}",
        e.explain_string
    );
    let status = analyze(
        &run,
        Analyze::Explain(Box::new(areq::Explain {
            plan: Some(plan.clone()),
            explain_mode: 0,
        })),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "CONNECT_INVALID_PLAN.INVALID_ENUM");

    let AResult::TreeString(t) = analyzed(
        &run,
        Analyze::TreeString(Box::new(areq::TreeString {
            plan: Some(plan.clone()),
            level: None,
        })),
    )
    .await
    else {
        panic!("tree string");
    };
    assert_eq!(
        t.tree_string,
        "root\n\
         \x20|-- id: long (nullable = false)\n\
         \x20|-- s: struct (nullable = false)\n\
         \x20|    |-- x: long (nullable = false)\n\
         \x20|    |-- y: array (nullable = false)\n\
         \x20|    |    |-- element: long (containsNull = false)\n"
    );
    let AResult::TreeString(t) = analyzed(
        &run,
        Analyze::TreeString(Box::new(areq::TreeString {
            plan: Some(plan.clone()),
            level: Some(1),
        })),
    )
    .await
    else {
        panic!("tree string");
    };
    assert_eq!(
        t.tree_string,
        "root\n |-- id: long (nullable = false)\n |-- s: struct (nullable = false)\n"
    );

    let AResult::IsLocal(l) = analyzed(
        &run,
        Analyze::IsLocal(Box::new(areq::IsLocal {
            plan: Some(plan.clone()),
        })),
    )
    .await
    else {
        panic!("is local");
    };
    assert!(!l.is_local);
    // Codex review (#738): SQL text is parsed during analysis, so a SQL
    // `VALUES` is not an inline table when `is_local` asks (Spark's answer
    // for it is to be checked against 4.2/4.3 before this changes).
    for query in [
        "VALUES (1), (2)",
        "SELECT * FROM VALUES (1), (2)",
        "SELECT 1",
    ] {
        let AResult::IsLocal(l) = analyzed(
            &run,
            Analyze::IsLocal(Box::new(areq::IsLocal {
                plan: Some(sql(query)),
            })),
        )
        .await
        else {
            panic!("is local");
        };
        assert!(!l.is_local, "{query}");
    }
    let AResult::IsStreaming(st) = analyzed(
        &run,
        Analyze::IsStreaming(Box::new(areq::IsStreaming { plan: Some(plan) })),
    )
    .await
    else {
        panic!("is streaming");
    };
    assert!(!st.is_streaming);
    let AResult::SparkVersion(v) =
        analyzed(&run, Analyze::SparkVersion(Box::new(areq::SparkVersion {}))).await
    else {
        panic!("version");
    };
    assert_eq!(v.version, "4.2.0");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyze_input_files_lists_file_scans() {
    let run = start().await;
    let dir = tempfile::tempdir().expect("dir");
    let out = dir.path().join("t");
    let out = out.to_str().expect("utf8").to_owned();
    let session = run.entry().session().clone();
    session
        .sql("SELECT 1 AS a")
        .await
        .expect("df")
        .write()
        .parquet(&out)
        .await
        .expect("write");
    let AResult::InputFiles(f) = analyzed(
        &run,
        Analyze::InputFiles(Box::new(areq::InputFiles {
            plan: Some(sql(&format!("SELECT * FROM parquet.`{out}`"))),
        })),
    )
    .await
    else {
        panic!("input files");
    };
    // The path qualified as the scan lists it, not the files under it (a
    // recorded difference).
    assert_eq!(f.files, vec![format!("file:{out}")]);
    let AResult::InputFiles(f) = analyzed(
        &run,
        Analyze::InputFiles(Box::new(areq::InputFiles {
            plan: Some(range(3)),
        })),
    )
    .await
    else {
        panic!("input files");
    };
    assert!(f.files.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyze_types_ddl_and_json() {
    let run = start().await;
    let AResult::DdlParse(d) = analyzed(
        &run,
        Analyze::DdlParse(Box::new(areq::DdlParse {
            ddl_string: "a INT, b STRING".to_owned(),
        })),
    )
    .await
    else {
        panic!("ddl parse");
    };
    let Some(pb::data_type::Kind::Struct(s)) = d.parsed.and_then(|t| t.kind) else {
        panic!("a struct");
    };
    assert_eq!(s.fields.len(), 2);
    assert_eq!(s.fields[1].name, "b");
    let status = analyze(
        &run,
        Analyze::DdlParse(Box::new(areq::DdlParse {
            ddl_string: "a nonsense_type".to_owned(),
        })),
    )
    .await
    .unwrap_err();
    assert!(!error_class(&status).is_empty());

    let json = r#"{"type":"struct","fields":[{"name":"a b","type":"long","nullable":false,"metadata":{"comment":"c"}}]}"#;
    let AResult::JsonToDdl(j) = analyzed(
        &run,
        Analyze::JsonToDdl(Box::new(areq::JsonToDdl {
            json_string: json.to_owned(),
        })),
    )
    .await
    else {
        panic!("json to ddl");
    };
    assert_eq!(j.ddl_string, "`a b` BIGINT NOT NULL COMMENT 'c'");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyze_same_semantics_and_semantic_hash() {
    let run = start().await;
    let same = |a: pb::Plan, b: pb::Plan| {
        Analyze::SameSemantics(Box::new(areq::SameSemantics {
            target_plan: Some(a),
            other_plan: Some(b),
        }))
    };
    let hash = |p: pb::Plan| Analyze::SemanticHash(Box::new(areq::SemanticHash { plan: Some(p) }));
    let q = "SELECT id + 1 AS x FROM range(10) WHERE id > 2";
    let AResult::SameSemantics(r) = analyzed(&run, same(sql(q), sql(q))).await else {
        panic!("same");
    };
    assert!(r.result, "a plan has the same semantics as itself");
    let AResult::SameSemantics(r) = analyzed(&run, same(sql(q), sql("SELECT 1"))).await else {
        panic!("same");
    };
    assert!(!r.result);
    let AResult::SemanticHash(h1) = analyzed(&run, hash(sql(q))).await else {
        panic!("hash");
    };
    let AResult::SemanticHash(h2) = analyzed(&run, hash(sql(q))).await else {
        panic!("hash");
    };
    let AResult::SemanticHash(h3) = analyzed(&run, hash(sql("SELECT 1"))).await else {
        panic!("hash");
    };
    assert_eq!(h1.result, h2.result);
    assert_ne!(h1.result, h3.result);
    // Spark's canonicalization erases alias and attribute names and drops
    // `SubqueryAlias` (`QueryPlan.scala:708-729`,
    // `namedExpressions.scala:326-327`, `basicLogicalOperators.scala:2031`).
    for (a, b) in [
        (
            "SELECT id AS x FROM range(3)",
            "SELECT id AS y FROM range(3)",
        ),
        (
            "SELECT t.id + 1 AS v FROM range(3) t",
            "SELECT u.id + 1 AS w FROM range(3) u",
        ),
    ] {
        let AResult::SameSemantics(r) = analyzed(&run, same(sql(a), sql(b))).await else {
            panic!("same");
        };
        assert!(r.result, "{a} / {b}");
        let AResult::SemanticHash(ha) = analyzed(&run, hash(sql(a))).await else {
            panic!("hash");
        };
        let AResult::SemanticHash(hb) = analyzed(&run, hash(sql(b))).await else {
            panic!("hash");
        };
        assert_eq!(ha.result, hb.result, "{a} / {b}");
    }
    // A different expression is still different.
    let AResult::SameSemantics(r) = analyzed(
        &run,
        same(
            sql("SELECT id + 1 AS x FROM range(3)"),
            sql("SELECT id + 2 AS x FROM range(3)"),
        ),
    )
    .await
    else {
        panic!("same");
    };
    assert!(!r.result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyze_refusals() {
    let run = start().await;
    let status = unary::<_, pb::AnalyzePlanResponse>(
        &run.channel,
        "AnalyzePlan",
        pb::AnalyzePlanRequest {
            session_id: SESSION.to_owned(),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert_eq!(status.code(), tonic::Code::Internal);
    assert_eq!(
        error_class(&status),
        "CONNECT_INVALID_PLAN.UNKNOWN_ANALYZE_METHOD"
    );
    assert_eq!(info(&status).metadata["sqlState"], "56K00");
    let rel = Some(relation(RelType::Range(pb::Range {
        start: Some(0),
        end: 1,
        step: 1,
        num_partitions: None,
    })));
    for a in [
        Analyze::Persist(Box::new(areq::Persist {
            relation: rel.clone(),
            storage_level: None,
        })),
        Analyze::Unpersist(Box::new(areq::Unpersist {
            relation: rel.clone(),
            blocking: None,
        })),
        Analyze::GetStorageLevel(Box::new(areq::GetStorageLevel {
            relation: rel.clone(),
        })),
    ] {
        let status = analyze(&run, a).await.unwrap_err();
        assert_eq!(
            error_class(&status),
            "CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED"
        );
    }
    // An analysis error is the analyzer's.
    let status = analyze(
        &run,
        Analyze::Schema(Box::new(areq::Schema {
            plan: Some(sql("SELECT nope FROM range(1)")),
        })),
    )
    .await
    .unwrap_err();
    assert!(
        error_class(&status).starts_with("UNRESOLVED_COLUMN"),
        "{status:?}"
    );
}

// ----------------------------------------------------------------- Config

async fn config(run: &Running, op: OpType) -> Result<pb::ConfigResponse, tonic::Status> {
    unary(
        &run.channel,
        "Config",
        pb::ConfigRequest {
            session_id: SESSION.to_owned(),
            operation: Some(Operation { op_type: Some(op) }),
            ..Default::default()
        },
    )
    .await
}

fn kv(key: &str, value: Option<&str>) -> pb::KeyValue {
    pb::KeyValue {
        key: key.to_owned(),
        value: value.map(str::to_owned),
    }
}

fn keys(k: &[&str]) -> Vec<String> {
    k.iter().map(|s| (*s).to_owned()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_operations() {
    let run = start().await;
    let set = config(
        &run,
        OpType::Set(creq::Set {
            pairs: vec![
                kv("spark.sql.ansi.enabled", Some("false")),
                kv("my.app.flag", Some("on")),
                kv("spark.sql.execution.arrow.enabled", Some("true")),
            ],
            silent: None,
        }),
    )
    .await
    .expect("set");
    assert_eq!(set.session_id, SESSION);
    assert_eq!(set.server_side_session_id, run.entry().server_side_id());
    assert_eq!(
        set.warnings,
        vec![
            "The SQL config 'spark.sql.execution.arrow.enabled' is NOT supported in Spark Connect"
                .to_owned()
        ]
    );
    let get = config(
        &run,
        OpType::Get(creq::Get {
            keys: keys(&["spark.sql.ansi.enabled", "my.app.flag"]),
        }),
    )
    .await
    .expect("get");
    assert_eq!(
        get.pairs,
        vec![
            kv("spark.sql.ansi.enabled", Some("false")),
            kv("my.app.flag", Some("on"))
        ]
    );
    // An unset key without a default.
    let status = config(
        &run,
        OpType::Get(creq::Get {
            keys: keys(&["no.such.key"]),
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "SQL_CONF_NOT_FOUND");
    assert_eq!(info(&status).metadata["sqlState"], "42K0I");
    assert_eq!(
        info(&status).reason,
        "org.apache.spark.SparkNoSuchElementException"
    );
    // get_with_default: a set value, else the caller's default (not the
    // registered default).
    let gwd = config(
        &run,
        OpType::GetWithDefault(creq::GetWithDefault {
            pairs: vec![
                kv("my.app.flag", Some("d")),
                kv("no.such.key", Some("d")),
                kv("no.such.key", None),
                kv("spark.sql.caseSensitive", Some("true")),
            ],
        }),
    )
    .await
    .expect("get with default");
    assert_eq!(
        gwd.pairs,
        vec![
            kv("my.app.flag", Some("on")),
            kv("no.such.key", Some("d")),
            kv("no.such.key", None),
            kv("spark.sql.caseSensitive", Some("true")),
        ]
    );
    let opt = config(
        &run,
        OpType::GetOption(creq::GetOption {
            keys: keys(&["my.app.flag", "no.such.key", "spark.sql.ansi.enabled"]),
        }),
    )
    .await
    .expect("get option");
    assert_eq!(
        opt.pairs,
        vec![
            kv("my.app.flag", Some("on")),
            kv("no.such.key", None),
            kv("spark.sql.ansi.enabled", Some("false")),
        ]
    );
    let all = config(
        &run,
        OpType::GetAll(creq::GetAll {
            prefix: Some("my.app.".to_owned()),
        }),
    )
    .await
    .expect("get all");
    assert_eq!(all.pairs, vec![kv("flag", Some("on"))]);
    let all = config(&run, OpType::GetAll(creq::GetAll { prefix: None }))
        .await
        .expect("get all");
    assert!(
        all.pairs
            .contains(&kv("spark.sql.ansi.enabled", Some("false")))
    );
    assert!(all.pairs.contains(&kv("my.app.flag", Some("on"))));
    config(
        &run,
        OpType::Unset(creq::Unset {
            keys: keys(&["my.app.flag", "spark.sql.ansi.enabled"]),
        }),
    )
    .await
    .expect("unset");
    let opt = config(
        &run,
        OpType::GetOption(creq::GetOption {
            keys: keys(&["my.app.flag", "spark.sql.ansi.enabled"]),
        }),
    )
    .await
    .expect("get option");
    assert_eq!(
        opt.pairs,
        vec![
            kv("my.app.flag", None),
            kv("spark.sql.ansi.enabled", Some("true"))
        ]
    );
    let m = config(
        &run,
        OpType::IsModifiable(creq::IsModifiable {
            keys: keys(&["spark.sql.ansi.enabled", "krishiv.memory.limit"]),
        }),
    )
    .await
    .expect("is modifiable");
    assert_eq!(
        m.pairs,
        vec![
            kv("spark.sql.ansi.enabled", Some("true")),
            kv("krishiv.memory.limit", Some("false"))
        ]
    );
    let status = unary::<_, pb::ConfigResponse>(
        &run.channel,
        "Config",
        pb::ConfigRequest {
            session_id: SESSION.to_owned(),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert_eq!(
        error_class(&status),
        "CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_set_silent_and_static_keys() {
    let run = start().await;
    let status = config(
        &run,
        OpType::Set(creq::Set {
            pairs: vec![kv("krishiv.resultCache.maxBytes", Some("1g"))],
            silent: None,
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "CANNOT_MODIFY_STATIC_CONFIG");
    assert_eq!(info(&status).metadata["sqlState"], "46110");
    let status = config(
        &run,
        OpType::Unset(creq::Unset {
            keys: keys(&["krishiv.resultCache.maxBytes"]),
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "CANNOT_MODIFY_STATIC_CONFIG");
    let r = config(
        &run,
        OpType::Set(creq::Set {
            pairs: vec![
                kv("krishiv.resultCache.maxBytes", Some("1g")),
                kv("my.null", None),
                kv("my.app.flag", Some("ok")),
            ],
            silent: Some(true),
        }),
    )
    .await
    .expect("silent set");
    assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
    assert!(
        r.warnings[0].starts_with(
            "Failed to set krishiv.resultCache.maxBytes to Some(1g) due to \
             [CANNOT_MODIFY_STATIC_CONFIG]"
        ),
        "{}",
        r.warnings[0]
    );
    assert!(
        r.warnings[1].starts_with("Failed to set my.null to None due to "),
        "{}",
        r.warnings[1]
    );
    assert_eq!(
        run.entry().session().conf().get("my.app.flag").as_deref(),
        Some("ok")
    );
}

/// T-C8: a planted secret never comes back from `Config`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_redaction() {
    const PLANTED: &str = "hunter2-planted-value";
    let run = start().await;
    let conf = run.entry().session().conf();
    conf.set("spark.my.secret.value", PLANTED).expect("set");
    conf.set("my.url", "jdbc:pg://h/db?password=pw")
        .expect("set");
    conf.set("my.plain", "visible").expect("set");
    // Keys and values Krishiv's classifier marks (ADR 0038), beyond Spark's
    // pattern.
    conf.set("fs.azure.account.key.acct.dfs.core.windows.net", PLANTED)
        .expect("set");
    conf.set("kafka.sasl.jaas.config", PLANTED).expect("set");
    conf.set("my.db", "postgres://u:pw-planted@h/db")
        .expect("set");
    for key in [
        "fs.azure.account.key.acct.dfs.core.windows.net",
        "kafka.sasl.jaas.config",
        "my.db",
    ] {
        let status = config(&run, OpType::Get(creq::Get { keys: keys(&[key]) }))
            .await
            .unwrap_err();
        assert_eq!(error_class(&status), "SQL_CONF_NOT_FOUND", "{key}");
        let r = config(
            &run,
            OpType::GetOption(creq::GetOption { keys: keys(&[key]) }),
        )
        .await
        .expect("get option");
        assert_eq!(r.pairs, vec![kv(key, None)]);
        let r = config(
            &run,
            OpType::GetWithDefault(creq::GetWithDefault {
                pairs: vec![kv(key, Some("dflt"))],
            }),
        )
        .await
        .expect("get with default");
        assert_eq!(r.pairs, vec![kv(key, Some("dflt"))]);
    }
    let status = config(
        &run,
        OpType::Get(creq::Get {
            keys: keys(&["spark.my.secret.value"]),
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "SQL_CONF_NOT_FOUND");
    let gwd = config(
        &run,
        OpType::GetWithDefault(creq::GetWithDefault {
            pairs: vec![
                kv("spark.my.secret.value", Some("dflt")),
                kv("my.url", None),
            ],
        }),
    )
    .await
    .expect("get with default");
    assert_eq!(
        gwd.pairs,
        vec![
            kv("spark.my.secret.value", Some("dflt")),
            kv("my.url", None)
        ]
    );
    let opt = config(
        &run,
        OpType::GetOption(creq::GetOption {
            keys: keys(&["spark.my.secret.value", "my.url", "my.plain"]),
        }),
    )
    .await
    .expect("get option");
    assert_eq!(
        opt.pairs,
        vec![
            kv("spark.my.secret.value", None),
            kv("my.url", None),
            kv("my.plain", Some("visible"))
        ]
    );
    // The prefix is stripped after matching, so `value` is not shown.
    for prefix in [None, Some("spark.my.secret.".to_owned())] {
        let all = config(&run, OpType::GetAll(creq::GetAll { prefix }))
            .await
            .expect("get all");
        let text = format!("{:?}", all.pairs);
        assert!(!text.contains(PLANTED), "{text}");
        assert!(!text.contains("pw"), "{text}");
        assert!(!text.contains("pw-planted"), "{text}");
        assert!(!text.contains("account.key"), "{text}");
        assert!(!text.contains("jaas"), "{text}");
        assert!(!all.pairs.iter().any(|p| p.key == "value"));
    }
}

// -------------------------------------------------------------- Interrupt

async fn interrupt(
    run: &Running,
    session: &str,
    kind: InterruptType,
    which: Option<Interrupt>,
) -> Result<pb::InterruptResponse, tonic::Status> {
    unary(
        &run.channel,
        "Interrupt",
        pb::InterruptRequest {
            session_id: session.to_owned(),
            interrupt_type: kind as i32,
            interrupt: which,
            ..Default::default()
        },
    )
    .await
}

fn execute(id: &str) -> pb::ExecutePlanRequest {
    pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        operation_id: Some(id.to_owned()),
        plan: Some(endless()),
        ..Default::default()
    }
}

/// Reads `live` until it ends with an error.
async fn ends_with(live: &mut tonic::Streaming<pb::ExecutePlanResponse>) -> tonic::Status {
    loop {
        match live.message().await {
            Ok(Some(_)) => {}
            other => return other.expect_err("the stream ends with an error"),
        }
    }
}

async fn eventually(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(tokio::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_by_operation_id_stops_within_a_second_and_frees_memory() {
    let run = start().await;
    let entry = run.entry();
    let session = entry.session().clone();
    let memory = session.runtime().memory();
    let before = memory.reserved();
    let mut live = stream(&run.channel, execute(OP)).await;
    live.message().await.expect("ok").expect("schema");
    eventually("the query runs", || session.running_queries() == 1).await;
    // Unknown and malformed ids interrupt nothing.
    for id in [OP2, "not-a-uuid"] {
        let r = interrupt(
            &run,
            SESSION,
            InterruptType::OperationId,
            Some(Interrupt::OperationId(id.to_owned())),
        )
        .await
        .expect("interrupt");
        assert!(r.interrupted_ids.is_empty());
    }
    // Another session's interrupt does not reach this operation (T-Z5).
    let r = interrupt(
        &run,
        OTHER_SESSION,
        InterruptType::OperationId,
        Some(Interrupt::OperationId(OP.to_owned())),
    )
    .await
    .expect("interrupt");
    assert!(r.interrupted_ids.is_empty());

    let t0 = Instant::now();
    let r = interrupt(
        &run,
        SESSION,
        InterruptType::OperationId,
        Some(Interrupt::OperationId(OP.to_owned())),
    )
    .await
    .expect("interrupt");
    assert_eq!(r.interrupted_ids, vec![OP.to_owned()]);
    assert_eq!(r.server_side_session_id, entry.server_side_id());
    let status = ends_with(&mut live).await;
    assert!(
        t0.elapsed() < Duration::from_secs(1),
        "stopped after {:?}",
        t0.elapsed()
    );
    assert_eq!(error_class(&status), "OPERATION_CANCELED");
    // Interrupting it again interrupts nothing.
    let r = interrupt(
        &run,
        SESSION,
        InterruptType::OperationId,
        Some(Interrupt::OperationId(OP.to_owned())),
    )
    .await
    .expect("interrupt");
    assert!(r.interrupted_ids.is_empty());
    drop(live);
    eventually("the query stops", || session.running_queries() == 0).await;
    eventually("the operation leaves the map", || {
        entry.operations().is_empty()
    })
    .await;
    eventually("the memory returns", || memory.reserved() == before).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_all_lists_the_operations_it_interrupted() {
    let run = start().await;
    let entry = run.entry();
    let mut a = stream(&run.channel, execute(OP)).await;
    let mut b = stream(&run.channel, execute(OP2)).await;
    a.message().await.expect("ok").expect("schema");
    b.message().await.expect("ok").expect("schema");
    let t0 = Instant::now();
    let r = interrupt(&run, SESSION, InterruptType::All, None)
        .await
        .expect("interrupt");
    let mut ids = r.interrupted_ids.clone();
    ids.sort();
    let mut want = vec![OP.to_owned(), OP2.to_owned()];
    want.sort();
    assert_eq!(ids, want);
    assert_eq!(error_class(&ends_with(&mut a).await), "OPERATION_CANCELED");
    assert_eq!(error_class(&ends_with(&mut b).await), "OPERATION_CANCELED");
    assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
    eventually("the operations leave the map", || {
        entry.operations().is_empty()
    })
    .await;
    let r = interrupt(&run, SESSION, InterruptType::All, None)
        .await
        .expect("interrupt");
    assert!(r.interrupted_ids.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_refusals() {
    let run = start().await;
    let status = interrupt(&run, SESSION, InterruptType::OperationId, None)
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&status),
        "INVALID_PARAMETER_VALUE.INTERRUPT_TYPE_OPERATION_ID_REQUIRES_ID"
    );
    assert_eq!(info(&status).metadata["sqlState"], "22023");
    assert_eq!(info(&status).reason, "org.apache.spark.SparkSQLException");
    let status = interrupt(&run, SESSION, InterruptType::Tag, None)
        .await
        .unwrap_err();
    assert_eq!(
        error_class(&status),
        "INVALID_PARAMETER_VALUE.INTERRUPT_TYPE_TAG_REQUIRES_TAG"
    );
    // Missing-id checks look at the oneof case.
    let status = interrupt(
        &run,
        SESSION,
        InterruptType::Tag,
        Some(Interrupt::OperationId(OP.to_owned())),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error_class(&status),
        "INVALID_PARAMETER_VALUE.INTERRUPT_TYPE_TAG_REQUIRES_TAG"
    );
    let status = interrupt(
        &run,
        SESSION,
        InterruptType::Tag,
        Some(Interrupt::OperationTag("t".to_owned())),
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "UNSUPPORTED_FEATURE.INTERRUPT_TYPE");
    assert_eq!(info(&status).metadata["sqlState"], "0A000");
    assert_eq!(
        info(&status).metadata["messageParameters"],
        r#"{"interruptType":"INTERRUPT_TYPE_TAG"}"#
    );
    let status = interrupt(&run, SESSION, InterruptType::Unspecified, None)
        .await
        .unwrap_err();
    assert_eq!(error_class(&status), "UNSUPPORTED_FEATURE.INTERRUPT_TYPE");
}

// ------------------------------------------------- trace context and client type

/// `AnalyzePlan` with extra request metadata, as a client interceptor adds it.
async fn analyze_with_metadata(
    run: &Running,
    metadata: &[(&'static str, &str)],
    client_type: Option<&str>,
) -> Result<pb::AnalyzePlanResponse, tonic::Status> {
    let mut request = tonic::Request::new(pb::AnalyzePlanRequest {
        session_id: SESSION.to_owned(),
        client_type: client_type.map(str::to_owned),
        analyze: Some(Analyze::Schema(Box::new(areq::Schema {
            plan: Some(sql("SELECT 1 AS a")),
        }))),
        ..Default::default()
    });
    for (k, v) in metadata {
        request
            .metadata_mut()
            .insert(*k, v.parse().expect("metadata value"));
    }
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        request,
        path("AnalyzePlan"),
        tonic_prost::ProstCodec::<pb::AnalyzePlanRequest, pb::AnalyzePlanResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

/// `ExecutePlan` of `SELECT 1` under `id` with extra request metadata, read to its end.
async fn execute_with_metadata(run: &Running, id: &str, metadata: &[(&'static str, &str)]) {
    let mut request = tonic::Request::new(pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        operation_id: Some(id.to_owned()),
        plan: Some(sql("SELECT 1 AS a")),
        ..Default::default()
    });
    for (k, v) in metadata {
        request
            .metadata_mut()
            .insert(*k, v.parse().expect("metadata value"));
    }
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let mut live = grpc
        .server_streaming(
            request,
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await
        .expect("execute")
        .into_inner();
    while live.message().await.expect("message").is_some() {}
}

async fn trace_ids(run: &Running) -> BTreeMap<String, Option<String>> {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    let batches = run
        .entry()
        .session()
        .sql("select query_id, trace_id from system.queries where state = 'finished'")
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    let mut out = BTreeMap::new();
    for b in &batches {
        let (ids, traces) = (
            b.column(0).as_string::<i32>(),
            b.column(1).as_string::<i32>(),
        );
        for i in 0..b.num_rows() {
            out.insert(
                ids.value(i).to_owned(),
                (!traces.is_null(i)).then(|| traces.value(i).to_owned()),
            );
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_request_keeps_its_own_traceparent() {
    let run = start().await;
    let other = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-00f067aa0ba902b7-01";
    execute_with_metadata(&run, OP, &[("traceparent", TRACEPARENT)]).await;
    execute_with_metadata(&run, OP2, &[("traceparent", other)]).await;
    execute_with_metadata(&run, "00112233-4455-6677-8899-aabbccddee01", &[]).await;
    execute_with_metadata(
        &run,
        "00112233-4455-6677-8899-aabbccddee02",
        &[("traceparent", "junk")],
    )
    .await;
    let rows = trace_ids(&run).await;
    assert_eq!(
        rows[OP].as_deref(),
        Some("4bf92f3577b34da6a3ce929d0e0e4736"),
        "{rows:?}"
    );
    assert_eq!(
        rows[OP2].as_deref(),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert_eq!(rows["00112233-4455-6677-8899-aabbccddee01"], None);
    assert_eq!(
        rows["00112233-4455-6677-8899-aabbccddee02"], None,
        "a malformed header means no trace, and fails nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_client_type_and_transport_show_in_system_sessions() {
    use arrow_array::cast::AsArray;
    let run = start().await;
    analyze_with_metadata(&run, &[], Some("_SPARK_CONNECT_PYTHON spark/4.3.0"))
        .await
        .expect("analyze");
    let batches = run
        .entry()
        .session()
        .sql("select transport, client_type from system.sessions where client_type is not null")
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    let b = &batches[0];
    assert_eq!(b.column(0).as_string::<i32>().value(0), "connect");
    assert_eq!(
        b.column(1).as_string::<i32>().value(0),
        "_SPARK_CONNECT_PYTHON spark/4.3.0"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_requests_tags_reach_the_engine_queries_and_cancel_by_tag() {
    use arrow_array::cast::AsArray;
    let run = start().await;
    let mut live = stream(
        &run.channel,
        pb::ExecutePlanRequest {
            session_id: SESSION.to_owned(),
            operation_id: Some(OP.to_owned()),
            plan: Some(sql("SELECT 1 AS a")),
            tags: vec!["etl".to_owned(), "job=42".to_owned()],
            ..Default::default()
        },
    )
    .await;
    while live.message().await.expect("message").is_some() {}
    let batches = run
        .entry()
        .session()
        .sql("select count(*) from system.queries where tags['etl'] is not null and tags['job=42'] is not null")
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    assert_eq!(
        batches[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0),
        1,
        "both request tags are on the operation's query row"
    );
}
