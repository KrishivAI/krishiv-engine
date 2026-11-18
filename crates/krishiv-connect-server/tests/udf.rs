// SPDX-License-Identifier: Apache-2.0
//! Python UDFs over Spark Connect end to end over a raw tonic client (P3-T014, PR-26d part 1; ADR 0046
//! §3b, N1): `register_function` and inline UDFs (`common_inline_user_defined_function`) run on the
//! server's worker pool, in workers of the session's own owner; client code needs `Usage` on
//! `System("udf")` (admin only in server mode until the plan says otherwise) before anything is
//! registered or any worker starts; a UDF that raises is Spark's `PYTHON_EXCEPTION`; releasing the session
//! stops its workers. Relation UDFs (`mapInPandas`, `applyInPandas`, a cogroup's, PR-26d part 2) run on the
//! same workers under the same check.
//!
//! Workers need `KRISHIV_TEST_UDF_PYTHON` (Python >= 3.11 with `pyspark-client`, `pyarrow`, `pandas`), as in
//! `krishiv-udf-pool`'s own tests: without it the tests that run a UDF pass after a note on standard error,
//! unless `KRISHIV_UDF_POOL_REQUIRED` is set. The privilege tests always run (they start no worker).
//! Workers are started through this test binary ([`worker_child`]), as the server's hidden subcommand
//! starts them.
#![cfg(target_os = "linux")]
#![expect(
    clippy::expect_used,
    clippy::print_stderr,
    reason = "integration-test helpers are outside #[test] fns; the skip note"
)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, Int64Array, RecordBatch};
use krishiv_common::config::Config;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::command::CommandType;
use krishiv_connect::proto::v4_3::common_inline_user_defined_function::Function;
use krishiv_connect::proto::v4_3::data_type::{Boolean, Kind, Long};
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::expression::ExprType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::translate::PythonUdf;
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, RunnerConf, SessionLimits, SessionManager,
    SessionUdfRuntime, UdfHost, role_sessions,
};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use krishiv_session::Session;
use krishiv_session::auth::{AuthConfig, ServerAuth};
use krishiv_udf_pool::{Launcher, Pool, PoolConfig};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::Code;
use tonic::transport::Channel;

const SESSION: &str = "5b2d9a3e-1c4f-4e8a-9d6b-7f0e3a2c1b94";
const VIEWER: &str = "viewer-token-0123456789abcdefghijklmnopqrstuv";
const OPERATOR: &str = "operator-token-0123456789abcdefghijklmnopqrstu";
const ADMIN: &str = "admin-token-0123456789abcdefghijklmnopqrstuvw";

/// The child entry: the sandbox trampoline when started with `--` and spec arguments.
#[test]
fn worker_child() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let Some(split) = args.iter().position(|a| a == "--") else {
        return;
    };
    let spec: Vec<OsString> = args.get(split + 1..).unwrap_or_default().to_vec();
    if spec.is_empty() {
        return;
    }
    let code = krishiv_udf_sandbox::trampoline_main(spec);
    let failed = code == ExitCode::from(krishiv_udf_sandbox::EXIT_SANDBOX_FAILED);
    std::process::exit(if failed { 125 } else { 1 });
}

fn python() -> Option<PathBuf> {
    match std::env::var_os("KRISHIV_TEST_UDF_PYTHON") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            assert!(
                std::env::var_os("KRISHIV_UDF_POOL_REQUIRED").is_none(),
                "KRISHIV_UDF_POOL_REQUIRED is set but KRISHIV_TEST_UDF_PYTHON is not"
            );
            eprintln!(
                "note: KRISHIV_TEST_UDF_PYTHON is not set; this test ran no Python UDF worker"
            );
            None
        }
    }
}

struct Running {
    pool: Pool,
    channel: Channel,
    _stop: oneshot::Sender<()>,
    _dir: tempfile::TempDir,
}

fn secret(dir: &Path, name: &str, token: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(name);
    std::fs::write(&path, token).expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    format!("file:{}", path.display())
}

/// An authenticated server (`carol` viewer, `olga` operator, `dana` admin) whose sessions are built as
/// `krishiv server` builds them, its UDFs on a pool of `python` (none: a pool that refuses every call).
async fn start(python: Option<&Path>) -> Running {
    start_with(python, None).await
}

/// [`start`], with the sessions' UDFs on `host` instead of the pool when given.
async fn start_with(python: Option<&Path>, host: Option<Arc<dyn UdfHost>>) -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let v = secret(dir.path(), "v", VIEWER);
    let o = secret(dir.path(), "o", OPERATOR);
    let a = secret(dir.path(), "a", ADMIN);
    let auth = AuthConfig::from_settings([
        ("krishiv.auth.tokens.v.secret", v.as_str()),
        ("krishiv.auth.tokens.v.principal", "carol"),
        ("krishiv.auth.tokens.v.role", "viewer"),
        ("krishiv.auth.tokens.o.secret", o.as_str()),
        ("krishiv.auth.tokens.o.principal", "olga"),
        ("krishiv.auth.tokens.o.role", "operator"),
        ("krishiv.auth.tokens.a.secret", a.as_str()),
        ("krishiv.auth.tokens.a.principal", "dana"),
        ("krishiv.auth.tokens.a.role", "admin"),
    ])
    .expect("auth config");
    let auth = Arc::new(ServerAuth::new(&auth).expect("auth"));
    let mut config = PoolConfig::default();
    config.executables = python
        .map(|p| p.display().to_string())
        .into_iter()
        .collect();
    config.launcher = Launcher {
        program: std::env::current_exe().expect("test binary"),
        args: ["worker_child", "--exact", "--nocapture", "--"]
            .map(OsString::from)
            .to_vec(),
    };
    let pool = Pool::new(config).expect("pool");
    let denials = Arc::new(krishiv_session::auth::DeniedCounts::new());
    let factory = role_sessions(
        Session::new().expect("session"),
        &Config::new(),
        Arc::clone(&denials),
    )
    .expect("factory");
    let manager = Arc::new(
        SessionManager::new(factory, SessionLimits::default())
            .with_udf_host(host.unwrap_or_else(|| Arc::new(pool.clone()))),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(manager, ConnectServerConfig::default())
        .with_auth(ConnectAuth::new(auth).with_denials(denials));
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
        pool,
        channel,
        _stop: stop,
        _dir: dir,
    }
}

fn path(rpc: &str) -> tonic::codegen::http::uri::PathAndQuery {
    format!("/spark.connect.SparkConnectService/{rpc}")
        .parse()
        .expect("path")
}

fn with_token<T>(request: T, token: &str) -> tonic::Request<T> {
    let mut r = tonic::Request::new(request);
    r.metadata_mut().insert(
        "authorization",
        format!("Bearer {token}").parse().expect("ascii"),
    );
    r
}

async fn execute(
    channel: &Channel,
    plan: pb::Plan,
    token: &str,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    execute_in(channel, SESSION, plan, token).await
}

/// [`execute`] in the session `session`.
async fn execute_in(
    channel: &Channel,
    session: &str,
    plan: pb::Plan,
    token: &str,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let request = pb::ExecutePlanRequest {
        session_id: session.to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "someone".to_owned(),
            ..Default::default()
        }),
        plan: Some(plan),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let mut s = grpc
        .server_streaming(
            with_token(request, token),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(r) = s.message().await? {
        out.push(r);
    }
    Ok(out)
}

async fn analyze_schema(
    channel: &Channel,
    plan: pb::Plan,
    token: &str,
) -> Result<pb::AnalyzePlanResponse, tonic::Status> {
    let request = pb::AnalyzePlanRequest {
        session_id: SESSION.to_owned(),
        analyze: Some(pb::analyze_plan_request::Analyze::Schema(Box::new(
            pb::analyze_plan_request::Schema { plan: Some(plan) },
        ))),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        with_token(request, token),
        path("AnalyzePlan"),
        tonic_prost::ProstCodec::<pb::AnalyzePlanRequest, pb::AnalyzePlanResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

async fn release(channel: &Channel, token: &str) {
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        with_token(
            pb::ReleaseSessionRequest {
                session_id: SESSION.to_owned(),
                ..Default::default()
            },
            token,
        ),
        path("ReleaseSession"),
        tonic_prost::ProstCodec::<pb::ReleaseSessionRequest, pb::ReleaseSessionResponse>::default(),
    )
    .await
    .expect("release");
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

#[expect(
    clippy::disallowed_methods,
    reason = "the test reads the status detail the server built"
)]
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

/// Spark's `command` for `body` (a Python function `f`) returning `return_type` (a pyspark type), made by
/// the interpreter's own cloudpickle as a client makes it, and the interpreter's `major.minor`.
fn pickle(python: Option<&Path>, body: &str, return_type: &str) -> (Vec<u8>, String) {
    let Some(python) = python else {
        // Never unpickled: the tests without an interpreter start no worker.
        return (
            format!("not a pickle: {body}").into_bytes(),
            "3.12".to_owned(),
        );
    };
    let code = format!(
        "import sys\nfrom pyspark import cloudpickle\nfrom pyspark.sql.types import *\n{body}\n\
         sys.stdout.buffer.write(cloudpickle.dumps((f, {return_type})))"
    );
    let out = Command::new(python)
        .args(["-c", &code])
        .output()
        .expect("run the interpreter");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let version = Command::new(python)
        .args([
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2], end='')",
        ])
        .output()
        .expect("version");
    (
        out.stdout,
        String::from_utf8_lossy(&version.stdout).into_owned(),
    )
}

fn udf(
    python: Option<&Path>,
    name: &str,
    body: &str,
    returns: Kind,
    args: Vec<pb::Expression>,
) -> pb::CommonInlineUserDefinedFunction {
    let return_type = match returns {
        Kind::Boolean(_) => "BooleanType()",
        _ => "LongType()",
    };
    let (command, python_ver) = pickle(python, body, return_type);
    pb::CommonInlineUserDefinedFunction {
        function_name: name.to_owned(),
        deterministic: true,
        arguments: args,
        function: Some(Function::PythonUdf(pb::PythonUdf {
            output_type: Some(pb::DataType {
                kind: Some(returns),
            }),
            eval_type: 101,
            command,
            python_ver,
            additional_includes: Vec::new(),
        })),
        is_distinct: false,
    }
}

fn long() -> Kind {
    Kind::Long(Long::default())
}

fn boolean() -> Kind {
    Kind::Boolean(Boolean::default())
}

fn expr(kind: ExprType) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(kind),
    }
}

fn id() -> pb::Expression {
    expr(ExprType::UnresolvedAttribute(
        pb::expression::UnresolvedAttribute {
            unparsed_identifier: "id".to_owned(),
            ..Default::default()
        },
    ))
}

fn call(f: pb::CommonInlineUserDefinedFunction) -> pb::Expression {
    expr(ExprType::CommonInlineUserDefinedFunction(f))
}

fn alias(e: pb::Expression, name: &str) -> pb::Expression {
    expr(ExprType::Alias(Box::new(pb::expression::Alias {
        expr: Some(Box::new(e)),
        name: vec![name.to_owned()],
        metadata: None,
    })))
}

fn relation(kind: RelType) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(kind),
    }
}

fn range(end: i64) -> pb::Relation {
    relation(RelType::Range(pb::Range {
        start: Some(0),
        end,
        step: 1,
        num_partitions: None,
    }))
}

fn root(r: pb::Relation) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(r))),
    }
}

/// `range(end).select(<e> AS r)`.
fn select(end: i64, e: pb::Expression) -> pb::Plan {
    root(relation(RelType::Project(Box::new(pb::Project {
        input: Some(Box::new(range(end))),
        expressions: vec![alias(e, "r")],
    }))))
}

fn register(f: pb::CommonInlineUserDefinedFunction) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(CommandType::RegisterFunction(Box::new(f))),
        }))),
    }
}

fn sql(query: &str) -> pb::Plan {
    root(relation(RelType::Sql(pb::Sql {
        query: query.to_owned(),
        ..Default::default()
    })))
}

/// The single `Long` column of the result.
fn longs(responses: &[pb::ExecutePlanResponse]) -> Vec<Option<i64>> {
    let mut out = Vec::new();
    for r in responses {
        let Some(ResponseType::ArrowBatch(b)) = &r.response_type else {
            continue;
        };
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data.to_vec()), None)
                .expect("ipc");
        for batch in reader {
            let batch: RecordBatch = batch.expect("batch");
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("a long column");
            out.extend((0..col.len()).map(|i| (!col.is_null(i)).then(|| col.value(i))));
        }
    }
    out
}

const PLUS_ONE: &str = "def f(x):\n    return x + 1";
const IS_EVEN: &str = "def f(x):\n    return x % 2 == 0";
const BOOM: &str = "def f(x):\n    raise ValueError('boom from the udf')";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_code_needs_usage_on_udf_and_starts_no_worker_without_it() {
    let py = python();
    let run = start(py.as_deref()).await;
    for token in [VIEWER, OPERATOR] {
        // Registration: refused by the command's class.
        let s = execute(
            &run.channel,
            register(udf(py.as_deref(), "plus_one", PLUS_ONE, long(), Vec::new())),
            token,
        )
        .await
        .expect_err("no client code below admin (ADR 0046 N1)");
        assert_eq!(s.code(), Code::PermissionDenied);
        assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        // An inline UDF: refused after translation, before anything is registered.
        let s = execute(
            &run.channel,
            select(
                3,
                call(udf(py.as_deref(), "f", PLUS_ONE, long(), vec![id()])),
            ),
            token,
        )
        .await
        .expect_err("an inline UDF is client code");
        assert_eq!(s.code(), Code::PermissionDenied);
        assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        // The refused registration left nothing behind: the name does not resolve.
        let s = execute(
            &run.channel,
            sql("SELECT plus_one(id) FROM range(3)"),
            token,
        )
        .await
        .expect_err("never registered");
        assert_ne!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        // Typing a plan runs no client code: AnalyzePlan's schema needs no privilege.
        let schema = analyze_schema(
            &run.channel,
            select(
                3,
                call(udf(py.as_deref(), "f", PLUS_ONE, long(), vec![id()])),
            ),
            token,
        )
        .await
        .expect("a scalar UDF is typed without running it");
        assert!(schema.result.is_some());
        release(&run.channel, token).await;
    }
    assert_eq!(
        run.pool.live_workers(),
        0,
        "no worker started for a refused caller"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_registers_and_calls_a_python_udf() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    // A registration answers with no response but the end of the stream (Spark sends none).
    let responses = execute(
        &run.channel,
        register(udf(Some(&py), "plus_one", PLUS_ONE, long(), Vec::new())),
        ADMIN,
    )
    .await
    .expect("admin registers client code");
    assert!(
        responses.iter().all(|r| r.response_type.is_none()
            || matches!(r.response_type, Some(ResponseType::ResultComplete(_)))),
        "{responses:?}"
    );
    let responses = execute(
        &run.channel,
        sql("SELECT plus_one(id) AS r FROM range(3)"),
        ADMIN,
    )
    .await
    .expect("the registered UDF runs");
    assert_eq!(longs(&responses), [Some(1), Some(2), Some(3)]);
    assert!(run.pool.live_workers() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inline_udfs_in_a_projection_and_a_filter() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    let plus_one = || call(udf(Some(&py), "plus_one", PLUS_ONE, long(), vec![id()]));
    let responses = execute(&run.channel, select(4, plus_one()), ADMIN)
        .await
        .expect("an inline UDF in a projection");
    assert_eq!(longs(&responses), [Some(1), Some(2), Some(3), Some(4)]);
    // The same UDF again (a DataFrame re-run) and one in a filter.
    let filtered = root(relation(RelType::Project(Box::new(pb::Project {
        input: Some(Box::new(relation(RelType::Filter(Box::new(pb::Filter {
            input: Some(Box::new(range(6))),
            condition: Some(call(udf(
                Some(&py),
                "is_even",
                IS_EVEN,
                boolean(),
                vec![id()],
            ))),
        }))))),
        expressions: vec![alias(plus_one(), "r")],
    }))));
    let responses = execute(&run.channel, filtered, ADMIN)
        .await
        .expect("inline UDFs in a filter and a projection");
    assert_eq!(longs(&responses), [Some(1), Some(3), Some(5)]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_raising_udf_is_sparks_python_exception() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    let s = execute(
        &run.channel,
        select(2, call(udf(Some(&py), "boom", BOOM, long(), vec![id()]))),
        ADMIN,
    )
    .await
    .expect_err("the UDF raises");
    assert_eq!(s.code(), Code::Internal);
    assert!(
        s.message().starts_with("[PYTHON_EXCEPTION] "),
        "{}",
        s.message()
    );
    assert!(
        s.message().contains("ValueError: boom from the udf"),
        "{}",
        s.message()
    );
    let info = info(&s);
    // PySpark raises its `PythonException` when this class leads the list.
    assert_eq!(info.reason, "org.apache.spark.api.python.PythonException");
    assert!(
        info.metadata["classes"].starts_with(r#"["org.apache.spark.api.python.PythonException""#),
        "{}",
        info.metadata["classes"]
    );
    assert_eq!(info.metadata["sqlState"], "38000");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn releasing_the_session_stops_its_workers() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    let responses = execute(
        &run.channel,
        select(
            3,
            call(udf(Some(&py), "plus_one", PLUS_ONE, long(), vec![id()])),
        ),
        ADMIN,
    )
    .await
    .expect("runs");
    assert_eq!(longs(&responses), [Some(1), Some(2), Some(3)]);
    assert!(run.pool.live_workers() > 0, "the worker stays for reuse");
    release(&run.channel, ADMIN).await;
    // Idle workers stop at once; one still handing back its result stops when it does.
    let deadline = Instant::now() + Duration::from_secs(10);
    while run.pool.live_workers() > 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(run.pool.live_workers(), 0);
}

/// A host that starts nothing and records every payload registered with it.
#[derive(Default)]
struct Spy(Arc<std::sync::Mutex<Vec<String>>>);

impl std::fmt::Debug for Spy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Spy")
    }
}

#[derive(Debug)]
struct SpyRuntime(Arc<std::sync::Mutex<Vec<String>>>);

impl UdfRuntime for SpyRuntime {
    fn prepare(&self, spec: &UdfCallSpec) -> krishiv_exec::Result<Arc<dyn PreparedUdf>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(format!("prepare {}", spec.name));
        Err(krishiv_common::error::KrishivUdfNotAvailable {
            name: spec.name.clone(),
            version: spec.version.to_string(),
        }
        .into())
    }
}

impl SessionUdfRuntime for SpyRuntime {
    fn register(&self, name: &str, version: u32, _: &PythonUdf) {
        self.0
            .lock()
            .expect("spy")
            .push(format!("register {name}@{version}"));
    }
    fn release(&self) {}
}

impl UdfHost for Spy {
    fn session_runtime(&self, _: &str, _: &str, _: RunnerConf) -> Arc<dyn SessionUdfRuntime> {
        Arc::new(SpyRuntime(Arc::clone(&self.0)))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_caller_registers_nothing_and_gets_no_runtime() {
    let spy = Spy::default();
    let log = Arc::clone(&spy.0);
    let run = start_with(None, Some(Arc::new(spy))).await;
    for token in [VIEWER, OPERATOR] {
        let _ = execute(
            &run.channel,
            register(udf(None, "plus_one", PLUS_ONE, long(), Vec::new())),
            token,
        )
        .await
        .expect_err("refused");
        let _ = execute(
            &run.channel,
            select(3, call(udf(None, "f", PLUS_ONE, long(), vec![id()]))),
            token,
        )
        .await
        .expect_err("refused");
        // The refused registration's name resolves to nothing, so nothing reaches the host.
        let _ = execute(
            &run.channel,
            sql("SELECT plus_one(id) FROM range(3)"),
            token,
        )
        .await;
        release(&run.channel, token).await;
    }
    assert!(
        log.lock().expect("spy").is_empty(),
        "nothing registered or prepared for a refused caller: {:?}",
        log.lock().expect("spy")
    );
    // The same plans from the admin do reach the host, so the spy sees what it should.
    execute(
        &run.channel,
        register(udf(None, "plus_one", PLUS_ONE, long(), Vec::new())),
        ADMIN,
    )
    .await
    .expect("admin registers");
    let _ = execute(
        &run.channel,
        select(3, call(udf(None, "f", PLUS_ONE, long(), vec![id()]))),
        ADMIN,
    )
    .await;
    let seen = log.lock().expect("spy").clone();
    assert!(seen.iter().any(|e| e == "register plus_one@1"), "{seen:?}");
    assert!(
        seen.iter().any(|e| e.starts_with("register f@")),
        "{seen:?}"
    );
}

/// `range(end).mapInPandas(f, "x bigint")`: a `MapPartitions` over a Python UDF of eval type 205, `f` being
/// `body` pickled by `python`'s cloudpickle (none: a payload that is never unpickled).
fn map_in_pandas(python: Option<&Path>, body: &str, end: i64) -> pb::Plan {
    let (command, python_ver) = pickle(python, body, "StructType([StructField('x', LongType())])");
    let schema = pb::DataType {
        kind: Some(Kind::Struct(pb::data_type::Struct {
            fields: vec![pb::data_type::StructField {
                name: "x".to_owned(),
                data_type: Some(pb::DataType { kind: Some(long()) }),
                nullable: true,
                metadata: None,
            }],
            type_variation_reference: 0,
        })),
    };
    let func = pb::CommonInlineUserDefinedFunction {
        function_name: "f".to_owned(),
        deterministic: true,
        arguments: vec![id()],
        function: Some(Function::PythonUdf(pb::PythonUdf {
            output_type: Some(schema),
            eval_type: 205,
            command,
            python_ver,
            additional_includes: Vec::new(),
        })),
        is_distinct: false,
    };
    root(relation(RelType::MapPartitions(Box::new(
        pb::MapPartitions {
            input: Some(Box::new(range(end))),
            func: Some(func),
            is_barrier: None,
            profile_id: None,
        },
    ))))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_relation_udf_is_client_code_and_is_typed_without_a_privilege() {
    let spy = Spy::default();
    let log = Arc::clone(&spy.0);
    let run = start_with(None, Some(Arc::new(spy))).await;
    for token in [VIEWER, OPERATOR] {
        let s = execute(&run.channel, map_in_pandas(None, TIMES_TEN, 3), token)
            .await
            .expect_err("a relation UDF is client code (ADR 0046 N1)");
        assert_eq!(s.code(), Code::PermissionDenied);
        assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        // Typing it runs no client code: the schema needs no privilege.
        let schema = analyze_schema(&run.channel, map_in_pandas(None, TIMES_TEN, 3), token)
            .await
            .expect("a relation UDF is typed without running it");
        assert!(schema.result.is_some());
        release(&run.channel, token).await;
    }
    assert!(
        log.lock().expect("spy").is_empty(),
        "nothing registered or prepared for a refused caller: {:?}",
        log.lock().expect("spy")
    );
    // The admin passes the check and registers the payload; the spy prepares nothing, so the plan fails
    // when it is planned (the real pool runs it: `admin_runs_map_in_pandas_on_the_pool`).
    let s = execute(&run.channel, map_in_pandas(None, TIMES_TEN, 3), ADMIN)
        .await
        .expect_err("the spy runs no relation");
    assert_eq!(error_class(&s), "KRISHIV_UDF_NOT_AVAILABLE", "{s:?}");
    let seen = log.lock().expect("spy").clone();
    assert!(
        seen.iter().any(|e| e.starts_with("register f@")),
        "{seen:?}"
    );
}

const TIMES_TEN: &str =
    "def f(it):\n    for pdf in it:\n        yield pdf.assign(x=pdf.id * 10)[['x']]";
const MAP_BOOM: &str = "def f(it):\n    for pdf in it:\n        raise ValueError('boom from the map')\n        yield pdf";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_runs_map_in_pandas_on_the_pool() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    // A viewer is refused before any worker starts (ADR 0046 N1).
    let s = execute(&run.channel, map_in_pandas(Some(&py), TIMES_TEN, 4), VIEWER)
        .await
        .expect_err("client code");
    assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    assert_eq!(run.pool.live_workers(), 0, "no worker for a refused caller");
    // The id is the viewer's session until released (sessions are keyed by their owner).
    release(&run.channel, VIEWER).await;
    let responses = execute(&run.channel, map_in_pandas(Some(&py), TIMES_TEN, 4), ADMIN)
        .await
        .expect("mapInPandas runs on the session's workers");
    let mut got = longs(&responses);
    got.sort_unstable();
    assert_eq!(got, [Some(0), Some(10), Some(20), Some(30)]);
    assert!(run.pool.live_workers() > 0);
    // The same DataFrame again keeps its version and runs again.
    let again = execute(&run.channel, map_in_pandas(Some(&py), TIMES_TEN, 4), ADMIN)
        .await
        .expect("re-run");
    assert_eq!(longs(&again).len(), 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_raising_map_in_pandas_is_sparks_python_exception() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    let s = execute(&run.channel, map_in_pandas(Some(&py), MAP_BOOM, 3), ADMIN)
        .await
        .expect_err("the map raises");
    assert_eq!(error_class(&s), "PYTHON_EXCEPTION", "{s:?}");
    assert!(s.message().contains("boom from the map"), "{s:?}");
    // The session still runs relations afterwards.
    let responses = execute(&run.channel, map_in_pandas(Some(&py), TIMES_TEN, 2), ADMIN)
        .await
        .expect("a later map runs");
    assert_eq!(longs(&responses).len(), 2);
}

/// An unresolved column reference.
fn col(name: &str) -> pb::Expression {
    expr(ExprType::UnresolvedAttribute(
        pb::expression::UnresolvedAttribute {
            unparsed_identifier: name.to_owned(),
            ..Default::default()
        },
    ))
}

/// A Python function of `eval_type` returning the struct of long columns `fields`, pickled by `python`.
fn relation_udf(
    python: &Path,
    body: &str,
    eval_type: i32,
    fields: &[&str],
    args: Vec<pb::Expression>,
) -> pb::CommonInlineUserDefinedFunction {
    let struct_type = format!(
        "StructType([{}])",
        fields
            .iter()
            .map(|f| format!("StructField('{f}', LongType())"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let (command, python_ver) = pickle(Some(python), body, &struct_type);
    let schema = pb::DataType {
        kind: Some(Kind::Struct(pb::data_type::Struct {
            fields: fields
                .iter()
                .map(|f| pb::data_type::StructField {
                    name: (*f).to_owned(),
                    data_type: Some(pb::DataType { kind: Some(long()) }),
                    nullable: true,
                    metadata: None,
                })
                .collect(),
            type_variation_reference: 0,
        })),
    };
    pb::CommonInlineUserDefinedFunction {
        function_name: "f".to_owned(),
        deterministic: true,
        arguments: args,
        function: Some(Function::PythonUdf(pb::PythonUdf {
            output_type: Some(schema),
            eval_type,
            command,
            python_ver,
            additional_includes: Vec::new(),
        })),
        is_distinct: false,
    }
}

fn sql_relation(query: &str) -> pb::Relation {
    relation(RelType::Sql(pb::Sql {
        query: query.to_owned(),
        ..Default::default()
    }))
}

/// Every row of the result's first two (long) columns, sorted.
fn long_pairs(responses: &[pb::ExecutePlanResponse]) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for r in responses {
        let Some(ResponseType::ArrowBatch(b)) = &r.response_type else {
            continue;
        };
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data.to_vec()), None)
                .expect("ipc");
        for batch in reader {
            let batch: RecordBatch = batch.expect("batch");
            let column = |i: usize| {
                batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("a long column")
                    .clone()
            };
            let (a, b) = (column(0), column(1));
            out.extend((0..batch.num_rows()).map(|i| (a.value(i), b.value(i))));
        }
    }
    out.sort_unstable();
    out
}

const LEFT: &str = "SELECT id % 2 AS k, id AS v FROM range(6)";
const RIGHT: &str = "SELECT id % 2 AS k, id * 100 AS w FROM range(3)";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_runs_map_in_arrow() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    // range(4).mapInArrow(f, "x long"): eval type 207, pyarrow RecordBatches in and out.
    let f = relation_udf(
        &py,
        "import pyarrow as pa\nimport pyarrow.compute as pc\ndef f(it):\n    for b in it:\n        \
         yield pa.RecordBatch.from_arrays([pc.multiply(b.column('id'), 10)], ['x'])",
        207,
        &["x"],
        vec![id()],
    );
    let plan = root(relation(RelType::MapPartitions(Box::new(
        pb::MapPartitions {
            input: Some(Box::new(range(4))),
            func: Some(f),
            is_barrier: None,
            profile_id: None,
        },
    ))));
    let responses = execute(&run.channel, plan, ADMIN)
        .await
        .expect("mapInArrow runs on the session's workers");
    let mut got = longs(&responses);
    got.sort_unstable();
    assert_eq!(got, [Some(0), Some(10), Some(20), Some(30)]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_runs_each_grouped_map_kind_per_group() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    // df.groupBy("k").applyInPandas / applyInArrow (f, "k long, s long"), plain and iterator forms.
    let cases = [
        (
            201,
            "import pandas as pd\ndef f(pdf):\n    \
             return pd.DataFrame({'k': [pdf.k.iloc[0]], 's': [pdf.v.sum()]})",
        ),
        (
            201,
            "import pandas as pd\ndef f(key, pdf):\n    \
             return pd.DataFrame({'k': [key[0]], 's': [pdf.v.sum()]})",
        ),
        (
            209,
            "import pyarrow as pa\nimport pyarrow.compute as pc\ndef f(key, t):\n    \
             return pa.table({'k': [key[0].as_py()], 's': [pc.sum(t.column('v')).as_py()]})",
        ),
        (
            216,
            "import pandas as pd\ndef f(key, it):\n    s = sum(int(p.v.sum()) for p in it)\n    \
             yield pd.DataFrame({'k': [key[0]], 's': [s]})",
        ),
        (
            215,
            "import pyarrow as pa\nimport pyarrow.compute as pc\ndef f(it):\n    batches = list(it)\n    \
             s = sum(pc.sum(b.column('v')).as_py() for b in batches)\n    \
             yield pa.RecordBatch.from_pydict({'k': [batches[0].column('k')[0].as_py()], 's': [s]})",
        ),
    ];
    for (eval_type, body) in cases {
        let f = relation_udf(&py, body, eval_type, &["k", "s"], vec![col("k"), col("v")]);
        let plan = root(relation(RelType::GroupMap(Box::new(pb::GroupMap {
            input: Some(Box::new(sql_relation(LEFT))),
            grouping_expressions: vec![col("k")],
            func: Some(f),
            ..Default::default()
        }))));
        let responses = execute(&run.channel, plan, ADMIN)
            .await
            .unwrap_or_else(|s| panic!("{eval_type} runs on the session's workers: {s:?}"));
        assert_eq!(
            long_pairs(&responses),
            [(0, 6), (1, 9)],
            "{eval_type}: {body}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_runs_each_cogrouped_map_kind() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    // l.groupBy("k").cogroup(r.groupBy("k")).applyInPandas / applyInArrow (f, "k long, n long").
    let cases = [
        (
            206,
            "import pandas as pd\ndef f(l, r):\n    \
             return pd.DataFrame({'k': [l.k.iloc[0]], 'n': [len(l) * 10 + len(r)]})",
        ),
        (
            210,
            "import pyarrow as pa\ndef f(l, r):\n    \
             return pa.table({'k': [l.column('k')[0].as_py()], 'n': [l.num_rows * 10 + r.num_rows]})",
        ),
    ];
    for (eval_type, body) in cases {
        let f = relation_udf(
            &py,
            body,
            eval_type,
            &["k", "n"],
            vec![col("k"), col("v"), col("k"), col("w")],
        );
        let plan = root(relation(RelType::CoGroupMap(Box::new(pb::CoGroupMap {
            input: Some(Box::new(sql_relation(LEFT))),
            input_grouping_expressions: vec![col("k")],
            other: Some(Box::new(sql_relation(RIGHT))),
            other_grouping_expressions: vec![col("k")],
            func: Some(f),
            ..Default::default()
        }))));
        let responses = execute(&run.channel, plan, ADMIN)
            .await
            .unwrap_or_else(|s| panic!("{eval_type} runs on the session's workers: {s:?}"));
        // Key 0: three left rows (0, 2, 4) and two right (0, 2); key 1: three left and one right.
        assert_eq!(
            long_pairs(&responses),
            [(0, 32), (1, 31)],
            "{eval_type}: {body}"
        );
    }
}

async fn semantic_hash(channel: &Channel, plan: pb::Plan) -> i32 {
    let request = pb::AnalyzePlanRequest {
        session_id: SESSION.to_owned(),
        analyze: Some(pb::analyze_plan_request::Analyze::SemanticHash(Box::new(
            pb::analyze_plan_request::SemanticHash { plan: Some(plan) },
        ))),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let response = grpc
        .unary(
            with_token(request, ADMIN),
            path("AnalyzePlan"),
            tonic_prost::ProstCodec::<pb::AnalyzePlanRequest, pb::AnalyzePlanResponse>::default(),
        )
        .await
        .expect("semantic hash")
        .into_inner();
    match response.result {
        Some(pb::analyze_plan_response::Result::SemanticHash(h)) => Some(h.result),
        _ => None,
    }
    .expect("a semantic hash")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_semantic_hash_covers_the_udf_body() {
    let run = start(None).await;
    let plan = |body| select(3, call(udf(None, "f", body, long(), vec![id()])));
    let one = semantic_hash(&run.channel, plan(PLUS_ONE)).await;
    assert_eq!(one, semantic_hash(&run.channel, plan(PLUS_ONE)).await);
    // Same name, types and determinism, another function: not the same plan, as in Spark.
    assert_ne!(one, semantic_hash(&run.channel, plan(BOOM)).await);
}

const TOKYO: &str = "def f(x):\n    from krishiv._udf._host import runner_conf\n    \
                     return 1 if runner_conf().get('spark.sql.session.timeZone') == 'Asia/Tokyo' else 0";

async fn set_conf(channel: &Channel, key: &str, value: &str) {
    use pb::config_request::operation::OpType;
    let request = pb::ConfigRequest {
        session_id: SESSION.to_owned(),
        operation: Some(pb::config_request::Operation {
            op_type: Some(OpType::Set(pb::config_request::Set {
                pairs: vec![pb::KeyValue {
                    key: key.to_owned(),
                    value: Some(value.to_owned()),
                }],
                silent: None,
            })),
        }),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        with_token(request, ADMIN),
        path("Config"),
        tonic_prost::ProstCodec::<pb::ConfigRequest, pb::ConfigResponse>::default(),
    )
    .await
    .expect("config set");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_call_sends_the_session_conf_current_when_it_runs() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    execute(
        &run.channel,
        register(udf(Some(&py), "is_tokyo", TOKYO, long(), Vec::new())),
        ADMIN,
    )
    .await
    .expect("registered");
    let query = || sql("SELECT is_tokyo(id) AS r FROM range(1)");
    set_conf(&run.channel, "spark.sql.session.timeZone", "UTC").await;
    let r = execute(&run.channel, query(), ADMIN).await.expect("runs");
    assert_eq!(longs(&r), [Some(0)]);
    // Changed after the registration: the next call of the registered UDF sees it.
    set_conf(&run.channel, "spark.sql.session.timeZone", "Asia/Tokyo").await;
    let r = execute(&run.channel, query(), ADMIN).await.expect("runs");
    assert_eq!(longs(&r), [Some(1)]);
}

// ---------------------------------------------------------------------------------------------------------
// Python UDTFs (PR-26c-4, the Connect side): `register_table_function` (command 10) and a DataFrame call of a
// UDTF object (`common_inline_user_defined_table_function`, relation 38).
// ---------------------------------------------------------------------------------------------------------

/// Spark Connect's UDTF `command` for `body` (a Python class `F`): the class alone, cloudpickled, as a client
/// sends it, and the interpreter's `major.minor`.
fn pickle_udtf(python: Option<&Path>, body: &str) -> (Vec<u8>, String) {
    let Some(python) = python else {
        return (
            format!("not a pickle: {body}").into_bytes(),
            "3.12".to_owned(),
        );
    };
    let code = format!(
        "import sys\nfrom pyspark import cloudpickle\nfrom pyspark.sql.types import *\n\
         from pyspark.sql.udtf import AnalyzeArgument, AnalyzeResult\n{body}\n\
         sys.stdout.buffer.write(cloudpickle.dumps(F))"
    );
    let out = Command::new(python)
        .args(["-c", &code])
        .output()
        .expect("run the interpreter");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    (out.stdout, python_ver(python))
}

fn python_ver(python: &Path) -> String {
    let version = Command::new(python)
        .args([
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2], end='')",
        ])
        .output()
        .expect("version");
    String::from_utf8_lossy(&version.stdout).into_owned()
}

/// A Python UDTF definition as a client sends it: `return_type` as the string a client passes
/// (`@udtf(returnType="i bigint")` is an `unparsed` type), none for a UDTF with `analyze`.
fn udtf_message(
    name: &str,
    command: (Vec<u8>, String),
    eval_type: i32,
    return_type: Option<&str>,
    arguments: Vec<pb::Expression>,
) -> pb::CommonInlineUserDefinedTableFunction {
    pb::CommonInlineUserDefinedTableFunction {
        function_name: name.to_owned(),
        deterministic: false,
        arguments,
        function: Some(
            pb::common_inline_user_defined_table_function::Function::PythonUdtf(pb::PythonUdtf {
                return_type: return_type.map(|t| pb::DataType {
                    kind: Some(Kind::Unparsed(pb::data_type::Unparsed {
                        data_type_string: t.to_owned(),
                    })),
                }),
                eval_type,
                command: command.0,
                python_ver: command.1,
            }),
        ),
    }
}

fn register_udtf(f: pb::CommonInlineUserDefinedTableFunction) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(CommandType::RegisterTableFunction(Box::new(f))),
        }))),
    }
}

fn inline_udtf(f: pb::CommonInlineUserDefinedTableFunction) -> pb::Plan {
    root(relation(RelType::CommonInlineUserDefinedTableFunction(f)))
}

fn lit_long(v: i64) -> pb::Expression {
    expr(ExprType::Literal(pb::expression::Literal {
        literal_type: Some(pb::expression::literal::LiteralType::Long(v)),
        data_type: None,
    }))
}

fn lit_int(v: i32) -> pb::Expression {
    expr(ExprType::Literal(pb::expression::Literal {
        literal_type: Some(pb::expression::literal::LiteralType::Integer(v)),
        data_type: None,
    }))
}

/// The result's schema (`name:type`, as `simpleString`) and its rows of integer columns, sorted.
fn int_rows(responses: &[pb::ExecutePlanResponse]) -> (String, Vec<Vec<Option<i64>>>) {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type};
    let mut schema = String::new();
    let mut rows = Vec::new();
    for r in responses {
        let Some(ResponseType::ArrowBatch(b)) = &r.response_type else {
            continue;
        };
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data.to_vec()), None)
                .expect("ipc");
        for batch in reader {
            let batch: RecordBatch = batch.expect("batch");
            schema = format!(
                "struct<{}>",
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| format!(
                        "{}:{}",
                        f.name(),
                        match f.data_type() {
                            arrow_schema::DataType::Int32 => "int".to_owned(),
                            arrow_schema::DataType::Int64 => "bigint".to_owned(),
                            // Not an integer column: shown as such, so the comparison fails.
                            other => other.to_string(),
                        }
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            for i in 0..batch.num_rows() {
                rows.push(
                    batch
                        .columns()
                        .iter()
                        .map(|c| {
                            (!c.is_null(i)).then(|| match c.data_type() {
                                arrow_schema::DataType::Int32 => {
                                    i64::from(c.as_primitive::<Int32Type>().value(i))
                                }
                                _ => c.as_primitive::<Int64Type>().value(i),
                            })
                        })
                        .collect(),
                );
            }
        }
    }
    rows.sort();
    (schema, rows)
}

/// `eval(n)` yields `0..n` (row kinds 300 and 301).
const RANGE_ROWS: &str =
    "class F:\n    def eval(self, n):\n        for i in range(n):\n            yield (i,)";
/// The same for `@arrow_udtf` (302): `eval` gets arrays and yields tables.
const RANGE_ARROW: &str = "import pyarrow as pa\nclass F:\n    def eval(self, n):\n        \
                           k = n.to_pylist()[0]\n        \
                           yield pa.table({'i': pa.array(list(range(k)), type=pa.int64())})";
/// A UDTF with `analyze` and no return type: the result is `i bigint`, given per call.
const RANGE_ANALYZED: &str = "class F:\n    @staticmethod\n    def analyze(n):\n        \
                              return AnalyzeResult(StructType().add('i', LongType()))\n    \
                              def eval(self, n):\n        for i in range(n):\n            yield (i,)";

fn range_body(eval_type: i32) -> &'static str {
    if eval_type == 302 {
        RANGE_ARROW
    } else {
        RANGE_ROWS
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_udtf_is_client_code_and_a_viewer_starts_no_worker() {
    let py = python();
    let run = start(py.as_deref()).await;
    let typed = || {
        udtf_message(
            "range_udtf",
            pickle_udtf(py.as_deref(), RANGE_ROWS),
            301,
            Some("i bigint"),
            vec![lit_long(3)],
        )
    };
    let analyzed = || {
        udtf_message(
            "range_udtf",
            pickle_udtf(py.as_deref(), RANGE_ANALYZED),
            301,
            None,
            vec![lit_long(3)],
        )
    };
    for token in [VIEWER, OPERATOR] {
        // Command 10: refused by the command's class.
        let s = execute(&run.channel, register_udtf(typed()), token)
            .await
            .expect_err("no client code below admin (ADR 0046 N1)");
        assert_eq!(s.code(), Code::PermissionDenied);
        assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        // Relation 38: refused after translation, before anything runs.
        let s = execute(&run.channel, inline_udtf(typed()), token)
            .await
            .expect_err("an inline UDTF is client code");
        assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        // The refused registration left nothing behind.
        let s = execute(&run.channel, sql("SELECT * FROM range_udtf(3)"), token)
            .await
            .expect_err("never registered");
        assert_eq!(
            error_class(&s),
            "UNRESOLVABLE_TABLE_VALUED_FUNCTION",
            "{s:?}"
        );
        // Typing a UDTF with a return type runs no client code: no privilege needed.
        let schema = analyze_schema(&run.channel, inline_udtf(typed()), token)
            .await
            .expect("a typed UDTF is analyzed without running it");
        assert!(schema.result.is_some());
        // One with `analyze` runs Python to be typed: `Usage` on `System("udf")` first (ADR 0046 §3b).
        let s = analyze_schema(&run.channel, inline_udtf(analyzed()), token)
            .await
            .expect_err("analyze is client code");
        assert_eq!(error_class(&s), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        release(&run.channel, token).await;
    }
    assert_eq!(
        run.pool.live_workers(),
        0,
        "no worker started for a refused caller"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_registers_and_calls_a_udtf_of_each_eval_type() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    for eval_type in [300, 301, 302] {
        let name = format!("range_{eval_type}");
        let message = |args| {
            udtf_message(
                &name,
                pickle_udtf(Some(&py), range_body(eval_type)),
                eval_type,
                Some("i bigint"),
                args,
            )
        };
        // A registration answers with no response but the end of the stream.
        let responses = execute(&run.channel, register_udtf(message(Vec::new())), ADMIN)
            .await
            .unwrap_or_else(|e| panic!("{eval_type}: {e:?}"));
        assert!(
            responses.iter().all(|r| r.response_type.is_none()
                || matches!(r.response_type, Some(ResponseType::ResultComplete(_)))),
            "{responses:?}"
        );
        let by_sql = execute(
            &run.channel,
            sql(&format!("SELECT * FROM {name}(3)")),
            ADMIN,
        )
        .await
        .unwrap_or_else(|e| panic!("{eval_type}: {e:?}"));
        let want = (
            "struct<i:bigint>".to_owned(),
            vec![vec![Some(0)], vec![Some(1)], vec![Some(2)]],
        );
        assert_eq!(int_rows(&by_sql), want, "{eval_type} by SQL");
        // The same UDTF object called on columns: relation 38 gives the same rows.
        let inline = execute(&run.channel, inline_udtf(message(vec![lit_long(3)])), ADMIN)
            .await
            .unwrap_or_else(|e| panic!("{eval_type}: {e:?}"));
        assert_eq!(int_rows(&inline), want, "{eval_type} inline");
        // Sent again (a DataFrame re-run), it keeps its registration and runs again.
        let again = execute(&run.channel, inline_udtf(message(vec![lit_long(2)])), ADMIN)
            .await
            .unwrap_or_else(|e| panic!("{eval_type}: {e:?}"));
        assert_eq!(int_rows(&again).1.len(), 2);
    }
    assert!(run.pool.live_workers() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_udtf_with_analyze_and_no_return_type() {
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    let message = |args| {
        udtf_message(
            "ranged",
            pickle_udtf(Some(&py), RANGE_ANALYZED),
            301,
            None,
            args,
        )
    };
    execute(&run.channel, register_udtf(message(Vec::new())), ADMIN)
        .await
        .expect("registered");
    let want = (
        "struct<i:bigint>".to_owned(),
        vec![vec![Some(0)], vec![Some(1)]],
    );
    let by_sql = execute(&run.channel, sql("SELECT * FROM ranged(2)"), ADMIN)
        .await
        .expect("analyze types the call");
    assert_eq!(int_rows(&by_sql), want);
    let inline = execute(&run.channel, inline_udtf(message(vec![lit_long(2)])), ADMIN)
        .await
        .expect("inline, analyze types the call");
    assert_eq!(int_rows(&inline), want);
    // The admin's AnalyzePlan runs `analyze` too.
    let schema = analyze_schema(&run.channel, inline_udtf(message(vec![lit_long(2)])), ADMIN)
        .await
        .expect("typed by analyze");
    let Some(pb::analyze_plan_response::Result::Schema(s)) = schema.result else {
        panic!("a schema: {schema:?}");
    };
    // `analyze` gave `struct<i: bigint>`: one nullable long field `i`.
    let Some(Kind::Struct(row)) = s.schema.and_then(|t| t.kind) else {
        panic!("a struct schema");
    };
    let [field] = row.fields.as_slice() else {
        panic!("one field: {row:?}");
    };
    assert_eq!(field.name, "i");
    assert!(field.nullable);
    assert!(
        matches!(
            field.data_type.as_ref().and_then(|t| t.kind.as_ref()),
            Some(Kind::Long(_))
        ),
        "{field:?}"
    );
}

/// The DataFrame-path cases of `python/tests/pyspark_udf/udtf/` that relation 38 runs: `rows/udtf<k>/eval_dataframe`
/// (`make_repeat(api, kind)(lit(7), lit(3))`) for 300, 301 and 302, each the case's own class from `cases.py`
/// (cloudpickled by value, as `export_cases.py` pickles it) sent as a stock client sends the call, compared with
/// PySpark 4.2.0's answer in `goldens.json` (rows sorted, as the case orders them by every column; these rows
/// do not depend on the Spark version, so they stand for 4.3.0 too). The other
/// DataFrame cases need `t.asTable()` (a `TABLE` argument, refused) or `lateralJoin` (relation 44, not translated).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_dataframe_udtf_goldens_that_relation_38_runs() {
    let Some(py) = python() else { return };
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cases = repo.join("python/tests");
    let goldens: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(cases.join("pyspark_udf/udtf/goldens.json")).expect("goldens"),
    )
    .expect("json");
    let run = start(Some(&py)).await;
    for kind in ["300", "301", "302"] {
        let code = format!(
            "import sys\nsys.path.insert(0, {:?})\nfrom pyspark import cloudpickle\n\
             from pyspark_udf.udtf import cases\ncloudpickle.register_pickle_by_value(cases)\n\
             f = cases.make_repeat(cases.Api('pyspark'), {kind:?})\n\
             assert f.evalType == {kind}, f.evalType\n\
             sys.stdout.buffer.write(cloudpickle.dumps(f.func))",
            cases.display().to_string()
        );
        // `pyspark.sql.connect` imports `pyspark.testing.sqlutils`, which looks for a Spark home at import time
        // (a client-only install has none): any directory satisfies it.
        let out = Command::new(&py)
            .args(["-c", &code])
            .env("SPARK_HOME", repo.join("python"))
            .output()
            .expect("run the interpreter");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let eval_type: i32 = kind.parse().expect("kind");
        let message = udtf_message(
            "Repeat",
            (out.stdout, python_ver(&py)),
            eval_type,
            Some("a int, i int"),
            vec![lit_int(7), lit_int(3)],
        );
        let responses = execute(&run.channel, inline_udtf(message), ADMIN)
            .await
            .unwrap_or_else(|e| panic!("{kind}: {e:?}"));
        let want = &goldens["cases"][format!("rows/udtf{kind}/eval_dataframe")];
        let (schema, rows) = int_rows(&responses);
        assert_eq!(schema, want["schema"].as_str().expect("schema"), "{kind}");
        let mut want_rows: Vec<Vec<Option<i64>>> = want["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .map(|r| {
                r.as_array()
                    .expect("row")
                    .iter()
                    .map(serde_json::Value::as_i64)
                    .collect()
            })
            .collect();
        want_rows.sort();
        assert_eq!(rows, want_rows, "{kind}");
    }
}

/// A `TABLE` argument (`udtf(df.asTable())`) is refused with `KRISHIV_UNSUPPORTED_FEATURE` before anything is
/// registered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_table_argument_is_refused() {
    let run = start(None).await;
    let table = expr(ExprType::SubqueryExpression(pb::SubqueryExpression {
        plan_id: 1,
        subquery_type: pb::subquery_expression::SubqueryType::TableArg as i32,
        table_arg_options: None,
        in_subquery_values: Vec::new(),
    }));
    let message = udtf_message(
        "t",
        pickle_udtf(None, RANGE_ROWS),
        301,
        Some("i bigint"),
        vec![table],
    );
    let s = execute(&run.channel, inline_udtf(message), ADMIN)
        .await
        .expect_err("TABLE arguments are not supported yet");
    assert_eq!(error_class(&s), "KRISHIV_UNSUPPORTED_FEATURE", "{s:?}");
}

/// A `CloneSession` clone runs the source's inline UDTF: the same relation 38 in the clone gives the same
/// rows, on the clone's own workers (its runtime holds the source's payloads, `SessionPython::copy_for`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_clone_runs_the_sources_inline_udtf() {
    const CLONE: &str = "8c1f0e7a-4b2d-4e9a-a3c5-2d7f6b1e9a40";
    let Some(py) = python() else { return };
    let run = start(Some(&py)).await;
    let call = || {
        inline_udtf(udtf_message(
            "range_udtf",
            pickle_udtf(Some(&py), RANGE_ROWS),
            301,
            Some("i bigint"),
            vec![lit_long(3)],
        ))
    };
    let want = (
        "struct<i:bigint>".to_owned(),
        vec![vec![Some(0)], vec![Some(1)], vec![Some(2)]],
    );
    let source = execute(&run.channel, call(), ADMIN).await.expect("source");
    assert_eq!(int_rows(&source), want);
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let cloned = grpc
        .unary(
            with_token(
                pb::CloneSessionRequest {
                    session_id: SESSION.to_owned(),
                    new_session_id: Some(CLONE.to_owned()),
                    client_type: Some("python".to_owned()),
                    ..Default::default()
                },
                ADMIN,
            ),
            path("CloneSession"),
            tonic_prost::ProstCodec::<pb::CloneSessionRequest, pb::CloneSessionResponse>::default(),
        )
        .await
        .expect("clone")
        .into_inner();
    assert_eq!(cloned.new_session_id, CLONE);
    let in_clone = execute_in(&run.channel, CLONE, call(), ADMIN)
        .await
        .expect("the clone runs the inline UDTF");
    assert_eq!(int_rows(&in_clone), want);
    // The source still runs it after the clone.
    let again = execute(&run.channel, call(), ADMIN)
        .await
        .expect("source again");
    assert_eq!(int_rows(&again), want);
}
