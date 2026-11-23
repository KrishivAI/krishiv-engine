// SPDX-License-Identifier: Apache-2.0
//! The `external` provider end to end (ADR 0049 §3, review B2): SQL through a session whose
//! policy is what `krishiv server` builds, `PathPrefix(Intersection[role matrix, external])`
//! (`krishiv_connect_server::provider_sessions`), against an in-process `krishiv.access.v1`
//! service. Each refusal is a 42501 that never names the service's reason, and a service that is
//! slow or down refuses everything.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_connect_server::{ExtraProvider, provider_sessions};
use krishiv_server::access_external::{
    AccessPolicyClient, AccessPolicyServer, AccessPolicyService, Endpoint, ExternalAccess, proto,
};
use krishiv_session::auth::{Authenticated, DeniedCounts, Method, Role};
use krishiv_session::{
    AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session, Transport,
};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";
const CANARY: &str = "Canary-reason-5e1f";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Deny every securable whose name starts with `secret` (tables, views, columns' tables) and
    /// every path with a `secret` segment.
    Rule,
    /// Answer after the deadline.
    Slow,
    /// Fail every check with a `PermissionDenied` status whose message is `CANARY`.
    Canary,
}

struct Service {
    mode: Mode,
    items: Arc<Mutex<Vec<String>>>,
    /// The `(statement_class, query_id)` of every check.
    statements: Arc<Mutex<Vec<(String, String)>>>,
}

fn name_of(item: &proto::Item) -> String {
    use proto::securable::Kind;
    match item.securable.as_ref().and_then(|s| s.kind.as_ref()) {
        Some(Kind::Table(t) | Kind::View(t)) => t.name.clone(),
        Some(Kind::Column(c)) => c.table.as_ref().map(|t| t.name.clone()).unwrap_or_default(),
        Some(Kind::Path(p)) => p.clone(),
        _ => String::new(),
    }
}

#[tonic::async_trait]
impl AccessPolicyService for Service {
    async fn check(
        &self,
        request: tonic::Request<proto::CheckRequest>,
    ) -> Result<tonic::Response<proto::CheckResponse>, tonic::Status> {
        match self.mode {
            Mode::Slow => tokio::time::sleep(Duration::from_secs(2)).await,
            Mode::Canary => return Err(tonic::Status::permission_denied(CANARY)),
            Mode::Rule => {}
        }
        let message = request.into_inner();
        self.statements
            .lock()
            .unwrap()
            .push((message.statement_class.clone(), message.query_id.clone()));
        let decisions = message
            .items
            .iter()
            .map(|i| {
                let name = name_of(i);
                self.items.lock().unwrap().push(name.clone());
                if name.starts_with("secret") || name.contains("/secret") {
                    proto::Decision::Deny.into()
                } else {
                    proto::Decision::Allow.into()
                }
            })
            .collect();
        Ok(tonic::Response::new(proto::CheckResponse {
            decisions,
            policy_version: 9,
        }))
    }

    async fn version(
        &self,
        _: tonic::Request<proto::VersionRequest>,
    ) -> Result<tonic::Response<proto::VersionResponse>, tonic::Status> {
        if self.mode == Mode::Canary {
            return Err(tonic::Status::permission_denied(CANARY));
        }
        Ok(tonic::Response::new(proto::VersionResponse {
            policy_version: 9,
        }))
    }
}

struct World {
    rt: Arc<EngineRuntime>,
    dir: tempfile::TempDir,
    tokio: tokio::runtime::Runtime,
}

impl World {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let rt = EngineRuntime::builder()
            .config(
                "spark.sql.warehouse.dir",
                dir.path().join("wh").to_str().unwrap(),
            )
            .build()
            .unwrap();
        let tokio = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let w = Self { rt, dir, tokio };
        let admin = w.session_with(Arc::new(AllowAll));
        for sql in [
            "CREATE TABLE t (a INT, b INT) USING parquet",
            "INSERT INTO t VALUES (1, 2)",
            "CREATE TABLE secret_t (a INT) USING parquet",
            "INSERT INTO secret_t VALUES (3)",
            "CREATE VIEW secret_v AS SELECT a FROM t",
            "CREATE VIEW v AS SELECT a FROM t",
        ] {
            w.run(&admin, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        }
        w
    }

    fn session_with(&self, policy: Arc<dyn AccessPolicy>) -> Session {
        Session::builder()
            .runtime(Arc::clone(&self.rt))
            .principal(Principal::new("ana", PrincipalKind::User))
            .access_policy(policy)
            .build()
            .unwrap()
    }

    /// A service in `mode` and ana's session under the server's composition for `role`; with
    /// `roles_decide_data` the role matrix decides data securables too (`roles+external`).
    fn ana(
        &self,
        mode: Mode,
        role: Role,
        roles_decide_data: bool,
    ) -> (Session, Arc<Mutex<Vec<String>>>) {
        let (session, items, _) = self.ana_seen(mode, role, roles_decide_data);
        (session, items)
    }

    /// [`Self::ana`], and the statements the service was told of.
    #[expect(clippy::type_complexity, reason = "a test helper's tuple")]
    fn ana_seen(
        &self,
        mode: Mode,
        role: Role,
        roles_decide_data: bool,
    ) -> (
        Session,
        Arc<Mutex<Vec<String>>>,
        Arc<Mutex<Vec<(String, String)>>>,
    ) {
        let items = Arc::new(Mutex::new(Vec::new()));
        let statements = Arc::new(Mutex::new(Vec::new()));
        let service = Service {
            mode,
            items: Arc::clone(&items),
            statements: Arc::clone(&statements),
        };
        let addr = self.tokio.block_on(async {
            let incoming =
                tonic::transport::server::TcpIncoming::bind("127.0.0.1:0".parse().unwrap())
                    .unwrap();
            let addr = incoming.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = tonic::transport::Server::builder()
                    .add_service(AccessPolicyServer::new(service))
                    .serve_with_incoming(incoming)
                    .await;
            });
            addr
        });
        let access = {
            let _guard = self.tokio.enter();
            let channel = Endpoint::parse(&format!("http://{addr}"))
                .unwrap()
                .channel()
                .unwrap();
            ExternalAccess::over(
                AccessPolicyClient::new(channel),
                Duration::from_millis(300),
                None,
                false,
            )
        };
        // Server mode confines local paths to the configured roots: the warehouse here.
        let mut config = Config::new();
        config
            .set(
                "krishiv.io.local.allowedRoots",
                self.dir.path().canonicalize().unwrap().to_str().unwrap(),
            )
            .unwrap();
        // The factory `krishiv server` builds (`krishiv-cli` `caller_sessions`).
        let extra = ExtraProvider {
            policy: Arc::new(move || Arc::new(access.policy()) as Arc<dyn AccessPolicy>),
            roles_decide_data,
        };
        let base = self.session_with(Arc::new(AllowAll));
        let factory = provider_sessions(
            base,
            &config,
            Arc::new(DeniedCounts::default()),
            Some(extra),
        )
        .unwrap();
        let caller = Authenticated::new(
            Principal::new("ana", PrincipalKind::User),
            Some(role),
            Method::Token,
        );
        (
            factory(&caller, Transport::Connect).unwrap(),
            items,
            statements,
        )
    }

    fn run(&self, s: &Session, sql: &str) -> Result<usize, KrishivError> {
        self.tokio.block_on(async {
            let batches = s.sql(sql).await?.collect().await?;
            Ok(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
        })
    }

    /// The `(a, b)` rows of `sql`, whose two columns are INT.
    fn pairs(&self, s: &Session, sql: &str) -> Vec<(i32, i32)> {
        use arrow_array::Array as _;
        use arrow_array::cast::AsArray as _;
        use arrow_array::types::Int32Type;
        let batches = self
            .tokio
            .block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let mut out = Vec::new();
        for b in &batches {
            let a = b.column(0).as_primitive::<Int32Type>();
            let c = b.column(1).as_primitive::<Int32Type>();
            assert_eq!(a.null_count() + c.null_count(), 0, "{sql}");
            out.extend(a.values().iter().copied().zip(c.values().iter().copied()));
        }
        out
    }
}

fn refused(r: Result<usize, KrishivError>, sql: &str) {
    match r {
        Err(e) => {
            assert_eq!(e.error_class().name(), DENIED, "{sql}: {e}");
            assert!(!e.to_string().contains(CANARY), "{sql}: {e}");
            assert!(
                !e.to_string().contains("external policy service"),
                "{sql}: {e}"
            );
        }
        Ok(n) => panic!("{sql}: expected a denial, got {n} rows"),
    }
}

#[test]
fn the_service_decides_reads_writes_and_views() {
    let w = World::new();
    let (ana, items) = w.ana(Mode::Rule, Role::Operator, false);
    assert_eq!(w.run(&ana, "SELECT a, b FROM t").unwrap(), 1);
    assert_eq!(w.run(&ana, "SELECT * FROM v").unwrap(), 1);
    for sql in [
        "SELECT * FROM secret_t",
        "SELECT a FROM t WHERE a IN (SELECT a FROM secret_t)",
        "SELECT * FROM t UNION ALL SELECT * FROM secret_t",
        "WITH c AS (SELECT * FROM secret_t) SELECT * FROM c",
        "SELECT * FROM secret_v",
        "INSERT INTO secret_t VALUES (4)",
        "DROP TABLE secret_t",
        "DESCRIBE secret_t",
        "SELECT a FROM secret_t",
        "SELECT t.a FROM t JOIN secret_t s ON t.a = s.a",
        "SELECT (SELECT max(a) FROM secret_t) FROM t",
        "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM secret_t)",
        "CREATE TABLE copied AS SELECT * FROM secret_t",
        "SHOW CREATE TABLE secret_t",
        "CACHE TABLE secret_t",
        "ALTER TABLE secret_t ADD COLUMNS (c INT)",
        "TRUNCATE TABLE secret_t",
    ] {
        refused(w.run(&ana, sql), sql);
    }
    // A read of the table's files by path is the path's decision, not the table's.
    let files = w
        .dir
        .path()
        .canonicalize()
        .unwrap()
        .join("wh")
        .join("secret_t");
    let by_path = format!("SELECT * FROM parquet.`{}`", files.display());
    refused(w.run(&ana, &by_path), &by_path);
    assert!(!items.lock().unwrap().is_empty(), "the service was asked");
    // With `external` alone the role matrix keeps only `System(..)`: an operator may write where
    // the service allows it.
    w.run(&ana, "INSERT INTO t VALUES (5, 6)").unwrap();
    assert_eq!(
        w.pairs(&ana, "SELECT a, b FROM t ORDER BY a"),
        vec![(1, 2), (5, 6)]
    );
}

#[test]
fn roles_and_external_both_decide_data() {
    let w = World::new();
    // A viewer may not read rows (`Select`) under the role matrix, whatever the service says.
    let (viewer, _) = w.ana(Mode::Rule, Role::Viewer, true);
    refused(w.run(&viewer, "SELECT * FROM t"), "viewer read");
    // An operator reads what both allow, and nothing the service denies.
    let (operator, _) = w.ana(Mode::Rule, Role::Operator, true);
    assert_eq!(w.run(&operator, "SELECT * FROM t").unwrap(), 1);
    refused(w.run(&operator, "SELECT * FROM secret_t"), "secret read");
    // Writes stay the role matrix's: an operator may not.
    refused(
        w.run(&operator, "INSERT INTO t VALUES (7, 8)"),
        "operator write",
    );
}

#[test]
fn a_service_error_refuses_without_its_message() {
    let w = World::new();
    let (ana, _) = w.ana(Mode::Canary, Role::Admin, false);
    // `refused` checks the class is exactly the public denial and that `CANARY` is absent.
    for sql in [
        "SELECT * FROM t",
        "SELECT * FROM v",
        "INSERT INTO t VALUES (9, 9)",
    ] {
        refused(w.run(&ana, sql), sql);
    }
}

#[test]
fn a_slow_or_absent_service_refuses_everything() {
    let w = World::new();
    let (ana, _) = w.ana(Mode::Slow, Role::Admin, false);
    for sql in [
        "SELECT * FROM t",
        "SELECT * FROM v",
        "INSERT INTO t VALUES (9, 9)",
    ] {
        refused(w.run(&ana, sql), sql);
    }
}

/// ADR 0049 §3, queue row 75: each check carries the statement's class and, with the caller's
/// operation id, its query id; checks of the frame's action and of the queries a command runs
/// carry them too.
#[test]
fn a_check_names_the_statement_it_is_asked_for() {
    let w = World::new();
    let (ana, _, seen) = w.ana_seen(Mode::Rule, Role::Operator, false);
    let run = |sql: &str, op: &str| {
        let options = krishiv_session::QueryOptions::new().operation_id(op);
        let r = w
            .tokio
            .block_on(async { ana.sql_with(sql, options).await?.collect().await });
        let checks = std::mem::take(&mut *seen.lock().unwrap());
        assert!(!checks.is_empty(), "{sql}: the service was asked");
        (r, checks)
    };
    let op = "6f1d2c1e-8b8a-4c55-9a8e-2f6a3c1b7d90";
    let (read, checks) = run("SELECT a, b FROM t", op);
    assert_eq!(read.unwrap().len(), 1);
    assert!(
        checks.iter().all(|(c, id)| c == "READ" && id == op),
        "{checks:?}"
    );
    let op = "0b7e4f3a-2d51-4e6b-8f0c-9a1d3e5b7c22";
    let (write, checks) = run("INSERT INTO t VALUES (5, 6)", op);
    write.unwrap();
    assert!(
        checks.iter().all(|(c, id)| c == "WRITE" && id == op),
        "{checks:?}"
    );
    let op = "4c2a9e1b-7f3d-4a8c-b6e5-1d0f2a3b4c5d";
    let (drop, checks) = run("DROP TABLE secret_t", op);
    refused(drop.map(|b| b.len()), "drop");
    assert!(
        checks.iter().all(|(c, id)| c == "DDL" && id == op),
        "{checks:?}"
    );
    // Without an operation id the class is still sent; the id only once the query's row exists.
    w.run(&ana, "SELECT a FROM t").unwrap();
    let checks = std::mem::take(&mut *seen.lock().unwrap());
    assert!(
        !checks.is_empty() && checks.iter().all(|(c, _)| c == "READ"),
        "{checks:?}"
    );
}
