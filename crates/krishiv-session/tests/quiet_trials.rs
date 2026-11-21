// SPDX-License-Identifier: Apache-2.0
//! T063 review R2-1: the analyze API's quiet trials of a run's checks (the target trial before
//! listing directories, the `BROWSE` trial of `DROP NAMESPACE ... CASCADE`) leave no trace: no
//! `krishiv::audit` record, no session denial count, no provider denial count
//! (`Grants::denied_counts`, the `krishiv_access_denied_total` counters) and no
//! `AccessPolicy::denied` call; nor does plain `EXPLAIN`'s trial analysis of a read (P3 PR-14d). A run
//! of the same statement still counts and audits its refusal. The `Usage` check on `System("udf")` an
//! analysis makes before a Python UDTF's `analyze` (PR-26c-4 S-2) is quiet in a trial too.
//! One test per binary: the subscriber is process-wide.
#![cfg(test)]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::auth::{DeniedCounts, Surface};
use krishiv_session::grants::{DeniedCount, Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, BoxFuture, EngineRuntime, PolicyVersion, Principal,
    PrincipalKind, Privilege, Securable, Session,
};
use tracing::field::{Field as TraceField, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Keeps the `krishiv::audit` events as `field=value` lines.
struct Capture;

struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &TraceField, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if event.metadata().target() == "krishiv::audit" {
            let mut fields = Fields(String::new());
            event.record(&mut fields);
            RECORDS.lock().unwrap().push(fields.0);
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

fn take() -> Vec<String> {
    std::mem::take(&mut *RECORDS.lock().unwrap())
}

fn run(s: &Session, sql: &str) -> Result<usize, KrishivError> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        Ok(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
    })
}

fn total(counts: &[DeniedCount]) -> u64 {
    counts.iter().map(|c| c.count).sum()
}

/// The provider's own counters (`Grants::denied_counts`), the session's, and the audit records,
/// after `analyze` and `EXPLAIN FORMAT JSON` of `sql`: all must stay still.
fn total_session(counts: &DeniedCounts) -> u64 {
    counts.counts().iter().map(|c| c.count).sum()
}

fn assert_trials_quiet(s: &Session, session: &DeniedCounts, provider: &dyn Fn() -> u64, sql: &str) {
    let counted = || total_session(session);
    let before_session = counted();
    take();
    let before = provider();
    block_on(s.analyze(sql)).unwrap_or_else(|e| panic!("analyze {sql}: {e}"));
    run(s, &format!("EXPLAIN FORMAT JSON {sql}")).unwrap_or_else(|e| panic!("explain {sql}: {e}"));
    assert_eq!(
        provider(),
        before,
        "{sql}: a trial counted a provider denial"
    );
    assert!(
        counted() == before_session,
        "{sql}: a trial counted a session denial"
    );
    assert_eq!(
        take(),
        Vec::<String>::new(),
        "{sql}: a trial wrote an audit record"
    );
}

/// Allows everything but `BROWSE` on a namespace (and `SELECT`, its fallback), `USAGE` on a secret
/// scope and `MANAGE` on the cluster, and records every refusal it is told of.
#[derive(Default)]
struct NoBrowseRecording(Mutex<Vec<String>>);

impl AccessPolicy for NoBrowseRecording {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let d = if matches!(securable, Securable::Namespace(_))
            && matches!(privilege, Privilege::Browse | Privilege::Select)
            || matches!(securable, Securable::Secret { .. }) && privilege == Privilege::Usage
            || matches!(securable, Securable::System(_)) && privilege == Privilege::Manage
        {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }

    fn denied(&self, _: &Principal, securable: &Securable, privilege: Privilege) {
        self.0
            .lock()
            .unwrap()
            .push(format!("{privilege} {securable}"));
    }
}

#[test]
fn quiet_trials_leave_no_trace() {
    let _ = tracing::subscriber::set_global_default(Capture);

    // The built-in grants provider: `ana` holds nothing on the admin's table and namespace.
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config(
            "spark.sql.warehouse.dir",
            dir.path().join("wh").to_str().unwrap(),
        )
        .build()
        .unwrap();
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let session_of = |name: &str, admin: bool| {
        let policy = Arc::new(GrantsPolicy::new(
            Arc::clone(&grants),
            Arc::new(AllowAll),
            admin,
        ));
        Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(Principal::new(name, PrincipalKind::User))
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .grants(policy)
            .build()
            .unwrap()
    };
    let root = session_of("root", true);
    for sql in [
        "CREATE NAMESPACE d",
        "CREATE TABLE d.t (a INT) USING parquet",
        "INSERT INTO d.t VALUES (1)",
    ] {
        run(&root, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    let ana = session_of("ana", false);
    let ana_counts = Arc::new(DeniedCounts::new());
    assert!(ana.count_denials(Arc::clone(&ana_counts), Surface::Connect));
    let provider = || total(&grants.denied_counts());
    for sql in [
        "INSERT INTO d.t VALUES (2)",
        "DROP TABLE d.t",
        "DROP NAMESPACE d CASCADE",
    ] {
        assert_trials_quiet(&ana, &ana_counts, &provider, sql);
        // The run of the same statement is refused, counted once and audited once.
        let before = provider();
        let e = run(&ana, sql).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{sql}"
        );
        assert_eq!(provider(), before + 1, "{sql}: the run counts one denial");
        assert_eq!(take().len(), 1, "{sql}: the run writes one audit record");
    }

    // Plain EXPLAIN (P3 PR-14d) first tries the ordinary analysis, whose `SELECT` denial is an
    // answer: with `BROWSE` only, it shows the withheld plan, and the provider counts nothing.
    run(&root, "GRANT BROWSE ON TABLE d.t TO user:ana").unwrap();
    for sql in [
        "EXPLAIN SELECT * FROM d.t",
        "EXPLAIN SELECT a FROM d.t WHERE a = 1",
    ] {
        take();
        let (before, before_session) = (provider(), total_session(&ana_counts));
        assert_eq!(run(&ana, sql).unwrap(), 1, "{sql}");
        assert_eq!(
            provider(),
            before,
            "{sql}: the trial counted a provider denial"
        );
        assert_eq!(total_session(&ana_counts), before_session, "{sql}");
        assert_eq!(
            take(),
            Vec::<String>::new(),
            "{sql}: the trial wrote an audit record"
        );
    }
    // The run of the read is refused, counted once and audited once.
    let before = provider();
    run(&ana, "SELECT * FROM d.t").unwrap_err();
    assert_eq!(provider(), before + 1, "the read counts one denial");
    assert_eq!(take().len(), 1, "the read writes one audit record");

    // #868 review 9fcd13d57 F1: the stored-function calls of a definer view's text are asked of
    // its definer through the analysis's policy, so plain EXPLAIN's trial of a read whose definer
    // lost a function tells the provider nothing: every denial the provider counts is one the
    // audited fallback reports (one audit record each), none is the trial's.
    let carl = session_of("carl", false);
    for sql in [
        "GRANT CREATE ON SCHEMA d TO user:carl",
        "GRANT SELECT ON TABLE d.t TO user:carl",
        "CREATE FUNCTION d.pos(x INT) RETURNS BOOLEAN RETURN x > 0",
        "GRANT SELECT ON FUNCTION d.pos TO user:carl",
    ] {
        run(&root, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    for sql in [
        "CREATE VIEW d.dvf SQL SECURITY DEFINER AS SELECT a FROM d.t WHERE d.pos(a)",
        "GRANT SELECT ON VIEW d.dvf TO user:ana",
    ] {
        run(&carl, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    assert_eq!(run(&ana, "SELECT a FROM d.dvf").unwrap(), 1);
    run(&root, "REVOKE SELECT ON FUNCTION d.pos FROM user:carl").unwrap();
    take();
    let before = provider();
    let explained = run(&ana, "EXPLAIN SELECT a FROM d.dvf");
    let audits = take().len() as u64;
    assert_eq!(
        provider() - before,
        audits,
        "a trial counted a denial no audit record reports ({explained:?})"
    );
    // The read itself is the definer's refusal: counted once, audited once.
    let before = provider();
    run(&ana, "SELECT a FROM d.dvf").unwrap_err();
    assert_eq!(provider(), before + 1, "the definer's refusal counts once");
    assert_eq!(take().len(), 1, "the definer's refusal is audited once");

    // A custom policy that may drop but not browse: the CASCADE's BROWSE trial tells it nothing.
    let warehouse = tempfile::tempdir().unwrap();
    let policy = Arc::new(NoBrowseRecording::default());
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .config(
            "spark.sql.warehouse.dir",
            warehouse.path().to_str().unwrap(),
        )
        .build()
        .unwrap();
    let s_counts = Arc::new(DeniedCounts::new());
    assert!(s.count_denials(Arc::clone(&s_counts), Surface::Connect));
    run(&s, "CREATE NAMESPACE d").unwrap();
    run(&s, "CREATE TABLE d.a1 (a INT)").unwrap();
    let told = || policy.0.lock().unwrap().len() as u64;
    assert_trials_quiet(&s, &s_counts, &told, "DROP NAMESPACE d CASCADE");
    assert!(policy.0.lock().unwrap().is_empty());
    // The policy is told of a run's refusal.
    run(&s, "SHOW TABLES IN d").unwrap_err();
    assert_eq!(told(), 1, "{:?}", policy.0.lock().unwrap());

    // One statement, one refusal report: a statement refused `MANAGE` on the cluster (a
    // server-scoped store option, or any query while the session changed a server-scoped store
    // setting, `check_store_options`) tells the policy once, writes one audit record and counts
    // one denial, whatever path it takes. Plain EXPLAIN's trial adds none (review Q3), and its
    // withheld fallback and its result frame do not report the one refusal twice. (A secret's
    // `USAGE` is reached only through the IR: `session/env.rs` tests.)
    let reported_once = |what: &str, r: Result<(), KrishivError>| {
        let e = r.unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{what}: {e}"
        );
        assert!(e.to_string().contains("MANAGE"), "{what}: {e}");
    };
    let once = |what: &str, act: &dyn Fn() -> Result<(), KrishivError>| {
        take();
        let (before, before_session) = (told(), total_session(&s_counts));
        reported_once(what, act());
        let audits = take().len();
        assert_eq!(
            (
                told() - before,
                total_session(&s_counts) - before_session,
                audits
            ),
            (1, 1, 1),
            "{what}: {:?}",
            policy.0.lock().unwrap()
        );
    };
    let s = &s;
    let sql = |q: &'static str| move || run(s, q).map(|_| ());
    // The store option is the statement's own.
    once("SET", &sql("SET fs.s3a.endpoint = 'http://10.0.0.1:9000'"));
    once(
        "CREATE TABLE OPTIONS",
        &sql("CREATE TABLE d.o (a INT) USING csv OPTIONS ('fs.s3a.region' = 'eu-west-1')"),
    );
    // Every query while the session changed a server-scoped store setting.
    s.conf()
        .set("fs.s3a.endpoint", "http://10.0.0.1:9000")
        .unwrap();
    for q in [
        "SELECT 1",
        "SELECT * FROM d.a1",
        "SET",
        "EXPLAIN SELECT 1",
        "EXPLAIN EXTENDED SELECT * FROM d.a1",
        "EXPLAIN FORMATTED SELECT 1",
        "EXPLAIN CODEGEN SELECT 1",
        "EXPLAIN COST SELECT 1",
        "EXPLAIN ANALYZE SELECT 1",
        "EXPLAIN VERBOSE SELECT 1",
        "EXPLAIN FORMAT JSON SELECT 1",
        "EXPLAIN SET",
    ] {
        once(q, &sql(q));
    }
    once("saveAsTable", &|| {
        block_on(s.range(0, 1, 1, None).write().save_as_table("d.w"))
    });
    // Connect `AnalyzePlan`: the schema and the explain of a frame.
    once("schema_for_metadata", &|| {
        block_on(s.range(0, 1, 1, None).schema_for_metadata()).map(|_| ())
    });
    once("explain_for_metadata", &|| {
        block_on(
            s.range(0, 1, 1, None)
                .explain_for_metadata(krishiv_session::ExplainMode::Simple),
        )
        .map(|_| ())
    });

    // PR-26c-4 S-2: a principal without `USAGE` on `System("udf")` calls a UDTF with `analyze`. The
    // `EXPLAIN` trial's check is quiet and no `analyze` runs.
    let policy = Arc::new(NoPythonRecording::default());
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .build()
        .unwrap();
    let s_counts = Arc::new(DeniedCounts::new());
    assert!(s.count_denials(Arc::clone(&s_counts), Surface::Connect));
    let analyzed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    s.set_python_udf_runtime(Arc::new(Analyzing(Arc::clone(&analyzed))));
    s.register_python_udtf(krishiv_session::PythonUdtfDef {
        name: "describe".to_owned(),
        eval_type: krishiv_plan::PythonEvalType::TableUdf,
        return_type: None,
        deterministic: true,
    })
    .unwrap();
    let told = || policy.0.lock().unwrap().len() as u64;
    // The `EXPLAIN` trial's check is quiet: the statement reports its refusal once, the withheld plan's
    // (which cannot be made without `analyze`), as for any refusal the fallback meets.
    take();
    let (before, before_session) = (told(), total_session(&s_counts));
    let shown = run(&s, "EXPLAIN SELECT * FROM describe(1)");
    let audits = take().len();
    assert_eq!(
        (
            told() - before,
            total_session(&s_counts) - before_session,
            audits
        ),
        (1, 1, 1),
        "EXPLAIN ({shown:?}): {:?}",
        policy.0.lock().unwrap()
    );
    // A run is refused, told, counted and audited once.
    let (before, before_session) = (told(), total_session(&s_counts));
    let e = run(&s, "SELECT * FROM describe(1)").unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "KRISHIV_INSUFFICIENT_PERMISSIONS",
        "{e}"
    );
    let audits = take().len();
    assert_eq!(
        (
            told() - before,
            total_session(&s_counts) - before_session,
            audits
        ),
        (1, 1, 1),
        "run: {:?}",
        policy.0.lock().unwrap()
    );
    assert_eq!(
        analyzed.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "an analyze ran"
    );
}

/// Denies `USAGE` on `System("udf")` (running client Python) and records every refusal it is told of.
#[derive(Default)]
struct NoPythonRecording(Mutex<Vec<String>>);

impl AccessPolicy for NoPythonRecording {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let d =
            if *securable == Securable::System("udf".to_owned()) && privilege == Privilege::Usage {
                AccessDecision::Deny {
                    reason: "no Python".to_owned(),
                }
            } else {
                AccessDecision::Allow
            };
        Box::pin(std::future::ready(Ok(d)))
    }

    fn denied(&self, _: &Principal, securable: &Securable, privilege: Privilege) {
        self.0
            .lock()
            .unwrap()
            .push(format!("{privilege} {securable}"));
    }
}

/// A Python UDF runtime whose `analyze` counts its calls (and answers one `INT` column).
struct Analyzing(Arc<std::sync::atomic::AtomicUsize>);

impl std::fmt::Debug for Analyzing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Analyzing")
    }
}

impl krishiv_exec::python_eval::UdfRuntime for Analyzing {
    fn prepare(
        &self,
        spec: &krishiv_exec::python_eval::UdfCallSpec,
    ) -> krishiv_exec::Result<Arc<dyn krishiv_exec::python_eval::PreparedUdf>> {
        Err(KrishivError::from(
            krishiv_common::error::KrishivUdfNotAvailable {
                name: spec.name.clone(),
                version: spec.version.to_string(),
            },
        ))
    }

    fn analyze_udtf(
        &self,
        _: &krishiv_exec::python_udtf::UdtfAnalyzeSpec,
    ) -> futures::future::BoxFuture<
        'static,
        krishiv_exec::Result<krishiv_exec::python_udtf::UdtfAnalysis>,
    > {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let schema = krishiv_types::DataType::Struct(krishiv_types::StructType::new([
            krishiv_types::StructField::new("c".to_owned(), krishiv_types::DataType::Integer, true),
        ]));
        Box::pin(std::future::ready(Ok(
            krishiv_exec::python_udtf::UdtfAnalysis {
                schema,
                pickled: Arc::from(b"p".to_vec()),
                with_single_partition: false,
                partition_by: Vec::new(),
                order_by: Vec::new(),
                select: Vec::new(),
            },
        )))
    }
}
