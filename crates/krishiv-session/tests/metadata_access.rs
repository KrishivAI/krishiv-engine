// SPDX-License-Identifier: Apache-2.0
//! Metadata analysis (ADR 0046 §3b, P3 PR-14c): `DataFrame::schema_for_metadata` asks `Browse` on
//! every table, view and path a plan reads, where `schema`, actions and `explain` ask `Select`. A
//! viewer-like policy (`Browse` allowed, `Select` denied) sees the schema of a table, a temporary
//! view, a permanent view and files inside the allowed roots, never outside them, and never reads
//! rows: no metadata analysis is reused by a later action of the same frame or session.
#![cfg(test)]

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, DataFrame, ExplainMode, PathPrefixPolicy, Principal,
    PrincipalKind, Privilege, Securable, Session,
};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

const ADMIN: u8 = 0;
const VIEWER: u8 = 1;
const OPERATOR: u8 = 2;

/// Admin (everything), viewer (`Browse` only) or operator (`Browse` and `Select`); a role change
/// bumps the version, as a policy's must. Records every question.
struct Roles {
    role: AtomicU8,
    version: AtomicU64,
    asked: Mutex<Vec<(String, Privilege)>>,
}

impl Roles {
    fn new(role: u8) -> Arc<Self> {
        Arc::new(Self {
            role: AtomicU8::new(role),
            version: AtomicU64::new(1),
            asked: Mutex::new(Vec::new()),
        })
    }

    fn become_(&self, role: u8) {
        self.role.store(role, Ordering::SeqCst);
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    fn take(&self) -> Vec<(String, Privilege)> {
        std::mem::take(&mut *self.asked.lock().unwrap())
    }
}

impl AccessPolicy for Roles {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(self.version.load(Ordering::SeqCst))
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        self.asked
            .lock()
            .unwrap()
            .push((securable.to_string(), privilege));
        let allowed = match self.role.load(Ordering::SeqCst) {
            ADMIN => true,
            VIEWER => privilege == Privilege::Browse,
            _ => matches!(privilege, Privilege::Browse | Privilege::Select),
        };
        Box::pin(std::future::ready(Ok(if allowed {
            AccessDecision::Allow
        } else {
            AccessDecision::Deny {
                reason: "the reason stays private".to_owned(),
            }
        })))
    }
}

/// Registers the table `t` (column `a`, three rows) in `s`.
fn register_t(s: &Session) {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
}

fn class<T>(r: Result<T, KrishivError>) -> String {
    match r {
        Ok(_) => "OK".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    }
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// The fixture, made by an unconfined admin session of the same runtime: files of `t` inside the
/// root (`root/data`) and outside it; a permanent table `p` and a permanent view `pv` over it. The
/// session under test (`vic`, confined to `root`, under `Roles`) has `t` and a temporary view `tv`.
struct Fixture {
    _tmp: tempfile::TempDir,
    inside: String,
    outside: String,
}

fn setup(role: u8) -> (Arc<Roles>, Session, Fixture) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let inside = root.join("data").to_str().unwrap().to_owned();
    let outside = tmp.path().join("outside/data").to_str().unwrap().to_owned();
    let admin = Session::builder()
        .access_policy(Arc::new(AllowAll))
        .build()
        .unwrap();
    admin
        .conf()
        .set(
            "spark.sql.warehouse.dir",
            tmp.path().join("warehouse").to_str().unwrap(),
        )
        .unwrap();
    register_t(&admin);
    for dir in [&inside, &outside] {
        block_on(admin.table("t").write().parquet(dir)).unwrap();
    }
    run(&admin, "CREATE TABLE p (a INT) USING parquet");
    run(&admin, "INSERT INTO p VALUES (1)");
    run(&admin, "CREATE VIEW pv AS SELECT a FROM p");

    let policy = Roles::new(ADMIN);
    let confined =
        PathPrefixPolicy::new([&root], Arc::clone(&policy) as Arc<dyn AccessPolicy>).unwrap();
    let s = Session::builder()
        .runtime(Arc::clone(admin.runtime()))
        .principal(Principal::new("vic", PrincipalKind::User))
        .access_policy(Arc::new(confined))
        .build()
        .unwrap();
    register_t(&s);
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT a FROM t");
    policy.become_(role);
    policy.take();
    (
        policy,
        s,
        Fixture {
            _tmp: tmp,
            inside,
            outside,
        },
    )
}

fn frames(s: &Session, f: &Fixture) -> Vec<(&'static str, DataFrame)> {
    vec![
        ("table", s.table("t")),
        ("temporary view", s.table("tv")),
        ("permanent view", s.table("pv")),
        ("permanent table", s.table("p")),
        ("path inside the roots", s.read().parquet([&f.inside])),
    ]
}

#[test]
fn a_viewer_sees_schemas_and_never_rows() {
    let (policy, s, f) = setup(VIEWER);
    for (name, df) in frames(&s, &f) {
        policy.take();
        let schema = block_on(df.schema_for_metadata())
            .unwrap_or_else(|e| panic!("{name}: schema_for_metadata: {e}"));
        assert_eq!(schema.field(0).name(), "a", "{name}");
        let asked = policy.take();
        assert!(
            !asked.is_empty() && asked.iter().all(|(_, p)| *p == Privilege::Browse),
            "{name}: {asked:?}"
        );
        // The same frame afterwards: nothing of the metadata analysis is reused.
        assert_eq!(class(block_on(df.schema())), DENIED, "{name}: schema");
        assert_eq!(class(block_on(df.collect())), DENIED, "{name}: collect");
        assert_eq!(class(block_on(df.count())), DENIED, "{name}: count");
        // `explain` plans the query (distinct-value samples read rows): `Select`.
        assert_eq!(
            class(block_on(df.explain(ExplainMode::Simple))),
            DENIED,
            "{name}: explain"
        );
        // And again: a metadata analysis after the denials still works.
        assert_eq!(class(block_on(df.schema_for_metadata())), "OK", "{name}");
    }
    // The permanent view is asked as a view too, with `Browse`.
    policy.take();
    block_on(s.table("pv").schema_for_metadata()).unwrap();
    let asked = policy.take();
    assert!(
        asked
            .iter()
            .any(|(s, p)| s.starts_with("view ") && s.contains("pv") && *p == Privilege::Browse),
        "{asked:?}"
    );
}

#[test]
fn paths_outside_the_roots_are_refused_for_metadata_too() {
    let (_, s, f) = setup(VIEWER);
    let outside = s.read().parquet([&f.outside]);
    assert_eq!(class(block_on(outside.schema_for_metadata())), DENIED);
    let escaped = format!("{}/../../outside/data", f.inside);
    let escaped = s.read().parquet([&escaped]);
    assert_eq!(class(block_on(escaped.schema_for_metadata())), DENIED);
}

#[test]
fn an_operator_run_after_a_metadata_analysis_is_checked_with_select() {
    let (policy, s, f) = setup(OPERATOR);
    for (name, df) in frames(&s, &f) {
        policy.take();
        block_on(df.schema_for_metadata()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            policy.take().iter().all(|(_, p)| *p == Privilege::Browse),
            "{name}"
        );
        let rows: usize = block_on(df.collect())
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        assert!(rows > 0, "{name}");
        let asked = policy.take();
        assert!(
            asked.iter().any(|(_, p)| *p == Privilege::Select),
            "{name}: the run asked {asked:?}"
        );
    }
    // A viewer's metadata analysis of the same text, then a run by the operator: asked again.
    policy.become_(VIEWER);
    block_on(s.table("t").select(["a"]).schema_for_metadata()).unwrap();
    policy.become_(OPERATOR);
    policy.take();
    block_on(s.table("t").select(["a"]).collect()).unwrap();
    assert!(
        policy
            .take()
            .iter()
            .any(|(s, p)| s.contains("`t`") && *p == Privilege::Select)
    );
}
