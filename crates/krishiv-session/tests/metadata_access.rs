// SPDX-License-Identifier: Apache-2.0
//! Metadata analysis (ADR 0046 §3b, P3 PR-14c): `DataFrame::schema_for_metadata` asks `Browse` on
//! every catalog table and view a plan reads, where `schema`, actions and `explain` ask `Select`. A
//! viewer-like policy (`Browse` allowed, `Select` denied) sees the stored schema of a table, a
//! temporary view, a permanent view and a permanent table, and never reads rows: no metadata
//! analysis is reused by a later action of the same frame or session. A read by path keeps
//! `Select` even for metadata (inference reads file contents: CSV `header=true` names are the
//! file's first line), and `PathPrefixPolicy` refuses paths outside its roots for every role; a
//! permanent view over a path needs `Select` on the path. Reads inside a CTE, a subquery or a SQL
//! function body ask `Browse` too, and nothing of the plan runs.
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

fn frames(s: &Session) -> Vec<(&'static str, DataFrame)> {
    vec![
        ("table", s.table("t")),
        ("temporary view", s.table("tv")),
        ("permanent view", s.table("pv")),
        ("permanent table", s.table("p")),
    ]
}

#[test]
fn a_viewer_sees_schemas_and_never_rows() {
    let (policy, s, _fixture) = setup(VIEWER);
    for (name, df) in frames(&s) {
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
fn a_read_by_path_needs_select_for_metadata_too() {
    let escaped = |f: &Fixture| format!("{}/../../outside/data", f.inside);
    // Viewer: inside the roots too (asked as `Select` on the path), and outside them.
    let (policy, s, f) = setup(VIEWER);
    let inside = s.read().parquet([&f.inside]);
    assert_eq!(class(block_on(inside.schema_for_metadata())), DENIED);
    let asked = policy.take();
    assert!(
        asked
            .iter()
            .any(|(s, p)| s.starts_with("path ") && *p == Privilege::Select),
        "{asked:?}"
    );
    assert!(
        asked.iter().all(|(_, p)| *p == Privilege::Select),
        "{asked:?}"
    );
    for path in [f.outside.clone(), escaped(&f)] {
        let df = s.read().parquet([&path]);
        assert_eq!(class(block_on(df.schema_for_metadata())), DENIED, "{path}");
    }
    // Operator: inside the roots only.
    let (_, s, f) = setup(OPERATOR);
    let inside = block_on(s.read().parquet([&f.inside]).schema_for_metadata()).unwrap();
    assert_eq!(inside.field(0).name(), "a");
    for path in [f.outside.clone(), escaped(&f)] {
        let df = s.read().parquet([&path]);
        assert_eq!(class(block_on(df.schema_for_metadata())), DENIED, "{path}");
    }
}

#[test]
fn a_viewer_cannot_read_a_csv_header_through_its_schema() {
    // Why a path keeps `Select` for metadata: with `header=true` the column names a schema shows are
    // the file's first line, so `Browse` on a path would show a viewer file contents.
    let csv =
        |s: &Session, path: &str| s.read().format("csv").option("header", "true").load([path]);
    for (role, expected) in [(VIEWER, DENIED), (OPERATOR, "OK")] {
        let (_, s, f) = setup(role);
        let dir = std::path::Path::new(&f.inside).with_file_name("secrets");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("s.csv"), "first_line_secret,b\n1,2\n").unwrap();
        let path = dir.to_str().unwrap().to_owned();
        let r = block_on(csv(&s, &path).schema_for_metadata());
        if role == OPERATOR {
            // What a viewer would have seen: the first line, as column names.
            let schema = r.as_ref().unwrap();
            assert_eq!(schema.field(0).name(), "first_line_secret");
        } else if let Err(e) = &r {
            assert!(!e.to_string().contains("first_line_secret"), "{e}");
        }
        assert_eq!(class(r), expected);
    }
}

#[test]
fn an_operator_run_after_a_metadata_analysis_is_checked_with_select() {
    let (policy, s, _fixture) = setup(OPERATOR);
    for (name, df) in frames(&s) {
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
}

#[test]
fn a_run_after_a_metadata_analysis_of_the_same_plan_asks_select() {
    // One policy version throughout (no role change): the same plan text, built again, first for
    // metadata and then run. The guard is that Bind asks the policy on every analysis (and that the
    // prefix cache keys on the mode, tested in krishiv-plan's `metadata_mode`), not a version bump.
    let (policy, s, _fixture) = setup(OPERATOR);
    let version = policy.version();
    for _ in 0..2 {
        policy.take();
        block_on(s.table("t").select(["a"]).schema_for_metadata()).unwrap();
        let asked = policy.take();
        assert!(
            asked
                .iter()
                .any(|(s, p)| s.contains("`t`") && *p == Privilege::Browse),
            "{asked:?}"
        );
        assert!(
            asked.iter().all(|(_, p)| *p == Privilege::Browse),
            "{asked:?}"
        );
        block_on(s.table("t").select(["a"]).collect()).unwrap();
        let asked = policy.take();
        assert!(
            asked
                .iter()
                .any(|(s, p)| s.contains("`t`") && *p == Privilege::Select),
            "{asked:?}"
        );
        assert!(
            asked.iter().all(|(_, p)| *p == Privilege::Select),
            "{asked:?}"
        );
    }
    assert_eq!(policy.version(), version);
}

#[test]
fn a_permanent_view_over_a_path_needs_select_on_the_path() {
    // The key case for views over paths: `Browse` on the view does not show a viewer what inference
    // reads from the files (the body's path read keeps `Select`).
    for (role, expected) in [(VIEWER, DENIED), (OPERATOR, "OK")] {
        let (policy, s, f) = setup(role);
        let admin = Session::builder()
            .runtime(Arc::clone(s.runtime()))
            .access_policy(Arc::new(AllowAll))
            .build()
            .unwrap();
        run(
            &admin,
            &format!("CREATE VIEW pathv AS SELECT a FROM parquet.`{}`", f.inside),
        );
        policy.take();
        let r = block_on(s.table("pathv").schema_for_metadata());
        let asked = policy.take();
        assert!(
            asked
                .iter()
                .any(|(s, p)| s.starts_with("path ") && *p == Privilege::Select),
            "{role}: {asked:?}"
        );
        assert_eq!(class(r), expected, "{role}");
    }
}

#[test]
fn a_cte_a_subquery_and_a_function_body_ask_browse_and_run_nothing() {
    let (policy, s, _fixture) = setup(ADMIN);
    // A body that reads the permanent table `p`; created while the role may.
    run(
        &s,
        "CREATE TEMPORARY FUNCTION top_a() RETURNS INT RETURN (SELECT max(a) FROM p)",
    );
    policy.become_(VIEWER);
    // Each reads `p` only inside the CTE, the subquery or the function body; `t` is the frame.
    // Built lazily (`Session::sql` analyzes its text at once, with `Select`). Running any of these
    // fails: `raise_error` on every row, and `Select` is denied anyway.
    let boom = "raise_error('executed') AS boom";
    let t = || s.table("t");
    let frames = [
        (
            "cte",
            t().select_expr([
                "(WITH c AS (SELECT a FROM p) SELECT max(a) FROM c) AS a",
                boom,
            ]),
        ),
        (
            "subquery",
            t().filter_expr("a IN (SELECT a FROM p)")
                .and_then(|df| df.select_expr(["a", boom])),
        ),
        ("function body", t().select_expr(["top_a() AS a", boom])),
    ];
    for (text, df) in frames {
        let df = df.unwrap_or_else(|e| panic!("{text}: {e}"));
        policy.take();
        let schema = block_on(df.schema_for_metadata()).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(schema.field(0).name(), "a", "{text}");
        let asked = policy.take();
        assert!(
            asked
                .iter()
                .any(|(s, p)| s.contains("`p`") && *p == Privilege::Browse),
            "{text}: the read inside asks `Browse`: {asked:?}"
        );
        assert!(
            asked.iter().all(|(_, p)| *p == Privilege::Browse),
            "{text}: {asked:?}"
        );
        // A run of the same frame is still asked `Select`, and denied.
        assert_eq!(class(block_on(df.collect())), DENIED, "{text}");
    }
}
