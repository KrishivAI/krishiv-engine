// SPDX-License-Identifier: Apache-2.0
//! Metadata analysis (ADR 0046 §3b, P3 PR-14c): `DataFrame::schema_for_metadata` asks `Browse` on
//! every catalog table and view a plan reads, where `schema`, actions and `explain` ask `Select`. A
//! viewer-like policy (`Browse` allowed, `Select` denied) sees the stored schema of a table, a
//! temporary view, a permanent view and a permanent table, and never reads rows: no metadata
//! analysis is reused by a later action of the same frame or session. A read by path keeps
//! `Select` even for metadata (inference reads file contents: CSV `header=true` names are the
//! file's first line), and `PathPrefixPolicy` refuses paths outside its roots for every role; a
//! permanent view over a path needs `Select` on the path. Reads inside a CTE, a subquery or a SQL
//! function body ask `Browse` too, and nothing of the plan runs. P3 PR-14d: `explain_for_metadata`
//! gives a viewer the plan with statistics, samples and the cache note withheld (the same plan
//! whatever the tables hold), an operator the ordinary text; `cost`, permanent views and reads
//! by path keep `Select`; SQL `EXPLAIN` the same, its refusals in the text.
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
/// `Browse` allowed; a `Select` question is a policy error.
const BROKEN_SELECT: u8 = 3;

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
        let role = self.role.load(Ordering::SeqCst);
        if role == BROKEN_SELECT && privilege == Privilege::Select {
            return Box::pin(std::future::ready(Err(KrishivError::from(
                krishiv_common::error::KrishivUnsupportedFeature {
                    feature: "a policy that fails".to_owned(),
                },
            ))));
        }
        let allowed = match role {
            ADMIN => true,
            VIEWER | BROKEN_SELECT => privilege == Privilege::Browse,
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
    /// The unconfined admin session that made the fixture (same runtime and catalog).
    admin: Session,
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
        .config(
            "spark.sql.warehouse.dir",
            tmp.path().join("warehouse").to_str().unwrap(),
        )
        .build()
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
            admin,
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

// P3 PR-14d: `explain_for_metadata`, the explain a viewer may see.

const NOTE: &str = "== Note ==\nStatistics withheld";

const MODES: [ExplainMode; 4] = [
    ExplainMode::Simple,
    ExplainMode::Extended,
    ExplainMode::Formatted,
    ExplainMode::Codegen,
];

/// What a withheld explain never prints: statistics, samples, the cache note, the join's sizes.
fn assert_withheld(text: &str, what: &str) {
    assert!(text.contains(NOTE), "{what}: {text}");
    for data in [
        "Statistics(",
        "Distinct-Value",
        "Result Cache",
        "builds left",
        "MiB",
        "rowCount",
    ] {
        assert!(!text.contains(data), "{what}: {data} in {text}");
    }
}

#[test]
fn a_viewer_explains_tables_with_the_data_withheld() {
    let (policy, s, _fixture) = setup(VIEWER);
    for (name, df) in [
        ("table", s.table("t")),
        ("temporary view", s.table("tv")),
        ("permanent table", s.table("p")),
    ] {
        for mode in MODES {
            policy.take();
            let text = block_on(df.explain_for_metadata(mode))
                .unwrap_or_else(|e| panic!("{name} {mode:?}: {e}"));
            assert_withheld(&text, &format!("{name} {mode:?}"));
            // The trial asks `Select` first; what the viewer is shown was checked with `Browse`.
            let asked = policy.take();
            assert!(
                asked.iter().any(|(_, p)| *p == Privilege::Select)
                    && asked.iter().any(|(_, p)| *p == Privilege::Browse),
                "{name}: {asked:?}"
            );
        }
        // `cost` is the statistics: `Select`.
        assert_eq!(
            class(block_on(df.explain_for_metadata(ExplainMode::Cost))),
            DENIED,
            "{name}: cost"
        );
        // Nothing of it is kept: the frame's own actions still ask `Select`.
        assert_eq!(class(block_on(df.collect())), DENIED, "{name}: collect");
    }
}

#[test]
fn an_operator_gets_the_ordinary_explain() {
    let (_policy, s, _fixture) = setup(OPERATOR);
    for (name, df) in frames(&s) {
        for mode in [
            ExplainMode::Simple,
            ExplainMode::Extended,
            ExplainMode::Cost,
        ] {
            let text = block_on(df.explain_for_metadata(mode))
                .unwrap_or_else(|e| panic!("{name} {mode:?}: {e}"));
            assert!(!text.contains(NOTE), "{name} {mode:?}: {text}");
            assert_eq!(text, block_on(df.explain(mode)).unwrap(), "{name} {mode:?}");
        }
    }
}

#[test]
fn a_viewer_cannot_explain_a_permanent_view_whose_body_the_plan_shows() {
    let (_policy, s, _fixture) = setup(VIEWER);
    let df = s.table("pv");
    for mode in MODES {
        assert_eq!(
            class(block_on(df.explain_for_metadata(mode))),
            DENIED,
            "{mode:?}"
        );
    }
    // Its schema stays metadata (PR-14c): only the body is withheld.
    assert!(block_on(df.schema_for_metadata()).is_ok());
}

#[test]
fn a_read_by_path_needs_select_for_a_withheld_explain_too() {
    let (_policy, s, fixture) = setup(VIEWER);
    let df = s.read().parquet([fixture.inside.as_str()]);
    assert_eq!(
        class(block_on(df.explain_for_metadata(ExplainMode::Simple))),
        DENIED
    );
}

#[test]
fn a_policy_that_fails_on_select_gets_the_withheld_explain() {
    let (_policy, s, _fixture) = setup(BROKEN_SELECT);
    let text = block_on(s.table("t").explain_for_metadata(ExplainMode::Simple)).unwrap();
    assert_withheld(&text, "broken select");
}

#[test]
fn a_viewer_sees_the_same_plan_whatever_the_tables_hold() {
    let (_policy, s, _fixture) = setup(VIEWER);
    let explain = |s: &Session| {
        block_on(
            s.table("t")
                .join_using(&s.table("u"), ["a"], krishiv_session::JoinType::Inner)
                .explain_for_metadata(ExplainMode::Simple),
        )
        .unwrap()
    };
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let rows = |n: i32| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from((0..n).collect::<Vec<_>>()))],
        )
        .unwrap()
    };
    s.register_batches("u", Arc::clone(&schema), vec![rows(1)])
        .unwrap();
    let small = explain(&s);
    s.register_batches("u", Arc::clone(&schema), vec![rows(100_000)])
        .unwrap();
    let large = explain(&s);
    assert_withheld(&small, "small");
    assert_eq!(small, large);
}

#[test]
fn sql_explain_as_a_viewer() {
    let (_policy, s, _fixture) = setup(VIEWER);
    let text = |sql: &str| {
        let batches = block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        use arrow_array::cast::AsArray as _;
        let c = batches[0].column(0);
        match c.as_string_view_opt() {
            Some(v) => v.value(0).to_owned(),
            None => c.as_string::<i32>().value(0).to_owned(),
        }
    };
    assert_withheld(&text("EXPLAIN SELECT a FROM p WHERE a = 1"), "EXPLAIN");
    assert_withheld(
        &text("EXPLAIN EXTENDED SELECT a FROM p WHERE a = 1"),
        "EXPLAIN EXTENDED",
    );
    // SQL `EXPLAIN` is lenient: a refusal is the text, as Spark prints a failed analysis.
    let cost = text("EXPLAIN COST SELECT a FROM p");
    assert!(cost.contains(DENIED), "{cost}");
    // `EXTENDED` shows the failure (plain `EXPLAIN` prints only Spark's planning-error line).
    let view = text("EXPLAIN EXTENDED SELECT a FROM pv");
    assert!(view.contains(DENIED) && !view.contains(NOTE), "{view}");
    let plain = text("EXPLAIN SELECT a FROM pv");
    assert!(!plain.contains(NOTE) && !plain.contains("Scan"), "{plain}");
}

#[test]
fn a_view_behind_a_temporary_view_a_cte_or_a_subquery_needs_select_too() {
    let (policy, s, _fixture) = setup(ADMIN);
    run(&s, "CREATE TEMPORARY VIEW over_pv AS SELECT a FROM pv");
    policy.become_(VIEWER);
    assert_eq!(
        class(block_on(
            s.table("over_pv").explain_for_metadata(ExplainMode::Simple)
        )),
        DENIED
    );
    // SQL frames analyze with `Select` when made, so these go through SQL `EXPLAIN`.
    for sql in [
        "SELECT a FROM over_pv",
        "WITH c AS (SELECT a FROM pv) SELECT a FROM c",
        "SELECT a FROM t WHERE a IN (SELECT a FROM pv)",
    ] {
        let text = text_of(&s, &format!("EXPLAIN EXTENDED {sql}"));
        assert!(
            text.contains(DENIED) && !text.contains(NOTE),
            "{sql}: {text}"
        );
    }
    // The same plans over a table are shown.
    let text = text_of(&s, "EXPLAIN WITH c AS (SELECT a FROM t) SELECT a FROM c");
    assert_withheld(&text, "a CTE over a table");
}

/// Review B1: a permanent view whose text no longer analyzes. Its error names the text
/// (`secretcol`), so a viewer gets the view's denial, an operator the error.
#[test]
fn a_broken_view_shows_its_text_only_with_select() {
    let (policy, s, fixture) = setup(VIEWER);
    let admin = &fixture.admin;
    run(admin, "CREATE TABLE q (secretcol INT) USING parquet");
    run(
        admin,
        "CREATE VIEW qv AS SELECT secretcol FROM q WHERE secretcol = 424242",
    );
    run(admin, "DROP TABLE q");
    run(admin, "CREATE TABLE q (b INT) USING parquet");
    policy.become_(VIEWER);
    let df = s.table("qv");
    let message = |r: Result<String, KrishivError>| match r {
        Ok(text) => panic!("an error, not {text}"),
        Err(e) => (e.error_class().name().to_owned(), e.to_string()),
    };
    for (what, (class, text)) in [
        (
            "explain",
            message(block_on(df.explain_for_metadata(ExplainMode::Simple))),
        ),
        (
            "schema",
            message(block_on(df.schema_for_metadata()).map(|s| format!("{s:?}"))),
        ),
    ] {
        assert_eq!(class, DENIED, "{what}: {text}");
        assert!(
            !text.contains("secretcol") && !text.contains("424242"),
            "{what}: {text}"
        );
    }
    let sql = text_of(&s, "EXPLAIN EXTENDED SELECT * FROM qv");
    assert!(!sql.contains("secretcol") && sql.contains(DENIED), "{sql}");
    policy.become_(OPERATOR);
    let (class, text) = message(block_on(
        s.table("qv").explain_for_metadata(ExplainMode::Simple),
    ));
    assert_ne!(class, DENIED, "{text}");
    assert!(text.contains("secretcol"), "{text}");
}

/// The one string value of a `EXPLAIN` statement's result.
fn text_of(s: &Session, sql: &str) -> String {
    use arrow_array::cast::AsArray as _;
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let c = batches[0].column(0);
    match c.as_string_view_opt() {
        Some(v) => v.value(0).to_owned(),
        None => c.as_string::<i32>().value(0).to_owned(),
    }
}

/// A withheld explain takes no distinct-value sample: a later `EXPLAIN COST` samples afresh,
/// never "sampled by an earlier query".
#[test]
fn a_withheld_explain_samples_nothing() {
    let (policy, s, fixture) = setup(VIEWER);
    s.conf()
        .set("krishiv.optimizer.ndvSampling.enabled", "true")
        .unwrap();
    run(&fixture.admin, "CREATE TABLE ps (s STRING) USING parquet");
    run(
        &fixture.admin,
        "INSERT INTO ps SELECT CAST(id % 7 AS STRING) FROM range(100)",
    );
    let sql = "SELECT s FROM ps WHERE s = '3'";
    assert_withheld(&text_of(&s, &format!("EXPLAIN {sql}")), "withheld");
    policy.become_(OPERATOR);
    let cost = text_of(&s, &format!("EXPLAIN COST {sql}"));
    assert!(cost.contains("Distinct-Value Samples"), "{cost}");
    assert!(!cost.contains("sampled by an earlier query"), "{cost}");
}

/// A withheld explain counts no key (design P2b A5 (d)): with the key counts on, a later
/// operator's `EXPLAIN VERBOSE` counts afresh, never "cached".
#[test]
fn a_withheld_explain_counts_no_key() {
    let (policy, s, fixture) = setup(VIEWER);
    for (k, v) in [
        ("krishiv.optimizer.payloadAboveAggregate", "true"),
        ("krishiv.optimizer.discoverKeys", "true"),
    ] {
        s.conf().set(k, v).unwrap();
    }
    run(
        &fixture.admin,
        "CREATE TABLE pc (c_custkey BIGINT, c_name STRING) USING parquet",
    );
    run(
        &fixture.admin,
        "INSERT INTO pc SELECT id, concat('C', CAST(id AS STRING)) FROM range(1, 201)",
    );
    run(
        &fixture.admin,
        "CREATE TABLE po (o_custkey BIGINT, o_price DECIMAL(12, 2)) USING parquet",
    );
    run(
        &fixture.admin,
        "INSERT INTO po SELECT id % 200 + 1, CAST(id % 97 AS DECIMAL(12, 2)) FROM range(2000)",
    );
    let sql = "SELECT c_custkey, c_name, sum(o_price) FROM po, pc WHERE o_custkey = c_custkey \
               GROUP BY c_custkey, c_name";
    let withheld = text_of(&s, &format!("EXPLAIN {sql}"));
    assert_withheld(&withheld, "withheld");
    assert!(!withheld.contains("discoverKeys"), "{withheld}");
    policy.become_(OPERATOR);
    let verbose = text_of(&s, &format!("EXPLAIN VERBOSE {sql}"));
    assert!(
        verbose.contains("key [c_custkey]: discovered unique"),
        "{verbose}"
    );
    assert!(!verbose.contains("cached unique"), "{verbose}");
}

/// A withheld explain whose footers do not read: the class stays, the paths (which can carry
/// partition values) do not.
#[test]
fn a_withheld_planning_failure_shows_no_path() {
    let (policy, s, fixture) = setup(VIEWER);
    run(&fixture.admin, "CREATE TABLE broken (a INT) USING parquet");
    run(&fixture.admin, "INSERT INTO broken VALUES (1)");
    let mut corrupted = 0;
    for entry in walk(fixture._tmp.path()) {
        if entry.extension().is_some_and(|e| e == "parquet")
            && entry.to_string_lossy().contains("broken")
        {
            std::fs::write(&entry, b"not a parquet file").unwrap();
            corrupted += 1;
        }
    }
    assert!(corrupted > 0);
    let root = fixture._tmp.path().to_string_lossy().into_owned();
    let e = block_on(s.table("broken").explain_for_metadata(ExplainMode::Simple)).unwrap_err();
    assert!(!e.to_string().contains(&root), "{e}");
    policy.become_(OPERATOR);
    let ordinary =
        block_on(s.table("broken").explain_for_metadata(ExplainMode::Simple)).unwrap_err();
    assert_eq!(e.error_class(), ordinary.error_class());
    assert_eq!(e.sql_state(), ordinary.sql_state());
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out
}
