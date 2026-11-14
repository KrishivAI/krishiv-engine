// SPDX-License-Identifier: Apache-2.0
//! `Session::analyze` and the `krishiv.analyze.v1` JSON (P3-T063 N1b): goldens over the parser corpus and the
//! TPC-DS queries (each checked against the schema description), the schema file, the command result
//! schemas, `partial` statements, the `Manage`-on-cluster target of `ADMIN`, denials and redaction. The
//! securables match what a run asks `authorize()` for, both ways (`securables_equal_what_a_run_authorizes`).
#![cfg(test)]

mod analyze_support;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use analyze_support::{
    assert_valid, corpus, corpus_session, layout, root, tpcds_query, tpcds_session,
};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::lineage::schema::{STATEMENT_CLASSES, WARNING_CODES, json_schema};
use krishiv_session::{
    AccessDecision, AccessPolicy, AnalysisWarning, BoxFuture, PolicyVersion, Principal,
    PrincipalKind, Privilege, Securable, Session, StatementClass,
};

fn corpus_ready() -> Session {
    let s = Session::new().unwrap();
    corpus_session(&s);
    s
}

/// One block per corpus statement: the label and SQL, then the layout JSON or the error class.
#[test]
fn corpus_goldens() {
    let s = corpus_ready();
    let mut out = String::new();
    for (label, sql) in corpus() {
        writeln!(out, "-- {label}: {sql}").unwrap();
        match block_on(s.analyze(&sql)) {
            Ok(a) => {
                let json = a.to_json();
                assert_valid(&json);
                writeln!(out, "{}\n", layout(&json)).unwrap();
            }
            Err(e) => writeln!(out, "error {}\n", e.error_class().name()).unwrap(),
        }
    }
    insta::assert_snapshot!(out);
}

/// The JSON of every TPC-DS query that analyzes, one golden each; the others are listed with their
/// error class.
#[test]
fn tpcds_goldens() {
    let s = tpcds_session();
    let mut failures = String::new();
    let mut analyzed = 0;
    for q in 1..=99 {
        match block_on(s.analyze(&tpcds_query(q))) {
            Ok(a) => {
                assert_eq!(a.info.class, StatementClass::Read);
                assert!(!a.partial);
                assert_eq!(a.lineage.len(), a.output_schema.len(), "q{q:02}");
                let json = a.to_json();
                assert_valid(&json);
                // Deterministic: a second analysis writes the same bytes.
                assert_eq!(
                    block_on(s.analyze(&tpcds_query(q))).unwrap().to_json(),
                    json
                );
                insta::assert_snapshot!(format!("tpcds_q{q:02}"), layout(&json));
                analyzed += 1;
            }
            Err(e) => writeln!(failures, "q{q:02} {}", e.error_class().name()).unwrap(),
        }
    }
    assert!(analyzed >= 90, "{analyzed} analyzed:\n{failures}");
    insta::assert_snapshot!("tpcds_not_analyzed", failures);
}

#[test]
fn json_schema_snapshot() {
    let path = root().join("docs/reference/analyze/krishiv.analyze.v1.schema.json");
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == json_schema(),
        "{} is stale: run `cargo xtask docs-analyze`",
        path.display()
    );
}

/// Every analyze golden on disk is a document the schema description accepts.
#[test]
fn goldens_validate_against_schema() {
    let dir = PathBufExt::snapshots();
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.starts_with("analyze__") || !name.ends_with(".snap") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        // A document starts at a line `{` begins and ends at the blank line (or the end).
        let body = text.splitn(3, "---\n").nth(2).unwrap_or_default();
        for block in body.split("\n\n") {
            let doc: String = block
                .lines()
                .skip_while(|l| !l.starts_with('{'))
                .collect::<Vec<_>>()
                .join("\n");
            if !doc.is_empty() {
                assert_valid(&doc);
                checked += 1;
            }
        }
    }
    assert!(checked > 150, "{checked} documents");
}

struct PathBufExt;

impl PathBufExt {
    fn snapshots() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
    }
}

/// The vocabularies the schema lists match the session's enums.
#[test]
fn schema_vocabularies_match_the_session() {
    let classes = [
        StatementClass::Read,
        StatementClass::Write,
        StatementClass::Ddl,
        StatementClass::Session,
        StatementClass::Admin,
    ];
    // An exhaustive match: a sixth class does not compile until it is listed above.
    let listed = |c: StatementClass| match c {
        StatementClass::Read
        | StatementClass::Write
        | StatementClass::Ddl
        | StatementClass::Session
        | StatementClass::Admin => classes.contains(&c),
    };
    assert!(classes.iter().all(|c| listed(*c)));
    let names: Vec<&str> = classes.iter().map(|c| c.name()).collect();
    assert_eq!(names, STATEMENT_CLASSES);
    let codes: Vec<&str> = AnalysisWarning::ALL.iter().map(|w| w.code()).collect();
    assert_eq!(codes, WARNING_CODES);
}

/// The result schema of every `SHOW`/`DESCRIBE` command kind (the corpus has one statement per kind,
/// `krishiv-sql`'s classify test checks it) and of the `EXPLAIN`, `SET` and `RESET` forms, with the column
/// names and types the command returns (`RESET` and `SET PATH`: none).
#[test]
fn command_output_schemas() {
    let s = corpus_ready();
    block_on(s.sql("CREATE TABLE pt (a INT, p INT) USING parquet PARTITIONED BY (p)")).unwrap();
    let mut statements: Vec<String> = corpus()
        .into_iter()
        .filter(|(label, _)| {
            (label.starts_with("Show::") || label.starts_with("Describe::"))
                && label != "Describe::Query"
        })
        .map(|(_, sql)| sql.replace("SHOW PARTITIONS t", "SHOW PARTITIONS pt"))
        .collect();
    statements.extend(
        [
            "DESCRIBE QUERY SELECT a, b FROM t",
            "DESCRIBE TABLE t a",
            "DESCRIBE TABLE EXTENDED t AS JSON",
            "SHOW TBLPROPERTIES t ('k')",
            "SHOW CURRENT SCHEMA",
            "EXPLAIN SELECT a FROM t",
            "EXPLAIN EXTENDED SELECT a FROM t",
            "EXPLAIN SET",
            "SET",
            "SET -v",
            "SET spark.sql.ansi.enabled",
            "SET spark.sql.ansi.enabled = true",
            "SET TIME ZONE 'UTC'",
            "RESET spark.sql.ansi.enabled",
            "RESET",
        ]
        .map(str::to_owned),
    );
    let mut out = String::new();
    for sql in &statements {
        let a = block_on(s.analyze(sql)).unwrap();
        assert_valid(&a.to_json());
        // A command the session refuses has no result schema.
        let df = match block_on(s.sql(sql)) {
            Ok(df) => df,
            Err(e) => {
                assert_eq!(
                    e.error_class().name(),
                    "KRISHIV_UNSUPPORTED_FEATURE",
                    "{sql}"
                );
                assert!(a.output_schema.is_empty(), "{sql}");
                writeln!(out, "{sql}\n  (refused by the session)").unwrap();
                continue;
            }
        };
        let cols: Vec<String> = a
            .output_schema
            .iter()
            .map(|c| format!("{}: {}", c.name, c.data_type.simple_string()))
            .collect();
        if cols.is_empty() {
            writeln!(out, "{sql}\n  (no columns)").unwrap();
        } else {
            writeln!(out, "{sql}\n  {}", cols.join(", ")).unwrap();
        }
        // The command's own frame has the same names and types.
        let run = block_on(df.schema()).unwrap();
        let ran: Vec<String> = run
            .fields()
            .iter()
            .map(|f| {
                let ty = match f.data_type() {
                    arrow_schema::DataType::Boolean => "boolean",
                    arrow_schema::DataType::Utf8
                    | arrow_schema::DataType::LargeUtf8
                    | arrow_schema::DataType::Utf8View => "string",
                    other => panic!("{sql}: {other}"),
                };
                format!("{}: {ty}", f.name())
            })
            .collect();
        assert_eq!(cols, ran, "{sql}");
    }
    insta::assert_snapshot!(out);
}

/// An `ADMIN` statement needs `Manage` on the cluster, and `statement_info` lists it (review m3).
#[test]
fn admin_statements_list_manage_on_the_cluster() {
    let s = corpus_ready();
    for sql in [
        "CLEAR CACHE",
        "REFRESH TABLE t",
        "EXPLAIN ANALYZE CLEAR CACHE",
    ] {
        let info = block_on(s.statement_info(sql)).unwrap();
        assert_eq!(info.class, StatementClass::Admin, "{sql}");
        let manage = info
            .securables
            .iter()
            .filter(|u| {
                u.privilege == Privilege::Manage
                    && u.securable == Securable::System("cluster".to_owned())
            })
            .count();
        assert_eq!(manage, 1, "{sql}: {:?}", info.securables);
    }
    let info = block_on(s.statement_info("SELECT a FROM t")).unwrap();
    assert!(
        info.securables
            .iter()
            .all(|u| u.privilege != Privilege::Manage)
    );
}

/// The privilege and securable of each use, as text.
fn uses(a: &krishiv_session::Analysis) -> Vec<String> {
    a.info
        .securables
        .iter()
        .map(|u| {
            let columns: Vec<&str> = u.columns.iter().map(String::as_str).collect();
            format!("{} {} {columns:?}", u.privilege, u.securable)
        })
        .collect()
}

/// A statement whose reads are not known from its text is `partial`: its targets,
/// `UNSUPPORTED_READ_SHAPE`, and no analysis (`statement_info` refuses it).
#[test]
fn unsupported_read_shapes_are_partial() {
    let s = corpus_ready();
    for sql in [
        "EXECUTE IMMEDIATE 'SELECT ?' USING 1",
        "EXECUTE IMMEDIATE 'SELECT 1' || ''",
        "SET VAR v = (SELECT 1)",
    ] {
        let a = block_on(s.analyze(sql)).unwrap();
        assert!(a.partial, "{sql}");
        assert_eq!(
            a.warnings.iter().copied().collect::<Vec<_>>(),
            [AnalysisWarning::UnsupportedReadShape]
        );
        assert!(a.output_schema.is_empty() && a.lineage.is_empty());
        assert!(a.to_json().contains("\"partial\":true"));
        assert!(block_on(s.statement_info(sql)).is_err());
    }
}

/// A MERGE with a subquery in `ON`, a clause condition or a `SET` value fails closed (review m1): `partial`,
/// with its targets only; one without keeps its reads (design §3.2).
#[test]
fn merge_subquery_condition_is_partial() {
    let s = corpus_ready();
    let head = "MERGE INTO t USING s ON t.k = s.k";
    for sql in [
        "MERGE INTO t USING s ON t.k = (SELECT max(k) FROM u) WHEN MATCHED THEN DELETE".to_owned(),
        format!("{head} WHEN MATCHED AND EXISTS (SELECT 1 FROM u) THEN DELETE"),
        format!("{head} WHEN MATCHED THEN UPDATE SET a = (SELECT max(a) FROM u)"),
    ] {
        let a = block_on(s.analyze(&sql)).unwrap();
        assert!(a.partial, "{sql}");
        assert!(
            a.warnings.contains(&AnalysisWarning::UnsupportedReadShape),
            "{sql}"
        );
        assert!(
            !a.to_json().contains("`u`"),
            "{sql}: a subquery's table is not listed"
        );
    }
    let a = block_on(s.analyze(&format!("{head} WHEN MATCHED THEN DELETE"))).unwrap();
    assert!(!a.partial);
    assert_eq!(
        uses(&a),
        [
            "MODIFY table `spark_catalog`.`default`.`t` []",
            "SELECT table `spark_catalog`.`default`.`s` []",
        ]
    );
}

/// `UPDATE`, `DELETE` with a subquery predicate and a multi-table `INSERT` with a shared `FROM` list their
/// reads, subqueries included (design §3.2).
#[test]
fn local_read_shapes_list_their_reads() {
    let s = corpus_ready();
    let a = block_on(s.analyze(
        "UPDATE t AS x SET a = (SELECT max(a) FROM u), b = DEFAULT WHERE x.k IN (SELECT k FROM s)",
    ))
    .unwrap();
    assert!(!a.partial && a.warnings.is_empty());
    assert_eq!(
        uses(&a),
        [
            "SELECT table `spark_catalog`.`default`.`t` [\"a\", \"b\", \"k\"]",
            "SELECT table `spark_catalog`.`default`.`u` [\"a\"]",
            "SELECT table `spark_catalog`.`default`.`s` [\"k\"]",
            "MODIFY table `spark_catalog`.`default`.`t` []",
        ]
    );
    let a = block_on(s.analyze("DELETE FROM t WHERE k IN (SELECT k FROM s)")).unwrap();
    assert!(!a.partial);
    assert_eq!(
        uses(&a),
        [
            "SELECT table `spark_catalog`.`default`.`s` [\"k\"]",
            "SELECT table `spark_catalog`.`default`.`t` [\"k\"]",
            "MODIFY table `spark_catalog`.`default`.`t` []",
        ]
    );
    let sql = "FROM s INSERT INTO t SELECT a, b, k INSERT INTO u SELECT a, b, k WHERE k > 0";
    let a = block_on(s.analyze(sql)).unwrap();
    assert!(!a.partial);
    assert_eq!(
        uses(&a),
        [
            "SELECT table `spark_catalog`.`default`.`s` [\"a\", \"b\", \"k\"]",
            "MODIFY table `spark_catalog`.`default`.`t` []",
            "MODIFY table `spark_catalog`.`default`.`u` []",
        ]
    );
    assert!(block_on(s.statement_info(sql)).is_ok());
}

/// `EXECUTE IMMEDIATE` of literal text reports the class and securables of the statement it holds (review
/// m2), in analyze, `statement_info` and `system.queries.statement_class`; text from a variable is `ADMIN`
/// and `partial`.
#[test]
fn execute_immediate_literal_reports_inner_class() {
    let s = corpus_ready();
    let a = block_on(s.analyze("EXECUTE IMMEDIATE 'SELECT a FROM t'")).unwrap();
    assert_eq!(a.info.class, StatementClass::Read);
    assert!(!a.partial);
    assert_eq!(
        uses(&a),
        ["SELECT table `spark_catalog`.`default`.`t` [\"a\"]"]
    );
    assert_eq!(a.output_schema.len(), 1);
    let a = block_on(s.analyze("EXECUTE IMMEDIATE 'DROP TABLE u'")).unwrap();
    assert_eq!(a.info.class, StatementClass::Ddl);
    assert_eq!(uses(&a), ["DROP table `spark_catalog`.`default`.`u` []"]);
    let info = block_on(s.statement_info("EXECUTE IMMEDIATE 'CLEAR CACHE'")).unwrap();
    assert_eq!(info.class, StatementClass::Admin);
    assert_eq!(info.securables.len(), 1, "{:?}", info.securables);
    block_on(s.sql("DECLARE stmt STRING DEFAULT 'SELECT 1'")).unwrap();
    let a = block_on(s.analyze("EXECUTE IMMEDIATE stmt")).unwrap();
    assert_eq!(a.info.class, StatementClass::Admin);
    assert!(a.partial && a.info.securables.is_empty());
    // A run records the statement it ran under its own text and class (the outer text is not
    // recorded); dynamic text needs no `Manage` of its own.
    block_on(
        block_on(s.sql("EXECUTE IMMEDIATE 'SELECT a FROM t'"))
            .unwrap()
            .collect(),
    )
    .unwrap();
    block_on(block_on(s.sql("EXECUTE IMMEDIATE stmt")).unwrap().collect()).unwrap();
    let rows = block_on(
        block_on(s.sql(
            "SELECT concat(statement_class, ' ', query_text) FROM system.queries \
             WHERE query_text IN ('SELECT a FROM t', 'SELECT 1') ORDER BY submitted_at",
        ))
        .unwrap()
        .collect(),
    )
    .unwrap();
    let classes: Vec<String> = rows
        .iter()
        .flat_map(|b| {
            use arrow_array::cast::AsArray as _;
            let c = b.column(0);
            (0..c.len())
                .map(|i| match c.as_string_view_opt() {
                    Some(v) => v.value(i).to_owned(),
                    None => c.as_string::<i32>().value(i).to_owned(),
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        classes,
        ["READ SELECT a FROM t", "READ SELECT 1"],
        "{classes:?}"
    );
}

/// Denies `Manage` (on anything), allows the rest: an ordinary user under a policy that keeps cluster
/// administration to admins.
struct DenyManage;

impl AccessPolicy for DenyManage {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        _: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let d = if privilege == Privilege::Manage {
            AccessDecision::Deny {
                reason: "admins only".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }
}

/// Dynamic SQL needs no `Manage` on the cluster wherever it stands (review B1, design §4 N3 decisions):
/// a script running it, `EXPLAIN ANALYZE` of it and marker text that does not parse with its markers
/// replaced all run for a user denied `Manage`, and analyze lists no `MANAGE` for them, though their class
/// is `ADMIN`. Literal `CLEAR CACHE` still needs `Manage`, at the top and in a script.
#[test]
fn dynamic_sql_needs_no_manage() {
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(DenyManage))
        .build()
        .unwrap();
    corpus_session(&s);
    block_on(s.sql("DECLARE stmt STRING DEFAULT 'SELECT 1'")).unwrap();
    let run = |sql: &str| -> Result<(), KrishivError> {
        block_on(async { s.sql(sql).await?.collect().await.map(|_| ()) })
    };
    // The run's outcome: `None` runs; otherwise the error class it fails with after the up-front check,
    // which asked no `Manage` (never `KRISHIV_INSUFFICIENT_PERMISSIONS`).
    for (sql, fails) in [
        (
            "BEGIN DECLARE x STRING DEFAULT 'SELECT 1'; EXECUTE IMMEDIATE x; END",
            None,
        ),
        ("EXECUTE IMMEDIATE stmt", None),
        // Not supported yet.
        (
            "EXPLAIN ANALYZE EXECUTE IMMEDIATE stmt",
            Some("KRISHIV_UNSUPPORTED_FEATURE"),
        ),
        // Text that does not parse with its markers replaced (so `ADMIN`); bound, it names two
        // parts, which Spark refuses as well.
        (
            "EXECUTE IMMEDIATE 'SELECT 1 AS IDENTIFIER(:s ''.'' :c)' USING 'd' AS s, 'c' AS c",
            Some("IDENTIFIER_TOO_MANY_NAME_PARTS"),
        ),
    ] {
        let a = block_on(s.analyze(sql)).unwrap();
        assert_eq!(a.info.class, StatementClass::Admin, "{sql}");
        assert!(
            a.info
                .securables
                .iter()
                .all(|u| u.privilege != Privilege::Manage),
            "{sql}: {:?}",
            uses(&a)
        );
        let outcome = run(sql).err().map(|e| e.error_class().name().to_owned());
        assert_eq!(outcome.as_deref(), fails, "{sql}");
    }
    for sql in [
        "EXECUTE IMMEDIATE 'CLEAR CACHE'",
        "BEGIN EXECUTE IMMEDIATE 'CLEAR CACHE'; END",
    ] {
        let e = run(sql).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{sql}: {e}"
        );
    }
}

/// A partial `EXECUTE IMMEDIATE ... USING` lists the targets of its text, which parses with its markers
/// (review m4); its reads stay unlisted.
#[test]
fn execute_immediate_using_lists_its_targets() {
    let s = corpus_ready();
    let a = block_on(
        s.analyze("EXECUTE IMMEDIATE 'INSERT INTO t SELECT a, b, k FROM s WHERE k = ?' USING 1"),
    )
    .unwrap();
    assert_eq!(a.info.class, StatementClass::Write);
    assert!(a.partial);
    assert_eq!(uses(&a), ["MODIFY table `spark_catalog`.`default`.`t` []"]);
}

/// Denies everything on the table `canary` once armed (its version moves with it), allows the rest.
struct DenyCanary(std::sync::atomic::AtomicBool);

impl AccessPolicy for DenyCanary {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(u64::from(self.0.load(Ordering::SeqCst)) + 1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let d = if self.0.load(Ordering::SeqCst)
            && matches!(securable, Securable::Table(t) if t.name.eq_ignore_ascii_case("canary"))
        {
            AccessDecision::Deny {
                reason: "canary".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }
}

/// The analyze JSON as a surface of T062's leak matrix (design §3.5, review M3): a securable the principal
/// has nothing on and that is not written in the statement never appears in it. Every statement that would
/// read the canary fails analysis with the 42501 a run gives (so no JSON), whatever shape reaches it
/// (subqueries, a temporary view, `UPDATE` and `DELETE` expressions, a multi-table `INSERT`'s shared
/// `FROM`, `EXECUTE IMMEDIATE`); every other one analyzes, and its JSON never names the canary, the
/// `partial` shapes that read it included (a `MERGE` subquery, `EXECUTE IMMEDIATE ... USING`: their reads
/// are not listed at all).
///
/// The one exception, by design (§1, §3.5): a target (what the statement writes, creates or drops) is
/// listed without being authorized, so the canary as a target in the user's own text appears, as the user
/// wrote it; the JSON reveals nothing the text does not hold.
#[test]
fn analyze_canary_never_in_json() {
    let admin = corpus_ready();
    block_on(admin.sql("CREATE TABLE canary (a INT, k INT)")).unwrap();
    let policy = Arc::new(DenyCanary(std::sync::atomic::AtomicBool::new(false)));
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .runtime(Arc::clone(admin.runtime()))
        .build()
        .unwrap();
    // A temporary view over the canary, made while it was readable: it is re-analyzed under the
    // session's principal, so reading through it reads the canary.
    block_on(s.sql("CREATE TEMPORARY VIEW over_canary AS SELECT * FROM canary")).unwrap();
    policy.0.store(true, Ordering::SeqCst);
    let reaches = [
        "SELECT a FROM t WHERE k IN (SELECT k FROM canary)",
        "SELECT (SELECT max(a) FROM canary) AS m",
        "UPDATE t SET a = (SELECT max(a) FROM canary)",
        "DELETE FROM t WHERE EXISTS (SELECT 1 FROM canary)",
        "FROM canary INSERT INTO t SELECT a, a, k INSERT INTO u SELECT a, a, k",
        "EXECUTE IMMEDIATE 'SELECT * FROM canary'",
        "EXPLAIN FORMAT JSON SELECT * FROM t JOIN canary USING (k)",
        "CREATE TABLE x AS SELECT * FROM canary",
        "CREATE TEMPORARY VIEW mine AS SELECT * FROM canary",
        "SELECT * FROM over_canary",
        "DELETE FROM t WHERE k = (SELECT max(k) FROM canary)",
    ];
    for sql in reaches {
        let e = block_on(s.analyze(sql)).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{sql}"
        );
    }
    let clean = [
        "SELECT a FROM t WHERE k IN (SELECT k FROM s)",
        "UPDATE t SET a = 1 WHERE k IN (SELECT k FROM s)",
        "MERGE INTO t USING s ON t.k = (SELECT max(k) FROM s) WHEN MATCHED THEN DELETE",
        "EXECUTE IMMEDIATE 'SELECT * FROM t'",
        "EXPLAIN FORMAT JSON DELETE FROM t WHERE k IN (SELECT k FROM s)",
        "SHOW TABLES",
        // Partial: their reads are not listed, so the canary they read is not either.
        "MERGE INTO t USING s ON t.k = (SELECT max(k) FROM canary) WHEN MATCHED THEN DELETE",
        "EXECUTE IMMEDIATE 'SELECT * FROM canary WHERE a = ?' USING 1",
    ];
    for sql in clean {
        let json = block_on(s.analyze(sql)).unwrap().to_json();
        assert!(!json.to_lowercase().contains("canary"), "{sql}: {json}");
    }
    // The target exception: the user's own text names it.
    let json = block_on(s.analyze("DROP TABLE canary")).unwrap().to_json();
    assert!(json.contains("canary"), "{json}");
}

/// Denies `Select` on table `t` (by name), allows the rest.
struct DenyT(AtomicU64);

impl AccessPolicy for DenyT {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(self.0.load(Ordering::SeqCst))
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let denied = privilege == Privilege::Select
            && matches!(securable, Securable::Table(t) if t.name == "t");
        let d = if denied {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }
}

/// A denied read fails analyze with the same error a run gives.
#[test]
fn denied_read_fails_like_run() {
    let admin = corpus_ready();
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(DenyT(AtomicU64::new(1))))
        .runtime(Arc::clone(admin.runtime()))
        .build()
        .unwrap();
    for sql in [
        "SELECT a FROM t",
        "SELECT * FROM s WHERE k IN (SELECT k FROM t)",
        "INSERT INTO u SELECT * FROM t",
    ] {
        let analyzed = block_on(s.analyze(sql)).unwrap_err();
        let ran = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            analyzed.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS"
        );
        assert_eq!(analyzed.to_string(), ran.to_string(), "{sql}");
    }
    assert!(block_on(s.analyze("SELECT a FROM s")).is_ok());
}

/// Denies `Modify` on paths, allows the rest.
struct DenyPaths;

impl AccessPolicy for DenyPaths {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let d = if matches!(securable, Securable::Path(_)) && privilege == Privilege::Modify {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }
}

/// A securable's `display` in the JSON is byte-equal to the 42501 text the run's denial prints (ADR 0038's
/// redaction through `Securable`'s `Display`, reused); credentialed URLs and secret references are checked
/// in the unit test `analyze::tests::redaction_matches_the_denial_text` (this build has no cloud schemes).
#[test]
fn redaction() {
    let dir = tempfile::tempdir().unwrap();
    let sql = format!(
        "CREATE TABLE x (a INT) USING parquet LOCATION '{}'",
        dir.path().join("x").display()
    );
    let s = corpus_ready();
    let a = block_on(s.analyze(&sql)).unwrap();
    let json = a.to_json();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let path = v["securables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|u| u["kind"] == "path")
        .unwrap_or_else(|| panic!("{json}"));
    let display = path["display"].as_str().unwrap();
    let denied = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(DenyPaths))
        .runtime(Arc::clone(s.runtime()))
        .build()
        .unwrap();
    let e = block_on(denied.sql(&sql)).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    let text = e.to_string();
    assert!(text.contains(&format!(" {display}")), "{text}\n{display}");
}

/// The `krishiv::authorize` events (one per `authorize()` call, `krishiv-plan`), per principal.
static ASKED: std::sync::Mutex<Vec<(String, String, String)>> = std::sync::Mutex::new(Vec::new());

/// Records the `krishiv::authorize` events.
struct Recorder;

#[derive(Default)]
struct AuthorizeFields {
    principal: String,
    privilege: String,
    securable: String,
}

impl tracing::field::Visit for AuthorizeFields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let v = format!("{value:?}");
        match field.name() {
            "principal" => self.principal = v,
            "privilege" => self.privilege = v,
            "securable" => self.securable = v,
            _ => {}
        }
    }
}

impl tracing::Subscriber for Recorder {
    fn enabled(&self, m: &tracing::Metadata<'_>) -> bool {
        m.target() == "krishiv::authorize"
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let mut f = AuthorizeFields::default();
        event.record(&mut f);
        ASKED
            .lock()
            .unwrap()
            .push((f.principal, f.privilege, f.securable));
    }

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

/// What `principal` asked since the last call, as (privilege, securable) with folded case.
fn asked_by(principal: &str) -> BTreeSet<(String, String)> {
    let mut all = ASKED.lock().unwrap();
    let (mine, rest): (Vec<_>, Vec<_>) = all.drain(..).partition(|(p, _, _)| p == principal);
    *all = rest;
    mine.into_iter()
        .map(|(_, privilege, s)| (privilege, warehouse_free(&s.to_lowercase())))
        .collect()
}

/// `s` with the session's temporary warehouse directory (`<tmp>/krishiv-warehouse-<n>`) as `<warehouse>`.
fn warehouse_free(s: &str) -> String {
    let Some(at) = s.find("krishiv-warehouse-") else {
        return s.to_owned();
    };
    let mut start = s[..at].rfind('\'').map_or(0, |q| q + 1);
    if s[start..].starts_with("file:") {
        start += "file:".len();
    }
    let end = s[at..].find('/').map_or(s.len(), |e| at + e);
    format!("{}<warehouse>{}", &s[..start], &s[end..])
}

/// The securable set of `analyze` is what a run asks `authorize()` about (review m3), over every corpus
/// statement whose analysis is complete and that runs: recorded at the `authorize()` call (the
/// `krishiv::authorize` event), not by a policy double, and with folded-case duplicates as one.
///
/// Both ways (N1b review M3): what analyze itself asks `authorize()` is also recorded, and it asks no more
/// than the run does and nothing it does not list.
///
/// An `UPDATE` or `DELETE` run, which fails with `UNSUPPORTED_FEATURE.TABLE_OPERATION` (no table can change
/// yet), is compared too: it fails after it has asked about its reads. A `MERGE` run is not: it analyzes
/// only `SELECT 1 FROM target` before it fails, not its source, which analyze lists (a follow-up for
/// when MERGE runs).
///
/// The snapshot pins the known differences: a run of `DROP TABLE`, `INSERT` and `TRUNCATE` on a managed
/// table also asks for its directory (`table_commands/files.rs`), which analyze does not list: the check
/// runs only when the directory exists, on its resolved path (documented in the `analyze` module doc and
/// the schema description).
///
/// The event stream is not every policy question: `system.*` tables' `Monitor` row filter
/// (`monitor.rs`) calls the policy's `check` directly, bypassing `authorize()`. It filters rows, it is
/// not a requirement, so analyze does not list it and this test does not see it.
#[test]
fn securables_equal_what_a_run_authorizes() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::subscriber::set_global_default(Recorder).unwrap());
    let mut compared = 0;
    let mut differences = String::new();
    for (i, (label, sql)) in corpus().into_iter().enumerate() {
        let principal = format!("prop{i}");
        let s = Session::builder()
            .principal(Principal::new(&principal, PrincipalKind::User))
            .build()
            .unwrap();
        corpus_session(&s);
        // The setup's own questions are not the statement's.
        asked_by(&principal);
        let Ok(a) = block_on(s.analyze(&sql)) else {
            continue;
        };
        let analyze_asked = asked_by(&principal);
        if a.partial {
            continue;
        }
        let analyzed: BTreeSet<(String, String)> = a
            .info
            .securables
            .iter()
            .map(|u| {
                (
                    u.privilege.to_string(),
                    u.securable.to_string().to_lowercase(),
                )
            })
            .collect();
        assert!(
            analyze_asked.is_subset(&analyzed),
            "{label}: {sql}\n  analyze asked but did not list: {:?}",
            analyze_asked.difference(&analyzed).collect::<Vec<_>>()
        );
        // `UPDATE` and `DELETE` analyze their queries, then find that the table cannot change (review
        // m2): what they asked is compared all the same.
        match block_on(s.sql(&sql)) {
            Ok(_) => {}
            Err(e)
                if matches!(label.as_str(), "Update" | "Delete")
                    && e.error_class().name() == "UNSUPPORTED_FEATURE.TABLE_OPERATION" => {}
            Err(_) => continue,
        }
        let ran = asked_by(&principal);
        compared += 1;
        assert!(
            analyze_asked.is_subset(&ran),
            "{label}: {sql}\n  analyze asked more than the run: {:?}",
            analyze_asked.difference(&ran).collect::<Vec<_>>()
        );
        if analyzed != ran {
            let only_analyze: Vec<_> = analyzed.difference(&ran).collect();
            let only_run: Vec<_> = ran.difference(&analyzed).collect();
            writeln!(
                differences,
                "{label}: {sql}\n  analyze only: {only_analyze:?}\n  run only: {only_run:?}"
            )
            .unwrap();
        }
    }
    assert!(compared >= 60, "{compared} compared");
    insta::assert_snapshot!(differences);
}

/// The one `plan` value of `EXPLAIN FORMAT JSON ...` (P3-T063 N2): one row, one string column.
fn explain_json(s: &Session, sql: &str) -> String {
    use arrow_array::cast::AsArray as _;
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.fields().len(), 1, "{sql}");
    assert_eq!(schema.field(0).name(), "plan", "{sql}");
    let batches = block_on(df.collect()).unwrap();
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert_eq!(rows, 1, "{sql}");
    let column = batches[0].column(0);
    match column.data_type() {
        arrow_schema::DataType::Utf8View => column.as_string_view().value(0).to_owned(),
        arrow_schema::DataType::Utf8 => column.as_string::<i32>().value(0).to_owned(),
        other => panic!("{sql}: plan is {other}"),
    }
}

/// The rows of table `name`.
fn row_count(s: &Session, name: &str) -> usize {
    let batches = block_on(async { s.table(name).collect().await }).unwrap();
    batches.iter().map(arrow_array::RecordBatch::num_rows).sum()
}

/// `EXPLAIN FORMAT JSON <statement>` returns `analyze(<statement>)`'s document as one `plan` row, and
/// runs nothing: no insert, no drop, not the `EXPLAIN ANALYZE` it explains.
#[test]
fn explain_format_json_returns_the_analysis_and_runs_nothing() {
    let s = corpus_ready();
    block_on(s.sql("INSERT INTO u VALUES (1, 2, 3)")).unwrap();
    for inner in [
        "SELECT a + 1 AS b FROM t WHERE k > 0",
        "INSERT INTO u SELECT * FROM t",
        "EXPLAIN ANALYZE INSERT INTO u SELECT * FROM t",
        "DROP TABLE u",
        "SET spark.sql.ansi.enabled = false",
    ] {
        let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
        assert_valid(&json);
        assert_eq!(
            json,
            block_on(s.analyze(inner)).unwrap().to_json(),
            "{inner}"
        );
    }
    assert_eq!(row_count(&s, "u"), 1);
    // The explained statement's class and explain kind, and the drop target, listed.
    let json = explain_json(
        &s,
        "EXPLAIN FORMAT JSON EXPLAIN ANALYZE INSERT INTO u SELECT * FROM t",
    );
    assert!(json.contains("\"statement_class\":\"WRITE\""), "{json}");
    assert!(json.contains("\"explain\":\"ANALYZE\""), "{json}");
    assert_eq!(row_count(&s, "u"), 1);
    let json = explain_json(&s, "EXPLAIN FORMAT JSON DROP TABLE u");
    assert!(json.contains("\"statement_class\":\"DDL\""), "{json}");
    assert!(
        json.contains("\"display\":\"table `spark_catalog`.`default`.`u`\",\"privilege\":\"DROP\""),
        "{json}"
    );
    assert_eq!(row_count(&s, "u"), 1);
}

/// The outer statement is `READ` with the `plan STRING` result of every `EXPLAIN`.
#[test]
fn explain_format_json_is_a_read_with_a_plan_column() {
    let s = corpus_ready();
    for sql in [
        "EXPLAIN FORMAT JSON SELECT a FROM t",
        "EXPLAIN FORMAT JSON DROP TABLE t",
        "EXPLAIN FORMAT JSON EXPLAIN ANALYZE INSERT INTO u SELECT * FROM t",
    ] {
        let a = block_on(s.analyze(sql)).unwrap();
        assert_eq!(a.info.class, StatementClass::Read, "{sql}");
        assert!(!a.partial, "{sql}");
        let cols: Vec<String> = a
            .output_schema
            .iter()
            .map(|c| format!("{}: {}", c.name, c.data_type.simple_string()))
            .collect();
        assert_eq!(cols, ["plan: string"], "{sql}");
        assert!(a.to_json().contains("\"explain\":\"PLAN\""), "{sql}");
        // Nothing is dropped or written by the outer statement.
        assert!(
            a.info
                .securables
                .iter()
                .all(|u| u.privilege == Privilege::Select),
            "{sql}: {:?}",
            a.info.securables
        );
    }
}

/// The explained statement's reads are checked: a denied read fails as the run of that statement does.
#[test]
fn explain_format_json_applies_the_read_checks() {
    let admin = corpus_ready();
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(DenyT(AtomicU64::new(1))))
        .runtime(Arc::clone(admin.runtime()))
        .build()
        .unwrap();
    for sql in ["SELECT a FROM t", "INSERT INTO u SELECT * FROM t"] {
        let explained = block_on(s.sql(&format!("EXPLAIN FORMAT JSON {sql}"))).unwrap_err();
        let ran = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            explained.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS"
        );
        assert_eq!(explained.to_string(), ran.to_string(), "{sql}");
    }
    assert!(block_on(s.sql("EXPLAIN FORMAT JSON SELECT a FROM s")).is_ok());
    // `PHYSICAL` is reserved.
    let e = block_on(s.sql("EXPLAIN FORMAT JSON PHYSICAL SELECT a FROM s")).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}
