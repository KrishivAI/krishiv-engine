// SPDX-License-Identifier: Apache-2.0
//! `Session::analyze` and the `krishiv.analyze.v1` JSON (P3-T063 N1b): goldens over the parser corpus and the
//! TPC-DS queries (each checked against the schema description), the schema file, the command result
//! schemas, `partial` statements, the `Manage`-on-cluster target of `ADMIN`, denials and redaction. The
//! securables match what a run asks `authorize()` for, both ways (`securables_equal_what_a_run_authorizes`).
#![cfg(test)]
mod literal_path;
use literal_path::sql_path;

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
                writeln!(out, "{}\n", warehouses_free(&layout(&json))).unwrap();
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
        // The snapshot also pins Spark 4.2.0's nullability (`command_schema.rs`), and the run's frame must
        // report the same (names, types and nullability are compared with it below).
        let shown: Vec<String> = a
            .output_schema
            .iter()
            .zip(&cols)
            .map(|(c, s)| {
                if c.nullable {
                    s.clone()
                } else {
                    format!("{s} not null")
                }
            })
            .collect();
        if shown.is_empty() {
            writeln!(out, "{sql}\n  (no columns)").unwrap();
        } else {
            writeln!(out, "{sql}\n  {}", shown.join(", ")).unwrap();
        }
        // The command's own frame has the same names, types and nullability.
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
                if f.is_nullable() {
                    format!("{}: {ty}", f.name())
                } else {
                    format!("{}: {ty} not null", f.name())
                }
            })
            .collect();
        assert_eq!(shown, ran, "{sql}");
    }
    insta::assert_snapshot!(out);
}

/// `SHOW GRANTS` (a Krishiv extension, the `grants` provider) reports the 7 columns its run returns,
/// with the same names, types and nullability (review #803 n3).
#[test]
fn show_grants_reports_its_result_columns() {
    use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
    let policy = Arc::new(GrantsPolicy::new(
        Grants::new(Arc::new(MemoryGrantStore::new())),
        Arc::new(krishiv_session::AllowAll),
        true,
    ));
    let s = Session::builder()
        .principal(Principal::new("root", PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .grants(policy)
        .build()
        .unwrap();
    corpus_session(&s);
    for sql in [
        "SHOW GRANTS",
        "SHOW GRANTS ON TABLE t",
        "SHOW GRANTS TO user:ana",
    ] {
        let a = block_on(s.analyze(sql)).unwrap();
        assert_valid(&a.to_json());
        let analyzed: Vec<(String, String, bool)> = a
            .output_schema
            .iter()
            .map(|c| (c.name.clone(), c.data_type.simple_string(), c.nullable))
            .collect();
        assert_eq!(analyzed.len(), 7, "{sql}: {analyzed:?}");
        let run = block_on(block_on(s.sql(sql)).unwrap().schema()).unwrap();
        let ran: Vec<(String, String, bool)> = run
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
                (f.name().clone(), ty.to_owned(), f.is_nullable())
            })
            .collect();
        assert_eq!(analyzed, ran, "{sql}");
    }
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
    let listed: Vec<String> = uses(&a).iter().map(|u| warehouse_free(u)).collect();
    assert_eq!(
        listed,
        [
            "SELECT table `spark_catalog`.`default`.`s` [\"a\", \"b\", \"k\"]",
            "MODIFY table `spark_catalog`.`default`.`t` []",
            "MODIFY table `spark_catalog`.`default`.`u` []",
            "MODIFY path 'file:<warehouse>/t' []",
            "CREATE path 'file:<warehouse>/t' []",
            "MODIFY path 'file:<warehouse>/u' []",
            "CREATE path 'file:<warehouse>/u' []",
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
    let listed: Vec<String> = uses(&a).iter().map(|u| warehouse_free(u)).collect();
    assert_eq!(
        listed,
        [
            "DROP table `spark_catalog`.`default`.`u` []",
            "MODIFY path 'file:<warehouse>/u' []"
        ]
    );
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
    // A target named by a marker cannot be placed before it is bound: left out, still `partial`
    // (review m6).
    let a = block_on(s.analyze("EXECUTE IMMEDIATE 'DROP TABLE IDENTIFIER(:t)' USING 'x' AS t"))
        .unwrap();
    assert_eq!(a.info.class, StatementClass::Ddl);
    assert!(a.partial && a.info.securables.is_empty(), "{:?}", uses(&a));
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
        sql_path(&dir.path().join("x"))
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

/// `s` with every temporary warehouse directory (`<tmp>/krishiv-warehouse-<n>`) as `<warehouse>`.
fn warehouses_free(s: &str) -> String {
    let (mut out, mut rest) = (String::new(), s);
    while let Some(at) = rest.find("krishiv-warehouse-") {
        let start = rest[..at].rfind('/').map_or(0, |i| {
            // The start of the absolute path: after `file:` or a quote.
            rest[..i].rfind(['\'', '"', ':']).map_or(0, |q| q + 1)
        });
        let end = rest[at..].find('/').map_or(rest.len(), |e| at + e);
        out.push_str(&rest[..start]);
        out.push_str("<warehouse>");
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
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
/// The snapshot pins the differences that remain. A run of `DROP TABLE`, `INSERT`, `TRUNCATE`, `CREATE
/// TABLE` (CTAS too), `ALTER TABLE ... RENAME TO` and `DROP NAMESPACE ... CASCADE` also asks about
/// directories (`table_commands/files.rs`), which analyze lists since T063 N1b (`analyze/dirs.rs`), in the
/// form the run's denial names (the same text here, with no link on the warehouse path): no difference.
/// [`DIRECTORY_CASES`] are statements of that DDL which succeed (each must), so the property compares
/// each rather than skipping a failed run. The partition directories of `ADD` and `DROP PARTITION` are
/// not listed yet (`analyze/dirs.rs` module doc): the snapshot pins those two differences.
///
/// `EXPLAIN FORMAT JSON` of a write or drop: its run analyzes the statement it explains, which asks the
/// policy about that statement's targets in a quiet trial before it lists any directory (review B2). The
/// trial requires nothing (a denial only lists no directory), so analyze of the `EXPLAIN` does not list
/// them; the snapshot pins those two questions.
///
/// Analyze of a `DROP NAMESPACE ... CASCADE` asks `BROWSE` on the namespace before it lists the
/// directories of its tables (review D1); the run does not ask it, so that one question is left out.
///
/// The event stream is not every policy question: `system.*` tables' `Monitor` row filter
/// (`monitor.rs`) calls the policy's `check` directly, bypassing `authorize()`. It filters rows, it is
/// not a requirement, so analyze does not list it and this test does not see it.
#[test]
fn securables_equal_what_a_run_authorizes() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::subscriber::set_global_default(Recorder).unwrap());
    let mut compared = 0;
    let mut row_level = BTreeSet::new();
    let mut differences = String::new();
    let dir = tempfile::tempdir().unwrap();
    let dir = dir.path().to_str().unwrap().to_owned();
    let extra = DIRECTORY_CASES
        .iter()
        .map(|(label, setup, sql)| ((*label).to_owned(), *setup, sql.replace("{dir}", &dir)));
    let cases = corpus()
        .into_iter()
        .map(|(label, sql)| (label, &[][..], sql))
        .chain(extra);
    for (i, (label, setup, sql)) in cases.enumerate() {
        let principal = format!("prop{i}");
        let s = Session::builder()
            .principal(Principal::new(&principal, PrincipalKind::User))
            .build()
            .unwrap();
        corpus_session(&s);
        for sql in setup {
            block_on(s.sql(sql)).unwrap();
        }
        let must_run = DIRECTORY_CASES.iter().any(|(l, _, _)| *l == label);
        // The setup's own questions are not the statement's.
        asked_by(&principal);
        let Ok(a) = block_on(s.analyze(&sql)) else {
            continue;
        };
        let mut analyze_asked = asked_by(&principal);
        // Review D1: before it lists a `CASCADE`'s table directories, analyze asks `BROWSE` on the
        // namespace (the right to list its tables), which the run does not ask and analyze does not
        // list: the one question it may ask beyond the run's.
        if a_cascade(&sql) {
            analyze_asked.retain(|(p, s)| !(p == "BROWSE" && s.starts_with("namespace ")));
        }
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
                    warehouse_free(&u.securable.to_string().to_lowercase()),
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
                if matches!(label.as_str(), "Update" | "Delete" | "DirReplaceTable")
                    && e.error_class().name() == "UNSUPPORTED_FEATURE.TABLE_OPERATION" => {}
            Err(e) if must_run => panic!("{label}: {sql}: {e}"),
            Err(_) => continue,
        }
        let ran = asked_by(&principal);
        compared += 1;
        if matches!(label.as_str(), "Update" | "Delete") {
            row_level.insert(label.clone());
        }
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
    // At least one `UPDATE` and one `DELETE` were compared (review m2).
    assert_eq!(
        row_level.into_iter().collect::<Vec<_>>(),
        ["Delete", "Update"]
    );
    insta::assert_snapshot!(differences);
}

/// Whether `sql` is a `DROP NAMESPACE ... CASCADE`, whose analysis asks `BROWSE` on the namespace (D1).
fn a_cascade(sql: &str) -> bool {
    let sql = sql.to_ascii_uppercase();
    sql.starts_with("DROP NAMESPACE")
        || sql.starts_with("DROP SCHEMA")
        || sql.starts_with("DROP DATABASE")
}

/// Statements of DDL whose run asks about a directory (T063 N1b review M1): (label, setup, statement), each of
/// which succeeds on the corpus session after its setup (`{dir}` is a new temporary directory), so the
/// property test compares what analyze lists with what the run asks for each. `REPLACE TABLE` of the
/// session catalog's file tables is `UNSUPPORTED_FEATURE.TABLE_OPERATION` today (asked before any
/// directory), so its case is compared through that error. `ADD PARTITION` and `DROP PARTITION` are not
/// listed yet (`analyze/dirs.rs`): the snapshot pins their difference.
const DIRECTORY_CASES: &[(&str, &[&str], &str)] = &[
    ("DirCreateTable", &[], "CREATE TABLE c_new (a INT)"),
    (
        "DirCreateTableIn",
        &[],
        "CREATE TABLE d.c_new (a INT, k INT) USING parquet PARTITIONED BY (k)",
    ),
    (
        "DirCreateExternal",
        &[],
        "CREATE TABLE c_ext (a INT) USING parquet LOCATION '{dir}/e'",
    ),
    (
        "DirCreatePathOption",
        &[],
        "CREATE TABLE c_opt (a INT) USING parquet OPTIONS (path '{dir}/o')",
    ),
    (
        "DirCreateStoreOptions",
        &[],
        "CREATE TABLE c_so (a INT) USING parquet OPTIONS ('fs.s3a.region' = 'eu-west-1')",
    ),
    ("DirCtas", &[], "CREATE TABLE c_ctas AS SELECT * FROM t"),
    (
        "DirCtasExternal",
        &[],
        "CREATE TABLE c_ctas_e USING parquet LOCATION '{dir}/c' AS SELECT * FROM t",
    ),
    ("DirReplaceTable", &[], "CREATE OR REPLACE TABLE t (a INT)"),
    ("DirRenameTable", &[], "ALTER TABLE u RENAME TO u_renamed"),
    (
        "DirDropNamespaceCascade",
        &[
            "CREATE TABLE d.dt (a INT)",
            "CREATE TABLE d.da (a INT)",
            "CREATE VIEW d.dv AS SELECT 1 AS a",
        ],
        "DROP NAMESPACE d CASCADE",
    ),
    (
        "DirAddPartition",
        &["CREATE TABLE p (a INT, k INT) USING parquet PARTITIONED BY (k)"],
        "ALTER TABLE p ADD PARTITION (k = 1)",
    ),
    (
        "DirDropPartition",
        &[
            "CREATE TABLE p (a INT, k INT) USING parquet PARTITIONED BY (k)",
            "ALTER TABLE p ADD PARTITION (k = 1)",
        ],
        "ALTER TABLE p DROP PARTITION (k = 1)",
    ),
    (
        "DirSetPartitionLocation",
        &[
            "CREATE TABLE p (a INT, k INT) USING parquet PARTITIONED BY (k)",
            "ALTER TABLE p ADD PARTITION (k = 1)",
        ],
        "ALTER TABLE p PARTITION (k = 1) SET LOCATION '{dir}/l'",
    ),
];

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

/// Whether `name` resolves to a table or view in `s`.
fn resolves(s: &Session, name: &str) -> bool {
    block_on(async { s.table(name).collect().await }).is_ok()
}

/// `EXPLAIN FORMAT JSON CREATE TABLE … AS SELECT` lists the target and its reads, and creates and
/// writes nothing.
#[test]
fn explain_format_json_ctas_creates_nothing() {
    let s = corpus_ready();
    block_on(s.sql("INSERT INTO t VALUES (1, 2, 3)")).unwrap();
    let inner = "CREATE TABLE c AS SELECT a, k FROM t";
    let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
    assert_valid(&json);
    assert_eq!(json, block_on(s.analyze(inner)).unwrap().to_json());
    assert!(json.contains("\"statement_class\":\"DDL\""), "{json}");
    assert!(
        json.contains("table `spark_catalog`.`default`.`c`"),
        "{json}"
    );
    assert!(
        json.contains("table `spark_catalog`.`default`.`t`"),
        "{json}"
    );
    assert!(!resolves(&s, "c"));
    // The name is still free: the real statement then runs.
    block_on(async { s.sql(inner).await?.collect().await }).unwrap();
    assert_eq!(row_count(&s, "c"), 1);
}

/// A temporary view is neither created by `EXPLAIN FORMAT JSON CREATE TEMPORARY VIEW`, nor replaced
/// by `… OR REPLACE`; a read through an existing temporary view lists the view's base table.
#[test]
fn explain_format_json_temp_views() {
    let s = corpus_ready();
    let inner = "CREATE TEMPORARY VIEW tv AS SELECT a FROM t";
    let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
    assert_valid(&json);
    assert_eq!(json, block_on(s.analyze(inner)).unwrap().to_json());
    assert!(!resolves(&s, "tv"));
    // `v1` keeps its two columns.
    let inner = "CREATE OR REPLACE TEMPORARY VIEW v1 AS SELECT a FROM t";
    let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
    assert_eq!(json, block_on(s.analyze(inner)).unwrap().to_json());
    let schema = block_on(async { s.table("v1").schema().await }).unwrap();
    assert_eq!(schema.fields().len(), 2);
    // A read through a temporary view: `READ`, the view expanded to its base table.
    let inner = "SELECT a FROM v1 WHERE k > 0";
    let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
    assert_valid(&json);
    assert_eq!(json, block_on(s.analyze(inner)).unwrap().to_json());
    assert!(json.contains("\"statement_class\":\"READ\""), "{json}");
    assert!(
        json.contains("table `spark_catalog`.`default`.`t`"),
        "{json}"
    );
}

/// `EXPLAIN FORMAT JSON CACHE TABLE` caches nothing.
#[test]
fn explain_format_json_cache_table_caches_nothing() {
    let s = corpus_ready();
    for inner in [
        "CACHE TABLE t",
        "CACHE LAZY TABLE t",
        "CACHE TABLE ct AS SELECT a FROM t",
    ] {
        let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
        assert_valid(&json);
        assert_eq!(
            json,
            block_on(s.analyze(inner)).unwrap().to_json(),
            "{inner}"
        );
        assert!(!block_on(s.table("t").is_cached()).unwrap(), "{inner}");
    }
    assert!(!resolves(&s, "ct"));
}

/// Nested `EXPLAIN`: the explained statement is the inner `EXPLAIN`, analyzed as `analyze` analyzes it.
/// An `EXPLAIN` in any mode but `ANALYZE` is a `READ` of kind `PLAN` that lists no write targets (§3.3.1,
/// `EXPLAIN_KINDS`); nothing runs at either level.
#[test]
fn explain_format_json_of_an_explain() {
    let s = corpus_ready();
    for inner in [
        "EXPLAIN SELECT a FROM t",
        "EXPLAIN EXTENDED SELECT a FROM t",
        "EXPLAIN FORMAT JSON SELECT a FROM t",
        "EXPLAIN FORMAT JSON EXPLAIN FORMAT JSON SELECT 1",
        "EXPLAIN FORMAT JSON INSERT INTO u SELECT * FROM t",
    ] {
        let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
        assert_valid(&json);
        assert_eq!(
            json,
            block_on(s.analyze(inner)).unwrap().to_json(),
            "{inner}"
        );
        assert!(
            json.contains("\"statement_class\":\"READ\""),
            "{inner}: {json}"
        );
        assert!(json.contains("\"explain\":\"PLAN\""), "{inner}: {json}");
        assert!(
            !json.contains("\"privilege\":\"INSERT\""),
            "{inner}: {json}"
        );
    }
    let json = explain_json(&s, "EXPLAIN FORMAT JSON EXPLAIN SELECT a FROM t");
    assert!(
        json.contains("table `spark_catalog`.`default`.`t`"),
        "{json}"
    );
    // One level up, the explained statement is `EXPLAIN FORMAT JSON EXPLAIN ANALYZE …`: a `READ`; nothing runs.
    let json = explain_json(
        &s,
        "EXPLAIN FORMAT JSON EXPLAIN FORMAT JSON EXPLAIN ANALYZE INSERT INTO u SELECT * FROM t",
    );
    assert!(json.contains("\"statement_class\":\"READ\""), "{json}");
    assert_eq!(row_count(&s, "u"), 0);
}

/// `EXPLAIN FORMAT JSON SET` leaves the session conf as it was.
#[test]
fn explain_format_json_set_changes_no_conf() {
    let s = corpus_ready();
    let keys = ["spark.sql.ansi.enabled", "spark.sql.shuffle.partitions"];
    let before = keys.map(|k| s.conf().get(k));
    for inner in [
        "SET spark.sql.ansi.enabled = false",
        "SET spark.sql.ansi.enabled = true",
        "SET spark.sql.shuffle.partitions = 7",
        "RESET",
    ] {
        let json = explain_json(&s, &format!("EXPLAIN FORMAT JSON {inner}"));
        assert_valid(&json);
        assert_eq!(
            json,
            block_on(s.analyze(inner)).unwrap().to_json(),
            "{inner}"
        );
    }
    assert_eq!(keys.map(|k| s.conf().get(k)), before);
    assert_ne!(before[1].as_deref(), Some("7"));
    // The real `SET` does change it, so the check above can fail.
    block_on(s.sql("SET spark.sql.shuffle.partitions = 7")).unwrap();
    assert_eq!(
        s.conf().get("spark.sql.shuffle.partitions").as_deref(),
        Some("7")
    );
}

/// The directory securables of `a`, warehouse-free: (privilege, display).
fn directories(a: &krishiv_session::Analysis) -> Vec<String> {
    a.info
        .securables
        .iter()
        .filter(|u| matches!(u.securable, Securable::Path(_)))
        .map(|u| {
            format!(
                "{} {}",
                u.privilege,
                warehouse_free(&u.securable.to_string())
            )
        })
        .collect()
}

/// T063 N1b item 1: `DROP TABLE`, `TRUNCATE TABLE` and `INSERT` list the directory a run asks about, after
/// their targets, and only where the run asks: not for a managed table whose directory is gone, an external
/// table's drop, a temporary view or a missing table.
#[test]
fn writes_and_drops_list_the_directory_a_run_asks() {
    let s = corpus_ready();
    let dirs = |sql: &str| directories(&block_on(s.analyze(sql)).unwrap());
    assert_eq!(dirs("DROP TABLE t"), ["MODIFY path 'file:<warehouse>/t'"]);
    assert_eq!(
        dirs("TRUNCATE TABLE t"),
        ["MODIFY path 'file:<warehouse>/t'"]
    );
    assert_eq!(
        dirs("INSERT INTO t SELECT * FROM s"),
        [
            "MODIFY path 'file:<warehouse>/t'",
            "CREATE path 'file:<warehouse>/t'"
        ]
    );
    assert_eq!(
        dirs("INSERT OVERWRITE TABLE t SELECT * FROM s"),
        [
            "MODIFY path 'file:<warehouse>/t'",
            "CREATE path 'file:<warehouse>/t'"
        ]
    );
    // Each clause of a multi-table insert, after the targets.
    let a = block_on(s.analyze("FROM s INSERT INTO t SELECT * INSERT INTO u SELECT *")).unwrap();
    assert_eq!(
        directories(&a),
        [
            "MODIFY path 'file:<warehouse>/t'",
            "CREATE path 'file:<warehouse>/t'",
            "MODIFY path 'file:<warehouse>/u'",
            "CREATE path 'file:<warehouse>/u'"
        ]
    );
    // The targets come first.
    let a = block_on(s.analyze("DROP TABLE t")).unwrap();
    assert!(matches!(
        a.info.securables[0].securable,
        Securable::Table(_)
    ));
    // Nothing a run would not ask: a missing table, a temporary view, a plain EXPLAIN.
    assert!(dirs("DROP TABLE IF EXISTS nope").is_empty());
    assert!(dirs("DROP TABLE v0").is_empty());
    // A multi-table insert with a missing target: the run resolves every target first and fails, so it
    // asks about no directory (review m4).
    assert!(dirs("FROM s INSERT INTO t SELECT * INSERT INTO nope SELECT *").is_empty());
    assert!(dirs("EXPLAIN INSERT INTO t SELECT * FROM s").is_empty());
    // A managed table whose directory is gone: DROP and TRUNCATE ask nothing, INSERT still asks.
    let drop_u = block_on(s.analyze("DROP TABLE u")).unwrap();
    let dir = drop_u
        .info
        .securables
        .iter()
        .find_map(|u| match &u.securable {
            Securable::Path(p) => p.as_plain().map(str::to_owned),
            _ => None,
        })
        .unwrap();
    let dir = dir.strip_prefix("file:").unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(dirs("DROP TABLE u").is_empty());
    assert!(dirs("TRUNCATE TABLE u").is_empty());
    assert_eq!(dirs("INSERT INTO u SELECT * FROM s").len(), 2);
    // An external table: its drop deletes nothing, so asks nothing; its insert writes its location.
    let external = tempfile::tempdir().unwrap();
    let ext = external.path().join("ext");
    block_on(s.sql(&format!(
        "CREATE TABLE e (a INT) USING parquet LOCATION '{}'",
        sql_path(&ext)
    )))
    .unwrap();
    assert!(dirs("DROP TABLE e").is_empty());
    assert_eq!(dirs("INSERT INTO e SELECT a FROM s").len(), 2);
}

/// A Delta table (PR-37a, review n4 of #782): each statement finds it as its run does. `DROP TABLE` of a
/// managed one deletes its directory, so lists it; `TRUNCATE TABLE` is refused by name, so lists
/// nothing; `INSERT` commits to the log and asks about no directory, so its clause lists nothing while
/// the other clauses of a multi-table insert still list theirs.
#[cfg(feature = "delta")]
#[test]
fn a_delta_table_lists_what_its_run_asks() {
    let s = corpus_ready();
    block_on(s.sql("CREATE TABLE d USING delta AS SELECT * FROM s")).unwrap();
    let dirs = |sql: &str| directories(&block_on(s.analyze(sql)).unwrap());
    assert_eq!(dirs("DROP TABLE d"), ["MODIFY path 'file:<warehouse>/d'"]);
    assert!(dirs("TRUNCATE TABLE d").is_empty());
    assert!(dirs("INSERT INTO d SELECT * FROM s").is_empty());
    assert!(dirs("INSERT OVERWRITE TABLE d SELECT * FROM s").is_empty());
    assert_eq!(
        dirs("FROM s INSERT INTO d SELECT * INSERT INTO t SELECT *"),
        [
            "MODIFY path 'file:<warehouse>/t'",
            "CREATE path 'file:<warehouse>/t'"
        ]
    );
    // The runs agree: the insert writes, the truncate is refused, the drop deletes the directory.
    block_on(s.sql("INSERT INTO d SELECT * FROM s")).unwrap();
    assert!(block_on(s.sql("TRUNCATE TABLE d")).is_err());
    block_on(s.sql("DROP TABLE d")).unwrap();
}

/// The directories of DDL (T063 N1b review M1): `CREATE TABLE` (managed and with a location), CTAS,
/// `ALTER TABLE ... RENAME TO` and `DROP NAMESPACE ... CASCADE` list the directories their run asks about,
/// after their targets, in the run's order and in the form its denial names; nothing where the run fails
/// before it asks (the table or the location already exists, `REPLACE`, a namespace that is not dropped
/// with its tables). A partition's `SET LOCATION` asks only its target (the location as written).
#[test]
fn ddl_lists_the_directories_a_run_asks() {
    let s = corpus_ready();
    let dirs = |sql: &str| directories(&block_on(s.analyze(sql)).unwrap());
    let none: [&str; 0] = [];
    assert_eq!(
        dirs("CREATE TABLE c_new (a INT)"),
        ["CREATE path 'file:<warehouse>/c_new'"]
    );
    // The namespace's `CREATE` (the target) comes first.
    let a = block_on(s.analyze("CREATE TABLE c_new (a INT)")).unwrap();
    assert!(matches!(
        a.info.securables[0].securable,
        Securable::Namespace(_)
    ));
    assert_eq!(
        dirs("CREATE TABLE d.c_new (a INT, k INT) USING parquet PARTITIONED BY (k)"),
        ["CREATE path 'file:<warehouse>/d.db/c_new'"]
    );
    assert_eq!(
        dirs("CREATE TABLE c_ctas AS SELECT * FROM t"),
        [
            "MODIFY path 'file:<warehouse>/c_ctas'",
            "CREATE path 'file:<warehouse>/c_ctas'"
        ]
    );
    // The table exists: the run fails (or, `IF NOT EXISTS`, does nothing) before it asks.
    for sql in [
        "CREATE TABLE t (a INT)",
        "CREATE TABLE IF NOT EXISTS t (a INT)",
        "CREATE TABLE t AS SELECT 1 AS a",
        "CREATE TABLE IF NOT EXISTS t AS SELECT 1 AS a",
        // Refused before any directory (`UNSUPPORTED_FEATURE.TABLE_OPERATION`).
        "CREATE OR REPLACE TABLE t (a INT)",
        "REPLACE TABLE t (a INT) USING parquet",
        "CREATE OR REPLACE TABLE c_new AS SELECT 1 AS a",
        // A missing namespace, a managed table without columns, a temporary table.
        "CREATE TABLE nope.c (a INT)",
        "CREATE TABLE c_none USING parquet",
        // A layout the run refuses: every column a partition column.
        "CREATE TABLE c_all (a INT) USING parquet PARTITIONED BY (a)",
        "ALTER TABLE u RENAME TO t",
        "ALTER TABLE nope RENAME TO c_x",
        "ALTER TABLE v RENAME TO c_x",
        "DROP NAMESPACE d",
        "DROP NAMESPACE nope CASCADE",
    ] {
        assert_eq!(dirs(sql), none, "{sql}");
    }
    // A managed table whose directory is already there: `LOCATION_ALREADY_EXISTS` before the question.
    let taken = block_on(s.analyze("DROP TABLE t")).unwrap();
    let t_dir = taken
        .info
        .securables
        .iter()
        .find_map(|u| match &u.securable {
            Securable::Path(p) => p.as_plain().map(str::to_owned),
            _ => None,
        })
        .unwrap();
    let warehouse = std::path::PathBuf::from(
        t_dir
            .strip_prefix("file:")
            .unwrap()
            .strip_suffix("/t")
            .unwrap(),
    );
    std::fs::create_dir(warehouse.join("c_taken")).unwrap();
    assert_eq!(dirs("CREATE TABLE c_taken (a INT)"), none);
    assert_eq!(dirs("CREATE TABLE c_taken AS SELECT 1 AS a"), none);
    // With a location: `CREATE` on it after its `MODIFY` target, as the run asks it.
    let external = tempfile::tempdir().unwrap();
    let ext = external.path().join("e");
    let ext = ext.display();
    assert_eq!(
        dirs(&format!(
            "CREATE TABLE c_e (a INT) USING parquet LOCATION '{ext}'"
        )),
        [
            format!("MODIFY path 'file:{ext}'"),
            format!("CREATE path 'file:{ext}'")
        ]
    );
    assert_eq!(
        dirs(&format!(
            "CREATE TABLE c_e (a INT) USING parquet OPTIONS (path '{ext}/../o')"
        )),
        [format!(
            "CREATE path 'file:{}'",
            external.path().join("o").display()
        )]
    );
    assert_eq!(
        dirs(&format!(
            "CREATE TABLE c_e USING parquet LOCATION '{ext}' AS SELECT * FROM t"
        )),
        [
            format!("MODIFY path 'file:{ext}'"),
            format!("CREATE path 'file:{ext}'")
        ]
    );
    // A relative location is qualified against the working directory, as the run's 42501 names it
    // (review N3); an object store's location is its URL.
    let cwd = std::env::current_dir().unwrap();
    let listed = dirs("CREATE TABLE c_rel (a INT) USING parquet LOCATION 'rel-q3/d'");
    assert_eq!(
        listed.last().map(String::as_str),
        Some(format!("CREATE path 'file:{}/rel-q3/d'", cwd.display()).as_str()),
        "{listed:?}"
    );
    assert_eq!(
        dirs("CREATE TABLE c_s3 (a INT) USING parquet LOCATION 's3a://lake/tables/c_s3'"),
        [
            "MODIFY path 's3a://lake/tables/c_s3'",
            "CREATE path 's3a://lake/tables/c_s3'"
        ]
    );
    // Server-scoped store options (review D2): the run asks `MANAGE` on the cluster after the targets,
    // before the location.
    let a = block_on(s.analyze(
        "CREATE TABLE c_so (a INT) USING parquet OPTIONS ('fs.s3a.region' = 'eu-west-1')",
    ))
    .unwrap();
    let listed: Vec<String> = uses(&a).iter().map(|u| warehouse_free(u)).collect();
    assert_eq!(
        listed,
        [
            "CREATE namespace `spark_catalog`.`default` []",
            "MANAGE system `cluster` []",
            "CREATE path 'file:<warehouse>/c_so' []"
        ]
    );
    // A rename moves a managed table's directory: `MODIFY` on it, then `CREATE` on the new one.
    assert_eq!(
        dirs("ALTER TABLE u RENAME TO u_renamed"),
        [
            "MODIFY path 'file:<warehouse>/u'",
            "CREATE path 'file:<warehouse>/u_renamed'"
        ]
    );
    // `CASCADE` deletes the directory of every managed table in the namespace (not an external
    // table's, not a view's), in the run's order.
    for sql in [
        "CREATE TABLE d.b2 (a INT)".to_owned(),
        "CREATE TABLE d.a1 (a INT)".to_owned(),
        format!("CREATE TABLE d.ext (a INT) USING parquet LOCATION '{ext}'"),
        "CREATE VIEW d.dv AS SELECT 1 AS a".to_owned(),
    ] {
        block_on(s.sql(&sql)).unwrap();
    }
    assert_eq!(
        dirs("DROP NAMESPACE d CASCADE"),
        [
            "MODIFY path 'file:<warehouse>/d.db/a1'",
            "MODIFY path 'file:<warehouse>/d.db/b2'"
        ]
    );
    // A partition's `SET LOCATION` changes the catalog only: its target, nothing more.
    block_on(s.sql("CREATE TABLE p (a INT, k INT) USING parquet PARTITIONED BY (k)")).unwrap();
    block_on(s.sql("ALTER TABLE p ADD PARTITION (k = 1)")).unwrap();
    assert_eq!(
        dirs(&format!(
            "ALTER TABLE p PARTITION (k = 1) SET LOCATION '{ext}'"
        )),
        [format!("MODIFY path 'file:{ext}'")]
    );
}

/// Review D3: analyze never makes the runtime's own warehouse directory; it lists the path a run then
/// makes and asks about.
#[test]
fn analyze_makes_no_warehouse_directory() {
    let s = Session::new().unwrap();
    let a = block_on(s.analyze("CREATE TABLE w_new (a INT)")).unwrap();
    let listed: Vec<String> = a
        .info
        .securables
        .iter()
        .filter_map(|u| match &u.securable {
            Securable::Path(p) => p.as_plain().map(str::to_owned),
            _ => None,
        })
        .collect();
    assert_eq!(listed.len(), 1, "{listed:?}");
    let dir = std::path::PathBuf::from(listed[0].strip_prefix("file:").unwrap());
    let warehouse = dir.parent().unwrap();
    assert!(
        warehouse
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("krishiv-warehouse-"),
        "{listed:?}"
    );
    assert!(!warehouse.exists(), "analyze made {}", warehouse.display());
    // The run makes the same directory.
    block_on(s.sql("CREATE TABLE w_new (a INT)")).unwrap();
    assert!(dir.is_dir(), "{}", dir.display());
}

/// T063 N1b item 3: a write of one query has that query's output schema and lineage (the query's own
/// columns, not the target table's); a multi-table insert and an EXPLAIN of a write have none.
#[test]
fn writes_of_one_query_give_its_schema_and_lineage() {
    let s = corpus_ready();
    for sql in [
        "INSERT INTO u SELECT a + 1 AS x, b, k FROM t",
        "INSERT OVERWRITE TABLE u SELECT a + 1 AS x, b, k FROM t",
        "CREATE TABLE c2 AS SELECT a + 1 AS x, b, k FROM t",
    ] {
        let a = block_on(s.analyze(sql)).unwrap();
        assert_valid(&a.to_json());
        let names: Vec<&str> = a.output_schema.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["x", "b", "k"], "{sql}");
        let transforms: Vec<&str> = a.lineage.iter().map(|l| l.transform.name()).collect();
        assert_eq!(transforms, ["expression", "identity", "identity"], "{sql}");
        let json = a.to_json();
        assert!(
            json.contains("{\"ordinal\":0,\"output\":\"x\",\"sources\":[{\"table\":[\"spark_catalog\",\"default\",\"t\"],\"column\":\"a\""),
            "{sql}: {json}"
        );
    }
    for sql in [
        "FROM s INSERT INTO t SELECT * INSERT INTO u SELECT *",
        "EXPLAIN INSERT INTO u SELECT a, b, k FROM t",
    ] {
        let a = block_on(s.analyze(sql)).unwrap();
        assert!(a.lineage.is_empty(), "{sql}");
    }
    let a = block_on(s.analyze("EXPLAIN INSERT INTO u SELECT a, b, k FROM t")).unwrap();
    let names: Vec<&str> = a.output_schema.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["plan"]);
}

/// T063 N1b item 4: every securable carries `display`, the 42501 text of it.
#[test]
fn securables_carry_their_denial_text() {
    let s = corpus_ready();
    let json = block_on(s.analyze("SELECT a FROM t")).unwrap().to_json();
    assert!(
        json.contains("\"display\":\"table `spark_catalog`.`default`.`t`\""),
        "{json}"
    );
}

/// Denies the write privileges (`Modify`, `Create`, `Drop`), or with `all` every privilege, once `on`;
/// counts each question.
struct DenyTargets {
    on: std::sync::atomic::AtomicBool,
    all: bool,
}

impl AccessPolicy for DenyTargets {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(u64::from(self.on.load(Ordering::SeqCst)))
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        _: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let write = matches!(
            privilege,
            Privilege::Modify | Privilege::Create | Privilege::Drop
        );
        let d = if self.on.load(Ordering::SeqCst) && (self.all || write) {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }
}

/// T063 N1b review B2: analyze (and `EXPLAIN FORMAT JSON`) reveals nothing about a write target the caller
/// may not write. Under a policy that denies the target (and under deny-all), `DROP TABLE`, `TRUNCATE
/// TABLE`, every form of `INSERT` (review r2-n2) and the DDL whose directories are listed (`CREATE TABLE`,
/// CTAS, `REPLACE`, `RENAME TO`, `DROP NAMESPACE ... CASCADE`, review M1) give byte-identical output whether `x` is an existing managed table, a
/// managed table whose directory is gone, an external table (whose distinctive location never appears),
/// a missing name, a permanent view or a bucketed table; and the trial writes no audit record (no denial
/// is counted).
#[test]
fn a_denied_target_reveals_nothing() {
    use krishiv_session::auth::{DeniedCounts, Surface};
    let external = tempfile::tempdir().unwrap();
    let location = external.path().join("distinctive-location-q7");
    let setups: Vec<(&str, Vec<String>)> = vec![
        (
            "managed",
            vec!["CREATE TABLE x (a INT, b INT, k INT)".to_owned()],
        ),
        (
            "managed, no directory",
            vec!["CREATE TABLE x (a INT, b INT, k INT)".to_owned()],
        ),
        (
            "external",
            vec![format!(
                "CREATE TABLE x (a INT, b INT, k INT) USING parquet LOCATION '{}'",
                sql_path(&location)
            )],
        ),
        ("missing", vec![]),
        (
            "view",
            vec!["CREATE VIEW x AS SELECT 1 AS a, 2 AS b, 3 AS k".to_owned()],
        ),
        (
            "bucketed",
            vec![
                "CREATE TABLE x (a INT, b INT, k INT) USING parquet CLUSTERED BY (a) INTO 2 BUCKETS"
                    .to_owned(),
            ],
        ),
    ];
    // Review r2-n2: the other INSERT forms; review M1: the DDL whose directories are listed. `DROP
    // NAMESPACE default CASCADE` varies with `x` too (its run asks about every managed table in it
    // before it refuses the default namespace).
    let statements = [
        "DROP TABLE x",
        "TRUNCATE TABLE x",
        "INSERT INTO x VALUES (1, 2, 3)",
        "INSERT OVERWRITE TABLE x VALUES (1, 2, 3)",
        "INSERT INTO x SELECT 1, 2, 3",
        "FROM (SELECT 1 AS a, 2 AS b, 3 AS k) q INSERT INTO x SELECT * INSERT OVERWRITE TABLE x SELECT *",
        "CREATE TABLE x (a INT, b INT, k INT)",
        "CREATE TABLE IF NOT EXISTS x (a INT, b INT, k INT)",
        "CREATE TABLE x AS SELECT 1 AS a, 2 AS b, 3 AS k",
        "CREATE OR REPLACE TABLE x (a INT, b INT, k INT)",
        "ALTER TABLE x RENAME TO x_renamed",
        "DROP NAMESPACE default CASCADE",
    ]
    .map(str::to_owned);
    // Review N2: a location the caller supplies is not echoed under a denied namespace `CREATE`.
    let elsewhere = external.path().join("distinctive-elsewhere-q8");
    let elsewhere = elsewhere.display();
    let statements: Vec<String> = statements
        .into_iter()
        .chain([
            format!("CREATE TABLE x (a INT, b INT, k INT) USING parquet LOCATION '{elsewhere}'"),
            format!("CREATE TABLE x (a INT, b INT, k INT) USING parquet OPTIONS (path '{elsewhere}')"),
            format!(
                "CREATE TABLE x USING parquet LOCATION '{elsewhere}' AS SELECT 1 AS a, 2 AS b, 3 AS k"
            ),
        ])
        .collect();
    for all in [false, true] {
        let mut seen: Vec<Option<(String, String)>> = vec![None; statements.len()];
        for (scenario, setup) in &setups {
            let warehouse = tempfile::tempdir().unwrap();
            let policy = Arc::new(DenyTargets {
                on: std::sync::atomic::AtomicBool::new(false),
                all,
            });
            let s = Session::builder()
                .principal(Principal::new("ana", PrincipalKind::User))
                .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
                .config(
                    "spark.sql.warehouse.dir",
                    warehouse.path().to_str().unwrap(),
                )
                .build()
                .unwrap();
            for sql in setup {
                block_on(s.sql(sql)).unwrap();
            }
            if *scenario == "managed, no directory" {
                std::fs::remove_dir_all(warehouse.path().join("x")).unwrap();
            }
            if *scenario == "managed" {
                // The gate opens for an allowed target: the directory is listed.
                for (sql, listed) in [
                    ("DROP TABLE x", 1),
                    ("ALTER TABLE x RENAME TO x_renamed", 2),
                    ("DROP NAMESPACE default CASCADE", 1),
                ] {
                    let a = block_on(s.analyze(sql)).unwrap();
                    assert_eq!(directories(&a).len(), listed, "{scenario}: {sql}");
                }
            }
            if *scenario == "missing" {
                for (sql, listed) in [
                    ("CREATE TABLE x (a INT, b INT, k INT)", 1),
                    ("CREATE TABLE x AS SELECT 1 AS a, 2 AS b, 3 AS k", 2),
                ] {
                    let a = block_on(s.analyze(sql)).unwrap();
                    assert_eq!(directories(&a).len(), listed, "{scenario}: {sql}");
                }
            }
            policy.on.store(true, Ordering::SeqCst);
            let counts = Arc::new(DeniedCounts::new());
            assert!(s.count_denials(Arc::clone(&counts), Surface::Connect));
            for (i, sql) in statements.iter().enumerate() {
                let analysis = block_on(s.analyze(sql)).unwrap();
                let analyzed = analysis.to_json();
                let explained = explain_json(&s, &format!("EXPLAIN FORMAT JSON {sql}"));
                // A `LOCATION` is a target (`MODIFY`, the caller's own text): listed, nothing more.
                let own = sql.contains("distinctive-elsewhere");
                for json in [&analyzed, &explained] {
                    assert!(
                        own || !json.contains("\"path\""),
                        "{scenario} {sql}: {json}"
                    );
                    assert!(
                        !json.contains("distinctive-location"),
                        "{scenario} {sql}: {json}"
                    );
                }
                assert!(
                    analysis
                        .info
                        .securables
                        .iter()
                        .filter(|u| matches!(u.securable, Securable::Path(_)))
                        .all(|u| u.privilege == Privilege::Modify && sql.contains("LOCATION")),
                    "{scenario} {sql}: {:?}",
                    uses(&analysis)
                );
                match &seen[i] {
                    None => seen[i] = Some((analyzed, explained)),
                    Some((a, e)) => {
                        assert_eq!(&analyzed, a, "all={all} {scenario}: {sql}");
                        assert_eq!(&explained, e, "all={all} {scenario}: {sql}");
                    }
                }
            }
            assert!(
                counts.counts().iter().all(|c| c.count == 0),
                "all={all} {scenario}: analyze counted a denial: {:?}",
                counts.counts()
            );
        }
    }
}

/// Allows everything but `BROWSE` on a namespace (and `SELECT`, its fallback): `DROP` without listing.
struct NoBrowse;

impl AccessPolicy for NoBrowse {
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
        {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }
}

/// Review D1: `DROP NAMESPACE ... CASCADE` lists its tables' directories only to a caller who may list
/// the namespace's tables (`BROWSE`, as `SHOW TABLES IN` asks): under a policy that grants `DROP` but
/// not `BROWSE`, the output is the same whether the namespace holds tables or not, and the trial
/// counts no denial.
#[test]
fn cascade_lists_no_directory_without_browse() {
    use krishiv_session::auth::{DeniedCounts, Surface};
    let sql = "DROP NAMESPACE d CASCADE";
    let mut seen: Option<(String, String)> = None;
    for tables in [false, true] {
        let warehouse = tempfile::tempdir().unwrap();
        let s = Session::builder()
            .principal(Principal::new("ana", PrincipalKind::User))
            .access_policy(Arc::new(NoBrowse))
            .config(
                "spark.sql.warehouse.dir",
                warehouse.path().to_str().unwrap(),
            )
            .build()
            .unwrap();
        block_on(s.sql("CREATE NAMESPACE d")).unwrap();
        if tables {
            block_on(s.sql("CREATE TABLE d.a1 (a INT)")).unwrap();
            block_on(s.sql("CREATE TABLE d.b2 (a INT)")).unwrap();
        }
        // The policy does refuse the listing.
        let listed = block_on(async { s.sql("SHOW TABLES IN d").await?.collect().await });
        assert!(listed.is_err(), "tables={tables}");
        let counts = Arc::new(DeniedCounts::new());
        assert!(s.count_denials(Arc::clone(&counts), Surface::Connect));
        let a = block_on(s.analyze(sql)).unwrap();
        assert!(
            directories(&a).is_empty(),
            "tables={tables}: {:?}",
            uses(&a)
        );
        let analyzed = a.to_json();
        let explained = explain_json(&s, &format!("EXPLAIN FORMAT JSON {sql}"));
        assert!(!analyzed.contains("\"path\""), "{analyzed}");
        match &seen {
            None => seen = Some((analyzed, explained)),
            Some((a, e)) => {
                assert_eq!(&analyzed, a);
                assert_eq!(&explained, e);
            }
        }
        assert!(
            counts.counts().iter().all(|c| c.count == 0),
            "tables={tables}: analyze counted a denial: {:?}",
            counts.counts()
        );
    }
}

/// Denies `Modify` on paths and records every path the policy is asked about.
#[derive(Default)]
struct RecordPaths(std::sync::Mutex<Vec<String>>);

impl AccessPolicy for RecordPaths {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let path = matches!(securable, Securable::Path(_));
        if path {
            self.0.lock().unwrap().push(securable.to_string());
        }
        let d = if path && privilege == Privilege::Modify {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(d)))
    }
}

/// Review m2: under a warehouse reached through a link, analyze lists the path as the caller wrote it
/// (the text the run's 42501 names), while the policy is asked about it with the link resolved.
#[cfg(unix)]
#[test]
fn a_linked_warehouse_lists_the_path_as_written() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real");
    std::fs::create_dir(&real).unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let policy = Arc::new(RecordPaths::default());
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .config("spark.sql.warehouse.dir", link.to_str().unwrap())
        .build()
        .unwrap();
    block_on(s.sql("CREATE TABLE w (a INT)")).unwrap();
    let real = std::fs::canonicalize(&real).unwrap();
    for sql in ["INSERT INTO w VALUES (1)", "DROP TABLE w"] {
        let a = block_on(s.analyze(sql)).unwrap();
        let listed: Vec<String> = a
            .info
            .securables
            .iter()
            .filter(|u| matches!(u.securable, Securable::Path(_)))
            .map(|u| u.securable.to_string())
            .collect();
        assert!(!listed.is_empty(), "{sql}");
        policy.0.lock().unwrap().clear();
        let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        let text = e.to_string();
        assert!(
            text.contains(&format!(" {}", listed[0])),
            "{sql}: {text}\n{listed:?}"
        );
        assert!(
            listed[0].contains(link.to_str().unwrap()),
            "{sql}: {listed:?}"
        );
        let asked = policy.0.lock().unwrap().clone();
        assert!(
            asked.iter().any(|p| p.contains(real.to_str().unwrap())),
            "{sql}: the policy saw {asked:?}"
        );
    }
}

/// Review m1: an external table on an object store lists the store URL the run's 42501 names (locating
/// it builds the store, no I/O), not a `file:` path.
#[cfg(feature = "cloud")]
#[test]
fn an_object_store_table_lists_its_url() {
    let admin = Session::builder()
        .config("fs.s3a.region", "us-east-1")
        .build()
        .unwrap();
    block_on(admin.sql("CREATE TABLE o (a INT) USING parquet LOCATION 's3://lake/tables/o'"))
        .unwrap();
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(DenyPaths))
        .config("fs.s3a.region", "us-east-1")
        .runtime(Arc::clone(admin.runtime()))
        .build()
        .unwrap();
    let sql = "INSERT INTO o VALUES (1)";
    let a = block_on(s.analyze(sql)).unwrap();
    let listed: Vec<String> = a
        .info
        .securables
        .iter()
        .filter(|u| matches!(u.securable, Securable::Path(_)))
        .map(|u| u.securable.to_string())
        .collect();
    assert_eq!(listed.len(), 2, "{listed:?}");
    assert!(listed[0].contains("s3://lake/tables/o"), "{listed:?}");
    assert!(!listed[0].contains("file:"), "{listed:?}");
    let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    let text = e.to_string();
    assert!(
        text.contains(&format!(" {}", listed[0])),
        "{text}\n{listed:?}"
    );
}
