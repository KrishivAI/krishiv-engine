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

/// A statement whose reads are not whole queries is `partial`: its targets, `UNSUPPORTED_READ_SHAPE`, and
/// no analysis (`statement_info` refuses it).
#[test]
fn unsupported_read_shapes_are_partial() {
    let s = corpus_ready();
    for sql in [
        "UPDATE t SET a = 1 WHERE b = 2",
        "FROM s INSERT INTO t SELECT a, b, k INSERT INTO u SELECT a, b, k",
        "EXECUTE IMMEDIATE 'SELECT 1'",
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
    let a = block_on(s.analyze("UPDATE t SET a = 1")).unwrap();
    let uses: Vec<String> = a
        .info
        .securables
        .iter()
        .map(|u| format!("{} {}", u.privilege, u.securable))
        .collect();
    assert_eq!(uses, ["MODIFY table `spark_catalog`.`default`.`t`"]);
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
        if block_on(s.sql(&sql)).is_err() {
            continue;
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
