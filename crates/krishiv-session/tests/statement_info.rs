// SPDX-License-Identifier: Apache-2.0
//! `Session::statement_info`: the class of a statement and what it reads,
//! with columns, found by analysis (temporary views expanded, CTEs,
//! reads by path). The 22 TPC-H queries' securables are a checked-in
//! snapshot.
#![cfg(test)]

mod common;

use std::fmt::Write as _;

use futures::executor::block_on;
use krishiv_session::{ExplainKind, Securable, Session, StatementClass};

fn session() -> Session {
    let s = Session::new().unwrap();
    common::register(&s, 0);
    s
}

/// `securable: columns` per read, one per line.
fn reads(s: &Session, sql: &str) -> String {
    let info = block_on(s.statement_info(sql)).unwrap();
    let mut out = String::new();
    for u in &info.securables {
        let columns: Vec<&str> = u.columns.iter().map(String::as_str).collect();
        writeln!(
            out,
            "{} {}: {}",
            u.privilege,
            u.securable,
            columns.join(", ")
        )
        .unwrap();
    }
    out
}

#[test]
fn tpch_securables_with_columns_match_the_snapshot() {
    let s = session();
    let mut out = String::new();
    for q in 1..=22 {
        let sql = common::query(q);
        let info = block_on(s.statement_info(&sql)).unwrap();
        assert_eq!(info.class, StatementClass::Read, "q{q}");
        assert_eq!(info.explain, None);
        writeln!(out, "q{q:02}\n{}", reads(&s, &sql)).unwrap();
    }
    insta::assert_snapshot!(out);
}

#[test]
fn classes_explain_kinds_and_reads_of_statements() {
    let s = session();
    let info = |sql: &str| block_on(s.statement_info(sql)).unwrap();
    let i = info("EXPLAIN ANALYZE SELECT n_name FROM nation");
    assert_eq!(
        (i.class, i.explain, i.securables.len()),
        (StatementClass::Read, Some(ExplainKind::Analyze), 1)
    );
    let i = info("EXPLAIN SELECT 1");
    assert_eq!(
        (i.class, i.explain),
        (StatementClass::Read, Some(ExplainKind::Plan))
    );
    // A write's reads are the reads of its query.
    let i = info("INSERT INTO region SELECT * FROM nation");
    assert_eq!(i.class, StatementClass::Write);
    assert!(
        matches!(&i.securables[..], [u] if matches!(&u.securable, Securable::Table(t) if t.name == "nation"))
    );
    assert_eq!(info("DROP TABLE nation").class, StatementClass::Ddl);
    assert_eq!(info("SET a = b").class, StatementClass::Session);
    assert_eq!(info("CLEAR CACHE").class, StatementClass::Admin);
    // Nothing ran: the statements are only classified.
    assert_eq!(s.conf().get("a"), None);
}

#[test]
fn reads_see_through_temporary_views_and_ctes() {
    let s = session();
    block_on(s.sql("CREATE TEMPORARY VIEW v1 AS SELECT r_name FROM region")).unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW v2 AS SELECT * FROM v1")).unwrap();
    // A temporary view is read with the reader's rights: its tables are
    // the statement's reads.
    assert_eq!(
        reads(&s, "WITH c AS (SELECT * FROM v2) SELECT * FROM c"),
        "SELECT table `spark_catalog`.`default`.`region`: r_name\n"
    );
    // A table read twice is one securable with the union of columns.
    assert_eq!(
        reads(
            &s,
            "CREATE TEMPORARY VIEW w AS SELECT r_name FROM region UNION ALL SELECT r_comment FROM region"
        ),
        "SELECT table `spark_catalog`.`default`.`region`: r_comment, r_name\n"
    );
}

#[test]
fn reads_by_path_are_qualified_paths() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let out = dir.path().join("out");
    block_on(s.table("region").write().parquet(out.to_str().unwrap())).unwrap();
    let sql = format!("SELECT * FROM parquet.`{}/../out`", out.display());
    let text = reads(&s, &sql);
    assert!(
        text.starts_with(&format!("SELECT path 'file:{}'", out.display())),
        "{text}"
    );
}

#[test]
fn statements_without_standalone_reads_are_unsupported() {
    let s = session();
    for sql in [
        "FROM nation INSERT INTO region SELECT n_name",
        "UPDATE region SET r_name = 'x' WHERE r_regionkey = 1",
    ] {
        let e = block_on(s.statement_info(sql)).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{sql}"
        );
    }
    let e = block_on(s.statement_info("SELECT 1; DROP TABLE nation")).unwrap_err();
    assert_eq!(e.error_class().name(), "PARSE_SYNTAX_ERROR");
}
