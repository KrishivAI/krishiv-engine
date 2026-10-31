// SPDX-License-Identifier: Apache-2.0
//! `IDENTIFIER(expr)` names (evaluated before the statement is parsed, so access checks see the real
//! names) and `CACHE TABLE`, through `Session::sql`. Classes follow Spark 4.2.0's
//! `identifier-clause.sql` golden.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_session::{
    AccessDecision, AccessPolicy, Principal, PrincipalKind, Privilege, Securable, Session,
};

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn class(s: &Session, sql: &str) -> String {
    block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_err()
        .error_class()
        .name()
        .to_owned()
}

fn first(s: &Session, sql: &str) -> String {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let c = batches[0].column(0);
    match c.data_type() {
        ArrowType::Int64 => c
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0)
            .to_string(),
        ArrowType::Int32 => c
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0)
            .to_string(),
        ArrowType::Utf8View => c.as_string_view().value(0).to_owned(),
        other => panic!("unexpected {other}"),
    }
}

#[test]
fn names_come_from_constant_string_expressions() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE IDENTIFIER('t' || '1') (a INT) USING parquet",
    );
    run(
        &s,
        "INSERT INTO IDENTIFIER('default.' || 't1') VALUES (7), (8)",
    );
    assert_eq!(first(&s, "SELECT sum(a) FROM IDENTIFIER('t1')"), "15");
    // A column, an alias and a function name too; a variable is a constant.
    run(&s, "DECLARE tbl STRING DEFAULT 't1'");
    assert_eq!(
        first(&s, "SELECT max(IDENTIFIER('a')) FROM IDENTIFIER(tbl)"),
        "8"
    );
    assert_eq!(first(&s, "SELECT IDENTIFIER('ab' || 's')(-3)"), "3");
    run(&s, "DROP TABLE IDENTIFIER('t' || '1')");
    // A quoted part stays one name, here a column alias with a space.
    assert_eq!(first(&s, "SELECT 5 AS IDENTIFIER('`my col`')"), "5");
    // A string literal that only looks like the clause is just a string.
    assert_eq!(first(&s, "SELECT 'IDENTIFIER(x)'"), "IDENTIFIER(x)");
}

#[test]
fn the_expression_must_be_a_constant_string() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    assert_eq!(
        class(&s, "SELECT * FROM IDENTIFIER(CAST(NULL AS STRING))"),
        "NOT_A_CONSTANT_STRING.NULL"
    );
    assert_eq!(
        class(&s, "SELECT * FROM IDENTIFIER(1)"),
        "NOT_A_CONSTANT_STRING.WRONG_TYPE"
    );
    assert_eq!(
        class(
            &s,
            "SELECT * FROM IDENTIFIER(substr('HELLO', 1, rand() + 1))"
        ),
        "NOT_A_CONSTANT_STRING.NOT_CONSTANT"
    );
    assert_eq!(
        class(&s, "SELECT * FROM IDENTIFIER((SELECT 't'))"),
        "NOT_A_CONSTANT_STRING.NOT_CONSTANT"
    );
}

/// Denies reading, modifying and creating `secret`; everything else is allowed.
struct NoSecret;

impl AccessPolicy for NoSecret {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<
        'a,
        Result<AccessDecision, krishiv_common::error::KrishivError>,
    > {
        let denied = matches!(securable, Securable::Table(t) if t.name == "secret");
        Box::pin(async move {
            Ok(if denied {
                AccessDecision::Deny {
                    reason: "private".to_owned(),
                }
            } else {
                AccessDecision::Allow
            })
        })
    }
}

#[test]
fn a_dynamic_name_is_authorized_as_the_name_it_evaluates_to() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(NoSecret))
        .build()
        .unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    run(&s, "CREATE TABLE open_t (a INT) USING parquet");
    let denied = "KRISHIV_INSUFFICIENT_PERMISSIONS";
    for sql in [
        "INSERT INTO IDENTIFIER('sec' || 'ret') VALUES (1)",
        "SELECT * FROM IDENTIFIER('sec' || 'ret')",
        "DROP TABLE IDENTIFIER('SEC' || 'RET')",
    ] {
        assert_eq!(class(&s, sql), denied, "{sql}");
    }
    // The same text with an allowed name works, and the denied table was never created.
    run(&s, "INSERT INTO IDENTIFIER('open' || '_t') VALUES (1)");
    assert_eq!(class(&s, "SELECT * FROM secret"), denied);
}

#[test]
fn cache_table_reads_a_relation_into_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    run(&s, "INSERT INTO t VALUES (1), (2), (3)");
    run(&s, "CACHE TABLE t");
    assert_eq!(first(&s, "SELECT sum(a) FROM t"), "6");
    run(
        &s,
        "CACHE LAZY TABLE t OPTIONS ('storageLevel'='DISK_ONLY')",
    );
    // A write drops the cached rows of its table.
    run(&s, "INSERT INTO t VALUES (4)");
    assert_eq!(first(&s, "SELECT sum(a) FROM t"), "10");
    run(&s, "UNCACHE TABLE t");
    // `AS query` makes the temporary view first.
    run(&s, "CACHE TABLE v AS SELECT a * 2 AS b FROM t");
    assert_eq!(first(&s, "SELECT sum(b) FROM v"), "20");
    assert_eq!(
        class(&s, "CACHE TABLE v AS SELECT 1"),
        "TEMP_TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    assert_eq!(class(&s, "CACHE TABLE missing"), "TABLE_OR_VIEW_NOT_FOUND");
    run(&s, "CACHE TABLE IDENTIFIER('v')");
}

#[test]
fn text_that_is_not_a_clause_is_left_to_the_parser() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    // A column list is not an expression: the parser decides (as it did before this change), the
    // evaluation never runs, so there is no `NOT_A_CONSTANT_STRING` or unresolved column.
    assert_eq!(
        class(&s, "CREATE TABLE identifier (a INT) USING parquet"),
        "PARSE_SYNTAX_ERROR"
    );
    // A script's clauses are not rewritten before it runs (its locals are the interpreter's).
    run(&s, "BEGIN DECLARE v INT DEFAULT 1; SELECT v; END");
    // `:IDENTIFIER(...)` is a parameter marker, not a clause.
    assert_eq!(
        class(&s, "SELECT :IDENTIFIER('p') FROM VALUES (1) AS T(c)"),
        "PARSE_SYNTAX_ERROR"
    );
}
