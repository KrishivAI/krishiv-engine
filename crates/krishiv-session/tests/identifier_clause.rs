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
    // The permanent table's entry is held until `UNCACHE` removes it.
    assert!(s.cache_reserved() > 0);
    run(&s, "UNCACHE TABLE t");
    assert_eq!(s.cache_reserved(), 0);
    // A write drops the cached rows of its table.
    run(&s, "INSERT INTO t VALUES (4)");
    assert_eq!(first(&s, "SELECT sum(a) FROM t"), "10");
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

/// A keyword before a parenthesis is not an alias: an `IDENTIFIER` with a qualified name there is
/// an ordinary column reference (Spark 4.2.0 accepts each of these), and `JOIN .. USING (..)` takes
/// one-part names.
#[test]
fn a_keyword_before_a_parenthesis_is_not_an_alias_column_list() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES (1), (2) AS t(c)",
    );
    run(
        &s,
        "CREATE TEMPORARY VIEW u AS SELECT * FROM VALUES (1), (3) AS u(c)",
    );
    for q in [
        "SELECT count(IDENTIFIER('t.c')) FROM t",
        "SELECT c FROM t WHERE abs(c) IN (IDENTIFIER('t.c'))",
        "SELECT sum(c) OVER (PARTITION BY IDENTIFIER('t.c')) FROM t",
        "SELECT c FROM t WHERE (IDENTIFIER('t.c') > 0)",
        "SELECT t.c FROM t JOIN u ON (IDENTIFIER('t.c') = IDENTIFIER('u.c'))",
        "SELECT c FROM t UNION (SELECT IDENTIFIER('u.c') FROM u)",
    ] {
        run(&s, q);
    }
    assert_eq!(
        class(&s, "SELECT * FROM t JOIN u USING (IDENTIFIER('a.b'))"),
        "IDENTIFIER_TOO_MANY_NAME_PARTS"
    );
}

#[test]
fn positions_that_take_no_clause_are_refused_as_spark_refuses_them() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    // An alias or a window name is one part.
    for q in [
        "SELECT row_number() OVER IDENTIFIER('x.win') FROM VALUES (1) AS T(c1) WINDOW win AS (ORDER BY c1)",
        "SELECT * FROM VALUES (1, 2) AS IDENTIFIER('schema.table')(c1, c2)",
        "SELECT 1 AS IDENTIFIER('col1.col2')",
        // `tableAlias` is `AS? strictIdentifier identifierList?`: no `AS`, and a CTE's column list
        // (Spark 4.2.0: IDENTIFIER_TOO_MANY_NAME_PARTS for each).
        "SELECT * FROM VALUES (1) t(IDENTIFIER('a.b'))",
        "WITH t(IDENTIFIER('a.b')) AS (SELECT 1) SELECT * FROM t",
        "SELECT * FROM VALUES (1) AS t(IDENTIFIER('a.b'))",
    ] {
        assert_eq!(class(&s, q), "IDENTIFIER_TOO_MANY_NAME_PARTS", "{q}");
    }
    // Hint arguments, `SHOW <scope> FUNCTIONS` and `EXTRACT(<field> FROM ...)` take no clause.
    for q in [
        "SELECT /*+ IDENTIFIER('BROADCAST')(t) */ 1",
        "SHOW IDENTIFIER('USER') FUNCTIONS",
        "SELECT EXTRACT(IDENTIFIER('YEAR') FROM DATE'2024-01-15')",
    ] {
        assert_eq!(class(&s, q), "PARSE_SYNTAX_ERROR", "{q}");
    }
    // A one-part alias still resolves.
    assert_eq!(first(&s, "SELECT 1 AS IDENTIFIER('one')"), "1");
}

#[test]
fn braces_substitute_configuration_values_before_parsing() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET hivevar:colname = 'c'");
    assert_eq!(
        first(
            &s,
            "SELECT IDENTIFIER(${colname} || '_1') FROM VALUES(1) AS T(c_1)"
        ),
        "1"
    );
    assert_eq!(first(&s, "SELECT ${hivevar:colname}"), "c");
    // An unknown name stays as written; switching the feature off leaves everything as written.
    assert_eq!(first(&s, "SELECT '${nothing}'"), "${nothing}");
    run(&s, "SET spark.sql.variable.substitute = false");
    assert_eq!(first(&s, "SELECT '${colname}'"), "${colname}");
    // The switch is a case-insensitive boolean.
    run(&s, "SET spark.sql.variable.substitute = true");
    assert_eq!(first(&s, "SELECT ${colname}"), "c");
    run(&s, "SET spark.sql.variable.substitute = FALSE");
    assert_eq!(first(&s, "SELECT '${colname}'"), "${colname}");
}

fn names(s: &Session, sql: &str) -> Vec<String> {
    block_on(async { s.sql(sql).await?.schema().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// identifier-clause 177 and 186: a select item written `IDENTIFIER(..)` is named by the SQL of
/// what it resolves to; the same name written plainly keeps its short name, and an attribute
/// keeps its own name either way.
#[test]
fn a_select_item_written_identifier_is_named_by_what_it_resolves_to() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "DECLARE my_var = 'new_value'");
    assert_eq!(
        names(&s, "SELECT IDENTIFIER('my_var'), my_var"),
        [
            "variablereference(system.session.my_var='new_value')",
            "my_var"
        ]
    );
    let structs = "FROM VALUES (named_struct('c2', 42)) AS T(c1)";
    assert_eq!(
        names(&s, &format!("SELECT IDENTIFIER('c1.c2'), c1.c2 {structs}")),
        ["c1.c2", "c2"]
    );
    assert_eq!(
        names(
            &s,
            &format!(
                "EXECUTE IMMEDIATE 'SELECT IDENTIFIER(:col1 ''.c2'') {}' USING 'c1' AS col1",
                structs.replace('\'', "''")
            )
        ),
        ["c1.c2"]
    );
    assert_eq!(
        names(&s, "SELECT IDENTIFIER('a') FROM VALUES (1) AS T(a)"),
        ["a"]
    );
    // a multibyte literal before the clause, two clauses in one item list, and a clause in a
    // derived table: each substituted name is found at its own byte span
    assert_eq!(
        names(
            &s,
            &format!("SELECT 'é—x' AS k, IDENTIFIER('c1.c2'), IDENTIFIER('c1') {structs}")
        ),
        ["k", "c1.c2", "c1"]
    );
    assert_eq!(
        names(
            &s,
            &format!("SELECT * FROM (SELECT IDENTIFIER('c1.c2'), c1.c2 {structs})")
        ),
        ["c1.c2", "c2"]
    );
    assert_eq!(
        names(
            &s,
            &format!("WITH w AS (SELECT IDENTIFIER('c1.c2') {structs}) SELECT * FROM w")
        ),
        ["c1.c2"]
    );
    // a clause followed by fields is named by the whole path, as Spark 4.2.0 names it; the
    // same path written plainly keeps its last part
    let nested = "FROM VALUES (named_struct('c2', named_struct('c3', 42))) AS T(c1)";
    assert_eq!(
        names(
            &s,
            &format!(
                "SELECT IDENTIFIER('c1').c2, IDENTIFIER('c1').c2.c3, IDENTIFIER('c1.c2').c3, \
                 IDENTIFIER('T.c1').c2, (c1).c2 {nested}"
            )
        ),
        ["c1.c2", "c1.c2.c3", "c1.c2.c3", "c1.c2", "c2"]
    );
    // the legacy clause (`spark.sql.legacy.identifierClause`) names the items the same way
    // (`research/spark/identifier-names-outside-query.md`)
    run(&s, "SET spark.sql.legacy.identifierClause = true");
    assert_eq!(
        names(&s, &format!("SELECT IDENTIFIER('c1.c2'), c1.c2 {structs}")),
        ["c1.c2", "c2"]
    );
    assert_eq!(
        names(&s, "SELECT IDENTIFIER('my_var'), my_var"),
        [
            "variablereference(system.session.my_var='new_value')",
            "my_var"
        ]
    );
}

/// Spark 4.2.0 (`research/spark/identifier-names-outside-query.md`): the name of a select item
/// written `IDENTIFIER(..)` depends only on the expression, so commands that take a query name
/// it as a query does.
#[test]
fn commands_over_a_query_name_identifier_items_as_the_query_does() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let structs = "FROM VALUES (named_struct('c2', 42)) AS T(c1)";
    let items = format!("SELECT IDENTIFIER('c1.c2'), c1.c2 AS x {structs}");
    // CTAS
    run(&s, &format!("CREATE TABLE t1 USING parquet AS {items}"));
    assert_eq!(names(&s, "SELECT * FROM t1"), ["c1.c2", "x"]);
    // INSERT ... BY NAME matches the item to the column `c1.c2`
    run(
        &s,
        &format!("INSERT INTO t1 BY NAME SELECT c1.c2 AS x, IDENTIFIER('c1.c2') {structs}"),
    );
    assert_eq!(first(&s, "SELECT count(*) FROM t1 WHERE `c1.c2` = 42"), "2");
    // views: temporary, permanent, and a column list that renames
    run(&s, &format!("CREATE TEMPORARY VIEW v AS {items}"));
    assert_eq!(names(&s, "SELECT * FROM v"), ["c1.c2", "x"]);
    run(&s, &format!("CREATE VIEW pv AS {items}"));
    assert_eq!(names(&s, "SELECT * FROM pv"), ["c1.c2", "x"]);
    run(
        &s,
        &format!("ALTER VIEW pv AS SELECT IDENTIFIER('c1.c2') {structs}"),
    );
    assert_eq!(names(&s, "SELECT * FROM pv"), ["c1.c2"]);
    run(
        &s,
        &format!("CREATE VIEW pv3 (a) AS SELECT IDENTIFIER('c1.c2') {structs}"),
    );
    assert_eq!(names(&s, "SELECT * FROM pv3"), ["a"]);
    // DESCRIBE QUERY and EXPLAIN show the analyzed name
    let described = first(&s, &format!("DESCRIBE QUERY {items}"));
    assert_eq!(described, "c1.c2");
    let explained = first(&s, &format!("EXPLAIN EXTENDED {items}"));
    assert!(explained.contains("c1.c2: int"), "{explained}");
    // two unaliased items: a query keeps both, CTAS refuses the duplicate
    let twice = format!("SELECT IDENTIFIER('c1.c2'), IDENTIFIER('c1.c2') {structs}");
    assert_eq!(names(&s, &twice), ["c1.c2", "c1.c2"]);
    assert_eq!(
        class(&s, &format!("CREATE TABLE t2 USING parquet AS {twice}")),
        "COLUMN_ALREADY_EXISTS"
    );
    // a permanent view with the duplicate is refused as CTAS is
    assert_eq!(
        class(&s, &format!("CREATE VIEW pv4 AS {twice}")),
        "COLUMN_ALREADY_EXISTS"
    );
    // global temporary views, CACHE TABLE ... AS and EXECUTE IMMEDIATE CTAS
    run(&s, &format!("CREATE GLOBAL TEMPORARY VIEW g AS {items}"));
    assert_eq!(names(&s, "SELECT * FROM global_temp.g"), ["c1.c2", "x"]);
    run(&s, &format!("CACHE TABLE cached AS {items}"));
    assert_eq!(names(&s, "SELECT * FROM cached"), ["c1.c2", "x"]);
    run(
        &s,
        &format!(
            "EXECUTE IMMEDIATE 'CREATE TABLE t3 USING parquet AS SELECT IDENTIFIER(:c ''.c2'') {}' USING 'c1' AS c",
            structs.replace('\'', "''")
        ),
    );
    assert_eq!(names(&s, "SELECT * FROM t3"), ["c1.c2"]);
}
