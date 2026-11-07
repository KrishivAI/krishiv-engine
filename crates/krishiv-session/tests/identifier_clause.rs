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

#[test]
fn positions_that_take_no_clause_are_refused_as_spark_refuses_them() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    // An alias or a window name is one part.
    for q in [
        "SELECT row_number() OVER IDENTIFIER('x.win') FROM VALUES (1) AS T(c1) WINDOW win AS (ORDER BY c1)",
        "SELECT * FROM VALUES (1, 2) AS IDENTIFIER('schema.table')(c1, c2)",
        "SELECT 1 AS IDENTIFIER('col1.col2')",
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

/// Every row of `sql` as its first two columns, as strings.
fn pairs(s: &Session, sql: &str) -> Vec<(String, String)> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let text = |c: &arrow_array::ArrayRef, i: usize| match c.data_type() {
        ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
        ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
        other => panic!("unexpected {other}"),
    };
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let second = if b.num_columns() > 1 {
                text(b.column(1), i)
            } else {
                String::new()
            };
            out.push((text(b.column(0), i), second));
        }
    }
    out
}

/// Spark 4.2.0 (`research/spark/identifier-leftovers.md`): a permanent view keeps its query as
/// written, so `View Text`, `View Original Text` and `SHOW CREATE TABLE` print the
/// `IDENTIFIER(..)` clauses, while the columns are named by what they evaluate to.
#[test]
fn a_view_keeps_its_identifier_clauses_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (c1 STRUCT<c2: INT>, x INT) USING parquet",
    );
    let query = "SELECT IDENTIFIER('c1' || '.c2'), IDENTIFIER(concat('x')) FROM IDENTIFIER('t')";
    run(&s, &format!("CREATE VIEW v1 AS {query}"));
    assert_eq!(names(&s, "SELECT * FROM v1"), ["c1.c2", "x"]);
    let text = |view: &str, key: &str| {
        pairs(&s, &format!("DESCRIBE EXTENDED {view}"))
            .into_iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("no {key} for {view}"))
    };
    assert_eq!(text("v1", "View Text"), query);
    assert_eq!(text("v1", "View Original Text"), query);
    assert_eq!(text("v1", "View Query Output Columns"), "[`c1.c2`, `x`]");
    assert_eq!(
        pairs(&s, "SHOW CREATE TABLE v1")[0].0,
        format!(
            "CREATE VIEW default.v1 (\n  `c1.c2`,\n  x)\nWITH SCHEMA COMPENSATION\nAS {query}\n"
        )
    );
    // The view's name is a clause too, and a clause followed by fields.
    run(
        &s,
        "CREATE VIEW IDENTIFIER('v2') (a COMMENT 'k', b) AS SELECT IDENTIFIER('c1').c2, x FROM t",
    );
    assert_eq!(
        text("v2", "View Text"),
        "SELECT IDENTIFIER('c1').c2, x FROM t"
    );
    // ALTER VIEW ... AS stores the new text as written; spacing inside the clause is kept.
    run(
        &s,
        "CREATE VIEW v6 AS SELECT  IDENTIFIER( 'x' )  FROM t -- trailing",
    );
    assert_eq!(text("v6", "View Text"), "SELECT  IDENTIFIER( 'x' )  FROM t");
    run(&s, "ALTER VIEW v6 AS SELECT IDENTIFIER('c1.c2') FROM t");
    assert_eq!(text("v6", "View Text"), "SELECT IDENTIFIER('c1.c2') FROM t");
    assert_eq!(names(&s, "SELECT * FROM v6"), ["c1.c2"]);
    // A parameter marker is bound into the text before the view is made, as Spark binds it.
    run(
        &s,
        "EXECUTE IMMEDIATE 'CREATE VIEW v4 AS SELECT IDENTIFIER(:p) FROM t' USING 'c1.c2' AS p",
    );
    assert_eq!(text("v4", "View Text"), "SELECT IDENTIFIER('c1.c2') FROM t");
    assert_eq!(names(&s, "SELECT * FROM v4"), ["c1.c2"]);
}

/// Spark 4.2.0 (`research/spark/identifier-leftovers.md`, `CreateSQLFunctionCommand`): a table
/// function without a column list needs a named column for every item of a bare `SELECT` body.
/// An `IDENTIFIER(..)` item is an expression there, so it is refused as `x + 1` is; a body of
/// another shape is not checked, and a subquery names its `IDENTIFIER(..)` items as a query does.
#[test]
fn a_table_function_body_names_its_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (c1 STRUCT<c2: INT>, x INT, m MAP<STRING, INT>) USING parquet",
    );
    run(
        &s,
        "INSERT INTO t SELECT named_struct('c2', 1), 2, map('k', 3)",
    );
    let missing = "USER_DEFINED_FUNCTIONS.SQL_TABLE_UDF_MISSING_COLUMN_NAMES";
    let refused = [
        "SELECT IDENTIFIER('x') FROM t",
        "SELECT IDENTIFIER('t.x') FROM t",
        "SELECT IDENTIFIER('c1.c2') FROM t",
        "SELECT IDENTIFIER('c1').c2 FROM t",
        "SELECT (IDENTIFIER('x')) FROM t",
        "SELECT IDENTIFIER('x' || '') FROM t",
        "SELECT x + 1 FROM t WHERE x > 0",
        "SELECT 1",
        "SELECT *, x + 1 FROM t",
        "SELECT m['k'] FROM t",
        "SELECT CAST(x AS INT) FROM t",
        "SELECT sum(x) FROM t",
        "(SELECT x + 1 FROM t)",
        "SELECT ALL x + 1 FROM t",
        "SELECT x AS a, x + 1 FROM t",
        "SELECT x + 1 FROM t LATERAL VIEW explode(array(1)) e AS y",
        "FROM t SELECT x + 1",
        "FROM t |> SELECT x + 1",
        "FROM t |> SELECT IDENTIFIER('x')",
        "TABLE t |> EXTEND x + 1",
    ];
    for (i, body) in refused.iter().enumerate() {
        let sql = format!("CREATE TEMPORARY FUNCTION r{i}() RETURNS TABLE RETURN {body}");
        assert_eq!(class(&s, &sql), missing, "{body}");
    }
    let err = block_on(async {
        s.sql("CREATE FUNCTION pz() RETURNS TABLE RETURN SELECT IDENTIFIER('x') FROM t")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_eq!(err.error_class().name(), missing);
    assert_eq!(err.sql_state(), Some("42601"));
    assert!(err.to_string().contains("statement for `pz` with"), "{err}");
    // The function is named as written.
    let err = block_on(async {
        s.sql("CREATE TEMPORARY FUNCTION MixedF() RETURNS TABLE RETURN SELECT x + 1 FROM t")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert!(
        err.to_string().contains("statement for `MixedF` with"),
        "{err}"
    );
    // A stored function over a temporary view: its column names are checked first.
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT * FROM t");
    assert_eq!(
        class(
            &s,
            "CREATE FUNCTION pt() RETURNS TABLE RETURN SELECT x + 1 FROM tv"
        ),
        missing
    );
    assert_eq!(
        class(
            &s,
            "CREATE FUNCTION pt() RETURNS TABLE RETURN SELECT x FROM tv"
        ),
        "INVALID_TEMP_OBJ_REFERENCE"
    );
    let accepted = [
        ("SELECT IDENTIFIER('c1.c2') AS a FROM t", vec!["a"]),
        ("SELECT IDENTIFIER('c1').c2 AS a FROM t", vec!["a"]),
        ("SELECT IDENTIFIER('x') + 1 AS a FROM t", vec!["a"]),
        ("SELECT x FROM IDENTIFIER('t')", vec!["x"]),
        (
            "SELECT c1.c2, (c1).c2 AS d, ((x)), `x` AS y FROM t",
            vec!["c2", "d", "x", "y"],
        ),
        ("SELECT t.x, t.c1.c2 FROM t", vec!["x", "c2"]),
        ("SELECT IDENTIFIER('x') FROM t ORDER BY 1", vec!["x"]),
        ("SELECT IDENTIFIER('x') FROM t GROUP BY x", vec!["x"]),
        ("SELECT x + 1 FROM t ORDER BY 1", vec!["(x + 1)"]),
        ("SELECT DISTINCT x + 1 FROM t", vec!["(x + 1)"]),
        ("SELECT x + 1 FROM t LIMIT 1", vec!["(x + 1)"]),
        (
            "WITH q AS (SELECT x FROM t) SELECT x + 1 FROM q",
            vec!["(x + 1)"],
        ),
        (
            "SELECT x + 1 FROM t UNION ALL SELECT x FROM t",
            vec!["(x + 1)"],
        ),
        ("FROM t SELECT x + 1 ORDER BY 1", vec!["(x + 1)"]),
        (
            "SELECT * FROM (SELECT IDENTIFIER('c1.c2') FROM t)",
            vec!["c1.c2"],
        ),
        ("SELECT t.* FROM t", vec!["c1", "x", "m"]),
        ("VALUES (1)", vec!["col1"]),
        ("SELECT /*+ REPARTITION(2) */ x + 1 FROM t", vec!["(x + 1)"]),
        (
            "SELECT /*+ BROADCAST(t) */ IDENTIFIER('x') FROM t",
            vec!["x"],
        ),
        ("FROM t |> SELECT x", vec!["x"]),
        ("FROM t |> SELECT DISTINCT x + 1", vec!["(x + 1)"]),
        ("FROM t |> SELECT x + 1 |> WHERE true", vec!["(x + 1)"]),
        (
            "FROM t |> SELECT x + 1 |> EXTEND 1 AS b",
            vec!["(x + 1)", "b"],
        ),
        ("TABLE t |> EXTEND x + 1 AS a", vec!["c1", "x", "m", "a"]),
        ("FROM t |> DROP x", vec!["c1", "m"]),
        // `RETURN (query)` is the query, in as many brackets as written.
        ("(SELECT x FROM t)", vec!["x"]),
        ("((SELECT x AS a FROM t))", vec!["a"]),
    ];
    for (i, (body, columns)) in accepted.iter().enumerate() {
        run(
            &s,
            &format!("CREATE TEMPORARY FUNCTION a{i}() RETURNS TABLE RETURN {body}"),
        );
        assert_eq!(
            names(&s, &format!("SELECT * FROM a{i}()")),
            *columns,
            "{body}"
        );
    }
    // A declared column list names the columns: nothing to check.
    run(
        &s,
        "CREATE TEMPORARY FUNCTION d() RETURNS TABLE (a INT) RETURN SELECT IDENTIFIER('c1.c2') FROM t",
    );
    assert_eq!(names(&s, "SELECT * FROM d()"), ["a"]);
    // A scalar function's query body is its value.
    run(
        &s,
        "CREATE TEMPORARY FUNCTION s1() RETURN SELECT IDENTIFIER('x') FROM t",
    );
    assert_eq!(first(&s, "SELECT s1()"), "2");
    // A scalar function's bracketed subquery is its value, as before.
    run(
        &s,
        "CREATE TEMPORARY FUNCTION s2() RETURN (SELECT max(x) FROM t)",
    );
    assert_eq!(first(&s, "SELECT s2()"), "2");
    // Both modes read the clause as an expression.
    run(&s, "SET spark.sql.legacy.identifierClause=true");
    assert_eq!(
        class(
            &s,
            "CREATE TEMPORARY FUNCTION l() RETURNS TABLE RETURN SELECT IDENTIFIER('x') FROM t"
        ),
        missing
    );
}
