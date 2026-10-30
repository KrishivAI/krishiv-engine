// SPDX-License-Identifier: Apache-2.0
//! `EXECUTE IMMEDIATE` through `Session::sql`; the classes and values are Spark 4.2.0's
//! (`tests/spark-golden/v4.2.0/results/execute-immediate.sql.out`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_session::{DenyAll, Session};

fn session() -> Session {
    Session::new().unwrap()
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

/// The error of `sql`: its class and message.
fn failure(s: &Session, sql: &str) -> (String, String) {
    let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
    (e.error_class().name().to_owned(), e.to_string())
}

/// Every cell of `sql` as a string (`NULL` for a null), row by row; columns must be strings.
fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            "NULL".to_owned()
                        } else {
                            match c.data_type() {
                                ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                                ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                                other => panic!("unexpected {other}"),
                            }
                        }
                    })
                    .collect(),
            );
        }
    }
    out
}

#[test]
fn positional_and_named_arguments_bind() {
    let s = session();
    run(&s, "DECLARE a STRING");
    run(&s, "SET VAR a = 'name2'");
    assert_eq!(
        rows(
            &s,
            "EXECUTE IMMEDIATE 'SELECT CAST(? + 1 AS STRING), ?' USING 41, a"
        ),
        [["42", "name2"]]
    );
    // Positional markers ignore the names of the arguments.
    assert_eq!(
        rows(
            &s,
            "EXECUTE IMMEDIATE 'SELECT CAST(? AS STRING), CAST(? AS STRING)' USING 1 AS y, 2 AS x"
        ),
        [["1", "2"]]
    );
    assert_eq!(
        rows(
            &s,
            "EXECUTE IMMEDIATE 'SELECT CAST(:x AS STRING), :y' USING 'o''k' AS y, 7 AS x"
        ),
        [["7", "o'k"]]
    );
    // A variable named by the argument binds the marker of that name, and the
    // list may be parenthesized.
    assert_eq!(
        rows(&s, "EXECUTE IMMEDIATE 'SELECT :a' USING (a)"),
        [["name2"]]
    );
    // The statement text may be any string expression.
    run(&s, "DECLARE q STRING");
    run(&s, "SET VAR q = 'SELECT ''v'''");
    assert_eq!(rows(&s, "EXECUTE IMMEDIATE q"), [["v"]]);
    assert_eq!(
        rows(&s, "EXECUTE IMMEDIATE concat('SELECT ', '''c''')"),
        [["c"]]
    );
    // A marker inside a string literal or a comment is text.
    assert_eq!(
        rows(&s, "EXECUTE IMMEDIATE 'SELECT ''?'' /* :n */' USING 1"),
        [["?"]]
    );
}

#[test]
fn argument_and_marker_errors() {
    let s = session();
    assert_eq!(
        class(
            &s,
            "EXECUTE IMMEDIATE 'SELECT :first' USING 10 AS first, 20 AS FIRST"
        ),
        "EXEC_IMMEDIATE_DUPLICATE_ARGUMENT_ALIASES"
    );
    assert_eq!(
        class(
            &s,
            "EXECUTE IMMEDIATE 'SELECT 42 WHERE ? = :first' USING 1, 2 AS first"
        ),
        "INVALID_QUERY_MIXED_QUERY_PARAMETERS"
    );
    let (c, m) = failure(&s, "EXECUTE IMMEDIATE 'SELECT :first' USING 2, 3");
    assert_eq!(c, "ALL_PARAMETERS_MUST_BE_NAMED");
    assert!(m.contains("\"2\", \"3\""), "{m}");
    let (c, m) = failure(&s, "EXECUTE IMMEDIATE 'SELECT :param'");
    assert_eq!(c, "UNBOUND_SQL_PARAMETER");
    assert!(m.contains("param"), "{m}");
    let (c, m) = failure(&s, "EXECUTE IMMEDIATE 'SELECT ?'");
    assert_eq!(c, "UNBOUND_SQL_PARAMETER");
    assert!(m.contains("_7"), "{m}");
    for sql in [
        "EXECUTE IMMEDIATE 'SELECT ?' USING (SELECT 1)",
        "EXECUTE IMMEDIATE (SELECT 'SELECT 1')",
        "EXECUTE IMMEDIATE 'SELECT :a' USING 1 + (SELECT 1) AS a",
    ] {
        assert_eq!(class(&s, sql), "UNSUPPORTED_EXPR_FOR_PARAMETER", "{sql}");
    }
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT 1' USING missing_column"),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
}

#[test]
fn the_statement_text_must_be_a_string_that_is_not_null() {
    let s = session();
    run(&s, "DECLARE n STRING");
    run(&s, "DECLARE i INT");
    let (c, m) = failure(&s, "EXECUTE IMMEDIATE n");
    assert_eq!(c, "NULL_QUERY_STRING_EXECUTE_IMMEDIATE");
    assert!(m.contains("`n`"), "{m}");
    let (c, m) = failure(&s, "EXECUTE IMMEDIATE i");
    assert_eq!(c, "INVALID_EXPR_TYPE_FOR_QUERY_EXECUTE_IMMEDIATE");
    assert!(m.contains("\"INT\""), "{m}");
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE length('abc')"),
        "INVALID_EXPR_TYPE_FOR_QUERY_EXECUTE_IMMEDIATE"
    );
    // The statement's own syntax error is the immediate string's, not the command's.
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELCT 1'"),
        "PARSE_SYNTAX_ERROR"
    );
}

#[test]
fn into_assigns_the_one_row_through_a_cast() {
    let s = session();
    run(&s, "DECLARE id INT");
    run(&s, "DECLARE nm STRING");
    run(
        &s,
        "EXECUTE IMMEDIATE 'SELECT 7, ''seven'' WHERE ? = 7' INTO id, nm USING 7",
    );
    assert_eq!(rows(&s, "SELECT CAST(id AS STRING), nm"), [["7", "seven"]]);
    // A query without rows sets NULL.
    run(
        &s,
        "EXECUTE IMMEDIATE 'SELECT 1, ''x'' WHERE 1 = 2' INTO id, nm",
    );
    assert_eq!(
        rows(&s, "SELECT CAST(id AS STRING), nm"),
        [["NULL", "NULL"]]
    );
    // Values go through a store-assignment cast.
    run(&s, "EXECUTE IMMEDIATE 'SELECT ''1707''' INTO id");
    assert_eq!(rows(&s, "SELECT CAST(id AS STRING)"), [["1707"]]);
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT ''invalid''' INTO id"),
        "CAST_INVALID_INPUT"
    );
    assert_eq!(
        class(
            &s,
            "EXECUTE IMMEDIATE 'SELECT id FROM VALUES (1), (2) AS t(id)' INTO id"
        ),
        "ROW_SUBQUERY_TOO_MANY_ROWS"
    );
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT 1, 2' INTO id"),
        "ASSIGNMENT_ARITY_MISMATCH"
    );
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT 1' INTO id, nm"),
        "ASSIGNMENT_ARITY_MISMATCH"
    );
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT 1, 2' INTO id, id"),
        "DUPLICATE_ASSIGNMENTS"
    );
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT 1' INTO undeclared"),
        "VARIABLE_NOT_FOUND"
    );
    // Only a query has a row to assign.
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SET VAR id = 1' INTO id"),
        "INVALID_STATEMENT_FOR_EXECUTE_INTO"
    );
}

#[test]
fn any_statement_runs_and_nesting_is_bounded() {
    let s = session();
    run(&s, "DECLARE sql_string STRING");
    run(
        &s,
        "EXECUTE IMMEDIATE 'SET VAR sql_string = ?' USING 'SELECT 5'",
    );
    assert_eq!(rows(&s, "SELECT sql_string"), [["SELECT 5"]]);
    run(
        &s,
        "EXECUTE IMMEDIATE 'CREATE TEMPORARY VIEW v AS SELECT 3 AS c'",
    );
    assert_eq!(rows(&s, "SELECT CAST(c AS STRING) FROM v"), [["3"]]);
    // One level of nesting works; the inner statement has its own arguments.
    assert_eq!(
        rows(
            &s,
            "EXECUTE IMMEDIATE 'EXECUTE IMMEDIATE ''SELECT CAST(? AS STRING)'' USING 10'"
        ),
        [["10"]]
    );
    // A statement that executes itself stops at the second level.
    run(&s, "SET VAR sql_string = 'EXECUTE IMMEDIATE sql_string'");
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE sql_string"),
        "NESTED_EXECUTE_IMMEDIATE"
    );
}

#[test]
fn the_statement_is_authorized_as_if_typed() {
    let denied = Session::builder()
        .access_policy(std::sync::Arc::new(DenyAll))
        .build()
        .unwrap();
    // The same text is refused whether typed or run immediately, with the same class.
    let typed = class(&denied, "SELECT * FROM secret");
    assert_eq!(typed, "KRISHIV_INSUFFICIENT_PERMISSIONS");
    assert_eq!(
        class(&denied, "EXECUTE IMMEDIATE 'SELECT * FROM secret'"),
        typed
    );
    assert_eq!(
        class(&denied, "EXECUTE IMMEDIATE 'DROP TABLE secret'"),
        class(&denied, "DROP TABLE secret")
    );
}

#[test]
fn values_keep_their_type_and_their_own_token() {
    let s = session();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    // A negative value after `-` is not a comment, and `?` before a word is not glued to it.
    assert_eq!(
        rows(
            &s,
            "EXECUTE IMMEDIATE 'SELECT CAST(5-? AS STRING), ''tail'' AS t' USING -3"
        ),
        [["8", "tail"]]
    );
    assert_eq!(
        rows(&s, "EXECUTE IMMEDIATE 'SELECT CAST(?AS STRING)' USING 5"),
        [["5"]]
    );
    // Decimals keep their precision and scale.
    assert_eq!(
        rows(
            &s,
            "EXECUTE IMMEDIATE 'SELECT typeof(?), CAST(? AS STRING)' USING 123.45BD, 123.45BD"
        ),
        [["decimal(5,2)", "123.45"]]
    );
    // Timestamps travel exactly, into a variable too.
    assert_eq!(
        rows(
            &s,
            "EXECUTE IMMEDIATE 'SELECT typeof(?), CAST(? AS STRING)' USING TIMESTAMP '2023-12-25 10:30:45.5', TIMESTAMP '2023-12-25 10:30:45.5'"
        ),
        [["timestamp", "2023-12-25 10:30:45.5"]]
    );
    run(&s, "DECLARE ts TIMESTAMP");
    run(
        &s,
        "EXECUTE IMMEDIATE 'SELECT TIMESTAMP ''2020-02-29 01:02:03''' INTO ts",
    );
    assert_eq!(
        rows(&s, "SELECT CAST(ts AS STRING)"),
        [["2020-02-29 01:02:03"]]
    );
    // Types the text cannot carry exactly are refused.
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT ?' USING INTERVAL '3' DAY"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn only_written_aliases_can_collide() {
    let s = session();
    run(&s, "DECLARE a STRING");
    run(&s, "SET VAR a = 'x'");
    assert_eq!(
        rows(&s, "EXECUTE IMMEDIATE 'SELECT ?, ?' USING a, a"),
        [["x", "x"]]
    );
    assert_eq!(
        class(&s, "EXECUTE IMMEDIATE 'SELECT :a' USING 1 AS a, 2 AS A"),
        "EXEC_IMMEDIATE_DUPLICATE_ARGUMENT_ALIASES"
    );
}

#[test]
fn a_backquoted_variable_is_named_once_and_a_char_index_names_the_marker() {
    let s = session();
    run(&s, "DECLARE nq STRING");
    let (c, m) = failure(&s, "EXECUTE IMMEDIATE `nq`");
    assert_eq!(c, "NULL_QUERY_STRING_EXECUTE_IMMEDIATE");
    assert!(m.contains("`nq`") && !m.contains("``nq``"), "{m}");
    let (_, m) = failure(&s, "EXECUTE IMMEDIATE 'SELECT ''é'', ?'");
    assert!(m.contains("_12"), "{m}");
}
