// SPDX-License-Identifier: Apache-2.0
//! `ARRAY`, `STRUCT` and `MAP` session and script variables (T196), as Spark
//! 4.2.0 runs them (`sql-session-variables.sql`, `randomly_generated_scripts`
//! #27): the value is kept and read back whole, by element and by field.
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn text(s: &Session, sql: &str) -> String {
    let out = run(s, &format!("SELECT CAST(({sql}) AS STRING)"));
    let c = out[0].column(0);
    if c.is_null(0) {
        return "NULL".to_owned();
    }
    c.as_string_view().value(0).to_owned()
}

#[test]
fn session_variables_hold_nested_values() {
    let s = Session::new().unwrap();
    run(
        &s,
        "DECLARE VARIABLE a ARRAY<STRING> DEFAULT split('Dan Smith', ' ')",
    );
    assert_eq!(text(&s, "a"), "[Dan, Smith]");
    assert_eq!(text(&s, "a[0]"), "Dan");
    run(&s, "SET VARIABLE a = array('x', NULL)");
    assert_eq!(text(&s, "a"), "[x, null]");
    assert_eq!(text(&s, "size(a)"), "2");
    run(&s, "SET VARIABLE a = NULL");
    assert_eq!(text(&s, "a"), "NULL");
    run(&s, "SET VARIABLE a = array()");
    assert_eq!(text(&s, "size(a)"), "0");

    run(
        &s,
        "DECLARE VARIABLE m MAP<STRING, DOUBLE> DEFAULT MAP('Hello', 5.1, 'World', -7.1E10)",
    );
    assert_eq!(text(&s, "to_json(m)"), "{\"Hello\":5.1,\"World\":-7.1E10}");
    assert_eq!(text(&s, "typeof(m)"), "map<string,double>");

    run(
        &s,
        "DECLARE VARIABLE st STRUCT<a INT, b ARRAY<INT>> DEFAULT named_struct('a', 1, 'b', array(2, 3))",
    );
    assert_eq!(text(&s, "st.a"), "1");
    assert_eq!(text(&s, "session.st.b[1]"), "3");
    assert_eq!(text(&s, "system.session.st.b"), "[2, 3]");

    // Names as Spark 4.2.0 gives them.
    run(&s, "SET VARIABLE a = array('x', NULL)");
    let names = |sql: &str| {
        let df = block_on(s.sql(sql)).unwrap();
        let schema = block_on(df.schema()).unwrap();
        schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names("SELECT size(a), st.a, CAST(session.st.b AS STRING)"),
        [
            "size(variablereference(system.session.a=ARRAY('x', CAST(NULL AS STRING))))",
            "a",
            "CAST(variablereference(system.session.st=NAMED_STRUCT('a', 1, 'b', ARRAY(2, 3))).b AS STRING)",
        ]
    );
}

#[test]
fn a_null_struct_variable_still_has_its_fields() {
    // Spark 4.2.0: the declared type names the fields, so they read NULL.
    let s = Session::new().unwrap();
    run(
        &s,
        "DECLARE VARIABLE st STRUCT<a: INT> DEFAULT named_struct('a', 1)",
    );
    run(&s, "SET VARIABLE st = NULL");
    assert_eq!(text(&s, "st.a"), "NULL");
    assert_eq!(text(&s, "st"), "NULL");
    run(
        &s,
        "DECLARE VARIABLE st2 STRUCT<a: INT, b: STRUCT<c: STRING>> DEFAULT NULL",
    );
    assert_eq!(text(&s, "st2.b.c"), "NULL");
    let err = block_on(async { s.sql("SELECT st.z").await?.collect().await })
        .err()
        .map(|e| e.error_class().name().to_owned());
    assert_eq!(err.as_deref(), Some("FIELD_NOT_FOUND"));
}

#[test]
fn script_variables_hold_nested_values() {
    let s = Session::new().unwrap();
    let out = run(
        &s,
        "BEGIN
           DECLARE parts ARRAY<STRING>;
           DECLARE first STRING;
           SET parts = split('Dan Smith', ' ');
           CASE parts[0] WHEN 'Dan' THEN SET first = 'manager'; ELSE SET first = 'other'; END CASE;
           FOR r AS SELECT array(1, 2) AS xs DO
             SET first = concat(first, ':', CAST(size(r.xs) AS STRING));
           END FOR;
           SELECT first;
         END",
    );
    assert_eq!(out[0].column(0).as_string_view().value(0), "manager:2");
}

#[test]
fn a_script_returns_and_row_assigns_nested_values() {
    // Spark 4.2.0: a script's last query returns nested variables as they
    // are, and `SET (a, b) = (SELECT ...)` assigns an ARRAY.
    let s = Session::new().unwrap();
    let int_of = |sql: &str| {
        let out = run(&s, sql);
        out[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0)
    };
    let out = run(
        &s,
        "BEGIN DECLARE a ARRAY<INT> DEFAULT array(1, 2); SELECT a; END",
    );
    let list = out[0].column(0).as_list::<i32>().value(0);
    let items: Vec<i32> = list
        .as_primitive::<arrow_array::types::Int32Type>()
        .values()
        .to_vec();
    assert_eq!(items, [1, 2]);
    let out = run(
        &s,
        "BEGIN DECLARE st STRUCT<x: INT> DEFAULT named_struct('x', 3); SELECT st; END",
    );
    assert_eq!(out[0].column(0).as_struct().num_columns(), 1);
    assert_eq!(
        int_of("BEGIN DECLARE st STRUCT<x: INT> DEFAULT named_struct('x', 3); SELECT st.x; END"),
        3
    );
    assert_eq!(
        int_of("BEGIN DECLARE m MAP<STRING, INT> DEFAULT map('k', 1); SELECT m['k']; END"),
        1
    );
    assert_eq!(
        int_of("BEGIN DECLARE a ARRAY<INT>; SET (a) = (SELECT array(1, 2)); SELECT size(a); END"),
        2
    );
    assert_eq!(
        int_of(
            "BEGIN DECLARE a ARRAY<INT>; DECLARE b INT; SET (a, b) = (SELECT array(3), 4); \
             SELECT a[0] + b; END"
        ),
        7
    );
}
