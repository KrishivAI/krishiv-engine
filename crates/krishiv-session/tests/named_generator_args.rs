// SPDX-License-Identifier: Apache-2.0
//! A generator in `FROM` takes its one parameter by name (`explode(collection => ...)`,
//! `inline(input => ...)`), as Spark 4.2.0's `named-function-arguments.sql` golden does.
#![cfg(test)]

use arrow_array::RecordBatch;
use futures::executor::block_on;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> usize {
    let out = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    out.iter().map(RecordBatch::num_rows).sum()
}

#[test]
fn a_named_argument_names_the_generators_one_parameter() {
    let s = Session::new().unwrap();
    for (sql, n) in [
        ("SELECT * FROM explode(collection => array(1, 2))", 2),
        (
            "SELECT * FROM explode_outer(collection => map('a', 1, 'b', 2))",
            2,
        ),
        ("SELECT * FROM posexplode(collection => array(1, 2))", 2),
        (
            "SELECT * FROM posexplode_outer(collection => map('a', 1, 'b', 2))",
            2,
        ),
        (
            "SELECT * FROM inline(input => array(struct(1, 'a'), struct(2, 'b')))",
            2,
        ),
        (
            "SELECT * FROM inline_outer(input => array(struct(1, 'a'), struct(2, 'b')))",
            2,
        ),
    ] {
        assert_eq!(rows(&s, sql), n, "{sql}");
    }
}

#[test]
fn another_name_is_still_refused() {
    let s = Session::new().unwrap();
    let r = block_on(async {
        s.sql("SELECT * FROM explode(input => array(1))")
            .await?
            .collect()
            .await
    });
    assert!(r.is_err());
}

/// The error class `sql` fails with.
fn class(s: &Session, sql: &str) -> String {
    let r = block_on(async { s.sql(sql).await?.collect().await });
    match r {
        Ok(_) => panic!("{sql} ran"),
        Err(e) => e.error_class().name().to_string(),
    }
}

// The expectations below are Spark 4.2.0's, from a local PySpark run.
#[test]
fn a_select_list_generator_takes_its_parameter_by_name() {
    let s = Session::new().unwrap();
    for sql in [
        "SELECT explode(collection => array(1, 2))",
        "SELECT explode_outer(collection => array(1, 2))",
        "SELECT posexplode(collection => array(1, 2))",
        "SELECT inline(input => array(struct(1 AS a), struct(2 AS a)))",
    ] {
        assert_eq!(rows(&s, sql), 2, "{sql}");
    }
    for (sql, expected) in [
        (
            "SELECT explode(foo => array(1))",
            "UNRECOGNIZED_PARAMETER_NAME",
        ),
        ("SELECT stack(n => 1, 2)", "NAMED_PARAMETERS_NOT_SUPPORTED"),
        (
            "SELECT abs(explode(collection => array(1)))",
            "UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS",
        ),
        (
            "SELECT * FROM explode(collection => explode(collection => array(1)))",
            "UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS",
        ),
    ] {
        assert_eq!(class(&s, sql), expected, "{sql}");
    }
}

#[test]
fn a_generator_refuses_a_table_argument_by_its_row_type() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TEMP VIEW v AS SELECT id FROM range(0, 8)")).unwrap();
    for sql in [
        "SELECT * FROM explode(collection => TABLE(v))",
        "SELECT * FROM inline(input => TABLE(v))",
        "SELECT * FROM posexplode(TABLE(v))",
    ] {
        assert_eq!(
            class(&s, sql),
            "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
            "{sql}"
        );
    }
}

#[test]
fn range_over_an_outer_column_is_not_foldable() {
    let s = Session::new().unwrap();
    for sql in [
        "SELECT * FROM VALUES (0, 1) t(c1, c2), LATERAL range(0, c2)",
        "SELECT * FROM range(3) t, LATERAL range(0, t.id)",
        "SELECT (SELECT count(*) FROM range(0, c2)) FROM VALUES (1) t(c2)",
    ] {
        assert_eq!(class(&s, sql), "NON_FOLDABLE_ARGUMENT", "{sql}");
    }
    assert_eq!(
        class(
            &s,
            "SELECT * FROM VALUES (0, 1) t(c1, c2), LATERAL range(0, nope)"
        ),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT * FROM VALUES (0), (1) t(c1), LATERAL range(0, 2) r"
        ),
        4
    );
}
