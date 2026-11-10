// SPDX-License-Identifier: Apache-2.0
//! The SQL context of a run-time error: a cast, an arithmetic operation or a function that raises
//! reports the fragment of the expression and its start and stop in the statement, and the error
//! message shows Spark's `== SQL (line, position) ==` block. Every vector is from a Spark 4.2.0 run
//! (`getQueryContext()` and the message of the raised exception); `spark.sql.ansi.enabled` is on.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::{KrishivError, render_with_context};
use krishiv_session::Session;

/// The error `sql` raises while it runs.
fn error_of(sql: &str) -> KrishivError {
    let s = Session::new().unwrap();
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(b) => panic!("{sql} ran: {b:?}"),
        Err(e) => e,
    }
}

/// `(class, fragment, start, stop)` of the first query context of the error `sql` raises.
fn context_of(sql: &str) -> (String, Option<(String, usize, usize)>) {
    let e = error_of(sql);
    let ctx = e
        .query_contexts()
        .first()
        .map(|c| (c.fragment.clone(), c.start, c.stop));
    (e.error_class().name().to_owned(), ctx)
}

fn check(sql: &str, class: &str, fragment: &str, start: usize, stop: usize) {
    let (got_class, ctx) = context_of(sql);
    assert_eq!(got_class, class, "{sql}");
    assert_eq!(ctx, Some((fragment.to_owned(), start, stop)), "{sql}");
}

fn check_none(sql: &str, class: &str) {
    let (got_class, ctx) = context_of(sql);
    assert_eq!(got_class, class, "{sql}");
    assert_eq!(ctx, None, "{sql}");
}

#[test]
fn a_failing_cast_quotes_itself() {
    check(
        "SELECT CAST(x AS INT) FROM VALUES ('a') AS t(x)",
        "CAST_INVALID_INPUT",
        "CAST(x AS INT)",
        7,
        20,
    );
    // The innermost raising expression, not the operator above it.
    check(
        "SELECT CAST(x AS INT) + 1 FROM VALUES ('a') AS t(x)",
        "CAST_INVALID_INPUT",
        "CAST(x AS INT)",
        7,
        20,
    );
    check(
        "SELECT abs(CAST(x AS INT)) FROM VALUES ('a') AS t(x)",
        "CAST_INVALID_INPUT",
        "CAST(x AS INT)",
        11,
        24,
    );
}

#[test]
fn an_implicit_cast_quotes_the_operation_that_needed_it() {
    check(
        "SELECT x + 'a' FROM VALUES (1) AS t(x)",
        "CAST_INVALID_INPUT",
        "x + 'a'",
        7,
        13,
    );
    check(
        "SELECT to_date(x) FROM VALUES ('zz') AS t(x)",
        "CAST_INVALID_INPUT",
        "to_date(x)",
        7,
        16,
    );
}

#[test]
fn arithmetic_errors_quote_the_operation() {
    check(
        "SELECT x / 0 FROM VALUES (1) AS t(x)",
        "DIVIDE_BY_ZERO",
        "x / 0",
        7,
        11,
    );
    check(
        "SELECT x DIV 0 FROM VALUES (1) AS t(x)",
        "DIVIDE_BY_ZERO",
        "x DIV 0",
        7,
        13,
    );
    check(
        "SELECT x % 0 FROM VALUES (1) AS t(x)",
        "REMAINDER_BY_ZERO",
        "x % 0",
        7,
        11,
    );
    check(
        "SELECT pmod(x, 0) FROM VALUES (1) AS t(x)",
        "REMAINDER_BY_ZERO",
        "pmod(x, 0)",
        7,
        16,
    );
    check(
        "SELECT 2147483647 + x FROM VALUES (1) AS t(x)",
        "ARITHMETIC_OVERFLOW",
        "2147483647 + x",
        7,
        20,
    );
    check(
        "SELECT x * 3000000000 FROM VALUES (9000000000000) AS t(x)",
        "ARITHMETIC_OVERFLOW",
        "x * 3000000000",
        7,
        20,
    );
    check(
        "SELECT INTERVAL '1' DAY / x FROM VALUES (0) AS t(x)",
        "INTERVAL_DIVIDED_BY_ZERO",
        "INTERVAL '1' DAY / x",
        7,
        26,
    );
}

#[test]
fn element_and_range_errors_quote_the_call() {
    check(
        "SELECT element_at(array(x), 5) FROM VALUES (1) AS t(x)",
        "INVALID_ARRAY_INDEX_IN_ELEMENT_AT",
        "element_at(array(x), 5)",
        7,
        29,
    );
    check(
        "SELECT array(x)[5] FROM VALUES (1) AS t(x)",
        "INVALID_ARRAY_INDEX",
        "array(x)[5]",
        7,
        17,
    );
    check(
        "SELECT element_at(array(x), 0) FROM VALUES (1) AS t(x)",
        "INVALID_INDEX_OF_ZERO",
        "element_at(array(x), 0)",
        7,
        29,
    );
    check(
        "SELECT CAST(x AS DECIMAL(3,1)) FROM VALUES (1000) AS t(x)",
        "NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION",
        "CAST(x AS DECIMAL(3,1))",
        7,
        29,
    );
}

/// Spark raises these without a query context; the engine must not invent one.
#[test]
fn errors_spark_raises_without_a_context_get_none() {
    check_none(
        "SELECT CAST(x AS TINYINT) FROM VALUES (1000) AS t(x)",
        "CAST_OVERFLOW",
    );
    check_none(
        "SELECT CAST(x AS INT) FROM VALUES (9999999999) AS t(x)",
        "CAST_OVERFLOW",
    );
    check_none(
        "SELECT -x FROM VALUES (-2147483648) AS t(x)",
        "ARITHMETIC_OVERFLOW",
    );
    check_none(
        "SELECT abs(x) FROM VALUES (-2147483648) AS t(x)",
        "ARITHMETIC_OVERFLOW",
    );
}

#[test]
fn the_position_counts_lines_and_the_message_shows_the_span() {
    check(
        "SELECT\n  1,\n  CAST(x AS INT)\nFROM VALUES ('a') AS t(x)",
        "CAST_INVALID_INPUT",
        "CAST(x AS INT)",
        14,
        27,
    );
    let sql = "SELECT CAST(x AS INT) FROM VALUES ('a') AS t(x)";
    let one = render_with_context(&error_of(sql), sql);
    assert!(
        one.contains(
            "== SQL (line 1, position 8) ==\nSELECT CAST(x AS INT) FROM VALUES ('a') AS t(x)\n       ^^^^^^^^^^^^^^"
        ),
        "{one}"
    );
    let sql = "SELECT\n  1,\n  CAST(x AS INT)\nFROM VALUES ('a') AS t(x)";
    let many = render_with_context(&error_of(sql), sql);
    assert!(
        many.contains("== SQL (line 3, position 3) ==\n  CAST(x AS INT)\n  ^^^^^^^^^^^^^^"),
        "{many}"
    );
}

/// `try_cast` and a row the cast never reaches raise nothing.
#[test]
fn no_error_no_context() {
    let s = Session::new().unwrap();
    let rows = block_on(async {
        s.sql("SELECT try_cast(x AS INT) FROM VALUES ('a') AS t(x)")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(
        rows.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
}

/// A plan kept from an earlier statement points into that statement's text, which a later plan
/// does not carry: its expressions get no context, never another statement's fragment.
#[test]
fn a_plan_from_an_earlier_statement_is_never_quoted_against_the_wrong_text() {
    let s = Session::new().unwrap();
    // The view's cast is the first statement's text; the second statement is shorter and different.
    let first = block_on(s.sql("SELECT CAST(x AS INT) AS y FROM VALUES ('a') AS t(x)")).unwrap();
    block_on(first.create_or_replace_temp_view("v")).unwrap();
    let e = match block_on(async { s.sql("SELECT 1, y FROM v").await?.collect().await }) {
        Ok(b) => panic!("ran: {b:?}"),
        Err(e) => e,
    };
    assert_eq!(e.error_class().name(), "CAST_INVALID_INPUT");
    assert!(
        e.query_contexts().is_empty(),
        "quoted the wrong text: {:?}",
        e.query_contexts()
    );
    // The same holds for a DataFrame built from another statement and joined into this one.
    let a = block_on(s.sql("SELECT CAST(x AS INT) AS y FROM VALUES ('a') AS t(x)")).unwrap();
    let b = block_on(s.sql("SELECT 1 AS k")).unwrap();
    let joined = a.cross_join(&b);
    let e = block_on(joined.collect()).unwrap_err();
    assert!(e.query_contexts().is_empty(), "{:?}", e.query_contexts());
}
