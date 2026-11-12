// SPDX-License-Identifier: Apache-2.0
//! Errors with their SQL query context rendered (`render_with_context`):
//! a snapshot per error class, and the contexts the renderer receives.
//! The attached spans are checked against Spark 4.2.0's goldens
//! (`tests/spark-golden/v4.2.0/results/*.sql.out`); the summary text itself
//! is engine-defined (see `krishiv-common/src/error/render.rs`).
//!
//! The goldens' JSON shows `startIndex` and `stopIndex` one higher than
//! `QueryContext.startIndex()` (Spark's JSON writer adds one), so a golden's
//! `startIndex` 8 is the offset 7 here.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError, render_with_context};
use krishiv_session::Session;

/// The error `sql` raises.
fn fails(sql: &str) -> KrishivError {
    let s = Session::new().unwrap();
    match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err("the query fails"),
    }
}

/// The first context of `e` as `(start, stop, fragment)`.
fn span(e: &KrishivError) -> (usize, usize, &str) {
    let c = &e.query_contexts()[0];
    (c.start, c.stop, c.fragment.as_str())
}

#[test]
fn unresolved_column_with_suggestion() {
    let sql = "SELECT nme FROM (SELECT 1 AS name, 2 AS id) t";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::UnresolvedColumnWithSuggestion);
    insta::assert_snapshot!(render_with_context(&e, sql));
}

#[test]
fn unresolved_column_without_suggestion() {
    let sql = "SELECT a";
    let e = fails(sql);
    assert_eq!(
        e.error_class(),
        ErrorClass::UnresolvedColumnWithoutSuggestion
    );
    insta::assert_snapshot!(render_with_context(&e, sql));
}

#[test]
fn qualified_unresolved_column_marks_the_whole_name() {
    let sql = "SELECT t.nme\nFROM (SELECT 1 AS name) t";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::UnresolvedColumnWithSuggestion);
    assert_eq!(span(&e), (7, 11, "t.nme"));
    insta::assert_snapshot!(render_with_context(&e, sql));
}

#[test]
fn unresolved_routine_on_a_later_line() {
    // Spark's golden (`double-quoted-identifiers`) for "SELECT `not_exist`()":
    // startIndex 8, stopIndex 20, fragment "`not_exist`()".
    assert_eq!(
        span(&fails("SELECT `not_exist`()")),
        (7, 19, "`not_exist`()")
    );
    let sql = "SELECT 1,\n  not_exist(2)\nFROM (SELECT 1 AS a) t";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::UnresolvedRoutine);
    insta::assert_snapshot!(render_with_context(&e, sql));
}

#[test]
fn table_or_view_not_found() {
    // Golden `double-quoted-identifiers`: startIndex 15, stopIndex 23, "not_exist".
    let sql = "SELECT 1 FROM not_exist AS X(`hello`)";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(span(&e), (14, 22, "not_exist"));
    insta::assert_snapshot!(render_with_context(&e, sql));
}

#[test]
fn parse_error() {
    let sql = "SELECT *\nFORM t";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::ParseSyntaxError);
    insta::assert_snapshot!(render_with_context(&e, sql));
}

#[test]
fn multi_byte_text_counts_characters() {
    let sql = "SELECT 'é' AS a, nope";
    let e = fails(sql);
    insta::assert_snapshot!(render_with_context(&e, sql));
}

#[test]
fn a_constant_division_by_zero_has_the_division_span() {
    // Spark 4.2.0 (`SELECT 1/0`): fragment `1/0`, start 7, stop 9.
    let e = fails("SELECT 1/0");
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert_eq!(span(&e), (7, 9, "1/0"));
}

#[test]
fn a_runtime_error_in_a_projection_has_its_span() {
    let e = fails("SELECT x / 0 FROM VALUES (1) AS t(x)");
    assert_eq!(span(&e), (7, 11, "x / 0"));
}

/// `Display` and the message parameters stay Spark's: the unknown function
/// and the unknown table carry no proposal (golden `double-quoted-identifiers`).
#[test]
fn not_found_errors_keep_spark_messages_and_parameters() {
    const SEARCH_PATH: &str = "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]";
    let routine = fails("SELECT `not_exist`()");
    assert_eq!(
        routine.to_string(),
        format!(
            "[UNRESOLVED_ROUTINE] Cannot resolve routine `not_exist` on search path \
             {SEARCH_PATH}. SQLSTATE: 42883"
        )
    );
    assert_eq!(
        routine.message_parameters(),
        [
            ("routineName", "`not_exist`".to_owned()),
            ("searchPath", SEARCH_PATH.to_owned())
        ]
    );
    let table = fails("SELECT 1 FROM not_exist AS X(`hello`)");
    assert_eq!(
        table.message_parameters(),
        [
            ("relationName", "`not_exist`".to_owned()),
            ("searchPath", SEARCH_PATH.to_owned())
        ]
    );
    assert!(!table.to_string().contains("Did you mean"), "{table}");
    // The rendering only appends the context.
    let sql = "SELECT `not_exist`()";
    assert!(render_with_context(&routine, sql).starts_with(&routine.to_string()));
}

/// The column proposals of the goldens' ranking: closest first, ties by name.
#[test]
fn column_proposals_are_ranked_by_edit_distance() {
    let e = fails("SELECT nme FROM (SELECT 1 AS id, 2 AS name, 3 AS age) t");
    assert_eq!(
        e.message_parameters(),
        [
            ("objectName", "`nme`".to_owned()),
            ("proposal", "`name`, `age`, `id`".to_owned())
        ]
    );
}

// Run-time errors carry the span of the expression that raised them (P3 design
// P2-runtime-error-context, unit 1).
#[test]
fn runtime_divide_by_zero_has_the_division_span() {
    let sql = "SELECT a / b FROM VALUES (1, 0) AS t(a, b)";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert_eq!(span(&e), (7, 11, "a / b"));
}

#[test]
fn runtime_cast_invalid_input_has_the_cast_span() {
    let sql = "SELECT CAST(a AS INT) FROM VALUES ('x') AS t(a)";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::CastInvalidInput);
    assert_eq!(span(&e), (7, 20, "CAST(a AS INT)"));
}

/// Spark 4.2.0 (`cast.sql.out`): `CAST_OVERFLOW` carries no query context.
#[test]
fn runtime_cast_overflow_has_no_context() {
    let e = fails("SELECT CAST(a AS TINYINT) FROM VALUES (1000) AS t(a)");
    assert!(e.query_contexts().is_empty());
    assert!(e.pending_site().is_none());
}

#[test]
fn runtime_error_innermost_span_wins() {
    let sql = "SELECT 1 + (a / b) FROM VALUES (1, 0) AS t(a, b)";
    let e = fails(sql);
    assert_eq!(e.query_contexts().len(), 1);
    assert_eq!(span(&e), (12, 16, "a / b"));
}

/// The error `sql` raises when the physical plan is encoded and decoded before it
/// runs, as a worker receives it (`krishiv.exec.debug.planRoundTrip`).
fn fails_round_tripped(sql: &str) -> KrishivError {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.exec.debug.planRoundTrip", "true")
        .unwrap();
    match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err("the query fails"),
    }
}

/// A plan that crossed the codec keeps the spans of its expressions, so a worker's
/// run-time error renders the same context as in-process.
#[test]
fn a_plan_through_the_codec_renders_the_same_runtime_contexts() {
    for sql in [
        "SELECT a / b FROM VALUES (1, 0) AS t(a, b)",
        "SELECT CAST(a AS INT) FROM VALUES ('x') AS t(a)",
        "SELECT 1 + (a / b) FROM VALUES (1, 0) AS t(a, b)",
    ] {
        let direct = fails(sql);
        let shipped = fails_round_tripped(sql);
        assert_eq!(shipped.error_class(), direct.error_class(), "{sql}");
        assert_eq!(span(&shipped), span(&direct), "{sql}");
    }
}

/// A DataFrame expression has a source of its own, not the statement's; a view made from it and read by
/// SQL must not render its span against the reading statement's text.
#[test]
fn a_dataframe_expression_span_is_not_rendered_against_another_statement() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT * FROM VALUES (1, 0) AS t(a, b)")).unwrap();
    let df = df.select_expr(["a / b AS q"]).unwrap();
    block_on(df.create_or_replace_temp_view("v_ctx")).unwrap();
    let sql = "SELECT q FROM v_ctx";
    let e = match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err("the query fails"),
    };
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    // No statement's text holds `a / b`, so the error has no context (a wrong span would cut the reading statement).
    assert!(e.query_contexts().is_empty(), "{:?}", e.query_contexts());
}

/// A view made from a SQL DataFrame was lowered from another text than the statement reading it.
#[test]
fn a_sql_view_expression_span_is_not_rendered_against_the_reading_statement() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT a / b AS q FROM VALUES (1, 0) t(a, b)")).unwrap();
    block_on(df.create_or_replace_temp_view("v_sql_ctx")).unwrap();
    let e = match block_on(s.sql("SELECT q FROM v_sql_ctx")) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err("the query fails"),
    };
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert!(e.query_contexts().is_empty(), "{:?}", e.query_contexts());
}

/// A scalar subquery in a stored view keeps no span of the view's text either.
#[test]
fn a_view_subquery_expression_span_is_not_rendered_against_the_reading_statement() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT (SELECT a / b FROM VALUES (1, 0) t(a, b)) AS q")).unwrap();
    block_on(df.create_or_replace_temp_view("v_sub_ctx")).unwrap();
    let e = match block_on(s.sql("SELECT q FROM v_sub_ctx")) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err("the query fails"),
    };
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert!(e.query_contexts().is_empty(), "{:?}", e.query_contexts());
}

/// Element access: Spark 4.2.0 `array.sql.out` (golden indexes are one higher).
#[test]
fn element_access_errors_carry_the_call_span() {
    let e = fails("select element_at(array(1, 2, 3), 5)");
    assert_eq!(e.error_class(), ErrorClass::InvalidArrayIndexInElementAt);
    assert_eq!(span(&e), (7, 35, "element_at(array(1, 2, 3), 5)"));
    let e = fails("select elt(4, '123', '456')");
    assert_eq!(e.error_class(), ErrorClass::InvalidArrayIndex);
    assert_eq!(span(&e), (7, 26, "elt(4, '123', '456')"));
    let e = fails("select array(1, 2, 3)[5]");
    assert_eq!(e.error_class(), ErrorClass::InvalidArrayIndex);
    assert_eq!(span(&e), (7, 23, "array(1, 2, 3)[5]"));
}

/// Spark 4.2.0 gives these errors no context (`try_element_at.sql.out`,
/// `string-functions.sql.out`, `interval.sql.out`).
#[test]
fn context_free_errors_stay_context_free() {
    for sql in [
        "SELECT try_element_at(array(1, 2, 3), 0)",
        "SELECT split_part('11.12.13', '.', 0)",
        // An enclosing call must not give the context-free error its own span.
        "SELECT abs(try_element_at(array(1, 2, 3), 0))",
        "select b + interval '1 month' from values (interval '-2147483648 months', interval '2147483647 months') t(a, b)",
    ] {
        let e = fails(sql);
        assert!(
            e.query_contexts().is_empty(),
            "{sql}: {:?}",
            e.query_contexts()
        );
    }
}
