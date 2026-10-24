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
fn an_error_without_a_context_is_the_message() {
    let sql = "SELECT 1/0";
    let e = fails(sql);
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert!(e.query_contexts().is_empty());
    assert_eq!(render_with_context(&e, sql), e.to_string());
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
