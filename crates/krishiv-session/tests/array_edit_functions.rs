// SPDX-License-Identifier: Apache-2.0
//! The array element-access and editing functions end to end: output names
//! and error classes of Spark 4.2.0 (`array.sql`, and a Spark 4.2.0 run for
//! the type errors), over a session.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

fn error_of(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Ok(df) => match block_on(df.collect()) {
            Ok(b) => panic!("{sql} ran: {b:?}"),
            Err(e) => e,
        },
        Err(e) => e,
    }
}

fn first_name(s: &Session, sql: &str) -> String {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.schema()).unwrap().field(0).name().clone()
}

/// An item that does not meet the element type, or a first argument that is
/// not an array, is `DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES`.
#[test]
fn mismatched_items_are_refused() {
    let s = session();
    for sql in [
        "SELECT array_append(array(1), 'x')",
        "SELECT array_prepend(array(1), 2.5)",
        "SELECT array_insert(array(1, 2, 3), 3, '4')",
        "SELECT array_insert(1, 1, 2)",
    ] {
        let class = error_of(&s, sql).error_class();
        assert_eq!(
            class.name(),
            "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
            "{sql}"
        );
    }
    // Spark 4.2.0: an append or prepend to a non-array, and a position that
    // is not an integer, are input-type errors.
    for sql in [
        "SELECT array_append(1, 2)",
        "SELECT array_prepend(NULL, NULL)",
        "SELECT array_insert(array(1), 'a', 2)",
    ] {
        assert_eq!(
            error_of(&s, sql).error_class().name(),
            "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
            "{sql}"
        );
    }
    let e = error_of(&s, "SELECT array_insert(array(1, 2, 3), 3, '4')");
    let text = e.to_string();
    assert!(
        text.contains("array_insert") && text.contains("ARRAY<INT>"),
        "{text}"
    );
}

#[test]
fn compact_and_size_refuse_non_arrays() {
    let s = session();
    for sql in [
        "SELECT array_compact(1)",
        "SELECT array_size(map('a', 1))",
        "SELECT get(1, 0)",
        "SELECT get(NULL, 0)",
        "SELECT element_at(1, 1)",
    ] {
        assert_eq!(
            error_of(&s, sql).error_class().name(),
            "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
            "{sql}"
        );
    }
}

/// The widening of the item shows in the result: `array_append(array(1), 5L)`
/// is `array<bigint>`; a typed NULL item keeps the element type.
#[test]
fn items_widen_to_the_element_type() {
    let s = session();
    let df = block_on(
        s.sql("SELECT array_append(array(1), 5L), array_prepend(array('a'), CAST(NULL AS STRING))"),
    )
    .unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert!(
        matches!(schema.field(0).data_type(), arrow_schema::DataType::List(f) if f.data_type() == &arrow_schema::DataType::Int64)
    );
    let batches = block_on(df.collect()).unwrap();
    let col = batches[0].column(0).as_list::<i32>();
    assert_eq!(col.value_length(0), 2);
    assert!(batches[0].column(1).as_list::<i32>().value(0).is_null(0));
}

/// Output names: the omitted pieces print as Spark prints them.
#[test]
fn output_names_follow_spark() {
    let s = session();
    assert_eq!(
        first_name(&s, "SELECT array_append(array(1, 2), 3)"),
        "array_append(array(1, 2), 3)"
    );
    assert_eq!(
        first_name(&s, "SELECT element_at(array(1, 2), 1)"),
        "element_at(array(1, 2), 1)"
    );
    assert_eq!(
        first_name(&s, "SELECT array_size(array(1))"),
        "array_size(array(1))"
    );
    assert_eq!(
        first_name(
            &s,
            "SELECT array_append(CAST(array() AS ARRAY<STRING>), 'a')"
        ),
        "array_append(array(), a)"
    );
}

/// Spark 4.2.0 shows a cast to an array, map or struct type as its operand,
/// written or not: the output name and the quoted expression of a refused
/// cast (`CAST(1 AS ARRAY<INT>)` quotes `"1"`).
#[test]
fn nested_casts_show_their_operand() {
    let s = session();
    assert_eq!(
        first_name(&s, "SELECT CAST(array(1) AS ARRAY<BIGINT>)"),
        "array(1)"
    );
    for (sql, quoted) in [
        ("SELECT CAST(1 AS ARRAY<INT>)", "\"1\""),
        ("SELECT CAST(array(1) AS MAP<INT, INT>)", "\"array(1)\""),
        ("SELECT CAST(map(1, 2) AS ARRAY<INT>)", "\"map(1, 2)\""),
    ] {
        let e = error_of(&s, sql);
        assert_eq!(
            e.error_class().name(),
            "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION",
            "{sql}"
        );
        let shown = e
            .message_parameters()
            .iter()
            .find(|(k, _)| *k == "sqlExpr")
            .map(|(_, v)| v.as_str());
        assert_eq!(shown, Some(quoted), "{sql}");
    }
}

/// Spark 4.2.0 types the result of `array_append`, `array_prepend` and
/// `array_insert` with `containsNull = true` whatever the inputs (padding adds
/// NULLs), and `array_compact`'s with `false`.
#[test]
fn result_elements_are_nullable_as_spark_types_them() {
    let s = session();
    let element_nullable = |sql: &str| {
        let df = block_on(s.sql(sql)).unwrap();
        match block_on(df.schema()).unwrap().field(0).data_type() {
            arrow_schema::DataType::List(f) => f.is_nullable(),
            other => panic!("{sql}: {other}"),
        }
    };
    for sql in [
        "SELECT array_append(array(1), 2)",
        "SELECT array_prepend(array(1), 2)",
        "SELECT array_insert(array(1), 1, 2)",
        "SELECT array_insert(array(1), 5, 2)",
    ] {
        assert!(element_nullable(sql), "{sql}");
    }
    assert!(!element_nullable("SELECT array_compact(array(1, NULL))"));
}
