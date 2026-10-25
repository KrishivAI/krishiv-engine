// SPDX-License-Identifier: Apache-2.0
//! Struct functions end to end: `named_struct` and `struct` construction,
//! `STRUCT(...)` syntax, field access under `spark.sql.caseSensitive` and
//! struct comparison. Construction, field access and error classes follow
//! the matrix and golden fixtures (real Spark 4.2 runs); comparison follows
//! Kleene three-valued logic for equality and lexicographic order, which no
//! fixture pins (open question: Spark's exact null ordering for `<`).
//! Not here: star-except in `STRUCT`, `MAX`/`MIN` over structs.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

/// Every row of `sql`'s first column, rendered as Spark renders values
/// (`{"a":1}`, `NULL` at the top level).
fn rows(s: &Session, sql: &str) -> Vec<String> {
    use arrow_array::cast::AsArray as _;

    fn cell(col: &arrow_array::ArrayRef, row: usize) -> String {
        if col.is_null(row) {
            return "NULL".to_owned();
        }
        match col.data_type() {
            arrow_schema::DataType::Null => "null".to_owned(),
            arrow_schema::DataType::Boolean => col.as_boolean().value(row).to_string(),
            arrow_schema::DataType::Int32 => col
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(row)
                .to_string(),
            arrow_schema::DataType::Utf8View => col.as_string_view().value(row).to_owned(),
            arrow_schema::DataType::Utf8 => col.as_string::<i32>().value(row).to_owned(),
            arrow_schema::DataType::Struct(_) => {
                let s = col.as_struct();
                let fields = s
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(i, child)| {
                        let name = s.column_names().get(i).cloned().unwrap_or_default();
                        let value = if child.is_null(row) {
                            "null".to_owned()
                        } else {
                            match child.data_type() {
                                arrow_schema::DataType::Utf8View => {
                                    format!("\"{}\"", child.as_string_view().value(row))
                                }
                                arrow_schema::DataType::Utf8 => {
                                    format!("\"{}\"", child.as_string::<i32>().value(row))
                                }
                                _ => cell(child, row),
                            }
                        };
                        format!("\"{name}\":{value}")
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{{fields}}}")
            }
            other => panic!("unrendered test value of {other}"),
        }
    }

    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        let col = batch.column(0);
        for row in 0..batch.num_rows() {
            out.push(cell(col, row));
        }
    }
    out
}

fn error(s: &Session, sql: &str) -> ErrorClass {
    match block_on(s.sql(sql)) {
        Ok(_) => panic!("{sql} resolved"),
        Err(e) => e.error_class(),
    }
}

#[test]
fn named_struct_and_struct_construct_structs() {
    let s = session();
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1, 'b', 'x')"),
        ["{\"a\":1,\"b\":\"x\"}"]
    );
    assert_eq!(
        rows(&s, "SELECT struct(1, 'a')"),
        ["{\"col1\":1,\"col2\":\"a\"}"]
    );
    assert_eq!(
        rows(&s, "SELECT STRUCT('alpha', 'beta')"),
        ["{\"col1\":\"alpha\",\"col2\":\"beta\"}"]
    );
    assert_eq!(
        rows(&s, "SELECT STRUCT('alpha' AS A, 'beta' AS B)"),
        ["{\"A\":\"alpha\",\"B\":\"beta\"}"]
    );
    // Null values are fields, not null structs.
    assert_eq!(
        rows(&s, "SELECT named_struct('a', NULL, 'b', 2)"),
        ["{\"a\":null,\"b\":2}"]
    );
    // Nesting.
    assert_eq!(
        rows(&s, "SELECT named_struct('a', named_struct('x', 1))"),
        ["{\"a\":{\"x\":1}}"]
    );
    // No arguments: the empty struct.
    assert_eq!(rows(&s, "SELECT struct()"), ["{}"]);
}

#[test]
fn struct_field_access_reads_by_name() {
    let s = session();
    assert_eq!(rows(&s, "SELECT named_struct('a', 1, 'b', 'x').b"), ["x"]);
    assert_eq!(rows(&s, "SELECT STRUCT(1 AS a, 2 AS b).b"), ["2"]);
    // A null struct gives a null field; a null field stays null.
    assert_eq!(rows(&s, "SELECT CAST(NULL AS STRUCT<a: INT>).a"), ["NULL"]);
    assert_eq!(
        rows(&s, "SELECT named_struct('a', CAST(NULL AS INT)).a"),
        ["NULL"]
    );
}

#[test]
fn struct_field_names_follow_case_sensitivity() {
    let s = session();
    assert_eq!(rows(&s, "SELECT named_struct('A', 1).a"), ["1"]);
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    assert_eq!(
        error(&s, "SELECT named_struct('A', 1).a"),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
    assert_eq!(rows(&s, "SELECT named_struct('A', 1).A"), ["1"]);
}

#[test]
fn struct_errors_name_their_classes() {
    let s = session();
    // Odd positions take only foldable strings.
    assert_eq!(
        error(&s, "SELECT named_struct(1, 2)"),
        ErrorClass::DatatypeMismatchCreateNamedStructWithoutFoldableString
    );
    assert_eq!(
        error(&s, "SELECT named_struct('a', 1, true, 2)"),
        ErrorClass::DatatypeMismatchCreateNamedStructWithoutFoldableString
    );
    // A null name is unexpected.
    assert_eq!(
        error(&s, "SELECT named_struct(CAST(NULL AS STRING), 1)"),
        ErrorClass::DatatypeMismatchUnexpectedNull
    );
    // An untyped null is not a string at all.
    assert_eq!(
        error(&s, "SELECT named_struct(NULL, 1)"),
        ErrorClass::DatatypeMismatchCreateNamedStructWithoutFoldableString
    );
    // Odd counts are an arity error.
    assert_eq!(
        error(&s, "SELECT named_struct('a')"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    // Field access on a non-struct.
    assert_eq!(
        error(&s, "SELECT (1).a"),
        ErrorClass::InvalidExtractBaseFieldType
    );
    assert_eq!(
        error(&s, "SELECT named_struct('a', 1).b"),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
}

#[test]
fn struct_comparison_follows_field_order() {
    let s = session();
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) = named_struct('a', 1)"),
        ["true"]
    );
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) = named_struct('a', 2)"),
        ["false"]
    );
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) <> named_struct('a', 2)"),
        ["true"]
    );
    // Null fields make equality null unless a field decides otherwise.
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT)) = named_struct('a', 1)"
        ),
        ["NULL"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) = named_struct('a', 2, 'b', 1)"
        ),
        ["NULL"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) = named_struct('a', 2, 'b', 2)"
        ),
        ["false"]
    );
    // Ordering is lexicographic.
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) < named_struct('a', 2)"),
        ["true"]
    );
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 2) < named_struct('a', 1)"),
        ["false"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', 1, 'b', 2) < named_struct('a', 1, 'b', 3)"
        ),
        ["true"]
    );
}
