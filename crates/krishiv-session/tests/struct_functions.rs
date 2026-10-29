// SPDX-License-Identifier: Apache-2.0
//! Struct functions end to end: `named_struct` and `struct` construction,
//! `STRUCT(...)` syntax, field access under `spark.sql.caseSensitive` and
//! struct comparison. Construction, field access and error classes follow
//! the matrix and golden fixtures (real Spark 4.2 runs); comparison treats a
//! null field as the smallest value and never returns NULL for a null field
//! (checked against a Spark 4.2 run).
//! `MAX`/`MIN` over structs follow the comparison order (field by field,
//! NULL fields smallest); window `MAX`/`MIN` over structs too.
//! Not here: star-except in `STRUCT`.
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

/// One value of `col` at `row`, as Spark renders it (`{"a":1}`, `NULL` at
/// the top level, `null` for a NULL field).
fn cell(col: &arrow_array::ArrayRef, row: usize) -> String {
    use arrow_array::cast::AsArray as _;

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
        arrow_schema::DataType::Int64 => col
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(row)
            .to_string(),
        arrow_schema::DataType::Utf8View => col.as_string_view().value(row).to_owned(),
        arrow_schema::DataType::Utf8 => col.as_string::<i32>().value(row).to_owned(),
        arrow_schema::DataType::Float64 => format!(
            "{:?}",
            col.as_primitive::<arrow_array::types::Float64Type>()
                .value(row)
        ),
        arrow_schema::DataType::Decimal128(_, scale) => {
            let v = col
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(row);
            dec_text(v, *scale)
        }
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

/// The unscaled decimal `v` at `scale`, as Spark prints it.
fn dec_text(v: i128, scale: i8) -> String {
    let scale = usize::try_from(scale).unwrap_or(0);
    let digits = v.unsigned_abs().to_string();
    let sign = if v < 0 { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = padded.split_at(padded.len() - scale);
    format!("{sign}{int}.{frac}")
}

/// Every row of `sql`, all columns joined by `|` (top-level NULLs are
/// `NULL`).
fn rows_all(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        for row in 0..batch.num_rows() {
            out.push(
                (0..batch.num_columns())
                    .map(|c| cell(batch.column(c), row))
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    out
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
    // A null field is a value like any other for equality (it equals only
    // another null), never a NULL result (Spark 4.2).
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT)) = named_struct('a', 1)"
        ),
        ["false"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) = named_struct('a', 2, 'b', 1)"
        ),
        ["false"]
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
    // A null field orders before every value and equals another null.
    for (sql, want) in [
        (
            "SELECT named_struct('a', 1, 'b', CAST(NULL AS INT)) < named_struct('a', 1, 'b', 2)",
            "true",
        ),
        (
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) > named_struct('a', 1, 'b', 1)",
            "false",
        ),
        (
            "SELECT named_struct('a', 1, 'b', 2) > named_struct('a', 1, 'b', CAST(NULL AS INT))",
            "true",
        ),
        (
            "SELECT named_struct('a', 1, 'b', CAST(NULL AS INT)) = named_struct('a', 1, 'b', CAST(NULL AS INT))",
            "true",
        ),
        // Equal null fields defer to the next field (Spark 4.2.0 run).
        (
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) < named_struct('a', CAST(NULL AS INT), 'b', 2)",
            "true",
        ),
        // Untyped NULL fields (a `NullArray` underneath) behave the same.
        (
            "SELECT named_struct('a', NULL) = named_struct('a', NULL)",
            "true",
        ),
        (
            "SELECT named_struct('a', NULL) < named_struct('a', NULL)",
            "false",
        ),
        (
            "SELECT named_struct('a', NULL) > named_struct('a', NULL)",
            "false",
        ),
        (
            "SELECT named_struct('a', NULL, 'b', 1) < named_struct('a', NULL, 'b', 2)",
            "true",
        ),
    ] {
        assert_eq!(rows(&s, sql), [want], "{sql}");
    }
}

/// `struct(...)` over a qualified star and over an aggregate (Spark 4.2.0:
/// the star's columns keep their names; a non-named child is `colN`).
#[test]
fn struct_of_a_star_and_of_an_aggregate() {
    let s = session();
    assert_eq!(
        rows(&s, "SELECT STRUCT(t.*) FROM (SELECT 1 AS a, 'x' AS b) AS t"),
        ["{\"a\":1,\"b\":\"x\"}"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT struct(k, sum(v)) FROM VALUES (1, 2), (1, 3) AS t(k, v) GROUP BY k"
        ),
        ["{\"k\":1,\"col2\":5}"]
    );
}

/// `MIN`/`MAX` over structs (Spark 4.2: field-by-field ordering, NULL
/// fields smallest and equal to one another, NaN the largest float; a NULL
/// struct row is skipped like any NULL input, all-NULL gives NULL).
#[test]
fn min_and_max_over_structs() {
    let s = session();
    // The golden `struct.sql` statement: schema and rows.
    assert_eq!(
        rows_all(&s, "SELECT MAX(STRUCT(1 AS a)), 2 AS b, b AS c GROUP BY b"),
        ["{\"a\":1}|2|2"]
    );
    // Simple, grouped and global.
    assert_eq!(
        rows(&s, "SELECT MAX(STRUCT(1 AS a, 2 AS b))"),
        ["{\"a\":1,\"b\":2}"]
    );
    assert_eq!(
        rows_all(
            &s,
            "SELECT k, MIN(s), MAX(s) FROM VALUES (1, named_struct('a', 1, 'b', 'y')), (1, named_struct('a', 1, 'b', 'x')), (1, named_struct('a', 2, 'b', 'a')) AS t(k, s) GROUP BY k"
        ),
        ["1|{\"a\":1,\"b\":\"x\"}|{\"a\":2,\"b\":\"a\"}"]
    );
    // A NULL field sorts before every value; NULL structs are skipped.
    assert_eq!(
        rows_all(
            &s,
            "SELECT MIN(s), MAX(s) FROM VALUES (named_struct('a', 1, 'b', CAST(NULL AS STRING))), (named_struct('a', 1, 'b', 'x')), (CAST(NULL AS STRUCT<a: INT, b: STRING>)) AS t(s)"
        ),
        ["{\"a\":1,\"b\":null}|{\"a\":1,\"b\":\"x\"}"]
    );
    // Only NULL structs: NULL.
    assert_eq!(
        rows_all(
            &s,
            "SELECT MIN(s), MAX(s) FROM VALUES (CAST(NULL AS STRUCT<a: INT>)) AS t(s)"
        ),
        ["NULL|NULL"]
    );
    // Nested structs, strings, decimals and floats with NaN.
    assert_eq!(
        rows_all(
            &s,
            "SELECT MIN(s), MAX(s) FROM (SELECT named_struct('n', named_struct('x', x), 'd', d, 't', t) AS s FROM VALUES (CAST('NaN' AS DOUBLE), CAST(1.00 AS DECIMAL(10, 2)), 'a'), (1.0, CAST(2.00 AS DECIMAL(10, 2)), 'z'), (CAST(NULL AS DOUBLE), CAST(0.50 AS DECIMAL(10, 2)), 'm') AS v(x, d, t))"
        ),
        ["{\"n\":{\"x\":null},\"d\":0.50,\"t\":\"m\"}|{\"n\":{\"x\":NaN},\"d\":1.00,\"t\":\"a\"}"]
    );
    // `MAX` and `MIN` over a struct with a map field is an ordering error
    // (Spark's `DATATYPE_MISMATCH.INVALID_ORDERING_TYPE`).
    for sql in [
        "SELECT MAX(CAST(NULL AS STRUCT<m: MAP<STRING, INT>>)) FROM VALUES (1) AS t(x)",
        "SELECT MIN(CAST(NULL AS STRUCT<m: MAP<STRING, INT>>)) FROM VALUES (1) AS t(x)",
    ] {
        assert_eq!(
            error(&s, sql),
            ErrorClass::DatatypeMismatchInvalidOrderingType
        );
    }
}

/// Window `MIN`/`MAX` over structs: whole partitions, running frames and
/// `ROWS` frames all use the aggregate's struct comparison.
#[test]
fn min_and_max_over_structs_in_windows() {
    let s = session();
    let input = "VALUES (1, 1, named_struct('a', 1)), (1, 2, CAST(NULL AS STRUCT<a: INT>)), (1, 3, named_struct('a', 2)), (2, 1, CAST(NULL AS STRUCT<a: INT>)) AS t(k, o, s)";
    let mut whole = rows_all(
        &s,
        &format!("SELECT k, o, MAX(s) OVER (PARTITION BY k) FROM {input}"),
    );
    whole.sort();
    assert_eq!(
        whole,
        [
            "1|1|{\"a\":2}",
            "1|2|{\"a\":2}",
            "1|3|{\"a\":2}",
            "2|1|NULL",
        ]
    );
    // Running peers (the default frame): NULL structs are skipped.
    let mut running = rows_all(
        &s,
        &format!("SELECT k, o, MIN(s) OVER (PARTITION BY k ORDER BY o) FROM {input}"),
    );
    running.sort();
    assert_eq!(
        running,
        [
            "1|1|{\"a\":1}",
            "1|2|{\"a\":1}",
            "1|3|{\"a\":1}",
            "2|1|NULL",
        ]
    );
    // Running rows: each row's frame is itself and its predecessor (the
    // NULL row contributes nothing).
    let mut rows_frame = rows_all(
        &s,
        &format!(
            "SELECT k, MAX(s) OVER (PARTITION BY k ORDER BY o ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM {input}"
        ),
    );
    rows_frame.sort();
    assert_eq!(
        rows_frame,
        ["1|{\"a\":1}", "1|{\"a\":1}", "1|{\"a\":2}", "2|NULL"]
    );
    // A `ROWS` frame around the row.
    let mut frame = rows_all(
        &s,
        &format!(
            "SELECT k, o, MIN(s) OVER (PARTITION BY k ORDER BY o ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM {input}"
        ),
    );
    frame.sort();
    assert_eq!(
        frame,
        [
            "1|1|{\"a\":1}",
            "1|2|{\"a\":1}",
            "1|3|{\"a\":2}",
            "2|1|NULL",
        ]
    );
}
