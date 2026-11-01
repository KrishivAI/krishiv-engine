// SPDX-License-Identifier: Apache-2.0
//! `array_max`, `array_min`, `array_position`, `array_remove`,
//! `array_repeat` and `cardinality` end to end: every row, name and error class below is what a
//! Spark 4.2.0 run returned for the statement.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, Decimal128Array};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

/// One cell as Spark prints it: `NULL`, `true`, numbers (floats with `.0`),
/// strings as they are, `[a, b]` for a list and `{a, b}` for a struct.
fn render(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) || *c.data_type() == DataType::Null {
        return "NULL".to_owned();
    }
    let float = |v: f64| {
        if v.is_nan() {
            "NaN".to_owned()
        } else {
            format!("{v:?}")
        }
    };
    match c.data_type() {
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Float32 => float(f64::from(c.as_primitive::<Float32Type>().value(row))),
        DataType::Float64 => float(c.as_primitive::<Float64Type>().value(row)),
        DataType::Decimal128(..) => c
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .map(|a| a.value_as_string(row))
            .unwrap_or_default(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        DataType::List(_) => {
            let items = c.as_list::<i32>().value(row);
            let cells: Vec<String> = (0..items.len())
                .map(|i| render(items.as_ref(), i))
                .collect();
            format!("[{}]", cells.join(", "))
        }
        DataType::Struct(_) => {
            let s = c.as_struct();
            let cells: Vec<String> = s
                .columns()
                .iter()
                .map(|col| render(col.as_ref(), row))
                .collect();
            format!("{{{}}}", cells.join(", "))
        }
        other => format!("<{other}>"),
    }
}

fn run(s: &Session, sql: &str) -> Result<Vec<String>, KrishivError> {
    let df = block_on(s.sql(sql))?;
    let mut out = Vec::new();
    for batch in block_on(df.collect())? {
        for row in 0..batch.num_rows() {
            out.push(render(batch.column(0).as_ref(), row));
        }
    }
    Ok(out)
}

const ROWS: &[(&str, &str)] = &[
    ("array_max(array(1,NULL,3))", "3"),
    ("array_min(array(1,NULL,3))", "1"),
    ("array_max(array())", "NULL"),
    ("array_min(array(NULL))", "NULL"),
    ("array_max(cast(NULL as array<int>))", "NULL"),
    ("array_max(NULL)", "NULL"),
    ("array_max(array(double('nan'), 1.0D))", "NaN"),
    ("array_min(array(double('nan'), 1.0D))", "1.0"),
    ("array_max(array('a','b'))", "b"),
    ("array_max(array(array(1),array(2)))", "[2]"),
    (
        "array_max(array(named_struct('a',1,'b',2), named_struct('a',1,'b',3)))",
        "{1, 3}",
    ),
    ("array_min(array(double('-0.0'), double('0.0')))", "-0.0"),
    ("array_min(array(double('0.0'), double('-0.0')))", "0.0"),
    ("array_max(array(double('0.0'), double('-0.0')))", "0.0"),
    ("array_position(array(1,2,3), 3)", "3"),
    ("array_position(array(1,2,3), 9)", "0"),
    ("array_position(array(1,NULL), NULL)", "NULL"),
    ("array_position(array(NULL), 1)", "0"),
    ("array_position(array(1,2), cast(2 as bigint))", "2"),
    ("array_position(array(1,2), 1.0D)", "1"),
    ("array_position(array(1.5D,2.0D), 1)", "0"),
    ("array_position(cast(NULL as array<int>), 1)", "NULL"),
    ("array_position(array(array(1),array(2)), array(2))", "2"),
    ("array_remove(array(1,NULL,2,1), 1)", "[NULL, 2]"),
    ("array_remove(array(1,NULL), cast(NULL as int))", "NULL"),
    ("array_remove(cast(NULL as array<int>), 1)", "NULL"),
    ("array_remove(array(1,2), cast(1 as bigint))", "[2]"),
    ("array_remove(array(1,2), NULL)", "NULL"),
    (
        "array_remove(array(double('nan'), 1.0D, 0.0D, double('-0.0')), 0.0D)",
        "[NaN, 1.0]",
    ),
    ("array_repeat(1, 3)", "[1, 1, 1]"),
    ("array_repeat('a', 0)", "[]"),
    ("array_repeat(1, -1)", "[]"),
    ("array_repeat(NULL, 2)", "[NULL, NULL]"),
    ("array_repeat(1, NULL)", "NULL"),
    ("array_repeat(array(1), 2)", "[[1], [1]]"),
    ("array_repeat(cast(NULL as int), 1)", "[NULL]"),
    ("cardinality(array(1,2))", "2"),
    ("cardinality(map(1,2))", "1"),
    ("cardinality(NULL)", "NULL"),
    ("cardinality(cast(NULL as array<int>))", "NULL"),
];

#[test]
fn rows_follow_spark() {
    let s = Session::new().unwrap();
    for (sql, expected) in ROWS {
        let got = run(&s, &format!("SELECT {sql}")).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert_eq!(got, [(*expected).to_owned()], "{sql}");
    }
}

const ERRORS: &[(&str, &str)] = &[
    ("array_max(1)", "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"),
    (
        "array_max(array(map(1,2)))",
        "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE",
    ),
    (
        "array_position(1, 1)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    ("array_position(NULL, 1)", "DATATYPE_MISMATCH.NULL_TYPE"),
    (
        "array_position(array(1,2), '2')",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_position(array(1,2), 2.0)",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_position(array(map(1,2)), NULL)",
        "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE",
    ),
    (
        "array_remove(1, 1)",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_remove(NULL, 1)",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_remove(array(map(1,2)), map(1,2))",
        "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE",
    ),
    (
        "array_repeat(1, '2')",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "array_repeat(1, cast(2 as bigint))",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "array_repeat(1, 3.0)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    ("cardinality(1)", "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"),
];

#[test]
fn errors_follow_spark() {
    let s = Session::new().unwrap();
    for (sql, class) in ERRORS {
        let sql = format!("SELECT {sql}");
        let err = match block_on(s.sql(&sql)) {
            Ok(df) => block_on(df.collect()).expect_err(&sql),
            Err(e) => e,
        };
        assert_eq!(err.error_class().name(), *class, "{sql}: {err}");
    }
}

#[test]
fn rows_over_columns() {
    let s = Session::new().unwrap();
    let got = run(
        &s,
        "SELECT array_position(a, v) FROM VALUES (array(1,2), 2), (NULL, 1), (array(3), NULL), (array(5,6), 6) t(a, v)",
    )
    .unwrap();
    assert_eq!(got, ["2", "NULL", "NULL", "2"]);
    let got = run(
        &s,
        "SELECT array_max(a) FROM VALUES (array(1,9,3)), (NULL), (array()), (array(4,NULL)) t(a)",
    )
    .unwrap();
    assert_eq!(got, ["9", "NULL", "NULL", "4"]);
    let got = run(
        &s,
        "SELECT array_repeat(x, n) FROM VALUES ('a', 2), ('b', 0), (NULL, 1) t(x, n)",
    )
    .unwrap();
    assert_eq!(got, ["[a, a]", "[]", "[NULL]"]);
    let got = run(
        &s,
        "SELECT array_remove(a, v) FROM VALUES (array(1,2,1), 1), (array(3,4), 9) t(a, v)",
    )
    .unwrap();
    assert_eq!(got, ["[2]", "[3, 4]"]);
}
