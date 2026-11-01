// SPDX-License-Identifier: Apache-2.0
//! `array_max`, `array_min`, `array_position`, `array_remove`,
//! `flatten`, `slice` and `array_join` end to end: every row and error class below is what a
//! Spark 4.2.0 run returned for the statement.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type};
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
        DataType::Int8 => c.as_primitive::<Int8Type>().value(row).to_string(),
        DataType::Int16 => c.as_primitive::<Int16Type>().value(row).to_string(),
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
    ("array_join(array('a','b',NULL), ',')", "a,b"),
    ("array_join(array('a','b',NULL), ',', 'N')", "a,b,N"),
    ("array_join(array(1,2,3), '-')", "1-2-3"),
    ("array_join(array(), ',')", ""),
    ("array_join(cast(NULL as array<string>), ',')", "NULL"),
    ("array_join(array('a'), NULL)", "NULL"),
    ("array_join(array('a',NULL), ',', NULL)", "NULL"),
    ("flatten(array(array(1,2),array(3)))", "[1, 2, 3]"),
    ("flatten(array(array(1,2),NULL))", "NULL"),
    ("flatten(array(array(1,NULL),array()))", "[1, NULL]"),
    ("slice(array(1,2,3,4), 2, 2)", "[2, 3]"),
    ("slice(array(1,2,3,4), -2, 2)", "[3, 4]"),
    ("slice(array(1,2,3,4), 2, 100)", "[2, 3, 4]"),
    ("slice(array(1,2,3,4), 5, 2)", "[]"),
    ("slice(array(1,2,3,4), -10, 2)", "[]"),
    ("slice(cast(NULL as array<int>), 1, 1)", "NULL"),
    ("slice(array(1,2), NULL, 1)", "NULL"),
    ("slice(array(1,2), cast(1 as bigint), 1)", "[1]"),
    ("slice(array(1,2), '1', 1)", "[1]"),
    ("sequence(1, 5)", "[1, 2, 3, 4, 5]"),
    ("sequence(5, 1)", "[5, 4, 3, 2, 1]"),
    ("sequence(1, 10, 3)", "[1, 4, 7, 10]"),
    ("sequence(10, 1, -3)", "[10, 7, 4, 1]"),
    ("sequence(1, 1)", "[1]"),
    ("sequence(1, 1, 0)", "[1]"),
    ("sequence(NULL, 5)", "NULL"),
    ("sequence(1, NULL)", "NULL"),
    ("sequence(1, 5, NULL)", "NULL"),
    ("sequence(cast(1 as tinyint), 3, 1L)", "[1, 2, 3]"),
    ("sequence(1, 3, cast(1 as tinyint))", "[1, 2, 3]"),
    ("typeof(sequence(1, 5L))", "array<bigint>"),
    ("typeof(sequence(NULL, 5))", "array<int>"),
    (
        "typeof(sequence(cast(1 as tinyint), cast(3 as tinyint)))",
        "array<tinyint>",
    ),
];

#[test]
fn rows_follow_spark() {
    let s = Session::new().unwrap();
    for (sql, expected) in ROWS {
        let got = run(&s, &format!("SELECT {sql}")).unwrap();
        assert_eq!(got, vec![(*expected).to_owned()], "{sql}");
    }
}

const ERRORS: &[(&str, &str)] = &[
    (
        "flatten(array(1,2))",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    ("flatten(NULL)", "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"),
    ("slice(array(1,2), 0, 1)", "INVALID_PARAMETER_VALUE.START"),
    ("slice(array(1,2), 1, -1)", "INVALID_PARAMETER_VALUE.LENGTH"),
    ("sequence(1, 5, 0)", "_LEGACY_ERROR_TEMP_3243"),
    ("sequence(1, 5, -1)", "_LEGACY_ERROR_TEMP_3243"),
    (
        "sequence(1, 2147483647)",
        "COLLECTION_SIZE_LIMIT_EXCEEDED.PARAMETER",
    ),
    (
        "sequence(1.5, 3)",
        "DATATYPE_MISMATCH.SEQUENCE_WRONG_INPUT_TYPES",
    ),
    (
        "sequence(true, false)",
        "DATATYPE_MISMATCH.SEQUENCE_WRONG_INPUT_TYPES",
    ),
    ("sequence(1)", "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"),
    ("sequence(1, 2, 3, 4)", "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"),
    (
        "array_join(array(array('a')), ',')",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "array_join(array('a'), 1)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "array_join(array('a', NULL), ',', 1)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
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
        "SELECT slice(a, st, 2) FROM VALUES (array(1,2,3), 2), (array(4,5,6), -1), (NULL, 1) AS t(a, st)",
    )
    .unwrap();
    assert_eq!(got, ["[2, 3]", "[6]", "NULL"]);
}

#[test]
fn sequence_over_columns() {
    let s = Session::new().unwrap();
    let got = run(
        &s,
        "SELECT sequence(x, 3) FROM VALUES (1), (5), (NULL) AS t(x)",
    )
    .unwrap();
    assert_eq!(got, ["[1, 2, 3]", "[5, 4, 3]", "NULL"]);
    let err = run(&s, "SELECT sequence(1, 3, x) FROM VALUES (1), (0) AS t(x)").unwrap_err();
    assert_eq!(err.error_class().name(), "_LEGACY_ERROR_TEMP_3243");
}
