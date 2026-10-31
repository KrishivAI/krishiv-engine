// SPDX-License-Identifier: Apache-2.0
//! `array_contains`, `array_distinct`, `array_union`, `sort_array` and
//! `arrays_zip` end to end: every row, name and error class below is what a
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
    if c.is_null(row) {
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
    ("array_contains(array(1,2,3), 2)", "true"),
    ("array_contains(array(1,2,3), 2L)", "true"),
    ("array_contains(array(1,2,3), 2.5)", "false"),
    ("array_contains(array(1,2,NULL), 5)", "NULL"),
    ("array_contains(array(1,2,NULL), 2)", "true"),
    ("array_contains(array(1,2), cast(NULL as int))", "NULL"),
    ("array_contains(cast(NULL as array<int>), 1)", "NULL"),
    ("array_contains(array(array(1),array(2)), array(2))", "true"),
    ("array_contains(array(0.0D), -0.0D)", "true"),
    (
        "array_contains(array(double('nan')), double('nan'))",
        "true",
    ),
    (
        "array_contains(array(named_struct('a',1)), named_struct('a',1))",
        "true",
    ),
    (
        "array_contains(array(named_struct('a', -0.0D)), named_struct('a', 0.0D))",
        "true",
    ),
    (
        "array_contains(array(array(double('nan'))), array(double('nan')))",
        "true",
    ),
    (
        "array_contains(array(date'2020-01-01'), timestamp'2020-01-01 00:00:00')",
        "true",
    ),
    ("array_contains(array(cast(1 as decimal(5,2))), 1)", "true"),
    ("array_contains(array(cast(1 as tinyint)), 300)", "false"),
    ("array_contains(array(1), 2147483648)", "false"),
    ("array_contains(array(1.5D), 1)", "false"),
    ("array_contains(array(1,2), 2.0D)", "true"),
    ("array_contains(array('a','b'), 'a')", "true"),
    ("sort_array(array(3,1,NULL,2))", "[NULL, 1, 2, 3]"),
    ("sort_array(array(3,1,NULL,2), false)", "[3, 2, 1, NULL]"),
    ("sort_array(array('b','a',NULL))", "[NULL, a, b]"),
    (
        "sort_array(array(double('nan'), 1.0D, -0.0D, 0.0D, NULL))",
        "[NULL, -0.0, 0.0, 1.0, NaN]",
    ),
    (
        "sort_array(array(double('nan'), 1.0D, -0.0D, 0.0D, NULL), false)",
        "[NaN, 1.0, -0.0, 0.0, NULL]",
    ),
    ("sort_array(array(0.0D, -0.0D))", "[0.0, -0.0]"),
    ("sort_array(array(-0.0D, 0.0D), false)", "[-0.0, 0.0]"),
    (
        "sort_array(array(array(2,1),array(1,5),NULL))",
        "[NULL, [1, 5], [2, 1]]",
    ),
    (
        "sort_array(array(array(1,NULL), array(1)))",
        "[[1], [1, NULL]]",
    ),
    (
        "sort_array(array(array(1,2), array(1)), false)",
        "[[1, 2], [1]]",
    ),
    (
        "sort_array(array(named_struct('a', 1, 'b', 'x'), named_struct('a', 1, 'b', NULL), named_struct('a', NULL, 'b', 'y')))",
        "[{NULL, y}, {1, NULL}, {1, x}]",
    ),
    ("sort_array(NULL)", "NULL"),
    ("sort_array(array(1), NULL)", "NULL"),
    ("sort_array(array(3,1), cast(NULL as boolean))", "NULL"),
    ("sort_array(cast(NULL as array<int>))", "NULL"),
    ("sort_array(array(true,false,NULL))", "[NULL, false, true]"),
    ("sort_array(array(2.5BD,1.5BD))", "[1.5, 2.5]"),
    ("array_distinct(array(1,2,2,NULL,NULL,1))", "[1, 2, NULL]"),
    ("array_distinct(array('b','a','b'))", "[b, a]"),
    ("array_distinct(cast(NULL as array<int>))", "NULL"),
    ("array_distinct(NULL)", "NULL"),
    (
        "array_distinct(array(array(1),array(1),array(2,NULL),array(2,NULL)))",
        "[[1], [2, NULL]]",
    ),
    (
        "array_distinct(array(double('nan'), 0.0D, -0.0D, double('nan')))",
        "[NaN, 0.0]",
    ),
    ("array_distinct(array(-0.0D, 0.0D))", "[0.0]"),
    ("array_distinct(array(-0.0D))", "[0.0]"),
    ("array_distinct(array(1.0D, -0.0D, 0.0D))", "[1.0, 0.0]"),
    (
        "array_distinct(array(cast(-0.0 as float), cast(0.0 as float)))",
        "[0.0]",
    ),
    (
        "array_distinct(array(array(-0.0D), array(0.0D)))",
        "[[0.0]]",
    ),
    (
        "array_distinct(array(named_struct('a', -0.0D), named_struct('a', 0.0D)))",
        "[{0.0}]",
    ),
    (
        "array_distinct(array(named_struct('a', double('nan')), named_struct('a', double('nan'))))",
        "[{NaN}]",
    ),
    ("array_distinct(array(1.5BD,1.50BD))", "[1.50]"),
    (
        "array_union(array(1,2,2), array(2,3,NULL,NULL))",
        "[1, 2, 3, NULL]",
    ),
    ("array_union(array(1), array(1L, 2L))", "[1, 2]"),
    ("array_union(array(1), cast(NULL as array<int>))", "NULL"),
    ("array_union(array(1, NULL), array(NULL))", "[1, NULL]"),
    (
        "array_union(array(1, NULL), array(2, NULL))",
        "[1, NULL, 2]",
    ),
    ("array_union(array(3,1,2), array(2,5))", "[3, 1, 2, 5]"),
    (
        "array_union(array(array(1),array(1)), array(array(2)))",
        "[[1], [2]]",
    ),
    (
        "array_union(array(named_struct('a',1)), array(named_struct('a',1)))",
        "[{1}]",
    ),
    (
        "array_union(array(double('nan'), 0.0D), array(double('nan'), -0.0D))",
        "[NaN, 0.0]",
    ),
    ("array_union(array(-0.0D), array(1.0D))", "[0.0, 1.0]"),
    ("array_union(array(1.0D, -0.0D), array(0.0D))", "[1.0, 0.0]"),
    (
        "arrays_zip(array(1,2,3), array('a','b'))",
        "[{1, a}, {2, b}, {3, NULL}]",
    ),
    (
        "arrays_zip(array(1,2,3), array('a','b'), array(1.5D))",
        "[{1, a, 1.5}, {2, b, NULL}, {3, NULL, NULL}]",
    ),
    ("arrays_zip(array(1), NULL)", "NULL"),
    ("arrays_zip(array(1), cast(NULL as array<int>))", "NULL"),
    ("arrays_zip()", "[]"),
    ("arrays_zip(array(1))", "[{1}]"),
    (
        "arrays_zip(array(1,NULL), array(NULL, 'x'))",
        "[{1, NULL}, {NULL, x}]",
    ),
    (
        "arrays_zip(array(1,2), array(3,4,5))",
        "[{1, 3}, {2, 4}, {NULL, 5}]",
    ),
    (
        "arrays_zip(array(named_struct('a',1)), array(2))",
        "[{{1}, 2}]",
    ),
    (
        "size(arrays_zip(array(1, 2, 3), array(4), array(7, 8, 9, 10)))",
        "4",
    ),
    (
        "size(arrays_zip(array(), array(1, 2, 3), array(4), array(7, 8, 9, 10)))",
        "4",
    ),
    (
        "size(arrays_zip(array(1, 2, 3), array(4), null, array(7, 8, 9, 10)))",
        "NULL",
    ),
    (
        "isnotnull(arrays_zip(array(1, 2, 3), NULL, array(4), array(7, 8, 9, 10)))",
        "false",
    ),
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
    (
        "array_contains(array(1,2,3), '2')",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_contains(array('a','b'), 1)",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_contains(array(true), 1)",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_contains(array(date'2020-01-01'), '2020-01-01')",
        "DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES",
    ),
    (
        "array_contains(array(1,2,NULL), NULL)",
        "DATATYPE_MISMATCH.NULL_TYPE",
    ),
    ("array_contains(NULL, 1)", "DATATYPE_MISMATCH.NULL_TYPE"),
    (
        "array_contains(array(1), NULL)",
        "DATATYPE_MISMATCH.NULL_TYPE",
    ),
    (
        "array_contains(1, 1)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "array_contains(array(map(1,2)), map(1,2))",
        "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE",
    ),
    (
        "array_contains(array(1,2), 1, 2)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "sort_array(array(map(1,2)))",
        "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE",
    ),
    ("sort_array(1)", "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"),
    (
        "sort_array(array(3,1), 1)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "sort_array(array(1), 'x')",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "sort_array(array(1,2), true, true)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "sort_array(array(1,2,3), (select true))",
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT",
    ),
    (
        "array_distinct(1)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "array_distinct(array(map(1,2)))",
        "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE",
    ),
    (
        "array_distinct(array(1,2), 3)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "array_union(array(1), array('a'))",
        "DATATYPE_MISMATCH.BINARY_ARRAY_DIFF_TYPES",
    ),
    (
        "array_union(array(1), NULL)",
        "DATATYPE_MISMATCH.BINARY_ARRAY_DIFF_TYPES",
    ),
    (
        "array_union(NULL, array(1))",
        "DATATYPE_MISMATCH.BINARY_ARRAY_DIFF_TYPES",
    ),
    (
        "array_union(array(1), 1)",
        "DATATYPE_MISMATCH.BINARY_ARRAY_DIFF_TYPES",
    ),
    (
        "array_union(array(map(1,2)), array(map(1,2)))",
        "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE",
    ),
    (
        "array_union(array(1,2),array(3,4),array(5))",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "arrays_zip(1, array(1))",
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

/// The error quotes the call as written, without the field names the
/// analyzer passes to `arrays_zip`.
#[test]
fn zip_errors_quote_the_call_as_written() {
    let s = Session::new().unwrap();
    let err = block_on(s.sql("SELECT arrays_zip(1, array(1))")).expect_err("not arrays");
    assert!(
        err.to_string().contains("\"arrays_zip(1, array(1))\""),
        "{err}"
    );
}

fn first_name(s: &Session, sql: &str) -> String {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.schema()).unwrap().field(0).name().clone()
}

/// `arrays_zip` names its fields after columns and struct fields; `sort_array`
/// shows its default order.
#[test]
fn names_follow_spark() {
    let s = Session::new().unwrap();
    for (sql, name) in [
        (
            "SELECT sort_array(a) FROM VALUES (array(1)) t(a)",
            "sort_array(a, true)",
        ),
        (
            "SELECT arrays_zip(a, b) FROM VALUES (array(1), array('x')) t(a, b)",
            "arrays_zip(a, b)",
        ),
        (
            "SELECT arrays_zip(s.x, s.y) FROM VALUES (named_struct('x', array(1), 'y', array(2))) t(s)",
            "arrays_zip(s.x, s.y)",
        ),
    ] {
        assert_eq!(first_name(&s, sql), name, "{sql}");
    }
}

/// The struct fields of `arrays_zip` are the column names, field names, and
/// the position for any other expression.
#[test]
fn zip_fields_are_named_like_spark() {
    let s = Session::new().unwrap();
    let ty = |sql: &str| {
        let df = block_on(s.sql(sql)).unwrap();
        let schema = block_on(df.schema()).unwrap();
        let DataType::List(f) = schema.field(0).data_type().clone() else {
            panic!("{sql}: not a list")
        };
        let DataType::Struct(fields) = f.data_type().clone() else {
            panic!("{sql}: not a struct")
        };
        fields.iter().map(|f| f.name().clone()).collect::<Vec<_>>()
    };
    assert_eq!(
        ty("SELECT arrays_zip(a, b, array(1)) FROM VALUES (array(1,2), array('x')) t(a,b)"),
        ["a", "b", "2"]
    );
    assert_eq!(
        ty("SELECT arrays_zip(t.a, b) FROM VALUES (array(1,2), array('x')) t(a,b)"),
        ["a", "b"]
    );
    assert_eq!(
        ty(
            "SELECT arrays_zip(s.x, s.y) FROM VALUES (named_struct('x', array(1), 'y', array(2))) t(s)"
        ),
        ["x", "y"]
    );
    assert_eq!(
        ty("SELECT arrays_zip(a, a) FROM VALUES (array(1,2)) t(a)"),
        ["a", "a"]
    );
    assert_eq!(ty("SELECT arrays_zip(array(1), array(2))"), ["0", "1"]);
}

/// Over rows: the three-valued `array_contains`, a column input for the
/// set functions, and a NULL row.
#[test]
fn rows_over_columns() {
    let s = Session::new().unwrap();
    assert_eq!(
        run(
            &s,
            "SELECT array_contains(a, x) FROM VALUES (array(1,2), 2), (array(1,NULL), 3), (array(1,2), NULL), (NULL, 1) t(a,x)"
        )
        .unwrap(),
        ["true", "NULL", "NULL", "NULL"]
    );
    assert_eq!(
        run(
            &s,
            "SELECT array_union(a, b) FROM VALUES (array(-0.0D, 1.0D), array(0.0D)), (NULL, array(1.0D)) t(a,b)"
        )
        .unwrap(),
        ["[0.0, 1.0]", "NULL"]
    );
    assert_eq!(
        run(
            &s,
            "SELECT array_distinct(a) FROM VALUES (array(1,1,2)), (CAST(NULL AS ARRAY<INT>)), (array(3,3)) t(a)"
        )
        .unwrap(),
        ["[1, 2]", "NULL", "[3]"]
    );
    assert_eq!(
        run(
            &s,
            "SELECT sort_array(a, false) FROM VALUES (array(1,3,2)), (CAST(NULL AS ARRAY<INT>)) t(a)"
        )
        .unwrap(),
        ["[3, 2, 1]", "NULL"]
    );
    assert_eq!(
        run(
            &s,
            "SELECT arrays_zip(a, b) FROM VALUES (array(1,2), array('x')), (NULL, array('y')) t(a,b)"
        )
        .unwrap(),
        ["[{1, x}, {2, NULL}]", "NULL"]
    );
}

/// A lambda variable names its field as a column does, and string columns
/// meet string literals (Spark 4.2.0).
#[test]
fn lambda_variables_and_string_columns() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT transform(array(array(1,2)), x -> arrays_zip(x, x))")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert!(
        format!("{:?}", schema.field(0).data_type()).contains("\"x\""),
        "{schema:?}"
    );
    assert_eq!(
        run(
            &s,
            "SELECT array_union(a, array('x')) FROM VALUES (array('y','x','y')) t(a)"
        )
        .unwrap(),
        ["[y, x]"]
    );
    assert_eq!(
        run(
            &s,
            "SELECT array_contains(a, 'x') FROM VALUES (array('y','x')), (array('y')) t(a)"
        )
        .unwrap(),
        ["true", "false"]
    );
}
