// SPDX-License-Identifier: Apache-2.0
//! Subscripts (`base[index]`) on arrays, maps and structs, end to end: the
//! rows below are Spark 4.2.0's results.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, Decimal128Array};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
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
    ("array(10, 20, 30)[0]", "10"),
    ("array(10, 20, 30)[2]", "30"),
    ("array(10, NULL)[1]", "NULL"),
    ("array(10, 20)[cast(NULL as int)]", "NULL"),
    ("cast(NULL as array<int>)[0]", "NULL"),
    ("array(array(1, 2), array(3, 4))[1][0]", "3"),
    ("map(1, 'a', 2, 'b')[2]", "b"),
    ("map(1, 'a', 2, 'b')[5]", "NULL"),
    ("map('k', array(7, 8))['k'][1]", "8"),
    ("named_struct('a', 1, 'b', 2)['b']", "2"),
    ("arrays_zip(array(1, 2), array('x', 'y'))[1]", "{2, y}"),
    ("arrays_zip(array(1, 2), array('x', 'y'))[1]['1']", "y"),
];

#[test]
fn subscripts_match_spark() {
    let s = Session::new().unwrap();
    for (expr, want) in ROWS {
        let got = run(&s, &format!("SELECT {expr}")).unwrap_or_else(|e| panic!("{expr}: {e}"));
        assert_eq!(got, [*want], "{expr}");
    }
}

#[test]
fn array_index_out_of_range_is_an_error_under_ansi() {
    let s = Session::new().unwrap();
    for expr in ["array(1, 2)[2]", "array(1, 2)[-1]"] {
        let err = run(&s, &format!("SELECT {expr}")).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InvalidArrayIndex, "{expr}");
    }
}

#[test]
fn a_subscript_is_named_as_written() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT array(1, 2)[0] AS x, map(1, 2)[1]")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(names, ["x", "map(1, 2)[1]"]);
}

#[test]
fn a_subscript_of_another_type_is_refused() {
    let s = Session::new().unwrap();
    let err = run(&s, "SELECT 1[0]").unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidExtractBaseFieldType);
    let err = run(&s, "SELECT named_struct('a', 1)[1]").unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidExtractFieldType);
}

#[test]
fn an_array_index_must_be_integral() {
    let s = Session::new().unwrap();
    for ansi in ["true", "false"] {
        block_on(s.sql(&format!("SET spark.sql.ansi.enabled = {ansi}"))).unwrap();
        for expr in ["array(1, 2)['1']", "array(1, 2)[1.0]", "array(1, 2)['x']"] {
            let err = run(&s, &format!("SELECT {expr}")).unwrap_err();
            assert_eq!(
                err.error_class(),
                ErrorClass::DatatypeMismatchUnexpectedInputType,
                "{expr}"
            );
        }
    }
    // Any integral type is an index.
    assert_eq!(
        run(
            &s,
            "SELECT array(1, 2)[1L], array(1, 2)[cast(0 as tinyint)]"
        )
        .unwrap()
        .len(),
        1
    );
}

#[test]
fn a_field_of_an_array_of_structs_is_refused_not_misread() {
    let s = Session::new().unwrap();
    for expr in [
        "array(named_struct('a', 1))['a']",
        "arrays_zip(array(1, 2), array('x', 'y'))['1']",
    ] {
        let err = run(&s, &format!("SELECT {expr}")).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{expr}"
        );
    }
}
