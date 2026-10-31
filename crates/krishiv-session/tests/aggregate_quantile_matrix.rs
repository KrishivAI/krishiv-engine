// SPDX-License-Identifier: Apache-2.0
//! The percentile family and `approx_count_distinct` over inline data:
//! every row and error class below is what a Spark 4.2.0 run returned for the
//! statement (checked on the oracle, not written from memory).
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, Decimal64Array, Decimal128Array};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

/// One cell as the oracle's rows are written: `NULL`, a number without a
/// trailing `.0`, decimal text, or `[a, b]` for a list.
fn render(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".to_owned();
    }
    let float = |v: f64| {
        let t = format!("{v:?}");
        t.strip_suffix(".0").map_or(t.clone(), str::to_owned)
    };
    match c.data_type() {
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
        DataType::Decimal64(..) => c
            .as_any()
            .downcast_ref::<Decimal64Array>()
            .map(|a| a.value_as_string(row))
            .unwrap_or_default(),
        DataType::List(_) => {
            let items = c.as_list::<i32>().value(row);
            let cells: Vec<String> = (0..items.len())
                .map(|i| render(items.as_ref(), i))
                .collect();
            format!("[{}]", cells.join(", "))
        }
        other => format!("<{other}>"),
    }
}

fn rows(s: &Session, sql: &str) -> Result<Vec<Vec<String>>, krishiv_common::error::KrishivError> {
    let df = block_on(s.sql(sql))?;
    let mut out = Vec::new();
    for batch in block_on(df.collect())? {
        for row in 0..batch.num_rows() {
            out.push(
                (0..batch.num_columns())
                    .map(|c| render(batch.column(c).as_ref(), row))
                    .collect(),
            );
        }
    }
    Ok(out)
}

type Expected = (&'static str, &'static [&'static [&'static str]]);

const ROWS: &[Expected] = &[
    (
        "SELECT percentile(v, array(0.25, 0.75)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[3, 8]"]],
    ),
    (
        "SELECT percentile(v, 0.5, 2) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(v, array(0.1, 0.9), 3) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[1.6, 9.400000000000002]"]],
    ),
    (
        "SELECT percentile_approx(v, array(0.25, 0.5, 0.75), 1000) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[3, 5, 8]"]],
    ),
    (
        "SELECT percentile_approx(v, 0.5, 100) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_approx(CAST(v AS DECIMAL(10,2)), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5.00"]],
    ),
    (
        "SELECT percentile_approx(CAST(v AS DOUBLE), 0.9) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["10"]],
    ),
    (
        "SELECT percentile_approx(CAST(v AS BIGINT), array(0.5)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[5]"]],
    ),
    (
        "SELECT percentile_approx(CAST(v AS SMALLINT), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_approx(CAST(v AS TINYINT), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_approx(CAST(v AS FLOAT), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT approx_percentile(v, 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT median(v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT median(CAST(v AS DECIMAL(10,2))) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY v DESC) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["3"]],
    ),
    (
        "SELECT percentile_disc(0.3) WITHIN GROUP (ORDER BY v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["3"]],
    ),
    (
        "SELECT percentile_disc(0.3) WITHIN GROUP (ORDER BY v DESC) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["8"]],
    ),
    (
        "SELECT percentile(CAST(v AS DECIMAL(10,2)), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(CAST(v AS BIGINT), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(CAST(v AS FLOAT), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(CAST(v AS TINYINT), 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(CAST(v AS SMALLINT), array(0.5, 1.0)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[5, 10]"]],
    ),
    (
        "SELECT percentile(v, 0.5) FILTER (WHERE v > 4) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["8"]],
    ),
    (
        "SELECT percentile_approx(v, 0.5) FILTER (WHERE v > 4) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["8"]],
    ),
    (
        "SELECT approx_count_distinct(v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(v, 0.01) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(v, 0.1) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["8"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS STRING)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS BIGINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS DOUBLE)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS FLOAT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS DECIMAL(10,2))) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS DECIMAL(30,2))) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(v > 4) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["2"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS SMALLINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(v AS TINYINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(CAST(CAST(v AS STRING) AS BINARY)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(DATE'2020-01-01') FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["1"]],
    ),
    (
        "SELECT k, approx_count_distinct(v) FROM VALUES (1,1),(1,2),(2,3),(2,3) t(k,v) GROUP BY k ORDER BY k",
        &[&["1", "2"], &["2", "1"]],
    ),
    (
        "SELECT k, percentile(v, 0.5), percentile_approx(v, 0.5), median(v) FROM VALUES (1,1),(1,2),(1,9),(2,3),(2,4) t(k,v) GROUP BY k ORDER BY k",
        &[&["1", "2", "2", "2"], &["2", "3.5", "3", "3.5"]],
    ),
    (
        "SELECT k, percentile(v, array(0.5, 0.9)) FROM VALUES (1,1),(1,2),(1,9),(2,3),(2,4) t(k,v) GROUP BY k ORDER BY k",
        &[&["1", "[2, 7.6]"], &["2", "[3.5, 3.9]"]],
    ),
    (
        "SELECT percentile(v, 0.5) FROM VALUES (CAST(NULL AS INT)) t(v)",
        &[&["NULL"]],
    ),
    (
        "SELECT percentile_approx(v, 0.5) FROM VALUES (CAST(NULL AS INT)) t(v)",
        &[&["NULL"]],
    ),
    (
        "SELECT approx_count_distinct(v) FROM VALUES (CAST(NULL AS INT)) t(v)",
        &[&["0"]],
    ),
    (
        "SELECT percentile(v, 0.5, f) FROM VALUES (1, 2), (3, 0), (5, 1) t(v, f)",
        &[&["1"]],
    ),
    (
        "SELECT percentile(v, 0.5, f) FROM VALUES (1, 2), (3, CAST(NULL AS INT)), (5, 1) t(v, f)",
        &[&["1"]],
    ),
];

const ERRORS: &[(&str, &str)] = &[
    (
        "SELECT percentile(v, 1.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile(v, -0.1) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile(v, v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT",
    ),
    (
        "SELECT percentile(v, NULL) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_NULL",
    ),
    (
        "SELECT percentile(v, 'a') FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "CAST_INVALID_INPUT",
    ),
    (
        "SELECT percentile(v, array(0.5, 2.0)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile(v, array(0.5, NULL)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "SELECT percentile(v, 0.5, 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "SELECT percentile_approx(v, 0.5, 0) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile_approx(v, 0.5, -5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile_approx(v, 0.5, v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT",
    ),
    (
        "SELECT percentile_approx(v, 1.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile_approx(v, NULL) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_NULL",
    ),
    (
        "SELECT percentile_approx('a', 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "CAST_INVALID_INPUT",
    ),
    (
        "SELECT percentile('a', 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "CAST_INVALID_INPUT",
    ),
    (
        "SELECT percentile_cont(1.5) WITHIN GROUP (ORDER BY v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile_disc(v) WITHIN GROUP (ORDER BY v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT",
    ),
    (
        "SELECT approx_count_distinct(v, v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "_LEGACY_ERROR_TEMP_1104",
    ),
    (
        "SELECT approx_count_distinct(v, NULL) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "_LEGACY_ERROR_TEMP_1104",
    ),
    (
        "SELECT approx_count_distinct(v, 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "FAILED_FUNCTION_CALL",
    ),
    (
        "SELECT percentile(v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "SELECT median(v, v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "SELECT percentile(v, 0.5, -3) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "NEGATIVE_VALUES_IN_FREQUENCY_EXPRESSION",
    ),
];

#[test]
fn rows_match_spark() {
    let s = Session::new().unwrap();
    let mut wrong = Vec::new();
    for (sql, want) in ROWS {
        let want: Vec<Vec<String>> = want
            .iter()
            .map(|r| r.iter().map(|c| (*c).to_owned()).collect())
            .collect();
        match rows(&s, sql) {
            Ok(got) if got == want => {}
            Ok(got) => wrong.push(format!("{sql}\n  want {want:?}\n  got  {got:?}")),
            Err(e) => wrong.push(format!(
                "{sql}\n  want {want:?}\n  error {}",
                e.error_class().name()
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} statements differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn error_classes_match_spark() {
    let s = Session::new().unwrap();
    let mut wrong = Vec::new();
    for (sql, class) in ERRORS {
        match rows(&s, sql) {
            Err(e) if e.error_class().name() == *class => {}
            Err(e) => wrong.push(format!(
                "{sql}\n  want {class}\n  got  {}",
                e.error_class().name()
            )),
            Ok(got) => wrong.push(format!("{sql}\n  want {class}\n  got rows {got:?}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} statements differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
