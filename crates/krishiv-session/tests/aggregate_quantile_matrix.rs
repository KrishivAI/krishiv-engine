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
    (
        "SELECT percentile(v, 0.25 + 0.25) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(v, CAST('0.5' AS DOUBLE)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(v, array(0.5, 1)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[5, 10]"]],
    ),
    (
        "SELECT percentile(v, array(CAST(0.5 AS FLOAT), CAST(0.9 AS DECIMAL(3,2)))) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[5, 9.2]"]],
    ),
    (
        "SELECT percentile_approx(v, array(0.5 + 0.25, 0.1)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["[8, 1]"]],
    ),
    (
        "SELECT percentile_approx(v, 0.5, 100 + 100) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_approx(v, 0.5, CAST(500 AS BIGINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_approx(v, 0.5, CAST(500 AS SMALLINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(v, 0.5, CAST(2 AS BIGINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(v, 0.5, CAST(2 AS SMALLINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(v, 0.5, CAST(2 AS TINYINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile(v, 0.5, 0) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["NULL"]],
    ),
    (
        "SELECT approx_count_distinct(v, 0.2) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
    ),
    (
        "SELECT approx_count_distinct(v, 0.3) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["11"]],
    ),
    (
        "SELECT median(CAST(v AS DOUBLE)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT median(CAST(v AS FLOAT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT median(CAST(v AS BIGINT)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY CAST(v AS DECIMAL(10,2))) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY CAST(v AS DOUBLE)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY CAST(v AS DECIMAL(10,2))) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY v) FILTER (WHERE v > 3) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["7.5"]],
    ),
    (
        "SELECT percentile(id, 0.5), percentile(id, array(0.1, 0.9)) FROM range(100000)",
        &[&["49999.5", "[9999.900000000001, 89999.1]"]],
    ),
    (
        "SELECT approx_count_distinct(id), approx_count_distinct(id, 0.01), approx_count_distinct(CAST(id AS STRING)) FROM range(100000)",
        &[&["95546", "100425", "97299"]],
    ),
    (
        "SELECT id % 3 g, approx_count_distinct(id), percentile(id, 0.5) FROM range(30000) GROUP BY id % 3 ORDER BY g",
        &[
            &["0", "10009", "14998.5"],
            &["1", "10092", "14999.5"],
            &["2", "10116", "15000.5"],
        ],
    ),
    (
        "SELECT approx_count_distinct(CAST(id AS DOUBLE)), approx_count_distinct(CAST(id AS DECIMAL(30,2))), approx_count_distinct(CAST(id AS DECIMAL(10,2))) FROM range(50000)",
        &[&["49115", "49152", "51286"]],
    ),
    (
        "SELECT approx_count_distinct(id % 7), approx_count_distinct(id % 1000) FROM range(100000)",
        &[&["7", "1026"]],
    ),
    (
        "SELECT percentile_approx(CAST(id AS TINYINT), 0.5) FROM range(100)",
        &[&["49"]],
    ),
    (
        "SELECT percentile_approx(CAST(id AS SMALLINT), 0.5), percentile_approx(CAST(id AS FLOAT), 0.5) FROM range(1000)",
        &[&["499", "499"]],
    ),
    (
        "SELECT percentile(DISTINCT v, 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT percentile_approx(DISTINCT v, 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["5"]],
    ),
    (
        "SELECT approx_count_distinct(DISTINCT v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        &[&["9"]],
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
    (
        "SELECT approx_count_distinct(v, CAST(0.05 AS DOUBLE)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "_LEGACY_ERROR_TEMP_1104",
    ),
    (
        "SELECT percentile(v, CAST('x' AS DOUBLE)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "CAST_INVALID_INPUT",
    ),
    (
        "SELECT percentile(v, array('x')) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "SELECT percentile_approx(v, array(0.5, -1.0)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile_approx(v, 0.5, 2147483648) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
    ),
    (
        "SELECT percentile_approx(v, 0.5, 0.5) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "SELECT percentile_approx(v, 0.5, NULL) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_NULL",
    ),
    (
        "SELECT percentile(v, 0.5, NULL, 1) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "SELECT percentile_cont(0.5, 0.6) WITHIN GROUP (ORDER BY v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "SELECT percentile_cont(v) WITHIN GROUP (ORDER BY v) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT",
    ),
    (
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY 'a') FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "CAST_INVALID_INPUT",
    ),
    (
        "SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY array(v)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "SELECT median('a') FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "CAST_INVALID_INPUT",
    ),
    (
        "SELECT median(array(v)) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    (
        "SELECT approx_count_distinct(v, 0.05, 1) FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
];

/// Window percentile aggregates are not supported here (Spark evaluates them
/// over the frame): they are refused, not evaluated.
const REFUSED: &[&str] = &[
    "SELECT percentile(v, 0.5) OVER () FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
    "SELECT percentile_approx(v, 0.5) OVER () FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
    "SELECT approx_count_distinct(v) OVER () FROM VALUES (1),(2),(3),(4),(5),(CAST(NULL AS INT)),(7),(8),(9),(10) t(v)",
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

#[test]
fn window_percentiles_are_refused() {
    let s = Session::new().unwrap();
    for sql in REFUSED {
        let err = rows(&s, sql).expect_err(sql);
        assert_eq!(
            err.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{sql}"
        );
    }
}

/// The approximate percentiles over inputs big enough to compress and merge
/// their summaries stay within the rank error `rows / accuracy` (the exact
/// answer depends on how Spark splits the input, so no engine's value is
/// "the" value).
#[test]
fn large_approximate_percentiles_stay_within_their_rank_error() {
    let s = Session::new().unwrap();
    let near = |got: &str, want: f64, slack: f64| {
        let v: f64 = got.parse().unwrap();
        assert!((v - want).abs() <= slack, "{got} vs {want} +- {slack}");
    };
    let got = &rows(
        &s,
        "SELECT percentile_approx(id, 0.5), percentile_approx(id, 0.99, 50) FROM range(300000)",
    )
    .unwrap()[0];
    near(&got[0], 149_999.5, 60.0);
    near(&got[1], 296_999.0, 6_000.0);
    let got = &rows(
        &s,
        "SELECT percentile_approx(id, array(0.1, 0.9), 200) FROM range(300000)",
    )
    .unwrap()[0][0];
    let (lo, hi) = got.trim_matches(['[', ']']).split_once(", ").unwrap();
    near(lo, 30_000.0, 3_000.0);
    near(hi, 270_000.0, 3_000.0);
    let got = &rows(
        &s,
        "SELECT percentile_approx(CAST(id AS DOUBLE), array(0.25, 0.5, 0.75)), \
         percentile_approx(CAST(id AS DECIMAL(12,2)), 0.5) FROM range(100000)",
    )
    .unwrap()[0];
    let parts: Vec<&str> = got[0].trim_matches(['[', ']']).split(", ").collect();
    near(parts[0], 25_000.0, 20.0);
    near(parts[1], 50_000.0, 20.0);
    near(parts[2], 75_000.0, 20.0);
    near(&got[1], 50_000.0, 20.0);
}
