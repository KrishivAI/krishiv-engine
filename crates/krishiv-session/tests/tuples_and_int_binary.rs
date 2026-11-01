// SPDX-License-Identifier: Apache-2.0
//! `(a, b)` is `struct(a, b)` and, without ANSI, an integral casts to its
//! big-endian bytes. Every value below is what a Spark 4.2.0 run returned.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn row(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let b = &batches[0];
    (0..b.num_columns())
        .map(|c| {
            let col = b.column(c);
            if col.is_null(0) {
                "NULL".to_owned()
            } else if let Some(a) = col.as_string_opt::<i32>() {
                a.value(0).to_owned()
            } else if let Some(a) = col.as_string_view_opt() {
                a.value(0).to_owned()
            } else {
                format!("<{}>", col.data_type())
            }
        })
        .collect()
}

#[test]
fn a_parenthesized_list_is_a_struct() {
    let s = Session::new().unwrap();
    let got = row(
        &s,
        "SELECT CAST(count((1, NULL)) AS STRING), CAST((1, 2) = (1, 2) AS STRING), \
         CAST((1, 2) IN ((1, 2), (3, 4)) AS STRING), CAST((1, 'a') AS STRING)",
    );
    assert_eq!(got, ["1", "true", "true", "{1, a}"]);
}

#[test]
fn integrals_cast_to_binary_without_ansi() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    let got = row(
        &s,
        "SELECT hex(CAST(CAST(-123 AS TINYINT) AS BINARY)), hex(CAST(-123S AS BINARY)), \
         hex(CAST(-123 AS BINARY)), hex(CAST(-123L AS BINARY)), \
         hex(CAST(CAST(NULL AS INT) AS BINARY))",
    );
    assert_eq!(got, ["85", "FF85", "FFFFFF85", "FFFFFFFFFFFFFF85", "NULL"]);
}

#[test]
fn a_tuple_of_lambda_variables_names_its_fields_after_them() {
    let s = Session::new().unwrap();
    let got = row(
        &s,
        "SELECT typeof(zip_with(array(1, 2), array('a', 'b'), (x, y) -> (y, x))), \
         typeof(transform(array(1, 2), x -> (x, 1)))",
    );
    assert_eq!(
        got,
        [
            "array<struct<y:string,x:int>>",
            "array<struct<x:int,col2:int>>"
        ]
    );
}
