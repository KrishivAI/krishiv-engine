// SPDX-License-Identifier: Apache-2.0
//! `concat` of arrays: the elements in order, NULL if any array is NULL
//! (Spark 4.2.0 `higher-order-functions`).
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn text(sql: &str) -> String {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).unwrap();
    let b = block_on(df.collect()).unwrap();
    arrow_array::cast::AsArray::as_string_view(b[0].column(0))
        .iter()
        .map(|v| v.unwrap_or("NULL").to_owned())
        .collect::<Vec<_>>()
        .join("|")
}

#[test]
fn concat_joins_arrays_in_order() {
    assert_eq!(
        text("concat(array(1, 2), array(3), array(4, 5))"),
        "[1, 2, 3, 4, 5]"
    );
    assert_eq!(text("concat(array(1), CAST(array() AS ARRAY<INT>))"), "[1]");
    assert_eq!(text("concat(array(1), CAST(NULL AS ARRAY<INT>))"), "NULL");
}

#[test]
fn concat_works_per_row_and_in_lambdas() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(
        "SELECT CAST(concat(a, b) AS STRING) FROM VALUES (1, array(1), array(2, 3)), \
         (2, array(4), NULL), (3, CAST(array() AS ARRAY<INT>), array(5)) AS t(i, a, b) ORDER BY i",
    ))
    .unwrap();
    let b = block_on(df.collect()).unwrap();
    let got: Vec<_> = arrow_array::cast::AsArray::as_string_view(b[0].column(0))
        .iter()
        .map(|v| v.unwrap_or("NULL").to_owned())
        .collect();
    assert_eq!(got, ["[1, 2, 3]", "NULL", "[5]"]);
    assert_eq!(
        text("transform(array(array(1), array(2)), z -> concat(array(0), z))"),
        "[[0, 1], [0, 2]]"
    );
}
