// SPDX-License-Identifier: Apache-2.0
//! `count(DISTINCT *)` counts distinct rows of the input columns (Spark
//! expands the star); plain `count(*)` stays `count(1)`.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn one(sql: &str) -> i64 {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(sql)).unwrap();
    let b = block_on(df.collect()).unwrap();
    b[0].columns()[0].as_primitive::<Int64Type>().value(0)
}

const T: &str = "FROM VALUES (1, 1), (1, 1), (1, 2), (NULL, 2), (1, NULL) AS t(a, b)";

#[test]
fn distinct_star_counts_distinct_complete_rows() {
    assert_eq!(one(&format!("SELECT count(DISTINCT *) {T}")), 2);
    assert_eq!(one(&format!("SELECT count(DISTINCT a, b) {T}")), 2);
    assert_eq!(one(&format!("SELECT count(*) {T}")), 5);
}
