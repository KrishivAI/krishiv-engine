// SPDX-License-Identifier: Apache-2.0
//! Mixed `DISTINCT` aggregates leave constant arguments in place (Spark's
//! `RewriteDistinctAggregates` skips foldable children), so a quantile's
//! percentage stays foldable.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use futures::executor::block_on;
use krishiv_session::Session;

const T: &str = "FROM VALUES (1), (2), (2) AS t(a)";

#[test]
fn a_quantile_percentage_survives_the_distinct_rewrite() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!(
        "SELECT count(DISTINCT 0.8), percentile_approx(DISTINCT a, 0.8), count(DISTINCT a) {T}"
    )))
    .unwrap();
    let b = block_on(df.collect()).unwrap();
    let c = b[0].columns();
    assert_eq!(c[0].as_primitive::<Int64Type>().value(0), 1);
    assert_eq!(c[1].as_primitive::<Int32Type>().value(0), 2);
    assert_eq!(c[2].as_primitive::<Int64Type>().value(0), 2);
}
