// SPDX-License-Identifier: Apache-2.0
//! `width_bucket`. Expected values come from the Spark 4.2.0 oracle and
//! `operators.sql`.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::Session;

fn value(s: &Session, sql: &str) -> Option<i64> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches: Vec<RecordBatch> = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let col = batches[0].column(0).as_primitive::<Int64Type>();
    (!col.is_null(0)).then(|| col.value(0))
}

#[test]
fn buckets_follow_sparks_rules() {
    let s = Session::builder().build().unwrap();
    let q = |args: &str| value(&s, &format!("select width_bucket({args})"));
    assert_eq!(q("5.35, 0.024, 10.06, 5"), Some(3));
    assert_eq!(q("'5.35', '0.024', '10.06', '5'"), Some(3));
    // The bucket count is cast to bigint: 2.5 is 2, 0.5 is 0 (NULL).
    assert_eq!(q("5.35, 0.024, 10.06, 2.5"), Some(2));
    assert_eq!(q("5.35, 0.024, 10.06, 0.5"), None);
    assert_eq!(q("5.35, 0.024, 10.06, -5"), None);
    assert_eq!(q("5.35, 0.024, 10.06, 9223372036854775807L"), None);
    assert_eq!(q("null, 0.024, 10.06, 5"), None);
    assert_eq!(q("5.35, 1.0, 1.0, 5"), None);
    // Out of range: below and above the interval, and a descending one.
    assert_eq!(q("-1.0, 0.0, 10.0, 5"), Some(0));
    assert_eq!(q("10.0, 0.0, 10.0, 5"), Some(6));
    assert_eq!(q("5.0, 10.0, 0.0, 5"), Some(3));
    assert_eq!(q("11.0, 10.0, 0.0, 5"), Some(0));
    assert_eq!(q("0.0, 10.0, 0.0, 5"), Some(6));
}
