// SPDX-License-Identifier: Apache-2.0
//! `m.k` and `t.m.k` on a map column are the lookups `m['k']` (Spark's `ExtractValue`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn first_int(s: &Session, sql: &str) -> Option<i32> {
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let col = batches[0].column(0).as_primitive::<Int32Type>();
    (!col.is_null(0)).then(|| col.value(0))
}

#[test]
fn a_dotted_name_on_a_map_looks_the_key_up() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE OR REPLACE TEMPORARY VIEW m AS SELECT map('a', 1, 'b', 2) AS col1"))
        .unwrap();
    assert_eq!(first_int(&s, "SELECT col1.a FROM m"), Some(1));
    assert_eq!(first_int(&s, "SELECT m.col1.b FROM m"), Some(2));
    assert_eq!(first_int(&s, "SELECT col1.zz FROM m"), None);
    assert_eq!(first_int(&s, "SELECT col1['a'] FROM m"), Some(1));
}
