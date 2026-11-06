// SPDX-License-Identifier: Apache-2.0
//! `shuffle(array, seed)` against values from a real Spark 4.2.0 run (one partition).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The first row's columns, each cast to a string (`array<..>` renders as `[a,b]`).
fn first(sql: &str) -> Vec<String> {
    let s = Session::new().unwrap();
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let b = &batches[0];
    b.columns()
        .iter()
        .map(|c| {
            if c.is_null(0) {
                "NULL".to_owned()
            } else if let Some(v) = c.as_string_opt::<i32>() {
                v.value(0).to_owned()
            } else {
                c.as_string_view().value(0).to_owned()
            }
        })
        .collect()
}

fn err(sql: &str) -> String {
    let s = Session::new().unwrap();
    match block_on(s.sql(sql)) {
        Err(e) => e.to_string(),
        Ok(df) => block_on(df.collect()).unwrap_err().to_string(),
    }
}

#[test]
fn seeded_shuffles_match_spark() {
    let r = first(
        "SELECT cast(shuffle(array(1,2,3,4,5,6,7,8,9,10), 0) AS STRING), \
                cast(shuffle(array(1,2,3,4,5,6,7,8,9,10), 42) AS STRING), \
                cast(shuffle(array(1,2,3,4,5,6,7,8,9,10), -7) AS STRING)",
    );
    assert_eq!(r[0], "[4, 1, 2, 9, 7, 10, 5, 3, 6, 8]");
    assert_eq!(r[1], "[6, 8, 2, 1, 4, 7, 9, 3, 5, 10]");
    assert_eq!(r[2], "[5, 2, 1, 8, 10, 4, 3, 7, 9, 6]");
}

#[test]
fn nulls_empty_and_other_types() {
    let r = first(
        "SELECT cast(shuffle(array(1,null,3,null), 5) AS STRING), \
                cast(shuffle(array('a','b','c','d'), 9) AS STRING), \
                cast(shuffle(cast(null AS array<int>), 1) AS STRING), \
                cast(shuffle(array(1), 3) AS STRING), \
                cast(shuffle(array(array(1,2),array(3),null,array(4,5,6)), 11) AS STRING)",
    );
    assert_eq!(r[0], "[3, 1, null, null]");
    assert_eq!(r[1], "[c, b, d, a]");
    assert_eq!(r[2], "NULL");
    assert_eq!(r[3], "[1]");
    assert_eq!(r[4], "[[3], [4, 5, 6], [1, 2], null]");
}

#[test]
fn seed_must_be_a_literal() {
    for sql in [
        "SELECT shuffle(array(1,2,3), 1.5)",
        "SELECT shuffle(array(1,2,3), '5')",
        "SELECT shuffle(array(1,2,3), cast(5 AS bigint))",
        "SELECT shuffle(array(1,2,3), id) FROM range(1)",
    ] {
        assert!(err(sql).contains("INVALID_PARAMETER_VALUE"), "{sql}");
    }
    assert!(err("SELECT shuffle(1)").contains("UNEXPECTED_INPUT_TYPE"));
}

#[test]
fn an_untyped_null_array_is_a_null_array() {
    let r = first(
        "SELECT cast(shuffle(null) AS STRING), cast(shuffle(null, 42) AS STRING), \
                typeof(shuffle(null, 42))",
    );
    assert_eq!(r, ["NULL", "NULL", "array<void>"]);
}
