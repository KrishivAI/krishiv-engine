// SPDX-License-Identifier: Apache-2.0
//! Named arguments of the Spark 4.3 VARIANT functions that declare a `functionSignature` (`variant_set`,
//! `try_variant_set`, `variant_strip_nulls`); the expected values are from a Spark 4.3.0 run (review of 4.3-A).

#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn one(sql: &str) -> String {
    let s = Session::new().expect("session");
    let batches = block_on(async { s.sql(sql).await?.collect().await }).expect("query");
    let col = batches[0].column(0);
    arrow_cast::display::array_value_to_string(col, 0).expect("cell")
}

#[test]
fn variant_set_takes_create_if_missing_by_name() {
    assert_eq!(
        one(
            "SELECT to_json(variant_set(parse_json('{\"a\":1}'), '$.b', 9, create_if_missing => false))"
        ),
        r#"{"a":1}"#
    );
    assert_eq!(
        one(
            "SELECT to_json(try_variant_set(parse_json('{\"a\":1}'), '$.b', 9, create_if_missing => false))"
        ),
        r#"{"a":1}"#
    );
    assert_eq!(
        one("SELECT to_json(variant_set(parse_json('{\"a\":1}'), '$.b', 9))"),
        r#"{"a":1,"b":9}"#
    );
}

#[test]
fn variant_strip_nulls_takes_include_arrays_by_name() {
    assert_eq!(
        one("SELECT to_json(variant_strip_nulls(parse_json('[null]'), include_arrays => false))"),
        "[null]"
    );
    assert_eq!(
        one("SELECT to_json(variant_strip_nulls(parse_json('[null]'), include_arrays => true))"),
        "[]"
    );
}
