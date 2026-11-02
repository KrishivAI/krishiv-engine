// SPDX-License-Identifier: Apache-2.0
//! Comparison operators over arrays use Spark's element order: NULL elements
//! are smallest and equal each other, a shorter prefix is smaller, a NULL
//! array gives NULL.
#![cfg(test)]

use arrow_array::Array;
use futures::executor::block_on;
use krishiv_session::Session;

fn one(sql: &str) -> String {
    let s = Session::new().unwrap();
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let col = batches[0].column(0);
    if col.is_null(0) {
        return "null".into();
    }
    arrow_array::cast::AsArray::as_boolean(col)
        .value(0)
        .to_string()
}

#[test]
fn arrays_with_null_elements_compare_equal() {
    assert_eq!(
        one("SELECT array(1, 2, NULL, 4) = array(1, 2, NULL, 4)"),
        "true"
    );
    assert_eq!(one("SELECT array(1, 2, NULL) = array(1, 2, 3)"), "false");
    assert_eq!(one("SELECT array(1, 2) != array(1, 2, 3)"), "true");
}

#[test]
fn arrays_order_by_element_then_length() {
    assert_eq!(one("SELECT array(1, 2) < array(1, 2, 3)"), "true");
    assert_eq!(one("SELECT array(1, 3) < array(1, 2, 3)"), "false");
    assert_eq!(one("SELECT array(NULL, 1) < array(0)"), "true");
    assert_eq!(one("SELECT array(1, 2) >= array(1, 2)"), "true");
    assert_eq!(one("SELECT array(2) > array(1, 9)"), "true");
    assert_eq!(one("SELECT array(2) <= array(1, 9)"), "false");
}

#[test]
fn a_null_array_gives_null() {
    assert_eq!(one("SELECT CAST(NULL AS ARRAY<INT>) = array(1)"), "null");
}

#[test]
fn nullability_differing_below_the_top_level_still_compares() {
    assert_eq!(
        one("SELECT array(array(1)) = array(array(1, NULL))"),
        "false"
    );
    assert_eq!(
        one("SELECT array(array(1, NULL)) = array(array(1, NULL))"),
        "true"
    );
    assert_eq!(
        one("SELECT array(named_struct('a', 1)) = array(named_struct('a', CAST(NULL AS INT)))"),
        "false"
    );
    assert_eq!(
        one("SELECT array(named_struct('a', 1)) = array(named_struct('a', 1))"),
        "true"
    );
}

#[test]
fn arrays_of_untyped_nulls_compare() {
    assert_eq!(one("SELECT array(NULL) = array(NULL)"), "true");
    assert_eq!(one("SELECT array(NULL, NULL) < array(NULL)"), "false");
    assert_eq!(one("SELECT array(NULL) < array(NULL, NULL)"), "true");
}
