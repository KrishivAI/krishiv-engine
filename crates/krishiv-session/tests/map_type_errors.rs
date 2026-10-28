// SPDX-License-Identifier: Apache-2.0
//! Type errors the map constructors and `map_concat` raise at analysis time.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn class_of(sql: &str) -> ErrorClass {
    let s = Session::new().unwrap();
    match block_on(s.sql(sql)) {
        Ok(_) => panic!("{sql} resolved"),
        Err(e) => e.error_class(),
    }
}

#[test]
fn map_keys_without_a_common_type_are_rejected() {
    assert_eq!(
        class_of("SELECT map(true, 'a', DATE '2020-01-01', 'b')"),
        ErrorClass::DatatypeMismatchCreateMapKeyDiffTypes
    );
}

#[test]
fn map_values_without_a_common_type_are_rejected() {
    assert_eq!(
        class_of("SELECT map('a', 1, 'b', map(1, 2))"),
        ErrorClass::DatatypeMismatchCreateMapValueDiffTypes
    );
}

#[test]
fn map_concat_of_a_non_map_argument_is_rejected() {
    assert_eq!(
        class_of("SELECT map_concat(map(1, 'a'), 1)"),
        ErrorClass::DatatypeMismatchMapConcatDiffTypes
    );
}
