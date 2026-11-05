// SPDX-License-Identifier: Apache-2.0
//! `COLLATE` and `collate()` through the analyzer (collations unit 2a, ADR 0043), with the
//! names, types and errors of the Spark 4.2.0 `collations-basic` golden.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn run(sql: &str) -> Result<Vec<String>, KrishivError> {
    let s = Session::new()?;
    let df = block_on(s.sql(sql))?;
    let batches = block_on(df.collect())?;
    let mut out = Vec::new();
    for b in &batches {
        let c = &b.columns()[0];
        for i in 0..c.len() {
            out.push(if c.is_null(i) {
                "NULL".to_owned()
            } else {
                c.as_string_view().value(i).to_owned()
            });
        }
    }
    Ok(out)
}

fn one(sql: &str) -> String {
    run(sql).unwrap().remove(0)
}

fn class(sql: &str) -> ErrorClass {
    run(sql).unwrap_err().error_class()
}

#[test]
fn collate_gives_the_collated_type() {
    assert_eq!(
        one("SELECT typeof('a' COLLATE utf8_lcase)"),
        "string collate UTF8_LCASE"
    );
    assert_eq!(
        one("SELECT typeof(collate('a', 'UNICODE_CI'))"),
        "string collate UNICODE_CI"
    );
    // UTF8_BINARY is the plain string type.
    assert_eq!(one("SELECT typeof('a' COLLATE UTF8_BINARY)"), "string");
    // Names are case-insensitive and may be qualified.
    assert_eq!(
        one("SELECT typeof('a' COLLATE SYSTEM.BUILTIN.utf8_lcase)"),
        "string collate UTF8_LCASE"
    );
}

#[test]
fn collate_keeps_the_value() {
    assert_eq!(one("SELECT 'AbC' COLLATE utf8_lcase"), "AbC");
}

#[test]
fn collate_checks_its_input_and_its_name() {
    assert_eq!(
        class("SELECT 1 COLLATE utf8_lcase"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        class("SELECT 'a' COLLATE no_such_collation"),
        ErrorClass::CollationInvalidName
    );
}
