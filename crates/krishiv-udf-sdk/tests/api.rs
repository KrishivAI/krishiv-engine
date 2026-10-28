// SPDX-License-Identifier: Apache-2.0
//! The SDK's public surface prints and converts as Spark does.

use krishiv_types::DataType;
use krishiv_udf_sdk::{
    NullHandling, Signature, UdfError, UdfInputType, UdfLibrary, UdfProperties, guard_call,
};

#[test]
fn input_types_print_as_spark() {
    assert_eq!(UdfInputType::Any.sql(), r#""ANY""#);
    assert_eq!(
        UdfInputType::Numeric { intervals: false }.sql(),
        r#""NUMERIC""#
    );
    assert_eq!(
        UdfInputType::Numeric { intervals: true }.sql(),
        r#""NUMERIC" or "ANSI INTERVAL""#
    );
    assert_eq!(
        UdfInputType::OneOf(vec![DataType::Date, DataType::String]).sql(),
        r#"("DATE" or "STRING")"#
    );
    assert_eq!(
        UdfInputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz]).sql(),
        r#""(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)""#
    );
}

#[test]
fn signatures_print_arity_and_inputs() {
    assert_eq!(
        Signature::Exact(vec![
            UdfInputType::Exact(DataType::Integer),
            UdfInputType::Exact(DataType::String),
        ])
        .arity_text(),
        "2"
    );
    assert_eq!(
        Signature::Coercible(vec![UdfInputType::Exact(DataType::Integer)]).arity_text(),
        "1"
    );
    assert_eq!(
        Signature::Variadic {
            element: UdfInputType::Exact(DataType::String),
            min_arity: 0,
        }
        .arity_text(),
        ""
    );
}

#[test]
fn panics_become_failed_execute_udf_with_their_payload() {
    let via_string = guard_call("f", "invoke", || -> i32 {
        std::panic::panic_any("owned".to_owned())
    })
    .unwrap_err();
    assert!(via_string.message().contains("owned"), "{via_string}");
    let via_other =
        guard_call("f", "invoke", || -> i32 { std::panic::panic_any(42_i32) }).unwrap_err();
    assert!(
        via_other.message().contains("without a message"),
        "{via_other}"
    );
}

#[test]
fn udf_errors_print_and_convert() {
    let e = UdfError::new("boom");
    assert_eq!(format!("{e}"), "boom");
    assert_eq!(UdfError::from("x".to_owned()), UdfError::new("x"));
    assert_eq!(UdfError::from("y"), UdfError::new("y"));
    assert_eq!(UdfProperties::default().null_handling, NullHandling::Strict);
}

#[test]
fn libraries_start_empty_and_debug() {
    let lib = UdfLibrary::new();
    assert!(lib.is_empty());
    assert_eq!(lib.version(), 0);
    assert_eq!(format!("{}", UdfLibrary::default().version()), "0");
    assert_eq!(format!("{lib:?}"), "UdfLibrary { udfs: [], epoch: 0 }");
}
