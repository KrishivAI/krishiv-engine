// SPDX-License-Identifier: Apache-2.0
//! `ErrorClass` can be named from the `krishiv` crate (acceptance RS-30).

use krishiv::{Error, ErrorClass, Session};

#[test]
fn an_error_class_is_matched_without_naming_krishiv_common() {
    let session = Session::new().expect("session");
    let err: Error =
        futures::executor::block_on(async { session.sql("select 1/0").await?.collect().await })
            .expect_err("division by zero");
    assert!(
        matches!(err.error_class(), ErrorClass::DivideByZero),
        "{err}"
    );
    assert_eq!(err.error_class().name(), "DIVIDE_BY_ZERO");
}
