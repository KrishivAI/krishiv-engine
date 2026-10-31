// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.legacy.javaCharsets` and `spark.sql.legacy.codingErrorAction`
//! reach `encode` and `decode`, for literals the analyzer folds and for
//! columns the workers evaluate. Expected values: Spark 4.2.0's
//! `string-functions` golden and runs of the same statements.
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::cast::AsArray;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

fn first(session: &Session, sql: &str) -> Result<Vec<u8>, KrishivError> {
    let batches = futures::executor::block_on(async { session.sql(sql).await?.collect().await })?;
    let a = batches[0].column(0);
    Ok(if let Some(b) = a.as_binary_opt::<i32>() {
        b.value(0).to_vec()
    } else if let Some(b) = a.as_binary_view_opt() {
        b.value(0).to_vec()
    } else if let Some(t) = a.as_string_opt::<i32>() {
        t.value(0).as_bytes().to_vec()
    } else {
        a.as_string_view().value(0).as_bytes().to_vec()
    })
}

fn run(session: &Session, sql: &str) {
    futures::executor::block_on(async { session.sql(sql).await?.collect().await }).unwrap();
}

/// A statement over a literal (folded) and the same over a column.
fn both(session: &Session, call: &str, arg: &str) -> [Result<Vec<u8>, KrishivError>; 2] {
    [
        first(session, &format!("select {call}({arg})")),
        first(
            session,
            &format!("select {call}(a, b) from values ({arg}) as t(a, b)"),
        ),
    ]
}

#[test]
fn java_charsets_need_the_switch() {
    let s = Session::new().unwrap();
    let arg = "'caf\u{e9}', 'WINDOWS-1252'";
    for r in both(&s, "encode", arg) {
        assert_eq!(
            r.unwrap_err().error_class().name(),
            "INVALID_PARAMETER_VALUE.CHARSET"
        );
    }
    run(&s, "set spark.sql.legacy.javaCharsets=true");
    for r in both(&s, "encode", arg) {
        assert_eq!(r.unwrap(), b"caf\xe9");
    }
    for r in both(&s, "decode", "X'636166E9', 'windows-1252'") {
        assert_eq!(r.unwrap(), "caf\u{e9}".as_bytes());
    }
    run(&s, "reset spark.sql.legacy.javaCharsets");
    assert!(both(&s, "encode", arg).iter().all(Result::is_err));
}

#[test]
fn the_coding_error_action_replaces_instead_of_raising() {
    let s = Session::new().unwrap();
    let unmappable = "'\u{4e2d}', 'US-ASCII'";
    let malformed = "X'E4', 'UTF-8'";
    for r in both(&s, "encode", unmappable) {
        assert_eq!(
            r.unwrap_err().error_class().name(),
            "MALFORMED_CHARACTER_CODING"
        );
    }
    for r in both(&s, "decode", malformed) {
        assert_eq!(
            r.unwrap_err().error_class().name(),
            "MALFORMED_CHARACTER_CODING"
        );
    }
    run(&s, "set spark.sql.legacy.codingErrorAction=true");
    for r in both(&s, "encode", unmappable) {
        assert_eq!(r.unwrap(), b"?");
    }
    for r in both(&s, "decode", malformed) {
        assert_eq!(r.unwrap(), "\u{fffd}".as_bytes());
    }
}
