// SPDX-License-Identifier: Apache-2.0
//! `concat` and `||` of binaries return binary, or a string with
//! `spark.sql.function.concatBinaryAsString`; a mix with strings is a string.
//! Expected values: Spark 4.2.0's `typeCoercion/native/concat` golden.
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::cast::AsArray;
use krishiv_session::Session;

/// The result as one batch.
fn run(session: &Session, sql: &str) -> Vec<arrow_array::RecordBatch> {
    let b = futures::executor::block_on(async { session.sql(sql).await?.collect().await }).unwrap();
    match b.first() {
        Some(f) => vec![arrow_select::concat::concat_batches(&f.schema(), &b).unwrap()],
        None => b,
    }
}

const COLUMNS: &str = "select c1 || c2 || c3 from (select encode(string(id), 'utf-8') c1, \
     encode(string(id + 1), 'utf-8') c2, encode(string(id + 2), 'utf-8') c3 from range(3))";

#[test]
fn binaries_concatenate_to_binary_over_columns_and_literals() {
    let s = Session::new().unwrap();
    let b = run(&s, COLUMNS);
    let a = b[0].column(0);
    assert!(a.as_binary_view_opt().is_some(), "{:?}", a.data_type());
    let v = a.as_binary_view();
    assert_eq!(v.value(0), b"012");
    assert_eq!(v.value(2), b"234");
    let l = run(
        &s,
        "select concat(encode('a', 'utf-8'), encode('b', 'utf-8'), cast(null as binary))",
    );
    assert!(l[0].column(0).is_null(0));
    let l = run(
        &s,
        "select concat(encode('a', 'utf-8'), encode('b', 'utf-8'))",
    );
    assert_eq!(l[0].column(0).as_binary_view().value(0), b"ab");
}

#[test]
fn the_switch_turns_the_result_into_a_string() {
    let s = Session::new().unwrap();
    run(&s, "set spark.sql.function.concatBinaryAsString=true");
    let b = run(&s, COLUMNS);
    let a = b[0].column(0);
    assert_eq!(a.as_string_view().value(1), "123");
}

#[test]
fn a_binary_with_a_string_is_a_string() {
    let s = Session::new().unwrap();
    let b = run(&s, "select concat(encode('a', 'utf-8'), 'b')");
    assert_eq!(b[0].column(0).as_string_view().value(0), "ab");
}
