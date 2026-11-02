// SPDX-License-Identifier: Apache-2.0
//! `DATE`/`TIMESTAMP` literals with Spark's special values (`epoch`, `now`,
//! `today`, `tomorrow`, `yesterday`) read the query clock.
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn truth(s: &Session, sql: &str) -> bool {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    batches[0].column(0).as_boolean().value(0)
}

#[test]
fn the_special_values_equal_the_clock_expressions() {
    let s = Session::new().unwrap();
    for sql in [
        "select date 'now' = current_date()",
        "select date 'today' = current_date()",
        "select date 'tomorrow' = date_add(current_date(), 1)",
        "select date 'yesterday' = date_sub(current_date(), 1)",
        "select timestamp 'now' = current_timestamp()",
        "select timestamp 'today' = cast(current_date() as timestamp)",
        "select timestamp 'tomorrow' = cast(date_add(current_date(), 1) as timestamp)",
        "select timestamp 'yesterday' = cast(date_sub(current_date(), 1) as timestamp)",
        "select timestamp_ntz 'now' = localtimestamp()",
        "select timestamp_ntz 'today' = cast(current_date() as timestamp_ntz)",
        " select TIMESTAMP ' Today ' = cast(current_date() as timestamp)",
    ] {
        assert!(truth(&s, sql), "{sql}");
    }
}

#[test]
fn epoch_is_a_constant() {
    let s = Session::new().unwrap();
    for sql in [
        "select date 'epoch' = date '1970-01-01'",
        "select timestamp 'epoch' = timestamp '1970-01-01 00:00:00+00:00'",
        "select timestamp_ntz 'epoch' = timestamp_ntz '1970-01-01 00:00:00'",
    ] {
        assert!(truth(&s, sql), "{sql}");
    }
}

#[test]
fn under_ntz_a_plain_timestamp_special_is_local() {
    let s = Session::builder()
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .build()
        .unwrap();
    assert!(truth(&s, "select timestamp 'now' = localtimestamp()"));
}
