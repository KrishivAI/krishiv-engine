// SPDX-License-Identifier: Apache-2.0
//! `width_bucket` over doubles: every row below is what a Spark 4.2.0 run
//! returned for the statement (checked on the oracle, not from memory).
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn column(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    let mut out = Vec::new();
    for batch in block_on(df.collect()).unwrap() {
        let c = batch.column(0);
        let a = c.as_primitive::<Int64Type>();
        for i in 0..a.len() {
            out.push(if a.is_null(i) {
                "NULL".to_owned()
            } else {
                a.value(i).to_string()
            });
        }
    }
    out
}

#[test]
fn buckets_follow_spark() {
    let s = Session::new().unwrap();
    let cases: &[(&str, &str)] = &[
        ("5.35, 0.024, 10.06, 5", "3"),
        ("5.35, 0.024, 10.06, '5'", "3"),
        ("5.35, 0.024, 10.06, 2.5", "2"),
        ("1, 0, 3, 4.7", "2"),
        ("5.35, 0.024, 10.06, 0", "NULL"),
        ("5.35, 0.024, 10.06, 0.5", "NULL"),
        ("5.35, 0.024, 10.06, -5", "NULL"),
        ("5.35, 0.024, 10.06, 9223372036854775807L", "NULL"),
        (
            "5.35, 0.024, 10.06, 9223372036854775807L - 1",
            "4894746858139549697",
        ),
        ("1.0, 1.0, 1.0, 5", "NULL"),
        ("double('NaN'), 0, 10, 5", "NULL"),
        ("1.0, double('NaN'), 10, 5", "NULL"),
        ("1.0, 0, double('Infinity'), 5", "NULL"),
        ("double('Infinity'), 0, 10, 5", "6"),
        ("double('-Infinity'), 0, 10, 5", "0"),
        ("1.0, 0, double('-Infinity'), 5", "NULL"),
        // The double overflows to infinity; Java's `(long)` saturates and the
        // `+ 1` wraps.
        ("1.0e308, 0.0, 1.7e308, 10", "-9223372036854775808"),
        ("1.0e308, 1.7e308, 0.0, 10", "-9223372036854775808"),
        ("1.0e308, -1.7e308, 1.7e308, 9223372036854775806L", "1"),
        ("-1, 0, 10, 5", "0"),
        ("10, 0, 10, 5", "6"),
        ("5, 10, 0, 5", "3"),
        ("11, 10, 0, 5", "0"),
        ("0, 10, 0, 5", "6"),
        ("null, 0.024, 10.06, 5", "NULL"),
        ("5.35, null, 10.06, 5", "NULL"),
        ("5.35, 0.024, null, -5", "NULL"),
        ("5.35, 0.024, 10.06, null", "NULL"),
    ];
    for (args, want) in cases {
        let got = column(&s, &format!("SELECT width_bucket({args})"));
        assert_eq!(got, [*want], "width_bucket({args})");
    }
}

#[test]
fn buckets_over_columns() {
    let s = Session::new().unwrap();
    let got = column(
        &s,
        "SELECT width_bucket(v, 0, 10, 5) FROM VALUES (-1.0), (0.0), (3.5), (9.99), (10.0), (CAST(NULL AS DOUBLE)) t(v)",
    );
    assert_eq!(got, ["0", "1", "2", "5", "6", "NULL"]);
}

#[test]
fn a_string_that_is_no_number_is_a_cast_error() {
    let s = Session::new().unwrap();
    let e = block_on(s.sql("SELECT width_bucket('a', 0, 10, 5)"))
        .and_then(|df| block_on(df.collect()))
        .unwrap_err();
    assert!(e.to_string().contains("CAST_INVALID_INPUT"), "{e}");
}

#[test]
fn legacy_mode_and_arity() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    assert_eq!(column(&s, "SELECT width_bucket('a', 0, 10, 5)"), ["NULL"]);
    assert_eq!(
        column(&s, "SELECT width_bucket(5.35, 0.024, 10.06, 5)"),
        ["3"]
    );
    let e = block_on(s.sql("SELECT width_bucket(1, 0, 3)")).unwrap_err();
    assert!(e.to_string().contains("WRONG_NUM_ARGS"), "{e}");
}
