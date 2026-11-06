// SPDX-License-Identifier: Apache-2.0
//! Casts with a `VARIANT` inside an array, struct or map (Spark 4.2.0: the
//! analyzer accepts them element by element, and so does execution).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn one(sql: &str) -> String {
    let s = Session::new().unwrap();
    let out = block_on(async {
        s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))
            .await?
            .collect()
            .await
    })
    .unwrap();
    out[0].column(0).as_string_view().value(0).to_owned()
}

#[test]
fn nested_variant_casts_run() {
    assert_eq!(
        one("CAST(array(parse_json('1'), parse_json('\"2\"')) AS ARRAY<INT>)"),
        "[1, 2]"
    );
    assert_eq!(one("to_json(CAST(array(1, 2) AS ARRAY<VARIANT>)[1])"), "2");
    assert_eq!(
        one("CAST(named_struct('a', parse_json('1.5')) AS STRUCT<a: DOUBLE>)"),
        "{1.5}"
    );
    assert_eq!(
        one("CAST(map('k', parse_json('true')) AS MAP<STRING, BOOLEAN>)"),
        "{k -> true}"
    );
}

/// Spark 4.2.0 (oracle): containers with variants as strings and as JSON, no
/// ordering of variants, NULL parents' hidden values never cast, a NOT NULL
/// target field of a missing value is NULL.
#[test]
fn review_cases_match_spark() {
    assert_eq!(
        one("CAST(array(parse_json('1'), parse_json('\"x\"')) AS STRING)"),
        "[1, \"x\"]"
    );
    assert_eq!(
        one("CAST(named_struct('a', parse_json('[1]')) AS STRING)"),
        "{[1]}"
    );
    assert_eq!(
        one("to_json(array(parse_json('1'), parse_json('{\"a\":\"b\"}')))"),
        "[1,{\"a\":\"b\"}]"
    );
    assert_eq!(
        one("to_json(named_struct('v', parse_json('2')))"),
        "{\"v\":2}"
    );
    // The NULL row's hidden `x` is never cast to INT.
    assert_eq!(
        one(
            "isnull(CAST(CASE WHEN rand() < 2 THEN NULL ELSE named_struct('a', parse_json('\"x\"')) \
             END AS STRUCT<a: INT>))"
        ),
        "true"
    );
    assert_eq!(
        one("try_variant_get(parse_json('{}'), '$', 'struct<a int not null>')"),
        "{null}"
    );
    let s = Session::new().unwrap();
    let e = block_on(async {
        s.sql("SELECT array_min(array(parse_json('2'), parse_json('1')))")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert!(e.to_string().contains("INVALID_ORDERING_TYPE"), "{e}");
}
