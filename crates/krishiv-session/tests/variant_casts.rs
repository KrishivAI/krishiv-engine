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

fn error_class(sql: &str) -> String {
    let s = Session::new().unwrap();
    let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
    e.error_class().name().to_owned()
}

/// Spark 4.2.0 (oracle), third review round: intervals beside a variant print
/// as intervals, a time of day out of a variant is on the query's date, no
/// map key holds a variant.
#[test]
fn third_review_cases_match_spark() {
    assert_eq!(
        one("CAST(named_struct('v', parse_json('1'), 'i', INTERVAL '1' YEAR) AS STRING)"),
        "{1, INTERVAL '1' YEAR}"
    );
    let today = "timestamp(concat(current_date(), ' 10:11:12'))";
    assert_eq!(
        one(&format!(
            "CAST(parse_json('\"10:11:12\"') AS TIMESTAMP) = {today}"
        )),
        "true"
    );
    assert_eq!(
        one(&format!(
            "try_cast(parse_json('\"10:11:12\"') AS TIMESTAMP) = {today}"
        )),
        "true"
    );
    assert_eq!(
        one(&format!(
            "variant_get(parse_json('\"10:11:12\"'), '$', 'timestamp') = {today}"
        )),
        "true"
    );
    for key in [
        "parse_json('1')",
        "named_struct('v', parse_json('1'))",
        "array(parse_json('1'))",
    ] {
        assert_eq!(
            error_class(&format!("SELECT map({key}, 'a')")),
            "DATATYPE_MISMATCH.INVALID_MAP_KEY_TYPE",
            "{key}"
        );
    }
    // Spark reads these; the JSON reader has no variant leaves yet.
    assert_eq!(
        error_class("SELECT from_json('{\"v\":{\"a\":1}}', 'v variant')"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}
