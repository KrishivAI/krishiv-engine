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
