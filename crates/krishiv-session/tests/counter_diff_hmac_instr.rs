// SPDX-License-Identifier: Apache-2.0
//! `counter_diff`, `hmac` and `instr` with a position and an occurrence, through SQL (Spark 4.3 forms).
#![cfg(test)]

use arrow_cast::display::{ArrayFormatter, FormatOptions};
use futures::executor::block_on;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let fmt: Vec<_> = b
            .columns()
            .iter()
            .map(|c| {
                ArrayFormatter::try_new(c.as_ref(), &FormatOptions::default().with_null("NULL"))
                    .unwrap()
            })
            .collect();
        for r in 0..b.num_rows() {
            out.push(
                fmt.iter()
                    .map(|f| f.value(r).to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    out
}

#[test]
fn counter_diff_over_a_partition_window() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "SELECT k, v, counter_diff(v) OVER (PARTITION BY k ORDER BY i) FROM VALUES \
         ('a', 1, 100), ('a', 2, 200), ('a', 3, 50), ('a', 4, NULL), ('a', 5, 80), ('b', 1, 7), ('b', 2, 9) \
         AS t(k, i, v) ORDER BY k, i",
    );
    assert_eq!(
        got,
        [
            "a|100|NULL",
            "a|200|100",
            "a|50|NULL",
            "a|NULL|NULL",
            "a|80|30",
            "b|7|NULL",
            "b|9|2"
        ]
    );
}

#[test]
fn hmac_and_the_default_algorithm() {
    let s = Session::new().unwrap();
    let want = "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8";
    let got = rows(
        &s,
        "SELECT hex(hmac(CAST('key' AS BINARY), CAST('The quick brown fox jumps over the lazy dog' AS BINARY), 'SHA-256')), \
         hex(hmac(CAST('key' AS BINARY), CAST('The quick brown fox jumps over the lazy dog' AS BINARY)))",
    );
    assert_eq!(
        got,
        [format!("{}|{}", want.to_uppercase(), want.to_uppercase())]
    );
}

#[test]
fn instr_with_a_position_and_an_occurrence() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "SELECT instr('abcabcabc', 'abc', 2), instr('abcabcabc', 'abc', 1, 3), instr('abcabcabc', 'abc', 1, 4), \
         instr(NULL, 'a', 1)",
    );
    assert_eq!(got, ["4|7|0|NULL"]);
}

#[test]
fn the_default_arguments_show_in_the_column_names() {
    let s = Session::new().unwrap();
    let batches = block_on(async {
        s.sql(
            "SELECT hmac(CAST('k' AS BINARY), CAST('m' AS BINARY)), instr('abc', 'b', 1), \
             counter_diff(v) OVER (ORDER BY v) FROM VALUES (1), (2) AS t(v)",
        )
        .await?
        .collect()
        .await
    })
    .unwrap();
    let names: Vec<String> = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(
        names,
        [
            "hmac(CAST(k AS BINARY), CAST(m AS BINARY), SHA-256)",
            "instr(abc, b, 1, 1)",
            "counter_diff(v) OVER (ORDER BY v ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
        ]
    );
}
