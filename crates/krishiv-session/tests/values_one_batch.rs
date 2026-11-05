// SPDX-License-Identifier: Apache-2.0
//! A small `VALUES` list is one local relation: its rows reach the operators above in one batch, as in Spark, where
//! a lower projection (a `BETWEEN`'s common subexpression) is computed over every row before the upper one starts.
//! With one batch per row, an error in a later row's `v` surfaced after an earlier row's bound had already
//! raised (the nightly differential, 2026-11-05). The expected answers are those of a real Spark 4.2.0 run of the
//! same statements, ANSI on.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use krishiv_session::Session;

/// `v` overflows `BIGINT` for `k = 2` only.
const V: &str = "CAST(CAST(k AS DECIMAL(20,0)) * 9223372036854775807 AS BIGINT)";

fn answer(sql: &str) -> String {
    let s = Session::new().unwrap();
    match futures::executor::block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(batches) => batches
            .iter()
            .flat_map(|b| {
                let a = b.column(0).as_string_view();
                (0..a.len())
                    .map(|i| {
                        if a.is_null(i) {
                            "NULL".to_owned()
                        } else {
                            a.value(i).to_owned()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .join("|"),
        Err(e) => format!("ERR:{}", e.error_class().name()),
    }
}

#[test]
fn between_computes_its_value_over_every_row_before_the_bounds() {
    // Row 1: the bound `10 / l` divides by zero; row 2: `v` overflows. Spark computes `v` for all rows first.
    let sql = format!(
        "SELECT CAST({V} BETWEEN 10 / l AND 100 AS STRING) FROM (VALUES (1, 0), (2, 1)) AS t(k, l)"
    );
    assert_eq!(answer(&sql), "ERR:CAST_OVERFLOW");
    // Without a failing `v`, the bound's own error is the one raised.
    let sql =
        "SELECT CAST(k BETWEEN 10 / l AND 100 AS STRING) FROM (VALUES (1, 0), (2, 1)) AS t(k, l)";
    assert_eq!(answer(sql), "ERR:DIVIDE_BY_ZERO");
    // And nothing raises where nothing fails.
    let sql = "SELECT CAST(k BETWEEN 10 / l AND 100 AS STRING) FROM (VALUES (1, 1), (2, 5)) AS t(k, l) ORDER BY k";
    assert_eq!(answer(sql), "false|true");
}

#[test]
fn a_small_values_list_arrives_in_one_batch() {
    let s = Session::new().unwrap();
    let batches = futures::executor::block_on(async {
        s.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(a)")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let sizes: Vec<usize> = batches
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .filter(|n| *n > 0)
        .collect();
    assert_eq!(sizes, [3]);
}
