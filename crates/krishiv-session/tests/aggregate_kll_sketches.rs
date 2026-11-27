// SPDX-License-Identifier: Apache-2.0
//! `kll_sketch_agg_*` and `kll_merge_agg_*` with the `kll_sketch_*` scalars through SQL (P3 goldens, sketches unit
//! 2): counts and quantiles do not depend on the partition count, and a bad `k` is Spark's analysis error.
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn cell(b: &RecordBatch, c: usize, row: usize) -> String {
    let a = b.column(c);
    if a.is_null(row) {
        return "NULL".into();
    }
    if let Some(s) = a.as_primitive_opt::<Int64Type>() {
        return s.value(row).to_string();
    }
    if let Some(s) = a.as_primitive_opt::<Float64Type>() {
        return s.value(row).to_string();
    }
    if let Some(s) = a.as_string_opt::<i32>() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_string_view_opt() {
        return s.value(row).to_owned();
    }
    format!("<{}>", a.data_type())
}

fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                out.push(
                    (0..b.num_columns())
                        .map(|c| cell(b, c, r))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
        }
        out.sort();
        Ok(out)
    })
}

#[test]
fn counts_and_quantiles_do_not_depend_on_partitions() -> R {
    let s = Session::new()?;
    for n in [1_u32, 8] {
        let from = format!("(SELECT /*+ REPARTITION({n}) */ id FROM range(0, 100))");
        let got = rows(
            &s,
            &format!(
                "SELECT kll_sketch_get_n_bigint(kll_sketch_agg_bigint(id)), \
                 kll_sketch_get_quantile_bigint(kll_sketch_agg_bigint(id), CAST(0.0 AS DOUBLE)), \
                 kll_sketch_get_quantile_bigint(kll_sketch_agg_bigint(id), CAST(1.0 AS DOUBLE)) FROM {from}"
            ),
        )?;
        assert_eq!(got, ["100|0|99"]);
        let merged = rows(
            &s,
            &format!(
                "SELECT kll_sketch_get_n_double(kll_merge_agg_double(sk)) FROM \
                 (SELECT kll_sketch_agg_double(CAST(id AS DOUBLE)) AS sk FROM {from} GROUP BY id % 4)"
            ),
        )?;
        assert_eq!(merged, ["100"]);
    }
    Ok(())
}

#[test]
fn nulls_are_skipped_and_an_empty_group_is_the_empty_sketch() -> R {
    let s = Session::new()?;
    let got = rows(
        &s,
        "SELECT kll_sketch_get_n_bigint(kll_sketch_agg_bigint(CAST(NULL AS BIGINT))), \
         kll_sketch_get_n_float(kll_sketch_agg_float(CAST(x AS FLOAT))) \
         FROM VALUES (1), (NULL), (3) AS t(x)",
    )?;
    assert_eq!(got, ["0|2"]);
    let none = rows(
        &s,
        "SELECT kll_sketch_get_n_bigint(kll_merge_agg_bigint(sk)) FROM \
         (SELECT kll_sketch_agg_bigint(id) AS sk FROM range(0, 5)) WHERE false",
    )?;
    assert_eq!(none, ["0"]);
    Ok(())
}

#[test]
fn a_k_out_of_range_is_an_analysis_error() -> R {
    let s = Session::new()?;
    for k in [7, 65_536] {
        let e = block_on(async {
            s.sql(&format!(
                "SELECT kll_sketch_agg_bigint(id, {k}) FROM range(0, 3)"
            ))
            .await?
            .collect()
            .await
        })
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(e.contains("KLL_SKETCH_K_OUT_OF_RANGE"), "{e}");
    }
    Ok(())
}

#[test]
fn a_bad_buffer_and_a_rank_outside_zero_one_are_runtime_errors() -> R {
    let s = Session::new()?;
    for (call, class) in [
        (
            "kll_sketch_get_n_bigint(X'616263')",
            "KLL_INVALID_INPUT_SKETCH_BUFFER",
        ),
        (
            "kll_sketch_get_quantile_bigint(kll_sketch_agg_bigint(id), CAST(1.5 AS DOUBLE))",
            "KLL_SKETCH_INVALID_QUANTILE_RANGE",
        ),
    ] {
        let e = block_on(async {
            s.sql(&format!("SELECT {call} FROM range(0, 3)"))
                .await?
                .collect()
                .await
        })
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(e.contains(class), "{call}: {e}");
    }
    Ok(())
}

#[test]
fn a_folded_constant_k_lowers_like_its_literal() -> R {
    let s = Session::new()?;
    let got = rows(
        &s,
        "SELECT kll_sketch_get_n_bigint(kll_sketch_agg_bigint(id, 100 + 100)) FROM range(0, 100)",
    )?;
    assert_eq!(got, ["100"]);
    Ok(())
}
