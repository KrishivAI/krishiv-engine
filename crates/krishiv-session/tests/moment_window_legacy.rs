// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.legacy.statisticalAggregate` reaches the moment aggregates run as window functions: a sample moment over
//! one row is NULL by default and NaN under the flag (as the grouped aggregate gives), for a whole-partition window,
//! a running frame and a sliding frame.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use futures::executor::block_on;
use krishiv_session::Session;

/// `Some(is_nan)` per row of the single DOUBLE column of `sql`, `None` for NULL.
fn nan_or_null(legacy: bool, sql: &str) -> Vec<Option<bool>> {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.legacy.statisticalAggregate", &legacy.to_string())
        .unwrap();
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        let a = b.column(0).as_primitive::<Float64Type>();
        out.extend((0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i).is_nan())));
    }
    out
}

#[test]
fn a_one_row_sample_moment_window_follows_the_legacy_flag() {
    for sql in [
        "SELECT var_samp(CAST(id AS DOUBLE)) OVER () FROM range(1)",
        "SELECT stddev_samp(CAST(id AS DOUBLE)) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM range(1)",
        "SELECT var_samp(CAST(id AS DOUBLE)) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM range(1)",
    ] {
        assert_eq!(nan_or_null(false, sql), [None], "{sql}");
        assert_eq!(nan_or_null(true, sql), [Some(true)], "{sql}");
    }
}
