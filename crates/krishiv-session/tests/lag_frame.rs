// SPDX-License-Identifier: Apache-2.0
//! `lag`/`lead` with a window frame fail at analysis (`_LEGACY_ERROR_TEMP_1035`), a named window's frame
//! included, as in Spark 4.2.0.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

#[test]
fn offset_functions_refuse_a_frame_at_analysis() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW nat AS SELECT * FROM VALUES (1), (2) AS t(a)")).unwrap();
    for (sql, function) in [
        (
            "SELECT lag(a) OVER w FROM nat WINDOW w AS (ORDER BY a ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
            "lag",
        ),
        (
            "SELECT lag(a) OVER (ORDER BY a ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM nat",
            "lag",
        ),
        (
            "SELECT lag(a) OVER (ORDER BY a ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) FROM nat",
            "lag",
        ),
        (
            "SELECT lag(a) OVER (ORDER BY a RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM nat",
            "lag",
        ),
        (
            "SELECT lead(a) OVER (ORDER BY a ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM nat",
            "lead",
        ),
    ] {
        // At `sql`, not at collect: analysis raises it.
        let e = block_on(s.sql(sql))
            .err()
            .unwrap_or_else(|| panic!("{sql} analyzed"));
        assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp1035, "{sql}");
        assert!(
            e.to_string().contains(&format!(
                "Cannot specify window frame for {function} function."
            )),
            "{e}"
        );
    }
}

/// The acceptance repro: a table, a partitioned and ordered named window.
#[test]
fn named_window_over_a_table_fails_at_analysis() {
    let s = Session::new().unwrap();
    block_on(s.sql(
        "CREATE TABLE orders (o_orderkey BIGINT, o_custkey BIGINT, o_orderdate DATE) USING parquet",
    ))
    .unwrap();
    let sql = "SELECT lag(o_orderkey) OVER w FROM orders WINDOW w AS (PARTITION BY o_custkey ORDER BY \
               o_orderdate ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)";
    let e = block_on(s.sql(sql))
        .err()
        .unwrap_or_else(|| panic!("analyzed"));
    assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp1035);
}
