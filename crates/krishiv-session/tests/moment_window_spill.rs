// SPDX-License-Identifier: Apache-2.0
//! The moment aggregates as window functions over partitions evaluated from disk (a memory limit far below a
//! partition): the result equals the in-memory one, and `spark.sql.legacy.statisticalAggregate` reaches the
//! spilled path (a one-row frame is NULL by default and NaN under the flag).
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// Two partitions of 100,000 rows each: neither fits in a 6 MB budget next to its sorted copy.
const ROWS: i64 = 200_000;

fn session(legacy: bool, limited: Option<&std::path::Path>) -> Session {
    let mut b = Session::builder().config(
        "spark.sql.legacy.statisticalAggregate",
        if legacy { "true" } else { "false" },
    );
    if let Some(dir) = limited {
        b = b
            .config("krishiv.spill.dir", dir.to_str().unwrap())
            .config("krishiv.memory.limit", "6m");
    }
    let s = b.build().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]));
    let rows: Vec<i64> = (0..ROWS).collect();
    let batches = rows
        .chunks(4096)
        .map(|c| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % 2))),
                    Arc::new(Int64Array::from_iter(
                        c.iter().map(|i| (i % 7 != 0).then_some(i % 1000)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("t", schema, batches).unwrap();
    s
}

/// The single row of `sql` as text, plus how many spill files the query wrote.
fn run(s: &Session, sql: &str) -> (String, u64) {
    let df = block_on(s.sql(sql)).unwrap();
    let (batches, profile) = block_on(df.collect_with_profile()).unwrap();
    let spilled = profile
        .metrics
        .operators
        .values()
        .map(|e| e.metrics.spill_files)
        .sum();
    let b = &batches[0];
    let cells: Vec<String> = (0..b.num_columns())
        .map(|c| {
            let a = b.column(c);
            if a.is_null(0) {
                "None".to_owned()
            } else if let DataType::Int64 = a.data_type() {
                a.as_primitive::<Int64Type>().value(0).to_string()
            } else {
                format!("{:?}", a.as_primitive::<Float64Type>().value(0))
            }
        })
        .collect();
    (cells.join(" "), spilled)
}

fn query(function: &str, frame: &str) -> String {
    format!(
        "SELECT count(*), count(c), count(CASE WHEN isnan(c) THEN 1 END), max(c), min(c) FROM \
         (SELECT {function} OVER (PARTITION BY k ORDER BY v {frame}) AS c FROM t)"
    )
}

#[test]
fn moment_windows_from_disk_equal_the_in_memory_result() {
    let dir = tempfile::tempdir().unwrap();
    let limited = session(false, Some(&dir.path().join("spill")));
    let memory = session(false, None);
    for function in [
        "var_samp(CAST(v AS DOUBLE))",
        "skewness(CAST(v AS DOUBLE))",
        // (`corr` over a frame of one distinct x raises DIVIDE_BY_ZERO under ANSI, in memory and from disk alike.)
        "covar_samp(CAST(v AS DOUBLE), CAST(k AS DOUBLE) + CAST(v AS DOUBLE))",
    ] {
        for frame in [
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            "ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING",
        ] {
            let q = query(function, frame);
            let (got, spilled) = run(&limited, &q);
            assert!(spilled > 0, "{function} {frame}: the window did not spill");
            assert_eq!(got, run(&memory, &q).0, "{function} {frame}");
        }
    }
}

#[test]
fn the_legacy_statistical_flag_reaches_a_window_evaluated_from_disk() {
    let dir = tempfile::tempdir().unwrap();
    // A frame of the current row only: a sample variance over one row divides by zero.
    let q = query(
        "var_samp(CAST(v AS DOUBLE))",
        "ROWS BETWEEN CURRENT ROW AND CURRENT ROW",
    );
    let rows = ROWS.to_string();
    let (plain, spilled) = run(&session(false, Some(&dir.path().join("a"))), &q);
    assert!(spilled > 0, "the window did not spill");
    // NULL (or NaN where `v` is NULL, which `CAST` keeps NULL): no row is NaN without the flag.
    assert!(plain.starts_with(&format!("{rows} 0 0 ")), "{plain}");
    let (legacy, spilled) = run(&session(true, Some(&dir.path().join("b"))), &q);
    assert!(spilled > 0, "the window did not spill");
    // Every row with a value is NaN under the flag.
    let nan_rows: i64 = (0..ROWS).filter(|i| i % 7 != 0).count().try_into().unwrap();
    assert!(
        legacy.starts_with(&format!("{rows} {nan_rows} {nan_rows} ")),
        "{legacy}"
    );
}
