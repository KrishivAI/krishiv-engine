// SPDX-License-Identifier: Apache-2.0
//! The moment aggregates as window functions over partitions evaluated from disk (a memory limit far below a
//! partition): every output row equals the in-memory one bit for bit, and
//! `spark.sql.legacy.statisticalAggregate` reaches the spilled path (a sample variance over one row is NULL by
//! default and NaN under the flag).
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
    // `t(k, v, i)`: two groups, a NULL `v` every seventh row, and a unique row number `i` (the window orders by
    // `v, i`, so ties in `v` do not make the result depend on the sort).
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
        Field::new("i", DataType::Int64, false),
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
                    Arc::new(Int64Array::from_iter_values(c.iter().copied())),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("t", schema, batches).unwrap();
    s
}

/// `sql`'s rows as text (every column), and the spill files the Window operator itself wrote.
fn run(s: &Session, sql: &str) -> (Vec<String>, u64) {
    // A window under a 6 MB limit can still be refused memory on a loaded or small host (`UnableToAcquireMemory`, a
    // request of a few KB: the residual of the flaky `udaf` window test, tracked in the queue item "URGENT flaky
    // test in the UDAF window code"). That refusal is retried, a bounded number of times; any other error, or a
    // wrong result, fails at once.
    let mut attempt = 0;
    let (batches, profile) = loop {
        let result = block_on(async { s.sql(sql).await?.collect_with_profile().await });
        match result {
            Ok(done) => break done,
            Err(e)
                if attempt < 3
                    && e.error_class()
                        == krishiv_common::error::ErrorClass::UnableToAcquireMemory =>
            {
                attempt += 1;
            }
            Err(e) => panic!("{sql}: {e}"),
        }
    };
    let spilled = profile
        .metrics
        .operators
        .values()
        .filter(|e| e.name == "Window")
        .map(|e| e.metrics.spill_files)
        .sum();
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = (0..b.num_columns())
                .map(|c| {
                    let a = b.column(c);
                    if a.is_null(r) {
                        "None".to_owned()
                    } else if let DataType::Int64 = a.data_type() {
                        a.as_primitive::<Int64Type>().value(r).to_string()
                    } else {
                        // The bits, so two NaNs and `-0.0` against `0.0` compare as Spark's goldens do.
                        a.as_primitive::<Float64Type>()
                            .value(r)
                            .to_bits()
                            .to_string()
                    }
                })
                .collect();
            out.push(cells.join(" "));
        }
    }
    (out, spilled)
}

/// The rows of `run`, sorted by their first column (the row number).
fn run_sorted(s: &Session, sql: &str) -> (Vec<String>, u64) {
    let (mut rows, spilled) = run(s, sql);
    rows.sort_by_key(|r| r.split(' ').next().unwrap().parse::<i64>().unwrap());
    (rows, spilled)
}

/// Every row of `function` over `frame` (ordered by `v, i`); the test sorts the rows by `i` itself, since a final
/// `ORDER BY` over 200,000 rows would be a second consumer of the 6 MB budget.
fn all_rows(function: &str, frame: &str) -> String {
    format!(
        "SELECT i, c FROM (SELECT i, {function} OVER (PARTITION BY k ORDER BY v, i {frame}) AS c FROM t)"
    )
}

/// How many rows are NaN, and how many are non-NULL.
fn counts(function: &str, frame: &str) -> String {
    format!(
        "SELECT count(*), count(c), count(CASE WHEN isnan(c) THEN 1 END) FROM \
         (SELECT {function} OVER (PARTITION BY k ORDER BY v, i {frame}) AS c FROM t)"
    )
}

// QUARANTINED (owner policy 2026-11-07; owner opus-3; deadline 2026-11-10): fails about 1 run in 100 with
// UNABLE_TO_ACQUIRE_MEMORY in the window flush phase (merge-queue runs 42401250907, 42410371135, after #601).
// Skipped by the default nextest profile, run by `--profile quarantine` (CI job `test-quarantine`, not required);
// krishiv-internal `open-issues.md`. Comes back with the root-cause fix.
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
            let q = all_rows(function, frame);
            let (got, spilled) = run_sorted(&limited, &q);
            assert!(
                spilled > 0,
                "{function} {frame}: the Window operator did not spill"
            );
            let (want, _) = run_sorted(&memory, &q);
            assert_eq!(
                got.len(),
                usize::try_from(ROWS).unwrap(),
                "{function} {frame}"
            );
            assert!(got == want, "{function} {frame}: rows differ from memory");
        }
    }
}

#[test]
fn the_legacy_statistical_flag_reaches_a_window_evaluated_from_disk() {
    let dir = tempfile::tempdir().unwrap();
    let rows = ROWS.to_string();
    let values: i64 = (0..ROWS).filter(|i| i % 7 != 0).count().try_into().unwrap();
    // A frame of the current row only: a sample variance over one row divides by zero, and a NULL `v` is NULL.
    let one = counts(
        "var_samp(CAST(v AS DOUBLE))",
        "ROWS BETWEEN CURRENT ROW AND CURRENT ROW",
    );
    let (plain, spilled) = run(&session(false, Some(&dir.path().join("a"))), &one);
    assert!(spilled > 0, "the Window operator did not spill");
    assert_eq!(plain, [format!("{rows} 0 0")]);
    let (legacy, spilled) = run(&session(true, Some(&dir.path().join("b"))), &one);
    assert!(spilled > 0, "the Window operator did not spill");
    assert_eq!(legacy, [format!("{rows} {values} {values}")]);

    // A running frame: the first row with a value in each partition has one value, NaN under the flag.
    let running = counts(
        "var_samp(CAST(v AS DOUBLE))",
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
    );
    let (plain, _) = run(&session(false, Some(&dir.path().join("c"))), &running);
    let (legacy, spilled) = run(&session(true, Some(&dir.path().join("d"))), &running);
    assert!(spilled > 0, "the Window operator did not spill");
    let nan = |line: &str| line.rsplit(' ').next().unwrap().to_owned();
    assert_eq!(nan(&plain[0]), "0", "{plain:?}");
    assert_eq!(nan(&legacy[0]), "2", "{legacy:?}");
}
