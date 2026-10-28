// SPDX-License-Identifier: Apache-2.0
//! Spilling never changes a result: random tables, joined, grouped and sorted under a memory limit
//! small enough to spill, return exactly what an unlimited session returns.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_session::Session;
use proptest::prelude::*;

/// `a(k, v, s)` and `b(k, w)` from a seed: `rows` rows, keys below `keys`.
fn tables(seed: u64, rows: usize, keys: i64) -> [(&'static str, RecordBatch); 2] {
    let mut x = seed | 1;
    let mut next = move || {
        // xorshift64*: deterministic, and enough spread for keys and values.
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut col = |n: usize, m: i64| -> Vec<i64> {
        (0..n)
            .map(|_| i64::try_from(next() >> 2).unwrap() % m)
            .collect()
    };
    let (ak, av, bk, bw) = (
        col(rows, keys),
        col(rows, 1_000_000),
        col(rows, keys),
        col(rows, 1_000),
    );
    let s: Vec<String> = av.iter().map(|v| format!("s{:08}", v % 50_000)).collect();
    let a = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from(ak)),
            Arc::new(Int64Array::from(av)),
            Arc::new(StringArray::from(s)),
        ],
    )
    .unwrap();
    let b = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("w", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(bk)),
            Arc::new(Int64Array::from(bw)),
        ],
    )
    .unwrap();
    [("a", a), ("b", b)]
}

/// Join, aggregation and sort, each returning only BIGINT columns.
const QUERIES: [&str; 3] = [
    "SELECT count(*), sum(a.v * 7 + b.w), min(a.v - b.w) FROM a JOIN b ON a.k = b.k",
    "SELECT count(*), sum(c), sum(m), max(c) FROM (SELECT s, count(*) AS c, max(k) AS m FROM a GROUP BY s)",
    "SELECT k, v FROM a ORDER BY v DESC, k",
];

/// Every result value of `sql`, row by row, and the spill files the query wrote.
fn run(session: &Session, sql: &str) -> (Vec<Vec<Option<i64>>>, u64) {
    let df = futures::executor::block_on(session.sql(sql)).unwrap();
    let (batches, metrics) = futures::executor::block_on(df.collect_with_metrics()).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let cols: Vec<_> = b
            .columns()
            .iter()
            .map(AsArray::as_primitive::<Int64Type>)
            .collect();
        for r in 0..b.num_rows() {
            rows.push(
                cols.iter()
                    .map(|c| arrow_array::Array::is_valid(c, r).then(|| c.value(r)))
                    .collect(),
            );
        }
    }
    let spilled = metrics
        .operators
        .values()
        .map(|e| e.metrics.spill_files)
        .sum();
    (rows, spilled)
}

fn session(limit: &str, data: &[(&str, RecordBatch)]) -> Session {
    // Two drivers: each holds a share of the limit, so a small limit stays workable.
    let executor = krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig {
        compute_threads: 2,
        io_threads: 1,
        blocking_threads: 2,
        io_blocking_threads: 2,
    })
    .unwrap();
    let s = Session::builder()
        .executor(executor)
        .config("krishiv.memory.limit", limit)
        .build()
        .unwrap();
    for (name, batch) in data {
        s.register_batches(name, batch.schema(), vec![batch.clone()])
            .unwrap();
    }
    s
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 4, ..ProptestConfig::default() })]

    #[test]
    fn spilled_results_equal_in_memory_results(
        seed in any::<u64>(),
        rows in 100_000usize..200_000,
        // Join keys per row: about 2, 1 or 0.1 matches per probe row.
        per_key in prop_oneof![Just(2usize), Just(1), Just(0)],
    ) {
        let keys = match per_key {
            0 => rows * 10,
            n => rows / n,
        };
        let data = tables(seed, rows, i64::try_from(keys).unwrap());
        let unlimited = session("1g", &data);
        let tight = session("4m", &data);
        let mut spilled_any = 0;
        for sql in QUERIES {
            let (expected, none) = run(&unlimited, sql);
            prop_assert_eq!(none, 0, "{} spilled with 1 GiB", sql);
            let (got, spilled) = run(&tight, sql);
            spilled_any += spilled;
            prop_assert_eq!(got, expected, "{}", sql);
        }
        prop_assert!(spilled_any > 0, "4 MiB must make something spill");
    }
}
