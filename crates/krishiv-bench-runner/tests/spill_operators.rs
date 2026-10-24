// SPDX-License-Identifier: Apache-2.0
//! The memory gate's spill coverage on TPC-H-shaped data: under a limit far below their state,
//! TPC-H Q18's per-order aggregation spills in the hash aggregate and a full `ORDER BY` over
//! `lineitem` spills in the sort, and both return exactly what an unlimited session returns.
//! (At SF10 the gate runs the same shapes on the TPC-H data: `q18` and `sort_lineitem`, see
//! `bench/README.md`.)
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_session::Session;

const ROWS: i64 = 600_000;

/// `lineitem(l_orderkey, l_linenumber, l_quantity, l_extendedprice)`: about four lines per
/// order, deterministic.
fn lineitem() -> RecordBatch {
    let mix = |i: i64| i.wrapping_mul(0x5851_f42d_4c95_7f2d).rotate_left(23);
    let order: Vec<i64> = (0..ROWS).map(|i| i / 4).collect();
    let line: Vec<i64> = (0..ROWS).map(|i| i % 4 + 1).collect();
    let quantity: Vec<i64> = (0..ROWS).map(|i| mix(i).rem_euclid(50) + 1).collect();
    let price: Vec<i64> = (0..ROWS)
        .map(|i| (mix(i) >> 8).rem_euclid(10_000_000))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("l_linenumber", DataType::Int64, false),
            Field::new("l_quantity", DataType::Int64, false),
            Field::new("l_extendedprice", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(order)),
            Arc::new(Int64Array::from(line)),
            Arc::new(Int64Array::from(quantity)),
            Arc::new(Int64Array::from(price)),
        ],
    )
    .unwrap()
}

/// `(query, the operator that must spill)`.
const QUERIES: [(&str, &str); 2] = [
    // TPC-H Q18's aggregation: one group per order, kept above a quantity threshold.
    (
        "SELECT count(*), sum(q), max(q), sum(l_orderkey) FROM \
         (SELECT l_orderkey, sum(l_quantity) AS q FROM lineitem GROUP BY l_orderkey) \
         WHERE q > 120",
        "HashAggregate",
    ),
    // A full sort: the limit is above the top-N operator's bound, so every row is sorted.
    (
        "SELECT l_orderkey, l_linenumber, l_extendedprice FROM lineitem \
         ORDER BY l_extendedprice DESC, l_orderkey, l_linenumber LIMIT 100000",
        "Sort",
    ),
];

fn session(limit: &str, data: &RecordBatch) -> Session {
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
    s.register_batches("lineitem", data.schema(), vec![data.clone()])
        .unwrap();
    s
}

/// Every result value of `sql` row by row, and the spill files of each operator named `op`.
fn run(session: &Session, sql: &str, op: &str) -> (Vec<Vec<i64>>, u64) {
    let df = futures::executor::block_on(session.sql(sql)).unwrap();
    let (batches, metrics) = futures::executor::block_on(df.collect_with_metrics()).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let cols: Vec<_> = b
            .columns()
            .iter()
            .map(AsArray::as_primitive::<Int64Type>)
            .collect();
        rows.extend((0..b.num_rows()).map(|r| cols.iter().map(|c| c.value(r)).collect()));
    }
    let spilled = metrics
        .operators
        .values()
        .filter(|e| e.name == op)
        .map(|e| e.metrics.spill_files)
        .sum();
    (rows, spilled)
}

#[test]
fn hash_aggregate_and_sort_spill_and_return_the_unspilled_result() {
    let data = lineitem();
    let unlimited = session("1g", &data);
    let tight = session("8m", &data);
    for (sql, op) in QUERIES {
        let (expected, none) = run(&unlimited, sql, op);
        assert_eq!(none, 0, "{op} spilled with 1 GiB");
        assert!(!expected.is_empty() && expected[0][0] > 0, "{sql}");
        let (got, spilled) = run(&tight, sql, op);
        assert!(spilled > 0, "{op} must spill under 8 MiB");
        assert_eq!(got, expected, "{sql}");
    }
}
