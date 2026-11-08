// SPDX-License-Identifier: Apache-2.0
//! The serialize-through mode (`krishiv.exec.debug.planRoundTrip`, P2-T173,
//! design P2-T171 section 8.1): with it on, every query's physical plan is
//! encoded to `krishiv.plan.v1` and decoded before it runs, and the results
//! are those of the plan as planned. The 22 TPC-H queries run both ways
//! over the SF 0.01 Parquet fixture.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::builder().build().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("x", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values((0..200).map(|i| i % 7))),
            Arc::new(StringArray::from_iter(
                (0..200).map(|i| (i % 5 != 0).then(|| format!("n{}", i % 11))),
            )),
            Arc::new(Float64Array::from_iter(
                (0..200).map(|i| (i % 13 != 0).then_some(f64::from(i) * 0.5)),
            )),
        ],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

/// The rows of `sql`, rendered and sorted (the plans may differ in order
/// only where the query does not fix it).
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |r| {
                b.columns()
                    .iter()
                    .map(|c| format!("{:?}", c.slice(r, 1)))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    out.sort();
    out
}

const QUERIES: &[&str] = &[
    "SELECT k, count(*), sum(x), avg(x) FROM t GROUP BY k",
    "SELECT k, stddev_samp(x), var_pop(x), corr(x, x * x), covar_samp(x, k) FROM t GROUP BY k",
    "SELECT k, max_by(name, x), min_by(name, x), any_value(name, true) FROM t GROUP BY k",
    "SELECT a.k, b.name FROM t a JOIN t b ON a.k = b.k AND a.x < b.x WHERE a.k < 2",
    "SELECT name, rank() OVER (PARTITION BY k ORDER BY x) FROM t",
    "SELECT k FROM t WHERE x > (SELECT avg(x) FROM t) AND name IN (SELECT name FROM t WHERE k = 3)",
    "SELECT k, count(DISTINCT name) FROM t GROUP BY ROLLUP (k)",
    // The partial state of a percentile crosses the exchange as an array: its schema is nested.
    "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x), percentile_disc(0.5) WITHIN GROUP (ORDER BY k) FROM t",
];

#[test]
fn queries_give_the_same_rows_through_the_wire_format() {
    let s = session();
    let planned: Vec<_> = QUERIES.iter().map(|q| rows(&s, q)).collect();
    s.conf()
        .set("krishiv.exec.debug.planRoundTrip", "true")
        .unwrap();
    for (sql, want) in QUERIES.iter().zip(&planned) {
        assert!(!want.is_empty(), "{sql}: no rows");
        assert_eq!(&rows(&s, sql), want, "{sql}");
    }
}

/// The SF 0.01 TPC-H tables of the planner tests, as temporary views.
fn tpch_session() -> Session {
    let s = Session::builder().build().unwrap();
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../krishiv-exec/tests/fixtures/tpch-sf0.01-parquet"
    );
    for t in [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ] {
        let df = s.read().parquet([format!("{dir}/{t}.parquet")]);
        block_on(df.create_or_replace_temp_view(t)).unwrap();
    }
    s
}

#[test]
fn tpch_gives_the_same_rows_through_the_wire_format() {
    let s = tpch_session();
    let queries: Vec<(String, String)> = (1..=22)
        .map(|q| {
            let path = format!(
                "{}/../../bench/queries/tpch/q{q:02}.sql",
                env!("CARGO_MANIFEST_DIR")
            );
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            (format!("q{q:02}"), text)
        })
        .collect();
    let planned: Vec<_> = queries.iter().map(|(_, sql)| rows(&s, sql)).collect();
    s.conf()
        .set("krishiv.exec.debug.planRoundTrip", "true")
        .unwrap();
    for ((name, sql), want) in queries.iter().zip(&planned) {
        assert_eq!(&rows(&s, sql), want, "tpch {name}");
    }
}
