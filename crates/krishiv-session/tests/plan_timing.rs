// SPDX-License-Identifier: Apache-2.0
//! Planning cost of the 22 TPC-H queries over the SF1 files in `KRISHIV_TPCH_SF1` (P3-T073's "planning is
//! unchanged" A/B): each query is analyzed, optimized and physically planned `KRISHIV_PLAN_RUNS` times
//! (default 1000) through `explain`, never executed. Run the same file on two builds and compare the
//! lines: `cargo nextest run -p krishiv-session --test plan_timing --run-ignored only --no-capture`.

use std::time::{Duration, Instant};

use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};

#[test]
#[ignore = "needs TPC-H SF1 files in KRISHIV_TPCH_SF1; prints p50 and p99 planning time per query"]
fn tpch_sf1_plan_only_timing() {
    let dir = std::env::var("KRISHIV_TPCH_SF1")
        .expect("KRISHIV_TPCH_SF1 must name a directory with the TPC-H SF1 parquet files");
    let runs: usize = std::env::var("KRISHIV_PLAN_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let s = Session::new().unwrap();
    // The tables are read once and registered in memory (first `KRISHIV_PLAN_ROWS` rows, default 50,000), so
    // the loop measures analysis, optimization and physical planning, not parquet footer reads; the memory
    // tables carry row counts, so statistics-driven rules and hints still fire.
    let rows: usize = std::env::var("KRISHIV_PLAN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    for t in [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ] {
        // The limit goes below the collect, so only the sample's rows are read, not the whole SF1 table.
        let n = i32::try_from(rows).unwrap_or(i32::MAX);
        let batches = block_on(
            s.read()
                .parquet([format!("{dir}/{t}.parquet")])
                .limit(n)
                .collect(),
        )
        .unwrap();
        let schema = batches[0].schema();
        let (mut kept, mut left) = (Vec::new(), rows);
        for b in batches {
            if left == 0 {
                break;
            }
            let n = b.num_rows().min(left);
            kept.push(b.slice(0, n));
            left -= n;
        }
        s.register_batches(t, schema, kept).unwrap();
    }
    let (mut sum_p50, mut sum_p99) = (Duration::ZERO, Duration::ZERO);
    for q in 1..=22 {
        let path = format!(
            "{}/../../bench/queries/tpch/q{q:02}.sql",
            env!("CARGO_MANIFEST_DIR")
        );
        let sql = std::fs::read_to_string(path).unwrap();
        let mut times = Vec::with_capacity(runs);
        // One unmeasured run warms the statistics and file-listing caches.
        block_on(async {
            s.sql(&sql)
                .await
                .unwrap()
                .explain(ExplainMode::Simple)
                .await
        })
        .unwrap();
        for _ in 0..runs {
            let started = Instant::now();
            block_on(async {
                s.sql(&sql)
                    .await
                    .unwrap()
                    .explain(ExplainMode::Simple)
                    .await
            })
            .unwrap();
            times.push(started.elapsed());
        }
        times.sort();
        let (p50, p99) = (times[runs / 2], times[(runs * 99 / 100).min(runs - 1)]);
        sum_p50 += p50;
        sum_p99 += p99;
        // The minimum and p10 are what a busy machine disturbs least; compare them across builds first.
        println!(
            "q{q:02}: min {:?} p10 {:?} p50 {p50:?} p99 {p99:?}",
            times[0],
            times[runs / 10]
        );
    }
    println!("sum of p50 {sum_p50:?}, sum of p99 {sum_p99:?} over {runs} runs per query");
}
