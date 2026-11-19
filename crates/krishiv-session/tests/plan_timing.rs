// SPDX-License-Identifier: Apache-2.0
//! Planning cost of the 22 TPC-H queries over the SF1 files in `KRISHIV_TPCH_SF1` (P3-T073's "planning is
//! unchanged" A/B): each query is analyzed, optimized and physically planned `KRISHIV_PLAN_RUNS` times
//! (default 1000) through `explain`, never executed. Run the same file on two builds and compare the
//! lines: `cargo nextest run -p krishiv-session --test plan_timing --run-ignored only --no-capture`.
#![cfg(test)]

use std::time::{Duration, Instant};

use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};

#[test]
#[ignore = "needs TPC-H SF1 files in KRISHIV_TPCH_SF1; prints p50 and p99 planning time per query"]
fn tpch_sf1_plan_only_timing() {
    let Ok(dir) = std::env::var("KRISHIV_TPCH_SF1") else {
        return;
    };
    let runs: usize = std::env::var("KRISHIV_PLAN_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let s = Session::new().unwrap();
    for t in [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ] {
        let df = s.read().parquet([format!("{dir}/{t}.parquet")]);
        block_on(df.create_or_replace_temp_view(t)).unwrap();
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
        println!("q{q:02}: p50 {p50:?} p99 {p99:?}");
    }
    println!("sum of p50 {sum_p50:?}, sum of p99 {sum_p99:?} over {runs} runs per query");
}
