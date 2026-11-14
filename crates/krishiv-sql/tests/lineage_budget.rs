// SPDX-License-Identifier: Apache-2.0
//! The lineage budget (design `P3-T063-analyze-api.md` §3.5 n7: the pass costs at most 10 % of analysis):
//! the TPC-DS queries analyzed, then their lineage computed, each timed. A measurement, not a gate: run
//! `cargo nextest run -p krishiv-sql --test lineage_budget --run-ignored only --no-capture` and read the
//! printed ratio (timings on a shared machine are too noisy to assert).
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use krishiv_plan::SourceId;
use krishiv_plan::analyzer::lineage::lineage;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_sql::{ParserOptions, parse_and_lower};
use serde_json::Value;

/// The repository root.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The TPC-DS tables (the Spark 4.2.0 golden schemas), `CHAR`/`VARCHAR` read as `STRING`.
fn catalog() -> MemoryCatalog {
    let path = root().join("tests/fixtures/spark-4.2.0/golden-tables/tpcds-schemas.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut catalog = MemoryCatalog::new("spark_catalog");
    for table in doc["tables"].as_array().unwrap() {
        let columns: Vec<TableColumn> = table["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let ty = c["type"].as_str().unwrap();
                let upper = ty.to_ascii_uppercase();
                let ty = if upper.starts_with("CHAR") || upper.starts_with("VARCHAR") {
                    "string"
                } else {
                    ty
                };
                TableColumn {
                    name: c["name"].as_str().unwrap().to_owned(),
                    data_type: krishiv_types::parse_data_type(ty).unwrap(),
                    nullable: c["nullable"].as_bool().unwrap(),
                    char_varchar: None,
                }
            })
            .collect();
        catalog = catalog.with_table(&["default"], table["name"].as_str().unwrap(), columns, None);
    }
    catalog
}

/// The fastest of `n` runs of `f`.
fn best(n: usize, mut f: impl FnMut()) -> Duration {
    (0..n)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .min()
        .unwrap()
}

#[test]
#[ignore = "a measurement: run it alone and read the ratio"]
fn lineage_costs_little_of_analysis_on_tpcds() {
    let ctx = context(catalog(), ResolutionConf::default());
    let (mut analysis, mut walk, mut ratios) = (Duration::ZERO, Duration::ZERO, Vec::new());
    for q in 1..=99 {
        let path = root().join(format!("bench/queries/tpcds/q{q:02}.sql"));
        let text = std::fs::read_to_string(path).unwrap();
        let Ok(plan) = parse_and_lower(&text, &ParserOptions::default(), SourceId(0)) else {
            continue;
        };
        let Ok(analyzed) = block_on(DefaultAnalyzer.analyze(&plan, &ctx)) else {
            continue;
        };
        // Every analyzed query has a lineage.
        lineage(&analyzed).unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        let a = best(5, || {
            let _ = block_on(DefaultAnalyzer.analyze(&plan, &ctx)).unwrap();
        });
        let l = best(5, || {
            let _ = lineage(&analyzed).unwrap();
        });
        analysis += a;
        walk += l;
        ratios.push(l.as_secs_f64() / a.as_secs_f64());
    }
    ratios.sort_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];
    let p90 = ratios[ratios.len() * 9 / 10];
    let max = ratios[ratios.len() - 1];
    eprintln!(
        "lineage budget: {} queries; analysis {analysis:?}, lineage {walk:?}, total ratio {:.2} %, \
         per-query median {:.2} %, p90 {:.2} %, max {:.2} %",
        ratios.len(),
        100.0 * walk.as_secs_f64() / analysis.as_secs_f64(),
        100.0 * median,
        100.0 * p90,
        100.0 * max,
    );
    assert!(ratios.len() >= 90, "{} queries analyzed", ratios.len());
}
