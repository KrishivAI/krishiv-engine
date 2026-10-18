// SPDX-License-Identifier: Apache-2.0
//! Planning latency of TPC-H queries, run under CodSpeed in CI: parse +
//! lower + analyze (warm in-memory catalog), per query. The target is far
//! inside the < 100 ms planning bound and tracks the < 5 ms planning
//! budget (analysis <= 2.5 ms p99).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::path::Path;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_sql::{ParserOptions, parse_and_lower};

#[expect(
    clippy::expect_used,
    reason = "a benchmark without its inputs cannot run"
)]
fn catalog(root: &Path) -> MemoryCatalog {
    let text =
        std::fs::read_to_string(root.join("tests/fixtures/spark-4.2.0/analyzer-goldens.json"))
            .expect("analyzer goldens");
    let goldens: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    let mut c = MemoryCatalog::new("spark_catalog");
    for (name, cols) in goldens["tables"].as_object().expect("tables") {
        let columns = cols
            .as_array()
            .expect("columns")
            .iter()
            .map(|f| TableColumn {
                name: f["name"].as_str().expect("name").to_owned(),
                data_type: krishiv_types::parse_data_type(f["type"].as_str().expect("type"))
                    .expect("type"),
                nullable: f["nullable"].as_bool().expect("nullable"),
            })
            .collect();
        c = c.with_table(&["default"], name, columns, None);
    }
    c
}

#[expect(
    clippy::expect_used,
    reason = "a benchmark without its inputs cannot run"
)]
fn bench(c: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ctx = context(catalog(&root), ResolutionConf::default());
    let options = ParserOptions::default();
    for q in ["q01", "q05", "q09"] {
        let sql = std::fs::read_to_string(root.join(format!("bench/queries/tpch/{q}.sql")))
            .expect("query file");
        let sql = sql.trim().trim_end_matches(';').to_owned();
        c.bench_function(&format!("plan/tpch_{q}"), |b| {
            b.iter(|| {
                let plan = parse_and_lower(black_box(&sql), &options, SourceId(0)).expect("lowers");
                black_box(block_on(DefaultAnalyzer.analyze(&plan, &ctx)).expect("resolves"))
            });
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
