// SPDX-License-Identifier: Apache-2.0
//! Parsing and lowering TPC-H queries into the logical-plan IR, run under
//! CodSpeed in CI: the front-end share of planning latency (analysis is
//! measured with the analyzer).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::path::Path;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_plan::SourceId;
use krishiv_sql::{ParserOptions, parse_and_lower};

#[expect(
    clippy::expect_used,
    reason = "a benchmark without its inputs cannot run"
)]
fn bench(c: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/queries/tpch");
    let options = ParserOptions::default();
    for q in ["q01", "q05", "q09"] {
        let sql = std::fs::read_to_string(root.join(format!("{q}.sql"))).expect("query file");
        let sql = sql.trim().trim_end_matches(';').to_owned();
        c.bench_function(&format!("lower/tpch_{q}"), |b| {
            b.iter(|| black_box(parse_and_lower(black_box(&sql), &options, SourceId(0))));
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
