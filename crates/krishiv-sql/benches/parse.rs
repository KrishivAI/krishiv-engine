// SPDX-License-Identifier: Apache-2.0
//! Parsing large TPC-DS queries (lexing included); run under CodSpeed in
//! CI. `q64` is the widest join/CTE query, `q08` the longest text (a
//! 400-element `IN` list); `all` parses the 99 queries one after another.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::path::Path;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_sql::{ParserOptions, parse_query};

/// The TPC-DS queries of `bench/queries/tpcds`, split into statements.
#[expect(
    clippy::expect_used,
    reason = "a benchmark without its inputs cannot run"
)]
fn tpcds() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/queries/tpcds");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("bench/queries/tpcds")
        .map(|e| e.expect("directory entry").path())
        .collect();
    files.sort();
    let mut out = vec![];
    for path in files {
        let text = std::fs::read_to_string(&path).expect("query file");
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.extend(
            text.split(';')
                .filter(|s| {
                    s.lines()
                        .any(|l| !l.trim().is_empty() && !l.trim().starts_with("--"))
                })
                .map(|s| (name.clone(), s.to_owned())),
        );
    }
    out
}

fn bench(c: &mut Criterion) {
    // The queries quote aliases with `"`, so they parse as Spark does with
    // `spark.sql.ansi.doubleQuotedIdentifiers`.
    let options = ParserOptions {
        double_quoted_identifiers: true,
        ..ParserOptions::default()
    };
    let queries = tpcds();
    for name in ["q64", "q08"] {
        let sql: Vec<&str> = queries
            .iter()
            .filter(|(n, _)| n == name)
            .map(|(_, s)| s.as_str())
            .collect();
        c.bench_function(&format!("parse/tpcds_{name}"), |b| {
            b.iter(|| {
                for s in &sql {
                    let _ = black_box(parse_query(black_box(s), &options));
                }
            });
        });
    }
    c.bench_function("parse/tpcds_all", |b| {
        b.iter(|| {
            for (_, s) in &queries {
                let _ = black_box(parse_query(black_box(s), &options));
            }
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
