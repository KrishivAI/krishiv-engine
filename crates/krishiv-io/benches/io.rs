// SPDX-License-Identifier: Apache-2.0
//! File-scan planning and I/O request shaping, run under CodSpeed in CI
//! (single-threaded, no I/O):
//! - Spark split planning of 10,000 listed files (sizes up to 300 MiB, a
//!   third not splittable);
//! - coalescing 10,000 column-chunk ranges of one file into requests;
//! - encoding 16,384 rows (an integer and a string column) as one Parquet
//!   file with the default (snappy) settings, footer and page indexes
//!   included: arrow-rs's encoder plus the writer's own per-batch work.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;
use std::ops::Range;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_common::config::Config;
use krishiv_io::io::coalesce;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::{SplitConfig, SplitFile, plan};
use krishiv_io::{FileFormat, FormatOptions};

fn files() -> Vec<SplitFile> {
    (0..10_000_u64)
        .map(|i| SplitFile {
            size: (i.wrapping_mul(2_654_435_761)) % (300 << 20) + 1,
            splittable: i % 3 != 0,
        })
        .collect()
}

/// Column chunks of 500 row groups × 20 columns, a few KiB to a few MiB
/// apart, listed in column-major order (as a projection asks for them).
fn ranges() -> Vec<Range<u64>> {
    let mut out = Vec::with_capacity(10_000);
    for col in 0..20_u64 {
        for rg in 0..500_u64 {
            let start = rg * (8 << 20) + col * (300 << 10) + (rg * col) % 4096;
            out.push(start..start + (200 << 10) + (col * 977) % 65_536);
        }
    }
    out
}

/// 16,384 rows in batches of 2,048: a key and a short, repetitive string.
fn rows() -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8View, true),
    ]));
    (0..8_i64)
        .map(|b| {
            let keys = (b * 2048..(b + 1) * 2048).map(|i| i.wrapping_mul(2_654_435_761));
            let strings = (b * 2048..(b + 1) * 2048).map(|i| Some(format!("value-{}", i % 97)));
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(keys)),
                    Arc::new(StringViewArray::from_iter(strings)),
                ],
            )
            .expect("bench batch")
        })
        .collect()
}

/// The size of the Parquet file of `batches`.
fn encode(batches: &[RecordBatch], o: &FormatOptions) -> usize {
    let mut enc = ParquetFormat::encoder_with(&batches[0].schema(), o, |p| p).expect("encoder");
    let mut size = 0;
    for b in batches {
        enc.write(b).expect("write");
        while let Some(bytes) = enc.take_bytes() {
            size += bytes.len();
        }
    }
    size + enc.finish().expect("finish").tail.len()
}

fn bench(c: &mut Criterion) {
    let files = files();
    let cfg = SplitConfig::from_config(&Config::new(), 16);
    let ranges = ranges();
    let mut group = c.benchmark_group("io");
    group.bench_function("split_plan_10k_files", |b| {
        b.iter(|| plan(black_box(&cfg), black_box(&files)));
    });
    group.bench_function("coalesce_10k_ranges", |b| {
        b.iter(|| coalesce(black_box(&ranges), 1 << 20, 16 << 20));
    });
    let batches = rows();
    let o = FormatOptions::defaults(ParquetFormat.options());
    group.bench_function("parquet_encode_16k_rows", |b| {
        b.iter(|| encode(black_box(&batches), &o));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
