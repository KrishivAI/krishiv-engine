// SPDX-License-Identifier: Apache-2.0
//! Row-table benchmarks (P2-T157 prototype), single thread: the salted
//! slot table and `hashbrown` over the same rows, with today's arrow-row
//! `KeyTable` as the reference.
//!
//! - `row_table/<index>_build_64k`: 64 Ki distinct `Int64` keys, each
//!   inserted twice (aggregate-shaped find-or-insert), batches of 4,096.
//! - `row_table/<index>_probe_64k`: probe 128 Ki keys into a 64 Ki-key
//!   table, all hits (join-shaped find).
//! - `row_table/<index>_build_strings_64k`: the build with 24-byte string
//!   keys (long views, heap compare).
//!
//! These are cache-resident sizes for CodSpeed; the go/no-go matrix at
//! 1e4–1.5e7 keys is `cargo run --release --example row_table_matrix`.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, StringViewArray};
use arrow_schema::DataType;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use krishiv_exec::hash::{HashSeed, KeyEncoder, KeyTable};
use krishiv_exec::rows::{HashbrownTable, RowIndex, RowLayout, RowPages, SaltedTable};

const KEYS: u64 = 1 << 16;
const BATCH: usize = 4096;

/// A bijective scramble so keys are not dense small integers.
fn scramble(x: u64) -> u64 {
    x.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Batches of keys `order[i] % KEYS`, as `Int64` or as 24-byte strings.
fn batches(order: &[u64], strings: bool) -> Vec<ArrayRef> {
    order
        .chunks(BATCH)
        .map(|c| -> ArrayRef {
            if strings {
                Arc::new(StringViewArray::from_iter_values(
                    c.iter().map(|&r| format!("{:024x}", scramble(r))),
                ))
            } else {
                Arc::new(Int64Array::from_iter_values(
                    c.iter().map(|&r| scramble(r).cast_signed()),
                ))
            }
        })
        .collect()
}

/// Every key twice, the second pass in a different order.
fn build_order() -> Vec<u64> {
    (0..KEYS)
        .chain((0..KEYS).map(|i| (i * 7919) % KEYS))
        .collect()
}

fn index(name: &str) -> Box<dyn RowIndex> {
    if name == "salted" {
        Box::new(SaltedTable::new(5))
    } else {
        Box::new(HashbrownTable::new())
    }
}

fn run_build(layout: &RowLayout, input: &[ArrayRef], name: &str) -> usize {
    let mut table = index(name);
    let mut pages = RowPages::new(layout);
    let mut ids = Vec::new();
    for a in input {
        let keys = layout.encode(&[Arc::clone(a)], 42).expect("encode");
        table
            .find_or_insert(layout, &keys, &mut pages, &mut ids)
            .expect("insert");
    }
    table.len()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("row_table");
    g.sample_size(10);
    let build = build_order();
    let probe: Vec<u64> = (0..2 * KEYS).map(|i| (i * 104_729) % KEYS).collect();
    for strings in [false, true] {
        let t = if strings {
            DataType::Utf8View
        } else {
            DataType::Int64
        };
        let layout = RowLayout::new(&[t], &[false]).expect("layout");
        let input = batches(&build, strings);
        let suffix = if strings { "_strings" } else { "" };
        g.throughput(Throughput::Elements(2 * KEYS));
        for name in ["salted", "hashbrown"] {
            g.bench_function(format!("{name}_build{suffix}_64k"), |b| {
                b.iter(|| black_box(run_build(&layout, &input, name)));
            });
        }
    }
    let layout = RowLayout::new(&[DataType::Int64], &[false]).expect("layout");
    let input = batches(&build, false);
    let probes = batches(&probe, false);
    for name in ["salted", "hashbrown"] {
        let mut table = index(name);
        let mut pages = RowPages::new(&layout);
        let mut ids = Vec::new();
        for a in &input {
            let keys = layout.encode(&[Arc::clone(a)], 42).expect("encode");
            table
                .find_or_insert(&layout, &keys, &mut pages, &mut ids)
                .expect("insert");
        }
        g.bench_function(format!("{name}_probe_64k"), |b| {
            b.iter(|| {
                let mut out = Vec::new();
                for a in &probes {
                    let keys = layout.encode(&[Arc::clone(a)], 42).expect("encode");
                    table.find(&layout, &keys, &pages, &mut out);
                    black_box(&out);
                }
            });
        });
    }
    // Today's path: arrow-row keys into `KeyTable`.
    let encoder = KeyEncoder::new(&[DataType::Int64]).expect("encoder");
    let seed = HashSeed::new(Some(42));
    g.bench_function("keytable_build_64k", |b| {
        b.iter(|| {
            let mut t = KeyTable::new();
            let mut ids = Vec::new();
            for a in &input {
                let rows = encoder.encode(&[Arc::clone(a)]).expect("encode");
                t.find_or_insert(&rows, &seed, &mut ids).expect("insert");
            }
            black_box(t.len())
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
