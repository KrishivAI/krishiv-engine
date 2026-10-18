// SPDX-License-Identifier: Apache-2.0
//! Cast kernel benchmarks (64K rows, one thread; CodSpeed in CI): the
//! TPC-H load and DataFrame casts — string to int/decimal/date, decimal
//! rescale, int to decimal and decimal to string.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::{Decimal128Array, Int64Array, StringArray};
use arrow_schema::DataType;
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: i64 = 1 << 16;

fn casts(c: &mut Criterion) {
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: &FixedOffset(0),
    };
    let mut bench = |name: &str, input: &dyn arrow_array::Array, to: DataType| {
        c.bench_function(name, |b| {
            b.iter(|| cast(black_box(input), &to, &opts).unwrap());
        });
    };
    let ints: StringArray = (0..ROWS)
        .map(|i| Some((i * 7919 % 1_000_000).to_string()))
        .collect();
    bench("string_to_bigint", &ints, DataType::Int64);
    let decs: StringArray = (0..ROWS)
        .map(|i| Some(format!("{}.{:02}", i * 7919 % 100_000, i % 100)))
        .collect();
    bench("string_to_decimal_15_2", &decs, DataType::Decimal128(15, 2));
    let dates: StringArray = (0..ROWS)
        .map(|i| Some(format!("199{}-{:02}-{:02}", i % 8, i % 12 + 1, i % 28 + 1)))
        .collect();
    bench("string_to_date", &dates, DataType::Date32);
    let d: Decimal128Array = (0..ROWS).map(|i| Some(i128::from(i) * 7919)).collect();
    let d = d.with_precision_and_scale(15, 2).unwrap();
    bench("decimal_15_2_to_12_1", &d, DataType::Decimal128(12, 1));
    bench("decimal_to_string", &d, DataType::Utf8);
    let l: Int64Array = (0..ROWS).map(Some).collect();
    bench("bigint_to_decimal_20_2", &l, DataType::Decimal128(20, 2));
}

criterion_group!(benches, casts);
criterion_main!(benches);
