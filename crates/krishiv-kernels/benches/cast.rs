// SPDX-License-Identifier: Apache-2.0
//! Cast kernel benchmarks (64K rows, one thread; CodSpeed in CI): the
//! TPC-H load and DataFrame casts — string to int/decimal/date, decimal
//! rescale, int to decimal, decimal to string, double to decimal and to string (fares), and the
//! widening integral casts (ClickBench's `CAST(smallint AS INT)`).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::{
    Decimal128Array, Float64Array, Int16Array, Int32Array, Int64Array, StringArray,
    Time64NanosecondArray,
};
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
    // Taxi fares: dollars and cents read as doubles (PY-063).
    #[expect(clippy::cast_precision_loss, reason = "small integers")]
    let fares: Float64Array = (0..ROWS)
        .map(|i| Some((i * 7919 % 20_000) as f64 / 100.0 + 2.5))
        .collect();
    bench(
        "double_to_decimal_12_2",
        &fares,
        DataType::Decimal128(12, 2),
    );
    bench("double_to_string", &fares, DataType::Utf8);
    let times: StringArray = (0..ROWS)
        .map(|i| {
            Some(format!(
                "{:02}:{:02}:{:02}.{:06}",
                i % 24,
                i % 60,
                i * 7 % 60,
                i * 7919 % 1_000_000
            ))
        })
        .collect();
    bench(
        "string_to_time",
        &times,
        DataType::Time64(arrow_schema::TimeUnit::Nanosecond),
    );
    let t: Time64NanosecondArray = (0..ROWS).map(|i| Some(i * 1_319_000)).collect();
    bench("time_to_string", &t, DataType::Utf8);
    bench("time_to_bigint", &t, DataType::Int64);
    // ClickBench's `CAST(smallint AS INT)` (q30, q02, q08), with one NULL in 16.
    let shorts: Int16Array = (0..ROWS)
        .map(|i| (i % 16 != 0).then_some(i16::try_from(i * 7919 % 2_000).unwrap()))
        .collect();
    bench("smallint_to_int", &shorts, DataType::Int32);
    let ints: Int32Array = (0..ROWS)
        .map(|i| Some(i32::try_from(i * 7919).unwrap()))
        .collect();
    bench("int_to_bigint", &ints, DataType::Int64);
}

criterion_group!(benches, casts);
criterion_main!(benches);
