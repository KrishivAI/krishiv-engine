// SPDX-License-Identifier: Apache-2.0
//! Theta sketch update cost per value (P3-T024, PR-54c): one million distinct non-null values of
//! each of `int64`, `decimal(18,2)`, a 16-byte `string` and `date` through the writer's
//! `theta_value` path (Iceberg single-value serialisation, then the default sketch, lg_k 12).
//! Criterion reports the time per run and, through `Throughput::Elements`, per value.

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

use arrow_array::{Array, ArrayRef, Date32Array, Decimal128Array, Int64Array, StringArray};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use krishiv_connectors_iceberg::plan::{IcebergType, bench_estimate};

/// Rows per column.
const ROWS: i32 = 1_000_000;

fn theta_update(c: &mut Criterion) {
    let cases: [(&str, ArrayRef, IcebergType); 4] = [
        (
            "int64",
            Arc::new(Int64Array::from_iter_values(
                (0..i64::from(ROWS)).map(|i| i.wrapping_mul(0x9E37_79B9)),
            )),
            IcebergType::Long,
        ),
        (
            "decimal_18_2",
            Arc::new(
                Decimal128Array::from_iter_values(
                    (0..i128::from(ROWS)).map(|i| i * 37 - 5_000_000),
                )
                .with_precision_and_scale(18, 2)
                .expect("precision 18 scale 2"),
            ),
            IcebergType::Decimal {
                precision: 18,
                scale: 2,
            },
        ),
        (
            "string_16b",
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|i| format!("{i:016}")),
            )),
            IcebergType::String,
        ),
        (
            "date",
            Arc::new(Date32Array::from_iter_values(0..ROWS)),
            IcebergType::Date,
        ),
    ];
    let mut group = c.benchmark_group("theta_update");
    group.sample_size(10);
    group.throughput(Throughput::Elements(u64::try_from(ROWS).unwrap_or(0)));
    for (name, array, ty) in &cases {
        assert_eq!(array.len(), ROWS as usize);
        group.bench_function(*name, |b| {
            b.iter(|| black_box(bench_estimate(array.as_ref(), ty)));
        });
    }
    group.finish();
}

criterion_group!(benches, theta_update);
criterion_main!(benches);
