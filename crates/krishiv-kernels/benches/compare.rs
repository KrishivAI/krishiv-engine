// SPDX-License-Identifier: Apache-2.0
//! Comparison kernel benchmarks (64K rows, one thread; CodSpeed in CI):
//! integral, decimal and string `<`/`=`, and `<=>` on top of `=`.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::types::{Decimal128Type, Int64Type};
use arrow_array::{Array, Decimal128Array, Int64Array, StringArray};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_kernels::compare::{
    BytesOperand, Equal, Less, Validity, compare, compare_bytes, null_safe,
};
use krishiv_kernels::{KernelCtx, Operand};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: i64 = 1 << 16;

fn compare_benches(c: &mut Criterion) {
    let ctx = KernelCtx::new();
    let a: Int64Array = (0..ROWS)
        .map(|i| Some(i.wrapping_mul(7919) % 1000))
        .collect();
    let b: Int64Array = (0..ROWS)
        .map(|i| (i % 10 != 0).then_some(i.wrapping_mul(104_729) % 1000))
        .collect();
    c.bench_function("i64_lt", |bench| {
        bench.iter(|| {
            compare::<Int64Type, Less>(Operand::Array(black_box(&a)), Operand::Array(&b), &ctx)
                .unwrap()
        });
    });
    c.bench_function("i64_null_safe_eq", |bench| {
        bench.iter(|| {
            let eq = compare::<Int64Type, Equal>(
                Operand::Array(black_box(&a)),
                Operand::Array(&b),
                &ctx,
            )
            .unwrap();
            null_safe(&eq, Validity::Array(a.nulls()), Validity::Array(b.nulls())).unwrap()
        });
    });
    let d: Decimal128Array = a.iter().map(|v| v.map(i128::from)).collect();
    c.bench_function("decimal_eq/scalar", |bench| {
        bench.iter(|| {
            compare::<Decimal128Type, Equal>(
                Operand::Array(black_box(&d)),
                Operand::Scalar(Some(500)),
                &ctx,
            )
            .unwrap()
        });
    });
    let s: StringArray = (0..ROWS)
        .map(|i| Some(format!("item-{:05}", i % 997)))
        .collect();
    c.bench_function("string_lt/scalar", |bench| {
        bench.iter(|| {
            compare_bytes::<_, Less>(
                BytesOperand::Array(black_box(&s)),
                BytesOperand::Scalar(Some(b"item-00500")),
            )
            .unwrap()
        });
    });
}

criterion_group!(benches, compare_benches);
criterion_main!(benches);
