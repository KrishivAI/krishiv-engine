// SPDX-License-Identifier: Apache-2.0
//! Arithmetic kernel benchmarks per operator family (64K rows, one
//! thread; run under CodSpeed in CI): integral `% div`, unary ops and
//! narrow-type `+`.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::types::{Int8Type, Int32Type, Int64Type};
use arrow_array::{Int8Array, Int32Array, Int64Array};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_kernels::integer::{UnaryOp, binary, integral_divide, remainder, unary};
use krishiv_kernels::{Add, AnsiError, JavaWrap, KernelCtx, Operand};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: i64 = 1 << 16;

/// Deterministic values; every `null_every`-th row is NULL.
fn longs(null_every: Option<i64>, seed: i64, nonzero: bool) -> Int64Array {
    (0..ROWS)
        .map(|i| {
            let v = i.wrapping_mul(seed) % 1_000_003;
            let v = if nonzero && v == 0 { 7 } else { v };
            match null_every {
                Some(n) if i % n == 0 => None,
                _ => Some(v),
            }
        })
        .collect()
}

fn integral(c: &mut Criterion) {
    let ctx = KernelCtx::new();
    let a = longs(None, 7919, false);
    let b = longs(None, 104_729, true);
    let bn = longs(Some(10), 104_729, true);
    c.bench_function("i64_div/ansi", |bench| {
        bench.iter(|| {
            integral_divide::<AnsiError>(Operand::Array(black_box(&a)), Operand::Array(&b), &ctx)
                .unwrap()
        });
    });
    c.bench_function("i64_div/nulls_10pct/ansi", |bench| {
        bench.iter(|| {
            integral_divide::<AnsiError>(Operand::Array(black_box(&a)), Operand::Array(&bn), &ctx)
                .unwrap()
        });
    });
    c.bench_function("i64_rem/java_wrap", |bench| {
        bench.iter(|| {
            remainder::<Int64Type, JavaWrap>(
                Operand::Array(black_box(&a)),
                Operand::Array(&b),
                &ctx,
            )
            .unwrap()
        });
    });
    c.bench_function("i64_abs/ansi", |bench| {
        bench.iter(|| unary::<Int64Type, AnsiError>(UnaryOp::Abs, black_box(&a), &ctx).unwrap());
    });
    let i: Int32Array = (0..ROWS)
        .map(|v| Some(i32::try_from(v % 50_000).unwrap()))
        .collect();
    c.bench_function("i32_add/ansi", |bench| {
        bench.iter(|| {
            binary::<Int32Type, Add, AnsiError>(
                Operand::Array(black_box(&i)),
                Operand::Scalar(Some(3)),
                &ctx,
            )
            .unwrap()
        });
    });
    let t: Int8Array = (0..ROWS)
        .map(|v| Some(i8::try_from(v % 60).unwrap()))
        .collect();
    c.bench_function("i8_add/ansi", |bench| {
        bench.iter(|| {
            binary::<Int8Type, Add, AnsiError>(
                Operand::Array(black_box(&t)),
                Operand::Array(&t),
                &ctx,
            )
            .unwrap()
        });
    });
}

criterion_group!(benches, integral);
criterion_main!(benches);
