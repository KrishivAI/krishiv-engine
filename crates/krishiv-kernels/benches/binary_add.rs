// SPDX-License-Identifier: Apache-2.0
//! 1M-row kernel benchmarks, one per specialization path, against the
//! row-at-a-time reference ("generic path"); run under CodSpeed in CI.
//!
//! Design goals: the no-null fast path is at least 1.2x the generic
//! (row-at-a-time reference) path; `Unchecked`/`JavaWrap` cost the same;
//! `AnsiError` stays close to them; the decimal 64-bit hook beats the
//! 128-bit path. `.../nulls_path` runs the same no-null data with an
//! all-valid bitmap attached: parity with the plain path is why elementwise
//! kernels have no has-nulls axis (keeps that decision measurable).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::types::{Decimal128Type, Float64Type, Int64Type};
use arrow_array::{Decimal128Array, Float64Array, Int64Array, PrimitiveArray};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_kernels::decimal::{DecimalPlan, DecimalStorage, decimal_binary};
use krishiv_kernels::float::{Equal, compare};
use krishiv_kernels::integer::{binary, binary_nulls_path};
use krishiv_kernels::reference;
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, NullOnOverflow, Operand, Overflow, SparkVersion,
    Target, Unchecked,
};

/// The engine's shipped allocator, so 8 MB outputs reuse memory instead of
/// paying fresh-page faults on every iteration.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: i64 = 1 << 20;

fn ints(null_every: Option<i64>, seed: i64) -> Int64Array {
    (0..ROWS)
        .map(|i| {
            let v = i.wrapping_mul(seed) % 1_000_000;
            match null_every {
                Some(n) if i % n == 0 => None,
                _ => Some(v),
            }
        })
        .collect()
}

fn add<O: Overflow>(
    c: &mut Criterion,
    name: &str,
    a: &Int64Array,
    b: &Int64Array,
    ctx: &KernelCtx,
) {
    c.bench_function(name, |bench| {
        bench.iter(|| {
            binary::<Int64Type, Add, O>(
                Operand::Array(black_box(a)),
                Operand::Array(black_box(b)),
                ctx,
            )
            .unwrap()
        });
    });
}

fn nulls_path<O: Overflow>(
    c: &mut Criterion,
    name: &str,
    a: &Int64Array,
    b: &Int64Array,
    ctx: &KernelCtx,
) {
    c.bench_function(name, |bench| {
        bench.iter(|| {
            binary_nulls_path::<Int64Type, Add, O>(
                Operand::Array(black_box(a)),
                Operand::Array(black_box(b)),
                ctx,
            )
            .unwrap()
        });
    });
}

fn integer(c: &mut Criterion) {
    let ctx = KernelCtx::new();
    let (a, b) = (ints(None, 7919), ints(None, 104_729));
    let (an, bn) = (ints(Some(10), 7919), ints(Some(13), 104_729));
    add::<Unchecked>(c, "i64_add/no_nulls/unchecked", &a, &b, &ctx);
    add::<JavaWrap>(c, "i64_add/no_nulls/java_wrap", &a, &b, &ctx);
    add::<AnsiError>(c, "i64_add/no_nulls/ansi", &a, &b, &ctx);
    add::<NullOnOverflow>(c, "i64_add/no_nulls/null_on_overflow", &a, &b, &ctx);
    add::<Unchecked>(c, "i64_add/nulls_10pct/unchecked", &an, &bn, &ctx);
    add::<JavaWrap>(c, "i64_add/nulls_10pct/java_wrap", &an, &bn, &ctx);
    add::<AnsiError>(c, "i64_add/nulls_10pct/ansi", &an, &bn, &ctx);
    // Same no-null data with an all-valid bitmap: should match the plain path.
    nulls_path::<Unchecked>(c, "i64_add/no_nulls/unchecked/nulls_path", &a, &b, &ctx);
    nulls_path::<AnsiError>(c, "i64_add/no_nulls/ansi/nulls_path", &a, &b, &ctx);
    let base = KernelCtx::with_target(Target::Baseline).unwrap();
    add::<AnsiError>(c, "i64_add/no_nulls/ansi/baseline_target", &a, &b, &base);
    c.bench_function("i64_add/no_nulls/ansi/scalar", |bench| {
        bench.iter(|| {
            binary::<Int64Type, Add, AnsiError>(
                Operand::Array(black_box(&a)),
                Operand::Scalar(Some(3)),
                &ctx,
            )
            .unwrap()
        });
    });
    c.bench_function("i64_add/no_nulls/ansi/generic_reference", |bench| {
        bench.iter(|| {
            reference::integer_binary::<Int64Type>(
                ArithOp::Add,
                AnsiError::MODE,
                SparkVersion::V4_2,
                Operand::Array(black_box(&a)),
                Operand::Array(black_box(&b)),
            )
            .unwrap()
        });
    });
}

fn decimals(seed: i128) -> Decimal128Array {
    let a: PrimitiveArray<Decimal128Type> = (0..i128::from(ROWS))
        .map(|i| Some(i.wrapping_mul(seed) % 10_000_000_000))
        .collect();
    a.with_precision_and_scale(10, 2).unwrap()
}

fn decimal(c: &mut Criterion) {
    let ctx = KernelCtx::new();
    let (a, b) = (decimals(7919), decimals(104_729));
    let plan = DecimalPlan::new(ArithOp::Add, (10, 2), (10, 2), (11, 2)).unwrap();
    for (name, storage) in [
        ("i64_hook", DecimalStorage::I64),
        ("i128", DecimalStorage::I128),
    ] {
        let p = plan.with_storage(storage).unwrap();
        c.bench_function(&format!("decimal_add/{name}/ansi"), |bench| {
            bench.iter(|| {
                decimal_binary::<Add, AnsiError>(
                    Operand::Array(black_box(&a)),
                    Operand::Array(black_box(&b)),
                    &p,
                    &ctx,
                )
                .unwrap()
            });
        });
    }
}

fn float(c: &mut Criterion) {
    let ctx = KernelCtx::new();
    #[expect(clippy::cast_precision_loss, reason = "bench data")]
    let mk =
        |s: i64| -> Float64Array { (0..ROWS).map(|i| Some(((i * s) % 1000) as f64)).collect() };
    let (a, b) = (mk(7919), mk(104_729));
    c.bench_function("f64_eq/no_nulls", |bench| {
        bench.iter(|| {
            compare::<Float64Type, Equal>(
                Operand::Array(black_box(&a)),
                Operand::Array(black_box(&b)),
                &ctx,
            )
            .unwrap()
        });
    });
}

criterion_group!(benches, integer, decimal, float);
criterion_main!(benches);
