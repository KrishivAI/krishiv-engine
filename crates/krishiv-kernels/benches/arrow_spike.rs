// SPDX-License-Identifier: Apache-2.0
//! Spike: Krishiv's `LIKE` and string comparison kernels against arrow-rs's
//! `arrow-string` / `arrow-ord` on the same 64K-row column.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::{Scalar, StringArray};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_kernels::compare::{BytesOperand, Equal, Less, compare_bytes};
use krishiv_kernels::string::{StrOperand, like};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: usize = 1 << 16;

fn column() -> StringArray {
    (0..ROWS)
        .map(|i| {
            (i % 17 != 0).then(|| {
                format!(
                    "Customer#{:09}-{}-{}",
                    i.wrapping_mul(7919) % 150_000,
                    ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD"][i % 4],
                    if i % 5 == 0 {
                        "special requests"
                    } else {
                        "regular"
                    }
                )
            })
        })
        .collect()
}

fn spike(c: &mut Criterion) {
    let col = column();
    for (name, pat) in [
        ("prefix", "Customer#0000%"),
        ("suffix", "%regular"),
        ("contains", "%special%"),
        ("general", "%#00_1%-MACH%special%"),
    ] {
        c.bench_function(&format!("like_{name}/krishiv"), |b| {
            b.iter(|| {
                like::<i32>(
                    StrOperand::Array(black_box(&col)),
                    StrOperand::Scalar(Some(pat)),
                    '\\',
                )
                .unwrap()
            });
        });
        let p = Scalar::new(StringArray::from(vec![pat]));
        c.bench_function(&format!("like_{name}/arrow"), |b| {
            b.iter(|| arrow_string::like::like(black_box(&col), &p).unwrap());
        });
    }
    let lit = "Customer#000050000-MACHINERY-regular";
    let s = Scalar::new(StringArray::from(vec![lit]));
    c.bench_function("str_eq_scalar/krishiv", |b| {
        b.iter(|| {
            compare_bytes::<_, Equal>(
                BytesOperand::Array(black_box(&col)),
                BytesOperand::Scalar(Some(lit.as_bytes())),
            )
            .unwrap()
        });
    });
    c.bench_function("str_eq_scalar/arrow", |b| {
        b.iter(|| arrow_ord::cmp::eq(black_box(&col), &s).unwrap());
    });
    c.bench_function("str_lt_scalar/krishiv", |b| {
        b.iter(|| {
            compare_bytes::<_, Less>(
                BytesOperand::Array(black_box(&col)),
                BytesOperand::Scalar(Some(lit.as_bytes())),
            )
            .unwrap()
        });
    });
    c.bench_function("str_lt_scalar/arrow", |b| {
        b.iter(|| arrow_ord::cmp::lt(black_box(&col), &s).unwrap());
    });
    let other: StringArray = col
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.map(|v| {
                if i % 3 == 0 {
                    v.to_owned()
                } else {
                    format!("{v}x")
                }
            })
        })
        .collect();
    c.bench_function("str_eq_array/krishiv", |b| {
        b.iter(|| {
            compare_bytes::<_, Equal>(
                BytesOperand::Array(black_box(&col)),
                BytesOperand::Array(&other),
            )
            .unwrap()
        });
    });
    c.bench_function("str_eq_array/arrow", |b| {
        b.iter(|| arrow_ord::cmp::eq(black_box(&col), &other).unwrap());
    });
}

criterion_group!(benches, spike);
criterion_main!(benches);
