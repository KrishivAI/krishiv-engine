// SPDX-License-Identifier: Apache-2.0
//! An identity UDF against the built-in it mirrors: invoking each over
//! 8,192-row batches, through the same [`ScalarFunction`] interface the
//! evaluator uses. The UDF's per-batch cost over `abs` is the wrapper
//! (checks, null masks, unwind boundary): the identity itself copies its
//! input in both cases.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array};
use arrow_schema::DataType;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use krishiv_functions::scalar::{Arg, BindArg, BindOptions, ScalarFunction, bind_scalar, bind_udf};
use krishiv_types::DataType as SparkType;
use krishiv_udf_sdk::{
    ArgField, InvokeContext, ReturnField, ScalarUdf, Signature, UdfArg, UdfError, UdfInputType,
};

const ROWS: usize = 8_192;

/// The identity over `INT`.
#[derive(Debug)]
struct Identity;

impl ScalarUdf for Identity {
    fn name(&self) -> &str {
        "identity_i32"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(SparkType::Integer)]))
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: SparkType::Integer,
            nullable: false,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        _: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        Ok(args[0].array())
    }
}

fn batch() -> ArrayRef {
    // Negative and positive values, so `abs` computes on every row.
    Arc::new(Int32Array::from_iter_values(
        (0..ROWS).map(|i| i32::try_from(i).unwrap_or(0).wrapping_sub(4096)),
    ))
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn bound_abs() -> Arc<dyn ScalarFunction> {
    bind_scalar(
        "abs",
        &[BindArg::Column(DataType::Int32)],
        &BindOptions::new(),
    )
    .expect("abs(int)")
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn bound_identity() -> Arc<dyn ScalarFunction> {
    let udf: Arc<dyn ScalarUdf> = Arc::new(Identity);
    bind_udf(
        &udf,
        &[DataType::Int32],
        &SparkType::Integer,
        false,
        true,
        None,
    )
    .expect("identity")
}

#[expect(
    clippy::expect_used,
    reason = "a failed invocation must fail the bench loudly, not skip it"
)]
fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("udf");
    group.throughput(Throughput::Elements(ROWS as u64));
    let input = batch();
    for (id, bound) in [("abs", bound_abs()), ("identity_udf", bound_identity())] {
        group.bench_with_input(
            BenchmarkId::new("invoke_8192_rows", id),
            &bound,
            |b, bound: &Arc<dyn ScalarFunction>| {
                b.iter(|| {
                    let out = bound
                        .invoke(black_box(&[Arg::Array(Arc::clone(&input))]), ROWS)
                        .expect("invoke");
                    black_box(out.len())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
