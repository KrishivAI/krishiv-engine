// SPDX-License-Identifier: Apache-2.0
//! Differential fuzz target: every specialization of the framework's
//! kernels (integer and decimal `+ - *` under every overflow policy, Spark
//! float comparisons) on arbitrary sliced, NULL-bearing arrays, on every
//! available CPU target, must equal the row-at-a-time reference.

#![no_main]

use arrow_array::types::{Decimal128Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::{ArrowNativeType, NullBuffer};
use krishiv_kernels::decimal::{DecimalPlan, DecimalStorage, decimal_binary};
use krishiv_kernels::compare::{self, CompareKind, CompareOp, Equal, Greater, GreaterEq, Less, LessEq};
use krishiv_kernels::integer::{SparkIntegral, binary};
use krishiv_kernels::ops::BinaryOp;
use krishiv_kernels::reference;
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, Mul, NullOnOverflow, Operand, Overflow,
    SparkVersion, Sub, Target, Unchecked,
};
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;

/// A column: raw values and a separate validity, so arbitrary (possibly
/// overflowing) values sit under NULL slots, as Arrow allows.
type Col<T> = (Vec<T>, Vec<bool>);

/// Two sliced arrays and a shape: 0 array/array, 1 array/scalar, 2 scalar/array.
struct Input<T> {
    l: Col<T>,
    r: Col<T>,
    offset: usize,
    shape: u8,
    scalar: Option<T>,
}

/// `value(u, valid)` draws one value; `valid` is false under a NULL.
fn input<T>(
    u: &mut Unstructured<'_>,
    mut value: impl FnMut(&mut Unstructured<'_>, bool) -> Result<T>,
) -> Result<Input<T>> {
    let len = u.int_in_range(0..=300_usize)?;
    let offset = u.int_in_range(0..=7_usize)?;
    let null_mod = u.int_in_range(0..=4_u8)?;
    let mut col = |u: &mut Unstructured<'_>| -> Result<Col<T>> {
        let (mut values, mut valid) = (Vec::new(), Vec::new());
        for i in 0..len + offset {
            let null = match null_mod {
                0 => false,
                1 => true,
                m => i % usize::from(m) == 0 || u.ratio(1, 16)?,
            };
            values.push(value(u, !null)?);
            valid.push(!null);
        }
        Ok((values, valid))
    };
    let l = col(u)?;
    let r = col(u)?;
    let shape = u.int_in_range(0..=2)?;
    let scalar = if u.ratio(1, 8)? {
        None
    } else {
        Some(value(u, true)?)
    };
    Ok(Input {
        l,
        r,
        offset,
        shape,
        scalar,
    })
}

fn array<P: ArrowPrimitiveType>(c: &Col<P::Native>, offset: usize) -> PrimitiveArray<P> {
    let a = PrimitiveArray::<P>::new(c.0.clone().into(), Some(NullBuffer::from(c.1.clone())));
    a.slice(offset, c.0.len() - offset)
}

impl<T: ArrowNativeType> Input<T> {
    fn run<P: ArrowPrimitiveType<Native = T>, R>(
        &self,
        f: impl Fn(Operand<'_, P>, Operand<'_, P>) -> R,
        fix: impl Fn(PrimitiveArray<P>) -> PrimitiveArray<P>,
    ) -> R {
        let (l, r) = (
            fix(array::<P>(&self.l, self.offset)),
            fix(array::<P>(&self.r, self.offset)),
        );
        match self.shape {
            0 => f(Operand::Array(&l), Operand::Array(&r)),
            1 => f(Operand::Array(&l), Operand::Scalar(self.scalar)),
            _ => f(Operand::Scalar(self.scalar), Operand::Array(&r)),
        }
    }
}

fn ctxs() -> impl Iterator<Item = KernelCtx> {
    Target::available().filter_map(|t| KernelCtx::with_target(t).ok())
}

fn int<P: SparkIntegral, Op: BinaryOp, O: Overflow>(i: &Input<P::Native>) {
    let slow = i.run::<P, _>(
        |l, r| reference::integer_binary::<P>(Op::OP, O::MODE, SparkVersion::V4_2, l, r),
        |a| a,
    );
    for ctx in ctxs() {
        let fast = i.run::<P, _>(|l, r| binary::<P, Op, O>(l, r, &ctx), |a| a);
        assert_eq!(fast, slow);
    }
}

fn int_all<P: SparkIntegral>(op: u8, i: &Input<P::Native>) {
    macro_rules! policies {
        ($op:ty) => {{
            int::<P, $op, Unchecked>(i);
            int::<P, $op, JavaWrap>(i);
            int::<P, $op, NullOnOverflow>(i);
            int::<P, $op, AnsiError>(i);
        }};
    }
    match op % 3 {
        0 => policies!(Add),
        1 => policies!(Sub),
        _ => policies!(Mul),
    }
}

fn cmp<Op: CompareOp>(kind: CompareKind, i: &Input<f64>) {
    let slow = i.run::<Float64Type, _>(|l, r| reference::compare::<Float64Type>(kind, l, r), |a| a);
    for ctx in ctxs() {
        let fast =
            i.run::<Float64Type, _>(|l, r| compare::compare::<Float64Type, Op>(l, r, &ctx), |a| a);
        assert_eq!(fast, slow);
    }
}

fn dec<Op: BinaryOp, O: krishiv_kernels::decimal::DecimalOverflow>(
    plan: &DecimalPlan,
    l: (u8, i8),
    r: (u8, i8),
    i: &Input<i128>,
) {
    // Values were drawn within the smaller of the two precisions.
    let fix = |t: (u8, i8)| {
        move |a: PrimitiveArray<Decimal128Type>| a.with_precision_and_scale(t.0, t.1).unwrap()
    };
    let call = |f: &dyn Fn(Operand<'_, Decimal128Type>, Operand<'_, Decimal128Type>) -> _| {
        let (la, ra) = (
            fix(l)(array(&i.l, i.offset)),
            fix(r)(array(&i.r, i.offset)),
        );
        match i.shape {
            0 => f(Operand::Array(&la), Operand::Array(&ra)),
            1 => f(Operand::Array(&la), Operand::Scalar(i.scalar)),
            _ => f(Operand::Scalar(i.scalar), Operand::Array(&ra)),
        }
    };
    let slow = call(&|a, b| reference::decimal_binary(plan, O::MODE, a, b));
    for s in [DecimalStorage::I128, DecimalStorage::I64] {
        let Ok(p) = plan.with_storage(s) else {
            continue;
        };
        for ctx in ctxs() {
            let fast = call(&|a, b| decimal_binary::<Op, O>(a, b, &p, &ctx));
            assert_eq!(fast, slow);
        }
    }
}

fn decimal(u: &mut Unstructured<'_>) -> Result<()> {
    let ty = |u: &mut Unstructured<'_>| -> Result<(u8, i8)> {
        let p = u.int_in_range(1..=38_u8)?;
        let s = u.int_in_range(0..=p.min(19))?;
        Ok((p, i8::try_from(s).unwrap_or(0)))
    };
    let l = ty(u)?;
    let r = ty(u)?;
    let op = u.int_in_range(0..=2_u8)?;
    let (arith, os) = match op {
        0 => (ArithOp::Add, l.1.max(r.1)),
        1 => (ArithOp::Subtract, l.1.max(r.1)),
        _ => (ArithOp::Multiply, l.1 + r.1),
    };
    let out_p = u.int_in_range(u8::try_from(os).unwrap_or(38).max(1)..=38)?;
    let Ok(plan) = DecimalPlan::new(arith, l, r, (out_p, os)) else {
        return Ok(());
    };
    let m = 10_i128.pow(u32::from(l.0.min(r.0))) - 1;
    // Valid values respect the precision; values under NULLs are any i128.
    let i = input(u, |u, valid| if valid { u.int_in_range(-m..=m) } else { u.arbitrary() })?;
    macro_rules! policies {
        ($op:ty) => {{
            dec::<$op, NullOnOverflow>(&plan, l, r, &i);
            dec::<$op, AnsiError>(&plan, l, r, &i);
            if plan.overflow_impossible() {
                dec::<$op, Unchecked>(&plan, l, r, &i);
            }
        }};
    }
    match arith {
        ArithOp::Add => policies!(Add),
        ArithOp::Subtract => policies!(Sub),
        ArithOp::Multiply => policies!(Mul),
    }
    Ok(())
}

fn one(data: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(data);
    match u.int_in_range(0..=3_u8)? {
        0 => {
            let op = u.arbitrary()?;
            let i = input(&mut u, |u, _| u.arbitrary::<i64>())?;
            int_all::<Int64Type>(op, &i);
        }
        1 => {
            let op = u.arbitrary()?;
            let i = input(&mut u, |u, _| u.arbitrary::<i32>())?;
            int_all::<Int32Type>(op, &i);
        }
        2 => {
            let i = input(&mut u, |u, _| {
                Ok(match u.int_in_range(0..=4_u8)? {
                    0 => f64::NAN,
                    1 => -0.0,
                    2 => 0.0,
                    _ => u.arbitrary::<f64>()?,
                })
            })?;
            cmp::<Equal>(CompareKind::Equal, &i);
            cmp::<Less>(CompareKind::Less, &i);
            cmp::<LessEq>(CompareKind::LessEq, &i);
            cmp::<Greater>(CompareKind::Greater, &i);
            cmp::<GreaterEq>(CompareKind::GreaterEq, &i);
        }
        _ => decimal(&mut u)?,
    }
    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let _ = one(data);
});
