// SPDX-License-Identifier: Apache-2.0
//! Fast path == reference for every specialization (the `kernel_props!`
//! harness of the kernel framework).
//!
//! Each case draws a length from {0, 1, 63, 64, 65, 4095, 4096, random}, a
//! slice offset 0..8 (non-byte-aligned validity bitmaps), a null density
//! from {none, sparse, half, all}, edge values (MIN, MAX, -1, 0, NaN, ±0.0,
//! decimal precision bounds) mixed with random ones, and an operand shape
//! (array/array, array/scalar, scalar/array, NULL scalar). Every monomorph
//! runs on every CPU target this machine supports and must equal the
//! row-at-a-time reference: same values, same NULLs, and on error the same
//! error class, message and row. Under Miri the case count and lengths are
//! reduced (the cases are interpreted).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::fmt::Debug;

use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::{NullBuffer, ScalarBuffer};
use krishiv_kernels::decimal::{DecimalPlan, DecimalStorage, decimal_binary};
use krishiv_kernels::float::{self, CompareOp, Equal, Greater, GreaterEq, Less, LessEq};
use krishiv_kernels::integer::{SparkIntegral, UnaryOp, binary, integral_divide, remainder, unary};
use krishiv_kernels::ops::BinaryOp;
use krishiv_kernels::reference::{self, CompareKind};
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, KernelError, Mul, NullOnOverflow, Operand,
    Overflow, OverflowMode, SparkVersion, Sub, Target, Unchecked,
};
use proptest::prelude::*;

#[cfg(not(miri))]
const LENGTHS: &[usize] = &[0, 1, 63, 64, 65, 4095, 4096];
#[cfg(miri)]
const LENGTHS: &[usize] = &[0, 1, 63, 64, 65, 130];

#[cfg(not(miri))]
const MAX_RANDOM_LEN: usize = 300;
#[cfg(miri)]
const MAX_RANDOM_LEN: usize = 70;

fn config() -> ProptestConfig {
    // `PROPTEST_CASES` (default 256) sets the count; Miri interprets each
    // case, so it runs two.
    let default = ProptestConfig::default();
    ProptestConfig {
        cases: if cfg!(miri) { 2 } else { default.cases },
        failure_persistence: None,
        ..default
    }
}

/// Every available target. `KRISHIV_REQUIRE_TARGETS` (e.g.
/// `baseline,x86_64v3`) makes a missing target a failure, not a skip.
fn contexts() -> Vec<KernelCtx> {
    let required = std::env::var("KRISHIV_REQUIRE_TARGETS").unwrap_or_default();
    Target::check_required(&required).unwrap();
    Target::available()
        .map(|t| KernelCtx::with_target(t).unwrap())
        .collect()
}

/// A column before slicing: `offset` leading junk rows, then `len` rows.
#[derive(Debug, Clone)]
struct Column<T> {
    values: Vec<T>,
    valid: Vec<bool>,
    offset: usize,
}

impl<T: arrow_buffer::ArrowNativeType> Column<T> {
    /// The sliced array. The drawn value stays in the slot under a NULL
    /// (Arrow leaves those slots arbitrary), so overflowing values hide
    /// under NULLs and must never raise.
    fn array<P: ArrowPrimitiveType<Native = T>>(&self) -> PrimitiveArray<P> {
        let all = PrimitiveArray::<P>::new(
            ScalarBuffer::from(self.values.clone()),
            Some(NullBuffer::from(self.valid.clone())),
        );
        all.slice(self.offset, self.values.len() - self.offset)
    }
}

#[derive(Debug, Clone, Copy)]
enum Shape<T> {
    ArrayArray,
    ArrayScalar(Option<T>),
    ScalarArray(Option<T>),
}

#[derive(Debug, Clone)]
struct Case<T> {
    l: Column<T>,
    r: Column<T>,
    shape: Shape<T>,
}

impl<T: arrow_buffer::ArrowNativeType> Case<T> {
    fn with<P: ArrowPrimitiveType<Native = T>, R>(
        &self,
        f: impl Fn(Operand<'_, P>, Operand<'_, P>) -> R,
    ) -> R {
        let (l, r) = (self.l.array::<P>(), self.r.array::<P>());
        match self.shape {
            Shape::ArrayArray => f(Operand::Array(&l), Operand::Array(&r)),
            Shape::ArrayScalar(s) => f(Operand::Array(&l), Operand::Scalar(s)),
            Shape::ScalarArray(s) => f(Operand::Scalar(s), Operand::Array(&r)),
        }
    }
}

fn len_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![proptest::sample::select(LENGTHS), 0..=MAX_RANDOM_LEN,]
}

/// Validity for `n` rows at a density drawn from {none, sparse, half, all}.
fn validity(n: usize) -> impl Strategy<Value = Vec<bool>> {
    prop_oneof![Just(0_u8), Just(1), Just(2), Just(3)].prop_flat_map(move |d| {
        proptest::collection::vec(
            (0..20_u8).prop_map(move |x| match d {
                0 => true,
                1 => x != 0,
                2 => x % 2 == 0,
                _ => false,
            }),
            n,
        )
    })
}

fn case<T: Debug + Clone + Copy + 'static>(
    value: impl Strategy<Value = T> + Clone + 'static,
) -> impl Strategy<Value = Case<T>> {
    (len_strategy(), 0..8_usize, 0..8_usize).prop_flat_map(move |(len, lo, ro)| {
        let col = |off: usize| {
            (
                proptest::collection::vec(value.clone(), len + off),
                validity(len + off),
            )
                .prop_map(move |(values, valid)| Column {
                    values,
                    valid,
                    offset: off,
                })
        };
        let scalar = proptest::option::weighted(0.9, value.clone());
        let shape = prop_oneof![
            2 => Just(Shape::ArrayArray),
            1 => scalar.clone().prop_map(Shape::ArrayScalar),
            1 => scalar.prop_map(Shape::ScalarArray),
        ];
        (col(lo), col(ro), shape).prop_map(|(l, r, shape)| Case { l, r, shape })
    })
}

fn int_value<T>(
    edges: &'static [T],
    small: impl Strategy<Value = T> + Clone + 'static,
    any: impl Strategy<Value = T> + Clone + 'static,
) -> impl Strategy<Value = T> + Clone
where
    T: Debug + Clone + Copy + 'static,
{
    prop_oneof![proptest::sample::select(edges), small, any]
}

fn assert_same<A: PartialEq + Debug>(fast: &Result<A, KernelError>, slow: &Result<A, KernelError>) {
    match (fast, slow) {
        (Ok(f), Ok(s)) => assert_eq!(f, s),
        (Err(f), Err(s)) => {
            assert_eq!(f, s, "error (class, message, row) differs");
            assert!(f.row().is_some());
        }
        _ => panic!("fast {fast:?} vs reference {slow:?}"),
    }
}

/// Every available target, under every Spark version (errors differ by
/// version; values do not); under Miri, one version.
fn version_contexts() -> Vec<KernelCtx> {
    // Miri interprets every case: there one older version stands for all.
    let versions: &[SparkVersion] = if cfg!(miri) {
        &[SparkVersion::V4_0]
    } else {
        &SparkVersion::ALL
    };
    contexts()
        .into_iter()
        .flat_map(|c| versions.iter().map(move |&v| c.with_spark_version(v)))
        .collect()
}

fn check_int<P, Op, O>(c: &Case<P::Native>)
where
    P: SparkIntegral,
    Op: BinaryOp,
    O: Overflow,
{
    for ctx in version_contexts() {
        let v = ctx.spark_version();
        let slow = c.with::<P, _>(|l, r| reference::integer_binary::<P>(Op::OP, O::MODE, v, l, r));
        let fast = c.with::<P, _>(|l, r| binary::<P, Op, O>(l, r, &ctx));
        assert_same(&fast, &slow);
        let slow = c.with::<P, _>(|l, r| reference::integer_remainder::<P>(O::MODE, v, l, r));
        let fast = c.with::<P, _>(|l, r| remainder::<P, O>(l, r, &ctx));
        assert_same(&fast, &slow);
        let left = c.l.array::<P>();
        for op in [UnaryOp::Negate, UnaryOp::Abs] {
            let slow = reference::integer_unary::<P>(op, O::MODE, v, &left);
            assert_same(&unary::<P, O>(op, &left, &ctx), &slow);
        }
    }
}

/// `div` is defined on `bigint` only (Spark widens its operands first).
fn check_div<O: Overflow>(c: &Case<i64>) {
    let slow = c.with::<Int64Type, _>(|l, r| reference::integer_divide(O::MODE, l, r));
    for ctx in contexts() {
        let fast = c.with::<Int64Type, _>(|l, r| integral_divide::<O>(l, r, &ctx));
        assert_same(&fast, &slow);
    }
}

fn check_int_all<P: SparkIntegral, Op: BinaryOp>(c: &Case<P::Native>) {
    check_int::<P, Op, Unchecked>(c);
    check_int::<P, Op, JavaWrap>(c);
    check_int::<P, Op, NullOnOverflow>(c);
    check_int::<P, Op, AnsiError>(c);
}

fn i8_value() -> impl Strategy<Value = i8> + Clone {
    int_value(&[i8::MIN, i8::MAX, -1, 0, 1], -10..10_i8, any::<i8>())
}

fn i16_value() -> impl Strategy<Value = i16> + Clone {
    int_value(&[i16::MIN, i16::MAX, -1, 0, 1], -100..100_i16, any::<i16>())
}

fn i32_value() -> impl Strategy<Value = i32> + Clone {
    int_value(
        &[i32::MIN, i32::MAX, -1, 0, 1],
        -1000..1000_i32,
        any::<i32>(),
    )
}

fn i64_value() -> impl Strategy<Value = i64> + Clone {
    int_value(
        &[i64::MIN, i64::MAX, -1, 0, 1],
        -1000..1000_i64,
        any::<i64>(),
    )
}

macro_rules! int_props {
    ($($name:ident: $p:ty, $op:ty, $v:expr;)*) => {
        proptest! {
            #![proptest_config(config())]
            $(
                #[test]
                fn $name(c in case($v)) {
                    check_int_all::<$p, $op>(&c);
                }
            )*
        }
    };
}

int_props! {
    i8_add: Int8Type, Add, i8_value();
    i8_mul: Int8Type, Mul, i8_value();
    i16_sub: Int16Type, Sub, i16_value();
    i16_mul: Int16Type, Mul, i16_value();
    i32_add: Int32Type, Add, i32_value();
    i32_sub: Int32Type, Sub, i32_value();
    i32_mul: Int32Type, Mul, i32_value();
    i64_add: Int64Type, Add, i64_value();
    i64_sub: Int64Type, Sub, i64_value();
    i64_mul: Int64Type, Mul, i64_value();
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn i64_div(c in case(i64_value())) {
        check_div::<JavaWrap>(&c);
        check_div::<NullOnOverflow>(&c);
        check_div::<AnsiError>(&c);
        check_div::<Unchecked>(&c);
    }
}

fn f64_value() -> impl Strategy<Value = f64> + Clone {
    prop_oneof![
        proptest::sample::select(
            &[
                f64::NAN,
                -f64::NAN,
                0.0,
                -0.0,
                1.0,
                -1.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::MIN,
                f64::MAX,
                f64::MIN_POSITIVE,
            ][..]
        ),
        any::<f64>(),
    ]
}

fn f32_value() -> impl Strategy<Value = f32> + Clone {
    prop_oneof![
        proptest::sample::select(
            &[
                f32::NAN,
                -f32::NAN,
                0.0,
                -0.0,
                1.0,
                -1.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MIN,
                f32::MAX,
            ][..]
        ),
        any::<f32>(),
    ]
}

fn check_cmp<P, Op>(kind: CompareKind, c: &Case<P::Native>)
where
    P: ArrowPrimitiveType<Native: float::SparkFloat>,
    Op: CompareOp,
{
    let slow = c.with::<P, _>(|l, r| reference::compare::<P>(kind, l, r));
    for ctx in contexts() {
        let fast = c.with::<P, _>(|l, r| float::compare::<P, Op>(l, r, &ctx));
        assert_eq!(fast.unwrap(), *slow.as_ref().unwrap());
    }
}

fn check_cmp_all<P: ArrowPrimitiveType<Native: float::SparkFloat>>(c: &Case<P::Native>) {
    check_cmp::<P, Equal>(CompareKind::Equal, c);
    check_cmp::<P, Less>(CompareKind::Less, c);
    check_cmp::<P, LessEq>(CompareKind::LessEq, c);
    check_cmp::<P, Greater>(CompareKind::Greater, c);
    check_cmp::<P, GreaterEq>(CompareKind::GreaterEq, c);
}

/// A decimal type, its values and a result precision.
/// `(op, left type, right type, result precision, values)`.
type DecimalCase = (ArithOp, (u8, i8), (u8, i8), u8, Case<i128>);

type DecOperand<'a> = Operand<'a, Decimal128Type>;
type DecResult = Result<PrimitiveArray<Decimal128Type>, KernelError>;

fn decimal_case() -> impl Strategy<Value = DecimalCase> {
    let ty = (1..=38_u8)
        .prop_flat_map(|p| (Just(p), 0..=p.min(20)))
        .prop_map(|(p, s)| (p, i8::try_from(s).unwrap()));
    let op = proptest::sample::select(&[ArithOp::Add, ArithOp::Subtract, ArithOp::Multiply][..]);
    (op, ty.clone(), ty, 1..=38_u8).prop_flat_map(|(op, l, r, out_p)| {
        let max = |p: u8| 10_i128.pow(u32::from(p)) - 1;
        let (ml, mr) = (max(l.0), max(r.0));
        // One value strategy covering both sides' precisions. Valid values
        // are clamped to their side's precision below; values under NULLs
        // keep any i128, including ones outside the declared precision.
        let m = ml.max(mr);
        let v = prop_oneof![
            Just(m),
            Just(-m),
            Just(0_i128),
            Just(1),
            -1000..1000_i128,
            -m..=m,
            any::<i128>(),
        ];
        case(v).prop_map(move |mut c| {
            for (x, ok) in c.l.values.iter_mut().zip(&c.l.valid) {
                if *ok {
                    *x = (*x).clamp(-ml, ml);
                }
            }
            for (x, ok) in c.r.values.iter_mut().zip(&c.r.valid) {
                if *ok {
                    *x = (*x).clamp(-mr, mr);
                }
            }
            c.shape = match c.shape {
                Shape::ArrayScalar(s) => Shape::ArrayScalar(s.map(|x| x.clamp(-mr, mr))),
                Shape::ScalarArray(s) => Shape::ScalarArray(s.map(|x| x.clamp(-ml, ml))),
                s => s,
            };
            (op, l, r, out_p, c)
        })
    })
}

fn decimal_array(c: &Column<i128>, t: (u8, i8)) -> PrimitiveArray<Decimal128Type> {
    c.array::<Decimal128Type>()
        .with_precision_and_scale(t.0, t.1)
        .unwrap()
}

fn check_decimal<Op: BinaryOp, O: krishiv_kernels::decimal::DecimalOverflow>(
    plan: &DecimalPlan,
    l: (u8, i8),
    r: (u8, i8),
    c: &Case<i128>,
) {
    let (la, ra) = (decimal_array(&c.l, l), decimal_array(&c.r, r));
    let run = |f: &dyn Fn(DecOperand<'_>, DecOperand<'_>) -> DecResult| match c.shape {
        Shape::ArrayArray => f(Operand::Array(&la), Operand::Array(&ra)),
        Shape::ArrayScalar(s) => f(Operand::Array(&la), Operand::Scalar(s)),
        Shape::ScalarArray(s) => f(Operand::Scalar(s), Operand::Array(&ra)),
    };
    let slow = run(&|a, b| reference::decimal_binary(plan, O::MODE, a, b));
    let mut storages = vec![DecimalStorage::I128];
    if plan.storage() == DecimalStorage::I64 {
        storages.push(DecimalStorage::I64);
    }
    for s in storages {
        let p = plan.with_storage(s).unwrap();
        for ctx in contexts() {
            let fast = run(&|a, b| decimal_binary::<Op, O>(a, b, &p, &ctx));
            if O::MODE == OverflowMode::Unchecked {
                assert_eq!(fast.unwrap(), slow.clone().unwrap());
            } else {
                assert_same(&fast, &slow);
            }
        }
    }
}

fn check_decimal_op<Op: BinaryOp>(plan: &DecimalPlan, l: (u8, i8), r: (u8, i8), c: &Case<i128>) {
    check_decimal::<Op, NullOnOverflow>(plan, l, r, c);
    check_decimal::<Op, AnsiError>(plan, l, r, c);
    if plan.overflow_impossible() {
        check_decimal::<Op, Unchecked>(plan, l, r, c);
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn f64_compare(c in case(f64_value())) {
        check_cmp_all::<Float64Type>(&c);
    }

    #[test]
    fn f32_compare(c in case(f32_value())) {
        check_cmp_all::<Float32Type>(&c);
    }

    #[test]
    fn decimal_arith((op, l, r, out_p, c) in decimal_case()) {
        let (ls, rs) = (l.1, r.1);
        let os = match op {
            ArithOp::Multiply => ls + rs,
            _ => ls.max(rs),
        };
        prop_assume!(os <= 38);
        let out = (out_p.max(u8::try_from(os).unwrap()).max(1), os);
        let plan = DecimalPlan::new(op, l, r, out).unwrap();
        match op {
            ArithOp::Add => check_decimal_op::<Add>(&plan, l, r, &c),
            ArithOp::Subtract => check_decimal_op::<Sub>(&plan, l, r, &c),
            ArithOp::Multiply => check_decimal_op::<Mul>(&plan, l, r, &c),
        }
    }
}
