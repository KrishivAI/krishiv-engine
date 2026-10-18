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
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, PrimitiveArray};
use arrow_buffer::{NullBuffer, ScalarBuffer, ToByteSlice};
use krishiv_kernels::compare::{
    self, BoolOperand, BytesOperand, CompareKind, CompareOp, Equal, Greater, GreaterEq, Less,
    LessEq, NotEqual, SparkOrd, Validity,
};
use krishiv_kernels::decimal::{DecimalPlan, DecimalStorage, decimal_binary};
use krishiv_kernels::float::{self, SparkFloat};
use krishiv_kernels::integer::{SparkIntegral, UnaryOp, binary, integral_divide, remainder, unary};
use krishiv_kernels::ops::BinaryOp;
use krishiv_kernels::reference;
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

/// A reference comparison.
type CmpRef<P> =
    dyn Fn(CompareKind, Operand<'_, P>, Operand<'_, P>) -> Result<BooleanArray, KernelError>;

/// Two optional byte strings.
type BytesRow = (Option<Vec<u8>>, Option<Vec<u8>>);

/// Checks one comparison and, for `=`, the null-safe `<=>` built on it.
fn check_cmp<P, Op>(c: &Case<P::Native>, slow: &CmpRef<P>)
where
    P: ArrowPrimitiveType<Native: SparkOrd>,
    Op: CompareOp,
{
    let expect = c.with::<P, _>(|l, r| slow(Op::KIND, l, r)).unwrap();
    for ctx in contexts() {
        let fast = c.with::<P, _>(|l, r| compare::compare::<P, Op>(l, r, &ctx));
        assert_eq!(fast.unwrap(), expect);
    }
    if Op::KIND == CompareKind::Equal {
        let (la, ra) = (c.l.array::<P>(), c.r.array::<P>());
        let eq = c.with::<P, _>(|l, r| compare::compare::<P, Equal>(l, r, &KernelCtx::new()));
        let rows = |a: &PrimitiveArray<P>| {
            (0..a.len())
                .map(|i| a.is_valid(i).then(|| a.value(i)))
                .collect::<Vec<_>>()
        };
        let (lv, rv, ls, rs) = match c.shape {
            Shape::ArrayArray => (
                rows(&la),
                rows(&ra),
                Validity::Array(la.nulls()),
                Validity::Array(ra.nulls()),
            ),
            Shape::ArrayScalar(v) => (
                rows(&la),
                vec![v; la.len()],
                Validity::Array(la.nulls()),
                Validity::Scalar { null: v.is_none() },
            ),
            Shape::ScalarArray(v) => (
                vec![v; ra.len()],
                rows(&ra),
                Validity::Scalar { null: v.is_none() },
                Validity::Array(ra.nulls()),
            ),
        };
        // `<=>` equality of present values follows Spark's order too.
        let expect: BooleanArray = lv
            .iter()
            .zip(&rv)
            .map(|(a, b)| {
                Some(match (a, b) {
                    (Some(a), Some(b)) => P::Native::spark_eq(*a, *b),
                    (None, None) => true,
                    _ => false,
                })
            })
            .collect();
        assert_eq!(compare::null_safe(&eq.unwrap(), ls, rs).unwrap(), expect);
    }
}

fn check_cmp_all<P: ArrowPrimitiveType<Native: SparkOrd>>(c: &Case<P::Native>, slow: &CmpRef<P>) {
    check_cmp::<P, Equal>(c, slow);
    check_cmp::<P, NotEqual>(c, slow);
    check_cmp::<P, Less>(c, slow);
    check_cmp::<P, LessEq>(c, slow);
    check_cmp::<P, Greater>(c, slow);
    check_cmp::<P, GreaterEq>(c, slow);
}

fn check_float_arith<P: ArrowPrimitiveType<Native: SparkFloat>>(c: &Case<P::Native>) {
    let same = |fast: Result<PrimitiveArray<P>, KernelError>,
                slow: Result<PrimitiveArray<P>, KernelError>| {
        match (fast, slow) {
            // Compare bit patterns (-0.0 included); any NaN equals any NaN
            // (IEEE leaves NaN payloads unspecified and Spark never shows them).
            (Ok(f), Ok(s)) => {
                let nulls =
                    |a: &PrimitiveArray<P>| (0..a.len()).map(|i| a.is_null(i)).collect::<Vec<_>>();
                assert_eq!(nulls(&f), nulls(&s));
                for i in (0..f.len()).filter(|&i| f.is_valid(i)) {
                    let (a, b) = (f.value(i), s.value(i));
                    if !(a.is_nan() && b.is_nan()) {
                        assert_eq!(a.to_byte_slice(), b.to_byte_slice(), "row {i}");
                    }
                }
            }
            (f, s) => assert_same(&f, &s),
        }
    };
    for ctx in version_contexts() {
        let v = ctx.spark_version();
        for op in [ArithOp::Add, ArithOp::Subtract, ArithOp::Multiply] {
            let slow = c.with::<P, _>(|l, r| {
                reference::float_arith(Some(op), false, OverflowMode::AnsiError, v, l, r)
            });
            same(c.with::<P, _>(|l, r| float::binary(op, l, r, &ctx)), slow);
        }
        for rem in [false, true] {
            for mode in [OverflowMode::JavaWrap, OverflowMode::AnsiError] {
                let slow = c.with::<P, _>(|l, r| reference::float_arith(None, rem, mode, v, l, r));
                let fast = c.with::<P, _>(|l, r| match (rem, mode) {
                    (false, OverflowMode::JavaWrap) => float::divide::<P, JavaWrap>(l, r, &ctx),
                    (false, _) => float::divide::<P, AnsiError>(l, r, &ctx),
                    (true, OverflowMode::JavaWrap) => float::remainder::<P, JavaWrap>(l, r, &ctx),
                    (true, _) => float::remainder::<P, AnsiError>(l, r, &ctx),
                });
                same(fast, slow);
            }
        }
    }
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
        check_cmp_all::<Float64Type>(&c, &reference::compare);
        check_float_arith::<Float64Type>(&c);
    }

    #[test]
    fn i32_compare(c in case(i32_value())) {
        check_cmp_all::<Int32Type>(&c, &reference::compare_ord);
    }

    #[test]
    fn i64_compare(c in case(i64_value())) {
        check_cmp_all::<Int64Type>(&c, &reference::compare_ord);
    }

    #[test]
    fn i128_compare(c in case(any::<i128>())) {
        check_cmp_all::<Decimal128Type>(&c, &reference::compare_ord);
    }

    #[test]
    fn bytes_compare(
        rows in proptest::collection::vec(
            (proptest::option::weighted(0.8, proptest::collection::vec(any::<u8>(), 0..4)),
             proptest::option::weighted(0.8, proptest::collection::vec(any::<u8>(), 0..4))),
            0..150,
        ),
        scalar in proptest::option::weighted(0.9, proptest::collection::vec(any::<u8>(), 0..4)),
    ) {
        check_bytes(&rows, scalar.as_deref());
    }

    #[test]
    fn bool_compare(
        rows in proptest::collection::vec(
            (proptest::option::weighted(0.8, any::<bool>()), proptest::option::weighted(0.8, any::<bool>())),
            0..150,
        ),
        scalar in proptest::option::weighted(0.9, any::<bool>()),
    ) {
        check_bool(&rows, scalar);
    }

    #[test]
    fn f32_compare(c in case(f32_value())) {
        check_cmp_all::<Float32Type>(&c, &reference::compare);
        check_float_arith::<Float32Type>(&c);
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

const KINDS: [CompareKind; 6] = [
    CompareKind::Equal,
    CompareKind::NotEqual,
    CompareKind::Less,
    CompareKind::LessEq,
    CompareKind::Greater,
    CompareKind::GreaterEq,
];

/// Strings and binary against byte-slice order, array/array and
/// array/scalar, plus `<=>`.
fn check_bytes(rows: &[BytesRow], scalar: Option<&[u8]>) {
    use arrow_array::BinaryArray;
    let l: Vec<Option<&[u8]>> = rows.iter().map(|r| r.0.as_deref()).collect();
    let r: Vec<Option<&[u8]>> = rows.iter().map(|r| r.1.as_deref()).collect();
    let (la, ra) = (BinaryArray::from(l.clone()), BinaryArray::from(r.clone()));
    let s = vec![scalar; l.len()];
    for kind in KINDS {
        macro_rules! run {
            ($op:ty) => {
                (
                    compare::compare_bytes::<_, $op>(
                        BytesOperand::Array(&la),
                        BytesOperand::Array(&ra),
                    )
                    .unwrap(),
                    compare::compare_bytes::<_, $op>(
                        BytesOperand::Array(&la),
                        BytesOperand::Scalar(scalar),
                    )
                    .unwrap(),
                )
            };
        }
        let (aa, asc) = match kind {
            CompareKind::Equal => run!(Equal),
            CompareKind::NotEqual => run!(NotEqual),
            CompareKind::Less => run!(Less),
            CompareKind::LessEq => run!(LessEq),
            CompareKind::Greater => run!(Greater),
            CompareKind::GreaterEq => run!(GreaterEq),
        };
        assert_eq!(aa, reference::compare_bytes(kind, &l, &r));
        assert_eq!(asc, reference::compare_bytes(kind, &l, &s));
    }
    let eq = compare::compare_bytes::<_, Equal>(BytesOperand::Array(&la), BytesOperand::Array(&ra))
        .unwrap();
    let ns = compare::null_safe(
        &eq,
        Validity::Array(la.nulls()),
        Validity::Array(ra.nulls()),
    );
    assert_eq!(ns.unwrap(), reference::null_safe_equal(&l, &r));
}

/// Booleans against `false < true`, array/array and array/scalar.
fn check_bool(rows: &[(Option<bool>, Option<bool>)], scalar: Option<bool>) {
    let l: Vec<Option<bool>> = rows.iter().map(|r| r.0).collect();
    let r: Vec<Option<bool>> = rows.iter().map(|r| r.1).collect();
    let (la, ra) = (BooleanArray::from(l.clone()), BooleanArray::from(r.clone()));
    let expect = |kind: CompareKind, rhs: &[Option<bool>]| -> BooleanArray {
        l.iter()
            .zip(rhs)
            .map(|(a, b)| Some(kind.holds((*a)?.cmp(&(*b)?))))
            .collect()
    };
    for kind in KINDS {
        macro_rules! run {
            ($op:ty) => {
                (
                    compare::compare_bool::<$op>(BoolOperand::Array(&la), BoolOperand::Array(&ra))
                        .unwrap(),
                    compare::compare_bool::<$op>(
                        BoolOperand::Array(&la),
                        BoolOperand::Scalar(scalar),
                    )
                    .unwrap(),
                    compare::compare_bool::<$op>(
                        BoolOperand::Scalar(scalar),
                        BoolOperand::Array(&ra),
                    )
                    .unwrap(),
                )
            };
        }
        let (aa, asc, sca) = match kind {
            CompareKind::Equal => run!(Equal),
            CompareKind::NotEqual => run!(NotEqual),
            CompareKind::Less => run!(Less),
            CompareKind::LessEq => run!(LessEq),
            CompareKind::Greater => run!(Greater),
            CompareKind::GreaterEq => run!(GreaterEq),
        };
        assert_eq!(aa, expect(kind, &r));
        assert_eq!(asc, expect(kind, &vec![scalar; l.len()]));
        let flipped: BooleanArray = r
            .iter()
            .map(|b| Some(kind.holds(scalar?.cmp(&(*b)?))))
            .collect();
        assert_eq!(sca, flipped);
    }
}
