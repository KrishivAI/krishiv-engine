// SPDX-License-Identifier: Apache-2.0
//! Decimal `+ - *` on `Decimal128` arrays, with the 64-bit storage hook.
//!
//! A [`DecimalPlan`] fixes the input and result types once per expression.
//! Inputs are rescaled to the result scale, combined, and range-checked
//! against the result precision, matching Spark's
//! `checkDecimalOverflow(... toPrecision(p, s, ROUND_HALF_UP, ...))` for
//! plans that need no rounding. Plans that would reduce the result scale
//! (Spark's precision-loss adjustment, which rounds) are not supported yet.
//!
//! **64-bit hook (ADR 0009):** when every precision is at most 18, values fit
//! in an `i64`, so the loop computes on the low 64 bits of the `Decimal128`
//! lanes ([`DecimalStorage::I64`]) — cheaper multiplies and range checks —
//! and widens the result back. Arrays stay `Decimal128` between operators.

use arrow_array::types::Decimal128Type;
use arrow_array::{Array, Decimal128Array, PrimitiveArray};
use arrow_buffer::{ScalarBuffer, i256};

use crate::chunk::{self, LaneOp, Resolved, Side};
use crate::dispatch::KernelCtx;
use crate::error::{KernelError, zeroed};
use crate::ops::{ArithOp, BinaryOp, IntNative, Operand};
use crate::policy::{AnsiError, NullOnOverflow, Overflow, OverflowMode, Unchecked};

/// Spark's maximum decimal precision.
pub const MAX_PRECISION: u8 = 38;
/// Largest precision whose values fit in an `i64`.
pub const MAX_I64_PRECISION: u8 = 18;
/// Spark's `Decimal.MATH_CONTEXT` precision for products (`MAX_PRECISION + 1`).
const PRODUCT_DIGITS: usize = 39;

mod sealed {
    pub trait Sealed {}
}

/// Overflow policies that apply to decimals (Java wrapping does not).
pub trait DecimalOverflow: sealed::Sealed + Overflow {}
impl sealed::Sealed for Unchecked {}
impl sealed::Sealed for NullOnOverflow {}
impl sealed::Sealed for AnsiError {}
impl DecimalOverflow for Unchecked {}
impl DecimalOverflow for NullOnOverflow {}
impl DecimalOverflow for AnsiError {}

/// How the kernel computes a decimal plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimalStorage {
    /// On the low 64 bits of each lane (every precision at most 18).
    I64,
    /// On the full 128-bit lanes.
    I128,
}

/// A decimal type `(precision, scale)`.
pub type DecimalType = (u8, i8);

/// The types of one decimal arithmetic expression, validated once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalPlan {
    op: ArithOp,
    left: DecimalType,
    right: DecimalType,
    out: DecimalType,
    l_mul: i128,
    r_mul: i128,
    storage: DecimalStorage,
}

/// `10^exp` for `exp <= 38` (larger exponents saturate at `10^38`).
fn pow10(exp: u8) -> i128 {
    const POW10: [i128; 39] = {
        let mut t = [1_i128; 39];
        let mut i = 1;
        while i < 39 {
            t[i] = t[i - 1].wrapping_mul(10);
            i += 1;
        }
        t
    };
    POW10[usize::from(exp.min(MAX_PRECISION))]
}

/// `t` as an unsigned scale, rejecting invalid and negative-scale types.
fn check_type(t: DecimalType) -> Result<u8, KernelError> {
    let (p, s) = t;
    if !(1..=MAX_PRECISION).contains(&p) {
        return Err(KernelError::internal(format!(
            "invalid decimal precision {p}"
        )));
    }
    let s = u8::try_from(s)
        .map_err(|_| KernelError::unsupported(format!("decimal with negative scale {s}")))?;
    if s > p {
        return Err(KernelError::internal(format!(
            "decimal scale {s} exceeds precision {p}"
        )));
    }
    Ok(s)
}

impl DecimalPlan {
    /// Plans `left op right -> out`. The result scale must be Spark's exact
    /// one (`max(s1, s2)` for `+`/`-`, `s1 + s2` for `*`); the result
    /// precision is the caller's (Spark's result type, possibly capped).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an invalid decimal type; `KRISHIV_UNSUPPORTED_FEATURE`
    /// for a negative scale or a result scale that needs rounding.
    pub fn new(
        op: ArithOp,
        left: DecimalType,
        right: DecimalType,
        out: DecimalType,
    ) -> Result<Self, KernelError> {
        let ls = check_type(left)?;
        let rs = check_type(right)?;
        let os = check_type(out)?;
        let exact = match op {
            ArithOp::Add | ArithOp::Subtract => Some(ls.max(rs)),
            ArithOp::Multiply => ls.checked_add(rs),
        };
        if exact != Some(os) {
            return Err(KernelError::unsupported(format!(
                "decimal {op:?} with result scale {os} (needs rounding from the exact scale)"
            )));
        }
        // Scales are at most 38 (checked above), so the factors fit.
        let (l_mul, r_mul) = match op {
            ArithOp::Multiply => (1, 1),
            _ => (pow10(os.saturating_sub(ls)), pow10(os.saturating_sub(rs))),
        };
        let storage = if [left.0, right.0, out.0]
            .iter()
            .all(|&p| p <= MAX_I64_PRECISION)
        {
            DecimalStorage::I64
        } else {
            DecimalStorage::I128
        };
        Ok(Self {
            op,
            left,
            right,
            out,
            l_mul,
            r_mul,
            storage,
        })
    }

    /// The storage the kernel computes in.
    #[must_use]
    pub fn storage(&self) -> DecimalStorage {
        self.storage
    }

    /// The result type.
    #[must_use]
    pub fn out(&self) -> DecimalType {
        self.out
    }

    /// The same plan computed in `storage` (tests and benchmarks compare the
    /// two paths).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `I64` is asked for a plan with a precision above 18.
    pub fn with_storage(mut self, storage: DecimalStorage) -> Result<Self, KernelError> {
        if storage == DecimalStorage::I64 && self.storage != DecimalStorage::I64 {
            return Err(KernelError::internal(
                "decimal I64 storage needs every precision <= 18",
            ));
        }
        self.storage = storage;
        Ok(self)
    }

    /// Whether the types alone prove the result always fits the result
    /// precision, which is what licenses [`Unchecked`] (decision 3 of the
    /// kernel framework: no runtime range proofs).
    #[must_use]
    pub fn overflow_impossible(&self) -> bool {
        let int_digits = |(p, s): DecimalType| i16::from(p).saturating_sub(i16::from(s));
        let needed = match self.op {
            ArithOp::Add | ArithOp::Subtract => int_digits(self.left)
                .max(int_digits(self.right))
                .saturating_add(1)
                .saturating_add(i16::from(self.out.1)),
            ArithOp::Multiply => i16::from(self.left.0).saturating_add(i16::from(self.right.0)),
        };
        needed <= i16::from(self.out.0)
    }

    /// `(op, left, right, out)`.
    pub(crate) fn types(&self) -> (ArithOp, DecimalType, DecimalType, DecimalType) {
        (self.op, self.left, self.right, self.out)
    }

    /// The exact result of one row (no overflow is possible in 256 bits).
    fn exact(&self, a: i128, b: i128) -> i256 {
        let (a, b) = (i256::from_i128(a), i256::from_i128(b));
        let (la, rb) = (
            a.wrapping_mul(i256::from_i128(self.l_mul)),
            b.wrapping_mul(i256::from_i128(self.r_mul)),
        );
        match self.op {
            ArithOp::Add => la.wrapping_add(rb),
            ArithOp::Subtract => la.wrapping_sub(rb),
            ArithOp::Multiply => la.wrapping_mul(rb),
        }
    }

    fn out_of_range(&self, v: i256) -> bool {
        let bound = i256::from_i128(pow10(self.out.0));
        v >= bound || v <= bound.wrapping_neg()
    }

    /// Spark's `toPlainString` of the value its `Decimal` arithmetic produced
    /// before the range check: exact for `+`/`-`; for `*`, rounded down to
    /// 39 significant digits (`Decimal.MATH_CONTEXT`), with a negative
    /// scale normalized to 0.
    pub(crate) fn spark_value(&self, v: i256) -> String {
        let text = v.to_string();
        let (sign, digits) = text
            .strip_prefix('-')
            .map_or(("", text.as_str()), |d| ("-", d));
        let mut digits = digits.to_owned();
        let mut scale = usize::from(self.out.1.unsigned_abs());
        if self.op == ArithOp::Multiply && digits.len() > PRODUCT_DIGITS {
            let dropped = digits.len().saturating_sub(PRODUCT_DIGITS);
            digits.truncate(PRODUCT_DIGITS);
            if dropped > scale {
                digits.push_str(&"0".repeat(dropped.saturating_sub(scale)));
                scale = 0;
            } else {
                scale = scale.saturating_sub(dropped);
            }
        }
        if scale == 0 {
            return format!("{sign}{digits}");
        }
        if digits.len() <= scale {
            let zeros = "0".repeat(scale.saturating_sub(digits.len()));
            return format!("{sign}0.{zeros}{digits}");
        }
        let (int, frac) = digits.split_at(digits.len().saturating_sub(scale));
        format!("{sign}{int}.{frac}")
    }
}

/// 128-bit lanes.
#[derive(Clone, Copy)]
struct Lane128<'p, Op> {
    plan: &'p DecimalPlan,
    bound: i128,
    op: std::marker::PhantomData<Op>,
}

impl<Op: BinaryOp> LaneOp<i128> for Lane128<'_, Op> {
    #[inline(always)]
    fn wrapping(self, a: i128, b: i128) -> i128 {
        Op::wrapping(
            a.wrapping_mul(self.plan.l_mul),
            b.wrapping_mul(self.plan.r_mul),
        )
    }
    #[inline(always)]
    fn checked(self, a: i128, b: i128) -> (i128, bool) {
        let (a, fa) = a.overflowing_mul(self.plan.l_mul);
        let (b, fb) = b.overflowing_mul(self.plan.r_mul);
        let (v, fv) = Op::overflowing(a, b);
        let bad = fa | fb | fv | (v >= self.bound) | (v <= self.bound.wrapping_neg());
        (v, bad)
    }
    #[inline(always)]
    fn is_bad(self, a: i128, b: i128) -> bool {
        self.plan.out_of_range(self.plan.exact(a, b))
    }
}

/// The 64-bit hook: computes on the low 64 bits of 128-bit lanes. A lane
/// that does not fit in 64 bits (an input outside its declared precision)
/// is flagged and settled exactly by `is_bad`. Truncation to 64 bits
/// commutes with wrapping `+ - *`, so whenever the exact result is in range
/// (hence fits an `i64`) the wrapped value is exact, as [`LaneOp`] requires;
/// the same argument modulo 2^128 covers [`Lane128`].
#[derive(Clone, Copy)]
struct Lane64<'p, Op> {
    plan: &'p DecimalPlan,
    l_mul: i64,
    r_mul: i64,
    bound: i64,
    op: std::marker::PhantomData<Op>,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "taking the low 64 bits is the point of the hook; lanes that do not fit are flagged"
)]
#[inline(always)]
fn low64(v: i128) -> (i64, bool) {
    let low = v as i64;
    (low, i128::from(low) != v)
}

impl<Op: BinaryOp> LaneOp<i128> for Lane64<'_, Op> {
    #[inline(always)]
    fn wrapping(self, a: i128, b: i128) -> i128 {
        let (a, b) = (low64(a).0, low64(b).0);
        i128::from(Op::wrapping(
            a.wrapping_mul(self.l_mul),
            b.wrapping_mul(self.r_mul),
        ))
    }
    #[inline(always)]
    fn checked(self, a: i128, b: i128) -> (i128, bool) {
        let ((a, wa), (b, wb)) = (low64(a), low64(b));
        let (a, fa) = IntNative::overflowing_mul(a, self.l_mul);
        let (b, fb) = IntNative::overflowing_mul(b, self.r_mul);
        let (v, fv) = Op::overflowing(a, b);
        let bad = wa | wb | fa | fb | fv | (v >= self.bound) | (v <= self.bound.wrapping_neg());
        (i128::from(v), bad)
    }
    #[inline(always)]
    fn is_bad(self, a: i128, b: i128) -> bool {
        self.plan.out_of_range(self.plan.exact(a, b))
    }
}

/// The `(precision, scale)` of a decimal array.
fn type_of(a: &Decimal128Array) -> DecimalType {
    (a.precision(), a.scale())
}

/// `l op r` under `plan` with overflow policy `O`.
///
/// # Errors
/// - `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION` (with [`KernelError::row`])
///   under `AnsiError`;
/// - `INTERNAL_ERROR` when `Op`, an array type or the policy does not match
///   the plan (`Unchecked` needs [`DecimalPlan::overflow_impossible`]), for
///   two scalars or arrays of different lengths;
/// - `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn decimal_binary<Op, O>(
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
    plan: &DecimalPlan,
    ctx: &KernelCtx,
) -> Result<Decimal128Array, KernelError>
where
    Op: BinaryOp,
    O: DecimalOverflow,
{
    if Op::OP != plan.op {
        return Err(KernelError::internal(format!(
            "decimal kernel {:?} called with a {:?} plan",
            Op::OP,
            plan.op
        )));
    }
    if O::MODE == OverflowMode::Unchecked && !plan.overflow_impossible() {
        return Err(KernelError::internal(
            "Unchecked decimal arithmetic on types that can overflow",
        ));
    }
    for (o, want) in [(l, plan.left), (r, plan.right)] {
        if let Operand::Array(a) = o
            && type_of(a) != want
        {
            return Err(KernelError::internal(format!(
                "decimal operand is {:?}, plan expects {want:?}",
                a.data_type()
            )));
        }
    }
    let (sl, sr, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return finish(PrimitiveArray::new_null(len), plan),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let out = zeroed(len)?;
    let computed = match plan.storage {
        DecimalStorage::I128 => {
            let bound = pow10(plan.out.0);
            let lane = Lane128::<Op> {
                plan,
                bound,
                op: std::marker::PhantomData,
            };
            chunk::compute::<i128, _, O>(ctx, sl, sr, nulls, out, lane)
        }
        DecimalStorage::I64 => {
            let narrow = |v: i128| i64::try_from(v).unwrap_or(i64::MAX);
            let lane = Lane64::<Op> {
                plan,
                l_mul: narrow(plan.l_mul),
                r_mul: narrow(plan.r_mul),
                bound: narrow(pow10(plan.out.0)),
                op: std::marker::PhantomData,
            };
            chunk::compute::<i128, _, O>(ctx, sl, sr, nulls, out, lane)
        }
    };
    match computed {
        Ok(c) => finish(
            PrimitiveArray::try_new(ScalarBuffer::from(c.values), c.nulls)
                .map_err(|e| KernelError::internal(e.to_string()))?,
            plan,
        ),
        Err(row) => {
            let at = |s: Side<'_, i128>| match s {
                Side::Array(v) => v.get(row).copied().unwrap_or_default(),
                Side::Scalar(v) => v,
            };
            let value = plan.spark_value(plan.exact(at(sl), at(sr)));
            Err(KernelError::decimal_out_of_range(
                row, value, plan.out.0, plan.out.1,
            ))
        }
    }
}

fn finish(a: Decimal128Array, plan: &DecimalPlan) -> Result<Decimal128Array, KernelError> {
    a.with_precision_and_scale(plan.out.0, plan.out.1)
        .map_err(|e| KernelError::internal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(op: ArithOp, l: DecimalType, r: DecimalType, o: DecimalType) -> DecimalPlan {
        DecimalPlan::new(op, l, r, o).unwrap()
    }

    #[test]
    fn plan_validation() {
        let p = plan(ArithOp::Add, (10, 2), (5, 4), (13, 4));
        assert_eq!((p.l_mul, p.r_mul), (100, 1));
        assert_eq!(p.storage(), DecimalStorage::I64);
        assert_eq!(p.out(), (13, 4));
        assert!(p.overflow_impossible());
        assert!(!plan(ArithOp::Add, (10, 2), (5, 4), (12, 4)).overflow_impossible());
        assert!(plan(ArithOp::Multiply, (10, 2), (5, 4), (15, 6)).overflow_impossible());
        assert!(!plan(ArithOp::Multiply, (38, 2), (5, 4), (38, 6)).overflow_impossible());
        assert_eq!(
            plan(ArithOp::Subtract, (19, 0), (5, 0), (20, 0)).storage(),
            DecimalStorage::I128
        );
        let i128p = p.with_storage(DecimalStorage::I128).unwrap();
        assert_eq!(i128p.storage(), DecimalStorage::I128);
        let big = plan(ArithOp::Add, (30, 0), (5, 0), (31, 0));
        assert!(big.with_storage(DecimalStorage::I64).is_err());
        use krishiv_common::error::ErrorClass as C;
        let class = |r: Result<DecimalPlan, KernelError>| r.unwrap_err().error_class();
        assert_eq!(
            class(DecimalPlan::new(ArithOp::Add, (0, 0), (1, 0), (2, 0))),
            C::InternalError
        );
        assert_eq!(
            class(DecimalPlan::new(ArithOp::Add, (39, 0), (1, 0), (2, 0))),
            C::InternalError
        );
        assert_eq!(
            class(DecimalPlan::new(ArithOp::Add, (5, 6), (1, 0), (2, 0))),
            C::InternalError
        );
        assert_eq!(
            class(DecimalPlan::new(ArithOp::Add, (5, -1), (1, 0), (2, 0))),
            C::KrishivUnsupportedFeature
        );
        assert_eq!(
            class(DecimalPlan::new(ArithOp::Add, (5, 1), (1, 0), (2, 0))),
            C::KrishivUnsupportedFeature
        );
        assert_eq!(
            class(DecimalPlan::new(
                ArithOp::Multiply,
                (38, 20),
                (38, 20),
                (38, 38)
            )),
            C::KrishivUnsupportedFeature
        );
    }

    #[test]
    fn spark_plain_string() {
        let p = plan(ArithOp::Add, (5, 2), (5, 2), (6, 2));
        let s = |v: i128| p.spark_value(i256::from_i128(v));
        assert_eq!(s(12_345), "123.45");
        assert_eq!(s(-5), "-0.05");
        assert_eq!(s(100), "1.00");
        let p0 = plan(ArithOp::Add, (5, 0), (5, 0), (6, 0));
        assert_eq!(p0.spark_value(i256::from_i128(-42)), "-42");
        // Products keep 39 significant digits, rounded towards zero.
        let m = plan(ArithOp::Multiply, (38, 1), (38, 1), (38, 2));
        let v = i256::from_i128(10_i128.pow(38)).wrapping_mul(i256::from_i128(123_456));
        assert_eq!(m.spark_value(v), format!("123456{}", "0".repeat(36)));
        assert_eq!(
            m.spark_value(v.wrapping_neg()),
            format!("-123456{}", "0".repeat(36))
        );
        let w = i256::from_i128(10_i128.pow(38)).wrapping_mul(i256::from_i128(7));
        assert_eq!(m.spark_value(w), format!("7{}.00", "0".repeat(36)));
        let mm = plan(ArithOp::Multiply, (38, 19), (20, 19), (38, 38));
        let x = i256::from_i128(10_i128.pow(38).wrapping_add(1)).wrapping_mul(i256::from_i128(100));
        assert_eq!(mm.spark_value(x), format!("100.{}1", "0".repeat(35)));
    }

    #[test]
    fn mismatched_plan_is_an_engine_error() {
        let a = Decimal128Array::from(vec![1_i128])
            .with_precision_and_scale(5, 0)
            .unwrap();
        let p = plan(ArithOp::Add, (5, 0), (5, 0), (6, 0));
        let ctx = KernelCtx::new();
        let e = decimal_binary::<crate::Mul, AnsiError>(
            Operand::Array(&a),
            Operand::Array(&a),
            &p,
            &ctx,
        )
        .unwrap_err();
        assert_eq!(e.row(), None);
        let narrow = plan(ArithOp::Add, (5, 0), (5, 0), (5, 0));
        assert!(
            decimal_binary::<crate::Add, Unchecked>(
                Operand::Array(&a),
                Operand::Array(&a),
                &narrow,
                &ctx
            )
            .is_err()
        );
        let other = plan(ArithOp::Add, (6, 0), (5, 0), (7, 0));
        assert!(
            decimal_binary::<crate::Add, AnsiError>(
                Operand::Array(&a),
                Operand::Array(&a),
                &other,
                &ctx
            )
            .is_err()
        );
    }
}
