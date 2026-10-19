// SPDX-License-Identifier: Apache-2.0
//! Decimal arithmetic on `Decimal128` arrays: `+ - * / %`, `div` and unary
//! `-`/`abs`, with Spark's result types ([`result_type`]) and the 64-bit
//! storage hook.
//!
//! A [`DecimalPlan`] fixes the input and result types once per expression.
//! Exact `+ - *` (result scale = exact scale) rescale, combine and
//! range-check on 128-bit (or 64-bit) lanes; everything that rounds
//! (Spark's precision-loss adjustment), `/` and `%` computes the exact value
//! in 256 bits and rounds `HALF_UP`, matching Spark's
//! `checkDecimalOverflow(... toPrecision(p, s, ROUND_HALF_UP, ...))`. Spark
//! truncates products to 39 digits and quotients to 39 decimals before
//! that rounding; for every result that fits its type this equals rounding
//! the exact value (the property tests check it against a reference that
//! follows Spark's steps literally).
//!
//! **64-bit hook (ADR 0009):** when every precision is at most 18, values fit
//! in an `i64`, so the loop computes on the low 64 bits of the `Decimal128`
//! lanes ([`DecimalStorage::I64`]) — cheaper multiplies and range checks —
//! and widens the result back. Arrays stay `Decimal128` between operators.
//!
//! **Per-batch range proof (P2-T160):** exact `+ - *` that the types leave
//! on checked 128-bit lanes first bound each batch's results from its
//! operands' largest magnitudes. When the bound is below `10^p` of the
//! result type, no row can overflow and the batch runs without checks;
//! when it also fits 63 bits, it runs on the 64-bit lanes whatever the
//! declared precisions (TPC-H's `price * (1 - discount) * (1 + tax)` is
//! `decimal(38,6)` by type but small by value). Otherwise the batch takes
//! the checked lanes as before. The proof only selects lanes, so results
//! and errors are those of the checked path.

use arrow_array::types::Decimal128Type;
use arrow_array::{Array, Decimal128Array, Int64Array, PrimitiveArray};
use arrow_buffer::{ScalarBuffer, i256};

use crate::chunk::{self, LaneOp, Resolved, Side};
use crate::dispatch::KernelCtx;
use crate::error::{KernelError, with_capacity};
use crate::integer::{UnaryOp, at};
use crate::ops::{ArithOp, BinaryOp, IntNative, Operand};

mod rules;
mod wide;

pub use rules::result_type;

/// `n / d` truncated to `scale` decimals, as plain text (the reference's
/// view of Spark's quotient).
pub(crate) fn quotient_text(n: i256, d: i256, scale: u32) -> String {
    wide::quotient_string(n, d, scale)
}
use crate::policy::{AnsiError, Fault, NullOnOverflow, Overflow, OverflowMode, Unchecked};

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

/// A decimal arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimalOp {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `/`
    Divide,
    /// `%`
    Remainder,
    /// `div` (a `bigint` result; see [`decimal_integral_divide`]).
    IntegralDivide,
}

impl From<ArithOp> for DecimalOp {
    fn from(op: ArithOp) -> Self {
        match op {
            ArithOp::Add => Self::Add,
            ArithOp::Subtract => Self::Subtract,
            ArithOp::Multiply => Self::Multiply,
        }
    }
}

/// The types of one decimal arithmetic expression, validated once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalPlan {
    op: DecimalOp,
    left: DecimalType,
    right: DecimalType,
    out: DecimalType,
    /// The scale of the exact result (`max(s1, s2)` for `+ - %`, `s1 + s2`
    /// for `*`; the result scale for `/`).
    exact_scale: u8,
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
    /// Plans `left op right -> out` (`out` is normally [`result_type`]).
    /// `+ - *` may round to a result scale below the exact one (Spark's
    /// precision-loss adjustment); `%` keeps the exact scale; `/` takes any
    /// result scale. Results round `HALF_UP`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an invalid decimal type, `div` (it has no
    /// decimal result) or a result scale above the exact one (or, for `%`,
    /// different from it); `KRISHIV_UNSUPPORTED_FEATURE` for a negative scale.
    pub fn new(
        op: impl Into<DecimalOp>,
        left: DecimalType,
        right: DecimalType,
        out: DecimalType,
    ) -> Result<Self, KernelError> {
        let op = op.into();
        let ls = check_type(left)?;
        let rs = check_type(right)?;
        let os = check_type(out)?;
        let exact_scale = match op {
            DecimalOp::Add | DecimalOp::Subtract | DecimalOp::Remainder => ls.max(rs),
            // Scales are at most 38, so the sum fits.
            DecimalOp::Multiply => ls.saturating_add(rs),
            DecimalOp::Divide => os,
            DecimalOp::IntegralDivide => {
                return Err(KernelError::internal(
                    "decimal div has no decimal result type",
                ));
            }
        };
        if os > exact_scale || (op == DecimalOp::Remainder && os != exact_scale) {
            return Err(KernelError::internal(format!(
                "decimal {op:?} of scales {ls} and {rs} cannot have result scale {os}"
            )));
        }
        let (l_mul, r_mul) = match op {
            DecimalOp::Add | DecimalOp::Subtract | DecimalOp::Remainder => (
                pow10(exact_scale.saturating_sub(ls)),
                pow10(exact_scale.saturating_sub(rs)),
            ),
            _ => (1, 1),
        };
        let fast = os == exact_scale
            && matches!(
                op,
                DecimalOp::Add | DecimalOp::Subtract | DecimalOp::Multiply
            );
        let storage = if fast
            && [left.0, right.0, out.0]
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
            exact_scale,
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

    /// Whether the plan runs on the 128/64-bit lanes (exact `+ - *`) rather
    /// than the 256-bit path (rounding, `/`, `%`).
    fn is_fast(&self) -> bool {
        u8::try_from(self.out.1) == Ok(self.exact_scale)
            && matches!(
                self.op,
                DecimalOp::Add | DecimalOp::Subtract | DecimalOp::Multiply
            )
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
    /// kernel framework: no runtime range proofs). A zero divisor is not an
    /// overflow: `Unchecked` still raises it.
    #[must_use]
    pub fn overflow_impossible(&self) -> bool {
        let int_digits = |(p, s): DecimalType| i16::from(p).saturating_sub(i16::from(s));
        let needed = match self.op {
            DecimalOp::Add | DecimalOp::Subtract => int_digits(self.left)
                .max(int_digits(self.right))
                .saturating_add(1)
                .saturating_add(i16::from(self.out.1)),
            // Rounding a product down to the result scale can carry into a
            // new digit, so only exact products are proven.
            DecimalOp::Multiply => i16::from(self.left.0)
                .saturating_add(i16::from(self.right.0))
                .saturating_add(i16::from(self.exact_scale).saturating_sub(i16::from(self.out.1))),
            // |a % b| < |b|.
            DecimalOp::Remainder => int_digits(self.left)
                .min(int_digits(self.right))
                .saturating_add(i16::from(self.out.1)),
            DecimalOp::Divide | DecimalOp::IntegralDivide => i16::MAX,
        };
        needed <= i16::from(self.out.0)
    }

    /// The operator.
    #[must_use]
    pub fn types_op(&self) -> DecimalOp {
        self.op
    }

    /// `(op, left, right, out)`.
    pub(crate) fn types(&self) -> (DecimalOp, DecimalType, DecimalType, DecimalType) {
        (self.op, self.left, self.right, self.out)
    }

    /// The exact result of `+ - *` at the exact scale (no overflow is
    /// possible in 256 bits).
    fn exact(&self, a: i128, b: i128) -> i256 {
        let (a, b) = (i256::from_i128(a), i256::from_i128(b));
        let (la, rb) = (
            a.wrapping_mul(i256::from_i128(self.l_mul)),
            b.wrapping_mul(i256::from_i128(self.r_mul)),
        );
        match self.op {
            DecimalOp::Subtract => la.wrapping_sub(rb),
            DecimalOp::Multiply => la.wrapping_mul(rb),
            _ => la.wrapping_add(rb),
        }
    }

    /// One row of any plan, exactly: the result, or why there is none.
    /// Computes in 128 bits when the rescaled operands fit (the common
    /// case), else in 256.
    fn eval(&self, a: i128, b: i128) -> Result<i128, Fault> {
        let os = u8::try_from(self.out.1).unwrap_or_default();
        if b == 0
            && matches!(
                self.op,
                DecimalOp::Remainder | DecimalOp::Divide | DecimalOp::IntegralDivide
            )
        {
            return Err(Fault::DivideByZero);
        }
        if let Some(v) = self.eval128(a, b, os) {
            return if v >= pow10(self.out.0) || v <= pow10(self.out.0).wrapping_neg() {
                Err(Fault::Overflow)
            } else {
                Ok(v)
            };
        }
        let v = match self.op {
            DecimalOp::Add | DecimalOp::Subtract | DecimalOp::Multiply => wide::round_half_up(
                self.exact(a, b),
                u32::from(self.exact_scale.saturating_sub(os)),
            ),
            DecimalOp::Remainder => {
                let (n, d) = (
                    i256::from_i128(a).wrapping_mul(i256::from_i128(self.l_mul)),
                    i256::from_i128(b).wrapping_mul(i256::from_i128(self.r_mul)),
                );
                // `BigDecimal.remainder(d, MathContext(39, DOWN))` cuts the
                // integral quotient to 39 significant digits and fails
                // ("Division impossible") only if a dropped digit is not 0.
                let q = n.wrapping_div(d).wrapping_abs();
                if wide::drops_nonzero_digits(q, u32::from(MAX_PRECISION).saturating_add(1)) {
                    return Err(Fault::Impossible);
                }
                n.wrapping_rem(d)
            }
            DecimalOp::Divide | DecimalOp::IntegralDivide => {
                let (n, d) = self.quotient_terms(a, b);
                wide::half_up_div(n, d)
            }
        };
        if self.out_of_range(v) {
            return Err(Fault::Overflow);
        }
        v.to_i128().ok_or(Fault::Overflow)
    }

    /// [`Self::eval`] in 128 bits, or `None` when an intermediate does not
    /// fit (`b != 0` for the dividing operators).
    fn eval128(&self, a: i128, b: i128, os: u8) -> Option<i128> {
        let drop = self.exact_scale.saturating_sub(os);
        match self.op {
            DecimalOp::Add | DecimalOp::Subtract | DecimalOp::Multiply => {
                let x = a.checked_mul(self.l_mul)?;
                let y = b.checked_mul(self.r_mul)?;
                let exact = match self.op {
                    DecimalOp::Add => x.checked_add(y),
                    DecimalOp::Subtract => x.checked_sub(y),
                    _ => x.checked_mul(y),
                }?;
                // drop <= 38 here only when 10^drop fits.
                (drop <= MAX_PRECISION).then(|| half_up_div128(exact, pow10(drop)))
            }
            DecimalOp::Remainder => {
                let n = a.checked_mul(self.l_mul)?;
                n.checked_rem(b.checked_mul(self.r_mul)?)
            }
            DecimalOp::Divide | DecimalOp::IntegralDivide => {
                let (s1, s2) = (self.left.1.unsigned_abs(), self.right.1.unsigned_abs());
                let up = i32::from(self.exact_scale)
                    .saturating_sub(i32::from(s1))
                    .saturating_add(i32::from(s2));
                let k = u8::try_from(up.unsigned_abs())
                    .ok()
                    .filter(|&k| k <= MAX_PRECISION)?;
                let (n, d) = if up >= 0 {
                    (a.checked_mul(pow10(k))?, b)
                } else {
                    (a, b.checked_mul(pow10(k))?)
                };
                Some(half_up_div128(n, d))
            }
        }
    }

    /// `(n, d)` with `a / b` at the result scale equal to `n / d`: Spark's
    /// quotient `(a / 10^s1) / (b / 10^s2)` scaled by `10^s`. A dividend
    /// beyond 256 bits saturates, which still overflows the result.
    fn quotient_terms(&self, a: i128, b: i128) -> (i256, i256) {
        let (s1, s2) = (self.left.1.unsigned_abs(), self.right.1.unsigned_abs());
        // n / d = a * 10^(s - s1 + s2) / b.
        let up = i32::from(self.exact_scale)
            .saturating_sub(i32::from(s1))
            .saturating_add(i32::from(s2));
        let (a, b) = (i256::from_i128(a), i256::from_i128(b));
        let k = up.unsigned_abs();
        if up >= 0 {
            let n = a.checked_mul(wide::pow10(k)).unwrap_or(if a.is_negative() {
                i256::MIN
            } else {
                i256::MAX
            });
            (n, b)
        } else {
            (a, b.wrapping_mul(wide::pow10(k)))
        }
    }

    fn out_of_range(&self, v: i256) -> bool {
        let bound = i256::from_i128(pow10(self.out.0));
        v >= bound || v <= bound.wrapping_neg()
    }

    /// Spark's `toPlainString` of the value its `Decimal` arithmetic
    /// produced for row `(a, b)` before the range check (the error value).
    pub(crate) fn error_value(&self, a: i128, b: i128) -> String {
        if self.op == DecimalOp::Divide {
            // `BigDecimal.divide(b, 39, DOWN)`.
            let (s1, s2) = (self.left.1.unsigned_abs(), self.right.1.unsigned_abs());
            let n = i256::from_i128(a).wrapping_mul(wide::pow10(u32::from(s2)));
            let d = i256::from_i128(b).wrapping_mul(wide::pow10(u32::from(s1)));
            return wide::quotient_string(n, d, u32::from(MAX_PRECISION).saturating_add(1));
        }
        if self.op == DecimalOp::Remainder {
            let n = i256::from_i128(a).wrapping_mul(i256::from_i128(self.l_mul));
            let d = i256::from_i128(b).wrapping_mul(i256::from_i128(self.r_mul));
            return self.spark_value(n.wrapping_rem(d));
        }
        self.spark_value(self.exact(a, b))
    }

    /// An operand's plain decimal text.
    pub(crate) fn spark_operand(&self, v: i128, scale: i8) -> String {
        let neg = if v < 0 { "-" } else { "" };
        let digits = v.unsigned_abs().to_string();
        let s = usize::from(scale.unsigned_abs());
        if s == 0 {
            return format!("{neg}{digits}");
        }
        let padded = format!("{digits:0>width$}", width = s.saturating_add(1));
        let (int, frac) = padded.split_at(padded.len().saturating_sub(s));
        format!("{neg}{int}.{frac}")
    }

    /// Spark's `toPlainString` of an exact `+ - *` result `v`: as is for
    /// `+`/`-`; for `*`, rounded down to 39 significant digits
    /// (`Decimal.MATH_CONTEXT`), with a negative scale normalized to 0.
    pub(crate) fn spark_value(&self, v: i256) -> String {
        let text = v.to_string();
        let (sign, digits) = text
            .strip_prefix('-')
            .map_or(("", text.as_str()), |d| ("-", d));
        let mut digits = digits.to_owned();
        let mut scale = usize::from(self.exact_scale);
        if self.op == DecimalOp::Multiply && digits.len() > PRODUCT_DIGITS {
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

/// `n / d` (`d != 0`, `n / d` representable), rounded half away from zero.
fn half_up_div128(n: i128, d: i128) -> i128 {
    let (q, r) = (
        n.checked_div(d).unwrap_or_default(),
        n.checked_rem(d).unwrap_or_default(),
    );
    let (r, d_abs) = (r.unsigned_abs(), d.unsigned_abs());
    // 2|r| >= |d|, without overflow.
    if r >= d_abs.wrapping_sub(r) {
        if (n < 0) == (d < 0) {
            q.wrapping_add(1)
        } else {
            q.wrapping_sub(1)
        }
    } else {
        q
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
    fn fault(self, a: i128, b: i128) -> Option<Fault> {
        self.plan.eval(a, b).err()
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
    fn fault(self, a: i128, b: i128) -> Option<Fault> {
        self.plan.eval(a, b).err()
    }
}

/// The 256-bit path: one exact evaluation per row (rounding `+ - *`, `/`,
/// `%`, `div`).
#[derive(Clone, Copy)]
struct LaneWide<'p> {
    plan: &'p DecimalPlan,
}

impl LaneOp<i128> for LaneWide<'_> {
    // `/`, `%` and `div` divide; the per-row flag word also keeps NULL
    // slots off the slow path for the rounding `+ - *` that share the lane.
    const MAY_DIVIDE_BY_ZERO: bool = true;
    #[inline(always)]
    fn wrapping(self, a: i128, b: i128) -> i128 {
        self.plan.eval(a, b).unwrap_or_default()
    }
    #[inline(always)]
    fn checked(self, a: i128, b: i128) -> (i128, bool) {
        match self.plan.eval(a, b) {
            Ok(v) => (v, false),
            Err(_) => (0, true),
        }
    }
    #[inline(always)]
    fn fault(self, a: i128, b: i128) -> Option<Fault> {
        self.plan.eval(a, b).err()
    }
}

/// What a batch's values prove about an exact `+ - *` plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchProof {
    /// Nothing beyond the types.
    None,
    /// Every result fits the result precision: no row can overflow.
    Fits,
    /// As `Fits`, and every result, rescaled operand and rescale factor
    /// fits an `i64`, so the 64-bit lanes compute it exactly.
    Fits64,
}

/// The OR of the ones'-complement magnitudes of `a` into `m`: branch-free,
/// unlike a 128-bit maximum, and `|v| <= OR + 1` for every `v`.
fn or_magnitudes(m: u128, a: &[i128]) -> u128 {
    a.iter()
        .fold(m, |m, &v| m | (v ^ (v >> 127)).cast_unsigned())
}

/// An upper bound of `|v|` over every lane of a side, NULL slots included
/// (their values are arbitrary; a large one only costs the proof).
#[cfg(test)]
fn max_abs(s: Side<'_, i128>) -> u128 {
    match s {
        Side::Scalar(v) => v.unsigned_abs(),
        Side::Array(a) => or_magnitudes(0, a).saturating_add(1),
    }
}

/// Rows per step of [`batch_proof`]'s scan: it stops at the first step
/// whose running bound already fails, so an unprovable batch costs little.
const PROOF_STEP: usize = 1024;

/// The per-batch range proof of an exact `+ - *` plan: with `A = max|a|`
/// and `B = max|b|` over the batch, `|a·l_mul ± b·r_mul| <= A·l_mul +
/// B·r_mul` and `|a·b| <= A·B`. When that bound is below `10^p_out` no row
/// overflows, so the policy has nothing to check; when it also fits an
/// `i64`, wrapping 64-bit lanes are exact (truncation commutes with
/// wrapping `+ - *`). The proof is local to the batch.
fn batch_proof(plan: &DecimalPlan, l: Side<'_, i128>, r: Side<'_, i128>) -> BatchProof {
    let (lm, rm) = (plan.l_mul.unsigned_abs(), plan.r_mul.unsigned_abs());
    let combine = |a: u128, b: u128| {
        if plan.op == DecimalOp::Multiply {
            a.saturating_mul(b)
        } else {
            a.saturating_mul(lm).saturating_add(b.saturating_mul(rm))
        }
    };
    let limit = pow10(plan.out.0).unsigned_abs();
    // Arrays: the OR so far (the bound is OR + 1); scalars: |v| exactly.
    let (mut oa, mut ob) = (0_u128, 0_u128);
    let side = |s: Side<'_, i128>, or: u128| match s {
        Side::Scalar(v) => v.unsigned_abs(),
        Side::Array(_) => or.saturating_add(1),
    };
    let len = match (l, r) {
        (Side::Array(a), _) | (_, Side::Array(a)) => a.len(),
        _ => 0,
    };
    let mut start = 0_usize;
    loop {
        let end = start.saturating_add(PROOF_STEP).min(len);
        for (s, or) in [(l, &mut oa), (r, &mut ob)] {
            if let Side::Array(a) = s {
                *or = or_magnitudes(*or, a.get(start..end).unwrap_or_default());
            }
        }
        if combine(side(l, oa), side(r, ob)) >= limit {
            return BatchProof::None;
        }
        if end >= len {
            break;
        }
        start = end;
    }
    let bound = combine(side(l, oa), side(r, ob));
    let i64_max = u128::from(i64::MAX.unsigned_abs());
    if bound <= i64_max && lm <= i64_max && rm <= i64_max {
        BatchProof::Fits64
    } else {
        BatchProof::Fits
    }
}

/// The `(precision, scale)` of a decimal array.
fn type_of(a: &Decimal128Array) -> DecimalType {
    (a.precision(), a.scale())
}

/// Checks the operand types and the policy against `plan`.
fn check_plan<O: Overflow>(
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
    left: DecimalType,
    right: DecimalType,
    overflow_impossible: bool,
) -> Result<(), KernelError> {
    if O::MODE == OverflowMode::Unchecked && !overflow_impossible {
        return Err(KernelError::internal(
            "Unchecked decimal arithmetic on types that can overflow",
        ));
    }
    for (o, want) in [(l, left), (r, right)] {
        if let Operand::Array(a) = o
            && type_of(a) != want
        {
            return Err(KernelError::internal(format!(
                "decimal operand is {:?}, plan expects {want:?}",
                a.data_type()
            )));
        }
    }
    Ok(())
}

/// `l op r` under an exact `+ - *` `plan` (the result scale is the exact
/// one) with overflow policy `O`, on the 128-bit lanes or the 64-bit hook.
/// [`decimal_arith`] takes every plan. A scalar operand is a raw unscaled
/// value and must already be in the plan's type for its side (scalars
/// carry no type, so this is not checked; arrays are).
///
/// # Errors
/// - `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION` (with [`KernelError::row`])
///   under `AnsiError`;
/// - `INTERNAL_ERROR` when `Op`, an array type or the policy does not match
///   the plan (`Unchecked` needs [`DecimalPlan::overflow_impossible`]), the
///   plan rounds, for two scalars or arrays of different lengths;
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
    if DecimalOp::from(Op::OP) != plan.op || !plan.is_fast() {
        return Err(KernelError::internal(format!(
            "decimal kernel {:?} called with a {:?} plan to scale {}",
            Op::OP,
            plan.op,
            plan.out.1
        )));
    }
    check_plan::<O>(l, r, plan.left, plan.right, plan.overflow_impossible())?;
    let (sl, sr, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return finish(PrimitiveArray::new_null(len), plan),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let raise = |row, fault| plan_error(plan, ctx, sl, sr, row, fault);
    let narrow = |v: i128| i64::try_from(v).unwrap_or(i64::MAX);
    let lane64 = Lane64::<Op> {
        plan,
        l_mul: narrow(plan.l_mul),
        r_mul: narrow(plan.r_mul),
        bound: narrow(pow10(plan.out.0)),
        op: std::marker::PhantomData,
    };
    let lane128 = Lane128::<Op> {
        plan,
        bound: pow10(plan.out.0),
        op: std::marker::PhantomData,
    };
    // Per-batch range proof (design P2-T160), for checked 128-bit plans
    // only: the checked 64-bit lanes and unchecked lanes cost about what
    // the proof's extra pass over the operands costs (measured).
    let proof = if O::MODE == OverflowMode::Unchecked || plan.storage == DecimalStorage::I64 {
        BatchProof::None
    } else {
        batch_proof(plan, sl, sr)
    };
    let c = match (proof, plan.storage) {
        (BatchProof::Fits64, _) => {
            chunk::compute::<i128, _, Unchecked>(ctx, sl, sr, nulls, len, lane64, raise)
        }
        (BatchProof::Fits, _) => {
            chunk::compute::<i128, _, Unchecked>(ctx, sl, sr, nulls, len, lane128, raise)
        }
        (BatchProof::None, DecimalStorage::I128) => {
            chunk::compute::<i128, _, O>(ctx, sl, sr, nulls, len, lane128, raise)
        }
        (BatchProof::None, DecimalStorage::I64) => {
            chunk::compute::<i128, _, O>(ctx, sl, sr, nulls, len, lane64, raise)
        }
    }?;
    finish(
        PrimitiveArray::try_new(ScalarBuffer::from(c.values), c.nulls)
            .map_err(|e| KernelError::internal(e.to_string()))?,
        plan,
    )
}

/// The error of a raised row of `plan`.
fn plan_error(
    plan: &DecimalPlan,
    ctx: &KernelCtx,
    l: Side<'_, i128>,
    r: Side<'_, i128>,
    row: usize,
    fault: Fault,
) -> KernelError {
    match fault {
        Fault::DivideByZero if plan.op == DecimalOp::Remainder => {
            KernelError::remainder_by_zero(row, ctx.spark_version())
        }
        Fault::DivideByZero => KernelError::divide_by_zero(row),
        Fault::Impossible => KernelError::decimal_remainder_impossible(
            row,
            plan.spark_operand(at(l, row), plan.left.1),
            plan.spark_operand(at(r, row), plan.right.1),
        ),
        Fault::Overflow => {
            let value = plan.error_value(at(l, row), at(r, row));
            KernelError::decimal_out_of_range(row, value, plan.out.0, plan.out.1)
        }
    }
}

/// `l op r` under any `plan` (`+ - * / %`), with overflow policy `O`:
/// exact `+ - *` run on [`decimal_binary`]'s lanes, everything else exactly
/// in 256 bits and rounded `HALF_UP`. A zero divisor is NULL under
/// `NullOnOverflow` (legacy mode and `try_divide`/`try_mod`).
///
/// # Errors
/// As [`decimal_binary`], plus `DIVIDE_BY_ZERO` (`/`) and
/// `REMAINDER_BY_ZERO` (`%`; `DIVIDE_BY_ZERO` in Spark 4.0) under
/// `AnsiError` and `Unchecked`.
///
/// ```
/// use arrow_array::Decimal128Array;
/// use krishiv_kernels::decimal::{DecimalOp, DecimalPlan, decimal_arith, result_type};
/// use krishiv_kernels::{AnsiError, KernelCtx, Operand};
///
/// // CAST(1 AS DECIMAL(38,0)) / CAST(3 AS DECIMAL(38,0)) = 0.333333 (decimal(38,6)).
/// let one = Decimal128Array::from(vec![1_i128]).with_precision_and_scale(38, 0)?;
/// let out = result_type(DecimalOp::Divide, (38, 0), (38, 0), true)?;
/// let plan = DecimalPlan::new(DecimalOp::Divide, (38, 0), (38, 0), out)?;
/// let q = decimal_arith::<AnsiError>(Operand::Array(&one), Operand::Scalar(Some(3)), &plan, &KernelCtx::new())?;
/// assert_eq!((q.value(0), q.scale()), (333_333, 6));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn decimal_arith<O: DecimalOverflow>(
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
    plan: &DecimalPlan,
    ctx: &KernelCtx,
) -> Result<Decimal128Array, KernelError> {
    if plan.is_fast() {
        return match plan.op {
            DecimalOp::Subtract => decimal_binary::<crate::Sub, O>(l, r, plan, ctx),
            DecimalOp::Multiply => decimal_binary::<crate::Mul, O>(l, r, plan, ctx),
            _ => decimal_binary::<crate::Add, O>(l, r, plan, ctx),
        };
    }
    check_plan::<O>(l, r, plan.left, plan.right, plan.overflow_impossible())?;
    let (sl, sr, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return finish(PrimitiveArray::new_null(len), plan),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let lane = LaneWide { plan };
    let c = chunk::compute::<i128, _, O>(ctx, sl, sr, nulls, len, lane, |row, fault| {
        plan_error(plan, ctx, sl, sr, row, fault)
    })?;
    finish(
        PrimitiveArray::try_new(ScalarBuffer::from(c.values), c.nulls)
            .map_err(|e| KernelError::internal(e.to_string()))?,
        plan,
    )
}

fn finish(a: Decimal128Array, plan: &DecimalPlan) -> Result<Decimal128Array, KernelError> {
    a.with_precision_and_scale(plan.out.0, plan.out.1)
        .map_err(|e| KernelError::internal(e.to_string()))
}

/// `l div r` on decimals (Spark `IntegralDivide`): the quotient truncated
/// towards zero must fit `decimal(38, 0)` (else `NUMERIC_VALUE_OUT_OF_RANGE`
/// under ANSI, NULL in legacy mode), then becomes a `bigint` keeping its
/// low 64 bits (Spark's `BigDecimal.longValue`: no error). A zero divisor
/// is NULL under `NullOnOverflow` (legacy mode).
///
/// # Errors
/// `DIVIDE_BY_ZERO` and `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION` under
/// `AnsiError`; engine and memory errors as [`decimal_binary`].
pub fn decimal_integral_divide<O: DecimalOverflow>(
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
    left: DecimalType,
    right: DecimalType,
    ctx: &KernelCtx,
) -> Result<Int64Array, KernelError> {
    check_type(left)?;
    check_type(right)?;
    check_plan::<O>(l, r, left, right, false)?;
    let (sl, sr, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(Int64Array::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let lane = DivLane { left, right };
    let c = chunk::compute::<i128, _, O>(ctx, sl, sr, nulls, len, lane, |row, fault| {
        KernelError::for_fault(row, fault, |row| {
            let q = lane
                .quotient(at(sl, row), at(sr, row))
                .unwrap_or(i256::ZERO);
            KernelError::decimal_out_of_range(row, q.to_string(), MAX_PRECISION, 0)
        })
    })?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Spark keeps the low 64 bits of the quotient"
    )]
    let values: Vec<i64> = c.values.iter().map(|&v| v as i64).collect();
    Int64Array::try_new(ScalarBuffer::from(values), c.nulls)
        .map_err(|e| KernelError::internal(e.to_string()))
}

/// `div` lanes: the truncated quotient (range-checked against
/// `decimal(38, 0)`).
#[derive(Clone, Copy)]
struct DivLane {
    left: DecimalType,
    right: DecimalType,
}

impl DivLane {
    /// `trunc(a / b)` (`None` for `b == 0`): `(a / 10^s1) / (b / 10^s2) =
    /// (a * 10^s2) / (b * 10^s1)`, each term below `10^76`.
    fn quotient(self, a: i128, b: i128) -> Option<i256> {
        let n =
            i256::from_i128(a).wrapping_mul(wide::pow10(u32::from(self.right.1.unsigned_abs())));
        let d = i256::from_i128(b).wrapping_mul(wide::pow10(u32::from(self.left.1.unsigned_abs())));
        n.checked_div(d)
    }

    fn eval(self, a: i128, b: i128) -> Result<i128, Fault> {
        let q = self.quotient(a, b).ok_or(Fault::DivideByZero)?;
        let bound = i256::from_i128(pow10(MAX_PRECISION));
        if q >= bound || q <= bound.wrapping_neg() {
            return Err(Fault::Overflow);
        }
        q.to_i128().ok_or(Fault::Overflow)
    }
}

impl LaneOp<i128> for DivLane {
    const MAY_DIVIDE_BY_ZERO: bool = true;
    #[inline(always)]
    fn wrapping(self, a: i128, b: i128) -> i128 {
        self.eval(a, b).unwrap_or_default()
    }
    #[inline(always)]
    fn checked(self, a: i128, b: i128) -> (i128, bool) {
        match self.eval(a, b) {
            Ok(v) => (v, false),
            Err(_) => (0, true),
        }
    }
    #[inline(always)]
    fn fault(self, a: i128, b: i128) -> Option<Fault> {
        self.eval(a, b).err()
    }
}

/// `-a` or `abs(a)` on decimals: the type is unchanged and the range is
/// symmetric, so nothing can fail.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn decimal_unary(op: UnaryOp, a: &Decimal128Array) -> Result<Decimal128Array, KernelError> {
    let mut out = with_capacity::<i128>(a.len())?;
    out.extend(a.values().iter().map(|&v| match op {
        UnaryOp::Negate => v.wrapping_neg(),
        UnaryOp::Abs => v.wrapping_abs(),
    }));
    Decimal128Array::try_new(ScalarBuffer::from(out), a.nulls().cloned())
        .and_then(|d| d.with_precision_and_scale(a.precision(), a.scale()))
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
        // Rounding plans are accepted (precision loss); scales above the
        // exact one, a remainder scale that differs and `div` are not.
        assert!(!plan(ArithOp::Add, (5, 1), (1, 0), (2, 0)).is_fast());
        assert!(!plan(ArithOp::Multiply, (38, 20), (38, 20), (38, 38)).overflow_impossible());
        for bad in [
            DecimalPlan::new(ArithOp::Add, (5, 1), (1, 0), (3, 2)),
            DecimalPlan::new(DecimalOp::Remainder, (5, 1), (3, 2), (3, 1)),
            DecimalPlan::new(DecimalOp::IntegralDivide, (5, 1), (3, 2), (3, 1)),
        ] {
            assert_eq!(class(bad), C::InternalError);
        }
        let rem = DecimalPlan::new(DecimalOp::Remainder, (5, 1), (3, 2), (3, 2)).unwrap();
        assert!(rem.overflow_impossible() && rem.types_op() == DecimalOp::Remainder);
        let div = DecimalPlan::new(DecimalOp::Divide, (5, 1), (3, 2), (10, 6)).unwrap();
        assert!(!div.overflow_impossible() && div.storage() == DecimalStorage::I128);
    }

    #[test]
    fn batch_proof_boundaries() {
        let mul = plan(ArithOp::Multiply, (20, 0), (20, 0), (38, 0));
        // `max_abs` of an array is `OR + 1`: 9 bounds as 10.
        assert_eq!(max_abs(Side::Array(&[9, -3])), 12);
        assert_eq!(max_abs(Side::Array(&[i128::MIN])), i128::MIN.unsigned_abs());
        assert_eq!(max_abs(Side::Scalar(-7)), 7);
        let p = |a: &[i128], b: i128| batch_proof(&mul, Side::Array(a), Side::Scalar(b));
        // Just below and at 2^63 and 10^38.
        let two63 = 1_i128 << 63;
        // (2^62 - 2 + 1) * 2 = 2^63 - 2 fits; (2^62 - 1 + 1) * 2 = 2^63 does not.
        assert_eq!(p(&[two63 / 2 - 2], 2), BatchProof::Fits64);
        assert_eq!(p(&[two63 / 2 - 1], 2), BatchProof::Fits);
        let e19 = 10_i128.pow(19);
        assert_eq!(p(&[e19 - 1], e19 - 1), BatchProof::Fits);
        assert_eq!(p(&[e19 - 1], e19), BatchProof::None);
        // `+` rescales: 10^2 * (A) + B.
        let add = plan(ArithOp::Add, (20, 0), (20, 2), (22, 2));
        let q = |a: i128, b: i128| batch_proof(&add, Side::Scalar(a), Side::Scalar(b));
        assert_eq!(q(5, 7), BatchProof::Fits64);
        assert_eq!(
            q(10_i128.pow(20) - 1, 10_i128.pow(22) - 1),
            BatchProof::None
        );
        // The scan runs in steps; a large value in a late step still counts.
        let mut long = vec![3_i128; 3 * PROOF_STEP + 5];
        assert_eq!(p(&long, 7), BatchProof::Fits64);
        *long.last_mut().unwrap() = -(10_i128.pow(37));
        assert_eq!(p(&long, 100), BatchProof::None);
        assert_eq!(p(&[], 5), BatchProof::Fits64);
        // A rescale factor beyond `i64` never takes the 64-bit lanes.
        let far = plan(ArithOp::Add, (38, 0), (38, 37), (38, 37));
        let r = batch_proof(&far, Side::Scalar(0), Side::Scalar(1));
        assert_eq!(r, BatchProof::Fits);
    }

    #[test]
    fn wide_lane_forms() {
        let p = plan(ArithOp::Add, (5, 1), (1, 0), (2, 0));
        let lane = LaneWide { plan: &p };
        // 1.5 + 2 = 3.5 rounds half up to 4; overflow wraps to 0 here.
        assert_eq!(lane.wrapping(15, 2), 4);
        assert_eq!(lane.wrapping(9_999, 0), 0);
        assert_eq!(lane.checked(9_999, 0), (0, true));
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
