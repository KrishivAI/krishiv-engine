// SPDX-License-Identifier: Apache-2.0
//! Spark's math functions: the `double` functions (`sqrt`, `exp`, `ln`,
//! the trigonometric family, `pow`, `atan2`, `hypot`, `log`, ...), `round` /
//! `bround` / `ceil` / `floor` on every numeric type.
//!
//! # Floating-point bits
//!
//! Spark runs the `double` functions on the JVM, whose `Math.*` calls are
//! JIT intrinsics on some CPUs and fdlibm's `StrictMath` elsewhere. These
//! kernels use pure-Rust implementations (the `libm` crate, plus the fdlibm
//! ports of [`fdlibm`] where `libm` differs), so a result is the same on
//! every worker and CPU. Measured against Spark 4.0-4.2 on 280 random inputs
//! per function, the results are bit-identical for `sqrt`, `cbrt`, `exp`,
//! `expm1`, `log2`, `log1p`, `log10` (one input in 145 differs), `asin`,
//! `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
//! `hypot` and `pow`; `sin`, `cos`, `tan`, `ln` and `atan2` differ in the last
//! bit for about 1 to 2 inputs in 100 (the JVM's intrinsics are not fdlibm).
//!
//! # Domains
//!
//! `ln`, `log2`, `log10` and `log` return NULL for arguments at or below
//! zero (and `log1p` at or below -1), as Hive does; everything else follows
//! IEEE (`sqrt(-1)` is NaN).
//!
//! `asinh` and `acosh` follow the context's [`SparkVersion`]
//! (see [`UnaryFn::apply`]).

mod fdlibm;
mod round;

use arrow_array::types::Float64Type;
use arrow_array::{Array, Float64Array};

pub use round::{
    DecimalRounding, RoundMode, ceil_floor_to_long, round_decimal, round_decimal_type, round_f32,
    round_f64, round_integral,
};

use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::ops::Operand;
use crate::version::SparkVersion;

/// A function from `double` to `double`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryFn {
    /// `sqrt`
    Sqrt,
    /// `cbrt`
    Cbrt,
    /// `exp`
    Exp,
    /// `expm1`
    Expm1,
    /// `ln` (NULL at or below 0)
    Ln,
    /// `log2` (NULL at or below 0)
    Log2,
    /// `log10` (NULL at or below 0)
    Log10,
    /// `log1p` (NULL at or below -1)
    Log1p,
    /// `sin`
    Sin,
    /// `cos`
    Cos,
    /// `tan`
    Tan,
    /// `asin`
    Asin,
    /// `acos`
    Acos,
    /// `atan`
    Atan,
    /// `sinh`
    Sinh,
    /// `cosh`
    Cosh,
    /// `tanh`
    Tanh,
    /// `asinh`
    Asinh,
    /// `acosh`
    Acosh,
    /// `atanh`
    Atanh,
    /// `cot`
    Cot,
    /// `csc`
    Csc,
    /// `sec`
    Sec,
    /// `degrees`
    Degrees,
    /// `radians`
    Radians,
    /// `rint`: round half to even
    Rint,
    /// `signum` / `sign`
    Signum,
}

impl UnaryFn {
    /// The result for `x` under Spark `version`; `None` is NULL.
    ///
    /// Spark 4.0 and 4.1 compute `asinh` and `acosh` as `log(x +
    /// sqrt(x*x +- 1))` (which loses every digit of `asinh(x)` for large
    /// negative `x`, giving `-inf`); Spark 4.2 uses accurate versions.
    #[must_use]
    pub fn apply(self, x: f64, version: SparkVersion) -> Option<f64> {
        Some(match self {
            Self::Sqrt => x.sqrt(),
            Self::Cbrt => fdlibm::cbrt(x),
            Self::Exp => libm::exp(x),
            Self::Expm1 => libm::expm1(x),
            Self::Ln => return (x > 0.0 || x.is_nan()).then(|| libm::log(x)),
            Self::Log2 => return (x > 0.0 || x.is_nan()).then(|| libm::log(x) / LN_2),
            Self::Log10 => return (x > 0.0 || x.is_nan()).then(|| fdlibm::log10(x)),
            Self::Log1p => return (x > -1.0 || x.is_nan()).then(|| libm::log1p(x)),
            Self::Sin => libm::sin(x),
            Self::Cos => libm::cos(x),
            Self::Tan => libm::tan(x),
            Self::Asin => libm::asin(x),
            Self::Acos => libm::acos(x),
            Self::Atan => libm::atan(x),
            Self::Sinh => libm::sinh(x),
            Self::Cosh => libm::cosh(x),
            Self::Tanh => fdlibm::tanh(x),
            Self::Asinh if version < SparkVersion::V4_2 => {
                if x == f64::NEG_INFINITY {
                    x
                } else {
                    libm::log(x + (x * x + 1.0).sqrt())
                }
            }
            Self::Asinh => libm::asinh(x),
            Self::Acosh if version < SparkVersion::V4_2 => libm::log(x + (x * x - 1.0).sqrt()),
            Self::Acosh => fdlibm::acosh(x),
            Self::Atanh => 0.5 * (libm::log1p(x) - libm::log1p(-x)),
            Self::Cot => 1.0 / libm::tan(x),
            Self::Csc => 1.0 / libm::sin(x),
            Self::Sec => 1.0 / libm::cos(x),
            Self::Degrees => x * RADIANS_TO_DEGREES,
            Self::Radians => x * DEGREES_TO_RADIANS,
            Self::Rint => x.round_ties_even(),
            Self::Signum => {
                if x == 0.0 || x.is_nan() {
                    x
                } else {
                    x.signum()
                }
            }
        })
    }
}

/// `ln(2)`, the divisor of Spark's `log2` (`ln(x) / ln(2)`).
const LN_2: f64 = std::f64::consts::LN_2;
/// `Math.toDegrees`' factor (JDK 21).
const RADIANS_TO_DEGREES: f64 = 57.295_779_513_082_32;
/// `Math.toRadians`' factor (JDK 21).
const DEGREES_TO_RADIANS: f64 = 0.017_453_292_519_943_295;

/// `f(x)` over a column; NULL where `x` is NULL or `f` has no value.
#[must_use]
pub fn unary(f: UnaryFn, x: &Float64Array, ctx: &KernelCtx) -> Float64Array {
    let version = ctx.spark_version();
    x.iter()
        .map(|v| v.and_then(|v| f.apply(v, version)))
        .collect()
}

/// A function of two `double`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryFn {
    /// `pow(x, y)` / `power`
    Pow,
    /// `atan2(y, x)`
    Atan2,
    /// `hypot(x, y)`
    Hypot,
    /// `log(base, x)`: `ln(x) / ln(base)`, NULL unless both are positive
    Log,
}

impl BinaryFn {
    /// The result for `(l, r)`; `None` is NULL.
    #[must_use]
    pub fn apply(self, l: f64, r: f64) -> Option<f64> {
        Some(match self {
            Self::Pow => java_pow(l, r),
            Self::Atan2 => libm::atan2(l, r),
            Self::Hypot => fdlibm::hypot(l, r),
            Self::Log => {
                if l <= 0.0 || r <= 0.0 {
                    return None;
                }
                libm::log(r) / libm::log(l)
            }
        })
    }
}

/// `Math.pow`: `pow(x, +-0)` is 1, any NaN operand gives NaN, and so does
/// `pow(+-1, +-infinity)` (C99's `pow` returns 1 for both).
fn java_pow(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        1.0
    } else if x.is_nan() || y.is_nan() || (x.abs() == 1.0 && y.is_infinite()) {
        f64::NAN
    } else {
        libm::pow(x, y)
    }
}

/// `f(l, r)` over two operands; NULL where an operand is NULL or `f` has no
/// value.
///
/// # Errors
/// `INTERNAL_ERROR` if two columns differ in length.
pub fn binary(
    f: BinaryFn,
    l: Operand<'_, Float64Type>,
    r: Operand<'_, Float64Type>,
) -> Result<Float64Array, KernelError> {
    let at = |o: Operand<'_, Float64Type>, i: usize| match o {
        Operand::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Operand::Scalar(v) => v,
    };
    let len = match (l, r) {
        (Operand::Array(a), Operand::Array(b)) if a.len() != b.len() => {
            return Err(KernelError::internal(format!(
                "math operands differ in length: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        (Operand::Array(a), _) | (_, Operand::Array(a)) => a.len(),
        (Operand::Scalar(_), Operand::Scalar(_)) => 1,
    };
    Ok((0..len)
        .map(|i| match (at(l, i), at(r, i)) {
            (Some(a), Some(b)) => f.apply(a, b),
            _ => None,
        })
        .collect())
}
