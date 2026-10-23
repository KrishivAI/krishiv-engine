// SPDX-License-Identifier: Apache-2.0
//! Runtime CPU-feature dispatch.
//!
//! Hot loops are written once and compiled three times on x86-64: for the
//! build baseline, for x86-64-v3 (AVX2, BMI1/2, FMA, LZCNT, MOVBE) and for
//! x86-64-v4 (v3 plus AVX-512 F/BW/CD/DQ/VL). The best target is detected
//! once per process; a [`KernelCtx`] carries the chosen target, and
//! [`KernelCtx::with_target`] forces any target the CPU supports (tests run
//! every available target, benchmarks compare them). Other architectures
//! run the baseline only.

use std::sync::OnceLock;

use crate::error::KernelError;
use crate::version::SparkVersion;

/// A code-generation target for kernel loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    /// The build's own target features (always available).
    Baseline,
    /// x86-64-v3: AVX2, BMI1, BMI2, FMA, LZCNT, MOVBE, POPCNT.
    X86_64V3,
    /// x86-64-v4: x86-64-v3 plus AVX-512 F, BW, CD, DQ and VL.
    X86_64V4,
}

impl Target {
    /// Every target, from the most portable to the most specific.
    pub const ALL: [Self; 3] = [Self::Baseline, Self::X86_64V3, Self::X86_64V4];

    /// Whether this CPU can run code compiled for the target.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Baseline => true,
            Self::X86_64V3 => has_v3(),
            Self::X86_64V4 => has_v3() && has_v4_extra(),
        }
    }

    /// The most specific available target (detected once, then cached).
    #[must_use]
    pub fn best() -> Self {
        static BEST: OnceLock<Target> = OnceLock::new();
        *BEST.get_or_init(|| Self::select(has_v3(), has_v4_extra()))
    }

    /// The target chosen for a CPU with (`v3`) the x86-64-v3 features and
    /// (`v4_extra`) the AVX-512 additions of x86-64-v4. Pure, so the choice
    /// is testable on any machine.
    #[must_use]
    pub fn select(v3: bool, v4_extra: bool) -> Self {
        match (v3, v4_extra) {
            (true, true) => Self::X86_64V4,
            (true, false) => Self::X86_64V3,
            (false, _) => Self::Baseline,
        }
    }

    /// The targets this CPU supports, baseline first.
    pub fn available() -> impl Iterator<Item = Self> {
        Self::ALL.into_iter().filter(|t| t.is_available())
    }

    /// The target's name: `baseline`, `x86_64v3` or `x86_64v4`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::X86_64V3 => "x86_64v3",
            Self::X86_64V4 => "x86_64v4",
        }
    }

    /// Test-harness guard: the targets named in `required` (a comma
    /// separated list of [`Target::name`]s, typically the
    /// `KRISHIV_REQUIRE_TARGETS` environment variable) must all be available,
    /// so a CI job cannot silently skip a code path its hardware should run.
    ///
    /// # Errors
    /// A message naming unknown or unavailable targets.
    pub fn check_required(required: &str) -> Result<(), String> {
        let bad: Vec<&str> = required
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .filter(|n| {
                !Self::ALL
                    .into_iter()
                    .any(|t| t.name() == *n && t.is_available())
            })
            .collect();
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "required kernel targets unknown or unavailable on this CPU: {}",
                bad.join(", ")
            ))
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn has_v3() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("movbe")
        && std::arch::is_x86_feature_detected!("popcnt")
}

#[cfg(target_arch = "x86_64")]
fn has_v4_extra() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

#[cfg(not(target_arch = "x86_64"))]
fn has_v3() -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
fn has_v4_extra() -> bool {
    false
}

/// Per-call kernel context. Kernels never read configuration; the context
/// carries the code-generation target, which is always one the CPU supports
/// (the invariant that makes dispatch sound), and the Spark version whose
/// error classes and messages kernels raise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCtx {
    target: Target,
    version: SparkVersion,
}

impl KernelCtx {
    /// A context using the best available target.
    #[must_use]
    pub fn new() -> Self {
        Self {
            target: Target::best(),
            version: SparkVersion::default(),
        }
    }

    /// A context forced to `target` (tests and benchmarks).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if this CPU cannot run `target`.
    pub fn with_target(target: Target) -> Result<Self, KernelError> {
        if target.is_available() {
            Ok(Self {
                target,
                version: SparkVersion::default(),
            })
        } else {
            Err(KernelError::internal(format!(
                "kernel target {target:?} is not supported by this CPU"
            )))
        }
    }

    /// The target kernels run on.
    #[must_use]
    pub fn target(&self) -> Target {
        self.target
    }

    /// This context raising the errors of Spark `version`.
    #[must_use]
    pub fn with_spark_version(mut self, version: SparkVersion) -> Self {
        self.version = version;
        self
    }

    /// The Spark version whose errors kernels raise.
    #[must_use]
    pub fn spark_version(&self) -> SparkVersion {
        self.version
    }
}

/// How a kernel entry point runs its loops; every dispatching entry point
/// takes `ctx: &impl Dispatch`. Sealed: the implementations are
/// [`KernelCtx`] (the CPU target chosen at run time, so each loop is
/// compiled once per [`Target`]) and [`Baseline`].
pub trait Dispatch: crate::unsafe_dispatch::Run {
    /// The Spark version whose errors kernels raise.
    fn spark_version(&self) -> SparkVersion;
}

impl Dispatch for KernelCtx {
    #[inline]
    fn spark_version(&self) -> SparkVersion {
        self.version
    }
}

/// A dispatch that always runs the build-baseline code: an entry point
/// called with it instantiates one loop, without the per-target clones and
/// the scalar-operand loops a [`KernelCtx`] needs (a scalar operand is
/// repeated into an array first). For evaluating one row or a constant
/// (plan-time folding), where vector width does not matter but code size
/// does. The results are a [`KernelCtx`]'s: every target and operand shape
/// computes the same values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Baseline {
    version: SparkVersion,
}

impl Baseline {
    /// Raising the errors of Spark `version`.
    #[must_use]
    pub fn new(version: SparkVersion) -> Self {
        Self { version }
    }
}

impl Dispatch for Baseline {
    #[inline]
    fn spark_version(&self) -> SparkVersion {
        self.version
    }
}

impl KernelCtx {
    /// This context's Spark version, on the baseline code only.
    #[must_use]
    pub fn baseline(&self) -> Baseline {
        Baseline::new(self.version)
    }
}

impl Default for KernelCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_always_available_and_best_is_available() {
        assert!(Target::Baseline.is_available());
        assert!(Target::best().is_available());
        assert_eq!(Target::available().next(), Some(Target::Baseline));
        assert_eq!(KernelCtx::default().target(), Target::best());
        let v40 = KernelCtx::new().with_spark_version(SparkVersion::V4_0);
        assert_eq!(
            (v40.spark_version(), KernelCtx::new().spark_version()),
            (SparkVersion::V4_0, SparkVersion::V4_2)
        );
        for t in Target::ALL {
            match KernelCtx::with_target(t) {
                Ok(ctx) => assert_eq!(ctx.target(), t),
                Err(e) => {
                    assert!(!t.is_available());
                    assert_eq!(
                        e.error_class(),
                        krishiv_common::error::ErrorClass::InternalError
                    );
                }
            }
        }
        // v4 implies v3.
        assert!(!Target::X86_64V4.is_available() || Target::X86_64V3.is_available());
    }

    /// `Baseline` computes what every available target computes, for every
    /// operand shape, over several 64-row chunks, errors (with their row and
    /// Spark version) included. `KRISHIV_REQUIRE_TARGETS` (set in CI) makes
    /// a missing target a failure instead of a skipped comparison.
    #[test]
    fn baseline_matches_every_available_target() {
        use arrow_array::types::{Decimal128Type, Float64Type, Int32Type};
        use arrow_array::{Array, Decimal128Array, Float64Array, Int32Array};

        use crate::compare::{Less, compare};
        use crate::decimal::{DecimalOp, DecimalPlan, decimal_arith};
        use crate::integer::{self, UnaryOp};
        use crate::{Add, AnsiError, ArithOp, JavaWrap, Mul, NullOnOverflow, Operand, float};

        let required = std::env::var("KRISHIV_REQUIRE_TARGETS").unwrap_or_default();
        Target::check_required(&required).unwrap();

        // 150 rows: two full chunks and a partial one, with NULLs, zeros
        // and overflowing values in each.
        let a: Int32Array = (0..150)
            .map(|i: i32| (i % 11 != 5).then_some(if i % 13 == 0 { i32::MAX } else { i * 7 - 300 }))
            .collect();
        let b: Int32Array = (0..150)
            .map(|i: i32| (i % 17 != 3).then_some(if i % 19 == 0 { 0 } else { i - 70 }))
            .collect();
        // The only overflow is at row 100, in the second chunk.
        let late: Int32Array = (0..150)
            .map(|i| Some(if i == 100 { i32::MAX } else { i }))
            .collect();
        let x: Float64Array = (0..150)
            .map(|i| match i % 9 {
                0 => None,
                1 => Some(-0.0),
                2 => Some(f64::NAN),
                3 => Some(0.0),
                _ => Some(f64::from(i) * 1.5 - 40.0),
            })
            .collect();
        let d = Decimal128Array::from(vec![Some(99_999), None, Some(-5)])
            .with_precision_and_scale(5, 2)
            .unwrap();
        let plan = DecimalPlan::new(DecimalOp::Multiply, (5, 2), (5, 2), (5, 2)).unwrap();
        let shapes: [(Operand<'_, Int32Type>, Operand<'_, Int32Type>); 6] = [
            (Operand::Array(&a), Operand::Array(&b)),
            (Operand::Array(&a), Operand::Scalar(Some(7))),
            (Operand::Array(&a), Operand::Scalar(Some(0))),
            (Operand::Scalar(Some(-7)), Operand::Array(&b)),
            (Operand::Scalar(None), Operand::Array(&b)),
            (Operand::Array(&late), Operand::Scalar(Some(1))),
        ];
        let fz = Operand::<Float64Type>::Scalar(Some(0.0));
        let float_shapes: [(Operand<'_, Float64Type>, Operand<'_, Float64Type>); 4] = [
            (Operand::Array(&x), Operand::Array(&x)),
            (Operand::Array(&x), Operand::Scalar(Some(2.5))),
            (Operand::Array(&x), fz),
            (Operand::Scalar(Some(-3.0)), Operand::Array(&x)),
        ];
        let null_row = Int32Array::from(vec![None, Some(1)]);
        for t in Target::available() {
            let ctx = KernelCtx::with_target(t)
                .unwrap()
                .with_spark_version(SparkVersion::V4_0);
            let base = ctx.baseline();
            assert_eq!(Dispatch::spark_version(&base), SparkVersion::V4_0);
            for (l, r) in shapes {
                assert_eq!(
                    integer::binary::<Int32Type, Add, JavaWrap>(l, r, &ctx),
                    integer::binary::<Int32Type, Add, JavaWrap>(l, r, &base)
                );
                assert_eq!(
                    integer::binary::<Int32Type, Add, AnsiError>(l, r, &ctx),
                    integer::binary::<Int32Type, Add, AnsiError>(l, r, &base)
                );
                assert_eq!(
                    integer::binary::<Int32Type, Mul, NullOnOverflow>(l, r, &ctx),
                    integer::binary::<Int32Type, Mul, NullOnOverflow>(l, r, &base)
                );
                assert_eq!(
                    integer::remainder::<Int32Type, JavaWrap>(l, r, &ctx),
                    integer::remainder::<Int32Type, JavaWrap>(l, r, &base)
                );
                assert_eq!(
                    integer::remainder::<Int32Type, AnsiError>(l, r, &ctx),
                    integer::remainder::<Int32Type, AnsiError>(l, r, &base)
                );
                assert_eq!(
                    compare::<Int32Type, Less>(l, r, &ctx),
                    compare::<Int32Type, Less>(l, r, &base)
                );
            }
            // The cases above do raise: overflow and remainder by zero, the
            // late overflow at its own row, past the first chunk.
            let (l, r) = (Operand::Array(&a), Operand::Array(&b));
            assert!(integer::binary::<Int32Type, Add, AnsiError>(l, r, &base).is_err());
            let by_zero = (Operand::Array(&a), Operand::Scalar(Some(0)));
            assert!(
                integer::remainder::<Int32Type, AnsiError>(by_zero.0, by_zero.1, &base).is_err()
            );
            let late_add = integer::binary::<Int32Type, Add, AnsiError>(
                Operand::Array(&late),
                Operand::Scalar(Some(1)),
                &base,
            );
            assert_eq!(late_add.unwrap_err().row(), Some(100));
            // A scalar zero divisor never raises on a NULL row: NULL there,
            // the error at the first valid row.
            let one_null = Int32Array::from(vec![None::<i32>]);
            for (l, ok) in [(&one_null, true), (&null_row, false)] {
                let (l, r) = (Operand::Array(l), Operand::Scalar(Some(0)));
                let got = integer::remainder::<Int32Type, AnsiError>(l, r, &base);
                assert_eq!(got, integer::remainder::<Int32Type, AnsiError>(l, r, &ctx));
                assert_eq!(got.map_err(|e| e.row()).err(), (!ok).then_some(Some(1)));
            }
            assert!(
                integer::remainder::<Int32Type, AnsiError>(
                    Operand::Array(&one_null),
                    Operand::Scalar(Some(0)),
                    &base
                )
                .unwrap()
                .is_null(0)
            );
            assert_eq!(
                integer::unary::<Int32Type, AnsiError>(UnaryOp::Negate, &a, &ctx),
                integer::unary::<Int32Type, AnsiError>(UnaryOp::Negate, &a, &base)
            );
            for (fl, fr) in float_shapes {
                for op in [ArithOp::Add, ArithOp::Subtract, ArithOp::Multiply] {
                    assert_eq!(
                        float::binary(op, fl, fr, &ctx),
                        float::binary(op, fl, fr, &base)
                    );
                }
                assert_eq!(
                    float::divide::<Float64Type, JavaWrap>(fl, fr, &ctx),
                    float::divide::<Float64Type, JavaWrap>(fl, fr, &base)
                );
                assert_eq!(
                    float::remainder::<Float64Type, JavaWrap>(fl, fr, &ctx),
                    float::remainder::<Float64Type, JavaWrap>(fl, fr, &base)
                );
                assert_eq!(
                    float::remainder::<Float64Type, AnsiError>(fl, fr, &ctx),
                    float::remainder::<Float64Type, AnsiError>(fl, fr, &base)
                );
                assert_eq!(
                    compare::<Float64Type, Less>(fl, fr, &ctx),
                    compare::<Float64Type, Less>(fl, fr, &base)
                );
            }
            assert!(
                float::remainder::<Float64Type, AnsiError>(Operand::Array(&x), fz, &base).is_err()
            );
            let (dl, dr) = (
                Operand::Array(&d),
                Operand::<Decimal128Type>::Scalar(Some(-200)),
            );
            assert_eq!(
                decimal_arith::<NullOnOverflow>(dl, dr, &plan, &ctx),
                decimal_arith::<NullOnOverflow>(dl, dr, &plan, &base)
            );
            assert_eq!(
                decimal_arith::<AnsiError>(dr, dl, &plan, &ctx),
                decimal_arith::<AnsiError>(dr, dl, &plan, &base)
            );
        }
    }

    #[test]
    fn selection_prefers_avx512_when_present() {
        assert_eq!(Target::select(true, true), Target::X86_64V4);
        assert_eq!(Target::select(true, false), Target::X86_64V3);
        assert_eq!(Target::select(false, true), Target::Baseline);
        assert_eq!(Target::select(false, false), Target::Baseline);
        assert_eq!(Target::best(), Target::select(has_v3(), has_v4_extra()));
    }

    #[test]
    fn required_targets() {
        assert!(Target::check_required("").is_ok());
        assert!(Target::check_required(" baseline ,").is_ok());
        let e = Target::check_required("baseline,avx9").unwrap_err();
        assert!(e.contains("avx9") && !e.contains("baseline"));
        for t in Target::ALL {
            assert_eq!(Target::check_required(t.name()).is_ok(), t.is_available());
        }
    }
}
