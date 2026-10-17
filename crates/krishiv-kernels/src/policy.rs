// SPDX-License-Identifier: Apache-2.0
//! Overflow policies: zero-sized types chosen once per expression instance
//! (by `krishiv-functions` from the Spark function, the types and
//! `spark.sql.ansi.enabled`), so each policy is its own monomorph.

/// The runtime name of a policy, for code that must branch on it outside the
/// hot loop (and for the reference implementation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverflowMode {
    /// Overflow is impossible by the types; nothing is checked.
    Unchecked,
    /// Two's-complement wrap (Spark legacy integral arithmetic, Java semantics).
    JavaWrap,
    /// An overflowing row becomes NULL (Spark legacy decimal arithmetic).
    NullOnOverflow,
    /// The first overflowing non-NULL row raises (Spark ANSI mode, the default).
    AnsiError,
}

mod sealed {
    pub trait Sealed {}
}

/// An overflow policy (one of the four zero-sized types in this module).
pub trait Overflow: sealed::Sealed + Copy + Send + Sync + 'static {
    /// The policy's runtime name.
    const MODE: OverflowMode;
    /// Whether the kernel computes overflow flags at all.
    const CHECKS: bool = matches!(
        Self::MODE,
        OverflowMode::NullOnOverflow | OverflowMode::AnsiError
    );
}

/// Overflow is impossible by the operand types (decided at plan time, e.g. a
/// decimal sum whose result precision covers both inputs). No flag is
/// computed; if the plan's proof were wrong the result would wrap, never
/// panic or invoke undefined behaviour.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unchecked;

/// Legacy integral arithmetic: wrap like Java.
#[derive(Debug, Clone, Copy, Default)]
pub struct JavaWrap;

/// Legacy decimal arithmetic: an out-of-range result is NULL.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullOnOverflow;

/// ANSI arithmetic: raise `ARITHMETIC_OVERFLOW` /
/// `NUMERIC_VALUE_OUT_OF_RANGE` at the first failing non-NULL row.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnsiError;

macro_rules! policy {
    ($($t:ident => $m:ident),*) => {$(
        impl sealed::Sealed for $t {}
        impl Overflow for $t {
            const MODE: OverflowMode = OverflowMode::$m;
        }
    )*};
}
policy!(Unchecked => Unchecked, JavaWrap => JavaWrap, NullOnOverflow => NullOnOverflow, AnsiError => AnsiError);
