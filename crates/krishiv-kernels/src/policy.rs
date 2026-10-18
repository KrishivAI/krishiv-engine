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

/// Why one row of a kernel has no exact result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fault {
    /// The exact result does not fit the result type.
    Overflow,
    /// A division or remainder by zero.
    DivideByZero,
    /// No result in any mode (a decimal remainder whose integral quotient
    /// exceeds 39 significant digits: Spark's "Division impossible").
    Impossible,
}

/// What a policy does with a faulting row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    /// Keep the wrapped (two's-complement) value.
    Wrap,
    /// The row becomes NULL.
    Null,
    /// The kernel fails at the first such non-NULL row.
    Raise,
}

impl OverflowMode {
    /// The action this policy takes for `fault`. Overflow follows the policy
    /// name. Division by zero has no wrapped value: Spark's legacy mode makes
    /// it NULL (`JavaWrap`, `NullOnOverflow`), ANSI raises, and `Unchecked`
    /// (a proof about overflow only) raises rather than guess. An impossible
    /// row raises under every policy.
    #[must_use]
    pub const fn action(self, fault: Fault) -> Action {
        match (self, fault) {
            (Self::Unchecked | Self::JavaWrap, Fault::Overflow) => Action::Wrap,
            (Self::NullOnOverflow | Self::JavaWrap, Fault::DivideByZero)
            | (Self::NullOnOverflow, Fault::Overflow) => Action::Null,
            (Self::AnsiError, _)
            | (Self::Unchecked, Fault::DivideByZero)
            | (_, Fault::Impossible) => Action::Raise,
        }
    }
}

/// An overflow policy (one of the four zero-sized types in this module).
pub trait Overflow: sealed::Sealed + Copy + Send + Sync + 'static {
    /// The policy's runtime name.
    const MODE: OverflowMode;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_per_fault() {
        use Action::{Null, Raise, Wrap};
        use Fault::{DivideByZero as Div, Overflow as Ovf};
        let table = [
            (OverflowMode::Unchecked, Wrap, Raise),
            (OverflowMode::JavaWrap, Wrap, Null),
            (OverflowMode::NullOnOverflow, Null, Null),
            (OverflowMode::AnsiError, Raise, Raise),
        ];
        for (mode, ovf, div) in table {
            assert_eq!((mode.action(Ovf), mode.action(Div)), (ovf, div), "{mode:?}");
            assert_eq!(mode.action(Fault::Impossible), Raise);
        }
    }
}
