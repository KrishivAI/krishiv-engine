// SPDX-License-Identifier: Apache-2.0
//! Native Rust aggregate UDFs (ADR 0045): [`AggregateUdf`] declares the
//! function, [`GroupsAccumulator`] holds its state for every group of one
//! hash-aggregate table.
//!
//! The engine drives an accumulator exactly as it drives its built-in
//! aggregates: groups are dense indices that [`resize`](GroupsAccumulator::resize)
//! creates, raw rows fold in with [`update`](GroupsAccumulator::update),
//! partial states (Arrow columns of the declared [`StateField`]s) cross
//! exchanges and spill files and fold in with
//! [`merge`](GroupsAccumulator::merge), and
//! [`evaluate`](GroupsAccumulator::evaluate) gives the final values.
//! `state` and `evaluate` never consume: the engine may read a group's state
//! and keep folding into it.

use std::fmt;

use arrow_array::ArrayRef;
use krishiv_types::DataType;

use super::{ArgField, InvokeContext, NullHandling, ReturnField, Signature, UdfArg, UdfError};

/// One column of an aggregate's partial state: what
/// [`GroupsAccumulator::state`] returns and
/// [`GroupsAccumulator::merge`] reads, one row per group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateField {
    /// The column's name (shown in `EXPLAIN`; need not be unique across
    /// aggregates, the engine prefixes it).
    pub name: String,
    /// Its type.
    pub data_type: DataType,
    /// Whether it may hold nulls.
    pub nullable: bool,
}

/// What never changes about an aggregate UDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct AggregateProperties {
    /// Whether the result follows from the input rows alone (any order
    /// within a group, any partitioning). A nondeterministic aggregate is
    /// never moved or duplicated by the optimizer.
    pub deterministic: bool,
    /// How null inputs reach the UDF: with [`Strict`](NullHandling::Strict)
    /// a row with any null argument is never passed to
    /// [`update`](GroupsAccumulator::update) (as Spark's built-in
    /// aggregates skip them); with [`Custom`](NullHandling::Custom) every
    /// row is.
    pub null_handling: NullHandling,
    /// Whether `f(DISTINCT x)` is allowed (the engine removes duplicate
    /// argument rows per group before `update`). When `false`, a `DISTINCT`
    /// call is `INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX`.
    pub distinct: bool,
}

impl AggregateProperties {
    /// The defaults: deterministic, [`Strict`](NullHandling::Strict),
    /// `DISTINCT` allowed.
    pub const DEFAULT: Self = Self {
        deterministic: true,
        null_handling: NullHandling::Strict,
        distinct: true,
    };

    /// These properties with `deterministic` set.
    #[must_use]
    pub const fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// These properties with `null_handling` set.
    #[must_use]
    pub const fn with_null_handling(mut self, null_handling: NullHandling) -> Self {
        self.null_handling = null_handling;
        self
    }

    /// These properties with `distinct` set.
    #[must_use]
    pub const fn with_distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }
}

impl Default for AggregateProperties {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A native Rust aggregate function: many rows of a group in, one value
/// out.
///
/// Every method is called through [`guard_call`](crate::guard_call), so a
/// panic becomes `FAILED_EXECUTE_UDF` (after unwinding the worker); don't
/// panic. `return_field` and `state_fields` must be pure functions of the
/// argument fields: analysis and every worker must agree on them.
pub trait AggregateUdf: Send + Sync + fmt::Debug + 'static {
    /// The function name, resolved case-insensitively, in the same
    /// namespace as scalar UDFs (see [`UdfLibrary`](crate::UdfLibrary)).
    fn name(&self) -> &str;

    /// The implementation version; plans reference the UDF by name and
    /// version (bump it on any change of behaviour or state layout).
    fn version(&self) -> u32;

    /// The arity and argument types (the analyzer inserts the implicit
    /// casts, as for scalar UDFs).
    fn signature(&self) -> &Signature;

    /// The result type and nullability for arguments of these types.
    ///
    /// # Errors
    /// A [`UdfError`] when these argument types have no result (raised at
    /// analysis).
    fn return_field(&self, args: &[ArgField]) -> Result<ReturnField, UdfError>;

    /// The partial state's columns for arguments of these types: at least
    /// one.
    ///
    /// # Errors
    /// A [`UdfError`] when these argument types have no state.
    fn state_fields(&self, args: &[ArgField]) -> Result<Vec<StateField>, UdfError>;

    /// Determinism, null handling and `DISTINCT` support; see
    /// [`AggregateProperties::DEFAULT`].
    fn properties(&self) -> AggregateProperties {
        AggregateProperties::DEFAULT
    }

    /// A new, empty accumulator for arguments of these types (one per
    /// hash-aggregate table; the engine creates many per query).
    ///
    /// # Errors
    /// A [`UdfError`] when it cannot be created.
    fn accumulator(
        &self,
        args: &[ArgField],
        ctx: &InvokeContext,
    ) -> Result<Box<dyn GroupsAccumulator>, UdfError>;
}

/// An aggregate's state for every group of one table, over dense group
/// indices `0..n`.
///
/// Contract the engine relies on (and checks where it can):
/// - groups come into being through [`resize`](Self::resize) only, empty;
///   an empty group evaluates to the aggregate of no rows;
/// - [`state`](Self::state) and [`evaluate`](Self::evaluate) return one row
///   per entry of `groups`, in that order, and do not change the state;
/// - [`merge`](Self::merge) of a group's [`state`](Self::state) into an
///   empty group gives the same result as folding the same rows into it;
/// - [`size`](Self::size) reports the heap bytes held (the engine reserves
///   them from the query's memory pool after each fold).
pub trait GroupsAccumulator: Send + fmt::Debug {
    /// Makes room for `groups` groups; new ones start empty. Never shrinks.
    fn resize(&mut self, groups: usize);

    /// Folds rows into their groups: row `i` of every argument belongs to
    /// group `groups[i]`. Each argument is a column of `groups.len()` rows
    /// or a one-row literal ([`UdfArg::Scalar`]).
    ///
    /// # Errors
    /// A [`UdfError`] fails the query with `FAILED_EXECUTE_UDF`.
    fn update(&mut self, args: &[UdfArg<'_>], groups: &[u32]) -> Result<(), UdfError>;

    /// Folds partial states into their groups: row `i` of every state
    /// column belongs to group `groups[i]`.
    ///
    /// # Errors
    /// A [`UdfError`] fails the query with `FAILED_EXECUTE_UDF`.
    fn merge(&mut self, states: &[ArrayRef], groups: &[u32]) -> Result<(), UdfError>;

    /// The partial state of `groups`, one column per declared
    /// [`StateField`], one row per entry.
    ///
    /// # Errors
    /// A [`UdfError`] fails the query with `FAILED_EXECUTE_UDF`.
    fn state(&self, groups: &[u32]) -> Result<Vec<ArrayRef>, UdfError>;

    /// The final values of `groups`, one row per entry, of the declared
    /// result type.
    ///
    /// # Errors
    /// A [`UdfError`] fails the query with `FAILED_EXECUTE_UDF`.
    fn evaluate(&self, groups: &[u32]) -> Result<ArrayRef, UdfError>;

    /// Removes rows previously folded with [`update`](Self::update) (input
    /// rows of weight −1, incremental maintenance only; batch queries never
    /// call it). `Ok(false)`, the default, means retraction is not
    /// supported and fails the query with `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// # Errors
    /// A [`UdfError`] fails the query with `FAILED_EXECUTE_UDF`.
    fn retract(&mut self, args: &[UdfArg<'_>], groups: &[u32]) -> Result<bool, UdfError> {
        let _ = (args, groups);
        Ok(false)
    }

    /// The heap bytes this accumulator holds.
    fn size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn properties_default_and_builders() {
        let p = AggregateProperties::default();
        assert_eq!(p, AggregateProperties::DEFAULT);
        assert!(p.deterministic && p.distinct);
        assert_eq!(p.null_handling, NullHandling::Strict);
        let q = p
            .with_deterministic(false)
            .with_null_handling(NullHandling::Custom)
            .with_distinct(false);
        assert!(!q.deterministic && !q.distinct);
        assert_eq!(q.null_handling, NullHandling::Custom);
    }

    #[derive(Debug)]
    struct Nothing;

    impl GroupsAccumulator for Nothing {
        fn resize(&mut self, _: usize) {}
        fn update(&mut self, _: &[UdfArg<'_>], _: &[u32]) -> Result<(), UdfError> {
            Ok(())
        }
        fn merge(&mut self, _: &[ArrayRef], _: &[u32]) -> Result<(), UdfError> {
            Ok(())
        }
        fn state(&self, _: &[u32]) -> Result<Vec<ArrayRef>, UdfError> {
            Ok(Vec::new())
        }
        fn evaluate(&self, _: &[u32]) -> Result<ArrayRef, UdfError> {
            Err(UdfError::new("none"))
        }
        fn size(&self) -> usize {
            0
        }
    }

    #[test]
    fn retract_defaults_to_unsupported() {
        let mut a = Nothing;
        assert_eq!(a.retract(&[], &[]), Ok(false));
    }

    #[test]
    fn a_do_nothing_accumulator_follows_the_contract() {
        let mut a = Nothing;
        a.resize(3);
        assert_eq!(a.update(&[], &[]), Ok(()));
        assert_eq!(a.merge(&[], &[]), Ok(()));
        assert!(a.state(&[]).unwrap().is_empty());
        assert_eq!(a.evaluate(&[]).unwrap_err(), UdfError::new("none"));
        assert_eq!(a.size(), 0);
    }
}
