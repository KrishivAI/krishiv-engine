// SPDX-License-Identifier: Apache-2.0
//! [`PlanData`]: the marker of the types a physical plan may hold.
//!
//! A physical plan is data (design P2-T171 invariant I1): it can be
//! printed, compared, encoded and sent to another process, and the same
//! plan can be instantiated any number of times. Every type a plan node
//! holds therefore implements [`PlanData`], a **sealed** trait implemented
//! only in this module and only for value types: numbers, strings, schemas,
//! resolved IR expressions, ids and the plan's own node types. Service
//! handles (`Arc<dyn Operator>`, table providers, spill scopes), locks,
//! atomics, cells and closures do not implement it, so a plan node that
//! holds one does not compile.
//!
//! The plan's own types are declared with the `plan_data!` macro, which
//! destructures every field (or every variant's fields) without `..`: a
//! field added to a node without a [`PlanData`] type, or without being
//! listed, fails to compile.
//!
//! ```
//! use krishiv_exec::physical::{PhysNode, PhysicalPlan, assert_plan_data};
//!
//! assert_plan_data::<PhysicalPlan>();
//! assert_plan_data::<PhysNode>();
//! assert_plan_data::<std::sync::Arc<krishiv_plan::Expr>>();
//! ```
//!
//! None of these are plan data:
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<std::sync::Mutex<u64>>();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<std::sync::atomic::AtomicU64>();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<std::sync::Arc<dyn krishiv_exec::Operator>>();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<std::sync::Arc<dyn krishiv_exec::Source>>();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<krishiv_exec::GlobalState>();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<krishiv_exec::RuntimeFilters>();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<krishiv_memory::SpillScope>();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<
//!     std::sync::Arc<dyn krishiv_exec::planner::TableProviderResolver>,
//! >();
//! ```
//!
//! ```compile_fail
//! krishiv_exec::physical::assert_plan_data::<Box<dyn Fn() + Send + Sync>>();
//! ```
//!
//! ```compile_fail
//! // The marker is sealed: no implementation outside this crate.
//! struct Handle(std::sync::Mutex<u64>);
//! impl krishiv_exec::physical::PlanData for Handle {}
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
}

/// A value type a physical plan may hold (see the module docs). Sealed:
/// implemented in this module only.
pub trait PlanData: sealed::Sealed {}

/// Compiles only for [`PlanData`] types (static checks and tests).
pub const fn assert_plan_data<T: PlanData + ?Sized>() {}

/// The checks the `plan_data!` macro generates call this on every field.
pub(crate) const fn is_plan_data<T: PlanData + ?Sized>(_: &T) {}

/// Implements [`PlanData`] for value types outside the plan's own
/// declarations (scalars, foreign IR and Arrow types).
macro_rules! value_types {
    ($($t:ty),* $(,)?) => {
        $(
            impl sealed::Sealed for $t {}
            impl PlanData for $t {}
        )*
    };
}

value_types!(
    bool,
    u8,
    u16,
    u32,
    u64,
    usize,
    i32,
    i64,
    f64,
    str,
    String,
    // Arrow schemas and types are values.
    arrow_schema::Schema,
    arrow_schema::Field,
    arrow_schema::DataType,
    // The resolved logical IR is data with a canonical encoding
    // (`krishiv_plan::encode`); a registry function id is a name plus an
    // in-process lookup handle that is never part of a key.
    krishiv_plan::Expr,
    krishiv_plan::Scan,
    krishiv_plan::AttrId,
    krishiv_plan::GeneratorKind,
    // Scan and runtime-filter descriptions.
    krishiv_io::ColumnPath,
    krishiv_io::FilterId,
    krishiv_io::KeyFilterOptions,
    // Write-plan option values (references only on the wire; inline
    // secrets are refused at encode time).
    krishiv_common::secret::OptionValue,
    krishiv_common::secret::SecretValue,
    krishiv_common::secret::SecretString,
    krishiv_common::secret::SecretUrl,
    krishiv_common::secret::UrlPart,
    // Operator options that are plain values.
    crate::operators::SortKey,
    crate::operators::RowStageSettings,
    crate::aggregate::AggregateExpr,
    crate::aggregate::AggregateOutput,
    crate::join::JoinType,
    crate::planner::ExplainNode,
);

impl<T: PlanData> sealed::Sealed for Vec<T> {}
impl<T: PlanData> PlanData for Vec<T> {}
impl<T: PlanData> sealed::Sealed for Option<T> {}
impl<T: PlanData> PlanData for Option<T> {}
impl<K: PlanData + Ord, V: PlanData> sealed::Sealed for BTreeMap<K, V> {}
impl<K: PlanData + Ord, V: PlanData> PlanData for BTreeMap<K, V> {}
impl<T: PlanData> sealed::Sealed for Box<T> {}
impl<T: PlanData> PlanData for Box<T> {}
// Shared ownership of an immutable value (`Arc<Expr>`, `SchemaRef`) is
// still a value; `Arc<dyn Trait>` is not, since no trait object is
// `PlanData` (only `str` among the unsized types).
impl<T: PlanData + ?Sized> sealed::Sealed for Arc<T> {}
impl<T: PlanData + ?Sized> PlanData for Arc<T> {}

/// Declares types of the physical plan as [`PlanData`], checking every
/// field: `plan_data!(struct Name { a, b });` lists all fields of a struct,
/// `plan_data!(enum Name { A { x, y }, B(p, q), C });` all variants with
/// all their fields. The check destructures without `..`, so an omitted
/// field or variant does not compile, nor does a field of a type that is
/// not [`PlanData`].
macro_rules! plan_data {
    (tuple $t:ident ( $($p:ident),* $(,)? )) => {
        impl $crate::physical::data::Sealed for $t {}
        impl $crate::physical::data::PlanData for $t {}
        const _: () = {
            #[expect(dead_code, reason = "a compile-time check, never called")]
            fn check(v: &$t) {
                let $t($($p),*) = v;
                $( $crate::physical::data::is_plan_data($p); )*
            }
        };
    };
    (struct $t:ident { $($f:ident),* $(,)? }) => {
        impl $crate::physical::data::Sealed for $t {}
        impl $crate::physical::data::PlanData for $t {}
        const _: () = {
            #[expect(dead_code, reason = "a compile-time check, never called")]
            fn check(v: &$t) {
                let $t { $($f),* } = v;
                $( $crate::physical::data::is_plan_data($f); )*
            }
        };
    };
    (enum $t:ident {
        $( $v:ident $( { $($f:ident),* $(,)? } )? $( ( $($p:ident),* $(,)? ) )? ),* $(,)?
    }) => {
        impl $crate::physical::data::Sealed for $t {}
        impl $crate::physical::data::PlanData for $t {}
        const _: () = {
            #[expect(dead_code, reason = "a compile-time check, never called")]
            fn check(v: &$t) {
                match v {
                    $(
                        $t::$v $( { $($f),* } )? $( ( $($p),* ) )? => {
                            $( $( $crate::physical::data::is_plan_data($f); )* )?
                            $( $( $crate::physical::data::is_plan_data($p); )* )?
                        }
                    )*
                }
            }
        };
    };
}

pub(crate) use plan_data;
pub(crate) use sealed::Sealed;
