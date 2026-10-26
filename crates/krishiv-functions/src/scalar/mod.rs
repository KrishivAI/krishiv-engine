// SPDX-License-Identifier: Apache-2.0
//! The execution binding of built-in scalar functions: [`bind_scalar`] turns
//! a function name and its argument types into a [`ScalarFunction`] the
//! physical planner calls once per batch.
//!
//! The binding is free of the plan IR: it sees Arrow types and values only,
//! so it sits next to the kernels (layer L2) and any evaluator can use it.
//! Arguments arrive **after** the analyzer's implicit casts, so a binding
//! accepts exactly the types its function's signature produces (plus the
//! storage alternatives of ADR 0009 for strings and binary: `Utf8View`,
//! `LargeUtf8`, `BinaryView`, `LargeBinary` are read as `Utf8`/`Binary`).
//! Results use the plain carriers: `Utf8`, `Binary`, `Date32`,
//! `Timestamp(Microsecond, "UTC")` for `timestamp`.
//!
//! The canonical operators (`+`, `==`, `like`, `when`, `concat`, ...) and
//! `CAST` are not registry functions; the evaluator implements them, and the
//! conditionals (`if`, `coalesce`, ...), which evaluate their arguments
//! lazily ([`crate::registry::Execution::Evaluator`]).
//!
//! # Calling convention
//!
//! [`ScalarFunction::invoke`] takes one [`Arg`] per bound argument: a column
//! of `rows` values or a literal as a one-row array. The result always has
//! `rows` values, also when every argument is a literal.

mod args;
mod arith;
mod array;
mod bitwise;
mod chars;
mod codec;
mod datetime;
mod hash;
mod math;
mod number;
mod predicate;
mod regex;
mod session;
mod string;
mod url;

pub use regex::{extract_all_pieces, split_regex};

use std::fmt;
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::KernelCtx;
use krishiv_kernels::cast::{FixedOffset, TimeZoneRules};

pub use args::Arg;
use args::{Inputs, broadcast, internal, unsupported};

/// An argument as the planner knows it when binding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BindArg {
    /// A value computed per row; `nullable` is the analyzer's nullability
    /// (a nested field's `nullable` is part of its Arrow type, so
    /// constructors of nested values read it).
    Column {
        /// The value's Arrow type.
        data_type: DataType,
        /// Whether the value may be null.
        nullable: bool,
    },
    /// A literal, as a one-row array. Some functions need the value to
    /// bind (the field of `extract`); others treat it as a column.
    Literal(ArrayRef),
}

impl BindArg {
    /// The argument's Arrow type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Column { data_type, .. } => data_type.clone(),
            Self::Literal(a) => a.data_type().clone(),
        }
    }

    /// Whether the argument may be null: the analyzer's nullability for a
    /// column, the array's for a literal (an untyped `NULL` literal is
    /// always null: its `NullArray` carries no validity bitmap).
    #[must_use]
    pub fn nullable(&self) -> bool {
        match self {
            Self::Column { nullable, .. } => *nullable,
            Self::Literal(a) => a.data_type() == &DataType::Null || a.null_count() > 0,
        }
    }
}

impl From<DataType> for BindArg {
    fn from(t: DataType) -> Self {
        Self::Column {
            data_type: t,
            nullable: true,
        }
    }
}

/// The session state a binding may read. Everything a function's result
/// depends on besides its arguments is here, so a bound function is a pure
/// function of its inputs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BindOptions {
    /// `spark.sql.ansi.enabled`: whether overflow and invalid input raise.
    pub ansi: bool,
    /// The kernel context: CPU target and the Spark version whose error
    /// classes and messages kernels raise.
    pub kernel: KernelCtx,
    /// The session time zone (`spark.sql.session.timeZone`), resolved.
    pub time_zone: Arc<dyn TimeZoneRules>,
    /// The query's start instant, in microseconds since the epoch (UTC):
    /// `current_date` and `current_timestamp` read it, so one query sees one
    /// clock.
    pub query_start_micros: i64,
    /// `spark.sql.decimalOperations.allowPrecisionLoss`: the result types of
    /// decimal arithmetic (`pmod`, `try_add`, ...).
    pub decimal_precision_loss: bool,
    /// The session's user (`current_user()`).
    pub user: Arc<str>,
}

impl BindOptions {
    /// ANSI mode on, the best CPU target, Spark 4.2, `UTC`, query start at
    /// the epoch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ansi: true,
            kernel: KernelCtx::new(),
            time_zone: Arc::new(FixedOffset(0)),
            query_start_micros: 0,
            decimal_precision_loss: true,
            user: Arc::from("unknown"),
        }
    }

    /// With `spark.sql.ansi.enabled = ansi`.
    #[must_use]
    pub fn with_ansi(mut self, ansi: bool) -> Self {
        self.ansi = ansi;
        self
    }

    /// With this kernel context.
    #[must_use]
    pub fn with_kernel(mut self, kernel: KernelCtx) -> Self {
        self.kernel = kernel;
        self
    }

    /// With this session time zone.
    #[must_use]
    pub fn with_time_zone(mut self, zone: Arc<dyn TimeZoneRules>) -> Self {
        self.time_zone = zone;
        self
    }

    /// With `spark.sql.decimalOperations.allowPrecisionLoss = loss`.
    #[must_use]
    pub fn with_decimal_precision_loss(mut self, loss: bool) -> Self {
        self.decimal_precision_loss = loss;
        self
    }

    /// With this session user (`current_user()`, `user()`, `session_user()`).
    #[must_use]
    pub fn with_user(mut self, user: impl Into<Arc<str>>) -> Self {
        self.user = user.into();
        self
    }

    /// With this query start instant (microseconds since the epoch, UTC).
    #[must_use]
    pub fn with_query_start(mut self, micros: i64) -> Self {
        self.query_start_micros = micros;
        self
    }
}

impl Default for BindOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// A built-in scalar function bound to its argument types: what the
/// evaluator calls per batch.
pub trait ScalarFunction: Send + Sync + fmt::Debug {
    /// The function's canonical name.
    fn name(&self) -> &str;
    /// The type of every result.
    fn return_type(&self) -> &DataType;
    /// The function over one batch: `args` in the bound order, each a
    /// column of `rows` values or a one-row literal. The result has `rows`
    /// values.
    ///
    /// # Errors
    /// The function's Spark errors (`ARITHMETIC_OVERFLOW` under ANSI, ...),
    /// and `INTERNAL_ERROR` for arguments that do not match the binding.
    fn invoke(&self, args: &[Arg], rows: usize) -> Result<ArrayRef, KrishivError>;
}

/// The body of a bound function: computes over [`Inputs`] that hold at
/// least one column when `rows > 1`.
type Body = Box<dyn Fn(&Inputs<'_>) -> Result<ArrayRef, KrishivError> + Send + Sync>;

/// The one [`ScalarFunction`] implementation: a name, a type and a body.
struct Bound {
    name: &'static str,
    types: Vec<DataType>,
    return_type: DataType,
    body: Body,
}

impl fmt::Debug for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScalarFunction")
            .field("name", &self.name)
            .field("args", &self.types)
            .field("return_type", &self.return_type)
            .finish_non_exhaustive()
    }
}

impl ScalarFunction for Bound {
    fn name(&self) -> &str {
        self.name
    }

    fn return_type(&self) -> &DataType {
        &self.return_type
    }

    fn invoke(&self, args: &[Arg], rows: usize) -> Result<ArrayRef, KrishivError> {
        if args.len() != self.types.len() {
            return Err(internal(format!(
                "{} bound to {} arguments called with {}",
                self.name,
                self.types.len(),
                args.len()
            )));
        }
        for (a, t) in args.iter().zip(&self.types) {
            a.check(t, rows, self.name)?;
        }
        // No rows: nothing is computed, so nothing raises.
        if rows == 0 {
            return Ok(arrow_array::new_empty_array(&self.return_type));
        }
        // Every argument a literal: compute one row, then repeat it.
        let all_literal = args.iter().all(|a| matches!(a, Arg::Scalar(_)));
        let out = if all_literal {
            let one: Vec<Arg> = args.iter().map(Arg::as_column).collect();
            let row = (self.body)(&Inputs::new(&one, 1))?;
            if rows == 1 {
                row
            } else {
                broadcast(&row, rows)?
            }
        } else {
            (self.body)(&Inputs::new(args, rows))?
        };
        if out.len() != rows || out.data_type() != &self.return_type {
            return Err(internal(format!(
                "{} returned {} rows of {} for {rows} rows of {}",
                self.name,
                out.len(),
                out.data_type(),
                self.return_type
            )));
        }
        Ok(out)
    }
}

/// A binding under construction: the body and return type a family's
/// binder chooses for the argument types.
struct Binding {
    return_type: DataType,
    body: Body,
}

impl Binding {
    fn new(
        return_type: DataType,
        body: impl Fn(&Inputs<'_>) -> Result<ArrayRef, KrishivError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            return_type,
            body: Box::new(body),
        }
    }
}

/// What a family's binder answers for a name: `None` when the name is not
/// one of its functions, else the name as listed in the family's `NAMES`
/// (a `'static` copy) and the binding or why there is none.
type FamilyResult = Option<(&'static str, Result<Binding, KrishivError>)>;

/// A family's binder.
type Binder = fn(&str, &[BindArg], &BindOptions) -> FamilyResult;

/// The families (their names and binder), in lookup order.
const FAMILIES: [(&[&str], Binder); 14] = [
    (array::NAMES, array::bind),
    (string::NAMES, string::bind),
    (datetime::NAMES, datetime::bind),
    (math::NAMES, math::bind),
    (regex::NAMES, regex::bind),
    (arith::NAMES, arith::bind),
    (hash::NAMES, hash::bind),
    (session::NAMES, session::bind),
    (bitwise::NAMES, bitwise::bind),
    (predicate::NAMES, predicate::bind),
    (number::NAMES, number::bind),
    (chars::NAMES, chars::bind),
    (codec::NAMES, codec::bind),
    (url::NAMES, url::bind),
];

/// The name of every scalar function [`bind_scalar`] binds, aliases
/// included, sorted.
#[must_use]
pub fn scalar_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = FAMILIES
        .iter()
        .flat_map(|(names, _)| names.iter().copied())
        .collect();
    names.sort_unstable();
    names
}

/// Whether the result of the scalar function `name` depends only on its
/// arguments and `spark.sql.ansi.enabled`: the string (with its character and
/// encoding families), math and arithmetic families. The datetime, session and
/// hash families read the session time zone, the user or the query start, so a
/// caller that evaluates a call before execution (constant folding) must not
/// evaluate those.
#[must_use]
pub fn is_argument_pure(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [
        string::NAMES,
        chars::NAMES,
        codec::NAMES,
        math::NAMES,
        arith::NAMES,
    ]
    .iter()
    .any(|names| names.contains(&lower.as_str()))
}

/// Binds the scalar function `name` (ASCII case-insensitive) to `args`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a name with no binding, or argument
/// types the binding does not take (the analyzer's signatures make the
/// latter unreachable for analyzed plans); `INTERNAL_ERROR` for a literal
/// argument that is not one row.
pub fn bind_scalar(
    name: &str,
    args: &[BindArg],
    opts: &BindOptions,
) -> Result<Arc<dyn ScalarFunction>, KrishivError> {
    let lower = name.to_ascii_lowercase();
    if let Some(BindArg::Literal(a)) = args
        .iter()
        .find(|a| matches!(a, BindArg::Literal(a) if a.len() != 1))
    {
        return Err(internal(format!(
            "{lower}: a literal argument of {} rows",
            a.len()
        )));
    }
    let Some((name, binding)) = FAMILIES
        .iter()
        .find_map(|(_, bind)| bind(&lower, args, opts))
    else {
        return Err(unsupported(format!("the function {lower}")));
    };
    let binding = binding?;
    Ok(Arc::new(Bound {
        name,
        types: args.iter().map(BindArg::data_type).collect(),
        return_type: binding.return_type,
        body: binding.body,
    }))
}

/// The `'static` entry of `names` equal to `name`.
fn listed(names: &'static [&'static str], name: &str) -> Option<&'static str> {
    names.iter().copied().find(|n| *n == name)
}

/// `KRISHIV_UNSUPPORTED_FEATURE` for argument types `name` does not take.
fn bad_types(name: &str, args: &[BindArg]) -> KrishivError {
    let types: Vec<String> = args.iter().map(|a| a.data_type().to_string()).collect();
    unsupported(format!("{name}({})", types.join(", ")))
}

#[cfg(test)]
mod tests;
