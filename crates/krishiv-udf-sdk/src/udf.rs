// SPDX-License-Identifier: Apache-2.0
//! The UDF boundary types: what a UDF declares ([`Signature`],
//! [`UdfProperties`]), what it computes over ([`UdfArg`],
//! [`InvokeContext`]), and what it reports ([`UdfError`]).
//!
//! Every type here is plain data over Arrow arrays and
//! [`krishiv_types::DataType`], so it crosses threads and processes without
//! handles. The analyzer maps a [`Signature`] onto its own arity and input
//! types (so implicit casts come from the existing coercion, never from the
//! UDF), and the executor checks the result the UDF returns against what
//! [`ScalarUdf::return_field`] declared.

use std::fmt;
use std::sync::Arc;

use arrow_array::ArrayRef;
use krishiv_types::DataType;

/// How a UDF takes its arguments: one variant per arity shape, over the
/// same input-type forms the built-in registry uses (see
/// `krishiv_functions::signature::InputType`, which these mirror so this
/// layer-1 crate stays free of layer-2 types).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Signature {
    /// Exactly one argument per entry, in order, each implicitly cast to
    /// its declared type where the analyzer's coercion allows (as a
    /// built-in with the same declared inputs).
    Exact(Vec<UdfInputType>),
    /// `min_arity` or more arguments, each implicitly cast to `element`
    /// (the last declared type repeats for further arguments, as with
    /// built-ins).
    Variadic {
        /// The type every argument takes.
        element: UdfInputType,
        /// The fewest arguments the call takes.
        min_arity: usize,
    },
    /// Exactly one argument per entry, in order, with no implicit casts:
    /// each argument must already have its declared type (an untyped `NULL`
    /// still becomes it), else the call fails with
    /// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` as a built-in's does.
    Coercible(Vec<UdfInputType>),
}

/// The type an argument must have: what the analyzer casts to (or checks,
/// for [`Signature::Coercible`]). Mirrors
/// `krishiv_functions::signature::InputType` entry for entry; the session
/// layer maps each form onto it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UdfInputType {
    /// Any type, as it is.
    Any,
    /// Exactly this type (implicitly cast where the analyzer allows).
    Exact(DataType),
    /// This type and no other: an untyped `NULL` becomes it, any other
    /// type is `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`.
    Strict(DataType),
    /// Any numeric type; a string or untyped `NULL` becomes `DOUBLE`.
    Numeric {
        /// Whether the required type also names ANSI intervals.
        intervals: bool,
    },
    /// One of these types: an argument of one of them stays as it is,
    /// another is cast to the first it may be cast to implicitly.
    OneOf(Vec<DataType>),
}

impl UdfInputType {
    /// The required type as Spark's errors print it.
    #[must_use]
    pub fn sql(&self) -> String {
        match self {
            Self::Any => "\"ANY\"".to_owned(),
            Self::Exact(t) | Self::Strict(t) => format!("\"{}\"", t.sql()),
            Self::Numeric { intervals: false } => "\"NUMERIC\"".to_owned(),
            Self::Numeric { intervals: true } => "\"NUMERIC\" or \"ANSI INTERVAL\"".to_owned(),
            Self::OneOf(types)
                if matches!(
                    types.as_slice(),
                    [DataType::Timestamp, DataType::TimestampNtz]
                ) =>
            {
                "\"(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)\"".to_owned()
            }
            Self::OneOf(types) => {
                let names: Vec<String> = types.iter().map(|t| format!("\"{}\"", t.sql())).collect();
                format!("({})", names.join(" or "))
            }
        }
    }
}

impl Signature {
    /// Whether `n` arguments pass the arity check.
    #[must_use]
    pub fn accepts(&self, n: usize) -> bool {
        match self {
            Self::Exact(types) | Self::Coercible(types) => n == types.len(),
            Self::Variadic { min_arity, .. } => n >= *min_arity,
        }
    }

    /// Spark's `expectedNum` text for `WRONG_NUM_ARGS` (`"1"`, `"> 0"`).
    #[must_use]
    pub fn arity_text(&self) -> String {
        match self {
            Self::Exact(types) | Self::Coercible(types) => types.len().to_string(),
            Self::Variadic { min_arity, .. } => {
                if *min_arity == 0 {
                    String::new()
                } else {
                    format!("> {}", min_arity.saturating_sub(1))
                }
            }
        }
    }

    /// The declared input types as Spark's errors print them (`"INT,
    /// \"STRING\""`; a variadic tail prints as `"..."`).
    #[must_use]
    pub fn inputs_sql(&self) -> String {
        match self {
            Self::Exact(types) | Self::Coercible(types) => types
                .iter()
                .map(UdfInputType::sql)
                .collect::<Vec<_>>()
                .join(", "),
            Self::Variadic { element, .. } => format!("{}, ...", element.sql()),
        }
    }
}

/// An argument as [`ScalarUdf::return_field`] sees it: its type after the
/// analyzer's implicit casts (`None` for an untyped `NULL`) and whether it
/// may be null. Pure: the declared result follows from these alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgField {
    /// The type; `None` for an untyped `NULL`.
    pub data_type: Option<DataType>,
    /// Whether it may be null.
    pub nullable: bool,
}

/// The declared result of a call: its type and whether it may be null.
/// The executor checks every result against this (wrong type, length or
/// nullability is `FAILED_EXECUTE_UDF`, never silent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnField {
    /// The result type.
    pub data_type: DataType,
    /// Whether the result may be null. A UDF that declares non-nullable
    /// results must never return a null; the executor fails the query if
    /// it does.
    pub nullable: bool,
}

/// How a UDF treats null inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum NullHandling {
    /// A row with a null argument is null without invoking the UDF (as
    /// Spark's strict functions are). The UDF then never sees a null.
    #[default]
    Strict,
    /// The UDF sees every row, nulls included, and decides its own nulls.
    Custom,
}

/// What never changes about a call: determinism and null handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdfProperties {
    /// Whether the result follows from the arguments alone. A
    /// nondeterministic UDF is never moved, duplicated or folded; only a
    /// deterministic one folds over literal arguments.
    pub deterministic: bool,
    /// How null inputs reach the UDF.
    pub null_handling: NullHandling,
}

impl Default for UdfProperties {
    fn default() -> Self {
        Self {
            deterministic: true,
            null_handling: NullHandling::Strict,
        }
    }
}

/// An argument at invocation: a column of `rows` values, or a literal as a
/// one-row array (as the built-in bindings see them). String columns may
/// use any UTF-8 carrier (`Utf8`, `Utf8View`, `LargeUtf8`) and binary
/// columns any binary carrier, whichever the evaluator produced; results
/// use the plain carriers (`Utf8`, `Binary`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum UdfArg<'a> {
    /// A column of `rows` values.
    Column(&'a ArrayRef),
    /// A literal, as a one-row array.
    Scalar(&'a ArrayRef),
}

impl UdfArg<'_> {
    /// The values.
    #[must_use]
    pub fn array(self) -> ArrayRef {
        match self {
            Self::Column(a) | Self::Scalar(a) => Arc::clone(a),
        }
    }
}

/// What a UDF's result may depend on besides its arguments: plain values
/// only, never a session or catalog handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvokeContext {
    /// The partition the batch belongs to. The engine runs each query in
    /// one stage without stable partition ids, so this is always zero; a
    /// UDF that reads it must declare itself nondeterministic.
    pub partition: u32,
    /// `spark.sql.ansi.enabled`.
    pub ansi: bool,
    /// The session time zone as a fixed offset in seconds, when it is one
    /// (`UTC`, `+hh:mm`); `None` in a region zone, which the engine does
    /// not resolve yet.
    pub time_zone: Option<i32>,
}

/// What a UDF reports when it cannot compute its result: a plain message.
/// The executor raises it as Spark's `FAILED_EXECUTE_UDF` (SQLSTATE 39000),
/// naming the UDF, its signature and its result type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdfError {
    message: String,
}

impl UdfError {
    /// An error with `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for UdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for UdfError {}

impl From<String> for UdfError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for UdfError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// A native Rust scalar function: one row in, one value out.
///
/// The implementation must be pure besides what [`InvokeContext`] carries:
/// the same arguments (and context) give the same result, unless
/// [`properties`](Self::properties) declares otherwise. It must not retain
/// state between invocations, and it must never panic (a panic fails the
/// query with `FAILED_EXECUTE_UDF`, but only after unwinding the worker).
/// Results use the plain Arrow carriers (`Utf8` for `STRING`, `Binary` for
/// `BINARY`); the engine converts strings and binaries to views itself.
pub trait ScalarUdf: Send + Sync + fmt::Debug + 'static {
    /// The function name, resolved case-insensitively (stored lower-case).
    /// Built-ins win over a same-name session UDF on the bare search path.
    fn name(&self) -> &str;

    /// The implementation version. Plans reference the UDF by name and
    /// version, so bump it whenever the behaviour changes: a cached plan
    /// naming an older version never runs the newer code, and a worker
    /// holding another version refuses the plan instead of misreading it.
    fn version(&self) -> u32;

    /// The arity and argument types (see [`Signature`]).
    fn signature(&self) -> &Signature;

    /// The result type and nullability for arguments of these types (after
    /// the analyzer's implicit casts). Pure: the same argument fields give
    /// the same field, so analysis and execution always agree.
    ///
    /// # Errors
    /// A [`UdfError`] when these argument types have no result (raised as
    /// Spark's error for the call, at analysis).
    fn return_field(&self, args: &[ArgField]) -> Result<ReturnField, UdfError>;

    /// Determinism and null handling; deterministic and [`Strict`](NullHandling::Strict)
    /// unless overridden.
    fn properties(&self) -> UdfProperties {
        UdfProperties::default()
    }

    /// The function over one batch: `args` in the bound order, each a
    /// column of `rows` values or a one-row literal (see [`UdfArg`]). The
    /// result has `rows` values, of the declared type; with
    /// [`Strict`](NullHandling::Strict) handling the engine only invokes
    /// this for rows without nulls (and returns null without invoking when
    /// every row has one).
    ///
    /// # Errors
    /// A [`UdfError`] when the values cannot be computed (raised as
    /// `FAILED_EXECUTE_UDF`, naming the UDF).
    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        ctx: &InvokeContext,
    ) -> Result<ArrayRef, UdfError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_types_print_as_spark() {
        assert_eq!(UdfInputType::Any.sql(), r#""ANY""#);
        assert_eq!(UdfInputType::Exact(DataType::Date).sql(), r#""DATE""#);
        assert_eq!(UdfInputType::Strict(DataType::Integer).sql(), r#""INT""#);
        assert_eq!(
            UdfInputType::Numeric { intervals: true }.sql(),
            r#""NUMERIC" or "ANSI INTERVAL""#
        );
        assert_eq!(
            UdfInputType::OneOf(vec![DataType::String, DataType::Binary]).sql(),
            r#"("STRING" or "BINARY")"#
        );
        assert_eq!(
            UdfInputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz]).sql(),
            r#""(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)""#
        );
    }

    #[test]
    fn properties_default_to_deterministic_strict() {
        let props = UdfProperties::default();
        assert!(props.deterministic);
        assert_eq!(props.null_handling, NullHandling::Strict);
        assert_eq!(NullHandling::default(), NullHandling::Strict);
    }

    #[test]
    fn udf_errors_carry_their_message() {
        let e = UdfError::new("boom");
        assert_eq!(e.message(), "boom");
        assert_eq!(format!("{e}"), "boom");
        assert_eq!(UdfError::from("x".to_owned()), UdfError::new("x"));
        assert_eq!(UdfError::from("y"), UdfError::new("y"));
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn args_borrow_their_arrays() {
        let array: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
        let arg = UdfArg::Column(&array);
        assert!(Arc::ptr_eq(&arg.array(), &array));
        let lit = UdfArg::Scalar(&array);
        assert!(Arc::ptr_eq(&lit.array(), &array));
    }
}
