// SPDX-License-Identifier: Apache-2.0
//! How the analyzer types a function call: the kinds of functions, the
//! Spark profile, argument types and constants, the implicit casts a
//! function asks for and its result type.

use std::fmt;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

/// What kind of function a call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum FunctionKind {
    /// One row in, one value out.
    Scalar,
    /// An aggregate (`sum`, `count`, ...).
    Aggregate,
    /// A window function (`row_number`, `rank`, ...).
    Window,
    /// A generator (`explode`, ...): one row in, many rows out.
    Generator,
}

impl FunctionKind {
    /// A stable code per kind, for canonical encodings.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Scalar => 1,
            Self::Aggregate => 2,
            Self::Window => 3,
            Self::Generator => 4,
        }
    }
}

/// The Spark version whose behaviour a session follows (ADR 0003, 0008).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum SparkProfile {
    /// Spark 3.5.
    V3_5,
    /// Spark 4.0.
    V4_0,
    /// Spark 4.1.
    V4_1,
    /// Spark 4.2 (the default).
    #[default]
    V4_2,
}

impl SparkProfile {
    /// Every profile, oldest first.
    pub const ALL: [Self; 4] = [Self::V3_5, Self::V4_0, Self::V4_1, Self::V4_2];

    /// The release line, as `"4.2"`.
    #[must_use]
    pub const fn line(self) -> &'static str {
        match self {
            Self::V3_5 => "3.5",
            Self::V4_0 => "4.0",
            Self::V4_1 => "4.1",
            Self::V4_2 => "4.2",
        }
    }

    /// A stable code per profile (`42` for 4.2), for fingerprints.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::V3_5 => 35,
            Self::V4_0 => 40,
            Self::V4_1 => 41,
            Self::V4_2 => 42,
        }
    }

    /// The profile of a release line (`"4.2"`).
    #[must_use]
    pub fn from_line(line: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.line() == line)
    }
}

/// A literal argument's value, for functions whose typing reads it (the
/// field of `extract`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Constant {
    /// `NULL`.
    Null,
    /// A `BOOLEAN`.
    Boolean(bool),
    /// A `TINYINT` .. `BIGINT`.
    Integral(i64),
    /// A `STRING`.
    String(String),
    /// A `FLOAT`, `DOUBLE` or `DECIMAL`, as the integer it truncates to (what
    /// a cast to `INT` gives: the scale of `round(x, 1.5)`).
    Fractional(i64),
    /// A cast of a literal that Spark raises while folding it (overflow,
    /// malformed input); the function that reads the value raises it.
    Error(KrishivError),
}

/// The type of a value, as function signatures see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgType {
    /// The type; `None` for an untyped `NULL`.
    pub data_type: Option<DataType>,
    /// Whether it may be null.
    pub nullable: bool,
}

impl ArgType {
    /// A value of type `data_type`.
    #[must_use]
    pub const fn new(data_type: DataType, nullable: bool) -> Self {
        Self {
            data_type: Some(data_type),
            nullable,
        }
    }
}

/// The arguments of a call as typing sees them: their types and, for
/// literal arguments, their values.
#[derive(Clone, Copy)]
pub struct CallTypes<'a> {
    args: &'a [ArgType],
    constants: &'a [Option<Constant>],
    literals: &'a [bool],
    sql: Option<&'a dyn Fn(usize) -> Option<String>>,
    ansi: bool,
    decimal_loss: bool,
}

impl fmt::Debug for CallTypes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallTypes")
            .field("args", &self.args)
            .field("constants", &self.constants)
            .field("literals", &self.literals)
            .field("ansi", &self.ansi)
            .field("decimal_loss", &self.decimal_loss)
            .finish_non_exhaustive()
    }
}

impl<'a> CallTypes<'a> {
    /// `args`, with `constants[i]` the value of literal argument `i` (a
    /// shorter `constants` leaves the rest unknown).
    #[must_use]
    pub const fn new(args: &'a [ArgType], constants: &'a [Option<Constant>]) -> Self {
        Self {
            args,
            constants,
            literals: &[],
            sql: None,
            ansi: true,
            decimal_loss: true,
        }
    }

    /// With `literals[i]` telling whether argument `i` is a plain literal
    /// (not merely foldable). Without it, every constant counts as a
    /// literal.
    #[must_use]
    pub const fn with_literals(mut self, literals: &'a [bool]) -> Self {
        self.literals = literals;
        self
    }

    /// With `spark.sql.ansi.enabled = ansi` (on by default): some legacy
    /// typing rules differ.
    #[must_use]
    pub const fn with_ansi(mut self, ansi: bool) -> Self {
        self.ansi = ansi;
        self
    }

    /// `spark.sql.ansi.enabled`.
    #[must_use]
    pub const fn ansi(&self) -> bool {
        self.ansi
    }

    /// With `spark.sql.decimalOperations.allowPrecisionLoss = loss` (on by
    /// default): decimal arithmetic result types follow it.
    #[must_use]
    pub const fn with_decimal_loss(mut self, loss: bool) -> Self {
        self.decimal_loss = loss;
        self
    }

    /// `spark.sql.decimalOperations.allowPrecisionLoss`.
    #[must_use]
    pub const fn decimal_loss(&self) -> bool {
        self.decimal_loss
    }

    /// With `sql(i)` rendering argument `i` as Spark's errors quote it
    /// (`extract`'s `INVALID_EXTRACT_FIELD` names the source); rendered
    /// only when a rule asks.
    #[must_use]
    pub const fn with_sql(mut self, sql: &'a dyn Fn(usize) -> Option<String>) -> Self {
        self.sql = Some(sql);
        self
    }

    /// The SQL text of argument `i`, when known.
    #[must_use]
    pub fn sql(&self, i: usize) -> Option<String> {
        self.sql.and_then(|f| f(i))
    }

    /// The argument types.
    #[must_use]
    pub const fn args(&self) -> &'a [ArgType] {
        self.args
    }

    /// The type of argument `i`.
    #[must_use]
    pub fn data_type(&self, i: usize) -> Option<&'a DataType> {
        self.args.get(i).and_then(|a| a.data_type.as_ref())
    }

    /// The value of argument `i`, when it is a literal.
    #[must_use]
    pub fn constant(&self, i: usize) -> Option<&'a Constant> {
        self.constants.get(i).and_then(Option::as_ref)
    }

    /// Whether argument `i` is a plain literal: what Spark's rules that
    /// match `Literal(..)` (the `ignoreNulls` flag of `first`) require, as
    /// opposed to the foldable expressions [`Self::constant`] also gives.
    #[must_use]
    pub fn is_literal(&self, i: usize) -> bool {
        match self.literals.get(i) {
            Some(plain) => *plain && self.constant(i).is_some(),
            None => self.constant(i).is_some(),
        }
    }

    /// Whether argument `i` is a plain literal, whether or not it folds to
    /// a [`Constant`] (a timestamp or binary literal has no constant, but
    /// Spark still folds it): what `schema_of_csv` needs to tell an
    /// unfoldable literal from a non-foldable expression.
    #[must_use]
    pub fn is_plain_literal(&self, i: usize) -> bool {
        self.literals.get(i).is_some_and(|plain| *plain)
    }

    /// Whether any argument may be null.
    #[must_use]
    pub fn any_nullable(&self) -> bool {
        self.args.iter().any(|a| a.nullable)
    }
}

/// A function's result type for its (cast) arguments. A closure, so
/// catalog and session functions can carry state.
pub type ResultType = Arc<dyn Fn(&CallTypes<'_>) -> Result<ArgType, KrishivError> + Send + Sync>;

/// A type an argument must have (Spark's `ExpectsInputTypes`); the analyzer
/// casts an argument of another type to it where Spark's ANSI implicit
/// casts allow, else fails with `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputType {
    /// Any type, as it is.
    Any,
    /// Exactly this type.
    Exact(DataType),
    /// This type and no other: an untyped `NULL` becomes it, any other
    /// type is `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` (Spark's `if`
    /// checks its condition's type instead of casting it).
    Strict(DataType),
    /// Any numeric type; a string or untyped `NULL` becomes `DOUBLE`.
    /// `intervals`: Spark also takes ANSI intervals (it names them in the
    /// error; Krishiv has no interval values yet).
    Numeric {
        /// Whether Spark's required type names ANSI intervals.
        intervals: bool,
    },
    /// One of these types (Spark's `TypeCollection`): an argument of one of
    /// them stays as it is, another is cast to the first it may be cast to
    /// implicitly.
    OneOf(Vec<DataType>),
}

impl InputType {
    /// The required type as Spark's errors print it.
    #[must_use]
    pub fn sql(&self) -> String {
        match self {
            Self::Any => "\"ANY\"".to_owned(),
            Self::Exact(t) | Self::Strict(t) => format!("\"{}\"", t.sql()),
            Self::Numeric { intervals: false } => "\"NUMERIC\"".to_owned(),
            Self::Numeric { intervals: true } => "\"NUMERIC\" or \"ANSI INTERVAL\"".to_owned(),
            // Spark's `AnyTimestampType` names itself in words.
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

/// Chooses a function's input types from its arguments, as Spark's
/// expression builders choose an overload (`lpad` on binary is the binary
/// pad, else the string pad).
#[derive(Clone, Copy)]
pub struct InputBuilder(pub fn(&CallTypes<'_>) -> Vec<InputType>);

impl fmt::Debug for InputBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("InputBuilder")
    }
}

impl PartialEq for InputBuilder {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.0, other.0)
    }
}

impl Eq for InputBuilder {}

/// An arithmetic operator whose argument typing a function reuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operator {
    /// `+`.
    Add,
    /// `-`.
    Subtract,
    /// `*`.
    Multiply,
    /// `/`.
    Divide,
    /// `%`.
    Remainder,
}

/// How a function's arguments are cast before its result type is computed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Inputs {
    /// As they are.
    Any,
    /// Argument `i` takes `types[i]`; the last type repeats for further
    /// arguments.
    Expects(Vec<InputType>),
    /// Every argument takes the type all of them meet at (`coalesce`);
    /// `DATATYPE_MISMATCH.DATA_DIFF_TYPES` when there is none.
    Common,
    /// As [`Inputs::Common`], but a string does not meet another type
    /// (`greatest`, `least`): `DATATYPE_MISMATCH.DATA_DIFF_TYPES`.
    CommonNoStrings,
    /// The first arguments take these types, one entry each; the others
    /// take the type they meet at (`if(cond, a, b)`, `nvl2(x, a, b)`).
    Leading(Vec<InputType>),
    /// As [`Inputs::Common`] with a hidden `0` argument (`zeroifnull(x)` is
    /// `coalesce(x, 0)`).
    CommonWithZero,
    /// The arguments must be comparable with `=`
    /// (`DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES` otherwise) but are not
    /// cast: the evaluator compares them, and the result has the first
    /// argument's type (`nullif(a, b)`). With `zero`, the single argument is
    /// compared with a hidden `0` (`nullifzero`).
    Comparable {
        /// Compare the argument with `0`.
        zero: bool,
    },
    /// Like the null-safe `<=>` operator: the arguments must be comparable
    /// (`DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES` otherwise) and are cast
    /// to their common type (`equal_null`).
    NullSafeEqual,
    /// The arguments of the arithmetic operator `op`: the implicit casts of
    /// `a op b` (`mod`, `try_add`, ...).
    Operator(Operator),
    /// As [`Inputs::Operator`], with implicit casts that are `TRY_CAST`s:
    /// `try_add('x', 1)` is NULL, not an error.
    TryOperator(Operator),
    /// As [`Inputs::Expects`], with the types chosen from the arguments.
    Built(InputBuilder),
}

impl Inputs {
    /// The input types for `call`: `None` for [`Inputs::Any`] and
    /// [`Inputs::Common`], which have no per-argument types.
    #[must_use]
    pub fn expected(&self, call: &CallTypes<'_>) -> Option<Vec<InputType>> {
        match self {
            Self::Any
            | Self::Common
            | Self::CommonNoStrings
            | Self::Leading(_)
            | Self::CommonWithZero
            | Self::Comparable { .. }
            | Self::NullSafeEqual
            | Self::Operator(_)
            | Self::TryOperator(_) => None,
            Self::Expects(types) => Some(types.clone()),
            Self::Built(b) => Some((b.0)(call)),
        }
    }
}

/// The argument counts a function takes, as Spark's arity check sees them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arity {
    /// Bit `n` (for `n` up to [`Arity::PROBED`]): whether `n` arguments
    /// pass; counts above take the answer for [`Arity::PROBED`].
    counts: u8,
    /// Spark's `expectedNum` text for `WRONG_NUM_ARGS` (`"1"`, `"[2, 3]"`,
    /// `"> 0"`).
    expected: String,
}

impl Arity {
    /// The largest argument count probed one by one.
    pub const PROBED: usize = 6;

    /// `counts[n]` says whether `n` arguments pass, for `n` up to
    /// [`Arity::PROBED`]; `expected` is Spark's `expectedNum` text.
    #[must_use]
    pub fn new(counts: [bool; Self::PROBED + 1], expected: impl Into<String>) -> Self {
        let counts = counts
            .iter()
            .enumerate()
            .fold(0_u8, |acc, (n, ok)| acc | (u8::from(*ok) << n));
        Self {
            counts,
            expected: expected.into(),
        }
    }

    /// Any number of arguments.
    #[must_use]
    pub fn any() -> Self {
        Self::new([true; Self::PROBED + 1], "")
    }

    /// Whether `n` arguments pass the arity check.
    #[must_use]
    pub fn accepts(&self, n: usize) -> bool {
        self.counts & (1 << n.min(Self::PROBED)) != 0
    }

    /// Spark's `expectedNum` text.
    #[must_use]
    pub fn expected(&self) -> &str {
        &self.expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_counts() {
        let a = Arity::new([false, false, true, true, false, false, false], "[2, 3]");
        let ok: Vec<bool> = (0..9).map(|n| a.accepts(n)).collect();
        assert_eq!(
            ok,
            [false, false, true, true, false, false, false, false, false]
        );
        assert_eq!(a.expected(), "[2, 3]");
        let v = Arity::new([false, true, true, true, true, true, true], "> 0");
        assert!(v.accepts(40) && !v.accepts(0));
        assert!(Arity::any().accepts(0) && Arity::any().accepts(99));
    }

    #[test]
    fn required_types_print_as_spark() {
        assert_eq!(InputType::Any.sql(), r#""ANY""#);
        assert_eq!(InputType::Exact(DataType::Date).sql(), r#""DATE""#);
        assert_eq!(
            InputType::Numeric { intervals: true }.sql(),
            r#""NUMERIC" or "ANSI INTERVAL""#
        );
        assert_eq!(
            InputType::OneOf(vec![DataType::String, DataType::Binary]).sql(),
            r#"("STRING" or "BINARY")"#
        );
    }

    #[test]
    fn profiles_and_kinds() {
        for p in SparkProfile::ALL {
            assert_eq!(SparkProfile::from_line(p.line()), Some(p));
            assert_eq!(p.line().replace('.', ""), p.code().to_string());
        }
        assert_eq!(SparkProfile::from_line("3.4"), None);
        let codes: Vec<u8> = [
            FunctionKind::Scalar,
            FunctionKind::Aggregate,
            FunctionKind::Window,
            FunctionKind::Generator,
        ]
        .map(FunctionKind::code)
        .into();
        assert_eq!(codes, [1, 2, 3, 4]);
    }

    #[test]
    fn call_types_and_builders() {
        let args = [
            ArgType::new(DataType::String, false),
            ArgType {
                data_type: None,
                nullable: true,
            },
        ];
        let consts = [Some(Constant::String("x".to_owned()))];
        let call = CallTypes::new(&args, &consts);
        assert_eq!(call.data_type(0), Some(&DataType::String));
        assert_eq!(call.data_type(1), None);
        assert_eq!(call.constant(0), Some(&Constant::String("x".to_owned())));
        assert_eq!(call.constant(1), None);
        assert!(call.any_nullable());
        assert_eq!(call.args().len(), 2);
        assert_eq!(call.sql(0), None);
        assert!(call.ansi() && !call.with_ansi(false).ansi());
        assert!(format!("{call:?}").starts_with("CallTypes"));
        let sql = |i: usize| (i == 0).then(|| "a".to_owned());
        assert_eq!(call.with_sql(&sql).sql(0).as_deref(), Some("a"));
        fn pick(c: &CallTypes<'_>) -> Vec<InputType> {
            vec![InputType::Exact(
                c.data_type(0).cloned().unwrap_or(DataType::Binary),
            )]
        }
        let built = Inputs::Built(InputBuilder(pick));
        assert_eq!(
            built.expected(&call),
            Some(vec![InputType::Exact(DataType::String)])
        );
        assert_eq!(built, Inputs::Built(InputBuilder(pick)));
        assert_eq!(format!("{built:?}"), "Built(InputBuilder)");
        assert_eq!(Inputs::Common.expected(&call), None);
        assert_eq!(
            Inputs::Expects(vec![InputType::Any]).expected(&call),
            Some(vec![InputType::Any])
        );
    }
}
