// SPDX-License-Identifier: Apache-2.0
//! Canonical operator names and resolved function identities.
//!
//! Unresolved calls ([`crate::FunctionCall`]) name their function by
//! string, so the SQL and DataFrame front ends meet one function registry.
//! Operators use one canonical name each, the form PySpark's Spark Connect
//! client sends. Front ends lower other spellings to it:
//!
//! | Source | Canonical form |
//! |---|---|
//! | `a = b`, `a == b` | `==(a, b)` |
//! | `a <> b`, `a != b` | `not(==(a, b))` |
//! | `a <=> b`, `a IS NOT DISTINCT FROM b` | `<=>(a, b)` |
//! | `a IS DISTINCT FROM b` | `not(<=>(a, b))` |
//! | `a < b`, `<=`, `>`, `>=` | `<`, `<=`, `>`, `>=` |
//! | `a AND b`, `a OR b`, `NOT a` | `and`, `or`, `not` |
//! | `a + b`, `-`, `*`, `/`, `%` | `+`, `-`, `*`, `/`, `%` |
//! | `-a` | `negative(a)` |
//! | `a IS NULL`, `a IS NOT NULL` | `isnull(a)`, `isnotnull(a)` |
//! | `a IN (x, y)` | `in(a, x, y)` |
//! | `a BETWEEN x AND y` | `between(a, x, y)` |
//! | `a LIKE p`, `ILIKE`, `RLIKE` | `like(a, p)`, `ilike`, `rlike` |
//! | `a [NOT] LIKE ANY (p, ...)` and kin | one `*_any`/`*_all` call each, rewritten to `AND`/`OR` |
//! | `CASE WHEN c THEN v ... ELSE e END` | `when(c, v, ..., e)` |
//! | `a \|\| b` | `concat(a, b)` |
//!
//! A `NOT` in front of `IN`, `BETWEEN`, `LIKE`, `EXISTS` or `IN (subquery)`
//! becomes `not(...)` around the canonical form.
//!
//! Resolved calls name their function by a typed [`FunctionId`] and say its
//! [`FunctionKind`], so rules match on values, never on strings.

/// The canonical operators. [`Builtin::name`] is the canonical name an
/// unresolved call uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Builtin {
    /// `==`.
    Eq,
    /// `<=>` (null-safe equality).
    EqNullSafe,
    /// `<`.
    Lt,
    /// `<=`.
    LtEq,
    /// `>`.
    Gt,
    /// `>=`.
    GtEq,
    /// `and`.
    And,
    /// `or`.
    Or,
    /// `not`.
    Not,
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
    /// `div`: integral division, a `bigint`.
    IntegralDivide,
    /// `negative` (unary minus).
    Negative,
    /// `isnull`.
    IsNull,
    /// `isnotnull`.
    IsNotNull,
    /// `in`: the value first, then the list.
    In,
    /// `between`: value, lower bound, upper bound.
    Between,
    /// `like`.
    Like,
    /// `ilike`.
    ILike,
    /// `rlike`.
    RLike,
    /// `x LIKE ANY (...)`: an OR of `LIKE`s, as the analyzer rewrites it.
    LikeAny,
    /// `x LIKE ALL (...)`: an AND of `LIKE`s, as the analyzer rewrites it.
    LikeAll,
    /// `x NOT LIKE ANY (...)`: an OR of `NOT LIKE`s.
    NotLikeAny,
    /// `x NOT LIKE ALL (...)`: an AND of `NOT LIKE`s.
    NotLikeAll,
    /// `x ILIKE ANY (...)`: an OR of `ILIKE`s (lowered `LIKE`s).
    ILikeAny,
    /// `x ILIKE ALL (...)`: an AND of `ILIKE`s (lowered `LIKE`s).
    ILikeAll,
    /// `x NOT ILIKE ANY (...)`: an OR of `NOT ILIKE`s.
    NotILikeAny,
    /// `x NOT ILIKE ALL (...)`: an AND of `NOT ILIKE`s.
    NotILikeAll,
    /// `when`: condition/value pairs, then an optional `ELSE` value.
    When,
    /// `concat`.
    Concat,
    /// A date plus a year-month interval (Spark's `DateAddYMInterval`).
    DateAddYmInterval,
    /// A timestamp plus a year-month interval (`TimestampAddYMInterval`).
    TimestampAddYmInterval,
    /// A timestamp plus a day-time interval (`TimestampAddInterval`).
    TimestampAddInterval,
    /// A date plus a calendar interval (`DateAddInterval`); its third argument
    /// is Spark's `ansiEnabled`, a boolean literal, as for the calendar operators
    /// below.
    DateAddInterval,
    /// A calendar interval times a number (`MultiplyInterval`).
    MultiplyInterval,
    /// A calendar interval over a number (`DivideInterval`).
    DivideInterval,
    /// `+` of two calendar intervals.
    CalendarAdd,
    /// `-` of two calendar intervals.
    CalendarSubtract,
    /// Unary `-` of a calendar interval; its second argument is the flag.
    CalendarNegate,
    /// How Spark's `DatetimeSub` shows `l - r` in a name; only ever in a
    /// bound expression's shown form, never evaluated.
    DatetimeSub,
    /// `date - date` (`SubtractDates`).
    SubtractDates,
    /// `timestamp - timestamp` (`SubtractTimestamps`).
    SubtractTimestamps,
    /// The whole days of a day-time interval (`ExtractANSIIntervalDays`).
    ExtractAnsiIntervalDays,
    /// Its argument, NULL in a row where evaluating it fails (`TryEval`, the
    /// `try_*` forms of interval arithmetic).
    TryEval,
}

/// The three axes of a [`Builtin`] `[NOT] LIKE|ILIKE ANY|ALL` operator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantifiedLike {
    /// `ILIKE` rather than `LIKE`.
    pub insensitive: bool,
    /// `NOT LIKE`: each pattern match is negated.
    pub negated: bool,
    /// `ALL` (an `AND` chain) rather than `ANY`/`SOME` (an `OR` chain).
    pub all: bool,
}

impl Builtin {
    /// Every operator, in declaration order.
    pub const ALL: [Self; 47] = [
        Self::Eq,
        Self::EqNullSafe,
        Self::Lt,
        Self::LtEq,
        Self::Gt,
        Self::GtEq,
        Self::And,
        Self::Or,
        Self::Not,
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Remainder,
        Self::IntegralDivide,
        Self::Negative,
        Self::IsNull,
        Self::IsNotNull,
        Self::In,
        Self::Between,
        Self::Like,
        Self::ILike,
        Self::RLike,
        Self::LikeAny,
        Self::LikeAll,
        Self::NotLikeAny,
        Self::NotLikeAll,
        Self::ILikeAny,
        Self::ILikeAll,
        Self::NotILikeAny,
        Self::NotILikeAll,
        Self::When,
        Self::Concat,
        Self::DateAddYmInterval,
        Self::TimestampAddYmInterval,
        Self::TimestampAddInterval,
        Self::DateAddInterval,
        Self::MultiplyInterval,
        Self::DivideInterval,
        Self::CalendarAdd,
        Self::CalendarSubtract,
        Self::CalendarNegate,
        Self::DatetimeSub,
        Self::SubtractDates,
        Self::SubtractTimestamps,
        Self::ExtractAnsiIntervalDays,
        Self::TryEval,
    ];

    /// Whether this operator only comes from the analyzer's rewrites of
    /// interval arithmetic: SQL cannot call it by name.
    #[must_use]
    pub const fn is_analyzer_internal(self) -> bool {
        matches!(
            self,
            Self::DateAddYmInterval
                | Self::TimestampAddYmInterval
                | Self::TimestampAddInterval
                | Self::DateAddInterval
                | Self::MultiplyInterval
                | Self::DivideInterval
                | Self::CalendarAdd
                | Self::CalendarSubtract
                | Self::CalendarNegate
                | Self::DatetimeSub
                | Self::SubtractDates
                | Self::SubtractTimestamps
                | Self::ExtractAnsiIntervalDays
                | Self::TryEval
        )
    }

    /// The canonical name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::EqNullSafe => "<=>",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Remainder => "%",
            Self::IntegralDivide => "div",
            Self::Negative => "negative",
            Self::IsNull => "isnull",
            Self::IsNotNull => "isnotnull",
            Self::In => "in",
            Self::Between => "between",
            Self::Like => "like",
            Self::ILike => "ilike",
            Self::RLike => "rlike",
            Self::LikeAny => "like_any",
            Self::LikeAll => "like_all",
            Self::NotLikeAny => "not_like_any",
            Self::NotLikeAll => "not_like_all",
            Self::ILikeAny => "ilike_any",
            Self::ILikeAll => "ilike_all",
            Self::NotILikeAny => "not_ilike_any",
            Self::NotILikeAll => "not_ilike_all",
            Self::When => "when",
            Self::Concat => "concat",
            Self::DateAddYmInterval => "date_add_ym_interval",
            Self::TimestampAddYmInterval => "timestamp_add_ym_interval",
            Self::TimestampAddInterval => "timestamp_add_interval",
            Self::DateAddInterval => "date_add_interval",
            Self::MultiplyInterval => "multiply_interval",
            Self::DivideInterval => "divide_interval",
            Self::CalendarAdd => "calendar_add",
            Self::CalendarSubtract => "calendar_subtract",
            Self::CalendarNegate => "calendar_negate",
            Self::DatetimeSub => "datetime_sub",
            Self::SubtractDates => "subtract_dates",
            Self::SubtractTimestamps => "subtract_timestamps",
            Self::ExtractAnsiIntervalDays => "extractansiintervaldays",
            Self::TryEval => "tryeval",
        }
    }

    /// The shape of a `[NOT] LIKE|ILIKE ANY|ALL` operator, `None` for any
    /// other. One mapping for the analyzer rewrite and the printers.
    #[must_use]
    pub const fn quantified_like(self) -> Option<QuantifiedLike> {
        let (insensitive, negated, all) = match self {
            Self::LikeAny => (false, false, false),
            Self::LikeAll => (false, false, true),
            Self::NotLikeAny => (false, true, false),
            Self::NotLikeAll => (false, true, true),
            Self::ILikeAny => (true, false, false),
            Self::ILikeAll => (true, false, true),
            Self::NotILikeAny => (true, true, false),
            Self::NotILikeAll => (true, true, true),
            _ => return None,
        };
        Some(QuantifiedLike {
            insensitive,
            negated,
            all,
        })
    }

    /// The operator with canonical name `name`, ASCII case-insensitively
    /// (Spark function names are case-insensitive). Non-canonical spellings
    /// such as `=` or `<>` are not accepted.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|b| b.name().eq_ignore_ascii_case(name))
    }
}

/// The identity of a resolved function.
///
/// A function is identified by its canonical name: [`Builtin::name`], the
/// registry's lower-case name, or a UDF's lower-case name and version. Only
/// the name (with the kind), and for a UDF its version, enter the canonical
/// encoding and the structural hash; a registry id is an in-process lookup
/// handle and never part of a key, so keys stay valid across builds whose
/// registries number functions differently.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum FunctionId {
    /// A canonical operator.
    Builtin(Builtin),
    /// An entry of the function registry.
    Registry {
        /// The canonical (lower-case) name: the identity.
        name: String,
        /// The registry's handle for this process; not part of any key.
        id: u32,
    },
    /// A session-registered native Rust UDF: its lower-case name and the
    /// implementation [`version`](krishiv_udf_sdk::ScalarUdf::version) the
    /// call was analyzed with.
    Udf {
        /// The canonical (lower-case) name: the identity, with `version`.
        name: String,
        /// The implementation version: part of the identity.
        version: u32,
    },
    /// A session-registered Python UDF (design P2-T059): a plan reference
    /// by name and registration version, never the callable (I1, I13). The
    /// worker resolves it through its UDF runtime; one without a runtime, or
    /// without that version, refuses the call (`KRISHIV_UDF_NOT_AVAILABLE`).
    /// The return type and determinism are the call's
    /// ([`ResolvedFunction`](crate::ResolvedFunction)).
    PythonUdf {
        /// The canonical (lower-case) name: the identity, with `version`.
        name: String,
        /// The registration version: re-registering a name bumps it.
        version: u32,
        /// How the worker converts and calls it (Spark's `PythonEvalType`).
        eval_type: PythonEvalType,
        /// Whether calls may run concurrently; `false` serialises them.
        thread_safe: bool,
    },
}

/// The Python UDF kinds a plan can name, with Spark's `PythonEvalType`
/// codes (`pyspark/util.py` L655-688, Spark 4.2.0; the same at 4.3.0).
/// The scalar kinds (101, 200, 250) run in a `PythonEval`; the relation
/// kinds (design P3-T085 §2) see a relation, not a column, and run in the
/// relation nodes ([`PythonUdfKind`]). Streaming-state kinds (208,
/// 211-214) are not here (P4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PythonEvalType {
    /// `@udf`, Arrow-batched (`SQL_ARROW_BATCHED_UDF`, 101): rows are
    /// converted to Python values and back per batch.
    ArrowBatched,
    /// A scalar `pandas_udf` (`SQL_SCALAR_PANDAS_UDF`, 200): pandas Series in
    /// and out.
    ScalarPandas,
    /// `groupBy().applyInPandas` (`SQL_GROUPED_MAP_PANDAS_UDF`, 201).
    GroupedMapPandas,
    /// A grouped-aggregate `pandas_udf` (`SQL_GROUPED_AGG_PANDAS_UDF`, 202).
    GroupedAggPandas,
    /// A grouped-aggregate `pandas_udf` over a window
    /// (`SQL_WINDOW_AGG_PANDAS_UDF`, 203).
    WindowAggPandas,
    /// A scalar iterator `pandas_udf` (`SQL_SCALAR_PANDAS_ITER_UDF`, 204).
    ScalarPandasIter,
    /// `mapInPandas` (`SQL_MAP_PANDAS_ITER_UDF`, 205).
    MapPandasIter,
    /// `cogroup().applyInPandas` (`SQL_COGROUPED_MAP_PANDAS_UDF`, 206).
    CogroupedMapPandas,
    /// `mapInArrow` (`SQL_MAP_ARROW_ITER_UDF`, 207).
    MapArrowIter,
    /// `groupBy().applyInArrow` (`SQL_GROUPED_MAP_ARROW_UDF`, 209).
    GroupedMapArrow,
    /// `cogroup().applyInArrow` (`SQL_COGROUPED_MAP_ARROW_UDF`, 210).
    CogroupedMapArrow,
    /// `applyInArrow` with iterator hints (`SQL_GROUPED_MAP_ARROW_ITER_UDF`,
    /// 215).
    GroupedMapArrowIter,
    /// `applyInPandas` with iterator hints
    /// (`SQL_GROUPED_MAP_PANDAS_ITER_UDF`, 216).
    GroupedMapPandasIter,
    /// A grouped-aggregate `pandas_udf` with iterator hints
    /// (`SQL_GROUPED_AGG_PANDAS_ITER_UDF`, 217).
    GroupedAggPandasIter,
    /// A scalar `arrow_udf` (`SQL_SCALAR_ARROW_UDF`, 250): `pyarrow.Array`
    /// in and out, no conversion.
    ScalarArrow,
    /// A scalar iterator `arrow_udf` (`SQL_SCALAR_ARROW_ITER_UDF`, 251).
    ScalarArrowIter,
    /// A grouped-aggregate `arrow_udf` (`SQL_GROUPED_AGG_ARROW_UDF`, 252).
    GroupedAggArrow,
    /// A grouped-aggregate `arrow_udf` over a window
    /// (`SQL_WINDOW_AGG_ARROW_UDF`, 253).
    WindowAggArrow,
    /// A grouped-aggregate `arrow_udf` with iterator hints
    /// (`SQL_GROUPED_AGG_ARROW_ITER_UDF`, 254).
    GroupedAggArrowIter,
    /// `@udtf` (`SQL_TABLE_UDF`, 300).
    TableUdf,
    /// `@udtf(useArrow=True)` (`SQL_ARROW_TABLE_UDF`, 301).
    ArrowTableUdf,
    /// `@arrow_udtf` (`SQL_ARROW_UDTF`, 302).
    ArrowUdtf,
}

/// What a Python UDF sees and returns, by [`PythonEvalType`] (design P3-T085
/// §2's table): which plan node runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PythonUdfKind {
    /// A column per argument in, a column out, row for row (`PythonEval`).
    Scalar,
    /// As [`Self::Scalar`], called once per partition with an iterator of
    /// batches.
    ScalarIter,
    /// A partition's batches in, any batches out (`MapPartitions`).
    Map,
    /// A group's rows in, any rows out (`GroupMap`).
    GroupedMap,
    /// A key's rows of two relations in, any rows out (`CoGroupMap`).
    CogroupedMap,
    /// A group's argument columns in, one value out (an `Aggregate`).
    GroupedAgg,
    /// A window partition's argument columns in, one value per row out (a
    /// `Window`).
    WindowAgg,
    /// A table function (`PythonTableFunction`, `Generate`).
    Udtf,
}

impl PythonEvalType {
    /// Every kind, in code order.
    pub const ALL: [Self; 22] = [
        Self::ArrowBatched,
        Self::ScalarPandas,
        Self::GroupedMapPandas,
        Self::GroupedAggPandas,
        Self::WindowAggPandas,
        Self::ScalarPandasIter,
        Self::MapPandasIter,
        Self::CogroupedMapPandas,
        Self::MapArrowIter,
        Self::GroupedMapArrow,
        Self::CogroupedMapArrow,
        Self::GroupedMapArrowIter,
        Self::GroupedMapPandasIter,
        Self::GroupedAggPandasIter,
        Self::ScalarArrow,
        Self::ScalarArrowIter,
        Self::GroupedAggArrow,
        Self::WindowAggArrow,
        Self::GroupedAggArrowIter,
        Self::TableUdf,
        Self::ArrowTableUdf,
        Self::ArrowUdtf,
    ];

    /// Spark's `PythonEvalType` code.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::ArrowBatched => 101,
            Self::ScalarPandas => 200,
            Self::GroupedMapPandas => 201,
            Self::GroupedAggPandas => 202,
            Self::WindowAggPandas => 203,
            Self::ScalarPandasIter => 204,
            Self::MapPandasIter => 205,
            Self::CogroupedMapPandas => 206,
            Self::MapArrowIter => 207,
            Self::GroupedMapArrow => 209,
            Self::CogroupedMapArrow => 210,
            Self::GroupedMapArrowIter => 215,
            Self::GroupedMapPandasIter => 216,
            Self::GroupedAggPandasIter => 217,
            Self::ScalarArrow => 250,
            Self::ScalarArrowIter => 251,
            Self::GroupedAggArrow => 252,
            Self::WindowAggArrow => 253,
            Self::GroupedAggArrowIter => 254,
            Self::TableUdf => 300,
            Self::ArrowTableUdf => 301,
            Self::ArrowUdtf => 302,
        }
    }

    /// The kind with Spark's code `code`, if a plan can name it.
    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.code() == code)
    }

    /// What the UDF sees and returns.
    #[must_use]
    pub const fn kind(self) -> PythonUdfKind {
        match self {
            Self::ArrowBatched | Self::ScalarPandas | Self::ScalarArrow => PythonUdfKind::Scalar,
            Self::ScalarPandasIter | Self::ScalarArrowIter => PythonUdfKind::ScalarIter,
            Self::MapPandasIter | Self::MapArrowIter => PythonUdfKind::Map,
            Self::GroupedMapPandas
            | Self::GroupedMapArrow
            | Self::GroupedMapArrowIter
            | Self::GroupedMapPandasIter => PythonUdfKind::GroupedMap,
            Self::CogroupedMapPandas | Self::CogroupedMapArrow => PythonUdfKind::CogroupedMap,
            Self::GroupedAggPandas
            | Self::GroupedAggPandasIter
            | Self::GroupedAggArrow
            | Self::GroupedAggArrowIter => PythonUdfKind::GroupedAgg,
            Self::WindowAggPandas | Self::WindowAggArrow => PythonUdfKind::WindowAgg,
            Self::TableUdf | Self::ArrowTableUdf | Self::ArrowUdtf => PythonUdfKind::Udtf,
        }
    }

    /// Whether a `PythonEval` evaluates it (101, 200, 250): the kinds of
    /// design P2-T059. Every other kind is a relation kind (design P3-T085).
    #[must_use]
    pub const fn is_scalar(self) -> bool {
        matches!(self.kind(), PythonUdfKind::Scalar)
    }
}

/// A reference to a session-registered Python UDF run by a relation node
/// (design P3-T085 §3.1): the same identity as
/// [`FunctionId::PythonUdf`], never the callable (I13).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PythonUdfRef {
    /// The canonical (lower-case) name: the identity, with `version`.
    pub name: String,
    /// The registration version.
    pub version: u32,
    /// How the worker converts and calls it.
    pub eval_type: PythonEvalType,
    /// Whether calls may run concurrently; `false` serialises them.
    pub thread_safe: bool,
    /// Whether the UDF gives the same rows for the same input (Spark's
    /// `PythonUDF.udfDeterministic`): a relation over one that does not is
    /// nondeterministic ([`crate::python_relation_is_deterministic`]), so a
    /// CTE holding it is computed once, not once per reference. Not part of
    /// the identity, as a scalar call's determinism is the call's
    /// ([`ResolvedFunction`](crate::ResolvedFunction)).
    pub deterministic: bool,
}

impl PythonUdfRef {
    /// The function identity it stands for.
    #[must_use]
    pub fn function_id(&self) -> FunctionId {
        FunctionId::PythonUdf {
            name: self.name.clone(),
            version: self.version,
            eval_type: self.eval_type,
            thread_safe: self.thread_safe,
        }
    }

    /// The reference of a [`FunctionId::PythonUdf`] called with
    /// `deterministic` (the call's determinism); `None` for any other
    /// function.
    #[must_use]
    pub fn of(id: &FunctionId, deterministic: bool) -> Option<Self> {
        match id {
            FunctionId::PythonUdf {
                name,
                version,
                eval_type,
                thread_safe,
            } => Some(Self {
                name: name.clone(),
                version: *version,
                eval_type: *eval_type,
                thread_safe: *thread_safe,
                deterministic,
            }),
            _ => None,
        }
    }
}

impl FunctionId {
    /// The canonical name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Builtin(b) => b.name(),
            Self::Registry { name, .. } | Self::Udf { name, .. } | Self::PythonUdf { name, .. } => {
                name
            }
        }
    }
}

pub use krishiv_functions::signature::FunctionKind;

/// The name a call of `name` (written under `qualifier`) prints with. A name
/// starting with NUL is an encoded name (an unregistered Python UDF's call,
/// `krishiv-session` `python_udfs.rs`: fields separated by NUL, the shown
/// name last): it prints as that last field, unqualified, as Spark's
/// `PythonUDF` prints `name(args)` (review 2c I2). Any other name prints as
/// written, qualified.
pub(crate) fn shown_call_name(qualifier: &str, name: &str) -> String {
    if name.starts_with('\0') {
        return name.rsplit('\0').next().unwrap_or_default().to_owned();
    }
    if qualifier.is_empty() {
        name.to_owned()
    } else {
        format!("{qualifier}.{name}")
    }
}

#[cfg(test)]
mod shown_name_tests {
    use super::shown_call_name;

    #[test]
    fn an_encoded_name_shows_its_last_field_unqualified_and_others_as_written() {
        assert_eq!(shown_call_name("session", "\0tag\0x\0<lambda>"), "<lambda>");
        assert_eq!(shown_call_name("", "\0tag\0\0f"), "f");
        assert_eq!(shown_call_name("", "upper"), "upper");
        assert_eq!(shown_call_name("db", "f"), "db.f");
    }
}
