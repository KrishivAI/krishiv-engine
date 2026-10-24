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
}

impl Builtin {
    /// Every operator, in declaration order.
    pub const ALL: [Self; 33] = [
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
    ];

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
        }
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
/// A function is identified by its canonical name: [`Builtin::name`] or the
/// registry's lower-case name. Only the name (with the kind) enters the
/// canonical encoding and the structural hash; a registry id is an
/// in-process lookup handle and never part of a key, so keys stay valid
/// across builds whose registries number functions differently.
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
}

impl FunctionId {
    /// The canonical name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Builtin(b) => b.name(),
            Self::Registry { name, .. } => name,
        }
    }
}

pub use krishiv_functions::signature::FunctionKind;
