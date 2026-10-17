// SPDX-License-Identifier: Apache-2.0
//! Where a node came from: a DataFrame `plan_id`, a span of SQL text, or both.
//!
//! An origin is metadata. The canonical encoding and the structural hash
//! leave it out, so the same query written twice, or sent twice by a client
//! that numbers its plans afresh, has the same key.

use std::fmt;

/// A half-open byte range `start..end` into a SQL text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// Byte offset of the first byte.
    pub start: u32,
    /// Byte offset one past the last byte.
    pub end: u32,
}

impl Span {
    /// The span `start..end`.
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// Identifies the SQL text a [`Span`] indexes: a `spark.sql(...)` string, a
/// view body, a SQL function body. The front end that lowers the text
/// assigns it and keeps the texts it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(pub u32);

/// A DataFrame plan identifier: Spark Connect's `RelationCommon.plan_id`, or
/// the identifier the native Python and Rust APIs give each DataFrame.
///
/// Column references such as PySpark's `df["a"]` name the DataFrame they
/// belong to by its `PlanId` ([`crate::ColumnRef::plan_id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanId(pub i64);

impl fmt::Display for PlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The source of a plan or expression node.
///
/// Both parts are optional and independent: a `spark.sql(...)` root has the
/// Connect `plan_id` of the SQL relation and the span of the statement.
/// More sources (for example a Connect call site) may be added, so build
/// origins with the constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct Origin {
    /// The DataFrame plan identifier, for relations built by a DataFrame API.
    pub plan_id: Option<PlanId>,
    /// The SQL text and the span of the node in it, for nodes lowered from
    /// SQL.
    pub sql: Option<(SourceId, Span)>,
}

impl Origin {
    /// No source information (for example, a node built by a rewrite rule).
    pub const UNKNOWN: Self = Self {
        plan_id: None,
        sql: None,
    };

    /// A node built by a DataFrame API.
    #[must_use]
    pub const fn plan(id: PlanId) -> Self {
        Self {
            plan_id: Some(id),
            sql: None,
        }
    }

    /// A node lowered from `span` of the SQL text `source`.
    #[must_use]
    pub const fn sql(source: SourceId, span: Span) -> Self {
        Self {
            plan_id: None,
            sql: Some((source, span)),
        }
    }

    /// The same origin with a `plan_id`.
    #[must_use]
    pub const fn with_plan_id(mut self, id: PlanId) -> Self {
        self.plan_id = Some(id);
        self
    }

    /// The same origin with a SQL span.
    #[must_use]
    pub const fn with_sql(mut self, source: SourceId, span: Span) -> Self {
        self.sql = Some((source, span));
        self
    }
}
