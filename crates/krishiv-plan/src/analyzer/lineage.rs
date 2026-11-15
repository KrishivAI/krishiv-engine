// SPDX-License-Identifier: Apache-2.0
//! Column lineage of an analyzed plan (P3-T063 unit N1a, design `P3-T063-analyze-api.md` §2 and §3.1).
//!
//! [`lineage`] is pure: it reads an [`AnalyzedPlan`] (the resolved plan and its view side table,
//! [`AnalyzedPlan::views`]) and returns, per output ordinal, the leaves the value derives from and the
//! strongest step on any path ([`Transform`]), plus the statement's [`filters`](Lineage::filters): the
//! columns that decide *which* rows or groups, not the value. It never runs anything and never reads a row.
//!
//! The vocabulary (the [`Leaf`] kinds, [`Transform`], [`FilterKind`], [`LineageWarning`]) lives in this
//! file only, so a change of the design's open answers (Q2, Q4) stays local.
//!
//! The walk (`lineage/walk.rs`) matches every [`PlanNode`](crate::PlanNode) and expression kind without a
//! wildcard arm (`#[deny(clippy::wildcard_enum_match_arm)]`), so a new node kind does not compile until it
//! states its rule. The Python relation nodes of T085 (`MapPartitions`, `GroupMap`, `CoGroupMap`,
//! `PythonTableFunction`, `GeneratorKind::PythonUdtf`) are not on main yet: PR-26a or PR-26c adds their rows.
//!
//! Views (design Q4): permanent and temporary views are inlined before binding, so view inlining records a
//! side table ([`ViewExpansion`](super::views::ViewExpansion)) instead of an IR node. A `SubqueryAlias` is a
//! view read when an entry has its (alias, origin); ids play no part, so a view inlined twice (`df.join(df)`)
//! or renumbered is still found. A view whose base tables were authorized for the caller (temporary and
//! `INVOKER` views) is expanded through, and its sources record [`via_view`](Source::via_view); any other
//! (`DEFINER`, T062) is a [`Leaf::View`] with transform [`Transform::View`] and the warning
//! [`LineageWarning::ViewNotExpanded`], and nothing under it appears. When entries of one (alias, origin)
//! disagree, the node is not expanded. A view that is not expanded and that no node carries fails the pass
//! (fail closed): its base tables could otherwise show.
//!
//! Rules where the design's table leaves a choice (design v3):
//! - a condition that decides a value (`CASE WHEN`, `IF`, an `IN` subquery's values, an existence join's
//!   condition) is a source of that value (`expression`) and a `conditional` (or `join`) filter;
//! - every set operation (`UNION`, `INTERSECT`, `EXCEPT`, `DISTINCT` and `ALL`) gives an output column the
//!   union of both sides' column at its position, `identity`; the `join` filters of the rewrite stay;
//! - `sources: []` is not always `literal`/`generated`: `count(*)` is `aggregate`, `rand()` `expression`,
//!   an uncorrelated `EXISTS` `expression`.
//!
//! The walk evaluates each expression once per (node, `Shift` offset). A view's `via_view` tag rewrites the
//! walk's map for the view's ids, so the invariant is that no expression node is shared across a view
//! boundary (a view's body is lowered on its own, so it holds today); a rewrite that shares expressions
//! across one must key the memo by the view too.

use std::collections::BTreeSet;
use std::fmt;

use krishiv_common::error::KrishivError;

use super::AnalyzedPlan;

mod exprs;
pub mod schema;
mod walk;

/// Where a value comes from: a column of a catalog table, of files read by path, or of a view that is
/// not expanded.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Leaf {
    /// A column of a catalog table (or a session table), by its name parts.
    Table {
        /// The table's name parts, as the scan names it (`["spark_catalog", "default", "t"]`).
        name: Vec<String>,
        /// The column.
        column: String,
    },
    /// A column of files read by path (`Scan` of `ReadSource::Files`); the hidden `_metadata` column's
    /// fields are leaves too (`_metadata.file_path`, ...).
    Path {
        /// The path, redacted as every securable path prints (`OptionValue`'s `Display`: a URL's
        /// credential parts are `*********(redacted)`).
        url: String,
        /// The column.
        column: String,
    },
    /// A column of a view that is not expanded (a `DEFINER` view): nothing under it is shown.
    View {
        /// The view's name parts.
        name: Vec<String>,
        /// The column.
        column: String,
    },
}

/// One source of an output column: a leaf, and the outermost expanded view it was read through.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Source {
    /// The leaf.
    pub leaf: Leaf,
    /// The outermost expanded view (temporary or `INVOKER`) the leaf was read through, by its name parts.
    pub via_view: Option<Vec<String>>,
}

/// How an output column derives from its sources: the strongest step on any path.
///
/// The variants are declared weakest first, and the derived order is the strength order:
/// `literal < identity < expression < generated < window < aggregate < opaque < masked < view`. Where paths
/// meet (a union, an expression over several columns) the strongest wins. `literal`, `generated`, `opaque`,
/// `view` and `masked` are the leaf-only kinds of the design: a value with no leaf is `literal` (or
/// `generated` for `range()`), a UDF hides its derivation (`opaque`), a `DEFINER` view hides its query
/// (`view`). `masked` is unused until P5-T045.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Transform {
    /// No leaf: a constant.
    Literal,
    /// The leaf's value, as it is.
    Identity,
    /// A row-wise expression over the sources.
    Expression,
    /// A generated row's value (`explode`, `range()`).
    Generated,
    /// A window function over the sources (arguments, partition and order keys).
    Window,
    /// An aggregate over the sources.
    Aggregate,
    /// A user-defined function: its derivation is unknown, its arguments are the sources.
    Opaque,
    /// A masked column (P5-T045; unused).
    Masked,
    /// A column of a view that is not expanded.
    View,
}

impl Transform {
    /// The stable name (`identity`, `expression`, ...).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Literal => "literal",
            Self::Identity => "identity",
            Self::Expression => "expression",
            Self::Generated => "generated",
            Self::Window => "window",
            Self::Aggregate => "aggregate",
            Self::Opaque => "opaque",
            Self::Masked => "masked",
            Self::View => "view",
        }
    }
}

impl fmt::Display for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why a column decides which rows or groups a statement returns (design Q2, a closed list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum FilterKind {
    /// `WHERE`, `HAVING`, `QUALIFY`; an aggregate's `FILTER (WHERE ...)`.
    Filter,
    /// A join condition, rewritten subquery conditions included (semi, anti, existence), and `IN`/`EXISTS`
    /// subqueries.
    Join,
    /// A grouping key.
    GroupBy,
    /// An `ORDER BY` / `SORT BY` key or a `DISTRIBUTE BY` expression.
    Sort,
    /// A window's partition or order key.
    Window,
    /// A `CASE WHEN` or `IF` condition (also a source of the value it decides).
    Conditional,
}

impl FilterKind {
    /// The stable name (`filter`, `join`, `group_by`, ...).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Filter => "filter",
            Self::Join => "join",
            Self::GroupBy => "group_by",
            Self::Sort => "sort",
            Self::Window => "window",
            Self::Conditional => "conditional",
        }
    }
}

impl fmt::Display for FilterKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A fixed warning code; never free text with user data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum LineageWarning {
    /// A user-defined function's derivation is unknown (its outputs are `opaque`).
    LineageOpaqueUdf,
    /// A view was not expanded (its columns are [`Leaf::View`]s).
    ViewNotExpanded,
}

impl LineageWarning {
    /// The stable code (`LINEAGE_OPAQUE_UDF`, `VIEW_NOT_EXPANDED`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::LineageOpaqueUdf => "LINEAGE_OPAQUE_UDF",
            Self::ViewNotExpanded => "VIEW_NOT_EXPANDED",
        }
    }
}

impl fmt::Display for LineageWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// The lineage of one output column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnLineage {
    /// The output position (`SELECT a, a` has two entries).
    pub ordinal: usize,
    /// The output column's name.
    pub name: String,
    /// The leaves the value depends on, sorted.
    pub sources: BTreeSet<Source>,
    /// The strongest step on any path.
    pub transform: Transform,
}

/// The lineage of a statement's result.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Lineage {
    /// One entry per output ordinal, in order.
    pub columns: Vec<ColumnLineage>,
    /// The columns that decide which rows or groups, sorted.
    pub filters: BTreeSet<(FilterKind, Leaf)>,
    /// Fixed warning codes, sorted.
    pub warnings: BTreeSet<LineageWarning>,
}

/// The column lineage of `analyzed` (see the module doc).
///
/// # Errors
/// - `INTERNAL_ERROR` for a node an analyzed plan never holds (an unresolved node, the optimizer's
///   `PartialAggregate`), a reference to an attribute nothing defines, or a view that is not expanded and
///   that resolution left no node for (fail closed);
/// - `FAILED_TO_PARSE_TOO_COMPLEX` when the plan nests deeper than [`crate::MAX_DEPTH`].
pub fn lineage(analyzed: &AnalyzedPlan) -> Result<Lineage, KrishivError> {
    walk::run(analyzed).map(|(l, _)| l)
}

/// [`lineage`], and the number of (node, `Shift` offset) pairs and of (expression, `Shift` offset) pairs
/// the walk computed (each at most once).
#[cfg(test)]
pub(crate) fn lineage_counting(
    analyzed: &AnalyzedPlan,
) -> Result<(Lineage, walk::Counts), KrishivError> {
    walk::run(analyzed)
}
