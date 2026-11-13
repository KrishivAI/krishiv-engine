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
//! side table ([`ViewExpansion`]) instead of an IR node. A view whose base tables were authorized for the
//! caller (temporary and `INVOKER` views) is expanded through, and its sources record
//! [`via_view`](Source::via_view); any other (`DEFINER`, T062) is a [`Leaf::View`] with transform
//! [`Transform::View`] and the warning [`LineageWarning::ViewNotExpanded`], and nothing under it appears.

use std::collections::BTreeSet;
use std::fmt;

use krishiv_common::error::KrishivError;

use super::AnalyzedPlan;
use crate::plan::{LogicalPlan, PlanNode};
use crate::{AttrId, Origin};

mod exprs;
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
    /// A `CASE WHEN` or `IF` condition.
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

/// How a view's base tables are authorized (T062 sets [`Self::Definer`]; today every permanent view is
/// read as its invoker reads its query).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ViewSecurity {
    /// A temporary view: its query is the caller's.
    Temporary,
    /// `SQL SECURITY INVOKER`: base tables are checked against the caller.
    Invoker,
    /// `SQL SECURITY DEFINER`: base tables are checked against the view's owner.
    Definer,
}

/// The view side table: one view read the expansion inlined, recorded by view inlining (there is no IR node
/// for it), and located in the resolved plan by [`locate_views`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewExpansion {
    /// The view's name parts.
    pub name: Vec<String>,
    /// The alias of the `SubqueryAlias` the inlining built.
    pub alias: String,
    /// That node's origin (the read's), which resolution keeps.
    pub origin: Origin,
    /// How its base tables are authorized.
    pub security: ViewSecurity,
    /// Whether its base securables were authorized for the caller.
    pub bases_authorized: bool,
    /// The output attributes of the inlined subtree in the resolved plan; `None` until located (or when
    /// resolution left no node for it).
    pub output: Option<Vec<AttrId>>,
}

impl ViewExpansion {
    /// Whether lineage expands through the view: only views whose base tables were checked against the
    /// caller (temporary and `INVOKER` views), never a `DEFINER` view.
    #[must_use]
    pub fn expands(&self) -> bool {
        match self.security {
            ViewSecurity::Temporary | ViewSecurity::Invoker => self.bases_authorized,
            ViewSecurity::Definer => false,
        }
    }
}

/// `views` with their [`output`](ViewExpansion::output) set from the resolved plan `root`: the outputs of
/// the `SubqueryAlias` nodes with the view's alias and origin. A view resolution left no node for keeps
/// `None`.
#[must_use]
pub fn locate_views(root: &LogicalPlan, mut views: Vec<ViewExpansion>) -> Vec<ViewExpansion> {
    if views.is_empty() {
        return views;
    }
    let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(root), |node| {
        if let crate::walk::Node::Plan(p) = node
            && let PlanNode::SubqueryAlias { alias, .. } = &p.node
        {
            for v in views
                .iter_mut()
                .filter(|v| v.output.is_none() && v.alias == *alias && v.origin == p.origin)
            {
                v.output = p
                    .resolved_output()
                    .map(|out| out.into_iter().map(|c| c.id).collect());
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    views
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

/// [`lineage`], and the number of (node, `Shift` offset) pairs the walk computed (each at most once).
#[cfg(test)]
pub(crate) fn lineage_counting(analyzed: &AnalyzedPlan) -> Result<(Lineage, usize), KrishivError> {
    walk::run(analyzed)
}
