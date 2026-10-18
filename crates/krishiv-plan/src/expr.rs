// SPDX-License-Identifier: Apache-2.0
//! Expression nodes, unresolved and resolved.

use std::fmt;
use std::sync::Arc;

use krishiv_types::DataType;

use crate::dag::{self, Memo};
use crate::walk::{self, Node};
use crate::{Builtin, FunctionId, FunctionKind, Literal, LogicalPlan, Origin, PlanId};

/// An expression: a node and its [`Origin`].
///
/// Children are `Arc<Expr>`, so rebuilding a node shares its unchanged
/// subexpressions. `PartialEq` compares everything, origins included; the
/// structural comparison that ignores origins is the canonical encoding
/// (`Plan::structurally_eq`). Both take time linear in the distinct nodes
/// when subexpressions are shared, and `Debug` prints a shared
/// subexpression once. Dropping an expression never recurses, so a
/// too-deep tree can still be rejected and dropped ([`crate::check_depth`]).
#[derive(Clone)]
pub struct Expr {
    /// What the expression is.
    pub node: ExprNode,
    /// Where it came from.
    pub origin: Origin,
}

/// The expression node kinds.
///
/// Operators are [`ExprNode::Function`] calls with canonical names (see
/// [`crate::function`]), so SQL and DataFrame front ends meet one function
/// registry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExprNode {
    /// An unresolved column reference (`a`, `t.a`, PySpark `df["a"]`).
    Column(ColumnRef),
    /// A literal value.
    Literal(Literal),
    /// An unresolved `*` or `t.*`.
    Star(Star),
    /// An unresolved function call, operators included.
    Function(FunctionCall),
    /// `CAST(expr AS to)`; `TRY_CAST` is [`CastMode::Try`].
    Cast {
        /// The value to cast.
        expr: Arc<Expr>,
        /// The target type.
        to: DataType,
        /// Error behaviour on invalid input.
        mode: CastMode,
    },
    /// An unresolved `expr AS name`.
    Alias {
        /// The aliased expression.
        expr: Arc<Expr>,
        /// The output name.
        name: String,
    },
    /// A subquery used as an expression.
    Subquery {
        /// Scalar, `EXISTS` or `IN`.
        kind: SubqueryKind,
        /// The subquery plan.
        plan: Arc<LogicalPlan>,
    },
    /// An expression bound by the analyzer.
    Resolved(ResolvedExpr),
}

/// An unresolved column reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnRef {
    /// The name parts, e.g. `["t", "a"]` for `t.a`.
    pub name: Vec<String>,
    /// The DataFrame the column belongs to, for PySpark's `df["a"]`. The
    /// analyzer resolves it among the relations carrying this `plan_id` in
    /// the subtree of the operator that holds the reference (then in
    /// enclosing queries, for a reference inside a subquery). The canonical
    /// encoding records all of those carriers, so plans with equal keys
    /// resolve the reference alike, whichever rule picks among them.
    pub plan_id: Option<PlanId>,
}

/// An unresolved `*` (empty qualifier) or `t.*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Star {
    /// The qualifier parts; empty for a bare `*`.
    pub qualifier: Vec<String>,
    /// The DataFrame the star expands, for PySpark's `df["*"]`; resolved
    /// as for [`ColumnRef::plan_id`].
    pub plan_id: Option<PlanId>,
}

/// An unresolved function call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    /// The function name; operators use their canonical names.
    pub name: String,
    /// The arguments.
    pub args: Vec<Arc<Expr>>,
    /// `f(DISTINCT ...)`.
    pub distinct: bool,
    /// `f(...) FILTER (WHERE filter)`.
    pub filter: Option<Arc<Expr>>,
}

/// How a cast handles invalid input.
///
/// [`CastMode::Default`] appears only in unresolved plans: the analyzer
/// pins it to [`CastMode::Ansi`] or [`CastMode::Legacy`] from
/// `spark.sql.ansi.enabled`, so a resolved plan's key does not depend on
/// configuration outside the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CastMode {
    /// As `spark.sql.ansi.enabled` says (unresolved plans only).
    #[default]
    Default,
    /// Non-ANSI: invalid input becomes `NULL`.
    Legacy,
    /// ANSI: invalid input is an error.
    Ansi,
    /// `TRY_CAST`: invalid input becomes `NULL`, even under ANSI.
    Try,
}

/// The kind of a subquery expression.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SubqueryKind {
    /// A scalar subquery: one row, one column.
    Scalar,
    /// `EXISTS (subquery)`.
    Exists,
    /// `(values) IN (subquery)`.
    In(Vec<Arc<Expr>>),
}

/// A sort key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortOrder {
    /// The key.
    pub expr: Arc<Expr>,
    /// Ascending or descending.
    pub direction: SortDirection,
    /// Where nulls go.
    pub nulls: NullOrdering,
}

impl SortOrder {
    /// `expr ASC NULLS FIRST`, Spark's default for `ASC`.
    #[must_use]
    pub fn asc(expr: impl Into<Arc<Expr>>) -> Self {
        Self {
            expr: expr.into(),
            direction: SortDirection::Ascending,
            nulls: NullOrdering::NullsFirst,
        }
    }

    /// `expr DESC NULLS LAST`, Spark's default for `DESC`.
    #[must_use]
    pub fn desc(expr: impl Into<Arc<Expr>>) -> Self {
        Self {
            expr: expr.into(),
            direction: SortDirection::Descending,
            nulls: NullOrdering::NullsLast,
        }
    }
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortDirection {
    /// `ASC`.
    Ascending,
    /// `DESC`.
    Descending,
}

/// Placement of nulls in a sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NullOrdering {
    /// `NULLS FIRST`.
    NullsFirst,
    /// `NULLS LAST`.
    NullsLast,
}

/// The identity of a resolved attribute (Spark's `ExprId`).
///
/// The analyzer numbers attributes deterministically per plan, so resolving
/// the same plan twice yields the same identifiers. Identifiers are unique
/// within a plan but not dense (a self-join instance reserves a span):
/// never index by them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttrId(pub u64);

/// A resolved attribute: a column of some plan's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundColumn {
    /// Identity, unique within a resolved plan.
    pub id: AttrId,
    /// Name, for display and `*` expansion.
    pub name: String,
    /// Spark type.
    pub data_type: DataType,
    /// Whether it may be null.
    pub nullable: bool,
}

/// A function call bound by the analyzer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFunction {
    /// Which function.
    pub id: FunctionId,
    /// Scalar, aggregate, window or generator.
    pub kind: FunctionKind,
    /// The arguments, after implicit casts.
    pub args: Vec<Arc<Expr>>,
    /// `f(DISTINCT ...)`.
    pub distinct: bool,
    /// `f(...) FILTER (WHERE filter)`.
    pub filter: Option<Arc<Expr>>,
    /// Result type.
    pub data_type: DataType,
    /// Whether the result may be null.
    pub nullable: bool,
    /// Whether the call is deterministic (from the function registry; the
    /// canonical operators are). Not part of the canonical encoding: it
    /// follows from the function's identity in a registry version, which
    /// the analysis fingerprint covers.
    pub deterministic: bool,
}

/// Expression forms produced by the analyzer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResolvedExpr {
    /// A column of the input plans.
    Column(BoundColumn),
    /// A column of an enclosing query, inside a correlated subquery.
    OuterColumn(BoundColumn),
    /// A function bound to its identity and result type.
    Function(ResolvedFunction),
    /// `expr AS name`, producing a new attribute.
    Alias {
        /// The aliased expression.
        expr: Arc<Expr>,
        /// The attribute it produces.
        output: BoundColumn,
    },
}

impl Expr {
    /// An expression of unknown origin.
    #[must_use]
    pub const fn new(node: ExprNode) -> Self {
        Self {
            node,
            origin: Origin::UNKNOWN,
        }
    }

    /// The same expression with `origin`.
    #[must_use]
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// An unresolved column reference from its name parts.
    #[must_use]
    pub fn column<S: Into<String>>(parts: impl IntoIterator<Item = S>) -> Self {
        Self::new(ExprNode::Column(ColumnRef {
            name: parts.into_iter().map(Into::into).collect(),
            plan_id: None,
        }))
    }

    /// A literal.
    #[must_use]
    pub const fn literal(value: Literal) -> Self {
        Self::new(ExprNode::Literal(value))
    }

    /// An unresolved, non-distinct, unfiltered call of `name`.
    #[must_use]
    pub fn function(name: impl Into<String>, args: Vec<Self>) -> Self {
        Self::new(ExprNode::Function(FunctionCall {
            name: name.into(),
            args: args.into_iter().map(Arc::new).collect(),
            distinct: false,
            filter: None,
        }))
    }

    /// An unresolved call of a canonical operator.
    #[must_use]
    pub fn builtin(op: Builtin, args: Vec<Self>) -> Self {
        Self::function(op.name(), args)
    }

    /// `self AS name`.
    #[must_use]
    pub fn alias(self, name: impl Into<String>) -> Self {
        Self::new(ExprNode::Alias {
            expr: Arc::new(self),
            name: name.into(),
        })
    }

    /// `CAST(self AS to)` in the default mode.
    #[must_use]
    pub fn cast(self, to: DataType) -> Self {
        Self::new(ExprNode::Cast {
            expr: Arc::new(self),
            to,
            mode: CastMode::Default,
        })
    }

    /// A subquery expression.
    #[must_use]
    pub fn subquery(kind: SubqueryKind, plan: LogicalPlan) -> Self {
        Self::new(ExprNode::Subquery {
            kind,
            plan: Arc::new(plan),
        })
    }

    /// Whether no unresolved node remains, subquery plans included. It does
    /// not recurse.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        walk::all_resolved(Node::Expr(self))
    }

    /// Calls `f` on each direct child expression, in order. Subquery plans
    /// are not expressions and are not visited.
    pub fn for_each_child<'a>(&'a self, f: &mut impl FnMut(&'a Arc<Self>)) {
        match &self.node {
            ExprNode::Column(_)
            | ExprNode::Literal(_)
            | ExprNode::Star(_)
            | ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_)) => {}
            ExprNode::Function(FunctionCall { args, filter, .. })
            | ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                args, filter, ..
            })) => {
                args.iter().for_each(&mut *f);
                filter.iter().for_each(f);
            }
            ExprNode::Cast { expr, .. }
            | ExprNode::Alias { expr, .. }
            | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => f(expr),
            ExprNode::Subquery { kind, .. } => match kind {
                SubqueryKind::In(values) => values.iter().for_each(f),
                SubqueryKind::Scalar | SubqueryKind::Exists => {}
            },
        }
    }

    /// Calls `f` on each direct child expression slot, in the order of
    /// [`Self::for_each_child`].
    pub(crate) fn for_each_child_mut(&mut self, f: &mut dyn FnMut(&mut Arc<Self>)) {
        match &mut self.node {
            ExprNode::Column(_)
            | ExprNode::Literal(_)
            | ExprNode::Star(_)
            | ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_)) => {}
            ExprNode::Function(FunctionCall { args, filter, .. })
            | ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                args, filter, ..
            })) => {
                args.iter_mut().for_each(&mut *f);
                filter.iter_mut().for_each(f);
            }
            ExprNode::Cast { expr, .. }
            | ExprNode::Alias { expr, .. }
            | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => f(expr),
            ExprNode::Subquery { kind, .. } => match kind {
                SubqueryKind::In(values) => values.iter_mut().for_each(f),
                SubqueryKind::Scalar | SubqueryKind::Exists => {}
            },
        }
    }

    /// Replaces each direct child by `f(child)`. Returns `self` itself
    /// (the same `Arc`) when every child comes back unchanged
    /// (`Arc::ptr_eq`), so unchanged trees are never copied.
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn map_children<E>(
        self: Arc<Self>,
        mut f: impl FnMut(Arc<Self>) -> Result<Arc<Self>, E>,
    ) -> Result<Arc<Self>, E> {
        let mut old = Vec::new();
        self.for_each_child(&mut |c| old.push(Arc::clone(c)));
        let mut new = Vec::with_capacity(old.len());
        let mut changed = false;
        for child in old {
            let next = f(Arc::clone(&child))?;
            changed |= !Arc::ptr_eq(&next, &child);
            new.push(next);
        }
        if !changed {
            return Ok(self);
        }
        let mut expr = Arc::unwrap_or_clone(self);
        let mut new = new.into_iter();
        expr.for_each_child_mut(&mut |slot| {
            if let Some(next) = new.next() {
                *slot = next;
            }
        });
        Ok(Arc::new(expr))
    }

    /// Applies `f` to every node, children first. Subquery plans are not
    /// entered. A node shared by several parents (`Arc`) is rewritten once
    /// and its rewrite shared, so `f` runs once per distinct node. Recurses
    /// once per level: call it only on expressions within
    /// [`crate::MAX_DEPTH`].
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn transform_up<E>(
        self: Arc<Self>,
        f: &mut impl FnMut(Arc<Self>) -> Result<Arc<Self>, E>,
    ) -> Result<Arc<Self>, E> {
        self.up(f, &mut Memo::new())
    }

    fn up<E>(
        self: Arc<Self>,
        f: &mut impl FnMut(Arc<Self>) -> Result<Arc<Self>, E>,
        done: &mut Memo<Self>,
    ) -> Result<Arc<Self>, E> {
        if let Some((_, out)) = done.get(&Arc::as_ptr(&self)) {
            return Ok(Arc::clone(out));
        }
        let rebuilt = Arc::clone(&self).map_children(|c| c.up(f, done))?;
        let out = f(rebuilt)?;
        done.insert(Arc::as_ptr(&self), (self, Arc::clone(&out)));
        Ok(out)
    }

    /// Applies `f` to every node, parents first; see [`Self::transform_up`].
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn transform_down<E>(
        self: Arc<Self>,
        f: &mut impl FnMut(Arc<Self>) -> Result<Arc<Self>, E>,
    ) -> Result<Arc<Self>, E> {
        self.down(f, &mut Memo::new())
    }

    fn down<E>(
        self: Arc<Self>,
        f: &mut impl FnMut(Arc<Self>) -> Result<Arc<Self>, E>,
        done: &mut Memo<Self>,
    ) -> Result<Arc<Self>, E> {
        if let Some((_, out)) = done.get(&Arc::as_ptr(&self)) {
            return Ok(Arc::clone(out));
        }
        let out = f(Arc::clone(&self))?.map_children(|c| c.down(f, done))?;
        done.insert(Arc::as_ptr(&self), (self, Arc::clone(&out)));
        Ok(out)
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        dag::eq(Node::Expr(self), Node::Expr(other), || {
            self.origin == other.origin && self.node == other.node
        })
    }
}

impl Eq for Expr {}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        dag::debug(Node::Expr(self), f, "Expr", |f, name| {
            f.debug_struct(name)
                .field("node", &self.node)
                .field("origin", &self.origin)
                .finish()
        })
    }
}

impl From<Literal> for Expr {
    fn from(value: Literal) -> Self {
        Self::literal(value)
    }
}

impl Drop for Expr {
    fn drop(&mut self) {
        walk::teardown_expr(self);
    }
}
