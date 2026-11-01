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

/// The kind of an unresolved typed datetime literal (`TIMESTAMP '...'`),
/// as the parser wrote it. The analyzer resolves it to a [`Literal`]
/// with the session's time zone and timestamp type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TypedLiteralKind {
    /// `DATE '...'`.
    Date,
    /// `TIMESTAMP '...'`.
    Timestamp,
    /// `TIMESTAMP_NTZ '...'`.
    TimestampNtz,
    /// `TIMESTAMP_LTZ '...'`.
    TimestampLtz,
    /// `TIME '...'`.
    Time,
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
    /// An unresolved typed datetime literal: the type keyword and the raw
    /// string, as written. The analyzer lowers it to a [`Literal`].
    TypedLiteral {
        /// Which type keyword was written.
        kind: TypedLiteralKind,
        /// The raw string, as written.
        value: String,
    },
    /// An unresolved `*` or `t.*`.
    Star(Star),
    /// An unresolved function call, operators included.
    Function(FunctionCall),
    /// `x -> body` or `(x, y) -> body`: a lambda, bound by the analyzer
    /// only as a direct argument of a higher-order function (`transform`,
    /// `filter`, `exists`, `forall`, ...), where its variables take the
    /// argument's element types (Spark's `NamedLambdaVariable`). The node
    /// stays in place when bound: its `params` are then the variables
    /// themselves (with stable [`AttrId`] identities), and its body holds
    /// [`ResolvedExpr::LambdaVar`] leaves reading them.
    Lambda {
        /// The variables, in parameter order (with their stable identities
        /// once bound; placeholders before).
        params: Vec<BoundColumn>,
        /// The body, resolved against the variables and the enclosing
        /// query's columns.
        body: Arc<Expr>,
    },
    /// An unresolved `expr.field`: the field `name` of the struct `base`
    /// (Spark's `GetStructField`). A `Column` base lowers to a longer
    /// [`ColumnRef`] instead; anything else becomes this node.
    Field {
        /// The struct value.
        base: Arc<Expr>,
        /// The field name, as written.
        name: String,
    },
    /// An unresolved struct update: the fields `ops` adds to, replaces in
    /// or drops from the struct `base` (Spark's `UpdateFields`, built by
    /// `Column.withField`/`Column.dropFields`). The analyzer rewrites it to
    /// a `CASE WHEN base IS NULL ... ELSE named_struct(...) END` over field
    /// extractions of the resolved base, so no resolved plan holds it.
    UpdateFields {
        /// The struct value.
        base: Arc<Expr>,
        /// The field updates, in order.
        ops: Vec<FieldOp>,
    },
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
    /// `function OVER window` (Spark's `WindowExpression`). Unresolved,
    /// `function` is an [`ExprNode::Function`] call; resolved, it is a
    /// window or aggregate function and the window is a
    /// [`WindowSpecRef::Spec`] with its frame set (Spark's defaults
    /// applied).
    Window(WindowExpr),
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

/// One field update of an unresolved [`ExprNode::UpdateFields`] (Spark's
/// `StructFieldsOperation`): a [`FieldOp::With`] carries its value as a
/// child, a [`FieldOp::Drop`] carries none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldOp {
    /// Add the field `name` with `value`, or replace every field matching
    /// it (Spark's `WithField`). The name is plain, never a dotted path.
    With {
        /// The field name, as written.
        name: String,
        /// The new value.
        value: Arc<Expr>,
    },
    /// Drop every field matching `name`, ignoring names that match nothing
    /// (Spark's `DropField`).
    Drop {
        /// The field name, as written.
        name: String,
    },
}

/// An unresolved `*` (empty qualifier) or `t.*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Star {
    /// The qualifier parts; empty for a bare `*`.
    pub qualifier: Vec<String>,
    /// `* EXCEPT (cols)`: excluded column paths, each as written (a
    /// top-level column like `id`, or qualified like `t.id`; longer paths
    /// name nested struct fields and are rejected by the analyzer).
    pub except: Vec<Vec<String>>,
    /// The DataFrame the star expands, for PySpark's `df["*"]`; resolved
    /// as for [`ColumnRef::plan_id`].
    pub plan_id: Option<PlanId>,
}

/// An unresolved function call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    /// The function name; operators use their canonical names.
    pub name: String,
    /// The qualifier parts (`builtin` in `builtin.abs(x)`), outermost
    /// first; empty for an unqualified call. The analyzer resolves
    /// `builtin.` / `system.builtin.` (case-insensitively) to the builtin
    /// function, `session.` / `system.session.` to a session UDF, and
    /// reports any other qualifier as an unresolved routine.
    pub qualifier: Vec<String>,
    /// The arguments.
    pub args: Vec<Arc<Expr>>,
    /// The parameter name of each argument (`name => value`), if written;
    /// `None` for a positional argument. Empty when every argument is
    /// positional (as every front end but SQL lowering writes); otherwise
    /// as long as [`Self::args`].
    pub arg_names: Vec<Option<String>>,
    /// `f(DISTINCT ...)`.
    pub distinct: bool,
    /// `f(...) FILTER (WHERE filter)`.
    pub filter: Option<Arc<Expr>>,
    /// `f(...) WITHIN GROUP (ORDER BY ...)` (ordered-set aggregates:
    /// `listagg`, `mode`, `percentile_cont`, `percentile_disc`); empty for other calls. The analyzer lowers a
    /// valid one into the aggregate's arguments and keeps this form only
    /// for naming (Spark's `toPrettySQL` renders the call as written).
    pub within_group: Vec<SortOrder>,
}

impl FunctionCall {
    /// The name when it is unqualified (no qualifier), as operators and
    /// the name-based function checks require.
    #[must_use]
    pub fn unqualified(&self) -> Option<&str> {
        self.qualifier.is_empty().then_some(&self.name)
    }
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

/// `function OVER window`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowExpr {
    /// The window or aggregate function call.
    pub function: Arc<Expr>,
    /// The window it is evaluated over.
    pub window: WindowSpecRef,
}

/// The window of a [`WindowExpr`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WindowSpecRef {
    /// An inline window (a named one, after SQL lowering substituted its
    /// definition).
    Spec(WindowSpec),
    /// `OVER w` with no `WINDOW` clause defining `w` around it
    /// (unresolved only; the analyzer reports
    /// `MISSING_WINDOW_SPECIFICATION` once the operator's input resolved,
    /// as Spark's `WindowsSubstitution` does).
    Named(String),
}

/// `(PARTITION BY ... ORDER BY ... frame)` (Spark's
/// `WindowSpecDefinition`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowSpec {
    /// The partitioning expressions.
    pub partition_by: Vec<Arc<Expr>>,
    /// The ordering within a partition.
    pub order_by: Vec<SortOrder>,
    /// The frame; `None` is Spark's unspecified frame (unresolved plans
    /// only: the analyzer sets the function's frame or Spark's default).
    pub frame: Option<WindowFrame>,
}

/// A window frame (Spark's `SpecifiedWindowFrame`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowFrame {
    /// `ROWS` or `RANGE`.
    pub units: FrameUnits,
    /// The start bound.
    pub lower: FrameBound,
    /// The end bound.
    pub upper: FrameBound,
}

/// The units of a [`WindowFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameUnits {
    /// `ROWS`: bounds count rows.
    Rows,
    /// `RANGE`: bounds are values of the ordering key.
    Range,
}

/// A bound of a [`WindowFrame`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameBound {
    /// `UNBOUNDED PRECEDING`: the partition's first row.
    UnboundedPreceding,
    /// `CURRENT ROW` (with its peers, in `RANGE` units).
    CurrentRow,
    /// `UNBOUNDED FOLLOWING`: the partition's last row.
    UnboundedFollowing,
    /// A value bound, signed as Spark keeps it: SQL's `n PRECEDING` is
    /// `-(n)` (the canonical `negative` call), `n FOLLOWING` is `n`, and a
    /// DataFrame `rowsBetween(-1, 1)` is the literals `-1` and `1`.
    Offset(Arc<Expr>),
}

impl WindowFrame {
    /// `units BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`.
    #[must_use]
    pub const fn unbounded(units: FrameUnits) -> Self {
        Self {
            units,
            lower: FrameBound::UnboundedPreceding,
            upper: FrameBound::UnboundedFollowing,
        }
    }

    /// `units BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
    #[must_use]
    pub const fn running(units: FrameUnits) -> Self {
        Self {
            units,
            lower: FrameBound::UnboundedPreceding,
            upper: FrameBound::CurrentRow,
        }
    }
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
    /// `f(...) WITHIN GROUP (ORDER BY ...)` as written, for naming only;
    /// the analyzer lowers a valid one into [`Self::args`].
    pub within_group: Vec<SortOrder>,
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
    /// A lambda variable, bound by the analyzer to one of the enclosing
    /// [`ExprNode::Lambda`] node's parameters (Spark's
    /// `NamedLambdaVariable`). It is not an output attribute: only the
    /// higher-order function compiled around the lambda reads it.
    LambdaVar(BoundColumn),
    /// A function bound to its identity and result type.
    Function(ResolvedFunction),
    /// The field `name` (position `ordinal`) of the struct `base` (Spark's
    /// `GetStructField`), resolved under `spark.sql.caseSensitive`.
    Field {
        /// The struct value.
        base: Arc<Expr>,
        /// The field's position in the struct.
        ordinal: usize,
        /// The field's name.
        name: String,
        /// The field's type.
        data_type: DataType,
        /// Whether the field (or the struct) may be null.
        nullable: bool,
    },
    /// `expr AS name`, producing a new attribute.
    Alias {
        /// The aliased expression.
        expr: Arc<Expr>,
        /// The attribute it produces.
        output: BoundColumn,
    },
    /// The only argument of an aggregate call that merges the states of a
    /// [`PlanNode::PartialAggregate`](crate::PlanNode) instead of reading
    /// raw values. It is not a value: only the aggregate planned around it
    /// reads it.
    StateInput {
        /// The state columns of the call, in order.
        columns: Vec<Arc<Expr>>,
        /// The type of the raw argument the states were built from: their
        /// types and the result derive from it.
        raw_type: DataType,
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
            qualifier: Vec::new(),
            args: args.into_iter().map(Arc::new).collect(),
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
        }))
    }

    /// An unresolved `params -> body` lambda. Parameter identities are
    /// placeholders (`AttrId(u64::MAX)`) until the analyzer binds them;
    /// passes that reserve fresh ids (see `first_free_id`) skip them.
    #[must_use]
    pub fn lambda(params: Vec<String>, body: Self) -> Self {
        Self::new(ExprNode::Lambda {
            params: params
                .into_iter()
                .map(|name| BoundColumn {
                    id: AttrId(u64::MAX),
                    name,
                    data_type: DataType::Null,
                    nullable: true,
                })
                .collect(),
            body: Arc::new(body),
        })
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
            | ExprNode::TypedLiteral { .. }
            | ExprNode::Star(_)
            | ExprNode::Resolved(
                ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_) | ResolvedExpr::LambdaVar(_),
            ) => {}
            ExprNode::Function(FunctionCall {
                args,
                filter,
                within_group,
                ..
            })
            | ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                args,
                filter,
                within_group,
                ..
            })) => {
                args.iter().for_each(&mut *f);
                if let Some(filter) = filter {
                    f(filter);
                }
                within_group.iter().for_each(|o| f(&o.expr));
            }
            ExprNode::Resolved(ResolvedExpr::StateInput { columns, .. }) => {
                columns.iter().for_each(&mut *f);
            }
            ExprNode::Cast { expr, .. }
            | ExprNode::Alias { expr, .. }
            | ExprNode::Lambda { body: expr, .. }
            | ExprNode::Field { base: expr, .. }
            | ExprNode::Resolved(
                ResolvedExpr::Alias { expr, .. } | ResolvedExpr::Field { base: expr, .. },
            ) => f(expr),
            ExprNode::UpdateFields { .. } => self.update_fields_children(f),
            ExprNode::Subquery { kind, .. } => match kind {
                SubqueryKind::In(values) => values.iter().for_each(f),
                SubqueryKind::Scalar | SubqueryKind::Exists => {}
            },
            ExprNode::Window(w) => {
                f(&w.function);
                if let WindowSpecRef::Spec(spec) = &w.window {
                    spec.partition_by.iter().for_each(&mut *f);
                    spec.order_by.iter().for_each(|o| f(&o.expr));
                    if let Some(frame) = &spec.frame {
                        for b in [&frame.lower, &frame.upper] {
                            if let FrameBound::Offset(e) = b {
                                f(e);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Calls `f` on each direct child expression slot, in the order of
    /// [`Self::for_each_child`].
    pub(crate) fn for_each_child_mut(&mut self, f: &mut dyn FnMut(&mut Arc<Self>)) {
        match &mut self.node {
            ExprNode::Column(_)
            | ExprNode::Literal(_)
            | ExprNode::TypedLiteral { .. }
            | ExprNode::Star(_)
            | ExprNode::Resolved(
                ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_) | ResolvedExpr::LambdaVar(_),
            ) => {}
            ExprNode::Function(FunctionCall {
                args,
                filter,
                within_group,
                ..
            })
            | ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                args,
                filter,
                within_group,
                ..
            })) => {
                args.iter_mut().for_each(&mut *f);
                if let Some(filter) = filter {
                    f(filter);
                }
                within_group.iter_mut().for_each(|o| f(&mut o.expr));
            }
            ExprNode::Resolved(ResolvedExpr::StateInput { columns, .. }) => {
                columns.iter_mut().for_each(&mut *f);
            }
            ExprNode::Cast { expr, .. }
            | ExprNode::Alias { expr, .. }
            | ExprNode::Lambda { body: expr, .. }
            | ExprNode::Field { base: expr, .. }
            | ExprNode::Resolved(
                ResolvedExpr::Alias { expr, .. } | ResolvedExpr::Field { base: expr, .. },
            ) => f(expr),
            ExprNode::UpdateFields { .. } => update_fields_children_mut(&mut self.node, f),
            ExprNode::Subquery { kind, .. } => match kind {
                SubqueryKind::In(values) => values.iter_mut().for_each(f),
                SubqueryKind::Scalar | SubqueryKind::Exists => {}
            },
            ExprNode::Window(w) => {
                f(&mut w.function);
                if let WindowSpecRef::Spec(spec) = &mut w.window {
                    spec.partition_by.iter_mut().for_each(&mut *f);
                    spec.order_by.iter_mut().for_each(|o| f(&mut o.expr));
                    if let Some(frame) = &mut spec.frame {
                        for b in [&mut frame.lower, &mut frame.upper] {
                            if let FrameBound::Offset(e) = b {
                                f(e);
                            }
                        }
                    }
                }
            }
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

/// The operand of `-(x)` (the canonical `negative`, unresolved or
/// resolved): a frame bound written `x PRECEDING`.
impl Expr {
    /// Calls `f` on the base then on each [`FieldOp::With`] value of an
    /// [`ExprNode::UpdateFields`], in child order (see
    /// [`Self::for_each_child`]). Out of line so the traversal keeps its
    /// frame: plans at the depth cap must still fit a 2 MiB stack (see
    /// `tests::depth`).
    #[inline(never)]
    fn update_fields_children<'a, 'b, F>(&'a self, f: &'b mut F)
    where
        F: FnMut(&'a Arc<Self>),
    {
        if let ExprNode::UpdateFields { base, ops } = &self.node {
            f(base);
            for op in ops {
                if let FieldOp::With { value, .. } = op {
                    f(value);
                }
            }
        }
    }
}

/// As [`Expr::update_fields_children`], for [`Expr::for_each_child_mut`].
#[inline(never)]
fn update_fields_children_mut(node: &mut ExprNode, f: &mut dyn FnMut(&mut Arc<Expr>)) {
    if let ExprNode::UpdateFields { base, ops } = node {
        f(base);
        for op in ops.iter_mut() {
            if let FieldOp::With { value, .. } = op {
                f(value);
            }
        }
    }
}

pub(crate) fn negated(e: &Expr) -> Option<&Arc<Expr>> {
    let (name, args) = match &e.node {
        ExprNode::Function(f) => (f.unqualified()?, &f.args),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => (f.id.name(), &f.args),
        _ => return None,
    };
    match &args[..] {
        [x] if name == crate::Builtin::Negative.name() => Some(x),
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Literal;

    #[test]
    fn negated_finds_negative_operands_resolved_or_not() {
        let inner = Arc::new(Expr::literal(Literal::Integer(3)));
        // Unresolved `negative(x)`.
        let unresolved = Expr::new(ExprNode::Function(FunctionCall {
            within_group: Vec::new(),
            name: "negative".to_owned(),
            qualifier: Vec::new(),
            args: vec![Arc::clone(&inner)],
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
        }));
        assert!(negated(&unresolved).is_some());
        // A qualified name is never an operator.
        let qualified = Expr::new(ExprNode::Function(FunctionCall {
            within_group: Vec::new(),
            name: "negative".to_owned(),
            qualifier: vec!["session".to_owned()],
            args: vec![Arc::clone(&inner)],
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
        }));
        assert!(negated(&qualified).is_none());
        // Resolved `-x` (a UDF id works like any other: only the name matters).
        let resolved = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                within_group: Vec::new(),
                id: FunctionId::Builtin(Builtin::Negative),
                kind: FunctionKind::Scalar,
                args: vec![Arc::clone(&inner)],
                distinct: false,
                filter: None,
                data_type: DataType::Integer,
                nullable: false,
                deterministic: true,
            },
        )));
        assert!(negated(&resolved).is_some());
        // Anything else is not negated.
        assert!(negated(&Expr::literal(Literal::Integer(1))).is_none());
    }
}
