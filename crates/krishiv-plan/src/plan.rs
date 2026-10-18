// SPDX-License-Identifier: Apache-2.0
//! Relation nodes, unresolved and resolved, and the versioned [`Plan`].

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_common::secret::OptionValue;

use crate::dag::{self, Memo};
use crate::walk::{self, Node};
use crate::{BoundColumn, Expr, Origin, SortOrder};

/// The IR version a [`Plan`] is written in. It changes when a node changes
/// meaning; adding a node or field does not change it before 1.0.
pub const IR_VERSION: u32 = 1;

/// A logical plan with the IR version it is written in, checked to be
/// within [`crate::MAX_DEPTH`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    ir_version: u32,
    root: Arc<LogicalPlan>,
}

impl Plan {
    /// A plan in the current [`IR_VERSION`].
    ///
    /// # Errors
    /// `FAILED_TO_PARSE_TOO_COMPLEX` when the plan nests deeper than
    /// [`crate::MAX_DEPTH`] (see [`crate::check_depth`]).
    pub fn try_new(root: impl Into<Arc<LogicalPlan>>) -> Result<Self, KrishivError> {
        let root = root.into();
        crate::check_depth(&root)?;
        Ok(Self {
            ir_version: IR_VERSION,
            root,
        })
    }

    /// The IR version.
    #[must_use]
    pub const fn ir_version(&self) -> u32 {
        self.ir_version
    }

    /// The root node.
    #[must_use]
    pub const fn root(&self) -> &Arc<LogicalPlan> {
        &self.root
    }
}

/// A relation: a node and its [`Origin`]. Children are shared through `Arc`.
///
/// `PartialEq` compares everything, origins included (see [`Expr`]), in
/// time linear in the distinct nodes even when subtrees are shared;
/// `Debug` prints a shared subtree once. Dropping a plan never recurses.
#[derive(Clone)]
pub struct LogicalPlan {
    /// What the relation is.
    pub node: PlanNode,
    /// Where it came from.
    pub origin: Origin,
}

/// The relation node kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanNode {
    /// An unresolved read of a table, view, CTE or files.
    Read(ReadSource),
    /// Unresolved SQL text, as `spark.sql(...)` sends it.
    Sql {
        /// The statement.
        query: String,
        /// Parameter values.
        args: SqlArgs,
    },
    /// A resolved read with its output attributes and pinned version.
    Scan(Scan),
    /// `SELECT exprs`.
    Project {
        /// Input.
        input: Arc<LogicalPlan>,
        /// Output expressions.
        exprs: Vec<Arc<Expr>>,
    },
    /// DataFrame `withColumn(s)`: adds or replaces columns by name. The
    /// analyzer rewrites it to a [`PlanNode::Project`].
    WithColumns {
        /// Input.
        input: Arc<LogicalPlan>,
        /// `(name, value)` pairs, in order.
        columns: Vec<(String, Arc<Expr>)>,
    },
    /// `WHERE` / `HAVING` / `filter`.
    Filter {
        /// Input.
        input: Arc<LogicalPlan>,
        /// The predicate.
        predicate: Arc<Expr>,
    },
    /// `GROUP BY` with aggregate output expressions.
    Aggregate {
        /// Input.
        input: Arc<LogicalPlan>,
        /// Grouping keys.
        grouping: Grouping,
        /// Output expressions: keys and aggregates.
        aggregates: Vec<Arc<Expr>>,
    },
    /// A join; a comma in `FROM` is an inner join without a condition.
    Join {
        /// Left input.
        left: Arc<LogicalPlan>,
        /// Right input.
        right: Arc<LogicalPlan>,
        /// Join type.
        kind: JoinKind,
        /// Join condition.
        condition: JoinCondition,
    },
    /// `ORDER BY` (global) or `SORT BY` / `sortWithinPartitions`.
    Sort {
        /// Input.
        input: Arc<LogicalPlan>,
        /// Sort keys, most significant first.
        order: Vec<SortOrder>,
        /// Whether the sort is total rather than per partition.
        global: bool,
    },
    /// `LIMIT n`; `n` is a foldable expression or a parameter.
    Limit {
        /// Input.
        input: Arc<LogicalPlan>,
        /// Row count.
        limit: Arc<Expr>,
    },
    /// `input AS alias (columns)`, a derived table or `df.alias(...)`.
    SubqueryAlias {
        /// Input.
        input: Arc<LogicalPlan>,
        /// The alias.
        alias: String,
        /// Output column names; empty to keep the input's.
        columns: Vec<String>,
    },
    /// `WITH name AS (plan), ... input`.
    WithCte {
        /// The main query.
        input: Arc<LogicalPlan>,
        /// The definitions, in order.
        ctes: Vec<Cte>,
    },
    /// A resolved reference to a `WITH` definition (see [`CteRef`]).
    CteRef(CteRef),
    /// A fresh instance of a resolved subtree: every attribute id inside
    /// `input` (outputs, references and aliases, subquery plans included;
    /// not a [`CteRef::source`]) reads as `id + offset`. The analyzer adds
    /// it where a subtree shared through `Arc` meets itself, as in a
    /// DataFrame self-join, so the instances need no copy and the plan
    /// stays linear in its distinct nodes. The subtree has no outer
    /// references.
    Shift {
        /// The shared subtree.
        input: Arc<LogicalPlan>,
        /// What each attribute id inside gains.
        offset: u64,
    },
}

/// A resolved reference to a `WITH` definition: the definition is planned
/// once, and each reference reads its rows under attributes of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CteRef {
    /// The definition's name.
    pub name: String,
    /// The definition's output attributes, which identify it.
    pub source: Vec<crate::AttrId>,
    /// This reference's attributes, one per source attribute.
    pub output: Vec<BoundColumn>,
}

/// What an unresolved read reads.
///
/// Option values are typed ([`OptionValue`], ADR 0038): front ends build
/// them with [`OptionValue::from_user`], so secrets never print, not even
/// through `Debug`, and never enter the canonical encoding in plaintext.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadSource {
    /// A table, view or CTE by name (`spark.read.table`, `FROM t`).
    Table {
        /// Name parts, e.g. `["cat", "db", "t"]`.
        name: Vec<String>,
        /// Data source options.
        options: BTreeMap<String, OptionValue>,
    },
    /// Files of a format (`spark.read.parquet(path)`).
    Files {
        /// Format name, e.g. `parquet`, as written (compared
        /// case-insensitively; a resolved [`Scan`] holds it in lower case).
        format: String,
        /// Paths or globs, typed like options (built with
        /// [`OptionValue::from_user`] under the key `path`, as Spark keeps
        /// them as the `path` option): a signed URL's credentials are secret
        /// parts.
        paths: Vec<OptionValue>,
        /// Data source options.
        options: BTreeMap<String, OptionValue>,
    },
}

/// A resolved read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scan {
    /// What was read.
    pub source: ReadSource,
    /// Output attributes.
    pub output: Vec<BoundColumn>,
    /// The table version pinned for the whole query (ADR 0037); `None` for
    /// sources without versions.
    pub version: Option<TableVersion>,
}

/// A table version: an Iceberg snapshot id or a Delta table version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableVersion(pub i64);

/// Parameters of a [`PlanNode::Sql`] statement.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SqlArgs {
    /// No parameters.
    #[default]
    None,
    /// `?` markers, in order.
    Positional(Vec<Arc<Expr>>),
    /// `:name` markers.
    Named(BTreeMap<String, Arc<Expr>>),
}

/// Grouping of an [`PlanNode::Aggregate`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Grouping {
    /// `GROUP BY exprs`; empty for a global aggregate.
    GroupBy(Vec<Arc<Expr>>),
    /// `GROUP BY ROLLUP(exprs)`.
    Rollup(Vec<Arc<Expr>>),
    /// `GROUP BY CUBE(exprs)`.
    Cube(Vec<Arc<Expr>>),
    /// `GROUP BY exprs GROUPING SETS (sets)`.
    GroupingSets {
        /// All grouping expressions.
        exprs: Vec<Arc<Expr>>,
        /// The sets.
        sets: Vec<Vec<Arc<Expr>>>,
    },
}

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JoinKind {
    /// `INNER`.
    Inner,
    /// `LEFT OUTER`.
    LeftOuter,
    /// `RIGHT OUTER`.
    RightOuter,
    /// `FULL OUTER`.
    FullOuter,
    /// `LEFT SEMI`.
    LeftSemi,
    /// `LEFT ANTI`.
    LeftAnti,
    /// `CROSS`.
    Cross,
}

/// Join condition.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum JoinCondition {
    /// No condition.
    None,
    /// `ON expr`.
    On(Arc<Expr>),
    /// `USING (columns)`.
    Using(Vec<String>),
}

/// One `WITH` definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cte {
    /// The name.
    pub name: String,
    /// The definition.
    pub plan: Arc<LogicalPlan>,
}

impl LogicalPlan {
    /// A relation of unknown origin.
    #[must_use]
    pub const fn new(node: PlanNode) -> Self {
        Self {
            node,
            origin: Origin::UNKNOWN,
        }
    }

    /// The same relation with `origin`.
    #[must_use]
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// An unresolved read of a table by its name parts, without options.
    #[must_use]
    pub fn table<S: Into<String>>(name: impl IntoIterator<Item = S>) -> Self {
        Self::new(PlanNode::Read(ReadSource::Table {
            name: name.into_iter().map(Into::into).collect(),
            options: BTreeMap::new(),
        }))
    }

    /// `SELECT exprs FROM self`.
    #[must_use]
    pub fn project(self, exprs: Vec<Expr>) -> Self {
        Self::new(PlanNode::Project {
            input: Arc::new(self),
            exprs: arcs(exprs),
        })
    }

    /// `self WHERE predicate`.
    #[must_use]
    pub fn filter(self, predicate: impl Into<Arc<Expr>>) -> Self {
        Self::new(PlanNode::Filter {
            input: Arc::new(self),
            predicate: predicate.into(),
        })
    }

    /// `SELECT aggregates FROM self GROUP BY group_by`.
    #[must_use]
    pub fn aggregate(self, group_by: Vec<Expr>, aggregates: Vec<Expr>) -> Self {
        Self::new(PlanNode::Aggregate {
            input: Arc::new(self),
            grouping: Grouping::GroupBy(arcs(group_by)),
            aggregates: arcs(aggregates),
        })
    }

    /// `self kind JOIN right condition`.
    #[must_use]
    pub fn join(self, right: Self, kind: JoinKind, condition: JoinCondition) -> Self {
        Self::new(PlanNode::Join {
            left: Arc::new(self),
            right: Arc::new(right),
            kind,
            condition,
        })
    }

    /// `self ORDER BY order` (a global sort).
    #[must_use]
    pub fn sort(self, order: Vec<SortOrder>) -> Self {
        Self::new(PlanNode::Sort {
            input: Arc::new(self),
            order,
            global: true,
        })
    }

    /// `self LIMIT limit`.
    #[must_use]
    pub fn limit(self, limit: impl Into<Arc<Expr>>) -> Self {
        Self::new(PlanNode::Limit {
            input: Arc::new(self),
            limit: limit.into(),
        })
    }

    /// `(self) AS alias`.
    #[must_use]
    pub fn alias(self, alias: impl Into<String>) -> Self {
        Self::new(PlanNode::SubqueryAlias {
            input: Arc::new(self),
            alias: alias.into(),
            columns: Vec::new(),
        })
    }

    /// Calls `f` on each child relation, in order: CTE definitions before
    /// the main query. Plans inside subquery expressions are not children.
    pub fn for_each_child<'a>(&'a self, f: &mut impl FnMut(&'a Arc<Self>)) {
        match &self.node {
            PlanNode::Read(_) | PlanNode::Sql { .. } | PlanNode::Scan(_) | PlanNode::CteRef(_) => {}
            PlanNode::Project { input, .. }
            | PlanNode::WithColumns { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Aggregate { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. }
            | PlanNode::SubqueryAlias { input, .. }
            | PlanNode::Shift { input, .. } => f(input),
            PlanNode::Join { left, right, .. } => {
                f(left);
                f(right);
            }
            PlanNode::WithCte { input, ctes } => {
                ctes.iter().for_each(|c| f(&c.plan));
                f(input);
            }
        }
    }

    fn for_each_child_mut(&mut self, f: &mut dyn FnMut(&mut Arc<Self>)) {
        match &mut self.node {
            PlanNode::Read(_) | PlanNode::Sql { .. } | PlanNode::Scan(_) | PlanNode::CteRef(_) => {}
            PlanNode::Project { input, .. }
            | PlanNode::WithColumns { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Aggregate { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. }
            | PlanNode::SubqueryAlias { input, .. }
            | PlanNode::Shift { input, .. } => f(input),
            PlanNode::Join { left, right, .. } => {
                f(left);
                f(right);
            }
            PlanNode::WithCte { input, ctes } => {
                ctes.iter_mut().for_each(|c| f(&mut c.plan));
                f(input);
            }
        }
    }

    /// Calls `f` on each expression this node holds (not its children's),
    /// in order.
    pub fn for_each_expr<'a>(&'a self, f: &mut impl FnMut(&'a Arc<Expr>)) {
        match &self.node {
            PlanNode::Read(_)
            | PlanNode::Scan(_)
            | PlanNode::SubqueryAlias { .. }
            | PlanNode::WithCte { .. }
            | PlanNode::CteRef(_)
            | PlanNode::Shift { .. } => {}
            PlanNode::Sql { args, .. } => match args {
                SqlArgs::None => {}
                SqlArgs::Positional(v) => v.iter().for_each(f),
                SqlArgs::Named(m) => m.values().for_each(f),
            },
            PlanNode::Project { exprs, .. } => exprs.iter().for_each(f),
            PlanNode::WithColumns { columns, .. } => columns.iter().for_each(|(_, e)| f(e)),
            PlanNode::Filter { predicate: e, .. } | PlanNode::Limit { limit: e, .. } => f(e),
            PlanNode::Aggregate {
                grouping,
                aggregates,
                ..
            } => {
                match grouping {
                    Grouping::GroupBy(v) | Grouping::Rollup(v) | Grouping::Cube(v) => {
                        v.iter().for_each(&mut *f);
                    }
                    Grouping::GroupingSets { exprs, sets } => {
                        exprs.iter().chain(sets.iter().flatten()).for_each(&mut *f);
                    }
                }
                aggregates.iter().for_each(f);
            }
            PlanNode::Join { condition, .. } => match condition {
                JoinCondition::On(e) => f(e),
                JoinCondition::None | JoinCondition::Using(_) => {}
            },
            PlanNode::Sort { order, .. } => order.iter().for_each(|o| f(&o.expr)),
        }
    }

    fn for_each_expr_mut(&mut self, f: &mut dyn FnMut(&mut Arc<Expr>)) {
        match &mut self.node {
            PlanNode::Read(_)
            | PlanNode::Scan(_)
            | PlanNode::SubqueryAlias { .. }
            | PlanNode::WithCte { .. }
            | PlanNode::CteRef(_)
            | PlanNode::Shift { .. } => {}
            PlanNode::Sql { args, .. } => match args {
                SqlArgs::None => {}
                SqlArgs::Positional(v) => v.iter_mut().for_each(f),
                SqlArgs::Named(m) => m.values_mut().for_each(f),
            },
            PlanNode::Project { exprs, .. } => exprs.iter_mut().for_each(f),
            PlanNode::WithColumns { columns, .. } => {
                columns.iter_mut().for_each(|(_, e)| f(e));
            }
            PlanNode::Filter { predicate: e, .. } | PlanNode::Limit { limit: e, .. } => f(e),
            PlanNode::Aggregate {
                grouping,
                aggregates,
                ..
            } => {
                match grouping {
                    Grouping::GroupBy(v) | Grouping::Rollup(v) | Grouping::Cube(v) => {
                        v.iter_mut().for_each(&mut *f);
                    }
                    Grouping::GroupingSets { exprs, sets } => exprs
                        .iter_mut()
                        .chain(sets.iter_mut().flatten())
                        .for_each(&mut *f),
                }
                aggregates.iter_mut().for_each(f);
            }
            PlanNode::Join { condition, .. } => match condition {
                JoinCondition::On(e) => f(e),
                JoinCondition::None | JoinCondition::Using(_) => {}
            },
            PlanNode::Sort { order, .. } => order.iter_mut().for_each(|o| f(&mut o.expr)),
        }
    }

    /// The output attributes of a resolved relation, as the analyzer
    /// produced them; `None` when the relation (or a relation it reads) is
    /// unresolved. Later passes use it rather than re-deriving outputs:
    /// inside a [`PlanNode::Shift`], every attribute id reads `id +
    /// offset`; a [`PlanNode::CteRef`] outputs its own attributes, one per
    /// attribute of the definition its `source` names; outer joins make the
    /// other side nullable. Attribute ids are unique within a plan but not
    /// dense: never index by them.
    #[must_use]
    pub fn resolved_output(&self) -> Option<Vec<BoundColumn>> {
        let mut p = self;
        loop {
            match &p.node {
                PlanNode::Filter { input, .. }
                | PlanNode::Sort { input, .. }
                | PlanNode::Limit { input, .. }
                | PlanNode::WithCte { input, .. } => p = input,
                PlanNode::SubqueryAlias { input, columns, .. } => {
                    let out = input.resolved_output()?;
                    if columns.is_empty() {
                        return Some(out);
                    }
                    if columns.len() != out.len() {
                        return None;
                    }
                    let renamed = out.into_iter().zip(columns);
                    return Some(
                        renamed
                            .map(|(c, name)| BoundColumn {
                                name: name.clone(),
                                ..c
                            })
                            .collect(),
                    );
                }
                PlanNode::Scan(s) => return Some(s.output.clone()),
                PlanNode::CteRef(r) => return Some(r.output.clone()),
                PlanNode::Shift { input, offset } => {
                    let out = input.resolved_output()?;
                    return out
                        .into_iter()
                        .map(|c| {
                            Some(BoundColumn {
                                id: crate::AttrId(c.id.0.checked_add(*offset)?),
                                ..c
                            })
                        })
                        .collect();
                }
                PlanNode::Project { exprs, .. }
                | PlanNode::Aggregate {
                    aggregates: exprs, ..
                } => {
                    return exprs
                        .iter()
                        .map(|e| match &e.node {
                            crate::ExprNode::Resolved(
                                crate::ResolvedExpr::Column(c)
                                | crate::ResolvedExpr::Alias { output: c, .. },
                            ) => Some(c.clone()),
                            _ => None,
                        })
                        .collect();
                }
                PlanNode::Join {
                    left, right, kind, ..
                } => {
                    let nullable = |v: Vec<BoundColumn>, yes: bool| {
                        v.into_iter()
                            .map(|c| BoundColumn {
                                nullable: c.nullable || yes,
                                ..c
                            })
                            .collect::<Vec<_>>()
                    };
                    let (l, r) = (left.resolved_output()?, right.resolved_output()?);
                    let mut out = nullable(
                        l,
                        matches!(kind, JoinKind::RightOuter | JoinKind::FullOuter),
                    );
                    if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
                        let r =
                            nullable(r, matches!(kind, JoinKind::LeftOuter | JoinKind::FullOuter));
                        out.extend(r);
                    }
                    return Some(out);
                }
                PlanNode::Read(_) | PlanNode::Sql { .. } | PlanNode::WithColumns { .. } => {
                    return None;
                }
            }
        }
    }

    /// Whether no unresolved node remains anywhere in the plan, subquery
    /// plans included. It does not recurse.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        walk::all_resolved(Node::Plan(self))
    }

    /// Replaces each child relation by `f(child)`; returns `self` itself
    /// when every child comes back unchanged (`Arc::ptr_eq`).
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn map_children<E>(
        self: Arc<Self>,
        mut f: impl FnMut(Arc<Self>) -> Result<Arc<Self>, E>,
    ) -> Result<Arc<Self>, E> {
        let mut old = Vec::new();
        self.for_each_child(&mut |c| old.push(Arc::clone(c)));
        rebuild(self, old, &mut f, Self::for_each_child_mut)
    }

    /// Replaces each expression of this node by `f(expr)`; returns `self`
    /// itself when every expression comes back unchanged.
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn map_expressions<E>(
        self: Arc<Self>,
        mut f: impl FnMut(Arc<Expr>) -> Result<Arc<Expr>, E>,
    ) -> Result<Arc<Self>, E> {
        let mut old = Vec::new();
        self.for_each_expr(&mut |e| old.push(Arc::clone(e)));
        rebuild(self, old, &mut f, Self::for_each_expr_mut)
    }

    /// Applies `f` to every relation, children first. Subquery plans are not
    /// entered. A relation shared by several parents (`Arc`) is rewritten
    /// once and its rewrite shared, so `f` runs once per distinct relation.
    /// Recurses once per level: call it only on plans within
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

    /// Applies `f` to every relation, parents first; see
    /// [`Self::transform_up`].
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

impl PartialEq for LogicalPlan {
    fn eq(&self, other: &Self) -> bool {
        dag::eq(Node::Plan(self), Node::Plan(other), || {
            self.origin == other.origin && self.node == other.node
        })
    }
}

impl Eq for LogicalPlan {}

impl fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        dag::debug(Node::Plan(self), f, "LogicalPlan", |f, name| {
            f.debug_struct(name)
                .field("node", &self.node)
                .field("origin", &self.origin)
                .finish()
        })
    }
}

/// Visits a node's `Arc` slots of one kind (children or expressions).
type Slots<T> = fn(&mut LogicalPlan, &mut dyn FnMut(&mut Arc<T>));

/// Maps `old` (the current slots, in visiting order) through `f` and writes
/// the results back with `slots` when any changed.
fn rebuild<T, E>(
    plan: Arc<LogicalPlan>,
    old: Vec<Arc<T>>,
    f: &mut impl FnMut(Arc<T>) -> Result<Arc<T>, E>,
    slots: Slots<T>,
) -> Result<Arc<LogicalPlan>, E> {
    let mut new = Vec::with_capacity(old.len());
    let mut changed = false;
    for item in old {
        let next = f(Arc::clone(&item))?;
        changed |= !Arc::ptr_eq(&next, &item);
        new.push(next);
    }
    if !changed {
        return Ok(plan);
    }
    let mut plan = Arc::unwrap_or_clone(plan);
    let mut new = new.into_iter();
    slots(&mut plan, &mut |slot| {
        if let Some(next) = new.next() {
            *slot = next;
        }
    });
    Ok(Arc::new(plan))
}

fn arcs(exprs: Vec<Expr>) -> Vec<Arc<Expr>> {
    exprs.into_iter().map(Arc::new).collect()
}

impl Drop for LogicalPlan {
    fn drop(&mut self) {
        walk::teardown_plan(self);
    }
}
