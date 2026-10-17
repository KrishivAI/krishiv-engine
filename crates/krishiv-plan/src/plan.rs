// SPDX-License-Identifier: Apache-2.0
//! Relation nodes, unresolved and resolved, and the versioned [`Plan`].

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{BoundColumn, Expr, Origin, SortOrder};

/// The IR version a [`Plan`] is written in. It changes when a node changes
/// meaning; adding a node or field does not change it before 1.0.
pub const IR_VERSION: u32 = 1;

/// A logical plan with the IR version it is written in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    ir_version: u32,
    root: Arc<LogicalPlan>,
}

impl Plan {
    /// A plan in the current [`IR_VERSION`].
    #[must_use]
    pub fn new(root: impl Into<Arc<LogicalPlan>>) -> Self {
        Self {
            ir_version: IR_VERSION,
            root: root.into(),
        }
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
/// `PartialEq` compares everything, origins included (see [`Expr`]).
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

/// What an unresolved read reads.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadSource {
    /// A table, view or CTE by name (`spark.read.table`, `FROM t`).
    Table {
        /// Name parts, e.g. `["cat", "db", "t"]`.
        name: Vec<String>,
        /// Data source options.
        options: BTreeMap<String, String>,
    },
    /// Files of a format (`spark.read.parquet(path)`).
    Files {
        /// Format name, e.g. `parquet`.
        format: String,
        /// Paths or globs.
        paths: Vec<String>,
        /// Data source options.
        options: BTreeMap<String, String>,
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
            PlanNode::Read(_) | PlanNode::Sql { .. } | PlanNode::Scan(_) => {}
            PlanNode::Project { input, .. }
            | PlanNode::WithColumns { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Aggregate { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. }
            | PlanNode::SubqueryAlias { input, .. } => f(input),
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

    /// Calls `f` on each expression this node holds (not its children's),
    /// in order.
    pub fn for_each_expr<'a>(&'a self, f: &mut impl FnMut(&'a Arc<Expr>)) {
        match &self.node {
            PlanNode::Read(_)
            | PlanNode::Scan(_)
            | PlanNode::SubqueryAlias { .. }
            | PlanNode::WithCte { .. } => {}
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
}

fn arcs(exprs: Vec<Expr>) -> Vec<Arc<Expr>> {
    exprs.into_iter().map(Arc::new).collect()
}
