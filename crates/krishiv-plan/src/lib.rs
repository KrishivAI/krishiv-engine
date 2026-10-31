// SPDX-License-Identifier: Apache-2.0
//! Krishiv's logical-plan IR (layer L3; ADR 0003, updates of 2026-10-17).
//!
//! Every front end lowers into one model: the SQL parser's AST, Spark
//! Connect protobuf (translated at the API boundary in `krishiv-connect`),
//! and the native Python and Rust DataFrame APIs. The analyzer turns an
//! unresolved plan into a resolved one in the same model. This crate never
//! depends on `prost`, `tonic` or `krishiv-connect` (`cargo xtask layering`).
//!
//! # Model
//!
//! - A [`Plan`] is a root [`LogicalPlan`] and its [`IR_VERSION`], checked to
//!   nest no deeper than [`MAX_DEPTH`].
//! - A [`LogicalPlan`] is a [`PlanNode`] plus its [`Origin`]; an [`Expr`]
//!   is an [`ExprNode`] plus its [`Origin`]. Children are `Arc`s, so
//!   rewrites (`map_children`, `transform_up`, `transform_down`) share
//!   unchanged subtrees. Dropping never recurses.
//! - An [`Origin`] holds an optional DataFrame [`PlanId`] and an optional
//!   SQL [`SourceId`] and [`Span`].
//! - Unresolved operators are [`ExprNode::Function`] calls with canonical
//!   names ([`function`]); resolved calls carry a typed [`FunctionId`] and
//!   [`FunctionKind`].
//! - Data types are [`krishiv_types::DataType`].
//! - `PartialEq` compares everything, origins included, and like `Debug`
//!   and `transform_*` takes time linear in the distinct nodes when
//!   subtrees are shared. The canonical
//!   encoding ([`Plan::canonical_encoding`]) leaves origins out and is the
//!   structural equality ([`Plan::structurally_eq`]); [`StructuralHash`] is
//!   its 128-bit Merkle hash, stable across processes and platforms
//!   (within one process for plans holding inline secrets, ADR 0038).
//! - Read options are [`krishiv_common::secret::OptionValue`]s: secrets are
//!   typed, so no `Display`, `Debug` or encoding of a plan holds their
//!   plaintext.
//! - `Display` prints a plan as a tree in the style of Spark's `treeString`.
//!
//! ```
//! use krishiv_plan::{Builtin, Expr, Literal, LogicalPlan, Origin, Plan, SourceId, Span};
//!
//! // SELECT a FROM t WHERE 1 < a
//! let lt = Expr::builtin(Builtin::Lt, vec![Literal::Integer(1).into(), Expr::column(["a"])]);
//! let plan = LogicalPlan::table(["t"]).filter(lt).project(vec![Expr::column(["a"])]);
//! assert_eq!(plan.to_string(), "Project [a]\n+- Filter (1 < a)\n   +- Read t\n");
//! // Origins change neither the encoding nor the hash.
//! let spanned = plan.clone().with_origin(Origin::sql(SourceId(0), Span::new(0, 28)));
//! let (a, b) = (Plan::try_new(plan)?, Plan::try_new(spanned)?);
//! assert!(a.structurally_eq(&b));
//! assert_eq!(a.structural_hash(), b.structural_hash());
//! # Ok::<(), krishiv_common::error::KrishivError>(())
//! ```
//!
//! Unresolved and resolved nodes share one tree. The analyzer replaces the
//! unresolved-only nodes (marked U below) by resolved ones (R);
//! [`LogicalPlan::is_resolved`] is true when none is left.
//!
//! # Node inventory
//!
//! Which nodes the TPC-H queries and the DataFrame slice need. Tests build
//! the 22 TPC-H plans by hand and each DataFrame-slice operation, and check
//! both columns against the nodes they use.
//!
//! | Node | Form | TPC-H | DataFrame slice |
//! |---|---|---|---|
//! | `PlanNode::Read` | U | all | `read.table`, `read.parquet` |
//! | `PlanNode::Sql` | U | - | `sql` |
//! | `PlanNode::OneRowRelation` | U, R | - | - |
//! | `PlanNode::Scan` | R | - | - |
//! | `PlanNode::Project` | U, R | Q2, Q4, Q6-Q9, Q11, Q14-Q17, Q19-Q22 | `select` |
//! | `PlanNode::WithColumns` | U | - | `withColumn` |
//! | `PlanNode::Filter` | U, R | Q1-Q12, Q14-Q22 | `filter`, `where` |
//! | `PlanNode::Aggregate` | U, R | Q1, Q3-Q5, Q7-Q13, Q15, Q16, Q18, Q21, Q22 | `groupBy.agg`, `rollup.agg`, `cube.agg` |
//! | `PlanNode::PartialAggregate` | R | - | - |
//! | `PlanNode::Join` | U, R | Q2, Q3, Q5, Q7-Q21 | `join` |
//! | `PlanNode::Sort` | U, R | Q1-Q5, Q7-Q13, Q15, Q16, Q18, Q20-Q22 | `orderBy`, `sort` |
//! | `PlanNode::Limit` | U, R | Q2, Q3, Q10, Q18, Q21 | `limit` |
//! | `PlanNode::Offset` | U, R | - | `offset` |
//! | `PlanNode::SubqueryAlias` | U, R | Q7-Q9, Q13, Q21, Q22 | `df.alias` |
//! | `PlanNode::WithCte` | U, R | Q15 | - |
//! | `PlanNode::CteRef` | R | - | - |
//! | `PlanNode::Shift` | R | - | - |
//! | `PlanNode::Hint` | U | - | - |
//! | `PlanNode::JoinHint` | R | - | - |
//! | `PlanNode::Union` | U, R | - | - |
//! | `PlanNode::Distinct` | U | - | - |
//! | `PlanNode::SetOp` | U | - | - |
//! | `PlanNode::ReplicateRows` | R | - | - |
//! | `PlanNode::Pivot` | U | - | - |
//! | `PlanNode::Unpivot` | U | - | - |
//! | `PlanNode::Expand` | R | - | - |
//! | `PlanNode::Generate` | U, R | - | - |
//! | `PlanNode::InlineTable` | U | - | - |
//! | `PlanNode::Window` | R | - | - |
//! | `ExprNode::Column` | U | all | `col`, `df["a"]` |
//! | `ExprNode::Literal` | U, R | all | `lit` |
//! | `ExprNode::Star` | U | Q1, Q4, Q13, Q21, Q22 | `select("*")` |
//! | `ExprNode::Function` | U | all | `operators`, `functions` |
//! | `ExprNode::Cast` | U, R | Q1, Q3-Q8, Q10, Q12, Q14, Q15, Q20 | `cast` |
//! | `ExprNode::Alias` | U | Q1, Q3-Q17, Q19, Q21, Q22 | `col.alias` |
//! | `ExprNode::Field` | U | - | - |
//! | `ExprNode::UpdateFields` | U | - | - |
//! | `ExprNode::Subquery` | U, R | Q2, Q4, Q11, Q15-Q18, Q20-Q22 | - |
//! | `ExprNode::Window` | U, R | - | - |
//! | `ExprNode::Lambda` | U, R | - | - |
//! | `ResolvedExpr::Column` | R | - | - |
//! | `ResolvedExpr::OuterColumn` | R | - | - |
//! | `ResolvedExpr::Function` | R | - | - |
//! | `ResolvedExpr::LambdaVar` | R | - | - |
//! | `ResolvedExpr::Alias` | R | - | - |
//! | `ResolvedExpr::Field` | R | - | - |
//! | `ResolvedExpr::StateInput` | R | - | - |
//!
//! Not in v1 (spec 13 §9; added with the features that need them):
//! `Generate`, `Repartition`, `Hint`, `Pivot`, `Command`,
//! `Extension`, `ToDF`, `Offset`, `Deduplicate`, `LocalRelation` and `Range`
//! relations; lambda, extract-value and extension expressions.

#![forbid(unsafe_code)]

pub mod analyzer;
pub mod cache;
mod dag;
mod display;
mod encode;
mod expr;
pub mod function;
mod literal;
mod origin;
mod partial;
mod plan;
mod raises;
pub(crate) mod spark_fmt;
#[cfg(test)]
mod tests;
mod walk;

pub use cache::{CacheKey, normalized_plan_hash, normalized_subtree_hash};
pub use display::Annotated;
pub use encode::{ENCODING_VERSION, StructuralHash};
pub use expr::{
    AttrId, BoundColumn, CastMode, ColumnRef, Expr, ExprNode, FieldOp, FrameBound, FrameUnits,
    FunctionCall, NullOrdering, ResolvedExpr, ResolvedFunction, SortDirection, SortOrder, Star,
    SubqueryKind, TypedLiteralKind, WindowExpr, WindowFrame, WindowSpec, WindowSpecRef,
};
pub use function::{Builtin, FunctionId, FunctionKind, QuantifiedLike};
pub use literal::{DayTimeField, Literal, YearMonthField};
pub use origin::{Origin, PlanId, SourceId, Span};
pub use partial::partial_state_columns;
/// The qualifier prefix of the columns a select-list generator adds: an
/// unqualified `*` skips them (the generator's own star reads them).
pub const GENERATE_QUALIFIER_PREFIX: &str = "__krishiv_generate_";

pub use plan::{
    Cte, CteRef, GeneratorKind, Grouping, IR_VERSION, JoinCondition, JoinKind, JoinStrategy,
    LogicalPlan, PivotValue, Plan, PlanNode, ReadSource, Scan, SetOperation, SqlArgs,
    TABLE_FUNCTION_NAME_SEP, TableVersion, UnpivotColumnSet,
};
pub use raises::{conjuncts, may_raise};
pub use walk::{MAX_DEPTH, check_depth};
