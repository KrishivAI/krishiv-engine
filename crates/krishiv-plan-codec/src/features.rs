// SPDX-License-Identifier: Apache-2.0
//! Plan features (ADR 0040): what a reader must implement to read a plan.
//!
//! A construct needs a feature when a reader without it would misread it or
//! reject it with an unhelpful error: a new node kind, a new project item or
//! enum value, a new field whose absence changes results. Everything else is
//! *ignorable* and needs no feature: an older reader skips the unknown field
//! (a hint, an explain detail) and still computes the same result. There is
//! no third class.
//!
//! The header of an encoded plan lists exactly the features the plan uses,
//! computed from the plan alone ([`plan_features`]), so the same plan always
//! gets the same header wherever it is encoded. A reader checks the header
//! before it reads a node and refuses a plan that uses a feature it lacks,
//! naming it ([`CodecError::UnsupportedFeatures`](crate::CodecError)).
//!
//! # Adding a feature
//! 1. One line in the registry below, at its **alphabetical** place (two
//!    changes adding features then touch different lines). Ids are
//!    `snake_case`, never renamed and never reused; a retired id moves to
//!    [`RETIRED`], which happens only at an epoch change.
//! 2. The arms of [`feature_of`] for the new constructs: its matches have no
//!    wildcard, so a new node kind, project item, join type, partitioning,
//!    window function or aggregate function does not compile until it is
//!    classified.
//! 3. A golden directory `tests/golden/<id>/` with the cases that use it.
//! 4. A `// feature: <id>` annotation on each new proto element
//!    (`tools/plan-protogen --check` enforces it).
//!
//! A new feature's legacy level is [`POST_MIGRATION_READER_VERSION`].
//!
//! Semantic drift inside an existing construct (a function that computes
//! something else under the same name) is not a codec matter: functions
//! travel by name and registry id, and the function registry's version
//! covers them.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use krishiv_exec::JoinType;
use krishiv_exec::aggregate::AggregateFunction;
use krishiv_exec::physical::{Partitioning, PhysNode, PhysicalPlan, ProjectItemPlan, Residual};
use krishiv_exec::window::{FrameBound, WindowFrame, WindowFunction};
use krishiv_plan::{Expr, ExprNode, FunctionId, ResolvedExpr, SubqueryKind};

/// The `min_reader_version` of a plan that uses no feature (the epoch-1
/// baseline): the format of version 2.
pub const MIN_READER_VERSION: u32 = 2;

/// The legacy `min_reader_version` of every feature added after the format
/// version was frozen: above every frozen format version, so a reader that
/// predates features refuses the plan instead of misreading it.
pub const POST_MIGRATION_READER_VERSION: u32 = 6;

/// Whether the planner can avoid a feature, which decides what happens while
/// some workers lack it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FeatureKind {
    /// The planner has an equivalent plan without it: it masks the feature
    /// while a live worker lacks it.
    Optimization,
    /// No alternative plan exists: such a plan runs only on workers that
    /// have the feature.
    Semantic,
}

macro_rules! registry {
    ($(($id:ident, $variant:ident, $legacy:expr, $kind:ident, $meaning:literal),)*) => {
        /// A plan feature. The order is the registry's (alphabetical by id).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Feature {
            $(#[doc = $meaning] $variant,)*
        }

        impl Feature {
            /// Every feature this reader implements, in registry order.
            pub const ALL: &[Self] = &[$(Self::$variant),*];

            /// The stable id written in a plan's header.
            #[must_use]
            pub const fn id(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($id),)*
                }
            }

            /// The `min_reader_version` a plan using this feature carries, for
            /// readers that predate features.
            #[must_use]
            pub const fn legacy_level(self) -> u32 {
                match self {
                    $(Self::$variant => $legacy,)*
                }
            }

            /// Whether the planner can avoid it.
            #[must_use]
            pub const fn kind(self) -> FeatureKind {
                match self {
                    $(Self::$variant => FeatureKind::$kind,)*
                }
            }

            /// What it covers, in one line.
            #[must_use]
            pub const fn meaning(self) -> &'static str {
                match self {
                    $(Self::$variant => $meaning,)*
                }
            }
        }
    };
}

// The registry: one line per feature, kept alphabetically sorted by id.
// (id, variant, legacy level, kind, meaning)
registry! {
    (expand, Expand, 5, Semantic, "The `Expand` node and its `NullOf` project item (ROLLUP, CUBE, GROUPING SETS)."),
    (join_build_preserved, JoinBuildPreserved, POST_MIGRATION_READER_VERSION, Optimization, "Joins that build their preserved side (`RightSemi`, `RightAnti`, `RightOuter`); without it the planner builds the other side (`LeftSemi`, `LeftAnti`, `LeftOuter`)."),
    (limit_offset, LimitOffset, 3, Semantic, "A `Limit` that skips leading rows (a nonzero offset)."),
    (rust_udf, RustUdf, POST_MIGRATION_READER_VERSION, Semantic, "A call of a session-registered native Rust UDF (`UdfFunction`, by name and version)."),
    (window, Window, 4, Semantic, "The `Window` node and its functions."),
    (window_frame_offsets, WindowFrameOffsets, POST_MIGRATION_READER_VERSION, Semantic, "`ROWS` window frame bounds at an offset from the current row (`n PRECEDING`, `n FOLLOWING`)."),
    (window_group_limit, WindowGroupLimit, POST_MIGRATION_READER_VERSION, Optimization, "The `WindowGroupLimit` node (the rows of each partition within a rank)."),
    (window_offset_functions, WindowOffsetFunctions, POST_MIGRATION_READER_VERSION, Semantic, "The window functions `lag`, `lead` and `nth_value`."),
}

/// Ids retired at an epoch change: never reused.
pub const RETIRED: &[&str] = &[];

impl Feature {
    /// The feature with id `id`, if this reader implements it.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|f| f.id() == id)
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// A construct of a plan that may need a feature.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Construct<'a> {
    /// A plan node (not its children).
    Node(&'a PhysNode),
    /// A project item.
    Item(&'a ProjectItemPlan),
    /// A join type.
    Join(JoinType),
    /// An exchange's partitioning.
    Partitioning(&'a Partitioning),
    /// A window function.
    Window(&'a WindowFunction),
    /// A window function's frame.
    Frame(&'a WindowFrame),
    /// An aggregate function (of an aggregate or a window).
    Aggregate(AggregateFunction),
    /// A resolved function call: only a session UDF needs a feature.
    Function(&'a FunctionId),
}

/// The feature `construct` needs, or `None` for the epoch-1 baseline.
#[must_use]
#[expect(
    clippy::match_same_arms,
    reason = "every current FunctionId is listed explicitly"
)]
pub fn feature_of(construct: Construct<'_>) -> Option<Feature> {
    match construct {
        Construct::Node(node) => match node {
            PhysNode::Limit { offset, .. } => (*offset != 0).then_some(Feature::LimitOffset),
            PhysNode::Window { .. } => Some(Feature::Window),
            PhysNode::Expand { .. } => Some(Feature::Expand),
            PhysNode::WindowGroupLimit { .. } => Some(Feature::WindowGroupLimit),
            PhysNode::Scan(_)
            | PhysNode::OneRow
            | PhysNode::Filter { .. }
            | PhysNode::Project { .. }
            | PhysNode::ReplicateRows { .. }
            | PhysNode::Sort { .. }
            | PhysNode::TopN { .. }
            | PhysNode::Aggregate { .. }
            | PhysNode::HashJoin(_)
            | PhysNode::NestedLoopJoin(_)
            | PhysNode::Union { .. }
            | PhysNode::WriteFiles { .. }
            | PhysNode::Exchange(_) => None,
        },
        Construct::Item(item) => match item {
            ProjectItemPlan::Column { .. } | ProjectItemPlan::Expr { .. } => None,
            ProjectItemPlan::NullOf { .. } => Some(Feature::Expand),
        },
        Construct::Join(join) => match join {
            JoinType::Inner
            | JoinType::LeftOuter
            | JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::NullAwareAnti => None,
            JoinType::RightSemi | JoinType::RightAnti | JoinType::RightOuter => {
                Some(Feature::JoinBuildPreserved)
            }
        },
        Construct::Partitioning(p) => match p {
            Partitioning::Hash { .. }
            | Partitioning::Single
            | Partitioning::Broadcast
            | Partitioning::RoundRobin
            | Partitioning::Range { .. } => None,
        },
        // Every window function came with the `Window` node.
        Construct::Window(w) => match w {
            WindowFunction::RowNumber
            | WindowFunction::Rank
            | WindowFunction::DenseRank
            | WindowFunction::PercentRank
            | WindowFunction::CumeDist
            | WindowFunction::Ntile(_)
            | WindowFunction::Aggregate(_) => Some(Feature::Window),
            WindowFunction::Offset { .. } | WindowFunction::NthValue { .. } => {
                Some(Feature::WindowOffsetFunctions)
            }
        },
        // The `Window` node's format carries every frame between unbounded
        // and current-row bounds; an offset bound is newer. `FrameBound` is
        // non-exhaustive, so a bound added later counts as newer too.
        Construct::Frame(frame) => {
            let newer = |b: FrameBound| {
                !matches!(
                    b,
                    FrameBound::UnboundedPreceding
                        | FrameBound::CurrentRow
                        | FrameBound::UnboundedFollowing
                )
            };
            (newer(frame.lower) || newer(frame.upper)).then_some(Feature::WindowFrameOffsets)
        }
        Construct::Aggregate(a) => match a {
            AggregateFunction::Count
            | AggregateFunction::Sum
            | AggregateFunction::Avg
            | AggregateFunction::Min
            | AggregateFunction::Max
            | AggregateFunction::First { .. }
            | AggregateFunction::Last { .. }
            | AggregateFunction::AnyValue { .. }
            | AggregateFunction::MaxBy
            | AggregateFunction::MinBy => None,
        },
        // Built-ins and registry functions travel by name; only a
        // session UDF (by name and version) needs a reader with it.
        Construct::Function(id) => match id {
            FunctionId::Udf { .. } => Some(Feature::RustUdf),
            FunctionId::Builtin(_) => None,
            FunctionId::Registry { .. } => None,
            // A future identity is baseline until it is classified
            // (encoding refuses it first).
            _ => None,
        },
    }
}

/// Calls `f` with `node` and each construct inside it (not its children).
fn constructs_of<'a>(node: &'a PhysNode, f: &mut impl FnMut(Construct<'a>)) {
    f(Construct::Node(node));
    match node {
        PhysNode::Project { items, .. } => items.iter().for_each(|i| f(Construct::Item(i))),
        PhysNode::Expand { projections, .. } => projections
            .iter()
            .flatten()
            .for_each(|i| f(Construct::Item(i))),
        PhysNode::Window { window, .. } => {
            for function in &window.functions {
                f(Construct::Window(&function.function));
                f(Construct::Frame(&function.frame));
                if let WindowFunction::Aggregate(a) = &function.function {
                    f(Construct::Aggregate(a.function));
                }
            }
        }
        PhysNode::Aggregate { aggregate, .. } => aggregate
            .aggregates
            .iter()
            .for_each(|a| f(Construct::Aggregate(a.function))),
        PhysNode::HashJoin(j) => f(Construct::Join(j.join_type)),
        PhysNode::NestedLoopJoin(j) => f(Construct::Join(j.join_type)),
        PhysNode::Exchange(e) => f(Construct::Partitioning(&e.partitioning)),
        PhysNode::Scan(_)
        | PhysNode::OneRow
        | PhysNode::Filter { .. }
        | PhysNode::Limit { .. }
        | PhysNode::ReplicateRows { .. }
        | PhysNode::Sort { .. }
        | PhysNode::TopN { .. }
        | PhysNode::WindowGroupLimit { .. }
        | PhysNode::Union { .. }
        | PhysNode::WriteFiles { .. } => {}
    }
    // A UDF call hides in any expression of the node.
    for e in exprs_of(node) {
        functions_in(e, f);
    }
}

/// Every expression `node` holds (not its children's or its subqueries').
/// Every [`PhysNode`] variant appears here explicitly, so a new node kind
/// holding an expression cannot silently skip UDF feature detection.
fn exprs_of(node: &PhysNode) -> Vec<&Arc<Expr>> {
    match node {
        PhysNode::Filter { predicate, .. } => vec![predicate],
        PhysNode::Project { items, .. } => items
            .iter()
            .filter_map(|i| match i {
                ProjectItemPlan::Expr { expr, .. } => Some(expr),
                ProjectItemPlan::Column { .. } | ProjectItemPlan::NullOf { .. } => None,
            })
            .collect(),
        PhysNode::ReplicateRows { count, .. } => vec![count],
        PhysNode::Expand { projections, .. } => projections
            .iter()
            .flatten()
            .filter_map(|i| match i {
                ProjectItemPlan::Expr { expr, .. } => Some(expr),
                ProjectItemPlan::Column { .. } | ProjectItemPlan::NullOf { .. } => None,
            })
            .collect(),
        PhysNode::Scan(s) => s
            .filter
            .as_ref()
            .map(|f| &f.predicate)
            .into_iter()
            .collect(),
        PhysNode::OneRow
        | PhysNode::Limit { .. }
        | PhysNode::Sort { .. }
        | PhysNode::TopN { .. }
        | PhysNode::Window { .. }
        | PhysNode::WindowGroupLimit { .. }
        | PhysNode::Aggregate { .. }
        | PhysNode::HashJoin(_)
        | PhysNode::NestedLoopJoin(_)
        | PhysNode::Union { .. }
        | PhysNode::WriteFiles { .. }
        | PhysNode::Exchange(_) => Vec::new(),
    }
    .into_iter()
    .chain(residual_of(node))
    .collect()
}

/// The conjuncts of a join's residual, if it has one. Every [`PhysNode`]
/// variant appears here explicitly, so a new join kind cannot silently skip
/// UDF feature detection.
fn residual_of(node: &PhysNode) -> Vec<&Arc<Expr>> {
    match node {
        PhysNode::HashJoin(j) => residual(j.residual.as_ref()),
        PhysNode::NestedLoopJoin(j) => residual(j.residual.as_ref()),
        PhysNode::Scan(_)
        | PhysNode::OneRow
        | PhysNode::Filter { .. }
        | PhysNode::Project { .. }
        | PhysNode::Limit { .. }
        | PhysNode::ReplicateRows { .. }
        | PhysNode::Expand { .. }
        | PhysNode::Sort { .. }
        | PhysNode::TopN { .. }
        | PhysNode::Window { .. }
        | PhysNode::WindowGroupLimit { .. }
        | PhysNode::Aggregate { .. }
        | PhysNode::Union { .. }
        | PhysNode::WriteFiles { .. }
        | PhysNode::Exchange(_) => Vec::new(),
    }
}

fn residual(residual: Option<&Residual>) -> Vec<&Arc<Expr>> {
    residual
        .map(|r| r.conjuncts.iter().collect())
        .unwrap_or_default()
}

/// Calls `f` with every function call `e` holds: only a session UDF needs
/// a feature, so only those arms matter. Every current [`ExprNode`] variant
/// appears here explicitly, so a new expression kind holding a call cannot
/// silently skip UDF feature detection (a future variant hits the wildcard,
/// whose children are still walked below; encoding refuses unknown nodes
/// first). Physical expressions hold no subquery nodes (the codec refuses
/// them); values of one are still entered, so nothing hides.
#[expect(
    clippy::match_same_arms,
    clippy::collapsible_match,
    reason = "every current expression kind is listed explicitly"
)]
fn functions_in<'a>(e: &'a Arc<Expr>, f: &mut impl FnMut(Construct<'a>)) {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(Arc::as_ptr(e)) {
            continue;
        }
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Function(c)) => f(Construct::Function(&c.id)),
            ExprNode::Resolved(
                ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_) | ResolvedExpr::Alias { .. },
            ) => {}
            // Physical expressions hold no subqueries; the values of one
            // are still entered, so nothing hides (`SubqueryKind` is
            // non-exhaustive, so a future kind is ignored here but still
            // walked through `for_each_child` below).
            ExprNode::Subquery { kind, .. } => {
                if let SubqueryKind::In(values) = kind {
                    stack.extend(values.iter());
                }
            }
            ExprNode::Column(_)
            | ExprNode::Literal(_)
            | ExprNode::TypedLiteral { .. }
            | ExprNode::Star(_)
            | ExprNode::Function(_)
            | ExprNode::Cast { .. }
            | ExprNode::Alias { .. }
            | ExprNode::Window(_) => {}
            // A future expression kind: its children are still walked below,
            // so nested calls cannot hide; encoding refuses it first.
            _ => {}
        }
        e.for_each_child(&mut |c| stack.push(c));
    }
}

/// The features `plan` uses: a pure function of its content. Subquery plans
/// count; nothing recurses.
#[must_use]
pub fn plan_features(plan: &PhysicalPlan) -> BTreeSet<Feature> {
    let mut out = BTreeSet::new();
    let mut stack = vec![plan.root()];
    while let Some(node) = stack.pop() {
        constructs_of(node, &mut |c| out.extend(feature_of(c)));
        stack.extend(node.children());
    }
    out
}

/// The legacy `min_reader_version` of a plan that uses `features`.
pub(crate) fn legacy_level(features: &BTreeSet<Feature>) -> u32 {
    features
        .iter()
        .map(|f| f.legacy_level())
        .fold(MIN_READER_VERSION, u32::max)
}

/// The features a reader implements: by default every compiled-in one. A
/// smaller set simulates an older or partial reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderFeatures {
    features: BTreeSet<Feature>,
}

impl Default for ReaderFeatures {
    fn default() -> Self {
        Self::all()
    }
}

impl ReaderFeatures {
    /// Every feature this build implements.
    #[must_use]
    pub fn all() -> Self {
        Self::only(Feature::ALL.iter().copied())
    }

    /// Exactly `features`.
    #[must_use]
    pub fn only(features: impl IntoIterator<Item = Feature>) -> Self {
        Self {
            features: features.into_iter().collect(),
        }
    }

    /// Whether the reader implements `feature`.
    #[must_use]
    pub fn contains(&self, feature: Feature) -> bool {
        self.features.contains(&feature)
    }

    /// The ids of `ids` this reader does not implement, unknown ids
    /// included, in their order.
    #[must_use]
    pub fn missing<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> Vec<String> {
        ids.into_iter()
            .filter(|id| Feature::from_id(id).is_none_or(|f| !self.contains(f)))
            .map(str::to_owned)
            .collect()
    }
}
