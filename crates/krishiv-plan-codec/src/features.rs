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
use krishiv_exec::physical::{Partitioning, PhysNode, PhysicalPlan, ProjectItemPlan};
use krishiv_exec::window::{FrameBound, WindowFrame, WindowFunction};
use krishiv_plan::{Expr, ExprNode, Literal, ResolvedExpr};
use krishiv_types::DataType;

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
    (lambda_functions, LambdaFunctions, POST_MIGRATION_READER_VERSION, Semantic, "Lambda expressions (`x -> body`) and variables: the bound bodies of higher-order function arguments (`transform`, `filter`, `exists`, `forall`)."),
    (limit_offset, LimitOffset, 3, Semantic, "A `Limit` that skips leading rows (a nonzero offset)."),
    (nested_types, NestedTypes, POST_MIGRATION_READER_VERSION, Semantic, "Array, map and struct types (`DataType.KIND_NESTED`, written as Spark's JSON form of the type)."),
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
    /// A Spark type in an expression or a column.
    Type(&'a DataType),
    /// An Arrow type the plan carries as IPC bytes (an aggregate's argument,
    /// a subquery's value, a runtime filter's key, a window offset function's
    /// value, the plan's output schema).
    ArrowType(&'a arrow_schema::DataType),
    /// A lambda expression or variable: the bound body of a higher-order
    /// function's argument.
    Lambda,
}

/// Whether an Arrow type is or contains an array, map or struct.
fn arrow_nested(t: &arrow_schema::DataType) -> bool {
    use arrow_schema::DataType as A;
    match t {
        A::List(_)
        | A::LargeList(_)
        | A::FixedSizeList(..)
        | A::ListView(_)
        | A::LargeListView(_)
        | A::Map(..)
        | A::Struct(_) => true,
        A::Dictionary(_, value) => arrow_nested(value),
        A::RunEndEncoded(_, value) => arrow_nested(value.data_type()),
        _ => false,
    }
}

/// Whether the encoder must list the nested-types feature for an Arrow type
/// it writes (kept in step with [`feature_of`] for [`Construct::ArrowType`]).
pub(crate) fn arrow_type_is_nested(t: &arrow_schema::DataType) -> bool {
    arrow_nested(t)
}

/// The feature `construct` needs, or `None` for the epoch-1 baseline.
#[must_use]
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
        // A lambda is plan data behind its own feature: a reader without
        // it misreads the higher-order call's argument.
        Construct::Lambda => Some(Feature::LambdaFunctions),
        // A type Krishiv does not know cannot be encoded at all. A later
        // feature for a type that can sit inside an array, map or struct
        // (variant, char/varchar, intervals) must look through them: its
        // arm walks the element, key, value and field types, so that
        // `array<variant>` needs it too.
        Construct::Type(t) => t.is_nested().then_some(Feature::NestedTypes),
        Construct::ArrowType(t) => arrow_nested(t).then_some(Feature::NestedTypes),
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
    for t in types_of(node) {
        f(Construct::Type(t));
    }
    for t in arrow_types_of(node) {
        f(Construct::ArrowType(t));
    }
    if exprs_have_lambda(node) {
        f(Construct::Lambda);
    }
}

/// Whether any expression `node` holds is or contains a lambda: the bound
/// body of a higher-order function's argument.
fn exprs_have_lambda(node: &PhysNode) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = expr_roots(node);
    while let Some(e) = stack.pop() {
        if !seen.insert(Arc::as_ptr(e)) {
            continue;
        }
        match &e.node {
            ExprNode::Lambda { .. } | ExprNode::Resolved(ResolvedExpr::LambdaVar(_)) => {
                return true;
            }
            _ => {}
        }
        e.for_each_child(&mut |c| stack.push(c));
    }
    false
}

/// The expressions `node` holds (its scan filter, project items and the
/// like): [`types_of`] and [`exprs_have_lambda`] walk these.
fn expr_roots(node: &PhysNode) -> Vec<&Arc<Expr>> {
    use krishiv_exec::physical::ProjectItemPlan as Item;
    let mut roots: Vec<&Arc<Expr>> = Vec::new();
    match node {
        PhysNode::Scan(scan) => {
            roots.extend(scan.filter.iter().map(|f| &f.predicate));
        }
        PhysNode::Filter { predicate, .. } => roots.push(predicate),
        PhysNode::Project { items: i, .. } => roots.extend(i.iter().filter_map(|p| match p {
            Item::Expr { expr, .. } => Some(expr),
            Item::Column { .. } | Item::NullOf { .. } => None,
        })),
        PhysNode::Expand { projections, .. } => {
            roots.extend(projections.iter().flat_map(|p| {
                p.iter().filter_map(|p| match p {
                    Item::Expr { expr, .. } => Some(expr),
                    Item::Column { .. } | Item::NullOf { .. } => None,
                })
            }));
        }
        PhysNode::ReplicateRows { count, .. } => roots.push(count),
        PhysNode::HashJoin(j) => roots.extend(j.residual.iter().flat_map(|r| &r.conjuncts)),
        PhysNode::NestedLoopJoin(j) => roots.extend(j.residual.iter().flat_map(|r| &r.conjuncts)),
        PhysNode::OneRow
        | PhysNode::Limit { .. }
        | PhysNode::Sort { .. }
        | PhysNode::TopN { .. }
        | PhysNode::Window { .. }
        | PhysNode::WindowGroupLimit { .. }
        | PhysNode::Aggregate { .. }
        | PhysNode::Union { .. }
        | PhysNode::WriteFiles { .. }
        | PhysNode::Exchange(_) => {}
    }
    roots
}

/// The Arrow types `node` carries as IPC bytes.
fn arrow_types_of(node: &PhysNode) -> Vec<&arrow_schema::DataType> {
    fn window<'a>(function: &'a WindowFunction, out: &mut Vec<&'a arrow_schema::DataType>) {
        match function {
            WindowFunction::Aggregate(a) => out.push(&a.arg_type),
            WindowFunction::Offset { data_type, .. }
            | WindowFunction::NthValue { data_type, .. } => out.push(data_type),
            WindowFunction::RowNumber
            | WindowFunction::Rank
            | WindowFunction::DenseRank
            | WindowFunction::PercentRank
            | WindowFunction::CumeDist
            | WindowFunction::Ntile(_) => {}
        }
    }
    let mut out = Vec::new();
    let subqueries = match node {
        PhysNode::Filter { subqueries, .. }
        | PhysNode::Project { subqueries, .. }
        | PhysNode::Expand { subqueries, .. } => Some(subqueries),
        _ => None,
    };
    match node {
        PhysNode::Window { window: w, .. } => {
            w.functions
                .iter()
                .for_each(|f| window(&f.function, &mut out));
        }
        PhysNode::Aggregate { aggregate, .. } => {
            out.extend(aggregate.aggregates.iter().map(|a| &a.arg_type));
        }
        PhysNode::Scan(scan) => out.extend(scan.runtime_filters.iter().map(|f| &f.key_type)),
        _ => {}
    }
    out.extend(subqueries.into_iter().flatten().map(|s| &s.data_type));
    out
}

/// The Spark types `node` holds: those of its scan columns and of every
/// expression in it (columns, casts, functions, aliases, typed NULLs).
fn types_of<'a>(node: &'a PhysNode) -> Vec<&'a DataType> {
    let mut out: Vec<&'a DataType> = Vec::new();
    if let PhysNode::Scan(scan) = node {
        out.extend(scan.scan.output.iter().map(|c| &c.data_type));
    }
    let roots: Vec<&'a Arc<Expr>> = expr_roots(node);
    // Expressions share subtrees: each is visited once.
    let mut seen = std::collections::HashSet::new();
    let mut stack = roots;
    while let Some(e) = stack.pop() {
        if !seen.insert(Arc::as_ptr(e)) {
            continue;
        }
        match &e.node {
            ExprNode::Cast { to, .. } => out.push(to),
            ExprNode::Literal(Literal::Null(Some(t))) => out.push(t),
            ExprNode::Resolved(
                ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) | ResolvedExpr::LambdaVar(c),
            ) => {
                out.push(&c.data_type);
            }
            ExprNode::Resolved(ResolvedExpr::Function(f)) => out.push(&f.data_type),
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => out.push(&output.data_type),
            _ => {}
        }
        e.for_each_child(&mut |c| stack.push(c));
    }
    out
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
    for field in plan.schema().fields() {
        out.extend(feature_of(Construct::ArrowType(field.data_type())));
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
