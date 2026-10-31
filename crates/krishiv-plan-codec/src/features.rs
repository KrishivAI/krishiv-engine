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
use krishiv_exec::aggregate::{AggregateExpr, AggregateFunction, is_moment};
use krishiv_exec::physical::{HashScheme, Partitioning, PhysNode, PhysicalPlan, ProjectItemPlan};
use krishiv_exec::window::{FrameBound, WindowFrame, WindowFunction};
use krishiv_plan::{Expr, ExprNode, FunctionId, Literal, ResolvedExpr};
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
    (aggregate_filter, AggregateFilter, POST_MIGRATION_READER_VERSION, Semantic, "An aggregate's `FILTER (WHERE ...)` mask column."),
    (collect_aggregates, CollectAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The collect and vector aggregates (`collect_list`, `collect_set`, `vector_sum`, `vector_avg`)."),
    (existence_join, ExistenceJoin, POST_MIGRATION_READER_VERSION, Semantic, "The `Existence` join type (Spark's `ExistenceJoin`: `EXISTS` and `IN` outside a filter conjunct), hash or nested loop."),
    (expand, Expand, 5, Semantic, "The `Expand` node and its `NullOf` project item (ROLLUP, CUBE, GROUPING SETS)."),
    (geo_types, GeoTypes, POST_MIGRATION_READER_VERSION, Semantic, "The `GEOMETRY` and `GEOGRAPHY` types, with their SRID."),
    (join_build_preserved, JoinBuildPreserved, POST_MIGRATION_READER_VERSION, Optimization, "Joins that build their preserved side (`RightSemi`, `RightAnti`, `RightOuter`); without it the planner builds the other side (`LeftSemi`, `LeftAnti`, `LeftOuter`)."),
    (join_full_outer, JoinFullOuter, POST_MIGRATION_READER_VERSION, Semantic, "The `FULL_OUTER` hash join: both sides preserved, with no equivalent plan without it."),
    (join_single_match, JoinSingleMatch, POST_MIGRATION_READER_VERSION, Semantic, "The single-match flag of a `LeftOuter` hash join (a scalar subquery: more than one match per probe row is an error)."),
    (key_group_hash, KeyGroupHash, POST_MIGRATION_READER_VERSION, Semantic, "Hash exchanges that route by key-group ranges of the routing hash `H_kg` version 1 (XXH64, seed 0, of the normalized key bytes, folded over several columns)."),
    (lambda_functions, LambdaFunctions, POST_MIGRATION_READER_VERSION, Semantic, "Lambda expressions (`x -> body`) and variables: the bound bodies of higher-order function arguments (`transform`, `filter`, `exists`, `forall`)."),
    (legacy_statistical_aggregate, LegacyStatisticalAggregate, POST_MIGRATION_READER_VERSION, Semantic, "The `legacy_statistical_aggregate` expression environment flag (`spark.sql.legacy.statisticalAggregate`): the sample moment aggregates give NaN instead of NULL where Spark divides by zero."),
    (limit_offset, LimitOffset, 3, Semantic, "A `Limit` that skips leading rows (a nonzero offset)."),
    (map_key_dedup_policy, MapKeyDedupPolicy, POST_MIGRATION_READER_VERSION, Semantic, "The session's map-key dedup policy (`LAST_WIN`); without it a worker keeps the first value on duplicate map keys."),
    (mode_regr_bit_listagg_aggs, ModeRegrBitListaggAggs, POST_MIGRATION_READER_VERSION, Semantic, "The `mode`, regression (`regr_*`), bitwise (`bit_and`, `bit_or`, `bit_xor`), `listagg`/`string_agg` and `try_sum`/`try_avg` aggregates."),
    (nested_types, NestedTypes, POST_MIGRATION_READER_VERSION, Semantic, "Array, map and struct types (`DataType.KIND_NESTED`, written as Spark's JSON form of the type)."),
    (partitioned_writes, PartitionedWrites, POST_MIGRATION_READER_VERSION, Semantic, "A `WritePlan` laid out by partition columns (Spark's directory layout)."),
    (range, Range, POST_MIGRATION_READER_VERSION, Semantic, "The `Range` node (`range(start, end, step, numSlices)`)."),
    (rust_udf, RustUdf, POST_MIGRATION_READER_VERSION, Semantic, "A call of a session-registered native Rust UDF (`UdfFunction`, by name and version)."),
    (session_zone_name, SessionZoneName, POST_MIGRATION_READER_VERSION, Semantic, "The configured session time zone name (`spark.sql.session.timeZone`) where it differs from the rendered offset; without it a worker shows the offset in `current_timezone()` and `VV`/`z` patterns."),
    (shared_subplan, SharedSubplan, POST_MIGRATION_READER_VERSION, Semantic, "Subplans run once per query and read by several `SharedRead` nodes (`WITH` bodies computed once)."),
    (struct_field, StructField, POST_MIGRATION_READER_VERSION, Semantic, "A struct field access (`GetStructField`: the analyzer-resolved field of a struct)."),
    (time_type, TimeType, POST_MIGRATION_READER_VERSION, Semantic, "The `TIME` type, with its precision."),
    (window, Window, 4, Semantic, "The `Window` node and its functions."),
    (window_frame_offsets, WindowFrameOffsets, POST_MIGRATION_READER_VERSION, Semantic, "`ROWS` window frame bounds at an offset from the current row (`n PRECEDING`, `n FOLLOWING`)."),
    (window_group_limit, WindowGroupLimit, POST_MIGRATION_READER_VERSION, Optimization, "The `WindowGroupLimit` node (the rows of each partition within a rank)."),
    (window_offset_functions, WindowOffsetFunctions, POST_MIGRATION_READER_VERSION, Semantic, "The window functions `lag`, `lead` and `nth_value`."),
    (write_plan, WritePlan, POST_MIGRATION_READER_VERSION, Semantic, "The `WriteFiles` node's `WritePlan` (format, store, directory, job id and file limits): the file sink is built at instantiation from it with the task's partition and attempt."),
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
    /// An aggregate expression (of an aggregate or a window): whether it
    /// reads only the rows of a mask column.
    Aggregate(&'a AggregateExpr),
    /// An aggregate function (of an aggregate or a window).
    AggregateFunction(&'a AggregateFunction),
    /// A resolved function call: only a session UDF needs a feature.
    Function(&'a FunctionId),
    /// A resolved expression (a struct field access needs its feature).
    Expr(&'a Expr),
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

/// The feature a type needs on its own. The walk reports a type inside an
/// array, map or struct as a construct of its own ([`visit_type`]), so
/// `array<time(3)>` needs `nested_types` (the array) and `time_type` (its
/// element). `DataType` is non-exhaustive, so this cannot list every type: a
/// type added later fails to encode (`encode::spark_type`) until it is
/// mapped there, and classified here.
fn type_feature(t: &DataType) -> Option<Feature> {
    match t {
        DataType::Time { .. } => Some(Feature::TimeType),
        DataType::Geometry { .. } | DataType::Geography { .. } => Some(Feature::GeoTypes),
        _ => t.is_nested().then_some(Feature::NestedTypes),
    }
}

/// `t` and every type inside it (array elements, map keys and values, struct
/// fields) as constructs.
fn visit_type<'a>(t: &'a DataType, f: &mut impl FnMut(Construct<'a>)) {
    f(Construct::Type(t));
    match t {
        DataType::Array(a) => visit_type(a.element_type(), f),
        DataType::Map(m) => {
            visit_type(m.key_type(), f);
            visit_type(m.value_type(), f);
        }
        DataType::Struct(s) => s.fields().iter().for_each(|x| visit_type(x.data_type(), f)),
        _ => {}
    }
}

/// `t` and every Arrow type inside it as constructs.
fn visit_arrow<'a>(t: &'a arrow_schema::DataType, f: &mut impl FnMut(Construct<'a>)) {
    use arrow_schema::DataType as A;
    f(Construct::ArrowType(t));
    match t {
        A::List(x)
        | A::LargeList(x)
        | A::FixedSizeList(x, _)
        | A::ListView(x)
        | A::LargeListView(x)
        | A::Map(x, _) => visit_arrow(x.data_type(), f),
        A::Struct(fields) => fields.iter().for_each(|x| visit_arrow(x.data_type(), f)),
        A::Dictionary(_, v) => visit_arrow(v, f),
        A::RunEndEncoded(_, v) => visit_arrow(v.data_type(), f),
        _ => {}
    }
}

/// Whether an Arrow type is a time-of-day column, `TIME`'s carrier.
fn arrow_time(t: &arrow_schema::DataType) -> bool {
    matches!(
        t,
        arrow_schema::DataType::Time32(_) | arrow_schema::DataType::Time64(_)
    )
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
#[expect(
    clippy::match_same_arms,
    reason = "every current FunctionId is listed explicitly"
)]
pub fn feature_of(construct: Construct<'_>) -> Option<Feature> {
    match construct {
        Construct::Node(node) => match node {
            PhysNode::Limit { offset, .. } => (*offset != 0).then_some(Feature::LimitOffset),
            PhysNode::HashJoin(j) => j.single_match.then_some(Feature::JoinSingleMatch),
            PhysNode::Window { .. } => Some(Feature::Window),
            PhysNode::Expand { .. } => Some(Feature::Expand),
            PhysNode::Range(_) => Some(Feature::Range),
            PhysNode::SharedRead { .. } => Some(Feature::SharedSubplan),
            PhysNode::WindowGroupLimit { .. } => Some(Feature::WindowGroupLimit),
            // A partitioned write needs a reader that
            // lays files out by partition (the older fields would be
            // ignored and the rows written flat).
            PhysNode::WriteFiles { write, .. } => write.as_ref().map(|w| {
                if w.partition_by.is_empty() {
                    Feature::WritePlan
                } else {
                    Feature::PartitionedWrites
                }
            }),
            PhysNode::Scan(_)
            | PhysNode::OneRow
            | PhysNode::Filter { .. }
            | PhysNode::Project { .. }
            | PhysNode::ReplicateRows { .. }
            | PhysNode::Sort { .. }
            | PhysNode::TopN { .. }
            | PhysNode::Aggregate { .. }
            | PhysNode::NestedLoopJoin(_)
            | PhysNode::Union { .. }
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
            JoinType::Existence => Some(Feature::ExistenceJoin),
            JoinType::RightSemi | JoinType::RightAnti | JoinType::RightOuter => {
                Some(Feature::JoinBuildPreserved)
            }
            JoinType::FullOuter => Some(Feature::JoinFullOuter),
        },
        Construct::Partitioning(p) => match p {
            Partitioning::Hash { scheme, .. } => match scheme {
                HashScheme::Unpinned => None,
                HashScheme::KeyGroupV1 => Some(Feature::KeyGroupHash),
            },
            Partitioning::Single
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
        Construct::Type(t) => type_feature(t),
        Construct::ArrowType(t) => {
            if arrow_time(t) {
                Some(Feature::TimeType)
            } else {
                arrow_nested(t).then_some(Feature::NestedTypes)
            }
        }
        Construct::Expr(e) => match &e.node {
            ExprNode::Resolved(ResolvedExpr::Field { .. }) => Some(Feature::StructField),
            _ => None,
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
        Construct::Aggregate(a) => a.mask.is_some().then_some(Feature::AggregateFilter),
        Construct::AggregateFunction(f) => match f {
            AggregateFunction::Count
            | AggregateFunction::Sum
            | AggregateFunction::Avg
            | AggregateFunction::Min
            | AggregateFunction::Max
            | AggregateFunction::First { .. }
            | AggregateFunction::Last { .. }
            // These are not encodable yet: the encoder refuses them, so
            // none reaches a worker.
            | AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarSamp
            | AggregateFunction::VarPop
            | AggregateFunction::Skewness
            | AggregateFunction::Kurtosis
            | AggregateFunction::Corr
            | AggregateFunction::CovarSamp
            | AggregateFunction::CovarPop
            | AggregateFunction::AnyValue { .. }
            | AggregateFunction::MaxBy
            | AggregateFunction::MinBy => None,
            AggregateFunction::CollectList
            | AggregateFunction::CollectSet
            | AggregateFunction::VectorSum
            | AggregateFunction::VectorAvg => Some(Feature::CollectAggregates),
            AggregateFunction::BitAnd
            | AggregateFunction::BitOr
            | AggregateFunction::BitXor
            | AggregateFunction::RegrCount
            | AggregateFunction::RegrAvgx
            | AggregateFunction::RegrAvgy
            | AggregateFunction::RegrSxx
            | AggregateFunction::RegrSyy
            | AggregateFunction::RegrSxy
            | AggregateFunction::RegrSlope
            | AggregateFunction::RegrIntercept
            | AggregateFunction::RegrR2
            | AggregateFunction::Mode
            | AggregateFunction::ModeMin
            | AggregateFunction::ModeMax
            | AggregateFunction::ListAgg { .. }
            | AggregateFunction::TrySum
            | AggregateFunction::TryAvg => Some(Feature::ModeRegrBitListaggAggs),
        },
        // Built-ins and registry functions travel by name; only a
        // session UDF (by name and version) needs a reader with it.
        Construct::Function(id) => match id {
            FunctionId::Udf { .. } => Some(Feature::RustUdf),
            FunctionId::Builtin(_) => None,
            // A future identity is baseline until it is classified
            // (encoding refuses it first).
            FunctionId::Registry { .. } | _ => None,
        },
    }
}

/// Calls `f` with an aggregate expression and its function.
fn aggregate_constructs<'a>(a: &'a AggregateExpr, f: &mut impl FnMut(Construct<'a>)) {
    f(Construct::Aggregate(a));
    f(Construct::AggregateFunction(&a.function));
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
                    aggregate_constructs(a, f);
                }
            }
        }
        PhysNode::Aggregate { aggregate, .. } => aggregate
            .aggregates
            .iter()
            .for_each(|a| aggregate_constructs(a, f)),
        PhysNode::HashJoin(j) => f(Construct::Join(j.join_type)),
        PhysNode::NestedLoopJoin(j) => f(Construct::Join(j.join_type)),
        PhysNode::Exchange(e) => f(Construct::Partitioning(&e.partitioning)),
        PhysNode::Scan(_)
        | PhysNode::OneRow
        | PhysNode::SharedRead { .. }
        | PhysNode::Range(_)
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
        visit_type(t, f);
    }
    for t in arrow_types_of(node) {
        visit_arrow(t, f);
    }
    for e in exprs_of(node) {
        if matches!(
            &e.node,
            ExprNode::Lambda { .. } | ExprNode::Resolved(ResolvedExpr::LambdaVar(_))
        ) {
            f(Construct::Lambda);
        }
        f(Construct::Expr(e));
        // A UDF call hides in any expression of the node (`exprs_of`
        // already yields every descendant, so one node check is enough).
        functions_in(e, f);
    }
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
    for e in exprs_of(node) {
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
            ExprNode::Resolved(ResolvedExpr::Field { data_type, .. }) => out.push(data_type),
            _ => {}
        }
    }
    if let PhysNode::Scan(scan) = node {
        out.extend(scan.scan.output.iter().map(|c| &c.data_type));
    }
    out
}

/// Every expression `node` holds, each once (shared subtrees visited once).
fn exprs_of<'a>(node: &'a PhysNode) -> Vec<&'a Arc<Expr>> {
    fn items(items: &[ProjectItemPlan]) -> impl Iterator<Item = &Arc<Expr>> {
        items.iter().filter_map(|i| match i {
            ProjectItemPlan::Expr { expr, .. } => Some(expr),
            ProjectItemPlan::Column { .. } | ProjectItemPlan::NullOf { .. } => None,
        })
    }
    let mut roots: Vec<&'a Arc<Expr>> = Vec::new();
    match node {
        PhysNode::Scan(scan) => {
            roots.extend(scan.filter.iter().map(|f| &f.predicate));
        }
        PhysNode::Filter { predicate, .. } => roots.push(predicate),
        PhysNode::Project { items: i, .. } => roots.extend(items(i)),
        PhysNode::Expand { projections, .. } => {
            roots.extend(projections.iter().flat_map(|p| items(p)));
        }
        PhysNode::ReplicateRows { count, .. } => roots.push(count),
        PhysNode::HashJoin(j) => roots.extend(j.residual.iter().flat_map(|r| &r.conjuncts)),
        PhysNode::NestedLoopJoin(j) => roots.extend(j.residual.iter().flat_map(|r| &r.conjuncts)),
        PhysNode::OneRow
        | PhysNode::SharedRead { .. }
        | PhysNode::Limit { .. }
        | PhysNode::Range(_)
        | PhysNode::Sort { .. }
        | PhysNode::TopN { .. }
        | PhysNode::Window { .. }
        | PhysNode::WindowGroupLimit { .. }
        | PhysNode::Aggregate { .. }
        | PhysNode::Union { .. }
        | PhysNode::WriteFiles { .. }
        | PhysNode::Exchange(_) => {}
    }
    // Expressions share subtrees: each is visited once.
    let mut seen = std::collections::HashSet::new();
    let mut stack = roots;
    let mut out = Vec::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(Arc::as_ptr(e)) {
            continue;
        }
        out.push(e);
        e.for_each_child(&mut |c| stack.push(c));
    }
    out
}

/// Calls `f` with the function call `e` holds, if one: only a session UDF
/// needs a feature, so only that arm matters. The call site already visits
/// every descendant (see `exprs_of`), so one node check covers the whole
/// subtree. Every current [`ExprNode`] variant appears in the call-site
/// walk explicitly, so a new expression kind holding a call cannot
/// silently skip UDF feature detection (encoding refuses unknown nodes
/// first). Physical expressions hold no subquery nodes (the codec refuses
/// them).
fn functions_in<'a>(e: &'a Arc<Expr>, f: &mut impl FnMut(Construct<'a>)) {
    if let ExprNode::Resolved(ResolvedExpr::Function(c)) = &e.node {
        f(Construct::Function(&c.id));
    }
}

/// The features `plan` uses: a pure function of its content. Subquery plans
/// count; nothing recurses. The expression environment counts too, but only
/// around a moment aggregate: a plan whose session sets
/// `spark.sql.legacy.statisticalAggregate` needs a reader that carries the
/// flag, or it would compute NULL where Spark gives NaN. A plan without one
/// runs the same everywhere, so the flag alone lists no feature and readers
/// without it keep reading the plan.
#[must_use]
pub fn plan_features(plan: &PhysicalPlan) -> BTreeSet<Feature> {
    let mut out = BTreeSet::new();
    if plan.env().map_key_dedup_last_win {
        out.insert(Feature::MapKeyDedupPolicy);
    }
    if plan.env().zone_name.is_some() {
        out.insert(Feature::SessionZoneName);
    }
    let mut has_moment = false;
    if !plan.shared().is_empty() {
        out.insert(Feature::SharedSubplan);
    }
    let mut stack: Vec<&PhysNode> = plan.roots().collect();
    while let Some(node) = stack.pop() {
        constructs_of(node, &mut |c| {
            if matches!(c, Construct::Aggregate(a) if is_moment(&a.function)) {
                has_moment = true;
            }
            out.extend(feature_of(c));
        });
        stack.extend(node.children());
    }
    for field in plan.schema().fields() {
        visit_arrow(field.data_type(), &mut |c| out.extend(feature_of(c)));
    }
    if has_moment && plan.env().legacy_statistical_aggregate {
        out.insert(Feature::LegacyStatisticalAggregate);
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
