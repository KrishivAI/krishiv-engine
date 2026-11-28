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
use krishiv_exec::aggregate::{AggregateExpr, AggregateFunction, AggregateOutput, is_moment};
use krishiv_exec::physical::{
    HashScheme, Partitioning, PhysNode, PhysicalPlan, ProjectItemPlan, TableWriteRequest,
};
use krishiv_exec::planner::WriteMode;
use krishiv_exec::window::{FrameBound, WindowFrame, WindowFunction};
use krishiv_plan::{Expr, ExprNode, FunctionId, Literal, ResolvedExpr};
use krishiv_types::DataType;
use krishiv_types::arrow::SPARK_TYPE_METADATA_KEY;

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
    (any_value_by_aggregates, AnyValueByAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The `any_value`, `max_by` and `min_by` aggregates."),
    (as_of_join, AsOfJoin, POST_MIGRATION_READER_VERSION, Semantic, "A sort-merge join's as-of match (`SortMergeJoinPlan.as_of`, Spark 4.3's `ASOF JOIN`): each left row with the nearest right row of its keys by the as-of columns; without it the join would return every matching pair."),
    (binary_output_style, BinaryOutputStyle, POST_MIGRATION_READER_VERSION, Semantic, "The `binary_output_style` expression environment setting (`spark.sql.binaryOutputStyle`): `to_csv` and `to_xml` print binary fields as UTF-8 text, signed bytes, base64 or hex instead of `[45 61]`."),
    (bitmap_aggregates, BitmapAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The bitmap aggregates (`bitmap_construct_agg`, `bitmap_or_agg`, `bitmap_and_agg`): a 4,096-byte bitmap per group."),
    (calendar_struct, CalendarStruct, POST_MIGRATION_READER_VERSION, Semantic, "CALENDAR INTERVAL columns on the struct carrier (months, days, microseconds, each child marked `krishiv.calendar`; ADR 0009 update 2026-11-06). A reader without it would read the column as a plain STRUCT."),
    (captured_eval_mode, CapturedEvalMode, POST_MIGRATION_READER_VERSION, Semantic, "An expression evaluated in a fixed ANSI mode whatever the query's: the body of a SQL function created under the other mode (T184). A reader without it would evaluate the body in the query's mode."),
    (collated_min_max, CollatedMinMax, POST_MIGRATION_READER_VERSION, Semantic, "`max` and `min` of a collated string (`MAX_BY_KEY_FIRST`, `MIN_BY_KEY_FIRST`): the value of the largest or smallest collation key, the first of equal keys kept."),
    (collated_strings, CollatedStrings, POST_MIGRATION_READER_VERSION, Semantic, "STRING types with a collation other than UTF8_BINARY, by canonical name."),
    (collect_aggregates, CollectAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The collect and vector aggregates (`collect_list`, `collect_set`, `vector_sum`, `vector_avg`)."),
    (exchange_filters, ExchangeFilters, POST_MIGRATION_READER_VERSION, Optimization, "Runtime filters an exchange collects from the rows it writes, for scans of other stages (`ExchangePlan.filters`, distributed stages); without them the fragmenter drops the filters, which never changes rows."),
    (exchange_source, ExchangeSource, POST_MIGRATION_READER_VERSION, Semantic, "The `ExchangeSource` node: a plan fragment's read of an exchange another stage wrote (distributed stages)."),
    (existence_join, ExistenceJoin, POST_MIGRATION_READER_VERSION, Semantic, "The `Existence` join type (Spark's `ExistenceJoin`: `EXISTS` and `IN` outside a filter conjunct), hash or nested loop."),
    (expand, Expand, 5, Semantic, "The `Expand` node and its `NullOf` project item (ROLLUP, CUBE, GROUPING SETS)."),
    (file_source_schema, FileSourceSchema, POST_MIGRATION_READER_VERSION, Semantic, "A user-declared schema on a file read (`ReadSource::Files.schema`). A reader without it would infer the columns again and misread formats such as CSV. The schema is a struct type, so a plan with one lists `nested_types` too."),
    (fixed_size_list_carrier, FixedSizeListCarrier, POST_MIGRATION_READER_VERSION, Semantic, "A vector column on the `FixedSizeList` carrier (design P3-T075): an `array<float>` or `array<double>` stored with its length. A reader without it would read the column through code that expects a plain list."),
    (fused_probe, FusedProbe, POST_MIGRATION_READER_VERSION, Optimization, "A hash join's probe selection (`HashJoinPlan.select`, design P2-E3 U2): a filter and a scan's runtime-filter row stage fused into the probe, which joins the selected rows by id; without it the planner keeps the `Filter` node and the scan's own row stage."),
    (generate, Generate, POST_MIGRATION_READER_VERSION, Semantic, "The `Generate` node (`explode`, `posexplode`, `inline` and their `_outer` forms)."),
    (geo_types, GeoTypes, POST_MIGRATION_READER_VERSION, Semantic, "The `GEOMETRY` and `GEOGRAPHY` types, with their SRID."),
    (histogram_numeric, HistogramNumeric, POST_MIGRATION_READER_VERSION, Semantic, "The `histogram_numeric` aggregate (Spark's `NumericHistogram`, with its folded number of bins)."),
    (hll_sketch_aggregates, HllSketchAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The DataSketches HLL aggregates (`hll_sketch_agg` with its folded lgConfigK, `hll_union_agg` with its folded allowDifferentLgConfigK): a sketch per group."),
    (indeterminate_collation, IndeterminateCollation, POST_MIGRATION_READER_VERSION, Semantic, "STRING types of the INDETERMINATE collation (`string collate null`: a mixed expression's operands' collations conflict), by its name `null`, alone or inside a nested type."),
    (interval_types, IntervalTypes, POST_MIGRATION_READER_VERSION, Semantic, "The year-month, day-time and calendar interval types, with their fields; calendar literals, and year-month and day-time literals as a cast or function argument. A bare year-month or day-time literal stays baseline."),
    (join_build_preserved, JoinBuildPreserved, POST_MIGRATION_READER_VERSION, Optimization, "Joins that build their preserved side (`RightSemi`, `RightAnti`, `RightOuter`); without it the planner builds the other side (`LeftSemi`, `LeftAnti`, `LeftOuter`)."),
    (join_filter_bound, JoinFilterBound, POST_MIGRATION_READER_VERSION, Optimization, "A hash join's runtime filter collected from a build column as a one-sided range (`JoinFilter.bound_column`, `KeyFilterOptions.side`; design P3-0.3.1 D2): the bound of a comparison conjunct of the residual, `p < b` publishing `p <= max(b)`; without it the planner publishes no such filter, which never changes rows."),
    (join_full_outer, JoinFullOuter, POST_MIGRATION_READER_VERSION, Semantic, "The `FULL_OUTER` hash join: both sides preserved, with no equivalent plan without it."),
    (join_single_match, JoinSingleMatch, POST_MIGRATION_READER_VERSION, Semantic, "The single-match flag of a `LeftOuter` hash join (a scalar subquery: more than one match per probe row is an error)."),
    (key_group_hash, KeyGroupHash, POST_MIGRATION_READER_VERSION, Semantic, "Hash exchanges that route by key-group ranges of the routing hash `H_kg` version 1 (XXH64, seed 0, of the normalized key bytes, folded over several columns)."),
    (lambda_functions, LambdaFunctions, POST_MIGRATION_READER_VERSION, Semantic, "Lambda expressions (`x -> body`) and variables: the bound bodies of higher-order function arguments (`transform`, `filter`, `exists`, `forall`)."),
    (late_fetch, LateFetch, POST_MIGRATION_READER_VERSION, Optimization, "The `LateFetch` node and a scan's row id: a top-N that ranks a narrow scan and reads the other columns of its rows afterwards."),
    (legacy_charset_switches, LegacyCharsetSwitches, POST_MIGRATION_READER_VERSION, Semantic, "The session's `encode`/`decode` switches (`spark.sql.legacy.javaCharsets`, `spark.sql.legacy.codingErrorAction`); without them a worker would refuse the charset or raise on malformed input."),
    (legacy_statistical_aggregate, LegacyStatisticalAggregate, POST_MIGRATION_READER_VERSION, Semantic, "The `legacy_statistical_aggregate` expression environment flag (`spark.sql.legacy.statisticalAggregate`): the sample moment aggregates give NaN instead of NULL where Spark divides by zero."),
    (legacy_time_parser, LegacyTimeParser, POST_MIGRATION_READER_VERSION, Semantic, "The `legacy_time_parser` expression environment flag (`spark.sql.legacy.timeParserPolicy=LEGACY`): the datetime pattern functions use `SimpleDateFormat`, and the JSON, CSV and XML functions the lenient `FastDateFormat`."),
    (limit_offset, LimitOffset, 3, Semantic, "A `Limit` that skips leading rows (a nonzero offset)."),
    (local_relation_source, LocalRelationSource, POST_MIGRATION_READER_VERSION, Semantic, "A read of Arrow data held in the session's relation store (`ReadSource::Local`, Spark's `LocalRelation`), named by content hash; the data is not in the plan. A reader without it has no way to read the relation."),
    (map_key_dedup_policy, MapKeyDedupPolicy, POST_MIGRATION_READER_VERSION, Semantic, "The session's map-key dedup policy (`LAST_WIN`); without it a worker keeps the first value on duplicate map keys."),
    (mode_regr_bit_listagg_aggs, ModeRegrBitListaggAggs, POST_MIGRATION_READER_VERSION, Semantic, "The `mode`, regression (`regr_*`), bitwise (`bit_and`, `bit_or`, `bit_xor`), `listagg`/`string_agg` and `try_sum`/`try_avg` aggregates."),
    (moment_aggregates, MomentAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The moment aggregates (`stddev_samp`, `stddev_pop`, `var_samp`, `var_pop`, `skewness`, `kurtosis`, `corr`, `covar_samp`, `covar_pop`)."),
    (nested_literals, NestedLiterals, POST_MIGRATION_READER_VERSION, Semantic, "An array, map or struct literal value (`Literal::{Array, KeyValueMap, Struct}`). The literal's type is a nested type, so a plan with one lists `nested_types` too."),
    (nested_types, NestedTypes, POST_MIGRATION_READER_VERSION, Semantic, "Array, map and struct types (`DataType.KIND_NESTED`, written as Spark's JSON form of the type)."),
    (no_second_offsets, NoSecondOffsets, POST_MIGRATION_READER_VERSION, Semantic, "The `no_second_offsets` expression environment setting (`spark.sql.files.supportSecondOffsetFormat = false`): the default `timestampFormat` of `to_json`, `to_csv` and `to_xml` drops the seconds of a zone offset."),
    (observe, Observe, POST_MIGRATION_READER_VERSION, Semantic, "The `CollectMetrics` node (`Dataset.observe`): rows pass through while the metrics of a named observation are folded; a worker without it cannot report them."),
    (options_map_literals, OptionsMapLiterals, POST_MIGRATION_READER_VERSION, Semantic, "A `MAP<STRING, STRING>` literal: the options of one call (`from_json`, `from_csv`, `from_xml`). Entries are classified as source options are: a sensitive key or a URL with credentials does not encode (ADR 0038)."),
    (partial_aggregate, PartialAggregate, POST_MIGRATION_READER_VERSION, Optimization, "Partial aggregates below a join, whose states a later aggregate merges (eager aggregation)."),
    (partitioned_writes, PartitionedWrites, POST_MIGRATION_READER_VERSION, Semantic, "A `WritePlan` laid out by partition columns (Spark's directory layout)."),
    (percentile_aggregates, PercentileAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The percentile family and `approx_count_distinct` aggregates (`percentile`, `percentile_approx`, `percentile_disc`, `approx_count_distinct` with their folded parameters)."),
    (python_udf, PythonUdf, POST_MIGRATION_READER_VERSION, Semantic, "Python UDF calls (`PythonUdfFunction`, by name and registration version) and the `PythonEval` node evaluating them; a worker without a UDF runtime refuses the plan."),
    (python_udf_relations, PythonUdfRelations, POST_MIGRATION_READER_VERSION, Semantic, "The Python UDF relation kinds (design P3-T085): the `PythonMap`, `PythonGroupMap`, `PythonCoGroupMap` and `PythonAggregate` nodes, the Python UDTF generator and the eval types beyond the scalar ones (Spark's codes 201-217, 251-254, 300-302); a worker without a runtime for them refuses the plan. A `PythonEval` of the scalar iterator codes 204 and 251 runs from the reader that also has `python_udf_windows` on; an earlier reader with this feature decodes it and fails at run time with `KRISHIV_PYTHON_UDF_EVAL_TYPE_MISMATCH` (no such reader was released)."),
    (python_udf_windows, PythonUdfWindows, POST_MIGRATION_READER_VERSION, Semantic, "A `PythonAggregate` node in window form (design P3-T085 §3.1, Spark's `ArrowWindowPythonExec`): a window aggregate of Python UDFs (codes 203, 253), every input row with one value per call, a bounded frame's rows with their frame bounds; an older reader would run it as a grouped aggregate."),
    (python_udtf, PythonUdtf, POST_MIGRATION_READER_VERSION, Semantic, "The `PythonUdtf` node (a Python table function, eval types 300-302, design P3-T085 §3.1): one UDTF task per partition fed the call's arguments (a `TABLE` argument as one struct column per row, its `PARTITION BY` values among its fields), a `LATERAL` call's input columns carried ahead of the results, and `analyze`'s pickled result; a worker without a UDTF runtime refuses the plan."),
    (range, Range, POST_MIGRATION_READER_VERSION, Semantic, "The `Range` node (`range(start, end, step, numSlices)`)."),
    (rust_udaf, RustUdaf, POST_MIGRATION_READER_VERSION, Semantic, "A call of a session-registered native Rust aggregate UDF (`UdafCall`, by name and version, with its argument and state columns)."),
    (rust_udf, RustUdf, POST_MIGRATION_READER_VERSION, Semantic, "A call of a session-registered native Rust UDF (`UdfFunction`, by name and version)."),
    (sample, Sample, POST_MIGRATION_READER_VERSION, Semantic, "The `Sample` node (`TABLESAMPLE`: Spark's Bernoulli sample of each input partition from `XORShiftRandom(seed + partition)`)."),
    (session_zone_name, SessionZoneName, POST_MIGRATION_READER_VERSION, Semantic, "The configured session time zone name (`spark.sql.session.timeZone`) where it differs from the rendered offset; without it a worker shows the offset in `current_timezone()` and `VV`/`z` patterns."),
    (shared_subplan, SharedSubplan, POST_MIGRATION_READER_VERSION, Semantic, "Subplans run once per query and read by several `SharedRead` nodes (`WITH` bodies computed once)."),
    (sort_merge_join, SortMergeJoin, POST_MIGRATION_READER_VERSION, Semantic, "The `SortMergeJoin` node: a join of its inputs sorted on the keys (a `MERGE` hint or `krishiv.exec.join.strategy`)."),
    (spark_version, SparkVersion, POST_MIGRATION_READER_VERSION, Semantic, "The `spark_version` expression environment setting (the session's Spark profile when it is not 4.2, which every default 0.3 session sets as 4.3; absent is 4.2, what plans written before the feature ran, so a 0.2 writer's plan keeps its meaning on a 0.3 reader): the kernels and function bindings reproduce that release, e.g. a Spark 4.2 `from_json`/`from_csv`/`from_xml` `TIME(p)` read keeps six fraction digits whatever `p`, and Spark 4.3 cuts a string cast to `TIME(p)` to `p` digits."),
    (struct_field, StructField, POST_MIGRATION_READER_VERSION, Semantic, "A struct field access (`GetStructField`: the analyzer-resolved field of a struct)."),
    (table_write, TableWrite, POST_MIGRATION_READER_VERSION, Semantic, "The `TableWrite` node: rows written into a table provider's sink (its writer, or the rewriter of its row-level table), resolved at instantiation; design P3-PR-DML §7."),
    (tail, Tail, POST_MIGRATION_READER_VERSION, Semantic, "The `Tail` node (the last rows of the input in plan order, Spark's `Dataset.tail`)."),
    (theta_sketch_aggregates, ThetaSketchAggregates, POST_MIGRATION_READER_VERSION, Semantic, "The DataSketches Theta aggregates (`theta_sketch_agg` and `theta_union_agg` with their folded lgNomEntries, `theta_intersection_agg`): a sketch per group."),
    (time_precision_nanos, TimePrecisionNanos, POST_MIGRATION_READER_VERSION, Semantic, "A `TIME(p)` type with p in 7..9 (Spark 4.3's nanosecond `TIME`), alone or inside an array, map, struct or user-defined type: a reader that knows only `time_type` (p <= 6) would refuse it as corrupt. Such a type contributes this feature alone, not `time_type`; a `TIME(p <= 6)` elsewhere in the plan still adds `time_type` beside it."),
    (time_type, TimeType, POST_MIGRATION_READER_VERSION, Semantic, "The `TIME` type, with its precision."),
    (udt_type, UdtType, POST_MIGRATION_READER_VERSION, Semantic, "User-defined types (Spark `UserDefinedType`): stored as their SQL type, with the class names in the plan."),
    (union_loop, UnionLoop, POST_MIGRATION_READER_VERSION, Semantic, "The `UnionLoop` and `LoopRef` nodes (recursive common table expressions run as a loop)."),
    (variant_explode, VariantExplode, POST_MIGRATION_READER_VERSION, Semantic, "The `variant_explode` generator (and `variant_explode_outer`): the fields of a variant object or the elements of a variant array."),
    (variant_type, VariantType, POST_MIGRATION_READER_VERSION, Semantic, "The `VARIANT` type (Parquet Variant v1 values, unshredded in memory)."),
    (window, Window, 4, Semantic, "The `Window` node and its functions."),
    (window_counter_diff, WindowCounterDiff, POST_MIGRATION_READER_VERSION, Semantic, "The window function `counter_diff` (Spark 4.3): the increase of a cumulative counter between rows, with an optional start time that marks resets."),
    (window_frame_offsets, WindowFrameOffsets, POST_MIGRATION_READER_VERSION, Semantic, "`ROWS` window frame bounds at an offset from the current row (`n PRECEDING`, `n FOLLOWING`)."),
    (window_group_limit, WindowGroupLimit, POST_MIGRATION_READER_VERSION, Optimization, "The `WindowGroupLimit` node (the rows of each partition within a rank)."),
    (window_offset_functions, WindowOffsetFunctions, POST_MIGRATION_READER_VERSION, Semantic, "The window functions `lag`, `lead` and `nth_value`."),
    (window_range_bounds, WindowRangeBounds, POST_MIGRATION_READER_VERSION, Semantic, "`RANGE` window frame bounds at a value distance from the current row's ordering key (`RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING`)."),
    (window_range_day_time_bounds, WindowRangeDayTimeBounds, POST_MIGRATION_READER_VERSION, Semantic, "`RANGE` window frame bounds that are a day-time interval over a timestamp ordering key (`RANGE BETWEEN INTERVAL '1' DAY PRECEDING AND CURRENT ROW`): whole days on the session zone's clock for a `TIMESTAMP`, elapsed time below a day."),
    (window_range_month_bounds, WindowRangeMonthBounds, POST_MIGRATION_READER_VERSION, Semantic, "`RANGE` window frame bounds that are a year-month interval, moving a date or timestamp ordering key by calendar months (`RANGE BETWEEN CURRENT ROW AND INTERVAL '1' MONTH FOLLOWING`)."),
    (write_bucketed, WriteBucketed, POST_MIGRATION_READER_VERSION, Semantic, "A `WritePlan` with a bucket column: the writer drops it from the files, names each file with its bucket id and expects the rows sorted by partition, bucket and `sortBy` columns; a reader without it would write the bucket id as data."),
    (write_column_types, WriteColumnTypes, POST_MIGRATION_READER_VERSION, Semantic, "A `WritePlan` naming the Spark types of input columns whose Arrow type doesn't say them (an interval's fields): without them a file's footer and text forms would use other fields."),
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
    /// An Arrow field the plan carries as IPC bytes (a field of the plan's
    /// output schema or of an aggregate's result element, or a child field
    /// of an Arrow type): its metadata can name the Spark type.
    ArrowField(&'a arrow_schema::Field),
    /// The same Arrow field, for the `TIME` precision its metadata names: a
    /// field can hold both a collated string and a nanosecond `TIME`.
    ArrowFieldTime(&'a arrow_schema::Field),
    /// A lambda expression or variable: the bound body of a higher-order
    /// function's argument.
    Lambda,
    /// An expression pinned to an ANSI mode (`ResolvedExpr::EvalMode`).
    EvalMode,
    /// A `WritePlan` with a bucket column.
    WriteBucketed,
    /// A `WritePlan` naming the Spark types of input columns.
    WriteColumnTypes,
    /// The type of a nested literal (array, map or struct): a nested type
    /// the literal carries itself, so `nested_types` is listed with
    /// `nested_literals`.
    NestedLiteralType,
    /// The schema a file read declares (`ReadSource::Files.schema`).
    FileSourceSchema,
    /// A scan of a local relation (`ReadSource::Local`).
    LocalRelationSource,
    /// A declared schema's struct type, written as one nested type (so
    /// `nested_types` is listed with `file_source_schema`).
    SchemaType,
    /// A `Generate` node's generator.
    Generator(krishiv_plan::GeneratorKind),
    /// A hash join's probe selection (`HashJoinPlan.select`).
    ProbeSelect,
    /// A hash join's runtime filter collected as a bound from a build
    /// column (`JoinFilterPlan::column`, a one-sided range).
    JoinFilterBound,
    /// A Python UDF's eval type (of a call or a relation node's UDF).
    PythonEvalType(krishiv_plan::PythonEvalType),
}

/// The feature a type needs on its own. The walk reports a type inside an
/// array, map or struct as a construct of its own ([`visit_type`]), so
/// `array<time(3)>` needs `nested_types` (the array) and `time_type` (its
/// element). `DataType` is non-exhaustive, so this cannot list every type: a
/// type added later fails to encode (`encode::spark_type`) until it is
/// mapped there, and classified here.
fn type_feature(t: &DataType) -> Option<Feature> {
    match t {
        DataType::Time { precision } if *precision > MICROS_TIME_PRECISION => {
            Some(Feature::TimePrecisionNanos)
        }
        DataType::Time { .. } => Some(Feature::TimeType),
        DataType::Geometry { .. } | DataType::Geography { .. } => Some(Feature::GeoTypes),
        DataType::YearMonthInterval { .. }
        | DataType::DayTimeInterval { .. }
        | DataType::CalendarInterval => Some(Feature::IntervalTypes),
        DataType::CollatedString(c) if c.is_indeterminate() => {
            Some(Feature::IndeterminateCollation)
        }
        DataType::CollatedString(c) if !c.is_utf8_binary() => Some(Feature::CollatedStrings),
        DataType::Variant => Some(Feature::VariantType),
        DataType::Udt(_) => Some(Feature::UdtType),
        _ => t.is_nested().then_some(Feature::NestedTypes),
    }
}

/// The highest `TIME` precision a reader with only `time_type` decodes.
const MICROS_TIME_PRECISION: u8 = 6;

/// Whether `t` is or holds a `TIME` with a precision past microseconds.
fn nanos_time(t: &DataType) -> bool {
    match t {
        DataType::Time { precision } => *precision > MICROS_TIME_PRECISION,
        DataType::Array(a) => nanos_time(a.element_type()),
        DataType::Map(m) => nanos_time(m.key_type()) || nanos_time(m.value_type()),
        DataType::Struct(s) => s.fields().iter().any(|x| nanos_time(x.data_type())),
        DataType::Udt(u) => nanos_time(u.sql_type()),
        _ => false,
    }
}

/// The Spark type an Arrow field's [`SPARK_TYPE_METADATA_KEY`] tag names,
/// read as `DataType::from_arrow_field` reads it: the engine's JSON form for
/// a type that is or holds a user-defined type, DDL otherwise. A malformed
/// tag is the reader's to refuse, not a feature.
fn tag_type(field: &arrow_schema::Field) -> Option<DataType> {
    let tag = field.metadata().get(SPARK_TYPE_METADATA_KEY)?;
    if tag.starts_with('{') {
        DataType::from_canonical_json(tag).ok()
    } else {
        krishiv_types::parse_data_type(tag).ok()
    }
}

/// Whether an Arrow field's tag names a type that is or holds a nanosecond
/// `TIME` (a user-defined type over one included): the carrier
/// (`Time64(ns)`) is the same at every precision.
fn nanos_time_field(field: &arrow_schema::Field) -> bool {
    tag_type(field).is_some_and(|t| nanos_time(&t))
}

/// Whether `t` is or holds a collated string.
fn collated(t: &DataType) -> bool {
    match t {
        DataType::CollatedString(c) => !c.is_utf8_binary(),
        DataType::Array(a) => collated(a.element_type()),
        DataType::Map(m) => collated(m.key_type()) || collated(m.value_type()),
        DataType::Struct(s) => s.fields().iter().any(|x| collated(x.data_type())),
        DataType::Udt(u) => collated(u.sql_type()),
        _ => false,
    }
}

/// Whether an Arrow field's [`SPARK_TYPE_METADATA_KEY`] tag names a type
/// that is or holds a collated string: the carrier (`Utf8View`) alone says
/// `string`. A malformed tag is the reader's to refuse, not a feature.
fn collated_field(field: &arrow_schema::Field) -> bool {
    tag_type(field).is_some_and(|t| collated(&t))
}

/// Whether `t` is or holds a string of the INDETERMINATE collation.
fn indeterminate(t: &DataType) -> bool {
    match t {
        DataType::CollatedString(c) => c.is_indeterminate(),
        DataType::Array(a) => indeterminate(a.element_type()),
        DataType::Map(m) => indeterminate(m.key_type()) || indeterminate(m.value_type()),
        DataType::Struct(s) => s.fields().iter().any(|x| indeterminate(x.data_type())),
        DataType::Udt(u) => indeterminate(u.sql_type()),
        _ => false,
    }
}

/// Whether an Arrow field's tag names the INDETERMINATE collation, alone or inside an array, map,
/// struct or user-defined type. A DDL tag (`string collate null`) is matched as text, since
/// `parse_data_type` refuses it; a JSON tag is parsed, since its form may be `spark.null` in a
/// field's `__COLLATIONS` (a user-defined type over a struct) with no `collate null` in the text.
fn indeterminate_field(field: &arrow_schema::Field) -> bool {
    field
        .metadata()
        .get(SPARK_TYPE_METADATA_KEY)
        .is_some_and(|tag| tag.contains("collate null"))
        || tag_type(field).is_some_and(|t| indeterminate(&t))
}

/// The constructs of a source a plan reads or writes: a declared file schema, or the stored
/// batches of a local relation. Inferred file reads and table reads need nothing here (the
/// encoder refuses a table read it cannot write).
fn visit_read_source<'a>(source: &'a krishiv_plan::ReadSource, f: &mut impl FnMut(Construct<'a>)) {
    match source {
        krishiv_plan::ReadSource::Files {
            schema: Some(schema),
            ..
        } => {
            f(Construct::FileSourceSchema);
            f(Construct::SchemaType);
            schema
                .fields()
                .iter()
                .for_each(|x| visit_type(x.data_type(), f));
        }
        // The stored batches' Arrow schema travels in the plan, so its fields are classified like
        // any schema the plan writes.
        krishiv_plan::ReadSource::Local(l) => {
            f(Construct::LocalRelationSource);
            l.schema()
                .fields()
                .iter()
                .for_each(|field| visit_arrow_field(field, f));
        }
        _ => {}
    }
}

/// `field`, its type and every field and type inside it as constructs: the
/// tag of a field at any depth can name a collated type.
fn visit_arrow_field<'a>(field: &'a arrow_schema::Field, f: &mut impl FnMut(Construct<'a>)) {
    f(Construct::ArrowField(field));
    f(Construct::ArrowFieldTime(field));
    visit_arrow(field.data_type(), f);
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
        // A user-defined type is stored as its SQL type: what that holds is a construct too.
        DataType::Udt(u) => visit_type(u.sql_type(), f),
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
        | A::Map(x, _) => visit_arrow_field(x, f),
        A::Struct(fields) => fields.iter().for_each(|x| visit_arrow_field(x, f)),
        A::Dictionary(_, v) => visit_arrow(v, f),
        A::RunEndEncoded(_, v) => visit_arrow_field(v, f),
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

/// Whether `e` is a year-month or day-time literal.
fn interval_literal(e: &Expr) -> bool {
    matches!(
        e.node,
        ExprNode::Literal(Literal::YearMonthInterval { .. } | Literal::DayTimeInterval { .. })
    )
}

/// Whether an Arrow type is an interval's carrier.
fn arrow_interval(t: &arrow_schema::DataType) -> bool {
    matches!(
        t,
        arrow_schema::DataType::Interval(_) | arrow_schema::DataType::Duration(_)
    )
}

/// Whether an Arrow type is or contains an array, map or struct.
fn arrow_nested(t: &arrow_schema::DataType) -> bool {
    use arrow_schema::DataType as A;
    // A calendar interval's struct is one value, classified as
    // `calendar_struct` by `feature_of`; a list of them is still nested.
    if krishiv_types::arrow::is_calendar_struct(t) {
        return false;
    }
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
/// The calendar interval's struct carrier is not nested: `arrow_nested`
/// excludes it (it is `calendar_struct`).
pub(crate) fn arrow_type_is_nested(t: &arrow_schema::DataType) -> bool {
    // A variant's carrier is one value, not a struct: it needs `variant_type`, not `nested_types`.
    // Nor is a vector carrier: it needs `fixed_size_list_carrier`.
    !krishiv_types::arrow::is_variant_struct(t)
        && !matches!(t, arrow_schema::DataType::FixedSizeList(..))
        && arrow_nested(t)
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
            PhysNode::LateFetch { .. } => Some(Feature::LateFetch),
            PhysNode::Scan(s) => s.row_id.then_some(Feature::LateFetch),
            PhysNode::HashJoin(j) => j.single_match.then_some(Feature::JoinSingleMatch),
            PhysNode::Window { .. } => Some(Feature::Window),
            PhysNode::Expand { .. } => Some(Feature::Expand),
            PhysNode::Generate { .. } => Some(Feature::Generate),
            PhysNode::Range(_) => Some(Feature::Range),
            PhysNode::SharedRead { .. } => Some(Feature::SharedSubplan),
            PhysNode::ExchangeSource { .. } => Some(Feature::ExchangeSource),
            PhysNode::UnionLoop { .. } | PhysNode::LoopRef { .. } => Some(Feature::UnionLoop),
            PhysNode::SortMergeJoin(j) if j.as_of.is_some() => Some(Feature::AsOfJoin),
            PhysNode::SortMergeJoin(_) => Some(Feature::SortMergeJoin),
            PhysNode::WindowGroupLimit { .. } => Some(Feature::WindowGroupLimit),
            PhysNode::PythonEval { .. } => Some(Feature::PythonUdf),
            PhysNode::Sample { .. } => Some(Feature::Sample),
            PhysNode::Tail { .. } => Some(Feature::Tail),
            PhysNode::CollectMetrics { .. } => Some(Feature::Observe),
            PhysNode::PythonAggregate {
                window: Some(_), ..
            } => Some(Feature::PythonUdfWindows),
            PhysNode::PythonMap { .. }
            | PhysNode::PythonGroupMap { .. }
            | PhysNode::PythonCoGroupMap { .. }
            | PhysNode::PythonAggregate { window: None, .. } => Some(Feature::PythonUdfRelations),
            PhysNode::PythonUdtf { .. } => Some(Feature::PythonUdtf),
            PhysNode::TableWrite { .. } => Some(Feature::TableWrite),
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
            PhysNode::OneRow
            | PhysNode::Filter { .. }
            | PhysNode::Project { .. }
            | PhysNode::ReplicateRows { .. }
            | PhysNode::Sort { .. }
            | PhysNode::TopN { .. }
            | PhysNode::Aggregate { .. }
            | PhysNode::NestedLoopJoin(_)
            | PhysNode::Union { .. } => None,
            PhysNode::Exchange(e) if !e.filters.is_empty() => Some(Feature::ExchangeFilters),
            PhysNode::Exchange(_) => None,
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
            WindowFunction::CounterDiff { .. } => Some(Feature::WindowCounterDiff),
        },
        // A lambda is plan data behind its own feature: a reader without
        // it misreads the higher-order call's argument.
        Construct::Lambda => Some(Feature::LambdaFunctions),
        Construct::EvalMode => Some(Feature::CapturedEvalMode),
        Construct::WriteBucketed => Some(Feature::WriteBucketed),
        Construct::WriteColumnTypes => Some(Feature::WriteColumnTypes),
        Construct::NestedLiteralType | Construct::SchemaType => Some(Feature::NestedTypes),
        Construct::FileSourceSchema => Some(Feature::FileSourceSchema),
        Construct::LocalRelationSource => Some(Feature::LocalRelationSource),
        Construct::ProbeSelect => Some(Feature::FusedProbe),
        Construct::JoinFilterBound => Some(Feature::JoinFilterBound),
        // The scalar kinds came with `python_udf`; every other kind is newer.
        Construct::PythonEvalType(t) => (!t.is_scalar()).then_some(Feature::PythonUdfRelations),
        Construct::Generator(kind) => match kind {
            krishiv_plan::GeneratorKind::VariantExplode => Some(Feature::VariantExplode),
            krishiv_plan::GeneratorKind::PythonUdtf => Some(Feature::PythonUdfRelations),
            krishiv_plan::GeneratorKind::Explode
            | krishiv_plan::GeneratorKind::PosExplode
            | krishiv_plan::GeneratorKind::Inline => None,
        },
        // A type Krishiv does not know cannot be encoded at all. A later
        // feature for a type that can sit inside an array, map or struct
        // (variant, char/varchar, intervals) must look through them: its
        // arm walks the element, key, value and field types, so that
        // `array<variant>` needs it too.
        Construct::Type(t) => type_feature(t),
        Construct::ArrowField(field) => {
            if indeterminate_field(field) {
                Some(Feature::IndeterminateCollation)
            } else {
                collated_field(field).then_some(Feature::CollatedStrings)
            }
        }
        Construct::ArrowFieldTime(field) => {
            nanos_time_field(field).then_some(Feature::TimePrecisionNanos)
        }
        Construct::ArrowType(t) => {
            if arrow_time(t) {
                Some(Feature::TimeType)
            } else if arrow_interval(t) {
                Some(Feature::IntervalTypes)
            } else if krishiv_types::arrow::is_calendar_struct(t) {
                // A calendar interval's carrier is one value, not a struct.
                Some(Feature::CalendarStruct)
            } else if krishiv_types::arrow::is_variant_struct(t) {
                // A variant's carrier is one value, not a struct.
                Some(Feature::VariantType)
            } else if matches!(t, arrow_schema::DataType::FixedSizeList(..)) {
                // A vector carrier is one value, not a nested type: a reader needs the carrier, not `nested_types`.
                Some(Feature::FixedSizeListCarrier)
            } else {
                arrow_nested(t).then_some(Feature::NestedTypes)
            }
        }
        Construct::Expr(e) => match &e.node {
            ExprNode::Resolved(ResolvedExpr::Field { .. }) => Some(Feature::StructField),
            ExprNode::Literal(Literal::CalendarInterval { .. }) => Some(Feature::IntervalTypes),
            ExprNode::Literal(Literal::Map(_)) => Some(Feature::OptionsMapLiterals),
            ExprNode::Literal(
                Literal::Array { .. } | Literal::KeyValueMap { .. } | Literal::Struct { .. },
            ) => Some(Feature::NestedLiterals),
            // The year-month and day-time literal kinds predate the feature,
            // so a bare one stays baseline: whatever reads its column names
            // the interval type. As an argument nothing else may name it, and
            // an older reader would reach a value it has no type for.
            ExprNode::Cast { expr, .. } if interval_literal(expr) => Some(Feature::IntervalTypes),
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.args.iter().any(|a| interval_literal(a)) =>
            {
                Some(Feature::IntervalTypes)
            }
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
            let range =
                |b: FrameBound| matches!(b, FrameBound::RangeInt(_) | FrameBound::RangeFloat(_));
            let months = |b: FrameBound| matches!(b, FrameBound::RangeMonths(_));
            let day_time = |b: FrameBound| matches!(b, FrameBound::RangeDayTime(_));
            if day_time(frame.lower) || day_time(frame.upper) {
                Some(Feature::WindowRangeDayTimeBounds)
            } else if months(frame.lower) || months(frame.upper) {
                Some(Feature::WindowRangeMonthBounds)
            } else if range(frame.lower) || range(frame.upper) {
                Some(Feature::WindowRangeBounds)
            } else {
                (newer(frame.lower) || newer(frame.upper)).then_some(Feature::WindowFrameOffsets)
            }
        }
        Construct::Aggregate(a) => a.mask.is_some().then_some(Feature::AggregateFilter),
        Construct::AggregateFunction(f) => match f {
            AggregateFunction::Count
            | AggregateFunction::Sum
            | AggregateFunction::Avg
            | AggregateFunction::Min
            | AggregateFunction::Max
            | AggregateFunction::First { .. }
            | AggregateFunction::Last { .. } => None,
            AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarSamp
            | AggregateFunction::VarPop
            | AggregateFunction::Skewness
            | AggregateFunction::Kurtosis
            | AggregateFunction::Corr
            | AggregateFunction::CovarSamp
            | AggregateFunction::CovarPop => Some(Feature::MomentAggregates),
            AggregateFunction::AnyValue { .. }
            | AggregateFunction::MaxBy
            | AggregateFunction::MinBy => Some(Feature::AnyValueByAggregates),
            AggregateFunction::MaxByKeyFirst | AggregateFunction::MinByKeyFirst => {
                Some(Feature::CollatedMinMax)
            }
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
            AggregateFunction::Percentile { .. }
            | AggregateFunction::PercentileApprox { .. }
            | AggregateFunction::PercentileDisc { .. }
            | AggregateFunction::ApproxCountDistinct { .. } => Some(Feature::PercentileAggregates),
            AggregateFunction::HistogramNumeric { .. } => Some(Feature::HistogramNumeric),
            AggregateFunction::BitmapConstructAgg
            | AggregateFunction::BitmapOrAgg
            | AggregateFunction::BitmapAndAgg => Some(Feature::BitmapAggregates),
            AggregateFunction::HllSketchAgg { .. } | AggregateFunction::HllUnionAgg { .. } => {
                Some(Feature::HllSketchAggregates)
            }
            AggregateFunction::ThetaSketchAgg { .. }
            | AggregateFunction::ThetaUnionAgg { .. }
            | AggregateFunction::ThetaIntersectionAgg => Some(Feature::ThetaSketchAggregates),
            AggregateFunction::Udf(_) => Some(Feature::RustUdaf),
        },
        // Built-ins and registry functions travel by name; only a
        // session UDF (by name and version) needs a reader with it.
        Construct::Function(id) => match id {
            FunctionId::Udf { .. } => Some(Feature::RustUdf),
            FunctionId::PythonUdf { .. } => Some(Feature::PythonUdf),
            FunctionId::Builtin(b) if b.is_analyzer_internal() => Some(Feature::IntervalTypes),
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
    if let Some(element) = &a.element {
        visit_arrow_field(element, f);
    }
    if let AggregateFunction::Udf(call) = &a.function {
        call.args
            .iter()
            .chain(&call.state)
            .for_each(|x| visit_arrow_field(x, f));
        visit_type(&call.result, f);
        call.arg_types.iter().for_each(|t| visit_type(t, f));
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
                    aggregate_constructs(a, f);
                }
            }
        }
        PhysNode::Aggregate { aggregate, .. } => aggregate
            .aggregates
            .iter()
            .for_each(|a| aggregate_constructs(a, f)),
        PhysNode::HashJoin(j) => {
            f(Construct::Join(j.join_type));
            if j.select.is_some() {
                f(Construct::ProbeSelect);
            }
            if j.runtime_filters
                .iter()
                .any(|r| r.column.is_some() || r.options.side != krishiv_io::RangeSide::Both)
            {
                f(Construct::JoinFilterBound);
            }
        }
        PhysNode::SortMergeJoin(j) => f(Construct::Join(j.join_type)),
        PhysNode::NestedLoopJoin(j) => f(Construct::Join(j.join_type)),
        PhysNode::Generate { kind, .. } => f(Construct::Generator(*kind)),
        // The planner's schema is classified by field, as an exchange
        // source's is.
        PhysNode::Exchange(e) => {
            f(Construct::Partitioning(&e.partitioning));
            if let Some(schema) = &e.schema {
                schema
                    .fields()
                    .iter()
                    .for_each(|field| visit_arrow_field(field, f));
            }
        }
        // The request's schema is classified by field, as an exchange's is.
        // The target's source is written too (`TableRef.source`), so it is classified as a
        // scan's is.
        PhysNode::TableWrite { request, table, .. } => {
            visit_read_source(&table.source, f);
            request
                .input_schema()
                .fields()
                .iter()
                .for_each(|field| visit_arrow_field(field, f));
        }
        // A reduce fragment may name a type only in the schema it reads
        // (a sort by column index above it), so the schema is classified
        // by field, as the plan's output schema is.
        PhysNode::ExchangeSource {
            partitioning,
            schema,
            ..
        } => {
            f(Construct::Partitioning(partitioning));
            schema
                .fields()
                .iter()
                .for_each(|field| visit_arrow_field(field, f));
        }
        PhysNode::Scan(scan) | PhysNode::LateFetch { scan, .. } => {
            visit_read_source(&scan.scan.source, f);
        }
        PhysNode::OneRow
        | PhysNode::SharedRead { .. }
        | PhysNode::UnionLoop { .. }
        | PhysNode::LoopRef { .. }
        | PhysNode::Range(_)
        | PhysNode::Filter { .. }
        | PhysNode::Limit { .. }
        | PhysNode::Tail { .. }
        | PhysNode::ReplicateRows { .. }
        | PhysNode::Sample { .. }
        | PhysNode::Sort { .. }
        | PhysNode::TopN { .. }
        | PhysNode::WindowGroupLimit { .. }
        | PhysNode::PythonEval { .. }
        | PhysNode::CollectMetrics { .. }
        | PhysNode::PythonAggregate { .. }
        | PhysNode::Union { .. } => {}
        PhysNode::PythonMap { udf, .. }
        | PhysNode::PythonGroupMap { udf, .. }
        | PhysNode::PythonCoGroupMap { udf, .. }
        | PhysNode::PythonUdtf { udf, .. } => {
            f(Construct::PythonEvalType(udf.udf.eval_type));
        }
        PhysNode::WriteFiles { write, .. } => {
            if write.as_ref().is_some_and(|w| !w.column_types.is_empty()) {
                f(Construct::WriteColumnTypes);
            }
            if write.as_ref().is_some_and(|w| w.bucket.is_some()) {
                f(Construct::WriteBucketed);
            }
        }
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
        if matches!(&e.node, ExprNode::Resolved(ResolvedExpr::EvalMode { .. })) {
            f(Construct::EvalMode);
        }
        f(Construct::Expr(e));
        nested_literal_types(e, f);
        // A UDF call hides in any expression of the node (`exprs_of`
        // already yields every descendant, so one node check is enough).
        functions_in(e, f);
    }
}

/// A nested literal's type as constructs: the literal itself is a nested type
/// (`nested_types`), and the types inside it are visited as any type is
/// (`array<time(3)>` needs `time_type` too).
fn nested_literal_types<'a>(e: &'a Expr, f: &mut impl FnMut(Construct<'a>)) {
    let ExprNode::Literal(l) = &e.node else {
        return;
    };
    match l {
        Literal::Array { data_type, .. } => {
            f(Construct::NestedLiteralType);
            visit_type(data_type.element_type(), f);
        }
        Literal::KeyValueMap { data_type, .. } => {
            f(Construct::NestedLiteralType);
            visit_type(data_type.key_type(), f);
            visit_type(data_type.value_type(), f);
        }
        Literal::Struct { data_type, .. } => {
            f(Construct::NestedLiteralType);
            data_type
                .fields()
                .iter()
                .for_each(|x| visit_type(x.data_type(), f));
        }
        // Scalars and the options map have no type of their own to visit;
        // `Literal` is non-exhaustive, and a kind added later fails to encode
        // (`encode::literal`) until it is handled there.
        _ => {}
    }
}

/// The Arrow types `node` carries as IPC bytes.
fn arrow_types_of(node: &PhysNode) -> Vec<&arrow_schema::DataType> {
    fn window<'a>(function: &'a WindowFunction, out: &mut Vec<&'a arrow_schema::DataType>) {
        match function {
            WindowFunction::Aggregate(a) => out.push(&a.arg_type),
            WindowFunction::Offset { data_type, .. }
            | WindowFunction::NthValue { data_type, .. }
            | WindowFunction::CounterDiff { data_type, .. } => out.push(data_type),
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
        PhysNode::Exchange(e) => out.extend(e.filters.iter().map(|f| &f.key_type)),
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
            ExprNode::Resolved(
                ResolvedExpr::Field { data_type, .. } | ResolvedExpr::EvalMode { data_type, .. },
            ) => out.push(data_type),
            _ => {}
        }
    }
    if let PhysNode::Scan(scan) | PhysNode::LateFetch { scan, .. } = node {
        out.extend(scan.scan.output.iter().map(|c| &c.data_type));
    }
    if let PhysNode::TableWrite { table, .. } = node {
        out.extend(table.output.iter().map(|c| &c.data_type));
    }
    if let PhysNode::WriteFiles { write: Some(w), .. } = node {
        out.extend(w.column_types.values());
    }
    if let PhysNode::PythonMap { udf, .. }
    | PhysNode::PythonGroupMap { udf, .. }
    | PhysNode::PythonCoGroupMap { udf, .. }
    | PhysNode::PythonUdtf { udf, .. } = node
    {
        out.extend(udf.sides.iter().flat_map(|s| &s.types));
        out.extend(udf.output.iter().map(|c| &c.data_type));
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
        PhysNode::Generate { arg, .. } => roots.push(arg),
        PhysNode::PythonEval { calls, .. } | PhysNode::PythonAggregate { calls, .. } => {
            roots.extend(calls.iter().map(|c| &c.call));
        }
        PhysNode::HashJoin(j) => {
            roots.extend(j.residual.iter().flat_map(|r| &r.conjuncts));
            roots.extend(
                j.select
                    .iter()
                    .flat_map(|s| &s.filter)
                    .map(|f| &f.predicate),
            );
        }
        PhysNode::SortMergeJoin(j) => {
            roots.extend(j.residual.iter().flat_map(|r| &r.conjuncts));
            // The as-of distance is an expression too: it may carry an `EvalMode`.
            roots.extend(
                j.as_of
                    .iter()
                    .flat_map(|a| a.distance.iter())
                    .flat_map(|d| &d.conjuncts),
            );
        }
        PhysNode::NestedLoopJoin(j) => roots.extend(j.residual.iter().flat_map(|r| &r.conjuncts)),
        PhysNode::TableWrite { request, .. } => match request {
            TableWriteRequest::Write(w) => {
                if let WriteMode::OverwriteByFilter(filter) = &w.mode {
                    roots.push(filter);
                }
            }
            TableWriteRequest::RowLevel(r) => roots.extend(r.read_filter.iter()),
        },
        PhysNode::OneRow
        | PhysNode::SharedRead { .. }
        | PhysNode::LateFetch { .. }
        | PhysNode::UnionLoop { .. }
        | PhysNode::LoopRef { .. }
        | PhysNode::Limit { .. }
        | PhysNode::Tail { .. }
        | PhysNode::Sample { .. }
        | PhysNode::Range(_)
        | PhysNode::Sort { .. }
        | PhysNode::TopN { .. }
        | PhysNode::Window { .. }
        | PhysNode::WindowGroupLimit { .. }
        | PhysNode::Aggregate { .. }
        | PhysNode::CollectMetrics { .. }
        | PhysNode::Union { .. }
        | PhysNode::WriteFiles { .. }
        | PhysNode::Exchange(_)
        | PhysNode::ExchangeSource { .. }
        | PhysNode::PythonMap { .. }
        | PhysNode::PythonGroupMap { .. }
        | PhysNode::PythonCoGroupMap { .. }
        | PhysNode::PythonUdtf { .. } => {}
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
        if let FunctionId::PythonUdf { eval_type, .. } = &c.id {
            f(Construct::PythonEvalType(*eval_type));
        }
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
    if plan.env().legacy_java_charsets || plan.env().legacy_coding_error_action {
        out.insert(Feature::LegacyCharsetSwitches);
    }
    if plan.env().legacy_time_parser {
        out.insert(Feature::LegacyTimeParser);
    }
    if plan.env().no_second_offsets {
        out.insert(Feature::NoSecondOffsets);
    }
    if plan.env().binary_output_style.is_some() {
        out.insert(Feature::BinaryOutputStyle);
    }
    if plan.env().spark_version.is_some() {
        out.insert(Feature::SparkVersion);
    }
    if plan.env().zone_name.is_some() {
        out.insert(Feature::SessionZoneName);
    }
    if plan.roots().any(partial_below_join) {
        out.insert(Feature::PartialAggregate);
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
        stack.extend(node.measure());
    }
    for field in plan.schema().fields() {
        visit_arrow_field(field, &mut |c| out.extend(feature_of(c)));
    }
    if has_moment && plan.env().legacy_statistical_aggregate {
        out.insert(Feature::LegacyStatisticalAggregate);
    }
    out
}

/// Whether a partial aggregate runs below a join under `root`.
fn partial_below_join(root: &PhysNode) -> bool {
    let mut stack = vec![(root, false)];
    while let Some((node, below)) = stack.pop() {
        if below
            && matches!(
                node,
                PhysNode::Aggregate { aggregate, .. }
                    if aggregate.output == AggregateOutput::Partial
            )
        {
            return true;
        }
        let below = below
            || matches!(
                node,
                PhysNode::HashJoin(_) | PhysNode::SortMergeJoin(_) | PhysNode::NestedLoopJoin(_)
            );
        stack.extend(node.children().into_iter().map(|c| (c, below)));
    }
    false
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType as A, Field};
    use krishiv_types::DataType;

    use super::*;

    #[test]
    fn the_encoders_nested_flag_agrees_with_the_feature_walk_for_a_variant_carrier() {
        let carrier = DataType::Variant.to_arrow();
        // The walk names `variant_type` only; the encoder must not also demand `nested_types`.
        assert_eq!(
            feature_of(Construct::ArrowType(&carrier)),
            Some(Feature::VariantType)
        );
        assert!(!arrow_type_is_nested(&carrier));
        // A list of variants is nested (and the element is a construct of its own).
        let list = A::List(Arc::new(Field::new("item", carrier, true)));
        assert_eq!(
            feature_of(Construct::ArrowType(&list)),
            Some(Feature::NestedTypes)
        );
        assert!(arrow_type_is_nested(&list));
        // A plain struct stays nested.
        let plain = A::Struct(vec![Field::new("a", A::Int32, true)].into());
        assert!(arrow_type_is_nested(&plain));
        assert_eq!(
            feature_of(Construct::ArrowType(&plain)),
            Some(Feature::NestedTypes)
        );
    }
}
