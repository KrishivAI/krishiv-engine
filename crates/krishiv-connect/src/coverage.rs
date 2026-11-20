// SPDX-License-Identifier: Apache-2.0
//! The coverage table of the Spark Connect translation (P3-T012a, owner decision 7): one row per arm of
//! every `oneof` the translator dispatches on, for the vendored protocol version (4.3). Each row's
//! match is exhaustive and has no wildcard, so a regenerated proto with a new arm does not compile
//! until the arm has a row. `cargo xtask connect-coverage` prints these rows into
//! `docs/reference/connect/coverage.md` and `tests/connect/coverage.json`; the test
//! `coverage_matches_translator` ties the rows to the translator (a row that is not `Done` or
//! `Partial` must answer `INVALID_ONE_OF_FIELD_NOT_SUPPORTED`). The rows are maintained here by hand
//! as arms land; the generated files are never edited.

use crate::proto::v4_3::{catalog, command, data_type, expression, relation};

/// How much of an arm the translator handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Translated, with no field refused that a stock client sends.
    Done,
    /// Translated, with the named fields refused.
    Partial,
    /// Not translated yet: answers `INVALID_ONE_OF_FIELD_NOT_SUPPORTED`.
    Missing,
    /// Never translated (decision 11 or a later phase): the same answer.
    OutOfScope,
}

impl Status {
    /// The word the generated table prints.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::Partial => "partial",
            Self::Missing => "missing",
            Self::OutOfScope => "out of scope",
        }
    }
}

/// One arm of a `oneof` and what the translator does with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coverage {
    /// The `oneof` (`Relation.rel_type`).
    pub oneof: &'static str,
    /// The proto field name.
    pub field: &'static str,
    /// The proto field number.
    pub number: u32,
    /// The status.
    pub status: Status,
    /// The fields refused, for `Partial`.
    pub refused: &'static str,
    /// What the arm lowers to in the IR.
    pub target: &'static str,
    /// The plan task that owns the row (empty for out of scope).
    pub task: &'static str,
    /// The design group (`design/P3-connect-mapping.md` s5), 0 for none.
    pub group: u8,
}

macro_rules! oneof {
    ($fname:ident, $rows:ident, $ty:path, $name:literal; $( $v:ident => ($field:literal, $num:literal, $st:ident, $refused:literal, $target:literal, $task:literal, $group:literal) ),* $(,)?) => {
        #[doc = concat!("The row of an arm of `", $name, "`: an exhaustive match, so a new arm does not compile.")]
        #[must_use]
        pub fn $fname(arm: &$ty) -> Coverage {
            use $ty as Arm;
            match arm {
                $( Arm::$v(_) => Coverage { oneof: $name, field: $field, number: $num, status: Status::$st, refused: $refused, target: $target, task: $task, group: $group }, )*
            }
        }

        #[doc = concat!("Every row of `", $name, "`, in field-number order.")]
        pub static $rows: &[Coverage] = &[
            $( Coverage { oneof: $name, field: $field, number: $num, status: Status::$st, refused: $refused, target: $target, task: $task, group: $group }, )*
        ];
    };
}

/// Every row of every `oneof`.
#[must_use]
pub fn all() -> Vec<Coverage> {
    [
        RELATION_ROWS,
        EXPRESSION_ROWS,
        COMMAND_ROWS,
        CATALOG_ROWS,
        LITERAL_ROWS,
        DATA_KIND_ROWS,
    ]
    .into_iter()
    .flat_map(|rows| {
        let mut rows = rows.to_vec();
        rows.sort_by_key(|r| r.number);
        rows
    })
    .collect()
}

oneof! { relation, RELATION_ROWS, relation::RelType, "Relation.rel_type";
    Read => ("read", 2, Partial, "DataSource.predicates (JDBC) and is_streaming", "Read(ReadSource::{Table,Files}); DataSource.schema → ReadSource::Files schema (DDL or JSON, a non-struct is INVALID_SCHEMA_NON_STRUCT_TYPE); predicates → JDBC partitions (refuse as Spark does for non-JDBC formats: PREDICATES_NOT_SUPPORTED_FOR_DATA_SOURCE)", "P3-T012", 1),
    Project => ("project", 3, Done, "", "Project; missing input → OneRowRelation", "P3-T012", 1),
    Filter => ("filter", 4, Done, "", "Filter", "P2-T091a", 0),
    Join => ("join", 5, Partial, "join_data_type (joinWith, typed)", "Join; join_data_type is Scala joinWith only → unsupported row", "P3-T012", 1),
    SetOp => ("set_op", 6, Done, "", "SetOp / Union (by name, allow missing) / Distinct", "P3-T012", 1),
    Sort => ("sort", 7, Done, "", "Sort", "P2-T091a", 0),
    Limit => ("limit", 8, Done, "", "Limit", "P2-T091a", 0),
    Aggregate => ("aggregate", 9, Partial, "GROUP_TYPE_PIVOT", "Aggregate; pivot → Pivot (exists)", "P3-T012", 2),
    Sql => ("sql", 10, Partial, "DataFrame arguments (with_relations)", "Sql; the with_relations form (#41) for DataFrame arguments", "P3-T012", 4),
    LocalRelation => ("local_relation", 11, Done, "", "Read(ReadSource::Local) from the Arrow IPC data plus the DDL or JSON schema (createDataFrame), built by the host before translation (TranslateConf::local_relations); no data and no schema is SCHEMA_REQUIRED_FOR_LOCAL_RELATION; the inline size is bounded by the gRPC message limit", "P3-T012a", 1),
    Sample => ("sample", 12, Partial, "with_replacement (Poisson) and deterministic_order (randomSplit)", "Sample (Bernoulli, seed or one drawn per plan, Spark's XORShiftRandom order); Poisson needs a new IR field; deterministic_order needs the per-partition sort", "P3-T012", 3),
    Offset => ("offset", 13, Done, "", "Offset", "P2-T091a", 0),
    Deduplicate => ("deduplicate", 14, Partial, "a column subset and within_watermark", "Distinct (all columns); a subset is Aggregate with first per non-key column and needs the input columns (analyzer node); within_watermark → P4", "P3-T012", 3),
    Range => ("range", 15, Done, "", "Range", "P2-T091a", 0),
    SubqueryAlias => ("subquery_alias", 16, Done, "", "SubqueryAlias; the qualifier and the alias are the alias parts joined by ALIAS_PART_SEP", "P3-T012", 1),
    Repartition => ("repartition", 17, Partial, "the partition count (spark_partition_id, getNumPartitions)", "Hint(REPARTITION n | COALESCE n): rows pass through unchanged; the count is not applied to the plan; each call adds the warning \"Hint REPARTITION(n) has no effect in a single-node plan.\" to the plan's warnings (Spark logs nothing; whether a client sees them is not checked)", "P3-T012", 3),
    ToDf => ("to_df", 18, Done, "", "ColumnNames(ToDf): the analyzer renames by position (count mismatch is the derived-table alias error)", "P3-T012", 4),
    WithColumnsRenamed => ("with_columns_renamed", 19, Partial, "rename_columns_map when its result depends on entry order (wire order is lost on decode)", "ColumnNames(Rename): the pairs in turn, case rule of the session, unknown names ignored", "P3-T012", 4),
    ShowString => ("show_string", 20, Missing, "", "server-side: execute, then format as Spark's Dataset.showString (truncate, vertical); returns a one-row, one-column relation", "P3-T012", 4),
    Drop => ("drop", 21, Partial, "columns that are qualified or tied to a DataFrame", "ColumnNames(Drop): by name; a plain one-part column expression resolves over the input (ambiguous is an error); unknown names ignored", "P3-T012", 4),
    Tail => ("tail", 22, Done, "", "Tail (the last n rows in plan order; not Sort + Limit); a negative n is INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE from the analyzer", "P3-T012", 3),
    WithColumns => ("with_columns", 23, Partial, "Alias.metadata", "WithColumns; metadata on the column field", "P3-T012", 4),
    Hint => ("hint", 24, Done, "", "Hint; as Spark 4.2: parameters are literals or lists of them, a partitioning hint is dropped with a warning (3.5 reads parameters differently)", "P3-T012", 3),
    Unpivot => ("unpivot", 25, Partial, "absent values, values that are not columns, and a column that is also an id or listed twice", "Project(ids, values) then Unpivot with nulls kept; values absent → every non-id column, known only after analysis", "P3-T012", 2),
    ToSchema => ("to_schema", 26, Missing, "", "lowering: Project of by-name Cast and reorder (Spark's Dataset.to)", "P3-T012", 4),
    RepartitionByExpression => ("repartition_by_expression", 27, Partial, "num_partitions, no expressions (round robin), repartitionByRange", "Repartition (hash); a partition count, a round-robin repartition and SortOrder expressions need IR fields", "P3-T012", 3),
    MapPartitions => ("map_partitions", 28, Partial, "is_barrier = true, profile_id, Scala/Java functions (56K00), eval types other than 205/207 (FUNCTION_EVAL_TYPE_NOT_SUPPORTED), additional_includes", "MapPartitions (mapInPandas 205, mapInArrow 207) over the PythonUdfRef the server's UdfHost names (Usage on System(udf), as for inline UDFs); runs on the session's Python workers", "P3-T014", 6),
    CollectMetrics => ("collect_metrics", 29, Done, "", "CollectMetrics (observe); the server returns ObservedMetrics after the action", "P3-T084", 6),
    Parse => ("parse", 30, Missing, "", "lowering: Project of from_json / from_csv over a string relation", "P3-T012", 4),
    GroupMap => ("group_map", 31, Partial, "sorting_expressions, initial_input and the other stateful fields, transformWithState eval types 211-214, Scala/Java functions (56K00), additional_includes", "GroupMap (applyInPandas 201/216, applyInArrow 209/215) over the PythonUdfRef the server's UdfHost names (Usage on System(udf), as for inline UDFs); nondeterministic UDF refused at analysis; runs on the session's Python workers", "P3-T014", 6),
    CoGroupMap => ("co_group_map", 32, Partial, "input_sorting_expressions, other_sorting_expressions, unequal key counts, Scala/Java functions (56K00), additional_includes", "CoGroupMap (cogroup applyInPandas 206, applyInArrow 210) over the PythonUdfRef the server's UdfHost names (Usage on System(udf), as for inline UDFs); nondeterministic UDF refused at analysis; runs on the session's Python workers", "P3-T014", 6),
    WithWatermark => ("with_watermark", 33, OutOfScope, "", "P4", "", 0),
    ApplyInPandasWithState => ("apply_in_pandas_with_state", 34, OutOfScope, "", "P4", "", 0),
    HtmlString => ("html_string", 35, Missing, "", "as show_string, HTML form", "P3-T012", 4),
    CachedLocalRelation => ("cached_local_relation", 36, Done, "", "Read(ReadSource::Local) through the same host request as local_relation; the host reads the session's own cache/<hash> artifact (CANNOT_FIND_CACHED_LOCAL_RELATION without one) and builds it as an inline local_relation", "P3-T012", 7),
    CachedRemoteRelation => ("cached_remote_relation", 37, Done, "", "a read of the reserved table `__krishiv_cached_remote.<id>`; the server resolves the id in its session (DATAFRAME_NOT_FOUND when absent) and splices the frame's plan in (`Session::from_plan_over`); a lazy checkpoint's rows are taken by the first analysis that reads it, AnalyzePlan included (Spark runs no job on analysis: a recorded difference); tests/cache.rs", "P3-T083", 7),
    CommonInlineUserDefinedTableFunction => ("common_inline_user_defined_table_function", 38, Missing, "", "TableFunction with a Python UDTF (T014)", "P3-T014", 6),
    AsOfJoin => ("as_of_join", 39, Missing, "", "new AsOfJoin, shared with the ASOF JOIN SQL syntax", "P3-T012", 8),
    CommonInlineUserDefinedDataSource => ("common_inline_user_defined_data_source", 40, Missing, "", "the Python data source scan (T040)", "P3-T040", 6),
    WithRelations => ("with_relations", 41, Missing, "", "Sql with DataFrame references → WithCte/CteRef; errors INVALID_WITH_RELATION_REFERENCE, INVALID_SQL_WITH_REFERENCES; a cached_remote_relation inside a subquery or with_relations is recorded by the translator but not spliced by Session::from_plan_over (it walks plan nodes, not expression subqueries): T012 must splice it", "P3-T012", 4),
    Transpose => ("transpose", 42, Missing, "", "new Transpose (output schema depends on data; Spark resolves it in the analyzer)", "P3-T012", 3),
    UnresolvedTableValuedFunction => ("unresolved_table_valued_function", 43, Missing, "", "TableFunction", "P3-T012", 4),
    LateralJoin => ("lateral_join", 44, Missing, "", "Join with a lateral right side; the IR has no lateral flag, so this is new", "P3-T012", 8),
    ChunkedCachedLocalRelation => ("chunked_cached_local_relation", 45, Missing, "", "as #36, from chunked artifacts (4.1+); CHUNKED_CACHED_LOCAL_RELATION_WITHOUT_DATA", "P3-T083", 7),
    RelationChanges => ("relation_changes", 46, Missing, "", "CDC read (DataFrameReader.changes()): Delta change data feed through table_changes; the Iceberg changelog and is_streaming come later", "P3-T021b", 8),
    NearestByJoin => ("nearest_by_join", 47, Missing, "", "top-k per left row by a ranking expression: new NearestByJoin, planned on T077's top-k (num_results 1–100,000)", "P3-T077", 8),
    Zip => ("zip", 48, Done, "", "PlanNode::Zip; the analyzer merges the two projection chains over their shared base (Spark's ResolveZip), ZIP_PLANS_NOT_MERGEABLE otherwise", "P3-T012", 3),
    FillNa => ("fill_na", 90, Done, "", "ColumnNames(FillNa): the analyzer projects coalesce / nanvl of the cast value in the columns the value's type reaches; NA_FILL_VALUES_EMPTY, NA_FILL_VALUES_LENGTH_MISMATCH; several values are Spark's fillMap (every named column, the last value of a repeated name); a BYTE, SHORT, INT, FLOAT or DECIMAL value is accepted where Spark's one-value fill takes only Long, Double, String, Boolean", "P3-T012", 5),
    DropNa => ("drop_na", 91, Done, "", "ColumnNames(DropNa): the analyzer filters on the count of values that are neither NULL nor NaN (min_non_nulls, else every column of the subset); a nested field in the subset (`s.a`) is UNRESOLVED_COLUMN where Spark resolves it", "P3-T012", 5),
    Replace => ("replace", 92, Done, "", "ColumnNames(Replace): the analyzer projects CASE WHEN col = old THEN new per column the first old value's kind reaches (a Spark 4.0 run); a NULL or mixed-type old value is refused; the last pair of a repeated old value wins; named columns are resolved even with no pairs", "P3-T012", 5),
    Summary => ("summary", 100, Missing, "", "lowering as Spark's StatFunctions.summary: Aggregate then a stack to rows", "P3-T012", 5),
    Crosstab => ("crosstab", 101, Missing, "", "lowering: Aggregate count + Pivot", "P3-T012", 5),
    Describe => ("describe", 102, Missing, "", "as summary with the fixed statistic list", "P3-T012", 5),
    Cov => ("cov", 103, Partial, "a non-numeric column is answered when its values cast to double (Spark refuses by type; a value that is not a number is an error here, never 0.0)", "Aggregate of coalesce(covar_samp over NULL-as-0.0 doubles, 0.0), as StatFunctions.calculateCov (a Spark 4.0 run)", "P3-T012", 5),
    Corr => ("corr", 104, Partial, "method other than pearson; a non-numeric column is answered when its values cast to double (Spark refuses by type; a value that is not a number is an error here, never 0.0)", "Aggregate of coalesce(corr over NULL-as-0.0 doubles, NaN); recorded difference: zero variance is NaN (Spark 4.0 ANSI: DIVIDE_BY_ZERO)", "P3-T012", 5),
    ApproxQuantile => ("approx_quantile", 105, Missing, "", "Aggregate approx_percentile with relative error", "P3-T012", 5),
    FreqItems => ("freq_items", 106, Missing, "", "new aggregate (Spark's frequent-items sketch, support)", "P3-T012", 5),
    SampleBy => ("sample_by", 107, Done, "", "Filter(rand(seed) < CASE WHEN col = stratum THEN fraction ... ELSE 0.0 END), the draw made for every row as Spark does; a repeated stratum keeps its last fraction; strata compare with SQL = (numeric coercion), whether Spark's boxed lookup differs for an Int stratum on a Long column is not verified against a Spark run", "P3-T012", 5),
    Catalog => ("catalog", 200, Done, "", "a call of the session's `spark.catalog` run by the host while the plan is prepared (`LocalRequest::Catalog`), its rows the local relation the plan reads; `execute/catalog_call.rs`, tests/catalog.rs", "P3-T082", 9),
    MlRelation => ("ml_relation", 300, OutOfScope, "", "machine learning is out of scope", "", 0),
    Extension => ("extension", 998, Partial, "every krishiv.v1 message but Echo, and any other package (KRISHIV_UNSUPPORTED_FEATURE naming the capability)", "the krishiv.v1 registry (`extension::resolve`); Echo is a one-row Project over OneRowRelation", "P3-T015", 10),
    Unknown => ("unknown", 999, OutOfScope, "", "Spark falls to case other and answers 56K00 with UNKNOWN(999)", "", 0),
}

oneof! { expression, EXPRESSION_ROWS, expression::ExprType, "Expression.expr_type";
    Literal => ("literal", 1, Partial, "Time", "Literal/TypedLiteral; nested literals are Literal::{Array,KeyValueMap,Struct}; nanosecond timestamps are refused until the types exist", "P3-T012", 1),
    UnresolvedAttribute => ("unresolved_attribute", 2, Partial, "is_metadata_column", "Column; metadata columns (_metadata) → the scan's metadata columns", "P3-T012", 1),
    UnresolvedFunction => ("unresolved_function", 3, Partial, "is_user_defined_function (qualified names) and is_internal", "Function with qualifier; internal functions map to Spark's internal names where Krishiv has them, otherwise 56K00-style refusal naming the function", "P3-T012", 1),
    ExpressionString => ("expression_string", 4, Done, "", "krishiv_sql::parse_and_lower_named_expression, as Session::select_expr and Session::expr (parser options from TranslateConf)", "P3-T012a", 1),
    UnresolvedStar => ("unresolved_star", 5, Done, "", "Star", "P2-T091a", 0),
    Alias => ("alias", 6, Partial, "metadata and several names", "Alias; several names → MultiAlias (exists)", "P3-T012", 1),
    Cast => ("cast", 7, Done, "", "Cast", "P2-T091a", 0),
    UnresolvedRegex => ("unresolved_regex", 8, Missing, "", "Star { regex } (the field exists)", "P3-T012", 2),
    SortOrder => ("sort_order", 9, Missing, "", "SortOrder, only inside Sort and Window", "P3-T012", 2),
    LambdaFunction => ("lambda_function", 10, Missing, "", "Lambda", "P3-T012", 2),
    Window => ("window", 11, Missing, "", "ExprNode::Window (frames: LOWER_BOUND_REQUIRED_IN_WINDOW_FRAME, UPPER_BOUND_…, WINDOW_FUNCTION_REQUIRED)", "P3-T012", 2),
    UnresolvedExtractValue => ("unresolved_extract_value", 12, Missing, "", "Field", "P3-T012", 2),
    UpdateFields => ("update_fields", 13, Missing, "", "UpdateFields", "P3-T012", 2),
    UnresolvedNamedLambdaVariable => ("unresolved_named_lambda_variable", 14, Missing, "", "lambda-scoped Column (UNRESOLVED_NAMED_LAMBDA_VARIABLE_REQUIRES_NAME_PART)", "P3-T012", 2),
    CommonInlineUserDefinedFunction => ("common_inline_user_defined_function", 15, Partial, "scalar_scala_udf and java_udf (56K00: no JVM code; T081), Python eval types other than 100/101/200/250, additional_includes", "Function over the session's encoded name of an unregistered Python UDF (PythonUdf), its payload registered with the session's UDF workers; Usage on System(udf) before any worker starts", "P3-T014", 6),
    CallFunction => ("call_function", 16, Done, "", "Function (as #3, by name)", "P3-T012", 1),
    NamedArgumentExpression => ("named_argument_expression", 17, Partial, "outside a function call's arguments", "FunctionCall.arg_names", "P3-T012", 1),
    MergeAction => ("merge_action", 19, Missing, "", "MERGE actions for merge_into_table_command (T023, T021, T027)", "P3-T023", 11),
    TypedAggregateExpression => ("typed_aggregate_expression", 20, OutOfScope, "", "Scala typed Dataset aggregation; unsupported row", "", 0),
    SubqueryExpression => ("subquery_expression", 21, Missing, "", "Subquery (scalar, EXISTS, IN) referencing with_relations", "P3-T012", 4),
    DirectShufflePartitionId => ("direct_shuffle_partition_id", 22, Missing, "", "partition id passed through for repartition_by_expression (expressions.proto §DirectShufflePartitionID); Repartition by explicit id", "P3-T012", 3),
    Extension => ("extension", 999, Partial, "every krishiv.v1 message but Echo, and any other package (KRISHIV_UNSUPPORTED_FEATURE naming the capability)", "the krishiv.v1 registry (`extension::resolve`); Echo is a string Literal", "P3-T015", 10),
}

oneof! { command, COMMAND_ROWS, command::CommandType, "Command.command_type";
    RegisterFunction => ("register_function", 1, Partial, "scalar_scala_udf and java_udf (56K00: no JVM code; T081), Python eval types other than 100/101/200/250, additional_includes", "Session::register_python_udf plus the payload on the session's UDF workers (translate_command; run by the server)", "P3-T014", 6),
    WriteOperation => ("write_operation", 2, Partial, "clustering_columns, with_schema_evolution", "krishiv-session's DataFrameWriter (write_to_with: save; write_table_with: saveAsTable, insertInto), the facade's path; save() without a path takes the path option (_LEGACY_ERROR_TEMP_2050 without one); run by the server after the Write class check", "P3-T012", 11),
    CreateDataframeView => ("create_dataframe_view", 3, Missing, "", "create_or_replace_temp_view (exists), global temp views", "P3-T012", 4),
    WriteOperationV2 => ("write_operation_v2", 4, Partial, "with_schema_evolution, partition transforms (partitioning_columns other than a column), overwrite_condition (overwrite is refused by the session)", "krishiv-session's DataFrameWriterV2 (write_to_table_with), the facade's path; run by the server after the Write class check", "P3-T012", 11),
    SqlCommand => ("sql_command", 5, Partial, "input of WithRelations form (group 4; the plain SQL input is supported); named and positional arguments (the analyzer refuses SQL parameters)", "spark.sql() sends this. Run commands at once; for queries return sql_command_result.relation (Spark returns a SQL or LocalRelation relation the client re-sends). Translated here as the SQL relation of a query; the server runs commands (execute.rs)", "P3-T012", 1),
    WriteStreamOperationStart => ("write_stream_operation_start", 6, OutOfScope, "", "P4", "", 0),
    StreamingQueryCommand => ("streaming_query_command", 7, OutOfScope, "", "P4", "", 0),
    GetResourcesCommand => ("get_resources_command", 8, Missing, "", "an empty resource map (single node, no executors)", "P3-T012", 4),
    StreamingQueryManagerCommand => ("streaming_query_manager_command", 9, OutOfScope, "", "P4", "", 0),
    RegisterTableFunction => ("register_table_function", 10, Missing, "", "Python UDTF (T014)", "P3-T014", 6),
    StreamingQueryListenerBusCommand => ("streaming_query_listener_bus_command", 11, OutOfScope, "", "P4", "", 0),
    RegisterDataSource => ("register_data_source", 12, Missing, "", "T040", "P3-T040", 6),
    CreateResourceProfileCommand => ("create_resource_profile_command", 13, Missing, "", "refusal: Krishiv has no stage-level resource profiles (an unsupported row; no IR)", "P3-T012", 4),
    CheckpointCommand => ("checkpoint_command", 14, Done, "", "`DataFrame::checkpoint` / `local_checkpoint` (translate_command reads the flags; the server runs it and answers `checkpoint_command_result` with the id it holds the frame under; the operator floor of ADR 0046 §3b applies; the query carries no operation id or tags: an interrupt by operation id ends the command, one by tag does not reach it); tests/cache.rs, tests/auth.rs", "P3-T083", 7),
    RemoveCachedRemoteRelationCommand => ("remove_cached_remote_relation_command", 15, Done, "", "drops the session's frame under the id (an unknown id is not an error); tests/cache.rs", "P3-T083", 7),
    MergeIntoTableCommand => ("merge_into_table_command", 16, Missing, "", "DataFrame.mergeInto → MERGE (T023 for Parquet/Hive refusals, T021 Delta, T027 Iceberg)", "P3-T023", 11),
    MlCommand => ("ml_command", 17, OutOfScope, "", "machine learning is out of scope; a client may send it for cache cleanup when a session ends", "", 0),
    ExecuteExternalCommand => ("execute_external_command", 18, Missing, "", "refusal: runs a command on an external JDBC source (executeExternal); no JDBC connector in P3", "P3-T012", 4),
    PipelineCommand => ("pipeline_command", 19, OutOfScope, "", "pipelines are out of scope until P4", "", 0),
    Extension => ("extension", 999, Partial, "every krishiv.v1 message: none is registered for a command yet (KRISHIV_UNSUPPORTED_FEATURE naming the capability)", "the krishiv.v1 registry (`extension::resolve`); the server runs a command extension when one is registered", "P3-T015", 10),
}

oneof! { catalog, CATALOG_ROWS, catalog::CatType, "Catalog.cat_type";
    CurrentDatabase => ("current_database", 1, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    SetCurrentDatabase => ("set_current_database", 2, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ListDatabases => ("list_databases", 3, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ListTables => ("list_tables", 4, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ListFunctions => ("list_functions", 5, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ListColumns => ("list_columns", 6, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    GetDatabase => ("get_database", 7, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    GetTable => ("get_table", 8, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    GetFunction => ("get_function", 9, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    DatabaseExists => ("database_exists", 10, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    TableExists => ("table_exists", 11, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    FunctionExists => ("function_exists", 12, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    CreateExternalTable => ("create_external_table", 13, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    CreateTable => ("create_table", 14, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    DropTempView => ("drop_temp_view", 15, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    DropGlobalTempView => ("drop_global_temp_view", 16, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    RecoverPartitions => ("recover_partitions", 17, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    IsCached => ("is_cached", 18, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    CacheTable => ("cache_table", 19, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    UncacheTable => ("uncache_table", 20, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ClearCache => ("clear_cache", 21, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    RefreshTable => ("refresh_table", 22, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    RefreshByPath => ("refresh_by_path", 23, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    CurrentCatalog => ("current_catalog", 24, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    SetCurrentCatalog => ("set_current_catalog", 25, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ListCatalogs => ("list_catalogs", 26, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    DropTable => ("drop_table", 27, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    DropView => ("drop_view", 28, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    CreateDatabase => ("create_database", 29, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    DropDatabase => ("drop_database", 30, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ListPartitions => ("list_partitions", 31, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    ListViews => ("list_views", 32, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    GetTableProperties => ("get_table_properties", 33, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    GetCreateTableString => ("get_create_table_string", 34, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    TruncateTable => ("truncate_table", 35, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
    AnalyzeTable => ("analyze_table", 36, Done, "", "the session catalog\'s call of that name (`CatalogCall`); Spark\'s overload by `db_name`; the result rows of Spark\'s `Dataset`", "P3-T082", 9),
}

oneof! { literal, LITERAL_ROWS, expression::literal::LiteralType, "Expression.Literal.literal_type";
    Null => ("null", 1, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Binary => ("binary", 2, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Boolean => ("boolean", 3, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Byte => ("byte", 4, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Short => ("short", 5, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Integer => ("integer", 6, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Long => ("long", 7, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Float => ("float", 10, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Double => ("double", 11, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Decimal => ("decimal", 12, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    String => ("string", 13, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Date => ("date", 16, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Timestamp => ("timestamp", 17, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    TimestampNtz => ("timestamp_ntz", 18, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    CalendarInterval => ("calendar_interval", 19, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    YearMonthInterval => ("year_month_interval", 20, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    DayTimeInterval => ("day_time_interval", 21, Done, "", "Literal / TypedLiteral", "P2-T091a", 1),
    Array => ("array", 22, Done, "", "Literal::{Array,KeyValueMap,Struct}; the type is the root data_type, else the message\'s own type fields", "P3-T012a", 1),
    Map => ("map", 23, Done, "", "Literal::{Array,KeyValueMap,Struct}; the type is the root data_type, else the message\'s own type fields", "P3-T012a", 1),
    Struct => ("struct", 24, Done, "", "Literal::{Array,KeyValueMap,Struct}; the type is the root data_type, else the message\'s own type fields", "P3-T012a", 1),
    SpecializedArray => ("specialized_array", 25, Done, "", "Literal::{Array,KeyValueMap,Struct}; the type is the root data_type, else the message\'s own type fields", "P3-T012a", 1),
    Time => ("time", 26, Missing, "", "TIME literals wait for an IR value", "P3-T012", 1),
    TimestampNtzNanos => ("timestamp_ntz_nanos", 29, Missing, "", "no IR type for nanosecond timestamps", "P3-T012", 1),
    TimestampLtzNanos => ("timestamp_ltz_nanos", 30, Missing, "", "no IR type for nanosecond timestamps", "P3-T012", 1),
}

oneof! { data_kind, DATA_KIND_ROWS, data_type::Kind, "DataType.kind";
    Null => ("null", 1, Missing, "", "no IR type", "P3-T012", 1),
    Binary => ("binary", 2, Done, "", "DataType", "P2-T091a", 1),
    Boolean => ("boolean", 3, Done, "", "DataType", "P2-T091a", 1),
    Byte => ("byte", 4, Done, "", "DataType", "P2-T091a", 1),
    Short => ("short", 5, Done, "", "DataType", "P2-T091a", 1),
    Integer => ("integer", 6, Done, "", "DataType", "P2-T091a", 1),
    Long => ("long", 7, Done, "", "DataType", "P2-T091a", 1),
    Float => ("float", 8, Done, "", "DataType", "P2-T091a", 1),
    Double => ("double", 9, Done, "", "DataType", "P2-T091a", 1),
    Decimal => ("decimal", 10, Done, "", "DataType", "P2-T091a", 1),
    String => ("string", 11, Done, "", "DataType", "P2-T091a", 1),
    Char => ("char", 12, Missing, "", "no IR type", "P3-T012", 1),
    VarChar => ("var_char", 13, Missing, "", "no IR type", "P3-T012", 1),
    Date => ("date", 14, Done, "", "DataType", "P2-T091a", 1),
    Timestamp => ("timestamp", 15, Done, "", "DataType", "P2-T091a", 1),
    TimestampNtz => ("timestamp_ntz", 16, Done, "", "DataType", "P2-T091a", 1),
    CalendarInterval => ("calendar_interval", 17, Done, "", "DataType", "P2-T091a", 1),
    YearMonthInterval => ("year_month_interval", 18, Done, "", "DataType", "P2-T091a", 1),
    DayTimeInterval => ("day_time_interval", 19, Done, "", "DataType", "P2-T091a", 1),
    Array => ("array", 20, Done, "", "DataType", "P2-T091a", 1),
    Struct => ("struct", 21, Done, "", "DataType", "P2-T091a", 1),
    Map => ("map", 22, Done, "", "DataType", "P2-T091a", 1),
    Variant => ("variant", 25, Done, "", "DataType", "P2-T091a", 1),
    Udt => ("udt", 23, Partial, "a JVM UDT with no sql_type (Spark builds it from the JVM class alone; Krishiv cannot load JVM classes, so `sql_type` is required and the answer is INVALID_ONE_OF_FIELD_NOT_SET)", "DataType::Udt", "P3-T079", 1),
    Geometry => ("geometry", 26, Done, "", "DataType", "P2-T091a", 1),
    Geography => ("geography", 27, Done, "", "DataType", "P2-T091a", 1),
    Unparsed => ("unparsed", 24, Missing, "", "a type string parsed by the SQL type parser", "P3-T012", 1),
    Time => ("time", 28, Done, "", "DataType", "P2-T091a", 1),
    TimestampNtzNanos => ("timestamp_ntz_nanos", 29, Missing, "", "no IR type for nanosecond timestamps", "P3-T012", 1),
    TimestampLtzNanos => ("timestamp_ltz_nanos", 30, Missing, "", "no IR type for nanosecond timestamps", "P3-T012", 1),
}

fn quote(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The rows as the JSON `cargo xtask connect-coverage` writes to `tests/connect/coverage.json`.
#[must_use]
pub fn to_json() -> String {
    use std::fmt::Write as _;
    let rows = all();
    let mut out = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        let _ = write!(
            out,
            "  {{\"oneof\": {}, \"field\": {}, \"number\": {}, \"status\": {}, \"refused\": {}, \"target\": {}, \"task\": {}, \"group\": {}}}",
            quote(r.oneof),
            quote(r.field),
            r.number,
            quote(r.status.word()),
            quote(r.refused),
            quote(r.target),
            quote(r.task),
            r.group
        );
        out.push_str(if i + 1 < rows.len() { ",\n" } else { "\n" });
    }
    out.push_str("]\n");
    out
}

#[cfg(test)]
mod tests {
    use super::quote;

    #[test]
    fn json_strings_escape_quotes_backslashes_and_control_characters() {
        assert_eq!(quote("a\"b\\c\nd\u{1}"), "\"a\\\"b\\\\c\\nd\\u0001\"");
    }
}
