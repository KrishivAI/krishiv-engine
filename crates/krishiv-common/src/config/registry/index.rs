// SPDX-License-Identifier: Apache-2.0
//! The sorted lookup index of the configuration registry (name to entry), kept sorted by a test.

use super::*;

/// Every key and Spark alias with its entry, sorted by (length, name) for
/// binary search: lookup is O(log n) with no initialisation or allocation,
/// and most probes compare only lengths, never the shared `krishiv.` prefix.
/// A test checks the order and that it holds exactly the keys and aliases
/// of [`REGISTRY`]; add new entries here in order.
pub static INDEX: &[(&str, &ConfigEntry)] = &[
    ("krishiv.exec.seed", &EXEC_SEED),
    ("krishiv.spill.dir", &SPILL_DIR),
    ("krishiv.compute.id", &COMPUTE_ID),
    ("krishiv.ui.enabled", &UI_ENABLED),
    ("krishiv.udf.sandbox", &UDF_SANDBOX),
    ("krishiv.connect.port", &CONNECT_PORT),
    ("krishiv.history.path", &HISTORY_PATH),
    ("krishiv.memory.limit", &MEMORY_LIMIT),
    ("spark.checkpoint.dir", &CHECKPOINT_DIR),
    ("krishiv.io.cache.path", &IO_CACHE_PATH),
    ("krishiv.io.maxRetries", &IO_MAX_RETRIES),
    ("spark.network.timeout", &NETWORK_TIMEOUT),
    ("spark.sql.defaultPath", &SQL_DEFAULT_PATH),
    ("krishiv.cache.maxBytes", &CACHE_MAX_BYTES),
    ("krishiv.checkpoint.dir", &CHECKPOINT_DIR),
    ("krishiv.exec.batchSize", &EXEC_BATCH_SIZE),
    ("krishiv.spill.maxBytes", &SPILL_MAX_BYTES),
    ("spark.sql.ansi.enabled", &ANSI_ENABLED),
    ("spark.sql.path.enabled", &SQL_PATH_ENABLED),
    ("krishiv.access.provider", &ACCESS_PROVIDER),
    ("krishiv.advisor.enabled", &ADVISOR_ENABLED),
    ("krishiv.agent.gateBytes", &AGENT_GATE_BYTES),
    ("krishiv.connect.address", &CONNECT_ADDRESS),
    ("krishiv.io.egress.proxy", &IO_EGRESS_PROXY),
    ("krishiv.metrics.enabled", &METRICS_ENABLED),
    ("krishiv.network.timeout", &NETWORK_TIMEOUT),
    ("krishiv.resultCache.ttl", &RESULT_CACHE_TTL),
    ("krishiv.sql.defaultPath", &SQL_DEFAULT_PATH),
    ("krishiv.ui.allowedHosts", &UI_ALLOWED_HOSTS),
    ("spark.sql.timestampType", &SQL_TIMESTAMP_TYPE),
    ("krishiv.advisor.maxHints", &ADVISOR_MAX_HINTS),
    ("krishiv.delta.logThreads", &DELTA_LOG_THREADS),
    ("krishiv.events.file.path", &EVENTS_FILE_PATH),
    ("krishiv.experimental.ivm", &EXPERIMENTAL_IVM),
    ("krishiv.history.literals", &HISTORY_LITERALS),
    ("krishiv.history.maxBytes", &HISTORY_MAX_BYTES),
    ("krishiv.io.cache.enabled", &IO_CACHE_ENABLED),
    ("krishiv.io.coalesceBytes", &IO_COALESCE_BYTES),
    ("krishiv.io.prefetchBytes", &IO_PREFETCH_BYTES),
    ("krishiv.io.strictOptions", &IO_STRICT_OPTIONS),
    ("krishiv.spill.zstd.level", &SPILL_ZSTD_LEVEL),
    ("krishiv.sql.ansi.enabled", &ANSI_ENABLED),
    ("krishiv.sql.path.enabled", &SQL_PATH_ENABLED),
    ("krishiv.ui.embed.linkTtl", &UI_EMBED_LINK_TTL),
    ("krishiv.flightSql.enabled", &FLIGHT_SQL_ENABLED),
    ("krishiv.io.requestTimeout", &IO_REQUEST_TIMEOUT),
    ("krishiv.io.text.inferTime", &IO_TEXT_INFER_TIME),
    ("krishiv.io.text.timeNanos", &IO_TEXT_TIME_NANOS),
    ("krishiv.spill.compression", &SPILL_COMPRESSION),
    ("krishiv.sql.timestampType", &SQL_TIMESTAMP_TYPE),
    ("krishiv.ui.allowedOrigins", &UI_ALLOWED_ORIGINS),
    ("krishiv.catalog.maxColumns", &CATALOG_MAX_COLUMNS),
    ("krishiv.deterministicOrder", &DETERMINISTIC_ORDER),
    ("krishiv.exec.agg.rowLayout", &EXEC_AGG_ROW_LAYOUT),
    ("krishiv.exec.coalesceInput", &EXEC_COALESCE_INPUT),
    ("krishiv.exec.join.strategy", &EXEC_JOIN_STRATEGY),
    ("krishiv.io.blockingThreads", &IO_BLOCKING_THREADS),
    ("krishiv.io.cache.diskBytes", &IO_CACHE_DISK_BYTES),
    ("krishiv.io.footerReadBytes", &IO_FOOTER_READ_BYTES),
    ("krishiv.io.maxRequestBytes", &IO_MAX_REQUEST_BYTES),
    ("krishiv.optimizer.joinCost", &JOIN_COST),
    ("krishiv.python.udf.threads", &PYTHON_UDF_THREADS),
    ("spark.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("krishiv.advisor.skew.factor", &ADVISOR_SKEW_FACTOR),
    ("krishiv.connect.maxSessions", &CONNECT_MAX_SESSIONS),
    ("krishiv.events.textMaxBytes", &EVENTS_TEXT_MAX_BYTES),
    ("krishiv.exec.agg.directKeys", &EXEC_AGG_DIRECT_KEYS),
    ("krishiv.exec.join.buildSide", &EXEC_JOIN_BUILD_SIDE),
    ("krishiv.exec.join.radixBits", &EXEC_JOIN_RADIX_BITS),
    ("krishiv.exec.join.rowLayout", &EXEC_JOIN_ROW_LAYOUT),
    ("krishiv.flightSql.handleTtl", &FLIGHT_SQL_HANDLE_TTL),
    ("krishiv.io.cache.blockBytes", &IO_CACHE_BLOCK_BYTES),
    ("krishiv.query.maxQueuedTime", &QUERY_MAX_QUEUED_TIME),
    ("krishiv.resultCache.enabled", &RESULT_CACHE_ENABLED),
    ("krishiv.server.allowedHosts", &SERVER_ALLOWED_HOSTS),
    ("krishiv.session.memoryLimit", &SESSION_MEMORY_LIMIT),
    ("spark.sql.binaryOutputStyle", &BINARY_OUTPUT_STYLE),
    ("spark.sql.mapKeyDedupPolicy", &MAP_KEY_DEDUP_POLICY),
    ("krishiv.advisor.skew.minRows", &ADVISOR_SKEW_MIN_ROWS),
    ("krishiv.advisor.skew.minTime", &ADVISOR_SKEW_MIN_TIME),
    ("krishiv.connect.maxPlanDepth", &CONNECT_MAX_PLAN_DEPTH),
    ("krishiv.connect.sparkProfile", &CONNECT_SPARK_PROFILE),
    ("krishiv.events.file.maxBytes", &EVENTS_FILE_MAX_BYTES),
    ("krishiv.events.file.maxFiles", &EVENTS_FILE_MAX_FILES),
    ("krishiv.events.queueCapacity", &EVENTS_QUEUE_CAPACITY),
    ("krishiv.io.cache.memoryBytes", &IO_CACHE_MEMORY_BYTES),
    ("krishiv.memory.residentGuard", &MEMORY_RESIDENT_GUARD),
    ("krishiv.resultCache.maxBytes", &RESULT_CACHE_MAX_BYTES),
    ("krishiv.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("krishiv.write.idempotencyKey", &WRITE_IDEMPOTENCY_KEY),
    ("spark.sql.shuffle.partitions", &SHUFFLE_PARTITIONS_HINT),
    ("krishiv.access.external.token", &ACCESS_EXTERNAL_TOKEN),
    ("krishiv.exec.agg.outputFilter", &EXEC_AGG_OUTPUT_FILTER),
    ("krishiv.exec.join.directIndex", &EXEC_JOIN_DIRECT_INDEX),
    ("krishiv.exec.sort.hashBuckets", &EXEC_SORT_HASH_BUCKETS),
    ("krishiv.io.coalesceWasteBytes", &IO_COALESCE_WASTE_BYTES),
    ("krishiv.io.local.allowedRoots", &IO_LOCAL_ALLOWED_ROOTS),
    ("krishiv.io.metadataCacheBytes", &IO_METADATA_CACHE_BYTES),
    ("krishiv.io.multipartPartBytes", &IO_MULTIPART_PART_BYTES),
    ("krishiv.optimizer.joinDpLimit", &JOIN_DP_LIMIT),
    ("krishiv.runtimeFilter.enabled", &RUNTIME_FILTER_ENABLED),
    ("krishiv.server.allowCoreDumps", &SERVER_ALLOW_CORE_DUMPS),
    ("krishiv.server.metrics.public", &SERVER_METRICS_PUBLIC),
    ("krishiv.spill.dropBehindBytes", &SPILL_DROP_BEHIND_BYTES),
    ("krishiv.sql.binaryOutputStyle", &BINARY_OUTPUT_STYLE),
    ("krishiv.sql.mapKeyDedupPolicy", &MAP_KEY_DEDUP_POLICY),
    ("spark.sql.legacy.javaCharsets", &LEGACY_JAVA_CHARSETS),
    ("krishiv.advisor.spill.minBytes", &ADVISOR_SPILL_MIN_BYTES),
    ("krishiv.catalog.requestTimeout", &CATALOG_REQUEST_TIMEOUT),
    ("krishiv.connect.keepAlive.time", &CONNECT_KEEPALIVE_TIME),
    ("krishiv.connect.maxConnections", &CONNECT_MAX_CONNECTIONS),
    ("krishiv.exec.lateFetch.maxRows", &LATE_FETCH_MAX_ROWS),
    ("krishiv.flightSql.maxBatchRows", &FLIGHT_SQL_MAX_BATCH_ROWS),
    ("krishiv.io.egress.allowedCidrs", &IO_EGRESS_ALLOWED_CIDRS),
    ("krishiv.io.lateMaterialization", &IO_LATE_MATERIALIZATION),
    ("krishiv.optimizer.discoverKeys", &DISCOVER_KEYS),
    ("krishiv.optimizer.groupKeyCode", &GROUP_KEY_CODE),
    ("krishiv.python.worker.maxTotal", &PYTHON_WORKER_MAX_TOTAL),
    ("krishiv.query.maxExecutionTime", &QUERY_MAX_EXECUTION_TIME),
    ("krishiv.server.shutdownTimeout", &SERVER_SHUTDOWN_TIMEOUT),
    ("spark.sql.cteRecursionRowLimit", &CTE_RECURSION_ROW_LIMIT),
    ("krishiv.access.external.timeout", &ACCESS_EXTERNAL_TIMEOUT),
    (
        "krishiv.delta.maxCommitAttempts",
        &DELTA_MAX_COMMIT_ATTEMPTS,
    ),
    ("krishiv.exec.dataflowScheduling", &EXEC_DATAFLOW_SCHEDULING),
    (
        "krishiv.io.coalesceWasteDivisor",
        &IO_COALESCE_WASTE_DIVISOR,
    ),
    (
        "krishiv.memory.spillWaitTimeout",
        &MEMORY_SPILL_WAIT_TIMEOUT,
    ),
    ("krishiv.sql.legacy.javaCharsets", &LEGACY_JAVA_CHARSETS),
    ("krishiv.system.queries.capacity", &SYSTEM_QUERIES_CAPACITY),
    ("spark.connect.grpc.binding.port", &CONNECT_PORT),
    ("spark.sql.files.maxPartitionNum", &FILES_MAX_PARTITION_NUM),
    ("spark.sql.files.minPartitionNum", &FILES_MIN_PARTITION_NUM),
    ("spark.sql.files.openCostInBytes", &FILES_OPEN_COST_IN_BYTES),
    ("spark.sql.join.asofJoin.enabled", &SQL_ASOF_JOIN_ENABLED),
    ("spark.sql.storeAssignmentPolicy", &STORE_ASSIGNMENT_POLICY),
    (
        "krishiv.access.external.endpoint",
        &ACCESS_EXTERNAL_ENDPOINT,
    ),
    (
        "krishiv.advisor.spill.minPercent",
        &ADVISOR_SPILL_MIN_PERCENT,
    ),
    (
        "krishiv.catalog.maxMetadataBytes",
        &CATALOG_MAX_METADATA_BYTES,
    ),
    (
        "krishiv.catalog.vendedDefaultTtl",
        &CATALOG_VENDED_DEFAULT_TTL,
    ),
    (
        "krishiv.connect.handshakeTimeout",
        &CONNECT_HANDSHAKE_TIMEOUT,
    ),
    (
        "krishiv.connect.maxResponseBytes",
        &CONNECT_MAX_RESPONSE_BYTES,
    ),
    (
        "krishiv.exec.debug.planRoundTrip",
        &EXEC_DEBUG_PLAN_ROUND_TRIP,
    ),
    (
        "krishiv.exec.join.idleSpillAfter",
        &EXEC_JOIN_IDLE_SPILL_AFTER,
    ),
    (
        "krishiv.flightSql.provisionalTtl",
        &FLIGHT_SQL_PROVISIONAL_TTL,
    ),
    (
        "krishiv.io.maxConcurrentRequests",
        &IO_MAX_CONCURRENT_REQUESTS,
    ),
    (
        "krishiv.memory.releaseAtQueryEnd",
        &MEMORY_RELEASE_AT_QUERY_END,
    ),
    ("krishiv.optimizer.joinEnumerator", &JOIN_ENUMERATOR),
    ("krishiv.optimizer.joinPruneFirst", &JOIN_PRUNE_FIRST),
    (
        "krishiv.server.readiness.timeout",
        &SERVER_READINESS_TIMEOUT,
    ),
    ("krishiv.sql.cteRecursionRowLimit", &CTE_RECURSION_ROW_LIMIT),
    ("krishiv.sql.files.commitProtocol", &FILES_COMMIT_PROTOCOL),
    (
        "krishiv.ui.embed.hideAttribution",
        &UI_EMBED_HIDE_ATTRIBUTION,
    ),
    (
        "krishiv.write.commitRetryTimeout",
        &WRITE_COMMIT_RETRY_TIMEOUT,
    ),
    ("spark.sql.avro.compression.codec", &AVRO_COMPRESSION_CODEC),
    (
        "spark.sql.cteRecursionLevelLimit",
        &CTE_RECURSION_LEVEL_LIMIT,
    ),
    ("spark.sql.repl.eagerEval.enabled", &REPL_EAGER_EVAL_ENABLED),
    (
        "krishiv.advisor.memoryWait.factor",
        &ADVISOR_MEMORY_WAIT_FACTOR,
    ),
    (
        "krishiv.cache.defaultStorageLevel",
        &CACHE_DEFAULT_STORAGE_LEVEL,
    ),
    (
        "krishiv.catalog.vended.maxEntries",
        &CATALOG_VENDED_MAX_ENTRIES,
    ),
    (
        "krishiv.connect.keepAlive.enabled",
        &CONNECT_KEEPALIVE_ENABLED,
    ),
    (
        "krishiv.connect.keepAlive.timeout",
        &CONNECT_KEEPALIVE_TIMEOUT,
    ),
    (
        "krishiv.exec.join.buildProjection",
        &EXEC_JOIN_BUILD_PROJECTION,
    ),
    (
        "krishiv.exec.join.fuseProbeSelect",
        &EXEC_JOIN_FUSE_PROBE_SELECT,
    ),
    (
        "krishiv.exec.join.smj.bufferBytes",
        &EXEC_JOIN_SMJ_BUFFER_BYTES,
    ),
    ("krishiv.exec.targetPartitionBytes", &TARGET_PARTITION_BYTES),
    (
        "krishiv.flightSql.maxSchemaTables",
        &FLIGHT_SQL_MAX_SCHEMA_TABLES,
    ),
    (
        "krishiv.flightSql.metadataTimeout",
        &FLIGHT_SQL_METADATA_TIMEOUT,
    ),
    (
        "krishiv.memory.reservationQuantum",
        &MEMORY_RESERVATION_QUANTUM,
    ),
    ("krishiv.optimizer.latePayloadJoin", &LATE_PAYLOAD_JOIN),
    (
        "krishiv.python.worker.executables",
        &PYTHON_WORKER_EXECUTABLES,
    ),
    (
        "krishiv.python.worker.idleTimeout",
        &PYTHON_WORKER_IDLE_TIMEOUT,
    ),
    (
        "krishiv.python.worker.memoryLimit",
        &PYTHON_WORKER_MEMORY_LIMIT,
    ),
    (
        "krishiv.server.readiness.interval",
        &SERVER_READINESS_INTERVAL,
    ),
    (
        "krishiv.server.readiness.required",
        &SERVER_READINESS_REQUIRED,
    ),
    (
        "krishiv.sql.files.maxPartitionNum",
        &FILES_MAX_PARTITION_NUM,
    ),
    (
        "krishiv.sql.files.minPartitionNum",
        &FILES_MIN_PARTITION_NUM,
    ),
    (
        "krishiv.sql.files.openCostInBytes",
        &FILES_OPEN_COST_IN_BYTES,
    ),
    ("krishiv.sql.join.asofJoin.enabled", &SQL_ASOF_JOIN_ENABLED),
    (
        "krishiv.sql.storeAssignmentPolicy",
        &STORE_ASSIGNMENT_POLICY,
    ),
    (
        "krishiv.system.queryText.literals",
        &SYSTEM_QUERY_TEXT_LITERALS,
    ),
    (
        "krishiv.system.queryText.maxBytes",
        &SYSTEM_QUERY_TEXT_MAX_BYTES,
    ),
    ("spark.connect.grpc.keepAlive.time", &CONNECT_KEEPALIVE_TIME),
    (
        "spark.sql.files.maxPartitionBytes",
        &FILES_MAX_PARTITION_BYTES,
    ),
    (
        "spark.sql.legacy.timeParserPolicy",
        &LEGACY_TIME_PARSER_POLICY,
    ),
    (
        "spark.sql.repl.eagerEval.truncate",
        &REPL_EAGER_EVAL_TRUNCATE,
    ),
    (
        "spark.sql.unionOutputPartitioning",
        &UNION_OUTPUT_PARTITIONING,
    ),
    (
        "krishiv.access.external.securables",
        &ACCESS_EXTERNAL_SECURABLES,
    ),
    (
        "krishiv.advisor.memoryWait.minTime",
        &ADVISOR_MEMORY_WAIT_MIN_TIME,
    ),
    (
        "krishiv.connect.artifacts.maxBytes",
        &CONNECT_ARTIFACTS_MAX_BYTES,
    ),
    (
        "krishiv.connect.execute.aheadBytes",
        &CONNECT_EXECUTE_AHEAD_BYTES,
    ),
    (
        "krishiv.flightSql.fetchIdleTimeout",
        &FLIGHT_SQL_FETCH_IDLE_TIMEOUT,
    ),
    (
        "krishiv.iceberg.write.maxOpenFiles",
        &ICEBERG_WRITE_MAX_OPEN_FILES,
    ),
    (
        "krishiv.io.credentials.refreshSkew",
        &IO_CREDENTIALS_REFRESH_SKEW,
    ),
    (
        "krishiv.io.egress.privateAddresses",
        &IO_EGRESS_PRIVATE_ADDRESSES,
    ),
    (
        "krishiv.memory.abortReleaseTimeout",
        &MEMORY_ABORT_RELEASE_TIMEOUT,
    ),
    (
        "krishiv.memory.spillScratchReserve",
        &MEMORY_SPILL_SCRATCH_RESERVE,
    ),
    ("krishiv.optimizer.derivedKeysAbove", &DERIVED_KEYS_ABOVE),
    ("krishiv.optimizer.eagerAggregation", &EAGER_AGGREGATION),
    ("krishiv.optimizer.ndvSampling.rows", &NDV_SAMPLING_ROWS),
    ("krishiv.optimizer.rollupFromParent", &ROLLUP_FROM_PARENT),
    ("krishiv.optimizer.subqueryAsWindow", &SUBQUERY_AS_WINDOW),
    (
        "krishiv.python.worker.maxPerTenant",
        &PYTHON_WORKER_MAX_PER_TENANT,
    ),
    (
        "krishiv.runtimeFilter.equatedSites",
        &RUNTIME_FILTER_EQUATED_SITES,
    ),
    (
        "krishiv.sql.avro.compression.codec",
        &AVRO_COMPRESSION_CODEC,
    ),
    (
        "krishiv.sql.cteRecursionLevelLimit",
        &CTE_RECURSION_LEVEL_LIMIT,
    ),
    (
        "krishiv.sql.repl.eagerEval.enabled",
        &REPL_EAGER_EVAL_ENABLED,
    ),
    (
        "krishiv.sql.shuffle.partitionsHint",
        &SHUFFLE_PARTITIONS_HINT,
    ),
    ("spark.connect.grpc.binding.address", &CONNECT_ADDRESS),
    (
        "spark.sql.defaultCacheStorageLevel",
        &CACHE_DEFAULT_STORAGE_LEVEL,
    ),
    (
        "spark.sql.legacy.codingErrorAction",
        &LEGACY_CODING_ERROR_ACTION,
    ),
    (
        "krishiv.connect.session.idleTimeout",
        &CONNECT_SESSION_IDLE_TIMEOUT,
    ),
    (
        "krishiv.flightSql.maxStatementBytes",
        &FLIGHT_SQL_MAX_STATEMENT_BYTES,
    ),
    (
        "krishiv.iceberg.manifestCache.bytes",
        &ICEBERG_MANIFEST_CACHE_BYTES,
    ),
    ("krishiv.optimizer.semiAntiJoinOrder", &SEMI_ANTI_JOIN_ORDER),
    (
        "krishiv.python.worker.maxPerSession",
        &PYTHON_WORKER_MAX_PER_SESSION,
    ),
    (
        "krishiv.runtimeFilter.bloom.atScale",
        &RUNTIME_FILTER_BLOOM_AT_SCALE,
    ),
    (
        "krishiv.sql.files.maxPartitionBytes",
        &FILES_MAX_PARTITION_BYTES,
    ),
    (
        "krishiv.sql.legacy.timeParserPolicy",
        &LEGACY_TIME_PARSER_POLICY,
    ),
    (
        "krishiv.sql.repl.eagerEval.truncate",
        &REPL_EAGER_EVAL_TRUNCATE,
    ),
    (
        "krishiv.sql.unionOutputPartitioning",
        &UNION_OUTPUT_PARTITIONING,
    ),
    (
        "spark.sql.repl.eagerEval.maxNumRows",
        &REPL_EAGER_EVAL_MAX_NUM_ROWS,
    ),
    (
        "krishiv.advisor.missingStats.minRows",
        &ADVISOR_MISSING_STATS_MIN_ROWS,
    ),
    (
        "krishiv.delta.snapshotCache.maxBytes",
        &DELTA_SNAPSHOT_CACHE_MAX_BYTES,
    ),
    (
        "krishiv.exec.scan.instanceProjection",
        &EXEC_SCAN_INSTANCE_PROJECTION,
    ),
    (
        "krishiv.io.parquet.pageIndex.enabled",
        &IO_PARQUET_PAGE_INDEX_ENABLED,
    ),
    (
        "krishiv.optimizer.preAggregateExpand",
        &PRE_AGGREGATE_EXPAND,
    ),
    (
        "krishiv.optimizer.shareFilteredScans",
        &SHARE_FILTERED_SCANS,
    ),
    (
        "krishiv.runtimeFilter.bloom.maxBytes",
        &RUNTIME_FILTER_BLOOM_MAX_BYTES,
    ),
    (
        "krishiv.runtimeFilter.bloom.maxItems",
        &RUNTIME_FILTER_BLOOM_MAX_ITEMS,
    ),
    (
        "krishiv.runtimeFilter.residualBounds",
        &RUNTIME_FILTER_RESIDUAL_BOUNDS,
    ),
    (
        "krishiv.runtimeFilter.smallTableWait",
        &RUNTIME_FILTER_SMALL_TABLE_WAIT,
    ),
    (
        "krishiv.sql.legacy.codingErrorAction",
        &LEGACY_CODING_ERROR_ACTION,
    ),
    (
        "krishiv.workload.resourceGroups.file",
        &WORKLOAD_RESOURCE_GROUPS_FILE,
    ),
    (
        "spark.connect.grpc.keepAlive.enabled",
        &CONNECT_KEEPALIVE_ENABLED,
    ),
    (
        "spark.connect.grpc.keepAlive.timeout",
        &CONNECT_KEEPALIVE_TIMEOUT,
    ),
    (
        "spark.sql.autoBroadcastJoinThreshold",
        &AUTO_BROADCAST_JOIN_THRESHOLD,
    ),
    (
        "spark.sql.leafNodeDefaultParallelism",
        &LEAF_NODE_DEFAULT_PARALLELISM,
    ),
    (
        "spark.sql.legacy.ctePrecedencePolicy",
        &CTE_PRECEDENCE_POLICY,
    ),
    (
        "krishiv.connect.maxInboundMessageSize",
        &CONNECT_MAX_INBOUND_MESSAGE_SIZE,
    ),
    (
        "krishiv.exec.agg.distinctCountAtMerge",
        &EXEC_AGG_DISTINCT_COUNT_AT_MERGE,
    ),
    (
        "krishiv.iceberg.commit.maxIntervening",
        &ICEBERG_COMMIT_MAX_INTERVENING,
    ),
    (
        "krishiv.optimizer.ndvSampling.enabled",
        &NDV_SAMPLING_ENABLED,
    ),
    (
        "krishiv.optimizer.ndvSampling.timeout",
        &NDV_SAMPLING_TIMEOUT,
    ),
    (
        "krishiv.sql.repl.eagerEval.maxNumRows",
        &REPL_EAGER_EVAL_MAX_NUM_ROWS,
    ),
    (
        "spark.sql.legacy.statisticalAggregate",
        &LEGACY_STATISTICAL_AGGREGATE,
    ),
    (
        "spark.sql.timeTravel.atSyntax.enabled",
        &TIME_TRAVEL_AT_SYNTAX_ENABLED,
    ),
    (
        "krishiv.advisor.missingStats.minQError",
        &ADVISOR_MISSING_STATS_MIN_Q_ERROR,
    ),
    (
        "krishiv.exec.join.sharedBroadcastBytes",
        &EXEC_JOIN_SHARED_BROADCAST_BYTES,
    ),
    (
        "krishiv.experimental.groupKeyCodeForce",
        &EXPERIMENTAL_GROUP_KEY_CODE_FORCE,
    ),
    (
        "krishiv.flightSql.maxHandlesPerSession",
        &FLIGHT_SQL_MAX_HANDLES_PER_SESSION,
    ),
    (
        "krishiv.io.objectStore.allowedPrefixes",
        &IO_OBJECT_STORE_ALLOWED_PREFIXES,
    ),
    (
        "krishiv.io.parquet.bloomFilter.enabled",
        &IO_PARQUET_BLOOM_FILTER_ENABLED,
    ),
    (
        "krishiv.optimizer.discoverKeys.timeout",
        &DISCOVER_KEYS_TIMEOUT,
    ),
    (
        "krishiv.runtimeFilter.adaptive.minRows",
        &RUNTIME_FILTER_ADAPTIVE_MIN_ROWS,
    ),
    (
        "krishiv.runtimeFilter.bloom.memoryTier",
        &RUNTIME_FILTER_BLOOM_MEMORY_TIER,
    ),
    (
        "krishiv.runtimeFilter.inList.maxValues",
        &RUNTIME_FILTER_IN_LIST_MAX_VALUES,
    ),
    (
        "krishiv.sql.autoBroadcastJoinThreshold",
        &AUTO_BROADCAST_JOIN_THRESHOLD,
    ),
    (
        "krishiv.sql.leafNodeDefaultParallelism",
        &LEAF_NODE_DEFAULT_PARALLELISM,
    ),
    (
        "krishiv.sql.legacy.ctePrecedencePolicy",
        &CTE_PRECEDENCE_POLICY,
    ),
    (
        "spark.sql.legacy.viewSchemaBindingMode",
        &LEGACY_VIEW_SCHEMA_BINDING_MODE,
    ),
    (
        "spark.sql.sources.bucketing.maxBuckets",
        &BUCKETING_MAX_BUCKETS,
    ),
    (
        "krishiv.connect.artifacts.maxTotalBytes",
        &CONNECT_ARTIFACTS_MAX_TOTAL_BYTES,
    ),
    (
        "krishiv.connect.execute.detachedTimeout",
        &CONNECT_EXECUTE_DETACHED_TIMEOUT,
    ),
    (
        "krishiv.connect.execute.retryBufferSize",
        &CONNECT_EXECUTE_RETRY_BUFFER_SIZE,
    ),
    (
        "krishiv.connect.maxOperationsPerSession",
        &CONNECT_MAX_OPERATIONS_PER_SESSION,
    ),
    (
        "krishiv.connect.maxSessionsPerPrincipal",
        &CONNECT_MAX_SESSIONS_PER_PRINCIPAL,
    ),
    (
        "krishiv.connect.pythonFunctions.catalog",
        &CONNECT_PYTHON_FUNCTIONS_CATALOG,
    ),
    (
        "krishiv.flightSql.maxCommandResultBytes",
        &FLIGHT_SQL_MAX_COMMAND_RESULT_BYTES,
    ),
    (
        "krishiv.optimizer.payloadAboveAggregate",
        &PAYLOAD_ABOVE_AGGREGATE,
    ),
    (
        "krishiv.optimizer.preservedSideSemiJoin",
        &PRESERVED_SIDE_SEMI_JOIN,
    ),
    (
        "krishiv.optimizer.shareLiteralInstances",
        &SHARE_LITERAL_INSTANCES,
    ),
    (
        "krishiv.optimizer.shareRepeatedSubtrees",
        &SHARE_REPEATED_SUBTREES,
    ),
    (
        "krishiv.python.worker.sandbox.allowFork",
        &PYTHON_WORKER_ALLOW_FORK,
    ),
    (
        "krishiv.python.worker.tenantMemoryLimit",
        &PYTHON_WORKER_TENANT_MEMORY_LIMIT,
    ),
    (
        "krishiv.sql.legacy.statisticalAggregate",
        &LEGACY_STATISTICAL_AGGREGATE,
    ),
    (
        "krishiv.sql.optimizer.shareCteInstances",
        &SHARE_CTE_INSTANCES,
    ),
    (
        "krishiv.sql.timeTravel.atSyntax.enabled",
        &TIME_TRAVEL_AT_SYNTAX_ENABLED,
    ),
    (
        "spark.sql.avro.datetimeRebaseModeInRead",
        &AVRO_DATETIME_REBASE_MODE_IN_READ,
    ),
    (
        "spark.sql.collation.schemaLevel.enabled",
        &SCHEMA_LEVEL_COLLATIONS_ENABLED,
    ),
    (
        "spark.sql.legacy.viewSchemaCompensation",
        &LEGACY_VIEW_SCHEMA_COMPENSATION,
    ),
    (
        "krishiv.connect.http2.maxHeaderListBytes",
        &CONNECT_HTTP2_MAX_HEADER_LIST_BYTES,
    ),
    (
        "krishiv.connect.session.closedTombstones",
        &CONNECT_SESSION_TOMBSTONES,
    ),
    (
        "krishiv.flightSql.maxHandlesPerPrincipal",
        &FLIGHT_SQL_MAX_HANDLES_PER_PRINCIPAL,
    ),
    (
        "krishiv.iceberg.write.statistics.enabled",
        &ICEBERG_WRITE_STATISTICS_ENABLED,
    ),
    (
        "krishiv.runtimeFilter.exactSet.maxValues",
        &RUNTIME_FILTER_EXACT_SET_MAX_VALUES,
    ),
    (
        "krishiv.sql.legacy.viewSchemaBindingMode",
        &LEGACY_VIEW_SCHEMA_BINDING_MODE,
    ),
    (
        "krishiv.sql.sources.bucketing.maxBuckets",
        &BUCKETING_MAX_BUCKETS,
    ),
    (
        "spark.connect.grpc.maxInboundMessageSize",
        &CONNECT_MAX_INBOUND_MESSAGE_SIZE,
    ),
    (
        "spark.databricks.delta.maxCommitAttempts",
        &DELTA_MAX_COMMIT_ATTEMPTS,
    ),
    (
        "spark.sql.join.sortMergeAsOfJoin.enabled",
        &SQL_SORT_MERGE_ASOF_JOIN_ENABLED,
    ),
    (
        "spark.sql.session.localRelationSizeLimit",
        &LOCAL_RELATION_SIZE_LIMIT,
    ),
    (
        "spark.sql.sources.partitionOverwriteMode",
        &PARTITION_OVERWRITE_MODE,
    ),
    (
        "krishiv.experimental.unalignedCheckpoints",
        &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
    ),
    (
        "krishiv.server.readiness.failureThreshold",
        &SERVER_READINESS_FAILURE_THRESHOLD,
    ),
    (
        "krishiv.sql.avro.datetimeRebaseModeInRead",
        &AVRO_DATETIME_REBASE_MODE_IN_READ,
    ),
    (
        "krishiv.sql.collation.schemaLevel.enabled",
        &SCHEMA_LEVEL_COLLATIONS_ENABLED,
    ),
    (
        "krishiv.sql.legacy.viewSchemaCompensation",
        &LEGACY_VIEW_SCHEMA_COMPENSATION,
    ),
    (
        "krishiv.advisor.runtimeFilter.minProbeRows",
        &ADVISOR_RUNTIME_FILTER_MIN_PROBE_ROWS,
    ),
    (
        "krishiv.connect.artifacts.maxArtifactBytes",
        &CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES,
    ),
    (
        "krishiv.connect.http2.maxConcurrentStreams",
        &CONNECT_HTTP2_MAX_CONCURRENT_STREAMS,
    ),
    (
        "krishiv.sql.join.sortMergeAsOfJoin.enabled",
        &SQL_SORT_MERGE_ASOF_JOIN_ENABLED,
    ),
    (
        "krishiv.sql.session.localRelationSizeLimit",
        &LOCAL_RELATION_SIZE_LIMIT,
    ),
    (
        "krishiv.sql.sources.partitionOverwriteMode",
        &PARTITION_OVERWRITE_MODE,
    ),
    (
        "spark.sql.execution.arrow.useLargeVarTypes",
        &ARROW_USE_LARGE_VAR_TYPES,
    ),
    (
        "krishiv.connect.execute.abandonedTombstones",
        &CONNECT_EXECUTE_ABANDONED_TOMBSTONES,
    ),
    (
        "krishiv.connect.execute.maintenanceInterval",
        &CONNECT_EXECUTE_MAINTENANCE_INTERVAL,
    ),
    (
        "krishiv.connect.execute.senderMaxStreamSize",
        &CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE,
    ),
    (
        "krishiv.connect.session.maintenanceInterval",
        &CONNECT_SESSION_MAINTENANCE_INTERVAL,
    ),
    (
        "krishiv.io.parquet.dictionaryFilter.enabled",
        &IO_PARQUET_DICTIONARY_FILTER_ENABLED,
    ),
    (
        "spark.sql.execution.pythonUDF.arrow.enabled",
        &PYTHON_UDF_ARROW_ENABLED,
    ),
    (
        "krishiv.advisor.runtimeFilter.minDropPercent",
        &ADVISOR_RUNTIME_FILTER_MIN_DROP_PERCENT,
    ),
    (
        "krishiv.connect.http2.maxPendingAcceptResets",
        &CONNECT_HTTP2_MAX_PENDING_ACCEPT_RESETS,
    ),
    (
        "krishiv.flightSql.maxHandleBytesPerPrincipal",
        &FLIGHT_SQL_MAX_HANDLE_BYTES_PER_PRINCIPAL,
    ),
    (
        "krishiv.optimizer.groupKeyCodeMultiDimension",
        &GROUP_KEY_CODE_MULTI_DIMENSION,
    ),
    (
        "krishiv.sql.execution.arrow.useLargeVarTypes",
        &ARROW_USE_LARGE_VAR_TYPES,
    ),
    (
        "spark.sql.execution.arrow.maxRecordsPerBatch",
        &ARROW_MAX_RECORDS_PER_BATCH,
    ),
    (
        "spark.sql.execution.pythonUDTF.arrow.enabled",
        &PYTHON_UDTF_ARROW_ENABLED,
    ),
    (
        "spark.sql.session.localRelationChunkSizeRows",
        &LOCAL_RELATION_CHUNK_SIZE_ROWS,
    ),
    (
        "krishiv.runtimeFilter.adaptive.maxPassPercent",
        &RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT,
    ),
    (
        "krishiv.sql.execution.pythonUDF.arrow.enabled",
        &PYTHON_UDF_ARROW_ENABLED,
    ),
    (
        "spark.connect.execute.manager.detachedTimeout",
        &CONNECT_EXECUTE_DETACHED_TIMEOUT,
    ),
    (
        "spark.sql.execution.pandas.structHandlingMode",
        &PANDAS_STRUCT_HANDLING_MODE,
    ),
    (
        "spark.sql.optimizer.windowGroupLimitThreshold",
        &WINDOW_GROUP_LIMIT_THRESHOLD,
    ),
    (
        "spark.sql.session.localRelationCacheThreshold",
        &LOCAL_RELATION_CACHE_THRESHOLD,
    ),
    (
        "spark.sql.session.localRelationChunkSizeBytes",
        &LOCAL_RELATION_CHUNK_SIZE_BYTES,
    ),
    (
        "krishiv.runtimeFilter.bloom.minApplicationRows",
        &RUNTIME_FILTER_BLOOM_MIN_APPLICATION_ROWS,
    ),
    (
        "krishiv.sql.execution.arrow.maxRecordsPerBatch",
        &ARROW_MAX_RECORDS_PER_BATCH,
    ),
    (
        "krishiv.sql.execution.pythonUDTF.arrow.enabled",
        &PYTHON_UDTF_ARROW_ENABLED,
    ),
    (
        "krishiv.sql.session.localRelationChunkSizeRows",
        &LOCAL_RELATION_CHUNK_SIZE_ROWS,
    ),
    (
        "krishiv.connect.execute.senderMaxStreamDuration",
        &CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION,
    ),
    (
        "krishiv.sql.execution.pandas.structHandlingMode",
        &PANDAS_STRUCT_HANDLING_MODE,
    ),
    (
        "krishiv.sql.optimizer.windowGroupLimitThreshold",
        &WINDOW_GROUP_LIMIT_THRESHOLD,
    ),
    (
        "krishiv.sql.session.localRelationCacheThreshold",
        &LOCAL_RELATION_CACHE_THRESHOLD,
    ),
    (
        "krishiv.sql.session.localRelationChunkSizeBytes",
        &LOCAL_RELATION_CHUNK_SIZE_BYTES,
    ),
    (
        "spark.sql.execution.pandas.inferPandasDictAsMap",
        &PANDAS_INFER_DICT_AS_MAP,
    ),
    (
        "spark.sql.optimizer.runtime.bloomFilter.enabled",
        &RUNTIME_FILTER_BLOOM_ENABLED,
    ),
    (
        "krishiv.catalog.delegation.maxExchangesPerMinute",
        &CATALOG_DELEGATION_MAX_EXCHANGES_PER_MINUTE,
    ),
    (
        "krishiv.sql.execution.pandas.inferPandasDictAsMap",
        &PANDAS_INFER_DICT_AS_MAP,
    ),
    (
        "krishiv.sql.optimizer.runtime.bloomFilter.enabled",
        &RUNTIME_FILTER_BLOOM_ENABLED,
    ),
    (
        "spark.connect.execute.manager.maintenanceInterval",
        &CONNECT_EXECUTE_MAINTENANCE_INTERVAL,
    ),
    (
        "spark.connect.session.manager.maintenanceInterval",
        &CONNECT_SESSION_MAINTENANCE_INTERVAL,
    ),
    (
        "spark.sql.pyspark.inferNestedDictAsStruct.enabled",
        &PYSPARK_INFER_NESTED_DICT_AS_STRUCT,
    ),
    (
        "krishiv.sql.pyspark.inferNestedDictAsStruct.enabled",
        &PYSPARK_INFER_NESTED_DICT_AS_STRUCT,
    ),
    (
        "spark.sql.legacy.skipTypeValidationOnAlterPartition",
        &LEGACY_SKIP_TYPE_VALIDATION_ON_ALTER_PARTITION,
    ),
    (
        "spark.sql.execution.pandas.convertToArrowArraySafely",
        &PANDAS_CONVERT_TO_ARROW_ARRAY_SAFELY,
    ),
    (
        "krishiv.sql.legacy.skipTypeValidationOnAlterPartition",
        &LEGACY_SKIP_TYPE_VALIDATION_ON_ALTER_PARTITION,
    ),
    (
        "spark.connect.execute.manager.abandonedTombstonesSize",
        &CONNECT_EXECUTE_ABANDONED_TOMBSTONES,
    ),
    (
        "spark.sql.session.localRelationBatchOfChunksSizeBytes",
        &LOCAL_RELATION_BATCH_OF_CHUNKS_SIZE_BYTES,
    ),
    (
        "krishiv.sql.execution.pandas.convertToArrowArraySafely",
        &PANDAS_CONVERT_TO_ARROW_ARRAY_SAFELY,
    ),
    (
        "spark.connect.execute.reattachable.senderMaxStreamSize",
        &CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE,
    ),
    (
        "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
        &ARROW_PYSPARK_SELF_DESTRUCT_ENABLED,
    ),
    (
        "krishiv.sql.session.localRelationBatchOfChunksSizeBytes",
        &LOCAL_RELATION_BATCH_OF_CHUNKS_SIZE_BYTES,
    ),
    (
        "krishiv.catalog.delegation.maxDistinctExchangesPerMinute",
        &CATALOG_DELEGATION_MAX_DISTINCT_EXCHANGES_PER_MINUTE,
    ),
    (
        "krishiv.sql.execution.arrow.pyspark.selfDestruct.enabled",
        &ARROW_PYSPARK_SELF_DESTRUCT_ENABLED,
    ),
    (
        "spark.sql.execution.arrow.pyspark.validateSchema.enabled",
        &ARROW_PYSPARK_VALIDATE_SCHEMA,
    ),
    (
        "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled",
        &ARROW_PYSPARK_VALIDATE_SCHEMA,
    ),
    (
        "spark.connect.execute.reattachable.observerRetryBufferSize",
        &CONNECT_EXECUTE_RETRY_BUFFER_SIZE,
    ),
    (
        "spark.connect.execute.reattachable.senderMaxStreamDuration",
        &CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION,
    ),
    (
        "spark.connect.session.manager.closedSessionsTombstonesSize",
        &CONNECT_SESSION_TOMBSTONES,
    ),
    (
        "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
        &PYSPARK_LEGACY_INFER_MAP_FROM_FIRST,
    ),
    (
        "krishiv.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
        &PYSPARK_LEGACY_INFER_MAP_FROM_FIRST,
    ),
    (
        "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
        &PYSPARK_LEGACY_INFER_ARRAY_FROM_FIRST,
    ),
    (
        "krishiv.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
        &PYSPARK_LEGACY_INFER_ARRAY_FROM_FIRST,
    ),
];
