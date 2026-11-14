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
    ("krishiv.ui.enabled", &UI_ENABLED),
    ("krishiv.connect.port", &CONNECT_PORT),
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
    ("krishiv.agent.gateBytes", &AGENT_GATE_BYTES),
    ("krishiv.connect.address", &CONNECT_ADDRESS),
    ("krishiv.metrics.enabled", &METRICS_ENABLED),
    ("krishiv.network.timeout", &NETWORK_TIMEOUT),
    ("krishiv.resultCache.ttl", &RESULT_CACHE_TTL),
    ("krishiv.sql.defaultPath", &SQL_DEFAULT_PATH),
    ("krishiv.delta.logThreads", &DELTA_LOG_THREADS),
    ("krishiv.experimental.ivm", &EXPERIMENTAL_IVM),
    ("krishiv.io.cache.enabled", &IO_CACHE_ENABLED),
    ("krishiv.io.coalesceBytes", &IO_COALESCE_BYTES),
    ("krishiv.io.prefetchBytes", &IO_PREFETCH_BYTES),
    ("krishiv.io.strictOptions", &IO_STRICT_OPTIONS),
    ("krishiv.spill.zstd.level", &SPILL_ZSTD_LEVEL),
    ("krishiv.sql.ansi.enabled", &ANSI_ENABLED),
    ("krishiv.sql.path.enabled", &SQL_PATH_ENABLED),
    ("krishiv.io.requestTimeout", &IO_REQUEST_TIMEOUT),
    ("krishiv.spill.compression", &SPILL_COMPRESSION),
    ("krishiv.catalog.maxColumns", &CATALOG_MAX_COLUMNS),
    ("krishiv.deterministicOrder", &DETERMINISTIC_ORDER),
    ("krishiv.exec.agg.rowLayout", &EXEC_AGG_ROW_LAYOUT),
    ("krishiv.exec.join.strategy", &EXEC_JOIN_STRATEGY),
    ("krishiv.io.blockingThreads", &IO_BLOCKING_THREADS),
    ("krishiv.io.cache.diskBytes", &IO_CACHE_DISK_BYTES),
    ("krishiv.io.footerReadBytes", &IO_FOOTER_READ_BYTES),
    ("krishiv.io.maxRequestBytes", &IO_MAX_REQUEST_BYTES),
    ("krishiv.optimizer.joinCost", &JOIN_COST),
    ("krishiv.python.udf.threads", &PYTHON_UDF_THREADS),
    ("spark.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("krishiv.connect.maxSessions", &CONNECT_MAX_SESSIONS),
    ("krishiv.exec.agg.directKeys", &EXEC_AGG_DIRECT_KEYS),
    ("krishiv.exec.join.buildSide", &EXEC_JOIN_BUILD_SIDE),
    ("krishiv.exec.join.radixBits", &EXEC_JOIN_RADIX_BITS),
    ("krishiv.exec.join.rowLayout", &EXEC_JOIN_ROW_LAYOUT),
    ("krishiv.io.cache.blockBytes", &IO_CACHE_BLOCK_BYTES),
    ("krishiv.query.maxQueuedTime", &QUERY_MAX_QUEUED_TIME),
    ("krishiv.resultCache.enabled", &RESULT_CACHE_ENABLED),
    ("krishiv.session.memoryLimit", &SESSION_MEMORY_LIMIT),
    ("spark.sql.binaryOutputStyle", &BINARY_OUTPUT_STYLE),
    ("spark.sql.mapKeyDedupPolicy", &MAP_KEY_DEDUP_POLICY),
    ("krishiv.connect.maxPlanDepth", &CONNECT_MAX_PLAN_DEPTH),
    ("krishiv.io.cache.memoryBytes", &IO_CACHE_MEMORY_BYTES),
    ("krishiv.memory.residentGuard", &MEMORY_RESIDENT_GUARD),
    ("krishiv.resultCache.maxBytes", &RESULT_CACHE_MAX_BYTES),
    ("krishiv.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("spark.sql.shuffle.partitions", &SHUFFLE_PARTITIONS_HINT),
    ("krishiv.exec.sort.hashBuckets", &EXEC_SORT_HASH_BUCKETS),
    ("krishiv.io.coalesceWasteBytes", &IO_COALESCE_WASTE_BYTES),
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
    ("krishiv.connect.keepAlive.time", &CONNECT_KEEPALIVE_TIME),
    ("krishiv.connect.maxConnections", &CONNECT_MAX_CONNECTIONS),
    ("krishiv.exec.lateFetch.maxRows", &LATE_FETCH_MAX_ROWS),
    ("krishiv.io.lateMaterialization", &IO_LATE_MATERIALIZATION),
    ("krishiv.query.maxExecutionTime", &QUERY_MAX_EXECUTION_TIME),
    ("krishiv.server.shutdownTimeout", &SERVER_SHUTDOWN_TIMEOUT),
    ("spark.sql.cteRecursionRowLimit", &CTE_RECURSION_ROW_LIMIT),
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
    ("spark.sql.storeAssignmentPolicy", &STORE_ASSIGNMENT_POLICY),
    (
        "krishiv.catalog.maxMetadataBytes",
        &CATALOG_MAX_METADATA_BYTES,
    ),
    (
        "krishiv.connect.handshakeTimeout",
        &CONNECT_HANDSHAKE_TIMEOUT,
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
        "krishiv.io.maxConcurrentRequests",
        &IO_MAX_CONCURRENT_REQUESTS,
    ),
    ("krishiv.optimizer.joinEnumerator", &JOIN_ENUMERATOR),
    ("krishiv.sql.cteRecursionRowLimit", &CTE_RECURSION_ROW_LIMIT),
    ("krishiv.sql.files.commitProtocol", &FILES_COMMIT_PROTOCOL),
    ("spark.sql.avro.compression.codec", &AVRO_COMPRESSION_CODEC),
    (
        "spark.sql.cteRecursionLevelLimit",
        &CTE_RECURSION_LEVEL_LIMIT,
    ),
    (
        "krishiv.cache.defaultStorageLevel",
        &CACHE_DEFAULT_STORAGE_LEVEL,
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
        "krishiv.memory.reservationQuantum",
        &MEMORY_RESERVATION_QUANTUM,
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
        "krishiv.connect.execute.aheadBytes",
        &CONNECT_EXECUTE_AHEAD_BYTES,
    ),
    (
        "krishiv.io.credentials.refreshSkew",
        &IO_CREDENTIALS_REFRESH_SKEW,
    ),
    (
        "krishiv.memory.abortReleaseTimeout",
        &MEMORY_ABORT_RELEASE_TIMEOUT,
    ),
    (
        "krishiv.memory.spillScratchReserve",
        &MEMORY_SPILL_SCRATCH_RESERVE,
    ),
    ("krishiv.optimizer.eagerAggregation", &EAGER_AGGREGATION),
    ("krishiv.optimizer.ndvSampling.rows", &NDV_SAMPLING_ROWS),
    ("krishiv.optimizer.subqueryAsWindow", &SUBQUERY_AS_WINDOW),
    (
        "krishiv.sql.avro.compression.codec",
        &AVRO_COMPRESSION_CODEC,
    ),
    (
        "krishiv.sql.cteRecursionLevelLimit",
        &CTE_RECURSION_LEVEL_LIMIT,
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
        "krishiv.sql.files.maxPartitionBytes",
        &FILES_MAX_PARTITION_BYTES,
    ),
    (
        "krishiv.sql.legacy.timeParserPolicy",
        &LEGACY_TIME_PARSER_POLICY,
    ),
    (
        "krishiv.io.parquet.pageIndex.enabled",
        &IO_PARQUET_PAGE_INDEX_ENABLED,
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
        "krishiv.optimizer.ndvSampling.enabled",
        &NDV_SAMPLING_ENABLED,
    ),
    (
        "krishiv.optimizer.ndvSampling.timeout",
        &NDV_SAMPLING_TIMEOUT,
    ),
    (
        "spark.sql.legacy.statisticalAggregate",
        &LEGACY_STATISTICAL_AGGREGATE,
    ),
    (
        "krishiv.exec.join.sharedBroadcastBytes",
        &EXEC_JOIN_SHARED_BROADCAST_BYTES,
    ),
    (
        "krishiv.io.parquet.bloomFilter.enabled",
        &IO_PARQUET_BLOOM_FILTER_ENABLED,
    ),
    (
        "krishiv.runtimeFilter.adaptive.minRows",
        &RUNTIME_FILTER_ADAPTIVE_MIN_ROWS,
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
        "krishiv.connect.maxOperationsPerSession",
        &CONNECT_MAX_OPERATIONS_PER_SESSION,
    ),
    (
        "krishiv.connect.maxSessionsPerPrincipal",
        &CONNECT_MAX_SESSIONS_PER_PRINCIPAL,
    ),
    (
        "krishiv.optimizer.shareRepeatedSubtrees",
        &SHARE_REPEATED_SUBTREES,
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
        "spark.sql.sources.partitionOverwriteMode",
        &PARTITION_OVERWRITE_MODE,
    ),
    (
        "krishiv.experimental.unalignedCheckpoints",
        &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
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
        "krishiv.connect.http2.maxConcurrentStreams",
        &CONNECT_HTTP2_MAX_CONCURRENT_STREAMS,
    ),
    (
        "krishiv.sql.sources.partitionOverwriteMode",
        &PARTITION_OVERWRITE_MODE,
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
        "krishiv.connect.http2.maxPendingAcceptResets",
        &CONNECT_HTTP2_MAX_PENDING_ACCEPT_RESETS,
    ),
    (
        "spark.sql.execution.arrow.maxRecordsPerBatch",
        &ARROW_MAX_RECORDS_PER_BATCH,
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
        "spark.sql.optimizer.windowGroupLimitThreshold",
        &WINDOW_GROUP_LIMIT_THRESHOLD,
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
        "krishiv.sql.optimizer.windowGroupLimitThreshold",
        &WINDOW_GROUP_LIMIT_THRESHOLD,
    ),
    (
        "spark.sql.optimizer.runtime.bloomFilter.enabled",
        &RUNTIME_FILTER_BLOOM_ENABLED,
    ),
    (
        "krishiv.sql.optimizer.runtime.bloomFilter.enabled",
        &RUNTIME_FILTER_BLOOM_ENABLED,
    ),
    (
        "spark.connect.session.manager.maintenanceInterval",
        &CONNECT_SESSION_MAINTENANCE_INTERVAL,
    ),
    (
        "spark.sql.legacy.skipTypeValidationOnAlterPartition",
        &LEGACY_SKIP_TYPE_VALIDATION_ON_ALTER_PARTITION,
    ),
    (
        "krishiv.sql.legacy.skipTypeValidationOnAlterPartition",
        &LEGACY_SKIP_TYPE_VALIDATION_ON_ALTER_PARTITION,
    ),
    (
        "spark.connect.session.manager.closedSessionsTombstonesSize",
        &CONNECT_SESSION_TOMBSTONES,
    ),
];
