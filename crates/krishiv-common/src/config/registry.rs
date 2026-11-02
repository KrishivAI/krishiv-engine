// SPDX-License-Identifier: Apache-2.0
//! The config entries. Add new entries here and to [`REGISTRY`].

use std::borrow::Cow;
use std::time::Duration;

use super::{ConfigEntry, ConfigValue, Scope, ValueType};

const MIB: u64 = 1024 * 1024;

/// Every registered entry. Order is irrelevant; docs sort by key.
pub static REGISTRY: &[&ConfigEntry] = &[
    &ANSI_ENABLED,
    &SESSION_TIME_ZONE,
    &SQL_PATH_ENABLED,
    &SQL_DEFAULT_PATH,
    &STORE_ASSIGNMENT_POLICY,
    &CTE_PRECEDENCE_POLICY,
    &MAP_KEY_DEDUP_POLICY,
    &TARGET_PARTITION_BYTES,
    &EXEC_BATCH_SIZE,
    &EXEC_AGG_ROW_LAYOUT,
    &EXEC_JOIN_ROW_LAYOUT,
    &EXEC_JOIN_BUILD_SIDE,
    &EXEC_JOIN_RADIX_BITS,
    &EXEC_JOIN_STRATEGY,
    &EXEC_JOIN_SMJ_BUFFER_BYTES,
    &EXEC_SEED,
    &EXEC_SORT_HASH_BUCKETS,
    &SHUFFLE_PARTITIONS_HINT,
    &NETWORK_TIMEOUT,
    &MEMORY_LIMIT,
    &MEMORY_RESERVATION_QUANTUM,
    &MEMORY_SPILL_WAIT_TIMEOUT,
    &MEMORY_ABORT_RELEASE_TIMEOUT,
    &MEMORY_SPILL_SCRATCH_RESERVE,
    &MEMORY_RESIDENT_GUARD,
    &CACHE_MAX_BYTES,
    &SPILL_DIR,
    &SPILL_MAX_BYTES,
    &SPILL_COMPRESSION,
    &SPILL_ZSTD_LEVEL,
    &SPILL_DROP_BEHIND_BYTES,
    &EXPERIMENTAL_IVM,
    &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
    &IO_STRICT_OPTIONS,
    &FILES_MAX_PARTITION_BYTES,
    &FILES_OPEN_COST_IN_BYTES,
    &FILES_MIN_PARTITION_NUM,
    &FILES_MAX_PARTITION_NUM,
    &PARTITION_OVERWRITE_MODE,
    &FILES_COMMIT_PROTOCOL,
    &BUCKETING_MAX_BUCKETS,
    &LEAF_NODE_DEFAULT_PARALLELISM,
    &IO_MAX_CONCURRENT_REQUESTS,
    &IO_BLOCKING_THREADS,
    &IO_COALESCE_BYTES,
    &IO_COALESCE_WASTE_BYTES,
    &IO_COALESCE_WASTE_DIVISOR,
    &IO_MAX_REQUEST_BYTES,
    &IO_PREFETCH_BYTES,
    &IO_LATE_MATERIALIZATION,
    &IO_FOOTER_READ_BYTES,
    &IO_METADATA_CACHE_BYTES,
    &IO_MULTIPART_PART_BYTES,
    &IO_MAX_RETRIES,
    &IO_REQUEST_TIMEOUT,
    &RUNTIME_FILTER_ENABLED,
    &RUNTIME_FILTER_BLOOM_ENABLED,
    &RUNTIME_FILTER_IN_LIST_MAX_VALUES,
    &RUNTIME_FILTER_EXACT_SET_MAX_VALUES,
    &RUNTIME_FILTER_BLOOM_MAX_ITEMS,
    &RUNTIME_FILTER_BLOOM_MAX_BYTES,
    &RUNTIME_FILTER_BLOOM_MIN_APPLICATION_ROWS,
    &RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT,
    &RUNTIME_FILTER_ADAPTIVE_MIN_ROWS,
    &WINDOW_GROUP_LIMIT_THRESHOLD,
    &CTE_RECURSION_LEVEL_LIMIT,
    &CTE_RECURSION_ROW_LIMIT,
    &LEGACY_CODING_ERROR_ACTION,
    &LEGACY_JAVA_CHARSETS,
    &LEGACY_STATISTICAL_AGGREGATE,
    &LEGACY_VIEW_SCHEMA_BINDING_MODE,
    &LEGACY_VIEW_SCHEMA_COMPENSATION,
    &METRICS_ENABLED,
    &AVRO_COMPRESSION_CODEC,
    &AVRO_DATETIME_REBASE_MODE_IN_READ,
    &AGENT_GATE_BYTES,
    &SHARE_CTE_INSTANCES,
];

/// Every key and Spark alias with its entry, sorted by (length, name) for
/// binary search: lookup is O(log n) with no initialisation or allocation,
/// and most probes compare only lengths, never the shared `krishiv.` prefix.
/// A test checks the order and that it holds exactly the keys and aliases
/// of [`REGISTRY`]; add new entries here in order.
pub static INDEX: &[(&str, &ConfigEntry)] = &[
    ("krishiv.exec.seed", &EXEC_SEED),
    ("krishiv.spill.dir", &SPILL_DIR),
    ("krishiv.memory.limit", &MEMORY_LIMIT),
    ("krishiv.io.maxRetries", &IO_MAX_RETRIES),
    ("spark.network.timeout", &NETWORK_TIMEOUT),
    ("spark.sql.defaultPath", &SQL_DEFAULT_PATH),
    ("krishiv.cache.maxBytes", &CACHE_MAX_BYTES),
    ("krishiv.exec.batchSize", &EXEC_BATCH_SIZE),
    ("krishiv.spill.maxBytes", &SPILL_MAX_BYTES),
    ("spark.sql.ansi.enabled", &ANSI_ENABLED),
    ("spark.sql.path.enabled", &SQL_PATH_ENABLED),
    ("krishiv.agent.gateBytes", &AGENT_GATE_BYTES),
    ("krishiv.metrics.enabled", &METRICS_ENABLED),
    ("krishiv.network.timeout", &NETWORK_TIMEOUT),
    ("krishiv.sql.defaultPath", &SQL_DEFAULT_PATH),
    ("krishiv.experimental.ivm", &EXPERIMENTAL_IVM),
    ("krishiv.io.coalesceBytes", &IO_COALESCE_BYTES),
    ("krishiv.io.prefetchBytes", &IO_PREFETCH_BYTES),
    ("krishiv.io.strictOptions", &IO_STRICT_OPTIONS),
    ("krishiv.spill.zstd.level", &SPILL_ZSTD_LEVEL),
    ("krishiv.sql.ansi.enabled", &ANSI_ENABLED),
    ("krishiv.sql.path.enabled", &SQL_PATH_ENABLED),
    ("krishiv.io.requestTimeout", &IO_REQUEST_TIMEOUT),
    ("krishiv.spill.compression", &SPILL_COMPRESSION),
    ("krishiv.exec.agg.rowLayout", &EXEC_AGG_ROW_LAYOUT),
    ("krishiv.exec.join.strategy", &EXEC_JOIN_STRATEGY),
    ("krishiv.io.blockingThreads", &IO_BLOCKING_THREADS),
    ("krishiv.io.footerReadBytes", &IO_FOOTER_READ_BYTES),
    ("krishiv.io.maxRequestBytes", &IO_MAX_REQUEST_BYTES),
    ("spark.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("krishiv.exec.join.buildSide", &EXEC_JOIN_BUILD_SIDE),
    ("krishiv.exec.join.radixBits", &EXEC_JOIN_RADIX_BITS),
    ("krishiv.exec.join.rowLayout", &EXEC_JOIN_ROW_LAYOUT),
    ("spark.sql.mapKeyDedupPolicy", &MAP_KEY_DEDUP_POLICY),
    ("krishiv.memory.residentGuard", &MEMORY_RESIDENT_GUARD),
    ("krishiv.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("spark.sql.shuffle.partitions", &SHUFFLE_PARTITIONS_HINT),
    ("krishiv.exec.sort.hashBuckets", &EXEC_SORT_HASH_BUCKETS),
    ("krishiv.io.coalesceWasteBytes", &IO_COALESCE_WASTE_BYTES),
    ("krishiv.io.metadataCacheBytes", &IO_METADATA_CACHE_BYTES),
    ("krishiv.io.multipartPartBytes", &IO_MULTIPART_PART_BYTES),
    ("krishiv.runtimeFilter.enabled", &RUNTIME_FILTER_ENABLED),
    ("krishiv.spill.dropBehindBytes", &SPILL_DROP_BEHIND_BYTES),
    ("krishiv.sql.mapKeyDedupPolicy", &MAP_KEY_DEDUP_POLICY),
    ("spark.sql.legacy.javaCharsets", &LEGACY_JAVA_CHARSETS),
    ("krishiv.io.lateMaterialization", &IO_LATE_MATERIALIZATION),
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
    ("spark.sql.files.maxPartitionNum", &FILES_MAX_PARTITION_NUM),
    ("spark.sql.files.minPartitionNum", &FILES_MIN_PARTITION_NUM),
    ("spark.sql.files.openCostInBytes", &FILES_OPEN_COST_IN_BYTES),
    ("spark.sql.storeAssignmentPolicy", &STORE_ASSIGNMENT_POLICY),
    (
        "krishiv.io.maxConcurrentRequests",
        &IO_MAX_CONCURRENT_REQUESTS,
    ),
    ("krishiv.sql.cteRecursionRowLimit", &CTE_RECURSION_ROW_LIMIT),
    ("krishiv.sql.files.commitProtocol", &FILES_COMMIT_PROTOCOL),
    ("spark.sql.avro.compression.codec", &AVRO_COMPRESSION_CODEC),
    (
        "spark.sql.cteRecursionLevelLimit",
        &CTE_RECURSION_LEVEL_LIMIT,
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
        "spark.sql.files.maxPartitionBytes",
        &FILES_MAX_PARTITION_BYTES,
    ),
    (
        "krishiv.memory.abortReleaseTimeout",
        &MEMORY_ABORT_RELEASE_TIMEOUT,
    ),
    (
        "krishiv.memory.spillScratchReserve",
        &MEMORY_SPILL_SCRATCH_RESERVE,
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
        "krishiv.sql.shuffle.partitionsHint",
        &SHUFFLE_PARTITIONS_HINT,
    ),
    (
        "spark.sql.legacy.codingErrorAction",
        &LEGACY_CODING_ERROR_ACTION,
    ),
    (
        "krishiv.sql.files.maxPartitionBytes",
        &FILES_MAX_PARTITION_BYTES,
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
        "spark.sql.leafNodeDefaultParallelism",
        &LEAF_NODE_DEFAULT_PARALLELISM,
    ),
    (
        "spark.sql.legacy.ctePrecedencePolicy",
        &CTE_PRECEDENCE_POLICY,
    ),
    (
        "spark.sql.legacy.statisticalAggregate",
        &LEGACY_STATISTICAL_AGGREGATE,
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
        "spark.sql.legacy.viewSchemaCompensation",
        &LEGACY_VIEW_SCHEMA_COMPENSATION,
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
        "krishiv.sql.legacy.viewSchemaCompensation",
        &LEGACY_VIEW_SCHEMA_COMPENSATION,
    ),
    (
        "krishiv.sql.sources.partitionOverwriteMode",
        &PARTITION_OVERWRITE_MODE,
    ),
    (
        "krishiv.runtimeFilter.adaptive.maxPassPercent",
        &RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT,
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
];

/// ANSI SQL mode (Spark 4 default: on).
pub static ANSI_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.ansi.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "When true, follow ANSI SQL semantics: invalid casts and arithmetic overflow raise \
          errors instead of returning NULL.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.ansi.enabled"),
};

/// Session time zone.
pub static SESSION_TIME_ZONE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.session.timeZone",
    value_type: ValueType::TimeZone,
    default: None,
    doc: "Session time zone: a region ID such as `America/Los_Angeles` or a zone offset. \
          When unset, the local system time zone is used, as in Spark.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.session.timeZone"),
};

/// Whether `SET PATH` is allowed.
pub static SQL_PATH_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.path.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether `SET PATH` (the SQL standard routine and relation search path) is enabled.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.path.enabled"),
};

/// What `DEFAULT_PATH` expands to.
pub static SQL_DEFAULT_PATH: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.defaultPath",
    value_type: ValueType::String,
    default: Some(ConfigValue::String(Cow::Borrowed(""))),
    doc: "What `DEFAULT_PATH` expands to: a comma-separated `SET PATH` element list without `PATH`. \
          Empty means `system.builtin, system.session, current_schema`.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.defaultPath"),
};

/// Table-insertion type coercion policy.
pub static STORE_ASSIGNMENT_POLICY: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.storeAssignmentPolicy",
    value_type: ValueType::Enum(&["ANSI", "LEGACY", "STRICT"]),
    default: Some(ConfigValue::String(Cow::Borrowed("ANSI"))),
    doc: "Type coercion rules when inserting a value into a column of a different type.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.storeAssignmentPolicy"),
};

/// How nested `WITH` definitions with the same name resolve.
pub static CTE_PRECEDENCE_POLICY: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.ctePrecedencePolicy",
    value_type: ValueType::Enum(&["EXCEPTION", "LEGACY", "CORRECTED"]),
    default: Some(ConfigValue::String(Cow::Borrowed("CORRECTED"))),
    doc: "How nested WITH definitions with the same name resolve: EXCEPTION fails on a \
          nested definition shadowing an outer one, LEGACY inlines outer definitions first, \
          CORRECTED lets inner definitions shadow outer ones.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.legacy.ctePrecedencePolicy"),
};

/// Map key deduplication policy.
pub static MAP_KEY_DEDUP_POLICY: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.mapKeyDedupPolicy",
    value_type: ValueType::Enum(&["EXCEPTION", "LAST_WIN"]),
    default: Some(ConfigValue::String(Cow::Borrowed("EXCEPTION"))),
    doc: "How duplicate map keys are handled by the map functions (`map`, `map_concat`, \
          `map_from_arrays`, `map_from_entries`, `str_to_map`): `EXCEPTION` fails the query, \
          `LAST_WIN` keeps the value inserted last.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.mapKeyDedupPolicy"),
};

/// Target partition size, replacing a fixed shuffle partition count.
pub static TARGET_PARTITION_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.targetPartitionBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(128 * MIB)),
    doc: "Target size of a partition for scans and exchanges.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Rows per execution batch.
pub static EXEC_BATCH_SIZE: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.batchSize",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(4096)),
    doc: "Rows per batch passed between operators. Operators do about a millisecond of work \
          per batch, so the scheduler can switch tasks between batches.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Grouped aggregation in the row layout with radix-partitioned tables
/// (P2-T159).
pub static EXEC_AGG_ROW_LAYOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.agg.rowLayout",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: grouped aggregations keep their keys as fixed-width rows in \
          radix-partitioned tables and merge the partitions in parallel. Results are the \
          same either way; aggregations with nested or dictionary keys \
          keep the columnar path.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Hash joins build the smaller input of a semi, anti or left outer join
/// (P2-T166).
pub static EXEC_JOIN_BUILD_SIDE: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.buildSide",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a semi, anti or left outer join whose right input is estimated much \
          larger than its left builds its hash table from the left input (a right semi, \
          right anti or right outer join). Results are the same either way; false keeps \
          Spark's side (the right input).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Hash join keys in the row layout with a size-adaptive index (P2-T158).
pub static EXEC_JOIN_ROW_LAYOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.rowLayout",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: hash joins store their keys as fixed-width rows and index them with a \
          table that switches by size between a flat and a salted directory. Results are \
          the same either way; joins whose key types have no row encoding keep the \
          columnar path.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Hash buckets of a sort below a window (P2-T055).
pub static EXEC_SORT_HASH_BUCKETS: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.sort.hashBuckets",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: the sort below a window with a PARTITION BY splits its rows into buckets by \
          the hash of the partitioning keys and sorts, merges and evaluates the buckets in \
          parallel. Results are the same either way.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Level-0 radix partitions of a row-layout hash join build (P2-T158).
pub static EXEC_JOIN_RADIX_BITS: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.radixBits",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(3)),
    doc: "Internal: a row-layout hash join build splits its rows into 2^bits partitions by \
          the top bits of the key hash (3 to 6) and indexes them in parallel. Results are \
          the same for every value.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Forces the sort-merge join on equi-joins without a strategy hint (P2-T056).
pub static EXEC_JOIN_STRATEGY: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.strategy",
    value_type: ValueType::Enum(&["auto", "sortMerge"]),
    default: Some(ConfigValue::String(Cow::Borrowed("auto"))),
    doc: "Internal: `auto` runs equi-joins as hash joins unless a `MERGE` hint asks otherwise; \
          `sortMerge` runs every equi-join without a strategy hint as a sort-merge join, except \
          `NOT IN` and keys that cannot be ordered. Results are the same either way.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// The bytes one key's rows may hold in a sort-merge join (P2-T056).
pub static EXEC_JOIN_SMJ_BUFFER_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.smj.bufferBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(0)),
    doc: "Bytes of one join key's right-side rows a sort-merge join keeps in memory before it \
          moves them to a spill file; 0 takes a share of the memory pool. Spark bounds these \
          rows by count (`spark.sql.sortMergeJoinExec.buffer.in.memory.threshold` and \
          `spark.sql.sortMergeJoinExec.buffer.spill.threshold`), which have no effect here.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// The seed of deterministic execution (tests and the golden harness).
pub static EXEC_SEED: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.seed",
    value_type: ValueType::Int,
    default: None,
    doc: "Internal, for tests: seeds the key hashes of grouped aggregations and joins and \
          the order in which scan morsels are claimed, and runs aggregations in their \
          deterministic mode, so that the order of groups and of unordered `LIMIT` rows is \
          the same on every run. Unset, each query draws a random seed.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `spark.sql.shuffle.partitions`, accepted as a hint only.
pub static SHUFFLE_PARTITIONS_HINT: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.shuffle.partitionsHint",
    value_type: ValueType::Int,
    default: None,
    doc: "Shuffle partition count requested by the user. Treated as a hint: partitioning \
          is driven by `krishiv.exec.targetPartitionBytes`.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.shuffle.partitions"),
};

/// Default network timeout.
pub static NETWORK_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.network.timeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(120))),
    doc: "Default timeout for network interactions between cluster processes.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: Some("spark.network.timeout"),
};

/// Process-wide memory limit for query execution.
pub static MEMORY_LIMIT: ConfigEntry = ConfigEntry {
    key: "krishiv.memory.limit",
    value_type: ValueType::Size,
    default: None,
    doc: "Memory available to query execution in this process: the limit of the root memory \
          pool. Operators reserve memory before allocating and spill when the limit is reached. \
          When unset: 80% of the effective memory limit (the tightest cgroup limit, else \
          physical memory), or 2 GiB where that cannot be determined.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Granularity of memory reservations.
pub static MEMORY_RESERVATION_QUANTUM: ConfigEntry = ConfigEntry {
    key: "krishiv.memory.reservationQuantum",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(MIB)),
    doc: "Granularity of memory reservations: a consumer takes memory from its pools in steps \
          of this size (at most 1/64 of the smallest pool limit above it), so small reservations \
          rarely touch shared counters. Unused parts of a step are reclaimed before anything is \
          spilled. `1b` disables rounding.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// How long the memory arbitrator waits for requested spills.
pub static MEMORY_SPILL_WAIT_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.memory.spillWaitTimeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(2))),
    doc: "How long the memory arbitrator waits for a consumer asked to spill to begin spilling \
          before it gives up on it and aborts the query furthest above its fair share of memory. \
          A spill that has begun is awaited to its end.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Memory kept for spill writers' scratch buffers in every limited pool.
pub static MEMORY_SPILL_SCRATCH_RESERVE: ConfigEntry = ConfigEntry {
    key: "krishiv.memory.spillScratchReserve",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(512 * 1024)),
    doc: "Memory at the top of every limited memory pool (at most 1/8 of its limit) that only \
          spill writers' scratch buffers may use, so a full pool can still fund the spill that \
          frees it. Shared by every spill under the pool and released when each write ends. \
          `0` disables it.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Whether a limited memory pool spills allocator overhead (the resident-set guard).
pub static MEMORY_RESIDENT_GUARD: ConfigEntry = ConfigEntry {
    key: "krishiv.memory.residentGuard",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Whether the root memory pool keeps what the process holds beyond its reservations \
          (allocator overhead, memory used without a reservation) within an allowance of 1/8 of \
          the limit plus 64 MiB, by lowering the memory ordinary reservations may take so \
          operators spill. It needs a resident-set probe from the program that owns the \
          allocator: the `krishiv` CLI and the bench runner install one when started with a \
          limit below 4 GiB; elsewhere the setting has no effect. Spill I/O and arbitration \
          rounds whose spills are exhausted are exempt from it.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// How long a request waits for aborted queries to release memory.
pub static MEMORY_ABORT_RELEASE_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.memory.abortReleaseTimeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(30))),
    doc: "How long a memory request waits for queries the arbitrator aborted to release their \
          memory; aborted queries release at their next batch boundary. While their memory \
          covers the request, no other query is aborted. The request fails after this time.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Byte cap of the session relation cache.
pub static CACHE_MAX_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.cache.maxBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(512 * MIB)),
    doc: "Most bytes cached relations (`DataFrame.cache`, `persist`) may hold in memory, per \
          session, as Arrow batches under the `relation_cache` memory pool. Materialization is \
          lazy (on first use) and all-or-nothing: when caching a relation would exceed the cap, \
          the query runs uncached instead of failing. `0` disables caching: queries always run \
          uncached.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Directory for spill files.
pub static SPILL_DIR: ConfigEntry = ConfigEntry {
    key: "krishiv.spill.dir",
    value_type: ValueType::String,
    default: None,
    doc: "Directory for spill files (operator state written to local disk under memory \
          pressure). Each process creates its own subdirectory and removes it on exit; \
          subdirectories left behind by crashed processes are removed at the next start. \
          When unset: the first of `$XDG_CACHE_HOME` (or `~/.cache`) `/krishiv/spill`, \
          `/var/tmp/krishiv-spill-<uid>` and `<temp dir>/krishiv-spill-<uid>` that is not on a \
          RAM file system (tmpfs). Prefer a local SSD; network and FUSE file systems are \
          refused.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Byte cap on spill files.
pub static SPILL_MAX_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.spill.maxBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 * 1024 * MIB)),
    doc: "Most bytes spill files may occupy on disk at once, over all queries of the process. \
          A spill that would exceed it fails with `KRISHIV_SPILL_CAP_EXCEEDED`.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Spill file compression.
pub static SPILL_COMPRESSION: ConfigEntry = ConfigEntry {
    key: "krishiv.spill.compression",
    value_type: ValueType::Enum(&["lz4", "zstd", "none"]),
    default: Some(ConfigValue::String(Cow::Borrowed("lz4"))),
    doc: "Compression of the Arrow IPC buffers in spill files: `lz4` (fastest), `zstd` \
          (smaller files, more CPU) or `none`. Files are self-describing, so the setting \
          only affects files written afterwards.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Zstandard level for spill files.
pub static SPILL_ZSTD_LEVEL: ConfigEntry = ConfigEntry {
    key: "krishiv.spill.zstd.level",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(1)),
    doc: "Zstandard compression level for spill files when `krishiv.spill.compression` is \
          `zstd`: 1 is fastest; higher levels (up to 22) give smaller files and cost more CPU. \
          Levels from -131072 (ultra-fast) are accepted.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Spill-file page-cache drop-behind window.
pub static SPILL_DROP_BEHIND_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.spill.dropBehindBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(8 << 20)),
    doc: "How many bytes a spill writer writes before it asks the kernel to write them to disk \
          and drop them from the page cache (Linux). Dirty spill pages count against a memory \
          cgroup's limit and cannot be reclaimed until written back, so without this a spilling \
          query can push its cgroup into the OOM killer. A finished file is written back and \
          dropped too. `0` disables it; other platforms ignore it.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Experimental: incremental view maintenance.
pub static EXPERIMENTAL_IVM: ConfigEntry = ConfigEntry {
    key: "krishiv.experimental.ivm",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Experimental: incremental maintenance of materialized views.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Experimental: unaligned streaming checkpoints.
pub static EXPERIMENTAL_UNALIGNED_CHECKPOINTS: ConfigEntry = ConfigEntry {
    key: "krishiv.experimental.unalignedCheckpoints",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Experimental: allow unaligned checkpoints under sustained backpressure.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.strictOptions`.
pub static IO_STRICT_OPTIONS: ConfigEntry = ConfigEntry {
    key: "krishiv.io.strictOptions",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, a data source option that the format does not know is an error \
          (`KRISHIV_UNKNOWN_FORMAT_OPTION`). When false (the default, as in Spark), it is \
          ignored with a warning that suggests the closest known option.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.sql.files.maxPartitionBytes`.
pub static FILES_MAX_PARTITION_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.files.maxPartitionBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(128 * MIB)),
    doc: "Most bytes packed into one file-scan partition (Spark's split planning).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.files.maxPartitionBytes"),
};

/// `krishiv.sql.files.openCostInBytes`.
pub static FILES_OPEN_COST_IN_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.files.openCostInBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(4 * MIB)),
    doc: "Estimated cost of opening a file, in bytes that could be scanned in the same time; \
          added per file when files are packed into scan partitions.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.files.openCostInBytes"),
};

/// `krishiv.sql.files.minPartitionNum`.
pub static FILES_MIN_PARTITION_NUM: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.files.minPartitionNum",
    value_type: ValueType::PositiveInt,
    default: None,
    doc: "Suggested minimum number of file-scan partitions. When unset: \
          `krishiv.sql.leafNodeDefaultParallelism`, else the number of worker slots.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.files.minPartitionNum"),
};

/// `krishiv.sql.files.maxPartitionNum`.
pub static FILES_MAX_PARTITION_NUM: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.files.maxPartitionNum",
    value_type: ValueType::PositiveInt,
    default: None,
    doc: "Suggested maximum number of file-scan partitions: when splitting yields more, the \
          split size is raised so that about this many remain. Unset: no maximum.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.files.maxPartitionNum"),
};

/// `spark.sql.sources.bucketing.maxBuckets` (Spark's default: 100,000).
pub static BUCKETING_MAX_BUCKETS: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.sources.bucketing.maxBuckets",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(100_000)),
    doc: "Maximum `CLUSTERED BY ... INTO n BUCKETS` bucket count: a count above it fails \
          with `INVALID_BUCKET_COUNT`.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.sources.bucketing.maxBuckets"),
};

/// `krishiv.sql.leafNodeDefaultParallelism`.
pub static LEAF_NODE_DEFAULT_PARALLELISM: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.leafNodeDefaultParallelism",
    value_type: ValueType::PositiveInt,
    default: None,
    doc: "Default parallelism of leaf scans (used when `krishiv.sql.files.minPartitionNum` is \
          unset). When unset: the number of worker slots.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.leafNodeDefaultParallelism"),
};

/// `krishiv.io.maxConcurrentRequests`.
pub static IO_MAX_CONCURRENT_REQUESTS: ConfigEntry = ConfigEntry {
    key: "krishiv.io.maxConcurrentRequests",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(64)),
    doc: "Most object-store requests in flight at once per store, over all queries of the process. \
          The local file store also runs each request on a blocking thread of the I/O runtime, so \
          at most `krishiv.io.blockingThreads` of them run at once.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.blockingThreads`.
pub static IO_BLOCKING_THREADS: ConfigEntry = ConfigEntry {
    key: "krishiv.io.blockingThreads",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(8)),
    doc: "Most blocking threads of the I/O runtime, where local-file reads and writes run: at most \
          this many local-file requests run at once per executor (`krishiv.io.maxConcurrentRequests` \
          still bounds those in flight). Each such thread keeps freed read buffers in its own \
          allocator cache, so the bound also bounds that memory. Read when the session starts its \
          executor.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.coalesceBytes`.
pub static IO_COALESCE_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.coalesceBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(MIB)),
    doc: "Byte ranges of one file closer than this are fetched with one request, as long as \
          the request reads few enough bytes outside the ranges \
          (`krishiv.io.coalesceWasteBytes`, `krishiv.io.coalesceWasteDivisor`).",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Gap bytes any coalesced request may read.
pub static IO_COALESCE_WASTE_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.coalesceWasteBytes",
    value_type: ValueType::Size,
    default: None,
    doc: "Bytes outside the requested ranges that any coalesced read may fetch (besides \
          `krishiv.io.coalesceWasteDivisor`'s share). A read's whole span is reserved while it is \
          in flight, so this trades bytes read for fewer requests. When unset: 64 KiB for local \
          files, 1 MiB for object stores and HTTP, where a request costs latency.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// A coalesced request's share of gap bytes.
pub static IO_COALESCE_WASTE_DIVISOR: ConfigEntry = ConfigEntry {
    key: "krishiv.io.coalesceWasteDivisor",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(16)),
    doc: "A coalesced read may fetch up to 1/this of its bytes outside the requested ranges \
          (or `krishiv.io.coalesceWasteBytes`, whichever is larger).",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.maxRequestBytes`.
pub static IO_MAX_REQUEST_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.maxRequestBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(16 * MIB)),
    doc: "Ranges are not coalesced into requests larger than this (a single larger range is \
          still fetched whole).",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.prefetchBytes`.
pub static IO_PREFETCH_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.prefetchBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 * MIB)),
    doc: "Bytes a file scan may fetch ahead of decoding, per scan driver. Prefetched bytes are \
          reserved from the query's memory pool and dropped first under memory pressure.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.lateMaterialization`.
pub static IO_LATE_MATERIALIZATION: ConfigEntry = ConfigEntry {
    key: "krishiv.io.lateMaterialization",
    value_type: ValueType::Enum(&["auto", "on", "off"]),
    default: Some(ConfigValue::String(Cow::Borrowed("auto"))),
    doc: "Whether a Parquet scan evaluates its pushed filter on the filter's columns before \
          decoding the others for the rows that pass: `auto` decides per row group from the \
          fraction of rows the filter kept before, `on` and `off` force it. Returned rows are \
          the same either way; with the filter, decode errors confined to rows it rejects \
          (a corrupt page or invalid UTF-8 in a column it does not read) are not raised.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.footerReadBytes`.
pub static IO_FOOTER_READ_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.footerReadBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 * 1024)),
    doc: "Size of the speculative read at the end of a file that fetches its footer in one \
          request; a format may ask for more (up to 1 MiB).",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.metadataCacheBytes`.
pub static IO_METADATA_CACHE_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.metadataCacheBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 * MIB)),
    doc: "Size of the in-memory cache of decoded file footers and metadata (least recently used \
          entries are evicted). `0` disables it.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.multipartPartBytes`.
pub static IO_MULTIPART_PART_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.multipartPartBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(8 * MIB)),
    doc: "Part size of multipart uploads (at least 5 MiB; doubled every 1,000 parts so a file \
          never needs more than the 10,000 parts object stores allow).",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.maxRetries`.
pub static IO_MAX_RETRIES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.maxRetries",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(5)),
    doc: "Retries of an object-store request that failed transiently (throttling, unavailable, \
          timeout), with jittered exponential backoff honouring `Retry-After`.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.io.requestTimeout`.
pub static IO_REQUEST_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.io.requestTimeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(30))),
    doc: "Timeout of one object-store request attempt; a timed-out attempt is retried.",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.enabled`.
pub static RUNTIME_FILTER_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Plan runtime filters (design P2-T163): the build side of an inner or left semi hash join \
          publishes the min/max, a small value set or a Bloom filter of its keys to the scans \
          feeding the probe side, which skip splits, row groups and rows no build key can match. \
          Results are the same either way.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.sql.optimizer.runtime.bloomFilter.enabled`.
pub static RUNTIME_FILTER_BLOOM_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.optimizer.runtime.bloomFilter.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Allow Bloom filters among the runtime filters a join build publishes (min/max and value \
          sets are not affected). Krishiv's cost model decides when one is built; Spark's creation \
          and application size thresholds are not used.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.optimizer.runtime.bloomFilter.enabled"),
};

/// `krishiv.runtimeFilter.inList.maxValues`.
pub static RUNTIME_FILTER_IN_LIST_MAX_VALUES: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.inList.maxValues",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(50)),
    doc: "Most distinct build keys a runtime filter keeps as a value set; with more, a Bloom \
          filter (when planned) or only the min/max range is published.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.exactSet.maxValues`.
pub static RUNTIME_FILTER_EXACT_SET_MAX_VALUES: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.exactSet.maxValues",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(4096)),
    doc: "Most distinct build keys a runtime filter tests rows against exactly (a small set or \
          a hash set); with more, a Bloom filter (when planned) or only the min/max range is \
          used on rows.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.bloom.maxItems`.
pub static RUNTIME_FILTER_BLOOM_MAX_ITEMS: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.maxItems",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(4_000_000)),
    doc: "Most build keys a runtime Bloom filter is built for: a build estimated larger plans \
          none, and one that turns out larger publishes none.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.bloom.maxBytes`.
pub static RUNTIME_FILTER_BLOOM_MAX_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.maxBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(8 * MIB)),
    doc: "Largest runtime Bloom filter. A filter capped below its 1% false-positive size is still \
          built unless its expected false-positive rate exceeds 5%.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.bloom.minApplicationRows`.
pub static RUNTIME_FILTER_BLOOM_MIN_APPLICATION_ROWS: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.minApplicationRows",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(1_000_000)),
    doc: "A runtime Bloom filter is planned only for a scan estimated at this many rows or more \
          (and at least four times the build).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.adaptive.maxPassPercent`.
pub static RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.adaptive.maxPassPercent",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(90)),
    doc: "A scan driver stops evaluating a runtime filter on rows once it passes more than this \
          percentage of a measurement window (pruning splits and row groups with it continues).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.adaptive.minRows`.
pub static RUNTIME_FILTER_ADAPTIVE_MIN_ROWS: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.adaptive.minRows",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(65_536)),
    doc: "Rows per window over which a scan driver measures a runtime filter's pass rate.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `spark.sql.optimizer.windowGroupLimitThreshold`.
pub static WINDOW_GROUP_LIMIT_THRESHOLD: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.optimizer.windowGroupLimitThreshold",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(1000)),
    doc: "Highest rank a filter on row_number, rank or dense_rank may keep for the rows past it \
          in each window partition to be dropped before the window is evaluated (Spark's window \
          group limit); -1 turns it off. Results are the same either way.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.optimizer.windowGroupLimitThreshold"),
};

/// `spark.sql.cteRecursionLevelLimit`.
pub static CTE_RECURSION_LEVEL_LIMIT: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.cteRecursionLevelLimit",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(100)),
    doc: "Most rounds a recursive common table expression may run before the query fails with \
          RECURSION_LEVEL_LIMIT_EXCEEDED; -1 removes the limit. A MAX RECURSION LEVEL clause on \
          the expression overrides it.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.cteRecursionLevelLimit"),
};

/// `spark.sql.cteRecursionRowLimit`.
pub static CTE_RECURSION_ROW_LIMIT: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.cteRecursionRowLimit",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(1_000_000)),
    doc: "Most rows a recursive common table expression may produce in all rounds before the \
          query fails with RECURSION_ROW_LIMIT_EXCEEDED; -1 removes the limit.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.cteRecursionRowLimit"),
};

/// `spark.sql.legacy.codingErrorAction`.
pub static LEGACY_CODING_ERROR_ACTION: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.codingErrorAction",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, `encode` writes `?` for a character the charset cannot hold and `decode` \
          reads malformed input as U+FFFD; when false both raise \
          `MALFORMED_CHARACTER_CODING`.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.legacy.codingErrorAction"),
};

/// `spark.sql.legacy.javaCharsets`.
pub static LEGACY_JAVA_CHARSETS: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.javaCharsets",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, `encode` and `decode` also accept `windows-1252` and the ISO-8859 family \
          under Java's names; other Java charsets are not supported. When false they accept \
          only Spark's default seven.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.legacy.javaCharsets"),
};

/// `spark.sql.legacy.statisticalAggregate`.
pub static LEGACY_STATISTICAL_AGGREGATE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.statisticalAggregate",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, `stddev_samp`, `var_samp`, `covar_samp` and `corr` of a single row, and \
          `skewness` and `kurtosis` of values without variance, return NaN; when false they \
          return NULL, as Spark 4 does.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.legacy.statisticalAggregate"),
};

/// `spark.sql.legacy.viewSchemaBindingMode`.
pub static LEGACY_VIEW_SCHEMA_BINDING_MODE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.viewSchemaBindingMode",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "When true, a permanent view may take `WITH SCHEMA BINDING | COMPENSATION | TYPE \
          EVOLUTION | EVOLUTION`, records its schema binding mode (shown by `DESCRIBE EXTENDED` \
          and `SHOW CREATE TABLE`) and reads by it; a view stored without a mode reads as \
          `BINDING`. When false, the clause is `FEATURE_NOT_ENABLED` and every view reads as \
          before the modes existed (see `viewSchemaCompensation`).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.legacy.viewSchemaBindingMode"),
};

/// `spark.sql.legacy.viewSchemaCompensation`.
pub static LEGACY_VIEW_SCHEMA_COMPENSATION: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.viewSchemaCompensation",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "When true, a view created without `WITH SCHEMA` is `COMPENSATION` (a changed column \
          type is cast to the stored one with an ANSI cast) and a view without a mode casts so \
          when `viewSchemaBindingMode` is false; when false, such views are `BINDING` (only \
          up-casts are allowed).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.legacy.viewSchemaCompensation"),
};

/// `krishiv.metrics.enabled`.
pub static METRICS_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.metrics.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, every query that reaches execution folds its metrics into the engine's \
          metrics registry once, at query end (`Session::metrics`, `krishiv.metrics()`); \
          when false the fold is skipped entirely. Engine gauges (memory, spill disk, \
          metadata cache size, build info) are evaluated when the snapshot is read either way.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `spark.sql.avro.compression.codec`.
pub static AVRO_COMPRESSION_CODEC: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.avro.compression.codec",
    value_type: ValueType::Enum(&[
        "uncompressed",
        "deflate",
        "snappy",
        "bzip2",
        "xz",
        "zstandard",
    ]),
    default: Some(ConfigValue::String(Cow::Borrowed("snappy"))),
    doc: "The block codec of Avro files written without a `compression` option. Writing with \
          `bzip2`, `xz` or `zstandard` is refused.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.avro.compression.codec"),
};

/// `spark.sql.avro.datetimeRebaseModeInRead`.
pub static AVRO_DATETIME_REBASE_MODE_IN_READ: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.avro.datetimeRebaseModeInRead",
    value_type: ValueType::Enum(&["EXCEPTION", "CORRECTED", "LEGACY"]),
    default: Some(ConfigValue::String(Cow::Borrowed("EXCEPTION"))),
    doc: "How an Avro read treats dates and timestamps before 1582-10-15 in files whose writer \
          is unknown (files written by Spark 3.0 or later say how they were written): \
          `EXCEPTION` fails, `CORRECTED` reads the stored values, `LEGACY` is refused. Used when \
          the `datetimeRebaseMode` option is not given.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.avro.datetimeRebaseModeInRead"),
};

/// `spark.sql.sources.partitionOverwriteMode`.
pub static PARTITION_OVERWRITE_MODE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.sources.partitionOverwriteMode",
    value_type: ValueType::Enum(&["STATIC", "DYNAMIC"]),
    default: Some(ConfigValue::String(Cow::Borrowed("STATIC"))),
    doc: "Which partitions an overwrite of partitioned output replaces: `STATIC` (the default, \
          as in Spark) replaces the whole output directory once the new files are complete; \
          `DYNAMIC` replaces only the partition directories the write produced.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.sources.partitionOverwriteMode"),
};

/// `krishiv.sql.files.commitProtocol`.
pub static FILES_COMMIT_PROTOCOL: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.files.commitProtocol",
    value_type: ValueType::Enum(&["AUTO", "LOG", "NONE"]),
    default: Some(ConfigValue::String(Cow::Borrowed("NONE"))),
    doc: "How a write to a plain directory is committed: `LOG` records every commit in \
          `_krishiv_log/` (put-if-absent, so concurrent writers cannot both take a version), and \
          Krishiv readers of the directory then see only committed files; `NONE` is Spark's plain \
          commit (files under their final names, `_SUCCESS`; the default until the log is the \
          default); `AUTO` uses the log \
          where the store supports put-if-absent and the plain commit, with a warning, where it \
          does not.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.agent.gateBytes`.
pub static AGENT_GATE_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.agent.gateBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(1024 * MIB)),
    doc: "Agent mode (`krishiv sql --agent`, the REPL's agent output, MCP): a statement whose estimated \
          table scans reach this many bytes is not run; the answer is the estimate and the plan \
          outline, and the caller repeats it with `--allow-heavy` to run it anyway. A statement whose \
          tables report no statistics runs. 0 turns the gate off.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.sql.optimizer.shareCteInstances`.
pub static SHARE_CTE_INSTANCES: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.optimizer.shareCteInstances",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Compute a deterministic CTE body once per query when two or more references to it are \
          identical after optimization and the body holds a pipeline breaker (an aggregate, join, \
          window, sort, distinct or set operation); every reference reads the one result. A group \
          is kept as separate copies when a runtime filter reaches one of its members. Results \
          are the same either way. Nondeterministic CTE bodies referenced twice or more are \
          always computed once, as Spark does: this switch does not affect them.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};
