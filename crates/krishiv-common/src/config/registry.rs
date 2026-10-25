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
    &STORE_ASSIGNMENT_POLICY,
    &MAP_KEY_DEDUP_POLICY,
    &TARGET_PARTITION_BYTES,
    &EXEC_BATCH_SIZE,
    &EXEC_AGG_ROW_LAYOUT,
    &EXEC_JOIN_ROW_LAYOUT,
    &EXEC_JOIN_RADIX_BITS,
    &EXEC_SORT_HASH_BUCKETS,
    &SHUFFLE_PARTITIONS_HINT,
    &NETWORK_TIMEOUT,
    &MEMORY_LIMIT,
    &MEMORY_RESERVATION_QUANTUM,
    &MEMORY_SPILL_WAIT_TIMEOUT,
    &MEMORY_ABORT_RELEASE_TIMEOUT,
    &MEMORY_SPILL_SCRATCH_RESERVE,
    &SPILL_DIR,
    &SPILL_MAX_BYTES,
    &SPILL_COMPRESSION,
    &SPILL_ZSTD_LEVEL,
    &EXPERIMENTAL_IVM,
    &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
    &IO_STRICT_OPTIONS,
    &FILES_MAX_PARTITION_BYTES,
    &FILES_OPEN_COST_IN_BYTES,
    &FILES_MIN_PARTITION_NUM,
    &FILES_MAX_PARTITION_NUM,
    &BUCKETING_MAX_BUCKETS,
    &LEAF_NODE_DEFAULT_PARALLELISM,
    &IO_MAX_CONCURRENT_REQUESTS,
    &IO_BLOCKING_THREADS,
    &IO_COALESCE_BYTES,
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
];

/// Every key and Spark alias with its entry, sorted by (length, name) for
/// binary search: lookup is O(log n) with no initialisation or allocation,
/// and most probes compare only lengths, never the shared `krishiv.` prefix.
/// A test checks the order and that it holds exactly the keys and aliases
/// of [`REGISTRY`]; add new entries here in order.
pub static INDEX: &[(&str, &ConfigEntry)] = &[
    ("krishiv.spill.dir", &SPILL_DIR),
    ("krishiv.memory.limit", &MEMORY_LIMIT),
    ("krishiv.io.maxRetries", &IO_MAX_RETRIES),
    ("spark.network.timeout", &NETWORK_TIMEOUT),
    ("krishiv.exec.batchSize", &EXEC_BATCH_SIZE),
    ("krishiv.spill.maxBytes", &SPILL_MAX_BYTES),
    ("spark.sql.ansi.enabled", &ANSI_ENABLED),
    ("krishiv.network.timeout", &NETWORK_TIMEOUT),
    ("krishiv.experimental.ivm", &EXPERIMENTAL_IVM),
    ("krishiv.io.coalesceBytes", &IO_COALESCE_BYTES),
    ("krishiv.io.prefetchBytes", &IO_PREFETCH_BYTES),
    ("krishiv.io.strictOptions", &IO_STRICT_OPTIONS),
    ("krishiv.spill.zstd.level", &SPILL_ZSTD_LEVEL),
    ("krishiv.sql.ansi.enabled", &ANSI_ENABLED),
    ("krishiv.io.requestTimeout", &IO_REQUEST_TIMEOUT),
    ("krishiv.spill.compression", &SPILL_COMPRESSION),
    ("krishiv.exec.agg.rowLayout", &EXEC_AGG_ROW_LAYOUT),
    ("krishiv.io.blockingThreads", &IO_BLOCKING_THREADS),
    ("krishiv.io.footerReadBytes", &IO_FOOTER_READ_BYTES),
    ("krishiv.io.maxRequestBytes", &IO_MAX_REQUEST_BYTES),
    ("spark.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("krishiv.exec.join.radixBits", &EXEC_JOIN_RADIX_BITS),
    ("krishiv.exec.join.rowLayout", &EXEC_JOIN_ROW_LAYOUT),
    ("spark.sql.mapKeyDedupPolicy", &MAP_KEY_DEDUP_POLICY),
    ("krishiv.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("spark.sql.shuffle.partitions", &SHUFFLE_PARTITIONS_HINT),
    ("krishiv.exec.sort.hashBuckets", &EXEC_SORT_HASH_BUCKETS),
    ("krishiv.io.metadataCacheBytes", &IO_METADATA_CACHE_BYTES),
    ("krishiv.io.multipartPartBytes", &IO_MULTIPART_PART_BYTES),
    ("krishiv.runtimeFilter.enabled", &RUNTIME_FILTER_ENABLED),
    ("krishiv.sql.mapKeyDedupPolicy", &MAP_KEY_DEDUP_POLICY),
    ("krishiv.io.lateMaterialization", &IO_LATE_MATERIALIZATION),
    (
        "krishiv.memory.spillWaitTimeout",
        &MEMORY_SPILL_WAIT_TIMEOUT,
    ),
    ("spark.sql.files.maxPartitionNum", &FILES_MAX_PARTITION_NUM),
    ("spark.sql.files.minPartitionNum", &FILES_MIN_PARTITION_NUM),
    ("spark.sql.files.openCostInBytes", &FILES_OPEN_COST_IN_BYTES),
    ("spark.sql.storeAssignmentPolicy", &STORE_ASSIGNMENT_POLICY),
    (
        "krishiv.io.maxConcurrentRequests",
        &IO_MAX_CONCURRENT_REQUESTS,
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
        "krishiv.sql.shuffle.partitionsHint",
        &SHUFFLE_PARTITIONS_HINT,
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
        "spark.sql.leafNodeDefaultParallelism",
        &LEAF_NODE_DEFAULT_PARALLELISM,
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
        "spark.sql.sources.bucketing.maxBuckets",
        &BUCKETING_MAX_BUCKETS,
    ),
    (
        "krishiv.runtimeFilter.exactSet.maxValues",
        &RUNTIME_FILTER_EXACT_SET_MAX_VALUES,
    ),
    (
        "krishiv.sql.sources.bucketing.maxBuckets",
        &BUCKETING_MAX_BUCKETS,
    ),
    (
        "krishiv.experimental.unalignedCheckpoints",
        &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
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
    default: Some(ConfigValue::Bool(false)),
    doc: "Internal: grouped aggregations keep their keys as fixed-width rows in \
          radix-partitioned tables and merge the partitions in parallel. Results are the \
          same either way; aggregations with string, binary, nested or dictionary keys \
          keep the default path.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Hash join keys in the row layout with a size-adaptive index (P2-T158).
pub static EXEC_JOIN_ROW_LAYOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.rowLayout",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Internal: hash joins store their keys as fixed-width rows and index them with a \
          table that switches by size between a flat and a salted directory. Results are \
          the same either way; joins whose key types have no row encoding keep the \
          default path.",
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
    doc: "Byte ranges of one file closer than this are fetched with one request.",
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
