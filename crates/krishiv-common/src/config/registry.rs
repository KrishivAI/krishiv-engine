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
    &TARGET_PARTITION_BYTES,
    &SHUFFLE_PARTITIONS_HINT,
    &NETWORK_TIMEOUT,
    &MEMORY_LIMIT,
    &MEMORY_RESERVATION_QUANTUM,
    &MEMORY_SPILL_WAIT_TIMEOUT,
    &MEMORY_ABORT_RELEASE_TIMEOUT,
    &SPILL_DIR,
    &SPILL_MAX_BYTES,
    &SPILL_COMPRESSION,
    &SPILL_ZSTD_LEVEL,
    &EXPERIMENTAL_IVM,
    &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
];

/// Every key and Spark alias with its entry, sorted by (length, name) for
/// binary search: lookup is O(log n) with no initialisation or allocation,
/// and most probes compare only lengths, never the shared `krishiv.` prefix.
/// A test checks the order and that it holds exactly the keys and aliases
/// of [`REGISTRY`]; add new entries here in order.
pub static INDEX: &[(&str, &ConfigEntry)] = &[
    ("krishiv.spill.dir", &SPILL_DIR),
    ("krishiv.memory.limit", &MEMORY_LIMIT),
    ("spark.network.timeout", &NETWORK_TIMEOUT),
    ("krishiv.spill.maxBytes", &SPILL_MAX_BYTES),
    ("spark.sql.ansi.enabled", &ANSI_ENABLED),
    ("krishiv.network.timeout", &NETWORK_TIMEOUT),
    ("krishiv.experimental.ivm", &EXPERIMENTAL_IVM),
    ("krishiv.spill.zstd.level", &SPILL_ZSTD_LEVEL),
    ("krishiv.sql.ansi.enabled", &ANSI_ENABLED),
    ("krishiv.spill.compression", &SPILL_COMPRESSION),
    ("spark.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("krishiv.sql.session.timeZone", &SESSION_TIME_ZONE),
    ("spark.sql.shuffle.partitions", &SHUFFLE_PARTITIONS_HINT),
    (
        "krishiv.memory.spillWaitTimeout",
        &MEMORY_SPILL_WAIT_TIMEOUT,
    ),
    ("spark.sql.storeAssignmentPolicy", &STORE_ASSIGNMENT_POLICY),
    ("krishiv.exec.targetPartitionBytes", &TARGET_PARTITION_BYTES),
    (
        "krishiv.memory.reservationQuantum",
        &MEMORY_RESERVATION_QUANTUM,
    ),
    (
        "krishiv.sql.storeAssignmentPolicy",
        &STORE_ASSIGNMENT_POLICY,
    ),
    (
        "krishiv.memory.abortReleaseTimeout",
        &MEMORY_ABORT_RELEASE_TIMEOUT,
    ),
    (
        "krishiv.sql.shuffle.partitionsHint",
        &SHUFFLE_PARTITIONS_HINT,
    ),
    (
        "krishiv.experimental.unalignedCheckpoints",
        &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
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
    doc: "How long the memory arbitrator waits for requested spills to free memory before it \
          aborts the query furthest above its fair share of memory.",
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
