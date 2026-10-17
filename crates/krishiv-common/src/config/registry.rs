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
    &EXPERIMENTAL_IVM,
    &EXPERIMENTAL_UNALIGNED_CHECKPOINTS,
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
