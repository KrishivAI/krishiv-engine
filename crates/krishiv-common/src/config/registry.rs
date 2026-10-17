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
    value_type: ValueType::String,
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
