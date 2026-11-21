// SPDX-License-Identifier: Apache-2.0
//! The config entries of runtime filters: on/off, IN-list and exact-set sizes,
//! the Bloom filter caps and the adaptive pass-rate switch.

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

const MIB: u64 = 1024 * 1024;

/// `krishiv.runtimeFilter.enabled`.
pub static RUNTIME_FILTER_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Plan runtime filters: the build side of an inner or left semi hash join \
          publishes the min/max, a small value set or a Bloom filter of its keys to the scans \
          feeding the probe side, which skip splits, row groups and rows no build key can match. \
          Results are the same either way.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
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
    settable_by: SettableBy::User,
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
    settable_by: SettableBy::Admin,
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
    settable_by: SettableBy::Admin,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.bloom.maxItems`.
pub static RUNTIME_FILTER_BLOOM_MAX_ITEMS: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.maxItems",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(4_000_000)),
    doc: "Most build keys a runtime Bloom filter is built for: a build estimated larger plans \
          none, and one that turns out larger publishes none. Left at its default, it is raised to \
          20,000,000 for a scan estimated at least 20 times its build \
          (`krishiv.runtimeFilter.bloom.atScale`), and for a build past that under a scan 8 times it to the keys \
          the memory tier holds (`krishiv.runtimeFilter.bloom.memoryTier`).",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.1.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.bloom.maxBytes`.
pub static RUNTIME_FILTER_BLOOM_MAX_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.maxBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(8 * MIB)),
    doc: "Largest runtime Bloom filter. A filter capped below its 1% false-positive size is still \
          built unless its expected false-positive rate exceeds 5%. Left at its default, it is raised to 32 MiB \
          for a scan estimated at least 20 times its build \
          (`krishiv.runtimeFilter.bloom.atScale`), and for a build past 20,000,000 keys under a scan 8 times it \
          to 1/64 of the query memory limit (`krishiv.runtimeFilter.bloom.memoryTier`).",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
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
    settable_by: SettableBy::User,
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
    settable_by: SettableBy::User,
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
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};
