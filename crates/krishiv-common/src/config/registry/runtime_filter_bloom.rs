// SPDX-License-Identifier: Apache-2.0
//! The config entries of the at-scale runtime Bloom tiers (gap ledger §4
//! item 13, study 36 §4.4): `atScale`, its ratio and the memory tier.

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `krishiv.runtimeFilter.bloom.atScale`: larger runtime Bloom filters for scans far larger than the build.
pub static RUNTIME_FILTER_BLOOM_AT_SCALE: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.atScale",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "A join whose probe-side scan is estimated at least `krishiv.runtimeFilter.bloom.atScaleRatio` times its \
          build plans its runtime Bloom filter with caps of 20,000,000 keys and 32 MiB instead of \
          `krishiv.runtimeFilter.bloom.maxItems` and `krishiv.runtimeFilter.bloom.maxBytes`, when those are lower \
          (and larger ones under `krishiv.runtimeFilter.bloom.memoryTier`). A cap set explicitly still binds unless \
          this key is also set explicitly to true. Results do not change.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.bloom.atScaleRatio`: the probe-to-build ratio from which `bloom.atScale` applies.
pub static RUNTIME_FILTER_BLOOM_AT_SCALE_RATIO: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.atScaleRatio",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(8)),
    doc: "How many times larger than a join's build its probe-side scan must be estimated for \
          `krishiv.runtimeFilter.bloom.atScale` (and `krishiv.runtimeFilter.bloom.memoryTier`) to raise the \
          runtime Bloom filter's caps. Results do not change.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.bloom.memoryTier`: Bloom caps from the memory limit for scans far larger than the build.
pub static RUNTIME_FILTER_BLOOM_MEMORY_TIER: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.memoryTier",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Where `krishiv.runtimeFilter.bloom.atScale` raises a join's runtime Bloom filter caps, raise the size cap \
          further to 1/64 of the query memory limit (`krishiv.memory.limit`, or `krishiv.session.memoryLimit` \
          when lower), and the key cap to the keys that size holds at the filter's false-positive rate. Results do \
          not change.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.0",
    spark_alias: None,
};
