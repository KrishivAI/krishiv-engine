// SPDX-License-Identifier: Apache-2.0
//! The config entries of the at-scale runtime Bloom tiers (gap ledger §4
//! item 13, study 36 §4.4): `atScale` and the memory tier.

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `krishiv.runtimeFilter.bloom.atScale`: larger runtime Bloom filters for scans far larger than the build.
pub static RUNTIME_FILTER_BLOOM_AT_SCALE: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.bloom.atScale",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "A join whose probe-side scan is estimated at least 20 times its build plans its runtime Bloom filter with \
          caps of 20,000,000 keys and 32 MiB instead of `krishiv.runtimeFilter.bloom.maxItems` and \
          `krishiv.runtimeFilter.bloom.maxBytes`, when those are lower (larger builds can get more under \
          `krishiv.runtimeFilter.bloom.memoryTier`). A cap set explicitly still binds unless this key is also set \
          explicitly to true. Results do not change.",
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
    doc: "A join whose build is estimated past the 20,000,000 keys of `krishiv.runtimeFilter.bloom.atScale`, \
          under a probe-side scan estimated at least 8 times it, plans its runtime Bloom filter with a size cap \
          of 1/64 of the query memory limit (`krishiv.memory.limit`, or `krishiv.session.memoryLimit` \
          when lower) and a key cap of the keys that size holds at the filter's false-positive rate. It applies \
          only while `krishiv.runtimeFilter.bloom.atScale` is on. Results do not change.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.0",
    spark_alias: None,
};
