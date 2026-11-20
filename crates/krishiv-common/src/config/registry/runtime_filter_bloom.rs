// SPDX-License-Identifier: Apache-2.0
//! The config entries of the at-scale runtime Bloom tiers (gap ledger §4
//! item 13, study 36 §4.4): `atScale` and the memory tier; and of runtime
//! filter targets through join key equalities (`equatedSites`).

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

/// `krishiv.runtimeFilter.equatedSites`: runtime filters also reach scans through inner-join key equalities.
pub static RUNTIME_FILTER_EQUATED_SITES: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.equatedSites",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "A runtime filter on a join key also targets the scans of the columns an inner equi-join below \
          equates with it (a key pair of two plain columns of one type), as on the key's own scan: for \
          `a.k = b.k` below, a filter on `b.k` also filters `a.k` at `a`'s scan. Results do not change.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.0",
    spark_alias: None,
};

#[cfg(test)]
mod tests {
    use super::RUNTIME_FILTER_EQUATED_SITES;
    use crate::config::{Config, ConfigValue};

    #[test]
    fn equated_sites_default_on_and_parse() {
        let mut c = Config::new();
        let get = |c: &Config| {
            c.get(&RUNTIME_FILTER_EQUATED_SITES)
                .and_then(ConfigValue::as_bool)
        };
        assert_eq!(get(&c), Some(true));
        c.set("krishiv.runtimeFilter.equatedSites", "false")
            .unwrap();
        assert_eq!(get(&c), Some(false));
        assert!(
            c.set("krishiv.runtimeFilter.equatedSites", "often")
                .is_err()
        );
    }
}
