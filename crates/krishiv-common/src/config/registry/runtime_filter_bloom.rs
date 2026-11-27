// SPDX-License-Identifier: Apache-2.0
//! The config entries of the at-scale runtime Bloom tiers (gap ledger §4
//! item 13, study 36 §4.4): `atScale` and the memory tier; and of runtime
//! filter targets through join key equalities (`equatedSites`); residual bounds; the small-table wait.

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

/// `krishiv.runtimeFilter.residualBounds` (design P3-0.3.1 D2).
pub static RUNTIME_FILTER_RESIDUAL_BOUNDS: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.residualBounds",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "An inner, semi or right hash join whose condition also compares a probe-side scan column with \
          a build column (`p < b`, `p <= b`, `p > b`, `p >= b`, numeric, date or timestamp) publishes the \
          build's largest (or smallest) value of that column as a runtime filter: the scan then drops the \
          rows no build row can match (`p <= max(b)`) before the join. Results do not change.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.1",
    spark_alias: None,
};

/// `krishiv.runtimeFilter.smallTableWait` (study 59, lever W).
pub static RUNTIME_FILTER_SMALL_TABLE_WAIT: ConfigEntry = ConfigEntry {
    key: "krishiv.runtimeFilter.smallTableWait",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "A scan of a table estimated at 1,000,000 rows or fewer starts after the join build whose runtime \
          filter it receives when that build only reads the same table and filters it to at most a tenth \
          of the table's estimated rows (a dimension filtered twice, as in TPC-DS q02): the wait is one \
          scan of a small table. Off, such a scan waits only for a build at least 1,000 times smaller, and otherwise \
          applies the filter only to the rows it reads after the build ends. Results do not change.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.1",
    spark_alias: None,
};

#[cfg(test)]
mod tests {
    use super::{
        RUNTIME_FILTER_EQUATED_SITES, RUNTIME_FILTER_RESIDUAL_BOUNDS,
        RUNTIME_FILTER_SMALL_TABLE_WAIT,
    };
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

    #[test]
    fn residual_bounds_default_on_and_parse() {
        let mut c = Config::new();
        let get = |c: &Config| {
            c.get(&RUNTIME_FILTER_RESIDUAL_BOUNDS)
                .and_then(ConfigValue::as_bool)
        };
        assert_eq!(get(&c), Some(true));
        c.set("krishiv.runtimeFilter.residualBounds", "false")
            .unwrap();
        assert_eq!(get(&c), Some(false));
        assert!(
            c.set("krishiv.runtimeFilter.residualBounds", "often")
                .is_err()
        );
    }

    #[test]
    fn small_table_wait_default_on_and_parse() {
        let mut c = Config::new();
        let get = |c: &Config| {
            c.get(&RUNTIME_FILTER_SMALL_TABLE_WAIT)
                .and_then(ConfigValue::as_bool)
        };
        assert_eq!(get(&c), Some(true));
        c.set("krishiv.runtimeFilter.smallTableWait", "false")
            .unwrap();
        assert_eq!(get(&c), Some(false));
    }
}
