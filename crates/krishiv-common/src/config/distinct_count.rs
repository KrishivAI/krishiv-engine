// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.agg.distinctCountAtMerge` and `krishiv.exec.agg.distinctRouted` entries (kept
//! out of `registry.rs`, which is at its size cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `count(DISTINCT v)` inside a `GROUP BY` counts each distinct pair where the
/// pairs' merge finds it, instead of a second hash aggregation over the pairs
/// (design `P3-cb-distinct-in-group`).
pub static EXEC_AGG_DISTINCT_COUNT_AT_MERGE: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.agg.distinctCountAtMerge",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a grouped aggregation whose only DISTINCT aggregate is a count, beside at \
          least one other aggregate, on more than one executor slot, runs one table of distinct \
          (key, value) pairs with no payload and one table of the plain aggregates by key, and \
          counts each pair where the pairs' tables merge, instead of a second hash aggregation \
          over the pairs. Results are the same either way; false keeps the two aggregations in \
          series.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};

/// A `count(DISTINCT v)` the planner marked as having many group keys (a
/// sample saw at least 1,024 distinct values) routes its pairs by key and is
/// counted at the merge even alone (design P3-A 1d).
pub static EXEC_AGG_DISTINCT_ROUTED: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.agg.distinctRouted",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a grouped aggregation whose only DISTINCT aggregate is a count, and whose \
          group keys the planner knows to hold at least 1,024 distinct values (a plan-time sample \
          saw them), puts all the (key, value) pairs of one key in one radix partition of the \
          pairs table and counts them where the pairs' tables merge, also when the count is the \
          only aggregate. Needs krishiv.exec.agg.distinctCountAtMerge. Results are the same \
          either way; false keeps the plan without routing.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

#[cfg(test)]
mod tests {
    use super::super::Config;
    use super::*;

    #[test]
    fn distinct_routed_defaults_and_parses_booleans() {
        let mut c = Config::new();
        let get = |c: &Config| {
            c.get(&EXEC_AGG_DISTINCT_ROUTED)
                .and_then(ConfigValue::as_bool)
        };
        assert_eq!(get(&c), Some(true));
        c.set("krishiv.exec.agg.distinctRouted", "false").unwrap();
        assert_eq!(get(&c), Some(false));
        assert!(
            c.set("krishiv.exec.agg.distinctRouted", "sometimes")
                .is_err()
        );
    }
}
