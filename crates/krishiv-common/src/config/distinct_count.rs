// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.agg.distinctCountAtMerge` entry (kept out of `registry.rs`, which is at its
//! size cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `count(DISTINCT v)` inside a `GROUP BY` counts each distinct pair where the
/// pairs' merge finds it, instead of a second hash aggregation over the pairs
/// (design `P3-cb-distinct-in-group`).
pub static EXEC_AGG_DISTINCT_COUNT_AT_MERGE: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.agg.distinctCountAtMerge",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a grouped aggregation whose only DISTINCT aggregate is a count runs one \
          table of distinct (key, value) pairs with no payload and one table of the plain \
          aggregates by key, and counts each pair where the pairs' tables merge, instead of a \
          second hash aggregation over the pairs. Results are the same either way; false keeps \
          the two aggregations in series.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};
