// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.agg.directKeys` entry (kept out of `registry.rs`, which is at its size cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Grouped aggregations address the groups of a batch with single-byte keys
/// directly (study 32 D1).
pub static EXEC_AGG_DIRECT_KEYS: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.agg.directKeys",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a grouped aggregation whose key columns in a batch are all one-byte \
          strings or small integers without NULLs finds each row's group through a table \
          indexed by the packed key, and hashes only each new key's first row. Results are \
          the same either way; false hashes every row.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};
