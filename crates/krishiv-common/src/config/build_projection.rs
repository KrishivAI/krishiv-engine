// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.join.buildProjection` entry (kept out of `registry.rs`, which is at its size cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Hash joins compute a build-side expression before building
/// (design S4', `P2b-S4p-q09-amount-below-join.md`).
pub static EXEC_JOIN_BUILD_PROJECTION: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.buildProjection",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: an expression above an inner hash join that reads only the build \
          input's columns, cannot raise and is narrower than the columns it frees is \
          computed on the build input, so the hash table holds its result instead of \
          those columns. Results are the same either way; false computes it above the join.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};
