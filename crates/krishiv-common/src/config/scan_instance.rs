// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.scan.instanceProjection` entry (kept out of `registry.rs`, which is at its size cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Each instance of a table read several times reads only the columns its parent uses
/// (study 38, lever A).
pub static EXEC_SCAN_INSTANCE_PROJECTION: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.scan.instanceProjection",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a table the query reads more than once is one scan in the logical plan, \
          whose columns are the union of what every reader needs. With this on, each \
          instance under a projection, an aggregate or a join (through filters and aliases) \
          reads only the columns that parent uses. Results are the same either way, except \
          which CSV/JSON rows count as malformed (on follows Spark, which parses each \
          instance's own columns); false reads the union in every instance.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};
