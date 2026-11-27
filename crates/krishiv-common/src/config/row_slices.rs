// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.scan.rowSlices` entry (kept out of `registry.rs`, which is at its size cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// A small file scan in fewer row groups than drivers cuts them into row
/// slices (study 59).
pub static EXEC_SCAN_ROW_SLICES: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.scan.rowSlices",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a small file scan whose kept units (Parquet row groups) are fewer than \
          its drivers cuts each unit into row slices of at least 4,096 rows, one morsel \
          each, so the operators above it run on every driver. Each slice decodes the rows \
          of its unit before it when the file has no offset index. Results are the same \
          either way; false reads each unit on one driver.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.1",
    spark_alias: None,
};
