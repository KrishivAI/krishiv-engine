// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.sql.legacy.skipTypeValidationOnAlterPartition` entry (kept out of `registry.rs`, which is at its
//! size cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Skips casting the partition values of `ALTER TABLE .. ADD | DROP PARTITION` to the partition column types.
pub static LEGACY_SKIP_TYPE_VALIDATION_ON_ALTER_PARTITION: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.skipTypeValidationOnAlterPartition",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, the partition values of `ALTER TABLE .. ADD PARTITION`, `DROP PARTITION` and \
          `RENAME PARTITION` are kept as written instead of being cast to the partition column types \
          (`'007'` stays `007`, `'abc'` for an INT column is accepted).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: Some("spark.sql.legacy.skipTypeValidationOnAlterPartition"),
};
