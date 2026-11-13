// SPDX-License-Identifier: Apache-2.0
//! The config entries of the subquery shared-scan rules (P3 W1).

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Whether a correlated aggregate subquery over the outer block's own join runs as a window.
pub static SUBQUERY_AS_WINDOW: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.subqueryAsWindow",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Compute a correlated scalar aggregate subquery over the same relations and filters as \
          the outer block (TPC-H q17, q02) as a window over the outer join's rows, so the big \
          table is read once. Only `min`, `max` and exact decimal `avg` of non-floating, \
          non-collated keys qualify, and only when statistics say the rewrite pays. Results are \
          the same either way; `false` turns it off.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Whether a subtree the main block and a subquery both read is computed once.
pub static SHARE_REPEATED_SUBTREES: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.shareRepeatedSubtrees",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Compute an uncorrelated subtree once when the main block reads it directly and a \
          subquery reads it again (TPC-H q11), and let both read the stored result. Needs \
          `krishiv.sql.optimizer.shareCteInstances`. Results are the same either way; `false` turns it off.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};

/// A dimension's payload joins above the semi and anti joins that never read it
/// (`krishiv.optimizer.latePayloadJoin`).
pub static LATE_PAYLOAD_JOIN: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.latePayloadJoin",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Join a dimension's payload columns (such as a name read only by the aggregate on top) \
          after the semi, anti and outer joins that never read them, and leave a semi join on the \
          dimension's keys where the join was, so those joins build and probe narrower rows. Applies \
          when statistics show the payload is wide and the dimension is the smaller side. Results \
          are the same either way.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};
