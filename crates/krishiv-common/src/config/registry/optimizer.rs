// SPDX-License-Identifier: Apache-2.0
//! The config entries of the subquery shared-scan rules (P3 W1) and of the group-key code (P2b D2).

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

/// Whether a stack of semi/anti joins over one input runs its most selective join first.
pub static SEMI_ANTI_JOIN_ORDER: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.semiAntiJoinOrder",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Run the `LEFT SEMI` and `LEFT ANTI` joins stacked over one input (`EXISTS` and \
          `NOT EXISTS` subqueries of one block, TPC-H q21) in the order of their estimated \
          survivors, fewest first, instead of the order of the text, when their conditions \
          cannot raise, and only when every input is estimated from statistics. Results are the same either way; `false` turns it off.",
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

/// Whether an aggregate's group keys from one dimension travel as one integer code (P2b D2).
pub static GROUP_KEY_CODE: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.groupKeyCode",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Internal. Let an aggregate whose group keys include several columns of one dimension \
          table (TPC-H q16's `p_brand, p_type, p_size`) group by one integer code made on the \
          dimension's side, and read the columns back from the dimension's distinct values above \
          the aggregate, so the joins below it carry the code instead of the columns. Applies only \
          when statistics say it pays. Results are the same either way. Off until its timed \
          A/B passes.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Experimental: the group-key code applies wherever the plan's shape qualifies, without the statistics test
/// (a measurement switch for P2b D2's timed A/B).
pub static EXPERIMENTAL_GROUP_KEY_CODE_FORCE: ConfigEntry = ConfigEntry {
    key: "krishiv.experimental.groupKeyCodeForce",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Experimental: apply `krishiv.optimizer.groupKeyCode` wherever an aggregate's shape qualifies, \
          also when a key's distinct count is a guess or the estimated ratios say it does not pay. \
          For measuring the rewrite; results are the same either way.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};
