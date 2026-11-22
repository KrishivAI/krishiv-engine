// SPDX-License-Identifier: Apache-2.0
//! The config entries of the subquery shared-scan rules (P3 W1), the group-key code (P2b D2) and the
//! payload joins (P2b S5, A5).

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
          cannot raise, and only when every input is estimated from statistics. Results are the \
          same either way; `false` turns it off.",
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

/// A table scan read only by filters, two or more, is read once under the
/// disjunction of their predicates (`krishiv.optimizer.shareFilteredScans`).
pub static SHARE_FILTERED_SCANS: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.shareFilteredScans",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Read a table once when two or more filters are its only readers (TPC-DS q28): the rows \
          any of the filters keeps are stored, and each filter reads them. Only when statistics \
          say the filters keep at most a quarter of the table, and no predicate can raise. The stored \
          rows are charged to the query's memory and spill to disk under pressure. Needs \
          `krishiv.sql.optimizer.shareCteInstances`. Results are the same either way; `false` turns it off.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// A group key computed from other group keys moves above the aggregate
/// (`krishiv.optimizer.derivedKeysAbove`, P3-A part 2c).
pub static DERIVED_KEYS_ABOVE: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.derivedKeysAbove",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal. Let an aggregate that groups by a column and by expressions of it only \
          (ClickBench q36's `ClientIP, ClientIP - 1, ClientIP - 2`) group by the column alone and \
          compute the expressions from it above the aggregate, so the hash table holds one narrow \
          key instead of several. Applies only to deterministic built-in expressions that cannot \
          raise (or whose range statistics prove no overflow) over keys whose equality is identity. \
          Results are the same either way. On since its timed A/B (ClickBench q36: 0.82x).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
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

/// Whether a dimension's payload columns join above the aggregate that groups by them (P2b A5,
/// `krishiv.optimizer.payloadAboveAggregate`).
pub static PAYLOAD_ABOVE_AGGREGATE: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.payloadAboveAggregate",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal. Let an aggregate that groups by a dimension's key and its other columns \
          (TPC-H q10's customer and nation columns) group by the joining key of the other side \
          instead, and join the dimension above the aggregate, so the aggregate and the joins \
          below it hold narrow rows. Applies only where the dimension's key is proven unique \
          (the plan groups it, the table's statistics declare it, or a key count proved it: \
          `krishiv.optimizer.discoverKeys`), no aggregate argument can \
          raise or overflow, and statistics say the join keeps most rows. Results are the same \
          either way. On by default (TPC-H q10: 0.75x at SF100, 0.59x at SF1000).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Whether the session counts a table's key columns to prove them unique (P2b A5 unit 2,
/// `krishiv.optimizer.discoverKeys`).
pub static DISCOVER_KEYS: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.discoverKeys",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal. Let the session prove a file table's key columns unique by reading them once \
          before the query runs, when `krishiv.optimizer.payloadAboveAggregate` needs that fact \
          for a join over plain files (TPC-H q10's `customer`). The answer is cached per file \
          version: a later query over the same files reads nothing. A count that runs out of \
          time or memory, or fails, proves nothing and the plan stays as it was. Results are the \
          same either way. On by default; set false to never read data before planning.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Time one query's key counts may take (`krishiv.optimizer.discoverKeys.timeout`).
pub static DISCOVER_KEYS_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.discoverKeys.timeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(std::time::Duration::from_secs(60))),
    doc: "The time one query's key counts (`krishiv.optimizer.discoverKeys`) may take together: \
          its tables are counted in turn, each with what the ones before it left. A count that \
          runs out of time proves nothing; it is tried again only by a query given more time.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
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
