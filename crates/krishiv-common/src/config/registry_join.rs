// SPDX-License-Identifier: Apache-2.0
//! Hash join entries of the configuration registry (`registry.rs`).

use std::time::Duration;

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Hash join keys in the row layout with a size-adaptive index (P2-T158).
pub static EXEC_JOIN_ROW_LAYOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.rowLayout",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: hash joins store their keys as fixed-width rows and index them with a \
          table that switches by size between a flat and a salted directory. Results are \
          the same either way; joins whose key types have no row encoding keep the \
          columnar path.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Probe a tiny build with one integer key through an array indexed by the key.
pub static EXEC_JOIN_DIRECT_INDEX: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.directIndex",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a hash join build of at most 4,096 rows with one integer key whose values \
          span at most four slots a row is probed through an array indexed by the key, without \
          encoding or hashing the probe keys (dimension joins such as TPC-DS q62's). Results \
          are the same either way; `false` keeps the hashed probe.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.1",
    spark_alias: None,
};

/// Fuse a filter and a scan's row stage into the hash probe above them
/// (design P2-E3 U2).
pub static EXEC_JOIN_FUSE_PROBE_SELECT: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.fuseProbeSelect",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a filter directly below a hash join's probe side, and a scan's runtime \
          filters there, run inside the probe, which then joins the rows they keep without \
          copying them first. Results are the same either way; turn it off to plan the \
          filter as an operator of its own and keep the runtime filters in the scan.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};

/// How long a finished hash join build waits for its probe before the
/// memory arbitrator spills it ahead of running consumers (study 30, M2).
pub static EXEC_JOIN_IDLE_SPILL_AFTER: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.join.idleSpillAfter",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(1))),
    doc: "How long a finished hash join build table may wait for its probe side to start \
          before it counts as idle. When memory runs short, an idle table is spilled first, \
          one partition at a time, before an operator that is running spills its own state; \
          the probe reads the spilled partitions back from disk. A shorter wait keeps less \
          memory idle; a longer one spills less often when the probe starts soon.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Small batches joined before a hash join's build, an aggregation and the
/// operators of their pipelines (Polars item 2, J1).
pub static EXEC_COALESCE_INPUT: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.coalesceInput",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: in a pipeline that ends in a hash join's build or an aggregation, the \
          driver joins small batches (fewer than a quarter of `krishiv.exec.batchSize` rows, and \
          of 4,096, as a selective filter or join leaves them) into batches of about \
          `krishiv.exec.batchSize` \
          rows before the build, the aggregation and a hash join's probe, so \
          each pays its per-batch cost once per batch of that size. Rows keep their order, \
          and only rows of one scan split are joined. The rows held are not charged to the \
          query's memory, like any batch between two operators: each driver holds at most \
          about one batch, and 1 MiB (or a 64th of the memory limit, if less), \
          for each operator that joins its input. Results are the same either way.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};
