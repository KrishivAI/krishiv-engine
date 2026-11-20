// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.advisor.*` entries (P3-T073): the thresholds of the advisor hints.
//!
//! Design note, the defaults and why. Every rule is a threshold on a measured ratio plus a floor, so a
//! trivial query raises nothing: SPILL, spilled bytes over 10% of the query's peak memory and at least 64 MiB
//! (a smaller spill costs less than the advice); MEMORY_WAIT, blocked time over 1x compute time and at least
//! 100 ms; SKEW, the slowest driver at least 4x the mean of the others, over 200 ms (or 100,000 rows); MISSING_STATISTICS, a
//! q-error of at least 4 over at least 10,000 rows on a table without statistics; RUNTIME_FILTER_OPPORTUNITY,
//! at least 50% of at least 1,000,000 probe rows dropped by a join with no filter. The floors were tuned against
//! the hints the 22 TPC-H SF1 queries raise (`tpch_sf1_hint_review`
//! in `krishiv-session/tests/advisor_hints.rs` prints the list). `maxHints` (32) bounds the stored row.

use std::time::Duration;

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `krishiv.advisor.enabled`.
pub static ADVISOR_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Compute advisor hints (spill, missing statistics, skew, runtime-filter opportunity, memory wait) for every finished query. They appear in `EXPLAIN ANALYZE`, in `system.queries.hints` and in the stored profile. Off: none is computed and the `hints` column is NULL. Hints never change a plan or a result.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.spill.minPercent`.
pub static ADVISOR_SPILL_MIN_PERCENT: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.spill.minPercent",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(10)),
    doc: "An operator that spilled raises a SPILL hint when the bytes it wrote to disk are more than this percentage of the query's peak reserved memory (and at least `krishiv.advisor.spill.minBytes`).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.spill.minBytes`.
pub static ADVISOR_SPILL_MIN_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.spill.minBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 << 20)),
    doc: "Floor on the bytes an operator wrote to disk before a SPILL hint is raised.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.memoryWait.factor`.
pub static ADVISOR_MEMORY_WAIT_FACTOR: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.memoryWait.factor",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(1)),
    doc: "An operator raises a MEMORY_WAIT hint when its time blocked waiting for memory exceeds this multiple of its compute time (and `krishiv.advisor.memoryWait.minTime`).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.memoryWait.minTime`.
pub static ADVISOR_MEMORY_WAIT_MIN_TIME: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.memoryWait.minTime",
    value_type: ValueType::Millis,
    default: Some(ConfigValue::Duration(Duration::from_millis(100))),
    doc: "Floor on the time an operator waited for memory before a MEMORY_WAIT hint is raised.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.skew.factor`.
pub static ADVISOR_SKEW_FACTOR: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.skew.factor",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(4)),
    doc: "A key-partitioned operator raises a SKEW hint when its slowest driver did at least this multiple of the mean work of the other drivers (time, or rows when the row floor holds); at factor 4 that is 80% of the work at 2 drivers, 67% at 3 and 57% at 4.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.skew.minTime`.
pub static ADVISOR_SKEW_MIN_TIME: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.skew.minTime",
    value_type: ValueType::Millis,
    default: Some(ConfigValue::Duration(Duration::from_millis(200))),
    doc: "Floor on the slowest driver's compute time before a SKEW hint is raised.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.skew.minRows`.
pub static ADVISOR_SKEW_MIN_ROWS: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.skew.minRows",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(100_000)),
    doc: "Floor on an operator's output rows for the row-based SKEW test.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.missingStats.minQError`.
pub static ADVISOR_MISSING_STATS_MIN_Q_ERROR: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.missingStats.minQError",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(4)),
    doc: "A table without statistics raises a MISSING_STATISTICS hint when the planner's row estimate for a join or aggregate input missed the actual rows by at least this factor (q-error).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.missingStats.minRows`.
pub static ADVISOR_MISSING_STATS_MIN_ROWS: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.missingStats.minRows",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(10_000)),
    doc: "Floor on the larger of the estimated and the actual rows before a MISSING_STATISTICS hint is raised.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.runtimeFilter.minProbeRows`.
pub static ADVISOR_RUNTIME_FILTER_MIN_PROBE_ROWS: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.runtimeFilter.minProbeRows",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(1_000_000)),
    doc: "Floor on a hash join's probe rows before a RUNTIME_FILTER_OPPORTUNITY hint is raised.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.runtimeFilter.minDropPercent`.
pub static ADVISOR_RUNTIME_FILTER_MIN_DROP_PERCENT: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.runtimeFilter.minDropPercent",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(50)),
    doc: "A hash join with an unfiltered key raises a RUNTIME_FILTER_OPPORTUNITY hint when it dropped at least this percentage of its probe rows (a lower bound: probe rows in minus rows out).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.advisor.maxHints`.
pub static ADVISOR_MAX_HINTS: ConfigEntry = ConfigEntry {
    key: "krishiv.advisor.maxHints",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(32)),
    doc: "Most hints kept per query, in order; bounds the size of the `system.queries` row.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};
