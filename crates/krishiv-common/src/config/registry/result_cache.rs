// SPDX-License-Identifier: Apache-2.0
//! The config entries of the process-wide result cache.

use std::time::Duration;

use super::MIB;
use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Whether the process-wide result cache serves and stores query results.
pub static RESULT_CACHE_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.resultCache.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Keep the rows of a query for later runs of the same query by any session of the process \
          with the same principal and access-policy version. Only queries over permanent \
          catalog tables with no non-deterministic function are cached. Entries are dropped by \
          `REFRESH TABLE`, `CLEAR CACHE` and every write the engine commits to a table they \
          read; a change made outside the engine is not seen until the entry expires. Off by \
          default.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: None,
};

/// How long a result-cache entry is served.
pub static RESULT_CACHE_TTL: ConfigEntry = ConfigEntry {
    key: "krishiv.resultCache.ttl",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(24 * 60 * 60))),
    doc: "How long the result cache serves an entry, counted from when it was stored; an older \
          entry is dropped before the next lookup and the query runs again. `0s` never \
          expires. Needs `krishiv.resultCache.enabled`.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.1.0",
    spark_alias: None,
};

/// Byte cap of the process-wide result cache.
pub static RESULT_CACHE_MAX_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.resultCache.maxBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 * MIB)),
    doc: "Most bytes the result cache of an engine runtime holds (the process-wide store of \
          sessions built alone, or a supplied runtime's own), as Arrow batches outside the query \
          memory pool; fixed when the runtime is built. A result \
          larger than the cap is not stored; when a new result would pass the cap the oldest \
          entries go first, and entries past their ttl go whenever a result is stored. Needs `krishiv.resultCache.enabled`.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.1.0",
    spark_alias: None,
};
