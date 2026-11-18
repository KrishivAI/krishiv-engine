// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.history.*` and `krishiv.ui.*` entries, re-exported by `registry.rs` (moved out to keep it under 2,000 lines).

use std::borrow::Cow;

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Where finished queries' profiles are kept (P3-T051).
pub static HISTORY_PATH: ConfigEntry = ConfigEntry {
    key: "krishiv.history.path",
    value_type: ValueType::String,
    default: None,
    doc: "A directory where the engine keeps one compact profile for every finished query, as \
          Parquet files, so `system.queries` history survives a restart and a stored profile can \
          be opened later. A profile has the plan, the operators' counters and the query's \
          identity, and no result data; literals are masked unless `krishiv.history.literals` \
          is `keep`. Every query that reached execution is stored, as the engine metrics count \
          them: a statement refused in parsing, analysis or an access check is not. Writing happens after the result stream \
          ends and never fails the query; a failed write is counted in \
          `krishiv_history_write_failures`. When unset, nothing is stored.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// Whether the persisted history keeps literals (P3-T051).
pub static HISTORY_LITERALS: ConfigEntry = ConfigEntry {
    key: "krishiv.history.literals",
    value_type: ValueType::Enum(&["keep", "mask"]),
    default: Some(ConfigValue::String(Cow::Borrowed("mask"))),
    doc: "`mask` (the default) replaces every string and number literal with `?` in the \
          statement text and in the plan lines of the profiles stored under \
          `krishiv.history.path`, so a stored profile carries no data values. `keep` stores \
          them as `EXPLAIN ANALYZE` prints them. Credentials in a statement are redacted either \
          way.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// Most bytes the profile history keeps (P3-T051).
pub static HISTORY_MAX_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.history.maxBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(256 << 20)),
    doc: "Most bytes the profile history under `krishiv.history.path` keeps. When a new file \
          takes the store past this, the oldest files go first, so the newest queries stay.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// The names the web UI answers to (P3-T050, threat model T-W5, DNS rebinding).
pub static UI_ALLOWED_HOSTS: ConfigEntry = ConfigEntry {
    key: "krishiv.ui.allowedHosts",
    value_type: ValueType::String,
    default: None,
    doc: "Comma-separated host names (no ports; an IPv6 address in brackets) the web UI's \
          `/api/internal` routes answer to, by the request's `Host` header, so a page on \
          another name that resolves to this server cannot read it (DNS rebinding). A server \
          bound to a loopback address always also answers to `localhost`, `127.0.0.1` and \
          `[::1]`. A server bound to another address checks the `Host` header only when this is \
          set (it already requires authentication, and TLS for bearer methods). The \
          environment variable `KRISHIV_UI_ALLOWED_HOSTS` adds names to the list.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// Whether `krishiv server` serves the web UI backend (design T133).
pub static UI_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.ui.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "When false, `krishiv server` refuses to start: there is no listener and no UI \
          backend, and nothing is recorded beyond the `system` tables. The UI reads those \
          tables and keeps no state of its own.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};
