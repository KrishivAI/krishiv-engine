// SPDX-License-Identifier: Apache-2.0
//! The config entries of the web UI's embeddable views: the framing origins,
//! the attribution mark and the signed link's life (P3-T071).

use std::time::Duration;

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

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

/// The origins that may frame the web UI's embeddable views (P3-T071).
pub static UI_ALLOWED_ORIGINS: ConfigEntry = ConfigEntry {
    key: "krishiv.ui.allowedOrigins",
    value_type: ValueType::String,
    default: None,
    doc: "Comma-separated origins (`https://console.example.com`, scheme and host, an optional \
          port) that may put the UI's embeddable views (`?embed=1`) in a frame: they become the \
          `frame-ancestors` of those pages, and the same origins may send the embed theme with \
          `postMessage`. The page reads the list from `GET /api/internal/embed/config`, which needs no \
          token (the embedded view has no session), so the origins are not secret. Unset, the default, is `'none'`: no page can be framed. Every other \
          page of the UI is never framable. An entry that is not an origin (a path, a wildcard, \
          a quote or a space inside it) makes the server refuse to start.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// Whether an embedded UI view drops the "Powered by Krishiv" mark (P3-T071).
pub static UI_EMBED_HIDE_ATTRIBUTION: ConfigEntry = ConfigEntry {
    key: "krishiv.ui.embed.hideAttribution",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, an embedded view (`?embed=1`) does not show the small \"Powered by Krishiv\" \
          mark. The embedded view never shows the Krishiv header either way.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// The longest life of a signed embed link (P3-T071).
pub static UI_EMBED_LINK_TTL: ConfigEntry = ConfigEntry {
    key: "krishiv.ui.embed.linkTtl",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(300))),
    doc: "How long a signed embed link stays valid, at most. A link grants read-only access to \
          one query's profile and nothing else; a request for a longer life gets this one. The \
          signing key is made at start-up, so every link also ends when the server restarts.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};
