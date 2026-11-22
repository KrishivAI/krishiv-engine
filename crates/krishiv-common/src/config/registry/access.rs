// SPDX-License-Identifier: Apache-2.0
//! The config entries of server-mode access control and egress: the access-policy provider
//! (`krishiv.access.provider`) and the outbound-address policy (`krishiv.io.egress.*`). All are
//! fixed when the server starts.

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `krishiv.access.provider`.
pub static ACCESS_PROVIDER: ConfigEntry = ConfigEntry {
    key: "krishiv.access.provider",
    value_type: ValueType::String,
    default: None,
    doc: "Server mode: the access-policy provider that decides data securables (tables, views, \
          columns, namespaces, functions, paths, secrets, connections). `grants`: \
          `GRANT`/`REVOKE` decide them, and the role matrix keeps deciding the server's own \
          objects; an admin passes every data check. `roles`: the role matrix decides \
          everything. Unset: `grants` when authentication is configured, otherwise `roles`. \
          `external`: the policy service of `krishiv.access.external.endpoint` decides them \
          (`krishiv.access.v1`, ADR 0049) and the role matrix keeps only the server's own objects, \
          so a viewer may write when the service allows it; `roles+external` keeps the role \
          matrix on data too. A `+`-separated list (`grants+external`) is an \
          intersection: every listed provider that decides a securable must allow it, and one no \
          listed provider decides is refused. `catalog` is refused until it is available. Path \
          confinement (`krishiv.io.local.allowedRoots`, `krishiv.io.objectStore.allowedPrefixes`) \
          applies first under every provider.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// `krishiv.access.external.endpoint` (ADR 0049 §3).
pub static ACCESS_EXTERNAL_ENDPOINT: ConfigEntry = ConfigEntry {
    key: "krishiv.access.external.endpoint",
    value_type: ValueType::String,
    default: None,
    doc: "Server mode, provider `external`: the `krishiv.access.v1` policy service. An \
          `https://` URI, a Unix socket `unix:/path`, or `http://` to a loopback address only. \
          Required when `krishiv.access.provider` lists `external`.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.access.external.timeout` (ADR 0049 §3).
pub static ACCESS_EXTERNAL_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.access.external.timeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(std::time::Duration::from_millis(200))),
    doc: "Server mode, provider `external`: the deadline of one call to the policy service. \
          Past it, every pair the call carried is refused (fail closed).",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.access.external.token` (ADR 0049 §3).
pub static ACCESS_EXTERNAL_TOKEN: ConfigEntry = ConfigEntry {
    key: "krishiv.access.external.token",
    value_type: ValueType::String,
    default: None,
    doc: "Server mode, provider `external`: the bearer token sent to the policy service, as a \
          secret reference (`env:NAME` or `file:/path`, as `krishiv.auth.tokens.*`), never a \
          literal. Unset: no `authorization` header.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.access.external.securables` (ADR 0049 §3).
pub static ACCESS_EXTERNAL_SECURABLES: ConfigEntry = ConfigEntry {
    key: "krishiv.access.external.securables",
    value_type: ValueType::String,
    default: Some(ConfigValue::String(std::borrow::Cow::Borrowed("data"))),
    doc: "Server mode, provider `external`: what the policy service decides, a comma-separated \
          list. `data`: tables, views, columns, namespaces, functions, paths, secrets and \
          connections. `system` (the server's own objects, `System(..)`) is refused until the \
          admin API and UI guards ask the service too; the role matrix decides them.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.io.egress.privateAddresses` (design P3-T067 §5.1).
pub static IO_EGRESS_PRIVATE_ADDRESSES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.egress.privateAddresses",
    value_type: ValueType::Enum(&["allow", "deny"]),
    default: None,
    doc: "Whether the server's outbound connections for catalogs and data (catalog APIs, token \
          endpoints, object-store requests, metastore connections) may reach addresses that are \
          not globally routable. `deny` (the server-mode default when unset) allows only \
          globally routable unicast addresses (IANA special-purpose registries), the networks of \
          `krishiv.io.egress.allowedCidrs`, and private (RFC 1918, unique-local) or loopback \
          addresses of a host an administrator declared, by its exact name and only for the \
          client that owns it; link-local addresses (cloud metadata) never pass by name. The \
          rule is checked after DNS resolution and on every redirect hop. Under `deny`, in \
          server and embedded mode alike, certificate verification stays on: a store option \
          that turns it off is refused. `allow` (the embedded \
          default) applies no rule.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// `krishiv.io.egress.allowedCidrs` (design P3-T067 §5.1).
pub static IO_EGRESS_ALLOWED_CIDRS: ConfigEntry = ConfigEntry {
    key: "krishiv.io.egress.allowedCidrs",
    value_type: ValueType::String,
    default: None,
    doc: "A comma-separated list of networks (`10.20.0.0/16`, `fd12:3456::/48`) the egress \
          policy always allows, for every client. The only way to open a link-local address. \
          Unset: none.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// `krishiv.io.egress.proxy` (design P3-T067 §5.2).
pub static IO_EGRESS_PROXY: ConfigEntry = ConfigEntry {
    key: "krishiv.io.egress.proxy",
    value_type: ValueType::String,
    default: None,
    doc: "An HTTP(S) proxy URL every egress-policy client sends through; the proxy then decides \
          where requests may go (logged at startup). Unset: the policy clients use no proxy, \
          environment proxy variables included.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};
