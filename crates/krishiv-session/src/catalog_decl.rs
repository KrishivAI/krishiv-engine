// SPDX-License-Identifier: Apache-2.0
//! Declared catalogs: the `krishiv.catalog.<name>.*` keys
//! and Spark's `spark.sql.catalog.<name>` alias, parsed and validated by one pure function, the
//! same for the server at startup and for an embedded session builder. Validation never
//! connects: a catalog that is down does not stop the server.
//!
//! ```text
//! krishiv.catalog.<name>.type            = iceberg-rest | hive | unity | glue | sqlite | memory
//! krishiv.catalog.<name>.uri             = https://catalog.example.com/api/catalog
//! krishiv.catalog.<name>.warehouse       = analytics
//! krishiv.catalog.<name>.auth            = none | token | client-credentials | delegated (default none)
//! krishiv.catalog.<name>.token           = env:NAME | file:/path               (auth=token)
//! krishiv.catalog.<name>.clientId        = krishiv                              (client-credentials)
//! krishiv.catalog.<name>.clientSecret    = env:NAME | file:/path
//! krishiv.catalog.<name>.tokenEndpoint   = https://idp.example.com/oauth2/token (required)
//! krishiv.catalog.<name>.scope / .audience
//! krishiv.catalog.<name>.delegation      = exchange | forward                    (delegated; default exchange)
//! krishiv.catalog.<name>.actorToken      = client-credentials | none             (exchange; default client-credentials)
//! krishiv.catalog.<name>.storage         = vended | allowed-prefixes
//! krishiv.catalog.<name>.tenants         = acme, globex                         (absent: every tenant)
//! krishiv.catalog.<name>.allowConfigKeys = s3.endpoint
//! krishiv.catalog.<name>.allowInsecure   = true                                  (an http:// uri, server mode)
//! krishiv.catalog.<name>.option.<key>    = <value>                               (per type, fixed list)
//! krishiv.catalog.default                = <name>
//! ```
//!
//! `auth = delegated` needs `.audience`; `exchange` needs `.tokenEndpoint`, and
//! with `actorToken = client-credentials` also `.clientId` and `.clientSecret` (the server's own
//! client, the exchange's actor). `forward` takes no token endpoint, client or actor setting. The
//! names `delegation` and `vended` are reserved: `krishiv.catalog.delegation.*` and
//! `krishiv.catalog.vended.*` are engine settings, never a catalog.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue, KrishivUnsupportedFeature};
use krishiv_io::egress::EgressPolicy;

use crate::auth::SecretRef;

/// The prefix of every Krishiv catalog key.
pub const CATALOG_PREFIX: &str = "krishiv.catalog.";
/// The prefix of Spark's catalog keys (the alias).
pub const SPARK_CATALOG_PREFIX: &str = "spark.sql.catalog.";
/// `krishiv.catalog.default` (Spark's `spark.sql.defaultCatalog`).
pub const DEFAULT_CATALOG_KEY: &str = "krishiv.catalog.default";

/// `key` after `prefix`, the prefix matched in any ASCII letter case: the one prefix match of
/// [`is_catalog_key`] and [`CatalogDeclarations::parse`], so no key one routes to the
/// declarations is missed by the other (review 58a2 B1).
fn strip_prefix_ci<'a>(key: &'a str, prefix: &str) -> Option<&'a str> {
    let head = key.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| key.get(prefix.len()..))
        .flatten()
}

/// Whether `key` declares a catalog: a `krishiv.catalog.*` or `spark.sql.catalog.*` key, the
/// prefix in any letter case. In server mode such keys are startup only, for every role (§2.2,
/// decision 1).
#[must_use]
pub fn is_catalog_key(key: &str) -> bool {
    strip_prefix_ci(key, CATALOG_PREFIX).is_some()
        || strip_prefix_ci(key, SPARK_CATALOG_PREFIX).is_some()
}

/// Whether `key` is part of a catalog declaration: a catalog key ([`is_catalog_key`]) that is
/// not a registered engine setting (`krishiv.catalog.requestTimeout`, `maxMetadataBytes`, ...
/// configure every catalog client and stay engine configuration). The server reads these at
/// startup and never stores them as configuration; a session in server mode cannot set them.
#[must_use]
pub fn is_declaration_key(key: &str) -> bool {
    is_catalog_key(key) && krishiv_common::config::lookup(key).is_none()
}

/// Where the declarations come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclMode {
    /// `krishiv server`: references only; an HTTP catalog uri must be `https` unless `allowInsecure`,
    /// and a Hive catalog takes a `thrift://` uri.
    Server,
    /// An embedded session (no principal boundary).
    Embedded,
}

/// A catalog type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CatalogType {
    /// An Iceberg REST catalog.
    IcebergRest,
    /// A Hive metastore (Thrift).
    Hive,
    /// Unity Catalog (read only).
    Unity,
    /// AWS Glue.
    Glue,
    /// A SQLite catalog file.
    Sqlite,
    /// An in-memory catalog.
    Memory,
}

impl CatalogType {
    /// The name in `type`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::IcebergRest => "iceberg-rest",
            Self::Hive => "hive",
            Self::Unity => "unity",
            Self::Glue => "glue",
            Self::Sqlite => "sqlite",
            Self::Memory => "memory",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "iceberg-rest" => Self::IcebergRest,
            "hive" => Self::Hive,
            "unity" => Self::Unity,
            "glue" => Self::Glue,
            "sqlite" => Self::Sqlite,
            "memory" => Self::Memory,
            _ => return None,
        })
    }

    /// The `option.<key>` names this type takes (§2.1: anything else fails startup).
    fn options(self) -> &'static [&'static str] {
        match self {
            Self::IcebergRest => &["prefix"],
            Self::Hive => &["requestTimeout"],
            Self::Glue => &["region", "endpoint", "catalogId"],
            Self::Unity | Self::Sqlite | Self::Memory => &[],
        }
    }

    fn needs_uri(self) -> bool {
        matches!(self, Self::IcebergRest | Self::Unity | Self::Hive)
    }
}

/// How a delegated catalog receives the end user's identity (§3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delegation {
    /// RFC 8693 token exchange at the declared token endpoint.
    Exchange,
    /// The user's own token, when its audience names the catalog.
    Forward,
}

impl Delegation {
    /// The mode name (`system.catalogs.delegation`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Exchange => "exchange",
            Self::Forward => "forward",
        }
    }
}

/// The actor token of a token exchange: the server's own client credentials at the declared
/// token endpoint.
#[derive(Clone, PartialEq, Eq)]
pub struct ActorClient {
    /// `clientId`.
    pub client_id: String,
    /// `clientSecret`.
    pub client_secret: SecretRef,
}

impl std::fmt::Debug for ActorClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActorClient")
            .field("client_id", &self.client_id)
            .field("client_secret", &self.client_secret)
            .finish()
    }
}

/// How the server authenticates to the catalog (§3.1). `Debug` shows the token endpoint without
/// its userinfo.
#[derive(Clone, PartialEq, Eq)]
pub enum CatalogAuth {
    /// Anonymous.
    None,
    /// One static bearer token.
    Token(SecretRef),
    /// OAuth2 client credentials at the declared token endpoint.
    ClientCredentials {
        /// `clientId`.
        client_id: String,
        /// `clientSecret`.
        client_secret: SecretRef,
        /// `tokenEndpoint` (required: the catalog's own `/v1/oauth/tokens` is deprecated).
        token_endpoint: String,
        /// `scope`.
        scope: Option<String>,
        /// `audience`.
        audience: Option<String>,
    },
    /// The end user's identity (§3.3): never the server's.
    Delegated {
        /// `delegation`.
        delegation: Delegation,
        /// `audience` (required): the catalog's audience.
        audience: String,
        /// `tokenEndpoint` (required for `exchange`).
        token_endpoint: Option<String>,
        /// `scope`.
        scope: Option<String>,
        /// The actor of an exchange (`actorToken = client-credentials`); `None` sends none.
        actor: Option<ActorClient>,
    },
}

impl std::fmt::Debug for CatalogAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Token(t) => f.debug_tuple("Token").field(t).finish(),
            Self::ClientCredentials {
                client_id,
                client_secret,
                token_endpoint,
                scope,
                audience,
            } => f
                .debug_struct("ClientCredentials")
                .field("client_id", client_id)
                .field("client_secret", client_secret)
                .field("token_endpoint", &shown_uri(token_endpoint))
                .field("scope", scope)
                .field("audience", audience)
                .finish(),
            Self::Delegated {
                delegation,
                audience,
                token_endpoint,
                scope,
                actor,
            } => f
                .debug_struct("Delegated")
                .field("delegation", delegation)
                .field("audience", audience)
                .field("token_endpoint", &token_endpoint.as_deref().map(shown_uri))
                .field("scope", scope)
                .field("actor", actor)
                .finish(),
        }
    }
}

impl CatalogAuth {
    /// The mode name (`system.catalogs.auth`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Token(_) => "token",
            Self::ClientCredentials { .. } => "client-credentials",
            Self::Delegated { .. } => "delegated",
        }
    }

    /// The delegation form of a `delegated` catalog.
    #[must_use]
    pub fn delegation(&self) -> Option<Delegation> {
        match self {
            Self::Delegated { delegation, .. } => Some(*delegation),
            _ => None,
        }
    }

    /// The declared token endpoint, if any.
    #[must_use]
    pub fn token_endpoint(&self) -> Option<&str> {
        match self {
            Self::ClientCredentials { token_endpoint, .. } => Some(token_endpoint),
            Self::Delegated { token_endpoint, .. } => token_endpoint.as_deref(),
            _ => None,
        }
    }
}

/// Where table data credentials come from (§4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogStorage {
    /// Credentials the catalog vends.
    Vended,
    /// The server's credentials under `krishiv.io.objectStore.allowedPrefixes`.
    AllowedPrefixes,
}

impl CatalogStorage {
    /// The mode name (`system.catalogs.storage`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Vended => "vended",
            Self::AllowedPrefixes => "allowed-prefixes",
        }
    }
}

/// One declared catalog. `Debug` shows the `uri` without its userinfo.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CatalogDecl {
    /// The catalog name (a lowercase identifier).
    pub name: String,
    /// The type.
    pub kind: CatalogType,
    /// `uri`.
    pub uri: Option<String>,
    /// `warehouse`.
    pub warehouse: Option<String>,
    /// `auth` with its credentials.
    pub auth: CatalogAuth,
    /// `storage`.
    pub storage: CatalogStorage,
    /// `tenants`: `None` means every tenant (and principals without one).
    pub tenants: Option<BTreeSet<String>>,
    /// `allowConfigKeys`.
    pub allow_config_keys: BTreeSet<String>,
    /// `allowInsecure`.
    pub allow_insecure: bool,
    /// `option.<key>` values.
    pub options: BTreeMap<String, String>,
}

impl std::fmt::Debug for CatalogDecl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogDecl")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("uri", &self.uri.as_deref().map(shown_uri))
            .field("warehouse", &self.warehouse)
            .field("auth", &self.auth)
            .field("storage", &self.storage)
            .field("tenants", &self.tenants)
            .field("allow_config_keys", &self.allow_config_keys)
            .field("allow_insecure", &self.allow_insecure)
            .field("options", &self.options)
            .finish()
    }
}

/// `uri` as errors, warnings and `Debug` show it (T067 review 58a1 nit 1, re-check 3 n2): the
/// userinfo replaced by the redaction marker, then any credential parameter redacted. The
/// userinfo runs to the last `@` after `//`, wherever it is: a password may hold unencoded `/`,
/// `?`, `#`, `@` or `:`, so no delimiter is trusted to end it (an `@` in the path or query
/// over-redacts, never leaks).
pub(crate) fn shown_uri(uri: &str) -> String {
    let (scheme, rest) = match uri.find("://") {
        Some(i) => uri.split_at(i + 3),
        None => ("", uri),
    };
    let shown = match rest.rfind('@') {
        Some(at) => format!(
            "{scheme}{}{}",
            krishiv_common::secret::REDACTED,
            &rest[at..]
        ),
        None => uri.to_owned(),
    };
    krishiv_common::secret::OptionValue::from_user("uri", shown).to_string()
}

impl CatalogDecl {
    /// Whether a principal of `tenant` sees this catalog (§2.4: hidden means absent).
    #[must_use]
    pub fn visible_to(&self, tenant: Option<&str>) -> bool {
        match (&self.tenants, tenant) {
            (None, _) => true,
            (Some(list), Some(t)) => list.contains(t),
            (Some(_), None) => false,
        }
    }

    /// The hosts this catalog's own clients may reach at private addresses by name.
    ///
    /// The hosts this catalog's own clients may reach at private addresses by name: its
    /// `uri` and token endpoint, and nothing a catalog response names.
    #[must_use]
    pub fn declared_hosts(&self) -> Vec<String> {
        let mut hosts: Vec<String> = self
            .uri
            .iter()
            .map(String::as_str)
            .chain(self.auth.token_endpoint())
            .filter_map(krishiv_io::egress::host_of)
            .collect();
        if self.kind == CatalogType::Glue
            && let Some(endpoint) = self.options.get("endpoint")
            && let Some(host) = krishiv_io::egress::host_of(endpoint)
        {
            hosts.push(host);
        }
        hosts
    }
}

/// The parsed declarations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CatalogDeclarations {
    /// The catalogs, by name.
    pub catalogs: Vec<CatalogDecl>,
    /// `krishiv.catalog.default` (`None`: `spark_catalog`).
    pub default: Option<String>,
    /// Startup warnings (options accepted and ignored), never carrying a value.
    pub warnings: Vec<String>,
}

fn invalid(key: &str, shown: &str, reason: &str) -> KrishivError {
    KrishivInvalidConfigValue {
        value: format!("'{shown}'"),
        key: format!("\"{key}\""),
        reason: reason.to_owned(),
    }
    .into()
}

/// Refuses `key` when `earlier`, another spelling of the same field, already set it: the two
/// would merge with the last one winning silently (review 58a2 N4). The same key given twice
/// keeps its meaning, the last value.
fn once(earlier: Option<&str>, key: &str) -> Result<(), KrishivError> {
    match earlier {
        Some(earlier) if earlier != key => Err(invalid(
            key,
            krishiv_common::secret::REDACTED,
            &format!("set twice: also as {earlier}"),
        )),
        _ => Ok(()),
    }
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// Iceberg's own FileIO classes: Krishiv reads the same bytes through its store layer (§2.1).
const ICEBERG_FILE_IO: [&str; 5] = [
    "org.apache.iceberg.aws.s3.S3FileIO",
    "org.apache.iceberg.gcp.gcs.GCSFileIO",
    "org.apache.iceberg.azure.adlsv2.ADLSFileIO",
    "org.apache.iceberg.io.ResolvingFileIO",
    "org.apache.iceberg.hadoop.HadoopFileIO",
];
const ICEBERG_SPARK_CATALOG: &str = "org.apache.iceberg.spark.SparkCatalog";
/// (unverified: the Unity Catalog Spark class at the UC release PR-16 pins, §2.1.)
const UNITY_SPARK_CATALOG: &str = "io.unitycatalog.spark.UCSingleCatalog";

/// The fields of one catalog as given, before validation.
#[derive(Debug, Default)]
struct Raw {
    /// key -> (the key as written, the value).
    fields: BTreeMap<String, (String, String)>,
    /// From Spark's alias: the class key and its value.
    spark_class: Option<(String, String)>,
}

impl CatalogDeclarations {
    /// Parses every catalog declaration key of `settings`.
    ///
    /// Parses every catalog declaration key of `settings` (other keys, registered
    /// `krishiv.catalog.*` settings included, are ignored). `compiled` lists the catalog types
    /// this build carries; `egress`, when given, checks every declared endpoint whose host is an
    /// address literal by the address rule, with the catalog's own by-name exemption: the `uri`,
    /// the `tokenEndpoint` and a Glue `endpoint` (N3: Glue's AWS SDK client is not a policy
    /// client, so its endpoint is checked only here). A name is checked when it resolves.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` naming the key for every startup refusal of §2.3 (unknown
    /// type, option or class, missing key, literal secret, reserved or malformed name, empty
    /// `tenants`, an `http://` uri in server mode without `allowInsecure`, an endpoint the egress
    /// rule refuses, named by its key and never by its address);
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a type compiled out, `delegated`, an S3 Tables warehouse
    /// or remote signing.
    pub fn parse<'a>(
        settings: impl IntoIterator<Item = (&'a str, &'a str)>,
        mode: DeclMode,
        compiled: &[CatalogType],
        egress: Option<&Arc<EgressPolicy>>,
    ) -> Result<Self, KrishivError> {
        let mut raw: BTreeMap<String, Raw> = BTreeMap::new();
        let mut default: Option<(String, String)> = None;
        for (key, value) in settings {
            if key.eq_ignore_ascii_case(DEFAULT_CATALOG_KEY)
                || key.eq_ignore_ascii_case("spark.sql.defaultCatalog")
            {
                once(default.as_ref().map(|(k, _)| k.as_str()), key)?;
                default = Some((key.to_owned(), value.trim().to_owned()));
            } else if !is_declaration_key(key) {
                // A registered engine setting (`krishiv.catalog.requestTimeout`) or no catalog key.
            } else if let Some(rest) = strip_prefix_ci(key, CATALOG_PREFIX) {
                let Some((name, field)) = rest.split_once('.') else {
                    return Err(invalid(
                        key,
                        krishiv_common::secret::REDACTED,
                        "expected krishiv.catalog.<name>.<setting>",
                    ));
                };
                // `:` marks the parser's own fields (`clientSecret:inline`, `spark:<opt>`):
                // configuration cannot set them (review 58a1 M1).
                if field.contains(':') {
                    return Err(invalid(
                        key,
                        krishiv_common::secret::REDACTED,
                        &format!("catalog `{name}` has no setting `{field}`"),
                    ));
                }
                let entry = raw.entry(name.to_owned()).or_default();
                once(entry.fields.get(field).map(|(k, _)| k.as_str()), key)?;
                entry
                    .fields
                    .insert(field.to_owned(), (key.to_owned(), value.to_owned()));
            } else if let Some(rest) = strip_prefix_ci(key, SPARK_CATALOG_PREFIX) {
                match rest.split_once('.') {
                    None => {
                        let entry = raw.entry(rest.to_owned()).or_default();
                        once(entry.spark_class.as_ref().map(|(k, _)| k.as_str()), key)?;
                        entry.spark_class = Some((key.to_owned(), value.trim().to_owned()));
                    }
                    Some((name, opt)) => {
                        let entry = raw.entry(name.to_owned()).or_default();
                        let field = format!("spark:{opt}");
                        once(entry.fields.get(&field).map(|(k, _)| k.as_str()), key)?;
                        entry
                            .fields
                            .insert(field, (key.to_owned(), value.to_owned()));
                    }
                }
            } else {
                // `is_declaration_key` routed it here, so it must not be dropped: a dropped
                // `tenants` would widen the catalog (review 58a2 B1).
                return Err(invalid(
                    key,
                    krishiv_common::secret::REDACTED,
                    "not a catalog declaration key",
                ));
            }
        }
        let mut out = Self {
            default: default.map(|(_, v)| v),
            ..Self::default()
        };
        for (name, entry) in raw {
            let decl = decl(&name, entry, mode, compiled, &mut out.warnings)?;
            if let Some(policy) = egress {
                check_endpoints(&decl, policy)?;
            }
            out.catalogs.push(decl);
        }
        if let Some(d) = &out.default
            && d != "spark_catalog"
            && !out.catalogs.iter().any(|c| &c.name == d)
        {
            return Err(invalid(DEFAULT_CATALOG_KEY, d, "names no declared catalog"));
        }
        Ok(out)
    }
}

/// The declared endpoints of `decl` checked by the address rule under the catalog's own scope
/// (its declared hosts exempt by name, so an on-premises catalog at a private address passes;
/// link-local and special-purpose addresses never do, in any spelling).
fn check_endpoints(decl: &CatalogDecl, policy: &Arc<EgressPolicy>) -> Result<(), KrishivError> {
    let scope = policy.scope(decl.declared_hosts());
    let token_endpoint = decl.auth.token_endpoint();
    let glue_endpoint = (decl.kind == CatalogType::Glue)
        .then(|| decl.options.get("endpoint").map(String::as_str))
        .flatten();
    for (field, endpoint) in [
        ("uri", decl.uri.as_deref()),
        ("tokenEndpoint", token_endpoint),
        ("option.endpoint", glue_endpoint),
    ] {
        if let Some(endpoint) = endpoint
            && scope.check_endpoint(endpoint).is_err()
        {
            return Err(invalid(
                &format!("{CATALOG_PREFIX}{}.{field}", decl.name),
                "<endpoint>",
                "the egress policy refuses this endpoint's address (private address); list its \
                 network in krishiv.io.egress.allowedCidrs if it is meant",
            ));
        }
    }
    Ok(())
}

fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Maps Spark's alias fields onto the Krishiv ones (§2.1), in place.
fn map_spark(
    name: &str,
    entry: &mut Raw,
    mode: DeclMode,
    warnings: &mut Vec<String>,
) -> Result<(), KrishivError> {
    let spark: Vec<(String, (String, String))> = entry
        .fields
        .iter()
        .filter(|(k, _)| k.starts_with("spark:"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if spark.is_empty() && entry.spark_class.is_none() {
        return Ok(());
    }
    let Some((class_key, class)) = entry.spark_class.clone() else {
        let (key, _) = &spark[0].1;
        return Err(invalid(
            key,
            "",
            "the catalog class spark.sql.catalog.<name> is not set",
        ));
    };
    let mut auth_set = entry.fields.contains_key("auth");
    // A field a `krishiv.catalog.` key also sets is one field under two spellings (review 58a2
    // N4); the alias's own options may set one field together (`token` and `rest.auth.type` both
    // set `auth`).
    let mut clash = None;
    let mut put = |field: &str, key: &str, value: &str| {
        if let Some((earlier, _)) = entry.fields.get(field)
            && strip_prefix_ci(earlier, CATALOG_PREFIX).is_some()
            && clash.is_none()
        {
            clash = Some((key.to_owned(), earlier.clone()));
        }
        entry
            .fields
            .insert(field.to_owned(), (key.to_owned(), value.to_owned()));
    };
    let unity = match class.as_str() {
        ICEBERG_SPARK_CATALOG => false,
        UNITY_SPARK_CATALOG => true,
        _ => {
            return Err(invalid(
                &class_key,
                &class,
                "this catalog class is not implemented by Krishiv",
            ));
        }
    };
    if unity {
        put("type", &class_key, "unity");
    }
    let mut credential = None;
    for (field, (key, value)) in &spark {
        let opt = field.trim_start_matches("spark:");
        if opt == "rest.auth.type" {
            auth_set = true;
        }
        match (unity, opt) {
            (false, "type") => {
                let kind = match value.as_str() {
                    "rest" => "iceberg-rest",
                    "hive" => "hive",
                    "glue" => "glue",
                    _ => return Err(invalid(key, value, "expected rest, hive or glue")),
                };
                put("type", key, kind);
            }
            (_, "uri") => put("uri", key, value),
            (false, "warehouse") => put("warehouse", key, value),
            (_, "token") => {
                auth_set = true;
                put("auth", key, "token");
                put("token", key, value);
            }
            (false, "credential") => credential = Some((key.clone(), value.clone())),
            (false, "oauth2-server-uri") => put("tokenEndpoint", key, value),
            (false, "scope") => put("scope", key, value),
            (false, "rest.auth.type") => match value.as_str() {
                "oauth2" => put("auth", key, "client-credentials"),
                "none" => put("auth", key, "none"),
                _ => return Err(invalid(key, value, "expected oauth2 or none")),
            },
            (false, "header.X-Iceberg-Access-Delegation") => match value.as_str() {
                "vended-credentials" => put("storage", key, "vended"),
                "remote-signing" => {
                    return Err(unsupported(format!(
                        "Iceberg REST remote signing (catalog `{name}`)"
                    )));
                }
                _ => return Err(invalid(key, value, "expected vended-credentials")),
            },
            (false, "rest-metrics-reporting-enabled") => warnings.push(format!(
                "catalog `{name}`: {key} has no meaning for Krishiv and is ignored"
            )),
            (false, "io-impl") if ICEBERG_FILE_IO.contains(&value.as_str()) => {
                warnings.push(format!(
                    "catalog `{name}`: {key} names an Iceberg FileIO; Krishiv reads through its \
                     own store layer and ignores it"
                ));
            }
            (false, "io-impl") => {
                return Err(invalid(
                    key,
                    value,
                    "a custom FileIO can change what the bytes mean; only Iceberg's own FileIO \
                     classes are accepted",
                ));
            }
            _ => {
                return Err(invalid(
                    key,
                    krishiv_common::secret::REDACTED,
                    &format!("catalog `{name}` has no option `{opt}` in Krishiv"),
                ));
            }
        }
    }
    if let Some((key, value)) = credential {
        // `client_id:client_secret` in one string: a reference in server mode.
        let secret = match SecretRef::parse(&key, &value, mode == DeclMode::Embedded)? {
            SecretRef::Inline(s) => {
                let text = s.expose_secret();
                let Some((id, secret)) = text.split_once(':') else {
                    return Err(invalid(
                        &key,
                        krishiv_common::secret::REDACTED,
                        "expected client_id:client_secret",
                    ));
                };
                put("clientId", &key, id);
                put("clientSecret:inline", &key, secret);
                None
            }
            reference => Some(reference),
        };
        if secret.is_some() {
            // A referenced Spark credential holds both halves in one secret: not split yet.
            return Err(unsupported(format!(
                "a referenced Spark `credential` for catalog `{name}`; set \
                 krishiv.catalog.{name}.clientId and .clientSecret instead"
            )));
        }
        if !auth_set {
            put("auth", &key, "client-credentials");
        }
    }
    if let Some((key, earlier)) = clash {
        return Err(invalid(
            &key,
            krishiv_common::secret::REDACTED,
            &format!("set twice: also as {earlier}"),
        ));
    }
    entry.fields.retain(|k, _| !k.starts_with("spark:"));
    Ok(())
}

fn decl(
    name: &str,
    mut entry: Raw,
    mode: DeclMode,
    compiled: &[CatalogType],
    warnings: &mut Vec<String>,
) -> Result<CatalogDecl, KrishivError> {
    let first_key = entry
        .fields
        .values()
        .map(|(k, _)| k.clone())
        .chain(entry.spark_class.iter().map(|(k, _)| k.clone()))
        .next()
        .unwrap_or_default();
    // `delegation` and `vended` are the engine settings `krishiv.catalog.delegation.*` and
    // `krishiv.catalog.vended.*`: a mistyped one must not declare a catalog.
    if matches!(name, "spark_catalog" | "system" | "delegation" | "vended") {
        return Err(invalid(&first_key, name, "the catalog name is reserved"));
    }
    if !valid_name(name) {
        return Err(invalid(
            &first_key,
            name,
            "a catalog name is a lowercase identifier",
        ));
    }
    map_spark(name, &mut entry, mode, warnings)?;
    let fields = entry.fields;
    let get = |f: &str| fields.get(f).map(|(k, v)| (k.as_str(), v.trim()));
    let key_of = |f: &str| format!("{CATALOG_PREFIX}{name}.{f}");
    let Some((type_key, type_text)) = get("type") else {
        return Err(invalid(
            &key_of("type"),
            "",
            "every declared catalog needs a type",
        ));
    };
    let kind = CatalogType::parse(type_text).ok_or_else(|| {
        invalid(
            type_key,
            type_text,
            "expected iceberg-rest, hive, unity, glue, sqlite or memory",
        )
    })?;
    if !compiled.contains(&kind) {
        return Err(unsupported(format!(
            "catalog type `{}` (catalog `{name}`) in this build",
            kind.name()
        )));
    }
    const KNOWN: [&str; 17] = [
        "type",
        "uri",
        "warehouse",
        "auth",
        "token",
        "clientId",
        "clientSecret",
        "clientSecret:inline",
        "tokenEndpoint",
        "scope",
        "audience",
        "storage",
        "tenants",
        "allowConfigKeys",
        "allowInsecure",
        "delegation",
        "actorToken",
    ];
    let mut options = BTreeMap::new();
    for (field, (key, value)) in &fields {
        if let Some(opt) = field.strip_prefix("option.") {
            if !kind.options().contains(&opt) {
                return Err(invalid(
                    key,
                    krishiv_common::secret::REDACTED,
                    &format!("catalog type `{}` has no option `{opt}`", kind.name()),
                ));
            }
            options.insert(opt.to_owned(), value.trim().to_owned());
        } else if !KNOWN.contains(&field.as_str()) {
            return Err(invalid(
                key,
                krishiv_common::secret::REDACTED,
                &format!("catalog `{name}` has no setting `{field}`"),
            ));
        }
    }
    let allow_insecure = match get("allowInsecure") {
        None => false,
        Some((_, v)) if v.eq_ignore_ascii_case("true") => true,
        Some((_, v)) if v.eq_ignore_ascii_case("false") => false,
        Some((k, v)) => return Err(invalid(k, v, "expected true or false")),
    };
    let uri = get("uri")
        .map(|(_, v)| v.to_owned())
        .filter(|v| !v.is_empty());
    if kind.needs_uri() && uri.is_none() {
        return Err(invalid(&key_of("uri"), "", "this catalog type needs a uri"));
    }
    if let Some(u) = &uri {
        let (k, _) = get("uri").unwrap_or_default();
        let lower = u.to_ascii_lowercase();
        let scheme_ok = match kind {
            CatalogType::Hive => lower.starts_with("thrift://"),
            _ => lower.starts_with("https://") || lower.starts_with("http://"),
        };
        if !scheme_ok {
            return Err(invalid(
                k,
                &shown_uri(u),
                "not a uri this catalog type speaks",
            ));
        }
        if lower.starts_with("http://") && mode == DeclMode::Server && !allow_insecure {
            return Err(invalid(
                k,
                &shown_uri(u),
                "an http:// catalog uri needs allowInsecure = true in server mode",
            ));
        }
        if lower.starts_with("http://") && allow_insecure {
            warnings.push(format!(
                "catalog `{name}` is reached over plain http (allowInsecure)"
            ));
        }
    }
    let warehouse = get("warehouse").map(|(_, v)| v.to_owned());
    if kind == CatalogType::IcebergRest
        && warehouse
            .as_deref()
            .is_some_and(|w| w.starts_with("arn:aws:s3tables:"))
    {
        return Err(unsupported(format!(
            "an S3 Tables warehouse through iceberg-rest (catalog `{name}`)"
        )));
    }
    let secret = |field: &str| -> Result<Option<SecretRef>, KrishivError> {
        if let Some((_, v)) = fields.get(&format!("{field}:inline"))
            && mode == DeclMode::Embedded
        {
            // Only the embedded Spark `credential` reaches here inline.
            return Ok(Some(SecretRef::Inline(
                krishiv_common::secret::SecretString::new(v.as_str()),
            )));
        }
        match get(field) {
            None => Ok(None),
            Some((k, v)) => {
                // Spark's `token` option may be inline in embedded mode; Krishiv's keys never.
                let spark_inline = mode == DeclMode::Embedded
                    && strip_prefix_ci(k, SPARK_CATALOG_PREFIX).is_some();
                SecretRef::parse(k, v, spark_inline).map(Some)
            }
        }
    };
    let auth = match get("auth").map(|(k, v)| (k, v.to_owned())) {
        None => CatalogAuth::None,
        Some((_, v)) if v == "none" => CatalogAuth::None,
        Some((k, v)) if v == "token" => CatalogAuth::Token(
            secret("token")?
                .ok_or_else(|| invalid(k, "token", "auth = token needs a token reference"))?,
        ),
        Some((k, v)) if v == "client-credentials" => {
            let need = |f: &str| {
                get(f)
                    .map(|(_, v)| v.to_owned())
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| {
                        invalid(
                            &key_of(f),
                            "",
                            &format!("auth = client-credentials needs {f}"),
                        )
                    })
            };
            let token_endpoint = need("tokenEndpoint")?;
            let te = token_endpoint.to_ascii_lowercase();
            if !(te.starts_with("https://") || (te.starts_with("http://") && allow_insecure)) {
                return Err(invalid(
                    &key_of("tokenEndpoint"),
                    &shown_uri(&token_endpoint),
                    "the token endpoint must be https (or http with allowInsecure)",
                ));
            }
            CatalogAuth::ClientCredentials {
                client_id: need("clientId")?,
                client_secret: secret("clientSecret")?.ok_or_else(|| {
                    invalid(
                        k,
                        "client-credentials",
                        "auth = client-credentials needs clientSecret",
                    )
                })?,
                token_endpoint,
                scope: get("scope").map(|(_, v)| v.to_owned()),
                audience: get("audience").map(|(_, v)| v.to_owned()),
            }
        }
        Some((_, v)) if v == "delegated" => delegated(&fields, &key_of, allow_insecure, &secret)?,
        Some((k, v)) => {
            return Err(invalid(
                k,
                &v,
                "expected none, token, client-credentials or delegated",
            ));
        }
    };
    if auth.delegation().is_none() {
        for f in ["delegation", "actorToken"] {
            if let Some((k, _)) = get(f) {
                return Err(invalid(
                    k,
                    "",
                    "this setting applies to auth = delegated only",
                ));
            }
        }
    }
    let storage = match get("storage") {
        None if matches!(kind, CatalogType::IcebergRest | CatalogType::Unity) => {
            CatalogStorage::Vended
        }
        None | Some((_, "allowed-prefixes")) => CatalogStorage::AllowedPrefixes,
        Some((_, "vended")) => CatalogStorage::Vended,
        Some((_, "remote-signing")) => {
            return Err(unsupported(format!(
                "Iceberg REST remote signing (catalog `{name}`)"
            )));
        }
        Some((k, v)) => return Err(invalid(k, v, "expected vended or allowed-prefixes")),
    };
    let list = |v: &str| -> BTreeSet<String> {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    };
    let tenants = match get("tenants") {
        None => None,
        Some((k, v)) => {
            let set = list(v);
            // A blank value must not be the widest setting (S6).
            if set.is_empty() {
                return Err(invalid(
                    k,
                    v,
                    "an empty tenant list; remove the key for every tenant",
                ));
            }
            Some(set)
        }
    };
    Ok(CatalogDecl {
        name: name.to_owned(),
        kind,
        uri,
        warehouse,
        auth,
        storage,
        tenants,
        allow_config_keys: get("allowConfigKeys")
            .map(|(_, v)| list(v))
            .unwrap_or_default(),
        allow_insecure,
        options,
    })
}

/// The `auth = delegated` fields (§3.3).
fn delegated(
    fields: &BTreeMap<String, (String, String)>,
    key_of: &dyn Fn(&str) -> String,
    allow_insecure: bool,
    secret: &dyn Fn(&str) -> Result<Option<SecretRef>, KrishivError>,
) -> Result<CatalogAuth, KrishivError> {
    let get = |f: &str| fields.get(f).map(|(k, v)| (k.as_str(), v.trim()));
    let need = |f: &str, why: &str| {
        get(f)
            .map(|(_, v)| v.to_owned())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid(&key_of(f), "", why))
    };
    let delegation = match get("delegation") {
        None | Some((_, "exchange")) => Delegation::Exchange,
        Some((_, "forward")) => Delegation::Forward,
        Some((k, v)) => return Err(invalid(k, v, "expected exchange or forward")),
    };
    let audience = need("audience", "auth = delegated needs the catalog's audience")?;
    let scope = get("scope").map(|(_, v)| v.to_owned());
    if delegation == Delegation::Forward {
        // Nothing of an exchange applies: a forwarded token is the user's own.
        for f in ["tokenEndpoint", "actorToken", "clientId", "clientSecret"] {
            if let Some((k, _)) = get(f) {
                return Err(invalid(
                    k,
                    "",
                    "delegation = forward sends the user's own token; this setting applies to \
                     delegation = exchange",
                ));
            }
        }
        return Ok(CatalogAuth::Delegated {
            delegation,
            audience,
            token_endpoint: None,
            scope,
            actor: None,
        });
    }
    let token_endpoint = need(
        "tokenEndpoint",
        "delegation = exchange needs the token endpoint",
    )?;
    let te = token_endpoint.to_ascii_lowercase();
    if !(te.starts_with("https://") || (te.starts_with("http://") && allow_insecure)) {
        return Err(invalid(
            &key_of("tokenEndpoint"),
            &shown_uri(&token_endpoint),
            "the token endpoint must be https (or http with allowInsecure)",
        ));
    }
    let actor = match get("actorToken") {
        None | Some((_, "client-credentials")) => Some(ActorClient {
            client_id: need("clientId", "actorToken = client-credentials needs clientId")?,
            client_secret: secret("clientSecret")?.ok_or_else(|| {
                invalid(
                    &key_of("clientSecret"),
                    "",
                    "actorToken = client-credentials needs clientSecret",
                )
            })?,
        }),
        Some((_, "none")) => {
            for f in ["clientId", "clientSecret"] {
                if let Some((k, _)) = get(f) {
                    return Err(invalid(k, "", "actorToken = none sends no client"));
                }
            }
            None
        }
        Some((k, v)) => return Err(invalid(k, v, "expected client-credentials or none")),
    };
    Ok(CatalogAuth::Delegated {
        delegation,
        audience,
        token_endpoint: Some(token_endpoint),
        scope,
        actor,
    })
}

#[cfg(test)]
mod tests;
