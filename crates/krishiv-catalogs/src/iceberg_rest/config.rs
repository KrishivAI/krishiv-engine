// SPDX-License-Identifier: Apache-2.0
//! How to reach an Iceberg REST catalog and how much to trust it (design P3-lakehouse §5.6, T067 §3.4, decision 8).
//!
//! Everything here comes from the catalog's declaration, never from the server: the base `uri`, the warehouse, the
//! credentials and the token endpoint. A server's `/v1/config` may set `prefix` and `namespace-separator` (both
//! checked) and nothing else this client acts on; a `uri` it returns is ignored, because following it would send
//! the caller's token to a host the declaration never named (the reason `iceberg-catalog-rest` 0.11 is not used).

use std::time::Duration;

use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue, KrishivUnsupportedFeature};
use krishiv_common::secret::SecretString;
use url::Url;

/// The OneLake (Microsoft Fabric) Iceberg REST endpoint.
pub const ONELAKE_URI: &str = "https://onelake.table.fabric.microsoft.com/iceberg";

/// The OAuth2 scope the Java client asks for when none is configured (`OAuth2Properties.CATALOG_SCOPE`).
pub const DEFAULT_SCOPE: &str = "catalog";

/// How a request is authenticated. Secrets are [`SecretString`]s from the declaration on.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RestAuth {
    /// No `Authorization` header (a local test catalog).
    None,
    /// A static bearer token, sent as it is on every request (OneLake: an Entra ID token for
    /// `https://storage.azure.com/`).
    Token(SecretString),
    /// The OAuth2 client-credentials grant: the client id and secret go in the form body, as the Java client sends
    /// them (`OAuth2Util.clientCredentialsRequest`), to `token_endpoint`; the token is reused until shortly before
    /// it expires, or until the catalog answers 401.
    ClientCredentials {
        /// The client id; `None` sends only the secret (the Java `credential` form without `id:`).
        client_id: Option<String>,
        /// The client secret.
        client_secret: SecretString,
        /// The token endpoint. `None` is the catalog's own `v1/oauth/tokens` (deprecated in the spec; set
        /// `oauth2-server-uri` when the IdP is elsewhere).
        token_endpoint: Option<Url>,
        /// The scope; `None` is [`DEFAULT_SCOPE`].
        scope: Option<String>,
        /// Extra form fields the IdP needs (`audience`, `resource`).
        extra: Vec<(String, String)>,
    },
    /// The end user's identity (T067 §3.3): the caller's own token (`forward`) or an RFC 8693 exchange of it at
    /// `token_endpoint`, never the server's identity.
    Delegated {
        /// Send the caller's token unchanged (only when its audience names `audience`).
        forward: bool,
        /// The catalog's audience.
        audience: String,
        /// The token endpoint of the exchange (required unless `forward`).
        token_endpoint: Option<Url>,
        /// The `scope` of the exchange (and of the actor's client-credentials token).
        scope: Option<String>,
        /// The actor's client id and secret (`actorToken = client-credentials`); `None` sends no actor token.
        actor: Option<(String, SecretString)>,
        /// The exchange limits.
        limits: super::delegated::DelegationLimits,
    },
}

/// The limits of one client (design P3-lakehouse §5.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestLimits {
    /// Longest one call may take, first byte sent to last byte read, token fetch and every page of a listing
    /// included (`krishiv.catalog.requestTimeout`; more than zero, at most an hour).
    pub request_timeout: Duration,
    /// Most bytes one response may carry (`krishiv.catalog.maxMetadataBytes`). The client never asks for or
    /// accepts a compressed body, so this counts the bytes the parser sees.
    pub max_response_bytes: usize,
    /// Deepest JSON nesting accepted (64; at most 128).
    pub max_depth: usize,
    /// Most entries one listing may return over all its pages.
    pub max_list_entries: usize,
    /// Most pages one listing may take.
    pub max_pages: usize,
}

/// The longest `request_timeout` accepted.
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(3600);

impl RestLimits {
    /// Refuses a zero limit, a zero or over-an-hour timeout, and a depth past 128 (the parser's own recursion
    /// limit), so no limit can make every call fail or overflow a deadline (LH §5.6).
    fn validate(&self) -> Result<(), KrishivError> {
        if self.request_timeout.is_zero() || self.request_timeout > MAX_REQUEST_TIMEOUT {
            return Err(invalid(
                "requestTimeout",
                "it must be more than zero and at most an hour",
            ));
        }
        for (key, value) in [
            ("maxMetadataBytes", self.max_response_bytes),
            ("maxDepth", self.max_depth),
            ("maxListEntries", self.max_list_entries),
            ("maxPages", self.max_pages),
        ] {
            if value == 0 {
                return Err(invalid(key, "it must be more than zero"));
            }
        }
        if self.max_depth > 128 {
            return Err(invalid("maxDepth", "it must be at most 128"));
        }
        Ok(())
    }
}

impl Default for RestLimits {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            max_response_bytes: 256 << 20,
            max_depth: 64,
            max_list_entries: 1_000_000,
            max_pages: 10_000,
        }
    }
}

/// One Iceberg REST catalog as its declaration describes it.
#[derive(Debug, Clone)]
pub struct RestConfig {
    /// The base URI (`https://host/path`); requests go to `<uri>/v1/...`.
    pub uri: Url,
    /// The `warehouse` sent to `/v1/config`, if any.
    pub warehouse: Option<String>,
    /// How requests are authenticated.
    pub auth: RestAuth,
    /// Ask the catalog for vended storage credentials (`X-Iceberg-Access-Delegation: vended-credentials`).
    pub vended_credentials: bool,
    /// Allow `http://` for the catalog and the token endpoint. Off by default: a token over plain HTTP is a token
    /// anyone on the path can read. Local test catalogs turn it on.
    pub allow_http: bool,
    /// The limits.
    pub limits: RestLimits,
    /// The egress policy connector every request goes out on (design P3-T067 §5.2): the server's policy scoped to
    /// this catalog's declared hosts, no redirects. `None` is a connector under no rule (embedded use, tests), still
    /// with no redirect followed and no ambient proxy.
    pub egress: Option<krishiv_store::PolicyConnector>,
}

/// Why a declaration is refused.
fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

impl RestConfig {
    /// A catalog at `uri` with no authentication, no warehouse and the default limits.
    ///
    /// # Errors
    /// `uri` is not an absolute `http(s)` URL without credentials, query or fragment.
    pub fn new(uri: &str) -> Result<Self, KrishivError> {
        let uri = Url::parse(uri).map_err(|_| invalid("uri", "it is not a URL"))?;
        Ok(Self {
            uri,
            warehouse: None,
            auth: RestAuth::None,
            vended_credentials: false,
            allow_http: false,
            limits: RestLimits::default(),
            egress: None,
        })
    }

    /// The connector requests go out on: [`Self::egress`], or one under no rule.
    pub(crate) fn connector(&self) -> krishiv_store::PolicyConnector {
        self.egress.clone().unwrap_or_else(|| {
            use krishiv_store::egress::{EgressPolicy, PrivateAddresses, ProviderEndpoints};
            let policy = EgressPolicy::new(
                PrivateAddresses::Allow,
                Vec::new(),
                ProviderEndpoints::default(),
            );
            krishiv_store::PolicyConnector::new(
                policy.scope(Vec::<String>::new()),
                krishiv_store::Redirects::None,
                None,
            )
            .without_provider_literals()
        })
    }

    /// The OneLake form (design P3-lakehouse-benchmark-iceberg): [`ONELAKE_URI`], warehouse
    /// `<workspace>/<item>` (e.g. `ws/sales.Lakehouse`, names or GUIDs) and a static Entra ID bearer token.
    ///
    /// # Errors
    /// An empty workspace or item, or one containing `/`.
    pub fn onelake(workspace: &str, item: &str, token: SecretString) -> Result<Self, KrishivError> {
        for (what, part) in [("workspace", workspace), ("item", item)] {
            if part.trim().is_empty() || part.contains('/') {
                return Err(invalid(what, "it is empty or contains `/`"));
            }
        }
        let mut config = Self::new(ONELAKE_URI)?;
        config.warehouse = Some(format!("{workspace}/{item}"));
        config.auth = RestAuth::Token(token);
        Ok(config)
    }

    /// Checks the declaration before anything connects: scheme, credentials in the URL, S3 Tables.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an S3 Tables catalog (decision 15) or `http://` without `allow_http`;
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a malformed URI or token endpoint.
    pub fn validate(&self) -> Result<(), KrishivError> {
        check_url(&self.uri, self.allow_http, "uri")?;
        self.limits.validate()?;
        if is_s3_tables(&self.uri, self.warehouse.as_deref()) {
            return Err(unsupported("Iceberg S3 Tables catalog"));
        }
        match &self.auth {
            RestAuth::ClientCredentials {
                token_endpoint: Some(endpoint),
                ..
            } => check_url(endpoint, self.allow_http, "oauth2-server-uri")?,
            RestAuth::Delegated {
                forward,
                token_endpoint,
                ..
            } => match token_endpoint {
                Some(endpoint) => check_url(endpoint, self.allow_http, "tokenEndpoint")?,
                None if !forward => {
                    return Err(invalid(
                        "tokenEndpoint",
                        "a token exchange needs the token endpoint",
                    ));
                }
                None => {}
            },
            _ => {}
        }
        Ok(())
    }
}

/// An S3 Tables catalog: an `arn:aws:s3tables:` warehouse or the `s3tables.<region>.amazonaws.com` endpoint.
fn is_s3_tables(uri: &Url, warehouse: Option<&str>) -> bool {
    let arn = warehouse.is_some_and(|w| {
        w.trim()
            .to_ascii_lowercase()
            .starts_with("arn:aws:s3tables:")
    });
    let host = uri.host_str().is_some_and(|h| {
        let h = h.to_ascii_lowercase();
        h.starts_with("s3tables.")
            && (h.ends_with(".amazonaws.com") || h.ends_with(".amazonaws.com.cn"))
    });
    arn || host
}

/// A declaration value this client cannot use. The value itself is never printed: a URI may carry credentials.
fn invalid(key: &str, reason: &str) -> KrishivError {
    KrishivInvalidConfigValue {
        value: "(not shown)".to_owned(),
        key: format!("\"{key}\""),
        reason: reason.to_owned(),
    }
    .into()
}

fn check_url(url: &Url, allow_http: bool, what: &str) -> Result<(), KrishivError> {
    match url.scheme() {
        "https" => {}
        "http" if allow_http => {}
        "http" => {
            return Err(unsupported(&format!(
                "a plain `http://` Iceberg REST `{what}` (tokens would cross the network in clear)"
            )));
        }
        _ => return Err(invalid(what, "it is not an http(s) URL")),
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err(invalid(what, "it has no host"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(invalid(what, "it carries credentials"));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(invalid(what, "it carries a query or fragment"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    #[test]
    fn s3_tables_is_refused_cleanly() {
        let mut c = RestConfig::new("https://catalog.example.com/iceberg").unwrap();
        c.warehouse = Some("arn:aws:s3tables:us-east-1:111122223333:bucket/b".into());
        let e = c.validate().unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(e.to_string().contains("Iceberg S3 Tables catalog"), "{e}");

        let c = RestConfig::new("https://s3tables.us-east-1.amazonaws.com/iceberg").unwrap();
        let e = c.validate().unwrap_err();
        assert!(e.to_string().contains("Iceberg S3 Tables catalog"), "{e}");
    }

    #[test]
    fn plain_http_needs_the_declaration_to_allow_it() {
        let mut c = RestConfig::new("http://localhost:8181").unwrap();
        assert_eq!(
            c.validate().unwrap_err().error_class(),
            ErrorClass::KrishivUnsupportedFeature
        );
        c.allow_http = true;
        c.validate().unwrap();
    }

    #[test]
    fn malformed_declarations_are_refused() {
        for uri in [
            "ftp://h/x",
            "https://user:pw@h/x",
            "https://h/x?token=1",
            "https://h/x#f",
        ] {
            let c = RestConfig::new(uri).unwrap();
            assert!(c.validate().is_err(), "{uri}");
        }
        assert!(RestConfig::new("not a url").is_err());
        let mut c = RestConfig::new("https://h").unwrap();
        c.auth = RestAuth::ClientCredentials {
            client_id: Some("id".into()),
            client_secret: SecretString::new("s"),
            token_endpoint: Some(Url::parse("http://idp/token").unwrap()),
            scope: None,
            extra: Vec::new(),
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn the_onelake_form_names_the_fabric_endpoint_and_warehouse() {
        let c = RestConfig::onelake("ws", "sales.Lakehouse", SecretString::new("t")).unwrap();
        assert_eq!(c.uri.as_str(), ONELAKE_URI);
        assert_eq!(c.warehouse.as_deref(), Some("ws/sales.Lakehouse"));
        assert!(matches!(c.auth, RestAuth::Token(_)));
        c.validate().unwrap();
        assert!(RestConfig::onelake("a/b", "x.Lakehouse", SecretString::new("t")).is_err());
        assert!(RestConfig::onelake("ws", " ", SecretString::new("t")).is_err());
    }
}
