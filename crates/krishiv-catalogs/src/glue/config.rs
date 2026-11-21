// SPDX-License-Identifier: Apache-2.0
//! The declaration of one AWS Glue Data Catalog: region (or an endpoint override), static credentials, an optional
//! catalog id and the limits (design P3-lakehouse §2.3). Roles, web identity and instance profiles are T067, not here.

use std::time::Duration;

use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue, KrishivUnsupportedFeature};
use krishiv_common::secret::SecretString;
use url::Url;

/// The limits of one call, checked before or while a response is parsed. Callers set the fields directly:
/// the `krishiv.catalog.maxMetadataBytes` and `krishiv.catalog.requestTimeout` settings do not apply to Glue.
#[derive(Debug, Clone)]
pub struct GlueLimits {
    /// The whole call, a paged listing included.
    pub request_timeout: Duration,
    /// The largest response body.
    pub max_response_bytes: usize,
    /// The deepest JSON nesting.
    pub max_depth: usize,
    /// Entries in one listing over all its pages.
    pub max_list_entries: usize,
    /// Pages in one listing.
    pub max_pages: usize,
    /// Columns of one table, nested fields included (`krishiv.catalog.maxColumns`).
    pub max_columns: usize,
}

impl Default for GlueLimits {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            max_response_bytes: 256 << 20,
            max_depth: 64,
            max_list_entries: 1_000_000,
            max_pages: 10_000,
            max_columns: 10_000,
        }
    }
}

fn invalid(key: &str, reason: &str) -> KrishivError {
    KrishivInvalidConfigValue {
        value: "(not shown)".to_owned(),
        key: format!("\"{key}\""),
        reason: reason.to_owned(),
    }
    .into()
}

impl GlueLimits {
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a zero limit, a zero timeout or a depth over 127.
    pub fn validate(&self) -> Result<(), KrishivError> {
        if self.request_timeout.is_zero() {
            return Err(invalid("requestTimeout", "it must be more than zero"));
        }
        if std::time::Instant::now()
            .checked_add(self.request_timeout)
            .is_none()
        {
            return Err(invalid(
                "requestTimeout",
                "it is too long to form a deadline",
            ));
        }
        for (key, value) in [
            ("maxMetadataBytes", self.max_response_bytes),
            ("maxDepth", self.max_depth),
            ("maxListEntries", self.max_list_entries),
            ("maxPages", self.max_pages),
            ("maxColumns", self.max_columns),
        ] {
            if value == 0 {
                return Err(invalid(key, "it must be more than zero"));
            }
        }
        // serde_json refuses the 128th nested container, so 127 is the deepest that always decodes.
        if self.max_depth > 127 {
            return Err(invalid("maxDepth", "it must be at most 127"));
        }
        Ok(())
    }
}

/// One Glue Data Catalog as its declaration describes it.
#[derive(Debug, Clone)]
pub struct GlueConfig {
    /// The AWS region (`us-east-1`); it names the default endpoint and is part of every signature.
    pub region: String,
    /// The endpoint, `https://glue.<region>.amazonaws.com` unless a VPC endpoint, LocalStack or a test server is
    /// named. Its path must be empty or `/`.
    pub endpoint: Url,
    /// The access key id.
    pub access_key_id: String,
    /// The secret access key.
    pub secret_access_key: SecretString,
    /// The session token of temporary credentials.
    pub session_token: Option<SecretString>,
    /// The Glue catalog id (an AWS account id); `None` is the caller's own account.
    pub catalog_id: Option<String>,
    /// Allow an `http://` endpoint. Off by default: a signed request over plain HTTP is replayable within its five
    /// minutes. LocalStack and test servers turn it on.
    pub allow_http: bool,
    /// The limits.
    pub limits: GlueLimits,
    /// The egress policy connector every request goes out on (design P3-T067 §5.2). `None` is a connector under no
    /// rule (embedded use, tests), still with no redirect followed and no ambient proxy.
    pub egress: Option<krishiv_store::PolicyConnector>,
}

impl GlueConfig {
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

    /// A declaration for `region` with the default endpoint, the given static keys and default limits.
    ///
    /// # Errors
    /// `region` is not a region name, or the keys are empty.
    pub fn new(
        region: &str,
        access_key_id: &str,
        secret_access_key: SecretString,
    ) -> Result<Self, KrishivError> {
        if !is_region(region) {
            return Err(invalid("region", "it is not an AWS region name"));
        }
        let suffix = if region.starts_with("cn-") { ".cn" } else { "" };
        let endpoint = Url::parse(&format!("https://glue.{region}.amazonaws.com{suffix}"))
            .map_err(|_| invalid("region", "it is not an AWS region name"))?;
        let config = Self {
            region: region.to_owned(),
            endpoint,
            access_key_id: access_key_id.to_owned(),
            secret_access_key,
            session_token: None,
            catalog_id: None,
            allow_http: false,
            limits: GlueLimits::default(),
            egress: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// Checks the declaration before anything connects: region, keys, endpoint scheme and shape, limits.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for `http://` without `allow_http`; `KRISHIV_INVALID_CONFIG_VALUE` otherwise.
    pub fn validate(&self) -> Result<(), KrishivError> {
        if !is_region(&self.region) {
            return Err(invalid("region", "it is not an AWS region name"));
        }
        if self.access_key_id.is_empty()
            || self.access_key_id.len() > 128
            || !self
                .access_key_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric())
        {
            return Err(invalid("accessKeyId", "it is not an access key id"));
        }
        if self.secret_access_key.expose_secret().is_empty() {
            return Err(invalid("secretAccessKey", "it must not be empty"));
        }
        if let Some(id) = &self.catalog_id
            && (id.is_empty() || id.len() > 255 || id.chars().any(char::is_control))
        {
            return Err(invalid("catalogId", "it is not a catalog id"));
        }
        match self.endpoint.scheme() {
            "https" => {}
            "http" if self.allow_http => {}
            "http" => {
                return Err(KrishivUnsupportedFeature {
                    feature: "a plain `http://` AWS Glue `endpoint` (the signed request would cross the network in clear)"
                        .to_owned(),
                }
                .into());
            }
            _ => return Err(invalid("endpoint", "it is not an http(s) URL")),
        }
        if self.endpoint.host_str().is_none_or(str::is_empty) {
            return Err(invalid("endpoint", "it has no host"));
        }
        if !self.endpoint.username().is_empty() || self.endpoint.password().is_some() {
            return Err(invalid("endpoint", "it carries credentials"));
        }
        if self.endpoint.query().is_some() || self.endpoint.fragment().is_some() {
            return Err(invalid("endpoint", "it carries a query or fragment"));
        }
        if self.endpoint.path() != "/" && !self.endpoint.path().is_empty() {
            return Err(invalid("endpoint", "it carries a path"));
        }
        self.limits.validate()
    }
}

/// Whether `region` is shaped like an AWS region: lower-case letters, digits and `-`, at most 32 characters.
fn is_region(region: &str) -> bool {
    !region.is_empty()
        && region.len() <= 32
        && region
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Whether `name` is one Glue name Krishiv will send: not empty, at most 255 characters, no control characters.
pub(crate) fn is_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 255 && !name.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn config() -> GlueConfig {
        GlueConfig::new("us-east-1", "AKIDEXAMPLE", SecretString::new("sekret")).unwrap()
    }

    #[test]
    fn the_default_endpoint_names_the_region_and_is_https() {
        let c = config();
        assert_eq!(c.endpoint.as_str(), "https://glue.us-east-1.amazonaws.com/");
        c.validate().unwrap();
    }

    #[test]
    fn plain_http_needs_the_declaration_to_allow_it() {
        let mut c = config();
        c.endpoint = Url::parse("http://localhost:4566").unwrap();
        assert_eq!(
            c.validate().unwrap_err().error_class(),
            ErrorClass::KrishivUnsupportedFeature
        );
        c.allow_http = true;
        c.validate().unwrap();
    }

    #[test]
    fn malformed_declarations_are_refused() {
        for endpoint in [
            "ftp://h",
            "https://u:p@h",
            "https://h/?t=1",
            "https://h/#f",
            "https://h/prefix",
        ] {
            let mut c = config();
            c.endpoint = Url::parse(endpoint).unwrap();
            assert!(c.validate().is_err(), "{endpoint}");
        }
        for region in ["", "US-EAST-1", "us east", "../x"] {
            assert!(
                GlueConfig::new(region, "AKID", SecretString::new("s")).is_err(),
                "{region:?}"
            );
        }
        let mut c = config();
        c.access_key_id = "bad key".into();
        assert!(c.validate().is_err());
        let mut c = config();
        c.secret_access_key = SecretString::new("");
        assert!(c.validate().is_err());
        let mut c = config();
        c.limits.max_depth = 129;
        assert!(c.validate().is_err());
    }

    #[test]
    fn the_keys_are_not_printed() {
        let mut c = config();
        c.session_token = Some(SecretString::new("tok-secret"));
        let shown = format!("{c:?}");
        assert!(
            !shown.contains("sekret") && !shown.contains("tok-secret"),
            "{shown}"
        );
    }
}
