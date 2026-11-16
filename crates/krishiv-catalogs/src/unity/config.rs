// SPDX-License-Identifier: Apache-2.0
//! The declaration of one Unity Catalog: the server, the UC catalog it serves, a static bearer token and the limits
//! (design P3-lakehouse §3, §5.6). Per-user tokens and vended credentials are T067, not here.

use std::time::Duration;

use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue, KrishivUnsupportedFeature};
use krishiv_common::secret::SecretString;
use url::Url;

/// The limits of one call, checked before or while a response is parsed. `max_response_bytes` is
/// `krishiv.catalog.maxMetadataBytes`, `request_timeout` is `krishiv.catalog.requestTimeout`.
#[derive(Debug, Clone)]
pub struct UnityLimits {
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

impl Default for UnityLimits {
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

impl UnityLimits {
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a zero limit, a zero timeout or a depth over 128.
    pub fn validate(&self) -> Result<(), KrishivError> {
        if self.request_timeout.is_zero() {
            return Err(invalid("requestTimeout", "it must be more than zero"));
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
        if self.max_depth > 128 {
            return Err(invalid("maxDepth", "it must be at most 128"));
        }
        Ok(())
    }
}

/// One Unity Catalog as its declaration describes it.
#[derive(Debug, Clone)]
pub struct UnityConfig {
    /// The server (`https://host[:port][/prefix]`); requests go to `<uri>/api/2.1/unity-catalog/...`.
    pub uri: Url,
    /// The UC catalog this Krishiv catalog serves (one Krishiv catalog per UC catalog).
    pub catalog: String,
    /// A static bearer token, if the server wants one.
    pub token: Option<SecretString>,
    /// Allow `http://`. Off by default: a token over plain HTTP is a token anyone on the path can read. Local test
    /// servers turn it on.
    pub allow_http: bool,
    /// The limits.
    pub limits: UnityLimits,
}

impl UnityConfig {
    /// A declaration for the UC catalog `catalog` at `uri`, no token, default limits.
    ///
    /// # Errors
    /// `uri` is not a URL, or `catalog` is not a usable name.
    pub fn new(uri: &str, catalog: &str) -> Result<Self, KrishivError> {
        let uri = Url::parse(uri).map_err(|_| invalid("uri", "it is not a URL"))?;
        if !is_name(catalog) {
            return Err(invalid("catalog", "it is not a usable catalog name"));
        }
        Ok(Self {
            uri,
            catalog: catalog.to_owned(),
            token: None,
            allow_http: false,
            limits: UnityLimits::default(),
        })
    }

    /// Checks the declaration before anything connects: scheme, credentials in the URL, limits.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for `http://` without `allow_http`; `KRISHIV_INVALID_CONFIG_VALUE` for a
    /// malformed URI or limit.
    pub fn validate(&self) -> Result<(), KrishivError> {
        match self.uri.scheme() {
            "https" => {}
            "http" if self.allow_http => {}
            "http" => {
                return Err(KrishivUnsupportedFeature {
                    feature: "a plain `http://` Unity Catalog `uri` (the token would cross the network in clear)"
                        .to_owned(),
                }
                .into());
            }
            _ => return Err(invalid("uri", "it is not an http(s) URL")),
        }
        if self.uri.host_str().is_none_or(str::is_empty) {
            return Err(invalid("uri", "it has no host"));
        }
        if !self.uri.username().is_empty() || self.uri.password().is_some() {
            return Err(invalid("uri", "it carries credentials"));
        }
        if self.uri.query().is_some() || self.uri.fragment().is_some() {
            return Err(invalid("uri", "it carries a query or fragment"));
        }
        if !is_name(&self.catalog) {
            return Err(invalid("catalog", "it is not a usable catalog name"));
        }
        self.limits.validate()
    }
}

/// Whether `name` is one UC name part Krishiv will put in a URL: not empty, not `.` or `..`, no `.` (it would make the
/// three-level `full_name` ambiguous), `/`, `\` or control characters.
pub(crate) fn is_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name != ".."
        && !name
            .chars()
            .any(|c| c.is_control() || matches!(c, '.' | '/' | '\\'))
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    #[test]
    fn plain_http_needs_the_declaration_to_allow_it() {
        let mut c = UnityConfig::new("http://localhost:8080", "unity").unwrap();
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
            "https://u:p@h/x",
            "https://h/x?t=1",
            "https://h/#f",
        ] {
            assert!(
                UnityConfig::new(uri, "unity").unwrap().validate().is_err(),
                "{uri}"
            );
        }
        assert!(UnityConfig::new("not a url", "unity").is_err());
        for name in ["", "..", "a.b", "a/b", "a\nb"] {
            assert!(UnityConfig::new("https://h", name).is_err(), "{name:?}");
        }
        let mut c = UnityConfig::new("https://h", "unity").unwrap();
        c.limits.max_depth = 0;
        assert!(c.validate().is_err());
        c.limits.max_depth = 129;
        assert!(c.validate().is_err());
    }

    #[test]
    fn the_token_is_not_printed() {
        let mut c = UnityConfig::new("https://h", "unity").unwrap();
        c.token = Some(SecretString::new("hunter2"));
        assert!(!format!("{c:?}").contains("hunter2"));
    }
}
