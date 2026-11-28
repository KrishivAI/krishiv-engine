// SPDX-License-Identifier: Apache-2.0
//! What a call to an Iceberg REST catalog can fail with, and the Krishiv error each becomes (design P3-lakehouse
//! §5.6): a limit is `KRISHIV_CATALOG_LIMIT_EXCEEDED`, anything else `KRISHIV_CATALOG_FAILURE`, both naming the
//! catalog and the table or namespace, with the detail in the log only (cut short). A server's own message, and an
//! IdP's `error_description`, never reach the error text: either may carry what it should not. The error text does say the kind of failure: the HTTP status, and an
//! `ErrorModel.type` only when it is one of the spec's names.

use krishiv_common::error::{
    KrishivCatalogDelegationUnavailable, KrishivCatalogFailure, KrishivCatalogLimitExceeded,
    KrishivError,
};

use super::config::RestLimits;

/// Which limit a call ran into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestLimit {
    /// The response body is larger than `max_response_bytes`.
    Bytes,
    /// The JSON nests deeper than `max_depth`.
    Depth,
    /// A listing returned more than `max_list_entries` entries over its pages.
    Entries,
    /// A listing took more than `max_pages` pages.
    Pages,
}

/// What a call can fail with.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RestError {
    /// The call did not finish within `request_timeout` (token fetch included).
    Timeout,
    /// A limit of [`RestLimits`] was exceeded: nothing was loaded.
    Limit(RestLimit),
    /// The catalog answered with a redirect, which is never followed.
    Redirect(u16),
    /// The catalog answered with an error status. `kind` and `message` are the `ErrorModel`'s, for the log.
    Status {
        /// The HTTP status.
        status: u16,
        /// The `ErrorModel.type`, when the body had one.
        kind: Option<String>,
        /// The `ErrorModel.message`, when the body had one (log only).
        message: Option<String>,
    },
    /// The token endpoint refused the client credentials; `error` is the OAuth `error` code only.
    TokenRefused {
        /// The HTTP status.
        status: u16,
        /// The OAuth2 `error` code (`invalid_client`, ...), never `error_description`.
        error: Option<String>,
    },
    /// The response is not what the protocol says (bad JSON, a missing field, a compressed body, an unsafe
    /// `prefix` from `/v1/config`, a listing that repeats its page token).
    Protocol(String),
    /// Connecting or reading failed.
    Transport(String),
    /// The HTTP client could not be built, so nothing was ever sent (SK3 E1).
    Client {
        /// The builder's reason, for the log.
        detail: String,
        /// The system trust store is empty (`krishiv_store::is_missing_tls_roots`).
        no_tls_roots: bool,
    },
    /// The egress policy refused the host, before anything was sent (SK3 E1).
    Egress(krishiv_store::egress::EgressRefusal),
    /// A delegated catalog has no usable user token, or the IdP refused the exchange (T067 §3.3). The reason is
    /// Krishiv's own text: at most an OAuth `error` code, never a token or `error_description`.
    Delegation(String),
    /// A token-exchange limit was reached.
    DelegationLimit {
        /// What is counted.
        what: &'static str,
        /// The limit.
        value: usize,
        /// The key that sets it.
        config: &'static str,
    },
}

/// The `ErrorModel.type` names the Iceberg REST spec and the Java client define: the only server text an error shows.
const SPEC_ERROR_TYPES: &[&str] = &[
    "BadRequestException",
    "NotAuthorizedException",
    "ForbiddenException",
    "NoSuchNamespaceException",
    "NoSuchTableException",
    "NoSuchViewException",
    "NoSuchIcebergTableException",
    "AlreadyExistsException",
    "NamespaceNotEmptyException",
    "CommitFailedException",
    "CommitStateUnknownException",
    "ValidationException",
    "UnsupportedOperationException",
    "ServiceFailureException",
    "ServiceUnavailableException",
    "InternalServerError",
    "RESTException",
    "AuthorizationExpiredException",
];

/// Longest detail a log line carries.
const LOGGED_DETAIL_CHARS: usize = 300;

fn shortened(text: &str) -> String {
    let mut out: String = text.chars().take(LOGGED_DETAIL_CHARS).collect();
    if text.chars().count() > LOGGED_DETAIL_CHARS {
        out.push('…');
    }
    out
}

pub(crate) fn quoted(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

impl RestError {
    /// A short description for the log; never a token, never a server's free text beyond the cut.
    #[must_use]
    pub fn detail(&self) -> String {
        match self {
            Self::Timeout => "the request timed out".to_owned(),
            Self::Limit(l) => format!("limit exceeded: {l:?}"),
            Self::Redirect(s) => format!("a redirect ({s}), which is not followed"),
            Self::Status {
                status,
                kind,
                message,
            } => format!(
                "HTTP {status} {}: {}",
                kind.as_deref().unwrap_or("(no error type)"),
                shortened(message.as_deref().unwrap_or(""))
            ),
            Self::TokenRefused { status, error } => format!(
                "the token endpoint refused the client credentials: HTTP {status} {}",
                error.as_deref().unwrap_or("(no error code)")
            ),
            Self::Protocol(d) => format!("protocol: {}", shortened(d)),
            Self::Transport(d) => format!("transport: {}", shortened(d)),
            Self::Client { detail, .. } => format!("HTTP client: {}", shortened(detail)),
            Self::Egress(refusal) => format!("egress policy: {refusal}"),
            Self::Delegation(d) => format!("delegation: {}", shortened(d)),
            Self::DelegationLimit { what, value, .. } => format!("limit exceeded: {value} {what}"),
        }
    }

    /// What a user may be told: Krishiv's own words for the kind of failure, and the HTTP status. Never a
    /// server's message; an `ErrorModel.type` only when it is a plain identifier.
    fn reason(&self) -> String {
        match self {
            Self::Timeout => "the request timed out".to_owned(),
            Self::Limit(_) => "a limit was exceeded".to_owned(),
            Self::Redirect(s) => {
                format!("the catalog answered with a redirect (HTTP {s}), which is not followed")
            }
            Self::Status { status, kind, .. } => {
                let kind = kind.as_deref().filter(|k| SPEC_ERROR_TYPES.contains(k));
                match kind {
                    Some(k) => format!("the catalog answered HTTP {status} ({k})"),
                    None => format!("the catalog answered HTTP {status}"),
                }
            }
            Self::TokenRefused { status, .. } => {
                format!("the token endpoint refused the client credentials (HTTP {status})")
            }
            Self::Protocol(_) => "the answer is not what the Iceberg REST protocol says".to_owned(),
            Self::Transport(_) => "the connection failed or was cut".to_owned(),
            Self::Client {
                no_tls_roots: true, ..
            } => "the HTTP client could not be built: no TLS root certificates were found \
                  (install the CA certificates, or point SSL_CERT_FILE at a bundle)"
                .to_owned(),
            Self::Client { .. } => "the HTTP client could not be built".to_owned(),
            Self::Egress(_) => "the egress policy refused the host".to_owned(),
            Self::Delegation(_) | Self::DelegationLimit { .. } => {
                "delegated access failed".to_owned()
            }
        }
    }

    /// The Krishiv error for this failure of `catalog` while it loaded `relation` (a quoted table or namespace
    /// name). Logs the detail on `krishiv::catalog`.
    #[must_use]
    pub fn into_error(self, catalog: &str, relation: &str, limits: &RestLimits) -> KrishivError {
        match self {
            Self::Delegation(reason) => {
                return KrishivCatalogDelegationUnavailable {
                    catalog: quoted(catalog),
                    reason,
                }
                .into();
            }
            Self::DelegationLimit {
                what,
                value,
                config,
            } => {
                return KrishivCatalogLimitExceeded {
                    source: format!(
                        "Delegated access to the catalog {} for this user",
                        quoted(catalog)
                    ),
                    limit: what.to_owned(),
                    value: value.to_string(),
                    config: format!("a higher `{config}`"),
                }
                .into();
            }
            _ => {}
        }
        if let Self::Limit(limit) = self {
            let (what, value, config) = match limit {
                RestLimit::Bytes => (
                    "bytes in one response",
                    limits.max_response_bytes.to_string(),
                    "a higher `krishiv.catalog.maxMetadataBytes`",
                ),
                RestLimit::Depth => (
                    "the nesting depth of a response",
                    limits.max_depth.to_string(),
                    "a higher depth limit, which is not a setting yet",
                ),
                RestLimit::Entries => (
                    "entries in one listing",
                    limits.max_list_entries.to_string(),
                    "a higher listing limit, which is not a setting yet",
                ),
                RestLimit::Pages => (
                    "pages in one listing",
                    limits.max_pages.to_string(),
                    "a higher page limit, which is not a setting yet",
                ),
            };
            return KrishivCatalogLimitExceeded {
                source: format!(
                    "The response of the Iceberg REST catalog {} for {relation}",
                    quoted(catalog)
                ),
                limit: what.to_owned(),
                value,
                config: config.to_owned(),
            }
            .into();
        }
        tracing::warn!(
            target: "krishiv::catalog",
            catalog,
            table = relation,
            detail = self.detail().as_str(),
            "an Iceberg REST catalog call failed"
        );
        let reason = self.reason();
        KrishivCatalogFailure {
            catalog: quoted(catalog),
            relation_name: relation.to_owned(),
            reason,
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    #[test]
    fn a_failure_says_what_kind_it_was_and_never_the_servers_text() {
        let limits = RestLimits::default();
        let text = RestError::Status {
            status: 500,
            kind: Some("ServiceFailureException".into()),
            message: Some("secret=hunter2".into()),
        }
        .into_error("rest", "`db`.`t`", &limits)
        .to_string();
        assert!(
            text.contains("HTTP 500 (ServiceFailureException)"),
            "{text}"
        );
        assert!(!text.contains("hunter2"), "{text}");
        let text = RestError::Status {
            status: 400,
            kind: Some("a b; drop".into()),
            message: None,
        }
        .into_error("rest", "`db`.`t`", &limits)
        .to_string();
        assert!(
            text.contains("HTTP 400") && !text.contains("drop"),
            "{text}"
        );
        let text = RestError::Timeout
            .into_error("rest", "`db`.`t`", &limits)
            .to_string();
        assert!(text.contains("timed out"), "{text}");
    }

    #[test]
    fn limits_name_the_limit_and_the_key_and_the_rest_name_only_the_catalog() {
        let limits = RestLimits::default();
        let e = RestError::Limit(RestLimit::Bytes).into_error("rest", "`db`.`t`", &limits);
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLimitExceeded);
        let text = e.to_string();
        assert!(
            text.contains("krishiv.catalog.maxMetadataBytes") && text.contains("`rest`"),
            "{text}"
        );

        for err in [
            RestError::Timeout,
            RestError::Redirect(302),
            RestError::Status {
                status: 500,
                kind: Some("X".into()),
                message: Some("secret=hunter2".into()),
            },
            RestError::TokenRefused {
                status: 401,
                error: Some("invalid_client".into()),
            },
            RestError::Protocol("p".into()),
            RestError::Transport("t".into()),
        ] {
            let e = err.into_error("rest", "`db`.`t`", &limits);
            assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure);
            let text = e.to_string();
            assert!(
                text.contains("`rest`") && text.contains("`db`.`t`"),
                "{text}"
            );
            assert!(!text.contains("hunter2"), "{text}");
        }
    }

    #[test]
    fn details_are_cut_short() {
        let long = "x".repeat(10_000);
        let d = RestError::Protocol(long).detail();
        assert!(d.chars().count() < 400, "{}", d.len());
    }
}
