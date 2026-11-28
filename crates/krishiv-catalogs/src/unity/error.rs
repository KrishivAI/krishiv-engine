// SPDX-License-Identifier: Apache-2.0
//! What a call to a Unity Catalog server can fail with, and the Krishiv error each becomes (design P3-lakehouse §5.6):
//! a limit is `KRISHIV_CATALOG_LIMIT_EXCEEDED`, anything else `KRISHIV_CATALOG_FAILURE`, both naming the catalog and
//! the relation, with the detail in the log only (cut short). A server's own message never reaches the error text.

use krishiv_common::error::{KrishivCatalogFailure, KrishivCatalogLimitExceeded, KrishivError};

use super::config::UnityLimits;

/// Which limit a call ran into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnityLimit {
    /// The response body is larger than `max_response_bytes`.
    Bytes,
    /// The JSON nests deeper than `max_depth`.
    Depth,
    /// A listing returned more than `max_list_entries` entries over its pages.
    Entries,
    /// A listing took more than `max_pages` pages.
    Pages,
    /// A table has more columns than `max_columns`.
    Columns,
}

/// What a call can fail with.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnityError {
    /// The call did not finish within `request_timeout`.
    Timeout,
    /// A limit of [`UnityLimits`] was exceeded: nothing was loaded.
    Limit(UnityLimit),
    /// The server answered with a redirect, which is never followed.
    Redirect(u16),
    /// The server answered with an error status; `code` and `message` are the body's `error_code` and `message`.
    Status {
        /// The HTTP status.
        status: u16,
        /// The body's `error_code`, when it had one.
        code: Option<String>,
        /// The body's `message`, when it had one (log only).
        message: Option<String>,
    },
    /// The response is not what the API says (bad JSON, a missing field, a compressed body, a listing that repeats
    /// its page token).
    Protocol(String),
    /// Connecting or reading failed.
    Transport(String),
}

/// Longest detail a log line carries.
const LOGGED_DETAIL_CHARS: usize = 300;

pub(crate) fn shortened(text: &str) -> String {
    let mut out: String = text.chars().take(LOGGED_DETAIL_CHARS).collect();
    if text.chars().count() > LOGGED_DETAIL_CHARS {
        out.push('…');
    }
    out
}

pub(crate) fn quoted(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

impl UnityError {
    /// A short description for the log; never the token.
    #[must_use]
    pub fn detail(&self) -> String {
        match self {
            Self::Timeout => "the request timed out".to_owned(),
            Self::Limit(l) => format!("limit exceeded: {l:?}"),
            Self::Redirect(s) => format!("a redirect ({s}), which is not followed"),
            Self::Status {
                status,
                code,
                message,
            } => format!(
                "HTTP {status} {}: {}",
                code.as_deref().unwrap_or("(no error code)"),
                shortened(message.as_deref().unwrap_or(""))
            ),
            Self::Protocol(d) => format!("protocol: {}", shortened(d)),
            Self::Transport(d) => format!("transport: {}", shortened(d)),
        }
    }

    /// The Krishiv error for this failure of `catalog` while it loaded `relation` (a quoted name). Logs the detail
    /// on `krishiv::catalog`.
    #[must_use]
    pub fn into_error(self, catalog: &str, relation: &str, limits: &UnityLimits) -> KrishivError {
        if let Self::Limit(limit) = self {
            let (what, value, config) = match limit {
                UnityLimit::Bytes => (
                    "bytes in one response",
                    limits.max_response_bytes.to_string(),
                    "a higher `krishiv.catalog.maxMetadataBytes`",
                ),
                UnityLimit::Depth => (
                    "the nesting depth of a response",
                    limits.max_depth.to_string(),
                    "a higher depth limit, which is not a setting yet",
                ),
                UnityLimit::Entries => (
                    "entries in one listing",
                    limits.max_list_entries.to_string(),
                    "a higher listing limit, which is not a setting yet",
                ),
                UnityLimit::Pages => (
                    "pages in one listing",
                    limits.max_pages.to_string(),
                    "a higher page limit, which is not a setting yet",
                ),
                UnityLimit::Columns => (
                    "columns in one table",
                    limits.max_columns.to_string(),
                    "`krishiv.catalog.maxColumns`",
                ),
            };
            return KrishivCatalogLimitExceeded {
                source: format!(
                    "The response of the Unity catalog {} for {relation}",
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
            "a Unity catalog call failed"
        );
        KrishivCatalogFailure {
            catalog: quoted(catalog),
            relation_name: relation.to_owned(),
            reason: "the call failed; the server log has the detail (target `krishiv::catalog`)"
                .to_owned(),
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    #[test]
    fn limits_name_the_limit_and_the_rest_name_only_the_catalog_and_the_relation() {
        let limits = UnityLimits::default();
        let e = UnityError::Limit(UnityLimit::Bytes).into_error("uc", "`s`.`t`", &limits);
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLimitExceeded);
        assert!(
            e.to_string().contains("krishiv.catalog.maxMetadataBytes"),
            "{e}"
        );
        for err in [
            UnityError::Timeout,
            UnityError::Redirect(302),
            UnityError::Status {
                status: 500,
                code: Some("X".into()),
                message: Some("secret=hunter2".into()),
            },
            UnityError::Protocol("p".into()),
            UnityError::Transport("t".into()),
        ] {
            let e = err.into_error("uc", "`s`.`t`", &limits);
            assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure);
            let text = e.to_string();
            assert!(text.contains("`uc`") && text.contains("`s`.`t`"), "{text}");
            assert!(!text.contains("hunter2"), "{text}");
        }
    }

    #[test]
    fn details_are_cut_short() {
        let d = UnityError::Protocol("x".repeat(10_000)).detail();
        assert!(d.chars().count() < 400, "{}", d.len());
    }
}
