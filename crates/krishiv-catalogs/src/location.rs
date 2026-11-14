// SPDX-License-Identifier: Apache-2.0
//! Locations a catalog returns (design P3-lakehouse §5.6, review S19, TM T-C2). A catalog is input, not a trusted
//! party: the storage location of a table it names is checked before any request is made, and a refusal is a fault
//! of the catalog, not of the user: `KRISHIV_CATALOG_LOCATION_REFUSED`, which names the catalog and the table and
//! never the location, and is logged as a catalog fault (not an authorization denial).
//!
//! This module holds the rules that need no allowlist: the scheme, credentials in the URL, escapes from the
//! location, malformed text. The confinement allowlists of ADR 0046 §3a (`krishiv.io.local.allowedRoots`,
//! `krishiv.io.objectStore.allowedPrefixes`) and the private-address rule of the egress policy (T067) join them
//! through [`LocationPolicy`] when they exist; a catalog never widens either list.

use krishiv_common::error::{KrishivCatalogLocationRefused, KrishivError};

/// Which rule refused a location. The name goes in the log record and the error's `<rule>`; the location itself
/// never does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LocationRule {
    /// The catalog returned nothing, or only blanks.
    Empty,
    /// Control characters or a backslash: not a location Krishiv would build.
    Malformed,
    /// No `scheme://`: a relative or bare path is not a place a remote catalog may point at.
    NoScheme,
    /// `file://` from a remote catalog: it would read the server's own disk.
    FileFromRemoteCatalog,
    /// A scheme that is not one of the object stores the catalog may name.
    SchemeNotAllowed,
    /// User information in the URL (`s3://key:secret@bucket/...`): credentials come from the store's own
    /// configuration, never from a catalog.
    CredentialsInLocation,
    /// A `..` segment (plain or percent-encoded), or an encoded slash or backslash inside a segment: the location
    /// would leave the prefix it appears to name once something decodes it.
    PathEscape,
    /// No host or bucket after `scheme://` (`s3:///key`).
    MissingHost,
}

impl LocationRule {
    /// The rule's name as the log and the error print it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Empty => "empty location",
            Self::Malformed => "malformed location",
            Self::NoScheme => "no scheme",
            Self::FileFromRemoteCatalog => "file location from a remote catalog",
            Self::SchemeNotAllowed => "scheme not allowed",
            Self::CredentialsInLocation => "credentials in the location",
            Self::PathEscape => "path escape",
            Self::MissingHost => "no bucket or host",
        }
    }
}

/// What decides whether a catalog-supplied location may be used. The allowlists of ADR 0046 §3a and the egress
/// policy implement it when they land; [`RemoteLocationPolicy`] is the part that needs neither.
pub trait LocationPolicy {
    /// Passes a location, or names the rule that refuses it.
    ///
    /// # Errors
    /// The [`LocationRule`] that refused `location`.
    fn check(&self, location: &str) -> Result<(), LocationRule>;
}

/// The schemes a remote catalog may name by default: the object stores Krishiv reads.
pub const DEFAULT_REMOTE_SCHEMES: &[&str] = &["s3", "s3a", "gs", "abfs", "abfss", "az"];

/// The rules for a **remote** catalog (Hive metastore, Unity, Iceberg REST): object-store schemes only, no
/// credentials, no escapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteLocationPolicy {
    schemes: Vec<String>,
}

impl Default for RemoteLocationPolicy {
    fn default() -> Self {
        Self {
            schemes: DEFAULT_REMOTE_SCHEMES
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }
}

impl RemoteLocationPolicy {
    /// The default policy with `schemes` as the only schemes allowed (lower case, no `://`).
    #[must_use]
    pub fn with_schemes<I, S>(schemes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self {
            schemes: schemes
                .into_iter()
                .map(|s| s.as_ref().to_ascii_lowercase())
                .collect(),
        }
    }
}

impl LocationPolicy for RemoteLocationPolicy {
    fn check(&self, location: &str) -> Result<(), LocationRule> {
        if location.trim().is_empty() {
            return Err(LocationRule::Empty);
        }
        if location.chars().any(|c| c.is_control() || c == '\\') {
            return Err(LocationRule::Malformed);
        }
        let Some((scheme, rest)) = location.split_once("://") else {
            return Err(LocationRule::NoScheme);
        };
        let scheme = scheme.to_ascii_lowercase();
        if scheme == "file" {
            return Err(LocationRule::FileFromRemoteCatalog);
        }
        if !self.schemes.contains(&scheme) {
            return Err(LocationRule::SchemeNotAllowed);
        }
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        if authority.is_empty() || authority.starts_with('@') {
            return Err(LocationRule::MissingHost);
        }
        if let Some((userinfo, _host)) = authority.split_once('@') {
            // `abfs[s]://container@account.dfs.core.windows.net` names the container before the `@`: that is
            // an address, not a credential. A password, or user information on any other scheme, is one.
            let names_a_container = matches!(scheme.as_str(), "abfs" | "abfss");
            if userinfo.contains(':') || !names_a_container {
                return Err(LocationRule::CredentialsInLocation);
            }
        }
        let escapes = path.split(['/', '?', '#']).any(|segment| {
            let segment = segment.to_ascii_lowercase();
            segment == ".."
                || segment == "%2e%2e"
                || segment == ".%2e"
                || segment == "%2e."
                || segment.contains("%2f")
                || segment.contains("%5c")
        });
        if escapes {
            return Err(LocationRule::PathEscape);
        }
        Ok(())
    }
}

/// Checks a location `catalog` returned for `relation_name` (the quoted table name) before anything reads it.
///
/// # Errors
/// `KRISHIV_CATALOG_LOCATION_REFUSED` naming the catalog, the table and the rule, never the location. A refusal is
/// logged on `krishiv::catalog` as a catalog fault; it is not an authorization denial and counts against no
/// principal.
pub fn check_location(
    catalog: &str,
    relation_name: &str,
    location: &str,
    policy: &dyn LocationPolicy,
) -> Result<(), KrishivError> {
    policy.check(location).map_err(|rule| {
        tracing::warn!(
            target: "krishiv::catalog",
            catalog,
            table = relation_name,
            rule = rule.name(),
            "a catalog returned a location Krishiv refuses"
        );
        KrishivCatalogLocationRefused {
            catalog: format!("`{}`", catalog.replace('`', "``")),
            relation_name: relation_name.to_owned(),
            rule: rule.name().to_owned(),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn rule(location: &str) -> Result<(), LocationRule> {
        RemoteLocationPolicy::default().check(location)
    }

    #[test]
    fn object_store_locations_pass() {
        for ok in [
            "s3://bucket/warehouse/db/t",
            "S3A://bucket/wh/t/",
            "gs://b/x",
            "abfss://container@account.dfs.core.windows.net/t",
            "abfs://container@account.dfs.core.windows.net",
            "az://c/t",
            "s3://bucket",
        ] {
            assert_eq!(rule(ok), Ok(()), "{ok}");
        }
    }

    #[test]
    fn each_refused_shape_names_its_rule() {
        let cases = [
            ("", LocationRule::Empty),
            ("   ", LocationRule::Empty),
            ("s3://b/a\nb", LocationRule::Malformed),
            ("s3://b\\x", LocationRule::Malformed),
            ("/etc/passwd", LocationRule::NoScheme),
            ("warehouse/t", LocationRule::NoScheme),
            ("file:///etc", LocationRule::FileFromRemoteCatalog),
            ("FILE://host/x", LocationRule::FileFromRemoteCatalog),
            (
                "http://169.254.169.254/latest",
                LocationRule::SchemeNotAllowed,
            ),
            ("hdfs://nn/t", LocationRule::SchemeNotAllowed),
            ("ftp://h/t", LocationRule::SchemeNotAllowed),
            (
                "s3://key:secret@bucket/t",
                LocationRule::CredentialsInLocation,
            ),
            ("s3://user@bucket/t", LocationRule::CredentialsInLocation),
            (
                "abfss://container:key@account.dfs.core.windows.net/t",
                LocationRule::CredentialsInLocation,
            ),
            ("s3://bucket/a/../b", LocationRule::PathEscape),
            ("s3://bucket/a/%2E%2e/b", LocationRule::PathEscape),
            ("s3://bucket/a/.%2e/b", LocationRule::PathEscape),
            ("gs://b/../x", LocationRule::PathEscape),
            ("s3://bucket/a%2F..%2Fb", LocationRule::PathEscape),
            ("s3://bucket/a%5cb", LocationRule::PathEscape),
            ("s3:///key", LocationRule::MissingHost),
            ("gs://", LocationRule::MissingHost),
        ];
        for (location, expected) in cases {
            assert_eq!(rule(location), Err(expected), "{location:?}");
        }
    }

    #[test]
    fn a_dot_inside_a_name_is_not_an_escape() {
        assert_eq!(rule("s3://bucket/a..b/c...d/.hidden"), Ok(()));
    }

    #[test]
    fn the_allowed_schemes_can_be_narrowed() {
        let only_s3 = RemoteLocationPolicy::with_schemes(["S3"]);
        assert_eq!(only_s3.check("s3://b/t"), Ok(()));
        assert_eq!(
            only_s3.check("gs://b/t"),
            Err(LocationRule::SchemeNotAllowed)
        );
    }

    #[test]
    fn a_refusal_names_the_catalog_and_the_table_and_never_the_location() {
        let location = "file:///etc/shadow";
        let e = check_location(
            "uc",
            "`uc`.`main`.`t`",
            location,
            &RemoteLocationPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLocationRefused);
        assert_eq!(e.sql_state(), Some("58030"));
        let text = e.to_string();
        assert!(
            text.contains("`uc`") && text.contains("`uc`.`main`.`t`"),
            "{text}"
        );
        assert!(
            text.contains("file location from a remote catalog"),
            "{text}"
        );
        assert!(!text.contains("shadow") && !text.contains("/etc"), "{text}");
    }

    #[test]
    fn an_accepted_location_is_no_error() {
        check_location(
            "hms",
            "`hms`.`db`.`t`",
            "s3a://bucket/db/t",
            &RemoteLocationPolicy::default(),
        )
        .unwrap();
    }

    #[test]
    fn a_catalog_name_is_quoted_so_it_cannot_inject_text() {
        let e = check_location("a`b", "`x`", "", &RemoteLocationPolicy::default()).unwrap_err();
        assert!(e.to_string().contains("`a``b`"), "{e}");
    }
}
