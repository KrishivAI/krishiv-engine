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
    /// `storage = vended`: the catalog vended no credential for the table (design P3-T067 §4.1, no fallback).
    NoVendedCredential,
    /// `storage = vended`: a file of the table lies outside every prefix the catalog vended credentials for.
    OutsideVendedPrefixes,
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
            Self::NoVendedCredential => "no vended credential",
            Self::OutsideVendedPrefixes => "outside the vended credential prefixes",
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

    /// Passes a bare scheme (`s3`, `abfss`), as an Iceberg storage-credential prefix names one (T067 §4.1): may
    /// this catalog's locations use it at all? The default checks a placeholder location of that scheme, which a
    /// policy that confines hosts or buckets should override.
    ///
    /// # Errors
    /// The [`LocationRule`] that refused the scheme.
    fn check_scheme(&self, scheme: &str) -> Result<(), LocationRule> {
        self.check(&format!("{}://scheme-prefix/", scheme.to_ascii_lowercase()))
    }
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
    fn check_scheme(&self, scheme: &str) -> Result<(), LocationRule> {
        let scheme = scheme.to_ascii_lowercase();
        if scheme == "file" {
            return Err(LocationRule::FileFromRemoteCatalog);
        }
        if !self.schemes.contains(&scheme) {
            return Err(LocationRule::SchemeNotAllowed);
        }
        Ok(())
    }

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
    policy
        .check(location)
        .map_err(|rule| refused(catalog, relation_name, rule))
}

/// As [`check_location`], for a bare scheme (an Iceberg storage-credential prefix such as `s3`;
/// [`LocationPolicy::check_scheme`]).
///
/// `scheme` must be a scheme as RFC 3986 §3.1 spells one (a letter, then letters, digits, `+`, `-` or `.`); anything
/// else is refused as empty or malformed before the policy sees it.
///
/// # Errors
/// `KRISHIV_CATALOG_LOCATION_REFUSED`, as [`check_location`].
pub fn check_scheme(
    catalog: &str,
    relation_name: &str,
    scheme: &str,
    policy: &dyn LocationPolicy,
) -> Result<(), KrishivError> {
    is_scheme(scheme)
        .and_then(|()| policy.check_scheme(scheme))
        .map_err(|rule| refused(catalog, relation_name, rule))
}

fn is_scheme(scheme: &str) -> Result<(), LocationRule> {
    let mut chars = scheme.chars();
    match chars.next() {
        None => Err(LocationRule::Empty),
        Some(first)
            if first.is_ascii_alphabetic()
                && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')) =>
        {
            Ok(())
        }
        Some(_) => Err(LocationRule::Malformed),
    }
}

pub(crate) fn refused(catalog: &str, relation_name: &str, rule: LocationRule) -> KrishivError {
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
    fn a_bare_scheme_is_checked_against_the_allowed_schemes() {
        let p = RemoteLocationPolicy::default();
        assert_eq!(p.check_scheme("S3"), Ok(()));
        assert_eq!(p.check_scheme("abfss"), Ok(()));
        assert_eq!(
            p.check_scheme("file"),
            Err(LocationRule::FileFromRemoteCatalog)
        );
        assert_eq!(p.check_scheme("http"), Err(LocationRule::SchemeNotAllowed));
    }

    /// A policy with only `check`, so `check_scheme` is the trait's default.
    struct OnlyCheck(std::cell::RefCell<Vec<String>>);

    impl LocationPolicy for OnlyCheck {
        fn check(&self, location: &str) -> Result<(), LocationRule> {
            self.0.borrow_mut().push(location.to_owned());
            RemoteLocationPolicy::default().check(location)
        }
    }

    #[test]
    fn the_default_scheme_check_checks_a_placeholder_location_of_that_scheme() {
        let p = OnlyCheck(std::cell::RefCell::new(Vec::new()));
        assert_eq!(p.check_scheme("S3"), Ok(()));
        assert_eq!(p.check_scheme("hdfs"), Err(LocationRule::SchemeNotAllowed));
        assert_eq!(
            *p.0.borrow(),
            ["s3://scheme-prefix/", "hdfs://scheme-prefix/"]
        );
    }

    #[test]
    fn a_scheme_check_names_the_catalog_and_the_rule() {
        let policy = RemoteLocationPolicy::default();
        check_scheme("rest", "`rest`.`db`.`t`", "s3", &policy).unwrap();
        let e = check_scheme("rest", "`rest`.`db`.`t`", "file", &policy).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLocationRefused);
        let text = e.to_string();
        assert!(
            text.contains("`rest`.`db`.`t`")
                && text.contains("file location from a remote catalog"),
            "{text}"
        );
    }

    #[test]
    fn a_scheme_check_refuses_what_is_not_a_scheme() {
        let policy = RemoteLocationPolicy::default();
        let refused_by = |scheme: &str| {
            check_scheme("rest", "`t`", scheme, &policy)
                .unwrap_err()
                .to_string()
        };
        assert!(refused_by("").contains("empty location"));
        for not_a_scheme in ["s3://b", "3s", "s3/x", "s 3", "-s3"] {
            assert!(
                refused_by(not_a_scheme).contains("malformed location"),
                "{not_a_scheme:?}"
            );
        }
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
