// SPDX-License-Identifier: Apache-2.0
//! Path identity for `remove_orphan_files` (design P3-T027 PR-36e §4.4, F4): one parse applied
//! to the strings of the valid set (manifests, metadata) and to the listed object paths, so a
//! file is kept whichever spelling its writer used. As Hadoop's `new Path(s).toUri()`: the
//! scheme is lower-cased, repeated `/` collapse, a trailing `/` goes, and `file:/x`,
//! `file:///x` and `/x` name one path. The path is compared as the raw key (no percent
//! decoding on either side: `p=a%20b` is the bytes the writer used).

use std::collections::BTreeMap;

/// A file location, split into the parts the comparison needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileIdentity {
    /// Lower-case; empty when the string had none (`/x`).
    pub(crate) scheme: String,
    /// The part after `//` up to the next `/`; empty when none (`file:/x`, `file:///x`).
    pub(crate) authority: String,
    /// Absolute, `/`-separated, no empty segment, no trailing `/`.
    pub(crate) path: String,
}

/// How a listed path relates to a valid one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Relation {
    /// The same file.
    Same,
    /// The same path under a different scheme or authority: `prefix_mismatch_mode` decides.
    PrefixMismatch,
    /// Another file.
    Different,
}

/// `equal_schemes` and `equal_authorities`: a key is replaced by its value before comparing.
#[derive(Debug, Clone, Default)]
pub(crate) struct Equivalences {
    schemes: BTreeMap<String, String>,
    authorities: BTreeMap<String, String>,
}

impl Equivalences {
    /// Iceberg's default `equal_schemes`: `s3n` and `s3a` are `s3`.
    pub(crate) fn defaults() -> Self {
        Self {
            schemes: [("s3n", "s3"), ("s3a", "s3")]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
            authorities: BTreeMap::new(),
        }
    }

    /// The defaults overridden by the call's maps (keys and values of schemes lower-cased).
    pub(crate) fn with(
        schemes: &BTreeMap<String, String>,
        authorities: &BTreeMap<String, String>,
    ) -> Self {
        let mut e = Self::defaults();
        for (k, v) in schemes {
            e.schemes.insert(k.to_lowercase(), v.to_lowercase());
        }
        for (k, v) in authorities {
            e.authorities.insert(k.clone(), v.clone());
        }
        e
    }

    fn scheme<'a>(&'a self, s: &'a str) -> &'a str {
        self.schemes.get(s).map_or(s, String::as_str)
    }

    fn authority<'a>(&'a self, a: &'a str) -> &'a str {
        self.authorities.get(a).map_or(a, String::as_str)
    }
}

impl FileIdentity {
    /// Reads `s`; `None` for a string that is not an absolute location: empty, a NUL, a
    /// relative path, a scheme with no path.
    pub(crate) fn parse(s: &str) -> Option<Self> {
        if s.is_empty() || s.contains('\0') {
            return None;
        }
        let (scheme, rest) = match s.find(':') {
            Some(i)
                if i > 0
                    && s[..i]
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
                    && s[..i].starts_with(|c: char| c.is_ascii_alphabetic()) =>
            {
                (s[..i].to_lowercase(), &s[i + 1..])
            }
            _ => (String::new(), s),
        };
        let (authority, path) = if let Some(after) = rest.strip_prefix("//") {
            match after.find('/') {
                Some(i) => (after[..i].to_owned(), &after[i..]),
                None => (after.to_owned(), ""),
            }
        } else {
            (String::new(), rest)
        };
        // An authority with no path is the root of its store.
        let path = if path.is_empty() && !authority.is_empty() {
            "/"
        } else {
            path
        };
        if !path.starts_with('/') {
            return None;
        }
        let segments: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        Some(Self {
            scheme,
            authority,
            path: format!("/{}", segments.join("/")),
        })
    }

    /// How `listed` relates to `self`, the valid side. As Java's `FileURI.uriComponentMatch`:
    /// an empty scheme or authority on the valid side matches any listed one; an empty one
    /// on the listed side against a non-empty valid one is a mismatch.
    pub(crate) fn relation(&self, listed: &Self, eq: &Equivalences) -> Relation {
        if self.path != listed.path {
            return Relation::Different;
        }
        let part = |valid: &str, listed: &str, map: &dyn Fn(&str) -> String| {
            valid.is_empty() || map(valid) == map(listed)
        };
        let scheme_ok = part(&self.scheme, &listed.scheme, &|s| eq.scheme(s).to_owned());
        let authority_ok = part(&self.authority, &listed.authority, &|a| {
            eq.authority(a).to_owned()
        });
        if scheme_ok && authority_ok {
            Relation::Same
        } else {
            Relation::PrefixMismatch
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> FileIdentity {
        FileIdentity::parse(s).unwrap_or_else(|| panic!("{s} must parse"))
    }

    #[test]
    fn file_spellings_are_one_path() {
        let a = id("file:/x/y");
        let b = id("file:///x/y");
        assert_eq!((&a.authority, &a.path), (&b.authority, &b.path));
        assert_eq!(id("/x/y").path, "/x/y");
        assert_eq!(id("/x/y").scheme, "");
    }

    #[test]
    fn slashes_collapse_and_a_trailing_one_goes() {
        assert_eq!(id("s3://b//a///c/").path, "/a/c");
        assert_eq!(id("s3://b/").path, "/");
        assert_eq!(id("S3://B/a").scheme, "s3");
        assert_eq!(id("S3://B/a").authority, "B");
    }

    #[test]
    fn escaped_partition_values_stay_as_written() {
        assert_eq!(id("s3://b/t/p=a%20b/f").path, "/t/p=a%20b/f");
        assert_ne!(id("s3://b/t/p=a%20b/f").path, id("s3://b/t/p=a b/f").path);
        assert_eq!(id("s3://b/t/p=a+b/f").path, "/t/p=a+b/f");
        assert_eq!(id("s3://b/t/p=é/f").path, "/t/p=é/f");
    }

    #[test]
    fn what_is_not_a_location_does_not_parse() {
        for s in ["", "a/b", "s3:rel", "x\0y"] {
            assert!(FileIdentity::parse(s).is_none(), "{s:?}");
        }
        // A scheme and authority with no path is the root.
        assert_eq!(id("s3://bucket").path, "/");
    }

    #[test]
    fn schemes_are_equal_by_the_map() {
        let eq = Equivalences::defaults();
        let valid = id("s3a://b/t/f");
        assert_eq!(valid.relation(&id("s3://b/t/f"), &eq), Relation::Same);
        assert_eq!(valid.relation(&id("s3n://b/t/f"), &eq), Relation::Same);
        assert_eq!(
            valid.relation(&id("gs://b/t/f"), &eq),
            Relation::PrefixMismatch
        );
        assert_eq!(valid.relation(&id("s3://b/t/g"), &eq), Relation::Different);
        let custom = Equivalences::with(
            &BTreeMap::from([("GS".to_owned(), "S3".to_owned())]),
            &BTreeMap::from([("other".to_owned(), "b".to_owned())]),
        );
        assert_eq!(
            id("s3://b/t/f").relation(&id("gs://other/t/f"), &custom),
            Relation::Same
        );
    }

    #[test]
    fn an_empty_valid_side_matches_and_an_empty_listed_side_mismatches() {
        let eq = Equivalences::defaults();
        assert_eq!(id("/x/f").relation(&id("file:/x/f"), &eq), Relation::Same);
        assert_eq!(
            id("file:/x/f").relation(&id("/x/f"), &eq),
            Relation::PrefixMismatch
        );
        assert_eq!(
            id("s3://b/f").relation(&id("s3:/f"), &eq),
            Relation::PrefixMismatch
        );
    }
}
