// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2's string collations (ADR 0043): the 32-bit id with Spark's bit
//! layout, the canonical name, and Spark's name grammar.
//!
//! Names are exactly Spark 4.2's: the table is
//! `tests/fixtures/spark-4.2.0/collations.tsv` (`SELECT * FROM collations()`
//! from a real run, `tests/scripts/gen_collations.py`). Ids are for use in
//! one process only: the ICU locale index is the locale's position in that
//! table, not Spark's, so an id is never written anywhere (names are).

use std::fmt;
use std::sync::LazyLock;

use krishiv_common::error::{CollationInvalidName, KrishivError};

use crate::error::Result;

const FIXTURE: &str = include_str!("../../../tests/fixtures/spark-4.2.0/collations.tsv");

/// Bit 29 of an id: set for an ICU collation.
const ICU_BIT: u32 = 1 << 29;
/// Bit 18: `RTRIM`.
const RTRIM_BIT: u32 = 1 << 18;
/// Bit 17 (ICU): case-insensitive.
const CI_BIT: u32 = 1 << 17;
/// Bit 16 (ICU): accent-insensitive.
const AI_BIT: u32 = 1 << 16;
/// Bit 0 (`UTF8_*`): lowercase.
const LCASE_BIT: u32 = 1;
/// Bits 0 to 11 (ICU): the locale.
const LOCALE_MASK: u32 = 0xFFF;
/// Spark's id of an indeterminate collation (`-1`).
const INDETERMINATE_ID: u32 = u32::MAX;

/// The prefix of a qualified name.
const QUALIFIER: &str = "SYSTEM.BUILTIN.";

/// The ICU locales Spark 4.2 knows, spelled as `collations()` prints them
/// (`UNICODE` first). A locale's index is its id's locale field.
static LOCALES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    FIXTURE
        .lines()
        .filter_map(|line| {
            let mut cols = line.split('\t');
            let name = cols.nth(2)?;
            let pad = cols.nth(4)?;
            let plain = !name.starts_with("UTF8_")
                && pad == "NO_PAD"
                && !line.contains("ACCENT_INSENSITIVE")
                && !line.contains("CASE_INSENSITIVE");
            plain.then_some(name)
        })
        .collect()
});

/// Every name in `collations()`, for proposals.
static NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    FIXTURE
        .lines()
        .filter_map(|l| l.split('\t').nth(2))
        .collect()
});

/// A string collation other than the one every `STRING` has by default
/// (`UTF8_BINARY`), or `UTF8_BINARY` itself where one is needed on its own.
///
/// The value is Spark's 32-bit collation id: bit 29 is the provider (0 for
/// `UTF8_*`, 1 for ICU), bit 18 `RTRIM`; for `UTF8_*` bit 0 is lowercase; for
/// ICU bit 17 is case-insensitive, bit 16 accent-insensitive and bits 0 to 11
/// the locale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Collation(u32);

/// What kind of comparison a collation makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollationFamily {
    /// `UTF8_BINARY`: by UTF-8 bytes.
    Utf8Binary,
    /// `UTF8_LCASE`: by lower-cased code points.
    Utf8Lcase,
    /// An ICU collation.
    Icu {
        /// Case-insensitive (`_CI`).
        ci: bool,
        /// Accent-insensitive (`_AI`).
        ai: bool,
    },
}

impl Collation {
    /// `UTF8_BINARY`, the default.
    pub const UTF8_BINARY: Self = Self(0);
    /// `UTF8_LCASE`.
    pub const UTF8_LCASE: Self = Self(LCASE_BIT);
    /// The collation of an expression whose operands' collations conflict
    /// (`INDETERMINATE`; Spark's id `-1`).
    pub const INDETERMINATE: Self = Self(INDETERMINATE_ID);

    /// Spark's id.
    #[must_use]
    pub const fn id(self) -> u32 {
        self.0
    }

    /// Parses a collation name as Spark does: case-insensitive, an optional
    /// `SYSTEM.BUILTIN.` qualifier, `UTF8_BINARY` or `UTF8_LCASE`, or an ICU
    /// `<locale>[_CI][_AI]`, each with an optional `_RTRIM`. `_CS` and `_AS`
    /// are accepted and dropped.
    ///
    /// # Errors
    /// `COLLATION_INVALID_NAME`, with up to three proposals.
    pub fn parse(name: &str) -> Result<Self> {
        Self::try_parse(name).ok_or_else(|| invalid_name(name))
    }

    fn try_parse(name: &str) -> Option<Self> {
        let bare = match name.get(..QUALIFIER.len()) {
            Some(head) if head.eq_ignore_ascii_case(QUALIFIER) => &name[QUALIFIER.len()..],
            _ => name,
        };
        let mut rest = bare;
        let mut bits = 0;
        if let Some(head) = strip_suffix_ci(rest, "_RTRIM") {
            rest = head;
            bits |= RTRIM_BIT;
        }
        if rest.eq_ignore_ascii_case("UTF8_BINARY") {
            return Some(Self(bits));
        }
        if rest.eq_ignore_ascii_case("UTF8_LCASE") {
            return Some(Self(bits | LCASE_BIT));
        }
        if let Some(head) = strip_suffix_ci(rest, "_AI") {
            rest = head;
            bits |= AI_BIT;
        } else if let Some(head) = strip_suffix_ci(rest, "_AS") {
            rest = head;
        }
        if let Some(head) = strip_suffix_ci(rest, "_CI") {
            rest = head;
            bits |= CI_BIT;
        } else if let Some(head) = strip_suffix_ci(rest, "_CS") {
            rest = head;
        }
        let index = LOCALES.iter().position(|l| l.eq_ignore_ascii_case(rest))?;
        Some(Self(ICU_BIT | bits | u32::try_from(index).ok()?))
    }

    /// Whether this is [`Self::INDETERMINATE`].
    #[must_use]
    pub const fn is_indeterminate(self) -> bool {
        self.0 == INDETERMINATE_ID
    }

    /// Whether this is `UTF8_BINARY` (without `RTRIM`).
    #[must_use]
    pub const fn is_utf8_binary(self) -> bool {
        self.0 == 0
    }

    /// The comparison family. [`Self::INDETERMINATE`] reports `Utf8Binary`.
    #[must_use]
    pub const fn family(self) -> CollationFamily {
        if self.0 & ICU_BIT != 0 && !self.is_indeterminate() {
            CollationFamily::Icu {
                ci: self.0 & CI_BIT != 0,
                ai: self.0 & AI_BIT != 0,
            }
        } else if self.0 & LCASE_BIT != 0 && !self.is_indeterminate() {
            CollationFamily::Utf8Lcase
        } else {
            CollationFamily::Utf8Binary
        }
    }

    /// Whether trailing spaces are ignored (`_RTRIM`).
    #[must_use]
    pub const fn rtrim(self) -> bool {
        self.0 & RTRIM_BIT != 0 && !self.is_indeterminate()
    }

    /// Whether equal strings are exactly the equal byte strings
    /// (`UTF8_BINARY`).
    #[must_use]
    pub const fn supports_binary_equality(self) -> bool {
        self.is_utf8_binary()
    }

    /// Whether `Utf8Lcase` equality applies (`UTF8_LCASE` or `UTF8_BINARY`
    /// compared after lower-casing is the same as `UTF8_LCASE`).
    #[must_use]
    pub const fn supports_lowercase_equality(self) -> bool {
        matches!(self.family(), CollationFamily::Utf8Lcase)
    }

    /// `collations().PAD_ATTRIBUTE`: `NO_PAD`, or `RTRIM`.
    #[must_use]
    pub const fn pad_attribute(self) -> &'static str {
        if self.rtrim() { "RTRIM" } else { "NO_PAD" }
    }

    /// The canonical name, as `collations()` prints it (the locale as ICU
    /// spells it; modifiers in the order case, accent, trim; `_CS` and `_AS`
    /// dropped). [`Self::INDETERMINATE`] is `null`.
    #[must_use]
    pub fn name(self) -> String {
        if self.is_indeterminate() {
            return "null".to_owned();
        }
        let mut out = match self.family() {
            CollationFamily::Utf8Binary => "UTF8_BINARY".to_owned(),
            CollationFamily::Utf8Lcase => "UTF8_LCASE".to_owned(),
            CollationFamily::Icu { ci, ai } => {
                let locale = usize::try_from(self.0 & LOCALE_MASK)
                    .ok()
                    .and_then(|i| LOCALES.get(i))
                    .copied()
                    .unwrap_or("UNICODE");
                let mut s = locale.to_owned();
                if ci {
                    s.push_str("_CI");
                }
                if ai {
                    s.push_str("_AI");
                }
                s
            }
        };
        if self.rtrim() {
            out.push_str("_RTRIM");
        }
        out
    }

    /// `SYSTEM.BUILTIN.<name>`: what `collation(x)` returns.
    #[must_use]
    pub fn qualified_name(self) -> String {
        format!("{QUALIFIER}{}", self.name())
    }

    /// Every collation name `collations()` lists, with its language, country,
    /// accent and case sensitivity, pad attribute and ICU version, in Spark's
    /// order.
    pub fn catalog() -> impl Iterator<Item = CatalogEntry> {
        FIXTURE.lines().filter_map(|line| {
            let c: Vec<&str> = line.split('\t').collect();
            Some(CatalogEntry {
                name: c.get(2)?,
                language: c.get(3).filter(|s| !s.is_empty()).copied(),
                country: c.get(4).filter(|s| !s.is_empty()).copied(),
                accent_sensitivity: c.get(5)?,
                case_sensitivity: c.get(6)?,
                pad_attribute: c.get(7)?,
                icu_version: c.get(8).filter(|s| !s.is_empty()).copied(),
            })
        })
    }
}

/// One row of `SELECT * FROM collations()`; the catalog is `SYSTEM` and the
/// schema `BUILTIN` for every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogEntry {
    /// The canonical name.
    pub name: &'static str,
    /// The language, or `None` for `UTF8_*` and `UNICODE`.
    pub language: Option<&'static str>,
    /// The country, if the locale has one.
    pub country: Option<&'static str>,
    /// `ACCENT_SENSITIVE` or `ACCENT_INSENSITIVE`.
    pub accent_sensitivity: &'static str,
    /// `CASE_SENSITIVE` or `CASE_INSENSITIVE`.
    pub case_sensitivity: &'static str,
    /// `NO_PAD` or `RTRIM`.
    pub pad_attribute: &'static str,
    /// The ICU version, or `None` for `UTF8_*`.
    pub icu_version: Option<&'static str>,
}

impl fmt::Display for Collation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

fn strip_suffix_ci<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    let cut = s.len().checked_sub(suffix.len())?;
    let tail = s.get(cut..)?;
    tail.eq_ignore_ascii_case(suffix).then(|| &s[..cut])
}

/// Levenshtein distance between the upper-cased texts.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.to_uppercase().chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.to_uppercase().chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(ca != *cb)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// `COLLATION_INVALID_NAME`: the (at most three) closest names.
pub(crate) fn invalid_name(name: &str) -> KrishivError {
    let mut ranked: Vec<(usize, &str)> = NAMES.iter().map(|n| (distance(name, n), *n)).collect();
    ranked.sort();
    let proposals: Vec<&str> = ranked.iter().take(3).map(|(_, n)| *n).collect();
    CollationInvalidName {
        collation_name: name.to_owned(),
        proposals: proposals.join(", "),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_and_lcase_ids_are_spark_s() {
        assert_eq!(
            Collation::parse("utf8_binary").unwrap(),
            Collation::UTF8_BINARY
        );
        assert_eq!(Collation::parse("UTF8_LCASE").unwrap().id(), 1);
        assert_eq!(Collation::parse("utf8_binary_rtrim").unwrap().id(), 1 << 18);
        assert_eq!(
            Collation::parse("UTF8_LCASE_RTRIM").unwrap().id(),
            (1 << 18) | 1
        );
    }

    #[test]
    fn names_are_canonical() {
        let n = |s: &str| Collation::parse(s).unwrap().name();
        assert_eq!(n("unicode_ci_ai"), "UNICODE_CI_AI");
        assert_eq!(n("UNICODE_CS_AS"), "UNICODE");
        assert_eq!(n("system.builtin.utf8_lcase"), "UTF8_LCASE");
        assert_eq!(n("en_ci"), "en_CI");
        assert_eq!(n("SR_cyrl_srb_ai_rtrim"), "sr_Cyrl_SRB_AI_RTRIM");
        assert_eq!(
            Collation::parse("sv").unwrap().qualified_name(),
            "SYSTEM.BUILTIN.sv"
        );
    }

    #[test]
    fn every_listed_name_round_trips() {
        for e in Collation::catalog() {
            let c = Collation::parse(e.name).unwrap();
            assert_eq!(c.name(), e.name);
            assert_eq!(c.pad_attribute(), e.pad_attribute);
            assert_eq!(
                matches!(c.family(), CollationFamily::Icu { ai: true, .. }),
                e.accent_sensitivity == "ACCENT_INSENSITIVE"
            );
        }
        assert_eq!(Collation::catalog().count(), 1164);
    }

    #[test]
    fn families_follow_the_bits() {
        let f = |s: &str| Collation::parse(s).unwrap().family();
        assert_eq!(f("UTF8_BINARY_RTRIM"), CollationFamily::Utf8Binary);
        assert_eq!(f("UTF8_LCASE_RTRIM"), CollationFamily::Utf8Lcase);
        assert_eq!(
            f("UNICODE_CI"),
            CollationFamily::Icu {
                ci: true,
                ai: false
            }
        );
        assert!(Collation::parse("UTF8_BINARY_RTRIM").unwrap().rtrim());
        assert!(
            Collation::parse("UTF8_BINARY")
                .unwrap()
                .supports_binary_equality()
        );
        assert!(Collation::UTF8_LCASE.supports_lowercase_equality());
        assert!(Collation::INDETERMINATE.is_indeterminate());
        assert_eq!(Collation::INDETERMINATE.name(), "null");
    }

    #[test]
    fn an_unknown_name_proposes_the_closest() {
        let e = Collation::parse("utf8_lcas").unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::CollationInvalidName
        );
        assert!(e.to_string().contains("UTF8_LCASE"), "{e}");
        assert!(Collation::parse("UNICODE_AI_CI").is_err());
        assert!(Collation::parse("").is_err());
    }
}
