// SPDX-License-Identifier: Apache-2.0
//! Spark's `spark.sql.legacy.*` settings that Krishiv does not honour (P3 PR-05, ADR 0008).
//!
//! A legacy setting restores an older Spark behaviour. The ones Krishiv implements are registered config
//! entries (`registry.rs`, with a `spark_alias`) and are found before this module is asked. For every other
//! `spark.sql.legacy.*` key any supported Spark release registers (`legacy_table.rs`, generated from the
//! releases by `tests/scripts/gen_legacy_settings.py`):
//!
//! * set to a value Spark itself defaults to, in any supported release, it changes nothing and is accepted;
//! * set to any other value it asks for behaviour Krishiv does not have, and `SET` is refused with
//!   `KRISHIV_UNSUPPORTED_FEATURE`: an unhonoured legacy setting never gives a silently different result;
//! * a key the engine's own code handles (`HANDLED_IN_CODE`) is left to that code;
//! * a key no release registers (a typo, a vendor key) keeps the unknown-key warning.
//!
//! [`ALREADY_THE_BEHAVIOUR`] lists the settings whose non-default value Krishiv already behaves as, with the
//! evidence; they are accepted too.

mod table {
    include!("legacy_table.rs");
}
use table::SPARK_LEGACY_SETTINGS;

/// `(key, value, evidence)`: a non-default value that Krishiv's one behaviour already matches, so a `SET`
/// to it is accepted. The evidence is the golden files or the Spark probe that shows it.
pub(super) const ALREADY_THE_BEHAVIOUR: &[(&str, &str, &str)] = &[];

/// Keys the engine reads or refuses itself, outside the config registry (the parser, the analyzer and the session
/// check them at the point they matter, some by refusing one value), so `SET` of them is left to that code.
/// `handled_keys_are_the_ones_the_sources_name` keeps this list equal to the keys the crates' sources name.
pub(super) const HANDLED_IN_CODE: &[&str] = &[
    "spark.sql.legacy.allowHashOnMapType",
    "spark.sql.legacy.allowParameterlessCount",
    "spark.sql.legacy.allowStarWithSingleTableIdentifierInCount",
    "spark.sql.legacy.allowUdfParameterToShadowParameterlessFunction",
    "spark.sql.legacy.bangEqualsNot",
    "spark.sql.legacy.charVarcharAsString",
    "spark.sql.legacy.codingErrorAction",
    "spark.sql.legacy.collationAwareHashFunctions",
    "spark.sql.legacy.consecutiveStringLiterals.enabled",
    "spark.sql.legacy.ctePrecedencePolicy",
    "spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName",
    "spark.sql.legacy.extraOptionsBehavior.enabled",
    "spark.sql.legacy.identifierClause",
    "spark.sql.legacy.interval.enabled",
    "spark.sql.legacy.javaCharsets",
    "spark.sql.legacy.negativeIndexInArrayInsert",
    "spark.sql.legacy.noCharPaddingInPredicate",
    "spark.sql.legacy.nullInEmptyListBehavior",
    "spark.sql.legacy.parser.havingWithoutGroupByAsWhere",
    "spark.sql.legacy.percentileDiscCalculation",
    "spark.sql.legacy.scalarSubqueryAllowGroupByNonEqualityCorrelatedPredicate",
    "spark.sql.legacy.scalarSubqueryCountBugBehavior",
    "spark.sql.legacy.setopsPrecedence.enabled",
    "spark.sql.legacy.skipTypeValidationOnAlterPartition",
    "spark.sql.legacy.statisticalAggregate",
    "spark.sql.legacy.timeParserPolicy",
    "spark.sql.legacy.viewSchemaBindingMode",
    "spark.sql.legacy.viewSchemaCompensation",
];

/// What [`check`] decides for a `spark.sql.legacy.*` key that is not a registered entry.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The value is a Spark default, or Krishiv already behaves as the setting asks.
    Accept,
    /// Spark has the setting; the value asks for behaviour Krishiv does not implement. `since` is the first
    /// supported release that registers it.
    Refuse { since: &'static str },
    /// The engine's own code handles the key: left to it, as before.
    Handled,
    /// No supported release registers this key.
    Unknown,
}

fn norm(v: &str) -> String {
    v.trim().to_ascii_lowercase()
}

/// Decides a `SET key = value` of an unregistered `spark.sql.legacy.*` key.
pub(super) fn check(key: &str, value: &str) -> Verdict {
    if HANDLED_IN_CODE.binary_search(&key).is_ok() {
        return Verdict::Handled;
    }
    let Ok(i) = SPARK_LEGACY_SETTINGS.binary_search_by(|(k, _, _)| (*k).cmp(key)) else {
        return Verdict::Unknown;
    };
    let (_, defaults, since) = SPARK_LEGACY_SETTINGS[i];
    let v = norm(value);
    // `<undefined>` is Spark printing "no default": no value equals it.
    if defaults.iter().any(|d| *d != "<undefined>" && norm(d) == v)
        || ALREADY_THE_BEHAVIOUR
            .iter()
            .any(|(k, val, _)| *k == key && norm(val) == v)
    {
        Verdict::Accept
    } else {
        Verdict::Refuse { since }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_and_prefixed() {
        assert!(SPARK_LEGACY_SETTINGS.len() > 100);
        assert!(SPARK_LEGACY_SETTINGS.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(
            SPARK_LEGACY_SETTINGS
                .iter()
                .all(|(k, d, _)| k.starts_with("spark.sql.legacy.") && !d.is_empty())
        );
    }

    /// The keys the crates name in their sources (not this module's table) are the handled ones.
    #[test]
    fn handled_keys_are_the_ones_the_sources_name() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut named = std::collections::BTreeSet::new();
        let mut dirs = vec![];
        for c in std::fs::read_dir(&crates).expect("crates dir").flatten() {
            dirs.push(c.path().join("src"));
        }
        while let Some(d) = dirs.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    dirs.push(p);
                } else if p.extension().is_some_and(|x| x == "rs")
                    && !p.ends_with("legacy_table.rs")
                    && !p.ends_with("legacy.rs")
                {
                    let text = std::fs::read_to_string(&p).unwrap_or_default();
                    for part in text.split("\"spark.sql.legacy.").skip(1) {
                        if let Some(end) = part.find('"') {
                            named.insert(format!("spark.sql.legacy.{}", &part[..end]));
                        }
                    }
                }
            }
        }
        let handled: std::collections::BTreeSet<String> =
            HANDLED_IN_CODE.iter().map(|k| (*k).to_owned()).collect();
        assert!(HANDLED_IN_CODE.windows(2).all(|w| w[0] < w[1]), "sorted");
        // A source naming a key (a registered one, or a test's made-up one) is handled or registered; every
        // handled key is named somewhere.
        for k in &handled {
            assert!(named.contains(k), "{k} is listed but no source names it");
        }
        for k in &named {
            let registered = crate::config::lookup(k).is_some();
            let known = SPARK_LEGACY_SETTINGS
                .binary_search_by(|(t, _, _)| (*t).cmp(k.as_str()))
                .is_ok();
            assert!(
                handled.contains(k) || registered || !known,
                "{k} is named in a source but neither handled nor registered"
            );
        }
    }

    #[test]
    fn a_default_is_accepted_and_another_value_refused() {
        let key = "spark.sql.legacy.allowNegativeScaleOfDecimal";
        assert_eq!(check(key, "false"), Verdict::Accept);
        assert_eq!(check(key, " FALSE "), Verdict::Accept);
        assert!(matches!(check(key, "true"), Verdict::Refuse { .. }));
        assert_eq!(
            check("spark.sql.legacy.noSuchThing", "true"),
            Verdict::Unknown
        );
    }

    #[test]
    fn a_key_without_a_default_refuses_every_value() {
        let key = "spark.sql.legacy.json.enableDateTimeParsingFallback";
        assert!(matches!(check(key, "true"), Verdict::Refuse { .. }));
        assert!(matches!(check(key, "false"), Verdict::Refuse { .. }));
        assert!(matches!(check(key, "<undefined>"), Verdict::Refuse { .. }));
    }

    #[test]
    fn a_default_of_any_release_is_accepted() {
        // `sizeOfNull` defaults to true in every release that has it.
        let key = "spark.sql.legacy.sizeOfNull";
        assert_eq!(check(key, "true"), Verdict::Accept);
        assert!(matches!(
            check(key, "false"),
            Verdict::Refuse { since: "3.5.9" }
        ));
        // A key the engine's code handles is left to it.
        assert_eq!(
            check("spark.sql.legacy.identifierClause", "true"),
            Verdict::Handled
        );
    }

    #[test]
    fn set_refuses_a_legacy_value_and_stays_quiet_at_a_default() {
        let mut config = crate::config::Config::new();
        let key = "spark.sql.legacy.allowNegativeScaleOfDecimal";
        assert_eq!(config.set(key, "false").expect("a default"), None);
        let e = config.set(key, "true").expect_err("refused");
        assert_eq!(
            e.error_class(),
            crate::error::ErrorClass::KrishivUnsupportedFeature
        );
        assert!(e.message().contains(key), "{}", e.message());
        // An unknown key keeps its warning.
        assert!(
            config
                .set("spark.sql.legacy.noSuchThing", "true")
                .expect("warns")
                .is_some()
        );
    }
}
