// SPDX-License-Identifier: Apache-2.0
//! The delegated-catalog exchange limits (kept out of `tests.rs`, which is at its size cap).

use super::*;

/// P3-T067 §3.3: the delegated-catalog exchange limits default, parse, refuse a zero, and are
/// registered (so `krishiv.catalog.delegation.*` is never read as a catalog named `delegation`).
#[test]
fn catalog_delegation_limits_default_parse_and_are_registered() {
    let mut c = Config::new();
    let get = |c: &Config| {
        (
            c.get(&CATALOG_DELEGATION_MAX_EXCHANGES_PER_MINUTE)
                .and_then(ConfigValue::as_int),
            c.get(&CATALOG_DELEGATION_MAX_DISTINCT_EXCHANGES_PER_MINUTE)
                .and_then(ConfigValue::as_int),
        )
    };
    assert_eq!(get(&c), (Some(60), Some(20)));
    c.set("krishiv.catalog.delegation.maxExchangesPerMinute", "5")
        .unwrap();
    c.set(
        "krishiv.catalog.delegation.maxDistinctExchangesPerMinute",
        "2",
    )
    .unwrap();
    assert_eq!(get(&c), (Some(5), Some(2)));
    assert!(
        c.set("krishiv.catalog.delegation.maxExchangesPerMinute", "0")
            .is_err()
    );
    for e in [
        &CATALOG_DELEGATION_MAX_EXCHANGES_PER_MINUTE,
        &CATALOG_DELEGATION_MAX_DISTINCT_EXCHANGES_PER_MINUTE,
    ] {
        assert!(lookup(e.key).is_some(), "{}", e.key);
        assert_eq!(e.scope, Scope::Cluster, "{}", e.key);
        assert_eq!(e.settable_by, SettableBy::Startup, "{}", e.key);
    }
}

/// P3-T067 §4.1, §4.3: the vended-credential settings default, parse and are registered (so
/// `krishiv.catalog.vended.*` is never read as a catalog named `vended`).
#[test]
fn catalog_vended_settings_default_parse_and_are_registered() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&CATALOG_VENDED_DEFAULT_TTL)
            .and_then(ConfigValue::as_duration),
        Some(std::time::Duration::from_secs(900))
    );
    assert_eq!(
        c.get(&CATALOG_VENDED_MAX_ENTRIES)
            .and_then(ConfigValue::as_int),
        Some(10_000)
    );
    c.set("krishiv.catalog.vendedDefaultTtl", "2m").unwrap();
    c.set("krishiv.catalog.vended.maxEntries", "3").unwrap();
    assert_eq!(
        c.get(&CATALOG_VENDED_DEFAULT_TTL)
            .and_then(ConfigValue::as_duration),
        Some(std::time::Duration::from_secs(120))
    );
    assert_eq!(
        c.get(&CATALOG_VENDED_MAX_ENTRIES)
            .and_then(ConfigValue::as_int),
        Some(3)
    );
    assert!(c.set("krishiv.catalog.vended.maxEntries", "0").is_err());
    for e in [&CATALOG_VENDED_DEFAULT_TTL, &CATALOG_VENDED_MAX_ENTRIES] {
        assert!(lookup(e.key).is_some(), "{}", e.key);
        assert_eq!(e.scope, Scope::Cluster, "{}", e.key);
        assert_eq!(e.settable_by, SettableBy::Startup, "{}", e.key);
    }
}
