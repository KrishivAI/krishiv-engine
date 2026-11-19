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
