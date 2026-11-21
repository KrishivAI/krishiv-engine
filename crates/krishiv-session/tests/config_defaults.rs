// SPDX-License-Identifier: Apache-2.0
//! `EngineRuntime::set_config_defaults` (P3-T066, design §5.5): session defaults set while running.
#![cfg(test)]

use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use krishiv_session::{EngineRuntime, Session};

fn pair(k: &str, v: &str) -> (String, String) {
    (k.to_owned(), v.to_owned())
}

fn runtime() -> Arc<EngineRuntime> {
    EngineRuntime::builder()
        .config("krishiv.sql.ansi.enabled", "false")
        .build()
        .unwrap_or_else(|e| panic!("{e}"))
}

fn value(rt: &Arc<EngineRuntime>, key: &str) -> Option<String> {
    Session::builder()
        .runtime(Arc::clone(rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"))
        .conf()
        .get(key)
}

#[test]
fn a_default_reaches_new_sessions_and_not_existing_ones() {
    let rt = runtime();
    let before = Session::builder()
        .runtime(Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    rt.set_config_defaults(&[pair("krishiv.sql.ansi.enabled", "true")], &[])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        before.conf().get("krishiv.sql.ansi.enabled").as_deref(),
        Some("false"),
        "a session that exists keeps its value"
    );
    assert_eq!(
        value(&rt, "krishiv.sql.ansi.enabled").as_deref(),
        Some("true")
    );
    assert_eq!(
        rt.config_defaults(),
        vec![pair("krishiv.sql.ansi.enabled", "true")]
    );
}

#[test]
fn unset_returns_a_key_to_its_startup_value() {
    let rt = runtime();
    rt.set_config_defaults(&[pair("krishiv.sql.ansi.enabled", "true")], &[])
        .unwrap_or_else(|e| panic!("{e}"));
    rt.set_config_defaults(&[], &["krishiv.sql.ansi.enabled".to_owned()])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        value(&rt, "krishiv.sql.ansi.enabled").as_deref(),
        Some("false"),
        "the startup value, not the registry default"
    );
    assert!(rt.config_defaults().is_empty());
}

#[test]
fn a_batch_with_one_bad_key_changes_nothing() {
    let rt = runtime();
    for bad in [
        pair("krishiv.memory.limit", "1g"),
        pair("krishiv.no.such.key", "1"),
        pair("krishiv.sql.ansi.enabled", "maybe"),
    ] {
        let err = rt
            .set_config_defaults(&[pair("krishiv.sql.session.timeZone", "UTC"), bad], &[])
            .err()
            .unwrap_or_else(|| panic!("the bad key was accepted"));
        drop(err);
        assert!(rt.config_defaults().is_empty(), "the good key was applied");
    }
}

#[test]
fn a_cluster_key_is_static_and_an_unknown_key_is_unknown() {
    let rt = runtime();
    let static_err = rt
        .set_config_defaults(&[pair("krishiv.memory.limit", "1g")], &[])
        .err()
        .unwrap_or_else(|| panic!("a cluster key was accepted"));
    assert_eq!(
        static_err.error_class().name(),
        ErrorClass::CannotModifyStaticConfig.name()
    );
    let unknown = rt
        .set_config_defaults(&[pair("krishiv.no.such.key", "1")], &[])
        .err()
        .unwrap_or_else(|| panic!("an unknown key was accepted"));
    assert_eq!(
        unknown.error_class().name(),
        ErrorClass::KrishivUnknownConfig.name()
    );
}

#[test]
fn no_security_key_can_be_set_at_run_time() {
    let rt = runtime();
    for key in [
        "krishiv.auth.udf.minRole",
        "krishiv.auth.defaultRole",
        "krishiv.auth.tokens.x.secret",
        "krishiv.io.local.allowedRoots",
        "krishiv.io.objectStore.allowedPrefixes",
    ] {
        let err = rt
            .set_config_defaults(&[pair(key, "/tmp")], &[])
            .err()
            .unwrap_or_else(|| panic!("{key} was accepted"));
        assert_eq!(
            err.error_class().name(),
            ErrorClass::CannotModifyStaticConfig.name(),
            "{key}"
        );
        let err = rt
            .set_config_defaults(&[], &[key.to_owned()])
            .err()
            .unwrap_or_else(|| panic!("unset of {key} was accepted"));
        assert_eq!(
            err.error_class().name(),
            ErrorClass::CannotModifyStaticConfig.name(),
            "{key}"
        );
    }
}

#[test]
fn a_key_set_and_unset_in_one_batch_is_refused() {
    let rt = runtime();
    let err = rt
        .set_config_defaults(
            &[pair("krishiv.sql.ansi.enabled", "true")],
            &["krishiv.sql.ansi.enabled".to_owned()],
        )
        .err()
        .unwrap_or_else(|| panic!("the conflicting batch was accepted"));
    drop(err);
    assert!(rt.config_defaults().is_empty());
}

#[test]
fn an_invalid_time_zone_is_refused_as_set_refuses_it() {
    let rt = runtime();
    assert!(
        rt.set_config_defaults(&[pair("krishiv.sql.session.timeZone", "Mars/Base")], &[])
            .is_err()
    );
    assert!(rt.config_defaults().is_empty());
}

fn listed(rt: &Arc<EngineRuntime>, key: &str, reveal: bool) -> krishiv_session::ConfigListing {
    rt.config_listing(Some(key), reveal)
        .into_iter()
        .find(|r| r.key == key)
        .unwrap_or_else(|| panic!("{key} is not listed"))
}

#[test]
fn the_listing_says_where_each_value_comes_from() {
    use krishiv_session::ConfigSource;
    let rt = runtime();
    rt.set_config_defaults(&[pair("krishiv.sql.session.timeZone", "UTC")], &[])
        .unwrap_or_else(|e| panic!("{e}"));
    let startup = listed(&rt, "krishiv.sql.ansi.enabled", false);
    assert_eq!(
        (startup.source, startup.value.as_deref()),
        (ConfigSource::Startup, Some("false"))
    );
    let admin = listed(&rt, "krishiv.sql.session.timeZone", false);
    assert_eq!(
        (admin.source, admin.value.as_deref()),
        (ConfigSource::Admin, Some("UTC"))
    );
    assert_eq!(admin.settable_by, Some("user"));
    assert_eq!(admin.scope, Some("session"));
    let default = listed(&rt, "krishiv.deterministicOrder", false);
    assert_eq!(default.source, ConfigSource::Default);
    let all = rt.config_listing(None, false);
    assert!(all.windows(2).all(|w| w[0].key < w[1].key), "key order");
    assert!(
        rt.config_listing(Some("krishiv.sql."), false)
            .iter()
            .all(|r| r.key.starts_with("krishiv.sql."))
    );
}

#[test]
fn a_path_allowlist_value_is_withheld_without_manage() {
    let dir = std::env::temp_dir();
    let rt = EngineRuntime::builder()
        .config(
            "krishiv.io.local.allowedRoots",
            dir.to_string_lossy().into_owned(),
        )
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let hidden = listed(&rt, "krishiv.io.local.allowedRoots", false);
    assert!(hidden.withheld && hidden.value.is_none());
    let shown = listed(&rt, "krishiv.io.local.allowedRoots", true);
    assert!(!shown.withheld && shown.value.is_some());
}

#[test]
fn a_credential_in_a_registered_value_is_withheld_without_manage() {
    let rt = EngineRuntime::builder()
        .config("krishiv.io.egress.proxy", "http://user:password@proxy:8080")
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let hidden = listed(&rt, "krishiv.io.egress.proxy", false);
    assert!(hidden.withheld && hidden.value.is_none());
    let shown = listed(&rt, "krishiv.io.egress.proxy", true);
    assert!(!shown.withheld && shown.value.is_some());
}

#[test]
fn concurrent_sets_all_survive() {
    let rt = runtime();
    let real = [
        ("krishiv.sql.ansi.enabled", "true"),
        ("krishiv.sql.session.timeZone", "UTC"),
        ("krishiv.deterministicOrder", "true"),
        ("krishiv.cache.maxBytes", "1m"),
    ];
    std::thread::scope(|scope| {
        for (k, v) in real {
            let rt = Arc::clone(&rt);
            scope.spawn(move || {
                rt.set_config_defaults(&[pair(k, v)], &[])
                    .unwrap_or_else(|e| panic!("{e}"));
            });
        }
    });
    let set: Vec<String> = rt.config_defaults().into_iter().map(|(k, _)| k).collect();
    for (k, _) in real {
        assert!(set.iter().any(|s| s == k), "{k} was lost: {set:?}");
    }
}

#[test]
fn a_spark_alias_sets_and_unsets_the_canonical_key() {
    let rt = runtime();
    rt.set_config_defaults(&[pair("spark.sql.ansi.enabled", "true")], &[])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        rt.config_defaults(),
        vec![pair("krishiv.sql.ansi.enabled", "true")]
    );
    rt.set_config_defaults(&[], &["spark.sql.ansi.enabled".to_owned()])
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(rt.config_defaults().is_empty());
}

#[test]
fn a_build_time_session_key_is_refused() {
    let rt = runtime();
    let err = rt
        .set_config_defaults(&[pair("krishiv.session.memoryLimit", "64m")], &[])
        .err()
        .unwrap_or_else(|| panic!("a build-time key was accepted"));
    assert_eq!(
        err.error_class().name(),
        ErrorClass::CannotModifyStaticConfig.name()
    );
}

#[tokio::test]
async fn new_clone_and_reset_see_the_defaults() {
    let rt = runtime();
    let base = Session::builder()
        .runtime(Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    rt.set_config_defaults(&[pair("krishiv.sql.ansi.enabled", "true")], &[])
        .unwrap_or_else(|e| panic!("{e}"));
    let key = "krishiv.sql.ansi.enabled";
    let fresh = base.new_session();
    assert_eq!(
        fresh.conf().get(key).as_deref(),
        Some("true"),
        "new_session"
    );
    fresh
        .conf()
        .set(key, "false")
        .unwrap_or_else(|e| panic!("{e}"));
    fresh
        .sql("RESET")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(fresh.conf().get(key).as_deref(), Some("true"), "RESET");
    let clone = base.clone_session().unwrap_or_else(|e| panic!("{e}"));
    clone
        .conf()
        .set(key, "false")
        .unwrap_or_else(|e| panic!("{e}"));
    clone
        .sql("RESET")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        clone.conf().get(key).as_deref(),
        Some("true"),
        "clone RESET"
    );
}

#[tokio::test]
async fn a_clone_keeps_the_source_snapshot_until_reset() {
    // Spark's `cloneSession` copies the source's configuration, and its `RESET` goes to the shared
    // state's: a clone of a session built before a defaults change starts from the source's value
    // and returns to the new default only on `RESET`.
    let rt = runtime();
    let key = "krishiv.sql.ansi.enabled";
    let base = Session::builder()
        .runtime(Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    rt.set_config_defaults(&[pair(key, "true")], &[])
        .unwrap_or_else(|e| panic!("{e}"));
    let clone = base.clone_session().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(clone.conf().get(key).as_deref(), Some("false"));
    clone
        .sql("RESET")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(clone.conf().get(key).as_deref(), Some("true"));
}

#[test]
fn a_change_reports_the_values_it_replaced() {
    let rt = runtime();
    let key = "krishiv.sql.ansi.enabled";
    let first = rt
        .set_config_defaults(&[pair(key, "true")], &[])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(first, vec![(key.to_owned(), Some("false".to_owned()))]);
    let second = rt
        .set_config_defaults(&[], &[key.to_owned()])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(second, vec![(key.to_owned(), Some("true".to_owned()))]);
}
