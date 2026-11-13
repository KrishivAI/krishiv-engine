// SPDX-License-Identifier: Apache-2.0
//! The snapshot cache ([`SnapshotCache`]) and the table's name in errors, on the oracle table
//! `session-appends` (three commits; the last one is put back while the cache holds the second)
//! and on delta-spark's session probes (`SESSION.json`).

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_io::StoreId;
use serde_json::Value;

use super::oracle_tests::{Env, fixtures};
use super::*;

const LAST: &str = "_delta_log/00000000000000000002";

/// `session-appends` without its last commit, which is kept aside in the table directory's parent.
fn two_commits() -> Env {
    let env = Env::fixture("session-appends");
    for ext in ["json", "crc"] {
        let from = env.dir.join(format!("{LAST}.{ext}"));
        std::fs::rename(&from, env.dir.with_extension(ext)).unwrap();
    }
    env
}

/// Puts the last commit back.
fn commit_last(env: &Env) {
    for ext in ["crc", "json"] {
        std::fs::rename(
            env.dir.with_extension(ext),
            env.dir.join(format!("{LAST}.{ext}")),
        )
        .unwrap();
    }
}

fn key(env: &Env, credential: &str) -> CacheKey {
    CacheKey {
        root: env.engine.table_root().as_str().to_owned(),
        store: StoreId::new("file"),
        credential: credential.to_owned(),
    }
}

fn open_cached(env: &Env, cache: &Arc<SnapshotCache>, key: CacheKey) -> DeltaTable {
    let options = OpenOptions {
        cache: Some((Arc::clone(cache), key)),
        ..OpenOptions::default()
    };
    env.block_on(DeltaTable::open_with(
        env.engine.clone(),
        options,
        &Config::new(),
    ))
    .unwrap()
}

#[test]
fn a_cached_snapshot_is_brought_up_to_date() {
    let env = two_commits();
    let cache = Arc::new(SnapshotCache::new(4));
    let first = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!((first.version(), cache.hits(), cache.misses()), (1, 0, 1));
    // Unchanged: the cached listing is reused, no new listing.
    let again = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!((again.version(), cache.hits()), (1, 1));
    assert!(Arc::ptr_eq(&first.inner.units, &again.inner.units));
    // A commit lands: the next open sees it, and the cache now holds it.
    commit_last(&env);
    let newer = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!(
        (newer.version(), newer.units().len(), cache.hits()),
        (2, 2, 2)
    );
    assert_eq!(cache.cached_version(&key(&env, "alice")), Some(2));
    // The table opened before the commit still reads its own version.
    assert_eq!((first.version(), first.units().len()), (1, 1));
}

#[test]
fn another_credential_misses() {
    let env = two_commits();
    let cache = Arc::new(SnapshotCache::new(4));
    open_cached(&env, &cache, key(&env, "alice"));
    open_cached(&env, &cache, key(&env, "bob"));
    assert_eq!((cache.hits(), cache.misses(), cache.len()), (0, 2, 2));
    let mut other_store = key(&env, "alice");
    other_store.store = StoreId::new("file#cfg:1");
    open_cached(&env, &cache, other_store);
    assert_eq!(cache.misses(), 3);
}

#[test]
fn the_cache_is_bounded_and_time_travel_bypasses_it() {
    let env = two_commits();
    let cache = Arc::new(SnapshotCache::new(1));
    open_cached(&env, &cache, key(&env, "alice"));
    open_cached(&env, &cache, key(&env, "bob"));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.cached_version(&key(&env, "alice")), None);
    let options = OpenOptions {
        version: Some(0),
        cache: Some((Arc::clone(&cache), key(&env, "carol"))),
        ..OpenOptions::default()
    };
    let old = env
        .block_on(DeltaTable::open_with(
            env.engine.clone(),
            options,
            &Config::new(),
        ))
        .unwrap();
    assert_eq!(old.version(), 0);
    assert_eq!((cache.hits(), cache.misses(), cache.len()), (0, 2, 1));
}

#[test]
fn hadoop_paths() {
    let url = |s: &str| url::Url::parse(s).unwrap();
    assert_eq!(hadoop_path(&url("file:///a/b/")), "file:/a/b");
    assert_eq!(hadoop_path(&url("file:///a/b%20c/")), "file:/a/b c");
    assert_eq!(hadoop_path(&url("file:///")), "file:/");
    assert_eq!(hadoop_path(&url("s3://bucket/k/t/")), "s3://bucket/k/t");
}

fn session_probe(name: &str) -> Value {
    let all: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("SESSION.json")).unwrap()).unwrap();
    all[name]["error"].clone()
}

/// `session-appends` with the probe's unknown reader feature added by a new commit.
fn refused() -> Env {
    let env = two_commits();
    std::fs::write(
        env.dir.join("_delta_log/00000000000000000002.json"),
        r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["krishivProbeFeature"],"writerFeatures":["krishivProbeFeature"]}}
"#,
    )
    .unwrap();
    env
}

fn table_name_or_path(e: &KrishivError) -> String {
    e.message_parameters()
        .iter()
        .find(|(k, _)| *k == "tableNameOrPath")
        .map(|(_, v)| v.clone())
        .unwrap()
}

#[test]
fn a_refused_table_is_named_as_delta_spark_names_it() {
    let env = refused();
    let path = env.dir.to_string_lossy().into_owned();
    // By path: the Hadoop form, also when the path was written with a trailing slash.
    let e = env.open().unwrap_err();
    let want = session_probe("refused_path_sql");
    assert_eq!(e.error_class().name(), want["condition"].as_str().unwrap());
    assert_eq!(
        table_name_or_path(&e).replace(&path, "<path>"),
        want["parameters"]["tableNameOrPath"].as_str().unwrap()
    );
    assert_eq!(
        session_probe("refused_path_trailing_slash")["parameters"]["tableNameOrPath"],
        want["parameters"]["tableNameOrPath"]
    );
    // A catalog table: its `namespace.table`, as the session passes it.
    let options = OpenOptions {
        name: Some("default.refused".to_owned()),
        ..OpenOptions::default()
    };
    let e = env
        .block_on(DeltaTable::open_with(
            env.engine.clone(),
            options,
            &Config::new(),
        ))
        .unwrap_err();
    let want = session_probe("refused_catalog_table");
    assert_eq!(e.error_class().name(), want["condition"].as_str().unwrap());
    assert_eq!(e.sql_state(), want["sqlstate"].as_str());
    assert_eq!(
        table_name_or_path(&e),
        want["parameters"]["tableNameOrPath"].as_str().unwrap()
    );
}

#[test]
fn the_delta_log_is_found_or_not() {
    let env = two_commits();
    assert!(env.block_on(has_delta_log(&env.engine)).unwrap());
    let empty = Env::empty();
    assert!(!empty.block_on(has_delta_log(&empty.engine)).unwrap());
}
