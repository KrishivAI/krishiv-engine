// SPDX-License-Identifier: Apache-2.0
//! The snapshot cache ([`SnapshotCache`]) and the table's name in errors, on the oracle table
//! `session-appends` (three commits; the last one is put back while the cache holds the second)
//! and on delta-spark's session probes (`SESSION.json`).

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_io::StoreId;
use serde_json::Value;

use super::oracle_tests::{Env, copy_dir, fixtures};
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

fn try_open_cached(
    env: &Env,
    cache: &Arc<SnapshotCache>,
    key: CacheKey,
) -> Result<DeltaTable, KrishivError> {
    let options = OpenOptions {
        cache: Some((Arc::clone(cache), key)),
        ..OpenOptions::default()
    };
    env.block_on(DeltaTable::open_with(
        env.engine.clone(),
        options,
        &Config::new(),
    ))
}

/// The table at `env`'s root replaced by the oracle table `name` (same path, a new log).
fn recreate(env: &Env, name: &str) {
    std::fs::remove_dir_all(&env.dir).unwrap();
    copy_dir(&fixtures().join(name), &env.dir);
}

/// A table removed after a cached open is not served from the cache: the path holds no Delta
/// table now, and the entry is dropped.
#[test]
fn a_removed_table_is_not_served_from_the_cache() {
    let env = Env::fixture("session-appends");
    let cache = Arc::new(SnapshotCache::new(4));
    let first = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!(first.version(), 2);
    std::fs::remove_dir_all(env.dir.join("_delta_log")).unwrap();
    let fresh = env.open().map(|t| t.version());
    let e = try_open_cached(&env, &cache, key(&env, "alice")).map(|t| t.version());
    assert!(fresh.is_err(), "{fresh:?}");
    // The same outcome as an uncached open, never the old snapshot.
    assert_eq!(
        e.as_ref().unwrap_err().error_class(),
        fresh.as_ref().unwrap_err().error_class(),
    );
    assert_eq!(cache.cached_version(&key(&env, "alice")), None);
}

/// A table recreated at the same path with fewer commits reads as the new table, not as the
/// kernel's "older than the cached snapshot" error, and its entry is replaced.
#[test]
fn a_table_recreated_with_fewer_commits_reads_the_new_one() {
    let env = Env::fixture("session-appends");
    let cache = Arc::new(SnapshotCache::new(4));
    assert_eq!(open_cached(&env, &cache, key(&env, "alice")).version(), 2);
    recreate(&env, "partitioned");
    let want = env.open().unwrap();
    let got = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!(got.version(), want.version());
    assert_eq!(got.schema(), want.schema());
    assert_eq!(got.units().len(), want.units().len());
    assert_eq!(
        cache.cached_version(&key(&env, "alice")),
        Some(want.version())
    );
}

/// A table recreated with as many commits: the cached last commit no longer matches (its
/// modification time moved), so the listing is read again rather than reused.
#[test]
fn a_rewritten_last_commit_is_read_again() {
    let env = Env::fixture("session-appends");
    let cache = Arc::new(SnapshotCache::new(4));
    let first = open_cached(&env, &cache, key(&env, "alice"));
    recreate(&env, "session-appends");
    let last = std::fs::File::options()
        .write(true)
        .open(env.dir.join(format!("{LAST}.json")))
        .unwrap();
    last.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(3_600))
        .unwrap();
    let again = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!(again.version(), 2);
    assert!(!Arc::ptr_eq(&first.inner.units, &again.inner.units));
    // Unchanged since: the listing is reused.
    let third = open_cached(&env, &cache, key(&env, "alice"));
    assert!(Arc::ptr_eq(&again.inner.units, &third.inner.units));
}

/// A cached entry whose refresh fails is dropped, so the table reads again once it is fixed.
#[test]
fn a_cached_entry_whose_refresh_fails_is_dropped() {
    let env = two_commits();
    let cache = Arc::new(SnapshotCache::new(4));
    assert_eq!(open_cached(&env, &cache, key(&env, "alice")).version(), 1);
    // A broken next commit: the refresh from the cached snapshot fails, and so does a fresh build.
    std::fs::write(env.dir.join(format!("{LAST}.json")), "{not json\n").unwrap();
    assert!(try_open_cached(&env, &cache, key(&env, "alice")).is_err());
    assert_eq!(cache.cached_version(&key(&env, "alice")), None);
    std::fs::remove_file(env.dir.join(format!("{LAST}.json"))).unwrap();
    commit_last(&env);
    assert_eq!(open_cached(&env, &cache, key(&env, "alice")).version(), 2);
}

/// The cache is bounded by its estimated bytes: a second table that does not fit beside the
/// first pushes it out.
#[test]
fn the_cache_is_bounded_by_bytes() {
    let env = two_commits();
    let one = cache::ENTRY_BYTES + listing_bytes(env.open().unwrap().units());
    assert!(listing_bytes(env.open().unwrap().units()) > 0);
    let cache = Arc::new(SnapshotCache::bounded(8, one + one / 2));
    open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!(cache.bytes(), one);
    open_cached(&env, &cache, key(&env, "bob"));
    assert_eq!((cache.len(), cache.bytes()), (1, one));
    assert_eq!(cache.cached_version(&key(&env, "alice")), None);
    assert_eq!(cache.cached_version(&key(&env, "bob")), Some(1));
}

/// A table whose listing alone exceeds the bound keeps only its snapshot: the next query
/// starts from it and lists the files again.
#[test]
fn a_listing_over_the_bound_is_not_kept() {
    let env = two_commits();
    let cache = Arc::new(SnapshotCache::bounded(8, cache::ENTRY_BYTES + 1));
    let first = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!(cache.cached_version(&key(&env, "alice")), Some(1));
    assert!(!cache.keeps_listing(&key(&env, "alice")));
    assert_eq!(cache.bytes(), cache::ENTRY_BYTES);
    let again = open_cached(&env, &cache, key(&env, "alice"));
    assert_eq!((again.version(), cache.hits()), (1, 1));
    assert!(!Arc::ptr_eq(&first.inner.units, &again.inner.units));
    assert_eq!(again.units().len(), first.units().len());
    // Nothing fits under a bound below one snapshot.
    let tiny = Arc::new(SnapshotCache::bounded(8, 1));
    open_cached(&env, &tiny, key(&env, "alice"));
    assert!(tiny.is_empty());
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

#[test]
fn a_corrupt_commit_is_a_failed_read_of_that_file() {
    let env = two_commits();
    let commit = env.dir.join("_delta_log/00000000000000000001.json");
    std::fs::write(&commit, "{not json\n").unwrap();
    for entry in std::fs::read_dir(env.dir.join("_delta_log")).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().is_some_and(|e| e == "crc") {
            std::fs::remove_file(p).unwrap();
        }
    }
    let e = env.open().unwrap_err();
    let want = &session_probe("corrupt_commit");
    assert_eq!(
        e.error_class().name(),
        want["condition"].as_str().unwrap(),
        "{e}"
    );
    assert_eq!(e.sql_state(), want["sqlstate"].as_str());
    let path = e
        .message_parameters()
        .iter()
        .find(|(k, _)| *k == "path")
        .map(|(_, v)| v.clone())
        .unwrap();
    assert_eq!(
        path.replace(&env.dir.to_string_lossy().into_owned(), "<path>"),
        want["parameters"]["path"].as_str().unwrap()
    );
}

/// A query that found an entry stale drops only that entry: a newer one another query stored
/// meanwhile stays (re-review N2).
#[test]
fn a_stale_drop_keeps_a_newer_entry() {
    let env = two_commits();
    let cache = Arc::new(SnapshotCache::new(4));
    assert_eq!(open_cached(&env, &cache, key(&env, "alice")).version(), 1);
    let seen = cache.lookup(&key(&env, "alice")).unwrap();
    commit_last(&env);
    assert_eq!(open_cached(&env, &cache, key(&env, "alice")).version(), 2);
    cache.invalidate_if(&key(&env, "alice"), seen.snapshot.version(), seen.last);
    assert_eq!(cache.cached_version(&key(&env, "alice")), Some(2));
    let now = cache.lookup(&key(&env, "alice")).unwrap();
    cache.invalidate_if(&key(&env, "alice"), now.snapshot.version(), now.last);
    assert_eq!(cache.cached_version(&key(&env, "alice")), None);
}
