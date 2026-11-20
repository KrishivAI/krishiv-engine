// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.delta.*` keys and the table-format commit retry bounds (kept out of `tests.rs`, which is at its size cap).

use super::*;

/// P3-T021: the Delta log pool and the catalog parse limits.
#[test]
fn delta_and_catalog_limit_entries_parse_and_default() {
    let mut c = Config::new();
    let get = |c: &Config| {
        (
            c.get(&DELTA_LOG_THREADS).and_then(ConfigValue::as_int),
            c.get(&CATALOG_MAX_METADATA_BYTES)
                .and_then(ConfigValue::as_size),
            c.get(&CATALOG_MAX_COLUMNS).and_then(ConfigValue::as_int),
        )
    };
    assert_eq!(get(&c), (Some(4), Some(256 << 20), Some(10_000)));
    c.set("krishiv.delta.logThreads", "8").unwrap();
    c.set("krishiv.catalog.maxMetadataBytes", "1g").unwrap();
    c.set("krishiv.catalog.maxColumns", "50000").unwrap();
    assert_eq!(get(&c), (Some(8), Some(1 << 30), Some(50_000)));
    assert!(c.set("krishiv.delta.logThreads", "0").is_err());
    assert!(c.set("krishiv.catalog.maxColumns", "-1").is_err());
    assert!(c.set("krishiv.catalog.maxMetadataBytes", "lots").is_err());
    let cache = |c: &Config| {
        c.get(&ICEBERG_MANIFEST_CACHE_BYTES)
            .and_then(ConfigValue::as_size)
    };
    assert_eq!(cache(&c), Some(64 << 20));
    c.set("krishiv.iceberg.manifestCache.bytes", "0").unwrap();
    assert_eq!(cache(&c), Some(0));
    c.set("krishiv.iceberg.manifestCache.bytes", "1g").unwrap();
    assert_eq!(cache(&c), Some(1 << 30));
    assert!(
        c.set("krishiv.iceberg.manifestCache.bytes", "lots")
            .is_err()
    );
    for e in [
        &DELTA_LOG_THREADS,
        &CATALOG_MAX_METADATA_BYTES,
        &CATALOG_MAX_COLUMNS,
        &ICEBERG_MANIFEST_CACHE_BYTES,
    ] {
        assert_eq!(e.scope, Scope::Cluster, "{}", e.key);
    }
}

/// P3-T021 PR-37a: the bounds of the Delta commit loop, `maxCommitAttempts` under Spark's name
/// too.
#[test]
fn delta_commit_retry_entries_parse_and_default() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&DELTA_MAX_COMMIT_ATTEMPTS)
            .and_then(ConfigValue::as_int),
        Some(10_000_000)
    );
    assert_eq!(
        c.get(&WRITE_COMMIT_RETRY_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(std::time::Duration::from_secs(1800))
    );
    c.set("spark.databricks.delta.maxCommitAttempts", "3")
        .unwrap();
    assert_eq!(
        c.get(&DELTA_MAX_COMMIT_ATTEMPTS)
            .and_then(ConfigValue::as_int),
        Some(3)
    );
    c.set("krishiv.write.commitRetryTimeout", "90s").unwrap();
    assert_eq!(
        c.get(&WRITE_COMMIT_RETRY_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(std::time::Duration::from_secs(90))
    );
    assert!(c.set("krishiv.delta.maxCommitAttempts", "0").is_err());
}
