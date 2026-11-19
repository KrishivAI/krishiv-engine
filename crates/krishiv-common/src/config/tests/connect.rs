// SPDX-License-Identifier: Apache-2.0
//! The Spark Connect reattach and artifact keys (kept out of `tests.rs`, which is at its size cap).

use super::*;

#[test]
fn connect_reattach_entries_parse_default_and_take_sparks_names() {
    let d = Config::new();
    let size = |c: &Config, e| c.get(e).and_then(ConfigValue::as_size);
    let millis = |c: &Config, e| c.get(e).and_then(ConfigValue::as_duration);
    // Spark 4.3.0 `Connect.scala` lines 161-227 (static, internal).
    assert_eq!(size(&d, &CONNECT_EXECUTE_RETRY_BUFFER_SIZE), Some(10 << 20));
    assert_eq!(
        millis(&d, &CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION),
        Some(Duration::from_secs(120))
    );
    assert_eq!(
        size(&d, &CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE),
        Some(1 << 30)
    );
    assert_eq!(
        millis(&d, &CONNECT_EXECUTE_DETACHED_TIMEOUT),
        Some(Duration::from_secs(300))
    );
    assert_eq!(
        millis(&d, &CONNECT_EXECUTE_MAINTENANCE_INTERVAL),
        Some(Duration::from_secs(30))
    );
    assert_eq!(
        d.get(&CONNECT_EXECUTE_ABANDONED_TOMBSTONES)
            .and_then(ConfigValue::as_int),
        Some(10_000)
    );
    assert_eq!(size(&d, &CONNECT_MAX_RESPONSE_BYTES), Some(1 << 30));

    let mut c = Config::new();
    c.set(
        "spark.connect.execute.reattachable.observerRetryBufferSize",
        "1m",
    )
    .unwrap();
    c.set(
        "spark.connect.execute.reattachable.senderMaxStreamDuration",
        "1500",
    )
    .unwrap();
    c.set(
        "spark.connect.execute.reattachable.senderMaxStreamSize",
        "0",
    )
    .unwrap();
    c.set("spark.connect.execute.manager.detachedTimeout", "1s")
        .unwrap();
    c.set("spark.connect.execute.manager.maintenanceInterval", "100")
        .unwrap();
    c.set("spark.connect.execute.manager.abandonedTombstonesSize", "5")
        .unwrap();
    c.set("krishiv.connect.maxResponseBytes", "2m").unwrap();
    assert_eq!(size(&c, &CONNECT_EXECUTE_RETRY_BUFFER_SIZE), Some(1 << 20));
    assert_eq!(
        millis(&c, &CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION),
        Some(Duration::from_millis(1500))
    );
    assert_eq!(size(&c, &CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE), Some(0));
    assert_eq!(
        millis(&c, &CONNECT_EXECUTE_DETACHED_TIMEOUT),
        Some(Duration::from_secs(1))
    );
    assert_eq!(
        millis(&c, &CONNECT_EXECUTE_MAINTENANCE_INTERVAL),
        Some(Duration::from_millis(100))
    );
    assert_eq!(
        c.get(&CONNECT_EXECUTE_ABANDONED_TOMBSTONES)
            .and_then(ConfigValue::as_int),
        Some(5)
    );
    assert_eq!(size(&c, &CONNECT_MAX_RESPONSE_BYTES), Some(2 << 20));
    assert!(c.set("krishiv.connect.maxResponseBytes", "-1").is_err());
    for e in [
        &CONNECT_EXECUTE_RETRY_BUFFER_SIZE,
        &CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION,
        &CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE,
        &CONNECT_EXECUTE_DETACHED_TIMEOUT,
        &CONNECT_EXECUTE_MAINTENANCE_INTERVAL,
        &CONNECT_EXECUTE_ABANDONED_TOMBSTONES,
        &CONNECT_MAX_RESPONSE_BYTES,
    ] {
        assert_eq!(e.settable_by, SettableBy::Startup, "{}", e.key);
    }
    // Krishiv's own names, which the Spark names above alias.
    for (e, key) in [
        (
            &CONNECT_EXECUTE_RETRY_BUFFER_SIZE,
            "krishiv.connect.execute.retryBufferSize",
        ),
        (
            &CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION,
            "krishiv.connect.execute.senderMaxStreamDuration",
        ),
        (
            &CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE,
            "krishiv.connect.execute.senderMaxStreamSize",
        ),
        (
            &CONNECT_EXECUTE_DETACHED_TIMEOUT,
            "krishiv.connect.execute.detachedTimeout",
        ),
        (
            &CONNECT_EXECUTE_MAINTENANCE_INTERVAL,
            "krishiv.connect.execute.maintenanceInterval",
        ),
        (
            &CONNECT_EXECUTE_ABANDONED_TOMBSTONES,
            "krishiv.connect.execute.abandonedTombstones",
        ),
    ] {
        assert_eq!(e.key, key);
        let mut k = Config::new();
        k.set(key, &d.get(e).map(ToString::to_string).unwrap_or_default())
            .unwrap();
        assert_eq!(k.get(e), d.get(e), "{key}");
    }
}

#[test]
fn connect_artifact_quotas_parse_and_default() {
    let d = Config::new();
    let size = |c: &Config, e| c.get(e).and_then(ConfigValue::as_size);
    // Design P3-T011a §3.9: 1 GiB per session, 256 MiB per artifact.
    assert_eq!(size(&d, &CONNECT_ARTIFACTS_MAX_BYTES), Some(1 << 30));
    assert_eq!(
        size(&d, &CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES),
        Some(256 << 20)
    );
    let mut c = Config::new();
    c.set("krishiv.connect.artifacts.maxBytes", "2m").unwrap();
    c.set("krishiv.connect.artifacts.maxArtifactBytes", "64k")
        .unwrap();
    assert_eq!(size(&c, &CONNECT_ARTIFACTS_MAX_BYTES), Some(2 << 20));
    assert_eq!(
        size(&c, &CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES),
        Some(64 << 10)
    );
    assert!(c.set("krishiv.connect.artifacts.maxBytes", "-1").is_err());
    assert!(
        c.set("krishiv.connect.artifacts.maxArtifactBytes", "lots")
            .is_err()
    );
    for e in [
        &CONNECT_ARTIFACTS_MAX_BYTES,
        &CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES,
        &CONNECT_ARTIFACTS_MAX_TOTAL_BYTES,
    ] {
        assert_eq!(e.settable_by, SettableBy::Startup, "{}", e.key);
        assert!(e.spark_alias.is_none(), "{}", e.key);
    }
}
