// SPDX-License-Identifier: Apache-2.0
use std::collections::BTreeSet;
use std::time::Duration;

use super::*;
use crate::error::ErrorClass;

#[test]
fn defaults_are_returned_when_unset() {
    let c = Config::new();
    assert_eq!(c.get(&ANSI_ENABLED), Some(&ConfigValue::Bool(true)));
    assert_eq!(
        c.get(&TARGET_PARTITION_BYTES),
        Some(&ConfigValue::Size(128 * 1024 * 1024))
    );
    assert_eq!(c.get(&SESSION_TIME_ZONE), None);
    assert_eq!(
        c.get(&ANSI_ENABLED).and_then(ConfigValue::as_bool),
        Some(true)
    );
}

#[test]
fn parses_each_value_type() {
    let mut c = Config::new();
    c.set("krishiv.sql.ansi.enabled", " FALSE ").unwrap();
    c.set("krishiv.exec.targetPartitionBytes", "64m").unwrap();
    c.set("krishiv.network.timeout", "90s").unwrap();
    c.set("krishiv.sql.shuffle.partitionsHint", "400").unwrap();
    c.set("krishiv.sql.storeAssignmentPolicy", "legacy")
        .unwrap();
    c.set("krishiv.sql.session.timeZone", "America/Los_Angeles")
        .unwrap();
    assert_eq!(c.get(&ANSI_ENABLED), Some(&ConfigValue::Bool(false)));
    assert_eq!(
        c.get(&TARGET_PARTITION_BYTES),
        Some(&ConfigValue::Size(64 * 1024 * 1024))
    );
    assert_eq!(
        c.get(&NETWORK_TIMEOUT).and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(90))
    );
    assert_eq!(
        c.get(&SHUFFLE_PARTITIONS_HINT),
        Some(&ConfigValue::Int(400))
    );
    // Enum values match case-insensitively and are stored canonically.
    assert_eq!(
        c.get(&STORE_ASSIGNMENT_POLICY)
            .and_then(ConfigValue::as_str),
        Some("LEGACY")
    );
    assert_eq!(
        c.get(&SESSION_TIME_ZONE).and_then(ConfigValue::as_str),
        Some("America/Los_Angeles")
    );
}

#[test]
fn size_and_duration_units() {
    let size = |s| ValueType::Size.parse(s);
    assert_eq!(size("1024"), Ok(ConfigValue::Size(1024)));
    assert_eq!(size("2k"), Ok(ConfigValue::Size(2048)));
    assert_eq!(size("1GB"), Ok(ConfigValue::Size(1 << 30)));
    assert!(size("12xb").is_err());
    assert!(size("-1").is_err());
    assert!(size("99999999999999999999t").is_err());
    let dur = |s| ValueType::Duration.parse(s);
    assert_eq!(
        dur("120"),
        Ok(ConfigValue::Duration(Duration::from_secs(120)))
    );
    assert_eq!(
        dur("250ms"),
        Ok(ConfigValue::Duration(Duration::from_millis(250)))
    );
    assert_eq!(
        dur("2min"),
        Ok(ConfigValue::Duration(Duration::from_secs(120)))
    );
    assert_eq!(
        dur("1h"),
        Ok(ConfigValue::Duration(Duration::from_secs(3600)))
    );
    assert!(dur("1 fortnight").is_err());
}

#[test]
fn metrics_entries_parse_and_default() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&METRICS_ENABLED).and_then(ConfigValue::as_bool),
        Some(false)
    );
    c.set("krishiv.metrics.enabled", "true").unwrap();
    assert_eq!(
        c.get(&METRICS_ENABLED).and_then(ConfigValue::as_bool),
        Some(true)
    );
    for bad in ["yes", "1", ""] {
        let e = c.set("krishiv.metrics.enabled", bad).unwrap_err();
        assert_eq!(
            e.error_class(),
            crate::error::ErrorClass::KrishivInvalidConfigValue,
            "{bad}"
        );
    }
    // A rejected value leaves the setting.
    assert_eq!(
        c.get(&METRICS_ENABLED).and_then(ConfigValue::as_bool),
        Some(true)
    );
}

#[test]
fn memory_entries_parse_and_default() {
    let mut c = Config::new();
    assert_eq!(c.get(&MEMORY_LIMIT), None);
    assert_eq!(
        c.get(&MEMORY_RESERVATION_QUANTUM)
            .and_then(ConfigValue::as_size),
        Some(1024 * 1024)
    );
    assert_eq!(
        c.get(&MEMORY_SPILL_WAIT_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(2))
    );
    c.set("krishiv.memory.limit", "2g").unwrap();
    c.set("krishiv.memory.reservationQuantum", "64k").unwrap();
    c.set("krishiv.memory.spillWaitTimeout", "500ms").unwrap();
    assert_eq!(
        c.get(&MEMORY_ABORT_RELEASE_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(30))
    );
    c.set("krishiv.memory.abortReleaseTimeout", "1min").unwrap();
    assert_eq!(
        c.get(&MEMORY_ABORT_RELEASE_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(60))
    );
    assert_eq!(
        c.get(&MEMORY_LIMIT).and_then(ConfigValue::as_size),
        Some(2 << 30)
    );
    assert_eq!(
        c.get(&MEMORY_RESERVATION_QUANTUM)
            .and_then(ConfigValue::as_size),
        Some(64 << 10)
    );
    assert_eq!(
        c.get(&MEMORY_SPILL_WAIT_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_millis(500))
    );
    assert!(c.set("krishiv.memory.limit", "lots").is_err());
}

#[test]
fn the_exec_seed_is_unset_until_given() {
    let mut c = Config::new();
    assert_eq!(c.get(&EXEC_SEED), None);
    c.set("krishiv.exec.seed", "-7").unwrap();
    assert_eq!(c.get(&EXEC_SEED).and_then(ConfigValue::as_int), Some(-7));
    assert!(c.set("krishiv.exec.seed", "abc").is_err());
}

#[test]
fn spill_entries_parse_and_default() {
    let mut c = Config::new();
    let get = |c: &Config| {
        (
            c.get(&SPILL_DIR)
                .and_then(ConfigValue::as_str)
                .map(str::to_owned),
            c.get(&SPILL_MAX_BYTES).and_then(ConfigValue::as_size),
            c.get(&SPILL_COMPRESSION)
                .and_then(ConfigValue::as_str)
                .map(str::to_owned),
            c.get(&SPILL_ZSTD_LEVEL).and_then(ConfigValue::as_int),
        )
    };
    assert_eq!(get(&c), (None, Some(64 << 30), Some("lz4".into()), Some(1)));
    c.set("krishiv.spill.dir", "/mnt/ssd/spill").unwrap();
    c.set("krishiv.spill.maxBytes", "10g").unwrap();
    c.set("krishiv.spill.compression", "ZSTD").unwrap();
    c.set("krishiv.spill.zstd.level", "9").unwrap();
    assert_eq!(
        get(&c),
        (
            Some("/mnt/ssd/spill".into()),
            Some(10 << 30),
            Some("zstd".into()),
            Some(9)
        )
    );
    assert!(c.set("krishiv.spill.compression", "gzip").is_err());
    assert!(c.set("krishiv.spill.maxBytes", "lots").is_err());
}

#[test]
fn avro_entries_parse_and_default() {
    let mut c = Config::new();
    let get = |c: &Config| {
        (
            c.get(&AVRO_COMPRESSION_CODEC)
                .and_then(ConfigValue::as_str)
                .map(str::to_owned),
            c.get(&AVRO_DATETIME_REBASE_MODE_IN_READ)
                .and_then(ConfigValue::as_str)
                .map(str::to_owned),
        )
    };
    assert_eq!(get(&c), (Some("snappy".into()), Some("EXCEPTION".into())));
    c.set("spark.sql.avro.compression.codec", "DEFLATE")
        .unwrap();
    c.set("spark.sql.avro.datetimeRebaseModeInRead", "corrected")
        .unwrap();
    assert_eq!(get(&c), (Some("deflate".into()), Some("CORRECTED".into())));
    // The Krishiv keys address the same entries as the Spark aliases.
    c.set("krishiv.sql.avro.compression.codec", "uncompressed")
        .unwrap();
    c.set("krishiv.sql.avro.datetimeRebaseModeInRead", "LEGACY")
        .unwrap();
    assert_eq!(
        get(&c),
        (Some("uncompressed".into()), Some("LEGACY".into()))
    );
    assert!(c.set("spark.sql.avro.compression.codec", "gzip").is_err());
    assert!(
        c.set("spark.sql.avro.datetimeRebaseModeInRead", "never")
            .is_err()
    );
}

#[test]
fn map_key_dedup_policy_parses_and_defaults() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&MAP_KEY_DEDUP_POLICY).and_then(ConfigValue::as_str),
        Some("EXCEPTION")
    );
    c.set("krishiv.sql.mapKeyDedupPolicy", "LAST_WIN").unwrap();
    assert_eq!(
        c.get(&MAP_KEY_DEDUP_POLICY).and_then(ConfigValue::as_str),
        Some("LAST_WIN")
    );
    // The Spark alias sets the same entry.
    c.set("spark.sql.mapKeyDedupPolicy", "EXCEPTION").unwrap();
    assert_eq!(
        c.get(&MAP_KEY_DEDUP_POLICY).and_then(ConfigValue::as_str),
        Some("EXCEPTION")
    );
    assert!(c.set("spark.sql.mapKeyDedupPolicy", "FIRST_WIN").is_err());
    assert_eq!(
        c.get(&MAP_KEY_DEDUP_POLICY).and_then(ConfigValue::as_str),
        Some("EXCEPTION")
    );
}

#[test]
fn invalid_values_are_typed_errors() {
    let mut c = Config::new();
    for (k, v) in [
        ("krishiv.sql.ansi.enabled", "1"),
        ("krishiv.exec.targetPartitionBytes", "big"),
        ("krishiv.sql.shuffle.partitionsHint", "4.5"),
        ("krishiv.sql.storeAssignmentPolicy", "LOOSE"),
        ("spark.sql.ansi.enabled", "yes"),
        ("spark.sql.legacy.ctePrecedencePolicy", "sometimes"),
    ] {
        let e = c.set(k, v).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivInvalidConfigValue,
            "{k}"
        );
        assert_eq!(e.sql_state(), Some("22023"));
    }
    // A failed set leaves the previous value in place.
    assert_eq!(c.get(&ANSI_ENABLED), Some(&ConfigValue::Bool(true)));
}

#[test]
fn cte_precedence_policy_parses_and_defaults() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&CTE_PRECEDENCE_POLICY).and_then(ConfigValue::as_str),
        Some("CORRECTED")
    );
    for (raw, canonical) in [
        ("EXCEPTION", "EXCEPTION"),
        ("legacy", "LEGACY"),
        (" Corrected ", "CORRECTED"),
    ] {
        c.set("spark.sql.legacy.ctePrecedencePolicy", raw).unwrap();
        assert_eq!(
            c.get(&CTE_PRECEDENCE_POLICY).and_then(ConfigValue::as_str),
            Some(canonical),
            "{raw}"
        );
    }
    let e = c
        .set("spark.sql.legacy.ctePrecedencePolicy", "sometimes")
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(e.message().contains("sometimes"), "{e}");
    assert!(e.message().contains("EXCEPTION, LEGACY, CORRECTED"), "{e}");
    // A rejected value leaves the previous one in place.
    assert_eq!(
        c.get(&CTE_PRECEDENCE_POLICY).and_then(ConfigValue::as_str),
        Some("CORRECTED")
    );
    // The Krishiv key takes the same values as its Spark alias.
    c.set("krishiv.sql.legacy.ctePrecedencePolicy", "exception")
        .unwrap();
    assert_eq!(
        c.get(&CTE_PRECEDENCE_POLICY).and_then(ConfigValue::as_str),
        Some("EXCEPTION")
    );
}

#[test]
fn spark_aliases_map_to_krishiv_keys() {
    let mut c = Config::new();
    assert_eq!(c.set("spark.sql.ansi.enabled", "false").unwrap(), None);
    assert_eq!(c.get(&ANSI_ENABLED), Some(&ConfigValue::Bool(false)));
    assert_eq!(
        lookup("spark.sql.session.timeZone").map(|e| e.key),
        Some(SESSION_TIME_ZONE.key)
    );
    assert_eq!(
        lookup("spark.sql.legacy.ctePrecedencePolicy").map(|e| e.key),
        Some(CTE_PRECEDENCE_POLICY.key)
    );
    assert_eq!(
        lookup("krishiv.sql.ansi.enabled").map(|e| e.key),
        Some(ANSI_ENABLED.key)
    );
}

#[test]
fn unknown_spark_key_warns_and_is_kept_verbatim() {
    let mut c = Config::new();
    let w = c.set("spark.some.unknown", "x").unwrap();
    assert_eq!(
        w,
        Some(ConfigWarning::UnknownKey {
            key: "spark.some.unknown".to_owned()
        })
    );
    assert_eq!(c.get_unregistered("spark.some.unknown"), Some("x"));
}

#[test]
fn unknown_krishiv_key_is_an_error() {
    let mut c = Config::new();
    let e = c.set("krishiv.no.such.key", "1").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnknownConfig);
    assert!(e.message().contains("krishiv.no.such.key"));
    let e = c
        .set("krishiv.experimental.noSuchFeature", "true")
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnknownConfig);
}

#[test]
fn from_pairs_collects_warnings_and_stops_at_first_error() {
    let (c, w) = Config::from_pairs([
        ("spark.sql.ansi.enabled", "false"),
        ("spark.unknown.a", "1"),
    ])
    .unwrap();
    assert_eq!(c.get(&ANSI_ENABLED), Some(&ConfigValue::Bool(false)));
    assert_eq!(w.len(), 1);
    assert!(Config::from_pairs([("krishiv.bogus", "1")]).is_err());
}

#[test]
fn registry_is_well_formed() {
    let mut keys = BTreeSet::new();
    for e in REGISTRY {
        assert!(e.key.starts_with("krishiv."), "{}", e.key);
        assert!(keys.insert(e.key), "duplicate key {}", e.key);
        if let Some(alias) = e.spark_alias {
            assert!(alias.starts_with("spark."), "{alias}");
            assert!(keys.insert(alias), "duplicate alias {alias}");
        }
        assert!(
            !e.doc.is_empty() && !e.version_added.is_empty(),
            "{}",
            e.key
        );
        if let Some(d) = &e.default {
            // Defaults have the entry's type and round-trip through the parser.
            assert_eq!(
                e.value_type.parse(&d.to_string()).as_ref(),
                Ok(d),
                "{}",
                e.key
            );
        }
        if e.key.starts_with(crate::experimental::PREFIX) {
            assert_eq!(e.value_type, ValueType::Bool, "{}", e.key);
            assert_eq!(e.default, Some(ConfigValue::Bool(false)), "{}", e.key);
        }
    }
}

#[test]
fn docs_markdown_lists_every_entry_deterministically() {
    let md = docs_markdown();
    assert!(md.starts_with("# Krishiv configuration reference\n"));
    assert!(md.contains("| Key | Type | Default | Scope | Since | Spark alias | Description |"));
    for e in REGISTRY {
        assert!(md.contains(&format!("| `{}` |", e.key)), "{}", e.key);
    }
    assert!(md.contains("| `krishiv.exec.targetPartitionBytes` | size | `128MB` | session |"));
    assert!(md.contains("`spark.sql.ansi.enabled`"));
    assert!(md.contains("enum (`ANSI`, `LEGACY`, `STRICT`)"));
    let rows: Vec<&str> = md.lines().filter(|l| l.starts_with("| `")).collect();
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    assert_eq!(rows, sorted);
    assert_eq!(md, docs_markdown());
}

#[test]
fn session_time_zone_is_validated() {
    let mut c = Config::new();
    for ok in [
        "UTC",
        "Z",
        "America/Los_Angeles",
        "Etc/GMT+8",
        "+08:00",
        "-0530",
        "+8",
        "GMT+02:00",
        "PST",
    ] {
        c.set("spark.sql.session.timeZone", ok).unwrap();
        assert_eq!(
            c.get(&SESSION_TIME_ZONE).and_then(ConfigValue::as_str),
            Some(ok)
        );
    }
    for bad in [
        "",
        "Mars/ Olympus",
        "+25:00",
        "UTC+25",
        "GMT-19:00",
        "UT+08:75",
        "+08:75",
        "+8:0",
        "1/2",
        "a//b",
    ] {
        let e = c.set("spark.sql.session.timeZone", bad).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivInvalidConfigValue,
            "{bad}"
        );
    }
}

#[test]
fn partition_overwrite_mode_defaults_to_static() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&PARTITION_OVERWRITE_MODE)
            .and_then(ConfigValue::as_str),
        Some("STATIC")
    );
    c.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .unwrap();
    assert_eq!(
        c.get(&PARTITION_OVERWRITE_MODE)
            .and_then(ConfigValue::as_str),
        Some("DYNAMIC")
    );
    c.set("krishiv.sql.sources.partitionOverwriteMode", "STATIC")
        .unwrap();
    assert_eq!(
        c.get(&PARTITION_OVERWRITE_MODE)
            .and_then(ConfigValue::as_str),
        Some("STATIC")
    );
    let e = c
        .set("spark.sql.sources.partitionOverwriteMode", "SMART")
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidConfigValue);
}

#[test]
fn spark_shuffle_partitions_maps_to_the_hint() {
    let mut c = Config::new();
    assert_eq!(c.set("spark.sql.shuffle.partitions", "64").unwrap(), None);
    assert_eq!(c.get(&SHUFFLE_PARTITIONS_HINT), Some(&ConfigValue::Int(64)));
}

/// Counts `WARN` events seen by a thread-local subscriber.
struct WarnCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl tracing::Subscriber for WarnCounter {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[test]
fn unknown_spark_key_logs_exactly_one_warning_per_key() {
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut c = Config::new();
    tracing::subscriber::with_default(WarnCounter(std::sync::Arc::clone(&count)), || {
        // Mapped keys do not warn.
        c.set("spark.sql.ansi.enabled", "true").unwrap();
        assert!(c.set("spark.unknown.a", "1").unwrap().is_some());
        // Setting the same key again does not warn again.
        assert_eq!(c.set("spark.unknown.a", "2").unwrap(), None);
        assert!(c.set("spark.unknown.b", "1").unwrap().is_some());
    });
    assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(c.get_unregistered("spark.unknown.a"), Some("2"));
}

#[test]
fn the_lookup_index_is_sorted_and_complete() {
    use super::registry::{INDEX, REGISTRY};
    assert!(
        INDEX
            .windows(2)
            .all(|w| (w[0].0.len(), w[0].0) < (w[1].0.len(), w[1].0)),
        "INDEX must be strictly sorted by (length, name)"
    );
    let mut want: Vec<(&str, &str)> = REGISTRY
        .iter()
        .flat_map(|e| std::iter::once((e.key, e.key)).chain(e.spark_alias.map(|a| (a, e.key))))
        .collect();
    want.sort_unstable_by_key(|(n, _)| (n.len(), *n));
    let got: Vec<(&str, &str)> = INDEX.iter().map(|(n, e)| (*n, e.key)).collect();
    assert_eq!(got, want, "INDEX must hold exactly every key and alias");
}

#[test]
fn positive_ints_reject_zero_and_negatives() {
    for key in [
        "spark.sql.files.minPartitionNum",
        "spark.sql.files.maxPartitionNum",
        "spark.sql.leafNodeDefaultParallelism",
        "spark.sql.sources.bucketing.maxBuckets",
    ] {
        let mut c = Config::new();
        for bad in ["0", "-3", "x"] {
            let e = c.set(key, bad).unwrap_err();
            assert_eq!(
                e.error_class(),
                crate::error::ErrorClass::KrishivInvalidConfigValue,
                "{key}={bad}"
            );
        }
        assert!(c.set(key, "4").is_ok());
    }
    assert_eq!(ValueType::PositiveInt.to_string(), "positive int");
}

#[test]
fn unset_restores_defaults() {
    let mut c = Config::new();
    c.set("spark.sql.ansi.enabled", "false").unwrap();
    c.set("my.app.flag", "1").unwrap();
    c.unset("krishiv.sql.ansi.enabled");
    c.unset("my.app.flag");
    c.unset("never.set");
    assert_eq!(c, Config::new());
}

#[test]
fn io_blocking_threads_parses_and_defaults() {
    let mut c = Config::new();
    let get = |c: &Config| c.get(&IO_BLOCKING_THREADS).and_then(ConfigValue::as_int);
    assert_eq!(get(&c), Some(8));
    c.set("krishiv.io.blockingThreads", "3").unwrap();
    assert_eq!(get(&c), Some(3));
    for bad in ["0", "-1", "two", "1.5"] {
        let e = c.set("krishiv.io.blockingThreads", bad).unwrap_err();
        assert!(
            e.to_string().contains("krishiv.io.blockingThreads"),
            "{bad}: {e}"
        );
    }
    assert_eq!(get(&c), Some(3), "a rejected value leaves the setting");
}

#[test]
fn agent_gate_defaults_to_one_gib_and_parses_sizes() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&AGENT_GATE_BYTES).and_then(ConfigValue::as_size),
        Some(1 << 30)
    );
    c.set("krishiv.agent.gateBytes", "256m").unwrap();
    assert_eq!(
        c.get(&AGENT_GATE_BYTES).and_then(ConfigValue::as_size),
        Some(256 << 20)
    );
    c.set("krishiv.agent.gateBytes", "0").unwrap();
    assert_eq!(
        c.get(&AGENT_GATE_BYTES).and_then(ConfigValue::as_size),
        Some(0)
    );
    assert!(c.set("krishiv.agent.gateBytes", "lots").is_err());
}

#[test]
fn share_cte_instances_defaults_on_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&SHARE_CTE_INSTANCES).and_then(ConfigValue::as_bool),
        Some(true)
    );
    c.set("krishiv.sql.optimizer.shareCteInstances", "false")
        .unwrap();
    assert_eq!(
        c.get(&SHARE_CTE_INSTANCES).and_then(ConfigValue::as_bool),
        Some(false)
    );
    assert!(
        c.set("krishiv.sql.optimizer.shareCteInstances", "maybe")
            .is_err()
    );
}
