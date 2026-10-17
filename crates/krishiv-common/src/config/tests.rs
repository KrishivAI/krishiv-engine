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
fn invalid_values_are_typed_errors() {
    let mut c = Config::new();
    for (k, v) in [
        ("krishiv.sql.ansi.enabled", "1"),
        ("krishiv.exec.targetPartitionBytes", "big"),
        ("krishiv.sql.shuffle.partitionsHint", "4.5"),
        ("krishiv.sql.storeAssignmentPolicy", "LOOSE"),
        ("spark.sql.ansi.enabled", "yes"),
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
fn spark_aliases_map_to_krishiv_keys() {
    let mut c = Config::new();
    assert_eq!(c.set("spark.sql.ansi.enabled", "false").unwrap(), None);
    assert_eq!(c.get(&ANSI_ENABLED), Some(&ConfigValue::Bool(false)));
    assert_eq!(
        lookup("spark.sql.session.timeZone").map(|e| e.key),
        Some(SESSION_TIME_ZONE.key)
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
