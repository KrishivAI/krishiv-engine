// SPDX-License-Identifier: Apache-2.0
//! Flight SQL config entries: defaults, parsing and who may set them (P3-T101).

use std::time::Duration;

use super::*;

#[test]
fn flight_sql_entries_default_and_parse() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&FLIGHT_SQL_ENABLED).and_then(ConfigValue::as_bool),
        Some(true)
    );
    assert_eq!(
        c.get(&FLIGHT_SQL_HANDLE_TTL)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(600))
    );
    for (entry, default) in [
        (&FLIGHT_SQL_MAX_HANDLES_PER_SESSION, 64),
        (&FLIGHT_SQL_MAX_HANDLES_PER_PRINCIPAL, 1024),
        (&FLIGHT_SQL_MAX_BATCH_ROWS, 10_000),
        (&FLIGHT_SQL_MAX_SCHEMA_TABLES, 1000),
    ] {
        assert_eq!(c.get(entry).and_then(ConfigValue::as_int), Some(default));
    }
    assert_eq!(
        c.get(&FLIGHT_SQL_METADATA_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(60))
    );
    c.set("krishiv.flightSql.handleTtl", "30s").unwrap();
    assert_eq!(
        c.get(&FLIGHT_SQL_HANDLE_TTL)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(30))
    );
    assert!(
        c.set("krishiv.flightSql.maxHandlesPerSession", "0")
            .is_err()
    );
    assert!(c.set("krishiv.flightSql.enabled", "maybe").is_err());
}

#[test]
fn flight_sql_session_and_size_entries_default() {
    let c = Config::new();
    assert_eq!(
        c.get(&FLIGHT_SQL_PROVISIONAL_TTL)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(60))
    );
    assert_eq!(
        c.get(&FLIGHT_SQL_FETCH_IDLE_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(300))
    );
    for (entry, default) in [
        (&FLIGHT_SQL_MAX_STATEMENT_BYTES, 1_u64 << 20),
        (&FLIGHT_SQL_MAX_COMMAND_RESULT_BYTES, 16 << 20),
        (&FLIGHT_SQL_MAX_HANDLE_BYTES_PER_PRINCIPAL, 64 << 20),
    ] {
        assert_eq!(c.get(entry), Some(&ConfigValue::Size(default)));
        assert_eq!(entry.settable_by, SettableBy::Startup);
    }
}

#[test]
fn every_flight_sql_key_sets_its_entry_by_name() {
    let mut c = Config::new();
    for (key, value) in [
        ("krishiv.flightSql.maxHandlesPerPrincipal", "7"),
        ("krishiv.flightSql.maxBatchRows", "8"),
        ("krishiv.flightSql.maxSchemaTables", "9"),
        ("krishiv.flightSql.metadataTimeout", "11s"),
        ("krishiv.flightSql.provisionalTtl", "12s"),
        ("krishiv.flightSql.fetchIdleTimeout", "13s"),
        ("krishiv.flightSql.maxStatementBytes", "2m"),
        ("krishiv.flightSql.maxCommandResultBytes", "3m"),
        ("krishiv.flightSql.maxHandleBytesPerPrincipal", "4m"),
    ] {
        c.set(key, value).unwrap();
    }
    for (entry, want) in [
        (&FLIGHT_SQL_MAX_HANDLES_PER_PRINCIPAL, 7),
        (&FLIGHT_SQL_MAX_BATCH_ROWS, 8),
        (&FLIGHT_SQL_MAX_SCHEMA_TABLES, 9),
    ] {
        assert_eq!(c.get(entry).and_then(ConfigValue::as_int), Some(want));
    }
    for (entry, secs) in [
        (&FLIGHT_SQL_METADATA_TIMEOUT, 11),
        (&FLIGHT_SQL_PROVISIONAL_TTL, 12),
        (&FLIGHT_SQL_FETCH_IDLE_TIMEOUT, 13),
    ] {
        assert_eq!(
            c.get(entry).and_then(ConfigValue::as_duration),
            Some(Duration::from_secs(secs))
        );
    }
    for (entry, mib) in [
        (&FLIGHT_SQL_MAX_STATEMENT_BYTES, 2_u64),
        (&FLIGHT_SQL_MAX_COMMAND_RESULT_BYTES, 3),
        (&FLIGHT_SQL_MAX_HANDLE_BYTES_PER_PRINCIPAL, 4),
    ] {
        assert_eq!(c.get(entry), Some(&ConfigValue::Size(mib << 20)));
    }
    assert!(c.set("krishiv.flightSql.maxBatchRows", "0").is_err());
    assert!(c.set("krishiv.flightSql.provisionalTtl", "soon").is_err());
}
