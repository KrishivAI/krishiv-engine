// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.optimizer.payloadAboveAggregate` and `discoverKeys` entries (kept out of `tests.rs`, which is at
//! its size cap).

use std::time::Duration;

use super::*;

#[test]
fn payload_above_aggregate_defaults_on_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&PAYLOAD_ABOVE_AGGREGATE)
            .and_then(ConfigValue::as_bool),
        Some(true)
    );
    c.set("krishiv.optimizer.payloadAboveAggregate", "false")
        .unwrap();
    assert_eq!(
        c.get(&PAYLOAD_ABOVE_AGGREGATE)
            .and_then(ConfigValue::as_bool),
        Some(false)
    );
    assert!(
        c.set("krishiv.optimizer.payloadAboveAggregate", "maybe")
            .is_err()
    );
}

#[test]
fn discover_keys_entries_default_on_and_parse() {
    let mut c = Config::new();
    let get = |c: &Config| {
        (
            c.get(&DISCOVER_KEYS).and_then(ConfigValue::as_bool),
            c.get(&DISCOVER_KEYS_TIMEOUT)
                .and_then(ConfigValue::as_duration),
        )
    };
    assert_eq!(get(&c), (Some(true), Some(Duration::from_secs(60))));
    c.set("krishiv.optimizer.discoverKeys", "false").unwrap();
    c.set("krishiv.optimizer.discoverKeys.timeout", "250ms")
        .unwrap();
    assert_eq!(get(&c), (Some(false), Some(Duration::from_millis(250))));
    assert!(c.set("krishiv.optimizer.discoverKeys", "maybe").is_err());
    assert!(
        c.set("krishiv.optimizer.discoverKeys.timeout", "soon")
            .is_err()
    );
}
