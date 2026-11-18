// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.server.readiness.*` keys (design P3-T066 §5.2).
use std::time::Duration;

use super::super::*;

#[test]
fn readiness_keys_default_and_validate() {
    let c = Config::new();
    assert_eq!(
        c.get(&SERVER_READINESS_INTERVAL)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(10))
    );
    assert_eq!(
        c.get(&SERVER_READINESS_TIMEOUT)
            .and_then(ConfigValue::as_duration),
        Some(Duration::from_secs(2))
    );
    assert_eq!(
        c.get(&SERVER_READINESS_FAILURE_THRESHOLD)
            .and_then(ConfigValue::as_int),
        Some(3)
    );
    assert_eq!(
        c.get(&SERVER_READINESS_REQUIRED)
            .and_then(ConfigValue::as_str),
        Some("")
    );
    let mut c = Config::new();
    assert!(
        c.set("krishiv.server.readiness.failureThreshold", "0")
            .is_err()
    );
}
