// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.optimizer.derivedKeysAbove` switch (kept out of `tests.rs`, which is at its size cap).

use super::*;

#[test]
fn derived_keys_above_defaults_on_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&DERIVED_KEYS_ABOVE).and_then(ConfigValue::as_bool),
        Some(true)
    );
    c.set("krishiv.optimizer.derivedKeysAbove", "false")
        .unwrap();
    assert_eq!(
        c.get(&DERIVED_KEYS_ABOVE).and_then(ConfigValue::as_bool),
        Some(false)
    );
    assert!(
        c.set("krishiv.optimizer.derivedKeysAbove", "maybe")
            .is_err()
    );
}
