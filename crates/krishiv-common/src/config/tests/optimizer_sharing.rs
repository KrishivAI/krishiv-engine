// SPDX-License-Identifier: Apache-2.0
//! The optimizer sharing and group-key code keys (`shareFilteredScans`, `shareLiteralInstances`, `rollupFromParent`, `groupKeyCode*`).
use super::super::*;

#[test]
fn share_filtered_scans_defaults_on_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&SHARE_FILTERED_SCANS).and_then(ConfigValue::as_bool),
        Some(true)
    );
    c.set("krishiv.optimizer.shareFilteredScans", "false")
        .unwrap();
    assert_eq!(
        c.get(&SHARE_FILTERED_SCANS).and_then(ConfigValue::as_bool),
        Some(false)
    );
    assert!(
        c.set("krishiv.optimizer.shareFilteredScans", "maybe")
            .is_err()
    );
}

#[test]
fn share_literal_instances_defaults_on_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&SHARE_LITERAL_INSTANCES)
            .and_then(ConfigValue::as_bool),
        Some(true)
    );
    c.set("krishiv.optimizer.shareLiteralInstances", "false")
        .unwrap();
    assert_eq!(
        c.get(&SHARE_LITERAL_INSTANCES)
            .and_then(ConfigValue::as_bool),
        Some(false)
    );
    assert!(
        c.set("krishiv.optimizer.shareLiteralInstances", "maybe")
            .is_err()
    );
}

#[test]
fn group_key_code_defaults_off_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&GROUP_KEY_CODE).and_then(ConfigValue::as_bool),
        Some(false)
    );
    c.set("krishiv.optimizer.groupKeyCode", "true").unwrap();
    assert_eq!(
        c.get(&GROUP_KEY_CODE).and_then(ConfigValue::as_bool),
        Some(true)
    );
    assert!(c.set("krishiv.optimizer.groupKeyCode", "maybe").is_err());
}

#[test]
fn group_key_code_multi_dimension_defaults_off_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&GROUP_KEY_CODE_MULTI_DIMENSION)
            .and_then(ConfigValue::as_bool),
        Some(false)
    );
    c.set("krishiv.optimizer.groupKeyCodeMultiDimension", "true")
        .unwrap();
    assert_eq!(
        c.get(&GROUP_KEY_CODE_MULTI_DIMENSION)
            .and_then(ConfigValue::as_bool),
        Some(true)
    );
    assert!(
        c.set("krishiv.optimizer.groupKeyCodeMultiDimension", "maybe")
            .is_err()
    );
}

#[test]
fn group_key_code_force_is_experimental_and_off() {
    let mut c = Config::new();
    assert!(!crate::experimental::is_enabled(&c, "groupKeyCodeForce"));
    c.set("krishiv.experimental.groupKeyCodeForce", "true")
        .unwrap();
    assert!(crate::experimental::is_enabled(&c, "groupKeyCodeForce"));
    assert!(
        c.set("krishiv.experimental.groupKeyCodeForce", "maybe")
            .is_err()
    );
}

#[test]
fn rollup_from_parent_defaults_on_and_parses() {
    let mut c = Config::new();
    assert_eq!(
        c.get(&ROLLUP_FROM_PARENT).and_then(ConfigValue::as_bool),
        Some(true)
    );
    c.set("krishiv.optimizer.rollupFromParent", "false")
        .unwrap();
    assert_eq!(
        c.get(&ROLLUP_FROM_PARENT).and_then(ConfigValue::as_bool),
        Some(false)
    );
    assert!(
        c.set("krishiv.optimizer.rollupFromParent", "maybe")
            .is_err()
    );
}
