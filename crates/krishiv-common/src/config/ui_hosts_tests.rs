// SPDX-License-Identifier: Apache-2.0
//! `krishiv.ui.allowedHosts` (kept out of `tests.rs`, which is at its size cap).

use super::*;

#[test]
fn ui_allowed_hosts_is_a_startup_string_with_no_default() {
    let mut c = Config::new();
    assert_eq!(c.get(&UI_ALLOWED_HOSTS), None);
    c.set("krishiv.ui.allowedHosts", "ui.example.com,[::1]")
        .unwrap();
    assert_eq!(
        c.get(&UI_ALLOWED_HOSTS).and_then(ConfigValue::as_str),
        Some("ui.example.com,[::1]")
    );
    assert_eq!(UI_ALLOWED_HOSTS.scope, Scope::Cluster);
}
