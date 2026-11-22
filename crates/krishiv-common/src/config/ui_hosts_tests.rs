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

#[test]
fn ui_embed_keys_default_closed_and_parse() {
    let mut c = Config::new();
    assert_eq!(c.get(&UI_ALLOWED_ORIGINS), None);
    assert_eq!(
        c.get(&UI_EMBED_HIDE_ATTRIBUTION)
            .and_then(ConfigValue::as_bool),
        Some(false)
    );
    c.set("krishiv.ui.allowedOrigins", "https://console.example.com")
        .unwrap();
    c.set("krishiv.ui.embed.hideAttribution", "true").unwrap();
    c.set("krishiv.ui.embed.linkTtl", "5m").unwrap();
    assert_eq!(
        c.get(&UI_EMBED_LINK_TTL),
        Some(&ConfigValue::Duration(std::time::Duration::from_secs(300)))
    );
    assert!(c.set("krishiv.ui.embed.hideAttribution", "maybe").is_err());
    for e in [
        &UI_ALLOWED_ORIGINS,
        &UI_EMBED_HIDE_ATTRIBUTION,
        &UI_EMBED_LINK_TTL,
    ] {
        assert_eq!(e.scope, Scope::Cluster);
    }
}
