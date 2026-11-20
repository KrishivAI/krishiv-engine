// SPDX-License-Identifier: Apache-2.0
//! `krishiv doctor` (P3-T070): exit status and output of the binary.
#![cfg(feature = "server")]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::process::Command;

/// Whether the tests run as root, where the `user` check fails unless `--allow-root` is given.
fn is_root() -> bool {
    let out = Command::new("id").arg("-u").output().expect("run id -u");
    String::from_utf8_lossy(&out.stdout).trim() == "0"
}

fn doctor(args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_krishiv"));
    cmd.arg("doctor")
        .args(["--port", "0", "--connect-port", "0"]);
    if is_root() {
        cmd.arg("--allow-root");
    }
    cmd.args(args).output().expect("run krishiv doctor")
}

#[test]
fn a_default_configuration_passes_with_a_warning() {
    let out = doctor(&[]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("ok      config"), "{text}");
    assert!(text.contains("warn    authentication"), "{text}");
    let warnings = if is_root() { 2 } else { 1 };
    assert!(
        text.contains(&format!("no failures, {warnings} warning(s)")),
        "{text}"
    );
}

#[test]
fn a_non_loopback_address_without_authentication_fails() {
    let out = doctor(&["--host", "0.0.0.0"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("FAIL    listener"), "{text}");
    assert!(text.contains("needs an authentication method"), "{text}");
}

#[test]
fn a_bad_setting_fails_and_skips_the_rest() {
    let out = doctor(&["--conf", "krishiv.ui.enabled=maybe"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("FAIL    config"), "{text}");
    assert!(text.contains("skipped tls"), "{text}");
}

#[test]
fn an_address_that_does_not_resolve_fails_and_keeps_every_check_row() {
    let out = doctor(&["--json", "--conf", "krishiv.connect.address=not-an-ip"]);
    assert_eq!(out.status.code(), Some(1));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert_eq!(v["ok"], false);
    let checks: Vec<(&str, &str)> = v["checks"]
        .as_array()
        .expect("checks")
        .iter()
        .map(|c| {
            (
                c["name"].as_str().expect("name"),
                c["status"].as_str().expect("status"),
            )
        })
        .collect();
    assert_eq!(
        checks,
        [
            ("config", "ok"),
            ("listener", "fail"),
            ("authentication", "skipped"),
            ("tls", "skipped"),
            ("ports", "skipped"),
            ("user", "skipped"),
            ("directories", "skipped"),
            ("paths", "skipped"),
            ("object store", "skipped"),
            ("catalogs", "skipped"),
        ]
    );
}

#[test]
fn json_output_says_ok_and_lists_the_checks() {
    let out = doctor(&["--json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert_eq!(v["ok"], true);
    assert_eq!(out.status.code(), Some(0));
    let checks: Vec<(&str, &str)> = v["checks"]
        .as_array()
        .expect("checks")
        .iter()
        .map(|c| {
            (
                c["name"].as_str().expect("name"),
                c["status"].as_str().expect("status"),
            )
        })
        .collect();
    let user = if is_root() { "warn" } else { "ok" };
    assert_eq!(
        checks,
        [
            ("config", "ok"),
            ("listener", "ok"),
            ("authentication", "warn"),
            ("tls", "ok"),
            ("ports", "ok"),
            ("user", user),
            ("directories", "ok"),
            ("paths", "ok"),
            ("object store", "skipped"),
            ("catalogs", "skipped"),
        ]
    );
}
