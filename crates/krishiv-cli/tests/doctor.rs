// SPDX-License-Identifier: Apache-2.0
//! `krishiv doctor` (P3-T070): exit status and output of the binary.
#![cfg(feature = "server")]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::process::Command;

fn doctor(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .arg("doctor")
        .args(["--port", "0", "--connect-port", "0"])
        .args(args)
        .output()
        .expect("run krishiv doctor")
}

#[test]
fn a_default_configuration_passes_with_a_warning() {
    let out = doctor(&[]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("ok      config"), "{text}");
    assert!(text.contains("warn    authentication"), "{text}");
    assert!(text.contains("no failures, 1 warning(s)"), "{text}");
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
fn json_output_says_ok_and_lists_the_checks() {
    let out = doctor(&["--json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert_eq!(v["ok"], true);
    assert!(v["checks"].as_array().is_some_and(|c| c.len() >= 8));
}
