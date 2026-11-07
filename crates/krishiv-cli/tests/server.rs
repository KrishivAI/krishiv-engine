// SPDX-License-Identifier: Apache-2.0
//! `krishiv server` and `krishiv.ui.enabled`.
#![cfg(feature = "ui")]

use std::process::Command;

#[test]
fn a_disabled_ui_means_no_listener_and_a_clear_error() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .args([
            "server",
            "--port",
            "0",
            "--conf",
            "krishiv.ui.enabled=false",
        ])
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(err.contains("krishiv.ui.enabled is false"), "{err}");
    let text = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(!text.contains("listening"), "no listener: {text}");
}

#[test]
fn a_bad_setting_is_refused_before_anything_starts() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .args([
            "server",
            "--port",
            "0",
            "--conf",
            "krishiv.ui.enabled=maybe",
        ])
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("listening"));
}
