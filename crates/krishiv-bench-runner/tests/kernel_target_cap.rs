// SPDX-License-Identifier: Apache-2.0
//! `KRISHIV_KERNEL_TARGET` at start-up: an invalid value stops the runner (a per-target run must not
//! measure the detected target under the cap's name); a valid one is reported by `--build-info`.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::ffi::OsStr;
use std::process::{Command, Output};

use krishiv_kernels::{TARGET_CAP_ENV, Target};

fn build_info(cap: Option<&OsStr>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"));
    cmd.arg("--build-info");
    match cap {
        Some(value) => cmd.env(TARGET_CAP_ENV, value),
        None => cmd.env_remove(TARGET_CAP_ENV),
    };
    cmd.output().expect("runner starts")
}

fn kernel_target(out: &Output) -> String {
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace()
        .find_map(|f| f.strip_prefix("kernel-target="))
        .unwrap_or_else(|| panic!("no kernel-target field: {text}"))
        .to_owned()
}

#[test]
fn an_invalid_cap_stops_the_runner_and_names_the_value() {
    let out = build_info(Some(OsStr::new("x86-64-v3")));
    assert!(!out.status.success(), "{out:?}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains(TARGET_CAP_ENV) && err.contains("\"x86-64-v3\""),
        "{err}"
    );
    assert!(out.stdout.is_empty(), "{out:?}");
}

#[cfg(unix)]
#[test]
fn a_cap_that_is_not_unicode_stops_the_runner() {
    use std::os::unix::ffi::OsStrExt;
    let out = build_info(Some(OsStr::from_bytes(b"\xff")));
    assert!(!out.status.success(), "{out:?}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains(TARGET_CAP_ENV) && err.contains("Unicode"),
        "{err}"
    );
}

#[test]
fn no_cap_runs_at_the_detected_target_without_a_message() {
    for cap in [None, Some(OsStr::new(""))] {
        let out = build_info(cap);
        assert!(out.status.success(), "{cap:?}: {out:?}");
        assert!(out.stderr.is_empty(), "{cap:?}: {out:?}");
        assert_eq!(kernel_target(&out), Target::detected().name(), "{cap:?}");
    }
}

#[test]
fn a_valid_cap_is_the_reported_kernel_target() {
    let out = build_info(Some(OsStr::new("BASELINE")));
    assert!(out.status.success(), "{out:?}");
    assert!(out.stderr.is_empty(), "{out:?}");
    assert_eq!(kernel_target(&out), Target::Baseline.name());
}
