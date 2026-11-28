// SPDX-License-Identifier: Apache-2.0
//! `KRISHIV_KERNEL_TARGET` at start-up: `krishiv` warns about a value that names no kernel target
//! (the kernels ignore it) and goes on; no cap, an empty one or a valid one prints nothing.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::ffi::OsStr;
use std::process::{Command, Output};

use krishiv_kernels::TARGET_CAP_ENV;

fn version(cap: Option<&OsStr>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_krishiv"));
    cmd.arg("--version");
    match cap {
        Some(value) => cmd.env(TARGET_CAP_ENV, value),
        None => cmd.env_remove(TARGET_CAP_ENV),
    };
    cmd.output().expect("run krishiv --version")
}

#[test]
fn an_invalid_cap_is_reported_and_ignored() {
    let out = version(Some(OsStr::new("avx512")));
    assert!(out.status.success(), "{out:?}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains(TARGET_CAP_ENV) && err.contains("\"avx512\"") && err.contains("ignored"),
        "{err}"
    );
}

#[cfg(unix)]
#[test]
fn a_cap_that_is_not_unicode_is_reported() {
    use std::os::unix::ffi::OsStrExt;
    let out = version(Some(OsStr::from_bytes(b"\xff")));
    assert!(out.status.success(), "{out:?}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains(TARGET_CAP_ENV) && err.contains("Unicode"),
        "{err}"
    );
}

#[test]
fn no_cap_an_empty_cap_and_a_valid_cap_print_nothing() {
    for cap in [None, Some(""), Some("x86_64v3")] {
        let out = version(cap.map(OsStr::new));
        assert!(out.status.success(), "{cap:?}: {out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!err.contains(TARGET_CAP_ENV), "{cap:?}: {err}");
    }
}
