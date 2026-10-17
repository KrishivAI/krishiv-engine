// SPDX-License-Identifier: Apache-2.0
//! `krishiv --version` reports the allocator the binary was built with.

use std::process::Command;

#[test]
fn long_version_reports_allocator() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .arg("--version")
        .output()
        .expect("run krishiv --version");
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).expect("utf-8 output");
    let expected = if cfg!(feature = "mimalloc") {
        "allocator=mimalloc"
    } else {
        "allocator=system"
    };
    assert!(text.lines().any(|l| l == expected), "got: {text}");
    assert!(text.starts_with("krishiv "), "got: {text}");
}
