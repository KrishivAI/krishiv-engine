// SPDX-License-Identifier: Apache-2.0
//! `doctor` and `stack` against a faked `PATH` (P2-T123).
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::path::PathBuf;
use std::process::{Command, Output};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xtask-dev-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xtask_with_path(args: &[&str], path: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .env("PATH", path)
        .env("KRISHIV_DATA", path)
        .output()
        .unwrap()
}

#[test]
fn doctor_fails_with_a_named_fix_per_missing_prerequisite() {
    let empty = tmp("empty-path");
    let out = xtask_with_path(&["doctor"], &empty);
    assert!(
        !out.status.success(),
        "doctor must fail with nothing on PATH"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    for (name, fix) in [
        ("rustup", "https://sh.rustup.rs"),
        ("rustc", "rustup toolchain install"),
        ("git", "install git"),
        ("uv", "astral.sh/uv/install.sh"),
        ("cargo-nextest", "cargo install --locked cargo-nextest"),
    ] {
        assert!(
            stdout.contains(&format!("[FAIL] {name}:")),
            "{name} not reported:\n{stdout}"
        );
        assert!(stdout.contains(fix), "fix for {name} missing:\n{stdout}");
        assert!(
            stderr.contains(name),
            "{name} not in the failure summary: {stderr}"
        );
    }
    // Optional items warn but do not fail on their own.
    assert!(stdout.contains("[warn] docker compose:"));
}

#[cfg(unix)]
#[test]
fn stack_only_addresses_the_bench_project() {
    use std::os::unix::fs::PermissionsExt;
    let bin = tmp("fake-docker");
    let log = bin.join("calls.log");
    let script = format!("#!/bin/sh\necho \"$@\" >> {}\nexit 0\n", log.display());
    let docker = bin.join("docker");
    std::fs::write(&docker, script).unwrap();
    std::fs::set_permissions(&docker, std::fs::Permissions::from_mode(0o755)).unwrap();
    for action in ["up", "down", "status"] {
        let out = xtask_with_path(&["stack", action], &bin);
        assert!(out.status.success(), "{action}: {out:?}");
    }
    let calls = std::fs::read_to_string(&log).unwrap();
    let compose: Vec<&str> = calls
        .lines()
        .filter(|l| l.starts_with("compose ") && *l != "compose version")
        .collect();
    assert_eq!(compose.len(), 3, "{calls}");
    for line in compose {
        assert!(line.starts_with("compose -p krishiv-bench -f "), "{line}");
        assert!(
            line.contains("bench/baselines/docker-compose.yml"),
            "{line}"
        );
        assert!(!line.contains(" -v"), "{line}");
    }
}

#[test]
fn stack_without_docker_names_the_fix() {
    let out = xtask_with_path(&["stack", "up"], &tmp("no-docker"));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("install Docker Engine"));
}
