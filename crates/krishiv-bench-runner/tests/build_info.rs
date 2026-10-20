// SPDX-License-Identifier: Apache-2.0
//! `krishiv-bench-runner --build-info`: the line the bench harness records in a result.

use std::process::Command;

#[test]
fn build_info_names_the_profile_and_the_target_cpu() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"))
        .arg("--build-info")
        .output()
        .expect("runner starts");
    assert!(out.status.success(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(text.trim_end(), krishiv_bench_runner::BUILD_INFO);
    // The profile is the cargo output directory the binary sits in (`debug`, `dist`, ...).
    let profile_dir = std::path::Path::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"))
        .parent()
        .and_then(|d| d.file_name())
        .and_then(|d| d.to_str())
        .expect("the binary sits in <target>/<profile>/");
    let mut fields = text.split_whitespace();
    assert_eq!(
        fields.next(),
        Some(format!("profile={profile_dir}").as_str()),
        "{text}"
    );
    assert!(
        fields
            .next()
            .is_some_and(|f| f.starts_with("target-cpu=") && f.len() > "target-cpu=".len()),
        "{text}"
    );
    assert_eq!(fields.next(), None, "{text}");
}

#[test]
fn an_unknown_flag_fails_with_a_message() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"))
        .arg("--nope")
        .output()
        .expect("runner starts");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown argument --nope"));
}
