// SPDX-License-Identifier: Apache-2.0
//! `krishiv-bench-runner --build-info`: the line the bench harness records in a result.

use std::process::Command;

#[test]
fn build_info_names_the_profile_the_target_cpu_and_the_branch_alignment() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"))
        .arg("--build-info")
        .output()
        .expect("runner starts");
    assert!(out.status.success(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    // The run's kernel target follows the build fields; this test process sees the same environment.
    assert_eq!(
        text.trim_end(),
        format!(
            "{} kernel-target={}",
            krishiv_bench_runner::BUILD_INFO,
            krishiv_kernels::Target::best().name()
        )
    );
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
    assert!(
        fields
            .next()
            .is_some_and(|f| f == "branch-align=32B" || f == "branch-align=off"),
        "{text}"
    );
    assert!(
        fields.next().is_some_and(|f| f
            .strip_prefix("kernel-target=")
            .and_then(krishiv_kernels::Target::from_name)
            .is_some()),
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
