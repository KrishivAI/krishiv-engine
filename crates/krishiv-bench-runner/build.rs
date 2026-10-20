// SPDX-License-Identifier: Apache-2.0
//! Records the cargo profile and the `target-cpu` this runner is built with, so a bench result
//! can say what it measured (`krishiv-bench-runner --build-info`).

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-changed=build.rs");
    // OUT_DIR is <target>[/<triple>]/<profile dir>/build/<package>-<hash>/out.
    let profile = std::env::var_os("OUT_DIR")
        .as_deref()
        .map(Path::new)
        .and_then(|p| p.ancestors().nth(3))
        .and_then(Path::file_name)
        .map_or_else(
            || "unknown".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
    println!("cargo:rustc-env=KRISHIV_BUILD_PROFILE={profile}");
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    println!(
        "cargo:rustc-env=KRISHIV_BUILD_TARGET_CPU={}",
        target_cpu(&flags)
    );
}

/// The last `target-cpu` named in `\x1f`-separated rustc flags, or `default`.
fn target_cpu(encoded_flags: &str) -> String {
    let mut cpu = "default".to_owned();
    let mut args = encoded_flags.split('\u{1f}');
    while let Some(a) = args.next() {
        let value = if a == "-C" || a == "-Ctarget-cpu" {
            args.next().unwrap_or_default()
        } else {
            a.strip_prefix("-C").unwrap_or(a)
        };
        if let Some(v) = value.strip_prefix("target-cpu=") {
            v.clone_into(&mut cpu);
        }
    }
    cpu
}
