// SPDX-License-Identifier: Apache-2.0
//! Records the cargo profile, the `target-cpu` and the x86-64 branch alignment this runner is
//! built with, so a bench result can say what it measured (`krishiv-bench-runner --build-info`).

use std::path::Path;

#[path = "src/build_flags.rs"]
mod build_flags;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/build_flags.rs");
    // OUT_DIR is <target>[/<triple>]/<profile dir>/build/<package>-<hash>/out, or with the
    // newer layout (nightly, so under `cargo llvm-cov --branch`) .../build/<package>/<hash>/out:
    // the profile is the directory holding the `build` ancestor.
    let profile = std::env::var_os("OUT_DIR")
        .as_deref()
        .map(Path::new)
        .and_then(|p| {
            p.ancestors()
                .find(|a| a.file_name().is_some_and(|n| n == "build"))
        })
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .map_or_else(
            || "unknown".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
    println!("cargo:rustc-env=KRISHIV_BUILD_PROFILE={profile}");
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    println!(
        "cargo:rustc-env=KRISHIV_BUILD_TARGET_CPU={}",
        build_flags::target_cpu(&flags)
    );
    println!(
        "cargo:rustc-env=KRISHIV_BUILD_BRANCH_ALIGN={}",
        build_flags::branch_align(&flags)
    );
}
