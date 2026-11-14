// SPDX-License-Identifier: Apache-2.0
//! Build script: bakes the git commit into the crate as `KRISHIV_GIT_SHA`.
//!
//! A release build sets `KRISHIV_GIT_SHA` itself; without it the commit of the checkout is read with
//! `git`. When neither is available (a build from a published crate, no `git`) the variable stays unset
//! and the crate reports `unknown`.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=KRISHIV_GIT_SHA");
    if std::env::var_os("KRISHIV_GIT_SHA").is_some() {
        return;
    }
    let sha = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    if let Some(sha) = sha {
        println!("cargo:rustc-env=KRISHIV_GIT_SHA={sha}");
    }
}
