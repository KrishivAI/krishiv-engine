// SPDX-License-Identifier: Apache-2.0
//! Build script: bakes the git commit into the crate as `KRISHIV_GIT_SHA`.
//!
//! A release build sets `KRISHIV_GIT_SHA` itself (it wins). Without it the commit is read with `git`, but only
//! when the repository's top level is this workspace (a vendored or path build inside someone else's checkout
//! must not bake in their commit), and the script reruns when `HEAD`, the branch it names or `packed-refs`
//! change, so an incremental build never reports an older commit. With neither, the variable stays unset and
//! the crate reports `unknown`. (`krishiv-session` carries an identical copy.)

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(args: &[&str], dir: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?.trim().to_owned();
    (!text.is_empty()).then_some(text)
}

fn main() {
    println!("cargo:rerun-if-env-changed=KRISHIV_GIT_SHA");
    if std::env::var_os("KRISHIV_GIT_SHA").is_some() {
        return;
    }
    let Some(manifest) = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from) else {
        return;
    };
    let workspace = manifest.join("../..");
    let (Some(top), Ok(root)) = (
        git(&["rev-parse", "--show-toplevel"], &manifest),
        workspace.canonicalize(),
    ) else {
        return;
    };
    if PathBuf::from(top).canonicalize().ok() != Some(root) {
        return;
    }
    // The files a new commit changes: `HEAD`, the ref it points to, and `packed-refs`.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"], &manifest) {
        let git_dir = PathBuf::from(git_dir);
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join("packed-refs").display()
        );
        if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"], &manifest) {
            // A linked worktree keeps its refs in the common directory.
            let common = git(&["rev-parse", "--git-common-dir"], &manifest)
                .map_or_else(|| git_dir.clone(), |c| manifest.join(c));
            println!(
                "cargo:rerun-if-changed={}",
                common.join(&reference).display()
            );
        }
    }
    if let Some(sha) = git(&["rev-parse", "--short=12", "HEAD"], &manifest) {
        println!("cargo:rustc-env=KRISHIV_GIT_SHA={sha}");
    }
}
