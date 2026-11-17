// SPDX-License-Identifier: Apache-2.0
#![expect(clippy::disallowed_methods, reason = "the tests read what is on disk")]

use std::io::Write;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::sync::Arc;

use super::{ArtifactRoot, Install, SessionDir};

fn root(parent: &std::path::Path) -> Arc<ArtifactRoot> {
    Arc::new(ArtifactRoot::create(parent, "krishiv-artifacts-test").unwrap())
}

fn stage(dir: &SessionDir, name: &str, bytes: &[u8]) {
    let mut f = dir.create_staged(name).unwrap();
    f.write_all(bytes).unwrap();
}

#[test]
fn the_root_and_session_directories_are_private_and_removed_on_drop() {
    let tmp = tempfile::tempdir().unwrap();
    let root = root(tmp.path());
    let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(root.path()), 0o700);
    // A second root with the same name is refused (it exists).
    assert!(ArtifactRoot::create(tmp.path(), "krishiv-artifacts-test").is_err());
    let dir = SessionDir::create(&root, "s1").unwrap();
    assert_eq!(mode(dir.path()), 0o700);
    assert_eq!(mode(&dir.path().join(".staging")), 0o700);
    stage(&dir, "x", b"abc");
    let staged = dir.path().join(".staging/x");
    assert_eq!(mode(&staged), 0o600);
    // `O_EXCL`: a staged name is never reused.
    assert!(dir.create_staged("x").is_err());
    let root_path = root.path().to_owned();
    let dir_path = dir.path().to_owned();
    drop(root);
    assert!(root_path.exists(), "a session directory keeps the root");
    drop(dir);
    assert!(!dir_path.exists());
    assert!(!root_path.exists());
}

#[test]
fn an_existing_directory_is_never_taken_over() {
    let tmp = tempfile::tempdir().unwrap();
    // Planted before the server starts, open to the group.
    let path = tmp.path().join("planted");
    std::fs::create_dir(&path).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o770)).unwrap();
    assert!(ArtifactRoot::create(tmp.path(), "planted").is_err());
    assert!(path.exists(), "a directory it did not create is left alone");
    // Nor a link in its place.
    symlink(tmp.path(), tmp.path().join("link")).unwrap();
    assert!(ArtifactRoot::create(tmp.path(), "link").is_err());
}

#[test]
fn install_creates_directories_and_never_replaces() {
    let tmp = tempfile::tempdir().unwrap();
    let root = root(tmp.path());
    let dir = SessionDir::create(&root, "s").unwrap();
    stage(&dir, "a", b"one");
    assert_eq!(
        dir.install("a", &["files", "sub", "f.txt"]).unwrap(),
        Install::Added
    );
    assert_eq!(
        std::fs::read(dir.path().join("files/sub/f.txt")).unwrap(),
        b"one"
    );
    assert!(!dir.path().join(".staging/a").exists());
    stage(&dir, "b", b"one");
    assert_eq!(
        dir.install("b", &["files", "sub", "f.txt"]).unwrap(),
        Install::Identical
    );
    stage(&dir, "c", b"two");
    assert_eq!(
        dir.install("c", &["files", "sub", "f.txt"]).unwrap(),
        Install::Exists
    );
    assert_eq!(
        std::fs::read(dir.path().join("files/sub/f.txt")).unwrap(),
        b"one"
    );
    // A file where a directory is needed.
    stage(&dir, "d", b"x");
    assert_eq!(
        dir.install("d", &["files", "sub", "f.txt", "g"]).unwrap(),
        Install::Blocked
    );
    // A directory where the file goes.
    stage(&dir, "e", b"x");
    assert_eq!(
        dir.install("e", &["files", "sub"]).unwrap(),
        Install::Exists
    );
    let left: Vec<_> = std::fs::read_dir(dir.path().join(".staging"))
        .unwrap()
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

#[test]
fn planted_links_are_never_followed() {
    let tmp = tempfile::tempdir().unwrap();
    let outside = tmp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("victim"), b"keep").unwrap();
    let root = root(tmp.path());
    let dir = SessionDir::create(&root, "s").unwrap();
    // A link to a directory outside, where a directory of the path goes.
    symlink(&outside, dir.path().join("files")).unwrap();
    stage(&dir, "a", b"evil");
    assert_eq!(
        dir.install("a", &["files", "victim"]).unwrap(),
        Install::Blocked
    );
    stage(&dir, "b", b"evil");
    assert_eq!(
        dir.install("b", &["files", "new"]).unwrap(),
        Install::Blocked
    );
    assert_eq!(std::fs::read(outside.join("victim")).unwrap(), b"keep");
    assert!(!outside.join("new").exists());
    // A link where the file goes: never written through, never compared
    // as equal.
    std::fs::create_dir(dir.path().join("pyfiles")).unwrap();
    symlink(outside.join("victim"), dir.path().join("pyfiles/a.py")).unwrap();
    stage(&dir, "c", b"keep");
    assert_eq!(
        dir.install("c", &["pyfiles", "a.py"]).unwrap(),
        Install::Exists
    );
    assert_eq!(std::fs::read(outside.join("victim")).unwrap(), b"keep");
    // A link planted in the staging directory under the next staged name.
    symlink(outside.join("victim"), dir.path().join(".staging/d")).unwrap();
    assert!(dir.create_staged("d").is_err());
    assert_eq!(std::fs::read(outside.join("victim")).unwrap(), b"keep");
    // Removing the session's tree removes the links, not their targets.
    drop(dir);
    drop(root);
    assert_eq!(std::fs::read(outside.join("victim")).unwrap(), b"keep");
}
