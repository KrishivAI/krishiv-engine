// SPDX-License-Identifier: Apache-2.0
//! Tests of the owned directory guard.

use super::{OwnedDirectory, OwnedDirectoryError};

#[test]
fn a_dropped_guard_removes_only_the_directory_it_made() {
    let tmp = tempfile::tempdir().unwrap();
    // Dropped without `keep` (a failed or cancelled statement): gone, with what was written.
    let a = tmp.path().join("a");
    let guard = OwnedDirectory::create(a.clone()).unwrap();
    std::fs::write(a.join("f"), b"x").unwrap();
    drop(guard);
    assert!(!a.exists());
    // Kept: stays.
    let b = tmp.path().join("b");
    OwnedDirectory::create(b.clone()).unwrap().keep();
    assert!(b.exists());
    // A path that already exists is not ours.
    assert!(matches!(
        OwnedDirectory::create(b.clone()),
        Err(OwnedDirectoryError::Exists)
    ));
    assert!(b.exists());
    // The path now names another directory (swapped meanwhile): left alone.
    let c = tmp.path().join("c");
    let guard = OwnedDirectory::create(c.clone()).unwrap();
    // Renamed aside, not removed: a removed directory's inode number may be reused by the new one.
    std::fs::rename(&c, tmp.path().join("c-moved")).unwrap();
    std::fs::create_dir(&c).unwrap();
    std::fs::write(c.join("theirs"), b"y").unwrap();
    drop(guard);
    assert!(c.join("theirs").exists());
}
