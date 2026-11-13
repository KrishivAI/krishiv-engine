// SPDX-License-Identifier: Apache-2.0
//! One snapshot per table root per query while a commit lands between two scans' opens; the
//! next query sees the commit; another credential misses the runtime's cache.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::secret::OptionValue;
use krishiv_plan::ReadSource;

use super::{DELTA, DeltaInputs, DeltaReads};
use crate::session::Session;

#[expect(clippy::unwrap_used, reason = "test helper")]
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// `session-appends` without its last commit (kept beside the table), and a function landing it.
#[expect(clippy::unwrap_used, reason = "test helper")]
fn two_commits(tmp: &Path) -> (PathBuf, impl Fn()) {
    let dir = tmp.join("t");
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/delta-4.4.1/session-appends"),
        &dir,
    );
    let last = |ext: &str| dir.join(format!("_delta_log/00000000000000000002.{ext}"));
    for ext in ["json", "crc"] {
        std::fs::rename(last(ext), dir.with_extension(ext)).unwrap();
    }
    let land = {
        let dir = dir.clone();
        move || {
            for ext in ["crc", "json"] {
                std::fs::rename(
                    dir.with_extension(ext),
                    dir.join(format!("_delta_log/00000000000000000002.{ext}")),
                )
                .unwrap();
            }
        }
    };
    (dir, land)
}

fn read(path: &str) -> ReadSource {
    ReadSource::Files {
        format: DELTA.to_owned(),
        paths: vec![OptionValue::from_user("path", path)],
        options: std::collections::BTreeMap::new(),
    }
}

#[test]
fn two_scans_of_a_query_pin_one_version_while_a_commit_lands() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let cache = Arc::clone(&s.state.runtime.delta);
    let query = s.query_env();
    let first = block_on(query.catalog.delta.open(&read(&path), None)).unwrap();
    assert_eq!(first.version(), 1);
    land();
    // The second scan spells the root another way: the same table, the same version.
    let second = block_on(query.catalog.delta.open(&read(&format!("{path}/")), None)).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(second.version(), 1);
    assert_eq!(cache.misses(), 1);
    // The next query starts from the cached snapshot and sees the commit.
    let next = s.query_env();
    let newer = block_on(next.catalog.delta.open(&read(&path), None)).unwrap();
    assert_eq!(newer.version(), 2);
    assert_eq!((cache.hits(), cache.misses()), (1, 1));
    // A query under another credential identity never reads that entry.
    let other = DeltaReads::new(DeltaInputs {
        io: s.state.io.context(),
        paths: Arc::new(krishiv_exec::files::LocalPaths::new(
            krishiv_io::StoreId::new(crate::session::LOCAL_STORE),
            "/",
        )),
        runtime: s.state.runtime.executor.io_handle().clone(),
        config: Arc::new(krishiv_common::config::Config::new()),
        cache: Arc::clone(&cache),
        credential: "someone-else".to_owned(),
    });
    let theirs = block_on(other.open(&read(&path), None)).unwrap();
    assert_eq!(theirs.version(), 2);
    assert_eq!((cache.hits(), cache.misses()), (1, 2));
}

/// A query's reads dropped (its planning future dropped) end the planning-time Delta work: an
/// engine they made refuses the next kernel call.
#[test]
fn dropped_reads_cancel_their_kernel_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let query = s.query_env();
    let (store, root) = query.catalog.delta.root_of(&read(&path)).unwrap();
    let engine = query.catalog.delta.engine(store, root).unwrap();
    assert!(block_on(engine.delta_blocking(|_| ())).is_ok());
    drop(query);
    let e = block_on(engine.delta_blocking(|_| ())).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::OperationCanceled,
        "{e}"
    );
}
