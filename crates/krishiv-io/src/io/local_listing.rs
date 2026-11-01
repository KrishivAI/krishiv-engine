// SPDX-License-Identifier: Apache-2.0
//! The local store reads use: `LocalFileSystem`, with a listing that does not
//! lose a file replaced while it runs.
//!
//! `LocalFileSystem::list` reads a directory's names and then asks about each
//! one, and quietly skips a name that has gone by then. An overwrite writes its
//! new files before it deletes the old ones, so a reader whose names held only
//! the old files, and whose questions came after the overwrite finished, lists
//! nothing: an overwritten table reads as empty, with no error. Here a pass in
//! which an entry vanished is repeated, as in [`super::beneath`]: the pass that
//! follows a vanished old file lists the new files too.
//!
//! An entry can also go before its directory is read, as when an overwrite adds
//! a partition after the root was read and then empties an old one, and then no
//! question fails. So a pass also compares each directory's modification time
//! from before its names were read with the one after the walk, and is repeated
//! if one moved. Before Linux 6.13 that time moves with the timer tick, so a
//! change in the tick of the first stat can pass unseen. A directory that only
//! ever gains files can keep moving: after the last pass it is answered, where a
//! vanished entry is an error.
//!
//! A listing is known only when its last pass ends, so it answers all at once:
//! register this store with [`super::IoScheduler::register_whole_listing_store`].

use std::fmt;
use std::fs::Metadata;
use std::io::ErrorKind;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::{BoxStream, StreamExt};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as StoreResult,
};

use rustix::io::Errno;

use super::beneath::{SETTLE_PASSES, is_staging};

const STORE: &str = "SettledLocalFileSystem";

/// `LocalFileSystem` rooted at `/`, but for [`ObjectStore::list`].
#[derive(Debug, Default)]
pub struct SettledLocalFileSystem {
    inner: LocalFileSystem,
}

impl SettledLocalFileSystem {
    /// A store over the whole file system.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Display for SettledLocalFileSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

fn io_err(e: std::io::Error) -> object_store::Error {
    object_store::Error::Generic {
        store: STORE,
        source: Box::new(e),
    }
}

/// The identity of a directory, to find a link back to one above it.
fn id_of(m: &Metadata) -> (u64, u64) {
    (m.dev(), m.ino())
}

/// What `LocalFileSystem` says about a file at `location`, ETag included: a
/// conditional read compares it with the one a read of the file answers.
fn meta_of(location: Path, m: &Metadata) -> ObjectMeta {
    let modified = m.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let micros = modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    ObjectMeta {
        location,
        last_modified: DateTime::<Utc>::from(modified),
        size: m.len(),
        e_tag: Some(format!("\"{:x}-{:x}-{:x}\"", m.ino(), micros, m.len())),
        version: None,
    }
}

/// What one pass of [`walk_with`] saw.
#[derive(Default)]
struct Pass {
    /// An entry vanished between the read of the names and its question.
    vanished: bool,
    /// A directory is not as it was before its names were read.
    moved: bool,
}

/// What tells that a directory changed.
fn version_of(m: &Metadata) -> (u64, i64, i64) {
    (m.ino(), m.mtime(), m.mtime_nsec())
}

/// An entry that was there is not: the file or a directory on its path went.
fn gone(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory)
        || e.raw_os_error() == Some(Errno::STALE.raw_os_error())
}

/// The files below `root`, following links, with no staging file. `between`
/// runs after the read of each directory's names and before the questions
/// about them (a test replaces files there).
///
/// # Errors
/// An entry still vanished in the last pass: what it found may be short of
/// every file, even empty, so it is not an answer. The error is one the
/// scheduler retries after a wait. `cancelled` ends the walk with an error too.
fn walk_with(
    root: &FsPath,
    cancelled: &AtomicBool,
    mut between: impl FnMut(),
) -> StoreResult<Vec<ObjectMeta>> {
    let mut out = Vec::new();
    let mut vanished = false;
    for _ in 0..SETTLE_PASSES {
        out.clear();
        let pass = walk_once(root, cancelled, &mut between, &mut out)?;
        if !pass.vanished && !pass.moved {
            return Ok(out);
        }
        vanished = pass.vanished;
    }
    if !vanished {
        return Ok(out);
    }
    Err(io_err(std::io::Error::new(
        ErrorKind::Interrupted,
        format!(
            "files under {} kept vanishing during {SETTLE_PASSES} listings",
            root.display()
        ),
    )))
}

/// One pass of [`walk_with`]. Asks what `LocalFileSystem` asks, and answers
/// alike: a link to nothing is no file, a file that cannot be asked about is
/// left out, and a directory that cannot be read is an error.
fn walk_once(
    root: &FsPath,
    cancelled: &AtomicBool,
    between: &mut impl FnMut(),
    out: &mut Vec<ObjectMeta>,
) -> StoreResult<Pass> {
    let mut pass = Pass::default();
    let root_meta = match std::fs::metadata(root) {
        Ok(m) => m,
        // A prefix with nothing below it lists nothing.
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(pass),
        Err(e) => return Err(io_err(e)),
    };
    if !root_meta.is_dir() {
        return Ok(pass);
    }
    let mut read = Vec::new();
    // A directory still to read, its version, and the directories above it
    // and itself.
    let mut pending = vec![(
        root.to_owned(),
        version_of(&root_meta),
        vec![id_of(&root_meta)],
    )];
    while let Some((dir, version, above)) = pending.pop() {
        let reader = match std::fs::read_dir(&dir) {
            Ok(reader) => reader,
            Err(e) if gone(&e) => {
                pass.vanished |= dir != root;
                continue;
            }
            Err(e) => return Err(io_err(e)),
        };
        let mut names = Vec::new();
        for entry in reader {
            match entry.and_then(|e| Ok((e.path(), e.file_type()?))) {
                Ok(name) => names.push(name),
                Err(e) if gone(&e) => pass.vanished = true,
                Err(e) => return Err(io_err(e)),
            }
        }
        read.push((dir, version));
        between();
        for (path, file_type) in names {
            if cancelled.load(Ordering::Relaxed) {
                return Err(io_err(std::io::Error::other("the listing was dropped")));
            }
            let staging = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_staging);
            // A staging file is renamed at any moment, and is not listed.
            if staging && file_type.is_file() {
                continue;
            }
            let m = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) if file_type.is_symlink() => {
                    if e.kind() != ErrorKind::NotFound {
                        return Err(io_err(e));
                    }
                    // A link to nothing was never a file; a removed link was.
                    pass.vanished |= std::fs::symlink_metadata(&path).is_err();
                    continue;
                }
                Err(e) if gone(&e) => {
                    pass.vanished = true;
                    continue;
                }
                Err(e) if file_type.is_dir() => return Err(io_err(e)),
                Err(_) => continue,
            };
            if m.is_dir() {
                if file_type.is_symlink() && above.contains(&id_of(&m)) {
                    return Err(io_err(std::io::Error::other(format!(
                        "a link back to a directory above it at {}",
                        path.display()
                    ))));
                }
                let mut below = above.clone();
                below.push(id_of(&m));
                pending.push((path, version_of(&m), below));
            } else if m.is_file() && !staging {
                let location = Path::from_absolute_path(&path)
                    .map_err(|source| object_store::Error::InvalidPath { source })?;
                out.push(meta_of(location, &m));
            }
        }
    }
    pass.moved = read
        .iter()
        .any(|(dir, version)| std::fs::metadata(dir).map_or(true, |m| version_of(&m) != *version));
    Ok(pass)
}

/// Tells a walk on a blocking thread that nobody waits for it any more.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[async_trait]
impl ObjectStore for SettledLocalFileSystem {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> StoreResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> StoreResult<Vec<bytes::Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        // Through a child: `path_to_filesystem` refuses a path whose last
        // segment reads like a staging file, and a directory may be so named.
        let child = prefix.cloned().unwrap_or_default().join("child");
        let root = match self.inner.path_to_filesystem(&child) {
            Ok(child) => child.parent().map_or_else(PathBuf::new, FsPath::to_owned),
            Err(e) => return futures::stream::once(std::future::ready(Err(e))).boxed(),
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        let guard = CancelOnDrop(Arc::clone(&cancelled));
        let walk = move || walk_with(&root, &cancelled, || {});
        futures::stream::once(async move {
            let _guard = guard;
            match tokio::runtime::Handle::try_current() {
                Ok(runtime) => runtime.spawn_blocking(walk).await.unwrap_or_else(|source| {
                    Err(object_store::Error::Generic {
                        store: STORE,
                        source: Box::new(source),
                    })
                }),
                Err(_) => walk(),
            }
        })
        .flat_map(|r| match r {
            Ok(v) => futures::stream::iter(v.into_iter().map(Ok)).boxed(),
            Err(e) => futures::stream::once(std::future::ready(Err(e))).boxed(),
        })
        .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> StoreResult<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use futures::{FutureExt, TryStreamExt};
    use object_store::ObjectStoreExt;

    use super::*;

    fn names(dir: &FsPath, between: impl FnMut()) -> Vec<String> {
        let mut out: Vec<String> = walk_with(dir, &AtomicBool::new(false), between)
            .unwrap()
            .into_iter()
            .map(|m| m.location.filename().unwrap().to_owned())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn a_file_replaced_during_a_listing_is_replaced_in_it() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("old"), b"1").unwrap();
        let mut calls = 0;
        let listed = names(tmp.path(), || {
            calls += 1;
            if tmp.path().join("old").exists() {
                std::fs::write(tmp.path().join("new"), b"2").unwrap();
                std::fs::remove_file(tmp.path().join("old")).unwrap();
            }
        });
        assert_eq!(listed, ["new"]);
        assert_eq!(calls, 2);
    }

    #[test]
    fn a_partition_directory_replaced_during_a_listing_is_replaced_in_it() {
        let tmp = tempfile::tempdir().unwrap();
        let old = tmp.path().join("a=1");
        std::fs::create_dir(&old).unwrap();
        std::fs::write(old.join("f"), b"1").unwrap();
        let listed = names(tmp.path(), || {
            if old.exists() {
                std::fs::remove_dir_all(&old).unwrap();
                std::fs::create_dir(tmp.path().join("a=2")).unwrap();
                std::fs::write(tmp.path().join("a=2/g"), b"2").unwrap();
            }
        });
        assert_eq!(listed, ["g"]);
    }

    #[test]
    fn a_quiet_listing_reads_the_directory_once() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a"), b"1").unwrap();
        let mut calls = 0;
        assert_eq!(names(tmp.path(), || calls += 1), ["a"]);
        assert_eq!(calls, 1);
    }

    /// Lets a writer's change show in a directory's time: in the timer tick of
    /// the stat before it, a kernel before 6.13 would not move it.
    fn after_a_tick() {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    #[test]
    fn a_partition_emptied_for_a_new_one_is_listed_again() {
        // An overwrite creates `a=2/new` and deletes `a=1/old` after the names
        // of the root are read: `a=1` is there, with nothing in it.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("a=1")).unwrap();
        std::fs::write(tmp.path().join("a=1/old"), b"1").unwrap();
        let listed = names(tmp.path(), || {
            if tmp.path().join("a=1/old").exists() {
                after_a_tick();
                std::fs::create_dir(tmp.path().join("a=2")).unwrap();
                std::fs::write(tmp.path().join("a=2/new"), b"2").unwrap();
                std::fs::remove_file(tmp.path().join("a=1/old")).unwrap();
            }
        });
        assert_eq!(listed, ["new"]);
    }

    #[test]
    fn an_old_file_gone_before_its_partition_is_read_is_not_a_short_listing() {
        // `a=1` is read after `old1` went, so no question fails: the time of
        // the root, which gained `a=3`, tells.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("a=1")).unwrap();
        std::fs::create_dir(tmp.path().join("a=2")).unwrap();
        std::fs::write(tmp.path().join("a=1/old1"), b"1").unwrap();
        std::fs::write(tmp.path().join("a=1/old2"), b"1").unwrap();
        std::fs::write(tmp.path().join("a=2/old3"), b"1").unwrap();
        let listed = names(tmp.path(), || {
            if tmp.path().join("a=1/old1").exists() {
                after_a_tick();
                std::fs::create_dir(tmp.path().join("a=3")).unwrap();
                std::fs::write(tmp.path().join("a=3/new"), b"2").unwrap();
                std::fs::remove_file(tmp.path().join("a=1/old1")).unwrap();
            }
        });
        assert_eq!(listed, ["new", "old2", "old3"]);
    }

    #[test]
    fn a_link_removed_during_a_listing_is_replaced_in_it() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        std::fs::write(target.path().join("t"), b"1").unwrap();
        let old = tmp.path().join("old");
        std::os::unix::fs::symlink(target.path().join("t"), &old).unwrap();
        let listed = names(tmp.path(), || {
            if old.symlink_metadata().is_ok() {
                std::fs::write(tmp.path().join("new"), b"2").unwrap();
                std::fs::remove_file(&old).unwrap();
            }
        });
        assert_eq!(listed, ["new"]);
    }

    #[test]
    fn an_empty_directory_is_no_reason_to_list_again() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("a=1")).unwrap();
        std::fs::write(tmp.path().join("f"), b"1").unwrap();
        let mut calls = 0;
        assert_eq!(names(tmp.path(), || calls += 1), ["f"]);
        assert_eq!(calls, 2);
    }

    #[test]
    fn a_file_growing_during_a_listing_is_no_reason_to_list_again() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("f"), b"1").unwrap();
        let mut calls = 0;
        let listed = names(tmp.path(), || {
            calls += 1;
            after_a_tick();
            std::fs::write(tmp.path().join("f"), b"12").unwrap();
        });
        assert_eq!(listed, ["f"]);
        assert_eq!(calls, 1);
    }

    #[test]
    fn a_directory_that_keeps_gaining_files_is_answered() {
        // A writer publishes a file and starts the next on every pass: no
        // data file went, and a staging file is not asked about.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("part-0#1"), b"1").unwrap();
        let mut n = 0;
        let listed = names(tmp.path(), || {
            after_a_tick();
            std::fs::rename(
                tmp.path().join(format!("part-{n}#1")),
                tmp.path().join(format!("part-{n}")),
            )
            .unwrap();
            n += 1;
            std::fs::write(tmp.path().join(format!("part-{n}#1")), b"1").unwrap();
        });
        assert_eq!(n, SETTLE_PASSES);
        assert_eq!(listed.len(), SETTLE_PASSES - 1, "{listed:?}");
    }

    #[test]
    fn files_that_keep_vanishing_are_an_error_the_scheduler_retries() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("f0"), b"1").unwrap();
        let mut n = 0;
        let e = walk_with(tmp.path(), &AtomicBool::new(false), || {
            std::fs::write(tmp.path().join(format!("f{}", n + 1)), b"1").unwrap();
            std::fs::remove_file(tmp.path().join(format!("f{n}"))).unwrap();
            n += 1;
        })
        .unwrap_err();
        assert_eq!(n, SETTLE_PASSES);
        assert!(e.to_string().contains("kept vanishing"), "{e}");
        assert!(super::super::transient(&e).is_some(), "{e}");
    }

    #[test]
    fn a_dropped_listing_stops() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a"), b"1").unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        drop(CancelOnDrop(Arc::clone(&cancelled)));
        let e = walk_with(tmp.path(), &cancelled, || {}).unwrap_err();
        assert!(e.to_string().contains("dropped"), "{e}");
    }

    #[test]
    fn a_link_through_a_file_is_an_error_as_there() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("f"), b"1").unwrap();
        std::os::unix::fs::symlink(tmp.path().join("f/x"), tmp.path().join("l")).unwrap();
        walk_with(tmp.path(), &AtomicBool::new(false), || {}).unwrap_err();
    }

    #[test]
    fn an_entry_that_cannot_be_asked_about_is_as_there() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let shut = tmp.path().join("shut");
        std::fs::create_dir(&shut).unwrap();
        std::fs::write(shut.join("f"), b"1").unwrap();
        std::fs::write(tmp.path().join("a"), b"1").unwrap();
        let mode = |mode| {
            std::fs::set_permissions(&shut, std::fs::Permissions::from_mode(mode)).unwrap();
        };
        mode(0o444);
        // Permissions don't stop root, which runs some containers.
        if std::fs::metadata(shut.join("f")).is_ok() {
            mode(0o755);
            return;
        }
        // A file is left out.
        let listed = names(tmp.path(), || {});
        mode(0o755);
        assert_eq!(listed, ["a"]);
        // A directory is an error.
        std::fs::create_dir(shut.join("sub")).unwrap();
        mode(0o444);
        let walked = walk_with(tmp.path(), &AtomicBool::new(false), || {});
        mode(0o755);
        walked.unwrap_err();
    }

    #[tokio::test]
    async fn a_directory_named_like_a_staging_file_is_listed() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("events#2024");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("a"), b"1").unwrap();
        let prefix = Path::from_absolute_path(&dir).unwrap();
        let listed: Vec<ObjectMeta> = SettledLocalFileSystem::new()
            .list(Some(&prefix))
            .try_collect()
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
    }

    #[test]
    fn a_link_to_nothing_is_no_reason_to_list_again() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a"), b"1").unwrap();
        std::os::unix::fs::symlink(tmp.path().join("none"), tmp.path().join("broken")).unwrap();
        let mut calls = 0;
        assert_eq!(names(tmp.path(), || calls += 1), ["a"]);
        assert_eq!(calls, 1);
    }

    #[test]
    fn a_link_back_up_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("d")).unwrap();
        std::os::unix::fs::symlink(tmp.path(), tmp.path().join("d/up")).unwrap();
        let e = walk_with(tmp.path(), &AtomicBool::new(false), || {}).unwrap_err();
        assert!(e.to_string().contains("a link back"), "{e}");
    }

    #[test]
    fn the_root_as_a_file_or_missing_lists_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a"), b"1").unwrap();
        let mut calls = 0;
        assert!(names(&tmp.path().join("a"), || calls += 1).is_empty());
        assert!(names(&tmp.path().join("none"), || calls += 1).is_empty());
        assert_eq!(calls, 0);
    }

    /// Everything else a listing says is what `LocalFileSystem` says.
    #[tokio::test]
    async fn a_listing_answers_as_local_file_system_does() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("a=1 x/b%20c")).unwrap();
        std::fs::create_dir(root.join("linked")).unwrap();
        std::fs::write(root.join("plain.parquet"), b"12").unwrap();
        std::fs::write(root.join("_SUCCESS"), b"").unwrap();
        std::fs::write(root.join("sp ace#é.csv"), b"1").unwrap();
        std::fs::write(root.join("staging#1"), b"1").unwrap();
        std::fs::write(root.join("kept#x"), b"1").unwrap();
        std::fs::write(root.join("a=1 x/b%20c/f[1]"), b"123").unwrap();
        std::fs::write(root.join("linked/in"), b"1").unwrap();
        std::os::unix::fs::symlink(root.join("plain.parquet"), root.join("file-link")).unwrap();
        std::os::unix::fs::symlink(root.join("linked"), root.join("dir-link")).unwrap();
        std::os::unix::fs::symlink(root.join("none"), root.join("broken")).unwrap();
        let prefix = Path::from_absolute_path(root).unwrap();
        let settled = SettledLocalFileSystem::new();
        let plain = LocalFileSystem::new();
        let sorted = |mut v: Vec<ObjectMeta>| {
            v.sort_by(|a, b| a.location.cmp(&b.location));
            v
        };
        let ours = sorted(settled.list(Some(&prefix)).try_collect().await.unwrap());
        let theirs = sorted(plain.list(Some(&prefix)).try_collect().await.unwrap());
        assert_eq!(ours, theirs);
        assert_eq!(ours.len(), 8);
        // And a conditional read of a listed file takes the listing's tag.
        let plain_file = ours
            .iter()
            .find(|m| m.location.filename() == Some("plain.parquet"));
        let meta = plain_file.unwrap();
        assert_eq!(
            settled.head(&meta.location).await.unwrap().e_tag,
            meta.e_tag
        );
        // A missing prefix lists nothing, as a missing directory does there.
        let none = prefix.clone().join("none");
        let listed: Vec<ObjectMeta> = settled.list(Some(&none)).try_collect().await.unwrap();
        assert!(listed.is_empty());
        assert_eq!(settled.to_string(), plain.to_string());
    }

    /// A listing works with no Tokio runtime around it, as there.
    #[test]
    fn a_listing_runs_without_a_runtime() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a"), b"1").unwrap();
        let prefix = Path::from_absolute_path(tmp.path()).unwrap();
        let store = SettledLocalFileSystem::new();
        let listed: Vec<ObjectMeta> = store
            .list(Some(&prefix))
            .try_collect()
            .now_or_never()
            .unwrap()
            .unwrap();
        assert_eq!(listed.len(), 1);
    }
}
