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

use std::fmt;
use std::fs::Metadata;
use std::io::ErrorKind;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::{Path as FsPath, PathBuf};
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

/// The files below `root`, following links, with no staging file. `between`
/// runs after the read of each directory's names and before the questions
/// about them (a test replaces files there).
///
/// # Errors
/// An entry still vanished in the last pass: what that pass found may be
/// short of every file, even empty, so it is not an answer. The error is one
/// the scheduler retries after a wait.
fn walk_with(root: &FsPath, mut between: impl FnMut()) -> StoreResult<Vec<ObjectMeta>> {
    let mut out = Vec::new();
    for _ in 0..SETTLE_PASSES {
        out.clear();
        if !walk_once(root, &mut between, &mut out)? {
            return Ok(out);
        }
    }
    Err(io_err(std::io::Error::new(
        ErrorKind::Interrupted,
        format!(
            "files under {} kept vanishing during {SETTLE_PASSES} listings",
            root.display()
        ),
    )))
}

/// One pass of [`walk_with`]: whether an entry vanished during it.
fn walk_once(
    root: &FsPath,
    between: &mut impl FnMut(),
    out: &mut Vec<ObjectMeta>,
) -> StoreResult<bool> {
    let gone =
        |e: &std::io::Error| matches!(e.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory);
    let root_id = match std::fs::metadata(root) {
        Ok(m) => id_of(&m),
        // A prefix with nothing below it lists nothing.
        Err(e) if gone(&e) => return Ok(false),
        Err(e) => return Err(io_err(e)),
    };
    let mut vanished = false;
    // A directory still to read, with the directories above it and itself.
    let mut pending: Vec<(PathBuf, Vec<(u64, u64)>)> = vec![(root.to_owned(), vec![root_id])];
    while let Some((dir, above)) = pending.pop() {
        let names = match std::fs::read_dir(&dir) {
            Ok(names) => names
                .map(|e| e.and_then(|e| Ok((e.path(), e.file_type()?))))
                .collect::<Result<Vec<_>, _>>(),
            Err(e) => Err(e),
        };
        let names = match names {
            Ok(names) => names,
            // The root as a file holds nothing; a directory below it went.
            Err(e) if gone(&e) => {
                vanished |= dir != root;
                continue;
            }
            Err(e) => return Err(io_err(e)),
        };
        between();
        for (path, file_type) in names {
            let m = match std::fs::metadata(&path) {
                Ok(m) => m,
                // A link to nothing was never a file; anything else was.
                Err(e) if gone(&e) => {
                    vanished |= !file_type.is_symlink();
                    continue;
                }
                Err(e) => return Err(io_err(e)),
            };
            if m.is_dir() {
                if above.contains(&id_of(&m)) {
                    return Err(io_err(std::io::Error::other(format!(
                        "a link back to a directory above it at {}",
                        path.display()
                    ))));
                }
                let mut below = above.clone();
                below.push(id_of(&m));
                pending.push((path, below));
            } else if m.is_file()
                && !path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(is_staging)
            {
                let location = Path::from_absolute_path(&path)
                    .map_err(|source| object_store::Error::InvalidPath { source })?;
                out.push(meta_of(location, &m));
            }
        }
    }
    Ok(vanished)
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
        // A prefix named like a staging file is refused here.
        let Ok(root) = self
            .inner
            .path_to_filesystem(prefix.unwrap_or(&Path::default()))
        else {
            return self.inner.list(prefix);
        };
        let walk = move || walk_with(&root, || {});
        futures::stream::once(async move {
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
        let mut out: Vec<String> = walk_with(dir, between)
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

    #[test]
    fn a_directory_that_keeps_changing_is_an_error_the_scheduler_retries() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("f0"), b"1").unwrap();
        let mut n = 0;
        let e = walk_with(tmp.path(), || {
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
        let e = walk_with(tmp.path(), || {}).unwrap_err();
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
