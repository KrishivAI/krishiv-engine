// SPDX-License-Identifier: Apache-2.0
//! A local file system on which a read of a directory is "not found" on every
//! platform.
//!
//! `LocalFileSystem` opens the file before it looks at it. On Unix a
//! directory opens and its metadata says "directory", which it answers as
//! `NotFound`; on Windows a directory does not open as a file at all (access
//! denied), which it answers as a store failure. A `HEAD` of a directory is how
//! a write asks whether its target is a file ([`crate::DirectoryWrite`]), so
//! there a save into an existing directory failed instead of following its
//! save mode. Here a failed read of a path that is a directory is `NotFound`,
//! as on Unix.

use std::fmt;
use std::ops::Range;

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as StoreResult,
};

/// `LocalFileSystem`, but a read of a directory is `NotFound` on every platform.
#[derive(Debug)]
pub struct LocalDirsAsMissing {
    inner: LocalFileSystem,
}

impl LocalDirsAsMissing {
    /// Wraps `inner`.
    #[must_use]
    pub fn new(inner: LocalFileSystem) -> Self {
        Self { inner }
    }

    /// `e`, or `NotFound` when `location` is a directory.
    async fn directory_is_missing(
        &self,
        location: &Path,
        e: object_store::Error,
    ) -> object_store::Error {
        if matches!(e, object_store::Error::NotFound { .. }) {
            return e;
        }
        let is_dir = match self.inner.path_to_filesystem(location) {
            // The metadata call blocks, so it runs off the async worker; with no Tokio runtime
            // (a caller driving the store with `futures::executor::block_on`) it runs inline,
            // where `spawn_blocking` would panic (review of #1002).
            Ok(path) => match tokio::runtime::Handle::try_current() {
                Ok(runtime) => runtime
                    .spawn_blocking(move || path.is_dir())
                    .await
                    .unwrap_or(false),
                Err(_) => path.is_dir(),
            },
            Err(_) => false,
        };
        if is_dir {
            object_store::Error::NotFound {
                path: location.to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "is directory",
                )),
            }
        } else {
            e
        }
    }
}

impl fmt::Display for LocalDirsAsMissing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

#[async_trait]
impl ObjectStore for LocalDirsAsMissing {
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
        match self.inner.get_opts(location, options).await {
            Ok(r) => Ok(r),
            Err(e) => Err(self.directory_is_missing(location, e).await),
        }
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> StoreResult<Vec<bytes::Bytes>> {
        match self.inner.get_ranges(location, ranges).await {
            Ok(r) => Ok(r),
            Err(e) => Err(self.directory_is_missing(location, e).await),
        }
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.inner.list(prefix)
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
    use object_store::ObjectStoreExt;

    use super::*;

    #[tokio::test]
    async fn a_directory_reads_as_missing_and_a_file_as_itself() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("d")).unwrap();
        std::fs::write(dir.path().join("f"), b"hi").unwrap();
        let store = LocalDirsAsMissing::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let e = store.head(&Path::from("d")).await.unwrap_err();
        assert!(matches!(e, object_store::Error::NotFound { .. }), "{e}");
        let e = store.get(&Path::from("d")).await.unwrap_err();
        assert!(matches!(e, object_store::Error::NotFound { .. }), "{e}");
        assert!(matches!(
            store.head(&Path::from("nothing")).await.unwrap_err(),
            object_store::Error::NotFound { .. }
        ));
        assert_eq!(store.head(&Path::from("f")).await.unwrap().size, 2);
        let listed = store.list_with_delimiter(None).await.unwrap();
        assert_eq!(listed.common_prefixes, vec![Path::from("d")]);
        assert_eq!(listed.objects.len(), 1);
    }

    /// Windows' answer for a directory (a store failure, access denied) is `NotFound`; on a file
    /// or a missing path the error stays as it is. Run on every platform with the failure made up.
    #[tokio::test]
    async fn a_failure_on_a_directory_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("d")).unwrap();
        std::fs::write(dir.path().join("f"), b"hi").unwrap();
        let store = LocalDirsAsMissing::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let denied = || object_store::Error::Generic {
            store: "LocalFileSystem",
            source: Box::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
        };
        let e = store.directory_is_missing(&Path::from("d"), denied()).await;
        assert!(matches!(e, object_store::Error::NotFound { .. }), "{e}");
        let e = store.directory_is_missing(&Path::from("f"), denied()).await;
        assert!(matches!(e, object_store::Error::Generic { .. }), "{e}");
        let e = store
            .directory_is_missing(&Path::from("nothing"), denied())
            .await;
        assert!(matches!(e, object_store::Error::Generic { .. }), "{e}");
    }

    /// With no Tokio runtime (a caller driving the store with `futures::executor::block_on`) the
    /// directory check runs inline instead of panicking in `spawn_blocking` (review of #1002):
    /// polled once outside any runtime, it is ready.
    #[test]
    fn a_directory_is_found_missing_outside_a_runtime() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("d")).unwrap();
        let store = LocalDirsAsMissing::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let denied = object_store::Error::Generic {
            store: "LocalFileSystem",
            source: Box::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
        };
        let at = Path::from("d");
        let check = std::pin::pin!(store.directory_is_missing(&at, denied));
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let std::task::Poll::Ready(e) = check.poll(&mut cx) else {
            panic!("the check waited outside a runtime");
        };
        assert!(matches!(e, object_store::Error::NotFound { .. }), "{e}");
    }

    #[tokio::test]
    async fn writes_lists_copies_and_renames_reach_the_file_system() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalDirsAsMissing::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        assert!(store.to_string().contains("LocalFileSystem"));
        store
            .put(&Path::from("a/x"), b"one".to_vec().into())
            .await
            .unwrap();
        let mut up = store.put_multipart(&Path::from("a/y")).await.unwrap();
        up.put_part(b"two".to_vec().into()).await.unwrap();
        up.complete().await.unwrap();
        let got = store
            .get_ranges(&Path::from("a/y"), std::slice::from_ref(&(0..2)))
            .await
            .unwrap();
        assert_eq!(got[0].as_ref(), b"tw");
        let e = store
            .get_ranges(&Path::from("a"), std::slice::from_ref(&(0..1)))
            .await
            .unwrap_err();
        assert!(matches!(e, object_store::Error::NotFound { .. }), "{e}");
        store
            .copy(&Path::from("a/x"), &Path::from("a/z"))
            .await
            .unwrap();
        store
            .rename(&Path::from("a/z"), &Path::from("b"))
            .await
            .unwrap();
        let mut names: Vec<String> = futures::TryStreamExt::try_collect::<Vec<_>>(store.list(None))
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.location.to_string())
            .collect();
        names.sort();
        assert_eq!(names, ["a/x", "a/y", "b"]);
        let after: Vec<_> = futures::TryStreamExt::try_collect::<Vec<_>>(
            store.list_with_offset(None, &Path::from("a/x")),
        )
        .await
        .unwrap();
        assert_eq!(after.len(), 2);
        store.delete(&Path::from("b")).await.unwrap();
        assert!(store.head(&Path::from("b")).await.is_err());
    }
}
