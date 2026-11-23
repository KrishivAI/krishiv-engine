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
    fn directory_is_missing(&self, location: &Path, e: object_store::Error) -> object_store::Error {
        if matches!(e, object_store::Error::NotFound { .. }) {
            return e;
        }
        match self.inner.path_to_filesystem(location) {
            Ok(path) if path.is_dir() => object_store::Error::NotFound {
                path: location.to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "is directory",
                )),
            },
            _ => e,
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
        self.inner
            .get_opts(location, options)
            .await
            .map_err(|e| self.directory_is_missing(location, e))
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> StoreResult<Vec<bytes::Bytes>> {
        self.inner
            .get_ranges(location, ranges)
            .await
            .map_err(|e| self.directory_is_missing(location, e))
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
}
