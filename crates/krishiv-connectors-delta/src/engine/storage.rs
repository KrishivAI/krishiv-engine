// SPDX-License-Identifier: Apache-2.0
//! The kernel's `StorageHandler` over `krishiv-io` (design `P3-T021-delta.md` §3.2).
//!
//! One handler serves one table: it is bound to a store and to the table root URL, and refuses any URL
//! outside that root (ADR 0046 §3a; a path the log names is never trusted). Kernel calls are synchronous
//! and run on a blocking pool, never on a runtime worker, so each call bridges to the async `IoContext`
//! with `Handle::block_on`.

use std::ops::Range;

use bytes::Bytes;
use delta_kernel::{
    FileMeta, FileSlice, KernelError, KernelResult, KernelResultIteratorStatic, StorageHandler,
};
use krishiv_io::{IoContext, PutIfAbsent, StoreId};
use object_store::path::Path;
use tokio::runtime::Handle;
use url::Url;

/// A `StorageHandler` bound to one store and table root.
#[derive(Clone)]
pub struct KrishivStorageHandler {
    io: IoContext,
    store: StoreId,
    root: Url,
    handle: Handle,
}

impl KrishivStorageHandler {
    /// A handler for the table at `root` in `store`. `handle` is the runtime the I/O runs on.
    #[must_use]
    pub fn new(io: IoContext, store: StoreId, root: Url, handle: Handle) -> Self {
        let mut root = root;
        if !root.path().ends_with('/') {
            let with_slash = format!("{}/", root.path());
            root.set_path(&with_slash);
        }
        Self {
            io,
            store,
            root,
            handle,
        }
    }

    /// The object path of `url`, refused unless it lies under the table root.
    fn path_of(&self, url: &Url) -> KernelResult<Path> {
        let same_origin = url.scheme() == self.root.scheme()
            && url.host_str() == self.root.host_str()
            && url.port() == self.root.port();
        // A directory URL (`.../dir/`) is under the root when the root is not below it, so compare with
        // a trailing slash added.
        let within = url.path().starts_with(self.root.path())
            || format!("{}/", url.path()) == self.root.path();
        if !same_origin || !within {
            return Err(KernelError::generic(format!(
                "path {url} is outside the table root {}",
                self.root
            )));
        }
        Path::from_url_path(url.path())
            .map_err(|e| KernelError::generic(format!("path {url}: {e}")))
    }

    fn url_of(&self, location: &Path) -> Url {
        let mut url = self.root.clone();
        url.set_path(&format!("/{location}"));
        url
    }

    fn block<T>(&self, future: impl std::future::Future<Output = T>) -> T {
        debug_assert!(
            Handle::try_current().is_err(),
            "kernel storage calls run on the blocking pool, not on a runtime thread"
        );
        self.handle.block_on(future)
    }
}

impl std::fmt::Debug for KrishivStorageHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KrishivStorageHandler")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

fn io_error(e: impl std::error::Error + Send + Sync + 'static) -> KernelError {
    KernelError::generic_err(e)
}

fn slice_of(bytes: &Bytes, range: &Option<Range<u64>>, url: &Url) -> KernelResult<Bytes> {
    let Some(range) = range else {
        return Ok(bytes.clone());
    };
    let len = bytes.len() as u64;
    if range.start > range.end || range.start > len {
        return Err(KernelError::generic(format!(
            "range {range:?} is outside {url} ({len} bytes)"
        )));
    }
    let end = range.end.min(len);
    // Both bounds are at most `len`, which came from a usize.
    Ok(bytes.slice(
        usize::try_from(range.start).unwrap_or(0)..usize::try_from(end).unwrap_or(bytes.len()),
    ))
}

impl StorageHandler for KrishivStorageHandler {
    fn list_from(&self, path: &Url) -> KernelResult<KernelResultIteratorStatic<FileMeta>> {
        let full = self.path_of(path)?;
        let (prefix, strictly_after) = if path.path().ends_with('/') {
            (full, None)
        } else {
            let parent: Path = full
                .parts()
                .take(full.parts().count().saturating_sub(1))
                .collect();
            (parent, Some(full))
        };
        let mut metas = self
            .block(async {
                match &strictly_after {
                    Some(after) => self.io.list_from(&self.store, &prefix, after).await,
                    None => self.io.list(&self.store, &prefix).await,
                }
            })
            .map_err(io_error)?;
        // The kernel contract is UTF-8 order by full path, strictly after `path`.
        metas.sort_by(|a, b| a.location.as_ref().cmp(b.location.as_ref()));
        if let Some(after) = &strictly_after {
            metas.retain(|m| m.location.as_ref() > after.as_ref());
        }
        let out: Vec<KernelResult<FileMeta>> = metas
            .iter()
            .map(|m| {
                Ok(FileMeta {
                    location: self.url_of(&m.location),
                    last_modified: m.last_modified.timestamp_millis(),
                    size: m.size,
                })
            })
            .collect();
        Ok(Box::new(out.into_iter()))
    }

    fn read_files(&self, files: Vec<FileSlice>) -> KernelResult<KernelResultIteratorStatic<Bytes>> {
        let this = self.clone();
        Ok(Box::new(files.into_iter().map(move |(url, range)| {
            let path = this.path_of(&url)?;
            let bytes = this
                .block(this.io.get_object(&this.store, &path))
                .map_err(io_error)?
                .ok_or_else(|| KernelError::file_not_found(url.as_str()))?;
            slice_of(&bytes, &range, &url)
        })))
    }

    fn copy_atomic(&self, _src: &Url, _dest: &Url) -> KernelResult<()> {
        // Its only caller is the declarative-plan executor, which is not compiled in.
        Err(KernelError::unsupported("atomic copy"))
    }

    fn put(&self, path: &Url, data: Bytes, overwrite: bool) -> KernelResult<()> {
        let object = self.path_of(path)?;
        if overwrite {
            return self
                .block(self.io.put_object(&self.store, &object, data))
                .map_err(io_error);
        }
        match self
            .block(self.io.put_if_absent(&self.store, &object, data))
            .map_err(io_error)?
        {
            PutIfAbsent::Created => Ok(()),
            PutIfAbsent::Exists => Err(KernelError::FileAlreadyExists(path.to_string())),
            PutIfAbsent::Unsupported => Err(KernelError::unsupported(
                "Delta commit on a store without conditional put",
            )),
        }
    }

    fn head(&self, path: &Url) -> KernelResult<FileMeta> {
        let object = self.path_of(path)?;
        let meta = self
            .block(self.io.head(&self.store, &object))
            .map_err(io_error)?
            .ok_or_else(|| KernelError::file_not_found(path.as_str()))?;
        Ok(FileMeta {
            location: path.clone(),
            last_modified: meta.last_modified.timestamp_millis(),
            size: meta.size,
        })
    }

    fn delete(&self, path: &Url) -> KernelResult<()> {
        let object = self.path_of(path)?;
        self.block(self.io.delete(&self.store, &object))
            .map_err(io_error)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_io::{IoConfig, IoScheduler};
    use object_store::local::LocalFileSystem;

    use super::*;

    fn handler(dir: &std::path::Path, handle: Handle) -> KrishivStorageHandler {
        let io = IoScheduler::new(IoConfig::default());
        let store = StoreId::new("local");
        io.register_store(store.clone(), Arc::new(LocalFileSystem::new()));
        let root = Url::from_directory_path(dir).unwrap();
        KrishivStorageHandler::new(io.context(), store, root, handle)
    }

    fn url(h: &KrishivStorageHandler, rel: &str) -> Url {
        h.root.join(rel).unwrap()
    }

    // Kernel calls run off the runtime threads, as in production.
    fn on_pool<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::spawn(f).join().unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn put_if_absent_is_the_commit_primitive_and_list_from_is_sorted_and_strict() {
        let dir = tempfile::tempdir().unwrap();
        let h = handler(dir.path(), Handle::current());
        on_pool(move || {
            let a = url(&h, "_delta_log/00000000000000000000.json");
            h.put(&a, Bytes::from_static(b"one"), false).unwrap();
            let again = h.put(&a, Bytes::from_static(b"two"), false);
            assert!(matches!(again, Err(KernelError::FileAlreadyExists(_))));
            h.put(
                &url(&h, "_delta_log/00000000000000000002.json"),
                Bytes::from_static(b"3"),
                false,
            )
            .unwrap();
            h.put(
                &url(&h, "_delta_log/00000000000000000001.json"),
                Bytes::from_static(b"2"),
                true,
            )
            .unwrap();

            let listed: Vec<_> = h
                .list_from(&a)
                .unwrap()
                .map(|m| {
                    m.unwrap()
                        .location
                        .path()
                        .rsplit('/')
                        .next()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            assert_eq!(
                listed,
                ["00000000000000000001.json", "00000000000000000002.json"]
            );

            let read: Vec<_> = h
                .read_files(vec![(a.clone(), None), (a.clone(), Some(1..3))])
                .unwrap()
                .map(Result::unwrap)
                .collect();
            assert_eq!(
                read,
                [Bytes::from_static(b"one"), Bytes::from_static(b"ne")]
            );
            assert_eq!(h.head(&a).unwrap().size, 3);
            h.delete(&a).unwrap();
            h.delete(&a).unwrap();
            assert!(matches!(h.head(&a), Err(KernelError::FileNotFound(_))));
        });
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_url_outside_the_table_root_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let h = handler(dir.path(), Handle::current());
        on_pool(move || {
            let outside =
                Url::from_file_path(dir.path().parent().unwrap().join("other.json")).unwrap();
            assert!(h.head(&outside).is_err());
            let other_scheme = Url::parse("s3://bucket/x.json").unwrap();
            assert!(h.delete(&other_scheme).is_err());
            assert!(
                h.read_files(vec![(outside, None)])
                    .unwrap()
                    .next()
                    .unwrap()
                    .is_err()
            );
        });
    }
}
