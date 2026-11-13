// SPDX-License-Identifier: Apache-2.0
//! The kernel's `StorageHandler` over `IoContext` (design note P3-T021 section 3.2).
//!
//! | Kernel call | Krishiv side |
//! |---|---|
//! | `list_from(url)` | `IoContext::list_from`: recursive, UTF-8 sorted, strictly after `url` (a directory-like `url` lists everything under it) |
//! | `read_files` | `get_object` (whole file) or a ranged read; `_last_checkpoint` and `*.crc`, which the kernel parses itself, get the size and depth limits here |
//! | `put(overwrite)` | `put_object` when overwriting, else `put_if_absent` (an existing object is the kernel's `FileAlreadyExists`) |
//! | `head` | `head`; a missing object is `FileNotFound` |
//! | `delete` | `delete`, idempotent |
//! | `copy_atomic` | the kernel's `Unsupported`; nothing in the kernel calls it |
//!
//! Every call checks the query's cancellation first, confines every URL to the table root and
//! runs its I/O on the log pool thread it was called on (`Shared::block_on`).

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::{
    FileMeta, FileSlice, KernelError, KernelResult, KernelResultIteratorStatic, StorageHandler,
};
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_io::{FileRef, PutIfAbsent};
use object_store::ObjectMeta;
use object_store::path::Path;
use url::Url;

use super::Shared;
use super::json::{check_depth, check_size};
use crate::error::to_kernel;

/// The storage handler of a [`super::DeltaEngine`].
#[derive(Debug)]
pub struct KrishivStorageHandler {
    shared: Arc<Shared>,
}

impl KrishivStorageHandler {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }

    fn confine(&self, url: &Url) -> KernelResult<Path> {
        self.shared.root.confine(url).map_err(to_kernel)
    }

    fn file_meta(&self, meta: &ObjectMeta) -> KernelResult<FileMeta> {
        Ok(FileMeta {
            location: self.shared.root.url_of(&meta.location).map_err(to_kernel)?,
            last_modified: meta.last_modified.timestamp_millis(),
            size: meta.size,
        })
    }

    /// Writes `data` at `path`: an overwrite, or a create that fails with `FileAlreadyExists`
    /// when the object exists (the commit primitive).
    pub(crate) fn put_bytes(&self, url: &Url, data: Bytes, overwrite: bool) -> KernelResult<()> {
        self.shared.check_cancelled()?;
        let path = self.confine(url)?;
        let (io, store) = (&self.shared.io, &self.shared.store);
        if overwrite {
            return self.shared.block_on(io.put_object(store, &path, data));
        }
        match self.shared.block_on(io.put_if_absent(store, &path, data))? {
            PutIfAbsent::Created => Ok(()),
            PutIfAbsent::Exists => Err(KernelError::FileAlreadyExists(url.to_string())),
            PutIfAbsent::Unsupported => Err(to_kernel(
                KrishivUnsupportedFeature {
                    feature: "Delta commit on a store without conditional put".to_owned(),
                }
                .into(),
            )),
        }
    }
}

/// Reads `range` of `url` (all of it when `None`), with the limits of a file the kernel parses.
fn read_slice(shared: &Shared, url: &Url, range: Option<Range<u64>>) -> KernelResult<Bytes> {
    shared.check_cancelled()?;
    let path = shared.root.confine(url).map_err(to_kernel)?;
    let (io, store) = (&shared.io, &shared.store);
    let limited = parsed_by_kernel(&path);
    let bytes = if range.is_none() && !limited {
        shared
            .block_on(io.get_object(store, &path))?
            .ok_or_else(|| KernelError::file_not_found(url.as_str()))?
    } else {
        // A ranged read, or a limited file whose size is checked before it is read.
        let meta = shared
            .block_on(io.head(store, &path))?
            .ok_or_else(|| KernelError::file_not_found(url.as_str()))?;
        if limited {
            check_size(&shared.limits, url.as_str(), meta.size).map_err(to_kernel)?;
        }
        match range {
            None => shared
                .block_on(io.get_object(store, &path))?
                .ok_or_else(|| KernelError::file_not_found(url.as_str()))?,
            Some(range) if range.is_empty() => Bytes::new(),
            Some(range) => {
                let file = FileRef::from_meta(store.clone(), &meta);
                let mut parts =
                    shared.block_on(io.get_ranges(&file, std::slice::from_ref(&range)))?;
                parts
                    .pop()
                    .ok_or_else(|| KernelError::generic(format!("no bytes read from {url}")))?
            }
        }
    };
    if limited {
        check_size(&shared.limits, url.as_str(), bytes.len() as u64).map_err(to_kernel)?;
        check_depth(&shared.limits, url.as_str(), &bytes).map_err(to_kernel)?;
    }
    Ok(bytes)
}

/// Whether the kernel parses `path` itself with `serde_json`, so the limits apply here.
fn parsed_by_kernel(path: &Path) -> bool {
    path.filename()
        .is_some_and(|name| name == "_last_checkpoint" || name.ends_with(".crc"))
}

impl StorageHandler for KrishivStorageHandler {
    fn list_from(&self, url: &Url) -> KernelResult<KernelResultIteratorStatic<FileMeta>> {
        self.shared.check_cancelled()?;
        let (prefix, offset) = if url.path().ends_with('/') {
            let dir = self.confine(url)?;
            (dir.clone(), dir)
        } else {
            let offset = self.confine(url)?;
            let mut parts: Vec<_> = offset.parts().collect();
            parts.pop();
            (Path::from_iter(parts), offset)
        };
        let listed = self.shared.block_on(self.shared.io.list_from(
            &self.shared.store,
            &prefix,
            &offset,
        ))?;
        let metas = listed
            .iter()
            .map(|m| self.file_meta(m))
            .collect::<KernelResult<Vec<_>>>()?;
        Ok(Box::new(metas.into_iter().map(Ok)))
    }

    fn read_files(&self, files: Vec<FileSlice>) -> KernelResult<KernelResultIteratorStatic<Bytes>> {
        self.shared.check_cancelled()?;
        // Lazily, one file per pull; a failure (a missing file included) is that item's error,
        // as the kernel expects (it reads a missing `_last_checkpoint` as "no hint").
        let shared = Arc::clone(&self.shared);
        Ok(Box::new(
            files
                .into_iter()
                .map(move |(url, range)| read_slice(&shared, &url, range)),
        ))
    }

    fn copy_atomic(&self, src: &Url, dest: &Url) -> KernelResult<()> {
        Err(KernelError::unsupported(format!(
            "copying {src} to {dest} (copy_atomic is not offered by Krishiv's Delta engine)"
        )))
    }

    fn put(&self, path: &Url, data: Bytes, overwrite: bool) -> KernelResult<()> {
        self.put_bytes(path, data, overwrite)
    }

    fn head(&self, url: &Url) -> KernelResult<FileMeta> {
        self.shared.check_cancelled()?;
        let path = self.confine(url)?;
        let meta = self
            .shared
            .block_on(self.shared.io.head(&self.shared.store, &path))?
            .ok_or_else(|| KernelError::file_not_found(url.as_str()))?;
        self.file_meta(&meta)
    }

    fn delete(&self, url: &Url) -> KernelResult<()> {
        self.shared.check_cancelled()?;
        let path = self.confine(url)?;
        self.shared
            .block_on(self.shared.io.delete(&self.shared.store, &path))
    }
}
