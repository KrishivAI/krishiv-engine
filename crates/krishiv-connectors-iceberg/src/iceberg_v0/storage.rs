// SPDX-License-Identifier: Apache-2.0
//! iceberg-rust's file access (`Storage`/`StorageFactory`) over Krishiv's I/O
//! (`design/P3-lakehouse.md` §5.1): every byte iceberg-rust reads or writes, metadata
//! included, goes through an [`IoContext`], so it inherits the stores, retries,
//! counters and policy of every other read. `iceberg-storage-opendal` is never used.
//!
//! Both iceberg traits are `#[typetag::serde]`: a `FileIO` can be serialised, so the
//! factory and the storage carry only a registration name. The name resolves to the
//! process's [`IoContext`] and path placement through [`StorageRegistration`], which
//! names itself uniquely and unregisters on drop; a name that is not registered
//! (a deserialised `FileIO` that outlived its registration, or one from another
//! process) fails the call cleanly.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, PoisonError, RwLock};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use futures::stream::BoxStream;
use iceberg::io::{
    FileIO, FileIOBuilder, FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage,
    StorageConfig, StorageFactory,
};
use iceberg::{Error, ErrorKind, Runtime};
use krishiv_common::error::{FailedReadFileFileNotExist, InternalError, KrishivError};
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_io::{FileRef, IoContext};
use serde::{Deserialize, Serialize};

/// What a registration name resolves to.
#[derive(Clone)]
struct Registered {
    io: IoContext,
    paths: Arc<dyn PathLocator>,
}

impl fmt::Debug for Registered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registered")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

static REGISTRY: LazyLock<RwLock<HashMap<String, Registered>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static NEXT_NAME: AtomicU64 = AtomicU64::new(0);
/// A random value per process: names stay unique where pids repeat (every container's
/// first process is pid 1), so a `FileIO` serialised in another process never
/// resolves here.
static PROCESS_TOKEN: LazyLock<u64> = LazyLock::new(|| {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(std::process::id());
    h.finish()
});

/// Krishiv's I/O made available to iceberg-rust under a process-unique name, for as
/// long as this value lives. Every request iceberg-rust makes for a table loaded
/// through it ([`crate::IcebergTable::load`]) runs through `io`, with paths placed by
/// `paths` (the session's path placement, so the same stores and rules as a read by
/// path). Load tables on the I/O side, never on a compute worker. A table already
/// loaded keeps its storage (and so the `IoContext`) after this value is dropped; new
/// loads and deserialised `FileIO`s under the name fail.
#[derive(Debug)]
pub struct StorageRegistration {
    name: String,
}

impl StorageRegistration {
    /// Registers `io` and `paths` under a new name.
    #[must_use]
    pub fn new(io: IoContext, paths: Arc<dyn PathLocator>) -> Self {
        let name = format!(
            "krishiv-{:016x}-{}",
            *PROCESS_TOKEN,
            NEXT_NAME.fetch_add(1, Ordering::Relaxed)
        );
        REGISTRY
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name.clone(), Registered { io, paths });
        Self { name }
    }

    /// The registration name (what a serialised `FileIO` carries).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A `FileIO` over this registration.
    pub(crate) fn file_io(&self) -> FileIO {
        FileIOBuilder::new(Arc::new(KrishivStorageFactory {
            name: self.name.clone(),
        }))
        .build()
    }

    /// iceberg-rust's `Runtime` for this registration: the scheduler's I/O runtime for
    /// both of iceberg-rust's handles (its manifest decoding is metadata-sized), so
    /// nothing it spawns lands on a compute worker.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the scheduler has no I/O runtime
    /// (`krishiv_io::IoScheduler::with_runtime`) or the name is gone.
    pub(crate) fn runtime(&self) -> Result<Runtime, KrishivError> {
        let registered = resolve(&self.name).map_err(from_iceberg)?;
        let Some(handle) = registered.io.scheduler().runtime() else {
            return Err(InternalError {
                message: "iceberg-rust needs the I/O scheduler's runtime; this scheduler has none"
                    .to_owned(),
            }
            .into());
        };
        let _entered = handle.enter();
        Runtime::try_current().map_err(from_iceberg)
    }
}

impl Drop for StorageRegistration {
    fn drop(&mut self) {
        REGISTRY
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.name);
    }
}

fn resolve(name: &str) -> iceberg::Result<Registered> {
    REGISTRY
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(name)
        .cloned()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::PreconditionFailed,
                format!("no Krishiv storage is registered as {name:?} in this process"),
            )
        })
}

/// A Krishiv error as iceberg-rust's, kept as the source so [`from_iceberg`] can
/// recover its class.
fn to_iceberg(e: KrishivError) -> Error {
    Error::new(ErrorKind::Unexpected, e.to_string()).with_source(e)
}

/// An iceberg-rust error as Krishiv's: the Krishiv error it wraps when it came from
/// Krishiv's I/O (its class and SQLSTATE kept), else an internal error naming
/// iceberg-rust's message.
#[expect(
    clippy::needless_pass_by_value,
    reason = "a `map_err` function: takes the error it replaces"
)]
pub(crate) fn from_iceberg(e: Error) -> KrishivError {
    let mut source: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(&e);
    while let Some(s) = source {
        if let Some(k) = s.downcast_ref::<KrishivError>() {
            return k.clone();
        }
        source = s.source();
    }
    InternalError {
        message: format!("iceberg-rust: {e}"),
    }
    .into()
}

/// The factory iceberg-rust builds [`KrishivStorage`] from: a registration name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KrishivStorageFactory {
    pub(crate) name: String,
}

#[typetag::serde(name = "krishiv")]
impl StorageFactory for KrishivStorageFactory {
    fn build(&self, _config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        // Fail at build, not at the first request, when the name is gone. Properties
        // (`s3.*`, credentials) are ignored: the stores come from Krishiv's
        // configuration, never from a catalog (design/P3-lakehouse.md §5.6).
        let registered = resolve(&self.name)?;
        let resolved = OnceLock::new();
        let _ = resolved.set(registered);
        Ok(Arc::new(KrishivStorage {
            name: self.name.clone(),
            resolved,
        }))
    }
}

/// iceberg-rust's `Storage` over a registered [`IoContext`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KrishivStorage {
    name: String,
    #[serde(skip)]
    resolved: OnceLock<Registered>,
}

impl KrishivStorage {
    fn registered(&self) -> iceberg::Result<&Registered> {
        if let Some(r) = self.resolved.get() {
            return Ok(r);
        }
        let r = resolve(&self.name)?;
        Ok(self.resolved.get_or_init(|| r))
    }

    fn locate(&self, location: &str) -> iceberg::Result<(&Registered, LocatedPath)> {
        let r = self.registered()?;
        let at = r.paths.locate(location).map_err(to_iceberg)?;
        Ok((r, at))
    }

    /// The file at `location` as listed now (size and version).
    async fn file(&self, location: &str) -> iceberg::Result<(IoContext, FileRef)> {
        let (r, at) = self.locate(location)?;
        match r.io.head(&at.store, &at.path).await.map_err(to_iceberg)? {
            Some(m) => Ok((r.io.clone(), FileRef::from_meta(at.store, &m))),
            None => Err(not_found(location, &at.display)),
        }
    }
}

fn not_found(location: &str, display: &str) -> Error {
    // The kind and wording iceberg-rust's own storages use for a missing file, with
    // Spark's class underneath for `from_iceberg`.
    let k: KrishivError = FailedReadFileFileNotExist {
        path: display.to_owned(),
    }
    .into();
    Error::new(
        ErrorKind::DataInvalid,
        format!("File not found: {location}"),
    )
    .with_source(k)
}

#[async_trait]
#[typetag::serde(name = "krishiv")]
impl Storage for KrishivStorage {
    async fn exists(&self, path: &str) -> iceberg::Result<bool> {
        let (r, at) = self.locate(path)?;
        let meta = r.io.head(&at.store, &at.path).await.map_err(to_iceberg)?;
        Ok(meta.is_some())
    }

    async fn metadata(&self, path: &str) -> iceberg::Result<FileMetadata> {
        let (_, file) = self.file(path).await?;
        Ok(FileMetadata { size: file.size })
    }

    async fn read(&self, path: &str) -> iceberg::Result<Bytes> {
        let (r, at) = self.locate(path)?;
        r.io.get_object(&at.store, &at.path)
            .await
            .map_err(to_iceberg)?
            .ok_or_else(|| not_found(path, &at.display))
    }

    async fn reader(&self, path: &str) -> iceberg::Result<Box<dyn FileRead>> {
        let (io, file) = self.file(path).await?;
        Ok(Box::new(KrishivFileRead { io, file }))
    }

    async fn write(&self, path: &str, bs: Bytes) -> iceberg::Result<()> {
        let (r, at) = self.locate(path)?;
        r.io.put_object(&at.store, &at.path, bs)
            .await
            .map_err(to_iceberg)
    }

    async fn writer(&self, path: &str) -> iceberg::Result<Box<dyn FileWrite>> {
        let (r, at) = self.locate(path)?;
        Ok(Box::new(KrishivFileWrite {
            io: r.io.clone(),
            at,
            buf: BytesMut::new(),
            closed: false,
        }))
    }

    async fn delete(&self, path: &str) -> iceberg::Result<()> {
        let (r, at) = self.locate(path)?;
        r.io.delete(&at.store, &at.path).await.map_err(to_iceberg)
    }

    async fn delete_prefix(&self, path: &str) -> iceberg::Result<()> {
        let (r, at) = self.locate(path)?;
        // A directory prefix: `a/b` removes `a/b/...`, never `a/bc`.
        for meta in r.io.list(&at.store, &at.path).await.map_err(to_iceberg)? {
            r.io.delete(&at.store, &meta.location)
                .await
                .map_err(to_iceberg)?;
        }
        Ok(())
    }

    async fn delete_stream(&self, mut paths: BoxStream<'static, String>) -> iceberg::Result<()> {
        while let Some(path) = paths.next().await {
            self.delete(&path).await?;
        }
        Ok(())
    }

    fn new_input(&self, path: &str) -> iceberg::Result<InputFile> {
        self.registered()?;
        Ok(InputFile::new(Arc::new(self.clone()), path.to_owned()))
    }

    fn new_output(&self, path: &str) -> iceberg::Result<OutputFile> {
        self.registered()?;
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_owned()))
    }
}

/// Range reads of one file, pinned to the version listed when it was opened: a file
/// replaced underneath fails the read (`KRISHIV_FILE_CHANGED_DURING_READ`). The bytes
/// are not reserved from a memory pool; the catalog parse limits of PR-16
/// (`krishiv.catalog.maxMetadataBytes`) bound them.
struct KrishivFileRead {
    io: IoContext,
    file: FileRef,
}

#[async_trait]
impl FileRead for KrishivFileRead {
    async fn read(&self, range: Range<u64>) -> iceberg::Result<Bytes> {
        let mut parts = self
            .io
            .get_ranges(&self.file, std::slice::from_ref(&range))
            .await
            .map_err(to_iceberg)?;
        parts.pop().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("no bytes returned for {}", self.file.display_path()),
            )
        })
    }
}

/// A file written whole on `close` (iceberg-rust's metadata files and manifests are
/// small; data files are never written through iceberg-rust).
struct KrishivFileWrite {
    io: IoContext,
    at: LocatedPath,
    buf: BytesMut,
    closed: bool,
}

#[async_trait]
impl FileWrite for KrishivFileWrite {
    async fn write(&mut self, bs: Bytes) -> iceberg::Result<()> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("write after close: {}", self.at.display),
            ));
        }
        self.buf.extend_from_slice(&bs);
        Ok(())
    }

    async fn close(&mut self) -> iceberg::Result<FileMetadata> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("closed twice: {}", self.at.display),
            ));
        }
        self.closed = true;
        let bytes = std::mem::take(&mut self.buf).freeze();
        let size = bytes.len() as u64;
        self.io
            .put_object(&self.at.store, &self.at.path, bytes)
            .await
            .map_err(to_iceberg)?;
        Ok(FileMetadata { size })
    }
}
