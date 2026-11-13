// SPDX-License-Identifier: Apache-2.0
//! `AddArtifacts` and `ArtifactStatus` (design P3-T011a §3.9, threat model
//! T-U6; Spark's `SparkConnectAddArtifactsHandler`,
//! `SparkConnectArtifactStatusesHandler` and `ArtifactManager.addArtifact`,
//! v4.3.0).
//!
//! **The stream** (`AddArtifactsH:54-97, 176-195`): each request carries a
//! `batch` of whole artifacts, the `begin_chunk` of one artifact split in
//! chunks, or its next `chunk`. Each chunk carries a CRC-32 of its bytes; an
//! artifact whose chunk sums do not all match is discarded and reported with
//! `is_crc_successful = false` (`:114-141, 237-256, 290-294`). A chunked
//! artifact must get exactly `num_chunks` chunks holding `total_bytes`
//! (`:296-316`); more chunks, fewer, a `chunk` with no artifact open, or a new
//! artifact while one is open, end the stream with
//! `KRISHIV_INVALID_ARTIFACT_TRANSFER` (Spark throws classless errors). A
//! chunked artifact is complete with its last chunk, also when
//! `num_chunks` is 1 (Spark only closes it when a later `chunk` arrives).
//! The artifacts are installed when the client ends the stream
//! (`:145-171`): an existing name with other bytes is
//! `ARTIFACT_ALREADY_EXISTS` (42713) after the others are installed, the
//! same bytes are a no-op (`ArtifactManager.scala:234-246`). Any other
//! failure, or the client going away, discards every artifact of the stream.
//!
//! **Names** are checked by `krishiv_connect::artifact` before any file
//! exists, then by the privilege gate (`config::check_add_artifact`):
//! `cache/` needs only the session, every other prefix is client code
//! (`Usage` on `System("udf")`, ADR 0046 §3b).
//!
//! **Storage** (`dir`): a private directory per session under the server's
//! artifact root, removed when the session is released, swept or closed at
//! shutdown, and the root removed with the service. Bytes are staged in a
//! fresh `O_EXCL | O_NOFOLLOW` file and installed with `linkat`, which never
//! replaces a name; no link is ever followed. Archives are not extracted
//! (T014 does it, with its own checks). `cache/<hash>` artifacts are kept in
//! memory instead, reserved from the engine's pool (consumer
//! `connect_artifacts`), for T012's `CachedLocalRelation`
//! ([`SessionArtifacts::cached`]).
//!
//! **Quotas**: every byte is counted against the session's
//! `krishiv.connect.artifacts.maxBytes` before it is written, plus
//! [`ARTIFACT_OVERHEAD`] per artifact (so many empty artifacts cannot fill
//! a disk with inodes), and against `krishiv.connect.artifacts.maxArtifactBytes`
//! for its artifact (a chunked artifact's declared `total_bytes` first).
//! Past either, `KRISHIV_ARTIFACT_LIMIT` (`RESOURCE_EXHAUSTED`). A discarded
//! artifact returns its bytes; an installed one holds them until the session
//! ends.
//!
//! **ArtifactStatus** (`ArtifactStatusesH:43-70`): `exists` is true only for
//! a `cache/<hash>` the session holds; every other name is false.

mod dir;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use bytes::Bytes;
use krishiv_common::config::{
    CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES, CONNECT_ARTIFACTS_MAX_BYTES,
    CONNECT_ARTIFACTS_MAX_TOTAL_BYTES, Config, ConfigValue,
};
use krishiv_common::error::{
    ArtifactAlreadyExists, ConnectInvalidPlanInvalidOneOfFieldNotSet, ErrorClass, InternalError,
    InvalidArtifactPath, InvalidHandleSessionClosed, KrishivArtifactLimit, KrishivError,
    KrishivInvalidArtifactTransfer, KrishivInvalidConfigValue,
};
use krishiv_common::principal::Principal;
use krishiv_connect::artifact::{ArtifactKind, ArtifactName, parse_artifact_name};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::add_artifacts_request::{ArtifactChunk, Payload};
use krishiv_connect::proto::v4_3::add_artifacts_response::ArtifactSummary;
use krishiv_connect::wire::Decoded;
use krishiv_memory::{ConsumerHandle, MemoryPool, Reservation};
use tonic::{Status, Streaming};

use self::dir::{ArtifactRoot, Install, SessionDir};
use crate::config::{ConfigRole, check_add_artifact};
use crate::manager::SessionEntry;
use crate::operation::random_uuid;
use crate::operation::replay::uuid_text;

/// What each artifact counts against the session's quota besides its bytes:
/// a file system block (the inode and the directory entry of a file, the
/// map entry of a cached blob).
pub const ARTIFACT_OVERHEAD: u64 = 4096;

/// The artifact quotas and where the root goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactLimits {
    /// Bytes one session may hold (`krishiv.connect.artifacts.maxBytes`).
    pub max_session_bytes: u64,
    /// Bytes of one artifact (`krishiv.connect.artifacts.maxArtifactBytes`).
    pub max_artifact_bytes: u64,
    /// Bytes all sessions together may hold
    /// (`krishiv.connect.artifacts.maxTotalBytes`).
    pub max_total_bytes: u64,
    /// The directory the server's artifact root is created in; `None`: the
    /// system's temporary directory.
    pub parent: Option<PathBuf>,
}

impl Default for ArtifactLimits {
    fn default() -> Self {
        Self {
            max_session_bytes: 1 << 30,
            max_artifact_bytes: 256 << 20,
            max_total_bytes: 16 << 30,
            parent: None,
        }
    }
}

impl ArtifactLimits {
    /// The limits `config` gives; defaults otherwise.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let d = Self::default();
        let size = |e| config.get(e).and_then(ConfigValue::as_size);
        Self {
            max_session_bytes: size(&CONNECT_ARTIFACTS_MAX_BYTES).unwrap_or(d.max_session_bytes),
            max_artifact_bytes: size(&CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES)
                .unwrap_or(d.max_artifact_bytes),
            max_total_bytes: size(&CONNECT_ARTIFACTS_MAX_TOTAL_BYTES).unwrap_or(d.max_total_bytes),
            parent: None,
        }
    }

    /// Checks the limits fit together (at startup).
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` when `maxTotalBytes` is below
    /// `maxArtifactBytes` (no artifact of the allowed size could ever fit).
    pub fn validate(&self) -> Result<(), KrishivError> {
        if self.max_total_bytes < self.max_artifact_bytes {
            return Err(KrishivInvalidConfigValue {
                value: format!("'{}'", self.max_total_bytes),
                key: format!("\"{}\"", CONNECT_ARTIFACTS_MAX_TOTAL_BYTES.key),
                reason: format!(
                    "it must be at least {} ({} bytes)",
                    CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES.key, self.max_artifact_bytes
                ),
            }
            .into());
        }
        Ok(())
    }
}

/// The server-wide count of artifact bytes (`maxTotalBytes`).
#[derive(Debug)]
struct ServerQuota {
    used: Mutex<u64>,
    cap: u64,
}

impl ServerQuota {
    fn lock(&self) -> std::sync::MutexGuard<'_, u64> {
        self.used.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn reserve(&self, bytes: u64, name: &str) -> Result<(), KrishivError> {
        let mut used = self.lock();
        match used.checked_add(bytes) {
            Some(total) if total <= self.cap => {
                *used = total;
                Ok(())
            }
            _ => Err(limit(name, CONNECT_ARTIFACTS_MAX_TOTAL_BYTES.key, self.cap)),
        }
    }

    fn release(&self, bytes: u64) {
        let mut used = self.lock();
        *used = used.saturating_sub(bytes);
    }
}

/// The root directory and whether the store was shut down.
#[derive(Debug, Default)]
struct RootState {
    root: Option<Arc<ArtifactRoot>>,
    shut: bool,
}

/// The server's artifacts: the limits and the root directory, created by
/// the first artifact written and removed when the store and every session
/// directory are gone.
#[derive(Debug)]
pub struct ArtifactStore {
    limits: ArtifactLimits,
    total: Arc<ServerQuota>,
    root: Mutex<RootState>,
}

impl ArtifactStore {
    /// A store with `limits`; nothing is created yet.
    #[must_use]
    pub fn new(limits: ArtifactLimits) -> Self {
        Self {
            total: Arc::new(ServerQuota {
                used: Mutex::new(0),
                cap: limits.max_total_bytes,
            }),
            limits,
            root: Mutex::new(RootState::default()),
        }
    }

    /// The limits.
    #[must_use]
    pub fn limits(&self) -> &ArtifactLimits {
        &self.limits
    }

    /// The root directory, once created.
    #[must_use]
    pub fn root_path(&self) -> Option<PathBuf> {
        self.lock().root.as_ref().map(|r| r.path().to_owned())
    }

    /// Bytes all sessions hold now (against `maxTotalBytes`).
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        *self.total.lock()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RootState> {
        self.root.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Removes the root and everything under it, and creates none again:
    /// the server calls it after every session is closed, so the files are
    /// gone before `serve` returns (a removal queued on the blocking pool
    /// may never run once the runtime is dropped).
    pub async fn shutdown(&self) {
        let path = {
            let mut state = self.lock();
            state.shut = true;
            state.root.as_ref().map(|r| r.path().to_owned())
        };
        let Some(path) = path else {
            return;
        };
        let removed = tokio::task::spawn_blocking(move || {
            #[expect(
                clippy::disallowed_methods,
                reason = "runs on the blocking pool; std's remove_dir_all follows no link"
            )]
            let removed = std::fs::remove_dir_all(&path);
            removed
        })
        .await;
        match removed {
            Ok(Ok(())) => {}
            Ok(Err(e)) if e.kind() == io::ErrorKind::NotFound => {}
            Ok(Err(e)) => {
                tracing::warn!(target: "krishiv::connect", error = %e, "could not remove the artifact root");
            }
            Err(e) => {
                tracing::warn!(target: "krishiv::connect", error = %e, "could not remove the artifact root");
            }
        }
    }

    /// The root, created on first use (blocking file system calls).
    fn root(&self) -> Result<Arc<ArtifactRoot>, KrishivError> {
        let mut state = self.lock();
        if state.shut {
            return Err(SessionArtifacts::closed_error());
        }
        if let Some(r) = state.root.as_ref() {
            return Ok(Arc::clone(r));
        }
        let parent = self
            .limits
            .parent
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let mut last = None;
        // A name taken by someone else is skipped, never reused.
        for _ in 0..4 {
            let name = format!("krishiv-artifacts-{}", uuid_text(random_uuid()?));
            match ArtifactRoot::create(&parent, &name) {
                Ok(r) => {
                    let r = Arc::new(r);
                    state.root = Some(Arc::clone(&r));
                    return Ok(r);
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(io_failure("create the artifact root", &e)),
            }
        }
        let e = last.unwrap_or_else(|| io::Error::other("no free name"));
        Err(io_failure("create the artifact root", &e))
    }
}

/// A cached `cache/<hash>` artifact and the pool bytes it holds.
struct Cached {
    data: Bytes,
    /// What it counts against the quotas (its bytes and overhead).
    charge: u64,
    _reservation: Reservation,
}

/// An installed file artifact (a `CloneSession` copies it).
#[derive(Debug, Clone)]
struct Installed {
    name: ArtifactName,
    bytes: u64,
    /// What it counts against the quotas (its bytes and overhead).
    charge: u64,
}

/// What a `CloneSession` copies of a session's artifacts: the installed
/// ones when it was taken (never a transfer still in progress). Holds the
/// source's directory open until the copy is done.
pub(crate) struct ArtifactSnapshot {
    dir: Option<Arc<SessionDir>>,
    files: Vec<Installed>,
    cache: Vec<(String, Bytes, u64)>,
}

impl ArtifactSnapshot {
    /// Whether there is nothing to copy.
    pub(crate) fn is_empty(&self) -> bool {
        self.files.is_empty() && self.cache.is_empty()
    }
}

#[derive(Default)]
struct State {
    dir: Option<Arc<SessionDir>>,
    /// Bytes counted against `maxBytes`: staged and installed.
    used: u64,
    cache: HashMap<String, Cached>,
    /// The installed files, in the order installed.
    files: Vec<Installed>,
    consumer: Option<Arc<ConsumerHandle>>,
    /// The server-wide count `used` is also held in.
    total: Option<Arc<ServerQuota>>,
    closed: bool,
}

/// One session's artifacts: its directory, its cached blobs and the bytes
/// counted against its quota.
#[derive(Default)]
pub struct SessionArtifacts {
    state: Mutex<State>,
}

impl std::fmt::Debug for SessionArtifacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.lock();
        f.debug_struct("SessionArtifacts")
            .field("used", &s.used)
            .field("cached", &s.cache.len())
            .field("dir", &s.dir.is_some())
            .field("closed", &s.closed)
            .finish()
    }
}

impl SessionArtifacts {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The bytes of the `cache/<hash>` artifact `hash`, if the session holds
    /// it (T012's `CachedLocalRelation` reads local relations from here).
    #[must_use]
    pub fn cached(&self, hash: &str) -> Option<Bytes> {
        self.lock().cache.get(hash).map(|c| c.data.clone())
    }

    /// Bytes counted against the session's quota now.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.lock().used
    }

    /// The session's artifact directory, once its first file artifact was
    /// staged.
    #[must_use]
    pub fn dir_path(&self) -> Option<PathBuf> {
        self.lock().dir.as_ref().map(|d| d.path().to_owned())
    }

    /// Ends the session's artifacts: the cache is dropped (its pool bytes
    /// returned) and the directory removed once no transfer still writes
    /// to it. Nothing is added afterwards.
    pub(crate) fn close(&self) {
        let (dir, cache) = {
            let mut s = self.lock();
            s.closed = true;
            s.files.clear();
            if let Some(total) = s.total.take() {
                total.release(s.used);
            }
            s.used = 0;
            (s.dir.take(), std::mem::take(&mut s.cache))
        };
        // Outside the lock: removing a tree takes time.
        drop(cache);
        drop(dir);
    }

    fn closed_error() -> KrishivError {
        InvalidHandleSessionClosed {
            handle: "(released while adding artifacts)".to_owned(),
        }
        .into()
    }

    /// Counts `bytes` of artifact `name` against the session's quota and
    /// the server's (`store`).
    fn reserve(&self, bytes: u64, store: &ArtifactStore, name: &str) -> Result<(), KrishivError> {
        let limits = store.limits();
        let mut s = self.lock();
        if s.closed {
            return Err(Self::closed_error());
        }
        let total = match s.used.checked_add(bytes) {
            Some(total) if total <= limits.max_session_bytes => total,
            _ => {
                return Err(limit(
                    name,
                    CONNECT_ARTIFACTS_MAX_BYTES.key,
                    limits.max_session_bytes,
                ));
            }
        };
        // Lock order: the session's state, then the server's count.
        store.total.reserve(bytes, name)?;
        s.total.get_or_insert_with(|| Arc::clone(&store.total));
        s.used = total;
        Ok(())
    }

    fn release(&self, bytes: u64) {
        let mut s = self.lock();
        let bytes = bytes.min(s.used);
        s.used -= bytes;
        if let Some(total) = &s.total {
            total.release(bytes);
        }
    }

    /// The session's directory, created on first use (blocking; the file
    /// system calls run outside the session's lock).
    fn dir(&self, store: &ArtifactStore) -> Result<Arc<SessionDir>, KrishivError> {
        {
            let s = self.lock();
            if s.closed {
                return Err(Self::closed_error());
            }
            if let Some(d) = &s.dir {
                return Ok(Arc::clone(d));
            }
        }
        let root = store.root()?;
        let name = uuid_text(random_uuid()?);
        let made = Arc::new(
            SessionDir::create(&root, &name)
                .map_err(|e| io_failure("create a session's artifact directory", &e))?,
        );
        let mut s = self.lock();
        if s.closed {
            // Dropped (and removed) outside the lock.
            drop(s);
            drop(made);
            return Err(Self::closed_error());
        }
        // A racing transfer made one first: ours goes, theirs is used.
        Ok(Arc::clone(s.dir.get_or_insert(made)))
    }

    /// The session's memory consumer for cached artifacts.
    fn consumer(&self, pool: &Arc<dyn MemoryPool>) -> Arc<ConsumerHandle> {
        let mut s = self.lock();
        Arc::clone(s.consumer.get_or_insert_with(|| {
            ConsumerHandle::register(Arc::clone(pool), "connect_artifacts", None)
        }))
    }

    /// Keeps `data` as `cache/<hash>`; `false` when the hash is already
    /// held (the bytes are the same by construction: a hash names them).
    fn put_cached(
        &self,
        hash: &str,
        data: Bytes,
        charge: u64,
        reservation: Reservation,
    ) -> Result<bool, KrishivError> {
        let mut s = self.lock();
        if s.closed {
            return Err(Self::closed_error());
        }
        if s.cache.contains_key(hash) {
            return Ok(false);
        }
        s.cache.insert(
            hash.to_owned(),
            Cached {
                data,
                charge,
                _reservation: reservation,
            },
        );
        Ok(true)
    }

    fn is_closed(&self) -> bool {
        self.lock().closed
    }

    /// Records an installed file (unless the session closed meanwhile).
    fn record_installed(&self, installed: Installed) {
        let mut s = self.lock();
        if !s.closed {
            s.files.push(installed);
        }
    }

    /// The installed artifacts now (`CloneSession`); `None` once closed.
    pub(crate) fn snapshot(&self) -> Option<ArtifactSnapshot> {
        let s = self.lock();
        if s.closed {
            return None;
        }
        Some(ArtifactSnapshot {
            dir: s.dir.clone(),
            files: s.files.clone(),
            cache: s
                .cache
                .iter()
                .map(|(hash, c)| (hash.clone(), c.data.clone(), c.charge))
                .collect(),
        })
    }

    /// Copies the artifacts of `snapshot` into this (new) session's
    /// artifacts, each counted against this session's quota and the
    /// server's before its bytes are written, cached blobs reserved again
    /// from the pool (blocking; on the blocking pool). On a failure what it
    /// copied stays counted here: the caller closes this session, which
    /// returns it all and removes the directory.
    ///
    /// # Errors
    /// `KRISHIV_ARTIFACT_LIMIT` past a quota, the pool's refusal for a
    /// cached blob, `INVALID_HANDLE.SESSION_CLOSED` once this session is
    /// closed, `INTERNAL_ERROR` for a file system failure.
    pub(crate) fn copy_from(
        self: &Arc<Self>,
        snapshot: &ArtifactSnapshot,
        store: &ArtifactStore,
        pool: &Arc<dyn MemoryPool>,
    ) -> Result<(), KrishivError> {
        for (hash, data, charge) in &snapshot.cache {
            let name = format!("cache/{hash}");
            self.reserve(*charge, store, &name)?;
            let mut reservation = self.consumer(pool).reservation();
            let put = reservation
                .try_grow(data.len())
                .map_err(KrishivError::from)
                .and_then(|()| self.put_cached(hash, data.clone(), *charge, reservation));
            match put {
                Ok(true) => {}
                Ok(false) => self.release(*charge),
                Err(e) => {
                    self.release(*charge);
                    return Err(e);
                }
            }
        }
        if snapshot.files.is_empty() {
            return Ok(());
        }
        let Some(source) = &snapshot.dir else {
            return Err(InternalError {
                message: "CloneSession: installed artifacts without a directory".to_owned(),
            }
            .into());
        };
        for file in &snapshot.files {
            self.copy_file(store, source, file)?;
        }
        Ok(())
    }

    /// Copies one installed file of `source` here: counted first, staged,
    /// installed as an upload is.
    fn copy_file(
        self: &Arc<Self>,
        store: &ArtifactStore,
        source: &SessionDir,
        file: &Installed,
    ) -> Result<(), KrishivError> {
        let raw = file.name.path().to_owned();
        self.reserve(file.charge, store, &raw)?;
        let made = self.dir(store).and_then(|dir| {
            let staged = uuid_text(random_uuid()?);
            let out = dir
                .create_staged(&staged)
                .map_err(|e| io_failure("stage a cloned artifact", &e))?;
            Ok((dir, staged, out))
        });
        let (dir, staged_name, out) = match made {
            Ok(made) => made,
            Err(e) => {
                self.release(file.charge);
                return Err(e);
            }
        };
        // From here dropping `staged` removes the staged file and returns
        // the charge.
        let mut staged = Staged {
            raw,
            name: file.name.clone(),
            sink: Sink::File {
                dir,
                staged: staged_name,
                file: None,
            },
            crc_ok: true,
            first_chunk: false,
            chunks: None,
            written: 0,
            reserved: file.charge,
            owner: Arc::clone(self),
        };
        let segments: Vec<&str> = file.name.segments().collect();
        let input = source
            .open_installed(&segments)
            .map_err(|e| io_failure("read an artifact to clone", &e))?;
        let mut out = out;
        let copied = io::copy(&mut io::Read::take(input, file.bytes), &mut out)
            .map_err(|e| io_failure("copy an artifact", &e))?;
        if copied != file.bytes {
            return Err(InternalError {
                message: format!(
                    "CloneSession: artifact {} has {copied} bytes, {} were installed",
                    staged.raw, file.bytes
                ),
            }
            .into());
        }
        staged.written = copied;
        if let Sink::File { file: slot, .. } = &mut staged.sink {
            *slot = Some(out);
        }
        staged.complete()?;
        if staged.install()? {
            Ok(())
        } else {
            Err(InternalError {
                message: format!("CloneSession: artifact {} was already there", staged.raw),
            }
            .into())
        }
    }
}

fn limit(name: &str, key: &str, bytes: u64) -> KrishivError {
    KrishivArtifactLimit {
        name: name.to_owned(),
        limit_name: key.to_owned(),
        limit: bytes.to_string(),
    }
    .into()
}

fn transfer_error(name: &str, reason: String) -> KrishivError {
    KrishivInvalidArtifactTransfer {
        name: name.to_owned(),
        reason,
    }
    .into()
}

/// A file system failure: logged with its detail, sent without it (paths
/// are the server's).
fn io_failure(what: &str, e: &io::Error) -> KrishivError {
    tracing::warn!(target: "krishiv::connect", error = %e, "AddArtifacts could not {what}");
    InternalError {
        message: format!("AddArtifacts could not {what}."),
    }
    .into()
}

/// CRC-32 (ISO-HDLC, reflected polynomial `0xEDB88320`), as
/// `java.util.zip.CRC32` and Python's `zlib.crc32` compute it.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            #[expect(clippy::cast_possible_truncation, reason = "i < 256")]
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 == 1 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };
    let mut crc = !0u32;
    for &b in bytes {
        crc = TABLE[((crc ^ u32::from(b)) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Makes room in `buf` for `more` bytes, reserving every byte of capacity
/// it gains from the pool first: the pool holds what the blob allocates,
/// never less.
fn grow_exact(
    buf: &mut Vec<u8>,
    reservation: Option<&mut Reservation>,
    more: usize,
) -> Result<(), KrishivError> {
    let need = buf.len().saturating_add(more);
    if need <= buf.capacity() {
        return Ok(());
    }
    let gain = need - buf.capacity();
    if let Some(r) = reservation {
        r.try_grow(gain)?;
    }
    buf.reserve_exact(need - buf.len());
    Ok(())
}

/// Where a staged artifact's bytes go.
enum Sink {
    /// A staged file; `file` is closed once the artifact is complete.
    File {
        dir: Arc<SessionDir>,
        staged: String,
        file: Option<File>,
    },
    /// A `cache/` blob and its pool bytes.
    Memory {
        buf: Vec<u8>,
        reservation: Option<Reservation>,
    },
}

/// An artifact of a chunked transfer: its declared size and what is left.
#[derive(Debug, Clone, Copy)]
struct Chunks {
    expected: i64,
    remaining: i64,
    total_bytes: i64,
}

/// One artifact being rebuilt (Spark's `StagedArtifact`). Dropping it
/// removes its staged file and returns its quota, unless it was installed.
struct Staged {
    /// The name as the client sent it (the summary repeats it).
    raw: String,
    name: ArtifactName,
    sink: Sink,
    crc_ok: bool,
    first_chunk: bool,
    chunks: Option<Chunks>,
    written: u64,
    /// Bytes counted against the session's quota, returned on drop.
    reserved: u64,
    owner: Arc<SessionArtifacts>,
}

impl Drop for Staged {
    fn drop(&mut self) {
        if let Sink::File { dir, staged, file } = &mut self.sink {
            drop(file.take());
            dir.remove_staged(staged);
        }
        self.owner.release(self.reserved);
    }
}

impl Staged {
    fn write(&mut self, chunk: &ArtifactChunk, store: &ArtifactStore) -> Result<(), KrishivError> {
        let limits = store.limits();
        let n = chunk.data.len() as u64;
        if self
            .written
            .checked_add(n)
            .is_none_or(|w| w > limits.max_artifact_bytes)
        {
            return Err(limit(
                &self.raw,
                CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES.key,
                limits.max_artifact_bytes,
            ));
        }
        self.owner.reserve(n, store, &self.raw)?;
        self.reserved += n;
        match &mut self.sink {
            Sink::File { file: Some(f), .. } => f
                .write_all(&chunk.data)
                .map_err(|e| io_failure("write an artifact", &e))?,
            Sink::File { file: None, .. } => {
                return Err(transfer_error(
                    &self.raw,
                    "it is already complete".to_owned(),
                ));
            }
            Sink::Memory { buf, reservation } => {
                grow_exact(buf, reservation.as_mut(), chunk.data.len())?;
                buf.extend_from_slice(&chunk.data);
            }
        }
        self.written += n;
        let ok = i64::from(crc32(&chunk.data)) == chunk.crc;
        // A chunked artifact is good only if every chunk is
        // (`StagedChunkedArtifact.updateCrc`).
        self.crc_ok = if self.chunks.is_some() {
            ok && (self.crc_ok || self.first_chunk)
        } else {
            ok
        };
        self.first_chunk = false;
        Ok(())
    }

    /// Closes the staged file; a chunked artifact must have all its chunks.
    fn complete(&mut self) -> Result<(), KrishivError> {
        if let Some(c) = self.chunks
            && (c.remaining != 0 || u64::try_from(c.total_bytes).ok() != Some(self.written))
        {
            return Err(transfer_error(
                &self.raw,
                format!(
                    "expected {} chunks of {} bytes in all, received {} chunks of {} bytes",
                    c.expected,
                    c.total_bytes,
                    c.expected - c.remaining,
                    self.written
                ),
            ));
        }
        if let Sink::File { file, .. } = &mut self.sink
            && let Some(f) = file.take()
        {
            f.sync_data()
                .map_err(|e| io_failure("write an artifact", &e))?;
        }
        Ok(())
    }

    /// Installs the artifact (CRC good): `Ok(true)` when added, `Ok(false)`
    /// when the same bytes were there.
    fn install(&mut self) -> Result<bool, KrishivError> {
        if self.owner.is_closed() {
            return Err(SessionArtifacts::closed_error());
        }
        match &mut self.sink {
            Sink::Memory { buf, reservation } => {
                let hash = self.name.cache_hash().unwrap_or_default().to_owned();
                let data = Bytes::from(std::mem::take(buf));
                let Some(reservation) = reservation.take() else {
                    return Ok(false);
                };
                let added = self
                    .owner
                    .put_cached(&hash, data, self.reserved, reservation)?;
                if added {
                    self.reserved = 0;
                }
                Ok(added)
            }
            Sink::File { dir, staged, .. } => {
                let segments: Vec<&str> = self.name.segments().collect();
                let outcome = dir
                    .install(staged, &segments)
                    .map_err(|e| io_failure("install an artifact", &e))?;
                match outcome {
                    Install::Added => {
                        self.owner.record_installed(Installed {
                            name: self.name.clone(),
                            bytes: self.written,
                            charge: self.reserved,
                        });
                        self.reserved = 0;
                        Ok(true)
                    }
                    Install::Identical => Ok(false),
                    Install::Exists => Err(ArtifactAlreadyExists {
                        normalized_remote_relative_path: self.name.path().to_owned(),
                    }
                    .into()),
                    Install::Blocked => Err(InvalidArtifactPath {
                        name: self.raw.clone(),
                    }
                    .into()),
                }
            }
        }
    }
}

/// The receiving side of one `AddArtifacts` stream (blocking file system
/// calls: run off the async workers).
pub(crate) struct Transfer {
    store: Arc<ArtifactStore>,
    owner: Arc<SessionArtifacts>,
    pool: Arc<dyn MemoryPool>,
    principal: Principal,
    role: ConfigRole,
    /// Complete artifacts, installed when the stream ends.
    staged: Vec<Staged>,
    /// The chunked artifact still receiving chunks.
    open: Option<Staged>,
}

impl Transfer {
    pub(crate) fn new(
        store: Arc<ArtifactStore>,
        entry: &SessionEntry,
        principal: Principal,
        role: ConfigRole,
    ) -> Self {
        Self {
            store,
            owner: Arc::clone(entry.artifacts()),
            pool: entry.session().runtime().memory(),
            principal,
            role,
            staged: Vec::new(),
            open: None,
        }
    }

    /// A new artifact named `raw`: validated and allowed before anything is
    /// created.
    fn stage(&self, raw: &str, chunks: Option<Chunks>) -> Result<Staged, KrishivError> {
        let name = parse_artifact_name(raw)?;
        check_add_artifact(self.role, &self.principal, &name)?;
        let limits = self.store.limits();
        if let Some(c) = chunks {
            if c.expected < 1 || c.total_bytes < 0 {
                return Err(transfer_error(
                    raw,
                    format!(
                        "{} chunks of {} bytes were announced",
                        c.expected, c.total_bytes
                    ),
                ));
            }
            if u64::try_from(c.total_bytes).is_ok_and(|t| t > limits.max_artifact_bytes) {
                return Err(limit(
                    raw,
                    CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES.key,
                    limits.max_artifact_bytes,
                ));
            }
        }
        // One block per segment: the file and each directory the install
        // may create (review HIGH-1), reserved before any of them exists.
        let segments = u64::try_from(name.segments().count()).unwrap_or(u64::MAX);
        let overhead = ARTIFACT_OVERHEAD.saturating_mul(segments);
        self.owner.reserve(overhead, &self.store, raw)?;
        let sink = if name.kind() == ArtifactKind::Cache {
            // The blob grows chunk by chunk in `write`, each chunk counted
            // against the session's and the server's quota before the pool
            // reservation grows: an announced size reserves nothing, so a
            // stream that announces 256 MiB and stalls pins no memory.
            Sink::Memory {
                buf: Vec::new(),
                reservation: Some(self.owner.consumer(&self.pool).reservation()),
            }
        } else {
            let made = self.owner.dir(&self.store).and_then(|dir| {
                let staged = uuid_text(random_uuid()?);
                let file = dir
                    .create_staged(&staged)
                    .map_err(|e| io_failure("stage an artifact", &e))?;
                Ok(Sink::File {
                    dir,
                    staged,
                    file: Some(file),
                })
            });
            match made {
                Ok(sink) => sink,
                Err(e) => {
                    self.owner.release(overhead);
                    return Err(e);
                }
            }
        };
        Ok(Staged {
            raw: raw.to_owned(),
            name,
            sink,
            crc_ok: false,
            first_chunk: true,
            chunks,
            written: 0,
            reserved: overhead,
            owner: Arc::clone(&self.owner),
        })
    }

    /// One request of the stream.
    pub(crate) fn on_message(
        &mut self,
        request: &pb::AddArtifactsRequest,
    ) -> Result<(), KrishivError> {
        let store = Arc::clone(&self.store);
        match &request.payload {
            Some(Payload::Batch(batch)) => {
                self.refuse_while_open()?;
                for artifact in &batch.artifacts {
                    let mut staged = self.stage(&artifact.name, None)?;
                    let empty = ArtifactChunk::default();
                    staged.write(artifact.data.as_ref().unwrap_or(&empty), &store)?;
                    staged.complete()?;
                    self.staged.push(staged);
                }
            }
            Some(Payload::BeginChunk(begin)) => {
                self.refuse_while_open()?;
                let chunks = Chunks {
                    expected: begin.num_chunks,
                    remaining: begin.num_chunks,
                    total_bytes: begin.total_bytes,
                };
                let staged = self.stage(&begin.name, Some(chunks))?;
                self.open = Some(staged);
                let empty = ArtifactChunk::default();
                self.next_chunk(begin.initial_chunk.as_ref().unwrap_or(&empty), &store)?;
            }
            Some(Payload::Chunk(chunk)) => self.next_chunk(chunk, &store)?,
            None => {
                return Err(ConnectInvalidPlanInvalidOneOfFieldNotSet {
                    full_name: "spark.connect.AddArtifactsRequest".to_owned(),
                    name: "PAYLOAD_NOT_SET".to_owned(),
                }
                .into());
            }
        }
        Ok(())
    }

    fn refuse_while_open(&self) -> Result<(), KrishivError> {
        match &self.open {
            Some(open) => Err(transfer_error(
                &open.raw,
                "a new artifact began before its last chunk".to_owned(),
            )),
            None => Ok(()),
        }
    }

    fn next_chunk(
        &mut self,
        chunk: &ArtifactChunk,
        store: &ArtifactStore,
    ) -> Result<(), KrishivError> {
        let Some(open) = self.open.as_mut() else {
            return Err(transfer_error(
                "",
                "a chunk arrived with no chunked artifact begun".to_owned(),
            ));
        };
        let Some(chunks) = open.chunks.as_mut() else {
            return Err(transfer_error(&open.raw, "it is not chunked".to_owned()));
        };
        if chunks.remaining <= 0 {
            return Err(transfer_error(
                &open.raw,
                format!("more than the {} chunks announced", chunks.expected),
            ));
        }
        chunks.remaining -= 1;
        let done = chunks.remaining == 0;
        open.write(chunk, store)?;
        if done && let Some(mut finished) = self.open.take() {
            finished.complete()?;
            self.staged.push(finished);
        }
        Ok(())
    }

    /// The end of the stream: every complete artifact with good sums is
    /// installed; the summaries in the order the artifacts came.
    pub(crate) fn finish(mut self) -> Result<Vec<ArtifactSummary>, KrishivError> {
        if let Some(open) = &self.open {
            let c = open.chunks.unwrap_or(Chunks {
                expected: 0,
                remaining: 0,
                total_bytes: 0,
            });
            return Err(transfer_error(
                &open.raw,
                format!(
                    "the stream ended after {} of {} chunks",
                    c.expected - c.remaining,
                    c.expected
                ),
            ));
        }
        let mut summaries = Vec::with_capacity(self.staged.len());
        let mut exists: Option<KrishivError> = None;
        for staged in &mut self.staged {
            if staged.crc_ok {
                match staged.install() {
                    Ok(_) => {}
                    // The others are still installed (`AddArtifactsH:114-141`).
                    Err(e) if e.error_class() == ErrorClass::ArtifactAlreadyExists => {
                        exists.get_or_insert(e);
                    }
                    Err(e) => return Err(e),
                }
            }
            summaries.push(ArtifactSummary {
                name: staged.raw.clone(),
                is_crc_successful: staged.crc_ok,
            });
        }
        match exists {
            Some(e) => Err(e),
            None => Ok(summaries),
        }
    }
}

/// Runs `f` on the blocking pool.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, KrishivError> + Send + 'static,
) -> Result<T, KrishivError> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| -> KrishivError {
            InternalError {
                message: format!("AddArtifacts: {e}"),
            }
            .into()
        })?
}

/// `AddArtifacts` on the session `entry`, from its `first` request: the
/// summaries, or the status of the first failure (with an `errorId`).
pub(crate) async fn add_artifacts(
    entry: &Arc<SessionEntry>,
    store: &Arc<ArtifactStore>,
    principal: Principal,
    role: ConfigRole,
    first: Decoded<pb::AddArtifactsRequest>,
    stream: &mut Streaming<Decoded<pb::AddArtifactsRequest>>,
) -> Result<pb::AddArtifactsResponse, Status> {
    let errors = Arc::clone(entry.errors());
    let mut transfer = Transfer::new(Arc::clone(store), entry, principal, role);
    let session_id = first.session_id.clone();
    let mut next = Some(first);
    while let Some(request) = next.take() {
        // A failure drops the transfer there, which removes what it staged.
        transfer = blocking(move || {
            transfer.on_message(&request)?;
            Ok(transfer)
        })
        .await
        .map_err(|e| errors.status(&e))?;
        // A client that goes away ends the stream with an error: the
        // transfer is dropped here and its staged files with it.
        next = match stream.message().await {
            Ok(next) => next,
            Err(status) => {
                // Its staged files are removed on the blocking pool (a
                // handler dropped mid-await by tonic drops it in place).
                let _ = blocking(move || {
                    drop(transfer);
                    Ok(())
                })
                .await;
                return Err(status);
            }
        };
    }
    let artifacts = blocking(move || transfer.finish())
        .await
        .map_err(|e| errors.status(&e))?;
    Ok(pb::AddArtifactsResponse {
        session_id,
        server_side_session_id: entry.server_side_id().to_owned(),
        artifacts,
    })
}

/// `ArtifactStatus` on the session `entry`.
pub(crate) fn artifact_status(
    entry: &SessionEntry,
    request: &pb::ArtifactStatusesRequest,
) -> pb::ArtifactStatusesResponse {
    let statuses = request
        .names
        .iter()
        .map(|name| {
            let exists = name
                .strip_prefix("cache/")
                .is_some_and(|hash| entry.artifacts().cached(hash).is_some());
            (
                name.clone(),
                pb::artifact_statuses_response::ArtifactStatus { exists },
            )
        })
        .collect();
    pb::ArtifactStatusesResponse {
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        statuses,
    }
}

#[cfg(test)]
mod tests;
