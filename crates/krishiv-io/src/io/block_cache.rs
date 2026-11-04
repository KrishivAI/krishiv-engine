// SPDX-License-Identifier: Apache-2.0
//! The local block cache for remote object-store reads (P2-T083).
//!
//! [`CachingStore`] wraps an [`ObjectStore`] and answers bounded range reads from aligned blocks
//! (default 1 MiB) kept in a small memory tier and on local disk. One mechanism covers data pages,
//! footers and page indexes, which are all ordinary ranges. A block is keyed by the store (as
//! displayed, so without credentials), the credential scope ([`CredentialScope`]), the path, the object's version (the `if_match` ETag or the
//! `if_unmodified_since` time the scheduler sends, else none) and the block index: an overwritten
//! object has a new version and never reaches its old blocks.
//!
//! Misses are fetched in contiguous runs (one request per run, so a cold scan issues about the
//! requests an uncached one does), and readers of a block already being fetched wait for that fetch
//! (single flight). Every cache failure (a disk that is full or unreadable, a corrupt block) is
//! counted and falls back to the store; it never fails a read. The disk part is emptied at start:
//! nothing written by an earlier process is trusted. Writes and listings are not cached.
//!
//! Disk files are `<key hash>.blk`: a magic, the key's text and the bytes. The hash only names the
//! file; a read compares the stored key text, so a collision is a miss. A file is written under a
//! `.tmp` name and renamed, so a reader never sees half a block.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::Write as _;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::future::{BoxFuture, FutureExt, Shared};
use futures::stream::BoxStream;
use krishiv_memory::{MemoryPool, Reservation};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as StoreResult,
};

use super::fault::ScheduledRequest;

/// Settings of the block cache (`krishiv.io.cache.*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockCacheConfig {
    /// Directory of the disk part; the process empties and uses its own subdirectory.
    pub dir: PathBuf,
    /// Size cap of the disk part.
    pub disk_bytes: u64,
    /// Size cap of the memory tier (`0`: disk only).
    pub memory_bytes: u64,
    /// Block size.
    pub block_bytes: u64,
}

/// What the cache did, process-wide.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Blocks served from memory.
    pub memory_hits: u64,
    /// Blocks served from disk.
    pub disk_hits: u64,
    /// Blocks fetched from the store.
    pub misses: u64,
    /// Bytes fetched from the store for blocks.
    pub fetched_bytes: u64,
    /// Cache failures that fell back to the store (disk read/write, corrupt block).
    pub errors: u64,
    /// Blocks removed to stay under a cap.
    pub evictions: u64,
}

#[derive(Debug, Default)]
struct Counters {
    memory_hits: AtomicU64,
    disk_hits: AtomicU64,
    misses: AtomicU64,
    fetched_bytes: AtomicU64,
    errors: AtomicU64,
    evictions: AtomicU64,
}

fn bump(c: &AtomicU64, n: u64) {
    c.fetch_add(n, Ordering::Relaxed);
}

/// One tier's index: size and recency of each key, least recently used first.
#[derive(Default)]
struct Lru<V> {
    entries: HashMap<u128, (u64, V)>,
    order: BTreeMap<u64, u128>,
    used: u64,
}

impl<V> Lru<V> {
    fn touch(&mut self, key: u128, tick: u64) -> Option<&V> {
        let entry = self.entries.get_mut(&key)?;
        self.order.remove(&entry.0);
        entry.0 = tick;
        self.order.insert(tick, key);
        Some(&entry.1)
    }

    fn insert(&mut self, key: u128, tick: u64, size: u64, value: V) -> Vec<(u128, u64, V)> {
        let mut gone = Vec::new();
        if let Some((old_tick, old)) = self.entries.remove(&key) {
            self.order.remove(&old_tick);
            gone.push((key, 0, old));
        }
        self.entries.insert(key, (tick, value));
        self.order.insert(tick, key);
        self.used += size;
        gone
    }

    fn pop_oldest(&mut self) -> Option<(u128, V)> {
        let (_, key) = self.order.pop_first()?;
        let (_, value) = self.entries.remove(&key)?;
        Some((key, value))
    }
}

#[derive(Default)]
struct State {
    tick: u64,
    memory: Lru<Bytes>,
    disk: Lru<u64>,
    /// The memory tier's bytes, reserved from a pool when one is attached (I11).
    reservation: Option<Reservation>,
}

type RunResult = Result<Arc<Vec<Bytes>>, Arc<object_store::Error>>;
type Run = Shared<BoxFuture<'static, RunResult>>;

/// The cache itself: one per scheduler, shared by every store it wraps.
pub struct BlockCache {
    cfg: BlockCacheConfig,
    dir: PathBuf,
    /// The lock this cache holds on `<dir>/.lock` while it lives: a directory whose lock is free
    /// has no owner (its process is gone), whatever the process table says.
    _lock: std::fs::File,
    state: Mutex<State>,
    inflight: Mutex<HashMap<u128, (Run, usize)>>,
    counters: Counters,
    /// At most [`MAX_CONCURRENT_FETCHES`] runs are fetched from the store at a time, whatever the
    /// number of readers and of missing runs in their ranges.
    fetches: Arc<tokio::sync::Semaphore>,
    /// The memory the runs in flight hold, reserved from the pool when one is attached.
    fetch_memory: Mutex<Option<Reservation>>,
}

/// Most runs fetched from the store at the same time by one cache.
const MAX_CONCURRENT_FETCHES: usize = 8;

/// Whether `e` was raised by the cache itself (a refused reservation, a timeout, a short read, a
/// size mismatch, a failed task), not by the store.
fn is_cache_error(e: &object_store::Error) -> bool {
    matches!(
        e,
        object_store::Error::Generic {
            store: "block cache",
            ..
        }
    )
}

/// The error a failed run is reported as. A failure of the cache stays a cache error (the plain
/// read answers); a failure of the store is the store's: it is returned, not asked again at once,
/// with `NotFound` and `Precondition` kept as they are and any other error behind a source chain
/// that still reaches its `TransientError` for the scheduler's retry.
fn run_failure(e: &Arc<object_store::Error>, location: &Path) -> object_store::Error {
    if is_cache_error(e) {
        return object_store::Error::Generic {
            store: "block cache",
            source: e.to_string().into(),
        };
    }
    match e.as_ref() {
        object_store::Error::NotFound { .. } => object_store::Error::NotFound {
            path: location.to_string(),
            source: Box::new(Arc::clone(e)),
        },
        object_store::Error::Precondition { .. } => object_store::Error::Precondition {
            path: location.to_string(),
            source: Box::new(Arc::clone(e)),
        },
        _ => object_store::Error::Generic {
            store: "object store",
            source: Box::new(Arc::clone(e)),
        },
    }
}

/// A random part for a directory name (the hasher is seeded randomly per process).
fn random_suffix() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    h.finish()
}

impl fmt::Debug for BlockCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockCache")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A 128-bit name for `text`: two seeded XXH64 hashes.
fn name_of(text: &str) -> u128 {
    let hi = twox_hash::XxHash64::oneshot(0x9E37_79B9_7F4A_7C15, text.as_bytes());
    let lo = twox_hash::XxHash64::oneshot(0xC2B2_AE3D_27D4_EB4F, text.as_bytes());
    (u128::from(hi) << 64) | u128::from(lo)
}

const MAGIC: &[u8; 4] = b"KBC2";
/// Magic, key length (u32), payload length (u64), payload XXH64 (u64).
const HEADER: usize = 4 + 4 + 8 + 8;

/// The longest one fetch of blocks may take before its readers are told it failed (they fall back
/// to the store); it also ends the in-flight entry of a request that never answers.
const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// `n` as an index (a block number or offset within a request that already fits in memory).
fn at(n: u64) -> usize {
    usize::try_from(n).unwrap_or(usize::MAX)
}

impl BlockCache {
    /// A cache over `cfg`, with an empty disk part.
    ///
    /// # Errors
    /// The disk directory cannot be created.
    pub fn new(mut cfg: BlockCacheConfig) -> std::io::Result<Arc<Self>> {
        cfg.block_bytes = cfg.block_bytes.max(4096);
        // One directory per cache (a process runs many sessions, each with its own cache): the
        // process id, a sequence number and a random part, created exclusively. Processes in
        // different PID namespaces may share the directory and have the same pid and sequence, so
        // a name is never assumed free: an existing one is skipped, never emptied.
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let pid = std::process::id();
        purge_dead(&cfg.dir);
        // Owner-only: another local user must not be able to put blocks of their own into a
        // cache (the default root is under the shared temp directory).
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&cfg.dir)?;
        let mut attempt = 0_u32;
        let dir = loop {
            let dir = cfg.dir.join(format!(
                "p{pid}-{}-{:016x}",
                SEQUENCE.fetch_add(1, Ordering::Relaxed),
                random_suffix()
            ));
            let mut leaf = std::fs::DirBuilder::new();
            #[cfg(unix)]
            std::os::unix::fs::DirBuilderExt::mode(&mut leaf, 0o700);
            match leaf.create(&dir) {
                Ok(()) => {
                    // A root another user created (or owns the parent of) could be used to plant
                    // blocks: the root must be ours, as the directory just made in it is.
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        let (root, own) = (std::fs::metadata(&cfg.dir)?, std::fs::metadata(&dir)?);
                        if root.uid() != own.uid() {
                            drop(std::fs::remove_dir(&dir));
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::PermissionDenied,
                                "the cache directory belongs to another user",
                            ));
                        }
                    }
                    break dir;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempt < 16 => {
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        };
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK_FILE))?;
        lock.try_lock().map_err(|e| match e {
            std::fs::TryLockError::Error(e) => e,
            std::fs::TryLockError::WouldBlock => {
                std::io::Error::other("the cache directory is locked by another cache")
            }
        })?;
        Ok(Arc::new(Self {
            cfg,
            dir,
            _lock: lock,
            state: Mutex::new(State::default()),
            inflight: Mutex::new(HashMap::new()),
            counters: Counters::default(),
            fetches: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_FETCHES)),
            fetch_memory: Mutex::new(None),
        }))
    }

    /// What the cache has done so far.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        CacheStats {
            memory_hits: get(&self.counters.memory_hits),
            disk_hits: get(&self.counters.disk_hits),
            misses: get(&self.counters.misses),
            fetched_bytes: get(&self.counters.fetched_bytes),
            errors: get(&self.counters.errors),
            evictions: get(&self.counters.evictions),
        }
    }

    /// Bytes on disk now.
    #[must_use]
    pub fn disk_used(&self) -> u64 {
        lock(&self.state).disk.used
    }

    /// Accounts the memory tier in its own child of `pool` (limited to the tier's size): a block
    /// is held in memory only while the pool grants its bytes. Blocks held so far are dropped.
    pub fn attach_pool(&self, pool: &Arc<dyn MemoryPool>) {
        let limit = usize::try_from(self.cfg.memory_bytes).unwrap_or(usize::MAX);
        let child = pool.child("block-cache", Some(limit));
        let id = child.register("block-cache", None);
        let mut st = lock(&self.state);
        st.memory = Lru::default();
        st.reservation = Some(Reservation::new(child, id));
        drop(st);
        let fetch_child = pool.child("block-cache-fetch", None);
        let fetch_id = fetch_child.register("block-cache-fetch", None);
        *lock(&self.fetch_memory) = Some(Reservation::new(fetch_child, fetch_id));
    }

    /// Reserves `bytes` for a fetch in flight; `false` when an attached pool refuses.
    fn reserve_fetch(&self, bytes: usize) -> bool {
        match lock(&self.fetch_memory).as_mut() {
            Some(r) => r.try_grow(bytes).is_ok(),
            None => true,
        }
    }

    /// Returns the bytes reserved by [`Self::reserve_fetch`].
    fn release_fetch(&self, bytes: usize) {
        if let Some(r) = lock(&self.fetch_memory).as_mut() {
            r.shrink(bytes.min(r.size()));
        }
    }

    /// Bytes held in memory now.
    #[must_use]
    pub fn memory_used(&self) -> u64 {
        lock(&self.state).memory.used
    }

    /// Returns `bytes` of the memory tier to the pool.
    fn release(st: &mut State, bytes: usize) {
        if let Some(r) = st.reservation.as_mut() {
            r.shrink(bytes.min(r.size()));
        }
    }

    fn file(&self, name: u128) -> PathBuf {
        self.dir.join(format!("{name:032x}.blk"))
    }

    fn next_tick(state: &mut State) -> u64 {
        state.tick += 1;
        state.tick
    }

    /// The block `key` (named `name`), from memory or disk.
    async fn lookup(self: &Arc<Self>, name: u128, key: &str) -> Option<Bytes> {
        {
            let mut st = lock(&self.state);
            let tick = Self::next_tick(&mut st);
            if let Some(b) = st.memory.touch(name, tick) {
                bump(&self.counters.memory_hits, 1);
                return Some(b.clone());
            }
            let tick = Self::next_tick(&mut st);
            st.disk.touch(name, tick)?;
        }
        let (path, want) = (self.file(name), key.to_owned());
        let read = tokio::task::spawn_blocking(move || read_block(&path, &want)).await;
        match read {
            Ok(Ok(bytes)) => {
                bump(&self.counters.disk_hits, 1);
                self.remember(name, &bytes);
                Some(bytes)
            }
            _ => {
                // Unreadable or not the block we asked for: forget it.
                bump(&self.counters.errors, 1);
                self.forget_disk(name);
                None
            }
        }
    }

    fn forget_disk(&self, name: u128) {
        let mut st = lock(&self.state);
        // The recency entry goes with the block: a stale tick would later
        // evict the refetched copy and stop eviction at the missing key.
        if let Some((tick, size)) = st.disk.entries.remove(&name) {
            st.disk.order.remove(&tick);
            st.disk.used = st.disk.used.saturating_sub(size);
        }
        drop(st);
        let _ = std::fs::remove_file(self.file(name));
    }

    /// Puts `bytes` in the memory tier (least recently used blocks leave).
    fn remember(&self, name: u128, bytes: &Bytes) {
        let cap = self.cfg.memory_bytes;
        let size = bytes.len() as u64;
        if cap == 0 || size > cap {
            return;
        }
        let mut st = lock(&self.state);
        let tick = Self::next_tick(&mut st);
        for (_, _, old) in st.memory.insert(name, tick, size, bytes.clone()) {
            st.memory.used = st.memory.used.saturating_sub(old.len() as u64);
            Self::release(&mut st, old.len());
        }
        // With a pool, the block's memory must be granted too: evict older blocks while it
        // refuses, and keep the block on disk only when nothing is left to evict.
        let want = bytes.len();
        loop {
            match st.reservation.as_mut().map(|r| r.try_grow(want)) {
                None | Some(Ok(())) => break,
                Some(Err(_)) => {
                    let Some((key, old)) = st.memory.pop_oldest() else {
                        break;
                    };
                    bump(&self.counters.evictions, u64::from(key != name));
                    st.memory.used = st.memory.used.saturating_sub(old.len() as u64);
                    Self::release(&mut st, old.len());
                    if key == name {
                        return;
                    }
                }
            }
        }
        while st.memory.used > cap {
            let Some((key, old)) = st.memory.pop_oldest() else {
                break;
            };
            if key != name {
                bump(&self.counters.evictions, 1);
            }
            st.memory.used = st.memory.used.saturating_sub(old.len() as u64);
            Self::release(&mut st, old.len());
        }
    }

    /// Writes the block to disk, removing the least recently used ones to stay under the cap.
    async fn persist(self: &Arc<Self>, name: u128, key: String, bytes: Bytes) {
        let size = bytes.len() as u64;
        if size > self.cfg.disk_bytes {
            return;
        }
        // The file first, then the index: a lookup never sees an entry whose file is not there yet.
        let (path, dir) = (self.file(name), self.dir.clone());
        let wrote =
            tokio::task::spawn_blocking(move || write_block(&dir, &path, &key, &bytes)).await;
        if !matches!(wrote, Ok(Ok(()))) {
            bump(&self.counters.errors, 1);
            return;
        }
        let victims = {
            let mut st = lock(&self.state);
            let tick = Self::next_tick(&mut st);
            let mut victims = Vec::new();
            // A replaced entry has the name of the file just written: it is not a victim.
            for (_, _, old) in st.disk.insert(name, tick, size, size) {
                st.disk.used = st.disk.used.saturating_sub(old);
            }
            while st.disk.used > self.cfg.disk_bytes {
                let Some((k, old)) = st.disk.pop_oldest() else {
                    break;
                };
                st.disk.used = st.disk.used.saturating_sub(old);
                if k != name {
                    bump(&self.counters.evictions, 1);
                    victims.push(k);
                }
            }
            victims
        };
        let victim_files: Vec<PathBuf> = victims.iter().map(|k| self.file(*k)).collect();
        drop(
            tokio::task::spawn_blocking(move || {
                for v in victim_files {
                    let _ = std::fs::remove_file(v);
                }
            })
            .await,
        );
    }
}

impl Drop for BlockCache {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

/// The file a live cache locks in its directory.
const LOCK_FILE: &str = ".lock";

/// Removes the cache directories (`p<pid>-<n>`) that no cache holds any longer: one whose lock file
/// can be locked has no owner. A directory without a lock file (an older layout) stays, as does
/// everything that is not a cache directory.
fn purge_dead(root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(rest) = name.to_str().and_then(|n| n.strip_prefix('p')) else {
            continue;
        };
        let mut parts = rest.split('-');
        let (Ok(_pid), Some(Ok(_seq))) = (
            parts.next().unwrap_or("").parse::<u32>(),
            parts.next().map(str::parse::<u64>),
        ) else {
            continue;
        };
        // An optional random part follows; nothing else.
        if parts
            .next()
            .is_some_and(|r| u64::from_str_radix(r, 16).is_err())
            || parts.next().is_some()
        {
            continue;
        }
        let lock_path = entry.path().join(LOCK_FILE);
        // A cache that is being created has its lock file before it holds the lock: a lock file
        // made a moment ago belongs to a cache that has not locked it yet.
        let young = std::fs::metadata(&lock_path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_none_or(|age| age < std::time::Duration::from_secs(60));
        if young {
            continue;
        }
        let Ok(lock) = std::fs::OpenOptions::new().write(true).open(&lock_path) else {
            continue;
        };
        // A free lock: nobody owns the directory. Held (a live cache of this or another process,
        // possibly in another PID namespace sharing the directory): leave it.
        if lock.try_lock().is_ok() {
            drop(lock);
            drop(std::fs::remove_dir_all(entry.path()));
        }
    }
}

fn read_block(path: &std::path::Path, key: &str) -> std::io::Result<Bytes> {
    let raw = std::fs::read(path)?;
    let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "a corrupt cache block");
    let head = raw.get(..HEADER).ok_or_else(bad)?;
    if &head[..4] != MAGIC {
        return Err(bad());
    }
    let key_len = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let word = |at: usize| {
        let mut w = [0u8; 8];
        w.copy_from_slice(&head[at..at + 8]);
        u64::from_le_bytes(w)
    };
    let (len, sum) = (word(8), word(16));
    let stored = raw.get(HEADER..HEADER + key_len).ok_or_else(bad)?;
    if stored != key.as_bytes() {
        return Err(bad());
    }
    let payload = Bytes::from(raw).slice(HEADER + key_len..);
    // A truncated or damaged payload is a miss, never a short or wrong block.
    if payload.len() as u64 != len || twox_hash::XxHash64::oneshot(0, &payload) != sum {
        return Err(bad());
    }
    Ok(payload)
}

fn write_block(
    dir: &std::path::Path,
    path: &std::path::Path,
    key: &str,
    bytes: &[u8],
) -> std::io::Result<()> {
    static TMP: AtomicU64 = AtomicU64::new(0);
    let tmp = dir.join(format!(
        "{}.{}.tmp",
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("block"),
        TMP.fetch_add(1, Ordering::Relaxed)
    ));
    let key_len = u32::try_from(key.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "a long cache key"))?;
    let write = || -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(MAGIC)?;
        file.write_all(&key_len.to_le_bytes())?;
        file.write_all(&(bytes.len() as u64).to_le_bytes())?;
        file.write_all(&twox_hash::XxHash64::oneshot(0, bytes).to_le_bytes())?;
        file.write_all(key.as_bytes())?;
        file.write_all(bytes)?;
        drop(file);
        std::fs::rename(&tmp, path)
    };
    // Any failure (a full disk included) leaves no temporary file behind.
    write().inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Request extension: the size of the object a read names, which the scheduler knows from the
/// listing. The cache aligns blocks against it, so it needs no metadata request of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectSize(pub u64);

/// Whose credentials a [`CachingStore`] reads with, the part of a block's key that keeps a byte
/// from being served to a query whose credentials could not have read it (E15): a cache hit skips
/// the store's own access check.
///
/// The default of [`CachingStore::new`] is [`CredentialScope::Isolated`]: sharing across
/// principals is something a caller asks for, never something it gets by omission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialScope {
    /// One engine identity reads for every principal (a single-tenant compute): blocks are shared.
    /// The caller vouches that a store's credentials grant every principal the same reads.
    Shared,
    /// The credentials of one principal (its [`Principal::id`](krishiv_common::Principal::id) or a
    /// token's subject): blocks are shared only with the same scope text.
    Principal(String),
    /// Credentials seen by this registration alone (a random nonce): blocks are never shared.
    Isolated,
}

impl CredentialScope {
    /// The text joined to every block key. An isolated scope draws a fresh process-unique nonce.
    fn key_text(&self) -> String {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        match self {
            Self::Shared => "shared".to_owned(),
            Self::Principal(p) => format!("p:{p}"),
            Self::Isolated => format!("n:{}", NEXT.fetch_add(1, Ordering::Relaxed)),
        }
    }
}

/// A store whose bounded range reads go through a [`BlockCache`].
pub struct CachingStore {
    inner: Arc<dyn ObjectStore>,
    /// The store as displayed (no credentials), then the credential scope.
    identity: String,
    /// `identity`, a separator and the scope text: the head of every block key.
    key_prefix: String,
    cache: Arc<BlockCache>,
}

impl fmt::Debug for CachingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CachingStore({})", self.identity)
    }
}

impl fmt::Display for CachingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CachingStore({})", self.identity)
    }
}

/// The object's version as the scheduler pins it on a read.
fn version_text(options: &GetOptions) -> String {
    if let Some(e) = &options.if_match {
        format!("e:{e}")
    } else if let Some(t) = &options.if_unmodified_since {
        format!("m:{}", t.timestamp_nanos_opt().unwrap_or(0))
    } else {
        "i".to_owned()
    }
}

impl CachingStore {
    /// `inner` behind `cache`; `identity` names the store in keys (never with credentials).
    #[must_use]
    pub fn new(
        inner: Arc<dyn ObjectStore>,
        identity: impl Into<String>,
        cache: Arc<BlockCache>,
    ) -> Self {
        Self::with_scope(inner, identity, cache, &CredentialScope::Isolated)
    }

    /// As [`Self::new`], reading with the credentials of `scope`: blocks are shared only between
    /// stores of the same identity and an equal scope.
    #[must_use]
    pub fn with_scope(
        inner: Arc<dyn ObjectStore>,
        identity: impl Into<String>,
        cache: Arc<BlockCache>,
        scope: &CredentialScope,
    ) -> Self {
        // Both parts are in every key; the display text keeps only the store.
        let identity = identity.into();
        Self {
            inner,
            key_prefix: format!("{identity}\0{}", scope.key_text()),
            identity,
            cache,
        }
    }

    fn block_key(&self, location: &Path, version: &str, block: u64) -> String {
        format!("{}\0{location}\0{version}\0{block}", self.key_prefix)
    }

    /// The object a read names, from what the scheduler sent: its size ([`ObjectSize`]) and the
    /// version it pins. `None` when the read carries no size (it is then not cached).
    fn meta(location: &Path, options: &GetOptions) -> Option<ObjectMeta> {
        let size = options.extensions.get::<ObjectSize>()?.0;
        Some(ObjectMeta {
            location: location.clone(),
            last_modified: options
                .if_unmodified_since
                .unwrap_or(chrono::DateTime::UNIX_EPOCH),
            size,
            e_tag: options.if_match.clone(),
            version: None,
        })
    }

    /// Fetches blocks `first..=last` (contiguous, none cached) in one request, caches them and
    /// returns them in order.
    fn fetch_run(
        &self,
        location: &Path,
        options: &GetOptions,
        meta: &ObjectMeta,
        version: &str,
        first: u64,
        last: u64,
    ) -> Run {
        let (inner, cache) = (Arc::clone(&self.inner), Arc::clone(&self.cache));
        let bs = cache.cfg.block_bytes;
        let size = meta.size;
        let (start, end) = (first * bs, ((last + 1) * bs).min(size));
        let keys: Vec<(u128, String)> = (first..=last)
            .map(|b| {
                let k = self.block_key(location, version, b);
                (name_of(&k), k)
            })
            .collect();
        let opts = GetOptions {
            range: Some(GetRange::Bounded(start..end)),
            if_match: options.if_match.clone(),
            if_unmodified_since: options.if_unmodified_since,
            extensions: ScheduledRequest.into_extensions(),
            ..GetOptions::default()
        };
        let location = location.clone();
        let task = tokio::spawn(async move {
            // At most a few runs in flight; the buffer of a run is reserved from the pool (a
            // refusal ends the fetch, and the reader falls back to the plain read).
            let _permit = Arc::clone(&cache.fetches).acquire_owned().await;
            let held = usize::try_from(end - start).unwrap_or(usize::MAX);
            let reserved = cache.reserve_fetch(held);
            let result = tokio::time::timeout(FETCH_TIMEOUT, async {
                if !reserved {
                    return Err(object_store::Error::Generic {
                        store: "block cache",
                        source: "no memory for a block fetch".into(),
                    });
                }
                let got = inner.get_opts(&location, opts).await?;
                if got.meta.size != size {
                    return Err(object_store::Error::Generic {
                        store: "block cache",
                        source: "the object changed size under a read".into(),
                    });
                }
                got.bytes().await
            })
            .await
            .unwrap_or_else(|_| {
                Err(object_store::Error::Generic {
                    store: "block cache",
                    source: "a block fetch timed out".into(),
                })
            });
            let outcome: RunResult = match result {
                Ok(bytes) if bytes.len() as u64 == end - start => {
                    bump(&cache.counters.misses, keys.len() as u64);
                    bump(&cache.counters.fetched_bytes, bytes.len() as u64);
                    let blocks: Vec<Bytes> = (0..keys.len())
                        .map(|i| {
                            let from = i as u64 * bs;
                            bytes.slice(at(from)..at((from + bs).min(end - start)))
                        })
                        .collect();
                    for ((name, key), block) in keys.iter().zip(&blocks) {
                        // Its own allocation: a slice would keep the whole run's buffer alive while
                        // the pool is charged for the block only.
                        if cache.cfg.memory_bytes > 0 {
                            cache.remember(*name, &Bytes::copy_from_slice(block));
                        }
                        cache.persist(*name, key.clone(), block.clone()).await;
                    }
                    Ok(Arc::new(blocks))
                }
                Ok(_) => Err(Arc::new(object_store::Error::Generic {
                    store: "block cache",
                    source: "a short read".into(),
                })),
                Err(e) => Err(Arc::new(e)),
            };
            if reserved {
                cache.release_fetch(held);
            }
            // Done: later readers find the blocks cached, so the entries go.
            let mut flight = lock(&cache.inflight);
            for (name, _) in &keys {
                flight.remove(name);
            }
            drop(flight);
            outcome
        });
        // The fetch runs as a task of its own, so a reader that is dropped (a timeout, a cancelled
        // query) neither stalls it nor leaves its entries behind; it ends by `FETCH_TIMEOUT` at the latest.
        async move {
            task.await.unwrap_or_else(|e| {
                Err(Arc::new(object_store::Error::Generic {
                    store: "block cache",
                    source: e.to_string().into(),
                }))
            })
        }
        .boxed()
        .shared()
    }

    async fn cached_range(
        &self,
        location: &Path,
        options: &GetOptions,
        range: Range<u64>,
    ) -> StoreResult<Option<GetResult>> {
        let version = version_text(options);
        let Some(meta) = Self::meta(location, options) else {
            return Ok(None);
        };
        if range.start >= range.end || range.end > meta.size {
            return Ok(None);
        }
        let bs = self.cache.cfg.block_bytes;
        let (first, last) = (range.start / bs, (range.end - 1) / bs);
        let mut parts: Vec<Option<Bytes>> = vec![None; at(last - first + 1)];
        let mut missing: Vec<u64> = Vec::new();
        for b in first..=last {
            let key = self.block_key(location, &version, b);
            match self.cache.lookup(name_of(&key), &key).await {
                Some(bytes) => parts[at(b - first)] = Some(bytes),
                None => missing.push(b),
            }
        }
        // Group the missing blocks into contiguous runs; a block someone else is already fetching
        // is awaited, not fetched again.
        let mut waits: Vec<(Run, usize, u64)> = Vec::new();
        let mut i = 0;
        while i < missing.len() {
            let mut j = i;
            let mut owned: Vec<u64> = Vec::new();
            // One lock for the run: take the blocks nobody fetches, join the others' runs.
            let mut flight = lock(&self.cache.inflight);
            while j < missing.len() && (j == i || missing[j] == missing[j - 1] + 1) {
                let b = missing[j];
                let name = name_of(&self.block_key(location, &version, b));
                if let Some((run, index)) = flight.get(&name) {
                    waits.push((run.clone(), *index, b));
                } else {
                    owned.push(b);
                }
                j += 1;
            }
            // `owned` may hold several sub-runs when a foreign block sits between them.
            let mut k = 0;
            while k < owned.len() {
                let mut l = k;
                while l + 1 < owned.len() && owned[l + 1] == owned[l] + 1 {
                    l += 1;
                }
                let run = self.fetch_run(location, options, &meta, &version, owned[k], owned[l]);
                for (index, b) in (owned[k]..=owned[l]).enumerate() {
                    let name = name_of(&self.block_key(location, &version, b));
                    flight.insert(name, (run.clone(), index));
                    waits.push((run.clone(), index, b));
                }
                k = l + 1;
            }
            drop(flight);
            i = j;
        }
        for (run, index, b) in waits {
            let blocks = match run.await {
                Ok(blocks) => blocks,
                Err(e) => return Err(run_failure(&e, location)),
            };
            parts[at(b - first)] = blocks.get(index).cloned();
        }
        let mut out = BytesMut::with_capacity(at(range.end - range.start));
        for (n, part) in parts.into_iter().enumerate() {
            let Some(block) = part else {
                return Ok(None);
            };
            let block_start = (first + n as u64) * bs;
            let from = range.start.max(block_start) - block_start;
            let to = range.end.min(block_start + block.len() as u64) - block_start;
            out.extend_from_slice(&block[at(from)..at(to)]);
        }
        let bytes = out.freeze();
        let payload =
            GetResultPayload::Stream(Box::pin(futures::stream::once(async move { Ok(bytes) })));
        Ok(Some(GetResult {
            payload,
            meta,
            range,
            attributes: object_store::Attributes::new(),
            extensions: object_store::Extensions::new(),
        }))
    }
}

#[async_trait]
impl ObjectStore for CachingStore {
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
        let cacheable = !options.head
            && options.if_none_match.is_none()
            && options.if_modified_since.is_none()
            && options.version.is_none();
        if let (true, Some(GetRange::Bounded(range))) = (cacheable, options.range.clone()) {
            match self.cached_range(location, &options, range).await {
                Ok(Some(result)) => return Ok(result),
                Ok(None) => {}
                // A failure of the cache is answered by the plain read; one of the store is its own
                // answer (the scheduler classifies and retries it), not a reason to ask again.
                Err(e) if is_cache_error(&e) => bump(&self.cache.counters.errors, 1),
                Err(e) => return Err(e),
            }
        }
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> StoreResult<Vec<Bytes>> {
        self.inner.get_ranges(location, ranges).await
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

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
