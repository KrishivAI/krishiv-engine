// SPDX-License-Identifier: Apache-2.0
//! The I/O layer: [`IoScheduler`] (process-wide) and [`IoContext`] (per
//! query). Every byte a format reads or writes goes through them.
//!
//! - **Reads** ([`IoContext::get_ranges`]) are coalesced
//!   ([`coalesce`]), limited to `krishiv.io.maxConcurrentRequests` in flight
//!   per store, version-checked (`if_match` on the ETag, else
//!   `if_unmodified_since` on the modification time: a file that changed
//!   since listing fails with `KRISHIV_FILE_CHANGED_DURING_READ`, never
//!   mixes versions), timed out per attempt, and retried on transient
//!   failures ([`TransientError`], I/O timeouts) with full-jitter exponential
//!   backoff that honours `Retry-After`.
//! - **Footers** take one speculative tail read ([`IoContext::get_tail`]).
//! - **Metadata** is cached ([`IoContext::read_metadata`], [`cache`]).
//! - **Writes** go through [`Upload`].
//! - **Counters** ([`IoCounters`]) are kept per query and process-wide.
//!
//! The scheduler can run requests on a dedicated I/O runtime
//! ([`IoScheduler::with_runtime`]): fetches then progress while compute
//! workers decode. Without one they run on the caller's task. Either way,
//! every future here is drop-safe: dropping it cancels the requests.

pub mod cache;
mod coalesce;
pub mod fault;
mod pace;
pub mod upload;

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::{BoxFuture, try_join_all};
use krishiv_common::config::{
    Config, ConfigValue, IO_COALESCE_BYTES, IO_FOOTER_READ_BYTES, IO_MAX_CONCURRENT_REQUESTS,
    IO_MAX_REQUEST_BYTES, IO_MAX_RETRIES, IO_METADATA_CACHE_BYTES, IO_MULTIPART_PART_BYTES,
    IO_REQUEST_TIMEOUT,
};
use krishiv_common::secret::{OptionValue, SecretString};
use krishiv_memory::{ConsumerHandle, MemoryPool, Reservation};
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectMeta, ObjectStore, PutOptions, PutPayload};
use tokio::sync::Semaphore;

pub use self::cache::{CacheKey, MetadataCache};
pub use self::coalesce::coalesce;
pub use self::fault::{FaultStats, FaultStore, ScheduledRequest, TransientError};
pub use self::pace::StorePolicy;
use self::pace::{Direction, Pacer};
pub use self::upload::Upload;
use crate::error::{
    Result, cancelled, changed, internal, not_found, read_failed, store_failed, write_failed,
};
use crate::format::{FileFormat, FileMetadata, MetadataNeeds, ReadAt};
use crate::options::{FormatOptions, NoSecrets};

/// Largest footer read a format may ask for in one tail read.
pub const MAX_TAIL_BYTES: u64 = 1024 * 1024;

/// Identifies a registered store, e.g. `s3://bucket`. Its text is printed in
/// paths, so it must not hold credentials (they are redacted if it does).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct StoreId(Arc<str>);

impl StoreId {
    /// A store id.
    #[must_use]
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }
}

impl fmt::Display for StoreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Typed redaction (ADR 0038): URL credentials never print.
        OptionValue::from_user("path", self.0.as_ref()).fmt(f)
    }
}

impl fmt::Debug for StoreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StoreId({self})")
    }
}

/// The version of a file seen at listing; every read checks it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VersionToken {
    /// The store's ETag (`if_match`).
    ETag(Arc<str>),
    /// Last modification, microseconds since the epoch
    /// (`if_unmodified_since`), for stores without ETags.
    Modified(i64),
    /// The file never changes (table-format data files): no check.
    Immutable,
}

impl fmt::Display for VersionToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ETag(e) => write!(f, "ETag {e}"),
            Self::Modified(us) => write!(f, "modified at {us} us"),
            Self::Immutable => f.write_str("immutable"),
        }
    }
}

/// A file as listed: identity, size and version.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileRef {
    /// The store holding it.
    pub store: StoreId,
    /// Its path in the store.
    pub path: Path,
    /// Its size in bytes.
    pub size: u64,
    /// Its version at listing.
    pub version: VersionToken,
    /// Last modification, microseconds since the epoch, if known
    /// (`_metadata.file_modification_time`).
    pub modified_micros: Option<i64>,
}

impl FileRef {
    /// The file `meta` lists in `store`, versioned by its ETag when it has
    /// one, else its modification time.
    #[must_use]
    pub fn from_meta(store: StoreId, meta: &ObjectMeta) -> Self {
        let micros = meta.last_modified.timestamp_micros();
        Self {
            store,
            path: meta.location.clone(),
            size: meta.size,
            version: meta
                .e_tag
                .as_deref()
                .map_or(VersionToken::Modified(micros), |e| {
                    VersionToken::ETag(Arc::from(e))
                }),
            modified_micros: Some(micros),
        }
    }

    /// The path for messages: `<store>/<path>`, credentials redacted.
    #[must_use]
    pub fn display_path(&self) -> String {
        format!("{}/{}", self.store, self.path)
    }
}

/// A byte range of a file `[start, end)`: a split, or a unit's extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteRange {
    /// First byte.
    pub start: u64,
    /// One past the last byte.
    pub end: u64,
}

impl ByteRange {
    /// `[start, end)`.
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// The whole file.
    #[must_use]
    pub fn whole(size: u64) -> Self {
        Self {
            start: 0,
            end: size,
        }
    }

    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this split owns the unit spanning `unit`: the unit's midpoint
    /// lies in the split (parquet-mr's rule), so splits that tile a file
    /// own each unit exactly once.
    #[must_use]
    pub fn owns(&self, unit: &Range<u64>) -> bool {
        let mid = unit.start + (unit.end.saturating_sub(unit.start)) / 2;
        self.start <= mid && mid < self.end
    }
}

/// Resolves `secret('scope', 'key')` references (the `SecretProvider` of
/// ADR 0036 R12 implements it).
pub trait SecretResolver: Send + Sync + fmt::Debug {
    /// The secret's plaintext.
    ///
    /// # Errors
    /// Whatever the provider reports for an unknown or forbidden secret.
    fn resolve<'a>(&'a self, scope: &'a str, key: &'a str) -> BoxFuture<'a, Result<SecretString>>;
}

/// I/O settings (cluster scope).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoConfig {
    /// Requests in flight per store.
    pub max_concurrent_requests: usize,
    /// Ranges closer than this are fetched together.
    pub coalesce_bytes: u64,
    /// Coalesced requests stay below this.
    pub max_request_bytes: u64,
    /// Default speculative footer read.
    pub footer_read_bytes: u64,
    /// Metadata cache size.
    pub metadata_cache_bytes: u64,
    /// Multipart part size.
    pub multipart_part_bytes: u64,
    /// Retries of a transient failure.
    pub max_retries: u32,
    /// Timeout of one attempt.
    pub request_timeout: Duration,
    /// First retry backoff (doubles per retry, full jitter).
    pub retry_base: Duration,
    /// Largest backoff.
    pub retry_max: Duration,
    /// Seed of the jitter (deterministic tests).
    pub seed: u64,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self::from_config(&Config::new())
    }
}

fn size_of(c: &Config, e: &'static krishiv_common::config::ConfigEntry, default: u64) -> u64 {
    c.get(e).and_then(ConfigValue::as_size).unwrap_or(default)
}

impl IoConfig {
    /// The settings in `c` (defaults for unset keys).
    #[must_use]
    pub fn from_config(c: &Config) -> Self {
        let int = |e, d: i64| c.get(e).and_then(ConfigValue::as_int).unwrap_or(d);
        Self {
            max_concurrent_requests: usize::try_from(int(&IO_MAX_CONCURRENT_REQUESTS, 64))
                .unwrap_or(64)
                .max(1),
            coalesce_bytes: size_of(c, &IO_COALESCE_BYTES, 1 << 20),
            max_request_bytes: size_of(c, &IO_MAX_REQUEST_BYTES, 16 << 20).max(1),
            footer_read_bytes: size_of(c, &IO_FOOTER_READ_BYTES, 64 << 10).clamp(8, MAX_TAIL_BYTES),
            metadata_cache_bytes: size_of(c, &IO_METADATA_CACHE_BYTES, 64 << 20),
            multipart_part_bytes: size_of(c, &IO_MULTIPART_PART_BYTES, 8 << 20),
            max_retries: u32::try_from(int(&IO_MAX_RETRIES, 5)).unwrap_or(5),
            request_timeout: c
                .get(&IO_REQUEST_TIMEOUT)
                .and_then(ConfigValue::as_duration)
                .unwrap_or(Duration::from_secs(30)),
            retry_base: Duration::from_millis(50),
            retry_max: Duration::from_secs(10),
            seed: 0x5EED,
        }
    }
}

/// Request and byte counters (per query, and process-wide totals).
#[derive(Debug, Default)]
pub struct IoCounters {
    /// Store requests (attempts), including retries and parts.
    pub requests: AtomicU64,
    /// Bytes read.
    pub bytes_read: AtomicU64,
    /// Bytes written.
    pub bytes_written: AtomicU64,
    /// Retried attempts.
    pub retries: AtomicU64,
    /// Speculative tail reads.
    pub tail_reads: AtomicU64,
    /// Metadata cache hits.
    pub cache_hits: AtomicU64,
    /// Metadata cache misses.
    pub cache_misses: AtomicU64,
    /// Multipart uploads started.
    pub uploads_started: AtomicU64,
    /// Multipart uploads completed.
    pub uploads_completed: AtomicU64,
    /// Multipart uploads aborted.
    pub uploads_aborted: AtomicU64,
}

impl IoCounters {
    /// The value of one counter.
    #[must_use]
    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }
}

struct Registered {
    store: Arc<dyn ObjectStore>,
    lane: Arc<Lane>,
}

/// What every request to one store goes through: prefix pacing, then an
/// in-flight permit.
#[derive(Debug)]
pub(crate) struct Lane {
    /// The store's in-flight request cap, shared by reads and writes.
    permits: Semaphore,
    pacer: Pacer,
}

impl Lane {
    /// Waits for the prefix of `path` to allow a request, then for an
    /// in-flight permit (held until the guard drops).
    pub(crate) async fn admit(
        &self,
        direction: Direction,
        path: &Path,
    ) -> Result<tokio::sync::SemaphorePermit<'_>> {
        self.pacer.pace(direction, path).await;
        self.permits.acquire().await.map_err(|_| cancelled())
    }
}

/// The process-wide I/O scheduler (see the module docs).
pub struct IoScheduler {
    config: IoConfig,
    stores: RwLock<HashMap<StoreId, Registered>>,
    cache: MetadataCache,
    runtime: Option<tokio::runtime::Handle>,
    totals: IoCounters,
    jitter: AtomicU64,
}

impl fmt::Debug for IoScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoScheduler")
            .field("config", &self.config)
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
}

impl IoScheduler {
    /// A scheduler whose requests run on the caller's task.
    #[must_use]
    pub fn new(config: IoConfig) -> Arc<Self> {
        Self::build(config, None)
    }

    /// A scheduler whose requests run on `runtime` (the I/O runtime).
    #[must_use]
    pub fn with_runtime(config: IoConfig, runtime: tokio::runtime::Handle) -> Arc<Self> {
        Self::build(config, Some(runtime))
    }

    fn build(config: IoConfig, runtime: Option<tokio::runtime::Handle>) -> Arc<Self> {
        Arc::new(Self {
            cache: MetadataCache::new(config.metadata_cache_bytes),
            jitter: AtomicU64::new(config.seed),
            config,
            stores: RwLock::new(HashMap::new()),
            runtime,
            totals: IoCounters::default(),
        })
    }

    /// Registers (or replaces) the store `id`, unpaced (local files,
    /// memory).
    pub fn register_store(&self, id: StoreId, store: Arc<dyn ObjectStore>) {
        self.register_store_with(id, store, StorePolicy::UNPACED);
    }

    /// Registers (or replaces) the store `id` with a pacing `policy`
    /// ([`StorePolicy::S3`] for S3).
    pub fn register_store_with(
        &self,
        id: StoreId,
        store: Arc<dyn ObjectStore>,
        policy: StorePolicy,
    ) {
        let lane = Arc::new(Lane {
            permits: Semaphore::new(self.config.max_concurrent_requests),
            pacer: Pacer::new(policy),
        });
        self.stores
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, Registered { store, lane });
    }

    fn store(&self, id: &StoreId) -> Result<(Arc<dyn ObjectStore>, Arc<Lane>)> {
        self.stores
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .map(|r| (Arc::clone(&r.store), Arc::clone(&r.lane)))
            .ok_or_else(|| internal(format!("object store {id} is not registered")))
    }

    /// A context for one query, with its own counters and no secret
    /// resolver.
    #[must_use]
    pub fn context(self: &Arc<Self>) -> IoContext {
        IoContext {
            sched: Arc::clone(self),
            counters: Arc::new(IoCounters::default()),
            secrets: Arc::new(CachedSecrets::new(Arc::new(NoSecrets))),
            memory: None,
        }
    }

    /// Accounts the metadata cache in a child of `pool` (see
    /// [`MetadataCache::attach_pool`]).
    pub fn attach_memory(&self, pool: &Arc<dyn MemoryPool>) {
        self.cache.attach_pool(pool);
    }

    /// The metadata cache.
    #[must_use]
    pub fn cache(&self) -> &MetadataCache {
        &self.cache
    }

    /// Process-wide counters.
    #[must_use]
    pub fn totals(&self) -> &IoCounters {
        &self.totals
    }

    /// The settings.
    #[must_use]
    pub fn config(&self) -> &IoConfig {
        &self.config
    }

    /// Full-jitter backoff before retry `attempt` (0-based), at least
    /// `retry_after`.
    fn backoff(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        let cap = self
            .config
            .retry_base
            .saturating_mul(1 << attempt.min(16))
            .min(self.config.retry_max);
        let r = splitmix(
            self.jitter
                .fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed),
        );
        let nanos = u64::try_from(cap.as_nanos()).unwrap_or(u64::MAX);
        let jittered = Duration::from_nanos(if nanos == 0 { 0 } else { r % (nanos + 1) });
        retry_after.map_or(jittered, |ra| {
            jittered.max(ra.min(Duration::from_secs(300)))
        })
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Why one attempt failed.
enum Attempt {
    Retry(Option<Duration>),
    Fail(krishiv_common::error::KrishivError),
}

/// Classifies a store error: transient failures are retried.
fn classify(e: &object_store::Error, file: &FileRef) -> Attempt {
    let path = file.display_path();
    match e {
        object_store::Error::NotFound { .. } => Attempt::Fail(not_found(&path)),
        object_store::Error::Precondition { .. } | object_store::Error::NotModified { .. } => {
            Attempt::Fail(changed(&path, &file.version.to_string()))
        }
        object_store::Error::Generic { .. } => match transient(e) {
            Some(retry_after) => Attempt::Retry(retry_after),
            None => Attempt::Fail(store_failed(&path, 1, &e.to_string())),
        },
        other => Attempt::Fail(store_failed(&path, 1, &other.to_string())),
    }
}

/// `Some(retry_after)` when `e` is worth retrying: a [`TransientError`] or
/// a timed-out / reset I/O error anywhere in its source chain.
fn transient(e: &object_store::Error) -> Option<Option<Duration>> {
    let object_store::Error::Generic { source, .. } = e else {
        return None;
    };
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(source.as_ref());
    while let Some(err) = cur {
        if let Some(t) = err.downcast_ref::<TransientError>() {
            return Some(t.retry_after);
        }
        if let Some(io) = err.downcast_ref::<std::io::Error>()
            && matches!(
                io.kind(),
                std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
            )
        {
            return Some(None);
        }
        cur = err.source();
    }
    None
}

impl ScheduledRequest {
    /// Request extensions carrying the marker.
    pub(crate) fn into_extensions(self) -> object_store::Extensions {
        let mut e = object_store::Extensions::new();
        e.insert(self);
        e
    }
}

/// A query's handle on the scheduler (cheap to clone).
#[derive(Clone)]
pub struct IoContext {
    sched: Arc<IoScheduler>,
    pub(crate) counters: Arc<IoCounters>,
    secrets: Arc<CachedSecrets>,
    /// The query's consumer for metadata reads ([`Held`] bytes).
    memory: Option<Arc<ConsumerHandle>>,
}

/// Secrets resolved once per query: a later request for the same reference
/// costs no provider round trip. The plaintext stays in [`SecretString`]s
/// (zeroed on drop) and dies with the query's context.
#[derive(Debug)]
struct CachedSecrets {
    inner: Arc<dyn SecretResolver>,
    resolved: std::sync::Mutex<HashMap<(String, String), SecretString>>,
}

impl CachedSecrets {
    fn new(inner: Arc<dyn SecretResolver>) -> Self {
        Self {
            inner,
            resolved: std::sync::Mutex::new(HashMap::new()),
        }
    }

    fn cached(&self, scope: &str, key: &str) -> Option<SecretString> {
        self.resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&(scope.to_owned(), key.to_owned()))
            .cloned()
    }
}

impl SecretResolver for CachedSecrets {
    fn resolve<'a>(&'a self, scope: &'a str, key: &'a str) -> BoxFuture<'a, Result<SecretString>> {
        Box::pin(async move {
            if let Some(s) = self.cached(scope, key) {
                return Ok(s);
            }
            let s = self.inner.resolve(scope, key).await?;
            self.resolved
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert((scope.to_owned(), key.to_owned()), s.clone());
            Ok(s)
        })
    }
}

/// Bytes read for file metadata, with the memory they hold reserved from
/// the query's pool (when the context has one, [`IoContext::with_memory`]);
/// released when dropped. Data reads are reserved by the scan instead.
#[derive(Debug)]
pub struct Held<T> {
    value: T,
    _reservation: Option<Reservation>,
}

impl<T> std::ops::Deref for Held<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

/// A speculative footer read: the offset of its first byte and the bytes.
#[derive(Debug, Clone)]
pub struct Tail {
    /// Offset of the first byte.
    pub start: u64,
    /// The bytes, up to the end of the file.
    pub bytes: Bytes,
}

impl fmt::Debug for IoContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoContext")
            .field("counters", &self.counters)
            .finish_non_exhaustive()
    }
}

impl IoContext {
    /// This context resolving secret references with `secrets` (each
    /// reference once per context).
    #[must_use]
    pub fn with_secrets(mut self, secrets: Arc<dyn SecretResolver>) -> Self {
        self.secrets = Arc::new(CachedSecrets::new(secrets));
        self
    }

    /// This context reserving the bytes of metadata reads ([`Self::get_tail`],
    /// [`Self::get_ranges_held`]) from `pool`, the query's pool.
    #[must_use]
    pub fn with_memory(mut self, pool: Arc<dyn MemoryPool>) -> Self {
        self.memory = Some(ConsumerHandle::register(pool, "file-metadata-reads", None));
        self
    }

    async fn reserve(&self, bytes: u64) -> Result<Option<Reservation>> {
        let Some(h) = &self.memory else {
            return Ok(None);
        };
        let mut r = h.reservation();
        r.grow(usize::try_from(bytes).unwrap_or(usize::MAX)).await?;
        Ok(Some(r))
    }

    /// [`Self::get_ranges`] for metadata (footers, page indexes, bloom
    /// filters): the bytes are reserved (with waiting) before the request
    /// and released when the result is dropped.
    ///
    /// # Errors
    /// As [`Self::get_ranges`], and memory errors.
    pub async fn get_ranges_held(
        &self,
        file: &FileRef,
        ranges: &[Range<u64>],
    ) -> Result<Held<Vec<Bytes>>> {
        let bytes = ranges.iter().map(|r| r.end.saturating_sub(r.start)).sum();
        let reservation = self.reserve(bytes).await?;
        let value = self.get_ranges(file, ranges).await?;
        Ok(Held {
            value,
            _reservation: reservation,
        })
    }

    /// The query's counters.
    #[must_use]
    pub fn counters(&self) -> &IoCounters {
        &self.counters
    }

    /// The secret resolver.
    #[must_use]
    pub fn secrets(&self) -> &dyn SecretResolver {
        self.secrets.as_ref()
    }

    /// The scheduler's settings.
    #[must_use]
    pub fn config(&self) -> &IoConfig {
        &self.sched.config
    }

    /// The scheduler.
    #[must_use]
    pub fn scheduler(&self) -> &Arc<IoScheduler> {
        &self.sched
    }

    pub(crate) fn count(&self, f: impl Fn(&IoCounters) -> &AtomicU64, n: u64) {
        f(&self.counters).fetch_add(n, Ordering::Relaxed);
        f(&self.sched.totals).fetch_add(n, Ordering::Relaxed);
    }

    /// The bytes of each of `ranges` of `file` (absolute offsets), fetched
    /// with coalesced, version-checked, retried requests.
    ///
    /// # Errors
    /// `FAILED_READ_FILE.NO_HINT` for a range beyond the file or a store
    /// failure after retries; `FAILED_READ_FILE.FILE_NOT_EXIST`;
    /// `KRISHIV_FILE_CHANGED_DURING_READ`.
    pub async fn get_ranges(&self, file: &FileRef, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        if let Some(bad) = ranges.iter().find(|r| r.end > file.size || r.start > r.end) {
            return Err(read_failed(
                &file.display_path(),
                &format!("range {bad:?} is outside the {}-byte file", file.size),
            ));
        }
        let cfg = &self.sched.config;
        // Empty ranges need no bytes (and must never become a whole-object GET).
        let wanted: Vec<Range<u64>> = ranges.iter().filter(|r| r.start < r.end).cloned().collect();
        let (requests, owner) = coalesce(&wanted, cfg.coalesce_bytes, cfg.max_request_bytes);
        let fetched = try_join_all(requests.iter().map(|r| self.get_one(file, r.clone()))).await?;
        let mut owner = owner.into_iter();
        ranges
            .iter()
            .map(|r| {
                if r.start == r.end {
                    return Ok(Bytes::new());
                }
                let i = owner
                    .next()
                    .ok_or_else(|| internal("range without a request"))?;
                let req = &requests[i];
                let (a, b) = (r.start - req.start, r.end - req.start);
                usize::try_from(a)
                    .ok()
                    .zip(usize::try_from(b).ok())
                    .map(|(a, b)| fetched[i].slice(a..b))
                    .ok_or_else(|| internal("range offset overflows usize"))
            })
            .collect()
    }

    /// Like [`Self::get_ranges`], but owned: runs on the I/O runtime when
    /// the scheduler has one (so it progresses while the caller decodes),
    /// and is cancelled when dropped.
    pub fn fetch(
        &self,
        file: FileRef,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
        let ctx = self.clone();
        let fut = async move { ctx.get_ranges(&file, &ranges).await };
        match &self.sched.runtime {
            None => Box::pin(fut),
            Some(rt) => spawned(rt, fut),
        }
    }

    /// The last `len` bytes of `file` (at most [`MAX_TAIL_BYTES`], at least
    /// the configured `krishiv.io.footerReadBytes`, at most the file) in one
    /// request: the speculative footer read, reserved like
    /// [`Self::get_ranges_held`].
    ///
    /// # Errors
    /// As [`Self::get_ranges`], and memory errors.
    pub async fn get_tail(&self, file: &FileRef, len: u64) -> Result<Held<Tail>> {
        let len = len
            .max(self.sched.config.footer_read_bytes)
            .min(MAX_TAIL_BYTES)
            .min(file.size);
        let start = file.size - len;
        let reservation = self.reserve(len).await?;
        self.count(|c| &c.tail_reads, 1);
        let bytes = self.get_one(file, start..file.size).await?;
        Ok(Held {
            value: Tail { start, bytes },
            _reservation: reservation,
        })
    }

    async fn get_one(&self, file: &FileRef, range: Range<u64>) -> Result<Bytes> {
        if range.start >= range.end {
            return Ok(Bytes::new());
        }
        let (store, lane) = self.sched.store(&file.store)?;
        let cfg = &self.sched.config;
        let mut attempt = 0;
        loop {
            let outcome = {
                let _permit = lane.admit(Direction::Read, &file.path).await?;
                self.count(|c| &c.requests, 1);
                let mut opts = GetOptions {
                    range: Some(GetRange::Bounded(range.clone())),
                    extensions: ScheduledRequest.into_extensions(),
                    ..GetOptions::default()
                };
                match &file.version {
                    VersionToken::ETag(e) => opts.if_match = Some(e.to_string()),
                    VersionToken::Modified(us) => {
                        // The token is truncated to microseconds; allow the
                        // rest of that microsecond (stores keep nanoseconds).
                        opts.if_unmodified_since = us
                            .checked_mul(1000)
                            .and_then(|ns| ns.checked_add(999))
                            .map(chrono::DateTime::from_timestamp_nanos);
                    }
                    VersionToken::Immutable => {}
                }
                let request = async {
                    let r = store.get_opts(&file.path, opts).await?;
                    let size = r.meta.size;
                    r.bytes().await.map(|b| (size, b))
                };
                tokio::time::timeout(cfg.request_timeout, request).await
            };
            let (retry_after, reason) = match outcome {
                Ok(Ok((size, bytes))) => {
                    if size != file.size {
                        return Err(changed(&file.display_path(), &file.version.to_string()));
                    }
                    if bytes.len() as u64 != range.end - range.start {
                        return Err(read_failed(&file.display_path(), "short read"));
                    }
                    self.count(|c| &c.bytes_read, bytes.len() as u64);
                    return Ok(bytes);
                }
                Ok(Err(e)) => match classify(&e, file) {
                    Attempt::Fail(err) => return Err(err),
                    Attempt::Retry(ra) => (ra, e.to_string()),
                },
                Err(_elapsed) => (None, "the request timed out".to_owned()),
            };
            if attempt >= cfg.max_retries {
                return Err(store_failed(&file.display_path(), attempt + 1, &reason));
            }
            self.count(|c| &c.retries, 1);
            tokio::time::sleep(self.sched.backoff(attempt, retry_after)).await;
            attempt += 1;
        }
    }

    /// `file`'s metadata as read by `format`, from the cache when an entry
    /// loaded with at least `needs` exists; otherwise read (with the union
    /// of both needs) and cached. The cache key includes the fingerprint of
    /// any metadata-decoding key in `options`.
    ///
    /// # Errors
    /// What [`FileFormat::read_metadata`] reports, with the file's path.
    pub async fn read_metadata<F: FileFormat + ?Sized>(
        &self,
        format: &F,
        file: &FileRef,
        needs: MetadataNeeds,
        options: &FormatOptions,
    ) -> Result<Arc<dyn FileMetadata>> {
        let fingerprint = options.metadata_fingerprint(self.secrets()).await?;
        let key = CacheKey::new(format.name(), file, fingerprint);
        let cache = &self.sched.cache;
        if let Ok(m) = cache.get(&key, needs) {
            self.count(|c| &c.cache_hits, 1);
            return Ok(m);
        }
        // Single flight: wait for a read of this key in progress and take
        // its outcome, or read it ourselves while holding the slot.
        let slot = cache.load_slot(&key);
        let seen = slot.completed.load(Ordering::Acquire);
        let mut loaded = slot.loaded.lock().await;
        // Reads from here on start in the current generation; a read from
        // before an invalidation is not reused.
        let generation = cache.generation();
        let last = loaded.as_ref().filter(|l| l.generation == generation);
        match last.map(|l| (&l.outcome, l.completion)) {
            Some((Ok(m), _)) if m.needs().covers(&needs) => {
                self.count(|c| &c.cache_hits, 1);
                return Ok(Arc::clone(m));
            }
            // A read that ended while we waited failed: fail with it.
            Some((Err(e), completion)) if completion > seen => return Err(e.clone()),
            _ => {}
        }
        let previous = last.and_then(|l| l.outcome.as_ref().ok().map(|m| m.needs()));
        let needs = match cache.get(&key, needs) {
            Ok(m) => {
                self.count(|c| &c.cache_hits, 1);
                return Ok(m);
            }
            Err(stale) => stale.or(previous).map_or(needs, |s| s.union(needs)),
        };
        self.count(|c| &c.cache_misses, 1);
        let outcome = format
            .read_metadata(self, file, needs, options)
            .await
            .map_err(|e| e.into_error(&file.display_path()));
        if let Ok(meta) = &outcome {
            cache.insert_read_in(generation, key, Arc::clone(meta));
        }
        let completion = slot.completed.fetch_add(1, Ordering::AcqRel) + 1;
        *loaded = Some(cache::Loaded {
            generation,
            completion,
            outcome: outcome.clone(),
        });
        outcome
    }

    /// [`Self::read_metadata`] as an owned future that runs on the I/O
    /// runtime when the scheduler has one: a scan starts reading the next
    /// file's footer while it decodes the current file. The result lands in
    /// the cache (or is shared with a concurrent read of the same file), so
    /// the scan's own `read_metadata` then costs no request. Dropping the
    /// future cancels the read.
    pub fn prefetch_metadata(
        &self,
        format: Arc<dyn FileFormat>,
        file: FileRef,
        needs: MetadataNeeds,
        options: FormatOptions,
    ) -> BoxFuture<'static, Result<Arc<dyn FileMetadata>>> {
        let ctx = self.clone();
        let fut = async move {
            ctx.read_metadata(format.as_ref(), &file, needs, &options)
                .await
        };
        match &self.sched.runtime {
            None => Box::pin(fut),
            Some(rt) => spawned(rt, fut),
        }
    }

    /// A [`ReadAt`] over `file` through this context (for formats whose
    /// metadata reader is an async library).
    #[must_use]
    pub fn reader(&self, file: &FileRef) -> IoReadAt {
        IoReadAt {
            ctx: self.clone(),
            file: file.clone(),
        }
    }

    /// Starts an upload of `path` in `store`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the store is not registered.
    pub fn upload(&self, store: &StoreId, path: Path) -> Result<Upload> {
        let (s, lane) = self.sched.store(store)?;
        let display = format!("{store}/{path}");
        Ok(Upload::new(self.clone(), s, lane, path, display))
    }

    async fn put(
        &self,
        store: &Arc<dyn ObjectStore>,
        lane: &Lane,
        path: &Path,
        display: &str,
        payload: PutPayload,
    ) -> Result<()> {
        let cfg = &self.sched.config;
        let len = payload.content_length() as u64;
        let mut attempt = 0;
        loop {
            let r = {
                let _permit = lane.admit(Direction::Write, path).await?;
                self.count(|c| &c.requests, 1);
                let opts = PutOptions {
                    extensions: ScheduledRequest.into_extensions(),
                    ..PutOptions::default()
                };
                tokio::time::timeout(
                    cfg.request_timeout,
                    store.put_opts(path, payload.clone(), opts),
                )
                .await
            };
            let retry_after = match r {
                Ok(Ok(_)) => {
                    self.count(|c| &c.bytes_written, len);
                    return Ok(());
                }
                Ok(Err(e)) => match transient(&e) {
                    Some(retry_after) => retry_after,
                    None => return Err(write_failed(display, &e.to_string())),
                },
                Err(_elapsed) => None,
            };
            if attempt >= cfg.max_retries {
                return Err(write_failed(display, "gave up after retries"));
            }
            self.count(|c| &c.retries, 1);
            tokio::time::sleep(self.sched.backoff(attempt, retry_after)).await;
            attempt += 1;
        }
    }

    /// Deletes `path` in `store` (a staged file of an aborted write).
    ///
    /// # Errors
    /// `TASK_WRITE_FAILED` if the store refuses.
    pub async fn delete(&self, store: &StoreId, path: &Path) -> Result<()> {
        let (s, lane) = self.sched.store(store)?;
        let _permit = lane.admit(Direction::Write, path).await?;
        self.count(|c| &c.requests, 1);
        let mut deleted = s.delete_stream(Box::pin(futures::stream::once(std::future::ready(Ok(
            path.clone(),
        )))));
        match futures::StreamExt::next(&mut deleted).await {
            None | Some(Ok(_) | Err(object_store::Error::NotFound { .. })) => Ok(()),
            Some(Err(e)) => Err(write_failed(&format!("{store}/{path}"), &e.to_string())),
        }
    }
}

/// A [`ReadAt`] through an [`IoContext`].
#[derive(Debug, Clone)]
pub struct IoReadAt {
    ctx: IoContext,
    file: FileRef,
}

impl ReadAt for IoReadAt {
    fn size(&self) -> u64 {
        self.file.size
    }

    fn read_at(&self, range: Range<u64>) -> BoxFuture<'_, crate::FormatResult<Bytes>> {
        Box::pin(async move {
            let mut v = self.ctx.get_ranges(&self.file, &[range]).await?;
            v.pop()
                .ok_or_else(|| crate::FormatError::Other(internal("no bytes for a range")))
        })
    }
}

/// `fut` running as a task on `rt`, cancelled when the returned future is
/// dropped.
pub(crate) fn spawned<T: Send + 'static>(
    rt: &tokio::runtime::Handle,
    fut: impl Future<Output = Result<T>> + Send + 'static,
) -> BoxFuture<'static, Result<T>> {
    let mut handle = AbortOnDrop(rt.spawn(fut));
    Box::pin(async move {
        match (&mut handle.0).await {
            Ok(r) => r,
            Err(e) if e.is_cancelled() => Err(cancelled()),
            Err(e) => Err(internal(format!("I/O task failed: {e}"))),
        }
    })
}

/// A spawned task aborted when dropped.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests;
