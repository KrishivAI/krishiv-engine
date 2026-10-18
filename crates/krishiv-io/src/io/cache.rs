// SPDX-License-Identifier: Apache-2.0
//! The file-metadata cache: decoded footers, byte-bounded LRU.
//!
//! Keyed by (format, store, path, size, version token) and, for metadata
//! that needs a key to decode (encrypted footers), a keyed fingerprint of
//! that key: another key never hits the entry and no key is stored. A
//! changed file has a new version token, so validation costs no request.
//! Distinct from the table-metadata cache of ADR 0037; `REFRESH` and
//! `CLEAR CACHE` clear it ([`MetadataCache::invalidate_path`],
//! [`MetadataCache::clear`]).
//!
//! **Single flight.** Concurrent misses on one key share one read: the
//! first reader holds the key's load slot while it reads, the others
//! wait for it and take its result (even when the entry was too large to
//! cache), or its error if it failed: readers queued behind a failed read
//! fail with it instead of retrying one after another (a reader arriving
//! after the failure reads again). A reader dropped mid-read releases the
//! slot to the next one, which reads.
//! An invalidation (`REFRESH`, `CLEAR CACHE`) starts a new generation: a
//! read begun before it neither fills the cache nor serves later readers.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use krishiv_common::error::KrishivError;

use krishiv_memory::{MemoryPool, Reservation};
use object_store::path::Path;

use crate::format::{FileMetadata, MetadataNeeds};
use crate::io::{FileRef, StoreId, VersionToken};

/// A metadata cache key.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    format: &'static str,
    store: StoreId,
    path: Path,
    size: u64,
    version: VersionToken,
    key_fingerprint: Option<[u8; 16]>,
}

impl CacheKey {
    /// The key of `file` read by `format`, with the fingerprint of the
    /// metadata-decoding key, if any.
    #[must_use]
    pub fn new(format: &'static str, file: &FileRef, key_fingerprint: Option<[u8; 16]>) -> Self {
        Self {
            format,
            store: file.store.clone(),
            path: file.path.clone(),
            size: file.size,
            version: file.version.clone(),
            key_fingerprint,
        }
    }
}

impl fmt::Debug for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CacheKey")
            .field("format", &self.format)
            .field("store", &self.store)
            .field("path", &self.path.as_ref())
            .field("size", &self.size)
            .field("version", &self.version)
            .field("keyed", &self.key_fingerprint.is_some())
            .finish()
    }
}

struct Entry {
    meta: Arc<dyn FileMetadata>,
    bytes: u64,
    tick: u64,
}

#[derive(Default)]
struct Lru {
    map: HashMap<CacheKey, Entry>,
    /// Recency order: tick → key.
    order: BTreeMap<u64, CacheKey>,
    bytes: u64,
    tick: u64,
    /// The cache's memory, reserved from its own pool when one is attached.
    reservation: Option<Reservation>,
    /// Bumped by every invalidation.
    generation: u64,
}

impl Lru {
    fn remove(&mut self, key: &CacheKey) {
        if let Some(e) = self.map.remove(key) {
            self.bytes -= e.bytes;
            self.order.remove(&e.tick);
            if let Some(r) = self.reservation.as_mut() {
                r.shrink(usize::try_from(e.bytes).unwrap_or(usize::MAX));
            }
        }
    }

    fn evict_oldest(&mut self) -> bool {
        let Some((_, victim)) = self.order.first_key_value().map(|(t, k)| (*t, k.clone())) else {
            return false;
        };
        self.remove(&victim);
        true
    }
}

/// What a finished read left for the readers waiting on its key: the cache
/// generation it began in, its completion number and its outcome.
pub(crate) struct Loaded {
    pub(crate) generation: u64,
    pub(crate) completion: u64,
    pub(crate) outcome: Result<Arc<dyn FileMetadata>, KrishivError>,
}

/// The readers of one key in flight: the reads completed so far and the
/// last outcome, behind the lock the reading reader holds.
#[derive(Default)]
pub(crate) struct LoadSlot {
    /// Reads completed (a reader notes this before it waits: a failure
    /// completed after that is shared with it).
    pub(crate) completed: AtomicU64,
    pub(crate) loaded: tokio::sync::Mutex<Option<Loaded>>,
}

/// The process-wide file-metadata cache (see the module docs).
pub struct MetadataCache {
    limit: u64,
    inner: Mutex<Lru>,
    /// Reads in progress, by key (single flight); entries die with their
    /// last reader.
    loads: Mutex<HashMap<CacheKey, Weak<LoadSlot>>>,
}

impl fmt::Debug for MetadataCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lru = self.lock();
        f.debug_struct("MetadataCache")
            .field("limit", &self.limit)
            .field("entries", &lru.map.len())
            .field("bytes", &lru.bytes)
            .finish()
    }
}

impl MetadataCache {
    /// A cache of at most `limit` bytes (by [`FileMetadata::heap_size`]);
    /// `0` disables it.
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            inner: Mutex::new(Lru::default()),
            loads: Mutex::new(HashMap::new()),
        }
    }

    /// The load slot of `key`, shared by every reader of it in flight.
    pub(crate) fn load_slot(&self, key: &CacheKey) -> Arc<LoadSlot> {
        let mut loads = self.loads.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = loads.get(key).and_then(Weak::upgrade) {
            return slot;
        }
        loads.retain(|_, w| w.strong_count() > 0);
        let slot = Arc::new(LoadSlot::default());
        loads.insert(key.clone(), Arc::downgrade(&slot));
        slot
    }

    /// The current generation (see the module docs).
    pub(crate) fn generation(&self) -> u64 {
        self.lock().generation
    }

    /// Keys with a read in flight (tests).
    #[must_use]
    pub fn loads_in_flight(&self) -> usize {
        self.loads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .filter(|w| w.strong_count() > 0)
            .count()
    }

    fn lock(&self) -> MutexGuard<'_, Lru> {
        // Critical sections only move map entries and cannot panic.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The entry for `key` if it was loaded with at least `needs`, and the
    /// needs of a stale entry otherwise (to reload with both).
    pub fn get(
        &self,
        key: &CacheKey,
        needs: MetadataNeeds,
    ) -> Result<Arc<dyn FileMetadata>, Option<MetadataNeeds>> {
        let mut lru = self.lock();
        lru.tick += 1;
        let tick = lru.tick;
        let Some(e) = lru.map.get_mut(key) else {
            return Err(None);
        };
        let loaded = e.meta.needs();
        if !loaded.covers(&needs) {
            return Err(Some(loaded));
        }
        let old = std::mem::replace(&mut e.tick, tick);
        let meta = Arc::clone(&e.meta);
        lru.order.remove(&old);
        lru.order.insert(tick, key.clone());
        Ok(meta)
    }

    /// Caches `meta` under `key`, evicting least recently used entries to
    /// stay within the limit. Metadata larger than the whole cache is not
    /// cached.
    pub fn insert(&self, key: CacheKey, meta: Arc<dyn FileMetadata>) {
        let generation = self.generation();
        self.insert_read_in(generation, key, meta);
    }

    /// [`Self::insert`] for metadata read in cache generation `generation`:
    /// not cached if the cache was invalidated since.
    pub(crate) fn insert_read_in(
        &self,
        generation: u64,
        key: CacheKey,
        meta: Arc<dyn FileMetadata>,
    ) {
        let bytes = meta.heap_size() as u64;
        if bytes > self.limit {
            return;
        }
        let mut lru = self.lock();
        if lru.generation != generation {
            return;
        }
        lru.tick += 1;
        let tick = lru.tick;
        lru.remove(&key);
        while lru.bytes + bytes > self.limit && lru.evict_oldest() {}
        // With a pool, the entry's memory must be granted too; evict more
        // while the pool refuses, and do not cache when nothing is left.
        let want = usize::try_from(bytes).unwrap_or(usize::MAX);
        loop {
            match lru.reservation.as_mut().map(|r| r.try_grow(want)) {
                None | Some(Ok(())) => break,
                Some(Err(_)) => {
                    if !lru.evict_oldest() {
                        return;
                    }
                }
            }
        }
        lru.bytes += bytes;
        lru.order.insert(tick, key.clone());
        lru.map.insert(key, Entry { meta, bytes, tick });
    }

    /// Drops every entry of `path` in `store` (`REFRESH`).
    pub fn invalidate_path(&self, store: &StoreId, path: &Path) {
        let mut lru = self.lock();
        lru.generation += 1;
        let victims: Vec<CacheKey> = lru
            .map
            .keys()
            .filter(|k| &k.store == store && &k.path == path)
            .cloned()
            .collect();
        for k in victims {
            lru.remove(&k);
        }
    }

    /// Drops every entry (`CLEAR CACHE`).
    pub fn clear(&self) {
        let mut lru = self.lock();
        lru.generation += 1;
        lru.map.clear();
        lru.order.clear();
        lru.bytes = 0;
        if let Some(r) = lru.reservation.as_mut() {
            let held = r.size();
            r.shrink(held);
        }
    }

    /// Accounts the cache's memory in its own child of `pool` (limited to
    /// the cache size): entries are cached only while the pool grants
    /// their bytes. Entries cached so far are dropped.
    pub fn attach_pool(&self, pool: &Arc<dyn MemoryPool>) {
        let limit = usize::try_from(self.limit).unwrap_or(usize::MAX);
        let child = pool.child("file-metadata-cache", Some(limit));
        let id = child.register("file-metadata-cache", None);
        self.clear();
        self.lock().reservation = Some(Reservation::new(child, id));
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().map.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.lock().bytes
    }
}
