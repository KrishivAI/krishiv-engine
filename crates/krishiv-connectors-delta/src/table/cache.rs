// SPDX-License-Identifier: Apache-2.0
//! [`SnapshotCache`], the cross-query cache of Delta snapshots (ADR 0037, design note P3-T021
//! section 3.4).
//!
//! An entry is one table's latest snapshot and its live files, keyed by [`CacheKey`]: the
//! normalized root URL, the store and the credential identity the snapshot was read under, so a
//! snapshot read with one principal's credentials is never served under another's. Delta's
//! freshness policy is `per_query`: every query starts from the cached snapshot and asks the
//! kernel for newer commits (`Snapshot::builder_from`, which lists the log after the cached
//! version only); an unchanged version keeps the cached file listing, a newer one is listed
//! again. The query then pins the version it got for all its scans; the pinning itself is the
//! session's (one table per root per query). Authorization runs before the lookup, in the
//! session's analysis.
//!
//! The cache holds at most `capacity` tables; the least recently used one leaves first. Time
//! travel (a given version) neither reads nor fills it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use delta_kernel::SnapshotRef;
use krishiv_io::StoreId;

use super::ScanUnit;

/// The tables a cache holds by default.
pub const DEFAULT_CAPACITY: usize = 256;

/// What identifies a cached snapshot (design note section 3.4).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// The table root URL, normalized (a trailing `/`).
    pub root: String,
    /// The store the table is read from (a cloud store's id carries its configuration's hash).
    pub store: StoreId,
    /// The credential identity the snapshot is read under (the session's principal).
    pub credential: String,
}

/// One cached table: its latest snapshot and that snapshot's live files.
#[derive(Debug, Clone)]
pub(crate) struct Cached {
    pub(crate) snapshot: SnapshotRef,
    pub(crate) units: Arc<[ScanUnit]>,
}

/// The cross-query snapshot cache (see the module docs).
#[derive(Debug)]
pub struct SnapshotCache {
    capacity: usize,
    /// Most recently used last.
    entries: Mutex<VecDeque<(CacheKey, Cached)>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl Default for SnapshotCache {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl SnapshotCache {
    /// A cache of at most `capacity` tables (at least one).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Mutex::new(VecDeque::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Lookups that found an entry.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Lookups that found none.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// The tables held now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether it holds no table.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// The version cached for `key`, if any (no hit or miss is counted).
    #[must_use]
    pub fn cached_version(&self, key: &CacheKey) -> Option<u64> {
        self.lock()
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, c)| c.snapshot.version())
    }

    /// Drops every entry of the table root `root` (any store or credential).
    pub fn invalidate_root(&self, root: &str) {
        self.lock().retain(|(k, _)| k.root != root);
    }

    /// The entry of `key`, counted as a hit or a miss.
    pub(crate) fn lookup(&self, key: &CacheKey) -> Option<Cached> {
        let mut entries = self.lock();
        let found = entries.iter().position(|(k, _)| k == key);
        let Some(i) = found else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.hits.fetch_add(1, Ordering::Relaxed);
        let entry = entries.remove(i)?;
        let cached = entry.1.clone();
        entries.push_back(entry);
        Some(cached)
    }

    /// Keeps `cached` for `key`, unless a newer version is already cached (two queries
    /// refreshing at once). The least recently used entry leaves when the cache is full.
    pub(crate) fn store(&self, key: CacheKey, cached: Cached) {
        let mut entries = self.lock();
        if let Some(i) = entries.iter().position(|(k, _)| *k == key) {
            let newer = entries
                .get(i)
                .is_some_and(|(_, c)| c.snapshot.version() > cached.snapshot.version());
            if newer {
                return;
            }
            entries.remove(i);
        }
        while entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back((key, cached));
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(CacheKey, Cached)>> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
