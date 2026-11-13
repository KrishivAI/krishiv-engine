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
//! again. A hit first checks the cached last commit file with one `head` (its size and
//! modification time): a table removed or recreated at the root is built afresh, and so is any
//! entry whose refresh fails (the entry is dropped). The query then pins the version it got for all its scans; the pinning itself is the
//! session's (one table per root per query). Authorization runs before the lookup, in the
//! session's analysis.
//!
//! The cache holds at most `capacity` tables and at most `max_bytes` of them, estimated
//! ([`ENTRY_BYTES`] per table plus [`listing_bytes`] of its live files;
//! `krishiv.delta.snapshotCache.maxBytes`, 64 MiB by default, ADR 0037); the least recently
//! used one leaves first. A table whose listing alone does not fit keeps only its snapshot
//! (each query lists its files again). Time travel (a given version) neither reads nor fills it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use delta_kernel::SnapshotRef;
use krishiv_common::config::{Config, ConfigValue, DELTA_SNAPSHOT_CACHE_MAX_BYTES};
use krishiv_io::StoreId;

use super::ScanUnit;

/// The tables a cache holds by default.
pub const DEFAULT_CAPACITY: usize = 256;

/// The bytes a cache holds by default (`krishiv.delta.snapshotCache.maxBytes`).
pub const DEFAULT_MAX_BYTES: u64 = 64 << 20;

/// The estimated bytes of one cached table besides its listing: the snapshot (its log segment,
/// protocol, metadata and schema) and the entry itself.
pub const ENTRY_BYTES: u64 = 16 << 10;

/// The estimated bytes of a table's live-file listing: each unit's fixed size, its path and its
/// partition values (with a map entry's overhead each).
#[must_use]
pub fn listing_bytes(units: &[ScanUnit]) -> u64 {
    const MAP_ENTRY: usize = 2 * std::mem::size_of::<String>() + 16;
    let bytes: usize = units
        .iter()
        .map(|u| {
            std::mem::size_of::<ScanUnit>()
                + u.path.len()
                + u.partition_values
                    .iter()
                    .map(|(k, v)| MAP_ENTRY + k.len() + v.len())
                    .sum::<usize>()
        })
        .sum();
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

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

/// A commit file as the store last reported it: what a hit checks with one `head` that the
/// table at the root is still the one cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommitStamp {
    pub(crate) size: u64,
    /// Milliseconds since the epoch.
    pub(crate) last_modified: i64,
}

/// One cached table: its latest snapshot, that snapshot's live files (unless they were too
/// large to keep) and its last commit file.
#[derive(Debug, Clone)]
pub(crate) struct Cached {
    pub(crate) snapshot: SnapshotRef,
    pub(crate) units: Option<Arc<[ScanUnit]>>,
    pub(crate) last: CommitStamp,
}

impl Cached {
    /// Its estimated bytes.
    fn bytes(&self) -> u64 {
        ENTRY_BYTES.saturating_add(self.units.as_deref().map_or(0, listing_bytes))
    }
}

/// The entries, most recently used last, and their estimated bytes.
#[derive(Debug, Default)]
struct Entries {
    list: VecDeque<(CacheKey, Cached, u64)>,
    bytes: u64,
}

impl Entries {
    fn remove(&mut self, i: usize) -> Option<(CacheKey, Cached, u64)> {
        let entry = self.list.remove(i)?;
        self.bytes = self.bytes.saturating_sub(entry.2);
        Some(entry)
    }

    fn retain(&mut self, keep: impl Fn(&CacheKey) -> bool) {
        self.list.retain(|(k, _, _)| keep(k));
        self.bytes = self.list.iter().map(|(_, _, b)| *b).sum();
    }
}

/// The cross-query snapshot cache (see the module docs).
#[derive(Debug)]
pub struct SnapshotCache {
    capacity: usize,
    max_bytes: u64,
    entries: Mutex<Entries>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl Default for SnapshotCache {
    fn default() -> Self {
        Self::bounded(DEFAULT_CAPACITY, DEFAULT_MAX_BYTES)
    }
}

impl SnapshotCache {
    /// A cache of at most `capacity` tables (at least one) and [`DEFAULT_MAX_BYTES`].
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::bounded(capacity, DEFAULT_MAX_BYTES)
    }

    /// A cache of at most `capacity` tables (at least one) and `max_bytes`, estimated.
    #[must_use]
    pub fn bounded(capacity: usize, max_bytes: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            max_bytes,
            entries: Mutex::new(Entries::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// The cache `config` sizes (`krishiv.delta.snapshotCache.maxBytes`), of
    /// [`DEFAULT_CAPACITY`] tables.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let max_bytes = config
            .get(&DELTA_SNAPSHOT_CACHE_MAX_BYTES)
            .and_then(ConfigValue::as_size)
            .unwrap_or(DEFAULT_MAX_BYTES);
        Self::bounded(DEFAULT_CAPACITY, max_bytes)
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
        self.lock().list.len()
    }

    /// Whether it holds no table.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().list.is_empty()
    }

    /// The estimated bytes held now.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.lock().bytes
    }

    /// The version cached for `key`, if any (no hit or miss is counted).
    #[must_use]
    pub fn cached_version(&self, key: &CacheKey) -> Option<u64> {
        self.lock()
            .list
            .iter()
            .find(|(k, _, _)| k == key)
            .map(|(_, c, _)| c.snapshot.version())
    }

    /// Whether the entry of `key` keeps its file listing (no hit or miss is counted).
    #[must_use]
    pub fn keeps_listing(&self, key: &CacheKey) -> bool {
        self.lock()
            .list
            .iter()
            .any(|(k, c, _)| k == key && c.units.is_some())
    }

    /// Drops the entry of `key`, if any.
    pub fn invalidate(&self, key: &CacheKey) {
        self.lock().retain(|k| k != key);
    }

    /// Drops the entry of `key` only while it is still the one a query found stale (its
    /// `version` and last commit `last`): an entry another query stored meanwhile stays.
    pub(crate) fn invalidate_if(&self, key: &CacheKey, version: u64, last: CommitStamp) {
        let mut entries = self.lock();
        entries
            .list
            .retain(|(k, c, _)| !(k == key && c.snapshot.version() == version && c.last == last));
        entries.bytes = entries.list.iter().map(|(_, _, b)| *b).sum();
    }

    /// Drops every entry of the table root `root` (any store or credential).
    pub fn invalidate_root(&self, root: &str) {
        self.lock().retain(|k| k.root != root);
    }

    /// The entry of `key`, counted as a hit or a miss.
    pub(crate) fn lookup(&self, key: &CacheKey) -> Option<Cached> {
        let mut entries = self.lock();
        let found = entries.list.iter().position(|(k, _, _)| k == key);
        let Some(i) = found else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.hits.fetch_add(1, Ordering::Relaxed);
        let entry = entries.list.remove(i)?;
        let cached = entry.1.clone();
        entries.list.push_back(entry);
        Some(cached)
    }

    /// Keeps `cached` for `key`, unless a newer version is already cached (two queries
    /// refreshing at once). A listing that alone exceeds the byte bound is not kept (the
    /// snapshot is); least recently used entries leave until the new one fits.
    pub(crate) fn store(&self, key: CacheKey, mut cached: Cached) {
        let mut bytes = cached.bytes();
        if bytes > self.max_bytes {
            cached.units = None;
            bytes = cached.bytes();
        }
        if bytes > self.max_bytes {
            return;
        }
        let mut entries = self.lock();
        if let Some(i) = entries.list.iter().position(|(k, _, _)| *k == key) {
            let newer = entries
                .list
                .get(i)
                .is_some_and(|(_, c, _)| c.snapshot.version() > cached.snapshot.version());
            if newer {
                return;
            }
            entries.remove(i);
        }
        while !entries.list.is_empty()
            && (entries.list.len() >= self.capacity
                || entries.bytes.saturating_add(bytes) > self.max_bytes)
        {
            entries.remove(0);
        }
        entries.bytes = entries.bytes.saturating_add(bytes);
        entries.list.push_back((key, cached, bytes));
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Entries> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
