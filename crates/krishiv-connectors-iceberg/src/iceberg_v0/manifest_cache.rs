// SPDX-License-Identifier: Apache-2.0
//! The decoded-manifest cache (PR-25d; design note §2.3): the live entries of
//! a manifest after inheritance, as an `Arc<[CachedEntry]>` (each entry an
//! `Arc`, so a plan shares them instead of copying), least recently used
//! out under a byte limit (`krishiv.iceberg.manifestCache.bytes`; 0 disables
//! it). In memory only: the workspace has no Foyer tier yet.
//!
//! - **Scope**: one cache per storage registration (`Registered`), shared by
//!   every table opened through it, so a process that keeps its registration
//!   plans a snapshot's manifests once. There is no process-global cache.
//! - **Key**: the manifest's path and length (manifests are immutable), plus
//!   what inheritance reads from the manifest-list entry (sequence number,
//!   added snapshot id, first row id), so an entry is the same whichever
//!   snapshot lists the manifest.
//! - **Size**: an estimate of the decoded entries' heap bytes; the limit
//!   bounds the sum. A manifest larger than the limit is not kept. The limit
//!   is the inserting table's (`krishiv.iceberg.manifestCache.bytes` is a
//!   startup key, so in practice one value per process).
//! - **Parse limit**: a hit is checked against the reading table's
//!   `krishiv.catalog.maxMetadataBytes` with the manifest's size when it was
//!   read, as a miss is (a lowered limit is not bypassed).

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, PoisonError};

use crate::plan::{ContentFile, DataFileEntry, DeleteFileEntry, LiveEntry, ManifestFileEntry};

/// A live entry of a cached manifest, shared by every plan that reads it.
#[derive(Debug, Clone)]
pub(crate) enum CachedEntry {
    Data(Arc<DataFileEntry>),
    Delete(Arc<DeleteFileEntry>),
}

impl From<LiveEntry> for CachedEntry {
    fn from(e: LiveEntry) -> Self {
        match e {
            LiveEntry::Data(d) => Self::Data(Arc::new(d)),
            LiveEntry::Delete(d) => Self::Delete(Arc::new(d)),
        }
    }
}

/// A manifest's cached entries and the bytes read to decode them.
#[derive(Debug, Clone)]
pub(crate) struct Cached {
    pub(crate) entries: Arc<[CachedEntry]>,
    /// The manifest's size as read (checked against the parse limit on a
    /// hit).
    pub(crate) read_bytes: u64,
}

/// What identifies a manifest's decoded entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ManifestKey {
    path: String,
    length: i64,
    sequence_number: i64,
    added_snapshot_id: i64,
    first_row_id: Option<u64>,
}

impl ManifestKey {
    pub(crate) fn of(m: &ManifestFileEntry) -> Self {
        Self {
            path: m.path.clone(),
            length: m.length,
            sequence_number: m.sequence_number,
            added_snapshot_id: m.added_snapshot_id,
            first_row_id: m.first_row_id,
        }
    }
}

struct Slot {
    cached: Cached,
    bytes: usize,
    tick: u64,
}

#[derive(Default)]
struct Lru {
    slots: HashMap<ManifestKey, Slot>,
    /// Keys by last use, oldest first.
    order: BTreeMap<u64, ManifestKey>,
    bytes: usize,
    tick: u64,
}

/// Decoded manifests, least recently used out (see the module docs).
#[derive(Default)]
pub(crate) struct ManifestCache {
    lru: Mutex<Lru>,
}

impl std::fmt::Debug for ManifestCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lru = self.lru.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("ManifestCache")
            .field("manifests", &lru.slots.len())
            .field("bytes", &lru.bytes)
            .finish()
    }
}

impl ManifestCache {
    /// The entries of `key`, if cached (now the most recently used).
    pub(crate) fn get(&self, key: &ManifestKey) -> Option<Cached> {
        let mut lru = self.lru.lock().unwrap_or_else(PoisonError::into_inner);
        lru.tick += 1;
        let tick = lru.tick;
        let slot = lru.slots.get_mut(key)?;
        let old = std::mem::replace(&mut slot.tick, tick);
        let cached = slot.cached.clone();
        lru.order.remove(&old);
        lru.order.insert(tick, key.clone());
        Some(cached)
    }

    /// Keeps `entries` under `key`, evicting the least recently used until
    /// the cache holds at most `limit` bytes; nothing when they alone exceed
    /// it.
    pub(crate) fn insert(&self, key: ManifestKey, cached: Cached, limit: u64) {
        let bytes = entries_bytes(&cached.entries);
        let limit = usize::try_from(limit).unwrap_or(usize::MAX);
        if bytes > limit {
            return;
        }
        let mut lru = self.lru.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(old) = lru.slots.remove(&key) {
            lru.order.remove(&old.tick);
            lru.bytes -= old.bytes;
        }
        while lru.bytes + bytes > limit {
            let Some((_, oldest)) = lru.order.pop_first() else {
                break;
            };
            if let Some(s) = lru.slots.remove(&oldest) {
                lru.bytes -= s.bytes;
            }
        }
        lru.tick += 1;
        let tick = lru.tick;
        lru.order.insert(tick, key.clone());
        lru.bytes += bytes;
        lru.slots.insert(
            key,
            Slot {
                cached,
                bytes,
                tick,
            },
        );
    }

    /// The bytes held and the manifests cached.
    #[cfg(test)]
    pub(crate) fn usage(&self) -> (usize, usize) {
        let lru = self.lru.lock().unwrap_or_else(PoisonError::into_inner);
        (lru.bytes, lru.slots.len())
    }
}

/// The estimated heap bytes of decoded entries.
pub(crate) fn entries_bytes(entries: &[CachedEntry]) -> usize {
    entries
        .iter()
        .map(|e| {
            std::mem::size_of::<CachedEntry>()
                + std::mem::size_of::<DeleteFileEntry>()
                + file_bytes(match e {
                    CachedEntry::Data(d) => &d.file,
                    CachedEntry::Delete(d) => &d.entry.file,
                })
        })
        .fold(0, usize::saturating_add)
}

fn file_bytes(f: &ContentFile) -> usize {
    // A map entry: key, value and the node's share of overhead.
    const COUNT: usize = 32;
    const BOUND: usize = 48;
    f.path.len()
        + f.partition.len() * 32
        + (f.column_sizes.len() + f.value_counts.len() + f.null_counts.len() + f.nan_counts.len())
            * COUNT
        + f.lower_bounds
            .values()
            .chain(f.upper_bounds.values())
            .map(|b| BOUND + b.len())
            .sum::<usize>()
        + f.split_offsets.as_ref().map_or(0, |s| s.len() * 8)
        + f.equality_ids.as_ref().map_or(0, |s| s.len() * 4)
        + f.referenced_data_file.as_ref().map_or(0, String::len)
}

#[cfg(test)]
mod tests {
    use super::{Cached, ManifestCache, ManifestKey, entries_bytes};
    use crate::plan::tests::file;
    use crate::plan::{DataFileEntry, EntryStatus, LiveEntry};

    fn key(path: &str) -> ManifestKey {
        ManifestKey {
            path: path.to_owned(),
            length: 10,
            sequence_number: 1,
            added_snapshot_id: 1,
            first_row_id: None,
        }
    }

    fn entries(n: usize) -> Cached {
        let entries = (0..n)
            .map(|_| {
                LiveEntry::Data(DataFileEntry {
                    file: file(1),
                    status: EntryStatus::Added,
                    snapshot_id: 1,
                    data_sequence_number: Some(1),
                    file_sequence_number: Some(1),
                })
                .into()
            })
            .collect();
        Cached {
            entries,
            read_bytes: 1,
        }
    }

    /// The least recently used manifest goes first; a hit counts as a use;
    /// a manifest over the limit is not kept; a 0 limit keeps nothing.
    #[test]
    fn evicts_least_recently_used_under_the_limit() {
        let one = entries_bytes(&entries(1).entries);
        let cache = ManifestCache::default();
        let limit = u64::try_from(2 * one).unwrap();
        cache.insert(key("a"), entries(1), limit);
        cache.insert(key("b"), entries(1), limit);
        assert!(cache.get(&key("a")).is_some());
        cache.insert(key("c"), entries(1), limit);
        assert!(
            cache.get(&key("b")).is_none(),
            "b was the least recently used"
        );
        assert!(cache.get(&key("a")).is_some() && cache.get(&key("c")).is_some());
        assert_eq!(cache.usage(), (2 * one, 2));
        cache.insert(key("big"), entries(3), limit);
        assert!(cache.get(&key("big")).is_none());
        let off = ManifestCache::default();
        off.insert(key("a"), entries(1), 0);
        assert_eq!(off.usage(), (0, 0));
    }
}
