// SPDX-License-Identifier: Apache-2.0
//! The rows of `system.caches`: the file-metadata cache and the session's
//! relation cache and the session's checkpoints (per tier: memory and disk bytes, registrations waiting
//! for data, spills and evictions under memory pressure; design P3-T083 §3.6).

use arrow_array::ArrayRef;

use super::{LOCAL_WORKER, Sources, long, text};

/// One row of `system.caches`.
struct CacheRow {
    cache: &'static str,
    entries: i64,
    bytes: i64,
    capacity: Option<i64>,
    hits: i64,
    misses: i64,
    evictions: i64,
    invalidations: i64,
    loads_in_flight: i64,
    memory_bytes: i64,
    disk_bytes: i64,
    entries_pending: i64,
    spills: i64,
}

/// The columns of `system.caches`, in schema order.
pub(super) fn columns(sources: &Sources) -> Vec<ArrayRef> {
    let n = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
    let meta = sources.io.cache().stats();
    let io = sources.io.totals();
    let relation = sources.relations.stats();
    let checkpoints = sources.relations.checkpoints.counts();
    let rows = [
        CacheRow {
            cache: "file_metadata",
            entries: n(meta.entries),
            bytes: n(meta.bytes),
            capacity: Some(n(meta.limit)),
            hits: n(krishiv_io::IoCounters::get(&io.cache_hits)),
            misses: n(krishiv_io::IoCounters::get(&io.cache_misses)),
            evictions: n(meta.evictions),
            invalidations: n(meta.invalidations),
            loads_in_flight: n(meta.loads_in_flight),
            memory_bytes: n(meta.bytes),
            disk_bytes: 0,
            entries_pending: 0,
            spills: 0,
        },
        CacheRow {
            cache: "relation",
            entries: n(relation.entries),
            bytes: n(relation.memory_bytes.saturating_add(relation.disk_bytes)),
            capacity: None,
            hits: n(relation.hits),
            misses: n(relation.misses),
            evictions: n(relation.evictions),
            invalidations: n(relation.invalidations),
            loads_in_flight: 0,
            memory_bytes: n(relation.memory_bytes),
            disk_bytes: n(relation.disk_bytes),
            entries_pending: n(relation.entries_pending),
            spills: n(relation.spills),
        },
        CacheRow {
            cache: "checkpoint",
            entries: n(checkpoints.live),
            bytes: n(checkpoints
                .memory_bytes
                .saturating_add(checkpoints.disk_bytes)),
            capacity: None,
            hits: 0,
            misses: 0,
            evictions: 0,
            invalidations: 0,
            loads_in_flight: 0,
            memory_bytes: n(checkpoints.memory_bytes),
            disk_bytes: n(checkpoints.disk_bytes),
            entries_pending: 0,
            spills: 0,
        },
    ];
    let l = |f: fn(&CacheRow) -> i64| long(rows.iter().map(move |r| Some(f(r))));
    vec![
        text(rows.iter().map(|_| Some(LOCAL_WORKER))),
        text(rows.iter().map(|r| Some(r.cache))),
        l(|r| r.entries),
        l(|r| r.bytes),
        long(rows.iter().map(|r| r.capacity)),
        l(|r| r.hits),
        l(|r| r.misses),
        l(|r| r.evictions),
        l(|r| r.invalidations),
        l(|r| r.loads_in_flight),
        l(|r| r.memory_bytes),
        l(|r| r.disk_bytes),
        l(|r| r.entries_pending),
        l(|r| r.spills),
    ]
}
