// SPDX-License-Identifier: Apache-2.0
//! What `system.caches` shows of a session's checkpoints (design P3-T083 §3.6): the snapshots alive
//! and the bytes they hold. A checkpoint is not a cache entry; it only reports through the cache's
//! row source.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The live checkpoints of one session.
#[derive(Debug, Default)]
pub(crate) struct CheckpointStats {
    live: AtomicU64,
    memory_bytes: AtomicU64,
    disk_bytes: AtomicU64,
}

/// A snapshot of [`CheckpointStats`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct CheckpointCounts {
    pub(crate) live: u64,
    pub(crate) memory_bytes: u64,
    pub(crate) disk_bytes: u64,
}

impl CheckpointStats {
    /// Counts one more snapshot of `memory_bytes` in memory and `disk_bytes` on disk; the guard
    /// takes it off again when the last frame that owns it is gone.
    pub(crate) fn add(self: &Arc<Self>, memory_bytes: u64, disk_bytes: u64) -> CheckpointGuard {
        self.live.fetch_add(1, Ordering::Relaxed);
        self.memory_bytes.fetch_add(memory_bytes, Ordering::Relaxed);
        self.disk_bytes.fetch_add(disk_bytes, Ordering::Relaxed);
        CheckpointGuard {
            stats: Arc::clone(self),
            memory_bytes,
            disk_bytes,
        }
    }

    /// The counts now.
    pub(crate) fn counts(&self) -> CheckpointCounts {
        CheckpointCounts {
            live: self.live.load(Ordering::Relaxed),
            memory_bytes: self.memory_bytes.load(Ordering::Relaxed),
            disk_bytes: self.disk_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Owned (through `DataFrame::keep`) by every frame of a snapshot; the snapshot is uncounted when
/// the last one drops.
#[derive(Debug)]
pub(crate) struct CheckpointGuard {
    stats: Arc<CheckpointStats>,
    memory_bytes: u64,
    disk_bytes: u64,
}

impl Drop for CheckpointGuard {
    fn drop(&mut self) {
        self.stats.live.fetch_sub(1, Ordering::Relaxed);
        self.stats
            .memory_bytes
            .fetch_sub(self.memory_bytes, Ordering::Relaxed);
        self.stats
            .disk_bytes
            .fetch_sub(self.disk_bytes, Ordering::Relaxed);
    }
}
