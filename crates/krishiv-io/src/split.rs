// SPDX-License-Identifier: Apache-2.0
//! Spark's file-split planning, reproduced exactly (the partition count is
//! user-observable). From listing sizes only, no footer I/O:
//!
//! 1. `bytesPerCore = Σ(size + openCostInBytes) / minPartitionNum`, where
//!    `minPartitionNum` is `spark.sql.files.minPartitionNum`, else the leaf
//!    parallelism (`spark.sql.leafNodeDefaultParallelism`, else the worker
//!    slots).
//! 2. `maxSplitBytes = min(maxPartitionBytes, max(openCostInBytes, bytesPerCore))`.
//! 3. Splittable files are cut every `maxSplitBytes` bytes; others stay whole.
//! 4. Splits are sorted by length, descending (stable), and packed next-fit:
//!    a partition is closed when adding a split would exceed
//!    `maxSplitBytes`; each split costs its length plus `openCostInBytes`.
//! 5. If that yields more than `spark.sql.files.maxPartitionNum`
//!    partitions, the same splits are packed again with
//!    `ceil(Σ(length + openCostInBytes) / maxPartitionNum)` as the limit.
//!
//! Zero-length files are skipped (Spark's file index never lists them).
//! Checked against partitions Spark 4.2 planned
//! (`tests/fixtures/spark-4.2.0/file-splits.json`).

use krishiv_common::config::{
    Config, ConfigValue, FILES_MAX_PARTITION_BYTES, FILES_MAX_PARTITION_NUM,
    FILES_MIN_PARTITION_NUM, FILES_OPEN_COST_IN_BYTES, LEAF_NODE_DEFAULT_PARALLELISM,
};

use crate::io::ByteRange;

/// Split-planning settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitConfig {
    /// `spark.sql.files.maxPartitionBytes`.
    pub max_partition_bytes: u64,
    /// `spark.sql.files.openCostInBytes`.
    pub open_cost_in_bytes: u64,
    /// `spark.sql.files.minPartitionNum`, else the leaf parallelism.
    pub min_partition_num: u64,
    /// `spark.sql.files.maxPartitionNum`.
    pub max_partition_num: Option<u64>,
}

impl SplitConfig {
    /// The settings in `c`; `worker_slots` is the parallelism when neither
    /// `minPartitionNum` nor `leafNodeDefaultParallelism` is set.
    #[must_use]
    pub fn from_config(c: &Config, worker_slots: u64) -> Self {
        let int = |e| {
            c.get(e)
                .and_then(ConfigValue::as_int)
                .and_then(|v| u64::try_from(v).ok())
                .filter(|v| *v > 0)
        };
        let size = |e, d| c.get(e).and_then(ConfigValue::as_size).unwrap_or(d);
        Self {
            max_partition_bytes: size(&FILES_MAX_PARTITION_BYTES, 128 << 20),
            open_cost_in_bytes: size(&FILES_OPEN_COST_IN_BYTES, 4 << 20),
            min_partition_num: int(&FILES_MIN_PARTITION_NUM)
                .or_else(|| int(&LEAF_NODE_DEFAULT_PARALLELISM))
                .unwrap_or(worker_slots)
                .max(1),
            max_partition_num: int(&FILES_MAX_PARTITION_NUM),
        }
    }
}

/// One listed file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitFile {
    /// Size in bytes.
    pub size: u64,
    /// Whether it may be split (format and codec, see
    /// [`Capabilities::is_splittable`](crate::Capabilities::is_splittable)).
    pub splittable: bool,
}

/// A byte range of one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileSplit {
    /// Index of the file in the planner's input.
    pub file: usize,
    /// The bytes.
    pub range: ByteRange,
}

/// The splits one task (distributed) or morsel (embedded) reads, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilePartition {
    /// Position of the partition.
    pub index: usize,
    /// Its splits.
    pub splits: Vec<FileSplit>,
}

/// Spark's `maxSplitBytes` for `files`.
#[must_use]
pub fn max_split_bytes(cfg: &SplitConfig, files: &[SplitFile]) -> u64 {
    let total: u128 = files
        .iter()
        .filter(|f| f.size > 0)
        .map(|f| u128::from(f.size) + u128::from(cfg.open_cost_in_bytes))
        .sum();
    let per_core =
        u64::try_from(total / u128::from(cfg.min_partition_num.max(1))).unwrap_or(u64::MAX);
    cfg.max_partition_bytes
        .min(cfg.open_cost_in_bytes.max(per_core))
}

/// Plans the partitions of a scan of `files` (in listing order).
#[must_use]
pub fn plan(cfg: &SplitConfig, files: &[SplitFile]) -> Vec<FilePartition> {
    let max_split = max_split_bytes(cfg, files).max(1);
    let mut splits: Vec<FileSplit> = Vec::new();
    for (i, f) in files.iter().enumerate().filter(|(_, f)| f.size > 0) {
        if f.splittable {
            let mut at = 0;
            while at < f.size {
                let end = f.size.min(at.saturating_add(max_split));
                splits.push(FileSplit {
                    file: i,
                    range: ByteRange::new(at, end),
                });
                at = end;
            }
        } else {
            splits.push(FileSplit {
                file: i,
                range: ByteRange::whole(f.size),
            });
        }
    }
    // Stable: equal lengths keep listing order, as Scala's `sortBy`.
    splits.sort_by_key(|s| std::cmp::Reverse(s.range.len()));
    let parts = pack(&splits, max_split, cfg.open_cost_in_bytes);
    match cfg.max_partition_num {
        Some(max) if parts.len() as u64 > max => {
            let total: u128 = splits
                .iter()
                .map(|s| u128::from(s.range.len()) + u128::from(cfg.open_cost_in_bytes))
                .sum();
            let desired = u64::try_from(total.div_ceil(u128::from(max))).unwrap_or(u64::MAX);
            pack(&splits, desired, cfg.open_cost_in_bytes)
        }
        _ => parts,
    }
}

fn pack(splits: &[FileSplit], max_split: u64, open_cost: u64) -> Vec<FilePartition> {
    let mut parts = Vec::new();
    let mut current: Vec<FileSplit> = Vec::new();
    let mut size: u64 = 0;
    for s in splits {
        if size.saturating_add(s.range.len()) > max_split && !current.is_empty() {
            parts.push(FilePartition {
                index: parts.len(),
                splits: std::mem::take(&mut current),
            });
            size = 0;
        }
        size = size.saturating_add(s.range.len()).saturating_add(open_cost);
        current.push(*s);
    }
    if !current.is_empty() {
        parts.push(FilePartition {
            index: parts.len(),
            splits: current,
        });
    }
    parts
}

#[cfg(test)]
mod tests;
