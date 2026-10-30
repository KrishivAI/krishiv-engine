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

/// Morsels a scan cuts per driver: enough that a driver which finds its
/// morsels cheap (a filter that skips their row groups) claims more, so a
/// selective filter over clustered data still spreads over every driver.
pub const MORSELS_PER_DRIVER: u64 = 4;

/// A piece of a split, as one morsel reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorselSplit {
    /// Index of the file in the planner's input.
    pub file: usize,
    /// The bytes this piece reads: the units (row groups, lines) whose
    /// midpoint or start it holds, as for any split.
    pub range: ByteRange,
    /// The planned split the piece belongs to (what `_metadata`'s
    /// `file_block_start` and `file_block_length` report).
    pub split: ByteRange,
}

/// The morsel size for `partitions` read by `drivers` drivers:
/// [`MORSELS_PER_DRIVER`] morsels per driver, never less than `floor` bytes
/// (the cost of opening a split).
#[must_use]
pub fn morsel_bytes(partitions: &[FilePartition], drivers: usize, floor: u64) -> u64 {
    let total: u64 = partitions
        .iter()
        .flat_map(|p| &p.splits)
        .map(|s| s.range.len())
        .fold(0, u64::saturating_add);
    let morsels = (drivers.max(1) as u64).saturating_mul(MORSELS_PER_DRIVER);
    total.div_ceil(morsels).max(floor).max(1)
}

/// The morsels drivers claim for `partitions`: each partition's splits in
/// order, splittable ones (`splittable(file)`) cut into near-equal pieces
/// of at most `target` bytes, and consecutive pieces of one partition
/// packed up to `target` bytes. A morsel never spans two partitions, and
/// reading the morsels in order reads every partition's rows in its order:
/// a unit belongs to the piece holding its midpoint (lines: the piece
/// holding their start), as it belongs to one split. Partitions without
/// splits give no morsel.
#[must_use]
pub fn morsels(
    partitions: &[FilePartition],
    target: u64,
    splittable: impl Fn(usize) -> bool,
) -> Vec<Vec<MorselSplit>> {
    let target = target.max(1);
    let mut out = Vec::new();
    for p in partitions {
        let mut current: Vec<MorselSplit> = Vec::new();
        let mut size: u64 = 0;
        for s in &p.splits {
            let len = s.range.len();
            let pieces = if splittable(s.file) && len > target {
                len.div_ceil(target)
            } else {
                1
            };
            for i in 0..pieces {
                let at = |k: u64| {
                    // `len * k / pieces` without overflow.
                    let off = u128::from(len) * u128::from(k) / u128::from(pieces);
                    s.range.start + u64::try_from(off).unwrap_or(len)
                };
                let range = ByteRange::new(at(i), at(i + 1));
                if !current.is_empty() && size.saturating_add(range.len()) > target {
                    out.push(std::mem::take(&mut current));
                    size = 0;
                }
                size = size.saturating_add(range.len());
                current.push(MorselSplit {
                    file: s.file,
                    range,
                    split: s.range,
                });
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

#[cfg(test)]
mod tests;
