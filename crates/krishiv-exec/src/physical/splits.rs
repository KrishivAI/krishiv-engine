// SPDX-License-Identifier: Apache-2.0
//! Scan splits as task data: which files (or relation batches) one task
//! reads.
//!
//! Splits live in [`TaskSpec`](super::TaskSpec), never in the plan fragment:
//! the same fragment bytes run on every worker and every attempt, and a
//! retried task reads the same splits. A scan without an entry in
//! [`TaskSpec::scans`](super::TaskSpec) plans its own splits exactly as
//! before, so single-process behaviour is unchanged.
//!
//! - [`SplitFileRef`]: one listed file (path, size and version tag), the
//!   worker's [`FileRef`](krishiv_io::FileRef) with a pinned version: every
//!   read carries `if_match` on the version tag, so a changed file fails
//!   with `KRISHIV_FILE_CHANGED_DURING_READ` instead of mixing versions.
//! - [`plan_file_partitions`]: the resolver-side split planning over the
//!   files listed once, with [`SplitConfig`] over the total worker slots. A split carries a byte range, never
//!   row-group indices: the coordinator reads no footers.
//! - [`deal_ranges`]: the deterministic dealing of partitions (or batches)
//!   to tasks in contiguous index ranges: the same inputs give the same
//!   assignment, so a retried task reads the same splits.
//!
//! Splits need no plan-codec representation: they travel with the task that
//! runs a fragment, not in the fragment's bytes.

use std::ops::Range;
use std::sync::Arc;

use krishiv_io::split::{FilePartition, SplitConfig, SplitFile};
use krishiv_io::{StoreId, VersionToken};
use object_store::path::Path;

use crate::error::{Result, internal};

/// One listed file as a task sees it: identity, size and the version to pin
/// reads to (the store's ETag where it returns one, else the modification
/// time and size it was listed with).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitFileRef {
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

impl SplitFileRef {
    /// The task's view of a listed file.
    #[must_use]
    pub fn from_file_ref(file: &krishiv_io::FileRef) -> Self {
        Self {
            store: file.store.clone(),
            path: file.path.clone(),
            size: file.size,
            version: file.version.clone(),
            modified_micros: file.modified_micros,
        }
    }

    /// The listed file, its version pinned as at listing: reads through it
    /// fail with `KRISHIV_FILE_CHANGED_DURING_READ` when the file changed.
    #[must_use]
    pub fn to_file_ref(&self) -> krishiv_io::FileRef {
        krishiv_io::FileRef {
            store: self.store.clone(),
            path: self.path.clone(),
            size: self.size,
            version: self.version.clone(),
            modified_micros: self.modified_micros,
        }
    }
}

/// The splits of one scan that one task reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanSplits {
    /// File splits: the scan's files (in listing order, split indexes
    /// refer to them) and this task's partitions of them.
    Files {
        /// The scan's files, in listing order.
        files: Arc<[SplitFileRef]>,
        /// This task's partitions (possibly none).
        partitions: Vec<FilePartition>,
    },
    /// Relation batches: the batch range of an in-memory relation this
    /// task reads, resolved through the worker's relation store.
    Batches {
        /// The relation, by content hash.
        relation: krishiv_memory::RelationRef,
        /// The batch indexes this task reads (possibly empty).
        range: Range<usize>,
    },
}

/// Plans the [`FilePartition`]s of `files` (in listing order) with `cfg`:
/// Spark's split planning over sizes only, no footer I/O.
/// `splittable` says whether a file may be cut (the format's capabilities
/// over its read options, by path).
#[must_use]
pub fn plan_file_partitions(
    cfg: &SplitConfig,
    files: &[SplitFileRef],
    splittable: impl Fn(&SplitFileRef) -> bool,
) -> Vec<FilePartition> {
    let sizes: Vec<SplitFile> = files
        .iter()
        .map(|f| SplitFile {
            size: f.size,
            splittable: splittable(f),
        })
        .collect();
    krishiv_io::split::plan(cfg, &sizes)
}

/// Deals `count` partitions (or batches) to `tasks` tasks in contiguous
/// index ranges: task `t` reads `out[t]`. The first `count % tasks` tasks
/// take one extra, so the ranges cover every index exactly once and are
/// empty only when there are more tasks than partitions (or none at all).
/// The same inputs give the same assignment.
///
/// With no tasks there is nothing to deal to, so the result is empty.
pub fn deal_ranges(count: usize, tasks: usize) -> Vec<Range<usize>> {
    if tasks == 0 {
        return Vec::new();
    }
    let base = count / tasks;
    let extra = count % tasks;
    let mut start = 0_usize;
    (0..tasks)
        .map(|t| {
            let len = base.saturating_add(usize::from(t < extra));
            let end = start.saturating_add(len);
            let range = start..end;
            start = end;
            range
        })
        .collect()
}

/// The file splits of `files` dealt to `tasks` tasks: what each task's
/// [`ScanSplits::Files`] holds. `partitions` is
/// [`plan_file_partitions`] over the files in listing order; task `t` reads
/// `partitions` restricted to `deal_ranges(partitions.len(), tasks)[t]`,
/// with the files in listing order.
#[must_use]
pub fn assign_file_splits(
    files: &[SplitFileRef],
    partitions: &[FilePartition],
    tasks: usize,
) -> Vec<ScanSplits> {
    let shared: Arc<[SplitFileRef]> = Arc::from(files);
    deal_ranges(partitions.len(), tasks)
        .into_iter()
        .map(|range| {
            // Partitions keep the position they were planned at, so a
            // retried task claims the same morsels in the same order.
            let parts: Vec<FilePartition> = partitions.get(range).unwrap_or(&[]).to_vec();
            ScanSplits::Files {
                files: Arc::clone(&shared),
                partitions: parts,
            }
        })
        .collect()
}

/// The batch ranges of a relation of `batches` batches dealt to `tasks`
/// tasks: what each task's [`ScanSplits::Batches`] holds.
#[must_use]
pub fn assign_batch_splits(
    relation: krishiv_memory::RelationRef,
    batches: usize,
    tasks: usize,
) -> Vec<ScanSplits> {
    deal_ranges(batches, tasks)
        .into_iter()
        .map(|range| ScanSplits::Batches { relation, range })
        .collect()
}

/// The files of a [`ScanSplits::Files`] as listed files, in listing order.
///
/// # Errors
/// `INTERNAL_ERROR` for a [`ScanSplits::Batches`].
pub fn files_of(splits: &ScanSplits) -> Result<Vec<krishiv_io::FileRef>> {
    match splits {
        ScanSplits::Files { files, .. } => {
            Ok(files.iter().map(SplitFileRef::to_file_ref).collect())
        }
        ScanSplits::Batches { .. } => Err(internal("a batch split has no files".to_owned())),
    }
}

/// The partitions of a [`ScanSplits::Files`].
///
/// # Errors
/// `INTERNAL_ERROR` for a [`ScanSplits::Batches`].
pub fn partitions_of(splits: &ScanSplits) -> Result<Vec<FilePartition>> {
    match splits {
        ScanSplits::Files { partitions, .. } => Ok(partitions.clone()),
        ScanSplits::Batches { .. } => {
            Err(internal("a batch split has no file partitions".to_owned()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, size: u64) -> SplitFileRef {
        SplitFileRef {
            store: StoreId::new("file://"),
            path: Path::from(path),
            size,
            version: VersionToken::ETag("e".into()),
            modified_micros: Some(7),
        }
    }

    fn config(max_partition_bytes: u64) -> SplitConfig {
        SplitConfig {
            max_partition_bytes,
            open_cost_in_bytes: 4,
            min_partition_num: 2,
            max_partition_num: None,
        }
    }

    #[test]
    fn ranges_cover_every_index_exactly_once_in_order() {
        // The first `count % tasks` tasks take one extra.
        assert_eq!(
            deal_ranges(10, 4),
            vec![0..3, 3..6, 6..8, 8..10],
            "10 over 4"
        );
        assert_eq!(deal_ranges(4, 4), vec![0..1, 1..2, 2..3, 3..4]);
        assert_eq!(deal_ranges(1, 1), vec![0..1]);
        // More tasks than partitions: the rest read nothing (empty ranges
        // at the end).
        assert_eq!(deal_ranges(2, 4), vec![0..1, 1..2, 2..2, 2..2]);
        // Nothing planned: every task reads nothing.
        assert_eq!(deal_ranges(0, 3), vec![0..0, 0..0, 0..0]);
        // No tasks: nothing to deal to.
        assert!(deal_ranges(5, 0).is_empty());
        // Large counts neither overflow nor lose an index.
        let ranges = deal_ranges(usize::MAX, 3);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[2].end, usize::MAX);
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start, "contiguous");
        }
    }

    #[test]
    fn dealing_is_deterministic() {
        for (count, tasks) in [(0, 1), (1, 4), (7, 3), (100, 7)] {
            assert_eq!(deal_ranges(count, tasks), deal_ranges(count, tasks));
        }
    }

    #[test]
    fn splittable_files_are_cut_and_others_stay_whole() {
        let files = vec![file("a.parquet", 10), file("b.csv.gz", 10)];
        let splittable = |f: &SplitFileRef| !f.path.as_ref().ends_with(".gz");
        let parts = plan_file_partitions(&config(4), &files, splittable);
        // 10 bytes cut every 4: three splits of the parquet file, the gzip
        // file whole, packed together.
        let parquet: usize = parts
            .iter()
            .flat_map(|p| &p.splits)
            .filter(|s| s.file == 0)
            .count();
        assert_eq!(parquet, 3, "cut every 4 bytes: {parts:?}");
        let gz: Vec<_> = parts
            .iter()
            .flat_map(|p| &p.splits)
            .filter(|s| s.file == 1)
            .collect();
        assert_eq!(gz.len(), 1, "unsplittable stays whole: {parts:?}");
        assert_eq!(gz[0].range, krishiv_io::ByteRange::whole(10));
        // Every split is a byte range of its file.
        for part in &parts {
            for s in &part.splits {
                let size = files[s.file].size;
                assert!(s.range.end <= size, "{s:?} of {size}");
            }
        }
    }

    #[test]
    fn zero_and_empty_files_plan_no_partitions() {
        assert!(plan_file_partitions(&config(4), &[], |_| true).is_empty());
        // Spark's file index never lists zero-length files.
        let files = vec![file("empty.parquet", 0)];
        assert!(plan_file_partitions(&config(4), &files, |_| true).is_empty());
    }

    #[test]
    fn file_refs_keep_the_listed_version() {
        let listed = krishiv_io::FileRef {
            store: StoreId::new("s3://bucket"),
            path: Path::from("a/part-0.parquet"),
            size: 100,
            version: VersionToken::ETag("tag-1".into()),
            modified_micros: Some(42),
        };
        let back = SplitFileRef::from_file_ref(&listed).to_file_ref();
        assert_eq!(back, listed);
        let modified = krishiv_io::FileRef {
            store: listed.store.clone(),
            path: listed.path.clone(),
            size: listed.size,
            version: VersionToken::Modified(42),
            modified_micros: Some(42),
        };
        let back = SplitFileRef::from_file_ref(&modified).to_file_ref();
        assert_eq!(back, modified);
        assert_eq!(
            SplitFileRef::from_file_ref(&listed).version,
            VersionToken::ETag("tag-1".into())
        );
    }

    #[test]
    fn assigned_tasks_cover_every_partition_exactly_once() {
        let files = vec![file("a.parquet", 20)];
        let parts = plan_file_partitions(&config(4), &files, |_| true);
        assert!(!parts.is_empty());
        for tasks in [1, 2, 3, 5] {
            let assigned = assign_file_splits(&files, &parts, tasks);
            assert_eq!(assigned.len(), tasks);
            let mut seen: Vec<usize> = assigned
                .iter()
                .flat_map(|s| partitions_of(s).expect("file splits"))
                .map(|p| p.index)
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..parts.len()).collect::<Vec<_>>(), "{tasks} tasks");
            // The same listing and config give the same assignment.
            assert_eq!(assigned, assign_file_splits(&files, &parts, tasks));
        }
    }

    #[test]
    fn assigned_batch_ranges_cover_every_batch_exactly_once() {
        let relation = krishiv_memory::RelationRef {
            hash: krishiv_memory::RelationHash::from_bytes([1; 32]),
        };
        for (batches, tasks) in [(0, 3), (1, 4), (9, 4)] {
            let assigned = assign_batch_splits(relation, batches, tasks);
            assert_eq!(assigned.len(), tasks);
            let mut seen = Vec::new();
            for s in &assigned {
                match s {
                    ScanSplits::Batches { range, .. } => seen.push(range.clone()),
                    ScanSplits::Files { .. } => panic!("batch splits: {s:?}"),
                }
            }
            let covered: Vec<usize> = seen.iter().flat_map(Clone::clone).collect();
            assert_eq!(covered, (0..batches).collect::<Vec<_>>());
            assert_eq!(assigned, assign_batch_splits(relation, batches, tasks));
        }
    }

    #[test]
    fn splits_of_the_wrong_kind_are_an_internal_error() {
        let batches = ScanSplits::Batches {
            relation: krishiv_memory::RelationRef {
                hash: krishiv_memory::RelationHash::from_bytes([2; 32]),
            },
            range: 0..1,
        };
        assert!(files_of(&batches).is_err());
        assert!(partitions_of(&batches).is_err());
    }
}
