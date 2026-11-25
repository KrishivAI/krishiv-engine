// SPDX-License-Identifier: Apache-2.0
//! The planner of `rewrite_data_files` (design P3-T027 PR-36e §4.3): Java's
//! `BinPackRewriteFilePlanner` and `SizeBasedFileRewritePlanner` (apache-iceberg 1.12.0), pure
//! over the candidate files of the starting snapshot. Which files exist, their partitions and
//! their deletes come from the caller's scan; nothing here reads a table.
//!
//! Java keeps the partitions in a `StructLikeMap` (hash order) and applies `max-files-to-rewrite`
//! in that order; the order is unspecified there, so Krishiv uses the partition key's order.

#![expect(
    dead_code,
    reason = "group execution and the procedure's call arm (next commits of 36e-2a) call this"
)]

use std::collections::BTreeMap;

use super::rewrite_options::{JobOrder, RewriteOptions};

/// `SizeBasedFileRewritePlanner.SPLIT_OVERHEAD`: 5 KiB.
const SPLIT_OVERHEAD: i64 = 5 * 1024;

/// One data file the scan returned for the starting snapshot.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CandidateFile {
    /// The file's path (identity).
    pub path: String,
    /// The partition of the file's own spec, as a comparable key.
    pub partition: Vec<String>,
    /// Whether the file's spec is the table's default spec; other files are all planned under
    /// the empty partition.
    pub default_spec: bool,
    /// `file_size_in_bytes` (the task's `length()`).
    pub length: i64,
    /// The file's record count.
    pub record_count: i64,
    /// How many delete files apply to the file (`task.deletes().size()`).
    pub delete_files: usize,
    /// The sum of record counts of the applying deletes that are file-scoped (DVs and position
    /// deletes naming one data file).
    pub file_scoped_deleted_records: i64,
}

/// One file group the rewrite will execute.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlannedGroup {
    /// 1-based index over the whole plan, in selection order (`FileGroupInfo.globalIndex`).
    pub global_index: usize,
    /// 1-based index within the group's partition (`partitionIndex`).
    pub partition_index: usize,
    /// The partition key the group was planned under (empty for other specs' files).
    pub partition: Vec<String>,
    /// The files of the group, in packing order.
    pub files: Vec<CandidateFile>,
    /// The size the group's input is split at when read.
    pub input_split_size: i64,
    /// The size ceiling of an output file (`writeMaxFileSize`).
    pub write_max_file_size: i64,
    /// How many output files the group is expected to produce.
    pub expected_output_files: usize,
}

impl PlannedGroup {
    /// The total length of the group's files.
    pub(crate) fn input_bytes(&self) -> i64 {
        self.files.iter().map(|f| f.length).sum()
    }
}

/// The plan: groups in execution order, and how many groups each partition had before
/// `max-files-to-rewrite` cut the selection.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RewritePlan {
    /// The selected groups, sorted by `rewrite-job-order` (stable).
    pub groups: Vec<PlannedGroup>,
    /// Groups the filters kept, over all partitions, before the cut.
    pub total_group_count: usize,
}

/// Java's `ListPacker(target, 1, false, max_items).pack(items, weight)`: forward, one open
/// bin; an item that does not fit (by weight or count) closes the bin and opens the next. The
/// first item of a bin is always taken, so an item heavier than `target` is alone.
pub(crate) fn pack(weights: &[i64], target: i64, max_items: usize) -> Vec<Vec<usize>> {
    let target = i128::from(target);
    let mut bins: Vec<Vec<usize>> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    let mut weight: i128 = 0;
    for (i, w) in weights.iter().enumerate() {
        let w = i128::from(*w);
        if !open.is_empty() && (weight + w > target || open.len() >= max_items) {
            bins.push(std::mem::take(&mut open));
            weight = 0;
        }
        open.push(i);
        weight += w;
    }
    if !open.is_empty() {
        bins.push(open);
    }
    bins
}

impl RewriteOptions {
    /// `writeMaxFileSize`: the target plus half the distance to the maximum.
    pub(crate) fn write_max_file_size(&self) -> i64 {
        // Java: `(long) (target + ((max - target) * 0.5))`.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            reason = "Java's double arithmetic on sizes, truncated toward zero as `(long)` does"
        )]
        let v = (self.target_file_size as f64
            + (self.max_file_size - self.target_file_size) as f64 * 0.5) as i64;
        v
    }

    /// `expectedOutputFiles` of a group with `input` bytes.
    pub(crate) fn expected_output_files(&self, input: i64) -> usize {
        let target = self.target_file_size;
        if input < target {
            return 1;
        }
        let with_rem = (input + target - 1) / target;
        let without_rem = input / target;
        let avg_without = input / without_rem;
        #[expect(
            clippy::cast_precision_loss,
            reason = "Java compares the average against `Math.min(1.1 * target, writeMax)` as doubles"
        )]
        let limit = (1.1 * target as f64).min(self.write_max_file_size() as f64);
        let n = if input % target > self.min_file_size {
            with_rem
        } else {
            #[expect(clippy::cast_precision_loss, reason = "as above")]
            if (avg_without as f64) < limit {
                without_rem
            } else {
                with_rem
            }
        };
        usize::try_from(n).unwrap_or(usize::MAX)
    }

    /// `inputSplitSize`.
    pub(crate) fn input_split_size(&self, input: i64) -> i64 {
        let n = i64::try_from(self.expected_output_files(input)).unwrap_or(i64::MAX);
        let estimated = input / n.max(1) + SPLIT_OVERHEAD;
        if estimated < self.target_file_size {
            self.target_file_size
        } else {
            estimated.min(self.write_max_file_size())
        }
    }

    fn too_many_deletes(&self, f: &CandidateFile) -> bool {
        f.delete_files >= self.delete_file_threshold
    }

    fn too_high_delete_ratio(&self, f: &CandidateFile) -> bool {
        if f.delete_files == 0 {
            return false;
        }
        let deleted = f.file_scoped_deleted_records.min(f.record_count);
        #[expect(
            clippy::cast_precision_loss,
            reason = "Java divides two longs as doubles"
        )]
        let ratio = deleted as f64 / f.record_count as f64;
        // A zero-record file gives NaN in Java, and `NaN >= t` is false.
        ratio >= self.delete_ratio_threshold
    }

    fn outside_size_range(&self, f: &CandidateFile) -> bool {
        f.length < self.min_file_size || f.length > self.max_file_size
    }

    fn needs_rewrite(&self, f: &CandidateFile) -> bool {
        self.outside_size_range(f) || self.too_many_deletes(f) || self.too_high_delete_ratio(f)
    }

    fn keeps_group(&self, group: &[CandidateFile]) -> bool {
        let bytes: i64 = group.iter().map(|f| f.length).sum();
        let many = group.len() > 1;
        (many && group.len() >= self.min_input_files)
            || (many && bytes > self.target_file_size)
            || bytes > self.max_file_size
            || group
                .iter()
                .any(|f| self.too_many_deletes(f) || self.too_high_delete_ratio(f))
    }
}

/// Plans the rewrite of `files` (the scan of the starting snapshot after `where`).
pub(crate) fn plan(files: Vec<CandidateFile>, options: &RewriteOptions) -> RewritePlan {
    let mut by_partition: BTreeMap<Vec<String>, Vec<CandidateFile>> = BTreeMap::new();
    for f in files {
        let key = if f.default_spec {
            f.partition.clone()
        } else {
            Vec::new()
        };
        by_partition.entry(key).or_default().push(f);
    }

    let mut per_partition: Vec<(Vec<String>, Vec<Vec<CandidateFile>>)> = Vec::new();
    let mut total_group_count = 0;
    for (key, files) in by_partition {
        let candidates: Vec<CandidateFile> = if options.rewrite_all {
            files
        } else {
            files
                .into_iter()
                .filter(|f| options.needs_rewrite(f))
                .collect()
        };
        let weights: Vec<i64> = candidates.iter().map(|f| f.length).collect();
        let groups: Vec<Vec<CandidateFile>> = pack(
            &weights,
            options.max_group_size,
            options.max_group_input_files,
        )
        .into_iter()
        .map(|bin| bin.into_iter().map(|i| candidates[i].clone()).collect())
        .filter(|g: &Vec<CandidateFile>| options.rewrite_all || options.keeps_group(g))
        .collect();
        total_group_count += groups.len();
        per_partition.push((key, groups));
    }

    let mut selected = Vec::new();
    let mut taken = 0usize;
    let mut global = 0usize;
    for (key, groups) in per_partition {
        let mut in_partition = 0usize;
        for mut group in groups {
            if let Some(max) = options.max_files_to_rewrite {
                if taken >= max {
                    continue;
                }
                let take = group.len().min(max - taken);
                group.truncate(take);
                taken += take;
            }
            let input: i64 = group.iter().map(|f| f.length).sum();
            global += 1;
            in_partition += 1;
            selected.push(PlannedGroup {
                global_index: global,
                partition_index: in_partition,
                partition: key.clone(),
                input_split_size: options.input_split_size(input),
                write_max_file_size: options.write_max_file_size(),
                expected_output_files: options.expected_output_files(input),
                files: group,
            });
        }
    }
    match options.job_order {
        JobOrder::None => {}
        JobOrder::BytesAsc => selected.sort_by_key(PlannedGroup::input_bytes),
        JobOrder::BytesDesc => selected.sort_by_key(|g| std::cmp::Reverse(g.input_bytes())),
        JobOrder::FilesAsc => selected.sort_by_key(|g| g.files.len()),
        JobOrder::FilesDesc => selected.sort_by_key(|g| std::cmp::Reverse(g.files.len())),
    }
    RewritePlan {
        groups: selected,
        total_group_count,
    }
}

#[cfg(test)]
#[path = "rewrite_plan_tests.rs"]
mod tests;
