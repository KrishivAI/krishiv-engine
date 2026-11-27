// SPDX-License-Identifier: Apache-2.0
//! Tests of the rewrite planner: the expected values follow Java's
//! `BinPackRewriteFilePlanner` / `SizeBasedFileRewritePlanner` as read from apache-iceberg 1.12.0.

use std::collections::BTreeMap;

use super::*;

const MIB: i64 = 1024 * 1024;

fn opts(pairs: &[(&str, &str)]) -> RewriteOptions {
    let map: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    RewriteOptions::parse(&map, Some(100 * MIB)).unwrap()
}

fn file(path: &str, part: &str, length: i64) -> CandidateFile {
    CandidateFile {
        path: path.into(),
        partition: vec![part.into()],
        default_spec: true,
        length,
        record_count: 1000,
        delete_files: 0,
        file_scoped_deleted_records: 0,
    }
}

fn paths(g: &PlannedGroup) -> Vec<&str> {
    g.files.iter().map(|f| f.path.as_str()).collect()
}

#[test]
fn pack_is_forward_with_one_open_bin_and_a_count_cap() {
    assert_eq!(
        pack(&[4, 4, 4, 4, 4], 10, usize::MAX),
        [vec![0, 1], vec![2, 3], vec![4]]
    );
    assert_eq!(
        pack(&[3, 20, 3, 3], 10, usize::MAX),
        [vec![0], vec![1], vec![2, 3]]
    );
    assert_eq!(
        pack(&[1, 1, 1, 1, 1], 100, 2),
        [vec![0, 1], vec![2, 3], vec![4]]
    );
    assert!(pack(&[], 10, 2).is_empty());
}

#[test]
fn derived_sizes_follow_java() {
    let o = opts(&[]);
    assert_eq!(o.write_max_file_size(), 100 * MIB + (80 * MIB) / 2);
    assert_eq!(o.expected_output_files(50 * MIB), 1);
    // 250 MiB: remainder 50 MiB <= min 75 MiB, average without remainder 125 MiB >= min(110, 140)
    // is false (125 > 110), so the remainder keeps its own file: 3.
    assert_eq!(o.expected_output_files(250 * MIB), 3);
    // 205 MiB: remainder 5 MiB, average 102.5 MiB < 110 MiB: round down to 2.
    assert_eq!(o.expected_output_files(205 * MIB), 2);
    // 280 MiB: remainder 80 MiB > min: 3.
    assert_eq!(o.expected_output_files(280 * MIB), 3);
    // split size is the target when the estimate is below it, else capped at writeMax.
    assert_eq!(o.input_split_size(50 * MIB), 100 * MIB);
    assert_eq!(
        o.input_split_size(205 * MIB),
        102 * MIB + MIB / 2 + 5 * 1024
    );
}

#[test]
fn small_files_group_per_partition_and_need_min_input_files() {
    let o = opts(&[("min-input-files", "3")]);
    let files = vec![
        file("a1", "a", MIB),
        file("a2", "a", MIB),
        file("a3", "a", MIB),
        file("b1", "b", MIB),
        file("b2", "b", MIB),
        file("c1", "c", 100 * MIB),
    ];
    let p = plan(files, &o);
    assert_eq!(p.total_group_count, 1);
    assert_eq!(p.groups.len(), 1);
    assert_eq!(paths(&p.groups[0]), ["a1", "a2", "a3"]);
    assert_eq!(p.groups[0].expected_output_files, 1);
}

#[test]
fn a_delete_heavy_file_keeps_its_group_whatever_its_size() {
    let mut f = file("a1", "a", 100 * MIB);
    f.delete_files = 2;
    f.file_scoped_deleted_records = 400; // 0.4 >= 0.3
    let p = plan(vec![f], &opts(&[]));
    assert_eq!(p.groups.len(), 1);

    // equality deletes carry no file-scoped records: only the count threshold applies
    let mut g = file("a1", "a", 100 * MIB);
    g.delete_files = 2;
    assert!(plan(vec![g.clone()], &opts(&[])).groups.is_empty());
    assert_eq!(
        plan(vec![g], &opts(&[("delete-file-threshold", "2")]))
            .groups
            .len(),
        1
    );
}

#[test]
fn files_of_other_specs_are_one_empty_partition() {
    let mut a = file("a", "x", MIB);
    a.default_spec = false;
    let mut b = file("b", "y", MIB);
    b.default_spec = false;
    let p = plan(vec![a, b], &opts(&[("min-input-files", "2")]));
    assert_eq!(p.groups.len(), 1);
    assert!(p.groups[0].partition.is_empty());
}

#[test]
fn rewrite_all_skips_both_filters() {
    let p = plan(
        vec![file("a", "a", 100 * MIB)],
        &opts(&[("rewrite-all", "true")]),
    );
    assert_eq!(p.groups.len(), 1);
}

#[test]
fn max_files_to_rewrite_cuts_before_the_job_order_sort() {
    let o = opts(&[
        ("min-input-files", "2"),
        ("max-files-to-rewrite", "3"),
        ("rewrite-job-order", "files-asc"),
        ("max-file-group-input-files", "2"),
    ]);
    let files = vec![
        file("a1", "a", MIB),
        file("a2", "a", MIB),
        file("a3", "a", MIB),
        file("a4", "a", MIB),
        file("b1", "b", MIB),
        file("b2", "b", MIB),
    ];
    let p = plan(files, &o);
    assert_eq!(p.total_group_count, 3);
    // partition a: groups [a1,a2] and [a3,a4]; the cut takes a1,a2 then a3 alone; b is dropped.
    assert_eq!(p.groups.len(), 2);
    assert_eq!(paths(&p.groups[0]), ["a3"]);
    assert_eq!(paths(&p.groups[1]), ["a1", "a2"]);
    assert_eq!(
        (p.groups[0].global_index, p.groups[0].partition_index),
        (2, 2)
    );
}
