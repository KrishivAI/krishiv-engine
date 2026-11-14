// SPDX-License-Identifier: Apache-2.0
//! Tests of commit validation (`validate`, PR-36b; design note §7.3 steps 2
//! and 3): the walk over parent ids, each check over a hand-built history
//! (which operations, which entries, Java's texts), the conservative
//! evaluator, the history read from manifests written by iceberg-rust's
//! writers (a merged manifest's `EXISTING` entries, `DELETED` entries, only
//! the manifests an intervening snapshot added), and the commit loop with a
//! validation (`commit_tests`' catalog and clock).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use iceberg::TableUpdate;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestFile,
    ManifestListWriter, ManifestWriterBuilder, Operation, Struct, TableMetadata,
};
use krishiv_common::error::ErrorClass;
use krishiv_io::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};

use super::append::{WriteIds, prepare_attempt, write_new_manifests};
use super::append_tests::{MAX, create, file};
use super::commit::CommitClock;
use super::commit_tests::{Case, Fault, Foreign, INIT, MINE, OURS, When, case, exists};
use super::provider_tests::{Env, data_file, env};
use super::scan::read_manifest_files;
use super::validate::{
    CHECK_ADDED_DATA, CHECK_ADDED_DELETES, CHECK_ADDED_DVS, CHECK_HISTORY, CHECK_REMOVED, Change,
    Checks, Fired, History, Isolation, Judge, Validation, check, validate, walk,
};
use crate::plan::encode;
use crate::plan::entry::{
    ContentFile, ContentKind, DataFileEntry, DeleteFileEntry, DeleteKind, EntryStatus,
    FileFormatKind, LiveEntry, ManifestContent, ManifestFileEntry, PartitionValue,
};
use crate::plan::types::{
    IcebergType, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};

// ---------------------------------------------------------------- the walk

/// A chain of snapshots `(id, parent)`; ids absent are not in the table.
fn chain(links: &[(i64, Option<i64>)]) -> impl Fn(i64) -> Option<Option<i64>> + '_ {
    move |id| links.iter().find(|(i, _)| *i == id).map(|(_, p)| *p)
}

const LINKS: &[(i64, Option<i64>)] = &[(1, None), (2, Some(1)), (3, Some(2)), (4, Some(3))];

#[test]
fn the_walk_stops_at_s0_head_first() {
    assert_eq!(
        walk(Some(4), Some(1), 1_000, chain(LINKS)),
        Ok(vec![4, 3, 2])
    );
    assert_eq!(walk(Some(4), Some(4), 1_000, chain(LINKS)), Ok(vec![]));
    // No head (no current state): nothing to validate, as Java's
    // `parent == null`.
    assert_eq!(walk(None, Some(1), 1_000, chain(LINKS)), Ok(vec![]));
}

/// S0 = none (an empty table at the write's start) walks the whole chain;
/// a chain broken by an expired snapshot then does not end at "no parent".
#[test]
fn the_walk_from_no_snapshot_takes_the_whole_chain() {
    assert_eq!(
        walk(Some(4), None, 1_000, chain(LINKS)),
        Ok(vec![4, 3, 2, 1])
    );
    let expired = &[(3, Some(2)), (4, Some(3))];
    let f = walk(Some(4), None, 1_000, chain(expired)).unwrap_err();
    assert_eq!(f.check, CHECK_HISTORY);
    assert_eq!(
        f.message,
        "Cannot determine history between starting snapshot null and the last known ancestor 3"
    );
}

/// A rollback: S0 is not an ancestor of the head, so the walk runs off the
/// chain's end; Java's text (`MergingSnapshotProducer.java:992`).
#[test]
fn s0_not_an_ancestor_is_a_conflict_with_javas_text() {
    let rolled = &[(1, None), (2, Some(1)), (5, Some(1)), (6, Some(5))];
    let f = walk(Some(6), Some(2), 1_000, chain(rolled)).unwrap_err();
    assert_eq!(
        f,
        Fired {
            check: CHECK_HISTORY,
            snapshot_id: Some(6),
            message: "Cannot determine history between starting snapshot 2 and the last known \
                      ancestor 1"
                .to_owned(),
        }
    );
    // The head missing from the table.
    let f = walk(Some(9), Some(2), 1_000, chain(rolled)).unwrap_err();
    assert!(f.message.ends_with("ancestor null"), "{}", f.message);
}

/// More intervening snapshots than the bound: a conflict, never a partial
/// validation; exactly the bound is validated.
#[test]
fn the_walk_is_bounded() {
    assert_eq!(walk(Some(4), Some(1), 3, chain(LINKS)), Ok(vec![4, 3, 2]));
    let f = walk(Some(4), Some(1), 2, chain(LINKS)).unwrap_err();
    assert_eq!(f.check, CHECK_HISTORY);
    assert!(
        f.message.contains("more than 2 snapshots")
            && f.message.contains("krishiv.iceberg.commit.maxIntervening"),
        "{}",
        f.message
    );
    // Also when S0 is not reachable at all.
    assert!(walk(Some(4), Some(99), 2, chain(LINKS)).is_err());
}

// ------------------------------------------------ the checks, hand-built

/// `id long (1), x int (2), name string (3)`.
fn fields() -> Vec<SchemaField> {
    let f = |id, name: &str, ty| SchemaField {
        id,
        path: ColumnPath::new(name),
        ty,
    };
    vec![
        f(1, "id", IcebergType::Long),
        f(2, "x", IcebergType::Int),
        f(3, "name", IcebergType::String),
    ]
}

/// Spec 0 `identity(x)`; spec 1 an unknown transform of `x`.
fn specs() -> PartitionSpecs {
    let spec = |id, t| PartitionSpecInfo {
        spec_id: id,
        fields: vec![PartitionFieldInfo::new(
            1000,
            2,
            "x".to_owned(),
            t,
            Some(IcebergType::Int),
        )],
    };
    [
        (0, spec(0, Transform::Identity)),
        (1, spec(1, Transform::Unknown)),
    ]
    .into()
}

fn judge() -> Judge {
    Judge::new(fields(), specs())
}

/// A file of `content` at `path` in partition `x` of spec 0.
fn content(kind: ContentKind, path: &str, x: i32) -> ContentFile {
    ContentFile {
        content: kind,
        path: path.to_owned(),
        format: FileFormatKind::Parquet,
        spec_id: 0,
        partition: vec![PartitionValue::Value(
            encode(&ScalarValue::Int32(x), &IcebergType::Int).unwrap(),
        )],
        record_count: 3,
        size: 1,
        column_sizes: BTreeMap::new(),
        value_counts: BTreeMap::new(),
        null_counts: BTreeMap::new(),
        nan_counts: BTreeMap::new(),
        lower_bounds: BTreeMap::new(),
        upper_bounds: BTreeMap::new(),
        split_offsets: None,
        sort_order_id: None,
        first_row_id: None,
        equality_ids: None,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
    }
}

fn entry(status: EntryStatus, snapshot_id: i64, seq: i64, file: ContentFile) -> DataFileEntry {
    DataFileEntry {
        file,
        status,
        snapshot_id,
        data_sequence_number: Some(seq),
        file_sequence_number: Some(seq),
    }
}

/// A data entry of `path` in partition `x`.
fn data(status: EntryStatus, snapshot_id: i64, path: &str, x: i32) -> LiveEntry {
    LiveEntry::Data(entry(
        status,
        snapshot_id,
        snapshot_id,
        content(ContentKind::Data, path, x),
    ))
}

/// An ADDED position delete file in partition `x`, data sequence `seq`.
fn pos_delete(snapshot_id: i64, seq: i64, path: &str, x: i32) -> LiveEntry {
    LiveEntry::Delete(DeleteFileEntry {
        entry: entry(
            EntryStatus::Added,
            snapshot_id,
            seq,
            content(ContentKind::PositionDeletes, path, x),
        ),
        kind: DeleteKind::PositionDelete,
    })
}

/// An ADDED DV of the data file `of` in partition `x`.
fn dv(snapshot_id: i64, path: &str, of: &str, x: i32) -> LiveEntry {
    let mut f = content(ContentKind::PositionDeletes, path, x);
    f.format = FileFormatKind::Puffin;
    f.referenced_data_file = Some(of.to_owned());
    f.content_offset = Some(4);
    f.content_size_in_bytes = Some(40);
    LiveEntry::Delete(DeleteFileEntry {
        entry: entry(EntryStatus::Added, snapshot_id, snapshot_id, f),
        kind: DeleteKind::DeletionVector,
    })
}

fn manifest(snapshot_id: i64, content: ManifestContent) -> ManifestFileEntry {
    ManifestFileEntry {
        path: format!("m-{snapshot_id}.avro"),
        length: 1,
        spec_id: 0,
        content,
        sequence_number: snapshot_id,
        min_sequence_number: snapshot_id,
        added_snapshot_id: snapshot_id,
        added_files: None,
        existing_files: None,
        deleted_files: None,
        first_row_id: None,
        partitions: None,
        encrypted: false,
    }
}

/// One change per `(snapshot, operation, entries)`; the snapshots are the
/// intervening ones, S0's sequence number is 1.
fn history(changes: Vec<(i64, Operation, Vec<LiveEntry>)>) -> History {
    let snapshots = changes
        .iter()
        .map(|(id, op, _)| (*id, op.clone()))
        .collect();
    let changes = changes
        .into_iter()
        .map(|(id, op, entries)| {
            let deletes = entries.iter().any(|e| matches!(e, LiveEntry::Delete(_)));
            Change {
                snapshot_id: id,
                operation: op,
                manifest: manifest(
                    id,
                    if deletes {
                        ManifestContent::Deletes
                    } else {
                        ManifestContent::Data
                    },
                ),
                entries,
            }
        })
        .collect();
    History {
        snapshots,
        changes,
        start_sequence: 1,
        v1: false,
        default_spec: 0,
    }
}

fn x_is(x: i32) -> PushdownExpr {
    PushdownExpr::cmp(ColumnPath::new("x"), CmpOp::Eq, ScalarValue::Int32(x))
}

/// A validation of `checks` with the filter `x = 1`.
fn validation(checks: Checks) -> Validation {
    Validation {
        checks,
        filter: Some(x_is(1)),
        filter_text: "ref(name=\"x\") == 1".to_owned(),
        ..Validation::none()
    }
}

const ADDED_DATA: Checks = Checks {
    added_data_files: true,
    added_delete_files: false,
    added_dvs: false,
    removed_data_files: false,
};

fn added(id: i64, op: Operation, path: &str, x: i32) -> (i64, Operation, Vec<LiveEntry>) {
    (id, op, vec![data(EntryStatus::Added, id, path, x)])
}

#[test]
fn added_data_files_matching_the_filter_conflict() {
    let v = validation(ADDED_DATA);
    let h = history(vec![
        added(3, Operation::Append, "b.parquet", 2),
        added(2, Operation::Overwrite, "a.parquet", 1),
    ]);
    assert_eq!(
        check(&h, &v, &judge()),
        Some(Fired {
            check: CHECK_ADDED_DATA,
            snapshot_id: Some(2),
            message: "Found conflicting files that can contain records matching \
                      ref(name=\"x\") == 1: [a.parquet]"
                .to_owned(),
        })
    );
    // An append in the same partition too.
    let h = history(vec![added(3, Operation::Append, "b.parquet", 1)]);
    assert_eq!(check(&h, &v, &judge()).unwrap().snapshot_id, Some(3));
    // No filter: every added file.
    let all = Validation {
        filter: None,
        filter_text: "true".to_owned(),
        ..v
    };
    let h = history(vec![added(3, Operation::Append, "b.parquet", 2)]);
    assert!(check(&h, &all, &judge()).is_some());
}

#[test]
fn added_data_files_elsewhere_do_not_conflict() {
    let v = validation(ADDED_DATA);
    let h = history(vec![added(2, Operation::Append, "a.parquet", 2)]);
    assert_eq!(check(&h, &v, &judge()), None);
}

/// A foreign compaction (`replace`) or a `delete` never conflicts through
/// what it added (`VALIDATE_ADDED_FILES_OPERATIONS`).
#[test]
fn added_data_files_of_other_operations_do_not_conflict() {
    let v = validation(ADDED_DATA);
    for op in [Operation::Replace, Operation::Delete] {
        let h = history(vec![added(2, op, "a.parquet", 1)]);
        assert_eq!(check(&h, &v, &judge()), None);
    }
}

/// `EXISTING` entries (a merged manifest) and entries of snapshots outside
/// the intervening set are not changes.
#[test]
fn existing_entries_and_older_snapshots_are_not_changes() {
    let v = validation(ADDED_DATA);
    let h = history(vec![(
        3,
        Operation::Overwrite,
        vec![
            data(EntryStatus::Existing, 3, "merged.parquet", 1),
            // Added by snapshot 1 (S0's ancestry), carried as ADDED.
            data(EntryStatus::Added, 1, "old.parquet", 1),
        ],
    )]);
    assert_eq!(check(&h, &v, &judge()), None);
}

/// Snapshot isolation runs no "added data files" check.
#[test]
fn snapshot_isolation_ignores_added_data_files() {
    let v = Validation {
        isolation: Isolation::Snapshot,
        ..validation(ADDED_DATA)
    };
    assert!(!v.runs());
    let h = history(vec![added(2, Operation::Append, "a.parquet", 1)]);
    assert_eq!(check(&h, &v, &judge()), None);
}

const ADDED_DELETES: Checks = Checks {
    added_data_files: false,
    added_delete_files: true,
    added_dvs: false,
    removed_data_files: false,
};

#[test]
fn added_delete_files_matching_the_filter_conflict() {
    let v = validation(ADDED_DELETES);
    let h = history(vec![(
        2,
        Operation::Delete,
        vec![
            pos_delete(2, 2, "d1.parquet", 1),
            pos_delete(2, 2, "d2.parquet", 2),
            pos_delete(2, 2, "d3.parquet", 1),
        ],
    )]);
    assert_eq!(
        check(&h, &v, &judge()),
        Some(Fired {
            check: CHECK_ADDED_DELETES,
            snapshot_id: Some(2),
            message: "Found new conflicting delete files that can apply to records matching \
                      ref(name=\"x\") == 1: [d1.parquet, d3.parquet]"
                .to_owned(),
        })
    );
    let h = history(vec![(
        2,
        Operation::Overwrite,
        vec![pos_delete(2, 2, "d.parquet", 1)],
    )]);
    assert!(check(&h, &v, &judge()).is_some());
}

/// Not for another partition, an `append` or `replace` snapshot, a data
/// sequence number at or below S0's (`afterSequenceNumber`), or a v1 table.
#[test]
fn added_delete_files_that_cannot_apply_do_not_conflict() {
    let v = validation(ADDED_DELETES);
    let one = |op, seq, x| history(vec![(2, op, vec![pos_delete(2, seq, "d.parquet", x)])]);
    assert_eq!(check(&one(Operation::Delete, 2, 2), &v, &judge()), None);
    assert_eq!(check(&one(Operation::Append, 2, 1), &v, &judge()), None);
    assert_eq!(check(&one(Operation::Replace, 2, 1), &v, &judge()), None);
    assert_eq!(check(&one(Operation::Delete, 1, 1), &v, &judge()), None);
    let mut v1 = one(Operation::Delete, 2, 1);
    v1.v1 = true;
    assert_eq!(check(&v1, &v, &judge()), None);
    // A missing data sequence number counts.
    let mut unknown = one(Operation::Delete, 2, 1);
    if let LiveEntry::Delete(d) = &mut unknown.changes[0].entries[0] {
        d.entry.data_sequence_number = None;
    }
    assert!(check(&unknown, &v, &judge()).is_some());
}

const ADDED_DVS: Checks = Checks {
    added_data_files: false,
    added_delete_files: false,
    added_dvs: true,
    removed_data_files: false,
};

#[test]
fn a_dv_added_for_a_file_this_write_deletes_from_conflicts() {
    let v = Validation {
        dvs_for: BTreeSet::from(["a.parquet".to_owned()]),
        ..validation(ADDED_DVS)
    };
    for op in [Operation::Overwrite, Operation::Delete, Operation::Replace] {
        let h = history(vec![(2, op, vec![dv(2, "dv.puffin", "a.parquet", 1)])]);
        assert_eq!(
            check(&h, &v, &judge()),
            Some(Fired {
                check: CHECK_ADDED_DVS,
                snapshot_id: Some(2),
                message: "Found concurrently added DV for a.parquet: DV{location=dv.puffin, \
                          offset=4, length=40, referencedDataFile=a.parquet}"
                    .to_owned(),
            })
        );
    }
    // A DV of another file, an `append` snapshot, a position delete file.
    let other = history(vec![(
        2,
        Operation::Delete,
        vec![dv(2, "dv.puffin", "b.parquet", 1)],
    )]);
    assert_eq!(check(&other, &v, &judge()), None);
    let append = history(vec![(
        2,
        Operation::Append,
        vec![dv(2, "dv.puffin", "a.parquet", 1)],
    )]);
    assert_eq!(check(&append, &v, &judge()), None);
    let pos = history(vec![(
        2,
        Operation::Delete,
        vec![pos_delete(2, 2, "d.parquet", 1)],
    )]);
    assert_eq!(check(&pos, &v, &judge()), None);
    // No DV written: the check does not run.
    assert!(!validation(ADDED_DVS).runs());
}

const REMOVED: Checks = Checks {
    added_data_files: false,
    added_delete_files: false,
    added_dvs: false,
    removed_data_files: true,
};

#[test]
fn a_file_this_write_removes_removed_by_another_conflicts() {
    let v = Validation {
        removes: BTreeSet::from(["a.parquet".to_owned(), "b.parquet".to_owned()]),
        ..validation(REMOVED)
    };
    for op in [Operation::Overwrite, Operation::Replace, Operation::Delete] {
        // In any partition: the path decides, not the filter.
        let h = history(vec![(
            2,
            op,
            vec![
                data(EntryStatus::Deleted, 2, "a.parquet", 7),
                data(EntryStatus::Deleted, 2, "b.parquet", 1),
            ],
        )]);
        assert_eq!(
            check(&h, &v, &judge()),
            Some(Fired {
                check: CHECK_REMOVED,
                snapshot_id: Some(2),
                message: "Missing required files to delete: a.parquet, b.parquet".to_owned(),
            })
        );
    }
    // Added (a compaction's output named alike), another path, or removed
    // by a snapshot outside the set.
    let h = history(vec![(
        2,
        Operation::Replace,
        vec![
            data(EntryStatus::Added, 2, "a.parquet", 1),
            data(EntryStatus::Deleted, 2, "c.parquet", 1),
            data(EntryStatus::Deleted, 1, "b.parquet", 1),
        ],
    )]);
    assert_eq!(check(&h, &v, &judge()), None);
}

/// `validateDataFilesExist(skipDeletes)`: a referenced file removed by a
/// `delete` snapshot counts only without `skipDeletes`.
#[test]
fn a_file_this_write_references_removed_by_another_conflicts() {
    let v = Validation {
        references: BTreeSet::from(["a.parquet".to_owned()]),
        ..validation(REMOVED)
    };
    let by = |op| {
        history(vec![(
            2,
            op,
            vec![data(EntryStatus::Deleted, 2, "a.parquet", 1)],
        )])
    };
    assert_eq!(
        check(&by(Operation::Delete), &v, &judge()),
        Some(Fired {
            check: CHECK_REMOVED,
            snapshot_id: Some(2),
            message: "Cannot commit, missing data files: [a.parquet]".to_owned(),
        })
    );
    let skip = Validation {
        references_skip_deletes: true,
        ..v.clone()
    };
    assert_eq!(check(&by(Operation::Delete), &skip, &judge()), None);
    assert!(check(&by(Operation::Replace), &skip, &judge()).is_some());
    assert!(check(&by(Operation::Overwrite), &skip, &judge()).is_some());
    assert_eq!(check(&by(Operation::Append), &v, &judge()), None);
}

// ------------------------------------------------ the conservative evaluator

/// Every unknown is a match (the lost-update guard, R:S11): a file without
/// bounds, a filter column the schema lacks, a spec the table does not know,
/// an unknown transform. Each has a known counterpart that does not match,
/// so a flipped default fails here.
#[test]
fn every_unknown_counts_as_a_match() {
    let at = |e: LiveEntry, filter: PushdownExpr| {
        let v = Validation {
            filter: Some(filter),
            ..validation(ADDED_DATA)
        };
        check(
            &history(vec![(2, Operation::Append, vec![e])]),
            &v,
            &judge(),
        )
        .is_some()
    };
    let id_is = |n| PushdownExpr::cmp(ColumnPath::new("id"), CmpOp::Eq, ScalarValue::Int64(n));
    let file = |f: ContentFile| LiveEntry::Data(entry(EntryStatus::Added, 2, 2, f));

    // Bounds: none known → match; known and apart → no match.
    let plain = content(ContentKind::Data, "a.parquet", 1);
    assert!(at(file(plain.clone()), id_is(50)));
    let mut bounded = plain;
    bounded
        .lower_bounds
        .insert(1, 10_i64.to_le_bytes().to_vec());
    bounded
        .upper_bounds
        .insert(1, 20_i64.to_le_bytes().to_vec());
    assert!(!at(file(bounded.clone()), id_is(50)));
    // A bound that does not decode is dropped, not trusted.
    let mut garbage = bounded.clone();
    garbage.upper_bounds.insert(1, vec![1, 2, 3]);
    assert!(at(file(garbage), id_is(50)));

    // A column the schema lacks.
    let missing = PushdownExpr::cmp(ColumnPath::new("gone"), CmpOp::Eq, ScalarValue::Int32(1));
    assert!(at(
        file(content(ContentKind::Data, "a.parquet", 2)),
        missing
    ));
    assert!(!at(
        file(content(ContentKind::Data, "a.parquet", 2)),
        x_is(1)
    ));

    // A spec the table does not know.
    let mut unknown_spec = content(ContentKind::Data, "a.parquet", 2);
    unknown_spec.spec_id = 9;
    assert!(at(file(unknown_spec), x_is(1)));

    // An unknown transform (spec 1).
    let mut unknown_transform = content(ContentKind::Data, "a.parquet", 2);
    unknown_transform.spec_id = 1;
    assert!(at(file(unknown_transform), x_is(1)));

    // An unreadable partition value.
    let mut unreadable = content(ContentKind::Data, "a.parquet", 2);
    unreadable.partition = vec![PartitionValue::Unreadable];
    assert!(at(file(unreadable), x_is(1)));
}

// ---------------------------------------- the history read from manifests

/// How a fixture manifest lists a file.
enum E {
    Added(DataFile),
    /// Carried over: the snapshot that added it and its sequence number.
    Existing(DataFile, i64, i64),
    /// Removed: its sequence number.
    Deleted(DataFile, i64),
}

/// Snapshots written by hand on a v2 table partitioned by `identity(x)`:
/// manifests through iceberg-rust's writers, lists through its list writer,
/// the metadata by editing `create`'s JSON.
struct Fixture {
    env: Env,
    meta: TableMetadata,
    snapshots: Vec<serde_json::Value>,
    n: usize,
}

impl Fixture {
    fn new() -> Self {
        let env = env();
        let meta = create(&env, 2).metadata().clone();
        Self {
            env,
            meta,
            snapshots: Vec::new(),
            n: 0,
        }
    }

    fn data(&self, name: &str, x: i32) -> DataFile {
        data_file(
            &self.env.at(name),
            10,
            3,
            Struct::from_iter([Some(Literal::int(x))]),
        )
    }

    fn pos_delete(&self, name: &str, x: i32) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(self.env.at(name))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(10)
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::int(x))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    /// A manifest written by snapshot `by`.
    fn manifest(&mut self, by: i64, deletes: bool, entries: Vec<E>) -> ManifestFile {
        self.n += 1;
        let io = self.env.reg.file_io();
        let path = self.env.at(&format!("m{}.avro", self.n));
        let b = ManifestWriterBuilder::new(
            io.new_output(path).unwrap(),
            Some(by),
            Arc::new(self.meta.current_schema().as_ref().clone()),
            self.meta.default_partition_spec().as_ref().clone(),
        );
        let mut w = if deletes {
            b.build_v2_deletes()
        } else {
            b.build_v2_data()
        };
        for e in entries {
            match e {
                E::Added(f) => w.add_file(f, -1).unwrap(),
                E::Existing(f, s, seq) => w.add_existing_file(f, s, seq, Some(seq)).unwrap(),
                E::Deleted(f, seq) => w.add_delete_file(f, seq, Some(seq)).unwrap(),
            }
        }
        self.env.rt.block_on(w.write_manifest_file()).unwrap()
    }

    /// Snapshot `id` of `op` listing `manifests` (sequence number `id`); the
    /// manifests as listed (sequence numbers assigned).
    fn snapshot(
        &mut self,
        id: i64,
        parent: Option<i64>,
        op: &str,
        manifests: Vec<ManifestFile>,
    ) -> Vec<ManifestFile> {
        let io = self.env.reg.file_io();
        let list = self.env.at(&format!("snap-{id}.avro"));
        self.env.rt.block_on(async {
            let mut w = ManifestListWriter::v2(
                io.new_output(&list).unwrap().writer().await.unwrap(),
                id,
                parent,
                id,
            );
            w.add_manifests(manifests.into_iter()).unwrap();
            w.close().await.unwrap();
        });
        self.snapshots.push(serde_json::json!({
            "snapshot-id": id,
            "parent-snapshot-id": parent,
            "sequence-number": id,
            "timestamp-ms": self.meta.last_updated_ms() + id,
            "manifest-list": list,
            "summary": {"operation": op},
            "schema-id": 0
        }));
        self.env
            .rt
            .block_on(read_manifest_files(&io, &list, 2, MAX))
            .unwrap()
    }

    /// The table with `main` at `head`.
    fn table(&self, head: i64) -> TableMetadata {
        let mut json = serde_json::to_value(&self.meta).unwrap();
        let last = self
            .snapshots
            .iter()
            .map(|s| s["sequence-number"].as_i64().unwrap())
            .max()
            .unwrap();
        json["snapshots"] = serde_json::Value::Array(self.snapshots.clone());
        json["current-snapshot-id"] = serde_json::json!(head);
        json["last-sequence-number"] = serde_json::json!(last);
        json["last-updated-ms"] = serde_json::json!(self.meta.last_updated_ms() + last);
        json["refs"] = serde_json::json!({"main": {"snapshot-id": head, "type": "branch"}});
        serde_json::from_value(json).unwrap()
    }

    fn validate(&self, head: i64, s0: Option<i64>, v: &Validation) -> Option<Fired> {
        let io = self.env.reg.file_io();
        let m = self.table(head);
        self.env
            .rt
            .block_on(validate(&io, &m, Some(head), s0, v, MAX))
            .unwrap()
    }
}

/// S1 appends `old` (x=1); S2 (replace) adds `compacted` (x=1); S3
/// (overwrite) merges S1's and S2's manifests (both `EXISTING`), adds `new`
/// (x=2) and removes `gone` (x=1). Written by iceberg-rust's writers and read
/// back: only S3's own manifest is read, its `EXISTING` entries are not
/// changes, its `DELETED` entry is.
#[test]
fn the_history_reads_only_what_intervening_snapshots_added() {
    let mut fx = Fixture::new();
    let (old, compacted) = (fx.data("old.parquet", 1), fx.data("compacted.parquet", 1));
    let (gone, new) = (fx.data("gone.parquet", 1), fx.data("new.parquet", 2));
    let m1 = fx.manifest(
        1,
        false,
        vec![E::Added(old.clone()), E::Added(gone.clone())],
    );
    let l1 = fx.snapshot(1, None, "append", vec![m1]);
    let m2 = fx.manifest(2, false, vec![E::Added(compacted.clone())]);
    let mut l2 = l1;
    l2.push(m2);
    fx.snapshot(2, Some(1), "replace", l2);
    let merged = fx.manifest(
        3,
        false,
        vec![
            E::Added(new),
            E::Existing(old, 1, 1),
            E::Existing(compacted, 2, 2),
            E::Deleted(gone, 1),
        ],
    );
    fx.snapshot(3, Some(2), "overwrite", vec![merged]);

    // x = 1: S2's file is a replace, S3's merged x=1 files are EXISTING.
    assert_eq!(fx.validate(3, Some(1), &validation(ADDED_DATA)), None);
    // x = 2: S3's added file.
    let any = Validation {
        filter: Some(x_is(2)),
        ..validation(ADDED_DATA)
    };
    let f = fx.validate(3, Some(1), &any).unwrap();
    assert_eq!((f.check, f.snapshot_id), (CHECK_ADDED_DATA, Some(3)));
    assert!(
        f.message
            .ends_with(&format!("[{}]", fx.env.at("new.parquet")))
    );
    // From S0 = none, S1's own manifest is read: its x=1 files conflict.
    let f = fx.validate(3, None, &validation(ADDED_DATA)).unwrap();
    assert_eq!(f.snapshot_id, Some(1));
    // S3 removed `gone`.
    let removes = Validation {
        removes: BTreeSet::from([fx.env.at("gone.parquet")]),
        ..validation(REMOVED)
    };
    let f = fx.validate(3, Some(1), &removes).unwrap();
    assert_eq!((f.check, f.snapshot_id), (CHECK_REMOVED, Some(3)));
    // `old` is EXISTING in S3, not removed.
    let kept = Validation {
        removes: BTreeSet::from([fx.env.at("old.parquet")]),
        ..validation(REMOVED)
    };
    assert_eq!(fx.validate(3, Some(1), &kept), None);
    // A rollback: S0 = 2 is not an ancestor of a branch at 1.
    let f = fx.validate(1, Some(2), &validation(ADDED_DATA)).unwrap();
    assert_eq!(f.check, CHECK_HISTORY);
}

/// Delete manifests: S2 (delete) adds a position delete in x=1, S3
/// (overwrite) one in x=2; Java's sequence-number rule against S0's.
#[test]
fn the_history_reads_delete_manifests_by_sequence_number() {
    let mut fx = Fixture::new();
    let base = fx.data("base.parquet", 1);
    let m1 = fx.manifest(1, false, vec![E::Added(base)]);
    let l1 = fx.snapshot(1, None, "append", vec![m1]);
    let d2 = fx.pos_delete("d2.parquet", 1);
    let m2 = fx.manifest(2, true, vec![E::Added(d2)]);
    let mut l2 = l1;
    l2.push(m2);
    let l2 = fx.snapshot(2, Some(1), "delete", l2);
    let d3 = fx.pos_delete("d3.parquet", 2);
    let m3 = fx.manifest(3, true, vec![E::Added(d3)]);
    let mut l3 = l2;
    l3.push(m3);
    fx.snapshot(3, Some(2), "overwrite", l3);

    let f = fx.validate(3, Some(1), &validation(ADDED_DELETES)).unwrap();
    assert_eq!((f.check, f.snapshot_id), (CHECK_ADDED_DELETES, Some(2)));
    assert!(
        f.message
            .ends_with(&format!("[{}]", fx.env.at("d2.parquet")))
    );
    // From S0 = 2: only S3's, in x=2.
    assert_eq!(fx.validate(3, Some(2), &validation(ADDED_DELETES)), None);
    let x2 = Validation {
        filter: Some(x_is(2)),
        ..validation(ADDED_DELETES)
    };
    assert_eq!(fx.validate(3, Some(2), &x2).unwrap().snapshot_id, Some(3));
}

// ------------------------------------------------------ the commit loop

/// A foreign commit of `op` adding one row in partition `x` to `main` as
/// snapshot `id` (an append's request with its operation changed).
fn foreign_op(c: &Case, name: &str, x: i32, id: i64, op: &'static str) -> Foreign {
    let f = file(&c.env, &c.base, name, x, &[(id, name)]);
    let io = c.env.reg.file_io();
    let clock = Arc::clone(&c.clock);
    Box::new(move |m: TableMetadata| {
        Box::pin(async move {
            let uuid = format!("uuid-{id}");
            let mut new = write_new_manifests(&io, &m, vec![f], id, &uuid)
                .await
                .unwrap();
            let ts = clock.now_ms().max(m.last_updated_ms() + 1);
            let ids = WriteIds {
                branch: "main",
                write_id: &uuid,
                query_id: None,
            };
            let a = prepare_attempt(&io, &m, &mut new, ids, ts, MAX)
                .await
                .unwrap();
            let updates = a
                .updates
                .into_iter()
                .map(|u| {
                    let mut json = serde_json::to_value(&u).unwrap();
                    if json["action"] == "add-snapshot" {
                        json["snapshot"]["summary"]["operation"] = op.into();
                    }
                    serde_json::from_value::<TableUpdate>(json).unwrap()
                })
                .collect();
            (a.requirements, updates)
        })
    })
}

/// The write's validation: added data files matching `x = 1`.
fn loop_validation() -> Validation {
    validation(ADDED_DATA)
}

/// A foreign overwrite in the filter's partition lands first: the rebase
/// fails with the conflict class naming the foreign snapshot and the check,
/// and every request's outcome is known, so the write's files are removed
/// (§7.4).
#[test]
fn a_validated_commit_conflicts_and_removes_its_files() {
    let mut c = case();
    c.cat.on(
        When::BeforeRequest(1),
        foreign_op(&c, "f.parquet", 1, 900, "overwrite"),
    );
    let e = c.run_validated(&loop_validation()).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    let text = e.message();
    assert!(text.contains(&OURS.to_string()), "{text}");
    assert!(text.contains("snapshot 900"), "{text}");
    assert!(text.contains(CHECK_ADDED_DATA), "{text}");
    assert!(
        text.contains("Found conflicting files that can contain records matching"),
        "{text}"
    );
    assert!(text.contains("removed"), "{text}");
    assert_eq!(c.cat.requests(), 1);
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    c.check(&[INIT, "900|1|f.parquet"]);
}

/// A foreign overwrite elsewhere and a foreign compaction in the filter's
/// partition: nothing conflicts, the write rebases and commits.
#[test]
fn a_validation_that_does_not_fire_rebases_and_commits() {
    let mut c = case();
    c.cat.on(
        When::BeforeRequest(1),
        foreign_op(&c, "g.parquet", 3, 901, "overwrite"),
    );
    c.cat.on(
        When::BeforeRequest(1),
        foreign_op(&c, "h.parquet", 1, 902, "replace"),
    );
    let m = c.run_validated(&loop_validation()).unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(
        m.snapshot_by_id(OURS).unwrap().parent_snapshot_id(),
        Some(902)
    );
    assert_eq!(c.cat.requests(), 2);
    c.check(&[INIT, MINE, "901|3|g.parquet", "902|1|h.parquet"]);
}

/// A plain append (no validation) is unchanged by a foreign overwrite in any
/// partition.
#[test]
fn an_append_still_skips_validation() {
    let mut c = case();
    c.cat.on(
        When::BeforeRequest(1),
        foreign_op(&c, "f.parquet", 1, 900, "overwrite"),
    );
    c.run().unwrap();
    c.check(&[INIT, MINE, "900|1|f.parquet"]);
}

/// The bound through the loop: more intervening snapshots than
/// `max_intervening` is a conflict.
#[test]
fn too_many_intervening_snapshots_conflict_in_the_loop() {
    let mut c = case();
    c.cat.on(
        When::BeforeRequest(1),
        foreign_op(&c, "g.parquet", 3, 901, "overwrite"),
    );
    c.cat.on(
        When::BeforeRequest(1),
        foreign_op(&c, "h.parquet", 3, 902, "overwrite"),
    );
    let v = Validation {
        max_intervening: 1,
        ..loop_validation()
    };
    let e = c.run_validated(&v).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(e.message().contains(CHECK_HISTORY), "{e}");
    assert!(!exists(&c.data()));
}

/// The probe's path: a 503 before the apply (unknown), a foreign overwrite
/// in the filter's partition lands, the probe sees the branch moved and the
/// rebase validates: a conflict. A request had an unknown answer, so the
/// write's files are kept (`commit.rs`, review-1 pass 2 P1).
#[test]
fn a_moved_branch_after_an_unknown_answer_is_validated() {
    let mut c = case();
    c.cat.script(&[Fault::ServerErrorBeforeApply]);
    c.cat.on(
        When::AfterRequest(1),
        foreign_op(&c, "f.parquet", 1, 900, "overwrite"),
    );
    let e = c.run_validated(&loop_validation()).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(e.message().contains(CHECK_ADDED_DATA), "{e}");
    assert!(e.message().contains("kept"), "{e}");
    assert_eq!(c.cat.requests(), 1);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[INIT, "900|1|f.parquet"]);
}
