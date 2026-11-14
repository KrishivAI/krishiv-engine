// SPDX-License-Identifier: Apache-2.0
//! Tests of manifest merging on append (`merge`, PR-36a; design note §6.2):
//! appends committed as a catalog applies a request (`append_tests::commit`),
//! the list and every merged manifest read back through iceberg-rust's
//! parsers, and the table read through SQL and iceberg-rust's planning.

use std::collections::{HashMap, HashSet};

use arrow_schema::DataType as A;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Manifest,
    ManifestFile, ManifestListWriter, ManifestStatus, ManifestWriterBuilder, Operation, Snapshot,
    Summary, TableMetadata, TableMetadataBuilder, Transform, UnboundPartitionSpec,
};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::ErrorClass;

use super::append::{
    Attempt, MANIFESTS_CREATED, MANIFESTS_KEPT, MANIFESTS_REPLACED, NewManifests, prepare_attempt,
    write_new_manifests,
};
use super::append_tests::{MAX, commit, create, file, iceberg_rust_files, ids, now, rows};
use super::data_file::partition_tuple;
use super::merge::{
    MANIFEST_MERGE_ENABLED, MANIFEST_MIN_MERGE_COUNT, MANIFEST_TARGET_SIZE_BYTES, MergeConfig,
    pack_end, with_first_row_id,
};
use super::provider_tests::{Env, batch, env, id_field, int64s, strings, write_parquet};
use super::scan::{parse_manifest, read_manifest_files, read_manifest_list};
use crate::plan::entry::{LiveEntry, inherit};
use crate::plan::{DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID};

/// A table, its metadata and metadata location, and the commit count.
struct Table {
    env: Env,
    m: TableMetadata,
    loc: String,
    n: u32,
}

impl Table {
    fn new(version: u8) -> Self {
        let env = env();
        let t = create(&env, version);
        let m = t.metadata().clone();
        let loc = t.metadata_location().unwrap().to_owned();
        Self { env, m, loc, n: 0 }
    }

    /// Sets table properties (they persist with the next commit).
    fn set(&mut self, props: &[(&str, &str)]) {
        self.m = TableMetadataBuilder::new_from_metadata(self.m.clone(), None)
            .set_properties(
                props
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                    .collect(),
            )
            .unwrap()
            .build()
            .unwrap()
            .metadata;
    }

    /// A one-file write (`name`, rows `rows` in partition `x`) of snapshot
    /// `id`, its manifests written, not yet committed.
    fn write(&self, name: &str, x: i32, rows: &[(i64, &str)], id: i64) -> NewManifests {
        let f = file(&self.env, &self.m, name, x, rows);
        self.write_files(vec![f], id)
    }

    fn write_files(&self, files: Vec<(i32, DataFile)>, id: i64) -> NewManifests {
        let io = self.env.reg.file_io();
        self.env
            .rt
            .block_on(write_new_manifests(
                &io,
                &self.m,
                files,
                id,
                &format!("uuid-{id}"),
            ))
            .unwrap()
    }

    /// Prepares the next attempt of `new` and commits it.
    fn commit(&mut self, new: &mut NewManifests) -> Attempt {
        let io = self.env.reg.file_io();
        let w = format!("w-{}", new.snapshot_id);
        let a = self
            .env
            .rt
            .block_on(prepare_attempt(
                &io,
                &self.m,
                new,
                ids(&w),
                now(&self.m),
                MAX,
            ))
            .unwrap();
        self.apply(&a);
        a
    }

    fn apply(&mut self, a: &Attempt) {
        self.n += 1;
        let (m, loc) = commit(&self.env, &self.m, &self.loc, a, self.n);
        self.m = m;
        self.loc = loc;
    }

    /// One append of one file.
    fn append(&mut self, name: &str, x: i32, rows: &[(i64, &str)], id: i64) -> NewManifests {
        let mut new = self.write(name, x, rows, id);
        self.commit(&mut new);
        new
    }

    /// The current snapshot's manifest list.
    fn listed(&self) -> Vec<ManifestFile> {
        let io = self.env.reg.file_io();
        let s = self.m.current_snapshot().unwrap();
        self.env
            .rt
            .block_on(read_manifest_files(
                &io,
                s.manifest_list(),
                self.m.format_version() as u8,
                MAX,
            ))
            .unwrap()
    }

    fn summary(&self) -> &HashMap<String, String> {
        &self
            .m
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties
    }

    /// The manifest counts of the current snapshot's summary.
    fn counts(&self) -> [&str; 3] {
        let p = self.summary();
        [MANIFESTS_CREATED, MANIFESTS_KEPT, MANIFESTS_REPLACED].map(|k| p[k].as_str())
    }

    /// Every live entry of the current snapshot after inheritance: (file
    /// name, snapshot id, data and file sequence numbers, first row id).
    fn live(&self) -> Vec<Live> {
        let io = self.env.reg.file_io();
        let s = self.m.current_snapshot().unwrap();
        let version = self.m.format_version() as u8;
        let mut out = Vec::new();
        self.env.rt.block_on(async {
            for mf in read_manifest_list(&io, s.manifest_list(), version, MAX)
                .await
                .unwrap()
            {
                let bytes = io.new_input(&mf.path).unwrap().read().await.unwrap();
                for e in inherit(&mf, parse_manifest(&bytes, mf.spec_id).unwrap()).unwrap() {
                    let LiveEntry::Data(d) = e else { continue };
                    out.push((
                        name(&d.file.path),
                        d.snapshot_id,
                        d.data_sequence_number,
                        d.file_sequence_number,
                        d.file.first_row_id,
                    ));
                }
            }
        });
        out
    }

    /// Every live file listed once, and the rows `expected` through SQL and
    /// the same files through iceberg-rust's planning.
    fn check(&self, expected: &[&str]) {
        let files = iceberg_rust_files(&self.env, &self.m, &self.loc);
        let unique: HashSet<&String> = files.iter().collect();
        assert_eq!(unique.len(), files.len(), "a file listed twice: {files:?}");
        assert_eq!(files.len(), self.live().len());
        assert_eq!(rows(&self.env, &self.loc), expected);
    }
}

/// A live entry: (file name, snapshot id, data and file sequence numbers,
/// first row id).
type Live = (String, i64, Option<i64>, Option<i64>, Option<i64>);

/// The last path segment.
fn name(path: &str) -> String {
    path.rsplit('/').next().unwrap().to_owned()
}

/// A manifest's raw entries through iceberg-rust's `Manifest::parse_avro`:
/// (file name, status, snapshot id, data and file sequence numbers, first
/// row id).
type Raw = (
    String,
    ManifestStatus,
    Option<i64>,
    Option<i64>,
    Option<i64>,
    Option<i64>,
);

fn raw(env: &Env, path: &str) -> Vec<Raw> {
    let io = env.reg.file_io();
    let bytes = env
        .rt
        .block_on(async { io.new_input(path).unwrap().read().await })
        .unwrap();
    let (entries, _) = Manifest::parse_avro(&bytes).unwrap().into_parts();
    entries
        .iter()
        .map(|e| {
            (
                name(e.file_path()),
                e.status(),
                e.snapshot_id(),
                e.sequence_number(),
                e.file_sequence_number,
                e.data_file().first_row_id(),
            )
        })
        .collect()
}

fn exists(url: &str) -> bool {
    std::path::Path::new(url.strip_prefix("file://").unwrap()).exists()
}

#[test]
fn pack_end_packs_from_the_end_with_one_open_bin() {
    // Java's `ListPacker(10, 1, false).packEnd`: from the end, so the first
    // bin is the under-filled one.
    assert_eq!(
        pack_end(&[4, 4, 4, 4, 4], 10),
        [vec![0], vec![1, 2], vec![3, 4]]
    );
    // An item above the target is alone, and so are its neighbours' bins.
    assert_eq!(pack_end(&[3, 20, 3, 3], 10), [vec![0], vec![1], vec![2, 3]]);
    // Lookback 1: a later item never goes back into a closed bin.
    assert_eq!(pack_end(&[1, 9, 2], 10), [vec![0, 1], vec![2]]);
    assert_eq!(pack_end(&[5, 5], 10), [vec![0, 1]]);
    assert!(pack_end(&[], 10).is_empty());
    // A negative or zero target puts every item alone.
    assert_eq!(pack_end(&[1, 1], 0), [vec![0], vec![1]]);
    assert_eq!(
        pack_end(&[i64::MAX, i64::MAX], i64::MAX),
        [vec![0], vec![1]]
    );
}

#[test]
fn merge_properties_parse_as_java_and_refuse_garbage() {
    let props = |kv: &[(&str, &str)]| -> HashMap<String, String> {
        kv.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    };
    assert_eq!(
        MergeConfig::from_properties(&props(&[])).unwrap(),
        MergeConfig {
            target_size_bytes: 8_388_608,
            min_count_to_merge: 100,
            enabled: true,
        }
    );
    assert_eq!(
        MergeConfig::from_properties(&props(&[
            (MANIFEST_TARGET_SIZE_BYTES, "1024"),
            (MANIFEST_MIN_MERGE_COUNT, "2"),
            (MANIFEST_MERGE_ENABLED, "FALSE"),
        ]))
        .unwrap(),
        MergeConfig {
            target_size_bytes: 1024,
            min_count_to_merge: 2,
            enabled: false,
        }
    );
    for (k, v) in [
        (MANIFEST_TARGET_SIZE_BYTES, "8MB"),
        (MANIFEST_MIN_MERGE_COUNT, " 4"),
        (MANIFEST_MIN_MERGE_COUNT, "3000000000"),
        (MANIFEST_MERGE_ENABLED, "yes"),
    ] {
        let e = MergeConfig::from_properties(&props(&[(k, v)])).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        );
        assert!(e.to_string().contains(k), "{e}");
    }
}

/// A file rebuilt with its own `first_row_id` is equal to it: no field is
/// lost by the copy through the builder.
#[test]
fn with_first_row_id_copies_every_field() {
    let t = Table::new(3);
    let (spec, f) = file(&t.env, &t.m, "a.parquet", 1, &[(1, "a")]);
    assert_eq!(with_first_row_id(&f, None, spec).unwrap(), f);
    let g = with_first_row_id(&f, Some(7), spec).unwrap();
    assert_eq!(g.first_row_id(), Some(7));
    assert_eq!(with_first_row_id(&g, None, spec).unwrap(), f);
}

/// 100 one-file appends (Java's defaults): 99 manifests stay apart (the bin
/// holding the new manifest has fewer than `min-count-to-merge`); the 100th
/// append merges all 100 into one manifest of its snapshot: its own file
/// `ADDED` with inherited sequence numbers, the others `EXISTING` with their
/// own snapshot ids and explicit sequence numbers.
#[test]
fn a_hundred_appends_merge_into_one_manifest() {
    let mut t = Table::new(2);
    let mut expected = Vec::new();
    for i in 1..=100_i64 {
        let n = format!("f{i}.parquet");
        t.append(&n, 1, &[(i, "r")], 1000 + i);
        expected.push(format!("{i}|1|r"));
        if i == 99 {
            assert_eq!(t.listed().len(), 99);
            assert_eq!(t.counts(), ["1", "98", "0"]);
        }
    }
    let listed = t.listed();
    assert_eq!(listed.len(), 1, "{listed:?}");
    let merged = &listed[0];
    assert_eq!(merged.added_snapshot_id, 1100);
    assert_eq!(merged.sequence_number, 100);
    assert_eq!(merged.min_sequence_number, 1);
    assert_eq!(
        (merged.added_files_count, merged.existing_files_count),
        (Some(1), Some(99))
    );
    assert!(
        merged
            .manifest_path
            .ends_with("/metadata/uuid-1100-m1.avro")
    );
    assert_eq!(t.counts(), ["1", "0", "99"]);
    let entries = raw(&t.env, &merged.manifest_path);
    assert_eq!(
        entries[0],
        (
            "f100.parquet".to_owned(),
            ManifestStatus::Added,
            Some(1100),
            None,
            None,
            None
        )
    );
    // The base list is newest first, so the existing entries are too.
    for (k, e) in entries[1..].iter().enumerate() {
        let i = 99 - i64::try_from(k).unwrap();
        assert_eq!(
            e,
            &(
                format!("f{i}.parquet"),
                ManifestStatus::Existing,
                Some(1000 + i),
                Some(i),
                Some(i),
                None
            )
        );
    }
    // The absorbed manifests stay (they belong to earlier snapshots).
    assert!(exists(&t.env.at("db/t/metadata/uuid-1001-m0.avro")));
    let refs: Vec<&str> = expected.iter().map(String::as_str).collect();
    t.check(&refs);
}

/// Below `min-count-to-merge` the bin holding the new manifest is kept;
/// `commit.manifest-merge.enabled=false` never merges; a target below every
/// manifest's length keeps each alone.
#[test]
fn merging_needs_the_count_the_switch_and_room() {
    for (props, merged) in [
        (vec![(MANIFEST_MIN_MERGE_COUNT, "4")], false),
        (vec![(MANIFEST_MIN_MERGE_COUNT, "3")], true),
        (
            vec![
                (MANIFEST_MIN_MERGE_COUNT, "2"),
                (MANIFEST_MERGE_ENABLED, "false"),
            ],
            false,
        ),
        (
            vec![
                (MANIFEST_MIN_MERGE_COUNT, "2"),
                (MANIFEST_TARGET_SIZE_BYTES, "1"),
            ],
            false,
        ),
    ] {
        let mut t = Table::new(2);
        t.append("a.parquet", 1, &[(1, "a")], 101);
        t.append("b.parquet", 2, &[(2, "b")], 102);
        t.set(&props);
        t.append("c.parquet", 1, &[(3, "c")], 103);
        let listed = t.listed();
        if merged {
            assert_eq!(listed.len(), 1, "{props:?}");
            assert_eq!(t.counts(), ["1", "0", "2"]);
        } else {
            assert_eq!(listed.len(), 3, "{props:?}");
            assert_eq!(t.counts(), ["1", "2", "0"], "{props:?}");
            assert_eq!(
                listed
                    .iter()
                    .map(|m| m.added_snapshot_id)
                    .collect::<Vec<_>>(),
                [103, 102, 101]
            );
        }
        t.check(&["1|1|a", "2|2|b", "3|1|c"]);
    }
}

/// `packEnd` order: with room for two manifests per bin, the new manifest
/// and the newest base one share the first (under-filled) bin and stay apart
/// (fewer than `min-count-to-merge`); the bin of the two oldest, without the
/// new manifest, merges at a count of 2.
#[test]
fn a_bin_without_the_new_manifest_merges_at_two() {
    let mut t = Table::new(2);
    t.append("a.parquet", 1, &[(1, "a")], 101);
    t.append("b.parquet", 1, &[(2, "b")], 102);
    t.append("c.parquet", 1, &[(3, "c")], 103);
    let mut new = t.write("d.parquet", 1, &[(4, "d")], 104);
    let mut lengths: Vec<i64> = t.listed().iter().map(|m| m.manifest_length).collect();
    lengths.push(new.manifests[0].manifest_length);
    let (min, max) = (
        *lengths.iter().min().unwrap(),
        *lengths.iter().max().unwrap(),
    );
    let target = 2 * max;
    assert!(3 * min > target, "{lengths:?}");
    t.set(&[(MANIFEST_TARGET_SIZE_BYTES, &target.to_string())]);
    t.commit(&mut new);
    let listed = t.listed();
    assert_eq!(
        listed
            .iter()
            .map(|m| m.added_snapshot_id)
            .collect::<Vec<_>>(),
        [104, 103, 104]
    );
    assert_eq!(
        listed[1].manifest_path,
        t.env.at("db/t/metadata/uuid-103-m0.avro")
    );
    let merged = raw(&t.env, &listed[2].manifest_path);
    assert_eq!(
        merged
            .iter()
            .map(|e| (e.0.as_str(), e.1, e.2, e.3))
            .collect::<Vec<_>>(),
        [
            ("b.parquet", ManifestStatus::Existing, Some(102), Some(2)),
            ("a.parquet", ManifestStatus::Existing, Some(101), Some(1)),
        ]
    );
    assert_eq!(t.counts(), ["2", "1", "2"]);
    t.check(&["1|1|a", "2|1|b", "3|1|c", "4|1|d"]);
}

/// Two partition specs merge separately, the highest spec id first, whatever
/// the input order. The first append after the spec change merges the old
/// spec's manifests (their bin lacks the new manifest, so 2 suffice); a later
/// append whose new manifest is of spec 0 still lists spec 1 first.
#[test]
fn two_specs_merge_separately_highest_id_first() {
    let mut t = Table::new(2);
    t.append("a.parquet", 1, &[(1, "a")], 101);
    t.append("b.parquet", 2, &[(2, "b")], 102);
    let spec0 = t.m.clone();
    let bucket = UnboundPartitionSpec::builder()
        .add_partition_field(2, "x_bucket", Transform::Bucket(4))
        .unwrap()
        .build();
    t.m = TableMetadataBuilder::new_from_metadata(t.m.clone(), None)
        .add_partition_spec(bucket)
        .unwrap()
        .set_default_partition_spec(-1)
        .unwrap()
        .build()
        .unwrap()
        .metadata;
    assert_eq!(t.m.default_partition_spec_id(), 1);
    let spec_and_ids = |t: &Table| -> Vec<(i32, i64)> {
        t.listed()
            .iter()
            .map(|m| (m.partition_spec_id, m.added_snapshot_id))
            .collect()
    };
    let names = |t: &Table, m: &ManifestFile| -> Vec<(String, ManifestStatus)> {
        raw(&t.env, &m.manifest_path)
            .into_iter()
            .map(|e| (e.0, e.1))
            .collect()
    };
    use ManifestStatus::{Added, Existing};
    t.append("c.parquet", 1, &[(3, "c")], 103);
    assert_eq!(spec_and_ids(&t), [(1, 103), (0, 103)]);
    let old = t.listed()[1].clone();
    assert_eq!(
        names(&t, &old),
        [
            ("b.parquet".to_owned(), Existing),
            ("a.parquet".to_owned(), Existing)
        ]
    );
    assert_eq!(t.counts(), ["2", "0", "2"]);

    // Input [e (spec 0), d (1), c (1), old (0)]: spec 1's [d, c] lacks the
    // new manifest and merges; spec 0's [e, old] holds it and has fewer
    // than 3 manifests.
    t.append("d.parquet", 2, &[(4, "d")], 104);
    assert_eq!(spec_and_ids(&t), [(1, 104), (1, 103), (0, 103)]);
    t.set(&[(MANIFEST_MIN_MERGE_COUNT, "3")]);
    let e = file(&t.env, &spec0, "e.parquet", 1, &[(5, "e")]);
    let mut new = t.write_files(vec![e], 105);
    assert_eq!(new.manifests[0].partition_spec_id, 0);
    t.commit(&mut new);
    assert_eq!(spec_and_ids(&t), [(1, 105), (0, 105), (0, 103)]);
    let listed = t.listed();
    assert_eq!(
        names(&t, &listed[0]),
        [
            ("d.parquet".to_owned(), Existing),
            ("c.parquet".to_owned(), Existing)
        ]
    );
    assert_eq!(names(&t, &listed[1]), [("e.parquet".to_owned(), Added)]);
    assert_eq!(listed[2].manifest_path, old.manifest_path);
    assert_eq!(t.counts(), ["2", "1", "2"]);
    t.check(&["1|1|a", "2|2|b", "3|1|c", "4|2|d", "5|1|e"]);
}

/// Review-1 pass 2, P2: a base manifest holding a `DELETED` entry of an
/// earlier snapshot. The merged manifest carries the live entry only, under
/// its own snapshot id; the removed file does not come back and is not
/// listed as removed by this write.
#[test]
fn a_merge_drops_deleted_entries_of_older_snapshots() {
    let mut t = Table::new(2);
    let a = file(&t.env, &t.m, "a.parquet", 1, &[(1, "a")]);
    let mut new_a = t.write_files(vec![a.clone()], 101);
    t.commit(&mut new_a);
    let b = file(&t.env, &t.m, "b.parquet", 1, &[(2, "b")]);
    let mut new_b = t.write_files(vec![b.clone()], 102);
    t.commit(&mut new_b);
    t.check(&["1|1|a", "2|1|b"]);

    // Snapshot 103 removes b: one manifest, a EXISTING, b DELETED.
    let io = t.env.reg.file_io();
    let path = t.env.at("db/t/metadata/hand-103.avro");
    let mut w = ManifestWriterBuilder::new(
        io.new_output(&path).unwrap(),
        Some(103),
        std::sync::Arc::clone(t.m.current_schema()),
        t.m.default_partition_spec().as_ref().clone(),
    )
    .build_v2_data();
    w.add_existing_file(a.1, 101, 1, Some(1)).unwrap();
    w.add_delete_file(b.1, 2, Some(2)).unwrap();
    let hand = t.env.rt.block_on(w.write_manifest_file()).unwrap();
    let list = t.env.at("db/t/metadata/snap-103-hand.avro");
    t.env.rt.block_on(async {
        let out = io.new_output(&list).unwrap().writer().await.unwrap();
        let mut l = ManifestListWriter::v2(out, 103, Some(102), 3);
        l.add_manifests(vec![hand].into_iter()).unwrap();
        l.close().await.unwrap();
    });
    let snapshot = Snapshot::builder()
        .with_snapshot_id(103)
        .with_parent_snapshot_id(Some(102))
        .with_sequence_number(3)
        .with_timestamp_ms(now(&t.m))
        .with_manifest_list(list)
        .with_schema_id(t.m.current_schema_id())
        .with_summary(Summary {
            operation: Operation::Delete,
            additional_properties: HashMap::new(),
        })
        .build();
    let delete = Attempt {
        snapshot: snapshot.clone(),
        requirements: vec![TableRequirement::UuidMatch { uuid: t.m.uuid() }],
        updates: vec![
            TableUpdate::AddSnapshot { snapshot },
            TableUpdate::SetSnapshotRef {
                ref_name: "main".to_owned(),
                reference: iceberg::spec::SnapshotReference::new(
                    103,
                    iceberg::spec::SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                ),
            },
        ],
    };
    t.apply(&delete);
    t.check(&["1|1|a"]);

    t.set(&[(MANIFEST_MIN_MERGE_COUNT, "2")]);
    t.append("c.parquet", 1, &[(3, "c")], 104);
    let listed = t.listed();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].deleted_files_count, Some(0));
    assert_eq!(
        raw(&t.env, &listed[0].manifest_path),
        [
            (
                "c.parquet".to_owned(),
                ManifestStatus::Added,
                Some(104),
                None,
                None,
                None
            ),
            (
                "a.parquet".to_owned(),
                ManifestStatus::Existing,
                Some(101),
                Some(1),
                Some(1),
                None
            ),
        ]
    );
    t.check(&["1|1|a", "3|1|c"]);
}

/// v3: the merged manifest carries an explicit `first_row_id` on every
/// `EXISTING` entry, so each file keeps its row ids; the new file gets fresh
/// ids from the merged manifest's range (which, as Java's list writer, also
/// spans the existing rows), and `next-row-id` advances past them.
#[test]
fn v3_merging_keeps_row_ids() {
    let mut t = Table::new(3);
    t.append("a.parquet", 1, &[(1, "a1"), (2, "a2")], 301);
    t.append("b.parquet", 2, &[(3, "b")], 302);
    let before: Vec<_> = t.live().into_iter().map(|e| (e.0, e.4)).collect();
    assert_eq!(
        before,
        [
            ("b.parquet".to_owned(), Some(2)),
            ("a.parquet".to_owned(), Some(0))
        ]
    );
    assert_eq!(t.m.next_row_id(), 3);
    t.set(&[(MANIFEST_MIN_MERGE_COUNT, "3")]);
    t.append("c.parquet", 1, &[(4, "c")], 303);
    let listed = t.listed();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].first_row_id, Some(3));
    let entries = raw(&t.env, &listed[0].manifest_path);
    assert_eq!(
        entries
            .iter()
            .map(|e| (e.0.as_str(), e.1, e.5))
            .collect::<Vec<_>>(),
        [
            ("c.parquet", ManifestStatus::Added, None),
            ("b.parquet", ManifestStatus::Existing, Some(2)),
            ("a.parquet", ManifestStatus::Existing, Some(0)),
        ]
    );
    let after: Vec<_> = t.live().into_iter().map(|e| (e.0, e.4)).collect();
    assert_eq!(
        after,
        [
            ("c.parquet".to_owned(), Some(3)),
            ("b.parquet".to_owned(), Some(2)),
            ("a.parquet".to_owned(), Some(0))
        ]
    );
    let s = t.m.current_snapshot().unwrap();
    assert_eq!((s.first_row_id(), s.added_rows_count()), (Some(3), Some(4)));
    assert_eq!(t.m.next_row_id(), 7);
    t.check(&["1|1|a1", "2|1|a2", "3|2|b", "4|1|c"]);
}

/// The first attempt merges; a later attempt of the same write lists the
/// unmerged manifests (its new manifest, then the base's) and writes no
/// merged manifest; the first attempt's merged manifest is recorded for
/// cleanup.
#[test]
fn only_the_first_attempt_merges() {
    let mut t = Table::new(2);
    t.append("a.parquet", 1, &[(1, "a")], 101);
    t.set(&[(MANIFEST_MIN_MERGE_COUNT, "2")]);
    let mut new = t.write("b.parquet", 1, &[(2, "b")], 102);
    let io = t.env.reg.file_io();
    let first = t
        .env
        .rt
        .block_on(prepare_attempt(
            &io,
            &t.m,
            &mut new,
            ids("w"),
            now(&t.m),
            MAX,
        ))
        .unwrap();
    assert_eq!(new.merged.len(), 1);
    assert!(new.merged[0].ends_with("/metadata/uuid-102-m1.avro"));
    let p = &first.snapshot.summary().additional_properties;
    assert_eq!(p[MANIFESTS_REPLACED], "1");
    let second = t
        .env
        .rt
        .block_on(prepare_attempt(
            &io,
            &t.m,
            &mut new,
            ids("w"),
            now(&t.m),
            MAX,
        ))
        .unwrap();
    assert_eq!(new.merged.len(), 1);
    let p = &second.snapshot.summary().additional_properties;
    assert_eq!(
        [MANIFESTS_CREATED, MANIFESTS_KEPT, MANIFESTS_REPLACED].map(|k| p[k].as_str()),
        ["1", "1", "0"]
    );
    t.apply(&second);
    let listed = t.listed();
    assert_eq!(
        listed
            .iter()
            .map(|m| m.manifest_path.clone())
            .collect::<Vec<_>>(),
        [
            t.env.at("db/t/metadata/uuid-102-m0.avro"),
            t.env.at("db/t/metadata/uuid-101-m0.avro")
        ]
    );
    t.check(&["1|1|a", "2|1|b"]);
}

/// A position delete file in partition `x = 1` deleting `rows` (data file
/// name, position).
fn pos_delete(t: &Table, name: &str, rows: &[(&str, i64)]) -> DataFile {
    let paths: Vec<String> = rows.iter().map(|(n, _)| t.env.at(n)).collect();
    let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
    let pos: Vec<i64> = rows.iter().map(|r| r.1).collect();
    let b = batch(
        vec![
            id_field("file_path", A::Utf8, DELETE_FILE_PATH_ID),
            id_field("pos", A::Int64, DELETE_FILE_POS_ID),
        ],
        vec![strings(&paths), int64s(&pos)],
    );
    let path = t.env.at(name);
    let size = write_parquet(&path, &b);
    let spec = t.m.default_partition_spec();
    let partition = partition_tuple(
        spec,
        t.m.current_schema(),
        &[Some(krishiv_io::ScalarValue::Int32(1))],
    )
    .unwrap();
    let mut d = DataFileBuilder::default();
    d.content(DataContentType::PositionDeletes)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(u64::try_from(rows.len()).unwrap())
        .partition(partition)
        .partition_spec_id(spec.spec_id());
    d.build().unwrap()
}

/// A base snapshot (102) listing two delete manifests, a manifest with
/// neither added nor existing files, and the data manifest, in that order.
/// An append lists its data manifests first, leaves the empty manifest out
/// (Java's `shouldKeep`; the file stays) and the delete manifests last; with
/// `min-count-to-merge` 2, data and delete manifests merge separately, the
/// delete entries `EXISTING` with their own sequence numbers.
#[test]
fn empty_base_manifests_are_dropped_and_deletes_merge_apart() {
    let mut t = Table::new(2);
    t.append("a.parquet", 1, &[(1, "a1"), (2, "a2"), (3, "a3")], 101);
    let data_manifest = t.listed().remove(0);
    let io = t.env.reg.file_io();
    let schema = std::sync::Arc::clone(t.m.current_schema());
    let spec = t.m.default_partition_spec().as_ref().clone();
    let write = |path: &str, deletes: bool, f: &dyn Fn(&mut iceberg::spec::ManifestWriter)| {
        let b = ManifestWriterBuilder::new(
            io.new_output(path).unwrap(),
            Some(102),
            std::sync::Arc::clone(&schema),
            spec.clone(),
        );
        let mut w = if deletes {
            b.build_v2_deletes()
        } else {
            b.build_v2_data()
        };
        f(&mut w);
        t.env.rt.block_on(w.write_manifest_file()).unwrap()
    };
    let d1 = pos_delete(&t, "d1.parquet", &[("a.parquet", 0)]);
    let d2 = pos_delete(&t, "d2.parquet", &[("a.parquet", 2)]);
    let m1 = write(&t.env.at("db/t/metadata/hand-d1.avro"), true, &|w| {
        w.add_file(d1.clone(), 2).unwrap();
    });
    let m2 = write(&t.env.at("db/t/metadata/hand-d2.avro"), true, &|w| {
        w.add_file(d2.clone(), 2).unwrap();
    });
    let (_, gone) = file(&t.env, &t.m, "gone.parquet", 1, &[(9, "gone")]);
    let empty_path = t.env.at("db/t/metadata/hand-empty.avro");
    let empty = write(&empty_path, false, &|w| {
        w.add_delete_file(gone.clone(), 1, Some(1)).unwrap();
    });
    let list = t.env.at("db/t/metadata/snap-102-hand.avro");
    t.env.rt.block_on(async {
        let out = io.new_output(&list).unwrap().writer().await.unwrap();
        let mut l = ManifestListWriter::v2(out, 102, Some(101), 2);
        l.add_manifests(vec![m1, m2, empty, data_manifest].into_iter())
            .unwrap();
        l.close().await.unwrap();
    });
    let snapshot = Snapshot::builder()
        .with_snapshot_id(102)
        .with_parent_snapshot_id(Some(101))
        .with_sequence_number(2)
        .with_timestamp_ms(now(&t.m))
        .with_manifest_list(list)
        .with_schema_id(t.m.current_schema_id())
        .with_summary(Summary {
            operation: Operation::Delete,
            additional_properties: HashMap::new(),
        })
        .build();
    t.apply(&Attempt {
        snapshot: snapshot.clone(),
        requirements: vec![],
        updates: vec![
            TableUpdate::AddSnapshot { snapshot },
            TableUpdate::SetSnapshotRef {
                ref_name: "main".to_owned(),
                reference: iceberg::spec::SnapshotReference::new(
                    102,
                    iceberg::spec::SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                ),
            },
        ],
    });
    assert_eq!(t.listed().len(), 4);
    t.check(&["2|1|a2"]);

    t.append("b.parquet", 1, &[(4, "b")], 103);
    let listed = t.listed();
    assert_eq!(
        listed
            .iter()
            .map(|m| name(&m.manifest_path))
            .collect::<Vec<_>>(),
        [
            "uuid-103-m0.avro",
            "uuid-101-m0.avro",
            "hand-d1.avro",
            "hand-d2.avro"
        ]
    );
    assert!(exists(&empty_path));
    assert_eq!(t.counts(), ["1", "3", "0"]);
    t.check(&["2|1|a2", "4|1|b"]);

    t.set(&[(MANIFEST_MIN_MERGE_COUNT, "2")]);
    t.append("c.parquet", 1, &[(5, "c")], 104);
    let listed = t.listed();
    assert_eq!(
        listed
            .iter()
            .map(|m| (m.content, m.added_snapshot_id))
            .collect::<Vec<_>>(),
        [
            (iceberg::spec::ManifestContentType::Data, 104),
            (iceberg::spec::ManifestContentType::Deletes, 104)
        ]
    );
    assert_eq!(
        raw(&t.env, &listed[1].manifest_path),
        [
            (
                "d1.parquet".to_owned(),
                ManifestStatus::Existing,
                Some(102),
                Some(2),
                Some(2),
                None
            ),
            (
                "d2.parquet".to_owned(),
                ManifestStatus::Existing,
                Some(102),
                Some(2),
                Some(2),
                None
            ),
        ]
    );
    assert_eq!(t.counts(), ["2", "0", "4"]);
    t.check(&["2|1|a2", "4|1|b", "5|1|c"]);
}

/// A merge on a v1 table writes a v1 manifest whose entries read back with
/// sequence number 0.
#[test]
fn v1_tables_merge() {
    let mut t = Table::new(1);
    t.append("a.parquet", 1, &[(1, "a")], 101);
    t.set(&[(MANIFEST_MIN_MERGE_COUNT, "2")]);
    t.append("b.parquet", 2, &[(2, "b")], 102);
    let listed = t.listed();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].added_snapshot_id, 102);
    assert_eq!(
        t.live()
            .into_iter()
            .map(|e| (e.0, e.1, e.2))
            .collect::<Vec<_>>(),
        [
            ("b.parquet".to_owned(), 102, Some(0)),
            ("a.parquet".to_owned(), 101, Some(0))
        ]
    );
    assert_eq!(t.m.format_version(), FormatVersion::V1);
    t.check(&["1|1|a", "2|2|b"]);
}
