// SPDX-License-Identifier: Apache-2.0
//! Tests of the overwrite's manifest filter (`filter`, PR-36b part 2a):
//! manifests written by iceberg-rust's writers with chosen statuses, snapshot
//! ids and sequence numbers; a hit rewritten from live entries only, a miss
//! kept as is, duplicates, missing paths, dangling DVs, delete files older
//! than every data file, the overwrite summary, and a row filter's whole
//! and partial matches.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Manifest, ManifestContentType,
    ManifestFile, ManifestStatus, ManifestWriterBuilder, Struct, TableMetadata,
};

use super::append_tests::{MAX, create, file};
use super::filter::{
    DeleteKey, FilterContext, FilterError, Filtered, Removals, filter_manifests, overwrite_summary,
};
use super::provider_tests::{Env, env};

const OURS: i64 = 900;

/// One entry: status, file, snapshot id, data and file sequence numbers.
type Entry = (ManifestStatus, DataFile, i64, i64);

/// A manifest of `entries` added by `snapshot` at sequence number `seq`.
fn manifest(
    env: &Env,
    m: &TableMetadata,
    name: &str,
    content: ManifestContentType,
    snapshot: i64,
    seq: i64,
    entries: Vec<Entry>,
) -> ManifestFile {
    let io = env.reg.file_io();
    let b = ManifestWriterBuilder::new(
        io.new_output(env.at(name)).unwrap(),
        Some(snapshot),
        Arc::clone(m.current_schema()),
        m.default_partition_spec().as_ref().clone(),
    );
    let mut w = match (m.format_version() as u8, content) {
        (3, ManifestContentType::Data) => b.build_v3_data(),
        (3, ManifestContentType::Deletes) => b.build_v3_deletes(),
        (_, ManifestContentType::Data) => b.build_v2_data(),
        (_, ManifestContentType::Deletes) => b.build_v2_deletes(),
    };
    for (status, f, id, s) in entries {
        match status {
            ManifestStatus::Added => w.add_file(f, s).unwrap(),
            ManifestStatus::Existing => w.add_existing_file(f, id, s, Some(s)).unwrap(),
            ManifestStatus::Deleted => w.add_delete_file(f, s, Some(s)).unwrap(),
        }
    }
    let mut mf = env.rt.block_on(w.write_manifest_file()).unwrap();
    mf.sequence_number = seq;
    if mf.min_sequence_number < 0 {
        mf.min_sequence_number = seq;
    }
    if m.format_version() as u8 == 3 && content == ManifestContentType::Data {
        mf.first_row_id = Some(100);
    }
    mf
}

fn run(
    env: &Env,
    m: &TableMetadata,
    base: Vec<ManifestFile>,
    removals: &Removals,
    written: &mut Vec<String>,
) -> Result<Filtered, FilterError> {
    let io = env.reg.file_io();
    env.rt.block_on(filter_manifests(
        &mut FilterContext {
            io: &io,
            metadata: m,
            snapshot_id: OURS,
            prefix: env.at("ours-m"),
            next: 0,
            written,
            max_metadata_bytes: MAX,
        },
        base,
        removals,
    ))
}

/// The entries of `m`: status, path, snapshot id, data sequence number and
/// `first_row_id`.
fn entries(env: &Env, m: &ManifestFile) -> Vec<(ManifestStatus, String, i64, i64, Option<i64>)> {
    let io = env.reg.file_io();
    let bytes = env
        .rt
        .block_on(async { io.new_input(&m.manifest_path).unwrap().read().await })
        .unwrap();
    let (entries, _) = Manifest::parse_avro(&bytes).unwrap().into_parts();
    entries
        .iter()
        .map(|e| {
            (
                e.status(),
                e.file_path().rsplit('/').next().unwrap().to_owned(),
                e.snapshot_id().unwrap(),
                e.sequence_number().unwrap(),
                e.data_file().first_row_id(),
            )
        })
        .collect()
}

fn removals(data: &[&str], env: &Env) -> Removals {
    Removals {
        data_files: data.iter().map(|n| env.at(n)).collect::<BTreeSet<_>>(),
        delete_files: BTreeSet::new(),
        ..Removals::default()
    }
}

fn data(env: &Env, m: &TableMetadata, name: &str, rows: usize) -> DataFile {
    let r: Vec<(i64, &str)> = (0..rows).map(|i| (i as i64, "a")).collect();
    file(env, m, name, 1, &r).1
}

fn delete_file(env: &Env, name: &str, dv_of: Option<&str>, partition: Struct) -> DataFile {
    let mut b = DataFileBuilder::default();
    b.content(DataContentType::PositionDeletes)
        .file_path(env.at(name))
        .file_size_in_bytes(500)
        .record_count(2)
        .partition(partition)
        .partition_spec_id(0);
    match dv_of {
        Some(d) => {
            b.file_format(DataFileFormat::Puffin)
                .referenced_data_file(Some(env.at(d)))
                .content_offset(Some(4))
                .content_size_in_bytes(Some(40));
        }
        None => {
            b.file_format(DataFileFormat::Parquet);
        }
    }
    b.build().unwrap()
}

/// A hit manifest is rewritten from its live entries (the older `DELETED`
/// entry dropped), the hit `DELETED` under our id with its sequence number,
/// the rest `EXISTING` with their own ids and sequence numbers; a manifest
/// without a hit is kept as it is; the summary counts the removed file.
#[test]
fn a_hit_manifest_is_rewritten_from_live_entries_and_a_miss_is_kept() {
    let env = env();
    let t = create(&env, 2);
    let m = t.metadata().clone();
    let a = data(&env, &m, "a.parquet", 3);
    let b = data(&env, &m, "b.parquet", 2);
    let gone = data(&env, &m, "gone.parquet", 1);
    let c = data(&env, &m, "c.parquet", 4);
    let hit = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        2,
        vec![
            (ManifestStatus::Existing, a.clone(), 10, 1),
            (ManifestStatus::Added, b, 20, 2),
            (ManifestStatus::Deleted, gone, 20, 1),
        ],
    );
    let miss = manifest(
        &env,
        &m,
        "m2.avro",
        ManifestContentType::Data,
        30,
        3,
        vec![(ManifestStatus::Added, c, 30, 3)],
    );
    let mut written = Vec::new();
    let f = run(
        &env,
        &m,
        vec![hit, miss.clone()],
        &removals(&["a.parquet"], &env),
        &mut written,
    )
    .unwrap();
    assert_eq!(f.replaced, 1);
    assert_eq!(written, [env.at("ours-m0.avro")]);
    assert_eq!(f.manifests[1], miss);
    assert_eq!(f.manifests[0].manifest_path, env.at("ours-m0.avro"));
    assert_eq!(f.manifests[0].added_snapshot_id, OURS);
    assert_eq!(
        entries(&env, &f.manifests[0]),
        [
            (
                ManifestStatus::Deleted,
                "a.parquet".to_owned(),
                OURS,
                1,
                None
            ),
            (
                ManifestStatus::Existing,
                "b.parquet".to_owned(),
                20,
                2,
                None
            ),
        ]
    );
    let summary = overwrite_summary(&HashMap::new(), &HashSet::new(), &f.removed, 0);
    assert_eq!(
        summary.get("deleted-data-files").map(String::as_str),
        Some("1")
    );
    assert_eq!(
        summary.get("deleted-records").map(String::as_str),
        Some("3")
    );
    assert_eq!(
        summary.get("removed-files-size"),
        Some(&a.file_size_in_bytes().to_string())
    );
    assert_eq!(
        summary.get("changed-partition-count").map(String::as_str),
        Some("1")
    );
    assert!(!summary.contains_key("deleted-duplicate-files"));
}

/// v3: every rewritten data entry carries an explicit `first_row_id`
/// (inherited in order from the manifest's), so files keep their row ids.
#[test]
fn a_v3_rewrite_keeps_row_ids() {
    let env = env();
    let t = create(&env, 3);
    let m = t.metadata().clone();
    let a = data(&env, &m, "a.parquet", 3);
    let b = data(&env, &m, "b.parquet", 2);
    let hit = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        2,
        vec![
            (ManifestStatus::Added, a, 20, 2),
            (ManifestStatus::Added, b, 20, 2),
        ],
    );
    let f = run(
        &env,
        &m,
        vec![hit],
        &removals(&["a.parquet"], &env),
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(
        entries(&env, &f.manifests[0]),
        [
            (
                ManifestStatus::Deleted,
                "a.parquet".to_owned(),
                OURS,
                2,
                Some(100)
            ),
            (
                ManifestStatus::Existing,
                "b.parquet".to_owned(),
                20,
                2,
                Some(103)
            ),
        ]
    );
}

/// A path live twice in one manifest is removed twice and counted once.
#[test]
fn a_duplicate_is_counted_not_doubled() {
    let env = env();
    let t = create(&env, 2);
    let m = t.metadata().clone();
    let a = data(&env, &m, "a.parquet", 3);
    let hit = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        2,
        vec![
            (ManifestStatus::Added, a.clone(), 20, 2),
            (ManifestStatus::Added, a, 20, 2),
        ],
    );
    let f = run(
        &env,
        &m,
        vec![hit],
        &removals(&["a.parquet"], &env),
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(f.removed.duplicates, 1);
    let summary = overwrite_summary(&HashMap::new(), &HashSet::new(), &f.removed, 0);
    assert_eq!(
        summary.get("deleted-data-files").map(String::as_str),
        Some("1")
    );
    assert_eq!(
        summary.get("deleted-duplicate-files").map(String::as_str),
        Some("1")
    );
    assert_eq!(
        entries(&env, &f.manifests[0])
            .iter()
            .filter(|e| e.0 == ManifestStatus::Deleted)
            .count(),
        2
    );
}

/// A path no manifest holds live (only a `DELETED` entry) is Java's
/// "Missing required files to delete".
#[test]
fn a_file_not_live_is_missing() {
    let env = env();
    let t = create(&env, 2);
    let m = t.metadata().clone();
    let a = data(&env, &m, "a.parquet", 3);
    let b = data(&env, &m, "b.parquet", 3);
    let only = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        2,
        vec![
            (ManifestStatus::Deleted, a, 20, 1),
            (ManifestStatus::Added, b, 20, 2),
        ],
    );
    let err = run(
        &env,
        &m,
        vec![only],
        &removals(&["a.parquet"], &env),
        &mut Vec::new(),
    )
    .unwrap_err();
    match err {
        FilterError::Missing(text) => assert_eq!(
            text,
            format!("Missing required files to delete: {}", env.at("a.parquet"))
        ),
        other => panic!("{other:?}"),
    }
    // No manifest at all: still missing.
    assert!(matches!(
        run(
            &env,
            &m,
            Vec::new(),
            &removals(&["a.parquet"], &env),
            &mut Vec::new()
        ),
        Err(FilterError::Missing(_))
    ));
}

/// The DV of a removed data file goes with it, a delete file named goes,
/// a delete file older than every remaining data file goes, and the rest
/// stays.
#[test]
fn dangling_dvs_and_old_deletes_are_removed() {
    let env = env();
    let t = create(&env, 3);
    // The table's last sequence number above every file's.
    let mut json = serde_json::to_value(t.metadata()).unwrap();
    json["last-sequence-number"] = serde_json::json!(5);
    let m: TableMetadata = serde_json::from_value(json).unwrap();
    let a = data(&env, &m, "a.parquet", 3);
    let b = data(&env, &m, "b.parquet", 3);
    let part = a.partition().clone();
    let data_m = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        3,
        vec![
            (ManifestStatus::Added, a, 20, 3),
            (ManifestStatus::Existing, b, 10, 3),
        ],
    );
    let deletes = manifest(
        &env,
        &m,
        "d1.avro",
        ManifestContentType::Deletes,
        20,
        4,
        vec![
            (
                ManifestStatus::Added,
                delete_file(&env, "a.dv", Some("a.parquet"), part.clone()),
                20,
                4,
            ),
            (
                ManifestStatus::Added,
                delete_file(&env, "b.dv", Some("b.parquet"), part.clone()),
                20,
                4,
            ),
            (
                ManifestStatus::Existing,
                delete_file(&env, "old.parquet", None, part.clone()),
                10,
                2,
            ),
        ],
    );
    let f = run(
        &env,
        &m,
        vec![data_m, deletes],
        &removals(&["a.parquet"], &env),
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(f.replaced, 2);
    let e = entries(&env, &f.manifests[1]);
    assert_eq!(
        e.iter().map(|e| (e.0, e.1.as_str())).collect::<Vec<_>>(),
        [
            (ManifestStatus::Deleted, "a.dv"),
            (ManifestStatus::Existing, "b.dv"),
            (ManifestStatus::Deleted, "old.parquet"),
        ]
    );
    let summary = overwrite_summary(&HashMap::new(), &HashSet::new(), &f.removed, 0);
    assert_eq!(summary.get("removed-dvs").map(String::as_str), Some("1"));
    assert_eq!(
        summary
            .get("removed-position-delete-files")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(
        summary.get("removed-delete-files").map(String::as_str),
        Some("2")
    );
    assert_eq!(
        summary.get("removed-position-deletes").map(String::as_str),
        Some("4")
    );
    // A DV's size is its blob's (40), the position file's its own (500),
    // with the data file's.
    let removed: u64 = summary["removed-files-size"].parse().unwrap();
    assert!(removed > 540);
    // A delete file named outright goes too; a missing one fails.
    let c_dv = delete_file(&env, "c.dv", Some("c.parquet"), part.clone());
    let named = |f: &DataFile| Removals {
        data_files: BTreeSet::new(),
        delete_files: BTreeSet::from([DeleteKey::of(f)]),
        ..Removals::default()
    };
    let deletes2 = manifest(
        &env,
        &m,
        "d2.avro",
        ManifestContentType::Deletes,
        20,
        4,
        vec![(ManifestStatus::Added, c_dv.clone(), 20, 4)],
    );
    let f = run(
        &env,
        &m,
        vec![deletes2.clone()],
        &named(&c_dv),
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(entries(&env, &f.manifests[0])[0].0, ManifestStatus::Deleted);
    assert!(matches!(
        run(
            &env,
            &m,
            vec![deletes2],
            &named(&delete_file(&env, "x.dv", Some("c.parquet"), part)),
            &mut Vec::new()
        ),
        Err(FilterError::Missing(_))
    ));
}

/// The overwrite summary: removed values with the added ones, changed
/// partitions counted once over both, partition summaries merged in Java's
/// key order when they fit the limit.
#[test]
fn the_overwrite_summary_merges_added_and_removed() {
    let env = env();
    let t = create(&env, 2);
    let m = t.metadata().clone();
    let a = data(&env, &m, "a.parquet", 3);
    let hit = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        2,
        vec![(ManifestStatus::Added, a.clone(), 20, 2)],
    );
    let f = run(
        &env,
        &m,
        vec![hit],
        &removals(&["a.parquet"], &env),
        &mut Vec::new(),
    )
    .unwrap();
    let added = HashMap::from([
        ("added-data-files".to_owned(), "1".to_owned()),
        ("added-records".to_owned(), "5".to_owned()),
        (
            "partitions.x=1".to_owned(),
            "added-records=5,added-data-files=1".to_owned(),
        ),
        ("partitions.x=2".to_owned(), "added-data-files=1".to_owned()),
    ]);
    let changed = HashSet::from(["x=1".to_owned(), "x=2".to_owned()]);
    let s = overwrite_summary(&added, &changed, &f.removed, 10);
    assert_eq!(s["changed-partition-count"], "2");
    assert_eq!(s["partition-summaries-included"], "true");
    assert_eq!(
        s["partitions.x=1"],
        format!(
            "added-data-files=1,deleted-data-files=1,added-records=5,deleted-records=3,\
             removed-files-size={}",
            a.file_size_in_bytes()
        )
    );
    assert_eq!(s["added-records"], "5");
    assert_eq!(s["deleted-records"], "3");
    // Over the limit: no partition summaries.
    let s = overwrite_summary(&added, &changed, &f.removed, 1);
    assert!(!s.keys().any(|k| k.starts_with("partitions.")));
    assert!(!s.contains_key("partition-summaries-included"));
}

/// A DV of `of` at `offset` in the Puffin file `name`.
fn dv_at(env: &Env, name: &str, of: &str, offset: i64, partition: Struct) -> DataFile {
    let mut b = DataFileBuilder::default();
    b.content(DataContentType::PositionDeletes)
        .file_path(env.at(name))
        .file_size_in_bytes(500)
        .record_count(2)
        .partition(partition)
        .partition_spec_id(0)
        .file_format(DataFileFormat::Puffin)
        .referenced_data_file(Some(env.at(of)))
        .content_offset(Some(offset))
        .content_size_in_bytes(Some(40));
    b.build().unwrap()
}

/// One Puffin file holds the DVs of two data files (review 36b2a B1): the
/// rewrite of one removes that DV's blob only; the other file's DV stays
/// live, so its deleted rows stay deleted.
#[test]
fn a_shared_puffin_file_keeps_the_dv_of_an_untouched_file() {
    let env = env();
    let t = create(&env, 3);
    let m = t.metadata().clone();
    let a = data(&env, &m, "a.parquet", 3);
    let b = data(&env, &m, "b.parquet", 3);
    let part = a.partition().clone();
    let dv_a = dv_at(&env, "p.puffin", "a.parquet", 4, part.clone());
    let dv_b = dv_at(&env, "p.puffin", "b.parquet", 44, part);
    let data_m = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        1,
        vec![
            (ManifestStatus::Added, a, 20, 1),
            (ManifestStatus::Added, b, 20, 1),
        ],
    );
    let deletes = manifest(
        &env,
        &m,
        "d1.avro",
        ManifestContentType::Deletes,
        20,
        1,
        vec![
            (ManifestStatus::Added, dv_a.clone(), 20, 1),
            (ManifestStatus::Added, dv_b, 20, 1),
        ],
    );
    let r = Removals {
        data_files: BTreeSet::from([env.at("a.parquet")]),
        delete_files: BTreeSet::from([DeleteKey::of(&dv_a)]),
        ..Removals::default()
    };
    let f = run(&env, &m, vec![data_m, deletes], &r, &mut Vec::new()).unwrap();
    let e = entries(&env, &f.manifests[1]);
    assert_eq!(
        e.iter().map(|e| e.0).collect::<Vec<_>>(),
        [ManifestStatus::Deleted, ManifestStatus::Existing]
    );
    let io = env.reg.file_io();
    let bytes = env
        .rt
        .block_on(async {
            io.new_input(&f.manifests[1].manifest_path)
                .unwrap()
                .read()
                .await
        })
        .unwrap();
    let (parsed, _) = Manifest::parse_avro(&bytes).unwrap().into_parts();
    assert_eq!(
        parsed[1].data_file().referenced_data_file(),
        Some(env.at("b.parquet"))
    );
}

/// The touched file's DV was replaced concurrently (its blob is no longer
/// live, another DV of the same Puffin file is): the removal is missing.
#[test]
fn a_replaced_dv_in_a_shared_puffin_file_is_missing() {
    let env = env();
    let t = create(&env, 3);
    let m = t.metadata().clone();
    let a = data(&env, &m, "a.parquet", 3);
    let b = data(&env, &m, "b.parquet", 3);
    let part = a.partition().clone();
    let old_a = dv_at(&env, "p.puffin", "a.parquet", 4, part.clone());
    let dv_b = dv_at(&env, "p.puffin", "b.parquet", 44, part.clone());
    let new_a = dv_at(&env, "q.puffin", "a.parquet", 4, part);
    let data_m = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        1,
        vec![
            (ManifestStatus::Added, a, 20, 1),
            (ManifestStatus::Added, b, 20, 1),
        ],
    );
    let deletes = manifest(
        &env,
        &m,
        "d1.avro",
        ManifestContentType::Deletes,
        20,
        1,
        vec![
            (ManifestStatus::Added, dv_b, 20, 1),
            (ManifestStatus::Added, new_a, 20, 1),
        ],
    );
    let r = Removals {
        data_files: BTreeSet::from([env.at("a.parquet")]),
        delete_files: BTreeSet::from([DeleteKey::of(&old_a)]),
        ..Removals::default()
    };
    match run(&env, &m, vec![data_m, deletes], &r, &mut Vec::new()) {
        Err(FilterError::Missing(text)) => {
            // The Puffin path and the data file the DV applies to.
            assert!(text.contains("p.puffin"), "{text}");
            assert!(
                text.contains(&format!("the deletion vector of {}", env.at("a.parquet"))),
                "{text}"
            );
        }
        other => panic!("{other:?}"),
    }
}

/// A row filter (`deleteByRowFilter`, PR-36b part 2b): data files whose
/// partition the strict projection proves are removed; a data file it
/// matches only in part fails with Java's text; a delete file it matches
/// only in part is kept silently (`ManifestFilterManager.java:554`).
#[test]
fn a_row_filter_removes_whole_files_and_refuses_partial_data_files() {
    use iceberg::spec::Literal;
    use krishiv_io::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};
    let env = env();
    let t = create(&env, 2);
    let m = t.metadata().clone();
    // `data` writes partition x = 1 and no column metrics.
    let a = data(&env, &m, "a.parquet", 3);
    let data_m = manifest(
        &env,
        &m,
        "m1.avro",
        ManifestContentType::Data,
        20,
        1,
        vec![(ManifestStatus::Added, a, 20, 1)],
    );
    let x2 = Struct::from_iter([Some(Literal::int(2))]);
    let deletes = manifest(
        &env,
        &m,
        "d1.avro",
        ManifestContentType::Deletes,
        20,
        1,
        vec![(
            ManifestStatus::Added,
            delete_file(&env, "pd.parquet", None, x2),
            20,
            1,
        )],
    );
    let x_is = |v: i32| PushdownExpr::cmp(ColumnPath::new("x"), CmpOp::Eq, ScalarValue::Int32(v));
    let id_ge = PushdownExpr::cmp(ColumnPath::new("id"), CmpOp::GtEq, ScalarValue::Int64(0));
    let with = |f: PushdownExpr| Removals {
        row_filter: Some(f),
        ..Removals::default()
    };
    let base = vec![data_m, deletes];
    // x = 1: the data file removed whole.
    let f = run(&env, &m, base.clone(), &with(x_is(1)), &mut Vec::new()).unwrap();
    assert_eq!(f.replaced, 1);
    let summary = overwrite_summary(&HashMap::new(), &HashSet::new(), &f.removed, 0);
    assert_eq!(
        summary.get("deleted-data-files").map(String::as_str),
        Some("1")
    );
    // id >= 0: no metrics prove it for the data file.
    match run(
        &env,
        &m,
        base.clone(),
        &with(id_ge.clone()),
        &mut Vec::new(),
    ) {
        Err(FilterError::Partial { filter, file }) => assert_eq!(
            FilterError::partial_text(&filter, &file),
            format!(
                "Cannot delete file where some, but not all, rows match filter \
                 ref(name=\"id\") >= 0: {}",
                env.at("a.parquet")
            )
        ),
        other => panic!("{other:?}"),
    }
    // x = 2 and id >= 0: the data file ruled out by its partition, the
    // delete file in x = 2 matched in part: kept, nothing rewritten.
    let both = PushdownExpr::And(vec![x_is(2), id_ge]);
    let f = run(&env, &m, base, &with(both), &mut Vec::new()).unwrap();
    assert_eq!(f.replaced, 0);
}

/// Partition keys compare by value across a promotion (review 36b2b M3,
/// the scope of both the dropped partitions and the validation): an old
/// 4-byte `int` tuple equals a new 8-byte `long` one, whichever type the
/// spec reads them with, `float` and `double` alike; an unreadable value
/// never makes a key.
#[test]
fn partition_keys_compare_by_value_across_promotions() {
    use crate::plan::PartitionValue as V;
    use crate::plan::types::{IcebergType, PartitionFieldInfo, PartitionSpecInfo, Transform};

    use super::filter::partition_key;
    let specs = |ty: IcebergType| {
        let field = PartitionFieldInfo::new(1000, 1, "v".to_owned(), Transform::Identity, Some(ty));
        std::collections::BTreeMap::from([(
            0,
            PartitionSpecInfo {
                spec_id: 0,
                fields: vec![field],
            },
        )])
    };
    let int = V::Value(5_i32.to_le_bytes().to_vec());
    let long = V::Value(5_i64.to_le_bytes().to_vec());
    for ty in [IcebergType::Int, IcebergType::Long] {
        let s = specs(ty);
        assert_eq!(
            partition_key(&s, 0, std::slice::from_ref(&int)),
            partition_key(&s, 0, std::slice::from_ref(&long))
        );
        assert!(partition_key(&s, 0, std::slice::from_ref(&int)).is_some());
    }
    let float = V::Value(1.5_f32.to_le_bytes().to_vec());
    let double = V::Value(1.5_f64.to_le_bytes().to_vec());
    for ty in [IcebergType::Float, IcebergType::Double] {
        let s = specs(ty);
        assert_eq!(
            partition_key(&s, 0, std::slice::from_ref(&float)),
            partition_key(&s, 0, std::slice::from_ref(&double))
        );
    }
    let s = specs(IcebergType::Long);
    assert_ne!(
        partition_key(&s, 0, std::slice::from_ref(&long)),
        partition_key(&s, 0, &[V::Value(6_i64.to_le_bytes().to_vec())])
    );
    assert_eq!(partition_key(&s, 0, &[V::Null]), Some((0, vec![V::Null])));
    assert_eq!(partition_key(&s, 0, &[V::Unreadable]), None);
}
