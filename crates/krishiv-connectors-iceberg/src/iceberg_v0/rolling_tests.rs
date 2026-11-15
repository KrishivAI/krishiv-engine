// SPDX-License-Identifier: Apache-2.0
//! Tests of the new manifests' layout (`rolling`, PR-36a; owner decision
//! 2026-11-15): Java's `RollingManifestWriter` split points against a literal
//! transcription of its loop (`java_layout`, the length of a manifest being
//! iceberg-rust's encoding of the entries so far), Java's
//! `manifestWriterCount`/`divide` groups at the 14,999 / 15,000 / 25,000 /
//! 45,001-file boundaries, the names and the written-path list the sink cleans
//! up, and rolled manifests committed, merged and planned back.

use std::sync::Arc;

use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Manifest, ManifestWriterBuilder,
    TableMetadata, TableMetadataBuilder, UNASSIGNED_SEQUENCE_NUMBER,
};

use krishiv_common::error::ErrorClass;

use super::append::{
    MANIFESTS_CREATED, MANIFESTS_KEPT, MANIFESTS_REPLACED, NewManifests, prepare_attempt,
    write_new_manifests,
};
use super::append_tests::{MAX, commit, create, iceberg_rust_files, ids, now};
use super::data_file::partition_tuple;
use super::merge::{MANIFEST_MIN_MERGE_COUNT, MANIFEST_TARGET_SIZE_BYTES};
use super::provider_tests::{Env, env};
use super::rolling::{
    MIN_FILE_GROUP_SIZE, ROWS_DIVISOR, WRITER_POOL, divide, manifest_writer_count,
};

const ID: i64 = 901;

/// `metadata` with the table properties `props` set.
fn with(metadata: &TableMetadata, props: &[(&str, String)]) -> TableMetadata {
    TableMetadataBuilder::new_from_metadata(metadata.clone(), None)
        .set_properties(
            props
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        )
        .unwrap()
        .build()
        .unwrap()
        .metadata
}

/// `n` data files of the default spec (`identity(x)`, `x` = i mod 3), named
/// `f<tag><i>` so every file name has the same length.
fn files(m: &TableMetadata, tag: char, n: usize) -> Vec<(i32, DataFile)> {
    let spec = m.default_partition_spec();
    let schema = m.current_schema();
    (0..n)
        .map(|i| {
            let x = i32::try_from(i % 3).unwrap();
            let partition =
                partition_tuple(spec, schema, &[Some(krishiv_io::ScalarValue::Int32(x))]).unwrap();
            let mut b = DataFileBuilder::default();
            b.content(DataContentType::Data)
                .file_path(format!("file:///data/x={x}/f{tag}{i:06}.parquet"))
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(1000 + i as u64)
                .record_count(10)
                .partition(partition)
                .partition_spec_id(spec.spec_id());
            (spec.spec_id(), b.build().unwrap())
        })
        .collect()
}

/// The length of a manifest of `files` as iceberg-rust writes it (in memory).
fn encoded(env: &Env, m: &TableMetadata, files: &[DataFile]) -> i64 {
    let io = FileIO::new_with_memory();
    let mut w = ManifestWriterBuilder::new(
        io.new_output("memory:///oracle.avro").unwrap(),
        Some(ID),
        Arc::clone(m.current_schema()),
        m.default_partition_spec().as_ref().clone(),
    )
    .build_v2_data();
    for f in files {
        w.add_file(f.clone(), UNASSIGNED_SEQUENCE_NUMBER).unwrap();
    }
    env.rt
        .block_on(w.write_manifest_file())
        .unwrap()
        .manifest_length
}

/// Java's `RollingManifestWriter` loop over one group, transcribed: before
/// each entry but the first, roll when `rows % 250 == 0 && length >= target`.
/// The entry counts per manifest.
fn java_layout(env: &Env, m: &TableMetadata, files: &[DataFile], target: i64) -> Vec<usize> {
    let mut out = Vec::new();
    let mut current: Vec<DataFile> = Vec::new();
    let mut open = false;
    for f in files {
        if !open {
            open = true;
        } else if current.len().is_multiple_of(250) && encoded(env, m, &current) >= target {
            out.push(current.len());
            current.clear();
        }
        current.push(f.clone());
    }
    if open {
        out.push(current.len());
    }
    out
}

/// The write's manifests and the paths it reported as written.
fn write(env: &Env, m: &TableMetadata, files: Vec<(i32, DataFile)>) -> (NewManifests, Vec<String>) {
    let io = env.reg.file_io();
    let mut written = Vec::new();
    let new = env
        .rt
        .block_on(write_new_manifests(
            &io,
            m,
            files,
            ID,
            "uuid-roll",
            &mut written,
        ))
        .unwrap();
    (new, written)
}

/// Each manifest's file names, through iceberg-rust's `Manifest::parse_avro`.
fn entries(env: &Env, new: &NewManifests) -> Vec<Vec<String>> {
    let io = env.reg.file_io();
    new.manifests
        .iter()
        .map(|mf| {
            let bytes = env
                .rt
                .block_on(async { io.new_input(&mf.manifest_path).unwrap().read().await })
                .unwrap();
            let (es, _) = Manifest::parse_avro(&bytes).unwrap().into_parts();
            es.iter().map(|e| e.file_path().to_owned()).collect()
        })
        .collect()
}

fn counts(env: &Env, new: &NewManifests) -> Vec<usize> {
    entries(env, new).iter().map(Vec::len).collect()
}

/// The layout of `n` files at target 1 byte: every manifest rolls at 250.
fn layout_at_one_byte(env: &Env, m: &TableMetadata, n: usize) -> Vec<usize> {
    let m = with(m, &[(MANIFEST_TARGET_SIZE_BYTES, "1".to_owned())]);
    let fs = files(&m, 'a', n);
    let order: Vec<String> = fs.iter().map(|(_, f)| f.file_path().to_owned()).collect();
    let (new, written) = write(env, &m, fs);
    // Names m0, m1, ... in list order, each reported before it was written.
    let paths: Vec<String> = new
        .manifests
        .iter()
        .map(|mf| mf.manifest_path.clone())
        .collect();
    assert_eq!(paths, written);
    for (i, p) in paths.iter().enumerate() {
        assert!(p.ends_with(&format!("/uuid-roll-m{i}.avro")), "{p}");
    }
    // Entries in input order across the manifests.
    let es = entries(env, &new);
    assert_eq!(es.concat(), order);
    es.iter().map(Vec::len).collect()
}

/// `n` manifests of 250 then one of `rest` (when not 0).
fn rolled(n: usize, rest: usize) -> Vec<usize> {
    let mut v = vec![250; n];
    if rest > 0 {
        v.push(rest);
    }
    v
}

#[test]
fn constants_are_javas() {
    assert_eq!(ROWS_DIVISOR, 250);
    assert_eq!(MIN_FILE_GROUP_SIZE, 10_000);
    assert_eq!(WRITER_POOL, 4);
}

/// `SnapshotProducer.manifestWriterCount`: `max(1, min(pool, round_half_up(n / 10,000)))`.
#[test]
fn writer_count_is_javas() {
    for (n, groups) in [
        (0, 1),
        (1, 1),
        (4_999, 1),
        (5_000, 1),
        (14_999, 1),
        (15_000, 2),
        (24_999, 2),
        (25_000, 3),
        (34_999, 3),
        (35_000, 4),
        (45_001, 4),
        (10_000_000, 4),
    ] {
        assert_eq!(manifest_writer_count(WRITER_POOL, n), groups, "{n}");
    }
    assert_eq!(manifest_writer_count(1, 1_000_000), 1);
    assert_eq!(manifest_writer_count(16, 45_001), 5);
    assert_eq!(manifest_writer_count(0, 45_001), 1);
}

/// `ManifestFiles.divide`: `Lists.partition` by `ceil(n / groups)`, so fewer
/// groups than asked when the last would be empty.
#[test]
fn divide_is_javas_partition() {
    let sizes = |n: usize, g: usize| -> Vec<usize> {
        divide((0..n).collect::<Vec<_>>(), g)
            .iter()
            .map(Vec::len)
            .collect()
    };
    assert_eq!(sizes(10, 4), [3, 3, 3, 1]);
    assert_eq!(sizes(9, 4), [3, 3, 3]);
    assert_eq!(sizes(25_000, 3), [8_334, 8_334, 8_332]);
    assert_eq!(sizes(7, 1), [7]);
    assert!(sizes(0, 3).is_empty());
    let order: Vec<usize> = divide((0..10).collect(), 4).concat();
    assert_eq!(order, (0..10).collect::<Vec<_>>());
}

/// The split points of 1,300 files equal Java's loop at targets around the
/// lengths of 250, 500 and 750 entries, at 0 and 1 byte, and at the default.
#[test]
fn rolls_where_javas_loop_rolls() {
    let env = env();
    let t = create(&env, 2);
    let m0 = t.metadata().clone();
    let fs = files(&m0, 'a', 1_300);
    let plain: Vec<DataFile> = fs.iter().map(|(_, f)| f.clone()).collect();
    let l250 = encoded(&env, &m0, &plain[..250]);
    let l500 = encoded(&env, &m0, &plain[..500]);
    let l750 = encoded(&env, &m0, &plain[..750]);
    for (target, expected) in [
        (i64::MIN, Some(rolled(5, 50))),
        (0, Some(rolled(5, 50))),
        (1, Some(rolled(5, 50))),
        (l250, None),
        (l250 + 1, None),
        (l500 - 1, None),
        (l500, None),
        (l500 + 1, None),
        (l750 + 1, None),
        (8 * 1024 * 1024, Some(vec![1_300])),
    ] {
        let m = with(&m0, &[(MANIFEST_TARGET_SIZE_BYTES, target.to_string())]);
        let java = java_layout(&env, &m, &plain, target);
        if let Some(e) = expected {
            assert_eq!(java, e, "oracle at {target}");
        }
        let (new, _) = write(&env, &m, fs.clone());
        assert_eq!(counts(&env, &new), java, "target {target}");
        // Every written length is iceberg-rust's encoding of its entries.
        let mut at = 0;
        for (mf, n) in new.manifests.iter().zip(&java) {
            assert_eq!(mf.manifest_length, encoded(&env, &m, &plain[at..at + n]));
            at += n;
        }
    }
    // The boundaries themselves: `>=` rolls at the length reached.
    let at = |target: i64| {
        let m = with(&m0, &[(MANIFEST_TARGET_SIZE_BYTES, target.to_string())]);
        counts(&env, &write(&env, &m, fs.clone()).0)[0]
    };
    assert_eq!(at(l250), 250);
    assert_eq!(at(l250 + 1), 500);
    assert_eq!(at(l500), 500);
    assert_eq!(at(l500 + 1), 750);
}

/// The first writer is lazy: no files, no manifest; a roll needs a 251st
/// entry, so 250 and 500 files are full manifests and 251 leaves one entry.
#[test]
fn edges_of_the_250_check() {
    let env = env();
    let t = create(&env, 2);
    let m = with(
        t.metadata(),
        &[(MANIFEST_TARGET_SIZE_BYTES, "1".to_owned())],
    );
    let (new, written) = write(&env, &m, Vec::new());
    assert!(new.manifests.is_empty());
    assert!(written.is_empty());
    assert_eq!(layout_at_one_byte(&env, &m, 1), [1]);
    assert_eq!(layout_at_one_byte(&env, &m, 249), [249]);
    assert_eq!(layout_at_one_byte(&env, &m, 250), [250]);
    assert_eq!(layout_at_one_byte(&env, &m, 251), [250, 1]);
    assert_eq!(layout_at_one_byte(&env, &m, 500), [250, 250]);
    assert_eq!(layout_at_one_byte(&env, &m, 501), [250, 250, 1]);
}

/// A target that does not parse fails the write before any manifest (Java's
/// `propertyAsLong` in the `SnapshotProducer` constructor), so the sink has
/// nothing to delete but the data files.
#[test]
fn unparseable_target_writes_nothing() {
    let env = env();
    let t = create(&env, 2);
    let m = with(
        t.metadata(),
        &[(MANIFEST_TARGET_SIZE_BYTES, "8MB".to_owned())],
    );
    let mut written = Vec::new();
    let e = env
        .rt
        .block_on(write_new_manifests(
            &env.reg.file_io(),
            &m,
            files(&m, 'a', 300),
            ID,
            "uuid-bad",
            &mut written,
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(written.is_empty());
}

/// Groups: 14,999 files are one group; 15,000 two of 7,500; 25,000 three
/// (8,334, 8,334, 8,332); 45,001 four (the pool, `WRITER_POOL`) of 11,251 but
/// the last 11,248. Each rolled apart, in group order.
#[test]
fn groups_at_the_writer_count_boundaries() {
    let env = env();
    let t = create(&env, 2);
    let m = t.metadata().clone();
    assert_eq!(layout_at_one_byte(&env, &m, 14_999), rolled(59, 249));
    assert_eq!(layout_at_one_byte(&env, &m, 15_000), rolled(60, 0));
    let three = [rolled(33, 84), rolled(33, 84), rolled(33, 82)].concat();
    assert_eq!(layout_at_one_byte(&env, &m, 25_000), three);
    let four = [rolled(45, 1), rolled(45, 1), rolled(45, 1), rolled(44, 248)].concat();
    assert_eq!(layout_at_one_byte(&env, &m, 45_001), four);
}

/// Two specs: each spec's files are rolled (and grouped) apart, numbered on
/// across specs, spec by spec in id order.
#[test]
fn specs_roll_apart_and_number_on() {
    let env = env();
    let t = create(&env, 2);
    let m0 = t.metadata().clone();
    let spec = iceberg::spec::UnboundPartitionSpec::builder()
        .add_partition_field(1, "id_b", iceberg::spec::Transform::Bucket(4))
        .unwrap()
        .build();
    let m1 = TableMetadataBuilder::new_from_metadata(m0.clone(), None)
        .add_partition_spec(spec)
        .unwrap()
        .set_default_partition_spec(-1)
        .unwrap()
        .set_properties([(MANIFEST_TARGET_SIZE_BYTES.to_owned(), "1".to_owned())].into())
        .unwrap()
        .build()
        .unwrap()
        .metadata;
    let mut old: Vec<(i32, DataFile)> = files(&m0, 'o', 300);
    let new_spec = files(&m1, 'n', 260);
    old.extend(new_spec);
    let (new, written) = write(&env, &m1, old);
    let specs: Vec<i32> = new.manifests.iter().map(|m| m.partition_spec_id).collect();
    assert_eq!(specs, [0, 0, 1, 1]);
    assert_eq!(counts(&env, &new), [250, 50, 250, 10]);
    for (i, p) in written.iter().enumerate() {
        assert!(p.ends_with(&format!("-m{i}.avro")), "{p}");
    }
}

/// Rolled manifests committed and merged: a base of one small manifest; an
/// append of 260 files rolls into 250 + 10, and its 10-entry manifest merges
/// with the base's (`min-count-to-merge` 2) into `-m2`, numbered after the
/// rolled ones; a third append of 600 rolls into three manifests that stay,
/// `manifests-created` 3 and `-kept` 2. iceberg-rust plans every file once.
#[test]
fn rolled_manifests_commit_and_merge() {
    let env = env();
    let t = create(&env, 2);
    let loc0 = t.metadata_location().unwrap().to_owned();
    let m0 = t.metadata().clone();
    // 240 entries under snapshot 901: below 250 entries under the write's
    // own ids (their snapshot id takes a byte less per entry), so the
    // manifests roll at 250.
    let plain: Vec<DataFile> = files(&m0, 'a', 240).into_iter().map(|(_, f)| f).collect();
    let target = encoded(&env, &m0, &plain);
    let m0 = with(
        &m0,
        &[
            (MANIFEST_TARGET_SIZE_BYTES, target.to_string()),
            (MANIFEST_MIN_MERGE_COUNT, "2".to_owned()),
        ],
    );
    let io = env.reg.file_io();
    let append = |m: &TableMetadata, loc: &str, tag: char, n: usize, id: i64, k: u32| {
        let fs = files(m, tag, n);
        let uuid = format!("uuid-{id}");
        let mut new = env
            .rt
            .block_on(write_new_manifests(&io, m, fs, id, &uuid, &mut Vec::new()))
            .unwrap();
        let w = format!("w-{id}");
        let a = env
            .rt
            .block_on(prepare_attempt(&io, m, &mut new, ids(&w), now(m), MAX))
            .unwrap();
        let (m, loc) = commit(&env, m, loc, &a, k);
        (new, m, loc)
    };
    let listed = |m: &TableMetadata| -> Vec<String> {
        env.rt
            .block_on(super::scan::read_manifest_files(
                &io,
                m.current_snapshot().unwrap().manifest_list(),
                2,
                MAX,
            ))
            .unwrap()
            .into_iter()
            .map(|mf| mf.manifest_path.rsplit('/').next().unwrap().to_owned())
            .collect()
    };
    let summary = |m: &TableMetadata, k: &str| -> String {
        m.current_snapshot()
            .unwrap()
            .summary()
            .additional_properties[k]
            .clone()
    };

    let (_, m1, loc1) = append(&m0, &loc0, 'z', 1, 11, 1);
    assert_eq!(listed(&m1), ["uuid-11-m0.avro"]);

    let (second, m2, loc2) = append(&m1, &loc1, 'b', 260, 12, 2);
    assert_eq!(second.manifests.len(), 2);
    assert_eq!(second.merged.len(), 1);
    assert!(
        second.merged[0].ends_with("/uuid-12-m2.avro"),
        "{:?}",
        second.merged
    );
    assert_eq!(listed(&m2), ["uuid-12-m0.avro", "uuid-12-m2.avro"]);
    assert_eq!(summary(&m2, MANIFESTS_CREATED), "2");
    assert_eq!(summary(&m2, MANIFESTS_REPLACED), "1");
    assert_eq!(summary(&m2, "added-data-files"), "260");
    assert_eq!(iceberg_rust_files(&env, &m2, &loc2).len(), 261);

    let (third, m3, loc3) = append(&m2, &loc2, 'c', 600, 13, 3);
    assert_eq!(third.manifests.len(), 3);
    assert!(third.merged.is_empty());
    assert_eq!(
        listed(&m3),
        [
            "uuid-13-m0.avro",
            "uuid-13-m1.avro",
            "uuid-13-m2.avro",
            "uuid-12-m0.avro",
            "uuid-12-m2.avro"
        ]
    );
    assert_eq!(summary(&m3, MANIFESTS_CREATED), "3");
    assert_eq!(summary(&m3, MANIFESTS_KEPT), "2");
    assert_eq!(summary(&m3, "total-data-files"), "861");
    assert_eq!(iceberg_rust_files(&env, &m3, &loc3).len(), 861);
}
