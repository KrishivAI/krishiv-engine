// SPDX-License-Identifier: Apache-2.0
//! Tests of position deletes and deletion vectors through SQL (PR-25b; design
//! note §3.2–3.3): tables whose one snapshot lists data and delete files with
//! chosen sequence numbers, written with iceberg-rust's manifest writers; the
//! property test `delete_application_matches_model`; DV corruption, a missing
//! delete file, Java's assignment errors and equality deletes refused.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, StringArray};
use arrow_schema::DataType as A;
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestListWriter,
    ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Struct,
    Transform as IceTransform, Type,
};
use krishiv_common::error::{ErrorClass, KrishivError};
use proptest::prelude::*;
use roaring::RoaringTreemap;

use super::provider_tests::{Env, batch, env, id_field, provider_session, write_parquet};
use super::table::IcebergTable;
use crate::plan::{DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID, VersionSpec, encode_dv};

/// `id long (1), p int (2)`.
fn schema() -> iceberg::spec::Schema {
    iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "p", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap()
}

/// What one file of a test table is.
#[derive(Debug, Clone)]
enum Content {
    /// Rows with ids `first..first + rows` in partition `p`.
    Data { first: i64, rows: i64 },
    /// A position delete file: `(data file name, position)` rows.
    Positions {
        rows: Vec<(String, i64)>,
        /// Written with `referenced_data_file` (file-scoped).
        scoped: Option<String>,
    },
    /// A deletion vector of `data` (a Puffin blob at offset 4).
    Dv { data: String, positions: Vec<u64> },
    /// An equality delete file on `id` (never read: refused).
    Equality,
}

/// One file: its name, partition value (for a partitioned table), data
/// sequence number and content.
#[derive(Debug, Clone)]
struct File {
    name: String,
    p: i32,
    seq: i64,
    content: Content,
}

/// A table with one snapshot listing `files` (data in a v2 data manifest,
/// deletes in a v3 delete manifest so DVs carry their fields).
struct Table {
    env: Env,
    table: IcebergTable,
}

fn build(files: &[File], partitioned: bool) -> Table {
    let env = env();
    let schema = schema();
    let spec = if partitioned {
        PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field("p", "p", IceTransform::Identity)
            .unwrap()
            .build()
            .unwrap()
    } else {
        PartitionSpec::unpartition_spec()
    };
    let t = env.create("t", schema.clone(), partitioned.then(|| spec.clone()));
    let partition = |p: i32| {
        if partitioned {
            Struct::from_iter([Some(Literal::int(p))])
        } else {
            Struct::empty()
        }
    };
    let snapshot_id = 1000;
    let seq = files.iter().map(|f| f.seq).max().unwrap_or(1);
    let mut data_files = Vec::new();
    let mut delete_files = Vec::new();
    for f in files {
        let path = env.at(&f.name);
        let builder = |content, format, size, records| {
            let mut b = DataFileBuilder::default();
            b.content(content)
                .file_path(path.clone())
                .file_format(format)
                .file_size_in_bytes(size)
                .record_count(records)
                .partition(partition(f.p))
                .partition_spec_id(0);
            b
        };
        match &f.content {
            Content::Data { first, rows } => {
                let ids: Vec<i64> = (*first..first + rows).collect();
                let b = batch(
                    vec![id_field("id", A::Int64, 1), id_field("p", A::Int32, 2)],
                    vec![
                        Arc::new(Int64Array::from(ids.clone())),
                        Arc::new(Int32Array::from(vec![f.p; ids.len()])),
                    ],
                );
                let size = write_parquet(&path, &b);
                let rows = u64::try_from(*rows).unwrap();
                data_files.push((
                    builder(DataContentType::Data, DataFileFormat::Parquet, size, rows)
                        .build()
                        .unwrap(),
                    f.seq,
                ));
            }
            Content::Positions { rows, scoped } => {
                let mut rows = rows.clone();
                rows.sort();
                let names: Vec<String> = rows.iter().map(|(n, _)| env.at(n)).collect();
                let b = batch(
                    vec![
                        id_field("file_path", A::Utf8, DELETE_FILE_PATH_ID),
                        id_field("pos", A::Int64, DELETE_FILE_POS_ID),
                    ],
                    vec![
                        Arc::new(StringArray::from(names)),
                        Arc::new(Int64Array::from(
                            rows.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
                        )),
                    ],
                );
                let size = write_parquet(&path, &b);
                let mut d = builder(
                    DataContentType::PositionDeletes,
                    DataFileFormat::Parquet,
                    size,
                    u64::try_from(rows.len()).unwrap(),
                );
                if let Some(s) = scoped {
                    d.referenced_data_file(Some(env.at(s)));
                }
                delete_files.push((d.build().unwrap(), f.seq));
            }
            Content::Dv { data, positions } => {
                let bitmap: RoaringTreemap = positions.iter().copied().collect();
                let blob = encode_dv(&bitmap);
                let mut bytes = b"PFA1".to_vec();
                bytes.extend_from_slice(&blob);
                std::fs::write(path.strip_prefix("file://").unwrap(), &bytes).unwrap();
                let mut d = builder(
                    DataContentType::PositionDeletes,
                    DataFileFormat::Puffin,
                    u64::try_from(bytes.len()).unwrap(),
                    bitmap.len(),
                );
                d.referenced_data_file(Some(env.at(data)))
                    .content_offset(Some(4))
                    .content_size_in_bytes(Some(i64::try_from(blob.len()).unwrap()));
                delete_files.push((d.build().unwrap(), f.seq));
            }
            Content::Equality => {
                let mut d = builder(
                    DataContentType::EqualityDeletes,
                    DataFileFormat::Parquet,
                    10,
                    1,
                );
                d.equality_ids(Some(vec![1]));
                delete_files.push((d.build().unwrap(), f.seq));
            }
        }
    }
    let io = env.reg.file_io();
    let schema = Arc::new(schema);
    let list = env.at("snap.avro");
    env.rt.block_on(async {
        let mut manifests = Vec::new();
        if !data_files.is_empty() {
            let mut w = ManifestWriterBuilder::new(
                io.new_output(env.at("data-m.avro")).unwrap(),
                Some(snapshot_id),
                Arc::clone(&schema),
                spec.clone(),
            )
            .build_v2_data();
            for (d, s) in data_files {
                w.add_file(d, s).unwrap();
            }
            manifests.push(w.write_manifest_file().await.unwrap());
        }
        if !delete_files.is_empty() {
            let mut w = ManifestWriterBuilder::new(
                io.new_output(env.at("delete-m.avro")).unwrap(),
                Some(snapshot_id),
                Arc::clone(&schema),
                spec.clone(),
            )
            .build_v3_deletes();
            for (d, s) in delete_files {
                w.add_file(d, s).unwrap();
            }
            manifests.push(w.write_manifest_file().await.unwrap());
        }
        let mut lw = ManifestListWriter::v2(
            io.new_output(&list).unwrap().writer().await.unwrap(),
            snapshot_id,
            None,
            seq,
        );
        lw.add_manifests(manifests.into_iter()).unwrap();
        lw.close().await.unwrap();
    });
    let location = t.metadata_location().unwrap().to_owned();
    let text = env
        .rt
        .block_on(async { io.new_input(&location).unwrap().read().await.unwrap() });
    let mut json: serde_json::Value = serde_json::from_slice(&text).unwrap();
    let ts = json["last-updated-ms"].as_i64().unwrap() + 1;
    json["snapshots"] = serde_json::json!([{
        "snapshot-id": snapshot_id,
        "sequence-number": seq,
        "timestamp-ms": ts,
        "manifest-list": list,
        "summary": {"operation": "overwrite"},
        "schema-id": 0
    }]);
    json["current-snapshot-id"] = serde_json::json!(snapshot_id);
    json["last-sequence-number"] = serde_json::json!(seq);
    json["refs"] = serde_json::json!({"main": {"snapshot-id": snapshot_id, "type": "branch"}});
    json["snapshot-log"] = serde_json::json!([{"snapshot-id": snapshot_id, "timestamp-ms": ts}]);
    let edited = env.at("99-deletes.metadata.json");
    env.rt.block_on(async {
        io.new_output(&edited)
            .unwrap()
            .write(bytes::Bytes::from(serde_json::to_vec(&json).unwrap()))
            .await
            .unwrap();
    });
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, &edited, "db", "t"))
        .unwrap();
    Table { env, table }
}

impl Table {
    /// The ids `SELECT id FROM t` returns, sorted, or the error of opening
    /// the provider or running the query.
    fn ids(&self, sql: &str) -> Result<Vec<i64>, KrishivError> {
        let p = self.env.provider(&self.table, &VersionSpec::Current)?;
        let batches = provider_session(p).run(sql)?;
        let mut out: Vec<i64> = super::provider_tests::render(&batches)
            .iter()
            .map(|r| r.parse().unwrap())
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    fn open_err(&self) -> KrishivError {
        self.env
            .provider(&self.table, &VersionSpec::Current)
            .unwrap_err()
    }
}

fn data(name: &str, p: i32, seq: i64, first: i64, rows: i64) -> File {
    File {
        name: name.to_owned(),
        p,
        seq,
        content: Content::Data { first, rows },
    }
}

fn positions(name: &str, p: i32, seq: i64, rows: &[(&str, i64)], scoped: Option<&str>) -> File {
    File {
        name: name.to_owned(),
        p,
        seq,
        content: Content::Positions {
            rows: rows.iter().map(|(n, p)| ((*n).to_owned(), *p)).collect(),
            scoped: scoped.map(str::to_owned),
        },
    }
}

fn dv(name: &str, p: i32, seq: i64, data: &str, positions: &[u64]) -> File {
    File {
        name: name.to_owned(),
        p,
        seq,
        content: Content::Dv {
            data: data.to_owned(),
            positions: positions.to_vec(),
        },
    }
}

#[test]
fn position_deletes_and_dvs_remove_rows() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 5),
            data("b.parquet", 0, 1, 10, 5),
            data("c.parquet", 0, 2, 20, 5),
            // One partition-wide file for a and b, unordered; a file-scoped one for b.
            positions(
                "d1.parquet",
                0,
                2,
                &[("b.parquet", 4), ("a.parquet", 1)],
                None,
            ),
            positions("d2.parquet", 0, 3, &[("b.parquet", 0)], Some("b.parquet")),
            // A position past the end deletes nothing, as in Java.
            dv("c.puffin", 0, 2, "c.parquet", &[0, 2, 99]),
        ],
        false,
    );
    // c's DV holds 3 positions (its record count); 99 is past c's 5 rows.
    assert_eq!(
        t.ids("SELECT id FROM t").unwrap(),
        [0, 2, 3, 4, 11, 12, 13, 21, 23, 24]
    );
    assert_eq!(t.ids("SELECT count(*) FROM t").unwrap(), [10]);
    assert_eq!(
        t.ids("SELECT id FROM t WHERE id >= 20").unwrap(),
        [21, 23, 24]
    );
}

/// FC S1: a DV replaces the file's position delete files.
#[test]
fn a_dv_replaces_position_delete_files() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 4),
            positions(
                "d.parquet",
                0,
                2,
                &[("a.parquet", 0), ("a.parquet", 1)],
                None,
            ),
            dv("a.puffin", 0, 3, "a.parquet", &[3]),
        ],
        false,
    );
    assert_eq!(t.ids("SELECT id FROM t").unwrap(), [0, 1, 2]);
}

/// M1: a partition-scoped position delete applies only within its partition.
#[test]
fn partition_scoped_deletes_stay_in_their_partition() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            data("b.parquet", 1, 1, 10, 3),
            // Names b but is filed under partition 0: it never reaches b.
            positions(
                "d.parquet",
                0,
                2,
                &[("a.parquet", 0), ("b.parquet", 0)],
                None,
            ),
        ],
        true,
    );
    assert_eq!(t.ids("SELECT id FROM t").unwrap(), [1, 2, 10, 11, 12]);
}

/// A position delete older than the data file does not apply (`>=`), one
/// with the same sequence number does.
#[test]
fn position_delete_sequence_numbers() {
    let t = build(
        &[
            data("a.parquet", 0, 2, 0, 3),
            positions("old.parquet", 0, 1, &[("a.parquet", 0)], None),
            positions("same.parquet", 0, 2, &[("a.parquet", 1)], None),
        ],
        false,
    );
    assert_eq!(t.ids("SELECT id FROM t").unwrap(), [0, 2]);
}

#[test]
fn a_missing_delete_file_fails_the_query() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            positions("d.parquet", 0, 2, &[("a.parquet", 0)], None),
        ],
        false,
    );
    std::fs::remove_file(t.env.dir.path().join("d.parquet")).unwrap();
    let e = t.open_err();
    assert!(e.message().contains("d.parquet"), "{e}");
    assert_ne!(e.error_class(), ErrorClass::InternalError, "{e}");
}

#[test]
fn a_corrupt_dv_fails_with_javas_text() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            dv("a.puffin", 0, 2, "a.parquet", &[1]),
        ],
        false,
    );
    // Flip the last CRC byte.
    let path = t.env.dir.path().join("a.puffin");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, bytes).unwrap();
    let e = t.open_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergInvalidDeletes,
        "{e}"
    );
    assert!(
        e.message().starts_with("Invalid CRC for deletion vector "),
        "{e}"
    );
}

#[test]
fn two_dvs_for_one_file_fail() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            dv("a1.puffin", 0, 2, "a.parquet", &[1]),
            dv("a2.puffin", 0, 3, "a.parquet", &[2]),
        ],
        false,
    );
    let e = t.open_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergInvalidDeletes,
        "{e}"
    );
    assert!(
        e.message().starts_with("Can't index multiple DVs for "),
        "{e}"
    );
}

#[test]
fn a_dv_older_than_its_data_file_fails() {
    let t = build(
        &[
            data("a.parquet", 0, 3, 0, 3),
            dv("a.puffin", 0, 2, "a.parquet", &[1]),
        ],
        false,
    );
    let e = t.open_err();
    assert_eq!(
        e.message(),
        "DV data sequence number (2) must be greater than or equal to data file sequence number (3)"
    );
}

#[test]
fn equality_deletes_are_refused_only_where_they_apply() {
    let eq = File {
        name: "eq.parquet".to_owned(),
        p: 0,
        seq: 2,
        content: Content::Equality,
    };
    // Newer than the data: refused, naming both files.
    let t = build(&[data("a.parquet", 0, 1, 0, 3), eq.clone()], false);
    let e = t.open_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(
        e.message().contains("eq.parquet") && e.message().contains("a.parquet"),
        "{e}"
    );
    // Not newer (strictly): it applies to nothing, so the table reads.
    let t = build(&[data("a.parquet", 0, 2, 0, 3), eq], false);
    assert_eq!(t.ids("SELECT id FROM t").unwrap(), [0, 1, 2]);
}

/// One step of a generated history.
#[derive(Debug, Clone)]
enum Op {
    /// Append a data file of `rows` rows in partition `p`.
    Append { p: i32, rows: i64 },
    /// Delete some rows of live files: `picks` are (file index, position)
    /// candidates, reduced modulo the files that exist.
    Delete { kind: u8, picks: Vec<(usize, u64)> },
}

fn ops() -> impl Strategy<Value = Vec<Op>> {
    let op = prop_oneof![
        (0..2i32, 1..6i64).prop_map(|(p, rows)| Op::Append { p, rows }),
        (0..3u8, prop::collection::vec((0..8usize, 0..8u64), 1..5))
            .prop_map(|(kind, picks)| Op::Delete { kind, picks }),
    ];
    prop::collection::vec(op, 1..10)
}

/// Runs `ops` against a model and writes the files a v2/v3 writer would:
/// kind 0 a partition-scoped position delete file (rows of every picked file
/// in one partition), kind 1 a file-scoped one, kind 2 a DV that absorbs the
/// file's earlier deletes (as Java's DV writer does); once a file has a DV,
/// its later deletes are DVs too. Returns the files and the ids that survive.
fn history(ops: &[Op], partitioned: bool) -> (Vec<File>, BTreeSet<i64>) {
    struct Live {
        name: String,
        p: i32,
        first: i64,
        rows: i64,
        dv: Option<(String, i64)>,
        deleted: BTreeSet<u64>,
    }
    let mut files = Vec::new();
    let mut live: Vec<Live> = Vec::new();
    let mut next_id = 0;
    for (n, op) in ops.iter().enumerate() {
        let seq = i64::try_from(n).unwrap() + 1;
        match op {
            Op::Append { p, rows } => {
                let p = if partitioned { *p } else { 0 };
                let name = format!("d{n}.parquet");
                files.push(data(&name, p, seq, next_id, *rows));
                live.push(Live {
                    name,
                    p,
                    first: next_id,
                    rows: *rows,
                    dv: None,
                    deleted: BTreeSet::new(),
                });
                next_id += rows;
            }
            Op::Delete { kind, picks } => {
                if live.is_empty() {
                    continue;
                }
                // Positions may be past the end of the file (deleting nothing).
                let mut by_file: BTreeMap<usize, BTreeSet<u64>> = BTreeMap::new();
                for (f, pos) in picks {
                    by_file.entry(f % live.len()).or_default().insert(*pos);
                }
                let mut position_rows: HashMap<i32, Vec<(String, i64)>> = HashMap::new();
                for (k, (f, picked)) in by_file.into_iter().enumerate() {
                    let l = &mut live[f];
                    if *kind == 2 || l.dv.is_some() {
                        // A new DV with every earlier delete of the file;
                        // the old DV leaves the table.
                        if let Some((old, _)) = l.dv.take() {
                            files.retain(|x| x.name != old);
                        }
                        l.deleted.extend(picked.iter().copied());
                        let name = format!("dv{n}-{k}.puffin");
                        let all: Vec<u64> = l.deleted.iter().copied().collect();
                        files.push(dv(&name, l.p, seq, &l.name, &all));
                        l.dv = Some((name, seq));
                    } else {
                        l.deleted.extend(picked.iter().copied());
                        let rows = picked
                            .iter()
                            .map(|p| (l.name.clone(), i64::try_from(*p).unwrap()));
                        if *kind == 1 {
                            let rows: Vec<_> = rows.collect();
                            let refs: Vec<(&str, i64)> =
                                rows.iter().map(|(a, b)| (a.as_str(), *b)).collect();
                            files.push(positions(
                                &format!("pos{n}-{k}.parquet"),
                                l.p,
                                seq,
                                &refs,
                                Some(&l.name),
                            ));
                        } else {
                            position_rows.entry(l.p).or_default().extend(rows);
                        }
                    }
                }
                for (p, rows) in position_rows {
                    let refs: Vec<(&str, i64)> =
                        rows.iter().map(|(a, b)| (a.as_str(), *b)).collect();
                    files.push(positions(
                        &format!("part{n}-{p}.parquet"),
                        p,
                        seq,
                        &refs,
                        None,
                    ));
                }
            }
        }
    }
    let survivors = live
        .iter()
        .flat_map(|l| {
            (0..l.rows)
                .filter(|r| !l.deleted.contains(&u64::try_from(*r).unwrap()))
                .map(move |r| l.first + r)
        })
        .collect();
    (files, survivors)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// Design note §3.3: random histories of appends, partition- and
    /// file-scoped position deletes and DVs over random partitioned tables
    /// read the rows an in-memory model keeps.
    #[test]
    fn delete_application_matches_model(ops in ops(), partitioned in any::<bool>()) {
        let (files, survivors) = history(&ops, partitioned);
        let t = build(&files, partitioned);
        let got = t.ids("SELECT id FROM t").unwrap();
        prop_assert_eq!(got, survivors.into_iter().collect::<Vec<_>>());
    }
}
