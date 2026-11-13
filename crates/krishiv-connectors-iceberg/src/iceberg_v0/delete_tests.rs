// SPDX-License-Identifier: Apache-2.0
//! Tests of position deletes and deletion vectors through SQL (PR-25b; design
//! note §3.2–3.3): tables whose one snapshot lists data and delete files with
//! chosen sequence numbers, written with iceberg-rust's manifest writers; the
//! property test `delete_application_matches_model`; DV corruption, a missing
//! delete file, Java's assignment errors and equality deletes refused.

use std::collections::{BTreeMap, BTreeSet};
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
    /// Rows with ids `first..first + rows` in partition `p`; the row at
    /// position `null_at` has a null id instead.
    Data {
        first: i64,
        rows: i64,
        null_at: Option<i64>,
    },
    /// A position delete file: `(data file name, position)` rows.
    Positions {
        rows: Vec<(String, i64)>,
        /// Written with `referenced_data_file` (file-scoped).
        scoped: Option<String>,
    },
    /// A deletion vector of `data` (a Puffin blob at offset 4).
    Dv { data: String, positions: Vec<u64> },
    /// An equality delete file on `id` (field 1) holding these keys.
    Equality { ids: Vec<Option<i64>> },
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
            Content::Data {
                first,
                rows,
                null_at,
            } => {
                let ids: Vec<Option<i64>> = (*first..first + rows)
                    .map(|i| (Some(i - first) != *null_at).then_some(i))
                    .collect();
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
                let blob = encode_dv(&bitmap).unwrap();
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
            Content::Equality { ids } => {
                let b = batch(
                    vec![id_field("id", A::Int64, 1)],
                    vec![Arc::new(Int64Array::from(ids.clone()))],
                );
                let size = write_parquet(&path, &b);
                let mut d = builder(
                    DataContentType::EqualityDeletes,
                    DataFileFormat::Parquet,
                    size,
                    u64::try_from(ids.len()).unwrap(),
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

    /// The error of `SELECT id FROM t`: deletes are read, and refused, when
    /// a scan that reads their data files runs (PR-25d), not at open.
    fn query_err(&self) -> KrishivError {
        self.ids("SELECT id FROM t").unwrap_err()
    }
}

fn data(name: &str, p: i32, seq: i64, first: i64, rows: i64) -> File {
    File {
        name: name.to_owned(),
        p,
        seq,
        content: Content::Data {
            first,
            rows,
            null_at: None,
        },
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
    let e = t.query_err();
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
    let e = t.query_err();
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
    let e = t.query_err();
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
    let e = t.query_err();
    assert_eq!(
        e.message(),
        "DV data sequence number (2) must be greater than or equal to data file sequence number (3)"
    );
}

fn equality(name: &str, p: i32, seq: i64, ids: &[Option<i64>]) -> File {
    File {
        name: name.to_owned(),
        p,
        seq,
        content: Content::Equality { ids: ids.to_vec() },
    }
}

/// Design note §3.2–3.3: an equality delete removes the rows of strictly
/// older data files whose key it holds, in its partition or (unpartitioned
/// spec) everywhere; with position deletes and DVs the removed rows are a
/// union.
#[test]
fn equality_deletes_remove_rows_of_older_files() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 5),
            data("b.parquet", 0, 2, 10, 5),
            // Newer than a, not than b (strictly): 1, 3 and 11 go from a only.
            equality("e1.parquet", 0, 2, &[Some(1), Some(3), Some(11)]),
            // Newer than both.
            equality("e2.parquet", 0, 3, &[Some(12), Some(4), Some(99)]),
            positions("d1.parquet", 0, 3, &[("a.parquet", 0)], None),
            dv("b.puffin", 0, 3, "b.parquet", &[4]),
        ],
        false,
    );
    assert_eq!(t.ids("SELECT id FROM t").unwrap(), [2, 10, 11, 13]);
    assert_eq!(t.ids("SELECT count(*) FROM t").unwrap(), [4]);
    assert_eq!(t.ids("SELECT id FROM t WHERE id > 10").unwrap(), [11, 13]);
    // A limit is not pushed below the deletes (deleted rows would count
    // towards it); the equality columns and the file ordinal are read.
    assert_eq!(
        t.ids("SELECT count(*) FROM (SELECT id FROM t LIMIT 2)")
            .unwrap(),
        [2]
    );
    let p = t.env.provider(&t.table, &VersionSpec::Current).unwrap();
    let inner = p
        .equality_inner_pushdowns(krishiv_exec::ScanPushdowns {
            projection: Some(vec![krishiv_io::ColumnPath::new("p")]),
            limit: Some(2),
            ..krishiv_exec::ScanPushdowns::default()
        })
        .unwrap();
    assert_eq!(inner.limit, None);
    assert_eq!(
        inner.projection.unwrap(),
        ["p", "id", super::eq_deletes::FILE_ORDINAL].map(krishiv_io::ColumnPath::new)
    );
    // A column the query does not read is still the key.
    assert_eq!(t.ids("SELECT p FROM t").unwrap(), [0, 0, 0, 0]);
    // The query's columns and `_metadata` are what leaves the scan; the
    // equality columns and the file ordinal read for the deletes are not.
    let p = t.env.provider(&t.table, &VersionSpec::Current).unwrap();
    let scan = krishiv_exec::planner::TableProvider::scan(&p).unwrap();
    let pushdowns = krishiv_exec::ScanPushdowns {
        projection: Some(vec![krishiv_io::ColumnPath::new("p")]),
        metadata: krishiv_io::MetadataCols {
            file_name: true,
            ..krishiv_io::MetadataCols::default()
        },
        ..krishiv_exec::ScanPushdowns::default()
    };
    let schema = scan.output_schema(&pushdowns).unwrap();
    let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["p", "_metadata"]);
    assert_eq!(
        format!("{:?}", schema.field(1).data_type()),
        format!(
            "{:?}",
            A::Struct(vec![arrow_schema::Field::new("file_name", A::Utf8View, false)].into())
        )
    );
    assert_eq!(scan.build(pushdowns).unwrap().schema(), schema);
    assert_eq!(scan.table_schema().fields().len(), 2);
}

/// An equality delete of a partitioned spec stays in its partition.
#[test]
fn partitioned_equality_deletes_stay_in_their_partition() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            data("b.parquet", 1, 1, 10, 3),
            equality("e.parquet", 1, 2, &[Some(1), Some(11)]),
        ],
        true,
    );
    assert_eq!(t.ids("SELECT id FROM t").unwrap(), [0, 1, 2, 10, 12]);
}

/// Iceberg deletes compare `null` equal to `null` (not SQL's rule): a null
/// key removes the rows whose key is null, and only a null key does.
#[test]
fn a_null_equality_key_deletes_null_rows() {
    let with_null = |name: &str| File {
        name: name.to_owned(),
        p: 0,
        seq: 1,
        content: Content::Data {
            first: 0,
            rows: 3,
            null_at: Some(1),
        },
    };
    let rows = |t: &Table| {
        let p = t.env.provider(&t.table, &VersionSpec::Current).unwrap();
        let mut out =
            super::provider_tests::render(&provider_session(p).run("SELECT id FROM t").unwrap());
        out.sort();
        out
    };
    let t = build(
        &[
            with_null("a.parquet"),
            equality("e.parquet", 0, 2, &[None, Some(2)]),
        ],
        false,
    );
    assert_eq!(rows(&t), ["0"]);
    // Without a null key the null row stays.
    let t = build(
        &[
            with_null("a.parquet"),
            equality("e.parquet", 0, 2, &[Some(0)]),
        ],
        false,
    );
    assert_eq!(rows(&t), ["2", "NULL"]);
}

/// A missing equality delete file fails the query; it is never skipped.
#[test]
fn a_missing_equality_delete_file_fails() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            equality("e.parquet", 0, 2, &[Some(1)]),
        ],
        false,
    );
    std::fs::remove_file(t.env.at("e.parquet").strip_prefix("file://").unwrap()).unwrap();
    let e = t.query_err();
    assert!(e.message().contains("e.parquet"), "{e}");
}

/// Deletes are read for the data files a scan reads, after filter pruning
/// (PR-25d; review 25b S1): with every delete file of partition 0 gone, a
/// query whose filter prunes partition 0 reads none of them and succeeds,
/// and one that reads partition 0 fails naming a missing delete file.
#[test]
fn pruned_files_read_no_delete_files() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            data("c.parquet", 0, 1, 10, 3),
            data("b.parquet", 1, 1, 20, 3),
            dv("a.puffin", 0, 2, "a.parquet", &[0]),
            positions("pc.parquet", 0, 2, &[("c.parquet", 1)], None),
            equality("e.parquet", 0, 2, &[Some(12)]),
        ],
        true,
    );
    assert_eq!(t.ids("SELECT id FROM t").unwrap(), [1, 2, 10, 20, 21, 22]);
    for name in ["a.puffin", "pc.parquet", "e.parquet"] {
        std::fs::remove_file(t.env.dir.path().join(name)).unwrap();
    }
    assert_eq!(t.ids("SELECT id FROM t WHERE p = 1").unwrap(), [20, 21, 22]);
    let e = t.ids("SELECT id FROM t WHERE p = 0").unwrap_err();
    assert!(
        ["a.puffin", "pc.parquet", "e.parquet"]
            .iter()
            .any(|n| e.message().contains(n)),
        "{e}"
    );
}

/// The delete state is charged to the query's pool (PR-25d; reviews 25b
/// N2, 25c N7): it holds bytes for its keys while loaded and none once
/// dropped; a pool too small fails with the memory class; a load dropped
/// mid-read leaves the pool at zero with no consumer registered.
#[test]
fn delete_state_is_charged_to_the_query_pool() {
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
    let keys: Vec<Option<i64>> = (0..20_000).map(Some).collect();
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            dv("a.puffin", 0, 2, "a.parquet", &[0]),
            equality("e.parquet", 0, 2, &keys),
        ],
        false,
    );
    let p = t.env.provider(&t.table, &VersionSpec::Current).unwrap();
    let consumers = |pool: &Arc<dyn MemoryPool>| {
        pool.snapshot()
            .entries
            .iter()
            .filter(|e| e.is_consumer)
            .count()
    };
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let state = p.delete_state(&pool).unwrap();
    t.env.rt.block_on(state.ensure()).unwrap();
    assert!(pool.reserved() >= 20_000 * 8, "{}", pool.reserved());
    drop(state);
    assert_eq!((pool.reserved(), consumers(&pool)), (0, 0));

    let tiny: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
        limit: Some(64 << 10),
        ..MemoryConfig::default()
    });
    let state = p.delete_state(&tiny).unwrap();
    let Err(e) = t.env.rt.block_on(state.ensure()) else {
        panic!("a 64 KiB pool holds 20,000 keys")
    };
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
    assert_eq!((tiny.reserved(), consumers(&tiny)), (0, 0));

    // Cancelled: polled until it holds memory, then dropped mid-load.
    let state = p.delete_state(&pool).unwrap();
    let held = t.env.rt.block_on(async {
        let mut load = std::pin::pin!(state.ensure());
        loop {
            assert!(futures::poll!(load.as_mut()).is_pending());
            if pool.reserved() > 0 {
                break pool.reserved();
            }
            tokio::task::yield_now().await;
        }
    });
    assert!(held > 0 && state.loaded().is_none());
    assert_eq!((pool.reserved(), consumers(&pool)), (0, 0));
}

/// A distributed task's scan of a reordered subset of the files
/// (`with_file_splits`) applies each file's equality deletes, not those of
/// the file at the same position in the full list (review 25c N8): the file
/// ordinal is global, through `FileConstants` keyed by path.
#[test]
fn equality_deletes_follow_files_through_given_splits() {
    use krishiv_exec::ScanPushdown;
    use krishiv_exec::planner::TableProvider;
    use krishiv_io::split::{FilePartition, FileSplit};

    #[derive(Debug)]
    struct Splits(super::provider::IcebergTableProvider, Vec<usize>);
    impl TableProvider for Splits {
        fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
            let files: Vec<_> = self.1.iter().map(|&i| self.0.files()[i].clone()).collect();
            let partitions = files
                .iter()
                .enumerate()
                .map(|(i, f)| FilePartition {
                    index: i,
                    splits: vec![FileSplit {
                        file: i,
                        range: krishiv_io::ByteRange::whole(f.size),
                    }],
                })
                .collect();
            self.0.scan()?.with_file_splits(files, partitions)
        }
    }

    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            data("b.parquet", 0, 3, 10, 3),
            data("c.parquet", 0, 1, 20, 3),
            // Newer than a and c, not than b.
            equality("e.parquet", 0, 2, &[Some(1), Some(11), Some(21)]),
        ],
        false,
    );
    let p = t.env.provider(&t.table, &VersionSpec::Current).unwrap();
    let names: Vec<String> = p.files().iter().map(|f| f.path.to_string()).collect();
    let at = |n: &str| names.iter().position(|p| p.ends_with(n)).unwrap();
    let (b, c) = (at("b.parquet"), at("c.parquet"));
    assert_eq!((at("a.parquet"), b, c), (0, 1, 2));
    let schema = p.snapshot().schema.clone();
    let session =
        super::provider_tests::session(vec![("t", schema, Arc::new(Splits(p, vec![b, c])))]);
    let mut ids: Vec<i64> =
        super::provider_tests::render(&session.run("SELECT id FROM t").unwrap())
            .iter()
            .map(|r| r.parse().unwrap())
            .collect();
    ids.sort_unstable();
    assert_eq!(ids, [10, 11, 12, 20, 22]);
}

/// The row estimate comes from the manifests (PR-25d; design note §2.4):
/// the data files' record counts less the position deletes and DVs that
/// apply (each delete file once), equality deletes not subtracted; scan
/// units are the files' split offsets.
#[test]
fn row_estimate_from_iceberg_metadata() {
    use krishiv_exec::planner::TableProvider;
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 5),
            data("b.parquet", 0, 1, 10, 5),
            data("c.parquet", 0, 1, 20, 5),
            // Two rows, applying to a and b: subtracted once.
            positions(
                "d.parquet",
                0,
                2,
                &[("a.parquet", 0), ("b.parquet", 1)],
                None,
            ),
            dv("c.puffin", 0, 2, "c.parquet", &[3]),
            equality("e.parquet", 0, 2, &[Some(1), Some(2)]),
        ],
        false,
    );
    let p = t.env.provider(&t.table, &VersionSpec::Current).unwrap();
    assert_eq!(p.statistics().unwrap().rows, 15 - 2 - 1);
    assert_eq!(p.scan_units(), Some(3));
    let plain = build(&[data("a.parquet", 0, 1, 0, 5)], false);
    let p = plain
        .env
        .provider(&plain.table, &VersionSpec::Current)
        .unwrap();
    assert_eq!(p.statistics().unwrap().rows, 5);
}

/// `EXPLAIN` shows the Iceberg scan's snapshot, the data files read after
/// pruning out of all, and the delete files that apply to them by kind
/// (PR-25d).
#[test]
fn explain_shows_snapshot_files_and_deletes() {
    let t = build(
        &[
            data("a.parquet", 0, 1, 0, 3),
            data("c.parquet", 0, 1, 10, 3),
            data("b.parquet", 1, 1, 20, 3),
            dv("a.puffin", 0, 2, "a.parquet", &[0]),
            positions("pc.parquet", 0, 2, &[("c.parquet", 1)], None),
            equality("e.parquet", 0, 2, &[Some(12)]),
        ],
        true,
    );
    let p = t.env.provider(&t.table, &VersionSpec::Current).unwrap();
    let s = provider_session(p);
    let all = s.explain("SELECT id FROM t");
    assert!(
        all.contains("Snapshot: 1000, DataFiles: 3/3, DeleteFiles: [position=1, dv=1, equality=1]"),
        "{all}"
    );
    let one = s.explain("SELECT id FROM t WHERE p = 1");
    assert!(
        one.contains("Snapshot: 1000, DataFiles: 1/3, DeleteFiles: [position=0, dv=0, equality=0]"),
        "{one}"
    );
}

/// One step of a generated history (commit `n` has sequence number `n + 1`).
#[derive(Debug, Clone)]
enum Op {
    /// Append a data file of `rows` rows in partition `p`, with a data
    /// sequence number `back` below the commit's (a rewrite keeping its
    /// sequence number), and optionally a position delete of `same` in the
    /// same commit (the row-delta shape: equal sequence numbers).
    Append {
        p: i32,
        rows: i64,
        back: i64,
        same: Option<u64>,
    },
    /// A delete of the picked (file index, position) pairs, reduced modulo
    /// the live files: kind 0 one partition-scoped position delete file
    /// (filed under the first pick's partition, so picks in other partitions
    /// are rows Java never applies), 1 file-scoped files, 2 a DV absorbing
    /// the file's earlier DV, 3 a DV of the new picks only (earlier position
    /// delete files stay and are replaced). A `stale` position delete has
    /// data sequence number 1, below every file appended after the first
    /// commit (it does not apply to them).
    Delete {
        kind: u8,
        stale: bool,
        picks: Vec<(usize, u64)>,
    },
    /// An equality delete of the ids `keys` (reduced modulo the ids written
    /// so far, plus one past them), in partition `p`, with a data sequence
    /// number `back` below the commit's (so some are not newer than files
    /// they would otherwise apply to: equality deletes need strictly newer).
    Equality { p: i32, keys: Vec<i64>, back: i64 },
}

fn ops() -> impl Strategy<Value = Vec<Op>> {
    let op = prop_oneof![
        (0..2i32, 1..6i64, 0..3i64, prop::option::of(0..6u64)).prop_map(|(p, rows, back, same)| {
            Op::Append {
                p,
                rows,
                back,
                same,
            }
        }),
        (
            0..4u8,
            any::<bool>(),
            prop::collection::vec((0..8usize, 0..8u64), 1..5)
        )
            .prop_map(|(kind, stale, picks)| Op::Delete { kind, stale, picks }),
        (0..2i32, prop::collection::vec(0..64i64, 1..5), 0..3i64)
            .prop_map(|(p, keys, back)| Op::Equality { p, keys, back }),
    ];
    prop::collection::vec(op, 1..12)
}

/// A delete as the model keeps it.
enum ModelDelete {
    /// Position delete rows (data file, position), file-scoped to `scoped`
    /// or scoped to partition `p`.
    Positions {
        seq: i64,
        p: i32,
        scoped: Option<String>,
        rows: Vec<(String, u64)>,
    },
    /// Equality delete keys (ids) of partition `p` (`None`: global).
    Equality {
        seq: i64,
        p: Option<i32>,
        keys: BTreeSet<i64>,
    },
}

/// Writes `ops` as files and computes, independently of the reader, the ids
/// that survive under the spec's rules (design note §3.2 with FC S1, M1):
/// a data file with a DV loses exactly the DV's positions; otherwise it loses
/// the rows naming it in every position delete file with `seq >= data seq`
/// that is scoped to it or to its partition.
fn history(ops: &[Op], partitioned: bool) -> (Vec<File>, BTreeSet<i64>) {
    struct Live {
        name: String,
        p: i32,
        seq: i64,
        first: i64,
        rows: i64,
        /// The current DV's file name and positions.
        dv: Option<(String, BTreeSet<u64>)>,
    }
    let mut files = Vec::new();
    let mut live: Vec<Live> = Vec::new();
    let mut model: Vec<ModelDelete> = Vec::new();
    let mut next_id = 0;
    for (n, op) in ops.iter().enumerate() {
        let commit = i64::try_from(n).unwrap() + 1;
        match op {
            Op::Append {
                p,
                rows,
                back,
                same,
            } => {
                let p = if partitioned { *p } else { 0 };
                let seq = (commit - back).max(1);
                let name = format!("d{n}.parquet");
                files.push(data(&name, p, seq, next_id, *rows));
                if let Some(pos) = same {
                    let rows = vec![(name.clone(), *pos)];
                    files.push(positions(
                        &format!("same{n}.parquet"),
                        p,
                        seq,
                        &[(name.as_str(), i64::try_from(*pos).unwrap())],
                        None,
                    ));
                    model.push(ModelDelete::Positions {
                        seq,
                        p,
                        scoped: None,
                        rows,
                    });
                }
                live.push(Live {
                    name,
                    p,
                    seq,
                    first: next_id,
                    rows: *rows,
                    dv: None,
                });
                next_id += rows;
            }
            Op::Equality { p, keys, back } => {
                let seq = (commit - back).max(1);
                let keys: BTreeSet<i64> = keys.iter().map(|k| k % (next_id + 1)).collect();
                let ids: Vec<Option<i64>> = keys.iter().copied().map(Some).collect();
                let p = partitioned.then_some(*p);
                files.push(equality(
                    &format!("eq{n}.parquet"),
                    p.unwrap_or(0),
                    seq,
                    &ids,
                ));
                model.push(ModelDelete::Equality { seq, p, keys });
            }
            Op::Delete { kind, stale, picks } => {
                if live.is_empty() {
                    continue;
                }
                let seq = if *stale { 1 } else { commit };
                let mut by_file: BTreeMap<usize, BTreeSet<u64>> = BTreeMap::new();
                for (f, pos) in picks {
                    by_file.entry(f % live.len()).or_default().insert(*pos);
                }
                match kind {
                    0 => {
                        let p = live[*by_file.keys().next().unwrap()].p;
                        let rows: Vec<(String, u64)> = by_file
                            .iter()
                            .flat_map(|(f, ps)| ps.iter().map(|x| (live[*f].name.clone(), *x)))
                            .collect();
                        let refs: Vec<(&str, i64)> = rows
                            .iter()
                            .map(|(a, b)| (a.as_str(), i64::try_from(*b).unwrap()))
                            .collect();
                        files.push(positions(&format!("part{n}.parquet"), p, seq, &refs, None));
                        model.push(ModelDelete::Positions {
                            seq,
                            p,
                            scoped: None,
                            rows,
                        });
                    }
                    1 => {
                        for (k, (f, ps)) in by_file.iter().enumerate() {
                            let l = &live[*f];
                            let rows: Vec<(String, u64)> =
                                ps.iter().map(|x| (l.name.clone(), *x)).collect();
                            let refs: Vec<(&str, i64)> = rows
                                .iter()
                                .map(|(a, b)| (a.as_str(), i64::try_from(*b).unwrap()))
                                .collect();
                            files.push(positions(
                                &format!("pos{n}-{k}.parquet"),
                                l.p,
                                seq,
                                &refs,
                                Some(&l.name),
                            ));
                            model.push(ModelDelete::Positions {
                                seq,
                                p: l.p,
                                scoped: Some(l.name.clone()),
                                rows,
                            });
                        }
                    }
                    _ => {
                        // A DV has the commit's sequence number, never below
                        // its data file's.
                        for (k, (f, ps)) in by_file.into_iter().enumerate() {
                            let l = &mut live[f];
                            let mut all = ps;
                            if let Some((old, prior)) = l.dv.take() {
                                files.retain(|x| x.name != old);
                                if *kind == 2 {
                                    all.extend(prior);
                                }
                            }
                            let name = format!("dv{n}-{k}.puffin");
                            let v: Vec<u64> = all.iter().copied().collect();
                            files.push(dv(&name, l.p, commit, &l.name, &v));
                            l.dv = Some((name, all));
                        }
                    }
                }
            }
        }
    }
    let survivors = live
        .iter()
        .flat_map(|l| {
            let deleted: BTreeSet<u64> = match &l.dv {
                Some((_, ps)) => ps.clone(),
                None => model
                    .iter()
                    .flat_map(|m| -> Vec<u64> {
                        match m {
                            ModelDelete::Positions {
                                seq,
                                p,
                                scoped,
                                rows,
                            } => {
                                let applies = *seq >= l.seq
                                    && match scoped {
                                        Some(s) => *s == l.name,
                                        None => *p == l.p,
                                    };
                                rows.iter()
                                    .filter(|(f, _)| applies && *f == l.name)
                                    .map(|(_, x)| *x)
                                    .collect()
                            }
                            ModelDelete::Equality { .. } => Vec::new(),
                        }
                    })
                    .collect(),
            };
            // Equality deletes: strictly newer, same partition or global;
            // their rows are a union with the position deletes and the DV.
            let by_key: BTreeSet<i64> = model
                .iter()
                .filter_map(|m| match m {
                    ModelDelete::Equality { seq, p, keys }
                        if *seq > l.seq && p.is_none_or(|p| p == l.p) =>
                    {
                        Some(keys)
                    }
                    _ => None,
                })
                .flatten()
                .copied()
                .collect();
            (0..l.rows)
                .filter(move |r| !deleted.contains(&u64::try_from(*r).unwrap()))
                .map(move |r| l.first + r)
                .filter(move |id| !by_key.contains(id))
        })
        .collect();
    (files, survivors)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 96, ..ProptestConfig::default() })]

    /// Design note §3.3: random histories of appends (some with deletes in
    /// the same commit or with older sequence numbers), partition- and
    /// file-scoped position deletes (some older than the data, some naming
    /// files of other partitions) and DVs (absorbing or replacing older
    /// deletes) over random partitioned tables read the rows a model of the
    /// spec's rules keeps. Equality deletes (some not strictly newer, some
    /// partition-scoped) remove the rows of their keys besides.
    #[test]
    fn delete_application_matches_model(ops in ops(), partitioned in any::<bool>()) {
        let (files, survivors) = history(&ops, partitioned);
        let t = build(&files, partitioned);
        let got = t.ids("SELECT id FROM t").unwrap();
        prop_assert_eq!(got, survivors.into_iter().collect::<Vec<_>>());
    }
}
