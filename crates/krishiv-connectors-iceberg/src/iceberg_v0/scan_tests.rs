// SPDX-License-Identifier: Apache-2.0
//! Tests of scan planning against files iceberg-rust writes: manifests with null
//! sequence numbers read back and inherited, the transforms against iceberg-rust's
//! own, and a table planned and time-travelled end to end through the storage.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal,
    ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Type,
};
use krishiv_io::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};

use super::scan::parse_manifest;
use super::storage::StorageRegistration;
use super::table::IcebergTable;
use super::tests::local;
use crate::plan::{
    EntryStatus, IcebergType, LiveEntry, ManifestContent, ManifestFileEntry, ManifestFilter,
    PartitionValue, PruneOptions, Transform, VersionSpec, apply_transform, inherit,
};

fn schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "x", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap()
}

fn spec(schema: &Schema) -> PartitionSpec {
    PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("id", "id_bucket", iceberg::spec::Transform::Bucket(4))
        .unwrap()
        .add_partition_field("x", "x", iceberg::spec::Transform::Identity)
        .unwrap()
        .build()
        .unwrap()
}

fn data_file(path: &str, rows: u64, x: Option<i32>, lower: i32, upper: i32) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_owned())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(rows)
        .partition(Struct::from_iter([
            Some(Literal::int(2)),
            x.map(Literal::int),
        ]))
        .partition_spec_id(0)
        .value_counts(HashMap::from([(3, rows)]))
        .null_value_counts(HashMap::from([(3, 0)]))
        .lower_bounds(HashMap::from([(3, Datum::int(lower))]))
        .upper_bounds(HashMap::from([(3, Datum::int(upper))]))
        .build()
        .unwrap()
}

fn list_entry(path: &str, seq: i64, first_row_id: Option<u64>) -> ManifestFileEntry {
    ManifestFileEntry {
        path: path.to_owned(),
        length: 0,
        spec_id: 0,
        content: ManifestContent::Data,
        sequence_number: seq,
        min_sequence_number: seq,
        added_snapshot_id: 42,
        added_files: None,
        existing_files: None,
        deleted_files: None,
        first_row_id,
        partitions: None,
        encrypted: false,
    }
}

async fn read(reg: &StorageRegistration, path: &str) -> Vec<u8> {
    reg.file_io()
        .new_input(path)
        .unwrap()
        .read()
        .await
        .unwrap()
        .to_vec()
}

/// FC B2: manifests written with null sequence numbers and snapshot ids (as a
/// commit writes them), parsed with iceberg-rust's public parser and inherited by
/// Krishiv, give the spec's values; v1 manifests give 0; v3 assigns row ids.
#[tokio::test]
async fn manifest_inheritance_matches_spec() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let io = reg.file_io();
    let schema = Arc::new(schema());
    let spec = spec(&schema);
    let at = |n: &str| format!("file://{}/{n}", dir.path().display());

    // v2: an added file with no sequence number, an existing one, a deleted one
    // and an added one with an explicit data sequence number.
    let path = at("m2.avro");
    let mut w = ManifestWriterBuilder::new(
        io.new_output(&path).unwrap(),
        None,
        Arc::clone(&schema),
        spec.clone(),
    )
    .build_v2_data();
    w.add_file(data_file("a", 10, Some(1), 1, 5), -1).unwrap();
    w.add_existing_file(data_file("b", 20, Some(2), 1, 5), 5, 3, Some(3))
        .unwrap();
    w.add_delete_file(data_file("c", 30, Some(3), 1, 5), 2, Some(2))
        .unwrap();
    w.add_file(data_file("d", 40, None, 1, 5), 6).unwrap();
    w.write_manifest_file().await.unwrap();

    let parsed = parse_manifest(&read(&reg, &path).await, 0).unwrap();
    // As written: iceberg-rust's parser leaves them null (it does not inherit).
    assert_eq!(parsed[0].status, EntryStatus::Added);
    assert_eq!(
        (parsed[0].sequence_number, parsed[0].snapshot_id),
        (None, None)
    );
    assert_eq!(parsed[3].sequence_number, Some(6));
    assert_eq!(parsed[3].file_sequence_number, None);
    let x1 = PartitionValue::Value(1i32.to_le_bytes().to_vec());
    assert_eq!(
        parsed[0].file.partition,
        vec![PartitionValue::Value(2i32.to_le_bytes().to_vec()), x1]
    );
    assert_eq!(parsed[3].file.partition[1], PartitionValue::Null);
    assert_eq!(parsed[0].file.lower_bounds[&3], 1i32.to_le_bytes().to_vec());

    let live = inherit(&list_entry(&path, 7, None), parsed).unwrap();
    let got: Vec<_> = live
        .iter()
        .map(|e| match e {
            LiveEntry::Data(d) => (
                d.file.path.clone(),
                d.data_sequence_number,
                d.file_sequence_number,
                d.snapshot_id,
                d.status,
            ),
            LiveEntry::Delete(_) => unreachable!(),
        })
        .collect();
    assert_eq!(
        got,
        vec![
            ("a".to_owned(), Some(7), Some(7), 42, EntryStatus::Added),
            ("b".to_owned(), Some(3), Some(3), 5, EntryStatus::Existing),
            ("d".to_owned(), Some(6), Some(7), 42, EntryStatus::Added),
        ]
    );

    // v1: no sequence numbers anywhere; the manifest list gives 0.
    let path = at("m1.avro");
    let mut w = ManifestWriterBuilder::new(
        io.new_output(&path).unwrap(),
        Some(3),
        Arc::clone(&schema),
        spec.clone(),
    )
    .build_v1();
    w.add_file(data_file("a", 10, Some(1), 1, 5), -1).unwrap();
    w.write_manifest_file().await.unwrap();
    let live = inherit(
        &list_entry(&path, 0, None),
        parse_manifest(&read(&reg, &path).await, 0).unwrap(),
    )
    .unwrap();
    let LiveEntry::Data(d) = &live[0] else {
        unreachable!()
    };
    assert_eq!(
        (
            d.data_sequence_number,
            d.file_sequence_number,
            d.snapshot_id
        ),
        (Some(0), Some(0), 3)
    );

    // v3: row ids assigned in order from the manifest's first_row_id.
    let path = at("m3.avro");
    let mut w = ManifestWriterBuilder::new(
        io.new_output(&path).unwrap(),
        None,
        Arc::clone(&schema),
        spec,
    )
    .build_v3_data();
    w.add_file(data_file("a", 10, Some(1), 1, 5), -1).unwrap();
    w.add_file(data_file("b", 5, Some(1), 1, 5), -1).unwrap();
    w.write_manifest_file().await.unwrap();
    let live = inherit(
        &list_entry(&path, 4, Some(1000)),
        parse_manifest(&read(&reg, &path).await, 0).unwrap(),
    )
    .unwrap();
    let ids: Vec<_> = live
        .iter()
        .map(|e| match e {
            LiveEntry::Data(d) => d.file.first_row_id,
            LiveEntry::Delete(_) => None,
        })
        .collect();
    assert_eq!(ids, vec![Some(1000), Some(1010)]);
}

/// Krishiv's transforms of literals equal iceberg-rust's.
#[test]
fn transforms_match_iceberg_rust() {
    use iceberg::spec::Transform as T;
    use iceberg::transform::create_transform_function;
    let cases: Vec<(Datum, IcebergType, ScalarValue)> = vec![
        (Datum::int(34), IcebergType::Int, ScalarValue::Int32(34)),
        (Datum::int(-7), IcebergType::Int, ScalarValue::Int32(-7)),
        (Datum::long(34), IcebergType::Long, ScalarValue::Int64(34)),
        (
            Datum::long(-1_234_567),
            IcebergType::Long,
            ScalarValue::Int64(-1_234_567),
        ),
        (
            Datum::string("iceberg"),
            IcebergType::String,
            ScalarValue::Utf8("iceberg".to_owned()),
        ),
        (
            Datum::string("añb€x"),
            IcebergType::String,
            ScalarValue::Utf8("añb€x".to_owned()),
        ),
        (
            Datum::date(17_486),
            IcebergType::Date,
            ScalarValue::Date32(17_486),
        ),
        (Datum::date(-3), IcebergType::Date, ScalarValue::Date32(-3)),
        (
            Datum::timestamptz_micros(1_510_871_468_123_456),
            IcebergType::Timestamptz,
            ScalarValue::TimestampMicros(1_510_871_468_123_456),
        ),
        (
            Datum::timestamp_micros(-1),
            IcebergType::Timestamp,
            ScalarValue::TimestampMicros(-1),
        ),
        (
            Datum::timestamp_micros(-86_400_000_001),
            IcebergType::Timestamp,
            ScalarValue::TimestampMicros(-86_400_000_001),
        ),
        (
            Datum::binary([0u8, 1, 2, 3]),
            IcebergType::Binary,
            ScalarValue::Binary(vec![0, 1, 2, 3]),
        ),
        (
            Datum::decimal_from_str("14.20").unwrap(),
            IcebergType::Decimal {
                precision: 38,
                scale: 2,
            },
            ScalarValue::Decimal128 {
                unscaled: 1420,
                precision: 38,
                scale: 2,
            },
        ),
        (
            Datum::decimal_from_str("-0.05").unwrap(),
            IcebergType::Decimal {
                precision: 38,
                scale: 2,
            },
            ScalarValue::Decimal128 {
                unscaled: -5,
                precision: 38,
                scale: 2,
            },
        ),
    ];
    let transforms = [
        (T::Bucket(16), Transform::Bucket(16)),
        (T::Bucket(7), Transform::Bucket(7)),
        (T::Truncate(3), Transform::Truncate(3)),
        (T::Year, Transform::Year),
        (T::Month, Transform::Month),
        (T::Day, Transform::Day),
        (T::Hour, Transform::Hour),
    ];
    let mut compared = 0;
    for (datum, ty, value) in &cases {
        for (theirs, ours) in &transforms {
            // The spec defines truncate on int, long, decimal, string and binary
            // only (iceberg-rust also truncates a date's day count).
            if matches!(ours, Transform::Truncate(_))
                && matches!(
                    ty,
                    IcebergType::Date | IcebergType::Timestamp | IcebergType::Timestamptz
                )
            {
                continue;
            }
            let f = create_transform_function(theirs).unwrap();
            // iceberg-rust has no literal truncate for binaries; nothing to compare.
            let Ok(Some(expected)) = f.transform_literal(datum) else {
                continue;
            };
            let result_ty =
                crate::plan::PartitionFieldInfo::new(0, 0, String::new(), *ours, Some(ty.clone()))
                    .result_type
                    .unwrap();
            let expected = crate::plan::decode(&expected.to_bytes().unwrap(), &result_ty);
            assert_eq!(
                apply_transform(*ours, ty, value),
                expected,
                "{ours:?} {datum}"
            );
            compared += 1;
        }
    }
    assert!(compared >= 50, "{compared}");
}

/// A table with two appends through iceberg-rust's in-memory catalog: planning
/// prunes by partition and metrics, and each version reads its own files.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_prunes_and_time_travels() {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let factory = Arc::new(super::storage::KrishivStorageFactory {
        name: reg.name().to_owned(),
    });
    let catalog = iceberg::memory::MemoryCatalogBuilder::default()
        .with_storage_factory(factory)
        .with_runtime(reg.runtime().unwrap())
        .load(
            "mem",
            [(
                iceberg::memory::MEMORY_CATALOG_WAREHOUSE.to_owned(),
                format!("file://{}", dir.path().display()),
            )]
            .into(),
        )
        .await
        .unwrap();
    let ns = NamespaceIdent::new("db".to_owned());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let schema = schema();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("t".to_owned())
                .schema(schema.clone())
                .partition_spec(spec(&schema).into_unbound())
                .build(),
        )
        .await
        .unwrap();
    let at = |n: &str| format!("file://{}/db/t/data/{n}", dir.path().display());
    let mut table = table;
    let mut snapshots = Vec::new();
    for files in [
        vec![
            data_file(&at("a"), 10, Some(1), 1, 1),
            data_file(&at("b"), 5, Some(2), 2, 2),
        ],
        vec![data_file(&at("c"), 7, Some(3), 3, 3)],
    ] {
        let tx = Transaction::new(&table);
        let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
        table = tx.commit(&catalog).await.unwrap();
        let s = table.metadata().current_snapshot().unwrap();
        snapshots.push((s.snapshot_id(), s.timestamp_ms()));
    }
    let location = table.metadata_location().unwrap().to_owned();
    let t = IcebergTable::load(&reg, &location, "db", "t")
        .await
        .unwrap();

    let x = |op, v| PushdownExpr::cmp(ColumnPath::new("x"), op, ScalarValue::Int32(v));
    let paths = |p: &crate::plan::ScanPlan| {
        let mut v: Vec<_> = p
            .tasks
            .iter()
            .map(|t| t.file.file.path.rsplit('/').next().unwrap().to_owned())
            .collect();
        v.sort();
        v
    };
    let current = t.snapshot(&VersionSpec::Current).unwrap();
    assert_eq!(current.snapshot_id, Some(snapshots[1].0));
    assert_eq!(current.format_version, 2);
    let all = t
        .plan(&current, None, ManifestFilter::All, PruneOptions::default())
        .await
        .unwrap();
    assert_eq!(paths(&all), ["a", "b", "c"]);
    assert_eq!(all.manifests_total, 2);
    // Sequence numbers are inherited from each commit's manifest.
    let mut seqs: Vec<_> = all
        .tasks
        .iter()
        .map(|t| (t.file.data_sequence_number, t.file.snapshot_id))
        .collect();
    seqs.sort_unstable();
    assert_eq!(
        seqs,
        [
            (Some(1), snapshots[0].0),
            (Some(1), snapshots[0].0),
            (Some(2), snapshots[1].0)
        ]
    );
    // x >= 3 keeps only c; the first manifest's summary (x in 1..=2) prunes it.
    let p = t
        .plan(
            &current,
            Some(&x(CmpOp::GtEq, 3)),
            ManifestFilter::All,
            PruneOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(paths(&p), ["c"]);
    assert_eq!(p.manifests_pruned, 1);
    // x = 2 is only b (a is pruned by its partition value and metrics).
    let p = t
        .plan(
            &current,
            Some(&x(CmpOp::Eq, 2)),
            ManifestFilter::All,
            PruneOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(paths(&p), ["b"]);
    assert_eq!(p.files_pruned, 1);
    assert_eq!(p.tasks[0].residual, Some(x(CmpOp::Eq, 2)));

    // Time travel: by id, by timestamp (micros), and the errors.
    let first = t.snapshot(&VersionSpec::Id(snapshots[0].0)).unwrap();
    let p = t
        .plan(&first, None, ManifestFilter::All, PruneOptions::default())
        .await
        .unwrap();
    assert_eq!(paths(&p), ["a", "b"]);
    let by_ts = t
        .snapshot(&VersionSpec::TimestampMicros(snapshots[0].1 * 1000 + 999))
        .unwrap();
    assert_eq!(by_ts.snapshot_id, Some(snapshots[0].0));
    let e = t.snapshot(&VersionSpec::Id(12_345)).unwrap_err();
    assert_eq!(e.message(), "Cannot find snapshot with ID 12345");
    let e = t
        .snapshot(&VersionSpec::TimestampMicros((snapshots[0].1 - 1) * 1000))
        .unwrap_err();
    assert!(
        e.message()
            .starts_with("Cannot find a snapshot older than "),
        "{e}"
    );
    // `main` is a branch at the current snapshot.
    let main = t.snapshot(&VersionSpec::Ref("main".to_owned())).unwrap();
    assert_eq!(main.snapshot_id, Some(snapshots[1].0));

    // Refs written by another writer: a tag and a branch at the first snapshot.
    let text = String::from_utf8(read(&reg, &location).await).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["refs"]["v1"] = serde_json::json!({"snapshot-id": snapshots[0].0, "type": "tag"});
    json["refs"]["dev"] = serde_json::json!({"snapshot-id": snapshots[0].0, "type": "branch"});
    let tagged = format!(
        "file://{}/db/t/metadata/99-tagged.metadata.json",
        dir.path().display()
    );
    reg.file_io()
        .new_output(&tagged)
        .unwrap()
        .write(bytes::Bytes::from(serde_json::to_vec(&json).unwrap()))
        .await
        .unwrap();
    let t = IcebergTable::load(&reg, &tagged, "db", "t").await.unwrap();
    for name in ["v1", "dev"] {
        let s = t.snapshot(&VersionSpec::Ref(name.to_owned())).unwrap();
        assert_eq!(s.snapshot_id, Some(snapshots[0].0), "{name}");
    }
    let e = t
        .snapshot(&VersionSpec::Ref("nope".to_owned()))
        .unwrap_err();
    assert_eq!(
        e.message(),
        "Cannot find matching snapshot ID or reference name for version nope"
    );
    // A dangling ref never loads (iceberg-rust validates refs at parse), so the
    // resolver's own check (plan::tests) is the second line.
    json["refs"]["gone"] = serde_json::json!({"snapshot-id": 1, "type": "tag"});
    let dangling = format!(
        "file://{}/db/t/metadata/99-dangling.metadata.json",
        dir.path().display()
    );
    reg.file_io()
        .new_output(&dangling)
        .unwrap()
        .write(bytes::Bytes::from(serde_json::to_vec(&json).unwrap()))
        .await
        .unwrap();
    assert!(
        IcebergTable::load(&reg, &dangling, "db", "t")
            .await
            .is_err()
    );
}
