// SPDX-License-Identifier: Apache-2.0
//! Tests of the PR-25a review fixes against tables iceberg-rust writes: the
//! catalog parse limits on metadata, manifest lists and manifests
//! (`catalog_parse_limits`), refs and manifest lists of format-v1 tables and
//! of snapshots from before an upgrade, the table's metrics modes in a scan, and
//! the refusal of a mixed-spec identity column a file lacks.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::Int32Array;
use arrow_schema::DataType as A;
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
    ManifestListWriter, ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Struct,
    Transform as IceTransform, Type,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, NamespaceIdent, TableCreation};
use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;

use super::provider_tests::{
    Env, batch, data_file, env, id_field, int64s, ixn_schema, provider_session, strings,
    write_parquet,
};
use super::scan::read_manifest_list;
use super::table::IcebergTable;
use crate::plan::{ManifestFilter, PruneOptions, VersionSpec};

impl Env {
    fn create_with(&self, creation: TableCreation) -> iceberg::table::Table {
        self.rt.block_on(async {
            self.catalog
                .create_table(&NamespaceIdent::new("db".to_owned()), creation)
                .await
                .unwrap()
        })
    }

    /// `t` with one Parquet file of `(id, x, name)` rows appended.
    fn append_rows(&self, t: &iceberg::table::Table, name: &str) -> iceberg::table::Table {
        let rows = batch(
            vec![
                id_field("id", A::Int64, 1),
                id_field("x", A::Int32, 2),
                id_field("name", A::Utf8, 3),
            ],
            vec![
                int64s(&[1]),
                Arc::new(Int32Array::from(vec![1])),
                strings(&[name]),
            ],
        );
        let path = self.at(&format!("{name}.parquet"));
        let size = write_parquet(&path, &rows);
        self.append(t, vec![data_file(&path, size, 1, Struct::empty())])
    }
}

fn local_size(location: &str) -> u64 {
    std::fs::metadata(location.strip_prefix("file://").unwrap())
        .unwrap()
        .len()
}

/// Design note §10 row 25a: a metadata file, manifest list or manifest over
/// `krishiv.catalog.maxMetadataBytes` fails with `KRISHIV_CATALOG_LIMIT_EXCEEDED`
/// before it is parsed. (The decoded entries are charged to a memory
/// reservation in PR-25d.)
#[test]
fn catalog_parse_limits() {
    let env = env();
    let t = env.create("t", ixn_schema(), None);
    let t = env.append_rows(&t, "a");
    let location = t.metadata_location().unwrap().to_owned();
    let list = t
        .metadata()
        .current_snapshot()
        .unwrap()
        .manifest_list()
        .to_owned();
    let manifests = env
        .rt
        .block_on(read_manifest_list(&env.reg.file_io(), &list, 2, u64::MAX))
        .unwrap();
    let (meta, list_size) = (local_size(&location), local_size(&list));
    let manifest_size = u64::try_from(manifests[0].length).unwrap();

    // The metadata file, through the configuration.
    let mut config = Config::new();
    config
        .set("krishiv.catalog.maxMetadataBytes", &(meta - 1).to_string())
        .unwrap();
    let e = env
        .rt
        .block_on(IcebergTable::load_with_config(
            &env.reg, &location, "db", "t", &config,
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogLimitExceeded,
        "{e}"
    );
    assert!(e.message().contains("metadata file"), "{e}");
    assert!(
        e.message().contains("krishiv.catalog.maxMetadataBytes"),
        "{e}"
    );

    // The manifest list and the manifest, each just over the limit.
    let mut table = env.load(&t);
    let snapshot = table.snapshot(&VersionSpec::Current).unwrap();
    let mut plan_with = |max: u64| {
        table.max_metadata_bytes = max;
        env.rt.block_on(table.plan(
            &snapshot,
            None,
            ManifestFilter::All,
            PruneOptions::default(),
        ))
    };
    let e = plan_with(list_size - 1).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogLimitExceeded,
        "{e}"
    );
    assert!(e.message().contains("manifest list"), "{e}");
    if manifest_size > list_size {
        let e = plan_with(manifest_size - 1).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivCatalogLimitExceeded,
            "{e}"
        );
        assert!(e.message().contains("The Iceberg manifest "), "{e}");
    }
    let ok = plan_with(list_size.max(manifest_size)).unwrap();
    assert_eq!(ok.tasks.len(), 1);
}

/// Review F5 and F9: a format-v1 table answers `VERSION AS OF 'main'` (Java
/// makes `main` from the current snapshot); after an upgrade to v2, the
/// snapshot written under v1 reads its v1 manifest list (its header names the
/// version), and the current one reads the v1 manifest it carries.
#[test]
fn v1_refs_and_manifest_lists_from_before_an_upgrade() {
    let env = env();
    let t = env.create_with(
        TableCreation::builder()
            .name("t".to_owned())
            .schema(ixn_schema())
            .format_version(FormatVersion::V1)
            .build(),
    );
    let t = env.append_rows(&t, "v1");
    let first = t.metadata().current_snapshot().unwrap().snapshot_id();
    let table = env.load(&t);
    let main = table
        .snapshot(&VersionSpec::Ref("main".to_owned()))
        .unwrap();
    assert_eq!(main.snapshot_id, Some(first));
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    assert_eq!(provider_session(p).rows("SELECT name FROM t"), ["v1"]);

    let t = env.rt.block_on(async {
        let tx = Transaction::new(&t);
        let tx = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V2)
            .apply(tx)
            .unwrap();
        tx.commit(&env.catalog).await.unwrap()
    });
    let t = env.append_rows(&t, "v2");
    let table = env.load(&t);
    let old = env.provider(&table, &VersionSpec::Id(first)).unwrap();
    assert_eq!(provider_session(old).rows("SELECT name FROM t"), ["v1"]);
    let current = env.provider(&table, &VersionSpec::Current).unwrap();
    assert_eq!(
        provider_session(current).rows("SELECT name FROM t ORDER BY name"),
        ["v1", "v2"]
    );
}

/// Review F3 through a scan: the table writes `truncate(4)` string bounds and a
/// writer stored the upper bound "abcd" without rounding it up for a file that
/// holds "abcdz". The scan must not take the bound as exact.
#[test]
fn a_scan_reads_bounds_by_the_tables_metrics_mode() {
    let env = env();
    let schema = iceberg::spec::Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let t = env.create_with(
        TableCreation::builder()
            .name("t".to_owned())
            .schema(schema)
            .properties(HashMap::from([(
                "write.metadata.metrics.default".to_owned(),
                "truncate(4)".to_owned(),
            )]))
            .build(),
    );
    let rows = batch(vec![id_field("s", A::Utf8, 1)], vec![strings(&["abcdz"])]);
    let path = env.at("s.parquet");
    let size = write_parquet(&path, &rows);
    let file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(1)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .value_counts(HashMap::from([(1, 1)]))
        .null_value_counts(HashMap::from([(1, 0)]))
        .lower_bounds(HashMap::from([(1, Datum::string("abcd"))]))
        .upper_bounds(HashMap::from([(1, Datum::string("abcd"))]))
        .build()
        .unwrap();
    let t = env.append(&t, vec![file]);
    let table = env.load(&t);
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    let s = provider_session(p);
    assert_eq!(s.rows("SELECT s FROM t WHERE s = 'abcdz'"), ["abcdz"]);
    assert_eq!(s.rows("SELECT s FROM t WHERE s > 'abcd'"), ["abcdz"]);
}

/// Mixed specs (review of PR-25a): `x` is an identity partition in spec 1 and
/// not in spec 0, so it is read from the files; a spec-1 file that lacks it
/// would read NULL where Java reads the manifest, so the table is refused,
/// naming the column and the file. With the column in the file it reads.
#[test]
fn a_mixed_spec_identity_column_a_file_lacks_is_refused() {
    for lacks in [true, false] {
        let env = env();
        let schema = ixn_schema();
        // Spec 0 (the table's) is unpartitioned; spec 1, added by hand, is
        // identity(x).
        let t = env.create("t", schema.clone(), None);
        let spec_x = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_partition_field("x", "x", IceTransform::Identity)
            .unwrap()
            .build()
            .unwrap();
        let base = env.append_rows(&t, "base");
        // The identity(x) file without (or with) `x`, and an unpartitioned one.
        let path0 = env.at("p0.parquet");
        let rows0 = if lacks {
            batch(
                vec![id_field("id", A::Int64, 1), id_field("name", A::Utf8, 3)],
                vec![int64s(&[2]), strings(&["p0"])],
            )
        } else {
            batch(
                vec![
                    id_field("id", A::Int64, 1),
                    id_field("x", A::Int32, 2),
                    id_field("name", A::Utf8, 3),
                ],
                vec![
                    int64s(&[2]),
                    Arc::new(Int32Array::from(vec![7])),
                    strings(&["p0"]),
                ],
            )
        };
        let size0 = write_parquet(&path0, &rows0);
        let file0 = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path0)
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(size0)
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::int(7))]))
            .partition_spec_id(1)
            .build()
            .unwrap();
        let unpartitioned = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .build()
            .unwrap();
        let path1 = env.at("p1.parquet");
        let rows1 = batch(
            vec![
                id_field("id", A::Int64, 1),
                id_field("x", A::Int32, 2),
                id_field("name", A::Utf8, 3),
            ],
            vec![
                int64s(&[3]),
                Arc::new(Int32Array::from(vec![9])),
                strings(&["p1"]),
            ],
        );
        let size1 = write_parquet(&path1, &rows1);
        let file1 = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path1)
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(size1)
            .record_count(1)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // A snapshot by hand: one manifest per spec and a list, added to the
        // metadata with spec 1.
        let previous = Arc::clone(base.metadata().current_snapshot().unwrap());
        let io = env.reg.file_io();
        let snapshot_id = previous.snapshot_id() + 1;
        let seq = previous.sequence_number() + 1;
        let list = env.at("snap-mixed.avro");
        let schema = Arc::new(schema);
        env.rt.block_on(async {
            let mut manifests = Vec::new();
            for (name, spec, file) in [
                ("m0.avro", spec_x, file0),
                ("m1.avro", unpartitioned, file1),
            ] {
                let mut w = ManifestWriterBuilder::new(
                    io.new_output(env.at(name)).unwrap(),
                    Some(snapshot_id),
                    Arc::clone(&schema),
                    spec,
                )
                .build_v2_data();
                w.add_file(file, seq).unwrap();
                manifests.push(w.write_manifest_file().await.unwrap());
            }
            let mut lw = ManifestListWriter::v2(
                io.new_output(&list).unwrap().writer().await.unwrap(),
                snapshot_id,
                Some(previous.snapshot_id()),
                seq,
            );
            lw.add_manifests(manifests.into_iter()).unwrap();
            lw.close().await.unwrap();
        });
        let location = base.metadata_location().unwrap().to_owned();
        let text = env
            .rt
            .block_on(async { io.new_input(&location).unwrap().read().await.unwrap() });
        let mut json: serde_json::Value = serde_json::from_slice(&text).unwrap();
        let ts = previous.timestamp_ms() + 1;
        json["partition-specs"].as_array_mut().unwrap().push(
            serde_json::json!({"spec-id": 1, "fields": [
                {"source-id": 2, "field-id": 1000, "name": "x", "transform": "identity"}
            ]}),
        );
        json["last-partition-id"] = serde_json::json!(1000);
        json["snapshots"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "snapshot-id": snapshot_id,
                "parent-snapshot-id": previous.snapshot_id(),
                "sequence-number": seq,
                "timestamp-ms": ts,
                "manifest-list": list,
                "summary": {"operation": "overwrite"},
                "schema-id": 0
            }));
        json["current-snapshot-id"] = serde_json::json!(snapshot_id);
        json["last-sequence-number"] = serde_json::json!(seq);
        json["refs"]["main"] = serde_json::json!({"snapshot-id": snapshot_id, "type": "branch"});
        json["snapshot-log"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"snapshot-id": snapshot_id, "timestamp-ms": ts}));
        let edited = env.at("99-mixed.metadata.json");
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
        let opened = env.provider(&table, &VersionSpec::Current);
        if lacks {
            let e = opened.unwrap_err();
            assert_eq!(
                e.error_class(),
                ErrorClass::KrishivUnsupportedFeature,
                "{e}"
            );
            assert!(e.message().contains("column x"), "{e}");
            assert!(e.message().contains("p0.parquet"), "{e}");
        } else {
            let p = opened.unwrap();
            assert!(p.constant_columns().is_empty());
            assert_eq!(
                provider_session(p).rows("SELECT name, x FROM t ORDER BY name"),
                ["p0|7", "p1|9"]
            );
        }
    }
}
