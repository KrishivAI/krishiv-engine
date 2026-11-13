// SPDX-License-Identifier: Apache-2.0
//! Tests of an append's metadata (`append`, PR-36a): manifests and a manifest
//! list Krishiv wrote, committed the way a catalog applies a request (its
//! requirements checked, its updates applied to the metadata), then read back
//! by Krishiv's provider through SQL and by iceberg-rust's own scan planning.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::DataType as A;
use iceberg::spec::{
    FormatVersion, Operation, PartitionSpec, TableMetadata, TableMetadataBuilder, Transform,
};
use iceberg::{NamespaceIdent, TableCreation, TableIdent};

use super::append::{
    Attempt, NewManifests, QUERY_ID_PROP, WRITE_ID_PROP, WriteIds, prepare_attempt,
    write_new_manifests,
};
use super::data_file::{WrittenDataFile, data_file, partition_tuple};
use super::provider_tests::{
    Env, batch, env, id_field, int64s, ixn_schema, provider_session, render, strings, write_parquet,
};
use super::table::IcebergTable;
use crate::plan::VersionSpec;
use crate::plan::file_metrics::FileMetrics;

const MAX: u64 = 1 << 30;

/// A table `t` partitioned by `identity(x)`, of format `version`.
fn create(env: &Env, version: u8) -> iceberg::table::Table {
    let schema = ixn_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let creation = TableCreation::builder()
        .name("t".to_owned())
        .schema(schema)
        .partition_spec(spec.into_unbound())
        .format_version(match version {
            3 => FormatVersion::V3,
            _ => FormatVersion::V2,
        })
        .build();
    env.rt.block_on(async {
        use iceberg::Catalog;
        env.catalog
            .create_table(&NamespaceIdent::new("db".to_owned()), creation)
            .await
            .unwrap()
    })
}

/// A Parquet file of `rows` (id, name) in partition `x`, as a `DataFile`.
fn file(
    env: &Env,
    m: &TableMetadata,
    name: &str,
    x: i32,
    rows: &[(i64, &str)],
) -> (i32, iceberg::spec::DataFile) {
    let ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let names: Vec<&str> = rows.iter().map(|r| r.1).collect();
    let b: RecordBatch = batch(
        vec![
            id_field("id", A::Int64, 1),
            id_field("x", A::Int32, 2),
            id_field("name", A::Utf8, 3),
        ],
        vec![
            int64s(&ids),
            Arc::new(Int32Array::from(vec![x; rows.len()])),
            strings(&names),
        ],
    );
    let location = env.at(name);
    let size = write_parquet(&location, &b);
    let spec = m.default_partition_spec();
    let partition = partition_tuple(
        spec,
        m.current_schema(),
        &[Some(krishiv_io::ScalarValue::Int32(x))],
    )
    .unwrap();
    let df = data_file(
        &WrittenDataFile {
            location,
            size,
            rows: rows.len() as u64,
            metrics: FileMetrics::default(),
            partition,
            spec_id: spec.spec_id(),
            split_offsets: None,
            sort_order_id: None,
        },
        m.current_schema(),
    )
    .unwrap();
    (spec.spec_id(), df)
}

/// A catalog's check of the request's requirements against `m`.
fn check(m: &TableMetadata, a: &Attempt) -> Result<(), iceberg::Error> {
    a.requirements.iter().try_for_each(|r| r.check(Some(m)))
}

/// What a catalog does with the request: check the requirements against
/// `m`, apply the updates, write the new metadata file; its location.
fn commit(
    env: &Env,
    m: &TableMetadata,
    location: &str,
    a: &Attempt,
    n: u32,
) -> (TableMetadata, String) {
    check(m, a).unwrap();
    let mut b = TableMetadataBuilder::new_from_metadata(m.clone(), Some(location.to_owned()));
    for u in a.updates.clone() {
        b = u.apply(b).unwrap();
    }
    let next = b.build().unwrap().metadata;
    let path = env.at(&format!("v{n}.metadata.json"));
    std::fs::write(
        path.strip_prefix("file://").unwrap(),
        serde_json::to_vec(&next).unwrap(),
    )
    .unwrap();
    (next, path)
}

fn ids(write_id: &str) -> WriteIds<'_> {
    WriteIds {
        branch: "main",
        write_id,
        query_id: Some("q-1"),
    }
}

fn append(
    env: &Env,
    m: &TableMetadata,
    files: Vec<(i32, iceberg::spec::DataFile)>,
    snapshot_id: i64,
    write_id: &str,
) -> (NewManifests, Attempt) {
    let io = env.reg.file_io();
    env.rt.block_on(async {
        let new = write_new_manifests(&io, m, files, snapshot_id, &format!("uuid-{snapshot_id}"))
            .await
            .unwrap();
        let a = prepare_attempt(&io, m, &new, ids(write_id), 1, now(m), MAX)
            .await
            .unwrap();
        (new, a)
    })
}

/// A timestamp after the table's last update.
fn now(m: &TableMetadata) -> i64 {
    m.last_updated_ms() + 1
}

fn rows(env: &Env, location: &str) -> Vec<String> {
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, location, "db", "t"))
        .unwrap();
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    render(
        &provider_session(p)
            .run("SELECT id, x, name FROM t ORDER BY id")
            .unwrap(),
    )
}

/// iceberg-rust's own planning of the table at `location`: its data files.
fn iceberg_rust_files(env: &Env, m: &TableMetadata, location: &str) -> Vec<String> {
    let table = iceberg::table::Table::builder()
        .metadata(m.clone())
        .metadata_location(location)
        .identifier(TableIdent::from_strs(["db", "t"]).unwrap())
        .file_io(env.reg.file_io())
        .runtime(env.reg.runtime().unwrap())
        .build()
        .unwrap();
    let mut files: Vec<String> = env.rt.block_on(async {
        use futures::TryStreamExt;
        table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .map_ok(|t| t.data_file_path().to_owned())
            .try_collect()
            .await
            .unwrap()
    });
    files.sort();
    files
}

/// Two appends, each committed as a catalog applies the request: the rows of
/// both read back through SQL, iceberg-rust plans the same files, and the
/// snapshots carry the parent, sequence numbers and summary Java writes.
#[test]
fn appends_commit_and_read_back() {
    let env = env();
    let t = create(&env, 2);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    assert_eq!(m0.current_snapshot_id(), None);

    let a = file(&env, &m0, "a.parquet", 1, &[(1, "a1"), (2, "a2")]);
    let b = file(&env, &m0, "b.parquet", 2, &[(3, "b")]);
    let (_, first) = append(&env, &m0, vec![a, b], 101, "w-1");
    let (m1, loc1) = commit(&env, &m0, &loc0, &first, 1);
    assert_eq!(rows(&env, &loc1), ["1|1|a1", "2|1|a2", "3|2|b"]);

    let s1 = m1.current_snapshot().unwrap();
    assert_eq!(
        (
            s1.snapshot_id(),
            s1.parent_snapshot_id(),
            s1.sequence_number()
        ),
        (101, None, 1)
    );
    let p = &s1.summary().additional_properties;
    assert_eq!(s1.summary().operation, Operation::Append);
    for (k, v) in [
        ("added-data-files", "2"),
        ("added-records", "3"),
        ("total-data-files", "2"),
        ("total-records", "3"),
        ("total-delete-files", "0"),
        ("changed-partition-count", "2"),
        ("engine-name", "krishiv"),
        (WRITE_ID_PROP, "w-1"),
        (QUERY_ID_PROP, "q-1"),
    ] {
        assert_eq!(p.get(k).map(String::as_str), Some(v), "{k}: {p:?}");
    }

    let c = file(&env, &m1, "c.parquet", 1, &[(4, "c")]);
    let (_, second) = append(&env, &m1, vec![c], 102, "w-2");
    let (m2, loc2) = commit(&env, &m1, &loc1, &second, 2);
    assert_eq!(rows(&env, &loc2), ["1|1|a1", "2|1|a2", "3|2|b", "4|1|c"]);
    let s2 = m2.current_snapshot().unwrap();
    assert_eq!(
        (
            s2.snapshot_id(),
            s2.parent_snapshot_id(),
            s2.sequence_number()
        ),
        (102, Some(101), 2)
    );
    let p = &s2.summary().additional_properties;
    assert_eq!(p["total-records"], "4");
    assert_eq!(p["total-data-files"], "3");
    assert_eq!(
        iceberg_rust_files(&env, &m2, &loc2),
        ["a.parquet", "b.parquet", "c.parquet"].map(|n| env.at(n))
    );
    // Time travel to the first append still sees its rows only.
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, &loc2, "db", "t"))
        .unwrap();
    let old = env.provider(&table, &VersionSpec::Id(101)).unwrap();
    assert_eq!(
        render(&provider_session(old).run("SELECT count(*) FROM t").unwrap()),
        ["3"]
    );
}

/// An attempt built on a stale base fails the catalog's
/// `assert-ref-snapshot-id`; rebuilt on the refreshed metadata with the same
/// manifests and snapshot id, it commits on top of the other writer's
/// snapshot, and the new manifests take the new sequence number.
#[test]
fn a_stale_attempt_is_refused_and_rebuilt_on_the_new_base() {
    let env = env();
    let t = create(&env, 2);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    let a = file(&env, &m0, "a.parquet", 1, &[(1, "a")]);
    let b = file(&env, &m0, "b.parquet", 1, &[(2, "b")]);
    let (new_b, stale) = append(&env, &m0, vec![b], 202, "w-b");
    let (_, other) = append(&env, &m0, vec![a], 201, "w-a");
    let (m1, loc1) = commit(&env, &m0, &loc0, &other, 1);

    let e = check(&m1, &stale).unwrap_err();
    assert_eq!(e.kind(), iceberg::ErrorKind::CatalogCommitConflicts, "{e}");

    let io = env.reg.file_io();
    let retry = env
        .rt
        .block_on(prepare_attempt(
            &io,
            &m1,
            &new_b,
            ids("w-b"),
            2,
            now(&m1),
            MAX,
        ))
        .unwrap();
    assert!(
        retry
            .snapshot
            .manifest_list()
            .contains("snap-202-2-uuid-202")
    );
    let (m2, loc2) = commit(&env, &m1, &loc1, &retry, 2);
    let s = m2.current_snapshot().unwrap();
    assert_eq!(
        (s.snapshot_id(), s.parent_snapshot_id(), s.sequence_number()),
        (202, Some(201), 2)
    );
    assert_eq!(rows(&env, &loc2), ["1|1|a", "2|1|b"]);
    assert_eq!(s.summary().additional_properties["total-records"], "2");
}

/// v3: the snapshot's `first-row-id` is the table's `next-row-id` and its
/// `added-rows` the rows the list writer assigned, so the table's
/// `next-row-id` advances by the rows written (iceberg-rust refuses a v3
/// snapshot without them).
#[test]
fn v3_appends_assign_row_lineage() {
    let env = env();
    let t = create(&env, 3);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    assert_eq!(m0.next_row_id(), 0);
    let a = file(&env, &m0, "a.parquet", 1, &[(1, "a1"), (2, "a2")]);
    let (_, first) = append(&env, &m0, vec![a], 301, "w-1");
    let (m1, loc1) = commit(&env, &m0, &loc0, &first, 1);
    let s1 = m1.current_snapshot().unwrap();
    assert_eq!(s1.first_row_id(), Some(0));
    assert_eq!(s1.added_rows_count(), Some(2));
    assert_eq!(m1.next_row_id(), 2);

    let b = file(&env, &m1, "b.parquet", 2, &[(3, "b")]);
    let (_, second) = append(&env, &m1, vec![b], 302, "w-2");
    let (m2, loc2) = commit(&env, &m1, &loc1, &second, 2);
    let s2 = m2.current_snapshot().unwrap();
    // The base manifest already has its row ids; only the new one is assigned.
    assert_eq!(
        (s2.first_row_id(), s2.added_rows_count()),
        (Some(2), Some(1))
    );
    assert_eq!(m2.next_row_id(), 3);
    assert_eq!(rows(&env, &loc2), ["1|1|a1", "2|1|a2", "3|2|b"]);
}
