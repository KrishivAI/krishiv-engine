// SPDX-License-Identifier: Apache-2.0
//! Tests of v3 row lineage (`lineage`, PR-36b unit 2a): `_row_id` and
//! `_last_updated_sequence_number` read through a lineage scan (inherited,
//! stored, after deletes), written by a rewrite's data-file writer under
//! their reserved ids and read back unchanged from a file committed later
//! with another `first_row_id`; v1/v2 tables refused; the derivation against
//! a model of the spec's rule.

use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType as A, Field};
use iceberg::spec::TableMetadata;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::TableProvider;
use krishiv_exec::{ScanPushdown, ScanPushdowns};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};
use krishiv_types::{DataType, StructField, StructType};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use proptest::prelude::*;

use super::append::{prepare_attempt, write_new_manifests};
use super::append_tests::{MAX, commit, create, file, ids, now};
use super::data_file::{WrittenDataFile, data_file};
use super::delete_tests::{build_in, data, dv, equality, positions};
use super::lineage::{
    LAST_UPDATED, LAST_UPDATED_FIELD_ID, ROW_ID, ROW_ID_FIELD_ID, derive_last_updated,
    derive_row_ids,
};
use super::provider::IcebergTableProvider;
use super::provider_tests::{Env, env, session};
use super::table::IcebergTable;
use super::write::{DataFileWriter, TaskIds};
use crate::plan::VersionSpec;

/// A provider whose scan is `p`'s lineage scan, for SQL over it.
struct Lineage(Arc<dyn ScanPushdown>);

impl std::fmt::Debug for Lineage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Lineage")
    }
}

impl TableProvider for Lineage {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::clone(&self.0))
    }
}

/// A session with `p`'s lineage scan as table `t`.
fn lineage_session(p: &IcebergTableProvider) -> super::provider_tests::Session {
    let scan = p.lineage_scan().unwrap();
    let mut schema: Vec<StructField> = p.snapshot().schema.fields().to_vec();
    schema.push(StructField::new(ROW_ID, DataType::Long, true));
    schema.push(StructField::new(LAST_UPDATED, DataType::Long, true));
    session(vec![(
        "t",
        StructType::new(schema),
        Arc::new(Lineage(scan)),
    )])
}

/// The rows of `sql` over `p`'s lineage scan as table `t`.
fn lineage_rows(p: &IcebergTableProvider, sql: &str) -> Vec<String> {
    lineage_session(p).rows(sql)
}

/// The sum of the scan counter `name` over the operators of `m`.
fn scan_count(m: &krishiv_exec::QueryMetrics, name: &str) -> u64 {
    m.operators
        .values()
        .flat_map(|o| o.extra.iter())
        .filter(|(n, _)| n == name)
        .filter_map(|(_, v)| v.value())
        .sum()
}

fn load(env: &Env, location: &str) -> IcebergTable {
    env.rt
        .block_on(IcebergTable::load(&env.reg, location, "db", "t"))
        .unwrap()
}

/// The provider of the table at `location` reading only the data file at
/// `only` (as if the others were removed), else every file.
fn provider(env: &Env, location: &str, only: Option<&str>) -> IcebergTableProvider {
    let table = load(env, location);
    match only {
        None => env.provider(&table, &VersionSpec::Current).unwrap(),
        Some(path) => env
            .rt
            .block_on(IcebergTableProvider::open_edited(&table, |_, plan| {
                plan.tasks.retain(|t| t.file.file.path == path);
            }))
            .unwrap(),
    }
}

/// Appends `files` to `m` (at `loc`) as snapshot `id`, the `n`th metadata.
fn append(
    env: &Env,
    m: &TableMetadata,
    loc: &str,
    files: Vec<(i32, iceberg::spec::DataFile)>,
    id: i64,
    n: u32,
) -> (TableMetadata, String) {
    let io = env.reg.file_io();
    let attempt = env.rt.block_on(async {
        let mut new =
            write_new_manifests(&io, m, files, id, &format!("uuid-{id}"), &mut Vec::new())
                .await
                .unwrap();
        prepare_attempt(&io, m, &mut new, ids(&format!("w-{id}")), now(m), MAX)
            .await
            .unwrap()
    });
    commit(env, m, loc, &attempt, n)
}

/// A rewrite's writer of `m` (lineage columns on) as attempt `attempt`.
fn writer(env: &Env, m: &TableMetadata, attempt: u64) -> Result<DataFileWriter, KrishivError> {
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let id = pool.register("write", None);
    DataFileWriter::new(
        m,
        &env.reg,
        TaskIds {
            partition_id: 0,
            attempt_id: attempt,
            write_id: "rewrite",
        },
        4,
        Reservation::new(Arc::clone(&pool), id),
    )?
    .with_lineage()
}

/// One row `(id, x, name, _row_id, _last_updated_sequence_number)`.
type Row<'a> = (i64, i32, &'a str, Option<i64>, Option<i64>);

/// Rows of [`Row`]s.
fn rewrite_batch(rows: &[Row<'_>]) -> RecordBatch {
    super::provider_tests::batch(
        vec![
            Field::new("id", A::Int64, true),
            Field::new("x", A::Int32, true),
            Field::new("name", A::Utf8View, true),
            Field::new(ROW_ID, A::Int64, true),
            Field::new(LAST_UPDATED, A::Int64, true),
        ],
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
            Arc::new(StringViewArray::from(
                rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.4).collect::<Vec<_>>(),
            )),
        ],
    )
}

/// Writes `batch` through a lineage writer of `m`; the one file written.
fn write_rewrite(
    env: &Env,
    m: &TableMetadata,
    batch: &RecordBatch,
    attempt: u64,
) -> WrittenDataFile {
    let mut w = writer(env, m, attempt).unwrap();
    let mut files = env.rt.block_on(async {
        w.write(batch).await.unwrap();
        w.finish().await.unwrap()
    });
    assert_eq!(files.len(), 1);
    files.remove(0)
}

const SQL: &str = "SELECT id, _row_id, _last_updated_sequence_number FROM t ORDER BY id";

/// The round trip: rows appended to a v3 table read inherited ids and
/// sequence numbers; written into a new file with both columns and committed
/// later (another `first_row_id` and sequence number), they read the same
/// values from the new file. Then an unmodified, an updated and an inserted
/// row: stored values win, nulls inherit the new file's.
#[test]
fn lineage_survives_a_rewrite() {
    let env = env();
    let t = create(&env, 3);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    let a = file(&env, &m0, "a.parquet", 1, &[(1, "a"), (2, "b"), (3, "c")]);
    let (m1, loc1) = append(&env, &m0, &loc0, vec![a], 401, 1);
    assert_eq!(
        lineage_rows(&provider(&env, &loc1, None), SQL),
        ["1|0|1", "2|1|1", "3|2|1"]
    );
    // Columns in another order, one lineage column alone, and a filter
    // above the scan.
    assert_eq!(
        lineage_rows(
            &provider(&env, &loc1, None),
            "SELECT _last_updated_sequence_number, name FROM t WHERE id > 1 ORDER BY name"
        ),
        ["1|b", "1|c"]
    );

    // A copy-on-write rewrite of every row, none modified.
    let moved = write_rewrite(
        &env,
        &m1,
        &rewrite_batch(&[
            (1, 1, "a", Some(0), Some(1)),
            (2, 1, "b", Some(1), Some(1)),
            (3, 1, "c", Some(2), Some(1)),
        ]),
        1,
    );
    let moved_path = moved.location.clone();
    let df = data_file(&moved, m1.current_schema()).unwrap();
    let (m2, loc2) = append(&env, &m1, &loc1, vec![(moved.spec_id, df)], 402, 2);
    let s2 = m2.current_snapshot().unwrap();
    assert_eq!((s2.first_row_id(), s2.sequence_number()), (Some(3), 2));
    assert_eq!(
        lineage_rows(&provider(&env, &loc2, Some(&moved_path)), SQL),
        ["1|0|1", "2|1|1", "3|2|1"]
    );

    // Mixed: row 1 unmodified, row 2 updated (sequence number null), row 9
    // inserted (both null). The file's first_row_id is 6, sequence number 3.
    let mixed = write_rewrite(
        &env,
        &m2,
        &rewrite_batch(&[
            (1, 1, "a", Some(0), Some(1)),
            (2, 1, "b2", Some(1), None),
            (9, 1, "new", None, None),
        ]),
        2,
    );
    let mixed_path = mixed.location.clone();
    let df = data_file(&mixed, m2.current_schema()).unwrap();
    let (m3, loc3) = append(&env, &m2, &loc2, vec![(mixed.spec_id, df)], 403, 3);
    let s3 = m3.current_snapshot().unwrap();
    assert_eq!((s3.first_row_id(), s3.sequence_number()), (Some(6), 3));
    assert_eq!(
        lineage_rows(&provider(&env, &loc3, Some(&mixed_path)), SQL),
        ["1|0|1", "2|1|3", "9|8|3"]
    );
    // The plain scan does not see the columns; the table reads all rows.
    let p = provider(&env, &loc3, None);
    assert!(
        p.scan()
            .unwrap()
            .table_schema()
            .field_with_name(ROW_ID)
            .is_err()
    );
    assert_eq!(lineage_rows(&p, "SELECT count(*) FROM t"), ["9"]);
}

/// The written file carries the two columns under their reserved ids,
/// optional longs, and their metrics follow the table's default mode (Java's
/// `MetricsConfig.columnMode(id)` gives the default to an id outside the
/// table schema).
#[test]
fn a_rewrite_writes_the_reserved_ids_and_their_metrics() {
    let env = env();
    let t = create(&env, 3);
    let m = t.metadata().clone();
    let w = write_rewrite(
        &env,
        &m,
        &rewrite_batch(&[(1, 1, "a", Some(7), Some(1)), (2, 1, "b", None, None)]),
        1,
    );
    let local = super::sink_tests::local_path(&w.location);
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(local).unwrap()).unwrap();
    let ids: Vec<(String, String, bool)> = reader
        .schema()
        .fields()
        .iter()
        .map(|f| {
            (
                f.name().clone(),
                f.metadata()["PARQUET:field_id"].clone(),
                f.is_nullable(),
            )
        })
        .collect();
    assert_eq!(
        ids[3..],
        [
            (ROW_ID.to_owned(), ROW_ID_FIELD_ID.to_string(), true),
            (
                LAST_UPDATED.to_owned(),
                LAST_UPDATED_FIELD_ID.to_string(),
                true
            ),
        ]
    );
    let long = |v: i64| {
        crate::plan::bounds::encode(
            &krishiv_io::ScalarValue::Int64(v),
            &crate::plan::IcebergType::Long,
        )
        .unwrap()
    };
    let mm = &w.metrics;
    assert_eq!(mm.value_counts.get(&ROW_ID_FIELD_ID), Some(&2));
    assert_eq!(mm.null_counts.get(&ROW_ID_FIELD_ID), Some(&1));
    assert_eq!(mm.lower_bounds.get(&ROW_ID_FIELD_ID), Some(&long(7)));
    assert_eq!(mm.upper_bounds.get(&LAST_UPDATED_FIELD_ID), Some(&long(1)));

    // Under `write.metadata.metrics.default=counts` no bounds are kept.
    let m_counts = {
        let mut b = iceberg::spec::TableMetadataBuilder::new_from_metadata(m, None);
        b = b
            .set_properties(
                [(
                    "write.metadata.metrics.default".to_owned(),
                    "counts".to_owned(),
                )]
                .into(),
            )
            .unwrap();
        b.build().unwrap().metadata
    };
    let w = write_rewrite(
        &env,
        &m_counts,
        &rewrite_batch(&[(1, 1, "a", Some(7), Some(1))]),
        2,
    );
    assert_eq!(w.metrics.value_counts.get(&ROW_ID_FIELD_ID), Some(&1));
    assert!(!w.metrics.lower_bounds.contains_key(&ROW_ID_FIELD_ID));
}

/// A rewrite that does not carry the columns writes none of them (an
/// insert-only write), and a lineage writer refuses a batch without them.
#[test]
fn without_lineage_the_columns_are_not_written() {
    let env = env();
    let t = create(&env, 3);
    let m = t.metadata().clone();
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let id = pool.register("write", None);
    let mut plain = DataFileWriter::new(
        &m,
        &env.reg,
        TaskIds {
            partition_id: 0,
            attempt_id: 1,
            write_id: "w",
        },
        4,
        Reservation::new(Arc::clone(&pool), id),
    )
    .unwrap();
    let rows = super::provider_tests::batch(
        vec![
            Field::new("id", A::Int64, true),
            Field::new("x", A::Int32, true),
            Field::new("name", A::Utf8View, true),
        ],
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringViewArray::from(vec!["a"])),
        ],
    );
    let files = env.rt.block_on(async {
        plain.write(&rows).await.unwrap();
        plain.finish().await.unwrap()
    });
    assert!(!files[0].metrics.value_counts.contains_key(&ROW_ID_FIELD_ID));

    let mut w = writer(&env, &m, 2).unwrap();
    let e = env.rt.block_on(w.write(&rows)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    env.rt.block_on(w.abort());
}

/// Deletes apply before the values are observed: a DV and a position delete
/// file remove rows, an equality delete another; the survivors keep
/// `first_row_id + _pos` of their place in the file.
#[test]
fn deleted_rows_never_show_and_survivors_keep_their_ids() {
    let t = build_in(
        env(),
        &[
            data("a.parquet", 0, 1, 0, 6),
            data("b.parquet", 0, 1, 10, 4),
            dv("a.puffin", 0, 2, "a.parquet", &[1, 3]),
            positions("d.parquet", 0, 2, &[("b.parquet", 0)], None),
            equality("e.parquet", 0, 2, &[Some(12)]),
        ],
        false,
    );
    let p = t
        .env
        .rt
        .block_on(IcebergTableProvider::open_edited(&t.table, |snap, plan| {
            snap.format_version = 3;
            for task in &mut plan.tasks {
                let e = Arc::make_mut(&mut task.file);
                e.file.first_row_id = Some(if e.file.path.ends_with("a.parquet") {
                    100
                } else {
                    200
                });
            }
        }))
        .unwrap();
    assert_eq!(
        lineage_rows(&p, SQL),
        [
            "0|100|1", "2|102|1", "4|104|1", "5|105|1", "11|201|1", "13|203|1"
        ]
    );
}

/// A file without a `first_row_id` (from before the upgrade to v3) reads
/// null for both columns, as the spec says.
#[test]
fn a_file_without_first_row_id_reads_nulls() {
    let t = build_in(env(), &[data("a.parquet", 0, 4, 0, 2)], false);
    let p = t
        .env
        .rt
        .block_on(IcebergTableProvider::open_edited(&t.table, |snap, plan| {
            snap.format_version = 3;
            for task in &mut plan.tasks {
                Arc::make_mut(&mut task.file).file.first_row_id = None;
            }
        }))
        .unwrap();
    assert_eq!(lineage_rows(&p, SQL), ["0|NULL|NULL", "1|NULL|NULL"]);
}

/// v1 and v2 tables have no row lineage (the spec's "In v3 and later"):
/// the lineage scan and a lineage writer are refused, never invented.
#[test]
fn v1_and_v2_tables_refuse_the_columns() {
    for version in [1, 2] {
        let env = env();
        let t = create(&env, version);
        let m = t.metadata().clone();
        let loc = t.metadata_location().unwrap().to_owned();
        let p = provider(&env, &loc, None);
        let e = p.lineage_scan().err().unwrap();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        );
        assert!(e.to_string().contains("format version"), "{e}");
        let e = writer(&env, &m, 1).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        );
    }
}

/// A lineage scan with `_metadata` columns is refused (the row position is
/// read as the row id); its output schema matches its batches.
#[test]
fn lineage_scan_output_schema_and_metadata_refusal() {
    let env = env();
    let t = create(&env, 3);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    let a = file(&env, &m0, "a.parquet", 1, &[(1, "a")]);
    let (_, loc1) = append(&env, &m0, &loc0, vec![a], 401, 1);
    let scan = provider(&env, &loc1, None).lineage_scan().unwrap();
    let names: Vec<String> = scan
        .table_schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["id", "x", "name", ROW_ID, LAST_UPDATED]);
    let mut p = ScanPushdowns {
        projection: Some(vec![
            krishiv_io::ColumnPath::new(ROW_ID),
            krishiv_io::ColumnPath::new("id"),
        ]),
        row_id: true,
        ..ScanPushdowns::default()
    };
    let out = scan.output_schema(&p).unwrap();
    let out: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        out,
        [
            ROW_ID,
            "id",
            krishiv_exec::ROW_FILE_COLUMN,
            krishiv_exec::ROW_POSITION_COLUMN
        ]
    );
    assert_eq!(scan.build(p.clone()).unwrap().schema().fields().len(), 4);
    // A row id with `_metadata` and no lineage column: unsupported, not
    // the file scan's internal error.
    let mut q = p.clone();
    q.projection = Some(vec![krishiv_io::ColumnPath::new("id")]);
    q.metadata.file_path = true;
    let e = scan.build(q).err().unwrap();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    p.row_id = false;
    p.metadata.file_path = true;
    let e = scan.build(p).err().unwrap();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
}

/// A data file of `n` rows, ids `0..n` at their positions, x = 1, in row
/// groups of `group` rows; a `DataFile` of partition x = 1.
fn grouped_file(
    env: &Env,
    m: &TableMetadata,
    name: &str,
    n: i64,
    group: usize,
) -> (i32, iceberg::spec::DataFile) {
    let ids: Vec<i64> = (0..n).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let b = super::provider_tests::batch(
        vec![
            super::provider_tests::id_field("id", A::Int64, 1),
            super::provider_tests::id_field("x", A::Int32, 2),
            super::provider_tests::id_field("name", A::Utf8, 3),
        ],
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(Int32Array::from(vec![1; ids.len()])),
            Arc::new(arrow_array::StringArray::from(names)),
        ],
    );
    let location = env.at(name);
    let local = super::sink_tests::local_path(&location);
    let props = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_row_count(Some(group))
        .build();
    let mut w = parquet::arrow::ArrowWriter::try_new(
        std::fs::File::create(&local).unwrap(),
        b.schema(),
        Some(props),
    )
    .unwrap();
    w.write(&b).unwrap();
    let meta = w.close().unwrap();
    assert!(
        meta.row_groups().len() > 2,
        "{} row groups",
        meta.row_groups().len()
    );
    let size = std::fs::metadata(local).unwrap().len();
    let spec = m.default_partition_spec();
    let partition = super::data_file::partition_tuple(
        spec,
        m.current_schema(),
        &[Some(krishiv_io::ScalarValue::Int32(1))],
    )
    .unwrap();
    let df = data_file(
        &WrittenDataFile {
            location,
            size,
            rows: u64::try_from(n).unwrap(),
            metrics: crate::plan::file_metrics::FileMetrics::default(),
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

/// Review finding 1: a file of several row groups read in several splits,
/// under a filter that skips its first row group: every row's `_row_id` is
/// the file's `first_row_id` plus its physical position, in later row
/// groups too.
#[test]
fn row_ids_follow_physical_positions_across_row_groups_and_splits() {
    let env = env();
    let t = create(&env, 3);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    let a = file(
        &env,
        &m0,
        "a.parquet",
        1,
        &[(100, "a"), (101, "b"), (102, "c")],
    );
    let (m1, loc1) = append(&env, &m0, &loc0, vec![a], 501, 1);
    let big = grouped_file(&env, &m1, "big.parquet", 60, 10);
    let (m2, loc2) = append(&env, &m1, &loc1, vec![big], 502, 2);
    let s2 = m2.current_snapshot().unwrap();
    assert_eq!((s2.first_row_id(), s2.sequence_number()), (Some(3), 2));

    let mut config = krishiv_common::config::Config::new();
    config
        .set("spark.sql.files.maxPartitionBytes", "300")
        .unwrap();
    config.set("spark.sql.files.openCostInBytes", "1").unwrap();
    let table = load(&env, &loc2);
    let p = env
        .rt
        .block_on(IcebergTableProvider::open(
            &table,
            &VersionSpec::Current,
            &config,
            4,
        ))
        .unwrap();
    assert!(p.lineage_splits() > 2, "{} splits", p.lineage_splits());
    let want: Vec<String> = (15..60).map(|i| format!("{i}|{}|2", 3 + i)).collect();
    let (batches, metrics) = lineage_session(&p)
        .run_metrics(
            "SELECT id, _row_id, _last_updated_sequence_number FROM t \
             WHERE id >= 15 AND id < 100 ORDER BY id",
        )
        .unwrap();
    assert_eq!(super::provider_tests::render(&batches), want);
    // The pushed filter skipped row groups: fewer rows decoded than the big
    // file's 60 (its first group, 0..10, is never read), and the files were
    // read as several splits.
    assert!(
        scan_count(&metrics, "scan_rows_decoded") < 60,
        "{metrics:?}"
    );
    assert!(scan_count(&metrics, "scan_splits") > 2, "{metrics:?}");
    let all = lineage_rows(&p, "SELECT id, _row_id FROM t WHERE id < 100 ORDER BY id");
    assert_eq!(
        all,
        (0..60)
            .map(|i| format!("{i}|{}", 3 + i))
            .collect::<Vec<_>>()
    );
}

/// `m` upgraded to format v3, written as the `n`th metadata; its location.
fn upgrade(env: &Env, m: &TableMetadata, loc: &str, n: u32) -> (TableMetadata, String) {
    let next =
        iceberg::spec::TableMetadataBuilder::new_from_metadata(m.clone(), Some(loc.to_owned()))
            .upgrade_format_version(iceberg::spec::FormatVersion::V3)
            .unwrap()
            .build()
            .unwrap()
            .metadata;
    let path = env.at(&format!("v{n}.metadata.json"));
    std::fs::write(
        super::sink_tests::local_path(&path),
        serde_json::to_vec(&next).unwrap(),
    )
    .unwrap();
    (next, path)
}

/// Review finding 3: a real v2 → v3 upgrade. Files from the v2 snapshots
/// have no `first_row_id` and read null for both columns until the first v3
/// snapshot, which assigns row ids to every data manifest, the existing ones
/// included (spec "Row Lineage for Upgraded Tables": `next-row-id` starts at
/// 0, and "When writing the manifest list, all data manifests must be
/// assigned a `first_row_id`"); the old rows keep their data sequence number.
#[test]
fn an_upgraded_table_assigns_row_ids_at_its_first_v3_snapshot() {
    let env = env();
    let t = create(&env, 2);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    let a = file(&env, &m0, "a.parquet", 1, &[(1, "a"), (2, "b")]);
    let (m1, loc1) = append(&env, &m0, &loc0, vec![a], 601, 1);
    let e = provider(&env, &loc1, None).lineage_scan().err().unwrap();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );

    let (m2, loc2) = upgrade(&env, &m1, &loc1, 2);
    assert_eq!((m2.format_version() as u8, m2.next_row_id()), (3, 0));
    assert_eq!(
        lineage_rows(&provider(&env, &loc2, None), SQL),
        ["1|NULL|NULL", "2|NULL|NULL"]
    );

    let b = file(&env, &m2, "b.parquet", 2, &[(3, "c")]);
    let (m3, loc3) = append(&env, &m2, &loc2, vec![b], 602, 3);
    let s3 = m3.current_snapshot().unwrap();
    assert_eq!((s3.first_row_id(), s3.sequence_number()), (Some(0), 2));
    // The new manifest is listed first, then the base's: b gets 0, a 1..2.
    assert_eq!(s3.added_rows_count(), Some(3));
    assert_eq!(m3.next_row_id(), 3);
    assert_eq!(
        lineage_rows(&provider(&env, &loc3, None), SQL),
        ["1|1|1", "2|2|1", "3|0|2"]
    );
}

/// The spec's rule, row by row.
fn model(
    stored_id: Option<i64>,
    stored_seq: Option<i64>,
    first: Option<i64>,
    pos: i64,
    seq: i64,
) -> (Option<i64>, Option<i64>) {
    match first {
        None => (None, None),
        Some(f) => (
            Some(stored_id.unwrap_or(f + pos)),
            Some(stored_seq.unwrap_or(seq)),
        ),
    }
}

proptest! {
    /// Random stored/null patterns against [`model`].
    #[test]
    fn derivation_matches_the_spec(
        rows in proptest::collection::vec(
            (proptest::option::of(0_i64..1_000), proptest::option::of(0_i64..50)),
            0..40,
        ),
        first in proptest::option::of(0_i64..1_000_000),
        seq in 1_i64..100,
    ) {
        let n = rows.len();
        let stored_id: Int64Array = rows.iter().map(|r| r.0).collect();
        let stored_seq: Int64Array = rows.iter().map(|r| r.1).collect();
        let firsts: Int64Array = vec![first; n].into_iter().collect();
        let seqs = Int64Array::from(vec![seq; n]);
        let pos = Int64Array::from((0..n as i64).collect::<Vec<_>>());
        let ids = derive_row_ids(&stored_id, &firsts, &pos).unwrap();
        let lus = derive_last_updated(&stored_seq, &firsts, &seqs);
        for (i, r) in rows.iter().enumerate() {
            let want = model(r.0, r.1, first, i as i64, seq);
            let got = (
                ids.is_valid(i).then(|| ids.value(i)),
                lus.is_valid(i).then(|| lus.value(i)),
            );
            prop_assert_eq!(got, want, "row {}", i);
        }
    }
}
