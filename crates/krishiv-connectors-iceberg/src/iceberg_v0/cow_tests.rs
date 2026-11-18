// SPDX-License-Identifier: Apache-2.0
//! Tests of copy-on-write `DELETE`, `UPDATE` and `MERGE` (`row_level`,
//! `filter`, PR-36b part 2a) through the row-level table an Iceberg provider
//! offers, driven as the session's row-level planner drives it (the session
//! crate does not reach Iceberg tables here): the find query and the rewrite
//! query in SQL over [`RowLevelTable::scan`] and [`RowLevelTable::scan_files`],
//! the rewrite's rows into [`RowLevelTable::rewriter`], committed by
//! `sink_tests`' catalog; read back through SQL and iceberg-rust's planning,
//! with the table's invariants checked after each case (every live file
//! listed once, no DV of a file that is not live).

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType as A, Schema, SchemaRef};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, Manifest, ManifestContentType,
    ManifestListWriter, ManifestStatus, Operation, PartitionSpec, Snapshot, SnapshotReference,
    SnapshotRetention, Struct, Summary, TableMetadata, Transform, UNASSIGNED_SEQUENCE_NUMBER,
};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::{RowLevelOperation, RowLevelTable, RowLevelWrite, TableProvider};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdown};
use krishiv_types::batch::Batch;
use krishiv_types::{DataType, StructField, StructType};

use super::append_tests::{MAX, iceberg_rust_files};
use super::commit::CommitTable;
use super::lineage::{LAST_UPDATED, ROW_ID, lineage_fields};
use super::merge::manifest_writer;
use super::provider_tests::{Env, env, id_field, ixn_schema, render, session, write_parquet};
use super::scan::read_manifest_files;
use super::sink_tests::{
    Catalog, PerDriver, expected, input_schema, parquet_files, provider, range, read_back, run,
};
use crate::plan::{DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID, encode_dv};

/// A provider whose scan is `0`.
struct Over(Arc<dyn ScanPushdown>);

impl std::fmt::Debug for Over {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Over")
    }
}

impl TableProvider for Over {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::clone(&self.0))
    }
}

/// A table `t` of [`ixn_schema`] at format `version`, partitioned by
/// `identity(x)` when `partitioned`, with `props`.
fn create(
    env: &Env,
    version: u8,
    partitioned: bool,
    props: &[(&str, &str)],
) -> iceberg::table::Table {
    let schema = ixn_schema();
    let props: HashMap<String, String> = props
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let b = iceberg::TableCreation::builder()
        .name("t".to_owned())
        .properties(props)
        .format_version(if version == 3 {
            FormatVersion::V3
        } else {
            FormatVersion::V2
        });
    let creation = if partitioned {
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        b.schema(schema).partition_spec(spec.into_unbound()).build()
    } else {
        b.schema(schema).build()
    };
    env.rt.block_on(async {
        use iceberg::Catalog as _;
        env.catalog
            .create_table(&iceberg::NamespaceIdent::new("db".to_owned()), creation)
            .await
            .unwrap()
    })
}

/// Appends `batch` through Krishiv's append sink.
fn append(env: &Env, catalog: &Arc<Catalog>, batch: RecordBatch) {
    let p = provider(env, &catalog.location(), catalog);
    let mut r = krishiv_exec::planner::WriteRequest::new(
        krishiv_exec::planner::WriteMode::Append,
        "`db`.`t`",
        input_schema(),
    );
    r.write_id = None;
    let sink = p.writer(&r).unwrap();
    let input = krishiv_exec::testing::Batches::new(input_schema(), vec![Batch::new(batch)]);
    run(env, sink, input, None).0.unwrap();
}

/// One row-level statement against the table as the provider pinned it.
struct Dml<'a> {
    env: &'a Env,
    table: Arc<dyn RowLevelTable>,
    op: RowLevelOperation,
    lineage: bool,
}

impl<'a> Dml<'a> {
    fn open(env: &'a Env, catalog: &Arc<Catalog>, op: RowLevelOperation) -> Self {
        let p = provider(env, &catalog.location(), catalog);
        let lineage = p.snapshot().format_version >= 3;
        Self {
            env,
            table: p.row_level(op).expect("a row-level table"),
            op,
            lineage,
        }
    }

    /// The scan's columns: the table's, the lineage columns on v3, the row id.
    fn columns(&self) -> StructType {
        let mut fields = vec![
            StructField::new("id", DataType::Long, true),
            StructField::new("x", DataType::Integer, true),
            StructField::new("name", DataType::String, true),
        ];
        if self.lineage {
            fields.push(StructField::new(ROW_ID, DataType::Long, true));
            fields.push(StructField::new(LAST_UPDATED, DataType::Long, true));
        }
        fields.push(StructField::new(ROW_FILE_COLUMN, DataType::Binary, false));
        fields.push(StructField::new(ROW_POSITION_COLUMN, DataType::Long, false));
        StructType::new(fields)
    }

    fn query(&self, scan: Arc<dyn ScanPushdown>, sql: &str) -> Vec<RecordBatch> {
        session(vec![("t", self.columns(), Arc::new(Over(scan)))])
            .run(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    /// The touched files of `condition` (the session's find query).
    fn find(&self, condition: &str) -> Vec<Vec<u8>> {
        let batches = self.query(
            self.table.scan().unwrap(),
            &format!(
                "SELECT {ROW_FILE_COLUMN}, count(*) FROM t WHERE coalesce(({condition}), false) \
                 GROUP BY 1"
            ),
        );
        let mut out = Vec::new();
        for b in &batches {
            let c = b.column(0);
            for i in 0..c.len() {
                out.push(match c.data_type() {
                    A::BinaryView => c.as_binary_view().value(i).to_vec(),
                    _ => c.as_binary::<i32>().value(i).to_vec(),
                });
            }
        }
        out
    }

    /// The schema the rewriter takes.
    fn input(&self) -> SchemaRef {
        let mut fields: Vec<_> = input_schema().fields().iter().cloned().collect();
        if self.lineage {
            fields.extend(lineage_fields());
        }
        Arc::new(Schema::new(fields))
    }

    /// The rows `select` gives over the touched files, plus `extra`, into
    /// the rewriter.
    #[expect(
        clippy::unwrap_in_result,
        reason = "a test helper: a setup failure panics, the rewrite's own error is returned"
    )]
    fn rewrite(
        &self,
        touched: Vec<Vec<u8>>,
        select: &str,
        extra: Vec<RecordBatch>,
    ) -> Result<(), KrishivError> {
        let rows = self.query(self.table.scan_files(&touched).unwrap(), select);
        let schema = self.input();
        let mut batches: Vec<RecordBatch> = rows
            .into_iter()
            .chain(extra)
            .map(|b| RecordBatch::try_new(Arc::clone(&schema), b.columns().to_vec()).unwrap())
            .collect();
        if batches.is_empty() {
            batches.push(RecordBatch::new_empty(Arc::clone(&schema)));
        }
        let mut request = RowLevelWrite::new(self.op, "`db`.`t`", schema, touched, None);
        request.write_id = Some(format!("dml-{}", super::sink::unique_u64()));
        let sink = self.table.rewriter(&request)?;
        let input = Arc::new(PerDriver {
            schema: self.input(),
            batches,
            stall: false,
        });
        run(self.env, sink, input, None).0
    }

    /// `DELETE FROM t WHERE condition`.
    fn delete(&self, condition: &str) -> Result<(), KrishivError> {
        let touched = self.find(condition);
        self.rewrite(
            touched,
            &format!(
                "SELECT {} FROM t WHERE NOT coalesce(({condition}), false)",
                self.select_list(None)
            ),
            Vec::new(),
        )
    }

    /// `UPDATE t SET name = value WHERE condition`.
    fn update(&self, condition: &str, value: &str) -> Result<(), KrishivError> {
        let touched = self.find(condition);
        let hit = format!("coalesce(({condition}), false)");
        self.rewrite(
            touched,
            &format!("SELECT {} FROM t", self.select_list(Some((&hit, value)))),
            Vec::new(),
        )
    }

    /// The rewrite's columns: `name` replaced where `set.0` holds by `set.1`;
    /// the lineage columns kept, an updated row's sequence number null.
    fn select_list(&self, set: Option<(&str, &str)>) -> String {
        let mut cols = vec!["id".to_owned(), "x".to_owned()];
        cols.push(match set {
            Some((hit, v)) => format!("CASE WHEN {hit} THEN {v} ELSE name END"),
            None => "name".to_owned(),
        });
        if self.lineage {
            cols.push(ROW_ID.to_owned());
            cols.push(match set {
                Some((hit, _)) => {
                    format!("CASE WHEN {hit} THEN CAST(NULL AS BIGINT) ELSE {LAST_UPDATED} END")
                }
                None => LAST_UPDATED.to_owned(),
            });
        }
        cols.join(", ")
    }
}

/// The current snapshot's live entries of each content, by path, and the
/// table's invariants: every live file listed once, no DV of a data file
/// that is not live.
fn live(env: &Env, m: &TableMetadata) -> (Vec<String>, Vec<String>) {
    let Some(s) = m.current_snapshot() else {
        return (Vec::new(), Vec::new());
    };
    let io = env.reg.file_io();
    let list = env
        .rt
        .block_on(read_manifest_files(
            &io,
            s.manifest_list(),
            m.format_version() as u8,
            MAX,
        ))
        .unwrap();
    let mut data = Vec::new();
    let mut deletes = Vec::new();
    let mut dv_of = Vec::new();
    for mf in list {
        let bytes = env
            .rt
            .block_on(async { io.new_input(&mf.manifest_path).unwrap().read().await })
            .unwrap();
        let (entries, _) = Manifest::parse_avro(&bytes).unwrap().into_parts();
        for e in entries {
            if e.status() == ManifestStatus::Deleted {
                continue;
            }
            let name = e.file_path().rsplit('/').next().unwrap().to_owned();
            match mf.content {
                ManifestContentType::Data => data.push(name),
                ManifestContentType::Deletes => {
                    if let Some(r) = e.data_file().referenced_data_file() {
                        dv_of.push(r.rsplit('/').next().unwrap().to_owned());
                    }
                    deletes.push(name);
                }
            }
        }
    }
    let unique: BTreeSet<&String> = data.iter().chain(&deletes).collect();
    assert_eq!(
        unique.len(),
        data.len() + deletes.len(),
        "{data:?} {deletes:?}"
    );
    for d in &dv_of {
        assert!(data.contains(d), "a delete file of {d}, which is not live");
    }
    data.sort();
    deletes.sort();
    (data, deletes)
}

fn summary(m: &TableMetadata) -> (Operation, HashMap<String, String>) {
    let s = m.current_snapshot().unwrap().summary();
    (s.operation.clone(), s.additional_properties.clone())
}

fn get<'m>(props: &'m HashMap<String, String>, key: &str) -> Option<&'m str> {
    props.get(key).map(String::as_str)
}

fn names(m: &TableMetadata, env: &Env, location: &str) -> usize {
    iceberg_rust_files(env, m, location).len()
}

/// `DELETE` on an unpartitioned and a partitioned v2 table: the touched files
/// replaced by their surviving rows, the others kept; the snapshot an
/// `overwrite` with Java's summary; iceberg-rust plans the same files.
fn delete_case(partitioned: bool) {
    let env = env();
    let t = create(&env, 2, partitioned, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    append(&env, &catalog, range(10, 20));
    let before = live(&env, &catalog.metadata()).0;
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id < 3")
        .unwrap();
    let m = catalog.metadata();
    assert_eq!(
        read_back(&env, &catalog),
        expected(3..20),
        "partitioned {partitioned}"
    );
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Overwrite);
    // Unpartitioned: one file of 0..10 replaced; partitioned by x = id % 3:
    // the files of x = 0, 1, 2 of the first append.
    let replaced = if partitioned { 3 } else { 1 };
    assert_eq!(
        get(&s, "deleted-data-files"),
        Some(replaced.to_string().as_str())
    );
    assert_eq!(get(&s, "deleted-records"), Some("10"));
    assert_eq!(get(&s, "added-records"), Some("7"));
    assert_eq!(get(&s, "total-records"), Some("17"));
    assert_eq!(
        get(&s, "total-data-files"),
        Some(if partitioned { "6" } else { "2" })
    );
    let (now, deletes) = live(&env, &m);
    assert!(deletes.is_empty());
    let kept: Vec<_> = before.iter().filter(|f| now.contains(f)).collect();
    assert_eq!(kept.len(), before.len() - replaced);
    assert_eq!(names(&m, &env, &catalog.location()), now.len());
}

#[test]
fn delete_rewrites_the_touched_files_unpartitioned() {
    delete_case(false);
}

#[test]
fn delete_rewrites_the_touched_files_partitioned() {
    delete_case(true);
}

/// `UPDATE` on a partitioned v2 table: only the touched partition's file is
/// rewritten, with the new values.
#[test]
fn update_rewrites_the_touched_file() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    Dml::open(&env, &catalog, RowLevelOperation::Update)
        .update("id = 4", "'u'")
        .unwrap();
    let m = catalog.metadata();
    let mut want = expected(0..9);
    want[4] = "4|1|u".to_owned();
    assert_eq!(read_back(&env, &catalog), want);
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "deleted-data-files"), Some("1"));
    assert_eq!(get(&s, "added-data-files"), Some("1"));
    assert_eq!(get(&s, "changed-partition-count"), Some("1"));
    assert_eq!(live(&env, &m).0.len(), 3);
}

/// `MERGE`: matched rows updated or deleted and unmatched source rows
/// inserted, in one rewrite (the SQL layer's join is the session's; here
/// its result).
#[test]
fn merge_updates_deletes_and_inserts() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 6));
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Merge);
    let touched = dml.find("id IN (2, 3)");
    // WHEN MATCHED AND id = 2 THEN UPDATE SET name = 'm'
    // WHEN MATCHED AND id = 3 THEN DELETE
    // WHEN NOT MATCHED THEN INSERT (40, 1, 'n40')
    dml.rewrite(
        touched,
        "SELECT id, x, CASE WHEN id = 2 THEN 'm' ELSE name END FROM t WHERE id <> 3",
        vec![range(40, 41)],
    )
    .unwrap();
    let mut want = expected([0, 1, 2, 4, 5, 40]);
    want[2] = "2|2|m".to_owned();
    assert_eq!(read_back(&env, &catalog), want);
    assert_eq!(summary(&catalog.metadata()).0, Operation::Overwrite);
}

/// v3: a moved row keeps `_row_id` and `_last_updated_sequence_number`, an
/// updated row keeps its id and takes the new sequence number, an inserted
/// row new values; a rewrite without the lineage columns is refused.
#[test]
fn v3_rewrites_keep_row_lineage() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Merge);
    let touched = dml.find("id = 2");
    let insert = {
        let b = range(40, 41);
        let mut cols: Vec<ArrayRef> = b.columns().to_vec();
        cols.push(Arc::new(Int64Array::from(vec![None::<i64>])));
        cols.push(Arc::new(Int64Array::from(vec![None::<i64>])));
        RecordBatch::try_new(dml.input(), cols).unwrap()
    };
    dml.rewrite(
        touched,
        &format!(
            "SELECT id, x, CASE WHEN id = 2 THEN 'u' ELSE name END, {ROW_ID}, \
             CASE WHEN id = 2 THEN CAST(NULL AS BIGINT) ELSE {LAST_UPDATED} END FROM t"
        ),
        vec![insert],
    )
    .unwrap();
    let p = provider(&env, &catalog.location(), &catalog);
    let scan = p.lineage_scan().unwrap();
    let mut schema: Vec<StructField> = p.snapshot().schema.fields().to_vec();
    schema.push(StructField::new(ROW_ID, DataType::Long, true));
    schema.push(StructField::new(LAST_UPDATED, DataType::Long, true));
    let rows = render(
        &session(vec![("t", StructType::new(schema), Arc::new(Over(scan)))])
            .run(&format!(
                "SELECT id, name, {ROW_ID}, {LAST_UPDATED} FROM t ORDER BY id"
            ))
            .unwrap(),
    );
    // Appended at sequence 1 with row ids 0..5; the merge is sequence 2.
    // The inserted row takes a new id from the new files' range (5..11:
    // the rows moved keep theirs, stored).
    assert_eq!(
        rows[..5],
        ["0|n0|0|1", "1|n1|1|1", "2|u|2|2", "3|n3|3|1", "4|n4|4|1"]
    );
    let inserted: Vec<&str> = rows[5].split('|').collect();
    assert_eq!((inserted[0], inserted[1], inserted[3]), ("40", "n40", "2"));
    let id: i64 = inserted[2].parse().unwrap();
    assert!((5..11).contains(&id), "{id}");
    let m = catalog.metadata();
    assert_eq!(m.next_row_id(), 11);
    live(&env, &m);

    // Without the lineage columns: refused before anything is written.
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    let mut request = RowLevelWrite::new(
        RowLevelOperation::Delete,
        "`db`.`t`",
        input_schema(),
        Vec::new(),
        None,
    );
    request.write_id = Some("w".to_owned());
    let Err(err) = dml.table.rewriter(&request) else {
        panic!("a rewrite without lineage was accepted")
    };
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{err}"
    );
    assert!(err.to_string().contains("row lineage"), "{err}");
}

/// A `DELETE` that matches nothing still commits an empty `overwrite`
/// snapshot (Spark's copy-on-write commits it).
#[test]
fn an_empty_delete_commits_an_overwrite_snapshot() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let before = live(&env, &catalog.metadata());
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id > 100")
        .unwrap();
    let m = catalog.metadata();
    assert_eq!(m.snapshots().count(), 2);
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "deleted-data-files"), None);
    assert_eq!(get(&s, "added-data-files"), None);
    assert_eq!(get(&s, "total-records"), Some("5"));
    assert_eq!(live(&env, &m), before);
    assert_eq!(read_back(&env, &catalog), expected(0..5));
}

/// Commits a snapshot adding `deletes` (built by `make` from the data file
/// paths of the current snapshot) on top of the current one, as a foreign
/// `RowDelta` would.
fn add_deletes(
    env: &Env,
    catalog: &Arc<Catalog>,
    make: impl FnOnce(&[String]) -> Vec<iceberg::spec::DataFile>,
) {
    let m = catalog.metadata();
    let io = env.reg.file_io();
    let parent = m.current_snapshot().unwrap();
    let files: Vec<String> = iceberg_rust_files(env, &m, &catalog.location());
    let deletes = make(&files);
    let id = 555;
    let seq = m.last_sequence_number() + 1;
    let snapshot = env.rt.block_on(async {
        let mut base =
            read_manifest_files(&io, parent.manifest_list(), m.format_version() as u8, MAX)
                .await
                .unwrap();
        let mut w = manifest_writer(
            &io,
            &env.at("rowdelta-m0.avro"),
            &m,
            id,
            0,
            ManifestContentType::Deletes,
        )
        .unwrap();
        for d in deletes {
            w.add_file(d, UNASSIGNED_SEQUENCE_NUMBER).unwrap();
        }
        base.push(w.write_manifest_file().await.unwrap());
        let list = env.at("snap-rowdelta.avro");
        let out = io.new_output(&list).unwrap().writer().await.unwrap();
        let mut lw = if m.format_version() == FormatVersion::V3 {
            ManifestListWriter::v3(
                out,
                id,
                Some(parent.snapshot_id()),
                seq,
                Some(m.next_row_id()),
            )
        } else {
            ManifestListWriter::v2(out, id, Some(parent.snapshot_id()), seq)
        };
        lw.add_manifests(base.into_iter()).unwrap();
        let next = lw.next_row_id();
        lw.close().await.unwrap();
        let b = Snapshot::builder()
            .with_snapshot_id(id)
            .with_parent_snapshot_id(Some(parent.snapshot_id()))
            .with_sequence_number(seq)
            .with_timestamp_ms(m.last_updated_ms() + 1)
            .with_manifest_list(list)
            .with_schema_id(m.current_schema_id())
            .with_summary(Summary {
                operation: Operation::Delete,
                additional_properties: HashMap::new(),
            });
        match next {
            Some(n) if m.format_version() == FormatVersion::V3 => b
                .with_row_range(m.next_row_id(), n - m.next_row_id())
                .build(),
            _ => b.build(),
        }
    });
    let updates = vec![
        TableUpdate::AddSnapshot { snapshot },
        TableUpdate::SetSnapshotRef {
            ref_name: "main".to_owned(),
            reference: SnapshotReference::new(
                id,
                SnapshotRetention::Branch {
                    min_snapshots_to_keep: None,
                    max_snapshot_age_ms: None,
                    max_ref_age_ms: None,
                },
            ),
        },
    ];
    let requirements = vec![TableRequirement::RefSnapshotIdMatch {
        r#ref: "main".to_owned(),
        snapshot_id: Some(parent.snapshot_id()),
    }];
    let out = env.rt.block_on(catalog.commit(requirements, updates));
    assert!(
        matches!(out, super::commit::CommitOutcome::Committed(_)),
        "{out:?}"
    );
}

/// v3: the DV of a rewritten data file is removed with it; no DV is left
/// for a file that is not live.
#[test]
fn a_dv_of_a_rewritten_file_is_removed_with_it() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    add_deletes(&env, &catalog, |files| {
        let bitmap: roaring::RoaringTreemap = [0_u64, 1].into_iter().collect();
        let blob = encode_dv(&bitmap).unwrap();
        let mut bytes = b"PFA1".to_vec();
        bytes.extend_from_slice(&blob);
        let path = env.at("a.puffin");
        std::fs::write(path.strip_prefix("file://").unwrap(), &bytes).unwrap();
        let mut d = DataFileBuilder::default();
        d.content(DataContentType::PositionDeletes)
            .file_path(path)
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(bytes.len() as u64)
            .record_count(2)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .referenced_data_file(Some(files[0].clone()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(blob.len() as i64));
        vec![d.build().unwrap()]
    });
    assert_eq!(read_back(&env, &catalog), expected(2..10));
    assert_eq!(live(&env, &catalog.metadata()).1, ["a.puffin"]);
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 5")
        .unwrap();
    let m = catalog.metadata();
    assert_eq!(read_back(&env, &catalog), expected([2, 3, 4, 6, 7, 8, 9]));
    let (data, deletes) = live(&env, &m);
    assert_eq!(data.len(), 1);
    assert!(deletes.is_empty(), "{deletes:?}");
    let s = summary(&m).1;
    assert_eq!(get(&s, "removed-dvs"), Some("1"));
    assert_eq!(get(&s, "removed-delete-files"), Some("1"));
}

/// v2: a position delete file of a rewritten data file, older than every
/// data file left, is dropped in the same commit (Java's
/// `dropDeleteFilesOlderThan`).
#[test]
fn an_old_position_delete_file_goes_with_its_data_file() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    add_deletes(&env, &catalog, |files| {
        let path = env.at("a-deletes.parquet");
        let b = super::provider_tests::batch(
            vec![
                id_field("file_path", A::Utf8, DELETE_FILE_PATH_ID),
                id_field("pos", A::Int64, DELETE_FILE_POS_ID),
            ],
            vec![
                Arc::new(arrow_array::StringArray::from(vec![files[0].clone()])),
                Arc::new(Int64Array::from(vec![0_i64])),
            ],
        );
        let size = write_parquet(&path, &b);
        let mut d = DataFileBuilder::default();
        d.content(DataContentType::PositionDeletes)
            .file_path(path)
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(size)
            .record_count(1)
            .partition(Struct::empty())
            .partition_spec_id(0);
        vec![d.build().unwrap()]
    });
    append(&env, &catalog, range(20, 22));
    assert_eq!(read_back(&env, &catalog).len(), 11);
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 5")
        .unwrap();
    let m = catalog.metadata();
    let mut want = expected([1, 2, 3, 4, 6, 7, 8, 9]);
    want.extend(expected(20..22));
    assert_eq!(read_back(&env, &catalog), want);
    assert!(live(&env, &m).1.is_empty());
    let s = summary(&m).1;
    assert_eq!(get(&s, "removed-position-delete-files"), Some("1"));
    assert_eq!(get(&s, "removed-position-deletes"), Some("1"));
}

/// Concurrency: a foreign append committed after the statement read the
/// table fails it under `serializable` (the filter, absent, matches every
/// row), with the statement's files removed; under `snapshot` it commits on
/// top of the append.
#[test]
fn a_foreign_append_conflicts_only_under_serializable() {
    for level in ["serializable", "snapshot"] {
        let env = env();
        let t = create(&env, 2, false, &[("write.delete.isolation-level", level)]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 5));
        let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
        let touched = dml.find("id = 1");
        append(&env, &catalog, range(100, 102));
        let files_before = parquet_files(&env, false).len();
        let out = dml.rewrite(
            touched,
            "SELECT id, x, name FROM t WHERE NOT coalesce((id = 1), false)",
            Vec::new(),
        );
        let m = catalog.metadata();
        if level == "serializable" {
            let err = out.unwrap_err();
            assert_eq!(
                err.error_class(),
                ErrorClass::KrishivIcebergCommitConflict,
                "{err}"
            );
            assert!(
                err.to_string()
                    .contains("Found conflicting files that can contain records matching true"),
                "{err}"
            );
            let mut want = expected(0..5);
            want.extend(expected(100..102));
            assert_eq!(read_back(&env, &catalog), want);
            assert_eq!(m.snapshots().count(), 2);
            assert_eq!(parquet_files(&env, true).len(), files_before);
        } else {
            out.unwrap();
            let mut want = expected([0, 2, 3, 4]);
            want.extend(expected(100..102));
            assert_eq!(read_back(&env, &catalog), want);
            assert_eq!(m.snapshots().count(), 3);
            assert_eq!(summary(&m).0, Operation::Overwrite);
        }
        live(&env, &m);
    }
}

/// A foreign removal of a touched file (another `DELETE` of all its rows)
/// fails the statement: "Missing required files to delete".
#[test]
fn a_foreign_removal_of_a_touched_file_fails() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let mine = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    let touched = mine.find("id = 3");
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id < 100")
        .unwrap();
    assert_eq!(summary(&catalog.metadata()).0, Operation::Delete);
    let err = mine
        .rewrite(
            touched,
            "SELECT id, x, name FROM t WHERE NOT coalesce((id = 3), false)",
            Vec::new(),
        )
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{err}"
    );
    assert!(
        err.to_string().contains("Missing required files to delete"),
        "{err}"
    );
    assert!(read_back(&env, &catalog).is_empty());
    live(&env, &catalog.metadata());
}

/// Design note §7.3: a foreign merge append rewrites the manifest that holds
/// the touched file (and an older `DELETED` entry) between the read and the
/// commit; the rebased commit removes the file from the merged manifest, every
/// live file is listed once, and the older removal is not listed again.
#[test]
fn rebase_after_foreign_manifest_merge() {
    let env = env();
    let t = create(
        &env,
        2,
        false,
        &[
            ("commit.manifest.min-count-to-merge", "2"),
            ("write.delete.isolation-level", "snapshot"),
        ],
    );
    let catalog = Catalog::new(&t);
    // One manifest of two files; a DELETE of the first leaves it with a
    // DELETED entry and the second EXISTING.
    append(&env, &catalog, range(0, 3));
    append(&env, &catalog, range(3, 6));
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id < 3")
        .unwrap();
    let s0 = catalog.metadata();
    assert_eq!(live(&env, &s0).0.len(), 1);
    let mine = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    let touched = mine.find("id = 4");
    // The foreign merge append.
    append(&env, &catalog, range(10, 12));
    let merged = catalog.metadata();
    assert_eq!(
        get(&summary(&merged).1, "manifests-replaced").map(|v| v != "0"),
        Some(true)
    );
    mine.rewrite(
        touched,
        "SELECT id, x, name FROM t WHERE NOT coalesce((id = 4), false)",
        Vec::new(),
    )
    .unwrap();
    let m = catalog.metadata();
    let mut want = expected([3, 5]);
    want.extend(expected(10..12));
    assert_eq!(read_back(&env, &catalog), want);
    let (data, _) = live(&env, &m);
    assert_eq!(data.len(), 2, "{data:?}");
    assert_eq!(names(&m, &env, &catalog.location()), 2);
    // The commit removed exactly the touched file.
    assert_eq!(get(&summary(&m).1, "deleted-data-files"), Some("1"));
}

/// A cancelled statement leaves nothing visible: no snapshot, no file.
#[test]
fn a_cancelled_statement_leaves_nothing() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let files = parquet_files(&env, false);
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    let touched = dml.find("id = 1");
    let mut request = RowLevelWrite::new(
        RowLevelOperation::Delete,
        "`db`.`t`",
        input_schema(),
        touched,
        None,
    );
    request.write_id = Some("cancelled".to_owned());
    let sink = dml.table.rewriter(&request).unwrap();
    let input = Arc::new(PerDriver {
        schema: input_schema(),
        batches: vec![range(0, 1)],
        stall: true,
    });
    let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&started);
    let (out, _pool, _runner) = run(
        &env,
        sink,
        input,
        Some(Box::new(move || {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            true
        })),
    );
    assert!(out.is_err());
    assert_eq!(catalog.metadata().snapshots().count(), 1);
    assert_eq!(parquet_files(&env, true), files);
    assert_eq!(read_back(&env, &catalog), expected(0..5));
}

/// The Iceberg scan reads rows again by position (`supports_take`, what
/// `RowIdScan` requires), by field id; identity-partition columns are not in
/// the files and are refused.
#[test]
fn the_iceberg_scan_takes_rows_by_position() {
    use krishiv_memory::{MemoryConfig, MemoryPool as _, Reservation, TreePool};
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    let p = provider(&env, &catalog.location(), &catalog);
    let scan = p.scan().unwrap();
    assert!(scan.supports_take());
    let file = p.files()[0].clone();
    let pool = TreePool::new(&MemoryConfig::default());
    let take = |columns: &[&str]| {
        let id = pool.register("take", None);
        let rows = krishiv_exec::TakeRows {
            file: file.clone(),
            rows: vec![2, 0],
            projection: columns
                .iter()
                .map(|c| krishiv_io::ColumnPath::new(*c))
                .collect(),
            mem: Reservation::new(Arc::clone(&pool) as _, id),
            gate: Arc::new(krishiv_io::ScanGate::new(
                &(Arc::clone(&pool) as _),
                1 << 30,
            )),
        };
        env.rt.block_on(async {
            let mut s = scan.take(rows).await?;
            let mut out = Vec::new();
            while let Some(b) = s.next_batch().await? {
                out.push(b);
            }
            Ok::<_, KrishivError>(out)
        })
    };
    let batches = take(&["id", "name"]).unwrap();
    let ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    // The file of x = 0 holds 0, 3, 6: positions 0 and 2.
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|i| i % 3 == ids[0] % 3), "{ids:?}");
    let err = take(&["x"]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{err}"
    );
}
