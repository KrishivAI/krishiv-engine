// SPDX-License-Identifier: Apache-2.0
//! Tests of merge-on-read `DELETE`, `UPDATE` and `MERGE` (`row_delta`,
//! `delete_write`, PR-36c part 1) through [`MergeOnReadWrite`]: the row ids
//! found by SQL over the provider's row-id scan, deleted (and rows inserted)
//! per task, committed by `sink_tests`' catalog; read back through SQL and
//! iceberg-rust's planning, with `cow_tests`' table invariants.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as A, Schema, SchemaRef};
use iceberg::TableUpdate;
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, Manifest, ManifestContentType,
    ManifestStatus, Operation, Struct, TableMetadata,
};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::{RowIdScan, RowLevelOperation};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN};
use krishiv_memory::{ConsumerHandle, MemoryConfig, MemoryPool, TreePool};
use krishiv_types::{DataType, StructField, StructType};

use super::append_tests::MAX;
use super::commit::{CommitOutcome, CommitTable};
use super::cow_tests::{
    Over, add_deletes, append, create, dvs_in_one_puffin, get, live, position_delete,
    remove_all_deletes, summary,
};
use super::lineage::{LAST_UPDATED, ROW_ID, lineage_fields};
use super::provider::IcebergTableProvider;
use super::provider_tests::{Env, env, id_field, session, write_parquet};
use super::row_delta::{MergeOnReadWrite, RowLevelMode, TaskOutput, row_level_mode};
use super::scan::read_manifest_files;
use super::sink_tests::{Catalog, expected, input_schema, provider, range, read_back};
use crate::plan::{DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID, decode_dv};

/// A row id: the encoded file reference and the position.
pub(super) type RowId = (Vec<u8>, i64);

/// What one task of a statement does.
#[derive(Default)]
pub(super) struct Task {
    pub(super) deletes: Vec<RowId>,
    pub(super) inserts: Vec<RecordBatch>,
}

/// One merge-on-read statement against the table as a provider pinned it.
pub(super) struct Mor<'a> {
    pub(super) env: &'a Env,
    pub(super) p: IcebergTableProvider,
    pub(super) op: RowLevelOperation,
    lineage: bool,
}

impl<'a> Mor<'a> {
    pub(super) fn open(env: &'a Env, catalog: &Arc<Catalog>, op: RowLevelOperation) -> Self {
        Self::over(env, provider(env, &catalog.location(), catalog), op)
    }

    pub(super) fn over(env: &'a Env, p: IcebergTableProvider, op: RowLevelOperation) -> Self {
        let lineage = p.snapshot().format_version >= 3;
        Self {
            env,
            p,
            op,
            lineage,
        }
    }

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

    fn query(&self, sql: &str) -> Vec<RecordBatch> {
        let scan =
            Arc::new(RowIdScan::new(self.p.row_level_scan(self.lineage, None).unwrap()).unwrap());
        session(vec![("t", self.columns(), Arc::new(Over(scan)))])
            .run(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    /// The row ids where `condition` holds.
    pub(super) fn ids(&self, condition: &str) -> Vec<RowId> {
        let mut out = Vec::new();
        for b in self.query(&format!(
            "SELECT {ROW_FILE_COLUMN}, {ROW_POSITION_COLUMN} FROM t WHERE coalesce(({condition}), \
             false)"
        )) {
            let (f, p) = (b.column(0), b.column(1).as_primitive::<Int64Type>());
            for i in 0..b.num_rows() {
                let file = match f.data_type() {
                    A::BinaryView => f.as_binary_view().value(i).to_vec(),
                    _ => f.as_binary::<i32>().value(i).to_vec(),
                };
                out.push((file, p.value(i)));
            }
        }
        out
    }

    /// The schema of inserted rows: the table's, then lineage on v3.
    fn input(&self) -> SchemaRef {
        let mut fields: Vec<_> = input_schema().fields().iter().cloned().collect();
        if self.lineage {
            fields.extend(lineage_fields());
        }
        Arc::new(Schema::new(fields))
    }

    /// The rows where `condition` holds with `name` set to `value` (an
    /// update's new rows: lineage kept, the sequence number null).
    pub(super) fn updated(&self, condition: &str, value: &str) -> Vec<RecordBatch> {
        let mut cols = format!("id, x, {value}");
        if self.lineage {
            cols.push_str(&format!(", {ROW_ID}, CAST(NULL AS BIGINT)"));
        }
        self.query(&format!(
            "SELECT {cols} FROM t WHERE coalesce(({condition}), false)"
        ))
        .into_iter()
        .map(|b| RecordBatch::try_new(self.input(), b.columns().to_vec()).unwrap())
        .collect()
    }

    /// New rows `from..to` (nulls for lineage on v3).
    pub(super) fn new_rows(&self, from: i64, to: i64) -> RecordBatch {
        let b = range(from, to);
        let mut cols: Vec<ArrayRef> = b.columns().to_vec();
        if self.lineage {
            let n = b.num_rows();
            cols.push(Arc::new(Int64Array::from(vec![None::<i64>; n])));
            cols.push(Arc::new(Int64Array::from(vec![None::<i64>; n])));
        }
        RecordBatch::try_new(self.input(), cols).unwrap()
    }

    /// The statement's writer.
    pub(super) fn writer(&self) -> Arc<MergeOnReadWrite> {
        MergeOnReadWrite::new(
            &self.p,
            self.op,
            None,
            Some(format!("mor-{}", super::sink::unique_u64())),
        )
        .unwrap()
    }

    /// Runs `tasks` and commits them.
    pub(super) fn run(&self, tasks: Vec<Task>) -> Result<TableMetadata, KrishivError> {
        let w = self.writer();
        let outputs = self.write(&w, tasks)?;
        self.env.rt.block_on(w.commit(outputs, Some("q")))
    }

    /// Runs `tasks` up to their outputs.
    pub(super) fn write(
        &self,
        w: &Arc<MergeOnReadWrite>,
        tasks: Vec<Task>,
    ) -> Result<Vec<TaskOutput>, KrishivError> {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        self.write_in(w, tasks, &pool)
    }

    /// [`write`](Self::write) drawing on `pool`.
    pub(super) fn write_in(
        &self,
        w: &Arc<MergeOnReadWrite>,
        tasks: Vec<Task>,
        pool: &Arc<dyn MemoryPool>,
    ) -> Result<Vec<TaskOutput>, KrishivError> {
        self.env.rt.block_on(async {
            let mut out = Vec::new();
            for (i, t) in tasks.into_iter().enumerate() {
                let mut task = w.task(
                    u32::try_from(i).unwrap(),
                    ConsumerHandle::register(Arc::clone(pool), "mor", None),
                );
                for (f, p) in &t.deletes {
                    task.delete(f, *p)?;
                }
                task.settle().await?;
                for b in &t.inserts {
                    task.insert(b).await?;
                }
                out.push(task.finish().await?);
            }
            Ok(out)
        })
    }

    /// `DELETE FROM t WHERE condition` in one task.
    pub(super) fn delete(&self, condition: &str) -> Result<TableMetadata, KrishivError> {
        self.run(vec![Task {
            deletes: self.ids(condition),
            inserts: Vec::new(),
        }])
    }

    /// `UPDATE t SET name = value WHERE condition` in one task.
    pub(super) fn update(
        &self,
        condition: &str,
        value: &str,
    ) -> Result<TableMetadata, KrishivError> {
        self.run(vec![Task {
            deletes: self.ids(condition),
            inserts: self.updated(condition, value),
        }])
    }
}

/// Each live delete file entry of `m`'s current snapshot: its name, format,
/// referenced data file, `file_path` bounds and record count.
pub(super) struct DeleteEntry {
    pub(super) name: String,
    pub(super) format: DataFileFormat,
    pub(super) referenced: Option<String>,
    pub(super) bounds: Option<(Vec<u8>, Vec<u8>)>,
    pub(super) records: u64,
    pub(super) file: iceberg::spec::DataFile,
}

pub(super) fn delete_entries(env: &Env, m: &TableMetadata) -> Vec<DeleteEntry> {
    let io = env.reg.file_io();
    let s = m.current_snapshot().unwrap();
    let list = env
        .rt
        .block_on(read_manifest_files(
            &io,
            s.manifest_list(),
            m.format_version() as u8,
            MAX,
        ))
        .unwrap();
    let mut out = Vec::new();
    for mf in list
        .iter()
        .filter(|m| m.content == ManifestContentType::Deletes)
    {
        let bytes = env
            .rt
            .block_on(async { io.new_input(&mf.manifest_path).unwrap().read().await })
            .unwrap();
        for e in Manifest::parse_avro(&bytes).unwrap().entries() {
            if e.status() == ManifestStatus::Deleted {
                continue;
            }
            let f = e.data_file();
            let bound = |b: &HashMap<i32, iceberg::spec::Datum>| {
                b.get(&DELETE_FILE_PATH_ID)
                    .map(|d| d.to_bytes().unwrap().to_vec())
            };
            out.push(DeleteEntry {
                name: f.file_path().rsplit('/').next().unwrap().to_owned(),
                format: f.file_format(),
                referenced: f.referenced_data_file(),
                bounds: bound(f.lower_bounds()).zip(bound(f.upper_bounds())),
                records: f.record_count(),
                file: f.clone(),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// iceberg-rust's plan of `m`: each data file's name with the names of the
/// delete files it assigns, by name.
pub(super) fn iceberg_rust_plan(
    env: &Env,
    m: &TableMetadata,
    location: &str,
) -> Vec<(String, Vec<String>)> {
    let table = iceberg::table::Table::builder()
        .metadata(m.clone())
        .metadata_location(location)
        .identifier(iceberg::TableIdent::from_strs(["db", "t"]).unwrap())
        .file_io(env.reg.file_io())
        .runtime(env.reg.runtime().unwrap())
        .build()
        .unwrap();
    let name = |p: &str| p.rsplit('/').next().unwrap().to_owned();
    let mut out: Vec<(String, Vec<String>)> = env.rt.block_on(async {
        use futures::TryStreamExt;
        table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .map_ok(|t| {
                let mut d: Vec<String> = t.deletes().iter().map(|d| name(&d.file_path)).collect();
                d.sort();
                (name(t.data_file_path()), d)
            })
            .try_collect()
            .await
            .unwrap()
    });
    out.sort();
    out
}

/// The files under the test's directory whose names end with `suffix`.
pub(super) fn files_ending(env: &Env, suffix: &str) -> Vec<String> {
    fn walk(dir: &std::path::Path, suffix: &str, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, suffix, out);
            } else if p.display().to_string().ends_with(suffix) {
                out.push(p.display().to_string());
            }
        }
    }
    let mut out = Vec::new();
    walk(env.dir.path(), suffix, &mut out);
    out.sort();
    out
}

/// The paths of the catalog's live data files (sorted by path, so their
/// order is random across runs: pick one by identity, never by position
/// when there are several).
pub(super) fn data_files(env: &Env, catalog: &Arc<Catalog>) -> Vec<String> {
    super::append_tests::iceberg_rust_files(env, &catalog.metadata(), &catalog.location())
}

/// The one data file of a table with a single data file.
pub(super) fn only_data_file(env: &Env, catalog: &Arc<Catalog>) -> String {
    let mut files = data_files(env, catalog);
    assert_eq!(files.len(), 1, "{files:?}");
    files.pop().unwrap()
}

fn without(ids: impl IntoIterator<Item = i64>, gone: &[i64]) -> Vec<String> {
    expected(ids.into_iter().filter(|i| !gone.contains(i)))
}

#[test]
fn the_mode_follows_the_table_property() {
    let props = |k: &str, v: &str| HashMap::from([(k.to_owned(), v.to_owned())]);
    let none = HashMap::new();
    for op in [
        RowLevelOperation::Delete,
        RowLevelOperation::Update,
        RowLevelOperation::Merge,
    ] {
        assert_eq!(
            row_level_mode(&none, op).unwrap(),
            RowLevelMode::CopyOnWrite
        );
    }
    assert_eq!(
        row_level_mode(
            &props("write.update.mode", "Merge-On-Read"),
            RowLevelOperation::Update
        )
        .unwrap(),
        RowLevelMode::MergeOnRead
    );
    // Another statement's property does not apply.
    assert_eq!(
        row_level_mode(
            &props("write.update.mode", "merge-on-read"),
            RowLevelOperation::Delete
        )
        .unwrap(),
        RowLevelMode::CopyOnWrite
    );
    let e =
        row_level_mode(&props("write.merge.mode", "mor"), RowLevelOperation::Merge).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.to_string()
            .contains("Unknown row-level operation mode: mor"),
        "{e}"
    );
}

/// A format v1 table refuses merge-on-read with Java's text.
#[test]
fn a_v1_table_refuses_merge_on_read() {
    let env = env();
    let creation = iceberg::TableCreation::builder()
        .name("t".to_owned())
        .schema(super::provider_tests::ixn_schema())
        .format_version(FormatVersion::V1)
        .build();
    let t = env.rt.block_on(async {
        use iceberg::Catalog as _;
        env.catalog
            .create_table(&iceberg::NamespaceIdent::new("db".to_owned()), creation)
            .await
            .unwrap()
    });
    let catalog = Catalog::new(&t);
    let p = provider(&env, &catalog.location(), &catalog);
    let e = MergeOnReadWrite::new(&p, RowLevelOperation::Delete, None, None).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.to_string()
            .contains("Deletes are supported in V2 and above"),
        "{e}"
    );
}

/// v2 `DELETE`: one position delete file per data file, file-scoped by its
/// equal `file_path` bounds, `referenced_data_file` unset; operation
/// `delete`; Java's summary; iceberg-rust assigns each file its delete.
#[test]
fn a_v2_delete_writes_one_position_delete_file_per_data_file() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    append(&env, &catalog, range(10, 20));
    let m = Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id IN (1, 2, 12)")
        .unwrap();
    assert_eq!(read_back(&env, &catalog), without(0..20, &[1, 2, 12]));
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Delete);
    assert_eq!(get(&s, "added-position-delete-files"), Some("2"));
    assert_eq!(get(&s, "added-delete-files"), Some("2"));
    assert_eq!(get(&s, "added-position-deletes"), Some("3"));
    assert_eq!(get(&s, "added-dvs"), None);
    assert_eq!(get(&s, "added-data-files"), None);
    assert_eq!(get(&s, "total-position-deletes"), Some("3"));
    assert_eq!(get(&s, "total-delete-files"), Some("2"));
    let deletes = delete_entries(&env, &m);
    assert_eq!(deletes.len(), 2);
    let sizes: u64 = deletes.iter().map(|d| d.file.file_size_in_bytes()).sum();
    assert_eq!(
        get(&s, "added-files-size"),
        Some(sizes.to_string().as_str())
    );
    let (data, _) = live(&env, &m);
    for d in &deletes {
        assert_eq!(d.format, DataFileFormat::Parquet);
        assert!(
            d.name.ends_with("-00001-deletes.parquet")
                || d.name.ends_with("-00002-deletes.parquet"),
            "{}",
            d.name
        );
        assert_eq!(d.referenced, None);
        let (lo, hi) = d.bounds.clone().expect("file_path bounds");
        assert_eq!(lo, hi);
        let path = String::from_utf8(lo).unwrap();
        assert!(data.iter().any(|f| path.ends_with(f.as_str())), "{path}");
    }
    let plan = iceberg_rust_plan(&env, &m, &catalog.location());
    assert_eq!(plan.len(), 2);
    for (_, d) in &plan {
        assert_eq!(d.len(), 1, "{plan:?}");
    }
}

/// v2, partitioned by `x`: each position delete file sits in its data
/// file's partition directory and carries its tuple.
#[test]
fn a_v2_position_delete_file_is_written_in_its_partition() {
    let env = env();
    let t = create(&env, 2, true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    let m = Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id IN (4, 5)")
        .unwrap();
    assert_eq!(read_back(&env, &catalog), without(0..9, &[4, 5]));
    let files = files_ending(&env, "-deletes.parquet");
    assert_eq!(files.len(), 2, "{files:?}");
    assert!(files.iter().any(|f| f.contains("/x=1/")), "{files:?}");
    assert!(files.iter().any(|f| f.contains("/x=2/")), "{files:?}");
    assert_eq!(get(&summary(&m).1, "changed-partition-count"), Some("2"));
    live(&env, &m);
}

/// v3 `DELETE` in two tasks over two data files: one Puffin file per task,
/// one DV each; `added-files-size` counts the blobs, not the Puffin files;
/// the blobs decode to the deleted positions.
#[test]
fn a_v3_delete_writes_one_puffin_file_per_task() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    append(&env, &catalog, range(10, 20));
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
    let m = mor
        .run(vec![
            Task {
                deletes: mor.ids("id IN (3, 4)"),
                ..Task::default()
            },
            Task {
                deletes: mor.ids("id = 15"),
                ..Task::default()
            },
        ])
        .unwrap();
    assert_eq!(read_back(&env, &catalog), without(0..20, &[3, 4, 15]));
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Delete);
    assert_eq!(get(&s, "added-dvs"), Some("2"));
    assert_eq!(get(&s, "added-delete-files"), Some("2"));
    assert_eq!(get(&s, "added-position-deletes"), Some("3"));
    assert_eq!(get(&s, "added-position-delete-files"), None);
    let deletes = delete_entries(&env, &m);
    let names: BTreeSet<&str> = deletes.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names.len(), 2, "{names:?}");
    assert!(names.iter().all(|n| n.ends_with("-00001-deletes.puffin")));
    let blobs: i64 = deletes
        .iter()
        .map(|d| d.file.content_size_in_bytes().unwrap())
        .sum();
    assert_eq!(
        get(&s, "added-files-size"),
        Some(blobs.to_string().as_str())
    );
    for d in &deletes {
        assert_eq!(d.format, DataFileFormat::Puffin);
        assert!(d.referenced.is_some());
        let local = d
            .file
            .file_path()
            .strip_prefix("file://")
            .unwrap()
            .to_owned();
        let bytes = std::fs::read(local).unwrap();
        assert_eq!(d.file.file_size_in_bytes(), bytes.len() as u64);
        assert_eq!(&bytes[..4], b"PFA1");
        let off = usize::try_from(d.file.content_offset().unwrap()).unwrap();
        let len = d.file.content_size_in_bytes().unwrap();
        let blob = &bytes[off..off + usize::try_from(len).unwrap()];
        let p = decode_dv(blob, len, d.records, "dv").unwrap();
        assert_eq!(p.len(), d.records);
    }
    for (_, d) in iceberg_rust_plan(&env, &m, &catalog.location()) {
        assert_eq!(d.len(), 1);
    }
    live(&env, &m);
}

/// `UPDATE` on v2 and v3: the old row deleted, the new one written in the
/// same snapshot, an `overwrite`; on v3 the updated row keeps its row id.
#[test]
fn an_update_deletes_and_inserts_in_one_snapshot() {
    for version in [2_u8, 3] {
        let env = env();
        let t = create(&env, version, false, &[]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 6));
        let m = Mor::open(&env, &catalog, RowLevelOperation::Update)
            .update("id = 4", "'u'")
            .unwrap();
        let mut want = expected(0..6);
        want[4] = "4|1|u".to_owned();
        assert_eq!(read_back(&env, &catalog), want, "v{version}");
        let (op, s) = summary(&m);
        assert_eq!(op, Operation::Overwrite);
        assert_eq!(get(&s, "added-data-files"), Some("1"));
        assert_eq!(get(&s, "added-records"), Some("1"));
        assert_eq!(get(&s, "added-delete-files"), Some("1"));
        assert_eq!(get(&s, "total-records"), Some("7"));
        if version == 3 {
            let rows = super::cow_tests::lineage_rows(&env, &catalog);
            assert!(rows.contains(&"4|4|2".to_owned()), "{rows:?}");
        }
        live(&env, &m);
    }
}

/// `MERGE`: matched rows updated and deleted, unmatched rows inserted
/// (`overwrite`); a `MERGE` that only inserts is an `append`.
#[test]
fn a_merge_commits_overwrite_or_append() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 6));
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Merge);
    let mut deletes = mor.ids("id IN (2, 3)");
    deletes.sort();
    let mut inserts = mor.updated("id = 2", "'m'");
    inserts.push(mor.new_rows(40, 41));
    let m = mor.run(vec![Task { deletes, inserts }]).unwrap();
    let mut want = without([0, 1, 2, 4, 5, 40], &[]);
    want[2] = "2|2|m".to_owned();
    assert_eq!(read_back(&env, &catalog), want);
    assert_eq!(summary(&m).0, Operation::Overwrite);
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Merge);
    let m = mor
        .run(vec![Task {
            deletes: Vec::new(),
            inserts: vec![mor.new_rows(50, 52)],
        }])
        .unwrap();
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Append);
    assert_eq!(get(&s, "added-delete-files"), None);
    assert_eq!(read_back(&env, &catalog).len(), 8);
    live(&env, &m);
}

/// A `DELETE` that matches nothing commits an empty `overwrite` snapshot
/// (Java's empty `RowDelta`) and writes no delete file.
#[test]
fn an_empty_delete_commits_an_overwrite_snapshot() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let m = Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id > 100")
        .unwrap();
    assert_eq!(m.snapshots().count(), 2);
    let (op, s) = summary(&m);
    assert_eq!(op, Operation::Overwrite);
    assert_eq!(get(&s, "added-delete-files"), None);
    assert!(files_ending(&env, ".puffin").is_empty());
    assert_eq!(read_back(&env, &catalog), expected(0..5));
}

/// v3: a data file's old DV is absorbed into the new one and removed in the
/// same commit (`removed-dvs`); a second statement absorbs Krishiv's own.
#[test]
fn a_v3_delete_absorbs_the_old_dv() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    add_deletes(&env, &catalog, |files| {
        dvs_in_one_puffin(&env, "old.puffin", files, &[0, 1])
    });
    assert_eq!(read_back(&env, &catalog), expected(2..10));
    let m = Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 5")
        .unwrap();
    assert_eq!(read_back(&env, &catalog), without(2..10, &[5]));
    let s = summary(&m).1;
    assert_eq!(get(&s, "removed-dvs"), Some("1"));
    assert_eq!(get(&s, "added-dvs"), Some("1"));
    assert_eq!(get(&s, "added-position-deletes"), Some("3"));
    assert_eq!(get(&s, "removed-position-deletes"), Some("2"));
    let deletes = delete_entries(&env, &m);
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].records, 3);
    let m = Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 7")
        .unwrap();
    assert_eq!(read_back(&env, &catalog), without(2..10, &[5, 7]));
    let deletes = delete_entries(&env, &m);
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].records, 4);
    live(&env, &m);
}

/// v3: two data files whose old DVs share one Puffin file, both touched by one UPDATE: each blob
/// is its own removal (the filter tells DVs apart by blob, not by file), so the statement does
/// not fail with "Missing required files to delete", and the summary counts two removed DVs.
#[test]
fn an_update_absorbs_old_dvs_that_share_a_puffin_file() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    append(&env, &catalog, range(10, 20));
    add_deletes(&env, &catalog, |files| {
        dvs_in_one_puffin(&env, "old.puffin", files, &[0, 1])
    });
    let m = Mor::open(&env, &catalog, RowLevelOperation::Update)
        .update("id IN (5, 15)", "'u'")
        .unwrap();
    assert_eq!(get(&summary(&m).1, "removed-dvs"), Some("2"));
    let mut want = without(0..20, &[0, 1, 10, 11, 5, 15]);
    want.extend(["5|2|u".to_owned(), "15|0|u".to_owned()]);
    want.sort();
    let mut got = read_back(&env, &catalog);
    got.sort();
    assert_eq!(got, want);
    live(&env, &m);
}

/// v2: a data file's earlier file-scoped position delete file (Krishiv's)
/// is absorbed into the new one and removed; a position delete file
/// covering two data files is neither merged nor removed.
#[test]
fn a_v2_delete_absorbs_file_scoped_position_deletes_only() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    append(&env, &catalog, range(10, 20));
    add_deletes(&env, &catalog, |files| vec![wide_delete(&env, files, 9)]);
    let model = [9, 19];
    assert_eq!(read_back(&env, &catalog), without(0..20, &model));
    Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 3")
        .unwrap();
    let m = Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 4")
        .unwrap();
    assert_eq!(read_back(&env, &catalog), without(0..20, &[3, 4, 9, 19]));
    let s = summary(&m).1;
    assert_eq!(get(&s, "removed-position-delete-files"), Some("1"));
    let deletes = delete_entries(&env, &m);
    let names: Vec<&str> = deletes.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(deletes.len(), 2, "{names:?}");
    assert!(names.contains(&"wide-deletes.parquet"), "{names:?}");
    let mine = deletes
        .iter()
        .find(|d| d.name != "wide-deletes.parquet")
        .unwrap();
    assert_eq!(mine.records, 2);
    live(&env, &m);
}

/// A position delete file deleting position `pos` of every file of
/// `files` (partition-scoped: no bounds).
fn wide_delete(env: &Env, files: &[String], pos: i64) -> iceberg::spec::DataFile {
    let path = env.at("wide-deletes.parquet");
    let b = super::provider_tests::batch(
        vec![
            id_field("file_path", A::Utf8, DELETE_FILE_PATH_ID),
            id_field("pos", A::Int64, DELETE_FILE_POS_ID),
        ],
        vec![
            Arc::new(StringArray::from(files.to_vec())),
            Arc::new(Int64Array::from(vec![pos; files.len()])),
        ],
    );
    let size = write_parquet(&path, &b);
    let mut d = DataFileBuilder::default();
    d.content(DataContentType::PositionDeletes)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(files.len() as u64)
        .partition(Struct::empty())
        .partition_spec_id(0);
    d.build().unwrap()
}

/// Upgrades the catalog's table to format v3.
pub(super) fn upgrade(env: &Env, catalog: &Arc<Catalog>) {
    let out = env.rt.block_on(catalog.commit(
        Vec::new(),
        vec![TableUpdate::UpgradeFormatVersion {
            format_version: FormatVersion::V3,
        }],
    ));
    assert!(matches!(out, CommitOutcome::Committed(_)), "{out:?}");
}

/// Design note §8 item 4's model case: a v2 table with a partition-scoped
/// position delete and Krishiv's file-scoped ones, upgraded to v3, then
/// Krishiv DVs that absorb every position delete of their data file (the
/// partition-scoped one included) and remove the file-scoped ones only;
/// the rows read back equal the model after every statement.
#[test]
fn delete_application_matches_model_with_krishiv_deletes_across_an_upgrade() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    // Rows 0..10; file names carry a random attempt id, so never by position.
    let first = only_data_file(&env, &catalog);
    append(&env, &catalog, range(10, 20));
    add_deletes(&env, &catalog, |files| vec![wide_delete(&env, files, 0)]);
    let mut gone = vec![0, 10];
    let check = |gone: &[i64]| assert_eq!(read_back(&env, &catalog), without(0..20, gone));
    check(&gone);
    for (cond, ids) in [("id = 3", vec![3]), ("id IN (4, 13)", vec![4, 13])] {
        Mor::open(&env, &catalog, RowLevelOperation::Delete)
            .delete(cond)
            .unwrap();
        gone.extend(ids);
        check(&gone);
    }
    let before = delete_entries(&env, &catalog.metadata());
    // The wide file and one file-scoped file per data file.
    assert_eq!(
        before.len(),
        3,
        "{:?}",
        before.iter().map(|d| &d.name).collect::<Vec<_>>()
    );
    upgrade(&env, &catalog);
    let m = Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id IN (5, 15)")
        .unwrap();
    gone.extend([5, 15]);
    check(&gone);
    let s = summary(&m).1;
    assert_eq!(get(&s, "added-dvs"), Some("2"));
    assert_eq!(get(&s, "removed-position-delete-files"), Some("2"));
    let after = delete_entries(&env, &m);
    let after_names: Vec<(&str, Option<&String>)> = after
        .iter()
        .map(|d| (d.name.as_str(), d.referenced.as_ref()))
        .collect();
    // Per data file, keyed by whether it holds rows 0..10.
    let records: HashMap<bool, u64> = after
        .iter()
        .filter(|d| d.format == DataFileFormat::Puffin)
        .map(|d| (d.referenced.as_deref() == Some(first.as_str()), d.records))
        .collect();
    // The first file: 0 (wide), 3, 4, 5; the second: 10 (wide), 13, 15.
    assert_eq!(
        records,
        HashMap::from([(true, 4), (false, 3)]),
        "{after_names:?}"
    );
    assert!(after.iter().any(|d| d.name == "wide-deletes.parquet"));
    assert_eq!(after.len(), 3);
    // Another statement on v3 keeps the model.
    Mor::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 18")
        .unwrap();
    gone.push(18);
    check(&gone);
    live(&env, &catalog.metadata());
}

/// Two tasks delete from one data file: their DVs are merged into one
/// `merged-dvs-<id>-1.puffin` (Java's `mergeDVs`); the tasks' own Puffin
/// files, left with no DV, are deleted.
#[test]
fn two_tasks_dvs_of_one_file_are_merged() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
    let m = mor
        .run(vec![
            Task {
                deletes: mor.ids("id = 1"),
                ..Task::default()
            },
            Task {
                deletes: mor.ids("id = 2"),
                ..Task::default()
            },
        ])
        .unwrap();
    assert_eq!(read_back(&env, &catalog), without(0..10, &[1, 2]));
    let deletes = delete_entries(&env, &m);
    assert_eq!(deletes.len(), 1);
    assert!(
        deletes[0].name.starts_with("merged-dvs-"),
        "{}",
        deletes[0].name
    );
    assert_eq!(deletes[0].records, 2);
    assert_eq!(files_ending(&env, ".puffin").len(), 1);
    live(&env, &m);
}

/// A DV another writer added for the same data file between the read and
/// the commit fails the statement under both isolation levels, `DELETE`
/// included (`validateAddedDVs`); the statement's files are removed.
#[test]
fn a_concurrent_dv_on_the_same_file_conflicts_at_both_levels() {
    for level in ["serializable", "snapshot"] {
        let env = env();
        let t = create(&env, 3, false, &[("write.delete.isolation-level", level)]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 10));
        let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
        let ids = mor.ids("id = 4");
        add_deletes(&env, &catalog, |files| {
            dvs_in_one_puffin(&env, "foreign.puffin", files, &[0])
        });
        let err = mor
            .run(vec![Task {
                deletes: ids,
                ..Task::default()
            }])
            .unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivIcebergCommitConflict,
            "{level}: {err}"
        );
        assert!(
            err.to_string().contains("Found concurrently added DV for"),
            "{err}"
        );
        assert_eq!(files_ending(&env, "-deletes.puffin"), Vec::<String>::new());
        assert_eq!(read_back(&env, &catalog), expected(1..10));
    }
}

/// A foreign append: a conflict under `serializable` (the filter, none,
/// matches every row), a commit on top under `snapshot`.
#[test]
fn a_foreign_append_conflicts_only_under_serializable() {
    for level in ["serializable", "snapshot"] {
        let env = env();
        let t = create(&env, 2, false, &[("write.delete.isolation-level", level)]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 5));
        let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
        let ids = mor.ids("id = 1");
        append(&env, &catalog, range(100, 102));
        let out = mor.run(vec![Task {
            deletes: ids,
            ..Task::default()
        }]);
        let mut want = expected(0..5);
        want.extend(expected(100..102));
        if level == "serializable" {
            let err = out.unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::KrishivIcebergCommitConflict);
            assert!(
                err.to_string()
                    .contains("Found conflicting files that can contain records matching true"),
                "{err}"
            );
            assert!(files_ending(&env, "-deletes.parquet").is_empty());
        } else {
            out.unwrap();
            want.retain(|r| r != "1|1|n1");
        }
        assert_eq!(read_back(&env, &catalog), want, "{level}");
    }
}

/// A foreign position delete: `UPDATE` fails at both levels ("Found new
/// conflicting delete files", `validateNoConflictingDeleteFiles`), `DELETE`
/// commits (it may delete a row deleted concurrently).
#[test]
fn a_foreign_delete_file_conflicts_with_update_not_delete() {
    for op in [RowLevelOperation::Update, RowLevelOperation::Delete] {
        for level in ["serializable", "snapshot"] {
            let env = env();
            let t = create(
                &env,
                2,
                false,
                &[
                    ("write.update.isolation-level", level),
                    ("write.delete.isolation-level", level),
                ],
            );
            let catalog = Catalog::new(&t);
            append(&env, &catalog, range(0, 5));
            let mor = Mor::open(&env, &catalog, op);
            let task = Task {
                deletes: mor.ids("id = 3"),
                inserts: if op == RowLevelOperation::Update {
                    mor.updated("id = 3", "'u'")
                } else {
                    Vec::new()
                },
            };
            add_deletes(&env, &catalog, |files| {
                vec![position_delete(&env, "foreign.parquet", &files[0], 0)]
            });
            let out = mor.run(vec![task]);
            if op == RowLevelOperation::Update {
                let err = out.unwrap_err();
                assert_eq!(err.error_class(), ErrorClass::KrishivIcebergCommitConflict);
                assert!(
                    err.to_string().contains(
                        "Found new conflicting delete files that can apply to records matching true"
                    ),
                    "{level}: {err}"
                );
                assert_eq!(read_back(&env, &catalog), expected(1..5));
            } else {
                out.unwrap();
                assert_eq!(read_back(&env, &catalog), expected([1, 2, 4]));
            }
        }
    }
}

/// A foreign overwrite removing the data file the deletes reference:
/// "Cannot commit, missing data files" (`validateDataFilesExist`; an
/// `overwrite` counts for `DELETE` too).
#[test]
fn a_foreign_removal_of_a_referenced_file_fails() {
    for op in [RowLevelOperation::Delete, RowLevelOperation::Update] {
        let env = env();
        // `snapshot`: the foreign file added is not itself a conflict.
        let t = create(
            &env,
            2,
            false,
            &[
                ("write.delete.isolation-level", "snapshot"),
                ("write.update.isolation-level", "snapshot"),
            ],
        );
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 5));
        let mor = Mor::open(&env, &catalog, op);
        let task = Task {
            deletes: mor.ids("id = 3"),
            inserts: if op == RowLevelOperation::Update {
                mor.updated("id = 3", "'u'")
            } else {
                Vec::new()
            },
        };
        foreign_overwrite(&env, &catalog);
        let err = mor.run(vec![task]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivIcebergCommitConflict,
            "{err}"
        );
        assert!(
            err.to_string()
                .contains("Cannot commit, missing data files"),
            "{op:?}: {err}"
        );
    }
}

/// Another writer's `overwrite`: every data file replaced by one file holding only row 0
/// (a data file the statement did not read).
fn foreign_overwrite(env: &Env, catalog: &Arc<Catalog>) {
    use super::append::{WriteIds, prepare_attempt, write_new_manifests};
    use super::data_file::{WrittenDataFile, data_file};
    let m = catalog.metadata();
    let io = env.reg.file_io();
    let files = super::append_tests::iceberg_rust_files(env, &m, &catalog.location());
    let path = env.at("foreign-data.parquet");
    let b = super::provider_tests::batch(
        vec![
            id_field("id", A::Int64, 1),
            id_field("x", A::Int32, 2),
            id_field("name", A::Utf8, 3),
        ],
        vec![
            Arc::new(Int64Array::from(vec![0_i64])),
            Arc::new(arrow_array::Int32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["n0"])),
        ],
    );
    let size = write_parquet(&path, &b);
    let written = WrittenDataFile {
        location: path,
        size,
        rows: 1,
        metrics: crate::plan::file_metrics::FileMetrics::default(),
        partition: Struct::empty(),
        spec_id: 0,
        split_offsets: None,
        sort_order_id: None,
    };
    let f = data_file(&written, m.current_schema()).unwrap();
    let request = env.rt.block_on(async {
        let mut new = write_new_manifests(&io, &m, vec![(0, f)], 777, "uuid-777", &mut Vec::new())
            .await
            .unwrap();
        new.removals = Some(super::filter::Removals {
            data_files: files.into_iter().collect(),
            ..super::filter::Removals::default()
        });
        let ids = WriteIds {
            branch: "main",
            write_id: "foreign",
            query_id: None,
        };
        prepare_attempt(&io, &m, &mut new, ids, m.last_updated_ms() + 1, MAX)
            .await
            .unwrap()
    });
    assert_eq!(request.snapshot.summary().operation, Operation::Overwrite);
    let out = env
        .rt
        .block_on(catalog.commit(request.requirements, request.updates));
    assert!(matches!(out, CommitOutcome::Committed(_)), "{out:?}");
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

/// A statement replayed with the key of one that already landed on the base it opened at
/// publishes nothing again.
#[test]
fn a_replayed_write_on_a_head_holding_it_commits_nothing() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 3));
    let first = Mor::open(&env, &catalog, RowLevelOperation::Update);
    let w = MergeOnReadWrite::new(&first.p, first.op, None, Some("replayed".to_owned())).unwrap();
    let outputs = first
        .write(
            &w,
            vec![Task {
                deletes: Vec::new(),
                inserts: vec![first.new_rows(10, 12)],
            }],
        )
        .unwrap();
    env.rt.block_on(w.commit(outputs, Some("q"))).unwrap();
    let snapshots = catalog.metadata().snapshots().count();
    let files = sorted(files_ending(&env, ""));
    let replay = Mor::open(&env, &catalog, RowLevelOperation::Update);
    let w = MergeOnReadWrite::new(&replay.p, replay.op, None, Some("replayed".to_owned())).unwrap();
    let outputs = replay
        .write(
            &w,
            vec![Task {
                deletes: Vec::new(),
                inserts: vec![replay.new_rows(10, 12)],
            }],
        )
        .unwrap();
    env.rt.block_on(w.commit(outputs, Some("q"))).unwrap();
    assert_eq!(catalog.metadata().snapshots().count(), snapshots);
    assert_eq!(read_back(&env, &catalog), expected((0..3).chain(10..12)));
    // The replay's own data files are deleted, not orphaned.
    assert_eq!(sorted(files_ending(&env, "")), files);
}

/// A v2 statement whose table is upgraded to v3 and moved on before it
/// commits: the rebase refuses its position delete files with Java's text.
#[test]
fn a_rebase_onto_v3_refuses_position_delete_files() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
    let ids = mor.ids("id = 2");
    upgrade(&env, &catalog);
    append(&env, &catalog, range(10, 12));
    let err = mor
        .run(vec![Task {
            deletes: ids,
            ..Task::default()
        }])
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{err}"
    );
    assert!(
        err.to_string()
            .contains("Must use DVs for position deletes in V3"),
        "{err}"
    );
    assert!(files_ending(&env, "-deletes.parquet").is_empty());
}

/// On v3 an `UPDATE`'s rows must carry the lineage columns.
#[test]
fn a_v3_update_without_lineage_is_refused() {
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 3));
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Update);
    let err = mor
        .run(vec![Task {
            deletes: Vec::new(),
            inserts: vec![range(5, 6)],
        }])
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{err}"
    );
    assert!(err.to_string().contains("row lineage"), "{err}");
}

/// A `DELETE` from a file the snapshot does not hold, or at a negative
/// position, is refused before anything is written.
#[test]
fn bad_row_ids_are_refused() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 3));
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
    let (file, _) = mor.ids("id = 1").remove(0);
    let err = mor
        .run(vec![Task {
            deletes: vec![(file, -1)],
            ..Task::default()
        }])
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivIcebergInvalidDeletes,
        "{err}"
    );
    let err = mor
        .run(vec![Task {
            deletes: vec![(b"nope".to_vec(), 1)],
            ..Task::default()
        }])
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError, "{err}");
    assert_eq!(catalog.metadata().snapshots().count(), 1);
}

/// Review 36c1 M4: a file-scoped position delete file P1 (Krishiv's, v2)
/// the statement absorbs is removed by a foreign `delete` snapshot that adds
/// no delete file. A `DELETE` skips the missing-absorbed check and commits
/// (Java's `RowDelta` without `validateDeletedFiles`), its new file still
/// deleting P1's row; an `UPDATE` fails with "Missing required files to
/// delete" (`Removals::ignore_missing_deletes`, both directions).
#[test]
fn a_concurrently_removed_absorbed_delete_fails_update_not_delete() {
    for op in [RowLevelOperation::Delete, RowLevelOperation::Update] {
        let env = env();
        let t = create(&env, 2, false, &[]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 10));
        Mor::open(&env, &catalog, RowLevelOperation::Delete)
            .delete("id = 1")
            .unwrap();
        assert_eq!(delete_entries(&env, &catalog.metadata()).len(), 1);
        let mor = Mor::open(&env, &catalog, op);
        let task = Task {
            deletes: mor.ids("id = 3"),
            inserts: if op == RowLevelOperation::Update {
                mor.updated("id = 3", "'u'")
            } else {
                Vec::new()
            },
        };
        remove_all_deletes(&env, &catalog);
        assert!(delete_entries(&env, &catalog.metadata()).is_empty());
        assert_eq!(read_back(&env, &catalog), expected(0..10));
        let out = mor.run(vec![task]);
        if op == RowLevelOperation::Delete {
            let m = out.unwrap();
            assert_eq!(summary(&m).0, Operation::Delete);
            // P1's row 1 is in the new file, absorbed.
            assert_eq!(read_back(&env, &catalog), without(0..10, &[1, 3]));
            live(&env, &m);
        } else {
            let err = out.unwrap_err();
            assert_eq!(
                err.error_class(),
                ErrorClass::KrishivIcebergCommitConflict,
                "{err}"
            );
            assert!(
                err.to_string().contains("Missing required files to delete"),
                "{err}"
            );
            assert_eq!(read_back(&env, &catalog), expected(0..10));
        }
    }
}

/// R36d m1: on v3 a finished task's output holds its positions until the commit returns, so the
/// pool keeps their charge while the outputs live (`write_in`'s `settle` is what charges a fresh
/// table's positions), while the commit reaches the catalog, and returns it afterwards. A v2 task
/// holds no positions after `finish`, so nothing stays charged.
#[test]
fn a_v3_output_keeps_its_positions_charged_through_the_commit() {
    for (version, held) in [(3, true), (2, false)] {
        let env = env();
        let t = create(
            &env,
            version,
            false,
            &[("write.delete.mode", "merge-on-read")],
        );
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 100));
        let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
        let w = mor.writer();
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let outputs = mor
            .write_in(
                &w,
                vec![Task {
                    deletes: mor.ids("true"),
                    inserts: Vec::new(),
                }],
                &pool,
            )
            .unwrap();
        assert_eq!(pool.reserved() > 0, held, "v{version} before the commit");
        let seen = Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        {
            let (seen, pool) = (Arc::clone(&seen), Arc::clone(&pool));
            *catalog.observe.lock().unwrap() = Some(Box::new(move || {
                seen.store(pool.reserved(), std::sync::atomic::Ordering::SeqCst);
            }));
        }
        env.rt.block_on(w.commit(outputs, Some("q"))).unwrap();
        let at_commit = seen.load(std::sync::atomic::Ordering::SeqCst);
        assert_ne!(at_commit, usize::MAX, "the commit reached the catalog");
        assert_eq!(at_commit > 0, held, "v{version} at the catalog");
        assert_eq!(pool.reserved(), 0, "v{version} after the commit");
    }
}
