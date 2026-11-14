// SPDX-License-Identifier: Apache-2.0
//! Tests of the merge-on-read delete writers (`delete_write`, PR-36c unit 1):
//! deletion vectors read back through the scan's delete loader and
//! iceberg-rust's `PuffinReader` (offsets, properties), merged with an old DV
//! and v2 position deletes (only file-scoped ones superseded); position
//! delete files sorted, with Java's metrics per granularity, merged with
//! file-scoped previous deletes, rolled at the target size; every delete
//! file through iceberg-rust's `ManifestWriterBuilder` and the scan's
//! manifest parser; refusals; an injected upload failure leaving no file.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use iceberg::puffin::PuffinReader;
use iceberg::spec::{
    DataFile, FormatVersion, Literal, ManifestWriterBuilder, NestedField, PartitionSpec,
    PrimitiveType, Schema, SortOrder, Struct, TableMetadata, TableMetadataBuilder, Transform, Type,
};
use krishiv_common::error::KrishivError;
use krishiv_exec::FileSelections;
use krishiv_io::{FileRef, RowSelection, VersionToken};
use krishiv_memory::{ConsumerHandle, MemoryConfig, MemoryPool, Reservation, TreePool};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use roaring::RoaringTreemap;

use super::delete_write::{
    CARDINALITY_KEY, DELETE_FORMAT, DELETE_GRANULARITY, DELETE_PARQUET_CODEC,
    DELETE_TARGET_FILE_SIZE, DeleteFileWriter, DeleteFormat, DeleteWriteResult, Granularity,
    PositionDeletes, Previous, REFERENCED_DATA_FILE_KEY, ROW_POSITION_ID, delete_file_name,
    is_file_scoped, load_previous,
};
use super::deletes;
use super::provider_tests::{Env, env, env_on};
use super::scan::parse_manifest;
use super::write::TaskIds;
use crate::plan::{
    ContentFile, ContentKind, DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID, DataFileEntry,
    DeleteFileEntry, DeleteKind, EntryStatus, FileFormatKind, TaskDeletes,
};

/// id long (1), x int (2).
fn schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "x", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap()
}

fn spec() -> PartitionSpec {
    PartitionSpec::builder(schema())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .unwrap()
        .build()
        .unwrap()
}

/// A table at `<dir>/t` partitioned by `identity(x)`, of format `version`.
fn table(env: &Env, version: FormatVersion, props: &[(&str, &str)]) -> TableMetadata {
    TableMetadataBuilder::new(
        schema(),
        spec(),
        SortOrder::unsorted_order(),
        env.at("t"),
        version,
        props
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
    )
    .unwrap()
    .build()
    .unwrap()
    .metadata
}

fn part(x: i32) -> Struct {
    Struct::from_iter([Some(Literal::int(x))])
}

fn pool() -> Arc<dyn MemoryPool> {
    TreePool::new(&MemoryConfig::default())
}

fn writer(env: &Env, m: &TableMetadata, pool: &Arc<dyn MemoryPool>) -> DeleteFileWriter {
    let id = pool.register("deletes", None);
    DeleteFileWriter::new(
        m,
        env.reg.registered().unwrap(),
        env.reg.file_io(),
        TaskIds {
            partition_id: 3,
            attempt_id: 7,
            write_id: "w1",
        },
        Reservation::new(Arc::clone(pool), id),
    )
    .unwrap()
}

/// A data file path of the table, in partition `x`.
fn data_path(env: &Env, x: i32, name: &str) -> String {
    env.at(&format!("t/data/x={x}/{name}.parquet"))
}

fn write(
    env: &Env,
    m: &TableMetadata,
    deletes: &PositionDeletes,
    previous: &HashMap<String, Previous>,
) -> Result<DeleteWriteResult, KrishivError> {
    let pool = pool();
    let w = writer(env, m, &pool);
    let r = env.rt.block_on(w.write(deletes, previous));
    assert_eq!(pool.reserved(), 0);
    r
}

fn locate(env: &Env) -> impl Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync {
    let paths = env.reg.registered().unwrap().paths;
    move |path: &str| {
        let at = paths.locate(path)?;
        Ok(FileRef {
            store: at.store,
            path: at.path,
            size: 0,
            version: VersionToken::Immutable,
            modified_micros: None,
        })
    }
}

/// `files` (of spec 0) through a manifest written by iceberg-rust and
/// parsed back by the scan's parser, as live delete entries of sequence
/// number `seq`.
fn through_manifest(
    env: &Env,
    version: FormatVersion,
    files: &[DataFile],
    seq: i64,
) -> Vec<Arc<DeleteFileEntry>> {
    let io = env.reg.file_io();
    let path = env.at(&format!("m-{seq}.avro"));
    let bytes = env.rt.block_on(async {
        let b = ManifestWriterBuilder::new(
            io.new_output(&path).unwrap(),
            Some(seq),
            Arc::new(schema()),
            spec(),
        );
        let mut w = match version {
            FormatVersion::V3 => b.build_v3_deletes(),
            _ => b.build_v2_deletes(),
        };
        for f in files {
            w.add_file(f.clone(), seq).unwrap();
        }
        w.write_manifest_file().await.unwrap();
        io.new_input(&path).unwrap().read().await.unwrap()
    });
    parse_manifest(&bytes, 0)
        .unwrap()
        .into_iter()
        .map(|e| {
            let kind = if e.file.format == FileFormatKind::Puffin {
                DeleteKind::DeletionVector
            } else {
                DeleteKind::PositionDelete
            };
            Arc::new(DeleteFileEntry {
                entry: DataFileEntry {
                    file: e.file,
                    status: e.status,
                    snapshot_id: seq,
                    data_sequence_number: Some(seq),
                    file_sequence_number: Some(seq),
                },
                kind,
            })
        })
        .collect()
}

fn data_entry(path: &str, x: i32) -> DataFileEntry {
    DataFileEntry {
        file: ContentFile {
            content: ContentKind::Data,
            path: path.to_owned(),
            format: FileFormatKind::Parquet,
            spec_id: 0,
            partition: vec![crate::plan::PartitionValue::Value(x.to_le_bytes().to_vec())],
            record_count: 100_000,
            size: 1,
            column_sizes: BTreeMap::new(),
            value_counts: BTreeMap::new(),
            null_counts: BTreeMap::new(),
            nan_counts: BTreeMap::new(),
            lower_bounds: BTreeMap::new(),
            upper_bounds: BTreeMap::new(),
            split_offsets: None,
            sort_order_id: None,
            first_row_id: None,
            equality_ids: None,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
        },
        status: EntryStatus::Added,
        snapshot_id: 1,
        data_sequence_number: Some(1),
        file_sequence_number: Some(1),
    }
}

/// The selection the scan's delete loader builds for the data file `path`
/// from `task`.
fn selection(env: &Env, path: &str, task: &TaskDeletes) -> RowSelection {
    let locate = locate(env);
    let data = data_entry(path, 0);
    let file = locate(path).unwrap();
    let mem = ConsumerHandle::register(pool(), "read", None);
    let (s, _held) = env
        .rt
        .block_on(deletes::load(
            &env.reg.registered().unwrap().io,
            &locate,
            &mem,
            &[(&data, &file, task)],
        ))
        .unwrap();
    s.selection(&file).unwrap()
}

fn excluding(positions: impl IntoIterator<Item = u64>) -> RowSelection {
    RowSelection::excluding(u64::MAX, positions)
}

/// The previous deletes of `path` among `task` for a write of `format`.
fn previous(env: &Env, format: DeleteFormat, path: &str, task: &TaskDeletes) -> Option<Previous> {
    let locate = locate(env);
    let mem = ConsumerHandle::register(pool(), "previous", None);
    env.rt
        .block_on(load_previous(
            &env.reg.registered().unwrap().io,
            &locate,
            &mem,
            format,
            path,
            task,
        ))
        .unwrap()
}

/// Every file under `dir` with extension `ext`.
fn files_with(dir: &std::path::Path, ext: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(files_with(&p, ext));
        } else if p.extension().is_some_and(|x| x == ext) {
            out.push(p.display().to_string());
        }
    }
    out.sort();
    out
}

/// The rows (`file_path`, `pos`) of the position delete file at `location`.
fn rows(location: &str) -> Vec<(String, i64)> {
    let file = std::fs::File::open(location.strip_prefix("file://").unwrap()).unwrap();
    let r = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let ids: Vec<_> = r
        .parquet_schema()
        .columns()
        .iter()
        .map(|c| c.self_type().get_basic_info().id())
        .collect();
    assert_eq!(ids, vec![DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID]);
    let batches: Vec<RecordBatch> = r.build().unwrap().map(Result::unwrap).collect();
    let mut out = Vec::new();
    for b in batches {
        assert!(!b.schema().field(0).is_nullable() && !b.schema().field(1).is_nullable());
        let p = b.column(0).as_string::<i32>();
        let n = b.column(1).as_primitive::<Int64Type>();
        for i in 0..b.num_rows() {
            out.push((p.value(i).to_owned(), n.value(i)));
        }
    }
    out
}

fn treemap(p: &[u64]) -> RoaringTreemap {
    p.iter().copied().collect()
}

#[test]
fn file_names_as_javas_output_file_factory_with_the_deletes_suffix() {
    assert_eq!(
        delete_file_name(3, 7, "w1", 1, "puffin"),
        "00003-7-w1-00001-deletes.puffin"
    );
    assert_eq!(
        delete_file_name(12, 99, "q", 123_456, "parquet"),
        "00012-99-q-123456-deletes.parquet"
    );
}

#[test]
fn one_puffin_with_a_dv_per_data_file_read_back() {
    let env = env();
    let m = table(&env, FormatVersion::V3, &[]);
    let (a, b, c) = (
        data_path(&env, 1, "a"),
        data_path(&env, 1, "b"),
        data_path(&env, 2, "c"),
    );
    let mut d = PositionDeletes::default();
    d.delete(&b, 5, 0, &part(1)).unwrap();
    d.delete(&a, 3, 0, &part(1)).unwrap();
    d.delete(&a, 1, 0, &part(1)).unwrap();
    d.delete_all(&c, &treemap(&[0, 1 << 33, 9]), 0, &part(2))
        .unwrap();
    let r = write(&env, &m, &d, &HashMap::new()).unwrap();
    assert!(r.rewritten.is_empty());
    assert_eq!(
        r.referenced_data_files.iter().cloned().collect::<Vec<_>>(),
        vec![a.clone(), b.clone(), c.clone()]
    );
    // One Puffin, where an unpartitioned data file goes.
    let puffin = env.at("t/data/00003-7-w1-00001-deletes.puffin");
    assert_eq!(
        files_with(env.dir.path(), "puffin"),
        vec![puffin.strip_prefix("file://").unwrap().to_owned()]
    );
    assert_eq!(r.delete_files.len(), 3);
    let io = env.reg.file_io();
    let size = std::fs::metadata(puffin.strip_prefix("file://").unwrap())
        .unwrap()
        .len();
    let reader = env
        .rt
        .block_on(PuffinReader::new(io.new_input(&puffin).unwrap()))
        .unwrap();
    let meta = env.rt.block_on(reader.file_metadata()).unwrap().clone();
    assert!(
        meta.properties()["created-by"].starts_with("Krishiv "),
        "{:?}",
        meta.properties()
    );
    assert_eq!(meta.blobs().len(), 3);
    let expected = [(&a, part(1), 2), (&b, part(1), 1), (&c, part(2), 3)];
    for ((f, blob), (path, p, n)) in r.delete_files.iter().zip(meta.blobs()).zip(expected) {
        assert_eq!(f.file_path(), puffin);
        assert_eq!(f.file_size_in_bytes(), size);
        assert_eq!(f.file_format(), iceberg::spec::DataFileFormat::Puffin);
        assert_eq!(
            f.content_type(),
            iceberg::spec::DataContentType::PositionDeletes
        );
        assert_eq!(f.referenced_data_file().as_deref(), Some(path.as_str()));
        assert_eq!(f.partition(), &p);
        assert_eq!(f.record_count(), n);
        assert!(f.lower_bounds().is_empty() && f.value_counts().is_empty());
        assert_eq!(
            f.content_offset(),
            Some(i64::try_from(blob.offset()).unwrap())
        );
        assert_eq!(
            f.content_size_in_bytes(),
            Some(i64::try_from(blob.length()).unwrap())
        );
        assert_eq!(blob.blob_type(), "deletion-vector-v1");
        assert_eq!(blob.fields(), &[ROW_POSITION_ID]);
        assert_eq!((blob.snapshot_id(), blob.sequence_number()), (-1, -1));
        assert_eq!(
            blob.compression_codec(),
            iceberg::puffin::CompressionCodec::None
        );
        assert_eq!(blob.properties()[REFERENCED_DATA_FILE_KEY], *path);
        assert_eq!(blob.properties()[CARDINALITY_KEY], n.to_string());
    }
    // The first blob follows the magic; the rest back to back.
    assert_eq!(meta.blobs()[0].offset(), 4);
    assert_eq!(
        meta.blobs()[1].offset(),
        meta.blobs()[0].offset() + meta.blobs()[0].length()
    );
    // Read back through the manifest and the scan's loader.
    let entries = through_manifest(&env, FormatVersion::V3, &r.delete_files, 2);
    for (e, (path, want)) in
        entries
            .iter()
            .zip([(&a, vec![1, 3]), (&b, vec![5]), (&c, vec![0, 9, 1 << 33])])
    {
        assert_eq!(
            e.entry.file.referenced_data_file.as_deref(),
            Some(path.as_str())
        );
        assert!(is_file_scoped(e));
        let task = TaskDeletes {
            deletion_vector: Some(Arc::clone(e)),
            ..TaskDeletes::default()
        };
        assert_eq!(selection(&env, path, &task), excluding(want));
    }
}

#[test]
fn no_deletes_write_no_file() {
    let env = env();
    for version in [FormatVersion::V2, FormatVersion::V3] {
        let m = table(&env, version, &[]);
        let r = write(&env, &m, &PositionDeletes::default(), &HashMap::new()).unwrap();
        assert!(r.delete_files.is_empty() && r.referenced_data_files.is_empty());
    }
    assert!(files_with(env.dir.path(), "puffin").is_empty());
    assert!(files_with(env.dir.path(), "parquet").is_empty());
}

/// v2 position delete files of `rows` written by Krishiv's own writer with
/// `granularity`, as delete entries of sequence number `seq`.
fn v2_files(
    env: &Env,
    granularity: &str,
    rows: &[(&str, u64)],
    seq: i64,
) -> Vec<Arc<DeleteFileEntry>> {
    let m = table(env, FormatVersion::V2, &[(DELETE_GRANULARITY, granularity)]);
    let mut d = PositionDeletes::default();
    for (path, pos) in rows {
        d.delete(path, *pos, 0, &part(1)).unwrap();
    }
    let pool = pool();
    let id = pool.register("deletes", None);
    let w = DeleteFileWriter::new(
        &m,
        env.reg.registered().unwrap(),
        env.reg.file_io(),
        TaskIds {
            partition_id: 1,
            attempt_id: u64::try_from(seq).unwrap(),
            write_id: "old",
        },
        Reservation::new(Arc::clone(&pool), id),
    )
    .unwrap();
    let r = env.rt.block_on(w.write(&d, &HashMap::new())).unwrap();
    through_manifest(env, FormatVersion::V2, &r.delete_files, seq)
}

/// A v3 write unions the new positions with the file's old DV and every v2
/// position delete that applies to it; it supersedes the DV and the
/// file-scoped position delete file, never the partition-scoped one. The
/// input is synthetic (a scan never assigns position deletes next to a DV,
/// `TaskDeletes`); it checks the union and rewritten rules only. The real
/// upgrade case is `the_first_dv_after_an_upgrade_absorbs_v2_position_deletes`.
#[test]
fn a_dv_absorbs_the_old_dv_and_v2_position_deletes() {
    let env = env();
    let (a, other) = (data_path(&env, 1, "a"), data_path(&env, 1, "other"));
    // A partition-scoped file (two data paths) and a file-scoped one.
    let shared = v2_files(&env, "partition", &[(&a, 10), (&a, 11), (&other, 2)], 1);
    assert_eq!(shared.len(), 1);
    assert!(!is_file_scoped(&shared[0]));
    let scoped = v2_files(&env, "file", &[(&a, 20)], 2);
    assert!(is_file_scoped(&scoped[0]));
    assert_eq!(scoped[0].entry.file.referenced_data_file, None);
    // The old DV, written by the DV writer.
    let m = table(&env, FormatVersion::V3, &[]);
    let mut d = PositionDeletes::default();
    d.delete(&a, 30, 0, &part(1)).unwrap();
    let old = write(&env, &m, &d, &HashMap::new()).unwrap();
    let old = through_manifest(&env, FormatVersion::V3, &old.delete_files, 3);
    let task = TaskDeletes {
        deletion_vector: Some(Arc::clone(&old[0])),
        position_deletes: vec![Arc::clone(&shared[0]), Arc::clone(&scoped[0])],
        equality_deletes: Vec::new(),
    };
    let prev = previous(&env, DeleteFormat::DeletionVectors, &a, &task).unwrap();
    assert_eq!(prev.positions, treemap(&[10, 11, 20, 30]));
    assert_eq!(prev.files.len(), 3);

    let mut d = PositionDeletes::default();
    d.delete(&a, 40, 0, &part(1)).unwrap();
    let r = write(&env, &m, &d, &HashMap::from([(a.clone(), prev)])).unwrap();
    let rewritten: Vec<_> = r
        .rewritten
        .iter()
        .map(|f| f.entry.file.path.clone())
        .collect();
    assert_eq!(
        rewritten,
        vec![
            old[0].entry.file.path.clone(),
            scoped[0].entry.file.path.clone()
        ]
    );
    assert_eq!(r.delete_files[0].record_count(), 5);
    let new = through_manifest(&env, FormatVersion::V3, &r.delete_files, 4);
    let task = TaskDeletes {
        deletion_vector: Some(Arc::clone(&new[0])),
        ..TaskDeletes::default()
    };
    assert_eq!(selection(&env, &a, &task), excluding([10, 11, 20, 30, 40]));
}

/// The v2 → v3 upgrade case (design note §8 item 4, review S4): the data
/// file has no DV yet, a partition-scoped and a file-scoped v2 position
/// delete file apply to it (assigned by `DeleteIndex`, as a scan assigns
/// them). The first DV holds both files' positions and the new ones; only
/// the file-scoped file is superseded.
#[test]
fn the_first_dv_after_an_upgrade_absorbs_v2_position_deletes() {
    let env = env();
    let (a, other) = (data_path(&env, 1, "a"), data_path(&env, 1, "other"));
    let shared = v2_files(&env, "partition", &[(&a, 10), (&a, 11), (&other, 2)], 2);
    let scoped = v2_files(&env, "file", &[(&a, 20)], 3);
    let m = table(&env, FormatVersion::V3, &[]);
    let specs = super::scan::partition_specs(&m, m.current_schema());
    let index =
        crate::plan::DeleteIndex::new(&[Arc::clone(&shared[0]), Arc::clone(&scoped[0])], &specs)
            .unwrap();
    let task = index.for_data_file(&data_entry(&a, 1)).unwrap();
    assert!(task.deletion_vector.is_none());
    assert_eq!(task.position_deletes.len(), 2);

    let prev = previous(&env, DeleteFormat::DeletionVectors, &a, &task).unwrap();
    assert_eq!(prev.positions, treemap(&[10, 11, 20]));
    let mut d = PositionDeletes::default();
    d.delete(&a, 40, 0, &part(1)).unwrap();
    let r = write(&env, &m, &d, &HashMap::from([(a.clone(), prev)])).unwrap();
    let rewritten: Vec<_> = r
        .rewritten
        .iter()
        .map(|f| f.entry.file.path.clone())
        .collect();
    assert_eq!(rewritten, vec![scoped[0].entry.file.path.clone()]);
    let new = through_manifest(&env, FormatVersion::V3, &r.delete_files, 4);
    assert_eq!(new[0].entry.file.record_count, 4);
    let task = TaskDeletes {
        deletion_vector: Some(Arc::clone(&new[0])),
        ..TaskDeletes::default()
    };
    assert_eq!(selection(&env, &a, &task), excluding([10, 11, 20, 40]));
}

/// Granularity `file`: one delete file per data file, rows sorted, both
/// columns' bounds kept (equal `file_path` bounds: file-scoped), no counts,
/// no `referenced_data_file`; read back through the manifest and the loader.
#[test]
fn file_granularity_writes_a_file_scoped_delete_file_per_data_file() {
    let env = env();
    let m = table(&env, FormatVersion::V2, &[]);
    assert_eq!(
        DeleteFormat::for_table(&m).unwrap(),
        DeleteFormat::PositionFiles(Granularity::File)
    );
    assert_eq!(
        writer(&env, &m, &pool()).format(),
        DeleteFormat::PositionFiles(Granularity::File)
    );
    let (a, b, c) = (
        data_path(&env, 1, "a"),
        data_path(&env, 1, "b"),
        data_path(&env, 2, "c"),
    );
    let mut d = PositionDeletes::default();
    for (p, pos, x) in [(&b, 7, 1), (&a, 9, 1), (&c, 4, 2), (&a, 2, 1), (&b, 1, 1)] {
        d.delete(p, pos, 0, &part(x)).unwrap();
    }
    let r = write(&env, &m, &d, &HashMap::new()).unwrap();
    assert_eq!(r.delete_files.len(), 3);
    let want = [
        (
            &a,
            1,
            "t/data/x=1/00003-7-w1-00001-deletes.parquet",
            vec![2, 9],
        ),
        (
            &b,
            1,
            "t/data/x=1/00003-7-w1-00002-deletes.parquet",
            vec![1, 7],
        ),
        (
            &c,
            2,
            "t/data/x=2/00003-7-w1-00003-deletes.parquet",
            vec![4],
        ),
    ];
    for (f, (path, x, name, pos)) in r.delete_files.iter().zip(&want) {
        assert_eq!(f.file_path(), env.at(name));
        assert_eq!(f.file_format(), iceberg::spec::DataFileFormat::Parquet);
        assert_eq!(f.partition(), &part(*x));
        assert_eq!(f.record_count(), pos.len() as u64);
        assert_eq!(f.referenced_data_file(), None);
        assert!(f.value_counts().is_empty() && f.null_value_counts().is_empty());
        assert_eq!(
            rows(f.file_path()),
            pos.iter()
                .map(|p| ((*path).clone(), *p))
                .collect::<Vec<_>>()
        );
    }
    let entries = through_manifest(&env, FormatVersion::V2, &r.delete_files, 2);
    for (e, (path, _, _, pos)) in entries.iter().zip(&want) {
        let f = &e.entry.file;
        assert_eq!(f.lower_bounds[&DELETE_FILE_PATH_ID], path.as_bytes());
        assert_eq!(f.upper_bounds[&DELETE_FILE_PATH_ID], path.as_bytes());
        assert_eq!(
            f.lower_bounds[&DELETE_FILE_POS_ID],
            pos[0].to_le_bytes().to_vec()
        );
        assert_eq!(
            f.upper_bounds[&DELETE_FILE_POS_ID],
            pos[pos.len() - 1].to_le_bytes().to_vec()
        );
        assert!(is_file_scoped(e));
        let task = TaskDeletes {
            position_deletes: vec![Arc::clone(e)],
            ..TaskDeletes::default()
        };
        assert_eq!(
            selection(&env, path, &task),
            excluding(pos.iter().map(|p| u64::try_from(*p).unwrap()))
        );
    }
}

/// Granularity `partition`: one file per partition with its paths sorted;
/// a file naming several paths keeps no counts and no bounds.
#[test]
fn partition_granularity_writes_a_file_per_partition() {
    let env = env();
    let m = table(
        &env,
        FormatVersion::V2,
        &[(DELETE_GRANULARITY, "PARTITION")],
    );
    let (a, b, c) = (
        data_path(&env, 1, "a"),
        data_path(&env, 1, "b"),
        data_path(&env, 2, "c"),
    );
    let mut d = PositionDeletes::default();
    for (p, pos, x) in [(&b, 7, 1), (&a, 9, 1), (&c, 4, 2), (&a, 2, 1)] {
        d.delete(p, pos, 0, &part(x)).unwrap();
    }
    let r = write(&env, &m, &d, &HashMap::new()).unwrap();
    assert_eq!(r.delete_files.len(), 2);
    let first = &r.delete_files[0];
    assert_eq!(
        rows(first.file_path()),
        vec![(a.clone(), 2), (a.clone(), 9), (b.clone(), 7)]
    );
    assert!(first.lower_bounds().is_empty() && first.upper_bounds().is_empty());
    assert_eq!(first.record_count(), 3);
    // The second names one path: its bounds are kept.
    assert_eq!(r.delete_files[1].lower_bounds().len(), 2);
    let entries = through_manifest(&env, FormatVersion::V2, &r.delete_files, 2);
    assert!(!is_file_scoped(&entries[0]));
    let task = TaskDeletes {
        position_deletes: vec![Arc::clone(&entries[0])],
        ..TaskDeletes::default()
    };
    assert_eq!(selection(&env, &a, &task), excluding([2, 9]));
    assert_eq!(selection(&env, &b, &task), excluding([7]));
}

/// v2 `file`: only file-scoped previous files are loaded, merged and
/// superseded; v2 `partition` loads none.
#[test]
fn v2_merges_only_file_scoped_previous_deletes() {
    let env = env();
    let (a, other) = (data_path(&env, 1, "a"), data_path(&env, 1, "other"));
    let shared = v2_files(&env, "partition", &[(&a, 10), (&other, 2)], 1);
    let scoped = v2_files(&env, "file", &[(&a, 20), (&a, 21)], 2);
    let task = TaskDeletes {
        position_deletes: vec![Arc::clone(&shared[0]), Arc::clone(&scoped[0])],
        ..TaskDeletes::default()
    };
    let file = DeleteFormat::PositionFiles(Granularity::File);
    assert!(
        previous(
            &env,
            DeleteFormat::PositionFiles(Granularity::Partition),
            &a,
            &task
        )
        .is_none()
    );
    let prev = previous(&env, file, &a, &task).unwrap();
    assert_eq!(prev.positions, treemap(&[20, 21]));
    assert_eq!(prev.files.len(), 1);

    let m = table(&env, FormatVersion::V2, &[]);
    let mut d = PositionDeletes::default();
    d.delete(&a, 5, 0, &part(1)).unwrap();
    let r = write(&env, &m, &d, &HashMap::from([(a.clone(), prev)])).unwrap();
    assert_eq!(r.rewritten.len(), 1);
    assert_eq!(r.rewritten[0].entry.file.path, scoped[0].entry.file.path);
    assert_eq!(
        rows(r.delete_files[0].file_path()),
        vec![(a.clone(), 5), (a.clone(), 20), (a.clone(), 21)]
    );
}

/// Files roll at a multiple of 1,000 rows once past the target size.
#[test]
fn position_delete_files_roll_at_the_target_size() {
    let env = env();
    let m = table(&env, FormatVersion::V2, &[(DELETE_TARGET_FILE_SIZE, "1")]);
    let a = data_path(&env, 1, "a");
    let mut d = PositionDeletes::default();
    d.delete_all(&a, &(0..2_500).collect(), 0, &part(1))
        .unwrap();
    let r = write(&env, &m, &d, &HashMap::new()).unwrap();
    let counts: Vec<u64> = r.delete_files.iter().map(DataFile::record_count).collect();
    assert_eq!(counts, vec![1_000, 1_000, 500]);
    let second = rows(r.delete_files[1].file_path());
    assert_eq!(second.first().map(|r| r.1), Some(1_000));
    let entries = through_manifest(&env, FormatVersion::V2, &r.delete_files, 2);
    assert_eq!(
        entries[2].entry.file.lower_bounds[&DELETE_FILE_POS_ID],
        2_000i64.to_le_bytes().to_vec()
    );
}

#[test]
fn delete_codec_falls_back_to_the_data_codec() {
    let env = env();
    let codec = |props: &[(&str, &str)]| {
        let m = table(&env, FormatVersion::V2, props);
        let mut d = PositionDeletes::default();
        d.delete(&data_path(&env, 1, "a"), 1, 0, &part(1)).unwrap();
        let r = write(&env, &m, &d, &HashMap::new()).unwrap();
        let path = r.delete_files[0].file_path().to_owned();
        let file = std::fs::File::open(path.strip_prefix("file://").unwrap()).unwrap();
        let reader = parquet::file::reader::SerializedFileReader::new(file).unwrap();
        let meta = parquet::file::reader::FileReader::metadata(&reader).clone();
        std::fs::remove_file(path.strip_prefix("file://").unwrap()).unwrap();
        format!("{:?}", meta.row_group(0).column(0).compression())
    };
    assert!(codec(&[]).starts_with("GZIP"), "{}", codec(&[]));
    assert_eq!(
        codec(&[("write.parquet.compression-codec", "snappy")]),
        "SNAPPY"
    );
    assert_eq!(
        codec(&[
            ("write.parquet.compression-codec", "snappy"),
            (DELETE_PARQUET_CODEC, "uncompressed")
        ]),
        "UNCOMPRESSED"
    );
}

#[test]
fn refusals() {
    let env = env();
    let class = |r: Result<DeleteFormat, KrishivError>| {
        let e = r.unwrap_err();
        (e.error_class().name().to_owned(), e.to_string())
    };
    let (c, _) = class(DeleteFormat::for_table(&table(
        &env,
        FormatVersion::V1,
        &[],
    )));
    assert_eq!(c, "KRISHIV_UNSUPPORTED_FEATURE");
    for version in [FormatVersion::V2, FormatVersion::V3] {
        let (c, text) = class(DeleteFormat::for_table(&table(
            &env,
            version,
            &[(DELETE_GRANULARITY, "row")],
        )));
        assert_eq!(c, "KRISHIV_UNSUPPORTED_FEATURE");
        assert!(text.contains("Unknown delete granularity: row"), "{text}");
    }
    let (c, _) = class(DeleteFormat::for_table(&table(
        &env,
        FormatVersion::V2,
        &[(DELETE_FORMAT, "avro")],
    )));
    assert_eq!(c, "KRISHIV_UNSUPPORTED_FEATURE");
    let (c, _) = class(DeleteFormat::for_table(&table(
        &env,
        FormatVersion::V2,
        &[("write.format.default", "orc")],
    )));
    assert_eq!(c, "KRISHIV_UNSUPPORTED_FEATURE");
    // v3 writes DVs whatever the delete format says.
    assert_eq!(
        DeleteFormat::for_table(&table(&env, FormatVersion::V3, &[(DELETE_FORMAT, "avro")]))
            .unwrap(),
        DeleteFormat::DeletionVectors
    );
    // One data path given two partitions is a bug.
    let mut d = PositionDeletes::default();
    d.delete("p", 1, 0, &part(1)).unwrap();
    let e = d.delete("p", 2, 0, &part(2)).unwrap_err();
    assert_eq!(e.error_class().name(), "INTERNAL_ERROR");
    assert!(d.delete_all("p", &treemap(&[3]), 1, &part(1)).is_err());
    assert_eq!(d.paths().collect::<Vec<_>>(), vec!["p"]);
}

/// A failed upload (the second delete file's, or the Puffin's) fails the
/// write and leaves no file behind.
#[test]
fn an_upload_failure_leaves_no_file() {
    for (version, refused) in [
        (FormatVersion::V2, "-00002-deletes.parquet"),
        (FormatVersion::V3, "-deletes.puffin"),
    ] {
        let store = Arc::new(Faulty::new(refused, Fault::Refuse));
        let env = env_on(store, krishiv_io::IoConfig::default());
        let m = table(&env, version, &[]);
        let mut d = PositionDeletes::default();
        for name in ["a", "b", "c"] {
            d.delete(&data_path(&env, 1, name), 1, 0, &part(1)).unwrap();
        }
        let e = write(&env, &m, &d, &HashMap::new()).unwrap_err();
        let text = format!("{e} {e:?}");
        assert!(
            text.contains("refused by the test") || text.contains(refused),
            "{text}"
        );
        assert!(files_with(env.dir.path(), "parquet").is_empty());
        assert!(files_with(env.dir.path(), "puffin").is_empty());
    }
}

/// A Puffin that landed although its upload answered with an error (a lost
/// answer) is deleted by the failed write (review S3).
#[test]
fn a_v3_failure_after_the_puffin_landed_deletes_it() {
    let store = Arc::new(Faulty::new("-deletes.puffin", Fault::LandThenFail));
    let env = env_on(store, krishiv_io::IoConfig::default());
    let m = table(&env, FormatVersion::V3, &[]);
    let mut d = PositionDeletes::default();
    d.delete(&data_path(&env, 1, "a"), 1, 0, &part(1)).unwrap();
    let e = write(&env, &m, &d, &HashMap::new()).unwrap_err();
    assert!(format!("{e}").contains("-deletes.puffin"), "{e}");
    assert!(files_with(env.dir.path(), "puffin").is_empty());
}

/// Runs `w.write(d)` as a task until the store's targeted upload starts
/// (and blocks), samples `pool`'s charge, then aborts the task (a dropped
/// write future) and returns the sample.
fn charge_then_drop(
    env: &Env,
    started: &std::sync::atomic::AtomicBool,
    pool: &Arc<dyn MemoryPool>,
    w: DeleteFileWriter,
    d: PositionDeletes,
) -> usize {
    let task = env
        .rt
        .spawn(async move { w.write(&d, &HashMap::new()).await.map(drop) });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !started.load(std::sync::atomic::Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < deadline,
            "the upload never started"
        );
        assert!(!task.is_finished(), "the write ended before the upload");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let charged = pool.reserved();
    task.abort();
    let _ = env.rt.block_on(task);
    charged
}

/// Waits until no `.parquet` or `.puffin` file is left under the table.
fn wait_for_no_files(env: &Env) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let left = [
            files_with(env.dir.path(), "parquet"),
            files_with(env.dir.path(), "puffin"),
        ]
        .concat();
        if left.is_empty() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "left behind: {left:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// A write future dropped while its second delete file uploads deletes the
/// first (already finished) file in the background and releases its memory
/// (review S3).
#[test]
fn a_dropped_write_deletes_its_files() {
    let store = Arc::new(Faulty::new("-00002-deletes.parquet", Fault::Block));
    let started = Arc::clone(&store.started);
    let env = env_on(store, krishiv_io::IoConfig::default());
    let m = table(&env, FormatVersion::V2, &[]);
    let mut d = PositionDeletes::default();
    for name in ["a", "b", "c"] {
        d.delete(&data_path(&env, 1, name), 1, 0, &part(1)).unwrap();
    }
    let pool = pool();
    let w = writer(&env, &m, &pool);
    charge_then_drop(&env, &started, &pool, w, d);
    wait_for_no_files(&env);
    assert_eq!(pool.reserved(), 0);
}

/// The Puffin is buffered whole until `close` uploads it: its blobs stay
/// charged until then (review S1), and a write dropped during that upload
/// releases the charge.
#[test]
fn puffin_blobs_stay_charged_until_close() {
    let store = Arc::new(Faulty::new("-deletes.puffin", Fault::Block));
    let started = Arc::clone(&store.started);
    let env = env_on(store, krishiv_io::IoConfig::default());
    let m = table(&env, FormatVersion::V3, &[]);
    let mut d = PositionDeletes::default();
    for name in ["a", "b", "c"] {
        d.delete_all(
            &data_path(&env, 1, name),
            &(0..5_000).step_by(3).collect(),
            0,
            &part(1),
        )
        .unwrap();
    }
    let blobs: usize = d
        .paths()
        .map(|_| {
            crate::plan::encode_dv(&(0..5_000).step_by(3).collect())
                .unwrap()
                .len()
        })
        .sum();
    let pool = pool();
    let w = writer(&env, &m, &pool);
    let charged = charge_then_drop(&env, &started, &pool, w, d);
    assert!(
        charged >= blobs,
        "{charged} charged while the Puffin uploads, {blobs} buffered"
    );
    wait_for_no_files(&env);
    assert_eq!(pool.reserved(), 0);
}

/// With `write.object-storage.enabled` the Puffin gets the hash directories
/// of an unpartitioned data file of its name (review N6).
#[test]
fn the_puffin_goes_where_an_unpartitioned_data_file_goes() {
    let env = env();
    let m = table(
        &env,
        FormatVersion::V3,
        &[("write.object-storage.enabled", "true")],
    );
    let mut d = PositionDeletes::default();
    d.delete(&data_path(&env, 1, "a"), 1, 0, &part(1)).unwrap();
    let r = write(&env, &m, &d, &HashMap::new()).unwrap();
    let name = "00003-7-w1-00001-deletes.puffin";
    let want = super::data_file::DataLocations::for_table(&m)
        .unwrap()
        .location(
            &PartitionSpec::unpartition_spec(),
            m.current_schema(),
            &Struct::empty(),
            name,
        )
        .unwrap();
    assert_eq!(r.delete_files[0].file_path(), want);
    assert_ne!(want, env.at(&format!("t/data/{name}")));
    assert!(!want.contains("x=1"), "{want}");
}

/// How [`Faulty`] treats the uploads of the files it targets.
#[derive(Debug, Clone, Copy)]
enum Fault {
    /// Refused before anything is written.
    Refuse,
    /// Written, then answered with an error (a lost answer).
    LandThenFail,
    /// Never answered (blocks until dropped).
    Block,
}

/// The local file system with a [`Fault`] on the uploads of files whose
/// names end with a suffix.
#[derive(Debug)]
struct Faulty {
    inner: object_store::local::LocalFileSystem,
    suffix: &'static str,
    fault: Fault,
    /// Set once a targeted upload started.
    started: Arc<std::sync::atomic::AtomicBool>,
}

impl Faulty {
    fn new(suffix: &'static str, fault: Fault) -> Self {
        Self {
            inner: object_store::local::LocalFileSystem::new(),
            suffix,
            fault,
            started: Arc::default(),
        }
    }

    fn targets(&self, location: &object_store::path::Path) -> bool {
        location.as_ref().ends_with(self.suffix)
    }
}

impl std::fmt::Display for Faulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Faulty")
    }
}

fn refused(location: &object_store::path::Path) -> object_store::Error {
    object_store::Error::PermissionDenied {
        path: location.to_string(),
        source: "refused by the test".into(),
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for Faulty {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        if self.targets(location) {
            self.started
                .store(true, std::sync::atomic::Ordering::SeqCst);
            match self.fault {
                Fault::Refuse => return Err(refused(location)),
                Fault::LandThenFail => {
                    self.inner.put_opts(location, payload, opts).await?;
                    return Err(refused(location));
                }
                Fault::Block => std::future::pending::<()>().await,
            }
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        if self.targets(location) {
            self.started
                .store(true, std::sync::atomic::Ordering::SeqCst);
            match self.fault {
                Fault::Refuse | Fault::LandThenFail => return Err(refused(location)),
                Fault::Block => std::future::pending::<()>().await,
            }
        }
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
