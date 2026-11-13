// SPDX-License-Identifier: Apache-2.0
//! Tests of the write task (`write`, PR-36a): rows in files whose partition
//! tuple matches every row (checked against iceberg-rust's transforms over
//! the rows read back), field ids in every footer, Java's names and
//! partition directories, rollover, the open-file cap, Parquet settings from
//! the table properties, an append of the written files read back through
//! SQL and iceberg-rust, and failed writes leaving no file behind.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, Date32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringViewArray, StructArray,
};
use arrow_schema::{DataType as A, Field, Fields, Schema};
use iceberg::spec::{
    Datum, Literal, NestedField, PartitionSpec, PrimitiveType, Schema as IcebergSchema,
    TableMetadata, Transform, Type,
};
use iceberg::transform::create_transform_function;
use iceberg::{Catalog, NamespaceIdent, TableCreation};
use krishiv_common::error::ErrorClass;
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::append::{prepare_attempt, write_new_manifests};
use super::append_tests::{MAX, commit, create, iceberg_rust_files, ids, now, rows};
use super::data_file::{WrittenDataFile, data_file};
use super::provider_tests::{Env, env, env_on};
use super::table::IcebergTable;
use super::write::{DEFAULT_MAX_OPEN_FILES, DataFileWriter, TaskIds};
use crate::plan::bounds::encode;
use crate::plan::{IcebergType, ManifestFilter, PruneOptions, VersionSpec};

/// id long (1), name string (2), day date (3), v double (4), tag struct (5)
/// of k int (6).
fn schema() -> IcebergSchema {
    IcebergSchema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "day", Type::Primitive(PrimitiveType::Date)).into(),
            NestedField::optional(4, "v", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::optional(
                5,
                "tag",
                Type::Struct(iceberg::spec::StructType::new(vec![
                    NestedField::optional(6, "k", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .unwrap()
}

/// `n` rows from `start` (no field ids in the batch's schema): names repeat
/// every 5 rows and are null every 7th, days cycle over 3, `tag` is null
/// every 11th row.
fn rows_batch(start: i64, n: usize) -> RecordBatch {
    let ids: Vec<i64> = (start..start + n as i64).collect();
    let names: Vec<Option<String>> = ids
        .iter()
        .map(|i| (i % 7 != 0).then(|| format!("name-{}", i % 5)))
        .collect();
    let days: Vec<i32> = ids.iter().map(|i| 19_000 + (i % 3) as i32).collect();
    let k = Field::new("k", A::Int32, true);
    let tag = StructArray::try_new(
        Fields::from(vec![k.clone()]),
        vec![Arc::new(Int32Array::from(
            ids.iter().map(|i| (i % 4) as i32).collect::<Vec<_>>(),
        )) as ArrayRef],
        Some(ids.iter().map(|i| i % 11 != 0).collect::<Vec<_>>().into()),
    )
    .unwrap();
    let schema = Schema::new(vec![
        Field::new("id", A::Int64, true),
        Field::new("name", A::Utf8View, true),
        Field::new("day", A::Date32, true),
        Field::new("v", A::Float64, true),
        Field::new("tag", A::Struct(Fields::from(vec![k])), true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(StringViewArray::from(names)),
            Arc::new(Date32Array::from(days)),
            Arc::new(Float64Array::from(
                ids.iter().map(|i| *i as f64).collect::<Vec<_>>(),
            )),
            Arc::new(tag),
        ],
    )
    .unwrap()
}

/// A table `t<n>` of [`schema`] partitioned by `fields` (source name,
/// transform), with `props`.
fn table(
    env: &Env,
    name: &str,
    fields: &[(&str, Transform)],
    props: &[(&str, &str)],
) -> TableMetadata {
    let schema = schema();
    let mut spec = PartitionSpec::builder(schema.clone()).with_spec_id(0);
    for (i, (source, t)) in fields.iter().enumerate() {
        let name = format!("{}_p{i}", source.replace('.', "_"));
        spec = spec.add_partition_field(*source, name, *t).unwrap();
    }
    let creation = TableCreation::builder()
        .name(name.to_owned())
        .schema(schema)
        .partition_spec(spec.build().unwrap().into_unbound())
        .properties(
            props
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned())),
        )
        .build();
    env.rt.block_on(async {
        env.catalog
            .create_table(&NamespaceIdent::new("db".to_owned()), creation)
            .await
            .unwrap()
            .metadata()
            .clone()
    })
}

fn writer(env: &Env, m: &TableMetadata, max_open: usize) -> DataFileWriter {
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let id = pool.register("write", None);
    DataFileWriter::new(
        m,
        &env.reg,
        TaskIds {
            partition_id: 3,
            task_id: 7,
            write_id: "w1",
        },
        max_open,
        Reservation::new(pool, id),
    )
    .unwrap()
}

/// Writes `batches` and finishes.
fn write_all(env: &Env, mut w: DataFileWriter, batches: &[RecordBatch]) -> Vec<WrittenDataFile> {
    env.rt.block_on(async {
        for b in batches {
            w.write(b).await.unwrap();
        }
        w.finish().await.unwrap()
    })
}

fn local(location: &str) -> &str {
    location.strip_prefix("file://").unwrap()
}

/// The rows of the Parquet file at `location`.
fn read(location: &str) -> RecordBatch {
    let file = std::fs::File::open(local(location)).unwrap();
    let r = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = r.map(Result::unwrap).collect();
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// Every `.parquet` file under `dir`.
fn parquet_files(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(parquet_files(&p));
        } else if p.extension().is_some_and(|x| x == "parquet") {
            out.push(p.display().to_string());
        }
    }
    out.sort();
    out
}

/// The source value of row `row` of `batch` at the column `path`, as a
/// datum (`None`: null, or no such column).
fn source(batch: &RecordBatch, path: &[&str], row: usize) -> Option<Datum> {
    let mut a: ArrayRef = Arc::clone(batch.column_by_name(path[0])?);
    if a.is_null(row) {
        return None;
    }
    for name in &path[1..] {
        a = Arc::clone(a.as_struct().column_by_name(name)?);
        if a.is_null(row) {
            return None;
        }
    }
    Some(match a.data_type() {
        A::Int64 => Datum::long(a.as_primitive::<Int64Type>().value(row)),
        A::Int32 => Datum::int(a.as_primitive::<Int32Type>().value(row)),
        A::Date32 => Datum::date(a.as_primitive::<Date32Type>().value(row)),
        A::Utf8View => Datum::string(a.as_string_view().value(row)),
        A::Utf8 => Datum::string(a.as_string::<i32>().value(row)),
        other => panic!("{other}"),
    })
}

/// Every file of `files` holds rows whose transform (by iceberg-rust) is the
/// file's partition tuple, sits in its partition's directory under Java's
/// name, and the files hold `expected` rows in all.
fn check_files(
    m: &TableMetadata,
    files: &[WrittenDataFile],
    sources: &[(&[&str], Transform)],
    expected: usize,
) {
    let spec = m.default_partition_spec();
    let mut total = 0;
    for f in files {
        let batch = read(&f.location);
        assert_eq!(batch.num_rows() as u64, f.rows, "{}", f.location);
        total += batch.num_rows();
        let tuple: Vec<Option<Literal>> =
            f.partition.iter().map(Option::<&Literal>::cloned).collect();
        for row in 0..batch.num_rows() {
            for (i, (path, t)) in sources.iter().enumerate() {
                let want = source(&batch, path, row).and_then(|d| {
                    create_transform_function(t)
                        .unwrap()
                        .transform_literal(&d)
                        .unwrap()
                });
                let want = want.map(|d| Literal::Primitive(d.literal().clone()));
                assert_eq!(tuple[i], want, "{} row {row}", f.location);
            }
        }
        let dir = if spec.fields().is_empty() {
            String::new()
        } else {
            format!(
                "{}/",
                spec.partition_to_path(&f.partition, Arc::clone(m.current_schema()))
            )
        };
        let name = f.location.rsplit('/').next().unwrap();
        assert!(
            f.location.ends_with(&format!("/db/t/data/{dir}{name}"))
                || f.location.contains(&format!("/data/{dir}{name}")),
            "{}",
            f.location
        );
        assert!(
            name.starts_with("00003-7-w1-") && name.ends_with(".parquet") && name.len() == 24,
            "{name}"
        );
        assert_eq!(f.spec_id, spec.spec_id());
        assert_eq!((f.split_offsets.as_ref(), f.sort_order_id), (None, None));
    }
    assert_eq!(total, expected);
}

/// Each footer field's id, by its path.
fn footer_ids(location: &str) -> Vec<(String, i32)> {
    let file = std::fs::File::open(local(location)).unwrap();
    let b = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    fn walk(t: &parquet::schema::types::Type, prefix: &str, out: &mut Vec<(String, i32)>) {
        for f in t.get_fields() {
            let name = format!("{prefix}{}", f.name());
            let info = f.get_basic_info();
            out.push((name.clone(), if info.has_id() { info.id() } else { -1 }));
            if f.is_group() {
                walk(f, &format!("{name}."), out);
            }
        }
    }
    let mut out = Vec::new();
    walk(b.parquet_schema().root_schema(), "", &mut out);
    out
}

#[test]
fn unpartitioned_one_file_with_field_ids_and_metrics() {
    let env = env();
    let m = table(&env, "t", &[], &[]);
    let files = write_all(
        &env,
        writer(&env, &m, DEFAULT_MAX_OPEN_FILES),
        &[rows_batch(0, 40), rows_batch(40, 60)],
    );
    assert_eq!(files.len(), 1);
    check_files(&m, &files, &[], 100);
    let f = &files[0];
    assert!(
        f.location.ends_with("/db/t/data/00003-7-w1-00001.parquet"),
        "{}",
        f.location
    );
    assert_eq!(
        footer_ids(&f.location),
        [
            ("id".to_owned(), 1),
            ("name".to_owned(), 2),
            ("day".to_owned(), 3),
            ("v".to_owned(), 4),
            ("tag".to_owned(), 5),
            ("tag.k".to_owned(), 6),
        ]
    );
    assert_eq!(f.size, std::fs::metadata(local(&f.location)).unwrap().len());
    // Metrics from the data: 100 values, names null every 7th row, id 0..=99.
    assert_eq!(f.metrics.value_counts.get(&1), Some(&100));
    assert_eq!(f.metrics.null_counts.get(&2), Some(&15));
    assert_eq!(f.metrics.nan_counts.get(&4), Some(&0));
    let long = |v: i64| encode(&krishiv_io::ScalarValue::Int64(v), &IcebergType::Long);
    assert_eq!(f.metrics.lower_bounds.get(&1).cloned(), long(0));
    assert_eq!(f.metrics.upper_bounds.get(&1).cloned(), long(99));
    // gzip when the table sets no codec (Java's `PARQUET_COMPRESSION_DEFAULT`).
    let codec = codecs(&f.location);
    assert!(
        codec.len() == 1 && codec[0].starts_with("GZIP"),
        "{codec:?}"
    );
}

/// Each column chunk's codec of the first row group.
fn codecs(location: &str) -> Vec<String> {
    let file = std::fs::File::open(local(location)).unwrap();
    let b = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let mut out: Vec<String> = b.metadata().row_groups()[0]
        .columns()
        .iter()
        .map(|c| format!("{:?}", c.compression()))
        .collect();
    out.dedup();
    out
}

fn partitioned(fields: &[(&str, Transform)], sources: &[(&[&str], Transform)]) {
    let env = env();
    let m = table(&env, "t", fields, &[]);
    let files = write_all(
        &env,
        writer(&env, &m, DEFAULT_MAX_OPEN_FILES),
        &[rows_batch(0, 50), rows_batch(50, 70)],
    );
    assert!(files.len() > 1, "{}", files.len());
    check_files(&m, &files, sources, 120);
}

#[test]
fn identity_partitions_with_null_values() {
    partitioned(
        &[("name", Transform::Identity)],
        &[(&["name"], Transform::Identity)],
    );
}

#[test]
fn bucket_partitions() {
    partitioned(
        &[("id", Transform::Bucket(4))],
        &[(&["id"], Transform::Bucket(4))],
    );
}

#[test]
fn day_and_truncate_partitions() {
    partitioned(
        &[("day", Transform::Day), ("name", Transform::Truncate(3))],
        &[
            (&["day"], Transform::Day),
            (&["name"], Transform::Truncate(3)),
        ],
    );
}

#[test]
fn nested_source_and_void_partitions() {
    partitioned(
        &[("tag.k", Transform::Identity), ("id", Transform::Void)],
        &[
            (&["tag", "k"], Transform::Identity),
            (&["id"], Transform::Void),
        ],
    );
}

#[test]
fn uncomputable_transform_fails_the_write() {
    let env = env();
    let m = table(&env, "t", &[("id", Transform::Unknown)], &[]);
    let mut w = writer(&env, &m, DEFAULT_MAX_OPEN_FILES);
    let e = env.rt.block_on(w.write(&rows_batch(0, 10))).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    env.rt.block_on(w.abort());
    assert_eq!(parquet_files(env.dir.path()), Vec::<String>::new());
}

#[test]
fn rollover_at_the_target_size_every_thousand_rows() {
    let env = env();
    let m = table(
        &env,
        "t",
        &[("day", Transform::Day)],
        &[("write.target-file-size-bytes", "1")],
    );
    let files = write_all(
        &env,
        writer(&env, &m, DEFAULT_MAX_OPEN_FILES),
        &[rows_batch(0, 6_500)],
    );
    // 6,500 rows over 3 days: 2,167 + 2,167 + 2,166 rows, files of 1,000.
    assert_eq!(
        files.len(),
        9,
        "{:?}",
        files.iter().map(|f| f.rows).collect::<Vec<_>>()
    );
    check_files(&m, &files, &[(&["day"], Transform::Day)], 6_500);
    let mut names: Vec<&str> = files
        .iter()
        .map(|f| f.location.rsplit('/').next().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(names[0], "00003-7-w1-00001.parquet");
    assert_eq!(names[8], "00003-7-w1-00009.parquet");
    let mut rows: Vec<u64> = files.iter().map(|f| f.rows).collect();
    rows.sort_unstable();
    assert_eq!(rows, [166, 167, 167, 1000, 1000, 1000, 1000, 1000, 1000]);
}

#[test]
fn max_open_files_closes_the_oldest() {
    let env = env();
    let m = table(&env, "t", &[("day", Transform::Day)], &[]);
    let files = write_all(
        &env,
        writer(&env, &m, 2),
        &[rows_batch(0, 30), rows_batch(30, 30)],
    );
    // Each batch opens the three days in turn; with two open the oldest closes.
    assert_eq!(files.len(), 6);
    check_files(&m, &files, &[(&["day"], Transform::Day)], 60);
}

#[test]
fn parquet_settings_from_the_table_properties() {
    let env = env();
    let m = table(
        &env,
        "t",
        &[],
        &[
            ("write.parquet.compression-codec", "GZIP"),
            ("write.parquet.row-group-size-bytes", "4096"),
        ],
    );
    let files = write_all(
        &env,
        writer(&env, &m, DEFAULT_MAX_OPEN_FILES),
        &[rows_batch(0, 5_000)],
    );
    let file = std::fs::File::open(local(&files[0].location)).unwrap();
    let b = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    assert!(b.metadata().num_row_groups() > 1);
    assert!(codecs(&files[0].location)[0].starts_with("GZIP"));

    for (key, value) in [
        ("write.parquet.compression-codec", "lzo"),
        ("write.parquet.compression-codec", "foo"),
        ("write.parquet.row-group-size-bytes", "big"),
        ("write.target-file-size-bytes", "1.5"),
        ("write.format.default", "orc"),
    ] {
        let m = table(&env, &format!("t_{value}"), &[], &[(key, value)]);
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let id = pool.register("write", None);
        let e = DataFileWriter::new(
            &m,
            &env.reg,
            TaskIds {
                partition_id: 0,
                task_id: 0,
                write_id: "w",
            },
            100,
            Reservation::new(pool, id),
        )
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{key}={value}: {e}"
        );
    }
}

/// The written files appended (manifests, list, a catalog applying the
/// request) read back through SQL and iceberg-rust's planning; the metrics
/// in the manifest are the ones of the data.
#[test]
fn written_files_append_and_read_back() {
    let env = env();
    let t = create(&env, 2);
    let m0 = t.metadata().clone();
    let loc0 = t.metadata_location().unwrap().to_owned();
    let b = super::provider_tests::batch(
        vec![
            Field::new("id", A::Int64, true),
            Field::new("x", A::Int32, true),
            Field::new("name", A::Utf8View, true),
        ],
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                Some(1),
                None,
                Some(2),
            ])),
            Arc::new(StringViewArray::from(vec![
                Some("a"),
                Some("b"),
                None,
                Some("d"),
                Some("e"),
            ])),
        ],
    );
    let files = write_all(&env, writer(&env, &m0, DEFAULT_MAX_OPEN_FILES), &[b]);
    assert_eq!(files.len(), 3);
    let schema = m0.current_schema();
    let entries = files
        .iter()
        .map(|f| (f.spec_id, data_file(f, schema).unwrap()))
        .collect();
    let io = env.reg.file_io();
    let attempt = env.rt.block_on(async {
        let mut new = write_new_manifests(&io, &m0, entries, 501, "uuid-501")
            .await
            .unwrap();
        prepare_attempt(&io, &m0, &mut new, ids("w1"), now(&m0), MAX)
            .await
            .unwrap()
    });
    let (m1, loc1) = commit(&env, &m0, &loc0, &attempt, 1);
    assert_eq!(
        rows(&env, &loc1),
        ["1|1|a", "2|2|b", "3|1|NULL", "4|NULL|d", "5|2|e"]
    );
    let mut written: Vec<String> = files.iter().map(|f| f.location.clone()).collect();
    written.sort();
    assert_eq!(iceberg_rust_files(&env, &m1, &loc1), written);

    // The manifest's metrics are the data's.
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, &loc1, "db", "t"))
        .unwrap();
    let snap = table.snapshot(&VersionSpec::Current).unwrap();
    let plan = env
        .rt
        .block_on(table.plan(&snap, None, ManifestFilter::All, PruneOptions::default()))
        .unwrap();
    let by_path: HashMap<&str, &WrittenDataFile> =
        files.iter().map(|f| (f.location.as_str(), f)).collect();
    let long = |v: i64| encode(&krishiv_io::ScalarValue::Int64(v), &IcebergType::Long).unwrap();
    for task in &plan.tasks {
        let c = &task.file.file;
        let w = by_path[c.path.as_str()];
        let data = read(&c.path);
        let ids = data
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        let min = (0..ids.len()).map(|i| ids.value(i)).min().unwrap();
        let max = (0..ids.len()).map(|i| ids.value(i)).max().unwrap();
        let names = data.column_by_name("name").unwrap();
        assert_eq!(c.record_count, data.num_rows() as u64);
        assert_eq!(c.size, w.size);
        assert_eq!(c.value_counts.get(&1), Some(&(data.num_rows() as u64)));
        assert_eq!(c.null_counts.get(&3), Some(&(names.null_count() as u64)));
        assert_eq!(c.lower_bounds.get(&1), Some(&long(min)));
        assert_eq!(c.upper_bounds.get(&1), Some(&long(max)));
        assert_eq!(
            (
                &c.value_counts,
                &c.null_counts,
                &c.lower_bounds,
                &c.upper_bounds
            ),
            (
                &w.metrics.value_counts,
                &w.metrics.null_counts,
                &w.metrics.lower_bounds,
                &w.metrics.upper_bounds
            )
        );
    }
    assert_eq!(plan.tasks.len(), 3);
}

/// A batch of another schema after some files were finished: the write
/// fails as an internal error and the abort leaves no file.
#[test]
fn failed_write_leaves_no_file() {
    let env = env();
    let m = table(&env, "t", &[], &[("write.target-file-size-bytes", "1")]);
    let mut w = writer(&env, &m, DEFAULT_MAX_OPEN_FILES);
    env.rt.block_on(w.write(&rows_batch(0, 2_500))).unwrap();
    assert_eq!(parquet_files(env.dir.path()).len(), 2);
    let other = super::provider_tests::batch(
        vec![Field::new("id", A::Int64, true)],
        vec![Arc::new(Int64Array::from(vec![1]))],
    );
    let e = env.rt.block_on(w.write(&other)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    env.rt.block_on(w.abort());
    assert_eq!(parquet_files(env.dir.path()), Vec::<String>::new());
}

/// A writer dropped before `finish` deletes its finished files.
#[test]
fn dropped_writer_deletes_its_files() {
    let env = env();
    let m = table(&env, "t", &[], &[("write.target-file-size-bytes", "1")]);
    let mut w = writer(&env, &m, DEFAULT_MAX_OPEN_FILES);
    env.rt.block_on(w.write(&rows_batch(0, 2_500))).unwrap();
    assert_eq!(parquet_files(env.dir.path()).len(), 2);
    env.rt.block_on(async move { drop(w) });
    for _ in 0..500 {
        if parquet_files(env.dir.path()).is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(parquet_files(env.dir.path()), Vec::<String>::new());
}

/// An upload that fails (the third file's) fails the write; the files
/// finished before it are deleted.
#[test]
fn failed_upload_deletes_the_finished_files() {
    let store = Arc::new(FailThird(object_store::local::LocalFileSystem::new()));
    let env = env_on(store, krishiv_io::IoConfig::default());
    let m = table(&env, "t", &[], &[("write.target-file-size-bytes", "1")]);
    let w = writer(&env, &m, DEFAULT_MAX_OPEN_FILES);
    let e = env.rt.block_on(async {
        let mut w = w;
        let r = w.write(&rows_batch(0, 3_500)).await;
        match r {
            Ok(()) => w.finish().await.map(drop),
            Err(e) => {
                w.abort().await;
                Err(e)
            }
        }
    });
    let e = e.unwrap_err().to_string();
    assert!(e.contains("00003-7-w1-00003.parquet"), "{e}");
    assert_eq!(parquet_files(env.dir.path()), Vec::<String>::new());
}

/// The local file system refusing writes of files named `*-00003.parquet`.
#[derive(Debug)]
struct FailThird(object_store::local::LocalFileSystem);

impl std::fmt::Display for FailThird {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailThird")
    }
}

fn refused(location: &object_store::path::Path) -> object_store::Error {
    object_store::Error::PermissionDenied {
        path: location.to_string(),
        source: "refused by the test".into(),
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FailThird {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        if location.as_ref().ends_with("-00003.parquet") {
            return Err(refused(location));
        }
        self.0.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        if location.as_ref().ends_with("-00003.parquet") {
            return Err(refused(location));
        }
        self.0.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.0.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        self.0.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.0.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.0.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.0.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.0.copy_opts(from, to, options).await
    }
}
