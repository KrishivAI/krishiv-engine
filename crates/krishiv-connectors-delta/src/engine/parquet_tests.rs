// SPDX-License-Identifier: Apache-2.0
//! Parquet handler tests on a local temporary table: `engine_reports_written_file_size`, the
//! read contract (order, field ids, missing columns, widening, metadata columns, nested
//! subsets), the footer, the limits, confinement and cancellation.

use std::collections::HashMap;

use arrow_array::types::Int64Type;
use arrow_array::{Array, Int32Array, Int64Array, StructArray};
use delta_kernel::schema::{MetadataColumnSpec, MetadataValue, SchemaRef};
use delta_kernel::{FileDataReadResultIterator, KernelResult};

use super::*;

/// A batch with `id` (int, field id 7), `name` (string) and `add` (struct of `path` and
/// `size`, plus `extra` the reads below never ask for).
fn data(rows: i32) -> RecordBatch {
    let ids: Vec<i32> = (0..rows).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let add = StructArray::try_new(
        vec![
            Field::new("path", ArrowType::Utf8, false),
            Field::new("size", ArrowType::Int64, true),
            Field::new("extra", ArrowType::Utf8, true),
        ]
        .into(),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|i| format!("p{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                ids.iter().map(|i| i64::from(*i) * 10).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec!["x"; ids.len()])),
        ],
        Some(ids.iter().map(|i| i % 2 == 0).collect::<Vec<_>>().into()),
    )
    .unwrap();
    let schema = Schema::new(vec![
        Field::new("id", ArrowType::Int32, false).with_metadata(HashMap::from([(
            "PARQUET:field_id".to_owned(),
            "7".to_owned(),
        )])),
        Field::new("name", ArrowType::Utf8, true),
        Field::new("add", add.data_type().clone(), true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(add),
        ],
    )
    .unwrap()
}

fn one(batch: RecordBatch) -> delta_kernel::KernelResultIteratorStatic<Box<dyn EngineData>> {
    Box::new(std::iter::once(Ok(
        Box::new(KrishivEngineData::new(batch)) as Box<dyn EngineData>
    )))
}

/// The file's metadata as the kernel lists it: size, and the modification time in
/// milliseconds (truncated), so reads carry a `Modified` version.
fn meta(f: &Fixture, rel: &str) -> FileMeta {
    let m = std::fs::metadata(f.dir.join(rel)).unwrap();
    let millis = m
        .modified()
        .unwrap()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    FileMeta {
        location: f.url(rel),
        last_modified: i64::try_from(millis).unwrap(),
        size: m.len(),
    }
}

fn collect(it: KernelResult<FileDataReadResultIterator>) -> KernelResult<Vec<RecordBatch>> {
    it?.map(|d| d.and_then(|d| KrishivEngineData::from_engine_data(d).map(|d| d.into_batch())))
        .collect()
}

fn with_id(field: StructField, id: i64) -> StructField {
    field.with_metadata([("parquet.field.id".to_owned(), MetadataValue::Number(id))])
}

#[test]
fn engine_reports_written_file_size() {
    let f = Fixture::new();
    let url = f.url("_delta_log/00000000000000000002.checkpoint.parquet");
    let size = f
        .run(move |e| e.parquet().write_parquet_file(url, one(data(5))))
        .unwrap();
    let on_disk = std::fs::metadata(
        f.dir
            .join("_delta_log/00000000000000000002.checkpoint.parquet"),
    )
    .unwrap()
    .len();
    assert_eq!(size, on_disk);
    assert!(size > 0);
    // No data is an error, not an empty file.
    let url = f.url("_delta_log/empty.parquet");
    let e = f
        .run(move |e| {
            e.parquet()
                .write_parquet_file(url, Box::new(std::iter::empty()))
        })
        .unwrap_err();
    assert!(e.to_string().contains("no data"), "{e}");
}

#[test]
fn parquet_reads_follow_the_kernel_contract() {
    let f = Fixture::new();
    let rel = "_delta_log/00000000000000000002.checkpoint.parquet";
    let url = f.url(rel);
    f.run(move |e| e.parquet().write_parquet_file(url, one(data(5))))
        .unwrap();
    let file = meta(&f, rel);
    let physical: SchemaRef = Arc::new(
        StructType::try_new([
            // Matched by field id under another name, and widened from int.
            with_id(StructField::not_null("logical_id", DataType::LONG), 7),
            StructField::create_metadata_column("row_index", MetadataColumnSpec::RowIndex),
            StructField::nullable("missing", DataType::STRING),
            // A nested subset: `extra` is not asked for, `tags` is absent (NULL).
            StructField::nullable(
                "add",
                StructType::try_new([
                    StructField::not_null("path", DataType::STRING),
                    StructField::nullable("tags", DataType::STRING),
                ])
                .unwrap(),
            ),
            StructField::create_metadata_column("_file", MetadataColumnSpec::FilePath),
            StructField::nullable("name", DataType::STRING),
        ])
        .unwrap(),
    );
    let files = vec![file.clone(), file];
    let schema = Arc::clone(&physical);
    let batches = f
        .run(move |e| collect(e.parquet().read_parquet_files(&files, schema, None)))
        .unwrap();
    let out = arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(out.num_rows(), 10);
    assert_eq!(
        out.schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["logical_id", "row_index", "missing", "add", "_file", "name"]
    );
    assert_eq!(
        out.column(0).as_primitive::<Int64Type>().values().to_vec(),
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    );
    // The row index restarts with each file.
    assert_eq!(
        out.column(1).as_primitive::<Int64Type>().values().to_vec(),
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    );
    assert_eq!(out.column(2).null_count(), 10);
    let add = out.column(3).as_struct();
    assert_eq!(add.num_columns(), 2);
    assert!(add.is_valid(0) && add.is_null(1));
    assert_eq!(add.column(0).as_string::<i32>().value(2), "p2");
    assert_eq!(add.column(1).null_count(), 10);
    assert_eq!(
        out.column(4).as_string::<i32>().value(7),
        f.url(rel).as_str()
    );
    assert_eq!(out.column(5).as_string::<i32>().value(3), "n3");
    // The kernel's types, exactly.
    assert_eq!(
        out.schema().as_ref(),
        &super::schema::to_arrow_schema(&physical).unwrap()
    );
}

#[test]
fn parquet_reads_refuse_what_they_cannot_fill() {
    let f = Fixture::new();
    let rel = "_delta_log/1.parquet";
    let url = f.url(rel);
    f.run(move |e| e.parquet().write_parquet_file(url, one(data(3))))
        .unwrap();
    let read = |schema: StructType| {
        let files = vec![meta(&f, rel)];
        f.run(move |e| {
            collect(
                e.parquet()
                    .read_parquet_files(&files, Arc::new(schema), None),
            )
        })
    };
    // A missing non-nullable column.
    let e = read(StructType::try_new([StructField::not_null("nope", DataType::INTEGER)]).unwrap())
        .unwrap_err();
    assert!(e.to_string().contains("non-nullable column nope"), "{e}");
    // A narrowing. 08b: a corrupt or mistyped file gets a named class (FAILED_READ_FILE);
    // update this assertion with it.
    let e = read(StructType::try_new([StructField::nullable("add", DataType::STRING)]).unwrap())
        .unwrap_err();
    assert_eq!(f.class(&e), ErrorClass::InternalError, "{e}");
    // A row-tracking metadata column.
    let e = read(
        StructType::try_new([StructField::create_metadata_column(
            "row_id",
            MetadataColumnSpec::RowId,
        )])
        .unwrap(),
    )
    .unwrap_err();
    assert!(matches!(e, KernelError::Unsupported(_)), "{e:?}");
}

#[test]
fn the_parquet_footer_carries_field_ids() {
    let f = Fixture::new();
    let rel = "_delta_log/1.parquet";
    let url = f.url(rel);
    f.run(move |e| e.parquet().write_parquet_file(url, one(data(3))))
        .unwrap();
    let file = meta(&f, rel);
    let footer = f
        .run(move |e| e.parquet().read_parquet_footer(&file))
        .unwrap();
    let id = footer.schema.field("id").unwrap();
    assert_eq!(
        id.get_config_value(&delta_kernel::schema::ColumnMetadataKey::ParquetFieldId),
        Some(&MetadataValue::Number(7))
    );
    assert_eq!(id.data_type(), &DataType::INTEGER);
    assert!(footer.schema.field("add").is_some());
}

#[test]
fn parquet_reads_apply_the_footer_limit_and_confinement() {
    let limits = DeltaLimits {
        max_metadata_bytes: 64,
        ..DeltaLimits::default()
    };
    let f = Fixture::with(limits, pool(2), None);
    let rel = "_delta_log/1.parquet";
    let url = f.url(rel);
    f.run(move |e| e.parquet().write_parquet_file(url, one(data(3))))
        .unwrap();
    let file = meta(&f, rel);
    let e = f
        .run(move |e| e.parquet().read_parquet_footer(&file))
        .unwrap_err();
    assert_eq!(f.class(&e), ErrorClass::KrishivCatalogLimitExceeded, "{e}");
    let outside = FileMeta {
        location: Url::parse("file:///etc/passwd").unwrap(),
        last_modified: 0,
        size: 10,
    };
    let schema: SchemaRef =
        Arc::new(StructType::try_new([StructField::nullable("id", DataType::INTEGER)]).unwrap());
    let e = f
        .run(move |e| {
            e.parquet()
                .read_parquet_files(&[outside], schema, None)
                .map(|_| ())
        })
        .unwrap_err();
    assert_eq!(
        f.class(&e),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(e.to_string().contains("outside the table root"), "{e}");
    let url = Url::parse("file:///tmp/krishiv-outside.parquet").unwrap();
    let e = f
        .run(move |e| e.parquet().write_parquet_file(url, one(data(1))))
        .unwrap_err();
    assert_eq!(
        f.class(&e),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(e.to_string().contains("outside the table root"), "{e}");
}

#[test]
fn a_cancelled_query_stops_between_row_groups() {
    let f = Fixture::new();
    let rel = "_delta_log/1.parquet";
    // One row per row group.
    let batch = data(4);
    let mut enc = krishiv_io::parquet::ParquetFormat::encoder_with(
        &batch.schema(),
        &crate::engine::columnar::format_options().unwrap(),
        |p| p.set_max_row_group_row_count(Some(1)),
    )
    .unwrap();
    enc.write(&batch).unwrap();
    let mut bytes = Vec::new();
    while let Some(b) = enc.take_bytes() {
        bytes.extend_from_slice(&b);
    }
    bytes.extend_from_slice(&enc.finish().unwrap().tail);
    f.write_bytes(rel, &bytes);
    let files = vec![meta(&f, rel)];
    let cancel = Arc::clone(&f.cancel);
    let (first, rest) = f.run(move |e| {
        let schema: SchemaRef = Arc::new(
            StructType::try_new([StructField::nullable("id", DataType::INTEGER)]).unwrap(),
        );
        let mut it = e
            .parquet()
            .read_parquet_files(&files, schema, None)
            .unwrap();
        let first = it
            .next()
            .map(|r| r.map(|d| KrishivEngineData::batch_of(d.as_ref()).unwrap().num_rows()));
        cancel.cancel();
        let rest: Vec<KernelResult<()>> = it.map(|r| r.map(|_| ())).collect();
        (first, rest)
    });
    assert!(matches!(first, Some(Ok(1))), "{first:?}");
    // The next pull sees the cancel; the iterator then ends.
    assert_eq!(rest.len(), 1);
    assert!(matches!(rest[0], Err(KernelError::Cancelled)));
}

#[test]
fn reads_are_versioned_by_the_store() {
    // This handler overwrites checkpoints and another writer may rewrite one: each read is
    // versioned by the store's metadata (the metadata cache key and the read precondition), so
    // a cached footer of an older file is never used for a newer one. The kernel's own times
    // do not version it (a sidecar's come from the checkpoint): a `FileMeta` with no time, or a
    // wrong one, still reads.
    let f = Fixture::new();
    let rel = "_delta_log/1.parquet";
    let url = f.url(rel);
    f.run(move |e| e.parquet().write_parquet_file(url, one(data(3))))
        .unwrap();
    let version = |m: FileMeta| {
        let shared = Arc::clone(&f.engine.shared);
        f.run(move |_| crate::engine::columnar::locate(&shared, &m).map(|(file, ..)| file.version))
            .unwrap()
    };
    let first = version(meta(&f, rel));
    assert_ne!(first, krishiv_io::VersionToken::Immutable);
    let wrong = FileMeta {
        last_modified: 1,
        size: 0,
        ..meta(&f, rel)
    };
    assert_eq!(version(wrong), first);
    std::thread::sleep(std::time::Duration::from_millis(5));
    let url = f.url(rel);
    f.run(move |e| e.parquet().write_parquet_file(url, one(data(4))))
        .unwrap();
    assert_ne!(version(meta(&f, rel)), first, "a rewrite is a new version");
    let files = vec![meta(&f, rel)];
    let schema: SchemaRef =
        Arc::new(StructType::try_new([StructField::nullable("name", DataType::STRING)]).unwrap());
    let rows: usize = f
        .run(move |e| collect(e.parquet().read_parquet_files(&files, schema, None)))
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 4);
}
