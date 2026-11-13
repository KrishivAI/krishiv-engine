// SPDX-License-Identifier: Apache-2.0
//! Tests of a written file's manifest entry (`data_file`, PR-36a): partition
//! directories and data locations as Java's `LocationProviders`, the partition
//! tuple, and a `DataFile` written into a manifest by iceberg-rust and read back
//! by the scan's parser with the metrics it was built from.

use std::sync::Arc;

use iceberg::spec::{
    FormatVersion, ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Schema,
    SchemaRef, SortOrder, Struct, TableMetadata, TableMetadataBuilder, Transform, Type,
};
use krishiv_io::{ColumnPath, ColumnStatistics, FileStatistics, Precision, ScalarValue as S};

use super::data_file::{
    DataLocations, OBJECT_STORE_ENABLED, WrittenDataFile, data_file, file_name, partition_tuple,
};
use super::scan::{parse_manifest, schema_fields};
use super::tests::local;
use crate::plan::file_metrics::file_metrics;
use crate::plan::{EntryStatus, MetricsMode, PartitionValue};

fn schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "ts", Type::Primitive(PrimitiveType::Date)).into(),
            NestedField::optional(4, "score", Type::Primitive(PrimitiveType::Double)).into(),
        ])
        .build()
        .unwrap()
}

fn spec(schema: &Schema) -> PartitionSpec {
    PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("ts", "ts_day", Transform::Day)
        .unwrap()
        .add_partition_field("name", "name", Transform::Identity)
        .unwrap()
        .add_partition_field("id", "id_bucket", Transform::Bucket(4))
        .unwrap()
        .build()
        .unwrap()
}

fn metadata(props: &[(&str, &str)]) -> TableMetadata {
    let schema = schema();
    let spec = spec(&schema);
    TableMetadataBuilder::new(
        schema,
        spec,
        SortOrder::unsorted_order(),
        "s3://bucket/db/t".to_owned(),
        FormatVersion::V2,
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

/// The tuple of the table's default spec with `values`.
fn key(m: &TableMetadata, values: &[Option<S>]) -> Struct {
    partition_tuple(m.default_partition_spec(), m.current_schema(), values).unwrap()
}

/// Where `locs` puts `f.parquet` in the partition `k` of `m`'s default spec.
fn at(locs: &DataLocations, m: &TableMetadata, k: &Struct) -> String {
    locs.location(
        m.default_partition_spec(),
        m.current_schema(),
        k,
        "f.parquet",
    )
    .unwrap()
}

/// Where `locs` puts `f.parquet` under an unpartitioned spec.
fn unpartitioned(locs: &DataLocations, m: &TableMetadata) -> String {
    let spec = PartitionSpec::builder(Arc::clone(m.current_schema()))
        .with_spec_id(1)
        .build()
        .unwrap();
    locs.location(&spec, m.current_schema(), &Struct::empty(), "f.parquet")
        .unwrap()
}

/// 2017-07-05 as days from 1970-01-01.
const DAY: i32 = 17_352;

#[test]
fn partition_paths_as_java() {
    let m = metadata(&[]);
    let locs = DataLocations::for_table(&m).unwrap();
    let k = key(
        &m,
        &[
            Some(S::Date32(DAY)),
            Some(S::Utf8("a b/c=d".to_owned())),
            Some(S::Int32(3)),
        ],
    );
    // Java's `PartitionSpec.partitionToPath`: `URLEncoder.encode` of the name and
    // of the transform's human string (a date as ISO, a space as `+`).
    assert_eq!(
        at(&locs, &m, &k),
        "s3://bucket/db/t/data/ts_day=2017-07-05/name=a+b%2Fc%3Dd/id_bucket=3/f.parquet"
    );
    let nulls = key(&m, &[None, None, Some(S::Int32(0))]);
    assert_eq!(
        at(&locs, &m, &nulls),
        "s3://bucket/db/t/data/ts_day=null/name=null/id_bucket=0/f.parquet"
    );
    assert_eq!(unpartitioned(&locs, &m), "s3://bucket/db/t/data/f.parquet");
}

#[test]
fn data_path_properties_as_java() {
    let m = metadata(&[("write.data.path", "s3://other/place/")]);
    let locs = DataLocations::for_table(&m).unwrap();
    assert_eq!(unpartitioned(&locs, &m), "s3://other/place/f.parquet");

    // Object storage (`Boolean.parseBoolean`: any case): the low 20 bits of
    // the Murmur3 hash (seed 0) of the partition path and file name, as binary
    // digits in directories of 4, 4, 4 and 8 (Java's `computeHash` and
    // `dirsFromHash`), under the table's data directory (no context: it is
    // inside the table).
    let m = metadata(&[(OBJECT_STORE_ENABLED, "TRUE")]);
    let locs = DataLocations::for_table(&m).unwrap();
    let k = key(&m, &[Some(S::Date32(DAY)), None, Some(S::Int32(1))]);
    let name = "ts_day=2017-07-05/name=null/id_bucket=1/f.parquet";
    let hash = java_hash_dirs(name);
    assert_eq!(
        at(&locs, &m, &k),
        format!("s3://bucket/db/t/data/{hash}/{name}")
    );

    // Without partition paths the hash is joined to the file name with `-`.
    let m = metadata(&[
        (OBJECT_STORE_ENABLED, "true"),
        ("write.object-storage.partitioned-paths", "false"),
    ]);
    assert_eq!(
        at(&DataLocations::for_table(&m).unwrap(), &m, &k),
        format!(
            "s3://bucket/db/t/data/{}-f.parquet",
            java_hash_dirs("f.parquet")
        )
    );

    // A data path outside the table adds the table's last two path segments.
    let m = metadata(&[
        (OBJECT_STORE_ENABLED, "true"),
        ("write.data.path", "s3://other/place"),
    ]);
    assert_eq!(
        unpartitioned(&DataLocations::for_table(&m).unwrap(), &m),
        format!(
            "s3://other/place/{}/db/t/f.parquet",
            java_hash_dirs("f.parquet")
        )
    );

    // Any other value is false.
    let m = metadata(&[(OBJECT_STORE_ENABLED, "yes")]);
    assert_eq!(
        unpartitioned(&DataLocations::for_table(&m).unwrap(), &m),
        "s3://bucket/db/t/data/f.parquet"
    );
}

/// Java's `ObjectStoreLocationProvider.computeHash`, written from its source:
/// `Integer.toBinaryString(murmur3_32_fixed(name) | Integer.MIN_VALUE)`, its
/// last 20 digits, split by `dirsFromHash`.
fn java_hash_dirs(name: &str) -> String {
    let h = crate::plan::transform::murmur3_32(name.as_bytes()) | 0x8000_0000;
    let bits = format!("{h:b}");
    let hash = &bits[bits.len() - 20..];
    format!(
        "{}/{}/{}/{}",
        &hash[..4],
        &hash[4..8],
        &hash[8..12],
        &hash[12..]
    )
}

#[test]
fn file_names_are_unique_per_task_and_count() {
    assert_eq!(
        file_name(3, 17, "w-1", 2),
        "00003-17-w-1-00002.parquet".to_owned()
    );
    assert_ne!(file_name(0, 1, "w", 0), file_name(0, 2, "w", 0));
    assert_ne!(file_name(0, 1, "w", 0), file_name(0, 1, "w", 1));
}

/// A tuple that does not fit its spec is an error, never one of iceberg-rust's
/// panics (review 36a S1); a custom location provider is refused (S4).
#[test]
fn locations_refuse_what_iceberg_rust_would_panic_on() {
    let m = metadata(&[]);
    let locs = DataLocations::for_table(&m).unwrap();
    let spec = m.default_partition_spec();
    let schema = m.current_schema();
    let short = Struct::from_iter([Some(iceberg::spec::Literal::int(1))]);
    assert!(locs.location(spec, schema, &short, "f").is_err());
    // A long where the day field holds a date.
    let wrong = Struct::from_iter([
        Some(iceberg::spec::Literal::long(1)),
        None,
        Some(iceberg::spec::Literal::int(1)),
    ]);
    assert!(locs.location(spec, schema, &wrong, "f").is_err());

    let m = metadata(&[("write.location-provider.impl", "com.example.Provider")]);
    let e = DataLocations::for_table(&m).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.to_string().contains("com.example.Provider"), "{e}");
}

#[test]
fn partition_tuple_refuses_values_that_do_not_fit_the_spec() {
    let m = metadata(&[]);
    let schema = m.current_schema();
    let spec = m.default_partition_spec().as_ref().clone();
    assert!(partition_tuple(&spec, schema, &[None, None]).is_err());
    // A day is a date, not a long.
    assert!(partition_tuple(&spec, schema, &[Some(S::Int64(1)), None, None]).is_err());
}

fn col(name: &str, nulls: u64, min: S, max: S) -> ColumnStatistics {
    let mut c = ColumnStatistics::new(ColumnPath::new(name));
    c.null_count = Precision::Exact(nulls);
    c.min = Precision::Exact(min);
    c.max = Precision::Exact(max);
    c
}

/// A `DataFile` built from a written file's statistics, written into a v2
/// manifest by iceberg-rust and parsed back by the scan's parser, carries the
/// metrics, partition tuple, sort order and split offsets it was built from.
#[tokio::test]
async fn data_file_round_trips_through_a_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let (_sched, reg) = local(dir.path());
    let schema: SchemaRef = Arc::new(schema());
    let spec = spec(&schema);

    let mut stats = FileStatistics::with_rows(Precision::Exact(10));
    let mut score = col("score", 1, S::Float64(-1.5), S::Float64(2.5));
    score.nan_count = Precision::Exact(2);
    stats.columns = vec![
        col("id", 0, S::Int64(-7), S::Int64(70)),
        col(
            "name",
            3,
            S::Utf8("apple".to_owned()),
            S::Utf8("zucchini-and-more-long-text".to_owned()),
        ),
        col("ts", 0, S::Date32(DAY), S::Date32(DAY)),
        score,
    ];
    let metrics = file_metrics(&stats, &schema_fields(&schema), |_| {
        MetricsMode::Truncate(16)
    });
    let partition = partition_tuple(
        &spec,
        &schema,
        &[Some(S::Date32(DAY)), None, Some(S::Int32(2))],
    )
    .unwrap();
    let written = WrittenDataFile {
        location: "s3://bucket/db/t/data/a.parquet".to_owned(),
        size: 1234,
        rows: 10,
        metrics: metrics.clone(),
        partition,
        spec_id: 0,
        split_offsets: Some(vec![4]),
        sort_order_id: None,
    };
    let df = data_file(&written, &schema).unwrap();

    let path = format!("file://{}/m.avro", dir.path().display());
    let mut w = ManifestWriterBuilder::new(
        reg.file_io().new_output(&path).unwrap(),
        Some(9),
        Arc::clone(&schema),
        spec,
    )
    .build_v2_data();
    w.add_file(df, -1).unwrap();
    w.write_manifest_file().await.unwrap();
    let bytes = reg
        .file_io()
        .new_input(&path)
        .unwrap()
        .read()
        .await
        .unwrap();
    let parsed = parse_manifest(&bytes, 0).unwrap();
    assert_eq!(parsed.len(), 1);
    let e = &parsed[0];
    assert_eq!(e.status, EntryStatus::Added);
    assert_eq!((e.sequence_number, e.file_sequence_number), (None, None));
    let f = &e.file;
    assert_eq!(f.path, written.location);
    assert_eq!((f.record_count, f.size), (10, 1234));
    assert_eq!(
        f.partition,
        vec![
            PartitionValue::Value(DAY.to_le_bytes().to_vec()),
            PartitionValue::Null,
            PartitionValue::Value(2i32.to_le_bytes().to_vec()),
        ]
    );
    assert_eq!(f.value_counts, metrics.value_counts);
    assert_eq!(f.null_counts, metrics.null_counts);
    assert_eq!(f.nan_counts, std::collections::BTreeMap::from([(4, 2)]));
    assert_eq!(f.lower_bounds, metrics.lower_bounds);
    assert_eq!(f.upper_bounds, metrics.upper_bounds);
    // The long string's upper bound is truncated and incremented: 16 code points.
    assert_eq!(f.upper_bounds[&2], b"zucchini-and-mos".to_vec());
    assert_eq!(f.sort_order_id, Some(0));
    assert_eq!(f.split_offsets.as_deref(), Some(&[4][..]));
}

#[test]
fn a_bound_of_a_column_outside_the_schema_is_an_internal_error() {
    let schema: SchemaRef = Arc::new(schema());
    let mut written = WrittenDataFile {
        location: "x".to_owned(),
        size: 1,
        rows: 1,
        metrics: crate::plan::file_metrics::FileMetrics::default(),
        partition: iceberg::spec::Struct::empty(),
        spec_id: 0,
        split_offsets: None,
        sort_order_id: Some(1),
    };
    written.metrics.lower_bounds.insert(99, vec![0; 8]);
    let e = data_file(&written, &schema).unwrap_err();
    assert!(e.to_string().contains("99"), "{e}");
}
