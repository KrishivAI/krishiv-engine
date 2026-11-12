// SPDX-License-Identifier: Apache-2.0
//! Tests of the file writer's helpers: committed files and the records-per-file limit.

use super::{committed_files, max_records_per_file};
use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_common::secret::OptionValue;

fn options(pairs: &[(&str, &str)]) -> BTreeMap<String, OptionValue> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), OptionValue::from_user(k, *v)))
        .collect()
}

/// The access policy is asked about the qualified path: a cloud URL
/// keeps its canonical form (no store is built), a local path gets the
/// `file:` form.
#[test]
fn qualify_path_keeps_cloud_urls_and_qualifies_local_paths() {
    let session = super::Session::new().unwrap();
    assert_eq!(
        session.qualify_path("s3a://bucket/k").unwrap(),
        "s3a://bucket/k"
    );
    assert_eq!(session.qualify_path("/a/b").unwrap(), "file:/a/b");
}

#[test]
fn max_records_per_file_reads_the_write_option() {
    assert_eq!(
        max_records_per_file(&BTreeMap::new(), "parquet").unwrap(),
        0
    );
    assert_eq!(
        max_records_per_file(&options(&[("maxRecordsPerFile", "10")]), "parquet").unwrap(),
        10
    );
    assert_eq!(
        max_records_per_file(&options(&[("MAXRECORDSPERFILE", " 7 ")]), "parquet").unwrap(),
        7
    );
    let e = max_records_per_file(&options(&[("maxRecordsPerFile", "ten")]), "parquet").unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "KRISHIV_INVALID_FORMAT_OPTION_VALUE"
    );
    let secret = BTreeMap::from([(
        "maxRecordsPerFile".to_owned(),
        OptionValue::reference("s", "k"),
    )]);
    let e = max_records_per_file(&secret, "parquet").unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "KRISHIV_INVALID_FORMAT_OPTION_VALUE"
    );
}

fn result_batch(paths: &[&str], sizes: &[i64], rows: &[i64]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8View, false),
        Field::new("size", DataType::Int64, false),
        Field::new("rows", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringViewArray::from_iter_values(paths.iter().copied())),
            Arc::new(Int64Array::from(sizes.to_vec())),
            Arc::new(Int64Array::from(rows.to_vec())),
        ],
    )
    .unwrap()
}

#[test]
fn committed_files_reads_path_size_and_rows() {
    let batch = result_batch(&["a/p", "a/q"], &[10, 20], &[1, 2]);
    let files = committed_files(std::slice::from_ref(&batch)).unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].path.as_ref(), "a/p");
    assert_eq!((files[0].size, files[0].rows), (10, 1));
    assert_eq!((files[1].size, files[1].rows), (20, 2));
    assert!(committed_files(&[]).unwrap().is_empty());
}

#[test]
fn committed_files_refuses_a_malformed_result() {
    let batch = result_batch(&["a/p"], &[10], &[1]);
    // Fewer columns.
    let narrow = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "path",
            DataType::Utf8View,
            false,
        )])),
        vec![Arc::clone(batch.column(0))],
    )
    .unwrap();
    assert!(
        committed_files(std::slice::from_ref(&narrow))
            .unwrap_err()
            .to_string()
            .contains("path, size and rows")
    );
    // A path that is not a string.
    let schema = Arc::new(Schema::new(vec![
        Field::new("path", DataType::Int32, false),
        Field::new("size", DataType::Int64, false),
        Field::new("rows", DataType::Int64, false),
    ]));
    let bad = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![10])),
            Arc::new(Int64Array::from(vec![1])),
        ],
    )
    .unwrap();
    assert!(
        committed_files(std::slice::from_ref(&bad))
            .unwrap_err()
            .to_string()
            .contains("not a string")
    );
    // Negative sizes and rows.
    let bad = result_batch(&["a/p"], &[-1], &[1]);
    assert!(
        committed_files(std::slice::from_ref(&bad))
            .unwrap_err()
            .to_string()
            .contains("bytes")
    );
    let bad = result_batch(&["a/p"], &[10], &[-1]);
    assert!(
        committed_files(std::slice::from_ref(&bad))
            .unwrap_err()
            .to_string()
            .contains("rows")
    );
}
