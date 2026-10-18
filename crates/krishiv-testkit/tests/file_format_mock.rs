// SPDX-License-Identifier: Apache-2.0
//! The mock format passes the generic `FileFormat` conformance suite, with
//! the sans-IO decoder and through the async read-at bridge.

use krishiv_testkit::file_format::MockFixture;

krishiv_testkit::file_format_conformance!(mock, MockFixture::default());
krishiv_testkit::file_format_conformance!(
    mock_async,
    MockFixture {
        async_decoder: true
    }
);

/// A format without its own schema is read with the table's.
#[test]
fn files_without_a_schema_are_read_with_the_table_schema() {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use krishiv_io::{ScanOptions, ScanRequest};
    use krishiv_testkit::file_format::{Env, FormatFixture, data};

    #[derive(Debug)]
    struct Schemaless;
    impl FormatFixture for Schemaless {
        fn format(&self) -> Arc<dyn krishiv_io::FileFormat> {
            Arc::new(krishiv_testkit::file_format::mock::MockFormat)
        }
        fn options(&self) -> Vec<(String, String)> {
            vec![("selfDescribing".into(), "false".into())]
        }
        fn unit_rows(&self, rows: usize) -> Vec<(String, String)> {
            vec![("batchRows".into(), rows.to_string())]
        }
    }
    let env = Env::new(None).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let batch = data::batch(&schema, 10, 1).unwrap();
    let file = env
        .write(&Schemaless, "s/f", &schema, &[batch], &[])
        .unwrap();
    let got = env
        .scan(
            &Schemaless,
            &file,
            &ScanRequest::all(schema),
            None,
            ScanOptions::default(),
        )
        .unwrap();
    assert_eq!(
        got.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        10
    );
    assert!(krishiv_testkit::file_format::schema_evolution(&Schemaless).is_ok());
}

/// Schema inference through the cache, and a scan's spill path (dropping
/// prefetched bytes) with its accessors.
#[test]
fn schema_inference_and_prefetch_release() {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use krishiv_io::{ByteRange, ScanOptions, ScanRequest, ScanStream};
    use krishiv_memory::MemoryPool;
    use krishiv_testkit::file_format::{Env, FormatFixture, data};

    let fx = MockFixture::default();
    let env = Env::new(None).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let batch = data::batch(&schema, 5000, 3).unwrap();
    let file = env
        .write(&fx, "p/f", &schema, &[batch], &fx.unit_rows(100))
        .unwrap();
    let o = env.options(&fx, &[]).unwrap();
    let format = fx.format();
    let inferred = env
        .block_on(format.infer_schema(&env.io, std::slice::from_ref(&file), &o))
        .unwrap();
    assert_eq!(inferred.fields().len(), 1);
    assert!(env.block_on(format.infer_schema(&env.io, &[], &o)).is_err());
    let req = ScanRequest::all(Arc::clone(&schema));
    let rows = env.block_on(async {
        let mut s = ScanStream::open(
            &env.io,
            format.as_ref(),
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            &o,
            env.reservation("s"),
            ScanOptions::default(),
        )
        .await
        .unwrap();
        let mut rows = s.next_batch().await.unwrap().unwrap().num_rows();
        assert!(s.reserved() > 0, "prefetch is reserved");
        let freed = s.release_prefetch();
        assert!(freed > 0);
        assert_eq!(s.file().path, file.path);
        assert!(s.mapper().is_some());
        assert!(format!("{s:?}").contains("ScanStream"));
        let _ = (s.io(), s.metrics());
        while let Some(b) = s.next_batch().await.unwrap() {
            rows += b.num_rows();
        }
        assert_eq!(s.reserved(), 0);
        assert!(s.next_batch().await.unwrap().is_none());
        rows
    });
    assert_eq!(rows, 5000);
    assert_eq!(env.root.reserved(), 0);
}

/// The predicate generator and oracle handle empty batches and read nullness
/// from the array.
#[test]
fn predicate_helpers_on_edge_batches() {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use krishiv_io::{ColumnPath, PushdownExpr};
    use krishiv_testkit::file_format::data;

    let schema = Arc::new(Schema::new(vec![
        Field::new("s", DataType::Utf8View, true),
        Field::new("b", DataType::Boolean, true),
    ]));
    let empty = data::batch(&schema, 0, 1).unwrap();
    let mut r = data::Rng::new(1);
    for _ in 0..50 {
        let _ = data::predicate(&empty, &["s", "b"], &mut r, 1);
    }
    assert!(data::literal(empty.column(0), &mut r).is_none());
    let rows = data::batch(&schema, 50, 2).unwrap();
    let not_null = PushdownExpr::IsNotNull(ColumnPath::new("b"));
    let valid = (0..50)
        .filter(|&i| data::eval(&not_null, &rows, i) == Some(true))
        .count();
    assert_eq!(
        valid,
        50 - arrow_array::Array::null_count(rows.column(1).as_ref())
    );
}

/// `ignoreCorruptFiles` skips corrupt files only: a storage failure that
/// outlasts the retries is an error, with its own class.
#[test]
fn ignore_corrupt_files_never_hides_a_storage_failure() {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use krishiv_common::error::ErrorClass;
    use krishiv_io::{ScanOptions, ScanRequest};
    use krishiv_testkit::file_format::{Env, FormatFixture, data};

    let fx = MockFixture::default();
    let env = Env::new(None).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let batch = data::batch(&schema, 100, 1).unwrap();
    let file = env
        .write(&fx, "s/f", &schema, &[batch], &fx.unit_rows(10))
        .unwrap();
    env.store.fail_next_gets(1000, None);
    let ignore = ScanOptions {
        ignore_corrupt_files: true,
        ..ScanOptions::default()
    };
    let e = env
        .scan(&fx, &file, &ScanRequest::all(schema), None, ignore)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivObjectStoreRequestFailed);
}
