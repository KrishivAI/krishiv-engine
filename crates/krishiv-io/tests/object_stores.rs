// SPDX-License-Identifier: Apache-2.0
//! Parquet round-trips through cloud object stores, against emulators
//! (P2-T084).
//!
//! Each test runs the full round-trip when its endpoint variable is set,
//! and otherwise asserts the offline behaviour (store construction touches
//! no network; misconfiguration fails with the exact config error). The
//! tests never start or stop emulators; point them at one that is already
//! running and create the bucket or container first:
//!
//! - MinIO: `KRISHIV_TEST_S3_ENDPOINT=http://127.0.0.1:9000`, credentials
//!   `minioadmin`/`minioadmin`, bucket `krishiv-test`
//!   (`mc mb local/krishiv-test`).
//! - fake-gcs-server: `KRISHIV_TEST_GCS_ENDPOINT=http://127.0.0.1:4443`,
//!   bucket `krishiv-test` (the test creates it through the emulator's
//!   unauthenticated JSON API when it can, as its documented `curl` flow
//!   does).
//! - Azurite: `KRISHIV_TEST_AZURE_ENDPOINT=http://127.0.0.1:10000/devstoreaccount1`,
//!   container `krishiv-test`
//!   (`az storage container create --name krishiv-test --connection-string ...`).
//!
//! Every byte goes through the layer's wrappers ([`IoContext`](krishiv_io::IoContext)):
//! the round-trip writes with [`put_object`](krishiv_io::IoContext::put_object),
//! reads back with [`get_ranges`](krishiv_io::IoContext::get_ranges) after a
//! [`head`](krishiv_io::IoContext::head), and lists the file's directory.
#![expect(clippy::unwrap_used, reason = "test code")]
#![expect(
    clippy::single_range_in_vec_init,
    reason = "a whole-file read is the case under test"
)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;
use krishiv_io::{FileRef, IoConfig, IoScheduler, StoreRegistry};
use object_store::path::Path;

/// A two-column Parquet file: ids `1, 2, 3` and names `a, NULL, c`.
fn parquet_bytes() -> Vec<u8> {
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap();
    let mut buf = Vec::new();
    let mut writer =
        ArrowWriter::try_new(&mut buf, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    buf
}

/// The `(id, name)` columns of a Parquet file.
fn read_parquet(bytes: &[u8]) -> (Vec<i32>, Vec<Option<String>>) {
    use arrow_array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes.to_vec()))
        .unwrap()
        .build()
        .unwrap();
    let mut ids = Vec::new();
    let mut names: Vec<Option<String>> = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let name = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        ids.extend((0..batch.num_rows()).map(|i| id.value(i)));
        names.extend((0..batch.num_rows()).map(|i| {
            if name.is_null(i) {
                None
            } else {
                Some(name.value(i).to_owned())
            }
        }));
    }
    (ids, names)
}

/// A fresh directory per run, so parallel runs never share a file.
fn unique_dir() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("krishiv-test/rw-{}-{nanos}", std::process::id())
}

fn test_config(pairs: &[(&str, &str)]) -> Arc<Config> {
    let mut full = vec![("krishiv.io.requestTimeout", "10s")];
    full.extend(pairs.iter().copied());
    let (config, _) = Config::from_pairs(full).unwrap();
    Arc::new(config)
}

/// Writes a Parquet file under `bucket_url` (`scheme://bucket`) and reads it
/// back: same bytes, same rows, visible in a listing, deletable.
async fn write_then_read(config: Arc<Config>, bucket_url: &str) {
    let dir = unique_dir();
    let url = format!("{}/{dir}/t.parquet", bucket_url.trim_end_matches('/'));
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let registry = StoreRegistry::new(config);
    let (store, path, display) = registry.ensure(&sched, &url).unwrap();
    assert_eq!(display, url);
    let io = sched.context();
    let bytes = Bytes::from(parquet_bytes());
    io.put_object(&store, &path, bytes.clone()).await.unwrap();
    let meta = io.head(&store, &path).await.unwrap().unwrap();
    let file = FileRef::from_meta(store.clone(), &meta);
    let got = io.get_ranges(&file, &[0..file.size]).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(&got[0], &bytes);
    let (ids, names) = read_parquet(&got[0]);
    assert_eq!(ids, vec![1, 2, 3]);
    assert_eq!(
        names,
        vec![Some("a".to_owned()), None, Some("c".to_owned())]
    );
    let listed = io.list(&store, &Path::from(dir.as_str())).await.unwrap();
    assert!(
        listed.iter().any(|m| m.location.as_ref() == path.as_ref()),
        "the file is listed under {dir}"
    );
    io.delete(&store, &path).await.unwrap();
    assert!(io.head(&store, &path).await.unwrap().is_none());
}

/// The non-empty value of an emulator endpoint variable, if set.
fn endpoint(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Creates `bucket` on a fake-gcs-server through its unauthenticated JSON
/// API (its documented `curl` flow). Any failure is ignored: the bucket may
/// exist, and the upload below reports what the server refuses.
fn ensure_gcs_bucket(endpoint: &str, bucket: &str) {
    let host = endpoint
        .trim_end_matches('/')
        .strip_prefix("http://")
        .unwrap_or(endpoint)
        .split('/')
        .next()
        .unwrap_or_default();
    if host.is_empty() {
        return;
    }
    let body = format!(r#"{{"name":"{bucket}"}}"#);
    let request = format!(
        "POST /storage/v1/b?project=test-project HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let Ok(stream) = std::net::TcpStream::connect(host) else {
        return;
    };
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    use std::io::{Read as _, Write as _};
    let mut stream = stream;
    if stream.write_all(request.as_bytes()).is_err() {
        return;
    }
    let mut head = [0_u8; 12];
    let _ = stream.read_exact(&mut head);
}

#[tokio::test]
async fn s3_round_trip() {
    let Some(endpoint) = endpoint("KRISHIV_TEST_S3_ENDPOINT") else {
        // No emulator in this environment: store construction is still
        // offline (no I/O) and misconfiguration still fails with the exact
        // config error.
        let config = test_config(&[
            ("fs.s3a.endpoint", "http://127.0.0.1:1"),
            ("fs.s3a.access.key", "minioadmin"),
            ("fs.s3a.secret.key", "minioadmin"),
            ("fs.s3a.path.style.access", "true"),
            ("fs.s3a.endpoint.region", "us-east-1"),
        ]);
        let sched = IoScheduler::new(IoConfig::from_config(&config));
        let registry = StoreRegistry::new(config);
        let (_, _, display) = registry.ensure(&sched, "s3a://krishiv-test/k").unwrap();
        assert_eq!(display, "s3a://krishiv-test/k");
        let bad = test_config(&[("fs.s3a.path.style.access", "banana")]);
        let bad_sched = IoScheduler::new(IoConfig::from_config(&bad));
        let bad_registry = StoreRegistry::new(bad);
        let err = bad_registry
            .ensure(&bad_sched, "s3a://krishiv-test/k")
            .unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::KrishivInvalidConfigValue);
        return;
    };
    write_then_read(
        test_config(&[
            ("fs.s3a.endpoint", endpoint.as_str()),
            ("fs.s3a.access.key", "minioadmin"),
            ("fs.s3a.secret.key", "minioadmin"),
            ("fs.s3a.path.style.access", "true"),
            ("fs.s3a.endpoint.region", "us-east-1"),
        ]),
        "s3a://krishiv-test",
    )
    .await;
}

#[tokio::test]
async fn gcs_round_trip() {
    let Some(endpoint) = endpoint("KRISHIV_TEST_GCS_ENDPOINT") else {
        // No emulator in this environment: store construction is still
        // offline (no I/O) and conflicting credentials are still refused.
        let config = test_config(&[
            ("fs.gs.storage.root.url", "http://127.0.0.1:1"),
            (
                "fs.gs.auth.service.account.json",
                r#"{"private_key": "key", "private_key_id": "id", "client_email": "test@example.com", "disable_oauth": true}"#,
            ),
        ]);
        let sched = IoScheduler::new(IoConfig::from_config(&config));
        let registry = StoreRegistry::new(config);
        let (_, _, display) = registry.ensure(&sched, "gs://krishiv-test/k").unwrap();
        assert_eq!(display, "gs://krishiv-test/k");
        let bad = test_config(&[
            (
                "fs.gs.auth.service.account.json.keyfile",
                "/tmp/krishiv-test-key.json",
            ),
            (
                "fs.gs.auth.service.account.json",
                r#"{"private_key": "key", "private_key_id": "id", "client_email": "test@example.com", "disable_oauth": true}"#,
            ),
        ]);
        let bad_sched = IoScheduler::new(IoConfig::from_config(&bad));
        let bad_registry = StoreRegistry::new(bad);
        bad_registry
            .ensure(&bad_sched, "gs://krishiv-test/k")
            .unwrap_err();
        return;
    };
    ensure_gcs_bucket(&endpoint, "krishiv-test");
    write_then_read(
        test_config(&[
            ("fs.gs.storage.root.url", endpoint.as_str()),
            (
                "fs.gs.auth.service.account.json",
                r#"{"private_key": "key", "private_key_id": "id", "client_email": "test@example.com", "disable_oauth": true}"#,
            ),
        ]),
        "gs://krishiv-test",
    )
    .await;
}

#[tokio::test]
async fn azure_round_trip() {
    let Some(endpoint) = endpoint("KRISHIV_TEST_AZURE_ENDPOINT") else {
        // No emulator in this environment: store construction is still
        // offline (no I/O) and a malformed SAS token is still refused with
        // the exact request error.
        let config = test_config(&[
            ("fs.azure.endpoint", "http://127.0.0.1:1/devstoreaccount1"),
            ("fs.azure.account.name", "devstoreaccount1"),
            (
                "fs.azure.account.key.devstoreaccount1",
                "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
            ),
        ]);
        let sched = IoScheduler::new(IoConfig::from_config(&config));
        let registry = StoreRegistry::new(config);
        let (_, _, display) = registry
            .ensure(
                &sched,
                "abfss://krishiv-test@devstoreaccount1.dfs.core.windows.net/k",
            )
            .unwrap();
        assert_eq!(
            display,
            "abfss://krishiv-test@devstoreaccount1.dfs.core.windows.net/k"
        );
        let bad = test_config(&[
            ("fs.azure.account.name", "acct"),
            ("fs.azure.sas.cont.acct", "not-a-sas-token"),
        ]);
        let bad_sched = IoScheduler::new(IoConfig::from_config(&bad));
        let bad_registry = StoreRegistry::new(bad);
        let err = bad_registry
            .ensure(&bad_sched, "abfss://cont@acct.dfs.core.windows.net/k")
            .unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivObjectStoreRequestFailed
        );
        return;
    };
    write_then_read(
        test_config(&[
            ("fs.azure.endpoint", endpoint.as_str()),
            ("fs.azure.account.name", "devstoreaccount1"),
            (
                "fs.azure.account.key.devstoreaccount1",
                "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
            ),
        ]),
        "abfss://krishiv-test@devstoreaccount1.dfs.core.windows.net",
    )
    .await;
}

/// The commit log relies on `PutMode::Create` (`If-None-Match: *`): of two
/// creates of one object, the second must fail as "exists", and the first
/// writer's bytes must stay (P2-T082b).
#[tokio::test]
async fn s3_put_if_absent_is_exclusive() {
    let Some(endpoint) = endpoint("KRISHIV_TEST_S3_ENDPOINT") else {
        return;
    };
    let config = test_config(&[
        ("fs.s3a.endpoint", endpoint.as_str()),
        ("fs.s3a.access.key", "minioadmin"),
        ("fs.s3a.secret.key", "minioadmin"),
        ("fs.s3a.path.style.access", "true"),
        ("fs.s3a.endpoint.region", "us-east-1"),
    ]);
    let url = format!("s3a://krishiv-test/{}/log/0.json", unique_dir());
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let registry = StoreRegistry::new(config);
    let (store, path, _) = registry.ensure(&sched, &url).unwrap();
    let io = sched.context();
    let first = io
        .put_if_absent(&store, &path, Bytes::from_static(b"one"))
        .await
        .unwrap();
    let second = io
        .put_if_absent(&store, &path, Bytes::from_static(b"two"))
        .await
        .unwrap();
    assert_eq!(first, krishiv_io::PutIfAbsent::Created);
    assert_eq!(second, krishiv_io::PutIfAbsent::Exists);
    let got = io.get_object(&store, &path).await.unwrap().unwrap();
    assert_eq!(&got[..], b"one");
    io.delete(&store, &path).await.unwrap();
}
