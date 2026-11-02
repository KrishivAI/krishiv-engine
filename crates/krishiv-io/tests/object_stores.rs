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

/// A one-file web server on loopback that honours `Range`, logging each
/// request line and its `Range` header.
/// (request line, `Range` header, `Authorization` header) per request.
type RequestLog = Arc<std::sync::Mutex<Vec<(String, Option<String>, Option<String>)>>>;

struct RangeServer {
    port: u16,
    log: RequestLog,
}

fn serve(body: Vec<u8>) -> RangeServer {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = Arc::clone(&log);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let request = line.trim().to_owned();
                let mut range = None;
                let mut auth = None;
                loop {
                    let mut h = String::new();
                    reader.read_line(&mut h).unwrap();
                    let h = h.trim();
                    if h.is_empty() {
                        break;
                    }
                    if let Some(v) = h.to_ascii_lowercase().strip_prefix("range:") {
                        range = Some(v.trim().to_owned());
                    }
                    if h.to_ascii_lowercase().starts_with("authorization:") {
                        auth = h.split_once(':').map(|(_, v)| v.trim().to_owned());
                    }
                }
                seen.lock()
                    .unwrap()
                    .push((request.clone(), range.clone(), auth.clone()));
                let head = request.starts_with("HEAD");
                let (status, slice) = match range.as_deref().and_then(|r| r.strip_prefix("bytes="))
                {
                    Some(r) => {
                        let (a, b) = r.split_once('-').unwrap();
                        let a: usize = a.parse().unwrap();
                        let b: usize = b.parse::<usize>().unwrap().min(body.len() - 1);
                        (
                            format!(
                                "206 Partial Content\r\nContent-Range: bytes {a}-{b}/{}",
                                body.len()
                            ),
                            &body[a..=b],
                        )
                    }
                    None => ("200 OK".to_owned(), &body[..]),
                };
                let length = if head { body.len() } else { slice.len() };
                let mut out = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {length}\r\nAccept-Ranges: bytes\r\nLast-Modified: Tue, 03 Nov 2026 00:00:00 GMT\r\nConnection: keep-alive\r\n\r\n"
                )
                .into_bytes();
                if !head {
                    out.extend_from_slice(slice);
                }
                if stream.write_all(&out).is_err() {
                    break;
                }
            }
        }
    });
    RangeServer { port, log }
}

/// An `http://` URL to a loopback server reads like a file: `HEAD` for the
/// size, then only `Range` requests (P2-T195 unit 1).
#[tokio::test]
async fn https_store_reads_ranges_only() {
    let body = parquet_bytes();
    let server = serve(body.clone());
    let config = test_config(&[]);
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let registry = StoreRegistry::new(config);
    let url = format!("http://127.0.0.1:{}/data/t.parquet", server.port);
    let (store, path, display) = registry.ensure(&sched, &url).unwrap();
    assert_eq!(display, url);
    let io = sched.context();
    let meta = io.head(&store, &path).await.unwrap().unwrap();
    assert_eq!(meta.size, body.len() as u64);
    let file = FileRef::from_meta(store.clone(), &meta);
    let got = io.get_ranges(&file, &[0..file.size]).await.unwrap();
    assert_eq!(&got[0][..], &body[..]);
    let (ids, _) = read_parquet(&got[0]);
    assert_eq!(ids, vec![1, 2, 3]);
    for (request, range, _) in server.log.lock().unwrap().iter() {
        assert!(
            request.starts_with("HEAD") || range.is_some(),
            "a whole-file GET: {request}"
        );
    }
}

#[test]
fn http_urls_name_a_store_by_authority_and_refuse_credentials_and_remote_plain_http() {
    let ok = krishiv_io::parse_object_url("https://example.com:8443/a/b.parquet").unwrap();
    assert_eq!(ok.store_id.key(), "https://example.com:8443");
    assert_eq!(ok.key.as_ref(), "a/b.parquet");
    assert!(krishiv_io::parse_object_url("https://user:pw@example.com/x").is_err());
    assert!(krishiv_io::parse_object_url("https://example.com/x?token=1").is_err());
    let config = test_config(&[]);
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let registry = StoreRegistry::new(config);
    let e = registry
        .ensure(&sched, "http://example.com/x.parquet")
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// Reads `url` whole through a registry with `pairs` and checks the bytes;
/// returns the display path and the registry's `Debug` text.
async fn read_hf(body: &[u8], pairs: &[(&str, &str)], url: &str) -> (String, String) {
    let config = test_config(pairs);
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let registry = StoreRegistry::new(config);
    let (store, path, display) = registry.ensure(&sched, url).unwrap();
    let io = sched.context();
    let meta = io.head(&store, &path).await.unwrap().unwrap();
    let file = FileRef::from_meta(store, &meta);
    let got = io.get_ranges(&file, &[0..file.size]).await.unwrap();
    assert_eq!(&got[0][..], body);
    (display, format!("{registry:?}"))
}

/// `hf://datasets/<org>/<name>[@rev]/<path>` resolves to the `resolve`
/// endpoint, sends the token as a bearer header only when one is set, and
/// reads the same bytes as `https://` (P2-T195 unit 2).
#[tokio::test]
async fn hf_paths_read_through_the_resolve_endpoint_with_a_bearer_token() {
    let body = parquet_bytes();
    let server = serve(body.clone());
    let endpoint = format!("http://127.0.0.1:{}", server.port);
    let (display, debug) = read_hf(
        &body,
        &[
            ("fs.hf.endpoint", endpoint.as_str()),
            ("fs.hf.token", "hf_Canary-7"),
        ],
        "hf://datasets/acme/tiny@v2/data/t.parquet",
    )
    .await;
    assert_eq!(display, "hf://datasets/acme/tiny@v2/data/t.parquet");
    assert!(!debug.contains("Canary-7"), "{debug}");
    read_hf(
        &body,
        &[("fs.hf.endpoint", endpoint.as_str())],
        "hf://datasets/acme/tiny/data/t.parquet",
    )
    .await;
    let log = server.log.lock().unwrap();
    let first = log.iter().find(|(r, _, _)| r.starts_with("HEAD")).unwrap();
    assert!(
        first
            .0
            .contains("/datasets/acme/tiny/resolve/v2/data/t.parquet"),
        "{}",
        first.0
    );
    assert_eq!(first.2.as_deref(), Some("Bearer hf_Canary-7"));
    let tokenless: Vec<_> = log
        .iter()
        .filter(|(r, _, _)| r.contains("/resolve/main/"))
        .collect();
    assert!(!tokenless.is_empty());
    assert!(tokenless.iter().all(|(_, _, a)| a.is_none()));
    drop(log);
    for bad in [
        "hf://models/a/b/c",
        "hf://datasets/a",
        "hf://datasets/a b/c/d",
    ] {
        assert!(krishiv_io::parse_object_url(bad).is_err(), "{bad}");
    }
}
