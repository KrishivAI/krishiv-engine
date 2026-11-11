// SPDX-License-Identifier: Apache-2.0
//! `query_registry` (ADR 0044 S4): two per-query registries on one process-wide `IoScheduler`,
//! with different credential sources for the same table, each read with their own credentials;
//! both share one HTTP client; dropping one unregisters its scoped ids and leaves the other working. Runs against a
//! loopback S3 stand-in that answers 404 and records each request's `Authorization` header.
//! `krishiv.io.credentials.refreshSkew` is read from the config.
#![cfg(feature = "cloud")]
#![expect(clippy::unwrap_used, reason = "test code")]

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use krishiv_common::config::Config;
use krishiv_io::{IoConfig, IoScheduler, query_registry, register};
use krishiv_store::{CredentialScope, CredentialSource, ScopedCredential};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Debug)]
struct Fixed(&'static str);

#[async_trait]
impl CredentialSource for Fixed {
    async fn fetch(&self, _scope: &CredentialScope) -> krishiv_store::Result<ScopedCredential> {
        Ok(ScopedCredential::s3(
            format!("AKIA{}KEY", self.0),
            "secret",
            None,
            Some(SystemTime::now() + Duration::from_secs(3600)),
        ))
    }
}

/// Answers every request with a 404 and records its `Authorization` header.
async fn serve() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let log = Arc::clone(&seen);
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let log = Arc::clone(&log);
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..n]);
                }
                let text = String::from_utf8_lossy(&request).to_string();
                let auth = text
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
                    .unwrap_or_default()
                    .to_owned();
                log.lock().unwrap().push(auth);
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (endpoint, seen)
}

fn config(endpoint: &str, skew: &str) -> Arc<Config> {
    let (config, _) = Config::from_pairs([
        ("fs.s3a.endpoint", endpoint),
        ("fs.s3a.region", "us-east-1"),
        ("krishiv.io.credentials.refreshSkew", skew),
        ("krishiv.io.maxRetries", "0"),
    ])
    .unwrap();
    Arc::new(config)
}

#[tokio::test(flavor = "multi_thread")]
async fn two_queries_on_one_scheduler_read_with_their_own_credentials() {
    let (endpoint, seen) = serve().await;
    let config = config(&endpoint, "0s");
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let scope = || CredentialScope::new("s3://b/warehouse/t/").unwrap();
    let one =
        query_registry(Arc::clone(&config), &sched).with_source(scope(), Arc::new(Fixed("ONE")));
    let two =
        query_registry(Arc::clone(&config), &sched).with_source(scope(), Arc::new(Fixed("TWO")));

    let url = "s3://b/warehouse/t/f.parquet";
    let (id_one, key, _) = register(&sched, &one, url).unwrap();
    let clients = sched.http_clients().len();
    assert!(
        clients >= 1,
        "the first query's store built a shared client"
    );
    let (id_two, _, _) = register(&sched, &two, url).unwrap();
    assert_ne!(id_one.key(), id_two.key(), "one scheduler entry per query");
    assert_eq!(
        sched.http_clients().len(),
        clients,
        "the second query's store reuses the first's HTTP client"
    );
    let io = sched.context();
    assert!(io.head(&id_one, &key).await.unwrap().is_none());
    assert!(io.head(&id_two, &key).await.unwrap().is_none());
    {
        let seen = seen.lock().unwrap();
        assert!(seen.iter().any(|a| a.contains("AKIAONEKEY")), "{seen:?}");
        assert!(seen.iter().any(|a| a.contains("AKIATWOKEY")), "{seen:?}");
    }

    // Query one ends: its scoped id leaves the scheduler, the other keeps working.
    drop(one);
    let gone = io.head(&id_one, &key).await.unwrap_err();
    assert!(gone.to_string().contains("not registered"), "{gone}");
    assert!(io.head(&id_two, &key).await.unwrap().is_none());
    drop(two);
    assert!(io.head(&id_two, &key).await.is_err());
}

#[test]
fn the_refresh_skew_is_a_typed_duration() {
    assert!(Config::from_pairs([("krishiv.io.credentials.refreshSkew", "10m")]).is_ok());
    assert!(Config::from_pairs([("krishiv.io.credentials.refreshSkew", "soon")]).is_err());
}
