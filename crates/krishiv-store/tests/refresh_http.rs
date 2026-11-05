// SPDX-License-Identifier: Apache-2.0
//! The whole path against a loopback S3 stand-in: a scoped store signs with the source's
//! credential, a `403 ExpiredToken` forces exactly one refresh and one retry (signed with the
//! new key id), and any other 403 fails with no refresh.
#![cfg(feature = "cloud")]
#![expect(clippy::unwrap_used, reason = "test code")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use krishiv_store::{CredentialScope, CredentialSource, ScopedCredential, StoreRegistry};
use object_store::{ObjectStoreExt, path::Path};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Debug, Default)]
struct Counting(AtomicUsize);

#[async_trait]
impl CredentialSource for Counting {
    async fn fetch(&self, _scope: &CredentialScope) -> krishiv_store::Result<ScopedCredential> {
        let n = self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ScopedCredential::s3(
            format!("AKIAKEY{n}XYZ"),
            "secret",
            None,
            Some(SystemTime::now() + Duration::from_secs(3600)),
        ))
    }
}

/// Serves the scripted answers in order, one per request; returns the Authorization header of
/// every request it saw.
async fn serve(
    answers: Vec<(&'static str, &'static str)>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut seen = Vec::new();
        for (status, body) in answers {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut buf).await.unwrap();
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
            seen.push(auth);
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            let _ = socket.shutdown().await;
        }
        seen
    });
    (endpoint, task)
}

fn registry(endpoint: &str, source: Arc<Counting>) -> StoreRegistry {
    let mut settings = std::collections::BTreeMap::new();
    settings.insert("fs.s3a.endpoint".to_owned(), endpoint.to_owned());
    settings.insert("fs.s3a.region".to_owned(), "us-east-1".to_owned());
    StoreRegistry::new(Arc::new(settings))
        .with_source(CredentialScope::new("s3://b/warehouse/").unwrap(), source)
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expired_token_forces_one_refresh_and_one_retry() {
    let (endpoint, server) = serve(vec![
        (
            "403 Forbidden",
            "<Error><Code>ExpiredToken</Code><Message>The provided token has expired.</Message></Error>",
        ),
        ("200 OK", "hello"),
    ])
    .await;
    let source = Arc::new(Counting::default());
    let registry = registry(&endpoint, Arc::clone(&source));
    let opened = registry.open("s3://b/warehouse/f.txt").unwrap();
    let bytes = opened
        .store
        .get(&opened.key)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(bytes.as_ref(), b"hello");
    assert_eq!(
        source.0.load(Ordering::SeqCst),
        2,
        "one fetch, one forced refresh"
    );
    let seen = server.await.unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen[0].contains("AKIAKEY0XYZ"), "{}", seen[0]);
    assert!(seen[1].contains("AKIAKEY1XYZ"), "{}", seen[1]);
}

#[tokio::test(flavor = "multi_thread")]
async fn any_other_403_fails_without_a_refresh() {
    let (endpoint, server) = serve(vec![(
        "403 Forbidden",
        "<Error><Code>AccessDenied</Code></Error>",
    )])
    .await;
    let source = Arc::new(Counting::default());
    let registry = registry(&endpoint, Arc::clone(&source));
    let opened = registry.open("s3://b/warehouse/f.txt").unwrap();
    let e = opened
        .store
        .head(&Path::from("warehouse/f.txt"))
        .await
        .unwrap_err();
    assert!(
        matches!(e, object_store::Error::PermissionDenied { .. }),
        "{e:?}"
    );
    assert_eq!(source.0.load(Ordering::SeqCst), 1);
    assert_eq!(server.await.unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failing_source_fails_the_request_naming_the_scope() {
    #[derive(Debug)]
    struct Down;
    #[async_trait]
    impl CredentialSource for Down {
        async fn fetch(&self, _: &CredentialScope) -> krishiv_store::Result<ScopedCredential> {
            Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: "token service down".to_owned(),
            }
            .into())
        }
    }
    let mut settings = std::collections::BTreeMap::new();
    settings.insert(
        "fs.s3a.endpoint".to_owned(),
        "http://127.0.0.1:9".to_owned(),
    );
    let registry = StoreRegistry::new(Arc::new(settings)).with_source(
        CredentialScope::new("s3://b/warehouse/").unwrap(),
        Arc::new(Down),
    );
    let opened = registry.open("s3://b/warehouse/f.txt").unwrap();
    let e = opened.store.head(&opened.key).await.unwrap_err();
    let shown = e.to_string();
    assert!(shown.contains("s3://b/warehouse"), "{shown}");
}

/// Answers every request with `503` and counts them.
async fn serve_unavailable() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&count);
    let task = tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            seen.fetch_add(1, Ordering::SeqCst);
            let _ = socket
                .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            let _ = socket.shutdown().await;
        }
    });
    (endpoint, count, task)
}

async fn attempts_under(policy: krishiv_store::RetryPolicy) -> usize {
    let (endpoint, count, server) = serve_unavailable().await;
    let source = Arc::new(Counting::default());
    let registry = registry(&endpoint, source).with_retry_policy(policy);
    let opened = registry.open("s3://b/warehouse/f.txt").unwrap();
    assert!(opened.store.get(&opened.key).await.is_err());
    server.abort();
    count.load(Ordering::SeqCst)
}

#[tokio::test(flavor = "multi_thread")]
async fn the_retry_policy_bounds_the_attempts_of_the_client() {
    use krishiv_store::RetryPolicy;
    assert_eq!(attempts_under(RetryPolicy::NONE).await, 1);
    let two = RetryPolicy {
        max_retries: 2,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(2),
        ..RetryPolicy::DEFAULT
    };
    assert_eq!(
        attempts_under(two).await,
        3,
        "the first try and two retries"
    );
}
