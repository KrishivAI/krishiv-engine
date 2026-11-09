// SPDX-License-Identifier: Apache-2.0
//! Tests of the shared HTTP client cache: hits, distinct options, the bound and the fingerprint.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use object_store::client::{
    ClientConfigKey, ClientOptions, HttpClient, HttpConnector, ReqwestConnector,
};

use super::{KEYS, MAX_CLIENTS, SharedHttpClients, fingerprint};
use crate::StoreRegistry;

/// Builds real reqwest clients and counts the builds.
#[derive(Debug, Default)]
struct Counting(AtomicUsize);

impl HttpConnector for Counting {
    fn connect(&self, options: &ClientOptions) -> object_store::Result<HttpClient> {
        self.0.fetch_add(1, Ordering::SeqCst);
        ReqwestConnector::default().connect(options)
    }
}

fn counting() -> (Arc<Counting>, SharedHttpClients) {
    let counter = Arc::new(Counting::default());
    let clients = SharedHttpClients::with_connector(Arc::clone(&counter) as Arc<dyn HttpConnector>);
    (counter, clients)
}

fn builds(counter: &Counting) -> usize {
    counter.0.load(Ordering::SeqCst)
}

#[test]
fn equal_options_build_one_client() {
    let (counter, clients) = counting();
    let options = ClientOptions::new().with_allow_http(true);
    for _ in 0..3 {
        clients.connect(&options).unwrap();
        clients.clone().connect(&options.clone()).unwrap();
    }
    assert_eq!(builds(&counter), 1);
    assert_eq!(clients.len(), 1);
}

#[test]
fn options_that_differ_get_their_own_clients() {
    let (counter, clients) = counting();
    let base = ClientOptions::new();
    let sets = [
        base.clone(),
        base.clone().with_allow_http(true),
        base.clone().with_config(ClientConfigKey::Timeout, "7s"),
        base.with_config(ClientConfigKey::ProxyUrl, "http://127.0.0.1:3128"),
    ];
    for set in &sets {
        clients.connect(set).unwrap();
    }
    assert_eq!(builds(&counter), sets.len());
    for set in &sets {
        clients.connect(set).unwrap();
    }
    assert_eq!(builds(&counter), sets.len(), "the second round is all hits");
}

#[test]
fn past_the_bound_clients_are_built_uncached_and_still_work() {
    let (counter, clients) = counting();
    let set = |i: usize| {
        ClientOptions::new().with_config(ClientConfigKey::Timeout, format!("{}s", i + 1))
    };
    for i in 0..MAX_CLIENTS {
        clients.connect(&set(i)).unwrap();
    }
    assert_eq!(clients.len(), MAX_CLIENTS);
    let extra = set(MAX_CLIENTS);
    clients.connect(&extra).unwrap();
    clients.connect(&extra).unwrap();
    assert_eq!(clients.len(), MAX_CLIENTS);
    assert_eq!(builds(&counter), MAX_CLIENTS + 2);
    // A cached set is still a hit.
    clients.connect(&set(0)).unwrap();
    assert_eq!(builds(&counter), MAX_CLIENTS + 2);
}

#[test]
fn a_failed_build_is_not_cached() {
    let (counter, clients) = counting();
    // An unparseable timeout fails in the reqwest build.
    let bad = ClientOptions::new().with_config(ClientConfigKey::Timeout, "soon");
    assert!(clients.connect(&bad).is_err());
    assert!(clients.connect(&bad).is_err());
    assert_eq!(builds(&counter), 2);
    assert!(clients.is_empty());
}

/// A non-default value for every key, so a key missing from the fingerprint shows up.
fn changed(key: ClientConfigKey) -> ClientOptions {
    use ClientConfigKey as K;
    // Booleans flip their default (some default to true).
    let default = ClientOptions::new().get_config_value(&key);
    let flipped = match default.as_deref() {
        Some("true") => Some("false"),
        Some("false") => Some("true"),
        _ => None,
    };
    let value = flipped.unwrap_or(match key {
        K::ConnectTimeout
        | K::ReadTimeout
        | K::Http2KeepAliveInterval
        | K::Http2KeepAliveTimeout
        | K::PoolIdleTimeout
        | K::Timeout => "17s",
        K::Http2MaxFrameSize => "32768",
        K::PoolMaxIdlePerHost => "3",
        K::DefaultContentType => "text/plain",
        K::ProxyUrl => "http://127.0.0.1:3128",
        K::ProxyCaCertificate => "PEM",
        K::ProxyExcludes => "localhost",
        K::UserAgent => "krishiv-test",
        _ => "x",
    });
    ClientOptions::new().with_config(key, value)
}

#[test]
fn every_key_changes_the_fingerprint() {
    let distinct: HashSet<ClientConfigKey> = KEYS.iter().copied().collect();
    assert_eq!(distinct.len(), KEYS.len(), "no key is listed twice");
    let mut seen = HashSet::from([fingerprint(&ClientOptions::new()).unwrap()]);
    for key in KEYS {
        assert!(
            seen.insert(fingerprint(&changed(key)).unwrap()),
            "{key:?} does not change the fingerprint"
        );
    }
}

#[test]
fn options_with_default_headers_are_never_shared() {
    let (counter, clients) = counting();
    let mut headers = http::HeaderMap::new();
    headers.insert("authorization", http::HeaderValue::from_static("Bearer t"));
    let options = ClientOptions::new().with_default_headers(headers);
    clients.connect(&options).unwrap();
    clients.connect(&options).unwrap();
    assert_eq!(builds(&counter), 2);
    assert!(clients.is_empty());
}

#[test]
fn the_debug_form_shows_no_fingerprint() {
    let (_, clients) = counting();
    let secret_proxy = ClientOptions::new().with_config(
        ClientConfigKey::ProxyUrl,
        "http://user:hunter2@127.0.0.1:3128",
    );
    clients.connect(&secret_proxy).unwrap();
    let debug = format!("{clients:?}");
    assert!(!debug.contains("hunter2"), "{debug}");
    assert!(debug.contains("clients: 1"), "{debug}");
}

/// Two registries (two queries) given one cache build their S3 clients once between them;
/// a registry without one builds as before.
#[test]
fn registries_sharing_a_cache_build_one_client_set() {
    let (counter, clients) = counting();
    let settings = Arc::new(BTreeMap::from([
        (
            "fs.s3a.endpoint".to_owned(),
            "http://127.0.0.1:9".to_owned(),
        ),
        ("fs.s3a.access.key".to_owned(), "AKIDEXAMPLE".to_owned()),
        ("fs.s3a.secret.key".to_owned(), "wJalrXUtnFEMI".to_owned()),
        ("fs.s3a.endpoint.region".to_owned(), "us-east-1".to_owned()),
    ]));
    let first = StoreRegistry::new(Arc::clone(&settings)).with_http_clients(clients.clone());
    first.open("s3a://bucket/a.parquet").unwrap();
    let after_first = builds(&counter);
    assert!(after_first >= 1);
    let second = StoreRegistry::new(Arc::clone(&settings)).with_http_clients(clients);
    second.open("s3a://bucket/b.parquet").unwrap();
    second.open("s3a://other/c.parquet").unwrap();
    assert_eq!(
        builds(&counter),
        after_first,
        "the second registry's S3 stores reuse the first's clients"
    );
    let shared = builds(&counter);
    let unshared = StoreRegistry::new(settings);
    unshared.open("s3a://bucket/e.parquet").unwrap();
    assert_eq!(builds(&counter), shared, "no cache, no use of it");
}
