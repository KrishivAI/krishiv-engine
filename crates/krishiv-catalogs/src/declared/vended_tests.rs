// SPDX-License-Identifier: Apache-2.0
//! The vended-answer cache, the block-cache scope and the vended path locator (design P3-T067 §4.1, §4.3, §4.4,
//! §8 `vended_cache_per_identity`, `vended_cache_bounded`, `block_cache_scope_encoding_no_collision`,
//! `data_file_outside_vended_prefix_refused`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_common::secret::SecretString;
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_session::catalog_registry::IdentityKey;
use krishiv_store::CacheScope;

use super::{Key, VendedAnswer, VendedCache, VendedPaths, cache_scope};
use crate::iceberg_rest::vended::VendedPrefix;

fn user(name: &str, tenant: Option<&str>) -> Principal {
    let mut p = Principal::new(name, PrincipalKind::User);
    p.tenant = tenant.map(str::to_owned);
    p
}

fn key(identity: IdentityKey, table: &str) -> Key {
    key_of(1, identity, table)
}

/// The key of `table` in the catalog instance `instance`.
fn key_of(instance: u64, identity: IdentityKey, table: &str) -> Key {
    (instance, identity, vec!["db".to_owned()], table.to_owned())
}

/// A cache with the catalog instances the tests key by (0 to 15) alive.
fn test_cache() -> VendedCache {
    let cache = VendedCache::default();
    for instance in 0..16 {
        cache.register_instance(instance);
    }
    cache
}

const TTL: Duration = Duration::from_secs(3600);

fn answer(location: &str) -> Arc<VendedAnswer> {
    Arc::new(VendedAnswer::new(
        Vec::new(),
        BTreeMap::new(),
        location.to_owned(),
    ))
}

/// An answer for `location` fetched at `fetched`, received in the order of `fetched` (as with a steady clock).
fn answer_fetched(location: &str, fetched: SystemTime) -> Arc<VendedAnswer> {
    let mut a = VendedAnswer::new(Vec::new(), BTreeMap::new(), location.to_owned());
    a.fetched = fetched;
    a.received = fetched
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    Arc::new(a)
}

#[test]
fn vended_cache_per_identity() {
    let cache = test_cache();
    let (ana, bob) = (
        IdentityKey::of("cat", &user("ana", None)),
        IdentityKey::of("cat", &user("bob", None)),
    );
    cache.put(key(ana.clone(), "t"), answer("s3://b/ana"), TTL, 10);
    assert!(cache.get(&key(bob.clone(), "t")).is_none());
    cache.put(key(bob.clone(), "t"), answer("s3://b/bob"), TTL, 10);
    assert_eq!(
        cache.get(&key(ana.clone(), "t")).unwrap().location(),
        "s3://b/ana"
    );
    assert_eq!(cache.get(&key(bob, "t")).unwrap().location(), "s3://b/bob");
    // Another catalog's identity of the same principal is another key.
    assert!(
        cache
            .get(&key(IdentityKey::of("other", &user("ana", None)), "t"))
            .is_none()
    );
    // Another instance of a catalog of the same name (another runtime, maybe another server) is another key.
    assert!(cache.get(&key_of(2, ana.clone(), "t")).is_none());
    cache.put(key_of(2, ana.clone(), "t"), answer("s3://b/two"), TTL, 10);
    // REFRESH TABLE drops every identity's answer for the table, in that catalog instance only.
    cache.put(key(ana.clone(), "u"), answer("s3://b/u"), TTL, 10);
    cache.drop_table(1, &["db".to_owned()], "t");
    assert_eq!(cache.len(), 2);
    assert!(cache.get(&key(ana.clone(), "u")).is_some());
    assert!(cache.get(&key_of(2, ana.clone(), "t")).is_some());
    // A dropped catalog's answers go with it.
    cache.drop_instance(2);
    assert_eq!(cache.len(), 1);
    assert!(cache.get(&key(ana, "u")).is_some());
}

/// A `loadTable` answer replaces the held one only when it was fetched later.
#[test]
fn vended_cache_keeps_the_newer_answer() {
    let cache = test_cache();
    let k = || key(IdentityKey::of("cat", &user("ana", None)), "t");
    let now = SystemTime::now();
    let older = answer_fetched("s3://b/old", now - Duration::from_secs(10));
    let newer = answer_fetched("s3://b/new", now);
    cache.put_if_newer(k(), Arc::clone(&older), TTL, 10);
    cache.put_if_newer(k(), Arc::clone(&newer), TTL, 10);
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/new");
    cache.put_if_newer(k(), older, TTL, 10);
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/new");
}

/// Which answer is newer follows the order they were received in, not the wall clock: a replacement fetched after
/// the clock stepped back still replaces, and a refresh still publishes over it.
#[test]
fn vended_cache_orders_answers_by_receipt_not_by_the_clock() {
    let cache = test_cache();
    let k = || key(IdentityKey::of("cat", &user("ana", None)), "t");
    let first = answer("s3://b/first");
    let mut second = VendedAnswer::new(Vec::new(), BTreeMap::new(), "s3://b/second".to_owned());
    second.fetched = first.fetched - Duration::from_secs(60);
    let (first_fetched, second_received) = (first.fetched, second.received);
    let second = Arc::new(second);
    cache.put_if_newer(k(), first, TTL, 10);
    cache.put_if_newer(k(), second, TTL, 10);
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/second");
    let mut third = VendedAnswer::new(Vec::new(), BTreeMap::new(), "s3://b/third".to_owned());
    third.fetched -= Duration::from_secs(120);
    let published = cache.publish(k(), Arc::new(third), TTL, 10, cache.drops());
    assert_eq!(published.location(), "s3://b/third");
    // The cache now holds the refresh, not only `publish`'s answer.
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/third");
    // An answer received before the refresh no longer replaces it, whatever its fetch time.
    let mut late = VendedAnswer::new(Vec::new(), BTreeMap::new(), "s3://b/late".to_owned());
    late.fetched = first_fetched + Duration::from_secs(60);
    late.received = second_received;
    cache.put_if_newer(k(), Arc::new(late), TTL, 10);
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/third");
}

/// The live set holds only catalogs not yet dropped: registering and dropping many leaves nothing behind.
#[test]
fn vended_cache_forgets_dropped_instances() {
    let cache = VendedCache::default();
    for instance in 100..1100 {
        cache.register_instance(instance);
        cache.drop_instance(instance);
    }
    assert_eq!(cache.lock().live.len(), 0);
    // An unregistered instance keeps nothing either.
    let ana = IdentityKey::of("cat", &user("ana", None));
    cache.put(key_of(5000, ana, "t"), answer("s3://b/t"), TTL, 10);
    assert_eq!(cache.len(), 0);
}

#[test]
fn vended_cache_bounded() {
    let cache = test_cache();
    let id = |n: &str| key(IdentityKey::of("cat", &user(n, None)), "t");
    cache.put(id("a"), answer("s3://b/a"), TTL, 2);
    cache.put(id("b"), answer("s3://b/b"), TTL, 2);
    // `a` used last, so `b` is the least recently used when `c` comes.
    assert!(cache.get(&id("a")).is_some());
    cache.put(id("c"), answer("s3://b/c"), TTL, 2);
    assert_eq!(cache.len(), 2);
    assert!(cache.get(&id("b")).is_none());
    assert!(cache.get(&id("a")).is_some() && cache.get(&id("c")).is_some());
    // Replacing an entry never evicts another.
    cache.put(id("c"), answer("s3://b/c2"), TTL, 2);
    assert_eq!(cache.len(), 2);
}

/// An S3 answer for `s3://b/t` whose credential expires at `expires_ms` (epoch ms), or has no expiry key.
fn s3_answer(expires_ms: Option<u128>) -> Arc<VendedAnswer> {
    let mut config = BTreeMap::from([
        ("s3.access-key-id".to_owned(), SecretString::new("AKIA")),
        (
            "s3.secret-access-key".to_owned(),
            SecretString::new("secret"),
        ),
    ]);
    if let Some(ms) = expires_ms {
        config.insert(
            "s3.session-token-expires-at-ms".to_owned(),
            SecretString::new(ms.to_string()),
        );
    }
    Arc::new(VendedAnswer::new(
        vec![crate::iceberg_rest::types::StorageCredential {
            prefix: "s3://b/t".to_owned(),
            config,
        }],
        BTreeMap::new(),
        "s3://b/t".to_owned(),
    ))
}

fn ms(t: SystemTime) -> u128 {
    t.duration_since(UNIX_EPOCH).unwrap().as_millis()
}

/// Design §4.3: an entry is dropped at its credential's expiry (the expiry key, else fetched plus the default TTL),
/// in `get` and in any later `put`, so expired secrets do not stay in the process until LRU eviction.
#[test]
fn vended_cache_drops_expired_entries() {
    let cache = test_cache();
    let id = |n: &str| key(IdentityKey::of("cat", &user(n, None)), "t");
    let soon = || ms(cache.now() + Duration::from_millis(300));
    let future = ms(cache.now() + Duration::from_secs(3600));
    // Live when kept, expired by the time of the lookup: dropped on get.
    cache.put(id("expiring"), s3_answer(Some(soon())), TTL, 10);
    assert_eq!(cache.len(), 1, "kept while live");
    cache.clock().advance(Duration::from_millis(600));
    assert!(cache.get(&id("expiring")).is_none());
    assert_eq!(cache.len(), 0, "dropped on get");
    // Live when kept, expired by the time of a later put of another key: dropped there, without a get.
    cache.put(id("old"), s3_answer(Some(soon())), TTL, 10);
    assert_eq!(cache.len(), 1, "kept while live");
    cache.clock().advance(Duration::from_millis(600));
    cache.put(id("live"), s3_answer(Some(future)), TTL, 10);
    assert_eq!(cache.len(), 1);
    assert!(cache.get(&id("live")).is_some());
    // An answer already expired when offered is not kept at all.
    let past = ms(cache.now() - Duration::from_secs(1));
    cache.put(id("expired"), s3_answer(Some(past)), TTL, 10);
    assert!(cache.get(&id("expired")).is_none());
    // No expiry key: the default TTL counts from the fetch.
    cache.put(id("ttl"), s3_answer(None), Duration::ZERO, 10);
    assert!(cache.get(&id("ttl")).is_none());
    cache.put(id("ttl"), s3_answer(None), TTL, 10);
    assert!(cache.get(&id("ttl")).is_some());
}

/// An answer that is already expired is not inserted, so on a full cache it evicts no live entry; it still replaces
/// what its own key held.
#[test]
fn vended_cache_full_keeps_live_entries_over_an_expired_answer() {
    let cache = test_cache();
    let id = |n: &str| key(IdentityKey::of("cat", &user(n, None)), "t");
    let past = ms(SystemTime::now() - Duration::from_secs(1));
    let future = ms(SystemTime::now() + Duration::from_secs(3600));
    cache.put(id("a"), s3_answer(Some(future)), TTL, 2);
    cache.put(id("b"), s3_answer(Some(future)), TTL, 2);
    cache.put(id("late"), s3_answer(Some(past)), TTL, 2);
    assert_eq!(cache.len(), 2);
    assert!(cache.get(&id("a")).is_some() && cache.get(&id("b")).is_some());
    cache.put(id("a"), s3_answer(Some(past)), TTL, 2);
    assert!(
        cache.get(&id("a")).is_none(),
        "superseded by its expired answer"
    );
    assert!(cache.get(&id("b")).is_some());
}

/// An answer fetched for a dropped catalog instance (a query running across the drop) is never kept.
#[test]
fn vended_cache_keeps_nothing_for_a_dropped_instance() {
    let cache = test_cache();
    let ana = IdentityKey::of("cat", &user("ana", None));
    cache.put(key_of(7, ana.clone(), "t"), answer("s3://b/t"), TTL, 10);
    cache.drop_instance(7);
    cache.put(key_of(7, ana.clone(), "t"), answer("s3://b/t"), TTL, 10);
    cache.put_if_newer(key_of(7, ana.clone(), "u"), answer("s3://b/u"), TTL, 10);
    assert_eq!(cache.len(), 0);
    cache.put(key_of(8, ana, "t"), answer("s3://b/t"), TTL, 10);
    assert_eq!(cache.len(), 1);
}

fn scope_text(s: CacheScope) -> String {
    match s {
        CacheScope::Principal(p) => p,
        other => panic!("expected a principal scope, got {other:?}"),
    }
}

#[test]
fn block_cache_scope_encoding_no_collision() {
    // Delegated: catalog, principal and tenant, each length-prefixed, so no two identities share a scope.
    let pairs = [
        (user("a:b", None), user("a", Some("b"))),
        (user("x/y", None), user("x", Some("/y"))),
        (user("ana", Some("t1")), user("ana", Some("t2"))),
        (user("ana", None), user("ana", Some("-"))),
    ];
    for (p, q) in &pairs {
        let (a, b) = (
            scope_text(cache_scope(&IdentityKey::of("cat", p), true)),
            scope_text(cache_scope(&IdentityKey::of("cat", q), true)),
        );
        assert_ne!(a, b, "{p:?} vs {q:?}");
    }
    // `delegated_by` is left out: the bytes a user may read do not depend on who acts for them.
    let direct = cache_scope(&IdentityKey::of("cat", &user("ana", None)), true);
    let acted = cache_scope(
        &IdentityKey::of("cat", &user("ana", None).with_delegator("svc")),
        true,
    );
    assert_eq!(scope_text(direct), scope_text(acted));
    // Any other auth: one scope per catalog, whoever reads, and catalogs never share one.
    let (ana, bob) = (
        cache_scope(&IdentityKey::of("cat", &user("ana", None)), false),
        cache_scope(&IdentityKey::of("cat", &user("bob", None)), false),
    );
    assert_eq!(scope_text(ana.clone()), scope_text(bob));
    assert_ne!(
        scope_text(ana),
        scope_text(cache_scope(&IdentityKey::server("cat2"), false))
    );
}

/// A locator that counts the calls that reach it and places nothing.
#[derive(Debug, Default)]
struct Counting(AtomicUsize);

impl PathLocator for Counting {
    fn locate(&self, _path: &str) -> Result<LocatedPath, KrishivError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(krishiv_common::error::InternalError {
            message: "placed".to_owned(),
        }
        .into())
    }
}

#[test]
fn data_file_outside_vended_prefix_refused() {
    let inner = Arc::new(Counting::default());
    let paths = VendedPaths {
        inner: Arc::clone(&inner) as Arc<dyn PathLocator>,
        prefixes: vec![
            VendedPrefix::parse("s3://b/wh/t").unwrap(),
            VendedPrefix::parse("gs://g").unwrap(),
        ],
        catalog: "cat".to_owned(),
        relation: "`cat`.`db`.`t`".to_owned(),
    };
    for outside in [
        "s3://b/wh/t2/data/f.parquet",
        "s3://other/wh/t/f.parquet",
        "file:///etc/passwd",
        "/tmp/f",
        "https://h/x",
        "gs://h/k",
    ] {
        let e = paths.locate(outside).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivCatalogLocationRefused,
            "{outside}"
        );
        let text = e.to_string();
        assert!(
            text.contains("outside the vended credential prefixes"),
            "{text}"
        );
        // The location itself is never in the error.
        assert!(!text.contains(outside), "{text}");
        assert!(paths.qualified_display(outside).is_err());
    }
    assert_eq!(
        inner.0.load(Ordering::SeqCst),
        0,
        "nothing outside reached a store"
    );
    for inside in [
        "s3://b/wh/t/data/f.parquet",
        "s3a://b/wh/t/m.avro",
        "gs://g/any",
    ] {
        let e = paths.locate(inside).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{inside}");
    }
    assert_eq!(inner.0.load(Ordering::SeqCst), 3);
}

/// A refresh publishes under one lock: a live answer fetched later wins and is returned; a `REFRESH TABLE` since the
/// refresh began keeps the refreshed answer out of the cache (it serves only the refresh that fetched it).
#[test]
fn vended_cache_publish_keeps_the_newer_and_respects_refresh_table() {
    let cache = test_cache();
    let k = || key(IdentityKey::of("cat", &user("ana", None)), "t");
    let now = SystemTime::now();
    let late = answer_fetched("s3://b/late", now);
    cache.put_if_newer(k(), Arc::clone(&late), TTL, 10);
    let drops = cache.drops();
    let slow = answer_fetched("s3://b/slow", now - Duration::from_secs(5));
    let kept = cache.publish(k(), slow, TTL, 10, drops);
    assert_eq!(
        kept.location(),
        "s3://b/late",
        "the newer answer is returned"
    );
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/late");
    // A refresh newer than the held one replaces it.
    let fresh = answer_fetched("s3://b/fresh", now + Duration::from_secs(1));
    assert_eq!(
        cache.publish(k(), fresh, TTL, 10, drops).location(),
        "s3://b/fresh"
    );
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/fresh");
    // A refresh that began before a REFRESH TABLE is served, never kept.
    let before = cache.drops();
    cache.drop_table(1, &["db".to_owned()], "t");
    let stale = answer_fetched("s3://b/stale", now + Duration::from_secs(2));
    assert_eq!(
        cache.publish(k(), stale, TTL, 10, before).location(),
        "s3://b/stale"
    );
    assert!(
        cache.get(&k()).is_none(),
        "nothing kept across the REFRESH TABLE"
    );
    // One that began after it is kept.
    let after = answer_fetched("s3://b/after", now + Duration::from_secs(3));
    cache.publish(k(), after, TTL, 10, cache.drops());
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/after");
}

/// A `loadTable` answer requested before a `REFRESH TABLE` and offered after it (a query analyzed before the refresh
/// and run after it) is kept nowhere; one requested after it is kept (review 7e24f4a71 F1).
#[test]
fn vended_cache_keeps_no_load_answer_across_refresh_table() {
    let cache = test_cache();
    let k = || key(IdentityKey::of("cat", &user("ana", None)), "t");
    let loaded = |location: &str, drops: u64| {
        let mut a = VendedAnswer::new(Vec::new(), BTreeMap::new(), location.to_owned());
        a.drops = drops;
        Arc::new(a)
    };
    let before = cache.drops();
    cache.drop_table(1, &["db".to_owned()], "t");
    cache.put_if_newer(k(), loaded("s3://b/old", before), TTL, 10);
    assert!(
        cache.get(&k()).is_none(),
        "the pre-refresh answer is not put back"
    );
    cache.put_if_newer(k(), loaded("s3://b/new", cache.drops()), TTL, 10);
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/new");
}

/// A bare-scheme Azure prefix keeps the host of the table location (`blob` here), where the catalog keyed the SAS
/// token, not the store id's `dfs` alias.
#[test]
fn azure_scheme_scope_keeps_the_location_host() {
    let url = krishiv_store::parse_object_url("abfss://c@acct.blob.core.windows.net/t").unwrap();
    assert_eq!(
        super::scheme_scope(&url),
        "abfss://c@acct.blob.core.windows.net"
    );
    let reparsed = krishiv_store::parse_object_url(&super::scheme_scope(&url)).unwrap();
    assert_eq!(reparsed.host.as_deref(), Some("acct.blob.core.windows.net"));
    let s3 = krishiv_store::parse_object_url("s3://b/t").unwrap();
    assert_eq!(super::scheme_scope(&s3), s3.store_id.key());
}

/// A catalog serving `/credentials` with a new key, a `VendedTable` over it holding at most `max_entries`, and the
/// `loadTable` answer with the table's own token `table-token` (requested before any refresh).
fn token_fixture(
    max_entries: usize,
) -> (
    crate::http_test_server::Server,
    Arc<VendedCache>,
    super::VendedTable,
    VendedAnswer,
) {
    token_fixture_refreshing(max_entries, &["s3://bucket/wh"])
}

/// [`token_fixture`] whose `loadCredentials` answers one credential per prefix of `refreshed` (key ids
/// `AKIANEWKEY0`, `AKIANEWKEY1`, ...; a single prefix gets `AKIANEWKEY`).
fn token_fixture_refreshing(
    max_entries: usize,
    refreshed: &[&str],
) -> (
    crate::http_test_server::Server,
    Arc<VendedCache>,
    super::VendedTable,
    VendedAnswer,
) {
    use crate::declared::tests::{decl, deps, path};
    use crate::iceberg_rest::test_server::{Reply, serve};
    use krishiv_session::catalog_decl::DeclMode;
    use krishiv_store::egress::PrivateAddresses;

    let long = ms(SystemTime::now() + Duration::from_secs(3600));
    let credential = |prefix: &str, key_id: &str| {
        format!(
            r#"{{"prefix": "{prefix}", "config": {{"s3.access-key-id": "{key_id}", "s3.secret-access-key": "s",
                "s3.session-token-expires-at-ms": "{long}"}}}}"#
        )
    };
    let refreshed = match refreshed {
        [one] => credential(one, "AKIANEWKEY"),
        many => many
            .iter()
            .enumerate()
            .map(|(i, p)| credential(p, &format!("AKIANEWKEY{i}")))
            .collect::<Vec<_>>()
            .join(", "),
    };
    let config = r#"{"defaults": {}, "overrides": {}, "endpoints": ["GET /v1/{prefix}/namespaces/{namespace}/tables/{table}",
        "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"]}"#;
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, config.into()),
        "/v1/namespaces/db/tables/t/credentials" => {
            Reply::Json(200, format!(r#"{{"storage-credentials": [{refreshed}]}}"#))
        }
        _ => Reply::Json(
            404,
            r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#.into(),
        ),
    });
    let d = decl(
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            ("krishiv.catalog.lake.uri", server.base.as_str()),
        ],
        DeclMode::Embedded,
    );
    let config = super::super::IcebergRestFactory::config(
        &d,
        &deps(DeclMode::Embedded, PrivateAddresses::Allow),
        true,
    )
    .unwrap();
    let client = crate::iceberg_rest::RestClient::new(
        "lake",
        config,
        Arc::new(super::super::table::EmbeddedLocations::default()),
    )
    .unwrap();
    let cache = Arc::new(test_cache());
    let table = super::VendedTable {
        cache: Arc::clone(&cache),
        key: key_of(1, IdentityKey::server("lake".to_owned()), "t"),
        max_entries,
        default_ttl: TTL,
        skew: Duration::from_secs(300),
        refetch: super::Refetch {
            client: Arc::new(client),
            caller: crate::iceberg_rest::RestCaller::server(),
            namespace: vec!["db".to_owned()],
            table: "t".to_owned(),
            catalog: "lake".to_owned(),
            relation: "`lake`.`db`.`t`".to_owned(),
            location: "s3://bucket/wh/db/t".to_owned(),
        },
    };
    // The `loadTable` answer, with the table's own token.
    let first: crate::iceberg_rest::types::StorageCredential =
        crate::iceberg_rest::types::StorageCredential {
            prefix: "s3://bucket/wh".to_owned(),
            config: BTreeMap::from([
                (
                    "s3.access-key-id".to_owned(),
                    SecretString::new("AKIAOLDKEY"),
                ),
                ("s3.secret-access-key".to_owned(), SecretString::new("s")),
                (
                    "s3.session-token-expires-at-ms".to_owned(),
                    SecretString::new(long.to_string()),
                ),
            ]),
        };
    let mut loaded = VendedAnswer::new(
        vec![first],
        BTreeMap::new(),
        "s3://bucket/wh/db/t".to_owned(),
    );
    loaded.table_token = Some(SecretString::new("table-token"));
    (server, cache, table, loaded)
}

/// The `/credentials` requests the catalog saw, by bearer.
fn credential_bearers(server: &crate::http_test_server::Server) -> Vec<Option<String>> {
    use crate::declared::tests::path;
    server
        .seen()
        .iter()
        .filter(|r| path(r) == "/v1/namespaces/db/tables/t/credentials")
        .map(|r| r.header("authorization").map(str::to_owned))
        .collect()
}

/// Review 7e24f4a71 F2: once the answer a source served is evicted from the cache (another table's answer, past
/// `vended.maxEntries = 1`), the source's next refresh still sends that answer's table token, not the caller's.
#[test]
fn a_refresh_after_eviction_sends_the_served_answers_table_token() {
    use crate::declared::tests::path;
    let (server, cache, table, loaded) = token_fixture(1);
    table.offer(Arc::new(loaded));
    let source = table.source("s3://bucket/wh/db/t", None).unwrap();
    futures::executor::block_on(source.current()).unwrap();
    assert!(
        server.seen().iter().all(|r| path(r) == "/v1/config"),
        "served from the cache"
    );
    // Another table's answer evicts this one (one entry at most).
    cache.put_if_newer(
        key_of(1, IdentityKey::server("lake".to_owned()), "u"),
        answer("s3://bucket/wh/db/u"),
        TTL,
        1,
    );
    assert!(
        cache
            .get(&key_of(1, IdentityKey::server("lake".to_owned()), "t"))
            .is_none()
    );
    // The store asks again (it refused the credential): the refresh carries the served answer's token.
    futures::executor::block_on(source.current()).unwrap();
    assert_eq!(
        credential_bearers(&server),
        vec![Some("Bearer table-token".to_owned())]
    );
}

/// Review 2ed68fa67 F1: a `loadTable` answer offered after a `REFRESH TABLE` is kept nowhere, but its own query
/// still reads with it, so no `/credentials` request goes out, and the first refresh carries its table token.
#[test]
fn an_answer_the_cache_refused_still_serves_its_query_with_its_table_token() {
    use crate::declared::tests::path;
    let (server, cache, table, mut loaded) = token_fixture(10);
    loaded.drops = cache.drops();
    cache.drop_table(1, &["db".to_owned()], "t");
    let own = Arc::new(loaded);
    table.offer(Arc::clone(&own));
    let key = key_of(1, IdentityKey::server("lake".to_owned()), "t");
    assert!(cache.get(&key).is_none(), "refused by the cache");
    let answer = futures::executor::block_on(table.answer(Some(&own))).unwrap();
    assert!(Arc::ptr_eq(&answer, &own));
    let source = table
        .source("s3://bucket/wh/db/t", Some(Arc::clone(&own)))
        .unwrap();
    futures::executor::block_on(source.current()).unwrap();
    assert!(
        server.seen().iter().all(|r| path(r) == "/v1/config"),
        "read with its own answer: {:?}",
        server.seen().iter().map(path).collect::<Vec<_>>()
    );
    assert!(cache.get(&key).is_none(), "still kept nowhere");
    // The store asks again: the refresh carries the table's own token, not the caller's.
    futures::executor::block_on(source.current()).unwrap();
    assert_eq!(
        credential_bearers(&server),
        vec![Some("Bearer table-token".to_owned())]
    );
}

/// #930 thread (Codex, P2): a `loadTable` answer whose storage credential expired while it sat in the 30 s table
/// cache is not kept (`State::put` drops it), but the table opens with it and its first refresh carries its table
/// token, not the caller's.
#[test]
fn an_expired_loaded_answer_refreshes_with_its_table_token() {
    let (server, cache, table, mut loaded) = token_fixture(10);
    let past = ms(SystemTime::now() - Duration::from_secs(1));
    for c in &mut loaded.credentials {
        c.config.insert(
            "s3.session-token-expires-at-ms".to_owned(),
            SecretString::new(past.to_string()),
        );
    }
    let own = Arc::new(loaded);
    table.offer(Arc::clone(&own));
    let key = key_of(1, IdentityKey::server("lake".to_owned()), "t");
    assert!(cache.get(&key).is_none(), "an expired answer is not kept");
    let answer = futures::executor::block_on(table.answer(Some(&own))).unwrap();
    assert!(Arc::ptr_eq(&answer, &own), "no refresh to find the scopes");
    assert!(credential_bearers(&server).is_empty());
    let source = table.source("s3://bucket/wh/db/t", Some(own)).unwrap();
    futures::executor::block_on(source.current()).unwrap();
    assert_eq!(
        credential_bearers(&server),
        vec![Some("Bearer table-token".to_owned())]
    );
}

/// #930 thread (CodeRabbit): a refresh whose answer narrows the prefix below the scope's root (`s3://bucket/wh` to
/// `s3://bucket/wh/db/t`) still serves the scope, instead of refusing the long-running read.
#[test]
fn a_refresh_that_narrows_the_prefix_still_serves_the_scope() {
    let (server, _cache, table, loaded) = token_fixture_refreshing(10, &["s3://bucket/wh/db/t"]);
    table.offer(Arc::new(loaded));
    let source = table.source("s3://bucket/wh", None).unwrap();
    futures::executor::block_on(source.current()).unwrap();
    assert!(
        credential_bearers(&server).is_empty(),
        "served from the cache"
    );
    // The store asks again (it refused the credential): the refreshed answer names only the narrower prefix.
    let refreshed = futures::executor::block_on(source.current());
    assert!(refreshed.is_ok(), "{:?}", refreshed.err());
    assert_eq!(credential_bearers(&server).len(), 1);
}

/// Two narrower prefixes with different credentials: no one credential serves the scope, so the read is refused as
/// before (`KRISHIV_CATALOG_LOCATION_REFUSED`), never served one of them at random.
#[test]
fn a_refresh_narrowed_to_two_different_credentials_is_refused() {
    let (_server, _cache, table, loaded) = token_fixture_refreshing(
        10,
        &["s3://bucket/wh/db/t/data", "s3://bucket/wh/db/t/metadata"],
    );
    table.offer(Arc::new(loaded));
    let source = table.source("s3://bucket/wh", None).unwrap();
    futures::executor::block_on(source.current()).unwrap();
    let e = futures::executor::block_on(source.current()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLocationRefused);
}
