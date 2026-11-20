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

const TTL: Duration = Duration::from_secs(3600);

fn answer(location: &str) -> Arc<VendedAnswer> {
    Arc::new(VendedAnswer::new(
        Vec::new(),
        BTreeMap::new(),
        location.to_owned(),
    ))
}

/// An answer for `location` fetched at `fetched`.
fn answer_fetched(location: &str, fetched: SystemTime) -> Arc<VendedAnswer> {
    let mut a = VendedAnswer::new(Vec::new(), BTreeMap::new(), location.to_owned());
    a.fetched = fetched;
    Arc::new(a)
}

#[test]
fn vended_cache_per_identity() {
    let cache = VendedCache::default();
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
    let cache = VendedCache::default();
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

#[test]
fn vended_cache_bounded() {
    let cache = VendedCache::default();
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
    let cache = VendedCache::default();
    let id = |n: &str| key(IdentityKey::of("cat", &user(n, None)), "t");
    let soon = || ms(SystemTime::now() + Duration::from_millis(300));
    let future = ms(SystemTime::now() + Duration::from_secs(3600));
    // Live when kept, expired by the time of the lookup: dropped on get.
    cache.put(id("expiring"), s3_answer(Some(soon())), TTL, 10);
    assert_eq!(cache.len(), 1, "kept while live");
    std::thread::sleep(Duration::from_millis(600));
    assert!(cache.get(&id("expiring")).is_none());
    assert_eq!(cache.len(), 0, "dropped on get");
    // Live when kept, expired by the time of a later put of another key: dropped there, without a get.
    cache.put(id("old"), s3_answer(Some(soon())), TTL, 10);
    assert_eq!(cache.len(), 1, "kept while live");
    std::thread::sleep(Duration::from_millis(600));
    cache.put(id("live"), s3_answer(Some(future)), TTL, 10);
    assert_eq!(cache.len(), 1);
    assert!(cache.get(&id("live")).is_some());
    // An answer already expired when offered is not kept at all.
    let past = ms(SystemTime::now() - Duration::from_secs(1));
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
    let cache = VendedCache::default();
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
    let cache = VendedCache::default();
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
    let cache = VendedCache::default();
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
