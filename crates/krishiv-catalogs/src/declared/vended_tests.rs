// SPDX-License-Identifier: Apache-2.0
//! The vended-answer cache, the block-cache scope and the vended path locator (design P3-T067 §4.1, §4.3, §4.4,
//! §8 `vended_cache_per_identity`, `vended_cache_bounded`, `block_cache_scope_encoding_no_collision`,
//! `data_file_outside_vended_prefix_refused`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::principal::{Principal, PrincipalKind};
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

fn answer(location: &str) -> Arc<VendedAnswer> {
    Arc::new(VendedAnswer::new(
        Vec::new(),
        BTreeMap::new(),
        location.to_owned(),
    ))
}

#[test]
fn vended_cache_per_identity() {
    let cache = VendedCache::default();
    let (ana, bob) = (
        IdentityKey::of("cat", &user("ana", None)),
        IdentityKey::of("cat", &user("bob", None)),
    );
    cache.put(key(ana.clone(), "t"), answer("s3://b/ana"), 10);
    assert!(cache.get(&key(bob.clone(), "t")).is_none());
    cache.put(key(bob.clone(), "t"), answer("s3://b/bob"), 10);
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
    cache.put(key_of(2, ana.clone(), "t"), answer("s3://b/two"), 10);
    // REFRESH TABLE drops every identity's answer for the table, in that catalog instance only.
    cache.put(key(ana.clone(), "u"), answer("s3://b/u"), 10);
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
    let older = answer("s3://b/old");
    std::thread::sleep(std::time::Duration::from_millis(5));
    let newer = answer("s3://b/new");
    cache.put_if_newer(k(), Arc::clone(&older), 10);
    cache.put_if_newer(k(), Arc::clone(&newer), 10);
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/new");
    cache.put_if_newer(k(), older, 10);
    assert_eq!(cache.get(&k()).unwrap().location(), "s3://b/new");
}

#[test]
fn vended_cache_bounded() {
    let cache = VendedCache::default();
    let id = |n: &str| key(IdentityKey::of("cat", &user(n, None)), "t");
    cache.put(id("a"), answer("s3://b/a"), 2);
    cache.put(id("b"), answer("s3://b/b"), 2);
    // `a` used last, so `b` is the least recently used when `c` comes.
    assert!(cache.get(&id("a")).is_some());
    cache.put(id("c"), answer("s3://b/c"), 2);
    assert_eq!(cache.len(), 2);
    assert!(cache.get(&id("b")).is_none());
    assert!(cache.get(&id("a")).is_some() && cache.get(&id("c")).is_some());
    // Replacing an entry never evicts another.
    cache.put(id("c"), answer("s3://b/c2"), 2);
    assert_eq!(cache.len(), 2);
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
