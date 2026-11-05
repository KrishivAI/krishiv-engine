// SPDX-License-Identifier: Apache-2.0
//! S4: per-location credential scopes in `StoreRegistry`: the most specific prefix wins on path
//! segments, scoped store ids are process-unique, a per-read credential on a scoped location
//! fails, and dropping a registry releases only its own ids.
#![cfg(feature = "cloud")]
#![expect(clippy::unwrap_used, reason = "test code")]

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use krishiv_common::error::ErrorClass;
use krishiv_common::secret::OptionValue;
use krishiv_store::{CredentialScope, CredentialSource, ScopedCredential, StoreId, StoreRegistry};

const CANARY: &str = "Canary-91c0de";

#[derive(Debug)]
struct Fixed(&'static str);

#[async_trait]
impl CredentialSource for Fixed {
    async fn fetch(&self, _scope: &CredentialScope) -> krishiv_store::Result<ScopedCredential> {
        Ok(ScopedCredential::s3(
            format!("{CANARY}-{}", self.0),
            format!("{CANARY}-secret"),
            None,
            Some(SystemTime::now() + Duration::from_secs(3600)),
        ))
    }
}

fn registry() -> StoreRegistry {
    let mut settings = BTreeMap::new();
    settings.insert(
        "fs.s3a.endpoint".to_owned(),
        "http://127.0.0.1:9".to_owned(),
    );
    StoreRegistry::new(Arc::new(settings))
}

fn scope(location: &str) -> CredentialScope {
    CredentialScope::new(location).unwrap()
}

fn with(location: &str, who: &'static str) -> StoreRegistry {
    registry().with_source(scope(location), Arc::new(Fixed(who)))
}

#[test]
fn two_prefixes_in_one_bucket_get_different_stores() {
    let r = registry()
        .with_source(scope("s3://b/warehouse/a/"), Arc::new(Fixed("a")))
        .with_source(scope("s3://b/warehouse/b/"), Arc::new(Fixed("b")));
    let a = r.open("s3://b/warehouse/a/f.parquet").unwrap();
    let b = r.open("s3://b/warehouse/b/f.parquet").unwrap();
    assert_ne!(a.id.key(), b.id.key());
    assert!(!Arc::ptr_eq(&a.store, &b.store));
    // The same scope shares its store.
    let a2 = r.open("s3://b/warehouse/a/other.parquet").unwrap();
    assert_eq!(a.id, a2.id);
    assert!(Arc::ptr_eq(&a.store, &a2.store));
}

#[test]
fn the_longest_prefix_wins_on_segment_boundaries() {
    let r = registry()
        .with_source(scope("s3://b/warehouse/"), Arc::new(Fixed("wide")))
        .with_source(scope("s3://b/warehouse/t1/"), Arc::new(Fixed("narrow")));
    let wide = r.open("s3://b/warehouse/t2/f").unwrap();
    let narrow = r.open("s3://b/warehouse/t1/f").unwrap();
    let t10 = r.open("s3://b/warehouse/t10/f").unwrap();
    assert_ne!(wide.id.key(), narrow.id.key());
    // `t1/` does not cover `t10/`: it falls to the wide scope.
    assert_eq!(wide.id, t10.id);
    // Outside every scope the plain bucket id is used.
    let outside = r.open("s3://b/elsewhere/f").unwrap();
    assert_eq!(outside.id.key(), "s3://b");
    let other_bucket = r.open("s3://other/warehouse/t1/f").unwrap();
    assert_eq!(other_bucket.id.key(), "s3://other");
}

#[test]
fn registries_never_share_a_scoped_id_and_dropping_releases_only_its_own() {
    let one = with("s3://b/warehouse/t/", "one");
    let two = with("s3://b/warehouse/t/", "two");
    let released: Arc<Mutex<Vec<StoreId>>> = Arc::default();
    for r in [&one, &two] {
        let sink = Arc::clone(&released);
        r.on_release(move |id| sink.lock().unwrap().push(id.clone()));
    }
    let a = one.open("s3://b/warehouse/t/f").unwrap();
    let b = two.open("s3://b/warehouse/t/f").unwrap();
    assert_ne!(a.id.key(), b.id.key(), "same table, different sources");
    assert_eq!(one.scoped_ids(), vec![a.id.clone()]);
    drop(one);
    let seen = released.lock().unwrap().clone();
    assert_eq!(seen, vec![a.id], "only the dropped registry's id");
    // The survivor still works and still has its id.
    assert!(two.open("s3://b/warehouse/t/g").is_ok());
    assert_eq!(two.scoped_ids(), vec![b.id]);
    drop(two);
    let all: HashSet<_> = released.lock().unwrap().iter().cloned().collect();
    assert_eq!(all.len(), 2);
}

#[test]
fn a_per_read_credential_on_a_scoped_location_fails_naming_only_the_key() {
    let r = with("s3://b/warehouse/t/", "x");
    for key in [
        "fs.s3a.access.key",
        "FS.S3A.SECRET.KEY",
        "fs.s3a.session.token",
        "fs.azure.sas.acct",
    ] {
        let mut options = BTreeMap::new();
        options.insert(key.to_owned(), OptionValue::from_user(key, CANARY));
        let e = r
            .open_with_options("s3://b/warehouse/t/f", &options)
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivInvalidConfigValue,
            "{key}"
        );
        let shown = format!("{e} {e:?}");
        assert!(shown.contains(key), "{shown}");
        assert!(!shown.contains(CANARY), "{shown}");
    }
    // Outside the scope the same option is an ordinary static credential, and a non-credential
    // per-read option is fine inside it.
    let mut keyed = BTreeMap::new();
    keyed.insert(
        "fs.s3a.access.key".to_owned(),
        OptionValue::from_user("fs.s3a.access.key", "k"),
    );
    keyed.insert(
        "fs.s3a.secret.key".to_owned(),
        OptionValue::from_user("fs.s3a.secret.key", "s"),
    );
    assert!(r.open_with_options("s3://b/elsewhere/f", &keyed).is_ok());
    let mut region = BTreeMap::new();
    region.insert(
        "fs.s3a.region".to_owned(),
        OptionValue::Plain("eu-west-1".into()),
    );
    assert!(r.open_with_options("s3://b/warehouse/t/f", &region).is_ok());
}

#[test]
fn gcs_and_azure_locations_take_sources_and_web_schemes_do_not() {
    let r = registry()
        .with_source(scope("gs://bucket/data/"), Arc::new(Fixed("gcs")))
        .with_source(
            scope("abfss://container@acct.dfs.core.windows.net/data/"),
            Arc::new(Fixed("az")),
        );
    let g = r.open("gs://bucket/data/f").unwrap();
    assert!(g.id.key().contains("#scope:"));
    let a = r
        .open("abfss://container@acct.dfs.core.windows.net/data/f")
        .unwrap();
    assert!(a.id.key().contains("#scope:"));
    assert!(CredentialScope::new("https://example.com/x").is_err());
}

#[test]
fn nothing_scoped_prints_a_secret_or_a_key_id() {
    let r = with("s3://b/warehouse/t/", "x");
    let opened = r.open("s3://b/warehouse/t/f").unwrap();
    let shown = format!(
        "{r:?} {opened:?} {} {:?} {:?}",
        opened.id,
        opened.id,
        scope("s3://b/warehouse/t/")
    );
    assert!(!shown.contains(CANARY), "{shown}");
}
