// SPDX-License-Identifier: Apache-2.0
//! Credential scopes and credentials with default features (no `cloud`): the scope arithmetic,
//! the registry bookkeeping around `with_source`, and the credential constructors. Building a
//! cloud store is refused in this build, so `open` fails cleanly.
#![cfg(not(feature = "cloud"))]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use krishiv_store::{
    CredentialScope, CredentialSource, ObjectScheme, ScopedCredential, StoreRegistry,
};

#[derive(Debug)]
struct Fixed;

#[async_trait]
impl CredentialSource for Fixed {
    async fn fetch(&self, _scope: &CredentialScope) -> krishiv_store::Result<ScopedCredential> {
        Ok(ScopedCredential::gcs("t", None))
    }
}

#[test]
fn a_scope_knows_its_bucket_prefix_and_what_it_covers() {
    let s = CredentialScope::new("s3://b/warehouse/t1/").unwrap();
    assert_eq!(s.scheme(), ObjectScheme::S3);
    assert_eq!(s.bucket(), "b");
    assert_eq!(s.prefix(), "warehouse/t1");
    assert!(s.contains_key("warehouse/t1/f.parquet"));
    assert!(!s.contains_key("warehouse/t10/f"));
    assert_eq!(s.to_string(), "s3://b/warehouse/t1");
    let whole = CredentialScope::new("gs://g").unwrap();
    assert_eq!(whole.prefix(), "");
    assert_eq!(whole.to_string(), "gs://g");
    assert!(whole.contains_key("anything/at/all"));
}

#[test]
fn credential_constructors_keep_expiry_and_print_no_secret() {
    let at = SystemTime::UNIX_EPOCH + Duration::from_secs(42);
    let all = [
        ScopedCredential::s3(
            "Canary-id",
            "Canary-secret",
            Some("Canary-tok".into()),
            Some(at),
        ),
        ScopedCredential::gcs("Canary-bearer", Some(at)),
        ScopedCredential::azure_sas("?sv=1&sig=Canary-sig", Some(at)),
        ScopedCredential::azure_bearer("Canary-bearer", Some(at)),
    ];
    for c in &all {
        assert_eq!(c.expires_at(), Some(at));
        assert!(!format!("{c:?}").contains("Canary"));
    }
    assert_eq!(ScopedCredential::gcs("t", None).expires_at(), None);
}

#[test]
fn registry_bookkeeping_without_cloud_stores() {
    let r = StoreRegistry::new(Arc::new(BTreeMap::<String, String>::new()))
        .with_source(CredentialScope::new("s3://b/p/").unwrap(), Arc::new(Fixed));
    let released = Arc::new(std::sync::Mutex::new(0_usize));
    let sink = Arc::clone(&released);
    r.on_release(move |_| *sink.lock().unwrap() += 1);
    // The store can't be built here, so nothing is handed out and nothing is released.
    assert!(r.open("s3://b/p/f").is_err());
    assert!(r.open("s3://b/elsewhere/f").is_err());
    assert!(r.scoped_ids().is_empty());
    drop(r);
    assert_eq!(*released.lock().unwrap(), 0);
}
