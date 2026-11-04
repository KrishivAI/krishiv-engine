// SPDX-License-Identifier: Apache-2.0
//! `StoreSettings` without the cloud builders (so it runs in the default build): how each source
//! types a value, and how a key is looked up.
#![expect(clippy::unwrap_used, reason = "test code")]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::secret::OptionValue;
use krishiv_store::{S3_ENDPOINT, S3_SECRET_KEY, StoreSettings};

const CANARY: &str = "Canary-91c0de";

/// The same two pairs in each kind of source.
fn sources(pairs: &[(&str, &str)]) -> Vec<(&'static str, Arc<dyn StoreSettings>)> {
    let (config, _) = Config::from_pairs(pairs.to_vec()).unwrap();
    let btree: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let hash: HashMap<String, String> = btree.clone().into_iter().collect();
    let typed: BTreeMap<String, OptionValue> = btree
        .iter()
        .map(|(k, v)| (k.clone(), OptionValue::from_user(k, v.as_str())))
        .collect();
    vec![
        ("Config", Arc::new(config)),
        ("BTreeMap<String, String>", Arc::new(btree)),
        ("HashMap<String, String>", Arc::new(hash)),
        ("BTreeMap<String, OptionValue>", Arc::new(typed)),
    ]
}

#[test]
fn every_source_types_a_value_by_its_key() {
    for (name, source) in sources(&[
        (S3_ENDPOINT, "http://127.0.0.1:9000"),
        (S3_SECRET_KEY, CANARY),
    ]) {
        let endpoint = source.get(S3_ENDPOINT).unwrap();
        assert_eq!(endpoint.as_plain(), Some("http://127.0.0.1:9000"), "{name}");
        let secret = source.get(S3_SECRET_KEY).unwrap();
        assert!(
            secret.as_plain().is_none(),
            "{name}: a sensitive key is a secret"
        );
        assert!(
            !format!("{secret} {secret:?} {secret:#?}").contains(CANARY),
            "{name}: a secret never prints"
        );
        assert!(source.get("fs.s3a.region").is_none(), "{name}");
    }
}

#[test]
fn an_exact_key_wins_over_a_case_folded_one() {
    let mut map = BTreeMap::new();
    map.insert(
        "FS.S3A.ENDPOINT".to_owned(),
        "https://folded.example.com".to_owned(),
    );
    map.insert(
        S3_ENDPOINT.to_owned(),
        "https://exact.example.com".to_owned(),
    );
    assert_eq!(
        map.lookup(S3_ENDPOINT).unwrap().as_plain(),
        Some("https://exact.example.com")
    );
    assert_eq!(
        map.lookup("Fs.S3a.Endpoint").unwrap().as_plain(),
        Some("https://folded.example.com")
    );
    assert!(map.lookup("fs.s3a.region").is_none());
}

#[test]
fn a_case_folded_key_is_found_in_every_source() {
    for (name, source) in sources(&[("FS.S3A.ENDPOINT", "http://127.0.0.1:9000")]) {
        assert!(
            source.get(S3_ENDPOINT).is_none() || name == "Config",
            "{name}: exact only"
        );
        let found = source.lookup(S3_ENDPOINT);
        assert_eq!(
            found.as_ref().and_then(OptionValue::as_plain),
            Some("http://127.0.0.1:9000"),
            "{name}"
        );
    }
}

#[test]
fn a_hash_map_looks_up_a_folded_key_deterministically() {
    let mut map = HashMap::new();
    map.insert(
        "FS.S3A.ENDPOINT".to_owned(),
        "https://b.example.com".to_owned(),
    );
    map.insert(
        "Fs.S3a.Endpoint".to_owned(),
        "https://a.example.com".to_owned(),
    );
    // Keys are sorted before the first case-folded match is taken.
    for _ in 0..8 {
        assert_eq!(
            map.lookup(S3_ENDPOINT).unwrap().as_plain(),
            Some("https://b.example.com")
        );
    }
}
