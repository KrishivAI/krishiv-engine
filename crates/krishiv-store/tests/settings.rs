// SPDX-License-Identifier: Apache-2.0
//! `StoreSettings`: every `fs.*` key resolves the same from the engine's `Config` and from a plain
//! map of strings (the way an outside user supplies settings), with keys matched ignoring ASCII
//! case, and no secret reaches an error or a `Debug`.
#![cfg(feature = "cloud")]
#![expect(clippy::unwrap_used, reason = "test code")]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;
use krishiv_common::secret::OptionValue;
use krishiv_store::{
    AZURE_ACCOUNT_KEY, AZURE_ACCOUNT_NAME, AZURE_ENDPOINT, AZURE_SAS, GCS_ENDPOINT, GCS_KEY,
    GCS_KEYFILE, GCS_STORAGE_ROOT_URL, HF_ENDPOINT, HF_TOKEN, S3_ACCESS_KEY, S3_ENDPOINT,
    S3_ENDPOINT_REGION, S3_PATH_STYLE_ACCESS, S3_REGION, S3_SECRET_KEY, S3_SESSION_TOKEN,
    StoreRegistry, StoreSettings, parse_object_url,
};

const CANARY: &str = "Canary-5d21b7";

/// What building the store of `url` came to: `Ok` or the error's class and text.
fn outcome(registry: &StoreRegistry, url: &str) -> Result<(), (ErrorClass, String)> {
    let parsed = parse_object_url(url).map_err(|e| (e.error_class(), e.to_string()))?;
    registry
        .store(&parsed)
        .map(|_| ())
        .map_err(|e| (e.error_class(), e.to_string()))
}

/// The registry of `pairs` read from an engine `Config` and from a string map.
fn registries(pairs: &[(&str, &str)]) -> [StoreRegistry; 3] {
    let (config, _) = Config::from_pairs(pairs.to_vec()).unwrap();
    let btree: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let hash: HashMap<String, String> = btree.clone().into_iter().collect();
    [
        StoreRegistry::new(Arc::new(config)),
        StoreRegistry::new(Arc::new(btree)),
        StoreRegistry::new(Arc::new(hash)),
    ]
}

/// Every key of the module docs' table, with a good value, a bad one where the key has a
/// grammar, and the URL it applies to.
fn cases() -> Vec<(&'static str, Vec<(&'static str, &'static str)>)> {
    let s3 = "s3://bucket/k";
    let gs = "gs://bucket/k";
    let az = "abfss://cont@acct.dfs.core.windows.net/k";
    vec![
        (s3, vec![]),
        (s3, vec![(S3_ENDPOINT, "http://127.0.0.1:9000")]),
        (s3, vec![(S3_ENDPOINT, "https://s3.example.com")]),
        (
            s3,
            vec![(S3_ACCESS_KEY, "AKIAEXAMPLE"), (S3_SECRET_KEY, CANARY)],
        ),
        (
            s3,
            vec![
                (S3_SESSION_TOKEN, CANARY),
                (S3_ACCESS_KEY, "k"),
                (S3_SECRET_KEY, "s"),
            ],
        ),
        (
            s3,
            vec![
                (S3_PATH_STYLE_ACCESS, "true"),
                (S3_ENDPOINT, "http://127.0.0.1:9000"),
            ],
        ),
        (s3, vec![(S3_PATH_STYLE_ACCESS, "banana")]),
        (s3, vec![(S3_ENDPOINT_REGION, "eu-west-1")]),
        (s3, vec![(S3_REGION, "eu-west-2")]),
        (s3, vec![(S3_ENDPOINT, "  ")]),
        (gs, vec![(GCS_STORAGE_ROOT_URL, "http://127.0.0.1:4443")]),
        (gs, vec![(GCS_ENDPOINT, "http://127.0.0.1:4443")]),
        (gs, vec![(GCS_KEY, "not json")]),
        (gs, vec![(GCS_KEYFILE, "/nonexistent/krishiv-key.json")]),
        (az, vec![(AZURE_ACCOUNT_KEY, "a2V5")]),
        (
            az,
            vec![("fs.azure.account.key.acct.dfs.core.windows.net", "a2V5")],
        ),
        (az, vec![("fs.azure.account.key.acct", "a2V5")]),
        (az, vec![(AZURE_SAS, "sv=2020&sig=abc")]),
        (
            az,
            vec![(
                "fs.azure.sas.cont.acct.dfs.core.windows.net",
                "?sv=2020&sig=abc",
            )],
        ),
        (
            az,
            vec![("fs.azure.sas.cont.acct.dfs.core.windows.net", "garbage")],
        ),
        (az, vec![(AZURE_ENDPOINT, "http://127.0.0.1:10000/acct")]),
        ("az://cont/k", vec![(AZURE_ACCOUNT_NAME, "acct")]),
        ("az://cont/k", vec![]),
        (
            "hf://datasets/org/name/data.parquet",
            vec![(HF_ENDPOINT, "http://127.0.0.1:8080")],
        ),
        (
            "hf://datasets/org/name/data.parquet",
            vec![(HF_TOKEN, CANARY)],
        ),
        ("https://example.com/data.parquet", vec![]),
    ]
}

#[test]
fn every_key_resolves_the_same_from_a_config_and_from_maps() {
    for (url, pairs) in cases() {
        let [config, btree, hash] = registries(&pairs);
        let want = outcome(&config, url);
        assert_eq!(outcome(&btree, url), want, "{url} {pairs:?}: BTreeMap");
        assert_eq!(outcome(&hash, url), want, "{url} {pairs:?}: HashMap");
        if let Err((_, text)) = &want {
            assert!(!text.contains(CANARY), "{url} {pairs:?}: {text}");
        }
    }
}

#[test]
fn keys_match_ignoring_ascii_case_in_every_source() {
    for (url, pairs) in cases() {
        let upper: Vec<(String, &str)> = pairs
            .iter()
            .map(|(k, v)| (k.to_ascii_uppercase(), *v))
            .collect();
        let upper: Vec<(&str, &str)> = upper.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let [config, btree, hash] = registries(&pairs);
        let [config_u, btree_u, hash_u] = registries(&upper);
        let want = outcome(&config, url);
        // Error texts name the key as written, so compare the class and success only.
        let shape = |r: Result<(), (ErrorClass, String)>| r.map_err(|(class, _)| class);
        for (name, got) in [
            ("config", outcome(&config_u, url)),
            ("BTreeMap", outcome(&btree_u, url)),
            ("HashMap", outcome(&hash_u, url)),
        ] {
            assert_eq!(shape(got), shape(want.clone()), "{url} {pairs:?}: {name}");
        }
        drop((btree, hash));
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
        map.lookup(S3_ENDPOINT).as_deref(),
        Some("https://exact.example.com")
    );
    assert_eq!(
        map.lookup("Fs.S3a.Endpoint").as_deref(),
        Some("https://folded.example.com")
    );
    assert_eq!(map.lookup("fs.s3a.region"), None);
}

#[test]
fn a_secret_is_typed_by_its_key_and_never_prints() {
    // The map holds plain text; the sensitive key makes it a secret as it is read.
    let [config, btree, _] = registries(&[(S3_SECRET_KEY, CANARY), (S3_ACCESS_KEY, "k")]);
    for registry in [&config, &btree] {
        assert!(!format!("{registry:?}").contains(CANARY));
    }
    assert!(
        OptionValue::from_user(S3_SECRET_KEY, CANARY)
            .as_plain()
            .is_none()
    );
    // A secret where a plain value is required still fails, naming the key and not the value.
    let options: BTreeMap<String, OptionValue> = [(
        S3_ENDPOINT.to_owned(),
        OptionValue::from_user(S3_SECRET_KEY, CANARY),
    )]
    .into();
    let url = parse_object_url("s3://bucket/k").unwrap();
    for registry in [&config, &btree] {
        let error = registry.store_with_options(&url, &options).unwrap_err();
        assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
        assert!(!error.to_string().contains(CANARY));
        assert!(!format!("{error:?}").contains(CANARY));
    }
}
