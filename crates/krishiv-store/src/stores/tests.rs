// SPDX-License-Identifier: Apache-2.0
//! Tests of URL parsing, store building and the registry (child module of `stores`).

use super::*;
use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;

#[cfg(feature = "cloud")]
const CANARY: &str = "Canary-7f3a9c";

fn config(pairs: &[(&str, &str)]) -> Arc<Config> {
    let (c, _) = Config::from_pairs(pairs.to_vec()).unwrap();
    Arc::new(c)
}

#[cfg(feature = "cloud")]
fn shown(value: &OptionValue) -> String {
    format!("{value} {value:?} {value:#?}")
}

#[test]
fn urls_parse_for_every_scheme() {
    for (input, scheme, scheme_text, bucket, account, key, store, display) in [
        (
            "s3://bucket/k/d.parquet",
            ObjectScheme::S3,
            "s3",
            "bucket",
            None,
            "k/d.parquet",
            "s3://bucket",
            "s3://bucket/k/d.parquet",
        ),
        (
            "s3a://bucket/k",
            ObjectScheme::S3,
            "s3a",
            "bucket",
            None,
            "k",
            "s3://bucket",
            "s3a://bucket/k",
        ),
        (
            "s3://bucket",
            ObjectScheme::S3,
            "s3",
            "bucket",
            None,
            "",
            "s3://bucket",
            "s3://bucket",
        ),
        (
            "s3://bucket/",
            ObjectScheme::S3,
            "s3",
            "bucket",
            None,
            "",
            "s3://bucket",
            "s3://bucket",
        ),
        (
            "gs://bucket/k",
            ObjectScheme::Gcs,
            "gs",
            "bucket",
            None,
            "k",
            "gs://bucket",
            "gs://bucket/k",
        ),
        (
            "abfss://cont@acct.dfs.core.windows.net/k",
            ObjectScheme::Azure,
            "abfss",
            "cont",
            Some("acct"),
            "k",
            "abfss://cont@acct",
            "abfss://cont@acct.dfs.core.windows.net/k",
        ),
        (
            "abfs://cont@acct.blob.core.windows.net/",
            ObjectScheme::Azure,
            "abfs",
            "cont",
            Some("acct"),
            "",
            "abfss://cont@acct",
            "abfs://cont@acct.blob.core.windows.net",
        ),
        (
            "abfss://cont/k",
            ObjectScheme::Azure,
            "abfss",
            "cont",
            None,
            "k",
            "abfss://cont",
            "abfss://cont/k",
        ),
        (
            "az://cont/k",
            ObjectScheme::Azure,
            "az",
            "cont",
            None,
            "k",
            "abfss://cont",
            "az://cont/k",
        ),
    ] {
        let url = parse_object_url(input).unwrap();
        assert_eq!(url.scheme, scheme, "{input}");
        assert_eq!(url.scheme_text, scheme_text, "{input}");
        assert_eq!(url.bucket, bucket, "{input}");
        assert_eq!(url.account.as_deref(), account, "{input}");
        assert_eq!(url.key.as_ref(), key, "{input}");
        assert_eq!(url.store_id, StoreId::new(store), "{input}");
        assert_eq!(url.display, display, "{input}");
        // The display re-parses to the same store and key.
        assert_eq!(parse_object_url(&url.display).unwrap(), url, "{input}");
    }
    // Normalization (Hadoop-style) and scheme case.
    let url = parse_object_url("s3://b/a/./c/../d").unwrap();
    assert_eq!(url.key.as_ref(), "a/d");
    assert_eq!(url.display, "s3://b/a/d");
    assert_eq!(parse_object_url(&url.display).unwrap(), url);
    let url = parse_object_url("S3://b/k").unwrap();
    assert_eq!(url.scheme, ObjectScheme::S3);
    assert_eq!(url.display, "S3://b/k");
    assert_eq!(
        (
            ObjectScheme::S3.to_string(),
            ObjectScheme::Gcs.to_string(),
            ObjectScheme::Azure.to_string()
        ),
        ("s3".to_owned(), "gs".to_owned(), "abfss".to_owned())
    );
}

#[test]
fn unknown_schemes_and_bad_urls_are_unsupported() {
    for input in [
        "ftp://x/y",
        "hdfs://namenode/k",
        "file:///a",
        "s3:bucket/k",
        "s3:///k",
        "s3://",
        "gs://",
        "abfss://",
        "no-scheme/k",
        "s3a://AKIA:secret@bucket/k",
        "gs://user@bucket/k",
        "abfss://user:pass@cont/k",
    ] {
        let error = parse_object_url(input).unwrap_err();
        assert_eq!(
            error.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{input}"
        );
    }
}

#[cfg(feature = "cloud")]
#[test]
fn config_keys_are_typed_secrets() {
    // Keys the shared classifier marks sensitive classify as secrets
    // that never print (the redaction tests' style).
    for key in [
        S3_ACCESS_KEY,
        S3_SECRET_KEY,
        S3_SESSION_TOKEN,
        "fs.azure.account.key.acct.dfs.core.windows.net",
        "fs.azure.sas.cont.acct",
        AZURE_ACCOUNT_KEY,
        AZURE_SAS,
    ] {
        let value = OptionValue::from_user(key, CANARY);
        assert!(value.as_plain().is_none(), "{key}");
        assert!(!shown(&value).contains(CANARY), "{key}");
    }
    // Endpoints, regions and account names stay plain.
    for key in [S3_ENDPOINT, S3_REGION, AZURE_ENDPOINT, AZURE_ACCOUNT_NAME] {
        assert_eq!(
            OptionValue::from_user(key, CANARY).as_plain(),
            Some(CANARY),
            "{key}"
        );
    }
    // Every credential value the registry reads is a `SecretString`
    // that only exposes at the builder call, including the GCS ones
    // the shared classifier leaves plain.
    let cfg = config(&[
        (GCS_KEYFILE, CANARY),
        (GCS_KEY, CANARY),
        (S3_ACCESS_KEY, CANARY),
    ]);
    for key in [GCS_KEYFILE, GCS_KEY, S3_ACCESS_KEY] {
        let held = cred(&cfg, &BTreeMap::new(), key).unwrap().unwrap();
        assert_eq!(held.expose_secret(), CANARY);
        assert!(!format!("{held:?}").contains(CANARY), "{key}");
    }
}

#[cfg(feature = "cloud")]
#[test]
fn stores_build_offline_without_touching_the_network() {
    let s3 = StoreRegistry::new(config(&[
        (S3_ENDPOINT, "http://127.0.0.1:9000"),
        (S3_ACCESS_KEY, CANARY),
        (S3_SECRET_KEY, CANARY),
        (S3_SESSION_TOKEN, CANARY),
        (S3_ENDPOINT_REGION, "us-east-1"),
        (S3_PATH_STYLE_ACCESS, "true"),
    ]));
    let url = parse_object_url("s3a://bucket/k").unwrap();
    s3.store(&url).unwrap();
    assert_eq!(url.store_id, StoreId::new("s3://bucket"));

    let gcs = StoreRegistry::new(config(&[
        (
            GCS_KEY,
            r#"{"private_key": "key", "private_key_id": "id", "client_email": "mail", "disable_oauth": true}"#,
        ),
        (GCS_STORAGE_ROOT_URL, "http://127.0.0.1:4443"),
    ]));
    gcs.store(&parse_object_url("gs://bucket/k").unwrap())
        .unwrap();

    let azure = StoreRegistry::new(config(&[
        (AZURE_ACCOUNT_NAME, "devstoreaccount1"),
        (
            "fs.azure.account.key.devstoreaccount1",
            "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
        ),
        (AZURE_ENDPOINT, "http://127.0.0.1:10000/devstoreaccount1"),
    ]));
    azure
        .store(&parse_object_url("abfss://cont/k").unwrap())
        .unwrap();
}

#[cfg(feature = "cloud")]
#[test]
fn failures_name_keys_and_buckets_but_never_secrets() {
    // A store that cannot be built names what is missing, never a
    // secret: no account here, however the key is set. The process
    // environment can supply the missing account through the builders'
    // `from_env` fallback, turning the expected failure into a success;
    // either way nothing secret prints (no skipped case).
    let registry = StoreRegistry::new(config(&[(AZURE_ACCOUNT_KEY, CANARY)]));
    let url = parse_object_url("abfss://cont/k").unwrap();
    match registry.store(&url) {
        Ok(_) => assert!(!format!("{registry:?}").contains(CANARY)),
        Err(error) => {
            assert_eq!(
                error.error_class(),
                ErrorClass::KrishivObjectStoreRequestFailed
            );
            assert!(!error.to_string().contains(CANARY));
            assert!(!format!("{error:?}").contains(CANARY));
            assert!(!format!("{registry:?}").contains(CANARY));
        }
    }

    // An invalid flag names the key; the value is only the marker.
    let registry = StoreRegistry::new(config(&[(S3_PATH_STYLE_ACCESS, CANARY)]));
    let error = registry
        .store(&parse_object_url("s3://bucket/k").unwrap())
        .unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_PATH_STYLE_ACCESS));
    assert!(!error.to_string().contains(CANARY));

    // A malformed SAS token never reaches an error either.
    let registry = StoreRegistry::new(config(&[
        (AZURE_ACCOUNT_NAME, "acct"),
        ("fs.azure.sas.cont.acct", CANARY),
    ]));
    let error = registry
        .store(&parse_object_url("abfss://cont@acct.dfs.core.windows.net/k").unwrap())
        .unwrap_err();
    assert_eq!(
        error.error_class(),
        ErrorClass::KrishivObjectStoreRequestFailed
    );
    assert!(!error.to_string().contains(CANARY));

    // Both GCS credential forms at once is refused without echo.
    let registry = StoreRegistry::new(config(&[(GCS_KEYFILE, CANARY), (GCS_KEY, CANARY)]));
    let error = registry
        .store(&parse_object_url("gs://bucket/k").unwrap())
        .unwrap_err();
    assert!(!error.to_string().contains(CANARY));

    // Half a key pair is refused without echo. The environment can
    // complete the pair (`AWS_ACCESS_KEY_ID` through `from_env`),
    // turning the failure into a success; either way nothing secret
    // prints (no skipped case).
    let registry = StoreRegistry::new(config(&[(S3_SECRET_KEY, CANARY)]));
    match registry.store(&parse_object_url("s3://bucket/k").unwrap()) {
        Ok(_) => assert!(!format!("{registry:?}").contains(CANARY)),
        Err(error) => assert!(!error.to_string().contains(CANARY)),
    }
}

#[cfg(feature = "cloud")]
#[test]
fn stores_are_cached_per_bucket() {
    let registry = StoreRegistry::new(config(&[]));
    let first = parse_object_url("s3://b1/k").unwrap();
    let one = registry.store(&first).unwrap();
    assert!(Arc::ptr_eq(&one, &registry.store(&first).unwrap()));
    let other = registry
        .store(&parse_object_url("s3://b2/k").unwrap())
        .unwrap();
    assert!(!Arc::ptr_eq(&one, &other));
    assert!(format!("{registry:?}").contains("cached"));
}

fn read_options(pairs: &[(&str, &str)]) -> BTreeMap<String, OptionValue> {
    pairs
        .iter()
        .map(|&(k, v)| (k.to_owned(), OptionValue::from_user(k, v)))
        .collect()
}

#[cfg(feature = "cloud")]
#[test]
fn per_read_options_override_session_config() {
    use super::ObjectScheme;
    // An invalid session flag fails on its own ...
    let registry = StoreRegistry::new(config(&[(S3_PATH_STYLE_ACCESS, "banana")]));
    let url = parse_object_url("s3://bucket/k").unwrap();
    let error = registry.store(&url).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    // ... but a valid per-read option overrides it (keys match ASCII
    // case-insensitively, as Spark's `CaseInsensitiveStringMap`).
    let options = read_options(&[("FS.S3A.PATH.STYLE.ACCESS", "true")]);
    registry.store_with_options(&url, &options).unwrap();
    // And an invalid per-read option beats a valid session value.
    let registry = StoreRegistry::new(config(&[(S3_PATH_STYLE_ACCESS, "true")]));
    let error = registry
        .store_with_options(&url, &read_options(&[(S3_PATH_STYLE_ACCESS, "banana")]))
        .unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_PATH_STYLE_ACCESS));
    assert!(!error.to_string().contains("banana"));
    // An empty per-read value is a present value, so it fails rather
    // than falling back to the session value (Hadoop keeps `""` too).
    let error = registry
        .store_with_options(&url, &read_options(&[(S3_PATH_STYLE_ACCESS, "")]))
        .unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_PATH_STYLE_ACCESS));
    assert_eq!(url.scheme, ObjectScheme::S3);
}

#[cfg(feature = "cloud")]
#[test]
fn session_keys_match_case_insensitively_like_per_read_keys() {
    // The session side resolves the same case-insensitive way the
    // per-read side does: one spelling means the same thing in both.
    let registry = StoreRegistry::new(config(&[("FS.S3A.PATH.STYLE.ACCESS", "true")]));
    let url = parse_object_url("s3://bucket/k").unwrap();
    registry.store(&url).unwrap();
    let registry = StoreRegistry::new(config(&[("fs.s3a.path.style.access", "true")]));
    registry
        .store_with_options(
            &url,
            &read_options(&[("FS.S3A.PATH.STYLE.ACCESS", "false")]),
        )
        .unwrap();
    // An invalid value under another spelling still fails, naming the key.
    let error = registry
        .store_with_options(
            &url,
            &read_options(&[("FS.S3A.PATH.STYLE.ACCESS", "banana")]),
        )
        .unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
}

#[cfg(feature = "cloud")]
#[test]
fn mistyped_per_read_values_fail_naming_the_key() {
    // Every per-read value below would have fallen back to the valid
    // session value before: now each fails with a typed invalid-option
    // error that names the key but never the value.
    let session = config(&[
        (S3_ACCESS_KEY, "session-key"),
        (S3_ENDPOINT, "http://127.0.0.1:9000"),
        (S3_PATH_STYLE_ACCESS, "true"),
    ]);
    // A reference where a credential is required: this layer cannot
    // resolve it, so it fails rather than using the session value.
    let mut options = BTreeMap::new();
    options.insert(
        S3_ACCESS_KEY.to_owned(),
        OptionValue::reference("scope", CANARY),
    );
    let error = cred(&session, &options, S3_ACCESS_KEY).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_ACCESS_KEY));
    assert!(!error.to_string().contains(CANARY));
    // A URL value is never a credential for these keys.
    let mut options = BTreeMap::new();
    options.insert(
        S3_ACCESS_KEY.to_owned(),
        OptionValue::from_user(S3_ENDPOINT, "https://user:pw@host"),
    );
    let error = cred(&session, &options, S3_ACCESS_KEY).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_ACCESS_KEY));
    assert!(!error.to_string().contains("pw"));
    // A secret where a plain endpoint is required.
    let mut options = BTreeMap::new();
    options.insert(
        S3_ENDPOINT.to_owned(),
        OptionValue::from_user(S3_SECRET_KEY, CANARY),
    );
    let error = text(&session, &options, S3_ENDPOINT).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_ENDPOINT));
    assert!(!error.to_string().contains(CANARY));
    // A secret where a flag is required.
    let mut options = BTreeMap::new();
    options.insert(
        S3_PATH_STYLE_ACCESS.to_owned(),
        OptionValue::from_user(S3_SECRET_KEY, CANARY),
    );
    let error = flag(&session, &options, S3_PATH_STYLE_ACCESS).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_PATH_STYLE_ACCESS));
    assert!(!error.to_string().contains(CANARY));
}

#[cfg(feature = "cloud")]
#[test]
fn empty_per_read_values_are_present_values() {
    // Hadoop keeps `""` as a value: an empty per-read value is invalid
    // for these keys, so it fails rather than falling back to the
    // valid session value.
    let session = config(&[
        (S3_ACCESS_KEY, "session-key"),
        (S3_ENDPOINT, "http://127.0.0.1:9000"),
        (S3_PATH_STYLE_ACCESS, "true"),
    ]);
    let error = cred(
        &session,
        &read_options(&[(S3_ACCESS_KEY, "")]),
        S3_ACCESS_KEY,
    )
    .unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_ACCESS_KEY));
    let error = text(&session, &read_options(&[(S3_ENDPOINT, "")]), S3_ENDPOINT).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(error.to_string().contains(S3_ENDPOINT));
    // Whitespace-only endpoints trim to empty, so they fail too.
    let error = text(&session, &read_options(&[(S3_ENDPOINT, "  ")]), S3_ENDPOINT).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
}

#[cfg(feature = "cloud")]
#[test]
fn per_read_options_get_their_own_store_id() {
    let registry = StoreRegistry::new(config(&[]));
    let url = parse_object_url("s3://bucket/k").unwrap();
    let base = registry.store(&url).unwrap();
    let one = registry
        .store_with_options(
            &url,
            &read_options(&[(S3_ENDPOINT, "http://127.0.0.1:9001")]),
        )
        .unwrap();
    let same = registry
        .store_with_options(
            &url,
            &read_options(&[(S3_ENDPOINT, "http://127.0.0.1:9001")]),
        )
        .unwrap();
    let other = registry
        .store_with_options(
            &url,
            &read_options(&[(S3_ENDPOINT, "http://127.0.0.1:9002")]),
        )
        .unwrap();
    // Same effective options share the store; different ones do not,
    // and neither shares the session-configured store.
    assert!(Arc::ptr_eq(&one, &same));
    assert!(!Arc::ptr_eq(&base, &one));
    assert!(!Arc::ptr_eq(&one, &other));
    // Non-`fs.*` options never split the store.
    let plain = registry
        .store_with_options(&url, &read_options(&[("compression", "zstd")]))
        .unwrap();
    assert!(Arc::ptr_eq(&base, &plain));
}

#[cfg(feature = "cloud")]
#[test]
fn per_read_secrets_never_print() {
    // A per-read secret overrides the session value without ever
    // printing: not in the error, not in the registry's Debug, not in
    // the option value's Display (what EXPLAIN prints for the plan's
    // option map), and not in the suffixed store id.
    let registry = StoreRegistry::new(config(&[]));
    let url = parse_object_url("s3://bucket/k").unwrap();
    let options = read_options(&[(S3_SECRET_KEY, CANARY), (S3_PATH_STYLE_ACCESS, "banana")]);
    let error = registry.store_with_options(&url, &options).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(!error.to_string().contains(CANARY));
    assert!(!format!("{error:?}").contains(CANARY));
    assert!(!format!("{registry:?}").contains(CANARY));
    let held = OptionValue::from_user(S3_SECRET_KEY, CANARY);
    assert!(held.as_plain().is_none());
    assert!(!held.to_string().contains(CANARY));
    assert!(!format!("{held:?}").contains(CANARY));
    // `open_with_options` returns a suffixed id whose text is also free of the secret.
    let opened = registry
        .open_with_options(
            "s3://bucket/k",
            &read_options(&[
                (S3_ENDPOINT, "http://127.0.0.1:9001"),
                (S3_ACCESS_KEY, CANARY),
                (S3_SECRET_KEY, CANARY),
            ]),
        )
        .unwrap();
    assert!(!opened.id.to_string().contains(CANARY));
    assert!(!format!("{:?}", opened.id).contains(CANARY));
    assert!(!format!("{opened:?}").contains(CANARY));
    assert_eq!(opened.policy, StorePolicy::S3);
}

#[test]
fn unknown_schemes_still_fail_with_options() {
    let registry = StoreRegistry::new(config(&[]));
    let options = read_options(&[(S3_ENDPOINT, "http://127.0.0.1:9001")]);
    for input in ["ftp://x/y", "s3:bucket/k", "s3a://AKIA:secret@bucket/k"] {
        let error = parse_object_url(input).unwrap_err();
        assert_eq!(
            error.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{input}"
        );
        let error = registry.open_with_options(input, &options).unwrap_err();
        assert_eq!(
            error.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{input}"
        );
    }
}

#[test]
fn store_option_keys_never_slice_mid_char() {
    // `is_store_option` compares bytes, so keys with a multi-byte
    // character at byte 3 never panic (e.g. `"ééx"`).
    assert!(!is_store_option("ééx"));
    assert!(!is_store_option("é"));
    assert!(!is_store_option(""));
    assert!(!is_store_option("fs"));
    assert!(!is_store_option("fs."));
    assert!(is_store_option("fs.a"));
    assert!(is_store_option("FS.S3A.ENDPOINT"));
    // `relevant_options` (which calls it) skips non-`fs.*` keys without
    // panicking, whatever their bytes.
    let options = read_options(&[("ééx", "1"), ("compression", "zstd")]);
    assert!(relevant_options(&options).is_empty());
    let options = read_options(&[("FS.S3A.ENDPOINT", "http://127.0.0.1:9000")]);
    assert_eq!(relevant_options(&options).len(), 1);
}

#[cfg(feature = "cloud")]
#[test]
fn sas_tokens_allow_a_leading_question_mark() {
    let with_mark = SecretString::new("?sv=2024-01-01&sig=abc".to_owned());
    assert_eq!(
        sas_pairs(&with_mark).unwrap(),
        vec![
            ("sv".to_owned(), "2024-01-01".to_owned()),
            ("sig".to_owned(), "abc".to_owned())
        ]
    );
    let without_mark = SecretString::new("sv=2024-01-01&sig=abc".to_owned());
    assert_eq!(
        sas_pairs(&without_mark).unwrap(),
        vec![
            ("sv".to_owned(), "2024-01-01".to_owned()),
            ("sig".to_owned(), "abc".to_owned())
        ]
    );
    // Empty names or values are not pairs.
    for bad in ["", "?", "?&", "sv=&sig=abc", "sv=2024-01-01&sig="] {
        assert!(
            sas_pairs(&SecretString::new(bad.to_owned())).is_err(),
            "{bad}"
        );
    }
}

#[cfg(feature = "cloud")]
#[test]
fn http_scheme_matches_ascii_case_insensitively() {
    assert!(allows_http(Some("http://127.0.0.1:9000")));
    assert!(allows_http(Some("HTTP://127.0.0.1:9000")));
    assert!(!allows_http(Some("https://example.com")));
    assert!(!allows_http(None));
}

#[test]
fn cloud_schemes_pace_with_their_own_policy() {
    assert_eq!(policy_for(ObjectScheme::S3), StorePolicy::S3);
    assert_eq!(policy_for(ObjectScheme::Gcs), StorePolicy::GCS);
    assert_eq!(policy_for(ObjectScheme::Azure), StorePolicy::AZURE);
}

/// Per-read `fs.*` options isolate the store id by a hash of their
/// values: the same options always give the same id, any change gives
/// another, keys match case-insensitively, other options are ignored,
/// and no value (plain, inline secret, reference or credentialed URL)
/// reaches the id's text.
#[test]
fn per_read_options_isolate_store_ids_without_leaking_values() {
    let base = StoreId::new("s3://bucket");
    assert_eq!(effective_store_id(&base, &BTreeMap::new()), base);
    assert_eq!(
        effective_store_id(&base, &read_options(&[("header", "true")])),
        base
    );
    let id = |pairs: &[(&str, &str)]| effective_store_id(&base, &read_options(pairs));
    let plain = id(&[(S3_ENDPOINT, "http://a:9000")]);
    assert!(plain.key().starts_with("s3://bucket#"));
    // Messages name the location only.
    assert_eq!(plain.to_string(), "s3://bucket");
    assert_eq!(plain, id(&[(S3_ENDPOINT, "http://a:9000")]));
    assert_ne!(plain, id(&[(S3_ENDPOINT, "http://b:9000")]));
    assert_eq!(
        plain,
        id(&[("FS.S3A.ENDPOINT", "http://a:9000"), ("header", "true")])
    );
    let secret = id(&[(S3_SECRET_KEY, "Canary-1")]);
    assert_ne!(secret, id(&[(S3_SECRET_KEY, "Canary-2")]));
    assert_ne!(secret, plain);
    let mut by_reference = read_options(&[]);
    by_reference.insert(
        S3_SECRET_KEY.to_owned(),
        OptionValue::reference("scope", "k"),
    );
    let referenced = effective_store_id(&base, &by_reference);
    assert_ne!(referenced, secret);
    let mut other = read_options(&[]);
    other.insert(
        S3_SECRET_KEY.to_owned(),
        OptionValue::reference("scope", "k2"),
    );
    assert_ne!(effective_store_id(&base, &other), referenced);
    let url = id(&[(S3_ENDPOINT, "https://user:Canary-3@host:9000/x")]);
    assert_ne!(url, plain);
    for shown in [&secret, &referenced, &url] {
        assert!(
            !shown.to_string().contains("Canary"),
            "a store id carries a credential value"
        );
    }
}

/// Per-read options reach the same configuration gate as session
/// config: without the `cloud` feature nothing builds, and a failed
/// build registers nothing.
#[cfg(not(feature = "cloud"))]
#[test]
fn per_read_options_without_the_cloud_feature_fail_to_build() {
    let registry = StoreRegistry::new(config(&[]));
    let url = parse_object_url("s3://bucket/k").unwrap();
    let options = read_options(&[(S3_ENDPOINT, "http://127.0.0.1:9000")]);
    let error = registry.store_with_options(&url, &options).unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let error = registry
        .open_with_options("s3://bucket/k", &options)
        .unwrap_err();
    assert_eq!(error.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(format!("{registry:?}").contains("cached: 0"));
}

#[cfg(not(feature = "cloud"))]
#[test]
fn cloud_stores_need_the_cloud_feature() {
    // Without the feature, cloud URLs still parse (so plans qualify
    // paths) but no store is built: the exact unsupported error names
    // the missing feature, never a secret.
    let registry = StoreRegistry::new(config(&[]));
    for input in ["s3://bucket/k", "gs://bucket/k", "abfss://cont/k"] {
        let url = parse_object_url(input).unwrap();
        let error = registry.store(&url).unwrap_err();
        assert_eq!(
            error.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{input}"
        );
        assert!(error.to_string().contains("`cloud`"), "{input}");
    }
}
/// The web schemes' error paths and names, in the default build (parsing is not behind `cloud`).
#[test]
fn web_urls_and_hf_names_are_checked_when_parsed() {
    for bad in [
        "https://example.com/a?token=1",
        "http://example.com/a#frag",
        "hf://models/acme/tiny/x",
        "hf://datasets/../tiny/x",
        "hf://datasets/acme/bad name/x",
        "hf://datasets/acme/tiny@/x",
    ] {
        let e = parse_object_url(bad).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{bad}"
        );
    }
    let hf = parse_object_url("hf://datasets/acme/tiny@v2/data/t.parquet").unwrap();
    assert_eq!(hf.scheme.to_string(), "hf");
    assert_eq!(hf.revision.as_deref(), Some("v2"));
    let web = parse_object_url("https://example.com/a/b.parquet").unwrap();
    assert_eq!(web.scheme.to_string(), "https");
    // Neither web scheme is paced like a cloud service.
    for scheme in [ObjectScheme::Http, ObjectScheme::HuggingFace] {
        assert_eq!(
            format!("{:?}", policy_for(scheme)),
            format!("{:?}", StorePolicy::UNPACED)
        );
    }
}

/// One `IoScheduler` serves every session of an engine runtime (design
/// P2-E1 R3): a store built from a session's `fs.*` settings is registered
/// under an id carrying a hash of all of them, so sessions that differ in any
/// setting (here only the endpoint) never share a store or a cache entry,
/// equal settings share one, no settings keep the bare id, and no secret is
/// in the id.
#[test]
fn store_ids_carry_a_hash_of_every_fs_setting() {
    let url = parse_object_url("s3://bucket/k/d.parquet").unwrap();
    let none = BTreeMap::new();
    let id = |pairs: &[(&str, &str)]| {
        let registry = StoreRegistry::new(config(pairs));
        registry.id_of(&url, None, &none)
    };
    let a = id(&[("fs.s3a.endpoint", "http://a:9000")]);
    let b = id(&[("fs.s3a.endpoint", "http://b:9000")]);
    assert_ne!(a, b, "an endpoint alone separates the stores");
    assert_eq!(a, id(&[("fs.s3a.endpoint", "http://a:9000")]));
    assert!(a.key().starts_with("s3://bucket#cfg:"), "{}", a.key());
    assert_eq!(id(&[]), url.store_id, "no fs.* setting: the bare id");
    assert_eq!(id(&[("spark.sql.ansi.enabled", "false")]), url.store_id);
    // Keys match without case, as Spark's do.
    assert_eq!(a, id(&[("FS.S3A.ENDPOINT", "http://a:9000")]));
    // Secrets: different secrets differ, and none appears in the id.
    let canary = "Canary-31337-secret";
    let s1 = id(&[("fs.s3a.secret.key", canary)]);
    let s2 = id(&[("fs.s3a.secret.key", "another")]);
    assert_ne!(s1, s2);
    assert!(!s1.key().contains(canary));
    assert!(!format!("{s1} {s1:?}").contains(canary));
    // Messages show the location only: no hash, no fingerprint.
    assert_eq!(a.to_string(), "s3://bucket");
    // One scheduler serves many principals: equal settings, different
    // principals, different ids (a registration never replaces another
    // principal's, whose block-cache scope differs); the same principal shares.
    let as_principal = |who: &str| {
        StoreRegistry::new(config(&[("fs.s3a.endpoint", "http://a:9000")]))
            .with_cache_scope(CacheScope::Principal(who.to_owned()))
            .id_of(&url, None, &none)
    };
    assert_ne!(as_principal("alice"), as_principal("bob"));
    assert_eq!(as_principal("alice"), as_principal("alice"));
    assert!(as_principal("alice").key().contains("#p:"));
    assert!(!as_principal("alice").key().contains("alice"));
    assert_eq!(as_principal("alice").to_string(), "s3://bucket");
    // Per-read options still add their own hash.
    let mut options = BTreeMap::new();
    options.insert(
        "fs.s3a.endpoint".to_owned(),
        OptionValue::Plain("http://c:9000".to_owned()),
    );
    let registry = StoreRegistry::new(config(&[("fs.s3a.endpoint", "http://a:9000")]));
    assert_ne!(registry.id_of(&url, None, &options), a);
}

/// object_store formats an S3 region into the S3 and STS host names, so a region that could name
/// another host (`evil.example#`, `/`, `@`, upper case, too long) is refused, per read or in the
/// session, before a store is built (P3-T067b review B1).
#[cfg(feature = "cloud")]
#[test]
fn a_region_cannot_name_a_host() {
    let url = parse_object_url("s3://bucket/k").unwrap();
    let long = "a".repeat(33);
    for key in [S3_REGION, S3_ENDPOINT_REGION] {
        for bad in [
            "evil.example#",
            "localhost:9000/",
            "a@b",
            "EU-WEST-1",
            "",
            long.as_str(),
        ] {
            let error = StoreRegistry::new(config(&[]))
                .store_with_options(&url, &read_options(&[(key, bad)]))
                .err()
                .unwrap_or_else(|| panic!("{key}={bad} was accepted"));
            assert_eq!(
                error.error_class(),
                ErrorClass::KrishivInvalidConfigValue,
                "{key}={bad}"
            );
            // (An empty session value is an unset one: the default region.)
            if !bad.is_empty() {
                let session = StoreRegistry::new(config(&[(key, bad)]));
                assert!(session.store(&url).is_err(), "{key}={bad} in the session");
            }
        }
        StoreRegistry::new(config(&[(key, "eu-west-1")]))
            .store(&url)
            .unwrap();
    }
}
