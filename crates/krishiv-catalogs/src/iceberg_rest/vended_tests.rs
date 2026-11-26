// SPDX-License-Identifier: Apache-2.0
//! Vended prefixes, credential keys and expiries (design P3-T067 §4.1, §8 `vended_prefix_forms`).

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use krishiv_common::secret::SecretString;
use krishiv_store::parse_object_url;

use super::{
    VendedFault, VendedPrefix, answer_expires_at, covering, credential, keep_credentials, narrowed,
};
use crate::iceberg_rest::types::StorageCredential;

const TTL: Duration = Duration::from_secs(900);

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, SecretString> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), SecretString::new((*v).to_owned())))
        .collect()
}

fn covers(prefix: &str, location: &str) -> bool {
    VendedPrefix::parse(prefix)
        .unwrap()
        .covers(&parse_object_url(location).unwrap())
}

#[test]
fn vended_prefix_forms() {
    // A full URL matches on whole segments.
    assert!(covers("s3://b/wh/t", "s3://b/wh/t/data/f.parquet"));
    assert!(covers("s3://b/wh/t/", "s3://b/wh/t/data/f.parquet"));
    assert!(!covers("s3://b/wh/t", "s3://b/wh/t10/data/f.parquet"));
    assert!(!covers("s3://b/wh/t", "s3://other/wh/t/f.parquet"));
    // A bucket covers its every key, across the scheme family.
    assert!(covers("s3://b", "s3://b/x/y.parquet"));
    assert!(covers("s3://b/", "s3a://b/x/y.parquet"));
    assert!(covers("s3a://b/wh", "s3://b/wh/y.parquet"));
    assert!(!covers("s3://b", "gs://b/x"));
    // A bare scheme covers its family.
    assert!(covers("s3", "s3a://any/k"));
    assert!(covers("s3a", "s3://any/k"));
    assert!(covers("abfss", "abfs://c@acct.dfs.core.windows.net/k"));
    assert!(covers("gs", "gs://b/k"));
    assert!(!covers("gs", "s3://b/k"));
    // A partial key never matches across a segment boundary.
    assert!(!covers("s3://b/wh/ta", "s3://b/wh/table/f"));
    // Schemes with no vended credentials, and text that is no location, are faults.
    for bad in [
        "https",
        "https://h/p",
        "hf://datasets/o/n",
        "file",
        "file:///tmp/x",
        "s3:/b",
    ] {
        assert!(VendedPrefix::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn the_longest_prefix_wins_and_config_covers_only_the_table_location() {
    let creds = vec![
        StorageCredential {
            prefix: "s3".to_owned(),
            config: map(&[("s3.access-key-id", "wide")]),
        },
        StorageCredential {
            prefix: "s3://b/wh/t".to_owned(),
            config: map(&[("s3.access-key-id", "narrow")]),
        },
        StorageCredential {
            prefix: "s3://b".to_owned(),
            config: map(&[("s3.access-key-id", "bucket")]),
        },
    ];
    let config = map(&[("s3.access-key-id", "config")]);
    let pick = |loc: &str| {
        covering(
            &creds,
            &config,
            "s3://b/wh/t",
            &parse_object_url(loc).unwrap(),
        )
        .unwrap()
        .map(|(_, c)| c["s3.access-key-id"].expose_secret().to_owned())
    };
    assert_eq!(pick("s3://b/wh/t/data/f").as_deref(), Some("narrow"));
    assert_eq!(pick("s3://b/other/f").as_deref(), Some("bucket"));
    assert_eq!(pick("s3://c/f").as_deref(), Some("wide"));
    assert_eq!(pick("gs://b/f"), None);
    // With no storage-credentials entry, `config` covers the table's own location only.
    let own = |loc: &str| {
        covering(&[], &config, "s3://b/wh/t", &parse_object_url(loc).unwrap())
            .unwrap()
            .is_some()
    };
    assert!(own("s3://b/wh/t/data/f"));
    assert!(!own("s3://b/wh/t2/f"));
    assert!(!own("s3://b/f"));
}

#[test]
fn keys_of_each_scheme_become_its_credential() {
    let now = UNIX_EPOCH + Duration::from_secs(1_000);
    let at = |ms: u64| UNIX_EPOCH + Duration::from_millis(ms);
    let s3 = parse_object_url("s3://b/k").unwrap();
    let c = credential(
        &map(&[
            ("s3.access-key-id", "AKIA"),
            ("s3.secret-access-key", "sec"),
            ("s3.session-token", "tok"),
            ("s3.session-token-expires-at-ms", "5000000"),
        ]),
        &s3,
        now,
        TTL,
    )
    .unwrap()
    .unwrap();
    assert!(format!("{c:?}").contains("\"s3\""));
    assert_eq!(c.expires_at(), Some(at(5_000_000)));
    let gcs = parse_object_url("gs://b/k").unwrap();
    let c = credential(
        &map(&[
            ("gcs.oauth2.token", "ya29"),
            ("gcs.oauth2.token-expires-at", "7000000"),
        ]),
        &gcs,
        now,
        TTL,
    )
    .unwrap()
    .unwrap();
    assert!(format!("{c:?}").contains("\"gcs\""));
    assert_eq!(c.expires_at(), Some(at(7_000_000)));
    // ADLS: keyed by the account host exactly.
    let adls = parse_object_url("abfss://c@acct.dfs.core.windows.net/k").unwrap();
    let sas = map(&[
        ("adls.sas-token.acct.dfs.core.windows.net", "sv=1&sig=x"),
        (
            "adls.sas-token-expires-at-ms.acct.dfs.core.windows.net",
            "9000000",
        ),
    ]);
    let c = credential(&sas, &adls, now, TTL).unwrap().unwrap();
    assert!(format!("{c:?}").contains("azure-sas"));
    assert_eq!(c.expires_at(), Some(at(9_000_000)));
    let other = parse_object_url("abfss://c@other.dfs.core.windows.net/k").unwrap();
    assert!(credential(&sas, &other, now, TTL).unwrap().is_none());
    // A credential of another scheme is none for this one.
    assert!(credential(&sas, &s3, now, TTL).unwrap().is_none());
}

#[test]
fn a_missing_expiry_is_the_default_ttl_and_a_bad_one_a_fault() {
    let now = SystemTime::now();
    let s3 = parse_object_url("s3://b/k").unwrap();
    let pair = [
        ("s3.access-key-id", "AKIA"),
        ("s3.secret-access-key", "sec"),
    ];
    let c = credential(&map(&pair), &s3, now, TTL).unwrap().unwrap();
    assert_eq!(c.expires_at(), Some(now + TTL));
    for bad in ["soon", "-1", "1.5", ""] {
        let mut m = map(&pair);
        m.insert(
            "s3.session-token-expires-at-ms".to_owned(),
            SecretString::new(bad.to_owned()),
        );
        let VendedFault(detail) = credential(&m, &s3, now, TTL).unwrap_err();
        assert!(
            detail.contains("s3.session-token-expires-at-ms"),
            "{detail}"
        );
    }
}

#[test]
fn half_a_key_pair_is_a_fault_naming_keys_only() {
    let s3 = parse_object_url("s3://b/k").unwrap();
    for pair in [
        [("s3.access-key-id", "AKIA-canary-1")],
        [("s3.secret-access-key", "secret-canary-2")],
    ] {
        let VendedFault(detail) = credential(&map(&pair), &s3, SystemTime::now(), TTL).unwrap_err();
        assert!(!detail.contains("canary"), "{detail}");
    }
}

#[test]
fn catalog_config_token_is_secret() {
    // Only the credential keys are kept; the table `token`, its RFC 8693 type keys, refresh endpoints, `signer.*`
    // and store options are dropped, so nothing follows or forwards them.
    let kept = [
        "s3.access-key-id",
        "s3.secret-access-key",
        "s3.session-token",
        "s3.session-token-expires-at-ms",
        "gcs.oauth2.token",
        "gcs.oauth2.token-expires-at",
        "adls.sas-token.acct.dfs.core.windows.net",
        "adls.sas-token-expires-at-ms.acct.dfs.core.windows.net",
    ];
    let dropped = [
        "token",
        "urn:ietf:params:oauth:token-type:access_token",
        "gcs.oauth2.refresh-credentials-endpoint",
        "gcs.oauth2.refresh-credentials-enabled",
        "adls.refresh-credentials-endpoint",
        "adls.refresh-credentials-enabled",
        "signer.uri",
        "signer.endpoint",
        "s3.endpoint",
        "s3.region",
        "client.region",
    ];
    let all: Vec<(&str, &str)> = kept.iter().chain(&dropped).map(|k| (*k, "v")).collect();
    let out = keep_credentials(map(&all));
    assert_eq!(
        out.keys().map(String::as_str).collect::<Vec<_>>().len(),
        kept.len()
    );
    for k in kept {
        assert!(out.contains_key(k), "{k}");
    }
}

/// An answer expires at its earliest credential deadline, each credential's own expiry key or else the fetch plus
/// the default TTL: an S3 key pair without an expiry beside a GCS token expiring later lives the default TTL.
#[test]
fn answer_expiry_counts_a_credential_without_an_expiry_key() {
    let fetched = UNIX_EPOCH + Duration::from_secs(1_000_000);
    let later = fetched + Duration::from_secs(7200);
    let later_ms = later
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string();
    let mixed = map(&[
        ("s3.access-key-id", "AKIA"),
        ("s3.secret-access-key", "secret"),
        ("gcs.oauth2.token", "tok"),
        ("gcs.oauth2.token-expires-at", &later_ms),
    ]);
    assert_eq!(answer_expires_at([&mixed], fetched, TTL), fetched + TTL);
    // A GCS token alone keeps its own, later expiry.
    let gcs = map(&[
        ("gcs.oauth2.token", "tok"),
        ("gcs.oauth2.token-expires-at", &later_ms),
    ]);
    assert_eq!(answer_expires_at([&gcs], fetched, TTL), later);
    // Each Azure host counts on its own: one without an expiry key lives the default TTL.
    let azure = map(&[
        ("adls.sas-token.a.dfs.core.windows.net", "sas"),
        (
            "adls.sas-token-expires-at-ms.a.dfs.core.windows.net",
            &later_ms,
        ),
        ("adls.sas-token.b.dfs.core.windows.net", "sas"),
    ]);
    assert_eq!(answer_expires_at([&azure], fetched, TTL), fetched + TTL);
}

#[test]
fn a_narrower_prefix_serves_a_wider_scope_only_with_one_credential() {
    let entry = |prefix: &str, key: &str| StorageCredential {
        prefix: prefix.to_owned(),
        config: map(&[("s3.access-key-id", key)]),
    };
    let config = map(&[("s3.access-key-id", "config")]);
    let scope = parse_object_url("s3://b/wh").unwrap();
    let key = |creds: &[StorageCredential], location: &str| {
        narrowed(creds, &config, location, &scope)
            .unwrap()
            .map(|c| c["s3.access-key-id"].expose_secret().to_owned())
    };
    // One prefix inside the scope, or several naming the same keys: that credential.
    assert_eq!(
        key(&[entry("s3://b/wh/t", "t")], "s3://b/wh/t").as_deref(),
        Some("t")
    );
    assert_eq!(
        key(
            &[
                entry("s3://b/wh/t/data", "t"),
                entry("s3://b/wh/t/metadata", "t")
            ],
            "s3://b/wh/t"
        )
        .as_deref(),
        Some("t")
    );
    // Two that differ: none.
    assert_eq!(
        key(
            &[
                entry("s3://b/wh/t/data", "d"),
                entry("s3://b/wh/t/metadata", "m")
            ],
            "s3://b/wh/t"
        ),
        None
    );
    // Outside the scope, another bucket, a segment that only shares a stem, the scope itself, a wider one: none.
    assert_eq!(key(&[entry("s3://b/other/t", "o")], "s3://b/wh/t"), None);
    assert_eq!(key(&[entry("s3://c/wh/t", "c")], "s3://b/wh/t"), None);
    assert_eq!(key(&[entry("s3://b/wh2/t", "w")], "s3://b/wh/t"), None);
    assert_eq!(key(&[entry("s3://b/wh", "same")], "s3://b/wh/t"), None);
    assert_eq!(
        key(
            &[entry("s3://b", "bucket"), entry("s3", "scheme")],
            "s3://b/wh/t"
        ),
        None
    );
    // A prefix outside the scope next to one inside does not count against it.
    assert_eq!(
        key(
            &[entry("s3://b/other", "o"), entry("s3://b/wh/t", "t")],
            "s3://b/wh/t"
        )
        .as_deref(),
        Some("t")
    );
    // No storage-credentials: `config` when the table's own location is inside the scope.
    assert_eq!(key(&[], "s3://b/wh/t").as_deref(), Some("config"));
    assert_eq!(key(&[], "s3://b/other/t"), None);
}
