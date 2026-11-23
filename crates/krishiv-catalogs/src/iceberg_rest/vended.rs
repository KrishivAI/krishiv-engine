// SPDX-License-Identifier: Apache-2.0
//! Vended storage credentials of an Iceberg REST `loadTable` / `loadCredentials` answer (design P3-T067 §4.1):
//! which credential covers a data file, and the credential itself as a [`ScopedCredential`].
//!
//! - **Prefixes.** A `storage-credentials` prefix is a full URL (`s3://bucket/a/b`, matched on path segments, so
//!   `s3://bucket/a/bc` is not under it), a bucket (`s3://bucket`, every key of it) or a bare scheme (`s3`, `gs`,
//!   `abfss`: every location of that scheme family, as the store layer maps schemes: `s3` covers `s3a://`, `abfss`
//!   covers `abfs://` and `az://`). The longest matching prefix wins (the spec's rule); with no
//!   `storage-credentials` entry the answer's `config` map is the credential, for the table's own location only.
//! - **Keys** (Iceberg 1.12.0 `S3FileIOProperties`, `GCPProperties`, `AzureProperties`): S3 `s3.access-key-id`,
//!   `s3.secret-access-key`, `s3.session-token`, `s3.session-token-expires-at-ms`; GCS `gcs.oauth2.token`,
//!   `gcs.oauth2.token-expires-at` (epoch milliseconds, `new Date(long)`); ADLS `adls.sas-token.<host>` and
//!   `adls.sas-token-expires-at-ms.<host>`, keyed by the account host exactly as Java's
//!   `AzureProperties.applyClientConfiguration(account, ..)` looks it up (`https://` + account is its endpoint).
//! - **Expiry.** A credential without its expiry key is valid for the default TTL (`krishiv.catalog.vendedDefaultTtl`)
//!   and then fetched again; an expiry that does not parse is a catalog fault, never "no expiry".
//! - **Faults.** A key id without its secret (or the reverse), or an expiry that does not parse, is a catalog fault
//!   whose detail names keys only, never a value.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use krishiv_common::secret::SecretString;
use krishiv_store::{ObjectScheme, ParsedUrl, ScopedCredential, parse_object_url};

use super::types::StorageCredential;

const S3_KEY_ID: &str = "s3.access-key-id";
const S3_SECRET: &str = "s3.secret-access-key";
const S3_TOKEN: &str = "s3.session-token";
const S3_EXPIRES: &str = "s3.session-token-expires-at-ms";
const GCS_TOKEN: &str = "gcs.oauth2.token";
const GCS_EXPIRES: &str = "gcs.oauth2.token-expires-at";
const ADLS_SAS: &str = "adls.sas-token.";
const ADLS_EXPIRES: &str = "adls.sas-token-expires-at-ms.";

/// Why a vended answer cannot be read with: a catalog fault, its detail for the log only (keys, never values).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VendedFault(pub(crate) String);

/// One vended prefix in its parsed form (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VendedPrefix {
    /// Every location of a scheme family.
    Scheme(ObjectScheme),
    /// Every key of one bucket (store id `s3://bucket`, `abfss://container@host`).
    Bucket { scheme: ObjectScheme, store: String },
    /// Keys under a path, on whole segments.
    Path {
        scheme: ObjectScheme,
        store: String,
        segments: Vec<String>,
    },
}

/// The scheme family of a bare scheme prefix (`s3`, `s3a`, `gs`, `abfs`, `abfss`, `az`), ASCII case-insensitive.
fn family(scheme: &str) -> Option<ObjectScheme> {
    match scheme.to_ascii_lowercase().as_str() {
        "s3" | "s3a" => Some(ObjectScheme::S3),
        "gs" => Some(ObjectScheme::Gcs),
        "abfs" | "abfss" | "az" => Some(ObjectScheme::Azure),
        _ => None,
    }
}

fn segments(url: &ParsedUrl) -> Vec<String> {
    url.key
        .as_ref()
        .split('/')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

impl VendedPrefix {
    /// Parses a `storage-credentials` prefix.
    ///
    /// # Errors
    /// A fault for a scheme without credentials of its own (`https`, `hf`) or a prefix that is no URL.
    pub(crate) fn parse(prefix: &str) -> Result<Self, VendedFault> {
        if !prefix.contains(':') {
            return family(prefix).map(Self::Scheme).ok_or_else(|| {
                VendedFault(
                    "a storage-credential prefix names a scheme without vended credentials"
                        .to_owned(),
                )
            });
        }
        let url = parse_object_url(prefix).map_err(|_| {
            VendedFault("a storage-credential prefix is not an object-store location".to_owned())
        })?;
        if !matches!(
            url.scheme,
            ObjectScheme::S3 | ObjectScheme::Gcs | ObjectScheme::Azure
        ) {
            return Err(VendedFault(
                "a storage-credential prefix names a scheme without vended credentials".to_owned(),
            ));
        }
        let store = url.store_id.key().to_owned();
        let segments = segments(&url);
        Ok(if segments.is_empty() {
            Self::Bucket {
                scheme: url.scheme,
                store,
            }
        } else {
            Self::Path {
                scheme: url.scheme,
                store,
                segments,
            }
        })
    }

    /// Whether `url` is under this prefix.
    pub(crate) fn covers(&self, url: &ParsedUrl) -> bool {
        match self {
            Self::Scheme(s) => *s == url.scheme,
            Self::Bucket { scheme, store } => *scheme == url.scheme && store == url.store_id.key(),
            Self::Path {
                scheme,
                store,
                segments: want,
            } => {
                *scheme == url.scheme
                    && store == url.store_id.key()
                    && segments(url).starts_with(want)
            }
        }
    }

    /// How specific the prefix is: the longest match wins.
    pub(crate) fn depth(&self) -> usize {
        match self {
            Self::Scheme(_) => 0,
            Self::Bucket { .. } => 1,
            Self::Path { segments, .. } => 1 + segments.len(),
        }
    }
}

/// A prefix and the credential keys it names.
pub(crate) type Covering<'a> = (VendedPrefix, &'a BTreeMap<String, SecretString>);

/// The credential entry covering `url`: the longest matching `storage-credentials` prefix, else (with no entries at
/// all) `config` when `url` is under the table's own `location`. `Ok(None)`: nothing covers `url`.
///
/// # Errors
/// A fault for a prefix that does not parse (the client checked every prefix already, so this is a catalog
/// answering in a form it should not).
pub(crate) fn covering<'a>(
    credentials: &'a [StorageCredential],
    config: &'a BTreeMap<String, SecretString>,
    location: &str,
    url: &ParsedUrl,
) -> Result<Option<Covering<'a>>, VendedFault> {
    let mut best: Option<Covering<'a>> = None;
    for c in credentials {
        let prefix = VendedPrefix::parse(&c.prefix)?;
        if prefix.covers(url)
            && best
                .as_ref()
                .is_none_or(|(b, _)| prefix.depth() > b.depth())
        {
            best = Some((prefix, &c.config));
        }
    }
    if best.is_some() || !credentials.is_empty() {
        return Ok(best);
    }
    let own = VendedPrefix::parse(location)?;
    Ok(own.covers(url).then_some((own, config)))
}

fn expiry(
    config: &BTreeMap<String, SecretString>,
    key: &str,
    now: SystemTime,
    default_ttl: Duration,
) -> Result<SystemTime, VendedFault> {
    match config.get(key) {
        None => Ok(now + default_ttl),
        Some(v) => v
            .expose_secret()
            .trim()
            .parse::<u64>()
            .ok()
            .and_then(|ms| UNIX_EPOCH.checked_add(Duration::from_millis(ms)))
            .ok_or_else(|| {
                VendedFault(format!(
                    "the vended credential's {key} is not epoch milliseconds"
                ))
            }),
    }
}

/// The credential in `config` for `url`'s scheme (see the module docs). `Ok(None)`: `config` holds none for it.
///
/// # Errors
/// A fault for half a key pair or an expiry that does not parse.
pub(crate) fn credential(
    config: &BTreeMap<String, SecretString>,
    url: &ParsedUrl,
    now: SystemTime,
    default_ttl: Duration,
) -> Result<Option<ScopedCredential>, VendedFault> {
    match url.scheme {
        ObjectScheme::S3 => match (config.get(S3_KEY_ID), config.get(S3_SECRET)) {
            (None, None) => Ok(None),
            (Some(id), Some(secret)) => Ok(Some(ScopedCredential::s3(
                id.expose_secret().to_owned(),
                secret.expose_secret().to_owned(),
                config.get(S3_TOKEN).map(|t| t.expose_secret().to_owned()),
                Some(expiry(config, S3_EXPIRES, now, default_ttl)?),
            ))),
            _ => Err(VendedFault(format!(
                "a vended S3 credential needs both {S3_KEY_ID} and {S3_SECRET}"
            ))),
        },
        ObjectScheme::Gcs => config
            .get(GCS_TOKEN)
            .map(|t| {
                Ok(ScopedCredential::gcs(
                    t.expose_secret().to_owned(),
                    Some(expiry(config, GCS_EXPIRES, now, default_ttl)?),
                ))
            })
            .transpose(),
        ObjectScheme::Azure => {
            let Some(host) = url.host.as_deref() else {
                return Ok(None);
            };
            config
                .get(&format!("{ADLS_SAS}{host}"))
                .map(|t| {
                    Ok(ScopedCredential::azure_sas(
                        t.expose_secret().to_owned(),
                        Some(expiry(
                            config,
                            &format!("{ADLS_EXPIRES}{host}"),
                            now,
                            default_ttl,
                        )?),
                    ))
                })
                .transpose()
        }
        ObjectScheme::Http | ObjectScheme::HuggingFace => Ok(None),
    }
}

/// The catalog-supplied bearer token of one table (a `loadTable` answer's `config`), design §4.1.
const TABLE_TOKEN: &str = "token";

/// The table `token` of a `loadTable` answer's `config`, when it has one: used only as the bearer of that table's
/// requests to the declared catalog host, never sent to the token endpoint or to storage.
pub(crate) fn table_token(config: &BTreeMap<String, SecretString>) -> Option<SecretString> {
    config
        .get(TABLE_TOKEN)
        .filter(|t| !t.expose_secret().trim().is_empty())
        .cloned()
}

/// When the credential keys of `config` stop being good: the earliest deadline of the credentials it holds, each its
/// own expiry key, else `fetched` plus the default TTL (an S3 key pair without `s3.session-token-expires-at-ms`
/// beside a GCS token expiring later lives the default TTL). An expiry that does not parse counts as the default TTL
/// here (the read refuses it anyway). A map with no credential at all is `fetched` plus the default TTL.
fn expires_at(
    config: &BTreeMap<String, SecretString>,
    fetched: SystemTime,
    default_ttl: Duration,
) -> SystemTime {
    let deadline =
        |key: &str| expiry(config, key, fetched, default_ttl).unwrap_or(fetched + default_ttl);
    let mut deadlines = Vec::new();
    if config.contains_key(S3_KEY_ID) || config.contains_key(S3_SECRET) {
        deadlines.push(deadline(S3_EXPIRES));
    }
    if config.contains_key(GCS_TOKEN) {
        deadlines.push(deadline(GCS_EXPIRES));
    }
    for host in config.keys().filter_map(|k| k.strip_prefix(ADLS_SAS)) {
        deadlines.push(deadline(&format!("{ADLS_EXPIRES}{host}")));
    }
    deadlines.into_iter().min().unwrap_or(fetched + default_ttl)
}

/// When a vended answer stops being good (design §4.3: its cache entry is dropped then): the earliest expiry of any
/// map that holds a credential, else `fetched` plus the default TTL.
pub(crate) fn answer_expires_at<'a>(
    maps: impl IntoIterator<Item = &'a BTreeMap<String, SecretString>>,
    fetched: SystemTime,
    default_ttl: Duration,
) -> SystemTime {
    maps.into_iter()
        .filter(|m| !m.is_empty())
        .map(|m| expires_at(m, fetched, default_ttl))
        .min()
        .unwrap_or(fetched + default_ttl)
}

/// Whether `key` is one of the credential keys above (what a vended answer keeps of a `config` map).
fn is_credential_key(key: &str) -> bool {
    matches!(
        key,
        S3_KEY_ID | S3_SECRET | S3_TOKEN | S3_EXPIRES | GCS_TOKEN | GCS_EXPIRES
    ) || key.starts_with(ADLS_SAS)
        || key.starts_with(ADLS_EXPIRES)
}

/// The credential keys of `config`; every other key (refresh endpoints, `signer.*`, store options, and the table
/// `token`, which [`table_token`] takes first) is dropped here and never followed or sent anywhere.
pub(crate) fn keep_credentials(
    config: BTreeMap<String, SecretString>,
) -> BTreeMap<String, SecretString> {
    config
        .into_iter()
        .filter(|(k, _)| is_credential_key(k))
        .collect()
}

#[cfg(test)]
#[path = "vended_tests.rs"]
mod tests;
