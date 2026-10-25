// SPDX-License-Identifier: Apache-2.0
//! Cloud object stores: S3, GCS and Azure Blob Storage (P2-T084).
//!
//! [`StoreRegistry`] builds one [`ObjectStore`]
//! per bucket (S3, GCS) or container (Azure) from the session's Spark-style
//! Hadoop configuration, lazily on first use within a query. It is
//! session-scoped: it holds the query's [`Config`] snapshot, caches stores
//! only for that query, and is never a process-global cache keyed by
//! credentials.
//!
//! # URLs
//!
//! [`parse_object_url`] splits a user path into its scheme, bucket (or
//! container) and key: `s3://` and `s3a://`, `gs://`, `abfss://` and
//! `abfs://` (both `container@account.host/key` and `container/key`), and
//! `az://container/key`. Anything else with a scheme is
//! `KRISHIV_UNSUPPORTED_FEATURE`, as are paths that smuggle credentials in
//! the URL itself (user info): like the rest of the engine, this layer only
//! accepts credentials from configuration, never from paths (plans stay
//! pure data: a path string plus option values).
//!
//! # Configuration
//!
//! Unregistered `fs.*` keys of the session configuration (anything that is
//! not a `krishiv.*` key is kept verbatim by [`Config`]), with the
//! environment fallback of the `object_store` builders (`AWS_*`,
//! `GOOGLE_*`, `AZURE_*`):
//!
//! | Store | Key | Meaning |
//! |---|---|---|
//! | S3 | `fs.s3a.endpoint` | custom endpoint (MinIO, LocalStack); `http://` implies `allow_http` |
//! | S3 | `fs.s3a.access.key` | access key id |
//! | S3 | `fs.s3a.secret.key` | secret access key |
//! | S3 | `fs.s3a.session.token` | session token |
//! | S3 | `fs.s3a.path.style.access` | `true` forces path-style requests (the default for `http://` and `localhost` endpoints) |
//! | S3 | `fs.s3a.endpoint.region`, `fs.s3a.region` | region (the default is `us-east-1`) |
//! | GCS | `fs.gs.auth.service.account.json.keyfile` | service-account JSON file |
//! | GCS | `fs.gs.auth.service.account.json` | service-account JSON inline |
//! | GCS | `fs.gs.storage.root.url`, `fs.gs.endpoint` | custom endpoint (fake-gcs-server) |
//! | Azure | `fs.azure.account.key.<account>.dfs.core.windows.net` (also `.blob.`, also bare `.<account>`, also bare `fs.azure.account.key`) | account key |
//! | Azure | `fs.azure.sas.<container>.<account>.dfs.core.windows.net` (also `.blob.`, also `.<container>.<account>`, also bare `fs.azure.sas`) | shared access signature |
//! | Azure | `fs.azure.account.name` | account, when the URL carries none (`az://container/key`) |
//! | Azure | `fs.azure.endpoint` | custom endpoint (Azurite: `http://127.0.0.1:10000/<account>`); `http://` implies `allow_http` |
//!
//! Every value whose key is sensitive (see
//! [`is_sensitive_key`](krishiv_common::secret::is_sensitive_key): every key
//! above but the endpoints, the region and the account name) is wrapped in a
//! [`SecretString`] the moment it is read and is only exposed at the builder
//! call. [`StoreRegistry`]'s `Debug`, every error here and every store id
//! and path are free of secret values: builder errors name the bucket, the
//! endpoint or a credential file path at most.
//!
//! # Requests
//!
//! The registry hands out raw stores; the caller registers them on the
//! [`IoScheduler`], so every request still goes through
//! the layer's retry, timeout, pacing and accounting wrappers unchanged.
//! Cloud stores share [`StorePolicy::S3`]'s
//! per-prefix pacing until per-service rates are calibrated. Multipart
//! uploads rely on the cloud clients' own per-part retries (their default
//! `RetryConfig` is left enabled), as `upload.rs` documents.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::config::Config;
use krishiv_common::error::KrishivInvalidConfigValue;
use krishiv_common::secret::{OptionValue, REDACTED, SecretString, SecretValue};
use object_store::ObjectStore;
use object_store::path::Path;

use super::{IoScheduler, StoreId, StorePolicy};
use crate::error::{Result, store_failed, unsupported};

/// `fs.s3a.endpoint`: custom endpoint (MinIO, LocalStack).
pub const S3_ENDPOINT: &str = "fs.s3a.endpoint";
/// `fs.s3a.access.key`: the access key id.
pub const S3_ACCESS_KEY: &str = "fs.s3a.access.key";
/// `fs.s3a.secret.key`: the secret access key.
pub const S3_SECRET_KEY: &str = "fs.s3a.secret.key";
/// `fs.s3a.session.token`: the session token.
pub const S3_SESSION_TOKEN: &str = "fs.s3a.session.token";
/// `fs.s3a.path.style.access`: `true` forces path-style requests.
pub const S3_PATH_STYLE_ACCESS: &str = "fs.s3a.path.style.access";
/// `fs.s3a.endpoint.region`: the region.
pub const S3_ENDPOINT_REGION: &str = "fs.s3a.endpoint.region";
/// `fs.s3a.region`: the region (fallback).
pub const S3_REGION: &str = "fs.s3a.region";
/// `fs.gs.auth.service.account.json.keyfile`: service-account JSON file.
pub const GCS_KEYFILE: &str = "fs.gs.auth.service.account.json.keyfile";
/// `fs.gs.auth.service.account.json`: service-account JSON inline.
pub const GCS_KEY: &str = "fs.gs.auth.service.account.json";
/// `fs.gs.storage.root.url`: custom endpoint (fake-gcs-server).
pub const GCS_STORAGE_ROOT_URL: &str = "fs.gs.storage.root.url";
/// `fs.gs.endpoint`: custom endpoint (fallback).
pub const GCS_ENDPOINT: &str = "fs.gs.endpoint";
/// `fs.azure.account.name`: the account, when the URL carries none.
pub const AZURE_ACCOUNT_NAME: &str = "fs.azure.account.name";
/// `fs.azure.endpoint`: custom endpoint (Azurite).
pub const AZURE_ENDPOINT: &str = "fs.azure.endpoint";
/// `fs.azure.account.key`: the account key (fallback for the dotted forms).
pub const AZURE_ACCOUNT_KEY: &str = "fs.azure.account.key";
/// `fs.azure.sas`: the shared access signature (fallback for the dotted forms).
pub const AZURE_SAS: &str = "fs.azure.sas";

/// A cloud scheme [`parse_object_url`] recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectScheme {
    /// `s3://` and `s3a://`.
    S3,
    /// `gs://`.
    Gcs,
    /// `abfs://`, `abfss://` and `az://`.
    Azure,
}

impl fmt::Display for ObjectScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S3 => f.write_str("s3"),
            Self::Gcs => f.write_str("gs"),
            Self::Azure => f.write_str("abfss"),
        }
    }
}

/// A cloud URL split into its store and its key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedUrl {
    /// The scheme class.
    pub scheme: ObjectScheme,
    /// The scheme as written (`s3a` in `s3a://b/k`).
    pub scheme_text: String,
    /// The bucket (S3, GCS) or container (Azure).
    pub bucket: String,
    /// The Azure account, from a `container@account.host` URL.
    pub account: Option<String>,
    /// The key within the bucket.
    pub key: Path,
    /// The store id: `s3://bucket`, `gs://bucket`,
    /// `abfss://container` or `abfss://container@account`.
    pub store_id: StoreId,
    /// The URL in canonical form (`scheme://authority/key`); what errors
    /// print and what re-parses to the same store and key.
    pub display: String,
}

/// Splits `text` into its cloud store and key.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a scheme this layer does not speak, a
/// URL without `//`, a missing bucket, or credentials smuggled into the
/// URL itself.
pub fn parse_object_url(text: &str) -> Result<ParsedUrl> {
    let (scheme, after) = text
        .split_once(':')
        .ok_or_else(|| unsupported("object store paths without a scheme".to_owned()))?;
    let lower = scheme.to_ascii_lowercase();
    let kind = match lower.as_str() {
        "s3" | "s3a" => ObjectScheme::S3,
        "gs" => ObjectScheme::Gcs,
        "abfs" | "abfss" | "az" => ObjectScheme::Azure,
        _ => {
            return Err(unsupported(format!("paths with the scheme {scheme}")));
        }
    };
    let Some(rest) = after.strip_prefix("//") else {
        return Err(unsupported(format!("{scheme} paths without `//`")));
    };
    let (authority, key) = match rest.find('/') {
        Some(i) => rest.split_at(i),
        None => (rest, ""),
    };
    if authority.is_empty() {
        return Err(unsupported(format!("{scheme} paths without a bucket")));
    }
    let (bucket, account) = match kind {
        ObjectScheme::S3 | ObjectScheme::Gcs => {
            if authority.contains('@') {
                return Err(unsupported("paths with credentials".to_owned()));
            }
            (authority.to_owned(), None)
        }
        ObjectScheme::Azure => match authority.rsplit_once('@') {
            None => (authority.to_owned(), None),
            Some((container, host)) => {
                if container.is_empty() || container.contains(':') || host.is_empty() {
                    return Err(unsupported("paths with credentials".to_owned()));
                }
                // The account is the host's first label
                // (`account.dfs.core.windows.net`); a bare host carries no
                // account and one comes from `fs.azure.account.name`.
                let account = host
                    .contains('.')
                    .then(|| host.split('.').next().unwrap_or_default().to_owned())
                    .filter(|a| !a.is_empty());
                (container.to_owned(), account)
            }
        },
    };
    // Hadoop-style normalization: `.` is dropped, `..` pops, and empty
    // segments (doubled slashes) vanish; object_store paths reject `.` and
    // `..` segments outright.
    let mut parts: Vec<&str> = Vec::new();
    for segment in key.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            part => parts.push(part),
        }
    }
    let key = Path::parse(parts.join("/"))
        .map_err(|e| unsupported(format!("object store paths Spark cannot name ({e})")))?;
    let store_id = match (&kind, &account) {
        (ObjectScheme::S3, _) => StoreId::new(format!("s3://{bucket}")),
        (ObjectScheme::Gcs, _) => StoreId::new(format!("gs://{bucket}")),
        (ObjectScheme::Azure, Some(account)) => StoreId::new(format!("abfss://{bucket}@{account}")),
        (ObjectScheme::Azure, None) => StoreId::new(format!("abfss://{bucket}")),
    };
    let mut display = format!("{scheme}://{authority}");
    if !key.as_ref().is_empty() {
        display.push('/');
        display.push_str(key.as_ref());
    }
    Ok(ParsedUrl {
        scheme: kind,
        scheme_text: scheme.to_owned(),
        bucket,
        account,
        key,
        store_id,
        display,
    })
}

/// Builds cloud object stores from the session configuration, one per
/// bucket or container (see the module docs). Session-scoped: it holds the
/// query's [`Config`] snapshot and caches stores only for that query.
pub struct StoreRegistry {
    config: Arc<Config>,
    stores: Mutex<HashMap<StoreId, Arc<dyn ObjectStore>>>,
}

impl StoreRegistry {
    /// A registry building stores from `config` (the query's snapshot).
    #[must_use]
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            config,
            stores: Mutex::new(HashMap::new()),
        }
    }

    /// The store of `url`'s bucket or container, cached by store id within
    /// this query. Credentials come from the configuration (see the module
    /// docs) with the builders' environment fallback.
    ///
    /// # Errors
    /// Invalid configuration and storage errors (named bucket and endpoint
    /// at most, never a secret).
    pub fn store(&self, url: &ParsedUrl) -> Result<Arc<dyn ObjectStore>> {
        if let Some(store) = self
            .stores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&url.store_id)
            .cloned()
        {
            return Ok(store);
        }
        let store = build(&self.config, url)?;
        self.stores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(url.store_id.clone(), Arc::clone(&store));
        Ok(store)
    }

    /// Parses `text`, builds its store and registers it on `sched` under
    /// its store id, returning the id, the key and the display path.
    ///
    /// # Errors
    /// As [`parse_object_url`] and [`Self::store`].
    pub fn ensure(&self, sched: &IoScheduler, text: &str) -> Result<(StoreId, Path, String)> {
        let url = parse_object_url(text)?;
        let store = self.store(&url)?;
        sched.register_store_with(url.store_id.clone(), store, StorePolicy::S3);
        Ok((url.store_id, url.key, url.display))
    }
}

impl fmt::Debug for StoreRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never the configuration (it may hold secrets) or the stores
        // (their builders' Debug may echo endpoints): only the count.
        let cached = self.stores.lock().map(|c| c.len()).unwrap_or_default();
        f.debug_struct("StoreRegistry")
            .field("cached", &cached)
            .finish_non_exhaustive()
    }
}

/// The store of `url` built from `config` (see the module docs for the
/// keys).
fn build(config: &Config, url: &ParsedUrl) -> Result<Arc<dyn ObjectStore>> {
    // Configuration errors keep their class; only storage errors become
    // `KRISHIV_OBJECT_STORE_REQUEST_FAILED`.
    let path_style = match url.scheme {
        ObjectScheme::S3 => path_style_access(config)?,
        _ => None,
    };
    match url.scheme {
        ObjectScheme::S3 => s3_store(config, url, path_style),
        ObjectScheme::Gcs => gcs_store(config, url),
        ObjectScheme::Azure => azure_store(config, url),
    }
    .map_err(|e: object_store::Error| store_failed(&url.display, 1, &e.to_string()))
}

/// The path-style setting for an S3 store: explicit
/// `fs.s3a.path.style.access`, else path-style with a custom endpoint that
/// looks local (MinIO), else the builder default.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for anything but `true`/`false`.
fn path_style_access(config: &Config) -> Result<Option<bool>> {
    let explicit = flag(config, S3_PATH_STYLE_ACCESS)?;
    if explicit.is_some() {
        return Ok(explicit);
    }
    Ok(text(config, S3_ENDPOINT).map(|e| allows_http(Some(&e)) || e.contains("localhost")))
}

/// The secret (or plain) value of an unregistered `key`, typed so it never
/// prints; empty values count as unset. This is the only place the
/// registry's secrets cross from configuration into builders.
fn cred(config: &Config, key: &str) -> Option<SecretString> {
    let raw = config.get_unregistered(key).filter(|v| !v.is_empty())?;
    match OptionValue::from_user(key, raw) {
        OptionValue::Plain(plain) => Some(SecretString::new(plain)),
        OptionValue::Secret(SecretValue::Inline(secret)) => Some(secret),
        // Unreachable: `from_user` never builds a reference, and values of
        // sensitive keys never classify as URLs. Ignoring them is the
        // fail-safe direction: no value is used that was not typed as the
        // key demands.
        OptionValue::Secret(SecretValue::Reference { .. }) | OptionValue::Url(_) => None,
    }
}

/// The plain value of an unregistered `key`, trimmed; empty values count as
/// unset. Only for keys that are never sensitive (endpoints, regions).
fn text(config: &Config, key: &str) -> Option<String> {
    config
        .get_unregistered(key)
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
}

/// Whether `text` allows HTTP connections: only an `http://` endpoint does
/// (emulators); anything else keeps the builders' HTTPS default.
fn allows_http(text: Option<&str>) -> bool {
    text.is_some_and(|e| e.starts_with("http://"))
}

/// Parses a `true`/`false` flag of `key`.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for anything else. The error names the
/// key and the redaction marker, never the value.
fn flag(config: &Config, key: &str) -> Result<Option<bool>> {
    match config
        .get_unregistered(key)
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        None => Ok(None),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "true" => Ok(Some(true)),
            "false" => Ok(Some(false)),
            _ => Err(KrishivInvalidConfigValue {
                key: format!("\"{key}\""),
                value: format!("'{REDACTED}'"),
                reason: "expected `true` or `false`".to_owned(),
            }
            .into()),
        },
    }
}

/// An S3 store for `url`'s bucket.
fn s3_store(
    config: &Config,
    url: &ParsedUrl,
    path_style: Option<bool>,
) -> object_store::Result<Arc<dyn ObjectStore>> {
    use object_store::aws::AmazonS3Builder;
    let mut builder = AmazonS3Builder::from_env().with_bucket_name(&url.bucket);
    if let Some(endpoint) = text(config, S3_ENDPOINT) {
        if allows_http(Some(&endpoint)) {
            builder = builder.with_allow_http(true);
        }
        builder = builder.with_endpoint(endpoint);
    }
    if let Some(key) = cred(config, S3_ACCESS_KEY) {
        builder = builder.with_access_key_id(key.expose_secret());
    }
    if let Some(secret) = cred(config, S3_SECRET_KEY) {
        builder = builder.with_secret_access_key(secret.expose_secret());
    }
    if let Some(token) = cred(config, S3_SESSION_TOKEN) {
        builder = builder.with_token(token.expose_secret());
    }
    if let Some(region) = text(config, S3_ENDPOINT_REGION).or_else(|| text(config, S3_REGION)) {
        builder = builder.with_region(region);
    }
    if let Some(path_style) = path_style {
        builder = builder.with_virtual_hosted_style_request(!path_style);
    }
    builder.build().map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
}

/// A GCS store for `url`'s bucket.
fn gcs_store(config: &Config, url: &ParsedUrl) -> object_store::Result<Arc<dyn ObjectStore>> {
    use object_store::gcp::GoogleCloudStorageBuilder;
    let mut builder = GoogleCloudStorageBuilder::from_env().with_bucket_name(url.bucket.clone());
    if let Some(path) = cred(config, GCS_KEYFILE) {
        builder = builder.with_service_account_path(path.expose_secret());
    }
    if let Some(key) = cred(config, GCS_KEY) {
        builder = builder.with_service_account_key(key.expose_secret());
    }
    if let Some(base) = text(config, GCS_STORAGE_ROOT_URL).or_else(|| text(config, GCS_ENDPOINT)) {
        builder = builder.with_base_url(&base);
    }
    builder.build().map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
}

/// An Azure Blob store for `url`'s container.
fn azure_store(config: &Config, url: &ParsedUrl) -> object_store::Result<Arc<dyn ObjectStore>> {
    use object_store::azure::MicrosoftAzureBuilder;
    let mut builder = MicrosoftAzureBuilder::from_env().with_container_name(url.bucket.clone());
    let account = url
        .account
        .clone()
        .or_else(|| text(config, AZURE_ACCOUNT_NAME));
    if let Some(account) = &account {
        builder = builder.with_account(account);
    }
    if let Some(key) = azure_key(config, url, account.as_deref()) {
        builder = builder.with_access_key(key.expose_secret());
    }
    if let Some(sas) = azure_sas(config, url, account.as_deref()) {
        let pairs = sas_pairs(&sas).map_err(|reason| object_store::Error::Generic {
            store: "AZURE",
            source: reason.into(),
        })?;
        builder = builder.with_sas_authorization(pairs);
    }
    if let Some(endpoint) = text(config, AZURE_ENDPOINT) {
        if allows_http(Some(&endpoint)) {
            builder = builder.with_allow_http(true);
        }
        builder = builder.with_endpoint(endpoint);
    }
    builder.build().map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
}

/// The Azure account key: the dotted `fs.azure.account.key.<account>.*`
/// forms first, then the bare fallback.
fn azure_key(config: &Config, url: &ParsedUrl, account: Option<&str>) -> Option<SecretString> {
    dotted(config, "fs.azure.account.key", url, account).or_else(|| cred(config, AZURE_ACCOUNT_KEY))
}

/// The Azure shared access signature: the dotted `fs.azure.sas.*` forms
/// first, then the bare fallback.
fn azure_sas(config: &Config, url: &ParsedUrl, account: Option<&str>) -> Option<SecretString> {
    dotted(config, "fs.azure.sas", url, account).or_else(|| cred(config, AZURE_SAS))
}

/// The value of the most specific dotted key for `prefix`:
/// `<prefix>.<container>.<account>.dfs.core.windows.net`,
/// `<prefix>.<container>.<account>.blob.core.windows.net`,
/// `<prefix>.<container>.<account>`, then `<prefix>.<account>` when the URL
/// carries an account.
fn dotted(
    config: &Config,
    prefix: &str,
    url: &ParsedUrl,
    account: Option<&str>,
) -> Option<SecretString> {
    let account = account?;
    for key in [
        format!("{prefix}.{}.{account}.dfs.core.windows.net", url.bucket),
        format!("{prefix}.{}.{account}.blob.core.windows.net", url.bucket),
        format!("{prefix}.{}.{account}", url.bucket),
        format!("{prefix}.{account}"),
    ] {
        if let Some(value) = cred(config, &key) {
            return Some(value);
        }
    }
    None
}

/// Splits a SAS token (`sv=..&sig=..`) into its query pairs, redacting any
/// complaint: the token itself must never reach an error.
fn sas_pairs(secret: &SecretString) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    for part in secret.expose_secret().split('&') {
        match part.split_once('=') {
            Some((name, value)) if !name.is_empty() => {
                pairs.push((name.to_owned(), value.to_owned()));
            }
            _ => return Err("the Azure SAS token is not `name=value&...`".to_owned()),
        }
    }
    if pairs.is_empty() {
        return Err("the Azure SAS token is not `name=value&...`".to_owned());
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::ErrorClass;

    const CANARY: &str = "Canary-7f3a9c";

    fn config(pairs: &[(&str, &str)]) -> Arc<Config> {
        let (c, _) = Config::from_pairs(pairs.to_vec()).unwrap();
        Arc::new(c)
    }

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
            let secret = cred(&cfg, key).unwrap();
            assert_eq!(secret.expose_secret(), CANARY);
            assert!(!format!("{secret:?}").contains(CANARY), "{key}");
        }
    }

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

    #[test]
    fn failures_name_keys_and_buckets_but_never_secrets() {
        // A store that cannot be built names what is missing, never a
        // secret: no account here, however the key is set.
        let registry = StoreRegistry::new(config(&[(AZURE_ACCOUNT_KEY, CANARY)]));
        let url = parse_object_url("abfss://cont/k").unwrap();
        let error = registry.store(&url).unwrap_err();
        assert_eq!(
            error.error_class(),
            ErrorClass::KrishivObjectStoreRequestFailed
        );
        assert!(!error.to_string().contains(CANARY));
        assert!(!format!("{error:?}").contains(CANARY));
        assert!(!format!("{registry:?}").contains(CANARY));

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

        // Half a key pair is refused without echo.
        let registry = StoreRegistry::new(config(&[(S3_SECRET_KEY, CANARY)]));
        let error = registry
            .store(&parse_object_url("s3://bucket/k").unwrap())
            .unwrap_err();
        assert!(!error.to_string().contains(CANARY));
    }

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
}
