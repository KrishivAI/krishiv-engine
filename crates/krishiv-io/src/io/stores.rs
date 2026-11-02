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
//! `GOOGLE_*`, `AZURE_*`). Per-read data-source options (`spark.read.option`
//! / `df.write.option`, carried by the plan as option values) are honoured
//! exactly like session config: Spark merges data-source options into the
//! per-relation Hadoop configuration, so a per-read `fs.*` option overrides
//! the session value. Option keys match ASCII case-insensitively on both
//! sides (Spark's `CaseInsensitiveStringMap`); a per-read option is always
//! a present value, so an empty per-read value is invalid (it fails, never
//! silently falls back), as is a per-read value of the wrong kind for its
//! key (a secret, reference or URL where a plain value is required, or the
//! reverse): both fail with `KRISHIV_INVALID_CONFIG_VALUE`, naming the key
//! and the redaction marker but never the value. A read whose per-read
//! `fs.*` options are non-empty resolves to a store id with a non-secret
//! `#hash` suffix, so two reads of one bucket with different credentials
//! never share a store within the query.
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
//! [`SecretString`](krishiv_common::secret::SecretString) the moment it is read and is only exposed at the builder
//! call. [`StoreRegistry`]'s `Debug`, every error here and every store id
//! and path are free of secret values: builder errors name the bucket, the
//! endpoint or a credential file path at most.
//!
//! # Requests
//!
//! The registry hands out raw stores; the caller registers them on the
//! [`IoScheduler`], so every request still goes through
//! the layer's retry, timeout, pacing and accounting wrappers unchanged.
//! Each cloud scheme paces with its own [`StorePolicy`] (`S3`, `GCS`,
//! `AZURE`); the GCS and Azure budgets are conservative starting points
//! until per-service rates are calibrated. Multipart
//! uploads rely on the cloud clients' own per-part retries (their default
//! `RetryConfig` is left enabled), as `upload.rs` documents.
//!
//! Cloud builders live behind the crate's `cloud` feature (the `aws`,
//! `azure` and `gcp` features of `object_store`): without it, building a
//! cloud store fails with `KRISHIV_UNSUPPORTED_FEATURE`, while parsing URLs
//! and pacing policies keep working so plans still qualify paths.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::config::Config;
#[cfg(feature = "cloud")]
use krishiv_common::error::KrishivInvalidConfigValue;
use krishiv_common::secret::{OptionValue, SecretValue};
#[cfg(feature = "cloud")]
use krishiv_common::secret::{REDACTED, SecretString};
use object_store::ObjectStore;
use object_store::path::Path;

use super::{IoScheduler, StoreId, StorePolicy};
#[cfg(feature = "cloud")]
use crate::error::store_failed;
use crate::error::{Result, unsupported};

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

/// `fs.hf.token`: the Hugging Face access token (falls back to the
/// `HF_TOKEN` environment variable). A secret; never in a plan.
pub const HF_TOKEN: &str = "fs.hf.token";
/// `fs.hf.endpoint`: the Hugging Face host (default `https://huggingface.co`).
pub const HF_ENDPOINT: &str = "fs.hf.endpoint";

/// A cloud scheme [`parse_object_url`] recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectScheme {
    /// `s3://` and `s3a://`.
    S3,
    /// `gs://`.
    Gcs,
    /// `abfs://`, `abfss://` and `az://`.
    Azure,
    /// `https://` (and loopback `http://`): a read-only server of byte
    /// ranges. The "bucket" is the authority (`host[:port]`).
    Http,
    /// `hf://datasets/<org>/<name>[@revision]/<path>`: a Krishiv extension
    /// over the Hugging Face `resolve` endpoint (read-only, ranges).
    HuggingFace,
}

impl fmt::Display for ObjectScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S3 => f.write_str("s3"),
            Self::Gcs => f.write_str("gs"),
            Self::Azure => f.write_str("abfss"),
            Self::Http => f.write_str("https"),
            Self::HuggingFace => f.write_str("hf"),
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
    /// The revision of an `hf://` dataset (`main` unless `@rev` is written).
    pub revision: Option<String>,
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
        "https" | "http" => ObjectScheme::Http,
        "hf" => ObjectScheme::HuggingFace,
        _ => {
            return Err(unsupported(format!("paths with the scheme {scheme}")));
        }
    };
    let Some(rest) = after.strip_prefix("//") else {
        return Err(unsupported(format!("{scheme} paths without `//`")));
    };
    if kind == ObjectScheme::Http && rest.contains(['?', '#']) {
        return Err(unsupported(format!(
            "{scheme} paths with a query or fragment"
        )));
    }
    if kind == ObjectScheme::HuggingFace {
        return parse_hf(scheme, rest);
    }
    let (authority, key) = match rest.find('/') {
        Some(i) => rest.split_at(i),
        None => (rest, ""),
    };
    if authority.is_empty() {
        return Err(unsupported(format!("{scheme} paths without a bucket")));
    }
    let (bucket, account) = match kind {
        ObjectScheme::S3 | ObjectScheme::Gcs | ObjectScheme::Http | ObjectScheme::HuggingFace => {
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
    let key = normalize_key(key)?;
    let store_id = match (&kind, &account) {
        (ObjectScheme::S3, _) => StoreId::new(format!("s3://{bucket}")),
        (ObjectScheme::Gcs, _) => StoreId::new(format!("gs://{bucket}")),
        (ObjectScheme::Azure, Some(account)) => StoreId::new(format!("abfss://{bucket}@{account}")),
        (ObjectScheme::Azure, None) => StoreId::new(format!("abfss://{bucket}")),
        (ObjectScheme::Http | ObjectScheme::HuggingFace, _) => {
            StoreId::new(format!("{lower}://{bucket}"))
        }
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
        revision: None,
        key,
        store_id,
        display,
    })
}

/// Hadoop-style normalization of a key: `.` is dropped, `..` pops, and empty
/// segments (doubled slashes) vanish; object_store paths reject `.` and `..`
/// segments outright.
fn normalize_key(key: &str) -> Result<Path> {
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
    Path::parse(parts.join("/"))
        .map_err(|e| unsupported(format!("object store paths Spark cannot name ({e})")))
}

/// `hf://datasets/<org>/<name>[@revision]/<path>` (the part after `hf://`).
/// The store is the dataset at one revision; the key is the path in it.
fn parse_hf(scheme: &str, rest: &str) -> Result<ParsedUrl> {
    let mut it = rest.splitn(4, '/');
    let (kind, org, name_rev, key) = (
        it.next().unwrap_or_default(),
        it.next().unwrap_or_default(),
        it.next().unwrap_or_default(),
        it.next().unwrap_or_default(),
    );
    if kind != "datasets" {
        return Err(unsupported(format!(
            "{scheme} paths other than hf://datasets/<org>/<name>/<path>"
        )));
    }
    let (name, revision) = match name_rev.split_once('@') {
        Some((name, rev)) => (name, rev),
        None => (name_rev, "main"),
    };
    let plain = |t: &str| {
        !t.is_empty()
            && t.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    if !plain(org) || !plain(name) || !plain(revision) {
        return Err(unsupported(format!(
            "{scheme} paths other than hf://datasets/<org>/<name>[@revision]/<path>"
        )));
    }
    let key = normalize_key(key)?;
    let authority = format!("datasets/{org}/{name}@{revision}");
    let mut display = format!("hf://{authority}");
    if !key.as_ref().is_empty() {
        display.push('/');
        display.push_str(key.as_ref());
    }
    Ok(ParsedUrl {
        scheme: ObjectScheme::HuggingFace,
        scheme_text: scheme.to_owned(),
        bucket: format!("{org}/{name}"),
        account: None,
        revision: Some(revision.to_owned()),
        key,
        store_id: StoreId::new(format!("hf://{authority}")),
        display,
    })
}

/// Builds cloud object stores from the session configuration, one per
/// bucket or container (see the module docs), with per-read `fs.*` option
/// overrides. Session-scoped: it holds the query's [`Config`] snapshot and
/// caches stores only for that query.
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
        self.store_with_options(url, &BTreeMap::new())
    }

    /// As [`Self::store`], with per-read `fs.*` data-source options
    /// overriding the session configuration (Spark's per-relation Hadoop
    /// configuration). `options` is the plan's option map: only `fs.*` keys
    /// affect the store; every other option is ignored here.
    ///
    /// # Errors
    /// As [`Self::store`]. An invalid per-read value fails the same way an
    /// invalid session value does (naming the key and the redaction marker,
    /// never the value).
    pub fn store_with_options(
        &self,
        url: &ParsedUrl,
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<Arc<dyn ObjectStore>> {
        let id = effective_store_id(&url.store_id, options);
        if let Some(store) = self
            .stores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&id)
            .cloned()
        {
            return Ok(store);
        }
        let store = build(&self.config, options, url)?;
        self.stores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, Arc::clone(&store));
        Ok(store)
    }

    /// Parses `text`, builds its store and registers it on `sched` under
    /// its store id, returning the id, the key and the display path.
    ///
    /// # Errors
    /// As [`parse_object_url`] and [`Self::store`].
    pub fn ensure(&self, sched: &IoScheduler, text: &str) -> Result<(StoreId, Path, String)> {
        self.ensure_with_options(sched, text, &BTreeMap::new())
    }

    /// As [`Self::ensure`], with per-read `fs.*` data-source options
    /// overriding the session configuration (see
    /// [`Self::store_with_options`]).
    ///
    /// # Errors
    /// As [`parse_object_url`] and [`Self::store_with_options`].
    pub fn ensure_with_options(
        &self,
        sched: &IoScheduler,
        text: &str,
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<(StoreId, Path, String)> {
        let url = parse_object_url(text)?;
        let store = self.store_with_options(&url, options)?;
        let id = effective_store_id(&url.store_id, options);
        sched.register_store_with(id.clone(), store, policy_for(url.scheme));
        Ok((id, url.key, url.display))
    }
}

/// The pacing policy of one cloud scheme: each scheme paces with its own
/// budget (see [`StorePolicy`]), so a change to one service's rates never
/// retunes another's.
fn policy_for(scheme: ObjectScheme) -> StorePolicy {
    match scheme {
        ObjectScheme::S3 => StorePolicy::S3,
        ObjectScheme::Gcs => StorePolicy::GCS,
        ObjectScheme::Azure => StorePolicy::AZURE,
        // A plain web server publishes no per-prefix budget.
        ObjectScheme::Http | ObjectScheme::HuggingFace => StorePolicy::UNPACED,
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

/// The store of `url` built from `config` with per-read `options`
/// overriding it (see the module docs for the keys).
#[cfg(feature = "cloud")]
fn build(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
) -> Result<Arc<dyn ObjectStore>> {
    // Configuration errors keep their class; only builder failures become
    // `KRISHIV_OBJECT_STORE_REQUEST_FAILED` (converted in each scheme
    // builder below).
    let path_style = match url.scheme {
        ObjectScheme::S3 => path_style_access(config, options)?,
        _ => None,
    };
    match url.scheme {
        ObjectScheme::S3 => s3_store(config, options, url, path_style),
        ObjectScheme::Gcs => gcs_store(config, options, url),
        ObjectScheme::Azure => azure_store(config, options, url),
        ObjectScheme::Http => http_store(url),
        ObjectScheme::HuggingFace => hf_store(config, options, url),
    }
}

/// A Hugging Face dataset at one revision, read through the `resolve`
/// endpoint with byte ranges. The token (`fs.hf.token`, else `HF_TOKEN`) is
/// sent as a sensitive bearer header; reqwest drops it on a redirect to
/// another host (the CDN), and it is never in a plan, error or log.
#[cfg(feature = "cloud")]
fn hf_store(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
) -> Result<Arc<dyn ObjectStore>> {
    use object_store::http::HttpBuilder;
    let endpoint =
        text(config, options, HF_ENDPOINT)?.unwrap_or_else(|| "https://huggingface.co".to_owned());
    let endpoint = endpoint.trim_end_matches('/');
    let base = format!(
        "{endpoint}/datasets/{}/resolve/{}",
        url.bucket,
        url.revision.as_deref().unwrap_or("main")
    );
    let token = match cred(config, options, HF_TOKEN)? {
        Some(token) => Some(token),
        None => std::env::var("HF_TOKEN")
            .ok()
            .filter(|t| !t.is_empty())
            .map(SecretString::new),
    };
    let mut client =
        object_store::ClientOptions::new().with_allow_http(allows_http(Some(endpoint)));
    if let Some(token) = token {
        let mut value = http::HeaderValue::from_str(&format!("Bearer {}", token.expose_secret()))
            .map_err(|_| {
            invalid_option(HF_TOKEN, "expected a token without control characters")
        })?;
        value.set_sensitive(true);
        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::AUTHORIZATION, value);
        client = client.with_default_headers(headers);
    }
    HttpBuilder::new()
        .with_url(base)
        .with_client_options(client)
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))
}

/// A read-only store over `https://host[:port]` (`http://` only for a
/// loopback host, for tests): byte-range reads and `HEAD` for sizes. Nothing
/// is sent but the request itself: no credentials, and redirects are not
/// followed to another host with a header (the client sends none).
#[cfg(feature = "cloud")]
fn http_store(url: &ParsedUrl) -> Result<Arc<dyn ObjectStore>> {
    use object_store::http::HttpBuilder;
    let plain = url.store_id.key().starts_with("http://");
    let host = url
        .bucket
        .rsplit_once(':')
        .map_or(url.bucket.as_str(), |(h, _)| h);
    if plain && !matches!(host, "localhost" | "127.0.0.1" | "[::1]") {
        return Err(unsupported(format!(
            "http:// paths to a host other than the loopback ({})",
            url.display
        )));
    }
    HttpBuilder::new()
        .with_url(url.store_id.key())
        .with_client_options(object_store::ClientOptions::new().with_allow_http(plain))
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))
}

/// Without the `cloud` feature there are no builders: cloud URLs parse (so
/// plans still qualify paths) but no store is built.
#[cfg(not(feature = "cloud"))]
fn build(
    _config: &Config,
    _options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
) -> Result<Arc<dyn ObjectStore>> {
    Err(unsupported(format!(
        "cloud object stores are not enabled in this build (enable the `cloud` feature): {}",
        url.display
    )))
}

/// The store id for `base` with per-read `options`: `base` when no `fs.*`
/// option is set, else `base` with a non-secret `#hash` suffix over the
/// `fs.*` options (so reads with different credentials never share a
/// store or a scheduler entry within the query).
fn effective_store_id(base: &StoreId, options: &BTreeMap<String, OptionValue>) -> StoreId {
    let relevant = relevant_options(options);
    if relevant.is_empty() {
        return base.clone();
    }
    StoreId::new(format!("{base}#{:016x}", options_hash(&relevant)))
}

/// The `fs.*` entries of `options` (Spark's store configuration keys),
/// with their keys as written. Keys match ASCII case-insensitively, as
/// Spark's `CaseInsensitiveStringMap` does.
fn relevant_options<'a>(
    options: &'a BTreeMap<String, OptionValue>,
) -> Vec<(&'a str, &'a OptionValue)> {
    let mut out: Vec<(&'a str, &'a OptionValue)> = options
        .iter()
        .filter(|(k, _)| is_store_option(k))
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    out.sort_by(|a, b| {
        a.0.to_ascii_lowercase()
            .cmp(&b.0.to_ascii_lowercase())
            .then_with(|| a.0.cmp(b.0))
    });
    out
}

/// Whether `key` is store configuration (`fs.*`), matched ASCII
/// case-insensitively. Shared with [`crate::options`] (format option
/// parsing skips these keys the same way).
pub(crate) fn is_store_option(key: &str) -> bool {
    key.len() > 3 && key.as_bytes()[..3].eq_ignore_ascii_case(b"fs.")
}

/// Finds `key` in the per-read `options`, ASCII case-insensitively.
#[cfg(feature = "cloud")]
fn find_option<'a>(
    options: &'a BTreeMap<String, OptionValue>,
    key: &str,
) -> Option<&'a OptionValue> {
    options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v)
}

/// A non-secret hash of the relevant per-read options: plain values hash by
/// value, inline secrets by their keyed fingerprint, references by
/// scope and key, and URLs by their redacted text (a wrong-kind per-read
/// value fails when the store builds, so the hash only isolates the cache
/// entry). Only the hash ever reaches a store id.
fn options_hash(relevant: &[(&str, &OptionValue)]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = OFFSET;
    let mut feed = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    };
    for (key, value) in relevant {
        for byte in key.to_ascii_lowercase().as_bytes() {
            feed(*byte);
        }
        feed(0xff);
        match value {
            OptionValue::Plain(text) => {
                feed(0x01);
                for byte in text.as_bytes() {
                    feed(*byte);
                }
            }
            OptionValue::Secret(SecretValue::Inline(secret)) => {
                feed(0x02);
                for byte in secret.fingerprint() {
                    feed(byte);
                }
            }
            OptionValue::Secret(SecretValue::Reference { scope, key }) => {
                feed(0x03);
                for byte in scope.as_bytes().iter().chain(key.as_bytes()) {
                    feed(*byte);
                }
            }
            OptionValue::Url(url) => {
                feed(0x04);
                for byte in url.to_string().as_bytes() {
                    feed(*byte);
                }
            }
        }
        feed(0x00);
    }
    hash
}

/// The path-style setting for an S3 store: the per-read
/// `fs.s3a.path.style.access` option first, else the explicit session
/// value, else path-style with a custom endpoint that looks local (MinIO),
/// else the builder default.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for anything but `true`/`false`.
#[cfg(feature = "cloud")]
fn path_style_access(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
) -> Result<Option<bool>> {
    let explicit = flag(config, options, S3_PATH_STYLE_ACCESS)?;
    if explicit.is_some() {
        return Ok(explicit);
    }
    Ok(text(config, options, S3_ENDPOINT)?
        .map(|e| allows_http(Some(&e)) || e.to_ascii_lowercase().contains("localhost")))
}

/// The session value of an unregistered `key`: the exact session entry
/// first, else the first session entry matching ASCII case-insensitively,
/// so the session side resolves the same way the per-read side does (see
/// the module docs). Session keys are case-sensitive in Hadoop, but this
/// layer documents case-insensitive matching and the per-read side already
/// honours it; resolving both sides the same way keeps one spelling from
/// meaning different things depending on where it is set.
#[cfg(feature = "cloud")]
fn session_raw(config: &Config, key: &str) -> Option<String> {
    if let Some(value) = config.get_unregistered(key) {
        return Some(value.to_owned());
    }
    config
        .explicit_unregistered()
        .into_iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v.to_owned())
}

/// The secret (or plain) value of an unregistered `key`: the per-read
/// option first, else the session value. A per-read value is always
/// present, so an empty per-read value and a per-read value of the wrong
/// kind (a `secret('scope', 'key')` reference, which this layer cannot
/// resolve, or a URL, which is never a credential for these keys) fail
/// rather than silently falling back to the session value. This is the
/// only place the registry's secrets cross from configuration into
/// builders.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (naming the key and the redaction marker, never the value).
#[cfg(feature = "cloud")]
fn cred(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    key: &str,
) -> Result<Option<SecretString>> {
    if let Some(value) = find_option(options, key) {
        return match value {
            OptionValue::Plain(plain) => {
                if plain.is_empty() {
                    Err(invalid_option(key, "expected a non-empty value").into())
                } else {
                    Ok(Some(SecretString::new(plain.clone())))
                }
            }
            OptionValue::Secret(SecretValue::Inline(secret)) => {
                // An empty inline secret is a present value too (Hadoop
                // keeps `""`), and it is invalid for every key read here.
                if secret.expose_secret().is_empty() {
                    Err(invalid_option(key, "expected a non-empty value").into())
                } else {
                    Ok(Some(secret.clone()))
                }
            }
            OptionValue::Secret(SecretValue::Reference { .. }) => Err(invalid_option(
                key,
                "a `secret('scope', 'key')` reference cannot be resolved here",
            )
            .into()),
            OptionValue::Url(_) => {
                Err(invalid_option(key, "a URL value is not a credential for this key").into())
            }
        };
    }
    let raw = session_raw(config, key).filter(|v| !v.is_empty());
    Ok(match raw.as_deref() {
        None => None,
        Some(raw) => match OptionValue::from_user(key, raw) {
            OptionValue::Plain(plain) => Some(SecretString::new(plain)),
            OptionValue::Secret(SecretValue::Inline(secret)) => Some(secret),
            // Unreachable: `from_user` never builds a reference, and values of
            // sensitive keys never classify as URLs. Ignoring them is the
            // fail-safe direction: no value is used that was not typed as the
            // key demands.
            OptionValue::Secret(SecretValue::Reference { .. }) | OptionValue::Url(_) => None,
        },
    })
}

/// The plain value of an unregistered `key`: the per-read option first,
/// else the session value; trimmed, and empty values are invalid. Only
/// for keys that are never sensitive (endpoints, regions). A per-read
/// secret, reference or URL value for a plain key fails rather than
/// smuggling credentials into an endpoint or region.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (naming the key and the redaction marker, never the value).
#[cfg(feature = "cloud")]
fn text(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    key: &str,
) -> Result<Option<String>> {
    if let Some(value) = find_option(options, key) {
        let Some(raw) = value.as_plain() else {
            return Err(invalid_option(key, "a secret is not allowed here").into());
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(invalid_option(key, "expected a non-empty value").into());
        }
        return Ok(Some(trimmed.to_owned()));
    }
    Ok(session_raw(config, key)
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned))
}

/// Whether `text` allows HTTP connections: only an `http://` endpoint does
/// (emulators); anything else keeps the builders' HTTPS default. The scheme
/// matches ASCII case-insensitively, over bytes (never slicing a `&str` at a
/// non-boundary).
#[cfg(feature = "cloud")]
fn allows_http(text: Option<&str>) -> bool {
    text.is_some_and(|e| e.len() >= 7 && e.as_bytes()[..7].eq_ignore_ascii_case(b"http://"))
}

/// Parses a `true`/`false` flag of `key`: the per-read option first, else
/// the session value. A per-read value is always present, so an empty
/// per-read value and a per-read secret, reference or URL value fail
/// rather than silently falling back to the session value.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for anything else. The error names the
/// key and the redaction marker, never the value.
#[cfg(feature = "cloud")]
fn flag(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    key: &str,
) -> Result<Option<bool>> {
    if let Some(value) = find_option(options, key) {
        // Only a plain per-read value overrides: a secret or URL value for
        // a flag fails rather than falling back to the session value.
        let Some(raw) = value.as_plain() else {
            return Err(invalid_option(key, "a secret is not allowed here").into());
        };
        return match raw.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(Some(true)),
            "false" => Ok(Some(false)),
            _ => Err(invalid_option(key, "expected `true` or `false`").into()),
        };
    }
    match session_raw(config, key)
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        None => Ok(None),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "true" => Ok(Some(true)),
            "false" => Ok(Some(false)),
            _ => Err(invalid_option(key, "expected `true` or `false`").into()),
        },
    }
}

/// The `KRISHIV_INVALID_CONFIG_VALUE` for `key`: the key and the redaction
/// marker, never the value.
#[cfg(feature = "cloud")]
fn invalid_option(key: &str, reason: &str) -> KrishivInvalidConfigValue {
    KrishivInvalidConfigValue {
        key: format!("\"{key}\""),
        value: format!("'{REDACTED}'"),
        reason: reason.to_owned(),
    }
}

/// An S3 store for `url`'s bucket.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]); `KRISHIV_OBJECT_STORE_REQUEST_FAILED` when the
/// builder rejects the configuration (names the bucket and the endpoint
/// at most, never a secret).
#[cfg(feature = "cloud")]
fn s3_store(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    path_style: Option<bool>,
) -> Result<Arc<dyn ObjectStore>> {
    use object_store::aws::AmazonS3Builder;
    let mut builder = AmazonS3Builder::from_env().with_bucket_name(&url.bucket);
    if let Some(endpoint) = text(config, options, S3_ENDPOINT)? {
        builder = builder.with_allow_http(allows_http(Some(&endpoint)));
        builder = builder.with_endpoint(endpoint);
    }
    if let Some(key) = cred(config, options, S3_ACCESS_KEY)? {
        builder = builder.with_access_key_id(key.expose_secret());
    }
    if let Some(secret) = cred(config, options, S3_SECRET_KEY)? {
        builder = builder.with_secret_access_key(secret.expose_secret());
    }
    if let Some(token) = cred(config, options, S3_SESSION_TOKEN)? {
        builder = builder.with_token(token.expose_secret());
    }
    if let Some(region) =
        text(config, options, S3_ENDPOINT_REGION)?.or(text(config, options, S3_REGION)?)
    {
        builder = builder.with_region(region);
    }
    if let Some(path_style) = path_style {
        builder = builder.with_virtual_hosted_style_request(!path_style);
    }
    builder
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))
}

/// A GCS store for `url`'s bucket.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]); `KRISHIV_OBJECT_STORE_REQUEST_FAILED` when the
/// builder rejects the configuration (never a secret).
#[cfg(feature = "cloud")]
fn gcs_store(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
) -> Result<Arc<dyn ObjectStore>> {
    use object_store::gcp::GoogleCloudStorageBuilder;
    let mut builder = GoogleCloudStorageBuilder::from_env().with_bucket_name(url.bucket.clone());
    if let Some(path) = cred(config, options, GCS_KEYFILE)? {
        builder = builder.with_service_account_path(path.expose_secret());
    }
    if let Some(key) = cred(config, options, GCS_KEY)? {
        builder = builder.with_service_account_key(key.expose_secret());
    }
    if let Some(base) =
        text(config, options, GCS_STORAGE_ROOT_URL)?.or(text(config, options, GCS_ENDPOINT)?)
    {
        builder = builder.with_base_url(&base);
    }
    builder
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))
}

/// An Azure Blob store for `url`'s container.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]); `KRISHIV_OBJECT_STORE_REQUEST_FAILED` when the
/// builder rejects the configuration (never a secret).
#[cfg(feature = "cloud")]
fn azure_store(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
) -> Result<Arc<dyn ObjectStore>> {
    use object_store::azure::MicrosoftAzureBuilder;
    let mut builder = MicrosoftAzureBuilder::from_env().with_container_name(url.bucket.clone());
    let account = url
        .account
        .clone()
        .or(text(config, options, AZURE_ACCOUNT_NAME)?);
    if let Some(account) = &account {
        builder = builder.with_account(account);
    }
    if let Some(key) = azure_key(config, options, url, account.as_deref())? {
        builder = builder.with_access_key(key.expose_secret());
    }
    if let Some(sas) = azure_sas(config, options, url, account.as_deref())? {
        let pairs = sas_pairs(&sas).map_err(|reason| store_failed(&url.display, 1, &reason))?;
        builder = builder.with_sas_authorization(pairs);
    }
    if let Some(endpoint) = text(config, options, AZURE_ENDPOINT)? {
        builder = builder.with_allow_http(allows_http(Some(&endpoint)));
        builder = builder.with_endpoint(endpoint);
    }
    builder
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))
}

/// The Azure account key: the dotted `fs.azure.account.key.<account>.*`
/// forms first, then the bare fallback.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]).
#[cfg(feature = "cloud")]
fn azure_key(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    account: Option<&str>,
) -> Result<Option<SecretString>> {
    Ok(
        dotted(config, options, "fs.azure.account.key", url, account)?.or(cred(
            config,
            options,
            AZURE_ACCOUNT_KEY,
        )?),
    )
}

/// The Azure shared access signature: the dotted `fs.azure.sas.*` forms
/// first, then the bare fallback.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]).
#[cfg(feature = "cloud")]
fn azure_sas(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    account: Option<&str>,
) -> Result<Option<SecretString>> {
    Ok(
        dotted(config, options, "fs.azure.sas", url, account)?
            .or(cred(config, options, AZURE_SAS)?),
    )
}

/// The value of the most specific dotted key for `prefix`:
/// `<prefix>.<container>.<account>.dfs.core.windows.net`,
/// `<prefix>.<container>.<account>.blob.core.windows.net`,
/// `<prefix>.<container>.<account>`, then `<prefix>.<account>` when the URL
/// carries an account. Per-read options override the session value for each
/// candidate key.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]).
#[cfg(feature = "cloud")]
fn dotted(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
    prefix: &str,
    url: &ParsedUrl,
    account: Option<&str>,
) -> Result<Option<SecretString>> {
    let Some(account) = account else {
        return Ok(None);
    };
    for key in [
        format!("{prefix}.{}.{account}.dfs.core.windows.net", url.bucket),
        format!("{prefix}.{}.{account}.blob.core.windows.net", url.bucket),
        format!("{prefix}.{}.{account}", url.bucket),
        format!("{prefix}.{account}"),
    ] {
        if let Some(value) = cred(config, options, &key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// Splits a SAS token (`sv=..&sig=..`) into its query pairs, redacting any
/// complaint: the token itself must never reach an error. A leading `?` (as
/// Azure portals copy it) and surrounding whitespace are ignored.
#[cfg(feature = "cloud")]
fn sas_pairs(secret: &SecretString) -> Result<Vec<(String, String)>, String> {
    let token = secret.expose_secret().trim();
    let token = token.strip_prefix('?').unwrap_or(token);
    let mut pairs = Vec::new();
    for part in token.split('&') {
        match part.split_once('=') {
            Some((name, value)) if !name.is_empty() && !value.is_empty() => {
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
        // `ensure_with_options` registers under a suffixed id whose text is
        // also free of the secret.
        let sched = crate::IoScheduler::new(crate::IoConfig::from_config(&Config::new()));
        let (id, _, _) = registry
            .ensure_with_options(
                &sched,
                "s3://bucket/k",
                &read_options(&[
                    (S3_ENDPOINT, "http://127.0.0.1:9001"),
                    (S3_ACCESS_KEY, CANARY),
                    (S3_SECRET_KEY, CANARY),
                ]),
            )
            .unwrap();
        assert!(!id.to_string().contains(CANARY));
        assert!(!format!("{id:?}").contains(CANARY));
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
            let sched = crate::IoScheduler::new(crate::IoConfig::from_config(&Config::new()));
            let error = registry
                .ensure_with_options(&sched, input, &options)
                .unwrap_err();
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
        assert!(plain.to_string().starts_with("s3://bucket#"));
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
        let sched = crate::IoScheduler::new(crate::IoConfig::from_config(&Config::new()));
        let error = registry
            .ensure_with_options(&sched, "s3://bucket/k", &options)
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
}
