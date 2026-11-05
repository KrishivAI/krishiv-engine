// SPDX-License-Identifier: Apache-2.0
//! Cloud object stores: S3, GCS and Azure Blob Storage, and the read-only web stores (P2-T084,
//! T195).
//!
//! [`StoreRegistry`] builds one [`ObjectStore`]
//! per bucket (S3, GCS) or container (Azure) from Spark-style
//! Hadoop settings ([`StoreSettings`]: the engine's session configuration, or a map), lazily on
//! first use. It is scoped to its owner (a query, for the engine): it holds one settings
//! snapshot, caches stores only for itself, and is never a process-global cache keyed by
//! credentials.
//!
//! # URLs
//!
//! [`parse_object_url`] splits a user path into its scheme, bucket (or
//! container) and key: `s3://` and `s3a://`, `gs://`, `abfss://` and
//! `abfs://` (both `container@account.host/key` and `container/key`), and
//! `az://container/key`; the read-only web stores `https://host[:port]/key`
//! (`http://` only for a loopback host) and `hf://datasets/<org>/<name>[@rev]/<path>`
//! (a Krishiv extension, not Spark behaviour, so no `spark.*` key: reads
//! through the Hugging Face `resolve` endpoint with `fs.hf.token` or
//! `HF_TOKEN` as a bearer header, `fs.hf.endpoint` for the host; with an
//! endpoint override a token is sent only when the same per-read options
//! carry it, and only over https or to a loopback host). Both are
//! byte-range reads and `HEAD` only; no listing or globs. Anything else with a scheme is
//! `KRISHIV_UNSUPPORTED_FEATURE`, as are paths that smuggle credentials in
//! the URL itself (user info): like the rest of the engine, this layer only
//! accepts credentials from configuration, never from paths (plans stay
//! pure data: a path string plus option values).
//!
//! # Configuration
//!
//! Unregistered `fs.*` keys of the session configuration (anything that is
//! not a `krishiv.*` key is kept verbatim by the engine's `Config`), with the
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
//! [`krishiv_common::secret::SecretString`] the moment it is read and is only exposed at the builder
//! call. [`StoreRegistry`]'s `Debug`, every error here and every store id
//! and path are free of secret values: builder errors name the bucket, the
//! endpoint or a credential file path at most.
//!
//! # Requests
//!
//! The registry hands out raw stores; the caller registers them on its request scheduler (the
//! engine's `IoScheduler`, in `krishiv-io`), so every request still goes through that layer's
//! retry, timeout, pacing and accounting wrappers unchanged.
//! Each cloud scheme paces with its own [`StorePolicy`] (`S3`, `GCS`,
//! `AZURE`); the GCS and Azure budgets are conservative starting points
//! until per-service rates are calibrated. Multipart
//! uploads rely on the cloud clients' own per-part retries, set by [`RetryPolicy`]
//! ([`StoreRegistry::with_retry_policy`]; default: `object_store`'s own).
//!
//! Cloud builders live behind this crate's `cloud` feature (the `aws`,
//! `azure` and `gcp` features of `object_store`): without it, building a
//! cloud store fails with `KRISHIV_UNSUPPORTED_FEATURE`, while parsing URLs
//! and pacing policies keep working so plans still qualify paths.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

#[cfg(feature = "cloud")]
use krishiv_common::error::KrishivInvalidConfigValue;
use krishiv_common::secret::{OptionValue, SecretValue};
#[cfg(feature = "cloud")]
use krishiv_common::secret::{REDACTED, SecretString};
use object_store::ObjectStore;
use object_store::path::Path;

use crate::credentials::{CredentialCache, two_credential_sources};
#[cfg(feature = "cloud")]
use crate::error::store_failed;
use crate::error::{Result, unsupported};
use crate::{
    CredentialScope, CredentialSource, DEFAULT_REFRESH_SKEW, RetryPolicy, StoreId, StorePolicy,
    StoreSettings,
};
#[cfg(feature = "cloud")]
use crate::{RefreshingProvider, RefreshingStore};

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
/// `HF_TOKEN` environment variable, for the default host only). With an
/// `fs.hf.endpoint` override it is used only as a per-read option. A secret;
/// never in a plan.
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
    // `.` and `..` are refused: URL parsing would resolve them and leave the dataset's path.
    let plain = |t: &str| {
        !t.is_empty()
            && !t.chars().all(|c| c == '.')
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
/// overrides. Scoped to its owner: it holds one settings snapshot and caches stores only for
/// itself.
pub struct StoreRegistry {
    config: Arc<dyn StoreSettings>,
    stores: Mutex<HashMap<StoreId, Arc<dyn ObjectStore>>>,
    /// Locations with credentials of their own (see [`Self::with_source`]).
    scopes: Mutex<Vec<Arc<ScopeEntry>>>,
    /// Refresh skew of the sources added next.
    skew: Duration,
    /// Retry and backoff of the cloud clients (see [`Self::with_retry_policy`]).
    retry: RetryPolicy,
    /// Every scoped store id this registry handed out, for [`Self::on_release`].
    scoped_ids: Mutex<HashSet<StoreId>>,
    release: Mutex<Vec<ReleaseHook>>,
}

/// Called with each scoped store id when the registry is dropped.
type ReleaseHook = Arc<dyn Fn(&StoreId) + Send + Sync>;

/// One credential scope of a registry: the location, its process-unique id suffix and the
/// credential cache every store of the scope signs with.
struct ScopeEntry {
    scope: CredentialScope,
    nonce: String,
    cache: Arc<CredentialCache>,
}

/// A process-unique, unguessable-enough suffix for a scoped store id: the `IoScheduler` is
/// process-wide and registering an id again replaces its entry, so two registries (queries)
/// must never share a scoped id. A process-wide counter makes it unique; the random part keeps
/// ids apart across restarts that share a log.
fn scope_nonce() -> String {
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let random = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    format!("{random:016x}{:x}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

impl StoreRegistry {
    /// A registry building stores from `settings` (for the engine, the query's [`Config`]
    /// snapshot; for anyone else, a map of `fs.*` keys, see [`StoreSettings`]).
    ///
    /// [`Config`]: krishiv_common::config::Config
    #[must_use]
    pub fn new<S: StoreSettings + 'static>(settings: Arc<S>) -> Self {
        Self::from_shared(settings)
    }

    /// As [`Self::new`], for settings already behind a `dyn StoreSettings`.
    #[must_use]
    pub fn from_shared(settings: Arc<dyn StoreSettings>) -> Self {
        Self {
            config: settings,
            stores: Mutex::new(HashMap::new()),
            scopes: Mutex::new(Vec::new()),
            skew: DEFAULT_REFRESH_SKEW,
            retry: RetryPolicy::DEFAULT,
            scoped_ids: Mutex::new(HashSet::new()),
            release: Mutex::new(Vec::new()),
        }
    }

    /// Sets how long before their expiry the credentials of the sources added *after* this
    /// call are refreshed (default [`DEFAULT_REFRESH_SKEW`]; the engine's config
    /// `krishiv.io.credentials.refreshSkew`).
    #[must_use]
    pub fn with_refresh_skew(mut self, skew: Duration) -> Self {
        self.skew = skew;
        self
    }

    /// Sets the retry and backoff of the cloud clients of the stores built *after* this call
    /// (default [`RetryPolicy::DEFAULT`], `object_store`'s own). The engine sets fewer retries
    /// when its request scheduler retries as well, so the two layers do not multiply.
    #[must_use]
    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Gives the locations under `scope` credentials from `source`: their stores sign with a
    /// refreshing provider instead of static keys.
    ///
    /// - The longest matching prefix wins, on path segments (`warehouse/t1/` does not match
    ///   `warehouse/t10/`).
    /// - A scope is registered under a store id with a random nonce, so two registries (two
    ///   queries on one scheduler) never share a store, and neither reads with the other's
    ///   credentials.
    /// - Session-level static keys do not apply under a scope, and an explicit per-read `fs.*`
    ///   credential on a scoped location fails (`KRISHIV_INVALID_CONFIG_VALUE`: two
    ///   credential sources for one read). Outside every scope static configuration works as
    ///   before.
    #[must_use]
    pub fn with_source(self, scope: CredentialScope, source: Arc<dyn CredentialSource>) -> Self {
        self.add_source(scope, source);
        self
    }

    /// As [`Self::with_source`], on a registry that is already shared (behind an `Arc`, in use
    /// by a running query): an Iceberg `loadTable` vends credentials mid-query. The scope
    /// applies to stores opened after this call; a store already handed out keeps the
    /// credentials it was built with (it has a different store id, so it is never reused for the
    /// new scope). Adding a scope for a location already scoped replaces it for later opens.
    pub fn add_source(&self, scope: CredentialScope, source: Arc<dyn CredentialSource>) {
        let cache = CredentialCache::system(scope.clone(), source, self.skew);
        self.scopes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::new(ScopeEntry {
                scope,
                nonce: scope_nonce(),
                cache,
            }));
    }

    /// Registers `hook` to be called with each scoped store id this registry handed out when
    /// the registry is dropped (at query end), so the owner of a process-wide scheduler can
    /// unregister them. Plain, unscoped ids are not reported.
    pub fn on_release(&self, hook: impl Fn(&StoreId) + Send + Sync + 'static) {
        self.release
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::new(hook));
    }

    /// The scoped store ids handed out so far.
    #[must_use]
    pub fn scoped_ids(&self) -> Vec<StoreId> {
        self.scoped_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .cloned()
            .collect()
    }

    /// The scope covering `url`, the most specific first.
    fn scope_for(&self, url: &ParsedUrl) -> Option<Arc<ScopeEntry>> {
        self.scopes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|e| e.scope.store_id() == &url.store_id && e.scope.covers(&url.key))
            // On equal depth the newest wins (`max_by_key` keeps the last maximum).
            .max_by_key(|e| e.scope.depth())
            .cloned()
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
        let scoped = self.scope_for(url);
        if let Some(entry) = &scoped {
            reject_credential_options(options, &entry.scope)?;
        }
        let id = self.id_of(url, scoped.as_deref(), options);
        if let Some(store) = self
            .stores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&id)
            .cloned()
        {
            return Ok(store);
        }
        let store = build(
            self.config.as_ref(),
            options,
            url,
            scoped.as_ref().map(|e| &e.cache),
            &self.retry,
        )?;
        if scoped.is_some() {
            self.scoped_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(id.clone());
        }
        self.stores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, Arc::clone(&store));
        Ok(store)
    }

    /// The id of `url`'s store under `options`: the URL's store id (with the `#scope:` nonce
    /// of its credential scope, if any) plus the non-secret hash of the per-read options.
    fn id_of(
        &self,
        url: &ParsedUrl,
        scoped: Option<&ScopeEntry>,
        options: &BTreeMap<String, OptionValue>,
    ) -> StoreId {
        match scoped {
            Some(entry) => effective_store_id(
                &StoreId::new(format!("{}#scope:{}", url.store_id.key(), entry.nonce)),
                options,
            ),
            None => effective_store_id(&url.store_id, options),
        }
    }

    /// Parses `text` and builds its store: the id to register it under, the store, the key within
    /// it, the display path and the pacing policy of its scheme.
    ///
    /// # Errors
    /// As [`parse_object_url`] and [`Self::store`].
    pub fn open(&self, text: &str) -> Result<OpenedStore> {
        self.open_with_options(text, &BTreeMap::new())
    }

    /// As [`Self::open`], with per-read `fs.*` data-source options overriding the settings (see
    /// [`Self::store_with_options`]).
    ///
    /// # Errors
    /// As [`parse_object_url`] and [`Self::store_with_options`].
    pub fn open_with_options(
        &self,
        text: &str,
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<OpenedStore> {
        let url = parse_object_url(text)?;
        let store = self.store_with_options(&url, options)?;
        Ok(OpenedStore {
            id: self.id_of(&url, self.scope_for(&url).as_deref(), options),
            store,
            key: url.key,
            display: url.display,
            policy: policy_for(url.scheme),
        })
    }
}

impl Drop for StoreRegistry {
    fn drop(&mut self) {
        let hooks =
            std::mem::take(&mut *self.release.lock().unwrap_or_else(PoisonError::into_inner));
        let ids = std::mem::take(
            &mut *self
                .scoped_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        for id in &ids {
            for hook in &hooks {
                hook(id);
            }
        }
    }
}

/// Whether `key` carries a credential (`fs.*`, ASCII case-insensitive): static keys, tokens,
/// key files and the dotted Azure forms.
fn is_credential_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    [
        S3_ACCESS_KEY,
        S3_SECRET_KEY,
        S3_SESSION_TOKEN,
        GCS_KEYFILE,
        GCS_KEY,
        AZURE_ACCOUNT_KEY,
        AZURE_SAS,
    ]
    .contains(&lower.as_str())
        || lower.starts_with("fs.azure.account.key.")
        || lower.starts_with("fs.azure.sas.")
}

/// A per-read credential on a scoped location would be a second credential source for the
/// read: it fails, naming the key and never the value.
fn reject_credential_options(
    options: &BTreeMap<String, OptionValue>,
    scope: &CredentialScope,
) -> Result<()> {
    match options.keys().find(|k| is_credential_key(k)) {
        Some(key) => Err(two_credential_sources(key, scope)),
        None => Ok(()),
    }
}

/// A store opened from a URL by [`StoreRegistry::open`]: everything a scheduler needs to
/// register it and to address the object.
#[derive(Clone)]
#[non_exhaustive]
pub struct OpenedStore {
    /// The id to register the store under (the URL's store id, with a non-secret `#hash` suffix
    /// when per-read `fs.*` options are set).
    pub id: StoreId,
    /// The store.
    pub store: Arc<dyn ObjectStore>,
    /// The key within the store.
    pub key: Path,
    /// The URL in canonical form, safe to print.
    pub display: String,
    /// The pacing policy of the URL's scheme.
    pub policy: StorePolicy,
}

impl fmt::Debug for OpenedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenedStore")
            .field("id", &self.id)
            .field("key", &self.key)
            .field("display", &self.display)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
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
    config: &dyn StoreSettings,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    scope: Option<&Arc<CredentialCache>>,
    retry: &RetryPolicy,
) -> Result<Arc<dyn ObjectStore>> {
    // Configuration errors keep their class; only builder failures become
    // `KRISHIV_OBJECT_STORE_REQUEST_FAILED` (converted in each scheme
    // builder below).
    let path_style = match url.scheme {
        ObjectScheme::S3 => path_style_access(config, options)?,
        _ => None,
    };
    match url.scheme {
        ObjectScheme::S3 => s3_store(config, options, url, path_style, scope, retry),
        ObjectScheme::Gcs => gcs_store(config, options, url, scope, retry),
        ObjectScheme::Azure => azure_store(config, options, url, scope, retry),
        ObjectScheme::Http => http_store(url, retry),
        ObjectScheme::HuggingFace => hf_store(config, options, url, retry),
    }
}

/// A Hugging Face dataset at one revision, read through the `resolve`
/// endpoint with byte ranges. The token (`fs.hf.token`, else `HF_TOKEN`, both
/// for the default host only; with an endpoint override, only a per-read
/// `fs.hf.token`) is sent as a sensitive bearer header, over https or to a
/// loopback host; reqwest drops it on a redirect to
/// another host (the CDN), and it is never in a plan, error or log.
#[cfg(feature = "cloud")]
fn hf_store(
    config: &dyn StoreSettings,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    retry: &RetryPolicy,
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
    // A token goes only to the default host, or, with an endpoint override, only when the same per-read
    // options carry the token: `fs.*` keys can be changed by clients (`SET`), so a session endpoint is as
    // caller-controlled as a per-read one, and neither may collect the server's `HF_TOKEN` or a session
    // `fs.hf.token`.
    let custom_endpoint = text(config, options, HF_ENDPOINT)?.is_some();
    let token = if custom_endpoint {
        if find_option(options, HF_TOKEN).is_some() {
            cred(config, options, HF_TOKEN)?
        } else {
            None
        }
    } else {
        match cred(config, options, HF_TOKEN)? {
            Some(token) => Some(token),
            None => std::env::var("HF_TOKEN")
                .ok()
                .filter(|t| !t.is_empty())
                .map(SecretString::new),
        }
    };
    if token.is_some() && !endpoint.starts_with("https://") && !loopback_endpoint(endpoint) {
        return Err(invalid_option(
            HF_ENDPOINT,
            "a token is sent only over https (or to a loopback host)",
        )
        .into());
    }
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
        .with_retry(retry.to_config())
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))
}

/// Whether `endpoint` (`http://host[:port][/...]`) names a loopback host.
#[cfg(feature = "cloud")]
fn loopback_endpoint(endpoint: &str) -> bool {
    let rest = endpoint.split_once("://").map_or(endpoint, |(_, r)| r);
    let authority = rest.split('/').next().unwrap_or_default();
    // Userinfo (`http://localhost:x@evil.com`) changes the host the URL parser picks: not a loopback form.
    if authority.contains('@') || authority.contains('\\') {
        return false;
    }
    let host = if let Some(v6) = authority.strip_prefix('[') {
        v6.split(']').next().unwrap_or_default()
    } else {
        authority.split(':').next().unwrap_or_default()
    };
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

/// A read-only store over `https://host[:port]` (`http://` only for a
/// loopback host, for tests): byte-range reads and `HEAD` for sizes. Nothing
/// is sent but the request itself: no credentials, and redirects are not
/// followed to another host with a header (the client sends none).
#[cfg(feature = "cloud")]
fn http_store(url: &ParsedUrl, retry: &RetryPolicy) -> Result<Arc<dyn ObjectStore>> {
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
        .with_retry(retry.to_config())
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))
}

/// Without the `cloud` feature there are no builders: cloud URLs parse (so
/// plans still qualify paths) but no store is built.
#[cfg(not(feature = "cloud"))]
fn build(
    _config: &dyn StoreSettings,
    _options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    _scope: Option<&Arc<CredentialCache>>,
    _retry: &RetryPolicy,
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
/// case-insensitively. A reader of data-source options uses it to skip the keys this layer
/// owns (the engine's format option parsing does).
pub fn is_store_option(key: &str) -> bool {
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
    config: &dyn StoreSettings,
    options: &BTreeMap<String, OptionValue>,
) -> Result<Option<bool>> {
    let explicit = flag(config, options, S3_PATH_STYLE_ACCESS)?;
    if explicit.is_some() {
        return Ok(explicit);
    }
    Ok(text(config, options, S3_ENDPOINT)?
        .map(|e| allows_http(Some(&e)) || e.to_ascii_lowercase().contains("localhost")))
}

/// The typed session value of `key`: the exact entry first, else the first entry matching ASCII
/// case-insensitively ([`StoreSettings::lookup`]), so the session side resolves the same way the
/// per-read side does (see the module docs).
#[cfg(feature = "cloud")]
fn session_value(config: &dyn StoreSettings, key: &str) -> Option<OptionValue> {
    config.lookup(key)
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
    config: &dyn StoreSettings,
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
    Ok(match session_value(config, key) {
        None => None,
        Some(OptionValue::Plain(plain)) => (!plain.is_empty()).then(|| SecretString::new(plain)),
        Some(OptionValue::Secret(SecretValue::Inline(secret))) => {
            (!secret.expose_secret().is_empty()).then_some(secret)
        }
        Some(OptionValue::Secret(SecretValue::Reference { .. })) => {
            return Err(invalid_option(
                key,
                "a `secret('scope', 'key')` reference cannot be resolved here",
            )
            .into());
        }
        Some(OptionValue::Url(_)) => {
            return Err(invalid_option(key, "a URL value is not a credential for this key").into());
        }
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
    config: &dyn StoreSettings,
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
    Ok(plain_session_value(config, key)?
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned))
}

/// The session value of a key that is never sensitive, as text: a secret or a URL with
/// credentials fails (it must not reach an endpoint, region or flag), as a per-read one does.
#[cfg(feature = "cloud")]
fn plain_session_value(config: &dyn StoreSettings, key: &str) -> Result<Option<String>> {
    match session_value(config, key) {
        None => Ok(None),
        Some(OptionValue::Plain(plain)) => Ok(Some(plain)),
        Some(OptionValue::Secret(_) | OptionValue::Url(_)) => {
            Err(invalid_option(key, "a secret is not allowed here").into())
        }
    }
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
    config: &dyn StoreSettings,
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
    match plain_session_value(config, key)?
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
    config: &dyn StoreSettings,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    path_style: Option<bool>,
    scope: Option<&Arc<CredentialCache>>,
    retry: &RetryPolicy,
) -> Result<Arc<dyn ObjectStore>> {
    use object_store::aws::{AmazonS3Builder, AwsCredential};
    let mut builder = AmazonS3Builder::from_env().with_bucket_name(&url.bucket);
    if let Some(endpoint) = text(config, options, S3_ENDPOINT)? {
        builder = builder.with_allow_http(allows_http(Some(&endpoint)));
        builder = builder.with_endpoint(endpoint);
    }
    // Under a credential scope the source is the only credential: session keys do not apply.
    let provider = scope.map(|c| RefreshingProvider::<AwsCredential>::from_cache(Arc::clone(c)));
    if let Some(provider) = &provider {
        builder = builder.with_credentials(Arc::new(provider.clone()));
    } else {
        if let Some(key) = cred(config, options, S3_ACCESS_KEY)? {
            builder = builder.with_access_key_id(key.expose_secret());
        }
        if let Some(secret) = cred(config, options, S3_SECRET_KEY)? {
            builder = builder.with_secret_access_key(secret.expose_secret());
        }
        if let Some(token) = cred(config, options, S3_SESSION_TOKEN)? {
            builder = builder.with_token(token.expose_secret());
        }
    }
    if let Some(region) =
        text(config, options, S3_ENDPOINT_REGION)?.or(text(config, options, S3_REGION)?)
    {
        builder = builder.with_region(region);
    }
    if let Some(path_style) = path_style {
        builder = builder.with_virtual_hosted_style_request(!path_style);
    }
    let store = builder
        .with_retry(retry.to_config())
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))?;
    Ok(refreshing(store, provider.as_ref()))
}

/// Wraps `store` so a stale credential gets one forced refresh and one retry, when it signs
/// with a refreshing `provider`.
#[cfg(feature = "cloud")]
fn refreshing<C>(
    store: Arc<dyn ObjectStore>,
    provider: Option<&RefreshingProvider<C>>,
) -> Arc<dyn ObjectStore> {
    match provider {
        Some(provider) => Arc::new(RefreshingStore::new(store, provider)),
        None => store,
    }
}

/// A GCS store for `url`'s bucket.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]); `KRISHIV_OBJECT_STORE_REQUEST_FAILED` when the
/// builder rejects the configuration (never a secret).
#[cfg(feature = "cloud")]
fn gcs_store(
    config: &dyn StoreSettings,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    scope: Option<&Arc<CredentialCache>>,
    retry: &RetryPolicy,
) -> Result<Arc<dyn ObjectStore>> {
    use object_store::gcp::{GcpCredential, GoogleCloudStorageBuilder};
    let mut builder = GoogleCloudStorageBuilder::from_env().with_bucket_name(url.bucket.clone());
    let provider = scope.map(|c| RefreshingProvider::<GcpCredential>::from_cache(Arc::clone(c)));
    if let Some(provider) = &provider {
        builder = builder.with_credentials(Arc::new(provider.clone()));
    } else {
        if let Some(path) = cred(config, options, GCS_KEYFILE)? {
            builder = builder.with_service_account_path(path.expose_secret());
        }
        if let Some(key) = cred(config, options, GCS_KEY)? {
            builder = builder.with_service_account_key(key.expose_secret());
        }
    }
    if let Some(base) =
        text(config, options, GCS_STORAGE_ROOT_URL)?.or(text(config, options, GCS_ENDPOINT)?)
    {
        builder = builder.with_base_url(&base);
    }
    let store = builder
        .with_retry(retry.to_config())
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))?;
    Ok(refreshing(store, provider.as_ref()))
}

/// An Azure Blob store for `url`'s container.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]); `KRISHIV_OBJECT_STORE_REQUEST_FAILED` when the
/// builder rejects the configuration (never a secret).
#[cfg(feature = "cloud")]
fn azure_store(
    config: &dyn StoreSettings,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    scope: Option<&Arc<CredentialCache>>,
    retry: &RetryPolicy,
) -> Result<Arc<dyn ObjectStore>> {
    use object_store::azure::{AzureCredential, MicrosoftAzureBuilder};
    let mut builder = MicrosoftAzureBuilder::from_env().with_container_name(url.bucket.clone());
    let account = url
        .account
        .clone()
        .or(text(config, options, AZURE_ACCOUNT_NAME)?);
    if let Some(account) = &account {
        builder = builder.with_account(account);
    }
    let provider = scope.map(|c| RefreshingProvider::<AzureCredential>::from_cache(Arc::clone(c)));
    if let Some(provider) = &provider {
        builder = builder.with_credentials(Arc::new(provider.clone()));
    } else {
        if let Some(key) = azure_key(config, options, url, account.as_deref())? {
            builder = builder.with_access_key(key.expose_secret());
        }
        if let Some(sas) = azure_sas(config, options, url, account.as_deref())? {
            let pairs = sas_pairs(&sas).map_err(|reason| store_failed(&url.display, 1, &reason))?;
            builder = builder.with_sas_authorization(pairs);
        }
    }
    if let Some(endpoint) = text(config, options, AZURE_ENDPOINT)? {
        builder = builder.with_allow_http(allows_http(Some(&endpoint)));
        builder = builder.with_endpoint(endpoint);
    }
    let store = builder
        .with_retry(retry.to_config())
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))?;
    Ok(refreshing(store, provider.as_ref()))
}

/// The Azure account key: the dotted `fs.azure.account.key.<account>.*`
/// forms first, then the bare fallback.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read
/// value (see [`cred`]).
#[cfg(feature = "cloud")]
fn azure_key(
    config: &dyn StoreSettings,
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
    config: &dyn StoreSettings,
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
    config: &dyn StoreSettings,
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
mod tests;
