// SPDX-License-Identifier: Apache-2.0
//! Azure Blob Storage and OneLake stores: the endpoint from the URL's host and the credential from
//! Hadoop ABFS keys (child module of `stores`).
//!
//! # Endpoint
//! As object_store 0.14.2's `MicrosoftAzureBuilder::parse_url` (`azure/builder.rs` l.720-756)
//! sets it: `*.core.windows.net` hosts address `https://<account>.blob.core.windows.net`,
//! `*.fabric.microsoft.com` hosts the Fabric endpoint `https://<account>.blob.fabric.microsoft.com`
//! (`with_use_fabric_endpoint`), and a workspace private link
//! (`<workspace>.z<xy>.<service>.fabric.microsoft.com`) `https://<host>` itself, with the account
//! `<workspace>.z<xy>`. `fs.azure.endpoint` overrides all of them.
//!
//! # Credential
//! Hadoop ABFS 3.4.3 (`AbfsConfiguration.getTokenProvider`, `getSASTokenProvider`, `AuthType`):
//! `fs.azure.account.auth.type` is `SharedKey` (the account key), `SAS`
//! (`fs.azure.sas.fixed.token`) or `OAuth` with `fs.azure.account.oauth.provider.type` naming
//! `ClientCredsTokenProvider`, `WorkloadIdentityTokenProvider` or `MsiTokenProvider`; Krishiv adds
//! `fs.azure.account.oauth2.access.token`, a static bearer token, for `OAuth` without a provider
//! class. Without an auth type the credential keys are tried in the order Krishiv always did
//! (account key, SAS), then the bearer token, then the builder's environment.
//!
//! Every key is resolved as ABFS resolves it (`AbfsConfiguration.accountConf`/`containerConf`):
//! the account-host form `<key>.<host>` beats the bare `<key>`, and a secret
//! (`getPasswordString`: keys, tokens, client ids, endpoints, tenants) also has the container form
//! `<key>.<container>.<host>` first. The host is the URL's (`account.dfs.core.windows.net`), its
//! `blob` twin, and for Azure Storage accounts Krishiv's older `<key>.<container>.<account>` and
//! `<key>.<account>` forms. A provider class set only on the bare key applies only when the bare
//! auth type is `OAuth` too (`getTokenProviderClass`).
//!
//! With an explicit credential the builder starts empty (`MicrosoftAzureBuilder::new`), never from
//! the process environment, so an ambient `AZURE_STORAGE_TOKEN`, `AZURE_CLIENT_ID`,
//! `AZURE_TENANT_ID` or `AZURE_FEDERATED_TOKEN_FILE` (which `from_env`'s auto resolution would
//! prefer: bearer token, then access key, then workload identity, then client secret, then SAS)
//! never changes it; the credential type is pinned too.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::secret::{OptionValue, SecretString};
use object_store::ObjectStore;
use object_store::azure::{AzureCredential, MicrosoftAzureBuilder};

use super::{
    AZURE_ACCOUNT_KEY, AZURE_ACCOUNT_NAME, AZURE_AUTH_TYPE, AZURE_ENDPOINT,
    AZURE_OAUTH_ACCESS_TOKEN, AZURE_OAUTH_CLIENT_ENDPOINT, AZURE_OAUTH_CLIENT_ID,
    AZURE_OAUTH_CLIENT_SECRET, AZURE_OAUTH_MSI_AUTHORITY, AZURE_OAUTH_MSI_ENDPOINT,
    AZURE_OAUTH_MSI_TENANT, AZURE_OAUTH_PROVIDER_TYPE, AZURE_OAUTH_TOKEN_FILE, AZURE_SAS,
    AZURE_SAS_FIXED_TOKEN, AZURE_SAS_TOKEN_PROVIDER_TYPE, AzureHost, ParsedUrl, allows_http, cred,
    invalid_option, loopback_endpoint, refreshing, text,
};
use crate::credentials::CredentialCache;
use crate::error::{Result, store_failed, unsupported};
use crate::{RefreshingProvider, SharedHttpClients, StoreSettings};

/// Hadoop ABFS's client-credentials provider (`fs.azure.account.oauth.provider.type`).
const CLIENT_CREDS: &str = "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider";
/// Hadoop ABFS's workload-identity provider.
const WORKLOAD_IDENTITY: &str =
    "org.apache.hadoop.fs.azurebfs.oauth2.WorkloadIdentityTokenProvider";
/// Hadoop ABFS's managed-identity provider.
const MSI: &str = "org.apache.hadoop.fs.azurebfs.oauth2.MsiTokenProvider";
/// `AuthConfigurations.DEFAULT_FS_AZURE_ACCOUNT_OAUTH_MSI_AUTHORITY`.
const DEFAULT_AUTHORITY: &str = "https://login.microsoftonline.com/";
/// `AuthConfigurations.DEFAULT_FS_AZURE_ACCOUNT_OAUTH_TOKEN_FILE`.
const DEFAULT_TOKEN_FILE: &str = "/var/run/secrets/azure/tokens/azure-identity-token";

/// The credential an Azure store signs with, from explicit configuration.
enum Auth {
    /// Shared key (the account key).
    Key(SecretString),
    /// A shared access signature.
    Sas(SecretString),
    /// A static bearer token.
    Bearer(SecretString),
    /// Entra client credentials (`ClientCredsTokenProvider`).
    ClientSecret {
        client_id: SecretString,
        secret: SecretString,
        tenant: String,
        authority: String,
    },
    /// Workload identity federation (`WorkloadIdentityTokenProvider`).
    Workload {
        client_id: SecretString,
        tenant: SecretString,
        token_file: SecretString,
        authority: String,
    },
    /// The host's managed identity (`MsiTokenProvider`).
    Managed {
        client_id: Option<SecretString>,
        endpoint: Option<SecretString>,
    },
}

impl Auth {
    /// The object_store credential type this auth pins (`with_credential_type`).
    fn credential_type(&self) -> &'static str {
        match self {
            Self::Key(_) => "access_key",
            Self::Sas(_) => "sas_token",
            Self::Bearer(_) => "bearer_token",
            Self::ClientSecret { .. } => "client_secret",
            Self::Workload { .. } => "workload_identity",
            Self::Managed { .. } => "managed_identity",
        }
    }
}

/// The keys of one Azure location: settings, per-read options and the names a key is scoped by.
struct Lookup<'a> {
    config: &'a dyn StoreSettings,
    options: &'a BTreeMap<String, OptionValue>,
    container: &'a str,
    /// The account hosts, most specific first.
    hosts: Vec<String>,
    /// The account, for Krishiv's older `<key>.<account>` forms (Azure Storage only).
    account: Option<String>,
}

impl Lookup<'_> {
    /// The account-scoped forms of `base`: `<base>.<host>` then `<base>.<account>`.
    fn account_keys(&self, base: &str) -> Vec<String> {
        let mut keys: Vec<String> = self.hosts.iter().map(|h| format!("{base}.{h}")).collect();
        if let Some(account) = &self.account {
            keys.push(format!("{base}.{account}"));
        }
        keys
    }

    /// The forms of a secret `base`, as `getPasswordString` reads them: container, account, bare.
    fn secret_keys(&self, base: &str) -> Vec<String> {
        let c = self.container;
        let mut keys: Vec<String> = self
            .hosts
            .iter()
            .map(|h| format!("{base}.{c}.{h}"))
            .collect();
        keys.extend(self.hosts.iter().map(|h| format!("{base}.{h}")));
        if let Some(account) = &self.account {
            keys.push(format!("{base}.{c}.{account}"));
            keys.push(format!("{base}.{account}"));
        }
        keys.push(base.to_owned());
        keys
    }

    /// The most specific value of the secret `base`.
    fn secret(&self, base: &str) -> Result<Option<SecretString>> {
        for key in self.secret_keys(base) {
            if let Some(value) = cred(self.config, self.options, &key)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// The value of the account-scoped forms of the plain key `base`, without the bare key.
    fn plain_scoped(&self, base: &str) -> Result<Option<String>> {
        for key in self.account_keys(base) {
            if let Some(value) = text(self.config, self.options, &key)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// The most specific value of the plain key `base` (`AbfsConfiguration.get`/`getEnum`).
    fn plain(&self, base: &str) -> Result<Option<String>> {
        match self.plain_scoped(base)? {
            Some(value) => Ok(Some(value)),
            None => text(self.config, self.options, base),
        }
    }

    /// A required secret: present, or `KRISHIV_INVALID_CONFIG_VALUE` naming `base` and `why`.
    fn required(&self, base: &str, why: &str) -> Result<SecretString> {
        self.secret(base)?
            .ok_or_else(|| invalid_option(base, &format!("required by {why}")).into())
    }
}

/// The hosts `url`'s keys are scoped by: the URL's host and its `dfs`/`blob` twin, or, for a URL
/// without a host, both Azure Storage names of `account`.
fn account_hosts(url: &ParsedUrl, account: Option<&str>) -> Vec<String> {
    match (&url.host, account) {
        (Some(host), _) => {
            let mut hosts = vec![host.clone()];
            let labels: Vec<&str> = host.split('.').collect();
            if let [a, service @ ("dfs" | "blob"), rest @ ..] = labels.as_slice()
                && labels.len() == 5
            {
                let twin = if *service == "dfs" { "blob" } else { "dfs" };
                hosts.push(format!("{a}.{twin}.{}", rest.join(".")));
            }
            hosts
        }
        (None, Some(account)) => vec![
            format!("{account}.dfs.core.windows.net"),
            format!("{account}.blob.core.windows.net"),
        ],
        (None, None) => Vec::new(),
    }
}

/// The explicit credential of the location, if any (see the module docs).
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for an unknown auth type or a missing required key,
/// `KRISHIV_UNSUPPORTED_FEATURE` for `Custom` auth and token provider classes other than the three
/// supported, both naming the key and never a secret.
fn resolve_auth(l: &Lookup<'_>) -> Result<Option<Auth>> {
    match l.plain(AZURE_AUTH_TYPE)?.as_deref() {
        None => {
            if let Some(key) = l.secret(AZURE_ACCOUNT_KEY)? {
                return Ok(Some(Auth::Key(key)));
            }
            if let Some(sas) = sas(l)? {
                return Ok(Some(Auth::Sas(sas)));
            }
            Ok(l.secret(AZURE_OAUTH_ACCESS_TOKEN)?.map(Auth::Bearer))
        }
        Some("SharedKey") => l
            .required(AZURE_ACCOUNT_KEY, "fs.azure.account.auth.type SharedKey")
            .map(|k| Some(Auth::Key(k))),
        Some("SAS") => {
            if let Some(class) = l.plain(AZURE_SAS_TOKEN_PROVIDER_TYPE)? {
                return Err(unsupported(format!(
                    "the SAS token provider class {class} ({AZURE_SAS_TOKEN_PROVIDER_TYPE}); set \
                     {AZURE_SAS_FIXED_TOKEN}"
                )));
            }
            match sas(l)? {
                Some(sas) => Ok(Some(Auth::Sas(sas))),
                None => Err(invalid_option(
                    AZURE_SAS_FIXED_TOKEN,
                    "required by fs.azure.account.auth.type SAS",
                )
                .into()),
            }
        }
        Some("OAuth") => oauth(l).map(Some),
        Some("Custom") => Err(unsupported(format!(
            "{AZURE_AUTH_TYPE} Custom (a custom token provider class); use SharedKey, SAS or OAuth"
        ))),
        Some(_) => Err(invalid_option(AZURE_AUTH_TYPE, "expected SharedKey, SAS or OAuth").into()),
    }
}

/// The SAS: Hadoop's `fs.azure.sas.fixed.token` forms, then Krishiv's `fs.azure.sas` forms.
fn sas(l: &Lookup<'_>) -> Result<Option<SecretString>> {
    match l.secret(AZURE_SAS_FIXED_TOKEN)? {
        Some(sas) => Ok(Some(sas)),
        None => l.secret(AZURE_SAS),
    }
}

/// The `OAuth` credential: a token provider class (account-scoped, else the bare one when the bare
/// auth type is `OAuth` too, as `getTokenProviderClass`), or the static bearer token.
fn oauth(l: &Lookup<'_>) -> Result<Auth> {
    let provider = match l.plain_scoped(AZURE_OAUTH_PROVIDER_TYPE)? {
        Some(class) => Some(class),
        None if text(l.config, l.options, AZURE_AUTH_TYPE)?.as_deref() == Some("OAuth") => {
            text(l.config, l.options, AZURE_OAUTH_PROVIDER_TYPE)?
        }
        None => None,
    };
    let token = l.secret(AZURE_OAUTH_ACCESS_TOKEN)?;
    let class = match (provider, token) {
        (Some(_), Some(_)) => {
            return Err(invalid_option(
                AZURE_OAUTH_ACCESS_TOKEN,
                "set together with fs.azure.account.oauth.provider.type: one credential only",
            )
            .into());
        }
        (None, Some(token)) => return Ok(Auth::Bearer(token)),
        (None, None) => {
            return Err(invalid_option(
                AZURE_OAUTH_PROVIDER_TYPE,
                "fs.azure.account.auth.type OAuth needs a token provider class or \
                 fs.azure.account.oauth2.access.token",
            )
            .into());
        }
        (Some(class), None) => class,
    };
    match class.as_str() {
        CLIENT_CREDS => {
            let endpoint = l.required(AZURE_OAUTH_CLIENT_ENDPOINT, "ClientCredsTokenProvider")?;
            let (authority, tenant) = token_endpoint(endpoint.expose_secret())?;
            Ok(Auth::ClientSecret {
                client_id: l.required(AZURE_OAUTH_CLIENT_ID, "ClientCredsTokenProvider")?,
                secret: l.required(AZURE_OAUTH_CLIENT_SECRET, "ClientCredsTokenProvider")?,
                tenant,
                authority,
            })
        }
        WORKLOAD_IDENTITY => {
            let authority = l.secret(AZURE_OAUTH_MSI_AUTHORITY)?.map_or_else(
                || DEFAULT_AUTHORITY.to_owned(),
                |a| a.expose_secret().to_owned(),
            );
            Ok(Auth::Workload {
                client_id: l.required(AZURE_OAUTH_CLIENT_ID, "WorkloadIdentityTokenProvider")?,
                tenant: l.required(AZURE_OAUTH_MSI_TENANT, "WorkloadIdentityTokenProvider")?,
                token_file: l
                    .secret(AZURE_OAUTH_TOKEN_FILE)?
                    .unwrap_or_else(|| SecretString::new(DEFAULT_TOKEN_FILE.to_owned())),
                authority: checked_authority(AZURE_OAUTH_MSI_AUTHORITY, authority.trim())?,
            })
        }
        MSI => Ok(Auth::Managed {
            client_id: l.secret(AZURE_OAUTH_CLIENT_ID)?,
            endpoint: l.secret(AZURE_OAUTH_MSI_ENDPOINT)?,
        }),
        other => Err(unsupported(format!(
            "the token provider class {other} ({AZURE_OAUTH_PROVIDER_TYPE}); supported are \
             {CLIENT_CREDS}, {WORKLOAD_IDENTITY} and {MSI}, or a static \
             {AZURE_OAUTH_ACCESS_TOKEN}"
        ))),
    }
}

/// An Entra authority (`scheme://host[:port]`, any path kept) that a secret or a federated token
/// may be sent to: https, or http to a loopback host (tests). Trailing `/` dropped (object_store
/// appends `/<tenant>/oauth2/v2.0/token`).
fn checked_authority(key: &str, raw: &str) -> Result<String> {
    let lower = raw.to_ascii_lowercase();
    let host_part = raw.split_once("://").map_or("", |(_, rest)| rest);
    let authority = host_part.split('/').next().unwrap_or_default();
    let secure =
        lower.starts_with("https://") || (lower.starts_with("http://") && loopback_endpoint(raw));
    if !secure || authority.is_empty() || authority.contains('@') {
        return Err(invalid_option(
            key,
            "expected an https:// URL (http:// only to a loopback host)",
        )
        .into());
    }
    Ok(raw.trim_end_matches('/').to_owned())
}

/// The authority and tenant of a Hadoop `fs.azure.account.oauth2.client.endpoint`
/// (`https://login.microsoftonline.com/<tenant>/oauth2/token`, or `.../oauth2/v2.0/token`):
/// object_store asks `<authority>/<tenant>/oauth2/v2.0/token` (`ClientSecretOAuthProvider`).
fn token_endpoint(raw: &str) -> Result<(String, String)> {
    let raw = raw.trim();
    let invalid = || {
        invalid_option(
            AZURE_OAUTH_CLIENT_ENDPOINT,
            "expected https://<authority>/<tenant>/oauth2/token (http:// only to a loopback host)",
        )
    };
    let (scheme, rest) = raw.split_once("://").ok_or_else(invalid)?;
    let (authority, path) = rest.split_once('/').ok_or_else(invalid)?;
    let tenant = path.split('/').next().unwrap_or_default();
    let tenant_ok = !tenant.is_empty()
        && tenant
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_'));
    if !tenant_ok {
        return Err(invalid().into());
    }
    let authority = checked_authority(
        AZURE_OAUTH_CLIENT_ENDPOINT,
        &format!("{scheme}://{authority}"),
    )?;
    Ok((authority, tenant.to_owned()))
}

/// An Azure Blob (or OneLake) store for `url`'s container.
///
/// # Errors
/// As [`resolve_auth`]; `KRISHIV_INVALID_CONFIG_VALUE` for an empty or wrong-kind per-read value
/// (see [`cred`]); `KRISHIV_OBJECT_STORE_REQUEST_FAILED` when the builder rejects the configuration
/// (never a secret).
pub(super) fn azure_store(
    config: &dyn StoreSettings,
    options: &BTreeMap<String, OptionValue>,
    url: &ParsedUrl,
    scope: Option<&Arc<CredentialCache>>,
    clients: Option<&SharedHttpClients>,
) -> Result<Arc<dyn ObjectStore>> {
    let account = url
        .account
        .clone()
        .or(text(config, options, AZURE_ACCOUNT_NAME)?);
    let core = url.host.as_deref().is_none_or(AzureHost::is_core);
    let lookup = Lookup {
        config,
        options,
        container: &url.bucket,
        hosts: account_hosts(url, account.as_deref()),
        account: account.clone().filter(|_| core),
    };
    // Under a credential scope the source is the only credential: session keys do not apply.
    let auth = match scope {
        Some(_) => None,
        None => resolve_auth(&lookup)?,
    };
    let mut builder = match &auth {
        Some(_) => MicrosoftAzureBuilder::new(),
        None => MicrosoftAzureBuilder::from_env(),
    };
    builder = builder.with_container_name(url.bucket.clone());
    if let Some(account) = &account {
        builder = builder.with_account(account);
    }
    let mut endpoint = None;
    if let Some(host) = url.host.as_deref().filter(|h| !AzureHost::is_core(h)) {
        builder = builder.with_use_fabric_endpoint(true);
        if AzureHost::is_private_link(host) {
            endpoint = Some(format!("https://{host}"));
        }
    }
    if let Some(explicit) = text(config, options, AZURE_ENDPOINT)? {
        builder = builder.with_allow_http(allows_http(Some(&explicit)));
        endpoint = Some(explicit);
    }
    if let Some(endpoint) = endpoint {
        builder = builder.with_endpoint(endpoint);
    }
    let provider = scope.map(|c| RefreshingProvider::<AzureCredential>::from_cache(Arc::clone(c)));
    if let Some(provider) = &provider {
        builder = builder.with_credentials(Arc::new(provider.clone()));
    }
    if let Some(auth) = auth {
        builder = builder.with_credential_type(auth.credential_type());
        builder = match auth {
            Auth::Key(key) => builder.with_access_key(key.expose_secret()),
            Auth::Sas(sas) => {
                let pairs =
                    sas_pairs(&sas).map_err(|reason| store_failed(&url.display, 1, &reason))?;
                builder.with_sas_authorization(pairs)
            }
            Auth::Bearer(token) => builder.with_bearer_token_authorization(token.expose_secret()),
            Auth::ClientSecret {
                client_id,
                secret,
                tenant,
                authority,
            } => builder
                .with_client_id(client_id.expose_secret())
                .with_client_secret(secret.expose_secret())
                .with_tenant_id(tenant)
                .with_authority_host(authority),
            Auth::Workload {
                client_id,
                tenant,
                token_file,
                authority,
            } => builder
                .with_client_id(client_id.expose_secret())
                .with_tenant_id(tenant.expose_secret())
                .with_federated_token_file(token_file.expose_secret())
                .with_authority_host(authority),
            Auth::Managed {
                client_id,
                endpoint,
            } => {
                let mut b = builder;
                if let Some(id) = client_id {
                    b = b.with_client_id(id.expose_secret());
                }
                if let Some(endpoint) = endpoint {
                    b = b.with_msi_endpoint(endpoint.expose_secret());
                }
                b
            }
        };
    }
    if let Some(clients) = clients {
        builder = builder.with_http_connector(clients.clone());
    }
    let store = builder
        .build()
        .map(|s| Arc::new(s) as Arc<dyn ObjectStore>)
        .map_err(|e| store_failed(&url.display, 1, &e.to_string()))?;
    Ok(refreshing(store, provider.as_ref()))
}

/// Splits a SAS token (`sv=..&sig=..`) into its query pairs, redacting any
/// complaint: the token itself must never reach an error. A leading `?` (as
/// Azure portals copy it) and surrounding whitespace are ignored. Names and values are
/// percent-decoded, as object_store's `split_sas` decodes them (`azure/builder.rs` l.1353):
/// object_store encodes the pairs again when it adds them to a request.
pub(super) fn sas_pairs(secret: &SecretString) -> Result<Vec<(String, String)>, String> {
    let token = secret.expose_secret().trim();
    let token = token.strip_prefix('?').unwrap_or(token);
    let mut pairs = Vec::new();
    for part in token.split('&') {
        match part.split_once('=') {
            Some((name, value)) if !name.is_empty() && !value.is_empty() => {
                let decoded = percent_decoded(name).zip(percent_decoded(value));
                let Some((name, value)) = decoded else {
                    return Err("the Azure SAS token holds a broken `%` escape".to_owned());
                };
                pairs.push((name, value));
            }
            _ => return Err("the Azure SAS token is not `name=value&...`".to_owned()),
        }
    }
    if pairs.is_empty() {
        return Err("the Azure SAS token is not `name=value&...`".to_owned());
    }
    Ok(pairs)
}

/// `text` with its `%XY` escapes decoded (`+` stays `+`, as in a URL query's percent-decoding);
/// `None` for a broken escape or bytes that are not UTF-8.
fn percent_decoded(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            if !hex.iter().all(u8::is_ascii_hexdigit) {
                return None;
            }
            let hex = std::str::from_utf8(hex).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}
