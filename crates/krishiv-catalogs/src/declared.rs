// SPDX-License-Identifier: Apache-2.0
//! The factories of declared catalogs (feature `session`; design P3-T067 §2.3): the session's
//! `CatalogRegistry` calls them, the binary registers the ones it compiles in.
//!
//! [`IcebergRestFactory`] builds a [`RestClient`] per declared `iceberg-rest` catalog: `auth`
//! `none`, `token` (a reference resolved when the catalog is built), `client-credentials` at the
//! declared token endpoint or `delegated` (the caller's own token or an RFC 8693 exchange of it,
//! design P3-T067 §3.3; each call carries the session's caller), every request on the egress policy client scoped to the catalog's
//! declared hosts with no redirect followed (§5.2). `table` maps a loaded table to the
//! analyzer's metadata and opens its scans through the Iceberg connector (PR-58b), and, with
//! `option.write = true` (default `false`), commits its writes through the catalog (PR-36c
//! part 2).

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue, KrishivUnsupportedFeature};
use krishiv_session::auth::CredentialKind;
use krishiv_session::catalog_decl::{
    CatalogAuth, CatalogDecl, CatalogStorage, CatalogType, DeclMode, Delegation,
};
use krishiv_session::catalog_registry::{
    CallerCatalog, CatalogCaller, CatalogDeps, CatalogFactory, IdentityKey,
};
use krishiv_store::{PolicyConnector, Redirects};
use url::Url;

use crate::iceberg_rest::{
    DelegationLimits, RestAuth, RestCaller, RestClient, RestConfig, SubjectToken,
};
use crate::location::RemoteLocationPolicy;

/// Builds `iceberg-rest` catalogs over Krishiv's own REST client.
#[derive(Debug, Default, Clone, Copy)]
pub struct IcebergRestFactory;

fn key_of(decl: &CatalogDecl, field: &str) -> String {
    format!("krishiv.catalog.{}.{field}", decl.name)
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// `option.write` of `decl` (default `false`: the catalog's tables are read-only; PR-36c part 2).
fn writes(decl: &CatalogDecl) -> Result<bool, KrishivError> {
    match decl.options.get("write").map(|v| v.trim()) {
        None => Ok(false),
        Some(v) if v.eq_ignore_ascii_case("true") => Ok(true),
        Some(v) if v.eq_ignore_ascii_case("false") => Ok(false),
        Some(v) => Err(KrishivInvalidConfigValue {
            value: format!("'{v}'"),
            key: format!("\"{}\"", key_of(decl, "option.write")),
            reason: "expected true or false".to_owned(),
        }
        .into()),
    }
}

impl IcebergRestFactory {
    /// The client configuration `decl` declares; secrets are resolved only with `resolve`
    /// (startup validation checks the shape without reading them again).
    fn config(
        decl: &CatalogDecl,
        deps: &CatalogDeps,
        resolve: bool,
    ) -> Result<RestConfig, KrishivError> {
        writes(decl)?;
        if decl.options.contains_key("prefix") {
            // The client takes the prefix from the catalog's `/v1/config` only.
            return Err(unsupported(format!(
                "option.prefix for the Iceberg REST catalog `{}`",
                decl.name
            )));
        }
        let uri = decl.uri.as_deref().ok_or_else(|| -> KrishivError {
            KrishivInvalidConfigValue {
                value: "''".to_owned(),
                key: format!("\"{}\"", key_of(decl, "uri")),
                reason: "this catalog type needs a uri".to_owned(),
            }
            .into()
        })?;
        let mut config = RestConfig::new(uri)?;
        config.warehouse.clone_from(&decl.warehouse);
        let endpoint = |e: &str| -> Result<Url, KrishivError> {
            Url::parse(e).map_err(|_| -> KrishivError {
                KrishivInvalidConfigValue {
                    value: "(not shown)".to_owned(),
                    key: format!("\"{}\"", key_of(decl, "tokenEndpoint")),
                    reason: "it is not a URL".to_owned(),
                }
                .into()
            })
        };
        let secret = |r: &krishiv_session::auth::SecretRef, field: &str| {
            if resolve {
                r.resolve(&key_of(decl, field))
            } else {
                Ok(krishiv_common::secret::SecretString::new(""))
            }
        };
        config.auth = match &decl.auth {
            CatalogAuth::None => RestAuth::None,
            CatalogAuth::Token(r) => RestAuth::Token(secret(r, "token")?),
            CatalogAuth::ClientCredentials {
                client_id,
                client_secret,
                token_endpoint,
                scope,
                audience,
            } => RestAuth::ClientCredentials {
                client_id: Some(client_id.clone()),
                client_secret: secret(client_secret, "clientSecret")?,
                token_endpoint: Some(endpoint(token_endpoint)?),
                scope: scope.clone(),
                extra: audience
                    .iter()
                    .map(|a| ("audience".to_owned(), a.clone()))
                    .collect(),
            },
            CatalogAuth::Delegated {
                delegation,
                audience,
                token_endpoint,
                scope,
                actor,
            } => RestAuth::Delegated {
                forward: *delegation == Delegation::Forward,
                audience: audience.clone(),
                token_endpoint: token_endpoint.as_deref().map(endpoint).transpose()?,
                scope: scope.clone(),
                actor: match actor {
                    Some(a) => Some((
                        a.client_id.clone(),
                        secret(&a.client_secret, "clientSecret")?,
                    )),
                    None => None,
                },
                limits: DelegationLimits {
                    max_exchanges_per_minute: deps.max_exchanges_per_minute,
                    max_distinct_exchanges_per_minute: deps.max_distinct_exchanges_per_minute,
                },
            },
        };
        config.vended_credentials = decl.storage == CatalogStorage::Vended;
        // The declarations allow `http://` only in embedded mode or with `allowInsecure`.
        config.allow_http = decl.allow_insecure || uri.to_ascii_lowercase().starts_with("http://");
        config.limits.request_timeout = deps.request_timeout;
        config.limits.max_response_bytes =
            usize::try_from(deps.max_response_bytes).unwrap_or(usize::MAX);
        config.egress = Some(
            PolicyConnector::new(
                deps.egress.scope(decl.declared_hosts()),
                Redirects::None,
                deps.proxy.clone(),
            )
            .without_provider_literals(),
        );
        Ok(config)
    }
}

impl IcebergRestFactory {
    /// [`CatalogFactory::build`] with `ttl` as the `loadTable` cache lifetime.
    pub(crate) fn build_with_ttl(
        decl: &CatalogDecl,
        deps: &CatalogDeps,
        ttl: std::time::Duration,
    ) -> Result<Arc<dyn CallerCatalog>, KrishivError> {
        let config = Self::config(decl, deps, true)?;
        // Embedded use reads local tables too (as reads by path are not confined there); a
        // server never reads its own disk for a catalog (`FileFromRemoteCatalog`).
        let policy: Arc<dyn crate::location::LocationPolicy + Send + Sync> =
            if deps.mode == DeclMode::Embedded {
                Arc::new(table::EmbeddedLocations::default())
            } else {
                Arc::new(RemoteLocationPolicy::default())
            };
        let client = RestClient::new(decl.name.clone(), config, policy)?;
        Ok(Arc::new(RestCatalog::new(
            decl,
            Arc::new(client),
            ttl,
            writes(decl)?,
        )))
    }
}

impl CatalogFactory for IcebergRestFactory {
    fn kind(&self) -> CatalogType {
        CatalogType::IcebergRest
    }

    fn validate(&self, decl: &CatalogDecl, deps: &CatalogDeps) -> Result<(), KrishivError> {
        Self::config(decl, deps, false)?.validate()
    }

    fn build(
        &self,
        decl: &CatalogDecl,
        deps: &CatalogDeps,
    ) -> Result<Arc<dyn CallerCatalog>, KrishivError> {
        Self::build_with_ttl(decl, deps, table::METADATA_TTL)
    }
}

fn quoted(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

/// The identity key as the REST token source keys on it: every part length-prefixed, so no two
/// keys encode alike.
fn encode_identity(key: &IdentityKey) -> String {
    let mut out = String::new();
    for part in [
        Some(key.catalog.as_str()),
        key.principal_id.as_deref(),
        key.tenant.as_deref(),
        key.delegated_by.as_deref(),
    ] {
        match part {
            None => out.push('-'),
            Some(p) => out.push_str(&format!("{}:{p}", p.len())),
        }
    }
    out
}

/// The REST client's view of `caller` (T067 §3.2): the server for a caller without a principal.
pub(crate) fn rest_caller(caller: &CatalogCaller) -> RestCaller {
    let (Some(_), Some(limit_key)) = (&caller.identity.principal_id, &caller.identity.limit_key)
    else {
        return RestCaller::server();
    };
    let subject = caller.credential.as_ref().map(|c| {
        SubjectToken::new(
            c.token().clone(),
            c.kind() == CredentialKind::SignedAssertion,
            c.claims().audiences.clone(),
            c.claims_fingerprint(),
            c.expires_at(),
        )
    });
    RestCaller::user(
        encode_identity(&caller.identity),
        limit_key.clone(),
        caller.identity.delegated_by.clone(),
        caller.authenticated_with(),
        subject,
    )
}

mod table;
mod vended;

pub(crate) use table::RestCatalog;

#[cfg(test)]
mod read_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod vended_read_tests;
#[cfg(test)]
mod write_tests;
