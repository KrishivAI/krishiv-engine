// SPDX-License-Identifier: Apache-2.0
//! The factories of declared catalogs (feature `session`; design P3-T067 §2.3): the session's
//! `CatalogRegistry` calls them, the binary registers the ones it compiles in.
//!
//! [`IcebergRestFactory`] builds a [`RestClient`] per declared `iceberg-rest` catalog: `auth`
//! `none`, `token` (a reference resolved when the catalog is built) or `client-credentials` at the
//! declared token endpoint, every request on the egress policy client scoped to the catalog's
//! declared hosts with no redirect followed (§5.2). The catalog resolves names: a table the
//! catalog does not have is absent; a table it has is refused for now with
//! `KRISHIV_UNSUPPORTED_FEATURE`, since the metadata mapping and the table provider over the
//! client (through the Iceberg connector) are a later step. Nothing is ever read through an
//! unchecked path.

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue, KrishivUnsupportedFeature};
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{BoxFuture, Catalog, NameForm, TableIdent, TableMeta};
use krishiv_session::catalog_decl::{CatalogAuth, CatalogDecl, CatalogStorage, CatalogType};
use krishiv_session::catalog_registry::{CatalogDeps, CatalogFactory};
use krishiv_store::{PolicyConnector, Redirects};
use url::Url;

use crate::iceberg_rest::{RestAuth, RestClient, RestConfig};
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

impl IcebergRestFactory {
    /// The client configuration `decl` declares; secrets are resolved only with `resolve`
    /// (startup validation checks the shape without reading them again).
    fn config(
        decl: &CatalogDecl,
        deps: &CatalogDeps,
        resolve: bool,
    ) -> Result<RestConfig, KrishivError> {
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
                token_endpoint: Some(Url::parse(token_endpoint).map_err(|_| -> KrishivError {
                    KrishivInvalidConfigValue {
                        value: "(not shown)".to_owned(),
                        key: format!("\"{}\"", key_of(decl, "tokenEndpoint")),
                        reason: "it is not a URL".to_owned(),
                    }
                    .into()
                })?),
                scope: scope.clone(),
                extra: audience
                    .iter()
                    .map(|a| ("audience".to_owned(), a.clone()))
                    .collect(),
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
    ) -> Result<Arc<dyn Catalog>, KrishivError> {
        let config = Self::config(decl, deps, true)?;
        let client = RestClient::new(
            decl.name.clone(),
            config,
            Arc::new(RemoteLocationPolicy::default()),
        )?;
        Ok(Arc::new(RestCatalog {
            client: Arc::new(client),
        }))
    }
}

/// A declared Iceberg REST catalog as the analyzer sees it (name resolution only, see the module
/// docs).
#[derive(Debug)]
struct RestCatalog {
    client: Arc<RestClient>,
}

fn quoted(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

impl Catalog for RestCatalog {
    fn name(&self) -> &str {
        self.client.name()
    }

    fn epoch(&self) -> u64 {
        0
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            // No temporary relation lives here, and a table needs a namespace.
            if form.temporary_only || id.namespace.is_empty() {
                return Ok(None);
            }
            let client = Arc::clone(&self.client);
            let (namespace, name) = (id.namespace.clone(), id.name.clone());
            // The client blocks until its deadline: never on a compute worker. Outside a tokio
            // runtime (an embedded host on another executor) there is no blocking pool, and the
            // call runs on the caller's thread, which is blocking already.
            let loaded = match tokio::runtime::Handle::try_current() {
                Ok(handle) => handle
                    .spawn_blocking(move || client.load_table(&namespace, &name))
                    .await
                    .map_err(|_| {
                        crate::iceberg_rest::RestError::Transport(
                            "the catalog call did not finish".to_owned(),
                        )
                        .into_error(
                            self.client.name(),
                            "the table",
                            &crate::iceberg_rest::RestLimits::default(),
                        )
                    })??,
                Err(_) => client.load_table(&namespace, &name)?,
            };
            let Some(_table) = loaded else {
                return Ok(None);
            };
            let relation: Vec<String> = std::iter::once(id.catalog.as_str())
                .chain(id.namespace.iter().map(String::as_str))
                .chain(std::iter::once(id.name.as_str()))
                .map(quoted)
                .collect();
            let what = if version.is_some() {
                "time travel on"
            } else {
                "reading"
            };
            Err(unsupported(format!(
                "{what} the Iceberg REST table {} (the table provider over the declared catalog \
                 is not in this build yet)",
                relation.join(".")
            )))
        })
    }
}

#[cfg(test)]
mod tests;
