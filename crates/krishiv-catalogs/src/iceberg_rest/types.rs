// SPDX-License-Identifier: Apache-2.0
//! The response bodies this client reads, from `rest-catalog-open-api.yaml` at `apache-iceberg-1.12.0`:
//! `CatalogConfig`, `ListNamespacesResponse`, `ListTablesResponse`, `LoadTableResult`, `LoadCredentialsResponse`,
//! `IcebergErrorResponse` and the (deprecated) `OAuthTokenResponse`/`OAuthError`.
//!
//! Parsed by hand from a `serde_json::Value`, as the rest of the workspace does: a required field that is missing or
//! of the wrong type is a protocol error, an unknown field is ignored. Every value of a table's `config` and of a
//! storage credential's `config` becomes a [`SecretString`] when it is moved out of the parsed document (no copy is
//! made), and the caller wipes the body bytes after parsing, so vended keys and tokens are never plain `String`s
//! that outlive the parse (T067 §4.1).

use std::collections::BTreeMap;

use krishiv_common::secret::SecretString;
use serde_json::{Map, Value};

use super::error::RestError;

fn protocol(what: &str) -> RestError {
    RestError::Protocol(what.to_owned())
}

/// Parses a body that already passed the depth check.
pub(crate) fn parse(body: &[u8]) -> Result<Value, RestError> {
    serde_json::from_slice(body).map_err(|e| RestError::Protocol(format!("not JSON: {e}")))
}

fn object(v: Value, what: &str) -> Result<Map<String, Value>, RestError> {
    match v {
        Value::Object(m) => Ok(m),
        _ => Err(protocol(&format!("{what} is not an object"))),
    }
}

fn take_string(
    m: &mut Map<String, Value>,
    key: &str,
    what: &str,
) -> Result<Option<String>, RestError> {
    match m.remove(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(protocol(&format!("`{key}` of {what} is not a string"))),
    }
}

fn string_map(v: Option<Value>, what: &str) -> Result<BTreeMap<String, String>, RestError> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(BTreeMap::new());
    };
    object(v, what)?
        .into_iter()
        .map(|(k, v)| match v {
            Value::String(s) => Ok((k, s)),
            _ => Err(protocol(&format!("a value of {what} is not a string"))),
        })
        .collect()
}

fn secret_map(v: Option<Value>, what: &str) -> Result<BTreeMap<String, SecretString>, RestError> {
    Ok(string_map(v, what)?
        .into_iter()
        .map(|(k, v)| (k, SecretString::from(v)))
        .collect())
}

fn string_list(v: Value, what: &str) -> Result<Vec<String>, RestError> {
    match v {
        Value::Array(items) => items
            .into_iter()
            .map(|i| match i {
                Value::String(s) => Ok(s),
                _ => Err(protocol(&format!("an item of {what} is not a string"))),
            })
            .collect(),
        _ => Err(protocol(&format!("{what} is not a list"))),
    }
}

/// `CatalogConfig`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CatalogConfig {
    /// `defaults`.
    pub defaults: BTreeMap<String, String>,
    /// `overrides`.
    pub overrides: BTreeMap<String, String>,
    /// `endpoints`; `None` when the server sent none (the spec's default set then applies).
    pub endpoints: Option<Vec<String>>,
}

impl CatalogConfig {
    pub(crate) fn from_json(v: Value) -> Result<Self, RestError> {
        let mut m = object(v, "the catalog config")?;
        let endpoints = match m.remove("endpoints") {
            None | Some(Value::Null) => None,
            Some(v) => Some(string_list(v, "`endpoints`")?),
        };
        Ok(Self {
            defaults: string_map(m.remove("defaults"), "`defaults`")?,
            overrides: string_map(m.remove("overrides"), "`overrides`")?,
            endpoints,
        })
    }

    /// A key as the merged configuration has it: an override, else a default.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.overrides
            .get(key)
            .or_else(|| self.defaults.get(key))
            .map(String::as_str)
    }

    /// Whether the server supports `endpoint` (`"GET /v1/{prefix}/..."`). Without an `endpoints` list, only the
    /// spec's default set is assumed (it has no `loadCredentials`).
    #[must_use]
    pub fn supports(&self, endpoint: &str) -> bool {
        match &self.endpoints {
            Some(list) => list.iter().any(|e| e == endpoint),
            None => DEFAULT_ENDPOINTS.contains(&endpoint),
        }
    }
}

/// The endpoints a server that sends no `endpoints` list supports (the read ones of the spec's default set).
const DEFAULT_ENDPOINTS: &[&str] = &[
    "GET /v1/{prefix}/namespaces",
    "GET /v1/{prefix}/namespaces/{namespace}",
    "GET /v1/{prefix}/namespaces/{namespace}/tables",
    "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}",
];

/// One page of a listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Page<T> {
    pub(crate) items: Vec<T>,
    pub(crate) next: Option<String>,
}

fn next_token(m: &mut Map<String, Value>) -> Result<Option<String>, RestError> {
    take_string(m, "next-page-token", "a listing")
}

/// `ListNamespacesResponse`: each namespace is its levels.
pub(crate) fn namespaces_page(v: Value) -> Result<Page<Vec<String>>, RestError> {
    let mut m = object(v, "the namespace listing")?;
    let items = match m.remove("namespaces") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(list)) => list
            .into_iter()
            .map(|ns| string_list(ns, "a namespace"))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(protocol("`namespaces` is not a list")),
    };
    Ok(Page {
        items,
        next: next_token(&mut m)?,
    })
}

/// A `TableIdentifier`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableIdent {
    /// The namespace levels.
    pub namespace: Vec<String>,
    /// The table name.
    pub name: String,
}

/// `ListTablesResponse`.
pub(crate) fn tables_page(v: Value) -> Result<Page<TableIdent>, RestError> {
    let mut m = object(v, "the table listing")?;
    let items = match m.remove("identifiers") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(list)) => list
            .into_iter()
            .map(|id| {
                let mut id = object(id, "a table identifier")?;
                let namespace = string_list(
                    id.remove("namespace")
                        .ok_or_else(|| protocol("a table identifier has no namespace"))?,
                    "a table namespace",
                )?;
                let name = take_string(&mut id, "name", "a table identifier")?
                    .ok_or_else(|| protocol("a table identifier has no name"))?;
                Ok(TableIdent { namespace, name })
            })
            .collect::<Result<_, RestError>>()?,
        Some(_) => return Err(protocol("`identifiers` is not a list")),
    };
    Ok(Page {
        items,
        next: next_token(&mut m)?,
    })
}

/// A `StorageCredential`: a location prefix and the store configuration for it, every value secret-typed.
#[derive(Debug, Clone)]
pub struct StorageCredential {
    /// The location prefix it applies to (checked like any catalog-supplied location).
    pub prefix: String,
    /// The store configuration (`adls.sas-token.<account>`, `s3.access-key-id`, ...).
    pub config: BTreeMap<String, SecretString>,
}

fn storage_credentials(v: Option<Value>) -> Result<Vec<StorageCredential>, RestError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(list)) => list
            .into_iter()
            .map(|c| {
                let mut c = object(c, "a storage credential")?;
                let prefix = take_string(&mut c, "prefix", "a storage credential")?
                    .ok_or_else(|| protocol("a storage credential has no prefix"))?;
                let config = secret_map(
                    Some(
                        c.remove("config")
                            .ok_or_else(|| protocol("a storage credential has no config"))?,
                    ),
                    "a storage credential's config",
                )?;
                Ok(StorageCredential { prefix, config })
            })
            .collect(),
        Some(_) => Err(protocol("`storage-credentials` is not a list")),
    }
}

/// `LoadTableResult`, as far as Krishiv reads it.
#[derive(Debug, Clone)]
pub struct LoadedTable {
    /// `metadata-location` (absent for a staged table).
    pub metadata_location: Option<String>,
    /// `metadata.location`, the table's root, read here so it is checked before anything uses it.
    pub location: String,
    /// `metadata.format-version`.
    pub format_version: u64,
    /// The full `metadata` document (the table metadata JSON), for the Iceberg metadata parser.
    pub metadata: Value,
    /// The table's `config`, every value secret-typed.
    pub config: BTreeMap<String, SecretString>,
    /// `storage-credentials`.
    pub storage_credentials: Vec<StorageCredential>,
    /// Whether `read-restrictions` is present and imposes something (a row filter, or column projections).
    pub restricted: bool,
}

/// Whether a `ReadRestrictions` value imposes anything. Per the spec, a missing or empty object, no projections
/// and an absent, null or literal-true row filter impose nothing; anything else does, including a shape this
/// client does not know (it must fail closed).
fn restricts(v: Option<&Value>) -> bool {
    let Some(v) = v else { return false };
    let Value::Object(m) = v else {
        return !v.is_null();
    };
    m.iter().any(|(k, v)| match (k.as_str(), v) {
        (_, Value::Null) | ("required-row-filter", Value::Bool(true)) => false,
        ("required-column-projections", Value::Array(a)) => !a.is_empty(),
        ("required-row-filter", Value::Object(f)) => {
            !(f.get("type").and_then(Value::as_str) == Some("true") && f.len() == 1)
        }
        _ => true,
    })
}

impl LoadedTable {
    pub(crate) fn from_json(v: Value) -> Result<Self, RestError> {
        let mut m = object(v, "the load-table result")?;
        let restricted = restricts(m.get("read-restrictions"));
        let metadata = m
            .remove("metadata")
            .ok_or_else(|| protocol("the load-table result has no metadata"))?;
        let Value::Object(meta) = &metadata else {
            return Err(protocol("`metadata` is not an object"));
        };
        let location = meta
            .get("location")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol("the table metadata has no location"))?
            .to_owned();
        let format_version = meta
            .get("format-version")
            .and_then(Value::as_u64)
            .ok_or_else(|| protocol("the table metadata has no format-version"))?;
        Ok(Self {
            metadata_location: take_string(&mut m, "metadata-location", "the load-table result")?,
            location,
            format_version,
            config: secret_map(m.remove("config"), "the table config")?,
            storage_credentials: storage_credentials(m.remove("storage-credentials"))?,
            metadata,
            restricted,
        })
    }
}

/// `LoadCredentialsResponse`.
pub(crate) fn credentials(v: Value) -> Result<Vec<StorageCredential>, RestError> {
    let mut m = object(v, "the credentials response")?;
    let list = m
        .remove("storage-credentials")
        .ok_or_else(|| protocol("the credentials response has no storage-credentials"))?;
    storage_credentials(Some(list))
}

/// The `ErrorModel` of an `IcebergErrorResponse`, when the body is one: `(type, message)`.
pub(crate) fn error_model(body: &[u8]) -> (Option<String>, Option<String>) {
    let Ok(Value::Object(mut m)) = serde_json::from_slice::<Value>(body) else {
        return (None, None);
    };
    let Some(Value::Object(mut e)) = m.remove("error") else {
        return (None, None);
    };
    let kind = take_string(&mut e, "type", "an error").ok().flatten();
    let message = take_string(&mut e, "message", "an error").ok().flatten();
    (kind, message)
}

/// An `OAuthTokenResponse`: the token and how long it lives.
#[derive(Debug)]
pub(crate) struct TokenResponse {
    pub(crate) access_token: SecretString,
    pub(crate) expires_in: Option<u64>,
}

pub(crate) fn token_response(v: Value) -> Result<TokenResponse, RestError> {
    let mut m = object(v, "the token response")?;
    let access_token = take_string(&mut m, "access_token", "the token response")?
        .map(SecretString::from)
        .ok_or_else(|| protocol("the token response has no access_token"))?;
    let token_type = take_string(&mut m, "token_type", "the token response")?
        .ok_or_else(|| protocol("the token response has no token_type"))?;
    if !token_type.eq_ignore_ascii_case("bearer") {
        return Err(protocol("the token response is not a bearer token"));
    }
    let expires_in = match m.remove("expires_in") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .ok_or_else(|| protocol("`expires_in` is not a non-negative integer"))?,
        ),
    };
    Ok(TokenResponse {
        access_token,
        expires_in,
    })
}

/// The OAuth `error` code of a token endpoint's error body (never `error_description`).
pub(crate) fn oauth_error(body: &[u8]) -> Option<String> {
    let Ok(Value::Object(mut m)) = serde_json::from_slice::<Value>(body) else {
        return None;
    };
    take_string(&mut m, "error", "an OAuth error")
        .ok()
        .flatten()
        // A code is a short ASCII token; anything else is not one.
        .filter(|c| c.len() <= 64 && c.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn config_merges_overrides_over_defaults() {
        let c = CatalogConfig::from_json(json!({
            "defaults": {"prefix": "d", "clients": "4"},
            "overrides": {"prefix": "o"},
        }))
        .unwrap();
        assert_eq!(c.get("prefix"), Some("o"));
        assert_eq!(c.get("clients"), Some("4"));
        assert!(c.supports("GET /v1/{prefix}/namespaces/{namespace}/tables/{table}"));
        assert!(!c.supports("GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"));
        let c = CatalogConfig::from_json(json!({
            "defaults": {}, "overrides": {},
            "endpoints": ["GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"],
        }))
        .unwrap();
        assert!(c.supports("GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"));
        assert!(!c.supports("GET /v1/{prefix}/namespaces"));
        assert!(CatalogConfig::from_json(json!({"defaults": {"a": 1}})).is_err());
        assert!(CatalogConfig::from_json(json!([])).is_err());
    }

    #[test]
    fn listings_parse_and_refuse_wrong_shapes() {
        let p = namespaces_page(json!({"namespaces": [["a"], ["a", "b"]], "next-page-token": "n"}))
            .unwrap();
        assert_eq!(
            p.items,
            vec![vec!["a".to_owned()], vec!["a".to_owned(), "b".to_owned()]]
        );
        assert_eq!(p.next.as_deref(), Some("n"));
        assert!(namespaces_page(json!({"namespaces": [[1]]})).is_err());
        let p = tables_page(
            json!({"identifiers": [{"namespace": ["a"], "name": "t"}], "next-page-token": null}),
        )
        .unwrap();
        assert_eq!(p.items[0].name, "t");
        assert_eq!(p.next, None);
        assert!(tables_page(json!({"identifiers": [{"namespace": ["a"]}]})).is_err());
        assert!(tables_page(json!({"identifiers": {}})).is_err());
    }

    #[test]
    fn load_table_keeps_config_credentials_and_restrictions() {
        let t = LoadedTable::from_json(json!({
            "metadata-location": "s3://b/t/metadata/v1.json",
            "metadata": {"format-version": 2, "location": "s3://b/t", "table-uuid": "u"},
            "config": {"token": "tok", "s3.access-key-id": "AK"},
            "storage-credentials": [{"prefix": "s3://b/t", "config": {"s3.secret-access-key": "SK"}}],
        }))
        .unwrap();
        assert_eq!(t.location, "s3://b/t");
        assert_eq!(t.format_version, 2);
        assert_eq!(t.config["token"].expose_secret(), "tok");
        assert_eq!(
            t.storage_credentials[0].config["s3.secret-access-key"].expose_secret(),
            "SK"
        );
        assert!(!format!("{t:?}").contains("SK"));
        assert!(!t.restricted);
        assert!(LoadedTable::from_json(json!({"metadata": {"format-version": 2}})).is_err());
        assert!(LoadedTable::from_json(json!({"metadata-location": "x"})).is_err());
    }

    #[test]
    fn restrictions_fail_closed() {
        for none in [
            json!(null),
            json!({}),
            json!({"required-column-projections": []}),
            json!({"required-row-filter": null}),
            json!({"required-row-filter": true}),
            json!({"required-row-filter": {"type": "true"}}),
        ] {
            assert!(!restricts(Some(&none)), "{none}");
        }
        for some in [
            json!({"required-column-projections": [{"field-id": 4, "action": "show-last-4"}]}),
            json!({"required-row-filter": {"type": "eq", "left": {}, "right": {}}}),
            json!({"something-new": 1}),
            json!("x"),
        ] {
            assert!(restricts(Some(&some)), "{some}");
        }
    }

    #[test]
    fn errors_and_tokens() {
        let (k, m) =
            error_model(br#"{"error":{"message":"m","type":"NoSuchTableException","code":404}}"#);
        assert_eq!(
            (k.as_deref(), m.as_deref()),
            (Some("NoSuchTableException"), Some("m"))
        );
        assert_eq!(error_model(b"<html>"), (None, None));
        let t =
            token_response(json!({"access_token": "a", "token_type": "Bearer", "expires_in": 60}))
                .unwrap();
        assert_eq!(
            (t.access_token.expose_secret(), t.expires_in),
            ("a", Some(60))
        );
        assert!(token_response(json!({"access_token": "a", "token_type": "mac"})).is_err());
        assert!(token_response(json!({"token_type": "bearer"})).is_err());
        assert_eq!(
            oauth_error(br#"{"error":"invalid_client","error_description":"x"}"#).as_deref(),
            Some("invalid_client")
        );
        assert_eq!(oauth_error(br#"{"error":"has a token eyJ..."}"#), None);
    }
}
