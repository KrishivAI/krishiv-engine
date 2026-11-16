// SPDX-License-Identifier: Apache-2.0
//! The four read calls of the open-source Unity Catalog REST API (v0.7.0), under `/api/2.1/unity-catalog`:
//! `catalogs`, `schemas?catalog_name=`, `tables?catalog_name=&schema_name=` (paged by `next_page_token`) and
//! `tables/{full_name}`.
//!
//! One deadline covers a call, a paged listing included. A response is read within `max_response_bytes`, its nesting
//! depth is pre-scanned before it is parsed, and a listing is bounded in pages and entries; a page token that repeats
//! or is over 4 KiB ends it. Names that would change the URL (empty, `.`, `..`, with a `/`, `\`, `.` or control
//! character) never reach a request. A 404 on a table is "absent" only when the body's `error_code` says the thing
//! does not exist, so a wrong base URL answering 404 everywhere is a failure, not an empty catalog. Nothing a server
//! says in free text reaches a Krishiv error.

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use krishiv_common::error::KrishivError;
use krishiv_common::secret::SecretString;
use serde_json::Value;
use url::Url;

use super::config::{UnityConfig, UnityLimits, is_name};
use super::error::{UnityError, UnityLimit, quoted};
use super::transport::{Request, Response, Transport};

/// The API root under the server's URI.
const API: [&str; 3] = ["api", "2.1", "unity-catalog"];

/// The longest page token Krishiv follows.
const MAX_PAGE_TOKEN: usize = 4096;

/// One column of a table as UC reports it.
#[derive(Debug, Clone)]
pub struct UcColumn {
    /// The column name.
    pub name: String,
    /// `type_text`, a Spark catalog type string (`int`, `decimal(10,2)`, `array<int>`, ...).
    pub type_text: String,
    /// Whether it may be null (UC's default is true).
    pub nullable: bool,
    /// The ordinal position, if reported.
    pub position: Option<i64>,
    /// The partition index, if the column partitions the table.
    pub partition_index: Option<i64>,
    /// The column comment, if any.
    pub comment: Option<String>,
}

/// One table as UC reports it, parsed and not yet checked against a location policy or a format policy.
#[derive(Debug, Clone)]
pub struct UcTable {
    /// The columns in the order the server sent them.
    pub columns: Vec<UcColumn>,
    /// `data_source_format` (`DELTA`, `PARQUET`, `CSV`, ...).
    pub data_source_format: Option<String>,
    /// `storage_location`.
    pub storage_location: Option<String>,
    /// `table_type` (`MANAGED`, `EXTERNAL`).
    pub table_type: Option<String>,
    /// The string-valued entries of `properties`.
    pub properties: BTreeMap<String, String>,
}

/// The client of one Unity Catalog server.
#[derive(Debug)]
pub struct UnityClient {
    name: String,
    config: UnityConfig,
    transport: Transport,
}

/// Whether a JSON text nests objects and arrays no deeper than `max_depth`. Strings (with their escapes) are skipped.
pub(crate) fn json_depth_within(bytes: &[u8], max_depth: usize) -> bool {
    let (mut depth, mut in_string, mut escaped) = (0_usize, false, false);
    for &b in bytes {
        if in_string {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > max_depth {
                    return false;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    true
}

fn protocol(what: &str) -> UnityError {
    UnityError::Protocol(what.to_owned())
}

impl UnityClient {
    /// A client named `name` (the Krishiv catalog's name, which errors carry) for `config`. Starts the HTTP thread;
    /// nothing connects until the first call.
    ///
    /// # Errors
    /// An invalid declaration, or the client thread could not start.
    pub fn new(name: impl Into<String>, config: UnityConfig) -> Result<Self, KrishivError> {
        let name = name.into();
        config.validate()?;
        let transport = Transport::start(
            config.allow_http,
            config.limits.request_timeout,
            config.limits.max_response_bytes,
        )
        .map_err(|e| e.into_error(&name, &quoted(&config.catalog), &config.limits))?;
        Ok(Self {
            name,
            config,
            transport,
        })
    }

    /// The limits in force.
    #[must_use]
    pub fn limits(&self) -> &UnityLimits {
        &self.config.limits
    }

    /// The UC catalog this client serves.
    #[must_use]
    pub fn uc_catalog(&self) -> &str {
        &self.config.catalog
    }

    fn url(&self, segments: &[&str], query: &[(&str, &str)]) -> Result<Url, UnityError> {
        let mut url = self.config.uri.clone();
        url.path_segments_mut()
            .map_err(|()| protocol("the server URI cannot be a base"))?
            .pop_if_empty()
            .extend(API)
            .extend(segments);
        if !query.is_empty() {
            url.query_pairs_mut().extend_pairs(query);
        }
        Ok(url)
    }

    /// One GET, its body parsed as JSON when the status is 200; any other status is the `Err` it deserves, with
    /// the body's `error_code` kept for the caller's 404 rule.
    fn get_json(&self, url: Url, deadline: Instant) -> Result<Value, UnityError> {
        let Response { status, body } = self.transport.call(
            Request {
                url,
                bearer: self
                    .config
                    .token
                    .as_ref()
                    .map(|t| SecretString::new(t.expose_secret())),
            },
            deadline,
        )?;
        let limits = &self.config.limits;
        if status != 200 {
            // An error body is small and best-effort; a body nested too deeply is simply not read.
            let parsed = if json_depth_within(&body, limits.max_depth) {
                serde_json::from_slice::<Value>(&body).ok()
            } else {
                None
            };
            let text = |key: &str| {
                parsed
                    .as_ref()
                    .and_then(|v| v.get(key))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            };
            return Err(UnityError::Status {
                status,
                code: text("error_code"),
                message: text("message"),
            });
        }
        if !json_depth_within(&body, limits.max_depth) {
            return Err(UnityError::Limit(UnityLimit::Depth));
        }
        serde_json::from_slice(&body).map_err(|e| UnityError::Protocol(format!("not JSON: {e}")))
    }

    /// A listing: `key` names the array in each page; every entry's `name` is returned.
    fn list(
        &self,
        segments: &[&str],
        query: &[(&str, &str)],
        key: &str,
    ) -> Result<Vec<String>, UnityError> {
        let limits = &self.config.limits;
        let deadline = Instant::now() + limits.request_timeout;
        let mut names = Vec::new();
        let mut seen_tokens = HashSet::new();
        let mut token: Option<String> = None;
        for _ in 0..limits.max_pages {
            let mut q: Vec<(&str, &str)> = query.to_vec();
            if let Some(t) = token.as_deref() {
                q.push(("page_token", t));
            }
            let page = self.get_json(self.url(segments, &q)?, deadline)?;
            let entries = match page.get(key) {
                // UC leaves the array out of an empty listing.
                None | Some(Value::Null) => &[][..],
                Some(Value::Array(a)) => a.as_slice(),
                Some(_) => return Err(protocol("a listing that is not an array")),
            };
            if names.len().saturating_add(entries.len()) > limits.max_list_entries {
                return Err(UnityError::Limit(UnityLimit::Entries));
            }
            for entry in entries {
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|n| is_name(n))
                    .ok_or_else(|| protocol("a listed entry without a usable name"))?;
                names.push(name.to_owned());
            }
            match page.get("next_page_token").and_then(Value::as_str) {
                None | Some("") => return Ok(names),
                Some(next) => {
                    if next.len() > MAX_PAGE_TOKEN {
                        return Err(protocol("a page token over 4 KiB"));
                    }
                    if !seen_tokens.insert(next.to_owned()) {
                        return Err(protocol("a listing that repeats its page token"));
                    }
                    token = Some(next.to_owned());
                }
            }
        }
        Err(UnityError::Limit(UnityLimit::Pages))
    }

    /// The catalogs of the server.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_catalogs(&self) -> Result<Vec<String>, KrishivError> {
        self.list(&["catalogs"], &[], "catalogs")
            .map_err(|e| self.fail(e, "(catalogs)"))
    }

    /// The schemas of the UC catalog.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_schemas(&self) -> Result<Vec<String>, KrishivError> {
        self.list(
            &["schemas"],
            &[("catalog_name", &self.config.catalog)],
            "schemas",
        )
        .map_err(|e| self.fail(e, &quoted(&self.config.catalog)))
    }

    /// The tables of `schema`.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_tables(&self, schema: &str) -> Result<Vec<String>, KrishivError> {
        let relation = format!("{}.{}", quoted(&self.config.catalog), quoted(schema));
        if !is_name(schema) {
            return Err(self.fail(protocol("an unusable schema name"), &relation));
        }
        self.list(
            &["tables"],
            &[
                ("catalog_name", &self.config.catalog),
                ("schema_name", schema),
            ],
            "tables",
        )
        .map_err(|e| self.fail(e, &relation))
    }

    /// `schema.table` of the UC catalog; `None` when the server says it does not exist.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn get_table(&self, schema: &str, table: &str) -> Result<Option<UcTable>, KrishivError> {
        if !is_name(schema) || !is_name(table) {
            // No such table can exist, and the name must not shape a URL.
            return Ok(None);
        }
        let relation = format!(
            "{}.{}.{}",
            quoted(&self.config.catalog),
            quoted(schema),
            quoted(table)
        );
        let run = || -> Result<Option<UcTable>, UnityError> {
            let full = format!("{}.{schema}.{table}", self.config.catalog);
            let url = self.url(&["tables", &full], &[])?;
            let deadline = Instant::now() + self.config.limits.request_timeout;
            match self.get_json(url, deadline) {
                Ok(v) => parse_table(&v, self.config.limits.max_columns).map(Some),
                Err(UnityError::Status {
                    status: 404, code, ..
                }) if code.as_deref().is_some_and(is_absence) => Ok(None),
                Err(e) => Err(e),
            }
        };
        run().map_err(|e| self.fail(e, &relation))
    }

    fn fail(&self, e: UnityError, relation: &str) -> KrishivError {
        e.into_error(&self.name, relation, &self.config.limits)
    }
}

/// An `error_code` that says the thing asked for does not exist.
fn is_absence(code: &str) -> bool {
    let c = code.to_ascii_uppercase();
    c.contains("NOT_FOUND") || c.contains("DOES_NOT_EXIST")
}

fn opt_str(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn parse_table(v: &Value, max_columns: usize) -> Result<UcTable, UnityError> {
    if !v.is_object() {
        return Err(protocol("a table that is not an object"));
    }
    let raw = match v.get("columns") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(a)) => a.as_slice(),
        Some(_) => return Err(protocol("columns that are not an array")),
    };
    if raw.len() > max_columns {
        return Err(UnityError::Limit(UnityLimit::Columns));
    }
    let mut columns = Vec::with_capacity(raw.len());
    for c in raw {
        let name = opt_str(c, "name").ok_or_else(|| protocol("a column without a name"))?;
        let type_text =
            opt_str(c, "type_text").ok_or_else(|| protocol("a column without a type_text"))?;
        columns.push(UcColumn {
            name,
            type_text,
            nullable: c.get("nullable").and_then(Value::as_bool).unwrap_or(true),
            position: c.get("position").and_then(Value::as_i64),
            partition_index: c.get("partition_index").and_then(Value::as_i64),
            comment: opt_str(c, "comment"),
        });
    }
    let properties = v
        .get("properties")
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                .collect()
        })
        .unwrap_or_default();
    Ok(UcTable {
        columns,
        data_source_format: opt_str(v, "data_source_format"),
        storage_location: opt_str(v, "storage_location"),
        table_type: opt_str(v, "table_type"),
        properties,
    })
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn depth_counts_brackets_outside_strings_only() {
        assert!(json_depth_within(br#"{"a":[1,{"b":2}]}"#, 3));
        assert!(!json_depth_within(br#"{"a":[1,{"b":2}]}"#, 2));
        assert!(json_depth_within(br#"{"a":"[[[[\"[["}"#, 1));
    }

    #[test]
    fn only_a_does_not_exist_code_is_absence() {
        assert!(is_absence("TABLE_DOES_NOT_EXIST"));
        assert!(is_absence("NOT_FOUND"));
        assert!(!is_absence("INTERNAL_ERROR"));
    }
}
