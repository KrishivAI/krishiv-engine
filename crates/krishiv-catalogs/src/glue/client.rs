// SPDX-License-Identifier: Apache-2.0
//! The three read calls of the AWS Glue Data Catalog API (JSON 1.1, `X-Amz-Target: AWSGlue.<Action>`): `GetDatabases`,
//! `GetTables` (both paged by `NextToken`) and `GetTable`. Every request is signed with Signature Version 4
//! ([`super::sigv4`]).
//!
//! One deadline covers a call, a paged listing included. A response is read within `max_response_bytes`, its nesting
//! depth is pre-scanned before it is parsed, and a listing is bounded in pages and entries; a token that repeats or is
//! over 4 KiB ends it. Only `EntityNotFoundException` on `GetTable` is "absent", so a wrong endpoint answering
//! errors everywhere is a failure, not an empty catalog. Nothing a server says in free text reaches a Krishiv error.

use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use krishiv_common::error::KrishivError;
use krishiv_common::secret::SecretString;
use serde_json::{Value, json};

use super::config::{GlueConfig, GlueLimits, is_name};
use super::error::{GlueError, GlueLimit, quoted};
use super::sigv4::{Signing, now_amz_date};
use super::transport::{Request, Response, Transport};

/// The longest token Krishiv follows.
const MAX_PAGE_TOKEN: usize = 4096;

/// The longest column name accepted (the type string is bounded by `MAX_DECLARED_LEN` in `catalog.rs`).
const MAX_COLUMN_TEXT: usize = 4096;

/// Entries asked for per page (Glue's own maximum for `GetTables` and `GetDatabases`).
const PAGE_SIZE: u32 = 100;

/// One column of a table as Glue reports it.
#[derive(Debug, Clone)]
pub struct GlueColumn {
    /// The column name.
    pub name: String,
    /// `Type`, a Hive type string (`int`, `decimal(10,2)`, `array<int>`, ...).
    pub type_text: String,
    /// The column comment, if any.
    pub comment: Option<String>,
}

/// One table as Glue reports it, parsed and not yet checked against a location policy or a format policy.
#[derive(Debug, Clone)]
pub struct GlueTableRaw {
    /// `StorageDescriptor.Columns`, in order.
    pub columns: Vec<GlueColumn>,
    /// `PartitionKeys`, in order.
    pub partition_keys: Vec<GlueColumn>,
    /// `StorageDescriptor.Location`.
    pub location: Option<String>,
    /// `StorageDescriptor.InputFormat`.
    pub input_format: Option<String>,
    /// `StorageDescriptor.SerdeInfo.SerializationLibrary`.
    pub serde: Option<String>,
    /// `TableType` (`EXTERNAL_TABLE`, `MANAGED_TABLE`, `VIRTUAL_VIEW`, ...).
    pub table_type: Option<String>,
    /// The string-valued entries of `Parameters`.
    pub parameters: BTreeMap<String, String>,
}

/// The client of one Glue Data Catalog.
#[derive(Debug)]
pub struct GlueClient {
    name: String,
    config: GlueConfig,
    host: String,
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

fn protocol(what: &str) -> GlueError {
    GlueError::Protocol(what.to_owned())
}

impl GlueClient {
    /// A client named `name` (the Krishiv catalog's name, which errors carry) for `config`. Starts the HTTP thread;
    /// nothing connects until the first call.
    ///
    /// # Errors
    /// An invalid declaration, or the client thread could not start.
    pub fn new(name: impl Into<String>, config: GlueConfig) -> Result<Self, KrishivError> {
        let name = name.into();
        config.validate()?;
        let relation = "(glue)".to_owned();
        let host = match (config.endpoint.host_str(), config.endpoint.port()) {
            (Some(h), Some(p)) => format!("{h}:{p}"),
            (Some(h), None) => h.to_owned(),
            (None, _) => String::new(),
        };
        let transport = Transport::start(
            &config.connector(),
            config.allow_http,
            config.limits.request_timeout,
            config.limits.max_response_bytes,
        )
        .map_err(|e| e.into_error(&name, &relation, &config.limits))?;
        Ok(Self {
            name,
            config,
            host,
            transport,
        })
    }

    /// The limits in force.
    #[must_use]
    pub fn limits(&self) -> &GlueLimits {
        &self.config.limits
    }

    /// One signed call of `action` with `body`; the answer is parsed as JSON when the status is 200, and any other
    /// status is the `Err` it deserves, with the body's error type kept for the caller's absence rule.
    fn call(&self, action: &str, body: &Value, deadline: Instant) -> Result<Value, GlueError> {
        let body = serde_json::to_vec(body).map_err(|e| GlueError::Protocol(e.to_string()))?;
        let date = now_amz_date();
        let target = format!("AWSGlue.{action}");
        let content_type = "application/x-amz-json-1.1";
        let token = self
            .config
            .session_token
            .as_ref()
            .map(|t| t.expose_secret().to_owned());
        let mut signed: Vec<(&str, &str)> =
            vec![("content-type", content_type), ("x-amz-target", &target)];
        if let Some(t) = &token {
            signed.push(("x-amz-security-token", t));
        }
        let authorization = Signing {
            method: "POST",
            host: &self.host,
            service: "glue",
            region: &self.config.region,
            date: &date,
            headers: &signed,
            body: &body,
            access_key_id: &self.config.access_key_id,
            secret_access_key: self.config.secret_access_key.expose_secret(),
        }
        .authorization();
        let mut headers = vec![
            ("authorization", SecretString::new(&authorization)),
            ("content-type", SecretString::new(content_type)),
            ("x-amz-date", SecretString::new(&date)),
            ("x-amz-target", SecretString::new(&target)),
        ];
        if let Some(t) = &token {
            headers.push(("x-amz-security-token", SecretString::new(t)));
        }
        let Response {
            status,
            body,
            error_type,
        } = self.transport.call(
            Request {
                url: self.config.endpoint.clone(),
                headers,
                body,
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
            let text = |keys: &[&str]| {
                keys.iter()
                    .find_map(|k| parsed.as_ref()?.get(k)?.as_str())
                    .map(str::to_owned)
            };
            return Err(GlueError::Status {
                status,
                // `com.amazonaws.glue#EntityNotFoundException`, `EntityNotFoundException` or the header form
                // `EntityNotFoundException:http://internal.amazon.com/...`.
                code: text(&["__type", "code"]).or(error_type).map(|c| {
                    let c = c.split(':').next().unwrap_or_default();
                    c.rsplit('#').next().unwrap_or_default().to_owned()
                }),
                message: text(&["Message", "message"]),
            });
        }
        if !json_depth_within(&body, limits.max_depth) {
            return Err(GlueError::Limit(GlueLimit::Depth));
        }
        serde_json::from_slice(&body).map_err(|e| GlueError::Protocol(format!("not JSON: {e}")))
    }

    /// A listing: `key` names the array in each page, whose entries each carry a `Name`.
    fn list(&self, action: &str, request: &Value, key: &str) -> Result<Vec<String>, GlueError> {
        let limits = &self.config.limits;
        let deadline = deadline_in(limits.request_timeout)?;
        let mut names = Vec::new();
        let mut seen_tokens = HashSet::new();
        let mut token: Option<String> = None;
        for _ in 0..limits.max_pages {
            let mut body = request.clone();
            body["MaxResults"] = json!(PAGE_SIZE);
            if let Some(id) = &self.config.catalog_id {
                body["CatalogId"] = json!(id);
            }
            if let Some(t) = &token {
                body["NextToken"] = json!(t);
            }
            let page = self.call(action, &body, deadline)?;
            if !page.is_object() {
                return Err(protocol("a listing that is not a JSON object"));
            }
            let entries = match page.get(key) {
                None | Some(Value::Null) => &[][..],
                Some(Value::Array(a)) => a.as_slice(),
                Some(_) => return Err(protocol("a listing that is not an array")),
            };
            if names.len().saturating_add(entries.len()) > limits.max_list_entries {
                return Err(GlueError::Limit(GlueLimit::Entries));
            }
            for entry in entries {
                let name = entry
                    .get("Name")
                    .and_then(Value::as_str)
                    .filter(|n| is_name(n))
                    .ok_or_else(|| protocol("a listed entry without a usable name"))?;
                names.push(name.to_owned());
            }
            match page.get("NextToken") {
                None | Some(Value::Null) => return within(deadline).map(|()| names),
                Some(Value::String(t)) if t.is_empty() => return within(deadline).map(|()| names),
                Some(Value::String(t)) => {
                    if t.len() > MAX_PAGE_TOKEN {
                        return Err(protocol("a page token over 4 KiB"));
                    }
                    if !seen_tokens.insert(t.clone()) {
                        return Err(protocol("a listing that repeats its page token"));
                    }
                    token = Some(t.clone());
                }
                Some(_) => return Err(protocol("a page token that is not a string")),
            }
        }
        Err(GlueError::Limit(GlueLimit::Pages))
    }

    /// The databases of the Glue catalog.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_databases(&self) -> Result<Vec<String>, KrishivError> {
        self.list("GetDatabases", &json!({}), "DatabaseList")
            .map_err(|e| self.fail(e, "(databases)"))
    }

    /// The tables of `database`.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_tables(&self, database: &str) -> Result<Vec<String>, KrishivError> {
        let relation = quoted(database);
        if !is_name(database) {
            return Err(self.fail(protocol("an unusable database name"), &relation));
        }
        self.list(
            "GetTables",
            &json!({ "DatabaseName": database }),
            "TableList",
        )
        .map_err(|e| self.fail(e, &relation))
    }

    /// `database.table`; `None` when Glue says it does not exist.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn get_table(
        &self,
        database: &str,
        table: &str,
    ) -> Result<Option<GlueTableRaw>, KrishivError> {
        if !is_name(database) || !is_name(table) {
            return Ok(None);
        }
        let relation = format!("{}.{}", quoted(database), quoted(table));
        let run = || -> Result<Option<GlueTableRaw>, GlueError> {
            let mut body = json!({ "DatabaseName": database, "Name": table });
            if let Some(id) = &self.config.catalog_id {
                body["CatalogId"] = json!(id);
            }
            let deadline = deadline_in(self.config.limits.request_timeout)?;
            match self.call("GetTable", &body, deadline) {
                Ok(v) => {
                    let table = parse_table(&v, self.config.limits.max_columns)?;
                    within(deadline)?;
                    Ok(Some(table))
                }
                Err(GlueError::Status {
                    status: 400,
                    code: Some(code),
                    ..
                }) if code == "EntityNotFoundException" => Ok(None),
                Err(e) => Err(e),
            }
        };
        run().map_err(|e| self.fail(e, &relation))
    }

    fn fail(&self, e: GlueError, relation: &str) -> KrishivError {
        e.into_error(&self.name, relation, &self.config.limits)
    }
}

/// `now + timeout`, or a timeout when that cannot be represented.
fn deadline_in(timeout: Duration) -> Result<Instant, GlueError> {
    Instant::now()
        .checked_add(timeout)
        .ok_or(GlueError::Timeout)
}

/// A call that finished after its deadline timed out, whatever it returned.
fn within(deadline: Instant) -> Result<(), GlueError> {
    if Instant::now() > deadline {
        Err(GlueError::Timeout)
    } else {
        Ok(())
    }
}

fn opt_str(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn parse_columns(v: Option<&Value>, max: usize) -> Result<Vec<GlueColumn>, GlueError> {
    let raw = match v {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(a)) => a.as_slice(),
        Some(_) => return Err(protocol("columns that are not an array")),
    };
    if raw.len() > max {
        return Err(GlueError::Limit(GlueLimit::Columns));
    }
    raw.iter()
        .map(|c| {
            let name = opt_str(c, "Name")
                .filter(|n| {
                    !n.is_empty() && n.len() <= MAX_COLUMN_TEXT && !n.chars().any(char::is_control)
                })
                .ok_or_else(|| protocol("a column without a usable name"))?;
            let type_text =
                opt_str(c, "Type").ok_or_else(|| protocol("a column without a type"))?;
            Ok(GlueColumn {
                name,
                type_text,
                comment: opt_str(c, "Comment"),
            })
        })
        .collect()
}

fn parse_table(v: &Value, max_columns: usize) -> Result<GlueTableRaw, GlueError> {
    let t = v
        .get("Table")
        .filter(|t| t.is_object())
        .ok_or_else(|| protocol("an answer without a table"))?;
    let sd = t.get("StorageDescriptor").filter(|s| s.is_object());
    let columns = parse_columns(sd.and_then(|s| s.get("Columns")), max_columns)?;
    let partition_keys = parse_columns(t.get("PartitionKeys"), max_columns)?;
    if columns.len().saturating_add(partition_keys.len()) > max_columns {
        return Err(GlueError::Limit(GlueLimit::Columns));
    }
    // A Spark schema marker that is not a string would vanish below and let the placeholder columns be read.
    if let Some(o) = t.get("Parameters").and_then(Value::as_object)
        && o.iter().any(|(k, v)| {
            (k == "spark.sql.sources.schema" || k.starts_with("spark.sql.sources.schema."))
                && !v.is_string()
        })
    {
        return Err(protocol("a Spark schema property that is not a string"));
    }
    let parameters = t
        .get("Parameters")
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                .collect()
        })
        .unwrap_or_default();
    Ok(GlueTableRaw {
        columns,
        partition_keys,
        location: sd.and_then(|s| opt_str(s, "Location")),
        input_format: sd.and_then(|s| opt_str(s, "InputFormat")),
        serde: sd
            .and_then(|s| s.get("SerdeInfo"))
            .and_then(|s| opt_str(s, "SerializationLibrary")),
        table_type: opt_str(t, "TableType"),
        parameters,
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
}
