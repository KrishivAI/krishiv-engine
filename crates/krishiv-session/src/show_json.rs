// SPDX-License-Identifier: Apache-2.0
//! `SHOW TABLES | TABLE EXTENDED | PARTITIONS ... AS JSON` (Spark 4.3): the plain command runs, and its rows
//! become one `json_metadata` string. Shapes and key order are Spark 4.3.0's (a real run): `{"tables":[{"name",
//! "namespace":[..],"isTemporary"}]}`, the extended form adding `catalog` and `type` after `name`, and
//! `{"partitions":[{"<column>":"<value>"}]}`.

use arrow_array::cast::AsArray;
use arrow_cast::cast;
use arrow_schema::DataType;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_sql::ast::{Show, ShowKind};
use serde_json::Value;

use crate::analyze::command_schema as cols;
use crate::dataframe::DataFrame;
use crate::session::Session;
use crate::store::CatalogStore;

impl Session {
    /// The `AS JSON` form of `show` (`show.as_json`).
    pub(crate) async fn exec_show_json(
        &self,
        show: &Show,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        // `SHOW TABLE EXTENDED ... AS JSON` lists names like `SHOW TABLES` and builds no information text, so the
        // listing is what runs (and no refusal of the text form applies).
        let listed = match &show.kind {
            ShowKind::TableExtended {
                namespace,
                pattern,
                span,
                ..
            } => {
                // The JSON names the session catalog and a stored table type; the `system` tables and a
                // declared catalog's tables are neither, so their metadata would be made up (the listing
                // carries no catalog identity or table type). Refused rather than answered wrong.
                let outside_session_catalog = match namespace {
                    Some(n) => {
                        let env = self.query_env();
                        let parts = crate::access::ddl_parts(n, &env)?;
                        match parts.as_slice() {
                            [one] => crate::system::is_system(one),
                            [catalog, _] => {
                                crate::system::is_system(catalog)
                                    || self.declared_catalogs().canonical(catalog).is_some()
                            }
                            _ => false,
                        }
                    }
                    None => {
                        let current = self.state.catalog.current_catalog();
                        crate::system::is_system(&current)
                            || self.declared_catalogs().canonical(&current).is_some()
                    }
                };
                if outside_session_catalog {
                    return Err(KrishivUnsupportedFeature {
                        feature: "SHOW TABLE EXTENDED ... AS JSON over the system catalog or a declared catalog"
                            .to_owned(),
                    }
                    .into());
                }
                ShowKind::Tables {
                    namespace: namespace.clone(),
                    pattern: Some(pattern.clone()),
                    span: *span,
                }
            }
            kind => kind.clone(),
        };
        // A declared catalog's listing carries its namespace as one dotted label, so a multipart namespace
        // (`lake.a.b`) cannot be written back as `["a","b"]`: refused rather than answered wrong.
        if let ShowKind::Tables {
            namespace: Some(n), ..
        } = &listed
        {
            let parts = crate::access::ddl_parts(n, &self.query_env())?;
            if let [catalog, ns @ ..] = parts.as_slice()
                && ns.len() > 1
                && self.declared_catalogs().canonical(catalog).is_some()
            {
                return Err(KrishivUnsupportedFeature {
                    feature:
                        "SHOW TABLES ... AS JSON over a multipart namespace of a declared catalog"
                            .to_owned(),
                }
                .into());
            }
        }
        let plain = Show {
            kind: listed,
            as_json: false,
            span: show.span,
        };
        let rows = string_rows(Box::pin(self.exec_show(&plain, original)).await?).await?;
        let (key, items): (&str, Vec<String>) = match &show.kind {
            ShowKind::Tables { .. } => ("tables", rows.iter().map(|r| table(r, None)).collect()),
            ShowKind::TableExtended { .. } => (
                "tables",
                rows.iter()
                    .map(|r| table(r, Some(self.table_type(r))))
                    .collect(),
            ),
            _ => ("partitions", rows.iter().map(|r| partition(r)).collect()),
        };
        let doc = format!("{{\"{key}\":[{}]}}", items.join(","));
        self.string_frame(cols::SHOW_JSON, vec![vec![Some(doc)]])
            .await
    }

    /// Spark's `tableType` of a listed row: `VIEW` for a temporary or permanent view, else `MANAGED` or `EXTERNAL`.
    fn table_type(&self, row: &[String]) -> &'static str {
        let (ns, name) = (
            row.first().map_or("", String::as_str),
            row.get(1).map_or("", String::as_str),
        );
        if row.get(2).is_some_and(|t| t == "true") {
            return "VIEW";
        }
        match self.state.catalog.store.table(ns, name) {
            Some(t) if t.view.is_some() => "VIEW",
            Some(t) if !t.managed => "EXTERNAL",
            _ => "MANAGED",
        }
    }
}

/// Every cell of `df` as text.
async fn string_rows(df: DataFrame) -> Result<Vec<Vec<String>>, KrishivError> {
    let mut rows = Vec::new();
    for b in df.collect().await? {
        let cols: Vec<_> = b
            .columns()
            .iter()
            .map(|c| cast(c, &DataType::Utf8).map_err(|e| crate::session::internal(e.to_string())))
            .collect::<Result<_, _>>()?;
        for i in 0..b.num_rows() {
            rows.push(
                cols.iter()
                    .map(|c| c.as_string::<i32>().value(i).to_owned())
                    .collect(),
            );
        }
    }
    Ok(rows)
}

/// A JSON string literal.
fn js(s: &str) -> String {
    Value::String(s.to_owned()).to_string()
}

/// One `SHOW TABLES` row (`namespace`, `tableName`, `isTemporary`) as an object whose keys are in Spark's order
/// (serde_json's map would sort them). `extended` is the table type of `SHOW TABLE EXTENDED`, which also writes
/// the `catalog` (the session catalog's, temporary views included).
fn table(row: &[String], extended: Option<&str>) -> String {
    let temporary = row.get(2).is_some_and(|t| t == "true");
    let namespace = row
        .first()
        .filter(|n| !n.is_empty())
        .map_or(String::new(), |n| js(n));
    let mut fields = vec![format!(
        "\"name\":{}",
        js(row.get(1).map_or("", String::as_str))
    )];
    if extended.is_some() {
        fields.push(format!("\"catalog\":{}", js(crate::catalog::CATALOG)));
    }
    fields.push(format!("\"namespace\":[{namespace}]"));
    if let Some(ty) = extended {
        fields.push(format!("\"type\":{}", js(ty)));
    }
    fields.push(format!("\"isTemporary\":{temporary}"));
    format!("{{{}}}", fields.join(","))
}

/// One `SHOW PARTITIONS` row, `p=7/q=a`, as an object in column order. Spark splits the escaped name on `/` and
/// the first `=` and keeps the `%XX` escapes (`ShowPartitionsJsonCommand`); a segment without `=` is a key with
/// an empty value.
fn partition(row: &[String]) -> String {
    let fields: Vec<String> = row
        .first()
        .map_or("", String::as_str)
        .split('/')
        .map(|part| {
            let (k, v) = part.split_once('=').unwrap_or((part, ""));
            format!("{}:{}", js(k), js(v))
        })
        .collect();
    format!("{{{}}}", fields.join(","))
}
