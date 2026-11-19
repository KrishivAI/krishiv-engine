// SPDX-License-Identifier: Apache-2.0
//! `SHOW TABLES | TABLE EXTENDED | PARTITIONS ... AS JSON` (Spark 4.3): the plain command runs, and its rows
//! become one `json_metadata` string. Shapes and key order are Spark 4.3.0's (a real run): `{"tables":[{"name",
//! "namespace":[..],"isTemporary"}]}`, the extended form adding `catalog` and `type` after `name`, and
//! `{"partitions":[{"<column>":"<value>"}]}`.

use arrow_array::cast::AsArray;
use arrow_cast::cast;
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_sql::ast::{Show, ShowKind};
use serde_json::Value;

use crate::analyze::command_schema as cols;
use crate::dataframe::DataFrame;
use crate::session::Session;

impl Session {
    /// The `AS JSON` form of `show` (`show.as_json`).
    pub(crate) async fn exec_show_json(
        &self,
        show: &Show,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let plain = Show {
            kind: show.kind.clone(),
            as_json: false,
            span: show.span,
        };
        let rows = string_rows(Box::pin(self.exec_show(&plain, original)).await?).await?;
        let (key, items): (&str, Vec<String>) = match &show.kind {
            ShowKind::Tables { .. } => ("tables", rows.iter().map(|r| table(r, None)).collect()),
            ShowKind::TableExtended { .. } => {
                let catalog = self.state.catalog.current_catalog();
                (
                    "tables",
                    rows.iter().map(|r| table(r, Some(&catalog))).collect(),
                )
            }
            _ => ("partitions", rows.iter().map(|r| partition(r)).collect()),
        };
        let doc = format!("{{\"{key}\":[{}]}}", items.join(","));
        self.string_frame(cols::SHOW_JSON, vec![vec![Some(doc)]])
            .await
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

/// One `SHOW TABLES` row (`namespace`, `tableName`, `isTemporary`, and for `EXTENDED` the information text), as
/// an object whose keys are in Spark's order (serde_json's map would sort them).
fn table(row: &[String], catalog: Option<&str>) -> String {
    let temporary = row.get(2).is_some_and(|t| t == "true");
    let namespace = row
        .first()
        .filter(|n| !n.is_empty())
        .map_or(String::new(), |n| js(n));
    let mut fields = vec![format!(
        "\"name\":{}",
        js(row.get(1).map_or("", String::as_str))
    )];
    if let Some(catalog) = catalog.filter(|_| !temporary) {
        fields.push(format!("\"catalog\":{}", js(catalog)));
    }
    fields.push(format!("\"namespace\":[{namespace}]"));
    if let Some(ty) = row
        .get(3)
        .and_then(|info| info.lines().find_map(|l| l.strip_prefix("Type: ")))
    {
        fields.push(format!("\"type\":{}", js(ty)));
    }
    fields.push(format!("\"isTemporary\":{temporary}"));
    format!("{{{}}}", fields.join(","))
}

/// One `SHOW PARTITIONS` row, `p=7/q=a` (values percent-escaped), as an object in column order.
fn partition(row: &[String]) -> String {
    let fields: Vec<String> = row
        .first()
        .map_or("", String::as_str)
        .split('/')
        .filter_map(|part| part.split_once('='))
        .map(|(k, v)| format!("{}:{}", js(&unescape(k)), js(&unescape(v))))
        .collect();
    format!("{{{}}}", fields.join(","))
}

/// `%XX` escapes of a partition path part, undone.
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && let Some(h) = s.get(i + 1..i + 3)
            && let Ok(v) = u8::from_str_radix(h, 16)
        {
            out.push(v);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
