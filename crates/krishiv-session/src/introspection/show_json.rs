// SPDX-License-Identifier: Apache-2.0
//! `SHOW TABLES | SHOW TABLE EXTENDED | SHOW PARTITIONS ... AS JSON` (Spark 4.3).

use krishiv_common::error::{KrishivError, UnsupportedFeatureShowTableExtendedJsonWithPartition};
use krishiv_sql::ast::{Show, ShowKind};

use super::{json_list, json_str, s};
use crate::analyze::command_schema as cols;
use crate::dataframe::DataFrame;
use crate::session::Session;
use crate::store::CatalogStore;

/// The `SHOW` forms that take `AS JSON`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum JsonForm {
    Tables,
    TableExtended,
    Partitions,
}

/// Which `AS JSON` form `kind` is, if it was written with `AS JSON`.
pub(super) fn json_form(kind: &ShowKind) -> Option<JsonForm> {
    match kind {
        ShowKind::Tables { as_json: true, .. } => Some(JsonForm::Tables),
        ShowKind::TableExtended { as_json: true, .. } => Some(JsonForm::TableExtended),
        ShowKind::Partitions { as_json: true, .. } => Some(JsonForm::Partitions),
        _ => None,
    }
}

impl Session {
    /// `SHOW TABLES | SHOW TABLE EXTENDED | SHOW PARTITIONS ... AS JSON` (Spark 4.3): the listing the plain
    /// statement makes, as one `json_metadata` document. Everything the plain form checks (access, missing
    /// names, partition specs) is checked by running it.
    pub(super) fn show_as_json<'a>(
        &'a self,
        show: &'a Show,
        kind: JsonForm,
        original: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<DataFrame, KrishivError>> + Send + 'a>,
    > {
        Box::pin(async move {
            if kind == JsonForm::TableExtended
                && matches!(
                    &show.kind,
                    ShowKind::TableExtended {
                        partition: Some(_),
                        ..
                    }
                )
            {
                return Err(UnsupportedFeatureShowTableExtendedJsonWithPartition {}.into());
            }
            let mut plain = show.clone();
            match &mut plain.kind {
                ShowKind::Tables { as_json, .. }
                | ShowKind::TableExtended { as_json, .. }
                | ShowKind::Partitions { as_json, .. } => *as_json = false,
                _ => {}
            }
            let listing = match &plain.kind {
                ShowKind::TableExtended {
                    namespace, pattern, ..
                } => {
                    self.show_table_extended(namespace.as_ref(), pattern, None, original, true)
                        .await?
                }
                _ => self.exec_show_catalog(&plain, original).await?,
            };
            let batches = listing.collect().await?;
            let mut rows: Vec<Vec<String>> = Vec::new();
            for b in &batches {
                for i in 0..b.num_rows() {
                    let mut row = Vec::with_capacity(b.num_columns());
                    for c in 0..b.num_columns() {
                        row.push(
                            arrow_cast::display::array_value_to_string(b.column(c), i)
                                .map_err(|e| crate::session::internal(e.to_string()))?,
                        );
                    }
                    rows.push(row);
                }
            }
            let namespace = |n: &str| json_list(n.split('.').filter(|p| !p.is_empty()));
            let json = match kind {
                JsonForm::Tables => {
                    let items: Vec<String> = rows
                        .iter()
                        .map(|r| {
                            format!(
                                r#"{{"name":{},"namespace":{},"isTemporary":{}}}"#,
                                json_str(&r[1]),
                                namespace(&r[0]),
                                r[2]
                            )
                        })
                        .collect();
                    format!(r#"{{"tables":[{}]}}"#, items.join(","))
                }
                JsonForm::TableExtended => {
                    let mut items = Vec::with_capacity(rows.len());
                    for r in &rows {
                        let temp = r[2] == "true";
                        let kind = if temp {
                            "VIEW"
                        } else {
                            match self.state.catalog.store.table(&r[0], &r[1]) {
                                Some(t) if t.view.is_some() => "VIEW",
                                Some(t) if !t.managed => "EXTERNAL",
                                _ => "MANAGED",
                            }
                        };
                        items.push(format!(
                            r#"{{"name":{},"catalog":"spark_catalog","namespace":{},"type":"{kind}","isTemporary":{temp}}}"#,
                            json_str(&r[1]),
                            namespace(&r[0]),
                        ));
                    }
                    format!(r#"{{"tables":[{}]}}"#, items.join(","))
                }
                JsonForm::Partitions => {
                    let items: Vec<String> = rows
                        .iter()
                        .map(|r| {
                            // The catalog's own (escaped) names, split as Spark's
                            // `partitionNameToJson` does: on `/`, then the first `=`.
                            let fields: Vec<String> = r[0]
                                .split('/')
                                .map(|kv| kv.split_once('=').unwrap_or((kv, "")))
                                .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
                                .collect();
                            format!("{{{}}}", fields.join(","))
                        })
                        .collect();
                    format!(r#"{{"partitions":[{}]}}"#, items.join(","))
                }
            };
            self.ordered_frame(cols::DESCRIBE_JSON, &[], vec![vec![s(json)]])
                .await
        })
    }
}
