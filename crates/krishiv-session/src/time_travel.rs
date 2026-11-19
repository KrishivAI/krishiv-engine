// SPDX-License-Identifier: Apache-2.0
//! The error of a time-travel read (`t@v1`, `t VERSION AS OF 1`, `t TIMESTAMP AS OF ..`). Reading an old version
//! is not implemented, so the lowerer refuses every one; Spark's error depends on the relation, which only the
//! session can resolve (`V2SessionCatalog.loadTable` finds the table first, `resolveTempView` names a temporary
//! view as written):
//! - no such relation: `TABLE_OR_VIEW_NOT_FOUND`;
//! - a temporary view: `UNSUPPORTED_FEATURE.TIME_TRAVEL` naming it as written;
//! - a file-source table of the session catalog: the same class naming `spark_catalog`.`<db>`.`<t>`;
//! - Delta, Iceberg and views, which Spark can read at a version: `KRISHIV_UNSUPPORTED_FEATURE`, as before.

use krishiv_common::error::{KrishivError, UnsupportedFeatureTimeTravel};
use krishiv_sql::ParseError;

use crate::session::{QueryEnv, Session};
use crate::store::CatalogStore;
use crate::table_commands::TableName;

impl Session {
    /// Spark's error for the time-travel refusal `e` of the lowerer, whose span is the relation's name in `text`;
    /// `None` when `e` is another error (or the name cannot be read back).
    pub(crate) fn time_travel_error(
        &self,
        e: &ParseError,
        text: &str,
        env: &QueryEnv,
    ) -> Option<KrishivError> {
        let is_time_travel = e
            .error
            .message_parameters()
            .iter()
            .any(|(k, v)| *k == "feature" && v == krishiv_sql::TIME_TRAVEL_FEATURE);
        if !is_time_travel {
            return None;
        }
        let written = text.get(e.span.start..e.span.end)?;
        let parts: Vec<String> = krishiv_sql::parse_multipart_identifier(written, &env.parser)
            .ok()?
            .parts
            .into_iter()
            .map(|p| p.value)
            .collect();
        let name = TableName::resolve(&parts, &self.namespace_for(&parts, env).ok()?, env).ok()?;
        if self.temp_entry_of(&name, env) {
            return Some(
                UnsupportedFeatureTimeTravel {
                    relation_id: name.written,
                }
                .into(),
            );
        }
        let Some(table) = self.state.catalog.store.table(&name.namespace, &name.name) else {
            return Some(name.not_found());
        };
        let readable_at_a_version = table.view.is_some()
            || ["delta", "iceberg"]
                .iter()
                .any(|p| table.provider.eq_ignore_ascii_case(p));
        if readable_at_a_version {
            return None;
        }
        Some(
            UnsupportedFeatureTimeTravel {
                relation_id: name.quoted(),
            }
            .into(),
        )
    }
}
