// SPDX-License-Identifier: Apache-2.0
//! The error of a time-travel read (`t@v1`, `t VERSION AS OF 1`, `t TIMESTAMP AS OF ..`). Reading an old version
//! is not implemented, so the lowerer refuses every one; Spark's error depends on the relation, which only the
//! session can resolve (`V2SessionCatalog.loadTable` finds the table first, `resolveTempView` names a temporary
//! view as written):
//! - no such relation: `TABLE_OR_VIEW_NOT_FOUND`;
//! - a temporary view: `UNSUPPORTED_FEATURE.TIME_TRAVEL` naming it as written;
//! - a file-source table of the session catalog: the same class naming `spark_catalog`.`<db>`.`<t>`;
//! - a global temporary view: the same class naming it as written;
//! - a permanent view of the session catalog: the same class naming `spark_catalog`.`<db>`.`<v>` (a view
//!   loads as a `V1Table`, which `failTimeTravel` refuses);
//! - Delta and Iceberg, which Spark can read at a version: `KRISHIV_UNSUPPORTED_FEATURE`, as before.
//!
//! A CTE name is named by the lowerer, which knows the CTE scope. [`Session::lowering_error`] applies this to
//! every lowering of a statement of the session.

use krishiv_common::error::{KrishivError, UnsupportedFeatureTimeTravel};
use krishiv_sql::ParseError;

use crate::session::{QueryEnv, Session};
use crate::store::CatalogStore;
use crate::table_commands::TableName;

impl Session {
    /// Spark's error for the time-travel refusal `e` of the lowerer, whose span is the relation's name in `text`;
    /// `None` when `e` is another error (or the name cannot be read back).
    pub(crate) fn time_travel_error(&self, e: &ParseError, env: &QueryEnv) -> Option<KrishivError> {
        let is_time_travel = e
            .error
            .message_parameters()
            .iter()
            .any(|(k, v)| *k == "feature" && v == krishiv_sql::TIME_TRAVEL_FEATURE);
        if !is_time_travel {
            return None;
        }
        // The catalog is read only for a session that lets everyone read everything: before the
        // analysis asks the policy, an answer that depends on the relation existing would tell a
        // denied principal which names exist. Any other session gets the lowerer's refusal.
        if !self.state.policy_open {
            return None;
        }
        let parts = e.relation.as_deref()?;
        let name = TableName::resolve(parts, &self.namespace_for(parts, env).ok()?, env).ok()?;
        if self.temp_entry_of(&name, env) {
            return Some(
                UnsupportedFeatureTimeTravel {
                    relation_id: name.written,
                }
                .into(),
            );
        }
        let global = parts.len() == 2
            && parts[0].eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
            && self.state.catalog.exists(
                &crate::catalog::global_key(&name.name),
                env.analysis.conf.case_sensitive,
            );
        if global {
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
        let readable_at_a_version = table.view.is_none()
            && ["delta", "iceberg"]
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

    /// The error of a failed lowering `e` of a statement of `env`: Spark's time-travel error where it is one,
    /// else the lowerer's own.
    pub(crate) fn lowering_error(&self, e: ParseError, env: &QueryEnv) -> KrishivError {
        self.time_travel_error(&e, env).unwrap_or(e.error)
    }
}
