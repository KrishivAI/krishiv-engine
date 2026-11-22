// SPDX-License-Identifier: Apache-2.0
//! Session methods that render the SQL search path an unresolved-relation error names.

use super::*;

impl Session {
    /// The search path a missing table or routine names in its error: the SQL path (`SET PATH`, else the
    /// `spark.sql.path` default) resolved at the current catalog and namespace, as the analyzer reports it.
    pub(crate) fn search_path(&self) -> String {
        crate::catalog::render_search_path(&self.search_path_entries())
    }

    fn search_path_entries(&self) -> Vec<Vec<String>> {
        let (config, _) = self.state.conf.query_snapshot();
        let default = config
            .get(&krishiv_common::config::SQL_DEFAULT_PATH)
            .map(ToString::to_string)
            .unwrap_or_default();
        let (catalog, namespace) = self.state.catalog.position_pair();
        let namespace = if crate::system::is_system(&catalog) {
            Vec::new()
        } else {
            vec![namespace]
        };
        self.state
            .catalog
            .sql_path_in(&default, &catalog, &namespace)
    }

    /// The search path of a command that resolves only tables and views (`ALTER TABLE`, `SHOW COLUMNS`,
    /// `SHOW TABLE EXTENDED`): [`Self::search_path`] without `system.builtin`, which holds functions only.
    pub(crate) fn table_search_path(&self) -> String {
        let mut path = self.search_path_entries();
        path.retain(|e| {
            !(e.len() == 2
                && e[0].eq_ignore_ascii_case("system")
                && e[1].eq_ignore_ascii_case("builtin"))
        });
        crate::catalog::render_search_path(&path)
    }

    /// The error of a v1 `DROP VIEW` that finds nothing: the catalog and the namespace the name resolved
    /// to (Spark's `NoSuchTableException` carries no SQL path), with the relation qualified by them.
    pub(crate) fn drop_view_not_found(&self, namespace: &str, name: &str) -> KrishivError {
        let (catalog, _) = self.state.catalog.position_pair();
        let q = |s: &str| format!("`{}`", s.replace('`', "``"));
        krishiv_common::error::TableOrViewNotFound {
            relation_name: format!("{}.{}.{}", q(&catalog), q(namespace), q(name)),
            search_path: format!("[{}.{}]", q(&catalog), q(namespace)),
        }
        .into()
    }
}
