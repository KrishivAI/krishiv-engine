// SPDX-License-Identifier: Apache-2.0
//! `SHOW FUNCTIONS` and `DESCRIBE FUNCTION`.

use krishiv_common::error::KrishivError;
use krishiv_sql::ast::{DdlName, FunctionScope, ObjectName, ShowPattern};

use crate::analyze::command_schema as cols;
use crate::dataframe::DataFrame;
use crate::introspection::{s, unsupported};
use crate::session::Session;

impl Session {
    /// `SHOW [USER|SYSTEM|ALL] FUNCTIONS [IN ns] [[LIKE] pattern]` (Spark's `ShowFunctionsCommand`, probed on
    /// 4.2.0 and 4.3.0): one `function` column; built-in and temporary functions by their bare lower-case
    /// name, a stored one as `spark_catalog.ns.name`, all sorted together. `LIKE` is Spark's case-insensitive
    /// pattern (`*`, `|`), a dotted legacy pattern names the namespace in its leading parts.
    pub(crate) async fn show_functions(
        &self,
        scope: Option<FunctionScope>,
        namespace: Option<&DdlName>,
        pattern: Option<&ShowPattern>,
        legacy: Option<&ObjectName>,
    ) -> Result<DataFrame, KrishivError> {
        let (ns, pat): (Option<String>, Option<String>) = match legacy {
            Some(_) if namespace.is_some() || pattern.is_some() => {
                return Err(unsupported(
                    "SHOW FUNCTIONS with a namespace and a dotted pattern",
                ));
            }
            Some(name) => {
                let (last, lead) = name
                    .parts
                    .split_last()
                    .ok_or_else(|| unsupported("SHOW FUNCTIONS"))?;
                let ns = match lead {
                    [] => None,
                    [db] => Some(db.value.clone()),
                    [cat, db] if cat.value.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                        Some(db.value.clone())
                    }
                    _ => return Err(unsupported("SHOW FUNCTIONS with this pattern")),
                };
                (ns, Some(last.value.clone()))
            }
            None => {
                let ns = match namespace {
                    Some(_) => Some(self.show_namespace(namespace).await?),
                    None => None,
                };
                (ns, pattern.map(|p| p.pattern.clone()))
            }
        };
        let listed = self
            .catalog()
            .list_functions_in(ns.as_deref(), pat.as_deref())
            .await?;
        let builtins = krishiv_functions::registry::builtins();
        let mut names: Vec<String> = listed
            .into_iter()
            .filter(|f| {
                let system = f.is_temporary && builtins.spark_function(&f.name).is_some();
                match scope {
                    Some(FunctionScope::User) => !system,
                    Some(FunctionScope::System) => system,
                    _ => true,
                }
            })
            .map(|f| match (&f.catalog, &f.namespace) {
                (Some(c), Some(n)) => format!("{c}.{}.{}", n.join("."), f.name),
                _ => f.name,
            })
            .collect();
        names.sort();
        names.dedup();
        self.ordered_frame(
            cols::SHOW_FUNCTIONS,
            &[],
            names.into_iter().map(|n| vec![s(n)]).collect(),
        )
        .await
    }

    /// `DESCRIBE FUNCTION [EXTENDED] name`: a built-in prints Spark 4.2.0's own rows (`Function:`, `Class:`,
    /// `Usage:`, and with `EXTENDED` the `Extended Usage:` block; the text is vendored from a Spark run). A name
    /// that is no function is `UNRESOLVED_ROUTINE`. A SQL-bodied or native function of the session or the
    /// catalog is found but its description is not modelled yet (`KRISHIV_UNSUPPORTED_FEATURE`).
    pub(crate) async fn describe_function(
        &self,
        parts: &[String],
        extended: bool,
    ) -> Result<DataFrame, KrishivError> {
        // `builtin.f` and `system.builtin.f` name the same built-in as `f`.
        let parts = match parts {
            [b, f] if b.eq_ignore_ascii_case("builtin") => std::slice::from_ref(f),
            [sy, b, f]
                if sy.eq_ignore_ascii_case("system") && b.eq_ignore_ascii_case("builtin") =>
            {
                std::slice::from_ref(f)
            }
            other => other,
        };
        if let [one] = parts
            && let Some(rows) =
                krishiv_functions::registry::spark_function_description(&one.to_lowercase())
        {
            let rows = rows
                .iter()
                .filter(|r| extended || !r.starts_with("Extended Usage:"))
                .map(|r| vec![s(r)])
                .collect();
            return self.ordered_frame(cols::DESCRIBE_FUNCTION, &[], rows).await;
        }
        if let [one] = parts
            && krishiv_functions::registry::builtins()
                .spark_function(&one.to_lowercase())
                .is_some()
        {
            return Err(unsupported("DESCRIBE FUNCTION of this built-in"));
        }
        // Found: a function of the session or the catalog; not found: `UNRESOLVED_ROUTINE`.
        self.catalog().get_function(&parts.join(".")).await?;
        Err(unsupported("DESCRIBE FUNCTION of a user-defined function"))
    }
}
