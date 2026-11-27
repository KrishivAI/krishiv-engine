// SPDX-License-Identifier: Apache-2.0
//! `SHOW FUNCTIONS` and `DESCRIBE FUNCTION`.

use krishiv_common::error::KrishivError;
use krishiv_sql::ast::{DdlName, FunctionScope, ObjectName, ShowPattern};
use krishiv_types::DataType;

use crate::analyze::command_schema as cols;
use crate::dataframe::DataFrame;
use crate::introspection::{s, unsupported};
use crate::session::Session;
use crate::sql_functions::Entry;
use crate::store::CatalogStore as _;

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
            .list_functions_tagged(ns.as_deref(), pat.as_deref())
            .await?;
        let mut names: Vec<String> = listed
            .into_iter()
            .filter(|(_, system)| match scope {
                Some(FunctionScope::User) => !system,
                Some(FunctionScope::System) => *system,
                _ => true,
            })
            .map(|(f, _)| match (&f.catalog, &f.namespace) {
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
    /// that is no function is `UNRESOLVED_ROUTINE`. A SQL function of the session or the catalog prints
    /// Spark 4.3.0's rows ([`Self::describe_sql_function`]); a native one is found but its
    /// description is not modelled yet (`KRISHIV_UNSUPPORTED_FEATURE`).
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
        // A built-in the session's Spark profile does not have resolves as an unknown name.
        let in_profile = |name: &str| {
            !matches!(
                krishiv_functions::registry::builtins().lookup(name, self.state.profile),
                krishiv_functions::registry::Lookup::NotFound
            )
        };
        if let [one] = parts
            && in_profile(one)
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
            && in_profile(one)
            && krishiv_functions::registry::builtins()
                .spark_function(&one.to_lowercase())
                .is_some()
        {
            return Err(unsupported("DESCRIBE FUNCTION of this built-in"));
        }
        // `session.f` and `system.session.f` name the temporary function `f`.
        let session_qualified = match parts {
            [q, f] if q.eq_ignore_ascii_case("session") => Some(f),
            [sy, q, f]
                if sy.eq_ignore_ascii_case("system") && q.eq_ignore_ascii_case("session") =>
            {
                Some(f)
            }
            _ => None,
        };
        if let Some(f) = session_qualified
            && let Some(entry) = self.state.sql_functions.snapshot().get(f).cloned()
            // The same check as a bare name's (one the principal may not list reads as missing).
            && self
                .catalog()
                .get_function(f)
                .await
                .is_ok_and(|found| found.catalog.is_none())
        {
            return self
                .describe_sql_function(&entry.def.name, &entry, extended)
                .await;
        }
        // Found: a function of the session or the catalog; not found: `UNRESOLVED_ROUTINE`.
        let found = self.catalog().get_function(&parts.join(".")).await?;
        let entry = match (&found.catalog, &found.namespace) {
            (Some(c), Some(ns)) => self
                .state
                .catalog
                .store
                .functions()
                .get(&ns.join("."), &found.name)
                .map(|f| {
                    (
                        format!("{c}.{}.{}", ns.join("."), found.name),
                        f.entry.clone(),
                    )
                }),
            _ => self
                .state
                .sql_functions
                .snapshot()
                .get(&found.name)
                .map(|e| (found.name.clone(), e.clone())),
        };
        match entry {
            Some((name, entry)) => self.describe_sql_function(&name, &entry, extended).await,
            None => Err(unsupported("DESCRIBE FUNCTION of a user-defined function")),
        }
    }

    /// The rows of a SQL function (Spark 4.3.0's `DescribeFunctionCommand.describeSQLFunction`):
    /// `Function:`, `Type:`, `Input:` and `Returns:`, and with `EXTENDED` the comment,
    /// determinism, data access, captured configuration, creation time and body. Keys are padded
    /// to the longest; a parameter or column list continues on rows with an empty key.
    async fn describe_sql_function(
        &self,
        name: &str,
        entry: &Entry,
        extended: bool,
    ) -> Result<DataFrame, KrishivError> {
        let def = &entry.def;
        let info = &entry.described;
        let mut rows: Vec<(&str, String)> = vec![("Function:", name.to_owned())];
        let table = def.returns_table.is_some();
        rows.push(("Type:", if table { "TABLE" } else { "SCALAR" }.to_owned()));
        let input: Vec<Param<'_>> = def
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| Param {
                name: &p.name,
                data_type: p.data_type.sql(),
                default: info.defaults.get(i).and_then(Option::as_deref),
                comment: p.comment.as_deref(),
            })
            .collect();
        push_list(&mut rows, "Input:", &format_params(&input, extended), "()");
        match &def.returns_table {
            Some(columns) => {
                // `RETURNS TABLE` without columns returns the query's own.
                let columns = if columns.is_empty() {
                    &info.columns
                } else {
                    columns
                };
                let columns: Vec<(&str, String)> =
                    columns.iter().map(|(n, t)| (n.as_str(), t.sql())).collect();
                let params: Vec<Param<'_>> = columns
                    .into_iter()
                    .map(|(name, data_type)| Param {
                        name,
                        data_type,
                        default: None,
                        comment: None,
                    })
                    .collect();
                push_list(&mut rows, "Returns:", &format_params(&params, extended), "");
            }
            None => match &def.returns {
                Some(DataType::Null) | None => {}
                Some(t) => rows.push(("Returns:", t.sql())),
            },
        }
        if extended {
            if let Some(c) = &info.comment {
                rows.push(("Comment:", c.clone()));
            }
            rows.push(("Deterministic:", info.deterministic.to_string()));
            let access = if def.reads_data {
                "READS SQL DATA"
            } else {
                "CONTAINS SQL"
            };
            rows.push(("Data Access:", access.to_owned()));
            let configs: Vec<String> = info
                .configs
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            push_list(&mut rows, "Configs:", &configs, "");
            rows.push((
                "Create Time:",
                crate::introspection::created_time(info.created_ms),
            ));
            rows.push(("Body:", def.body_text.clone()));
        }
        let width = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let rows = rows
            .into_iter()
            .map(|(k, v)| vec![s(format!("{k:<width$} {v}"))])
            .collect();
        self.ordered_frame(cols::DESCRIBE_FUNCTION, &[], rows).await
    }
}

/// A parameter (or a returned column) as `DESCRIBE FUNCTION` lists it.
struct Param<'a> {
    name: &'a str,
    data_type: String,
    default: Option<&'a str>,
    comment: Option<&'a str>,
}

/// Spark's `formatParameters`: names and types each padded to the longest, then (with `EXTENDED`
/// only) ` DEFAULT d` and ` 'comment'`.
fn format_params(params: &[Param<'_>], extended: bool) -> Vec<String> {
    let name_w = params
        .iter()
        .map(|p| p.name.chars().count())
        .max()
        .unwrap_or(0);
    let type_w = params
        .iter()
        .map(|p| p.data_type.chars().count())
        .max()
        .unwrap_or(0);
    params
        .iter()
        .map(|p| {
            let mut line = format!("{:<name_w$} {:<type_w$}", p.name, p.data_type);
            if extended {
                if let Some(d) = p.default {
                    line.push_str(" DEFAULT ");
                    line.push_str(d);
                }
                if let Some(c) = p.comment {
                    line.push_str(&format!(" '{c}'"));
                }
            }
            line
        })
        .collect()
}

/// `key: first`, then one row with an empty key per further item; `empty` alone when there are
/// none (no row when `empty` is empty).
fn push_list<'k>(rows: &mut Vec<(&'k str, String)>, key: &'k str, items: &[String], empty: &str) {
    match items.split_first() {
        Some((first, rest)) => {
            rows.push((key, first.clone()));
            rows.extend(rest.iter().map(|r| ("", r.clone())));
        }
        None if !empty.is_empty() => rows.push((key, empty.to_owned())),
        None => {}
    }
}
