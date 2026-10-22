// SPDX-License-Identifier: Apache-2.0
//! `SET`, `RESET`, `CREATE TEMPORARY VIEW` and `DROP VIEW` as
//! `Session::sql` runs them: parsing is in `krishiv-sql`, and the session
//! state they touch (configuration, temporary views) is the session's own,
//! through the same APIs every other surface uses.

use std::cmp::Ordering;
use std::sync::Arc;

use krishiv_common::config::REGISTRY;
use krishiv_common::error::{
    CreateViewColumnArityMismatchNotEnoughDataColumns,
    CreateViewColumnArityMismatchTooManyDataColumns, CreateViewWithIfNotExistsAndReplace,
    KrishivError, TableOrViewNotFound, TempTableOrViewAlreadyExists,
};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::TableColumn;
use krishiv_plan::{LogicalPlan, PlanNode};
use krishiv_sql::ast::{CreateView, DropView, Reset, Set, SetKind};
use krishiv_types::DataType;

use crate::catalog::{CATALOG, NAMESPACE, TempTable};
use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::functions::{lit, lit_null};
use crate::session::Session;

/// What `SET key` shows for a key with no value, as in Spark.
const UNDEFINED: &str = "<undefined>";

impl Session {
    /// Runs `SET`, `SET -v`, `SET key` or `SET key = value` as Spark's
    /// `spark.sql` does: setting a key goes through the session
    /// configuration, so validation and error classes are its own, and the
    /// result is a frame of string `key` and `value` columns (`SET -v`
    /// adds `meaning` and `Since version`).
    pub(crate) async fn exec_set(&self, s: &Set) -> Result<DataFrame, KrishivError> {
        match &s.kind {
            SetKind::All => {
                // Every explicitly set key, in order.
                let config = self.state.conf.snapshot();
                let mut rows: Vec<(String, String)> = config
                    .explicit_registered()
                    .into_iter()
                    .map(|(key, value)| (key.to_owned(), value))
                    .collect();
                rows.extend(
                    config
                        .explicit_unregistered()
                        .into_iter()
                        .map(|(key, value)| (key.to_owned(), value.to_owned())),
                );
                rows.sort();
                self.string_frame(
                    &["key", "value"],
                    rows.into_iter()
                        .map(|(key, value)| vec![Some(key), Some(value)])
                        .collect(),
                )
                .await
            }
            SetKind::Verbose => {
                // Every registered entry with its meaning and version, and
                // every explicitly set unregistered key, in order.
                let config = self.state.conf.snapshot();
                let mut rows: Vec<Vec<Option<String>>> = REGISTRY
                    .iter()
                    .map(|entry| {
                        let value = config
                            .get(entry)
                            .map_or_else(|| UNDEFINED.to_owned(), std::string::ToString::to_string);
                        vec![
                            Some(entry.key.to_owned()),
                            Some(value),
                            Some(entry.doc.to_owned()),
                            Some(entry.version_added.to_owned()),
                        ]
                    })
                    .collect();
                rows.extend(
                    config
                        .explicit_unregistered()
                        .into_iter()
                        .map(|(key, value)| {
                            vec![Some(key.to_owned()), Some(value.to_owned()), None, None]
                        }),
                );
                rows.sort();
                self.string_frame(&["key", "value", "meaning", "Since version"], rows)
                    .await
            }
            SetKind::Show { key, .. } => {
                let config = self.state.conf.snapshot();
                let value = crate::conf::get(&config, key).unwrap_or_else(|| UNDEFINED.to_owned());
                self.string_frame(
                    &["key", "value"],
                    vec![vec![Some(key.clone()), Some(value)]],
                )
                .await
            }
            SetKind::Assign { key, value, .. } => {
                self.conf().set(key, value)?;
                let config = self.state.conf.snapshot();
                let shown = crate::conf::get(&config, key).unwrap_or_else(|| UNDEFINED.to_owned());
                self.string_frame(
                    &["key", "value"],
                    vec![vec![Some(key.clone()), Some(shown)]],
                )
                .await
            }
            // Later `SET` forms.
            _ => Err(crate::session::internal("unknown SET form".to_owned())),
        }
    }

    /// Runs `RESET` (every key the session set back to what the session
    /// was built with) or `RESET key`, returning an empty frame as in
    /// Spark.
    pub(crate) async fn exec_reset(&self, r: &Reset) -> Result<DataFrame, KrishivError> {
        match &r.key {
            None => self.state.conf.reset(),
            Some(key) => self.conf().unset(key)?,
        }
        self.empty_frame().await
    }

    /// Runs `CREATE [OR REPLACE] TEMPORARY VIEW name [(columns)] AS
    /// query`: the view's query is analyzed now, so its errors come first
    /// as in Spark, and the view over it reads like any other temporary
    /// view. Returns an empty frame as in Spark.
    pub(crate) async fn exec_create_view(&self, v: &CreateView) -> Result<DataFrame, KrishivError> {
        if v.or_replace && v.if_not_exists {
            return Err(CreateViewWithIfNotExistsAndReplace {
                view_name: quote(&v.name.value),
            }
            .into());
        }
        let env = self.query_env();
        if !v.or_replace
            && self
                .state
                .catalog
                .exists(&v.name.value, env.analysis.conf.case_sensitive)
        {
            // `IF NOT EXISTS` turns a clash into a no-op, as in Spark.
            if v.if_not_exists {
                return self.empty_frame().await;
            }
            return Err(TempTableOrViewAlreadyExists {
                relation_name: quote(&v.name.value),
            }
            .into());
        }
        let plan = krishiv_sql::lower(&v.query, SourceId(0)).map_err(|e| e.error)?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let analysis = df.analyzed_in(env).await?;
        let output: Vec<String> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let mut columns: Vec<TableColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        if !v.columns.is_empty() {
            let declared: Vec<String> = v.columns.iter().map(|c| c.name.value.clone()).collect();
            match declared.len().cmp(&columns.len()) {
                Ordering::Greater => {
                    return Err(CreateViewColumnArityMismatchNotEnoughDataColumns {
                        view_name: quote(&v.name.value),
                        view_columns: list(&declared),
                        data_columns: list(&output),
                    }
                    .into());
                }
                Ordering::Less => {
                    return Err(CreateViewColumnArityMismatchTooManyDataColumns {
                        view_name: quote(&v.name.value),
                        view_columns: list(&declared),
                        data_columns: list(&output),
                    }
                    .into());
                }
                Ordering::Equal => {}
            }
            for (column, name) in columns.iter_mut().zip(&declared) {
                name.clone_into(&mut column.name);
            }
        }
        self.state.catalog.put(
            &v.name.value,
            TempTable::View {
                plan: Arc::clone(&df.plan),
                columns,
            },
        );
        self.empty_frame().await
    }

    /// Runs `DROP VIEW [IF EXISTS] name` over the temporary views,
    /// returning an empty frame as in Spark.
    pub(crate) async fn exec_drop_view(&self, v: &DropView) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        if self
            .state
            .catalog
            .exists(&v.name.value, env.analysis.conf.case_sensitive)
        {
            self.state.catalog.drop_table(&v.name.value);
        } else if !v.if_exists {
            return Err(TableOrViewNotFound {
                relation_name: quote(&v.name.value),
                search_path: search_path(),
            }
            .into());
        }
        self.empty_frame().await
    }

    /// An empty result with no columns, as Spark's `CREATE VIEW`,
    /// `DROP VIEW` and `RESET` return.
    async fn empty_frame(&self) -> Result<DataFrame, KrishivError> {
        let df = DataFrame::new(
            self.clone(),
            Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        )
        .select(Vec::<Column>::new())
        .limit(0);
        let env = self.query_env();
        df.analyzed_in(env).await?;
        Ok(df)
    }

    /// A frame of the string columns `columns` with `rows` (`None` is
    /// NULL), as Spark's `SET` commands return.
    async fn string_frame(
        &self,
        columns: &[&str],
        rows: Vec<Vec<Option<String>>>,
    ) -> Result<DataFrame, KrishivError> {
        let one = || {
            DataFrame::new(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
            )
        };
        let row = |values: &[Option<String>]| {
            one().select(
                columns
                    .iter()
                    .zip(values.iter())
                    .map(|(&name, value)| match value {
                        Some(value) => lit(value.as_str()).alias(name),
                        None => lit_null().cast(DataType::String).alias(name),
                    }),
            )
        };
        let df = if rows.is_empty() {
            // No rows: one placeholder row with the schema, limited to
            // nothing, so the columns read as non-nullable strings.
            row(&vec![Some(String::new()); columns.len()]).limit(0)
        } else if rows.len() == 1 {
            row(&rows[0])
        } else {
            let inputs: Vec<Arc<LogicalPlan>> =
                rows.iter().map(|r| Arc::clone(&row(r).plan)).collect();
            DataFrame::new(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::Union {
                    inputs,
                    by_name: false,
                    allow_missing_columns: false,
                })),
            )
        };
        let env = self.query_env();
        df.analyzed_in(env).await?;
        Ok(df)
    }
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Column names as Spark lists them in errors: `` `a`, `b` ``.
fn list(names: &[String]) -> String {
    names
        .iter()
        .map(|n| quote(n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Spark's search path, as analysis reports it for a missing table.
fn search_path() -> String {
    format!("[`system`.`builtin`, `system`.`session`, `{CATALOG}`.`{NAMESPACE}`]")
}
