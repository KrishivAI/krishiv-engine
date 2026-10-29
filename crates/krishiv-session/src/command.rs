// SPDX-License-Identifier: Apache-2.0
//! `SET`, `RESET`, `CREATE TEMPORARY VIEW`, `DROP VIEW` and the session
//! variable commands (`DECLARE`, `SET VARIABLE`, `DROP TEMPORARY
//! VARIABLE`) as [`Session::sql`](crate::Session::sql) runs them: parsing
//! is in `krishiv-sql`, and the session state they touch (configuration,
//! temporary views, variables) is the session's own, through the same APIs
//! every other surface uses.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::StreamExt;
use krishiv_common::config::REGISTRY;
use krishiv_common::error::{
    AssignmentArityMismatch, ColumnAlreadyExists,
    CreateViewColumnArityMismatchNotEnoughDataColumns,
    CreateViewColumnArityMismatchTooManyDataColumns, CreateViewWithIfNotExistsAndReplace,
    DefaultPlacementInvalid, DuplicateAssignments, DuplicateVariableNameInsideDeclare, ErrorClass,
    ErrorQueryContext, InvalidDefaultValueSubqueryExpression, KrishivError,
    KrishivUnsupportedFeature, RowSubqueryTooManyRows, TableOrViewNotFound,
    TempTableOrViewAlreadyExists, UnresolvedVariable, UnsupportedFeatureSetVariableUsingSet,
    VariableAlreadyExists, VariableNotFound,
};
use krishiv_common::redact::RedactedSql;
use krishiv_plan::analyzer::{TableColumn, variable_name};
use krishiv_plan::{Literal, LogicalPlan, PlanNode, SourceId};
use krishiv_sql::ast::{
    Cache, CacheKind, CreateView, DataTypeSpec, DdlName, DeclareVariable, DropVariable, DropView,
    Expr, ExprKind, ObjectName, Refresh, RefreshKind, Reset, Set, SetKind, SetVariable,
    SetVariableKind, Span, TimeZone,
};
use krishiv_types::{DataType, DecimalType};

use crate::cache::is_cache_name;
use crate::catalog::{CATALOG, NAMESPACE, TempTable};
use crate::catalog::{is_hidden_name, reserved_temp_view_name};
use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::functions::{col, lit, lit_null};
use crate::session::Session;
use crate::variables::{
    DECLARE_SEARCH_PATH, RESOLVE_SEARCH_PATH, StoredVariable, quote_name, quote_normalized,
    variable_key,
};

/// What `SET key` shows for a key with no value, as in Spark.
const UNDEFINED: &str = "<undefined>";

/// The configuration key `SET TIME ZONE` sets and shows, as in Spark.
const TIME_ZONE_KEY: &str = "spark.sql.session.timeZone";

/// The system's time zone name, for `SET TIME ZONE LOCAL`: the `TZ`
/// environment variable when it names a valid zone, else the target of
/// `/etc/localtime` under the system zone directory, else `UTC`.
fn system_zone_name() -> String {
    if let Some(tz) = std::env::var_os("TZ")
        && let Some(tz) = tz.to_str()
        && !tz.is_empty()
        && krishiv_common::config::lookup(TIME_ZONE_KEY)
            .is_some_and(|entry| entry.parse(TIME_ZONE_KEY, tz).is_ok())
    {
        return tz.to_owned();
    }
    if let Ok(target) = std::fs::read_link("/etc/localtime")
        && let Some(target) = target.to_str()
        && let Some(zone) = target
            .split_once("/zoneinfo/")
            .map(|(_, zone)| zone)
            .filter(|zone| !zone.is_empty() && !zone.contains("/../"))
        && krishiv_common::config::lookup(TIME_ZONE_KEY)
            .is_some_and(|entry| entry.parse(TIME_ZONE_KEY, zone).is_ok())
    {
        return zone.to_owned();
    }
    "UTC".to_owned()
}

impl Session {
    /// Runs `SET`, `SET -v`, `SET key` or `SET key = value` as Spark's
    /// `spark.sql` does: setting a key goes through the session
    /// configuration, so validation and error classes are its own, and the
    /// result is a frame of string `key` and `value` columns (`SET -v`
    /// adds `meaning` and `Since version`).
    pub(crate) async fn exec_set(
        &self,
        s: &Set,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        match &s.kind {
            SetKind::All => {
                // Every explicitly set key (the frame sorts by key).
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
                self.string_frame(
                    &["key", "value"],
                    rows.into_iter()
                        .map(|(key, value)| vec![Some(key), Some(value)])
                        .collect(),
                )
                .await
            }
            SetKind::Verbose => {
                // Every registered entry with its meaning and version (the
                // frame sorts by key). Unregistered keys are not listed, as
                // in Spark.
                let config = self.state.conf.snapshot();
                let rows: Vec<Vec<Option<String>>> = REGISTRY
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
            SetKind::Assign {
                key,
                value,
                key_span,
                ..
            } => {
                // A variable is never set with `SET`: `v`, `session.v` and
                // `system.session.v` name the variable `v` when it exists,
                // as in Spark (a backquoted key is one part, whatever dots
                // it holds).
                if let Some(parts) = assign_variable_parts(key, *key_span, text)
                    && let Some(variable) = parts.last()
                    && self.state.variables.get(&variable.to_lowercase()).is_some()
                {
                    return Err(UnsupportedFeatureSetVariableUsingSet {
                        variable_name: parts
                            .iter()
                            .map(|p| format!("`{}`", p.replace('`', "``")))
                            .collect::<Vec<_>>()
                            .join("."),
                    }
                    .into());
                }
                // The assignment sets through the session configuration,
                // so validation and error classes are its own; the row
                // shows the value as written, as in Spark.
                self.conf().set(key, value)?;
                self.string_frame(
                    &["key", "value"],
                    vec![vec![Some(key.clone()), Some(value.clone())]],
                )
                .await
            }
            SetKind::TimeZone { zone, .. } => {
                // `SET TIME ZONE` sets the session time zone and shows it,
                // as in Spark.
                let value = match zone {
                    TimeZone::Local => system_zone_name(),
                    TimeZone::Named(zone) => zone.clone(),
                    _ => {
                        return Err(crate::session::internal(
                            "unknown SET TIME ZONE form".to_owned(),
                        ));
                    }
                };
                self.conf().set(TIME_ZONE_KEY, &value)?;
                self.string_frame(
                    &["key", "value"],
                    vec![vec![Some(TIME_ZONE_KEY.to_owned()), Some(value)]],
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
    /// query`: a view that reads itself (through `OR REPLACE`) fails when
    /// it is created with `RECURSIVE_VIEW`, as in Spark; otherwise the
    /// view's query is analyzed now, so its errors come first. The query
    /// lowers with a source id of its own, so errors raised while a later
    /// query expands the view point into the view's text, not the outer
    /// statement's. Returns an empty frame as in Spark.
    ///
    /// # Errors
    /// `KRISHIV_RESERVED_TEMP_VIEW_NAME` when `name` starts with the
    /// reserved snapshot prefix `` `__krishiv_snapshot_` `` or the cache
    /// prefix `` `__krishiv_cache_` `` (in any letter case), which queries
    /// cannot spell either.
    pub(crate) async fn exec_create_view(&self, v: &CreateView) -> Result<DataFrame, KrishivError> {
        if is_hidden_name(&v.name.value) || is_cache_name(&v.name.value) {
            return Err(reserved_temp_view_name(&v.name.value));
        }
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
            return Err(TempTableOrViewAlreadyExists {
                relation_name: quote(&v.name.value),
            }
            .into());
        }
        let plan = krishiv_sql::lower(&v.query, self.next_source_id()).map_err(|e| e.error)?;
        crate::catalog::check_view_cycle(
            &self.state.catalog,
            &v.name.value,
            plan.root(),
            env.analysis.conf.case_sensitive,
        )?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let case_sensitive = env.analysis.conf.case_sensitive;
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
        // Spark reports a repeated view column with `COLUMN_ALREADY_EXISTS`
        // (its view path checks field names with
        // `SchemaUtils.checkColumnNameDuplication`): after the aliases are
        // applied, so both a duplicated alias list and duplicate output
        // names fail, ignoring case unless `case_sensitive` (then the
        // repeated name is reported as written, else in lower case, as in
        // Spark 4.2 and as [`unique_columns`](crate::session) reports it
        // for writes).
        unique_view_columns(&columns, case_sensitive)?;
        self.state.catalog.put(
            &v.name.value,
            TempTable::View {
                plan: Arc::clone(&df.plan),
                columns,
                pins: Vec::new(),
            },
        );
        self.state.cache.invalidate_table(&v.name.value);
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
            self.state.cache.invalidate_table(&v.name.value);
        } else if !v.if_exists {
            return Err(TableOrViewNotFound {
                relation_name: quote(&v.name.value),
                search_path: search_path(),
            }
            .into());
        }
        self.empty_frame().await
    }

    /// Whether the table or view `parts` names exists: temporary tables and
    /// views have single-part names, so a qualified name never matches.
    fn cache_target_exists(&self, parts: &[String]) -> bool {
        let env = self.query_env();
        matches!(parts, [only] if self
            .state
            .catalog
            .exists(only, env.analysis.conf.case_sensitive))
    }

    /// Runs `CLEAR CACHE` and `UNCACHE TABLE [IF EXISTS] name`: both drop
    /// cached relations and return an empty frame, as in Spark. `CACHE
    /// TABLE ...` stays `KRISHIV_UNSUPPORTED_FEATURE`: caching through SQL
    /// is not supported, only `DataFrame.cache` and `persist`.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND` for `UNCACHE TABLE` of a missing table
    /// without `IF EXISTS`; `KRISHIV_UNSUPPORTED_FEATURE` for `CACHE TABLE`.
    pub(crate) async fn exec_cache(&self, c: &Cache) -> Result<DataFrame, KrishivError> {
        match &c.kind {
            CacheKind::Clear => {
                self.clear_cache();
            }
            CacheKind::Uncache {
                if_exists, name, ..
            } => {
                let Some(parts) = ddl_table_parts(name) else {
                    return Err(KrishivUnsupportedFeature {
                        feature: "UNCACHE TABLE with a dynamic table name".to_owned(),
                    }
                    .into());
                };
                if self.cache_target_exists(&parts) {
                    self.uncache_table(&parts[0]);
                } else if !if_exists {
                    return Err(table_not_found(&parts));
                }
            }
            CacheKind::Cache { .. } => {
                return Err(KrishivUnsupportedFeature {
                    feature: "CACHE TABLE in sessions".to_owned(),
                }
                .into());
            }
            _ => {
                return Err(crate::session::internal("unknown CACHE form".to_owned()));
            }
        }
        self.empty_frame().await
    }

    /// Runs `REFRESH TABLE name`, `REFRESH FUNCTION name` and `REFRESH
    /// path`: all drop the cached relations they cover and return an empty
    /// frame. Refreshing a function is a no-op (functions hold no cache),
    /// and unlike Spark it does not check that the function exists;
    /// refreshing a path never fails for a missing one.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND` for `REFRESH TABLE` of a missing table;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a dynamic `IDENTIFIER(...)` name.
    pub(crate) async fn exec_refresh(&self, r: &Refresh) -> Result<DataFrame, KrishivError> {
        match &r.kind {
            RefreshKind::Table { name, .. } => {
                let Some(parts) = ddl_table_parts(name) else {
                    return Err(KrishivUnsupportedFeature {
                        feature: "REFRESH TABLE with a dynamic table name".to_owned(),
                    }
                    .into());
                };
                if !self.cache_target_exists(&parts) {
                    return Err(table_not_found(&parts));
                }
                self.refresh_table(&parts[0]);
            }
            RefreshKind::Function { .. } => {}
            RefreshKind::Path { .. } => {
                self.refresh_files();
            }
            _ => {
                return Err(crate::session::internal("unknown REFRESH form".to_owned()));
            }
        }
        self.empty_frame().await
    }

    /// Runs `DECLARE [OR REPLACE] [VARIABLE | VAR] name [, ...] [type]
    /// [{DEFAULT | =} expr]`: every name resolves on the declare search
    /// path, duplicates and (without `OR REPLACE`) existing variables
    /// fail, the default is evaluated by running a one-row query through
    /// the normal pipeline and cast to the declared type, and all names
    /// share the value. Returns Spark's empty result shape.
    pub(crate) async fn exec_declare(
        &self,
        v: &DeclareVariable,
        original: &str,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let mut keys = Vec::with_capacity(v.names.len());
        for name in &v.names {
            let Some(key) = variable_key(name) else {
                return Err(unresolved_variable(name, DECLARE_SEARCH_PATH, text));
            };
            keys.push(key);
        }
        let mut seen = HashSet::new();
        for key in &keys {
            if !seen.insert(key) {
                return Err(DuplicateVariableNameInsideDeclare {
                    variable_name: quote_normalized(key),
                }
                .into());
            }
        }
        if !v.or_replace {
            for key in &keys {
                if self.state.variables.get(key).is_some() {
                    return Err(VariableAlreadyExists {
                        variable_name: quote_normalized(key),
                    }
                    .into());
                }
            }
        }
        let declared = v
            .data_type
            .as_ref()
            .map(DataTypeSpec::to_data_type)
            .transpose()?;
        let (data_type, value, default_query) = match (&declared, &v.default) {
            (_, None) => match declared {
                Some(t) => (t.clone(), Literal::Null(Some(t)), None),
                None => (DataType::Null, Literal::Null(None), None),
            },
            (declared, Some(default)) => {
                if contains_subquery(default) {
                    return Err(InvalidDefaultValueSubqueryExpression {
                        col_name: quote_normalized(&keys[0]),
                        default_value: text
                            .fragment(default.span.start..default.span.end)
                            .to_owned(),
                        statement: "DECLARE VARIABLE".to_owned(),
                    }
                    .into());
                }
                let slice = original
                    .get(default.span.start..default.span.end)
                    .unwrap_or_default();
                let sql = match declared {
                    Some(t) => padded_select(
                        "SELECT CAST(",
                        slice,
                        default.span.start,
                        &format!(" AS {})", t),
                    ),
                    None => padded_select("SELECT ", slice, default.span.start, ""),
                };
                let (inferred, value) = self
                    .eval_variable_value(&sql, text, v.span, default.span)
                    .await?;
                let data_type = declared.clone().unwrap_or(inferred);
                (data_type, value, Some(sql))
            }
        };
        for key in keys {
            self.state.variables.put(
                key,
                StoredVariable {
                    data_type: data_type.clone(),
                    value: value.clone(),
                    default_query: default_query.clone(),
                },
            );
        }
        self.empty_frame().await
    }

    /// Runs `SET VAR[IABLE] v = expr [, ...]` or `SET VAR[IABLE] (v, ...)
    /// = (query)`: every name resolves on the resolve search path, all
    /// right-hand sides are evaluated before any is assigned, a row query
    /// assigns its single row (no rows assign NULL) and `DEFAULT` reads
    /// the variable's default. Returns Spark's empty result shape.
    /// Evaluate-then-assign spans no lock: concurrent writers must be
    /// externally synchronized.
    pub(crate) async fn exec_set_variable(
        &self,
        v: &SetVariable,
        original: &str,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        match &v.kind {
            SetVariableKind::Single { assignments } => {
                let mut keys = Vec::with_capacity(assignments.len());
                for a in assignments {
                    let Some(key) = variable_key(&a.name) else {
                        return Err(unresolved_variable(&a.name, RESOLVE_SEARCH_PATH, text));
                    };
                    keys.push(key);
                }
                duplicate_targets(&keys)?;
                let mut stored = Vec::with_capacity(keys.len());
                for key in &keys {
                    let Some(var) = self.state.variables.get(key) else {
                        return Err(VariableNotFound {
                            variable_name: quote_normalized(key),
                        }
                        .into());
                    };
                    stored.push(var);
                }
                let has_default_var = self.state.variables.get("default").is_some();
                let mut pending = Vec::with_capacity(keys.len());
                for ((key, var), a) in keys.iter().zip(&stored).zip(assignments) {
                    let value = if is_bare_default(&a.value) && !has_default_var {
                        match &var.default_query {
                            Some(query) => {
                                let (_, value) = self
                                    .eval_variable_value(query, text, v.span, a.value.span)
                                    .await?;
                                value
                            }
                            None => Literal::Null(Some(var.data_type.clone())),
                        }
                    } else {
                        if contains_default(&a.value) && !has_default_var {
                            return Err(DefaultPlacementInvalid {}.into());
                        }
                        let slice = original
                            .get(a.value.span.start..a.value.span.end)
                            .unwrap_or_default();
                        let sql = padded_select(
                            "SELECT CAST(",
                            slice,
                            a.value.span.start,
                            &format!(" AS {})", var.data_type.simple_string()),
                        );
                        let (_, value) = self
                            .eval_variable_value(&sql, text, v.span, a.value.span)
                            .await?;
                        value
                    };
                    pending.push((key.clone(), var.clone(), value));
                }
                for (key, var, value) in pending {
                    self.state.variables.put(
                        key,
                        StoredVariable {
                            data_type: var.data_type,
                            value,
                            default_query: var.default_query,
                        },
                    );
                }
                self.empty_frame().await
            }
            SetVariableKind::Row { targets, query } => {
                let mut keys = Vec::with_capacity(targets.len());
                for name in targets {
                    let Some(key) = variable_key(name) else {
                        return Err(unresolved_variable(name, RESOLVE_SEARCH_PATH, text));
                    };
                    keys.push(key);
                }
                duplicate_targets(&keys)?;
                let mut stored = Vec::with_capacity(keys.len());
                for key in &keys {
                    let Some(var) = self.state.variables.get(key) else {
                        return Err(VariableNotFound {
                            variable_name: quote_normalized(key),
                        }
                        .into());
                    };
                    stored.push(var);
                }
                // The row query's static output: the arity check needs it
                // even when the query returns no rows.
                let inner = original
                    .get(query.span.start..query.span.end)
                    .unwrap_or_default();
                let inner_sql = padded_select("", inner, query.span.start, "");
                let inner_analysis = self
                    .analyze_variable_query(&inner_sql)
                    .await
                    .map_err(|e| map_variable_error(e, text, v.span, query.span))?;
                let width = inner_analysis.analyzed.output.len();
                if width != targets.len() {
                    return Err(AssignmentArityMismatch {
                        num_target: targets.len().to_string(),
                        num_expr: width.to_string(),
                    }
                    .into());
                }
                let batches = self
                    .run_analyzed_variable_query(&inner_analysis)
                    .await
                    .map_err(|e| map_variable_error(e, text, v.span, query.span))?;
                let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
                if rows > 1 {
                    return Err(RowSubqueryTooManyRows {}.into());
                }
                if rows == 0 {
                    for (key, var) in keys.iter().zip(&stored) {
                        self.state.variables.put(
                            key.clone(),
                            StoredVariable {
                                data_type: var.data_type.clone(),
                                value: Literal::Null(Some(var.data_type.clone())),
                                default_query: var.default_query.clone(),
                            },
                        );
                    }
                    return self.empty_frame().await;
                }
                // One row: assign through casts, so values keep Spark's
                // store-assignment semantics and error classes.
                let prefix = format!(
                    "SELECT {} FROM (",
                    stored
                        .iter()
                        .enumerate()
                        .map(|(i, var)| format!(
                            "CAST(q._c{i} AS {}) AS _c{i}",
                            var.data_type.simple_string()
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                let aliases = (0..stored.len())
                    .map(|i| format!("_c{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let wrapper = format!("{prefix}{inner_sql}) AS q({aliases})");
                let wrapper_analysis = self
                    .analyze_variable_query(&wrapper)
                    .await
                    .map_err(|e| map_variable_error(e, text, v.span, query.span))?;
                let values = self
                    .run_analyzed_variable_query(&wrapper_analysis)
                    .await
                    .map_err(|e| map_variable_error(e, text, v.span, query.span))?;
                let total: usize = values.iter().map(RecordBatch::num_rows).sum();
                if total != 1 {
                    return Err(RowSubqueryTooManyRows {}.into());
                }
                let batch = values.iter().find(|b| b.num_rows() > 0).ok_or_else(|| {
                    crate::session::internal("a row assignment returned no batches".to_owned())
                })?;
                for ((key, var), i) in keys.iter().zip(&stored).zip(0..) {
                    let (_, value) = arrow_value(batch.column(i), 0)?;
                    self.state.variables.put(
                        key.clone(),
                        StoredVariable {
                            data_type: var.data_type.clone(),
                            value,
                            default_query: var.default_query.clone(),
                        },
                    );
                }
                self.empty_frame().await
            }
            // Later `SET VARIABLE` forms.
            _ => Err(KrishivUnsupportedFeature {
                feature: "this SET VARIABLE form".to_owned(),
            }
            .into()),
        }
    }

    /// Runs `DROP TEMPORARY VAR[IABLE] [IF EXISTS] name`, returning an
    /// empty frame as in Spark.
    pub(crate) async fn exec_drop_variable(
        &self,
        v: &DropVariable,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let Some(key) = variable_key(&v.name) else {
            return Err(unresolved_variable(&v.name, RESOLVE_SEARCH_PATH, text));
        };
        if self.state.variables.remove(&key).is_none() && !v.if_exists {
            return Err(VariableNotFound {
                variable_name: quote_normalized(&key),
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
    /// NULL), as Spark's `SET` commands return. Rows come out ordered by
    /// `key`, as in Spark.
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
        // Rows come out ordered by key: unions read in any order, so the
        // frame sorts.
        let df = df.order_by([col("key")]);
        let env = self.query_env();
        df.analyzed_in(env).await?;
        Ok(df)
    }
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Every part of a table name as written; `None` for a dynamic
/// `IDENTIFIER(...)` name or an empty one.
fn ddl_table_parts(name: &DdlName) -> Option<Vec<String>> {
    match name {
        DdlName::Parts(parts) if !parts.parts.is_empty() => {
            Some(parts.parts.iter().map(|p| p.value.clone()).collect())
        }
        _ => None,
    }
}

/// `TABLE_OR_VIEW_NOT_FOUND` quoting the whole multipart name, as Spark does.
fn table_not_found(parts: &[String]) -> KrishivError {
    TableOrViewNotFound {
        relation_name: parts.iter().map(|p| quote(p)).collect::<Vec<_>>().join("."),
        search_path: search_path(),
    }
    .into()
}

/// Column names as Spark lists them in errors: `` `a`, `b` ``.
fn list(names: &[String]) -> String {
    names
        .iter()
        .map(|n| quote(n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A repeated view column, as Spark 4.2 reports it with
/// `COLUMN_ALREADY_EXISTS`: ignoring case unless `case_sensitive`, then
/// the repeated name in lower case, mirroring
/// [`unique_columns`](crate::session) for writes.
fn unique_view_columns(columns: &[TableColumn], case_sensitive: bool) -> Result<(), KrishivError> {
    let mut seen = std::collections::HashSet::new();
    for c in columns {
        let name = if case_sensitive {
            c.name.clone()
        } else {
            c.name.to_lowercase()
        };
        if !seen.insert(name.clone()) {
            return Err(ColumnAlreadyExists {
                column_name: format!("`{}`", name.replace('`', "``")),
            }
            .into());
        }
    }
    Ok(())
}

/// Spark's search path, as analysis reports it for a missing table.
fn search_path() -> String {
    format!("[`system`.`builtin`, `system`.`session`, `{CATALOG}`.`{NAMESPACE}`]")
}

/// `name` on `search_path`, which it is not shaped like, as Spark reports
/// an unresolvable variable name: the written parts quoted, at the name.
fn unresolved_variable(name: &ObjectName, search_path: &str, text: &RedactedSql) -> KrishivError {
    KrishivError::from(UnresolvedVariable {
        variable_name: quote_name(name),
        search_path: search_path.to_owned(),
    })
    .with_query_context(ErrorQueryContext::sql(
        text.fragment(name.span.start..name.span.end),
        name.span.start,
        name.span.end.saturating_sub(1),
    ))
}

/// The duplicate targets of one assignment, as Spark reports them: the
/// lower-case name quoted, like `` `var1` ``.
fn duplicate_targets(keys: &[String]) -> Result<(), KrishivError> {
    let mut seen = HashSet::new();
    for key in keys {
        if !seen.insert(key) {
            return Err(DuplicateAssignments {
                name_list: format!("`{}`", key.replace('`', "``")),
            }
            .into());
        }
    }
    Ok(())
}

/// The parts of a plain `SET key = ...` assignment when the key is shaped
/// like a variable (`v`, `session.v` or `system.session.v`, in any letter
/// case): a backquoted key is one part, whatever dots it holds. The shape
/// check is [`variable_name`]'s, so resolution here cannot drift from the
/// analyzer's.
fn assign_variable_parts(key: &str, key_span: Span, text: &RedactedSql) -> Option<Vec<String>> {
    if key.is_empty() {
        return None;
    }
    let parts: Vec<String> = if text.fragment(key_span.start..key_span.end).starts_with('`') {
        vec![key.to_owned()]
    } else {
        key.split('.').map(str::to_owned).collect()
    };
    let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
    variable_name(&refs).is_some().then_some(parts)
}

/// `prefix + slice + suffix` with `slice` starting at its offset in the
/// command: spaces pad the gap, so error spans inside `slice` keep the
/// command's coordinates.
fn padded_select(prefix: &str, slice: &str, slice_start: usize, suffix: &str) -> String {
    let mut sql = String::new();
    sql.push_str(prefix);
    for _ in 0..slice_start.saturating_sub(prefix.len()) {
        sql.push(' ');
    }
    sql.push_str(slice);
    sql.push_str(suffix);
    sql
}

impl Session {
    /// The analysis of `sql` (a synthesized variable query): the row
    /// assignment's arity check needs the static output even when the
    /// query returns no rows.
    async fn analyze_variable_query(
        &self,
        sql: &str,
    ) -> Result<Arc<crate::dataframe::Analysis>, KrishivError> {
        let env = self.query_env();
        let redacted = krishiv_sql::redact_sql(sql);
        let query = krishiv_sql::parse_query(sql, &env.parser).map_err(|e| {
            crate::session::sql_context(e.error, &redacted, e.span.start, e.span.end)
        })?;
        let plan = krishiv_sql::lower(&query, SourceId(0)).map_err(|e| {
            crate::session::sql_context(e.error, &redacted, e.span.start, e.span.end)
        })?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        df.analyzed_in_with_text(env, Some(&redacted)).await
    }

    /// Runs analyzed `analysis` (from [`Self::analyze_variable_query`])
    /// through the physical planner and the executor, collecting its
    /// batches.
    async fn run_analyzed_variable_query(
        &self,
        analysis: &Arc<crate::dataframe::Analysis>,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let mut stream = self.run(&physical, &analysis.env).await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            out.push(batch?);
        }
        Ok(out)
    }

    /// The value of `sql` (a synthesized one-row `SELECT`): its single
    /// row and column as a typed literal.
    ///
    /// Execution errors carry no query context of their own; they are
    /// attached where Spark reports them: an assignment cast that reads
    /// malformed input at the command, an overflow nowhere (as Spark
    /// leaves it), anything else at the default or assigned expression.
    /// Errors that already have a context (analysis errors, whose aligned
    /// spans point into the command) are kept.
    async fn eval_variable_value(
        &self,
        sql: &str,
        text: &RedactedSql,
        command_span: Span,
        expr_span: Span,
    ) -> Result<(DataType, Literal), KrishivError> {
        let analysis = self
            .analyze_variable_query(sql)
            .await
            .map_err(|e| map_variable_error(e, text, command_span, expr_span))?;
        let batches = self
            .run_analyzed_variable_query(&analysis)
            .await
            .map_err(|e| map_variable_error(e, text, command_span, expr_span))?;
        let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
        if total != 1 {
            return Err(crate::session::internal(
                "a variable expression returned more than one row".to_owned(),
            ));
        }
        let batch = batches.iter().find(|b| b.num_rows() > 0).ok_or_else(|| {
            crate::session::internal("a variable expression returned no batches".to_owned())
        })?;
        if batch.num_columns() != 1 {
            return Err(crate::session::internal(
                "a variable expression returned more than one column".to_owned(),
            ));
        }
        arrow_value(batch.column(0), 0)
    }
}

/// Attaches the query context Spark reports for an execution error from a
/// variable default or assignment (see
/// [`Session::eval_variable_value`](Session::eval_variable_value)).
fn map_variable_error(
    error: KrishivError,
    text: &RedactedSql,
    command_span: Span,
    expr_span: Span,
) -> KrishivError {
    if !error.query_contexts().is_empty() {
        return error;
    }
    let at = |span: Span| {
        ErrorQueryContext::sql(
            text.fragment(span.start..span.end),
            span.start,
            span.end.saturating_sub(1),
        )
    };
    match error.error_class() {
        ErrorClass::CastInvalidInput => error.with_query_context(at(command_span)),
        ErrorClass::CastOverflow => error,
        _ => error.with_query_context(at(expr_span)),
    }
}

/// The value of `array` at `row` as a typed literal. The type comes from
/// the single [`arrow_krishiv_type`] map, so a NULL slot and a value never
/// disagree; anything the engine cannot hold in a variable fails there,
/// naming its Arrow type.
fn arrow_value(
    array: &arrow_array::ArrayRef,
    row: usize,
) -> Result<(DataType, Literal), KrishivError> {
    use arrow_array::types::{
        Date32Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type,
        TimestampMicrosecondType,
    };
    let data_type = arrow_krishiv_type(array.data_type())?;
    // A `NULL` array holds no validity bitmap (`is_null` is false), but
    // every slot is logically NULL.
    if matches!(array.data_type(), ArrowType::Null) || array.is_null(row) {
        let value = if data_type == DataType::Null {
            Literal::Null(None)
        } else {
            Literal::Null(Some(data_type.clone()))
        };
        return Ok((data_type, value));
    }
    let value = match array.data_type() {
        ArrowType::Boolean => Literal::Boolean(array.as_boolean().value(row)),
        ArrowType::Int8 => Literal::Byte(array.as_primitive::<Int8Type>().value(row)),
        ArrowType::Int16 => Literal::Short(array.as_primitive::<Int16Type>().value(row)),
        ArrowType::Int32 => Literal::Integer(array.as_primitive::<Int32Type>().value(row)),
        ArrowType::Int64 => Literal::Long(array.as_primitive::<Int64Type>().value(row)),
        ArrowType::Float32 => Literal::Float(
            array
                .as_primitive::<arrow_array::types::Float32Type>()
                .value(row),
        ),
        ArrowType::Float64 => Literal::Double(
            array
                .as_primitive::<arrow_array::types::Float64Type>()
                .value(row),
        ),
        ArrowType::Decimal128(_, _) => {
            let DataType::Decimal(decimal) = &data_type else {
                return Err(crate::session::internal(
                    "a variable decimal changed type".to_owned(),
                ));
            };
            Literal::Decimal {
                unscaled: array.as_primitive::<Decimal128Type>().value(row),
                data_type: *decimal,
            }
        }
        ArrowType::Utf8 => Literal::String(array.as_string::<i32>().value(row).to_owned()),
        ArrowType::LargeUtf8 => Literal::String(array.as_string::<i64>().value(row).to_owned()),
        ArrowType::Utf8View => Literal::String(array.as_string_view().value(row).to_owned()),
        ArrowType::Binary => Literal::Binary(array.as_binary::<i32>().value(row).to_owned()),
        ArrowType::LargeBinary => Literal::Binary(array.as_binary::<i64>().value(row).to_owned()),
        ArrowType::BinaryView => Literal::Binary(array.as_binary_view().value(row).to_owned()),
        ArrowType::Date32 => Literal::Date(array.as_primitive::<Date32Type>().value(row)),
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            let micros = array.as_primitive::<TimestampMicrosecondType>().value(row);
            match &data_type {
                DataType::Timestamp => Literal::Timestamp(micros),
                DataType::TimestampNtz => Literal::TimestampNtz(micros),
                _ => {
                    return Err(crate::session::internal(
                        "a variable timestamp changed type".to_owned(),
                    ));
                }
            }
        }
        // `arrow_krishiv_type` above already failed on any other type.
        _ => {
            return Err(crate::session::internal(
                "a variable value changed type".to_owned(),
            ));
        }
    };
    Ok((data_type, value))
}

/// The Krishiv type of Arrow `data_type`: the single Arrow-to-Krishiv
/// type map [`arrow_value`] reads, so NULL slots and values agree.
fn arrow_krishiv_type(data_type: &ArrowType) -> Result<DataType, KrishivError> {
    match data_type {
        ArrowType::Null => Ok(DataType::Null),
        ArrowType::Boolean => Ok(DataType::Boolean),
        ArrowType::Int8 => Ok(DataType::Byte),
        ArrowType::Int16 => Ok(DataType::Short),
        ArrowType::Int32 => Ok(DataType::Integer),
        ArrowType::Int64 => Ok(DataType::Long),
        ArrowType::Float32 => Ok(DataType::Float),
        ArrowType::Float64 => Ok(DataType::Double),
        ArrowType::Decimal128(precision, scale) => {
            decimal_type(*precision, *scale).map(DataType::Decimal)
        }
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => Ok(DataType::String),
        ArrowType::Binary | ArrowType::LargeBinary | ArrowType::BinaryView => Ok(DataType::Binary),
        ArrowType::Date32 => Ok(DataType::Date),
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some(_)) => {
            Ok(DataType::Timestamp)
        }
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, None) => {
            Ok(DataType::TimestampNtz)
        }
        other => Err(KrishivUnsupportedFeature {
            feature: format!("session variable values of type {other}"),
        }
        .into()),
    }
}

/// A decimal type from Arrow's precision and scale, for variable values.
fn decimal_type(precision: u8, scale: i8) -> Result<DecimalType, KrishivError> {
    DecimalType::try_new(
        u32::from(precision),
        u32::try_from(scale).map_err(|_| {
            crate::session::internal("an invalid decimal type in a variable".to_owned())
        })?,
    )
    .map_err(|e| crate::session::internal(format!("an invalid decimal type in a variable: {e}")))
}

/// Whether `e` is exactly the bare `DEFAULT` keyword (through
/// parentheses): `SET var = DEFAULT` reads the variable's default.
fn is_bare_default(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Nested(inner) => is_bare_default(inner),
        ExprKind::Column(parts) => {
            matches!(&parts[..], [p] if !p.quoted && p.value.eq_ignore_ascii_case("default"))
        }
        _ => false,
    }
}

/// Whether any node of the top-level expression tree `e` (windows
/// included) matches `pred`. Subquery plans are not crossed: a `Query` is
/// not an [`Expr`], so a `DEFAULT` inside one still resolves as a column
/// there, and a subquery inside one is not the top-level subquery these
/// checks look for.
fn expr_any(e: &Expr, pred: &dyn Fn(&Expr) -> bool) -> bool {
    let mut stack = vec![e];
    while let Some(node) = stack.pop() {
        if pred(node) {
            return true;
        }
        node.for_each_child(&mut |child| stack.push(child));
    }
    false
}

/// Whether the top-level expression tree holds a bare `DEFAULT` that is
/// not the whole value: there it cannot read a default, as in Spark.
fn contains_default(e: &Expr) -> bool {
    expr_any(e, &is_bare_default)
}

/// Whether the top-level expression tree holds a subquery: a default
/// cannot hold one, as in Spark.
fn contains_subquery(e: &Expr) -> bool {
    expr_any(e, &|node| {
        matches!(
            node.kind,
            ExprKind::Exists(_) | ExprKind::Subquery(_) | ExprKind::InSubquery { .. }
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use futures::executor::block_on;
    use krishiv_plan::SourceId;

    use super::*;

    /// The SQL source ids of the nodes of the stored view `name`.
    fn view_sources(session: &Session, name: &str) -> HashSet<SourceId> {
        let entry = session.state.catalog.get(name, false).unwrap();
        let crate::catalog::TempTable::View { plan, .. } = &*entry else {
            panic!("{name} is not a view");
        };
        let mut ids = HashSet::new();
        let mut stack = vec![Arc::clone(plan)];
        while let Some(plan) = stack.pop() {
            if let Some((source, _)) = plan.origin.sql {
                ids.insert(source);
            }
            plan.for_each_child(&mut |child| stack.push(Arc::clone(child)));
            plan.for_each_expr(&mut |expr| {
                if let krishiv_plan::ExprNode::Subquery { plan, .. } = &expr.node {
                    stack.push(Arc::clone(plan));
                }
            });
        }
        ids
    }

    #[test]
    fn views_lower_with_their_own_source_id() {
        // Every created view lowers with a source id of its own, so errors
        // raised while expanding a view point into the view's text, not the
        // outer statement's (`SourceId(0)`).
        let session = Session::new().unwrap();
        block_on(session.sql("CREATE TEMP VIEW v AS SELECT 1 AS a")).unwrap();
        block_on(session.sql("CREATE TEMP VIEW w AS SELECT 2 AS b")).unwrap();
        let (v, w) = (view_sources(&session, "v"), view_sources(&session, "w"));
        assert!(!v.is_empty() && !w.is_empty());
        assert!(!v.contains(&SourceId(0)), "{v:?}");
        assert!(!w.contains(&SourceId(0)), "{w:?}");
        assert!(v.is_disjoint(&w), "{v:?} vs {w:?}");
    }

    /// A window over `bound` as its frame start, for the expression walks.
    fn window_with(bound: Expr) -> krishiv_sql::ast::WindowRef {
        use krishiv_sql::ast::{
            FrameBound, FrameBoundKind, FrameUnits, Span, WindowFrame, WindowRef, WindowSpec,
        };
        WindowRef::Spec(Box::new(WindowSpec {
            partitioning: None,
            partition_by: Vec::new(),
            order_by: Vec::new(),
            frame: Some(WindowFrame {
                units: FrameUnits::Rows,
                start: FrameBound {
                    kind: FrameBoundKind::Preceding(bound),
                    span: Span::default(),
                },
                end: None,
                span: Span::default(),
            }),
            span: Span::default(),
        }))
    }

    /// A `f(1)` call over `window`, for the expression walks.
    fn function_over(window: krishiv_sql::ast::WindowRef) -> Expr {
        use krishiv_sql::ast::{Function, FunctionArg, Ident, ObjectName, Span};
        let ident = |value: &str| Ident {
            value: value.to_owned(),
            quoted: false,
            span: Span::default(),
        };
        Expr {
            kind: ExprKind::Function(Box::new(Function {
                name: ObjectName {
                    parts: vec![ident("f")],
                    span: Span::default(),
                },
                quantifier: None,
                args: vec![FunctionArg {
                    name: None,
                    expr: named_expr("1"),
                    span: Span::default(),
                }],
                within_group: Vec::new(),
                filter: None,
                null_treatment: None,
                over: Some(window),
                dynamic_name: None,
                span: Span::default(),
            })),
            span: Span::default(),
        }
    }

    /// Parses `sql` as a named expression, for the expression walks.
    fn named_expr(sql: &str) -> Expr {
        let options = krishiv_sql::ParserOptions::default();
        krishiv_sql::parse_named_expression(sql, &options)
            .unwrap()
            .expr
    }

    /// Variable names: shapes, quoting and normalization.
    #[test]
    fn variable_names_quote_and_normalize() {
        use krishiv_sql::ast::{Ident, Span};
        let name = |parts: &[&str]| ObjectName {
            parts: parts
                .iter()
                .map(|p| Ident {
                    value: (*p).to_owned(),
                    quoted: false,
                    span: Span::default(),
                })
                .collect(),
            span: Span::default(),
        };
        assert_eq!(variable_key(&name(&["var1"])), Some("var1".to_owned()));
        assert_eq!(
            variable_key(&name(&["SESSION", "Var1"])),
            Some("var1".to_owned())
        );
        assert_eq!(
            variable_key(&name(&["system", "SESSION", "vAr1"])),
            Some("var1".to_owned())
        );
        assert_eq!(variable_key(&name(&["ses", "var1"])), None);
        assert_eq!(variable_key(&name(&["a", "b", "c", "d"])), None);
        assert_eq!(variable_key(&name(&[])), None);
        assert_eq!(quote_name(&name(&["vAr1"])), "`vAr1`");
        assert_eq!(quote_name(&name(&["s", "v"])), "`s`.`v`");
        assert_eq!(quote_normalized("var1"), "`system`.`session`.`var1`");
    }

    /// The parts of a plain `SET` assignment shaped like a variable.
    #[test]
    fn assign_variable_parts_shapes() {
        use krishiv_sql::ast::Span;
        let parts = |key: &str, quoted: bool| {
            let sql = if quoted {
                format!("SET `{key}` = 1")
            } else {
                format!("SET {key} = 1")
            };
            let text = krishiv_sql::redact_sql(&sql);
            let span = Span::new(4, 4 + key.len() + usize::from(quoted) * 2);
            assign_variable_parts(key, span, &text)
        };
        assert_eq!(parts("var1", false), Some(vec!["var1".to_owned()]));
        assert_eq!(
            parts("SESSION.var1", false),
            Some(vec!["SESSION".to_owned(), "var1".to_owned()])
        );
        assert_eq!(
            parts("system.session.var1", false).map(|p| p.len()),
            Some(3)
        );
        assert_eq!(parts("x.var1", false), None);
        assert_eq!(parts("x.session.var1", false), None);
        assert_eq!(parts("", false), None);
        // Backquoted, the same dots are one part.
        assert_eq!(parts("x.var1", true), Some(vec!["x.var1".to_owned()]));
        // A backquoted key is one part, whatever dots it holds.
        assert_eq!(
            parts("session.var1", true),
            Some(vec!["session.var1".to_owned()])
        );
    }

    /// Duplicate targets report the first repeat, lower-cased and quoted.
    #[test]
    fn duplicate_targets_reports_first_repeat() {
        assert!(duplicate_targets(&["a".to_owned(), "b".to_owned()]).is_ok());
        let e = duplicate_targets(&["a".to_owned(), "B".to_owned(), "a".to_owned()]).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::DuplicateAssignments);
        assert_eq!(e.message_parameters(), &[("nameList", "`a`".to_owned())]);
    }

    /// `padded_select` pads the gap so an expression keeps its offset.
    #[test]
    fn padded_select_aligns_slices() {
        assert_eq!(padded_select("SELECT ", "1", 7, ""), "SELECT 1");
        assert_eq!(
            padded_select("SELECT CAST(", "1", 45, " AS INT)"),
            format!("SELECT CAST({}1 AS INT)", " ".repeat(45 - 12))
        );
        // Before the prefix: no padding, no panic.
        assert_eq!(padded_select("SELECT CAST(", "1", 2, ")"), "SELECT CAST(1)");
    }

    /// Every Arrow type the engine produces extracts to its literal.
    #[test]
    fn arrow_values_extract_every_type() {
        use arrow_array::{
            BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
            Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
            LargeStringArray, NullArray, StringArray, StringViewArray, TimestampMicrosecondArray,
        };
        let value = |array: arrow_array::ArrayRef| arrow_value(&array, 0).unwrap();
        assert_eq!(
            value(Arc::new(BooleanArray::from(vec![true]))),
            (DataType::Boolean, Literal::Boolean(true))
        );
        assert_eq!(
            value(Arc::new(Int8Array::from(vec![7]))),
            (DataType::Byte, Literal::Byte(7))
        );
        assert_eq!(
            value(Arc::new(Int16Array::from(vec![7]))),
            (DataType::Short, Literal::Short(7))
        );
        assert_eq!(
            value(Arc::new(Int32Array::from(vec![7]))),
            (DataType::Integer, Literal::Integer(7))
        );
        assert_eq!(
            value(Arc::new(Int64Array::from(vec![7]))),
            (DataType::Long, Literal::Long(7))
        );
        assert_eq!(
            value(Arc::new(Float32Array::from(vec![1.5]))),
            (DataType::Float, Literal::Float(1.5))
        );
        assert_eq!(
            value(Arc::new(Float64Array::from(vec![1.5]))),
            (DataType::Double, Literal::Double(1.5))
        );
        let decimal = Decimal128Array::from(vec![15])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert_eq!(
            value(Arc::new(decimal)),
            (
                DataType::Decimal(DecimalType::try_new(10, 2).unwrap()),
                Literal::Decimal {
                    unscaled: 15,
                    data_type: DecimalType::try_new(10, 2).unwrap(),
                }
            )
        );
        assert_eq!(
            value(Arc::new(StringArray::from(vec!["a"]))),
            (DataType::String, Literal::String("a".to_owned()))
        );
        assert_eq!(
            value(Arc::new(LargeStringArray::from(vec!["a"]))),
            (DataType::String, Literal::String("a".to_owned()))
        );
        assert_eq!(
            value(Arc::new(StringViewArray::from(vec!["a"]))),
            (DataType::String, Literal::String("a".to_owned()))
        );
        assert_eq!(
            value(Arc::new(BinaryArray::from(vec![b"a".as_slice()]))),
            (DataType::Binary, Literal::Binary(vec![b'a']))
        );
        assert_eq!(
            value(Arc::new(LargeBinaryArray::from(vec![b"a".as_slice()]))),
            (DataType::Binary, Literal::Binary(vec![b'a']))
        );
        assert_eq!(
            value(Arc::new(BinaryViewArray::from(vec![b"a".as_slice()]))),
            (DataType::Binary, Literal::Binary(vec![b'a']))
        );
        assert_eq!(
            value(Arc::new(Date32Array::from(vec![0]))),
            (DataType::Date, Literal::Date(0))
        );
        assert_eq!(
            value(Arc::new(
                TimestampMicrosecondArray::from(vec![1_000_000]).with_timezone("UTC")
            )),
            (DataType::Timestamp, Literal::Timestamp(1_000_000))
        );
        assert_eq!(
            value(Arc::new(TimestampMicrosecondArray::from(vec![1]))),
            (DataType::TimestampNtz, Literal::TimestampNtz(1))
        );
        // Untyped and typed nulls.
        assert_eq!(
            value(Arc::new(NullArray::new(1))),
            (DataType::Null, Literal::Null(None))
        );
        assert_eq!(
            value(Arc::new(Int32Array::from(vec![None]))),
            (DataType::Integer, Literal::Null(Some(DataType::Integer)))
        );
        assert_eq!(
            value(Arc::new(StringViewArray::from(vec![None as Option<&str>]))),
            (DataType::String, Literal::Null(Some(DataType::String)))
        );
        // Anything else names its Arrow type.
        let other: arrow_array::ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
        let _ = arrow_krishiv_type(other.data_type()).unwrap();
        let e = arrow_value(
            &{
                let array: arrow_array::ArrayRef =
                    Arc::new(arrow_array::DurationSecondArray::from(vec![1]));
                array
            },
            0,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        let e = decimal_type(99, 99).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }

    /// The `DEFAULT` walks: a bare `DEFAULT`, one nested in an
    /// expression, and subqueries.
    #[test]
    fn default_and_subquery_walks() {
        assert!(is_bare_default(&named_expr("DEFAULT")));
        assert!(is_bare_default(&named_expr("(DEFAULT)")));
        assert!(is_bare_default(&named_expr("default")));
        assert!(!is_bare_default(&named_expr("`default`")));
        assert!(!is_bare_default(&named_expr("session.default")));
        assert!(!is_bare_default(&named_expr("'Hello' || DEFAULT")));
        for sql in [
            "DEFAULT",
            "(DEFAULT)",
            "'Hello' || DEFAULT",
            "CASE WHEN TRUE THEN DEFAULT ELSE 1 END",
            "CAST(DEFAULT AS INT)",
            "upper(DEFAULT)",
        ] {
            assert!(contains_default(&named_expr(sql)), "{sql}");
        }
        for sql in ["1", "default2", "session.default", "(SELECT 1)"] {
            assert!(!contains_default(&named_expr(sql)), "{sql}");
        }
        // Through a window: the function, its filter, the ordering and
        // the frame all count.
        assert!(contains_default(&named_expr(
            "sum(x) OVER (ORDER BY DEFAULT)"
        )));
        assert!(contains_default(&function_over(window_with(named_expr(
            "DEFAULT"
        )))));
        assert!(!contains_default(&function_over(window_with(named_expr(
            "1"
        )))));
        for sql in [
            "(SELECT 1)",
            "1 + (SELECT 1)",
            "x IN (SELECT 1)",
            "EXISTS (SELECT 1)",
        ] {
            assert!(contains_subquery(&named_expr(sql)), "{sql}");
        }
        for sql in ["1", "DEFAULT", "'Hello' || DEFAULT"] {
            assert!(!contains_subquery(&named_expr(sql)), "{sql}");
        }
        // A subquery through a window frame counts too.
        assert!(contains_subquery(&function_over(window_with(named_expr(
            "(SELECT 1)"
        )))));
        assert!(!contains_subquery(&function_over(window_with(named_expr(
            "1"
        )))));
    }

    /// Execution errors from variable evaluation gain Spark's contexts.
    #[test]
    fn map_variable_error_attaches_contexts() {
        use krishiv_common::error::{CastInvalidInput, CastOverflow};
        use krishiv_sql::ast::Span;
        let text = krishiv_sql::redact_sql("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 'hello'");
        let command = Span::new(0, text.as_str().len());
        let expr = Span::new(45, 52);
        let at = |span: Span| {
            ErrorQueryContext::sql(
                text.fragment(span.start..span.end),
                span.start,
                span.end.saturating_sub(1),
            )
        };
        // A malformed assignment cast reports the command ...
        let e = map_variable_error(
            CastInvalidInput {
                expression: "'hello'".to_owned(),
                source_type: "\"STRING\"".to_owned(),
                target_type: "\"INT\"".to_owned(),
            }
            .into(),
            &text,
            command,
            expr,
        );
        assert_eq!(e.error_class(), ErrorClass::CastInvalidInput);
        assert_eq!(e.query_contexts(), &[at(command)]);
        // ... while an overflow keeps no context and anything else
        // reports the expression.
        let e = map_variable_error(
            CastOverflow {
                value: "1".to_owned(),
                source_type: "\"INT\"".to_owned(),
                target_type: "\"INT\"".to_owned(),
            }
            .into(),
            &text,
            command,
            expr,
        );
        assert!(e.query_contexts().is_empty());
        let e = map_variable_error(
            VariableNotFound {
                variable_name: "v".to_owned(),
            }
            .into(),
            &text,
            command,
            expr,
        );
        assert_eq!(e.query_contexts(), &[at(expr)]);
        // An error that already has a context keeps it.
        let e = map_variable_error(
            KrishivError::from(VariableNotFound {
                variable_name: "v".to_owned(),
            })
            .with_query_context(at(command)),
            &text,
            command,
            expr,
        );
        assert_eq!(e.query_contexts(), &[at(command)]);
        // `SET`, `DROP` and row forms resolve variables the same way.
        let name = |parts: &[&str]| ObjectName {
            parts: parts
                .iter()
                .map(|p| krishiv_sql::ast::Ident {
                    value: (*p).to_owned(),
                    quoted: false,
                    span: Span::default(),
                })
                .collect(),
            span: Span::new(28, 32),
        };
        let e = unresolved_variable(&name(&["builtin", "var1"]), RESOLVE_SEARCH_PATH, &text);
        assert_eq!(e.error_class(), ErrorClass::UnresolvedVariable);
        assert_eq!(
            e.query_contexts(),
            &[ErrorQueryContext::sql("var1", 28, 31)]
        );
    }

    /// One-row expressions evaluate; anything else is an internal error.
    #[test]
    fn eval_variable_value_needs_one_row_one_column() {
        let session = Session::new().unwrap();
        let text = krishiv_common::redact::RedactedSql::opaque("SELECT 1");
        let span = krishiv_sql::ast::Span::new(0, 8);
        let eval = |sql: &str| block_on(session.eval_variable_value(sql, &text, span, span));
        assert_eq!(
            eval("SELECT 7").unwrap(),
            (DataType::Integer, Literal::Integer(7))
        );
        for sql in [
            "SELECT 1 UNION ALL SELECT 2",
            "SELECT 1 LIMIT 0",
            "SELECT 1, 2",
        ] {
            assert_eq!(
                eval(sql).unwrap_err().error_class(),
                ErrorClass::InternalError,
                "{sql}"
            );
        }
    }
}
