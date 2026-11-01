// SPDX-License-Identifier: Apache-2.0
//! `SET`, `RESET`, `CREATE TEMPORARY VIEW`, `DROP VIEW`, `CREATE`/`DROP
//! TEMPORARY FUNCTION` and the session variable commands (`DECLARE`, `SET
//! VARIABLE`, `DROP TEMPORARY VARIABLE`) as [`Session::sql`](crate::Session::sql) runs them: parsing
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
    DefaultPlacementInvalid, DuplicateAssignments, DuplicateRoutineParameterNames,
    DuplicateRoutineReturnsColumns, DuplicateVariableNameInsideDeclare, ErrorClass,
    ErrorQueryContext, InvalidDefaultValueSubqueryExpression, InvalidSqlFunctionDataAccess,
    KrishivError, KrishivUnsupportedFeature, RoutineAlreadyExists, RoutineNotFound,
    RowSubqueryTooManyRows, SchemaNotFound, TableOrViewNotFound, TempTableOrViewAlreadyExists,
    UnresolvedVariable, UnsupportedCharOrVarcharAsString, UnsupportedFeatureSetPathWhenDisabled,
    UnsupportedFeatureSetVariableUsingSet, UserDefinedFunctionsCannotReplaceNonSqlUdfWithSqlUdf,
    UserDefinedFunctionsNotAValidDefaultExpression,
    UserDefinedFunctionsNotAValidDefaultParameterPosition,
    UserDefinedFunctionsNotNullOnFunctionParameters,
    UserDefinedFunctionsSqlTableUdfBodyMustBeAQuery, VariableAlreadyExists, VariableNotFound,
};
use krishiv_common::redact::RedactedSql;
use krishiv_plan::analyzer::{FunctionLookup, SqlFunction, SqlParam, TableColumn, variable_name};
use krishiv_plan::{
    Expr as PlanExpr, ExprNode, Literal, LogicalPlan, Origin, PlanNode, SourceId, SubqueryKind,
};
use krishiv_sql::ast::{
    Cache, CacheKind, ColumnDef, ColumnName, CreateFunction, CreateView, DataTypeKind,
    DataTypeSpec, DdlName, DeclareVariable, DropFunction, DropVariable, DropView, Expr, ExprKind,
    FunctionBodyKind, FunctionReturnsKind, Ident, ObjectName, Query, Refresh, RefreshKind, Reset,
    Set, SetExprKind, SetKind, SetVariable, SetVariableKind, Span, SqlDataAccess, TimeZone,
};
use krishiv_types::{DataType, DecimalType};

use crate::catalog::{CATALOG, NAMESPACE, TempTable};
use crate::catalog::{is_global_key, is_reserved_view_name, reserved_temp_view_name};
use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::functions::{col, lit, lit_null};
use crate::session::{QueryEnv, Session};
use crate::sql_functions::{Entry, one_row_project};
use crate::store::{CatalogStore, StoredFunction};
use crate::variables::{
    DECLARE_SEARCH_PATH, RESOLVE_SEARCH_PATH, StoredVariable, quote_name, quote_normalized,
    variable_key,
};

/// What `SET key` shows for a key with no value, as in Spark.
const UNDEFINED: &str = "<undefined>";

/// The configuration key `SET TIME ZONE` sets and shows, as in Spark.
const TIME_ZONE_KEY: &str = "spark.sql.session.timeZone";

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
                let rows: Vec<Vec<Option<String>>> = REGISTRY
                    .iter()
                    .map(|entry| {
                        let value = self
                            .state
                            .conf
                            .value(entry.key)
                            .unwrap_or_else(|| UNDEFINED.to_owned());
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
                let value = self
                    .state
                    .conf
                    .value(key)
                    .unwrap_or_else(|| UNDEFINED.to_owned());
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
                    TimeZone::Local => self.state.conf.local_zone().to_owned(),
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
            SetKind::Path { elements } => {
                let config = self.state.conf.snapshot();
                let enabled = config
                    .get(&krishiv_common::config::SQL_PATH_ENABLED)
                    .is_some_and(|v| v.to_string() == "true");
                if !enabled {
                    return Err(UnsupportedFeatureSetPathWhenDisabled {
                        config: "spark.sql.path.enabled".to_owned(),
                    }
                    .into());
                }
                let default_conf = config
                    .get(&krishiv_common::config::SQL_DEFAULT_PATH)
                    .map(ToString::to_string)
                    .unwrap_or_default();
                let namespace = self.state.catalog.current_namespace();
                let live = self
                    .state
                    .catalog
                    .stored_path()
                    .unwrap_or_else(|| crate::sql_path::default_entries(&default_conf));
                let entries = crate::sql_path::expand(elements, &live, &default_conf, &namespace)?;
                self.state.catalog.set_path(entries);
                self.empty_frame().await
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
        self.create_temp_view(v, false).await
    }

    /// `CREATE [OR REPLACE] [GLOBAL] TEMPORARY VIEW`: a global view lives in `global_temp`.
    pub(crate) async fn create_temp_view(
        &self,
        v: &CreateView,
        global: bool,
    ) -> Result<DataFrame, KrishivError> {
        let key = if global {
            crate::catalog::global_key(&v.name.value)
        } else {
            v.name.value.clone()
        };
        let shown = if global {
            format!(
                "{}.{}",
                quote(crate::catalog::GLOBAL_NAMESPACE),
                quote(&v.name.value)
            )
        } else {
            quote(&v.name.value)
        };
        if is_reserved_view_name(&v.name.value) {
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
                .exists(&key, env.analysis.conf.case_sensitive)
        {
            return Err(TempTableOrViewAlreadyExists {
                relation_name: shown,
            }
            .into());
        }
        let plan = krishiv_sql::lower(&v.query, self.next_source_id()).map_err(|e| e.error)?;
        crate::catalog::check_view_cycle(
            &self.state.catalog,
            &key,
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
            &key,
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
        if !is_global_key(&v.name.value)
            && self
                .state
                .catalog
                .exists(&v.name.value, env.analysis.conf.case_sensitive)
        {
            self.state.catalog.drop_table(&v.name.value);
            self.state.cache.invalidate_table(&v.name.value);
        } else {
            self.drop_bare_stored_view(&v.name.value, v.if_exists)?;
        }
        self.empty_frame().await
    }

    /// The name the cache knows the table `parts` by: a temporary view or table by its name, a
    /// permanent table as `namespace.name`. `None` when there is no such relation.
    fn cache_key(&self, parts: &[String]) -> Option<String> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let fold = |s: &str| {
            if case_sensitive {
                s.to_owned()
            } else {
                s.to_lowercase()
            }
        };
        let store = &self.state.catalog.store;
        match parts {
            [only] => {
                if self.state.catalog.exists(only, case_sensitive) {
                    return Some(only.clone());
                }
                let namespace = self.state.catalog.current_namespace();
                store
                    .table(&namespace, &fold(only))
                    .map(|_| format!("{namespace}.{}", fold(only)))
            }
            [namespace, name]
                if namespace.eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
                    && self
                        .state
                        .catalog
                        .exists(&crate::catalog::global_key(name), case_sensitive) =>
            {
                // The cache records a view by the last part of the name it was read by.
                Some(name.clone())
            }
            [namespace, name] => store
                .table(&fold(namespace), &fold(name))
                .map(|_| format!("{}.{}", fold(namespace), fold(name))),
            _ => None,
        }
    }

    /// Runs `CACHE [LAZY] TABLE name [AS query]`, `CLEAR CACHE` and `UNCACHE TABLE [IF EXISTS] name`,
    /// all returning an empty frame as in Spark. `CACHE TABLE` reads the relation once into the
    /// cache (`LAZY` and `OPTIONS` change nothing observable here); with `AS query` it first creates
    /// the temporary view `name`.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND` for `UNCACHE TABLE` of a missing table
    /// without `IF EXISTS`; `KRISHIV_UNSUPPORTED_FEATURE` for `CACHE TABLE`.
    pub(crate) async fn exec_cache(
        &self,
        c: &Cache,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
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
                if let Some(key) = self.cache_key(&parts) {
                    self.uncache_table(&key);
                } else if !if_exists {
                    return Err(table_not_found(&parts));
                }
            }
            CacheKind::Cache { name, query, .. } => {
                let Some(parts) = ddl_table_parts(name) else {
                    return Err(KrishivUnsupportedFeature {
                        feature: "CACHE TABLE with a dynamic table name".to_owned(),
                    }
                    .into());
                };
                // `CACHE TABLE t AS query` first makes the temporary view `t`.
                if let Some(q) = query {
                    let [view] = parts.as_slice() else {
                        return Err(table_not_found(&parts));
                    };
                    let body = original.get(q.span.start..q.span.end).unwrap_or_default();
                    let create = format!(
                        "CREATE TEMPORARY VIEW `{}` AS {body}",
                        view.replace('`', "``")
                    );
                    Box::pin(self.sql_here(&create)).await?;
                }
                // The relation is cached by reading it once: `LAZY` is the same here, since a
                // snapshot is taken the first time anything reads it either way.
                let frame = self
                    .table(
                        &parts
                            .iter()
                            .map(|p| format!("`{}`", p.replace('`', "``")))
                            .collect::<Vec<_>>()
                            .join("."),
                    )
                    .cache();
                // Counting reads every batch into the cache without keeping the rows.
                frame.count().await?;
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
                let Some(key) = self.cache_key(&parts) else {
                    return Err(table_not_found(&parts));
                };
                self.refresh_table(&key);
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
                let mut inner_sql = padded_select("", inner, query.span.start, "");
                let mut first = self.analyze_variable_query(&inner_sql).await;
                // A bare `DEFAULT` that is no column of the query reads the target's default.
                if let Err(e) = &first
                    && e.error_class().name().starts_with("UNRESOLVED_COLUMN")
                    && let Some(rewritten) = default_row_query(query, inner, &stored)
                {
                    inner_sql = padded_select("", &rewritten, query.span.start, "");
                    first = self.analyze_variable_query(&inner_sql).await;
                }
                let inner_analysis =
                    first.map_err(|e| map_variable_error(e, text, v.span, query.span))?;
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

    /// Runs `CREATE [OR REPLACE] TEMPORARY FUNCTION name (params) [RETURNS
    /// type] RETURN body` (`redacted` is the statement, redacted): checks the declaration as Spark does, then resolves a call of
    /// the new function once, with every parameter a typed `NULL` and every
    /// default applied, so that the body, the defaults and the return type
    /// are checked now and recursion is found (the call is made against
    /// the functions as they will be). The function is defined from the
    /// statement's text, which the body's spans index. Returns an empty
    /// frame as in Spark.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for persisted functions (without
    /// `TEMPORARY`), `LANGUAGE` other than SQL, `IDENTIFIER(...)` names
    /// and table functions (`RETURNS TABLE`); `ROUTINE_ALREADY_EXISTS`
    /// without `OR REPLACE`; Spark's `USER_DEFINED_FUNCTIONS.*` and
    /// `DUPLICATE_ROUTINE_*` errors for the declaration; whatever the
    /// analysis of the body raises, including `CYCLIC_FUNCTION_REFERENCE`.
    pub(crate) async fn exec_create_function(
        &self,
        f: &CreateFunction,
        redacted: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        if let Some(language) = &f.characteristics.language
            && !language.value.eq_ignore_ascii_case("SQL")
        {
            return Err(unsupported(&format!(
                "LANGUAGE {} functions",
                language.value
            )));
        }
        let (namespace, part) = if f.temporary {
            let [part] = &f.name.parts[..] else {
                // The parser rejects a qualified temporary function name.
                return Err(unsupported("qualified function names"));
            };
            (None, part)
        } else {
            let (namespace, part) = self.stored_function_name(&f.name.parts)?;
            (Some(namespace), part)
        };
        let name = part.value.to_lowercase();
        let conf = self.query_env().analysis.conf;
        let case_sensitive = conf.case_sensitive;
        let source = self.next_source_id();
        let mut entry = self.declare(f, &name, source, case_sensitive, conf.ansi, redacted)?;
        let reads = self.reads_data(&entry.def);
        if reads && f.characteristics.data_access == Some(SqlDataAccess::ContainsSql) {
            return Err(InvalidSqlFunctionDataAccess {}.into());
        }
        entry.def = Arc::new(SqlFunction {
            reads_data: reads,
            ..(*entry.def).clone()
        });
        if let Some(namespace) = &namespace {
            entry.def = Arc::new(SqlFunction {
                qualifier: vec![crate::catalog::CATALOG.to_owned(), namespace.clone()],
                path: self.query_env().catalog.sql_path_snapshot(),
                ..(*entry.def).clone()
            });
            return self
                .create_stored_function(f, namespace, part, entry, redacted)
                .await;
        }
        // A built-in or a registered UDF of the name is not ours to replace
        // or shadow (Spark: `ROUTINE_ALREADY_EXISTS`, or with `OR REPLACE`
        // `CANNOT_REPLACE_NON_SQL_UDF_WITH_SQL_UDF`).
        if !matches!(
            self.state
                .functions
                .lookup(std::slice::from_ref(&name), self.state.profile),
            FunctionLookup::NotFound
        ) {
            return Err(if f.or_replace {
                UserDefinedFunctionsCannotReplaceNonSqlUdfWithSqlUdf {
                    name: quote(&part.value),
                }
                .into()
            } else {
                RoutineAlreadyExists {
                    new_routine_type: "function".to_owned(),
                    routine_name: quote(&part.value),
                    existing_routine_type: "function".to_owned(),
                }
                .into()
            });
        }
        // Another statement may change the functions while the body is
        // checked; a few laps are enough in practice, and a statement that
        // keeps losing the race fails instead of looping.
        for _ in 0..MAX_DEFINE_ATTEMPTS {
            let current = self.state.sql_functions.snapshot();
            if !f.or_replace && current.get(&name).is_some() {
                return Err(RoutineAlreadyExists {
                    new_routine_type: "function".to_owned(),
                    routine_name: quote(&part.value),
                    existing_routine_type: "function".to_owned(),
                }
                .into());
            }
            let mut env = self.query_env();
            let candidate = Arc::new(
                env.sql_functions
                    .over(Arc::new(current.with(entry.clone()))),
            );
            env.analysis.functions = Arc::clone(&candidate) as _;
            env.sql_functions = Arc::clone(&candidate);
            env.functions_version = krishiv_plan::analyzer::FunctionRegistry::version(&*candidate);
            let output = self.probe(&entry.def, &[], env, redacted).await?;
            let mut entry = entry.clone();
            if entry.def.returns.is_none() {
                let def = SqlFunction {
                    returns: Some(output),
                    ..(*entry.def).clone()
                };
                entry.def = Arc::new(def);
            }
            // Another statement changed the functions while the body was
            // checked: check again against the new ones.
            if self.state.sql_functions.define(entry, current.version()) {
                return self.empty_frame().await;
            }
        }
        Err(crate::session::internal(
            "the session's functions kept changing while CREATE FUNCTION was checked".to_owned(),
        ))
    }

    /// The function `f` declares, as an entry: the declaration's own
    /// checks, and the lowering of its defaults and body.
    fn declare(
        &self,
        f: &CreateFunction,
        name: &str,
        source: krishiv_plan::SourceId,
        case_sensitive: bool,
        ansi: bool,
        redacted: &RedactedSql,
    ) -> Result<Entry, KrishivError> {
        let fold = |n: &str| {
            if case_sensitive {
                n.to_owned()
            } else {
                n.to_lowercase()
            }
        };
        let params: Vec<(String, &ColumnDef)> = f
            .params
            .iter()
            .map(|p| match &p.name {
                ColumnName::Ident(i) => Ok((i.value.clone(), p)),
                _ => Err(unsupported("IDENTIFIER(...) as a parameter name")),
            })
            .collect::<Result<_, _>>()?;
        let returned = match f.returns.as_ref().map(|r| &r.kind) {
            Some(FunctionReturnsKind::Table(columns)) => Some(columns.as_deref().unwrap_or(&[])),
            _ => None,
        };
        if f.params.iter().any(|p| p.not_null) {
            return Err(not_null(redacted, f.params_span));
        }
        if let Some(columns) = returned
            && let (Some(first), Some(last)) = (columns.first(), columns.last())
            && columns.iter().any(|c| c.not_null)
        {
            return Err(not_null(redacted, first.span.to(last.span)));
        }
        let duplicates = |names: Vec<String>| {
            let mut seen = std::collections::HashSet::new();
            let mut out: Vec<String> = Vec::new();
            for n in names.iter().map(|n| fold(n)) {
                if !seen.insert(n.clone()) && !out.contains(&n) {
                    out.push(n);
                }
            }
            out
        };
        let repeated = duplicates(params.iter().map(|(n, _)| n.clone()).collect());
        if !repeated.is_empty() {
            return Err(DuplicateRoutineParameterNames {
                routine_name: name.to_owned(),
                names: list(&repeated),
            }
            .into());
        }
        if let Some(columns) = returned {
            let repeated = duplicates(columns.iter().map(|c| c.name.value.clone()).collect());
            if !repeated.is_empty() {
                return Err(DuplicateRoutineReturnsColumns {
                    routine_name: name.to_owned(),
                    columns: list(&repeated),
                }
                .into());
            }
        }
        // A default is followed only by defaults.
        for (i, (n, p)) in params.iter().enumerate() {
            if p.default.is_some()
                && let Some((next, _)) = params[i + 1..].iter().find(|(_, q)| q.default.is_none())
            {
                return Err(UserDefinedFunctionsNotAValidDefaultParameterPosition {
                    function_name: name.to_owned(),
                    parameter_name: n.clone(),
                    next_parameter_name: next.clone(),
                }
                .into());
            }
        }
        let lower_err = |e: krishiv_sql::ParseError| e.error;
        let mut out_params = Vec::with_capacity(params.len());
        for (n, p) in &params {
            let default = match &p.default {
                Some(d) => {
                    let lowered = krishiv_sql::lower_expression(d, source).map_err(lower_err)?;
                    if has_subquery(&lowered) {
                        return Err(UserDefinedFunctionsNotAValidDefaultExpression {
                            function_name: name.to_owned(),
                            parameter_name: n.clone(),
                        }
                        .into());
                    }
                    Some(Arc::new(lowered))
                }
                None => None,
            };
            out_params.push(SqlParam {
                name: n.clone(),
                data_type: data_type(&p.data_type)?,
                default,
                comment: p.comment.clone(),
            });
        }
        let returns = match f.returns.as_ref().map(|r| &r.kind) {
            Some(FunctionReturnsKind::Type(t)) => Some(data_type(t)?),
            _ => None,
        };
        let body = match (&f.body.kind, returned.is_some()) {
            (FunctionBodyKind::Expr(_), true) => {
                return Err(UserDefinedFunctionsSqlTableUdfBodyMustBeAQuery {
                    name: name.to_owned(),
                }
                .into());
            }
            (FunctionBodyKind::Expr(e), false) => {
                krishiv_sql::lower_expression(e, source).map_err(lower_err)?
            }
            // A function's query body is a scalar subquery: for a scalar
            // function its value, for a table function its rows.
            (FunctionBodyKind::Query(q), _) => {
                let plan = krishiv_sql::lower(q, source).map_err(lower_err)?;
                PlanExpr::new(ExprNode::Subquery {
                    kind: SubqueryKind::Scalar,
                    plan: Arc::clone(plan.root()),
                })
                .with_origin(origin(source, f.body.span))
            }
            _ => return Err(unsupported("this kind of function body")),
        };
        let returns_table = match returned {
            Some(columns) => Some(
                columns
                    .iter()
                    .map(|c| Ok((c.name.value.clone(), data_type(&c.data_type)?)))
                    .collect::<Result<Vec<_>, KrishivError>>()?,
            ),
            None => None,
        };
        let def = SqlFunction {
            name: name.to_owned(),
            qualifier: Vec::new(),
            path: Vec::new(),
            returns_table,
            params: out_params,
            returns,
            body: Arc::new(body),
            body_text: redacted.fragment(f.body.span.range()).to_owned(),
            source,
            deterministic: f.characteristics.deterministic.unwrap_or(true),
            reads_data: f.characteristics.data_access == Some(SqlDataAccess::ReadsSqlData),
            ansi: Some(ansi),
        };
        Ok(Entry {
            def: Arc::new(def),
            text: redacted.clone(),
        })
    }

    /// Resolves `SELECT name(NULL, ...)` in `env`, calling `def` with a
    /// typed `NULL` for each parameter it needs (defaults fill the rest):
    /// the type of the result.
    async fn probe(
        &self,
        def: &SqlFunction,
        qualifier: &[String],
        env: QueryEnv,
        redacted: &RedactedSql,
    ) -> Result<DataType, KrishivError> {
        let args: Vec<PlanExpr> = def.params[..def.required()]
            .iter()
            .map(|p| PlanExpr::literal(Literal::Null(Some(p.data_type.clone()))))
            .collect();
        // A table function is probed by reading its rows; only the check
        // matters, its columns are declared.
        if def.returns_table.is_some() {
            let name: Vec<&str> = qualifier
                .iter()
                .map(String::as_str)
                .chain(std::iter::once(def.name.as_str()))
                .collect();
            let plan = Arc::new(LogicalPlan::new(PlanNode::TableFunction {
                name: name.join(&krishiv_plan::TABLE_FUNCTION_NAME_SEP.to_string()),
                args: args.into_iter().map(Arc::new).collect(),
            }));
            DataFrame::new(self.clone(), plan)
                .analyzed_in_with_text(env, Some(redacted))
                .await?;
            return Ok(DataType::Null);
        }
        let mut call = PlanExpr::function(def.name.clone(), args);
        if let ExprNode::Function(function) = &mut call.node {
            function.qualifier = qualifier.to_vec();
        }
        let plan = one_row_project(vec![Arc::new(call)]);
        let df = DataFrame::new(self.clone(), plan);
        let analysis = df.analyzed_in_with_text(env, Some(redacted)).await?;
        analysis
            .analyzed
            .output
            .first()
            .map(|c| c.data_type.clone())
            .ok_or_else(|| crate::session::internal("a function call without a result".to_owned()))
    }

    /// The namespace and name a stored function is named by: `f` (the
    /// current namespace), `ns.f` or `spark_catalog.ns.f`.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND` for a namespace the catalog lacks;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for another catalog or an
    /// `IDENTIFIER(...)` name.
    fn stored_function_name<'a>(
        &self,
        parts: &'a [Ident],
    ) -> Result<(String, &'a Ident), KrishivError> {
        let (namespace, part) = match parts {
            [part] => (self.state.catalog.current_namespace(), part),
            [ns, part] => (ns.value.clone(), part),
            [catalog, ns, part] if catalog.value.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                (ns.value.clone(), part)
            }
            _ => {
                return Err(unsupported(
                    "functions of catalogs other than spark_catalog",
                ));
            }
        };
        let namespace = if self.query_env().analysis.conf.case_sensitive {
            namespace
        } else {
            namespace.to_lowercase()
        };
        if !self.state.catalog.store.has_namespace(&namespace) {
            return Err(SchemaNotFound {
                schema_name: quote(&namespace),
            }
            .into());
        }
        Ok((namespace, part))
    }

    /// The first temporary SQL function the body or a default of `def`
    /// calls by a bare or session-qualified name (built-ins rank first).
    fn reads_temporary_function(&self, def: &SqlFunction) -> Option<String> {
        let temporary = self.state.sql_functions.snapshot();
        let mut stack: Vec<&PlanExpr> = std::iter::once(&*def.body)
            .chain(def.params.iter().filter_map(|p| p.default.as_deref()))
            .collect();
        while let Some(e) = stack.pop() {
            if let ExprNode::Function(call) = &e.node {
                let bare = call.qualifier.is_empty();
                let session = matches!(call.qualifier.as_slice(), [q] if q.eq_ignore_ascii_case("session"))
                    || matches!(call.qualifier.as_slice(), [s, q]
                        if s.eq_ignore_ascii_case("system") && q.eq_ignore_ascii_case("session"));
                let builtin = bare
                    && !matches!(
                        self.state
                            .functions
                            .lookup(std::slice::from_ref(&call.name), self.state.profile),
                        FunctionLookup::NotFound
                    );
                if (bare || session) && !builtin && temporary.get(&call.name).is_some() {
                    return Some(call.name.to_lowercase());
                }
            }
            if let ExprNode::Subquery { plan, .. } = &e.node {
                let mut plans = vec![plan];
                while let Some(p) = plans.pop() {
                    // A table function of the session, called by a bare or
                    // session-qualified name.
                    if let PlanNode::TableFunction { name, .. } = &p.node {
                        let parts: Vec<&str> =
                            name.split(krishiv_plan::TABLE_FUNCTION_NAME_SEP).collect();
                        let function = match parts.as_slice() {
                            [f] => Some(*f),
                            [q, f] if q.eq_ignore_ascii_case("session") => Some(*f),
                            [s, q, f]
                                if s.eq_ignore_ascii_case("system")
                                    && q.eq_ignore_ascii_case("session") =>
                            {
                                Some(*f)
                            }
                            _ => None,
                        };
                        if let Some(f) = function
                            && temporary
                                .get(f)
                                .is_some_and(|t| t.def.returns_table.is_some())
                        {
                            return Some(f.to_lowercase());
                        }
                    }
                    p.for_each_expr(&mut |x| stack.push(x));
                    p.for_each_child(&mut |c| plans.push(c));
                }
            }
            e.for_each_child(&mut |c| stack.push(c));
        }
        None
    }

    /// Whether the function reads data: its declaration says so, or says
    /// nothing and its body reads a relation (not `range` or `VALUES`) or
    /// calls a function that reads data.
    fn reads_data(&self, def: &SqlFunction) -> bool {
        if def.reads_data {
            return true;
        }
        let functions = self.query_env().sql_functions;
        let reads_calls = |name: &[String]| {
            matches!(
                krishiv_plan::analyzer::FunctionRegistry::lookup(&*functions, name, self.state.profile),
                FunctionLookup::Sql(d) if d.reads_data
            )
        };
        let mut stack: Vec<&PlanExpr> = std::iter::once(&*def.body)
            .chain(def.params.iter().filter_map(|p| p.default.as_deref()))
            .collect();
        let mut plans = Vec::new();
        while let Some(e) = stack.pop() {
            if let ExprNode::Function(call) = &e.node {
                let mut name = call.qualifier.clone();
                name.push(call.name.clone());
                if reads_calls(&name) {
                    return true;
                }
            }
            if let ExprNode::Subquery { plan, .. } = &e.node {
                plans.push(plan);
                while let Some(p) = plans.pop() {
                    if let PlanNode::TableFunction { name, .. } = &p.node {
                        let parts: Vec<String> = name
                            .split(krishiv_plan::TABLE_FUNCTION_NAME_SEP)
                            .map(str::to_owned)
                            .collect();
                        if reads_calls(&parts) {
                            return true;
                        }
                    }
                    p.for_each_expr(&mut |x| stack.push(x));
                    p.for_each_child(&mut |c| plans.push(c));
                }
            }
            e.for_each_child(&mut |c| stack.push(c));
        }
        crate::sql_functions::reads_relation(
            &def.body,
            self.query_env().analysis.conf.case_sensitive,
        )
    }

    /// `CREATE [OR REPLACE] FUNCTION` of a function stored in `namespace`
    /// of the session's catalog: checked and installed as a temporary one
    /// is, against the stored functions as they will be.
    async fn create_stored_function(
        &self,
        f: &CreateFunction,
        namespace: &str,
        part: &Ident,
        entry: Entry,
        redacted: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let store = &self.state.catalog.store;
        let qualified = [
            crate::catalog::CATALOG.to_owned(),
            namespace.to_owned(),
            part.value.clone(),
        ];
        // A stored function may not depend on an object of the session; Spark
        // names the function by namespace and name, and the object by kind.
        let object = format!("`{namespace}`.{}", quote(&entry.def.name));
        if let Some(temp) = crate::sql_functions::reads_temporary_view(
            &self.state.catalog,
            &entry.def.body,
            self.query_env().analysis.conf.case_sensitive,
        ) {
            return Err(krishiv_common::error::InvalidTempObjReference {
                obj_name: object,
                obj: "FUNCTION".to_owned(),
                temp_obj_name: format!("`{temp}`"),
                temp_obj: "VIEW".to_owned(),
            }
            .into());
        }
        if let Some(temp) = self.reads_temporary_function(&entry.def) {
            return Err(krishiv_common::error::InvalidTempObjReference {
                obj_name: object,
                obj: "FUNCTION".to_owned(),
                temp_obj_name: quote(&temp),
                temp_obj: "FUNCTION".to_owned(),
            }
            .into());
        }
        for _ in 0..MAX_DEFINE_ATTEMPTS {
            // A namespace dropped while the body was checked is not one to
            // store into.
            if !store.has_namespace(namespace) {
                return Err(SchemaNotFound {
                    schema_name: quote(namespace),
                }
                .into());
            }
            let current = store.functions();
            if !f.or_replace && current.get(namespace, &entry.def.name).is_some() {
                return Err(RoutineAlreadyExists {
                    new_routine_type: "function".to_owned(),
                    routine_name: quote_all(&qualified),
                    existing_routine_type: "function".to_owned(),
                }
                .into());
            }
            let stored = StoredFunction {
                namespace: namespace.to_owned(),
                entry: entry.clone(),
            };
            let mut env = self.query_env();
            let candidate = Arc::new(
                env.sql_functions
                    .over_stored(Arc::new(current.with(stored))),
            );
            env.analysis.functions = Arc::clone(&candidate) as _;
            env.sql_functions = Arc::clone(&candidate);
            env.functions_version = krishiv_plan::analyzer::FunctionRegistry::version(&*candidate);
            let output = self
                .probe(&entry.def, &entry.def.qualifier, env, redacted)
                .await?;
            let mut entry = entry.clone();
            if entry.def.returns.is_none() {
                entry.def = Arc::new(SqlFunction {
                    returns: Some(output),
                    ..(*entry.def).clone()
                });
            }
            let stored = StoredFunction {
                namespace: namespace.to_owned(),
                entry,
            };
            if store.define_function(stored, current.version()) {
                return self.empty_frame().await;
            }
        }
        Err(crate::session::internal(
            "the catalog's functions kept changing while CREATE FUNCTION was checked".to_owned(),
        ))
    }

    /// Runs `DROP [TEMPORARY] FUNCTION [IF EXISTS] name`, returning an
    /// empty frame as in Spark. Functions that call the dropped one fail
    /// when next used.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` without `TEMPORARY` (persisted
    /// functions); `ROUTINE_NOT_FOUND` for a function the session does not
    /// define, unless `IF EXISTS`.
    pub(crate) async fn exec_drop_function(
        &self,
        f: &DropFunction,
    ) -> Result<DataFrame, KrishivError> {
        // Temporary functions have one-part names: a qualified name is a
        // function that does not exist.
        let dropped = if f.temporary {
            match &f.name.parts[..] {
                [part] => self.state.sql_functions.drop_function(&part.value),
                _ => false,
            }
        } else {
            let (namespace, part) = self.stored_function_name(&f.name.parts)?;
            self.state
                .catalog
                .store
                .drop_function(&namespace, &part.value)
        };
        if !dropped && !f.if_exists {
            let parts: Vec<String> = f.name.parts.iter().map(|p| quote(&p.value)).collect();
            return Err(RoutineNotFound {
                routine_name: parts.join("."),
            }
            .into());
        }
        self.empty_frame().await
    }

    /// An empty result with no columns, as Spark's `CREATE VIEW`,
    /// `DROP VIEW` and `RESET` return.
    pub(crate) async fn empty_frame(&self) -> Result<DataFrame, KrishivError> {
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

    /// `SHOW ...`: only `SHOW CURRENT NAMESPACE` runs for now; the rest is
    /// unsupported.
    pub(crate) async fn exec_show(
        &self,
        s: &krishiv_sql::ast::Show,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        match &s.kind {
            krishiv_sql::ast::ShowKind::CurrentNamespace { .. } => {
                self.string_frame(
                    &["catalog", "namespace"],
                    vec![vec![
                        Some(CATALOG.to_owned()),
                        Some(self.state.catalog.current_namespace()),
                    ]],
                )
                .await
            }
            _ => self.exec_show_catalog(s, original).await,
        }
    }

    /// A frame of the string columns `columns` with `rows` (`None` is
    /// NULL), as Spark's `SET` commands return. Rows come out ordered by
    /// `key`, as in Spark.
    pub(crate) async fn string_frame(
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
        let df = if columns.contains(&"key") {
            df.order_by([col("key")])
        } else {
            df // callers without a `key` column pass at most one row
        };
        let env = self.query_env();
        df.analyzed_in(env).await?;
        Ok(df)
    }
}

/// How often `CREATE FUNCTION` checks a body again after a concurrent change.
const MAX_DEFINE_ATTEMPTS: usize = 16;

/// `KRISHIV_UNSUPPORTED_FEATURE` for `feature`.
fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

/// `NOT_NULL_ON_FUNCTION_PARAMETERS`, quoting `span` of the statement.
fn not_null(redacted: &RedactedSql, span: Span) -> KrishivError {
    UserDefinedFunctionsNotNullOnFunctionParameters {
        input: redacted.fragment(span.range()).to_owned(),
    }
    .into()
}

/// An origin in the statement `source` names.
fn origin(source: krishiv_plan::SourceId, span: Span) -> Origin {
    let at = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    Origin::sql(
        source,
        krishiv_plan::Span::new(at(span.start), at(span.end)),
    )
}

/// The type `spec` names. `CHAR` and `VARCHAR` are for table schemas only:
/// Spark rejects them in a function's signature.
fn data_type(spec: &DataTypeSpec) -> Result<DataType, KrishivError> {
    if has_char_or_varchar(spec) {
        return Err(UnsupportedCharOrVarcharAsString {}.into());
    }
    spec.to_data_type()
}

fn has_char_or_varchar(spec: &DataTypeSpec) -> bool {
    match &spec.kind {
        DataTypeKind::Primitive { name, .. } => matches!(
            name.split(' ').next().unwrap_or(name),
            "char" | "character" | "varchar"
        ),
        DataTypeKind::Array(Some(e)) => has_char_or_varchar(e),
        DataTypeKind::Map(Some(kv)) => has_char_or_varchar(&kv.0) || has_char_or_varchar(&kv.1),
        DataTypeKind::Struct(Some(fields)) => {
            fields.iter().any(|f| has_char_or_varchar(&f.data_type))
        }
        _ => false,
    }
}

/// Whether `e` holds a subquery.
fn has_subquery(e: &PlanExpr) -> bool {
    let mut found = false;
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if matches!(x.node, ExprNode::Subquery { .. }) {
            found = true;
            break;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    found
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Every part of a name as Spark quotes it in errors: `` `a`.`b` ``.
fn quote_all(parts: &[String]) -> String {
    parts.iter().map(|p| quote(p)).collect::<Vec<_>>().join(".")
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
pub(crate) fn unresolved_variable(
    name: &ObjectName,
    search_path: &str,
    text: &RedactedSql,
) -> KrishivError {
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
pub(crate) fn duplicate_targets(keys: &[String]) -> Result<(), KrishivError> {
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
pub(crate) fn padded_select(prefix: &str, slice: &str, slice_start: usize, suffix: &str) -> String {
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
    pub(crate) async fn analyze_variable_query(
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
    pub(crate) async fn run_analyzed_variable_query(
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

    /// As [`Self::run_analyzed_variable_query`], but stops reading at the first row beyond
    /// `max_rows` and returns `None`: a caller that holds the rows never holds more.
    pub(crate) async fn run_analyzed_variable_query_capped(
        &self,
        analysis: &Arc<crate::dataframe::Analysis>,
        max_rows: usize,
    ) -> Result<Option<Vec<RecordBatch>>, KrishivError> {
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let mut stream = self.run(&physical, &analysis.env).await?;
        let (mut out, mut rows) = (Vec::new(), 0_usize);
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            rows = rows.saturating_add(batch.num_rows());
            if rows > max_rows {
                return Ok(None);
            }
            out.push(batch);
        }
        Ok(Some(out))
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
    pub(crate) async fn eval_variable_value(
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
pub(crate) fn arrow_value(
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
pub(crate) fn arrow_krishiv_type(data_type: &ArrowType) -> Result<DataType, KrishivError> {
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
/// The text of the row query `query` (`inner`) with each bare `DEFAULT` of its select list replaced by
/// the default of the variable it is assigned to; `None` unless the query is a plain `SELECT` with one
/// item per target and at least one bare `DEFAULT`.
fn default_row_query(query: &Query, inner: &str, stored: &[StoredVariable]) -> Option<String> {
    let SetExprKind::Select(select) = &query.body.kind else {
        return None;
    };
    if select.projection.len() != stored.len()
        || !select.projection.iter().any(|i| is_bare_default(&i.expr))
    {
        return None;
    }
    let mut text = inner.to_owned();
    for (item, var) in select.projection.iter().zip(stored).rev() {
        if !is_bare_default(&item.expr) {
            continue;
        }
        let replacement = match &var.default_query {
            Some(q) => format!("({q})"),
            None => format!("CAST(NULL AS {})", var.data_type.simple_string()),
        };
        let start = item.expr.span.start.checked_sub(query.span.start)?;
        let end = item.expr.span.end.checked_sub(query.span.start)?;
        if !text.is_char_boundary(start) || !text.is_char_boundary(end) || start > end {
            return None;
        }
        text.replace_range(start..end, &replacement);
    }
    Some(text)
}

pub(crate) fn is_bare_default(e: &Expr) -> bool {
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
pub(crate) fn contains_default(e: &Expr) -> bool {
    expr_any(e, &is_bare_default)
}

/// Whether the top-level expression tree holds a subquery: a default
/// cannot hold one, as in Spark.
pub(crate) fn contains_subquery(e: &Expr) -> bool {
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
