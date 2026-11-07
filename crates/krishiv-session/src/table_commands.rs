// SPDX-License-Identifier: Apache-2.0
//! `CREATE TABLE`, `DROP TABLE`, `INSERT` and `TRUNCATE TABLE` on the
//! permanent catalog ([`crate::store`]).
//!
//! DDL is metadata only and never enters a plan. `CREATE TABLE` creates the
//! table's directory before it inserts the metadata, so metadata never
//! points at missing data; `DROP TABLE` removes the metadata first and then
//! deletes a managed table's directory. `INSERT` and `CREATE TABLE ... AS
//! SELECT` are writes: the query is analysed, stored into the table's
//! columns by Spark's ANSI store-assignment rules, and written as files to
//! the table's directory with the session's writer. A failed or cancelled
//! write keeps the old data; `INSERT OVERWRITE` deletes the old files only
//! once every new file is complete, but a reader listing the directory in
//! between sees old and new files together: atomic publication is the
//! manifest commit's (P2-T082b), as `krishiv-io`'s commit module says. A
//! CTAS adds its metadata only after the commit.
//!
//! Names of new tables and namespaces are checked as Spark checks them
//! (letters, digits and `_`), so a name can never be a path; everything the
//! commands delete lies inside the warehouse, is never a link, and is
//! authorised against the session's policy first.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use krishiv_common::error::{
    CastOverflowInTableInsert, DataSourceNotFound, ErrorClass,
    IncompatibleDataForTableCannotSafelyCast, InsertColumnArityMismatchNotEnoughDataColumns,
    InsertColumnArityMismatchTooManyDataColumns, InternalError, InvalidAttributeNameSyntax,
    InvalidSchemaOrRelationName, KrishivError, KrishivUnsupportedFeature, LocationAlreadyExists,
    NotSupportedCommandWithoutHiveSupport, RequiresSinglePartNamespace, SchemaAlreadyExists,
    SchemaNotEmpty, SchemaNotFound, TableOrViewNotFound, UnresolvedColumnWithSuggestion,
    UnsupportedFeatureDropDatabase, UnsupportedFeatureTableOperation, UnsupportedOverwriteTable,
    WrongCommandForObjectType,
};
use krishiv_common::names::check_column_name_duplication;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{Privilege, Securable, TableColumn, authorize};
use krishiv_sql::StatementClass;
use krishiv_sql::ast::{
    AlterColumnAction, AlterNamespace, AlterNamespaceAction, AlterTable, AlterTableAction,
    AlterView, AlterViewAction, Analyze, AnalyzeColumns, ChangeColumnAction, ColumnName,
    CreateNamespace, CreatePermanentView, CreateTable, CreateTableLike, CreateView, DdlName,
    DropBehavior, DropNamespace, DropPermanentView, DropTable, Ident, Insert, InsertClause, Query,
    TableElement, TableOption, Truncate, Use, ViewColumn,
};
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
use crate::functions::{col, expr, lit_null};
use crate::query_fold::Admitted;
use crate::session::{QueryEnv, Session};
use crate::store::{
    CatalogStore, CatalogTable, CreateMode, DEFAULT_NAMESPACE, DropNamespaceError, NamespaceMeta,
    StoreColumn, TableLayout, ViewDef, managed_location,
};

/// The file formats a table can have.
const PROVIDERS: &[&str] = &["parquet", "csv", "json", "text", "orc", "avro"];

/// Spark's `CatalogUtils.makeQualifiedDBObjectPath` on the local file system: a location with a URI scheme
/// stays as written; any other is resolved against the warehouse directory (an absolute path stays itself)
/// and qualified as a `file:` URI.
fn qualified_namespace_location(location: &str, warehouse: &std::path::Path) -> String {
    let scheme = location.split_once(':').is_some_and(|(s, _)| {
        s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    });
    if scheme {
        location.to_owned()
    } else {
        format!("file:{}", warehouse.join(location).display())
    }
}

/// The short name of a data source written fully qualified, as Spark maps its built-in classes
/// (`org.apache.spark.sql.parquet`, `com.databricks.spark.csv`); other names stay as written.
fn short_provider(written: &str) -> String {
    let lower = written.to_lowercase();
    for prefix in [
        "org.apache.spark.sql.execution.datasources.",
        "org.apache.spark.sql.",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let name = rest.split('.').next().unwrap_or(rest);
            if PROVIDERS.contains(&name) {
                return name.to_owned();
            }
        }
    }
    if lower == "com.databricks.spark.csv" {
        return "csv".to_owned();
    }
    lower
}

/// Whether `name` is a name Spark accepts for a new table or namespace: letters, digits and `_`.
fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// `INVALID_SCHEMA_OR_RELATION_NAME` (and for `.` and `..`, which Spark's parser rejects as attribute names,
/// `INVALID_ATTRIBUTE_NAME_SYNTAX`) unless `name` is a valid new table or namespace name.
fn check_new_name(name: &str) -> Result<(), KrishivError> {
    if valid_name(name) {
        Ok(())
    } else if name.chars().all(|c| c == '.') && !name.is_empty() {
        Err(InvalidAttributeNameSyntax {
            name: format!("`{name}`"),
        }
        .into())
    } else {
        Err(InvalidSchemaOrRelationName {
            name: format!("`{name}`"),
        }
        .into())
    }
}

/// `spark_catalog`.`ns`.`name`, as Spark prints a table.
fn quoted(namespace: &str, name: &str) -> String {
    format!("`spark_catalog`.`{namespace}`.`{name}`")
}

/// What a `DELETE`, `UPDATE` or `MERGE` asks of a table, for the checks that come before Spark finds
/// that the table cannot change.
pub(crate) struct RowChanges<'a> {
    /// The columns assigned.
    pub(crate) targets: &'a [&'a krishiv_sql::ast::ObjectName],
    /// The new values.
    pub(crate) values: &'a [&'a krishiv_sql::ast::Expr],
    /// The `WHERE` condition.
    pub(crate) condition: Option<&'a krishiv_sql::ast::Expr>,
}

/// The scalar subqueries anywhere in the expression.
fn scalar_subqueries<'a>(
    e: &'a krishiv_sql::ast::Expr,
    out: &mut Vec<&'a krishiv_sql::ast::Query>,
) {
    if let krishiv_sql::ast::ExprKind::Subquery(q) = &e.kind {
        out.push(q);
    }
    e.for_each_child(&mut |c| scalar_subqueries(c, out));
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

/// An `ALTER` found its table dropped or altered by another statement while it ran.
fn concurrent_change(name: &TableName) -> KrishivError {
    internal(format!(
        "{} was changed by another statement while this one ran; retry",
        name.quoted()
    ))
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// A table named in a command: its namespace and name, folded unless the
/// session is case sensitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TableName {
    pub(crate) namespace: String,
    pub(crate) name: String,
    /// The name as written, quoted part by part, for Spark's "not found" errors.
    pub(crate) written: String,
    /// The name had no namespace part.
    pub(crate) bare: bool,
}

impl TableName {
    pub(crate) fn resolve(
        parts: &[String],
        current: &str,
        env: &QueryEnv,
    ) -> Result<Self, KrishivError> {
        let fold = |s: &str| {
            if env.analysis.conf.case_sensitive {
                s.to_owned()
            } else {
                s.to_lowercase()
            }
        };
        let (namespace, name, bare) = match parts {
            [name] => (current.to_owned(), fold(name), true),
            [ns, name] => (fold(ns), fold(name), false),
            [catalog, ns, name] if catalog.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                (fold(ns), fold(name), false)
            }
            // No other catalog is registered, so a longer name is a namespace of several parts in
            // the session catalog, which takes one.
            _ => {
                return Err(KrishivError::from(RequiresSinglePartNamespace {
                    session_catalog: crate::catalog::CATALOG.to_owned(),
                    identifier: format!("`{}`", parts.join("`.`")),
                }));
            }
        };
        Ok(Self {
            namespace,
            name,
            written: format!("`{}`", parts.join("`.`")),
            bare,
        })
    }

    pub(crate) fn quoted(&self) -> String {
        quoted(&self.namespace, &self.name)
    }

    /// Spark's `TABLE_OR_VIEW_NOT_FOUND` for this name: as written, with the search path.
    pub(crate) fn not_found(&self) -> KrishivError {
        TableOrViewNotFound {
            relation_name: self.written.clone(),
            search_path: crate::catalog::search_path(),
        }
        .into()
    }
}

impl Session {
    pub(crate) fn table_name(
        &self,
        name: &DdlName,
        env: &QueryEnv,
    ) -> Result<TableName, KrishivError> {
        TableName::resolve(
            &crate::access::ddl_parts(name, env)?,
            &self.state.catalog.current_namespace(),
            env,
        )
    }

    pub(crate) fn warehouse_dir(&self) -> Result<PathBuf, KrishivError> {
        let configured = crate::conf::get(&self.state.conf.snapshot(), "spark.sql.warehouse.dir");
        self.state
            .catalog
            .warehouse
            .root(configured.as_deref())
            .map_err(|e| internal(format!("cannot create the warehouse directory: {e}")))
    }

    /// Whether `name` is a temporary view (or table) that a bare table name
    /// resolves to before any permanent table.
    pub(crate) fn temp_entry_of(&self, name: &TableName, env: &QueryEnv) -> bool {
        name.bare
            && !crate::catalog::is_global_key(&name.name)
            && self
                .state
                .catalog
                .exists(&name.name, env.analysis.conf.case_sensitive)
    }

    /// The namespace named by `name` (one part, or the session catalog and one part).
    pub(crate) fn namespace_name(
        &self,
        name: &DdlName,
        env: &QueryEnv,
    ) -> Result<String, KrishivError> {
        let parts = crate::access::ddl_parts(name, env)?;
        let fold = |s: &str| {
            if env.analysis.conf.case_sensitive {
                s.to_owned()
            } else {
                s.to_lowercase()
            }
        };
        match parts.as_slice() {
            [ns] => Ok(fold(ns)),
            [catalog, ns] if catalog.eq_ignore_ascii_case(crate::catalog::CATALOG) => Ok(fold(ns)),
            [catalog, _] => Err(unsupported(&format!(
                "namespaces in the catalog `{catalog}`"
            ))),
            _ => Err(RequiresSinglePartNamespace {
                session_catalog: crate::catalog::CATALOG.to_owned(),
                identifier: format!("`{}`", parts.join("`.`")),
            }
            .into()),
        }
    }

    /// `CREATE DATABASE|SCHEMA|NAMESPACE [IF NOT EXISTS] name ...`.
    pub(crate) async fn exec_create_namespace(
        &self,
        n: &CreateNamespace,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.namespace_name(&n.name, &env)?;
        check_new_name(&name)?;
        // `global_temp` is the reserved home of global temporary views: it always exists.
        if name.eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE) {
            if n.if_not_exists {
                return self.empty_frame().await;
            }
            return Err(SchemaAlreadyExists {
                schema_name: format!("`{name}`"),
            }
            .into());
        }
        let meta = NamespaceMeta {
            comment: n.comment.clone(),
            location: n.location.clone(),
            properties: options_of(&n.properties, original),
        };
        if !self.state.catalog.store.create_namespace(&name, meta) && !n.if_not_exists {
            return Err(SchemaAlreadyExists {
                schema_name: format!("`{name}`"),
            }
            .into());
        }
        self.empty_frame().await
    }

    /// `ALTER DATABASE|SCHEMA|NAMESPACE name SET PROPERTIES | SET LOCATION`, and `COMMENT ON SCHEMA`.
    pub(crate) async fn exec_alter_namespace(
        &self,
        a: &AlterNamespace,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.namespace_name(&a.name, &env)?;
        let change: Box<dyn Fn(&mut NamespaceMeta)> = match &a.action {
            AlterNamespaceAction::SetProperties(properties) => {
                let properties = options_of(properties, original);
                Box::new(move |m| m.properties.extend(properties.clone()))
            }
            AlterNamespaceAction::SetLocation(l) => {
                let l = qualified_namespace_location(l, &self.warehouse_dir()?);
                Box::new(move |m| m.location = Some(l.clone()))
            }
            AlterNamespaceAction::SetComment(c) => {
                let c = c.clone();
                Box::new(move |m| m.comment.clone_from(&c))
            }
            _ => return Err(unsupported("this ALTER SCHEMA action")),
        };
        let found = self.state.catalog.store.alter_namespace(&name, &*change);
        drop(change);
        if !found {
            return Err(SchemaNotFound {
                schema_name: format!("`{name}`"),
            }
            .into());
        }
        self.empty_frame().await
    }

    /// `DROP DATABASE|SCHEMA|NAMESPACE [IF EXISTS] name [RESTRICT|CASCADE]`.
    pub(crate) async fn exec_drop_namespace(
        &self,
        n: &DropNamespace,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.namespace_name(&n.name, &env)?;
        let cascade = matches!(n.behavior, Some(DropBehavior::Cascade));
        // Every check on every table's files comes before the first change, so a refused delete leaves the
        // namespace as it was.
        let mut dirs = Vec::new();
        if cascade {
            for t in self.state.catalog.store.tables_in(&name) {
                dirs.extend(self.deletable_dir(&t).await?);
            }
        }
        match self.state.catalog.store.drop_namespace(&name, cascade) {
            Ok(tables) => {
                for t in &tables {
                    self.state
                        .cache
                        .invalidate_table(&format!("{}.{}", t.namespace, t.name));
                }
                // The metadata is gone: leave the dropped namespace first, then try every directory and report the
                // first failure.
                if self.state.catalog.current_namespace() == name {
                    self.state.catalog.use_namespace(DEFAULT_NAMESPACE);
                }
                let mut first_error = None;
                for dir in dirs {
                    if let Err(e) = remove_dir(&dir) {
                        first_error.get_or_insert(e);
                    }
                }
                if let Some(e) = first_error {
                    return Err(e);
                }
            }
            Err(DropNamespaceError::Default) => {
                return Err(UnsupportedFeatureDropDatabase {
                    database: format!("`{name}`"),
                }
                .into());
            }
            Err(DropNamespaceError::Missing) if n.if_exists => {}
            Err(DropNamespaceError::Missing) => {
                return Err(SchemaNotFound {
                    schema_name: format!("`{name}`"),
                }
                .into());
            }
            Err(DropNamespaceError::NotEmpty) => {
                return Err(SchemaNotEmpty {
                    schema_name: format!("`{name}`"),
                }
                .into());
            }
        }
        self.empty_frame().await
    }

    /// `USE [NAMESPACE|DATABASE|SCHEMA] name`.
    pub(crate) async fn exec_use(&self, u: &Use) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.namespace_name(&u.name, &env)?;
        if !self.state.catalog.store.has_namespace(&name) {
            return Err(SchemaNotFound {
                schema_name: format!("`{name}`"),
            }
            .into());
        }
        self.state.catalog.use_namespace(&name);
        self.empty_frame().await
    }

    /// `CREATE [OR REPLACE] VIEW ...` for a permanent view.
    pub(crate) async fn exec_create_permanent_view(
        &self,
        v: &CreatePermanentView,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        if v.global && v.using.is_none() {
            return self.create_global_temp_view(v, &env).await;
        }
        let frozen_path = env.catalog.sql_path_snapshot();
        let frozen_namespace = env
            .catalog
            .namespace()
            .first()
            .cloned()
            .unwrap_or_else(|| self.state.catalog.current_namespace());
        if v.temporary && !v.global && v.using.is_some() && v.query.is_none() {
            return self.create_datasource_temp_view(v, &env, original).await;
        }
        if v.temporary || v.global {
            return Err(unsupported("qualified and datasource temporary views"));
        }
        let (None, Some(query)) = (&v.using, &v.query) else {
            return Err(unsupported("datasource views"));
        };
        let schema_mode = crate::view_schema::create_mode(
            v.schema_binding.map(|b| b.mode),
            !v.columns.is_empty(),
            &env.config,
        );
        let name = self.table_name(&v.name, &env)?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(&name)?;
        let mode = match (v.or_replace, v.if_not_exists) {
            (true, true) => {
                return Err(krishiv_common::error::CreateViewWithIfNotExistsAndReplace {
                    view_name: format!("`{}`", name.name),
                }
                .into());
            }
            (true, false) => CreateMode::Replace,
            (false, true) => CreateMode::IfNotExists,
            (false, false) => CreateMode::ErrorIfExists,
        };
        if let Some(existing) = self.state.catalog.store.table(&name.namespace, &name.name) {
            if mode == CreateMode::IfNotExists {
                return self.empty_frame().await;
            }
            if existing.view.is_none() && mode == CreateMode::Replace {
                return Err(krishiv_common::error::ExpectViewNotTableNoAlternative {
                    table_name: name.quoted(),
                    operation: "CREATE OR REPLACE VIEW".to_owned(),
                }
                .into());
            }
        }
        let sql = original
            .get(query.span.start..query.span.end)
            .unwrap_or_default()
            .to_owned();
        let plan = krishiv_sql::lower_with(
            query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
        )
        .map_err(|e| e.error)?;
        if let Some(temp) = crate::catalog::reads_temporary(
            &self.state.catalog,
            plan.root(),
            env.analysis.conf.case_sensitive,
        ) {
            return Err(krishiv_common::error::InvalidTempObjReference {
                obj_name: name.quoted(),
                obj: "view".to_owned(),
                temp_obj_name: format!("`{temp}`"),
                temp_obj: "view".to_owned(),
            }
            .into());
        }
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let case_sensitive = env.analysis.conf.case_sensitive;
        let analysis = df.analyzed_in(env).await?;
        let query_columns: Vec<String> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let mut columns: Vec<StoreColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| StoreColumn {
                name: c.name.clone(),
                data_type: krishiv_plan::without_generated_marker(&c.data_type),
                nullable: c.nullable,
                default: None,
                comment: None,
            })
            .collect();
        if !v.columns.is_empty() {
            let mut declared = Vec::with_capacity(v.columns.len());
            for alias in &v.columns {
                let ColumnName::Ident(ident) = &alias.name else {
                    return Err(unsupported("IDENTIFIER() column names"));
                };
                declared.push(ident.value.clone());
            }
            let quote = |n: &str| format!("`{}`", n.replace('`', "``"));
            let list = |names: &[String]| {
                names
                    .iter()
                    .map(|n| quote(n))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            match declared.len().cmp(&columns.len()) {
                std::cmp::Ordering::Greater => {
                    return Err(
                        krishiv_common::error::CreateViewColumnArityMismatchNotEnoughDataColumns {
                            view_name: name.quoted(),
                            view_columns: list(&declared),
                            data_columns: list(&query_columns),
                        }
                        .into(),
                    );
                }
                std::cmp::Ordering::Less => {
                    return Err(
                        krishiv_common::error::CreateViewColumnArityMismatchTooManyDataColumns {
                            view_name: name.quoted(),
                            view_columns: list(&declared),
                            data_columns: list(&query_columns),
                        }
                        .into(),
                    );
                }
                std::cmp::Ordering::Equal => {}
            }
            for (column, alias) in columns.iter_mut().zip(&v.columns) {
                let ColumnName::Ident(ident) = &alias.name else {
                    return Err(unsupported("IDENTIFIER() column names"));
                };
                ident.value.clone_into(&mut column.name);
                column.comment.clone_from(&alias.comment);
            }
        }
        // Spark checks the names after the aliases: a repeated one is `COLUMN_ALREADY_EXISTS`.
        check_column_name_duplication(columns.iter().map(|c| c.name.as_str()), case_sensitive)?;
        let view = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: false,
            provider: String::new(),
            provider_written: String::new(),
            location: PathBuf::new(),
            columns,
            options: BTreeMap::new(),
            properties: options_of(&v.tblproperties, original),
            comment: v.comment.clone(),
            created_ms: now_ms(),
            root: PathBuf::new(),
            view: Some(ViewDef {
                sql,
                namespace: frozen_namespace,
                path: frozen_path,
                query_columns,
                schema_mode,
            }),
            layout: TableLayout::default(),
        };
        self.state.catalog.store.create_table(view, mode)?;
        self.empty_frame().await
    }

    /// `DROP VIEW [IF EXISTS] name` for a permanent view (a temporary view of that bare name is dropped first).
    pub(crate) async fn exec_drop_permanent_view(
        &self,
        d: &DropPermanentView,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        if let Some(done) =
            self.drop_global_temp_view(&crate::access::ddl_parts(&d.name, &env)?, d.if_exists)
        {
            done?;
            return self.empty_frame().await;
        }
        let name = self.table_name(&d.name, &env)?;
        self.drop_stored_view(&name, d.if_exists)?;
        self.empty_frame().await
    }

    /// `CREATE [OR REPLACE] GLOBAL TEMPORARY VIEW name [(columns)] AS query`, kept in `global_temp`.
    async fn create_global_temp_view(
        &self,
        v: &CreatePermanentView,
        env: &QueryEnv,
    ) -> Result<DataFrame, KrishivError> {
        let parts = crate::access::ddl_parts(&v.name, env)?;
        let (Some(query), [name]) = (&v.query, parts.as_slice()) else {
            return Err(unsupported("qualified global temporary views"));
        };
        let columns = v
            .columns
            .iter()
            .map(|c| match &c.name {
                ColumnName::Ident(name) => Ok(ViewColumn {
                    name: name.clone(),
                    comment: c.comment.clone(),
                    span: c.span,
                }),
                _ => Err(unsupported("IDENTIFIER() column names in views")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let view = CreateView {
            or_replace: v.or_replace,
            if_not_exists: v.if_not_exists,
            name: Ident {
                value: name.clone(),
                quoted: false,
                span: v.span,
            },
            columns,
            collation: None,
            query: query.clone(),
            span: v.span,
        };
        self.create_temp_view(&view, true).await
    }

    /// `CREATE TEMPORARY VIEW name (col type, ...) USING format [OPTIONS (...)]`: a read of the
    /// format's files (the `path` option), typed by the declared columns. Nothing is read until a
    /// query uses the view.
    async fn create_datasource_temp_view(
        &self,
        v: &CreatePermanentView,
        env: &QueryEnv,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let parts = crate::access::ddl_parts(&v.name, env)?;
        let [name] = parts.as_slice() else {
            return Err(unsupported("qualified datasource temporary views"));
        };
        if crate::catalog::is_reserved_view_name(name) {
            return Err(crate::catalog::reserved_temp_view_name(name));
        }
        // The parser rejects IF NOT EXISTS on temporary views, as Spark does.
        let exists = self
            .state
            .catalog
            .exists(name, env.analysis.conf.case_sensitive);
        if exists && !v.or_replace {
            return Err(krishiv_common::error::TempTableOrViewAlreadyExists {
                relation_name: format!("`{name}`"),
            }
            .into());
        }
        let columns = v
            .columns
            .iter()
            .map(|c| {
                let (ColumnName::Ident(ident), Some(spec)) = (&c.name, &c.data_type) else {
                    return Err(unsupported(
                        "datasource views without declared column types",
                    ));
                };
                Ok(TableColumn {
                    name: ident.value.clone(),
                    data_type: spec.to_data_type()?,
                    nullable: !c.not_null,
                })
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        if columns.is_empty() {
            return Err(unsupported(
                "datasource views without declared column types",
            ));
        }
        let mut options = options_of(&v.options, original);
        let path_key = options
            .keys()
            .find(|k| k.eq_ignore_ascii_case("path"))
            .cloned();
        let paths = path_key
            .and_then(|k| options.remove(&k))
            .map(|p| vec![OptionValue::from_user("path", p)])
            .unwrap_or_default();
        // Spark resolves the data source when the view is created.
        let written = v
            .using
            .as_ref()
            .map(|u| {
                u.parts
                    .iter()
                    .map(|p| p.value.clone())
                    .collect::<Vec<_>>()
                    .join(".")
            })
            .unwrap_or_default();
        let provider = self.provider_of(Some(short_provider(&written)))?;
        // The policy is asked before the path is looked at, so a denial never tells whether it exists.
        for path in paths.iter().filter_map(OptionValue::as_plain) {
            let qualified = self.qualify_path(path)?;
            self.authorize_path(
                Path::new(&qualified),
                &[Privilege::Select],
                StatementClass::Ddl,
            )
            .await?;
        }
        if let Some(path) = paths.first().and_then(OptionValue::as_plain) {
            let local = path.strip_prefix("file:").unwrap_or(path);
            if !path.contains("://") && !std::path::Path::new(local).exists() {
                return Err(krishiv_common::error::PathNotFound {
                    path: format!("file:{local}"),
                }
                .into());
            }
        }
        let source = krishiv_plan::ReadSource::Files {
            format: provider,
            paths,
            options: options
                .into_iter()
                .map(|(k, val)| {
                    let val = OptionValue::from_user(&k, val);
                    (k, val)
                })
                .collect(),
        };
        self.state.catalog.put(
            name,
            crate::catalog::TempTable::Files {
                source,
                columns,
                known: true,
            },
        );
        // A replaced view must not keep serving the rows of the old one.
        self.state.cache.invalidate_table(name);
        self.empty_frame().await
    }

    /// `DROP VIEW global_temp.name`: the global temporary view, when the name is one.
    fn drop_global_temp_view(
        &self,
        parts: &[String],
        if_exists: bool,
    ) -> Option<Result<(), KrishivError>> {
        let [namespace, name] = parts else {
            return None;
        };
        if !namespace.eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE) {
            return None;
        }
        let key = crate::catalog::global_key(name);
        let cs = self.query_env().analysis.conf.case_sensitive;
        if self.state.catalog.exists(&key, cs) {
            self.state.catalog.drop_table(&key);
            self.state.cache.invalidate_table(name);
            return Some(Ok(()));
        }
        Some(if if_exists {
            Ok(())
        } else {
            Err(TableOrViewNotFound {
                relation_name: format!("`{}`.`{name}`", crate::catalog::GLOBAL_NAMESPACE),
                search_path: crate::catalog::search_path(),
            }
            .into())
        })
    }

    /// `DROP VIEW name` after no temporary view of that bare name was found: the stored view in the current
    /// namespace.
    pub(crate) fn drop_bare_stored_view(
        &self,
        name: &str,
        if_exists: bool,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        let name = TableName::resolve(
            &[name.to_owned()],
            &self.state.catalog.current_namespace(),
            &env,
        )?;
        self.drop_stored_view(&name, if_exists)
    }

    /// Drops the stored view `name`; a table of that name is `WRONG_COMMAND_FOR_OBJECT_TYPE`.
    fn drop_stored_view(&self, name: &TableName, if_exists: bool) -> Result<(), KrishivError> {
        match self.state.catalog.store.table(&name.namespace, &name.name) {
            Some(t) if t.view.is_some() => {
                self.forget_cached(name);
                self.state
                    .catalog
                    .store
                    .drop_table(&name.namespace, &name.name);
                Ok(())
            }
            Some(t) => Err(WrongCommandForObjectType {
                operation: "DROP VIEW".to_owned(),
                required_type: "VIEW".to_owned(),
                object_name: format!("{}.{}", name.namespace, name.name),
                found_type: if t.managed { "MANAGED" } else { "EXTERNAL" }.to_owned(),
                alternative: "DROP TABLE".to_owned(),
            }
            .into()),
            None if if_exists => Ok(()),
            None => Err(name.not_found()),
        }
    }

    /// `ALTER VIEW name ...`: `AS query`, `WITH SCHEMA ...` and `SET|UNSET TBLPROPERTIES`; `RENAME TO` is not
    /// executed yet.
    pub(crate) async fn exec_alter_view(
        &self,
        a: &AlterView,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let frozen_path = env.catalog.sql_path_snapshot();
        let frozen_namespace = env
            .catalog
            .namespace()
            .first()
            .cloned()
            .unwrap_or_else(|| self.state.catalog.current_namespace());
        let name = self.table_name(&a.name, &env)?;
        if matches!(a.action, AlterViewAction::SetSchemaBinding(_))
            && self.temp_entry_of(&name, &env)
        {
            return Err(
                krishiv_common::error::UnsupportedFeatureTemporaryViewWithSchemaBindingMode {}
                    .into(),
            );
        }
        let Some(existing) = self.state.catalog.store.table(&name.namespace, &name.name) else {
            return Err(name.not_found());
        };
        if existing.view.is_none() {
            return Err(krishiv_common::error::ExpectViewNotTableUseAlterTable {
                table_name: name.quoted(),
                operation: "ALTER VIEW".to_owned(),
            }
            .into());
        }
        let mut changed = (*existing).clone();
        match &a.action {
            AlterViewAction::SetQuery {
                query,
                schema_binding: None,
                ..
            } => {
                let plan = krishiv_sql::lower_with(
                    query,
                    self.next_source_id(),
                    self.parser_options().case_sensitive,
                )
                .map_err(|e| e.error)?;
                if let Some(temp) = crate::catalog::reads_temporary(
                    &self.state.catalog,
                    plan.root(),
                    env.analysis.conf.case_sensitive,
                ) {
                    return Err(krishiv_common::error::InvalidTempObjReference {
                        obj_name: name.quoted(),
                        obj: "view".to_owned(),
                        temp_obj_name: format!("`{temp}`"),
                        temp_obj: "view".to_owned(),
                    }
                    .into());
                }
                let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
                let analysis = df.analyzed_in(env).await?;
                changed.columns = analysis
                    .analyzed
                    .output
                    .iter()
                    .map(|c| StoreColumn {
                        name: c.name.clone(),
                        data_type: krishiv_plan::without_generated_marker(&c.data_type),
                        nullable: c.nullable,
                        default: None,
                        comment: None,
                    })
                    .collect();
                let query_columns = analysis
                    .analyzed
                    .output
                    .iter()
                    .map(|c| c.name.clone())
                    .collect();
                // `ALTER VIEW ... AS` keeps the mode (`AlterViewAsCommand`).
                let schema_mode = existing.view.as_ref().and_then(|v| v.schema_mode);
                changed.view = Some(ViewDef {
                    sql: original
                        .get(query.span.start..query.span.end)
                        .unwrap_or_default()
                        .to_owned(),
                    namespace: frozen_namespace,
                    path: frozen_path,
                    query_columns,
                    schema_mode,
                });
            }
            AlterViewAction::SetSchemaBinding(binding) => {
                // `AlterViewSchemaBindingCommand`: the mode only (`EVOLUTION` stores no query names: the view's
                // own names stand for them).
                let mode = crate::view_schema::from_ast(binding.mode);
                let names: Vec<String> = changed.columns.iter().map(|c| c.name.clone()).collect();
                if let Some(view) = &mut changed.view {
                    if mode == crate::store::ViewSchemaMode::Evolution {
                        view.query_columns = names;
                    }
                    view.schema_mode = Some(mode);
                }
                self.forget_cached(&name);
            }
            AlterViewAction::SetTblProperties(props) => {
                changed.properties.extend(options_of(props, original));
            }
            _ => return Err(unsupported("this ALTER VIEW")),
        }
        if !self.state.catalog.store.replace_table(&existing, changed) {
            return Err(concurrent_change(&name));
        }
        self.empty_frame().await
    }

    /// `ALTER TABLE name ...`: `ADD COLUMNS`, `RENAME TO`, `SET` and `UNSET TBLPROPERTIES`. Spark's file-source
    /// tables refuse the other changes (or they need the files rewritten), which fail as unsupported here.
    pub(crate) async fn exec_alter_table(
        &self,
        a: &AlterTable,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&a.table, &env)?;
        if self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .is_some_and(|t| t.view.is_some())
        {
            if matches!(a.action, AlterTableAction::SetComment(_)) {
                // Spark's error for a view here is not recorded in the goldens yet.
                return Err(unsupported("COMMENT ON TABLE of a view"));
            }
            return Err(krishiv_common::error::ExpectTableNotViewUseAlterView {
                operation: "`ALTER TABLE`".to_owned(),
                view_name: name.quoted(),
            }
            .into());
        }
        let operation = if matches!(a.action, AlterTableAction::SetComment(_)) {
            "COMMENT ON TABLE"
        } else {
            "ALTER TABLE"
        };
        let table = self.table_for_write(&name, operation, &env)?;
        let mut changed = (*table).clone();
        match &a.action {
            AlterTableAction::AddColumns(defs) => {
                let mut added = Vec::with_capacity(defs.len());
                for def in defs {
                    let ColumnName::Ident(ident) = &def.name else {
                        return Err(unsupported("IDENTIFIER() column names"));
                    };
                    added.push(ident);
                }
                // Spark checks the new and the existing names together (`AlterTableAddColumnsCommand`).
                check_column_name_duplication(
                    changed
                        .columns
                        .iter()
                        .map(|c| c.name.as_str())
                        .chain(added.iter().map(|i| i.value.as_str())),
                    env.analysis.conf.case_sensitive,
                )?;
                for (def, ident) in defs.iter().zip(added) {
                    // Spark alters the data schema: new columns go before the partition columns.
                    let at = changed.data_columns().count();
                    changed.columns.insert(
                        at,
                        StoreColumn {
                            name: ident.value.clone(),
                            data_type: def.data_type.to_data_type()?,
                            // Existing files have no such column: it reads as NULL.
                            nullable: true,
                            default: None,
                            comment: def.comment.clone(),
                        },
                    );
                }
            }
            AlterTableAction::ChangeColumn {
                name: column,
                action,
                ..
            } => {
                let change = match action {
                    ChangeColumnAction::Type(t) => ColumnChange::Type(t.to_data_type()?),
                    ChangeColumnAction::Comment(c) => ColumnChange::Comment(c.clone()),
                    ChangeColumnAction::First | ChangeColumnAction::After(_) => {
                        ColumnChange::Position
                    }
                    ChangeColumnAction::SetNotNull | ChangeColumnAction::DropNotNull => {
                        ColumnChange::Nullability
                    }
                    _ => return Err(unsupported("this CHANGE COLUMN")),
                };
                self.change_column(&mut changed, column, change, &env)?;
            }
            AlterTableAction::AlterColumn {
                name: column,
                action,
                ..
            } => {
                let change = match action {
                    AlterColumnAction::Comment(c) => ColumnChange::Comment(c.clone()),
                    AlterColumnAction::SetNotNull | AlterColumnAction::DropNotNull => {
                        ColumnChange::Nullability
                    }
                    AlterColumnAction::DropDefault => ColumnChange::DropDefault,
                    _ => return Err(unsupported("ALTER COLUMN ... SET DEFAULT")),
                };
                self.change_column(&mut changed, column, change, &env)?;
            }
            AlterTableAction::RenameColumn { from, .. } => {
                resolve_column(&changed, from, &env)?;
                return Err(table_operation(&name, "RENAME COLUMN"));
            }
            AlterTableAction::DropColumns(columns) => {
                for c in columns {
                    resolve_column(&changed, c, &env)?;
                }
                return Err(table_operation(&name, "DROP COLUMN"));
            }
            AlterTableAction::AddPartition {
                if_not_exists,
                parts,
                ..
            } => {
                changed = self
                    .add_partitions(&table, parts, *if_not_exists, original)
                    .await?;
            }
            AlterTableAction::SetTblProperties(props) => {
                // The `comment` property is the table comment (`AlterTableSetPropertiesCommand`); the
                // catalog shows it as `Comment`, not among the properties (Spark 4.2 golden
                // describe-table-after-alter-table).
                for (k, v) in options_of(props, original) {
                    if k == "comment" {
                        changed.comment = Some(v);
                    } else {
                        changed.properties.insert(k, v);
                    }
                }
            }
            AlterTableAction::SetComment(comment) => {
                changed.comment = comment.clone();
            }
            AlterTableAction::UnsetTblProperties { keys, .. } => {
                for key in options_of(keys, original).keys() {
                    changed.properties.remove(key);
                    // Spark keeps the table comment as the `comment` property.
                    if key == "comment" {
                        changed.comment = None;
                    }
                }
            }
            AlterTableAction::RenameTo { name: to, .. } => {
                return self.rename_table(&table, &self.table_name(to, &env)?).await;
            }
            _ => return Err(unsupported("this ALTER TABLE")),
        }
        if !self.state.catalog.store.replace_table(&table, changed) {
            return Err(concurrent_change(&name));
        }
        self.empty_frame().await
    }

    /// `ALTER TABLE ... CHANGE | ALTER COLUMN` on a file-source table (`ResolveSessionCatalog` and
    /// `AlterTableChangeColumnCommand`): only the comment changes, and the default can be dropped; the type
    /// must stay the same, positions and nullability are refused, and partition columns cannot change.
    fn change_column(
        &self,
        table: &mut CatalogTable,
        column: &ColumnName,
        change: ColumnChange,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let at = resolve_column(table, column, env)?;
        let name = TableName {
            namespace: table.namespace.clone(),
            name: table.name.clone(),
            written: String::new(),
            bare: false,
        };
        match change {
            ColumnChange::Position => {
                return Err(table_operation(&name, "ALTER COLUMN ... FIRST | AFTER"));
            }
            ColumnChange::Nullability => {
                return Err(table_operation(&name, "ALTER COLUMN ... SET NOT NULL"));
            }
            _ => {}
        }
        let written = match column {
            ColumnName::Ident(i) => i.value.clone(),
            _ => table.columns[at].name.clone(),
        };
        if table
            .layout
            .partition_columns
            .contains(&table.columns[at].name)
        {
            return Err(krishiv_common::error::CannotAlterPartitionColumn {
                table_name: name.quoted(),
                column_name: format!("`{}`", written.replace('`', "``")),
            }
            .into());
        }
        let current = &mut table.columns[at];
        match change {
            ColumnChange::Type(t) if t != current.data_type => {
                let sql = |t: &DataType| format!("\"{}\"", t.sql());
                let id = format!("`{}`", current.name.replace('`', "``"));
                return Err(krishiv_common::error::NotSupportedChangeColumn {
                    table: name.quoted(),
                    origin_name: id.clone(),
                    origin_type: sql(&current.data_type),
                    new_name: id,
                    new_type: sql(&t),
                }
                .into());
            }
            ColumnChange::Comment(c) => current.comment = Some(c),
            ColumnChange::DropDefault => current.default = None,
            _ => {}
        }
        Ok(())
    }

    /// `ALTER TABLE old RENAME TO new`: a managed table's directory moves first, then the metadata; a failure in
    /// between puts the directory back.
    async fn rename_table(
        &self,
        table: &CatalogTable,
        to: &TableName,
    ) -> Result<DataFrame, KrishivError> {
        // A bare target stays in the source's namespace, and a table never changes namespace (Spark 4.2).
        let mut to = to.clone();
        if to.bare {
            to.namespace.clone_from(&table.namespace);
        }
        let to = &to;
        check_new_name(&to.namespace)?;
        check_new_name(&to.name)?;
        self.check_namespace(to)?;
        if to.namespace != table.namespace {
            return Err(krishiv_common::error::LegacyErrorTemp1073 {
                db: table.namespace.clone(),
                new_db: to.namespace.clone(),
            }
            .into());
        }
        let store = &self.state.catalog.store;
        if store.table(&to.namespace, &to.name).is_some() {
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: to.quoted(),
            }
            .into());
        }
        let mut renamed = table.clone();
        renamed.namespace.clone_from(&to.namespace);
        renamed.name.clone_from(&to.name);
        if table.managed {
            let warehouse = self.warehouse_dir()?;
            renamed.location = managed_location(&warehouse, &to.namespace, &to.name);
            renamed.root = warehouse;
            // Partitions inside the table's directory move with it.
            for p in &mut renamed.layout.partitions {
                if let Ok(rest) = p.location.strip_prefix(&table.location) {
                    p.location = renamed.location.join(rest);
                }
            }
            // Authorized before any file moves: the source must be deletable and the target creatable.
            self.deletable_dir(table).await?;
            self.authorize_path(&renamed.location, &[Privilege::Create], StatementClass::Ddl)
                .await?;
            if let Some(parent) = renamed.location.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| internal(format!("cannot create {}: {e}", parent.display())))?;
            }
            std::fs::rename(&table.location, &renamed.location).map_err(|e| {
                internal(format!(
                    "cannot move {} to {}: {e}",
                    table.location.display(),
                    renamed.location.display()
                ))
            })?;
        }
        let moved_to = renamed.location.clone();
        self.state
            .cache
            .invalidate_table(&format!("{}.{}", table.namespace, table.name));
        store.drop_table(&table.namespace, &table.name);
        if let Err(e) = store.create_table(renamed, CreateMode::ErrorIfExists) {
            let _ = store.create_table(table.clone(), CreateMode::Replace);
            if table.managed {
                let _ = std::fs::rename(&moved_to, &table.location);
            }
            return Err(e);
        }
        self.empty_frame().await
    }

    /// `CREATE [OR REPLACE] [EXTERNAL] TABLE ...`, with or without columns
    /// and a query.
    pub(crate) async fn exec_create_table(
        &self,
        t: &CreateTable,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let warehouse = self.warehouse_dir()?;
        if t.temporary {
            return Err(unsupported("CREATE TEMPORARY TABLE"));
        }
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: "CREATE Hive TABLE (AS SELECT)".to_owned(),
            }
            .into());
        }
        if t.skew.is_some() || !t.cluster_by.is_empty() {
            return Err(unsupported("skewed or clustered tables"));
        }
        let (partition_names, typed_partitions) = partition_fields(&t.partitioned_by)?;
        if t.as_query.is_some() && !typed_partitions.is_empty() {
            return Err(unsupported(
                "partition column types in CREATE TABLE ... AS SELECT",
            ));
        }
        let using = t.using.as_ref().map(|u| {
            u.parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join(".")
        });
        let provider_written = self.written_provider(using.clone());
        let provider = self.provider_of(using)?;
        let name = self.table_name(&t.name, &env)?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(&name)?;
        if t.or_replace {
            // Spark's file-source tables cannot be replaced either.
            return Err(UnsupportedFeatureTableOperation {
                table_name: name.quoted(),
                operation: if t.as_query.is_some() {
                    "REPLACE TABLE AS SELECT"
                } else {
                    "REPLACE TABLE"
                }
                .to_owned(),
            }
            .into());
        }
        let mode = if t.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        if self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .is_none()
        {
            // A name taken by a temporary view is not a table; the table is
            // created beside it in its namespace, as in Spark.
        } else if mode == CreateMode::IfNotExists {
            return self.empty_frame().await;
        } else {
            // Before anything is written (a CTAS writes first).
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: name.quoted(),
            }
            .into());
        }
        crate::access::refuse_secret_options(&t.options, original)?;
        let mut options = options_of(&t.options, original);
        let properties = options_of(&t.tblproperties, original);
        // A `path` option is the table's location, as in Spark.
        let path_key = options
            .keys()
            .find(|k| k.eq_ignore_ascii_case("path"))
            .cloned();
        let location_text = match (&t.location, path_key) {
            (Some(l), _) => Some(l.clone()),
            (None, Some(k)) => options.remove(&k),
            (None, None) => None,
        };
        let external = t.external || location_text.is_some();
        let location = match &location_text {
            Some(l) if l.is_empty() => {
                return Err(krishiv_common::error::InvalidEmptyLocation {
                    location: l.clone(),
                }
                .into());
            }
            Some(l) => PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            None => managed_location(&warehouse, &name.namespace, &name.name),
        };
        let mut table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: !external,
            provider,
            provider_written,
            location,
            columns: Vec::new(),
            options,
            properties,
            comment: t.comment.clone(),
            created_ms: now_ms(),
            root: warehouse.clone(),
            view: None,
            layout: TableLayout::default(),
        };
        let layout = LayoutRequest {
            partition_by: &partition_names,
            bucket: t.bucket.as_ref(),
        };
        match &t.as_query {
            Some(query) => self.create_table_as(table, query, mode, &layout).await?,
            None => {
                table.columns = self.columns_of(t, original)?;
                let mut typed = Vec::with_capacity(typed_partitions.len());
                for c in &typed_partitions {
                    let ColumnName::Ident(ident) = &c.name else {
                        return Err(unsupported("IDENTIFIER() column names"));
                    };
                    typed.push(ident);
                }
                // A typed partition column is a column of the table's schema: checked with the others.
                check_column_name_duplication(
                    table
                        .columns
                        .iter()
                        .map(|c| c.name.as_str())
                        .chain(typed.iter().map(|i| i.value.as_str())),
                    self.query_env().analysis.conf.case_sensitive,
                )?;
                for (c, ident) in typed_partitions.iter().zip(typed) {
                    table.columns.push(StoreColumn {
                        name: ident.value.clone(),
                        data_type: c.data_type.to_data_type()?,
                        nullable: true,
                        default: None,
                        comment: c.comment.clone(),
                    });
                }
                if table.columns.is_empty() {
                    return Err(unsupported("CREATE TABLE without columns"));
                }
                krishiv_io::check_declared_schema(
                    &table.provider,
                    &arrow_schema::Schema::new(
                        table
                            .columns
                            .iter()
                            .map(|c| c.data_type.to_arrow_field(&c.name, true))
                            .collect::<Vec<_>>(),
                    ),
                )?;
                table.layout = crate::partitions::table_layout(
                    &name,
                    &mut table.columns,
                    layout.partition_by,
                    layout.bucket,
                    env.analysis.conf.case_sensitive,
                )?;
                self.add_table(table, mode, false).await?;
            }
        }
        self.empty_frame().await
    }

    /// The provider as written in `using`, or `spark.sql.sources.default`'s
    /// value: what a table's metadata prints (Spark keeps the case).
    fn written_provider(&self, using: Option<String>) -> String {
        using.unwrap_or_else(|| {
            crate::conf::get(&self.state.conf.snapshot(), "spark.sql.sources.default")
                .unwrap_or_else(|| "parquet".to_owned())
        })
    }

    /// The provider named by `using`, or `spark.sql.sources.default`.
    fn provider_of(&self, using: Option<String>) -> Result<String, KrishivError> {
        let provider = using.unwrap_or_else(|| {
            crate::conf::get(&self.state.conf.snapshot(), "spark.sql.sources.default")
                .unwrap_or_else(|| "parquet".to_owned())
        });
        let lower = provider.to_lowercase();
        if PROVIDERS.contains(&lower.as_str()) {
            Ok(lower)
        } else {
            Err(DataSourceNotFound { provider }.into())
        }
    }

    fn check_namespace(&self, name: &TableName) -> Result<(), KrishivError> {
        if self.state.catalog.store.has_namespace(&name.namespace) {
            Ok(())
        } else {
            Err(SchemaNotFound {
                schema_name: format!("`{}`", name.namespace),
            }
            .into())
        }
    }

    /// The columns of a `CREATE TABLE` from its elements.
    fn columns_of(
        &self,
        t: &CreateTable,
        original: &str,
    ) -> Result<Vec<StoreColumn>, KrishivError> {
        let mut columns: Vec<StoreColumn> = Vec::new();
        for element in &t.elements {
            let TableElement::Column(def) = element else {
                continue;
            };
            let ColumnName::Ident(ident) = &def.name else {
                return Err(unsupported("IDENTIFIER() column names"));
            };
            if def.generated.is_some() {
                return Err(unsupported("generated columns"));
            }
            columns.push(StoreColumn {
                name: ident.value.clone(),
                data_type: def.data_type.to_data_type()?,
                // File-source tables are nullable whatever the DDL says (Spark 4.2 verified: NOT NULL is dropped
                // and NULLs are stored), so nothing is enforced on insert.
                nullable: true,
                default: def.default.as_ref().map(|e| {
                    original
                        .get(e.span.start..e.span.end)
                        .unwrap_or_default()
                        .to_owned()
                }),
                comment: def.comment.clone(),
            });
        }
        check_column_name_duplication(
            columns.iter().map(|c| c.name.as_str()),
            self.query_env().analysis.conf.case_sensitive,
        )?;
        Ok(columns)
    }

    /// Adds `table` to the store, creating its directory first. A replaced
    /// managed table's directory is deleted after the metadata changed.
    async fn add_table(
        &self,
        table: CatalogTable,
        mode: CreateMode,
        written: bool,
    ) -> Result<(), KrishivError> {
        let name = TableName {
            namespace: table.namespace.clone(),
            name: table.name.clone(),
            written: String::new(),
            bare: false,
        };
        if mode == CreateMode::ErrorIfExists
            && self
                .state
                .catalog
                .store
                .table(&table.namespace, &table.name)
                .is_some()
        {
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: name.quoted(),
            }
            .into());
        }
        if !written && table.managed && table.location.exists() {
            return Err(LocationAlreadyExists {
                identifier: name.quoted(),
                location: table.location.display().to_string(),
            }
            .into());
        }
        let location = table.location.clone();
        self.authorize_path(&location, &[Privilege::Create], StatementClass::Ddl)
            .await?;
        // Spark creates the directory of a managed table only; an external table's location is
        // made by its first write.
        if table.managed {
            std::fs::create_dir_all(&location)
                .map_err(|e| internal(format!("cannot create {}: {e}", location.display())))?;
        }
        // Tables are never replaced, so nothing comes back from the store to clean up.
        self.state.catalog.store.create_table(table, mode)?;
        Ok(())
    }

    /// Asks the session's policy about `path` for each privilege (the writer does the same for its files).
    async fn authorize_path(
        &self,
        path: &Path,
        privileges: &[Privilege],
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        let securable = Securable::Path(OptionValue::from_user("path", path.display().to_string()));
        for privilege in privileges {
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                &securable,
                *privilege,
            )
            .await;
            crate::access::audit(self, class, r)?;
        }
        Ok(())
    }

    /// Deletes the files of the managed `table` (all of its directory, or with `keep_dir` what is in it).
    ///
    /// The directory must lie inside the warehouse, must not be the warehouse and must not be a link: a name
    /// can never reach outside it (names are checked at creation, and this is the second line). The policy is
    /// asked about the directory before anything is removed.
    async fn delete_table_files(
        &self,
        table: &CatalogTable,
        keep_dir: bool,
    ) -> Result<(), KrishivError> {
        if let Some(dir) = self.deletable_dir(table).await? {
            if keep_dir {
                clear_dir(&dir)
            } else {
                remove_dir(&dir)
            }
        } else {
            Ok(())
        }
    }

    /// The directory whose files a delete of `table` would remove, after every check (containment in the root
    /// the table was created under, no links, the policy): `None` when there is nothing to delete. Run before
    /// the metadata changes, so a refused delete leaves the table in place.
    async fn deletable_dir(&self, table: &CatalogTable) -> Result<Option<PathBuf>, KrishivError> {
        if !table.managed {
            return Ok(None);
        }
        let location = &table.location;
        let meta = match location.symlink_metadata() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(internal(format!(
                    "cannot inspect {}: {e}",
                    location.display()
                )));
            }
        };
        let root = table
            .root
            .canonicalize()
            .map_err(|e| internal(format!("cannot resolve the warehouse: {e}")))?;
        let canonical = location
            .canonicalize()
            .map_err(|e| internal(format!("cannot resolve {}: {e}", location.display())))?;
        if meta.file_type().is_symlink() || !canonical.starts_with(&root) || canonical == root {
            return Err(internal(format!(
                "refusing to delete {}: it is not a directory inside the warehouse",
                location.display()
            )));
        }
        self.authorize_path(&canonical, &[Privilege::Modify], StatementClass::Write)
            .await?;
        Ok(Some(canonical))
    }

    /// `CREATE TABLE ... AS SELECT`: writes the query to the table's
    /// location, then adds the metadata (so a failed write leaves none).
    async fn create_table_as(
        &self,
        table: CatalogTable,
        query: &Query,
        mode: CreateMode,
        layout: &LayoutRequest<'_>,
    ) -> Result<(), KrishivError> {
        let plan = krishiv_sql::lower_with(
            query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
        )
        .map_err(|e| e.error)?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        self.create_table_from(table, &df, mode, layout, &mut None)
            .await
    }

    /// Writes `df` to the location of the new `table` (whose columns are
    /// the frame's), then adds the metadata: the shared tail of `CREATE
    /// TABLE ... AS SELECT` and `writeTo(..).create()`.
    async fn create_table_from(
        &self,
        mut table: CatalogTable,
        df: &DataFrame,
        mode: CreateMode,
        layout: &LayoutRequest<'_>,
        pre: &mut Option<Admitted>,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        let analysis = df.analyzed_in(self.query_env()).await?;
        table.columns = analysis
            .analyzed
            .output
            .iter()
            .map(|c| StoreColumn {
                name: c.name.clone(),
                data_type: krishiv_plan::without_generated_marker(&c.data_type),
                nullable: true,
                default: None,
                comment: None,
            })
            .collect();
        let name = TableName {
            namespace: table.namespace.clone(),
            name: table.name.clone(),
            written: String::new(),
            bare: false,
        };
        table.layout = crate::partitions::table_layout(
            &name,
            &mut table.columns,
            layout.partition_by,
            layout.bucket,
            env.analysis.conf.case_sensitive,
        )?;
        if table.managed && table.location.exists() {
            return Err(LocationAlreadyExists {
                identifier: quoted(&table.namespace, &table.name),
                location: table.location.display().to_string(),
            }
            .into());
        }
        // The partition columns go last, in the table's order.
        let df = if table.is_partitioned() {
            df.select(
                table
                    .columns
                    .iter()
                    .map(|c| col(&format!("`{}`", c.name.replace('`', "``")))),
            )
        } else {
            df.clone()
        };
        // A managed table's directory is made here, exclusively: that creation is the proof that this
        // statement owns it (a concurrent CTAS of the same table loses it and fails with
        // `LOCATION_ALREADY_EXISTS`), and the guard removes it again if the statement fails or is
        // cancelled, so a failed first write (and the commit log it started) leaves nothing and the
        // statement can be retried. The write goes into that empty directory (`append`).
        // An external location keeps `errorifexists`: a CTAS never replaces files already there.
        // The policy is asked about the destination first: a denied CTAS must not create the warehouse or
        // the table directory on the way to being refused (the writer asks again for the files it writes).
        if table.managed {
            self.authorize_path(
                &table.location,
                &[Privilege::Modify, Privilege::Create],
                StatementClass::Write,
            )
            .await?;
        }
        let owned = if table.managed {
            Some(
                OwnedDirectory::create(table.location.clone()).map_err(|e| match e {
                    OwnedDirectoryError::Exists => KrishivError::from(LocationAlreadyExists {
                        identifier: quoted(&table.namespace, &table.name),
                        location: table.location.display().to_string(),
                    }),
                    OwnedDirectoryError::Io(message) => internal(message),
                })?,
            )
        } else {
            None
        };
        let write_mode = if owned.is_some() {
            "append"
        } else {
            "errorifexists"
        };
        let dirs = self.write_table_rows(&table, &df, write_mode, pre).await?;
        if let Some(owned) = owned {
            owned.keep();
        }
        self.add_table(table, mode, true).await?;
        // `RepairTableCommand`: the partitions written, with their file statistics.
        self.register_partitions(&name, &dirs, true);
        Ok(())
    }

    /// `CREATE TABLE target LIKE source`.
    pub(crate) async fn exec_create_table_like(
        &self,
        t: &CreateTableLike,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let warehouse = self.warehouse_dir()?;
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: "CREATE Hive TABLE (AS SELECT)".to_owned(),
            }
            .into());
        }
        let name = self.table_name(&t.name, &env)?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(&name)?;
        let source = self.table_name(&t.source, &env)?;
        // Spark's `CreateTableLikeCommand` copies the partition columns, the bucket specification and the
        // storage options, under any provider.
        let mut layout = TableLayout::default();
        let mut options = BTreeMap::new();
        let (columns, source_provider) = if let Some(s) = self
            .state
            .catalog
            .store
            .table(&source.namespace, &source.name)
        {
            layout
                .partition_columns
                .clone_from(&s.layout.partition_columns);
            layout.bucket.clone_from(&s.layout.bucket);
            options.clone_from(&s.options);
            // A view has no provider of its own: Spark gives the new table
            // `spark.sql.sources.default` (`CreateTableLikeCommand`, `isViewLike`).
            let provider = s
                .view
                .is_none()
                .then(|| (s.provider.clone(), s.provider_written.clone()));
            (s.columns.clone(), provider)
        } else if self.temp_entry_of(&source, &env)
            && let Some(entry) = self
                .state
                .catalog
                .get(&source.name, env.analysis.conf.case_sensitive)
        {
            let columns = match &*entry {
                crate::catalog::TempTable::Memory { columns, .. }
                | crate::catalog::TempTable::Files { columns, .. }
                | crate::catalog::TempTable::View { columns, .. } => columns,
            };
            (
                columns
                    .iter()
                    .map(|c| StoreColumn {
                        name: c.name.clone(),
                        data_type: krishiv_plan::without_generated_marker(&c.data_type),
                        nullable: c.nullable,
                        default: None,
                        comment: None,
                    })
                    .collect(),
                None,
            )
        } else {
            return Err(source.not_found());
        };
        let (provider, provider_written) = match &t.using {
            Some(u) => {
                let using = u
                    .parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                (self.provider_of(Some(using.clone()))?, using)
            }
            // `None`: a view, which has no provider of its own. The default
            // is looked up only here, so an explicit `USING` never depends on it.
            None => match source_provider {
                Some(p) => p,
                // Validated, and written as the setting has it (`CSV`).
                None => (self.provider_of(None)?, self.written_provider(None)),
            },
        };
        let external = t.location.is_some();
        let location = match &t.location {
            Some(l) => PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            None => managed_location(&warehouse, &name.namespace, &name.name),
        };
        let table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: !external,
            provider,
            provider_written,
            location,
            columns,
            options,
            properties: options_of(&t.tblproperties, original),
            comment: None,
            created_ms: now_ms(),
            root: warehouse.clone(),
            view: None,
            layout,
        };
        let mode = if t.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        self.add_table(table, mode, false).await?;
        self.empty_frame().await
    }

    /// `DROP TABLE [IF EXISTS] name [PURGE]`.
    pub(crate) async fn exec_drop_table(&self, d: &DropTable) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&d.name, &env)?;
        self.drop_named_table(&name, &env, d.if_exists).await?;
        self.empty_frame().await
    }

    /// `DROP TABLE` for a resolved name; also what `saveAsTable` in overwrite
    /// mode runs before it creates the table again.
    async fn drop_named_table(
        &self,
        name: &TableName,
        env: &QueryEnv,
        if_exists: bool,
    ) -> Result<(), KrishivError> {
        // `DROP TABLE` on a permanent view is refused as Spark's `DropTableCommand` does.
        if self.temp_entry_of(name, env) {
            // A temporary view shadows a table of the same bare name, and `DROP TABLE` drops it, as in Spark.
            self.state.catalog.drop_table(&name.name);
            self.state.cache.invalidate_table(&name.name);
        } else if let Some(table) = self.state.catalog.store.table(&name.namespace, &name.name) {
            if table.view.is_some() {
                return Err(WrongCommandForObjectType {
                    operation: "DROP TABLE".to_owned(),
                    required_type: "EXTERNAL or MANAGED".to_owned(),
                    object_name: format!("{}.{}", name.namespace, name.name),
                    found_type: "VIEW".to_owned(),
                    alternative: "DROP VIEW".to_owned(),
                }
                .into());
            }
            // Checked before the metadata goes: a refused delete must not orphan the files.
            let dir = self.deletable_dir(&table).await?;
            self.forget_cached(name);
            self.state
                .catalog
                .store
                .drop_table(&name.namespace, &name.name);
            if let Some(dir) = dir {
                remove_dir(&dir)?;
            }
        } else if !if_exists {
            return Err(name.not_found());
        }
        Ok(())
    }

    /// `TRUNCATE TABLE name`: removes the table's files and keeps its
    /// directory.
    pub(crate) async fn exec_truncate(&self, t: &Truncate) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        if t.partition.is_some() {
            return Err(unsupported("TRUNCATE TABLE ... PARTITION"));
        }
        let name = self.table_name(&t.table, &env)?;
        let table = self.table_for_write(&name, "TRUNCATE TABLE", &env)?;
        if !table.managed {
            return Err(KrishivUnsupportedFeature {
                feature: "TRUNCATE TABLE on an external table".to_owned(),
            }
            .into());
        }
        self.forget_cached(&name);
        let deleted = self.delete_table_files(&table, true).await;
        // Data may have changed even when the cleanup failed part way.
        self.state.catalog.store.bump();
        self.forget_cached(&name);
        deleted?;
        self.empty_frame().await
    }

    /// `ANALYZE TABLE name COMPUTE STATISTICS [NOSCAN | FOR COLUMNS ...]`.
    ///
    /// The table (and the listed columns) must exist, as in Spark. The engine
    /// keeps no table statistics, so nothing is computed or stored: the
    /// command only validates its target.
    pub(crate) async fn exec_analyze(
        &self,
        a: &Analyze,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&a.table, &env)?;
        let table = self.table_for_write(&name, "ANALYZE TABLE", &env)?;
        if let Some(spec) = &a.partition {
            if !matches!(a.columns, AnalyzeColumns::All) {
                return Err(unsupported("ANALYZE TABLE ... PARTITION ... FOR COLUMNS"));
            }
            self.analyze_partitions(&name, &table, spec, a.noscan, original)
                .await?;
            return self.empty_frame().await;
        }
        let case_sensitive = env.analysis.conf.case_sensitive;
        let same = |x: &str, y: &str| {
            if case_sensitive {
                x == y
            } else {
                x.eq_ignore_ascii_case(y)
            }
        };
        // The columns whose statistics are asked for: the listed ones, or all.
        let chosen: Vec<&crate::store::StoreColumn> = match &a.columns {
            AnalyzeColumns::All => Vec::new(),
            AnalyzeColumns::ForAll => table.columns.iter().collect(),
            AnalyzeColumns::ForColumns(columns) => {
                let mut found = Vec::with_capacity(columns.len());
                for c in columns {
                    let Some(column) = table.columns.iter().find(|t| same(&t.name, &c.value))
                    else {
                        return Err(krishiv_common::error::ColumnNotFound {
                            col_name: format!("`{}`", c.value.replace('`', "``")),
                            case_sensitive_config: "\"spark.sql.caseSensitive\"".to_owned(),
                        }
                        .into());
                    };
                    found.push(column);
                }
                found
            }
            _ => return Err(unsupported("this ANALYZE TABLE form")),
        };
        // Spark gathers statistics of numbers, booleans, dates and times, strings and binary only.
        for column in chosen {
            let supported = matches!(
                column.data_type,
                DataType::Boolean
                    | DataType::Byte
                    | DataType::Short
                    | DataType::Integer
                    | DataType::Long
                    | DataType::Float
                    | DataType::Double
                    | DataType::Decimal(_)
                    | DataType::String
                    | DataType::Binary
                    | DataType::Date
                    | DataType::Timestamp
                    | DataType::TimestampNtz
                    | DataType::Time { .. }
            );
            if !supported {
                return Err(
                    krishiv_common::error::UnsupportedFeatureAnalyzeUnsupportedColumnType {
                        column_type: format!("\"{}\"", column.data_type.sql()),
                        column_name: format!("`{}`", column.name.replace('`', "``")),
                        table_name: format!("`{}`.`{}`", table.namespace, table.name),
                    }
                    .into(),
                );
            }
        }
        self.empty_frame().await
    }

    /// `DELETE FROM`, `UPDATE` and `MERGE INTO` a file-source table: Spark's
    /// v1 tables support none of them, and says so once the table resolves.
    pub(crate) async fn exec_row_level_change(
        &self,
        table: &DdlName,
        alias: Option<&krishiv_sql::ast::Ident>,
        operation: &str,
        changes: &RowChanges<'_>,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(table, &env)?;
        self.table_for_write(&name, operation, &env)?;
        // Spark analyzes the condition and the new values against the table before it finds that the
        // table cannot change: an unknown column is the error then.
        // Each assigned column is analyzed as a reference, so an unknown one is the analyzer's own
        // error (with its suggestions), not a message of ours.
        let plain = self.with_locals(Vec::new());
        for target in changes.targets {
            let written = target
                .parts
                .iter()
                .map(|p| format!("`{}`", p.value.replace('`', "``")))
                .collect::<Vec<_>>()
                .join(".");
            let sql = format!("SELECT {written} FROM {}", name.quoted());
            plain.analyze_variable_query(&sql).await?;
        }
        let slice = |e: &krishiv_sql::ast::Expr| {
            original
                .get(e.span.start..e.span.end)
                .unwrap_or_default()
                .to_owned()
        };
        // `SET c = DEFAULT` takes the column's default: nothing to resolve.
        let mut select: Vec<String> = changes
            .values
            .iter()
            .filter(|e| !crate::command::is_bare_default(e))
            .map(|e| slice(e))
            .collect();
        if select.is_empty() {
            select.push("1".to_owned());
        }
        let mut sql = format!("SELECT {} FROM {}", select.join(", "), name.quoted());
        if let Some(alias) = alias {
            sql.push_str(&format!(" AS `{}`", alias.value.replace('`', "``")));
        }
        if let Some(condition) = changes.condition {
            sql.push_str(&format!(" WHERE {}", slice(condition)));
        }
        // Spark 4.2 does not resolve a running script's local variables in these statements (the
        // record shows `UNRESOLVED_COLUMN` for one), so the analysis does not see them either.
        self.with_locals(Vec::new())
            .analyze_variable_query(&sql)
            .await?;
        Err(UnsupportedFeatureTableOperation {
            table_name: name.quoted(),
            operation: operation.to_owned(),
        }
        .into())
    }

    /// Drops the cached relations that read the permanent table `name`: its data or existence changed.
    fn forget_cached(&self, name: &TableName) {
        self.state
            .cache
            .invalidate_table(&format!("{}.{}", name.namespace, name.name));
    }

    /// The permanent table `name` for a command that writes it.
    fn table_for_write(
        &self,
        name: &TableName,
        operation: &str,
        env: &QueryEnv,
    ) -> Result<Arc<CatalogTable>, KrishivError> {
        if let Some(table) = self.state.catalog.store.table(&name.namespace, &name.name)
            && !self.temp_entry_of(name, env)
        {
            if table.view.is_none() {
                return Ok(table);
            }
            return Err(krishiv_common::error::ExpectTableNotViewNoAlternative {
                operation: format!("`{operation}`"),
                view_name: name.quoted(),
            }
            .into());
        }
        let global = name
            .namespace
            .eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
            && self.state.catalog.exists(
                &crate::catalog::global_key(&name.name),
                env.analysis.conf.case_sensitive,
            );
        if self.temp_entry_of(name, env) || global {
            return Err(krishiv_common::error::ExpectTableNotViewNoAlternative {
                operation: format!("`{operation}`"),
                view_name: name.quoted(),
            }
            .into());
        }
        Err(name.not_found())
    }

    /// `INSERT INTO|OVERWRITE [TABLE] name [(columns)] [BY NAME] query`, or the multi-table
    /// `FROM source INSERT ... SELECT ... [INSERT ... SELECT ...]`.
    ///
    /// A multi-table insert writes each clause's rows of the shared source (the clause reads as
    /// `FROM source SELECT ...`). Every clause is resolved and checked before the first write, so
    /// a missing table, an unknown column or a column count mismatch in any clause writes nothing,
    /// as in Spark 4.2.0; the clauses are then written in order. Spark plans them as one statement,
    /// so a failure while one runs also leaves the others unwritten; here the clauses before it
    /// stay written (`docs/known-differences.md`).
    pub(crate) async fn exec_insert(
        &self,
        i: &Insert,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let clauses: Vec<InsertClause> = match &i.source {
            None => i.clauses.clone(),
            Some(source) => i.clauses.iter().map(|c| from_source(source, c)).collect(),
        };
        if clauses.len() > 1 {
            // Spark resolves every target before any clause's columns: a missing table is the
            // error even when an earlier clause names an unknown column.
            let env = self.query_env();
            for clause in &clauses {
                let name = self.table_name(&clause.name, &env)?;
                self.table_for_write(&name, "INSERT", &env)?;
            }
            for clause in &clauses {
                self.insert_clause(clause, original, true).await?;
            }
        }
        for clause in &clauses {
            self.insert_clause(clause, original, false).await?;
        }
        self.empty_frame().await
    }

    /// `c` checked through to the rows it would write (`dry`), or written.
    async fn insert_clause(
        &self,
        c: &InsertClause,
        original: &str,
        dry: bool,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        if c.replace_where.is_some() || c.replace_on.is_some() || c.replace_using.is_some() {
            return Err(unsupported("INSERT ... REPLACE"));
        }
        let name = self.table_name(&c.name, &env)?;
        let table = self.table_for_write(&name, "INSERT", &env)?;
        let statics = match &c.partition {
            None => Vec::new(),
            Some(spec) => {
                if c.overwrite || c.if_not_exists {
                    return Err(unsupported("INSERT OVERWRITE ... PARTITION"));
                }
                self.insert_partition_values(&table, spec, original).await?
            }
        };
        // Spark resolves a scalar subquery in the rows of `INSERT ... VALUES` (an unknown column is
        // the error then) and refuses it afterwards (a plain `VALUES` table reports the cell it
        // cannot evaluate instead).
        if let krishiv_sql::ast::SetExprKind::Values(values) = &c.query.body.kind {
            let mut subqueries = Vec::new();
            values
                .rows
                .iter()
                .for_each(|r| scalar_subqueries(r, &mut subqueries));
            for query in &subqueries {
                let plan = krishiv_sql::lower_with(
                    query,
                    self.next_source_id(),
                    self.parser_options().case_sensitive,
                )
                .map_err(|e| e.error)?;
                DataFrame::new(self.clone(), Arc::clone(plan.root()))
                    .analyzed_in(self.query_env())
                    .await?;
            }
            if !subqueries.is_empty() {
                return Err(
                    krishiv_common::error::UnsupportedSubqueryExpressionCategoryScalarSubqueryInValues {}
                        .into(),
                );
            }
        }
        let plan = krishiv_sql::lower_with(
            &c.query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
        )
        .map_err(|e| e.error)?;
        let source = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let listed: Vec<String> = c
            .columns
            .iter()
            .map(|n| match n {
                ColumnName::Ident(i) => Ok(i.value.clone()),
                _ => Err(unsupported("IDENTIFIER() column names")),
            })
            .collect::<Result<_, _>>()?;
        self.insert_rows(
            &name,
            &table,
            &source,
            &InsertTarget {
                listed: &listed,
                by_name: c.by_name,
                overwrite: c.overwrite,
                statics: &statics,
                dry,
            },
        )
        .await
    }

    /// Inserts the rows of `source` into `table` (`INSERT INTO|OVERWRITE`,
    /// `writeTo(..).append()`): `listed` names the target columns (empty:
    /// all, by position, or by name with `by_name`).
    async fn insert_rows(
        &self,
        name: &TableName,
        table: &Arc<CatalogTable>,
        source: &DataFrame,
        target: &InsertTarget<'_>,
    ) -> Result<(), KrishivError> {
        let InsertTarget {
            listed,
            by_name,
            overwrite,
            statics,
            dry,
        } = *target;
        let env = self.query_env();
        let analysis = source.analyzed_in(self.query_env()).await?;
        if overwrite
            && crate::catalog::reads_stored_table(
                analysis.analyzed.plan.root(),
                &name.namespace,
                &name.name,
                crate::catalog::NAMESPACE,
            )
        {
            return Err(UnsupportedOverwriteTable {
                table: name.quoted(),
            }
            .into());
        }
        let output: Vec<TableColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| TableColumn {
                name: c.name.clone(),
                data_type: krishiv_plan::without_generated_marker(&c.data_type),
                nullable: c.nullable,
            })
            .collect();
        // The query provides every column but the static partition columns.
        let assigned: CatalogTable = if statics.is_empty() {
            (**table).clone()
        } else {
            let mut t = (**table).clone();
            t.columns
                .retain(|c| !statics.iter().any(|(k, _)| *k == c.name));
            t
        };
        let assignment = assign(
            &assigned,
            listed,
            by_name,
            &output,
            env.analysis.conf.case_sensitive,
        )?;
        let renamed: Vec<String> = (0..output.len())
            .map(|i| format!("__krishiv_c{i}"))
            .collect();
        let base = source.to_df(renamed.iter());
        let mut select = Vec::with_capacity(table.columns.len());
        let mut assigned_columns = assigned.columns.iter().zip(&assignment);
        for column in &table.columns {
            if let Some((_, value)) = statics.iter().find(|(k, _)| *k == column.name) {
                let value = if value == krishiv_io::partition::DEFAULT_PARTITION {
                    lit_null().cast(column.data_type.clone())
                } else {
                    crate::functions::lit(value.as_str()).cast_ansi(column.data_type.clone())
                };
                select.push(value.alias(&column.name));
                continue;
            }
            let Some((column, from)) = assigned_columns.next() else {
                return Err(internal("an INSERT lost a column".to_owned()));
            };
            let value = match from {
                Source::Position(i) => {
                    let value = col(&renamed[*i]);
                    if output[*i].data_type == column.data_type {
                        value
                    } else {
                        // Store assignment is ANSI whatever `spark.sql.ansi.enabled` says.
                        value.cast_ansi(column.data_type.clone())
                    }
                }
                Source::Default => match &column.default {
                    Some(text) => expr(text)?.cast_ansi(column.data_type.clone()),
                    None => lit_null().cast(column.data_type.clone()),
                },
            };
            select.push(value.alias(&column.name));
        }
        let df = base.select(select);
        if dry {
            // Every check of the write, without writing.
            return self.check_table_write(table, &df, overwrite).await;
        }
        let mode = if overwrite { "overwrite" } else { "append" };
        let written = self.write_table_rows(table, &df, mode, &mut None).await;
        // Data may have changed even when the write failed late.
        self.state.catalog.store.bump();
        self.forget_cached(name);
        let dirs =
            written.map_err(|e| overflow_in_table_insert(e, &assigned, &output, &assignment))?;
        // The partitions written (`AlterTableAddPartitionCommand` after an insert).
        self.register_partitions(name, &dirs, false);
        Ok(())
    }

    /// The access check the SQL path makes for the statement a table write
    /// stands for: `CREATE TABLE ... AS SELECT` (`Create` in the namespace) or
    /// `INSERT INTO` (`Modify` on the table). The DataFrame writers reach
    /// the command path without a statement, so the equivalent one is parsed
    /// and asked about through the same `authorize_statement`; the table name
    /// is quoted part by part, so it cannot add to the statement. The reads of
    /// the DataFrame are checked by its own analysis.
    async fn authorize_table_write(
        &self,
        name: &TableName,
        create: bool,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let text = if create {
            "CREATE TABLE {} AS SELECT 1"
        } else {
            "INSERT INTO {} SELECT 1"
        };
        self.authorize_equivalent(name, text, env).await
    }

    /// [`Self::authorize_table_write`] for any statement `text` (with `{}`
    /// standing for the quoted table name) the call stands for.
    async fn authorize_equivalent(
        &self,
        name: &TableName,
        text: &str,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let quoted = format!(
            "`{}`.`{}`",
            name.namespace.replace('`', "``"),
            name.name.replace('`', "``")
        );
        let text = text.replace("{}", &quoted);
        let statement = krishiv_sql::parse_statement(&text, &env.parser).map_err(|e| e.error)?;
        let class = krishiv_sql::classify(&statement);
        self.authorize_statement(&statement, class, env).await
    }

    /// What a `create` of a new table checks before it changes anything,
    /// so an overwrite can check it before it drops the old one.
    fn validate_new_table(&self, name: &TableName, spec: &V2WriteSpec) -> Result<(), KrishivError> {
        if !spec.partitioned_by.is_empty() || !spec.cluster_by.is_empty() {
            return Err(unsupported("partitioned tables"));
        }
        self.warehouse_dir()?;
        self.provider_of(spec.provider.clone())?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(name)?;
        if spec.options.values().any(|v| v.as_plain().is_none()) {
            return Err(unsupported("secret table options"));
        }
        Ok(())
    }

    /// The table a `DataFrameWriter` call names.
    fn writer_table_name(&self, table: &str, env: &QueryEnv) -> Result<TableName, KrishivError> {
        let parsed = krishiv_sql::parse_multipart_identifier(table, &self.parser_options())
            .map_err(|e| e.error)?;
        let parts: Vec<String> = parsed.parts.iter().map(|p| p.value.clone()).collect();
        TableName::resolve(&parts, &self.state.catalog.current_namespace(), env)
    }

    /// `DataFrameWriter.saveAsTable`: Spark's V1 save modes over a table. The
    /// table is created from the DataFrame's schema (an existing one in
    /// `errorifexists` mode fails with `TABLE_OR_VIEW_ALREADY_EXISTS`),
    /// `ignore` leaves an existing table alone, `append` adds the rows by
    /// column name (names that do not match the table's, a format other than
    /// its, or a `partition_by` fail with `KRISHIV_UNSUPPORTED_FEATURE`), and
    /// `overwrite` drops the table and creates it again, after the checks that
    /// can refuse the new table and the `UNSUPPORTED_OVERWRITE.TABLE` check
    /// (a query that reads the table it replaces); a temporary view of the
    /// same name makes it `KRISHIV_UNSUPPORTED_FEATURE`.
    pub(crate) async fn exec_save_as_table(
        &self,
        df: &DataFrame,
        spec: V2WriteSpec,
        mode: krishiv_io::SaveMode,
    ) -> Result<(), KrishivError> {
        use krishiv_io::SaveMode;
        let env = self.query_env();
        let name = self.writer_table_name(&spec.table, &env)?;
        let exists = self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .is_some();
        match (mode, exists) {
            (SaveMode::Ignore, true) => {
                // A no-op create is still authorized, as the SQL path does, so a denial
                // cannot tell an existing table from a missing one.
                self.authorize_table_write(&name, true, &env).await
            }
            (SaveMode::Append, true) => {
                // Asked first, as `insertInto` does: what is reported about a table the caller
                // may not write must not depend on its metadata.
                self.authorize_table_write(&name, false, &env).await?;
                let table = self.table_for_write(&name, "INSERT", &env)?;
                // Spark refuses an append whose format or partitioning differs
                // from the table's.
                if !spec.partitioned_by.is_empty() {
                    return Err(unsupported("partitioned tables"));
                }
                if spec
                    .provider
                    .as_deref()
                    .is_some_and(|p| !p.eq_ignore_ascii_case(&table.provider))
                {
                    return Err(unsupported(
                        "saveAsTable append with a format other than the table's",
                    ));
                }
                let schema = df.schema().await?;
                let case_sensitive = env.analysis.conf.case_sensitive;
                let same = |a: &str, b: &str| {
                    if case_sensitive {
                        a == b
                    } else {
                        a.eq_ignore_ascii_case(b)
                    }
                };
                let mut order = Vec::with_capacity(table.columns.len());
                for c in &table.columns {
                    let mut hits = schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .filter(|h| same(h, &c.name));
                    match (hits.next(), hits.next()) {
                        (Some(h), None) => order.push(h.to_owned()),
                        _ => break,
                    }
                }
                if order.len() != table.columns.len() || order.len() != schema.fields().len() {
                    return Err(unsupported(
                        "saveAsTable append with column names that differ from the table's",
                    ));
                }
                // Built directly: a name with dots or a star is one plain column.
                let by_name = df.select(order.iter().map(|h| {
                    crate::column::Column::new(krishiv_plan::Expr::new(
                        krishiv_plan::ExprNode::Column(krishiv_plan::ColumnRef::new(vec![
                            h.clone(),
                        ])),
                    ))
                }));
                self.exec_write_v2(&by_name, spec, V2Mode::Append, &mut None)
                    .await
            }
            (SaveMode::Overwrite, true) => {
                // Spark drops the table and creates it again. Everything that can
                // refuse the new table is checked before the old one goes.
                if self.temp_entry_of(&name, &env) {
                    return Err(unsupported(
                        "saveAsTable overwrite of a name a temporary view shadows",
                    ));
                }
                self.authorize_equivalent(&name, "DROP TABLE {}", &env)
                    .await?;
                self.authorize_table_write(&name, true, &env).await?;
                // `UNSUPPORTED_OVERWRITE.TABLE`: the query reads the table it replaces.
                let analysis = df.analysis().await?;
                if crate::catalog::reads_stored_table(
                    analysis.analyzed.plan.root(),
                    &name.namespace,
                    &name.name,
                    &self.state.catalog.current_namespace(),
                ) {
                    return Err(krishiv_common::error::UnsupportedOverwriteTable {
                        table: name.quoted(),
                    }
                    .into());
                }
                self.validate_new_table(&name, &spec)?;
                // What the write itself would refuse (a format the engine does not
                // write, duplicate output columns, options the format rejects) must
                // not come after the drop.
                let provider = self.provider_of(spec.provider.clone())?;
                let schema = df.schema().await?;
                self.check_write_ready(&provider, &spec.options, &schema, &env)?;
                // The write asks the policy about the table's directory, in the form
                // `write_files` asks it (`locate_write_path`), and the create refuses
                // a directory that is already there other than the old table's own;
                // both before the drop.
                let location =
                    managed_location(&self.warehouse_dir()?, &name.namespace, &name.name);
                let at =
                    self.locate_write_path(&location.display().to_string(), &spec.options, &env)?;
                self.authorize_write_path(&at, &spec.options).await?;
                // A managed table's own directory goes with it; any other one in the way stays.
                let own = self
                    .state
                    .catalog
                    .store
                    .table(&name.namespace, &name.name)
                    .is_some_and(|t| t.managed && t.location == location);
                if location.exists() && !own {
                    return Err(LocationAlreadyExists {
                        identifier: name.quoted(),
                        location: location.display().to_string(),
                    }
                    .into());
                }
                // What is left after this point is the write itself (query errors, I/O): as in
                // Spark's V1 overwrite, a failure there leaves no table.
                // Admitted before the drop: a controller that refuses the write, or a caller that
                // gives up while it queues, must not have lost the old table. The write takes
                // this admission instead of asking again.
                let mut pre = Some(self.admit_query(&analysis.env, false).await?);
                let done = async {
                    self.drop_named_table(&name, &env, true).await?;
                    self.exec_write_v2(df, spec, V2Mode::Create, &mut pre).await
                }
                .await;
                // Not taken by a write (the drop or the create failed first): settle its row. A
                // statement dropped meanwhile drops `pre`, which closes its row `cancelled`.
                if let Some(unrun) = pre.take() {
                    match &done {
                        Ok(()) => unrun.finish_ok(),
                        Err(error) => unrun.fail(error),
                    }
                }
                done
            }
            // `ErrorIfExists` on an existing table fails in the create.
            _ => {
                self.exec_write_v2(df, spec, V2Mode::Create, &mut None)
                    .await
            }
        }
    }

    /// `DataFrameWriter.insertInto`: `errorifexists`, `ignore` and `append`
    /// are all `INSERT INTO` by position (Spark's `insertInto` ignores the
    /// first two); `overwrite` is `INSERT OVERWRITE` by position, as the SQL
    /// statement runs it.
    pub(crate) async fn exec_insert_into(
        &self,
        df: &DataFrame,
        table: &str,
        mode: krishiv_io::SaveMode,
    ) -> Result<(), KrishivError> {
        let overwrite = mode == krishiv_io::SaveMode::Overwrite;
        let env = self.query_env();
        let name = self.writer_table_name(table, &env)?;
        self.authorize_equivalent(
            &name,
            if overwrite {
                "INSERT OVERWRITE TABLE {} SELECT 1"
            } else {
                "INSERT INTO {} SELECT 1"
            },
            &env,
        )
        .await?;
        let found = self.table_for_write(
            &name,
            if overwrite {
                "INSERT OVERWRITE"
            } else {
                "INSERT"
            },
            &env,
        )?;
        let target = InsertTarget {
            listed: &[],
            by_name: false,
            overwrite,
            statics: &[],
            dry: false,
        };
        self.insert_rows(&name, &found, df, &target).await
    }

    /// Runs a [`DataFrameWriterV2`](crate::DataFrameWriterV2) request: `Create`
    /// is `CREATE TABLE name AS SELECT` over `df` (one code path, a failed
    /// write leaves no table), `Append` is `INSERT INTO name` by position.
    /// `Replace` and `CreateOrReplace` fail as `REPLACE TABLE AS SELECT` does
    /// on file-source tables; overwrites and partitioned or clustered tables
    /// are unsupported, as in `INSERT` and `CREATE TABLE`.
    pub(crate) async fn exec_write_v2(
        &self,
        df: &DataFrame,
        spec: V2WriteSpec,
        mode: V2Mode,
        pre: &mut Option<Admitted>,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        let parsed = krishiv_sql::parse_multipart_identifier(&spec.table, &self.parser_options())
            .map_err(|e| e.error)?;
        let parts: Vec<String> = parsed.parts.iter().map(|p| p.value.clone()).collect();
        let name = TableName::resolve(&parts, &self.state.catalog.current_namespace(), &env)?;
        self.authorize_table_write(&name, mode != V2Mode::Append, &env)
            .await?;
        match mode {
            // An append ignores the layout the builder was given: the
            // existing table's layout rules.
            V2Mode::Append => {
                let table = self.table_for_write(&name, "INSERT", &env)?;
                let target = InsertTarget {
                    listed: &[],
                    by_name: false,
                    overwrite: false,
                    statics: &[],
                    dry: false,
                };
                return self.insert_rows(&name, &table, df, &target).await;
            }
            V2Mode::Replace | V2Mode::CreateOrReplace => {
                return Err(UnsupportedFeatureTableOperation {
                    table_name: name.quoted(),
                    operation: "REPLACE TABLE AS SELECT".to_owned(),
                }
                .into());
            }
            V2Mode::Create => {}
        }
        if !spec.partitioned_by.is_empty() || !spec.cluster_by.is_empty() {
            return Err(unsupported("partitioned tables"));
        }
        let warehouse = self.warehouse_dir()?;
        let provider_written = self.written_provider(spec.provider.clone());
        let provider = self.provider_of(spec.provider)?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(&name)?;
        if self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .is_some()
        {
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: name.quoted(),
            }
            .into());
        }
        let mut options = BTreeMap::new();
        for (k, v) in &spec.options {
            // Table metadata is stored as plain text: a secret never is.
            let Some(text) = v.as_plain() else {
                return Err(unsupported("secret table options"));
            };
            options.insert(k.clone(), text.to_owned());
        }
        let table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: true,
            provider,
            provider_written,
            location: managed_location(&warehouse, &name.namespace, &name.name),
            columns: Vec::new(),
            options,
            properties: spec.properties,
            comment: None,
            created_ms: now_ms(),
            root: warehouse,
            view: None,
            layout: TableLayout::default(),
        };
        let layout = LayoutRequest {
            partition_by: &[],
            bucket: None,
        };
        self.create_table_from(table, df, CreateMode::ErrorIfExists, &layout, pre)
            .await
    }

    /// Writes `df` to the files of `table` with the writer's save `mode`, in partition directories when the
    /// table is partitioned, and returns the partition directories written. Bucketed files are not written:
    /// a bucketed table refuses the write.
    async fn write_table_rows(
        &self,
        table: &CatalogTable,
        df: &DataFrame,
        mode: &str,
        pre: &mut Option<Admitted>,
    ) -> Result<Vec<String>, KrishivError> {
        table_writer(table, df, mode)?
            .save_reporting(&table.location.display().to_string(), pre.take())
            .await
    }

    /// The checks [`Self::write_table_rows`] makes before it writes, without writing.
    async fn check_table_write(
        &self,
        table: &CatalogTable,
        df: &DataFrame,
        overwrite: bool,
    ) -> Result<(), KrishivError> {
        let mode = if overwrite { "overwrite" } else { "append" };
        table_writer(table, df, mode)?
            .check_save(&table.location.display().to_string())
            .await
    }
}

/// What an `ALTER TABLE ... CHANGE | ALTER COLUMN` asks for.
enum ColumnChange {
    Type(DataType),
    Comment(String),
    Position,
    Nullability,
    DropDefault,
}

/// The writer of `df` into `table` with `mode` (bucketed tables refused).
fn table_writer(
    table: &CatalogTable,
    df: &DataFrame,
    mode: &str,
) -> Result<crate::dataframe::DataFrameWriter, KrishivError> {
    if table.layout.bucket.is_some() {
        return Err(unsupported("writes to bucketed tables"));
    }
    let mut writer = df.write().format(&table.provider).mode(mode);
    if table.is_partitioned() {
        writer = writer.partition_by(&table.layout.partition_columns);
    }
    for (k, v) in &table.options {
        writer = writer.option(k, v);
    }
    Ok(writer)
}

/// `UNSUPPORTED_FEATURE.TABLE_OPERATION` for `operation` on the table `name`.
fn table_operation(name: &TableName, operation: &str) -> KrishivError {
    UnsupportedFeatureTableOperation {
        table_name: name.quoted(),
        operation: operation.to_owned(),
    }
    .into()
}

/// The index of the column `column` names in `table` (Spark's `ResolveAlterTableCommands`): an unknown one is
/// `UNRESOLVED_COLUMN.WITH_SUGGESTION` listing the table's columns.
fn resolve_column(
    table: &CatalogTable,
    column: &ColumnName,
    env: &QueryEnv,
) -> Result<usize, KrishivError> {
    let ColumnName::Ident(ident) = column else {
        return Err(unsupported("IDENTIFIER() column names"));
    };
    let case_sensitive = env.analysis.conf.case_sensitive;
    table
        .columns
        .iter()
        .position(|c| {
            if case_sensitive {
                c.name == ident.value
            } else {
                c.name.eq_ignore_ascii_case(&ident.value)
            }
        })
        .ok_or_else(|| {
            UnresolvedColumnWithSuggestion {
                object_name: format!("`{}`", ident.value.replace('`', "``")),
                proposal: table
                    .columns
                    .iter()
                    .map(|c| format!("`{}`", c.name.replace('`', "``")))
                    .collect::<Vec<_>>()
                    .join(", "),
            }
            .into()
        })
}

/// Where an insert's rows go: the listed columns (empty: all, by position, or by name with `by_name`), and
/// the static partition values of `PARTITION (...)`.
#[derive(Clone, Copy)]
struct InsertTarget<'a> {
    listed: &'a [String],
    by_name: bool,
    overwrite: bool,
    statics: &'a [(String, String)],
    /// Checked through to the rows it would write, not written.
    dry: bool,
}

/// A clause of a multi-table insert as a single-table one: its query reads `FROM source`, as
/// Spark's `visitMultiInsertQuery` gives each body the shared relation.
fn from_source(source: &krishiv_sql::ast::FromClause, c: &InsertClause) -> InsertClause {
    use krishiv_sql::ast::{FromStatement, Query, SetExpr, SetExprKind};
    let mut body = (*c.query).clone();
    let with = body.with.take();
    let span = c.query.span;
    let query = Query {
        with,
        body: SetExpr {
            kind: SetExprKind::FromStatement(Box::new(FromStatement {
                from: source.clone(),
                bodies: vec![body],
                span,
            })),
            span,
        },
        order_by: Vec::new(),
        order_span: None,
        cluster_by: Vec::new(),
        distribute_by: Vec::new(),
        sort_by: Vec::new(),
        windows: Vec::new(),
        limit: None,
        offset: None,
        span,
    };
    InsertClause {
        query: Box::new(query),
        ..c.clone()
    }
}

/// The `PARTITIONED BY` column names and `CLUSTERED BY` of a new table.
struct LayoutRequest<'a> {
    partition_by: &'a [String],
    bucket: Option<&'a krishiv_sql::ast::BucketSpec>,
}

/// The `PARTITIONED BY` elements as column names and typed columns (Spark refuses a mix of both).
fn partition_fields(
    fields: &[krishiv_sql::ast::PartitionField],
) -> Result<(Vec<String>, Vec<&krishiv_sql::ast::PartitionColumn>), KrishivError> {
    use krishiv_sql::ast::PartitionFieldKind;
    let mut names = Vec::new();
    let mut typed = Vec::new();
    let mut references = Vec::new();
    for f in fields {
        match &f.kind {
            PartitionFieldKind::Column(c) => {
                let ColumnName::Ident(ident) = &c.name else {
                    return Err(unsupported("IDENTIFIER() column names"));
                };
                names.push(ident.value.clone());
                typed.push(c);
            }
            PartitionFieldKind::Transform(t) => match (&t.name, t.args.is_empty()) {
                (DdlName::Parts(n), true) if n.parts.len() == 1 => {
                    names.push(n.parts[0].value.clone());
                    references.push(n.parts[0].value.clone());
                }
                _ => return Err(unsupported("partition transforms")),
            },
            _ => return Err(unsupported("this PARTITIONED BY element")),
        }
    }
    if !typed.is_empty() && !references.is_empty() {
        // `AstBuilder.partitionExpressions` (`operationNotAllowed`).
        let columns = typed
            .iter()
            .map(|c| {
                let name = match &c.name {
                    ColumnName::Ident(i) => i.value.clone(),
                    _ => String::new(),
                };
                let t = c
                    .data_type
                    .to_data_type()
                    .map(|t| t.catalog_string())
                    .unwrap_or_default();
                format!("{name} {t}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(krishiv_common::error::LegacyErrorTemp0035 {
            message: format!(
                "PARTITION BY: Cannot mix partition expressions and partition columns:\nExpressions: {}\nColumns: {columns}",
                references.join(", ")
            ),
        }
        .into());
    }
    Ok((names, typed))
}

/// What [`Session::exec_write_v2`] writes: the table name as written and
/// the builder state of [`DataFrameWriterV2`](crate::DataFrameWriterV2).
/// Tables are managed (the writer has no `LOCATION`).
pub(crate) struct V2WriteSpec {
    /// The table name as written (`t`, `db.t` or `spark_catalog.db.t`).
    pub table: String,
    /// `using`: `None` for `spark.sql.sources.default`.
    pub provider: Option<String>,
    /// `option`/`options`: the table's options and its read options.
    pub options: BTreeMap<String, OptionValue>,
    /// `tableProperty`: the `TBLPROPERTIES`.
    pub properties: BTreeMap<String, String>,
    /// `partitionedBy` columns.
    pub partitioned_by: Vec<String>,
    /// `clusterBy` columns.
    pub cluster_by: Vec<String>,
}

/// What [`Session::exec_write_v2`] does with the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum V2Mode {
    /// Create it; an existing table fails with `TABLE_OR_VIEW_ALREADY_EXISTS`.
    Create,
    /// Replace it (unsupported for file-source tables).
    Replace,
    /// Create or replace it (unsupported for file-source tables).
    CreateOrReplace,
    /// Append the rows to the existing table.
    Append,
}

/// Where a table column's value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// The query's column at this position.
    Position(usize),
    /// The column's `DEFAULT`, else NULL.
    Default,
}

/// Matches the query's `output` to the columns of `table`: by position, by
/// the column list, or by name. Spark's arity and store-assignment errors.
fn assign(
    table: &CatalogTable,
    listed: &[String],
    by_name: bool,
    output: &[TableColumn],
    case_sensitive: bool,
) -> Result<Vec<Source>, KrishivError> {
    let eq = |a: &str, b: &str| {
        if case_sensitive {
            a == b
        } else {
            a.eq_ignore_ascii_case(b)
        }
    };
    let quoted_names = |names: Vec<&str>| {
        names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let table_name = quoted(&table.namespace, &table.name);
    let table_cols = || quoted_names(table.columns.iter().map(|c| c.name.as_str()).collect());
    let data_cols = || quoted_names(output.iter().map(|c| c.name.as_str()).collect());
    let mut out = vec![Source::Default; table.columns.len()];
    if by_name {
        for (i, o) in output.iter().enumerate() {
            let Some(at) = table.columns.iter().position(|c| eq(&c.name, &o.name)) else {
                // A data column the table does not have: Spark counts the table's columns against the data's.
                return Err(InsertColumnArityMismatchTooManyDataColumns {
                    table_name,
                    table_columns: table_cols(),
                    data_columns: data_cols(),
                }
                .into());
            };
            out[at] = Source::Position(i);
        }
    } else if !listed.is_empty() {
        if listed.len() != output.len() {
            let (not_enough, table_cols_text) = (
                listed.len() > output.len(),
                quoted_names(listed.iter().map(String::as_str).collect()),
            );
            return Err(if not_enough {
                InsertColumnArityMismatchNotEnoughDataColumns {
                    table_name,
                    table_columns: table_cols_text,
                    data_columns: data_cols(),
                }
                .into()
            } else {
                InsertColumnArityMismatchTooManyDataColumns {
                    table_name,
                    table_columns: table_cols_text,
                    data_columns: data_cols(),
                }
                .into()
            });
        }
        for (i, name) in listed.iter().enumerate() {
            let Some(at) = table.columns.iter().position(|c| eq(&c.name, name)) else {
                return Err(UnresolvedColumnWithSuggestion {
                    object_name: format!("`{name}`"),
                    proposal: table_cols(),
                }
                .into());
            };
            out[at] = Source::Position(i);
        }
    } else {
        match output.len().cmp(&table.columns.len()) {
            std::cmp::Ordering::Less => {
                return Err(InsertColumnArityMismatchNotEnoughDataColumns {
                    table_name,
                    table_columns: table_cols(),
                    data_columns: data_cols(),
                }
                .into());
            }
            std::cmp::Ordering::Greater => {
                return Err(InsertColumnArityMismatchTooManyDataColumns {
                    table_name,
                    table_columns: table_cols(),
                    data_columns: data_cols(),
                }
                .into());
            }
            std::cmp::Ordering::Equal => {
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = Source::Position(i);
                }
            }
        }
    }
    for (column, from) in table.columns.iter().zip(&out) {
        if let Source::Position(i) = from
            && !can_store(&output[*i].data_type, &column.data_type)
        {
            return Err(IncompatibleDataForTableCannotSafelyCast {
                table_name,
                col_name: format!("`{}`", column.name),
                src_type: format!(
                    "\"{}\"",
                    output[*i].data_type.catalog_string().to_uppercase()
                ),
                target_type: format!("\"{}\"", column.data_type.catalog_string().to_uppercase()),
            }
            .into());
        }
    }
    Ok(out)
}

/// Spark's ANSI store assignment (`Cast.canANSIStoreAssign`): whether a
/// value of type `from` may be stored into a column of type `to`.
fn can_store(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    if from == to {
        return true;
    }
    let numeric = |t: &T| {
        matches!(
            t,
            T::Byte | T::Short | T::Integer | T::Long | T::Float | T::Double | T::Decimal(_)
        )
    };
    // Anything atomic may be stored as a string.
    let to_string = matches!(to, T::String | T::CollatedString(_))
        && (numeric(from)
            || matches!(
                from,
                T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary
            ));
    match (from, to) {
        (T::Null, _) | (T::String | T::CollatedString(_), T::String | T::CollatedString(_)) => true,
        (a, b) if (numeric(a) && numeric(b)) || to_string => true,
        // Dates and timestamps among themselves.
        (T::Date, T::Timestamp | T::TimestampNtz)
        | (T::Timestamp | T::TimestampNtz, T::Date | T::Timestamp | T::TimestampNtz) => true,
        (T::Array(a), T::Array(b)) => {
            can_store(a.element_type(), b.element_type())
                && (b.contains_null() || !a.contains_null())
        }
        (T::Map(a), T::Map(b)) => {
            can_store(a.key_type(), b.key_type())
                && can_store(a.value_type(), b.value_type())
                && (b.value_contains_null() || !a.value_contains_null())
        }
        (T::Struct(a), T::Struct(b)) => {
            a.fields().len() == b.fields().len()
                && a.fields().iter().zip(b.fields()).all(|(x, y)| {
                    can_store(x.data_type(), y.data_type()) && (y.nullable() || !x.nullable())
                })
        }
        _ => false,
    }
}

/// A `CAST_OVERFLOW` raised while a column is written is `CAST_OVERFLOW_IN_TABLE_INSERT`, naming the column,
/// whatever `spark.sql.ansi.enabled` is; any other error is returned as it came.
fn overflow_in_table_insert(
    e: KrishivError,
    table: &CatalogTable,
    output: &[TableColumn],
    assignment: &[Source],
) -> KrishivError {
    if e.error_class() != ErrorClass::CastOverflow {
        return e;
    }
    let params = e.message_parameters();
    let param = |key: &str| {
        params
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, v)| v.clone())
    };
    let (Some(source_type), Some(target_type)) = (param("sourceType"), param("targetType")) else {
        return e;
    };
    let sql = |t: &DataType| format!("\"{}\"", t.catalog_string().to_uppercase());
    let column = table
        .columns
        .iter()
        .zip(assignment)
        .find(|(c, from)| {
            matches!(from, Source::Position(i)
                if sql(&c.data_type) == target_type && sql(&output[*i].data_type) == source_type)
        })
        .map(|(c, _)| c.name.clone());
    // No column matches: the overflow is the query's own, and Spark reports it as it is.
    let Some(column) = column else {
        return e;
    };
    CastOverflowInTableInsert {
        source_type,
        target_type,
        column_name: format!("`{column}`"),
    }
    .into()
}

/// `OPTIONS (...)` / `TBLPROPERTIES (...)` as text pairs, keys lower-cased
/// for options as Spark does not (kept as written).
fn options_of(options: &[TableOption], original: &str) -> BTreeMap<String, String> {
    options
        .iter()
        .map(|o| {
            let key = o
                .key
                .parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join(".");
            let value = o
                .value
                .as_ref()
                .map(|v| literal_text(original.get(v.span.start..v.span.end).unwrap_or_default()))
                .unwrap_or_default();
            (key, value)
        })
        .collect()
}

/// A literal's text without its quotes.
fn literal_text(fragment: &str) -> String {
    let f = fragment.trim();
    for q in ['\'', '"'] {
        if f.len() >= 2 && f.starts_with(q) && f.ends_with(q) {
            return f[1..f.len() - 1].to_owned();
        }
    }
    f.to_owned()
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// Deletes a managed table's directory; a directory that is already gone
/// is fine.
fn remove_dir(dir: &Path) -> Result<(), KrishivError> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(internal(format!("cannot delete {}: {e}", dir.display()))),
    }
}

/// Removes everything in `dir` and keeps `dir`.
fn clear_dir(dir: &Path) -> Result<(), KrishivError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(internal(format!("cannot list {}: {e}", dir.display()))),
    };
    for entry in entries {
        let path = entry
            .map_err(|e| internal(format!("cannot list {}: {e}", dir.display())))?
            .path();
        let removed = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        removed.map_err(|e| internal(format!("cannot delete {}: {e}", path.display())))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

/// Why an [`OwnedDirectory`] could not be made.
#[derive(Debug)]
enum OwnedDirectoryError {
    /// Something is already at the path (not ours to touch).
    Exists,
    Io(String),
}

/// A directory this statement created exclusively. Dropped without [`Self::keep`] (the statement
/// failed, or its future was dropped), it removes what is in it, provided the path still names the
/// directory that was created (same device and inode): a path swapped for another directory in the
/// meantime is left alone.
struct OwnedDirectory {
    path: PathBuf,
    identity: Option<(u64, u64)>,
    kept: bool,
}

impl OwnedDirectory {
    fn create(path: PathBuf) -> Result<Self, OwnedDirectoryError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                OwnedDirectoryError::Io(format!("cannot create {}: {e}", parent.display()))
            })?;
        }
        match std::fs::create_dir(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(OwnedDirectoryError::Exists);
            }
            Err(e) => {
                return Err(OwnedDirectoryError::Io(format!(
                    "cannot create {}: {e}",
                    path.display()
                )));
            }
        }
        let identity = Self::identity_of(&path);
        Ok(Self {
            path,
            identity,
            kept: false,
        })
    }

    fn identity_of(path: &Path) -> Option<(u64, u64)> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::symlink_metadata(path).ok()?;
            meta.is_dir().then(|| (meta.dev(), meta.ino()))
        }
        #[cfg(not(unix))]
        {
            let meta = std::fs::symlink_metadata(path).ok()?;
            meta.is_dir().then_some((0, 0))
        }
    }

    /// The statement succeeded: the directory stays.
    fn keep(mut self) {
        self.kept = true;
    }
}

impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        if !self.kept && self.identity.is_some() && Self::identity_of(&self.path) == self.identity {
            drop(std::fs::remove_dir_all(&self.path));
        }
    }
}

#[cfg(test)]
mod owned_directory_tests {
    use super::{OwnedDirectory, OwnedDirectoryError};

    #[test]
    fn a_dropped_guard_removes_only_the_directory_it_made() {
        let tmp = tempfile::tempdir().unwrap();
        // Dropped without `keep` (a failed or cancelled statement): gone, with what was written.
        let a = tmp.path().join("a");
        let guard = OwnedDirectory::create(a.clone()).unwrap();
        std::fs::write(a.join("f"), b"x").unwrap();
        drop(guard);
        assert!(!a.exists());
        // Kept: stays.
        let b = tmp.path().join("b");
        OwnedDirectory::create(b.clone()).unwrap().keep();
        assert!(b.exists());
        // A path that already exists is not ours.
        assert!(matches!(
            OwnedDirectory::create(b.clone()),
            Err(OwnedDirectoryError::Exists)
        ));
        assert!(b.exists());
        // The path now names another directory (swapped meanwhile): left alone.
        let c = tmp.path().join("c");
        let guard = OwnedDirectory::create(c.clone()).unwrap();
        // Renamed aside, not removed: a removed directory's inode number may be reused by the new one.
        std::fs::rename(&c, tmp.path().join("c-moved")).unwrap();
        std::fs::create_dir(&c).unwrap();
        std::fs::write(c.join("theirs"), b"y").unwrap();
        drop(guard);
        assert!(c.join("theirs").exists());
    }
}
