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
use krishiv_plan::analyzer::{Privilege, TableColumn};
use krishiv_sql::StatementClass;
use krishiv_sql::ast::{
    AlterColumnAction, AlterNamespace, AlterNamespaceAction, AlterTable, AlterTableAction,
    AlterView, AlterViewAction, Analyze, AnalyzeColumns, ChangeColumnAction, ColumnName,
    CreateNamespace, CreatePermanentView, CreateTable, CreateTableLike, CreateView, DdlName,
    DropBehavior, DropNamespace, DropPermanentView, DropTable, Ident, Insert, InsertClause, Query,
    TableElement, TableOption, Truncate, Use, ViewColumn,
};
use krishiv_types::DataType;

use crate::access::refuse_secret_options;
use crate::dataframe::DataFrame;
use crate::functions::{col, expr, lit_null};
use crate::query_fold::Admitted;
use crate::session::{QueryEnv, Session};
use crate::store::{
    CatalogStore, CatalogTable, CreateMode, DEFAULT_NAMESPACE, DropNamespaceError, NamespaceMeta,
    StoreColumn, TableLayout, ViewDef, managed_location,
};

/// The file formats a table can have.
pub(crate) const PROVIDERS: &[&str] = &["parquet", "csv", "json", "text", "orc", "avro"];

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
pub(crate) fn check_new_name(name: &str) -> Result<(), KrishivError> {
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
pub(crate) fn concurrent_change(name: &TableName) -> KrishivError {
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
        let parts = crate::access::ddl_parts(name, env)?;
        TableName::resolve(&parts, &self.namespace_for(&parts, env)?, env)
    }

    /// Refuses a table, view or SQL function in a schema with a default collation (`spark.sql.collation.schemaLevel.enabled`):
    /// Krishiv does not apply the default to the new columns yet, and would type them `UTF8_BINARY`.
    pub(crate) fn refuse_collated_namespace(&self, name: &TableName) -> Result<(), KrishivError> {
        let collated = self
            .state
            .catalog
            .store
            .namespaces()
            .into_iter()
            .any(|(n, m)| n.eq_ignore_ascii_case(&name.namespace) && m.collation.is_some());
        if collated {
            return Err(unsupported(
                "a table or view in a schema with a DEFAULT COLLATION",
            ));
        }
        Ok(())
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
            [ns] => self.namespace_arg(&fold(ns)),
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
        let collation = match &n.collation {
            Some(c) => {
                let parts: Vec<&str> = c.parts.iter().map(|p| p.value.as_str()).collect();
                Some(krishiv_types::Collation::parse(&parts.join("."))?.name())
            }
            None => None,
        };
        let meta = NamespaceMeta {
            collation,
            comment: n.comment.clone(),
            location: n
                .location
                .as_deref()
                .map(|l| qualified_namespace_location(l, || self.warehouse_dir()))
                .transpose()?,
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
                let l = qualified_namespace_location(l, || self.warehouse_dir())?;
                Box::new(move |m| m.location = Some(l.clone()))
            }
            AlterNamespaceAction::SetComment(c) => {
                let c = c.clone();
                Box::new(move |m| m.comment.clone_from(&c))
            }
            // Reaches here only with `spark.sql.collation.schemaLevel.enabled` (the parser refuses it
            // otherwise).
            AlterNamespaceAction::DefaultCollation(c) => {
                let parts: Vec<&str> = c.parts.iter().map(|p| p.value.as_str()).collect();
                let c = krishiv_types::Collation::parse(&parts.join("."))?.name();
                Box::new(move |m| m.collation = Some(c.clone()))
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
            Err(DropNamespaceError::Changed) => {
                return Err(internal(format!(
                    "`{name}` was changed by another statement while this one ran; retry"
                )));
            }
        }
        self.empty_frame().await
    }

    /// `USE [NAMESPACE|DATABASE|SCHEMA] name`.
    pub(crate) async fn exec_use(&self, u: &Use) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        // Spark's catalog-first rule: unless the statement says NAMESPACE, a
        // first part that names a catalog is the catalog.
        let parts = crate::access::ddl_parts(&u.name, &env)?;
        let canonical = |p: &str| {
            if crate::system::is_system(p) {
                Some("system")
            } else if p.eq_ignore_ascii_case(crate::catalog::CATALOG) {
                Some(crate::catalog::CATALOG)
            } else {
                None
            }
        };
        let quoted = |ps: &[String]| format!("`{}`", ps.join("`.`"));
        if u.kind.is_none()
            && let Some(catalog) = parts.first().and_then(|p| canonical(p))
        {
            match (catalog, &parts[1..]) {
                (_, []) => {
                    self.state.catalog.use_catalog(catalog);
                    return self.empty_frame().await;
                }
                ("system", rest) => {
                    return Err(SchemaNotFound {
                        schema_name: quoted(&[vec!["system".to_owned()], rest.to_vec()].concat()),
                    }
                    .into());
                }
                (_, [_]) => {}
                (_, rest) => {
                    return Err(SchemaNotFound {
                        schema_name: quoted(
                            &[vec![crate::catalog::CATALOG.to_owned()], rest.to_vec()].concat(),
                        ),
                    }
                    .into());
                }
            }
        } else if self.names_a_system_namespace(&parts) {
            return Err(SchemaNotFound {
                schema_name: quoted(&[vec!["system".to_owned()], parts].concat()),
            }
            .into());
        }
        let name = self.namespace_name(&u.name, &env)?;
        if !self.state.catalog.store.has_namespace(&name) {
            return Err(SchemaNotFound {
                schema_name: format!("`{name}`"),
            }
            .into());
        }
        self.state
            .catalog
            .use_position(crate::catalog::CATALOG, &name);
        self.empty_frame().await
    }

    /// `CREATE [OR REPLACE] VIEW ...` for a permanent view.
    pub(crate) async fn exec_create_permanent_view(
        &self,
        v: &CreatePermanentView,
        original: &str,
        (ids, written): (&[krishiv_sql::ast::Span], &[String]),
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        if v.global && v.using.is_none() {
            return self.create_global_temp_view(v, &env, ids).await;
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
        self.refuse_collated_namespace(&name)?;
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
        let plan = krishiv_sql::lower_identified(
            query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
            ids,
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
        crate::char_write::refuse_indeterminate_schema(&analysis.analyzed)?;
        let mut columns = crate::char_write::output_columns(&analysis.analyzed, None);
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
                // Spark's `aliasPlan`: a listed comment replaces the column's; without one the alias keeps the
                // query column's metadata, its comment included.
                if alias.comment.is_some() {
                    column.comment.clone_from(&alias.comment);
                }
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
                identifier_spans: crate::identifier_clause::spans_within(ids, query.span),
                written: crate::identifier_clause::written_within(
                    original, ids, written, query.span,
                ),
            }),
            layout: TableLayout::default(),
            column_stats: BTreeMap::new(),
            incarnation: 0,
            data_version: 0,
            stamp: 0,
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
        ids: &[krishiv_sql::ast::Span],
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
        self.create_temp_view(&view, true, ids, None).await
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
                Ok(TableColumn::new(
                    ident.value.clone(),
                    spec.to_data_type()?,
                    !c.not_null,
                ))
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        if columns.is_empty() {
            return Err(unsupported(
                "datasource views without declared column types",
            ));
        }
        let mut options = options_of(refuse_secret_options(&v.options, original)?, original);
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
            schema: None,
        };
        self.state.catalog.put(
            name,
            crate::catalog::TempTable::Files {
                source,
                columns,
                known: true,
                permanent: false,
                partitions: Some(Default::default()),
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
        env: &crate::session::QueryEnv,
    ) -> Result<(), KrishivError> {
        let name = TableName::resolve(&[name.to_owned()], &self.stored_namespace_in(env)?, env)?;
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
        (ids, written): (&[krishiv_sql::ast::Span], &[String]),
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
                let plan = krishiv_sql::lower_identified(
                    query,
                    self.next_source_id(),
                    self.parser_options().case_sensitive,
                    ids,
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
                crate::char_write::refuse_indeterminate_schema(&analysis.analyzed)?;
                changed.columns = crate::char_write::output_columns(&analysis.analyzed, None);
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
                    identifier_spans: crate::identifier_clause::spans_within(ids, query.span),
                    written: crate::identifier_clause::written_within(
                        original, ids, written, query.span,
                    ),
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
            AlterViewAction::UnsetTblProperties { keys, .. } => {
                for key in options_of(keys, original).keys() {
                    changed.properties.remove(key);
                }
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
                    // Existing files have no such column: it reads as NULL.
                    let column = StoreColumn::declared(
                        ident.value.clone(),
                        &def.data_type,
                        None,
                        def.comment.clone(),
                    )?;
                    changed.columns.insert(at, column);
                }
            }
            AlterTableAction::ChangeColumn {
                name: column,
                action,
                ..
            } => {
                let change = match action {
                    ChangeColumnAction::Type(t) => {
                        let (ty, declared) = t.to_column_type()?;
                        ColumnChange::Type(ty, declared)
                    }
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
            AlterTableAction::AddPartition { .. }
            | AlterTableAction::SetPartitionLocation { .. } => {
                changed = self.alter_partitions(&table, &a.action, original).await?;
            }
            // It swaps the catalog entry itself, between its directory move and the statement's end.
            AlterTableAction::RenamePartition { from, to, .. } => {
                self.rename_partition(&table, &name, [from, to], original)
                    .await?;
                return self.empty_frame().await;
            }
            // It swaps the catalog entry itself, between its log commit and its deletes.
            AlterTableAction::DropPartition {
                if_exists, specs, ..
            } => {
                self.drop_partitions(&table, &name, specs, *if_exists, original)
                    .await?;
                return self.empty_frame().await;
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
            AlterTableAction::SetSerdeProperties(props) => {
                // The storage properties are the table's options (what the reader and writer get):
                // a server-scoped store key in them is `Manage` on the cluster (P3-T067b).
                let props = options_of(props, original);
                self.authorize_store_options(
                    krishiv_sql::StatementClass::Ddl,
                    &self.state.conf.snapshot(),
                    props.keys().map(String::as_str),
                )
                .await?;
                changed.options.extend(props);
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
            ColumnChange::Type(t, declared)
                if t != current.data_type || declared != current.char_varchar =>
            {
                return Err(current.change_type_error(name.quoted(), &t, declared.as_deref()));
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
}

/// What an `ALTER TABLE ... CHANGE | ALTER COLUMN` asks for.
enum ColumnChange {
    /// The type and the `CHAR(n)` / `VARCHAR(n)` string the new type was written with.
    Type(DataType, Option<String>),
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
    // Stored when an admin (or a `Manage` holder) created or altered the table: not asked again.
    Ok(writer.options_are_the_tables())
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
        organization_span: None,
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
    let e = crate::char_write::map_error(e);
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

mod files;
use files::remove_dir;
mod location;
use location::qualified_namespace_location;

#[cfg(test)]
mod tests;

/// Why an [`OwnedDirectory`] could not be made.
#[derive(Debug)]
pub(crate) enum OwnedDirectoryError {
    /// Something is already at the path (not ours to touch).
    Exists,
    Io(String),
}

/// A directory this statement created exclusively. Dropped without [`Self::keep`] (the statement
/// failed, or its future was dropped), it removes what is in it, provided the path still names the
/// directory that was created (same device and inode): a path swapped for another directory in the
/// meantime is left alone.
pub(crate) struct OwnedDirectory {
    path: PathBuf,
    identity: Option<(u64, u64)>,
    kept: bool,
}

impl OwnedDirectory {
    pub(crate) fn create(path: PathBuf) -> Result<Self, OwnedDirectoryError> {
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
    pub(crate) fn keep(mut self) {
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
mod owned_directory_tests;

mod writes;

mod create;
