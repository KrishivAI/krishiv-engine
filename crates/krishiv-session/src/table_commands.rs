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
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{Privilege, Securable, TableColumn, authorize};
use krishiv_sql::StatementClass;
use krishiv_sql::ast::{
    AlterTable, AlterTableAction, AlterView, AlterViewAction, Analyze, AnalyzeColumns, ColumnName,
    CreateNamespace, CreatePermanentView, CreateTable, CreateTableLike, DdlName, DropBehavior,
    DropNamespace, DropPermanentView, DropTable, Insert, InsertClause, Query, TableElement,
    TableOption, Truncate, Use,
};
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
use crate::functions::{col, expr, lit_null};
use crate::session::{QueryEnv, Session};
use crate::store::{
    CatalogStore, CatalogTable, CreateMode, DEFAULT_NAMESPACE, DropNamespaceError, NamespaceMeta,
    StoreColumn, ViewDef, managed_location,
};

/// The file formats a table can have.
const PROVIDERS: &[&str] = &["parquet", "csv", "json", "text", "orc"];

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
    fn resolve(parts: &[String], current: &str, env: &QueryEnv) -> Result<Self, KrishivError> {
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
            [catalog, _, _] => {
                return Err(unsupported(&format!("tables in the catalog `{catalog}`")));
            }
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
        if v.temporary || v.global {
            return Err(unsupported("global and qualified temporary views"));
        }
        let (None, Some(query)) = (&v.using, &v.query) else {
            return Err(unsupported("datasource views"));
        };
        if v.schema_binding.is_some() {
            return Err(unsupported("WITH SCHEMA binding modes of views"));
        }
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
        let plan = krishiv_sql::lower(query, self.next_source_id()).map_err(|e| e.error)?;
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
                data_type: c.data_type.clone(),
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
        // Spark checks the names after the aliases: a repeated one is `COLUMN_ALREADY_EXISTS`, ignoring case
        // unless `caseSensitive` (then as written, else in lower case).
        let mut seen = std::collections::BTreeSet::new();
        for c in &columns {
            let key = if case_sensitive {
                c.name.clone()
            } else {
                c.name.to_lowercase()
            };
            if !seen.insert(key.clone()) {
                return Err(krishiv_common::error::ColumnAlreadyExists {
                    column_name: format!("`{key}`"),
                }
                .into());
            }
        }
        let view = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: false,
            provider: String::new(),
            location: PathBuf::new(),
            columns,
            options: BTreeMap::new(),
            properties: options_of(&v.tblproperties, original),
            comment: v.comment.clone(),
            created_ms: now_ms(),
            root: PathBuf::new(),
            view: Some(ViewDef {
                sql,
                namespace: self.state.catalog.current_namespace(),
                query_columns,
            }),
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
        let name = self.table_name(&d.name, &env)?;
        self.drop_stored_view(&name, d.if_exists)?;
        self.empty_frame().await
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

    /// `ALTER VIEW name ...`: `AS query` and `SET|UNSET TBLPROPERTIES`; `RENAME TO` and the schema binding forms
    /// are not executed yet.
    pub(crate) async fn exec_alter_view(
        &self,
        a: &AlterView,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&a.name, &env)?;
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
                let plan = krishiv_sql::lower(query, self.next_source_id()).map_err(|e| e.error)?;
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
                        data_type: c.data_type.clone(),
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
                changed.view = Some(ViewDef {
                    sql: original
                        .get(query.span.start..query.span.end)
                        .unwrap_or_default()
                        .to_owned(),
                    namespace: self.state.catalog.current_namespace(),
                    query_columns,
                });
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
            return Err(krishiv_common::error::ExpectTableNotViewUseAlterView {
                operation: "`ALTER TABLE`".to_owned(),
                view_name: name.quoted(),
            }
            .into());
        }
        let table = self.table_for_write(&name, "ALTER TABLE", &env)?;
        let mut changed = (*table).clone();
        match &a.action {
            AlterTableAction::AddColumns(defs) => {
                for def in defs {
                    let ColumnName::Ident(ident) = &def.name else {
                        return Err(unsupported("IDENTIFIER() column names"));
                    };
                    if changed
                        .columns
                        .iter()
                        .any(|c| c.name.eq_ignore_ascii_case(&ident.value))
                    {
                        return Err(krishiv_common::error::ColumnAlreadyExists {
                            column_name: format!("`{}`", ident.value.to_lowercase()),
                        }
                        .into());
                    }
                    changed.columns.push(StoreColumn {
                        name: ident.value.clone(),
                        data_type: def.data_type.to_data_type()?,
                        // Existing files have no such column: it reads as NULL.
                        nullable: true,
                        default: None,
                        comment: def.comment.clone(),
                    });
                }
            }
            AlterTableAction::SetTblProperties(props) => {
                changed.properties.extend(options_of(props, original));
            }
            AlterTableAction::UnsetTblProperties { keys, .. } => {
                for key in options_of(keys, original).keys() {
                    changed.properties.remove(key);
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
        if !t.partitioned_by.is_empty() {
            return Err(unsupported("partitioned tables"));
        }
        if t.bucket.is_some() || t.skew.is_some() || !t.cluster_by.is_empty() {
            return Err(unsupported("bucketed, skewed or clustered tables"));
        }
        let provider = self.provider_of(t.using.as_ref().map(|u| {
            u.parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join(".")
        }))?;
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
        let mut options = options_of(&t.options, original);
        let properties = options_of(&t.tblproperties, original);
        let external = t.external || t.location.is_some();
        let location = match &t.location {
            Some(l) if l.is_empty() => {
                return Err(krishiv_common::error::InvalidEmptyLocation {
                    location: l.clone(),
                }
                .into());
            }
            Some(l) => PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            None => managed_location(&warehouse, &name.namespace, &name.name),
        };
        if provider == "csv" {
            options
                .entry("header".to_owned())
                .or_insert_with(|| "true".to_owned());
        }
        let mut table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: !external,
            provider,
            location,
            columns: Vec::new(),
            options,
            properties,
            comment: t.comment.clone(),
            created_ms: now_ms(),
            root: warehouse.clone(),
            view: None,
        };
        match &t.as_query {
            Some(query) => self.create_table_as(table, query, mode).await?,
            None => {
                table.columns = self.columns_of(t, original)?;
                if table.columns.is_empty() {
                    return Err(unsupported("CREATE TABLE without columns"));
                }
                self.add_table(table, mode, false).await?;
            }
        }
        self.empty_frame().await
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
            if columns
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(&ident.value))
            {
                return Err(krishiv_common::error::ColumnAlreadyExists {
                    column_name: format!("`{}`", ident.value.to_lowercase()),
                }
                .into());
            }
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
        std::fs::create_dir_all(&location)
            .map_err(|e| internal(format!("cannot create {}: {e}", location.display())))?;
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
        mut table: CatalogTable,
        query: &Query,
        mode: CreateMode,
    ) -> Result<(), KrishivError> {
        let plan = krishiv_sql::lower(query, self.next_source_id()).map_err(|e| e.error)?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let analysis = df.analyzed_in(self.query_env()).await?;
        table.columns = analysis
            .analyzed
            .output
            .iter()
            .map(|c| StoreColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: true,
                default: None,
                comment: None,
            })
            .collect();
        if table.managed && table.location.exists() {
            return Err(LocationAlreadyExists {
                identifier: quoted(&table.namespace, &table.name),
                location: table.location.display().to_string(),
            }
            .into());
        }
        // `errorifexists`: a CTAS never replaces files that are already there (an external LOCATION with data).
        self.write_table_rows(&table, &df, "errorifexists").await?;
        self.add_table(table, mode, true).await
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
        let (columns, source_provider) = if let Some(s) = self
            .state
            .catalog
            .store
            .table(&source.namespace, &source.name)
        {
            (s.columns.clone(), s.provider.clone())
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
                        data_type: c.data_type.clone(),
                        nullable: c.nullable,
                        default: None,
                        comment: None,
                    })
                    .collect(),
                "parquet".to_owned(),
            )
        } else {
            return Err(source.not_found());
        };
        let provider = match &t.using {
            Some(u) => self.provider_of(Some(
                u.parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join("."),
            ))?,
            None => source_provider,
        };
        let external = t.location.is_some();
        let location = match &t.location {
            Some(l) => PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            None => managed_location(&warehouse, &name.namespace, &name.name),
        };
        let mut options = BTreeMap::new();
        if provider == "csv" {
            options.insert("header".to_owned(), "true".to_owned());
        }
        let table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: !external,
            provider,
            location,
            columns,
            options,
            properties: options_of(&t.tblproperties, original),
            comment: None,
            created_ms: now_ms(),
            root: warehouse.clone(),
            view: None,
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
        // `DROP TABLE` on a permanent view is refused as Spark's `DropTableCommand` does.
        if self.temp_entry_of(&name, &env) {
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
            self.forget_cached(&name);
            self.state
                .catalog
                .store
                .drop_table(&name.namespace, &name.name);
            if let Some(dir) = dir {
                remove_dir(&dir)?;
            }
        } else if !d.if_exists {
            return Err(name.not_found());
        }
        self.empty_frame().await
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
    pub(crate) async fn exec_analyze(&self, a: &Analyze) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&a.table, &env)?;
        let table = self.table_for_write(&name, "ANALYZE TABLE", &env)?;
        if a.partition.is_some() {
            return Err(unsupported("ANALYZE TABLE ... PARTITION"));
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
    pub(crate) fn exec_row_level_change(
        &self,
        table: &DdlName,
        operation: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(table, &env)?;
        self.table_for_write(&name, operation, &env)?;
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
        if self.temp_entry_of(name, env) {
            return Err(krishiv_common::error::ExpectTableNotViewNoAlternative {
                operation: format!("`{operation}`"),
                view_name: name.quoted(),
            }
            .into());
        }
        Err(name.not_found())
    }

    /// `INSERT INTO|OVERWRITE [TABLE] name [(columns)] [BY NAME] query`.
    pub(crate) async fn exec_insert(&self, i: &Insert) -> Result<DataFrame, KrishivError> {
        if i.source.is_some() || i.clauses.len() != 1 {
            return Err(unsupported("multi-table INSERT"));
        }
        for clause in &i.clauses {
            self.insert_clause(clause).await?;
        }
        self.empty_frame().await
    }

    async fn insert_clause(&self, c: &InsertClause) -> Result<(), KrishivError> {
        let env = self.query_env();
        if c.partition.is_some() {
            return Err(unsupported("INSERT ... PARTITION"));
        }
        if c.replace_where.is_some() || c.replace_on.is_some() || c.replace_using.is_some() {
            return Err(unsupported("INSERT ... REPLACE"));
        }
        let name = self.table_name(&c.name, &env)?;
        let table = self.table_for_write(&name, "INSERT", &env)?;
        let plan = krishiv_sql::lower(&c.query, self.next_source_id()).map_err(|e| e.error)?;
        let source = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let analysis = source.analyzed_in(self.query_env()).await?;
        if c.overwrite
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
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        let listed: Vec<String> = c
            .columns
            .iter()
            .map(|n| match n {
                ColumnName::Ident(i) => Ok(i.value.clone()),
                _ => Err(unsupported("IDENTIFIER() column names")),
            })
            .collect::<Result<_, _>>()?;
        let assignment = assign(
            &table,
            &listed,
            c.by_name,
            &output,
            env.analysis.conf.case_sensitive,
        )?;
        let renamed: Vec<String> = (0..output.len())
            .map(|i| format!("__krishiv_c{i}"))
            .collect();
        let base = source.to_df(renamed.iter());
        let mut select = Vec::with_capacity(table.columns.len());
        for (column, from) in table.columns.iter().zip(&assignment) {
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
        let mode = if c.overwrite { "overwrite" } else { "append" };
        let written = self.write_table_rows(&table, &df, mode).await;
        // Data may have changed even when the write failed late.
        self.state.catalog.store.bump();
        self.forget_cached(&name);
        written.map_err(|e| overflow_in_table_insert(e, &table, &output, &assignment))
    }

    /// Writes `df` to the files of `table` with the writer's save `mode`.
    async fn write_table_rows(
        &self,
        table: &CatalogTable,
        df: &DataFrame,
        mode: &str,
    ) -> Result<(), KrishivError> {
        let mut writer = df.write().format(&table.provider).mode(mode);
        for (k, v) in &table.options {
            writer = writer.option(k, v);
        }
        writer.save(&table.location.display().to_string()).await
    }
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
    let to_string = matches!(to, T::String)
        && (numeric(from)
            || matches!(
                from,
                T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary
            ));
    match (from, to) {
        (T::Null, _) => true,
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

fn now_ms() -> i64 {
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
mod tests {
    use super::*;
    use krishiv_types::{ArrayType, DecimalType};

    #[test]
    fn ansi_store_assignment() {
        use DataType as T;
        assert!(can_store(&T::Integer, &T::Integer));
        assert!(can_store(&T::Null, &T::Date));
        assert!(can_store(&T::Integer, &T::Byte));
        assert!(can_store(
            &T::Long,
            &T::Decimal(DecimalType::try_new(5, 2).unwrap())
        ));
        assert!(can_store(&T::Integer, &T::String));
        assert!(can_store(&T::Date, &T::String));
        assert!(can_store(&T::Date, &T::Timestamp));
        assert!(!can_store(&T::String, &T::Binary));
        assert!(!can_store(&T::String, &T::Integer));
        assert!(!can_store(&T::String, &T::Date));
        assert!(!can_store(&T::Boolean, &T::Integer));
        assert!(can_store(&T::Timestamp, &T::Date));
        let a = |t: T| T::Array(Arc::new(ArrayType::new(t, true)));
        assert!(can_store(&a(T::Integer), &a(T::Long)));
        assert!(!can_store(&a(T::String), &a(T::Long)));
    }

    #[test]
    fn literal_text_strips_one_pair_of_quotes() {
        assert_eq!(literal_text("'a b'"), "a b");
        assert_eq!(literal_text("\"x\""), "x");
        assert_eq!(literal_text(" true "), "true");
        assert_eq!(literal_text("'"), "'");
    }
}
