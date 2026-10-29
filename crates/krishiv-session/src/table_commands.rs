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
//! the table's directory with the session's writer (`INSERT OVERWRITE` is
//! its atomic overwrite). A CTAS adds its metadata only after the commit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use krishiv_common::error::{
    DataSourceNotFound, IncompatibleDataForTableCannotSafelyCast,
    IncompatibleDataForTableExtraColumns, InsertColumnArityMismatchNotEnoughDataColumns,
    InsertColumnArityMismatchTooManyDataColumns, InternalError, KrishivError,
    KrishivUnsupportedFeature, LocationAlreadyExists, NotSupportedCommandWithoutHiveSupport,
    SchemaNotFound, TableOrViewNotFound, WrongCommandForObjectType,
};
use krishiv_common::redact::RedactedSql;
use krishiv_plan::analyzer::TableColumn;
use krishiv_sql::ast::{
    ColumnName, CreateTable, CreateTableLike, DdlName, DropTable, Insert, InsertClause, Query,
    TableElement, TableOption, Truncate,
};
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
use crate::functions::{col, expr, lit_null};
use crate::session::{QueryEnv, Session};
use crate::store::{
    CatalogStore, CatalogTable, CreateMode, CreateOutcome, DEFAULT_NAMESPACE, StoreColumn,
    managed_location,
};

/// The file formats a table can have.
const PROVIDERS: &[&str] = &["parquet", "csv", "json", "text", "orc"];

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

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// A table named in a command: its namespace and name, folded unless the
/// session is case sensitive.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TableName {
    namespace: String,
    name: String,
    /// The name had no namespace part.
    bare: bool,
}

impl TableName {
    fn resolve(parts: &[String], env: &QueryEnv) -> Result<Self, KrishivError> {
        let fold = |s: &str| {
            if env.analysis.conf.case_sensitive {
                s.to_owned()
            } else {
                s.to_lowercase()
            }
        };
        let (namespace, name, bare) = match parts {
            [name] => (DEFAULT_NAMESPACE.to_owned(), fold(name), true),
            [ns, name] => (fold(ns), fold(name), false),
            [catalog, ns, name] if catalog.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                (fold(ns), fold(name), false)
            }
            [catalog, _, _] => {
                return Err(unsupported(&format!("tables in the catalog `{catalog}`")));
            }
            _ => {
                return Err(KrishivError::from(
                    krishiv_common::error::RequiresSinglePartNamespace {
                        session_catalog: crate::catalog::CATALOG.to_owned(),
                        identifier: format!("`{}`", parts.join("`.`")),
                    },
                ));
            }
        };
        Ok(Self {
            namespace,
            name,
            bare,
        })
    }

    fn quoted(&self) -> String {
        quoted(&self.namespace, &self.name)
    }
}

impl Session {
    fn table_name(&self, name: &DdlName, env: &QueryEnv) -> Result<TableName, KrishivError> {
        TableName::resolve(&crate::access::ddl_parts(name, env)?, env)
    }

    fn warehouse_dir(&self) -> Result<PathBuf, KrishivError> {
        let configured = crate::conf::get(&self.state.conf.snapshot(), "spark.sql.warehouse.dir");
        self.state
            .catalog
            .warehouse
            .root(configured.as_deref())
            .map_err(|e| internal(format!("cannot create the warehouse directory: {e}")))
    }

    /// Whether `name` is a temporary view (or table) that a bare table name
    /// resolves to before any permanent table.
    fn temp_entry_of(&self, name: &TableName, env: &QueryEnv) -> bool {
        name.bare
            && self
                .state
                .catalog
                .exists(&name.name, env.analysis.conf.case_sensitive)
    }

    /// `CREATE [OR REPLACE] [EXTERNAL] TABLE ...`, with or without columns
    /// and a query.
    pub(crate) async fn exec_create_table(
        &self,
        t: &CreateTable,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
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
        self.check_namespace(&name)?;
        let mode = match (t.or_replace, t.if_not_exists) {
            (true, _) => CreateMode::Replace,
            (false, true) => CreateMode::IfNotExists,
            (false, false) => CreateMode::ErrorIfExists,
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
        }
        let mut options = options_of(&t.options, text);
        let properties = options_of(&t.tblproperties, text);
        let external = t.external || t.location.is_some();
        let location = match &t.location {
            Some(l) if l.is_empty() => {
                return Err(krishiv_common::error::InvalidEmptyLocation {
                    location: l.clone(),
                }
                .into());
            }
            Some(l) => PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            None => managed_location(&self.warehouse_dir()?, &name.namespace, &name.name),
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
        };
        match &t.as_query {
            Some(query) => self.create_table_as(table, query, mode).await?,
            None => {
                table.columns = self.columns_of(t, text)?;
                if table.columns.is_empty() {
                    return Err(unsupported("CREATE TABLE without columns"));
                }
                self.add_table(table, mode, false)?;
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
        text: &RedactedSql,
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
                nullable: !def.not_null,
                default: def
                    .default
                    .as_ref()
                    .map(|e| text.fragment(e.span.start..e.span.end).to_owned()),
                comment: def.comment.clone(),
            });
        }
        Ok(columns)
    }

    /// Adds `table` to the store, creating its directory first. A replaced
    /// managed table's directory is deleted after the metadata changed.
    fn add_table(
        &self,
        table: CatalogTable,
        mode: CreateMode,
        written: bool,
    ) -> Result<(), KrishivError> {
        let name = TableName {
            namespace: table.namespace.clone(),
            name: table.name.clone(),
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
        if !written && table.managed && table.location.exists() && mode != CreateMode::Replace {
            return Err(LocationAlreadyExists {
                identifier: name.quoted(),
                location: table.location.display().to_string(),
            }
            .into());
        }
        let location = table.location.clone();
        std::fs::create_dir_all(&location)
            .map_err(|e| internal(format!("cannot create {}: {e}", location.display())))?;
        let outcome = self.state.catalog.store.create_table(table, mode)?;
        if let CreateOutcome::Created(Some(old)) = &outcome
            && old.managed
        {
            if old.location == location {
                // `OR REPLACE` over a managed table of the same location:
                // its old files are replaced by an empty table.
                clear_dir(&old.location)?;
            } else {
                remove_dir(&old.location)?;
            }
        }
        Ok(())
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
        if table.managed && table.location.exists() && mode != CreateMode::Replace {
            return Err(LocationAlreadyExists {
                identifier: quoted(&table.namespace, &table.name),
                location: table.location.display().to_string(),
            }
            .into());
        }
        self.write_table_rows(&table, &df, "overwrite").await?;
        self.add_table(table, mode, true)
    }

    /// `CREATE TABLE target LIKE source`.
    pub(crate) async fn exec_create_table_like(
        &self,
        t: &CreateTableLike,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: "CREATE Hive TABLE (AS SELECT)".to_owned(),
            }
            .into());
        }
        let name = self.table_name(&t.name, &env)?;
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
            return Err(TableOrViewNotFound {
                relation_name: source.quoted(),
                search_path: "[]".to_owned(),
            }
            .into());
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
            None => managed_location(&self.warehouse_dir()?, &name.namespace, &name.name),
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
            properties: options_of(&t.tblproperties, text),
            comment: None,
            created_ms: now_ms(),
        };
        let mode = if t.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        self.add_table(table, mode, false)?;
        self.empty_frame().await
    }

    /// `DROP TABLE [IF EXISTS] name [PURGE]`.
    pub(crate) async fn exec_drop_table(&self, d: &DropTable) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&d.name, &env)?;
        match self
            .state
            .catalog
            .store
            .drop_table(&name.namespace, &name.name)
        {
            Some(table) => {
                if table.managed {
                    remove_dir(&table.location)?;
                }
            }
            None if self.temp_entry_of(&name, &env) => {
                return Err(WrongCommandForObjectType {
                    operation: "DROP TABLE".to_owned(),
                    required_type: "EXTERNAL or MANAGED".to_owned(),
                    object_name: name.quoted(),
                    found_type: "VIEW".to_owned(),
                    alternative: "DROP VIEW".to_owned(),
                }
                .into());
            }
            None if d.if_exists => {}
            None => {
                return Err(TableOrViewNotFound {
                    relation_name: name.quoted(),
                    search_path: "[]".to_owned(),
                }
                .into());
            }
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
        clear_dir(&table.location)?;
        self.state.catalog.store.bump();
        self.empty_frame().await
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
            return Ok(table);
        }
        if self.temp_entry_of(name, env) {
            return Err(krishiv_common::error::ExpectTableNotViewNoAlternative {
                operation: format!("`{operation}`"),
                view_name: name.quoted(),
            }
            .into());
        }
        Err(TableOrViewNotFound {
            relation_name: name.quoted(),
            search_path: "[]".to_owned(),
        }
        .into())
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
                        value.cast(column.data_type.clone())
                    }
                }
                Source::Default => match &column.default {
                    Some(text) => expr(text)?.cast(column.data_type.clone()),
                    None => lit_null().cast(column.data_type.clone()),
                },
            };
            select.push(value.alias(&column.name));
        }
        let df = base.select(select);
        let mode = if c.overwrite { "overwrite" } else { "append" };
        self.write_table_rows(&table, &df, mode).await?;
        self.state.catalog.store.bump();
        Ok(())
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
                return Err(IncompatibleDataForTableExtraColumns {
                    table_name,
                    extra_columns: format!("`{}`", o.name),
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
                return Err(IncompatibleDataForTableExtraColumns {
                    table_name,
                    extra_columns: format!("`{name}`"),
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
        // Binary from a string, dates and timestamps among themselves.
        (T::String, T::Binary)
        | (T::Date, T::Timestamp | T::TimestampNtz)
        | (T::Timestamp, T::TimestampNtz)
        | (T::TimestampNtz, T::Timestamp) => true,
        (T::Array(a), T::Array(b)) => can_store(a.element_type(), b.element_type()),
        _ => false,
    }
}

/// `OPTIONS (...)` / `TBLPROPERTIES (...)` as text pairs, keys lower-cased
/// for options as Spark does not (kept as written).
fn options_of(options: &[TableOption], text: &RedactedSql) -> BTreeMap<String, String> {
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
                .map(|v| literal_text(text.fragment(v.span.start..v.span.end)))
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
        assert!(can_store(&T::String, &T::Binary));
        assert!(!can_store(&T::String, &T::Integer));
        assert!(!can_store(&T::String, &T::Date));
        assert!(!can_store(&T::Boolean, &T::Integer));
        assert!(!can_store(&T::Timestamp, &T::Date));
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
