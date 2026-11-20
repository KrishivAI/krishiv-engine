// SPDX-License-Identifier: Apache-2.0
//! `Relation.catalog` (Spark's `SparkConnectPlanner.transformCatalog`, 36 operations): the call a
//! Connect `Catalog` message makes on the session's `spark.catalog`.
//!
//! The translator cannot run the call (it is synchronous and the catalog is the session's), so it
//! reads the message into a [`CatalogCall`] and records it as a [`super::LocalRequest::Catalog`]; the
//! host runs the call before the second translation and answers the relation with its result rows
//! (Spark likewise runs the call while planning and wraps the result in a local relation). Which
//! arguments a call has, and the choice between the one-name and the database-and-name overload,
//! are Spark's: an arm with a `db_name` calls the two-argument form, one without it the current
//! database's.

use std::collections::BTreeMap;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_types::DataType;

use super::errors::assertion;
use super::types::data_type;
use crate::proto::v4_3 as pb;
use crate::proto::v4_3::catalog::CatType;

/// One call of Spark's `Catalog` API, as a Connect `Catalog` message makes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogCall {
    /// `currentDatabase`: one string row.
    CurrentDatabase,
    /// `setCurrentDatabase(dbName)`: no rows.
    SetCurrentDatabase(String),
    /// `listDatabases([pattern])`.
    ListDatabases(Option<String>),
    /// `listTables([dbName], [pattern])`.
    ListTables(Option<String>, Option<String>),
    /// `listFunctions([dbName], [pattern])`.
    ListFunctions(Option<String>, Option<String>),
    /// `listColumns([dbName,] tableName)`.
    ListColumns {
        /// The database, when the message names one.
        db: Option<String>,
        /// The table.
        table: String,
    },
    /// `getDatabase(dbName)`.
    GetDatabase(String),
    /// `getTable([dbName,] tableName)`.
    GetTable {
        /// The database, when the message names one.
        db: Option<String>,
        /// The table.
        table: String,
    },
    /// `getFunction([dbName,] functionName)`.
    GetFunction {
        /// The database, when the message names one.
        db: Option<String>,
        /// The function.
        function: String,
    },
    /// `databaseExists(dbName)`: one boolean row.
    DatabaseExists(String),
    /// `tableExists([dbName,] tableName)`: one boolean row.
    TableExists {
        /// The database, when the message names one.
        db: Option<String>,
        /// The table.
        table: String,
    },
    /// `functionExists([dbName,] functionName)`: one boolean row.
    FunctionExists {
        /// The database, when the message names one.
        db: Option<String>,
        /// The function.
        function: String,
    },
    /// `createTable` (also `createExternalTable`, which differs only in the missing description).
    CreateTable(CreateTable),
    /// `dropTempView(viewName)`: one boolean row.
    DropTempView(String),
    /// `dropGlobalTempView(viewName)`: one boolean row.
    DropGlobalTempView(String),
    /// `recoverPartitions(tableName)`: no rows.
    RecoverPartitions(String),
    /// `isCached(tableName)`: one boolean row.
    IsCached(String),
    /// `cacheTable(tableName[, storageLevel])`: no rows.
    CacheTable {
        /// The table.
        table: String,
        /// The storage level, when the message sets one.
        level: Option<pb::StorageLevel>,
    },
    /// `uncacheTable(tableName)`: no rows.
    UncacheTable(String),
    /// `clearCache()`: no rows.
    ClearCache,
    /// `refreshTable(tableName)`: no rows.
    RefreshTable(String),
    /// `refreshByPath(path)`: no rows.
    RefreshByPath(String),
    /// `currentCatalog()`: one string row.
    CurrentCatalog,
    /// `setCurrentCatalog(catalogName)`: no rows.
    SetCurrentCatalog(String),
    /// `listCatalogs([pattern])`.
    ListCatalogs(Option<String>),
    /// `dropTable(tableName, ifExists, purge)`: no rows.
    DropTable {
        /// The table.
        table: String,
        /// `IF EXISTS`.
        if_exists: bool,
        /// `PURGE`.
        purge: bool,
    },
    /// `dropView(viewName, ifExists)`: no rows.
    DropView {
        /// The view.
        view: String,
        /// `IF EXISTS`.
        if_exists: bool,
    },
    /// `createDatabase(dbName, ifNotExists, properties)`: no rows.
    CreateDatabase {
        /// The database.
        db: String,
        /// `IF NOT EXISTS`.
        if_not_exists: bool,
        /// `WITH DBPROPERTIES`.
        properties: BTreeMap<String, String>,
    },
    /// `dropDatabase(dbName, ifExists, cascade)`: no rows.
    DropDatabase {
        /// The database.
        db: String,
        /// `IF EXISTS`.
        if_exists: bool,
        /// `CASCADE`.
        cascade: bool,
    },
    /// `listPartitions(tableName)`.
    ListPartitions(String),
    /// `listViews([dbName], [pattern])`.
    ListViews(Option<String>, Option<String>),
    /// `getTableProperties(tableName)`: `key`, `value` rows.
    GetTableProperties(String),
    /// `getCreateTableString(tableName, asSerde)`: one string row.
    GetCreateTableString {
        /// The table or view.
        table: String,
        /// `AS SERDE`.
        as_serde: bool,
    },
    /// `truncateTable(tableName)`: no rows.
    TruncateTable(String),
    /// `analyzeTable(tableName, noScan)`: no rows.
    AnalyzeTable {
        /// The table.
        table: String,
        /// `NOSCAN`.
        no_scan: bool,
    },
}

/// What `createTable` and `createExternalTable` ask for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTable {
    /// The table.
    pub table: String,
    /// The data source (`spark.sql.sources.default` when absent).
    pub source: Option<String>,
    /// The columns as a DDL column list; `None` for no schema (the files give it).
    pub schema: Option<String>,
    /// A column comment holds a quote or backslash: it cannot be written exactly under
    /// `spark.sql.parser.escapedStringLiterals`, so the host refuses it in that mode.
    pub lossy_in_escaped_mode: bool,
    /// The description (`createTable` only; Spark sends an empty one when unset).
    pub description: String,
    /// The options, with the `path` option last when the message has a `path`.
    pub options: Vec<(String, String)>,
}

/// Reads a `Catalog` message.
///
/// # Errors
/// `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET` when no operation is set;
/// `CONNECT_INVALID_PLAN.ASSERTION_FAILURE` for a schema that is not a struct (Spark's `assertPlan`),
/// and the data-type errors of a schema.
pub fn catalog_call(catalog: &pb::Catalog) -> Result<CatalogCall, KrishivError> {
    let Some(kind) = &catalog.cat_type else {
        return Err(super::errors::not_set(
            "spark.connect.Catalog",
            "CATTYPE_NOT_SET",
        ));
    };
    Ok(match kind {
        CatType::CurrentDatabase(_) => CatalogCall::CurrentDatabase,
        CatType::SetCurrentDatabase(m) => CatalogCall::SetCurrentDatabase(m.db_name.clone()),
        CatType::ListDatabases(m) => CatalogCall::ListDatabases(m.pattern.clone()),
        CatType::ListTables(m) => CatalogCall::ListTables(m.db_name.clone(), m.pattern.clone()),
        CatType::ListFunctions(m) => {
            CatalogCall::ListFunctions(m.db_name.clone(), m.pattern.clone())
        }
        CatType::ListColumns(m) => CatalogCall::ListColumns {
            db: m.db_name.clone(),
            table: m.table_name.clone(),
        },
        CatType::GetDatabase(m) => CatalogCall::GetDatabase(m.db_name.clone()),
        CatType::GetTable(m) => CatalogCall::GetTable {
            db: m.db_name.clone(),
            table: m.table_name.clone(),
        },
        CatType::GetFunction(m) => CatalogCall::GetFunction {
            db: m.db_name.clone(),
            function: m.function_name.clone(),
        },
        CatType::DatabaseExists(m) => CatalogCall::DatabaseExists(m.db_name.clone()),
        CatType::TableExists(m) => CatalogCall::TableExists {
            db: m.db_name.clone(),
            table: m.table_name.clone(),
        },
        CatType::FunctionExists(m) => CatalogCall::FunctionExists {
            db: m.db_name.clone(),
            function: m.function_name.clone(),
        },
        CatType::CreateExternalTable(m) => CatalogCall::CreateTable(create_table(
            &m.table_name,
            m.source.as_ref(),
            m.schema.as_ref(),
            None,
            &m.options,
            m.path.as_ref(),
        )?),
        CatType::CreateTable(m) => CatalogCall::CreateTable(create_table(
            &m.table_name,
            m.source.as_ref(),
            m.schema.as_ref(),
            Some(m.description.as_deref().unwrap_or("")),
            &m.options,
            m.path.as_ref(),
        )?),
        CatType::DropTempView(m) => CatalogCall::DropTempView(m.view_name.clone()),
        CatType::DropGlobalTempView(m) => CatalogCall::DropGlobalTempView(m.view_name.clone()),
        CatType::RecoverPartitions(m) => CatalogCall::RecoverPartitions(m.table_name.clone()),
        CatType::IsCached(m) => CatalogCall::IsCached(m.table_name.clone()),
        CatType::CacheTable(m) => CatalogCall::CacheTable {
            table: m.table_name.clone(),
            level: m.storage_level,
        },
        CatType::UncacheTable(m) => CatalogCall::UncacheTable(m.table_name.clone()),
        CatType::ClearCache(_) => CatalogCall::ClearCache,
        CatType::RefreshTable(m) => CatalogCall::RefreshTable(m.table_name.clone()),
        CatType::RefreshByPath(m) => CatalogCall::RefreshByPath(m.path.clone()),
        CatType::CurrentCatalog(_) => CatalogCall::CurrentCatalog,
        CatType::SetCurrentCatalog(m) => CatalogCall::SetCurrentCatalog(m.catalog_name.clone()),
        CatType::ListCatalogs(m) => CatalogCall::ListCatalogs(m.pattern.clone()),
        CatType::DropTable(m) => CatalogCall::DropTable {
            table: m.table_name.clone(),
            if_exists: m.if_exists,
            purge: m.purge,
        },
        CatType::DropView(m) => CatalogCall::DropView {
            view: m.view_name.clone(),
            if_exists: m.if_exists,
        },
        CatType::CreateDatabase(m) => CatalogCall::CreateDatabase {
            db: m.db_name.clone(),
            if_not_exists: m.if_not_exists,
            properties: m.properties.clone(),
        },
        CatType::DropDatabase(m) => CatalogCall::DropDatabase {
            db: m.db_name.clone(),
            if_exists: m.if_exists,
            cascade: m.cascade,
        },
        CatType::ListPartitions(m) => CatalogCall::ListPartitions(m.table_name.clone()),
        CatType::ListViews(m) => CatalogCall::ListViews(m.db_name.clone(), m.pattern.clone()),
        CatType::GetTableProperties(m) => CatalogCall::GetTableProperties(m.table_name.clone()),
        CatType::GetCreateTableString(m) => CatalogCall::GetCreateTableString {
            table: m.table_name.clone(),
            as_serde: m.as_serde,
        },
        CatType::TruncateTable(m) => CatalogCall::TruncateTable(m.table_name.clone()),
        CatType::AnalyzeTable(m) => CatalogCall::AnalyzeTable {
            table: m.table_name.clone(),
            no_scan: m.no_scan,
        },
    })
}

/// Visits every comment of the struct fields nested inside `t` (arrays, maps and structs): the type's
/// `sql()` writes them with only the quote escaped, as Spark's `DataType.sql` does.
fn nested_comments<'a>(t: &'a DataType, out: &mut Vec<&'a str>) {
    match t {
        DataType::Array(a) => nested_comments(a.element_type(), out),
        DataType::Map(m) => {
            nested_comments(m.key_type(), out);
            nested_comments(m.value_type(), out);
        }
        DataType::Struct(s) => {
            for f in s.fields() {
                if let Some(c) = f.metadata().comment() {
                    out.push(c);
                }
                nested_comments(f.data_type(), out);
            }
        }
        _ => {}
    }
}

/// `transformCreateTable` / `transformCreateExternalTable`: the `path` becomes the `path` option
/// (it wins over one in `options`), and the schema must be a struct.
fn create_table(
    table: &str,
    source: Option<&String>,
    schema: Option<&pb::DataType>,
    description: Option<&str>,
    options: &BTreeMap<String, String>,
    path: Option<&String>,
) -> Result<CreateTable, KrishivError> {
    let mut lossy_in_escaped_mode = false;
    let schema = schema
        .map(|t| match data_type(t)? {
            DataType::Struct(s) => Ok(s
                .fields()
                .iter()
                .map(|f| -> Result<String, KrishivError> {
                    let mut nested = Vec::new();
                    nested_comments(f.data_type(), &mut nested);
                    // A nested comment is written with its quote escaped only (Spark's `DataType.sql`):
                    // a backslash in it would be read as an escape, so it cannot be stored as given.
                    if nested.iter().any(|c| c.contains('\\')) {
                        return Err(KrishivError::from(KrishivUnsupportedFeature {
                            feature: "a backslash in the comment of a nested struct field"
                                .to_owned(),
                        }));
                    }
                    lossy_in_escaped_mode |= nested.iter().any(|c| c.contains('\''));
                    let mut column =
                        format!("`{}` {}", f.name().replace('`', "``"), f.data_type().sql());
                    if !f.nullable() {
                        column.push_str(" NOT NULL");
                    }
                    // A field's comment is the column's `COMMENT` (`listColumns` shows it).
                    if let Some(comment) = f.metadata().comment() {
                        lossy_in_escaped_mode |= comment.contains(['\\', '\'']);
                        column.push_str(&format!(
                            " COMMENT '{}'",
                            comment.replace('\\', "\\\\").replace('\'', "\\'")
                        ));
                    }
                    Ok(column)
                })
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")),
            _ => Err(assertion("the schema of a table must be a struct")),
        })
        .transpose()?
        // An empty struct is Spark's `new StructType`: no schema.
        .filter(|s| !s.is_empty());
    let mut options: Vec<(String, String)> = options
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let Some(path) = path {
        options.push(("path".to_owned(), path.clone()));
    }
    Ok(CreateTable {
        table: table.to_owned(),
        source: source.cloned(),
        schema,
        description: description.unwrap_or("").to_owned(),
        options,
        lossy_in_escaped_mode,
    })
}

#[cfg(test)]
mod tests;
