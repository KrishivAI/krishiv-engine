// SPDX-License-Identifier: Apache-2.0
//! The `Catalog` relation (Spark's `SparkConnectPlanner.transformCatalog`): a call of the session's
//! `spark.catalog` ([`krishiv_session::Session::catalog`]) run while the plan is prepared, its
//! result rows held as the local relation the plan reads. Spark does the same: it makes the call
//! while planning and wraps the result (a `Dataset` of `Database`, `Table`, `Column`, `Function`,
//! `CatalogMetadata` or `TablePartition`, a one-value `Dataset[String]` or `Dataset[Boolean]`, or the
//! empty relation of a call that returns nothing). Column names and nullability are those of the
//! Scala classes Spark derives the rows' encoders from: a `String` or an array is nullable, a
//! `Boolean` is not.

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_connect::translate::{CatalogCall, CreateTable};
use krishiv_plan::LocalRelation;
use krishiv_session::Session;
use krishiv_session::spark_catalog::{Catalog, Column, Database, Function, Table};

use crate::auth::{Check, enforce};
use crate::cache::storage_level_of;

fn internal(what: &str, e: impl std::fmt::Display) -> KrishivError {
    InternalError {
        message: format!("{what}: {e}"),
    }
    .into()
}

fn string_field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Utf8, nullable)
}

fn bool_field(name: &str) -> Field {
    Field::new(name, DataType::Boolean, false)
}

fn namespace_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    )
}

fn strings<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(StringArray::from_iter(values))
}

fn bools(values: impl IntoIterator<Item = bool>) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(values.into_iter().map(Some)))
}

fn namespaces<'a>(values: impl IntoIterator<Item = Option<&'a Vec<String>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for v in values {
        match v {
            Some(parts) => {
                for p in parts {
                    builder.values().append_value(p);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> Result<RecordBatch, KrishivError> {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| internal("a catalog result", e))
}

/// One nullable string column `value`: `Dataset[String]`.
fn one_string(value: &str) -> Result<RecordBatch, KrishivError> {
    batch(
        vec![string_field("value", true)],
        vec![strings([Some(value)])],
    )
}

/// One boolean column `value`: `Dataset[Boolean]`.
fn one_bool(value: bool) -> Result<RecordBatch, KrishivError> {
    batch(vec![bool_field("value")], vec![bools([value])])
}

/// Spark's `emptyLocalRelation`: no rows, a `value` string column that is not nullable.
///
/// `createTable` / `createExternalTable` answer this too; Spark answers a read of the new table.
/// The PySpark client discards the eager result, so the difference shows only on a second use of
/// the returned frame.
fn nothing() -> Result<RecordBatch, KrishivError> {
    batch(
        vec![string_field("value", false)],
        vec![strings(std::iter::empty())],
    )
}

fn databases(list: &[Database]) -> Result<RecordBatch, KrishivError> {
    batch(
        vec![
            string_field("name", true),
            string_field("catalog", true),
            string_field("description", true),
            string_field("locationUri", true),
        ],
        vec![
            strings(list.iter().map(|d| Some(d.name.as_str()))),
            strings(list.iter().map(|d| Some(d.catalog.as_str()))),
            strings(list.iter().map(|d| d.description.as_deref())),
            strings(list.iter().map(|d| Some(d.location_uri.as_str()))),
        ],
    )
}

fn tables(list: &[Table]) -> Result<RecordBatch, KrishivError> {
    batch(
        vec![
            string_field("name", true),
            string_field("catalog", true),
            namespace_field("namespace"),
            string_field("description", true),
            string_field("tableType", true),
            bool_field("isTemporary"),
        ],
        vec![
            strings(list.iter().map(|t| Some(t.name.as_str()))),
            strings(list.iter().map(|t| t.catalog.as_deref())),
            namespaces(list.iter().map(|t| t.namespace.as_ref())),
            strings(list.iter().map(|t| t.description.as_deref())),
            strings(list.iter().map(|t| Some(t.table_type.as_str()))),
            bools(list.iter().map(|t| t.is_temporary)),
        ],
    )
}

fn columns(list: &[Column]) -> Result<RecordBatch, KrishivError> {
    batch(
        vec![
            string_field("name", true),
            string_field("description", true),
            string_field("dataType", true),
            bool_field("nullable"),
            bool_field("isPartition"),
            bool_field("isBucket"),
            bool_field("isCluster"),
        ],
        vec![
            strings(list.iter().map(|c| Some(c.name.as_str()))),
            strings(list.iter().map(|c| c.description.as_deref())),
            strings(list.iter().map(|c| Some(c.data_type.as_str()))),
            bools(list.iter().map(|c| c.nullable)),
            bools(list.iter().map(|c| c.is_partition)),
            bools(list.iter().map(|c| c.is_bucket)),
            bools(list.iter().map(|c| c.is_cluster)),
        ],
    )
}

fn functions(list: &[Function]) -> Result<RecordBatch, KrishivError> {
    batch(
        vec![
            string_field("name", true),
            string_field("catalog", true),
            namespace_field("namespace"),
            string_field("description", true),
            string_field("className", true),
            bool_field("isTemporary"),
        ],
        vec![
            strings(list.iter().map(|f| Some(f.name.as_str()))),
            strings(list.iter().map(|f| f.catalog.as_deref())),
            namespaces(list.iter().map(|f| f.namespace.as_ref())),
            strings(list.iter().map(|f| f.description.as_deref())),
            strings(list.iter().map(|f| Some(f.class_name.as_str()))),
            bools(list.iter().map(|f| f.is_temporary)),
        ],
    )
}

/// The name of a call's table or function: `db.name` when the message names a database.
fn named(db: Option<&str>, name: &str) -> String {
    db.map_or_else(|| name.to_owned(), |d| Catalog::qualified(d, name))
}

/// Runs `call` on the session and returns its result rows.
///
/// # Errors
/// The call's own (`TABLE_OR_VIEW_NOT_FOUND`, `SCHEMA_NOT_FOUND`, the access errors of the SQL
/// statement that stands for it, ...), and the operator floor (`Check::SideEffect`) for the
/// calls that change the cache.
async fn rows(session: &Session, call: CatalogCall) -> Result<RecordBatch, KrishivError> {
    let c = session.catalog();
    match call {
        CatalogCall::CurrentDatabase => one_string(&c.current_database()),
        CatalogCall::SetCurrentDatabase(db) => {
            c.set_current_database(&db).await?;
            nothing()
        }
        CatalogCall::ListDatabases(pattern) => {
            databases(&c.list_databases(pattern.as_deref()).await?)
        }
        CatalogCall::ListTables(db, pattern) => {
            tables(&c.list_tables(db.as_deref(), pattern.as_deref()).await?)
        }
        CatalogCall::ListFunctions(db, pattern) => {
            // Spark's `listFunctions(db)` requires the database to exist.
            functions(
                &c.list_functions_in(db.as_deref(), pattern.as_deref())
                    .await?,
            )
        }
        CatalogCall::ListColumns { db, table } => {
            columns(&c.list_columns(&named(db.as_deref(), &table)).await?)
        }
        CatalogCall::GetDatabase(db) => databases(&[c.get_database(&db).await?]),
        CatalogCall::GetTable { db, table } => {
            tables(&[c.get_table(&named(db.as_deref(), &table)).await?])
        }
        CatalogCall::GetFunction { db, function } => {
            functions(&[c.get_function(&named(db.as_deref(), &function)).await?])
        }
        CatalogCall::DatabaseExists(db) => one_bool(c.database_exists(&db).await?),
        CatalogCall::TableExists { db, table } => {
            one_bool(c.table_exists(&named(db.as_deref(), &table)).await?)
        }
        CatalogCall::FunctionExists { db, function } => {
            one_bool(c.function_exists(&named(db.as_deref(), &function)).await?)
        }
        CatalogCall::CreateTable(CreateTable {
            table,
            source,
            schema,
            description,
            options,
            lossy_in_escaped_mode,
        }) => {
            // Under `escapedStringLiterals` a quote or backslash in a column comment cannot be
            // written back as given: refuse it rather than store altered documentation.
            if lossy_in_escaped_mode
                && session
                    .conf()
                    .get("spark.sql.parser.escapedStringLiterals")
                    .is_some_and(|v| v.eq_ignore_ascii_case("true"))
            {
                return Err(krishiv_common::error::KrishivUnsupportedFeature {
                    feature: "a column comment holding a quote or backslash under \
                              spark.sql.parser.escapedStringLiterals"
                        .to_owned(),
                }
                .into());
            }
            let options: Vec<(&str, &str)> = options
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            c.create_table(
                &table,
                source.as_deref(),
                schema.as_deref(),
                Some(description.as_str()),
                &options,
            )
            .await?;
            // Spark answers the created table's rows (an empty relation of its schema); the
            // client reads no rows from it.
            nothing()
        }
        CatalogCall::DropTempView(view) => one_bool(c.drop_temp_view(&view)),
        CatalogCall::DropGlobalTempView(view) => one_bool(c.drop_global_temp_view(&view).await?),
        CatalogCall::RecoverPartitions(table) => {
            c.recover_partitions(&table).await?;
            nothing()
        }
        CatalogCall::IsCached(table) => one_bool(c.is_cached(&table).await?),
        CatalogCall::CacheTable { table, level } => {
            enforce(&Check::SideEffect, session).await?;
            match level {
                Some(level) => {
                    // A missing or forbidden table fails this request, as `CACHE TABLE` does.
                    c.check_cache_target(&table).await?;
                    // The frame holds the cache entry; the level is the client's.
                    let _held = session.table(&table).persist(storage_level_of(&level));
                }
                None => c.cache_table(&table).await?,
            }
            nothing()
        }
        CatalogCall::UncacheTable(table) => {
            enforce(&Check::SideEffect, session).await?;
            c.uncache_table(&table).await?;
            nothing()
        }
        CatalogCall::ClearCache => {
            enforce(&Check::SideEffect, session).await?;
            c.clear_cache().await?;
            nothing()
        }
        CatalogCall::RefreshTable(table) => {
            c.refresh_table(&table).await?;
            nothing()
        }
        CatalogCall::RefreshByPath(path) => {
            c.refresh_by_path(&path).await?;
            nothing()
        }
        CatalogCall::CurrentCatalog => one_string(&c.current_catalog()),
        CatalogCall::SetCurrentCatalog(name) => {
            c.set_current_catalog(&name).await?;
            nothing()
        }
        CatalogCall::ListCatalogs(pattern) => {
            let names = c.list_catalogs_matching(pattern.as_deref());
            batch(
                vec![
                    string_field("name", true),
                    string_field("description", true),
                ],
                vec![
                    strings(names.iter().map(|n| Some(n.as_str()))),
                    strings(names.iter().map(|_| None)),
                ],
            )
        }
        CatalogCall::DropTable {
            table,
            if_exists,
            purge,
        } => {
            c.drop_table(&table, if_exists, purge).await?;
            nothing()
        }
        CatalogCall::DropView { view, if_exists } => {
            c.drop_view(&view, if_exists).await?;
            nothing()
        }
        CatalogCall::CreateDatabase {
            db,
            if_not_exists,
            properties,
        } => {
            c.create_database(&db, if_not_exists, &properties).await?;
            nothing()
        }
        CatalogCall::DropDatabase {
            db,
            if_exists,
            cascade,
        } => {
            c.drop_database(&db, if_exists, cascade).await?;
            nothing()
        }
        CatalogCall::ListPartitions(table) => {
            let parts = c.list_partitions(&table).await?;
            batch(
                vec![string_field("partition", true)],
                vec![strings(parts.iter().map(|p| Some(p.as_str())))],
            )
        }
        CatalogCall::ListViews(db, pattern) => {
            tables(&c.list_views(db.as_deref(), pattern.as_deref()).await?)
        }
        CatalogCall::GetTableProperties(table) => {
            // Rows come in key order; Spark's come from a `HashMap` (unordered), so the order is Krishiv's.
            let props = c.get_table_properties(&table).await?;
            batch(
                vec![string_field("key", false), string_field("value", true)],
                vec![
                    strings(props.keys().map(|k| Some(k.as_str()))),
                    strings(props.values().map(|v| Some(v.as_str()))),
                ],
            )
        }
        CatalogCall::GetCreateTableString { table, as_serde } => {
            one_string(&c.get_create_table_string(&table, as_serde).await?)
        }
        CatalogCall::TruncateTable(table) => {
            c.truncate_table(&table).await?;
            nothing()
        }
        CatalogCall::AnalyzeTable { table, no_scan } => {
            c.analyze_table(&table, no_scan).await?;
            nothing()
        }
    }
}

/// The relation a `Catalog` call answers: the call run, its rows encoded as an Arrow IPC stream
/// and held by the session as a local relation.
///
/// # Errors
/// The call's errors, see [`rows`].
pub(super) async fn run(
    session: &Session,
    call: CatalogCall,
) -> Result<LocalRelation, KrishivError> {
    let result = rows(session, call).await?;
    let mut data = Vec::new();
    {
        let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut data, &result.schema())
            .map_err(|e| internal("encoding a catalog result", e))?;
        writer
            .write(&result)
            .map_err(|e| internal("encoding a catalog result", e))?;
        writer
            .finish()
            .map_err(|e| internal("encoding a catalog result", e))?;
    }
    session.local_relation(Some(&data), None).await
}
