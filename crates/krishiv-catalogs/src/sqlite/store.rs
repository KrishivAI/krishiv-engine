// SPDX-License-Identifier: Apache-2.0
//! The SQLite side of the embedded catalog: the schema, its version, the rows and their JSON columns. Nothing here
//! knows the catalog traits; every function takes a connection (or a transaction, which derefs to one) and returns
//! a [`Failure`] the catalog turns into a Spark error class.

use std::collections::BTreeMap;

use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::parse_data_type;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};

/// The on-disk format of the database (`PRAGMA user_version`). A catalog reads its own version and refuses a newer one
/// (ADR 0026: N reads N-1). Version 2 added `python_functions`; a version 1 file is upgraded when opened.
pub(super) const FORMAT_VERSION: i64 = 2;

/// Why a step failed: the database, or a rule of the catalog (already a Spark error).
#[derive(Debug)]
pub(super) enum Failure {
    /// SQLite said no (busy past the timeout, disk, corruption); the detail is logged, never returned.
    Sql(String),
    /// The step's own refusal.
    Krishiv(KrishivError),
}

impl From<rusqlite::Error> for Failure {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sql(e.to_string())
    }
}

impl From<KrishivError> for Failure {
    fn from(e: KrishivError) -> Self {
        Self::Krishiv(e)
    }
}

/// The tables of format version 2 (version 1 had no `python_functions`). `version` of a table is the catalog epoch of the change that made it what it is,
/// so a table dropped and created again, or altered, is never at the version a reader saw before.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS epoch (
    id INTEGER PRIMARY KEY CHECK (id = 0),
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO epoch (id, value) VALUES (0, 0);
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id = 0),
    case_sensitive INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS namespaces (
    name TEXT PRIMARY KEY,
    comment TEXT,
    location TEXT,
    properties TEXT NOT NULL
) WITHOUT ROWID;
INSERT OR IGNORE INTO namespaces (name, comment, location, properties) VALUES ('default', NULL, NULL, '{}');
CREATE TABLE IF NOT EXISTS tables (
    namespace TEXT NOT NULL REFERENCES namespaces (name),
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    comment TEXT,
    location TEXT,
    properties TEXT NOT NULL,
    partition_columns TEXT NOT NULL,
    columns TEXT NOT NULL,
    PRIMARY KEY (namespace, name)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS python_functions (
    tenant TEXT NOT NULL,
    name TEXT NOT NULL,
    eval_type INTEGER NOT NULL,
    deterministic INTEGER NOT NULL,
    return_type TEXT NOT NULL,
    payload BLOB NOT NULL,
    digest TEXT NOT NULL,
    python_ver TEXT NOT NULL,
    owner TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY (tenant, name)
) WITHOUT ROWID;
";

/// Creates the schema when the database is new and refuses one written by a newer format. Run in an immediate
/// transaction, so two processes opening a new file at once make it once.
///
/// Names are folded by one rule for the life of the file: `case_sensitive` is stored when the file is made, and a file
/// is refused when opened with the other value.
pub(super) fn migrate(conn: &mut Connection, case_sensitive: bool) -> Result<(), Failure> {
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version > FORMAT_VERSION {
        return Err(Failure::Sql(format!(
            "the catalog file has format version {version}; this build reads up to {FORMAT_VERSION}"
        )));
    }
    if version < FORMAT_VERSION {
        // Every statement is idempotent, so a version 1 file only gains the tables added since.
        tx.execute_batch(SCHEMA)?;
        tx.execute(
            "INSERT OR IGNORE INTO settings (id, case_sensitive) VALUES (0, ?1)",
            [case_sensitive],
        )?;
        tx.pragma_update(None, "user_version", FORMAT_VERSION)?;
    }
    let stored: bool = tx.query_row(
        "SELECT case_sensitive FROM settings WHERE id = 0",
        [],
        |r| r.get(0),
    )?;
    if stored != case_sensitive {
        return Err(Failure::Sql(format!(
            "the catalog file folds names with case_sensitive = {stored}; it was opened with {case_sensitive}"
        )));
    }
    tx.commit()?;
    Ok(())
}

/// The epoch stored in the database.
pub(super) fn read_epoch(conn: &Connection) -> Result<i64, Failure> {
    Ok(conn.query_row("SELECT value FROM epoch WHERE id = 0", [], |r| r.get(0))?)
}

/// Bumps the epoch and returns the new value; the caller commits it with the change.
pub(super) fn bump_epoch(conn: &Connection) -> Result<i64, Failure> {
    conn.execute("UPDATE epoch SET value = value + 1 WHERE id = 0", [])?;
    read_epoch(conn)
}

/// A namespace's row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NamespaceRow {
    pub(super) comment: Option<String>,
    pub(super) location: Option<String>,
    pub(super) properties: BTreeMap<String, String>,
}

pub(super) fn read_namespace(
    conn: &Connection,
    name: &str,
) -> Result<Option<NamespaceRow>, Failure> {
    let row: Option<(Option<String>, Option<String>, String)> = conn
        .query_row(
            "SELECT comment, location, properties FROM namespaces WHERE name = ?1",
            [name],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    row.map(|(comment, location, properties)| {
        Ok(NamespaceRow {
            comment,
            location,
            properties: strings(&properties)?,
        })
    })
    .transpose()
}

pub(super) fn write_namespace(
    conn: &Connection,
    name: &str,
    row: &NamespaceRow,
    create: bool,
) -> Result<(), Failure> {
    let properties = encode_strings(&row.properties)?;
    let sql = if create {
        "INSERT INTO namespaces (comment, location, properties, name) VALUES (?1, ?2, ?3, ?4)"
    } else {
        "UPDATE namespaces SET comment = ?1, location = ?2, properties = ?3 WHERE name = ?4"
    };
    conn.execute(sql, params![row.comment, row.location, properties, name])?;
    Ok(())
}

/// One column as stored: the table column, its comment and its default (SQL text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StoredColumn {
    pub(super) column: TableColumn,
    pub(super) comment: Option<String>,
    pub(super) default: Option<String>,
}

/// A table's row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TableRow {
    pub(super) version: i64,
    pub(super) comment: Option<String>,
    pub(super) location: Option<String>,
    pub(super) properties: BTreeMap<String, String>,
    pub(super) partition_columns: Vec<String>,
    pub(super) columns: Vec<StoredColumn>,
}

pub(super) fn read_table(
    conn: &Connection,
    namespace: &str,
    name: &str,
) -> Result<Option<TableRow>, Failure> {
    type Raw = (i64, Option<String>, Option<String>, String, String, String);
    let raw: Option<Raw> = conn
        .query_row(
            "SELECT version, comment, location, properties, partition_columns, columns \
             FROM tables WHERE namespace = ?1 AND name = ?2",
            [namespace, name],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            },
        )
        .optional()?;
    raw.map(
        |(version, comment, location, properties, partitions, columns)| {
            Ok(TableRow {
                version,
                comment,
                location,
                properties: strings(&properties)?,
                partition_columns: serde_json::from_str(&partitions).map_err(damaged)?,
                columns: decode_columns(&columns)?,
            })
        },
    )
    .transpose()
}

/// Inserts (`create`) or replaces a table's row.
pub(super) fn write_table(
    conn: &Connection,
    namespace: &str,
    name: &str,
    row: &TableRow,
    create: bool,
) -> Result<(), Failure> {
    let sql = if create {
        "INSERT INTO tables (version, comment, location, properties, partition_columns, columns, namespace, name) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"
    } else {
        "UPDATE tables SET version = ?1, comment = ?2, location = ?3, properties = ?4, partition_columns = ?5, \
         columns = ?6 WHERE namespace = ?7 AND name = ?8"
    };
    conn.execute(
        sql,
        params![
            row.version,
            row.comment,
            row.location,
            encode_strings(&row.properties)?,
            serde_json::to_string(&row.partition_columns).map_err(damaged)?,
            encode_columns(&row.columns)?,
            namespace,
            name,
        ],
    )?;
    Ok(())
}

pub(super) fn table_names(conn: &Connection, namespace: &str) -> Result<Vec<String>, Failure> {
    let mut stmt = conn.prepare("SELECT name FROM tables WHERE namespace = ?1 ORDER BY name")?;
    let names = stmt.query_map([namespace], |r| r.get(0))?;
    Ok(names.collect::<Result<_, _>>()?)
}

pub(super) fn namespace_names(conn: &Connection) -> Result<Vec<String>, Failure> {
    let mut stmt = conn.prepare("SELECT name FROM namespaces ORDER BY name")?;
    let names = stmt.query_map([], |r| r.get(0))?;
    Ok(names.collect::<Result<_, _>>()?)
}

pub(super) fn delete_table(
    conn: &Connection,
    namespace: &str,
    name: &str,
) -> Result<bool, Failure> {
    Ok(conn.execute(
        "DELETE FROM tables WHERE namespace = ?1 AND name = ?2",
        [namespace, name],
    )? > 0)
}

pub(super) fn delete_tables_of(conn: &Connection, namespace: &str) -> Result<(), Failure> {
    conn.execute("DELETE FROM tables WHERE namespace = ?1", [namespace])?;
    Ok(())
}

pub(super) fn delete_namespace(conn: &Connection, name: &str) -> Result<(), Failure> {
    conn.execute("DELETE FROM namespaces WHERE name = ?1", [name])?;
    Ok(())
}

/// A row that does not decode is a damaged catalog, not a user mistake.
fn damaged(e: impl std::fmt::Display) -> Failure {
    Failure::Sql(format!("a stored value does not decode: {e}"))
}

fn strings(json: &str) -> Result<BTreeMap<String, String>, Failure> {
    serde_json::from_str(json).map_err(damaged)
}

fn encode_strings(map: &BTreeMap<String, String>) -> Result<String, Failure> {
    serde_json::to_string(map).map_err(damaged)
}

fn encode_columns(columns: &[StoredColumn]) -> Result<String, Failure> {
    let list: Vec<Value> = columns
        .iter()
        .map(|c| {
            json!({
                "name": c.column.name,
                "type": c.column.data_type.catalog_string(),
                "nullable": c.column.nullable,
                "char_varchar": c.column.char_varchar,
                "comment": c.comment,
                "default": c.default,
            })
        })
        .collect();
    serde_json::to_string(&list).map_err(damaged)
}

fn decode_columns(json: &str) -> Result<Vec<StoredColumn>, Failure> {
    let list: Vec<Value> = serde_json::from_str(json).map_err(damaged)?;
    list.iter().map(decode_column).collect()
}

fn decode_column(v: &Value) -> Result<StoredColumn, Failure> {
    let text = |key: &str| v.get(key).and_then(Value::as_str).map(str::to_owned);
    let (Some(name), Some(ty), Some(nullable)) = (
        text("name"),
        text("type"),
        v.get("nullable").and_then(Value::as_bool),
    ) else {
        return Err(damaged("a column without a name, type or nullability"));
    };
    let data_type = parse_data_type(&ty).map_err(damaged)?;
    Ok(StoredColumn {
        column: TableColumn {
            name,
            data_type,
            nullable,
            char_varchar: text("char_varchar"),
        },
        comment: text("comment"),
        default: text("default"),
    })
}
