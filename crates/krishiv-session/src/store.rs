// SPDX-License-Identifier: Apache-2.0
//! The permanent catalog: namespaces and tables, apart from the session's
//! temporary views ([`SessionCatalog`](crate::catalog::SessionCatalog)).
//!
//! The metadata of a table (its columns, provider, location and options)
//! lives in a [`CatalogStore`]; its rows are files in the table's
//! directory, so a table is read as a read of files and any worker that can
//! reach the location can read it. `CREATE TABLE` never creates an
//! in-memory table.
//!
//! P2b has one implementation, [`InMemoryCatalogStore`] (Spark's
//! `in-memory` catalog, the one the Spark 4.2 goldens were produced with).
//! Other catalogs implement the same trait and the commands do not change.
//! Every change bumps the store's epoch, which is part of what a cached
//! plan was analyzed against.
//!
//! Table data is ordered after its metadata so that metadata never points
//! at missing data: the directory is created before the table is
//! inserted, and a dropped managed table's directory is deleted after its
//! metadata is gone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use krishiv_common::error::{KrishivError, TableOrViewAlreadyExists};
use krishiv_types::DataType;
use std::sync::Arc;

/// The namespace that always exists and cannot be dropped.
pub(crate) const DEFAULT_NAMESPACE: &str = "default";

/// A column of a catalog table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StoreColumn {
    pub(crate) name: String,
    pub(crate) data_type: DataType,
    pub(crate) nullable: bool,
    /// The `DEFAULT` expression as SQL text, analysed again at each
    /// `INSERT` (Spark keeps column defaults as text).
    pub(crate) default: Option<String>,
    pub(crate) comment: Option<String>,
}

/// A table in the catalog.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CatalogTable {
    pub(crate) namespace: String,
    pub(crate) name: String,
    /// A managed table owns its directory: dropping the table deletes it.
    pub(crate) managed: bool,
    /// The file format: `parquet`, `csv`, `json` or `text`.
    pub(crate) provider: String,
    /// The directory the table's files live in.
    pub(crate) location: PathBuf,
    pub(crate) columns: Vec<StoreColumn>,
    /// `OPTIONS (...)` as written, plus the options the engine adds so that
    /// the files it writes read back as the table's columns.
    pub(crate) options: BTreeMap<String, String>,
    pub(crate) properties: BTreeMap<String, String>,
    pub(crate) comment: Option<String>,
    pub(crate) created_ms: i64,
}

/// What `CREATE TABLE` does when the table exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CreateMode {
    ErrorIfExists,
    IfNotExists,
}

/// The contract of a permanent catalog (see the module docs).
pub(crate) trait CatalogStore: Send + Sync {
    /// Bumped on every change.
    fn epoch(&self) -> u64;
    /// Whether the namespace exists.
    fn has_namespace(&self, namespace: &str) -> bool;
    /// The table `namespace.name`.
    fn table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>>;
    /// Adds `table`; `false` when nothing was added (`IfNotExists` on an existing table).
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_ALREADY_EXISTS` for `ErrorIfExists`.
    fn create_table(&self, table: CatalogTable, mode: CreateMode) -> Result<bool, KrishivError>;
    /// Removes the table, returning it.
    fn drop_table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>>;
}

/// Spark's `in-memory` catalog: metadata in memory, per session.
#[derive(Debug)]
pub(crate) struct InMemoryCatalogStore {
    state: Mutex<StoreState>,
    epoch: AtomicU64,
}

#[derive(Debug)]
struct StoreState {
    namespaces: Vec<String>,
    tables: BTreeMap<(String, String), Arc<CatalogTable>>,
}

impl Default for InMemoryCatalogStore {
    fn default() -> Self {
        Self {
            state: Mutex::new(StoreState {
                namespaces: vec![DEFAULT_NAMESPACE.to_owned()],
                tables: BTreeMap::new(),
            }),
            epoch: AtomicU64::new(0),
        }
    }
}

impl InMemoryCatalogStore {
    /// Marks a change to a table's data (an `INSERT`, a `TRUNCATE`): cached
    /// analyses are stale though the metadata is the same.
    pub(crate) fn bump(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl CatalogStore for InMemoryCatalogStore {
    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    fn has_namespace(&self, namespace: &str) -> bool {
        self.lock().namespaces.iter().any(|n| n == namespace)
    }

    fn table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>> {
        self.lock()
            .tables
            .get(&(namespace.to_owned(), name.to_owned()))
            .cloned()
    }

    fn create_table(&self, table: CatalogTable, mode: CreateMode) -> Result<bool, KrishivError> {
        let key = (table.namespace.clone(), table.name.clone());
        let mut state = self.lock();
        match (state.tables.contains_key(&key), mode) {
            (true, CreateMode::ErrorIfExists) => {
                return Err(TableOrViewAlreadyExists {
                    relation_name: format!(
                        "`spark_catalog`.`{}`.`{}`",
                        table.namespace, table.name
                    ),
                }
                .into());
            }
            (true, CreateMode::IfNotExists) => return Ok(false),
            (false, _) => {}
        }
        state.tables.insert(key, Arc::new(table));
        self.epoch.fetch_add(1, Ordering::AcqRel);
        Ok(true)
    }

    fn drop_table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>> {
        let removed = self
            .lock()
            .tables
            .remove(&(namespace.to_owned(), name.to_owned()));
        if removed.is_some() {
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
        removed
    }
}

/// Where managed tables live: `spark.sql.warehouse.dir` when set, else a
/// directory of the session's own, removed when the session ends (the
/// tables of an in-memory catalog do not outlive it, and a leftover
/// directory would make the next session's `CREATE TABLE` fail on an
/// existing location).
#[derive(Debug, Default)]
pub(crate) struct Warehouse {
    own: Mutex<Option<PathBuf>>,
}

impl Warehouse {
    /// The warehouse directory for `configured`, creating the session's own
    /// on first use.
    ///
    /// # Errors
    /// The I/O error creating the directory.
    pub(crate) fn root(&self, configured: Option<&str>) -> std::io::Result<PathBuf> {
        if let Some(dir) = configured.filter(|d| !d.is_empty()) {
            let dir = dir.strip_prefix("file:").unwrap_or(dir);
            return Ok(PathBuf::from(dir));
        }
        let mut own = self.own.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(dir) = &*own {
            return Ok(dir.clone());
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir =
            std::env::temp_dir().join(format!("krishiv-warehouse-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        *own = Some(dir.clone());
        Ok(dir)
    }
}

impl Drop for Warehouse {
    fn drop(&mut self) {
        let own = self.own.get_mut().unwrap_or_else(PoisonError::into_inner);
        if let Some(dir) = own.take() {
            // Best effort: a leftover directory is the temp directory's to clean.
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Spark's managed location of `namespace.name` under `warehouse`:
/// `<warehouse>/<table>` in `default`, else `<warehouse>/<namespace>.db/<table>`.
pub(crate) fn managed_location(warehouse: &Path, namespace: &str, name: &str) -> PathBuf {
    if namespace == DEFAULT_NAMESPACE {
        warehouse.join(name)
    } else {
        warehouse.join(format!("{namespace}.db")).join(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(ns: &str, name: &str) -> CatalogTable {
        CatalogTable {
            namespace: ns.to_owned(),
            name: name.to_owned(),
            managed: true,
            provider: "parquet".to_owned(),
            location: PathBuf::from("/w").join(name),
            columns: vec![StoreColumn {
                name: "a".to_owned(),
                data_type: DataType::Integer,
                nullable: true,
                default: None,
                comment: None,
            }],
            options: BTreeMap::new(),
            properties: BTreeMap::new(),
            comment: None,
            created_ms: 0,
        }
    }

    #[test]
    fn create_and_drop_bump_the_epoch() {
        let s = InMemoryCatalogStore::default();
        assert!(s.has_namespace("default"));
        assert!(!s.has_namespace("other"));
        assert_eq!(s.epoch(), 0);
        assert!(
            s.create_table(table("default", "t"), CreateMode::ErrorIfExists)
                .unwrap()
        );
        assert_eq!(s.epoch(), 1);
        let err = s
            .create_table(table("default", "t"), CreateMode::ErrorIfExists)
            .unwrap_err();
        assert_eq!(err.error_class().name(), "TABLE_OR_VIEW_ALREADY_EXISTS");
        assert_eq!(s.epoch(), 1);
        assert!(
            !s.create_table(table("default", "t"), CreateMode::IfNotExists)
                .unwrap()
        );
        assert_eq!(s.epoch(), 1);
        assert!(s.table("default", "t").is_some());
        assert!(s.table("default", "u").is_none());
        s.bump();
        assert_eq!(s.epoch(), 2);
        assert!(s.drop_table("default", "t").is_some());
        assert_eq!(s.epoch(), 3);
        assert!(s.drop_table("default", "t").is_none());
        assert_eq!(s.epoch(), 3);
    }

    #[test]
    fn managed_locations_follow_sparks_layout() {
        let w = Path::new("/w");
        assert_eq!(managed_location(w, "default", "t"), Path::new("/w/t"));
        assert_eq!(managed_location(w, "db", "t"), Path::new("/w/db.db/t"));
    }

    #[test]
    fn the_warehouse_is_the_configured_dir_or_a_session_dir_removed_on_drop() {
        let w = Warehouse::default();
        assert_eq!(
            w.root(Some("file:/data/wh")).unwrap(),
            Path::new("/data/wh")
        );
        let own = w.root(None).unwrap();
        assert!(own.is_dir());
        assert_eq!(w.root(Some("")).unwrap(), own);
        drop(w);
        assert!(!own.exists());
    }
}
