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

use crate::sql_functions::Entry;

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

/// A permanent view: its query as SQL text, parsed again at each read (so it follows the tables it reads), and the
/// namespace that was current when it was created, in which its bare table names resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ViewDef {
    pub(crate) sql: String,
    pub(crate) namespace: String,
    /// The names the query's output had when the view was created (before a column list renamed them): a
    /// read projects the stored columns from the query's output by these names, so a base table that gained
    /// columns does not change the view (Spark's schema compensation).
    pub(crate) query_columns: Vec<String>,
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
    /// The warehouse root the table was created under: `spark.sql.warehouse.dir` can change later, and the
    /// files must be deleted against the root they were created in.
    pub(crate) root: PathBuf,
    /// Set for a view: it has no provider, location or files.
    pub(crate) view: Option<ViewDef>,
}

/// A namespace's metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct NamespaceMeta {
    pub(crate) comment: Option<String>,
    pub(crate) location: Option<String>,
    pub(crate) properties: BTreeMap<String, String>,
}

/// A SQL-bodied function of a namespace (`CREATE FUNCTION` without
/// `TEMPORARY`), with the statement that defined it.
#[derive(Clone)]
pub(crate) struct StoredFunction {
    pub(crate) namespace: String,
    pub(crate) entry: Entry,
}

/// The stored functions at one version: replaced, never edited, so a query
/// reads one map whatever other statements run meanwhile.
#[derive(Clone, Default)]
pub(crate) struct StoredFunctions {
    version: u64,
    /// By namespace and lower-case name.
    entries: BTreeMap<(String, String), StoredFunction>,
}

impl std::fmt::Debug for StoredFunctions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredFunctions")
            .field("version", &self.version)
            .field("functions", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl StoredFunctions {
    /// The function `name` of `namespace`.
    pub(crate) fn get(&self, namespace: &str, name: &str) -> Option<&StoredFunction> {
        self.entries
            .get(&(namespace.to_owned(), name.to_lowercase()))
    }

    /// Every function, by namespace and name.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &StoredFunction> {
        self.entries.values()
    }

    /// How many changes the store has made to its functions.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// This map with `function` defined (replacing any of its name).
    pub(crate) fn with(&self, function: StoredFunction) -> Self {
        let mut next = self.clone();
        next.entries.insert(
            (function.namespace.clone(), function.entry.def.name.clone()),
            function,
        );
        next.version += 1;
        next
    }
}

/// Why a namespace could not be dropped.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DropNamespaceError {
    /// `default` is never dropped.
    Default,
    /// The namespace does not exist.
    Missing,
    /// It holds tables and the drop was a `RESTRICT`.
    NotEmpty,
}

/// What `CREATE TABLE` does when the table exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CreateMode {
    ErrorIfExists,
    IfNotExists,
    /// Replaces an existing entry (views only: file tables cannot be replaced).
    Replace,
}

/// The contract of a permanent catalog (see the module docs).
pub(crate) trait CatalogStore: Send + Sync {
    /// Bumped on every change.
    fn epoch(&self) -> u64;
    /// Whether the namespace exists.
    fn has_namespace(&self, namespace: &str) -> bool;
    /// Adds a namespace; `false` when it exists already (nothing changes).
    fn create_namespace(&self, namespace: &str, meta: NamespaceMeta) -> bool;
    /// Removes a namespace; with `cascade` also its tables, which are returned so that the caller can delete
    /// the managed ones' data after the metadata change.
    ///
    /// # Errors
    /// [`DropNamespaceError`] for `default`, a missing namespace, or a `RESTRICT` of a namespace with tables.
    fn drop_namespace(
        &self,
        namespace: &str,
        cascade: bool,
    ) -> Result<Vec<Arc<CatalogTable>>, DropNamespaceError>;
    /// Every namespace with its metadata, by name.
    fn namespaces(&self) -> Vec<(String, NamespaceMeta)>;
    /// The tables and views of `namespace`, by name.
    fn tables_in(&self, namespace: &str) -> Vec<Arc<CatalogTable>>;
    /// The table `namespace.name`.
    fn table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>>;
    /// Adds `table`; `false` when nothing was added (`IfNotExists` on an existing table).
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_ALREADY_EXISTS` for `ErrorIfExists`.
    fn create_table(&self, table: CatalogTable, mode: CreateMode) -> Result<bool, KrishivError>;
    /// Replaces the entry `expected` was read as with `table`; `false` when that entry was dropped or replaced
    /// since, so a concurrent `DROP` or `ALTER` is never undone by a stale copy.
    fn replace_table(&self, expected: &Arc<CatalogTable>, table: CatalogTable) -> bool;
    /// Removes the table, returning it.
    fn drop_table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>>;
    /// The stored functions as of now.
    fn functions(&self) -> Arc<StoredFunctions>;
    /// Defines `function`, replacing one of its name. `expected` is the
    /// version of [`functions`](Self::functions) the caller checked the
    /// definition against: when another statement changed the functions
    /// meanwhile nothing is defined and `false` is returned.
    fn define_function(&self, function: StoredFunction, expected: u64) -> bool;
    /// Removes the function `namespace.name`; whether it existed.
    fn drop_function(&self, namespace: &str, name: &str) -> bool;
}

/// Spark's `in-memory` catalog: metadata in memory, per session.
#[derive(Debug)]
pub(crate) struct InMemoryCatalogStore {
    state: Mutex<StoreState>,
    epoch: AtomicU64,
}

#[derive(Debug)]
struct StoreState {
    namespaces: BTreeMap<String, NamespaceMeta>,
    tables: BTreeMap<(String, String), Arc<CatalogTable>>,
    functions: Arc<StoredFunctions>,
}

impl Default for InMemoryCatalogStore {
    fn default() -> Self {
        Self {
            state: Mutex::new(StoreState {
                namespaces: BTreeMap::from([(
                    DEFAULT_NAMESPACE.to_owned(),
                    NamespaceMeta::default(),
                )]),
                tables: BTreeMap::new(),
                functions: Arc::new(StoredFunctions::default()),
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
        self.lock().namespaces.contains_key(namespace)
    }

    fn namespaces(&self) -> Vec<(String, NamespaceMeta)> {
        self.lock()
            .namespaces
            .iter()
            .map(|(n, m)| (n.clone(), m.clone()))
            .collect()
    }

    fn tables_in(&self, namespace: &str) -> Vec<Arc<CatalogTable>> {
        self.lock()
            .tables
            .iter()
            .filter(|((ns, _), _)| ns == namespace)
            .map(|(_, t)| Arc::clone(t))
            .collect()
    }

    fn create_namespace(&self, namespace: &str, meta: NamespaceMeta) -> bool {
        let mut state = self.lock();
        if state.namespaces.contains_key(namespace) {
            return false;
        }
        state.namespaces.insert(namespace.to_owned(), meta);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
    }

    fn drop_namespace(
        &self,
        namespace: &str,
        cascade: bool,
    ) -> Result<Vec<Arc<CatalogTable>>, DropNamespaceError> {
        if namespace == DEFAULT_NAMESPACE {
            return Err(DropNamespaceError::Default);
        }
        let mut state = self.lock();
        if !state.namespaces.contains_key(namespace) {
            return Err(DropNamespaceError::Missing);
        }
        let held: Vec<(String, String)> = state
            .tables
            .keys()
            .filter(|(ns, _)| ns == namespace)
            .cloned()
            .collect();
        let has_functions = state.functions.iter().any(|f| f.namespace == namespace);
        if (!held.is_empty() || has_functions) && !cascade {
            return Err(DropNamespaceError::NotEmpty);
        }
        if has_functions {
            let mut next = StoredFunctions::clone(&state.functions);
            next.entries.retain(|(ns, _), _| ns != namespace);
            next.version += 1;
            state.functions = Arc::new(next);
        }
        let dropped = held
            .into_iter()
            .filter_map(|key| state.tables.remove(&key))
            .collect();
        state.namespaces.remove(namespace);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        Ok(dropped)
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
            (true, CreateMode::Replace) | (false, _) => {}
        }
        state.tables.insert(key, Arc::new(table));
        self.epoch.fetch_add(1, Ordering::AcqRel);
        Ok(true)
    }

    fn replace_table(&self, expected: &Arc<CatalogTable>, table: CatalogTable) -> bool {
        let key = (table.namespace.clone(), table.name.clone());
        let mut state = self.lock();
        match state.tables.get(&key) {
            Some(current) if Arc::ptr_eq(current, expected) => {}
            _ => return false,
        }
        state.tables.insert(key, Arc::new(table));
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
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

    fn functions(&self) -> Arc<StoredFunctions> {
        Arc::clone(&self.lock().functions)
    }

    fn define_function(&self, function: StoredFunction, expected: u64) -> bool {
        let mut state = self.lock();
        if state.functions.version() != expected {
            return false;
        }
        state.functions = Arc::new(state.functions.with(function));
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
    }

    fn drop_function(&self, namespace: &str, name: &str) -> bool {
        let mut state = self.lock();
        if state.functions.get(namespace, name).is_none() {
            return false;
        }
        let mut next = StoredFunctions::clone(&state.functions);
        next.entries
            .remove(&(namespace.to_owned(), name.to_lowercase()));
        next.version += 1;
        state.functions = Arc::new(next);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
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
            root: PathBuf::new(),
            view: None,
        }
    }

    #[test]
    fn replace_table_refuses_a_stale_copy() {
        let s = InMemoryCatalogStore::default();
        s.create_table(table("default", "t"), CreateMode::ErrorIfExists)
            .unwrap();
        let read = s.table("default", "t").unwrap();
        let epoch = s.epoch();
        assert!(s.replace_table(&read, table("default", "t")));
        assert_eq!(s.epoch(), epoch + 1);
        // `read` is stale now.
        assert!(!s.replace_table(&read, table("default", "t")));
        // And so is a copy of a dropped table: it is not recreated.
        let current = s.table("default", "t").unwrap();
        s.drop_table("default", "t");
        assert!(!s.replace_table(&current, table("default", "t")));
        assert!(s.table("default", "t").is_none());
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
