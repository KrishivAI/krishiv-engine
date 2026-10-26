// SPDX-License-Identifier: Apache-2.0
//! The session catalog: temporary tables and views, by name, in
//! `spark_catalog.default` (Spark's session catalog), and what one query
//! sees of it.
//!
//! A temporary table holds batches in memory ([`Session::register_batches`](crate::Session::register_batches));
//! a temporary view over files is a read by path
//! ([`DataFrame::create_or_replace_temp_view`](crate::DataFrame::create_or_replace_temp_view))
//! with the columns it had when it was created: as in Spark, files changed
//! later do not change the view's schema, so analysing a query over it reads
//! no files. Each query loads an entry once and keeps it ([`QueryCatalog`]),
//! so a view replaced while the query runs does not change what it reads.
//! A query lists the files of each read (and view) once, before it is
//! planned, and scans them as a [`FileTable`] whose footers give the
//! optimizer its statistics.
//!
//! Permanent state lives in a [`CatalogStore`] (one per session, owned by
//! [`SessionCatalog`], never shared between sessions or processes): namespaces
//! and file-backed tables created by `CREATE DATABASE` and `CREATE TABLE`.
//! A table read plans as an ordinary read by path ([`ReadSource::Files`])
//! with the stored columns as its schema, the same way a temporary view
//! over files works today: analysis reads no files, and a plan never holds
//! a handle into the catalog. The store's epoch is part of the
//! query-context fingerprint, so cached analyses do not survive DDL.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_schema::{Schema, SchemaRef};
use krishiv_common::config::Config;
use krishiv_common::error::{
    ErrorClass, InternalError, KrishivError, KrishivReservedTempViewName,
    KrishivUnsupportedFeature, RecursiveView, SchemaAlreadyExists, SchemaNotEmpty, SchemaNotFound,
    TableOrViewAlreadyExists, TableOrViewNotFound,
};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::FileReads;
use krishiv_exec::planner::{FileTable, MemoryTable, TableProvider, TableProviderResolver};
use krishiv_io::IoContext;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, CatalogResolver, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_plan::{
    Cte, Expr, ExprNode, LogicalPlan, Origin, PlanNode, ReadSource, Scan, TableVersion,
};

/// The session catalog's name, as Spark names it.
pub(crate) const CATALOG: &str = "spark_catalog";
/// The namespace temporary tables live in.
pub(crate) const NAMESPACE: &str = "default";

/// A temporary table or view.
#[derive(Debug)]
pub(crate) enum TempTable {
    /// Batches in memory.
    Memory {
        columns: Vec<TableColumn>,
        table: Arc<MemoryTable>,
    },
    /// A read by path, with the columns inferred when the view was created.
    Files {
        source: ReadSource,
        columns: Vec<TableColumn>,
    },
    /// A view over a query: the unresolved plan and the output columns it
    /// had when it was created (renamed when the view declares aliases).
    /// Queries inline the plan as a subquery, so a view replaced while a
    /// query runs does not change what it reads, as for tables.
    ///
    /// A DataFrame view also owns the hidden names its snapshot pinned
    /// ([`pin_temp_tables`]): replacing or dropping the view removes the
    /// pins no other live view still owns. SQL views keep the unresolved
    /// query, follow the tables they read, and own no pins.
    View {
        plan: Arc<LogicalPlan>,
        columns: Vec<TableColumn>,
        /// The hidden names this view pinned, owned by this view.
        pins: Vec<String>,
    },
}

/// The prefix of the hidden names snapshots pin tables under. No user
/// input spells them: [`expand_temp_views`] reports a read of one as a
/// missing table, and creating a temporary table or view with such a
/// name fails with `KRISHIV_RESERVED_TEMP_VIEW_NAME`, which names this
/// prefix. [`SessionCatalog::names`] hides them.
pub(crate) const HIDDEN_PREFIX: &str = "__krishiv_snapshot_";

/// Whether `name` is a hidden snapshot pin: it starts with
/// [`HIDDEN_PREFIX`], in any letter case.
pub(crate) fn is_hidden_name(name: &str) -> bool {
    name.get(..HIDDEN_PREFIX.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(HIDDEN_PREFIX))
}

/// The temporary tables and views of a session, by lower-case name:
/// single-part names only, resolved before the [`CatalogStore`].
#[derive(Debug, Default)]
pub(crate) struct TempViews {
    tables: RwLock<BTreeMap<String, (String, Arc<TempTable>)>>,
    epoch: AtomicU64,
    /// Hidden names handed out for snapshot pins (see [`SessionCatalog::put_view`]).
    hidden: AtomicU64,
}

impl TempViews {
    /// Adds or replaces `name`, removing the snapshot pins the old entry
    /// owned that no other live view still owns.
    pub(crate) fn put(&self, name: &str, table: TempTable) {
        let mut tables = self.tables.write().unwrap_or_else(PoisonError::into_inner);
        tables.insert(name.to_lowercase(), (name.to_owned(), Arc::new(table)));
        Self::collect_garbage(&mut tables);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// Adds or replaces `name` with a view over `plan` and `columns` that
    /// owns `pins`, inserting `new_pins` (the hidden entries the view pins)
    /// in the same critical section: no concurrent `put` or `drop_table`
    /// can collect a new pin between its insert and the view's, which would
    /// leave the view reading a missing entry.
    pub(crate) fn put_view(
        &self,
        name: &str,
        plan: Arc<LogicalPlan>,
        columns: Vec<TableColumn>,
        pins: Vec<String>,
        new_pins: Vec<(String, Arc<TempTable>)>,
    ) {
        let mut tables = self.tables.write().unwrap_or_else(PoisonError::into_inner);
        for (hidden, table) in new_pins {
            tables.insert(hidden.clone(), (hidden, table));
        }
        tables.insert(
            name.to_lowercase(),
            (
                name.to_owned(),
                Arc::new(TempTable::View {
                    plan,
                    columns,
                    pins,
                }),
            ),
        );
        Self::collect_garbage(&mut tables);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// Removes every hidden entry no live view pins anymore. The caller
    /// holds the write lock and bumps the epoch itself.
    fn collect_garbage(tables: &mut BTreeMap<String, (String, Arc<TempTable>)>) {
        let mut live = BTreeSet::new();
        for (_, table) in tables.values() {
            if let TempTable::View { pins, .. } = &**table {
                live.extend(pins.iter().cloned());
            }
        }
        tables.retain(|key, _| !is_hidden_name(key) || live.contains(key));
    }

    /// Hands out the next hidden name snapshot pins use: no SQL text
    /// spells it, so snapshot views read it however the session's names
    /// change afterwards. The caller inserts the entry with
    /// [`SessionCatalog::put_view`], which stores the pins and the owning
    /// view atomically.
    fn next_hidden_name(&self) -> String {
        format!(
            "{HIDDEN_PREFIX}{}",
            self.hidden.fetch_add(1, Ordering::AcqRel)
        )
    }

    /// Removes `name` (and the snapshot pins it owned that no other live
    /// view still owns); whether it existed. Hidden names are never
    /// removed this way: dropping one behaves as if it did not exist.
    pub(crate) fn drop_table(&self, name: &str) -> bool {
        if is_hidden_name(name) {
            return false;
        }
        let mut tables = self.tables.write().unwrap_or_else(PoisonError::into_inner);
        let removed = tables.remove(&name.to_lowercase()).is_some();
        if removed {
            Self::collect_garbage(&mut tables);
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
        removed
    }

    /// The names, as registered, in order, without snapshot pins.
    pub(crate) fn names(&self) -> Vec<String> {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .map(|(n, _)| n.clone())
            .filter(|n| !n.starts_with(HIDDEN_PREFIX))
            .collect()
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// The entry `name`: case-insensitively unless `case_sensitive`, when
    /// the name must be as registered. Unlike [`exists`](Self::exists),
    /// hidden names resolve here: snapshot views read their pins through
    /// this. User input never reaches a pin: [`expand_temp_views`]
    /// reports a read of one as a missing table before analysis loads
    /// anything, and creating or dropping one behaves as missing too.
    pub(crate) fn get(&self, name: &str, case_sensitive: bool) -> Option<Arc<TempTable>> {
        let tables = self.tables.read().unwrap_or_else(PoisonError::into_inner);
        let (registered, table) = tables.get(&name.to_lowercase())?;
        (!case_sensitive || registered == name).then(|| Arc::clone(table))
    }

    /// Whether an entry `name` exists: case-insensitively unless
    /// `case_sensitive`, when the name must be as registered. Hidden
    /// names never exist for user input, however they are spelled.
    pub(crate) fn exists(&self, name: &str, case_sensitive: bool) -> bool {
        if is_hidden_name(name) {
            return false;
        }
        let tables = self.tables.read().unwrap_or_else(PoisonError::into_inner);
        match tables.get(&name.to_lowercase()) {
            Some((registered, _)) => !case_sensitive || registered == name,
            None => false,
        }
    }

    /// The number of catalog entries, snapshot pins included: tests only.
    /// [`names`](Self::names) stays the user-visible count.
    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Whether a table is managed (its directory is the warehouse's, and
/// dropping the table deletes it) or external (its `LOCATION` is the
/// user's, and dropping the table keeps it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CatalogTableKind {
    /// A managed table.
    Managed,
    /// An external table.
    External,
}

/// A permanent, file-backed table: its files live at `location`, read in
/// `provider` with `options` under the stored `columns` (plans hold this
/// read, never a handle into the catalog). Comments, defaults, properties
/// and the rest are metadata only.
#[derive(Debug, Clone)]
pub(crate) struct CatalogTable {
    /// Its identity, as the catalog spells it: `spark_catalog`,
    /// one namespace part, the name.
    pub(crate) ident: TableIdent,
    /// Managed or external.
    pub(crate) kind: CatalogTableKind,
    /// The file format, lower-case (`parquet`, `csv`, `json`, `text`).
    pub(crate) provider: String,
    /// The table directory, absolute.
    pub(crate) location: String,
    /// The columns, as declared or as the CTAS query returned them.
    pub(crate) columns: Vec<TableColumn>,
    /// Each column's comment, if written.
    pub(crate) col_comments: Vec<Option<String>>,
    /// Each column's default, as SQL text, if written.
    pub(crate) col_defaults: Vec<Option<String>>,
    /// The `OPTIONS`, typed as secrets where Spark redacts them.
    pub(crate) options: BTreeMap<String, OptionValue>,
    /// The `TBLPROPERTIES`.
    pub(crate) properties: BTreeMap<String, String>,
    /// The `COMMENT`, if written.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by DESCRIBE and SHOW, which the engine does not run yet"
        )
    )]
    pub(crate) comment: Option<String>,
    /// When the table was created, as seconds (and nanos) since the epoch.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by DESCRIBE and SHOW, which the engine does not run yet"
        )
    )]
    pub(crate) created_at: String,
    /// Who created the table.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by DESCRIBE and SHOW, which the engine does not run yet"
        )
    )]
    pub(crate) owner: String,
}

/// How [`CatalogStore::create_table`] treats an existing table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CreateMode {
    /// Fail with `TABLE_OR_VIEW_ALREADY_EXISTS`.
    ErrorIfExists,
    /// Keep the existing table.
    IfNotExists,
    /// Replace the existing table.
    Replace,
}

/// A namespace: its comment, its directory and its properties.
#[derive(Debug, Clone)]
pub(crate) struct NamespaceMeta {
    /// The name, as written.
    pub(crate) name: String,
    /// The `COMMENT`, if written.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by DESCRIBE, which the engine does not run yet"
        )
    )]
    pub(crate) comment: Option<String>,
    /// The namespace directory, absolute.
    pub(crate) location: String,
    /// The `DBPROPERTIES`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by DESCRIBE, which the engine does not run yet"
        )
    )]
    pub(crate) properties: BTreeMap<String, String>,
}

/// The permanent state of a session's catalog: namespaces and file-backed
/// tables. One store per session, owned by [`SessionCatalog`]: no shared
/// process state, no static. Every change bumps [`epoch`](Self::epoch),
/// which is part of the query-context fingerprint, so cached analyses do
/// not survive DDL.
pub(crate) trait CatalogStore {
    /// Bumped on every change.
    fn epoch(&self) -> u64;
    /// The namespaces, as written, in order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by SHOW NAMESPACES, which the engine does not run yet"
        )
    )]
    fn namespaces(&self) -> Vec<String>;
    /// Whether namespace `name` exists: ignoring case unless
    /// `case_sensitive`, when the name must be as created.
    fn namespace_exists(&self, name: &str, case_sensitive: bool) -> bool;
    /// The directory of namespace `name`, if it exists.
    fn namespace_location(&self, name: &str, case_sensitive: bool) -> Option<String>;
    /// Creates namespace `ns`; `true` when it was created. An existing
    /// namespace fails with `SCHEMA_ALREADY_EXISTS` unless `if_not_exists`.
    fn create_namespace(
        &mut self,
        ns: NamespaceMeta,
        if_not_exists: bool,
        case_sensitive: bool,
    ) -> Result<bool, KrishivError>;
    /// Drops namespace `name` with `behavior`: without `CASCADE` a
    /// non-empty namespace fails with `SCHEMA_NOT_EMPTY`; with it the
    /// namespace and its tables' metadata go. Returns the dropped tables
    /// (the caller deletes managed data afterwards). A missing namespace
    /// fails with `SCHEMA_NOT_FOUND` unless `if_exists`.
    fn drop_namespace(
        &mut self,
        name: &str,
        if_exists: bool,
        cascade: bool,
        case_sensitive: bool,
    ) -> Result<Vec<CatalogTable>, KrishivError>;
    /// The tables of namespace `name`, as written, in order, or
    /// `SCHEMA_NOT_FOUND`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by SHOW TABLES, which the engine does not run yet"
        )
    )]
    fn list_tables(&self, name: &str, case_sensitive: bool) -> Result<Vec<String>, KrishivError>;
    /// The table `name` in namespace `ns`: ignoring case unless
    /// `case_sensitive`, when both names must be as created.
    fn table(&self, ns: &str, name: &str, case_sensitive: bool) -> Option<CatalogTable>;
    /// Inserts `table`; `true` when it was inserted or replaced. An
    /// existing table fails with `TABLE_OR_VIEW_ALREADY_EXISTS` under
    /// [`CreateMode::ErrorIfExists`], is kept under
    /// [`CreateMode::IfNotExists`], and is replaced under
    /// [`CreateMode::Replace`]. The namespace must exist
    /// (`SCHEMA_NOT_FOUND` otherwise).
    fn create_table(
        &mut self,
        table: CatalogTable,
        mode: CreateMode,
        case_sensitive: bool,
    ) -> Result<bool, KrishivError>;
    /// Runs `update` on the table `name` in namespace `ns` (bumping the
    /// epoch); `false` when there is no such table.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "run by ALTER TABLE, which the engine does not run yet"
        )
    )]
    fn alter_table(
        &mut self,
        ns: &str,
        name: &str,
        case_sensitive: bool,
        update: impl FnOnce(&mut CatalogTable),
    ) -> bool;
    /// Removes the table `name` in namespace `ns` (and bumps the epoch);
    /// the removed table, if any.
    fn drop_table(&mut self, ns: &str, name: &str, case_sensitive: bool) -> Option<CatalogTable>;
}

/// The [`CatalogStore`]: namespaces and tables in memory, keyed
/// lower-case like Spark's session catalog.
#[derive(Debug, Default)]
pub(crate) struct InMemoryCatalogStore {
    /// Bumped on every change.
    epoch: u64,
    /// The namespaces, by lower-case name.
    namespaces: BTreeMap<String, NamespaceMeta>,
    /// The tables, by (lower-case namespace, lower-case name).
    tables: BTreeMap<(String, String), CatalogTable>,
}

impl InMemoryCatalogStore {
    /// An empty store with the `default` namespace.
    fn with_default() -> Self {
        let mut store = Self::default();
        store.namespaces.insert(
            NAMESPACE.to_owned(),
            NamespaceMeta {
                name: NAMESPACE.to_owned(),
                comment: None,
                location: String::new(),
                properties: BTreeMap::new(),
            },
        );
        store
    }

    /// Bumps the epoch.
    fn bump(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// The entry of namespace `name`: ignoring case unless
    /// `case_sensitive`, when the name must be as created.
    fn namespace_entry(&self, name: &str, case_sensitive: bool) -> Option<&NamespaceMeta> {
        let entry = self.namespaces.get(&name.to_lowercase())?;
        (!case_sensitive || entry.name == name).then_some(entry)
    }

    /// The key of table `name` in namespace `ns`.
    fn table_key(ns: &str, name: &str) -> (String, String) {
        (ns.to_lowercase(), name.to_lowercase())
    }

    /// Whether `stored` (as created) matches `written` (as read).
    fn matches(stored: &str, written: &str, case_sensitive: bool) -> bool {
        stored == written || !case_sensitive && stored.eq_ignore_ascii_case(written)
    }
}

impl CatalogStore for InMemoryCatalogStore {
    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn namespaces(&self) -> Vec<String> {
        self.namespaces.values().map(|n| n.name.clone()).collect()
    }

    fn namespace_exists(&self, name: &str, case_sensitive: bool) -> bool {
        self.namespace_entry(name, case_sensitive).is_some()
    }

    fn namespace_location(&self, name: &str, case_sensitive: bool) -> Option<String> {
        self.namespace_entry(name, case_sensitive)
            .map(|n| n.location.clone())
    }

    fn create_namespace(
        &mut self,
        ns: NamespaceMeta,
        if_not_exists: bool,
        case_sensitive: bool,
    ) -> Result<bool, KrishivError> {
        if self.namespace_entry(&ns.name, case_sensitive).is_some() {
            if if_not_exists {
                return Ok(false);
            }
            return Err(SchemaAlreadyExists {
                schema_name: quote(&ns.name),
            }
            .into());
        }
        self.namespaces.insert(ns.name.to_lowercase(), ns);
        self.bump();
        Ok(true)
    }

    fn drop_namespace(
        &mut self,
        name: &str,
        if_exists: bool,
        cascade: bool,
        case_sensitive: bool,
    ) -> Result<Vec<CatalogTable>, KrishivError> {
        if name.eq_ignore_ascii_case(NAMESPACE) {
            // The `default` namespace always exists: it cannot be dropped.
            return Err(unsupported("dropping the default namespace"));
        }
        let Some(entry) = self.namespace_entry(name, case_sensitive) else {
            if if_exists {
                return Ok(Vec::new());
            }
            return Err(SchemaNotFound {
                schema_name: quote(name),
            }
            .into());
        };
        let key = entry.name.to_lowercase();
        let tables: Vec<(String, String)> = self
            .tables
            .keys()
            .filter(|(ns, _)| *ns == key)
            .cloned()
            .collect();
        if !cascade && !tables.is_empty() {
            return Err(SchemaNotEmpty {
                schema_name: quote(name),
            }
            .into());
        }
        let mut dropped = Vec::with_capacity(tables.len());
        for k in tables {
            if let Some(t) = self.tables.remove(&k) {
                dropped.push(t);
            }
        }
        self.namespaces.remove(&key);
        self.bump();
        Ok(dropped)
    }

    fn list_tables(&self, name: &str, case_sensitive: bool) -> Result<Vec<String>, KrishivError> {
        let Some(entry) = self.namespace_entry(name, case_sensitive) else {
            return Err(SchemaNotFound {
                schema_name: quote(name),
            }
            .into());
        };
        let key = entry.name.to_lowercase();
        Ok(self
            .tables
            .iter()
            .filter(|((ns, _), _)| *ns == key)
            .map(|(_, t)| t.ident.name.clone())
            .collect())
    }

    fn table(&self, ns: &str, name: &str, case_sensitive: bool) -> Option<CatalogTable> {
        let found = self.tables.get(&Self::table_key(ns, name))?;
        let ident_ns = found
            .ident
            .namespace
            .first()
            .map(String::as_str)
            .unwrap_or("");
        (Self::matches(ident_ns, ns, case_sensitive)
            && Self::matches(&found.ident.name, name, case_sensitive))
        .then(|| found.clone())
    }

    fn create_table(
        &mut self,
        table: CatalogTable,
        mode: CreateMode,
        case_sensitive: bool,
    ) -> Result<bool, KrishivError> {
        let ns = table
            .ident
            .namespace
            .first()
            .map(String::as_str)
            .unwrap_or("");
        if self.namespace_entry(ns, case_sensitive).is_none() {
            return Err(SchemaNotFound {
                schema_name: quote(ns),
            }
            .into());
        }
        let key = Self::table_key(ns, &table.ident.name);
        if let Some(existing) = self.tables.get(&key)
            && Self::matches(
                existing
                    .ident
                    .namespace
                    .first()
                    .map(String::as_str)
                    .unwrap_or(""),
                ns,
                case_sensitive,
            )
            && Self::matches(&existing.ident.name, &table.ident.name, case_sensitive)
        {
            match mode {
                CreateMode::ErrorIfExists => {
                    return Err(TableOrViewAlreadyExists {
                        relation_name: quote(&table.ident.name),
                    }
                    .into());
                }
                CreateMode::IfNotExists => return Ok(false),
                CreateMode::Replace => {}
            }
        }
        self.tables.insert(key, table);
        self.bump();
        Ok(true)
    }

    fn alter_table(
        &mut self,
        ns: &str,
        name: &str,
        case_sensitive: bool,
        update: impl FnOnce(&mut CatalogTable),
    ) -> bool {
        let key = Self::table_key(ns, name);
        let Some(found) = self.tables.get_mut(&key) else {
            return false;
        };
        let ident_ns = found
            .ident
            .namespace
            .first()
            .map(String::as_str)
            .unwrap_or("");
        if !Self::matches(ident_ns, ns, case_sensitive)
            || !Self::matches(&found.ident.name.clone(), name, case_sensitive)
        {
            return false;
        }
        update(found);
        self.bump();
        true
    }

    fn drop_table(&mut self, ns: &str, name: &str, case_sensitive: bool) -> Option<CatalogTable> {
        let key = Self::table_key(ns, name);
        let found = self.tables.get(&key)?;
        let ident_ns = found
            .ident
            .namespace
            .first()
            .map(String::as_str)
            .unwrap_or("");
        if !Self::matches(ident_ns, ns, case_sensitive)
            || !Self::matches(&found.ident.name.clone(), name, case_sensitive)
        {
            return None;
        }
        let removed = self.tables.remove(&key);
        if removed.is_some() {
            self.bump();
        }
        removed
    }
}

/// The session's warehouse directory: created under the system temporary
/// directory when the session is built (unless
/// `spark.sql.warehouse.dir` names one), and deleted when the session is
/// dropped.
#[derive(Debug)]
pub(crate) struct WarehouseDir {
    /// The directory; created by [`Self::new`] unless creation failed, in
    /// which case table creation fails with the I/O error when it creates
    /// the table's directory.
    dir: PathBuf,
    /// Whether this session created `dir` (and deletes it on drop).
    owned: bool,
}

impl WarehouseDir {
    /// A fresh directory for one session.
    fn new() -> Self {
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let n = COUNT.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!(
            "krishiv-warehouse-{}-{n}-{nanos}",
            std::process::id()
        ));
        let owned = std::fs::create_dir_all(&dir).is_ok();
        Self { dir, owned }
    }

    /// The directory.
    fn root(&self) -> &PathBuf {
        &self.dir
    }
}

impl Drop for WarehouseDir {
    fn drop(&mut self) {
        if self.owned {
            drop(std::fs::remove_dir_all(&self.dir));
        }
    }
}

/// The catalog of a session: the [`TempViews`], the [`CatalogStore`], the
/// current namespace (`USE`), and the warehouse directory. Each session
/// owns one store: no shared process state, and plans hold reads, never
/// handles into it.
#[derive(Debug)]
pub(crate) struct SessionCatalog {
    /// The temporary tables and views.
    pub(crate) temp: TempViews,
    /// The namespaces and file-backed tables.
    store: RwLock<InMemoryCatalogStore>,
    /// The current namespace, as written.
    current: RwLock<String>,
    /// The warehouse directory.
    warehouse: WarehouseDir,
}

impl Default for SessionCatalog {
    fn default() -> Self {
        Self {
            temp: TempViews::default(),
            store: RwLock::new(InMemoryCatalogStore::with_default()),
            current: RwLock::new(NAMESPACE.to_owned()),
            warehouse: WarehouseDir::new(),
        }
    }
}

impl SessionCatalog {
    /// The temporary tables and views.
    pub(crate) fn temp_views(&self) -> &TempViews {
        &self.temp
    }

    /// The current namespace, as written.
    pub(crate) fn current_namespace(&self) -> String {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Sets the current namespace (`USE` checks it exists first), bumping
    /// the epoch so cached analyses (which resolved under the old
    /// namespace) are not reused.
    pub(crate) fn set_current_namespace(&self, ns: &str) {
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = ns.to_owned();
        self.store
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .bump();
    }

    /// The warehouse root: `spark.sql.warehouse.dir` when it names one
    /// (relative to `cwd`), else the session's temporary directory.
    pub(crate) fn warehouse_root(&self, config: &Config, cwd: &str) -> PathBuf {
        let configured =
            crate::conf::get(config, "spark.sql.warehouse.dir").filter(|d| !d.is_empty());
        match configured {
            Some(dir) => {
                let path = PathBuf::from(&dir);
                if path.is_absolute() {
                    path
                } else {
                    PathBuf::from(cwd).join(path)
                }
            }
            None => self.warehouse.root().clone(),
        }
    }

    /// The session's temporary warehouse directory (tests only).
    #[cfg(test)]
    pub(crate) fn warehouse_dir(&self) -> PathBuf {
        self.warehouse.root().clone()
    }

    /// Every temporary or store change bumps the epoch (see
    /// [`TempViews::put`] and [`CatalogStore::epoch`]).
    pub(crate) fn epoch(&self) -> u64 {
        self.temp.epoch().wrapping_add(
            self.store
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .epoch(),
        )
    }

    /// Adds or replaces `name`, removing the snapshot pins the old entry
    /// owned that no other live view still owns.
    pub(crate) fn put(&self, name: &str, table: TempTable) {
        self.temp.put(name, table);
    }

    /// Adds or replaces `name` with a view over `plan` and `columns` that
    /// owns `pins`, inserting `new_pins` in the same critical section.
    pub(crate) fn put_view(
        &self,
        name: &str,
        plan: Arc<LogicalPlan>,
        columns: Vec<TableColumn>,
        pins: Vec<String>,
        new_pins: Vec<(String, Arc<TempTable>)>,
    ) {
        self.temp.put_view(name, plan, columns, pins, new_pins);
    }

    /// Removes `name` (and the snapshot pins it owned that no other live
    /// view still owns); whether it existed.
    pub(crate) fn drop_table(&self, name: &str) -> bool {
        self.temp.drop_table(name)
    }

    /// The names, as registered, in order, without snapshot pins.
    pub(crate) fn names(&self) -> Vec<String> {
        self.temp.names()
    }

    /// The entry `name`: case-insensitively unless `case_sensitive`.
    pub(crate) fn get(&self, name: &str, case_sensitive: bool) -> Option<Arc<TempTable>> {
        self.temp.get(name, case_sensitive)
    }

    /// Whether an entry `name` exists: case-insensitively unless
    /// `case_sensitive`.
    pub(crate) fn exists(&self, name: &str, case_sensitive: bool) -> bool {
        self.temp.exists(name, case_sensitive)
    }

    /// The number of temporary entries, snapshot pins included: tests only.
    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.temp.entry_count()
    }

    /// Whether namespace `name` exists.
    pub(crate) fn namespace_exists(&self, name: &str, case_sensitive: bool) -> bool {
        self.store
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .namespace_exists(name, case_sensitive)
    }

    /// The directory of namespace `name`, if it exists.
    pub(crate) fn namespace_location(&self, name: &str, case_sensitive: bool) -> Option<String> {
        self.store
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .namespace_location(name, case_sensitive)
    }

    /// The namespaces, as written, in order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by SHOW NAMESPACES, which the engine does not run yet"
        )
    )]
    pub(crate) fn namespaces(&self) -> Vec<String> {
        self.store
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .namespaces()
    }

    /// Creates namespace `ns`; `true` when it was created.
    pub(crate) fn create_namespace(
        &self,
        ns: NamespaceMeta,
        if_not_exists: bool,
        case_sensitive: bool,
    ) -> Result<bool, KrishivError> {
        self.store
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .create_namespace(ns, if_not_exists, case_sensitive)
    }

    /// Drops namespace `name` (see [`CatalogStore::drop_namespace`]).
    pub(crate) fn drop_namespace(
        &self,
        name: &str,
        if_exists: bool,
        cascade: bool,
        case_sensitive: bool,
    ) -> Result<Vec<CatalogTable>, KrishivError> {
        self.store
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .drop_namespace(name, if_exists, cascade, case_sensitive)
    }

    /// The tables of namespace `name`, as written, in order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by SHOW TABLES, which the engine does not run yet"
        )
    )]
    pub(crate) fn list_tables(
        &self,
        name: &str,
        case_sensitive: bool,
    ) -> Result<Vec<String>, KrishivError> {
        self.store
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .list_tables(name, case_sensitive)
    }

    /// The table `name` in namespace `ns`.
    pub(crate) fn get_table(
        &self,
        ns: &str,
        name: &str,
        case_sensitive: bool,
    ) -> Option<CatalogTable> {
        self.store
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .table(ns, name, case_sensitive)
    }

    /// Inserts `table` (see [`CatalogStore::create_table`]).
    pub(crate) fn create_table(
        &self,
        table: CatalogTable,
        mode: CreateMode,
        case_sensitive: bool,
    ) -> Result<bool, KrishivError> {
        self.store
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .create_table(table, mode, case_sensitive)
    }

    /// Runs `update` on the table `name` in namespace `ns`; `false` when
    /// there is no such table.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "run by ALTER TABLE, which the engine does not run yet"
        )
    )]
    pub(crate) fn alter_store_table(
        &self,
        ns: &str,
        name: &str,
        case_sensitive: bool,
        update: impl FnOnce(&mut CatalogTable),
    ) -> bool {
        self.store
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .alter_table(ns, name, case_sensitive, update)
    }

    /// Removes the table `name` in namespace `ns`; the removed table, if
    /// any.
    pub(crate) fn drop_store_table(
        &self,
        ns: &str,
        name: &str,
        case_sensitive: bool,
    ) -> Option<CatalogTable> {
        self.store
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .drop_table(ns, name, case_sensitive)
    }
}

/// What a query reads files for: a temporary view (by lower-case name), a
/// permanent table (by lower-case namespace and name), or a read by
/// path.
#[derive(Debug, Clone, PartialEq)]
enum FilesKey {
    View(String),
    Table(String, String),
    Read(ReadSource),
}

/// How a query opens the file tables of its scans.
#[derive(Debug, Clone)]
pub(crate) struct FileScans {
    /// The I/O the scans read through.
    pub(crate) io: IoContext,
    /// The query's configuration (split sizes, ...).
    pub(crate) config: Arc<Config>,
    /// The executor's drivers, for split planning.
    pub(crate) slots: u64,
    /// The executor's I/O runtime, which opens the file tables.
    pub(crate) runtime: tokio::runtime::Handle,
}

/// What a query loaded: a temporary entry, or a permanent table. Scans
/// find their entry by its canonical parts (see
/// [`LoadedEntry::parts`]).
#[derive(Debug, Clone)]
enum LoadedEntry {
    /// A temporary table or view, with the identity the analyzer bound.
    Temp {
        /// The identity the analyzer bound.
        ident: TableIdent,
        /// The entry.
        table: Arc<TempTable>,
    },
    /// A permanent, file-backed table (boxed: the temporary entry is far
    /// smaller).
    Table(Box<CatalogTable>),
}

impl LoadedEntry {
    /// The canonical parts of the entry: what the analyzer names its
    /// scans, so a scan always finds the entry that produced it, even
    /// when a temporary table and a permanent table share every part.
    fn parts(&self) -> Vec<String> {
        match self {
            Self::Temp { ident, .. } => ident.parts(),
            Self::Table(t) => t.ident.parts(),
        }
    }
}

/// What a query loaded, by lower-case name: a temporary entry by name, a
/// permanent table by namespace and name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LoadedKey {
    /// A temporary entry.
    Temp(String),
    /// A permanent table.
    Table(String, String),
}

/// The session catalog as one query sees it: the catalog resolver and the
/// table providers of its scans.
pub(crate) struct QueryCatalog {
    session: Arc<SessionCatalog>,
    files: Arc<FileReads>,
    scans: FileScans,
    /// Entries loaded by this query.
    loaded: Mutex<HashMap<LoadedKey, LoadedEntry>>,
    /// The file tables of the query's scans, opened before planning.
    opened: Mutex<Vec<(FilesKey, Arc<FileTable>)>>,
    epoch: u64,
    /// The current namespace the query reads in.
    current: Vec<String>,
}

impl fmt::Debug for QueryCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryCatalog")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl QueryCatalog {
    pub(crate) fn new(
        session: Arc<SessionCatalog>,
        files: Arc<FileReads>,
        scans: FileScans,
        current: Vec<String>,
    ) -> Self {
        let epoch = session.epoch();
        Self {
            session,
            files,
            scans,
            loaded: Mutex::new(HashMap::new()),
            opened: Mutex::new(Vec::new()),
            epoch,
            current,
        }
    }

    /// The loaded entry a scan of `name` reads: by canonical parts, as
    /// the analyzer names the scans it produces. A single-part scan is a
    /// re-bound snapshot pin (see [`pin_temp_tables`]): pins are the only
    /// single-part scans, matched by name.
    fn find_loaded(&self, name: &[String]) -> Option<LoadedEntry> {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .find(|e| match (name, e) {
                ([one], LoadedEntry::Temp { ident, .. }) => *one == ident.name,
                _ => e.parts() == name,
            })
            .cloned()
    }

    /// Lists the files of every read by path and file view that the
    /// analyzed `plan` scans (subqueries included), and opens them as file
    /// tables for the planner. A read the analyzer listed keeps that
    /// listing; a view lists its files now.
    ///
    /// # Errors
    /// Listing errors (`PATH_NOT_FOUND` once a view's files are gone, ...)
    /// and errors reading the files' footers.
    pub(crate) async fn open_files(&self, plan: &LogicalPlan) -> Result<(), KrishivError> {
        for (key, source, table) in self.file_scans(plan) {
            if self.opened(&key).is_some() {
                continue;
            }
            let ReadSource::Files {
                format,
                paths,
                options,
            } = &source
            else {
                continue;
            };
            match self.files.listing(format, paths, options).await {
                Ok(listing) => {
                    let opened = open_table(
                        Arc::clone(&listing.format),
                        self.scans.clone(),
                        table,
                        listing.files.to_vec(),
                        listing.options.clone(),
                    )
                    .await?;
                    self.opened
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push((key, Arc::new(opened)));
                }
                // An empty table has no files to infer a schema from: its
                // stored columns are the schema. By-path reads and
                // temporary file views keep the inference error.
                Err(e)
                    if matches!(key, FilesKey::Table(..))
                        && e.error_class() == ErrorClass::UnableToInferSchema =>
                {
                    let (format, options) = self.files.prepared(format, options)?;
                    let opened =
                        open_table(format, self.scans.clone(), table, Vec::new(), options).await?;
                    self.opened
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push((key, Arc::new(opened)));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// The scans of files in `plan`: what each reads, and the columns the
    /// analyzer bound (a view's, or a table's, for those), which the files
    /// map to by name.
    fn file_scans(&self, plan: &LogicalPlan) -> Vec<(FilesKey, ReadSource, SchemaRef)> {
        let mut scans = Vec::new();
        collect_scans(plan, &mut scans);
        scans
            .into_iter()
            .filter_map(|scan| {
                let (key, source) = match &scan.source {
                    ReadSource::Table { name, .. } => match self.find_loaded(name)? {
                        LoadedEntry::Temp { table, ident: _ } => {
                            let TempTable::Files { source, .. } = &*table else {
                                return None;
                            };
                            let last = name.last().map_or("", String::as_str);
                            (FilesKey::View(last.to_lowercase()), source.clone())
                        }
                        LoadedEntry::Table(t) => {
                            let ns = t
                                .ident
                                .namespace
                                .first()
                                .map_or("", String::as_str)
                                .to_lowercase();
                            (
                                FilesKey::Table(ns, t.ident.name.to_lowercase()),
                                files_source(&t),
                            )
                        }
                    },
                    source @ ReadSource::Files { .. } => {
                        (FilesKey::Read(source.clone()), source.clone())
                    }
                    _ => return None,
                };
                let table = Arc::new(Schema::new(
                    scan.output
                        .iter()
                        .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
                        .collect::<Vec<_>>(),
                ));
                Some((key, source, table))
            })
            .collect()
    }

    /// Whether this query loaded a file view or a permanent table: their
    /// files are listed again by the next query, so an analysis that did
    /// must not be reused.
    pub(crate) fn loaded_file_views(&self) -> bool {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .any(|e| match e {
                LoadedEntry::Temp { table, .. } => matches!(&**table, TempTable::Files { .. }),
                LoadedEntry::Table(_) => true,
            })
    }

    fn opened(&self, key: &FilesKey) -> Option<Arc<FileTable>> {
        self.opened
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, t)| Arc::clone(t))
    }

    /// The file table of `key`, opened by [`Self::open_files`].
    fn files_provider(&self, key: &FilesKey) -> Result<Arc<dyn TableProvider>, KrishivError> {
        match self.opened(key) {
            Some(t) => Ok(t as Arc<dyn TableProvider>),
            None => Err(InternalError {
                message: "a scan of files that was not opened before planning".to_owned(),
            }
            .into()),
        }
    }
}

/// [`FileTable::open`] on the executor's I/O runtime: footer reads are
/// object-store requests, which belong there and not on the compute
/// runtime. Dropping the returned future aborts the open.
async fn open_table(
    format: Arc<dyn krishiv_io::FileFormat>,
    scans: FileScans,
    table: SchemaRef,
    files: Vec<krishiv_io::FileRef>,
    options: krishiv_io::FormatOptions,
) -> Result<FileTable, KrishivError> {
    let runtime = scans.runtime.clone();
    run_on(&runtime, async move {
        FileTable::open(
            format,
            scans.io,
            table,
            files,
            options,
            &scans.config,
            scans.slots,
        )
        .await
    })
    .await
}

/// Aborts its task when dropped.
struct AbortGuard<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortGuard<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Runs `work` as a task of `runtime` and waits for it; dropping the
/// returned future aborts the task.
async fn run_on<T: Send + 'static>(
    runtime: &tokio::runtime::Handle,
    work: impl Future<Output = Result<T, KrishivError>> + Send + 'static,
) -> Result<T, KrishivError> {
    let mut guard = AbortGuard(runtime.spawn(work));
    match (&mut guard.0).await {
        Ok(result) => result,
        Err(e) => Err(KrishivError::from(InternalError {
            message: format!("reading file footers ended without a result: {e}"),
        })),
    }
}

/// The scans of `plan` and of the subqueries in its expressions.
fn collect_scans<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a Scan>) {
    fn in_expr<'a>(e: &'a Arc<krishiv_plan::Expr>, out: &mut Vec<&'a Scan>) {
        if let ExprNode::Subquery { plan, .. } = &e.node {
            collect_scans(plan, out);
        }
        e.for_each_child(&mut |c| in_expr(c, out));
    }
    if let PlanNode::Scan(scan) = &plan.node {
        out.push(scan);
    }
    plan.for_each_expr(&mut |e| in_expr(e, out));
    plan.for_each_child(&mut |c| collect_scans(c, out));
}

/// A permanent table as a read by path: its provider, its location and
/// its options, with the stored columns as its schema. Analysis reads no
/// files for it, the same way a temporary view over files works.
fn files_source(t: &CatalogTable) -> ReadSource {
    ReadSource::Files {
        format: t.provider.clone(),
        paths: vec![OptionValue::from_user("path", &t.location)],
        options: t.options.clone(),
    }
}

/// The temporary table `name`: the one this query already loaded, so every
/// spelling of a name binds one table even if another thread replaces it
/// meanwhile, else the session's. Returns the identity the analyzer binds
/// (in [`NAMESPACE`], lower-case, as before) and the entry.
fn resolve(
    loaded: &Mutex<HashMap<LoadedKey, LoadedEntry>>,
    session: &SessionCatalog,
    name: &str,
    form: NameForm,
) -> Option<(TableIdent, Arc<TempTable>)> {
    let key = LoadedKey::Temp(name.to_lowercase());
    let loaded = loaded.lock().unwrap_or_else(PoisonError::into_inner);
    let first = loaded.get(&key);
    // A temporary entry under another spelling does not shadow: with case
    // sensitivity the loaded entry only counts when the session still has
    // it under this spelling.
    match first {
        Some(LoadedEntry::Temp { table, .. })
            if !form.case_sensitive || session.get(name, true).is_some() =>
        {
            Some(Arc::clone(table))
        }
        _ => session.get(name, form.case_sensitive),
    }
    .map(|table| {
        (
            TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![NAMESPACE.to_owned()],
                name: name.to_lowercase(),
            },
            table,
        )
    })
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

impl Catalog for QueryCatalog {
    fn name(&self) -> &str {
        CATALOG
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn single_part_namespace(&self) -> bool {
        true
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            // Temporary entries resolve for unqualified names only: Spark
            // finds `v` but not `default.v` or `spark_catalog.default.v`.
            if !form.qualified
                && let Some((ident, table)) = resolve(&self.loaded, &self.session, &id.name, form)
            {
                let (columns, kind) = match &*table {
                    TempTable::Memory { columns, .. } | TempTable::Files { columns, .. } => {
                        (columns.clone(), TableKind::Table)
                    }
                    TempTable::View { columns, .. } => (columns.clone(), TableKind::View),
                };
                self.loaded
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(
                        LoadedKey::Temp(id.name.to_lowercase()),
                        LoadedEntry::Temp {
                            ident: ident.clone(),
                            table,
                        },
                    );
                return Ok(Some(TableMeta {
                    ident,
                    columns,
                    version: None,
                    kind,
                }));
            }
            // Anything else is a permanent table in its namespace: a
            // single-part name reads the current namespace, after the
            // temporary entries above. (A namespace of more than one part
            // never reaches the catalog: the analyzer rejects it first.)
            if id.namespace.len() != 1 {
                return Ok(None);
            }
            let ns = id.namespace[0].clone();
            let Some(found) = self.session.get_table(&ns, &id.name, form.case_sensitive) else {
                return Ok(None);
            };
            let meta = TableMeta {
                ident: found.ident.clone(),
                columns: found.columns.clone(),
                version: None,
                kind: TableKind::Table,
            };
            self.loaded
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(
                    LoadedKey::Table(ns.to_lowercase(), id.name.to_lowercase()),
                    LoadedEntry::Table(Box::new(found)),
                );
            Ok(Some(meta))
        })
    }
}

/// The resolver of one query: the session catalog only.
#[derive(Debug)]
pub(crate) struct QueryCatalogs(pub(crate) Arc<QueryCatalog>);

impl CatalogResolver for QueryCatalogs {
    fn current_catalog(&self) -> &str {
        CATALOG
    }

    fn current_namespace(&self) -> &[String] {
        &self.0.current
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>> {
        name.eq_ignore_ascii_case(CATALOG)
            .then(|| Arc::clone(&self.0) as Arc<dyn Catalog>)
    }

    fn epoch(&self) -> u64 {
        0
    }
}

impl TableProviderResolver for QueryCatalog {
    fn provider(&self, scan: &Scan) -> Result<Arc<dyn TableProvider>, KrishivError> {
        match &scan.source {
            ReadSource::Table { name, .. } => {
                // Analysis loaded every table a scan names; the scan
                // carries the canonical parts of its entry.
                let Some(entry) = self.find_loaded(name) else {
                    return Err(InternalError {
                        message: format!("the scanned table {} was not loaded", name.join(".")),
                    }
                    .into());
                };
                match entry {
                    LoadedEntry::Temp { table, .. } => match &*table {
                        TempTable::Memory { table, .. } => {
                            Ok(Arc::clone(table) as Arc<dyn TableProvider>)
                        }
                        TempTable::Files { .. } => {
                            let last = name.last().map_or("", String::as_str);
                            self.files_provider(&FilesKey::View(last.to_lowercase()))
                        }
                        // Views are inlined as subqueries before analysis, so
                        // no scan ever reads one.
                        TempTable::View { .. } => Err(InternalError {
                            message: "a scan of a temporary view that was not inlined".to_owned(),
                        }
                        .into()),
                    },
                    LoadedEntry::Table(t) => {
                        let ns = t
                            .ident
                            .namespace
                            .first()
                            .map_or("", String::as_str)
                            .to_lowercase();
                        self.files_provider(&FilesKey::Table(ns, t.ident.name.to_lowercase()))
                    }
                }
            }
            source @ ReadSource::Files { .. } => {
                self.files_provider(&FilesKey::Read(source.clone()))
            }
            // `ReadSource` is non-exhaustive: a later variant reads no
            // catalog entry.
            _ => Err(unsupported("this scan")),
        }
    }
}

/// Inlines the temporary views `plan` reads: a read of a view (a
/// single-part name with no options, not shadowed by a `WITH` definition)
/// becomes the view's plan as a subquery named for the view, with the
/// columns the view had when it was created. Tables, file views and reads
/// by path are left for the analyzer, as are reads a `WITH` definition
/// shadows (definitions shadow tables, as in Spark).
///
/// Views over views inline recursively; a view reading itself (through
/// `OR REPLACE`) is an error, as the engine has no recursive views.
///
/// # Errors
/// `INTERNAL_ERROR` for a view that reads itself.
pub(crate) fn expand_temp_views(
    catalog: &TempViews,
    current: &str,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    Expander {
        catalog,
        current,
        case_sensitive,
        path: Vec::new(),
    }
    .go(Arc::clone(plan), &mut Vec::new())
}

/// [`expand_temp_views`]: the views on the current expansion path, by
/// lower-case name, for cycle detection.
struct Expander<'a> {
    catalog: &'a TempViews,
    current: &'a str,
    case_sensitive: bool,
    path: Vec<String>,
}

impl Expander<'_> {
    /// `plan` with its views inlined; `scope` holds the `WITH`
    /// definitions visible where `plan` reads.
    fn go(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        // Snapshot pins are unreachable from user input however a read
        // spells them: a read of one behaves exactly as a missing table,
        // with or without read options. (The pins snapshot views read are
        // scans, which pass through untouched below.)
        if let PlanNode::Read(ReadSource::Table { name, .. }) = &plan.node
            && let [view] = name.as_slice()
            && !scope
                .iter()
                .any(|c| same_name(c, view, self.case_sensitive))
            && is_hidden_name(view)
        {
            return Err(TableOrViewNotFound {
                relation_name: quote(view),
                search_path: search_path(self.current),
            }
            .into());
        }
        if let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && let [view] = name.as_slice()
            && options.is_empty()
            && !scope
                .iter()
                .any(|c| same_name(c, view, self.case_sensitive))
        {
            // (Hidden names are rejected above, before this view lookup.)
            if let Some(entry) = self.catalog.get(view, self.case_sensitive)
                && let TempTable::View {
                    plan: definition,
                    columns,
                    ..
                } = &*entry
            {
                let key = view.to_lowercase();
                if self.path.contains(&key) {
                    return Err(InternalError {
                        message: format!("temporary view `{key}` reads itself"),
                    }
                    .into());
                }
                self.path.push(key);
                // A view's definitions are its own: outer `WITH` names do not
                // reach into it.
                let input = self.go(Arc::clone(definition), &mut Vec::new());
                self.path.pop();
                let columns = columns.iter().map(|c| c.name.clone()).collect();
                return Ok(Arc::new(
                    LogicalPlan::new(PlanNode::SubqueryAlias {
                        input: input?,
                        alias: view.clone(),
                        columns,
                    })
                    .with_origin(plan.origin),
                ));
            }
        }
        if let PlanNode::WithCte { input, ctes } = &plan.node {
            let (input, ctes) = (Arc::clone(input), ctes.clone());
            return self.go_with_cte(plan.origin, input, ctes, scope);
        }
        self.children_and_exprs(plan, scope)
    }

    /// `WITH` query parts (of `origin`) with their views inlined: each
    /// definition expands with only the preceding definitions in scope, as
    /// in Spark, so a body still reads a view of its own name.
    fn go_with_cte(
        &mut self,
        origin: Origin,
        input: Arc<LogicalPlan>,
        ctes: Vec<Cte>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let outer = scope.len();
        let mut expanded = Vec::with_capacity(ctes.len());
        for cte in ctes {
            let definition = self.go(Arc::clone(&cte.plan), scope)?;
            scope.push(cte.name.clone());
            expanded.push(Cte {
                name: cte.name,
                plan: definition,
            });
        }
        let input = self.go(input, scope)?;
        scope.truncate(outer);
        Ok(Arc::new(
            LogicalPlan::new(PlanNode::WithCte {
                input,
                ctes: expanded,
            })
            .with_origin(origin),
        ))
    }

    /// `plan` with its child relations and the subquery plans in its
    /// expressions expanded.
    fn children_and_exprs(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let plan = plan.map_children(|child| self.go(child, scope))?;
        plan.map_expressions(|expr| self.expand_expr(expr, scope))
    }

    /// `expr` with the views read by the subquery plans inside it inlined.
    fn expand_expr(
        &mut self,
        expr: Arc<Expr>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<Expr>, KrishivError> {
        let expr = expr.map_children(|child| self.expand_expr(child, scope))?;
        if let ExprNode::Subquery { kind, plan } = &expr.node {
            let expanded = self.go(Arc::clone(plan), scope)?;
            if !Arc::ptr_eq(&expanded, plan) {
                return Ok(Arc::new(Expr {
                    node: ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: expanded,
                    },
                    origin: expr.origin,
                }));
            }
        }
        Ok(expr)
    }
}

/// Whether creating the temporary view `name` over `plan` would close a
/// recursive reference: `plan` reads `name` itself, or a view that reads
/// it transitively. Spark reports that when the view is created, as
/// `RECURSIVE_VIEW`; the read-time guard in [`expand_temp_views`] stays
/// for anything analysis meets another way.
///
/// A read of `name` counts only when the view exists (an `OR REPLACE`
/// whose query sees the old definition): a fresh view over its own name
/// resolves nowhere, and analysis reports it as missing.
///
/// # Errors
/// `RECURSIVE_VIEW` naming the view and the cycle.
pub(crate) fn check_view_cycle(
    catalog: &TempViews,
    name: &str,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<(), KrishivError> {
    Checker {
        catalog,
        case_sensitive,
        target: name,
        stack: vec![name.to_owned()],
        scope: Vec::new(),
    }
    .check(plan)
}

/// [`check_view_cycle`]: the views on the current expansion path from the
/// new view down (by name as read), and the `WITH` definitions in scope.
struct Checker<'a> {
    catalog: &'a TempViews,
    case_sensitive: bool,
    target: &'a str,
    stack: Vec<String>,
    scope: Vec<String>,
}

impl Checker<'_> {
    fn check(&mut self, plan: &Arc<LogicalPlan>) -> Result<(), KrishivError> {
        if let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && let [view] = name.as_slice()
            && options.is_empty()
            && !self
                .scope
                .iter()
                .any(|c| same_name(c, view, self.case_sensitive))
        {
            if same_name(view, self.target, self.case_sensitive) {
                if self.catalog.get(self.target, self.case_sensitive).is_some() {
                    return Err(recursive_view(self.target, &self.stack));
                }
            } else if let Some(entry) = self.catalog.get(view, self.case_sensitive)
                && let TempTable::View {
                    plan: definition, ..
                } = &*entry
            {
                let key = view.to_lowercase();
                if self.stack.iter().any(|v| v.to_lowercase() == key) {
                    return Err(recursive_view(self.target, &self.stack));
                }
                self.stack.push(view.clone());
                // A view's definitions are its own: outer `WITH` names do
                // not reach into it.
                let outer = std::mem::take(&mut self.scope);
                let found = self.check(definition);
                self.scope = outer;
                self.stack.pop();
                found?;
            }
            return Ok(());
        }
        if let PlanNode::WithCte { input, ctes } = &plan.node {
            let outer = self.scope.len();
            for cte in ctes.iter() {
                self.check(&cte.plan)?;
                self.scope.push(cte.name.clone());
            }
            self.check(input)?;
            self.scope.truncate(outer);
            return Ok(());
        }
        let mut children = Vec::new();
        plan.for_each_child(&mut |child| children.push(Arc::clone(child)));
        for child in &children {
            self.check(child)?;
        }
        let mut exprs = Vec::new();
        plan.for_each_expr(&mut |expr| exprs.push(Arc::clone(expr)));
        for expr in &exprs {
            self.check_expr(expr)?;
        }
        Ok(())
    }

    /// `expr` with the views read by the subquery plans inside it checked.
    fn check_expr(&mut self, expr: &Arc<Expr>) -> Result<(), KrishivError> {
        let mut children = Vec::new();
        expr.for_each_child(&mut |child| children.push(Arc::clone(child)));
        for child in &children {
            self.check_expr(child)?;
        }
        if let ExprNode::Subquery { plan, .. } = &expr.node {
            self.check(plan)?;
        }
        Ok(())
    }
}

/// `RECURSIVE_VIEW` for a cycle through `stack` back to `target`.
fn recursive_view(target: &str, stack: &[String]) -> KrishivError {
    let mut path: Vec<String> = stack.iter().map(|v| quote(v)).collect();
    path.push(quote(target));
    RecursiveView {
        view_ident: quote(target),
        new_path: path.join(" -> "),
    }
    .into()
}

/// The pinned plan, the hidden names it reads (the new view owns them,
/// see [`TempTable::View`]) and the hidden entries to store with it.
type PinnedPlan = (Arc<LogicalPlan>, Vec<String>, Vec<(String, Arc<TempTable>)>);

/// Snapshots the temporary tables the analyzed `plan` reads: pins each
/// memory or file-backed entry under a hidden name and rewrites its scans
/// to it, so the plan stays bound to the tables it was analyzed against
/// however the session's names change afterwards (a view built from itself
/// reads the old definition; a view over a file-backed view survives its
/// replacement or removal). Reads by path are self-contained and need no
/// pinning.
///
/// A scan that already reads a pin (inlined from another view) reuses it,
/// so views over views share the pin instead of stacking a new one per
/// view. Returns the pinned plan, the hidden names it reads (the new view
/// owns them, see [`TempTable::View`]) and the hidden entries to store
/// with it: the caller inserts both atomically with
/// [`SessionCatalog::put_view`].
///
/// # Errors
/// `INTERNAL_ERROR` for a scan of a temporary view that was not inlined.
pub(crate) fn pin_temp_tables(
    catalog: &TempViews,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<PinnedPlan, KrishivError> {
    let mut pinner = Pinner {
        catalog,
        case_sensitive,
        pins: HashMap::new(),
        owned: BTreeSet::new(),
        new_pins: Vec::new(),
    };
    let plan = pinner.run(plan)?;
    Ok((plan, pinner.owned.into_iter().collect(), pinner.new_pins))
}

/// [`pin_temp_tables`]: the pins handed out so far, by lower-case table
/// name, every hidden name the pinned plan reads, and the hidden entries
/// to store with the owning view.
struct Pinner<'a> {
    catalog: &'a TempViews,
    case_sensitive: bool,
    pins: HashMap<String, String>,
    owned: BTreeSet<String>,
    new_pins: Vec<(String, Arc<TempTable>)>,
}

impl Pinner<'_> {
    fn run(&mut self, plan: &Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
        if let PlanNode::Scan(scan) = &plan.node
            && let ReadSource::Table { name, .. } = &scan.source
            && let Some(last) = name.last()
        {
            if is_hidden_name(last) {
                self.owned.insert(last.to_lowercase());
                return Ok(Arc::clone(plan));
            }
            let key = last.to_lowercase();
            if !self.pins.contains_key(&key) {
                let pin = match self.catalog.get(last, self.case_sensitive) {
                    Some(table)
                        if matches!(
                            &*table,
                            TempTable::Memory { .. } | TempTable::Files { .. }
                        ) =>
                    {
                        let hidden = self.catalog.next_hidden_name();
                        self.owned.insert(hidden.clone());
                        self.new_pins.push((hidden.clone(), table));
                        hidden
                    }
                    _ => String::new(),
                };
                self.pins.insert(key.clone(), pin);
            }
            if let Some(hidden) = self.pins.get(&key).filter(|h| !h.is_empty()) {
                let mut pinned = scan.clone();
                if let ReadSource::Table { name, .. } = &mut pinned.source {
                    *name = vec![hidden.clone()];
                }
                return Ok(Arc::new(
                    LogicalPlan::new(PlanNode::Scan(pinned)).with_origin(plan.origin),
                ));
            }
            return Ok(Arc::clone(plan));
        }
        let plan = Arc::clone(plan).map_children(|child| self.run(&child))?;
        plan.map_expressions(|expr| self.run_expr(&expr))
    }

    /// `expr` with the scans read by the subquery plans inside it pinned.
    fn run_expr(&mut self, expr: &Arc<Expr>) -> Result<Arc<Expr>, KrishivError> {
        let expr = Arc::clone(expr).map_children(|child| self.run_expr(&child))?;
        if let ExprNode::Subquery { kind, plan } = &expr.node {
            let pinned = self.run(plan)?;
            if !Arc::ptr_eq(&pinned, plan) {
                return Ok(Arc::new(Expr {
                    node: ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: pinned,
                    },
                    origin: expr.origin,
                }));
            }
        }
        Ok(expr)
    }
}

/// Whether `a` and `b` name the same view: exactly when `case_sensitive`,
/// else ignoring ASCII case, as the session catalog stores names.
fn same_name(a: &str, b: &str, case_sensitive: bool) -> bool {
    a == b || !case_sensitive && a.eq_ignore_ascii_case(b)
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Spark's search path, as analysis reports it for a missing table in
/// `current`.
fn search_path(current: &str) -> String {
    format!("[`system`.`builtin`, `system`.`session`, `{CATALOG}`.`{current}`]")
}

/// `name` rejected as a temporary table or view name: it starts with the
/// reserved snapshot prefix ([`HIDDEN_PREFIX`]), in any letter case. The
/// `KRISHIV_RESERVED_TEMP_VIEW_NAME` error names the reserved prefix in
/// its `prefix` parameter.
pub(crate) fn reserved_temp_view_name(name: &str) -> KrishivError {
    KrishivReservedTempViewName {
        relation_name: quote(name),
        prefix: format!("`{HIDDEN_PREFIX}`"),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array as _;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::Field;
    use futures::executor::block_on;
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DataType;

    use crate::session::Session;

    #[test]
    fn work_runs_on_the_given_runtime_and_a_drop_aborts_it() {
        let io = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("test-io")
            .enable_all()
            .build()
            .unwrap();
        let name = futures::executor::block_on(run_on(io.handle(), async {
            Ok(std::thread::current().name().map(str::to_owned))
        }))
        .unwrap();
        assert_eq!(name.as_deref(), Some("test-io"));
        // Dropping the waiting future aborts a task that never finishes.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let mut waiting = Box::pin(run_on(io.handle(), async move {
            let _held = tx;
            std::future::pending::<Result<(), KrishivError>>().await
        }));
        let waker = futures::task::noop_waker();
        let _ = waiting
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker));
        drop(waiting);
        assert_eq!(
            rx.recv_timeout(std::time::Duration::from_secs(5)),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected),
            "the aborted task must drop its sender"
        );
    }

    fn files(col: &str) -> TempTable {
        TempTable::Files {
            source: ReadSource::Files {
                format: "parquet".to_owned(),
                paths: vec![],
                options: Default::default(),
            },
            columns: vec![TableColumn {
                name: col.to_owned(),
                data_type: DataType::Integer,
                nullable: false,
            }],
        }
    }

    fn first_column(t: &TempTable) -> &str {
        match t {
            TempTable::Files { columns, .. }
            | TempTable::Memory { columns, .. }
            | TempTable::View { columns, .. } => &columns[0].name,
        }
    }

    #[test]
    fn expanding_a_cycle_is_an_internal_error() {
        // `CREATE TEMPORARY VIEW` rejects recursive views itself; the
        // expansion guard stays for anything else that closes a cycle.
        let session = TempViews::default();
        let view = |name: &str| TempTable::View {
            plan: Arc::new(LogicalPlan::table([name])),
            columns: Vec::new(),
            pins: Vec::new(),
        };
        session.put("a", view("b"));
        session.put("b", view("a"));
        let plan = Arc::new(LogicalPlan::table(["a"]));
        let e = expand_temp_views(&session, "default", &plan, false).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(e.message().contains("`a`"), "{e}");
    }

    #[test]
    fn every_spelling_binds_the_table_loaded_first() {
        let session = SessionCatalog::default();
        session.put("people", files("a"));
        let loaded = Mutex::new(HashMap::new());
        let form = NameForm {
            case_sensitive: false,
            qualified: false,
        };
        let (ident, t) = resolve(&loaded, &session, "people", form).unwrap();
        loaded.lock().unwrap().insert(
            LoadedKey::Temp("people".to_owned()),
            LoadedEntry::Temp { ident, table: t },
        );
        // Another thread replaces the table.
        session.put("people", files("b"));
        let (_, again) = resolve(&loaded, &session, "PEOPLE", form).unwrap();
        assert_eq!(first_column(&again), "a");
        // A case-sensitive spelling that is not registered does not resolve
        // through the loaded entry.
        let cs = NameForm {
            case_sensitive: true,
            ..form
        };
        assert!(resolve(&loaded, &session, "PEOPLE", cs).is_none());
    }

    /// A session with the memory table `t(id)` holding `values`.
    fn int_session(values: Vec<i32>) -> Session {
        let session = Session::new().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )
        .unwrap();
        session.register_batches("t", schema, vec![batch]).unwrap();
        session
    }

    /// The `id` column of `sql`, ordered.
    fn select_ids(session: &Session, sql: &str) -> Vec<i32> {
        let df = block_on(session.sql(sql)).unwrap();
        let batches = block_on(df.collect()).unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            out.extend((0..batch.num_rows()).map(|i| ids.value(i)));
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn view_creation_stores_pins_and_view_together() {
        // The pins and the owning view insert in one critical section, so
        // one epoch bump: no concurrent `put` or `drop_table` can collect
        // a new pin before its view arrives.
        let s = int_session(vec![1]);
        let before = s.state.catalog.epoch();
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        assert_eq!(s.state.catalog.epoch(), before + 1);
        assert_eq!(s.state.catalog.entry_count(), 3);
        assert_eq!(select_ids(&s, "SELECT id FROM v"), vec![1]);
    }

    #[test]
    fn reregistering_a_snapshot_view_does_not_grow_the_catalog() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        // The table, its pin and the view: replacing the view must not add
        // to that.
        assert_eq!(s.state.catalog.entry_count(), 3);
        for _ in 0..100 {
            block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        }
        assert_eq!(s.state.catalog.entry_count(), 3);
        assert_eq!(select_ids(&s, "SELECT id FROM v"), vec![1]);
    }

    #[test]
    fn dropping_a_view_removes_the_pins_it_owned() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        assert_eq!(s.state.catalog.entry_count(), 3);
        block_on(s.sql("DROP VIEW v")).unwrap();
        assert_eq!(s.state.catalog.entry_count(), 1);
    }

    #[test]
    fn a_pin_shared_by_two_views_survives_dropping_one() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v1")).unwrap();
        // A view over a view reuses the pin instead of stacking a new one.
        block_on(s.table("v1").create_or_replace_temp_view("v2")).unwrap();
        assert_eq!(s.state.catalog.entry_count(), 4);
        assert!(s.drop_temp_view("v1"));
        assert_eq!(s.state.catalog.entry_count(), 3);
        // The surviving view still reads the snapshot, however `t` changes.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![99]))],
        )
        .unwrap();
        s.register_batches("t", schema, vec![batch]).unwrap();
        assert_eq!(select_ids(&s, "SELECT id FROM v2"), vec![1]);
    }

    #[test]
    fn hidden_names_resolve_as_missing_tables() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        // Exactly as a missing table: same class, state and message shape.
        let missing = block_on(s.sql("SELECT * FROM nope")).unwrap_err();
        assert_eq!(missing.error_class(), ErrorClass::TableOrViewNotFound);
        let e = block_on(s.sql("SELECT * FROM `__krishiv_snapshot_0`")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
        assert_eq!(e.sql_state(), missing.sql_state());
        assert_eq!(
            e.message().replace("`__krishiv_snapshot_0`", "`nope`"),
            missing.message()
        );
        // Any letter case, quoted or not.
        let e = block_on(s.sql("SELECT * FROM __KRISHIV_SNAPSHOT_0")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
        // With read options too, while a pin of that name exists: the
        // options do not reach the pinned snapshot.
        let missing_options = block_on(s.sql("SELECT * FROM nope WITH ('a' = 'b')")).unwrap_err();
        assert_eq!(
            missing_options.error_class(),
            ErrorClass::TableOrViewNotFound
        );
        let e = block_on(s.sql("SELECT * FROM __krishiv_snapshot_0 WITH ('a' = 'b')")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
        assert_eq!(e.sql_state(), missing_options.sql_state());
        assert_eq!(
            e.message().replace("`__krishiv_snapshot_0`", "`nope`"),
            missing_options.message()
        );
    }

    #[test]
    fn view_built_from_itself_reads_the_old_definition() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("t")).unwrap();
        assert_eq!(select_ids(&s, "SELECT id FROM t"), vec![1]);
        // Rebuilding the self-view reuses the pin instead of orphaning one.
        block_on(s.table("t").create_or_replace_temp_view("t")).unwrap();
        assert_eq!(select_ids(&s, "SELECT id FROM t"), vec![1]);
        assert_eq!(s.state.catalog.entry_count(), 2);
    }

    #[test]
    fn hidden_names_are_rejected_for_new_views_and_tables() {
        let s = int_session(vec![1]);
        let e = block_on(
            s.table("t")
                .create_or_replace_temp_view("__krishiv_snapshot_7"),
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
        assert_eq!(e.error_class().name(), "KRISHIV_RESERVED_TEMP_VIEW_NAME");
        assert_eq!(e.sql_state(), Some("42602"));
        assert!(e.message().contains("`__krishiv_snapshot_`"), "{e}");
        let e = block_on(s.sql("CREATE TEMP VIEW `__krishiv_snapshot_7` AS SELECT id FROM t"))
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
        let e =
            block_on(s.sql("CREATE OR REPLACE TEMP VIEW V__KRISHIV_SNAPSHOT_ AS SELECT id FROM t"));
        // A name that merely contains the prefix is fine.
        assert!(e.is_ok(), "{e:?}");
        block_on(s.sql("DROP VIEW V__KRISHIV_SNAPSHOT_")).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let e = s
            .register_batches("__KRISHIV_SNAPSHOT_7", schema, vec![batch])
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
    }

    /// A table for the store tests: `ns.name` over `provider` at
    /// `/tmp/name`.
    fn store_table(ns: &str, name: &str) -> CatalogTable {
        CatalogTable {
            ident: TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![ns.to_owned()],
                name: name.to_owned(),
            },
            kind: CatalogTableKind::Managed,
            provider: "parquet".to_owned(),
            location: format!("/tmp/{name}"),
            columns: Vec::new(),
            col_comments: Vec::new(),
            col_defaults: Vec::new(),
            options: BTreeMap::new(),
            properties: BTreeMap::new(),
            comment: None,
            created_at: "0".to_owned(),
            owner: "u".to_owned(),
        }
    }

    /// A namespace for the store tests.
    fn store_namespace(name: &str) -> NamespaceMeta {
        NamespaceMeta {
            name: name.to_owned(),
            comment: None,
            location: format!("/tmp/{name}.db"),
            properties: BTreeMap::new(),
        }
    }

    #[test]
    fn store_namespaces_tables_and_modes() {
        let mut store = InMemoryCatalogStore::with_default();
        let epoch = store.epoch();
        // `default` always exists.
        assert!(store.namespace_exists("default", false));
        assert!(store.namespace_exists("DEFAULT", false));
        assert!(!store.namespace_exists("DEFAULT", true));
        assert!(
            !store
                .create_namespace(store_namespace("default"), true, false)
                .unwrap(),
            "IF NOT EXISTS keeps the existing namespace"
        );
        assert_eq!(store.epoch(), epoch, "no change bumps no epoch");
        assert!(
            store
                .create_namespace(store_namespace("MyDb"), false, false)
                .unwrap()
        );
        assert!(store.namespace_exists("mydb", false));
        assert!(store.namespace_exists("MyDb", false));
        assert!(store.namespace_exists("MyDb", true));
        assert!(!store.namespace_exists("MYDB", true));
        // Comments and properties round-trip.
        assert!(
            store
                .create_namespace(
                    NamespaceMeta {
                        name: "WithMeta".to_owned(),
                        comment: Some("c".to_owned()),
                        location: "/tmp/withmeta.db".to_owned(),
                        properties: BTreeMap::from([("k".to_owned(), "v".to_owned())]),
                    },
                    false,
                    false,
                )
                .unwrap()
        );
        let meta = store.namespaces.get("withmeta").unwrap();
        assert_eq!(meta.comment.as_deref(), Some("c"));
        assert_eq!(meta.properties.get("k").map(String::as_str), Some("v"));
        assert_eq!(meta.location, "/tmp/withmeta.db");
        store
            .drop_namespace("withmeta", false, true, false)
            .unwrap();
        assert_eq!(
            store
                .create_namespace(store_namespace("mydb"), false, false)
                .unwrap_err()
                .error_class(),
            ErrorClass::SchemaAlreadyExists
        );
        assert!(store.namespaces().contains(&"MyDb".to_owned()));
        // Tables need their namespace.
        assert_eq!(
            store
                .create_table(
                    store_table("missing", "t"),
                    CreateMode::ErrorIfExists,
                    false
                )
                .unwrap_err()
                .error_class(),
            ErrorClass::SchemaNotFound
        );
        assert!(
            store
                .create_table(store_table("MyDb", "T"), CreateMode::ErrorIfExists, false)
                .unwrap()
        );
        assert!(store.table("mydb", "t", false).is_some());
        assert!(store.table("MyDb", "T", false).is_some());
        assert!(store.table("MyDb", "T", true).is_some());
        assert!(store.table("MYDB", "T", true).is_none());
        assert!(store.table("mydb", "T", true).is_none());
        assert_eq!(store.list_tables("mydb", false).unwrap(), ["T"]);
        assert_eq!(
            store
                .list_tables("missing", false)
                .unwrap_err()
                .error_class(),
            ErrorClass::SchemaNotFound
        );
        assert_eq!(
            store
                .create_table(store_table("mydb", "t"), CreateMode::ErrorIfExists, false)
                .unwrap_err()
                .error_class(),
            ErrorClass::TableOrViewAlreadyExists
        );
        assert!(
            !store
                .create_table(store_table("mydb", "t"), CreateMode::IfNotExists, false)
                .unwrap()
        );
        assert!(
            store
                .create_table(store_table("mydb", "t"), CreateMode::Replace, false)
                .unwrap()
        );
        assert!(store.alter_table("mydb", "t", false, |t| {
            t.comment = Some("c".to_owned());
        }));
        assert_eq!(
            store.table("mydb", "t", false).unwrap().comment.as_deref(),
            Some("c")
        );
        assert!(!store.alter_table("mydb", "missing", false, |_| {}));
        assert!(store.drop_table("mydb", "missing", false).is_none());
        assert!(store.drop_table("mydb", "t", false).is_some());
        assert!(store.table("mydb", "t", false).is_none());
        // A non-empty namespace needs `CASCADE`.
        assert!(
            store
                .create_table(store_table("MyDb", "u"), CreateMode::ErrorIfExists, false)
                .unwrap()
        );
        assert_eq!(
            store
                .drop_namespace("mydb", false, false, false)
                .unwrap_err()
                .error_class(),
            ErrorClass::SchemaNotEmpty
        );
        let dropped = store.drop_namespace("mydb", false, true, false).unwrap();
        assert_eq!(dropped.len(), 1);
        assert!(!store.namespace_exists("mydb", false));
        // `default` cannot be dropped, however it is spelled; a missing
        // namespace fails unless `IF EXISTS`.
        assert_eq!(
            store
                .drop_namespace("default", false, true, false)
                .unwrap_err()
                .error_class(),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(
            store
                .drop_namespace("DEFAULT", true, true, false)
                .unwrap_err()
                .error_class(),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(
            store
                .drop_namespace("missing", false, true, false)
                .unwrap_err()
                .error_class(),
            ErrorClass::SchemaNotFound
        );
        assert!(
            store
                .drop_namespace("missing", true, true, false)
                .unwrap()
                .is_empty()
        );
    }
}
