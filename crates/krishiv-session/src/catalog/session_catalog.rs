// SPDX-License-Identifier: Apache-2.0
//! `SessionCatalog`'s methods: temporary tables and views, the current catalog, namespace and SQL path.

use super::*;

impl SessionCatalog {
    /// A session's catalog over the runtime's permanent `store` and
    /// `warehouse`.
    pub(crate) fn over(store: Arc<InMemoryCatalogStore>, warehouse: Arc<Warehouse>) -> Self {
        Self {
            tables: RwLock::default(),
            epoch: AtomicU64::new(0),
            hidden: AtomicU64::new(0),
            store,
            warehouse,
            position: RwLock::default(),
            path: RwLock::default(),
        }
    }

    /// A catalog over the same permanent store and warehouse holding this
    /// one's temporary tables and views, current catalog, namespace and
    /// path; later changes to either do not reach the other. Script
    /// results no copied view pins are left out: their frames belong to
    /// this catalog's session.
    pub(crate) fn copy(&self) -> Self {
        let mut tables = self
            .tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        script_results::leave_out(&mut tables);
        let position = self
            .position
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let path = self
            .path
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        Self {
            tables: RwLock::new(tables),
            epoch: AtomicU64::new(0),
            hidden: AtomicU64::new(self.hidden.load(Ordering::Acquire)),
            store: Arc::clone(&self.store),
            warehouse: Arc::clone(&self.warehouse),
            position: RwLock::new(position),
            path: RwLock::new(path),
        }
    }

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
        keeps: ViewKeeps,
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
                    text: None,
                    keeps,
                }),
            ),
        );
        Self::collect_garbage(&mut tables);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// Removes the hidden entries nothing needs; the caller holds the write lock and bumps the epoch.
    pub(super) fn collect_garbage(tables: &mut BTreeMap<String, (String, Arc<TempTable>)>) {
        let mut live = BTreeSet::new();
        for (_, table) in tables.values() {
            if let TempTable::View { pins, .. } = &**table {
                live.extend(pins.iter().cloned());
            }
        }
        script_results::retain(tables, &live);
    }

    /// Hands out the next hidden name snapshot pins use: no SQL text
    /// spells it, so snapshot views read it however the session's names
    /// change afterwards. The caller inserts the entry with
    /// [`SessionCatalog::put_view`], which stores the pins and the owning
    /// view atomically.
    pub(super) fn next_hidden_name(&self) -> String {
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
            .filter(|n| !n.starts_with(HIDDEN_PREFIX) && !n.starts_with(GLOBAL_PREFIX))
            .collect()
    }

    /// The names of the global temporary views, as registered, in order.
    pub(crate) fn global_names(&self) -> Vec<String> {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .filter_map(|(n, _)| n.strip_prefix(GLOBAL_PREFIX).map(str::to_owned))
            .collect()
    }

    /// The namespace bare table names resolve in.
    pub(crate) fn current_namespace(&self) -> String {
        self.position_pair().1
    }

    /// The current catalog's canonical name and namespace, read together.
    pub(crate) fn position_pair(&self) -> (String, String) {
        let p = self.position.read().unwrap_or_else(PoisonError::into_inner);
        let catalog = if p.0.is_empty() { CATALOG } else { &p.0 };
        let namespace = if p.1.is_empty() { NAMESPACE } else { &p.1 };
        (catalog.to_owned(), namespace.to_owned())
    }

    /// The stored path entries, `None` for the default.
    pub(crate) fn stored_path(&self) -> Option<Vec<crate::sql_path::Entry>> {
        self.path
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// `SET PATH`: stores the expanded entries.
    pub(crate) fn set_path(&self, entries: Vec<crate::sql_path::Entry>) {
        *self.path.write().unwrap_or_else(PoisonError::into_inner) = Some(entries);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// The SQL path as parts, `CURRENT_SCHEMA` resolved; `default_conf` is
    /// `krishiv.sql.defaultPath`, read when no `SET PATH` ran.
    pub(crate) fn sql_path(&self, default_conf: &str) -> Vec<Vec<String>> {
        let entries = self
            .stored_path()
            .unwrap_or_else(|| crate::sql_path::default_entries(default_conf));
        let current = self.current_position();
        entries.iter().map(|e| e.parts(&current)).collect()
    }

    /// The current catalog and namespace as path parts: `[spark_catalog,
    /// db]`, or `[system]`.
    pub(crate) fn current_position(&self) -> Vec<String> {
        let (catalog, namespace) = self.position_pair();
        if crate::system::is_system(&catalog) {
            vec![catalog]
        } else {
            vec![catalog, namespace]
        }
    }

    /// The current catalog's canonical name.
    pub(crate) fn current_catalog(&self) -> String {
        self.position_pair().0
    }

    /// `USE <catalog>`: a no-op when `catalog` is current, else its default
    /// namespace (the default database for `spark_catalog`, none for `system`).
    pub(crate) fn use_catalog(&self, catalog: &str) {
        {
            let mut p = self
                .position
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            let current = if p.0.is_empty() {
                CATALOG
            } else {
                p.0.as_str()
            };
            if current == catalog {
                return;
            }
            *p = (catalog.to_owned(), String::new());
        }
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// `USE namespace`.
    pub(crate) fn use_namespace(&self, namespace: &str) {
        namespace.clone_into(
            &mut self
                .position
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .1,
        );
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// What a cached analysis depends on: temporary views and permanent
    /// tables both bump it.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire) + self.store.epoch()
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
