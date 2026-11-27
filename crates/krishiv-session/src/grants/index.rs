// SPDX-License-Identifier: Apache-2.0
//! What the grants provider needs to know about the session catalog, as lookup tables rebuilt only
//! when the catalog changes (its epoch; review re-check 2 P1): case-folded names with their case
//! twins (for [`GrantsPolicy::exact_for`](super::GrantsPolicy::exact_for)) and the local
//! directories of tables (for a managed table's directory rights). A check reads the tables; it
//! never walks the catalog or touches the filesystem. Resolving a location's links happens when
//! the tables are rebuilt, once per warehouse root and once per explicit location (both kept
//! across rebuilds).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::Securable;

use super::model::{KeyKind, ObjectKey};
use crate::store::{CatalogStore as _, InMemoryCatalogStore};

/// The lookup tables at one catalog epoch.
#[derive(Debug, Default)]
pub(crate) struct CatalogIndex {
    /// Namespaces by folded name: how many spellings exist.
    namespaces: HashMap<String, usize>,
    /// Tables and views by folded (namespace, name): each spelling's columns.
    relations: HashMap<(String, String), Vec<Vec<String>>>,
    /// Every table's local directory (physical path key) and its relation key, most specific
    /// first (longest key).
    dirs: Vec<(ObjectKey, ObjectKey, bool)>,
}

impl CatalogIndex {
    /// Whether `key` (of the session catalog) or a parent has a case twin.
    pub(crate) fn has_twin(&self, key: &ObjectKey) -> bool {
        let p = &key.parts;
        let Some(ns) = p.get(1) else {
            return false;
        };
        let ns = ns.to_lowercase();
        if self.namespaces.get(&ns).copied().unwrap_or(0) > 1 || p.len() > 4 {
            return true;
        }
        let Some(name) = p.get(2) else {
            return false;
        };
        let spellings = self
            .relations
            .get(&(ns, name.to_lowercase()))
            .map_or(&[][..], Vec::as_slice);
        if spellings.len() > 1 {
            return true;
        }
        let Some(column) = p.get(3) else {
            return false;
        };
        spellings.first().is_some_and(|columns| {
            columns
                .iter()
                .filter(|c| c.eq_ignore_ascii_case(column))
                .count()
                > 1
        })
    }

    /// The table whose directory is the most specific one holding `path`: its relation key and
    /// whether it is managed. A directory that holds another table's location belongs to that
    /// table (review re-check 2 r2-m1).
    pub(crate) fn owner_of_dir(&self, path: &ObjectKey) -> Option<(&ObjectKey, bool)> {
        self.dirs
            .iter()
            .find(|(dir, _, _)| path.covered_by(dir, true))
            .map(|(_, relation, managed)| (relation, *managed))
    }
}

/// The index, rebuilt when the catalog's epoch moves, and the resolved locations it reuses.
#[derive(Debug, Default)]
pub(crate) struct IndexCache {
    current: Mutex<Option<(u64, Arc<CatalogIndex>)>>,
    resolved: Mutex<HashMap<PathBuf, Option<ObjectKey>>>,
}

impl IndexCache {
    /// The index of `store` at its current epoch.
    pub(crate) fn get(&self, store: &InMemoryCatalogStore) -> Arc<CatalogIndex> {
        let epoch = store.epoch();
        if let Ok(g) = self.current.lock()
            && let Some((e, index)) = g.as_ref()
            && *e == epoch
        {
            return Arc::clone(index);
        }
        let index = Arc::new(self.build(store));
        if let Ok(mut g) = self.current.lock() {
            *g = Some((epoch, Arc::clone(&index)));
        }
        index
    }

    fn build(&self, store: &InMemoryCatalogStore) -> CatalogIndex {
        let mut index = CatalogIndex::default();
        for (ns, _) in store.namespaces() {
            *index.namespaces.entry(ns.to_lowercase()).or_default() += 1;
            for t in store.tables_in(&ns) {
                index
                    .relations
                    .entry((ns.to_lowercase(), t.name.to_lowercase()))
                    .or_default()
                    .push(t.columns.iter().map(|c| c.name.clone()).collect());
                if t.view.is_some() {
                    continue;
                }
                // A managed directory is its warehouse root (resolved once) joined with the
                // rest; an explicit location is resolved once.
                let dir = match t.location.strip_prefix(&t.root) {
                    Ok(rest) if t.managed => self
                        .resolved_dir(&t.root)
                        .and_then(|root| local_key(&root_path(&root)?.join(rest))),
                    _ => self.resolved_dir(&t.location),
                };
                let Some(dir) = dir else {
                    continue;
                };
                let relation = ObjectKey::new(
                    KeyKind::Relation,
                    vec![
                        crate::catalog::CATALOG.to_owned(),
                        t.namespace.clone(),
                        t.name.clone(),
                    ],
                );
                index.dirs.push((dir, relation, t.managed));
            }
        }
        index
            .dirs
            .sort_by_key(|(dir, _, _)| std::cmp::Reverse(dir.parts.first().map_or(0, String::len)));
        index
    }

    /// The physical path key of the local directory `path`, resolved once.
    fn resolved_dir(&self, path: &Path) -> Option<ObjectKey> {
        if let Ok(g) = self.resolved.lock()
            && let Some(k) = g.get(path)
        {
            return k.clone();
        }
        let key = local_key(path).and_then(super::physical_key);
        if let Ok(mut g) = self.resolved.lock() {
            g.insert(path.to_path_buf(), key.clone());
        }
        key
    }
}

/// The path key of an absolute local path (lexical).
fn local_key(path: &Path) -> Option<ObjectKey> {
    if !path.is_absolute() {
        return None;
    }
    // `file:///C:/x` on Windows: a `\` or a bare drive would not read as a path key.
    let text = krishiv_exec::files::file_url(path);
    ObjectKey::of(&Securable::Path(OptionValue::from_user("path", text)))
}

/// The local path a `file:///...` key names (`C:/x` for `file:///C:/x` on Windows).
fn root_path(key: &ObjectKey) -> Option<PathBuf> {
    key.parts
        .first()
        .filter(|p| p.starts_with("file://"))
        .map(|p| krishiv_exec::files::native_path(p))
}
