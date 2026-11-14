// SPDX-License-Identifier: Apache-2.0
//! The session's in-memory store (`CatalogStore`) seen through the public
//! catalog traits (design P3-lakehouse §2.1, P3-T020a): `StoreCatalog` is a
//! `Catalog` and a `CatalogWrite`, so the shared contract suite and the DDL
//! commands that follow can talk to every catalog the same way. It changes no
//! existing caller; the store's own methods stay what the session uses until
//! the commands move onto the traits.
//!
//! A table's version is [`CatalogTable::version`], the store's stamp of the
//! entry: `alter_table` fails on a table that changed (or was dropped and
//! created again, or analyzed) since it was read, as `replace_table` always
//! did; the class is `INTERNAL_ERROR`, as the session raises today.
//!
//! The catalog authorizes nothing: the caller checks the policy (the session
//! does so per path before it calls). What it does check is the safety of a
//! delete: only a managed table's directory, inside the warehouse root the
//! table was created under, never through a link, and before any metadata
//! changes, so a refused delete leaves the table in place.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the DDL commands move onto the catalog traits in the next commits of P3-T020a"
    )
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use krishiv_common::error::{
    DataSourceNotFound, InternalError, KrishivError, LegacyErrorTemp1073, LocationAlreadyExists,
    SchemaAlreadyExists, SchemaNotEmpty, SchemaNotFound, TableOrViewAlreadyExists,
    TableOrViewNotFound, UnsupportedFeatureCatalogOperation,
};
use krishiv_common::names::check_column_name_duplication;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    AlterInput, AlterRules, BoxFuture, Catalog, CatalogCapabilities, CatalogWrite, CreateMode,
    NameForm, NamespaceChange, NamespaceInfo, TableChange, TableColumn, TableIdent, TableInfo,
    TableKind, TableMeta, apply_changes,
};

use crate::catalog::CATALOG;
use crate::store::CreateMode as StoreCreateMode;
use crate::store::{
    CatalogStore, CatalogTable, DropNamespaceError, NamespaceMeta, StoreColumn, TableLayout,
    managed_location,
};
use crate::table_commands::{
    OwnedDirectory, OwnedDirectoryError, PROVIDERS, TableName, check_new_name,
};

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// The directory whose files a delete of `table` removes, after every check
/// (a managed table, inside the root it was created under, not that root, no
/// link at the end, resolved paths): `None` when there is nothing to delete.
/// Run before the metadata changes. The session's own `deletable_dir`, less
/// the policy check.
fn deletable_dir(table: &CatalogTable) -> Result<Option<PathBuf>, KrishivError> {
    if !table.managed || table.view.is_some() {
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
    Ok(Some(canonical))
}

/// Deletes a directory; one that is already gone is fine.
fn remove_dir(dir: &Path) -> Result<(), KrishivError> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(internal(format!("cannot delete {}: {e}", dir.display()))),
    }
}

fn to_column(c: &StoreColumn) -> TableColumn {
    TableColumn {
        name: c.name.clone(),
        data_type: c.data_type.clone(),
        nullable: c.nullable,
        char_varchar: c.char_varchar.clone(),
    }
}

fn concurrent(id: &TableIdent) -> KrishivError {
    internal(format!(
        "{id} was changed by another statement while this one ran; retry"
    ))
}

/// The in-memory store as a catalog named `spark_catalog`. Names fold as
/// `case_sensitive` says, which the session fixes per query, so a catalog is
/// made for one query's view of the configuration.
pub(crate) struct StoreCatalog {
    store: Arc<dyn CatalogStore>,
    /// Where managed tables live.
    warehouse: PathBuf,
    case_sensitive: bool,
}

impl std::fmt::Debug for StoreCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreCatalog")
            .field("warehouse", &self.warehouse)
            .field("case_sensitive", &self.case_sensitive)
            .finish_non_exhaustive()
    }
}

impl StoreCatalog {
    /// `store` seen as a catalog whose managed tables live under `warehouse`.
    pub(crate) fn new(
        store: Arc<dyn CatalogStore>,
        warehouse: PathBuf,
        case_sensitive: bool,
    ) -> Self {
        Self {
            store,
            warehouse,
            case_sensitive,
        }
    }

    fn fold(&self, name: &str) -> String {
        if self.case_sensitive {
            name.to_owned()
        } else {
            name.to_lowercase()
        }
    }

    fn unsupported(&self, operation: &str) -> KrishivError {
        UnsupportedFeatureCatalogOperation {
            catalog_name: format!("`{CATALOG}`"),
            operation: operation.to_owned(),
        }
        .into()
    }

    /// The one part of a namespace; the session catalog has no nested ones.
    fn namespace(&self, namespace: &[String]) -> Result<String, KrishivError> {
        match namespace {
            [one] => Ok(self.fold(one)),
            _ => Err(self.unsupported("a namespace of more than one part")),
        }
    }

    fn table(&self, id: &TableIdent) -> Result<(String, String), KrishivError> {
        Ok((self.namespace(&id.namespace)?, self.fold(&id.name)))
    }

    fn not_found(id: &TableIdent) -> KrishivError {
        TableOrViewNotFound {
            relation_name: id.to_string(),
            search_path: crate::catalog::search_path(),
        }
        .into()
    }

    fn to_meta(&self, table: &CatalogTable) -> TableMeta {
        TableMeta {
            ident: TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![table.namespace.clone()],
                name: table.name.clone(),
            },
            columns: table.columns.iter().map(to_column).collect(),
            version: Some(table.version()),
            kind: if table.view.is_some() {
                TableKind::View
            } else {
                TableKind::Table
            },
            file_metadata: crate::catalog::file_metadata(&table.provider),
            column_comments: table
                .columns
                .iter()
                .filter_map(|c| Some((c.name.clone(), c.comment.clone()?)))
                .collect(),
        }
    }

    /// The new table's columns and layout, checked as `CREATE TABLE` checks them: no repeated column, the
    /// declared schema writable by `provider`, and (through the session's `table_layout`) partition columns
    /// that exist, are not repeated, are not every column and have a partitionable type, moved last in
    /// partition order and spelled as the table spells them.
    fn arranged(
        &self,
        info: &TableInfo,
        namespace: &str,
        name: &str,
        provider: &str,
    ) -> Result<(Vec<StoreColumn>, TableLayout), KrishivError> {
        check_column_name_duplication(
            info.columns.iter().map(|c| c.name.as_str()),
            self.case_sensitive,
        )?;
        krishiv_io::check_declared_schema(
            provider,
            &arrow_schema::Schema::new(
                info.columns
                    .iter()
                    .map(|c| c.data_type.to_arrow_field(&c.name, true))
                    .collect::<Vec<_>>(),
            ),
        )?;
        let mut columns: Vec<StoreColumn> = info
            .columns
            .iter()
            .map(|c| StoreColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                // File-source tables are nullable whatever the definition says: nothing is enforced on write,
                // and a non-nullable flag would let the optimizer fold `IS NULL` over files that hold NULLs.
                nullable: true,
                default: None,
                comment: None,
                char_varchar: c.char_varchar.clone(),
            })
            .collect();
        let table = TableName {
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            written: info.ident.to_string(),
            bare: false,
        };
        let layout = crate::partitions::table_layout(
            &table,
            &mut columns,
            &info.partition_columns,
            None,
            self.case_sensitive,
        )?;
        Ok((columns, layout))
    }
}

impl Catalog for StoreCatalog {
    fn name(&self) -> &str {
        CATALOG
    }

    fn epoch(&self) -> u64 {
        self.store.epoch()
    }

    fn single_part_namespace(&self) -> bool {
        true
    }

    fn capabilities(&self) -> CatalogCapabilities {
        CatalogCapabilities::read_write()
    }

    fn as_write(&self) -> Option<&dyn CatalogWrite> {
        Some(self)
    }

    /// The table at its current version: this catalog keeps no history, so a
    /// `version` other than the current one finds nothing.
    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            // The path names no schema of this catalog: only temporary relations resolve, and it has none.
            if form.temporary_only {
                return Ok(None);
            }
            let [namespace] = id.namespace.as_slice() else {
                return Ok(None);
            };
            let (namespace, name) = if form.case_sensitive {
                (namespace.clone(), id.name.clone())
            } else {
                (namespace.to_lowercase(), id.name.to_lowercase())
            };
            Ok(self
                .store
                .table(&namespace, &name)
                .filter(|t| version.is_none_or(|v| v == t.version()))
                .map(|t| self.to_meta(&t)))
        })
    }
}

impl CatalogWrite for StoreCatalog {
    fn create_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        info: NamespaceInfo,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let name = self.namespace(namespace)?;
            check_new_name(&name)?;
            // `global_temp` is the reserved home of global temporary views: it always exists.
            if name.eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE) {
                return Err(SchemaAlreadyExists {
                    schema_name: format!("`{name}`"),
                }
                .into());
            }
            let meta = NamespaceMeta {
                collation: None,
                comment: info.comment,
                location: info.location,
                properties: info.properties,
            };
            if self.store.create_namespace(&name, meta) {
                Ok(())
            } else {
                Err(SchemaAlreadyExists {
                    schema_name: format!("`{name}`"),
                }
                .into())
            }
        })
    }

    fn alter_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        changes: &'a [NamespaceChange],
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let name = self.namespace(namespace)?;
            if changes.iter().any(|c| {
                !matches!(
                    c,
                    NamespaceChange::SetProperty { .. } | NamespaceChange::RemoveProperty { .. }
                )
            }) {
                return Err(self.unsupported("this namespace change"));
            }
            let apply = |meta: &mut NamespaceMeta| {
                for change in changes {
                    match change {
                        NamespaceChange::SetProperty { key, value } => {
                            meta.properties.insert(key.clone(), value.clone());
                        }
                        NamespaceChange::RemoveProperty { key } => {
                            meta.properties.remove(key);
                        }
                        _ => {}
                    }
                }
            };
            if self.store.alter_namespace(&name, &apply) {
                Ok(())
            } else {
                Err(SchemaNotFound {
                    schema_name: format!("`{name}`"),
                }
                .into())
            }
        })
    }

    /// Drops the namespace; with `cascade` its managed tables' directories
    /// too. Every directory is checked before the metadata changes; a delete
    /// that fails after it does not stop the others, and the first failure is
    /// returned.
    fn drop_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        cascade: bool,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let name = self.namespace(namespace)?;
            // The directories to delete are decided here, once, before the metadata changes, and
            // are the only ones deleted after it: `drop_namespace_matching` drops only if the
            // namespace still holds exactly these entries, so nothing found later (a table made
            // again meanwhile, a directory that appeared) can be deleted.
            let (checked, dirs) = if cascade {
                let held = self.store.tables_in(&name);
                let mut dirs = Vec::new();
                for table in &held {
                    dirs.extend(deletable_dir(table)?);
                }
                (Some(held), dirs)
            } else {
                (None, Vec::new())
            };
            match self
                .store
                .drop_namespace_matching(&name, cascade, checked.as_deref())
            {
                Ok(_dropped) => {}
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
                Err(DropNamespaceError::Default) => {
                    return Err(self.unsupported("dropping the default namespace"));
                }
                Err(DropNamespaceError::Changed) => {
                    return Err(internal(format!(
                        "`{name}` was changed by another statement while this one ran; retry"
                    )));
                }
            };
            let mut first = None;
            for dir in &dirs {
                if let Err(e) = remove_dir(dir) {
                    first.get_or_insert(e);
                }
            }
            first.map_or(Ok(()), Err)
        })
    }

    fn create_table<'a>(
        &'a self,
        info: TableInfo,
        mode: CreateMode,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        Box::pin(async move {
            let store_mode = match mode {
                CreateMode::ErrorIfExists => StoreCreateMode::ErrorIfExists,
                CreateMode::IgnoreIfExists => StoreCreateMode::IfNotExists,
                _ => return Err(self.unsupported("this create mode")),
            };
            let (namespace, name) = self.table(&info.ident)?;
            check_new_name(&name)?;
            if !self.store.has_namespace(&namespace) {
                return Err(SchemaNotFound {
                    schema_name: format!("`{namespace}`"),
                }
                .into());
            }
            // As the SQL create does (`Session::refuse_collated_namespace`): the schema's default collation
            // is not applied to new columns yet, so a table here would silently compare as `UTF8_BINARY`.
            if self
                .store
                .namespaces()
                .into_iter()
                .any(|(n, m)| n.eq_ignore_ascii_case(&namespace) && m.collation.is_some())
            {
                return Err(self.unsupported("a table in a schema with a DEFAULT COLLATION"));
            }
            // A table that exists settles the call before the definition is looked at.
            if self.store.table(&namespace, &name).is_some() {
                return match store_mode {
                    StoreCreateMode::IfNotExists => Ok(false),
                    _ => Err(TableOrViewAlreadyExists {
                        relation_name: info.ident.to_string(),
                    }
                    .into()),
                };
            }
            let mut properties = info.properties.clone();
            let written = properties.remove("provider");
            let provider = written.clone().unwrap_or_else(|| "parquet".to_owned());
            if !PROVIDERS.contains(&provider.to_lowercase().as_str()) {
                return Err(DataSourceNotFound { provider }.into());
            }
            if info.location.as_deref().is_some_and(str::is_empty) {
                return Err(self.unsupported("an empty location"));
            }
            let (columns, layout) =
                self.arranged(&info, &namespace, &name, &provider.to_lowercase())?;
            let managed = info.location.is_none();
            let location = info.location.as_deref().map_or_else(
                || managed_location(&self.warehouse, &namespace, &name),
                |l| PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            );
            let table = CatalogTable {
                namespace,
                name,
                managed,
                provider_written: written.unwrap_or_else(|| provider.clone()),
                provider: provider.to_lowercase(),
                location: location.clone(),
                columns,
                options: BTreeMap::new(),
                properties,
                comment: info.comment,
                created_ms: crate::table_commands::now_ms(),
                root: self.warehouse.clone(),
                view: None,
                layout,
                column_stats: BTreeMap::new(),
                incarnation: 0,
                data_version: 0,
                stamp: 0,
            };
            // The directory first, made exclusively (a directory that is there is not ours to take), as the
            // SQL create does: a drop that sees the entry finds its files, and a refused registration removes
            // the directory this call made and nothing else.
            let owned = managed
                .then(|| OwnedDirectory::create(location.clone()))
                .transpose()
                .map_err(|e| match e {
                    OwnedDirectoryError::Exists => KrishivError::from(LocationAlreadyExists {
                        identifier: info.ident.to_string(),
                        location: location.display().to_string(),
                    }),
                    OwnedDirectoryError::Io(message) => internal(message),
                })?;
            let created = self.store.create_table(table, store_mode)?;
            if created && let Some(owned) = owned {
                owned.keep();
            }
            Ok(created)
        })
    }

    fn alter_table<'a>(
        &'a self,
        id: &'a TableIdent,
        expected: TableVersion,
        changes: &'a [TableChange],
    ) -> BoxFuture<'a, Result<TableMeta, KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table(id)?;
            let current = self
                .store
                .table(&namespace, &name)
                .ok_or_else(|| Self::not_found(id))?;
            if current.view.is_some() {
                return Err(self.unsupported("altering a view as a table"));
            }
            if current.version() != expected {
                return Err(concurrent(id));
            }
            let columns: Vec<TableColumn> = current.columns.iter().map(to_column).collect();
            let comments: BTreeMap<String, String> = current
                .columns
                .iter()
                .filter_map(|c| Some((c.name.clone(), c.comment.clone()?)))
                .collect();
            let defaults: BTreeMap<String, String> = current
                .columns
                .iter()
                .filter_map(|c| Some((c.name.clone(), c.default.clone()?)))
                .collect();
            let table_id = TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![namespace],
                name,
            };
            let input = AlterInput {
                table: &table_id,
                columns: &columns,
                comments: &comments,
                defaults: &defaults,
                partition_columns: &current.layout.partition_columns,
            };
            let altered = apply_changes(&input, changes, AlterRules::new(self.case_sensitive))?;
            // The provider must be able to write the columns that result, as at create.
            krishiv_io::check_declared_schema(
                &current.provider,
                &arrow_schema::Schema::new(
                    altered
                        .columns
                        .iter()
                        .map(|c| c.data_type.to_arrow_field(&c.name, true))
                        .collect::<Vec<_>>(),
                ),
            )?;
            let mut changed = (*current).clone();
            changed.columns = altered
                .columns
                .iter()
                .map(|c| StoreColumn {
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    // Nullable, as at create; an added column is NULL in the files written before it.
                    nullable: true,
                    default: altered.defaults.get(&c.name).cloned(),
                    comment: altered.comments.get(&c.name).cloned(),
                    char_varchar: c.char_varchar.clone(),
                })
                .collect();
            for (key, value) in altered.properties {
                // Spark's `comment` property is the table's comment.
                if key.eq_ignore_ascii_case("comment") {
                    changed.comment = value;
                } else if let Some(v) = value {
                    changed.properties.insert(key, v);
                } else {
                    changed.properties.remove(&key);
                }
            }
            // The entry the compare-and-swap stored, with its stamp: never another thread's later one.
            let stored = self
                .store
                .replace_table_entry(&current, changed)
                .ok_or_else(|| concurrent(id))?;
            Ok(self.to_meta(&stored))
        })
    }

    /// Drops the table. A managed table's directory is deleted with it
    /// (Spark has no trash, so `purge` changes nothing here), an external
    /// table's never; the delete is checked before the metadata changes.
    fn drop_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _purge: bool,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table(id)?;
            let Some(current) = self.store.table(&namespace, &name) else {
                return Ok(false);
            };
            if current.view.is_some() {
                return Err(self.unsupported("dropping a view as a table"));
            }
            let dir = deletable_dir(&current)?;
            // Only the entry that passed the checks: a table dropped and created again meanwhile is not ours.
            if self.store.drop_table_entry(&current).is_none() {
                return if self.store.table(&namespace, &name).is_some() {
                    Err(concurrent(id))
                } else {
                    Ok(false)
                };
            }
            if let Some(dir) = dir {
                remove_dir(&dir)?;
            }
            Ok(true)
        })
    }

    /// Renames a table within its namespace, as `ALTER TABLE .. RENAME TO` does: a table never changes namespace
    /// (Spark's session catalog refuses it, so `rename_across_namespaces` is off). A managed table's directory moves
    /// first, then the metadata; a failure in between puts the directory back and restores the entry. The move is
    /// checked first (inside the warehouse, no link), and the entry swapped is the one that was checked.
    fn rename_table<'a>(
        &'a self,
        from: &'a TableIdent,
        to: &'a TableIdent,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table(from)?;
            let (to_namespace, to_name) = self.table(to)?;
            check_new_name(&to_namespace)?;
            check_new_name(&to_name)?;
            if !self.store.has_namespace(&to_namespace) {
                return Err(SchemaNotFound {
                    schema_name: format!("`{to_namespace}`"),
                }
                .into());
            }
            let table = self
                .store
                .table(&namespace, &name)
                .ok_or_else(|| Self::not_found(from))?;
            if table.view.is_some() {
                return Err(self.unsupported("renaming a view as a table"));
            }
            if to_namespace != namespace {
                return Err(LegacyErrorTemp1073 {
                    db: namespace,
                    new_db: to_namespace,
                }
                .into());
            }
            if self.store.table(&to_namespace, &to_name).is_some() {
                return Err(TableOrViewAlreadyExists {
                    relation_name: to.to_string(),
                }
                .into());
            }
            let mut renamed = (*table).clone();
            renamed.namespace.clone_from(&to_namespace);
            renamed.name.clone_from(&to_name);
            let source_dir = deletable_dir(&table)?;
            if table.managed {
                renamed.location = managed_location(&self.warehouse, &to_namespace, &to_name);
                renamed.root.clone_from(&self.warehouse);
                // Partitions inside the table's directory move with it.
                for p in &mut renamed.layout.partitions {
                    if let Ok(rest) = p.location.strip_prefix(&table.location) {
                        p.location = renamed.location.join(rest);
                    }
                }
                // A directory that is there is not ours to take (an empty one would be replaced by the move), and
                // it is not ours whether or not the source directory still exists.
                if renamed.location.symlink_metadata().is_ok() {
                    return Err(LocationAlreadyExists {
                        identifier: to.to_string(),
                        location: renamed.location.display().to_string(),
                    }
                    .into());
                }
                if let (Some(source), Some(parent)) = (&source_dir, renamed.location.parent()) {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        internal(format!("cannot create {}: {e}", parent.display()))
                    })?;
                    std::fs::rename(source, &renamed.location).map_err(|e| {
                        internal(format!(
                            "cannot move {} to {}: {e}",
                            source.display(),
                            renamed.location.display()
                        ))
                    })?;
                }
            }
            let moved_to = renamed.location.clone();
            let dir_moved = table.managed && source_dir.is_some();
            // The entry that was checked, swapped for the renamed one under one lock: both names or neither.
            let swapped = self.store.rename_table_entry(&table, renamed);
            if !matches!(swapped, Ok(Some(_))) {
                if dir_moved && let Some(source) = &source_dir {
                    // The source path may have a new owner by now (a table dropped and created again under the old
                    // name); a move back would replace its directory, so the files stay where they are and the
                    // failure says so.
                    if source.symlink_metadata().is_ok() {
                        return Err(internal(format!(
                            "the rename failed and {} is in use again; the files stay at {}",
                            source.display(),
                            moved_to.display()
                        )));
                    }
                    std::fs::rename(&moved_to, source).map_err(|e| {
                        internal(format!(
                            "the rename failed and {} could not be moved back to {}: {e}",
                            moved_to.display(),
                            source.display()
                        ))
                    })?;
                }
                return match swapped {
                    Err(e) => Err(e),
                    _ => Err(concurrent(from)),
                };
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DataType;

    use super::*;
    use crate::store::InMemoryCatalogStore;

    struct Fixture {
        catalog: StoreCatalog,
        warehouse: tempfile::TempDir,
    }

    fn fixture() -> Fixture {
        let warehouse = tempfile::tempdir().unwrap();
        let store: Arc<dyn CatalogStore> = Arc::new(InMemoryCatalogStore::default());
        let catalog = StoreCatalog::new(store, warehouse.path().to_path_buf(), false);
        Fixture { catalog, warehouse }
    }

    fn ns(name: &str) -> Vec<String> {
        vec![name.to_owned()]
    }

    fn ident(namespace: &str, name: &str) -> TableIdent {
        TableIdent {
            catalog: CATALOG.to_owned(),
            namespace: ns(namespace),
            name: name.to_owned(),
        }
    }

    fn info(namespace: &str, name: &str) -> TableInfo {
        let mut info = TableInfo::new(
            ident(namespace, name),
            vec![
                TableColumn::new("a", DataType::Integer, true),
                TableColumn::new("p", DataType::Integer, true),
            ],
        );
        info.partition_columns = vec!["p".to_owned()];
        info
    }

    fn form() -> NameForm {
        NameForm {
            case_sensitive: false,
            qualified: true,
            temporary_only: false,
        }
    }

    fn load(c: &StoreCatalog, id: &TableIdent) -> Option<TableMeta> {
        block_on(c.load_table(id, None, form())).unwrap()
    }

    fn class(e: &KrishivError) -> ErrorClass {
        e.error_class()
    }

    fn add_column(name: &str) -> TableChange {
        TableChange::AddColumn {
            path: ns(name),
            data_type: DataType::Long,
            declared: None,
            nullable: true,
            comment: Some("new".into()),
            position: None,
            default: None,
        }
    }

    #[test]
    fn the_write_capability_agrees_with_the_write_half() {
        let f = fixture();
        assert_eq!(
            f.catalog.capabilities().writes,
            f.catalog.as_write().is_some()
        );
        assert!(f.catalog.single_part_namespace());
        assert_eq!(f.catalog.name(), "spark_catalog");
    }

    #[test]
    fn namespaces_create_alter_and_drop() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let again = block_on(c.create_namespace(&ns("DB"), NamespaceInfo::default())).unwrap_err();
        assert_eq!(class(&again), ErrorClass::SchemaAlreadyExists);
        block_on(c.alter_namespace(
            &ns("db"),
            &[NamespaceChange::SetProperty {
                key: "k".into(),
                value: "v".into(),
            }],
        ))
        .unwrap();
        let missing = block_on(c.alter_namespace(&ns("nope"), &[])).unwrap_err();
        assert_eq!(class(&missing), ErrorClass::SchemaNotFound);
        assert!(
            block_on(c.create_namespace(&["a".into(), "b".into()], NamespaceInfo::default()))
                .is_err()
        );

        assert!(block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap());
        let not_empty = block_on(c.drop_namespace(&ns("db"), false)).unwrap_err();
        assert_eq!(class(&not_empty), ErrorClass::SchemaNotEmpty);
        block_on(c.drop_namespace(&ns("db"), true)).unwrap();
        let gone = block_on(c.drop_namespace(&ns("db"), true)).unwrap_err();
        assert_eq!(class(&gone), ErrorClass::SchemaNotFound);
        assert!(load(c, &ident("db", "t")).is_none());
    }

    #[test]
    fn a_table_in_a_schema_with_a_default_collation_is_refused() {
        let f = fixture();
        let c = &f.catalog;
        let meta = NamespaceMeta {
            collation: Some("UTF8_LCASE".to_owned()),
            ..NamespaceMeta::default()
        };
        assert!(c.store.create_namespace("coll", meta));
        let err =
            block_on(c.create_table(info("coll", "t"), CreateMode::ErrorIfExists)).unwrap_err();
        assert_eq!(
            err.error_class().name(),
            "UNSUPPORTED_FEATURE.CATALOG_OPERATION"
        );
        assert!(c.store.table("coll", "t").is_none());
        assert!(
            !managed_location(f.warehouse.path(), "coll", "t").exists(),
            "no directory is made for a refused table"
        );
    }

    #[test]
    fn a_created_table_loads_with_its_columns_a_version_and_a_managed_location() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let epoch = c.epoch();
        assert!(block_on(c.create_table(info("db", "T"), CreateMode::ErrorIfExists)).unwrap());
        assert!(c.epoch() > epoch, "a change bumps the epoch");
        let meta = load(c, &ident("db", "t")).unwrap();
        assert_eq!(meta.ident, ident("db", "t"));
        assert_eq!(meta.columns.len(), 2);
        assert!(meta.version.is_some());
        let table = c.store.table("db", "t").unwrap();
        assert!(table.managed);
        assert_eq!(table.layout.partition_columns, ["p"]);
        assert!(table.location.starts_with(f.warehouse.path()));

        let dup = block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap_err();
        assert_eq!(class(&dup), ErrorClass::TableOrViewAlreadyExists);
        assert!(!block_on(c.create_table(info("db", "t"), CreateMode::IgnoreIfExists)).unwrap());
        let no_ns =
            block_on(c.create_table(info("nope", "t"), CreateMode::ErrorIfExists)).unwrap_err();
        assert_eq!(class(&no_ns), ErrorClass::SchemaNotFound);
    }

    #[test]
    fn alter_table_applies_the_changes_and_moves_the_version() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let before = load(c, &id).unwrap();
        let epoch = c.epoch();
        let after = block_on(c.alter_table(
            &id,
            before.version.unwrap(),
            &[
                add_column("b"),
                TableChange::SetProperty {
                    key: "k".into(),
                    value: "v".into(),
                },
            ],
        ))
        .unwrap();
        let names: Vec<_> = after.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            ["a", "b", "p"],
            "a new column goes before the partition columns"
        );
        assert_eq!(after.column_comments["b"], "new");
        assert_ne!(after.version, before.version);
        assert!(c.epoch() > epoch);
        assert_eq!(c.store.table("db", "t").unwrap().properties["k"], "v");

        // A caller holding the old version is refused, not overwritten.
        let stale =
            block_on(c.alter_table(&id, before.version.unwrap(), &[add_column("c")])).unwrap_err();
        assert_eq!(class(&stale), ErrorClass::InternalError);
        assert!(
            stale.to_string().contains("changed by another statement"),
            "{stale}"
        );
        assert_eq!(load(c, &id).unwrap().columns.len(), 3);
    }

    #[test]
    fn a_failing_change_list_leaves_the_table_unchanged() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let before = load(c, &id).unwrap();
        let epoch = c.epoch();
        let err = block_on(c.alter_table(
            &id,
            before.version.unwrap(),
            &[add_column("b"), add_column("a")],
        ))
        .unwrap_err();
        assert_eq!(class(&err), ErrorClass::ColumnAlreadyExists);
        assert_eq!(
            load(c, &id).unwrap(),
            before,
            "not even the first change stayed"
        );
        assert_eq!(c.epoch(), epoch);

        // A file-source table refuses what Spark's session catalog refuses.
        for (change, op) in [
            (
                TableChange::RenameColumn {
                    path: ns("a"),
                    new_name: "x".into(),
                },
                "RENAME COLUMN",
            ),
            (
                TableChange::DeleteColumn {
                    path: ns("a"),
                    if_exists: false,
                },
                "DROP COLUMN",
            ),
        ] {
            let e = block_on(c.alter_table(&id, before.version.unwrap(), &[change])).unwrap_err();
            assert_eq!(class(&e), ErrorClass::UnsupportedFeatureTableOperation);
            assert!(e.to_string().contains(op), "{e}");
        }
        let widen = TableChange::UpdateColumnType {
            path: ns("a"),
            data_type: DataType::Long,
            declared: None,
        };
        let e = block_on(c.alter_table(&id, before.version.unwrap(), &[widen])).unwrap_err();
        assert_eq!(class(&e), ErrorClass::NotSupportedChangeColumn);
    }

    #[test]
    fn a_table_dropped_and_created_again_is_not_the_one_a_caller_read() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let read = load(c, &id).unwrap().version.unwrap();
        assert!(block_on(c.drop_table(&id, false)).unwrap());
        assert!(!block_on(c.drop_table(&id, false)).unwrap(), "already gone");
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        assert_ne!(load(c, &id).unwrap().version.unwrap(), read);
        let e = block_on(c.alter_table(&id, read, &[add_column("b")])).unwrap_err();
        assert_eq!(class(&e), ErrorClass::InternalError);
        let missing = block_on(c.alter_table(&ident("db", "nope"), read, &[])).unwrap_err();
        assert_eq!(class(&missing), ErrorClass::TableOrViewNotFound);
    }

    #[test]
    fn a_managed_tables_files_go_with_it_and_an_external_ones_stay() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "m"), CreateMode::ErrorIfExists)).unwrap();
        let managed = c.store.table("db", "m").unwrap().location.clone();
        assert!(
            managed.is_dir(),
            "a managed table's directory is made with it"
        );
        std::fs::write(managed.join("part.parquet"), b"x").unwrap();
        // There is no trash: `purge` or not, the data goes.
        assert!(block_on(c.drop_table(&ident("db", "m"), false)).unwrap());
        assert!(!managed.exists());

        let outside = tempfile::tempdir().unwrap();
        let mut external = info("db", "ext");
        external.location = Some(outside.path().display().to_string());
        block_on(c.create_table(external, CreateMode::ErrorIfExists)).unwrap();
        assert!(!c.store.table("db", "ext").unwrap().managed);
        std::fs::write(outside.path().join("f"), b"x").unwrap();
        assert!(block_on(c.drop_table(&ident("db", "ext"), true)).unwrap());
        assert!(
            outside.path().join("f").exists(),
            "an external table's data is never deleted"
        );
    }

    #[test]
    fn a_name_that_could_leave_the_warehouse_is_refused_at_create() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        for bad in ["..", ".", "", "a/b", "a b", "..%2f"] {
            let e =
                block_on(c.create_table(info("db", bad), CreateMode::ErrorIfExists)).unwrap_err();
            assert!(
                matches!(
                    class(&e),
                    ErrorClass::InvalidAttributeNameSyntax
                        | ErrorClass::InvalidSchemaOrRelationName
                ),
                "{bad:?}: {e}"
            );
            let e = block_on(c.create_namespace(&ns(bad), NamespaceInfo::default())).unwrap_err();
            assert!(
                matches!(
                    class(&e),
                    ErrorClass::InvalidAttributeNameSyntax
                        | ErrorClass::InvalidSchemaOrRelationName
                ),
                "{bad:?}: {e}"
            );
        }
        assert!(c.store.tables_in("db").is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn a_delete_through_a_link_or_outside_the_root_is_refused_and_changes_nothing() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("keep"), b"x").unwrap();

        // The table's directory is replaced by a link to somewhere else.
        block_on(c.create_table(info("db", "linked"), CreateMode::ErrorIfExists)).unwrap();
        let dir = c.store.table("db", "linked").unwrap().location.clone();
        std::fs::remove_dir_all(&dir).unwrap();
        std::os::unix::fs::symlink(outside.path(), &dir).unwrap();
        let e = block_on(c.drop_table(&ident("db", "linked"), true)).unwrap_err();
        assert_eq!(class(&e), ErrorClass::InternalError);
        assert!(e.to_string().contains("refusing to delete"), "{e}");
        assert!(c.store.table("db", "linked").is_some(), "the table stays");
        assert!(outside.path().join("keep").exists());

        // So does dropping the namespace: nothing changes, not even the tables before the bad one.
        block_on(c.create_table(info("db", "fine"), CreateMode::ErrorIfExists)).unwrap();
        let fine = c.store.table("db", "fine").unwrap().location.clone();
        let e = block_on(c.drop_namespace(&ns("db"), true)).unwrap_err();
        assert_eq!(class(&e), ErrorClass::InternalError);
        assert!(c.store.has_namespace("db") && fine.is_dir());
        assert!(outside.path().join("keep").exists());

        // A table whose directory is outside the root it was created under.
        std::fs::remove_file(&dir).unwrap();
        assert!(block_on(c.drop_table(&ident("db", "linked"), true)).unwrap());
        let mut stray = (*c.store.table("db", "fine").unwrap()).clone();
        stray.location = outside.path().to_path_buf();
        let current = c.store.table("db", "fine").unwrap();
        assert!(c.store.replace_table(&current, stray));
        let e = block_on(c.drop_table(&ident("db", "fine"), true)).unwrap_err();
        assert!(e.to_string().contains("refusing to delete"), "{e}");
        assert!(outside.path().join("keep").exists());
    }

    #[test]
    fn create_checks_what_create_table_checks() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let create = |i: TableInfo| block_on(c.create_table(i, CreateMode::ErrorIfExists));

        let mut bad_provider = info("db", "p");
        bad_provider
            .properties
            .insert("provider".into(), "nope".into());
        assert_eq!(
            class(&create(bad_provider).unwrap_err()),
            ErrorClass::DataSourceNotFound
        );

        let mut empty = info("db", "e");
        empty.location = Some(String::new());
        assert_eq!(
            class(&create(empty).unwrap_err()),
            ErrorClass::UnsupportedFeatureCatalogOperation
        );

        let mut dup = info("db", "d");
        dup.columns
            .push(TableColumn::new("A", DataType::Long, true));
        assert_eq!(
            class(&create(dup).unwrap_err()),
            ErrorClass::ColumnAlreadyExists
        );

        let mut missing = info("db", "m");
        missing.partition_columns = vec!["nope".into()];
        assert_eq!(
            class(&create(missing).unwrap_err()),
            ErrorClass::ColumnNotDefinedInTable
        );

        // Partition-layout rules of CREATE TABLE: not every column, only partitionable types.
        let mut all = info("db", "all");
        all.partition_columns = vec!["a".into(), "p".into()];
        assert_eq!(
            class(&create(all).unwrap_err()),
            ErrorClass::AllPartitionColumnsNotAllowed
        );
        let mut complex = TableInfo::new(
            ident("db", "cx"),
            vec![
                TableColumn::new("a", DataType::Integer, true),
                TableColumn::new(
                    "arr",
                    DataType::Array(Arc::new(krishiv_types::ArrayType::new(
                        DataType::Integer,
                        true,
                    ))),
                    true,
                ),
            ],
        );
        complex.partition_columns = vec!["arr".into()];
        assert_eq!(
            class(&create(complex).unwrap_err()),
            ErrorClass::InvalidPartitionColumnDataType
        );

        // A schema the provider cannot write fails before anything is published.
        let interval = TableInfo::new(
            ident("db", "iv"),
            vec![TableColumn::new("i", DataType::CalendarInterval, true)],
        );
        assert!(create(interval).is_err());
        for refused in ["m", "all", "cx", "iv"] {
            assert!(c.store.table("db", refused).is_none(), "{refused}");
            assert!(!managed_location(f.warehouse.path(), "db", refused).exists());
        }

        // A managed directory that exists already is not taken over.
        let taken = managed_location(f.warehouse.path(), "db", "t");
        std::fs::create_dir_all(&taken).unwrap();
        assert_eq!(
            class(&create(info("db", "t")).unwrap_err()),
            ErrorClass::LocationAlreadyExists
        );
        assert!(c.store.table("db", "t").is_none());

        // The partition columns go last, in partition order, spelled as the table spells them.
        let mut info3 = TableInfo::new(
            ident("db", "x"),
            vec![
                TableColumn::new("P2", DataType::Integer, true),
                TableColumn::new("a", DataType::Integer, true),
                TableColumn::new("p1", DataType::Integer, true),
                TableColumn::new("b", DataType::Integer, true),
            ],
        );
        info3.partition_columns = vec!["p1".into(), "p2".into()];
        assert!(create(info3).unwrap());
        let t = c.store.table("db", "x").unwrap();
        let names: Vec<_> = t.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "p1", "P2"]);
        assert_eq!(t.layout.partition_columns, ["p1", "P2"]);

        let mut csv = info("db", "csv");
        csv.properties.insert("provider".into(), "CSV".into());
        csv.comment = Some("hello".into());
        create(csv).unwrap();
        let t = c.store.table("db", "csv").unwrap();
        assert_eq!(
            (t.provider.as_str(), t.provider_written.as_str()),
            ("csv", "CSV")
        );
        assert!(!t.properties.contains_key("provider"));
        assert_eq!(t.comment.as_deref(), Some("hello"));
    }

    #[test]
    fn a_temporary_only_lookup_finds_no_stored_table() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        assert!(load(c, &id).is_some());
        let temporary_only = NameForm {
            temporary_only: true,
            ..form()
        };
        assert!(
            block_on(c.load_table(&id, None, temporary_only))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn an_existing_table_settles_a_create_before_its_definition_is_looked_at() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let mut broken = info("db", "t");
        broken.properties.insert("provider".into(), "nope".into());
        broken.partition_columns = vec!["nope".into()];
        broken.location = Some(String::new());
        // IF NOT EXISTS is a no-op whatever the definition says.
        assert!(!block_on(c.create_table(broken.clone(), CreateMode::IgnoreIfExists)).unwrap());
        let e = block_on(c.create_table(broken, CreateMode::ErrorIfExists)).unwrap_err();
        assert_eq!(class(&e), ErrorClass::TableOrViewAlreadyExists);
        assert!(e.to_string().contains("`spark_catalog`.`db`.`t`"), "{e}");
        // A missing table does validate it.
        let mut fresh = info("db", "u");
        fresh.properties.insert("provider".into(), "nope".into());
        let e = block_on(c.create_table(fresh, CreateMode::IgnoreIfExists)).unwrap_err();
        assert_eq!(class(&e), ErrorClass::DataSourceNotFound);
    }

    #[test]
    fn alter_returns_the_entry_it_committed_not_a_later_ones() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let v1 = load(c, &id).unwrap().version.unwrap();
        let after = block_on(c.alter_table(&id, v1, &[add_column("b")])).unwrap();
        // The returned version is the committed entry's stamp, so it guards the next change.
        assert_eq!(
            after.version,
            Some(c.store.table("db", "t").unwrap().version())
        );
        let again =
            block_on(c.alter_table(&id, after.version.unwrap(), &[add_column("c")])).unwrap();
        assert_ne!(again.version, after.version);
        // The store hands back the stored entry with its stamp, under the lock.
        let current = c.store.table("db", "t").unwrap();
        let stored = c
            .store
            .replace_table_entry(&current, (*current).clone())
            .unwrap();
        assert_ne!(stored.stamp, current.stamp);
        assert!(Arc::ptr_eq(&stored, &c.store.table("db", "t").unwrap()));
        assert!(
            c.store
                .replace_table_entry(&current, (*current).clone())
                .is_none()
        );
    }

    #[test]
    fn file_source_columns_are_nullable_whatever_the_definition_says() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let mut strict = info("db", "t");
        strict.columns[0].nullable = false;
        block_on(c.create_table(strict, CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let before = load(c, &id).unwrap();
        assert!(
            before.columns.iter().all(|c| c.nullable),
            "{:?}",
            before.columns
        );
        let not_null = TableChange::AddColumn {
            path: ns("b"),
            data_type: DataType::Long,
            declared: None,
            nullable: false,
            comment: None,
            position: None,
            default: None,
        };
        let after = block_on(c.alter_table(&id, before.version.unwrap(), &[not_null])).unwrap();
        assert!(
            after.columns.iter().all(|c| c.nullable),
            "{:?}",
            after.columns
        );
    }

    #[test]
    fn a_cascading_namespace_drop_invalidates_the_caches_of_its_tables() {
        let results = Arc::new(crate::result_cache::ResultCache::default());
        let hub = crate::cache::Invalidation::new(Arc::clone(&results));
        let store: Arc<dyn CatalogStore> = Arc::new(InMemoryCatalogStore::with_hub(hub));
        let warehouse = tempfile::tempdir().unwrap();
        let c = StoreCatalog::new(store, warehouse.path().to_path_buf(), false);
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let before = results.generation();
        block_on(c.drop_namespace(&ns("db"), true)).unwrap();
        assert!(
            results.generation() > before,
            "the dropped table's caches are invalidated"
        );
    }

    #[test]
    fn the_directory_exists_when_the_table_does_and_a_refused_create_leaves_none() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let dir = managed_location(f.warehouse.path(), "db", "t");
        assert!(dir.is_dir());
        // A second create of the name is refused before it touches the directory.
        std::fs::write(dir.join("keep"), b"x").unwrap();
        assert!(block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).is_err());
        assert!(!block_on(c.create_table(info("db", "t"), CreateMode::IgnoreIfExists)).unwrap());
        assert!(dir.join("keep").exists());
    }

    #[test]
    fn a_cascading_namespace_drop_deletes_exactly_the_directories_it_checked() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "m"), CreateMode::ErrorIfExists)).unwrap();
        block_on(c.create_table(info("db", "gone"), CreateMode::ErrorIfExists)).unwrap();
        let outside = tempfile::tempdir().unwrap();
        let mut external = info("db", "ext");
        external.location = Some(outside.path().display().to_string());
        block_on(c.create_table(external, CreateMode::ErrorIfExists)).unwrap();
        std::fs::write(outside.path().join("f"), b"x").unwrap();
        let managed = managed_location(f.warehouse.path(), "db", "m");
        std::fs::write(managed.join("part"), b"x").unwrap();
        // One managed table's directory is already gone: nothing is looked up for it afterwards.
        let gone = managed_location(f.warehouse.path(), "db", "gone");
        std::fs::remove_dir_all(&gone).unwrap();

        block_on(c.drop_namespace(&ns("db"), true)).unwrap();
        assert!(!c.store.has_namespace("db"));
        assert!(
            !managed.exists(),
            "a managed table's files go with the namespace"
        );
        assert!(
            outside.path().join("f").exists(),
            "an external table's never"
        );
        assert!(!gone.exists());
        // A directory made at a dropped table's path after the checks is not deleted by this call: the
        // set to delete was fixed first (here, a namespace dropped without cascade deletes nothing).
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "n"), CreateMode::ErrorIfExists)).unwrap();
        let kept = managed_location(f.warehouse.path(), "db", "n");
        let e = block_on(c.drop_namespace(&ns("db"), false)).unwrap_err();
        assert_eq!(class(&e), ErrorClass::SchemaNotEmpty);
        assert!(kept.is_dir() && c.store.table("db", "n").is_some());
    }

    #[test]
    fn global_temp_is_not_a_permanent_namespace() {
        let f = fixture();
        let e = block_on(
            f.catalog
                .create_namespace(&ns("Global_Temp"), NamespaceInfo::default()),
        )
        .unwrap_err();
        assert_eq!(class(&e), ErrorClass::SchemaAlreadyExists);
        assert!(!f.catalog.store.has_namespace("global_temp"));
    }

    #[test]
    fn an_added_column_the_provider_cannot_write_is_refused_before_the_table_changes() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let before = load(c, &id).unwrap();
        let interval = TableChange::AddColumn {
            path: ns("iv"),
            data_type: DataType::CalendarInterval,
            declared: None,
            nullable: true,
            comment: None,
            position: None,
            default: None,
        };
        let altered =
            block_on(c.alter_table(&id, before.version.unwrap(), &[interval])).unwrap_err();
        // The same refusal create gives for the same column.
        let created = block_on(c.create_table(
            TableInfo::new(
                ident("db", "iv"),
                vec![TableColumn::new("iv", DataType::CalendarInterval, true)],
            ),
            CreateMode::ErrorIfExists,
        ))
        .unwrap_err();
        assert_eq!(class(&altered), class(&created));
        assert_eq!(load(c, &id).unwrap(), before, "the table did not change");
    }

    #[test]
    fn a_drop_never_removes_a_table_created_after_the_one_it_checked() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let old = c.store.table("db", "t").unwrap();
        // The table is dropped and created again between the checks and the removal.
        assert!(block_on(c.drop_table(&id, true)).unwrap());
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        assert!(c.store.drop_table_entry(&old).is_none());
        assert!(load(c, &id).is_some(), "the new table stays");
        let dir = managed_location(f.warehouse.path(), "db", "t");
        assert!(dir.is_dir(), "and so do its files");
    }

    #[test]
    fn a_managed_directory_that_is_there_is_not_taken_over() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let taken = managed_location(f.warehouse.path(), "db", "t");
        std::fs::create_dir_all(&taken).unwrap();
        std::fs::write(taken.join("other"), b"x").unwrap();
        let e = block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap_err();
        assert_eq!(class(&e), ErrorClass::LocationAlreadyExists);
        assert!(
            taken.join("other").exists(),
            "another statement's files are left alone"
        );
        assert!(c.store.table("db", "t").is_none());
    }

    #[test]
    fn a_view_is_not_a_table_to_alter_or_drop() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "v"), CreateMode::ErrorIfExists)).unwrap();
        let mut view = (*c.store.table("db", "v").unwrap()).clone();
        view.view = Some(crate::store::ViewDef {
            sql: "SELECT 1".into(),
            namespace: "db".into(),
            path: Vec::new(),
            query_columns: Vec::new(),
            identifier_spans: Vec::new(),
            written: "SELECT 1".into(),
            schema_mode: None,
        });
        let current = c.store.table("db", "v").unwrap();
        assert!(c.store.replace_table(&current, view));
        let id = ident("db", "v");
        let version = load(c, &id).unwrap().version.unwrap();
        for e in [
            block_on(c.drop_table(&id, true)).unwrap_err(),
            block_on(c.alter_table(&id, version, &[])).unwrap_err(),
        ] {
            assert_eq!(class(&e), ErrorClass::UnsupportedFeatureCatalogOperation);
        }
        assert!(c.store.table("db", "v").is_some());
    }

    #[test]
    fn the_comment_property_is_the_table_comment_and_a_load_at_another_version_finds_nothing() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "t"), CreateMode::ErrorIfExists)).unwrap();
        let id = ident("db", "t");
        let v1 = load(c, &id).unwrap().version.unwrap();
        block_on(c.alter_table(
            &id,
            v1,
            &[TableChange::SetProperty {
                key: "Comment".into(),
                value: "hi".into(),
            }],
        ))
        .unwrap();
        let t = c.store.table("db", "t").unwrap();
        assert_eq!(t.comment.as_deref(), Some("hi"));
        assert!(!t.properties.contains_key("Comment"));
        let v2 = t.version();
        assert_ne!(v1, v2);
        assert!(
            block_on(c.load_table(&id, Some(v1), form()))
                .unwrap()
                .is_none()
        );
        assert!(
            block_on(c.load_table(&id, Some(v2), form()))
                .unwrap()
                .is_some()
        );
        block_on(c.alter_table(
            &id,
            v2,
            &[TableChange::RemoveProperty {
                key: "comment".into(),
            }],
        ))
        .unwrap();
        assert!(c.store.table("db", "t").unwrap().comment.is_none());
    }

    /// The tests every catalog shares (`krishiv_plan::analyzer::contract`): this one is the first to meet them.
    #[test]
    fn the_store_catalog_meets_the_write_contract() {
        let f = fixture();
        block_on(krishiv_plan::analyzer::contract::write_contract(&f.catalog)).unwrap();
        block_on(krishiv_plan::analyzer::contract::capabilities_agree(
            &f.catalog,
        ))
        .unwrap();
    }

    /// A catalog that is not what it says: the contract has to be able to tell.
    struct Wrapped<'c> {
        inner: &'c StoreCatalog,
        /// Reports the same epoch whatever happens.
        frozen_epoch: bool,
        /// Declares `writes: false` while offering the write half.
        liar: bool,
    }

    impl Catalog for Wrapped<'_> {
        fn name(&self) -> &str {
            self.inner.name()
        }
        fn epoch(&self) -> u64 {
            if self.frozen_epoch {
                0
            } else {
                self.inner.epoch()
            }
        }
        fn capabilities(&self) -> CatalogCapabilities {
            let mut caps = self.inner.capabilities();
            if self.liar {
                caps = CatalogCapabilities::default();
            }
            caps
        }
        fn as_write(&self) -> Option<&dyn CatalogWrite> {
            self.inner.as_write()
        }
        fn load_table<'a>(
            &'a self,
            id: &'a TableIdent,
            version: Option<TableVersion>,
            form: NameForm,
        ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
            self.inner.load_table(id, version, form)
        }
    }

    #[test]
    fn the_contract_tells_a_catalog_that_is_not_what_it_says() {
        use krishiv_plan::analyzer::contract::{capabilities_agree, write_contract};
        let f = fixture();
        let liar = Wrapped {
            inner: &f.catalog,
            frozen_epoch: false,
            liar: true,
        };
        let why = block_on(capabilities_agree(&liar)).unwrap_err();
        assert!(
            why.contains("writes is false but as_write() is Some"),
            "{why}"
        );

        let f = fixture();
        let frozen = Wrapped {
            inner: &f.catalog,
            frozen_epoch: true,
            liar: false,
        };
        let why = block_on(write_contract(&frozen)).unwrap_err();
        assert!(
            why.starts_with("namespaces:") && why.contains("epoch"),
            "{why}"
        );
    }

    fn rename(c: &StoreCatalog, from: &str, to: &str) -> Result<(), KrishivError> {
        block_on(c.rename_table(&ident("db", from), &ident("db", to)))
    }

    #[test]
    fn a_renamed_managed_table_keeps_its_files_columns_partitions_and_comments() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let mut commented = info("db", "old");
        commented.comment = Some("hello".into());
        block_on(c.create_table(commented, CreateMode::ErrorIfExists)).unwrap();
        let from_dir = managed_location(f.warehouse.path(), "db", "old");
        let partition = from_dir.join("p=1");
        std::fs::create_dir_all(&partition).unwrap();
        std::fs::write(partition.join("part-0.parquet"), b"x").unwrap();
        // The table tracks the partition it has, under its own directory.
        let current = c.store.table("db", "old").unwrap();
        let mut with_partition = (*current).clone();
        with_partition
            .layout
            .partitions
            .push(crate::store::CatalogPartition {
                spec: vec![("p".to_owned(), "1".to_owned())],
                location: partition,
                parameters: BTreeMap::new(),
                created_ms: 0,
                stats: None,
            });
        assert!(c.store.replace_table(&current, with_partition));
        let before = load(c, &ident("db", "old")).unwrap();
        let epoch = c.epoch();

        rename(c, "OLD", "new").unwrap();

        assert!(c.epoch() > epoch, "a rename bumps the epoch");
        assert!(
            load(c, &ident("db", "old")).is_none(),
            "the old name is gone"
        );
        let after = load(c, &ident("db", "new")).unwrap();
        assert_eq!(after.columns, before.columns, "the same columns");
        assert_ne!(
            after.version, before.version,
            "a renamed entry is a new version"
        );
        let to_dir = managed_location(f.warehouse.path(), "db", "new");
        assert!(!from_dir.exists());
        assert!(
            to_dir.join("p=1").join("part-0.parquet").exists(),
            "the files moved"
        );
        let moved = c.store.table("db", "new").unwrap();
        assert_eq!(moved.location, to_dir);
        assert_eq!(moved.layout.partitions[0].location, to_dir.join("p=1"));
        assert_eq!(moved.layout.partition_columns, ["p"]);
        assert_eq!(moved.comment.as_deref(), Some("hello"));
    }

    #[test]
    fn an_external_tables_data_stays_where_it_is() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        let outside = tempfile::tempdir().unwrap();
        let mut external = info("db", "ext");
        external.location = Some(outside.path().display().to_string());
        block_on(c.create_table(external, CreateMode::ErrorIfExists)).unwrap();
        std::fs::write(outside.path().join("f"), b"x").unwrap();
        rename(c, "ext", "moved").unwrap();
        assert!(outside.path().join("f").exists());
        let t = c.store.table("db", "moved").unwrap();
        assert!(!t.managed);
        assert_eq!(t.location, outside.path());
    }

    #[test]
    fn a_rename_the_session_catalog_refuses_is_refused_with_its_class() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_namespace(&ns("other"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "a"), CreateMode::ErrorIfExists)).unwrap();
        block_on(c.create_table(info("db", "b"), CreateMode::ErrorIfExists)).unwrap();
        let a_dir = managed_location(f.warehouse.path(), "db", "a");
        let epoch = c.epoch();

        let exists = rename(c, "a", "b").unwrap_err();
        assert_eq!(class(&exists), ErrorClass::TableOrViewAlreadyExists);
        let missing = rename(c, "nope", "z").unwrap_err();
        assert_eq!(class(&missing), ErrorClass::TableOrViewNotFound);
        for bad in ["..", "a b", ""] {
            let e = rename(c, "a", bad).unwrap_err();
            assert!(
                matches!(
                    class(&e),
                    ErrorClass::InvalidAttributeNameSyntax
                        | ErrorClass::InvalidSchemaOrRelationName
                ),
                "{bad:?}: {e}"
            );
        }
        // A table never changes namespace (Spark's session catalog, `_LEGACY_ERROR_TEMP_1073`).
        let across = block_on(c.rename_table(&ident("db", "a"), &ident("other", "a"))).unwrap_err();
        assert_eq!(class(&across), ErrorClass::LegacyErrorTemp1073);
        let no_ns = block_on(c.rename_table(&ident("db", "a"), &ident("nope", "a"))).unwrap_err();
        assert_eq!(class(&no_ns), ErrorClass::SchemaNotFound);
        assert!(!c.capabilities().rename_across_namespaces);

        assert_eq!(c.epoch(), epoch, "nothing changed");
        assert!(a_dir.is_dir() && c.store.table("db", "a").is_some());
    }

    /// The target directory is in the way: the move fails before any metadata changes and nothing is lost.
    #[test]
    fn a_rename_whose_files_cannot_move_changes_nothing() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "a"), CreateMode::ErrorIfExists)).unwrap();
        let from_dir = managed_location(f.warehouse.path(), "db", "a");
        std::fs::write(from_dir.join("f"), b"x").unwrap();
        let target = managed_location(f.warehouse.path(), "db", "z");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("other-statements-file"), b"x").unwrap();
        let before = load(c, &ident("db", "a")).unwrap();

        let e = rename(c, "a", "z").unwrap_err();

        assert_eq!(class(&e), ErrorClass::LocationAlreadyExists);
        assert!(
            from_dir.join("f").exists(),
            "the source files are where they were"
        );
        assert!(
            target.join("other-statements-file").exists(),
            "and the other directory is untouched"
        );
        assert_eq!(load(c, &ident("db", "a")).unwrap(), before);
        assert!(load(c, &ident("db", "z")).is_none());
    }

    /// An empty directory at the target is as much in the way as a full one: the move would replace it.
    #[test]
    fn an_empty_target_directory_is_not_replaced_by_a_rename() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "a"), CreateMode::ErrorIfExists)).unwrap();
        let target = managed_location(f.warehouse.path(), "db", "z");
        std::fs::create_dir_all(&target).unwrap();

        let e = rename(c, "a", "z").unwrap_err();

        assert_eq!(class(&e), ErrorClass::LocationAlreadyExists);
        assert!(target.is_dir());
        assert!(load(c, &ident("db", "a")).is_some() && load(c, &ident("db", "z")).is_none());
    }

    /// The name is taken between the check and the swap: the directory goes back, the source entry is untouched.
    #[test]
    fn a_name_taken_during_a_rename_puts_the_files_back() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "a"), CreateMode::ErrorIfExists)).unwrap();
        block_on(c.create_table(info("db", "z"), CreateMode::ErrorIfExists)).unwrap();
        let a = c.store.table("db", "a").unwrap();
        let mut renamed = (*a).clone();
        renamed.name = "z".to_owned();
        let epoch = c.epoch();

        let e = c.store.rename_table_entry(&a, renamed).unwrap_err();

        assert_eq!(class(&e), ErrorClass::TableOrViewAlreadyExists);
        assert_eq!(c.epoch(), epoch, "a refused swap changes nothing");
        assert!(Arc::ptr_eq(&c.store.table("db", "a").unwrap(), &a));
    }

    /// The source directory is gone (a managed table whose files were removed): the target is still not taken over.
    #[test]
    fn a_missing_source_directory_does_not_let_a_rename_take_over_the_target() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "a"), CreateMode::ErrorIfExists)).unwrap();
        std::fs::remove_dir_all(managed_location(f.warehouse.path(), "db", "a")).unwrap();
        let target = managed_location(f.warehouse.path(), "db", "z");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("other"), b"x").unwrap();

        let e = rename(c, "a", "z").unwrap_err();

        assert_eq!(class(&e), ErrorClass::LocationAlreadyExists);
        assert!(target.join("other").exists());
        assert!(load(c, &ident("db", "z")).is_none());
    }

    #[test]
    fn a_view_is_not_renamed_as_a_table_and_a_changed_table_is_a_retry() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "v"), CreateMode::ErrorIfExists)).unwrap();
        let current = c.store.table("db", "v").unwrap();
        let mut view = (*current).clone();
        view.view = Some(crate::store::ViewDef {
            sql: "SELECT 1".into(),
            namespace: "db".into(),
            path: Vec::new(),
            query_columns: Vec::new(),
            identifier_spans: Vec::new(),
            written: "SELECT 1".into(),
            schema_mode: None,
        });
        assert!(c.store.replace_table(&current, view));
        let e = rename(c, "v", "w").unwrap_err();
        assert_eq!(class(&e), ErrorClass::UnsupportedFeatureCatalogOperation);
        assert!(c.store.table("db", "v").is_some());
    }
}
