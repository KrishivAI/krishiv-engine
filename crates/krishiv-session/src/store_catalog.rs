// SPDX-License-Identifier: Apache-2.0
//! The session's in-memory store (`CatalogStore`) seen through the public
//! catalog traits (design P3-lakehouse §2.1, P3-T020a): `StoreCatalog` is a
//! `Catalog` and a `CatalogWrite`, so the shared contract suite and the DDL
//! commands that follow can talk to every catalog the same way. It changes no
//! existing caller; the store's own methods stay what the session uses until
//! the commands move onto the traits.
//!
//! A table's version is [`CatalogTable::version`]: the incarnation and the
//! metadata revision, so `alter_table` fails on a table that changed or was
//! dropped and created again since it was read, as `replace_table` always did.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the DDL commands move onto the catalog traits in the next commits of P3-T020a"
    )
)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use krishiv_common::error::{
    InternalError, KrishivError, SchemaAlreadyExists, SchemaNotEmpty, SchemaNotFound,
    TableOrViewNotFound, UnsupportedFeatureCatalogOperation,
};
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

/// The in-memory store as a catalog named `spark_catalog`.
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

    /// Deletes a managed table's directory, only inside the root it was created under and never through a link.
    fn remove_data(table: &CatalogTable) -> Result<(), KrishivError> {
        if !table.managed || table.view.is_some() {
            return Ok(());
        }
        let inside = table.location.starts_with(&table.root) && table.location != table.root;
        let linked = std::fs::symlink_metadata(&table.location)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        if !inside || linked {
            return Ok(());
        }
        match std::fs::remove_dir_all(&table.location) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(InternalError {
                message: format!("deleting {}: {e}", table.location.display()),
            }
            .into()),
        }
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
    InternalError {
        message: format!("{id} was changed by another statement while this one ran; retry"),
    }
    .into()
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

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
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

    fn drop_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        cascade: bool,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let name = self.namespace(namespace)?;
            match self.store.drop_namespace(&name, cascade) {
                Ok(tables) => tables.iter().try_for_each(|t| Self::remove_data(t)),
                Err(DropNamespaceError::Missing) => Err(SchemaNotFound {
                    schema_name: format!("`{name}`"),
                }
                .into()),
                Err(DropNamespaceError::NotEmpty) => Err(SchemaNotEmpty {
                    schema_name: format!("`{name}`"),
                }
                .into()),
                Err(DropNamespaceError::Default) => {
                    Err(self.unsupported("dropping the default namespace"))
                }
            }
        })
    }

    fn create_table<'a>(
        &'a self,
        info: TableInfo,
        mode: CreateMode,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table(&info.ident)?;
            if !self.store.has_namespace(&namespace) {
                return Err(SchemaNotFound {
                    schema_name: format!("`{namespace}`"),
                }
                .into());
            }
            let external = info.location.is_some();
            let location = info.location.as_deref().map_or_else(
                || managed_location(&self.warehouse, &namespace, &name),
                |l| PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            );
            let provider = info
                .properties
                .get("provider")
                .cloned()
                .unwrap_or_else(|| "parquet".to_owned());
            let columns = info
                .columns
                .iter()
                .map(|c| StoreColumn {
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    nullable: c.nullable,
                    default: None,
                    comment: None,
                    char_varchar: c.char_varchar.clone(),
                })
                .collect();
            let table = CatalogTable {
                namespace,
                name,
                managed: !external,
                provider_written: provider.clone(),
                provider,
                location,
                columns,
                options: BTreeMap::new(),
                properties: info.properties,
                comment: info.comment,
                created_ms: crate::table_commands::now_ms(),
                root: self.warehouse.clone(),
                view: None,
                layout: TableLayout {
                    partition_columns: info.partition_columns,
                    ..TableLayout::default()
                },
                column_stats: BTreeMap::new(),
                incarnation: 0,
                data_version: 0,
                revision: 0,
            };
            let mode = match mode {
                CreateMode::IgnoreIfExists => StoreCreateMode::IfNotExists,
                _ => StoreCreateMode::ErrorIfExists,
            };
            self.store.create_table(table, mode)
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
            let mut changed = (*current).clone();
            changed.columns = altered
                .columns
                .iter()
                .map(|c| StoreColumn {
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    nullable: c.nullable,
                    default: altered.defaults.get(&c.name).cloned(),
                    comment: altered.comments.get(&c.name).cloned(),
                    char_varchar: c.char_varchar.clone(),
                })
                .collect();
            for (key, value) in altered.properties {
                match value {
                    Some(v) => changed.properties.insert(key, v),
                    None => changed.properties.remove(&key),
                };
            }
            if !self.store.replace_table(&current, changed) {
                return Err(concurrent(id));
            }
            self.store
                .table(&table_id.namespace[0], &table_id.name)
                .map(|t| self.to_meta(&t))
                .ok_or_else(|| Self::not_found(id))
        })
    }

    fn drop_table<'a>(
        &'a self,
        id: &'a TableIdent,
        purge: bool,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table(id)?;
            match self.store.drop_table(&namespace, &name) {
                None => Ok(false),
                Some(table) => {
                    if purge {
                        Self::remove_data(&table)?;
                    }
                    Ok(true)
                }
            }
        })
    }

    fn rename_table<'a>(
        &'a self,
        from: &'a TableIdent,
        to: &'a TableIdent,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            // A managed table's directory moves with the rename, which `Session::rename_table` still does
            // (log first, then the metadata, rolled back on failure); it moves here with the fold.
            let _ = (from, to);
            Err(self.unsupported("renaming a table"))
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
    fn purge_deletes_a_managed_tables_files_and_an_external_ones_stay() {
        let f = fixture();
        let c = &f.catalog;
        block_on(c.create_namespace(&ns("db"), NamespaceInfo::default())).unwrap();
        block_on(c.create_table(info("db", "m"), CreateMode::ErrorIfExists)).unwrap();
        let managed = c.store.table("db", "m").unwrap().location.clone();
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(managed.join("part.parquet"), b"x").unwrap();
        // Without purge the data stays.
        let kept = ident("db", "keep");
        block_on(c.create_table(info("db", "keep"), CreateMode::ErrorIfExists)).unwrap();
        let kept_dir = c.store.table("db", "keep").unwrap().location.clone();
        std::fs::create_dir_all(&kept_dir).unwrap();
        assert!(block_on(c.drop_table(&kept, false)).unwrap());
        assert!(kept_dir.exists());
        assert!(block_on(c.drop_table(&ident("db", "m"), true)).unwrap());
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
    fn a_rename_is_refused_until_the_directory_move_is_folded_in() {
        let f = fixture();
        let e = block_on(f.catalog.rename_table(&ident("db", "a"), &ident("db", "b"))).unwrap_err();
        assert_eq!(class(&e), ErrorClass::UnsupportedFeatureCatalogOperation);
    }
}
