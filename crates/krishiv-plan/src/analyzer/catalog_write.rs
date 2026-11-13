// SPDX-License-Identifier: Apache-2.0
//! The write half of a catalog (design P3-lakehouse §2.1): `CatalogWrite`,
//! and what its calls take. A read-only catalog (Unity in P3) has no write
//! half, so no method exists that could half-work.

use std::collections::BTreeMap;

use krishiv_common::error::KrishivError;

use super::{
    BoxFuture, Catalog, CatalogCapabilities, NamespaceChange, TableChange, TableColumn, TableIdent,
    TableMeta,
};
use crate::TableVersion;

/// What `CREATE TABLE` does when the table exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CreateMode {
    /// `TABLE_OR_VIEW_ALREADY_EXISTS`.
    ErrorIfExists,
    /// Nothing happens (`IF NOT EXISTS`).
    IgnoreIfExists,
}

/// A namespace's metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct NamespaceInfo {
    /// Its comment.
    pub comment: Option<String>,
    /// Its location.
    pub location: Option<String>,
    /// Its properties.
    pub properties: BTreeMap<String, String>,
}

/// What `create_table` needs to make an empty table (Spark's `TableInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TableInfo {
    /// The table.
    pub ident: TableIdent,
    /// Its columns.
    pub columns: Vec<TableColumn>,
    /// Its comment.
    pub comment: Option<String>,
    /// Its location; managed under the warehouse when absent.
    pub location: Option<String>,
    /// Its properties.
    pub properties: BTreeMap<String, String>,
    /// The partition columns, in order.
    pub partition_columns: Vec<String>,
}

impl TableInfo {
    /// A table of `columns` with nothing else set.
    #[must_use]
    pub fn new(ident: TableIdent, columns: Vec<TableColumn>) -> Self {
        Self {
            ident,
            columns,
            comment: None,
            location: None,
            properties: BTreeMap::new(),
            partition_columns: Vec::new(),
        }
    }
}

/// The calls that change a catalog. Every successful change bumps
/// [`Catalog::epoch`]; a call that fails changes nothing.
pub trait CatalogWrite: Catalog {
    /// Creates the namespace `namespace`.
    ///
    /// # Errors
    /// `SCHEMA_ALREADY_EXISTS` when it exists.
    fn create_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        info: NamespaceInfo,
    ) -> BoxFuture<'a, Result<(), KrishivError>>;

    /// Applies `changes` to the namespace's properties, all or none.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND` when it does not exist.
    fn alter_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        changes: &'a [NamespaceChange],
    ) -> BoxFuture<'a, Result<(), KrishivError>>;

    /// Drops the namespace; with `cascade` its tables too.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND`, or `SCHEMA_NOT_EMPTY` without `cascade`.
    fn drop_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        cascade: bool,
    ) -> BoxFuture<'a, Result<(), KrishivError>>;

    /// Creates an empty table (metadata only, as `CREATE TABLE` without
    /// `AS`). Returns whether it was created.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_ALREADY_EXISTS` under [`CreateMode::ErrorIfExists`];
    /// `SCHEMA_NOT_FOUND`.
    fn create_table<'a>(
        &'a self,
        info: TableInfo,
        mode: CreateMode,
    ) -> BoxFuture<'a, Result<bool, KrishivError>>;

    /// Applies `changes` in order, all or none (the `alterTable` contract),
    /// and returns the table after them. `expected` is the version the
    /// analyzer pinned: a concurrent change fails the statement instead of
    /// being overwritten.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND`; the first change's error (see
    /// [`apply_column_changes`](super::apply_column_changes)); a version
    /// that moved.
    fn alter_table<'a>(
        &'a self,
        id: &'a TableIdent,
        expected: Option<TableVersion>,
        changes: &'a [TableChange],
    ) -> BoxFuture<'a, Result<TableMeta, KrishivError>>;

    /// Drops the table; `purge` also deletes its data. Returns whether it
    /// existed.
    ///
    /// # Errors
    /// Errors deleting the data.
    fn drop_table<'a>(
        &'a self,
        id: &'a TableIdent,
        purge: bool,
    ) -> BoxFuture<'a, Result<bool, KrishivError>>;

    /// Renames a table within the catalog. Across namespaces only when
    /// [`CatalogCapabilities::rename_across_namespaces`].
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND`, `TABLE_OR_VIEW_ALREADY_EXISTS`,
    /// `CANNOT_RENAME_ACROSS_SCHEMA`.
    fn rename_table<'a>(
        &'a self,
        from: &'a TableIdent,
        to: &'a TableIdent,
    ) -> BoxFuture<'a, Result<(), KrishivError>>;
}

impl CatalogCapabilities {
    /// A catalog that writes tables and namespaces, renames within a
    /// namespace and alters columns (without widening or column ids): what
    /// the in-memory catalog declares.
    #[must_use]
    pub fn read_write() -> Self {
        Self {
            namespaces: true,
            alter_columns: true,
            writes: true,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_write_half_is_object_safe() {
        fn takes(_: Option<&dyn CatalogWrite>) {}
        takes(None);
    }

    #[test]
    fn table_info_starts_with_nothing_set() {
        let id = TableIdent {
            catalog: "c".into(),
            namespace: vec!["db".into()],
            name: "t".into(),
        };
        let info = TableInfo::new(id.clone(), Vec::new());
        assert_eq!(info.ident, id);
        assert!(info.properties.is_empty() && info.partition_columns.is_empty());
        assert!(info.comment.is_none() && info.location.is_none());
    }

    #[test]
    fn a_read_write_catalog_declares_writes_and_alters() {
        let caps = CatalogCapabilities::read_write();
        assert!(caps.writes && caps.namespaces && caps.alter_columns);
        assert!(!caps.staged_replace && !caps.type_widening && !caps.column_ids);
    }
}
