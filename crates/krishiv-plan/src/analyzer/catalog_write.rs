// SPDX-License-Identifier: Apache-2.0
//! The write half of a catalog (design P3-lakehouse §2.1): `CatalogWrite`,
//! and what its calls take. A read-only catalog (Unity in P3) has no write
//! half, so no method exists that could half-work.

use std::collections::BTreeMap;

use krishiv_common::error::KrishivError;

use super::{BoxFuture, Catalog, NamespaceChange, TableChange, TableColumn, TableIdent, TableMeta};
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

    /// Applies `changes` in order, all or none (the `alterTable` contract;
    /// [`apply_changes`](super::apply_changes) is the one application), and
    /// returns the table after them. `expected` is the version the analyzer
    /// pinned: a concurrent change fails the statement instead of being
    /// overwritten. A catalog whose tables have no format version gives each a
    /// version of its own (an incarnation and a change counter) in
    /// [`TableMeta::version`], so the guard holds for every catalog.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND` when the table is gone; the first change's
    /// error (see [`apply_changes`](super::apply_changes)); `INTERNAL_ERROR`
    /// ("was changed by another statement while this one ran; retry") when
    /// the table is no longer at `expected`, as the session raises today.
    fn alter_table<'a>(
        &'a self,
        id: &'a TableIdent,
        expected: TableVersion,
        changes: &'a [TableChange],
    ) -> BoxFuture<'a, Result<TableMeta, KrishivError>>;

    /// Drops the table and returns whether it existed. A managed table's
    /// data goes with it and an external table's never does; a catalog that
    /// keeps a trash may keep a managed table's data unless `purge`. The
    /// delete is checked before the metadata changes.
    ///
    /// # Errors
    /// A refused or failed delete; a view (it is not a table to drop).
    fn drop_table<'a>(
        &'a self,
        id: &'a TableIdent,
        purge: bool,
    ) -> BoxFuture<'a, Result<bool, KrishivError>>;

    /// Renames a table within the catalog. Across namespaces only when
    /// [`CatalogCapabilities::rename_across_namespaces`](super::CatalogCapabilities::rename_across_namespaces).
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
}
