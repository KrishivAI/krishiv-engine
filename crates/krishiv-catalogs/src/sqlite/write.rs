// SPDX-License-Identifier: Apache-2.0
//! The write half of [`SqliteCatalog`]: each call is one transaction that bumps the epoch when it changes something.

use std::collections::BTreeMap;

use krishiv_common::error::{
    InternalError, InvalidAttributeNameSyntax, InvalidSchemaOrRelationName, KrishivError,
    LegacyErrorTemp1073, PartitionColumnNotFoundInSchema, SchemaAlreadyExists, SchemaNotEmpty,
    SchemaNotFound, TableOrViewAlreadyExists,
};
use krishiv_common::names::check_column_name_duplication;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    AlterInput, AlterRules, BoxFuture, CatalogWrite, CreateMode, NamespaceChange, NamespaceInfo,
    TableChange, TableColumn, TableIdent, TableInfo, TableMeta, apply_changes,
};
use krishiv_types::parse_data_type;

use super::store::{
    self, Failure, NamespaceRow, StoredColumn, TableRow, bump_epoch, read_namespace, read_table,
};
use super::{SqliteCatalog, quoted};

/// `INVALID_SCHEMA_OR_RELATION_NAME` (`INVALID_ATTRIBUTE_NAME_SYNTAX` for dots only) unless `name` is a valid new
/// namespace or table name: `[A-Za-z0-9_]+`, as the session catalog takes them.
fn check_new_name(name: &str) -> Result<(), KrishivError> {
    if !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        Ok(())
    } else if !name.is_empty() && name.chars().all(|c| c == '.') {
        Err(InvalidAttributeNameSyntax { name: quoted(name) }.into())
    } else {
        Err(InvalidSchemaOrRelationName { name: quoted(name) }.into())
    }
}

fn concurrent(id: &TableIdent) -> Failure {
    Failure::Krishiv(
        InternalError {
            message: format!("{id} was changed by another statement while this one ran; retry"),
        }
        .into(),
    )
}

fn schema_not_found(namespace: &str) -> Failure {
    Failure::Krishiv(
        SchemaNotFound {
            schema_name: quoted(namespace),
        }
        .into(),
    )
}

impl SqliteCatalog {
    fn table_key(&self, id: &TableIdent) -> Result<(String, String), KrishivError> {
        Ok((self.namespace(&id.namespace)?, self.fold(&id.name)))
    }

    /// The columns of a new table, checked as `CREATE TABLE` checks them (no repeated name, every type stored and
    /// read back as the same type), partition columns last in partition order and spelled as the table spells them.
    fn arranged(&self, info: &TableInfo) -> Result<(Vec<StoredColumn>, Vec<String>), KrishivError> {
        check_column_name_duplication(
            info.columns.iter().map(|c| c.name.as_str()),
            self.case_sensitive,
        )?;
        for c in &info.columns {
            let text = c.data_type.catalog_string();
            if !matches!(parse_data_type(&text), Ok(back) if back == c.data_type) {
                return Err(self.unsupported(&format!("a column of type {text}")));
            }
        }
        let same = |a: &str, b: &str| self.fold(a) == self.fold(b);
        let mut partitions: Vec<String> = Vec::new();
        for p in &info.partition_columns {
            let Some(column) = info.columns.iter().find(|c| same(&c.name, p)) else {
                return Err(PartitionColumnNotFoundInSchema {
                    column: quoted(p),
                    schema: format!(
                        "[{}]",
                        info.columns
                            .iter()
                            .map(|c| quoted(&c.name))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }
                .into());
            };
            if partitions.iter().any(|q| same(q, &column.name)) {
                return Err(self.unsupported("a repeated partition column"));
            }
            partitions.push(column.name.clone());
        }
        if !partitions.is_empty() && partitions.len() == info.columns.len() {
            return Err(self.unsupported("a table partitioned by every column"));
        }
        let stored = |c: &TableColumn| StoredColumn {
            column: c.clone(),
            comment: None,
            default: None,
        };
        let mut columns: Vec<StoredColumn> = info
            .columns
            .iter()
            .filter(|c| !partitions.iter().any(|p| same(p, &c.name)))
            .map(stored)
            .collect();
        for p in &partitions {
            if let Some(c) = info.columns.iter().find(|c| same(&c.name, p)) {
                columns.push(stored(c));
            }
        }
        Ok((columns, partitions))
    }
}

impl CatalogWrite for SqliteCatalog {
    fn create_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        info: NamespaceInfo,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let name = self.namespace(namespace)?;
            check_new_name(&name)?;
            self.change("", |c| {
                if read_namespace(c, &name)?.is_some() {
                    return Err(Failure::Krishiv(
                        SchemaAlreadyExists {
                            schema_name: quoted(&name),
                        }
                        .into(),
                    ));
                }
                bump_epoch(c)?;
                let row = NamespaceRow {
                    comment: info.comment,
                    location: info.location,
                    properties: info.properties,
                };
                store::write_namespace(c, &name, &row, true)?;
                Ok(((), true))
            })
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
            self.change("", |c| {
                let Some(mut row) = read_namespace(c, &name)? else {
                    return Err(schema_not_found(&name));
                };
                for change in changes {
                    match change {
                        NamespaceChange::SetProperty { key, value } => {
                            row.properties.insert(key.clone(), value.clone());
                        }
                        NamespaceChange::RemoveProperty { key } => {
                            row.properties.remove(key);
                        }
                        _ => {}
                    }
                }
                bump_epoch(c)?;
                store::write_namespace(c, &name, &row, false)?;
                Ok(((), true))
            })
        })
    }

    fn drop_namespace<'a>(
        &'a self,
        namespace: &'a [String],
        cascade: bool,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let name = self.namespace(namespace)?;
            if name == "default" {
                return Err(self.unsupported("dropping the default namespace"));
            }
            self.change("", |c| {
                if read_namespace(c, &name)?.is_none() {
                    return Err(schema_not_found(&name));
                }
                if !cascade && !store::table_names(c, &name)?.is_empty() {
                    return Err(Failure::Krishiv(
                        SchemaNotEmpty {
                            schema_name: quoted(&name),
                        }
                        .into(),
                    ));
                }
                bump_epoch(c)?;
                store::delete_tables_of(c, &name)?;
                store::delete_namespace(c, &name)?;
                Ok(((), true))
            })
        })
    }

    fn create_table<'a>(
        &'a self,
        info: TableInfo,
        mode: CreateMode,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        Box::pin(async move {
            let ignore = match mode {
                CreateMode::ErrorIfExists => false,
                CreateMode::IgnoreIfExists => true,
                _ => return Err(self.unsupported("this create mode")),
            };
            let (namespace, name) = self.table_key(&info.ident)?;
            check_new_name(&name)?;
            if info.location.as_deref().is_some_and(str::is_empty) {
                return Err(self.unsupported("an empty location"));
            }
            self.change(&name, |c| {
                if read_namespace(c, &namespace)?.is_none() {
                    return Err(schema_not_found(&namespace));
                }
                // A table that exists settles the call before its definition is looked at.
                if read_table(c, &namespace, &name)?.is_some() {
                    return if ignore {
                        Ok((false, false))
                    } else {
                        Err(Failure::Krishiv(
                            TableOrViewAlreadyExists {
                                relation_name: info.ident.to_string(),
                            }
                            .into(),
                        ))
                    };
                }
                let (columns, partition_columns) = self.arranged(&info)?;
                let mut properties = info.properties.clone();
                // Spark's `comment` property is the table's comment.
                let comment = info
                    .comment
                    .clone()
                    .or_else(|| properties.remove("comment"));
                let version = bump_epoch(c)?;
                let row = TableRow {
                    version,
                    comment,
                    location: info.location.clone(),
                    properties,
                    partition_columns,
                    columns,
                };
                store::write_table(c, &namespace, &name, &row, true)?;
                Ok((true, true))
            })
        })
    }

    fn alter_table<'a>(
        &'a self,
        id: &'a TableIdent,
        expected: TableVersion,
        changes: &'a [TableChange],
    ) -> BoxFuture<'a, Result<TableMeta, KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table_key(id)?;
            self.change(&name, |c| {
                let Some(current) = read_table(c, &namespace, &name)? else {
                    return Err(Failure::Krishiv(self.not_found(&namespace, &name)));
                };
                if TableVersion(current.version) != expected {
                    return Err(concurrent(id));
                }
                let columns: Vec<TableColumn> =
                    current.columns.iter().map(|s| s.column.clone()).collect();
                let pick = |f: fn(&StoredColumn) -> Option<&String>| -> BTreeMap<String, String> {
                    current
                        .columns
                        .iter()
                        .filter_map(|s| Some((s.column.name.clone(), f(s)?.clone())))
                        .collect()
                };
                let comments = pick(|s| s.comment.as_ref());
                let defaults = pick(|s| s.default.as_ref());
                let table_id = self.ident(&namespace, &name);
                let input = AlterInput {
                    table: &table_id,
                    columns: &columns,
                    comments: &comments,
                    defaults: &defaults,
                    partition_columns: &current.partition_columns,
                };
                let altered = apply_changes(&input, changes, AlterRules::new(self.case_sensitive))?;
                for column in &altered.columns {
                    let text = column.data_type.catalog_string();
                    if !matches!(parse_data_type(&text), Ok(back) if back == column.data_type) {
                        return Err(Failure::Krishiv(
                            self.unsupported(&format!("a column of type {text}")),
                        ));
                    }
                }
                let mut row = current.clone();
                row.columns = altered
                    .columns
                    .iter()
                    .map(|col| StoredColumn {
                        column: col.clone(),
                        comment: altered.comments.get(&col.name).cloned(),
                        default: altered.defaults.get(&col.name).cloned(),
                    })
                    .collect();
                for (key, value) in altered.properties {
                    if key.eq_ignore_ascii_case("comment") {
                        row.comment = value;
                    } else if let Some(v) = value {
                        row.properties.insert(key, v);
                    } else {
                        row.properties.remove(&key);
                    }
                }
                row.version = bump_epoch(c)?;
                store::write_table(c, &namespace, &name, &row, false)?;
                Ok((self.meta(&namespace, &name, &row), true))
            })
        })
    }

    fn drop_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _purge: bool,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table_key(id)?;
            self.change(&name, |c| {
                if read_table(c, &namespace, &name)?.is_none() {
                    return Ok((false, false));
                }
                bump_epoch(c)?;
                store::delete_table(c, &namespace, &name)?;
                Ok((true, true))
            })
        })
    }

    fn rename_table<'a>(
        &'a self,
        from: &'a TableIdent,
        to: &'a TableIdent,
    ) -> BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            let (namespace, name) = self.table_key(from)?;
            let (to_namespace, to_name) = self.table_key(to)?;
            check_new_name(&to_name)?;
            self.change(&name, |c| {
                if read_namespace(c, &to_namespace)?.is_none() {
                    return Err(schema_not_found(&to_namespace));
                }
                let Some(mut row) = read_table(c, &namespace, &name)? else {
                    return Err(Failure::Krishiv(self.not_found(&namespace, &name)));
                };
                if to_namespace != namespace {
                    return Err(Failure::Krishiv(
                        LegacyErrorTemp1073 {
                            db: namespace.clone(),
                            new_db: to_namespace.clone(),
                        }
                        .into(),
                    ));
                }
                if read_table(c, &to_namespace, &to_name)?.is_some() {
                    return Err(Failure::Krishiv(
                        TableOrViewAlreadyExists {
                            relation_name: to.to_string(),
                        }
                        .into(),
                    ));
                }
                row.version = bump_epoch(c)?;
                store::delete_table(c, &namespace, &name)?;
                store::write_table(c, &to_namespace, &to_name, &row, true)?;
                Ok(((), true))
            })
        })
    }
}
