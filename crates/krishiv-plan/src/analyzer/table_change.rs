// SPDX-License-Identifier: Apache-2.0
//! What a catalog can be asked to change (design P3-lakehouse §2.1-2.2): its
//! capabilities, Spark's `TableChange` list for `ALTER TABLE`, namespace
//! changes, and the one place a column list is changed all-or-none, so every
//! catalog applies the same rules.

use krishiv_common::error::{
    ColumnAlreadyExists, FieldNotFound, KrishivError, NotSupportedChangeColumn,
    UnsupportedFeatureCatalogOperation,
};
use krishiv_types::DataType;

use super::TableColumn;

/// What a catalog supports; the analyzer and the DDL commands check it before
/// calling a method, so the refusal is uniform. Every field starts `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per capability, as the design lists them"
)]
pub struct CatalogCapabilities {
    /// Namespaces can be created, altered and dropped.
    pub namespaces: bool,
    /// A namespace name has more than one part.
    pub nested_namespaces: bool,
    /// `rename_table` may move a table to another namespace.
    pub rename_across_namespaces: bool,
    /// `alter_table` changes columns.
    pub alter_columns: bool,
    /// A column type may widen (Iceberg promotions, Delta type widening).
    pub type_widening: bool,
    /// Columns carry stable ids, so a rename needs no data rewrite.
    pub column_ids: bool,
    /// A table loads at a version or timestamp.
    pub time_travel: bool,
    /// Staged replace is native (`stage_replace` is one atomic update).
    pub staged_replace: bool,
    /// The catalog has a write half at all.
    pub writes: bool,
}

/// A change to a namespace (`ALTER NAMESPACE`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NamespaceChange {
    /// Sets a property.
    SetProperty {
        /// The key.
        key: String,
        /// The value.
        value: String,
    },
    /// Removes a property.
    RemoveProperty {
        /// The key.
        key: String,
    },
}

/// Where a column goes (`FIRST`, `AFTER col`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColumnPosition {
    /// First in the table.
    First,
    /// After the named column.
    After(String),
}

/// One step of an `ALTER TABLE`, mirroring Spark's `TableChange` factories so
/// the SQL and Connect front ends translate one to one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TableChange {
    /// `setProperty`.
    SetProperty {
        /// The key.
        key: String,
        /// The value.
        value: String,
    },
    /// `removeProperty`.
    RemoveProperty {
        /// The key.
        key: String,
    },
    /// `addColumn`; a nested `path` adds a struct field.
    AddColumn {
        /// The column's path; the last part is its name.
        path: Vec<String>,
        /// The type.
        data_type: DataType,
        /// Whether it may be null.
        nullable: bool,
        /// Its comment.
        comment: Option<String>,
        /// Its position; the end by default.
        position: Option<ColumnPosition>,
        /// Its default expression, as SQL text.
        default: Option<String>,
    },
    /// `renameColumn`.
    RenameColumn {
        /// The column.
        path: Vec<String>,
        /// Its new name.
        new_name: String,
    },
    /// `updateColumnType`; widening only.
    UpdateColumnType {
        /// The column.
        path: Vec<String>,
        /// The new type.
        data_type: DataType,
    },
    /// `updateColumnNullability`.
    UpdateColumnNullability {
        /// The column.
        path: Vec<String>,
        /// Whether it may be null.
        nullable: bool,
    },
    /// `updateColumnComment`.
    UpdateColumnComment {
        /// The column.
        path: Vec<String>,
        /// The comment; `None` drops it.
        comment: Option<String>,
    },
    /// `updateColumnPosition`.
    UpdateColumnPosition {
        /// The column.
        path: Vec<String>,
        /// Where it goes.
        position: ColumnPosition,
    },
    /// `updateColumnDefaultValue`.
    UpdateColumnDefault {
        /// The column.
        path: Vec<String>,
        /// The default as SQL text; `None` drops it.
        default: Option<String>,
    },
    /// `deleteColumn`.
    DeleteColumn {
        /// The column.
        path: Vec<String>,
        /// No error when it is missing.
        if_exists: bool,
    },
    /// `clusterBy`: not supported in P3.
    ClusterBy,
    /// `addConstraint`: not supported in P3.
    AddConstraint,
    /// `dropConstraint`: not supported in P3.
    DropConstraint,
}

/// Whether a column of type `from` may become `to` without rewriting its
/// data: the promotions Iceberg allows (`int` to `long`, `float` to `double`,
/// a `decimal` whose precision grows at the same scale).
#[must_use]
pub fn widens(from: &DataType, to: &DataType) -> bool {
    match (from, to) {
        (DataType::Integer, DataType::Long) | (DataType::Float, DataType::Double) => true,
        (DataType::Decimal(a), DataType::Decimal(b)) => {
            a.scale() == b.scale() && a.precision() < b.precision()
        }
        _ => false,
    }
}

fn quoted(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// The columns of a table after `changes`, or the first error with the table
/// untouched (the `alterTable` contract: in order, all or none). Changes that
/// are not about the column list (properties, comments, defaults) are the
/// catalog's to store and pass through here; a nested path, a constraint or
/// `CLUSTER BY` is `UNSUPPORTED_FEATURE.CATALOG_OPERATION`, and a type change
/// that does not [`widen`] (or any change when `widening` is false) is
/// `NOT_SUPPORTED_CHANGE_COLUMN`.
///
/// # Errors
/// As above; `COLUMN_ALREADY_EXISTS` for an added or renamed column that
/// exists, and `FIELD_NOT_FOUND` for one that does not.
pub fn apply_column_changes(
    catalog: &str,
    table: &str,
    columns: &[TableColumn],
    changes: &[TableChange],
    case_sensitive: bool,
    widening: bool,
) -> Result<Vec<TableColumn>, KrishivError> {
    let mut out = columns.to_vec();
    let same = |a: &str, b: &str| {
        if case_sensitive {
            a == b
        } else {
            a.eq_ignore_ascii_case(b)
        }
    };
    let unsupported = |what: &str| -> KrishivError {
        UnsupportedFeatureCatalogOperation {
            catalog_name: quoted(catalog),
            operation: what.to_owned(),
        }
        .into()
    };
    for change in changes {
        match change {
            TableChange::SetProperty { .. }
            | TableChange::RemoveProperty { .. }
            | TableChange::UpdateColumnComment { .. }
            | TableChange::UpdateColumnDefault { .. } => {}
            TableChange::ClusterBy => return Err(unsupported("CLUSTER BY")),
            TableChange::AddConstraint | TableChange::DropConstraint => {
                return Err(unsupported("table constraints"));
            }
            TableChange::AddColumn {
                path,
                data_type,
                nullable,
                position,
                ..
            } => {
                let name = top_level(path, &unsupported)?;
                if out.iter().any(|c| same(&c.name, name)) {
                    return Err(ColumnAlreadyExists {
                        column_name: quoted(name),
                    }
                    .into());
                }
                let at = match position {
                    None => out.len(),
                    Some(p) => position_index(&out, p, &same)?,
                };
                out.insert(at, TableColumn::new(name, data_type.clone(), *nullable));
            }
            TableChange::RenameColumn { path, new_name } => {
                let i = find(&out, top_level(path, &unsupported)?, &same)?;
                if out
                    .iter()
                    .enumerate()
                    .any(|(j, c)| j != i && same(&c.name, new_name))
                {
                    return Err(ColumnAlreadyExists {
                        column_name: quoted(new_name),
                    }
                    .into());
                }
                new_name.clone_into(&mut out[i].name);
            }
            TableChange::UpdateColumnType { path, data_type } => {
                let i = find(&out, top_level(path, &unsupported)?, &same)?;
                if out[i].data_type != *data_type {
                    if !(widening && widens(&out[i].data_type, data_type)) {
                        let id = quoted(&out[i].name);
                        return Err(NotSupportedChangeColumn {
                            table: quoted_table(catalog, table),
                            origin_name: id.clone(),
                            origin_type: format!("\"{}\"", out[i].data_type.sql()),
                            new_name: id,
                            new_type: format!("\"{}\"", data_type.sql()),
                        }
                        .into());
                    }
                    out[i].data_type = data_type.clone();
                    out[i].char_varchar = None;
                }
            }
            TableChange::UpdateColumnNullability { path, nullable } => {
                let i = find(&out, top_level(path, &unsupported)?, &same)?;
                out[i].nullable = *nullable;
            }
            TableChange::UpdateColumnPosition { path, position } => {
                let i = find(&out, top_level(path, &unsupported)?, &same)?;
                let column = out.remove(i);
                let at = position_index(&out, position, &same)?;
                out.insert(at, column);
            }
            TableChange::DeleteColumn { path, if_exists } => {
                let name = top_level(path, &unsupported)?;
                match out.iter().position(|c| same(&c.name, name)) {
                    Some(i) => {
                        out.remove(i);
                    }
                    None if *if_exists => {}
                    None => return Err(missing(&out, name)),
                }
            }
        }
    }
    Ok(out)
}

fn quoted_table(catalog: &str, table: &str) -> String {
    format!("{}.{}", quoted(catalog), quoted(table))
}

/// The one-part name of `path`; a nested path is not supported yet.
fn top_level<'p>(
    path: &'p [String],
    unsupported: &dyn Fn(&str) -> KrishivError,
) -> Result<&'p str, KrishivError> {
    match path {
        [name] => Ok(name),
        _ => Err(unsupported("changing a nested column")),
    }
}

fn missing(columns: &[TableColumn], name: &str) -> KrishivError {
    FieldNotFound {
        field_name: quoted(name),
        fields: columns
            .iter()
            .map(|c| quoted(&c.name))
            .collect::<Vec<_>>()
            .join(", "),
    }
    .into()
}

fn find(
    columns: &[TableColumn],
    name: &str,
    same: &dyn Fn(&str, &str) -> bool,
) -> Result<usize, KrishivError> {
    columns
        .iter()
        .position(|c| same(&c.name, name))
        .ok_or_else(|| missing(columns, name))
}

fn position_index(
    columns: &[TableColumn],
    position: &ColumnPosition,
    same: &dyn Fn(&str, &str) -> bool,
) -> Result<usize, KrishivError> {
    match position {
        ColumnPosition::First => Ok(0),
        ColumnPosition::After(name) => Ok(find(columns, name, same)? + 1),
    }
}

#[cfg(test)]
mod tests {
    use krishiv_types::DecimalType;

    use super::*;

    fn cols() -> Vec<TableColumn> {
        vec![
            TableColumn::new("a", DataType::Integer, true),
            TableColumn::new("b", DataType::String, true),
        ]
    }

    fn names(columns: &[TableColumn]) -> Vec<&str> {
        columns.iter().map(|c| c.name.as_str()).collect()
    }

    fn apply(changes: &[TableChange]) -> Result<Vec<TableColumn>, KrishivError> {
        apply_column_changes("c", "t", &cols(), changes, false, true)
    }

    fn add(name: &str, position: Option<ColumnPosition>) -> TableChange {
        TableChange::AddColumn {
            path: vec![name.to_owned()],
            data_type: DataType::Long,
            nullable: true,
            comment: None,
            position,
            default: None,
        }
    }

    #[test]
    fn changes_apply_in_order() {
        let out = apply(&[
            add("c", None),
            TableChange::RenameColumn {
                path: vec!["c".into()],
                new_name: "z".into(),
            },
            add("y", Some(ColumnPosition::First)),
            TableChange::UpdateColumnPosition {
                path: vec!["a".into()],
                position: ColumnPosition::After("b".into()),
            },
        ])
        .unwrap();
        assert_eq!(names(&out), ["y", "b", "a", "z"]);
    }

    #[test]
    fn a_failing_change_leaves_the_table_unchanged() {
        let before = cols();
        let err = apply_column_changes(
            "c",
            "t",
            &before,
            &[add("c", None), add("a", None)],
            false,
            true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("`a`"), "{err}");
        assert_eq!(before, cols());
    }

    #[test]
    fn names_match_case_insensitively_unless_told_otherwise() {
        let rename = |case_sensitive| {
            apply_column_changes(
                "c",
                "t",
                &cols(),
                &[TableChange::RenameColumn {
                    path: vec!["A".into()],
                    new_name: "x".into(),
                }],
                case_sensitive,
                true,
            )
        };
        assert_eq!(names(&rename(false).unwrap()), ["x", "b"]);
        assert!(rename(true).is_err());
    }

    #[test]
    fn a_type_only_widens() {
        let to = |t: DataType, widening| {
            apply_column_changes(
                "c",
                "t",
                &cols(),
                &[TableChange::UpdateColumnType {
                    path: vec!["a".into()],
                    data_type: t,
                }],
                false,
                widening,
            )
        };
        assert_eq!(
            to(DataType::Long, true).unwrap()[0].data_type,
            DataType::Long
        );
        assert!(to(DataType::Long, false).is_err());
        assert!(to(DataType::String, true).is_err());
        assert!(to(DataType::Integer, false).is_ok());
    }

    #[test]
    fn decimals_widen_at_the_same_scale() {
        let d = |p, s| DataType::Decimal(DecimalType::try_new(p, s).unwrap());
        assert!(widens(&d(10, 2), &d(12, 2)));
        assert!(!widens(&d(10, 2), &d(12, 3)));
        assert!(!widens(&d(12, 2), &d(10, 2)));
        assert!(widens(&DataType::Float, &DataType::Double));
        assert!(!widens(&DataType::Long, &DataType::Integer));
    }

    #[test]
    fn deleting_a_missing_column_needs_if_exists() {
        let delete = |if_exists| {
            apply(&[TableChange::DeleteColumn {
                path: vec!["nope".into()],
                if_exists,
            }])
        };
        assert_eq!(delete(true).unwrap().len(), 2);
        assert!(delete(false).is_err());
    }

    #[test]
    fn nested_paths_constraints_and_clustering_are_refused() {
        let nested = TableChange::UpdateColumnNullability {
            path: vec!["a".into(), "x".into()],
            nullable: false,
        };
        for change in [
            nested,
            TableChange::ClusterBy,
            TableChange::AddConstraint,
            TableChange::DropConstraint,
        ] {
            let err = apply(&[change]).unwrap_err();
            assert!(err.to_string().contains("does not support"), "{err}");
        }
    }

    #[test]
    fn properties_comments_and_defaults_pass_through() {
        let out = apply(&[
            TableChange::SetProperty {
                key: "k".into(),
                value: "v".into(),
            },
            TableChange::UpdateColumnComment {
                path: vec!["a".into()],
                comment: Some("c".into()),
            },
        ])
        .unwrap();
        assert_eq!(out, cols());
    }
}
