// SPDX-License-Identifier: Apache-2.0
//! What a catalog can be asked to change (design P3-lakehouse §2.1-2.2): its
//! capabilities, Spark's `TableChange` list for `ALTER TABLE`, namespace
//! changes, and [`apply_changes`], the one place a table's columns, column
//! comments, defaults and properties are changed all-or-none, so every
//! catalog applies the same rules. What differs between formats is passed in
//! as [`AlterRules`].

use std::collections::BTreeMap;

use krishiv_common::error::{
    CannotAlterPartitionColumn, ColumnAlreadyExists, KrishivError, NotSupportedChangeColumn,
    UnresolvedColumnWithSuggestion, UnsupportedFeatureCatalogOperation,
    UnsupportedFeatureTableOperation,
};
use krishiv_types::DataType;

use super::{TableColumn, TableIdent};

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
    /// The catalog has a write half at all; a contract test checks it agrees
    /// with [`Catalog::as_write`](super::Catalog::as_write).
    pub writes: bool,
}

impl CatalogCapabilities {
    /// A catalog that writes tables and namespaces and alters columns without
    /// widening or column ids: what the in-memory catalog declares.
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
/// the SQL and Connect front ends translate one to one. A column type is a
/// `data_type` and, for `CHAR(n)` / `VARCHAR(n)`, the `declared` text
/// (`char(5)`; the type is `string`), as the catalog stores it.
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
        /// The declared `CHAR(n)` / `VARCHAR(n)` text, if any.
        declared: Option<String>,
        /// Whether it may be null.
        nullable: bool,
        /// Its comment.
        comment: Option<String>,
        /// Its position; the end of the data columns by default.
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
        /// The declared `CHAR(n)` / `VARCHAR(n)` text, if any.
        declared: Option<String>,
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
    ClusterBy {
        /// The clustering columns, each a path.
        columns: Vec<Vec<String>>,
    },
    /// `addConstraint`: not supported in P3.
    AddConstraint {
        /// The constraint's name.
        name: String,
        /// Its definition, as SQL text.
        definition: String,
    },
    /// `dropConstraint`: not supported in P3.
    DropConstraint {
        /// The constraint's name.
        name: String,
        /// No error when it is missing.
        if_exists: bool,
    },
}

/// Which type changes a format accepts. The set lives with the format; a
/// catalog's `type_widening` capability only declares that one exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TypeWidening {
    /// No type changes: the type must stay the same.
    #[default]
    None,
    /// Iceberg's promotions: `int` to `long`, `float` to `double`, a
    /// `decimal` whose precision grows at one scale. (Version 3's
    /// `date` to `timestamp_ntz` is unverified, design §2.2.)
    Iceberg,
}

impl TypeWidening {
    /// Whether a column of type `from` may become `to` under this rule.
    #[must_use]
    pub fn allows(self, from: &DataType, to: &DataType) -> bool {
        match self {
            Self::None => false,
            Self::Iceberg => match (from, to) {
                (DataType::Integer, DataType::Long) | (DataType::Float, DataType::Double) => true,
                (DataType::Decimal(a), DataType::Decimal(b)) => {
                    a.scale() == b.scale() && a.precision() < b.precision()
                }
                _ => false,
            },
        }
    }
}

/// What differs between table formats when columns change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct AlterRules {
    /// Names match as written; otherwise case-insensitively.
    pub case_sensitive: bool,
    /// Which type changes are accepted.
    pub widening: TypeWidening,
    /// Columns carry stable ids, so a rename or a drop leaves the files
    /// readable. Without them both are `UNSUPPORTED_FEATURE.TABLE_OPERATION`
    /// (a file-source table's data is read by column name).
    pub column_ids: bool,
    /// A column may move and be relaxed to nullable. Without it both are
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION`, as for a file-source table.
    pub reshape: bool,
}

/// The table `changes` apply to.
#[derive(Debug, Clone, Copy)]
pub struct AlterInput<'a> {
    /// The table, for the errors.
    pub table: &'a TableIdent,
    /// Its columns, in order, partition columns last.
    pub columns: &'a [TableColumn],
    /// Column comments by column name.
    pub comments: &'a BTreeMap<String, String>,
    /// Column defaults by column name, as SQL text.
    pub defaults: &'a BTreeMap<String, String>,
    /// The partition columns: no change may touch one, and a new column goes
    /// before them.
    pub partition_columns: &'a [String],
}

/// A table after its changes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AlteredTable {
    /// The columns.
    pub columns: Vec<TableColumn>,
    /// Column comments by final column name.
    pub comments: BTreeMap<String, String>,
    /// Column defaults by final column name.
    pub defaults: BTreeMap<String, String>,
    /// The property edits in order: a value sets, `None` removes.
    pub properties: Vec<(String, Option<String>)>,
}

fn quoted(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

struct Alter<'a> {
    input: &'a AlterInput<'a>,
    rules: AlterRules,
    state: AlteredTable,
}

impl Alter<'_> {
    fn same(&self, a: &str, b: &str) -> bool {
        if self.rules.case_sensitive {
            a == b
        } else {
            a.eq_ignore_ascii_case(b)
        }
    }

    fn table_operation(&self, operation: &str) -> KrishivError {
        UnsupportedFeatureTableOperation {
            table_name: self.input.table.to_string(),
            operation: operation.to_owned(),
        }
        .into()
    }

    fn catalog_operation(&self, operation: &str) -> KrishivError {
        UnsupportedFeatureCatalogOperation {
            catalog_name: quoted(&self.input.table.catalog),
            operation: operation.to_owned(),
        }
        .into()
    }

    /// The name of a one-part `path`; a nested path is not supported yet.
    fn top_level<'p>(&self, path: &'p [String]) -> Result<&'p str, KrishivError> {
        match path {
            [name] => Ok(name),
            _ => Err(self.catalog_operation("changing a nested column")),
        }
    }

    fn position_of(&self, name: &str) -> Option<usize> {
        self.state
            .columns
            .iter()
            .position(|c| self.same(&c.name, name))
    }

    /// The index of the column `name` (Spark's `ResolveAlterTableCommands`):
    /// an unknown one is `UNRESOLVED_COLUMN.WITH_SUGGESTION` listing the
    /// table's columns.
    fn find(&self, name: &str) -> Result<usize, KrishivError> {
        self.position_of(name).ok_or_else(|| {
            UnresolvedColumnWithSuggestion {
                object_name: quoted(name),
                proposal: self
                    .state
                    .columns
                    .iter()
                    .map(|c| quoted(&c.name))
                    .collect::<Vec<_>>()
                    .join(", "),
            }
            .into()
        })
    }

    fn is_partition(&self, column: &str) -> bool {
        self.input
            .partition_columns
            .iter()
            .any(|p| self.same(p, column))
    }

    /// Finds `path`'s column and refuses a partition column.
    fn data_column(&self, path: &[String]) -> Result<usize, KrishivError> {
        let written = self.top_level(path)?;
        let at = self.find(written)?;
        if self.is_partition(&self.state.columns[at].name) {
            return Err(CannotAlterPartitionColumn {
                table_name: self.input.table.to_string(),
                column_name: quoted(written),
            }
            .into());
        }
        Ok(at)
    }

    fn insertion_point(&self, position: Option<&ColumnPosition>) -> Result<usize, KrishivError> {
        match position {
            Some(ColumnPosition::First) => Ok(0),
            Some(ColumnPosition::After(name)) => {
                let at = self.find(name)?;
                // Partition columns stay last: nothing goes between or after them.
                if self.is_partition(&self.state.columns[at].name) {
                    return Err(CannotAlterPartitionColumn {
                        table_name: self.input.table.to_string(),
                        column_name: quoted(name),
                    }
                    .into());
                }
                Ok(at + 1)
            }
            None => Ok(self
                .state
                .columns
                .iter()
                .position(|c| self.is_partition(&c.name))
                .unwrap_or(self.state.columns.len())),
        }
    }

    fn exists(&self, name: &str, except: Option<usize>) -> bool {
        self.state
            .columns
            .iter()
            .enumerate()
            .any(|(i, c)| Some(i) != except && self.same(&c.name, name))
    }

    fn add_column(&mut self, change: &TableChange) -> Result<(), KrishivError> {
        let TableChange::AddColumn {
            path,
            data_type,
            declared,
            nullable,
            comment,
            position,
            default,
        } = change
        else {
            return Ok(());
        };
        let name = self.top_level(path)?;
        if self.exists(name, None) {
            return Err(ColumnAlreadyExists {
                column_name: quoted(name),
            }
            .into());
        }
        let at = self.insertion_point(position.as_ref())?;
        let mut column = TableColumn::new(name, data_type.clone(), *nullable);
        column.char_varchar.clone_from(declared);
        self.state.columns.insert(at, column);
        if let Some(c) = comment {
            self.state.comments.insert(name.to_owned(), c.clone());
        }
        if let Some(d) = default {
            self.state.defaults.insert(name.to_owned(), d.clone());
        }
        Ok(())
    }

    fn update_type(
        &mut self,
        path: &[String],
        data_type: &DataType,
        declared: Option<&String>,
    ) -> Result<(), KrishivError> {
        let at = self.data_column(path)?;
        let current = &self.state.columns[at];
        if current.data_type == *data_type && current.char_varchar.as_ref() == declared {
            return Ok(());
        }
        if declared.is_none()
            && current.char_varchar.is_none()
            && self.rules.widening.allows(&current.data_type, data_type)
        {
            self.state.columns[at].data_type = data_type.clone();
            return Ok(());
        }
        let sql = |t: &DataType, declared: Option<&String>| {
            format!(
                "\"{}\"",
                declared.map_or_else(|| t.sql(), |d| d.to_uppercase())
            )
        };
        let id = quoted(&current.name);
        Err(NotSupportedChangeColumn {
            table: self.input.table.to_string(),
            origin_name: id.clone(),
            origin_type: sql(&current.data_type, current.char_varchar.as_ref()),
            new_name: id,
            new_type: sql(data_type, declared),
        }
        .into())
    }

    fn rename(&mut self, path: &[String], new_name: &str) -> Result<(), KrishivError> {
        let at = self.data_column(path)?;
        if !self.rules.column_ids {
            return Err(self.table_operation("RENAME COLUMN"));
        }
        if self.exists(new_name, Some(at)) {
            return Err(ColumnAlreadyExists {
                column_name: quoted(new_name),
            }
            .into());
        }
        let old = std::mem::replace(&mut self.state.columns[at].name, new_name.to_owned());
        if let Some(c) = self.state.comments.remove(&old) {
            self.state.comments.insert(new_name.to_owned(), c);
        }
        if let Some(d) = self.state.defaults.remove(&old) {
            self.state.defaults.insert(new_name.to_owned(), d);
        }
        Ok(())
    }

    fn delete(&mut self, path: &[String], if_exists: bool) -> Result<(), KrishivError> {
        let name = self.top_level(path)?;
        if self.position_of(name).is_none() && if_exists {
            return Ok(());
        }
        let at = self.data_column(path)?;
        if !self.rules.column_ids {
            return Err(self.table_operation("DROP COLUMN"));
        }
        let gone = self.state.columns.remove(at);
        self.state.comments.remove(&gone.name);
        self.state.defaults.remove(&gone.name);
        Ok(())
    }

    fn apply(&mut self, change: &TableChange) -> Result<(), KrishivError> {
        match change {
            TableChange::SetProperty { key, value } => self
                .state
                .properties
                .push((key.clone(), Some(value.clone()))),
            TableChange::RemoveProperty { key } => self.state.properties.push((key.clone(), None)),
            TableChange::ClusterBy { .. } => return Err(self.catalog_operation("CLUSTER BY")),
            TableChange::AddConstraint { .. } | TableChange::DropConstraint { .. } => {
                return Err(self.catalog_operation("table constraints"));
            }
            TableChange::AddColumn { .. } => self.add_column(change)?,
            TableChange::RenameColumn { path, new_name } => self.rename(path, new_name)?,
            TableChange::UpdateColumnType {
                path,
                data_type,
                declared,
            } => self.update_type(path, data_type, declared.as_ref())?,
            TableChange::UpdateColumnNullability { path, nullable } => {
                let at = self.data_column(path)?;
                if self.state.columns[at].nullable != *nullable {
                    if !self.rules.reshape || !*nullable {
                        return Err(self.table_operation("ALTER COLUMN ... SET NOT NULL"));
                    }
                    self.state.columns[at].nullable = true;
                }
            }
            TableChange::UpdateColumnComment { path, comment } => {
                let at = self.data_column(path)?;
                let name = self.state.columns[at].name.clone();
                match comment {
                    Some(c) => self.state.comments.insert(name, c.clone()),
                    None => self.state.comments.remove(&name),
                };
            }
            TableChange::UpdateColumnDefault { path, default } => {
                let at = self.data_column(path)?;
                let name = self.state.columns[at].name.clone();
                match default {
                    Some(d) => self.state.defaults.insert(name, d.clone()),
                    None => self.state.defaults.remove(&name),
                };
            }
            TableChange::UpdateColumnPosition { path, position } => {
                let at = self.data_column(path)?;
                if !self.rules.reshape {
                    return Err(self.table_operation("ALTER COLUMN ... FIRST | AFTER"));
                }
                let column = self.state.columns.remove(at);
                let to = self.insertion_point(Some(position))?;
                self.state.columns.insert(to, column);
            }
            TableChange::DeleteColumn { path, if_exists } => self.delete(path, *if_exists)?,
        }
        Ok(())
    }
}

/// `input` after `changes`, or the first failing change's error with nothing
/// applied (the `alterTable` contract: in order, all or none). Column names
/// resolve against the list as earlier changes left it.
///
/// A nested path, a constraint and `CLUSTER BY` are
/// `UNSUPPORTED_FEATURE.CATALOG_OPERATION`; a change to a partition column is
/// `CANNOT_ALTER_PARTITION_COLUMN`; an unknown column is
/// `UNRESOLVED_COLUMN.WITH_SUGGESTION`; an added or renamed column that exists
/// is `COLUMN_ALREADY_EXISTS`; a type change [`AlterRules::widening`] does not
/// allow is `NOT_SUPPORTED_CHANGE_COLUMN`.
///
/// # Errors
/// As above.
pub fn apply_changes(
    input: &AlterInput<'_>,
    changes: &[TableChange],
    rules: AlterRules,
) -> Result<AlteredTable, KrishivError> {
    let mut alter = Alter {
        input,
        rules,
        state: AlteredTable {
            columns: input.columns.to_vec(),
            comments: input.comments.clone(),
            defaults: input.defaults.clone(),
            properties: Vec::new(),
        },
    };
    for change in changes {
        alter.apply(change)?;
    }
    Ok(alter.state)
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DecimalType;

    use super::*;

    fn table() -> TableIdent {
        TableIdent {
            catalog: "c".into(),
            namespace: vec!["db".into()],
            name: "t".into(),
        }
    }

    fn cols() -> Vec<TableColumn> {
        vec![
            TableColumn::new("a", DataType::Integer, true),
            TableColumn::new("b", DataType::String, true),
            TableColumn::new("p", DataType::Integer, true),
        ]
    }

    fn names(t: &AlteredTable) -> Vec<&str> {
        t.columns.iter().map(|c| c.name.as_str()).collect()
    }

    fn full() -> AlterRules {
        AlterRules {
            widening: TypeWidening::Iceberg,
            column_ids: true,
            reshape: true,
            ..AlterRules::default()
        }
    }

    fn run(changes: &[TableChange], rules: AlterRules) -> Result<AlteredTable, KrishivError> {
        let (id, columns) = (table(), cols());
        let (comments, defaults) = (BTreeMap::new(), BTreeMap::new());
        let input = AlterInput {
            table: &id,
            columns: &columns,
            comments: &comments,
            defaults: &defaults,
            partition_columns: &["p".to_owned()],
        };
        apply_changes(&input, changes, rules)
    }

    fn fails(changes: &[TableChange], rules: AlterRules) -> (ErrorClass, Option<&'static str>) {
        let e = run(changes, rules).unwrap_err();
        (e.error_class(), e.sql_state())
    }

    fn add(name: &str, position: Option<ColumnPosition>) -> TableChange {
        TableChange::AddColumn {
            path: vec![name.to_owned()],
            data_type: DataType::Long,
            declared: None,
            nullable: true,
            comment: Some(format!("the {name}")),
            position,
            default: Some("1".into()),
        }
    }

    fn path(name: &str) -> Vec<String> {
        vec![name.to_owned()]
    }

    #[test]
    fn a_new_column_goes_before_the_partition_columns() {
        let t = run(&[add("c", None)], full()).unwrap();
        assert_eq!(names(&t), ["a", "b", "c", "p"]);
        assert_eq!(t.comments["c"], "the c");
        assert_eq!(t.defaults["c"], "1");
    }

    #[test]
    fn changes_apply_in_order_and_names_resolve_as_earlier_changes_left_them() {
        let t = run(
            &[
                add("c", None),
                TableChange::UpdateColumnComment {
                    path: path("c"),
                    comment: Some("kept".into()),
                },
                TableChange::RenameColumn {
                    path: path("c"),
                    new_name: "z".into(),
                },
                add("y", Some(ColumnPosition::First)),
                TableChange::UpdateColumnPosition {
                    path: path("a"),
                    position: ColumnPosition::After("b".into()),
                },
            ],
            full(),
        )
        .unwrap();
        assert_eq!(names(&t), ["y", "b", "a", "z", "p"]);
        assert_eq!(t.comments["z"], "kept", "a rename carries the comment");
        assert_eq!(t.defaults["z"], "1", "and the default");
        assert!(!t.comments.contains_key("c"));
    }

    #[test]
    fn the_first_failing_change_is_the_error() {
        let (class, state) = fails(
            &[
                add("c", None),
                add("a", None),
                TableChange::DeleteColumn {
                    path: path("nope"),
                    if_exists: false,
                },
            ],
            full(),
        );
        assert_eq!(class, ErrorClass::ColumnAlreadyExists);
        assert_eq!(state, Some("42711"));
    }

    #[test]
    fn a_comment_or_default_on_an_unknown_column_is_unresolved() {
        for change in [
            TableChange::UpdateColumnComment {
                path: path("nope"),
                comment: None,
            },
            TableChange::UpdateColumnDefault {
                path: path("nope"),
                default: None,
            },
        ] {
            let e = run(&[change], full()).unwrap_err();
            assert_eq!(e.error_class(), ErrorClass::UnresolvedColumnWithSuggestion);
            assert!(e.to_string().contains("`a`, `b`, `p`"), "{e}");
        }
    }

    #[test]
    fn names_match_case_insensitively_unless_told_otherwise() {
        let rename = |case_sensitive| {
            run(
                &[TableChange::RenameColumn {
                    path: path("A"),
                    new_name: "x".into(),
                }],
                AlterRules {
                    case_sensitive,
                    ..full()
                },
            )
        };
        assert_eq!(names(&rename(false).unwrap()), ["x", "b", "p"]);
        assert!(rename(true).is_err());
        let (class, _) = fails(&[add("A", None)], full());
        assert_eq!(class, ErrorClass::ColumnAlreadyExists);
        let t = run(
            &[add("A", None)],
            AlterRules {
                case_sensitive: true,
                ..full()
            },
        )
        .unwrap();
        assert_eq!(names(&t), ["a", "b", "A", "p"]);
    }

    #[test]
    fn a_rename_onto_another_column_is_refused_but_onto_itself_is_not() {
        let rename = |to: &str| {
            run(
                &[TableChange::RenameColumn {
                    path: path("a"),
                    new_name: to.into(),
                }],
                full(),
            )
        };
        assert!(rename("b").is_err());
        assert_eq!(names(&rename("A").unwrap()), ["A", "b", "p"]);
    }

    #[test]
    fn a_type_only_widens_under_the_formats_rule() {
        let to = |t: DataType, widening| {
            run(
                &[TableChange::UpdateColumnType {
                    path: path("a"),
                    data_type: t,
                    declared: None,
                }],
                AlterRules { widening, ..full() },
            )
        };
        assert_eq!(
            to(DataType::Long, TypeWidening::Iceberg).unwrap().columns[0].data_type,
            DataType::Long
        );
        for refused in [
            to(DataType::Long, TypeWidening::None),
            to(DataType::String, TypeWidening::Iceberg),
        ] {
            let e = refused.unwrap_err();
            assert_eq!(e.error_class(), ErrorClass::NotSupportedChangeColumn);
            assert_eq!(e.sql_state(), Some("0A000"));
            let text = e.to_string();
            assert!(
                text.contains("`c`.`db`.`t`") && text.contains("\"INT\""),
                "{text}"
            );
        }
        to(DataType::Integer, TypeWidening::None).unwrap();
    }

    #[test]
    fn a_char_or_varchar_length_is_kept_and_cannot_change() {
        let (id, empty) = (table(), BTreeMap::new());
        let mut c = TableColumn::new("a", DataType::String, true);
        c.char_varchar = Some("char(5)".into());
        let columns = [c];
        let input = AlterInput {
            table: &id,
            columns: &columns,
            comments: &empty,
            defaults: &empty,
            partition_columns: &[],
        };
        let to = |declared: Option<&str>| {
            apply_changes(
                &input,
                &[TableChange::UpdateColumnType {
                    path: path("a"),
                    data_type: DataType::String,
                    declared: declared.map(str::to_owned),
                }],
                full(),
            )
        };
        to(Some("char(5)")).unwrap();
        let e = to(Some("varchar(10)")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::NotSupportedChangeColumn);
        let text = e.to_string();
        assert!(
            text.contains("\"CHAR(5)\"") && text.contains("\"VARCHAR(10)\""),
            "{text}"
        );
        assert!(to(None).is_err());
        let added = apply_changes(
            &input,
            &[TableChange::AddColumn {
                path: path("c"),
                data_type: DataType::String,
                declared: Some("varchar(3)".into()),
                nullable: true,
                comment: None,
                position: None,
                default: None,
            }],
            full(),
        )
        .unwrap();
        assert_eq!(added.columns[1].char_varchar.as_deref(), Some("varchar(3)"));
    }

    #[test]
    fn decimals_widen_at_the_same_scale() {
        let d = |p, s| DataType::Decimal(DecimalType::try_new(p, s).unwrap());
        let ice = TypeWidening::Iceberg;
        assert!(ice.allows(&d(10, 2), &d(12, 2)));
        assert!(!ice.allows(&d(10, 2), &d(12, 3)));
        assert!(!ice.allows(&d(12, 2), &d(10, 2)));
        assert!(ice.allows(&DataType::Float, &DataType::Double));
        assert!(!ice.allows(&DataType::Long, &DataType::Integer));
        assert!(!TypeWidening::None.allows(&DataType::Integer, &DataType::Long));
    }

    #[test]
    fn a_partition_column_cannot_change() {
        for change in [
            TableChange::UpdateColumnComment {
                path: path("p"),
                comment: Some("x".into()),
            },
            TableChange::RenameColumn {
                path: path("p"),
                new_name: "q".into(),
            },
            TableChange::DeleteColumn {
                path: path("p"),
                if_exists: false,
            },
        ] {
            let (class, _) = fails(&[change], full());
            assert_eq!(class, ErrorClass::CannotAlterPartitionColumn);
        }
    }

    #[test]
    fn a_format_without_column_ids_refuses_rename_and_drop() {
        let rules = AlterRules {
            column_ids: false,
            ..full()
        };
        for (change, op) in [
            (
                TableChange::RenameColumn {
                    path: path("a"),
                    new_name: "x".into(),
                },
                "RENAME COLUMN",
            ),
            (
                TableChange::DeleteColumn {
                    path: path("a"),
                    if_exists: false,
                },
                "DROP COLUMN",
            ),
        ] {
            let e = run(&[change], rules).unwrap_err();
            assert_eq!(
                e.error_class(),
                ErrorClass::UnsupportedFeatureTableOperation
            );
            assert!(e.to_string().contains(op), "{e}");
        }
        let (class, _) = fails(
            &[TableChange::RenameColumn {
                path: path("nope"),
                new_name: "x".into(),
            }],
            rules,
        );
        assert_eq!(
            class,
            ErrorClass::UnresolvedColumnWithSuggestion,
            "an unknown column is reported first"
        );
    }

    #[test]
    fn nullability_only_relaxes_and_position_needs_reshape() {
        let nullability = |nullable| TableChange::UpdateColumnNullability {
            path: path("a"),
            nullable,
        };
        run(&[nullability(true)], full()).unwrap();
        let (class, _) = fails(&[nullability(false)], full());
        assert_eq!(class, ErrorClass::UnsupportedFeatureTableOperation);

        let mut not_null = cols();
        not_null[0].nullable = false;
        let (id, empty) = (table(), BTreeMap::new());
        let input = AlterInput {
            table: &id,
            columns: &not_null,
            comments: &empty,
            defaults: &empty,
            partition_columns: &[],
        };
        let flat = AlterRules {
            reshape: false,
            ..full()
        };
        assert!(apply_changes(&input, &[nullability(true)], flat).is_err());
        let relaxed = apply_changes(&input, &[nullability(true)], full()).unwrap();
        assert!(relaxed.columns[0].nullable);
        let (class, _) = fails(
            &[TableChange::UpdateColumnPosition {
                path: path("a"),
                position: ColumnPosition::First,
            }],
            flat,
        );
        assert_eq!(class, ErrorClass::UnsupportedFeatureTableOperation);
    }

    #[test]
    fn deleting_a_missing_column_needs_if_exists() {
        let delete = |if_exists| {
            run(
                &[TableChange::DeleteColumn {
                    path: path("nope"),
                    if_exists,
                }],
                full(),
            )
        };
        assert_eq!(delete(true).unwrap().columns.len(), 3);
        assert!(delete(false).is_err());
        let t = run(
            &[TableChange::DeleteColumn {
                path: path("a"),
                if_exists: false,
            }],
            full(),
        )
        .unwrap();
        assert_eq!(names(&t), ["b", "p"]);
    }

    #[test]
    fn nothing_is_placed_after_a_partition_column() {
        let (class, _) = fails(&[add("c", Some(ColumnPosition::After("p".into())))], full());
        assert_eq!(class, ErrorClass::CannotAlterPartitionColumn);
        let (class, _) = fails(
            &[TableChange::UpdateColumnPosition {
                path: path("a"),
                position: ColumnPosition::After("P".into()),
            }],
            full(),
        );
        assert_eq!(class, ErrorClass::CannotAlterPartitionColumn);
        // After the last data column is the edge that is allowed.
        let t = run(&[add("c", Some(ColumnPosition::After("b".into())))], full()).unwrap();
        assert_eq!(names(&t), ["a", "b", "c", "p"]);
        let t = run(
            &[TableChange::UpdateColumnPosition {
                path: path("a"),
                position: ColumnPosition::After("b".into()),
            }],
            full(),
        )
        .unwrap();
        assert_eq!(names(&t), ["b", "a", "p"]);
    }

    #[test]
    fn a_position_after_a_missing_column_is_unresolved() {
        let (class, _) = fails(
            &[add("c", Some(ColumnPosition::After("nope".into())))],
            full(),
        );
        assert_eq!(class, ErrorClass::UnresolvedColumnWithSuggestion);
    }

    #[test]
    fn nested_paths_constraints_and_clustering_are_unsupported_catalog_operations() {
        let nested = TableChange::UpdateColumnNullability {
            path: vec!["a".into(), "x".into()],
            nullable: false,
        };
        for change in [
            nested,
            TableChange::ClusterBy {
                columns: vec![path("a")],
            },
            TableChange::AddConstraint {
                name: "k".into(),
                definition: "CHECK (a > 0)".into(),
            },
            TableChange::DropConstraint {
                name: "k".into(),
                if_exists: true,
            },
        ] {
            let (class, state) = fails(&[change], full());
            assert_eq!(class, ErrorClass::UnsupportedFeatureCatalogOperation);
            assert_eq!(state, Some("0A000"));
        }
    }

    #[test]
    fn properties_are_returned_in_order() {
        let t = run(
            &[
                TableChange::SetProperty {
                    key: "k".into(),
                    value: "v".into(),
                },
                TableChange::RemoveProperty { key: "k".into() },
            ],
            full(),
        )
        .unwrap();
        assert_eq!(
            t.properties,
            [
                ("k".to_owned(), Some("v".to_owned())),
                ("k".to_owned(), None)
            ]
        );
        assert_eq!(t.columns, cols());
    }
}
