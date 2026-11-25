// SPDX-License-Identifier: Apache-2.0
//! The result schemas of the `SHOW`, `DESCRIBE`, `EXPLAIN`, `SET` and `RESET` commands (design
//! `P3-T063-analyze-api.md` §2, review m6, N1b review m1): they come from the session's command code, not
//! from analysis, so they are listed here per command, as Spark 4.2's goldens print them (`-- !query schema`
//! lines) and the commands return them. `EXPLAIN` is one `plan` string (`explain.rs`); `SET` is `key`,
//! `value` (`SET -v` adds `meaning` and `Since version`; `SET PATH` has none) and `RESET` has none
//! (`command.rs`).
//!
//! The integration test `command_output_schemas` (`tests/analyze.rs`) runs every `SHOW`/`DESCRIBE`
//! statement of the parser corpus (which has one per command kind, as `krishiv-sql`'s classify test
//! checks), and the `EXPLAIN`, `SET` and `RESET` forms, and compares the names and types with the frame the
//! command returns. A command kind added to `krishiv-sql` later falls into the wildcard arms below (the
//! enums are `#[non_exhaustive]` in another crate) and has no result schema until it is listed here.
//!
//! The tables below are the one list: the session builds each command's frame from the same table
//! ([`CommandColumn`], `Session::string_frame`), so the frame and `analyze` cannot disagree.
//!
//! Nullability is Spark 4.2.0's command attributes (`DescribeCommandSchema.scala`, `v2Commands.scala` and the v1
//! commands' `output`; table in krishiv-internal `research/spark/show-describe-nullability/notes.md`): every
//! column is non-null except `comment` of `DESCRIBE TABLE`/`DESCRIBE QUERY` and `info_value` of
//! `DESCRIBE NAMESPACE`; `EXPLAIN`'s `plan` is nullable (`commands.scala:166`).

use krishiv_sql::ast::{DescribeKind, SetKind, ShowKind, Statement};
use krishiv_types::DataType;

use super::OutputColumn;

/// One result column of a command: its name and whether Spark 4.2.0 declares it nullable. `isTemporary`
/// and `SHOW GRANTS`'s `grant_option` are `BOOLEAN`, every other column `STRING`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandColumn {
    /// The column name.
    pub(crate) name: &'static str,
    /// Whether Spark declares it nullable.
    pub(crate) nullable: bool,
}

impl CommandColumn {
    /// A non-null column.
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            nullable: false,
        }
    }

    /// A nullable column.
    const fn nullable(name: &'static str) -> Self {
        Self {
            name,
            nullable: true,
        }
    }

    /// Its type in the frame.
    fn data_type(self) -> DataType {
        if matches!(self.name, "isTemporary" | "grant_option") {
            DataType::Boolean
        } else {
            DataType::String
        }
    }
}

const fn c(name: &'static str) -> CommandColumn {
    CommandColumn::new(name)
}

/// `SHOW FUNCTIONS`.
pub(crate) const SHOW_FUNCTIONS: &[CommandColumn] = &[c("function")];
/// `DESCRIBE FUNCTION`.
pub(crate) const DESCRIBE_FUNCTION: &[CommandColumn] = &[c("function_desc")];

/// `SHOW TABLES`.
pub(crate) const SHOW_TABLES: &[CommandColumn] =
    &[c("namespace"), c("tableName"), c("isTemporary")];
/// `SHOW TABLE EXTENDED`.
pub(crate) const SHOW_TABLE_EXTENDED: &[CommandColumn] = &[
    c("namespace"),
    c("tableName"),
    c("isTemporary"),
    c("information"),
];
/// `SHOW VIEWS`.
pub(crate) const SHOW_VIEWS: &[CommandColumn] = &[c("namespace"), c("viewName"), c("isTemporary")];
/// `SHOW COLUMNS`.
pub(crate) const SHOW_COLUMNS: &[CommandColumn] = &[c("col_name")];
/// `SHOW NAMESPACES`.
pub(crate) const SHOW_NAMESPACES: &[CommandColumn] = &[c("namespace")];
/// `SHOW CREATE TABLE`.
pub(crate) const SHOW_CREATE_TABLE: &[CommandColumn] = &[c("createtab_stmt")];
/// `SHOW TBLPROPERTIES`.
pub(crate) const SHOW_TBLPROPERTIES: &[CommandColumn] = &[c("key"), c("value")];
/// `SHOW PARTITIONS`.
pub(crate) const SHOW_PARTITIONS: &[CommandColumn] = &[c("partition")];
/// `SHOW CATALOGS`.
pub(crate) const SHOW_CATALOGS: &[CommandColumn] = &[c("catalog")];
/// `SHOW CURRENT NAMESPACE`.
pub(crate) const SHOW_CURRENT_NAMESPACE: &[CommandColumn] = &[c("catalog"), c("namespace")];

/// `SHOW GRANTS` (the `grants` provider, `grants/exec.rs`): an ownership row has no grantor or time.
pub(crate) const SHOW_GRANTS: &[CommandColumn] = &[
    c("principal"),
    c("privilege"),
    c("object_type"),
    c("object_key"),
    c("grant_option"),
    CommandColumn::nullable("granted_by"),
    CommandColumn::nullable("granted_at"),
];
/// `DESCRIBE TABLE` (no column) and `DESCRIBE QUERY`: `comment` is nullable.
pub(crate) const DESCRIBE_TABLE: &[CommandColumn] = &[
    c("col_name"),
    c("data_type"),
    CommandColumn::nullable("comment"),
];
/// `DESCRIBE TABLE .. AS JSON`.
pub(crate) const DESCRIBE_JSON: &[CommandColumn] = &[c("json_metadata")];
/// `DESCRIBE TABLE t col`: both non-null.
pub(crate) const DESCRIBE_COLUMN: &[CommandColumn] = &[c("info_name"), c("info_value")];
/// `DESCRIBE NAMESPACE`: `info_value` is nullable.
pub(crate) const DESCRIBE_NAMESPACE: &[CommandColumn] =
    &[c("info_name"), CommandColumn::nullable("info_value")];
/// `EXPLAIN`: `plan` is nullable (`commands.scala:166`).
pub(crate) const EXPLAIN: &[CommandColumn] = &[CommandColumn::nullable("plan")];
/// `SET`, `SET key`, `SET key = value`, `SET TIME ZONE`.
pub(crate) const SET: &[CommandColumn] = &[c("key"), c("value")];
/// `SET -v`.
pub(crate) const SET_VERBOSE: &[CommandColumn] =
    &[c("key"), c("value"), c("meaning"), c("Since version")];

/// The result columns of a `SHOW` (`SHOW GRANTS` too), `DESCRIBE`, `EXPLAIN`, `SET` or `RESET`
/// statement; `None` for any other statement.
pub(crate) fn command_schema(statement: &Statement) -> Option<Vec<OutputColumn>> {
    let columns: &[CommandColumn] = match statement {
        Statement::Show(s) => show_columns(&s.kind)?,
        Statement::Describe(d) => describe_columns(&d.kind)?,
        Statement::Explain(_) => EXPLAIN,
        Statement::Set(s) => set_columns(&s.kind)?,
        Statement::Reset(_) => &[],
        // `SHOW GRANTS` (review #803 n3); the other access-control statements return no rows.
        Statement::AccessControl(ac) => match ac.command {
            krishiv_plan::analyzer::access_control::AccessCommand::ShowGrants { .. } => SHOW_GRANTS,
            _ => return None,
        },
        _ => return None,
    };
    Some(
        columns
            .iter()
            .map(|c| OutputColumn::new(c.name, c.data_type(), c.nullable))
            .collect(),
    )
}

/// `SET`'s columns (`Session::exec_set`).
fn set_columns(kind: &SetKind) -> Option<&'static [CommandColumn]> {
    Some(match kind {
        SetKind::All | SetKind::Show { .. } | SetKind::Assign { .. } | SetKind::TimeZone { .. } => {
            SET
        }
        SetKind::Verbose => SET_VERBOSE,
        SetKind::Path { .. } => &[],
        // `SET` forms added later.
        _ => return None,
    })
}

fn show_columns(kind: &ShowKind) -> Option<&'static [CommandColumn]> {
    Some(match kind {
        ShowKind::Tables { as_json: true, .. }
        | ShowKind::TableExtended { as_json: true, .. }
        | ShowKind::Partitions { as_json: true, .. } => DESCRIBE_JSON,
        ShowKind::Tables { .. } => SHOW_TABLES,
        ShowKind::TableExtended { .. } => SHOW_TABLE_EXTENDED,
        ShowKind::Views { .. } => SHOW_VIEWS,
        ShowKind::Columns { .. } => SHOW_COLUMNS,
        ShowKind::Namespaces { .. } => SHOW_NAMESPACES,
        ShowKind::CreateTable { .. } => SHOW_CREATE_TABLE,
        ShowKind::TblProperties { .. } => SHOW_TBLPROPERTIES,
        ShowKind::Partitions { .. } => SHOW_PARTITIONS,
        ShowKind::Catalogs { .. } => SHOW_CATALOGS,
        ShowKind::CurrentNamespace { .. } => SHOW_CURRENT_NAMESPACE,
        ShowKind::Functions { .. } => SHOW_FUNCTIONS,
        // Kinds added later have no result schema here.
        _ => return None,
    })
}

/// `DESCRIBE`'s columns.
fn describe_columns(kind: &DescribeKind) -> Option<&'static [CommandColumn]> {
    Some(match kind {
        DescribeKind::Table {
            as_json, column, ..
        } => {
            if *as_json {
                DESCRIBE_JSON
            } else if column.is_empty() {
                DESCRIBE_TABLE
            } else {
                DESCRIBE_COLUMN
            }
        }
        DescribeKind::Namespace { .. } => DESCRIBE_NAMESPACE,
        DescribeKind::Query { .. } => DESCRIBE_TABLE,
        DescribeKind::Function { .. } => DESCRIBE_FUNCTION,
        // Kinds added later.
        _ => return None,
    })
}
