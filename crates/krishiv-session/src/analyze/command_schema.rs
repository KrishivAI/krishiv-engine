// SPDX-License-Identifier: Apache-2.0
//! The result schemas of the `SHOW` and `DESCRIBE` commands (design `P3-T063-analyze-api.md` §2, review m6):
//! they come from the session's command code, not from analysis, so they are listed here per command, as
//! Spark 4.2's goldens print them (`-- !query schema` lines) and the commands return them.
//!
//! `tests/analyze_commands.rs` runs one statement per command kind and compares the names and types with the
//! frame the command returns, and reads the `ShowKind`/`DescribeKind` variants from `krishiv-sql`'s source,
//! so a new command kind fails there until it is listed (the enums are `#[non_exhaustive]` in another crate,
//! so the match below needs a wildcard arm).
//!
//! Nullability: the commands build their frames from literal rows, so the run's nullability depends on the
//! rows; analyze reports every command column as nullable (Spark's command attributes are not checked here).

use krishiv_sql::ast::{DescribeKind, ShowKind, Statement};
use krishiv_types::DataType;

use super::OutputColumn;

/// The result columns of a `SHOW` or `DESCRIBE` statement; `None` for any other statement.
pub(crate) fn command_schema(statement: &Statement) -> Option<Vec<OutputColumn>> {
    let names: &[&str] = match statement {
        Statement::Show(s) => show_columns(&s.kind)?,
        Statement::Describe(d) => describe_columns(&d.kind)?,
        _ => return None,
    };
    Some(
        names
            .iter()
            .map(|&n| {
                let ty = if n == "isTemporary" {
                    DataType::Boolean
                } else {
                    DataType::String
                };
                OutputColumn::new(n, ty, true)
            })
            .collect(),
    )
}

const TABLES: &[&str] = &["namespace", "tableName", "isTemporary"];
const DESCRIBE: &[&str] = &["col_name", "data_type", "comment"];
const INFO: &[&str] = &["info_name", "info_value"];

fn show_columns(kind: &ShowKind) -> Option<&'static [&'static str]> {
    Some(match kind {
        ShowKind::Tables { .. } => TABLES,
        ShowKind::TableExtended { .. } => &["namespace", "tableName", "isTemporary", "information"],
        ShowKind::Views { .. } => &["namespace", "viewName", "isTemporary"],
        ShowKind::Columns { .. } => &["col_name"],
        ShowKind::Namespaces { .. } => &["namespace"],
        ShowKind::CreateTable { .. } => &["createtab_stmt"],
        ShowKind::TblProperties { .. } => &["key", "value"],
        ShowKind::Partitions { .. } => &["partition"],
        ShowKind::Catalogs { .. } => &["catalog"],
        ShowKind::CurrentNamespace { .. } => &["catalog", "namespace"],
        // `SHOW FUNCTIONS` (refused by the session, `KRISHIV_UNSUPPORTED_FEATURE`) and kinds added later have no
        // result schema here.
        _ => return None,
    })
}

fn describe_columns(kind: &DescribeKind) -> Option<&'static [&'static str]> {
    Some(match kind {
        DescribeKind::Table {
            as_json, column, ..
        } => {
            if *as_json {
                &["json_metadata"]
            } else if column.is_empty() {
                DESCRIBE
            } else {
                INFO
            }
        }
        DescribeKind::Namespace { .. } => INFO,
        DescribeKind::Query { .. } => DESCRIBE,
        // `DESCRIBE FUNCTION` (refused by the session, `KRISHIV_UNSUPPORTED_FEATURE`) and kinds added later.
        _ => return None,
    })
}
