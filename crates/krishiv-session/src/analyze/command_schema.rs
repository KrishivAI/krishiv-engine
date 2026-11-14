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
//! Provisional nullability: the commands build their frames from literal rows, so the run's nullability
//! depends on the rows; analyze reports every command column as nullable. Spark's command attributes are
//! not checked yet (a spark-semantics follow-up pins them at the Spark tags).

use krishiv_sql::ast::{DescribeKind, SetKind, ShowKind, Statement};
use krishiv_types::DataType;

use super::OutputColumn;

/// The result columns of a `SHOW`, `DESCRIBE`, `EXPLAIN`, `SET` or `RESET` statement; `None` for any other
/// statement.
pub(crate) fn command_schema(statement: &Statement) -> Option<Vec<OutputColumn>> {
    let names: &[&str] = match statement {
        Statement::Show(s) => show_columns(&s.kind)?,
        Statement::Describe(d) => describe_columns(&d.kind)?,
        Statement::Explain(_) => &["plan"],
        Statement::Set(s) => set_columns(&s.kind)?,
        Statement::Reset(_) => &[],
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

/// `SET`'s columns (`Session::exec_set`).
fn set_columns(kind: &SetKind) -> Option<&'static [&'static str]> {
    Some(match kind {
        SetKind::All | SetKind::Show { .. } | SetKind::Assign { .. } | SetKind::TimeZone { .. } => {
            &["key", "value"]
        }
        SetKind::Verbose => &["key", "value", "meaning", "Since version"],
        SetKind::Path { .. } => &[],
        // `SET` forms added later.
        _ => return None,
    })
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
