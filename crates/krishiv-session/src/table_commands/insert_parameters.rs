// SPDX-License-Identifier: Apache-2.0
//! The types of the markers of an `INSERT ... VALUES` that are whole cells: their target
//! columns' (P3-T101 note §4.4; [`crate::parameter_types`]).

use super::*;
use krishiv_sql::ast::{Expr, ExprKind, SetExprKind, Statement};

impl Session {
    /// For each of the `markers` positional markers of `sql`: the type of the target column a
    /// marker that is a whole cell of a single-table `INSERT INTO|OVERWRITE t [(columns)] VALUES`
    /// (no `PARTITION`) is assigned to, as the `INSERT` assigns it (by position or by its column
    /// list); `None` for any other marker or statement. The types are read only when the caller
    /// may run the `INSERT` (its target check, asked quietly), so they say nothing about a table
    /// it may not write.
    #[cfg(feature = "unstable-ir")]
    pub async fn insert_parameter_types(&self, sql: &str, markers: usize) -> Vec<Option<DataType>> {
        let mut out = vec![None; markers];
        let env = self.query_env();
        let Ok(statement) = krishiv_sql::parse_statement(sql, &env.parser) else {
            return out;
        };
        let Statement::Insert(insert) = &statement else {
            return out;
        };
        let [c] = insert.clauses.as_slice() else {
            return out;
        };
        if insert.source.is_some()
            || c.partition.is_some()
            || c.replace_where.is_some()
            || c.replace_on.is_some()
            || c.replace_using.is_some()
        {
            return out;
        }
        let SetExprKind::Values(values) = &c.query.body.kind else {
            return out;
        };
        let rows: Vec<Vec<&Expr>> = values.rows.iter().map(cells).collect();
        let width = rows.first().map_or(0, Vec::len);
        if width == 0 || rows.iter().any(|r| r.len() != width) {
            return out;
        }
        let Ok(listed) = c
            .columns
            .iter()
            .map(|n| match n {
                ColumnName::Ident(i) => Ok(i.value.clone()),
                _ => Err(()),
            })
            .collect::<Result<Vec<_>, _>>()
        else {
            return out;
        };
        if !self.targets_allowed_quietly(&statement, &env).await {
            return out;
        }
        let Ok(name) = self.table_name(&c.name, &env) else {
            return out;
        };
        let Ok(table) = self.table_for_write(&name, "INSERT", &env) else {
            return out;
        };
        // `VALUES` names its columns `col1`, `col2`, ...; only positions matter here.
        let output: Vec<TableColumn> = (1..=width)
            .map(|j| TableColumn::new(format!("col{j}"), DataType::Null, true))
            .collect();
        let Ok(assignment) = assign(
            &table,
            &listed,
            c.by_name,
            &output,
            env.analysis.conf.case_sensitive,
            false,
        ) else {
            return out;
        };
        let mut by_position: Vec<Option<DataType>> = vec![None; width];
        for (column, from) in table.columns.iter().zip(&assignment) {
            if let Source::Position(j) = from
                && let Some(slot) = by_position.get_mut(*j)
            {
                *slot = Some(column.data_type.clone());
            }
        }
        for row in &rows {
            for (j, cell) in row.iter().enumerate() {
                if let ExprKind::Parameter(k) = cell.kind
                    && let Some(slot) = out.get_mut(k)
                {
                    slot.clone_from(&by_position[j]);
                }
            }
        }
        out
    }
}

/// The cells of one `VALUES` row: a row constructor's items, else the row itself (parentheses
/// around a single value are not a constructor).
fn cells(row: &Expr) -> Vec<&Expr> {
    let mut row = row;
    while let ExprKind::Nested(inner) = &row.kind {
        row = inner;
    }
    match &row.kind {
        ExprKind::Row(items) => items.iter().map(|i| unnest(&i.expr)).collect(),
        _ => vec![row],
    }
}

/// `e` without the parentheses around it.
fn unnest(e: &Expr) -> &Expr {
    let mut e = e;
    while let ExprKind::Nested(inner) = &e.kind {
        e = inner;
    }
    e
}
