// SPDX-License-Identifier: Apache-2.0
//! The server console's Advisor page: the partition columns the query history suggests for each table
//! (`system.partition_advice`, P3-T025), the same suggestions `EXPLAIN` prints.
//!
//! The advice is folded from every principal's finished queries, so it is a server's page: a console over
//! one session and one over stored profiles list nothing. Reading it reads the history store, so the page
//! asks once and on request, not on a timer. Nothing here changes a table.

use std::collections::HashMap;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::{Value, json};

use crate::internal::{Row, SystemTable, UiState, int, json_response, pick, table, text};

/// Tables one answer lists; the table's own order is kept, so the cut is its tail.
const MAX_TABLES: usize = 500;

/// What the page gets of one suggestion (the table's name and its query count are the group's).
const COLUMN_COLUMNS: &[&str] = &[
    "rank",
    "column_name",
    "column_queries",
    "partitions",
    "pruned_fraction",
];

/// The suggestions grouped by table in the order the table lists them (the best suggestion of each table
/// first), each table's by rank.
fn by_table(rows: &[Row]) -> Vec<Value> {
    let mut tables: Vec<(&str, i64, Vec<&Row>)> = Vec::new();
    // Where each table is in `tables`: a history may cover many thousands of tables.
    let mut at: HashMap<&str, usize> = HashMap::new();
    for row in rows {
        let name = text(row, "table_name");
        match at.get(name).and_then(|i| tables.get_mut(*i)) {
            Some((.., columns)) => columns.push(row),
            None => {
                at.insert(name, tables.len());
                tables.push((name, int(row, "table_queries"), vec![row]));
            }
        }
    }
    tables
        .into_iter()
        .map(|(name, queries, mut columns)| {
            columns.sort_by_key(|r| int(r, "rank"));
            json!({
                "table_name": name,
                "table_queries": queries,
                "columns": columns
                    .into_iter()
                    .map(|r| Value::Object(pick(r, COLUMN_COLUMNS)))
                    .collect::<Vec<_>>(),
            })
        })
        .collect()
}

/// `GET /api/internal/advice`: the partition advice per table; `truncated` says that more than
/// [`MAX_TABLES`] tables have advice.
pub(crate) async fn advice(State(s): State<UiState>) -> Response {
    let rows = match table(&s, SystemTable::PartitionAdvice).await {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let mut tables = by_table(&rows);
    let truncated = tables.len() > MAX_TABLES;
    tables.truncate(MAX_TABLES);
    json_response(
        StatusCode::OK,
        &json!({"tables": tables, "truncated": truncated}),
    )
}
