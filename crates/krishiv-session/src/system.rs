// SPDX-License-Identifier: Apache-2.0
//! The `system` catalog (P2-T126 unit 1): read-only tables about the engine
//! itself, addressed `system.<table>` (empty namespace, ADR 0035).
//!
//! In-process there is one worker, so a system table has one split: each
//! query builds the rows it reads from the process's state when its scan is
//! planned, and the table is a snapshot, never an update. `system` is a
//! reserved catalog name (`KRISHIV_RESERVED_NAME`): nothing can be created in
//! it, and it is no database.
//!
//! This unit holds `system.capabilities` and `system.memory`; the query,
//! operator, cache and metric tables arrive with their recorders (units 2
//! and 3).

use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::planner::MemoryTable;
use krishiv_memory::MemoryPool;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

/// The catalog's name.
pub(crate) const SYSTEM: &str = "system";

/// The schema version of every table in this unit (`krishiv.schema.version`).
const SCHEMA_VERSION: &str = "1";

/// The worker this process is, in the `worker_id` column.
const LOCAL_WORKER: &str = "local";

/// Every table of the catalog.
const TABLES: [&str; 2] = ["capabilities", "memory"];

/// Whether `name` is the reserved catalog name, in any letter case.
pub(crate) fn is_system(name: &str) -> bool {
    name.eq_ignore_ascii_case(SYSTEM)
}

/// Whether a scan of `name` reads a system table: `system.<table>`.
pub(crate) fn is_system_table(name: &[String]) -> bool {
    matches!(name, [catalog, table] if is_system(catalog) && table_known(table))
}

fn table_known(table: &str) -> bool {
    TABLES.iter().any(|t| t.eq_ignore_ascii_case(table))
}

/// The cargo features this build was made with and whether each is on, set
/// once by the binary (the library cannot see its dependents' features).
static BUILD_FEATURES: OnceLock<Vec<(String, bool)>> = OnceLock::new();

/// Records the cargo features of the running binary for
/// `system.capabilities`: `(name, enabled)` for every feature that can be
/// switched. The first call wins; later calls change nothing.
pub fn register_build_features(features: &[(&str, bool)]) {
    let owned = features
        .iter()
        .map(|(name, on)| ((*name).to_owned(), *on))
        .collect();
    let _ = BUILD_FEATURES.set(owned);
}

/// The rows of `system.capabilities`: `(kind, name, value, detail)`.
fn capability_rows() -> Vec<[String; 4]> {
    let row = |kind: &str, name: &str, value: &str, detail: &str| {
        [
            kind.to_owned(),
            name.to_owned(),
            value.to_owned(),
            detail.to_owned(),
        ]
    };
    let state = |on: bool| if on { "enabled" } else { "disabled" };
    let mut rows = vec![
        row("build", "engine", "krishiv", env!("CARGO_PKG_VERSION")),
        row("build", "spark_versions", "3.5,4.0,4.1,4.2", ""),
        row("surface", "system_tables", SCHEMA_VERSION, ""),
        row("surface", "agent_result", "1", ""),
        row("feature", "cloud", state(cfg!(feature = "cloud")), ""),
    ];
    if let Some(features) = BUILD_FEATURES.get() {
        let mut sorted: Vec<&(String, bool)> = features.iter().collect();
        sorted.sort();
        rows.extend(
            sorted
                .into_iter()
                .filter(|(name, _)| name != "cloud")
                .map(|(name, on)| row("feature", name, state(*on), "")),
        );
    }
    rows
}

/// What a column holds.
#[derive(Clone, Copy)]
enum Kind {
    Text,
    Long,
    Flag,
}

/// A table's columns: `(name, kind, nullable)`.
fn layout(table: &str) -> Option<Vec<(&'static str, Kind, bool)>> {
    use Kind::{Flag, Long, Text};
    Some(match table.to_ascii_lowercase().as_str() {
        "capabilities" => vec![
            ("worker_id", Text, false),
            ("kind", Text, false),
            ("name", Text, false),
            ("value", Text, false),
            ("detail", Text, true),
        ],
        "memory" => vec![
            ("worker_id", Text, false),
            ("path", Text, false),
            ("parent", Text, true),
            ("kind", Text, false),
            ("query_id", Text, true),
            ("reserved_bytes", Long, false),
            ("peak_bytes", Long, false),
            ("limit_bytes", Long, true),
            ("spillable", Flag, false),
        ],
        _ => return None,
    })
}

/// The columns of `table`, `None` for an unknown table.
fn columns(table: &str) -> Option<Vec<TableColumn>> {
    Some(
        layout(table)?
            .into_iter()
            .map(|(name, kind, nullable)| TableColumn {
                name: name.to_owned(),
                data_type: match kind {
                    Kind::Text => DataType::String,
                    Kind::Long => DataType::Long,
                    Kind::Flag => DataType::Boolean,
                },
                nullable,
            })
            .collect(),
    )
}

/// Rows of text cells, column by column.
fn text<'a>(cells: impl Iterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(StringArray::from_iter(cells))
}

fn long(cells: impl Iterator<Item = Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from_iter(cells))
}

/// A byte count as `BIGINT`, saturating.
fn bytes(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// The columns' values for `table` now.
fn values(table: &str, sources: &Sources) -> Vec<ArrayRef> {
    if table.eq_ignore_ascii_case("memory") {
        let entries = sources.memory.snapshot().entries;
        let worker = vec![Some(LOCAL_WORKER); entries.len()];
        // `process/query-<id>/...`: the query of a pool or consumer below one.
        let query = |path: &str| {
            path.split('/')
                .nth(1)
                .and_then(|p| p.strip_prefix("query-"))
                .map(str::to_owned)
        };
        let queries: Vec<Option<String>> = entries.iter().map(|e| query(&e.path)).collect();
        return vec![
            text(worker.into_iter()),
            text(entries.iter().map(|e| Some(e.path.as_str()))),
            text(entries.iter().map(|e| e.parent.as_deref())),
            text(
                entries
                    .iter()
                    .map(|e| Some(if e.is_consumer { "consumer" } else { "pool" })),
            ),
            text(queries.iter().map(Option::as_deref)),
            long(entries.iter().map(|e| Some(bytes(e.reserved)))),
            long(entries.iter().map(|e| Some(bytes(e.peak)))),
            long(entries.iter().map(|e| e.limit.map(bytes))),
            Arc::new(BooleanArray::from_iter(
                entries.iter().map(|e| Some(e.spillable)),
            )),
        ];
    }
    let rows = capability_rows();
    vec![
        text(rows.iter().map(|_| Some(LOCAL_WORKER))),
        text(rows.iter().map(|r| Some(r[0].as_str()))),
        text(rows.iter().map(|r| Some(r[1].as_str()))),
        text(rows.iter().map(|r| Some(r[2].as_str()))),
        text(rows.iter().map(|r| Some(r[3].as_str()))),
    ]
}

/// What the tables read: the process's state.
pub(crate) struct Sources {
    /// The session's memory pool tree.
    pub(crate) memory: Arc<dyn MemoryPool>,
}

/// The table `table` as the planner reads it: its rows now.
pub(crate) fn provider(
    table: &str,
    sources: &Sources,
) -> Result<Option<Arc<MemoryTable>>, KrishivError> {
    let Some(layout) = layout(table) else {
        return Ok(None);
    };
    let plain = Arc::new(Schema::new(
        layout
            .iter()
            .map(|(name, kind, nullable)| {
                let ty = match kind {
                    Kind::Text => ArrowType::Utf8,
                    Kind::Long => ArrowType::Int64,
                    Kind::Flag => ArrowType::Boolean,
                };
                Field::new(*name, ty, *nullable)
            })
            .collect::<Vec<_>>(),
    ));
    let schema = crate::carrier::canonical_schema(Arc::clone(&plain));
    let batch = RecordBatch::try_new(plain, values(table, sources)).map_err(|e| InternalError {
        message: format!("system.{table}: {e}"),
    })?;
    let batch = crate::carrier::recarried(&batch, &schema)?;
    Ok(Some(Arc::new(MemoryTable::new(
        schema,
        vec![Batch::new(batch)],
    ))))
}

/// The `system` catalog: metadata only; the rows come from [`provider`].
#[derive(Debug)]
pub(crate) struct SystemCatalog;

impl Catalog for SystemCatalog {
    fn name(&self) -> &str {
        SYSTEM
    }

    fn epoch(&self) -> u64 {
        0
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _: Option<TableVersion>,
        _: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            // Two-part names only: `system.<table>`, with no namespace.
            if !id.namespace.is_empty() {
                return Ok(None);
            }
            Ok(columns(&id.name).map(|columns| TableMeta {
                ident: TableIdent {
                    catalog: SYSTEM.to_owned(),
                    namespace: Vec::new(),
                    name: id.name.to_lowercase(),
                },
                columns,
                version: None,
                kind: TableKind::Table,
                file_metadata: Default::default(),
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_has_one_row_per_fact() {
        let rows = capability_rows();
        assert!(rows.iter().any(|r| r[0] == "build" && r[1] == "engine"));
        assert!(rows.iter().any(|r| r[0] == "feature" && r[1] == "cloud"));
        let sources = Sources {
            memory: krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default()),
        };
        assert!(provider("capabilities", &sources).unwrap().is_some());
        assert!(provider("nope", &sources).unwrap().is_none());
    }

    #[test]
    fn names_are_matched_without_case() {
        assert!(is_system("SYSTEM"));
        assert!(is_system_table(&[
            "System".to_owned(),
            "Capabilities".to_owned()
        ]));
        assert!(!is_system_table(&["system".to_owned(), "x".to_owned()]));
        assert!(!is_system_table(&["system".to_owned()]));
    }
}
