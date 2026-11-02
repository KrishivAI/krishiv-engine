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
//! This unit holds `system.capabilities`; the query, operator, memory, cache
//! and metric tables arrive with their recorders (units 2 and 3).

use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::planner::MemoryTable;
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

/// The columns of `system.capabilities` (design `P2-T126` section 3).
const CAPABILITY_COLUMNS: [&str; 5] = ["worker_id", "kind", "name", "value", "detail"];

/// Every table of the catalog.
const TABLES: [&str; 1] = ["capabilities"];

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

/// The columns of `table`, `None` for an unknown table.
fn columns(table: &str) -> Option<Vec<TableColumn>> {
    if !table.eq_ignore_ascii_case("capabilities") {
        return None;
    }
    Some(
        CAPABILITY_COLUMNS
            .iter()
            .map(|name| TableColumn {
                name: (*name).to_owned(),
                data_type: DataType::String,
                nullable: *name == "detail",
            })
            .collect(),
    )
}

/// The table `table` as the planner reads it: its rows now.
pub(crate) fn provider(table: &str) -> Result<Option<Arc<MemoryTable>>, KrishivError> {
    let Some(cols) = columns(table) else {
        return Ok(None);
    };
    let rows = capability_rows();
    let fields: Vec<Field> = cols
        .iter()
        .map(|c| Field::new(&c.name, ArrowType::Utf8, c.nullable))
        .collect();
    let schema = crate::carrier::canonical_schema(Arc::new(Schema::new(fields)));
    let worker: ArrayRef = Arc::new(StringArray::from(vec![LOCAL_WORKER; rows.len()]));
    let column = |i: usize| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r[i].as_str()),
        ))
    };
    let plain = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| Field::new(f.name(), ArrowType::Utf8, f.is_nullable()))
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(
        plain,
        vec![worker, column(0), column(1), column(2), column(3)],
    )
    .map_err(|e| InternalError {
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
        let table = provider("capabilities").unwrap().unwrap();
        let _ = table;
        assert!(provider("nope").unwrap().is_none());
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
