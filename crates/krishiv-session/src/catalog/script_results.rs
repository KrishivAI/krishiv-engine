// SPDX-License-Identifier: Apache-2.0
//! The hidden tables that hold a SQL script's result (P2b row 15, R4): a frame reads one by a name no
//! SQL text spells, and it goes once nothing but the catalog holds it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, PoisonError};

use krishiv_exec::planner::MemoryTable;
use krishiv_plan::analyzer::TableColumn;

use super::{SessionCatalog, TempTable, is_hidden_name};

/// The start (in lower case) of the hidden names of script results: a kind of [`HIDDEN_PREFIX`] name
/// that no view pins and that lives as long as a frame reads it.
const SCRIPT_RESULT_PREFIX: &str = "__krishiv_snapshot_r";

impl SessionCatalog {
    /// Adds the hidden in-memory table `table` and returns its name, which no SQL text spells. It
    /// is removed once nothing but the catalog holds it ([`Self::collect_garbage`]), at the next
    /// change of the catalog.
    pub(crate) fn put_hidden_memory(
        &self,
        columns: Vec<TableColumn>,
        table: Arc<MemoryTable>,
    ) -> (String, Arc<TempTable>) {
        let name = format!(
            "{SCRIPT_RESULT_PREFIX}{}",
            self.hidden.fetch_add(1, Ordering::AcqRel)
        );
        let entry = Arc::new(TempTable::Memory { columns, table });
        let mut tables = self.tables.write().unwrap_or_else(PoisonError::into_inner);
        tables.insert(name.to_lowercase(), (name.clone(), Arc::clone(&entry)));
        // The caller's reference keeps the new entry through this collection.
        Self::collect_garbage(&mut tables);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        (name, entry)
    }
}

/// Drops the hidden entries nothing needs: those no live view pins (`live`) and, of the script
/// results, those nothing but the catalog holds: a result frame, and every frame derived from it
/// (`DataFrame::keep`), owns its table.
pub(super) fn retain(
    tables: &mut BTreeMap<String, (String, Arc<TempTable>)>,
    live: &BTreeSet<String>,
) {
    tables.retain(|key, (_, table)| {
        !is_hidden_name(key)
            || live.contains(key)
            || (key.starts_with(SCRIPT_RESULT_PREFIX) && Arc::strong_count(table) > 1)
    });
}

/// Drops the script results no view in `tables` pins (a copy of another catalog's map): the copy
/// has no frame that reads them, and keeping them would hold each result alive in both catalogs.
pub(super) fn leave_out(tables: &mut BTreeMap<String, (String, Arc<TempTable>)>) {
    let mut live = BTreeSet::new();
    for (_, table) in tables.values() {
        if let TempTable::View { pins, .. } = &**table {
            live.extend(pins.iter().cloned());
        }
    }
    tables.retain(|key, _| !key.starts_with(SCRIPT_RESULT_PREFIX) || live.contains(key));
    // A kept result gets a handle of the copy's own (the batches stay shared): `retain` counts
    // handles, so a handle shared with the source would keep the entry in both catalogs after each
    // drops the view that pins it.
    for (key, (_, table)) in tables.iter_mut() {
        if key.starts_with(SCRIPT_RESULT_PREFIX)
            && let TempTable::Memory {
                columns,
                table: data,
            } = &**table
        {
            *table = Arc::new(TempTable::Memory {
                columns: columns.clone(),
                table: Arc::clone(data),
            });
        }
    }
}
