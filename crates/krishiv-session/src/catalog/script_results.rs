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

/// How many of the newest script results the catalog keeps whether or not a frame holds them.
const RECENT_SCRIPT_RESULTS: usize = 16;

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
/// results, all but the newest few and those something besides the catalog holds (the result
/// frame, whose analysis keeps the table it loaded; a frame derived from a result, `limit` or
/// `filter`, has no analysis yet and reads the table by its name, hence the newest few).
pub(super) fn retain(
    tables: &mut BTreeMap<String, (String, Arc<TempTable>)>,
    live: &BTreeSet<String>,
) {
    let mut results: Vec<u64> = tables
        .keys()
        .filter_map(|k| k.strip_prefix(SCRIPT_RESULT_PREFIX)?.parse().ok())
        .collect();
    results.sort_unstable();
    let newest = results
        .len()
        .checked_sub(RECENT_SCRIPT_RESULTS)
        .map_or(0, |n| results[n]);
    tables.retain(|key, (_, table)| {
        !is_hidden_name(key)
            || live.contains(key)
            || key
                .strip_prefix(SCRIPT_RESULT_PREFIX)
                .and_then(|n| n.parse::<u64>().ok())
                .is_some_and(|n| n >= newest || Arc::strong_count(table) > 1)
    });
}
