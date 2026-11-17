// SPDX-License-Identifier: Apache-2.0
//! Partition advice from the persisted history (P3-T025, ADR 0020 layer 1): the finished queries of
//! `krishiv.history.path` say which tables were scanned and which columns were compared with values, the
//! profile's `scans` member carries the table statistics known then, and [`krishiv_optimizer::advise`] ranks the
//! columns worth partitioning by. Read-only and off the planning path: only `EXPLAIN` and
//! `system.partition_advice` read the history, and nothing here runs DDL.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use krishiv_exec::profile::QueryProfile;
use krishiv_optimizer::{AdvisedColumn, AdvisedTable, PartitionAdvice, advise};

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array};

use crate::profile_history::read_history;
use crate::system::{Kind, Sources, long, text};

/// The ranked suggestions for one table.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TableAdvice {
    /// The table, as `catalog.schema.name`.
    pub(crate) table: String,
    /// Finished queries that scanned it.
    pub(crate) queries: u64,
    /// The suggestions, best first (never empty).
    pub(crate) advice: Vec<PartitionAdvice>,
}

/// What the history says of one table.
#[derive(Default)]
struct Seen {
    queries: u64,
    /// Queries that compared each column with values.
    filters: HashMap<String, u64>,
    /// The latest row count and distinct counts a query planned with.
    rows: Option<u64>,
    distinct: BTreeMap<String, u64>,
}

/// The advice for every table the history under `dir` has finished queries on, by table name. A history that
/// does not read gives none; a profile that does not parse (a later major version) is skipped.
pub(crate) fn from_history(dir: &Path) -> Vec<TableAdvice> {
    let Ok(rows) = read_history(dir) else {
        return Vec::new();
    };
    let mut seen: BTreeMap<String, Seen> = BTreeMap::new();
    // Oldest first, so the statistics kept are the latest.
    for row in rows
        .iter()
        .filter(|r| r.state == "finished" && r.error_class.is_none())
    {
        let Ok(profile) = QueryProfile::from_json(&row.profile) else {
            continue;
        };
        for scan in profile.scans {
            let entry = seen.entry(scan.table).or_default();
            entry.queries += 1;
            if scan.rows.is_some() {
                entry.rows = scan.rows;
            }
            for c in scan.equality {
                *entry.filters.entry(c.name.clone()).or_default() += 1;
                if let Some(d) = c.distinct {
                    entry.distinct.insert(c.name, d);
                }
            }
        }
    }
    seen.into_iter()
        .filter_map(|(table, s)| {
            let table_stats = AdvisedTable {
                rows: s.rows?,
                columns: s
                    .distinct
                    .iter()
                    .map(|(name, d)| AdvisedColumn {
                        name: name.clone(),
                        distinct: Some(*d),
                    })
                    .collect(),
            };
            let advice = advise(&table_stats, &s.filters, s.queries);
            (!advice.is_empty()).then_some(TableAdvice {
                table,
                queries: s.queries,
                advice,
            })
        })
        .collect()
}

/// The `EXPLAIN` section for `tables`: one `partition advice` line per suggestion, the best first; `None` when
/// there is nothing to advise.
pub(crate) fn explain_section(dir: &Path, tables: &[String]) -> Option<String> {
    let lines: Vec<String> = from_history(dir)
        .into_iter()
        .filter(|t| tables.contains(&t.table))
        .flat_map(|t| {
            t.advice
                .iter()
                .map(|a| format!("{}: {}", t.table, a.explain_line()))
                .collect::<Vec<_>>()
        })
        .collect();
    (!lines.is_empty()).then(|| format!("== Partition Advice ==\n{}\n", lines.join("\n")))
}

/// The `system.partition_advice` rows: one per suggestion, the best of each table first. The history holds every
/// principal's queries, so only a caller with `Monitor` on `system` reads them.
pub(crate) fn columns(sources: &Sources) -> Vec<ArrayRef> {
    let advice = match (&sources.history, sources.monitor.sees_all()) {
        (Some(history), true) => from_history(history.dir()),
        _ => Vec::new(),
    };
    let rows: Vec<(&TableAdvice, i32, &PartitionAdvice)> = advice
        .iter()
        .flat_map(|t| t.advice.iter().zip(1..).map(move |(a, rank)| (t, rank, a)))
        .collect();
    let count = |n: u64| Some(i64::try_from(n).unwrap_or(i64::MAX));
    vec![
        text(rows.iter().map(|(t, _, _)| Some(t.table.as_str()))),
        Arc::new(Int32Array::from_iter(rows.iter().map(|(_, r, _)| Some(*r)))),
        text(rows.iter().map(|(_, _, a)| Some(a.column.as_str()))),
        long(rows.iter().map(|(t, _, _)| count(t.queries))),
        long(rows.iter().map(|(_, _, a)| count(a.queries))),
        long(rows.iter().map(|(_, _, a)| count(a.distinct))),
        Arc::new(Float64Array::from_iter(
            rows.iter().map(|(_, _, a)| Some(a.pruned_fraction)),
        )),
    ]
}

pub(crate) fn layout() -> Vec<(&'static str, Kind, bool)> {
    use Kind::{Double, Int, Long, Text};
    vec![
        ("table_name", Text, false),
        ("rank", Int, false),
        ("column_name", Text, false),
        ("table_queries", Long, false),
        ("column_queries", Long, false),
        ("partitions", Long, false),
        ("pruned_fraction", Double, false),
    ]
}

impl crate::session::Session {
    /// The `== Partition Advice ==` section of `EXPLAIN` for a query that scans `tables`, when the runtime
    /// keeps a history and it has advice for one of them.
    pub(crate) fn partition_advice_section(&self, tables: &[String]) -> Option<String> {
        let history = self.state.runtime.profile_history.as_ref()?;
        explain_section(history.dir(), tables)
    }
}

#[cfg(test)]
mod tests;
