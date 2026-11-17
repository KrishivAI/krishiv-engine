// SPDX-License-Identifier: Apache-2.0
//! Partition advice from the persisted history (P3-T025, ADR 0020 layer 1): the finished queries of
//! `krishiv.history.path` say which tables were scanned and which columns were compared with values, the
//! profile's `scans` member carries the table statistics known then, and [`krishiv_optimizer::advise`] ranks the
//! columns worth partitioning by. Read-only and off the planning path: only `EXPLAIN` and
//! `system.partition_advice` read the history, and nothing here runs DDL.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use krishiv_exec::Executor;
use krishiv_exec::profile::QueryProfile;
use krishiv_optimizer::{AdvisedColumn, AdvisedTable, PartitionAdvice, advise};

use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Float64Array, Int32Array};

use crate::profile_history::{ProfileHistory, read_history};
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
            let entry = seen.entry(scan.table.to_lowercase()).or_default();
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
fn section_of(advice: &[TableAdvice], tables: &[String]) -> Option<String> {
    let lines: Vec<String> = advice
        .iter()
        .filter(|t| tables.iter().any(|n| n.eq_ignore_ascii_case(&t.table)))
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
    let advice = match sources.monitor.sees_all() {
        true => sources.advice.get(),
        false => &[],
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

/// The history's advice, read once per query and off the executor: the read is file and Parquet I/O.
#[derive(Debug)]
pub(crate) struct AdviceCell {
    history: Option<Arc<ProfileHistory>>,
    executor: Arc<dyn Executor>,
    loaded: OnceLock<Vec<TableAdvice>>,
}

impl AdviceCell {
    pub(crate) fn new(history: Option<Arc<ProfileHistory>>, executor: Arc<dyn Executor>) -> Self {
        Self {
            history,
            executor,
            loaded: OnceLock::new(),
        }
    }

    /// Reads the history on the blocking pool, once; a read that cannot run leaves no advice.
    pub(crate) async fn load(&self) {
        if self.loaded.get().is_some() {
            return;
        }
        let advice = match &self.history {
            Some(history) => {
                let dir = history.dir().to_path_buf();
                let (tx, rx) = tokio::sync::oneshot::channel();
                let ran = self
                    .executor
                    .spawn_blocking(Box::new(move || drop(tx.send(from_history(&dir)))))
                    .await;
                match (ran, rx.await) {
                    (Ok(()), Ok(advice)) => advice,
                    _ => Vec::new(),
                }
            }
            None => Vec::new(),
        };
        let _ = self.loaded.set(advice);
    }

    /// The advice [`Self::load`] read (none before it ran).
    pub(crate) fn get(&self) -> &[TableAdvice] {
        self.loaded.get().map_or(&[], Vec::as_slice)
    }
}

impl crate::session::Session {
    /// The `== Partition Advice ==` section of `EXPLAIN` for a query that scans `tables`, when the runtime
    /// keeps a history and it has advice for one of them. The history spans principals, so the section is
    /// for a caller with `Monitor` on `system`, as `system.partition_advice` is.
    pub(crate) async fn partition_advice_section(&self, tables: &[String]) -> Option<String> {
        if tables.is_empty() {
            return None;
        }
        let history = self.state.runtime.profile_history.as_ref()?;
        let monitor = crate::system::Monitor::new(
            Arc::clone(&self.state.principal),
            Arc::clone(&self.state.policy),
        );
        monitor.decide().await;
        if !monitor.sees_all() {
            return None;
        }
        let cell = AdviceCell::new(
            Some(Arc::clone(history)),
            Arc::clone(&self.state.runtime.executor),
        );
        cell.load().await;
        section_of(cell.get(), tables)
    }
}

#[cfg(test)]
mod tests;
