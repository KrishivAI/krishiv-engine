// SPDX-License-Identifier: Apache-2.0
//! The partition advisor over the persisted query history (P3-T025, ADR 0020 layer 1): which columns of
//! which tables the finished queries filtered by equality or `IN`, and what the tables' statistics were
//! when they ran, ranked by `krishiv_optimizer::advise`.
//!
//! Read-only and off the planning path: it reads the history directory (`krishiv.history.path`) when
//! `EXPLAIN` or `system.partition_advice` asks, never at planning time, and it runs no DDL. The history
//! masks literals and drops every plan line's text, and nothing here needs either: a scan's planner facts name
//! the filtered columns, never their values.
//! A table's statistics come from the planner facts each stored profile's scan node carries (minor 6), so
//! the advice is from the statistics the queries planned with.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use krishiv_exec::profile::{ProfileNode, QueryProfile};
use krishiv_optimizer::{AdvisedColumn, AdvisedTable, PartitionAdvice, advise};

use crate::profile_history::read_history;

/// The advice for one table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TableAdvice {
    /// The table as the scan names it.
    pub(crate) table: String,
    /// Stored queries that read it.
    pub(crate) queries: u64,
    /// The ranked candidates, best first.
    pub(crate) advice: Vec<PartitionAdvice>,
}

/// What one query did to one table.
#[derive(Debug, Default)]
struct ScanUse {
    rows: u64,
    distinct: BTreeMap<String, u64>,
    columns: BTreeSet<String>,
}

fn collect_uses(node: &ProfileNode, out: &mut HashMap<String, ScanUse>) {
    if node.name == "Scan"
        && let Some(facts) = node.facts.as_deref()
        && let Some(table) = &facts.table
    {
        let u = out.entry(table.name.clone()).or_default();
        u.rows = table.rows;
        u.distinct = table.distinct.iter().cloned().collect();
        // The planner recorded the columns the pushed filter compares with a constant. The history masks every
        // plan line's text, so the advisor reads these facts, never the filter's text.
        u.columns.extend(table.filter_columns.iter().cloned());
    }
    for c in &node.children {
        collect_uses(c, out);
    }
}

/// The advice for every table the stored profiles under `dir` read, ordered by table name. Profiles are read
/// oldest first, so a table's statistics are the newest query's.
pub(crate) fn advise_history(dir: &Path) -> Vec<TableAdvice> {
    let Ok(stored) = read_history(dir) else {
        return Vec::new();
    };
    let profiles = stored
        .iter()
        .filter(|s| s.state == "finished")
        .filter_map(|s| QueryProfile::from_json(&s.profile).ok());
    advise_profiles(profiles)
}

/// [`advise_history`] over profiles already read, oldest first.
pub(crate) fn advise_profiles(profiles: impl Iterator<Item = QueryProfile>) -> Vec<TableAdvice> {
    struct Acc {
        queries: u64,
        filters: HashMap<String, u64>,
        latest: ScanUse,
    }
    let mut by_table: BTreeMap<String, Acc> = BTreeMap::new();
    for p in profiles {
        let mut uses = HashMap::new();
        collect_uses(&p.plan, &mut uses);
        for (table, u) in uses {
            let acc = by_table.entry(table).or_insert_with(|| Acc {
                queries: 0,
                filters: HashMap::new(),
                latest: ScanUse::default(),
            });
            acc.queries += 1;
            for c in &u.columns {
                *acc.filters.entry(c.clone()).or_default() += 1;
            }
            acc.latest = u;
        }
    }
    by_table
        .into_iter()
        .map(|(table, acc)| {
            let t = AdvisedTable {
                rows: acc.latest.rows,
                columns: acc
                    .latest
                    .distinct
                    .iter()
                    .map(|(name, d)| AdvisedColumn {
                        name: name.clone(),
                        distinct: Some(*d),
                    })
                    .collect(),
            };
            TableAdvice {
                advice: advise(&t, &acc.filters, acc.queries),
                queries: acc.queries,
                table,
            }
        })
        .collect()
}

/// The `EXPLAIN` lines for the tables `plan` scans that have advice: `partition advice (<table>): <column>
/// (<n> queries, <d> partitions, ~<p>% of bytes pruned)`, best candidate per table.
pub(crate) fn explain_lines(
    all: &[TableAdvice],
    plan: &krishiv_exec::planner::ExplainNode,
) -> Vec<String> {
    fn tables(n: &krishiv_exec::planner::ExplainNode, out: &mut BTreeSet<String>) {
        if let Some(t) = n.facts.as_deref().and_then(|f| f.table.as_ref()) {
            out.insert(t.name.clone());
        }
        for c in &n.children {
            tables(c, out);
        }
    }
    let mut read = BTreeSet::new();
    tables(plan, &mut read);
    all.iter()
        .filter(|a| read.contains(&a.table))
        .filter_map(|a| {
            let best = a.advice.first()?;
            let line = best.explain_line();
            let rest = line.strip_prefix("partition advice: ").unwrap_or(&line);
            Some(format!("partition advice ({}): {rest}", a.table))
        })
        .collect()
}

#[cfg(test)]
mod tests;
