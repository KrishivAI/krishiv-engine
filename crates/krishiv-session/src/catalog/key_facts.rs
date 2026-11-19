// SPDX-License-Identifier: Apache-2.0
//! Plan-time key counts (design P2b A5 (d), unit 2): the optimizer asks for
//! a base relation's key to be proven unique (`Statistics::want_unique`,
//! recorded by each opened [`FileTable`](krishiv_exec::planner::FileTable));
//! with `krishiv.optimizer.discoverKeys` on, the session counts each asked
//! key once ([`QueryCatalog::discover_keys`]) before it optimizes again, and
//! `EXPLAIN VERBOSE` notes what each count found ([`QueryCatalog::key_notes`]).

use super::QueryCatalog;
use super::ndv_sampling::files_label;
use super::run_on;

impl QueryCatalog {
    /// Forgets the key counts the optimizer asked for so far (a plan
    /// optimized before this one asked them).
    pub(crate) fn clear_key_requests(&self) {
        for (_, table) in self.opened_tables() {
            let _ = table.take_wanted();
        }
    }

    /// Counts every key the last optimization asked for, when
    /// `krishiv.optimizer.discoverKeys` is on, within one budget per query
    /// (`krishiv.optimizer.discoverKeys.timeout`: each count gets what the
    /// ones before it left). Returns whether a count proved a key, so the
    /// plan may change when optimized again. A failed count never fails the
    /// query: its answer is unknown. The requests are taken either way.
    pub(crate) async fn discover_keys(&self) -> bool {
        use krishiv_common::config::{ConfigValue, DISCOVER_KEYS, DISCOVER_KEYS_TIMEOUT};
        let wanted: Vec<_> = self
            .opened_tables()
            .into_iter()
            .map(|(_, t)| {
                let w = t.take_wanted();
                (t, w)
            })
            .filter(|(_, w)| !w.is_empty())
            .collect();
        let config = &self.scans.config;
        if wanted.is_empty()
            || config.get(&DISCOVER_KEYS).and_then(ConfigValue::as_bool) != Some(true)
        {
            return false;
        }
        let budget = config
            .get(&DISCOVER_KEYS_TIMEOUT)
            .and_then(ConfigValue::as_duration)
            .unwrap_or_default();
        let deadline = std::time::Instant::now() + budget;
        // Under the session's pool, so the arbitrator and
        // `krishiv.session.memoryLimit` see the counts (no query pool exists
        // while planning).
        let pool = self.scans.memory.child("key counts", None);
        let mut proved = false;
        for (table, sets) in wanted {
            for columns in sets {
                let options = krishiv_exec::planner::KeyOptions {
                    timeout: deadline.saturating_duration_since(std::time::Instant::now()),
                    pool: Some(std::sync::Arc::clone(&pool)),
                };
                let table = std::sync::Arc::clone(&table);
                match run_on(&self.scans.runtime, async move {
                    table.discover_unique(&columns, options).await
                })
                .await
                {
                    Ok(outcome) => proved |= outcome.is_unique(),
                    Err(e) => tracing::debug!("key count skipped: {e}"),
                }
            }
        }
        proved
    }

    /// One line per key count of the query, in order:
    /// `<table>: key [<columns>]: <what it found>` (with the read's time when
    /// `timed`). A line repeated is kept once; empty when nothing was counted.
    pub(crate) fn key_notes(&self, timed: bool) -> Vec<String> {
        let mut notes = Vec::new();
        for (key, table) in self.opened_tables() {
            let name = files_label(&key);
            for outcome in table.key_outcomes() {
                let text = if timed {
                    outcome.with_time()
                } else {
                    outcome.to_string()
                };
                let note = format!("{name}: {text}");
                if !notes.contains(&note) {
                    notes.push(note);
                }
            }
        }
        notes
    }
}
