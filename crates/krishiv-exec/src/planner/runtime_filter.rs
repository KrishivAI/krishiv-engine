// SPDX-License-Identifier: Apache-2.0
//! Runtime filter planning (design P2-T163): which scan columns a join's
//! build may filter, and what it collects for them.
//!
//! # Legal targets
//! Every planned relation carries its **sites**: the scan columns among its
//! output that a filter from above may target, by attribute id. A filter
//! drops only rows its producer would drop, so a site survives a node only
//! when the predicate `key ∈ build keys` could be pushed through it:
//! - filters, projections of the column (renamed, or cast by a widening
//!   that keeps values distinct and equal — `int` to `bigint`, decimal
//!   widening), sorts, the grouping keys of an aggregate, the branches of
//!   a `UNION ALL` and `WITH` references (each reference is planned anew);
//! - both inputs of an inner join; only the preserved (probe) side of a
//!   left outer, semi or anti join (never the null-supplying side);
//! - never through a `LIMIT` or Top-N, `INTERSECT ALL`/`EXCEPT ALL`
//!   replication, or into subqueries.
//!
//! Only inner and left semi hash joins produce filters (they drop unmatched
//! probe rows), one per key whose probe expression is a site's column.
//!
//! # Scans below pipeline breakers
//! A site below an aggregate, a sort or another join's build side is read
//! by a pipeline that may run while the producer's build does: the scan
//! starts with the filter pending and adopts it at its next check point
//! (row group, batch) — correct, only less effective. A scheduling-only
//! dependency that makes such scans wait for the build was measured and
//! dropped: the executor runs pipelines in dependency waves, so the extra
//! dependency adds a wave and serializes unrelated pipelines (TPC-DS q30,
//! q81 and q74 ran 1.6–1.9x slower with it).
//!
//! # Cost model
//! min/max and the value set (up to `krishiv.runtimeFilter.inList.maxValues`
//! keys) are always collected; a Bloom filter's hashes only when
//! `spark.sql.optimizer.runtime.bloomFilter.enabled`, the build is
//! estimated at most `krishiv.runtimeFilter.bloom.maxItems` rows, and the
//! scan at least four times the build and at least
//! `krishiv.runtimeFilter.bloom.minApplicationRows`. The build decides with
//! the actual key count at finalize. Spark's size guards are not used:
//! they are tuned for clusters.

use std::sync::Arc;

use arrow_schema::DataType;
use krishiv_common::config::{
    Config, ConfigValue, RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT,
    RUNTIME_FILTER_ADAPTIVE_MIN_ROWS, RUNTIME_FILTER_BLOOM_ENABLED, RUNTIME_FILTER_BLOOM_MAX_BYTES,
    RUNTIME_FILTER_BLOOM_MAX_ITEMS, RUNTIME_FILTER_BLOOM_MIN_APPLICATION_ROWS,
    RUNTIME_FILTER_ENABLED, RUNTIME_FILTER_IN_LIST_MAX_VALUES,
};
use krishiv_io::{ColumnPath, FilterId, KeyFilterOptions, RuntimeFilter};
use krishiv_plan::{AttrId, Expr, ExprNode, ResolvedExpr};

use super::build::{Planned, Planner, column_of};
use crate::join::JoinRuntimeFilter;
use crate::operators::{RowStageSettings, RuntimeFilters};

/// A scan column a runtime filter from above may target.
#[derive(Debug, Clone)]
pub(crate) struct Site {
    /// The column's attribute in the relation's output.
    pub(crate) attr: AttrId,
    /// The scan's table column and its type.
    pub(crate) column: ColumnPath,
    /// The scan's filter list.
    pub(crate) slot: RuntimeFilters,
    /// The scan's estimated rows.
    pub(crate) scan_rows: f64,
    /// The scan, for `EXPLAIN`.
    pub(crate) scan: String,
}

/// The settings planning reads (see the module docs).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    enabled: bool,
    bloom: bool,
    max_values: usize,
    bloom_max_items: u64,
    bloom_max_bytes: usize,
    min_application_rows: u64,
    row: RowStageSettings,
}

fn int(config: &Config, e: &krishiv_common::config::ConfigEntry) -> Option<i64> {
    config.get(e).and_then(ConfigValue::as_int)
}

impl Settings {
    pub(crate) fn new(config: &Config) -> Self {
        let flag = |e| config.get(e).and_then(ConfigValue::as_bool).unwrap_or(true);
        let count = |e, d: u64| {
            int(config, e)
                .and_then(|v| u64::try_from(v).ok())
                .unwrap_or(d)
        };
        let percent = count(&RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT, 90).min(100);
        #[expect(clippy::cast_precision_loss, reason = "a percentage")]
        let max_pass_rate = percent as f64 / 100.0;
        Self {
            enabled: flag(&RUNTIME_FILTER_ENABLED),
            bloom: flag(&RUNTIME_FILTER_BLOOM_ENABLED),
            max_values: usize::try_from(count(&RUNTIME_FILTER_IN_LIST_MAX_VALUES, 50))
                .unwrap_or(50),
            bloom_max_items: count(&RUNTIME_FILTER_BLOOM_MAX_ITEMS, 4_000_000),
            bloom_max_bytes: config
                .get(&RUNTIME_FILTER_BLOOM_MAX_BYTES)
                .and_then(ConfigValue::as_size)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(8 << 20),
            min_application_rows: count(&RUNTIME_FILTER_BLOOM_MIN_APPLICATION_ROWS, 1_000_000),
            row: RowStageSettings {
                min_rows: count(&RUNTIME_FILTER_ADAPTIVE_MIN_ROWS, 65_536).max(1),
                max_pass_rate,
            },
        }
    }

    /// Whether runtime filters are planned at all.
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }
}

/// The attribute `e` reads when it is a column or a widening cast of one
/// (`input_type` gives a column's type in the relation).
pub(crate) fn key_source(
    e: &Expr,
    input_type: &impl Fn(AttrId) -> Option<DataType>,
) -> Option<AttrId> {
    if let Some(id) = column_of(e) {
        return Some(id);
    }
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => key_source(expr, input_type),
        ExprNode::Cast { expr, to, .. } => {
            let id = column_of(expr)?;
            krishiv_io::runtime_filter::widens_to(&input_type(id)?, &to.to_arrow()).then_some(id)
        }
        _ => None,
    }
}

/// The type of `id`'s column in `p`.
pub(crate) fn column_type(p: &Planned, id: AttrId) -> Option<DataType> {
    let i = p.layout.iter().position(|a| *a == id)?;
    Some(p.schema.field(i).data_type().clone())
}

/// What one join publishes.
pub(crate) struct JoinFilters {
    pub(crate) filters: Vec<JoinRuntimeFilter>,
    /// `EXPLAIN` text.
    pub(crate) explain: Vec<String>,
}

impl Planner<'_> {
    /// The runtime filters an inner or left semi hash join publishes for
    /// its probe keys (`probe_exprs`, of types `key_types`), given the
    /// estimated build rows.
    pub(crate) fn runtime_filters(
        &mut self,
        probe: &Planned,
        probe_exprs: &[Arc<Expr>],
        key_types: &[DataType],
        build_rows: f64,
    ) -> JoinFilters {
        let mut out = JoinFilters {
            filters: Vec::new(),
            explain: Vec::new(),
        };
        let s = self.rf;
        if !s.enabled {
            return out;
        }
        let input_type = |id: AttrId| column_type(probe, id);
        for (key, (e, key_type)) in probe_exprs.iter().zip(key_types).enumerate() {
            if !krishiv_io::runtime_filter::supported_key_type(key_type) {
                continue;
            }
            let Some(source) = key_source(e, &input_type) else {
                continue;
            };
            // Every scan the key reaches gets its own filter (the branches
            // of a union, both inputs of an inner join below).
            for site in probe.sites.iter().filter(|s| s.attr == source) {
                #[expect(clippy::cast_precision_loss, reason = "estimates")]
                let bloom = s.bloom
                    && build_rows <= s.bloom_max_items as f64
                    && site.scan_rows >= (4.0 * build_rows).max(s.min_application_rows as f64);
                let id = FilterId(self.next_filter);
                self.next_filter += 1;
                let filter = Arc::new(RuntimeFilter::new(
                    id,
                    site.column.clone(),
                    key_type.clone(),
                ));
                site.slot.push(Arc::clone(&filter));
                site.slot.set_settings(s.row);
                out.explain.push(format!(
                    "RuntimeFilter {id}: {e} -> {}.{}{}",
                    site.scan,
                    site.column.0.join("."),
                    if bloom { " (bloom)" } else { "" }
                ));
                out.filters.push(JoinRuntimeFilter {
                    key,
                    filter,
                    options: KeyFilterOptions {
                        max_values: s.max_values,
                        bloom,
                        bloom_max_items: s.bloom_max_items,
                        bloom_max_bytes: s.bloom_max_bytes,
                        ..KeyFilterOptions::default()
                    },
                });
            }
        }
        out
    }
}

/// `sites` that are still columns of `layout`.
pub(crate) fn kept(sites: &[Site], layout: &[AttrId]) -> Vec<Site> {
    sites
        .iter()
        .filter(|s| layout.contains(&s.attr))
        .cloned()
        .collect()
}

/// `sites` renamed from the attributes `from` to `to` (by position);
/// sites of other attributes are dropped.
pub(crate) fn renamed(sites: &[Site], from: &[AttrId], to: &[AttrId]) -> Vec<Site> {
    sites
        .iter()
        .filter_map(|s| {
            let i = from.iter().position(|a| *a == s.attr)?;
            Some(Site {
                attr: *to.get(i)?,
                ..s.clone()
            })
        })
        .collect()
}
