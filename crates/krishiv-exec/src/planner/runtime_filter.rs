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
//! (row group, batch) — correct, only less effective. The executor makes
//! such a scan start after the build when the filter is worth the wait
//! (`runner.rs`, `with_filter_order`; [`scan_waits`]), which takes away the
//! time the two pipelines would have overlapped. It is not worth it when the
//! build reads the scan's own table, as the two sides of an inlined CTE do:
//! the keys come from one domain, so the filter drops little (TPC-DS q30,
//! q81, q64 and q74 ran 1.12–1.30x slower waiting). Estimates cannot tell
//! that case apart alone: q95 and q30 have the same ratio of scan to build
//! rows, and only q95's filter pays. Even then a build far smaller than the
//! scan is selective enough (q58's seven dates of `date_dim`).
//!
//! # Cost model
//! min/max and the distinct keys (up to the larger of
//! `krishiv.runtimeFilter.inList.maxValues`, the statistics `IN` list, and
//! `krishiv.runtimeFilter.exactSet.maxValues`, the exact row form) are
//! always collected; a Bloom filter's hashes only when
//! `spark.sql.optimizer.runtime.bloomFilter.enabled`, the build is
//! estimated at most `krishiv.runtimeFilter.bloom.maxItems` rows, and the
//! scan at least four times the build and at least
//! `krishiv.runtimeFilter.bloom.minApplicationRows`. The build decides with
//! the actual key count at finalize. A build whose estimate is a selectivity
//! guess over an aggregate's output (a `HAVING`, or a filter on a grouped
//! subquery: `filtered_aggregate`) counts as at most `bloom.maxItems`
//! rows for both tests, so its Bloom form is planned and its actual keys
//! decide (TPC-H q18; DuckDB also picks the form from the built table's
//! count, `JoinFilterPushdownInfo::FinalizeFilters`). Spark's size guards are not used:
//! they are tuned for clusters.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use krishiv_common::config::{
    Config, ConfigValue, RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT,
    RUNTIME_FILTER_ADAPTIVE_MIN_ROWS, RUNTIME_FILTER_BLOOM_ENABLED, RUNTIME_FILTER_BLOOM_MAX_BYTES,
    RUNTIME_FILTER_BLOOM_MAX_ITEMS, RUNTIME_FILTER_BLOOM_MIN_APPLICATION_ROWS,
    RUNTIME_FILTER_ENABLED, RUNTIME_FILTER_EXACT_SET_MAX_VALUES, RUNTIME_FILTER_IN_LIST_MAX_VALUES,
};
use krishiv_io::runtime_filter::RuntimeBloom;
use krishiv_io::{ColumnPath, FilterId, KeyFilterOptions};
use krishiv_plan::{AttrId, Expr, ExprNode, ReadSource, ResolvedExpr};

use super::build::{Planned, Planner, column_of};
use crate::operators::RowStageSettings;
use crate::physical::{JoinFilterPlan, PhysNode, ScanId, ScanRuntimeFilter};

/// A scan column a runtime filter from above may target.
#[derive(Debug, Clone)]
pub(crate) struct Site {
    /// The column's attribute in the relation's output.
    pub(crate) attr: AttrId,
    /// The scan's table column.
    pub(crate) column: ColumnPath,
    /// The scan.
    pub(crate) scan: ScanId,
    /// The scan's estimated rows.
    pub(crate) scan_rows: f64,
    /// The scan's source, for `EXPLAIN`.
    pub(crate) scan_name: String,
    /// What the scan reads.
    pub(crate) source: Arc<ReadSource>,
}

/// The settings planning reads (see the module docs).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    enabled: bool,
    bloom: bool,
    max_values: usize,
    exact_max_values: usize,
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
            exact_max_values: usize::try_from(count(
                &RUNTIME_FILTER_EXACT_SET_MAX_VALUES,
                krishiv_io::runtime_filter::EXACT_SET_MAX as u64,
            ))
            .unwrap_or(krishiv_io::runtime_filter::EXACT_SET_MAX),
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

    /// The scans' row-stage settings.
    pub(crate) fn row_stage(&self) -> RowStageSettings {
        self.row
    }
}

/// Moves the runtime filters' row stage of a scan to the `Filter` directly above it (design P2-E3 U1): one
/// copy of the surviving rows instead of two. The planner already gave the scan no runtime filter when that
/// filter's predicate may raise (`no_runtime_filters_before`), so the predicate seeing the rows the row stage
/// used to drop first cannot raise on them (ADR 0008 section 6). Only filters on a top-level column the
/// scan projects move (the Filter's row stage finds its column by name in its input); any other leaves the
/// scan as it is.
fn fuse_row_stages(roots: Vec<&mut PhysNode>) {
    let mut stack = roots;
    while let Some(n) = stack.pop() {
        if let PhysNode::Filter {
            input,
            runtime,
            row_stage,
            ..
        } = n
            && runtime.is_empty()
            && let PhysNode::Scan(s) = input.as_mut()
            && !s.row_id
            && !s.row_stage_in_filter
            && !s.runtime_filters.is_empty()
            && s.runtime_filters
                .iter()
                .all(|f| f.column.0.len() == 1 && s.projection.contains(&f.column))
        {
            *runtime = s.runtime_filters.clone();
            *row_stage = s.row_stage;
            s.row_stage_in_filter = true;
        }
        stack.extend(n.children_mut());
    }
}

/// Hands each scan under `roots` the runtime filters joins publish to it, then moves the row stage of a scan
/// under a `Filter` to that filter.
pub(crate) fn attach_runtime_filters<'a>(
    filters: &mut HashMap<ScanId, Vec<ScanRuntimeFilter>>,
    roots: impl IntoIterator<Item = &'a mut PhysNode>,
) {
    let mut roots: Vec<&mut PhysNode> = roots.into_iter().collect();
    let mut stack: Vec<&mut PhysNode> = roots.iter_mut().map(|r| &mut **r).collect();
    while let Some(n) = stack.pop() {
        if let PhysNode::Scan(s) = n
            && let Some(f) = filters.remove(&s.id)
        {
            s.runtime_filters = f;
        }
        stack.extend(n.children_mut());
    }
    fuse_row_stages(roots);
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
    pub(crate) filters: Vec<JoinFilterPlan>,
    /// `EXPLAIN` text.
    pub(crate) explain: Vec<String>,
}

impl Planner<'_> {
    /// The runtime filters an inner or left semi hash join publishes for
    /// its probe keys (`probe_exprs`, of types `key_types`), given the
    /// build side's plan and estimated rows. A `partitioned` build (one
    /// partition per task) gets a Bloom size fixed from its estimate, so
    /// that its partitions' filters merge (design P2-T177 §2.4).
    pub(crate) fn runtime_filters(
        &mut self,
        probe: &Planned,
        probe_exprs: &[Arc<Expr>],
        key_types: &[DataType],
        build: &PhysNode,
        (build_rows, partitioned): (f64, bool),
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
                // A build whose estimate is a guess over an aggregate's
                // output (`HAVING`) plans its Bloom filter anyway: the build
                // decides from the keys it holds, as it does for any planned
                // Bloom (TPC-H q18: estimated 150M, holds 6,398 at SF100).
                // Past `bloom.maxItems` it publishes none, so the guess is
                // capped there for the scan-size test and the planned size.
                #[expect(clippy::cast_precision_loss, reason = "estimates")]
                let rows = if s.bloom_max_items > 0 && filtered_aggregate(build) {
                    build_rows.min(s.bloom_max_items as f64)
                } else {
                    build_rows
                };
                #[expect(clippy::cast_precision_loss, reason = "estimates")]
                let bloom = s.bloom
                    && rows <= s.bloom_max_items as f64
                    && site.scan_rows >= (4.0 * rows).max(s.min_application_rows as f64);
                let reads_scan_table = reads_source(build, &site.source);
                let id = FilterId(self.next_filter);
                self.next_filter += 1;
                self.scan_filters
                    .entry(site.scan)
                    .or_default()
                    .push(ScanRuntimeFilter {
                        id,
                        column: site.column.clone(),
                        key_type: key_type.clone(),
                    });
                out.explain.push(format!(
                    "RuntimeFilter {id}: {e} -> {}.{}{}",
                    site.scan_name,
                    site.column.0.join("."),
                    if bloom { " (bloom)" } else { "" }
                ));
                let mut options = KeyFilterOptions {
                    max_values: s.max_values,
                    exact_max_values: s.exact_max_values,
                    bloom,
                    scan_waits: scan_waits(site.scan_rows, build_rows, reads_scan_table),
                    bloom_max_items: s.bloom_max_items,
                    bloom_max_bytes: s.bloom_max_bytes,
                    ..KeyFilterOptions::default()
                };
                // Sized at the false-positive rate the producers build with.
                options.bloom_bytes = (bloom && partitioned)
                    .then(|| planned_bloom_bytes(rows, options.bloom_fpp, options.bloom_max_bytes));
                out.filters.push(JoinFilterPlan { id, key, options });
            }
        }
        out
    }
}

/// The Bloom size in bytes for a build of `build_rows` estimated keys at
/// the false-positive rate `fpp`, at most `max_bytes`.
fn planned_bloom_bytes(build_rows: f64, fpp: f64, max_bytes: usize) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a non-negative estimate, capped below by the cost model's bloom_max_items"
    )]
    let keys = build_rows.max(1.0).ceil() as u64;
    let bytes = RuntimeBloom::bytes_for(keys, fpp).min(max_bytes);
    u32::try_from(bytes).unwrap_or(u32::MAX)
}

/// Whether `node`'s rows are an aggregate's output cut by a filter (a
/// `HAVING`, or a filter on a grouped subquery), looking down through
/// projections and exchanges: the estimate of such a build is a selectivity
/// guess, so it must not keep a runtime filter's Bloom form out of the plan.
/// The walk is shallow on purpose: any other node in between (a join, a
/// sort, a limit) answers `false`, and the estimate is used as it is.
fn filtered_aggregate(node: &PhysNode) -> bool {
    let mut filtered = false;
    let mut n = node;
    loop {
        match n {
            PhysNode::Filter { input, .. } => {
                filtered = true;
                n = input;
            }
            PhysNode::Project { input, .. } => n = input,
            PhysNode::Exchange(e) => n = &e.input,
            PhysNode::Aggregate { .. } => return filtered,
            _ => return false,
        }
    }
}

/// How many times the build must be smaller than a scan of the build's own
/// table for its filter to be worth the scan's wait.
const SELECTIVE_BUILD: f64 = 1_000.0;

/// Whether a scan of `scan_rows` estimated rows starts after a build of
/// `build_rows`, given whether the build reads the scan's table (see the
/// module docs).
fn scan_waits(scan_rows: f64, build_rows: f64, build_reads_scan_table: bool) -> bool {
    !build_reads_scan_table || scan_rows >= SELECTIVE_BUILD * build_rows.max(1.0)
}

/// Whether `node`'s plan scans `source`.
fn reads_source(node: &PhysNode, source: &ReadSource) -> bool {
    matches!(node, PhysNode::Scan(s) if s.scan.source == *source)
        || node.children().into_iter().any(|c| reads_source(c, source))
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use krishiv_common::secret::OptionValue;
    use krishiv_plan::{ReadSource, Scan};

    use super::{PhysNode, reads_source, scan_waits};
    use crate::physical::{ScanId, ScanNode};

    fn files(path: &str) -> ReadSource {
        ReadSource::Files {
            format: "parquet".to_owned(),
            paths: vec![OptionValue::from_user("path", path)],
            options: BTreeMap::new(),
        }
    }

    /// Two paths of one format are two sources.
    #[test]
    fn a_source_is_its_paths_not_its_format() {
        let scan = PhysNode::Scan(ScanNode {
            id: ScanId(0),
            scan: Scan {
                source: files("a.parquet"),
                output: Vec::new(),
                version: None,
            },
            projection: Vec::new(),
            filter: None,
            runtime_filters: Vec::new(),
            row_stage: crate::operators::RowStageSettings::default(),
            row_id: false,
            row_stage_in_filter: false,
        });
        assert!(reads_source(&scan, &files("a.parquet")));
        assert!(!reads_source(&scan, &files("b.parquet")));
    }

    #[test]
    fn a_scan_waits_unless_the_build_reads_its_table() {
        // Another table: whatever the estimates say (TPC-H q18's are wrong).
        assert!(scan_waits(15e6, 20e6, false));
        // The build's own table: a filter across one key domain drops little.
        assert!(!scan_waits(250_000.0, 120_000.0, true));
        assert!(!scan_waits(500_000.0, 4_750.0, true));
        // Unless the build is far smaller.
        assert!(scan_waits(73_049.0, 7.0, true));
        assert!(scan_waits(1_000.0, 1.0, true));
        assert!(!scan_waits(999.0, 1.0, true));
        // An estimate of less than one row counts as one.
        assert!(!scan_waits(500.0, 0.0, true));
    }
}
