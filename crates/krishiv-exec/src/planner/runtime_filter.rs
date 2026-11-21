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
//! scan is selective enough (q58's seven dates of `date_dim`), and a build
//! that is a bare scan of a small table costs too little to overlap (q59's
//! 365 dates of `date_dim`, [`waits_for_build`]).
//!
//! # Cost model
//! min/max and the distinct keys (up to the larger of
//! `krishiv.runtimeFilter.inList.maxValues`, the statistics `IN` list, and
//! `krishiv.runtimeFilter.exactSet.maxValues`, the exact row form) are
//! always collected; a Bloom filter's hashes only when
//! `spark.sql.optimizer.runtime.bloomFilter.enabled`, the build is
//! estimated at most `krishiv.runtimeFilter.bloom.maxItems` rows, and the
//! scan at least four times the build and at least
//! `krishiv.runtimeFilter.bloom.minApplicationRows`. A scan at least 20
//! times its build raises both caps (`bloom.atScale`: 20 M keys, 32 MiB).
//! A build estimated past those 20 M keys, under a scan at least 8 times it
//! (Polars' admission ratio), gets `bloom.memoryTier`'s caps: 1/64 of the
//! query memory limit and the keys that holds (DuckDB's share; study 36
//! §4.4, the L3 tier; measured at SF1000 by perf-1 2026-11-17, while the
//! same 8x ratio for smaller builds was killed at SF100). The build decides with
//! the actual key count at finalize. A build whose estimate is a selectivity
//! guess over an aggregate's output (a `HAVING`, or a filter on a grouped
//! subquery: `filtered_aggregate`) counts as at most `bloom.maxItems`
//! rows for both tests, so its Bloom form is planned and its actual keys
//! decide (TPC-H q18; DuckDB also picks the form from the built table's
//! count, `JoinFilterPushdownInfo::FinalizeFilters`). The same cap applies
//! to a build whose input is cut anywhere below by a predicate the
//! estimator cannot model (`LIKE`, `RLIKE`, a string function, a UDF:
//! [`krishiv_optimizer::selectivity_is_guessed`], in a `Filter`, a filter
//! pushed into a scan or one fused into a hash probe: `guessed_filter`):
//! its estimate rests on the default selectivity (TPC-H q20's `p_name LIKE
//! 'forest%'`: 0.2 guessed, about 0.01 actual, a build estimated at 16M
//! keys that holds 869K at SF100). A build whose key rows exceed
//! `bloom.maxItems` publishes no Bloom form (`KeyFilterBuilder::add` and
//! `merge` in `krishiv_io::runtime_filter` drop the hashes past it), so a
//! wrong guess costs only the hashing until then. Except for such a guessed
//! build while `bloom.memoryTier` is on: it gets a fixed
//! Bloom (`KeyFilterOptions::bloom_spill_bytes`) sized for a quarter of the
//! scan's rows within the tier. Its memory is reserved once the build's
//! keys reach a sixteenth of it as hashes, spill or not (early, while the
//! pool has room; freed at publish if unused), and the partials make the
//! filter on it once one passes the cap or is asked to spill, then drain
//! their hashes into it. Once the keys drained pass
//! what it holds within `bloom_max_fpp` it is freed and hashing stops, so
//! a wrong guess costs at most the filter's keys; else it is published
//! (TPC-H q09 at SF1000: `p_name LIKE '%green%'`, a lineitem-side build of
//! ~300M keys that always spills, had only a range on `orders`;
//! `join/build/spilled.rs`). A Bloom that drops little is switched off by
//! the scan's adaptive row stage. The join may be broadcast or shuffled
//! (q09's is shuffled on the key) when the plan runs on one worker: a cut
//! plan keeps none (`krishiv-scheduler` `fragment.rs` clears it on every
//! join of a fragment and on an exchange's collected filter, as a fragment
//! runs as tasks).
//! Whether the scan
//! waits for the filter ([`scan_waits`]) still reads the uncapped
//! estimate. Spark's size guards are not used: they are tuned for clusters.

use std::sync::Arc;

use super::explain::{FilterGap, FilterGapReason};

use arrow_schema::DataType;
use krishiv_common::config::{
    Config, ConfigValue, MEMORY_LIMIT, RUNTIME_FILTER_ADAPTIVE_MAX_PASS_PERCENT,
    RUNTIME_FILTER_ADAPTIVE_MIN_ROWS, RUNTIME_FILTER_BLOOM_AT_SCALE, RUNTIME_FILTER_BLOOM_ENABLED,
    RUNTIME_FILTER_BLOOM_MAX_BYTES, RUNTIME_FILTER_BLOOM_MAX_ITEMS,
    RUNTIME_FILTER_BLOOM_MEMORY_TIER, RUNTIME_FILTER_BLOOM_MIN_APPLICATION_ROWS,
    RUNTIME_FILTER_ENABLED, RUNTIME_FILTER_EQUATED_SITES, RUNTIME_FILTER_EXACT_SET_MAX_VALUES,
    RUNTIME_FILTER_IN_LIST_MAX_VALUES, SESSION_MEMORY_LIMIT,
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
    /// The estimated rows of the scan's table before the scan's pushed
    /// filter: what a bare scan of the same table reads.
    pub(crate) table_rows: f64,
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
    /// `krishiv.runtimeFilter.bloom.atScale` raises the key cap, the
    /// size cap: on, unless the key is on only by default and an admin
    /// set that cap (an explicit cap is a limit the default does not lift).
    at_scale_items: bool,
    at_scale_bytes: bool,
    /// `krishiv.runtimeFilter.bloom.memoryTier`: the at-scale size cap
    /// taken from the query memory limit, when larger.
    memory_tier: Option<usize>,
    /// `krishiv.runtimeFilter.equatedSites`: an inner join's key
    /// equalities add sites ([`equated_sites`]).
    pub(crate) equated_sites: bool,
    row: RowStageSettings,
}

/// The share of the query memory limit one Bloom filter may take under
/// `krishiv.runtimeFilter.bloom.memoryTier` (DuckDB's limit / 64).
const BLOOM_MEMORY_TIER_SHARE: usize = 64;
/// A scan at least this many times its build gets the at-scale caps.
const BLOOM_AT_SCALE_RATIO: f64 = 20.0;
/// A build past the at-scale key cap gets the memory tier's caps under a
/// scan at least this many times it (Polars' admission ratio).
const BLOOM_MEMORY_TIER_RATIO: f64 = 8.0;
/// The Bloom key cap `krishiv.runtimeFilter.bloom.atScale` raises to.
const BLOOM_AT_SCALE_MAX_ITEMS: u64 = 20_000_000;
/// The Bloom size cap `krishiv.runtimeFilter.bloom.atScale` raises to.
const BLOOM_AT_SCALE_MAX_BYTES: usize = 32 << 20;

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
        let at_scale = config
            .get(&RUNTIME_FILTER_BLOOM_AT_SCALE)
            .and_then(ConfigValue::as_bool)
            .unwrap_or(true);
        let at_scale_explicit = config.is_set(&RUNTIME_FILTER_BLOOM_AT_SCALE);
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
            at_scale_items: at_scale
                && (at_scale_explicit || !config.is_set(&RUNTIME_FILTER_BLOOM_MAX_ITEMS)),
            at_scale_bytes: at_scale
                && (at_scale_explicit || !config.is_set(&RUNTIME_FILTER_BLOOM_MAX_BYTES)),
            memory_tier: flag(&RUNTIME_FILTER_BLOOM_MEMORY_TIER)
                .then(|| memory_tier(query_memory_limit(config)))
                .flatten(),
            equated_sites: flag(&RUNTIME_FILTER_EQUATED_SITES),
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

    /// These settings for one scan of `scan_rows` estimated rows filtered by
    /// a build of `build_rows`: a scan at least 20 times the build gets the
    /// at-scale caps (`bloom.atScale`: 20 M keys, 32 MiB; with R4's
    /// prefetching probe a Bloom past the caches still pays), and a build
    /// past 20 M keys under a scan at least 8 times it the memory tier's too
    /// (study 36 §4.4). The tier's key cap is what the size cap in force
    /// holds, so an explicit `bloom.maxBytes` also bounds the keys a build
    /// collects hashes for (review M1).
    fn for_scan(mut self, scan_rows: f64, build_rows: f64) -> Self {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a cap compared with an estimate"
        )]
        let past_at_scale = build_rows > BLOOM_AT_SCALE_MAX_ITEMS as f64;
        let tier = self
            .memory_tier
            .filter(|_| past_at_scale && scan_rows >= BLOOM_MEMORY_TIER_RATIO * build_rows);
        if tier.is_none() && scan_rows < BLOOM_AT_SCALE_RATIO * build_rows {
            return self;
        }
        if self.at_scale_bytes {
            self.bloom_max_bytes = self
                .bloom_max_bytes
                .max(BLOOM_AT_SCALE_MAX_BYTES)
                .max(tier.unwrap_or(0));
        }
        if self.at_scale_items {
            let tier = tier.map_or(0, |t| keys_for(t.min(self.bloom_max_bytes)));
            self.bloom_max_items = self.bloom_max_items.max(BLOOM_AT_SCALE_MAX_ITEMS).max(tier);
        }
        self
    }

    /// The size of the fixed Bloom a guessed build keeps past its cap
    /// (`KeyFilterOptions::bloom_spill_bytes`) under a scan of `scan_rows`:
    /// sized for a quarter of the scan's rows (the planner's own 4x test: a
    /// build of more keys filters too little to pay), within the memory
    /// tier and an explicit `bloom.maxBytes`; `None` with the tier off.
    /// TPC-H q09 at SF1000 keeps the whole tier (375M keys).
    fn spill_bloom_bytes(&self, scan_rows: f64, fpp: f64) -> Option<u32> {
        let tier = self.memory_tier?;
        let cap = if self.at_scale_bytes {
            tier
        } else {
            tier.min(self.bloom_max_bytes)
        };
        Some(planned_bloom_bytes(scan_rows / 4.0, fpp, cap))
    }
}

/// The memory a query may use: `krishiv.memory.limit` (else the root pool's
/// default, `krishiv_memory::default_limit`), or `krishiv.session.memoryLimit`
/// when lower. Read where the plan is made; a worker with a smaller pool
/// still charges the Bloom to the build's reservation.
fn query_memory_limit(config: &Config) -> Option<usize> {
    let size = |e| {
        config
            .get(e)
            .and_then(ConfigValue::as_size)
            .map(|v| usize::try_from(v).unwrap_or(usize::MAX))
    };
    // The default reads `/proc` and `/sys`: once per process.
    static DEFAULT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let root =
        size(&MEMORY_LIMIT).or_else(|| *DEFAULT.get_or_init(krishiv_memory::default_limit))?;
    Some(size(&SESSION_MEMORY_LIMIT).map_or(root, |s| s.min(root)))
}

/// The memory tier's size cap for a query memory limit: 1/64 of it.
fn memory_tier(limit: Option<usize>) -> Option<usize> {
    let bytes = limit? / BLOOM_MEMORY_TIER_SHARE;
    (bytes > 0).then_some(bytes)
}

/// The keys a Bloom of `bytes` holds at the producers' false-positive rate.
fn keys_for(bytes: usize) -> u64 {
    let fpp = KeyFilterOptions::default().bloom_fpp;
    let per_mi_keys = RuntimeBloom::bytes_for(1 << 20, fpp) as u64;
    (bytes as u64).saturating_mul(1 << 20) / per_mi_keys.max(1)
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

/// The sites an inner equi-join adds through its key equalities: each key
/// pair of plain columns of one type holds equal values in every row the
/// join returns, so a filter from above on either column may drop the
/// other column's scan rows too. A side's site of its key column is
/// returned again under the other side's attribute (TPC-H q18: the
/// `HAVING` keys' filter on `o_orderkey` reaches `lineitem.l_orderkey`
/// through `l_orderkey = o_orderkey`). A dropped scan row either matched
/// nothing (the join drops it) or matched a row of the other column's
/// value, which the filter above drops; a residual only drops more. As
/// DuckDB's `ExpandEquivalentFilterColumns` (research E11), which also
/// takes column-to-column equalities of equal type only.
pub(crate) fn equated_sites(
    probe: &Planned,
    probe_exprs: &[Arc<Expr>],
    build: &Planned,
    build_exprs: &[Arc<Expr>],
) -> Vec<Site> {
    let mut out = Vec::new();
    for (p, b) in probe_exprs.iter().zip(build_exprs) {
        let (Some(pa), Some(ba)) = (column_of(p), column_of(b)) else {
            continue;
        };
        let ty = column_type(probe, pa);
        if pa == ba || ty.is_none() || ty != column_type(build, ba) {
            continue;
        }
        for (from, to, sites) in [(pa, ba, &probe.sites), (ba, pa, &build.sites)] {
            out.extend(sites.iter().filter(|s| s.attr == from).map(|s| Site {
                attr: to,
                ..s.clone()
            }));
        }
    }
    out
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
    /// Keys that got no filter, for the advisor (P3-T073).
    pub(crate) gaps: Vec<FilterGap>,
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
            gaps: Vec::new(),
        };
        let s = self.rf;
        let gap = |key: usize, e: &Expr, reason| FilterGap {
            key,
            probe_key: e.to_string(),
            reason,
        };
        let input_type = |id: AttrId| column_type(probe, id);
        for (key, (e, key_type)) in probe_exprs.iter().zip(key_types).enumerate() {
            if !krishiv_io::runtime_filter::supported_key_type(key_type) {
                continue;
            }
            if !s.enabled {
                out.gaps.push(gap(key, e, FilterGapReason::Disabled));
                continue;
            }
            let Some(source) = key_source(e, &input_type) else {
                out.gaps.push(gap(key, e, FilterGapReason::NoScanColumn));
                continue;
            };
            if !probe.sites.iter().any(|s| s.attr == source) {
                out.gaps.push(gap(key, e, FilterGapReason::Blocked));
            }
            // Every scan the key reaches gets its own filter (the branches
            // of a union, both inputs of an inner join below).
            for site in probe.sites.iter().filter(|s| s.attr == source) {
                // The at-scale caps (`bloom.atScale`, ledger §4 item 13;
                // `bloom.memoryTier`, the L3 tier).
                let s = s.for_scan(site.scan_rows, build_rows);
                // A build whose estimate is a guess over an aggregate's
                // output (`HAVING`) plans its Bloom filter anyway: the build
                // decides from the keys it holds, as it does for any planned
                // Bloom (TPC-H q18: estimated 150M, holds 6,398 at SF100).
                // Past `bloom.maxItems` it publishes none, so the guess is
                // capped there for the scan-size test and the planned size.
                // So is a build cut by a predicate the estimator cannot
                // model (TPC-H q20's `LIKE`: estimated 16M, holds 869K).
                let guessed = filtered_aggregate(build) || guessed_filter(build);
                #[expect(clippy::cast_precision_loss, reason = "estimates")]
                let rows = if s.bloom_max_items > 0 && guessed {
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
                    scan_waits: waits_for_build(
                        site.scan_rows,
                        site.table_rows,
                        build_rows,
                        reads_scan_table,
                        build,
                    ),
                    bloom_max_items: s.bloom_max_items,
                    bloom_max_bytes: s.bloom_max_bytes,
                    ..KeyFilterOptions::default()
                };
                // Sized at the false-positive rate the producers build with.
                options.bloom_bytes = (bloom && partitioned)
                    .then(|| planned_bloom_bytes(rows, options.bloom_fpp, options.bloom_max_bytes));
                // A guessed build may hold far more keys than its capped
                // estimate, and spill: past the cap, or when it sheds its
                // hashes, it keeps a Bloom of the memory tier. Broadcast or
                // shuffled alike (q09's join is shuffled) on one worker,
                // where the join is one build; cutting the plan clears it
                // (`krishiv-scheduler` `fragment.rs` `local_filters`), and
                // a build keeps it instead of the planned size
                // (`instantiate/join.rs` `build_options`).
                options.bloom_spill_bytes = (bloom && guessed)
                    .then(|| s.spill_bloom_bytes(site.scan_rows, options.bloom_fpp))
                    .flatten();
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

/// Whether a predicate anywhere in `node`'s plan (a `Filter`, a filter
/// pushed into a scan, a filter fused into a hash probe), through joins,
/// projections, exchanges and aggregates, is one whose selectivity the
/// estimator guesses ([`krishiv_optimizer::selectivity_is_guessed`]): the
/// estimate of `node`'s rows then rests on the default selectivity. A
/// shared plan read through `SharedRead` is not looked into.
fn guessed_filter(node: &PhysNode) -> bool {
    let own = match node {
        PhysNode::Filter { predicate, .. } => Some(predicate),
        PhysNode::Scan(s) => s.filter.as_ref().map(|f| &f.predicate),
        PhysNode::HashJoin(j) => j
            .select
            .as_ref()
            .and_then(|s| s.filter.as_ref())
            .map(|f| &f.predicate),
        _ => None,
    };
    own.is_some_and(|p| krishiv_optimizer::selectivity_is_guessed(p))
        || node.children().into_iter().any(guessed_filter)
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

/// The most estimated rows a scan may have for a bare scan of its own table
/// to be worth its wait whatever the ratio: reading such a table first costs
/// a few milliseconds.
const SMALL_TABLE_ROWS: f64 = 1_000_000.0;

/// Whether a scan of `scan_rows` estimated rows starts after the pipeline
/// of `build`, a build of `build_rows` estimated rows ([`scan_waits`]). A
/// build that reads the scan's own table also waits when it is a bare scan
/// ([`bare_scan`]) and the table is small (TPC-DS q59: 365 dates of
/// `date_dim` filter another `date_dim` scan, which otherwise starts with
/// the filter pending and, under load, passes 9 or more batches of dates
/// before it, so the `store_sales` join takes 20-27M rows instead of 5.5M;
/// study 43 §1). The table's size is `table_rows`, the scan's estimate
/// before its pushed filter: a bare scan of that table reads all of it
/// whatever its own filter (a filtered self-join of a fact table must not
/// wait for a full scan of the fact table).
fn waits_for_build(
    scan_rows: f64,
    table_rows: f64,
    build_rows: f64,
    build_reads_scan_table: bool,
    build: &PhysNode,
) -> bool {
    scan_waits(scan_rows, build_rows, build_reads_scan_table)
        || (table_rows <= SMALL_TABLE_ROWS && bare_scan(build))
}

/// Whether `node` is a scan under filters, projections and exchanges only
/// (q59's build is a broadcast of a filtered `date_dim` scan): a pipeline
/// waiting for it waits for one scan, not for a breaker's input. An
/// exchange adds no work of its own to wait for: on one worker it is
/// lowered into the operators around it, and in a cluster its map side is
/// that scan.
fn bare_scan(node: &PhysNode) -> bool {
    match node {
        PhysNode::Scan(_) => true,
        PhysNode::Filter { input, .. } | PhysNode::Project { input, .. } => bare_scan(input),
        PhysNode::Exchange(e) => bare_scan(&e.input),
        _ => false,
    }
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

    use krishiv_common::config::Config;

    use super::{
        BLOOM_AT_SCALE_MAX_BYTES, BLOOM_AT_SCALE_MAX_ITEMS, PhysNode, Settings, bare_scan,
        keys_for, memory_tier, query_memory_limit, reads_source, scan_waits, waits_for_build,
    };
    use crate::physical::{ScanId, ScanNode};

    fn files(path: &str) -> ReadSource {
        ReadSource::files(
            "parquet".to_owned(),
            vec![OptionValue::from_user("path", path)],
            BTreeMap::new(),
        )
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

    /// TPC-DS q59 (study 43 §1): a 365-row build from a filtered scan of
    /// `date_dim` filters another `date_dim` scan. The ratio (200) is under
    /// `SELECTIVE_BUILD`, but the build is a bare scan of a small table, so
    /// the wait costs one small scan, never a breaker's pipeline.
    #[test]
    fn a_scan_waits_for_a_small_bare_scan_of_its_own_table() {
        let dates = PhysNode::Scan(ScanNode {
            id: ScanId(1),
            scan: Scan {
                source: files("date_dim.parquet"),
                output: Vec::new(),
                version: None,
            },
            projection: Vec::new(),
            filter: None,
            runtime_filters: Vec::new(),
            row_stage: crate::operators::RowStageSettings::default(),
            row_id: false,
        });
        assert!(bare_scan(&dates));
        assert!(waits_for_build(73_049.0, 73_049.0, 365.0, true, &dates));
        // Another table waits as before.
        assert!(waits_for_build(15e6, 15e6, 20e6, false, &dates));
        // A large table read twice: the wait would lose the overlap.
        assert!(!waits_for_build(2e6, 2e6, 20_000.0, true, &dates));
        // A filtered scan of a large table: its own estimate is small, but
        // the build reads the whole table (a filtered fact self-join).
        assert!(!waits_for_build(5_000.0, 288e6, 1_000.0, true, &dates));
        // A breaker below the build (an inlined CTE's aggregate, q30, q81,
        // q64, q74): the wait would queue behind the breaker's pipeline.
        let sorted = PhysNode::Sort {
            input: Box::new(dates),
            keys: Vec::new(),
        };
        assert!(!bare_scan(&sorted));
        assert!(!waits_for_build(73_049.0, 73_049.0, 365.0, true, &sorted));
    }

    /// The L3 tier (study 36 §4.4): 1/64 of the limit, and the keys a
    /// Bloom of that size holds at 1% (about 9.7 bits a key).
    #[test]
    fn the_memory_tier_is_a_64th_of_the_limit() {
        let bytes = memory_tier(Some(20 << 30)).unwrap();
        assert_eq!(bytes, 320 << 20);
        let items = keys_for(bytes);
        assert!((250_000_000..=290_000_000).contains(&items), "{items}");
        // What it holds fits it.
        assert!(krishiv_io::runtime_filter::RuntimeBloom::bytes_for(items, 0.01) <= bytes + 4096);
        assert_eq!(memory_tier(None), None);
        assert_eq!(memory_tier(Some(10)), None);
    }

    /// The session limit binds when lower; the key turns the tier off.
    #[test]
    fn the_memory_tier_follows_the_session_limit_and_its_key() {
        let mut c = Config::new();
        c.set("krishiv.memory.limit", "20g").unwrap();
        assert_eq!(query_memory_limit(&c), Some(20 << 30));
        c.set("krishiv.session.memoryLimit", "4g").unwrap();
        assert_eq!(query_memory_limit(&c), Some(4 << 30));
        assert_eq!(Settings::new(&c).memory_tier, Some(64 << 20));
        c.set("krishiv.runtimeFilter.bloom.memoryTier", "false")
            .unwrap();
        assert_eq!(Settings::new(&c).memory_tier, None);
    }

    fn settings(pairs: &[(&str, &str)]) -> Settings {
        let mut c = Config::new();
        c.set("krishiv.memory.limit", "20g").unwrap();
        for (k, v) in pairs {
            c.set(k, v).unwrap();
        }
        Settings::new(&c)
    }

    fn caps(s: Settings) -> (u64, usize) {
        (s.bloom_max_items, s.bloom_max_bytes)
    }

    /// A build within the at-scale key cap gets the at-scale caps from a
    /// scan 20 times it (main's plans, part A's 8x killed); a build past it
    /// gets the tier's (320 MiB at 20g) from a scan 8 times it (part B), or
    /// the at-scale caps from 20 times with the tier off.
    #[test]
    fn the_caps_a_scan_gets() {
        let base = (4_000_000, 8 << 20);
        let tier = (keys_for(320 << 20), 320 << 20);
        let at_scale = (BLOOM_AT_SCALE_MAX_ITEMS, BLOOM_AT_SCALE_MAX_BYTES);
        let s = settings(&[]);
        assert_eq!(caps(s.for_scan(8e6, 1e6)), base);
        assert_eq!(caps(s.for_scan(19.9e6, 1e6)), base);
        assert_eq!(caps(s.for_scan(20e6, 1e6)), at_scale);
        assert_eq!(caps(s.for_scan(20e9, 1e6)), at_scale);
        assert_eq!(caps(s.for_scan(7.9 * 30e6, 30e6)), base);
        assert_eq!(caps(s.for_scan(8.0 * 30e6, 30e6)), tier);
        let off = settings(&[("krishiv.runtimeFilter.bloom.memoryTier", "false")]);
        assert_eq!(caps(off.for_scan(8.0 * 30e6, 30e6)), base);
        assert_eq!(caps(off.for_scan(20.0 * 30e6, 30e6)), at_scale);
    }

    /// The fixed Bloom of a guessed build is sized for a quarter of its scan
    /// within the tier: q09 at SF1000 (1.5B `orders`
    /// rows) keeps the whole 320 MiB at 20g, a 17M-row scan about 20 MiB;
    /// an explicit `bloom.maxBytes` binds; none with the tier off.
    #[test]
    fn the_fixed_bloom_follows_the_scan_within_the_tier() {
        let s = settings(&[]);
        assert_eq!(s.spill_bloom_bytes(1.5e9, 0.01), Some(320 << 20));
        let small = s.spill_bloom_bytes(17e6, 0.01).unwrap_or(0);
        assert!(small > 1 << 20 && small < 32 << 20, "{small}");
        let explicit = settings(&[("krishiv.runtimeFilter.bloom.maxBytes", "8m")]);
        assert_eq!(explicit.spill_bloom_bytes(1.5e9, 0.01), Some(8 << 20));
        let off = settings(&[("krishiv.runtimeFilter.bloom.memoryTier", "false")]);
        assert_eq!(off.spill_bloom_bytes(1.5e9, 0.01), None);
    }

    /// An explicit cap binds under the tier as under `atScale`; an explicit
    /// size cap also bounds the tier's key cap, so a build never collects
    /// hashes for more keys than its Bloom can hold (review M1).
    #[test]
    fn explicit_caps_bind_under_the_memory_tier() {
        let bytes = settings(&[("krishiv.runtimeFilter.bloom.maxBytes", "8m")]).for_scan(1e9, 30e6);
        assert_eq!(bytes.bloom_max_bytes, 8 << 20);
        assert_eq!(
            bytes.bloom_max_items,
            BLOOM_AT_SCALE_MAX_ITEMS.max(keys_for(8 << 20))
        );
        let items =
            settings(&[("krishiv.runtimeFilter.bloom.maxItems", "100")]).for_scan(1e9, 30e6);
        assert_eq!(caps(items), (100, 320 << 20));
        // Set explicitly, `atScale` lifts explicit caps again, to the tier's.
        let lifted = settings(&[
            ("krishiv.runtimeFilter.bloom.maxItems", "100"),
            ("krishiv.runtimeFilter.bloom.atScale", "true"),
        ])
        .for_scan(1e9, 30e6);
        assert_eq!(caps(lifted), (keys_for(320 << 20), 320 << 20));
    }
}
