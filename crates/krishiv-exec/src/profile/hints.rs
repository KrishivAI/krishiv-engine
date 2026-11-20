// SPDX-License-Identifier: Apache-2.0
//! The advisor's hints (P3-T073): one pure function from a finished query's plan tree, metrics and the
//! planner facts the nodes carry to a list of [`Hint`]s. `EXPLAIN ANALYZE`, `system.queries` and the stored
//! profile all read its output, so the three agree. It runs after execution and never touches planning.
//!
//! The codes are stable (`SPILL`, `MISSING_STATISTICS`, `SKEW`, `RUNTIME_FILTER_OPPORTUNITY`,
//! `MEMORY_WAIT`): messages may change, a new code is additive. Thresholds are `krishiv.advisor.*`.

use std::collections::HashMap;
use std::time::Duration;

use krishiv_common::config::{
    ADVISOR_ENABLED, ADVISOR_MAX_HINTS, ADVISOR_MEMORY_WAIT_FACTOR, ADVISOR_MEMORY_WAIT_MIN_TIME,
    ADVISOR_MISSING_STATS_MIN_Q_ERROR, ADVISOR_MISSING_STATS_MIN_ROWS,
    ADVISOR_RUNTIME_FILTER_MIN_DROP_PERCENT, ADVISOR_RUNTIME_FILTER_MIN_PROBE_ROWS,
    ADVISOR_SKEW_FACTOR, ADVISOR_SKEW_MIN_ROWS, ADVISOR_SKEW_MIN_TIME, ADVISOR_SPILL_MIN_BYTES,
    ADVISOR_SPILL_MIN_PERCENT, Config, ConfigEntry,
};

use super::ProfileNode;
use crate::context::OperatorId;
use crate::metrics::{OperatorEntry, QueryMetrics};
use crate::planner::{FilterGapReason, NodeFacts};

/// What a hint is about. The order is the order hints are listed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum HintCode {
    /// An operator wrote a large share of the query's memory to disk.
    Spill,
    /// A table without statistics made the planner misestimate a join input.
    MissingStatistics,
    /// One driver of a key-partitioned operator did most of the work.
    Skew,
    /// A hash join dropped many probe rows that a runtime filter could have skipped at the scan.
    RuntimeFilterOpportunity,
    /// An operator waited for memory for longer than it computed.
    MemoryWait,
}

impl HintCode {
    /// The stable code string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Spill => "SPILL",
            Self::MissingStatistics => "MISSING_STATISTICS",
            Self::Skew => "SKEW",
            Self::RuntimeFilterOpportunity => "RUNTIME_FILTER_OPPORTUNITY",
            Self::MemoryWait => "MEMORY_WAIT",
        }
    }

    /// The code named `s`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        [
            Self::Spill,
            Self::MissingStatistics,
            Self::Skew,
            Self::RuntimeFilterOpportunity,
            Self::MemoryWait,
        ]
        .into_iter()
        .find(|c| c.as_str() == s)
    }
}

/// One finding about a finished query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hint {
    /// What kind of finding.
    pub code: HintCode,
    /// The plan node (`ProfileNode::id`) it is about.
    pub node: Option<u32>,
    /// The operator it is about.
    pub operator: Option<OperatorId>,
    /// What happened, with the numbers, and the remedy.
    pub message: String,
    /// The numbers the rule compared.
    pub evidence: Vec<(String, u64)>,
}

impl Hint {
    /// `[CODE] node N Name: message`, the `EXPLAIN ANALYZE` line.
    #[must_use]
    pub fn line(&self, plan: &ProfileNode) -> String {
        let at = self
            .node
            .and_then(|id| find(plan, id))
            .map_or_else(String::new, |n| {
                format!("node {} {}: ", n.id.unwrap_or(0), n.name)
            });
        format!("[{}] {at}{}", self.code.as_str(), self.message)
    }
}

/// The thresholds of the rules (`krishiv.advisor.*`).
#[derive(Clone, Debug, PartialEq)]
pub struct HintSettings {
    /// `krishiv.advisor.enabled`.
    pub enabled: bool,
    /// Spill written over peak memory, in percent.
    pub spill_min_percent: u64,
    /// Floor on spill written.
    pub spill_min_bytes: u64,
    /// Blocked on memory over compute time.
    pub memory_wait_factor: u64,
    /// Floor on time blocked.
    pub memory_wait_min_time: Duration,
    /// Slowest driver times drivers over the total.
    pub skew_factor: u64,
    /// Floor on the slowest driver's time.
    pub skew_min_time: Duration,
    /// Floor on rows for the row-based test.
    pub skew_min_rows: u64,
    /// Q-error at which a missing statistic is worth a hint.
    pub missing_stats_min_q_error: u64,
    /// Floor on the larger of estimate and actual.
    pub missing_stats_min_rows: u64,
    /// Floor on probe rows.
    pub runtime_filter_min_probe_rows: u64,
    /// Lower bound on probe rows dropped, in percent.
    pub runtime_filter_min_drop_percent: u64,
    /// Hints kept per query.
    pub max_hints: usize,
}

impl Default for HintSettings {
    fn default() -> Self {
        Self::from_config(&Config::default())
    }
}

impl HintSettings {
    /// The settings in `config` (the registry defaults for what is unset).
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        // `Config::get` falls back to the entry's registry default, so no default is repeated here; the
        // `unwrap_or_default` is for an entry the registry validated, which always holds a value.
        let int = |e: &ConfigEntry| {
            config
                .get(e)
                .and_then(krishiv_common::config::ConfigValue::as_int)
                .and_then(|v| u64::try_from(v).ok())
                .unwrap_or_default()
        };
        let time = |e: &ConfigEntry| {
            config
                .get(e)
                .and_then(krishiv_common::config::ConfigValue::as_duration)
                .unwrap_or_default()
        };
        Self {
            enabled: config
                .get(&ADVISOR_ENABLED)
                .and_then(krishiv_common::config::ConfigValue::as_bool)
                .unwrap_or_default(),
            spill_min_percent: int(&ADVISOR_SPILL_MIN_PERCENT),
            spill_min_bytes: config
                .get(&ADVISOR_SPILL_MIN_BYTES)
                .and_then(krishiv_common::config::ConfigValue::as_size)
                .unwrap_or_default(),
            memory_wait_factor: int(&ADVISOR_MEMORY_WAIT_FACTOR),
            memory_wait_min_time: time(&ADVISOR_MEMORY_WAIT_MIN_TIME),
            skew_factor: int(&ADVISOR_SKEW_FACTOR),
            skew_min_time: time(&ADVISOR_SKEW_MIN_TIME),
            skew_min_rows: int(&ADVISOR_SKEW_MIN_ROWS),
            missing_stats_min_q_error: int(&ADVISOR_MISSING_STATS_MIN_Q_ERROR),
            missing_stats_min_rows: int(&ADVISOR_MISSING_STATS_MIN_ROWS),
            runtime_filter_min_probe_rows: int(&ADVISOR_RUNTIME_FILTER_MIN_PROBE_ROWS),
            runtime_filter_min_drop_percent: int(&ADVISOR_RUNTIME_FILTER_MIN_DROP_PERCENT),
            max_hints: usize::try_from(int(&ADVISOR_MAX_HINTS)).unwrap_or(usize::MAX),
        }
    }
}

/// Operators whose drivers each own a key range, so morsel stealing cannot rebalance them: a slow driver
/// is key skew. A scan's drivers, a hash join's build and probe and a hash aggregate's partial and final phases
/// share morsels on one node, so one driver doing most of the work there is scheduling, not key skew (TPC-H
/// SF1 q04, q05 and q21 raised it on those with the first list; they are not listed). What remains owns its
/// keys: a window's partitions and an exchange consumer's reduce partitions.
const KEY_PARTITIONED: [&str; 2] = ["Window", "ExchangeRead"];

fn find(plan: &ProfileNode, id: u32) -> Option<&ProfileNode> {
    if plan.id == Some(id) {
        return Some(plan);
    }
    plan.children.iter().find_map(|c| find(c, id))
}

fn bytes(n: u64) -> String {
    #[expect(clippy::cast_precision_loss, reason = "a rounded size for text")]
    let v = n as f64;
    if v >= 1_073_741_824.0 {
        format!("{:.1} GiB", v / 1_073_741_824.0)
    } else if v >= 1_048_576.0 {
        format!("{:.1} MiB", v / 1_048_576.0)
    } else {
        format!("{n} B")
    }
}

fn count(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn ratio(a: u64, b: u64) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "a ratio")]
    {
        a as f64 / (b.max(1)) as f64
    }
}

/// The plan's nodes by id, built once per `hints` call so each hint's `at` is a lookup, not a tree walk.
type Index<'a> = HashMap<u32, &'a ProfileNode>;

fn index(plan: &ProfileNode) -> Index<'_> {
    let mut out = HashMap::new();
    let mut stack = vec![plan];
    while let Some(n) = stack.pop() {
        if let Some(id) = n.id {
            out.entry(id).or_insert(n);
        }
        stack.extend(n.children.iter());
    }
    out
}

fn at(idx: &Index<'_>, node: Option<u32>) -> String {
    match node.and_then(|id| idx.get(&id)) {
        Some(n) => format!("{} (node {})", n.name, n.id.unwrap_or(0)),
        None => "an operator".to_owned(),
    }
}

fn ev(pairs: &[(&str, u64)]) -> Vec<(String, u64)> {
    pairs.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
}

/// The hints for a finished query: `plan` (with the planner facts its nodes carry), the query's `metrics`
/// and the thresholds. Pure and deterministic, ordered by (code, node, operator) and cut at `max_hints`.
#[must_use]
pub fn hints(plan: &ProfileNode, metrics: &QueryMetrics, settings: &HintSettings) -> Vec<Hint> {
    if !settings.enabled {
        return Vec::new();
    }
    let idx = index(plan);
    let mut out = Vec::new();
    let peak = metrics.memory.peak_reserved;
    for (id, e) in &metrics.operators {
        let m = &e.metrics;
        // SPILL.
        if m.spill_write_bytes >= settings.spill_min_bytes
            && m.spill_write_bytes > 0
            && u128::from(m.spill_write_bytes) * 100
                > u128::from(settings.spill_min_percent) * u128::from(peak)
        {
            let pct = ratio(m.spill_write_bytes.saturating_mul(100), peak);
            let idx = &idx;
            out.push(Cand::lazy(HintCode::Spill, e.node, *id, move || {
                (
                    format!(
                        "{} spilled {} to disk ({pct:.0}% of the query's {} peak memory). Raise \
                         krishiv.memory.limit, or reduce the rows it holds at once (filter or aggregate earlier).",
                        at(idx, e.node),
                        bytes(m.spill_write_bytes),
                        bytes(peak)
                    ),
                    ev(&[
                        ("spill_write_bytes", m.spill_write_bytes),
                        ("spilled_bytes", m.spilled_bytes),
                        ("peak_reserved_bytes", peak),
                    ]),
                )
            }));
        }
        // MEMORY_WAIT.
        if m.blocked_on_memory >= settings.memory_wait_min_time
            && m.blocked_on_memory
                > m.compute_time
                    .saturating_mul(u32::try_from(settings.memory_wait_factor).unwrap_or(u32::MAX))
        {
            let idx = &idx;
            out.push(Cand::lazy(HintCode::MemoryWait, e.node, *id, move || {
                (
                    format!(
                        "{} waited {} ms for memory and computed for {} ms. Raise krishiv.memory.limit or run fewer queries at once.",
                        at(idx, e.node),
                        m.blocked_on_memory.as_millis(),
                        m.compute_time.as_millis()
                    ),
                    ev(&[
                        ("blocked_on_memory_ns", nanos(m.blocked_on_memory)),
                        ("compute_time_ns", nanos(m.compute_time)),
                    ]),
                )
            }));
        }
        skew(&idx, *id, e, settings, &mut out);
    }
    let mut eager = Vec::new();
    joins(plan, &idx, &ops_index(metrics), settings, &mut eager);
    out.extend(eager.into_iter().map(Cand::ready));
    out.sort_by_key(|c| (c.code, c.node, c.operator.map(|o| o.0)));
    out.truncate(settings.max_hints);
    // Messages are built only for the hints that survive the cut.
    out.into_iter().map(Cand::render).collect()
}

type Rendered = (String, Vec<(String, u64)>);

/// A hint whose message and evidence are built when it is kept, not when it is found: a query with
/// hundreds of spilling operators formats `max_hints` messages, not hundreds.
struct Cand<'a> {
    code: HintCode,
    node: Option<u32>,
    operator: Option<OperatorId>,
    render: Box<dyn FnOnce() -> Rendered + 'a>,
}

impl<'a> Cand<'a> {
    fn lazy(
        code: HintCode,
        node: Option<u32>,
        operator: OperatorId,
        render: impl FnOnce() -> Rendered + 'a,
    ) -> Self {
        Self {
            code,
            node,
            operator: Some(operator),
            render: Box::new(render),
        }
    }

    fn ready(h: Hint) -> Self {
        Self {
            code: h.code,
            node: h.node,
            operator: h.operator,
            render: Box::new(move || (h.message, h.evidence)),
        }
    }

    fn render(self) -> Hint {
        let (message, evidence) = (self.render)();
        Hint {
            code: self.code,
            node: self.node,
            operator: self.operator,
            message,
            evidence,
        }
    }
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

fn skew<'a>(
    plan: &'a Index<'_>,
    id: OperatorId,
    e: &'a OperatorEntry,
    s: &HintSettings,
    out: &mut Vec<Cand<'a>>,
) {
    let m = &e.metrics;
    if !KEY_PARTITIONED.contains(&e.name.as_ref())
        || m.drivers < 2
        || m.max_driver_compute_time < s.skew_min_time
    {
        return;
    }
    let drivers = m.drivers;
    // The slowest driver against the mean of the others: `max >= factor * (total - max) / (drivers - 1)`.
    // Unlike a share of the even split, it can fire at 2 and 3 drivers (80% and 67% of the work at factor 4).
    let outlier = |max: u64, total: u64| {
        u128::from(max) * u128::from(drivers - 1)
            >= u128::from(s.skew_factor) * u128::from(total.saturating_sub(max))
    };
    let by_time = outlier(nanos(m.max_driver_compute_time), nanos(m.compute_time));
    let by_rows = m.rows_out >= s.skew_min_rows && outlier(m.max_driver_rows_out, m.rows_out);
    if !(by_time || by_rows) {
        return;
    }
    let share = if by_time {
        ratio(
            nanos(m.max_driver_compute_time).saturating_mul(100),
            nanos(m.compute_time),
        )
    } else {
        ratio(m.max_driver_rows_out.saturating_mul(100), m.rows_out)
    };
    out.push(Cand::lazy(HintCode::Skew, e.node, id, move || {
        (
            format!(
                "One of {drivers} drivers of {} did {share:.0}% of the work. Salting the key or filtering the hot values may help.",
                at(plan, e.node),
            ),
            ev(&[
                ("drivers", drivers),
                ("max_driver_compute_time_ns", nanos(m.max_driver_compute_time)),
                ("compute_time_ns", nanos(m.compute_time)),
                ("max_driver_rows_out", m.max_driver_rows_out),
                ("rows_out", m.rows_out),
            ]),
        )
    }));
}

/// The operators of a query by (plan node, operator name), the first of each, built once per `hints` call.
type Ops<'a> = HashMap<(u32, &'a str), (OperatorId, &'a OperatorEntry)>;

fn ops_index(metrics: &QueryMetrics) -> Ops<'_> {
    let mut out = HashMap::new();
    for (id, e) in &metrics.operators {
        if let Some(node) = e.node {
            out.entry((node, e.name.as_ref())).or_insert((*id, e));
        }
    }
    out
}

fn operator_at<'a>(
    ops: &Ops<'a>,
    node: u32,
    name: &str,
) -> Option<(OperatorId, &'a OperatorEntry)> {
    ops.get(&(node, name)).copied()
}

fn joins(
    plan: &ProfileNode,
    idx: &Index<'_>,
    ops: &Ops<'_>,
    s: &HintSettings,
    out: &mut Vec<Hint>,
) {
    let mut stats: Vec<(String, f64, Hint)> = Vec::new();
    let mut stack = vec![plan];
    while let Some(n) = stack.pop() {
        stack.extend(n.children.iter());
        let (Some(id), Some(facts)) = (n.id, n.facts.as_deref()) else {
            continue;
        };
        if n.name == "HashJoin" {
            missing_stats(idx, id, facts, ops, s, &mut stats);
            runtime_filters(idx, id, facts, ops, s, out);
        }
    }
    // One hint per table: the node where the estimate missed most.
    stats.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    stats.dedup_by(|b, a| a.0 == b.0);
    out.extend(stats.into_iter().map(|(_, _, h)| h));
}

fn missing_stats(
    plan: &Index<'_>,
    node: u32,
    facts: &NodeFacts,
    ops: &Ops<'_>,
    s: &HintSettings,
    out: &mut Vec<(String, f64, Hint)>,
) {
    let probe = operator_at(ops, node, "HashJoinProbe");
    let build = operator_at(ops, node, "HashJoinBuild");
    let actual = [
        probe.map(|(_, e)| e.metrics.rows_in),
        build.map(|(_, e)| e.metrics.rows_in),
    ];
    let side = ["probe", "build"];
    for (i, input) in facts.inputs.iter().take(2).enumerate() {
        let Some(actual) = actual[i] else { continue };
        if input.statistics || input.tables_without_stats.is_empty() {
            continue;
        }
        #[expect(clippy::cast_precision_loss, reason = "estimates")]
        let (est, act) = (input.estimated_rows.max(1.0), (actual.max(1)) as f64);
        let q = (est / act).max(act / est);
        #[expect(clippy::cast_precision_loss, reason = "a threshold")]
        if q < s.missing_stats_min_q_error as f64 {
            continue;
        }
        #[expect(clippy::cast_precision_loss, reason = "a threshold")]
        if est.max(act) < s.missing_stats_min_rows as f64 {
            continue;
        }
        let operator = if i == 0 { probe } else { build };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a row count"
        )]
        let est_rows = est as u64;
        for table in &input.tables_without_stats {
            let remedy = if table.ends_with(" files") || table == "LocalRelation" {
                "Register it as a table and run ANALYZE TABLE ... COMPUTE STATISTICS, or store it as Parquet."
                    .to_owned()
            } else {
                format!("Run ANALYZE TABLE {table} COMPUTE STATISTICS, or store it as Parquet.")
            };
            out.push((
                table.clone(),
                q,
                Hint {
                    code: HintCode::MissingStatistics,
                    node: Some(node),
                    operator: operator.map(|(id, _)| id),
                    message: format!(
                        "Table {table} has no statistics. The planner assumed {} rows for the {} input of {}; it had {}. {remedy}",
                        count(est_rows),
                        side[i],
                        at(plan, Some(node)),
                        count(actual),
                    ),
                    evidence: ev(&[("estimated_rows", est_rows), ("actual_rows", actual)]),
                },
            ));
        }
    }
}

fn runtime_filters(
    plan: &Index<'_>,
    node: u32,
    facts: &NodeFacts,
    ops: &Ops<'_>,
    s: &HintSettings,
    out: &mut Vec<Hint>,
) {
    if facts.filter_gaps.is_empty() {
        return;
    }
    let Some((id, probe)) = operator_at(ops, node, "HashJoinProbe") else {
        return;
    };
    let m = &probe.metrics;
    let dropped = m.rows_in.saturating_sub(m.rows_out);
    if m.rows_in < s.runtime_filter_min_probe_rows
        || u128::from(dropped) * 100
            < u128::from(s.runtime_filter_min_drop_percent) * u128::from(m.rows_in)
    {
        return;
    }
    let pct = ratio(dropped.saturating_mul(100), m.rows_in);
    for gap in &facts.filter_gaps {
        let why = match &gap.reason {
            FilterGapReason::Disabled => {
                "krishiv.runtimeFilter.enabled is false; a filter on the probe key could have skipped them at the scan.".to_owned()
            }
            FilterGapReason::NoScanColumn => format!(
                "join key #{} is computed, so no filter reaches the scan; compare the column itself.",
                gap.key + 1
            ),
            FilterGapReason::Blocked => format!(
                "an operator between the join and the scan (a LIMIT, an outer join or a subquery) stops a filter on join key #{}.",
                gap.key + 1
            ),
        };
        out.push(Hint {
            code: HintCode::RuntimeFilterOpportunity,
            node: Some(node),
            operator: Some(id),
            message: format!(
                "{} dropped at least {pct:.0}% of {} probe rows that a runtime filter could have skipped: {why}",
                at(plan, Some(node)),
                count(m.rows_in)
            ),
            evidence: ev(&[
                ("key", gap.key as u64),
                ("probe_rows_in", m.rows_in),
                ("probe_rows_out", m.rows_out),
            ]),
        });
    }
}

#[cfg(all(test, not(loom)))]
mod tests;
