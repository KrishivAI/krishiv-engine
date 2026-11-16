// SPDX-License-Identifier: Apache-2.0
//! The advisor's hints (P3-T073): one pure function from a finished query's plan tree, metrics and the
//! planner facts the nodes carry to a list of [`Hint`]s. `EXPLAIN ANALYZE`, `system.queries` and the stored
//! profile all read its output, so the three agree. It runs after execution and never touches planning.
//!
//! The codes are stable (`SPILL`, `MISSING_STATISTICS`, `SKEW`, `RUNTIME_FILTER_OPPORTUNITY`,
//! `MEMORY_WAIT`): messages may change, a new code is additive. Thresholds are `krishiv.advisor.*`.

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
    /// A table without statistics made the planner mis-estimate a join input.
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
                format!("node {} {}: ", self.node.unwrap_or(0), n.name)
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
        let int = |e: &ConfigEntry, d: u64| {
            config
                .get(e)
                .and_then(krishiv_common::config::ConfigValue::as_int)
                .and_then(|v| u64::try_from(v).ok())
                .unwrap_or(d)
        };
        let time = |e: &ConfigEntry, d: Duration| {
            config
                .get(e)
                .and_then(krishiv_common::config::ConfigValue::as_duration)
                .unwrap_or(d)
        };
        Self {
            enabled: config
                .get(&ADVISOR_ENABLED)
                .and_then(krishiv_common::config::ConfigValue::as_bool)
                .unwrap_or(true),
            spill_min_percent: int(&ADVISOR_SPILL_MIN_PERCENT, 10),
            spill_min_bytes: config
                .get(&ADVISOR_SPILL_MIN_BYTES)
                .and_then(krishiv_common::config::ConfigValue::as_size)
                .unwrap_or(64 << 20),
            memory_wait_factor: int(&ADVISOR_MEMORY_WAIT_FACTOR, 1),
            memory_wait_min_time: time(&ADVISOR_MEMORY_WAIT_MIN_TIME, Duration::from_millis(100)),
            skew_factor: int(&ADVISOR_SKEW_FACTOR, 4),
            skew_min_time: time(&ADVISOR_SKEW_MIN_TIME, Duration::from_millis(200)),
            skew_min_rows: int(&ADVISOR_SKEW_MIN_ROWS, 100_000),
            missing_stats_min_q_error: int(&ADVISOR_MISSING_STATS_MIN_Q_ERROR, 4),
            missing_stats_min_rows: int(&ADVISOR_MISSING_STATS_MIN_ROWS, 10_000),
            runtime_filter_min_probe_rows: int(&ADVISOR_RUNTIME_FILTER_MIN_PROBE_ROWS, 1_000_000),
            runtime_filter_min_drop_percent: int(&ADVISOR_RUNTIME_FILTER_MIN_DROP_PERCENT, 50),
            max_hints: usize::try_from(int(&ADVISOR_MAX_HINTS, 32)).unwrap_or(32),
        }
    }
}

/// Operators whose drivers each own a key range, so morsel stealing cannot rebalance them: a slow driver
/// is key skew. A scan's drivers share morsels, so a slow scan driver is I/O, not skew.
const KEY_PARTITIONED: [&str; 5] = [
    "HashJoinBuild",
    "HashJoinProbe",
    "HashAggregate",
    "Window",
    "ExchangeRead",
];

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

fn at(plan: &ProfileNode, node: Option<u32>) -> String {
    match node.and_then(|id| find(plan, id)) {
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
            out.push(Hint {
                code: HintCode::Spill,
                node: e.node,
                operator: Some(*id),
                message: format!(
                    "{} spilled {} to disk ({pct:.0}% of the query's {} peak memory). Raise \
                     krishiv.memory.limit, or reduce the rows it holds at once (filter or aggregate earlier).",
                    at(plan, e.node),
                    bytes(m.spill_write_bytes),
                    bytes(peak)
                ),
                evidence: ev(&[
                    ("spill_write_bytes", m.spill_write_bytes),
                    ("spilled_bytes", m.spilled_bytes),
                    ("peak_reserved_bytes", peak),
                ]),
            });
        }
        // MEMORY_WAIT.
        if m.blocked_on_memory >= settings.memory_wait_min_time
            && m.blocked_on_memory
                > m.compute_time
                    .saturating_mul(u32::try_from(settings.memory_wait_factor).unwrap_or(u32::MAX))
        {
            out.push(Hint {
                code: HintCode::MemoryWait,
                node: e.node,
                operator: Some(*id),
                message: format!(
                    "{} waited {} ms for memory and computed for {} ms. Raise krishiv.memory.limit or run fewer queries at once.",
                    at(plan, e.node),
                    m.blocked_on_memory.as_millis(),
                    m.compute_time.as_millis()
                ),
                evidence: ev(&[
                    ("blocked_on_memory_ns", nanos(m.blocked_on_memory)),
                    ("compute_time_ns", nanos(m.compute_time)),
                ]),
            });
        }
        skew(plan, *id, e, settings, &mut out);
    }
    joins(plan, metrics, settings, &mut out);
    out.sort_by(|a, b| {
        (a.code, a.node, a.operator.map(|o| o.0)).cmp(&(b.code, b.node, b.operator.map(|o| o.0)))
    });
    out.truncate(settings.max_hints);
    out
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

fn skew(
    plan: &ProfileNode,
    id: OperatorId,
    e: &OperatorEntry,
    s: &HintSettings,
    out: &mut Vec<Hint>,
) {
    let m = &e.metrics;
    if !KEY_PARTITIONED.contains(&e.name.as_ref())
        || m.drivers < 2
        || m.max_driver_compute_time < s.skew_min_time
    {
        return;
    }
    let drivers = m.drivers;
    let by_time = u128::from(nanos(m.max_driver_compute_time)) * u128::from(drivers)
        >= u128::from(s.skew_factor) * u128::from(nanos(m.compute_time));
    let by_rows = m.rows_out >= s.skew_min_rows
        && u128::from(m.max_driver_rows_out) * u128::from(drivers)
            >= u128::from(s.skew_factor) * u128::from(m.rows_out);
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
    let keys = node_keys(plan, e.node);
    out.push(Hint {
        code: HintCode::Skew,
        node: e.node,
        operator: Some(id),
        message: format!(
            "One of {drivers} drivers of {} did {share:.0}% of the work{keys}. Salting the key or filtering the hot values may help.",
            at(plan, e.node),
        ),
        evidence: ev(&[
            ("drivers", drivers),
            ("max_driver_compute_time_ns", nanos(m.max_driver_compute_time)),
            ("compute_time_ns", nanos(m.compute_time)),
            ("max_driver_rows_out", m.max_driver_rows_out),
            ("rows_out", m.rows_out),
        ]),
    });
}

/// `; the key [..] is skewed`, from the node's detail (the join or grouping keys).
fn node_keys(plan: &ProfileNode, node: Option<u32>) -> String {
    let Some(n) = node.and_then(|id| find(plan, id)) else {
        return String::new();
    };
    match (n.detail.find('['), n.detail.find(']')) {
        (Some(a), Some(b)) if a < b => format!("; the key {} is skewed", &n.detail[a..=b]),
        _ => String::new(),
    }
}

fn operator_at<'a>(
    metrics: &'a QueryMetrics,
    node: u32,
    name: &str,
) -> Option<(OperatorId, &'a OperatorEntry)> {
    metrics
        .operators
        .iter()
        .find(|(_, e)| e.node == Some(node) && e.name == name)
        .map(|(id, e)| (*id, e))
}

fn joins(plan: &ProfileNode, metrics: &QueryMetrics, s: &HintSettings, out: &mut Vec<Hint>) {
    let mut stats: Vec<(String, f64, Hint)> = Vec::new();
    let mut stack = vec![plan];
    while let Some(n) = stack.pop() {
        stack.extend(n.children.iter());
        let (Some(id), Some(facts)) = (n.id, n.facts.as_deref()) else {
            continue;
        };
        if n.name == "HashJoin" {
            missing_stats(plan, id, facts, metrics, s, &mut stats);
            runtime_filters(plan, id, facts, metrics, s, out);
        }
    }
    // One hint per table: the node where the estimate missed most.
    stats.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    stats.dedup_by(|b, a| a.0 == b.0);
    out.extend(stats.into_iter().map(|(_, _, h)| h));
}

fn missing_stats(
    plan: &ProfileNode,
    node: u32,
    facts: &NodeFacts,
    metrics: &QueryMetrics,
    s: &HintSettings,
    out: &mut Vec<(String, f64, Hint)>,
) {
    let probe = operator_at(metrics, node, "HashJoinProbe");
    let build = operator_at(metrics, node, "HashJoinBuild");
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
    plan: &ProfileNode,
    node: u32,
    facts: &NodeFacts,
    metrics: &QueryMetrics,
    s: &HintSettings,
    out: &mut Vec<Hint>,
) {
    if facts.filter_gaps.is_empty() {
        return;
    }
    let Some((id, probe)) = operator_at(metrics, node, "HashJoinProbe") else {
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
                "the probe key ({}) is computed, so no filter reaches the scan; compare the column itself.",
                gap.probe_key
            ),
            _ => format!(
                "an operator between the join and the scan (a LIMIT, an outer join or a subquery) stops a filter on {}.",
                gap.probe_key
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
