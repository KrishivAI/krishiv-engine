// SPDX-License-Identifier: Apache-2.0
//! Metrics for `EXPLAIN ANALYZE`: each driver counts in plain fields and
//! merges into the query once, when it ends (no per-batch atomics).
//!
//! Every type here is plain data owned by the query that produced it:
//! nothing is registered globally. The `merge` methods are associative and
//! commutative, so the metrics of the drivers of one worker, and of the
//! workers of one query, combine in any order (sums add, maxima take the
//! larger value).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::context::OperatorId;

/// Counters of one operator, summed over its drivers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorMetrics {
    /// Drivers that ran the operator.
    pub drivers: u64,
    /// Rows received (operators and sinks).
    pub rows_in: u64,
    /// Batches received.
    pub batches_in: u64,
    /// Rows produced (sources and operators).
    pub rows_out: u64,
    /// Batches produced.
    pub batches_out: u64,
    /// Time in `execute` / `sink` / `finalize`; for a source, in
    /// `get_data` while it was not waiting.
    pub compute_time: Duration,
    /// The largest [`compute_time`](Self::compute_time) of one driver (skew).
    pub max_driver_compute_time: Duration,
    /// Time a source's `get_data` was pending (I/O).
    pub blocked_on_io: Duration,
    /// The largest [`blocked_on_io`](Self::blocked_on_io) of one driver.
    pub max_driver_blocked_on_io: Duration,
    /// Morsels claimed (sources; reported by the source itself).
    pub morsels: u64,
    /// Peak bytes reserved through the operator's memory consumer, summed
    /// over its drivers (an upper bound of the true peak).
    pub peak_reserved: u64,
    /// Spills serviced.
    pub spills: u64,
    /// Bytes freed by spills.
    pub spilled_bytes: u64,
    /// Time waiting for memory after `NeedMemory`.
    pub blocked_on_memory: Duration,
    /// Time waiting on a `Blocked` sink (backpressure).
    pub blocked_on_sink: Duration,
}

impl OperatorMetrics {
    /// Adds `o` (another driver's or worker's counters), saturating (a
    /// merged profile may come from a document).
    pub fn merge(&mut self, o: &Self) {
        self.drivers = self.drivers.saturating_add(o.drivers);
        self.rows_in = self.rows_in.saturating_add(o.rows_in);
        self.batches_in = self.batches_in.saturating_add(o.batches_in);
        self.rows_out = self.rows_out.saturating_add(o.rows_out);
        self.batches_out = self.batches_out.saturating_add(o.batches_out);
        self.compute_time = self.compute_time.saturating_add(o.compute_time);
        self.max_driver_compute_time = self.max_driver_compute_time.max(o.max_driver_compute_time);
        self.blocked_on_io = self.blocked_on_io.saturating_add(o.blocked_on_io);
        self.max_driver_blocked_on_io = self
            .max_driver_blocked_on_io
            .max(o.max_driver_blocked_on_io);
        self.morsels = self.morsels.saturating_add(o.morsels);
        self.peak_reserved = self.peak_reserved.saturating_add(o.peak_reserved);
        self.spills = self.spills.saturating_add(o.spills);
        self.spilled_bytes = self.spilled_bytes.saturating_add(o.spilled_bytes);
        self.blocked_on_memory = self.blocked_on_memory.saturating_add(o.blocked_on_memory);
        self.blocked_on_sink = self.blocked_on_sink.saturating_add(o.blocked_on_sink);
    }

    /// One driver's counters as a contribution: one driver, whose times
    /// are also the per-driver maxima.
    pub(crate) fn as_driver(&mut self) {
        self.drivers = 1;
        self.max_driver_compute_time = self.compute_time;
        self.max_driver_blocked_on_io = self.blocked_on_io;
    }

    pub(crate) fn input(&mut self, rows: usize) {
        self.rows_in += rows as u64;
        self.batches_in += 1;
    }

    pub(crate) fn output(&mut self, rows: usize) {
        self.rows_out += rows as u64;
        self.batches_out += 1;
    }
}

/// Scheduling counters, summed (or maxed) over the query's drivers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DriverStats {
    /// Drivers run.
    pub drivers: u64,
    /// Source batches pushed.
    pub batches: u64,
    /// Yields to the scheduler.
    pub yields: u64,
    /// Longest single poll of a driver (a non-yielding slice).
    pub max_slice: Duration,
    /// Polls longer than the stall threshold.
    pub stalls: u64,
}

impl DriverStats {
    /// Adds `o`.
    pub fn merge(&mut self, o: &Self) {
        self.drivers = self.drivers.saturating_add(o.drivers);
        self.batches = self.batches.saturating_add(o.batches);
        self.yields = self.yields.saturating_add(o.yields);
        self.max_slice = self.max_slice.max(o.max_slice);
        self.stalls = self.stalls.saturating_add(o.stalls);
    }
}

/// The value of an operator-specific metric, with how it merges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Metric {
    /// A count: merges by addition. A name ending in `_bytes` is a size.
    Count(u64),
    /// A maximum (a recursion depth, a longest probe): merges by `max`.
    Max(u64),
    /// A description (a filter's form, a hash table's index kind): merging
    /// keeps each distinct `"; "`-separated part once, sorted.
    Text(String),
}

impl Metric {
    /// Merges `o` into this value, associatively and commutatively: two
    /// maxima take the larger; any other two numbers add as a count; texts
    /// keep their distinct `"; "`-separated parts, sorted; a text wins over
    /// a number.
    pub fn merge(&mut self, o: &Self) {
        match (&mut *self, o) {
            (Self::Max(a), Self::Max(b)) => *a = (*a).max(*b),
            (Self::Count(a) | Self::Max(a), Self::Count(b) | Self::Max(b)) => {
                *self = Self::Count(a.saturating_add(*b));
            }
            (Self::Text(a), Self::Text(b)) => {
                let mut parts: Vec<&str> = a.split("; ").chain(b.split("; ")).collect();
                parts.sort_unstable();
                parts.dedup();
                *a = parts.join("; ");
            }
            (Self::Text(_), _) => {}
            (_, Self::Text(b)) => *self = Self::Text(b.clone()),
        }
    }
}

impl Metric {
    /// The number of a count or maximum.
    #[must_use]
    pub fn value(&self) -> Option<u64> {
        match self {
            Self::Count(n) | Self::Max(n) => Some(*n),
            Self::Text(_) => None,
        }
    }
}

impl From<u64> for Metric {
    fn from(v: u64) -> Self {
        Self::Count(v)
    }
}

/// Operator-specific metrics: `(name, value)`, names unique per operator.
pub type ExtraMetrics = Vec<(Cow<'static, str>, Metric)>;

/// Counts under static names, as most operators report them.
pub(crate) fn counts<const N: usize>(values: [(&'static str, u64); N]) -> ExtraMetrics {
    values
        .into_iter()
        .map(|(n, v)| (Cow::Borrowed(n), Metric::Count(v)))
        .collect()
}

/// The probe-length metrics of `c`: nothing unless lookups were counted
/// (`ExecConfig::detailed_metrics`).
pub(crate) fn probe_metrics(c: &crate::hash::ProbeCount) -> ExtraMetrics {
    if c.lookups == 0 {
        return Vec::new();
    }
    vec![
        (Cow::Borrowed("probe_lookups"), Metric::Count(c.lookups)),
        (
            Cow::Borrowed("probe_key_compares"),
            Metric::Count(c.compares),
        ),
        (Cow::Borrowed("max_probe_length"), Metric::Max(c.max)),
    ]
}

/// Merges `from` into `into` by name.
pub(crate) fn merge_extra(into: &mut ExtraMetrics, from: &ExtraMetrics) {
    for (name, v) in from {
        match into.iter_mut().find(|(n, _)| n == name) {
            Some((_, have)) => have.merge(v),
            None => into.push((name.clone(), v.clone())),
        }
    }
}

/// Where an operator sits in its pipelines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperatorRole {
    /// Produces a pipeline's batches (a scan, a breaker's output side).
    Source,
    /// Transforms batches.
    #[default]
    Operator,
    /// Ends a pipeline (a breaker's input side, the result).
    Sink,
}

impl OperatorRole {
    /// `source`, `operator` or `sink`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Operator => "operator",
            Self::Sink => "sink",
        }
    }
}

/// One operator's metrics with its name and place in the plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorEntry {
    /// The operator's name.
    pub name: Cow<'static, str>,
    /// The physical plan node it implements (the node's pre-order index in
    /// [`PhysicalPlan::root`](crate::physical::PhysicalPlan::root)); `None`
    /// for the query's result sink and graphs built without a plan.
    pub node: Option<u32>,
    /// Its role.
    pub role: OperatorRole,
    /// The pipelines it runs in (several below a `UNION ALL`), ascending.
    pub pipelines: Vec<u32>,
    /// Its counters.
    pub metrics: OperatorMetrics,
    /// Operator-specific counters (`Operator::extra_metrics`,
    /// `Sink::extra_metrics`), e.g. an aggregate's groups and spill runs.
    pub extra: ExtraMetrics,
}

impl OperatorEntry {
    /// Adds `o`, the same operator's entry from another worker.
    pub fn merge(&mut self, o: &Self) {
        if self.name.is_empty() {
            self.name.clone_from(&o.name);
        }
        self.node = self.node.or(o.node);
        self.role = o.role;
        for p in &o.pipelines {
            if !self.pipelines.contains(p) {
                self.pipelines.push(*p);
            }
        }
        self.pipelines.sort_unstable();
        self.metrics.merge(&o.metrics);
        merge_extra(&mut self.extra, &o.extra);
    }

    /// The value of the extra metric `name`, if any.
    #[must_use]
    pub fn extra(&self, name: &str) -> Option<&Metric> {
        self.extra.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

/// One pipeline's run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipelineMetrics {
    /// Drivers run.
    pub drivers: u64,
    /// Distinct worker threads that polled them, summed over workers.
    pub threads: u64,
    /// When its first driver started, after the query started.
    pub started: Duration,
    /// When its last driver ended, after the query started.
    pub ended: Duration,
}

impl PipelineMetrics {
    /// Wall time from its first driver's start to its last driver's end.
    #[must_use]
    pub fn wall_time(&self) -> Duration {
        self.ended.saturating_sub(self.started)
    }

    /// Adds `o` (another worker's run of the pipeline).
    pub fn merge(&mut self, o: &Self) {
        if self.drivers == 0 {
            self.started = o.started;
        } else if o.drivers > 0 {
            self.started = self.started.min(o.started);
        }
        self.ended = self.ended.max(o.ended);
        self.drivers = self.drivers.saturating_add(o.drivers);
        self.threads = self.threads.saturating_add(o.threads);
    }
}

/// A query's metrics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    /// Per operator.
    pub operators: BTreeMap<OperatorId, OperatorEntry>,
    /// Per pipeline, by index.
    pub pipelines: BTreeMap<u32, PipelineMetrics>,
    /// Scheduling.
    pub drivers: DriverStats,
    /// Wall time from the start of execution to its end (the longest over
    /// workers).
    pub wall_time: Duration,
    /// Workers whose metrics this holds.
    pub workers: u64,
}

impl QueryMetrics {
    /// Adds `o`: the metrics of the same plan run by another worker (or of
    /// another set of drivers). Operators and pipelines are matched by id,
    /// which the plan fixes.
    pub fn merge(&mut self, o: &Self) {
        for (id, e) in &o.operators {
            self.operators.entry(*id).or_default().merge(e);
        }
        for (id, p) in &o.pipelines {
            self.pipelines.entry(*id).or_default().merge(p);
        }
        self.drivers.merge(&o.drivers);
        self.wall_time = self.wall_time.max(o.wall_time);
        self.workers = self.workers.saturating_add(o.workers);
    }
}

/// A small dense id of the current thread, for counting the threads that
/// ran a pipeline: the first thread to ask gets 0. A label only; nothing
/// is shared through it.
pub(crate) fn thread_ordinal() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::thread_local! {
        static ORDINAL: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    ORDINAL.with(|o| *o)
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn op(compute_ms: u64, rows: u64) -> OperatorMetrics {
        let mut m = OperatorMetrics {
            rows_out: rows,
            batches_out: 1,
            compute_time: Duration::from_millis(compute_ms),
            blocked_on_io: Duration::from_millis(compute_ms / 2),
            ..OperatorMetrics::default()
        };
        m.as_driver();
        m
    }

    #[test]
    fn drivers_merge_sums_and_maxima() {
        let mut total = OperatorMetrics::default();
        for (ms, rows) in [(5, 10), (40, 1), (7, 3)] {
            total.merge(&op(ms, rows));
        }
        assert_eq!(total.drivers, 3);
        assert_eq!(total.rows_out, 14);
        assert_eq!(total.batches_out, 3);
        assert_eq!(total.compute_time, Duration::from_millis(52));
        assert_eq!(total.max_driver_compute_time, Duration::from_millis(40));
        assert_eq!(total.max_driver_blocked_on_io, Duration::from_millis(20));
    }

    #[test]
    fn merge_is_associative_and_commutative() {
        let a = op(1, 2);
        let b = op(30, 4);
        let c = op(9, 8);
        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);
        let mut bc = c.clone();
        bc.merge(&b);
        let mut right = a;
        right.merge(&bc);
        assert_eq!(left, right);
    }

    #[test]
    fn metrics_merge_by_kind() {
        let mut m = Metric::Count(2);
        m.merge(&Metric::Count(3));
        assert_eq!(m, Metric::Count(5));
        let mut m = Metric::Max(2);
        m.merge(&Metric::Max(7));
        m.merge(&Metric::Max(4));
        assert_eq!(m, Metric::Max(7));
        let mut m = Metric::Text("bloom".to_owned());
        m.merge(&Metric::Text("bloom".to_owned()));
        assert_eq!(m, Metric::Text("bloom".to_owned()));
        m.merge(&Metric::Text("range".to_owned()));
        m.merge(&Metric::Text("bloom".to_owned()));
        assert_eq!(m, Metric::Text("bloom; range".to_owned()));
        // Merged texts merge part by part, in either order.
        let mut x = Metric::Text("range; bloom".to_owned());
        x.merge(&m);
        assert_eq!(x, m);
        let mut y = Metric::Text("small".to_owned());
        y.merge(&Metric::Text("bloom; range".to_owned()));
        let mut z = Metric::Text("bloom; range".to_owned());
        z.merge(&Metric::Text("small".to_owned()));
        assert_eq!(y, z);
        // A count and a maximum merge to a count, in either order.
        let mut c = Metric::Count(2);
        c.merge(&Metric::Max(5));
        let mut d = Metric::Max(5);
        d.merge(&Metric::Count(2));
        assert_eq!((c, d), (Metric::Count(7), Metric::Count(7)));
        m.merge(&Metric::Count(1));
        assert_eq!(m, Metric::Text("bloom; range".to_owned()));
        let mut m = Metric::Count(1);
        m.merge(&Metric::Text("x".to_owned()));
        assert_eq!(m, Metric::Text("x".to_owned()));
    }

    #[test]
    fn query_metrics_merge_across_workers() {
        let worker = |rows: u64, started: u64, ended: u64| {
            let mut q = QueryMetrics {
                workers: 1,
                wall_time: Duration::from_millis(ended),
                ..QueryMetrics::default()
            };
            q.operators.insert(
                OperatorId(1),
                OperatorEntry {
                    name: Cow::Borrowed("HashAggregate"),
                    node: Some(0),
                    role: OperatorRole::Sink,
                    pipelines: vec![1],
                    metrics: op(rows, rows),
                    extra: vec![
                        (Cow::Borrowed("groups"), Metric::Count(rows)),
                        (Cow::Borrowed("max_probe_length"), Metric::Max(rows)),
                    ],
                },
            );
            q.pipelines.insert(
                1,
                PipelineMetrics {
                    drivers: 2,
                    threads: 2,
                    started: Duration::from_millis(started),
                    ended: Duration::from_millis(ended),
                },
            );
            q
        };
        let mut q = worker(10, 5, 50);
        q.merge(&worker(4, 2, 30));
        assert_eq!(q.workers, 2);
        assert_eq!(q.wall_time, Duration::from_millis(50));
        let e = &q.operators[&OperatorId(1)];
        assert_eq!(e.metrics.rows_out, 14);
        assert_eq!(e.metrics.drivers, 2);
        assert_eq!(e.extra("groups"), Some(&Metric::Count(14)));
        assert_eq!(e.extra("max_probe_length"), Some(&Metric::Max(10)));
        assert_eq!(e.pipelines, vec![1]);
        let p = &q.pipelines[&1];
        assert_eq!((p.drivers, p.threads), (4, 4));
        assert_eq!(p.wall_time(), Duration::from_millis(48));
    }

    #[test]
    fn thread_ordinals_are_stable_per_thread() {
        let here = thread_ordinal();
        assert_eq!(here, thread_ordinal());
        let other = std::thread::spawn(thread_ordinal).join().unwrap_or(here);
        assert_ne!(here, other);
    }
}
