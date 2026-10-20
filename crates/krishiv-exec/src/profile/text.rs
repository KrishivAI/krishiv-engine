// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN ANALYZE`'s text: the physical plan tree, each node followed by
//! its operators' counters, then the pipelines.
//!
//! ```text
//! == Physical Plan (analyzed) ==
//! wall 12.3 ms, 3 pipelines, 6 drivers, 2 threads, 1 worker
//! HashAggregate [l_returnflag#8], [count(1)]
//! :  · sink   #2 HashAggregate p1: in 600 rows (1 batch); compute 1.2 ms (max 0.8 ms, 2 drivers); peak mem 64.0 KiB; groups=4 hash_entries=4 hash_buckets=8 (load 0.50) hash_index=swiss
//! :  · source #4 HashAggregate p0: out 4 rows (1 batch); compute 0.1 ms (max 0.1 ms, 1 driver)
//! +- Scan lineitem [...]
//!       · source #3 Scan p1: out 600 rows (1 batch); compute 2.0 ms (max 1.1 ms, 2 drivers); io 0.4 ms (max 0.3 ms)
//! ```
//!
//! Operators are listed in id order under their node; times are sums over
//! drivers with the largest single driver's share (skew) next to them.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Duration;

use super::{ProfileNode, QueryProfile};
use crate::context::OperatorId;
use crate::metrics::{Metric, OperatorEntry, OperatorRole};

/// How [`QueryProfile::to_text`] prints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TextOptions {
    /// Print `<t>` for times, `<mem>` for memory and `<n>` for driver and
    /// thread counts and the hash-seed-dependent table statistics: the
    /// values that change from run to run (golden tests).
    pub mask_volatile: bool,
}

impl TextOptions {
    /// With volatile values masked.
    #[must_use]
    pub fn masked() -> Self {
        Self {
            mask_volatile: true,
        }
    }
}

impl QueryProfile {
    /// The profile as `EXPLAIN ANALYZE` prints it.
    #[must_use]
    pub fn to_text(&self, opts: TextOptions) -> String {
        let p = Printer {
            opts,
            by_node: by_node(self),
        };
        let m = &self.metrics;
        let mut out = String::from("== Physical Plan (analyzed) ==\n");
        let threads: u64 = m.pipelines.values().map(|p| p.threads).max().unwrap_or(0);
        let _ = writeln!(
            out,
            "wall {}, {} pipeline{}, {} driver{}, {} thread{} (most in one pipeline), {} worker{}",
            p.time(m.wall_time),
            m.pipelines.len(),
            plural(m.pipelines.len() as u64),
            p.count(m.drivers.drivers),
            p.pl(m.drivers.drivers),
            p.count(threads),
            p.pl(threads),
            m.workers,
            plural(m.workers),
        );
        if !self.detailed {
            out.push_str("(probe lengths not collected)\n");
        }
        p.node(&self.plan, &mut String::new(), &mut out);
        let unplaced: Vec<_> = m
            .operators
            .iter()
            .filter(|(_, e)| e.node.is_none_or(|n| !p.placed(n)))
            .collect();
        if !unplaced.is_empty() {
            out.push_str("Result:\n");
            for (id, e) in unplaced {
                let _ = writeln!(out, "   {}", p.operator(*id, e));
            }
        }
        out.push_str("Pipelines:\n");
        for (id, pm) in &m.pipelines {
            let mut ops: Vec<_> = m
                .operators
                .iter()
                .filter(|(_, e)| e.pipelines.contains(id))
                .collect();
            // Source first, then operators, then the sink.
            ops.sort_by_key(|(oid, e)| (e.role != OperatorRole::Source, e.role, **oid));
            let names: Vec<String> = ops
                .iter()
                .map(|(oid, e)| format!("#{} {}", oid.0, e.name))
                .collect();
            let _ = writeln!(
                out,
                "   p{id}: wall {}, {} driver{}, {} thread{}: {}",
                p.time(pm.wall_time()),
                p.count(pm.drivers),
                p.pl(pm.drivers),
                p.count(pm.threads),
                p.pl(pm.threads),
                names.join(" -> ")
            );
        }
        out
    }
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn by_node(p: &QueryProfile) -> BTreeMap<u32, Vec<(OperatorId, &OperatorEntry)>> {
    let mut map: BTreeMap<u32, Vec<_>> = BTreeMap::new();
    for (id, e) in &p.metrics.operators {
        if let Some(n) = e.node {
            map.entry(n).or_default().push((*id, e));
        }
    }
    let mut ids = Vec::new();
    p.plan.walk(&mut |n| ids.extend(n.id));
    // Operators of nodes the tree does not show print under "Result".
    map.retain(|n, _| ids.contains(n));
    map
}

/// Extra metrics that depend on the query's hash seed (how keys spread
/// over partitions and collide), masked with the other volatile values.
const VOLATILE: [&str; 4] = [
    "hash_buckets",
    "grace_hash_buckets",
    "probe_key_compares",
    "max_probe_length",
];

struct Printer<'a> {
    opts: TextOptions,
    by_node: BTreeMap<u32, Vec<(OperatorId, &'a OperatorEntry)>>,
}

impl Printer<'_> {
    fn placed(&self, node: u32) -> bool {
        self.by_node.contains_key(&node)
    }

    fn node(&self, n: &ProfileNode, prefix: &mut String, out: &mut String) {
        out.push_str(&n.name);
        if !n.detail.is_empty() {
            out.push(' ');
            out.push_str(&n.detail);
        }
        out.push('\n');
        let guide = if n.children.is_empty() { "   " } else { ":  " };
        for (id, e) in
            n.id.and_then(|i| self.by_node.get(&i))
                .into_iter()
                .flatten()
        {
            let _ = writeln!(out, "{prefix}{guide}· {}", self.operator(*id, e));
        }
        let last = n.children.len().saturating_sub(1);
        for (i, child) in n.children.iter().enumerate() {
            let (head, tail) = if i == last {
                ("+- ", "   ")
            } else {
                (":- ", ":  ")
            };
            out.push_str(prefix);
            out.push_str(head);
            let len = prefix.len();
            prefix.push_str(tail);
            self.node(child, prefix, out);
            prefix.truncate(len);
        }
    }

    fn operator(&self, id: OperatorId, e: &OperatorEntry) -> String {
        let m = &e.metrics;
        let pipes: Vec<String> = e.pipelines.iter().map(|p| format!("p{p}")).collect();
        let mut parts = Vec::new();
        if e.role != OperatorRole::Source {
            parts.push(format!("in {} rows ({})", m.rows_in, batches(m.batches_in)));
        }
        if e.role != OperatorRole::Sink {
            parts.push(format!(
                "out {} rows ({})",
                m.rows_out,
                batches(m.batches_out)
            ));
        }
        parts.push(format!(
            "compute {} (max {}, {} driver{})",
            self.time(m.compute_time),
            self.time(m.max_driver_compute_time),
            self.count(m.drivers),
            self.pl(m.drivers)
        ));
        // Sources always show their I/O wait (a golden text stays stable
        // when a run happens not to wait).
        if e.role == OperatorRole::Source || !m.blocked_on_io.is_zero() {
            parts.push(format!(
                "io {} (max {})",
                self.time(m.blocked_on_io),
                self.time(m.max_driver_blocked_on_io)
            ));
        }
        if !m.blocked_on_memory.is_zero() {
            parts.push(format!("memory wait {}", self.time(m.blocked_on_memory)));
        }
        if !m.blocked_on_sink.is_zero() {
            parts.push(format!("sink wait {}", self.time(m.blocked_on_sink)));
        }
        if m.peak_reserved > 0 {
            parts.push(format!("peak mem {}", self.bytes(m.peak_reserved)));
        }
        if m.spills > 0 {
            parts.push(format!(
                "spilled {}x {}",
                m.spills,
                self.bytes(m.spilled_bytes)
            ));
        }
        if !e.extra.is_empty() {
            parts.push(self.extra(e));
        }
        format!(
            "{:<6} #{} {} {}: {}",
            e.role.as_str(),
            id.0,
            e.name,
            pipes.join(","),
            parts.join("; ")
        )
    }

    fn extra(&self, e: &OperatorEntry) -> String {
        let mut out = Vec::new();
        for (name, v) in &e.extra {
            if self.opts.mask_volatile && VOLATILE.contains(&name.as_ref()) {
                out.push(format!("{name}=<n>"));
                continue;
            }
            let text = match v {
                Metric::Count(n) | Metric::Max(n) if name.ends_with("_bytes") => self.bytes(*n),
                Metric::Count(n) | Metric::Max(n) => n.to_string(),
                Metric::Text(t) => t.clone(),
            };
            let mut item = format!("{name}={text}");
            if name == "hash_buckets"
                && let (Some(entries), Some(buckets)) = (
                    e.extra("hash_entries").and_then(Metric::value),
                    v.value().filter(|b| *b > 0),
                )
            {
                let _ = write!(item, " (load {:.2})", entries as f64 / buckets as f64);
            }
            if name == "probe_key_compares"
                && let (Some(lookups), Some(compares)) = (
                    e.extra("probe_lookups")
                        .and_then(Metric::value)
                        .filter(|l| *l > 0),
                    v.value(),
                )
            {
                let _ = write!(item, " (avg {:.2})", compares as f64 / lookups as f64);
            }
            out.push(item);
        }
        out.join(" ")
    }

    fn time(&self, d: Duration) -> String {
        if self.opts.mask_volatile {
            return "<t>".to_owned();
        }
        let ns = d.as_nanos();
        if ns >= 1_000_000_000 {
            format!("{:.3} s", d.as_secs_f64())
        } else if ns >= 1_000_000 {
            format!("{:.1} ms", d.as_secs_f64() * 1e3)
        } else {
            format!("{:.1} µs", d.as_secs_f64() * 1e6)
        }
    }

    /// The plural ending of a [`count`](Self::count): always `s` when the
    /// count is masked.
    fn pl(&self, n: u64) -> &'static str {
        if self.opts.mask_volatile {
            "s"
        } else {
            plural(n)
        }
    }

    fn count(&self, n: u64) -> String {
        if self.opts.mask_volatile {
            "<n>".to_owned()
        } else {
            n.to_string()
        }
    }

    fn bytes(&self, n: u64) -> String {
        if self.opts.mask_volatile {
            return "<mem>".to_owned();
        }
        human_bytes(n)
    }
}

fn batches(n: u64) -> String {
    format!("{n} batch{}", if n == 1 { "" } else { "es" })
}

/// `n` bytes as `B`, `KiB`, `MiB` or `GiB` with one decimal.
pub(crate) fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut v = n as f64 / 1024.0;
    let mut unit = 0;
    while v >= 1024.0 && unit + 1 < UNITS.len() {
        v /= 1024.0;
        unit += 1;
    }
    format!("{v:.1} {}", UNITS[unit])
}
