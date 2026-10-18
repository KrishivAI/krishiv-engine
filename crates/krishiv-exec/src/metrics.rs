// SPDX-License-Identifier: Apache-2.0
//! Metrics for `EXPLAIN ANALYZE`: each driver counts in plain fields and
//! merges into the query once, when it ends (no per-batch atomics).

use std::collections::BTreeMap;
use std::time::Duration;

use crate::context::OperatorId;

/// Counters of one operator, summed over its drivers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorMetrics {
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
    /// Time a source's `get_data` was pending (I/O).
    pub blocked_on_io: Duration,
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
    pub(crate) fn merge(&mut self, o: &Self) {
        self.rows_in += o.rows_in;
        self.batches_in += o.batches_in;
        self.rows_out += o.rows_out;
        self.batches_out += o.batches_out;
        self.compute_time += o.compute_time;
        self.blocked_on_io += o.blocked_on_io;
        self.morsels += o.morsels;
        self.peak_reserved += o.peak_reserved;
        self.spills += o.spills;
        self.spilled_bytes += o.spilled_bytes;
        self.blocked_on_memory += o.blocked_on_memory;
        self.blocked_on_sink += o.blocked_on_sink;
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
    pub(crate) fn merge(&mut self, o: &Self) {
        self.drivers += o.drivers;
        self.batches += o.batches;
        self.yields += o.yields;
        self.max_slice = self.max_slice.max(o.max_slice);
        self.stalls += o.stalls;
    }
}

/// One operator's metrics with its name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorEntry {
    /// The operator's name.
    pub name: &'static str,
    /// Its counters.
    pub metrics: OperatorMetrics,
}

/// A query's metrics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    /// Per operator.
    pub operators: BTreeMap<OperatorId, OperatorEntry>,
    /// Scheduling.
    pub drivers: DriverStats,
}
