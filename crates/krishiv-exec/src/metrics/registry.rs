// SPDX-License-Identifier: Apache-2.0
//! The engine metrics registry (P2-T125 unit 1): the v1 catalogue, the
//! per-engine registry and its snapshot.
//!
//! Metric names are a public surface, so they are fixed here. Names are
//! OTel-style dotted names (`krishiv.query.count`); the catalogue also shows
//! the mechanical Prometheus rendering (dots to underscores, `_seconds` or
//! `_bytes` for those units, `_total` for counters).
//!
//! # Design
//!
//! - The registry is an engine-owned object, held in an `Arc` in the
//!   session's shared state. It is never a `static` or a process-global
//!   recorder: two engines in one process do not collide. The `metrics`
//!   facade is not used.
//! - Counters and histograms are atomics indexed by (definition, bounded
//!   label tuple). A query's metrics fold into them exactly once per query,
//!   at query end, from the merged [`QueryMetrics`] the profile already
//!   uses; nothing is updated per batch or per row.
//! - Engine gauges (memory pool reserved/limit, metadata cache size, spill
//!   disk use, build info, active queries) are closures over the owning
//!   component, evaluated inside [`MetricsRegistry::snapshot`] at scrape
//!   time, never pushed.
//! - Labels are bounded enums only: `status`, `class`, `error_class`
//!   (failed queries only), `cache`, `op`, `store_kind`, plus the
//!   build-stamped `version` and `git_sha` of `krishiv.build.info`.
//!   Definitions whose source does not exist yet keep their catalogue entry
//!   (marked "no source yet") and emit nothing. The prefixes
//!   `krishiv.stream.*`, `krishiv.ivm.*` and `krishiv.cluster.*` are
//!   reserved and have no entries.
//!
//! Process-wide cumulative counters without a per-query breakdown (I/O
//! requests and bytes by operation and store, cache evictions) and the
//! per-operation I/O durations have no source yet: the I/O layer only keeps
//! totals. They are catalogue entries that emit nothing until a source
//! lands. Arbitration and cache counters fold per-query deltas of the
//! process totals read at query start and end; under concurrent queries a
//! delta may be attributed to whichever query ends first.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_common::error::KrishivError;

use super::{OperatorRole, QueryMetrics};
use crate::error::Result;
use crate::error::internal;
use crate::executor::WorkloadClass;
use crate::sync::{Mutex, lock};

/// The kind of a metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricKind {
    /// A sum that only grows (per series); merges by addition.
    Counter,
    /// A current value, read at scrape time; merges keep `worker_id`.
    Gauge,
    /// A distribution over [`BUCKETS`]; merges add buckets, count and sum.
    Histogram,
}

impl MetricKind {
    /// `counter`, `gauge` or `histogram`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Counter => "counter",
            Self::Gauge => "gauge",
            Self::Histogram => "histogram",
        }
    }
}

/// One label of a metric definition: its name and, when the values are
/// known statically, every value it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelDef {
    /// The label name.
    pub name: &'static str,
    /// The values it takes; empty when they are not statically known
    /// (the error-class registry, build-stamped labels, or a source that
    /// does not exist yet).
    pub values: &'static [&'static str],
}

/// Where a metric's values come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricSource {
    /// Folded once per query at query end, from the query's metrics.
    QueryFold,
    /// Read from a closure over the owning component at scrape time.
    Scrape,
    /// No source on main yet: a catalogue entry that emits nothing.
    NoSource,
}

impl MetricSource {
    /// How the catalogue shows it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::QueryFold => "query fold",
            Self::Scrape => "scrape",
            Self::NoSource => "no source yet",
        }
    }
}

/// One metric of the v1 catalogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricDef {
    /// The OTel-style dotted name.
    pub name: &'static str,
    /// What it is.
    pub kind: MetricKind,
    /// The unit in parentheses (`s`, `By`, `{query}`, `{row}`, `{file}`,
    /// `1`).
    pub unit: &'static str,
    /// The labels, in series-key order.
    pub labels: &'static [LabelDef],
    /// What it measures; starts with "No source yet:" when there is none.
    pub description: &'static str,
    /// The Krishiv version that added it.
    pub since: &'static str,
    /// Where its values come from.
    pub source: MetricSource,
}

impl MetricDef {
    /// The mechanical Prometheus rendering: dots to underscores, `_seconds`
    /// or `_bytes` for those units, `_total` for counters. Histograms expose
    /// this name with the usual `_bucket`/`_count`/`_sum` suffixes.
    #[must_use]
    pub fn prometheus_name(&self) -> String {
        let mut out = self.name.replace('.', "_");
        match self.unit {
            "s" => out.push_str("_seconds"),
            "By" => out.push_str("_bytes"),
            _ => {}
        }
        if self.kind == MetricKind::Counter {
            out.push_str("_total");
        }
        out
    }
}

/// The histogram bucket upper bounds, in the unit of the metric (seconds
/// for durations, bytes for sizes): the OTel database-client duration
/// buckets. One set serves every histogram.
pub const BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0,
];

/// `status`: how the query ended.
pub const STATUS_LABEL: LabelDef = LabelDef {
    name: "status",
    values: &["succeeded", "failed", "cancelled", "denied"],
};

/// `class`: the query's workload class.
pub const CLASS_LABEL: LabelDef = LabelDef {
    name: "class",
    values: &["interactive", "batch", "background", "agent"],
};

/// `error_class`: failed queries only, from the error-class registry
/// (at most [`MAX_ERROR_CLASS_VALUES`] distinct values per metric; further
/// values fold into `"OTHER"`).
pub const ERROR_CLASS_LABEL: LabelDef = LabelDef {
    name: "error_class",
    values: &[],
};

/// `cache`: which cache.
pub const CACHE_LABEL: LabelDef = LabelDef {
    name: "cache",
    values: &["metadata"],
};

/// `op`: the I/O operation (no source yet, so no values).
pub const OP_LABEL: LabelDef = LabelDef {
    name: "op",
    values: &[],
};

/// `store_kind`: the object-store kind (no source yet, so no values).
pub const STORE_KIND_LABEL: LabelDef = LabelDef {
    name: "store_kind",
    values: &[],
};

/// `version`: the build's version (build-stamped).
pub const VERSION_LABEL: LabelDef = LabelDef {
    name: "version",
    values: &[],
};

/// `git_sha`: the build's commit (build-stamped).
pub const GIT_SHA_LABEL: LabelDef = LabelDef {
    name: "git_sha",
    values: &[],
};

/// Every label name a catalogue entry or a gauge source may use. In
/// particular no principal, user, tenant, query id, session id, table,
/// path or query text ever labels a series.
pub const LABEL_ALLOWLIST: &[&str] = &[
    "status",
    "class",
    "error_class",
    "cache",
    "op",
    "store_kind",
    "version",
    "git_sha",
];

/// The most distinct `error_class` label values kept per metric; further
/// values fold into `"OTHER"`, so cardinality stays bounded while counts
/// stay exact.
pub const MAX_ERROR_CLASS_VALUES: usize = 200;

/// Overflow value for `error_class` past [`MAX_ERROR_CLASS_VALUES`].
pub const OTHER_ERROR_CLASS: &str = "OTHER";

/// The v1 catalogue, in a fixed order. Entries with
/// [`MetricSource::NoSource`] emit nothing.
pub const CATALOGUE: &[MetricDef] = &[
    MetricDef {
        name: "krishiv.query.count",
        kind: MetricKind::Counter,
        unit: "{query}",
        labels: &[STATUS_LABEL, CLASS_LABEL, ERROR_CLASS_LABEL],
        description: "Queries that reached execution, by outcome. `error_class` is present on \
                      failed queries only, from the error-class registry.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.query.duration",
        kind: MetricKind::Histogram,
        unit: "s",
        labels: &[STATUS_LABEL, CLASS_LABEL],
        description: "Wall time from the start of execution to its end, in seconds.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.query.active",
        kind: MetricKind::Gauge,
        unit: "{query}",
        labels: &[CLASS_LABEL],
        description: "Queries currently executing, by workload class.",
        since: "0.1.0",
        source: MetricSource::Scrape,
    },
    MetricDef {
        name: "krishiv.query.cpu.time",
        kind: MetricKind::Counter,
        unit: "s",
        labels: &[CLASS_LABEL],
        description: "Operator compute time summed over the query's drivers, in seconds.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.query.scan.rows",
        kind: MetricKind::Counter,
        unit: "{row}",
        labels: &[],
        description: "Rows file and memory scans produced, summed over the query's drivers.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.query.scan.bytes",
        kind: MetricKind::Counter,
        unit: "By",
        labels: &[],
        description: "No source yet: scans report no byte counter in the query metrics, so this \
                      entry emits nothing until one lands.",
        since: "0.1.0",
        source: MetricSource::NoSource,
    },
    MetricDef {
        name: "krishiv.query.output.rows",
        kind: MetricKind::Counter,
        unit: "{row}",
        labels: &[],
        description: "Rows the query returned: rows into the result sink, summed over drivers.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.query.spill.bytes",
        kind: MetricKind::Counter,
        unit: "By",
        labels: &[],
        description: "Bytes freed by serviced spills plus bytes written to spill files, summed \
                      over the query's operators.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.query.spill.files",
        kind: MetricKind::Counter,
        unit: "{file}",
        labels: &[],
        description: "Spill files the query's operators wrote.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.query.memory.peak",
        kind: MetricKind::Histogram,
        unit: "By",
        labels: &[CLASS_LABEL],
        description: "The most bytes the query held reserved at once (an upper bound, summed over \
                      workers), in bytes.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.memory.reserved",
        kind: MetricKind::Gauge,
        unit: "By",
        labels: &[],
        description: "Bytes the session's memory pool subtree holds now, in bytes.",
        since: "0.1.0",
        source: MetricSource::Scrape,
    },
    MetricDef {
        name: "krishiv.memory.limit",
        kind: MetricKind::Gauge,
        unit: "By",
        labels: &[],
        description: "The most the session's pool subtree can hold, in bytes; absent when \
                      unbounded.",
        since: "0.1.0",
        source: MetricSource::Scrape,
    },
    MetricDef {
        name: "krishiv.memory.arbitration.spills",
        kind: MetricKind::Counter,
        unit: "1",
        labels: &[],
        description: "Spill requests the memory arbitrator sent, as per-query deltas of the pool \
                      tree totals. Under concurrent queries a delta may be attributed to whichever \
                      query ends first.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.memory.arbitration.aborts",
        kind: MetricKind::Counter,
        unit: "1",
        labels: &[],
        description: "Queries the memory arbitrator aborted, as per-query deltas of the pool \
                      tree totals (see `krishiv.memory.arbitration.spills` on attribution).",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.spill.disk.used",
        kind: MetricKind::Gauge,
        unit: "By",
        labels: &[],
        description: "Bytes spill files occupy on disk now (written, not yet dropped), in bytes; \
                      zero before the first query opens the spill directory.",
        since: "0.1.0",
        source: MetricSource::Scrape,
    },
    MetricDef {
        name: "krishiv.cache.hits",
        kind: MetricKind::Counter,
        unit: "1",
        labels: &[CACHE_LABEL],
        description: "File-metadata cache hits, as per-query deltas of the scheduler totals \
                      (see `krishiv.memory.arbitration.spills` on attribution).",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.cache.misses",
        kind: MetricKind::Counter,
        unit: "1",
        labels: &[CACHE_LABEL],
        description: "File-metadata cache misses, as per-query deltas of the scheduler totals \
                      (see `krishiv.memory.arbitration.spills` on attribution).",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.cache.evictions",
        kind: MetricKind::Counter,
        unit: "1",
        labels: &[CACHE_LABEL],
        description: "No source yet: the metadata cache keeps no eviction counter, so this entry \
                      emits nothing until one lands.",
        since: "0.1.0",
        source: MetricSource::NoSource,
    },
    MetricDef {
        name: "krishiv.cache.size",
        kind: MetricKind::Gauge,
        unit: "By",
        labels: &[CACHE_LABEL],
        description: "Bytes the file-metadata cache holds now, in bytes.",
        since: "0.1.0",
        source: MetricSource::Scrape,
    },
    MetricDef {
        name: "krishiv.io.requests",
        kind: MetricKind::Counter,
        unit: "1",
        labels: &[OP_LABEL, STORE_KIND_LABEL],
        description: "No source yet: the I/O layer keeps only process-wide totals with no \
                      per-operation or per-store breakdown, so this entry emits nothing until one \
                      lands.",
        since: "0.1.0",
        source: MetricSource::NoSource,
    },
    MetricDef {
        name: "krishiv.io.bytes.read",
        kind: MetricKind::Counter,
        unit: "By",
        labels: &[OP_LABEL, STORE_KIND_LABEL],
        description: "No source yet: the I/O layer keeps only process-wide totals with no \
                      per-operation or per-store breakdown, so this entry emits nothing until one \
                      lands.",
        since: "0.1.0",
        source: MetricSource::NoSource,
    },
    MetricDef {
        name: "krishiv.io.request.duration",
        kind: MetricKind::Histogram,
        unit: "s",
        labels: &[OP_LABEL, STORE_KIND_LABEL],
        description: "No source yet: the I/O layer keeps no per-request durations, so this entry \
                      emits nothing until they land.",
        since: "0.1.0",
        source: MetricSource::NoSource,
    },
    MetricDef {
        name: "krishiv.exec.driver.stalls",
        kind: MetricKind::Counter,
        unit: "1",
        labels: &[],
        description: "Driver polls longer than the stall threshold, summed over the query's \
                      drivers.",
        since: "0.1.0",
        source: MetricSource::QueryFold,
    },
    MetricDef {
        name: "krishiv.build.info",
        kind: MetricKind::Gauge,
        unit: "1",
        labels: &[VERSION_LABEL, GIT_SHA_LABEL],
        description: "The build: always 1, labelled with its version and commit.",
        since: "0.1.0",
        source: MetricSource::Scrape,
    },
];

/// The query's outcome, as [`Status::from_error`] classifies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The query ran to the end.
    Succeeded,
    /// The query failed.
    Failed,
    /// The query was cancelled.
    Cancelled,
    /// The query was denied (`KRISHIV_INSUFFICIENT_PERMISSIONS`).
    Denied,
}

impl Status {
    /// `succeeded`, `failed`, `cancelled` or `denied`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Denied => "denied",
        }
    }

    /// Classifies the query's error: none is success, `OPERATION_CANCELED`
    /// is cancellation, `KRISHIV_INSUFFICIENT_PERMISSIONS` is denial, any
    /// other class is failure with its name as the `error_class` label.
    #[must_use]
    pub fn from_error(error: Option<&KrishivError>) -> (Self, Option<String>) {
        let Some(error) = error else {
            return (Self::Succeeded, None);
        };
        let class = error.error_class().name().to_owned();
        match class.as_str() {
            "OPERATION_CANCELED" => (Self::Cancelled, None),
            "KRISHIV_INSUFFICIENT_PERMISSIONS" => (Self::Denied, Some(class)),
            _ => (Self::Failed, Some(class)),
        }
    }
}

/// What [`MetricsRegistry::fold`] records for one ended query.
#[derive(Clone, Copy, Debug)]
pub struct QueryOutcome<'a> {
    /// The query's workload class.
    pub class: WorkloadClass,
    /// How the query ended.
    pub status: Status,
    /// The error class of a failed or denied query.
    pub error_class: Option<&'a str>,
    /// Wall time from the start of execution to the fold, in seconds.
    pub duration_secs: f64,
    /// The query's merged metrics.
    pub metrics: &'a QueryMetrics,
    /// New arbitration spill requests since the query started.
    pub arbitration_spills: u64,
    /// New arbitration aborts since the query started.
    pub arbitration_aborts: u64,
    /// New metadata-cache hits since the query started.
    pub cache_hits: u64,
    /// New metadata-cache misses since the query started.
    pub cache_misses: u64,
}

/// One series: its definition and its label values in definition order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SeriesKey {
    def: usize,
    labels: Vec<(String, String)>,
}

/// A folded counter cell.
#[derive(Debug, Default)]
struct CounterCell {
    /// The value, as `f64` bits.
    value: AtomicU64,
}

/// A folded histogram cell.
#[derive(Debug)]
struct HistogramCell {
    /// Cumulative counts: `buckets[i]` holds observations `<= BUCKETS[i]`.
    buckets: Vec<AtomicU64>,
    /// Observations.
    count: AtomicU64,
    /// Their sum, as `f64` bits.
    sum: AtomicU64,
}

impl HistogramCell {
    fn new() -> Self {
        Self {
            buckets: BUCKETS.iter().map(|_| AtomicU64::new(0)).collect(),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        }
    }
}

/// A scrape-time gauge: its definition, its labels and how to read it.
/// `None` means absent (e.g. an unbounded memory limit): no sample.
struct GaugeSource {
    def: usize,
    labels: Vec<(String, String)>,
    read: Box<dyn Fn() -> Option<u64> + Send + Sync>,
}

impl fmt::Debug for GaugeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaugeSource")
            .field("def", &self.def)
            .field("labels", &self.labels)
            .finish_non_exhaustive()
    }
}

/// Adds `delta` to the `f64` in `cell`.
fn add_f64(cell: &AtomicU64, delta: f64) {
    // Once per query per series: contention is not a concern.
    let mut bits = cell.load(Ordering::Relaxed);
    loop {
        let next = f64::from_bits(bits) + delta;
        match cell.compare_exchange_weak(bits, next.to_bits(), Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return,
            Err(seen) => bits = seen,
        }
    }
}

/// The per-engine metrics registry: the folded counters and histograms of
/// ended queries plus the scrape-time gauges. Held in an `Arc` in the
/// session's shared state; two engines never share one.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    counters: Mutex<HashMap<SeriesKey, CounterCell>>,
    histograms: Mutex<HashMap<SeriesKey, HistogramCell>>,
    gauges: Mutex<Vec<GaugeSource>>,
    /// Queries currently executing, by workload class
    /// (`Interactive`, `Batch`, `Background`, `Agent`).
    active: [AtomicU64; 4],
    /// Folds so far (queries that reached execution and ended).
    folds: AtomicU64,
    /// Distinct `error_class` values seen per metric definition.
    error_classes: Mutex<HashMap<usize, HashSet<String>>>,
}

impl MetricsRegistry {
    /// An empty registry: no series, no gauges, nothing active.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The catalogue.
    #[must_use]
    pub fn catalogue() -> &'static [MetricDef] {
        CATALOGUE
    }

    /// The index of `class` in [`Self::active`].
    fn class_index(class: WorkloadClass) -> usize {
        match class {
            WorkloadClass::Interactive => 0,
            WorkloadClass::Batch => 1,
            WorkloadClass::Background => 2,
            WorkloadClass::Agent => 3,
        }
    }

    /// Notes a query of `class` starting execution (balanced by
    /// [`Self::fold`], which runs exactly once per started query).
    pub fn query_begin(&self, class: WorkloadClass) {
        self.active[Self::class_index(class)].fetch_add(1, Ordering::Relaxed);
    }

    /// Queries of `class` currently executing.
    #[must_use]
    pub fn active_queries(&self, class: WorkloadClass) -> u64 {
        self.active[Self::class_index(class)].load(Ordering::Relaxed)
    }

    /// Folds so far.
    #[must_use]
    pub fn fold_count(&self) -> u64 {
        self.folds.load(Ordering::Relaxed)
    }

    /// Finds the catalogue index of `name`.
    fn def_index(name: &str) -> Option<usize> {
        CATALOGUE.iter().position(|d| d.name == name)
    }

    /// Registers a scrape-time gauge for `metric` with `labels`: `read`
    /// runs inside [`Self::snapshot`], never pushed. Label names and values
    /// must match the definition (an `error_class` value is capped like a
    /// folded one); `None` reads emit no sample.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an unknown metric, a non-gauge, or labels that
    /// do not match the definition.
    pub fn add_gauge(
        &self,
        metric: &str,
        labels: Vec<(String, String)>,
        read: impl Fn() -> Option<u64> + Send + Sync + 'static,
    ) -> Result<()> {
        let Some(def) = Self::def_index(metric) else {
            return Err(internal(format!("unknown metric {metric}")));
        };
        if CATALOGUE[def].kind != MetricKind::Gauge {
            return Err(internal(format!("metric {metric} is not a gauge")));
        }
        check_labels(def, &labels)?;
        lock(&self.gauges).push(GaugeSource {
            def,
            labels,
            read: Box::new(read),
        });
        Ok(())
    }

    /// Folds one ended query: every counter and histogram with a source,
    /// and one less active query of its class. Runs exactly once per
    /// started query, including submit errors (with empty metrics) and
    /// cancelled queries (with partial metrics).
    pub fn fold(&self, outcome: &QueryOutcome<'_>) {
        self.folds.fetch_add(1, Ordering::Relaxed);
        self.active[Self::class_index(outcome.class)].fetch_sub(1, Ordering::Relaxed);
        let m = outcome.metrics;
        let status = outcome.status.as_str().to_owned();
        let class = outcome.class.as_str().to_owned();
        let count_labels = match outcome.status {
            Status::Failed | Status::Denied => {
                let error_class = self.capped_error_class(
                    Self::def_index("krishiv.query.count"),
                    outcome.error_class,
                );
                vec![
                    ("status".to_owned(), status.clone()),
                    ("class".to_owned(), class.clone()),
                    ("error_class".to_owned(), error_class),
                ]
            }
            Status::Succeeded | Status::Cancelled => {
                vec![
                    ("status".to_owned(), status.clone()),
                    ("class".to_owned(), class.clone()),
                ]
            }
        };
        let status_class = vec![
            ("status".to_owned(), status),
            ("class".to_owned(), class.clone()),
        ];
        let class_only = vec![("class".to_owned(), class)];
        let cache = vec![("cache".to_owned(), "metadata".to_owned())];
        self.add_counter("krishiv.query.count", &count_labels, 1.0);
        self.observe(
            "krishiv.query.duration",
            &status_class,
            outcome.duration_secs.max(0.0),
        );
        self.add_counter("krishiv.query.cpu.time", &class_only, cpu_secs(m));
        self.add_counter("krishiv.query.scan.rows", &[], scan_rows(m) as f64);
        self.add_counter("krishiv.query.output.rows", &[], output_rows(m) as f64);
        self.add_counter("krishiv.query.spill.bytes", &[], spill_bytes(m) as f64);
        self.add_counter("krishiv.query.spill.files", &[], spill_files(m) as f64);
        self.observe(
            "krishiv.query.memory.peak",
            &class_only,
            m.memory.peak_reserved as f64,
        );
        self.add_counter(
            "krishiv.memory.arbitration.spills",
            &[],
            outcome.arbitration_spills as f64,
        );
        self.add_counter(
            "krishiv.memory.arbitration.aborts",
            &[],
            outcome.arbitration_aborts as f64,
        );
        self.add_counter("krishiv.cache.hits", &cache, outcome.cache_hits as f64);
        self.add_counter("krishiv.cache.misses", &cache, outcome.cache_misses as f64);
        self.add_counter("krishiv.exec.driver.stalls", &[], m.drivers.stalls as f64);
    }

    /// The `error_class` label value for `def`: the query's class, or
    /// [`OTHER_ERROR_CLASS`] past [`MAX_ERROR_CLASS_VALUES`] distinct
    /// values (missing classes cannot happen: only failed and denied
    /// queries label it, and both carry one).
    fn capped_error_class(&self, def: Option<usize>, class: Option<&str>) -> String {
        let class = class.unwrap_or(OTHER_ERROR_CLASS);
        let Some(def) = def else {
            return OTHER_ERROR_CLASS.to_owned();
        };
        let mut seen = lock(&self.error_classes);
        let set = seen.entry(def).or_default();
        if set.contains(class) {
            return class.to_owned();
        }
        if set.len() >= MAX_ERROR_CLASS_VALUES {
            return OTHER_ERROR_CLASS.to_owned();
        }
        set.insert(class.to_owned());
        class.to_owned()
    }

    /// Adds `delta` to the counter `metric` (a definition with a source).
    fn add_counter(&self, metric: &str, labels: &[(String, String)], delta: f64) {
        let Some(def) = Self::def_index(metric) else {
            return;
        };
        if CATALOGUE[def].source == MetricSource::NoSource {
            return;
        }
        let key = SeriesKey {
            def,
            labels: labels.to_vec(),
        };
        let mut counters = lock(&self.counters);
        let cell = counters.entry(key).or_default();
        add_f64(&cell.value, delta);
    }

    /// Observes `value` in the histogram `metric` (a definition with a
    /// source).
    fn observe(&self, metric: &str, labels: &[(String, String)], value: f64) {
        let Some(def) = Self::def_index(metric) else {
            return;
        };
        if CATALOGUE[def].source == MetricSource::NoSource {
            return;
        }
        let key = SeriesKey {
            def,
            labels: labels.to_vec(),
        };
        let mut histograms = lock(&self.histograms);
        let cell = histograms.entry(key).or_insert_with(HistogramCell::new);
        cell.count.fetch_add(1, Ordering::Relaxed);
        add_f64(&cell.sum, value);
        // Cumulative buckets: every bound at or above the value counts it.
        for (bound, bucket) in BUCKETS.iter().zip(cell.buckets.iter()) {
            if value <= *bound {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Reads every series and gauge: plain data. Counters and histograms
    /// hold what folds recorded; gauges evaluate their closures now.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let mut out = MetricsSnapshot::default();
        for (key, cell) in lock(&self.counters).iter() {
            let def = &CATALOGUE[key.def];
            out.counters.push(CounterSample {
                name: def.name,
                unit: def.unit,
                labels: key.labels.clone(),
                value: f64::from_bits(cell.value.load(Ordering::Relaxed)),
            });
        }
        for (key, cell) in lock(&self.histograms).iter() {
            let def = &CATALOGUE[key.def];
            out.histograms.push(HistogramSample {
                name: def.name,
                unit: def.unit,
                labels: key.labels.clone(),
                buckets: BUCKETS
                    .iter()
                    .zip(cell.buckets.iter())
                    .map(|(bound, bucket)| HistogramBucket {
                        le: *bound,
                        count: bucket.load(Ordering::Relaxed),
                    })
                    .collect(),
                count: cell.count.load(Ordering::Relaxed),
                sum: f64::from_bits(cell.sum.load(Ordering::Relaxed)),
            });
        }
        for gauge in lock(&self.gauges).iter() {
            let def = &CATALOGUE[gauge.def];
            if let Some(value) = (gauge.read)() {
                out.gauges.push(GaugeSample {
                    name: def.name,
                    unit: def.unit,
                    labels: gauge.labels.clone(),
                    worker_id: worker_id(),
                    value: value as f64,
                });
            }
        }
        out.sort();
        out
    }

    /// Renders the v1 catalogue reference (`docs/reference/metrics-v1.md`):
    /// every definition with its dotted name, Prometheus rendering, kind,
    /// unit, labels, source and description, plus the bucket table and the
    /// reserved prefixes.
    #[must_use]
    pub fn catalogue_markdown() -> String {
        let mut out = String::from(
            "# Krishiv engine metrics (v1)\n\n\
             <!-- @generated by `cargo xtask docs-metrics`; do not edit. -->\n\n\
             Engine metrics, scraped per engine (`Session::metrics`, `krishiv.metrics()`): \
             OTel-style dotted names with the mechanical Prometheus rendering next to them \
             (dots to underscores, `_seconds` or `_bytes` for those units, `_total` for \
             counters; histograms expose the base name with the usual `_bucket`, `_count` \
             and `_sum` suffixes). Units are in parentheses. Query metrics fold once per \
             query at query end and are skipped entirely when `krishiv.metrics.enabled` \
             is false; gauges evaluate at scrape time. Labels are bounded enums only: no \
             principal, user, tenant, query id, session id, table, path or query text ever \
             labels a series.\n\n\
             The prefixes `krishiv.stream.*`, `krishiv.ivm.*` and `krishiv.cluster.*` are \
             reserved for streaming, incremental view maintenance and multi-node clusters; \
             they have no entries yet.\n\n\
             ## Histogram buckets\n\n\
             One set serves every histogram (seconds for durations, bytes for sizes):\n\n\
             `0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300` \
             (an overflow bucket past 300 holds the rest).\n\n\
             ## Catalogue\n\n\
             | Dotted name | Prometheus | Kind | Unit | Labels | Source | Description |\n\
             |---|---|---|---|---|---|---|\n",
        );
        for def in CATALOGUE {
            let labels = def
                .labels
                .iter()
                .map(|l| {
                    if l.values.is_empty() {
                        l.name.to_owned()
                    } else {
                        format!("{}({})", l.name, l.values.join(", "))
                    }
                })
                .collect::<Vec<_>>()
                .join("; ");
            out.push_str(&format!(
                "| `{}` | `{}` | {} | `{}` | {} | {} | {} |\n",
                def.name,
                def.prometheus_name(),
                def.kind.as_str(),
                def.unit,
                if labels.is_empty() { "—" } else { &labels },
                def.source.as_str(),
                def.description.replace('|', "\\|"),
            ));
        }
        out
    }
}

/// Checks `labels` against the definition's label names and values: names
/// must match in order; values must be listed unless the definition leaves
/// them open (the error-class registry, build stamps, sources without one).
fn check_labels(def: usize, labels: &[(String, String)]) -> Result<()> {
    let wanted = CATALOGUE[def].labels;
    if labels.len() != wanted.len() {
        return Err(internal(format!(
            "metric {} takes {} labels, not {}",
            CATALOGUE[def].name,
            wanted.len(),
            labels.len()
        )));
    }
    for (have, want) in labels.iter().zip(wanted.iter()) {
        if have.0 != want.name {
            return Err(internal(format!(
                "metric {} takes label {}, not {}",
                CATALOGUE[def].name, want.name, have.0
            )));
        }
        if !want.values.is_empty() && !want.values.contains(&have.1.as_str()) {
            return Err(internal(format!(
                "metric {} label {} takes no value {}",
                CATALOGUE[def].name, want.name, have.1
            )));
        }
    }
    Ok(())
}

/// The worker id gauges carry: this process. Merged snapshots keep one
/// sample per worker.
fn worker_id() -> String {
    std::process::id().to_string()
}

/// Operator compute time summed over the query's drivers, in seconds.
#[must_use]
pub fn cpu_secs(m: &QueryMetrics) -> f64 {
    m.operators
        .values()
        .map(|e| e.metrics.compute_time.as_secs_f64())
        .sum()
}

/// Rows file and memory scans produced, summed over the query's drivers.
/// Breaker output sides are sources too, but under their operator's name,
/// never a scan's.
#[must_use]
pub fn scan_rows(m: &QueryMetrics) -> u64 {
    m.operators
        .values()
        .filter(|e| {
            e.role == OperatorRole::Source
                && (e.name.as_ref() == "FileScan" || e.name.as_ref() == "MemoryScan")
        })
        .map(|e| e.metrics.rows_out)
        .fold(0, u64::saturating_add)
}

/// Rows the query returned: rows into the result sink, summed over drivers.
#[must_use]
pub fn output_rows(m: &QueryMetrics) -> u64 {
    m.operators
        .values()
        .filter(|e| e.role == OperatorRole::Sink && e.name.as_ref() == "Result")
        .map(|e| e.metrics.rows_in)
        .fold(0, u64::saturating_add)
}

/// Bytes freed by serviced spills plus bytes written to spill files, summed
/// over the query's operators.
#[must_use]
pub fn spill_bytes(m: &QueryMetrics) -> u64 {
    m.operators
        .values()
        .map(|e| {
            e.metrics
                .spilled_bytes
                .saturating_add(e.metrics.spill_write_bytes)
        })
        .fold(0, u64::saturating_add)
}

/// Spill files the query's operators wrote.
#[must_use]
pub fn spill_files(m: &QueryMetrics) -> u64 {
    m.operators
        .values()
        .map(|e| e.metrics.spill_files)
        .fold(0, u64::saturating_add)
}

/// One folded counter series.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterSample {
    /// The dotted name.
    pub name: &'static str,
    /// The unit.
    pub unit: &'static str,
    /// Label values in definition order.
    pub labels: Vec<(String, String)>,
    /// The value.
    pub value: f64,
}

/// One scraped gauge sample.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeSample {
    /// The dotted name.
    pub name: &'static str,
    /// The unit.
    pub unit: &'static str,
    /// Label values in definition order.
    pub labels: Vec<(String, String)>,
    /// The worker that reported it.
    pub worker_id: String,
    /// The value.
    pub value: f64,
}

/// One cumulative histogram bucket: observations `<= le`.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramBucket {
    /// The upper bound.
    pub le: f64,
    /// Cumulative observations.
    pub count: u64,
}

/// One folded histogram series.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramSample {
    /// The dotted name.
    pub name: &'static str,
    /// The unit.
    pub unit: &'static str,
    /// Label values in definition order.
    pub labels: Vec<(String, String)>,
    /// Cumulative buckets in [`BUCKETS`] order.
    pub buckets: Vec<HistogramBucket>,
    /// Observations.
    pub count: u64,
    /// Their sum, in the unit.
    pub sum: f64,
}

/// The registry's plain-data snapshot: every series with its labels.
/// Counters and histogram buckets add on [`Self::merge`]; gauges keep
/// `worker_id`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetricsSnapshot {
    /// Folded counters.
    pub counters: Vec<CounterSample>,
    /// Scraped gauges.
    pub gauges: Vec<GaugeSample>,
    /// Folded histograms.
    pub histograms: Vec<HistogramSample>,
}

impl MetricsSnapshot {
    /// Sorts every series by (name, labels, worker id): snapshots compare
    /// and merge deterministically.
    fn sort(&mut self) {
        self.counters
            .sort_by(|a, b| (&a.name, &a.labels).cmp(&(&b.name, &b.labels)));
        self.gauges.sort_by(|a, b| {
            (&a.name, &a.labels, &a.worker_id).cmp(&(&b.name, &b.labels, &b.worker_id))
        });
        self.histograms
            .sort_by(|a, b| (&a.name, &a.labels).cmp(&(&b.name, &b.labels)));
    }

    /// Adds `other`: counters and histogram buckets, counts and sums of the
    /// same (name, labels) add; gauges keep one sample per
    /// (name, labels, `worker_id`), with `other` winning conflicts.
    pub fn merge(&mut self, other: &Self) {
        for c in &other.counters {
            match self
                .counters
                .iter_mut()
                .find(|s| s.name == c.name && s.labels == c.labels)
            {
                Some(s) => s.value += c.value,
                None => self.counters.push(c.clone()),
            }
        }
        for h in &other.histograms {
            match self
                .histograms
                .iter_mut()
                .find(|s| s.name == h.name && s.labels == h.labels)
            {
                Some(s) => {
                    for (to, from) in s.buckets.iter_mut().zip(h.buckets.iter()) {
                        // Bounds come from the one static table; a mismatch
                        // would be an engine bug, never a merge order.
                        debug_assert_eq!(to.le, from.le);
                        to.count = to.count.saturating_add(from.count);
                    }
                    s.count = s.count.saturating_add(h.count);
                    s.sum += h.sum;
                }
                None => self.histograms.push(h.clone()),
            }
        }
        for g in &other.gauges {
            match self
                .gauges
                .iter_mut()
                .find(|s| s.name == g.name && s.labels == g.labels && s.worker_id == g.worker_id)
            {
                Some(s) => s.value = g.value,
                None => self.gauges.push(g.clone()),
            }
        }
        self.sort();
    }

    /// The value of the counter `name` with `labels`, if any.
    #[must_use]
    pub fn counter(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        self.counters
            .iter()
            .find(|s| {
                s.name == name
                    && s.labels.len() == labels.len()
                    && s.labels
                        .iter()
                        .zip(labels.iter())
                        .all(|((n, v), (wn, wv))| n == wn && v == wv)
            })
            .map(|s| s.value)
    }

    /// The series as plain dicts (`krishiv.metrics()`): each has `name`,
    /// `kind`, `unit` and `labels`; counters and gauges add `value`
    /// (gauges also `worker_id`); histograms add cumulative `buckets` as
    /// `(le, count)` pairs with `count` and `sum`.
    #[must_use]
    pub fn dicts(&self) -> Vec<MetricDict> {
        let mut out = Vec::new();
        for c in &self.counters {
            out.push(MetricDict {
                name: c.name,
                kind: "counter",
                unit: c.unit,
                labels: c.labels.clone(),
                value: Some(c.value),
                buckets: None,
                count: None,
                sum: None,
                worker_id: None,
            });
        }
        for g in &self.gauges {
            out.push(MetricDict {
                name: g.name,
                kind: "gauge",
                unit: g.unit,
                labels: g.labels.clone(),
                value: Some(g.value),
                buckets: None,
                count: None,
                sum: None,
                worker_id: Some(g.worker_id.clone()),
            });
        }
        for h in &self.histograms {
            out.push(MetricDict {
                name: h.name,
                kind: "histogram",
                unit: h.unit,
                labels: h.labels.clone(),
                value: None,
                buckets: Some(h.buckets.iter().map(|b| (b.le, b.count)).collect()),
                count: Some(h.count),
                sum: Some(h.sum),
                worker_id: None,
            });
        }
        out
    }
}

/// One snapshot series as plain dict data (see
/// [`MetricsSnapshot::dicts`]).
#[derive(Clone, Debug, PartialEq)]
pub struct MetricDict {
    /// The dotted name.
    pub name: &'static str,
    /// `counter`, `gauge` or `histogram`.
    pub kind: &'static str,
    /// The unit.
    pub unit: &'static str,
    /// Label values in definition order.
    pub labels: Vec<(String, String)>,
    /// Counters and gauges.
    pub value: Option<f64>,
    /// Histograms: cumulative `(le, count)` pairs in [`BUCKETS`] order.
    pub buckets: Option<Vec<(f64, u64)>>,
    /// Histograms: observations.
    pub count: Option<u64>,
    /// Histograms: their sum, in the unit.
    pub sum: Option<f64>,
    /// Gauges: the reporting worker.
    pub worker_id: Option<String>,
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::context::OperatorId;
    use crate::metrics::{OperatorEntry, OperatorMetrics, QueryMemory};

    fn metrics_with(rows_out: u64, rows_in: u64, compute_ms: u64) -> QueryMetrics {
        let mut m = QueryMetrics::default();
        m.operators.insert(
            OperatorId(1),
            OperatorEntry {
                name: std::borrow::Cow::Borrowed("MemoryScan"),
                node: Some(0),
                role: OperatorRole::Source,
                pipelines: vec![0],
                metrics: OperatorMetrics {
                    rows_out,
                    batches_out: 1,
                    ..OperatorMetrics::default()
                },
                extra: Vec::new(),
            },
        );
        m.operators.insert(
            OperatorId(2),
            OperatorEntry {
                name: std::borrow::Cow::Borrowed("Result"),
                node: None,
                role: OperatorRole::Sink,
                pipelines: vec![0],
                metrics: OperatorMetrics {
                    rows_in,
                    batches_in: 1,
                    compute_time: std::time::Duration::from_millis(compute_ms),
                    ..OperatorMetrics::default()
                },
                extra: Vec::new(),
            },
        );
        m.memory = QueryMemory {
            limit: Some(1 << 30),
            peak_reserved: 1 << 20,
        };
        m
    }

    fn outcome<'a>(
        status: Status,
        error_class: Option<&'a str>,
        metrics: &'a QueryMetrics,
    ) -> QueryOutcome<'a> {
        QueryOutcome {
            class: WorkloadClass::Interactive,
            status,
            error_class,
            duration_secs: 0.25,
            metrics,
            arbitration_spills: 2,
            arbitration_aborts: 0,
            cache_hits: 3,
            cache_misses: 1,
        }
    }

    #[test]
    fn fold_records_every_sourced_series() {
        let registry = MetricsRegistry::new();
        let m = metrics_with(600, 4, 120);
        registry.query_begin(WorkloadClass::Interactive);
        assert_eq!(registry.active_queries(WorkloadClass::Interactive), 1);
        registry.fold(&outcome(Status::Succeeded, None, &m));
        assert_eq!(registry.active_queries(WorkloadClass::Interactive), 0);
        assert_eq!(registry.fold_count(), 1);
        let snap = registry.snapshot();
        assert_eq!(
            snap.counter(
                "krishiv.query.count",
                &[("status", "succeeded"), ("class", "interactive")]
            ),
            Some(1.0)
        );
        assert_eq!(
            snap.counter("krishiv.query.cpu.time", &[("class", "interactive")]),
            Some(0.12)
        );
        assert_eq!(snap.counter("krishiv.query.scan.rows", &[]), Some(600.0));
        assert_eq!(snap.counter("krishiv.query.output.rows", &[]), Some(4.0));
        assert_eq!(
            snap.counter("krishiv.memory.arbitration.spills", &[]),
            Some(2.0)
        );
        assert_eq!(
            snap.counter("krishiv.cache.hits", &[("cache", "metadata")]),
            Some(3.0)
        );
        assert_eq!(
            snap.counter("krishiv.cache.misses", &[("cache", "metadata")]),
            Some(1.0)
        );
        let duration = snap
            .histograms
            .iter()
            .find(|h| h.name == "krishiv.query.duration")
            .unwrap();
        assert_eq!(duration.count, 1);
        assert_eq!(duration.sum, 0.25);
        // 0.25 lands in the 0.25 bucket and every bucket above it.
        let at = |le: f64| {
            duration
                .buckets
                .iter()
                .find(|b| b.le == le)
                .map(|b| b.count)
                .unwrap()
        };
        assert_eq!(at(0.1), 0);
        assert_eq!(at(0.25), 1);
        assert_eq!(at(300.0), 1);
        let peak = snap
            .histograms
            .iter()
            .find(|h| h.name == "krishiv.query.memory.peak")
            .unwrap();
        assert_eq!(peak.count, 1);
        assert_eq!(peak.sum, f64::from(1 << 20));
        // A 1 MiB peak exceeds the largest bound: only the overflow holds it.
        assert!(peak.buckets.iter().all(|b| b.count == 0));
        // Entries without a source emit nothing.
        for name in [
            "krishiv.query.scan.bytes",
            "krishiv.cache.evictions",
            "krishiv.io.requests",
            "krishiv.io.bytes.read",
        ] {
            assert!(
                snap.counters.iter().all(|c| c.name != name),
                "{name} emitted without a source"
            );
        }
        assert!(
            snap.histograms
                .iter()
                .all(|h| h.name != "krishiv.io.request.duration")
        );
    }

    #[test]
    fn failures_carry_error_class_and_denials_map() {
        use krishiv_common::error::{KrishivError, OperationCanceled};
        let registry = MetricsRegistry::new();
        let m = QueryMetrics::default();
        let (status, class) = Status::from_error(None);
        assert_eq!((status, class), (Status::Succeeded, None));
        let cancelled: KrishivError = OperationCanceled {}.into();
        assert_eq!(
            Status::from_error(Some(&cancelled)),
            (Status::Cancelled, None)
        );
        registry.fold(&QueryOutcome {
            class: WorkloadClass::Interactive,
            status: Status::Failed,
            error_class: Some("DIVIDE_BY_ZERO"),
            duration_secs: 0.0,
            metrics: &m,
            arbitration_spills: 0,
            arbitration_aborts: 0,
            cache_hits: 0,
            cache_misses: 0,
        });
        // Denied queries keep their class too.
        registry.fold(&QueryOutcome {
            class: WorkloadClass::Interactive,
            status: Status::Denied,
            error_class: Some("KRISHIV_INSUFFICIENT_PERMISSIONS"),
            duration_secs: 0.0,
            metrics: &m,
            arbitration_spills: 0,
            arbitration_aborts: 0,
            cache_hits: 0,
            cache_misses: 0,
        });
        let snap = registry.snapshot();
        assert_eq!(
            snap.counter(
                "krishiv.query.count",
                &[
                    ("status", "failed"),
                    ("class", "interactive"),
                    ("error_class", "DIVIDE_BY_ZERO"),
                ]
            ),
            Some(1.0)
        );
        assert_eq!(
            snap.counter(
                "krishiv.query.count",
                &[
                    ("status", "denied"),
                    ("class", "interactive"),
                    ("error_class", "KRISHIV_INSUFFICIENT_PERMISSIONS"),
                ]
            ),
            Some(1.0)
        );
        // Succeeded and cancelled series carry no error_class.
        registry.fold(&outcome(Status::Cancelled, None, &m));
        assert_eq!(
            snap.counter(
                "krishiv.query.count",
                &[("status", "succeeded"), ("class", "interactive")]
            ),
            None
        );
        let snap = registry.snapshot();
        assert_eq!(
            snap.counter(
                "krishiv.query.count",
                &[("status", "cancelled"), ("class", "interactive")]
            ),
            Some(1.0)
        );
    }

    #[test]
    fn error_classes_are_capped_but_counts_are_exact() {
        let registry = MetricsRegistry::new();
        let m = QueryMetrics::default();
        let classes: Vec<String> = (0..250)
            .map(|n| format!("SYNTHETIC_CLASS_{n:03}"))
            .collect();
        for class in &classes {
            registry.fold(&QueryOutcome {
                class: WorkloadClass::Interactive,
                status: Status::Failed,
                error_class: Some(class.as_str()),
                duration_secs: 0.0,
                metrics: &m,
                arbitration_spills: 0,
                arbitration_aborts: 0,
                cache_hits: 0,
                cache_misses: 0,
            });
        }
        let snap = registry.snapshot();
        let failed: Vec<_> = snap
            .counters
            .iter()
            .filter(|c| {
                c.name == "krishiv.query.count"
                    && c.labels.iter().any(|(n, v)| n == "status" && v == "failed")
            })
            .collect();
        let named: Vec<_> = failed
            .iter()
            .filter(|c| {
                c.labels
                    .iter()
                    .any(|(n, v)| n == "error_class" && v != OTHER_ERROR_CLASS)
            })
            .collect();
        assert_eq!(named.len(), MAX_ERROR_CLASS_VALUES);
        let total: f64 = failed.iter().map(|c| c.value).sum();
        assert_eq!(total, 250.0);
        let overflow = snap.counter(
            "krishiv.query.count",
            &[
                ("status", "failed"),
                ("class", "interactive"),
                ("error_class", OTHER_ERROR_CLASS),
            ],
        );
        assert_eq!(overflow, Some(50.0));
    }

    #[test]
    fn snapshot_merge_adds_counters_and_histograms() {
        let m = metrics_with(10, 2, 10);
        let first = MetricsRegistry::new();
        first.fold(&outcome(Status::Succeeded, None, &m));
        let second = MetricsRegistry::new();
        second.fold(&outcome(Status::Succeeded, None, &m));
        second.fold(&outcome(Status::Failed, Some("DIVIDE_BY_ZERO"), &m));
        let mut merged = first.snapshot();
        let before = merged.clone();
        merged.merge(&second.snapshot());
        // Associative: folding both orders gives the same snapshot.
        let mut other = second.snapshot();
        other.merge(&before);
        assert_eq!(merged, other);
        assert_eq!(
            merged.counter(
                "krishiv.query.count",
                &[("status", "succeeded"), ("class", "interactive")]
            ),
            Some(2.0)
        );
        assert_eq!(merged.counter("krishiv.query.scan.rows", &[]), Some(30.0));
        let duration = merged
            .histograms
            .iter()
            .find(|h| {
                h.name == "krishiv.query.duration"
                    && h.labels
                        == vec![
                            ("status".to_owned(), "succeeded".to_owned()),
                            ("class".to_owned(), "interactive".to_owned()),
                        ]
            })
            .unwrap();
        assert_eq!(duration.count, 2);
        assert_eq!(duration.sum, 0.5);
    }

    #[test]
    fn snapshot_merge_keeps_gauges_per_worker() {
        let registry = MetricsRegistry::new();
        registry
            .add_gauge("krishiv.memory.reserved", vec![], || Some(100))
            .unwrap();
        let mut a = registry.snapshot();
        let mut b = registry.snapshot();
        // Another worker's sample survives the merge, keyed by worker.
        b.gauges.push(GaugeSample {
            name: "krishiv.memory.reserved",
            unit: "By",
            labels: Vec::new(),
            worker_id: "other".to_owned(),
            value: 50.0,
        });
        a.merge(&b);
        assert_eq!(a.gauges.len(), 2);
        // Same worker: the merged snapshot wins.
        let mut c = registry.snapshot();
        c.gauges[0].value = 7.0;
        a.merge(&c);
        assert_eq!(a.gauges.len(), 2);
        let here = a
            .gauges
            .iter()
            .find(|g| g.worker_id == worker_id())
            .unwrap();
        assert_eq!(here.value, 7.0);
    }

    #[test]
    fn gauges_evaluate_at_scrape_time_and_skip_absent() {
        let registry = MetricsRegistry::new();
        let cell = std::sync::Arc::new(AtomicU64::new(3));
        let read_cell = std::sync::Arc::clone(&cell);
        registry
            .add_gauge("krishiv.spill.disk.used", vec![], move || {
                Some(read_cell.load(Ordering::Relaxed))
            })
            .unwrap();
        registry
            .add_gauge("krishiv.memory.limit", vec![], || None)
            .unwrap();
        assert_eq!(registry.snapshot().gauges.len(), 1);
        cell.store(9, Ordering::Relaxed);
        // Never pushed: the scrape reads the current value.
        assert_eq!(registry.snapshot().gauges[0].value, 9.0);
        assert_eq!(registry.snapshot().gauges[0].worker_id, worker_id());
    }

    #[test]
    fn add_gauge_rejects_unknown_metrics_and_bad_labels() {
        let registry = MetricsRegistry::new();
        assert!(
            registry
                .add_gauge("krishiv.nope", vec![], || Some(1))
                .is_err()
        );
        assert!(
            registry
                .add_gauge("krishiv.query.count", vec![], || Some(1))
                .is_err()
        );
        assert!(
            registry
                .add_gauge(
                    "krishiv.cache.size",
                    vec![("cache".to_owned(), "nope".to_owned())],
                    || Some(1)
                )
                .is_err()
        );
        assert!(
            registry
                .add_gauge(
                    "krishiv.cache.size",
                    vec![("cache".to_owned(), "metadata".to_owned())],
                    || Some(1)
                )
                .is_ok()
        );
    }

    #[test]
    fn catalogue_is_stable_and_prometheus_names_collide_nowhere() {
        assert!(!CATALOGUE.is_empty());
        let mut names = HashSet::new();
        for def in CATALOGUE {
            assert!(def.name.starts_with("krishiv."), "{}", def.name);
            assert!(!def.description.is_empty(), "{}", def.name);
            assert!(!def.since.is_empty(), "{}", def.name);
            assert!(names.insert(def.name), "duplicate {}", def.name);
            for label in def.labels {
                assert!(
                    LABEL_ALLOWLIST.contains(&label.name),
                    "{} labels {}",
                    def.name,
                    label.name
                );
            }
            if def.source == MetricSource::NoSource {
                assert!(
                    def.description.starts_with("No source yet:"),
                    "{}",
                    def.name
                );
            }
        }
        // The mechanical Prometheus rendering collides nowhere.
        let mut prom = HashSet::new();
        for def in CATALOGUE {
            let rendered = def.prometheus_name();
            assert!(!rendered.contains('.'), "{rendered}");
            assert!(prom.insert(rendered.clone()), "collision on {rendered}");
        }
        assert_eq!(CATALOGUE[0].prometheus_name(), "krishiv_query_count_total");
        assert_eq!(
            CATALOGUE
                .iter()
                .find(|d| d.name == "krishiv.query.duration")
                .unwrap()
                .prometheus_name(),
            "krishiv_query_duration_seconds"
        );
    }

    #[test]
    fn report_lists_every_definition_for_the_snapshot_test() {
        // The insta snapshot guards names, units, kinds, labels and both
        // renderings; `cargo insta` updates it with the catalogue.
        let mut report = String::new();
        for def in CATALOGUE {
            let labels = def
                .labels
                .iter()
                .map(|l| {
                    if l.values.is_empty() {
                        l.name.to_owned()
                    } else {
                        format!("{}={}", l.name, l.values.join("|"))
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            report.push_str(&format!(
                "{} {} {} [{}] -> {}\n",
                def.name,
                def.kind.as_str(),
                def.unit,
                labels,
                def.prometheus_name()
            ));
        }
        insta::assert_snapshot!(report);
    }

    #[test]
    fn dicts_shape_matches_the_python_contract() {
        let registry = MetricsRegistry::new();
        let m = metrics_with(8, 1, 5);
        registry.fold(&outcome(Status::Succeeded, None, &m));
        registry
            .add_gauge("krishiv.memory.reserved", vec![], || Some(2))
            .unwrap();
        let snap = registry.snapshot();
        for d in snap.dicts() {
            assert!(!d.name.is_empty());
            assert!(!d.unit.is_empty());
            match d.kind {
                "counter" | "gauge" => {
                    assert!(d.value.is_some());
                    assert!(d.buckets.is_none());
                }
                "histogram" => {
                    assert!(d.value.is_none());
                    let buckets = d.buckets.as_ref().unwrap();
                    assert_eq!(buckets.len(), BUCKETS.len());
                    assert!(d.count.is_some() && d.sum.is_some());
                }
                kind => panic!("unknown kind {kind}"),
            }
        }
        let histogram = snap
            .dicts()
            .into_iter()
            .find(|d| d.kind == "histogram")
            .unwrap();
        // Cumulative: no bucket exceeds the observation count.
        assert!(
            histogram
                .buckets
                .as_ref()
                .unwrap()
                .iter()
                .all(|(_, c)| *c <= histogram.count.unwrap())
        );
    }

    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "a test reads the committed reference page; no worker runs here"
    )]
    fn the_reference_matches_the_generated_catalogue() {
        // `docs/reference/metrics-v1.md` is generated by
        // `cargo xtask docs-metrics`; regenerate it with the catalogue.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/reference/metrics-v1.md");
        let committed = std::fs::read_to_string(&path).unwrap();
        assert_eq!(committed, MetricsRegistry::catalogue_markdown());
    }

    #[test]
    fn totals_derive_from_the_query_metrics() {
        let m = metrics_with(600, 4, 120);
        assert_eq!(scan_rows(&m), 600);
        assert_eq!(output_rows(&m), 4);
        assert_eq!(cpu_secs(&m), 0.12);
        assert_eq!(spill_bytes(&QueryMetrics::default()), 0);
        assert_eq!(spill_files(&QueryMetrics::default()), 0);
        // Breaker output sides are sources but not scans.
        let mut mixed = m;
        mixed.operators.insert(
            OperatorId(3),
            OperatorEntry {
                name: std::borrow::Cow::Borrowed("HashAggregate"),
                node: Some(1),
                role: OperatorRole::Source,
                pipelines: vec![0],
                metrics: OperatorMetrics {
                    rows_out: 4,
                    batches_out: 1,
                    ..OperatorMetrics::default()
                },
                extra: Vec::new(),
            },
        );
        assert_eq!(scan_rows(&mixed), 600);
    }
}
