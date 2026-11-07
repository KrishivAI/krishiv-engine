// SPDX-License-Identifier: Apache-2.0
//! The `system` catalog (P2-T126 unit 1): read-only tables about the engine
//! itself, addressed `system.<table>` (empty namespace, ADR 0035).
//!
//! In-process there is one worker, so a system table has one split: each
//! query builds the rows it reads from the process's state when its scan is
//! planned, and the table is a snapshot, never an update. `system` is a
//! reserved catalog name (`KRISHIV_RESERVED_NAME`): nothing can be created in
//! it, and it is no database.
//!
//! This unit holds `system.capabilities` and `system.memory`; the query,
//! operator, cache and metric tables arrive with their recorders (units 2
//! and 3).

use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType as ArrowType, Field, Schema, TimeUnit};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::planner::MemoryTable;
use krishiv_memory::MemoryPool;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, BoxFuture, Catalog, NameForm, Principal, Privilege, Securable,
    TableColumn, TableIdent, TableKind, TableMeta,
};

use crate::history::QueryHistory;
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

/// The catalog's name.
pub(crate) const SYSTEM: &str = "system";

/// The schema version of every table in this unit (`krishiv.schema.version`).
const SCHEMA_VERSION: &str = "1";

/// The worker this process is, in the `worker_id` column.
const LOCAL_WORKER: &str = "local";

/// Every table of the catalog.
const TABLES: [&str; 6] = [
    "capabilities",
    "caches",
    "memory",
    "metrics",
    "operators",
    "queries",
];

/// Whether `name` is the reserved catalog name, in any letter case.
pub(crate) fn is_system(name: &str) -> bool {
    name.eq_ignore_ascii_case(SYSTEM)
}

/// Whether a scan of `name` reads a system table: `system.<table>`.
pub(crate) fn is_system_table(name: &[String]) -> bool {
    matches!(name, [catalog, table] if is_system(catalog) && table_known(table))
}

fn table_known(table: &str) -> bool {
    TABLES.iter().any(|t| t.eq_ignore_ascii_case(table))
}

/// The cargo features this build was made with and whether each is on, set
/// once by the binary (the library cannot see its dependents' features).
static BUILD_FEATURES: OnceLock<Vec<(String, bool)>> = OnceLock::new();

/// Records the cargo features of the running binary for
/// `system.capabilities`: `(name, enabled)` for every feature that can be
/// switched. The first call wins; later calls change nothing.
pub fn register_build_features(features: &[(&str, bool)]) {
    let owned = features
        .iter()
        .map(|(name, on)| ((*name).to_owned(), *on))
        .collect();
    let _ = BUILD_FEATURES.set(owned);
}

/// The rows of `system.capabilities`: `(kind, name, value, detail)`.
fn capability_rows() -> Vec<[String; 4]> {
    let row = |kind: &str, name: &str, value: &str, detail: &str| {
        [
            kind.to_owned(),
            name.to_owned(),
            value.to_owned(),
            detail.to_owned(),
        ]
    };
    let state = |on: bool| if on { "enabled" } else { "disabled" };
    let mut rows = vec![
        row("build", "engine", "krishiv", env!("CARGO_PKG_VERSION")),
        row("build", "spark_versions", "3.5,4.0,4.1,4.2", ""),
        row("surface", "system_tables", SCHEMA_VERSION, ""),
        row("surface", "agent_result", "1", ""),
        row("feature", "cloud", state(cfg!(feature = "cloud")), ""),
    ];
    if let Some(features) = BUILD_FEATURES.get() {
        let mut sorted: Vec<&(String, bool)> = features.iter().collect();
        sorted.sort();
        rows.extend(
            sorted
                .into_iter()
                .filter(|(name, _)| name != "cloud")
                .map(|(name, on)| row("feature", name, state(*on), "")),
        );
    }
    rows
}

/// What a column holds.
#[derive(Clone, Copy)]
enum Kind {
    Text,
    Long,
    /// `INT`.
    Int,
    Flag,
    /// `TIMESTAMP` (microseconds since the epoch).
    Time,
    /// `DOUBLE`.
    Double,
    /// `MAP<STRING, STRING>`.
    Labels,
    /// `ARRAY<INT>`.
    IntList,
    /// `ARRAY<STRUCT<le: DOUBLE, count: BIGINT>>`.
    Buckets,
}

/// A table's columns: `(name, kind, nullable)`.
fn layout(table: &str) -> Option<Vec<(&'static str, Kind, bool)>> {
    use Kind::{Buckets, Double, Flag, Int, IntList, Labels, Long, Text, Time};
    Some(match table.to_ascii_lowercase().as_str() {
        "capabilities" => vec![
            ("worker_id", Text, false),
            ("kind", Text, false),
            ("name", Text, false),
            ("value", Text, false),
            ("detail", Text, true),
        ],
        "memory" => vec![
            ("worker_id", Text, false),
            ("path", Text, false),
            ("parent", Text, true),
            ("kind", Text, false),
            ("query_id", Text, true),
            ("reserved_bytes", Long, false),
            ("peak_bytes", Long, false),
            ("limit_bytes", Long, true),
            ("spillable", Flag, false),
        ],
        "caches" => vec![
            ("worker_id", Text, false),
            ("cache", Text, false),
            ("entries", Long, false),
            ("bytes", Long, false),
            ("capacity_bytes", Long, true),
            ("hits", Long, false),
            ("misses", Long, false),
            ("evictions", Long, false),
            ("invalidations", Long, false),
            ("loads_in_flight", Long, false),
        ],
        "metrics" => vec![
            ("worker_id", Text, false),
            ("name", Text, false),
            ("kind", Text, false),
            ("unit", Text, false),
            ("labels", Labels, false),
            ("value", Double, true),
            ("count", Long, true),
            ("sum", Double, true),
            ("buckets", Buckets, true),
            ("sampled_at", Time, false),
        ],
        "operators" => vec![
            ("query_id", Text, false),
            ("operator_id", Int, false),
            ("node_id", Int, true),
            ("name", Text, false),
            ("role", Text, false),
            ("pipelines", IntList, false),
            ("drivers", Long, false),
            ("rows_in", Long, false),
            ("rows_out", Long, false),
            ("batches_in", Long, false),
            ("batches_out", Long, false),
            ("compute_ns", Long, false),
            ("max_driver_compute_ns", Long, false),
            ("blocked_io_ns", Long, false),
            ("blocked_memory_ns", Long, false),
            ("blocked_sink_ns", Long, false),
            ("peak_reserved_bytes", Long, false),
            ("spills", Long, false),
            ("spilled_bytes", Long, false),
            ("spill_files", Long, false),
            ("spill_write_bytes", Long, false),
            ("extra", Labels, false),
            ("workers", Int, false),
        ],
        "queries" => vec![
            ("query_id", Text, false),
            ("session_id", Text, false),
            ("principal", Text, false),
            ("tags", Labels, false),
            ("trace_id", Text, true),
            ("statement_class", Text, true),
            ("state", Text, false),
            ("query_text", Text, true),
            ("query_text_truncated", Flag, false),
            ("submitted_at", Time, false),
            ("ended_at", Time, true),
            ("wall_ns", Long, true),
            ("peak_memory_bytes", Long, true),
            ("error_class", Text, true),
            ("workload_class", Text, true),
            ("cpu_ns", Long, true),
            ("rows_produced", Long, true),
            ("bytes_read", Long, true),
            ("bytes_written", Long, true),
            ("spill_bytes", Long, true),
            ("spill_files", Long, true),
            ("pipelines", Long, true),
            ("workers", Long, true),
            ("cancel_reason", Text, true),
        ],
        _ => return None,
    })
}

/// The columns of `table`, `None` for an unknown table.
fn columns(table: &str) -> Option<Vec<TableColumn>> {
    Some(
        layout(table)?
            .into_iter()
            .map(|(name, kind, nullable)| TableColumn {
                name: name.to_owned(),
                data_type: match kind {
                    Kind::Text => DataType::String,
                    Kind::Long => DataType::Long,
                    Kind::Int => DataType::Integer,
                    Kind::Flag => DataType::Boolean,
                    Kind::Time => DataType::Timestamp,
                    Kind::Double => DataType::Double,
                    Kind::Labels => labels_type(),
                    Kind::IntList => DataType::Array(
                        krishiv_types::ArrayType::new(DataType::Integer, false).into(),
                    ),
                    Kind::Buckets => buckets_type(),
                },
                nullable,
                char_varchar: None,
            })
            .collect(),
    )
}

/// `MAP<STRING, STRING>`.
fn labels_type() -> DataType {
    DataType::Map(krishiv_types::MapType::new(DataType::String, DataType::String, true).into())
}

/// `ARRAY<STRUCT<le: DOUBLE, count: BIGINT>>`.
fn buckets_type() -> DataType {
    use krishiv_types::{ArrayType, StructField, StructType};
    DataType::Array(
        ArrayType::new(
            DataType::Struct(StructType::new([
                StructField::new("le", DataType::Double, false),
                StructField::new("count", DataType::Long, false),
            ])),
            false,
        )
        .into(),
    )
}

/// An `ARRAY<INT>` column.
fn int_list_column(rows: &[&[i32]]) -> ArrayRef {
    use arrow_array::ListArray;
    use arrow_buffer::OffsetBuffer;
    let ArrowType::List(element) =
        DataType::Array(krishiv_types::ArrayType::new(DataType::Integer, false).into()).to_arrow()
    else {
        unreachable!("an array type maps to a list")
    };
    let values = Int32Array::from_iter_values(rows.iter().flat_map(|r| r.iter().copied()));
    let offsets = OffsetBuffer::from_lengths(rows.iter().map(|r| r.len()));
    Arc::new(ListArray::new(element, offsets, Arc::new(values), None))
}

/// A `MAP<STRING, STRING>` column of `rows` label sets.
fn labels_column(rows: &[&[(String, String)]]) -> ArrayRef {
    use arrow_array::{MapArray, StringViewArray, StructArray};
    use arrow_buffer::OffsetBuffer;
    let ArrowType::Map(entries, _) = labels_type().to_arrow() else {
        unreachable!("a map type maps to a map")
    };
    let ArrowType::Struct(fields) = entries.data_type().clone() else {
        unreachable!("map entries are a struct")
    };
    let keys = StringViewArray::from_iter_values(
        rows.iter().flat_map(|r| r.iter().map(|(k, _)| k.as_str())),
    );
    let vals = StringViewArray::from_iter_values(
        rows.iter().flat_map(|r| r.iter().map(|(_, v)| v.as_str())),
    );
    let offsets = OffsetBuffer::from_lengths(rows.iter().map(|r| r.len()));
    let struct_array = StructArray::new(fields, vec![Arc::new(keys), Arc::new(vals)], None);
    Arc::new(MapArray::new(entries, offsets, struct_array, None, false))
}

/// An `ARRAY<STRUCT<le, count>>` column; a row with `None` is NULL.
fn buckets_column(rows: &[Option<&[(f64, i64)]>]) -> ArrayRef {
    use arrow_array::{Float64Array, ListArray, StructArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    let ArrowType::List(element) = buckets_type().to_arrow() else {
        unreachable!("an array type maps to a list")
    };
    let ArrowType::Struct(fields) = element.data_type().clone() else {
        unreachable!("the element is a struct")
    };
    let les =
        Float64Array::from_iter_values(rows.iter().flatten().flat_map(|r| r.iter().map(|b| b.0)));
    let counts =
        Int64Array::from_iter_values(rows.iter().flatten().flat_map(|r| r.iter().map(|b| b.1)));
    let offsets = OffsetBuffer::from_lengths(rows.iter().map(|r| r.map_or(0, <[_]>::len)));
    let values = StructArray::new(fields, vec![Arc::new(les), Arc::new(counts)], None);
    let valid = NullBuffer::from_iter(rows.iter().map(Option::is_some));
    Arc::new(ListArray::new(
        element,
        offsets,
        Arc::new(values),
        Some(valid),
    ))
}

/// Rows of text cells, column by column.
fn text<'a>(cells: impl Iterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(StringArray::from_iter(cells))
}

fn long(cells: impl Iterator<Item = Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from_iter(cells))
}

/// A byte count as `BIGINT`, saturating.
fn bytes(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// The trace id of a W3C `traceparent` (`version-trace_id-parent_id-flags`,
/// lower-case hex), `None` if the header is malformed: version `ff`, an
/// all-zero id, or the wrong lengths.
pub(crate) fn parse_traceparent(header: &str) -> Option<String> {
    let hex = |s: &str, n: usize| {
        s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    let parts: Vec<&str> = header.trim().split('-').collect();
    let [version, trace, parent, flags, ..] = parts.as_slice() else {
        return None;
    };
    let zero = |s: &str| s.bytes().all(|b| b == b'0');
    // Version 00 has exactly four fields; later versions may add more.
    let shape = hex(version, 2) && *version != "ff" && (*version != "00" || parts.len() == 4);
    (shape && hex(trace, 32) && !zero(trace) && hex(parent, 16) && !zero(parent) && hex(flags, 2))
        .then(|| (*trace).to_owned())
}

/// The name of the memory pool of the query whose `system.queries` row is `query_id`.
pub(crate) fn pool_name(query_id: &str) -> String {
    format!("query-{query_id}")
}

/// The query a memory path belongs to: `process/query-<id>/...` (or
/// `process/session:<s>/query-<id>/...` under a session limit) names the pool of query `<id>`,
/// or a consumer below it.
fn query_of_path(path: &str) -> Option<String> {
    let mut parts = path.split('/').skip(1);
    let first = parts.next()?;
    let pool = if first.starts_with("session:") {
        parts.next()?
    } else {
        first
    };
    pool.strip_prefix("query-").map(str::to_owned)
}

/// The columns' values for `table` now.
fn values(table: &str, sources: &Sources) -> Vec<ArrayRef> {
    if table.eq_ignore_ascii_case("memory") {
        let entries = sources.memory.snapshot().entries;
        let worker = vec![Some(LOCAL_WORKER); entries.len()];
        let queries: Vec<Option<String>> = entries.iter().map(|e| query_of_path(&e.path)).collect();
        return vec![
            text(worker.into_iter()),
            text(entries.iter().map(|e| Some(e.path.as_str()))),
            text(entries.iter().map(|e| e.parent.as_deref())),
            text(
                entries
                    .iter()
                    .map(|e| Some(if e.is_consumer { "consumer" } else { "pool" })),
            ),
            text(queries.iter().map(Option::as_deref)),
            long(entries.iter().map(|e| Some(bytes(e.reserved)))),
            long(entries.iter().map(|e| Some(bytes(e.peak)))),
            long(entries.iter().map(|e| e.limit.map(bytes))),
            Arc::new(BooleanArray::from_iter(
                entries.iter().map(|e| Some(e.spillable)),
            )),
        ];
    }
    if table.eq_ignore_ascii_case("metrics") {
        let snap = sources.metrics.snapshot();
        let now = now_ms().saturating_mul(1000);
        let mut rows: Vec<MetricRow> = Vec::new();
        for c in &snap.counters {
            rows.push(MetricRow::new(
                LOCAL_WORKER,
                c.name,
                "counter",
                c.unit,
                &c.labels,
                Some(c.value),
            ));
        }
        for g in &snap.gauges {
            rows.push(MetricRow::new(
                &g.worker_id,
                g.name,
                "gauge",
                g.unit,
                &g.labels,
                Some(g.value),
            ));
        }
        for h in &snap.histograms {
            let mut r = MetricRow::new(LOCAL_WORKER, h.name, "histogram", h.unit, &h.labels, None);
            r.count = Some(i64::try_from(h.count).unwrap_or(i64::MAX));
            r.sum = Some(h.sum);
            r.buckets = Some(
                h.buckets
                    .iter()
                    .map(|b| (b.le, i64::try_from(b.count).unwrap_or(i64::MAX)))
                    .collect(),
            );
            rows.push(r);
        }
        let labels: Vec<&[(String, String)]> = rows.iter().map(|r| r.labels.as_slice()).collect();
        let buckets: Vec<Option<&[(f64, i64)]>> =
            rows.iter().map(|r| r.buckets.as_deref()).collect();
        return vec![
            text(rows.iter().map(|r| Some(r.worker.as_str()))),
            text(rows.iter().map(|r| Some(r.name))),
            text(rows.iter().map(|r| Some(r.kind))),
            text(rows.iter().map(|r| Some(r.unit))),
            labels_column(&labels),
            Arc::new(Float64Array::from_iter(rows.iter().map(|r| r.value))),
            long(rows.iter().map(|r| r.count)),
            Arc::new(Float64Array::from_iter(rows.iter().map(|r| r.sum))),
            buckets_column(&buckets),
            Arc::new(TimestampMicrosecondArray::from_value(now, rows.len()).with_timezone("UTC")),
        ];
    }
    if table.eq_ignore_ascii_case("caches") {
        let n = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        let meta = sources.io.cache().stats();
        let io = sources.io.totals();
        let relation = sources.relations.stats();
        let rows = [
            CacheRow {
                cache: "file_metadata",
                entries: n(meta.entries),
                bytes: n(meta.bytes),
                capacity: Some(n(meta.limit)),
                hits: n(krishiv_io::IoCounters::get(&io.cache_hits)),
                misses: n(krishiv_io::IoCounters::get(&io.cache_misses)),
                evictions: n(meta.evictions),
                invalidations: n(meta.invalidations),
                loads_in_flight: n(meta.loads_in_flight),
            },
            CacheRow {
                cache: "relation",
                entries: n(relation.entries),
                bytes: n(relation.bytes),
                capacity: None,
                hits: n(relation.hits),
                misses: n(relation.misses),
                evictions: 0,
                invalidations: n(relation.invalidations),
                loads_in_flight: 0,
            },
        ];
        let l = |f: fn(&CacheRow) -> i64| long(rows.iter().map(move |r| Some(f(r))));
        return vec![
            text(rows.iter().map(|_| Some(LOCAL_WORKER))),
            text(rows.iter().map(|r| Some(r.cache))),
            l(|r| r.entries),
            l(|r| r.bytes),
            long(rows.iter().map(|r| r.capacity)),
            l(|r| r.hits),
            l(|r| r.misses),
            l(|r| r.evictions),
            l(|r| r.invalidations),
            l(|r| r.loads_in_flight),
        ];
    }
    if table.eq_ignore_ascii_case("operators") {
        let rows = sources.queries.operators(sources.monitor.sees_all());
        let l = |f: fn(&OperatorRow) -> i64| long(rows.iter().map(move |r| Some(f(r))));
        return vec![
            text(rows.iter().map(|r| Some(r.query_id.as_str()))),
            Arc::new(Int32Array::from_iter(
                rows.iter().map(|r| Some(r.operator_id)),
            )),
            Arc::new(Int32Array::from_iter(rows.iter().map(|r| r.node_id))),
            text(rows.iter().map(|r| Some(r.name.as_str()))),
            text(rows.iter().map(|r| Some(r.role))),
            int_list_column(
                &rows
                    .iter()
                    .map(|r| r.pipelines.as_slice())
                    .collect::<Vec<_>>(),
            ),
            l(|r| r.drivers),
            l(|r| r.rows_in),
            l(|r| r.rows_out),
            l(|r| r.batches_in),
            l(|r| r.batches_out),
            l(|r| r.compute_ns),
            l(|r| r.max_driver_compute_ns),
            l(|r| r.blocked_io_ns),
            l(|r| r.blocked_memory_ns),
            l(|r| r.blocked_sink_ns),
            l(|r| r.peak_reserved_bytes),
            l(|r| r.spills),
            l(|r| r.spilled_bytes),
            l(|r| r.spill_files),
            l(|r| r.spill_write_bytes),
            labels_column(&rows.iter().map(|r| r.extra.as_slice()).collect::<Vec<_>>()),
            Arc::new(Int32Array::from_iter(rows.iter().map(|r| Some(r.workers)))),
        ];
    }
    if table.eq_ignore_ascii_case("queries") {
        let rows = sources.queries.snapshot(sources.monitor.sees_all());
        let micros = |ms: Option<i64>| ms.map(|m| m.saturating_mul(1000));
        return vec![
            text(rows.iter().map(|r| Some(r.id.as_str()))),
            text(rows.iter().map(|r| Some(r.session_id.as_str()))),
            text(rows.iter().map(|r| Some(r.principal.as_str()))),
            labels_column(&rows.iter().map(|r| r.tags.as_slice()).collect::<Vec<_>>()),
            text(rows.iter().map(|r| r.trace_id.as_deref())),
            text(rows.iter().map(|r| r.statement_class)),
            text(rows.iter().map(|r| Some(r.state))),
            text(rows.iter().map(|r| r.text.as_deref())),
            Arc::new(BooleanArray::from_iter(
                rows.iter().map(|r| Some(r.truncated)),
            )),
            Arc::new(
                TimestampMicrosecondArray::from_iter(
                    rows.iter().map(|r| micros(Some(r.submitted_ms))),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(
                TimestampMicrosecondArray::from_iter(rows.iter().map(|r| micros(r.ended_ms)))
                    .with_timezone("UTC"),
            ),
            long(rows.iter().map(|r| r.wall_ns)),
            long(rows.iter().map(|r| r.peak_memory_bytes)),
            text(rows.iter().map(|r| r.error_class.as_deref())),
            text(rows.iter().map(|r| r.workload_class)),
            long(rows.iter().map(|r| r.cpu_ns)),
            long(rows.iter().map(|r| r.rows_produced)),
            long(rows.iter().map(|r| r.bytes_read)),
            long(rows.iter().map(|r| r.bytes_written)),
            long(rows.iter().map(|r| r.spill_bytes)),
            long(rows.iter().map(|r| r.spill_files)),
            long(rows.iter().map(|r| r.pipelines)),
            long(rows.iter().map(|r| r.workers)),
            text(rows.iter().map(|r| r.cancel_reason)),
        ];
    }
    let rows = capability_rows();
    vec![
        text(rows.iter().map(|_| Some(LOCAL_WORKER))),
        text(rows.iter().map(|r| Some(r[0].as_str()))),
        text(rows.iter().map(|r| Some(r[1].as_str()))),
        text(rows.iter().map(|r| Some(r[2].as_str()))),
        text(rows.iter().map(|r| Some(r[3].as_str()))),
    ]
}

/// One series of `system.metrics`.
struct MetricRow {
    worker: String,
    name: &'static str,
    kind: &'static str,
    unit: &'static str,
    labels: Vec<(String, String)>,
    value: Option<f64>,
    count: Option<i64>,
    sum: Option<f64>,
    buckets: Option<Vec<(f64, i64)>>,
}

impl MetricRow {
    fn new(
        worker: &str,
        name: &'static str,
        kind: &'static str,
        unit: &'static str,
        labels: &[(String, String)],
        value: Option<f64>,
    ) -> Self {
        Self {
            worker: worker.to_owned(),
            name,
            kind,
            unit,
            labels: labels.to_vec(),
            value,
            count: None,
            sum: None,
            buckets: None,
        }
    }
}

/// One row of `system.caches`.
struct CacheRow {
    cache: &'static str,
    entries: i64,
    bytes: i64,
    capacity: Option<i64>,
    hits: i64,
    misses: i64,
    evictions: i64,
    invalidations: i64,
    loads_in_flight: i64,
}

/// What the tables read: the process's state.
pub(crate) struct Sources {
    /// The session's memory pool tree.
    pub(crate) memory: Arc<dyn MemoryPool>,
    /// The queries the session ran.
    pub(crate) queries: Arc<QueryLog>,
    /// The I/O scheduler: the file-metadata cache and the request counters.
    pub(crate) io: Arc<krishiv_io::IoScheduler>,
    /// The relation cache.
    pub(crate) relations: Arc<crate::cache::RelationCache>,
    /// The engine's metrics.
    pub(crate) metrics: Arc<krishiv_exec::MetricsRegistry>,
    /// Whether the query's principal sees every principal's queries.
    pub(crate) monitor: Arc<Monitor>,
}

/// Whether one query's principal holds `MONITOR` on the `system` securable,
/// which shows every principal's rows in `system.queries` and
/// `system.operators` (ADR 0036 update 2026-11-05). Asked once per query,
/// when analysis loads one of those tables; until then, and when the policy
/// says anything but a plain allow, the principal sees its own rows only.
pub(crate) struct Monitor {
    principal: Arc<Principal>,
    policy: Arc<dyn AccessPolicy>,
    decided: OnceLock<bool>,
}

impl std::fmt::Debug for Monitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Monitor")
            .field("decided", &self.decided.get())
            .finish_non_exhaustive()
    }
}

impl Monitor {
    pub(crate) fn new(principal: Arc<Principal>, policy: Arc<dyn AccessPolicy>) -> Self {
        Self {
            principal,
            policy,
            decided: OnceLock::new(),
        }
    }

    /// Asks the policy, once.
    async fn decide(&self) {
        if self.decided.get().is_some() {
            return;
        }
        let securable = Securable::System(SYSTEM.to_owned());
        let allowed = matches!(
            self.policy
                .check(&self.principal, &securable, Privilege::Monitor)
                .await,
            Ok(AccessDecision::Allow)
        );
        let _ = self.decided.set(allowed);
    }

    /// Whether every principal's rows show.
    pub(crate) fn sees_all(&self) -> bool {
        self.decided.get().copied().unwrap_or(false)
    }
}

/// Most tags a session carries, and the longest key and value.
pub(crate) const MAX_TAGS: usize = 32;
pub(crate) const MAX_TAG_KEY: usize = 128;
pub(crate) const MAX_TAG_VALUE: usize = 256;

/// Most bytes of a statement `system.queries` keeps
/// (`query_text_truncated` says when it cut).
const TEXT_BYTES: usize = 4096;

/// One query of `system.queries`.
#[derive(Debug, Clone)]
pub(crate) struct QueryRow {
    pub(crate) id: String,
    pub(crate) session_id: String,
    pub(crate) principal: String,
    pub(crate) tags: Vec<(String, String)>,
    pub(crate) trace_id: Option<String>,
    /// `READ`, `WRITE`, `DDL`, `SESSION` or `ADMIN` (T137); `None` when the
    /// statement was refused before it was classified or came from no SQL.
    pub(crate) statement_class: Option<&'static str>,
    pub(crate) state: &'static str,
    pub(crate) text: Option<String>,
    pub(crate) truncated: bool,
    pub(crate) submitted_ms: i64,
    pub(crate) ended_ms: Option<i64>,
    pub(crate) wall_ns: Option<i64>,
    pub(crate) peak_memory_bytes: Option<i64>,
    pub(crate) error_class: Option<String>,
    /// The operators it ran (a finished query only); shared, so a copy of
    /// the history doesn't copy them.
    pub(crate) operators: Arc<[OperatorRow]>,
    pub(crate) workload_class: Option<&'static str>,
    /// Busy time of its drivers.
    pub(crate) cpu_ns: Option<i64>,
    pub(crate) rows_produced: Option<i64>,
    pub(crate) bytes_read: Option<i64>,
    pub(crate) bytes_written: Option<i64>,
    pub(crate) spill_bytes: Option<i64>,
    pub(crate) spill_files: Option<i64>,
    pub(crate) pipelines: Option<i64>,
    pub(crate) workers: Option<i64>,
    /// Why the query was cancelled ([`crate::CancelReason::as_str`]), for a
    /// query cancelled with a reason; `state` stays `cancelled`.
    pub(crate) cancel_reason: Option<&'static str>,
}

/// One operator of a finished query, as `system.operators` lists it.
#[derive(Debug, Clone)]
pub(crate) struct OperatorRow {
    pub(crate) query_id: String,
    pub(crate) operator_id: i32,
    pub(crate) node_id: Option<i32>,
    pub(crate) name: String,
    pub(crate) role: &'static str,
    pub(crate) drivers: i64,
    pub(crate) rows_in: i64,
    pub(crate) rows_out: i64,
    pub(crate) batches_in: i64,
    pub(crate) batches_out: i64,
    pub(crate) compute_ns: i64,
    pub(crate) max_driver_compute_ns: i64,
    pub(crate) blocked_io_ns: i64,
    pub(crate) blocked_memory_ns: i64,
    pub(crate) blocked_sink_ns: i64,
    pub(crate) peak_reserved_bytes: i64,
    pub(crate) spills: i64,
    pub(crate) spilled_bytes: i64,
    pub(crate) spill_files: i64,
    pub(crate) spill_write_bytes: i64,
    pub(crate) pipelines: Vec<i32>,
    pub(crate) extra: Vec<(String, String)>,
    pub(crate) workers: i32,
}

impl OperatorRow {
    /// The row of one operator entry of the query `query_id`.
    pub(crate) fn of(
        query_id: &str,
        id: krishiv_exec::OperatorId,
        e: &krishiv_exec::OperatorEntry,
        workers: u64,
    ) -> Self {
        let n = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        let ns = |d: std::time::Duration| i64::try_from(d.as_nanos()).unwrap_or(i64::MAX);
        let m = &e.metrics;
        Self {
            query_id: query_id.to_owned(),
            operator_id: i32::try_from(id.0).unwrap_or(i32::MAX),
            node_id: e.node.map(|v| i32::try_from(v).unwrap_or(i32::MAX)),
            name: e.name.to_string(),
            role: e.role.as_str(),
            drivers: n(m.drivers),
            rows_in: n(m.rows_in),
            rows_out: n(m.rows_out),
            batches_in: n(m.batches_in),
            batches_out: n(m.batches_out),
            compute_ns: ns(m.compute_time),
            max_driver_compute_ns: ns(m.max_driver_compute_time),
            blocked_io_ns: ns(m.blocked_on_io),
            blocked_memory_ns: ns(m.blocked_on_memory),
            blocked_sink_ns: ns(m.blocked_on_sink),
            peak_reserved_bytes: n(m.peak_reserved),
            spills: n(m.spills),
            spilled_bytes: n(m.spilled_bytes),
            spill_files: n(m.spill_files),
            spill_write_bytes: n(m.spill_write_bytes),
            pipelines: e
                .pipelines
                .iter()
                .map(|p| i32::try_from(*p).unwrap_or(i32::MAX))
                .collect(),
            extra: e
                .extra
                .iter()
                .map(|(name, v)| {
                    let value = match v {
                        krishiv_exec::Metric::Count(n) | krishiv_exec::Metric::Max(n) => {
                            n.to_string()
                        }
                        krishiv_exec::Metric::Text(t) => t.clone(),
                    };
                    (name.to_string(), value)
                })
                .collect(),
            workers: i32::try_from(workers).unwrap_or(i32::MAX),
        }
    }
}

/// What a finished query adds to its row.
#[derive(Debug)]
pub(crate) struct Finished<'a> {
    pub(crate) state: &'static str,
    pub(crate) wall_ns: i64,
    pub(crate) peak_memory_bytes: i64,
    pub(crate) error_class: Option<&'a str>,
    pub(crate) operators: Vec<OperatorRow>,
    /// Counters that need the query's metrics; `None` when it never ran.
    pub(crate) totals: Option<Totals>,
}

/// The query-wide counters of a finished query.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Totals {
    pub(crate) workload_class: &'static str,
    pub(crate) cpu_ns: i64,
    pub(crate) rows_produced: Option<i64>,
    /// The query's own I/O since it started (its I/O context; design P2-E1 R3).
    pub(crate) bytes_read: i64,
    pub(crate) bytes_written: i64,
    pub(crate) spill_bytes: i64,
    pub(crate) spill_files: i64,
    pub(crate) pipelines: i64,
    pub(crate) workers: i64,
}

/// The `system.queries` view of one session: its rows go into its runtime's
/// [`QueryHistory`] with its session id, principal, tags and trace id
/// (P2-T126 unit 2, design P2-E1 §5).
#[derive(Debug, Default)]
pub(crate) struct QueryLog {
    history: Arc<QueryHistory>,
    /// The session's operation ids ([`crate::QueryOptions::operation_id`]).
    operations: Arc<crate::history::Operations>,
    /// The session this log belongs to, and its principal (fixed for life).
    session_id: String,
    principal: String,
    /// The W3C trace the session's next queries belong to.
    trace_id: Mutex<Option<String>>,
    /// The session's tags, by key.
    tags: Mutex<std::collections::BTreeMap<String, String>>,
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

impl QueryLog {
    /// The log of a new session of the runtime that keeps `history`, run by
    /// `principal`.
    pub(crate) fn new(history: Arc<QueryHistory>, principal: &str) -> Self {
        Self {
            session_id: history.mint(now_ms()),
            principal: principal.to_owned(),
            history,
            ..Self::default()
        }
    }

    /// The session's id (UUIDv7).
    pub(crate) fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Removes the tag `key`, if there is one.
    pub(crate) fn remove_tag(&self, key: &str) {
        self.tags
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(key);
    }

    /// Sets the tag `key`; `false` (and nothing changed) when it is new and the session already has
    /// [`MAX_TAGS`]. The check and the insertion happen under one lock, so concurrent callers
    /// cannot both take the last place.
    pub(crate) fn try_set_tag(&self, key: &str, value: &str) -> bool {
        let mut tags = self.tags.lock().unwrap_or_else(PoisonError::into_inner);
        if !tags.contains_key(key) && tags.len() >= MAX_TAGS {
            return false;
        }
        tags.insert(key.to_owned(), value.to_owned());
        true
    }

    /// The tags now, by key.
    pub(crate) fn tags(&self) -> Vec<(String, String)> {
        self.tags
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Sets (or clears) the trace id the session's next queries record.
    pub(crate) fn set_trace_id(&self, trace_id: Option<String>) {
        *self.trace_id.lock().unwrap_or_else(PoisonError::into_inner) = trace_id;
    }

    /// The trace id now in effect.
    pub(crate) fn trace_id(&self) -> Option<String> {
        self.trace_id
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// A row for a query that starts now, named `id`.
    fn row(
        &self,
        id: String,
        ms: i64,
        text: Option<&str>,
        class: Option<&'static str>,
    ) -> QueryRow {
        let (text, truncated) = match text {
            None => (None, false),
            Some(t) if t.len() <= TEXT_BYTES => (Some(t.to_owned()), false),
            Some(t) => {
                let mut end = TEXT_BYTES;
                while !t.is_char_boundary(end) {
                    end -= 1;
                }
                (Some(t[..end].to_owned()), true)
            }
        };
        QueryRow {
            id,
            session_id: self.session_id.clone(),
            principal: self.principal.clone(),
            tags: self.tags(),
            trace_id: self.trace_id(),
            statement_class: class,
            state: "running",
            text,
            truncated,
            submitted_ms: ms,
            ended_ms: None,
            wall_ns: None,
            peak_memory_bytes: None,
            error_class: None,
            operators: Arc::default(),
            workload_class: None,
            cpu_ns: None,
            rows_produced: None,
            bytes_read: None,
            bytes_written: None,
            spill_bytes: None,
            spill_files: None,
            pipelines: None,
            workers: None,
            cancel_reason: None,
        }
    }

    /// A running row named `id` (tests of the history).
    #[cfg(test)]
    pub(crate) fn snapshot_row_for_tests(&self, id: &str) -> QueryRow {
        self.row(id.to_owned(), 0, None, None)
    }

    /// The session's operation ids.
    pub(crate) fn operations(&self) -> &Arc<crate::history::Operations> {
        &self.operations
    }

    /// Records a query that started now under a minted id and returns it.
    #[cfg(test)]
    pub(crate) fn begin(&self, text: Option<&str>, class: Option<&'static str>) -> String {
        self.record(text, class, None)
    }

    /// Records a query that started now, named `operation` when the caller
    /// gave the statement an id ([`crate::QueryOptions::operation_id`]), else
    /// a minted id, and returns the id.
    ///
    /// # Errors
    /// `INVALID_HANDLE.OPERATION_ALREADY_EXISTS` while another execution of
    /// this session runs under the same operation id.
    pub(crate) fn begin_as(
        &self,
        text: Option<&str>,
        class: Option<&'static str>,
        operation: Option<&crate::QueryId>,
    ) -> Result<String, KrishivError> {
        let Some(operation) = operation else {
            return Ok(self.record(text, class, None));
        };
        let id = operation.as_str().to_owned();
        self.history
            .push_given(self.row(id.clone(), now_ms(), text, class))?;
        Ok(id)
    }

    /// Records a row without the live-id check (a statement that never runs:
    /// refused, or answered from the result cache).
    fn record(
        &self,
        text: Option<&str>,
        class: Option<&'static str>,
        operation: Option<&crate::QueryId>,
    ) -> String {
        let ms = now_ms();
        let id = operation.map_or_else(|| self.history.mint(ms), |o| o.as_str().to_owned());
        self.history.push(self.row(id.clone(), ms, text, class));
        id
    }

    /// Closes the live row `id` as the outcome of `error` (a refused
    /// admission): `denied`, `cancelled` or `failed` by its class.
    pub(crate) fn record_failed_row(&self, id: &str, error: &KrishivError) {
        let class = error.error_class().name();
        self.finish(
            id,
            &Finished {
                state: outcome_state(class),
                wall_ns: 0,
                peak_memory_bytes: 0,
                error_class: (outcome_state(class) != "cancelled").then_some(class),
                operators: Vec::new(),
                totals: None,
            },
        );
    }

    /// Sets the state of the live row `id` (`queued` while admission waits,
    /// `running` once it is admitted).
    pub(crate) fn set_state(&self, id: &str, state: &'static str) {
        self.history
            .update(&self.session_id, id, |row| row.state = state);
    }

    /// The session's tags now.
    pub(crate) fn tags_now(&self) -> Vec<(String, String)> {
        self.tags()
    }

    /// Records a query answered from the result cache: a finished row of its own (with the
    /// session's current tags and trace id) that took no execution.
    pub(crate) fn record_replayed(
        &self,
        text: Option<&str>,
        class: Option<&'static str>,
        operation: Option<&crate::QueryId>,
    ) -> String {
        let id = self.record(text, class, operation);
        self.finish(
            &id,
            &Finished {
                state: "finished",
                wall_ns: 0,
                peak_memory_bytes: 0,
                error_class: None,
                operators: Vec::new(),
                totals: None,
            },
        );
        id
    }

    /// Records a statement that failed before it ran: parsing, analysis,
    /// an access check or physical planning. A denial is `denied`; a
    /// cancellation `cancelled`. Returns the row's id.
    pub(crate) fn record_refused(
        &self,
        text: Option<&str>,
        class: Option<&'static str>,
        error: &KrishivError,
        operation: Option<&crate::QueryId>,
    ) -> String {
        let id = self.record(text, class, operation);
        let class = error.error_class().name();
        let state = outcome_state(class);
        self.finish(
            &id,
            &Finished {
                state,
                wall_ns: 0,
                peak_memory_bytes: 0,
                error_class: (state != "cancelled").then_some(class),
                operators: Vec::new(),
                totals: None,
            },
        );
        id
    }

    /// Turns the finished query `id` into a failed one: the write it ran completed its stream but a
    /// step after it (the commit, a re-check of the path) failed, so the stream's `finished` was
    /// not the write's outcome. Its other figures stay.
    pub(crate) fn fail(&self, id: &str, error: &KrishivError) {
        let class = error.error_class().name();
        let state = outcome_state(class);
        self.history.update(&self.session_id, id, |row| {
            row.state = state;
            row.error_class = (state != "cancelled").then(|| class.to_owned());
        });
    }

    /// Marks the query `id` finished (a row evicted meanwhile is gone).
    pub(crate) fn finish(&self, id: &str, done: &Finished<'_>) {
        self.history.finish(&self.session_id, id, done);
    }

    /// The operators of the remembered queries of this session's principal,
    /// or of every principal when `all`, oldest query first.
    pub(crate) fn operators(&self, all: bool) -> Vec<OperatorRow> {
        self.history
            .operators((!all).then_some(self.principal.as_str()))
    }

    /// The rows of this session's principal, or of every principal when
    /// `all`, oldest first.
    pub(crate) fn snapshot(&self, all: bool) -> Vec<QueryRow> {
        self.history
            .snapshot((!all).then_some(self.principal.as_str()))
    }
}

/// The state a failed query shows for its error class.
fn outcome_state(class: &str) -> &'static str {
    match class {
        "KRISHIV_INSUFFICIENT_PERMISSIONS" => "denied",
        "OPERATION_CANCELED" | "KRISHIV_QUERY_TIMEOUT" | "KRISHIV_QUERY_KILLED" => "cancelled",
        _ => "failed",
    }
}

/// The table `table` as the planner reads it: its rows now.
pub(crate) fn provider(
    table: &str,
    sources: &Sources,
) -> Result<Option<Arc<MemoryTable>>, KrishivError> {
    let Some(layout) = layout(table) else {
        return Ok(None);
    };
    let plain = Arc::new(Schema::new(
        layout
            .iter()
            .map(|(name, kind, nullable)| {
                let ty = match kind {
                    Kind::Text => ArrowType::Utf8,
                    Kind::Long => ArrowType::Int64,
                    Kind::Int => ArrowType::Int32,
                    Kind::Flag => ArrowType::Boolean,
                    Kind::Time => ArrowType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    Kind::Double => ArrowType::Float64,
                    Kind::Labels => labels_type().to_arrow(),
                    Kind::IntList => DataType::Array(
                        krishiv_types::ArrayType::new(DataType::Integer, false).into(),
                    )
                    .to_arrow(),
                    Kind::Buckets => buckets_type().to_arrow(),
                };
                Field::new(*name, ty, *nullable)
            })
            .collect::<Vec<_>>(),
    ));
    let schema = crate::carrier::canonical_schema(Arc::clone(&plain));
    let batch = RecordBatch::try_new(plain, values(table, sources)).map_err(|e| InternalError {
        message: format!("system.{table}: {e}"),
    })?;
    let batch = crate::carrier::recarried(&batch, &schema)?;
    Ok(Some(Arc::new(MemoryTable::new(
        schema,
        vec![Batch::new(batch)],
    ))))
}

/// The `system` catalog: metadata only; the rows come from [`provider`].
#[derive(Debug)]
pub(crate) struct SystemCatalog {
    /// The query's `MONITOR` decision, made when a per-principal table loads.
    monitor: Arc<Monitor>,
}

impl SystemCatalog {
    /// The `system` catalog of the query that reads `sources`.
    pub(crate) fn of(sources: &Sources) -> Self {
        Self {
            monitor: Arc::clone(&sources.monitor),
        }
    }
}

impl Catalog for SystemCatalog {
    fn name(&self) -> &str {
        SYSTEM
    }

    fn epoch(&self) -> u64 {
        0
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _: Option<TableVersion>,
        _: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            // Two-part names only: `system.<table>`, with no namespace.
            if !id.namespace.is_empty() {
                return Ok(None);
            }
            if ["queries", "operators"]
                .iter()
                .any(|t| id.name.eq_ignore_ascii_case(t))
            {
                self.monitor.decide().await;
            }
            Ok(columns(&id.name).map(|columns| TableMeta {
                ident: TableIdent {
                    catalog: SYSTEM.to_owned(),
                    namespace: Vec::new(),
                    name: id.name.to_lowercase(),
                },
                columns,
                version: None,
                kind: TableKind::Table,
                file_metadata: Default::default(),
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_has_one_row_per_fact() {
        let rows = capability_rows();
        assert!(rows.iter().any(|r| r[0] == "build" && r[1] == "engine"));
        assert!(rows.iter().any(|r| r[0] == "feature" && r[1] == "cloud"));
        let memory: Arc<dyn MemoryPool> =
            krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
        let sources = Sources {
            queries: Arc::new(QueryLog::default()),
            io: krishiv_io::IoScheduler::new(krishiv_io::IoConfig::default()),
            relations: crate::cache::RelationCache::new(&memory),
            metrics: Arc::new(krishiv_exec::MetricsRegistry::new()),
            memory,
            monitor: Arc::new(Monitor::new(
                Arc::new(Principal::local_user()),
                Arc::new(krishiv_plan::analyzer::AllowAll),
            )),
        };
        assert!(provider("capabilities", &sources).unwrap().is_some());
        assert!(provider("nope", &sources).unwrap().is_none());
    }

    #[test]
    fn a_finished_query_can_be_turned_into_a_failed_one() {
        let log = QueryLog::default();
        let id = log.begin(Some("INSERT"), Some("write"));
        log.finish(
            &id,
            &Finished {
                state: "finished",
                wall_ns: 5,
                peak_memory_bytes: 7,
                error_class: None,
                operators: Vec::new(),
                totals: None,
            },
        );
        let error: KrishivError = krishiv_common::error::PathNotFound {
            path: "x".to_owned(),
        }
        .into();
        log.fail(&id, &error);
        let row = &log.snapshot(false)[0];
        assert_eq!(row.state, "failed");
        assert_eq!(row.error_class.as_deref(), Some("PATH_NOT_FOUND"));
        assert_eq!(row.wall_ns, Some(5));
        // A row evicted meanwhile is just gone.
        log.fail("unknown", &error);
        let cancelled: KrishivError = krishiv_common::error::OperationCanceled {}.into();
        log.fail(&id, &cancelled);
        let row = &log.snapshot(false)[0];
        assert_eq!(row.state, "cancelled");
        assert_eq!(row.error_class, None);
    }

    #[test]
    fn concurrent_tags_never_exceed_the_limit() {
        let log = QueryLog::default();
        let taken: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..MAX_TAGS * 2)
                .map(|i| {
                    let log = &log;
                    scope.spawn(move || log.try_set_tag(&format!("k{i}"), "v"))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| usize::from(h.join().unwrap_or(false)))
                .sum()
        });
        assert_eq!(taken, MAX_TAGS);
        assert_eq!(log.tags().len(), MAX_TAGS);
        // A known key can still change; an unknown one is refused until a tag is removed.
        assert!(log.try_set_tag("k0", "other") || log.tags().iter().all(|(k, _)| k != "k0"));
        assert!(!log.try_set_tag("new", "v"));
        let (key, _) = log.tags()[0].clone();
        log.remove_tag(&key);
        assert!(log.try_set_tag("new", "v"));
    }

    #[test]
    fn a_replayed_query_is_a_finished_row() {
        let log = QueryLog::default();
        log.set_trace_id(Some("trace".to_owned()));
        assert!(log.try_set_tag("team", "etl"));
        let id = log.record_replayed(Some("SELECT 1"), Some("READ"), None);
        let row = &log.snapshot(false)[0];
        assert_eq!(row.id, id);
        assert_eq!(row.state, "finished");
        assert_eq!(row.statement_class, Some("READ"));
        assert_eq!(row.tags, [("team".to_owned(), "etl".to_owned())]);
        assert_eq!(row.error_class, None);
    }

    #[test]
    fn a_query_pool_is_named_by_its_history_id_and_found_again() {
        let log = QueryLog::default();
        let id = log.begin(Some("SELECT 1"), None);
        let pool = pool_name(&id);
        assert_eq!(query_of_path(&format!("process/{pool}")), Some(id.clone()));
        assert_eq!(
            query_of_path(&format!("process/{pool}/hash-join")),
            Some(id)
        );
        assert_eq!(
            query_of_path(&format!("process/session:abc/{pool}/sort")),
            query_of_path(&format!("process/{pool}"))
        );
        assert_eq!(query_of_path("process/session:abc"), None);
        assert_eq!(query_of_path("process/shuffle"), None);
        assert_eq!(query_of_path("process"), None);
    }

    #[test]
    fn query_ids_are_uuid_v7_and_unique() {
        let log = QueryLog::default();
        let a = log.begin(Some("SELECT 1"), Some("read"));
        let b = log.begin(None, None);
        assert_ne!(a, b);
        for id in [&a, &b] {
            let p: Vec<&str> = id.split('-').collect();
            assert_eq!(
                p.iter().map(|x| x.len()).collect::<Vec<_>>(),
                [8, 4, 4, 4, 12],
                "{id}"
            );
            assert!(p[2].starts_with('7') && "89ab".contains(&p[3][..1]), "{id}");
        }
        log.finish(
            &a,
            &Finished {
                state: "finished",
                wall_ns: 5,
                peak_memory_bytes: 7,
                error_class: None,
                operators: Vec::new(),
                totals: None,
            },
        );
        let rows = log.snapshot(false);
        assert_eq!((rows[0].state, rows[1].state), ("finished", "running"));
    }

    #[test]
    fn a_history_copy_shares_the_operator_rows() {
        let log = QueryLog::default();
        let id = log.begin(None, None);
        let op = OperatorRow {
            query_id: id.clone(),
            operator_id: 0,
            node_id: None,
            name: "Scan".to_owned(),
            role: "source",
            drivers: 1,
            rows_in: 0,
            rows_out: 3,
            batches_in: 0,
            batches_out: 1,
            compute_ns: 0,
            max_driver_compute_ns: 0,
            blocked_io_ns: 0,
            blocked_memory_ns: 0,
            blocked_sink_ns: 0,
            peak_reserved_bytes: 0,
            spills: 0,
            spilled_bytes: 0,
            spill_files: 0,
            spill_write_bytes: 0,
            pipelines: vec![0],
            extra: Vec::new(),
            workers: 1,
        };
        log.finish(
            &id,
            &Finished {
                state: "finished",
                wall_ns: 1,
                peak_memory_bytes: 0,
                error_class: None,
                operators: vec![op],
                totals: None,
            },
        );
        let (a, b) = (log.snapshot(false), log.snapshot(false));
        assert!(Arc::ptr_eq(&a[0].operators, &b[0].operators));
        let ops = log.operators(false);
        assert_eq!((ops.len(), ops[0].rows_out), (1, 3));
    }

    #[test]
    fn history_is_bounded_and_long_text_is_cut_on_a_char() {
        let log = QueryLog::default();
        for _ in 0..(crate::history::PRINCIPAL_ROWS + 5) {
            let id = log.begin(None, None);
            log.finish(
                &id,
                &Finished {
                    state: "finished",
                    wall_ns: 1,
                    peak_memory_bytes: 0,
                    error_class: None,
                    operators: Vec::new(),
                    totals: None,
                },
            );
        }
        assert_eq!(log.snapshot(false).len(), crate::history::PRINCIPAL_ROWS);
        let long = "é".repeat(TEXT_BYTES);
        log.begin(Some(&long), None);
        let last = log.snapshot(false).pop().unwrap();
        assert!(last.truncated && last.text.unwrap().len() <= TEXT_BYTES);
    }

    #[test]
    fn traceparent_is_checked_as_w3c_says() {
        let ok = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        assert_eq!(
            parse_traceparent(ok).as_deref(),
            Some("4bf92f3577b34da6a3ce929d0e0e4736")
        );
        for bad in [
            "",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
        ] {
            assert!(parse_traceparent(bad).is_none(), "{bad}");
        }
        assert!(
            parse_traceparent(&format!("01-{}-00f067aa0ba902b7-01-more", &ok[3..35])).is_some()
        );
    }

    #[test]
    fn names_are_matched_without_case() {
        assert!(is_system("SYSTEM"));
        assert!(is_system_table(&[
            "System".to_owned(),
            "Capabilities".to_owned()
        ]));
        assert!(!is_system_table(&["system".to_owned(), "x".to_owned()]));
        assert!(!is_system_table(&["system".to_owned()]));
    }
}
