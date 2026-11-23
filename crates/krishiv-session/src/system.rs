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

use std::sync::{Arc, Mutex, PoisonError};
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
    BoxFuture, Catalog, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};

use crate::history::QueryHistory;
use crate::session::Session;
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

/// The catalog's name.
pub(crate) const SYSTEM: &str = "system";

/// The schema version of every table in this unit (`krishiv.schema.version`).
const SCHEMA_VERSION: &str = "1";

/// The worker this process is, in the `worker_id` column.
const LOCAL_WORKER: &str = "local";

/// Every table of the catalog.
const TABLES: [&str; 10] = [
    "capabilities",
    "caches",
    "catalogs",
    "memory",
    "metrics",
    "operators",
    "partition_advice",
    "queries",
    "resource_groups",
    "sessions",
];

/// The names of the tables of the catalog.
pub(crate) fn table_names() -> &'static [&'static str] {
    &TABLES
}

/// Whether `name` is the reserved catalog name, in any letter case.
pub(crate) fn is_system(name: &str) -> bool {
    name.eq_ignore_ascii_case(SYSTEM)
}

/// Whether a scan of `name` reads a system table: `system.<table>`.
pub(crate) fn is_system_table(name: &[String]) -> bool {
    matches!(name, [catalog, table] if is_system(catalog) && table_known(table))
}

/// Whether `plan` reads only system tables: every leaf a scan of a system table (at least one),
/// and no one-row relation (inline `VALUES`), range, generator or recursion that makes rows of
/// its own. Admission is skipped only for such a plan; anything else, such as
/// `system.queries CROSS JOIN range(..)`, is admitted as usual (fail closed).
pub(crate) fn plan_reads_only_system(plan: &krishiv_exec::physical::PhysicalPlan) -> bool {
    use krishiv_exec::physical::PhysNode;
    let mut stack: Vec<&PhysNode> = plan.roots().collect();
    let mut system_scan = false;
    while let Some(node) = stack.pop() {
        match node {
            PhysNode::Scan(scan) => match &scan.scan.source {
                krishiv_plan::ReadSource::Table { name, .. } if is_system_table(name) => {
                    system_scan = true;
                }
                _ => return false,
            },
            PhysNode::OneRow
            | PhysNode::Range(_)
            | PhysNode::Generate { .. }
            | PhysNode::UnionLoop { .. }
            | PhysNode::LoopRef { .. } => return false,
            other => stack.extend(other.children()),
        }
    }
    system_scan
}

fn table_known(table: &str) -> bool {
    TABLES.iter().any(|t| t.eq_ignore_ascii_case(table))
}

/// What a column holds.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
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
    /// `ARRAY<STRUCT<code: STRING, node_id: INT, operator_id: INT, message: STRING>>`.
    Hints,
}

/// A table's columns: `(name, kind, nullable)`.
fn layout(table: &str) -> Option<Vec<(&'static str, Kind, bool)>> {
    use Kind::{Buckets, Double, Flag, Hints, Int, IntList, Labels, Long, Text, Time};
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
            ("memory_bytes", Long, false),
            ("disk_bytes", Long, false),
            ("entries_pending", Long, false),
            ("spills", Long, false),
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
            ("session_id", Text, false),
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
            ("started_at", Time, true),
            ("ended_at", Time, true),
            ("wall_ns", Long, true),
            ("peak_memory_bytes", Long, true),
            ("error_class", Text, true),
            ("workload_class", Text, true),
            ("resource_group", Text, true),
            ("queued_ns", Long, true),
            ("cpu_ns", Long, true),
            ("rows_produced", Long, true),
            ("bytes_read", Long, true),
            ("rows_read", Long, true),
            ("rows_written", Long, true),
            ("files_read", Long, true),
            ("files_written", Long, true),
            ("result_cache_hit", Flag, false),
            ("bytes_written", Long, true),
            ("spill_bytes", Long, true),
            ("spill_files", Long, true),
            ("pipelines", Long, true),
            ("workers", Long, true),
            ("cancel_reason", Text, true),
            ("sqlstate", Text, true),
            ("error_message", Text, true),
            ("partial", Flag, false),
            ("delegated_by", Text, true),
            ("hints", Hints, true),
        ],
        "partition_advice" => crate::partition_advice::layout(),
        "resource_groups" => crate::system_groups::layout(),
        "catalogs" => crate::system_catalogs::layout(),
        "sessions" => crate::sessions::layout(),
        _ => return None,
    })
}

/// The columns of `table`, `None` for an unknown table (also what `DESCRIBE` and `SHOW COLUMNS`
/// print, which need `Browse`, not the `Select` that building its rows does).
pub(crate) fn columns(table: &str) -> Option<Vec<TableColumn>> {
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
                    Kind::Buckets => columns::buckets_type(),
                    Kind::Hints => columns::hints_type(),
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
pub(crate) fn labels_column(rows: &[&[(String, String)]]) -> ArrayRef {
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

/// Rows of text cells, column by column.
pub(crate) fn text<'a>(cells: impl Iterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(StringArray::from_iter(cells))
}

pub(crate) fn long(cells: impl Iterator<Item = Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from_iter(cells))
}

/// A byte count as `BIGINT`, saturating.
fn bytes(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

#[cfg(test)]
use crate::trace_context::{MAX_BAGGAGE_BYTES, MAX_BAGGAGE_ENTRIES};
pub(crate) use crate::trace_context::{RemoteParent, parse_baggage, parse_remote_parent};
#[cfg(test)]
use krishiv_plan::analyzer::Principal;

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
            columns::buckets_column(&buckets),
            Arc::new(TimestampMicrosecondArray::from_value(now, rows.len()).with_timezone("UTC")),
        ];
    }
    if table.eq_ignore_ascii_case("caches") {
        return caches::columns(sources);
    }
    if table.eq_ignore_ascii_case("operators") {
        let rows = sources.queries.operators(sources.monitor.sees_all());
        let l = |f: fn(&OperatorRow) -> i64| long(rows.iter().map(move |r| Some(f(r))));
        return vec![
            text(rows.iter().map(|r| Some(r.query_id.as_str()))),
            text(rows.iter().map(|r| Some(r.session_id.as_str()))),
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
    if table.eq_ignore_ascii_case("sessions") {
        return crate::sessions::columns(sources);
    }
    if table.eq_ignore_ascii_case("partition_advice") {
        return crate::partition_advice::columns(sources);
    }
    if table.eq_ignore_ascii_case("resource_groups") {
        return crate::system_groups::columns(sources);
    }
    if table.eq_ignore_ascii_case("catalogs") {
        return crate::system_catalogs::columns(sources);
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
                TimestampMicrosecondArray::from_iter(rows.iter().map(|r| micros(r.started_ms)))
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
            text(rows.iter().map(|r| r.resource_group.as_deref())),
            long(rows.iter().map(|r| r.queued_ns)),
            long(rows.iter().map(|r| r.cpu_ns)),
            long(rows.iter().map(|r| r.rows_produced)),
            long(rows.iter().map(|r| r.bytes_read)),
            long(rows.iter().map(|r| r.rows_read)),
            long(rows.iter().map(|r| r.rows_written)),
            long(rows.iter().map(|r| r.files_read)),
            long(rows.iter().map(|r| r.files_written)),
            Arc::new(BooleanArray::from_iter(
                rows.iter().map(|r| Some(r.result_cache_hit)),
            )),
            long(rows.iter().map(|r| r.bytes_written)),
            long(rows.iter().map(|r| r.spill_bytes)),
            long(rows.iter().map(|r| r.spill_files)),
            long(rows.iter().map(|r| r.pipelines)),
            long(rows.iter().map(|r| r.workers)),
            text(rows.iter().map(|r| r.cancel_reason)),
            // The class's SQLSTATE (Spark's registry); NULL for a class the registry lacks.
            text(rows.iter().map(|r| {
                r.error_class
                    .as_deref()
                    .and_then(krishiv_common::error::ErrorClass::from_name)
                    .and_then(krishiv_common::error::ErrorClass::sql_state)
            })),
            text(rows.iter().map(|r| r.error_message.as_deref())),
            // One process, one worker: every row is whole (design §2.7 for the merged rows).
            Arc::new(BooleanArray::from_iter(rows.iter().map(|_| Some(false)))),
            text(rows.iter().map(|r| r.delegated_by.as_deref())),
            columns::hints_column(&rows.iter().map(|r| r.hints.as_deref()).collect::<Vec<_>>()),
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
    /// The runtime's sessions.
    pub(crate) sessions: Arc<crate::sessions::SessionView>,
    /// The resource group controller (`krishiv.workload.resourceGroups.file`), when one is set.
    pub(crate) groups: Option<Arc<crate::resource_groups::ResourceGroups>>,
    /// The advice of the persisted query history (`krishiv.history.path`), read off the executor when the
    /// `partition_advice` table loads.
    pub(crate) advice: Arc<crate::partition_advice::AdviceCell>,
    /// The declared catalogs the query's principal sees (`system.catalogs`, name resolution).
    pub(crate) catalogs: crate::catalog_registry::DeclaredView,
}

pub(crate) use crate::monitor::Monitor;

/// Most tags a session carries, and the longest key and value.
pub(crate) const MAX_TAGS: usize = 32;
pub(crate) const MAX_TAG_KEY: usize = 128;
pub(crate) const MAX_TAG_VALUE: usize = 256;

/// One query of `system.queries`.
#[derive(Debug, Clone)]
pub(crate) struct QueryRow {
    pub(crate) id: String,
    pub(crate) session_id: String,
    pub(crate) principal: String,
    /// The service principal that ran the query for `principal` (trusted impersonation, P3-T061).
    pub(crate) delegated_by: Option<String>,
    /// [`crate::sessions::owner_key`] of the principal: what the own-rows filter compares.
    pub(crate) owner: String,
    pub(crate) tags: Vec<(String, String)>,
    pub(crate) trace_id: Option<String>,
    /// `READ`, `WRITE`, `DDL`, `SESSION` or `ADMIN` (T137); `None` when the
    /// statement was refused before it was classified or came from no SQL.
    pub(crate) statement_class: Option<&'static str>,
    pub(crate) state: &'static str,
    pub(crate) text: Option<String>,
    pub(crate) truncated: bool,
    pub(crate) submitted_ms: i64,
    /// When admission let the query run (`None` while queued, or for a row that never ran).
    pub(crate) started_ms: Option<i64>,
    /// When it was submitted on the monotonic clock: a queued query's wait
    /// does not move with the system clock.
    pub(crate) submitted_at: std::time::Instant,
    pub(crate) ended_ms: Option<i64>,
    /// `system.queries.wall_ns`: for a query that ran, its execution time (from the start of its run, after
    /// admission, to its end), so a queued wait is not in it; for one that never ran (refused, cancelled while
    /// queued, or ended before its run) the time since it was submitted, its wait included; 0 for a statement
    /// refused before it was admitted and for a result-cache replay. `ended_at - submitted_at` is the
    /// submit-to-end time of a query that reached admission (a statement refused earlier misses its parse and
    /// analysis there).
    pub(crate) wall_ns: Option<i64>,
    pub(crate) peak_memory_bytes: Option<i64>,
    pub(crate) error_class: Option<String>,
    /// The failure's message, cut and masked like the text.
    pub(crate) error_message: Option<String>,
    /// The text policy the query started under: its error message is cut and masked by it, whatever the
    /// session's policy is by the time it fails.
    pub(crate) policy: TextPolicy,
    /// The operators it ran (a finished query only); shared, so a copy of
    /// the history doesn't copy them.
    pub(crate) operators: Arc<[OperatorRow]>,
    pub(crate) workload_class: Option<&'static str>,
    /// The resource group the controller placed the query in (`None` without one).
    pub(crate) resource_group: Option<String>,
    /// How long admission held the query queued, in nanoseconds (`None` for a query that never
    /// reached admission; 0 when it started at once).
    pub(crate) queued_ns: Option<i64>,
    /// Busy time of its drivers.
    pub(crate) cpu_ns: Option<i64>,
    pub(crate) rows_produced: Option<i64>,
    pub(crate) bytes_read: Option<i64>,
    /// Rows the scan operators produced.
    pub(crate) rows_read: Option<i64>,
    /// Rows the file write sinks took.
    pub(crate) rows_written: Option<i64>,
    /// Files its scans opened.
    pub(crate) files_read: Option<i64>,
    /// Data files the query wrote to completion.
    pub(crate) files_written: Option<i64>,
    /// Answered from the result cache, with no execution.
    pub(crate) result_cache_hit: bool,
    pub(crate) bytes_written: Option<i64>,
    pub(crate) spill_bytes: Option<i64>,
    pub(crate) spill_files: Option<i64>,
    pub(crate) pipelines: Option<i64>,
    pub(crate) workers: Option<i64>,
    /// Why the query was cancelled ([`crate::CancelReason::as_str`]), for a
    /// query cancelled with a reason; `state` stays `cancelled`.
    pub(crate) cancel_reason: Option<&'static str>,
    /// The advisor's hints for a finished query; `None` for one that did not finish or ran with
    /// `krishiv.advisor.enabled=false` (`Some` of none: finished, nothing to say).
    pub(crate) hints: Option<Arc<[HintRow]>>,
}

/// One operator of a finished query, as `system.operators` lists it.
#[derive(Debug, Clone)]
pub(crate) struct OperatorRow {
    pub(crate) query_id: String,
    /// The session that ran the query (set when the query finishes).
    pub(crate) session_id: String,
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
            session_id: String::new(),
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
    /// The advisor's hints; `None` when none was computed.
    pub(crate) hints: Option<Vec<HintRow>>,
}

/// The query-wide counters of a finished query.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Totals {
    pub(crate) workload_class: &'static str,
    pub(crate) cpu_ns: i64,
    pub(crate) rows_produced: Option<i64>,
    /// The query's own I/O since it started (its I/O context; design P2-E1 R3). The totals in an
    /// `Option` are those of the plan's scans and pipelines: `None` for a query whose operators are
    /// withheld (`QueryEnv::withhold_profile`, review r2 item 5).
    pub(crate) bytes_read: Option<i64>,
    /// Rows the scans of the plan produced (not a breaker's output side).
    pub(crate) rows_read: Option<i64>,
    /// Rows the file write sinks of the plan took.
    pub(crate) rows_written: i64,
    /// Files its scans opened.
    pub(crate) files_read: Option<i64>,
    /// Files the query's writers completed.
    pub(crate) files_written: i64,
    pub(crate) bytes_written: i64,
    pub(crate) spill_bytes: Option<i64>,
    pub(crate) spill_files: Option<i64>,
    pub(crate) pipelines: Option<i64>,
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
    /// The service principal acting for `principal`, if any (fixed for life, as the principal).
    delegated_by: Option<String>,
    /// [`crate::sessions::owner_key`] of the principal.
    owner: String,
    /// The W3C trace the session's next queries belong to, and the caller's
    /// span they are children of.
    trace: Mutex<Option<RemoteParent>>,
    /// The `baggage` of the inbound request (untrusted, bounded).
    baggage: Mutex<Vec<(String, String)>>,
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
    pub(crate) fn new(
        history: Arc<QueryHistory>,
        principal: &krishiv_plan::analyzer::Principal,
    ) -> Self {
        Self {
            session_id: history.mint(now_ms()),
            principal: principal.name.clone(),
            delegated_by: principal.delegated_by.clone(),
            owner: crate::sessions::owner_key(principal),
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

    /// Sets (or clears) the trace the session's next queries record.
    pub(crate) fn set_trace(&self, trace: Option<RemoteParent>) {
        *self.trace.lock().unwrap_or_else(PoisonError::into_inner) = trace;
    }

    /// The trace now in effect.
    pub(crate) fn trace(&self) -> Option<RemoteParent> {
        self.trace
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The trace id now in effect.
    pub(crate) fn trace_id(&self) -> Option<String> {
        self.trace().map(|t| t.trace_id)
    }

    /// Replaces the inbound baggage.
    pub(crate) fn set_baggage(&self, baggage: Vec<(String, String)>) {
        *self.baggage.lock().unwrap_or_else(PoisonError::into_inner) = baggage;
    }

    /// The inbound baggage now.
    pub(crate) fn baggage(&self) -> Vec<(String, String)> {
        self.baggage
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// A row for a query that starts now, named `id`.
    fn row(
        &self,
        policy: TextPolicy,
        id: String,
        ms: i64,
        text: Option<&str>,
        class: Option<&'static str>,
    ) -> QueryRow {
        let (text, truncated) = match text {
            None => (None, false),
            Some(t) => {
                let (shown, cut) = policy.apply(t);
                (Some(shown), cut)
            }
        };
        QueryRow {
            id,
            session_id: self.session_id.clone(),
            principal: self.principal.clone(),
            delegated_by: self.delegated_by.clone(),
            owner: self.owner.clone(),
            tags: self.tags(),
            trace_id: self.trace_id(),
            statement_class: class,
            state: "running",
            text,
            truncated,
            submitted_ms: ms,
            started_ms: None,
            submitted_at: std::time::Instant::now(),
            ended_ms: None,
            wall_ns: None,
            peak_memory_bytes: None,
            error_class: None,
            error_message: None,
            policy,
            operators: Arc::default(),
            workload_class: None,
            resource_group: None,
            queued_ns: None,
            cpu_ns: None,
            rows_produced: None,
            bytes_read: None,
            rows_read: None,
            rows_written: None,
            files_read: None,
            files_written: None,
            result_cache_hit: false,
            bytes_written: None,
            spill_bytes: None,
            spill_files: None,
            pipelines: None,
            workers: None,
            cancel_reason: None,
            hints: None,
        }
    }

    /// A running row named `id` (tests of the history).
    #[cfg(test)]
    pub(crate) fn snapshot_row_for_tests(&self, id: &str) -> QueryRow {
        self.row(TextPolicy::default(), id.to_owned(), 0, None, None)
    }

    /// The session's operation ids.
    pub(crate) fn operations(&self) -> &Arc<crate::history::Operations> {
        &self.operations
    }

    /// Records a query that started now under a minted id and returns it.
    #[cfg(test)]
    pub(crate) fn begin(&self, text: Option<&str>, class: Option<&'static str>) -> String {
        self.record(TextPolicy::default(), text, class, None)
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
        policy: TextPolicy,
        text: Option<&str>,
        class: Option<&'static str>,
        operation: Option<&crate::QueryId>,
    ) -> Result<String, KrishivError> {
        let Some(operation) = operation else {
            return Ok(self.record(policy, text, class, None));
        };
        let id = operation.as_str().to_owned();
        self.history
            .push_given(self.row(policy, id.clone(), now_ms(), text, class))?;
        Ok(id)
    }

    /// Records a row without the live-id check (a statement that never runs:
    /// refused, or answered from the result cache).
    fn record(
        &self,
        policy: TextPolicy,
        text: Option<&str>,
        class: Option<&'static str>,
        operation: Option<&crate::QueryId>,
    ) -> String {
        let ms = now_ms();
        let id = operation.map_or_else(|| self.history.mint(ms), |o| o.as_str().to_owned());
        self.history
            .push(self.row(policy, id.clone(), ms, text, class));
        id
    }

    /// Closes the live row `id` as the outcome of `error` (a refused
    /// admission): `denied`, `cancelled` or `failed` by its class.
    pub(crate) fn record_failed_row(&self, id: &str, error: &KrishivError) {
        let class = error.error_class().name();
        let state = outcome_state(class);
        self.note_error(id, error);
        self.close_unrun(id, state, (state != "cancelled").then_some(class));
    }

    /// Closes the live row `id` of a query that never ran (refused, cancelled while queued,
    /// or ended before its run): its wall time is the time since it was submitted, so the
    /// queue time is not lost.
    pub(crate) fn close_unrun(&self, id: &str, state: &'static str, error_class: Option<&str>) {
        // A statement dropped while it waited because its action was cancelled
        // (`action.rs`; the drop runs with the action current): name the reason.
        if state == "cancelled"
            && let Some(reason) = crate::action::current_reason()
        {
            self.note_cancel_reason(id, reason);
        }
        let waited = self
            .history
            .submitted_at(&self.session_id, id)
            .map_or(0, |at| {
                i64::try_from(at.elapsed().as_nanos()).unwrap_or(i64::MAX)
            });
        self.finish(
            id,
            &Finished {
                state,
                wall_ns: waited,
                peak_memory_bytes: 0,
                error_class,
                operators: Vec::new(),
                totals: None,
                hints: None,
            },
        );
    }

    /// Records `reason` as the `cancel_reason` of row `id` unless it has one
    /// (the first reason wins, as for a running query).
    pub(crate) fn note_cancel_reason(&self, id: &str, reason: crate::CancelReason) {
        self.history.update(&self.session_id, id, |row| {
            if row.cancel_reason.is_none() {
                row.cancel_reason = Some(reason.as_str());
            }
        });
    }

    /// Sets the state of the live row `id` (`queued` while admission waits,
    /// `running` once it is admitted).
    pub(crate) fn set_state(&self, id: &str, state: &'static str) {
        self.history
            .update(&self.session_id, id, |row| row.state = state);
    }

    /// Names the group the controller queued live row `id` in.
    pub(crate) fn set_group(&self, id: &str, group: &str) {
        self.history.update(&self.session_id, id, |row| {
            row.resource_group = Some(group.to_owned());
        });
    }

    /// Stamps the start of live row `id`: admission let it run.
    pub(crate) fn mark_started(&self, id: &str, group: Option<&str>, queued_ns: u64) {
        self.history.update(&self.session_id, id, |row| {
            row.started_ms = Some(now_ms());
            // What the controller held the query in its queue (0 when it started at once).
            row.queued_ns = Some(i64::try_from(queued_ns).unwrap_or(i64::MAX));
            row.resource_group = group.map(str::to_owned);
        });
    }

    /// The session's tags now.
    pub(crate) fn tags_now(&self) -> Vec<(String, String)> {
        self.tags()
    }

    /// Records a query answered from the result cache: a finished row of its own (with the
    /// session's current tags and trace id) that took no execution.
    pub(crate) fn record_replayed(
        &self,
        policy: TextPolicy,
        text: Option<&str>,
        class: Option<&'static str>,
        operation: Option<&crate::QueryId>,
    ) -> String {
        let id = self.record(policy, text, class, operation);
        self.history
            .update(&self.session_id, &id, |row| row.result_cache_hit = true);
        self.finish(
            &id,
            &Finished {
                state: "finished",
                wall_ns: 0,
                peak_memory_bytes: 0,
                error_class: None,
                operators: Vec::new(),
                totals: None,
                hints: None,
            },
        );
        id
    }

    /// Records a statement that failed before it ran: parsing, analysis,
    /// an access check or physical planning. A denial is `denied`; a
    /// cancellation `cancelled`. Returns the row's id.
    pub(crate) fn record_refused(
        &self,
        policy: TextPolicy,
        text: Option<&str>,
        class: Option<&'static str>,
        error: &KrishivError,
        operation: Option<&crate::QueryId>,
    ) -> String {
        let id = self.record(policy, text, class, operation);
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
                hints: None,
            },
        );
        self.note_error(&id, error);
        id
    }

    /// Turns the finished query `id` into a failed one: the write it ran completed its stream but a
    /// step after it (the commit, a re-check of the path) failed, so the stream's `finished` was
    /// not the write's outcome. Its other figures stay.
    pub(crate) fn fail(&self, id: &str, error: &KrishivError) {
        let class = error.error_class().name();
        let state = outcome_state(class);
        self.note_error(id, error);
        self.history.update(&self.session_id, id, |row| {
            row.state = state;
            row.error_class = (state != "cancelled").then(|| class.to_owned());
        });
    }

    /// Records the message of `error` on the row `id`: the client form (operator-only parameters
    /// redacted, no cause), cut and masked by the text policy (a cancellation has none).
    pub(crate) fn note_error(&self, id: &str, error: &KrishivError) {
        if outcome_state(error.error_class().name()) == "cancelled" {
            return;
        }
        let client = error.for_client();
        let message = client.error().to_string();
        let parameters = client.error().message_parameters();
        self.history.update(&self.session_id, id, |row| {
            row.error_message = Some(row.policy.apply_message_with(&message, parameters));
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
            .operators((!all).then_some(self.owner.as_str()))
    }

    /// The rows of this session's principal, or of every principal when
    /// `all`, oldest first.
    pub(crate) fn snapshot(&self, all: bool) -> Vec<QueryRow> {
        self.history.snapshot((!all).then_some(self.owner.as_str()))
    }
}

/// The state a failed query shows for its error class.
pub(crate) fn outcome_state(class: &str) -> &'static str {
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
                    Kind::Buckets => columns::buckets_type().to_arrow(),
                    Kind::Hints => columns::hints_type().to_arrow(),
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
    /// The history's advice, which `partition_advice` reads once the decision is made.
    advice: Arc<crate::partition_advice::AdviceCell>,
    /// The session catalog of the query, asked first for a bare name: a temporary table or view of the
    /// session wins over a system table of the same name, as it does over any table.
    temps: Option<Arc<crate::catalog::QueryCatalog>>,
}

impl SystemCatalog {
    /// This catalog, with the session's temporary relations in front of it.
    pub(crate) fn with_temps(mut self, session: Arc<crate::catalog::QueryCatalog>) -> Self {
        self.temps = Some(session);
        self
    }

    /// The `system` catalog of the query that reads `sources`.
    pub(crate) fn of(sources: &Sources) -> Self {
        Self {
            monitor: Arc::clone(&sources.monitor),
            advice: Arc::clone(&sources.advice),
            temps: None,
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
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            if id.namespace.is_empty()
                && !form.qualified
                && let Some(temps) = &self.temps
            {
                let only_temporary = NameForm {
                    temporary_only: true,
                    ..form
                };
                if let Some(found) = temps.load_table(id, version, only_temporary).await? {
                    return Ok(Some(found));
                }
                // The path names no schema with tables: a bare name is a temporary relation or nothing.
                if form.temporary_only {
                    return Ok(None);
                }
            }
            // Two-part names only: `system.<table>`, with no namespace.
            if !id.namespace.is_empty() {
                return Ok(None);
            }
            if [
                "queries",
                "operators",
                "sessions",
                "resource_groups",
                "partition_advice",
            ]
            .iter()
            .any(|t| id.name.eq_ignore_ascii_case(t))
            {
                self.monitor.decide().await;
                // The history read is file I/O: done here, in the async load, on the blocking pool.
                if id.name.eq_ignore_ascii_case("partition_advice") && self.monitor.sees_all() {
                    self.advice.load().await;
                }
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
                column_comments: Default::default(),
            }))
        })
    }
}

impl QueryRow {
    /// What the row holds on the heap, about: its strings, tags and
    /// operators, plus the struct. Counted for the history's byte bound, so
    /// an estimate that is stable for a given row is all it needs.
    pub(crate) fn heap_bytes(&self) -> usize {
        let opt = |s: &Option<String>| s.as_ref().map_or(0, String::len);
        let operators: usize = self
            .operators
            .iter()
            .map(|o| {
                std::mem::size_of::<OperatorRow>()
                    + o.query_id.len()
                    + o.name.len()
                    + o.session_id.len()
                    + o.pipelines.len() * 4
                    + o.extra
                        .iter()
                        .map(|(k, v)| size_of::<(String, String)>() + k.len() + v.len())
                        .sum::<usize>()
            })
            .sum();
        std::mem::size_of::<Self>()
            + self.id.len()
            + self.session_id.len()
            + self.principal.len()
            + self.delegated_by.as_ref().map_or(0, String::len)
            + self.owner.len()
            + self
                .tags
                .iter()
                .map(|(k, v)| size_of::<(String, String)>() + k.len() + v.len())
                .sum::<usize>()
            + opt(&self.trace_id)
            + opt(&self.text)
            + opt(&self.error_class)
            + opt(&self.error_message)
            + self.hints.as_deref().map_or(0, |h| {
                h.iter()
                    .map(|h| size_of::<HintRow>() + h.message.len())
                    .sum::<usize>()
            })
            + operators
    }
}

impl crate::table_commands::TableName {
    /// The name as an authorization statement writes it: ``spark_catalog`.`ns`.`name` ``. A `TableName` is always
    /// in the session catalog, whatever catalog is current.
    pub(crate) fn quoted_in_session_catalog(&self) -> String {
        format!(
            "`{}`.`{}`.`{}`",
            crate::catalog::CATALOG,
            self.namespace.replace('`', "``"),
            self.name.replace('`', "``")
        )
    }
}

/// Resolution of bare names while the current catalog is `system`, which holds no stored objects:
/// a bare name must not fall through to `spark_catalog.default`, which authorization did not check.
impl Session {
    /// The namespace a bare name of a stored object resolves in, refused under `system` and under a
    /// declared catalog ([`Self::refuse_declared_current`]).
    pub(crate) fn stored_namespace(&self) -> Result<String, KrishivError> {
        self.refuse_declared_current()?;
        let (catalog, namespace) = self.state.catalog.position_pair();
        if is_system(&catalog) {
            return Err(krishiv_common::error::SchemaNotFound {
                schema_name: format!("`{catalog}`"),
            }
            .into());
        }
        Ok(namespace)
    }

    /// The permit of a query that reads only `system.*` (never queued behind the load it
    /// inspects): it starts at once, so its row has a start time.
    pub(crate) fn exempt_permit(&self, log_id: &str) -> crate::admission::Permit {
        self.state.queries.mark_started(log_id, None, 0);
        crate::admission::Permit::unlimited()
    }

    /// Whether `parts`, the name after `USE NAMESPACE|DATABASE|SCHEMA`, is a namespace of the current `system`
    /// catalog (which has none): any name while it is current, except one qualified with the session catalog.
    pub(crate) fn names_a_system_namespace(&self, parts: &[String]) -> bool {
        is_system(&self.state.catalog.current_catalog())
            && !(parts.len() == 2 && parts[0].eq_ignore_ascii_case(crate::catalog::CATALOG))
    }

    /// A namespace named on its own (`SHOW TABLES IN ns`, `DESCRIBE SCHEMA ns`): `system` itself stays
    /// nameable (it lists the system tables), any other is refused under `system`.
    pub(crate) fn namespace_arg(&self, ns: &str) -> Result<String, KrishivError> {
        if is_system(ns) || ns.eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE) {
            return Ok(ns.to_owned());
        }
        self.function_namespace(ns)
    }

    /// Refuses a namespace or table command that would resolve in the current catalog when that is
    /// a declared one: a declared catalog resolves names in queries only for now (design P3-T067
    /// PR-58a), and the embedded catalog is never read or changed in its name. `spark_catalog.<ns>`
    /// forms, `global_temp` and temporary views are not refused.
    pub(crate) fn refuse_declared_current(&self) -> Result<(), KrishivError> {
        let catalog = self.state.catalog.current_catalog();
        if is_system(&catalog) || catalog.eq_ignore_ascii_case(crate::catalog::CATALOG) {
            return Ok(());
        }
        Err(krishiv_common::error::KrishivUnsupportedFeature {
            feature: format!(
                "namespace and table commands in the declared catalog `{}`",
                catalog.replace('`', "``")
            ),
        }
        .into())
    }

    /// Whether `name` (any letter case) names a catalog this session sees: the session catalog,
    /// `system`, or a declared catalog visible to its principal (a hidden one is absent).
    pub(crate) fn is_catalog_name(&self, name: &str) -> bool {
        name.eq_ignore_ascii_case(crate::catalog::CATALOG)
            || is_system(name)
            || self.declared_catalogs().canonical(name).is_some()
    }

    /// The namespace `ns` of a stored object (a function, a namespace named on its own): refused under `system`, which has none,
    /// and under a declared catalog ([`Self::refuse_declared_current`]).
    pub(crate) fn function_namespace(&self, ns: &str) -> Result<String, KrishivError> {
        self.refuse_declared_current()?;
        if is_system(&self.state.catalog.current_catalog()) {
            return Err(krishiv_common::error::SchemaNotFound {
                schema_name: format!("`system`.`{ns}`"),
            }
            .into());
        }
        Ok(ns.to_owned())
    }

    /// Whether `name` (bare, as written) is a temporary view or table of the session, which
    /// resolves before any catalog.
    pub(crate) fn is_temp_name(&self, name: &str, env: &crate::session::QueryEnv) -> bool {
        !crate::catalog::is_global_key(name)
            && self
                .state
                .catalog
                .exists(name, env.analysis.conf.case_sensitive)
    }

    /// The namespace `parts` (a table or view name as written) resolve in: the current one, and a
    /// bare name is `TABLE_OR_VIEW_NOT_FOUND` under `system`.
    pub(crate) fn namespace_for(
        &self,
        parts: &[String],
        env: &crate::session::QueryEnv,
    ) -> Result<String, KrishivError> {
        self.namespace_with(parts, env, true)
    }

    /// [`namespace_for`](Self::namespace_for) for the target a writer creates or appends to: a stored table,
    /// so a temporary relation of the same bare name is no exemption under `system`.
    pub(crate) fn namespace_for_stored(
        &self,
        parts: &[String],
        env: &crate::session::QueryEnv,
    ) -> Result<String, KrishivError> {
        self.namespace_with(parts, env, false)
    }

    fn namespace_with(
        &self,
        parts: &[String],
        env: &crate::session::QueryEnv,
        temp_exempt: bool,
    ) -> Result<String, KrishivError> {
        // A bare name is the session's temporary view or nothing; a two-part name is a namespace of the
        // CURRENT catalog, and `system` has none.
        let global =
            parts.len() == 2 && parts[0].eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE);
        if !global
            && (parts.len() == 2
                || (parts.len() == 1 && !(temp_exempt && self.is_temp_name(&parts[0], env))))
            && env.catalog.in_system()
        {
            return Err(krishiv_common::error::TableOrViewNotFound {
                relation_name: format!("`{}`", parts.join("`.`")),
                // From the query's own snapshot, not the session as it is now.
                search_path: crate::catalog::render_search_path(&env.catalog.sql_path_snapshot()),
            }
            .into());
        }
        if !global
            && (parts.len() == 2
                || (parts.len() == 1 && !(temp_exempt && self.is_temp_name(&parts[0], env))))
        {
            self.refuse_declared_current()?;
        }
        Ok(env.catalog.namespace().join("."))
    }

    /// The namespace a bare name of a stored object resolves in, from the query's own catalog snapshot
    /// (the one the statement was authorized in), refused under `system`.
    pub(crate) fn stored_namespace_in(
        &self,
        env: &crate::session::QueryEnv,
    ) -> Result<String, KrishivError> {
        if env.catalog.in_system() {
            return Err(krishiv_common::error::SchemaNotFound {
                schema_name: format!("`{SYSTEM}`"),
            }
            .into());
        }
        self.refuse_declared_current()?;
        Ok(env.catalog.namespace().join("."))
    }
}

mod columns;
#[cfg(test)]
mod tests;
pub(crate) use columns::HintRow;

mod caches;
mod capabilities;
mod row_stamps;
mod text;
pub(crate) use text::TextPolicy;
#[cfg(test)]
use text::*;

use capabilities::capability_rows;
pub use capabilities::{capabilities, register_build_features, register_surface};
