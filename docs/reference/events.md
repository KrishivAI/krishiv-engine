---
description: "The query event stream: kinds, the krishiv.event.v1 JSON schema, usage records and delivery guarantees."
---

# Query events (`krishiv.event.v1`)

The engine emits one event for every transition of a query, from one choke point: the rows of `system.queries`.
A consumer subscribes with a sink: the JSON-lines file sink (`krishiv.events.file.path`) or a sink a host links
with `EngineRuntimeBuilder::event_sink`. With no sink registered no event is built.

## Kinds

| `kind` | When | Extra members |
|---|---|---|
| `submitted` | the statement was recorded (a refused one too) | `statement_class`, `text` (always literal-masked, whatever `krishiv.system.queryText.literals` says; cut to `krishiv.events.textMaxBytes`) |
| `queued` | admission holds the query in its queue | |
| `started` | admission let it run | `resource_group`, `queued_ns` |
| `finished` | it ended: `state` is `finished`, `failed` or `denied` | `state`, `usage`, `error_class`, `statement_class`, `result_cache_hit` |
| `cancelled` | it was cancelled | `reason`, `usage` |

A write whose commit fails after its result stream ended gets a second `finished` (state `failed`) with a higher
`seq`: keep the highest. Error messages are never carried, only the error class; no credential is.
Every event has `schema` (`krishiv.event.v1`), `minor`, `seq` (bus-wide: one counter per runtime, so interleaved queries see gaps and a reused operation id never repeats a `seq`; order the events of a query by `seq`), `emitted_at_ms`, `kind` and
`query` (`query_id`, `session_id`, `principal`, `delegated_by`, `compute_id`, `trace_id`, `tags`).
Consumers ignore unknown members and unknown kinds: minor versions only add.

## Usage record

`usage` is cumulative since the query started and keyed by `record_id = (session_id, query_id, seq)` (a caller's operation id is unique only per session). The
event schema is experimental until the design note's review findings are settled; do not treat `record_id` as a
billing-grade key yet. Members: `kind` (`final`), `fidelity` (`exact`), `principal`, `delegated_by`, `session_id`,
`compute_id`, `workload_class`, `resource_group`, `statement_class`, `window_start_ms`, `window_end_ms`,
`wall_ns` (queue time excluded for a query that started; a query cancelled before it started reports the time since its submission), `queued_ns`, `cpu_ns`, `bytes_scanned`, `rows_read`, `files_read`, `bytes_written`,
`rows_written`, `files_written`, `spill_bytes`, `spill_files`, `peak_memory_bytes`, `result_cache_hit`.
The figures are the `system.queries` row's.

## Delivery

Each sink has a bounded queue (`krishiv.events.queueCapacity`) and its own thread; emitting never blocks and never
fails a query. A full queue drops `progress` events first, then the oldest non-terminal event, and a terminal event
only when nothing else can go; drops are counted (`krishiv.events.dropped`, gauge `krishiv.events.queue_depth`).
`krishiv.events.dropped` counts queue drops (a full queue, a queue closed because its thread did not start, a queue not drained at shutdown). A sink's reported write failures are counted there too (`EventSink::lost`: the JSON-lines sink counts the events of a batch it could not write, less those already flushed); a failed write cuts the segment back to its last flushed whole line and moves the sink to a new segment. The contract is: never blocks a query, every queue loss is counted, the final usage is recoverable from
`system.queries` while the row is held. Dropping a runtime waits up to 5 s per sink for its queue to drain.

Not built yet: `progress` samples, audit events, an OTLP logs sink.
