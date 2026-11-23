---
description: "Record every query's lifecycle and usage in a JSON-lines file: turn the event sink on, read an event, and what is never recorded."
---

# Query events

Stability: experimental since 0.3. The event names and the `krishiv.event.v1` envelope stay; fields may be added in a
minor release.

A server can write one event for each step of every query (submitted, queued, started, finished or cancelled) to a
file. The final event carries the query's usage: wall time, CPU time, bytes and rows read and written, spill and peak
memory. Use it to feed a usage report, an audit trail or a dashboard without polling `system.queries`.

## Turn it on

Set a path prefix. The engine writes one JSON object per line to it:

```text
krishiv.events.file.path = /var/log/krishiv/events
```

The setting is a startup setting, so it goes on `krishiv server`, on `krishiv sql --conf` or on the embedded
session's builder, not on `SET`. With no path set no event is built and queries pay nothing.

| Key | Default | Meaning |
|---|---|---|
| `krishiv.events.file.path` | none | Path prefix of the event file. |
| `krishiv.events.file.maxBytes` | `256MB` | A new segment starts after this many bytes. |
| `krishiv.events.file.maxFiles` | `8` | Segments kept; older ones are removed. |
| `krishiv.events.queueCapacity` | `10000` | Events waiting for the writer. A full queue drops events and counts them. |
| `krishiv.events.textMaxBytes` | `4KB` | Most bytes of statement text in a `submitted` event. |
| `krishiv.compute.id` | host name | The `compute_id` every usage record carries. |

## Read an event

Run a statement with the sink on. `krishiv sql` takes the setting with `--conf`:

```bash
krishiv sql --conf krishiv.events.file.path=/tmp/events -e "SELECT 1 AS x, 'secret' AS s"
ls /tmp/events*        # events.000001.jsonl
```

The file holds three lines for this statement, `submitted`, `started` and `finished`. The first, formatted over
several lines here (the file has one line per event), shows the masked text:

```json
{
  "schema": "krishiv.event.v1", "minor": 0, "seq": 0, "kind": "submitted",
  "emitted_at_ms": 1795384799099,
  "query": {"query_id": "01a20545-ef7b-7001-95bd-056f047eeaf5", "session_id": "01a20545-ef26-7000-aa66-d5ab0940f24d",
            "principal": "krishiv", "compute_id": "krishiv2", "tags": {}, "delegated_by": null, "trace_id": null},
  "statement_class": "READ",
  "text": "SELECT ? AS x , ? AS s"
}
```

The `finished` event carries `state`, `error_class` and the `usage` record (here `rows_read` 1, `cpu_ns` 923988,
`wall_ns` 2941877, `peak_memory_bytes` 0).

Events of one query share `query.query_id`; order them by `seq`. A consumer ignores members and kinds it does not
know: later versions only add.

## What is never recorded

- Statement text always has its literals masked, whatever `krishiv.system.queryText.literals` says, and is cut to
  `krishiv.events.textMaxBytes`.
- Error messages are not recorded, only the error class. No credential is recorded.

## Limits

- Delivery never blocks or fails a query. A full queue drops events, counted in `krishiv.events.dropped`.
- The final usage of a query can still be read from `system.queries` while the row is held.
- Progress samples, audit events and an OTLP sink are not available yet. The event schema is experimental: do not
  treat `record_id` as a billing key.
