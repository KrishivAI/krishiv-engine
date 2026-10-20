# EXPLAIN ANALYZE and EXPLAIN VERBOSE

Two query-debugging statements that Spark does not have. Spark's own
`EXPLAIN [EXTENDED | CODEGEN | COST | FORMATTED]` works as in Spark 4.2.

| Statement | What it does |
|---|---|
| `EXPLAIN ANALYZE <query>` | Runs the query, drops its rows and prints the physical plan with each operator's metrics. |
| `EXPLAIN VERBOSE <query>` | Prints the plan at every stage: parsed, analyzed, after each optimizer rule that changed it (with the lines it changed), optimized and physical. Does not run the query. |

In Spark 4.2 both are syntax errors. `EXPLAIN ANALYZE TABLE ...` still parses
as Spark's `ANALYZE TABLE`, which Krishiv does not support yet. Unlike
Spark's `EXPLAIN`, a query that fails to analyze, plan or run is an error
here, not part of the text.

## Entry points

| Where | Profile (`ANALYZE`) | Stages (`VERBOSE`) |
|---|---|---|
| SQL | `EXPLAIN ANALYZE SELECT ...`: one row, column `plan` | `EXPLAIN VERBOSE SELECT ...` |
| Rust | `DataFrame::explain_analyze()` returns a `QueryProfile`; `collect_with_profile()` also returns the rows | `DataFrame::explain_verbose()` returns an `ExplainVerbose` |
| Python | `df.explain(analyze=True)`; `df.explainString(analyze=True, json=True)` | `df.explain(verbose=True)`; `df.explainString(verbose=True, json=True)` |
| CLI | `krishiv sql --explain-analyze [--json] -e '...'` | `krishiv sql --explain-verbose [--json] -e '...'` |

`QueryProfile::to_text` prints the text below; `to_json` writes the JSON
document. Pass `TextOptions::masked()` to replace the values that change
from run to run (times, memory, driver and thread counts, hash-seed
dependent table statistics) with `<t>`, `<mem>` and `<n>`, for golden tests.

## Reading the profile

TPC-H Q18 at scale factor 1 on 12 threads (excerpt):

```text
== Physical Plan (analyzed) ==
wall 508.6 ms, 7 pipelines, 54 drivers, 12 threads (most in one pipeline), 1 worker
TopN 100, [o_totalprice#11 DESC NULLS LAST, o_orderdate#12 ASC NULLS FIRST]
:  · sink   #1 TopN p1: in 57 rows (1 batch); compute 13.1 ms (max 13.1 ms, 1 driver); peak mem 7.3 KiB
:  · source #22 TopN p0: out 57 rows (1 batch); compute 5.3 µs (max 5.3 µs, 1 driver); io 0.0 µs (max 0.0 µs)
+- Project [c_name#1, c_custkey#0, o_orderkey#8, o_orderdate#12, o_totalprice#11, sum(l_quantity#21)#4611686018427387905 AS sum(l_quantity)#50]
   :  · operator #21 Project p1: in 57 rows (1 batch); out 57 rows (1 batch); compute 1.2 µs (max 1.2 µs, 1 driver)
   +- HashAggregate keys=[c_name#1, c_custkey#0, o_orderkey#8, o_orderdate#12, o_totalprice#11], functions=[sum(l_quantity#21)]
      :  · sink   #2 HashAggregate p2: in 399 rows (56 batches); compute 937.7 µs (max 222.0 µs, 12 drivers); peak mem 9.9 MiB; groups=57 spill_runs=0 spill_files=0 spill_bytes=0 B repartitions=0 hash_entries=57 hash_buckets=128 (load 0.45) hash_index=swiss probe_lookups=447 probe_key_compares=346 (avg 0.77) max_probe_length=1
      ...
                     :  · sink   #4 HashAggregate p4: in 6001215 rows (1484 batches); compute 966.7 ms (max 260.6 ms, 12 drivers); peak mem 655.1 MiB; groups=1500000 spill_runs=0 spill_files=0 spill_bytes=0 B repartitions=0 hash_entries=1500000 hash_buckets=2097152 (load 0.72) hash_index=swiss probe_lookups=7359710 probe_key_compares=4863528 (avg 0.66) max_probe_length=6
                     :  · source #6 HashAggregate p3: out 1500000 rows (367 batches); compute 128.3 ms (max 17.4 ms, 12 drivers); io 0.0 µs (max 0.0 µs)
                     +- Scan spark_catalog.default.lineitem [l_orderkey#17, l_quantity#21] (ids +16)
                           · source #5 FileScan p4: out 6001215 rows (1484 batches); compute 145.8 ms (max 19.0 ms, 12 drivers); io 117.2 ms (max 13.1 ms); peak mem 19.6 MiB
Result:
   sink   #0 Result p0: in 57 rows (1 batch); compute 1.3 µs (max 1.3 µs, 1 driver)
Pipelines:
   p2: wall 125.5 ms, 12 drivers, 12 threads: #16 FileScan -> #17 HashJoinProbe -> #18 Project -> #19 HashJoinProbe -> #2 HashAggregate
   ...
```

Each plan node is followed by the operators that implement it, one line
each: role (`source`, `operator`, `sink`), operator id, name and the
pipelines it runs in. A pipeline breaker (aggregate, sort, the build side
of a hash join) has a sink in one pipeline and a source in the next.

| Field | Meaning |
|---|---|
| `in` / `out` | Rows and batches received and produced. |
| `compute` | Time in the operator, summed over its drivers; `max` is the largest single driver's share, so `max` close to `compute / drivers` means an even spread and `max` close to `compute` means skew. |
| `io` | For a source: time waiting for data (I/O), summed, with the per-driver maximum. |
| `memory wait`, `sink wait` | Time waiting for a memory grant and for a full downstream buffer. |
| `peak mem` | Peak bytes reserved through the operator's memory consumer, summed over drivers (an upper bound of the true peak). |
| `spilled` | Spills serviced and the bytes they freed. |
| `runtime_filter_*`, `runtime_filters` | Rows a scan's runtime filters tested and dropped, units and splits they pruned, and per filter the form its producer chose (`range`, `small set of N`, `hash set of N`, `bloom N B`). |
| `hash_entries`, `hash_buckets` (load), `hash_index` | The operator's hash tables: distinct keys, buckets and the index kind (a join's probe side reports the tables its grace phase built after spilling as `grace_hash_entries` and `grace_hash_buckets`) (`swiss`: an open-addressing table of keys; `swiss+chain`: the same with duplicate build rows chained; `row-flat+chain` and `row-salted+chain`: the row-layout join index, flat or salted by size). |
| `probe_lookups`, `probe_key_compares` (avg), `max_probe_length` | Lookups, the keys compared to answer them (the probe length) and the longest probe. Collected only by `EXPLAIN ANALYZE` (`ExecConfig::detailed_metrics`), and not for the row-layout join index, because they cost a counter per lookup; other runs show `(probe lengths not collected)`. |
| Pipelines | Per pipeline: wall time from its first driver's start to its last driver's end, drivers, threads that ran them, and its operators from source to sink. |

Other operator-specific counters (an aggregate's `groups` and spill runs,
a join's `build_rows`, grace partitions and recursion depth) are printed
by name. A name ending in `_bytes` is a size.

## The JSON document

**Experimental: the JSON documents may change before 1.0.**

`QueryProfile::to_json` writes one JSON object with `"format":
"krishiv.query-profile"` and `"version": 1`; its JSON Schema is
[`query-profile-v1.schema.json`](query-profile-v1.schema.json). Times are
integer nanoseconds (`*_ns`) and sizes integer bytes. The plan is a flat
list in pre-order, each node naming its inputs by position, so deep plans
stay shallow. `QueryProfile::from_json` reads it back exactly and rejects
another format or version, missing or mistyped members, unknown roles or
metric kinds, duplicate ids and a plan that is not one tree. A change that
removes or retypes a member bumps the version.

`EXPLAIN VERBOSE`'s document has `"format": "krishiv.explain-verbose"`,
`"version": 1` and the members `parsed`, `analyzed`, `steps` (each with
`batch`, `round`, `rule`, `plan` and `diff`), `optimized` and `physical`.

## Metrics are data

Metrics belong to the query that produced them: each driver counts in
plain fields and merges into the query once when it ends, and nothing is
registered globally. `QueryMetrics::merge` and `QueryProfile::merge`
combine the metrics of one plan run by several workers (sums add, maxima
take the larger value, descriptions are kept once each), in any order.

## Cost

The always-on counters (rows, batches, times, per-driver maxima, pipeline
times and threads, hash-table sizes, runtime-filter counters) are updated
per batch or per driver, not per row. The per-lookup probe counters are
compiled out of the plain hash-table loops and run only under `EXPLAIN
ANALYZE`.
