---
description: "Two opt-in session settings: reuse the result of an identical query, and make runs reproducible."
---

<!-- SPDX-License-Identifier: Apache-2.0 -->
# The result cache and reproducible results

Two opt-in session settings. Neither changes any query's answer: the result cache returns what an
earlier run of the same query returned, and reproducible results make two runs of a query agree
byte for byte.

## `krishiv.resultCache.*`

| Key | Default | Meaning |
|---|---|---|
| `krishiv.resultCache.enabled` | `false` | Keep the rows of a query for later runs of the same query. |
| `krishiv.resultCache.ttl` | `1d` | How long an entry is served, from when it was stored. `0s` never expires. |
| `krishiv.resultCache.maxBytes` | `64MB` | Bytes the cache holds. A runtime setting: set it when the session (or its engine runtime) is built; `SET` of it fails with `CANNOT_MODIFY_STATIC_CONFIG`. |

The store belongs to an engine runtime and is shared by every session on it: sessions built alone share one store for the process, and a runtime a host builds (`EngineRuntime`) has its own. Sessions built alone each bring their own cap to that shared store. It is separate from `DataFrame.cache()`
and `persist()`, which are a session's explicit snapshots, never expire (as in Spark) and are not
changed by this.

### What is cached

A query is stored and served only when all of these hold:

- it reads **permanent catalog tables** and nothing else. A temporary view over permanent tables is
  expanded into the plan, so it qualifies; a table registered from in-memory data, a read by path
  (`parquet.`/data``) and a plan that used a relation cached with `DataFrame.cache()` do not;
- it calls no non-deterministic function (`rand`, `uuid`, `input_file_name`, ...), no clock or
  session function (`current_timestamp`, `now`, `current_date`, `current_user`, `version`, ...), and no
  function registered in the session (its identity is a name and a version, not the code);
- its result fits `krishiv.resultCache.maxBytes`.

The key is the query's normalized plan, the table versions it was pinned to, the principal and the
access-policy version, the analysis configuration (ANSI mode, session time zone, ...), the identity
of each table read (its location and creation time, so a dropped and re-created table is a different
one), and the settings that change rows without changing the plan (`krishiv.deterministicOrder`, the
file and Parquet read settings, the time parser, map-key and store-assignment policies). Reads are
authorized before an entry is served, as for any query.

### When an entry goes

- `krishiv.resultCache.ttl` after it was stored;
- `REFRESH TABLE`, `CLEAR CACHE`, `spark.catalog.clearCache()`;
- every change the engine makes to a table an entry read: `INSERT`, `INSERT OVERWRITE`, `CREATE TABLE`,
  `saveAsTable`, `TRUNCATE`, `ALTER TABLE` (including `ADD PARTITION`), `DROP TABLE`, a rename, and a
  write by path into the table's directory.

A change made **outside** the engine (files copied into a table's directory by another program) is not
seen until the entry expires or `REFRESH TABLE` runs. That is the price of the cache and the reason it
is off by default.

### Seeing it

With the cache enabled, `EXPLAIN` ends with a section:

```text
== Result Cache ==
hit (metadata_age_ms=1532)
```

`miss (the result is stored when the query has run)`, or `not cached: <why>` (reads files by path, calls
a non-deterministic, clock or session function, reads a temporary or in-memory table, ...). Without the
setting the plan text is unchanged.

A second line gives the store's state and its counters since the process started, for telling whether
the cache is earning its memory:

```text
store: 3 entries, 51584 bytes; since start: 12 hits, 4 misses, 7 stored, 2 invalidated
```

`invalidated` counts entries dropped by a write, `REFRESH TABLE`, `CLEAR CACHE` or a drop, one per
entry; entries that expired or were evicted for room are not counted there, and `misses` counts the
lookups of queries that could be cached (an `EXPLAIN` looks without counting).

### Limits

The cache holds batches outside the query memory pool, so `krishiv.memory.limit` does not bound it; its
own cap does, and the setting of the session that stores a result applies. A result is charged at its
whole buffers (a sliced batch keeps its parent alive) and an entry costs at least 1 KiB; entries past
their ttl are dropped whenever a result is stored.

## `krishiv.deterministicOrder`

Off by default. When on, every pipeline of a query runs on **one driver**, so:

- the partial states of a decimal `sum` or `avg` are merged in the order the input arrives (whether a
  transient overflow happens can depend on the order of the merges);
- a query without `ORDER BY` returns its rows in the same order, batch for batch, on every run, for the
  engine's own scans.

It costs the query's parallelism, so use it to reproduce a result, to compare two builds, and in tests.
Memory pressure from other queries in the process can still change when an operator spills, and so the
order of what it returns. A `UNION ALL` under an operator that has to see every branch (a hash join
probing a union, say) still runs its branches side by side, so their rows can interleave differently;
unions that only stream rows are ordered by branch. Sources that reorder their own rows are outside it.
