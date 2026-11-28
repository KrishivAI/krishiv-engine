---
description: "Profile the time and memory of a Python UDF with spark.sql.pyspark.udf.profiler and spark.profile."
---

# Profile a Python UDF

Set `spark.sql.pyspark.udf.profiler` to `perf` or `memory`, run the query, and read the result with `spark.profile`. It is off by default, and while it is off a UDF runs exactly as before.

```python
import krishiv
from krishiv.sql import functions as F

spark = krishiv.session()

@F.udf("long")
def plus_one(x):
    return x + 1

spark.conf.set("spark.sql.pyspark.udf.profiler", "perf")
spark.range(1000).select(plus_one("id")).collect()
spark.profile.show()
```

`show` prints one block per UDF, headed `Profile of UDF<id=N>`; with `perf` the block is the `cProfile` table (calls, time per function, sorted by time).

| Call | Does |
| --- | --- |
| `spark.profile.show(id=None, *, type=None)` | prints the profiles of UDF `id` (all when `None`) of `type` (`perf` and `memory` when `None`) |
| `spark.profile.showPerfProfiles(id=None)`, `showMemoryProfiles(id=None)` | `show` for one type |
| `spark.profile.dump(path, id=None, *, type=None)` | writes `udf_<id>_perf.pstats` (open with `pstats.Stats`) and `udf_<id>_memory.json` into `path` |
| `spark.profile.clear(id=None, *, type=None)` | forgets what was collected; the profiler keeps collecting |

## Differences from Spark

- `memory` reports, per UDF, the calls, the peak and the net bytes the calls allocated (`tracemalloc`), not Spark's line-by-line `memory_profiler` report, and it needs no extra package.
- A profiled call takes a process-wide lock (also across sessions), because `cProfile` and `tracemalloc` are one per process: while a profiler is on, calls of profiled UDFs do not overlap, free-threaded Python included. Turn it off for timing the whole query.
- Whether a call is profiled is decided when the column is built (`plus_one("id")`), by the setting at that moment, not when the query runs.
- Nested profiled calls are not supported: a profiled UDF must not run a query that calls another profiled UDF. The process lock is re-entrant, so on the same thread the inner call does not wait; on Python 3.12 and later enabling the inner `cProfile` raises `ValueError` (one profiler per process). The inner call waits for the lock only when it runs on another thread.
- The profiler setting is read from the session the builder created (`SparkSession.builder.getOrCreate()`), not from a session made with `newSession()`: set `spark.sql.pyspark.udf.profiler` on the builder's session. Profiling per session is a follow-up.
- The memory dump file is `.json`, where Spark writes `.txt`. `memory` resets the peak of a `tracemalloc` you started yourself.
- `spark.profile.render` is not available.
- A UDF registered with `spark.udf.register` and called from SQL is not profiled yet; call the UDF object from the DataFrame API.
- Another profiler or `tracemalloc` running at the same time is not supported: the UDF's profile also counts work other threads do while it runs.
- It profiles UDFs that run in your process. UDFs on a `krishiv server`'s isolated workers are not profiled yet.
