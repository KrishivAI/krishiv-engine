- A temporary view over a persisted file read (`spark.read.csv(p).persist()` then `createOrReplaceTempView`) reads
  the cached rows; it used to list and scan the files at every query.
- An explicit `krishiv.memory.limit` above the process's cgroup limit or physical memory is fitted to 80% of it
  with a warning, so a query spills instead of the kernel killing the process without an error (db-benchmark
  groupby q10 at 1e8 rows).
