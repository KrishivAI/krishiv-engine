- A `krishiv.memory.limit` above the process's cgroup limit or physical memory is no longer honoured: the default
  (80% of that limit) is used, with a warning, so a query spills instead of the kernel killing the process without
  an error (seen on db-benchmark's group-by question 10 at 1e8 rows).
