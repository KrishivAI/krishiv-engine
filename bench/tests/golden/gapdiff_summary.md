# gapdiff tpch SF100, 12 threads

Process CPU seconds of the profiled run for each engine (`basis` says when a query has operator seconds instead, which are wall time per thread summed over threads and inflate under load). `top gap` is the kind where Krishiv spends the most beyond DuckDB; `flags` counts pipelines with a parallelism flag; `load` marks a run that passed the harness's load bound.

| query | Krishiv s | DuckDB s | ratio | gap s | basis | top gap | flags | load |
|---|---:|---:|---:|---:|---|---|---:|---|
| q01 | 9.000 | 3.000 | 3.00x | +6.000 | cpu | scan | 0 | ok |
| q13 | 24.500 | 8.000 | 3.06x | +16.500 | cpu | join | 2 | INDICATIVE |
| q15 | - | - | - | - | - | not compared: RunnerError: [KRISHIV_UNSUPPORTED_FEATURE] a view over a CTE | - | - |

Geometric mean of the ratios over 2 queries: 3.03x.
