# gapdiff tpch SF100, 12 threads

CPU seconds summed over threads. `top gap` is the kind where Krishiv spends the most CPU beyond DuckDB; `flags` counts pipelines whose busiest driver does over half of the pipeline's CPU.

| query | Krishiv CPU s | DuckDB CPU s | ratio | gap s | top gap | flags |
|---|---:|---:|---:|---:|---|---:|
| q01 | 9.000 | 3.000 | 3.00x | +6.000 | scan | 0 |
| q13 | 24.500 | 8.000 | 3.06x | +16.500 | join | 2 |
| q15 | - | - | - | - | not compared: RunnerError: [KRISHIV_UNSUPPORTED_FEATURE] a view over a CTE | - |

Geometric mean of the CPU ratios over 2 queries: 3.03x.
