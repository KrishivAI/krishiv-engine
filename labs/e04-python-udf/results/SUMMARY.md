# E04 Python UDF runtime - results summary

Environment: **shared laptop** (i7-9750H, 12 HW threads) with other experiments running; every timed run held `/home/gopal/krishiv-data/.bench.lock`; loadavg recorded per run in the JSON (0 of 48 full runs started with load1 > 8). Relative numbers only. 32 threads is **oversubscribed** (> 12 HW threads).
Python 3.14.8 (GIL) and 3.14.8t (free-threaded, GIL stayed off after importing pyarrow/pandas/numpy: `sys._is_gil_enabled()==False`), pyarrow 26.0.0, pandas 3.0.6, numpy 2.5.4 (all have 3.14t wheels). Rust 1.99.1 (root toolchain pin), pyo3 0.30.1, arrow-rs 60.0.0. Arrow crosses via a hand-written PyCapsule bridge (`src/bridge.rs`, `arrow::ffi`); self-test proves buffer-address identity both directions (zero-copy).
Options: A in-process GIL; B in-process free-threaded; C subinterpreters (`concurrent.interpreters`, PEP 734; `Interpreter.prepare_main` + `exec`, writable memoryviews over Rust buffers); D one Python worker process per Rust thread, Arrow IPC through a memfd shared-memory region (host serialises batch into shm: one copy; worker reads zero-copy; result written to shm by pyarrow).
UDFs: cheap = Int64 x+1; expensive = Utf8 -> Int64 count of digit runs by regex. Batches of 8192 rows, checksum-verified. Cells = rows/s median (p50/p99 batch latency ms). "D vs X" = D slowdown, positive = D slower.

## Full (10M rows, 5 iters + 2 warmups; top-2 per kind/body by smoke, plus A and D for vectorised kinds)
| kind | body | thr | A GIL | B free-thr | C subinterp | D procs+shm | D vs A | D vs B |
|---|---|---|---|---|---|---|---|---|
| scalar | cheap | 1 | - | - | 8.68M [0.93/2.22] | 6.51M [1.15/2.92] | - | - |
| scalar | cheap | 8 | - | - | 32.87M [2.34/3.64] | 21.99M [3.05/4.64] | - | - |
| scalar | cheap | 32 | - | - | 30.80M [6.83/26.51] | 30.18M [8.64/27.29] | - | - |
| scalar | expensive | 1 | - | 1.66M [4.98/8.00] | - | 1.28M [6.45/9.85] | - | +22.5% |
| scalar | expensive | 8 | - | 3.81M [16.22/23.54] | - | 3.69M [19.17/23.77] | - | +3.1% |
| scalar | expensive | 32 | - | 3.81M [67.86/146.44] | - | 4.65M [56.54/116.52] | - | -22.0% |
| pandas | cheap | 1 | 58.61M [0.13/0.24] | 46.97M [0.16/0.26] | - | 29.91M [0.27/0.38] | +49.0% | +36.3% |
| pandas | cheap | 8 | 35.10M [1.53/4.52] | 191.16M [0.34/0.56] | - | 129.34M [0.49/0.95] | -268.5% | +32.3% |
| pandas | cheap | 32 | 33.65M [5.86/18.73] | 160.16M [1.38/4.90] | - | 118.09M [1.72/6.74] | -251.0% | +26.3% |
| pandas | expensive | 1 | 3.04M [2.64/4.65] | 2.99M [2.76/4.82] | - | 2.78M [2.86/4.62] | +8.6% | +7.0% |
| pandas | expensive | 8 | 9.45M [5.92/10.87] | 10.83M [4.48/9.41] | - | 8.71M [8.17/13.00] | +7.8% | +19.6% |
| pandas | expensive | 32 | 8.49M [27.53/67.86] | 9.77M [26.11/63.34] | - | 8.80M [29.00/66.85] | -3.7% | +9.9% |
| arrow | cheap | 1 | 311.56M [0.03/0.05] | 276.32M [0.03/0.03] | - | 71.12M [0.11/0.22] | +77.2% | +74.3% |
| arrow | cheap | 8 | 130.47M [0.42/1.54] | 1198.81M [0.04/0.11] | - | 259.67M [0.25/0.43] | -99.0% | +78.3% |
| arrow | cheap | 32 | 125.94M [1.76/7.25] | 880.20M [0.07/2.65] | - | 244.47M [0.90/2.97] | -94.1% | +72.2% |
| arrow | expensive | 1 | 3.29M [2.50/3.58] | 3.02M [2.65/4.52] | - | 2.94M [2.79/4.62] | +10.6% | +2.6% |
| arrow | expensive | 8 | 10.77M [4.40/9.70] | 11.30M [4.18/9.25] | - | 9.57M [6.17/12.54] | +11.1% | +15.3% |
| arrow | expensive | 32 | 9.95M [25.47/54.94] | 10.48M [23.92/76.81] | - | 9.19M [26.71/70.45] | +7.6% | +12.3% |

(`-` = not run in full. scalar/cheap top-2 were D and C; A/B scalar-cheap are in the smoke table.)

## Smoke (1M rows, 3 iters + 2 warmups, all options)
| kind | body | thr | A GIL | B free-thr | C subinterp | D procs+shm | D vs A | D vs B |
|---|---|---|---|---|---|---|---|---|
| scalar | cheap | 1 | 9.09M [0.85/1.73] | 6.09M [1.32/2.05] | 11.37M [0.71/0.88] | 9.46M [0.85/1.49] | -4.1% | -55.4% |
| scalar | cheap | 8 | 3.23M [4.74/152.42] | 8.84M [4.36/19.21] | 13.91M [3.29/10.31] | 18.71M [3.14/8.77] | -480.2% | -111.8% |
| scalar | cheap | 32 | 3.10M [41.37/250.04] | 12.92M [9.58/40.31] | 14.41M [12.52/42.07] | 26.57M [6.33/20.34] | -758.3% | -105.7% |
| scalar | expensive | 1 | 0.64M [10.18/18.83] | 0.77M [9.04/21.65] | 0.77M [10.66/20.36] | 1.00M [7.92/11.86] | -58.0% | -29.8% |
| scalar | expensive | 8 | 1.29M [22.65/254.77] | 3.95M [17.25/22.65] | 3.70M [14.71/27.24] | 4.36M [17.27/20.17] | -236.7% | -10.4% |
| scalar | expensive | 32 | 1.20M [147.98/707.39] | 3.79M [55.16/163.66] | 3.56M [64.20/194.37] | 4.54M [51.19/137.30] | -278.2% | -19.9% |
| pandas | cheap | 1 | 59.12M [0.13/0.22] | 45.90M [0.15/0.30] | unsupported | 32.84M [0.24/0.33] | +44.5% | +28.5% |
| pandas | cheap | 8 | 33.51M [1.54/4.94] | 189.38M [0.26/0.49] | - | 141.94M [0.45/0.53] | -323.6% | +25.1% |
| pandas | cheap | 32 | 25.81M [6.83/20.02] | 83.27M [0.76/6.55] | - | 103.50M [1.70/5.40] | -301.0% | -24.3% |
| pandas | expensive | 1 | 1.68M [4.70/7.67] | 1.65M [4.69/8.48] | unsupported | 1.59M [4.78/9.05] | +5.4% | +3.6% |
| pandas | expensive | 8 | 5.37M [10.80/22.74] | 7.14M [9.52/10.55] | - | 6.49M [10.33/15.20] | -21.0% | +9.1% |
| pandas | expensive | 32 | 8.48M [25.33/57.65] | 9.23M [19.64/93.97] | - | 8.27M [25.79/73.96] | +2.4% | +10.3% |
| arrow | cheap | 1 | 283.80M [0.02/0.07] | 190.77M [0.03/0.08] | unsupported | 70.96M [0.10/0.19] | +75.0% | +62.8% |
| arrow | cheap | 8 | 94.18M [0.58/1.73] | 328.50M [0.07/0.28] | - | 237.89M [0.25/0.33] | -152.6% | +27.6% |
| arrow | cheap | 32 | 92.43M [1.83/6.69] | 364.28M [0.07/1.16] | - | 170.19M [0.98/3.24] | -84.1% | +53.3% |
| arrow | expensive | 1 | 2.46M [3.20/5.56] | 2.41M [3.31/7.12] | unsupported | 2.02M [3.98/6.79] | +18.0% | +16.1% |
| arrow | expensive | 8 | 6.56M [9.47/17.84] | 5.84M [9.52/20.94] | - | 6.12M [9.96/18.96] | +6.6% | -4.9% |
| arrow | expensive | 32 | 6.78M [31.51/81.04] | 7.73M [23.75/77.81] | - | 8.56M [26.98/63.56] | -26.2% | -10.7% |

Cheap vectorised smoke cells run only a few ms per iteration and are noisy; trust the full table.

## Other findings
* **Option C is not viable for vectorised UDFs**: `pyarrow.lib`, numpy and pandas refuse to import in subinterpreters (single-phase init). Only pure-Python scalar UDFs on raw buffers work (that is what C measures); it beat A at >=8 threads (no shared GIL) and matches D.
* **Pin the worker thread state** (`results/attach_pin.json`): naive per-call `Python::attach` costs ~1.7 ms per call on 3.14t (arrow cheap, 1 thread: 4.9M vs 143M rows/s pinned) and ~0.02 ms on 3.14; all reported numbers use pinned threads (`PyThreadPin`). `results/prelim_smoke_naive_attach.json` is an earlier unpinned smoke run, kept for transparency (and shows B misleadingly slow).
* GIL build (A) does not scale: arrow-cheap 8 threads is 2.4x slower than 1 thread (130M vs 312M); B scales ~4x; D scales but is capped by IPC.
* Crash (`results/crash.json`, ctypes.string_at(0) in the UDF): A, B, C -> host process killed by SIGSEGV (exit 139), blast radius = whole engine. D -> only the worker died; host observed EOF + "killed by signal 11", respawned in ~223 ms, and processed the following 50 batches correctly.
* Memory visibility (`results/memvis_*.json`, 256 MB probes): tracemalloc sees Python objects and numpy but NOT pyarrow-pool buffers (visible only via `pa.total_allocated_bytes()`) nor native-extension mallocs; RSS sees everything but is process-wide. In-process, the host cannot attribute per-UDF memory or cap it. In D the host reads each worker's `/proc/<pid>/status` (saw +256 MB exactly) and RLIMIT_AS=1.5 GB turned a 2 GB allocation into a worker `MemoryError` with the host unaffected. Peak RSS: D costs ~30-100 MB per worker (pyarrow+pandas import) - 8 workers ~0.9 GB vs ~0.13 GB in-process.
* CPU per 1k rows (1 thread): arrow cheap A 3.2 / B 3.6 / D 14.0 us; expensive bodies ~equal across options (310-360 us).
