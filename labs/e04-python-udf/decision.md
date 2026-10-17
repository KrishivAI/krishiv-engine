# E04 decision - Python UDF runtime

## Data
Full 10M-row runs (results/SUMMARY.md, results/full.json; shared laptop, relative only):
* D overhead vs in-process, **cheap vectorised** (pandas/arrow x+1): +26% to +78% (fixed ~0.08-0.1 ms/batch IPC+serialise cost; arrow cheap 1 thread 71M vs 312M/276M rows/s).
* D overhead, **expensive vectorised** (regex): +2.6% to +11% at 1 thread, +8% to +20% at 8 threads, -4% to +12% at 32 (oversubscribed); i.e. straddles 10%.
* D vs A (GIL) at >=8 threads: D is 2-3.5x faster on cheap bodies and ~equal/slightly slower on expensive ones, because A serialises on the GIL. B (free-threaded) is the fastest option at >=8 threads for every kind.
* Crash: A/B/C lose the whole host (SIGSEGV); D loses one worker, ~223 ms respawn, host continues.
* Memory: in-process allocations are not attributable (tracemalloc misses Arrow pool/native); D workers are observable via /proc and capped via RLIMIT_AS/cgroups.
* C cannot import pyarrow/numpy/pandas -> scalar-only.

## Rule (binding)
Embedded = fastest safe in-process option. Server modes = isolation (D) unless its overhead > 10% for vectorised UDFs.

## Decision per mode
* **Embedded: B (free-threaded 3.14t, pinned worker threads, Arrow via PyCapsule zero-copy)**, falling back to A (GIL 3.14) when 3.14t or a UDF dependency is unavailable (a dependency re-enabling the GIL degrades B to A-like behaviour, still correct). At 1 thread A is up to ~12% faster on cheap bodies, but B is 2-9x faster at 8-32 threads. C rejected (no pyarrow/pandas).
* **Server modes: the rule's trigger is met as measured** - D overhead exceeds 10% for cheap vectorised UDFs in every cell (26-78%) and for roughly half of the expensive vectorised cells. Taken literally, the default is therefore in-process B, not D. **Flag for the controller:** that outcome gives up crash isolation (a UDF segfault kills the server) and per-UDF memory control, which this experiment shows only D provides. Recommended reading for the ADR: keep D as the supported *isolated* runtime selectable per UDF/tenant (overhead is a fixed per-batch cost, so it amortises with larger batches and expensive bodies), and use B as the default only for trusted/first-party UDFs. This is a recommendation beyond the literal rule; the literal rule result is B.

## Residual risks
* Numbers come from a shared, loaded 12-thread laptop; 32-thread cells are oversubscribed; cheap-body cells are noisy. Re-run on a dedicated box before locking the 10% threshold.
* D overhead was measured with a naive protocol (one copy into shm, StreamReader copy on result, 1:1 worker:thread, 8 MB regions); a shm-native Arrow allocator and larger batches would reduce it - not tested. Scalar-kind A/B were not in the full top-2 (smoke only).
* Free-threaded Python: ecosystem/thread-safety of third-party UDF libs unverified beyond pyarrow/pandas/numpy; unpinned `Python::attach` is a 100x trap on 3.14t (must pin).
* No subinterpreter support in pyarrow/numpy/pandas today; revisit when they adopt multi-phase init.
* Memory arbitration in-process remains blind to native allocations; only RSS (process-wide) is available.
* Arrow bridge uses `unsafe` (capsule pointer move-out); FFI-crate exemption per repo rules, needs SAFETY review before productising.
