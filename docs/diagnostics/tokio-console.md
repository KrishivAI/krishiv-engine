# tokio-console task diagnostics

When a query seems stuck but uses CPU, the cause is usually a driver that
does not yield: one poll runs too long on a compute worker. `tokio-console`
lists every live async task with its poll times, and Krishiv names its
tasks so a slow one points at its query, pipeline and driver. This is a
debugging aid only: it is off by default and changes nothing in default
builds.

## Enabling it

The `tokio-console` cargo feature (on `krishiv-cli` and `krishiv`,
forwarding to `krishiv-exec`) attaches `console-subscriber`. Task names
need `RUSTFLAGS="--cfg tokio_unstable"`: the feature still builds without
that cfg, but tasks stay unnamed. That cfg changes tokio itself, so build
into a separate target directory to keep default artifacts clean:

```sh
export CARGO_TARGET_DIR="$PWD/target/console" CARGO_BUILD_JOBS=3
export RUSTFLAGS="--cfg tokio_unstable"
cargo run -p krishiv-cli --features tokio-console -- sql -e 'SELECT ...'
```

Then point the console at the process:

```sh
KRISHIV_TOKIO_CONSOLE=1 cargo run -p krishiv-cli --features tokio-console -- sql -e 'SELECT ...'
```

With the variable set to `1`, the CLI (and every `Session` builder) installs
the subscriber at startup; with it unset, or without the feature, nothing
is installed. Connect with `tokio-console` (defaults to `http://127.0.0.1:6669`).

## What the task list shows

| Task name | What it is |
|---|---|
| `q<query id>/<pipeline>/driver<n>` | One pipeline driver; `<n>` is the driver's scheduler-assigned partition id. |
| `io/file-write` | A file-write writer draining one driver's batches. |
| `io/spill` | An output-spill hook spilling a finished breaker's unread output. |

The coordinator task stays unnamed. Sort by total poll time: a driver that
never yields shows one long poll where the rest show short ones. Warnings
(`waker missed`, `task long poll`) point at the same task.

## Walk-through: a planted slow operator

The `tokio-console-test` feature (implies `tokio-console`) adds
`testing::BusyLoop`, a pass-through operator that busy-loops 50 ms per
batch without yielding. It exists so the console has something to show;
it is compiled only with `tokio-console-test`, never in default builds:

```sh
export CARGO_TARGET_DIR="$PWD/target/console" RUSTFLAGS="--cfg tokio_unstable"
cargo nextest run -p krishiv-exec --features tokio-console-test a_busy_loop_counts_a_stall
```

The `a_busy_loop_counts_a_stall` test runs one batch through it and
asserts the driver's stall counter increments. In the console, the query's
`q1/0/driver0` task shows a ~50 ms poll while the other tasks stay idle:
that is what a non-yielding operator looks like live.

## The stall detector comes first

In production there is no console attached, so the driver's stall detector
is the first signal. It times every driver poll: a poll longer than the
stall threshold (10 ms by default) increments the driver's stall counter
(the profile JSON's `query.stalls`, `stalls` in a debug bundle). With
`fail_on_stall` (tests and CI), the query fails with `INTERNAL_ERROR`:

```text
a driver ran 52.3ms without yielding (stall threshold 10ms, batch size 4096)
```

The existing executor tests (`the_stall_detector_fails_a_non_yielding_driver`,
`the_stall_detector_fails_a_blocking_driver`) pin this behaviour. A fix
moves the work off the polling path: blocking or CPU-heavy work over ~1 ms
belongs on the bounded blocking pool (`TaskContext::run_blocking`), and a
batch that does too much per row should be chunked so the driver yields
between chunks.

## Cost

- Enabled, attached, idle query: added overhead per query to be recorded.
- Enabled, attached, running query: resident memory of the subscriber to be recorded.
- Disabled (any default or `dist` build): zero overhead and zero console
  symbols in the `krishiv` and `krishiv-cli` default feature trees, checked by
  `cargo xtask tokio-console-check` (wired into `cargo xtask precheck`). The subscriber accounts none of its memory to
  the query `MemoryPool`: it runs on its own runtime in a background
  thread, outside every query's pool.
