# krishiv-connectors-kafka

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P4.

## Purpose
Kafka source/sink connector (feature `kafka`, rdkafka).

## Invariants
- Behind an additive Cargo feature; exactly-once sinks per ADR 0016.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0016, 0028

## Test
`cargo nextest run -p krishiv-connectors-kafka --no-tests=pass` · `cargo clippy -p krishiv-connectors-kafka --all-targets -- -D warnings`
