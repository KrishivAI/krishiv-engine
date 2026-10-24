# krishiv-types

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).  
**Real code arrives:** P2.

## Purpose
Internal type representations: Spark data types mapped onto Arrow, schemas, scalar values, decimal/interval/timestamp rules.
The `Batch` newtype (`batch.rs`): Arrow `RecordBatch` + `Weights` (`Const(c)` | Int64 column); `ChangeBatch = Batch`.

## Key files
Paths are relative to this crate.
- `src/data_type.rs` Spark data types; `src/arrow.rs` mapping to Arrow
- `src/ddl.rs` type-string parsing used by `krishiv-sql`
- `src/batch.rs` the `Batch` newtype with `Weights` (`Const` or Int64 column)
- `src/error.rs` error type with Spark classes
- `tests/spark_type_strings.rs` type-string golden tests; `benches/batch_filter.rs` filter bench

## Extension points
- New Spark type: a variant in `src/data_type.rs`, its Arrow mapping in `src/arrow.rs`, and its string form in `src/ddl.rs`.

## Invariants
- Type mapping and precision/scale rules follow Spark exactly per supported version.
- 128-bit decimals with Spark precision/scale; binary-safe strings.
- `Batch`: no selection vector, no encoding enum; copy-on-filter. `Const(c)` never becomes a column except in
  `concat` of differing constants; operators route `Const(1)` to the no-weight path. A weight column has the row
  count and no nulls.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0009, 0008, 0006, 0017

## Tests
`cargo nextest run -p krishiv-types --no-tests=pass` · `cargo clippy -p krishiv-types --all-targets -- -D warnings`
