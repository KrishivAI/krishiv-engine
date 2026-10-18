# krishiv-functions

**Layer:** L2 (may depend only on crates in L2 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (core), coverage push in P3.

## Purpose
Spark-compatible function library: scalar, aggregate, window and generator functions with version-keyed semantics.

## Layout
- `scalar` (P2-T153): the execution binding of registry scalar functions. `bind_scalar(name, &[BindArg], &BindOptions)` →
  `Arc<dyn ScalarFunction>` (`return_type`, `invoke(&[Arg], rows)`), IR-free (Arrow types only) so the physical planner
  (krishiv-exec) calls it per batch. Arguments arrive after the analyzer's implicit casts; strings/binary in any Arrow
  carrier (read as `Utf8`/`Binary`), results in plain carriers. One family module per kernel family (`string`,
  `datetime`), each with `NAMES` and a `bind` over the argument shape. `BindOptions` holds ANSI, `KernelCtx` (Spark
  version), session zone and the query start instant. Operators, `CAST` and aggregates are not bound here.

## Invariants
- Behaviour keyed by Spark version (3.5, 4.0, 4.1, 4.2), never ad-hoc flags.
- Each function registered compatible / incompatible (opt-in `krishiv.expr.<name>.allowIncompatible`) / unsupported; no global switch.
- Golden tests from spark-semantics before implementation (skill `krishiv-spark-function`).
- Unselected rows never raise (CASE/IF/COALESCE, AND/OR short-circuit).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0008, 0009, 0010

## Test
`cargo nextest run -p krishiv-functions --no-tests=pass` · `cargo clippy -p krishiv-functions --all-targets -- -D warnings`
