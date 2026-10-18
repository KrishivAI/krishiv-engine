# krishiv-optimizer

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (batch: rule batches P2-T072a, join ordering P2-T073a), P4 (IVM pass).

## Purpose
Query optimizer: rule- and cost-based rewrites, including the IVM rewrite pass.

## Invariants
- Input and output are resolved plans with the same output attributes; the optimised plan is a fixed point.
- A rule returns its input `Arc` when it changes nothing (that is how a batch detects its fixed point) and
  rewrites a shared subtree once (`dag::Rewriter`), subquery plans included; never copy a shared subtree.
- Only deterministic predicates move; folding never changes what a query returns or raises (no overflow,
  no division, no time-zone-dependent cast; a raising call stays in the plan).
- Join ordering re-plans each inner-join tree (through deterministic filters and column selections) from
  `Statistics`; it never adds a cross product, places each conjunct at the lowest join that has its inputs
  (subquery conjuncts as filters), and restores column order only for positional consumers. Inputs are
  ordered canonically, so `optimize` stays a fixed point.
- Per-rule and TPC-H plan snapshots live in `tests/it/snapshots/`; differential tests run optimised and
  unoptimised plans through the reference interpreter in `tests/it/support/eval.rs`.
- Never changes a user's explicit `repartition(n)`.
- Rules are plan-snapshot tested; IVM pass satisfies Σ incremental == query(Σ inputs).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0012, 0017, 0020

## Test
`cargo nextest run -p krishiv-optimizer --no-tests=pass` · `cargo clippy -p krishiv-optimizer --all-targets -- -D warnings`
