# krishiv-optimizer

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (batch: rule batches P2-T072a, join ordering P2-T073a), P4 (IVM pass).

## Purpose
Query optimizer: rule- and cost-based rewrites, including the IVM rewrite pass.

## Key files
Paths are relative to this crate.
- `src/lib.rs` rule batches and the `optimize` entry point
- `src/dag.rs` `dag::Rewriter`, the rewrite-once rule for shared subtrees
- `src/constraints.rs` `InferJoinFilters` (P2-T072b unit 3): `isnotnull` on nullable `=` keys and literal comparisons moved across `=` keys of one type, on the sides Spark's `InferFiltersFromConstraints` filters (no substitution; differences in `docs/known-differences.md`); runs after the rewrite batch (then the batch again when it added) and inside `order_and_restrict`
- `src/fold.rs` constant folding; `src/pushdown.rs` predicate pushdown (below a window only on partition keys); `src/prune.rs` column pruning (removes unused window columns)
- `src/project.rs` projection rules; `src/repartition.rs` Spark's `CollapseRepartition` (a `DISTRIBUTE BY` directly over another drops the inner one; filters move below one in `pushdown.rs`, pruning keeps the columns its expressions read); `src/limit.rs` limit rules; `src/group_limit.rs` Spark's
  `InferWindowGroupLimit` decision (applied by the physical planner)
- `src/restrict.rs` subquery restriction rules
- `src/inline_cte.rs` CTE inlining (a nondeterministic definition read twice stays a `CteRef`); `src/share_cte.rs` `ShareCte` (P2-T198): equal deterministic instances with a pipeline breaker become one `WITH` definition, only when a session configuration is given and `krishiv.sql.optimizer.shareCteInstances` is on; with `krishiv.optimizer.shareRepeatedSubtrees` (off) a subtree the main block reads directly joins the group (P3 W1 shape (b))
- `src/win_magic.rs` `SubqueryAsWindow` (P3 W1 shape (a), form M): a decorrelated `min`/`max`/decimal `avg` subquery over the outer block's own relations becomes a `Window` over the outer rows (`krishiv.optimizer.subqueryAsWindow`, off; q17, q02; refusals in `tests/it/win_magic.rs`; guarded by estimated window bytes, priced on the join-ordered outer block, <= second-scan bytes; no firing without statistics)
- `src/eager.rs` `EagerAggregate` (P2-T167): an aggregate's calls run below a join and merge above it, gated on statistics; `krishiv.optimizer.eagerAggregation`, `Options::eager_aggregation`
- `src/group_code.rs` `GroupKeyCode` (design P2b D2): an aggregate's dimension group keys travel as one integer code (`row_number` over the dimension's distinct keys, read twice as `Shift`s of one body; `D_c` is a `Shift` of offset 0, the fence join ordering does not look through), decoded by a join above the aggregate; statistics-gated (rows(I) read through a null-aware `NOT IN` anti join only, `gate_rows`), one dimension per aggregate, `D`'s filters may not raise; a decode join it adds is ordered by `order_and_restrict` at once; `krishiv.optimizer.groupKeyCode`, `Options::group_key_code`; tests `tests/it/group_code.rs`, executor differential `krishiv-exec` `tests/planner/group_code.rs`
- `src/join_order.rs` inner-join ordering (estimates the whole plan first, so a `CteRef` in a tree reads its `WITH` definition; enumerator choice `krishiv.optimizer.joinEnumerator` / `joinDpLimit`, a `TreeStats` note per tree in `EXPLAIN VERBOSE`); `src/join_order/dp.rs` the exact enumerator DPsub over a hypergraph of input masks (P2-T113 kept it over DPhyp: `exp/t113-dphyp`); `src/join_order/reads.rs` which columns are read above a tree, for `krishiv.optimizer.joinCost=bytes` (E5b: a join costs its result rows plus its build's bytes, the fewer-rows input stays the build within 2x of the other's bytes; default `rows`, `tests/it/join_cost.rs`); `src/stats.rs` `Statistics` it reads and the estimate types (`RelStats`, `AttrStats`, `StatSource`)
- `src/estimate.rs` the `Estimator`: what each operator leaves known of its columns (a column a filter or join does not read keeps the distinct values the kept rows are expected to hold, `kept_distinct`, Cardenas); `src/narrow.rs` predicates over estimates (selectivity and narrowed ranges); `src/plan_stats.rs` `PlanStats`, one pass over an optimized plan (physical planner, `EXPLAIN COST`)
- `tests/it/plan_stats.rs` one test per propagation rule; `tests/it/qerror.rs` the q-error report (`cargo xtask qerror`)
- `tests/it/rules.rs` per-rule snapshots; `tests/it/join_order.rs` ordering snapshots; `tests/it/join_enumerator.rs` enumerator settings over TPC-H/TPC-DS; `tests/it/support/eval.rs` reference interpreter

## Extension points
- New rule: a module in `src/`, added to a batch in `src/lib.rs`, with snapshot cases in `tests/it/rules.rs` and a differential check through `tests/it/support/eval.rs`.

## Invariants
- Input and output are resolved plans with the same output attributes; the optimised plan is a fixed point.
- A rule returns its input `Arc` when it changes nothing (that is how a batch detects its fixed point) and
  rewrites a shared subtree once (`dag::Rewriter`), subquery plans included; never copy a shared subtree.
- Inferred join filters keep `optimize` a fixed point: a side that already implies a candidate (`constraints::held`: through aliases, `cse_*` projections, `Shift`, positional unions, grouped aggregates, joins, row-preserving nodes; a literal `false`) gets nothing, and a side with nothing to add is returned as it is.
- Only deterministic predicates move; folding never changes what a query returns or raises (no overflow,
  no division, no time-zone-dependent cast; a raising call stays in the plan).
- `SUM(x + c)` of a narrow integral column becomes `SUM(x) + c * COUNT(x)` (`sum_rewrite.rs`, one pass after the
  rewrite batch) only where the type range proves no per-row overflow and `|x|, |c| <= 2^16`; floating point and
  decimals are never rewritten. Conditions and the one residual assumption (fewer than 2^46 rows per aggregate)
  are in the module docs; the executor differential test is `krishiv-exec` `tests/planner/sum_rewrite.rs`.
- Eager aggregation moves only calls whose result cannot depend on row order and whose arguments cannot raise (`eager.rs` module docs); it reads statistics only, runs after join ordering, and states each decision in `EXPLAIN VERBOSE`. The executor differential is `krishiv-exec` `tests/planner/eager_aggregate.rs`.
- Join ordering re-plans each inner-join tree (through deterministic filters and column selections) from
  `Statistics`; it never adds a cross product, places each conjunct at the lowest join that has its inputs
  (subquery conjuncts as filters), and restores column order only for positional consumers. Inputs are
  ordered canonically, so `optimize` stays a fixed point. An input's own conjuncts are estimated over one view in a canonical
  order (`narrow::conjunct_selectivities`), so a two-sided range keeps the fraction between its bounds (E5a).
- Column equalities `a = b` of one type between join-tree inputs form equivalence classes (`join_order.rs`
  `canonical_equalities`): inputs with columns of one class can join on it even when the query relates them only
  through a third input, and a join or an estimate takes only the equalities that connect columns of a class not
  already equal below it (`Graph::independent`), chosen canonically (a composite key's members first, its
  selectivity on one canonical member, then column-id order), so the plan does not depend on how the class was
  written.
- Restriction (`restrict.rs`): an aggregate joined on its group keys may be reduced by a semi join with the other
  side's key domain; the domain follows inner-join `o = y` equalities (same type, never through an outer join) to the
  subtree that scans fewest rows, so where join ordering puts the aggregate does not decide it.
- Statistics propagation (`PlanStats`): a filter narrows `min`/`max` of the columns it reads and clears their null
  fraction when null-rejecting; `OR` takes the union of its branches' ranges, `NOT` and unknown predicates leave
  ranges alone, so a range only stays or widens (the proptest in `plan_stats.rs` holds it to that). An equality with
  a literal outside `[min, max]` estimates 0 rows and removes nothing. A join's keys take the smaller distinct count
  (inner: the ranges intersect, no nulls); an outer join's null-supplying side gains the unmatched fraction as nulls.
  `col = col` keeps `1 / max(NDV)` with an unknown count read as the rows (T166 note, section 2); `col = <scalar subquery>` (reading no column, deterministic) reads an unknown count as the rows too and leaves one value (study 26, E10: q15's `= (SELECT max(..))` over an aggregate output), and `col IN (..)` whose elements read no column, are deterministic and include a subquery leaves `k` values the same way (`k` counts a repeated literal once; a known count is clamped to `k` too); only `col = literal` over an unknown count takes the default selectivity. A count from `StatSource::Sampled` (P2-T164 U6: the session samples file columns an equality or `IN` reads before optimizing) is known like any other. Statistics never change results;
  `Options::stats_propagation` switches the propagation off for the differential tests.
- Per-rule and TPC-H plan snapshots live in `tests/it/snapshots/`; differential tests run optimised and
  unoptimised plans through the reference interpreter in `tests/it/support/eval.rs`.
- Never changes a user's explicit `repartition(n)`.
- Rules are plan-snapshot tested; IVM pass satisfies Σ incremental == query(Σ inputs).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0012, 0017, 0020

## Tests
`cargo nextest run -p krishiv-optimizer --no-tests=pass` · `cargo clippy -p krishiv-optimizer --all-targets -- -D warnings`
