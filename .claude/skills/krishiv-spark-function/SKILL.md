---
name: krishiv-spark-function
description: Use when adding or changing a Spark SQL/DataFrame function or expression in Krishiv (scalar, aggregate, window, generator, cast, operator expression like +, =, LIKE). Golden-first, so Spark behaviour is established from apache/spark source and a real Spark run before any code.
---

# Add a Spark function (ADR 0008, 0009, 0010, 0029)

Read by path: `../krishiv-internal/adr/0008-spark-semantics-layer.md`, `0009-internal-type-representations.md`,
`0010-spark-observable-hashing.md` (hash functions), `0029-code-quality-and-testing.md`;
`../krishiv-internal/spec/04-spark-compatibility.md`; `crates/krishiv-functions/CLAUDE.md`, `crates/krishiv-kernels/CLAUDE.md`.

## Checklist
1. **Scope.** Name, Spark class (e.g. `org.apache.spark.sql.catalyst.expressions.Substring`), signatures, which
   Spark versions have it (3.5, 4.0, 4.1, 4.2). Batch same-shape functions (e.g. a family of string functions) into
   one task; keep one crate per task.
2. **Ground truth first — spark-semantics agent** (`.claude/agents/spark-semantics.md`): source at each tag +
   real Spark 4.2 run → `tests/spark-golden/functions/<name>/` (committed; not under a `v<N>/` dir, which is gitignored except `v4.2.0`) with per-version expected outputs and a README
   citing tag/file/line. Never from memory. Include: ANSI on and off, NULLs, NaN/-0.0/±Inf, decimal precision/overflow,
   empty and non-UTF-8 strings, Unicode, timezone/DST, regex dialect, error class + SQLSTATE + message.
3. **Status.** Register as `compatible`, `incompatible` (documented diff; opt-in via
   `krishiv.expr.<name>.allowIncompatible=true`; no global switch) or `unsupported` (clear error).
4. **Failing tests.** Golden tests (sqllogictest / SQLQueryTestSuite file) + unit tests + `proptest` for algebraic
   properties (e.g. `upper(upper(x)) == upper(x)`); run them, see them fail.
5. **Implement** in `krishiv-functions` (semantics, type coercion, version keying — `match spark_version`, not flags)
   on top of `krishiv-kernels` (vectorised; generics inside kernels). Selection vectors respected: no errors from
   unselected rows (CASE/IF/COALESCE, AND/OR short-circuit). Spark float ordering in every compare/hash/sort.
   Errors: typed, with the error class from `crates/krishiv-common/error-classes.json` (add new classes there, copied
   from Spark's `error-conditions.json` at the tag).
6. **Kernel specialisations** (nullable/non-null, dictionary, Utf8 vs Utf8View, scalar-vs-array): each one tested.
7. **Verify:** `cargo nextest run -p krishiv-functions -p krishiv-kernels`, clippy `-D warnings`, `cargo xtask layering`,
   `cargo xtask docs-errors` if error classes changed. Micro-bench with criterion if the function is hot.
8. **Review:** reviewer agent; stage 1 checks goldens cover every version and edge listed above.
9. Update the function-coverage list/compat matrix (once it exists) and the plan checkbox.

## Never
Invent behaviour, "improve" on Spark silently, add a global incompatibility switch, or pull in a crate for one
function (own code preferred — ADR 0028; Java regex via translation + `fancy-regex`, datetime patterns in-house).
