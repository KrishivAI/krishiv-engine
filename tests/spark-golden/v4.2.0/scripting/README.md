# SQL scripting goldens (Spark 4.2.0, T196 unit 0)

Krishiv-authored; every expected value was produced by running `apache/spark:4.2.0` (`local[2]`, ANSI on, UTC session
time zone), never transcribed. Design: `krishiv-internal/design/P2-T196-sql-scripting.md` section 2.

- `semantics/*.sql` are the inputs, `semantics/*.json` the outputs (`schema` + `rows`, or `error`: `class`, `sqlstate`,
  `params`; a failure that is not a Spark error is recorded by its Java class). Format and the command:
  header of `tests/scripts/gen_scripting_semantics.py`. A case is a list of statements run in order in a fresh database
  with an empty scratch table `s(i INT)`. Random scope ids are normalised to `<scope>`, scratch database names to `<db>`.
  Files: `structural` (every section 2.5 row, each preceded by an `INSERT` to show the phase),
  `entry_results` (2.1), `variables` (2.2), `control_flow` (2.3), `handlers` (2.4).
- **Documented divergences** (cases with `-- KRISHIV-EXPECT:` lines; their JSON carries `divergence.krishiv_expected`, and Spark's
  behaviour sits under `spark_actual_not_expected`, which is not an expectation): `handler_in_if_body`, `condition_after_handler`
  (Spark crashes with `NoSuchElementException: None.get`; Krishiv raises the intended class) and `unaliased_column_name_over_local`
  (Spark names the column with a random scope id; Krishiv names it from the expression as written).
- `handler-record.json`: per script of `inputs/scripting/randomly_generated_scripts.sql` (95), on fresh copies of the six
  tables: `as_is` (schema, result rows, per-table `count`, `hash` = `sum(hash(*))`, `stable_hash` = same sum without TIMESTAMP columns),
  `fired` (error class, SQLSTATE and parameters Spark raises when the script's `DECLARE EXIT HANDLER` line is removed, or `"none"`),
  `without_handler_tables` and `deterministic` (false for scripts calling RAND/RANDOM or differing between two runs: compare
  nothing for those). The scripts insert `CURRENT_TIMESTAMP`, so compare `stable_hash`, not `hash`. Rows differ from the
  goldens in `results/scripting/` for scripts whose result depends on rows inserted by earlier scripts (the goldens run
  them in sequence on shared tables, this record on fresh tables).
- Phases and differences from the design note: `research/t196-scripting-phases.md`.

Version note: the run is Spark 4.2.0 only; scripting is new in 4.0 and the other versions were not run.

## Acceptance against Krishiv (`crates/krishiv-session/tests/scripting_goldens.rs`)

The 95 scripts run on fresh tables and are compared with `handler-record.json` (rows, per-table count and
checksum over the non-timestamp columns, and the error class that fires with the handler removed).
`known-failures.json` lists, with a hand-written reason each, the scripts that do not match yet; the test fails
for a new failure and for a listed script that now passes (`KRISHIV_RATCHET_UPDATE=1` rewrites the list and
keeps existing reasons). A script with a `FOR` over `DISTINCT`/`GROUP BY` and no `ORDER BY` depends on the
engine's row order, which differs between machines: it is compared and printed, never asserted.
`semantics/*.json` is checked by `tests/scripting_semantics.rs`.

## Triage of the `randomly_generated_scripts` golden remainders (opus-2, 2026-11-08, P2b row 15 unit R1)

The 7 statements still listed in `known-failures-cases/scripting/randomly_generated_scripts.json` (0-based 36, 61, 67,
69, 88, 103, 112) all fail on the result schema only: the script's `EXIT` handler fires in one engine and not in the other.
They run in sequence on shared tables, so a row inserted or deleted by an earlier script changes what a later one sees.
Six of the seven (36, 61, 67, 69, 88, 112) have a `FOR` query or scalar subquery with `LIMIT` and no `ORDER BY`, whose row
order is Spark's file order, not ours; scripts 3, 75 and 8 of the handler record (`known-failures.json`) are the
same effect with a proof (Spark's checksum is reproduced with an explicit `ORDER BY`). Statement 103 has no `LIMIT`; its
branches depend on the counts per region that earlier statements left behind. None is a scripting-engine defect; they
stay listed as order/state-dependent, and cannot be made to pass without Spark's file order.

## Exit handlers at the end of a body, and interrupts (opus-2, 2026-11-08, P2b row 15 R2 and R3)
- `semantics/handlers.sql` `exit_last_in_*` and `exit_*` cases (17, run on Spark 4.2.0): an `EXIT` handler whose compound is the
  last statement of an `IF`/`ELSEIF`/`ELSE` branch, a `WHILE` or `REPEAT` body or a searched `CASE` branch (also through
  compounds that end the same way) ends the whole script silently; a `FOR`, `LOOP` or simple `CASE` body, or a compound with
  statements after it, continues after the compound. The engine implements exactly this (`Body` in `scripting/mod.rs`).
- `semantics/control_flow.sql` `*_condition_text*`: `INVALID_BOOLEAN_STATEMENT` prints the condition's text in upper case.
- `interrupt/`: a Connect `Interrupt` into a running script ends it with `OPERATION_CANCELED` (`HY008`), between statements
  too, no handler fires and no later statement runs (`gen_scripting_interrupt.py`, `interrupt.out`).
