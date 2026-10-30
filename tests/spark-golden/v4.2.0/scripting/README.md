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
