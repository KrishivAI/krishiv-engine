# is_valid_variant, schema_of_variant_agg, try_reflect (+ variant edge cases) goldens (P3-T088 step 1)

Status target: `compatible` for every rule below; the "Quirks" list is Spark behaviour that looks like a bug
(recorded, not endorsed; `incompatible` candidates, opt-in via `krishiv.expr.<name>.allowIncompatible`).

Layout: `inputs/<name>.sql` (`;`-terminated statements, one session, so SET and views carry over),
`inputs/malformed.setup.py` (builds view `mv` of raw-byte variants through the Python API; exec'd by `../run_oracle.py`
before `malformed.sql`), `results/spark-<ver>/<name>.sql.out` (output of a real run), `run.sh <ver> [name]`.
Every expected value was produced by executing Spark; none is transcribed. Hand-written expectations: none.

## How the .out files were made

`apache/spark:<ver>` container `krishiv-s7pr34-oracle-<ver>` (`--rm`), `spark-submit ../run_oracle.py`, session time
zone UTC unless the file sets it, ANSI default (on). No extra packages needed. Runs: **4.2.0 and 4.3.0** (the only
versions where these functions exist; source read at tags `v4.2.0` and `v4.3.0`). Not run on 3.5.7/4.0.0/4.1.0: the
functions are absent there (`is_valid_variant` is "Since 4.2.0"; `try_reflect` and `schema_of_variant_agg` are registered
in `FunctionRegistry.scala` for the 4.x line; the older tags were not run, so 3.5/4.0/4.1 behaviour is `unverified-by-run`).
Executor host ids in `SparkException` "Job aborted" messages differ run to run: compare class/cause, not that text.
The `.out` files are root-owned (written from the container).

## Rules and sources (v4.2.0; v4.3.0 identical except `describe function extended` gains an `Arguments:` block)

Source files: `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/variant/variantExpressions.scala`
(IsValidVariant, SchemaOfVariant, SchemaOfVariantAgg), `.../variant/VariantExpressionEvalUtils.scala` (isValidVariant),
`.../CallMethodViaReflection.scala` (reflect/TryReflect), `.../analysis/FunctionRegistry.scala` (registration). Source line
numbers were not obtainable (files fetched without numbering); each rule cites the declaration name.

| Rule | Source | Case file |
|---|---|---|
| `is_valid_variant(v)`: BOOLEAN; true for valid bytes, false for malformed, NULL for SQL NULL. Variant null (`parse_json('null')`) is valid -> true. Registered `expression[IsValidVariant]("is_valid_variant")`; "Since 4.2.0" | `IsValidVariant` (RuntimeReplaceable -> `VariantExpressionEvalUtils.isValidVariant`); `describe function extended` | is_valid_variant, malformed |
| Validity is structural (`VariantUtil.isValidVariant(value, metadata)`): truncated value, unknown header, bad metadata version, empty value/metadata, out-of-range field id => false; trailing bytes after a complete value => still true | run (rows 1-8 of `mv`) | malformed |
| Arity exactly 1 (`WRONG_NUM_ARGS.WITHOUT_SUGGESTION`, 42605); non-VARIANT input (INT, STRING, BINARY) => `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` (42K09), no implicit cast from string | `ExpectsInputTypes(Seq(VariantType))`; run | is_valid_variant |
| `cast(binary as variant)` is legal and yields a BINARY variant (always valid) | run | is_valid_variant |
| Bad JSON in `parse_json` raises `MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION` (22023) before the function sees it; `try_parse_json` gives NULL => result NULL; `parse_json('"\u0000"')` is a Jackson control-char error | run | is_valid_variant |
| ANSI on/off, session time zone: no effect | run | is_valid_variant |
| Conditional evaluation (CASE/IF/COALESCE/WHERE with AND false) never raises on unselected rows | run | is_valid_variant, schema_of_variant_agg, try_reflect |
| `schema_of_variant_agg(v)`: STRING, "Since 4.0.0", never NULL (`nullable=false`): empty input, all-NULL input, filter-eliminated input and all variant-null input give `VOID`; SQL-NULL rows are skipped; `parse_json('null')` merges as VOID (`null` + 1 = BIGINT) | `SchemaOfVariantAgg` (`createAggregationBuffer = NullType`, `update` skips null, `eval = printSchema(buffer)`) | schema_of_variant_agg |
| Merge uses `JsonInferSchema.compatibleType(t1, t2, VariantType)`: int+decimal => `DECIMAL(21,1)`-style widening; int+string, int+bool, array+object => `VARIANT`; objects merge fields by name, sorted; arrays merge element types (`ARRAY<VOID>` for `[]`); `1e400`/oversize ints => DOUBLE when mixed with decimals; `DECIMAL(38,0)` caps | `SchemaOfVariant.mergeSchema`; run | schema_of_variant_agg |
| Printing: `OBJECT<...>`, `ARRAY<...>`, field names via `QuotingUtils.quoteIfNeeded` (backticks, embedded backtick doubled) | `SchemaOfVariant.printSchema` | schema_of_variant_agg |
| Works as group-by aggregate, `FILTER`, window with `PARTITION BY` and a running `ORDER BY` frame (quirk 2), `HAVING` | run | schema_of_variant_agg |
| Errors: INT/STRING input `UNEXPECTED_INPUT_TYPE`; arity `WRONG_NUM_ARGS.WITHOUT_SUGGESTION`; malformed bytes => `MALFORMED_VARIANT` (22023, `SparkRuntimeException`, empty parameters) | `schemaOf` throws `MALFORMED_VARIANT` on unsorted object fields | schema_of_variant_agg, malformed |
| `schema_of_variant(v)`: NULL for SQL NULL, `VOID` for variant null, `ARRAY<VARIANT>` for mixed arrays; leniency: a truncated int8 variant still reports `BIGINT` while `is_valid_variant` is false and `variant_get`/`to_json`/`cast` raise `MALFORMED_VARIANT` | `SchemaOfVariant.schemaOfVariant`; run | schema_of_variant_agg, malformed |
| `try_reflect(class, method[, args...])` = `reflect` with `failOnError=false`: STRING result; any `NonFatal` invocation exception (NumberFormatException, ArithmeticException, NPE on NULL arg to primitive param) => NULL. `reflect` with the same input raises a `SparkException` (job aborted) with `InvocationTargetException` cause. "Since" 4.0.0 per registry, usage text in `.out` | `CallMethodViaReflection` (`failOnError`, `case NonFatal(_) if !failOnError => null`); `FunctionRegistry` `expression[TryReflect]("try_reflect")` in the try_* section | try_reflect |
| try_reflect does NOT swallow analysis errors: unknown class `UNEXPECTED_CLASS_TYPE`, unknown/instance method or non-matching arg types `UNEXPECTED_STATIC_METHOD`, non-foldable class/method `NON_FOLDABLE_INPUT`, NULL class/method `UNEXPECTED_NULL`, array/decimal/timestamp arg `UNEXPECTED_INPUT_TYPE` (allowed: boolean, tinyint..bigint, float, double, string), arity < 2 `WRONG_NUM_ARGS` (expected `> 1`); all error messages name `reflect`, not `try_reflect` | `CallMethodViaReflection.checkInputDataTypes`; run | try_reflect |
| Non-deterministic: not allowed inside aggregates (`AGGREGATE_FUNCTION_WITH_NONDETERMINISTIC_EXPRESSION`, 42845); in a CASE the else-branch `reflect` result is widened with the literal type (`bigint`) | run | try_reflect |
| String args pass UTF-8 through; invalid UTF-8 (`cast(X'ff' as string)`) comes back as U+FFFD; `Integer.parseInt(' 12')` => NULL (no trimming); ANSI on/off and time zone make no difference | run | try_reflect |
| `variant_get`/`try_variant_get` edge cases (cheap subset): wrong-type => `INVALID_VARIANT_CAST` (22023, "use try_variant_get") regardless of ANSI; bad path `a` or `$[-1]` => `INVALID_VARIANT_GET_PATH` (22023); missing path/index => NULL; unknown type string => `UNSUPPORTED_DATATYPE` (0A000); `-0.0` -> `0.0` rendering, `"NaN"` string -> NaN; decimal rounds half up (1.55 -> 1.6) | run | variant_edge |

## Per-version differences

- 4.2.0 vs 4.3.0: no behavioural difference in any of the five files (diff = executor hostnames in job-abort
  messages, plus `describe function extended` for `is_valid_variant`, `schema_of_variant_agg`, `try_reflect` gaining an
  `Arguments:` section in 4.3.0). Function-description text is the only observable change.
- 3.5.x / 4.0.x / 4.1.x: not run; expected `UNRESOLVED_ROUTINE` for `is_valid_variant` (new in 4.2) and presumably for
  `try_reflect` where unregistered. Verify with a run before implementing per-version gating.

## Quirks to decide (Spark behaviour that looks like a bug)

1. `schema_of_variant_agg(distinct ...)` dies with `scala.MatchError` on `VariantVal` (job aborted) in 4.2/4.3.
2. (not a bug, noted) with `over (order by k)` the running frame at k=2 merges objects with an array, hence `VARIANT`; the all-NULL group k=3 still shows the merged prefix.
3. `try_reflect` error messages and analysis errors say `reflect` (the expression's `prettyName`).
4. `try_reflect` is non-deterministic so `count(try_reflect(...))` fails analysis; group-by on it works.
5. `group by ()` is a parse error in Spark SQL (recorded in schema_of_variant_agg.sql.out).

## Not covered / open

- 3.5.7, 4.0.0, 4.1.0 were not run (see above). `try_reflect` since-version was not independently confirmed beyond registry+usage text.
- Malformed bytes beyond the 8 constructed rows (corrupt dictionary offsets, unsorted object field ids at depth) are not enumerated.
- `is_valid_variant` over Parquet-shredded variants is out of scope.
