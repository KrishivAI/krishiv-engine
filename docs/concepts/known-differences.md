---
description: "Where Krishiv behaviour deliberately differs from Spark."
---

<!-- SPDX-License-Identifier: Apache-2.0 -->
# Known differences from Spark

Deliberate, documented divergences of Krishiv's engine from Apache Spark.
Each divergence is listed here next to the code that implements it.

## Error ordering between two raising expressions

When two expressions evaluated for the same row would both raise (for
example an ANSI division by zero in each of two select items), Spark does
not specify which error is raised first. Common-subexpression elimination
(`krishiv-optimizer`'s `cse` pass) evaluates a surely-evaluated
subexpression once in a projection below its operator, so the row raises at
the shared computation rather than at whichever use Spark happened to
evaluate first. When one row would raise from several expressions, the
error reported can differ in class as well as in the blamed expression (for
example `DIVIDE_BY_ZERO` against `ARITHMETIC_OVERFLOW`); a query that raises
in one engine still raises in the other.

## Spark 3.5 error-message parameters follow 4.x

The parameters inside an error message follow Spark 4.x, also for a Spark
3.5 client. For example, `SCHEMA_NOT_FOUND` names the schema with its catalog
(`` `spark_catalog`.`dummy` ``) where Spark 3.5 names it without
(`` `dummy` ``).

## Result cache, reproducible results (Krishiv extensions, off by default)

`krishiv.resultCache.enabled` and `krishiv.deterministicOrder` add behaviour Spark does not have; with
both off (the default) nothing differs. See [result cache](../reference/result-cache.md) for
what is cached, when entries go, and the one staleness the cache accepts (a change made outside the
engine, until the entry expires or `REFRESH TABLE`).

## `EXPLAIN FORMAT JSON`, `EXPLAIN ANALYZE`, `EXPLAIN VERBOSE` (Krishiv extensions)

Spark 4.2 has none of these forms, and its parser rejects each with a syntax error. Krishiv accepts them:
`EXPLAIN FORMAT JSON <statement>` returns the statement's analysis as JSON without running it (see the
[SQL reference](../reference/sql/index.md)); any other `EXPLAIN FORMAT ...` is a syntax error, as in Spark.

## Writers leave a `_krishiv_log` directory

A write to a plain directory records each commit in `<path>/_krishiv_log/` (one `<version>.json` per commit), so that
concurrent writers cannot both take a version and Krishiv readers see only committed files. Spark commits by
writing files under their final names plus `_SUCCESS`. The underscore prefix hides the directory from Krishiv and
Spark readers, but other tools that list the directory see it. `krishiv.sql.files.commitProtocol=NONE` selects
Spark's plain commit for a new directory; a directory that already has a log keeps being written through it, so
do not delete the log from a table you still append to.

## An unset `spark.sql.warehouse.dir` has no value

`spark.conf.get` of a Spark SQL config Krishiv does not use answers the default of the session profile's Spark
release, as Spark does. Spark computes the warehouse directory's default from its host (a path under its working
directory); unset, Krishiv answers no value for it (`<undefined>` in `SET`) and keeps managed tables in its own
default warehouse. Set it when the session is built (it is a static key) to read it back.

## A malformed value of a Spark config key fails at `SET`

A Spark key that Krishiv maps to a typed setting (for example
`spark.sql.autoBroadcastJoinThreshold`) is checked when it is set: a value
that does not parse fails the `SET` with `KRISHIV_INVALID_CONFIG_VALUE`.
Spark stores the string and fails when a query reads it, with
`INVALID_CONF_VALUE.TYPE_MISMATCH` (SQLSTATE 22022).

## `DISTRIBUTE BY` under a `LIMIT`, and partition numbers

`DISTRIBUTE BY exprs` (Spark's `RepartitionByExpression`, also under `CLUSTER BY` and `SORT BY ...
DISTRIBUTE BY`) places a hash exchange on the expressions' values, so they are evaluated, and raise,
on every row as Spark's shuffle evaluates them. Two things differ:

- On one worker the exchange runs inside each driver's pipeline, so a `LIMIT` above it stops reading
  once it has its rows, and a value in a batch that was never read is never evaluated:
  `SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) LIMIT 1` can return a row where
  Spark's shuffle first evaluates every row and raises `CAST_INVALID_INPUT`. Do not count on it to
  skip a bad value. The expressions are evaluated a batch at a time and several batches are read at
  once, so a bad value in any batch that was read raises as in Spark: with the bad value in the last
  row of a one-file Parquet table, a small table raises and a large one can return a row. A
  distributed run's map stage reads every row, so there it always raises as Spark does.
- Rows go to partitions by Krishiv's routing hash, not Spark's `pmod(murmur3(exprs), n)`, so what
  depends on the partition numbers (`spark_partition_id()`) can differ.

Pinned by `crates/krishiv-session/tests/distribute_by_goldens.rs` (`DIFFERENT`), against a Spark
4.2.0 run (3.5, 4.0 and 4.1 not run).

## `TABLESAMPLE` rows follow the input's partitions

`TABLESAMPLE (x PERCENT)` and `(BUCKET x OUT OF y)` sample as Spark's `SampleExec` does: each row of
input partition `i` takes one draw of `XORShiftRandom(seed + i)`, in input order, and is kept when it
falls below the fraction. The rows therefore equal Spark's when both engines split the input the same
way, and are otherwise a different Bernoulli sample at the same fraction:

- A session with one compute thread reads `range(0, n)`, `VALUES` and a small file as one partition, as
  Spark's `local[1]` does; `range(a, b, step, n)` is the same `n` slices in both engines at any thread
  count; `UNION ALL` numbers its inputs' partitions one after another, as Spark.
- An in-memory table (registered batches, a cached relation) is one partition per 128 Ki rows
  (`morsel_rows`), even at one thread: a larger one draws from several streams where Spark's
  `local[1]` draws from one.
- A file scan under a sample reads Spark's file partitions (the same split planning:
  `maxPartitionBytes`, `openCostInBytes`, the leaf parallelism) each whole, by one driver, in row order.
  So `REPEATABLE` gives the same rows on every run at any thread count; they equal Spark's when the
  files hold the same rows in the same order and the same sizes (files Krishiv writes differ in size
  from Spark's, which can pack them differently).
- With more threads, `range(0, n)` is cut into one slice per thread (Spark: `defaultParallelism`), so
  the rows differ from a Spark cluster's of another parallelism; the same session repeats them.
- After a join, an aggregate or an exchange the input partition is the driver's, and its rows' order
  depends on scheduling: as in Spark, a sample there is not repeatable.
- Without `REPEATABLE` the seed is a random value in `0..1000`, drawn when the plan is built. One
  DataFrame keeps its seeds across actions (`df.collect()` twice gives the same rows, as a Spark
  Dataset keeps its executed plan); each new statement draws again. Every reference of a `WITH`
  definition holding one samples with its own seed, as Spark's inlined references do.
- Inside a recursive CTE a sample draws from the stream of its iteration's input partition, without
  Spark's per-round partition shift, so its rows there differ from Spark's.
- `TABLESAMPLE SYSTEM (...)` (Spark 4.2's block sampling) is `KRISHIV_UNSUPPORTED_FEATURE`, and a
  `REPEATABLE` seed past `BIGINT` is refused as unsupported where Spark raises an unclassified
  `NumberFormatException`. A sampled plan travels through the plan codec (feature `sample`); a
  worker without that feature refuses it.

Pinned by `crates/krishiv-session/tests/tablesample_goldens.rs` against a Spark 4.2.0 run
(`tests/fixtures/spark-4.2.0/tablesample.json`).

## `histogram_numeric` bins depend on where partial histograms merge

`histogram_numeric(col, nb)` is Spark's `NumericHistogram` (add, merge, trim, the `java.util.Random(31183)`
tie-break, the same serialized partial state). Its bins depend on the order rows are added and on where partial
histograms merge, as Spark's do. The results equal Spark's when both engines split the input the same way, and
can differ by a bin or its counts otherwise:

- A local `VALUES` relation is one partition here; Spark's `local` run splits it into two. Merging two halves
  reproduces Spark's bins, a single pass does not (the 50-row case of `group-by.sql`).
- Krishiv merges at its own points: several drivers per partition, flushes of full partial tables, the adaptive
  bypass of a partial aggregate and spills. A merge of two partial histograms does not combine equal `x` values,
  so a bypassed two-phase aggregate over the rows `(1, 4), (1, 4)` gives two bins `{x: 4, y: 1}` where Spark gives
  one bin `{x: 4, y: 2}`.
- Not covered: `spark.sql.legacy.histogramNumericPropagateInputType=false`, TIME input, window use (refused).

Pinned by `crates/krishiv-session/tests/aggregate_histogram_numeric.rs` and the golden runner (`group-by.sql`).

## `SELECT *` next to `window(..)` or `session_window(..)`

`SELECT *, window(ts, '5 minutes') FROM t` returns the window column once, as Spark does, also with a
`QUALIFY` over a window. Krishiv adds the window column to the input before the star expands, where Spark
expands the star first, so it leaves the added column out of a bare `*`. Three cases differ:

- A table with a column named `window` (or `session_window`) next to a window call fails with
  `AMBIGUOUS_REFERENCE`.
- `SELECT * EXCEPT (window), window(..)` fails with `EXCEPT_OVERLAPPING_COLUMNS` where Spark answers
  `UNRESOLVED_COLUMN`.
- A `*` next to a window call in a `WHERE` or `ORDER BY`, and `struct(*)` next to one, are not pinned against a
  Spark run; the star may include a trailing `window` column.

## `session_window` over a string time column

`session_window(s, '5 minutes')` over a string column returns `start` and `end` as strings, with the
right values. Spark casts the column to `TIMESTAMP` first, so its fields are timestamps. Pinned (ignored) by
`crates/krishiv-session/tests/time_window_oracle.rs`.

## Filters move below only their deterministic prefix

Spark's predicate pushdown moves every deterministic conjunct of a filter below a node it can pass
(`splitConjunctivePredicates(condition).partition(_.deterministic)`), wherever the conjunct stands.
Krishiv moves only the conjuncts before the first nondeterministic one: in `WHERE rand() < 2 AND
x <> 'x'` over a `DISTRIBUTE BY CAST(x AS INT)`, Spark filters `x <> 'x'` before the shuffle and
returns rows, Krishiv keeps it above and raises `CAST_INVALID_INPUT`. Results differ only where a
conjunct after a nondeterministic one would have kept a raising expression from a row. Pinned by
`distribute_by_goldens.rs` (`DIFFERENT`).

## PySpark's `VariantVal`

Krishiv's Python `VariantVal` (`krishiv.sql.variant`) decodes a value as Spark's JVM `Variant` and `VariantUtil` do, so
`toJson` matches SQL `to_json`. PySpark 4.2.0's own `VariantVal` (`pyspark.sql.variant_utils`) is a separate decoder
and differs from it, and so from Krishiv, in these cases (checked on PySpark 4.2.0):

- A UUID (primitive type 20) decodes to a `uuid.UUID` and prints as its text; PySpark raises `MALFORMED_VARIANT`.
- Two values with the same bytes are equal and hash alike; PySpark compares by identity.
- A metadata header with a version other than 1 raises `MALFORMED_VARIANT` even for a primitive value; PySpark reads
  the primitive (`'true'`).
- An unknown primitive type (21 and up) raises `UNKNOWN_PRIMITIVE_TYPE_IN_VARIANT`, as the JVM does; PySpark raises
  `MALFORMED_VARIANT`.
- A double prints as Java does (`1.0E10`, as `to_json`); PySpark prints Python's form (`10000000000.0`). A
  timestamp's fraction drops trailing zeros as `to_json` does (`00:00:01.5`); PySpark prints six digits (`.500000`).
- Errors are `ValueError` whose message starts with the error class; PySpark raises `PySparkValueError` carrying the
  condition.

## `mapInPandas` and `mapInArrow` results unlike the declared schema

With `spark.sql.execution.arrow.pyspark.validateSchema.enabled` set (Spark 4.1 and later), Krishiv checks each
result batch as Spark does and raises `ARROW_TYPE_MISMATCH` with Spark's message. With it off (the default, and on
Spark 3.5 and 4.0, which have no such setting), Spark does not check the result's types and reads the batch as if it
had the declared ones. Krishiv refuses such a result instead, and the error depends on where the UDF runs:

- **Server mode (the Python worker pool).** `PYTHON_EXCEPTION` ("The Python worker returned an invalid result"),
  also for a null in a column declared non-nullable.
- **Embedded (`krishiv.session()`).** A `mapInArrow` column of another type is `KRISHIV_INVALID_ARROW_DATA`
  (SQLSTATE 22000), which names the returned and the declared type. `mapInPandas` converts each column to its
  declared type as PySpark's worker does (a `long` column declared `int` is returned as `int`) and raises
  `PYTHON_EXCEPTION` with the conversion's traceback when it cannot (a `long` column declared `string`). A null in
  a column declared `NOT NULL` fails with `PYTHON_EXCEPTION`: see the last paragraph of
  [Several grouped-aggregate pandas UDFs in one `agg`](#several-grouped-aggregate-pandas-udfs-in-one-agg).

A `mapInArrow` result with more columns than declared is refused too by the server (Spark reads the declared columns
and ignores the rest when the setting is off, as the embedded runtime does; with it on, all raise
`ARROW_TYPE_MISMATCH`). Krishiv does not know the client's Spark version
here, so the setting is honoured whatever the version: under a Spark 3.5 or 4.0 client, which has no such setting,
setting it still turns the check on.

## Python UDF workers per session (server mode)

In server mode each running Python relation UDF (`mapInPandas`, `applyInPandas`, a `pandas_udf` aggregate, ...)
holds one of its session's `krishiv.python.worker.maxPerSession` workers for as long as the operator runs, and one
worker is kept for scalar Python UDFs. An operator that needs more workers at once than that leaves (for example
nine `pandas_udf` aggregates in one `agg` with the default of 8) fails with `KRISHIV_PYTHON_WORKER_LIMIT`; with the
setting at 1, a scalar Python UDF call while a relation UDF of the same session runs (in any query) fails the same
way. All sessions' relation UDFs together hold at most `krishiv.python.worker.maxTotal` minus one workers; one
waits for another to finish when they would hold more, in arrival order. Spark runs such calls in one worker and has no such limit.

A scalar iterator UDF (`pandas_udf` or `arrow_udf` with `Iterator` hints) and a Python UDF over a window hold a
worker the same way, one per UDF of the operator.

A Python UDF over a window keeps each window partition's input rows in memory until the partition's values come
back from the worker; Spark queues them in a buffer that spills to disk. A partition larger than the query's memory
fails with the memory pool's error in Krishiv where Spark finishes.

A scalar iterator UDF runs one Python iterator per driver of the operator, not one per Spark partition, so a UDF
that keeps state across its iterator (a running total, a call count) sees other batch boundaries than in Spark. The
operator also reads all its input and keeps all its output before the next operator reads any, so a `LIMIT` above
it does not stop it early.

## Python artifacts in server mode

The Spark Connect server does not hand a session's Python artifacts to its UDF workers yet: pyfiles and archives a
Connect client adds (`addArtifacts(..., pyfile=True)`, `archive=True`) do not reach the workers, so a UDF that
imports one fails with Python's import error where Spark finds the module. The worker pool takes them
(`SessionUdfs::set_artifacts`); the server's forwarding to it is a separate change that follows. Once a session's
artifacts are set, the pool treats them as below.

Pyfiles go on the workers' `sys.path`, as in Spark: the files directory, then each zip or egg. Archives (zip, jar,
tar, tar.gz, tgz) are unpacked into the same files directory under their `#alias`, or else their file name, as
Spark's executor does; an archive's own directory is not on `sys.path` (as in Spark, its packages import as
`alias.pkg`), and each entry of the files directory is reachable by a path relative to the worker's working
directory (`./alias/...`), as in Spark. Two differences remain. The working directory is the worker's own writable
directory, where the entries are links to a read-only directory of the server, so code cannot write into an
unpacked archive. Spark takes any alias (a fragment like `../x` unpacks outside its directory) and lets a second
archive of one alias replace the first; Krishiv refuses an empty, hidden (`.`, `..`), over-long (more than 255
bytes) or `/`-containing alias with `INVALID_ARTIFACT_PATH`, and two artifacts of one name with
`ARTIFACT_ALREADY_EXISTS`. A tar member that is a link leaving its archive's directory, a device file, or artifacts
past the server's size limit fail the session's UDF calls with `KRISHIV_PYTHON_ARTIFACT_REFUSED`.

## Python table functions (UDTFs)

Python UDTFs (`@udtf`, `@udtf(useArrow=True)`, `@arrow_udtf`) run on the server's worker pool from SQL (`FROM f(..)`,
`TABLE(t)` arguments with `PARTITION BY` / `ORDER BY` / `WITH SINGLE PARTITION`, `LATERAL f(..)`), with `analyze` at
analysis time. They match PySpark 4.2.0's answers (`python/tests/pyspark_udf/udtf/goldens.json`) except:

- A UDTF over an input without rows (a `TABLE` argument over an empty relation without `PARTITION BY`, or a
  `LATERAL` call over an empty left side) whose class has a `terminate` method: Spark 4.2.0 fails with a JVM
  `NullPointerException` (`EvalPythonUDTFExec` joins `terminate`'s rows with an input row that never came). Krishiv
  starts no UDTF instance for such an input and returns no rows, as Spark does for `PARTITION BY` over an empty
  relation.
- A `@udtf` without Arrow (eval type 300) converts its rows as Spark's JVM does (`EvaluatePython.makeFromJava`: a
  value of another type than the declared one is NULL, an integer is cut to the column's width, a string column
  takes any value's text), but where Spark shows a JVM object's own `toString` in a string column (`[B@...` for
  `bytes`, `java.util.GregorianCalendar[...]` for a `date`), Krishiv shows the value's Python `str`; and a `Row`
  inside a result row, which Spark's JVM fails to unpickle (`PickleException`), is NULL in the column.
- A `cogroup(...).applyInPandas` / `applyInArrow` whose two sides group on keys of different Spark types (`int` against
  `bigint`) is refused with `KRISHIV_UNSUPPORTED_FEATURE`; the keys must have one type.
- An `@arrow_udtf` (302) result with a NULL in a column declared `NOT NULL`: Spark 4.2.0's JVM fails reading the
  batch (`IllegalStateException`, no error class); Krishiv refuses it as a result unlike the declared schema
  (`PYTHON_EXCEPTION`).
- A constant argument the analyzer does not fold (`array(1, 2)`) reaches `analyze` as no constant (`value` is
  `None`, `isConstantExpression` false); Spark folds it.
- `analyze`'s result is limited to 64 MiB (as the worker encodes it, the pickled `AnalyzeResult` as hex); a larger
  one fails the call with `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON`. Spark sets no limit.
- A constant `BINARY` argument reaches `analyze` as `bytes`, Spark's default
  (`spark.sql.execution.pyspark.binaryAsBytes=true`); the setting is not read, so its `false` (`bytearray`) is not
  available.
- `LEFT JOIN LATERAL f(..) ON cond` (and `USING` or `NATURAL`) with a Python UDTF `f` is refused
  (`KRISHIV_UNSUPPORTED_FEATURE`) until its rows are checked against a Spark run; an inner `JOIN LATERAL f(..) ON
  cond` runs as Spark runs it, and `LEFT JOIN LATERAL f(..)` without a condition gives Spark's rows (those of the
  inner join).

## `schema_of_variant` with `spark.sql.caseSensitive=true`

Spark's `schema_of_variant` merges two objects of the same size whose sorted field names match under the session's
name resolver (`findTypeForComplex`), keeping the first object's names: `[{"A":1},{"a":2}]` is
`ARRAY<OBJECT<A: BIGINT>>` with the default `spark.sql.caseSensitive=false`, and Krishiv gives the same. The function
does not read the setting: with it on, Spark keeps both spellings (`ARRAY<OBJECT<A: BIGINT, a: BIGINT>>`) and Krishiv
still merges them.

## `PIVOT` cases Spark answers from undefined state are refused

When every aggregate of a `PIVOT` has a numeric, boolean or decimal type, Spark takes its `PivotFirst` path and
Krishiv reproduces it, including Spark's slot map (a NaN value never matches, `0.0` and `-0.0` are one value).
Repeated pivot values depend on the version: under Spark 4.2 and earlier they raise `INVALID_ARRAY_INDEX` with ANSI
on and leave NULL cells with ANSI off; under 4.3 (the default) every output reads its own value's slot, so repeated
values give equal cells and nothing is raised. A few cases fail with `KRISHIV_UNSUPPORTED_FEATURE` instead of
returning a result:

- repeated pivot values with ANSI off and several aggregates under 4.2 and earlier: Spark writes the repeated
  value's result into the next aggregate's cells. This refusal is Krishiv's choice under 4.3 until a 4.3.0 golden
  for the case exists;
- several pivot values of a type the analyzer does not evaluate (timestamp, interval, time) when one of them is not
  a literal of the pivot column's type, for example `IN (TIMESTAMP'2020-01-01 00:00:00', '2021-01-01 00:00:00')`:
  Krishiv cannot tell whether two of them are the same value, which decides Spark's result;
- a collated pivot column with Spark 4.0 or 4.1 compatibility: those versions match the values by their bytes,
  not by the collation;
- a NULL pivot value for a binary or collated pivot column: Spark 4.2 compares it with the other values in a sorted
  map (and fails), earlier versions look it up in a hash map.

## `describe`, `summary`, `freqItems` and sampling in Python

- `describe`, `summary` and `freqItems` evaluate when called and return a local DataFrame (a snapshot: it is not
  re-evaluated if the source changes); Spark builds a lazy plan. The values, formats and errors are Spark's.
  `describe`/`summary` run one engine aggregate. A percentile is the engine's `percentile_approx` with Spark's accuracy
  of 10000, which is the exact nearest rank up to 10000 rows.
- `freqItems` reads each column into the Python driver and runs Spark's counting algorithm one row at a time, so it
  costs O(rows) single-threaded Python work per column and fails with `KRISHIV_UNSUPPORTED_FEATURE` past 10,000,000
  rows. It is exact on one partition; the order of the items is not Spark's hash order. An engine aggregate is the
  follow-up.
- `approxQuantile` runs one `percentile_approx` per column and probability (the engine takes no array of
  percentages): with `relativeError=0` each one buffers the whole column, so k probabilities use k times its memory.
  With `relativeError` above 0 the result may differ from Spark's Greenwald-Khanna summary in the last rank;
  `relativeError=0` is exact. An empty `probabilities` raises `CANNOT_BE_EMPTY` (Spark fails with a
  `NullPointerException`).
- `sample`, `randomSplit` and `sampleBy` draw one number per row from Spark's `XORShiftRandom` (`rand(seed)`, seeded
  with `seed + partition`) and keep the rows Spark keeps only when the partitions and the order of the rows inside
  them are Spark's. Krishiv ignores `numSlices` (`spark.range(0, 20, 1, 4)` is one partition), reads a `union` as one
  stream rather than one partition per input, and cannot match Spark after a shuffle (`repartition`) or on a file
  split Spark divides differently; then a sample holds different rows and a different number of them. Only the
  expected fraction holds. `randomSplit` sorts each partition by its scalar columns; Spark 4.2 also sorts array and struct columns whose
  children are orderable, so inputs with equal scalar keys and different array or struct values can split
  differently. Its splits stay disjoint and complete for the same input.
- `sampleBy` finds a stratum by type, as Spark's typed map does: a string column matches only string keys and a
  numeric column only numbers.
- `df.sample(0.3, seed=1)`: PySpark (3.5 to 4.2) drops the seed of this call form and samples at random. Krishiv
  honours it, so the result is deterministic.
- `sample(withReplacement=True, ...)` fails with `KRISHIV_UNSUPPORTED_FEATURE`: Spark draws each row's copies from a
  Poisson distribution (commons-math `Well19937c`) seeded per partition, which is not reproduced, and a different
  multiset of rows would be a silent difference. The Rust `DataFrame::sample` is not implemented either (no sample plan
  node: the Python layer filters on `rand`), and a Spark Connect `Sample` is not served yet.
- `GroupedData.pivot`: `count("*")` raises `INVALID_USAGE_OF_STAR_OR_REGEX` as Spark's DataFrame pivot does (SQL's
  `PIVOT` clause accepts it). A grouping key that is not a plain or aliased column cannot be pivoted beside an
  aggregate that reads a key of the same name (`KRISHIV_UNSUPPORTED_FEATURE`). A pivot's aggregate errors are raised by `agg`.
  An explicit `values=[]` gives no pivot columns, as classic PySpark does; the values are discovered only when `values` is omitted.

## Inferred join filters differ from Spark's
Spark's `InferFiltersFromConstraints` adds filters to the inputs of a join from the join condition and the filters
below it. Krishiv adds the same `isnotnull` on nullable join keys, and moves comparisons with a literal, literal `IN`
lists and `isnotnull` across `=` keys of one type, on the sides Spark filters. It never substitutes a column into
another expression, so:
- `EXPLAIN` lacks the filters Spark derives that way: `a + 1 > 5` from `b + 1 > 5` and `a = b`, `(a = 1 OR a = 2)`,
  `isnull(a)`, `StartsWith`/`LIKE`, cast forms such as `cast(a AS STRING) = '5'`, and equalities of two columns
  of one side;
- Spark also adds `isnotnull(c)` beside any filter whose own predicate rejects a null `c` (`c = 'x'` becomes
  `isnotnull(c) AND c = 'x'`), at joins or not; Krishiv adds `isnotnull` only for join keys (TPC-H: 51 of Spark's
  162 inferred `isnotnull` are of this kind);
- Spark adds `isnotnull` for every column a null-rejecting join condition reads (`l_quantity < 0.2 * avg(..)`) and
  for a key equality common to every branch of an `OR`; Krishiv only for top-level `=` keys;
- a join key's `isnotnull` is not pushed into a subtree that another part of the query reads too (TPC-H q15's
  `revenue0`, read by the join and by the `max` subquery): the filter would make the two reads differ, and they
  could no longer be computed once;
- such a filter can raise in Spark (a cast or arithmetic under ANSI) on rows the query's own predicates never
  evaluate; Krishiv does not raise there.

Krishiv also infers at the semi and anti joins that `IN`/`EXISTS`/`NOT EXISTS` subqueries become (TPC-H q04, q20,
q22: `isnotnull` on their keys); Spark rewrites those subqueries after its inference and adds none there. A null key
never matches, so these filters drop only rows the join drops.

The rows returned are the same. ADR 0023 (update 2026-11-06).

## A multi-table `INSERT` that fails while it runs keeps the clauses written before

`FROM source INSERT INTO a SELECT ... INSERT INTO b SELECT ...` writes each clause's rows of the shared source. Every
clause is resolved and checked first, so a missing table, an unknown column or a column count mismatch in any clause
writes nothing, as in Spark. Spark plans the clauses as one statement, so a failure while one of them runs (a cast of a
malformed value under ANSI, say) also leaves the other targets unwritten; Krishiv writes the clauses in order, and the
ones before the failing clause stay written.

## Correlated subqueries with `spark.sql.optimizer.decorrelateInnerQuery.enabled` off

With inner-query decorrelation off, Spark refuses some shapes of correlated subquery that it answers with the framework
on (it still answers simple ones). Krishiv reads the setting only for the nested count-bug shape (a correlated global
aggregate below another operator, as in `SELECT (SELECT sum(c) FROM (SELECT count(*) c FROM t2 WHERE t1.a = t2.b)) FROM t1`),
which stays refused with `KRISHIV_UNSUPPORTED_FEATURE`, as it does under `spark.sql.legacy.scalarSubqueryCountBugBehavior`
and `spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled`. Every other correlated shape is answered
as with the setting on, including those Spark would refuse with an analysis error.

## SQL scripting

The interpreter follows Spark 4.2.0 (`tests/spark-golden/v4.2.0/scripting`) except for these:

- A `FOR` loop runs its query to the end and holds the rows in memory for the loop (Spark reads them lazily). A script's rows are limited by memory only: `UNABLE_TO_ACQUIRE_MEMORY`.
- A handler declared inside an `IF` body, and a `DECLARE ... CONDITION` after a handler, fail with the intended error classes (`INVALID_HANDLER_DECLARATION.WRONG_PLACE_OF_DECLARATION`, `INVALID_ERROR_CONDITION_DECLARATION.NOT_AT_START_OF_COMPOUND_STATEMENT`) where Spark 4.2 crashes with a Java `NoSuchElementException`.
- An unaliased output column over a local variable is named from the expression as written, not with Spark's random scope id.
- `KRISHIV_*` errors, `OPERATION_CANCELED` and internal errors are never caught by a handler. Cancelling or timing out a script (`krishiv.query.maxExecutionTime`) ends the whole script, between statements too.
- Locals, labels and `FOR` columns are lower-cased whatever `spark.sql.caseSensitive` says.

## SQL function bodies and the ANSI mode they were created under

A SQL function, scalar or table, evaluates in the `spark.sql.ansi.enabled` mode of its `CREATE`, as Spark does: the body's arithmetic, casts and functions follow the creation mode, while the arguments keep the caller's. Where Krishiv differs:

- A table function's body is pinned in its projections and filters only. Joins, grouping and aggregate arguments, and sort keys inside the body follow the caller's mode (`SELECT max(1 / x)` as a body raises or not by the caller's setting).
- An aggregate or window call in a scalar function argument reads its operator's mode.
- Only `spark.sql.ansi.enabled` is captured; Spark captures every SQL configuration.

## The relation cache is per session

Spark's `CacheManager` lives in the `SharedState` of a `SparkContext` (believed shared by every session of it, not
yet confirmed against a Spark run): a frame cached in one session reads as cached, and is reused, in another.
Krishiv keeps `cache()` / `persist()` registrations and their data per session (owner decision 2026-11-12, design
P3-T083 §3.7), because each session can have its own memory pool (`krishiv.session.memoryLimit`) and a shared
entry would charge one tenant for another's cache. `isCached` and `storageLevel` in one session do not see a
`cache()` made in another, and closing a session drops its entries (memory and spill files). Invalidation does
cross sessions: a write or `REFRESH TABLE` in one session drops what every session cached over the table, as in
Spark. To be revisited with multi-tenant resource groups (P3-T003).

Storage levels: `_2`/`_3` replication and `OFF_HEAP` are accepted and reported as asked, but one copy is kept, in
the engine's memory or one local spill file. Spilled and `DISK_ONLY` entries are scratch data in the spill
directory (`krishiv.spill.*`), deleted with the entry or at the next start after a crash.

## Checkpoint files and where a checkpoint lives

`DataFrame.checkpoint()` writes the rows under `spark.checkpoint.dir` as a Krishiv spill-format file
and, unlike Spark, deletes the file when the result and every frame derived from it are gone (Spark
keeps reliable checkpoint files by default, and nothing in Krishiv reads them back from another
process). The rows are also held by the session, in its relation store (memory-accounted), so a
reliable checkpoint holds a memory copy; `localCheckpoint(storageLevel=...)` accepts a level but holds
the rows in the session's memory whatever it is. With `eager = false` the rows are taken when the
result is first analyzed (its first action, or the first call that needs its schema), not at its
first job. A checkpoint is not a cache entry: `is_cached`, `CLEAR CACHE`, `REFRESH` and `unpersist`
do not see it, as in Spark. `system.caches` lists the session's live checkpoints in a `checkpoint` row.

`spark.conf.set("spark.checkpoint.dir", ...)` succeeds in Krishiv. Spark 4.2 over Spark Connect refuses it
with `CANNOT_MODIFY_CONFIG` (SQLSTATE 46110), and a Connect client there has no other way to set the
directory, so a reliable `checkpoint()` fails with "Checkpoint directory has not been set in the SparkContext"
unless the server was started with one. Krishiv accepts the setting per session (or `krishiv.checkpoint.dir`
at server start) because a client with no `SparkContext` has no other route; code that relies on Spark's
refusal is the only code that sees a difference.

## `${env:NAME}` and `${system:NAME}` in a server's SQL stay as written

Spark replaces `${env:NAME}` and `${system:NAME}` in SQL text with the driver's environment variable and Java system
property before parsing (`VariableSubstitution`, `spark.sql.variable.substitute`). In a `krishiv server` session (Spark
Connect, Flight SQL, the console's SQL editor) Krishiv leaves both as written, as it leaves a name it does not know:
otherwise any signed-in client could read every secret the server process holds (`SELECT '${env:AWS_SECRET_ACCESS_KEY}'`,
or the variables that `env:` references in `krishiv.auth.*` and catalog tokens name). `${name}` of a configuration value
is still substituted, except a setting that `SET` and Connect `Config.get` hide: a key or value that looks secret (it
contains `secret`, `password`, `token`, `accesskey`, `access.key` or `credential`, or a credential such as
`postgres://user:password@host`). Such a `${name}` stays as written too (Spark substitutes it), so a client cannot read
`${fs.s3a.secret.key}` that the operator set at startup. An operator who wants Spark's behaviour back for the environment
sets `krishiv.server.sql.substituteEnvironment=true` at server start; no `SET` or Connect `Config` call can change it. That
key does not bring back hidden settings. Embedded sessions (`krishiv sql`, the Python and Rust
APIs) read the environment as Spark does; there, `${system:NAME}` reads the environment variable `NAME` too, since there
is no JVM.

## A second catalog, `system`, next to `spark_catalog`

`SHOW CATALOGS` lists `spark_catalog` and `system`. `system` is Krishiv's own catalog of introspection tables
(`system.capabilities`, `system.caches`, `system.queries`, `system.metrics`, among others); Spark has no such catalog and
lists only `spark_catalog`. A declared catalog (`krishiv.catalog.<name>.*`, or Spark's `spark.sql.catalog.<name>`) appears in the list as well. The names
`spark_catalog`, `system`, `delegation` and `vended` cannot be used for a declared catalog.

## `remote("local")` is not supported

In Spark 4.3, `SparkSession.builder.remote("local")` starts a JVM `SparkContext` with the Connect plugin and
needs the full `pyspark` package; `pyspark-client` cannot do it. Krishiv has no JVM and does not accept
`remote("local")`. Use `krishiv.start_local()`, which starts an in-process Connect server and returns the
connection string for a stock `pyspark-client` (see [the guide](../guides/embedded-connect.md)). Like Spark's
local mode, it protects the port with a token made for each start; unlike it, the URL has the host `localhost`
and the token in the connection string.

## A declared schema on a Delta read

`spark.read.schema(..).format("delta").load(path)` (and the same read through Spark Connect) is refused with
`KRISHIV_UNSUPPORTED_FEATURE` when the table is opened, because what delta-spark 4.4.1 does with a declared schema is
not pinned by the Delta oracle yet. A Delta table's schema always comes from its log. No call on such a read
answers with the declared schema: embedded, `load` itself raises; through Spark Connect, where `load` sends
nothing to the server, the first call that does raises, `df.schema` and `printSchema` included.

## Tags and global views

`spark.addTag(tag)` tags the queries of the calling thread. A DataFrame keeps the tags it was made with: tag
before `spark.sql`, `spark.table`, `spark.read` or `spark.range`, not before the action. A frame made before the
tag, or made on another thread, is not cancelled by `spark.interruptTag(tag)`, and a frame made under a tag keeps
it after `removeTag`. Frames derived from it (`filter`, `select`, ...) keep it. In Spark, the tags are those of the
thread that runs the action. `removeTag` of a tag the thread does not have does nothing, as in Spark 4.3.

Global temporary views (`createGlobalTempView`, `CREATE GLOBAL TEMPORARY VIEW`) belong to the session that
made them: another session, `spark.newSession()` included, does not see `global_temp.<name>`. In Spark they are
shared by every session of the application.

## `krishiv server` (Spark Connect)

> The Spark Connect server is not part of the 0.2 release; `krishiv server` in 0.2 serves the web UI and admin API
> only. The behaviour below is what the Spark Connect server is built to do, and it does not apply to 0.2.

- **The Connect listener binds `127.0.0.1` by default.** Spark binds every interface when
  `spark.connect.grpc.binding.address` is unset. A non-loopback address needs authentication, so `krishiv server`
  refuses to start on one until authentication is configured.
- **A session belongs to the authenticated principal, not to `UserContext.user_id`.** Spark keys a session by the
  `user_id` the client sends; Krishiv keys it by who authenticated, and records `user_id` without trusting it. Two
  clients of one principal with different `$USER` values and the same session id share a session in Krishiv but not
  in Spark. Clients pick random session ids, so this does not happen in practice. A session id another principal holds
  answers `INVALID_HANDLE.SESSION_NOT_FOUND`.
- **An idle session running a query is kept.** Spark releases a session by its last access alone; Krishiv also keeps
  one whose query is still running or whose results are still being sent.
- **A request over `spark.connect.grpc.maxInboundMessageSize` is refused with `OUT_OF_RANGE`**, read from the message
  length before the body; Spark (grpc-java) answers `RESOURCE_EXHAUSTED`.
- **`krishiv.connect.session.idleTimeout`** replaces `spark.connect.session.manager.defaultSessionTimeout`, whose
  "never" is `-1`; Krishiv's is `0s`. `krishiv server` refuses Spark's key, and any other `spark.connect.*` key it
  does not read, rather than ignore it.
- **Too many operations in one session** fail with `KRISHIV_OPERATION_LIMIT` (`RESOURCE_EXHAUSTED`) past
  `krishiv.connect.maxOperationsPerSession`; Spark has no such cap.
- **An invalid operation tag** fails with `KRISHIV_INVALID_OPERATION_TAG` (SQLSTATE 22023) and Spark's message; Spark
  sends an `INTERNAL` status without an error class.
- **A Connect `Sample` whose bounds span more than `[0, 1]`** (or NaN) fails with `KRISHIV_SAMPLE_FRACTION_OUT_OF_RANGE`
  (SQLSTATE 22023) and Spark's text (`requirement failed: Sampling fraction (1.5) must be on interval [0, 1] without
  replacement`); Spark raises a classless `IllegalArgumentException` from the `Sample` constructor. PySpark raises
  `IllegalArgumentException` for both.
- **Results arrive in the engine's batches** (4,096 rows by default), cut further at Spark's
  `spark.connect.grpc.arrow.maxBatchSize` and `spark.sql.execution.arrow.maxRecordsPerBatch` but never merged.
- **Timestamps are labelled UTC when `spark.sql.session.timeZone` is unset**; Spark labels them with the JVM's zone.
  The instants are the same.
- **Geometry, geography and variant columns** get Spark's Arrow shape at the top level of a result only, not inside a
  struct, array or map.
- **The `metrics` response of ExecutePlan is empty** until the operator tree is wired through.
- **A reattachable operation's query pauses** once `krishiv.connect.execute.aheadBytes` (64 MiB) of its responses are
  unread, whether or not a client is attached; Spark keeps producing and buffers every unread response.
- **A new operation can wait for admission.** Each operation reserves its worst case (`aheadBytes` plus, reattachable,
  `krishiv.connect.execute.retryBufferSize`) of `krishiv.connect.maxResponseBytes` (1 GiB) before its first response,
  until it is released, abandoned or ends; past the cap a new `ExecutePlan` waits (its stream sends nothing yet) until
  one does. Running operations never wait on it. Spark has no such cap. A server refuses to start with a cap below one
  operation's worst case (`KRISHIV_INVALID_CONFIG_VALUE`). There is no per-principal share yet, so one principal's
  open operations can hold the whole cap.
- **Column metadata travels in the `schema` response only**: Spark 4.3 also writes each field's user metadata into
  the Arrow schema of every result batch (`SPARK::metadata::json`); Krishiv's batches carry none.
- **ReleaseExecute without a release type** fails with `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET`; Spark throws
  an `UnsupportedOperationException` without an error class.
- **AnalyzePlan.** An `UNSPECIFIED` explain mode fails with `CONNECT_INVALID_PLAN.INVALID_ENUM`, and `json_to_ddl` of a
  type that is not a struct with `INVALID_SCHEMA.NON_STRUCT_TYPE`; Spark sends `INTERNAL` without an error class for
  both. `persist`, `unpersist` and `get_storage_level` fail with
  `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED` until DataFrame caching lands. `input_files` lists the paths
  the file scans read, qualified (`file:/data/t`), not the files under them, and nothing for a catalog table (Spark
  lists the files of a file-backed table). `same_semantics` and `semantic_hash` ignore column and alias names and
  subquery aliases, as Spark's canonicalization does, but not Spark's other rewrites: plans that differ only in the
  order of commutative operands (`a + b` and `b + a`), in a projection that only renames or repeats the input's
  columns, or in join hints are not the same in Krishiv and are in Spark. `semantic_hash` values are Krishiv's own
  (equal for plans `same_semantics` calls equal, within one server), not Spark's. `ddl_parse` refuses the clauses only
  a table schema takes (`DEFAULT`, `GENERATED`), and its parse errors are those of the text as a data type, where Spark
  reports the table-schema parse. `spark_version` is the session profile's release line (`4.2.0`).
- **Config.** Secret-looking keys and values (Spark 4.3's `spark.redaction.regex` default) are hidden for 4.2 sessions
  too; Spark 4.2 shows them. The pattern is the default only (Spark reads `spark.redaction.regex` from the startup
  configuration). Krishiv also hides what its own secret classifier finds, where Spark 4.3 shows it: keys such as
  `fs.azure.account.key.<account>`, `sasl.jaas.config` or `*.apiKey`, and values with credentials in a URL
  (`postgres://user:password@host`). SQL `SET` and `SET <key>` hide the same pairs and, as Spark's do, any pair with
  `url` in the key or the value (the default of `spark.sql.redaction.options.regex`; a changed pattern is not read, and `SET` of that key itself
  prints `<undefined>`, where Spark hides its default value because the value contains `url`);
  `SET -v` hides a registered key that holds a secret, where Spark's `SET -v` hides nothing. `set` of a key without a value fails with `INVALID_CONF_VALUE.REQUIREMENT`, and a request without an
  operation with `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET`; Spark sends `INTERNAL` without an error class.
  `is_modifiable` answers `true` for a key outside the engine's registry, where Spark answers `false` for a key it does
  not know. `get_all` lists a registered key under its Spark name when it has one, else under its `krishiv.*` name.
  Setting a key the registry marks admin-only answers `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501, gRPC
  `PERMISSION_DENIED`) for a non-admin role (no role exists before authentication, so the local user is the admin).
- **Interrupt by tag** fails with `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` until operation tags can be interrupted.
- **AddArtifacts.** Names are checked before anything is written and more strictly than Spark: one of the prefixes
  `cache/`, `pyfiles/`, `archives/` or `files/`, `/`-separated, without empty, `.` or `..` segments, `\`, control or
  invisible formatting characters, at most 32 segments of at most 255 bytes; a `cache/` name is one segment of ASCII
  letters and digits. Anything else is `INVALID_ARTIFACT_PATH` (Spark accepts an unknown prefix as a plain file and
  turns `\` into a separator). `jars/` and `classes/` (no JVM) and `forward_to_fs/` (no Hadoop file system) fail with
  `KRISHIV_UNSUPPORTED_FEATURE`. Each session may hold `krishiv.connect.artifacts.maxBytes` (1 GiB) of artifacts and
  one artifact `krishiv.connect.artifacts.maxArtifactBytes` (256 MiB); past either,
  `KRISHIV_ARTIFACT_LIMIT` (`RESOURCE_EXHAUSTED`); Spark has no cap. Each segment of a name (the file and every
  directory it may need) counts 4 KiB rather than each artifact, and all sessions together may hold
  `krishiv.connect.artifacts.maxTotalBytes` (16 GiB). A broken chunk sequence (too many or too few chunks, a chunk
  with no artifact begun, a new chunked artifact before the last chunk, a wrong `total_bytes`) fails with
  `KRISHIV_INVALID_ARTIFACT_TRANSFER` where Spark sends `INTERNAL` without an error class. A `batch` sent while a
  chunked artifact is still open fails the same way, which Spark accepts (the stock client never sends one). A
  chunked artifact of one chunk is accepted (Spark fails it). An artifact that collides with one already stored fails with
  `ARTIFACT_ALREADY_EXISTS` as in Spark, also when the name holds a directory or a link. Archives are stored, not
  unpacked, until Python UDFs use them.
- **FetchErrorDetails** sends no stack trace (Spark sends the JVM's): there is no JVM, and a Rust backtrace would
  expose server internals. A client that asks for the server stack trace gets the root error's class line only. The
  causes of an error are sent with the parameters of server-internal error classes redacted, as their own status would
  be. Errors of a Connect `Sql` relation carry no SQL query context yet.
- **CloneSession.** The clone gets copies of the configuration, temporary views, temporary functions, variables,
  current catalog and namespace, and the session's artifacts (charged to the clone's
  `krishiv.connect.artifacts.maxBytes` and to `maxTotalBytes`; a copy that does not fit fails with
  `KRISHIV_ARTIFACT_LIMIT` and creates no session). Differences from Spark: Rust UDFs (`register_udf`) are shared
  with the clone rather than copied; global temporary views are per session in Krishiv, so the clone gets a copy and
  the two diverge (Spark shares them); cached relations are not copied (Spark's cache is shared by both sessions); a
  clone counts against `krishiv.connect.maxSessions` and `maxSessionsPerPrincipal` (`KRISHIV_SESSION_LIMIT`), which
  Spark does not check; a target id another principal holds answers `INVALID_HANDLE.SESSION_NOT_FOUND`; and a
  target created by a racing request while the clone is made fails with
  `INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_ALREADY_EXISTS` where Spark returns the racing session.
- **GetStatus** ignores request and operation extensions: Krishiv has no `spark.connect.extensions.getStatus.classes`
  plugins. A session remembers the final state of at most 1,000 ended operations, each for 30 minutes after its last
  lookup; Spark keeps every one for the 30 minutes. Operation ids are matched in their canonical UUID form, so an
  upper-case spelling finds the operation; Spark keys them by the string as sent. Which state Spark reports in a few
  cases has not been checked against Spark yet: Krishiv reports `CANCELLED` for an `ExecutePlan` whose client went
  away during analysis, `FAILED` for one that failed there, and `TERMINATING` for an interrupted operation that has
  not stopped yet.
- **`ml_command.clean_cache` answers with no result.** PySpark sends it when its process exits; Krishiv has no ML
  cache, so the operation succeeds with no response (only `result_complete` when reattachable). Spark sends an
  `ml_command_result`, whose content for this command is not verified; PySpark ignores it. Its operation id is not
  remembered after it ends, so the same id may be reused (Spark refuses a reused id with `OPERATION_ABANDONED`;
  clients send fresh ids). Every other `ml_command` fails with SQLSTATE 56K00.

## Declared catalogs (`krishiv.catalog.<name>.*`, `spark.sql.catalog.<name>`)

- **`SHOW CATALOGS` and `spark.catalog.listCatalogs()` list every declared catalog the principal sees, used or
  not.** Spark lists `spark_catalog` and only the catalogs the session has already loaded. Spark loads a catalog when
  a multi-part name first names it, even in a lookup that then fails, or with `SHOW NAMESPACES IN <catalog>`.
  Setting `spark.sql.defaultCatalog` to it does not load it by itself. So a fresh Spark session answers
  `spark_catalog` only, and Krishiv answers `spark_catalog` plus its declared catalogs. A catalog declared for other
  tenants is never listed. `SHOW CATALOGS` also lists `system`; `spark.catalog.listCatalogs()` does not.
- **In `krishiv server`, a session cannot declare or change a catalog.** `SET`, `RESET` and `spark.conf.set` of
  `spark.sql.catalog.<name>`, `spark.sql.catalog.<name>.*` or `krishiv.catalog.<name>.*` fail with
  `CANNOT_MODIFY_STATIC_CONFIG` for every role. So does a session builder that sets one. Declare catalogs in the
  server's configuration instead. Spark lets a session set `spark.sql.catalog.<name>` at runtime and loads the
  catalog when it is first used. In server mode that would mean fetching from an endpoint the client chose. Embedded
  sessions are unchanged. Registered settings such as `krishiv.catalog.default` are not declaration keys.

## `Dataset.observe`: where the counts differ from Spark's (follow-up P3-T084c)

`DataFrame::observe` / PySpark `Observation` match Spark 4.2.0 on every case of
`tests/fixtures/spark-4.2.0/observe-goldens.json` except the ones below
(`crates/krishiv-session/tests/observe_goldens.rs` pins them in `KNOWN_GAPS`, each must still fail there, so a
fix removes the line). A `LIMIT` (and so `take`, `first`, `show`) directly above the observed node,
or above projections only, counts exactly the rows it takes, as Spark's `CollectMetricsExec` counts the rows
its consumer pulled; the cap is one budget per node instance, so over several partitions the rows
counted are the first the racing drivers hand over (Spark's are partition 0's: a `sum` metric can differ).
The budget is per instantiation, not global: in a staged or distributed run every task that instantiates the
observed node has its own, so the count can exceed `n` (P3-T084c).

- **Limit not last, or a filter in between:** Spark counts the rows its limit operator *pulled* through the node, and
  which limit operator runs depends on the action (`observe-goldens-followup.json`, `limit_counts`, run on Spark 4.2.0,
  `range(0,1000,1,P)`, every case stable over 3 runs). `count()` plans `GlobalLimit <- Exchange <- LocalLimit` and
  scans *every* partition (each pulls `min(n, partition rows)`: P x n); `collect()`/`take` scans one partition, then
  `ceil(1.5 * missing * scanned / got)` more, at most 4 x the scanned. The engine applies one global cap of `n` rows
  at the node (through projections only). Counts, `n = 5` / `n = 100`, count() then collect():

  | shape | P = 1 | P = 4 | P = 16 | engine |
  |---|---|---|---|---|
  | `observe.limit(n)` | 5, 5 / 100, 100 | 20, 5 / 400, 100 | 80, 5 / 1000, 125 | P = 1 only (else 5 / 100) |
  | `limit(n).observe` | n | n | n | matches |
  | `observe.filter(v%2=0).limit(n)` | 9 / 199 | 36, 9 / 796, 199 | 152, 9 / 1000, 312 | reads 1000 (750 at P16) |
  | `observe.filter(v%100=0).limit(n)` | 401 / 1000 | 1000, 500 / 1000 | 1000, 437 / 1000 | reads 1000 |
  | `observe.offset(3).limit(n)` | n + 3 | 32, 8 / 412, 103 | 128, 8 / 1000, 125 | P = 1 only |
  | `observe.orderBy(v desc).limit(n)` (TakeOrderedAndProject) | 1000 | 1000 | 1000 | matches |
  | `orderBy(v desc).observe.limit(n)` | n | n | n | matches |
  | `observe.union(range).limit(n)` | as `observe.limit(n)` | | | reads everything |
  | `observe(A).limit(n) UNION ALL observe(B).limit(n)` | n, n | 20, 20 / 400, 400 | 80, 80 / 1000, 1000 | P = 1 only |
  | `(limit(n) UNION ALL limit(n)).observe` | 2n | 2n | 2n | matches |
  | observed side streamed through a broadcast join, `limit(n)` above | as `observe.limit(n)` (collect: Connect gives P x n) | | | reads everything |
  | observed build side of a join | 1000 | 1000 | 1000 | matches |

  The runner (`observe_goldens.rs`, `limit_counts_match_spark_where_they_can`) asserts every row marked "matches" and the
  P = 1 cases, and pins the rest in `LIMIT_GAPS` with the reason; they must still differ, so a fix removes the line.
  Emulating the per-partition `LocalLimit` and `CollectLimit`'s incremental scan is not planned: the numbers are plan- and
  AQE-dependent (the post-shuffle partition count changes them), so only the deterministic shapes are matched.
- **Sort and limit:** `df.observe(..).orderBy(..).limit(n)` counts every row, as Spark's `TakeOrderedAndProject` does
  for a descending sort (1000). The first fixture's ascending `orderBy(id)` over a range gave 20 of 1000 in Spark (a
  partition-dependent count the engine does not reproduce: `observation_through_sort_limit_topn` in `KNOWN_GAPS`).
- **ANSI `sum` overflow:** modelled on Spark's mechanism (`AggregatingAccumulator`): an overflow inside one task's
  partition fails the action with `ARITHMETIC_OVERFLOW`; an overflow when the tasks' buffers are merged drops *that task's
  whole buffer* and the action succeeds (Spark merges in task-completion order, so which task survives varies from run to
  run: `{MAX-1, MAX}` for two rows in two tasks; the engine merges in partition order, so the first partition's buffer
  wins). An overflow in a metric's result (a decimal `sum`/`avg`, `max(v) + 2`) is raised by `get`, not the action. The
  decimal class is `ARITHMETIC_OVERFLOW` where Spark's is `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION` (both SQLSTATE
  22003). The ANSI setting read is the one at action time (as Connect; classic Spark bakes the `Column`'s mode in when it
  is built).
- **Scalar subquery in a metric:** Spark runs it; the engine refuses it with `KRISHIV_UNSUPPORTED_FEATURE`.
- **Empty relation above:** a relation proven empty (a filter that folds to `false` or NULL, `LIMIT 0`, and what
  Spark's `PropagateEmptyRelation` carries that through: project, sort, keyed aggregate, inner or semi join with an
  empty side, a union of empties, ...) drops the observed nodes inside it, as Spark does (`{}`); an empty input below the
  node, an outer join's preserved side and a union with a live branch keep it (`n = 0`). Not matched: a cached frame
  (`observe(..).cache().where(false)`: the engine fills the cache before the filter prunes it) and `sample` (not supported).
  Spark's classic Python `get` raises a JVM `AssertionError` for a pruned node (an upstream defect); the engine returns `{}` as Connect does.
- **Connect:** struct, map, array, binary and day-time interval metrics go out as Spark's `ObservedMetrics` literal forms
  (year-month and calendar intervals too: Spark's Python client raises `UNSUPPORTED_LITERAL` for those, the server sends them);
  a name with no plan id reports `plan_id` 0. A calendar-interval metric cannot be built yet (the engine has no aggregate over it).
- **Streaming:** the engine has no streaming `DataFrame` yet (a Connect `Read` with `is_streaming` is refused when it is translated,
  `UNSUPPORTED_FEATURE`), so there is nothing for `observe(Observation, ...)` to refuse. Spark refuses it at `observe()`: classic
  `IllegalArgumentException` (no class) "Observation does not support streaming Datasets.This is because ..." (sic, no space);
  Connect `UNSUPPORTED_OPERATION` "Streaming DataFrame with Observation is not supported."; `observe("name", ...)` on a stream is
  allowed and reports per micro-batch in `StreamingQueryProgress.observedMetrics`
  (`observe-goldens-followup.json`, `streaming`). Both belong to the streaming plan row, not here.
- **Cancelled actions:** a dropped result stream completes the observation with `{}` or the rows
  folded before the drivers stopped (Spark: a partial value); a failed finalization of a metric is what `get`
  raises.
- **Actions that fail before their query starts:** an action refused or dropped before its query
  runs (a closed session, a full resource group, a plan that cannot be submitted, a caller that gives up
  or a cancel, timeout or kill while it waits) completes the observation with that error, so `get` raises
  it. Spark completes an observation only when an execution of the query ends (Spark 4.2:
  `SQLExecution.withNewExecutionId0` calls `ObservationManager.tryComplete` as it finishes; 3.5 to 4.1:
  the session's `QueryExecutionListener`, on success and failure alike), with the metrics it collected,
  or (4.1, 4.2) an empty row when it has none (4.2 also completes it with a failure when collecting the
  metrics fails: `Try(qe.observedMetrics)`), never with the action's own error. An action that fails before an execution starts leaves
  it pending, and a later action can still complete it. In the engine the first action, failed or
  not, completes it.
- **`Observation.get` blocks** a thread; do not call it from async code on a runtime thread.
- **Result cache:** a plan that observes is never served from the result cache (every action runs it).
- **`spark.sql.timestampType = TIMESTAMP_NTZ`:** a bare `TIMESTAMP` follows the setting everywhere Spark's type parser reads it (SQL, views as of their `CREATE`, Spark Connect type and schema strings, `from_json` / `from_csv` / `from_xml` schemas, `variant_get` types, the return type of a UDF registered on a session) except the return type of an unregistered Python `udf(...)`, which is built without a session and still means `TIMESTAMP_LTZ`.

## Delta Lake writes

Krishiv writes Delta tables through `delta_kernel` and its own writer (`krishiv-connectors-delta`,
`src/write/`). The rows, partition values and statistics match delta-spark 4.4.1 (the oracle in
`tests/fixtures/delta-4.4.1/`); these parts of the log differ:

- **Protocol.** A table Krishiv creates carries the table-features protocol (reader 3, writer 7)
  where delta-spark writes (1, 2) for a table with no features. Spark 4.2 with delta-spark 4.4.1
  reads such tables and appends to them, keeping the protocol
  (`tests/fixtures/delta-4.4.1/krishiv-written/READBACK.json`).
- **Operation names.** A CTAS commit records the kernel's `CREATE TABLE` (delta-spark: `CREATE TABLE
  AS SELECT`) with the kernel's parameters; a v1 `saveAsTable` overwrite of an existing table is
  recorded as `WRITE` (delta-spark: `CREATE OR REPLACE TABLE AS SELECT`).
- **commitInfo fields.** `isBlindAppend` is omitted when false (the kernel owns the field);
  `isDynamicPartitionOverwrite` is the string `"true"`, not a boolean; a `replaceWhere` writes no
  `numDeletedRows`; `partitionBy` is always `"[]"`.
- **Constraint text.** The nested `DELTA_VIOLATE_CONSTRAINT_WITH_VALUES` message of a
  `replaceWhere` mismatch names the constraint by the filter's text, where delta-spark prints its
  expression tree.
- **Float bounds.** Statistics order floats as Spark does (every NaN above every number), except
  that `-0.0` sorts below `0.0`: a file holding both keeps the other one as its bound (both are
  written as `0.0`).
- **Files.** Data files are named `part-00000-<uuid>.c000.snappy.parquet` for every task and roll
  over at the `maxRecordsPerFile` write option (a value not above `0` is no limit) and at a fixed
  128 MiB, on every write (`insertInto`, `saveAsTable` and `writeTo` on a catalog table too). A value
  that is not a number fails with `KRISHIV_INVALID_FORMAT_OPTION_VALUE`, where delta-spark raises
  Java's `NumberFormatException`.
- **Checkpoints.** A checkpoint after a commit is written without the `.crc` file, and a V2
  checkpoint without sidecars; it runs after the statement returns, and a failure is logged,
  never raised.
- **Refused.** `replaceWhere` on data columns, schema evolution (`mergeSchema`), `overwriteSchema`
  on an existing table and every replace (RTAS, `REPLACE TABLE`, `createOrReplace`) are refused by
  name. A `replaceWhere` or `INSERT OVERWRITE` filter must name partition columns by one-part names:
  a qualified name (`t.p`) and a struct field (`s.p`, a data column for delta-spark) are refused, and
  so is a string compared with a `TIMESTAMP` partition column, which delta-spark casts in
  `spark.sql.session.timeZone` (Krishiv's filter translation does not know the zone yet).
- **Write options** (`tests/fixtures/delta-4.4.1/OPTIONS.json`). Values are checked as
  delta-spark checks them (`DELTA_ILLEGAL_OPTION`, `DELTA_INVALID_IDEMPOTENT_WRITES_OPTIONS`) and
  unknown names are accepted, as delta-spark only logs them; `userMetadata` and `mergeSchema`
  (`canMergeSchema`, the string `"true"`) reach the commit. Refused by name where delta-spark
  commits: idempotent writes (`txnAppId` + `txnVersion`), `dataChange = false`, `replaceOn`,
  `replaceUsing`, `targetAlias` and `useNullIntolerantEqualityWithDPO = true`
  (`writePartitionColumns` is accepted and has no effect, as in delta-spark). A DataFrame
  `save(path)` merges the frame's types with the table's as delta-spark's `SchemaMergingUtils`
  does, with or without `mergeSchema` (`OPTIONS.json` `merge_schema`): narrower integrals,
  NULL-typed columns and other nested nullability are taken, other types are
  `DELTA_FAILED_TO_MERGE_FIELDS`, and a merge that would change the table's schema is
  `DELTA_METADATA_MISMATCH` without `mergeSchema`; with it (type widening on a `typeWidening`
  table, a wider integral, a new nested field) it is refused by name, since the kernel commits no
  schema change on a write. Type widening follows delta-spark's default
  `spark.databricks.delta.typeWidening.allowAutomaticWidening = same_family_type`; the conf is
  not read. A non-numeric `maxRecordsPerFile` is
  `KRISHIV_INVALID_FORMAT_OPTION_VALUE` where delta-spark raises a classless
  `NumberFormatException`. `optimizeWrite` and `compression` are accepted and change nothing.
  `INSERT INTO delta.`<path>`` is not taken (`TABLE_OR_VIEW_NOT_FOUND`); a catalog table is.

`DELETE` and `UPDATE` of a Delta table (copy-on-write, `src/write/dml.rs`) leave the rows, the
command's `num_affected_rows` and the files removed that delta-spark leaves
(`tests/fixtures/delta-4.4.1/dml/DML.json`; Spark reads the tables back, `dml/krishiv-written/`).
They differ in:

- **Predicate text.** `operationParameters.predicate` holds the condition as written, in
  parentheses (`["(id = 2)"]`); delta-spark prints Catalyst's form with expression ids
  (`["(id#1231L = 2)"]`). Without a condition both write `["true"]` (`DELETE`) and `[]`
  (`UPDATE`).
- **Deletion vectors.** On a table with `delta.enableDeletionVectors = true` Krishiv marks deleted
  and updated rows in deletion vectors as delta-spark does (merge-on-read, under the same
  `spark.databricks.delta.{delete,update}.deletionVectors.persistent` confs, default true; see
  the `MERGE` list below for what differs).
- **Files per partition.** A rewrite writes one file per partition per driver; delta-spark one per
  partition per task.
- **Whole-file deletes.** A `DELETE` removes whole files without rewriting them only when its
  condition is a comparison, `IN` or `IS [NOT] NULL` of partition columns with literals (combined
  by `AND`, `OR`, `NOT`); delta-spark also takes other expressions of partition columns. Other
  conditions rewrite (the same rows go, with a rewrite's metrics). A column qualified by the
  statement's alias counts (`x.p`); a struct field (`s.p`) and a string compared with a
  `TIMESTAMP` partition column (Spark casts it in the session time zone) make the statement a
  rewrite. A partition file with no live row is removed only when its statistics prove it
  (`numRecords` less its deletion vector's cardinality); delta-spark removes every file of the
  matching partitions.
- **Partition-only `UPDATE`.** An `UPDATE` whose condition names only partition columns rewrites
  the files that hold a matching row; delta-spark rewrites every file of the matching partitions,
  which differs only for files with no live row.
- **Read predicates.** The conflict check reads by the condition's top-level `AND` conjuncts that
  translate as partition predicates, and every live file they select is a read file (as
  delta-spark's `filterFiles`); delta-spark also takes partition-only conjuncts of other forms,
  which Krishiv reads as the whole table (a concurrent append to any partition conflicts).
- **Subqueries.** A condition with a subquery is refused with `DELTA_UNSUPPORTED_SUBQUERY` before
  anything runs, as delta-spark does; its `cond` parameter is the condition as written, where
  delta-spark prints Catalyst's SQL of the resolved condition. `delete.allowExistsSubquery`
  (delta-spark's internal flag admitting `EXISTS` in a `DELETE`) is not read.
- **Write errors.** An error raised while the rewrite writes its rows is `TASK_WRITE_FAILED` (the
  table's root) caused by the error, as Spark's writer raises it; an overflow of the stored value's
  cast is `DELTA_CAST_OVERFLOW_IN_TABLE_WRITE`, naming the first assigned column whose new value's
  type casts to the overflowing type. Krishiv's own refusals, conflicts, internal errors and
  cancellations stay unwrapped.
- **Refused.** On a table with the change data feed on, `UPDATE` and a `DELETE` that rewrites rows
  are refused (`UNSUPPORTED_FEATURE.TABLE_OPERATION`) before any file is written: Krishiv writes no
  change data files yet. A `DELETE` of whole partitions commits.

`MERGE INTO` a Delta table (copy-on-write; the session's `table_commands/merge.rs`, the connector's
`src/write/merge.rs`) leaves the rows, the command's four counts, the files removed and the counted
`operationMetrics` that delta-spark leaves (`tests/fixtures/delta-4.4.1/merge/MERGE.json`; Spark
reads the tables back, `merge/krishiv-written/`). It differs in:

- **Predicate text.** `operationParameters.predicate` and the `predicate` of each entry of
  `matchedPredicates`, `notMatchedPredicates` and `notMatchedBySourcePredicates` are the
  conditions as written, in parentheses; delta-spark prints Catalyst's form with expression ids.
  The same holds for the condition in `DELTA_NON_DETERMINISTIC_FUNCTION_NOT_SUPPORTED` and
  `DELTA_UNSUPPORTED_SUBQUERY` messages.
- **Unresolved names.** A name that does not resolve in a `MERGE` is `INTERNAL_ERROR` ("Cannot find
  main error class 'DELTA_MERGE_UNRESOLVED_EXPRESSION'"), the analyzer's `UNRESOLVED_COLUMN.*` as
  its cause. This matches a delta-spark 4.4.1 defect: its error class file lacks the class it
  raises. Revisit when a delta-spark release ships the class. A SQL script's handler does not take
  it (Krishiv's handlers never take `INTERNAL_ERROR`).
- **Source.** delta-spark copies the source unless the `MERGE` only inserts (with
  `spark.databricks.delta.merge.optimizeInsertOnlyMerge.enabled`, on by default) or the source
  reads only Delta tables and is deterministic (`MergeIntoMaterializeSource`). Krishiv reads an
  insert-only `MERGE`'s source once, under the same setting (with it off, the source is chosen as
  for any other `MERGE`), and a source that is one bare scan of a Delta table at its snapshot; any
  other source (a query, files, a Delta read under a filter or a projection) is copied into a
  spill file of its own first, a batch at a time, and read back from it. An
  insert-only `MERGE` and a Delta-table source are read once more to count `numSourceRows`.
- **Read predicates.** The conflict check reads by the `ON` condition's partition-only conjuncts
  (the whole table with a `NOT MATCHED BY SOURCE` clause); delta-spark also skips data files by
  the conjuncts' statistics, so Krishiv's read set is wider (a concurrent append to a partition the
  `ON` condition does not exclude conflicts).
- **Change data feed.** On a table with it on, a `MERGE` that removes a file is refused
  (`UNSUPPORTED_FEATURE.TABLE_OPERATION`) before any file is written; delta-spark commits it with
  change data files. A `MERGE` that only inserts commits, as delta-spark's does (with no change
  data files).
- **Schema evolution.** Under `spark.databricks.delta.schema.autoMerge.enabled = true` a `MERGE`
  that would add a column (a `*` clause whose source has a column the target lacks, an
  assignment to a source column the target lacks) is refused (`KRISHIV_UNSUPPORTED_FEATURE`); other
  `MERGE`s run, with `*` taking the target columns the source has. `MERGE WITH SCHEMA EVOLUTION`
  does not parse.
- **Refused.** A `MERGE` under `spark.sql.caseSensitive = true`, `SET c = DEFAULT`, a nested
  assignment key (`SET s.f = ..`) and a table whose row-level rule is Spark's own
  `MERGE_CARDINALITY_VIOLATION` (no oracle yet) are refused by name.
- **Deletion vectors.** Marked as delta-spark marks them (`merge.deletionVectors.persistent`,
  default true).

Merge-on-read `DELETE`, `UPDATE` and `MERGE` on a table with deletion vectors (the connector's
`src/write/dv.rs`) leave the rows, the counts, the files removed whole, the files given a new
vector with the vector's cardinality and the counted `operationMetrics` that delta-spark leaves
(`tests/fixtures/delta-4.4.1/dvs/DVS.json`; Spark reads the tables back, `dvs/krishiv-written/`,
and derives the same change data feed rows from a vector-only `DELETE` on a change-data-feed
table). They differ in:

- **Vector bytes.** delta-spark run-optimizes its bitmaps before writing them; the kernel's writer
  Krishiv uses does not, so a vector holding runs of rows takes more bytes (`sizeInBytes`). The
  rows are the same.
- **Vector file layout.** One vector file per statement, the vectors in path order; delta-spark
  bin-packs them per task in its aggregation's order (other offsets).
- **When vectors apply.** A `DELETE` or `UPDATE` whose condition is partition-only by Krishiv's
  translation (the whole-file `DELETE` rule above) removes or rewrites whole files; a
  partition-only condition of another form marks rows, where delta-spark removes whole files.
- **Files without statistics.** As in delta-spark, a file is removed whole only when its
  `numRecords` proves every live row goes; a file without the statistic keeps a vector of all its
  rows, and its new add carries statistics (`tests/fixtures/delta-4.4.1/dvs-no-stats/`). Krishiv
  writes only `numRecords` there (the footer's row count, with `tightBounds: false`); delta-spark
  also computes the file's `minValues`, `maxValues` and `nullCount` by reading it.
- **Change data feed.** A vector-only `DELETE` commits; an `UPDATE` or a `MERGE` that changes rows
  stays refused (delta-spark writes change data files for them).

## Spark 4.3's nanosecond timestamps and nanosecond clocks

Spark 4.3.0 previews `TIMESTAMP_NTZ(7..9)` and `TIMESTAMP_LTZ(7..9)` behind the internal
`spark.sql.timestampNanosTypes.enabled` (off). With the key off Krishiv answers as Spark does:
`FEATURE_NOT_ENABLED` for a precision of 7 to 9 (also `current_timestamp(p)`, `now(p)`,
`localtimestamp(p)`), `INVALID_TIMESTAMP_PRECISION` for any precision but 6. Krishiv has no
nanosecond timestamp type yet (planned for 0.3.1), so:

- With `spark.sql.timestampNanosTypes.enabled=true`, a `TIMESTAMP*(7..9)` type is
  `KRISHIV_UNSUPPORTED_FEATURE` where Spark builds the type. `SET` and `RESET` of the key work under
  every profile; under the Spark 4.2 profile it changes nothing, as in Spark 4.2.
- A schema in Spark's JSON form naming `timestamp_ntz(p)` or `timestamp_ltz(p)` is
  `KRISHIV_UNSUPPORTED_FEATURE` (Spark's `DataType.fromJson` builds the type with the key off too).
  `timestamp_ntz(6)` and `timestamp_ltz(6)` there are the microsecond types under every profile;
  Spark 4.2's `DataType.fromJson` has no such names and refuses them.
- Spark Connect's `timestamp_ntz_nanos` and `timestamp_ltz_nanos` types and literals are refused
  as every arm Krishiv does not translate (`CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED`).

`TIME(p)` takes `p` up to 9 under the Spark 4.3 profile and keeps nanoseconds, but the query clock
is in microseconds: `current_time(7..9)` and `LOCALTIME(7..9)` have zero digits below the
microsecond, where Spark 4.3 reads a nanosecond clock.

## HLL sketches: the compact form (`hll_sketch_agg`, `hll_union_agg`, `hll_union`)

`hll_sketch_agg` returns the sketch in the compact serialized form; Spark returns the updatable one
(`HllSketch.toUpdatableByteArray`). Both are the Apache DataSketches `HLL_8` format, and both libraries read
both forms, so a sketch written by one engine can be estimated or unioned by the other, and
`hll_sketch_estimate`, `hll_union` and `hll_union_agg` give the same numbers. What differs is the bytes:
`length`, `hex`, `base64` or equality of an HLL sketch is not the same as Spark's. Compare sketches through
`hll_sketch_estimate` or `hll_union`, not by their bytes. A sketch built from collated strings
(`hll_sketch_agg(s COLLATE UTF8_LCASE)`) is refused: Spark hashes the collation key, Krishiv does not yet.

## Several grouped-aggregate pandas UDFs in one `agg`

Spark runs every grouped-aggregate pandas UDF of one `agg` in a single Python worker. Krishiv gives each UDF
its own Python UDF worker, because each is a task that holds its worker until its partition ends. On the
default (`krishiv.python.udf.threads` = 0, which is one worker on a GIL build, or when the value is set to 1),
`df.groupBy(k).agg(udf1(x), udf2(y))` with two such UDFs therefore fails with
`KRISHIV_PYTHON_WORKER_LIMIT` (SQLSTATE 54000), which names the setting. On a free-threaded build the default is the
executor thread count, so with two or more executor threads two grouped-aggregate UDFs can run without that error.
Raise `krishiv.python.udf.threads` to the number of grouped-aggregate UDFs in the `agg` plus one (one thread stays
free for scalar UDFs that feed them) and the query runs. The setting is startup-only: an existing session rejects a
runtime change with `CANNOT_MODIFY_STATIC_CONFIG`. The pool is made when the first Python UDF query starts and is not
resized, so put the value in the configuration passed when the session is created (or set it at engine startup, or
start a fresh interpreter with the higher value).
A single-thread pool starts one spare thread for scalar UDF calls (which still run one at a time), so a scalar UDF
call of another query does not wait for a running `mapInPandas` or `applyInPandas`. In one query a scalar
`pandas_udf` feeding such a node finishes before the node starts.

The embedded `mapInArrow` takes each result batch's columns by position and relabels them with the
declared schema; `mapInPandas` matches the result's columns to the declared schema by name, as Spark does. With `spark.sql.execution.arrow.pyspark.validateSchema.enabled` set to `true` the embedded
`mapInArrow` path compares the UDF's own field names with the declared schema and refuses a mismatch of names or
types with `ARROW_TYPE_MISMATCH`, as the server's worker pool does. A NULL in a result column declared `NOT NULL`
fails the query with `PYTHON_EXCEPTION`, where Spark 4.2.0's JVM fails reading the batch with no error class. A
nested `containsNull=false` or `valueContainsNull=false` is not kept by the embedded runtime, so a NULL element or
map value is accepted there.

## Metric views (`CREATE VIEW .. WITH METRICS LANGUAGE YAML`)

Metric views follow Spark 4.2.1 and 4.3.0 (goldens in `tests/fixtures/spark-{4.2.1,4.3.0}/metric-views/`), with
these differences:

- **Malformed YAML.** `INVALID_METRIC_VIEW_YAML` keeps Spark's `Failed to parse YAML:` prefix; the rest of the
  message is Krishiv's YAML parser's, not SnakeYAML's scanner text.
- **YAML Krishiv refuses where Spark reads on.** An alias to an anchor the document never defined is a YAML
  syntax error (Spark's binding reads the alias name as the value). A `name` or `expr` that is null is refused when
  the view is created, with a Krishiv message (Spark keeps it and fails later).
- **Spark internal errors, reproduced by class.** Where Spark fails with an internal error, Krishiv raises the same
  class and SQLSTATE (`INTERNAL_ERROR`, `XX000`, or the `MISSING_ATTRIBUTES.*` class) with Spark's message prefix;
  plan dumps, expression ids and Java exception text after it differ. This covers: a `CREATE VIEW` missing
  `WITH METRICS` or `LANGUAGE` under Spark 4.2 (which cannot find `MISSING_CLAUSES_FOR_OPERATION`; from 4.3 the
  class itself is raised, as in Spark); a read with no aggregate;
  `measure(<dimension>)` (`key not found: ExprId(..)`); a read through a view, a temporary view or a CTE over a
  metric view; `source: ~`; a constant dimension (`expr: 1`); a measure column read anywhere but as the argument of
  `measure(..)`.
- **Errors that are not Spark exceptions.** The 1,024-character expression limit, `measure(<dimension>)` and
  `SHOW CREATE TABLE .. AS SERDE` are plain JVM exceptions in Spark, with no error class: Spark Connect clients
  see `UnknownException` or `IllegalArgumentException`. Krishiv raises `INTERNAL_ERROR` with Spark's text, and its
  Connect server withholds the text of internal errors, as for any other.
- **`CANNOT_UP_CAST_DATATYPE` after a source column changed type.** The class, SQLSTATE and types are Spark's;
  the `expression` parameter is Krishiv's form of the dimension, where Spark qualifies its columns
  (`spark_catalog.default.t.k`).
- **Not matched yet:** `DESCRIBE EXTENDED .. AS JSON` of a metric view (refused, as for every view: Krishiv does not
  record the SQL configuration a view captures); a self-join of a metric view (Spark's
  `UNRESOLVED_COLUMN`, Krishiv's internal error); `WHERE measure(m)` under an aggregate (Spark's
  `MISSING_ATTRIBUTES.*`, Krishiv's `INVALID_WHERE_CONDITION`); the physical plan `EXPLAIN` prints.

## Pattern matching over collated strings (`LIKE`, `RLIKE`)

`LIKE` and `ILIKE` over a `UTF8_LCASE` string match as Spark does, character by character in Java's case folding
(final sigma, `ſ`, `ı`, `İ` and `µ` match their case partners), with or without an `ESCAPE` clause. The value
operand's collation alone decides: `'Ab' LIKE 'a%' COLLATE UTF8_LCASE` is false, as in Spark. Spark 4.0.0 differs
from 4.1 to 4.3 for `'ſ' ILIKE 'S'` over a column (false on 4.0); Krishiv follows 4.1 to 4.3. Still refused with a
`KRISHIV_UNSUPPORTED_FEATURE` error that names the call: `LIKE` over `UTF8_LCASE_RTRIM`, `LIKE` over an ICU collation
(Spark itself raises `DATATYPE_MISMATCH` there), `RLIKE` and the `regexp_*` family, and the `LIKE ANY`/`ALL` forms.
Krishiv never matches such strings by bytes.

Two smaller differences in `UTF8_LCASE` `LIKE`: case folding uses Rust's Unicode tables, which are newer than the JDK 17/21 tables Spark runs on, so a case pair added after Unicode 15 matches in Krishiv and not in Spark; and a malformed pattern (invalid or trailing escape) is reported with the case-folded pattern text, where Spark reports it as written. Java's matching is one-directional for a few characters (the pattern `ß` matches only `ß`, the pattern `ẞ` matches both), which a folded comparison cannot express: the value `ẞ` against the pattern `ß` matches in Krishiv and not in Spark.

## `NEAREST BY` joins run as a ranked cross join

Spark rewrites `JOIN ... (APPROX | EXACT) NEAREST [k] BY ...` to a `max_by`/`min_by` of `k` rows over a cross
join. Krishiv ranks the cross join's rows per left row with `row_number()` and keeps the first `k`; `APPROX` and `EXACT`
return the same exact rows. The results match Spark's, including NULL rankings (never a match; a left outer join pads
a left row left with none) and the `NEAREST_BY_JOIN.*` errors. `EXPLAIN` shows a window over a join, not Spark's
`Generate`/`Aggregate` shape, and the rewrite materialises every left-right pair before ranking, where Spark keeps a
top-k per left row. The `CROSS_JOIN_NOT_ENABLED` check runs when the statement is parsed, where Spark's runs in the
optimizer, so `CREATE VIEW v AS SELECT ... NEAREST BY ...` under `spark.sql.crossJoin.enabled=false` fails here and
succeeds in Spark.
