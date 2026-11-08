<!-- SPDX-License-Identifier: Apache-2.0 -->
# Known differences from Spark

Deliberate, documented divergences of Krishiv's engine from Apache Spark.
The register of record lives with ADR 0023 outside this repository; the
entries below are the engine's own copy so each divergence is visible next
to the code that implements it.

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

## Result cache, reproducible results (Krishiv extensions, off by default)

`krishiv.resultCache.enabled` and `krishiv.deterministicOrder` add behaviour Spark does not have; with
both off (the default) nothing differs. See [reference/result-cache.md](reference/result-cache.md) for
what is cached, when entries go, and the one staleness the cache accepts (a change made outside the
engine, until the entry expires or `REFRESH TABLE`).

## A malformed value of a Spark config key fails at `SET`

A Spark key that Krishiv maps to a typed setting (for example
`spark.sql.autoBroadcastJoinThreshold`) is checked when it is set: a value
that does not parse fails the `SET` with `KRISHIV_INVALID_CONFIG_VALUE`.
Spark stores the string and fails when a query reads it, with
`INVALID_CONF_VALUE.TYPE_MISMATCH` (SQLSTATE 22022).

## `DISTRIBUTE BY` and `CLUSTER BY` do not repartition

`DISTRIBUTE BY exprs` (Spark's `RepartitionByExpression`, also under `CLUSTER BY` and `SORT BY ...
DISTRIBUTE BY`) is checked as Spark checks it: the expressions resolve, and aggregates, windows,
generators and outer references are refused with Spark's errors. The repartition itself is then
dropped, as the `REPARTITION`-family hints are. The rows are the same, but two things differ:

- Spark evaluates the expressions while it shuffles, so one that fails at run time fails the query
  there (`SELECT x FROM t DISTRIBUTE BY x + 'a'` raises `CAST_INVALID_INPUT` under ANSI). Krishiv
  never evaluates them and returns the rows.
- On a distributed run the rows are not hash-partitioned by the expressions, so what depends on the
  partitioning (`spark_partition_id()`, the order `SORT BY` gives within each partition) can differ.

Pinned by `crates/krishiv-session/tests/distribute_by.rs` (`the_expressions_are_not_evaluated`); a real
exchange that evaluates them is queued for the execution lane.

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

## `schema_of_variant` with `spark.sql.caseSensitive=true`

Spark's `schema_of_variant` merges two objects of the same size whose sorted field names match under the session's
name resolver (`findTypeForComplex`), keeping the first object's names: `[{"A":1},{"a":2}]` is
`ARRAY<OBJECT<A: BIGINT>>` with the default `spark.sql.caseSensitive=false`, and Krishiv gives the same. The function
does not read the setting: with it on, Spark keeps both spellings (`ARRAY<OBJECT<A: BIGINT, a: BIGINT>>`) and Krishiv
still merges them.

## `PIVOT` cases Spark answers from undefined state are refused

When every aggregate of a `PIVOT` has a numeric, boolean or decimal type, Spark takes its `PivotFirst` path and
Krishiv reproduces it, including Spark's slot map (a NaN value never matches, `0.0` and `-0.0` are one value,
repeated values raise `INVALID_ARRAY_INDEX` with ANSI on and leave NULL cells with ANSI off). A few cases fail with
`KRISHIV_UNSUPPORTED_FEATURE` instead of returning a result:

- repeated pivot values with ANSI off and several aggregates: Spark writes the repeated value's result into the
  next aggregate's cells;
- several pivot values of a type the analyzer does not evaluate (timestamp, interval, time) when one of them is not
  a literal of the pivot column's type, for example `IN (TIMESTAMP'2020-01-01 00:00:00', '2021-01-01 00:00:00')`:
  Krishiv cannot tell whether two of them are the same value, which decides Spark's result;
- a collated pivot column with Spark 4.0 or 4.1 compatibility: those versions match the values by their bytes,
  not by the collation;
- a NULL pivot value for a binary or collated pivot column: Spark 4.2 compares it with the other values in a sorted
  map (and fails), earlier versions look it up in a hash map.

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
