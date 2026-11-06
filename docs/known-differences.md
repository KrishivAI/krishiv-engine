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
