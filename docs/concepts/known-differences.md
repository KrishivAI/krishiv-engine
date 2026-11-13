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

## Result cache, reproducible results (Krishiv extensions, off by default)

`krishiv.resultCache.enabled` and `krishiv.deterministicOrder` add behaviour Spark does not have; with
both off (the default) nothing differs. See [result cache](../reference/result-cache.md) for
what is cached, when entries go, and the one staleness the cache accepts (a change made outside the
engine, until the entry expires or `REFRESH TABLE`).

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
  after the rows it needs: `SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) LIMIT 1`
  returns a row where Spark's shuffle first evaluates every row and raises `CAST_INVALID_INPUT`. A
  distributed run's map stage reads every row, so there it raises as Spark does.
- Rows go to partitions by Krishiv's routing hash, not Spark's `pmod(murmur3(exprs), n)`, so what
  depends on the partition numbers (`spark_partition_id()`) can differ.

Pinned by `crates/krishiv-session/tests/distribute_by_goldens.rs` (`DIFFERENT`), against a Spark
4.2.0 run (3.5, 4.0 and 4.1 not run).

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

## `krishiv server` (Spark Connect)

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
- **Results arrive in the engine's batches** (4,096 rows by default), cut further at Spark's
  `spark.connect.grpc.arrow.maxBatchSize` and `spark.sql.execution.arrow.maxRecordsPerBatch` but never merged.
- **Timestamps are labelled UTC when `spark.sql.session.timeZone` is unset**; Spark labels them with the JVM's zone.
  The instants are the same.
- **Geometry, geography and variant columns** get Spark's Arrow shape at the top level of a result only, not inside a
  struct, array or map.
- **The `metrics` response of ExecutePlan is empty** until the operator tree is wired through.
- **Reattaching to an operation that is still sending** answers `INVALID_CURSOR.DISCONNECTED` to the new stream, and
  ExecutePlan re-sent with a live operation id answers the same; Spark ends the old stream and replays from the start.
  The stock client reattaches only after a dropped connection, when Krishiv has already cancelled the operation.
- **Column metadata travels in the `schema` response only**: Spark 4.3 also writes each field's user metadata into
  the Arrow schema of every result batch (`SPARK::metadata::json`); Krishiv's batches carry none.
- **A connection that breaks while results are streaming fails the query**: Krishiv cancels the operation and the
  client's reattach answers `INVALID_HANDLE.OPERATION_NOT_FOUND`, where Spark keeps the sent responses and resumes the
  stream. Krishiv's replay buffer and resume come in a later release.
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
  reports the table-schema parse. `spark_version` is the session profile's release line (`4.2.0`). `is_local` is false
  for a SQL `VALUES` relation (`spark.sql("VALUES (1), (2)")`), which Spark answers true.
- **Config.** Secret-looking keys and values (Spark 4.3's `spark.redaction.regex` default) are hidden for 4.2 sessions
  too; Spark 4.2 shows them. The pattern is the default only (Spark reads `spark.redaction.regex` from the startup
  configuration). Krishiv also hides what its own secret classifier finds, where Spark 4.3 shows it: keys such as
  `fs.azure.account.key.<account>`, `sasl.jaas.config` or `*.apiKey`, and values with credentials in a URL
  (`postgres://user:password@host`). `set` of a key without a value fails with `INVALID_CONF_VALUE.REQUIREMENT`, and a request without an
  operation with `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET`; Spark sends `INTERNAL` without an error class.
  `is_modifiable` answers `true` for a key outside the engine's registry, where Spark answers `false` for a key it does
  not know. `get_all` lists a registered key under its Spark name when it has one, else under its `krishiv.*` name.
  Setting a key the registry marks admin-only answers `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501, gRPC
  `PERMISSION_DENIED`) for a non-admin role (no role exists before authentication, so the local user is the admin).
- **Interrupt by tag** fails with `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` until operation tags can be interrupted.
