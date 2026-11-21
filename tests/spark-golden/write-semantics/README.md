# Spark 4.3.0 write semantics (P3-T023)

Status: **observed on a real Spark 4.3.0 server for the scripted cases**. `tests/scripts/gen_write_semantics_goldens.py
--remote sc://... --version 4.3.0` ran against a Spark Connect 4.3.0 server and wrote the `observed` block of every case
(the server build and the run date were not recorded; the files were committed 2026-11-20). Cases the server could
not script carry `observed.kind = "skipped"` with a reason: the V2 cases need a V2 catalog (`testcat`), the time-travel
cases are DataFrame-API only, and two `saveAsTable` mode-string cases are classic-only. For those, `expect` is derived
from apache/spark at tag `v4.3.0` (commit 98bc0212c) and the suite assertions cited per case, with nothing run.
Where a scripted case's `observed` differs from the source-derived text, `expect` is set to the run's result and the
old text is kept as `spark_expected_by_source` (`DML-hv-textfile-delete-alias-opts`, `DML-nowith-4.2-grammar`,
`SAT-50-existing-view-ignore`). `SAT-43-append-self-read` differs only in row order (rows are compared unordered).

Files: `replace_where.json` (40 cases), `dml_with_options.json` (26), `time_travel_writes.json` (20),
`save_as_table_modes.json` (51 cases, 20 of them the full mode x existing/missing x partitioned/flat matrix).
Case schema: `id`, setup (`setup` SQL or `existing`), statement (`stmt` or `write`), `expect`, `source` (path:line at
v4.3.0), `compat` (all `compatible`: nothing here needs `allowIncompatible`), optional `conf`, `note`.

## Rules (all line numbers at v4.3.0)

1. **REPLACE WHERE on a V1 table (Parquet, partitioned Parquet, Hive-serde parquet/textfile) is always rejected**:
   `UNSUPPORTED_FEATURE.TABLE_OPERATION`, SQLSTATE `0A000`, `tableName` = `` `spark_catalog`.`default`.`t` ``,
   `operation` = `INSERT INTO ... REPLACE WHERE`. `datasources/rules.scala:471-473` (PreprocessTableInsertion, runs
   before HiveAnalysis, `HiveSessionStateBuilder.scala:146-152`), `QueryCompilationErrors.scala:710-712`; suite proof on a
   V1 table: `InsertSuite.scala:998-1030`. The table is unchanged. `INSERT WITH SCHEMA EVOLUTION` is checked first
   (`rules.scala:467-469`) -> `UNSUPPORTED_INSERT_WITH_SCHEMA_EVOLUTION`. REPLACE ON/USING gets the operation text
   `INSERT INTO ... REPLACE ON/USING`.
2. **Column list** (SPARK-58065): grammar `SqlBaseParser.g4:600-601`
   (`identifierReference tableAlias optionsClause? ((BY NAME) | identifierList)? REPLACE (WHERE|ON) cond`); parsed in
   `AstBuilder.scala:1130-1152`; gated by `spark.sql.insertIntoReplaceWhereColumnList.enabled` (internal, default
   true, `SQLConf.scala:5784-5791`); off -> parse error `INSERT_REPLACE_WHERE_COLUMN_LIST_NOT_ENABLED` (0A000). On V1
   tables the rule-1 error wins over any column-list problem. A table alias is rejected at parse time:
   `INSERT_REPLACE_WHERE_TABLE_ALIAS_NOT_ALLOWED` (42000), `AstBuilder.scala:1132-1136`.
3. **REPLACE WHERE on DSv2**: becomes `OverwriteByExpression(cond)` (`Analyzer.scala:1465-1490`), never dynamic
   partition overwrite even with `partitionOverwriteMode=dynamic` (`Analyzer.scala:1447-1450`). Rows matching the
   condition on the target are deleted, all source rows are inserted (they need not satisfy it). Column list maps
   source columns by position to the listed columns; omitted columns take their default (NULL); struct fields map by
   position. Errors: `INSERT_COLUMN_ARITY_MISMATCH.{NOT_ENOUGH,TOO_MANY}_DATA_COLUMNS` (21S01),
   `UNRESOLVED_COLUMN.WITH_SUGGESTION`, `COLUMN_ALREADY_EXISTS`, `INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_SAFELY_CAST`,
   `UNSUPPORTED_FEATURE.OVERWRITE_BY_SUBQUERY`; all from `DataSourceV2SQLSuite.scala:4570-4921` (the result rows there are
   suite assertions, not runs of ours). A temp view target: `VIEW_WRITE_NOT_ALLOWED` (`Analyzer.scala:1203-1205`).
4. **WITH (...) on UPDATE / MERGE / DELETE** (SPARK-57681/58007/58008): grammar `SqlBaseParser.g4:750-754`
   (`DELETE FROM t [alias] [WITH (..)] [WHERE]`, `UPDATE t [alias] [WITH (..)] SET`, `MERGE INTO t [WITH (..)] ... USING
   s [WITH (..)]`), `AstBuilder.scala:1245-1275,1284-1385`. Target options live on an `UnresolvedRelation` with write
   privileges; the MERGE source options are a separate read bag. On Parquet/Hive V1 tables the options are accepted
   and the statement fails exactly as it does without them: DELETE `UNSUPPORTED_FEATURE.TABLE_OPERATION` operation
   `DELETE` (`DataSourceV2Strategy.scala:550-583`; Hive-serde: `HiveStrategies.scala` HiveAnalysis), UPDATE operation
   `UPDATE TABLE`, MERGE operation `MERGE INTO TABLE` (`SparkStrategies.scala:1251-1256`, raised at planning as
   SparkUnsupportedOperationException, 0A000). Table name `` `spark_catalog`.`default`.`t` `` for a LogicalRelation, but
   `` `unknown` `` for an unconverted `HiveTableRelation` (`SparkStrategies.scala:1282-1295`): textfile Hive tables are
   flagged as source-derived, to be confirmed by a run. On DSv2 the options reach the delete/row-level operation
   (`RewriteRowLevelCommand.scala:78-84`).
5. **Time travel on writes**: any of the two keys `timestampAsOf` / `versionAsOf` (names from
   `spark.sql.timeTravelTimestampKey` / `...VersionKey`, `SQLConf.scala:7723-7735`; matching is case-insensitive) in the
   write options of a SQL `WITH (...)` or a DataFrameWriter/V2 `option(...)` on a write target makes it fail with
   `UNSUPPORTED_FEATURE.TIME_TRAVEL` (0A000), message `Time travel on the relation: <relationId>.`;
   `relationId` = `toSQLId(catalog +: namespace :+ name)`, e.g. `` `testcat`.`ns1`.`ns2`.`table` ``, `` `temp_view` `` for a
   temp view. Sites: `CatalogV2Util.scala:595-610`, `RelationResolution.scala:165-176,248-258`, `ResolveCatalogs.scala:130-141`
   (CTAS/RTAS incl. `writeTo().create()`/`replace()`/`createOrReplace()` even for a new table),
   `DataFrameWriter.scala:361-370,508`; suite `DataSourceV2OptionSuite.scala:1839-1896`. The check does not parse the
   value, so a bad timestamp still gives this error. A missing target gives `TABLE_OR_VIEW_NOT_FOUND` first. SQL
   `OPTIONS(...)` on CTAS/RTAS are table properties and are not rejected. There is no `VERSION AS OF` clause on a write.
   **V1 DataFrameWriter paths ignore the options** (`insertInto`/`saveAsTable` fall back to V1 before the check;
   `DataSourceV2DataFrameSessionCatalogSuite.scala:102-125`), but a SQL write with `WITH (...)` to a V1 session-catalog
   table is rejected by `tryResolvePersistent` (`TT-v1-spark-catalog-sql`, derived, no suite).
6. **saveAsTable on the default (V1, `spark_catalog`) path** goes to `SaveAsV1TableCommand` (`ddl.scala:1025-1075`):
   - table missing: all five modes create it and write the rows;
   - existing + `error`/`errorifexists`/`default`: `TABLE_OR_VIEW_ALREADY_EXISTS` (42P07) `` `spark_catalog`.`default`.`t` ``
     (thrown at execution; `ddl.scala:1035-1036`); mode strings are lower-cased, `sql/api/.../DataFrameWriter.scala:58-66`;
   - existing + `ignore`: nothing happens, before any validation;
   - existing + `overwrite`: drop and recreate (schema, provider and partitioning follow the writer; all old data and
     partitions go regardless of `partitionOverwriteMode`); reading the target in the query ->
     `UNSUPPORTED_OVERWRITE.TABLE` (42902);
   - existing + `append`: by name; provider, partition columns, bucketing must match (`_LEGACY_ERROR_TEMP_1159/1161/1162/1163/1164`,
     message text only, `tableName` is `spark_catalog.default.t`), then ANSI store assignment (string -> int is
     `INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_SAFELY_CAST`; numeric narrowing allowed, overflow errors at run time).
7. A custom V2 session catalog or a V2 provider uses a different path (`DataFrameWriter.scala:477-550`:
   `CreateTableAsSelect`/`ReplaceTableAsSelect`/`AppendData.byName`); out of scope here, needs the Iceberg/Delta oracles.

## Per-version differences

- Spark 3.5.8 / 4.1 (4.0 not separately checked, expected same): no `REPLACE WHERE`, no `WITH` on UPDATE/MERGE/DELETE (grammar checked at v3.5.8 `:445-446`, v4.1.0
  `:677-678`); `rejectTimeTravelOptionsForWrite` does not exist (0 hits in `CatalogV2Util.scala` at v3.5.8/v4.1.0/v4.2.0).
- Spark 4.2.0: `REPLACE WHERE` exists in the grammar (`:593`) with no column list; `AstBuilder.scala:960-990` builds
  `OverwriteByExpression` directly; V1 rejection text in `rules.scala:470-471` is `unsupportedInsertReplaceOnOrUsing`
  (no `REPLACE WHERE` operation). UPDATE/DELETE have no `WITH` (`:742-743`). What 4.2 does for REPLACE WHERE on Parquet was
  not derived (unverified).
- Spark 4.3.0: everything in this directory. Behaviour is keyed by Spark version, so none of it applies to 4.2 and below.

## Cannot match / compat

No case needs `krishiv.expr.<name>.allowIncompatible`. Cases with `target: dsv2 ...` cannot be oracle-checked on stock
Spark (no V2 catalog); use the Iceberg 1.12 / Delta runtimes of the plan and re-derive `expect` from those runs.

## Regeneration

`gen_write_semantics_goldens.py` reads these input files and writes only the `observed` blocks to
`tests/fixtures/spark-<ver>/write-semantics/`; it never rewrites an input file. The curation step is by hand: where a
run differs from the source-derived text, set `expect` to the run's result and keep the old text beside it
(`spark_expected_by_source`, `old_expect`). The script refuses a server whose version is not `--version`, creates a
uniquely named scratch database per case (never dropping one it did not create), restores the session configuration
after each case, records field nullability, skips the classic-only writer mode strings and all V2 cases, and, for a
refused statement whose case states the table's rows, records the rows read back (`after`). `TT-v1-spark-catalog-sql`
now has a setup and a statement, so a run covers it; the fixture keeps its `skipped` record until one is made.
