---
description: "The SQL statements Krishiv runs, with an example of each that was run on Krishiv 0.2, and what is not supported yet."
---

# SQL statements

Statements are Spark SQL; semantics and error classes follow Spark. This page lists what Krishiv 0.2 runs, with the
statement forms that were checked by running them with `krishiv sql -e`. Functions and operators are in the
[function reference](functions/index.md); known gaps are in [known differences](../../concepts/known-differences.md).

## Queries

`SELECT`, `WITH` (CTEs), joins, aggregates, window functions, `VALUES`, `LATERAL`, set operations and table
functions such as `range(5)` run as in Spark.

```sql
SELECT * FROM t ORDER BY id;
```

`EXPLAIN` prints the physical plan; `EXPLAIN ANALYZE` runs the query and adds per-operator rows, time and bytes
(see [EXPLAIN ANALYZE](../explain-analyze.md)).

```sql
EXPLAIN SELECT * FROM t;
EXPLAIN ANALYZE SELECT * FROM t;
```

## Namespaces and tables

```sql
CREATE NAMESPACE IF NOT EXISTS demo;
USE demo;
SHOW NAMESPACES;
CREATE TABLE t (id INT, name STRING);        -- Parquet by default
CREATE TABLE t2 LIKE t;
ALTER TABLE t ADD COLUMNS (age INT);
DESCRIBE t;
SHOW TABLES;
SHOW CREATE TABLE t;
TRUNCATE TABLE t2;
DROP TABLE t2;
DROP NAMESPACE demo CASCADE;                 -- without CASCADE a non-empty namespace is refused (SCHEMA_NOT_EMPTY)
```

## information_schema

`spark_catalog.information_schema` (also `information_schema` while `spark_catalog` is current) has three
read-only views over the session catalog. Spark has no `information_schema`, so the column names are the
SQL-standard ones.

| View | Columns |
|---|---|
| `schemata` | `catalog_name`, `schema_name` |
| `tables` | `table_catalog`, `table_schema`, `table_name`, `table_type` (`BASE TABLE` or `VIEW`) |
| `columns` | `table_catalog`, `table_schema`, `table_name`, `column_name`, `ordinal_position` (from 1), `data_type` (Spark's simple string), `is_nullable` (`YES` or `NO`) |

```sql
SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'VIEW';
SELECT column_name, data_type FROM information_schema.columns
 WHERE table_schema = 'demo' AND table_name = 't' ORDER BY ordinal_position;
```

Rows follow the `SHOW TABLES` listing rule: an object the principal may not list is absent, as it is there.
Temporary views and declared catalogs are not listed yet.

## Writing data

```sql
INSERT INTO t VALUES (1, 'a'), (2, 'b');
INSERT OVERWRITE t SELECT 9, 'z';
```

`UPDATE`, `DELETE FROM` and `MERGE INTO` parse, but a table in the default session catalog (Parquet files) does not
support them: they fail with `UNSUPPORTED_FEATURE.TABLE_OPERATION`, as Spark does for a table without row-level
support. They need a table format that supports row-level changes.

## Views

```sql
CREATE TEMPORARY VIEW v AS SELECT * FROM t;
CREATE VIEW pv AS SELECT id FROM t;
DROP VIEW v;
SHOW VIEWS;
```

`ALTER VIEW ... RENAME TO` is not supported yet (`KRISHIV_UNSUPPORTED_FEATURE`).

## Variables, settings and scripting

```sql
DECLARE VARIABLE x INT DEFAULT 5;
SET VAR x = 7;
SELECT x;
DROP TEMPORARY VARIABLE x;
SET spark.sql.ansi.enabled = true;
RESET spark.sql.ansi.enabled;
EXECUTE IMMEDIATE 'SELECT 1';
```

## Caching and statistics

```sql
CACHE TABLE t;
REFRESH TABLE t;
ANALYZE TABLE t COMPUTE STATISTICS;
```

## Not supported yet

- `CREATE FUNCTION ... AS 'class'` (a function implemented by a JVM class): `KRISHIV_JVM_FUNCTION_UNSUPPORTED`. Register
  Python functions with `@udf` and Rust functions through the [Rust UDF guide](../rust-udfs.md).
- `ADD JAR | FILE | ARCHIVE` in a session: `KRISHIV_UNSUPPORTED_FEATURE`.
- `ALTER VIEW ... RENAME TO`, as above.
- A `BEGIN ... END` scripting block did not run through `krishiv sql -f` in 0.2 (`PARSE_SYNTAX_ERROR`).
