---
description: "UPDATE, DELETE and MERGE INTO on Delta tables: how to run them, what they cost, and what is refused."
---

# Changing rows in a Delta table

A Delta table takes `UPDATE`, `DELETE FROM` and `MERGE INTO` from SQL. Each statement is one Delta commit, and Spark
and other Delta readers see the result.

## Run a statement

```sql
CREATE TABLE trips (id BIGINT, region STRING, amount DOUBLE)
  USING delta PARTITIONED BY (region) LOCATION '/data/trips';
INSERT INTO trips VALUES (1, 'east', 10.0), (2, 'west', 20.0), (3, 'east', 30.0);

UPDATE trips SET amount = amount * 2 WHERE region = 'east';   -- num_affected_rows: 2
DELETE FROM trips WHERE region = 'west';                      -- num_affected_rows: 1
SELECT * FROM trips ORDER BY id;
```

```text
id,region,amount
1,east,20.0
3,east,60.0
```

Each statement returns `num_affected_rows`. The statements work on a table named in a catalog and on one addressed by
path, ``UPDATE delta.`/data/trips` SET ...``; a path needs the modify privilege on the path when
[access control](access-control.md) is on.

## Merge a batch of changes

`MERGE INTO` applies a source (a table, a view or a query) to the target in one commit. It takes `WHEN MATCHED`
(`UPDATE` or `DELETE`), `WHEN NOT MATCHED` (`INSERT`) and `WHEN NOT MATCHED BY SOURCE` clauses:

```sql
CREATE TEMPORARY VIEW updates AS SELECT * FROM VALUES
  (1, 'east', 99.0), (2, 'west', NULL), (4, 'north', 5.0) AS updates(id, region, amount);

MERGE INTO trips t USING updates s ON t.id = s.id
  WHEN MATCHED AND s.amount IS NULL THEN DELETE
  WHEN MATCHED THEN UPDATE SET t.amount = s.amount
  WHEN NOT MATCHED THEN INSERT (id, region, amount) VALUES (s.id, s.region, s.amount)
  WHEN NOT MATCHED BY SOURCE THEN UPDATE SET t.amount = 0;
```

On the three rows above the statement returns the four counts, and the table holds the rows `(1, east, 99.0)`,
`(3, east, 0.0)` and `(4, north, 5.0)`:

```text
num_affected_rows,num_updated_rows,num_deleted_rows,num_inserted_rows
4,2,1,1
```

A target row that matches more than one source row fails the statement with Delta's cardinality error
(SQLSTATE `21506`), as delta-spark does. A `MERGE` by path needs the modify privilege on the path. A source that is
not a bare scan of a Delta table is copied to a spill file first, so a large source costs disk. The
[known differences](../concepts/known-differences.md) page lists where the recorded commit differs from delta-spark's.

## How it runs

A `DELETE` without a condition, or with one on partition columns only, removes whole files and rewrites nothing. Otherwise
the write is copy-on-write: a data file that holds a changed row is rewritten without that row (or with the new
value), and the commit removes the old file and adds the new one. Files that hold no matching row are left alone, so
filter on a partition column to keep a change small. Deletion vectors are not written; a table that has them is read
correctly, but a change rewrites the whole file.

Writers commit with optimistic concurrency and retry after losing a race; `krishiv.write.commitRetryTimeout` bounds
all the attempts of one commit.

## What is refused

| You write | What happens |
|---|---|
| `UPDATE`, `DELETE` or `MERGE` while `krishiv.write.idempotencyKey` is set | `KRISHIV_UNSUPPORTED_FEATURE`; unset the key for the statement. See [make a retried write safe](lakehouse-tables.md#make-a-retried-write-safe) |
| `UPDATE`, or a `DELETE` that rewrites rows, on a table with the change data feed on | refused |
| A whole-file `DELETE` (no condition, or partition columns only) on such a table | commits |

See [Iceberg and Delta tables, and catalogs](lakehouse-tables.md) for the rest of what a Delta table supports.
