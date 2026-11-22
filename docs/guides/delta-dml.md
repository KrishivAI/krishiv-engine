---
description: "UPDATE and DELETE on Delta tables: what runs, what it costs, and what is refused (MERGE INTO, deletion vectors, change data feed)."
---

# Changing rows in a Delta table

A Delta table takes `UPDATE` and `DELETE FROM` from SQL. Each statement is one Delta commit, and Spark and other Delta
readers see the result. `MERGE INTO` is not available in 0.3.

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

## How it runs

The write is copy-on-write: a data file that holds a changed row is rewritten without that row (or with the new
value), and the commit removes the old file and adds the new one. Files that hold no matching row are left alone, so
filter on a partition column to keep a change small. Deletion vectors are not written; a table that has them is read
correctly, but a change rewrites the whole file.

Writers commit with optimistic concurrency and retry after losing a race; `krishiv.write.commitRetryTimeout` bounds
all the attempts of one commit.

## What is refused

| You write | What happens |
|---|---|
| `MERGE INTO` a Delta table | `UNSUPPORTED_FEATURE.TABLE_OPERATION` |
| `UPDATE` or `DELETE` while `krishiv.write.idempotencyKey` is set | `KRISHIV_UNSUPPORTED_FEATURE`; unset the key for the statement. See [make a retried write safe](lakehouse-tables.md#make-a-retried-write-safe) |
| `UPDATE` or `DELETE` on a table with the change data feed on | refused |

Put the new rows of a merge in a staging table and run an `UPDATE` and an `INSERT` for now.

See [Iceberg and Delta tables, and catalogs](lakehouse-tables.md) for the rest of what a Delta table supports.
