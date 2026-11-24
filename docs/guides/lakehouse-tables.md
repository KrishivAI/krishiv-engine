---
description: "What Krishiv reads and writes in Iceberg and Delta tables, how to declare an Iceberg REST catalog, and idempotent writes."
---

# Iceberg and Delta tables, and catalogs

Krishiv reads Apache Iceberg tables and reads and writes Delta Lake tables without a JVM. This page covers what each format
supports in 0.3, how to declare an external catalog, and how to make a retried write safe.

## Delta tables

A Delta table is addressed by path or by a catalog name:

```sql
CREATE TABLE trips (id BIGINT, region STRING, amount DOUBLE) USING delta LOCATION 's3://bucket/trips';
INSERT INTO trips VALUES (1, 'east', 10.0);
SELECT region, sum(amount) FROM delta.`s3://bucket/trips` GROUP BY region;
```

| Operation | Support |
|---|---|
| Read, including partitioned tables | yes |
| `INSERT`, CTAS, `df.write.format("delta").save`, `saveAsTable`, `insertInto`, `writeTo` | yes |
| `DELETE`, `UPDATE` and `MERGE INTO` | yes, copy-on-write (a changed file is rewritten); see [Delta DML](delta-dml.md) |
| Deletion vectors on write | not written; rows are rewritten |
| Change data feed read | not yet |

Delta support is the `delta` cargo feature. Of the release artifacts only the Docker image is built with it; the wheels and the release `krishiv` binary are not
(see [What each artifact includes](../get-started/install.md#what-each-artifact-includes)). A build
without it refuses a Delta table with
`DATA_SOURCE_NOT_FOUND`: write one small table to check yours.

A table that uses a protocol feature Krishiv cannot read is refused with Spark's own error
(`DELTA_UNSUPPORTED_FEATURES_FOR_READ`) instead of being read wrongly. Writers commit with optimistic concurrency and
retry after losing a race; `krishiv.write.commitRetryTimeout` bounds all the attempts of one commit.

A Delta write is single-writer in 0.3: the rows of one statement are gathered to one task, which writes every data
file and makes the commit. The scans and the rest of the query before the write still run in parallel. Writing a
table's files from several workers, with one commit made by the coordinator, comes with distributed mode.

## Iceberg tables

An Iceberg table is read through a [declared catalog](#declare-a-catalog) (`SELECT * FROM lake.sales.orders`).
Reads cover position deletes, deletion vectors and equality deletes.

Writing Iceberg tables is not available from SQL or the DataFrame API in 0.3: `INSERT`, `CREATE TABLE ... USING iceberg`
and the other writes do not reach the Iceberg writer yet. The session does not route a write statement to a declared catalog's table in 0.3, so every Iceberg table is read-only. `INSERT INTO lake.ns.t` is refused (`REQUIRES_SINGLE_PART_NAMESPACE`). Use Spark, or write the table as Delta or Parquet.

## Declare a catalog

A catalog is declared by settings. In server mode they are startup settings; an embedded session sets them on its
builder. The same keys under Spark's `spark.sql.catalog.<name>` prefix are accepted.

```text
krishiv.catalog.lake.type          = iceberg-rest
krishiv.catalog.lake.uri           = https://catalog.example.com/api/catalog
krishiv.catalog.lake.warehouse     = analytics
krishiv.catalog.lake.storage       = allowed-prefixes
krishiv.catalog.lake.auth          = client-credentials
krishiv.catalog.lake.clientId      = krishiv
krishiv.catalog.lake.clientSecret  = env:LAKE_SECRET
krishiv.catalog.lake.tokenEndpoint = https://idp.example.com/oauth2/token
krishiv.catalog.default            = lake
```

The only catalog type `krishiv server` serves in 0.3 is `iceberg-rest`, in a binary built with the
`catalog-iceberg-rest` feature. The other values (`hive`, `unity`, `glue`, `sqlite`, `memory`) are recognised in a
declaration but a server refuses to start with one. Catalog storage must be `allowed-prefixes`: `vended` credentials
are not supported yet, and a table of a catalog that is left on the default `vended` fails with
`KRISHIV_UNSUPPORTED_FEATURE`.

| Key | Meaning |
|---|---|
| `type`, `uri`, `warehouse` | the catalog kind, its endpoint and its warehouse |
| `auth` | `none` (default), `token`, `client-credentials` or `delegated` |
| `token`, `clientSecret` | a secret by reference: `env:NAME` or `file:/path`, never the value |
| `delegation` | with `auth = delegated`: `exchange` (RFC 8693 token exchange, default) or `forward` (the caller's own token); needs `audience` |
| `storage` | `allowed-prefixes` (set it); `vended` (the catalog hands out credentials) is not supported yet |
| `tenants` | the tenants that may use the catalog; absent means every tenant |
| `allowInsecure` | allows an `http://` uri, server mode only |

With `auth = delegated` every request to the catalog carries the calling user's identity, so the catalog decides what
that user may see. Declaring a catalog never connects: a catalog that is down does not stop the server. The names
`delegation` and `vended` are reserved for engine settings. `krishiv.catalog.requestTimeout`, `maxMetadataBytes` and
`maxColumns` bound one request, one piece of metadata and one schema.

## Make a retried write safe

A job that retries a statement after a timeout can append the same rows twice. Give the logical write a key and the
retry writes nothing:

```python
spark.conf.set("krishiv.write.idempotencyKey", "orders-2026-11-23")
df.write.format("delta").mode("append").save("s3://bucket/orders")
spark.conf.unset("krishiv.write.idempotencyKey")
```

A write whose key is already recorded in the target table's commits succeeds and writes nothing. The key is scoped to
the table, lasts as long as the commit that holds it, and is set per logical write: while it stays set, every later
write to the same table with the same key writes nothing. `UPDATE`, `DELETE` and `MERGE` refuse a key set on the
session.
