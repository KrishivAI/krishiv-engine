---
title: Iceberg and Delta tables, and catalogs
---

# Iceberg and Delta tables, and catalogs

Krishiv reads and writes Apache Iceberg and Delta Lake tables without a JVM. This page covers what each format
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
| `DELETE` and `UPDATE` | yes, copy-on-write (a changed file is rewritten) |
| Deletion vectors on write | not written; rows are rewritten |
| Change data feed read | not yet |

A table that uses a protocol feature Krishiv cannot read is refused with Spark's own error
(`DELTA_UNSUPPORTED_FEATURES_FOR_READ`) instead of being read wrongly. Writers commit with optimistic concurrency and
retry after losing a race; `krishiv.write.commitRetryTimeout` bounds all the attempts of one commit.

## Iceberg tables

An Iceberg table is read through a [declared catalog](#declare-a-catalog).

| Operation | Support |
|---|---|
| Read, including position deletes, deletion vectors and equality deletes | yes |
| `INSERT`, CTAS | yes |
| `INSERT OVERWRITE`: whole table, by filter, and dynamic partition overwrite | yes |
| `DELETE` and `UPDATE` | yes, merge-on-read (delete files; data files are not rewritten) |
| Commit through an Iceberg REST catalog | yes; `krishiv.iceberg.commit.maxIntervening` bounds how many snapshots may land on the branch meanwhile |

`krishiv.iceberg.write.statistics.enabled` (default `false`) makes an append also write a Theta sketch and a Puffin
statistics file, which later queries use for distinct-count estimates.

## Declare a catalog

A catalog is declared by settings. In server mode they are startup settings; an embedded session sets them on its
builder. The same keys under Spark's `spark.sql.catalog.<name>` prefix are accepted.

```text
krishiv.catalog.lake.type          = iceberg-rest
krishiv.catalog.lake.uri           = https://catalog.example.com/api/catalog
krishiv.catalog.lake.warehouse     = analytics
krishiv.catalog.lake.auth          = client-credentials
krishiv.catalog.lake.clientId      = krishiv
krishiv.catalog.lake.clientSecret  = env:LAKE_SECRET
krishiv.catalog.lake.tokenEndpoint = https://idp.example.com/oauth2/token
krishiv.catalog.default            = lake
```

Types are `iceberg-rest`, `hive`, `unity`, `glue`, `sqlite` and `memory`. Reading is
`SELECT * FROM lake.sales.orders`.

| Key | Meaning |
|---|---|
| `type`, `uri`, `warehouse` | the catalog kind, its endpoint and its warehouse |
| `auth` | `none` (default), `token`, `client-credentials` or `delegated` |
| `token`, `clientSecret` | a secret by reference: `env:NAME` or `file:/path`, never the value |
| `delegation` | with `auth = delegated`: `exchange` (RFC 8693 token exchange, default) or `forward` (the caller's own token); needs `audience` |
| `storage` | `vended` (the catalog hands out storage credentials) or `allowed-prefixes` |
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
