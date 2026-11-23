---
description: "Quick answers: reading from S3 and CSV files, Delta and Iceberg, the Spark version, streaming, the session time zone, ANSI mode and upgrade notes."
---

# Common questions

Each answer is short and links to the page that has the detail.

## How do I read data from S3?

Use an `s3://` (or `s3a://`) path: `spark.read.parquet("s3://my-bucket/events/")`. Google Cloud Storage and Azure Blob Storage
work the same way. Credentials and endpoint settings are in [Data sources](../guides/data-sources.md).

## How do I read a CSV file?

`spark.read.csv("trips.csv", header=True, inferSchema=True)`. `spark.read.json` and `spark.read.parquet` work the same
way, and `df.write` has matching writers. See [Data sources](../guides/data-sources.md).

## Is Delta Lake supported?

Yes. Krishiv reads and writes Delta tables, including `INSERT`, `DELETE`, `UPDATE` and `MERGE`, without a JVM. See
[Iceberg and Delta tables, and catalogs](../guides/lakehouse-tables.md) and
[Changing rows in a Delta table](../guides/delta-dml.md).

## Is Iceberg supported?

Yes. Krishiv reads Iceberg tables and can use an Iceberg REST catalog you declare. What each format supports is a table in
[Iceberg and Delta tables, and catalogs](../guides/lakehouse-tables.md).

## Which Spark version does Krishiv match?

Krishiv follows Spark 4.2 behaviour; the stock PySpark client connects over Spark Connect. The per-surface status is in
[Compatibility](../concepts/compatibility.md) and every deliberate difference is in
[Known differences](../concepts/known-differences.md).

## Does Krishiv support streaming?

No. Structured Streaming APIs are not available, and `CREATE STREAMING TABLE` is refused with `UNSUPPORTED_FEATURE`, as in
Spark. The [Compatibility](../concepts/compatibility.md) page lists what is not yet covered.

## How do I set the session time zone?

Set `krishiv.sql.session.timeZone` (or its Spark alias `spark.sql.session.timeZone`) to a region ID such as
`America/Los_Angeles` or an offset: `spark.conf.set("spark.sql.session.timeZone", "UTC")`. When unset, the local time zone
applies. All settings are in the [configuration reference](../reference/config/index.md).

## Is ANSI mode on?

Yes. `krishiv.sql.ansi.enabled` (alias `spark.sql.ansi.enabled`) defaults to `true`: invalid casts and arithmetic overflow
raise errors instead of returning NULL. Set it to `false` to turn it off. See the
[configuration reference](../reference/config/index.md) and [Known differences](../concepts/known-differences.md).

## What do I check when I upgrade?

Read the [release notes](../releases/index.md) for each version between yours and the new one, and the
[API stability](../concepts/stability.md) page: APIs are not stable yet.

## Next steps

[Quickstart](quickstart.md) · [Guides](../guides/index.md) · [Reference](../reference/index.md)
