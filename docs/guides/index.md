---
description: "Task-by-task guides for migrating, checking, deploying and diagnosing Krishiv."
---

# Guides

Each guide solves one task and is written for someone who already ran the [Quickstart](../get-started/quickstart.md).
Moving a Spark job? Start with [Migrate a PySpark job](pyspark-migration.md) and check your functions against the
list there. Choosing where Krishiv runs: [Deployment](deploy.md)
(embedded, single-node, distributed). To work with it interactively, see
[Run SQL from the command line](run-sql-from-the-command-line.md) and the
[notebooks and sample data](notebooks-and-datasets.md); to debug a stalled task, [tokio-console](tokio-console.md).

## All guides

**Migrate from Spark**
- [Migrate a PySpark job](pyspark-migration.md): unsupported functions, workarounds, behaviour differences.
- [Check a project with check-compat](check-compat.md): scan a PySpark or SQL project for functions that differ from Spark.
- [Migrating JVM UDFs](migrating-jvm-udfs.md).

**Read and write data**
- [Data sources](data-sources.md): Parquet and CSV files on local disk, S3, Google Cloud Storage and Azure Blob Storage.
- [Iceberg and Delta tables, and catalogs](lakehouse-tables.md): Iceberg REST catalogs, Delta tables, idempotent writes.
- [Changing rows in a Delta table](delta-dml.md): UPDATE, DELETE and MERGE INTO.
- [Python data sources](python-data-sources.md): your own formats through `spark.read.format`.

**SQL features**
- [As-of joins](asof-join.md), [Metric views](metric-views.md), [The TIME type](time-types.md),
  [Vector search with SQL](vector-search.md), [Collations and ICU](collations-and-icu.md).
- [Run SQL from the command line](run-sql-from-the-command-line.md).

**Python UDFs and table functions**
- [Python table functions](python-table-functions.md), [Profile a Python UDF](python-udf-profiler.md),
  [Python UDFs in the server: the sandbox](python-udf-sandbox.md), [Python UDFs per tenant](python-udf-tenants.md).

**Run and operate**
- [Deployment](deploy.md): embedded, single-node and distributed.
- [Connect stock PySpark to an in-process server](embedded-connect.md) and [Arrow Flight SQL](flight-sql.md).
- [Securing a single-node server](single-node-server-security.md), [Access control with grants](access-control.md),
  [Check a server configuration with krishiv doctor](krishiv-doctor.md), [Resource groups and request tags](resource-groups.md).
- [Query events](query-events.md), [Observability from Rust](observability.md), [tokio-console](tokio-console.md).
- [Notebooks and sample data](notebooks-and-datasets.md).
