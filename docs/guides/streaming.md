---
description: "Whether Krishiv runs Structured Streaming today, what to use instead, and what a streaming job needs from a migration."
---

# Streaming: what works today

Krishiv does not run Structured Streaming yet. This page says what that means for a job you want to move, so you can
decide before you start.

## What is not available

- `spark.readStream`, `DataFrame.writeStream`, `StreamingQuery` and `StreamingQueryManager`.
- Streaming sources and sinks such as Kafka, and `trigger`, `outputMode` and `checkpointLocation` options.
- Watermarks and stateful streaming operators (`withWatermark`, streaming aggregations, stream-stream joins).
- A Spark Connect `Read` marked as streaming: the server refuses it with an error rather than running it as a batch.

The [Compatibility](../concepts/compatibility.md) table lists the streaming APIs under "not yet", and
[Known differences](../concepts/known-differences.md) records how the batch-only engine treats streaming-only calls such as
`observe` on a stream.

## What you can do now

- **Run the batch part of a pipeline in Krishiv.** If a job reads a stream only to land files, you can often re-express
  the transform as a batch query over the landed files: [Data sources](data-sources.md) covers Parquet and CSV on local
  disk, S3, Google Cloud Storage and Azure Blob Storage, and [Iceberg and Delta tables](lakehouse-tables.md) covers
  table reads and writes.
- **Keep the streaming job on Spark.** Move only the batch stages. [Migrate a PySpark job](pyspark-migration.md) lists
  the functions that differ from Spark, and [Check a project](check-compat.md) describes a scanner for them.
- **Check the release you run.** [Stability](../concepts/stability.md) says which surfaces are stable, and the
  [release notes](../releases/index.md) list what each release adds.

## Related pages

- [Migrate a PySpark job](pyspark-migration.md)
- [Compatibility](../concepts/compatibility.md)
