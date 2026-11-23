---
description: "Run a local lakehouse with docker compose: the Krishiv server, MinIO object storage and a PySpark notebook that writes and reads a Delta table."
---

# Local lakehouse with docker compose

The `examples/docker-compose/` directory of the Krishiv repository starts three containers on your machine:

| Service | Role |
|---|---|
| `krishiv` | the [official image](run-krishiv-in-docker.md), serving Spark Connect on port `15002` |
| `minio` | S3-compatible object storage; the lakehouse bucket lives here |
| `jupyter` | a notebook with `pyspark-client`, connected to `krishiv` |

One notebook writes a Delta table to MinIO and reads it back through Krishiv. Nothing leaves your machine, and no JVM
runs anywhere.

<!-- FINALISE (front-1): service names, ports, bucket and notebook file name, how credentials are set in the compose file, and what the compose file publishes. -->

## Start it

You need Docker with the Compose plugin and about 2 GB of free memory.

```bash
git clone https://github.com/KrishivAI/krishiv-engine.git
cd krishiv-engine/examples/docker-compose
docker compose up -d
docker compose ps        # krishiv, minio and jupyter are running
```

Open the notebook at the address `docker compose logs jupyter` prints (it includes a one-time token). The MinIO
console is at `http://localhost:9001`.

## What the notebook does

The client connects to the `krishiv` service by name and points Krishiv at MinIO. Krishiv reads object-store settings
from the session (or from the standard `AWS_*` variables of the `krishiv` container), and `fs.s3a.endpoint` accepts an
`http://` URL for MinIO:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://krishiv:15002").create()

spark.conf.set("fs.s3a.endpoint", "http://minio:9000")
spark.conf.set("fs.s3a.access.key", "<MinIO user>")
spark.conf.set("fs.s3a.secret.key", "<MinIO password>")
spark.conf.set("fs.s3a.path.style.access", "true")

trips = spark.createDataFrame(
    [("east", 10.0), ("west", 20.0), ("east", 5.5)], "region string, amount double"
)
trips.write.format("delta").mode("overwrite").save("s3://lakehouse/trips")

spark.read.format("delta").load("s3://lakehouse/trips") \
    .groupBy("region").sum("amount").orderBy("region").show()
# east 15.5, west 20.0
```

The write creates a Delta table: Parquet data files and a `_delta_log/` directory of commits under
`s3://lakehouse/trips`. Browse it in the MinIO console. Append more rows with `mode("append")`, or change rows with
`UPDATE`, `DELETE` and `MERGE`; see [Changing rows in a Delta table](delta-dml.md) and
[Iceberg and Delta tables, and catalogs](lakehouse-tables.md).

## Keep it local

The example is for one machine. A server on a non-loopback address needs authentication, and a bearer token needs TLS
(see [Run Krishiv in Docker](run-krishiv-in-docker.md#start-a-server)); do not publish its ports beyond your own
machine without them. Give the `krishiv` service a memory limit (`mem_limit`): its pool is 80% of it. MinIO keeps its
data in a named volume, so the table survives `docker compose down`.

## Stop and clean up

```bash
docker compose down        # stops the containers, keeps the MinIO volume
docker compose down -v     # also deletes the volume and the table
```

## Next steps

- Point the notebook at your own bucket: change the endpoint and credentials, and use `s3://your-bucket/path`
  ([Data sources](data-sources.md)).
- Connect other tools to port `15002` with the Flight SQL driver ([Arrow Flight SQL](flight-sql.md)).
