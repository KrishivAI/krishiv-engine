---
description: "Run a local lakehouse with docker compose: the Krishiv server, MinIO object storage and a PySpark notebook that writes and reads a Delta table."
---

# Local lakehouse with docker compose

The `examples/docker-compose/` directory of the Krishiv repository starts three containers on your machine:

| Service | Role | On your machine |
|---|---|---|
| `krishiv` | the [official image](run-krishiv-in-docker.md): Spark Connect, Arrow Flight SQL and the web console | `sc://localhost:15002` (TLS), `https://localhost:8080/` |
| `minio` | S3-compatible object storage with one bucket, `lakehouse` | `http://127.0.0.1:9000`, console `http://127.0.0.1:9001` |
| `jupyter` | JupyterLab with `pyspark-client`, connected to `krishiv` | `http://127.0.0.1:8888/lab?token=krishiv` |

One notebook writes a Delta table to MinIO and reads it back through Krishiv. Nothing leaves your machine, and no JVM
runs anywhere.

## Start it

You need Docker with the Compose plugin and about 2 GB of free memory.

```bash
git clone https://github.com/KrishivAI/krishiv-engine.git
cd krishiv-engine/examples/docker-compose
docker compose up -d --wait        # returns when krishiv, minio and jupyter are healthy
```

On Linux, if `id -u` is not 1000, start it as yourself so that JupyterLab can save into `notebooks/`:
`JUPYTER_UID=$(id -u) JUPYTER_GID=$(id -g) docker compose up -d --wait`.

Open <http://127.0.0.1:8888/lab?token=krishiv> and run `lakehouse.ipynb` from top to bottom.

## What the notebook does

It connects with no arguments. The compose file sets `SPARK_REMOTE` in the `jupyter` container to the server's
address and token, and the container trusts the server's self-signed certificate:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

The notebook holds no storage keys either. The `krishiv` container reaches MinIO with its own credentials and
endpoint, the standard `AWS_*` variables in the compose file. So a table is just a path:

```python
path = "s3://lakehouse/trips"

trips = spark.sql("""
    SELECT * FROM VALUES
        (1, 'Pune', 4.2D, 96.0D),
        (2, 'Pune', 11.8D, 240.0D),
        (3, 'Mumbai', 7.5D, 210.0D)
    AS t(trip_id, city, distance_km, fare)
""")
trips.write.format("delta").mode("overwrite").save(path)

spark.read.format("delta").load(path).orderBy("trip_id").toPandas()
```

The write creates a Delta table: Parquet data files and a `_delta_log/` directory of commits under
`s3://lakehouse/trips`. The notebook then registers it by name and changes it with SQL, one commit per statement:

```python
spark.sql(f"CREATE TABLE IF NOT EXISTS trips USING delta LOCATION '{path}'")
spark.sql("INSERT INTO trips VALUES (6, 'Nashik', 8.9, 185.0)")
spark.sql("UPDATE trips SET fare = round(fare * 1.1, 2) WHERE city = 'Pune'")
spark.sql("DELETE FROM trips WHERE distance_km < 4")
```

See [Changing rows in a Delta table](delta-dml.md) and
[Iceberg and Delta tables, and catalogs](lakehouse-tables.md) for `MERGE` and the rest.

Two things differ from a Spark server in 0.3.0, and the notebook is written around them:

- `DataFrame.show()` is not served yet: use `toPandas()` (a table in the notebook) or `collect()`;
- `spark.createDataFrame(...)`, and `toPandas()` with the 4.2 and 4.3 clients, read settings the server does not
  answer yet. The example installs `pyspark-client` below 4.2 and builds its rows with SQL `VALUES`.

## Look around

- The Krishiv console, `https://localhost:8080/?token=krishiv-quickstart-token-change-me-please`: your browser warns
  about the self-signed certificate. The token goes in the address once; the page then keeps a session and removes
  it from the address. The notebook's queries are under Queries and History.
- The MinIO console, <http://127.0.0.1:9001> (user `krishiv`, password `krishiv-quickstart`): the table's Parquet
  files and its `_delta_log/` are in the `lakehouse` bucket.

## Connect from your own Python

Copy the server's certificate out and point gRPC at it:

```bash
docker compose cp krishiv:/var/lib/krishiv/tls/cert.pem .
pip install "pyspark-client>=4.0,<4.2"
GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=cert.pem python -c '
from pyspark.sql import SparkSession
spark = SparkSession.builder.remote("sc://localhost:15002/;use_ssl=true;token=krishiv-quickstart-token-change-me-please").getOrCreate()
print(spark.sql("SELECT count(*) AS trips FROM delta.`s3://lakehouse/trips`").collect())'
```

## What the compose file configures

- **Token and TLS.** A server on a non-loopback address needs authentication, and a bearer token needs TLS. The
  image's entry point supplies both: the token from `KRISHIV_AUTH_TOKEN` (at least 32 characters) and a self-signed
  certificate kept in a named volume. See [Run Krishiv in Docker](run-krishiv-in-docker.md#start-a-server).
- **The bucket.** A server with authentication reads and writes only the object-store prefixes it is told to:
  `krishiv.io.objectStore.allowedPrefixes=s3://lakehouse`.
- **A private address.** The server does not connect to private addresses unless told to, and MinIO on the compose
  network is one: `krishiv.io.egress.privateAddresses=allow`.

The token, the MinIO keys, the ports and the images are variables with defaults; set them in the shell or in a `.env`
file next to `compose.yaml`. The example's `README.md` lists them.

## Keep it local

The example is for one machine: default secrets, a self-signed certificate, every port published on `127.0.0.1` only.
Change the token and the MinIO keys before anything else can reach those ports; for a server other machines reach,
see [Deployment](deploy.md) and [Securing a single-node server](single-node-server-security.md). Give the `krishiv`
service a memory limit (`mem_limit`): its pool is 80% of it.

## Stop and clean up

```bash
docker compose down        # stops the containers, keeps the bucket and the certificate
docker compose down -v     # also deletes them
```

## Next steps

- Point Krishiv at your own bucket: change the `AWS_*` variables and the allowed prefix of the `krishiv` service, and
  use `s3://your-bucket/path` ([Data sources](data-sources.md)).
- JDBC and ADBC clients connect to port `15002` with the Flight SQL driver ([Arrow Flight SQL](flight-sql.md)).
