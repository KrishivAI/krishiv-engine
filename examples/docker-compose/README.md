# A local lakehouse with Docker Compose

Krishiv, an S3 bucket and a notebook on your machine, with one command:

| Service | What it is | On your machine |
|---|---|---|
| `krishiv` | the Krishiv server: Spark Connect, Arrow Flight SQL, the web console | `sc://localhost:15002` (TLS), `https://localhost:8080/` |
| `minio` | MinIO, an S3-compatible store, with one bucket, `lakehouse` | `http://127.0.0.1:9000`, console `http://127.0.0.1:9001` |
| `jupyter` | JupyterLab with the PySpark Connect client (`pyspark-client`, no JVM) | `http://127.0.0.1:8888/lab?token=krishiv` |

## Start

```bash
cd examples/docker-compose
docker compose up -d --wait
```

On Linux, if `id -u` is not 1000, start it as yourself so that JupyterLab can save into `notebooks/`:
`JUPYTER_UID=$(id -u) JUPYTER_GID=$(id -g) docker compose up -d --wait`.

Open <http://127.0.0.1:8888/lab?token=krishiv> and run `lakehouse.ipynb`. It writes a Delta table to
`s3://lakehouse/trips` on MinIO, reads it back by path and by name, and changes it with `INSERT`, `UPDATE` and
`DELETE`.

The notebook's first cell is all a notebook needs to connect:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

`compose.yaml` sets `SPARK_REMOTE` to `sc://krishiv:15002/;use_ssl=true;token=...` in the `jupyter` container, which trusts the
server's certificate (`jupyter/start.sh`).

Two things differ from a Spark server in 0.3, and the notebook is written around them:

- `DataFrame.show()` is not served yet: use `toPandas()` (a table in the notebook) or `collect()`;
- `toPandas()` of the 4.3 client reads a setting the server does not answer yet (`spark.sql.execution.arrow.pyspark.selfDestruct.enabled`),
  and fails with `SQL_CONF_NOT_FOUND`. `jupyter/requirements.txt` holds `pyspark-client` below 4.2, the range the notebook is
  tested with; `collect()` works with the 4.3 client too.

Look around while it runs:

- the Krishiv console, `https://localhost:8080/?token=krishiv-quickstart-token-change-me-please`: your browser
  warns about the self-signed certificate. The token goes in the address once (the page then keeps a session and
  removes it from the address). The notebook's queries are under Queries and History;
- the MinIO console, <http://127.0.0.1:9001> (`krishiv` / `krishiv-quickstart`): the table's Parquet files and
  its `_delta_log/` are in the `lakehouse` bucket.

## Stop

```bash
docker compose down        # keeps the bucket and the certificate
docker compose down -v     # deletes them
```

## Connect from your own Python

The server's certificate is self-signed, generated on first start. Copy it out and point gRPC at it:

```bash
docker compose cp krishiv:/var/lib/krishiv/tls/cert.pem .
pip install "pyspark-client>=4.0,<4.2"
GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=cert.pem python -c '
from pyspark.sql import SparkSession
spark = SparkSession.builder.remote("sc://localhost:15002/;use_ssl=true;token=krishiv-quickstart-token-change-me-please").getOrCreate()
print(spark.sql("SELECT count(*) AS trips FROM delta.`s3://lakehouse/trips`").collect())'
```

## Settings

Set these in the shell or in a `.env` file next to `compose.yaml`.

| Variable | Default | Meaning |
|---|---|---|
| `KRISHIV_IMAGE` | `ghcr.io/krishivai/krishiv:0.3.0` | the Krishiv image; a local build: `docker buildx build -f docker/Dockerfile -t krishiv .` at the repository root, then `KRISHIV_IMAGE=krishiv` |
| `KRISHIV_AUTH_TOKEN` | `krishiv-quickstart-token-change-me-please` | the server's bearer token (one admin principal); at least 32 characters |
| `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | `krishiv`, `krishiv-quickstart` | MinIO's root keys; the server reads the bucket with them |
| `JUPYTER_TOKEN` | `krishiv` | JupyterLab's token |
| `JUPYTER_UID`, `JUPYTER_GID` | `1000`, `1000` | the user JupyterLab runs as; it saves notebooks into `./notebooks`, so on Linux set them to your own when `id -u` is not 1000: `JUPYTER_UID=$(id -u) JUPYTER_GID=$(id -g) docker compose up -d --wait` |
| `MINIO_IMAGE` | `cgr.dev/chainguard/minio:latest` | the MinIO image. The quickstart is tested with `cgr.dev/chainguard/minio@sha256:304e0bbeb70a9aa0c40ff26d5a9c0a24296019c6d326ba8b75214fb73441d6e9` (the digest `smoke.sh` pins); set that if a newer `latest` does not start |
| `KRISHIV_CONNECT_PORT`, `KRISHIV_UI_PORT`, `MINIO_PORT`, `MINIO_CONSOLE_PORT`, `JUPYTER_PORT` | `15002`, `8080`, `9000`, `9001`, `8888` | the ports on your machine (all on `127.0.0.1`) |

## What is configured, and why

- **Token and TLS.** The server refuses to listen beyond loopback without authentication and TLS. The image's
  entrypoint supplies both: the token from `KRISHIV_AUTH_TOKEN` and a self-signed certificate kept in the
  `krishiv-state` volume (`docker/README.md`).
- **The bucket.** A server with authentication reads and writes only the object-store prefixes it is told to:
  `krishiv.io.objectStore.allowedPrefixes=s3://lakehouse`. It reaches MinIO with its own credentials and endpoint,
  the standard `AWS_*` variables, so the notebook holds no keys.
- **A private address.** The server does not connect to private addresses unless told to; MinIO on the compose
  network is one: `krishiv.io.egress.privateAddresses=allow`.

This is a quickstart for one machine: default secrets, a self-signed certificate, every port on `127.0.0.1`. For a
server other machines reach, see the [deployment guide](../../docs/guides/deploy.md) and
[single-node server security](../../docs/guides/single-node-server-security.md).

`smoke.sh` is the CI check: it brings the stack up, runs the notebook headless and checks the table in MinIO.
