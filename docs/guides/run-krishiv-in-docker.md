---
description: "Run the Krishiv server from the official Docker image: image tags, ports, settings, volumes, health checks and connecting PySpark."
---

# Run Krishiv in Docker

The official image runs `krishiv server`: one process that serves Spark Connect, Arrow Flight SQL and the web console.
It needs no JVM. This page covers the image tags, the ports, how to give it settings, what to mount, and how to connect
stock PySpark to it.

<!-- FINALISE (p3-2, docker/Dockerfile): image name and tags, non-root uid, HEALTHCHECK command, default flags. -->

## Image and tags

```bash
docker pull ghcr.io/krishivai/krishiv:0.3.0
```

| Tag | Moves | Use it for |
|---|---|---|
| `0.3.0` | never | a pinned deployment |
| `0.3` | to each 0.3.x release | patch fixes without a version edit |
| `latest` | to each release | trying Krishiv |

Images are built for `linux/amd64` and `linux/arm64`; Docker pulls the one that matches your machine. They are pushed
only when a release is tagged.

The entry point is `krishiv` and the default command is `server`, so `docker run IMAGE` starts the server and
`docker run IMAGE sql ...` runs the SQL command line instead (see
[Run SQL from the command line](run-sql-from-the-command-line.md)). The process runs as a non-root user.

## Ports

| Port | What | Setting |
|---|---|---|
| `15002` | Spark Connect and Arrow Flight SQL (one listener, two protocols) | `--connect-port`, `krishiv.connect.port` |
| `8080` | admin API (`/healthz`, `/readyz`, `/metrics`) and the web console | `--port` |

Publish the ports you use with `-p`. A server reached through a published port must listen on all addresses
(`--host 0.0.0.0`), and the server refuses to start on a non-loopback address unless authentication is configured; a
bearer token or OIDC also needs TLS. The next section shows a setup that starts.

## Start a server

This example uses one admin token and a certificate you already have. The token is read from an environment variable,
so the setting that names it never holds the secret.

```bash
docker run --rm --name krishiv \
  -p 15002:15002 -p 8080:8080 \
  -e KRISHIV_TOKEN="$(openssl rand -hex 24)" \
  -v "$PWD/certs:/certs:ro" \
  ghcr.io/krishivai/krishiv:0.3.0 server \
    --host 0.0.0.0 \
    --conf krishiv.server.tls.certFile=/certs/server.crt \
    --conf krishiv.server.tls.keyFile=/certs/server.key \
    --conf krishiv.auth.tokens.admin.secret=env:KRISHIV_TOKEN \
    --conf krishiv.auth.tokens.admin.principal=admin \
    --conf krishiv.auth.tokens.admin.role=admin
```

The key file must not be readable by group or others, or the server refuses it. Token roles, OIDC, client certificates
and path confinement are in [Securing a single-node server](single-node-server-security.md) and
[Deployment](deploy.md#authentication-roles-and-tls).

Every `--conf KEY=VALUE` is an engine setting; the keys are in the [configuration reference](../reference/config/index.md).
`--conf` is repeatable.

## Settings and environment variables

- **Engine settings** are `--conf` arguments after `server`. Put them in a Compose file's `command:` or a Kubernetes
  `args:` list.
- **Secrets** go in environment variables and are named with an `env:` reference (`secret=env:KRISHIV_TOKEN`) or a
  `file:` reference to a mounted file.
- **Cloud credentials.** When a read or write sets no credentials, Krishiv uses the standard variables of each cloud:
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `GOOGLE_APPLICATION_CREDENTIALS` and
  `AZURE_STORAGE_ACCOUNT_NAME`. See [Data sources](data-sources.md#credentials-and-options).
- **Logs.** `KRISHIV_LOG_RATE` limits how fast repeated log records are written.

## Memory

With no `krishiv.memory.limit`, the server's pool is 80% of the container's memory limit (`docker run --memory 4g` gives
about 3.2 GiB), and operators spill to disk past it. Set a memory limit on the container; without one the pool is sized
from the host's memory. See [the memory limit in a container](deploy.md#the-memory-limit-in-a-container).

## Volumes

Mount a volume only for data you want to keep or read:

| Mount | For |
|---|---|
| a data directory, for example `-v "$PWD/data:/data"` | files that queries read or write (`/data/...` paths) |
| `/certs`, read-only | the TLS certificate and key |

<!-- FINALISE (p3-2): the image's data and spill directories, if it declares any, and whether the non-root user can write to a bind mount. -->

To limit what clients can reach, set `krishiv.io.local.allowedRoots=/data`. See
[Securing a single-node server](single-node-server-security.md#path-confinement).

## Check that it is up

```bash
curl -s -o /dev/null -w '%{http_code}\n' --cacert certs/server.crt https://localhost:8080/healthz   # 200
curl -s -o /dev/null -w '%{http_code}\n' --cacert certs/server.crt https://localhost:8080/readyz     # 200
```

`/healthz` and `/readyz` need no token. `/metrics` does: send the admin token as `Authorization: Bearer <token>`.
`docker ps` shows the container's health state once the image's health check has run.

## Connect PySpark

Install the thin client (no JVM) and connect with the token and TLS:

```python
# pip install pyspark-client
import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(
    f"sc://localhost:15002/;use_ssl=true;token={os.environ['KRISHIV_TOKEN']}"
).create()
spark.sql("SELECT 1 AS one").show()
```

With a certificate that your system does not trust (a self-signed one), point gRPC at it before you connect:
`export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=$PWD/certs/server.crt`.

JDBC and ADBC clients connect to the same port with the Flight SQL driver; see [Arrow Flight SQL](flight-sql.md).

## Related

- [Local lakehouse with docker compose](local-lakehouse-docker-compose.md): the image, MinIO and a notebook together.
- [Deployment](deploy.md) and [Securing a single-node server](single-node-server-security.md).
