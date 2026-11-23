---
description: "Run the Krishiv server from the official Docker image: image tags, ports, settings, volumes, health checks and connecting PySpark."
---

# Run Krishiv in Docker

The official image runs `krishiv server`: one process that serves Spark Connect, Arrow Flight SQL and the web console.
It needs no JVM. This page covers the image tags, the ports, how to give it settings, what to mount, and how to connect
stock PySpark to it.

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

The entry point is `docker-entrypoint.sh` and the default command is `server`, so `docker run IMAGE` starts the server.
Any other command passes through to `krishiv`: `docker run --rm IMAGE sql -e "SELECT 1"` runs the SQL command line (see
[Run SQL from the command line](run-sql-from-the-command-line.md)). The process runs as a non-root user (uid and gid
`10001`) with `/var/lib/krishiv` as its working directory.

## Ports

| Port | What | Setting |
|---|---|---|
| `15002` | Spark Connect and Arrow Flight SQL (one listener, two protocols) | `--connect-port`, `krishiv.connect.port` |
| `8080` | admin API (`/healthz`, `/readyz`, `/metrics`) and the web console | `--port` |

Publish the ports you use with `-p`. The server inside the image listens on all addresses, and the server refuses a
non-loopback address without authentication and TLS (ADR 0046), so the image's entry point supplies both.

## Start a server

One command and one variable:

```bash
docker run -d --name krishiv -p 15002:15002 -p 8080:8080 -e KRISHIV_AUTH_TOKEN=change-me \
  ghcr.io/krishivai/krishiv:0.3.0
docker cp krishiv:/var/lib/krishiv/tls/cert.pem .   # the self-signed certificate it generated
```

| What | How the image chooses it |
|---|---|
| Token | `KRISHIV_AUTH_TOKEN`; else the file `KRISHIV_AUTH_TOKEN_FILE` (default `/run/secrets/krishiv_token`, a Docker or Compose secret); else a random token generated on first start and printed in the container log. It is one admin principal, `docker`, and expires on `KRISHIV_AUTH_TOKEN_EXPIRES` (`YYYY-MM-DD`, default one year from start). |
| Certificate | `KRISHIV_TLS_CERT` and `KRISHIV_TLS_KEY` (default `/etc/krishiv/tls/cert.pem` and `key.pem`); else a self-signed certificate for `localhost`, `127.0.0.1` and the container's host name, generated on first start in `KRISHIV_TLS_DIR` (default `/var/lib/krishiv/tls`). Mount that directory to keep the certificate across containers. |

The key and a token file must not be readable by group or others, or the server refuses them.

To configure authentication yourself, pass your own `--host`, `krishiv.auth.*` or `krishiv.server.tls.*` settings after
`server`; the entry point then adds nothing. This example uses a certificate you already have:

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
    --conf krishiv.auth.tokens.admin.kind=service \
    --conf krishiv.auth.tokens.admin.role=admin \
    --conf krishiv.auth.principals.admin.expires=2027-12-31
```

A static service token must carry an expiry date. Token roles, OIDC, client certificates and path confinement are in
[Securing a single-node server](single-node-server-security.md) and [Deployment](deploy.md#authentication-roles-and-tls).

Every `--conf KEY=VALUE` is an engine setting; the keys are in the [configuration reference](../reference/config/index.md).
`--conf` is repeatable.

## Settings and environment variables

- **Image settings** are the environment variables in the table above.
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
| `/certs`, read-only | your own TLS certificate and key |
| `/var/lib/krishiv/tls` | the generated certificate and token, to keep them across containers |

The image declares no volumes. The server runs as uid `10001`, so a bind-mounted directory it writes to must be writable
by that uid (`chown 10001:10001 data`), and the key and token files must be readable by it. Spill files go to `~/.cache/krishiv/spill`, under the home directory `/var/lib/krishiv`, inside the container; set
`krishiv.spill.dir` to a mounted local volume if the container's own disk is small.

To limit what clients can reach, set `krishiv.io.local.allowedRoots=/data`. See
[Securing a single-node server](single-node-server-security.md#path-confinement).

## Check that it is up

```bash
curl -s -o /dev/null -w '%{http_code}\n' --cacert cert.pem https://localhost:8080/healthz   # 200
curl -s -o /dev/null -w '%{http_code}\n' --cacert cert.pem https://localhost:8080/readyz     # 200
```

`/healthz` and `/readyz` need no token. `/metrics` does: send the admin token as `Authorization: Bearer <token>`.
The image's `HEALTHCHECK` calls `/healthz` every 15 seconds (over HTTPS when TLS is set), so `docker ps` shows `healthy` after the start period of 20 seconds.

## Connect PySpark

Install the thin client (no JVM) and connect with the token and TLS. With the one-variable start, the token is the value of `KRISHIV_AUTH_TOKEN` and `cert.pem` is the certificate you copied out:

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
`export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=$PWD/cert.pem`.

JDBC and ADBC clients connect to the same port with the Flight SQL driver; see [Arrow Flight SQL](flight-sql.md).

## Related

- [Local lakehouse with docker compose](local-lakehouse-docker-compose.md): the image, MinIO and a notebook together.
- [Deployment](deploy.md) and [Securing a single-node server](single-node-server-security.md).
