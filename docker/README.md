# The Krishiv image

`docker/Dockerfile` builds `krishiv` with the server, Spark Connect, Arrow Flight SQL, the web console, Delta
and Iceberg tables and the cloud object stores (`--features server,connect,flight-sql,cloud,ui,delta,iceberg`) on `debian:bookworm-slim`. It runs as
uid/gid 10001 (never root), exposes two ports and has a `HEALTHCHECK` on `/healthz`.

| Port | Serves |
|---|---|
| 15002 | Spark Connect and Arrow Flight SQL (one listener) |
| 8080 | admin: `/healthz`, `/readyz`, `/metrics`, the web console |

`ENTRYPOINT ["docker-entrypoint.sh"]`, `CMD ["server"]`: `docker-entrypoint.sh` runs `krishiv server --host 0.0.0.0`
with authentication and TLS supplied (the server refuses a non-loopback bind without both, ADR 0046). Any other
`krishiv` command passes through (`docker run --rm krishiv sql -e "SELECT 1"`).

Build (context: the repository root; amd64 and arm64 are built natively, one image each):

```text
docker buildx build -f docker/Dockerfile -t krishiv .
docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t krishiv .   # multi-arch manifest
```

## Running the server

One command, one variable:

```text
docker run -d --name krishiv -p 15002:15002 -p 8080:8080 -e KRISHIV_AUTH_TOKEN=change-me-to-32-or-more-characters-of-your-own krishiv
docker cp krishiv:/var/lib/krishiv/tls/cert.pem .      # the self-signed certificate it generated
```

Clients connect with `sc://localhost:15002/;use_ssl=true;token=change-me-to-32-or-more-characters-of-your-own` (PySpark: set
`GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=cert.pem` to trust the certificate). The console is `https://localhost:8080/`.

| Setting | Default |
|---|---|
| token | `KRISHIV_AUTH_TOKEN` (at least 32 characters: the server refuses a shorter token at start), else the file `KRISHIV_AUTH_TOKEN_FILE` (default `/run/secrets/krishiv_token`, a Docker or compose secret), else a random token generated on first start and printed in the container log. It is one admin principal, `docker`, whose token expires on `KRISHIV_AUTH_TOKEN_EXPIRES` (YYYY-MM-DD, default one year from start). Local files may be read and written only under `/data` (a volume; `KRISHIV_ALLOWED_ROOTS` sets a comma-separated list of other existing directories), because a server with authentication allows no local path otherwise. |
| certificate | `KRISHIV_TLS_CERT` and `KRISHIV_TLS_KEY` (default `/etc/krishiv/tls/cert.pem` and `key.pem`), else a self-signed certificate (`localhost`, `127.0.0.1`, the container's host name) generated on first start in `KRISHIV_TLS_DIR` (default `/var/lib/krishiv/tls`; its path is printed). Mount that directory to keep the certificate. |

The key and a token file must not be readable by group or others. To configure authentication yourself (several
tokens, OIDC, client certificates) pass your own `--conf krishiv.auth.*` or `krishiv.server.tls.*` settings after
`server`: the entrypoint then adds nothing. `.github/scripts/docker-smoke.sh` runs the one-variable start with PySpark 4.x
on both architectures in CI.

For a throwaway loopback-only run, publish no port and use `docker exec`, or run `--network host` with
`server --host 127.0.0.1`.
