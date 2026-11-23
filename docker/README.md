# The Krishiv image

`docker/Dockerfile` builds `krishiv` with the server, Spark Connect, Arrow Flight SQL, the web console, Delta
and Iceberg tables and the cloud object stores (`--features server,connect,flight-sql,cloud,ui,delta,iceberg`) on `debian:bookworm-slim`. It runs as
uid/gid 10001 (never root), exposes two ports and has a `HEALTHCHECK` on `/healthz`.

| Port | Serves |
|---|---|
| 15002 | Spark Connect and Arrow Flight SQL (one listener) |
| 8080 | admin: `/healthz`, `/readyz`, `/metrics`, the web console |

`ENTRYPOINT ["krishiv"]`, `CMD ["server", "--host", "0.0.0.0"]`: any other `krishiv` command works too
(`docker run --rm krishiv sql -e "SELECT 1"`).

Build (context: the repository root; amd64 and arm64 are built natively, one image each):

```text
docker buildx build -f docker/Dockerfile -t krishiv .
docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t krishiv .   # multi-arch manifest
```

## Running the server

The default command binds `0.0.0.0`, and a non-loopback server refuses to start without an authentication method
and, for tokens, TLS (ADR 0046, `docs/guides/deploy.md`). Mount a certificate and a token and name them:

```text
docker run -d --name krishiv -p 15002:15002 -p 8080:8080 -v ./secrets:/etc/krishiv:ro krishiv \
  server --host 0.0.0.0 \
  --conf krishiv.server.tls.certFile=/etc/krishiv/cert.pem \
  --conf krishiv.server.tls.keyFile=/etc/krishiv/key.pem \
  --conf krishiv.auth.tokens.me.secret=file:/etc/krishiv/token \
  --conf krishiv.auth.tokens.me.principal=me \
  --conf krishiv.auth.tokens.me.kind=user \
  --conf krishiv.auth.tokens.me.role=admin
```

The key and the token file must not be readable by group or others, and must be readable by uid 10001. Clients
connect with `sc://localhost:15002/;use_ssl=true;token=<token>`. `.github/scripts/docker-smoke.sh` is a working
example (self-signed certificate, PySpark 4.x, one query); CI runs it on both architectures.

For a throwaway loopback-only run, publish no port and use `docker exec`, or run `--network host` with
`server --host 127.0.0.1`.
