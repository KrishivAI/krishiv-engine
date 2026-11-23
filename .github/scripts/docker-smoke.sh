#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Release check: the built image serves Spark Connect to a PySpark 4.x client.
#
#   docker-smoke.sh IMAGE
#
# Generates a self-signed certificate and a token, starts IMAGE with both (the image binds
# 0.0.0.0, which the server refuses without authentication and TLS), waits for the container to
# report healthy, reads the console page, runs one query through PySpark over TLS with the token, writes and reads a Delta table, and checks that the
# image is set to run as a non-root user and that the admin port answers `/readyz`.
set -euo pipefail
image=$1
work=$(mktemp -d)
name=krishiv-smoke-$$
cleanup() {
  docker logs "$name" 2>&1 | tail -n 40 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT

openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=localhost \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
  -keyout "$work/key.pem" -out "$work/cert.pem" 2>/dev/null
printf '%s' "smoke-$(openssl rand -hex 16)" >"$work/token"
# The key and the token must not be readable by group or others; the container runs as the
# invoking user for this check (the image's own user, 10001, is checked below).
chmod 600 "$work/key.pem" "$work/token"
mkdir -m 777 "$work/data"

docker run -d --name "$name" --user "$(id -u):$(id -g)" -p 15002:15002 -p 8080:8080 -v "$work:/etc/krishiv:ro" -v "$work/data:/data" "$image" \
  server --host 0.0.0.0 \
  --conf krishiv.server.tls.certFile=/etc/krishiv/cert.pem \
  --conf krishiv.server.tls.keyFile=/etc/krishiv/key.pem \
  --conf krishiv.auth.tokens.smoke.secret=file:/etc/krishiv/token \
  --conf krishiv.auth.tokens.smoke.principal=smoke \
  --conf krishiv.auth.tokens.smoke.kind=service \
  --conf krishiv.auth.tokens.smoke.role=admin >/dev/null

status=
for _ in $(seq 1 90); do
  status=$(docker inspect -f '{{.State.Health.Status}}' "$name")
  [ "$status" = healthy ] && break
  [ "$(docker inspect -f '{{.State.Running}}' "$name")" = true ] || { echo "::error::container exited"; exit 1; }
  sleep 2
done
[ "$status" = healthy ] || { echo "::error::container never became healthy ($status)"; exit 1; }

user=$(docker image inspect -f '{{.Config.User}}' "$image")
case "$user" in ''|0|0:*|root|root:*) echo "::error::the image runs as root ($user)"; exit 1 ;; esac
curl -fsk https://127.0.0.1:8080/readyz >/dev/null
# The web console: the page at `/` of the UI port answers 200.
[ "$(curl -sk -o /dev/null -w '%{http_code}' https://127.0.0.1:8080/)" = 200 ] \
  || { echo "::error::the console page did not answer 200"; exit 1; }

venv=$work/venv
uv venv -q "$venv"
uv pip install -q --python "$venv/bin/python" "pyspark[connect]>=4.0,<5"
GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="$work/cert.pem" "$venv/bin/python" - "$(cat "$work/token")" <<'PY'
import sys
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(f"sc://localhost:15002/;use_ssl=true;token={sys.argv[1]}").getOrCreate()
rows = spark.sql("SELECT 1 + 1 AS two, upper('krishiv') AS name").collect()
assert rows[0]["two"] == 2 and rows[0]["name"] == "KRISHIV", rows
print("pyspark", spark.version, "->", rows[0])

# Delta: create, write and read a table in the mounted data directory.
path = "/data/delta-smoke"
spark.sql(f"CREATE TABLE smoke_delta (id BIGINT, region STRING) USING delta LOCATION '{path}'")
spark.sql("INSERT INTO smoke_delta VALUES (1, 'east'), (2, 'west')")
got = sorted(tuple(r) for r in spark.sql(f"SELECT id, region FROM delta.`{path}`").collect())
assert got == [(1, "east"), (2, "west")], got
print("delta ->", got)
PY
