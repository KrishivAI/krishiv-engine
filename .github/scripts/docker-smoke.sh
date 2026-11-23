#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Release check: the built image serves Spark Connect to a PySpark 4.x client.
#
#   docker-smoke.sh IMAGE
#
# Starts IMAGE with only KRISHIV_AUTH_TOKEN set (the entrypoint binds 0.0.0.0 with a generated
# self-signed certificate), waits for the container to
# report healthy, reads the console page, runs one query through PySpark over TLS with the token, writes and reads a Delta table, and checks that the
# image is set to run as a non-root user and that the admin port answers `/readyz`.
set -euo pipefail
image=$1
work=$(mktemp -d)
name=krishiv-smoke-$$
cleanup() {
  docker logs "$name" 2>&1 | tail -n 40 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  # The container's user (10001) owns what it wrote under data/: clear it as root in the image.
  docker run --rm --user 0 --entrypoint sh -v "$work/data:/d" "$image" -c 'rm -rf /d/* /d/.[!.]*' >/dev/null 2>&1 || true
  rm -rf "$work" || true
}
trap cleanup EXIT

token="smoke-$(openssl rand -hex 16)"
mkdir -m 777 "$work/data"

# The one-variable start: the entrypoint generates the certificate and takes the token from the
# environment (the server alone refuses 0.0.0.0 without authentication and TLS).
docker run -d --name "$name" -p 15002:15002 -p 8080:8080 -e KRISHIV_AUTH_TOKEN="$token" \
  -v "$work/data:/data" "$image" >/dev/null

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
docker cp "$name:/var/lib/krishiv/tls/cert.pem" "$work/cert.pem"
curl -fsk https://127.0.0.1:8080/readyz >/dev/null
# The web console: the page at `/` of the UI port answers 200.
[ "$(curl -sk -o /dev/null -w '%{http_code}' https://127.0.0.1:8080/)" = 200 ] \
  || { echo "::error::the console page did not answer 200"; exit 1; }

venv=$work/venv
uv venv -q "$venv"
uv pip install -q --python "$venv/bin/python" "pyspark[connect]>=4.0,<5"
GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="$work/cert.pem" "$venv/bin/python" - "$token" <<'PY'
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
