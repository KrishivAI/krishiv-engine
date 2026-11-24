#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Release check: the built image serves Spark Connect to a PySpark 4.x client.
#
#   docker-smoke.sh IMAGE
#
# Starts IMAGE with only KRISHIV_AUTH_TOKEN set (the entrypoint binds 0.0.0.0 with a generated
# self-signed certificate), waits for the container to
# report healthy, reads the console page, runs one query through PySpark over TLS with the token, writes and reads a Delta table, and checks that the
# image is set to run as a non-root user and that the admin port answers `/readyz`. Then it exercises
# the entrypoint's other branches (a token from a mounted file, a generated token, the refusals).
set -euo pipefail
image=$1
work=$(mktemp -d)
name=krishiv-smoke-$$
cleanup() {
  docker logs "$name" 2>&1 | tail -n 40 || true
  docker rm -f "$name" "$name-file" "$name-gen" >/dev/null 2>&1 || true
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
# `timeout`: a client that finished its checks must not hang the job on a gRPC thread at exit.
GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="$work/cert.pem" timeout 300 "$venv/bin/python" - "$token" <<'PY'
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
spark.stop()
PY

# Entrypoint branches. Each starts a second container on other host ports (the first still runs).
echo "== entrypoint: token from a mounted secret file (root-owned, mode 0444) =="
printf '%s' "file-$(openssl rand -hex 16)" >"$work/secret"
chmod 444 "$work/secret"
docker run -d --name "$name-file" -p 15012:15002 -p 8081:8080 \
  -v "$work/secret:/run/secrets/krishiv_token:ro" "$image" >/dev/null
for _ in $(seq 1 60); do
  curl -fsk https://127.0.0.1:8081/readyz >/dev/null 2>&1 && break
  [ "$(docker inspect -f '{{.State.Running}}' "$name-file")" = true ] || { docker logs "$name-file"; echo "::error::file-token container exited"; exit 1; }
  sleep 2
done
curl -fsk https://127.0.0.1:8081/readyz >/dev/null || { echo "::error::file-token container never became ready"; exit 1; }
docker rm -f "$name-file" >/dev/null

echo "== entrypoint: no token given generates one and prints it once =="
docker run -d --name "$name-gen" "$image" >/dev/null
for _ in $(seq 1 30); do
  docker logs "$name-gen" 2>&1 | grep -q 'generated a token in' && break
  sleep 1
done
docker logs "$name-gen" 2>&1 | grep -q 'generated a token in' || { echo "::error::no generated-token notice"; exit 1; }
docker rm -f "$name-gen" >/dev/null

refuses() { # refuses MESSAGE docker-run-args...: the container must exit non-zero with MESSAGE
  local msg=$1; shift
  local out
  if out=$(docker run --rm "$@" "$image" 2>&1); then echo "::error::expected a refusal: $msg"; exit 1; fi
  grep -q "$msg" <<<"$out" || { echo "$out"; echo "::error::refusal lacks '$msg'"; exit 1; }
}
echo "== entrypoint: refusals =="
: >"$work/empty"
refuses "is not readable by uid" -v "$work/empty:/run/secrets/krishiv_token:ro"
refuses "are set but" -e KRISHIV_AUTH_TOKEN=x -e KRISHIV_TLS_CERT=/nonexistent/cert.pem -e KRISHIV_TLS_KEY=/nonexistent/key.pem
