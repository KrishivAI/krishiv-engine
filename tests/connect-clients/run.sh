#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Smoke runs of the Apache Spark Connect Go and Scala clients against a Spark Connect server.
# Report-only: the exit status is 0 whenever the runs happened; failures are in the report.
#
#   tests/connect-clients/run.sh --server-bin target/debug/krishiv       # start `krishiv server`
#   tests/connect-clients/run.sh --remote sc://localhost:15002           # an already running server
#
# Options: --clients go,scala,delta,tpcds (default go,scala,delta)   --out DIR (default target/connect-clients)
#          --tpcds-data DIR   the TPC-DS Parquet files (<table>.parquet) for the `tpcds` run
# `delta` and `tpcds` drive the pinned pyspark client of bench/ (uv); `delta` reads the Spark-written tables of
# tests/fixtures/delta-4.4.1 (the server needs the `delta` cargo feature).
# `go` and `scala` run in containers (golang, virtuslab/scala-cli) on the host network and need Docker;
# `delta` and `tpcds` need uv; python3 is needed for every run. See README.md.
set -euo pipefail

# Pinned by tag and digest, so the report changes only with a commit here; Renovate bumps both (renovate.json).
go_image="golang:1.24.13@sha256:d2d2bc1c84f7e60d7d2438a3836ae7d0c847f4888464e7ec9ba3a1339a1ee804"
scala_image="virtuslab/scala-cli:1.19.0@sha256:d1715b5c5085db2d461a293757561f240ec6da955846a4f5d2c0f5cc30ae2f9d"

here="$(cd "$(dirname "$0")" && pwd)"
clients="go,scala,delta"
tpcds_data=""
out="target/connect-clients"
remote=""
server_bin=""
while [ $# -gt 0 ]; do
  case "$1" in
    --server-bin) server_bin="$2"; shift 2 ;;
    --remote) remote="$2"; shift 2 ;;
    --clients) clients="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --tpcds-data) tpcds_data="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$server_bin" ] && [ -z "$remote" ]; then
  echo "give --server-bin PATH or --remote sc://HOST:PORT" >&2
  exit 2
fi
mkdir -p "$out"
out="$(cd "$out" && pwd)"
if [ -n "$tpcds_data" ]; then tpcds_data="$(cd "$tpcds_data" && pwd)"; fi
# Reports of earlier runs must not pass for this run's. All are cleared, not only the selected clients':
# report.py reads whichever of go.json and scala.json exist.
rm -f "$out"/{go,scala,delta-read,tpcds}.json "$out"/summary.md "$out"/{delta-read,tpcds}-summary.md

server_pid=""
cleanup() {
  [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

if [ -n "$server_bin" ]; then
  # A free port for each of the two listeners; the admin port is not used by the clients.
  port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  admin="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  "$server_bin" server --connect-port "$port" --port "$admin" >"$out/server.log" 2>&1 &
  server_pid=$!
  for i in $(seq 1 60); do
    if python3 -c "import socket; socket.create_connection(('127.0.0.1', $port), 1)" 2>/dev/null; then break; fi
    kill -0 "$server_pid" 2>/dev/null || { echo "the server exited; see $out/server.log" >&2; exit 1; }
    sleep 1
    [ "$i" = 60 ] && { echo "the server did not open port $port in 60 s; see $out/server.log" >&2; exit 1; }
  done
  remote="sc://127.0.0.1:$port"
fi
echo "server: $remote"

for c in ${clients//,/ }; do
  case "$c" in
    go)
      docker run --rm --network host -v "$here:/src:ro" -v "$out:/out" -v krishiv-smoke-gomod:/go/pkg/mod \
        -e GOFLAGS=-buildvcs=false "$go_image" \
        sh -c "cp -r /src/go /tmp/go && cd /tmp/go && go mod tidy && go run . --remote '$remote' --queries /src/queries.json --out /out/go.json" \
        || echo "go client run failed" >&2
      ;;
    scala)
      docker run --rm --network host -v "$here:/src:ro" -v "$out:/out" -v krishiv-smoke-scala:/root/.cache \
        "$scala_image" run --workspace /tmp/scala-ws /src/scala/Smoke.scala -- "$remote" /src/queries.json /out/scala.json \
        || echo "scala client run failed" >&2
      ;;
    delta)
      (cd "$here/../../bench" && uv run --frozen python "$here/delta_read.py" --remote "$remote" --out "$out") \
        || echo "delta read run failed" >&2
      ;;
    tpcds)
      [ -n "$tpcds_data" ] || { echo "tpcds needs --tpcds-data DIR" >&2; continue; }
      (cd "$here/../../bench" && uv run --frozen python "$here/tpcds_connect.py" --remote "$remote" --data "$tpcds_data" --out "$out") \
        || echo "tpcds run failed" >&2
      ;;
    *) echo "unknown client: $c" >&2 ;;
  esac
done

python3 "$here/report.py" "$out" || true
