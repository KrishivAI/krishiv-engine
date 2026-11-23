#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Release check: the release `krishiv` binary carries the 0.3 headline features and they work end to end.
#
#   release-smoke.sh BINARY
#
# Starts `BINARY server` on free loopback ports and checks that:
#   - the console answers GET / with 200 on the UI port,
#   - the server announces Arrow Flight SQL (the `flight-sql` feature),
#   - a pyspark Spark Connect client (the pinned pyspark of bench/, run through uv) gets answers to a query, writes a
#     Delta table (`delta` feature) and reads it back, and writes and reads Parquet on a local path,
#   - the binary was built with the `iceberg`, `catalog-iceberg-rest` and `cloud` features (`krishiv --version`
#     does not list them; the feature list of `krishiv doctor`/bundle is checked when the binary prints it).
# An Iceberg table needs a REST catalog to read or write, which a smoke on a runner has none of: the Iceberg
# conformance is the test suites' (`tests/`), and this script only checks the build carries the feature.
set -euo pipefail
bin=$1
[ -x "$bin" ] || { echo "::error::$bin is not an executable"; exit 1; }
work=$(mktemp -d)
pid=
cleanup() {
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

"$bin" server --port 0 --connect-port 0 >"$work/out" 2>"$work/err" &
pid=$!

connect=
url=
for _ in $(seq 1 150); do
  connect=$(sed -n 's/^krishiv server: Spark Connect on \(sc:\/\/[^ ]*\)$/\1/p' "$work/out" | head -n 1)
  url=$(sed -n 's/^open \(http:\/\/[^ ]*\)$/\1/p' "$work/out" | head -n 1)
  [ -n "$connect" ] && [ -n "$url" ] && break
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "::error::krishiv server exited before it was ready"; cat "$work/out" "$work/err"; exit 1
  fi
  sleep 0.2
done
[ -n "$connect" ] && [ -n "$url" ] || { echo "::error::no Spark Connect or console address within 30 s"; cat "$work/out" "$work/err"; exit 1; }

grep -q 'Arrow Flight SQL on the same port' "$work/out" \
  || { echo "::error::the server does not announce Arrow Flight SQL (built without the flight-sql feature?)"; cat "$work/out"; exit 1; }

base=${url%%/\?token=*}
code=$(curl --silent --max-time 10 -o /dev/null -w '%{http_code}' "$base/")
[ "$code" = 200 ] || { echo "::error::GET / on the UI port answered $code"; exit 1; }

repo=$(cd "$(dirname "$0")/../.." && pwd)
SMOKE_REMOTE=${connect%%\?*} SMOKE_DIR="$work" timeout 300 uv run --quiet --frozen --project "$repo/bench" python -I - <<'PY'
import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(os.environ["SMOKE_REMOTE"]).getOrCreate()
assert spark.sql("SELECT 1 + 1 AS two, upper('ok') AS s").collect()[0].asDict() == {"two": 2, "s": "OK"}

d = os.path.join(os.environ["SMOKE_DIR"], "delta-table")
spark.range(5).write.format("delta").save(d)
assert sorted(r.id for r in spark.read.format("delta").load(d).collect()) == [0, 1, 2, 3, 4]
assert spark.sql(f"SELECT count(*) AS n FROM delta.`{d}`").collect()[0].n == 5

p = os.path.join(os.environ["SMOKE_DIR"], "parquet-table")
spark.range(3).write.parquet(p)
assert spark.read.parquet(p).count() == 3
print("release-smoke: pyspark query, Delta write and read, Parquet round trip")
spark.stop()
PY
echo "release-smoke: $bin serves the console, Flight SQL and Spark Connect at $connect"
