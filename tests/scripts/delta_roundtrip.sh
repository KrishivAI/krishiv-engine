#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Round trip of the Delta connector (P3-T028b, PR-53b): Spark 4.2.0 + delta-spark 4.4.1 makes tables (one
# partitioned, deletion vectors, change data feed), Krishiv reads them and commits an append, a whole-file DELETE
# and an append, Spark reads that (history, time travel, change feed) and runs DML on top, Krishiv reads Spark's
# result. Needs Docker and network (delta-spark is fetched by Spark's --packages). Run from the repository root; OUT
# defaults to a fresh temporary directory.
set -euo pipefail
OUT="${OUT:-$(mktemp -d)}"
OUT="$(cd "$OUT" && pwd -P)"
IMAGE="${SPARK_IMAGE:-apache/spark:4.2.0}"
export KRISHIV_DELTA_RT_OUT="$OUT"
rm -rf "$OUT/delta" "$OUT"/SPARK-*.json "$OUT/KRISHIV-RT.json"
spark() {
    # <out> is mounted at the same path, so both engines name the tables alike.
    docker run --rm --user "$(id -u)" -e HOME=/tmp -e TZ=UTC -v "$OUT:$OUT" -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro -v "$PWD/tests/scripts:/scripts:ro" --entrypoint /opt/spark/bin/spark-submit \
        "$IMAGE" --packages io.delta:delta-spark_2.13:4.4.1 --conf spark.jars.ivy=/tmp/ivy \
        /scripts/gen_delta_roundtrip.py "$1" "$OUT"
}
spark create
cargo nextest run -p krishiv-connectors-delta --run-ignored only -E 'test(krishiv_commits_on_spark_tables)'
spark readback
cargo nextest run -p krishiv-connectors-delta --run-ignored only -E 'test(krishiv_reads_spark_on_top)'
echo "delta round trip passed; artifacts in $OUT"
