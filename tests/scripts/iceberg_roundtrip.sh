#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Round trip of the Iceberg connector (P3-T028b, PR-53b): Spark 4.1.0 + Iceberg 1.12.0 makes tables (position
# deletes, a deletion vector, an evolved schema and spec), Krishiv appends and runs copy-on-write DELETE and UPDATE,
# Spark reads that and runs merge-on-read DML on top, Krishiv reads Spark's result. Needs Docker and network (the
# Iceberg runtime jar is fetched by Spark's --packages). Run from the repository root; OUT defaults to a fresh
# temporary directory.
set -euo pipefail
OUT="${OUT:-$(mktemp -d)}"
OUT="$(cd "$OUT" && pwd -P)"
IMAGE="${SPARK_IMAGE:-apache/spark:4.1.0}"
export KRISHIV_ICEBERG_RT_OUT="$OUT"
rm -rf "$OUT/wh" "$OUT"/SPARK-*.json "$OUT/KRISHIV-RT.json"
spark() {
    # <out> is mounted at the same path: Iceberg metadata holds absolute paths.
    docker run --rm --user "$(id -u)" -e HOME=/tmp -e TZ=UTC -v "$OUT:$OUT" -v /etc/passwd:/etc/passwd:ro \
        -v "$PWD/tests/scripts:/scripts:ro" --entrypoint /opt/spark/bin/spark-submit "$IMAGE" \
        --packages org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.12.0 --conf spark.jars.ivy=/tmp/ivy \
        /scripts/gen_iceberg_roundtrip.py "$1" "$OUT"
}
spark create
cargo nextest run -p krishiv-connectors-iceberg --run-ignored only -E 'test(krishiv_meets_spark_tables)'
spark readback
cargo nextest run -p krishiv-connectors-iceberg --run-ignored only -E 'test(krishiv_reads_spark_on_top)'
echo "iceberg round trip passed; artifacts in $OUT"
