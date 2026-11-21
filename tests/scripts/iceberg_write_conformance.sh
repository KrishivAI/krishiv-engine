#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Write conformance of the Iceberg connector (P3-T028b, PR-53b): Krishiv writes tables, Spark 4.1.0 + Iceberg 1.12.0
# reads them and appends on top, Krishiv reads Spark's result. Needs Docker and network (the Iceberg runtime jar is
# fetched by Spark's --packages). Run from the repository root; OUT defaults to a fresh temporary directory.
set -euo pipefail
OUT="${OUT:-$(mktemp -d)}"
OUT="$(cd "$OUT" && pwd -P)"
IMAGE="${SPARK_IMAGE:-apache/spark:4.1.0}"
export KRISHIV_ICEBERG_WRITE_OUT="$OUT"
rm -rf "$OUT/wh" "$OUT/KRISHIV.json" "$OUT/READBACK.json"
cargo nextest run -p krishiv-connectors-iceberg --run-ignored only -E 'test(krishiv_writes_the_warehouse)'
# <out> is mounted at the same path: the metadata Krishiv wrote holds absolute paths.
docker run --rm --user "$(id -u)" -e HOME=/tmp -e TZ=UTC -v "$OUT:$OUT" -v /etc/passwd:/etc/passwd:ro \
    -v "$PWD/tests/scripts:/scripts:ro" --entrypoint /opt/spark/bin/spark-submit "$IMAGE" \
    --packages org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.12.0 --conf spark.jars.ivy=/tmp/ivy \
    /scripts/gen_iceberg_write_readback.py "$OUT"
cargo nextest run -p krishiv-connectors-iceberg --run-ignored only -E 'test(krishiv_agrees_with_spark_readback)'
echo "write conformance passed; artifacts in $OUT"
