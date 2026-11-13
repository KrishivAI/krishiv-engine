#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Usage: run.sh <spark-version> (e.g. 4.2.0). Regenerates results/spark-<ver>/*.out in a throwaway container.
set -e
V=$1
ONLY=${2:-*}
HERE=$(cd "$(dirname "$0")" && pwd)
FN=$(dirname "$HERE")
N=krishiv-s7pr34-oracle-$V
mkdir -p "$HERE/results/spark-$V"
docker run --rm --name "$N" --user root -v "$FN":/g -w /g apache/spark:$V /bin/bash -c '
for f in /g/variant_misc/inputs/'"$ONLY"'.sql; do
  b=$(basename $f)
  /opt/spark/bin/spark-submit --conf spark.ui.enabled=false /g/run_oracle.py $f /g/variant_misc/results/spark-'"$V"'/$b.out >/dev/null 2>&1 || echo FAIL $b
done'
