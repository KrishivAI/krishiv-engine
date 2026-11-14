#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Usage: run.sh <spark-version> [name]. Regenerates results/spark-<ver>/*.out in a throwaway container.
set -e
V=$1
ONLY=${2:-*}
HERE=$(cd "$(dirname "$0")" && pwd)
FN=$(dirname "$HERE")
N=krishiv-s7sql456-oracle-$V
mkdir -p "$HERE/results/spark-$V"
docker run --rm --name "$N" --user root -v "$FN":/g -w /g apache/spark:$V /bin/bash -c '
for f in /g/sequence_xml/inputs/'"$ONLY"'.sql; do
  b=$(basename $f)
  /opt/spark/bin/spark-submit --conf spark.ui.enabled=false /g/run_oracle.py $f /g/sequence_xml/results/spark-'"$V"'/$b.out >/dev/null 2>&1 || echo FAIL $b
done'
