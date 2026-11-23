#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CI check of the compose quickstart: bring the stack up, run the notebook headless, check that the
# Delta table is in MinIO, take the stack down.
#
#   smoke.sh [KRISHIV_IMAGE]
#
# KRISHIV_IMAGE defaults to the compose file's (the released image). The stack is its own compose
# project (`krishiv-quickstart-smoke`), removed with its volumes on exit; host ports come from the
# compose file's variables (KRISHIV_CONNECT_PORT, KRISHIV_UI_PORT, MINIO_PORT, MINIO_CONSOLE_PORT,
# JUPYTER_PORT). SMOKE_KEEP=1 leaves the stack up afterwards, to look at a failure.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] && export KRISHIV_IMAGE=$1
compose=(docker compose -p krishiv-quickstart-smoke)

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    "${compose[@]}" ps -a || true
    "${compose[@]}" logs --tail 60 || true
  fi
  [ -n "${SMOKE_KEEP:-}" ] || "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

"${compose[@]}" up -d --build --wait --wait-timeout 300

# The notebook runs top to bottom; a failing cell fails the command. The executed copy stays in the
# container (the notebooks folder is the checkout).
"${compose[@]}" exec -T jupyter jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=300 --output-dir /tmp --output lakehouse-run.ipynb lakehouse.ipynb

# The table the notebook wrote is a Delta table in the bucket: its log has commits.
commits=$("${compose[@]}" exec -T minio mc ls local/lakehouse/trips/_delta_log/ | grep -c '\.json$')
[ "$commits" -ge 2 ] || { echo "::error::expected Delta commits in s3://lakehouse/trips, found $commits"; exit 1; }
echo "compose quickstart: notebook ran, $commits Delta commits in s3://lakehouse/trips"
