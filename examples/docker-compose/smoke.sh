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
# MinIO by digest: the compose default is a moving tag, and the service needs `sh`, `mkdir` and
# `mc` in the image. Bump it here and in README.md together.
export MINIO_IMAGE=${MINIO_IMAGE:-cgr.dev/chainguard/minio@sha256:304e0bbeb70a9aa0c40ff26d5a9c0a24296019c6d326ba8b75214fb73441d6e9}
# JupyterLab runs as this user, as README.md tells a Linux user whose uid is not 1000.
export JUPYTER_UID=${JUPYTER_UID:-$(id -u)} JUPYTER_GID=${JUPYTER_GID:-$(id -g)}
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

# JupyterLab can save into the notebooks folder (the checkout, owned by the user running this).
"${compose[@]}" exec -T jupyter sh -c 'touch .smoke-write && rm .smoke-write' \
  || { echo "::error::JupyterLab cannot write to the notebooks folder"; exit 1; }

# The table the notebook wrote is a Delta table in the bucket: its log has the four commits
# (the write, INSERT, UPDATE, DELETE).
commits=$("${compose[@]}" exec -T minio mc ls local/lakehouse/trips/_delta_log/ | grep -c '\.json$' || true)
[ "$commits" -ge 4 ] || { echo "::error::expected Delta commits in s3://lakehouse/trips, found $commits"; exit 1; }
echo "compose quickstart: notebook ran, $commits Delta commits in s3://lakehouse/trips"
