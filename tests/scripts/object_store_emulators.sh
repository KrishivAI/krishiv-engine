#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Starts throw-away S3 (MinIO), GCS (fake-gcs-server) and Azure (Azurite) emulators in Docker, runs the
# object-store tests against them (P2-T084) and removes the containers, on success, error and interrupt.
#
#   tests/scripts/object_store_emulators.sh                  # krishiv-io round trips + the https/hf tests
#   KRISHIV_TPCH_DIR=/path/to/tpch-sf1 tests/scripts/object_store_emulators.sh   # also TPC-H over S3
#
# Containers are named krishiv-test-<kind>-<pid> and bind 127.0.0.1 on free ports; only those names are ever
# removed (never bench-minio or any other container). Images are overridable and should be pinned by digest in CI:
#   MINIO_IMAGE (default minio/minio), FAKE_GCS_IMAGE (fsouza/fake-gcs-server),
#   AZURITE_IMAGE (mcr.microsoft.com/azure-storage/azurite), AZ_CLI_IMAGE (mcr.microsoft.com/azure-cli).
set -euo pipefail

MINIO_IMAGE="${MINIO_IMAGE:-minio/minio}"
FAKE_GCS_IMAGE="${FAKE_GCS_IMAGE:-fsouza/fake-gcs-server}"
AZURITE_IMAGE="${AZURITE_IMAGE:-mcr.microsoft.com/azure-storage/azurite}"
AZ_CLI_IMAGE="${AZ_CLI_IMAGE:-mcr.microsoft.com/azure-cli}"
# Azurite's published development account (public, fixed by Microsoft; not a secret).
AZURITE_KEY="Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"

# The names are fixed up front: `start` runs in a command substitution, so it cannot register them.
containers=("krishiv-test-minio-$$" "krishiv-test-gcs-$$" "krishiv-test-azurite-$$")
work="$(mktemp -d)"
cleanup() {
  status=$?
  for c in "${containers[@]:-}"; do
    [ -n "$c" ] && docker rm -f "$c" >/dev/null 2>&1 || true
  done
  rm -rf "$work"
  exit "$status"
}
trap cleanup EXIT INT TERM

# start <kind> <container port> <docker run args and image...> [container args...]: prints the host port.
start() {
  local kind="$1" port="$2"
  shift 2
  local name="krishiv-test-${kind}-$$"
  docker run -d --rm --name "$name" -p "127.0.0.1::${port}" "$@" >/dev/null
  docker port "$name" "${port}/tcp" | head -n1 | sed 's/.*://'
}

# wait_for <url>: polls until the server answers (any HTTP status).
wait_for() {
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null --max-time 2 "$1"; then return 0; fi
    sleep 1
  done
  echo "timed out waiting for $1" >&2
  return 1
}

# MinIO: the bucket is a directory of its data dir, created before the server starts.
mkdir -p "$work/minio/krishiv-test"
s3_port="$(start minio 9000 --user "$(id -u):$(id -g)" -v "$work/minio:/data" \
  "$MINIO_IMAGE" server /data)"
wait_for "http://127.0.0.1:${s3_port}/minio/health/live"

gcs_port="$(start gcs 4443 "$FAKE_GCS_IMAGE" -scheme http -port 4443)"
wait_for "http://127.0.0.1:${gcs_port}/storage/v1/b?project=test-project"

az_port="$(start azurite 10000 "$AZURITE_IMAGE" azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck)"
wait_for "http://127.0.0.1:${az_port}/devstoreaccount1?comp=list"
docker run --rm --network host "$AZ_CLI_IMAGE" az storage container create --name krishiv-test \
  --connection-string "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=${AZURITE_KEY};BlobEndpoint=http://127.0.0.1:${az_port}/devstoreaccount1;" >/dev/null

export KRISHIV_TEST_S3_ENDPOINT="http://127.0.0.1:${s3_port}"
export KRISHIV_TEST_GCS_ENDPOINT="http://127.0.0.1:${gcs_port}"
export KRISHIV_TEST_AZURE_ENDPOINT="http://127.0.0.1:${az_port}/devstoreaccount1"

cargo nextest run -p krishiv-io --features cloud --test object_stores --no-tests=pass
if [ -n "${KRISHIV_TPCH_DIR:-}" ]; then
  cargo nextest run -p krishiv-session --features cloud --test s3_tpch --no-tests=pass
fi
