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
#   MINIO_IMAGE (default cgr.dev/chainguard/minio: Docker Hub no longer serves minio/minio), FAKE_GCS_IMAGE (fsouza/fake-gcs-server, the image object_store's own CI
#   uses; the tustvold image answers the XML PUT with 405),
#   AZURITE_IMAGE (mcr.microsoft.com/azure-storage/azurite), AZ_CLI_IMAGE (mcr.microsoft.com/azure-cli).
set -euo pipefail

MINIO_IMAGE="${MINIO_IMAGE:-cgr.dev/chainguard/minio}"
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
  if [ "$status" -ne 0 ] && docker inspect "krishiv-test-gcs-$$" >/dev/null 2>&1; then
    echo "--- fake-gcs-server log (last 25 lines) ---" >&2
    docker logs --tail 25 "krishiv-test-gcs-$$" >&2 2>&1 || true
  fi
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

# GCS is opt-in (WITH_GCS=1): no fake-gcs-server image tried (tustvold: 405; fsouza latest, 2026-08: 400 "invalid
# uploadType") accepts the XML-API PUT that object_store uploads through, so gcs_round_trip cannot pass against an
# emulator yet; without the endpoint the test only checks offline store construction.
if [ "${WITH_GCS:-0}" = 1 ]; then
  # fake-gcs-server routes the XML API (what object_store uploads through) by Host header, so its public host must be
  # the address the client uses: pick a free port and publish the container on that same port.
  gcs_port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  docker run -d --rm --name "krishiv-test-gcs-$$" -p "127.0.0.1:${gcs_port}:${gcs_port}" "$FAKE_GCS_IMAGE" \
    -scheme http -backend memory -port "$gcs_port" -public-host "127.0.0.1:${gcs_port}" -log-level debug >/dev/null
  wait_for "http://127.0.0.1:${gcs_port}/storage/v1/b?project=test-project"
  # fake-gcs-server starts without buckets (an upload to a missing one is a 404), unlike the MinIO data dir above.
  curl -fsS -X POST -H 'Content-Type: application/json' -d '{"name":"krishiv-test"}' \
    "http://127.0.0.1:${gcs_port}/storage/v1/b?project=test-project" >/dev/null
  export KRISHIV_TEST_GCS_ENDPOINT="http://127.0.0.1:${gcs_port}"
fi

az_port="$(start azurite 10000 "$AZURITE_IMAGE" azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck)"
wait_for "http://127.0.0.1:${az_port}/devstoreaccount1?comp=list"
docker run --rm --network host "$AZ_CLI_IMAGE" az storage container create --name krishiv-test \
  --connection-string "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=${AZURITE_KEY};BlobEndpoint=http://127.0.0.1:${az_port}/devstoreaccount1;" >/dev/null

# A read+list container SAS for the CSV read signed with Hadoop ABFS's SAS keys (`azure_csv_reads_with_a_sas`).
KRISHIV_TEST_AZURE_SAS="$(docker run --rm --network host "$AZ_CLI_IMAGE" az storage container generate-sas \
  --name krishiv-test --permissions rl --expiry 2030-01-01T00:00:00Z -o tsv \
  --connection-string "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=${AZURITE_KEY};BlobEndpoint=http://127.0.0.1:${az_port}/devstoreaccount1;")"
export KRISHIV_TEST_AZURE_SAS

# A second Azurite in OAuth mode for the bearer-token read (`azure_csv_reads_with_a_bearer_token`, lakehouse gate 2).
# `--oauth basic` serves only https and checks a token's audience, issuer and lifetime, not its signature, so the
# tokens below are unsigned JWTs: one for Azure Storage's audience and one for another audience, which it must refuse.
# The certificate is a throw-away CA plus a 127.0.0.1 leaf; the test run trusts it through SSL_CERT_FILE (a bundle of
# the system roots and this CA), which rustls-platform-verifier reads on Linux.
mkdir -p "$work/certs"
(
  cd "$work/certs"
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 1 -subj "/CN=krishiv-test-ca" \
    -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign" \
    -keyout ca-key.pem -out ca.pem 2>/dev/null
  openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -subj "/CN=127.0.0.1" \
    -keyout key.pem -out leaf.csr 2>/dev/null
  printf 'subjectAltName=IP:127.0.0.1,DNS:localhost\nbasicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n' >leaf.ext
  openssl x509 -req -in leaf.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -days 1 -extfile leaf.ext \
    -out cert.pem 2>/dev/null
  chmod 644 ./*.pem
)
system_roots=""
for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
  if [ -f "$f" ]; then system_roots="$f"; break; fi
done
cat ${system_roots:+"$system_roots"} "$work/certs/ca.pem" >"$work/certs/bundle.pem"
containers+=("krishiv-test-azurite-oauth-$$")
az_oauth_port="$(start azurite-oauth 10000 -v "$work/certs:/certs:ro" "$AZURITE_IMAGE" azurite-blob --blobHost 0.0.0.0 \
  --skipApiVersionCheck --oauth basic --cert /certs/cert.pem --key /certs/key.pem)"
az_oauth="https://127.0.0.1:${az_oauth_port}/devstoreaccount1"
for _ in $(seq 1 60); do
  if curl -s -o /dev/null --max-time 2 --cacert "$work/certs/ca.pem" "${az_oauth}?comp=list"; then break; fi
  sleep 1
done
b64url() { base64 | tr -d '\n=' | tr '+/' '-_'; }
# azure_token <audience>: an unsigned JWT valid from 5 minutes ago for an hour.
azure_token() {
  local now tenant="00000000-0000-0000-0000-00000000c0de"
  now="$(date +%s)"
  printf '%s.%s.%s' "$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)" \
    "$(printf '{"aud":"%s","iss":"https://sts.windows.net/%s/","nbf":%d,"iat":%d,"exp":%d,"oid":"%s","tid":"%s"}' \
      "$1" "$tenant" $((now - 300)) $((now - 300)) $((now + 3600)) "$tenant" "$tenant" | b64url)" \
    "$(printf 'unsigned' | b64url)"
}
KRISHIV_TEST_AZURE_BEARER="$(azure_token https://storage.azure.com)"
KRISHIV_TEST_AZURE_BEARER_OTHER_AUDIENCE="$(azure_token https://example.com)"
curl -fsS -o /dev/null --cacert "$work/certs/ca.pem" -X PUT -H "Authorization: Bearer ${KRISHIV_TEST_AZURE_BEARER}" \
  -H 'x-ms-version: 2025-01-05' -H 'Content-Length: 0' "${az_oauth}/krishiv-test?restype=container"
export KRISHIV_TEST_AZURE_BEARER KRISHIV_TEST_AZURE_BEARER_OTHER_AUDIENCE
export KRISHIV_TEST_AZURE_OAUTH_ENDPOINT="$az_oauth"

export KRISHIV_TEST_S3_ENDPOINT="http://127.0.0.1:${s3_port}"
export KRISHIV_TEST_AZURE_ENDPOINT="http://127.0.0.1:${az_port}/devstoreaccount1"

# Every step runs even when an earlier one fails; the exit status is non-zero if any did.
status=0
SSL_CERT_FILE="$work/certs/bundle.pem" cargo nextest run -p krishiv-io --features cloud --test object_stores --locked --no-tests=pass --no-fail-fast || status=1
if [ -n "${KRISHIV_TPCH_DIR:-}" ]; then
  cargo nextest run -p krishiv-session --features cloud --test s3_tpch --locked --no-tests=pass --no-fail-fast || status=1
fi
exit "$status"
