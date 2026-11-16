#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Starts a throw-away Apache Iceberg REST fixture (the Java RESTCatalogServlet over an in-memory catalog) in Docker,
# runs the krishiv-catalogs REST client acceptance against it (PR-17) and removes the container, on success, error
# and interrupt.
#
#   tests/scripts/iceberg_rest_fixture.sh
#
# The container is named krishiv-test-iceberg-rest-<pid> and binds 127.0.0.1 on a free port; only that name is ever
# removed. The image is overridable and should be pinned by digest in CI: ICEBERG_REST_IMAGE
# (default apache/iceberg-rest-fixture:1.10.0, the newest tag published; the client follows the 1.12.0 spec, whose
# read endpoints are unchanged since).
set -euo pipefail

ICEBERG_REST_IMAGE="${ICEBERG_REST_IMAGE:-apache/iceberg-rest-fixture:1.10.0}"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"

name="krishiv-test-iceberg-rest-$$"
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && docker inspect "$name" >/dev/null 2>&1; then
    echo "--- iceberg-rest-fixture log (last 25 lines) ---" >&2
    docker logs --tail 25 "$name" >&2 2>&1 || true
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

docker run -d --rm --name "$name" -p 127.0.0.1::8181 -e CATALOG_WAREHOUSE=/tmp/warehouse "$ICEBERG_REST_IMAGE" >/dev/null
port="$(docker port "$name" 8181/tcp | head -n1 | sed 's/.*://')"
url="http://127.0.0.1:${port}"
for _ in $(seq 60); do
  curl -sf "${url}/v1/config" >/dev/null && break
  sleep 1
done
curl -sf "${url}/v1/config" >/dev/null || { echo "the fixture did not start" >&2; exit 1; }

KRISHIV_ICEBERG_REST_URL="$url" cargo nextest run -p krishiv-catalogs --features iceberg-rest \
  -E 'test(fixture_lists_and_loads_what_the_java_server_wrote)' --no-tests=fail
