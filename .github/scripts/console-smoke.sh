#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Release check: a built `krishiv` binary serves the web console.
#
#   console-smoke.sh BINARY
#
# Starts `BINARY server` on free loopback ports, reads the `open <url>` line it
# prints, and checks that:
#   - the console page (`/`) is HTML that loads the console's script,
#   - `/api/internal/health` answers 200 with the start-up token,
#   - the same route answers 401 without it,
#   - `/healthz` answers 200.
# A binary built without the `server` and `ui` features fails here: it has no
# `server` command, or prints no `open` line.
set -euo pipefail
bin=$1
[ -x "$bin" ] || { echo "::error::$bin is not an executable"; exit 1; }
work=$(mktemp -d)
pid=
cleanup() {
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

"$bin" server --port 0 --connect-port 0 >"$work/out" 2>"$work/err" &
pid=$!

url=
for _ in $(seq 1 150); do
  url=$(sed -n 's/^open \(http:\/\/[^ ]*\)$/\1/p' "$work/out" | head -n 1)
  [ -n "$url" ] && break
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "::error::krishiv server exited before it printed the console address"
    cat "$work/out" "$work/err"
    exit 1
  fi
  sleep 0.2
done
if [ -z "$url" ]; then
  echo "::error::krishiv server printed no 'open <url>' line within 30 s (built without the ui feature?)"
  cat "$work/out" "$work/err"
  exit 1
fi

base=${url%%/\?token=*}
token=${url##*token=}
[ -n "$token" ] && [ "$token" != "$url" ] || { echo "::error::no token in the printed address"; exit 1; }

fetch() { curl --silent --show-error --max-time 10 "$@"; }

fetch -o "$work/page" -D "$work/page.headers" "$base/"
grep -qi '^content-type: text/html' "$work/page.headers" \
  || { echo "::error::/ is not served as text/html"; cat "$work/page.headers"; exit 1; }
grep -qi '<!doctype html' "$work/page" || { echo "::error::/ is not an HTML document"; exit 1; }
grep -q '<script' "$work/page" || { echo "::error::the console page loads no script"; exit 1; }

code=$(fetch -o "$work/health" -w '%{http_code}' -H "Authorization: Bearer $token" "$base/api/internal/health")
[ "$code" = 200 ] || { echo "::error::/api/internal/health with the token answered $code"; cat "$work/health"; exit 1; }

code=$(fetch -o /dev/null -w '%{http_code}' "$base/api/internal/health")
[ "$code" = 401 ] || { echo "::error::/api/internal/health without a token answered $code, expected 401"; exit 1; }

code=$(fetch -o /dev/null -w '%{http_code}' "$base/healthz")
[ "$code" = 200 ] || { echo "::error::/healthz answered $code"; exit 1; }

echo "console-smoke: $bin serves the console at $base"
